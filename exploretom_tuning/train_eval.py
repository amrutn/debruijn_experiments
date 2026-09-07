"""
Fine-tune Qwen2.5-1.5B-Instruct on ExploreToM training targets and score it
on the ExploreToM test rows.

Adapters are trained on the same 1,500 training questions -- on the answer
alone (direct), on the question-specific chain of decisive steps (chain), on
the story restated with the question's slice of the state after each step
(state_tracking, the local De Bruijn trace), on the story restated with a
local per-step note that carries no running state (narration, the length-
matched non-De-Bruijn control; `exploretom_data`), optionally on a teacher's
reasoning (distill; `traces`) -- and evaluated alongside the untuned base
model. Training is LoRA on top
of the frozen base model with the math task's settings: a full fine-tune of
a 1.5B model needs ~25 GB for weights + gradients + AdamW state, whereas
LoRA produces ~40 MB adapters that are cheap to cache and fits any current
GPU.

Loss is next-token cross-entropy on the *assistant* tokens only -- the chat
template's system/user turns are context and are never scored.

Evaluation
----------
Ordinary batched greedy decoding of the chat prompt, up to `max_new_tokens`
new tokens, scored by `exploretom_data.is_correct` on the normalised text of
the answer field. A completion that hits the cap without answering is scored
wrong; the fraction that did so is reported as `capped`, and the fraction
that wrote an answer field as `answered`. Alongside plain accuracy a
*story-level* accuracy is reported: the fraction of test stories whose
every question is answered correctly.

Learning curve
--------------
While an adapter trains it is scored `TrainConfig.curve_evals` times, at
evenly spaced optimizer steps (the last at the end of training), on the very
test set and decoding settings of the final evaluation, so the last curve
point and the final score coincide. The curve is stored with the adapter.

Caching
-------
Three layers under ``cache/``: ``adapters/`` (a trained LoRA per (condition,
seed), keyed by the full training config and the target spec), ``decodes/``
(the test-set completions per model, keyed additionally by the decoding
config) and ``evals/`` (the score). An eval-only change reuses the adapter
and a rescoring reuses the decode, so a rerun recomputes only what is
missing.

Progress is also saved *within* a unit, so an interruption costs minutes, not
the unit: training writes a resumable checkpoint (adapter, optimizer,
schedule, RNG state, counters and the curve so far) every `CKPT_EVERY_STEPS`
optimizer steps and continues from it on the next run, and decoding appends
every finished batch to a partial file that the next run reads back before
decoding the rest. Both are removed once the final artefact is written.

Numerical safety
----------------
A non-finite loss or gradient norm aborts the unit at once (nothing is
saved), because one such step poisons the adapter through AdamW and the
finished adapter would be junk after a long evaluation. Which attention
kernels training may use is a setting (`TrainConfig.sdpa_backends`,
`ModelConfig.attn_implementation`); `check_train.py` tries each kernel on a
machine and says which train cleanly.
"""

import os
import json
import math
import time
import random
import shutil
from dataclasses import dataclass, asdict, replace

import torch
from tqdm.auto import tqdm

from exploretom_data import (
    get_problems, is_correct, extract_answer, user_message,
    train_spec, test_spec, _key, CACHE, SYSTEM_PROMPT,
)
from traces import target_spec, training_examples

ADAPTER_CACHE = os.path.join(CACHE, 'adapters')
EVAL_CACHE = os.path.join(CACHE, 'evals')
DECODE_CACHE = os.path.join(CACHE, 'decodes')

# The base models and the Hub commits the results are reported for.
BASE_MODELS = {
    'Qwen/Qwen2.5-1.5B-Instruct': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
    'Qwen/Qwen2.5-1.5B': '8faed761d45a263340a0528343f099c05c9a4323',
}
BASE_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
BASE_REVISION = BASE_MODELS[BASE_MODEL]

# 'base' is the untuned model; the others are adapters trained on that target.
# The order is the order of the figure and of the run queue; 'distill' needs
# teacher traces and is not run unless asked for.
# 'state_tracking' is the local, De Bruijn-structured reasoning trace (the
# story restated with the question's running world-and-belief state after each
# step); shown as "state-tracking". 'distill' needs teacher traces and is not
# run unless asked for.
CONDITIONS = ('base', 'direct', 'chain', 'state_tracking', 'narration', 'distill')
DEFAULT_CONDITIONS = ('base', 'direct', 'chain', 'state_tracking', 'narration')


# ----------------------------------------------------------------------------
# configs
# ----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Base model + LoRA adapter shape."""
    base_model: str = BASE_MODEL
    base_revision: str = BASE_REVISION      # None: whatever the Hub serves
    # 'sdpa' (torch's fused attention) or 'eager'; `check_train.py` says which
    # kernels train cleanly on a given machine.
    attn_implementation: str = 'sdpa'
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'gate_proj', 'up_proj', 'down_proj')


@dataclass
class TrainConfig:
    """Optimisation settings for the LoRA fine-tune (the math task's)."""
    lr: float = 1e-4
    weight_decay: float = 0.0
    # The budget is total samples consumed: the pool is reshuffled each time it
    # is exhausted, so `n_rows x passes` samples is `passes` epochs. Set per run
    # by `run_experiments.train_cfg`.
    total_samples: int = 6000       # 1,500 rows x 4 passes
    schedule: str = 'cosine'        # annealed over this run's own horizon
    batch_size: int = 8
    grad_accum: int = 4             # effective batch 32
    warmup_frac: float = 0.03
    grad_clip: float = 1.0
    # A story is at most ~180 words and a ledger target restates it with a
    # state line every two steps: the longest ledger examples are ~1,200
    # words, about 1,700 tokens with the prompt (2,900 at stride 1). Nothing
    # is truncated at 4,096; an example that would not fit is dropped and the
    # count is logged.
    max_seq_len: int = 4096
    seed: int = 0
    dtype: str = 'bfloat16'
    # Learning curve: `curve_evals` evaluations spread evenly over the optimizer
    # steps (the last one at the end of training), each a decode of the run's
    # test set under the run's EvalConfig. 0 disables the curve. This is part of
    # the adapter cache key (the curve is produced during training), so changing
    # it retrains the adapters.
    curve_evals: int = 20
    # The kernels torch's fused attention may use during training, a
    # comma-separated subset of `SDPA_BACKENDS` ('' = torch's own choice).
    # cuDNN's kernel is left out: on Hopper/Blackwell GPUs it has returned
    # non-finite gradients in bf16 for padded batches in several torch
    # releases while its forward pass is fine (every adapter of a GSM8K run
    # went NaN that way, the base model fine). The other kernels are as fast
    # for a 1.5B model. `check_train.py` tests each kernel on a machine.
    sdpa_backends: str = 'flash,efficient,math'


@dataclass
class EvalConfig:
    """Decoding settings for evaluation."""
    # A ledger reply for the longest test story is ~1,550 tokens (training
    # ledgers reach ~2,550); 2,048 leaves room without letting a runaway reply
    # hold the batch for long.
    max_new_tokens: int = 2048
    # Runtime only: not part of any cache key (greedy decoding is
    # batch-invariant) and halved on OOM by `run_unit`.
    batch_size: int = 64
    # Greedy keeps a rerun bit-identical; the sampling fields are only read
    # when `do_sample` is set. All of it is in the decode cache key.
    do_sample: bool = False
    temperature: float = 0.6
    top_p: float = 0.8
    eval_seed: int = 0


SDPA_BACKENDS = ('math', 'efficient', 'flash', 'cudnn')


def sdpa_context(names):
    """
    A context that restricts torch's scaled-dot-product attention to the
    kernels in `names` (a list or comma-separated string of `SDPA_BACKENDS`),
    or does nothing when `names` is empty.
    """
    import contextlib
    if isinstance(names, str):
        names = [n.strip() for n in names.split(',') if n.strip()]
    if not names:
        return contextlib.nullcontext()
    unknown = [n for n in names if n not in SDPA_BACKENDS]
    if unknown:
        raise ValueError(f'unknown sdpa backend(s) {unknown}; choose from {SDPA_BACKENDS}')
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
    except ImportError:                              # torch < 2.3: no per-kernel control
        return contextlib.nullcontext()
    table = {'math': 'MATH', 'efficient': 'EFFICIENT_ATTENTION',
             'flash': 'FLASH_ATTENTION', 'cudnn': 'CUDNN_ATTENTION'}
    backends = [getattr(SDPBackend, table[n]) for n in names if hasattr(SDPBackend, table[n])]
    if not backends:
        raise ValueError(f'none of the sdpa backends {names} exist in torch {torch.__version__}')
    return sdpa_kernel(backends)


def eval_spec(ecfg):
    """The decoding settings that determine a completion: an EvalConfig minus
    the batch size, which changes only how the prompts are batched."""
    spec = asdict(ecfg)
    spec.pop('batch_size', None)
    return spec


def model_spec(cond, dcfg, mcfg, tcfg, trcfg):
    """
    What identifies a model under evaluation: for the base condition the base
    model alone; for a fine-tuned one the training rows, adapter shape,
    optimisation settings and the condition's target spec (`traces.target_spec`).
    The test split is deliberately absent, so an adapter is shared by every
    evaluation of it.
    """
    if cond == 'base':
        return dict(task='exploretom', cond='base', base_model=mcfg.base_model,
                    base_revision=mcfg.base_revision)
    return dict(task='exploretom', cond=cond, data=train_spec(dcfg), model=asdict(mcfg),
                train=asdict(tcfg), targets=target_spec(cond, trcfg, dcfg))


# ----------------------------------------------------------------------------
# tokenisation (chat template, assistant-only loss)
# ----------------------------------------------------------------------------

def chat_prefix(tokenizer, problem):
    """The templated system+user turns plus the assistant generation header."""
    msgs = [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_message(problem)}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode_example(tokenizer, problem, target, max_seq_len):
    """
    One training example: chat-templated prompt + target, with the prompt
    tokens masked out of the loss.

    Returns
    -------
    dict | None
        ``{'input_ids', 'labels'}``, or None if it does not fit `max_seq_len`.
    """
    prefix_ids = tokenizer(chat_prefix(tokenizer, problem), add_special_tokens=False)['input_ids']
    # the assistant turn must be closed so the model learns to stop
    reply_ids = tokenizer(target + '<|im_end|>', add_special_tokens=False)['input_ids']
    ids = prefix_ids + reply_ids
    if len(ids) > max_seq_len:
        return None
    labels = [-100] * len(prefix_ids) + list(reply_ids)
    return {'input_ids': ids, 'labels': labels}


def collate(batch, pad_id):
    """Right-pad a list of encoded examples into tensors."""
    n = max(len(b['input_ids']) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        k = n - len(b['input_ids'])
        input_ids.append(b['input_ids'] + [pad_id] * k)
        labels.append(b['labels'] + [-100] * k)
        attn.append([1] * len(b['input_ids']) + [0] * k)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def training_pool(cond, dcfg, trcfg):
    """The ``[(problem, target)]`` a condition trains on (`traces.training_examples`)."""
    return training_examples(cond, dcfg, trcfg)


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

ADAPTER_DONE = 'done.json'          # written last; its presence means "complete"
CKPT_STATE = 'state.pt'             # optimizer, schedule, RNG and counters of a checkpoint
CKPT_EVERY_STEPS = 20               # optimizer steps between checkpoints (640 samples at batch 32)


def _adapter_path(cond, dcfg, mcfg, tcfg, trcfg):
    """Cache directory for the LoRA of a fine-tuned condition."""
    return os.path.join(ADAPTER_CACHE, _key(model_spec(cond, dcfg, mcfg, tcfg, trcfg)))


def _ckpt_path(final_path):
    """Where a training in progress keeps its resumable checkpoint."""
    return final_path + '.ckpt'


def adapter_is_cached(cond, dcfg, mcfg, tcfg, trcfg):
    """True if a *complete* adapter for this config is on disk."""
    return os.path.exists(os.path.join(_adapter_path(cond, dcfg, mcfg, tcfg, trcfg),
                                       ADAPTER_DONE))


def adapter_curve(cond, dcfg, mcfg, tcfg, trcfg, ecfg):
    """
    The learning curve recorded while the adapter was trained -- a list of
    ``{'step', 'samples', 'accuracy', 'robust', 'answered', 'capped',
    'mean_tokens', 'n_eval'}`` in training order -- provided it was recorded
    on the test set of `dcfg` under `ecfg`; else None.
    """
    path = os.path.join(_adapter_path(cond, dcfg, mcfg, tcfg, trcfg), ADAPTER_DONE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        meta = json.load(f)
    if meta.get('curve_test') != test_spec(dcfg) or meta.get('curve_eval') != eval_spec(ecfg):
        return None
    return meta.get('curve', [])


def train_adapter(cond, dcfg, mcfg, tcfg, trcfg, ecfg, device='cuda', force=False,
                  log=print, progress_pos=0):
    """
    Fine-tune a LoRA for `cond` ('direct', 'chain', 'state_tracking' or
    'distill'), or reuse the cached one.

    Training consumes `tcfg.total_samples` samples from the pool, reshuffling
    each time it is exhausted, under a cosine schedule over that budget. The
    learning curve is scored on the test set of `dcfg` under `ecfg`.

    Every `CKPT_EVERY_STEPS` optimizer steps the adapter, optimizer, schedule,
    RNG states, counters and curve are checkpointed next to the final path,
    and a run that finds such a checkpoint continues from it -- the same
    sample order and schedule as the uninterrupted run, up to floating-point
    nondeterminism. The finished adapter is written to a temporary directory
    and moved into place with a `done.json` marker as the completion signal,
    so an interrupted run cannot leave a half-written adapter behind; the
    checkpoint is then removed.

    Returns
    -------
    str
        Path to the adapter.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
    from peft import LoraConfig, get_peft_model, PeftModel

    tag = f'[{cond} s{tcfg.seed}]'
    final_path = _adapter_path(cond, dcfg, mcfg, tcfg, trcfg)
    ckpt = _ckpt_path(final_path)
    if not force and adapter_is_cached(cond, dcfg, mcfg, tcfg, trcfg):
        with open(os.path.join(final_path, ADAPTER_DONE)) as f:
            meta = json.load(f)
        log(f'{tag} adapter CACHED ({meta.get("total_samples", 0):,} samples, '
            f'loss {meta.get("final_loss", float("nan")):.4f})')
        return final_path
    if force:
        shutil.rmtree(ckpt, ignore_errors=True)
    state = None
    if os.path.exists(os.path.join(ckpt, CKPT_STATE)):
        try:
            state = torch.load(os.path.join(ckpt, CKPT_STATE), map_location='cpu',
                               weights_only=False)
        except Exception as e:                          # cut short while being written
            log(f'{tag} checkpoint unreadable ({e}); training from scratch')
            shutil.rmtree(ckpt, ignore_errors=True)

    torch.manual_seed(tcfg.seed)
    random.seed(tcfg.seed)
    dtype = getattr(torch, tcfg.dtype)

    examples = training_pool(cond, dcfg, trcfg)     # [(problem, given, target)]
    n_rows_unique = len({p.idx for p, _ in examples})

    log(f'{tag} loading {mcfg.base_model} on {device} ...')
    tok = AutoTokenizer.from_pretrained(mcfg.base_model, revision=mcfg.base_revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mcfg.base_model, revision=mcfg.base_revision,
                                                 dtype=dtype,
                                                 attn_implementation=mcfg.attn_implementation)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    if state is None:
        model = get_peft_model(model, LoraConfig(
            r=mcfg.lora_r, lora_alpha=mcfg.lora_alpha, lora_dropout=mcfg.lora_dropout,
            target_modules=list(mcfg.target_modules), bias='none', task_type='CAUSAL_LM'))
    else:
        model = PeftModel.from_pretrained(model, ckpt, is_trainable=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    data = []
    build = tqdm(examples, desc=f'{cond} s{tcfg.seed} build data', position=progress_pos,
                 leave=False, dynamic_ncols=True, mininterval=2.0, unit='ex')
    over = []
    for prob, target in build:
        e = encode_example(tok, prob, target, tcfg.max_seq_len)
        if e is not None:
            data.append(e)
        else:
            over.append(len(tok(chat_prefix(tok, prob) + target,
                                add_special_tokens=False)['input_ids']))
    build.close()
    if not data:
        raise RuntimeError(
            f'{tag} no {cond} training example fits max_seq_len={tcfg.max_seq_len} '
            f'({len(over)} examples, {min(over)}-{max(over)} tokens); this target is too long for '
            f'these stories. Raise --max-seq-len or drop {cond} from --conditions.')
    lens = [len(e['input_ids']) for e in data]
    mean_tok = sum(lens) / max(len(lens), 1)
    reply_tok = sum(sum(l != -100 for l in e['labels']) for e in data) / max(len(data), 1)
    budget = int(tcfg.total_samples)
    eff_batch = tcfg.batch_size * tcfg.grad_accum
    opt_steps = (budget + eff_batch - 1) // eff_batch
    n_too_long = len(examples) - len(data)

    log(f'{tag} LoRA {trainable/1e6:.1f}M trainable / {total/1e6:.0f}M total '
        f'({100 * trainable / total:.2f}%)')
    log(f'{tag} pool {len(data)} rows from {n_rows_unique} ({n_too_long} over '
        f'{tcfg.max_seq_len} tok dropped) | mean {mean_tok:.0f} tok ({reply_tok:.0f} scored), '
        f'max {max(lens) if lens else 0}')
    log(f'{tag} streaming {budget:,} samples ({budget * mean_tok / 1e6:.2f}M tokens, '
        f'{opt_steps:,} optimizer steps, effective batch {eff_batch})')

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = get_scheduler(tcfg.schedule, opt,
                          num_warmup_steps=int(tcfg.warmup_frac * opt_steps),
                          num_training_steps=opt_steps)

    # learning curve: which optimizer steps to evaluate at
    curve = []
    curve_steps = []
    if tcfg.curve_evals > 0:
        every = max(1, -(-opt_steps // tcfg.curve_evals))
        curve_steps = sorted(set(range(every, opt_steps, every)) | {opt_steps})
        curve_problems = get_problems(dcfg, 'test')
        tok_gen = AutoTokenizer.from_pretrained(mcfg.base_model, revision=mcfg.base_revision)
        if tok_gen.pad_token_id is None:
            tok_gen.pad_token = tok_gen.eos_token
        tok_gen.padding_side = 'left'
        log(f'{tag} learning curve: {len(curve_steps)} evaluations on the {len(curve_problems)} '
            f'test rows (<= {ecfg.max_new_tokens} new tokens) at steps {curve_steps}')

    def curve_eval(step, n_samples):
        """Score the adapter as it is now on the test set; RNG state is
        restored so the evaluation leaves the training run unchanged."""
        rng_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.startswith('cuda') else None
        model.eval()
        model.config.use_cache = True
        batch = ecfg.batch_size
        while True:
            try:
                outputs = generate_answers(
                    model, tok_gen, curve_problems, replace(ecfg, batch_size=batch),
                    device=device, desc=f'{cond} s{tcfg.seed} curve @{step}',
                    progress_pos=progress_pos)
                break
            except Exception as e:                  # noqa: BLE001 -- OOM retry
                if _is_oom(e) and batch > 1:
                    torch.cuda.empty_cache()
                    batch = max(1, batch // 2)
                    continue
                raise
        model.config.use_cache = False
        model.train()
        torch.set_rng_state(rng_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
        s = score(curve_problems, outputs)
        r = dict(step=step, samples=n_samples, **{k: s[k] for k in (
            'accuracy', 'robust', 'answered', 'capped', 'mean_tokens', 'n_eval')})
        curve.append(r)
        log(f'{tag} curve @ step {step} ({n_samples:,} samples): acc={r["accuracy"]:.3f} '
            f'story={r["robust"]:.3f} answered={r["answered"]:.3f} capped={r["capped"]:.3f} '
            f'mean {r["mean_tokens"]:.0f} tok')

    rng = random.Random(tcfg.seed)
    order = list(range(len(data)))
    rng.shuffle(order)
    cursor = 0
    consumed = tok_seen = micro = 0
    running, elapsed = [], 0.0
    if state is not None:
        if state['n_data'] != len(data):
            raise RuntimeError(f'{tag} checkpoint was trained on a pool of {state["n_data"]} '
                               f'examples but this run built {len(data)}; delete {ckpt}')
        opt.load_state_dict(state['opt'])
        sched.load_state_dict(state['sched'])
        rng.setstate(state['rng'])
        order, cursor = state['order'], state['cursor']
        consumed, micro, tok_seen = state['consumed'], state['micro'], state['tok_seen']
        running, elapsed = state['running'], state['elapsed']
        curve = state.get('curve', [])
        torch.set_rng_state(state['torch_rng'])
        if device.startswith('cuda') and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(state['cuda_rng'], device)
        log(f'{tag} RESUMING from the checkpoint at {consumed:,}/{budget:,} samples')

    def save(n_samples, loss, secs, toks):
        """Write the trained adapter to the exact path the cache check uses."""
        tmp = final_path + f'.tmp{os.getpid()}'
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(tmp)
        with open(os.path.join(tmp, ADAPTER_DONE), 'w') as f:
            json.dump(dict(cond=cond, seed=tcfg.seed, total_samples=n_samples,
                           final_loss=loss, seconds=secs, train_tokens=toks,
                           mean_tokens=mean_tok, reply_tokens=reply_tok,
                           n_examples=len(data), n_too_long=n_too_long, n_rows=n_rows_unique,
                           curve=curve, curve_test=test_spec(dcfg), curve_eval=eval_spec(ecfg)),
                      f, indent=1)
        os.makedirs(ADAPTER_CACHE, exist_ok=True)
        shutil.rmtree(final_path, ignore_errors=True)
        os.replace(tmp, final_path)
        log(f'{tag} saved @ {n_samples:,} samples ({toks/1e6:.2f}M tok, loss {loss:.4f}, '
            f'{secs/60:.1f} min)')

    def checkpoint():
        """Everything needed to continue from here, written atomically."""
        tmp = ckpt + f'.tmp{os.getpid()}'
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(tmp)
        torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), rng=rng.getstate(),
                        order=order, cursor=cursor, consumed=consumed, micro=micro,
                        tok_seen=tok_seen, running=running[-40:], elapsed=time.time() - t0,
                        curve=curve, torch_rng=torch.get_rng_state(),
                        cuda_rng=(torch.cuda.get_rng_state(device)
                                  if device.startswith('cuda') else None),
                        n_data=len(data)),
                   os.path.join(tmp, CKPT_STATE))
        shutil.rmtree(ckpt, ignore_errors=True)
        os.replace(tmp, ckpt)

    model.train()
    t0 = time.time() - elapsed
    if tcfg.sdpa_backends:
        log(f'{tag} attention restricted to the {tcfg.sdpa_backends} kernel(s) for training')
    bar = tqdm(total=budget, initial=consumed, desc=f'{cond} s{tcfg.seed} train',
               position=progress_pos, leave=True, dynamic_ncols=True, mininterval=2.0,
               unit='ex')
    kernels = sdpa_context(tcfg.sdpa_backends)
    kernels.__enter__()
    while consumed < budget:
        take = min(tcfg.batch_size, budget - consumed)
        if cursor + take > len(order):                  # exhausted: reshuffle
            rng.shuffle(order)
            cursor = 0
        idx = order[cursor:cursor + take]
        cursor += take
        ids, labels, attn = collate([data[i] for i in idx], tok.pad_token_id)
        out = model(input_ids=ids.to(device), attention_mask=attn.to(device),
                    labels=labels.to(device))
        loss_val = float(out.loss.item())
        if not math.isfinite(loss_val):
            # Fail here, not after thousands of samples: a non-finite loss
            # poisons the weights through AdamW, and the adapter would be junk.
            raise RuntimeError(
                f'{tag} non-finite loss ({loss_val}) at micro-batch {micro + 1}, sample '
                f'{consumed + take:,}/{budget:,}, sequence lengths {sorted(ids.shape[1:])}; '
                f'nothing saved. Run `python check_train.py` on this GPU to isolate the cause.')
        (out.loss / tcfg.grad_accum).backward()
        running.append(loss_val)
        tok_seen += int(attn.sum())
        consumed += take
        micro += 1
        if micro % tcfg.grad_accum == 0:
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], tcfg.grad_clip)
            if not torch.isfinite(gnorm):
                raise RuntimeError(
                    f'{tag} non-finite gradient norm at optimizer step {micro // tcfg.grad_accum} '
                    f'(loss {loss_val:.4f} was finite); nothing saved. Run `python check_train.py` '
                    'on this GPU to isolate the cause.')
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step = micro // tcfg.grad_accum
            if step in curve_steps and not any(c['step'] == step for c in curve):
                curve_eval(step, consumed)
            if step % CKPT_EVERY_STEPS == 0 and consumed < budget:
                checkpoint()
        bar.update(take)
        if micro % (tcfg.grad_accum * 5) == 0:
            el = time.time() - t0
            bar.set_postfix(loss=f'{sum(running[-40:]) / len(running[-40:]):.4f}',
                            tok_s=f'{tok_seen / max(el, 1e-9):.0f}', refresh=False)
    bar.close()
    kernels.__exit__(None, None, None)
    if curve_steps and not any(c['step'] == opt_steps for c in curve):
        curve_eval(opt_steps, consumed)         # the budget ended between two steps
    save(budget, sum(running[-40:]) / max(len(running[-40:]), 1),
         time.time() - t0, tok_seen)
    shutil.rmtree(ckpt, ignore_errors=True)

    log(f'{tag} TRAINED {consumed:,} samples in {(time.time()-t0)/60:.1f} min '
        f'({tok_seen/max(time.time()-t0,1e-9):.0f} tok/s)')
    del model, opt
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return final_path


# ----------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------

def load_for_eval(adapter_path, mcfg, tcfg, device='cuda'):
    """
    The base model, with `adapter_path` attached when it is not None, ready to
    generate. Returns ``(model, tokenizer)``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mcfg.base_model, revision=mcfg.base_revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'                       # required for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        mcfg.base_model, revision=mcfg.base_revision, dtype=getattr(torch, tcfg.dtype),
        attn_implementation=mcfg.attn_implementation)
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device).eval()
    model.config.use_cache = True
    return model, tok


def _eos_ids(model, tok):
    """Every token id that ends a generation, as a sorted list."""
    ids = {tok.convert_tokens_to_ids('<|im_end|>'), tok.eos_token_id}
    cfg = getattr(model, 'generation_config', None)
    eos = getattr(cfg, 'eos_token_id', None)
    ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
    return sorted(i for i in ids if isinstance(i, int) and i >= 0)


@torch.no_grad()
def generate_answers(model, tok, problems, ecfg, device='cuda', desc='decode',
                     progress_pos=0, done=None, on_batch=None):
    """
    Decode a completion for each problem with ordinary batched generation.

    Prompts are batched longest-first, so that a batch that does not fit fails
    on the first step rather than half-way through the test set.

    Params
    ------
    done : dict | None
        ``{index: output}`` of problems decoded earlier (a partial decode being
        resumed); they are returned as they are and not decoded again.
    on_batch : callable | None
        Called after every batch with the ``(index, output)`` pairs it
        produced -- the hook that appends them to the partial decode file.

    Returns
    -------
    list[dict]
        Per problem, in order: ``{'text', 'n_tokens', 'capped'}`` -- the decoded
        assistant turn, the number of tokens generated up to and including the
        end-of-turn token, and whether the cap was hit before one was produced.
    """
    torch.manual_seed(ecfg.eval_seed)              # only read when sampling
    prefixes = [chat_prefix(tok, p) for p in problems]
    out = [None] * len(problems)
    for i, o in (done or {}).items():
        out[i] = o
    todo = sorted((i for i in range(len(problems)) if out[i] is None),
                  key=lambda i: -len(prefixes[i]))
    eos = _eos_ids(model, tok)
    n_batches = (len(todo) + ecfg.batch_size - 1) // ecfg.batch_size
    bar = tqdm(total=n_batches, desc=desc, position=progress_pos, leave=False,
               dynamic_ncols=True, mininterval=2.0)
    for s0 in range(0, len(todo), ecfg.batch_size):
        idx = todo[s0:s0 + ecfg.batch_size]
        enc = tok([prefixes[i] for i in idx], return_tensors='pt', padding=True,
                  add_special_tokens=False).to(device)
        gen = model.generate(
            **enc, max_new_tokens=ecfg.max_new_tokens,
            do_sample=ecfg.do_sample,
            temperature=ecfg.temperature if ecfg.do_sample else None,
            top_p=ecfg.top_p if ecfg.do_sample else None,
            top_k=None,
            pad_token_id=tok.pad_token_id, eos_token_id=eos)
        new = gen[:, enc['input_ids'].shape[1]:].tolist()
        fresh = []
        for j, i in enumerate(idx):
            ids = new[j]
            n = next((k + 1 for k, t in enumerate(ids) if t in eos), len(ids))
            out[i] = dict(text=tok.decode(ids[:n], skip_special_tokens=True),
                          n_tokens=n, capped=not any(t in eos for t in ids))
            fresh.append((i, out[i]))
        if on_batch is not None:
            on_batch(fresh)
        bar.update(1)
        bar.set_postfix(capped=sum(out[i]['capped'] for i in idx), refresh=False)
    bar.close()
    return out


# ----------------------------------------------------------------------------
# decode cache
# ----------------------------------------------------------------------------
#
# Decoding the test set is the expensive part of evaluation, and a decode is
# worth keeping: a rescoring, a different answer extractor or a look at the
# failures should read a few MB from disk rather than occupy a GPU again.

def _decode_path(cond, dcfg, mcfg, tcfg, ecfg, trcfg):
    spec = dict(task='exploretom-decode', model=model_spec(cond, dcfg, mcfg, tcfg, trcfg),
                eval=eval_spec(ecfg), test=test_spec(dcfg))
    return os.path.join(DECODE_CACHE, _key(spec) + '.json')


def _read_partial(path, n):
    """
    The completions a partial decode file holds for a test set of `n`
    problems, as ``{index: output}``. A line cut short by a crash is skipped;
    a line written for a test set of another size is ignored.
    """
    done = {}
    with open(path) as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get('n') == n and isinstance(r.get('i'), int) and 0 <= r['i'] < n:
                done[r['i']] = dict(text=r['text'], n_tokens=r['n_tokens'], capped=r['capped'])
    return done


def get_or_decode(get_model, problems, cond, dcfg, mcfg, tcfg, ecfg, trcfg,
                  device='cuda', desc=None, progress_pos=0, force=False, log=print,
                  batch_override=None):
    """
    The test-set completions for one model, from cache when possible.

    While decoding, every finished batch is appended to
    ``<decode>.partial.jsonl``, and a run that finds that file decodes only
    the problems it lacks -- so an interruption, or an OOM retry at a smaller
    batch, costs at most one batch. The file is removed once the complete
    decode is written.

    Params
    ------
    get_model : callable
        Zero-argument callable returning ``(model, tokenizer)``. It is only
        invoked on a cache miss, so a fully cached condition never loads a
        model or touches the GPU.
    batch_override : int | None
        Decode with this batch size instead of ``ecfg.batch_size``. It changes
        only the runtime batching, never the result (greedy decoding is
        batch-invariant) and never the cache key, which is what lets a caller
        shrink the batch to survive an OOM without orphaning the decode.

    Returns
    -------
    list[dict]
        As `generate_answers` returns them.
    """
    path = _decode_path(cond, dcfg, mcfg, tcfg, ecfg, trcfg)
    partial = path + '.partial.jsonl'
    if os.path.exists(path) and not force:
        try:
            with open(path) as f:
                blob = json.load(f)
            if len(blob['outputs']) == len(problems):
                return blob['outputs']
            log(f'  [decode] cached decode has {len(blob["outputs"])} outputs, '
                f'expected {len(problems)}; re-decoding')
        except (json.JSONDecodeError, KeyError, OSError):
            log('  [decode] cached decode unreadable; re-decoding')

    done = {}
    if force:
        if os.path.exists(partial):
            os.remove(partial)
    elif os.path.exists(partial):
        done = _read_partial(partial, len(problems))
        if done:
            log(f'  [decode] resuming: {len(done)}/{len(problems)} completions already on disk')

    model, tok = get_model()
    run_ecfg = ecfg if batch_override is None else \
        EvalConfig(**dict(asdict(ecfg), batch_size=batch_override))
    os.makedirs(DECODE_CACHE, exist_ok=True)
    with open(partial, 'a') as pf:
        def on_batch(fresh):
            for i, o in fresh:
                pf.write(json.dumps(dict(i=i, n=len(problems), **o)) + '\n')
            pf.flush()
            os.fsync(pf.fileno())

        outputs = generate_answers(model, tok, problems, run_ecfg, device=device,
                                   desc=desc or f'{cond} decode', progress_pos=progress_pos,
                                   done=done, on_batch=on_batch)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(dict(cond=cond, seed=tcfg.seed, outputs=outputs), f)
    os.replace(tmp, path)
    if os.path.exists(partial):
        os.remove(partial)
    return outputs


# ----------------------------------------------------------------------------
# one (condition, seed) unit, cached
# ----------------------------------------------------------------------------

def _eval_path(cond, dcfg, mcfg, tcfg, ecfg, trcfg):
    spec = dict(task='exploretom-eval', model=model_spec(cond, dcfg, mcfg, tcfg, trcfg),
                eval=eval_spec(ecfg), test=test_spec(dcfg))
    return os.path.join(EVAL_CACHE, _key(spec) + '.json')


def cached_unit(cond, dcfg, mcfg, tcfg, ecfg, trcfg):
    """The cached result for a unit (with the adapter's learning curve for a
    fine-tuned condition), or None -- without loading any model."""
    path = _eval_path(cond, dcfg, mcfg, tcfg, ecfg, trcfg)
    if os.path.exists(path):
        with open(path) as f:
            r = json.load(f)
        r['cached'] = True
        if cond != 'base':
            r['curve'] = adapter_curve(cond, dcfg, mcfg, tcfg, trcfg, ecfg)
        return r
    return None


def score(problems, outputs):
    """
    Accuracy and the statistics behind it, over one decode: `accuracy`
    (rows), `robust` (test stories whose every row is correct), `answered`
    (rows that wrote an answer field), `capped`, `mean_tokens`, and the
    accuracy per question type and per true/false belief as
    ``{name: [n_correct, n]}``.
    """
    n = max(len(problems), 1)
    ok = [is_correct(o['text'], p.gold) for p, o in zip(problems, outputs)]
    stories, by_type, by_belief = {}, {}, {}
    for p, c in zip(problems, ok):
        stories.setdefault(p.story_id, []).append(c)
        by_type.setdefault(p.qtype, [0, 0])
        by_type[p.qtype][0] += int(c)
        by_type[p.qtype][1] += 1
        k = 'false belief' if p.false_belief else 'true belief / factual'
        by_belief.setdefault(k, [0, 0])
        by_belief[k][0] += int(c)
        by_belief[k][1] += 1
    return dict(accuracy=sum(ok) / n,
                robust=sum(all(v) for v in stories.values()) / max(len(stories), 1),
                n_groups=len(stories),
                answered=sum(extract_answer(o['text'])[1] for o in outputs) / n,
                capped=sum(o['capped'] for o in outputs) / n,
                mean_tokens=sum(o['n_tokens'] for o in outputs) / n,
                n_eval=len(problems), by_type=by_type, by_belief=by_belief)


def _is_oom(exc):
    """True for a CUDA out-of-memory error, however torch surfaces it."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower()


def run_unit(cond, dcfg, mcfg, tcfg, ecfg, trcfg, device='cuda', force=False,
             log=print, progress_pos=0, get_model=None):
    """
    Evaluate one condition on the test set, training its adapter first if it
    is not cached.

    A decode at the configured batch that runs out of memory is retried at half
    the batch until it fits (`batch_override` keeps the result keyed by the
    canonical `ecfg`), continuing from the batches already on disk; only a
    genuine batch-1 OOM raises.

    Returns
    -------
    dict
        {'cond', 'seed', 'accuracy', 'robust', 'answered', 'capped',
        'mean_tokens', 'n_eval', 'by_type', 'by_belief', 'seconds', 'cached'}
        plus 'curve' for a fine-tuned condition.
    """
    seed = None if cond == 'base' else tcfg.seed
    path = _eval_path(cond, dcfg, mcfg, tcfg, ecfg, trcfg)
    if os.path.exists(path) and not force:
        return cached_unit(cond, dcfg, mcfg, tcfg, ecfg, trcfg)

    problems = get_problems(dcfg, 'test')
    t0 = time.time()

    def _default_model():
        adapter = None if cond == 'base' else train_adapter(
            cond, dcfg, mcfg, tcfg, trcfg, ecfg, device=device, force=force, log=log,
            progress_pos=progress_pos)
        return load_for_eval(adapter, mcfg, tcfg, device=device)

    tag = f'[{cond}' + ('' if seed is None else f' s{seed}') + ']'
    batch = ecfg.batch_size
    while True:
        try:
            outputs = get_or_decode(
                get_model or _default_model, problems, cond, dcfg, mcfg, tcfg, ecfg,
                trcfg, device=device, desc=f'{tag[1:-1]} decode',
                progress_pos=progress_pos, force=force, log=log,
                batch_override=None if batch == ecfg.batch_size else batch)
            break
        except Exception as e:
            if _is_oom(e) and batch > 1:
                torch.cuda.empty_cache()
                batch = max(1, batch // 2)
                log(f'  {tag} OOM decoding; retrying at batch {batch}')
                continue
            raise
    r = dict(cond=cond, seed=seed, **score(problems, outputs),
             seconds=time.time() - t0, cached=False)

    os.makedirs(EVAL_CACHE, exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(dict(r, samples=[dict(idx=p.idx, story_id=p.story_id, qtype=p.qtype,
                                        false_belief=p.false_belief, question=p.question,
                                        gold=p.gold, pred=extract_answer(o['text'])[0],
                                        text=o['text'])
                                   for p, o in zip(problems, outputs)]),
                  f, indent=1)
    os.replace(tmp, path)
    log(f'{tag} acc={r["accuracy"]:.3f}  story={r["robust"]:.3f}  answered={r["answered"]:.3f}  '
        f'capped={r["capped"]:.3f}  mean {r["mean_tokens"]:.0f} tok  '
        f'({r["seconds"]:.0f}s for {len(problems)} rows)')
    if cond != 'base':
        r['curve'] = adapter_curve(cond, dcfg, mcfg, tcfg, trcfg, ecfg)
    return r
