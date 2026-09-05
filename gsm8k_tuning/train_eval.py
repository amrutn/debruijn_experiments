"""
Fine-tune Qwen2.5-1.5B-Instruct on GSM8K reasoning traces and score it on the
GSM8K test set.

Three adapters are trained on the same 1,500 training problems -- on the
standard traces, on the restated traces (`gsm8k_data`) and on a larger
model's solutions (`distill`) -- and evaluated alongside the untuned base
model. Training is LoRA on top of the frozen base model with the math task's
settings: a full fine-tune of a 1.5B model needs ~25 GB for weights +
gradients + AdamW state, whereas LoRA produces ~40 MB adapters that are cheap
to cache and fits any current GPU.

Loss is next-token cross-entropy on the *assistant* tokens only -- the chat
template's system/user turns are context and are never scored -- exactly as in
the math task.

Evaluation
----------
Ordinary batched greedy decoding of the chat prompt, up to `max_new_tokens`
new tokens, scored by `gsm8k_data.is_correct` on the number after ``####``. A
trace that hits the cap without answering is scored wrong; the fraction that
did so is reported as `capped` so a low accuracy can be read correctly.

Caching
-------
Three layers under ``cache/``: ``adapters/`` (a trained LoRA per (condition,
seed), keyed by the full training config), ``decodes/`` (the test-set
completions per model, keyed additionally by the decoding config) and
``evals/`` (the score). An eval-only change reuses the adapter and a rescoring
reuses the decode, so a rerun recomputes only what is missing.

Progress is also saved *within* a unit, so an interruption costs minutes, not
the unit: training writes a resumable checkpoint (adapter, optimizer,
schedule, RNG state and counters) every `CKPT_EVERY_STEPS` optimizer steps
and continues from it on the next run, and decoding appends every finished
batch to a partial file that the next run reads back before decoding the
rest. Both are removed once the final artefact is written.
"""

import os
import json
import time
import random
import shutil
from dataclasses import dataclass, asdict

import torch
from tqdm.auto import tqdm

from gsm8k_data import (
    DataConfig, get_problems, standard_trace, is_correct, extract_answer,
    train_spec, test_spec, _key, CACHE, SYSTEM_PROMPT, ANSWER_MARKER,
)
from traces import TraceConfig, trace_spec, load_traces
from distill import distill_spec, load_distill_traces

ADAPTER_CACHE = os.path.join(CACHE, 'adapters')
EVAL_CACHE = os.path.join(CACHE, 'evals')
DECODE_CACHE = os.path.join(CACHE, 'decodes')

BASE_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
# The Hub commit of the base model the results are reported for.
BASE_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'

# 'base' is the untuned model; the others are adapters trained on that trace
# source. The order is the order of the figure and of the run queue, so the
# distillation units come after everything else.
CONDITIONS = ('base', 'standard', 'restated', 'distill')


# ----------------------------------------------------------------------------
# configs
# ----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Base model + LoRA adapter shape."""
    base_model: str = BASE_MODEL
    base_revision: str = BASE_REVISION      # None: whatever the Hub serves
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'gate_proj', 'up_proj', 'down_proj')


@dataclass
class TrainConfig:
    """Optimisation settings for the LoRA fine-tune (the math task's)."""
    lr: float = 1e-4
    weight_decay: float = 0.0
    # The budget is total samples consumed, as in the math task: the pool is
    # reshuffled each time it is exhausted, so `n_train x passes` samples is
    # `passes` epochs. Set per run by `run_experiments.train_cfg`.
    total_samples: int = 4500       # 1,500 problems x 3 passes
    schedule: str = 'cosine'        # annealed over this run's own horizon
    batch_size: int = 8
    grad_accum: int = 4             # effective batch 32
    warmup_frac: float = 0.03
    grad_clip: float = 1.0
    # A standard example is ~195 tokens with the Qwen2.5 tokenizer and never
    # exceeds 470; the hand-written ledger examples average ~330 and top out
    # at 880; the default teacher's examples match the standard ones. Nothing
    # is truncated at 1,024; an example that would not fit is dropped and the
    # count is logged. (The math task used 896; its traces are shorter.) The
    # thinking-trace teacher overrides this per condition (`distill.budget`).
    max_seq_len: int = 1024
    seed: int = 0
    dtype: str = 'bfloat16'


@dataclass
class EvalConfig:
    """Decoding settings for evaluation."""
    # Standard ground-truth traces are at most ~340 assistant tokens, the
    # hand-written ledger traces at most ~740 (rule-based ones reach ~1,400).
    # 1,024 leaves the tail room without letting a runaway trace hold the
    # batch for long. The thinking-trace teacher overrides it (`distill.budget`).
    max_new_tokens: int = 1024
    # Runtime only: not part of any cache key (greedy decoding is
    # batch-invariant) and halved on OOM by `run_unit`.
    batch_size: int = 64
    # Greedy keeps a rerun bit-identical; the sampling fields are only read
    # when `do_sample` is set. All of it is in the decode cache key.
    do_sample: bool = False
    temperature: float = 0.6
    top_p: float = 0.8
    eval_seed: int = 0


def eval_spec(ecfg):
    """The decoding settings that determine a completion: an EvalConfig minus
    the batch size, which changes only how the prompts are batched."""
    spec = asdict(ecfg)
    spec.pop('batch_size', None)
    return spec


def model_spec(cond, dcfg, mcfg, tcfg, trcfg):
    """
    What identifies a model under evaluation: for the base condition the base
    model alone; for a fine-tuned one the training subset, adapter shape and
    optimisation settings, plus -- for the restated and distillation
    conditions -- the trace set. The test split is deliberately absent, so an
    adapter is shared by every evaluation of it.
    """
    if cond == 'base':
        return dict(task='gsm8k', cond='base', base_model=mcfg.base_model,
                    base_revision=mcfg.base_revision)
    spec = dict(task='gsm8k', cond=cond, data=train_spec(dcfg), model=asdict(mcfg),
                train=asdict(tcfg))
    if cond == 'restated':
        spec['traces'] = trace_spec(trcfg, dcfg)
    elif cond == 'distill':
        spec['traces'] = distill_spec(trcfg, dcfg)
    return spec


# ----------------------------------------------------------------------------
# tokenisation (chat template, assistant-only loss)
# ----------------------------------------------------------------------------

def chat_prefix(tokenizer, problem):
    """The templated system+user turns plus the assistant generation header."""
    msgs = [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': problem.question.strip()}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode_example(tokenizer, problem, trace, max_seq_len):
    """
    One training example: chat-templated prompt + trace, with the prompt tokens
    masked out of the loss.

    Returns
    -------
    dict | None
        ``{'input_ids', 'labels'}``, or None if it does not fit `max_seq_len`.
    """
    prefix_ids = tokenizer(chat_prefix(tokenizer, problem), add_special_tokens=False)['input_ids']
    # the assistant turn must be closed so the model learns to stop
    reply_ids = tokenizer(trace + '<|im_end|>', add_special_tokens=False)['input_ids']
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


def training_text(cond, problem, traces):
    """The assistant-side text a condition trains on for `problem`."""
    if cond == 'standard':
        return standard_trace(problem)
    if cond in ('restated', 'distill'):
        return traces[problem.idx]
    raise ValueError(f'condition {cond!r} is not trained')


def load_training_traces(cond, dcfg, trcfg):
    """The ``{idx: text}`` a condition trains on, validated, or None for the
    standard condition (which reads the ground truth directly)."""
    if cond == 'restated':
        return load_traces(trcfg, dcfg)
    if cond == 'distill':
        return load_distill_traces(trcfg, dcfg)
    return None


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


def train_adapter(cond, dcfg, mcfg, tcfg, trcfg, device='cuda', force=False,
                  log=print, progress_pos=0):
    """
    Fine-tune a LoRA for `cond` ('standard', 'restated' or 'distill'), or reuse
    the cached one.

    Training consumes `tcfg.total_samples` samples from the pool, reshuffling
    each time it is exhausted, under a cosine schedule over that budget. The
    restated and distillation conditions read their traces through
    `load_training_traces`, which validates them before anything is trained.

    Every `CKPT_EVERY_STEPS` optimizer steps the adapter, optimizer, schedule,
    RNG states and counters are checkpointed next to the final path, and a
    run that finds such a checkpoint continues from it -- the same sample
    order and schedule as the uninterrupted run, up to floating-point
    nondeterminism. The finished adapter is written to a temporary directory
    and moved into place with a `done.json` marker as the completion signal,
    so an interrupted run cannot leave a half-written adapter behind; the
    checkpoint is then removed.

    Params
    ------
    cond : str
        'standard', 'restated' or 'distill'.
    dcfg, mcfg, tcfg, trcfg : DataConfig, ModelConfig, TrainConfig, TraceConfig
    force : bool
        Retrain from scratch even if a cached adapter or a checkpoint exists.

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

    traces = load_training_traces(cond, dcfg, trcfg)
    problems = get_problems(dcfg, 'train')

    log(f'{tag} loading {mcfg.base_model} on {device} ...')
    tok = AutoTokenizer.from_pretrained(mcfg.base_model, revision=mcfg.base_revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mcfg.base_model, revision=mcfg.base_revision,
                                                 dtype=dtype)
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

    data, n_absent = [], 0
    build = tqdm(problems, desc=f'{cond} s{tcfg.seed} build data', position=progress_pos,
                 leave=False, dynamic_ncols=True, mininterval=2.0, unit='ex')
    for prob in build:
        if traces is not None and prob.idx not in traces:
            n_absent += 1                                # dropped by the trace set's fill policy
            continue
        e = encode_example(tok, prob, training_text(cond, prob, traces), tcfg.max_seq_len)
        if e is not None:
            data.append(e)
    build.close()
    lens = [len(e['input_ids']) for e in data]
    mean_tok = sum(lens) / max(len(lens), 1)
    reply_tok = sum(sum(l != -100 for l in e['labels']) for e in data) / max(len(data), 1)
    budget = int(tcfg.total_samples)
    eff_batch = tcfg.batch_size * tcfg.grad_accum
    opt_steps = (budget + eff_batch - 1) // eff_batch
    n_too_long = len(problems) - n_absent - len(data)

    log(f'{tag} LoRA {trainable/1e6:.1f}M trainable / {total/1e6:.0f}M total '
        f'({100 * trainable / total:.2f}%)')
    log(f'{tag} pool {len(data)}/{len(problems)} examples ({n_too_long} over '
        f'{tcfg.max_seq_len} tok dropped, {n_absent} without a trace) | mean {mean_tok:.0f} '
        f'tok ({reply_tok:.0f} scored), max {max(lens) if lens else 0}')
    log(f'{tag} streaming {budget:,} samples ({budget * mean_tok / 1e6:.2f}M tokens, '
        f'{opt_steps:,} optimizer steps, effective batch {eff_batch})')

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = get_scheduler(tcfg.schedule, opt,
                          num_warmup_steps=int(tcfg.warmup_frac * opt_steps),
                          num_training_steps=opt_steps)

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
                           n_examples=len(data), n_too_long=n_too_long, n_absent=n_absent),
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
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=(torch.cuda.get_rng_state(device)
                                  if device.startswith('cuda') else None),
                        n_data=len(data)),
                   os.path.join(tmp, CKPT_STATE))
        shutil.rmtree(ckpt, ignore_errors=True)
        os.replace(tmp, ckpt)

    model.train()
    t0 = time.time() - elapsed
    bar = tqdm(total=budget, initial=consumed, desc=f'{cond} s{tcfg.seed} train',
               position=progress_pos, leave=True, dynamic_ncols=True, mininterval=2.0,
               unit='ex')
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
        (out.loss / tcfg.grad_accum).backward()
        running.append(float(out.loss.item()))
        tok_seen += int(attn.sum())
        consumed += take
        micro += 1
        if micro % tcfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], tcfg.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if (micro // tcfg.grad_accum) % CKPT_EVERY_STEPS == 0 and consumed < budget:
                checkpoint()
        bar.update(take)
        if micro % (tcfg.grad_accum * 5) == 0:
            el = time.time() - t0
            bar.set_postfix(loss=f'{sum(running[-40:]) / len(running[-40:]):.4f}',
                            tok_s=f'{tok_seen / max(el, 1e-9):.0f}', refresh=False)
    bar.close()
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
        mcfg.base_model, revision=mcfg.base_revision, dtype=getattr(torch, tcfg.dtype))
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
    spec = dict(task='gsm8k-decode', model=model_spec(cond, dcfg, mcfg, tcfg, trcfg),
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
    spec = dict(task='gsm8k-eval', model=model_spec(cond, dcfg, mcfg, tcfg, trcfg),
                eval=eval_spec(ecfg), test=test_spec(dcfg))
    return os.path.join(EVAL_CACHE, _key(spec) + '.json')


def cached_unit(cond, dcfg, mcfg, tcfg, ecfg, trcfg):
    """The cached result for a unit, or None -- without loading any model."""
    path = _eval_path(cond, dcfg, mcfg, tcfg, ecfg, trcfg)
    if os.path.exists(path):
        with open(path) as f:
            r = json.load(f)
        r['cached'] = True
        return r
    return None


def score(problems, outputs):
    """Accuracy and the trace statistics behind it, over one decode."""
    n = max(len(problems), 1)
    n_ok = sum(is_correct(o['text'], p.gold) for p, o in zip(problems, outputs))
    return dict(accuracy=n_ok / n,
                answered=sum(ANSWER_MARKER in o['text'] for o in outputs) / n,
                capped=sum(o['capped'] for o in outputs) / n,
                mean_tokens=sum(o['n_tokens'] for o in outputs) / n,
                n_eval=len(problems))


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

    Params
    ------
    cond : str
        One of `CONDITIONS`. The seed of a fine-tuned condition is `tcfg.seed`.
    get_model : callable | None
        Zero-argument callable returning ``(model, tokenizer)``, invoked only
        if the decode is not cached. Defaults to training/loading on demand.
    force : bool
        Recompute even if cached.

    Returns
    -------
    dict
        {'cond', 'seed', 'accuracy', 'answered', 'capped', 'mean_tokens',
        'n_eval', 'seconds', 'cached'}
    """
    seed = None if cond == 'base' else tcfg.seed
    path = _eval_path(cond, dcfg, mcfg, tcfg, ecfg, trcfg)
    if os.path.exists(path) and not force:
        with open(path) as f:
            r = json.load(f)
        r['cached'] = True
        return r

    problems = get_problems(dcfg, 'test')
    t0 = time.time()

    def _default_model():
        adapter = None if cond == 'base' else train_adapter(
            cond, dcfg, mcfg, tcfg, trcfg, device=device, force=force, log=log,
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
        json.dump(dict(r, samples=[dict(question=p.question, gold=p.gold,
                                        pred=extract_answer(o['text']), text=o['text'])
                                   for p, o in zip(problems[:20], outputs[:20])]),
                  f, indent=1)
    os.replace(tmp, path)
    log(f'{tag} acc={r["accuracy"]:.3f}  answered={r["answered"]:.3f}  '
        f'capped={r["capped"]:.3f}  mean {r["mean_tokens"]:.0f} tok  '
        f'({r["seconds"]:.0f}s for {len(problems)} problems)')
    return r
