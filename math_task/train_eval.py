"""
Fine-tune Qwen2.5-1.5B-Instruct on equation-solving traces and evaluate it with
and without per-step operation injection.

One model is trained per state-emission interval k (`generate_data.encode_trace`),
including the standard condition (no state ever emitted). Training is LoRA on top
of the frozen base model: a full fine-tune of a 1.5B model needs ~25 GB for
weights + gradients + AdamW state alone, which does not fit a 24 GB 4090, whereas
LoRA fits comfortably and produces ~40 MB adapters that are cheap to cache.

Loss is next-token cross-entropy on the *assistant* tokens only -- the chat
template's system/user turns are context and are never scored -- mirroring the
answer-masked loss of the navigation experiment.

Evaluation
----------
`generate_clean` decodes an answer normally. `generate_injected` decodes one
reasoning step at a time and, with probability `inject_p`, replaces a step the
model just produced with a random solution-preserving operation
(`generate_data.random_injection`). Injection only ever replaces an *operation*
step: a step that opens with ``Current equation is`` (the model reporting its
state) or the final ``####`` answer is always left alone, matching the navigation
setup where only moves -- never the emitted cell -- are corrupted. Because every
injected operation preserves the solution set, the target answer is unchanged and
a model that really tracks the algebraic state can recover.

Caching
-------
Three layers under ``cache/``: ``datasets/`` (generated problems),
``adapters/`` (a trained LoRA per interval, keyed by the training config), and
``evals/`` (one JSON per (interval, inject_p) unit). An eval-only change reuses
the adapter, so no retraining.
"""

import os
import json
import time
import random
import shutil
import hashlib
from dataclasses import dataclass, asdict, replace

import torch
from tqdm.auto import tqdm

from generate_data import (
    make_dataset, encode_trace, prompt_for, Problem, SYSTEM_PROMPT,
    ANSWER_MARKER, STATE_PREFIX, is_correct, random_injection, GEN_VERSION,
    augmented_trace,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
DATA_CACHE = os.path.join(CACHE, 'datasets')
ADAPTER_CACHE = os.path.join(CACHE, 'adapters')
EVAL_CACHE = os.path.join(CACHE, 'evals')
DECODE_CACHE = os.path.join(CACHE, 'decodes')

BASE_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'


# ----------------------------------------------------------------------------
# configs
# ----------------------------------------------------------------------------

@dataclass
class DataConfig:
    """How the train / test problem sets are generated."""
    n_train: int = 40000
    n_test: int = 500
    train_seed: int = 0
    test_seed: int = 12345          # disjoint seed -> disjoint problems
    min_ops: int = 3
    max_ops: int = 12
    # Injection-augmented training: this fraction of the training traces are
    # rebuilt with random detours spliced in at `augment_p`, which the trace then
    # solves through (`generate_data.augmented_trace`). 0.0 is the plain
    # condition. The problems themselves are unchanged, so the dataset cache is
    # shared with the unaugmented runs; only the traces differ.
    augment_frac: float = 0.0
    augment_p: float = 0.3
    # Bumped when `augmented_trace` changes what it produces. It only enters the
    # cache key for runs that actually augment, so a fix here re-runs the
    # augmented conditions without disturbing anything else. v2 stopped augmented
    # traces using a different solving style (and coming out shorter) than clean.
    augment_version: str = '2'
    # part of every cache key, so changing the generator invalidates old data,
    # adapters and evals instead of silently reusing them
    gen_version: str = GEN_VERSION


@dataclass
class ModelConfig:
    """Base model + LoRA adapter shape."""
    base_model: str = BASE_MODEL
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'gate_proj', 'up_proj', 'down_proj')


@dataclass
class TrainConfig:
    """Optimisation settings for the LoRA fine-tune."""
    lr: float = 1e-4
    weight_decay: float = 0.0
    # The training budget is expressed as total samples consumed rather than
    # epochs, because the fixed-compute experiment gives each condition a
    # different sample count (equal tokens, not equal samples). The pool is
    # reshuffled each time it is exhausted.
    total_samples: int = 120000     # e.g. 40,000 traces x 3 passes
    schedule: str = 'cosine'        # annealed over this run's own horizon
    # Recovery fine-tuning: when > 0 the adapter is *not* trained from scratch.
    # The adapter for the same config with recovery_samples=0 is loaded -- the
    # already-trained model -- and carried on for this many further samples, of
    # which `DataConfig.augment_frac` carry injected detours. A short, cheap
    # continuation rather than a second full run.
    recovery_samples: int = 0
    recovery_lr: float = 5e-5       # lower than `lr`: this is a continuation
    batch_size: int = 8
    grad_accum: int = 4             # effective batch 32
    warmup_frac: float = 0.03
    grad_clip: float = 1.0
    # k=1 traces of a 12-operation problem reach ~760 tokens (measured with the
    # Qwen2.5 tokenizer); 896 keeps every condition inside the window.
    max_seq_len: int = 896
    seed: int = 0
    dtype: str = 'bfloat16'


@dataclass
class EvalConfig:
    """Decoding settings for evaluation."""
    # A k=1 trace of a 12-operation problem is 12 ops + 12 state reports + 1
    # answer = 25 steps when clean, but injections make k=1 derail and run long:
    # at 40 steps, 16.6% of k=1 traces at p=0.5 were truncated before emitting an
    # answer, which scores them wrong and inflates their injection count. 96 puts
    # the cap well clear of that tail. Raising it is nearly free for the other
    # conditions -- finished rows drop out of the decode loop, so only traces that
    # actually run long cost extra.
    max_steps: int = 96             # reasoning steps before giving up
    step_max_tokens: int = 64       # tokens allowed per reasoning step
    batch_size: int = 64
    # Decoding. Part of the eval cache key, so changing these re-runs the
    # evaluations and leaves the trained adapters untouched. Greedy keeps a
    # rerun bit-identical and makes the diagnostics' clean/injected comparison
    # exact rather than noisy.
    do_sample: bool = False
    temperature: float = 0.6        # only used when do_sample is True
    top_p: float = 0.8
    eval_seed: int = 0


def _key(obj):
    """Short stable hash of a JSON-able config."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# The test split the adapter keys were built against. Training never reads the
# test set, so an adapter does not depend on its size or seed -- but the fields sit
# in DataConfig and so in the key. They are pinned to these values rather than
# dropped, because dropping them would change the hash and orphan every adapter
# already trained, which is the cost this exists to avoid. Changing the test set
# now re-runs evaluation only.
_ADAPTER_KEY_TEST = dict(n_test=500, test_seed=12345)


def _train_data_spec(dcfg):
    """The dataset half of an adapter key: `_data_spec` with the test split pinned."""
    return _data_spec(replace(dcfg, **_ADAPTER_KEY_TEST))


def _data_spec(dcfg):
    """
    DataConfig as a cache key, with the augmentation fields omitted when nothing
    is augmented.

    Without this, adding the augmentation feature would change the key of every
    pre-existing entry and silently orphan adapters that took hours to train. A
    run that does not augment hashes exactly as it did before the feature existed;
    an augmented run carries the extra fields and is therefore distinct.
    """
    d = asdict(dcfg)
    if not d.get('augment_frac'):
        for f in ('augment_frac', 'augment_p', 'augment_version'):
            d.pop(f, None)
    return d


def _train_spec(tcfg):
    """TrainConfig as a cache key, omitting the recovery fields when there is no
    recovery phase (same reasoning as `_data_spec`)."""
    t = asdict(tcfg)
    if not t.get('recovery_samples'):
        for f in ('recovery_samples', 'recovery_lr'):
            t.pop(f, None)
    return t


# ----------------------------------------------------------------------------
# datasets (cached)
# ----------------------------------------------------------------------------

def get_problems(dcfg, split):
    """
    The `split` ('train' or 'test') problem list for `dcfg`, generated once and
    cached to ``cache/datasets``.
    """
    n = dcfg.n_train if split == 'train' else dcfg.n_test
    seed = dcfg.train_seed if split == 'train' else dcfg.test_seed
    spec = dict(split=split, n=n, seed=seed, min_ops=dcfg.min_ops,
                max_ops=dcfg.max_ops, gen_version=dcfg.gen_version)
    path = os.path.join(DATA_CACHE, _key(spec) + '.json')
    if os.path.exists(path):
        with open(path) as f:
            return [Problem.from_dict(d) for d in json.load(f)]
    # SymPy-bound and slow (~40 problems/s): announce it rather than going quiet
    print(f'generating the {split} set ({n} problems, ~{n / 40:.0f}s) ...', flush=True)
    probs = make_dataset(n, seed=seed, min_ops=dcfg.min_ops, max_ops=dcfg.max_ops)
    os.makedirs(DATA_CACHE, exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump([p.to_dict() for p in probs], f)
    os.replace(tmp, path)
    return probs


def mean_tokens_per_example(interval, dcfg, mcfg, tcfg, n_sample=600):
    """
    Mean tokenised length of a training example at this state-emission interval.

    Used to size the fixed-compute budgets: conditions differ in trace length, so
    equal *tokens* means unequal sample counts. Measured on a sample of the real
    training pool with the real tokenizer and cached, since it only depends on the
    data and the interval.
    """
    spec = dict(task='math-meantok', interval=interval, data=_train_data_spec(dcfg),
                base_model=mcfg.base_model, max_seq_len=tcfg.max_seq_len, n=n_sample)
    path = os.path.join(DATA_CACHE, _key(spec) + '.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)['mean_tokens']

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mcfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    problems = get_problems(dcfg, 'train')[:n_sample]
    lens = [len(e['input_ids']) for e in
            (encode_example(tok, p, interval, tcfg.max_seq_len) for p in problems)
            if e is not None]
    mean = sum(lens) / max(len(lens), 1)
    os.makedirs(DATA_CACHE, exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(dict(interval=interval, mean_tokens=mean, n=len(lens)), f)
    os.replace(tmp, path)
    return mean


# ----------------------------------------------------------------------------
# tokenisation (chat template, assistant-only loss)
# ----------------------------------------------------------------------------

def chat_prefix(tokenizer, problem):
    """The templated system+user turns plus the assistant generation header."""
    msgs = [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt_for(problem)}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode_example(tokenizer, problem, interval, max_seq_len, trace=None):
    """
    One training example: chat-templated prompt + trace, with the prompt tokens
    masked out of the loss.

    Returns
    -------
    dict | None
        ``{'input_ids', 'labels'}``, or None if it does not fit `max_seq_len`.
        `trace` overrides the clean trace, which is how augmented traces enter.
    """
    prefix = chat_prefix(tokenizer, problem)
    if trace is None:
        trace = encode_trace(problem, interval)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
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


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

def _adapter_path(interval, dcfg, mcfg, tcfg):
    """Cache directory for the LoRA trained at this interval."""
    spec = dict(task='math', interval=interval, data=_train_data_spec(dcfg),
                model=asdict(mcfg), train=_train_spec(tcfg))
    return os.path.join(ADAPTER_CACHE, _key(spec))


ADAPTER_DONE = 'done.json'          # written last; its presence means "complete"


def adapter_is_cached(interval, dcfg, mcfg, tcfg):
    """True if a *complete* adapter for this config is on disk."""
    return os.path.exists(os.path.join(
        _adapter_path(interval, dcfg, mcfg, tcfg), ADAPTER_DONE))


def train_adapter(interval, dcfg, mcfg, tcfg, device='cuda', force=False,
                  log=print, progress_pos=0):
    """
    Fine-tune a LoRA for one state-emission `interval`, or reuse the cached one.

    Training consumes `tcfg.total_samples` samples from the pool, reshuffling each
    time it is exhausted, under a cosine schedule over that budget. The budget is
    in samples rather than epochs because the fixed-compute experiment gives each
    condition a different sample count so that every condition sees the same
    number of *tokens*.

    The adapter is written to a temporary directory and moved into place only once
    complete, with a `done.json` marker as the completion signal, so an
    interrupted run cannot leave a half-written adapter behind.

    Params
    ------
    interval : int | None
        State-emission interval; None is the standard condition.
    dcfg, mcfg, tcfg : DataConfig, ModelConfig, TrainConfig
    force : bool
        Retrain even if a cached adapter exists.

    Returns
    -------
    str
        Path to the adapter for `tcfg.total_samples`.
    """
    from dataclasses import replace
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
    from peft import LoraConfig, get_peft_model, PeftModel

    recovery = tcfg.recovery_samples > 0
    final_path = _adapter_path(interval, dcfg, mcfg, tcfg)
    if not force and adapter_is_cached(interval, dcfg, mcfg, tcfg):
        with open(os.path.join(final_path, ADAPTER_DONE)) as f:
            meta = json.load(f)
        log(f'[k={interval}] adapter CACHED ({meta.get("total_samples", 0):,} samples, '
            f'loss {meta.get("final_loss", float("nan")):.4f})')
        return final_path

    torch.manual_seed(tcfg.seed)
    random.seed(tcfg.seed)
    dtype = getattr(torch, tcfg.dtype)

    log(f'[k={interval}] loading {mcfg.base_model} on {device} ...')
    tok = AutoTokenizer.from_pretrained(mcfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mcfg.base_model, dtype=dtype)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    if recovery:
        # continue from the already-trained adapter rather than starting over
        base_path = train_adapter(interval, replace(dcfg, augment_frac=0.0), mcfg,
                                  replace(tcfg, recovery_samples=0), device=device,
                                  force=False, log=log, progress_pos=progress_pos)
        log(f'[k={interval}] recovery: continuing from {os.path.basename(base_path)}')
        model = PeftModel.from_pretrained(model, base_path, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=mcfg.lora_r, lora_alpha=mcfg.lora_alpha, lora_dropout=mcfg.lora_dropout,
            target_modules=list(mcfg.target_modules), bias='none', task_type='CAUSAL_LM'))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    budget = int(tcfg.recovery_samples if recovery else tcfg.total_samples)
    pool = get_problems(dcfg, 'train')
    # Build only as many examples as the budget will actually consume. A recovery
    # phase takes 2,000 samples out of a 50,000-problem pool, and building an
    # augmented trace costs a SymPy solve, so materialising the whole pool would
    # be ~7 minutes of work per condition to use 4% of it.
    if budget < len(pool):
        sel = random.Random(f'pool|{tcfg.seed}|{interval}').sample(range(len(pool)), budget)
        problems = [pool[i] for i in sel]
    else:
        problems = pool

    arng = random.Random(f'augment|{tcfg.seed}|{interval}')
    data, n_aug, n_aug_failed = [], 0, 0
    # shown for the clean path too: tokenising 50,000 examples is another
    # minute of silence, and this is exactly the gap that looked like a hang
    build = tqdm(problems, desc=f'k={interval} build data', position=progress_pos,
                 leave=False, dynamic_ncols=True, mininterval=2.0, unit='ex',
                 unit_scale=True)
    for prob in build:
        trace = None
        if dcfg.augment_frac > 0 and arng.random() < dcfg.augment_frac:
            trace = augmented_trace(prob, interval, arng, dcfg.augment_p)
            if trace is None:
                n_aug_failed += 1        # detours did not resolve; use the clean trace
            else:
                n_aug += 1
        e = encode_example(tok, prob, interval, tcfg.max_seq_len, trace=trace)
        if e is not None:
            data.append(e)
    build.close()
    if dcfg.augment_frac > 0:
        log(f'[k={interval}] augmented {n_aug}/{len(problems)} traces at p={dcfg.augment_p} '
            f'({n_aug_failed} fell back to clean)')
    lens = [len(e['input_ids']) for e in data]
    mean_tok = sum(lens) / max(len(lens), 1)
    eff_batch = tcfg.batch_size * tcfg.grad_accum
    opt_steps = (budget + eff_batch - 1) // eff_batch

    log(f'[k={interval}] LoRA {trainable/1e6:.1f}M trainable / {total/1e6:.0f}M total '
        f'({100 * trainable / total:.2f}%)')
    log(f'[k={interval}] pool {len(data)}/{len(problems)} traces fit in {tcfg.max_seq_len} tok '
        f'| mean {mean_tok:.0f} tok, max {max(lens) if lens else 0}')
    log(f'[k={interval}] {"RECOVERY " if recovery else ""}streaming {budget:,} samples '
        f'({budget * mean_tok / 1e6:.1f}M tokens, {opt_steps:,} optimizer steps, '
        f'effective batch {eff_batch})')

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tcfg.recovery_lr if recovery else tcfg.lr,
                            weight_decay=tcfg.weight_decay)
    sched = get_scheduler(tcfg.schedule, opt,
                          num_warmup_steps=int(tcfg.warmup_frac * opt_steps),
                          num_training_steps=opt_steps)

    def save(n_samples, loss, secs, toks):
        """Write the trained adapter to the exact path the cache check uses."""
        path = final_path
        tmp = path + f'.tmp{os.getpid()}'
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(tmp)
        with open(os.path.join(tmp, ADAPTER_DONE), 'w') as f:
            json.dump(dict(interval=interval, total_samples=n_samples, final_loss=loss,
                           seconds=secs, train_tokens=toks, mean_tokens=mean_tok), f, indent=1)
        os.makedirs(ADAPTER_CACHE, exist_ok=True)
        shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)
        log(f'[k={interval}] saved @ {n_samples:,} samples '
            f'({toks/1e6:.1f}M tok, loss {loss:.4f}, {secs/60:.1f} min)')

    rng = random.Random(tcfg.seed)
    order = list(range(len(data)))
    rng.shuffle(order)
    cursor = 0

    model.train()
    t0 = time.time()
    consumed = tok_seen = micro = 0
    running = []
    bar = tqdm(total=budget, desc=f'k={interval} train', position=progress_pos,
               leave=True, dynamic_ncols=True, mininterval=2.0, unit='ex', unit_scale=True)
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
        bar.update(take)
        if micro % (tcfg.grad_accum * 20) == 0:
            el = time.time() - t0
            bar.set_postfix(loss=f'{sum(running[-200:]) / len(running[-200:]):.4f}',
                            tok_s=f'{tok_seen / max(el, 1e-9):.0f}', refresh=False)
    bar.close()
    save(budget, sum(running[-200:]) / max(len(running[-200:]), 1),
         time.time() - t0, tok_seen)

    log(f'[k={interval}] TRAINED {consumed:,} samples in {(time.time()-t0)/60:.1f} min '
        f'({tok_seen/max(time.time()-t0,1e-9):.0f} tok/s)')
    del model
    torch.cuda.empty_cache()
    return final_path


# ----------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------

def load_for_eval(adapter_path, mcfg, tcfg, device='cuda'):
    """Load the base model with a trained adapter attached, ready to generate."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(mcfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'                       # required for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        mcfg.base_model, dtype=getattr(torch, tcfg.dtype))
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device).eval()
    model.config.use_cache = True
    return model, tok


def _split_first_step(text):
    """
    Cut `text` at the end of its first reasoning step.

    Steps end at a period; the final answer step (``#### 1, 2``) instead runs to
    the end of the line. Returns ``(step, is_answer)``, or ``(None, False)`` if no
    complete step is present yet.
    """
    if ANSWER_MARKER in text:
        head, tail = text.split(ANSWER_MARKER, 1)
        if head.strip():                            # a normal step precedes the answer
            i = head.find('.')
            if i != -1:
                return head[:i + 1].strip(), False
        return (ANSWER_MARKER + tail.split('\n')[0]).strip(), True
    i = text.find('.')
    if i == -1:
        return None, False
    return text[:i + 1].strip(), False


@torch.no_grad()
def generate_traces(model, tok, problems, ecfg, inject_p=0.0, device='cuda',
                    desc='decode', progress_pos=0, return_records=False, seed_tag=0):
    """
    Decode a trace for each problem, optionally injecting random operations.

    With ``inject_p == 0`` this is ordinary batched greedy decoding. Otherwise the
    trace is built one reasoning step at a time and, with probability `inject_p`,
    a step the model just produced is replaced by a random solution-preserving
    operation. Only *operation* steps are eligible: steps beginning with
    ``Current equation is`` (the model reporting its state) and the final
    ``####`` answer are never injected into.

    Returns
    -------
    list[str]
        The assistant-side trace decoded for each problem, in order.
    """
    # sampled decoding draws from the global torch RNG, so seed it per pass to
    # keep a rerun (and the clean/injected pair a diagnostic compares) reproducible
    torch.manual_seed(abs(hash((ecfg.eval_seed, float(inject_p), seed_tag))) % (2 ** 31))

    out = [''] * len(problems)
    records = [[] for _ in problems]
    n_batches = (len(problems) + ecfg.batch_size - 1) // ecfg.batch_size
    bar = tqdm(total=n_batches, desc=desc, position=progress_pos, leave=False,
               dynamic_ncols=True, mininterval=2.0)
    for s0 in range(0, len(problems), ecfg.batch_size):
        chunk = problems[s0:s0 + ecfg.batch_size]
        prefixes = [chat_prefix(tok, p) for p in chunk]
        bodies = [''] * len(chunk)
        recs = [[] for _ in chunk]
        done = [False] * len(chunk)
        # a per-problem stream so injections do not depend on batch composition
        rngs = [random.Random(f'{ecfg.eval_seed}|{s0 + i}|{inject_p}')
                for i in range(len(chunk))]

        used = 0                      # decode steps this batch actually consumed
        for _ in range(ecfg.max_steps):
            live = [i for i in range(len(chunk)) if not done[i]]
            if not live:
                break
            used += 1
            texts = [prefixes[i] + bodies[i] for i in live]
            enc = tok(texts, return_tensors='pt', padding=True,
                      add_special_tokens=False).to(device)
            gen = model.generate(
                **enc, max_new_tokens=ecfg.step_max_tokens,
                do_sample=ecfg.do_sample,
                temperature=ecfg.temperature if ecfg.do_sample else None,
                top_p=ecfg.top_p if ecfg.do_sample else None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.convert_tokens_to_ids('<|im_end|>'))
            new = gen[:, enc['input_ids'].shape[1]:]
            for j, i in enumerate(live):
                piece = tok.decode(new[j], skip_special_tokens=True)
                step, is_answer = _split_first_step(piece)
                if step is None or not step:                 # nothing usable -> stop
                    done[i] = True
                    bodies[i] += piece
                    continue
                if is_answer:
                    bodies[i] += (' ' if bodies[i] else '') + step
                    recs[i].append(dict(text=step, kind='answer', injected=False))
                    done[i] = True
                    continue
                # inject only into operation steps, never state reports
                is_state = step.startswith(STATE_PREFIX)
                injected = False
                if inject_p > 0 and not is_state and rngs[i].random() < inject_p:
                    step = random_injection(rngs[i])
                    injected = True
                bodies[i] += (' ' if bodies[i] else '') + step
                recs[i].append(dict(text=step, kind='state' if is_state else 'op',
                                    injected=injected))
        for i in range(len(chunk)):
            out[s0 + i] = bodies[i]
            records[s0 + i] = recs[i]
        bar.update(1)
        # `steps` is decode steps used out of the cap; `done` is how many
        # sequences terminated on their own rather than being cut off
        bar.set_postfix(steps=f'{used}/{ecfg.max_steps}',
                        done=f'{sum(done)}/{len(chunk)}', refresh=False)
    bar.close()
    return (out, records) if return_records else out


# ----------------------------------------------------------------------------
# decode cache
# ----------------------------------------------------------------------------
#
# Decoding the test set is the expensive part of evaluation, and the same decode
# is wanted by more than one consumer: `run_unit` scores it for accuracy and the
# diagnostics replay it step by step. Caching the decode itself means the second
# consumer -- in this run or a later one -- reads a few MB from disk instead of
# occupying a GPU again. At ~1-2 MB per (interval, inject_p) the whole sweep is
# a few hundred MB.

def _decode_path(interval, inject_p, dcfg, mcfg, tcfg, ecfg):
    spec = dict(task='math-decode', interval=interval, inject_p=inject_p,
                data=_data_spec(dcfg), model=asdict(mcfg), train=_train_spec(tcfg),
                eval=asdict(ecfg))
    return os.path.join(DECODE_CACHE, _key(spec) + '.json')


def get_or_decode(get_model, problems, interval, inject_p, dcfg, mcfg, tcfg, ecfg,
                  device='cuda', desc=None, progress_pos=0, force=False, log=print,
                  batch_override=None):
    """
    The decoded traces for one (interval, inject_p), from cache when possible.

    Params
    ------
    get_model : callable
        Zero-argument callable returning ``(model, tokenizer)``. It is only
        invoked on a cache miss, so a fully cached condition never loads a model
        or touches the GPU.
    force : bool
        Decode again even if a cached decode exists.
    batch_override : int | None
        Decode with this generation batch size instead of ``ecfg.batch_size``.
        It changes only the runtime batching, never the result (greedy decoding
        is batch-invariant) and never the cache key -- the entry is still written
        and read under the canonical ``ecfg``. This is what lets a caller shrink
        the batch to survive an OOM without orphaning the decode: a later run at
        the default batch still finds it.

    Returns
    -------
    (list[str], list[list[dict]])
        Traces and per-step records, as `generate_traces` returns them.
    """
    path = _decode_path(interval, inject_p, dcfg, mcfg, tcfg, ecfg)
    if os.path.exists(path) and not force:
        try:
            with open(path) as f:
                blob = json.load(f)
            if len(blob['traces']) == len(problems):
                return blob['traces'], blob['records']
            log(f'  [decode] cached decode has {len(blob["traces"])} traces, '
                f'expected {len(problems)}; re-decoding')
        except (json.JSONDecodeError, KeyError, OSError):
            log('  [decode] cached decode unreadable; re-decoding')

    model, tok = get_model()
    run_ecfg = ecfg if batch_override is None else replace(ecfg, batch_size=batch_override)
    traces, records = generate_traces(
        model, tok, problems, run_ecfg, inject_p=inject_p, device=device,
        desc=desc or f'k={interval} p={inject_p} decode',
        progress_pos=progress_pos, return_records=True)

    os.makedirs(DECODE_CACHE, exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(dict(interval=interval, inject_p=inject_p,
                       traces=traces, records=records), f)
    os.replace(tmp, path)
    return traces, records


# ----------------------------------------------------------------------------
# one (interval, inject_p) unit, cached
# ----------------------------------------------------------------------------

def _eval_path(interval, inject_p, dcfg, mcfg, tcfg, ecfg):
    spec = dict(task='math', interval=interval, inject_p=inject_p,
                data=_data_spec(dcfg), model=asdict(mcfg),
                train=_train_spec(tcfg), eval=asdict(ecfg))
    return os.path.join(EVAL_CACHE, _key(spec) + '.json')


def cached_unit(interval, inject_p, dcfg, mcfg, tcfg, ecfg):
    """The cached result for a unit, or None -- without loading any model."""
    path = _eval_path(interval, inject_p, dcfg, mcfg, tcfg, ecfg)
    if os.path.exists(path):
        with open(path) as f:
            r = json.load(f)
        r['cached'] = True
        return r
    return None


def run_unit(interval, inject_p, dcfg, mcfg, tcfg, ecfg, device='cuda',
             force=False, model_pack=None, log=print, progress_pos=0,
             get_model=None):
    """
    Evaluate the model trained at `interval` under injection probability
    `inject_p`, training the adapter first if it is not cached.

    Params
    ------
    interval : int | None
        State-emission interval (None = standard).
    inject_p : float
        Per-operation-step injection probability (0 = clean).
    model_pack : (model, tok) | None
        A already-loaded model to reuse across the p values of one interval.
    get_model : callable | None
        Zero-argument callable returning ``(model, tokenizer)``, invoked only if
        the decode is not cached. Defaults to training/loading on demand.
    force : bool
        Recompute even if cached.

    Returns
    -------
    dict
        {'interval', 'inject_p', 'accuracy', 'n_eval', 'answered', 'cached'}
    """
    path = _eval_path(interval, inject_p, dcfg, mcfg, tcfg, ecfg)
    if os.path.exists(path) and not force:
        with open(path) as f:
            r = json.load(f)
        r['cached'] = True
        return r

    problems = get_problems(dcfg, 'test')
    t0 = time.time()

    def _default_model():
        adapter = train_adapter(interval, dcfg, mcfg, tcfg, device=device, force=force,
                                log=log, progress_pos=progress_pos)
        return model_pack or load_for_eval(adapter, mcfg, tcfg, device=device)

    traces, _ = get_or_decode(get_model or _default_model, problems, interval, inject_p,
                              dcfg, mcfg, tcfg, ecfg, device=device,
                              progress_pos=progress_pos, force=force, log=log)
    n_ok = sum(is_correct(p, t) for p, t in zip(problems, traces))
    answered = sum(ANSWER_MARKER in t for t in traces)
    r = dict(interval=interval, inject_p=inject_p,
             accuracy=n_ok / len(problems), answered=answered / len(problems),
             n_eval=len(problems), seconds=time.time() - t0, cached=False)

    os.makedirs(EVAL_CACHE, exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(dict(r, samples=traces[:20]), f, indent=1)
    os.replace(tmp, path)
    log(f'[k={interval} p={inject_p}] acc={r["accuracy"]:.3f}  '
        f'answered={r["answered"]:.3f}  ({r["seconds"]:.0f}s for {len(problems)} problems, '
        f'{r["seconds"]/max(len(problems),1)*1000:.0f} ms/problem)')
    return r
