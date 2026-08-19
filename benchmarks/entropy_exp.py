import os
import re
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import gc
import json

# -----------------------------------------------------------------------------
# COMMAND-LINE ARGUMENTS
# -----------------------------------------------------------------------------
# Parsed before any CUDA call so --devices can pin CUDA_VISIBLE_DEVICES; the
# model is then sharded across the selected GPUs by device_map="auto".

def parse_args():
    p = argparse.ArgumentParser(description="Teacher-forced analyses of Qwen3 models over "
                                            "reasoning datasets and a WikiText baseline: "
                                            "an attention-knockout perplexity-vs-window "
                                            "sweep and an entropy-vs-token-position curve.")
    p.add_argument('--experiment',
                   choices=['knockout', 'knockout-generation', 'entropy', 'all'],
                   default='all',
                   help="'knockout': perplexity of the ground-truth traces vs the "
                        "memory-window size n. 'knockout-generation': generate answers "
                        "under the knockout and plot accuracy vs n. 'entropy': mean "
                        "entropy vs answer-token position. 'all': run every "
                        "experiment in turn (VRAM is freed between them).")
    # ---- knockout-generation experiment ----
    p.add_argument('--gen-samples', type=int, default=192,
                   help="problems sampled without replacement per benchmark for the "
                        "generation experiment (GPQA uses all if it has fewer).")
    p.add_argument('--gen-max-new-tokens', type=int, default=12800,
                   help="max tokens to generate per answer.")
    p.add_argument('--gen-temperature', type=float, default=0.6, help="sampling temperature.")
    p.add_argument('--gen-top-p', type=float, default=0.95, help="nucleus (top-p) sampling.")
    p.add_argument('--gen-seed', type=int, default=42, help="seed for problem sampling + decoding.")
    p.add_argument('--gen-batch-size', type=int, default=32,
                   help="problems generated in parallel per batch (rolling-KV-cache path).")
    p.add_argument('--gen-full-batch-size', type=int, default=8,
                   help="batch size used only for the full-context reference column, "
                        "whose KV cache grows with the generation length (unlike the "
                        "window-bounded knockout columns). Set smaller than "
                        "--gen-batch-size to avoid OOM on long generations. "
                        )
    p.add_argument('--gen-no-cache', action='store_true',
                   help="use the slow per-problem path (no batching / KV cache) as a "
                        "robustness fallback if the batched path misbehaves.")
    p.add_argument('--devices', nargs='+', default=None, metavar='cuda:N',
                   help="GPUs to use, e.g. --devices cuda:0 cuda:1 cuda:2 cuda:3 "
                        "(bare indices like 0 1 also accepted). Sets "
                        "CUDA_VISIBLE_DEVICES; the model is sharded across them. "
                        "Default: all visible GPUs.")
    p.add_argument('--knockout-gen-ns', type=str, default="20,30,40,50,60,70,80,90,100,120,140,160,180,200,400,500,600,700,800,900,1000",
                   help="comma-separated memory-window sizes n (prompt + n most recent "
                        "tokens) to sweep, roughly 10 values.")
    # ---- knockout experiment ----
    p.add_argument('--knockout-ns', type=str, default="5,10,15,20,30,40,50,60,70,80,90,100",
                   help="comma-separated memory-window sizes n (prompt + n most recent "
                        "tokens) to sweep, roughly 10 values in 0-100.")
    p.add_argument('--min-answer-len', type=int, default=100,
                   help="knockout: only include sequences with >= this many answer "
                        "tokens (matches the 'filter answers shorter than 100' rule).")
    # ---- entropy experiment ----
    p.add_argument('--max-analysis-len', type=int, default=100,
                   help="entropy: analyse the first N answer tokens; only sequences with "
                        ">=N answer tokens contribute (constant N, no survivorship).")
    p.add_argument('--cache-min-len', type=int, default=32,
                   help="entropy: only compute/cache sequences with >= this many answer "
                        "tokens. Bounds forward-pass cost and sets how low "
                        "--max-analysis-len can go without recomputing traces.")
    # ---- shared ----
    p.add_argument('--model', type=str, default="Qwen/Qwen3-14B",
                   help="HF model id to evaluate (e.g. Qwen/Qwen3-32B, Qwen/Qwen3-14B). "
                        "Caches are namespaced per model, so switching does not clobber "
                        "another model's results.")
    p.add_argument('--num-samples', type=int, default=None,
                   help="cap problems per dataset before computing (default: all; "
                        "use to bound cost, especially for GSM8K).")
    p.add_argument('--plot-only', action='store_true',
                   help="only re-aggregate/plot from cache (no model load).")
    p.add_argument('--force', action='store_true',
                   help="recompute the cache even if present and fresh.")
    return p.parse_args()

ARGS = parse_args()
if ARGS.devices:
    _indices = [str(d)[len('cuda:'):] if str(d).lower().startswith('cuda:') else str(d)
                for d in ARGS.devices]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(_indices)
# expandable segments let the caching allocator grow/shrink blocks, cutting the
# reserved-but-unallocated fragmentation that otherwise wastes GiB during the
# batched prefill and the growing full-context KV cache.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_NAME = ARGS.model
# filesystem-safe per-model tag used to namespace every cache directory, so
# results for different models (e.g. Qwen3-32B vs Qwen3-14B) never collide.
MODEL_SLUG = re.sub(r'[^A-Za-z0-9._-]', '_', MODEL_NAME.split('/')[-1])
NUM_WIKI_SAMPLES = 5000         # WikiText passages to consider before filtering
MAX_SEQ_LEN = 1024              # forward-pass / cached-trace length cap
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Raw per-token entropy traces are cached here (one .npz per dataset). Because a
# forward pass yields the whole sequence anyway, we cache the FULL answer-region
# entropy per sequence (not a fixed window), so --max-analysis-len (and a future
# percent-of-solution view) can be recomputed from cache without re-running the
# model.
TRACES_DIR = "entropy_traces"
# Attention-knockout perplexity cache (one .npz per dataset: a per-sequence x
# per-n perplexity matrix), so the perplexity-vs-n plot re-aggregates cheaply.
KNOCKOUT_DIR = "knockout_ppl"
# Attention-knockout generation cache (one .npz per dataset: a per-problem x
# per-n correctness matrix, -1 = not-yet-computed, so runs resume cell-by-cell
# and nothing is regenerated).
KNOCKOUT_GEN_DIR = "knockout_gen"
FIGURES_DIR = "figures"          # all plots are written here

print(f"Running on device: {DEVICE}"
      + (f" | CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}" if ARGS.devices else ""))

# -----------------------------------------------------------------------------
# 1. LOAD MODEL AND TOKENIZER
# -----------------------------------------------------------------------------

def load_tokenizer():
    print(f"Loading tokenizer: {MODEL_NAME}...")
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def load_model(attn_implementation=None):
    print(f"Loading model: {MODEL_NAME}...")
    kwargs = dict(
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE == "cuda":
        # Spread the weights EVENLY across all visible GPUs. Two placement traps to
        # avoid: (a) plain device_map="auto" packs the model onto the fewest GPUs
        # that fit; (b) an explicit per-GPU max_memory cap makes accelerate fill the
        # low-index GPUs greedily and leave the rest empty. Both concentrate the
        # weights on GPU 0, so the batched prefill activation (~17 GiB on the hot
        # GPU at batch 64) lands on an already-loaded GPU and OOMs while GPUs sit
        # idle. get_balanced_memory computes an EVEN per-GPU weight budget: a 14B
        # model lands as ~5 GiB on each of 6 GPUs, leaving ~19 GiB free per GPU for
        # the prefill activations and the growing KV cache.
        try:
            from accelerate import init_empty_weights
            from accelerate.utils import get_balanced_memory
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(MODEL_NAME)
            with init_empty_weights():
                probe = AutoModelForCausalLM.from_config(cfg, torch_dtype=kwargs['dtype'])
            probe.tie_weights()
            bal = get_balanced_memory(
                probe, dtype=kwargs['dtype'], low_zero=True,   # keep GPU0 lighter
                no_split_module_classes=getattr(probe, "_no_split_modules", None))
            del probe
            gc.collect()
            kwargs['max_memory'] = {d: b for d, b in bal.items() if isinstance(d, int)}
        except Exception as e:
            print(f"  get_balanced_memory unavailable ({e}); using device_map='auto'")
    # the knockout experiment passes a custom 4D attention mask; eager attention
    # applies a user-supplied mask predictably (added to the scores)
    if attn_implementation is not None:
        kwargs['attn_implementation'] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 2. DATA LOADING & FILTERING
# -----------------------------------------------------------------------------

def get_gsm8k_data(n=None):
    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="train")
    data_items = []
    
    selection = ds if n is None else ds.select(range(min(n, len(ds))))
        
    for ex in selection:
        prompt = ex['question']
        # We append the answer to measure entropy of generating it
        full = f"{ex['question']}\n{ex['answer']}"
        data_items.append({"prompt": prompt, "full": full})

    return data_items

def get_math500_data(n=None):
    print("Loading MATH-500...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    data_items = []

    selection = ds if n is None else ds.select(range(min(n, len(ds))))

    for ex in selection:
        prompt = ex['problem']
        # Teacher-force the ground-truth worked solution, same as GSM8K
        full = f"{ex['problem']}\n{ex['solution']}"
        data_items.append({"prompt": prompt, "full": full})

    return data_items

def get_gpqa_data(n=None):
    # GPQA-Diamond: PhD-level *non-math* reasoning (physics, chemistry, biology).
    # Each question ships an author-written `Explanation`, which we teacher-force
    # as the ground-truth reasoning trace (the non-math analogue of the GSM8K /
    # MATH-500 worked solutions). NOTE: GPQA is a gated dataset -- accept its
    # terms on the Hub and run `huggingface-cli login` before loading.
    print("Loading GPQA-Diamond (non-math science reasoning)...")
    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as e:
        print(f"  could not load GPQA (gated? run huggingface-cli login): {e}")
        return []

    data_items = []
    selection = ds if n is None else ds.select(range(min(n, len(ds))))

    for ex in selection:
        question = ex.get('Question') or ex.get('question')
        explanation = ex.get('Explanation') or ex.get('explanation')
        if not question or not explanation:
            continue
        prompt = str(question).strip()
        # Teacher-force the ground-truth explanation, same idea as GSM8K / MATH-500
        full = f"{prompt}\n{str(explanation).strip()}"
        data_items.append({"prompt": prompt, "full": full})

    return data_items

def get_wikitext_data(n=100, min_char_len=200):
    print("Loading WikiText-2 (Normal Text)...")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        data_items = []

        # Filter for non-empty lines and meaningful content
        valid_indices = [i for i, x in enumerate(ds['text']) if len(x.strip()) > min_char_len and not x.strip().startswith('=')]

        count = min(n, len(valid_indices))
        selected_indices = valid_indices[:count]

        for idx in selected_indices:
            text = ds[idx]['text'].strip()

            # No prompt for WikiText initially
            prompt = ""
            full = text

            data_items.append({"prompt": prompt, "full": full})

        return data_items
    except Exception as e:
        print(f"Failed to load WikiText: {e}")
        return []

# -----------------------------------------------------------------------------
# 3. ENTROPY COMPUTATION
# -----------------------------------------------------------------------------

def calculate_entropy(logits):
    """
    Calculates Shannon entropy from logits.
    H(x) = - sum(p(x) * log(p(x)))
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    # Compute log probabilities (log_softmax is more numerically stable)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Entropy = - sum(p * log p)
    # We use sum(dim=-1) to sum over the vocabulary dimension
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def compute_traces(model, tokenizer, data_items, dataset_name, cache_min_len, fixed_start_idx=None):
    """
    Forward-pass each item and return the FULL per-token entropy (nats) over the
    answer region -- from the prompt end to the end of the (truncated) sequence
    -- as one np.array per sequence.

    Because a single forward pass already produces logits for the whole sequence,
    caching the entire answer-region trace (rather than a fixed 100-token window)
    is essentially free and lets the analysis window / percent view be recomputed
    later with no model calls. Sequences with fewer than `cache_min_len` answer
    tokens are skipped.

    Params
    ------
    model, tokenizer : the loaded model and its tokenizer.
    data_items : list[dict] with 'prompt' and 'full' text.
    dataset_name : str (drives the default start index).
    cache_min_len : int, minimum answer length (tokens) to keep.
    fixed_start_idx : int | None, explicit start logit index (WikiText alignment).

    Returns
    -------
    traces : list[np.ndarray], per-sequence answer-region entropy (float32, nats).
    """
    traces = []
    print(f"Computing entropy traces for {dataset_name}...")

    for item in tqdm(data_items):
        inputs = tokenizer(item['full'], return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to(model.device)
        seq_len = inputs.input_ids.shape[1]

        # Where the answer starts (logit at idx predicts the token at idx+1)
        if fixed_start_idx is not None:
            start_idx = fixed_start_idx
        elif dataset_name == "WikiText":
            start_idx = 10
        else:
            prompt_ids = tokenizer.encode(item['prompt'], add_special_tokens=True)
            start_idx = len(prompt_ids) - 1

        # Answer-region logit indices are start_idx .. seq_len-2
        n_answer = (seq_len - 1) - start_idx
        if start_idx < 0 or n_answer < cache_min_len:
            del inputs
            continue

        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(0)   # (seq_len, vocab)
        seq_entropy = calculate_entropy(logits)          # (seq_len,) nats
        trace = seq_entropy[start_idx: seq_len - 1].detach().to(torch.float32).cpu().numpy()
        traces.append(trace)

        del inputs, logits, seq_entropy
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"  cached {len(traces)} traces (answer len >= {cache_min_len})")
    return traces

# -----------------------------------------------------------------------------
# 3b. TRACE CACHE  (raw per-token entropy; the expensive-compute cache)
# -----------------------------------------------------------------------------

def traces_path(name):
    return os.path.join(TRACES_DIR, MODEL_SLUG, f"{name}.npz")

def save_traces(name, traces, meta):
    """Persist ragged per-sequence traces flat (values + lengths) with a meta dict."""
    os.makedirs(os.path.dirname(traces_path(name)), exist_ok=True)
    lengths = np.array([len(t) for t in traces], dtype=np.int64)
    flat = np.concatenate(traces).astype(np.float32) if traces else np.zeros(0, np.float32)
    np.savez_compressed(traces_path(name), flat=flat, lengths=lengths, meta=json.dumps(meta))

def load_traces(name):
    """Reconstruct the per-sequence traces and meta from cache."""
    data = np.load(traces_path(name), allow_pickle=False)
    flat, lengths = data['flat'], data['lengths']
    out, off = [], 0
    for n in lengths:
        out.append(flat[off:off + int(n)])
        off += int(n)
    meta = json.loads(str(data['meta'])) if 'meta' in data.files else {}
    return out, meta

def traces_fresh(name, cache_min_len, num_samples):
    """True iff a cached trace file exists that matches the current compute knobs."""
    if not os.path.exists(traces_path(name)):
        return False
    try:
        with np.load(traces_path(name), allow_pickle=False) as d:
            meta = json.loads(str(d['meta'])) if 'meta' in d.files else {}
    except Exception:
        return False
    return (meta.get('model') == MODEL_NAME
            and meta.get('cache_min_len') == cache_min_len
            and meta.get('num_samples') == num_samples)

# -----------------------------------------------------------------------------
# 3c. AGGREGATION  (cheap; recomputed at plot time from cached traces)
# -----------------------------------------------------------------------------

def aggregate_tokens(traces, max_len):
    """
    Fixed-window (raw token position) aggregation. Only sequences with at least
    `max_len` answer tokens contribute -- so N is constant across all steps and
    there is no survivorship bias -- and their first `max_len` tokens are stacked.

    Returns (steps, mean, sem) with steps 1..max_len; SEM is the across-sequence
    standard deviation / sqrt(N).
    """
    kept = [np.asarray(t[:max_len], dtype=np.float64) for t in traces if len(t) >= max_len]
    if not kept:
        return [], [], []
    M = np.vstack(kept)
    n = M.shape[0]
    mean = M.mean(axis=0)
    sem = M.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(max_len)
    return list(range(1, max_len + 1)), mean.tolist(), sem.tolist()

# NOTE: traces are cached in full (untruncated), so a percent-of-solution
# aggregation can be added later without recomputing anything -- not implemented
# yet by request.

# -----------------------------------------------------------------------------
# 3d. ATTENTION-KNOCKOUT PERPLEXITY
# -----------------------------------------------------------------------------
#
# For each memory-window size n we restrict every query position to attend only
# to (a) the prompt tokens and (b) the n most recent tokens, via a custom 4D
# attention mask. position_ids are left at their TRUE values, so masking out the
# middle tokens leaves a gap in the (RoPE) positional encoding between the prompt
# and the recent window -- exactly the intended "attention knockout with an
# appropriate positional gap". Perplexity = exp(mean NLL) over the answer tokens.

def parse_n_values(spec):
    """Parse a comma-separated list of window sizes n into a sorted int list."""
    vals = sorted({int(x) for x in str(spec).split(',') if x.strip() != ''})
    return vals

def knockout_perplexity(model, tokenizer, item, n_values, prompt_len_override=None):
    """
    Per-sequence perplexity of the ground-truth trace under prompt+window
    attention, for each n in `n_values`.

    Params
    ------
    model, tokenizer : loaded model and tokenizer.
    item : dict with 'prompt' and 'full' text.
    n_values : list[int], memory-window sizes.
    prompt_len_override : int | None, number of leading tokens to treat as the
        always-visible "prompt" (used for WikiText, which has no real prompt).

    Returns
    -------
    ppls : np.ndarray, shape (len(n_values),), of per-n perplexities. None if the
        sequence has no answer tokens. (Pass n_values=[MAX_SEQ_LEN] to get the
        full-context perplexity, since a window >= L reduces to full causal.)
    n_answer : int, the number of answer (trace) tokens for this sequence.
    """
    inputs = tokenizer(item['full'], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = inputs.input_ids.to(model.device)          # (1, L)
    L = input_ids.shape[1]
    P = (prompt_len_override if prompt_len_override is not None
         else len(tokenizer.encode(item['prompt'], add_special_tokens=True)))
    P = max(1, min(P, L - 1))                               # >=1 prompt tok, >=1 answer tok
    n_answer = L - P
    if n_answer < 1:
        return None, 0

    device = input_ids.device
    dtype = next(model.parameters()).dtype
    neg = torch.finfo(dtype).min
    position_ids = torch.arange(L, device=device).unsqueeze(0)      # TRUE positions -> RoPE gap
    q = torch.arange(L, device=device).unsqueeze(1)                 # (L,1) query index
    k = torch.arange(L, device=device).unsqueeze(0)                 # (1,L) key index
    causal = k <= q
    prompt_sink = k < P
    targets = input_ids[0, P:L]                                     # (n_answer,)
    idx = torch.arange(n_answer, device=device)

    ppls = []
    for n in n_values:
        # allowed: causal AND (prompt OR within the n most recent tokens, i.e. k > q-n)
        allowed = causal & (prompt_sink | (k > (q - n)))
        mask = torch.where(allowed,
                           torch.zeros((), dtype=dtype, device=device),
                           torch.full((), neg, dtype=dtype, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)                       # (1,1,L,L) additive
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=mask,
                           position_ids=position_ids).logits[0]     # (L, V)
        # predict answer token t in [P,L) from the logits at position t-1
        logp = torch.log_softmax(logits[P - 1:L - 1].float(), dim=-1)
        nll = -logp[idx, targets]
        ppls.append(float(torch.exp(nll.mean())))
        del logits, logp, nll
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return np.array(ppls, dtype=np.float32), n_answer

def compute_knockout(model, tokenizer, data_items, n_values, min_answer_len,
                     dataset_name, prompt_len_override=None):
    """
    Run `knockout_perplexity` over a dataset, keeping only sequences with at
    least `min_answer_len` answer tokens.

    Returns
    -------
    ppl_matrix : np.ndarray, shape (n_kept, len(n_values)); one row per sequence.
    mean_trace_len : float, the average answer (trace) length over kept sequences.
    """
    rows, lengths = [], []
    print(f"Computing knockout perplexity for {dataset_name} "
          f"({len(n_values)} window sizes)...")
    for item in tqdm(data_items):
        full_ids = tokenizer.encode(item['full'], add_special_tokens=True)
        P = (prompt_len_override if prompt_len_override is not None
             else len(tokenizer.encode(item['prompt'], add_special_tokens=True)))
        answer_len = min(len(full_ids), MAX_SEQ_LEN) - P
        if answer_len < min_answer_len:
            continue
        ppls, n_answer = knockout_perplexity(model, tokenizer, item, n_values, prompt_len_override)
        if ppls is not None:
            rows.append(ppls)
            lengths.append(n_answer)
    mean_trace_len = float(np.mean(lengths)) if lengths else 0.0
    print(f"  kept {len(rows)} sequences (answer len >= {min_answer_len}); "
          f"mean trace length = {mean_trace_len:.1f} tokens")
    matrix = np.vstack(rows) if rows else np.zeros((0, len(n_values)), np.float32)
    return matrix, mean_trace_len

def compute_knockout_full(model, tokenizer, data_items, min_answer_len,
                          dataset_name, prompt_len_override=None):
    """
    Full-context perplexity (entire context, full causal attention) per sequence.
    Uses the same filter as `compute_knockout` so the kept sequences match, but is
    cached separately so it can be computed without recomputing the window sweep.

    Returns
    -------
    ppl_full : np.ndarray, shape (n_kept,); the full-context perplexity per seq.
    mean_trace_len : float, the average answer (trace) length over kept sequences.
    """
    vals, lengths = [], []
    print(f"Computing full-context perplexity for {dataset_name}...")
    for item in tqdm(data_items):
        full_ids = tokenizer.encode(item['full'], add_special_tokens=True)
        P = (prompt_len_override if prompt_len_override is not None
             else len(tokenizer.encode(item['prompt'], add_special_tokens=True)))
        answer_len = min(len(full_ids), MAX_SEQ_LEN) - P
        if answer_len < min_answer_len:
            continue
        # n = MAX_SEQ_LEN (>= L) makes the window cover everything -> full causal
        ppls, n_answer = knockout_perplexity(model, tokenizer, item, [MAX_SEQ_LEN],
                                             prompt_len_override)
        if ppls is not None:
            vals.append(float(ppls[0]))
            lengths.append(n_answer)
    mean_trace_len = float(np.mean(lengths)) if lengths else 0.0
    print(f"  kept {len(vals)} sequences (answer len >= {min_answer_len})")
    return np.asarray(vals, dtype=np.float32), mean_trace_len

def count_and_mean_length(tokenizer, data_items, min_answer_len, prompt_len_override=None):
    """
    (kept count, mean answer length in tokens) using only the tokenizer -- no
    model. Matches compute_knockout's filter, so it annotates the summary even
    when the perplexities were loaded from cache.
    """
    lengths = []
    for item in data_items:
        full_ids = tokenizer.encode(item['full'], add_special_tokens=True)
        P = (prompt_len_override if prompt_len_override is not None
             else len(tokenizer.encode(item['prompt'], add_special_tokens=True)))
        answer_len = min(len(full_ids), MAX_SEQ_LEN) - P
        if answer_len >= min_answer_len:
            lengths.append(answer_len)
    return len(lengths), (float(np.mean(lengths)) if lengths else 0.0)

# The window sweep and the full-context reference are cached SEPARATELY (two
# files per dataset in the same folder: <name>.npz and <name>__full.npz), so
# adding / changing one never forces the other to recompute.

def knockout_path(name):
    return os.path.join(KNOCKOUT_DIR, MODEL_SLUG, f"{name}.npz")

def knockout_full_path(name):
    return os.path.join(KNOCKOUT_DIR, MODEL_SLUG, f"{name}__full.npz")

def save_knockout(name, ppl_matrix, n_values, meta):
    os.makedirs(os.path.dirname(knockout_path(name)), exist_ok=True)
    np.savez_compressed(knockout_path(name), ppl=ppl_matrix.astype(np.float32),
                        n_values=np.asarray(n_values, dtype=np.int64),
                        meta=json.dumps(meta))

def load_knockout(name):
    data = np.load(knockout_path(name), allow_pickle=False)
    meta = json.loads(str(data['meta'])) if 'meta' in data.files else {}
    return data['ppl'], list(data['n_values']), meta

def _sweep_meta_compatible(meta, min_answer_len, num_samples):
    """Whether a cached sweep was computed with the current model / filter knobs
    (independent of which n columns it holds)."""
    return (meta.get('model') == MODEL_NAME
            and meta.get('min_answer_len') == min_answer_len
            and meta.get('num_samples') == num_samples)

def knockout_covers(name, min_answer_len, num_samples, n_values):
    """
    True iff a compatible cached sweep already contains EVERY requested n. The
    cache stores one perplexity column per n, so a run whose n values are a subset
    of the cache needs no computation; a superset only computes the new columns.
    """
    if not os.path.exists(knockout_path(name)):
        return False
    try:
        with np.load(knockout_path(name), allow_pickle=False) as d:
            meta = json.loads(str(d['meta'])) if 'meta' in d.files else {}
            cached_ns = list(d['n_values'])
    except Exception:
        return False
    return (_sweep_meta_compatible(meta, min_answer_len, num_samples)
            and all(n in cached_ns for n in n_values))

def columns_for(matrix, have_ns, want_ns):
    """Select the columns of `matrix` corresponding to `want_ns`, in that order.
    Assumes every n in `want_ns` is present in `have_ns`."""
    have = list(have_ns)
    return matrix[:, [have.index(int(n)) for n in want_ns]]

def save_knockout_full(name, ppl_full, meta):
    os.makedirs(os.path.dirname(knockout_full_path(name)), exist_ok=True)
    np.savez_compressed(knockout_full_path(name),
                        ppl_full=np.asarray(ppl_full, dtype=np.float32),
                        meta=json.dumps(meta))

def load_knockout_full(name):
    data = np.load(knockout_full_path(name), allow_pickle=False)
    meta = json.loads(str(data['meta'])) if 'meta' in data.files else {}
    return data['ppl_full'], meta

def knockout_full_fresh(name, min_answer_len, num_samples):
    """True iff the cached full-context perplexities match the current knobs.
    (Independent of the n sweep -- full context does not depend on n.)"""
    if not os.path.exists(knockout_full_path(name)):
        return False
    try:
        with np.load(knockout_full_path(name), allow_pickle=False) as d:
            meta = json.loads(str(d['meta'])) if 'meta' in d.files else {}
    except Exception:
        return False
    return (meta.get('model') == MODEL_NAME
            and meta.get('min_answer_len') == min_answer_len
            and meta.get('num_samples') == num_samples)

def aggregate_knockout(ppl_matrix):
    """Mean and SEM of perplexity across sequences, per window size n."""
    if ppl_matrix.shape[0] == 0:
        return np.zeros(ppl_matrix.shape[1]), np.zeros(ppl_matrix.shape[1])
    mean = ppl_matrix.mean(axis=0)
    n = ppl_matrix.shape[0]
    sem = ppl_matrix.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(ppl_matrix.shape[1])
    return mean, sem

def resolve_sweep(name, n_values, min_answer_len, num_samples, avg,
                  model, tokenizer, data_items, prompt_override, plot_only, force):
    """
    Return the sweep perplexity matrix for exactly `n_values` (columns in that
    order), reusing any cached columns and computing only the window sizes that
    are missing. The cache is updated to the union of cached + newly computed n,
    so later runs reuse everything.

    Returns None when nothing usable is available (e.g. --plot-only with a cache
    that is missing some requested n).
    """
    requested = [int(n) for n in n_values]

    # load a compatible cache (matching model / filter knobs), if any
    cached_mat, cached_ns, cmeta = None, [], {}
    if os.path.exists(knockout_path(name)) and not force:
        try:
            m0, ns0, meta0 = load_knockout(name)
            if _sweep_meta_compatible(meta0, min_answer_len, num_samples):
                cached_mat, cached_ns, cmeta = m0, [int(n) for n in ns0], meta0
        except Exception:
            cached_mat = None

    missing = [n for n in requested if n not in cached_ns]

    def _save(mat, ns, mtl):
        save_knockout(name, mat, ns, dict(model=MODEL_NAME, dataset=name,
                      min_answer_len=min_answer_len, num_samples=num_samples,
                      n_sequences=int(mat.shape[0]), mean_trace_len=mtl, avg_prompt_len=avg))

    # ---- no computation allowed (plot-only, or model not loaded) ----
    if plot_only or model is None:
        if cached_mat is not None and not missing:
            return columns_for(cached_mat, cached_ns, requested)
        if cached_mat is not None and missing:
            print(f"  {name}: cache is missing n {missing}; run without --plot-only "
                  f"to compute them. Skipping this dataset.")
        return None

    # ---- compute path ----
    if cached_mat is None:                       # nothing reusable -> compute all
        mat, mtl = compute_knockout(model, tokenizer, data_items, requested,
                                    min_answer_len, name, prompt_override)
        if mat.shape[0] == 0:
            return None
        _save(mat, requested, mtl)
        return mat

    if not missing:                              # cache already covers the request
        return columns_for(cached_mat, cached_ns, requested)

    # compute ONLY the missing columns and merge with the cached ones
    print(f"  {name}: reusing cached n {cached_ns}; computing missing n {missing}")
    new_mat, mtl = compute_knockout(model, tokenizer, data_items, missing,
                                    min_answer_len, name, prompt_override)
    if new_mat.shape[0] != cached_mat.shape[0]:
        # sequence set unexpectedly changed -> recompute the whole sweep to stay aligned
        print(f"    row count changed ({cached_mat.shape[0]} -> {new_mat.shape[0]}); "
              f"recomputing the full sweep")
        mat, mtl = compute_knockout(model, tokenizer, data_items, requested,
                                    min_answer_len, name, prompt_override)
        _save(mat, requested, mtl)
        return mat
    union_ns = cached_ns + missing
    union_mat = np.concatenate([cached_mat, new_mat], axis=1)
    _save(union_mat, union_ns, cmeta.get('mean_trace_len', mtl))
    return columns_for(union_mat, union_ns, requested)

# -----------------------------------------------------------------------------
# 3e. ATTENTION-KNOCKOUT GENERATION (accuracy vs memory window)
# -----------------------------------------------------------------------------
#
# The model GENERATES an answer under the same knockout: at each decoding step it
# may attend only to the prompt (always) plus the n most recent generated tokens.
# We implement this by feeding, each step, [prompt tokens] + [last n generated
# tokens] with their TRUE position ids (so the dropped middle leaves the same
# positional-encoding gap as the perplexity experiment). Answers are graded and
# accuracy is plotted against n. Per-cell caching makes runs resumable.

# internal sentinel window size meaning "full context" (window >= any generation
# length, so nothing is evicted): the reference point plotted as "full".
GEN_FULL_N = 100000

# Pretty benchmark names for figure legends only; internal keys / cache paths keep
# the short name (e.g. "GPQA") so cached results stay valid.
DISPLAY_NAME = {"GPQA": "GPQA-Diamond"}
def disp(name):
    return DISPLAY_NAME.get(name, name)

GEN_MATH_INSTR = ("Please reason step by step, and put your final answer within "
                  "\\boxed{}.")
GEN_GSM8K_INSTR = ("Please reason step by step. End your response with '#### ' "
                   "followed by the final numeric answer.")
MCQ_LETTERS = "ABCD"

def get_generation_items(name, n_samples, seed):
    """
    Sample problems for the generation experiment and build {prompt, gold, kind}.

    Sampling is a STABLE prefix of a fixed seeded permutation -- the first
    `n_samples` of `permutation(seed)` -- so a larger `n_samples` is a strict
    superset of a smaller one (the first k problems never change). This lets the
    cache be reused row-for-row when you grow `--gen-samples`. GPQA options are
    shuffled with a per-problem seed (also stable).
    """
    def stable_idx(N):
        return np.random.default_rng(seed).permutation(N)[:min(n_samples, N)]

    items = []
    if name == "GSM8K":
        ds = load_dataset("gsm8k", "main", split="test")
        for i in stable_idx(len(ds)):
            ex = ds[int(i)]
            gold = str(ex['answer']).split('####')[-1].strip().replace(',', '')
            items.append({"prompt": f"{ex['question']}\n\n{GEN_GSM8K_INSTR}",
                          "gold": gold, "kind": "gsm8k"})
    elif name == "MATH-500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for i in stable_idx(len(ds)):
            ex = ds[int(i)]
            items.append({"prompt": f"{ex['problem']}\n\n{GEN_MATH_INSTR}",
                          "gold": str(ex['answer']), "kind": "math"})
    elif name == "GPQA":
        try:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        except Exception as e:
            print(f"  could not load GPQA (gated? huggingface-cli login): {e}")
            return []
        for i in stable_idx(len(ds)):
            ex = ds[int(i)]
            correct = str(ex['Correct Answer']).strip()
            opts = [correct] + [str(ex[f'Incorrect Answer {k}']).strip() for k in (1, 2, 3)]
            order = np.random.default_rng([seed, int(i)]).permutation(4)   # per-problem stable
            shuffled = [opts[j] for j in order]
            gold_letter = MCQ_LETTERS[int(np.where(order == 0)[0][0])]
            rendered = "\n".join(f"({MCQ_LETTERS[j]}) {o}" for j, o in enumerate(shuffled))
            prompt = (f"{ex['Question']}\n\n{rendered}\n\nReason step by step, then give "
                      f"the letter of the correct option within \\boxed{{}}.")
            items.append({"prompt": prompt, "gold": gold_letter, "kind": "mcq"})
    return items

def _extract_boxed(text):
    """Contents of the last \\boxed{...} (balanced braces), or None."""
    m = text.rfind('\\boxed')
    if m == -1:
        return None
    i = m + len('\\boxed')
    while i < len(text) and text[i] != '{':
        i += 1
    depth, start = 0, i
    for j in range(i, len(text)):
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:j]
    return None

def _extract_letter(text):
    b = _extract_boxed(text)
    if b:
        mm = re.search(r'[A-D]', b.upper())
        if mm:
            return mm.group(0)
    mm = re.search(r'answer\s*(?:is|:)?\s*\(?([A-D])\)?', text, re.IGNORECASE)
    return mm.group(1).upper() if mm else None

def _norm_math(s):
    s = str(s)
    b = _extract_boxed(s)
    if b is not None:
        s = b
    s = re.sub(r'\\left|\\right|\\!|\\,|\\;|\\ ', '', s)
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    s = s.replace('$', '').replace('%', '').replace(' ', '').rstrip('.')
    return s.lower().strip('{}')

def grade_answer(kind, text, gold):
    """True iff the generated answer matches the gold answer for this benchmark."""
    if kind == 'mcq':
        letter = _extract_letter(text)
        return letter is not None and letter == gold
    if kind == 'gsm8k':
        after = text.split('####')[-1] if '####' in text else text
        nums = re.findall(r'-?\d[\d,]*\.?\d*', after)
        pred = nums[-1].replace(',', '') if nums else ''
        try:
            return abs(float(pred) - float(str(gold).replace(',', ''))) < 1e-6
        except ValueError:
            return pred == str(gold)
    # math: prefer math_verify (semantics-aware), else normalised \boxed match
    try:
        from math_verify import parse, verify
        g = parse(gold if '\\boxed' in str(gold) else f'\\boxed{{{gold}}}')
        pr = parse(text)
        if g and pr:
            return bool(verify(g, pr))
    except Exception:
        pass
    return _norm_math(text) == _norm_math(gold)

def sample_top_p(logits, temperature, top_p):
    """Sample one token id from `logits` with temperature + nucleus (top-p)."""
    logits = logits.float() / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    cum = torch.cumsum(sp, dim=-1)
    keep = (cum - sp) < top_p          # nucleus: keep until cumulative reaches p
    keep[0] = True
    sp = torch.where(keep, sp, torch.zeros_like(sp))
    sp = sp / sp.sum()
    return int(si[torch.multinomial(sp, 1)])

_LOGITS_KEEP_OK = None

def _forward_last_logits(model, **kwargs):
    """
    Forward pass returning only the LAST position's logits when the transformers
    version supports it (`logits_to_keep=1`), which avoids materialising the huge
    (batch, prompt_len, vocab) logits tensor at prefill -- the usual OOM at large
    batch. Falls back to a normal forward on older versions.
    """
    global _LOGITS_KEEP_OK
    if _LOGITS_KEEP_OK is False:
        return model(**kwargs)
    try:
        out = model(logits_to_keep=1, **kwargs)
        _LOGITS_KEEP_OK = True
        return out
    except TypeError:
        _LOGITS_KEEP_OK = False
        return model(**kwargs)

def knockout_generate_one(model, tokenizer, prompt_ids, n, args, eos_ids):
    """
    Generate an answer where each step attends to the prompt + the n most recent
    generated tokens (true positions -> RoPE gap). Returns the decoded string.
    """
    device = model.device
    P = len(prompt_ids)
    generated = []
    for _ in range(args.gen_max_new_tokens):
        w = min(n, len(generated)) if n > 0 else 0
        window = generated[len(generated) - w:] if w > 0 else []
        input_ids = torch.tensor([prompt_ids + window], device=device)
        start = P + (len(generated) - w)
        pos = list(range(P)) + [start + i for i in range(w)]
        position_ids = torch.tensor([pos], device=device)
        with torch.no_grad():
            logits = _forward_last_logits(model, input_ids=input_ids,
                                          position_ids=position_ids).logits[0, -1]
        tok = sample_top_p(logits, args.gen_temperature, args.gen_top_p)
        generated.append(tok)
        if tok in eos_ids:
            break
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated)

def _evict_window(cache, keep_prefix, keep_suffix):
    """
    Trim every layer's K/V along the sequence axis to keep the first `keep_prefix`
    (prompt sink) + last `keep_suffix` (window) entries. Handles the differing
    transformers DynamicCache layouts: newer versions expose `cache.layers[i]`
    with `.keys`/`.values` tensors; older ones use `cache.key_cache` /
    `cache.value_cache` lists.
    """
    def trim(t):
        if keep_suffix <= 0:                       # keep only the prompt sink
            return t[:, :, :keep_prefix]
        return torch.cat([t[:, :, :keep_prefix], t[:, :, -keep_suffix:]], dim=2)
    layers = getattr(cache, 'layers', None)
    if layers:
        for layer in layers:
            for kn, vn in (('keys', 'values'), ('key_cache', 'value_cache')):
                k = getattr(layer, kn, None)
                if k is not None:
                    setattr(layer, kn, trim(k))
                    setattr(layer, vn, trim(getattr(layer, vn)))
                    break
        return
    kc = getattr(cache, 'key_cache', None)
    if kc is not None:
        vc = cache.value_cache
        for l in range(len(kc)):
            kc[l] = trim(kc[l])
            vc[l] = trim(vc[l])
        return
    raise AttributeError("cannot evict the KV cache (unrecognised DynamicCache "
                         "layout for this transformers version); rerun with --gen-no-cache")

def knockout_generate_batch(model, tokenizer, prompts_ids, n, args, eos_ids, pad_id):
    """
    Batched knockout generation with a rolling-window KV cache: the prompt KV is
    computed once (prefill) and kept as an attention sink; each decode step
    appends the new token's KV and evicts the oldest so the cache holds only
    [prompt] + [last n tokens]. Explicit true position ids preserve the RoPE gap.
    Decode is O(tokens) instead of O(tokens*window), and `len(prompts_ids)`
    sequences run in parallel.

    Returns a list of decoded answer strings, one per prompt.
    """
    from transformers import DynamicCache
    device = model.device
    B = len(prompts_ids)
    P = [len(p) for p in prompts_ids]
    Pmax = max(P)

    # left-pad prompts so every real prompt ends at column Pmax-1 (predictions
    # then line up at position -1); pad columns are masked and never attended
    ids = torch.full((B, Pmax), pad_id, dtype=torch.long, device=device)
    amask = torch.zeros((B, Pmax), dtype=torch.long, device=device)
    ppos = torch.zeros((B, Pmax), dtype=torch.long, device=device)
    for b, p in enumerate(prompts_ids):
        ids[b, Pmax - P[b]:] = torch.tensor(p, device=device)
        amask[b, Pmax - P[b]:] = 1
        ppos[b, Pmax - P[b]:] = torch.arange(P[b], device=device)   # true positions 0..P_b-1

    cache = DynamicCache()
    with torch.no_grad():
        out = _forward_last_logits(model, input_ids=ids, attention_mask=amask,
                                   position_ids=ppos, past_key_values=cache, use_cache=True)
    logits = out.logits[:, -1, :]                       # (B, V) -> predicts position P_b

    gens = [[] for _ in range(B)]
    finished = [False] * B
    cur = [P[b] for b in range(B)]                      # true position of the next token
    win_len = 0                                         # window tokens currently in cache
    for _ in range(args.gen_max_new_tokens):
        toks = [sample_top_p(logits[b], args.gen_temperature, args.gen_top_p) for b in range(B)]
        for b in range(B):
            if not finished[b]:
                gens[b].append(toks[b])
                if toks[b] in eos_ids:
                    finished[b] = True
        if all(finished):
            break
        new_ids = torch.tensor([[t] for t in toks], dtype=torch.long, device=device)
        new_pos = torch.tensor([[c] for c in cur], dtype=torch.long, device=device)
        full_mask = torch.cat([amask, torch.ones((B, win_len + 1), dtype=torch.long, device=device)],
                              dim=1)                     # prompt (pad-masked) + window + new
        with torch.no_grad():
            out = _forward_last_logits(model, input_ids=new_ids, attention_mask=full_mask,
                                       position_ids=new_pos, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1, :]
        cur = [c + 1 for c in cur]
        win_len += 1
        # keep n-1 window tokens in the cache: the next query attends to those
        # plus itself = the n most recent tokens (matching knockout_generate_one
        # and the perplexity mask's (q-n, q] window, not n+1).
        keep = max(n - 1, 0)
        if win_len > keep:                              # evict oldest -> [prompt] + [last n-1]
            _evict_window(cache, Pmax, keep)
            win_len = keep
    return ([tokenizer.decode(g, skip_special_tokens=True) for g in gens],
            [len(g) for g in gens])

# ---- resumable per-(problem, n) correctness cache ----

def gen_path(name):
    return os.path.join(KNOCKOUT_GEN_DIR, MODEL_SLUG, f"{name}.npz")

# Cache identity does NOT include gen_samples / n_problems: sampling is a stable
# permutation prefix, so a larger run reuses the smaller run's rows (extend the
# matrix) and only generates the new problems.

def _gen_meta(args, kind, n_problems):
    return dict(model=MODEL_NAME, kind=kind, n_problems=int(n_problems),
                gen_samples=args.gen_samples, gen_seed=args.gen_seed,
                gen_max_new_tokens=args.gen_max_new_tokens,
                gen_temperature=args.gen_temperature, gen_top_p=args.gen_top_p)

def load_gen(name):
    d = np.load(gen_path(name), allow_pickle=False)
    meta = json.loads(str(d['meta'])) if 'meta' in d.files else {}
    correct = d['correct']
    lengths = (d['lengths'] if 'lengths' in d.files
               else np.full(correct.shape, -1, np.int32))   # legacy caches: unknown lengths
    return correct, lengths, [int(x) for x in d['n_values']], meta

def save_gen(name, correct, lengths, n_values, meta):
    os.makedirs(os.path.dirname(gen_path(name)), exist_ok=True)
    np.savez_compressed(gen_path(name), correct=correct.astype(np.int8),
                        lengths=lengths.astype(np.int32),
                        n_values=np.asarray(n_values, dtype=np.int64), meta=json.dumps(meta))

def gen_compatible(meta, args):
    """Whether a cached generation matrix used the same model / decoding knobs
    (independent of gen_samples -- rows are reused by stable sampling)."""
    want = {'model': MODEL_NAME, 'gen_seed': args.gen_seed,
            'gen_max_new_tokens': args.gen_max_new_tokens,
            'gen_temperature': args.gen_temperature, 'gen_top_p': args.gen_top_p}
    return all(meta.get(k) == v for k, v in want.items())

def aggregate_gen(correct, n_values, requested):
    """Accuracy (mean) and SEM per requested n, over fully-computed cells."""
    have = list(n_values)
    cols = [have.index(int(n)) for n in requested]
    sub = correct[:, cols]
    acc, sem = [], []
    for c in range(sub.shape[1]):
        vals = sub[:, c]
        vals = vals[vals >= 0]                 # only computed cells
        if vals.size == 0:
            acc.append(np.nan); sem.append(0.0); continue
        p = float(vals.mean())
        acc.append(p)
        sem.append(float(np.sqrt(max(p * (1 - p), 0) / vals.size)))
    return np.array(acc), np.array(sem)

# -----------------------------------------------------------------------------
# 4. PLOTTING
# -----------------------------------------------------------------------------

def plot_entropy_results(results, max_analysis_len):
    print("Generating Entropy Plot...")
    
    # Update global font sizes for consistent look
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 7
    })

    fig, ax = plt.subplots(figsize=(3,2.5))
    
    # One line per teacher-forced dataset (the math reasoning sets plus the
    # non-math GPQA science set in warm/dark colours, WikiText baseline light blue)
    styles = {
        'GSM8K':    '#003366',
        'MATH-500': '#2ca02c',
        'GPQA':     '#d62728',
        'WikiText': '#5dade2',
    }

    for name, color in styles.items():
        if name in results and isinstance(results[name], dict):
            steps = np.asarray(results[name]["steps"], dtype=float)
            mean = np.asarray(results[name]["entropy"], dtype=float)
            ax.plot(steps, mean, label=disp(name), color=color,
                    linestyle='-', linewidth=2.0, alpha=1.0)
            # shaded +/- 1 SEM band, when standard errors are available
            if "sem" in results[name]:
                sem = np.asarray(results[name]["sem"], dtype=float)
                ax.fill_between(steps, mean - sem, mean + sem,
                                color=color, alpha=0.2, linewidth=0)

    ax.set_xlabel("Text Length (Tokens)")
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Entropy (nats)")
    ax.set_xlim(right=max_analysis_len)
    
    ax.legend(loc='center right', frameon=False, handlelength=1.5, bbox_to_anchor=(1.0, .8))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    output_path = os.path.join(FIGURES_DIR, "entropy_over_time")
    plt.savefig(output_path+".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_path+".png", dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()

def plot_knockout_results(results, n_values):
    print("Generating Knockout Perplexity Plot...")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 7,
    })

    fig, ax = plt.subplots(figsize=(3, 2.5))
    styles = {
        'GSM8K':    '#003366',
        'MATH-500': '#2ca02c',
        'GPQA':     '#d62728',
        'WikiText': '#5dade2',
    }
    # results[name] = {'mean','sem'} over the n sweep, plus optional 'full_mean'/
    # 'full_sem' (cached separately). The sweep is a line + SEM band; "full" is a
    # separate point (with a SEM error bar) placed to the right of the sweep.
    x = np.asarray(n_values, dtype=float)
    # place "full" a couple of sweep-steps past the last n so its tick label does
    # not collide with the last numeric tick
    step = (x[-1] - x[-2]) if len(x) > 1 else 10.0
    gap = max(2.0 * step, 20.0)
    x_full = x.max() + gap
    any_full = any(results.get(nm, {}).get('full_mean') is not None for nm in styles)
    for name, color in styles.items():
        if name in results:
            mean = np.asarray(results[name]["mean"], dtype=float)
            sem = np.asarray(results[name]["sem"], dtype=float)
            ax.plot(x, mean, label=disp(name), color=color, linestyle='-', linewidth=1.8, zorder=3)
            # +/- 1 SEM shaded band around the sweep line
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.3,
                            linewidth=0, zorder=2)
            # full-context reference point (entire context), with its SEM error bar
            fm = results[name].get('full_mean')
            if fm is not None:
                ax.errorbar([x_full], [fm], yerr=[results[name].get('full_sem', 0.0)],
                            fmt='o', color=color, ms=5, capsize=0, capthick=.7,
                            elinewidth=.7, zorder=5)
    if any_full:
        # subtle divider between the sweep and the "full" reference point
        ax.axvline(x.max() + gap / 2.0, color='0.85', lw=0.8, ls=(0, (2, 2)), zorder=0)

    ax.set_xlabel(r"Memory Size (Tokens)")
    ax.set_ylabel("Perplexity")
    ax.set_ylim(bottom=0.0, top=30.0)
    ax.set_xlim(left=min(n_values), right=(x_full + step) if any_full else x.max())
    # numeric ticks for the sweep (every 20), plus a labelled "full" tick if present
    base_ticks = [n for n in n_values if n % 20 == 0 and n >= 10]
    if any_full:
        ax.set_xticks(base_ticks + [x_full])
        ax.set_xticklabels([str(int(t)) for t in base_ticks] + ["full"])
    else:
        ax.set_xticks(base_ticks)
        ax.set_xticklabels([str(int(t)) for t in base_ticks])
    ax.legend(loc='upper right', frameon=False, handlelength=1.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    output_path = os.path.join(FIGURES_DIR, "knockout_perplexity")
    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_path + ".png", dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()

def plot_knockout_generation_results(results, n_values):
    """Accuracy vs memory window n, one line per benchmark, with +/-1 SEM shading."""
    print("Generating Knockout-Generation Accuracy Plot...")
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 7})
    fig, ax = plt.subplots(figsize=(3, 2.5))
    styles = {'GSM8K': '#003366', 'MATH-500': '#2ca02c', 'GPQA': '#d62728'}
    x = np.asarray(n_values, dtype=float)
    step = (x[-1] - x[-2]) if len(x) > 1 else 10.0
    gap = max(2.0 * step, 20.0)
    x_full = x.max() + gap
    any_full = any(np.isfinite(results.get(nm, {}).get('full_acc', np.nan)) for nm in styles)
    for name, color in styles.items():
        if name in results:
            acc = np.asarray(results[name]["acc"], dtype=float)
            sem = np.asarray(results[name]["sem"], dtype=float)
            ax.plot(x, acc, label=disp(name), color=color, linestyle='-', linewidth=1.8, zorder=3)
            ax.fill_between(x, acc - sem, acc + sem, color=color, alpha=0.3,
                            linewidth=0, zorder=2)
            fa = results[name].get('full_acc', np.nan)
            if np.isfinite(fa):
                ax.errorbar([x_full], [fa], yerr=[results[name].get('full_sem', 0.0)],
                            fmt='o', color=color, ms=5, capsize=0, elinewidth=.9, zorder=5)
    if any_full:
        ax.axvline(x.max() + gap / 2.0, color='0.85', lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.set_xlabel(r"Memory Size (Tokens)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_xlim(right=(x_full + step) if any_full else x.max())
    base_ticks = [int(t) for t in x if t % 40 == 0]
    if any_full:
        ax.set_xticks(base_ticks + [x_full]); ax.set_xticklabels([str(t) for t in base_ticks] + ["full"])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    output_path = os.path.join(FIGURES_DIR, "knockout_generation_accuracy")
    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_path + ".png", dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()

def plot_knockout_generation_lengths(table, n_values):
    """Mean reasoning length vs memory window n, split into correct vs incorrect
    answers (one figure per benchmark), with +/-1 SEM shading and a 'full'-context
    reference point at the right end. Reads the cached per-n length stats in
    `table`, so it regenerates from cache (--plot-only) without any rerun."""
    print("Generating Knockout-Generation Reasoning-Length Plots...")
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 7})
    x = np.asarray(n_values, dtype=float)
    step = (x[-1] - x[-2]) if len(x) > 1 else 10.0
    gap = max(2.0 * step, 20.0)
    x_full = x.max() + gap
    series = [('correct', 'Correct', '#2ca02c'), ('incorrect', 'Incorrect', '#d62728')]
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for name in table:
        by_n = table[name]['by_n']
        fig, ax = plt.subplots(figsize=(3, 2.5))
        any_full = False
        for key, lbl, color in series:
            mean = np.array([by_n[n][f'mean_{key}'] for n in n_values], dtype=float)
            sem = np.array([by_n[n][f'sem_{key}'] for n in n_values], dtype=float)
            ax.plot(x, mean, label=lbl, color=color, linestyle='-', linewidth=1.8, zorder=3)
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.3,
                            linewidth=0, zorder=2)
            fm, fs = by_n[GEN_FULL_N][f'mean_{key}'], by_n[GEN_FULL_N][f'sem_{key}']
            if np.isfinite(fm):
                ax.errorbar([x_full], [fm], yerr=[fs if np.isfinite(fs) else 0.0],
                            fmt='o', color=color, ms=5, capsize=0, elinewidth=.9, zorder=5)
                any_full = True
        if any_full:
            ax.axvline(x.max() + gap / 2.0, color='0.85', lw=0.8, ls=(0, (2, 2)), zorder=0)
        ax.set_xlabel(r"Memory Size (Tokens)")
        ax.set_ylabel("Mean Reasoning Length")
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(right=(x_full + step) if any_full else x.max())
        base_ticks = [int(t) for t in x if t % 40 == 0]
        if any_full:
            ax.set_xticks(base_ticks + [x_full])
            ax.set_xticklabels([str(t) for t in base_ticks] + ["full"])
        ax.legend(title=disp(name), loc='center left', frameon=False, handlelength=1.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        safe = re.sub(r'[^A-Za-z0-9._-]', '_', name)
        output_path = os.path.join(FIGURES_DIR, f"knockout_gen_length_{safe}")
        plt.savefig(output_path + ".pdf", dpi=300, bbox_inches="tight")
        plt.savefig(output_path + ".png", dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close()

def plot_knockout_generation_lengths_combined(table, n_values):
    """Mean reasoning length of CORRECT answers vs memory window n, ONE curve per
    benchmark on a single axis, with +/-1 SEM shading and a 'full'-context point
    at the right end. Restricting to correct answers avoids the non-terminating
    small-n loops (which run to the cap) inflating the means. Built from the cached
    per-n length stats in `table`, so it regenerates from cache (--plot-only)."""
    print("Generating Combined Knockout-Generation Reasoning-Length Plot...")
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 7})
    fig, ax = plt.subplots(figsize=(3, 2.5))
    styles = {'GSM8K': '#003366', 'MATH-500': '#2ca02c', 'GPQA': '#d62728'}
    x = np.asarray(n_values, dtype=float)
    step = (x[-1] - x[-2]) if len(x) > 1 else 10.0
    gap = max(2.0 * step, 20.0)
    x_full = x.max() + gap
    any_full = False
    for name, color in styles.items():
        if name not in table:
            continue
        by_n = table[name]['by_n']
        mean = np.array([by_n[n]['mean_correct'] for n in n_values], dtype=float)
        sem = np.array([by_n[n]['sem_correct'] for n in n_values], dtype=float)
        ax.plot(x, mean, label=disp(name), color=color, linestyle='-', linewidth=1.8, zorder=3)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.25,
                        linewidth=0, zorder=2)
        fm, fs = by_n[GEN_FULL_N]['mean_correct'], by_n[GEN_FULL_N]['sem_correct']
        if np.isfinite(fm):
            ax.errorbar([x_full], [fm], yerr=[fs if np.isfinite(fs) else 0.0],
                        fmt='o', color=color, ms=5, capsize=0, elinewidth=.9, zorder=5)
            any_full = True
    if any_full:
        ax.axvline(x.max() + gap / 2.0, color='0.85', lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.set_xlabel(r"Memory Size (Tokens)")
    ax.set_ylabel("Length (Tokens)")
    ax.set_ylim(bottom=0.0)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # ->  x10^4 offset, shorter labels
    ax.yaxis.get_offset_text().set_size(10)
    ax.set_xlim(right=(x_full + step) if any_full else x.max())
    base_ticks = list(range(50, int(x.max()) + 1, 50))   # sparse ticks so labels + "full" don't collide
    if any_full:
        ax.set_xticks(base_ticks + [x_full])
        ax.set_xticklabels([str(t) for t in base_ticks] + ["full"])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    output_path = os.path.join(FIGURES_DIR, "knockout_gen_length_combined")
    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_path + ".png", dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def load_datasets(tokenizer, num_samples, wt_min_answer_tokens):
    """
    Load all datasets (GSM8K, MATH-500, GPQA, WikiText) as {prompt, full} items
    and return (raw, order, gsm8k_avg_prompt_len). The WikiText baseline is
    aligned to the average GSM8K prompt length so its "answer" starts at a
    comparable absolute position.

    `wt_min_answer_tokens` sets how long WikiText passages must be (so they have
    enough tokens after the simulated prompt for the current experiment).
    """
    print("\n--- Loading datasets ---")
    raw = {
        "GSM8K":    get_gsm8k_data(num_samples),
        "MATH-500": get_math500_data(num_samples),
        "GPQA":     get_gpqa_data(num_samples),
    }
    p_lengths = [len(tokenizer.encode(it['prompt'], add_special_tokens=True))
                 for it in raw["GSM8K"]]
    gsm8k_avg_prompt_len = float(np.mean(p_lengths)) if p_lengths else 0.0
    print(f"Average GSM8K prompt length: {gsm8k_avg_prompt_len:.2f}")

    wt_target_tokens = int(gsm8k_avg_prompt_len) + wt_min_answer_tokens
    raw["WikiText"] = get_wikitext_data(NUM_WIKI_SAMPLES, min_char_len=wt_target_tokens * 3)
    if num_samples is not None:
        raw["WikiText"] = raw["WikiText"][:num_samples]

    return raw, ["GSM8K", "MATH-500", "GPQA", "WikiText"], gsm8k_avg_prompt_len

def run_entropy(tokenizer):
    """Entropy vs answer-token-position experiment (cached full traces)."""
    cache_min_len = ARGS.cache_min_len
    num_samples = ARGS.num_samples
    raw, order, avg = load_datasets(tokenizer, num_samples, cache_min_len)
    # WikiText analysis starts at logit index avg-1 (predicts token at avg)
    start_overrides = {"WikiText": max(0, int(avg) - 1)}

    need = [n for n in order if ARGS.force or not traces_fresh(n, cache_min_len, num_samples)]
    if ARGS.plot_only and need:
        print(f"[plot-only] no fresh trace cache for {need}; they will be skipped")
        need = []
    model = load_model() if (need and not ARGS.plot_only) else None
    if model is None and not ARGS.plot_only:
        print("All trace caches present and fresh -> skipping model load")

    traces_by_name = {}
    for name in order:
        if name in need:
            traces = compute_traces(model, tokenizer, raw[name], name, cache_min_len,
                                    fixed_start_idx=start_overrides.get(name))
            save_traces(name, traces, dict(model=MODEL_NAME, dataset=name,
                                           cache_min_len=cache_min_len,
                                           num_samples=num_samples,
                                           n_traces=len(traces),
                                           avg_prompt_len=avg))
            traces_by_name[name] = traces
        elif os.path.exists(traces_path(name)):
            traces_by_name[name], _ = load_traces(name)

    results = {}
    for name in order:
        traces = traces_by_name.get(name, [])
        if not traces:
            continue
        steps, mean, sem = aggregate_tokens(traces, ARGS.max_analysis_len)
        if not steps:
            print(f"  {name}: no sequences >= {ARGS.max_analysis_len} answer tokens; skipped")
            continue
        n_used = sum(1 for t in traces if len(t) >= ARGS.max_analysis_len)
        print(f"  {name}: {n_used} / {len(traces)} cached sequences used "
              f"(answer len >= {ARGS.max_analysis_len})")
        results[name] = {"steps": steps, "entropy": mean, "sem": sem}

    plot_entropy_results(results, ARGS.max_analysis_len)

def run_knockout(tokenizer):
    """Attention-knockout perplexity-vs-window-size experiment."""
    num_samples = ARGS.num_samples
    min_answer_len = ARGS.min_answer_len
    n_values = parse_n_values(ARGS.knockout_ns)
    print(f"Knockout window sizes n: {n_values}")

    raw, order, avg = load_datasets(tokenizer, num_samples, min_answer_len)
    # WikiText has no real prompt; treat its first ~avg tokens as the visible
    # "prompt" so it is comparable to the reasoning sets.
    prompt_overrides = {"WikiText": int(avg)}

    # Sweep and full-context caches are independent. The sweep cache stores one
    # perplexity column per n, so only the window sizes NOT already cached are
    # (re)computed -- e.g. adding n=5 to a cached 10..100 sweep computes only n=5.
    need_sweep = [n for n in order
                  if ARGS.force or not knockout_covers(n, min_answer_len, num_samples, n_values)]
    need_full = [n for n in order
                 if ARGS.force or not knockout_full_fresh(n, min_answer_len, num_samples)]
    if ARGS.plot_only and (need_sweep or need_full):
        print(f"[plot-only] missing/stale caches (sweep={need_sweep}, "
              f"full={need_full}); those points are skipped")
    # SDPA (the default) applies our custom 4D additive mask via attn_mask -- no
    # query is ever fully masked (every one attends to the prompt), so the
    # fully-masked-row NaN guard never fires; eager is not needed.
    model = (load_model() if (need_sweep or need_full) and not ARGS.plot_only else None)
    if model is None and not ARGS.plot_only:
        print("All knockout caches present and cover the requested n -> skipping model load")

    sweep_by_name, full_by_name = {}, {}
    for name in order:
        # ---- window sweep (per-n column reuse) ----
        mat = resolve_sweep(name, n_values, min_answer_len, num_samples, avg,
                            model, tokenizer, raw[name], prompt_overrides.get(name),
                            ARGS.plot_only, ARGS.force)
        if mat is not None:
            sweep_by_name[name] = mat

        # ---- full-context reference (separate cache) ----
        if name in need_full and not ARGS.plot_only:
            vec, mtl = compute_knockout_full(model, tokenizer, raw[name], min_answer_len,
                                             name, prompt_len_override=prompt_overrides.get(name))
            save_knockout_full(name, vec, dict(model=MODEL_NAME, dataset=name,
                                               min_answer_len=min_answer_len,
                                               num_samples=num_samples,
                                               n_sequences=int(vec.shape[0]),
                                               mean_trace_len=mtl))
            full_by_name[name] = vec
        elif os.path.exists(knockout_full_path(name)):
            vec, _ = load_knockout_full(name)
            full_by_name[name] = vec

    results = {}
    for name in order:
        mat = sweep_by_name.get(name)
        if mat is None or mat.shape[0] == 0:
            continue
        mean, sem = aggregate_knockout(mat)
        entry = {"mean": mean, "sem": sem}
        vec = full_by_name.get(name)
        if vec is not None and vec.shape[0] > 0:
            entry["full_mean"] = float(vec.mean())
            entry["full_sem"] = (float(vec.std(ddof=1) / np.sqrt(vec.shape[0]))
                                 if vec.shape[0] > 1 else 0.0)
        results[name] = entry

    # summary: sequence counts and average trace lengths. Lengths are derived from
    # the data + tokenizer (no model), so they print even when the perplexities
    # were loaded from an older cache that did not store them.
    print("\nAverage trace lengths (answer tokens):")
    for name in order:
        if name not in results:
            continue
        n_kept, mean_len = count_and_mean_length(
            tokenizer, raw[name], min_answer_len, prompt_overrides.get(name))
        print(f"  {name:10s} {n_kept:5d} sequences  |  mean trace length = {mean_len:.1f} tokens")

    plot_knockout_results(results, n_values)

def run_knockout_generation(tokenizer):
    """Generate answers under the knockout and plot accuracy vs memory window n."""
    n_values = parse_n_values(ARGS.knockout_gen_ns)
    all_cols = n_values + [GEN_FULL_N]        # sweep + the full-context reference point
    order = ["GSM8K", "MATH-500", "GPQA"]
    print(f"Knockout-generation window sizes n: {n_values} (+ full)")

    # sampled problems per benchmark (stable permutation prefix -> rows are reusable)
    items_by_name = {name: get_generation_items(name, ARGS.gen_samples, ARGS.gen_seed)
                     for name in order}

    # gate model load: need work if any requested cell (rows [0,nprob)) is uncomputed
    def _needs(name):
        items = items_by_name[name]
        if not items:
            return False
        nprob = len(items)
        if ARGS.force or not os.path.exists(gen_path(name)):
            return True
        try:
            corr, _lens, cns, meta = load_gen(name)
        except Exception:
            return True
        if not gen_compatible(meta, ARGS) or not all(n in set(cns) for n in all_cols):
            return True
        if corr.shape[0] < nprob:
            return True                              # cache has fewer problems than requested
        cols = [list(cns).index(int(n)) for n in all_cols]
        return bool((corr[:nprob, cols] < 0).any())

    any_need = any(_needs(n) for n in order)
    model = (load_model() if any_need and not ARGS.plot_only else None)
    eos_ids = {i for i in [getattr(tokenizer, 'eos_token_id', None),
                           tokenizer.convert_tokens_to_ids('<|im_end|>')
                           if tokenizer.convert_tokens_to_ids('<|im_end|>') is not None else None]
               if i is not None and i >= 0}

    results, table = {}, {}
    for name in order:
        items = items_by_name[name]
        if not items:
            continue
        nprob, kind = len(items), items[0]['kind']

        # load / initialise the correctness + length matrices, reusing cached rows
        # (stable sampling: cached row i is the same problem regardless of --gen-samples)
        correct = lengths = None
        cns = []
        if os.path.exists(gen_path(name)) and not ARGS.force:
            try:
                c0, l0, cns0, meta0 = load_gen(name)
                if gen_compatible(meta0, ARGS):
                    cns = list(cns0)
                    rows = max(c0.shape[0], nprob)         # keep all cached rows, extend to nprob
                    correct = np.full((rows, c0.shape[1]), -1, np.int8); correct[:c0.shape[0]] = c0
                    lengths = np.full((rows, c0.shape[1]), -1, np.int32); lengths[:l0.shape[0]] = l0
            except Exception:
                correct = None
        if correct is None:
            correct = np.full((nprob, 0), -1, np.int8)
            lengths = np.full((nprob, 0), -1, np.int32)
            cns = []
        for n in all_cols:                                  # ensure a column per requested n + full
            if n not in cns:
                correct = np.concatenate([correct, np.full((correct.shape[0], 1), -1, np.int8)], axis=1)
                lengths = np.concatenate([lengths, np.full((lengths.shape[0], 1), -1, np.int32)], axis=1)
                cns.append(int(n))

        # fill missing cells for the first nprob rows (resumable); checkpoint per batch
        if model is not None:
            pad_id = getattr(tokenizer, 'pad_token_id', None)
            if pad_id is None:
                pad_id = getattr(tokenizer, 'eos_token_id', 0) or 0
            prompt_ids_all = []                             # tokenize each prompt once (n-independent)
            for it in items:
                s = tokenizer.apply_chat_template(
                    [{"role": "user", "content": it['prompt']}],
                    add_generation_prompt=True, tokenize=False, enable_thinking=True)
                prompt_ids_all.append(tokenizer.encode(s, add_special_tokens=False))

            bs = max(1, ARGS.gen_batch_size)
            # the full column's KV cache grows with generation length (no eviction),
            # so it can need a smaller batch than the window-bounded knockout columns.
            full_bs = max(1, ARGS.gen_full_batch_size or ARGS.gen_batch_size)
            for n in all_cols:
                col = cns.index(int(n))
                todo = [i for i in range(nprob) if correct[i, col] < 0]
                if not todo:
                    continue
                todo.sort(key=lambda i: len(prompt_ids_all[i]))   # length-homogeneous batches
                bs_n = full_bs if n == GEN_FULL_N else bs
                label = "full" if n == GEN_FULL_N else f"n={n}"
                for s0 in tqdm(range(0, len(todo), bs_n), desc=f"{name} {label}", unit="batch"):
                    idxs = todo[s0:s0 + bs_n]
                    prompts = [prompt_ids_all[i] for i in idxs]
                    if ARGS.gen_no_cache:
                        pairs = [knockout_generate_one(model, tokenizer, p, n, ARGS, eos_ids)
                                 for p in prompts]
                        texts = [t for t, _ in pairs]
                        lens = [l for _, l in pairs]
                    else:
                        texts, lens = knockout_generate_batch(model, tokenizer, prompts, n, ARGS,
                                                              eos_ids, pad_id)
                    for i, text, ln in zip(idxs, texts, lens):
                        correct[i, col] = 1 if grade_answer(kind, text, items[i]['gold']) else 0
                        lengths[i, col] = int(ln)
                    save_gen(name, correct, lengths, cns, _gen_meta(ARGS, kind, correct.shape[0]))

        # accuracy uses the first nprob rows (this run's problem set)
        csub = correct[:nprob]
        acc, sem = aggregate_gen(csub, cns, n_values)
        full_acc, full_sem = aggregate_gen(csub, cns, [GEN_FULL_N])
        results[name] = {"acc": acc, "sem": sem,
                         "full_acc": float(full_acc[0]), "full_sem": float(full_sem[0])}

        # generation-length stats over this benchmark's rows, split by correctness.
        # Derived entirely from the cached `lengths` + `correct` matrices, so these
        # tables regenerate from cache (e.g. with --plot-only) without any rerun.
        def _len_stats(lv, cv):
            """lv/cv: 1D length + correctness arrays for a set of cells (-1 = todo).
            Returns count, mean and SEM overall and split by correctness."""
            valid = lv >= 0
            def stat(mask):
                s = lv[mask].astype(float)
                n = int(s.size)
                mean = float(s.mean()) if n else float('nan')
                sem = (float(s.std(ddof=1) / np.sqrt(n)) if n > 1
                       else (0.0 if n == 1 else float('nan')))
                return n, mean, sem
            n_all, m_all, se_all = stat(valid)
            n_cor, m_cor, se_cor = stat(valid & (cv == 1))
            n_inc, m_inc, se_inc = stat(valid & (cv == 0))
            return {'n': n_all, 'mean': m_all, 'sem': se_all,
                    'hit': int((lv[valid] >= ARGS.gen_max_new_tokens).sum()),
                    'n_correct': n_cor, 'mean_correct': m_cor, 'sem_correct': se_cor,
                    'n_incorrect': n_inc, 'mean_incorrect': m_inc, 'sem_incorrect': se_inc}
        by_n = {n: _len_stats(lengths[:nprob, cns.index(int(n))],
                              correct[:nprob, cns.index(int(n))]) for n in all_cols}
        gcols = [cns.index(int(n)) for n in all_cols]
        table[name] = {'overall': _len_stats(lengths[:nprob][:, gcols].ravel(),
                                             correct[:nprob][:, gcols].ravel()),
                       'by_n': by_n}

    # ---- summary tables ----
    col_labels = ["full" if n == GEN_FULL_N else str(n) for n in all_cols]
    _flen = lambda v: f"{v:.0f}" if v == v else "n/a"          # NaN-safe length
    _fint = lambda v: str(v)

    print(f"\nGeneration summary (max-new-tokens = {ARGS.gen_max_new_tokens}), "
          f"all n + full:")
    hdr = f"{'benchmark':<10}{'#sequences':>12}{'mean gen len':>14}{'hit cap':>9}"
    print(hdr); print('-' * len(hdr))
    for name in order:
        if name in table:
            o = table[name]['overall']
            print(f"{name:<10}{o['n']:>12d}{_flen(o['mean']):>14}{o['hit']:>9d}")

    def _print_by_n(title, key, fmt):
        w = max(8, max(len(l) for l in col_labels) + 2)
        head = f"{'benchmark':<10}" + "".join(f"{l:>{w}}" for l in col_labels)
        print(f"\n{title} by n:")
        print(head); print('-' * len(head))
        for name in order:
            if name not in table:
                continue
            cells = "".join(f"{fmt(table[name]['by_n'][n][key]):>{w}}" for n in all_cols)
            print(f"{name:<10}{cells}")

    _print_by_n("Mean generated length (all answers)", 'mean', _flen)
    _print_by_n("Mean generated length (CORRECT answers)", 'mean_correct', _flen)
    _print_by_n("Mean generated length (INCORRECT answers)", 'mean_incorrect', _flen)
    _print_by_n("# correct answers", 'n_correct', _fint)
    _print_by_n("# incorrect answers", 'n_incorrect', _fint)
    _print_by_n("Sequences hitting cap", 'hit', _fint)

    # persist the per-(benchmark, n) length stats so the tables are reusable as data
    # without rerunning (or even reloading the .npz matrices).
    summ_path = os.path.join(KNOCKOUT_GEN_DIR, MODEL_SLUG, "length_summary.csv")
    os.makedirs(os.path.dirname(summ_path), exist_ok=True)
    def _csv(x):
        return "" if (isinstance(x, float) and x != x) else \
               (f"{x:.2f}" if isinstance(x, float) else str(x))
    with open(summ_path, "w") as f:
        f.write("benchmark,n,n_all,mean_len_all,sem_all,n_correct,mean_len_correct,"
                "sem_correct,n_incorrect,mean_len_incorrect,sem_incorrect,hit_cap\n")
        for name in order:
            if name not in table:
                continue
            for n in all_cols:
                s = table[name]['by_n'][n]
                lbl = "full" if n == GEN_FULL_N else str(n)
                f.write(f"{name},{lbl},{s['n']},{_csv(s['mean'])},{_csv(s['sem'])},"
                        f"{s['n_correct']},{_csv(s['mean_correct'])},{_csv(s['sem_correct'])},"
                        f"{s['n_incorrect']},{_csv(s['mean_incorrect'])},"
                        f"{_csv(s['sem_incorrect'])},{s['hit']}\n")
    print(f"\nPer-n length summary (incl. correct/incorrect split) written to {summ_path}")

    plot_knockout_generation_results(results, n_values)
    plot_knockout_generation_lengths(table, n_values)
    plot_knockout_generation_lengths_combined(table, n_values)

EXPERIMENTS = {
    "knockout": run_knockout,
    "entropy": run_entropy,
    "knockout-generation": run_knockout_generation,
}

def main():
    tokenizer = load_tokenizer()
    to_run = list(EXPERIMENTS) if ARGS.experiment == "all" else [ARGS.experiment]
    for i, name in enumerate(to_run):
        if len(to_run) > 1:
            print(f"\n{'='*60}\n=== experiment: {name}\n{'='*60}")
        EXPERIMENTS[name](tokenizer)
        # each experiment loads its own model; release VRAM before the next so
        # running "all" doesn't stack 32B model copies and OOM.
        if len(to_run) > 1 and i < len(to_run) - 1 and DEVICE == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()