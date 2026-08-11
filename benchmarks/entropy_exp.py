import os
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
    p = argparse.ArgumentParser(description="Teacher-forced entropy vs token position "
                                            "over reasoning datasets and a WikiText baseline.")
    p.add_argument('--devices', nargs='+', default=None, metavar='cuda:N',
                   help="GPUs to use, e.g. --devices cuda:0 cuda:1 cuda:2 cuda:3 "
                        "(bare indices like 0 1 also accepted). Sets "
                        "CUDA_VISIBLE_DEVICES; the model is sharded across them. "
                        "Default: all visible GPUs.")
    # ---- plot-time knob (re-aggregate cached traces, no recompute) ----
    p.add_argument('--max-analysis-len', type=int, default=100,
                   help="analyse the first N answer tokens; only sequences with "
                        ">=N answer tokens contribute (constant N, no survivorship).")
    # ---- compute-time knobs (change these -> traces are recomputed) ----
    p.add_argument('--cache-min-len', type=int, default=32,
                   help="only compute/cache sequences with >= this many answer "
                        "tokens. Bounds forward-pass cost and sets how low "
                        "--max-analysis-len can go without recomputing traces.")
    p.add_argument('--num-samples', type=int, default=None,
                   help="cap problems per dataset before computing (default: all; "
                        "use to bound cost, especially for GSM8K).")
    p.add_argument('--plot-only', action='store_true',
                   help="only re-aggregate/plot from cached traces (no model load).")
    p.add_argument('--force', action='store_true',
                   help="recompute cached traces even if present and fresh.")
    return p.parse_args()

ARGS = parse_args()
if ARGS.devices:
    _indices = [str(d)[len('cuda:'):] if str(d).lower().startswith('cuda:') else str(d)
                for d in ARGS.devices]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(_indices)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-32B"
NUM_WIKI_SAMPLES = 5000         # WikiText passages to consider before filtering
MAX_SEQ_LEN = 1024              # forward-pass / cached-trace length cap
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Raw per-token entropy traces are cached here (one .npz per dataset). Because a
# forward pass yields the whole sequence anyway, we cache the FULL answer-region
# entropy per sequence (not a fixed window), so --max-analysis-len (and a future
# percent-of-solution view) can be recomputed from cache without re-running the
# model.
TRACES_DIR = "entropy_traces"

print(f"Running on device: {DEVICE}"
      + (f" | CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}" if ARGS.devices else ""))

# -----------------------------------------------------------------------------
# 1. LOAD MODEL AND TOKENIZER
# -----------------------------------------------------------------------------

def load_tokenizer():
    print(f"Loading tokenizer: {MODEL_NAME}...")
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def load_model():
    print(f"Loading model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None
    )
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
    return os.path.join(TRACES_DIR, f"{name}.npz")

def save_traces(name, traces, meta):
    """Persist ragged per-sequence traces flat (values + lengths) with a meta dict."""
    os.makedirs(TRACES_DIR, exist_ok=True)
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
            ax.plot(steps, mean, label=name, color=color,
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
    
    ax.legend(loc='center right', frameon=False, handlelength=1.5, bbox_to_anchor=(1.0, .7))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = "entropy_over_time"
    plt.savefig(output_path+".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_path+".png", dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    tokenizer = load_tokenizer()
    cache_min_len = ARGS.cache_min_len
    num_samples = ARGS.num_samples

    # --- 1. Load the raw datasets (text only; no model needed yet) ---
    print("\n--- Loading datasets ---")
    raw = {
        "GSM8K":    get_gsm8k_data(num_samples),
        "MATH-500": get_math500_data(num_samples),
        "GPQA":     get_gpqa_data(num_samples),
    }

    # Average GSM8K prompt length -> used to align the WikiText baseline so its
    # "answer" starts at the same absolute position (tokenizer only, no model).
    p_lengths = [len(tokenizer.encode(it['prompt'], add_special_tokens=True))
                 for it in raw["GSM8K"]]
    gsm8k_avg_prompt_len = float(np.mean(p_lengths)) if p_lengths else 0.0
    print(f"Average GSM8K prompt length: {gsm8k_avg_prompt_len:.2f}")
    wt_start_idx = max(0, int(gsm8k_avg_prompt_len) - 1)

    # WikiText: keep passages long enough to have >= cache_min_len answer tokens
    # after the simulated prompt (~3 chars/token is a cheap pre-filter).
    wt_target_tokens = int(gsm8k_avg_prompt_len) + cache_min_len
    raw["WikiText"] = get_wikitext_data(NUM_WIKI_SAMPLES, min_char_len=wt_target_tokens * 3)
    if num_samples is not None:
        raw["WikiText"] = raw["WikiText"][:num_samples]

    order = ["GSM8K", "MATH-500", "GPQA", "WikiText"]
    start_overrides = {"WikiText": wt_start_idx}

    # --- 2. Compute or load cached traces (the expensive step is cached) ---
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
                                           avg_prompt_len=gsm8k_avg_prompt_len))
            traces_by_name[name] = traces
        elif os.path.exists(traces_path(name)):
            traces_by_name[name], _ = load_traces(name)

    # --- 3. Aggregate (cheap; no model) and plot ---
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

if __name__ == "__main__":
    main()