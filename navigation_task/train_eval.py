"""
Train and evaluate a decoder-only Transformer on the grid navigation task.

The model (reused verbatim from `synthetic_experiment/model.py`) is trained to
emit a shortest-path trace given a ``s -> g`` prompt, under one of the two
encodings from `generate_data`. Cross-entropy is applied to the *answer* tokens
only (the prompt is context). The metric is held-out shortest-path accuracy:
greedy-decode each unseen test prompt and check that the emitted moves walk a
shortest path from ``s`` to ``g``.

This mirrors `synthetic_experiment/train_eval.py` (same `ModelConfig`,
`TrainConfig`, `build_model`, per-unit caching), specialised to the navigation
task: answer-masked training and a decode-based accuracy metric instead of the
illegal-mass / path-coverage probes.
"""

import os
import json
import hashlib
import importlib.util
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F

from generate_data import (
    NavGrid, split_prompts, sample_batch, max_tokens, dag_edges,
)

# Reuse the synthetic experiment's Transformer verbatim ("use their model.py").
# It is loaded by explicit path (it is standalone: only math/torch/einx) so this
# module keeps its own local `generate_data` without a sys.path name clash.
_SYN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'synthetic_experiment')
_spec = importlib.util.spec_from_file_location('syn_model', os.path.join(_SYN_DIR, 'model.py'))
_syn_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_syn_model)
TransformerLM = _syn_model.TransformerLM

_DTYPES = {'float32': torch.float32, 'float64': torch.float64, 'bfloat16': torch.bfloat16}


@dataclass
class ModelConfig:
    """
    Architecture of the decoder-only Transformer (forwarded to `TransformerLM`;
    mirrors `synthetic_experiment.train_eval.ModelConfig`). `vocab_size` and
    `max_seq_len` are supplied separately by `build_model`.
    """
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int | None = None          # None -> SwiGLU picks ~8/3 * d_model
    theta: float = 10000.0
    use_rope: bool = True


@dataclass
class TrainConfig:
    """
    Optimisation settings for the training loop (mirrors
    `synthetic_experiment.train_eval.TrainConfig`).
    """
    lr: float = 1e-3
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.98)
    batch_size: int = 256
    grad_clip: float | None = 1.0
    seed: int = 0                    # weight init + data-shuffle seed
    device: str | None = None        # None -> auto (cuda, then mps, then cpu)
    dtype: str = 'float32'


def resolve_device(device):
    """Turn a device spec into a concrete torch.device (auto: cuda, mps, cpu)."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_model(vocab_size, max_seq_len, mcfg, device, dtype, seed):
    """Instantiate a fresh, seeded `TransformerLM` on `device`."""
    device = resolve_device(device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return TransformerLM(
        d_model=mcfg.d_model, d_ff=mcfg.d_ff, num_heads=mcfg.num_heads,
        vocab_size=vocab_size, num_layers=mcfg.num_layers, theta=mcfg.theta,
        max_seq_len=max_seq_len, use_rope=mcfg.use_rope, rng=gen,
        dtype=dtype, device=device,
    )


# ----------------------------------------------------------------------------
# training (answer-masked cross-entropy)
# ----------------------------------------------------------------------------

def train(model, tokens, lengths, answer_start, tcfg, epochs=1):
    """
    Train `model` on a batch of encoded traces for `epochs` passes.

    Loss is next-token cross-entropy on the answer tokens only: for each sample,
    input positions ``[answer_start-1, length-2]`` predict the answer tokens at
    ``[answer_start, length-1]`` (the final one is EOS). Prompt positions are
    context and are never scored.

    Params
    ------
    model : TransformerLM
        Model to train in place.
    tokens : (num, T) int array
        EOS-padded token matrix from `sample_batch`.
    lengths : (num,) int array
        Real token count per sample (through EOS).
    answer_start : (num,) int array
        Index of the first answer token per sample.
    tcfg : TrainConfig
        Optimisation settings (lr, batch_size, device, seed, ...).
    epochs : int
        Number of passes over the data.

    Returns
    -------
    float
        Mean per-batch training loss over the last epoch (nan if no batches).
    """
    device = resolve_device(tcfg.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                            weight_decay=tcfg.weight_decay, betas=tuple(tcfg.betas))

    ft = torch.from_numpy(np.asarray(tokens, dtype=np.int64))
    lt = torch.from_numpy(np.asarray(lengths, dtype=np.int64))
    at = torch.from_numpy(np.asarray(answer_start, dtype=np.int64))
    num, K = ft.shape
    pos = torch.arange(K - 1, device=device)

    g = torch.Generator().manual_seed(int(tcfg.seed))
    losses = []
    for _ in range(epochs):
        order = torch.randperm(num, generator=g)
        for s in range(0, num, tcfg.batch_size):
            idx = order[s:s + tcfg.batch_size]
            batch = ft[idx].to(device)
            blen = lt[idx].to(device)
            bans = at[idx].to(device)

            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            logits = model(inp)                              # (b, K-1, vocab)

            # score input positions [answer_start-1, length-2] -> answer tokens
            mask = (pos[None, :] >= bans[:, None] - 1) & (pos[None, :] <= blen[:, None] - 2)
            loss = F.cross_entropy(logits[mask], tgt[mask])

            opt.zero_grad()
            loss.backward()
            if tcfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float('nan')


# ----------------------------------------------------------------------------
# greedy decoding
# ----------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(model, grid, prompts, interval, tcfg, batch_size=512):
    """
    Greedily decode an answer for each prompt.

    Each prompt ``s -> g`` is encoded to its prompt tokens, then the model
    autoregressively appends argmax tokens until EOS or the encoding's maximum
    length (for the given emission `interval`). Decoding is batched over prompts.

    Returns
    -------
    list[np.ndarray]
        The generated answer tokens (excluding the prompt, up to but not
        including EOS) for each prompt, in order.
    """
    device = resolve_device(tcfg.device)
    model.eval()
    v = grid.vocab
    T = max_tokens(grid, interval)

    out = [None] * len(prompts)
    for s0 in range(0, len(prompts), batch_size):
        chunk = prompts[s0:s0 + batch_size]
        # prompt tokens are identical for every interval: label(s) -> label(g)
        plen = 2 * grid.label_len + 1
        seq = torch.full((len(chunk), T), v.EOS, dtype=torch.long, device=device)
        for i, (s, g) in enumerate(chunk):
            ptoks = grid.label_tokens(s) + [v.ARROW] + grid.label_tokens(g)
            seq[i, :plen] = torch.tensor(ptoks, device=device)
        done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
        gen_start = plen
        for t in range(gen_start, T):
            logits = model(seq[:, :t])                 # (b, t, vocab)
            nxt = logits[:, -1, :].argmax(dim=-1)      # (b,)
            nxt = torch.where(done, torch.full_like(nxt, v.EOS), nxt)
            seq[:, t] = nxt
            done = done | (nxt == v.EOS)
            if bool(done.all()):
                break
        seqc = seq.cpu().numpy()
        for i in range(len(chunk)):
            row = seqc[i, gen_start:]
            eos = np.where(row == v.EOS)[0]
            end = eos[0] if len(eos) else len(row)
            out[s0 + i] = row[:end]
    return out


# ----------------------------------------------------------------------------
# shortest-path accuracy
# ----------------------------------------------------------------------------

def _decoded_moves(grid, answer):
    """The move tokens (N/E/S/W), in order, from a decoded answer sequence."""
    v = grid.vocab
    inv = {v.N: 'N', v.E: 'E', v.S: 'S', v.W: 'W'}
    return [inv[t] for t in answer if t in inv]


def shortest_path_accuracy(grid, prompts, decoded):
    """
    Fraction of prompts whose decoded moves walk a shortest path ``s -> g``.

    A decode is correct iff, applying its moves from ``s``, every step stays on
    the grid, the walk ends exactly at ``g``, and the number of moves equals the
    L1 distance (so the path is a shortest one). The emitted cell labels (De
    Bruijn) are not required for this metric -- see `worldmodel_consistency` for
    that -- so the two encodings are scored by the same criterion.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    ok = 0
    for (s, g), ans in zip(prompts, decoded):
        moves = _decoded_moves(grid, ans)
        if len(moves) != grid.dist(s, g):
            continue
        cell, valid = s, True
        for d in moves:
            if d not in grid.valid_dirs(cell):
                valid = False
                break
            cell = grid.step(cell, d)
        if valid and cell == g:
            ok += 1
    return ok / len(prompts) if prompts else float('nan')


def worldmodel_consistency(grid, prompts, decoded):
    """
    Fraction of decodes whose emitted cell labels all match the cells actually
    reached by the interleaved moves -- i.e. the model wrote out an internally
    coherent world-model trajectory. Applies to any emission interval: it walks
    the decoded answer, applying each move to a running position and checking that
    every emitted cell label equals that position (and that the run ends at the
    goal). For the standard encoding this checks only the start/goal bookend
    labels; for shorter intervals it checks every intermediate cell too.
    """
    v = grid.vocab
    dir_name = {v.N: 'N', v.E: 'E', v.S: 'S', v.W: 'W'}
    b, L = v.digit_base, grid.label_len

    ok = 0
    for (s, g), ans in zip(prompts, decoded):
        i, pos, good = 0, s, True
        while i < len(ans) and good:
            t = int(ans[i])
            if t in dir_name:                                # a move
                d = dir_name[t]
                if d not in grid.valid_dirs(pos):
                    good = False
                else:
                    pos = grid.step(pos, d)
                i += 1
            elif t < b and i + L <= len(ans) and all(ans[i + k] < b for k in range(L)):
                cell = sum(int(ans[i + k]) * b ** (L - 1 - k) for k in range(L))
                if cell != pos:                              # emitted label != true position
                    good = False
                i += L
            else:
                good = False                                 # malformed token
        if good and pos == g:
            ok += 1
    return ok / len(prompts) if prompts else float('nan')


# ----------------------------------------------------------------------------
# per-unit driver with caching (mirrors synthetic_experiment)
# ----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT_CACHE_DIR = os.path.join(HERE, 'cache', 'exp_results')


def _unit_key(spec):
    blob = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _unit_spec_path(m, interval, samples_per_cell, num_train, test_frac, eval_cap,
                    epochs, seed, prompt_seed, min_dist, mcfg, tcfg):
    """
    Resolve the training-set size and build the cache spec + file path for a unit.
    Shared by `run_unit` and `cached_unit` so the cache key never drifts between
    the writer and the up-front cache check.
    """
    tcfg.seed = seed
    if samples_per_cell is not None:
        num_train = int(round(samples_per_cell * (m * m)))
    if num_train is None:
        raise ValueError('provide samples_per_cell or num_train')
    spec = dict(task='nav', m=m, interval=interval, num_train=num_train,
                test_frac=test_frac, eval_cap=eval_cap, epochs=epochs, seed=seed,
                prompt_seed=prompt_seed, min_dist=min_dist,
                model=asdict(mcfg), train=asdict(tcfg))
    return spec, num_train, os.path.join(UNIT_CACHE_DIR, _unit_key(spec) + '.json')


def cached_unit(m, interval, samples_per_cell=None, num_train=None, test_frac=0.2,
                eval_cap=2000, epochs=1, seed=0, prompt_seed=0, mcfg=None, tcfg=None,
                min_dist=1):
    """
    Return the cached result dict for a unit if it exists, else None -- without
    building a model or touching a device. Lets a caller check the cache up front
    and skip the run machinery (and any GPU spawn) for units already computed. The
    arguments must match the corresponding `run_unit` call so the key agrees.
    """
    mcfg = mcfg or ModelConfig()
    tcfg = tcfg or TrainConfig()
    _, _, path = _unit_spec_path(m, interval, samples_per_cell, num_train, test_frac,
                                 eval_cap, epochs, seed, prompt_seed, min_dist, mcfg, tcfg)
    if os.path.exists(path):
        with open(path) as f:
            r = json.load(f)
        r['cached'] = True
        return r
    return None


def run_unit(m, interval, samples_per_cell=None, num_train=None, test_frac=0.2,
             eval_cap=2000, epochs=1, seed=0, prompt_seed=0, mcfg=None, tcfg=None,
             min_dist=1, force=False):
    """
    One point of the accuracy-vs-edges figure.

    Builds an ``m x m`` grid, holds out a fixed fraction of prompts as a test set,
    trains a fresh model on traces sampled at the given emission `interval`
    (`epochs` passes) and reports held-out shortest-path accuracy plus that
    encoding's exact minimal-DAG edge count over the training prompts. Cached by
    the full config.

    `interval` sweeps the encoding from De Bruijn (``interval=1``, cell every step,
    O(M) edges) to standard (``interval=None``, cell only at the ends, O(M^2)
    edges); intermediate values (cell every k steps) land between the two on the
    accuracy-vs-edges plot.

    Params
    ------
    m : int
        Grid side length.
    interval : int | None
        Cell-emission interval: 1 = De Bruijn, k = every k moves, None = standard.
    samples_per_cell : float | None
        Training traces per grid cell; the budget is ``round(samples_per_cell*M)``.
        Provide this or `num_train` (`samples_per_cell` takes precedence).
    num_train : int | None
        Explicit fixed training-set size, used when `samples_per_cell` is None.
    test_frac : float
        Fraction of prompts held out as the test set.
    eval_cap : int
        Evaluate accuracy on at most this many held-out prompts (a random subset
        of the held-out set, for decode speed on large grids).
    epochs : int
        Passes over the training samples.
    seed : int
        Model-init + training-shuffle + data-sampling seed.
    prompt_seed : int
        Seed for the train/test prompt split (shared across intervals so every
        condition on a grid sees the same task and the same held-out prompts).
    mcfg : ModelConfig | None
        Architecture (defaults to `ModelConfig()`).
    tcfg : TrainConfig | None
        Optimisation (defaults to `TrainConfig()`); its `seed`/`device` are set
        from `seed` and the caller's device.
    min_dist : int
        Minimum start-goal distance for prompts.
    force : bool
        Recompute even if a cached result exists.

    Returns
    -------
    dict
        {'m', 'M', 'interval', 'edges', 'accuracy', 'wm_consistency', 'train_loss',
         'num_train', 'num_train_prompts', 'num_eval', 'seed', 'cached'}.
    """
    mcfg = mcfg or ModelConfig()
    tcfg = tcfg or TrainConfig()
    tcfg.seed = seed

    spec, num_train, path = _unit_spec_path(m, interval, samples_per_cell, num_train,
                                            test_frac, eval_cap, epochs, seed,
                                            prompt_seed, min_dist, mcfg, tcfg)
    os.makedirs(UNIT_CACHE_DIR, exist_ok=True)
    if os.path.exists(path) and not force:
        with open(path) as f:
            r = json.load(f)
        r['cached'] = True
        return r

    grid = NavGrid(m)
    train_prompts, test_prompts = split_prompts(grid, test_frac=test_frac,
                                                seed=prompt_seed, min_dist=min_dist)
    edges = dag_edges(grid, train_prompts, interval)

    # evaluate on a capped random subset of the held-out prompts (decode speed)
    if len(test_prompts) > eval_cap:
        idx = np.random.default_rng([prompt_seed, 7]).choice(
            len(test_prompts), eval_cap, replace=False)
        eval_prompts = [test_prompts[i] for i in idx]
    else:
        eval_prompts = test_prompts

    device = resolve_device(tcfg.device)
    dtype = _DTYPES[tcfg.dtype]
    T = max_tokens(grid, interval)
    model = build_model(grid.vocab.size, T + 1, mcfg, device, dtype, seed)

    tok, ln, a0 = sample_batch(grid, train_prompts, num_train, interval,
                               np.random.default_rng([seed, 1]), T=T)
    train_loss = train(model, tok, ln, a0, tcfg, epochs=epochs)

    decoded = greedy_decode(model, grid, eval_prompts, interval, tcfg,
                            batch_size=tcfg.batch_size)
    acc = shortest_path_accuracy(grid, eval_prompts, decoded)
    wmc = worldmodel_consistency(grid, eval_prompts, decoded)

    r = dict(m=m, M=grid.M, interval=interval, edges=int(edges),
             accuracy=float(acc), wm_consistency=float(wmc),
             train_loss=float(train_loss), num_train=num_train,
             num_train_prompts=len(train_prompts), num_eval=len(eval_prompts),
             seed=seed, cached=False)
    with open(path, 'w') as f:
        json.dump(r, f, indent=1)
    return r
