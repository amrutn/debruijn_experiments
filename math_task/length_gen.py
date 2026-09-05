"""
Do the conditions generalise past the lengths they were trained on?

Training and test problems run 3-12 operations. This evaluates the trained
adapters across problems of 3-16 operations -- the whole trained range plus four
steps past it -- and reports test accuracy and derivation accuracy at every
length.

One figure puts problem length on the x-axis and draws two curves per condition:
final-answer accuracy as a solid line with circle markers and derivation accuracy
as a dashed line with diamond markers. Conditions are k = 1, 3 and the standard
condition, coloured along a warm ramp (dark red = small k, standard at the light
end), distinct from the blue perturbation ramp. No-CoT is added as a single black
final-answer line -- the no-reasoning baseline. The other intervals are dropped to
keep the figure legible. A light vertical line at length 12 marks where the
training data ends, so the trained range (left) and extrapolation range (right)
read at a glance.

The question is whether local structure buys length generalisation. A model that
genuinely chains single operations has no reason to care how many of them there
are: the k = 1 condition applies one operation from a stated equation, and that
subproblem is identical at length 16 and at length 6. A model that reconstructs
the answer from the whole context has every reason to care, because the context
it must integrate is now longer than any it was fitted on. So the prediction that
falls out of the two-regime picture is that accuracy should decay far more slowly
with length at small k than at large k, and that derivation accuracy should hold
up at small k even where answers start failing.

`run_experiments` runs this by default at the end of each budget mode, once the
adapters it needs are already trained, and writes both figures alongside the rest
(`--no-length-gen` skips it). It lives in its own module rather than in
`run_experiments` because it is the one figure that has to *decode* rather than
re-score what is cached, and because it carries its own problem generation and
the caching argument below. The module is also runnable on its own:

    python length_gen.py                 # every condition, every length
    python length_gen.py --cond 1 std --lengths 13 14
    python length_gen.py --plot-only     # cached decodes only, no GPU

Caching safety
--------------
Two things here could have been destructive and are deliberately not.

Adapters are looked up with the ORIGINAL `DATA` config, never with the long-eval
one. `min_ops` and `max_ops` sit in `DataConfig` and therefore in the adapter
cache key (`_ADAPTER_KEY_TEST` pins only `n_test` and `test_seed`), so evaluating
under a length-13 DataConfig would have changed every adapter key and silently
retrained all nine adapters. Passing `DATA` to `train_adapter` returns the cached
path and touches nothing.

Nothing about how the existing data is generated is modified. `_MIX` and
`DataConfig` are untouched, so every existing dataset, decode and score keeps its
hash. Long problems need a deeper onion than `_MIX` allows, so this builds its own
mix locally and hands it to `make_dataset` through the `mix` argument that already
exists for the purpose -- only the depth window moves, and every other generation
parameter (the five problem types, their proportions, and the x-shift, flip and
word-problem probabilities) is copied across verbatim.
"""

import os
# See the note in run_experiments: cut CUDA fragmentation before torch loads.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import json
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# Warm ramp for the interval curves, deliberately distinct from the blue injection
# ramp the k-axis figures use for p, so k is never read as p. Bright red (k=1) to
# light orange (largest k) -- the red end is kept clear of black so k=1 does not
# blend with the standard condition, which stays black.
_COND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'cond_warm', ['#e31a1c', '#fd8d3c', '#fdd49e'])

import generate_data as G
from generate_data import (derivation_correct, parse_answers, is_correct,
                           make_dataset, Problem, NO_COT)
from train_eval import (get_or_decode, train_adapter, load_for_eval, _key,
                        _decode_path, DATA_CACHE)
import run_experiments as R


def _is_oom(exc):
    """True for a CUDA out-of-memory error, however torch surfaces it."""
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower()


def _empty_cuda():
    """Best-effort release of the device cache after a caught OOM."""
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# Every operation count from the shortest trained problem to four past the
# longest. 3-12 is in-distribution and 13-16 is extrapolation; the figures mark
# the 12 boundary with a vertical line.
LENGTHS = tuple(range(3, 17))   # operations per problem; training ran 3-12
# The conditions the figure draws (and therefore the only ones decoded here):
# k = 1, 3, the standard condition, and no-CoT (the black no-reasoning baseline,
# plotted as a single final-answer line). The other intervals are dropped to keep
# the figure from over-crowding.
CONDS = [1, 3, None, NO_COT]
# 250 problems per length. This sets `_eval_cfg`, hence the decode cache key, so
# any decodes cached under a different count stay on disk unused rather than reused
# -- there is no way to extend a cached decode in place, so changing this re-decodes
# every (condition, length).
N_PER_LENGTH = 250
LONG_SEED = 900001              # disjoint from the train (0) and test (12345) seeds


# Operations a finished solution needs beyond the onion depth, by problem type.
# The onion contributes one operation per layer; the rest is the fixed cost of
# finishing that kind of equation -- a word problem opens by writing the equation
# down, a quadratic or cubic ends by factoring, a product expands before it
# factors. Measured, not assumed: see the table in `_check_overhead`.
_OVERHEAD = {('linear', False): 0, ('linear', True): 1,
             ('quadratic', False): 2, ('cubic', False): 2, ('product', False): 3}


def _long_mix(target):
    """
    `generate_data._MIX` with only the depth window moved.

    Each entry keeps its problem type and its x-shift / flip / word-problem
    probabilities exactly as the training mix has them; the depth range is
    re-centred so a finished solution lands on `target` operations. The window is
    kept two wide rather than pinned, because the overhead is a fixed cost per
    type but an x-shift can add a step -- `make_dataset`'s exact `min_ops` /
    `max_ops` filter does the final selection, so a loose window costs sampling
    attempts and never correctness.
    """
    out = []
    for core, _depth_range, opt in G._MIX:
        over = _OVERHEAD[(core, bool(opt['word']))]
        lo = max(0, target - over - 2)
        out.append((core, (lo, target - over), opt))
    return out


def long_problems(target, n=N_PER_LENGTH, seed=LONG_SEED, progress=True):
    """
    `n` problems whose canonical solution is exactly `target` operations.

    Cached in the same directory as the ordinary datasets but under a key that
    carries the target length and this module's mix, so it can never collide with
    a cached train or test split.
    """
    # the generator version rides along so a change to `generate_data`
    # invalidates these sets exactly as it invalidates the ordinary ones
    spec = dict(split='length-gen', n=n, seed=seed + target, target_ops=target,
                gen_version=R.DATA.gen_version, mix=_long_mix(target))
    path = os.path.join(DATA_CACHE, _key(spec) + '.json')
    if os.path.exists(path):
        with open(path) as f:
            return [Problem.from_dict(d) for d in json.load(f)]
    print(f'generating {n} problems of exactly {target} operations '
          f'(rejection-sampled, this is slow) ...', flush=True)
    probs = make_dataset(n, seed=seed + target, min_ops=target, max_ops=target,
                         mix=_long_mix(target), progress=progress)
    os.makedirs(DATA_CACHE, exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump([p.to_dict() for p in probs], f)
    os.replace(tmp, path)
    return probs


def _eval_cfg(target):
    """
    The DataConfig used to key long-eval decodes -- never to train or to look up
    an adapter.

    It differs from `DATA` in the test fields only, which is what makes the decode
    cache entry distinct; `test_seed` carries the length so two lengths can never
    share a key even if the op filters were to coincide.
    """
    return replace(R.DATA, n_test=N_PER_LENGTH, min_ops=target, max_ops=target,
                   test_seed=LONG_SEED + target)


def _sem(p, n):
    """Standard error of a proportion estimated from `n` independent trials."""
    return (p * (1.0 - p) / n) ** 0.5 if n else 0.0


def score(problems, records):
    """
    Test and derivation accuracy over one decoded condition, each with its SEM.

    Both are proportions over the same `n` problems, so their standard error is
    the binomial ``sqrt(p(1-p)/n)``. The SEMs are kept on the rows for reference
    even though the current figure no longer draws error bars.
    """
    n_ans = n_der = 0
    for prob, rec in zip(problems, records):
        ops = [x['text'] for x in rec if x['kind'] == 'op']
        ans = next((x['text'] for x in rec if x['kind'] == 'answer'), '')
        ok = bool(is_correct(prob, ans))
        n_ans += ok
        n_der += bool(derivation_correct(prob, ops, parse_answers(ans)))
    n = max(len(problems), 1)
    acc, der = n_ans / n, n_der / n
    return acc, der, _sem(acc, n), _sem(der, n)


def is_cached(k, L, mode):
    """True if this (condition, length) has already been decoded."""
    return os.path.exists(_decode_path(k, 0.0, _eval_cfg(L), R.MODEL,
                                       R.train_cfg(k, mode), R.EVAL))


def _length_jobs(conds, lengths, mode, device, force, decode, progress_pos=0,
                 log=print):
    """
    Score every (condition, length) on one device, sequentially.

    One condition is one task: the adapter is loaded once and kept resident
    across all of its lengths, because reloading the base model per length would
    cost far more than any imbalance it could save.
    """
    rows = []
    todo = [(k, L) for k in conds for L in lengths
            if force or not is_cached(k, L, mode)]

    for k in conds:
        need = [L for L in lengths if (k, L) in todo] if decode else []
        pack = {}

        # the adapter is only resolved when something actually has to be decoded,
        # so a fully cached run never trains, loads a model or touches the GPU
        def get_model(_k=k):
            # `decode=False` promises no model is loaded. `get_or_decode` can
            # still reach for one if a cached decode turns out to be unreadable
            # or the wrong length, so make that a loud failure rather than a
            # silent GPU load in what the caller asked to be a cache-only pass.
            if not decode:
                raise RuntimeError(
                    f'length_results(decode=False) needed to decode '
                    f'k={R._cond_label(_k)}: its cached decode is unusable. '
                    f'Re-run with decode=True to rebuild it.')
            if 'p' not in pack:
                # ORIGINAL data config: this resolves to the already trained
                # adapter. Passing the long-eval config would change the adapter
                # key -- `_ADAPTER_KEY_TEST` pins only n_test and test_seed, not
                # min_ops/max_ops -- and silently retrain every condition.
                adapter = train_adapter(_k, R.data_cfg(mode), R.MODEL,
                                        R.train_cfg(_k, mode), device=device,
                                        log=lambda m: log(f'[{mode}] {m}'))
                pack['p'] = load_for_eval(adapter, R.MODEL,
                                          R.train_cfg(_k, mode), device=device)
            return pack['p']

        for L in lengths:
            if not (is_cached(k, L, mode) or L in need):
                continue
            probs = long_problems(L)
            # A long trace at the eval batch on a shared GPU can OOM. Rather than
            # skip the unit (which would leave a hole in the figure), retry it at
            # a halved batch until it fits. `batch_override` keeps the decode keyed
            # by the canonical EVAL, so the result is cached normally and a later
            # run finds it instead of re-OOMing. Only a genuine batch-1 OOM raises.
            batch = R.EVAL.batch_size
            while True:
                try:
                    _, records = get_or_decode(
                        get_model, probs, k, 0.0, _eval_cfg(L), R.MODEL,
                        R.train_cfg(k, mode), R.EVAL, device=device,
                        # never force in a cache-only pass: forcing would re-decode
                        # every cached unit and defeat `decode=False`
                        force=force and decode,
                        desc=f'{R._cond_label(k)} len={L} decode', log=log,
                        progress_pos=progress_pos,
                        batch_override=None if batch == R.EVAL.batch_size else batch)
                    break
                except Exception as e:
                    if _is_oom(e) and batch > 1:
                        _empty_cuda()
                        batch = max(1, batch // 2)
                        log(f'  [length] OOM on k={R._cond_label(k)} len={L}; '
                            f'retrying at batch {batch}')
                        continue
                    raise
            acc, der, acc_sem, der_sem = score(probs, records)
            rows.append(dict(cond=k, length=L, n=len(probs), mode=mode,
                             accuracy=acc, derivation=der,
                             accuracy_sem=acc_sem, derivation_sem=der_sem))
        if pack:
            pack.clear()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
    return rows


_STOP = '__stop__'      # queue sentinel; None is a real condition (standard)


def _length_worker(queue, lengths, mode, device, force, decode, progress_pos):
    """Pull conditions off the shared queue until it is drained."""
    out = []
    while True:
        k = queue.get()
        if k == _STOP:
            break
        print(f'[{device}] starting {mode} length-gen k={R._cond_label(k)}',
              flush=True)
        out.extend(_length_jobs([k], lengths, mode, device, force, decode,
                                progress_pos))
    print(f'[{device}] length-gen done', flush=True)
    return out


def length_results(mode='samples', conds=None, lengths=LENGTHS, devices=('cuda',),
                   force=False, decode=True, log=print):
    """
    Test and derivation accuracy for every (condition, length), from cache when
    possible.

    Decoding is clean only. Injection asks about robustness, which is a separate
    axis, and mixing it in here would confound length with perturbation.

    Work is spread over `devices` with the same shared queue `run_all` uses: one
    condition per task, handed out on demand so a GPU that finishes early picks up
    the next condition instead of idling.

    Params
    ------
    devices : sequence[str]
        Torch devices. A single-element sequence runs in this process without
        spawning anything.
    decode : bool
        False evaluates from cached decodes alone and never loads a model, which
        is what `--plot-only` needs. Uncached units are skipped rather than
        decoded, so the figure is built from whatever exists.

    Returns
    -------
    list[dict]
        {'cond', 'length', 'n', 'accuracy', 'derivation', 'accuracy_sem',
        'derivation_sem', 'mode'}
    """
    # k = 1, 3, 5 and the standard condition are the curves the figure draws;
    # nothing else (the other intervals, no-CoT) is decoded here. See `CONDS`.
    conds = list(CONDS if conds is None else conds)
    lengths = tuple(lengths)
    devices = [devices] if isinstance(devices, str) else list(devices)

    todo = [(k, L) for k in conds for L in lengths
            if force or not is_cached(k, L, mode)]
    if todo and not decode:
        log(f'  [length] {len(todo)} unit(s) not cached; skipping them '
            f'(no model is loaded when decode=False)')
    if not todo or not decode or len(devices) == 1:
        # nothing to decode, or a cache-only pass, or one device: no point paying
        # for processes
        return _length_jobs(conds, lengths, mode, devices[0], force, decode,
                            log=log)

    # Generate (and cache) every long problem set HERE, in the parent. Each
    # worker calls `long_problems`, so without this every GPU would redo the same
    # rejection-sampled SymPy generation concurrently.
    for L in lengths:
        long_problems(L)

    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    queue = manager.Queue()
    for k in conds:
        queue.put(k)
    for _ in devices:
        queue.put(_STOP)

    rows = []
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as ex:
        futs = [ex.submit(_length_worker, queue, lengths, mode, d, force, decode, i)
                for i, d in enumerate(devices)]
        for f in futs:
            rows.extend(f.result())
    return rows


def plot_length_gen(rows, mode, subdir='', log=print):
    """The combined length-generalisation figure. Returns the paths written."""
    path = _plot(rows, mode, subdir=subdir)
    if not path:
        return []
    log(f'wrote {path}.pdf/.png')
    return [path]


def print_length_gen(rows, mode):
    """The table behind the two figures."""
    if not rows:
        return
    lengths = sorted({r['length'] for r in rows})
    conds = [k for k in CONDS
             if any(r['cond'] == k for r in rows)]
    by = {(r['cond'], r['length']): r for r in rows}
    print(f'\n=== length generalisation [{mode}, clean decodes] ===')
    print(f'  training problems ran {R.DATA.min_ops}-{R.DATA.max_ops} operations;'
          f' lengths past {R.DATA.max_ops} are extrapolation')
    print('  acc = final answer correct;  der = written operations derive it\n')
    print('  cond  ' + '  '.join(f'{L:>4d} ops     ' for L in lengths))
    print('        ' + '  '.join('  acc     der' for _ in lengths))
    for k in conds:
        cells = []
        for L in lengths:
            r = by.get((k, L))
            cells.append('    -       -' if r is None
                         else f"{r['accuracy']:6.3f}  {r['derivation']:6.3f}")
        print(f'  {R._cond_label(k):>4}  ' + '  '.join(cells))


# The interval conditions are packed into the darker part of the warm ramp (so the
# gaps between k = 1, 3, 5 stay small) and standard is placed at the light end,
# roughly where k = 5 used to sit -- not the extreme, so it stays visible. No-CoT is
# off this ramp entirely: it is the black no-reasoning baseline.
_K_RAMP_MAX = 0.6


def _ramp_colors(ks, has_std):
    """
    Warm colour per interval k (compressed into `[0, _K_RAMP_MAX]`, dark -> light)
    plus, if present, the standard condition at the light end of the ramp.
    """
    out = {}
    order = sorted(ks)
    m = max(len(order) - 1, 1)
    for i, k in enumerate(order):
        out[k] = _COND_CMAP(_K_RAMP_MAX * i / m)
    if has_std:
        out[None] = _COND_CMAP(1.0)
    return out


def _plot(rows, mode, subdir=''):
    """
    Test and derivation accuracy vs problem length, one colour per condition.

    Problem length is the x-axis. Each interval condition and the standard condition
    contribute two curves in their own colour: test accuracy as a solid line with
    circle markers and derivation accuracy as a dashed line with diamond markers.
    Interval conditions take warm colours (dark red = small k); standard sits at the
    light end of that ramp. No-CoT is drawn as a single black test-accuracy line (it
    emits no operations, so it has no derivation to score). A light vertical line at
    the last trained length (`R.DATA.max_ops`) separates the in-distribution range
    from the extrapolation range to its right.
    """
    ks = sorted({r['cond'] for r in rows if isinstance(r['cond'], int)})
    has_std = any(r['cond'] is None for r in rows)
    has_nocot = any(r['cond'] == NO_COT for r in rows)
    lengths = sorted({r['length'] for r in rows})
    if not lengths or (not ks and not has_std):
        return None
    by = {(r['cond'], r['length']): r for r in rows}
    colors = _ramp_colors(ks, has_std)

    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.axvline(R.DATA.max_ops, color='0.8', lw=1.0, zorder=0)   # training ends here
    series = [(k, colors[k], rf'$k={k}$') for k in ks]
    if has_std:
        series.append((None, colors[None], 'std.'))
    handles = []
    for cond, col, label in series:
        xs = [L for L in lengths if (cond, L) in by]
        if not xs:
            continue
        ax.plot(xs, [by[(cond, L)]['accuracy'] for L in xs],
                marker='o', ms=4.0, lw=1.5, color=col, zorder=2)
        ax.plot(xs, [by[(cond, L)]['derivation'] for L in xs],
                marker='D', ms=3.6, lw=1.3, ls=(0, (3, 2)), color=col, zorder=2)
        handles.append(Line2D([], [], color=col, marker='o', ms=4, lw=1.5,
                              label=label))
    # no-reasoning baseline: a single black test-accuracy line (no ops => no
    # derivation to score, so no dashed twin)
    if has_nocot:
        xs = [L for L in lengths if (NO_COT, L) in by]
        if xs:
            ax.plot(xs, [by[(NO_COT, L)]['accuracy'] for L in xs],
                    marker='o', ms=4.0, lw=1.5, color='black', zorder=2)
            handles.append(Line2D([], [], color='black', marker='o', ms=4, lw=1.5,
                                  label='no reasoning'))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Problem length (operations)', fontsize=R.LABEL_FS)
    ax.set_ylabel('Accuracy', fontsize=R.LABEL_FS)
    # label the even lengths only; the leftmost (3) is intentionally left unlabelled
    shown = {L for L in lengths if L % 2 == 0}
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(L) if L in shown else '' for L in lengths])
    R._style_axis(ax)
    handles += [Line2D([], [], color='0.35', lw=1.5, marker='o', ms=4,
                       label='final-answer'),
                Line2D([], [], color='0.35', lw=1.3, ls=(0, (3, 2)), marker='D',
                       ms=3.6, label='derivation')]
    leg = ax.legend(handles=handles, fontsize=R.LEGEND_FS - 1, frameon=True,
                    loc='lower left', ncol=2, handlelength=1.4,
                    handletextpad=0.5, labelspacing=0.25, columnspacing=1.0,
                    borderaxespad=0.2, facecolor='white', edgecolor='0.8',
                    framealpha=1.0, borderpad=0.35)
    leg.get_frame().set_linewidth(0.7)
    leg.set_zorder(10)
    return R._save(fig, f'math_length_gen_{mode}', subdir)


def _check_overhead(n=25):
    """Print the depth -> n_ops overhead per problem type, to justify `_OVERHEAD`."""
    import random
    rng = random.Random(0)
    print('  core            depth    median n_ops    overhead')
    for core, _dr, opt in G._MIX:
        for d in (10, 12):
            ns = [p.n_ops for p in
                  (G.make_problem(rng, depth=d, core=core,
                                  xshift=rng.random() < opt['xshift'],
                                  flipped=rng.random() < opt['flipped'],
                                  word=rng.random() < opt['word'])
                   for _ in range(n)) if p]
            med = sorted(ns)[len(ns) // 2] if ns else float('nan')
            tag = core + ('/word' if opt['word'] else '')
            print(f'  {tag:14s}  {d:5d}    {med:12.0f}    {med - d:+8.0f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=list(R.BUDGET_MODES), default='samples')
    ap.add_argument('--cond', nargs='*', default=None,
                    help='conditions to evaluate, e.g. 1 5 std none (default: all)')
    ap.add_argument('--lengths', nargs='*', type=int, default=list(LENGTHS))
    ap.add_argument('--devices', default=None,
                    help='comma-separated torch devices, e.g. cuda:0,cuda:1 '
                         '(default: every visible GPU, as run_experiments does)')
    ap.add_argument('--force', action='store_true', help='re-decode even if cached')
    ap.add_argument('--plot-only', action='store_true',
                    help='build the figures from cached decodes only')
    ap.add_argument('--check-overhead', action='store_true',
                    help='print the depth/n_ops table behind _OVERHEAD and exit')
    # default to the same directory `run_experiments` writes this mode's figures
    # into, so a standalone run overwrites the integrated one rather than leaving
    # two copies in different places
    ap.add_argument('--subdir', default=None)
    args = ap.parse_args()

    if args.check_overhead:
        _check_overhead()
        return

    def wanted(k):
        if args.cond is None:
            return True
        return R._cond_label(k) in args.cond

    devices = (args.devices.split(',') if args.devices
               else R._default_devices())
    rows = length_results(mode=args.mode,
                          conds=[k for k in CONDS if wanted(k)],
                          lengths=tuple(args.lengths), devices=devices,
                          force=args.force, decode=not args.plot_only)
    if not rows:
        print('nothing evaluated')
        return
    print_length_gen(rows, args.mode)
    plot_length_gen(rows, args.mode,
                    subdir=args.subdir if args.subdir is not None
                    else R.fig_subdir(args.mode))


if __name__ == '__main__':
    main()
