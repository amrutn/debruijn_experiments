"""
Run the equation-solving experiment: fine-tune one Qwen2.5-1.5B-Instruct LoRA per
state-emission interval k and measure held-out accuracy with and without
per-step operation injection.

Each condition k emits the current equation every k reasoning steps; the standard
condition never emits it (`generate_data.encode_trace`). A separate model is
fine-tuned per condition on the *same* problems, so the only difference between
models is how much of the algebraic state sits in their local context.

Unlike the navigation task the number of distinct transitions cannot be counted
exactly here, so the x-axis is k itself, with the standard condition drawn as a
separate point to its right.

Figure (pdf+png in figures/)
----------------------------
math_accuracy_vs_k    Held-out accuracy vs k. The solid line is clean decoding;
                      the dashed lines re-evaluate the same models while a random
                      solution-preserving operation is injected at each step with
                      probability p. Injected operations never change the answer,
                      so a model that tracks the algebraic state can recover.

Scheduling
----------
Each interval is pinned to one GPU, which trains its adapter and then evaluates
every injection probability with that model still resident. Everything is cached
(`train_eval`): datasets, LoRA adapters, and per-(k, p) eval results, so a rerun
recomputes only what is missing.

Usage
-----
    python run_experiments.py                                   # auto devices
    python run_experiments.py --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4
    python run_experiments.py --plot-only                       # figure from cache
    python run_experiments.py --force
"""

import os
# Reduce CUDA fragmentation before torch is imported (train_eval imports it):
# the decode loop re-prefills a growing context every step, and on a shared GPU
# the reserved-but-unallocated slack is what tips a long trace into OOM.
# `setdefault` so an explicit environment value still wins. Spawned workers
# re-import this module, so they inherit the setting before their own torch load.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import time
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from dataclasses import replace

import json

from train_eval import (
    DataConfig, ModelConfig, TrainConfig, EvalConfig,
    run_unit, cached_unit, train_adapter, load_for_eval, adapter_is_cached,
    get_problems, mean_tokens_per_example, _decode_path, _key, CACHE,
    _data_spec, _train_spec,
)
from generate_data import (derivation_correct, derivation_correct_replaced,
                           parse_answers, is_correct, NO_COT)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')


# ----------------------------------------------------------------------------
# the sweep
# ----------------------------------------------------------------------------

# State-emission intervals. 1 = emit the current equation after every operation;
# None = standard (never emitted).
#
# The sweep stops at 7 because past it the state metrics stop describing the test
# set. A report needs ops >= k, so the population each k is scored on shrinks as k
# grows: 53.3% of problems emit any state at k = 7, and only 6.9% by k = 11. The
# conditions beyond 7 also converge onto standard by construction rather than as a
# finding -- at k = 11 no state is emitted at all on 93% of problems, so the model
# is simply producing standard-shaped output most of the time.
#
# Nothing reads the higher intervals any more. They were once needed by a
# predicted-standard marker that mapped each problem's length onto the k = n_ops
# row; that marker presupposed the interval-k models were reasoning step by step,
# which the recovery gradient contradicts, and it has been removed.
INTERVALS = [1, 2, 3, 4, 5, 6, 7, None]

# Every condition that gets its own adapter. NO_COT is trained and evaluated like
# the rest but is not a k, so it is kept out of INTERVALS and off the k axis.
CONDITIONS = INTERVALS + [NO_COT]

# Per-step injection probabilities. 0.0 is clean decoding (the reference curve).
# The low end is sampled finely because that is where the conditions separate:
# at p=0.5 a mid-length trace takes ~4 injections and everything is degraded.
INJECT_PS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

# p = 1.0 is evaluated for the standard condition only. Replacing every operation
# leaves no model-authored content before the answer, so it is the direct test of
# whether the answer depends on the trace -- but only for standard. At k >= 1 the
# model still writes all of its own state reports, making it a different condition
# entirely, and one where traces frequently run to the step cap without answering.
STD_ONLY_PS = (1.0,)

# Perturbation probabilities kept out of the *figures* (still computed, still in the
# printed tables). p = 0.5 is dropped: under that much perturbation the traces are
# degraded enough that its curve adds clutter more than signal.
PLOT_EXCLUDE_PS = (0.5,)

# 1000 test problems rather than 500. State accuracy is measured per *report*, and
# a trace emits floor(ops / k) of them, so the denominator shrinks as k grows: at
# 500 problems the sparse high-k, high-p cells rested on so few reports that a zero
# count was unremarkable. Doubling the test set doubles every such denominator, and
# capping the sweep at k = 7 keeps the thinnest cells out of the sweep entirely.
# The adapter keys pin the test split (`train_eval._ADAPTER_KEY_TEST`), so this
# re-runs evaluation without retraining anything.
DATA = DataConfig(n_train=50000, n_test=1000, min_ops=3, max_ops=12)

# Two training budgets, run as two separate experiments over the same conditions.
#
#   'samples'  every condition trains on the same NUMBER OF TRACES. Traces get
#              longer as k shrinks, so small k also gets more tokens -- the
#              comparison holds data constant and lets compute vary.
#   'compute'  every condition trains on the same NUMBER OF TOKENS. The standard
#              condition (shortest traces) sets the budget at SAMPLES_STD traces;
#              every other k gets proportionally fewer traces.
#
# Reporting both separates "state in context helps" from "state in context buys
# more gradient signal per trace".
# A '_rec' suffix adds a short *recovery* fine-tune on top of that mode's already
# trained adapter: RECOVERY_SAMPLES further samples, half of them carrying random
# detours the trace solves through. It continues from the existing adapter rather
# than training a second model, so it costs ~2% of a full run.
BUDGET_MODES = ('samples', 'compute', 'samples_rec', 'compute_rec')
SAMPLES_PER_COND = 40000        # 'samples' mode: traces per condition
SAMPLES_STD = 50000             # 'compute' mode: traces for the standard condition
PASSES = 3                      # how many times the budget is swept over the pool
RECOVERY_SAMPLES = 2000         # '_rec': extra samples of continued fine-tuning
AUGMENT_FRAC = 0.5              # of those, the share carrying detours
AUGMENT_P = 0.3                 # injection rate used when building them
# r=32 roughly doubles the adapter to ~2.4% of the model. The task now has 50k
# traces behind it, so a little more capacity is cheap insurance against the
# adapter -- rather than the data -- being the binding constraint.
MODEL = ModelConfig(lora_r=32, lora_alpha=64)
# batch_size x grad_accum is the effective batch (32). Sequences reach ~760
# tokens, so batch 8 is the safe setting on a 24 GB card -- an OOM part-way
# through a long adapter costs far more than the lost throughput. Raise to
# 16 x 2 if memory allows. `total_samples` is set per condition by `train_cfg`.
TRAIN = TrainConfig(lr=1e-4, batch_size=8, grad_accum=4)


def is_recovery(mode):
    """True for modes that add a recovery fine-tune on top of the base adapter."""
    return mode.endswith('_rec')


def data_cfg(mode):
    """The DataConfig for a mode. Only ``*_rec`` modes augment, and only during
    their short continuation; the problems are identical either way, so the
    dataset cache is shared with the base runs."""
    if is_recovery(mode):
        return replace(DATA, augment_frac=AUGMENT_FRAC, augment_p=AUGMENT_P)
    return DATA


def scored_ps_for(interval):
    """
    Injection probabilities the derivation, state and diagnostic passes score.

    `STD_ONLY_PS` is excluded. At p = 1.0 every operation the model emitted is
    replaced, which makes those metrics degenerate rather than informative:
    `derivation` and `ignored` are near-zero by construction, and `replaced`
    reduces to "did the trace have the right number of steps and the right answer",
    since substituting every slot leaves the canonical prefix. The accuracy figure
    still carries the point -- it is meaningful there.
    """
    return [x for x in inject_ps_for(interval) if x not in STD_ONLY_PS]


def inject_ps_for(interval):
    """
    Which injection probabilities are meaningful for a condition.

    The no-CoT condition emits no operation steps, so there is nothing to inject
    into and every p would decode identically; it is evaluated clean only. The
    probabilities in `STD_ONLY_PS` are restricted to the standard condition.
    """
    if interval == NO_COT:
        return [0.0]
    if interval is None:
        return list(INJECT_PS)
    return [x for x in INJECT_PS if x not in STD_ONLY_PS]


def train_cfg(interval, mode):
    """
    The TrainConfig for one condition under one budget mode.

    'samples' gives every condition `SAMPLES_PER_COND` traces. 'compute' gives
    every condition the same number of *tokens* -- the budget the standard
    condition uses at `SAMPLES_STD` traces -- which means proportionally fewer
    traces the longer that condition's traces are. `total_samples` lives in
    TrainConfig, so it flows into the adapter, eval, decode and diagnostic cache
    keys automatically and the two modes never collide.

    NO_COT is the exception, and is trained on the same number of traces as the
    standard condition in whichever mode is running. It is a reference point for
    standard rather than another condition being compared under the budget rule,
    so matching traces is what makes "the same data, no reasoning" the question it
    answers. Token-matching it would be actively misleading: its traces are a
    handful of tokens, so the budget would buy several times the 50k problem pool
    and the extra would be repeated passes rather than more data.
    """
    base = mode[:-4] if is_recovery(mode) else mode
    if interval == NO_COT:
        n = SAMPLES_PER_COND if base == 'samples' else SAMPLES_STD
    elif base == 'samples':
        n = SAMPLES_PER_COND
    else:
        budget = mean_tokens_per_example(None, DATA, MODEL, TRAIN) * SAMPLES_STD
        n = round(budget / mean_tokens_per_example(interval, DATA, MODEL, TRAIN))
    tc = replace(TRAIN, total_samples=int(n) * PASSES)
    return replace(tc, recovery_samples=RECOVERY_SAMPLES) if is_recovery(mode) else tc
# max_steps must exceed the longest k=1 trace (ops + state reports + answer)
# plus room to recover from injections; see EvalConfig.
EVAL = EvalConfig(step_max_tokens=64, batch_size=128)


# ----------------------------------------------------------------------------
# plotting style (mirrors the navigation figures)
# ----------------------------------------------------------------------------

CURVE_COLORS = ['#2a78d6', '#e87ba4', '#eda100', '#008300']
# injection probability is ordered and continuous, shown as a colorbar rather
# than legend entries. Clean (p=0) is the darkest: it is the reference curve and
# should read strongest. Colours are assigned by *rank* rather than by the value
# of p: the probabilities run 0, 0.05, 0.1, 0.2, 0.3, 0.5, so a linear scale
# would squeeze the first three into the bottom fifth of the ramp and leave them
# nearly identical. Even spacing by rank gives every pair the same contrast.
_INJECT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'inject', ['#12294a', '#2160a8', '#57a0e5'])


# p = 1.0 sits outside the ramp: it is a different kind of condition (no
# model-authored operation survives at all) and is drawn for standard alone, so it
# gets black rather than the lightest blue, which would read as one more step in
# an ordered series.
_FULL_INJECT_COLOR = '#111111'


def _inject_colors(ps):
    """One colour per injection probability, evenly spaced along the ramp."""
    ramp = [p for p in sorted(ps) if p < 1.0]
    n = max(len(ramp) - 1, 1)
    out = {p: _INJECT_CMAP(i / n) for i, p in enumerate(ramp)}
    if 1.0 in ps:
        out[1.0] = _FULL_INJECT_COLOR
    return out
LABEL_FS = 14
TICK_FS = 12
LEGEND_FS = 8


def _style_axis(ax):
    ax.tick_params(labelsize=TICK_FS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _save(fig, name, subdir=''):
    out = os.path.join(FIG_DIR, subdir) if subdir else FIG_DIR
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path + '.pdf', bbox_inches='tight')
    fig.savefig(path + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# running
# ----------------------------------------------------------------------------

def _interval_jobs(intervals, inject_ps, device, force, progress_pos=0,
                   diagnostics=False, mode='samples'):
    """
    Train the adapter for each interval on `device` and evaluate every injection
    probability with the model still loaded. Returns the result dicts.
    """
    out = []
    for interval in intervals:
        TR = train_cfg(interval, mode)
        inject_ps = [p for p in inject_ps_for(interval) if p in set(inject_ps)]
        todo = [p for p in inject_ps
                if force or cached_unit(interval, p, data_cfg(mode), MODEL, TR, EVAL) is None]
        for p in inject_ps:                                # collect the cache hits
            if p not in todo:
                r = cached_unit(interval, p, data_cfg(mode), MODEL, TR, EVAL)
                r['mode'] = mode
                out.append(r)
        # no operations means no plan to replay and no states to check
        need_diag = ([p for p in inject_ps if p > 0]
                     if diagnostics and interval != NO_COT else [])
        if not todo and not need_diag:
            print(f'[{mode}] [k={interval}] all cached', flush=True)
            continue
        # the model is loaded lazily: a condition whose decodes are all cached
        # never occupies the GPU at all
        adapter = train_adapter(interval, data_cfg(mode), MODEL, TR, device=device,
                                force=force, log=lambda m: print(f'[{mode}] {m}', flush=True),
                                progress_pos=progress_pos)
        _pack = {}

        def get_model(_a=adapter):
            if 'p' not in _pack:
                _pack['p'] = load_for_eval(_a, MODEL, TR, device=device)
            return _pack['p']

        for p in todo:
            res = run_unit(interval, p, data_cfg(mode), MODEL, TR, EVAL, device=device,
                           force=force, get_model=get_model,
                           log=lambda m: print(f'[{mode}] {m}', flush=True),
                           progress_pos=progress_pos)
            res['mode'] = mode
            out.append(res)
        if diagnostics and interval != NO_COT:
            from diagnostics import run_diagnostic, diag_is_cached
            need = [p for p in inject_ps if p > 0 and
                    (force or not diag_is_cached(interval, p, data_cfg(mode), MODEL, TR, EVAL))]
            for p in need:
                # diagnostics are analysis, not results: never let one fail the
                # sweep and throw away trained adapters
                try:
                    run_diagnostic(interval, p, data_cfg(mode), MODEL, TR, EVAL, device=device,
                                   force=force, get_model=get_model,
                                   log=lambda m: print(f'[{mode}] {m}', flush=True),
                                   progress_pos=progress_pos)
                except Exception as exc:
                    print(f'[k={interval} p={p}] diagnostic FAILED '
                          f'({type(exc).__name__}: {exc}); continuing', flush=True)
        _pack.clear()
        import torch
        torch.cuda.empty_cache()
    return out


_STOP = '__stop__'          # queue sentinel (None is a real interval value)


def _worker(queue, inject_ps, device, force, progress_pos, diagnostics=False,
            mode='samples'):
    """Pull intervals off the shared queue until it is drained."""
    out = []
    while True:
        interval = queue.get()
        if interval == _STOP:
            break
        print(f'[{device}] starting {mode} k={interval}', flush=True)
        out.extend(_interval_jobs([interval], inject_ps, device, force, progress_pos,
                                  diagnostics, mode))
    print(f'[{device}] done', flush=True)
    return out


def run_all(intervals, inject_ps, devices, force, diagnostics=False, mode='samples'):
    """
    Run every (interval, inject_p) unit across `devices`.

    Work is handed out through a shared queue rather than pre-assigned, so a GPU
    that finishes early immediately picks up the next interval instead of idling
    while another device works through a longer queue. Each interval stays a
    single task (train, then evaluate every injection probability with the model
    still resident) because reloading the base model per evaluation would cost
    more than the imbalance it saves.
    """
    # Build (and cache) the datasets here, in the parent. Every worker calls
    # get_problems, so without this all of them would generate the same 10k
    # problems concurrently -- minutes of duplicated SymPy work per GPU.
    t0 = time.time()
    n_tr = len(get_problems(DATA, 'train'))
    n_te = len(get_problems(DATA, 'test'))
    print(f'datasets ready: {n_tr} train / {n_te} test ({time.time() - t0:.0f}s)', flush=True)

    if len(devices) == 1:
        return _interval_jobs(intervals, inject_ps, devices[0], force,
                              diagnostics=diagnostics, mode=mode)

    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    queue = manager.Queue()
    for interval in intervals:
        queue.put(interval)
    for _ in devices:
        queue.put(_STOP)

    results = []
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as ex:
        futs = [ex.submit(_worker, queue, inject_ps, d, force, i, diagnostics, mode)
                for i, d in enumerate(devices)]
        for f in futs:
            results.extend(f.result())
    return results


def print_plan(intervals, inject_ps, devices, force, mode='samples'):
    """
    What the run will actually do: which adapters must be trained, which
    (k, p) evaluations are missing, and how the work lands on the GPUs. Printed
    before anything loads so the cost is visible up front.
    """
    to_train = [k for k in intervals
                if force or not adapter_is_cached(k, data_cfg(mode), MODEL, train_cfg(k, mode))]
    units = [(k, p) for k in intervals for p in inject_ps_for(k) if p in set(inject_ps)]
    todo = [(k, p) for k, p in units
            if force or cached_unit(k, p, data_cfg(mode), MODEL, train_cfg(k, mode), EVAL) is None]
    n_tr = {k: train_cfg(k, mode).total_samples // PASSES for k in intervals}
    n_dev, n_iv = len(devices), len(intervals)
    per_dev = max(1, -(-n_iv // n_dev))
    print('-' * 66)
    print(f'intervals      : {intervals}   (one LoRA each)')
    print(f'injection probs: {inject_ps}')
    print(f'budget mode    : {mode} ({_MODE_LABEL[mode]})')
    print('traces / cond  : ' + ' '.join(
        f'{_cond_label(k)}={n_tr[k]:,}' for k in intervals))
    print(f'passes         : {PASSES}   (test set {DATA.n_test} problems)')
    print(f'adapters       : {len(to_train)}/{len(intervals)} to train '
          f'-> {[_cond_label(k) for k in to_train]}')
    print(f'evaluations    : {len(todo)}/{len(units)} to run')
    print(f'devices        : {devices}')
    if n_iv < n_dev:
        print(f'  NOTE: {n_dev - n_iv} GPU(s) will idle -- there are only {n_iv} intervals. '
              f'Add intervals or pass fewer --devices.')
    else:
        print(f'  {n_iv} intervals over {n_dev} GPUs, pulled from a shared queue '
              f'(<={per_dev} per device)')
    print(f'  per-GPU train batch {TRAIN.batch_size} x {TRAIN.grad_accum} accum '
          f'= {TRAIN.batch_size * TRAIN.grad_accum} effective; eval batch {EVAL.batch_size}')
    print('rough cost     : ~10-20 min per adapter + ~2-5 min per evaluation,')
    print(f'                 so about {per_dev} x (train + {len(inject_ps)} evals) per device')
    print('-' * 66, flush=True)


def load_cached(intervals, inject_ps, mode='samples'):
    """Every cached unit for one budget mode, for --plot-only."""
    out = []
    for interval in intervals:
        TR = train_cfg(interval, mode)
        for p in inject_ps_for(interval):
            if p not in set(inject_ps):
                continue
            r = cached_unit(interval, p, data_cfg(mode), MODEL, TR, EVAL)
            if r is not None:
                r['mode'] = mode
                out.append(r)
    return out


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------

# Figures stop here. This now matches `max(INTERVALS)`, so the filters that use it
# are no-ops -- it is kept as a guard so that raising INTERVALS to explore a wider
# sweep does not silently widen every published figure along with it.
PLOT_MAX_K = 7


def _cond_label(interval):
    """Short name for a condition in tables and logs."""
    if interval is None:
        return 'std'
    return 'none' if interval == NO_COT else str(interval)


_NO_INJ_CACHE = {}


def no_injection_fraction(inject_p):
    """
    Expected share of traces that receive no injection at all.

    Injection is Bernoulli per operation step, so a trace of `n` operations escapes
    entirely with probability ``(1 - p) ** n``, and the share over the test set is
    ``E[(1 - p) ** n]``. Those traces are scored identically by every derivation
    variant -- there is nothing to drop or to replace -- so this level is the floor
    all three share, and a metric sitting on it is getting only the untouched
    problems right.

    Computed from the canonical operation counts rather than the model's emitted
    ones, which is close: predicted 0.257 / 0.128 / 0.029 at p = 0.2 / 0.3 / 0.5
    against 0.242 / 0.126 / 0.030 measured on the decodes.

    The expectation is over whole per-problem probabilities, i.e. outside the
    power. Putting it in the exponent instead -- ``(1 - p) ** mean_ops`` -- gives
    0.222 at p = 0.2 rather than 0.257, and is wrong: ``(1 - p) ** n`` is convex in
    `n`, so by Jensen the plug-in always understates the true share. The same fact
    read the other way is that short traces are over-represented among the ones
    that escape injection.
    """
    if inject_p not in _NO_INJ_CACHE:
        ns = [pr.n_ops for pr in get_problems(DATA, 'test')]
        _NO_INJ_CACHE[inject_p] = sum((1 - inject_p) ** n for n in ns) / max(len(ns), 1)
    return _NO_INJ_CACHE[inject_p]


def _xpos(interval, kmax):
    """x position of a condition; standard sits one slot right of the largest k."""
    return kmax + 1.6 if interval is None else interval


def fig_subdir(mode):
    """
    Where a mode's figures go: {with|without}_recovery/{samples|compute}_fixed.

    Recovery is the outer split because it is the bigger manipulation; the budget
    mode is the inner one. `_save` creates the nested path. Shared with
    `length_gen` so a figure lands in the same directory however it is invoked.
    """
    return os.path.join(
        'with_recovery' if is_recovery(mode) else 'without_recovery',
        f'{mode[:-4] if is_recovery(mode) else mode}_fixed')


_MODE_LABEL = {'samples': 'fixed training samples',
               'compute': 'fixed token budget',
               'samples_rec': 'fixed training samples + recovery fine-tune',
               'compute_rec': 'fixed token budget + recovery fine-tune'}


def plot_accuracy_vs_k(results, name='math_accuracy_vs_k', ylabel='Test accuracy',
                       subdir='', legend=True):
    """
    Held-out accuracy vs the state-emission interval k, one line per injection
    probability.

    Injection probability is ordered, so it is encoded as colour, spaced evenly by
    rank (see `_inject_colors`) rather than by value -- clean decoding is simply
    p = 0 at the dark end of the same ramp. The standard condition (state never
    emitted) is drawn as detached diamonds to the right of the k axis, since it is
    not a finite k; each series is carried across to its standard point with a
    dashed connector, so the trend is followable while the break in the x axis
    stays visible.
    """
    # the no-CoT condition is a floor, not a point on the k axis: it is drawn as
    # a horizontal reference so it can be read against every curve at once
    nocot = next((r['accuracy'] for r in results if r['interval'] == NO_COT), None)
    results = [r for r in results if r['interval'] != NO_COT]
    ks = sorted({r['interval'] for r in results
                 if r['interval'] is not None and r['interval'] <= PLOT_MAX_K})
    kmax = max(ks) if ks else 1
    ps = [p for p in sorted({r['inject_p'] for r in results})
          if p not in PLOT_EXCLUDE_PS]
    colors = _inject_colors(ps)
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    if nocot is not None:
        # the no-reasoning floor, labelled inline. x is in data units: 1.1 puts the
        # text clear of both the spine and the k = 1 markers.
        ax.axhline(nocot, color='#e31a1c', lw=1.1, ls=(0, (5, 3)), zorder=1)
        ax.text(1.1, nocot + 0.022, 'no reasoning', fontsize=LEGEND_FS - 1,
                color='#e31a1c', va='bottom')

    handles = []
    for p in ps:
        col = colors[p]
        rows = {r['interval']: r for r in results if r['inject_p'] == p}
        xs = [_xpos(k, kmax) for k in ks if k in rows]
        ys = [rows[k]['accuracy'] for k in ks if k in rows]
        if xs:
            ax.plot(xs, ys, marker='o', ms=4.2, lw=1.5, color=col, zorder=2)
        if None in rows:                                   # standard: detached point
            xstd, ystd = _xpos(None, kmax), rows[None]['accuracy']
            if xs:
                ax.plot([xs[-1], xstd], [ys[-1], ystd], ls=(0, (3, 2)), lw=1.2,
                        color=col, zorder=1)
            ax.plot([xstd], [ystd], marker='D', ms=5.5, color=col, ls='none', zorder=3)
        # a probability evaluated only at standard has no curve, so show its
        # marker alone rather than a line the figure never draws
        handles.append(Line2D([], [], color=col, marker='o', ms=4, lw=1.5,
                              label=rf'$p={p:g}$') if xs else
                       Line2D([], [], color=col, marker='D', ms=4.5, ls='none',
                              label=rf'$p={p:g}$'))

    # every k is plotted, but at this width twelve labels collide -- label every
    # other one (always keeping the first and last) plus the standard point
    shown = set(ks[::2]) | {ks[0], ks[-1]} if ks else set()
    ax.set_xticks(ks + [_xpos(None, kmax)])
    ax.set_xticklabels([str(k) if k in shown else '' for k in ks] + ['std'])
    ax.axvline(kmax + 0.8, color='0.85', lw=0.8, zorder=0)   # separates std
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Interval $k$', fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    _style_axis(ax)

    # two columns: six entries stacked vertically reach down into the curves,
    # whereas the strip above y~0.7 is empty for all but the smallest k
    if legend:
        leg = ax.legend(handles=handles, title='Perturbation Probability', fontsize=LEGEND_FS - 1,
                        title_fontsize=LEGEND_FS - 1, frameon=True, loc='upper right',
                        ncol=2, handlelength=1.2, handletextpad=0.4,
                        labelspacing=0.25, columnspacing=1.0, borderaxespad=0.2,
                        facecolor='white', edgecolor='0.8', framealpha=1.0,
                        borderpad=0.35)
        leg.get_frame().set_linewidth(0.7)
        leg.set_zorder(10)
        leg._legend_box.align = 'left'
    return _save(fig, name, subdir)


STATE_CACHE = os.path.join(CACHE, 'state')
ORDER_CACHE = os.path.join(CACHE, 'order')


def order_results(mode, force=False, pass_label=''):
    """
    On the CLEAN decodes, how closely do correct traces follow the canonical order?

    Derivation accuracy asks whether the written operations *reach* a solved form,
    not whether they are the ones the problem was generated from -- a different
    valid route passes identically, and alternative routes exist: the general
    solver picks a different first operation from the canonical plan about 62% of
    the time. So "sound" and "canonical" are genuinely different properties and
    nothing else here separates them.

    Restricted to traces that got the answer right, since the question is what a
    *successful* trace looks like. Reports, for each condition:

    exact       the operation list equals `problem.ops` outright
    prefix      the model's operations are a prefix of the canonical list, or the
                canonical list is a prefix of theirs -- right order, stopped early
                or ran on
    lcs         mean longest-common-subsequence ratio against the canonical list,
                as partial credit for keeping most of the order
    len_ratio   mean operations emitted / canonical operations

    A re-scoring of cached decodes; no GPU.
    """
    problems = get_problems(DATA, 'test')
    ks = list(INTERVALS)
    prefix_lbl = f'[{pass_label}] ' if pass_label else ''
    bar = tqdm(ks, desc=f'{prefix_lbl}{mode} order re-score', leave=False,
               dynamic_ncols=True, mininterval=2.0)
    out = []
    for k in bar:
        TR = train_cfg(k, mode)
        dec = _decode_path(k, 0.0, data_cfg(mode), MODEL, TR, EVAL)
        if not os.path.exists(dec):
            continue
        spec = dict(task='math-order', interval=k, mode=mode,
                    data=_data_spec(data_cfg(mode)), model=MODEL.__dict__,
                    train=_train_spec(TR), eval=EVAL.__dict__)
        path = os.path.join(ORDER_CACHE, _key(spec) + '.json')
        if os.path.exists(path) and not force:
            with open(path) as f:
                out.append(json.load(f))
            continue
        with open(dec) as f:
            records = json.load(f)['records']
        n = exact = pref = 0
        lcs_sum = len_sum = 0.0
        for prob, rec in zip(problems, records):
            ans = next((x['text'] for x in rec if x['kind'] == 'answer'), '')
            if not is_correct(prob, ans):          # correct traces only
                continue
            got = [x['text'] for x in rec if x['kind'] == 'op']
            want = list(prob.ops)
            n += 1
            exact += got == want
            m = min(len(got), len(want))
            pref += got[:m] == want[:m]
            lcs_sum += _lcs_len(got, want) / max(len(want), 1)
            len_sum += len(got) / max(len(want), 1)
        row = dict(interval=k, mode=mode, n_correct=n,
                   exact=(exact / n) if n else None,
                   prefix=(pref / n) if n else None,
                   lcs=(lcs_sum / n) if n else None,
                   len_ratio=(len_sum / n) if n else None)
        os.makedirs(ORDER_CACHE, exist_ok=True)
        tmp = path + f'.tmp{os.getpid()}'
        with open(tmp, 'w') as f:
            json.dump(row, f)
        os.replace(tmp, path)
        out.append(row)
    bar.close()
    return out


def _lcs_len(a, b):
    """Length of the longest common subsequence of two operation lists."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def print_order(order, mode=None):
    """Canonical-order adherence among correct traces on the clean decodes."""
    rows = [r for r in order if r.get('exact') is not None]
    if not rows:
        return
    head = 'canonical-order adherence, CLEAN decodes, correct traces only'
    if mode:
        head += f'   [{_MODE_LABEL.get(mode, mode)}]'
    print('\n' + head)
    print('  exact     the operation list equals the canonical one')
    print('  prefix    same order as far as the shorter of the two runs')
    print('  lcs       mean longest-common-subsequence ratio (partial credit)')
    print('  len       mean operations emitted / canonical operations')
    print()
    print('  cond   n_correct    exact   prefix      lcs      len')
    for r in sorted(rows, key=lambda r: (99 if r['interval'] is None else r['interval'])):
        print(f'  {_cond_label(r["interval"]):>4}   {r["n_correct"]:>9}   '
              f'{r["exact"]:6.3f}   {r["prefix"]:6.3f}   {r["lcs"]:6.3f}   '
              f'{r["len_ratio"]:6.3f}')



def state_results(mode, force=False, pass_label=''):
    """
    Anchored and cumulative state accuracy per condition and injection probability.

    This is a first-class metric rather than part of `--diagnostics`, and it is
    cached under its own key. Keeping it separate matters: folding these fields
    into the D2 records would have made every diagnostic already on disk incomplete,
    forcing a rebuild of work that has not changed. Here nothing else is touched.

    Like the derivation variants it re-scores cached decodes, so it needs no GPU.
    Only k >= 1 has intermediate state reports; standard enters the figure through
    its clean test accuracy, supplied by the caller.
    """
    from diagnostics import state_tracking

    problems = get_problems(DATA, 'test')
    units = [(k, pp) for k in INTERVALS if k is not None
             for pp in scored_ps_for(k)]
    prefix = f'[{pass_label}] ' if pass_label else ''
    bar = tqdm(units, desc=f'{prefix}{mode} state re-score', leave=False,
               dynamic_ncols=True, mininterval=2.0)
    out = []
    for k, pp in bar:
        TR = train_cfg(k, mode)
        dec = _decode_path(k, pp, data_cfg(mode), MODEL, TR, EVAL)
        if not os.path.exists(dec):
            continue
        spec = dict(task='math-state', interval=k, mode=mode,
                    data=_data_spec(data_cfg(mode)), model=MODEL.__dict__,
                    train=_train_spec(TR), eval=EVAL.__dict__)
        # keep the clean entries already on disk valid: they were keyed without
        # an injection probability, which is what p = 0 meant
        if pp > 0:
            spec['inject_p'] = pp
        path = os.path.join(STATE_CACHE, _key(spec) + '.json')
        if os.path.exists(path) and not force:
            with open(path) as f:
                r = json.load(f)
            if 'cum_only' in r:
                out.append(r)
                continue
        with open(dec) as f:
            records = json.load(f)['records']
        tot = ok = cum = cum_ok = skip = 0
        cum_only = anch_only = 0
        for prob, rec in zip(problems, records):
            try:
                a, b_, _, _, sk, c, d, co, ao = state_tracking(prob, rec)
            except Exception:
                continue
            tot += a; ok += b_; skip += sk; cum += c; cum_ok += d
            cum_only += co; anch_only += ao
        row = dict(interval=k, inject_p=pp, mode=mode,
                   anchored=(ok / tot) if tot else None,
                   cumulative=(cum_ok / cum) if cum else None,
                   n_reports=tot, n_unscorable=skip,
                   # discordant pairs: reports one scoring accepted and the other
                   # did not. McNemar reads these and nothing else.
                   cum_only=cum_only, anch_only=anch_only)
        os.makedirs(STATE_CACHE, exist_ok=True)
        tmp = path + f'.tmp{os.getpid()}'
        with open(tmp, 'w') as f:
            json.dump(row, f)
        os.replace(tmp, path)
        out.append(row)
    bar.close()
    return out


def plot_state_tracking_vs_k(state, name='math_state_tracking_vs_k', subdir=''):
    """
    Raw state-report accuracy vs k, scored two ways, at every injection probability.

    Both series apply the model's own operations, so both judge state computation
    rather than choice of operation; they differ only in what counts as truth.
    `anchored` (solid) re-adopts the model's claim after every report, `cumulative`
    (dashed) never does. Their vertical gap at a given k is error propagation, and
    its sign says which way: cumulative above anchored means the model returns to
    the true state after a slip (anchoring double-charges a recovery), cumulative
    below means it drifts while staying locally self-consistent.

    No root is taken. A report at interval k spans k operations, so comparing across
    k is confounded by span -- the comparison this figure is for is vertical, within
    a single k, where both series cover identical operations and the confound
    cancels exactly. The per-operation conversion has its own figure.

    Above k = 6 the two series coincide exactly, and that is arithmetic rather than
    a result: they can only disagree on a report that has an earlier one before it,
    and from k = 7 a second report would need 2k <= 12 operations, which no problem
    has. Every condition is plotted so the convergence is visible, but a flat
    overlap on the right says only that traces rarely report state twice there.

    The standard condition does not appear. It emits no intermediate state, so it
    has no state accuracy to plot; its answer accuracy is a different kind of
    quantity -- one endpoint check against a per-report average -- and showing the
    two side by side invited reading a comparison the figure cannot support.

    It also carried a prediction: standard's accuracy if its answer were a single
    state prediction over the whole trace, ``E[state_acc(k = n_ops)]``. That reading
    presupposes the interval-k models are themselves reasoning step by step, which
    the recovery gradient contradicts -- 0% of discordant reports are recoveries at
    k = 1 against 92% at k = 5, so by the k the prediction reads (mean n_ops = 6.8)
    the reference models already show the signature the test is meant to detect. The
    prediction is not sound and is no longer drawn.
    """
    rows = {(d['interval'], d.get('inject_p') or 0.0): d for d in state
            if d.get('anchored') is not None and d.get('cumulative') is not None
            and d['interval'] <= PLOT_MAX_K}
    if not rows:
        return None
    ks = sorted({k for k, _ in rows})
    ps = [pp for pp in sorted({pp for _, pp in rows}) if pp not in PLOT_EXCLUDE_PS]
    colors = _inject_colors(ps)

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    handles = []
    for pp in ps:
        col = colors[pp]
        xs = [k for k in ks if (k, pp) in rows]
        if not xs:
            continue
        ax.plot(xs, [rows[(k, pp)]['anchored'] for k in xs], marker='o', ms=4.0,
                lw=1.5, color=col, zorder=2)
        ax.plot(xs, [rows[(k, pp)]['cumulative'] for k in xs], ls=(0, (3, 2)),
                marker='D', ms=3.6, lw=1.3, color=col, zorder=2)
        handles.append(Line2D([], [], color=col, marker='o', ms=4, lw=1.5,
                              label=rf'$p={pp:g}$'))
    shown = set(ks[::2]) | {ks[0], ks[-1]}
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) if k in shown else '' for k in ks])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Interval $k$', fontsize=LABEL_FS)
    ax.set_ylabel('State accuracy', fontsize=LABEL_FS)
    _style_axis(ax)
    handles += [Line2D([], [], color='0.35', lw=1.5, marker='o', ms=4,
                       label='anchored'),
                Line2D([], [], color='0.35', lw=1.3, ls=(0, (3, 2)), marker='D',
                       ms=3.6, label='cumulative')]
    leg = ax.legend(handles=handles, fontsize=LEGEND_FS - 1, frameon=True,
                    loc='upper right', ncol=2, handlelength=1.4, handletextpad=0.5,
                    labelspacing=0.25, columnspacing=1.0, borderaxespad=0.2,
                    facecolor='white', edgecolor='0.8', framealpha=1.0,
                    borderpad=0.35)
    leg.get_frame().set_linewidth(0.7)
    leg.set_zorder(10)
    return _save(fig, name, subdir)


def plot_state_discordance_vs_k(state, name='math_state_discordance_vs_k', subdir='',
                                ps_shown=None):
    """
    Where the anchored and cumulative scorings disagree, by how much and which way.

    Each state report is scored twice (see `plot_state_tracking_vs_k`); the reports
    that both accept or both reject carry no information about error propagation, so
    this figure plots only the discordant ones, split by direction:

    solid (circles)   `anch_only`: reports the anchored scoring accepted and the
                      cumulative one rejected -- the model stayed locally consistent
                      with its own last claim while off the true trajectory (drift).
    dashed (diamonds) `cum_only`: reports the cumulative scoring accepted and the
                      anchored one rejected -- the model was back on the true state
                      but disagreed with the wrong claim it had anchored to (a
                      recovery that anchoring double-charges).

    Each direction is reported as a fraction of the *discordant* reports
    (`anch_only + cum_only`), so the solid and dashed values at a given k sum to 1:
    the figure shows how the disagreements split into drift (anchored-only) and
    recovery (cumulative-only). A k, p with no discordant report has no defined
    split and is simply not drawn.

    Params
    ------
    ps_shown : sequence[float] | None
        Injection probabilities to draw; None (the default) draws every p present.
        Colours are taken from the full injection ramp either way, so a p keeps the
        same colour it has in the state accuracy figure whichever subset is drawn.
    """
    rows = {(d['interval'], d.get('inject_p') or 0.0): d for d in state
            if d.get('cum_only') is not None and d.get('anch_only') is not None
            and 1 <= d['interval'] <= 5}
    if not rows:
        return None

    def frac(d, field):
        """`field` as a share of the discordant total, or None if there is none."""
        denom = d['anch_only'] + d['cum_only']
        return d[field] / denom if denom else None

    ks = sorted({k for k, _ in rows})
    # colour from the full injection grid so each p keeps the colour it has in the
    # state accuracy figure, whether we draw all of them or only a subset.
    all_ps = [pp for pp in sorted({pp for _, pp in rows}) if pp not in PLOT_EXCLUDE_PS]
    colors = _inject_colors(all_ps)
    ps = all_ps if ps_shown is None else [pp for pp in ps_shown if pp in all_ps]

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    for pp in ps:
        col = colors[pp]
        # only k with at least one discordant report have a defined split
        xs = [k for k in ks if (k, pp) in rows
              and frac(rows[(k, pp)], 'anch_only') is not None]
        if not xs:
            continue
        ax.plot(xs, [frac(rows[(k, pp)], 'anch_only') for k in xs], marker='o',
                ms=4.0, lw=1.5, color=col, zorder=2)
        ax.plot(xs, [frac(rows[(k, pp)], 'cum_only') for k in xs], ls=(0, (3, 2)),
                marker='D', ms=3.6, lw=1.3, color=col, zorder=2)
    ax.set_xticks(ks)
    ax.set_xlim(ks[0] - 0.3, ks[-1] + 0.3)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Interval $k$', fontsize=LABEL_FS)
    ax.set_ylabel('Drift/Recover Fraction', fontsize=LABEL_FS)
    _style_axis(ax)
    # The colour -> p mapping is carried by a shared legend across the family of
    # figures that use the same injection ramp, so this legend lists only the two
    # line styles -- keeping it small enough to sit inside without covering curves.
    handles = [Line2D([], [], color='0.35', lw=1.5, marker='o', ms=4,
                      label='drift (anchored ✓; cumulative ✗)'),
               Line2D([], [], color='0.35', lw=1.3, ls=(0, (3, 2)), marker='D',
                      ms=3.6, label='recover (anchored ✗; cumulative ✓)')]
    leg = ax.legend(handles=handles, fontsize=LEGEND_FS - 1, frameon=True,
                    loc='upper right', bbox_to_anchor=(1.0, 1.08),
                    handlelength=1.4, handletextpad=0.5, labelspacing=0.25,
                    borderaxespad=0.0, facecolor='white', edgecolor='0.8',
                    framealpha=1.0, borderpad=0.35)
    leg.get_frame().set_linewidth(0.7)
    leg.set_zorder(10)
    return _save(fig, name, subdir)


def mcnemar(cum_only, anch_only):
    """
    Exact two-sided McNemar test on the anchored/cumulative discordance.

    The two scorings judge the *same* reports, so this is a paired comparison and
    a two-proportion test would be wrong -- it would assume independence and
    overstate the standard error. Only the reports where they disagree carry
    information: under the null "a report is equally likely to be accepted by one
    scoring alone as by the other", the count that only cumulative accepted is
    Binomial(n_discordant, 1/2).

    The exact binomial tail is used rather than the chi-square approximation,
    which is unreliable once the discordant total falls below ~25 -- and it does,
    at large k where few traces emit a second report at all.

    Returns (n_discordant, p_value). p is None when nothing disagreed.
    """
    from math import comb
    n = cum_only + anch_only
    if n == 0:
        return 0, None
    lo = min(cum_only, anch_only)
    tail = sum(comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return n, min(2 * tail, 1.0)


def print_state_tracking(state, results=None):
    """The two state-tracking scores side by side, with the sign of their gap."""
    rows = {d['interval']: d for d in state if d.get('cumulative') is not None
            and not d.get('inject_p')}
    if not rows:
        return
    print('\nstate tracking on the CLEAN decode (no injection), raw per report')
    print('  anchored   : truth re-adopts the model\'s claim after every report')
    print('  cumulative : truth is the consequence of the operations written so far')
    print('  cum > anch => the model recovers after a slip (anchoring charges it twice)')
    print('  cum < anch => the model drifts while staying locally self-consistent')
    print('  the figure carries every injection probability; this table is the clean')
    print('  column, where the sign is a claim about the models rather than the noise')
    print()
    print('  (from k = 7 no trace emits a second report -- 2k > 12 operations -- so')
    print('   the two scorings are the same computation there and the gap is 0)')
    print()
    print('  cum-only / anch-only  reports ONE scoring accepted and the other did not;')
    print('  McNemar tests whether those two counts differ (exact two-sided binomial)')
    print()
    print('  cond   anchored   cumulative      gap   cum-only  anch-only'
          '   n_disc        p   reading')
    for k in sorted(rows):
        r = rows[k]
        a, c = r['anchored'], r['cumulative']
        co, ao = r.get('cum_only'), r.get('anch_only')
        tag = 'recovers' if c > a else ('drifts' if c < a else '-')
        if co is None:
            print(f'  {k:>4}     {a:.3f}        {c:.3f}   {c - a:+.3f}'
                  f'         -          -        -        -   {tag}')
            continue
        n, pv = mcnemar(co, ao)
        star = '' if pv is None else (
            '***' if pv < 1e-3 else '**' if pv < 0.01 else '*' if pv < 0.05 else 'ns')
        ps = '     -   ' if pv is None else (f'{pv:9.2e}' if pv < 1e-4 else f'{pv:9.4f}')
        print(f'  {k:>4}     {a:.3f}        {c:.3f}   {c - a:+.3f}   {co:>8}  {ao:>9}'
              f'   {n:>6}  {ps} {star:<3} {tag}')
    print()
    print('  * p<0.05   ** p<0.01   *** p<0.001   ns = not significant')
    std = next((r['accuracy'] for r in (results or [])
                if r['interval'] is None and r['inject_p'] == 0.0), None)
    if std is not None:
        print(f'  {"std":>4}     {std:.3f}        {std:.3f}    0.000   '
              'final state only (test accuracy)')


DERIV_CACHE = os.path.join(CACHE, 'derivation')


def derivation_results(mode, force=False, variant='full', pass_label=''):
    """
    Derivation accuracy for every (k, p) whose decode is cached.

    A trace counts only if the operations it actually wrote, replayed symbolically,
    reduce the equation to a solved form matching the answer
    (`generate_data.derivation_correct`) -- strictly harder than getting the final
    answer right. Only operation steps are scored; the model's state reports are
    ignored, so the metric is defined identically for every condition including
    standard.

    Three variants, selected by `variant`:

    ``full``      score the sequence exactly as decoded.
    ``ignored``   drop the injected steps before replaying.
    ``replaced``  put the canonical operation back at each injected slot -- an
                  injection at index i becomes ``problem.ops[i]``
                  (`generate_data.derivation_correct_replaced`). This closes the
                  hole `ignored` leaves: injection *replaces* the step the model
                  was about to emit, so deleting it costs the trace one of its own
                  operations and it can fail for want of steps rather than for want
                  of a plan. A trace longer than the canonical solution is counted
                  incorrect, so the substitutions cannot carry a rambling trace to
                  the answer on their own.

    With ``ignored`` the injected steps are *dropped* before replaying, so
    only the model's own operations are scored. That asks directly whether a
    condition is simply skipping the perturbation: if the model ignored the
    injection and carried on with its original plan, its own steps still form a
    complete derivation and this scores high; if it genuinely re-planned around the
    detour, its own steps no longer reach the answer without the injected ones and
    this scores low. At p = 0 the two coincide, which is a built-in sanity check.

    Every decode record carries an ``injected`` flag, so both variants are a
    re-scoring of cached traces -- no decoding and no GPU.

    `pass_label` is cosmetic: the run makes one pass per (mode, variant), and
    labelling them ``[3/8]`` makes clear that the bar restarting is the next pass
    rather than the same work repeating.
    """
    problems = get_problems(DATA, 'test')
    units = [(k, p) for k in INTERVALS for p in scored_ps_for(k)]
    label = {'full': 'derivation', 'ignored': 'ignored',
             'replaced': 'replaced'}[variant]
    prefix = f'[{pass_label}] ' if pass_label else ''
    bar = tqdm(units, desc=f'{prefix}{mode} {label} re-score', leave=False,
               dynamic_ncols=True, mininterval=2.0)
    out = []
    for k, p in bar:
        if True:
            TR = train_cfg(k, mode)
            dec = _decode_path(k, p, data_cfg(mode), MODEL, TR, EVAL)
            if not os.path.exists(dec):
                continue
            spec = dict(task='math-deriv', interval=k, inject_p=p, mode=mode,
                        data=_data_spec(data_cfg(mode)), model=MODEL.__dict__,
                        train=_train_spec(TR), eval=EVAL.__dict__)
            # spelled so that the keys already on disk keep their meaning:
            # 'full' adds nothing, 'ignored' keeps the original flag name
            if variant == 'ignored':
                spec['ignore_injected'] = True
            elif variant != 'full':
                spec['variant'] = variant
            path = os.path.join(DERIV_CACHE, _key(spec) + '.json')
            if os.path.exists(path) and not force:
                with open(path) as f:
                    out.append(json.load(f))
                continue
            with open(dec) as f:
                records = json.load(f)['records']
            n_ok = 0
            for prob, rec in zip(problems, records):
                steps = [x for x in rec if x['kind'] == 'op']
                ans_step = next((x['text'] for x in rec if x['kind'] == 'answer'), '')
                claimed = parse_answers(ans_step)
                if variant == 'replaced':
                    ok = derivation_correct_replaced(
                        prob, [x['text'] for x in steps],
                        [x['injected'] for x in steps], claimed)
                else:
                    ops = [x['text'] for x in steps
                           if not (variant == 'ignored' and x['injected'])]
                    ok = derivation_correct(prob, ops, claimed)
                n_ok += bool(ok)
            row = dict(interval=k, inject_p=p, mode=mode, variant=variant,
                       accuracy=n_ok / max(len(problems), 1), n_eval=len(problems))
            os.makedirs(DERIV_CACHE, exist_ok=True)
            tmp = path + f'.tmp{os.getpid()}'
            with open(tmp, 'w') as f:
                json.dump(row, f)
            os.replace(tmp, path)
            out.append(row)
    bar.close()
    return out


def print_summary(results, mode=None, label='accuracy'):
    """Metric table: rows are conditions, columns injection probabilities."""
    nocot = next((r for r in results if r['interval'] == NO_COT), None)
    results = [r for r in results if r['interval'] != NO_COT]
    ks = sorted({r['interval'] for r in results if r['interval'] is not None})
    ps = sorted({r['inject_p'] for r in results})
    order = ks + ([None] if any(r['interval'] is None for r in results) else [])
    idx = {(r['interval'], r['inject_p']): r for r in results}
    head = f'{label} (rows: state-emission interval, cols: injection probability)'
    if mode:
        head += f'   [{_MODE_LABEL.get(mode, mode)}]'
    print('\n' + head)
    print('  cond  ' + '   n_train'
          + '  '.join((f'p={p:g}').rjust(8) for p in ps))
    for k in order:
        cells = []
        for p in ps:
            r = idx.get((k, p))
            cells.append((f'{r["accuracy"]:.3f}' if r else '-').rjust(8))
        n = train_cfg(k, mode).total_samples // PASSES if mode else 0
        ncol = f'{n:>8,}  ' if mode else ''
        print(f'  {_cond_label(k):<4}  {ncol}' + '  '.join(cells))
    if nocot is not None:
        n = train_cfg(NO_COT, mode).total_samples // PASSES if mode else 0
        print(f'  {"none":<4}  ' + (f'{n:>8,}  ' if mode else '')
              + f'{nocot["accuracy"]:.3f}'.rjust(8)
              + '   <- no reasoning emitted; injection does not apply')


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def _default_devices():
    try:
        import torch
        if torch.cuda.is_available():
            return [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            return ['mps']
    except Exception:
        pass
    return ['cpu']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--devices', default=None,
                    help='comma-separated torch devices, e.g. cuda:0,cuda:1,...')
    ap.add_argument('--force', action='store_true', help='retrain and re-evaluate')
    ap.add_argument('--plot-only', action='store_true',
                    help='build the figure from cached results only')
    ap.add_argument('--mode', choices=list(BUDGET_MODES), default=None,
                    help='run only one budget mode (default: both)')
    ap.add_argument('--diagnostics', action='store_true',
                    help='also run D1 (plan replay) and D2 (state tracking) and '
                         'print them; reuses the cached adapters')
    ap.add_argument('--no-length-gen', action='store_true',
                    help='skip the length-generalisation pass, which decodes '
                         'unseen problem lengths and so costs GPU time')
    args = ap.parse_args()

    devices = args.devices.split(',') if args.devices else _default_devices()
    modes = [args.mode] if args.mode else list(BUDGET_MODES)

    n_passes = 5 * len(modes)   # derivation, ignored, replaced, state, order
    for mi, mode in enumerate(modes):
        banner = f'  BUDGET MODE: {mode}  ({_MODE_LABEL[mode]})  '
        print('\n' + '=' * len(banner)); print(banner); print('=' * len(banner), flush=True)

        if args.plot_only:
            results = load_cached(CONDITIONS, INJECT_PS, mode)
            if not results:
                print(f'no cached results for mode={mode}; skipping')
                continue
        else:
            print_plan(CONDITIONS, INJECT_PS, devices, args.force, mode)
            t0 = time.time()
            results = run_all(CONDITIONS, INJECT_PS, devices, args.force,
                              diagnostics=args.diagnostics, mode=mode)
            print(f'\n[{mode}] wall clock: {(time.time() - t0) / 60:.1f} min')

        results = [r for r in results if r is not None]
        sub = fig_subdir(mode)
        path = plot_accuracy_vs_k(results, name=f'math_accuracy_vs_k_{mode}', subdir=sub)
        print(f'\nwrote {path}.pdf/.png')
        print_summary(results, mode)

        # re-score the same cached decodes on whether the *steps* were right
        deriv = derivation_results(mode, force=args.force,
                                   pass_label=f'{5 * mi + 1}/{n_passes}')
        if deriv:
            dpath = plot_accuracy_vs_k(deriv, name=f'math_derivation_vs_k_{mode}',
                                       ylabel='Derivation accuracy', subdir=sub,
                                       legend=False)
            print(f'wrote {dpath}.pdf/.png')
            print_summary(deriv, mode, label='derivation accuracy (step sequence only)')

        # the same score with the injected steps dropped: high means the model's
        # own plan was complete on its own, i.e. it ignored the perturbation
        ignored = derivation_results(mode, force=args.force, variant='ignored',
                                     pass_label=f'{5 * mi + 2}/{n_passes}')
        if ignored:
            ipath = plot_accuracy_vs_k(ignored, name=f'math_ignored_vs_k_{mode}',
                                       ylabel='Ignored accuracy', subdir=sub)
            print(f'wrote {ipath}.pdf/.png')
            print_summary(ignored, mode,
                          label='ignored accuracy (injected steps dropped)')

        # the same score with each injected step *repaired* rather than dropped,
        # which is what `ignored` cannot do: it asks whether the model's own
        # operations are right once the perturbation is neutralised for free
        replaced = derivation_results(mode, force=args.force, variant='replaced',
                                      pass_label=f'{5 * mi + 3}/{n_passes}')
        if replaced:
            rpath = plot_accuracy_vs_k(replaced, name=f'math_replaced_vs_k_{mode}',
                                       ylabel='Replaced accuracy', subdir=sub)
            print(f'wrote {rpath}.pdf/.png')
            print_summary(replaced, mode,
                          label='replaced accuracy (injected steps corrected)')

        # state tracking is a metric, not a diagnostic: it re-scores the clean
        # decodes like the derivation variants do, so it runs every time
        state = state_results(mode, force=args.force,
                              pass_label=f'{5 * mi + 4}/{n_passes}')
        if state:
            tpath = plot_state_tracking_vs_k(
                state, name=f'math_state_tracking_vs_k_{mode}', subdir=sub)
            if tpath:
                print(f'wrote {tpath}.pdf/.png')
            dpath = plot_state_discordance_vs_k(
                state, name=f'math_state_discordance_vs_k_{mode}', subdir=sub,
                ps_shown=None)          # all p (p=0.5 dropped via PLOT_EXCLUDE_PS)
            if dpath:
                print(f'wrote {dpath}.pdf/.png')
            print_state_tracking(state, results)

        # does a correct trace follow the canonical operation order, or merely
        # reach a valid answer by some other route?
        order = order_results(mode, force=args.force,
                              pass_label=f'{5 * mi + 5}/{n_passes}')
        print_order(order, mode)

        # Generalisation past the training lengths. This is the one figure that
        # decodes rather than re-scoring what is cached, so it comes last and can
        # be skipped; it reuses the adapters the sweep just trained.
        if not args.no_length_gen:
            import length_gen
            lrows = length_gen.length_results(
                mode=mode, devices=devices, force=args.force,
                decode=not args.plot_only)
            if lrows:
                length_gen.print_length_gen(lrows, mode)
                length_gen.plot_length_gen(lrows, mode, subdir=sub)
            else:
                print(f'[{mode}] no length-generalisation results '
                      f'({"nothing cached" if args.plot_only else "decode failed"})')

        if args.diagnostics or args.plot_only:
            from diagnostics import run_diagnostic, print_diagnostics, _diag_path
            # Every unit is re-read here for reporting. Entries written before a
            # field existed are recomputed rather than the cache key being bumped,
            # so this can be minutes of symbolic replay with no GPU involved --
            # hence a bar, since a silent stall looks like a hang.
            units = [(k, p) for k in INTERVALS for p in scored_ps_for(k)
                     if p > 0 and os.path.exists(
                         _diag_path(k, p, data_cfg(mode), MODEL, train_cfg(k, mode), EVAL))]
            diag = []
            for k, p in tqdm(units, desc=f'{mode} diagnostics', leave=False,
                             dynamic_ncols=True, mininterval=2.0):
                diag.append(run_diagnostic(k, p, data_cfg(mode), MODEL,
                                           train_cfg(k, mode), EVAL))
            if diag:
                print_diagnostics(diag)
                for d in diag:
                    d['mode'] = mode


    # refresh the worked-examples document alongside the figures
    import make_examples
    make_examples.main()


if __name__ == '__main__':
    main()
