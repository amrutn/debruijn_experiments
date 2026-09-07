"""
Run the ExploreToM state-tracking experiment: does fine-tuning on reasoning
that keeps the story's state in the local context (a restated step plus the
exact world-and-belief state after it) beat fine-tuning on a question-specific
chain of the decisive steps, or on the answers alone?

Models scored on the ExploreToM test rows (`exploretom_data`):

base            Qwen2.5-1.5B-Instruct as released, no fine-tuning, under the
                same prompt. ``--base-model Qwen/Qwen2.5-1.5B`` runs the
                pretrained model instead (`train_eval.BASE_MODELS`).
direct          a LoRA trained on the answer field alone of the training rows.
chain           a LoRA trained on the *same* rows with the steps at which the
                answer to the question changes, each with the answer after it.
state_tracking  a LoRA trained on the same rows with the story restated and
                the *question's slice* of the state -- the people asked about,
                the object or topic -- after every step (``--focus-stride``),
                the asked beliefs written on every line whether or not they
                depart from the truth (``--focus-beliefs``). The local,
                De Bruijn-structured trace: each state line is a function of
                the previous state and the steps since. Shown as
                "state-tracking".
narration       the length-matched control for state_tracking: the same story
                restated, but each step followed by a one-line ``Note:`` of
                that step's *local* event (what happened, who was present),
                not a running state. The final note is not sufficient, so the
                answer needs global integration across the notes -- the same
                length and restating as state_tracking, without the De Bruijn
                structure. If state_tracking beats it, the structure (not the
                length) is what helps.
distill         optional (``--conditions ...,distill``): a LoRA trained on a
                teacher's own reasoning about the training questions, kept
                where the teacher was right (`traces.py`; Qwen3-32B run
                locally by default, built by this script if the file is short).

The chain, state_tracking and narration targets are computed from the
replayed story, so the exact conditions are trained on identical rows with
identical labels and differ in the format of the reasoning only. Training is LoRA r=32 on every
projection, lr 1e-4 cosine, effective batch 32, over four passes of the rows,
one seed. While an adapter trains it is scored twenty times on the test set
(`TrainConfig.curve_evals`), which gives the learning curve.

Datasets
--------
By default the whole experiment is run twice and a separate set of figures is
written for each: on the larger locally-generated stories (`generated`, the
"positive" set that separates the formats) and on the released ExploreToM
sample (`sample`, the near-saturated "negative" baseline where the good formats
tie). Pass `--dataset generated` or `--dataset sample` to run just one.

Figures (pdf+png in figures/)
-----------------------------
exploretom_accuracy_<model>[_<teacher>][_gen-p<N>m<N>r<N>]
                          Test accuracy per condition as a bar, with the
                          binomial SEM over the test rows as an error bar
                          (the SEM over seeds when there are several) and the
                          untuned model's accuracy as a dashed line.
exploretom_curve_<model>[_<teacher>][_gen-p<N>m<N>r<N>]
                          Test accuracy against training samples, one line
                          per fine-tuned condition starting from the untuned
                          model at zero samples, with the untuned model's
                          accuracy as a dashed horizontal line.
<model> is the base model, e.g. 'qwen2.5-1.5b-instruct'; <teacher> is the
trace teacher's tag and appears only when the distill condition is run; the
`_gen-...` suffix (people/moves/rooms) marks the generated set, so the two
datasets' figures never overwrite each other.

Scheduling and robustness
-------------------------
One (condition, seed) is one unit of work: train the adapter, then decode the
test set with the model still resident. Units are queued in `CONDITIONS`
order; on several GPUs the queue is shared and a GPU that finishes early
takes the next unit. Everything is cached (`train_eval`): the row sets, the
adapters, the decodes and the scores, and within a unit the training
checkpoints and every finished decode batch, so an interrupted run resumes
within minutes of where it stopped. A unit that fails with an error the
decoder cannot retry is reported and skipped rather than ending the run: the
figures and the table are built from whatever finished, and the exit status
says whether anything failed.

Usage
-----
    python run_experiments.py --devices cuda:0                  # BOTH datasets, one GPU
    python run_experiments.py                                   # auto devices, both datasets
    python run_experiments.py --dataset sample                  # only the released sample
    python run_experiments.py --dataset generated --gen-people 6 --gen-moves 12 --gen-stories 500
    python run_experiments.py --conditions base,direct,chain,state_tracking,distill
    python run_experiments.py --plot-only                       # figures from cache (both sets)
    python run_experiments.py --n-train 16 --n-test 12          # smoke test
"""

import os
# Reduce CUDA fragmentation before torch is imported (train_eval imports it):
# batched decoding re-allocates a growing KV cache every batch, and on a shared
# GPU the reserved-but-unallocated slack is what tips a long batch into OOM.
# `setdefault` so an explicit environment value still wins. Spawned workers
# re-import this module, so they inherit the setting before their own torch load.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import gc
import sys
import time
import argparse
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from exploretom_data import (
    DataConfig, get_problems, train_pool, DATASETS, FOCUS_STRIDE, FOCUS_BELIEFS, BELIEF_MODES,
)
from traces import (
    TraceConfig, TEACHER, coverage, trace_summary, teacher_tag, make_teacher, build,
)
from train_eval import (
    ModelConfig, TrainConfig, EvalConfig, CONDITIONS, DEFAULT_CONDITIONS, BASE_MODELS,
    run_unit, cached_unit, train_adapter, load_for_eval, adapter_is_cached,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')


# ----------------------------------------------------------------------------
# the experiment
# ----------------------------------------------------------------------------

# 1,500 training questions from 375 stories and 200 test questions from 50
# other stories (at most 4 questions per story), drawn once with seed 0.
DATA = DataConfig(n_train=1500, n_test=200, q_per_story=4, seed=0)

# The math task's adapter and optimiser settings: r=32 (~2.4% of the model);
# batch 8 x 4 accumulation is the effective batch of 32 -- kept as 8 x 4 on
# larger GPUs too, so the loss is averaged exactly as in the math task.
MODEL = ModelConfig()
TRAIN = TrainConfig(lr=1e-4, batch_size=8, grad_accum=4)
PASSES = 4                      # sweeps over the training rows per adapter (epochs)

# Greedy, batch 64: the 1.5B model's KV cache is small (2 KV heads, ~28 KB per
# token), so 64 sequences of ~2.3k tokens use ~4 GB; `run_unit` halves the
# batch on OOM. Not a cache key.
EVAL = EvalConfig(max_new_tokens=2048, batch_size=64)

# Training seeds per fine-tuned condition (LoRA init and sample order; the
# rows are fixed). One: the error bar is the binomial one over the test rows.
SEEDS = 1

COND_LABEL = {'base': 'base', 'direct': 'direct', 'chain': 'chain',
              'state_tracking': 'state-tracking', 'narration': 'narration', 'distill': 'distill'}


def train_cfg(seed, base=None, n_rows=None):
    """The TrainConfig of one unit: `PASSES` sweeps over `n_rows` training
    rows (default the subset size) at `seed`, from `base` (default `TRAIN`)."""
    n = DATA.n_train if n_rows is None else n_rows
    return replace(TRAIN if base is None else base, total_samples=n * PASSES, seed=seed)


def unit_list(seeds, conditions=DEFAULT_CONDITIONS):
    """Every (condition, seed) to run, in queue order: the base model once,
    then each fine-tuned condition once per seed, in `CONDITIONS` order."""
    units = []
    for c in CONDITIONS:
        if c not in conditions:
            continue
        units += [('base', None)] if c == 'base' else [(c, s) for s in range(seeds)]
    return units


# The model and optimisation settings of the current run: `MODEL` / `TRAIN`
# with the command-line overrides applied by `main`, and the number of
# training rows per condition (which fixes the sample budget). Passed
# explicitly to the workers, which re-import this module and would otherwise
# see the defaults.
class Setup:
    """`(mcfg, tcfg, n_rows)` of a run; `n_rows` maps a condition to its
    training-row count when it differs from the subset size."""

    def __init__(self, mcfg=None, tcfg=None, n_rows=None):
        self.mcfg = MODEL if mcfg is None else mcfg
        self.tcfg = TRAIN if tcfg is None else tcfg
        self.n_rows = n_rows or {}

    def unit(self, cond, seed):
        """The (TrainConfig, EvalConfig) a unit is keyed by (seed 0 stands in
        for the base model, whose key ignores it)."""
        return train_cfg(0 if seed is None else seed, self.tcfg, self.n_rows.get(cond)), EVAL


# ----------------------------------------------------------------------------
# plotting style (matches the benchmarks/entropy_exp.py knockout figures:
# figsize 3x2.5, 14pt axis labels, 12pt ticks, 7pt frameless legend, a faint
# grid and no titles)
# ----------------------------------------------------------------------------

FIGSIZE = (3, 2.5)
LABEL_FS = 14
TICK_FS = 12
LEGEND_FS = 7
LINE_LW = 1.8
BAND_ALPHA = 0.3

# A distinct, publication-friendly colour per line: the untuned model is grey,
# the hero state-tracking condition is black, and the others take well-separated
# hues (direct blue, chain orange, narration red, distill green).
COND_COLOR = {'base': '#8c8c8c', 'direct': '#4c72b0', 'chain': '#dd8452',
              'state_tracking': '#000000', 'narration': '#c44e52', 'distill': '#55a868'}


def _apply_style():
    """rcParams shared by every figure, matching the knockout plots."""
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': LABEL_FS, 'axes.titlesize': 16,
                         'xtick.labelsize': TICK_FS, 'ytick.labelsize': TICK_FS,
                         'legend.fontsize': LEGEND_FS})


def _style_axis(ax, grid='both'):
    ax.tick_params(labelsize=TICK_FS)
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid, alpha=0.3)


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

def _label(cond, seed):
    return f'[{cond}' + ('' if seed is None else f' s{seed}') + ']'


_CURVE_KEYS = ('accuracy', 'robust', 'answered', 'capped', 'mean_tokens', 'n_eval')


def _base_point(r):
    """The untuned model's result as the zero-sample point of every curve."""
    return [dict(step=0, samples=0, **{k: r[k] for k in _CURVE_KEYS})]


def _unit_jobs(units, dcfg, trcfg, device, force, progress_pos=0, setup=None):
    """
    Run each (condition, seed) unit on `device`: train the adapter if needed,
    then decode the test set with the model still loaded. A unit that raises
    is recorded as ``{'cond', 'seed', 'error'}`` and the next one starts.
    Returns the results.
    """
    import torch
    setup = setup or Setup()
    mcfg = setup.mcfg
    out = []
    for cond, seed in units:
        TR, EV = setup.unit(cond, seed)
        r = None if force else cached_unit(cond, dcfg, mcfg, TR, EV, trcfg)
        if r is not None:
            if cond == 'base':
                r['curve'] = _base_point(r)
            print(f'{_label(cond, seed)} cached', flush=True)
            out.append(r)
            continue
        # the model is loaded lazily inside run_unit, so a unit whose decode is
        # cached but whose score is not never occupies the GPU
        _pack = {}

        def get_model(_cond=cond, _TR=TR, _EV=EV):  # noqa: B023
            if 'p' not in _pack:
                adapter = None if _cond == 'base' else train_adapter(
                    _cond, dcfg, mcfg, _TR, trcfg, _EV, device=device, force=force,
                    log=lambda m: print(m, flush=True), progress_pos=progress_pos)
                _pack['p'] = load_for_eval(adapter, mcfg, _TR, device=device)
            return _pack['p']

        try:
            res = run_unit(cond, dcfg, mcfg, TR, EV, trcfg, device=device, force=force,
                           get_model=get_model, log=lambda m: print(m, flush=True),
                           progress_pos=progress_pos)
            if cond == 'base':
                res['curve'] = _base_point(res)
        except Exception as e:                    # noqa: BLE001 -- isolate the unit
            print(f'{_label(cond, seed)} FAILED: {e!r}\n{traceback.format_exc()}',
                  flush=True)
            res = dict(cond=cond, seed=seed, error=repr(e))
        out.append(res)
        _pack.clear()
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    return out


_STOP = '__stop__'          # queue sentinel


def _worker(queue, dcfg, trcfg, device, force, progress_pos, setup):
    """Pull units off the shared queue until it is drained."""
    out = []
    while True:
        unit = queue.get()
        if unit == _STOP:
            break
        cond, seed = unit
        print(f'[{device}] starting {cond}' + ('' if seed is None else f' seed {seed}'),
              flush=True)
        out.extend(_unit_jobs([unit], dcfg, trcfg, device, force, progress_pos, setup))
    print(f'[{device}] done', flush=True)
    return out


def run_all(units, dcfg, trcfg, devices, force, setup=None):
    """
    Run every unit across `devices`.

    Work is handed out through a shared queue rather than pre-assigned, so a
    GPU that finishes early immediately picks up the next unit instead of
    idling; the queue holds the units in `unit_list` order. The row sets are
    built here, in the parent, so the workers never race to build them.
    """
    setup = setup or Setup()
    t0 = time.time()
    n_tr = len(get_problems(dcfg, 'train'))
    n_te = len(get_problems(dcfg, 'test'))
    print(f'datasets ready: {n_tr} train / {n_te} test rows ({time.time() - t0:.0f}s)', flush=True)

    if len(devices) == 1:
        return _unit_jobs(units, dcfg, trcfg, devices[0], force, setup=setup)

    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    queue = manager.Queue()
    for unit in units:
        queue.put(unit)
    for _ in devices:
        queue.put(_STOP)

    results = []
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as ex:
        futs = [ex.submit(_worker, queue, dcfg, trcfg, d, force, i, setup)
                for i, d in enumerate(devices)]
        for f in futs:
            results.extend(f.result())
    return results


def pending_units(units, dcfg, trcfg, force, setup=None):
    """The units that still have to train or decode."""
    setup = setup or Setup()
    return [(c, s) for c, s in units
            if force or cached_unit(c, dcfg, setup.mcfg, *setup.unit(c, s), trcfg) is None]


def print_plan(units, dcfg, trcfg, devices, force, setup=None):
    """
    What the run will actually do: which adapters must be trained, which
    evaluations are missing, and how the work lands on the GPUs. Printed
    before anything loads so the cost is visible up front.
    """
    setup = setup or Setup()
    todo = pending_units(units, dcfg, trcfg, force, setup)
    to_train = sorted({(c, s) for c, s in todo if c != 'base' and (
        force or not adapter_is_cached(c, dcfg, setup.mcfg, setup.unit(c, s)[0], trcfg))})
    tr, te = get_problems(dcfg, 'train'), get_problems(dcfg, 'test')
    conds = [c for c in CONDITIONS if any(u == c for u, _ in units)]
    seeds = sorted({s for _, s in units if s is not None})
    print('-' * 66)
    print(f'base model     : {setup.mcfg.base_model} @ {(setup.mcfg.base_revision or "latest")[:12]}')
    print(f'conditions     : {", ".join(conds)}')
    if dcfg.dataset == 'generated':
        print(f'dataset        : generated -- {dcfg.gen_stories} stories x {dcfg.gen_people} people, '
              f'{dcfg.gen_moves} moves, {dcfg.gen_rooms} rooms (seed {dcfg.gen_seed}'
              + (', interesting-only)' if dcfg.gen_interesting_only else ')'))
    else:
        print('dataset        : released ExploreToM sample')
    print(f'training rows  : {len(tr):,} questions from {len({p.story_id for p in tr})} stories '
          f'(seed {dcfg.seed}, <= {dcfg.q_per_story} per story) x {PASSES} passes = '
          f'{len(tr) * PASSES:,} samples per adapter')
    print(f'test set       : {len(te):,} questions from {len({p.story_id for p in te})} other '
          f'stories, greedy, <= {EVAL.max_new_tokens} new tokens, batch {EVAL.batch_size}')
    if 'distill' in conds:
        s = trace_summary(dcfg, trcfg)
        print(f'distill traces : {trcfg.teacher} -- ' + (
            f'{s["n_rows"]} of {s["n_train"]} rows ({s["n_asked"]} asked, {s["n_failed"]} '
            f'unusable), {s["distill_words"]:.0f} words per trace' if s else 'MISSING'))
    print(f'units          : {len(units)} ({"1 base + " if "base" in conds else ""}'
          f'{len(conds) - ("base" in conds)} conditions x {len(seeds)} seeds); '
          f'adapters {len(to_train)}/{sum(c != "base" for c, _ in units)} to train, '
          f'evaluations {len(todo)}/{len(units)} to run')
    print(f'devices        : {devices}')
    if not todo:
        print('  everything is cached; nothing will run')
    elif len(todo) < len(devices):
        print(f'  NOTE: {len(devices) - len(todo)} GPU(s) will idle -- only {len(todo)} '
              f'unit(s) need work. Add seeds or pass fewer --devices.')
    else:
        print(f'  {len(todo)} units over {len(devices)} device(s), in queue order '
              f'(<={-(-len(todo) // len(devices))} per device)')
    tc, mc = setup.tcfg, setup.mcfg
    print(f'  per-GPU train batch {tc.batch_size} x {tc.grad_accum} accum '
          f'= {tc.batch_size * tc.grad_accum} effective; eval batch {EVAL.batch_size}')
    if tc.curve_evals > 0:
        print(f'  learning curve : {tc.curve_evals} evaluations per adapter on the test set')
    print(f'  state-tracking : a state line every {trcfg.focus_stride} step(s), '
          f'beliefs {trcfg.focus_beliefs}')
    print(f'  attention      : {mc.attn_implementation}'
          + (f', training kernels {tc.sdpa_backends}' if tc.sdpa_backends else '')
          + '  (check_train.py tests them in a minute)')
    print('rough cost     : on an H100 or B200 ~3-10 min of training per adapter (the state-')
    print('                 tracking and narration targets are long) plus ~1-4 min per evaluation')
    print(f'                 of the test rows (x{tc.curve_evals + 1} with the curve), so ~20-60 min per unit;')
    print('                 the default runs both datasets (~10 units), several hours on one GPU')
    print('-' * 66, flush=True)


def load_cached(units, dcfg, trcfg, setup=None):
    """Every cached unit, for --plot-only."""
    setup = setup or Setup()
    out = []
    for cond, seed in units:
        r = cached_unit(cond, dcfg, setup.mcfg, *setup.unit(cond, seed), trcfg)
        if r is not None:
            if cond == 'base':
                r['curve'] = _base_point(r)
            out.append(r)
    return out


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

def _binomial_sem(p, n):
    """Standard error of a proportion estimated from `n` independent trials."""
    return (p * (1.0 - p) / n) ** 0.5 if n else 0.0


def summarise(results):
    """
    Per condition: the per-seed accuracies, their mean, and a standard error.

    With more than one seed the SEM is over seeds -- the spread that matters
    when asking whether two fine-tunes differ. With one seed (the default),
    and for the base model, it is the binomial SEM over the test rows.
    """
    out = {}
    for cond in CONDITIONS:
        rows = [r for r in results if r['cond'] == cond]
        if not rows:
            continue
        accs = [r['accuracy'] for r in rows]
        mean = float(np.mean(accs))
        if len(accs) > 1:
            sem = float(np.std(accs, ddof=1) / np.sqrt(len(accs)))
        else:
            sem = _binomial_sem(mean, rows[0]['n_eval'])
        out[cond] = dict(accs=accs, mean=mean, sem=sem, seeds=[r['seed'] for r in rows],
                         robust=float(np.mean([r['robust'] for r in rows])),
                         answered=float(np.mean([r['answered'] for r in rows])),
                         capped=float(np.mean([r['capped'] for r in rows])),
                         mean_tokens=float(np.mean([r['mean_tokens'] for r in rows])),
                         n_eval=rows[0]['n_eval'], n_groups=rows[0].get('n_groups', 0),
                         by_type=rows[0].get('by_type', {}),
                         by_belief=rows[0].get('by_belief', {}))
    return out


def plot_accuracy_bars(results, name='exploretom_accuracy', subdir=''):
    """Test accuracy per condition as bars, with the SEM of `summarise` as a
    black error bar, the untuned model's level dashed, and, with several
    seeds, the individual seeds as dots."""
    _apply_style()
    stats = summarise(results)
    conds = [c for c in CONDITIONS if c in stats]
    if not conds:
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xs = np.arange(len(conds))
    for x, c in zip(xs, conds):
        s = stats[c]
        ax.bar(x, s['mean'], width=0.62, color=COND_COLOR[c], zorder=3)
        ax.errorbar(x, s['mean'], yerr=s['sem'], fmt='none', ecolor='black',
                    elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
        if len(s['accs']) > 1:
            jit = np.linspace(-0.13, 0.13, len(s['accs']))
            ax.plot(x + jit, s['accs'], ls='none', marker='o', ms=2.8, color='black',
                    alpha=0.75, zorder=5)
    if 'base' in stats:
        ax.axhline(stats['base']['mean'], color=COND_COLOR['base'], lw=1.1,
                   ls=(0, (5, 3)), zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([COND_LABEL[c] for c in conds], rotation=30, ha='right')
    ax.set_xlim(-0.6, len(conds) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Test Accuracy')
    _style_axis(ax, grid='y')
    return _save(fig, name, subdir)


def curve_stats(results):
    """
    Per fine-tuned condition: the training-sample positions of the curve
    evaluations and the mean and SEM over seeds of the accuracy at each
    (seeds are aligned by evaluation index; they share the schedule). The
    base model's point is prepended to every condition: at zero samples the
    adapter is the base model.
    """
    base = next((r['curve'][0] for r in results if r['cond'] == 'base' and r.get('curve')), None)
    out = {}
    for cond in CONDITIONS:
        if cond == 'base':
            continue
        curves = [r['curve'] for r in results if r['cond'] == cond and r.get('curve')]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        pts = [[c[i] for c in curves] for i in range(n)]
        if base is not None:
            pts.insert(0, [base] * len(curves))
        xs = [float(np.mean([p['samples'] for p in row])) for row in pts]
        accs = [np.array([p['accuracy'] for p in row]) for row in pts]
        out[cond] = dict(samples=xs, mean=[float(a.mean()) for a in accs],
                         sem=[float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
                              for a in accs], n_seeds=len(curves))
    return out, (base['accuracy'] if base is not None else None)


def plot_accuracy_curve(results, name='exploretom_curve', subdir=''):
    """Test accuracy against training samples, one line per fine-tuned
    condition (mean over seeds, SEM shaded when there are several), with the
    untuned model's accuracy as a dashed horizontal line and as the common
    starting point."""
    _apply_style()
    stats, base_acc = curve_stats(results)
    if not stats:
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for cond, s in stats.items():
        x, m, e = np.array(s['samples']), np.array(s['mean']), np.array(s['sem'])
        ax.plot(x, m, marker='o', ms=3, lw=LINE_LW, color=COND_COLOR[cond],
                label=COND_LABEL[cond], zorder=3)
        if s['n_seeds'] > 1:
            ax.fill_between(x, m - e, m + e, color=COND_COLOR[cond], alpha=BAND_ALPHA,
                            lw=0, zorder=2)
    if base_acc is not None:
        ax.axhline(base_acc, color=COND_COLOR['base'], lw=1.1, ls=(0, (5, 3)),
                   label='base', zorder=1)
    ax.set_xlabel('Training samples')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    ax.legend(frameon=False, handlelength=1.5, loc='lower right')
    _style_axis(ax)
    return _save(fig, name, subdir)


def print_curve(results):
    """The learning curve behind the figure: test accuracy at every evaluation."""
    stats, base_acc = curve_stats(results)
    if not stats:
        return
    conds = list(stats)
    xs = max((s['samples'] for s in stats.values()), key=len)
    print('\nlearning curve (test accuracy; base = '
          + (f'{base_acc:.3f})' if base_acc is not None else 'n/a)'))
    print('  samples ' + ''.join(f'{COND_LABEL[c]:>10}' for c in conds))
    for i, x in enumerate(xs):
        row = ''.join(f'{stats[c]["mean"][i]:10.3f}' if i < len(stats[c]['mean']) else ' ' * 10
                      for c in conds)
        print(f'  {x:7.0f} {row}')


def print_summary(results, trcfg, mcfg=None):
    """The table behind the figures, and the accuracy per question type."""
    stats = summarise(results)
    if not stats:
        return
    print(f'\ntest accuracy of {(mcfg or MODEL).base_model}'
          + (f' (distill teacher {trcfg.teacher})' if 'distill' in stats else ''))
    print('  sem       over seeds when there are several, else binomial over the test rows')
    print('  story     fraction of test stories with every question correct')
    print('  answered  fraction of completions that wrote an Answer field')
    print('  capped    fraction that hit the token cap without finishing (scored wrong)')
    print('  tokens    mean generated tokens per completion')
    print()
    print('  condition   seeds      acc      sem    story   answered   capped   tokens')
    for cond, s in stats.items():
        print(f'  {cond:<10} {len(s["accs"]):>6}   {s["mean"]:6.3f}   {s["sem"]:6.3f}   '
              f'{s["robust"]:6.3f}   {s["answered"]:8.3f}   {s["capped"]:6.3f}   '
              f'{s["mean_tokens"]:6.0f}')
    for cond, s in stats.items():
        if len(s['accs']) > 1:
            print(f'  {cond} per seed: ' + '  '.join(
                f's{sd}={a:.3f}' for sd, a in zip(s['seeds'], s['accs'])))
    for a, b in (('state_tracking', 'narration'), ('state_tracking', 'chain'),
                 ('state_tracking', 'direct'), ('narration', 'chain'), ('narration', 'direct'),
                 ('chain', 'direct'), ('distill', 'state_tracking')):
        if a in stats and b in stats:
            d = stats[a]['mean'] - stats[b]['mean']
            e = (stats[a]['sem'] ** 2 + stats[b]['sem'] ** 2) ** 0.5
            print(f'  {a} - {b} = {d:+.3f} (+/- {e:.3f})')

    conds = list(stats)
    types = sorted({t for s in stats.values() for t in s['by_type']})
    if types:
        print('\naccuracy per question type (first seed; n rows in brackets; the suffix is the '
              'reasoning order)')
        print('  ' + ' ' * 26 + ''.join(f'{COND_LABEL[c]:>10}' for c in conds))
        for t in types:
            n = next((s['by_type'][t][1] for s in stats.values() if t in s['by_type']), 0)
            row = ''.join(f'{s["by_type"][t][0] / max(s["by_type"][t][1], 1):10.3f}'
                          if t in s['by_type'] else ' ' * 10 for s in stats.values())
            print(f'  {t:<20} ({n:>3}) {row}')
        for k in sorted({k for s in stats.values() for k in s['by_belief']}):
            n = next((s['by_belief'][k][1] for s in stats.values() if k in s['by_belief']), 0)
            row = ''.join(f'{s["by_belief"][k][0] / max(s["by_belief"][k][1], 1):10.3f}'
                          if k in s['by_belief'] else ' ' * 10 for s in stats.values())
            print(f'  {k:<20} ({n:>3}) {row}')


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def ensure_traces(dcfg, trcfg, device, allow_build=True, teacher_batch=8, thinking=False):
    """
    The number of training rows with a distill trace, after extending the
    trace file with the teacher on `device` if fewer than `n_train` have one
    (and `allow_build`). Exits with an error if no row can be trained on.
    """
    cov = coverage(dcfg, trcfg)
    if cov['n_rows'] < dcfg.n_train and allow_build and cov['n_candidates']:
        print(f'distill traces : {cov["n_rows"]} of {dcfg.n_train} rows have one; asking '
              f'{trcfg.teacher} about more rows on {device} first', flush=True)
        teacher = make_teacher(trcfg.teacher, device=device, batch_size=teacher_batch,
                               thinking=thinking)
        build(dcfg, trcfg, teacher, log=lambda m: print('  ' + m, flush=True))
        del teacher
        gc.collect()
        if device.startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
        cov = coverage(dcfg, trcfg)
    if cov['n_rows'] == 0:
        sys.exit(f'ERROR: no training rows have a distill trace for teacher {trcfg.teacher!r}; '
                 f'build them with `python traces.py --build --teacher {trcfg.teacher}`')
    if cov['n_rows'] < dcfg.n_train:
        print(f'WARNING: only {cov["n_rows"]} of {dcfg.n_train} rows have a distill trace '
              f'({cov["n_asked"]} asked, {cov["n_failed"]} unusable, {cov["n_candidates"]} never '
              'asked); the distill condition uses those rows', flush=True)
    return cov['n_rows']


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


def run_one_dataset(dcfg, conditions, args, devices, mcfg, tcfg, trcfg):
    """
    Train and evaluate every unit for one dataset, write its two figures
    (`exploretom_accuracy_<tag>`, `exploretom_curve_<tag>`, the tag carrying a
    `_gen-...` suffix for the generated set) and print its tables. Returns the
    list of failed unit-results (empty on full success).
    """
    n_rows = {}
    n_pool = len(train_pool(dcfg))
    # the exact conditions train on the first `n_train` rows of the pool (all of
    # it if smaller); pin their sample budget to that actual count so `--n-train`
    # is honoured (at 1,500 this is a no-op).
    n_train_actual = min(n_pool, dcfg.n_train)
    if n_pool < dcfg.n_train:
        print(f'WARNING: the training pool holds {n_pool} rows, fewer than the {dcfg.n_train} '
              'asked for; the exact conditions train on all of them', flush=True)
    for c in ('direct', 'chain', 'state_tracking', 'narration'):
        n_rows[c] = n_train_actual
    if 'distill' in conditions:
        try:
            if args.plot_only:
                n_rows['distill'] = coverage(dcfg, trcfg)['n_rows'] or dcfg.n_train
            else:
                n_rows['distill'] = ensure_traces(dcfg, trcfg, devices[0],
                                                  allow_build=not args.no_build,
                                                  teacher_batch=args.teacher_batch,
                                                  thinking=args.thinking)
        except ValueError as e:                       # a corrupt record: data changed?
            sys.exit(f'ERROR: {e}')
    setup = Setup(mcfg, tcfg, n_rows)

    model_tag = args.base_model.split('/')[-1].lower()
    tag = model_tag + (f'_{teacher_tag(args.teacher)}' if 'distill' in conditions else '')
    if dcfg.dataset == 'generated':
        tag += f'_gen-p{dcfg.gen_people}m{dcfg.gen_moves}r{dcfg.gen_rooms}'
    if trcfg.focus_stride != FOCUS_STRIDE:
        tag += f'_fs{trcfg.focus_stride}'
    if trcfg.focus_beliefs != FOCUS_BELIEFS:
        tag += f'_f{trcfg.focus_beliefs[:3]}'
    units = unit_list(args.seeds, conditions)
    lead = f'[{dcfg.dataset}] '

    if args.plot_only:
        results = load_cached(units, dcfg, trcfg, setup)
        if not results:
            print(f'{lead}no cached results; nothing to plot')
            return []
    else:
        print_plan(units, dcfg, trcfg, devices, args.force, setup)
        t0 = time.time()
        results = run_all(units, dcfg, trcfg, devices, args.force, setup)
        print(f'\n{lead}wall clock: {(time.time() - t0) / 60:.1f} min')

    failed = [r for r in results if r is not None and 'error' in r]
    results = [r for r in results if r is not None and 'error' not in r]
    path = plot_accuracy_bars(results, name=f'exploretom_accuracy_{tag}')
    if path:
        print(f'\n{lead}wrote {path}.pdf/.png')
    path = plot_accuracy_curve(results, name=f'exploretom_curve_{tag}')
    if path:
        print(f'{lead}wrote {path}.pdf/.png')
    print_summary(results, trcfg, setup.mcfg)
    print_curve(results)
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--devices', default=None,
                    help='comma-separated torch devices, e.g. cuda:0,cuda:1,...')
    ap.add_argument('--seeds', type=int, default=SEEDS,
                    help='training seeds per fine-tuned condition')
    ap.add_argument('--conditions', default=','.join(DEFAULT_CONDITIONS),
                    help=f'comma-separated subset of {", ".join(CONDITIONS)} '
                         f'(default {",".join(DEFAULT_CONDITIONS)})')
    ap.add_argument('--teacher', default=TEACHER,
                    help=f'whose traces the distill condition trains on (traces.py; default {TEACHER})')
    ap.add_argument('--n-train', type=int, default=DATA.n_train)
    ap.add_argument('--n-test', type=int, default=DATA.n_test)
    ap.add_argument('--q-per-story', type=int, default=DATA.q_per_story,
                    help='at most this many questions per story in either split')
    ap.add_argument('--dataset', choices=DATASETS, default=None,
                    help="'sample' (released ExploreToM) or 'generated' (larger local stories); "
                         "default runs BOTH and writes a set of figures for each")
    ap.add_argument('--gen-people', type=int, default=DATA.gen_people)
    ap.add_argument('--gen-rooms', type=int, default=DATA.gen_rooms)
    ap.add_argument('--gen-moves', type=int, default=DATA.gen_moves,
                    help='generated: actions per story after the initial placements')
    ap.add_argument('--gen-stories', type=int, default=DATA.gen_stories)
    ap.add_argument('--gen-seed', type=int, default=DATA.gen_seed)
    ap.add_argument('--gen-interesting-only', action='store_true',
                    help='generated: keep only questions whose answer depends on who is asked')
    ap.add_argument('--base-model', default=MODEL.base_model,
                    help='Hub id of the base model; one of ' + ', '.join(BASE_MODELS)
                         + ' is pinned to its recorded revision')
    ap.add_argument('--focus-stride', type=int, default=FOCUS_STRIDE,
                    help='state-tracking: a state line after every block of this many steps')
    ap.add_argument('--focus-beliefs', choices=BELIEF_MODES, default=FOCUS_BELIEFS,
                    help="state-tracking: write the asked beliefs on every line ('explicit') or "
                         "only where they depart from the truth ('departures')")
    ap.add_argument('--curve-evals', type=int, default=TRAIN.curve_evals,
                    help='evaluations per adapter during training (0: none)')
    ap.add_argument('--max-seq-len', type=int, default=TRAIN.max_seq_len,
                    help='training sequence cap (default 4096)')
    ap.add_argument('--attn', choices=('sdpa', 'eager'), default=MODEL.attn_implementation,
                    help='attention implementation of the model')
    ap.add_argument('--sdpa-backends', default=TRAIN.sdpa_backends, metavar='LIST',
                    help="kernels torch's fused attention may use in training, e.g. "
                         "'flash,efficient,math' (default) or 'math'; '' for torch's choice")
    ap.add_argument('--no-build', action='store_true',
                    help='distill: do not extend a short trace file; train on what exists')
    ap.add_argument('--teacher-batch', type=int, default=8,
                    help='decode batch of a local teacher when traces are built here')
    ap.add_argument('--thinking', action='store_true',
                    help='local teacher: thinking mode on when traces are built here')
    ap.add_argument('--force', action='store_true', help='retrain and re-evaluate')
    ap.add_argument('--plot-only', action='store_true',
                    help='build the figures from cached results only')
    args = ap.parse_args()

    conditions = tuple(c.strip() for c in args.conditions.split(',') if c.strip())
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        ap.error(f'unknown condition(s) {unknown}; choose from {", ".join(CONDITIONS)}')
    devices = args.devices.split(',') if args.devices else _default_devices()

    # the model/optimiser/trace settings are shared across datasets; only the
    # DataConfig changes. Default (`--dataset` unset) runs both datasets, the
    # generated ("positive") set first and the released sample ("negative"
    # baseline) second, each writing its own set of figures.
    mcfg = replace(MODEL, base_model=args.base_model,
                   base_revision=BASE_MODELS.get(args.base_model), attn_implementation=args.attn)
    tcfg = replace(TRAIN, sdpa_backends=args.sdpa_backends,
                   curve_evals=args.curve_evals, max_seq_len=args.max_seq_len)
    trcfg = TraceConfig(teacher=args.teacher,
                        focus_stride=args.focus_stride, focus_beliefs=args.focus_beliefs)
    base_dcfg = replace(DATA, n_train=args.n_train, n_test=args.n_test, q_per_story=args.q_per_story,
                        gen_people=args.gen_people, gen_rooms=args.gen_rooms,
                        gen_moves=args.gen_moves, gen_stories=args.gen_stories,
                        gen_seed=args.gen_seed, gen_interesting_only=args.gen_interesting_only)
    datasets = [args.dataset] if args.dataset else ['generated', 'sample']

    failed = []
    for i, ds in enumerate(datasets):
        if len(datasets) > 1:
            print('\n' + '=' * 66)
            print(f'DATASET {i + 1}/{len(datasets)}: {ds}')
            print('=' * 66, flush=True)
        dcfg = replace(base_dcfg, dataset=ds)
        failed += run_one_dataset(dcfg, conditions, args, devices, mcfg, tcfg, trcfg)

    if failed:
        print('\nFAILED units (cached work is kept; rerun to retry them):')
        for r in failed:
            print(f'  {_label(r["cond"], r["seed"])} {r["error"]}')
        sys.exit(1)


if __name__ == '__main__':
    main()
