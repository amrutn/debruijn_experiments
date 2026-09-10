"""
Run the ExploreToM state-tracking experiment: does fine-tuning on reasoning
that keeps the story's state in the local context (a restated step plus the
exact world-and-belief state after it) beat fine-tuning on the question-specific
key steps that change the answer, or on the answers alone?

Models scored on the ExploreToM test rows (`exploretom_data`):

base            Qwen2.5-1.5B-Instruct as released, no fine-tuning, under the
                same prompt. ``--base-model Qwen/Qwen2.5-1.5B`` runs the
                pretrained model instead (`train_eval.BASE_MODELS`).
direct          a LoRA trained on the answer field alone of the training rows.
key_steps       a LoRA trained on the *same* rows with the steps at which the
                answer to the question changes, each with the answer after it.
                Shown as "key-steps".
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

The key_steps, state_tracking and narration targets are computed from the
replayed story, so the exact conditions are trained on identical rows with
identical labels and differ in the format of the reasoning only. Training is LoRA r=32 on every
projection, lr 1e-4 cosine, effective batch 32, over three passes of the rows.
The whole thing is repeated over three seeds, each an independent train/test
split (and training seed); results are averaged and the error bars are the SEM
over the seeds. While an adapter trains it is scored twenty times on the test
set (`TrainConfig.curve_evals`), which gives the learning curve.

Datasets
--------
The default dataset is the larger locally-generated stories (`generated`,
`DataConfig`: 4 people / 12 moves / 3 rooms, 500 stories) -- the "positive" set
that separates the reasoning formats. 4 people (not 6) keeps the answer
distribution far less skewed (the majority baseline is ~0.38 rather than ~0.56;
see REPRODUCIBILITY.md §4.1). Pass `--dataset sample` for the released ExploreToM
sample -- the near-saturated "negative" baseline where the good formats tie
(kept as a comparison, no longer run by default).

Stride sweep (`--stride-sweep`)
-------------------------------
A separate ablation: train state_tracking at several state-emission intervals
(`STRIDE_SWEEP`: a state line after every block of k steps, plus the no-interval
"std." case where the state is written only once at the end), with a final
evaluation only (no learning curve), over `SEEDS` splits of the generated data.
Writes `exploretom_stride_<tag>` -- final accuracy vs interval, SEM over seeds,
in the same format as the `math_task` accuracy-vs-k figure. The bare default
(`python run_experiments.py`) runs the generated main experiment and then this
sweep; `--stride-sweep` runs only the sweep; an explicit `--dataset X` runs only
that dataset's main experiment (no sweep).

Figures (pdf+png in figures/)
-----------------------------
exploretom_accuracy_<model>[_<teacher>][_gen-p<N>m<N>r<N>]
                          Test accuracy per condition as a bar (mean over the
                          seeds), with the SEM over the seeds as an error bar
                          and each seed's own accuracy as a dot, and the untuned
                          model's mean accuracy as a dashed line.
exploretom_curve_<model>[_<teacher>][_gen-p<N>m<N>r<N>]
                          Test accuracy against training iterations (optimizer
                          steps), one line per fine-tuned condition (mean over
                          seeds) starting from
                          the untuned model at zero samples, a shaded +/-1 SEM
                          band over the seeds, and the untuned model's accuracy
                          as a dashed horizontal line.
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
    python run_experiments.py --devices cuda:0                  # generated main run + stride sweep
    python run_experiments.py --dataset sample                  # the released-sample baseline (no sweep)
    python run_experiments.py --stride-sweep --devices cuda:0   # only the state-emission-interval ablation
    python run_experiments.py --dataset generated --gen-people 6      # override: the more-skewed 6-person set
    python run_experiments.py --conditions base,direct,key_steps,state_tracking,distill
    python run_experiments.py --plot-only                       # figures from cache
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
from matplotlib.colors import to_rgba

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

# 1,500 training questions and 250 test questions from disjoint stories (at
# most 4 questions per story). The split seed varies per replicate (see SEEDS),
# so `seed=0` here is only the first split.
DATA = DataConfig(n_train=1500, n_test=250, q_per_story=4, seed=0)

# The math task's adapter and optimiser settings: r=32 (~2.4% of the model);
# batch 8 x 4 accumulation is the effective batch of 32 -- kept as 8 x 4 on
# larger GPUs too, so the loss is averaged exactly as in the math task.
MODEL = ModelConfig()
TRAIN = TrainConfig(lr=1e-4, batch_size=8, grad_accum=4)
PASSES = 3                      # sweeps over the training rows per adapter (epochs)

# Greedy, batch 64: the 1.5B model's KV cache is small (2 KV heads, ~28 KB per
# token), so 64 sequences of ~2.3k tokens use ~4 GB; `run_unit` halves the
# batch on OOM. Not a cache key.
EVAL = EvalConfig(max_new_tokens=2048, batch_size=64)

# Number of replicates. Each seed k is an INDEPENDENT run: a different
# train/test split (`dcfg.seed = k`, so the test rows differ) and a different
# training seed (LoRA init and sample order). Results are averaged over the
# seeds and the error bars are the SEM over them -- the spread that reflects
# both the data split and the training randomness. The untuned base model is
# re-evaluated on each split.
SEEDS = 3

# `--stride-sweep` ablation: the state-emission intervals to train state_tracking
# at (a state line after every block of this many steps). STRIDE_NONE is a
# sentinel larger than any story length, so `blocks()` yields a single block and
# the state is written only once, at the end of the story -- the "no interval"
# case. Plotted as accuracy vs interval, final eval only, over `SEEDS` seeds.
STRIDE_NONE = 10 ** 6
STRIDE_SWEEP = (1, 2, 3, 4, 5, 6, STRIDE_NONE)
STRIDE_LABELS = '1,2,3,4,5,6,std.'

COND_LABEL = {'base': 'base', 'direct': 'direct', 'key_steps': 'key-steps',
              'state_tracking': 'state-tracking', 'narration': 'narration', 'distill': 'distill'}


def train_cfg(seed, base=None, n_rows=None):
    """The TrainConfig of one unit: `PASSES` sweeps over `n_rows` training
    rows (default the subset size) at `seed`, from `base` (default `TRAIN`)."""
    n = DATA.n_train if n_rows is None else n_rows
    return replace(TRAIN if base is None else base, total_samples=n * PASSES, seed=seed)


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
# hues (direct blue, key-steps orange, narration red, distill green).
COND_COLOR = {'base': '#8c8c8c', 'direct': '#4c72b0', 'key_steps': '#dd8452',
              'state_tracking': '#000000', 'narration': '#c44e52', 'distill': '#55a868'}


def _apply_style():
    """rcParams shared by every figure, matching the knockout plots."""
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': LABEL_FS, 'axes.titlesize': 16,
                         'xtick.labelsize': TICK_FS, 'ytick.labelsize': TICK_FS,
                         'legend.fontsize': LEGEND_FS})


def _style_axis(ax, grid=None):
    ax.tick_params(labelsize=TICK_FS)
    if grid:
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
    idling; the queue holds the units in `CONDITIONS` order. The row sets are
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
    print(f'                 the bare default is the generated main run + the stride sweep x {SEEDS}')
    print('                 seeds -- many units, hours on one GPU (everything caches/resumes)')
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
    """Test accuracy per fine-tuned condition as bars, with the SEM of
    `summarise` as a black error bar and, with several seeds, the individual
    seeds as dots. The untuned model is the annotated dashed line, not a bar."""
    _apply_style()
    stats = summarise(results)
    conds = [c for c in CONDITIONS if c in stats and c != 'base']   # base is the dashed line
    if not conds:
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xs = np.arange(len(conds))
    for x, c in zip(xs, conds):
        s = stats[c]
        ax.bar(x, s['mean'], width=0.62, facecolor=to_rgba(COND_COLOR[c], 0.3),
               edgecolor=COND_COLOR[c], linewidth=1.6, zorder=3)
        ax.errorbar(x, s['mean'], yerr=s['sem'], fmt='none', ecolor='black',
                    elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
        if len(s['accs']) > 1:
            jit = np.linspace(-0.13, 0.13, len(s['accs']))
            ax.plot(x + jit, s['accs'], ls='none', marker='o', ms=2.8, color='black',
                    alpha=0.75, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([COND_LABEL[c] for c in conds], rotation=30, ha='right')
    ax.set_xlim(-0.6, len(conds) - 0.4)
    ax.set_ylim(0, 1)
    if 'base' in stats:
        b = stats['base']['mean']
        ax.axhline(b, color=COND_COLOR['base'], lw=1.1, ls=(0, (5, 3)), zorder=2)
        ax.text(len(conds) - 0.5, b + 0.015, 'not-tuned', ha='right', va='bottom',
                fontsize=LEGEND_FS, color=COND_COLOR['base'])
    ax.set_ylabel('Test Accuracy')
    _style_axis(ax)
    return _save(fig, name, subdir)


def curve_stats(results):
    """
    Per fine-tuned condition: the optimizer-step (training-iteration) positions
    of the curve evaluations and the mean and SEM over seeds of the accuracy at
    each (seeds are aligned by evaluation index; they share the schedule). The
    untuned model is prepended as the step-0 point of every curve, using each
    seed's own base evaluation so its mean and SEM match the base bar.
    """
    base_pts = [r['curve'][0] for r in results if r['cond'] == 'base' and r.get('curve')]
    out = {}
    for cond in CONDITIONS:
        if cond == 'base':
            continue
        curves = [r['curve'] for r in results if r['cond'] == cond and r.get('curve')]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        pts = [[c[i] for c in curves] for i in range(n)]
        if base_pts:
            pts.insert(0, base_pts)         # the untuned model on each seed's split
        xs = [float(np.mean([p['step'] for p in row])) for row in pts]
        accs = [np.array([p['accuracy'] for p in row]) for row in pts]
        out[cond] = dict(iterations=xs, mean=[float(a.mean()) for a in accs],
                         sem=[float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
                              for a in accs], n_seeds=len(curves))
    base_acc = float(np.mean([p['accuracy'] for p in base_pts])) if base_pts else None
    return out, base_acc


def plot_accuracy_curve(results, name='exploretom_curve', subdir=''):
    """Test accuracy against training iterations (optimizer steps), one line per
    fine-tuned condition (mean over seeds, SEM shaded when there are several),
    with the untuned model's accuracy as a dashed horizontal line and as the
    common starting point (iteration 0)."""
    _apply_style()
    stats, base_acc = curve_stats(results)
    if not stats:
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    handles = {}
    for cond, s in stats.items():
        x, m, e = np.array(s['iterations']), np.array(s['mean']), np.array(s['sem'])
        (handles[cond],) = ax.plot(x, m, lw=LINE_LW, color=COND_COLOR[cond],
                                   label=COND_LABEL[cond], zorder=3)
        if s['n_seeds'] > 1:
            ax.fill_between(x, m - e, m + e, color=COND_COLOR[cond], alpha=BAND_ALPHA,
                            lw=0, zorder=2)
    if base_acc is not None:
        handles['base'] = ax.axhline(base_acc, color=COND_COLOR['base'], lw=1.1,
                                     ls=(0, (5, 3)), label='not-tuned', zorder=1)
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    # state-tracking first (top-left of the legend), the not-tuned line last; the
    # 5 default entries fill 2 columns as a 2x2x1 block
    order = [c for c in (['state_tracking'] + [c for c in stats if c != 'state_tracking']
                         + ['base']) if c in handles]
    ax.legend([handles[c] for c in order], [handles[c].get_label() for c in order],
              frameon=False, handlelength=1.5, loc='upper left', ncol=2,
              columnspacing=1.0, handletextpad=0.5)
    _style_axis(ax)
    return _save(fig, name, subdir)


def print_curve(results):
    """The learning curve behind the figure: test accuracy at every evaluation."""
    stats, base_acc = curve_stats(results)
    if not stats:
        return
    conds = list(stats)
    xs = max((s['iterations'] for s in stats.values()), key=len)
    print('\nlearning curve (test accuracy; base = '
          + (f'{base_acc:.3f})' if base_acc is not None else 'n/a)'))
    print('  iters   ' + ''.join(f'{COND_LABEL[c]:>10}' for c in conds))
    for i, x in enumerate(xs):
        row = ''.join(f'{stats[c]["mean"][i]:10.3f}' if i < len(stats[c]['mean']) else ' ' * 10
                      for c in conds)
        print(f'  {x:7.0f} {row}')


def plot_stride_sweep(stride_accs, cond_accs, name='exploretom_stride', subdir=''):
    """
    Final-answer accuracy of state_tracking vs the state-emission interval, in the
    `math_task` accuracy-vs-k format: a line with markers over the finite
    intervals; the no-interval / final-state-only case ("std.", the `STRIDE_NONE`
    sentinel) as a detached diamond one slot past the largest interval, across a
    thin divider and joined by a dashed connector; and a **horizontal dashed
    reference line for each fixed (interval-independent) condition** in
    `cond_accs` -- base ("not-tuned"), direct, key_steps, narration -- at its
    mean-over-seeds accuracy, coloured by condition and named in the legend.
    Means over seeds with +/-1 SEM bars; `stride_accs` maps a stride to its
    per-seed accuracies, `cond_accs` a condition to its per-seed accuracies.
    """
    from matplotlib.lines import Line2D
    _apply_style()
    strides = [s for s in STRIDE_SWEEP if stride_accs.get(s)]
    if not strides:
        return None
    ks = [s for s in strides if s != STRIDE_NONE]
    kmax = max(ks) if ks else 1
    has_std = STRIDE_NONE in strides
    xstd = kmax + 1.6                                     # std sits one slot right of kmax
    color = COND_COLOR['state_tracking']
    mean = lambda v: float(np.mean(v))                                                # noqa: E731
    sem = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0  # noqa: E731
    fig, ax = plt.subplots(figsize=FIGSIZE)
    handles = [Line2D([0], [0], color=color, lw=1.5, marker='o', ms=4,
                      label=COND_LABEL['state_tracking'])]
    # a horizontal dashed reference line per fixed condition, coloured by condition
    for cond, lbl in (('base', 'not-tuned'), ('direct', COND_LABEL['direct']),
                      ('key_steps', COND_LABEL['key_steps']),
                      ('narration', COND_LABEL['narration'])):
        v = cond_accs.get(cond)
        if not v:
            continue
        ax.axhline(mean(v), color=COND_COLOR[cond], lw=1.1, ls=(0, (5, 3)), zorder=1)
        handles.append(Line2D([0], [0], color=COND_COLOR[cond], lw=1.1, ls=(0, (5, 3)), label=lbl))
    xs, ys = ks, [mean(stride_accs[s]) for s in ks]
    if ks:                                               # the finite-interval sweep
        ax.errorbar(xs, ys, yerr=[sem(stride_accs[s]) for s in ks], marker='o', ms=4.2,
                    lw=1.5, color=color, capsize=2.5, elinewidth=0.9, zorder=3)
    if has_std:                                          # the no-interval ("std.") case
        ystd = mean(stride_accs[STRIDE_NONE])
        if ks:
            ax.plot([xs[-1], xstd], [ys[-1], ystd], ls=(0, (3, 2)), lw=1.2, color=color, zorder=2)
        ax.axvline(kmax + 0.8, color='0.85', lw=0.8, zorder=0)   # separates std
        ax.errorbar([xstd], [ystd], yerr=[sem(stride_accs[STRIDE_NONE])], marker='D', ms=5.5,
                    color=color, ls='none', capsize=2.5, elinewidth=0.9, zorder=4)
    ax.set_xticks(ks + ([xstd] if has_std else []))
    ax.set_xticklabels([str(k) for k in ks] + (['std.'] if has_std else []))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Interval $k$', fontsize=LABEL_FS)
    ax.set_ylabel('Final-Answer Accuracy', fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leg = ax.legend(handles=handles, fontsize=LEGEND_FS - 1, frameon=True, loc='lower center',
                    ncol=2, handlelength=1.6, handletextpad=0.5, columnspacing=1.1,
                    labelspacing=0.3, borderpad=0.35, facecolor='white', edgecolor='0.8',
                    framealpha=1.0)
    leg.get_frame().set_linewidth(0.7)
    return _save(fig, name, subdir)


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
    for a, b in (('state_tracking', 'narration'), ('state_tracking', 'key_steps'),
                 ('state_tracking', 'direct'), ('narration', 'key_steps'), ('narration', 'direct'),
                 ('key_steps', 'direct'), ('distill', 'state_tracking')):
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


def _n_rows_for(dcfg, conditions, args, trcfg, devices):
    """The training-row count per condition for one train/test split (`dcfg`).
    The exact conditions train on the first `n_train` rows of the split's pool
    (all of it if smaller), pinning their sample budget to that actual count so
    `--n-train` is honoured (at 1,500 this is a no-op)."""
    n_rows = {}
    n_pool = len(train_pool(dcfg))
    n_train_actual = min(n_pool, dcfg.n_train)
    if n_pool < dcfg.n_train:
        print(f'WARNING: the training pool holds {n_pool} rows, fewer than the {dcfg.n_train} '
              'asked for; the exact conditions train on all of them', flush=True)
    for c in ('direct', 'key_steps', 'state_tracking', 'narration'):
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
    return n_rows


def run_one_dataset(dcfg, conditions, args, devices, mcfg, tcfg, trcfg):
    """
    Run the whole experiment for one dataset over `args.seeds` independent
    train/test splits (each seed k uses `dcfg.seed = k` for the split and k as
    the training seed; the base model is re-evaluated on each split), then
    average over the seeds and write the two figures (`exploretom_accuracy_<tag>`,
    `exploretom_curve_<tag>`, the tag carrying a `_gen-...` suffix for the
    generated set) with SEM error bars/bands over the seeds. Returns the list of
    failed unit-results (empty on full success).
    """
    model_tag = args.base_model.split('/')[-1].lower()
    tag = model_tag + (f'_{teacher_tag(args.teacher)}' if 'distill' in conditions else '')
    if dcfg.dataset == 'generated':
        tag += f'_gen-p{dcfg.gen_people}m{dcfg.gen_moves}r{dcfg.gen_rooms}'
    if trcfg.focus_stride != FOCUS_STRIDE:
        tag += f'_fs{trcfg.focus_stride}'
    if trcfg.focus_beliefs != FOCUS_BELIEFS:
        tag += f'_f{trcfg.focus_beliefs[:3]}'
    lead = f'[{dcfg.dataset}] '

    all_results = []
    for k in range(args.seeds):
        dcfg_k = replace(dcfg, seed=k)
        if args.seeds > 1:
            print(f'\n{lead}===== seed {k + 1}/{args.seeds} (train/test split seed {k}) =====',
                  flush=True)
        setup = Setup(mcfg, tcfg, _n_rows_for(dcfg_k, conditions, args, trcfg, devices))
        # every condition, base included, evaluated on THIS split
        units = [(c, k) for c in CONDITIONS if c in conditions]
        if args.plot_only:
            res = load_cached(units, dcfg_k, trcfg, setup)
        else:
            print_plan(units, dcfg_k, trcfg, devices, args.force, setup)
            t0 = time.time()
            res = run_all(units, dcfg_k, trcfg, devices, args.force, setup)
            print(f'\n{lead}seed {k} wall clock: {(time.time() - t0) / 60:.1f} min')
        for r in res:                       # stamp the split seed (base reports None)
            if r is not None:
                r['seed'] = k
        all_results += res

    if not all_results:
        print(f'{lead}no cached results; nothing to plot')
        return []
    failed = [r for r in all_results if r is not None and 'error' in r]
    results = [r for r in all_results if r is not None and 'error' not in r]
    path = plot_accuracy_bars(results, name=f'exploretom_accuracy_{tag}')
    if path:
        print(f'\n{lead}wrote {path}.pdf/.png  (mean over {args.seeds} seeds, SEM error bars)')
    path = plot_accuracy_curve(results, name=f'exploretom_curve_{tag}')
    if path:
        print(f'{lead}wrote {path}.pdf/.png')
    print_summary(results, trcfg, mcfg)
    print_curve(results)
    return failed


def run_stride_sweep(base_dcfg, args, devices, mcfg, tcfg, trcfg):
    """
    Ablation of the state-emission interval: train state_tracking at each
    `STRIDE_SWEEP` stride (a state line after every block of that many steps;
    the `STRIDE_NONE` sentinel emits the state only once, at the end), with a
    final evaluation only (no learning curve), over `args.seeds` independent
    splits of the generated data. Writes `exploretom_stride_<tag>` (accuracy vs
    interval, SEM over seeds) with a horizontal dashed reference line for each
    fixed condition -- not-tuned (base), direct, key_steps, narration -- taken
    from the main run's cache, and prints the table. Returns failed unit-results.
    """
    dcfg0 = replace(base_dcfg, dataset='generated')
    tcfg0 = replace(tcfg, curve_evals=0)                 # no intermediate evaluations
    model_tag = args.base_model.split('/')[-1].lower()
    tag = model_tag + f'_gen-p{dcfg0.gen_people}m{dcfg0.gen_moves}r{dcfg0.gen_rooms}'
    if trcfg.focus_beliefs != FOCUS_BELIEFS:
        tag += f'_f{trcfg.focus_beliefs[:3]}'
    labels = ['std.' if s == STRIDE_NONE else str(s) for s in STRIDE_SWEEP]
    print('\n' + '=' * 66)
    print(f'STRIDE SWEEP (state_tracking): intervals {labels}, {args.seeds} seed(s), '
          'final eval only, generated data')
    print('=' * 66, flush=True)

    # fixed (interval-independent) conditions overlaid as dashed reference lines:
    # `base` is evaluated here (cheap), the others are loaded from the main run's
    # cache with its own `tcfg` (never trained by the sweep) and omitted if absent.
    REF_CONDS = ('direct', 'key_steps', 'narration')
    stride_accs = {s: [] for s in STRIDE_SWEEP}
    cond_accs = {c: [] for c in ('base',) + REF_CONDS}
    failed = []
    for k in range(args.seeds):
        dcfg_k = replace(dcfg0, seed=k)
        nr = _n_rows_for(dcfg_k, ('direct', 'key_steps', 'state_tracking', 'narration'),
                         args, trcfg, devices)
        sweep_setup = Setup(mcfg, tcfg0, nr)             # sweep training: curve_evals=0
        ref_setup = Setup(mcfg, tcfg, nr)                # main tcfg -> reuse the main run's cache

        def collect(units, trc, setup, sink, load_only=False):
            res = (load_cached(units, dcfg_k, trc, setup) if (args.plot_only or load_only)
                   else run_all(units, dcfg_k, trc, devices, args.force, setup))
            for r in res:
                if r is None:
                    continue
                if 'error' in r:
                    r['seed'] = k
                    failed.append(r)
                else:
                    sink.append(r['accuracy'])

        collect([('base', k)], trcfg, sweep_setup, cond_accs['base'])   # untuned reference
        for c in REF_CONDS:                              # fixed-condition lines from the main run
            collect([(c, k)], trcfg, ref_setup, cond_accs[c], load_only=True)
        for s in STRIDE_SWEEP:                           # the interval sweep
            lbl = 'std.' if s == STRIDE_NONE else str(s)
            if not args.plot_only:
                print(f'\n[stride {lbl}] seed {k}', flush=True)
            collect([('state_tracking', k)], replace(trcfg, focus_stride=s), sweep_setup, stride_accs[s])

    stride_accs = {s: v for s, v in stride_accs.items() if v}
    missing = [c for c in REF_CONDS if not cond_accs[c]]
    if missing:
        print(f'  (no cached main-run results for {missing}; their reference lines are omitted -- '
              'run the generated main experiment to include them)', flush=True)
    path = plot_stride_sweep(stride_accs, cond_accs, name=f'exploretom_stride_{tag}')
    if path:
        print(f'\n[stride-sweep] wrote {path}.pdf/.png  (state_tracking vs interval; dashed lines '
              f'= fixed conditions; mean over {args.seeds} seed(s))')

    def _row(lbl, v):
        v = np.array(v)
        s = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        print(f'  {lbl:>16}    {v.mean():6.3f}   {s:6.3f}   ({len(v)})')
    print('\nstride sweep -- final test accuracy (mean +/- SEM over seeds):')
    print('  condition/interval      acc     sem   (seeds)')
    for s in STRIDE_SWEEP:
        if s in stride_accs:
            _row('k=' + ('std.' if s == STRIDE_NONE else str(s)), stride_accs[s])
    for c, lbl in (('base', 'not-tuned'), ('direct', 'direct'),
                   ('key_steps', 'key-steps'), ('narration', 'narration')):
        if cond_accs.get(c):
            _row(lbl, cond_accs[c])
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
                    help="'generated' (larger local stories, the default) or 'sample' (the "
                         "released ExploreToM baseline)")
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
    ap.add_argument('--stride-sweep', action='store_true',
                    help=f'run ONLY the ablation: train state_tracking at emission intervals '
                         f'{STRIDE_LABELS} (final accuracy only, no curve) on the generated data '
                         'and plot accuracy vs interval. The bare default runs this sweep after '
                         'the generated main experiment; an explicit --dataset skips it')
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
    # DataConfig changes. The default dataset is the generated one; pass
    # `--dataset sample` for the (saturated) released-sample baseline.
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

    # `--stride-sweep` alone runs only the state-emission-interval ablation. The
    # bare default (no explicit `--dataset`) runs the main experiment on the
    # generated data AND then the sweep; an explicit `--dataset X` runs only that
    # dataset's main experiment. The sweep trains state_tracking at several
    # intervals (final accuracy only, no learning curve) on the generated data.
    if args.stride_sweep:
        failed = run_stride_sweep(base_dcfg, args, devices, mcfg, tcfg, trcfg)
    else:
        ds = args.dataset or 'generated'
        failed = run_one_dataset(replace(base_dcfg, dataset=ds), conditions, args,
                                 devices, mcfg, tcfg, trcfg)
        if args.dataset is None:                 # bare default: main run + the sweep
            failed += run_stride_sweep(base_dcfg, args, devices, mcfg, tcfg, trcfg)

    if failed:
        print('\nFAILED units (cached work is kept; rerun to retry them):')
        for r in failed:
            print(f'  {_label(r["cond"], r["seed"])} {r["error"]}')
        sys.exit(1)


if __name__ == '__main__':
    main()
