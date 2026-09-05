"""
Run the GSM8K restated-reasoning experiment: does fine-tuning on reasoning that
keeps its state in the local context beat fine-tuning on the ground-truth
reasoning, from the same 1,500 problems -- and how does either compare with
distilling a much larger model's reasoning on those same problems?

Four models are scored on the GSM8K test set:

base        Qwen2.5-1.5B-Instruct as released, no fine-tuning.
standard    a LoRA trained on the ground-truth solutions of 1,500 training
            problems (calculator annotations stripped).
restated    a LoRA trained on the *same* 1,500 problems, the same solution
            steps, with a ledger of the problem's state (``Known so far: ...``)
            written before the first step and after every step (`gsm8k_data`).
            The ledger lines were written by Claude (`traces.py`,
            ``traces_ledger.txt``); ``--author rule`` swaps in a mechanical
            ledger as a control for their wording.
distill     a LoRA trained on the same 1,500 problems with a larger model's
            solution in place of the ground truth (`distill.py`,
            ``traces_distill_<teacher>.txt``). By default the concise
            reasoning of Qwen3.5-397B-A17B, which is as short as the ground
            truth, so the condition differs from 'standard' only in who wrote
            the reasoning; ``--teacher qwen3-8b-think`` trains on Qwen3-8B
            thinking traces instead, under the longer budgets those need.

The fine-tuning settings are the math task's (LoRA r=32 on every projection,
lr 1e-4 cosine, effective batch 32, three passes over the pool). Each
fine-tuned condition is trained with several seeds, so the figure carries the
seed-to-seed spread and the comparison is not a single draw.

Figure (pdf+png in figures/)
----------------------------
gsm8k_accuracy_<author>   Test accuracy per condition as a bar, with the SEM
                          over seeds as an error bar (binomial SEM over the test
                          set for the base model, which has no seed) and the
                          individual seeds as dots. <author> is who wrote the
                          ledger lines: 'claude' or 'rule'.

Scheduling and robustness
-------------------------
One (condition, seed) is one unit of work: train the adapter, then decode the
test set with the model still resident. Units are queued in `CONDITIONS`
order -- base, standard, restated, distill -- so the distillation units run
after everything else and the main comparison completes first; on several
GPUs the queue is shared and a GPU that finishes early takes the next unit.
Everything is cached (`train_eval`): the problem sets, the adapters, the
decodes and the scores, and within a unit the training checkpoints and every
finished decode batch, so an interrupted run resumes within minutes of where
it stopped. A unit that fails with an error the decoder cannot retry is
reported and skipped rather than ending the run: the figure and the table are
built from whatever finished, and the exit status says whether anything
failed. The restated and distillation traces are validated once in the
parent before any worker starts, so a bad trace file fails before a GPU is
touched.

Usage
-----
    python run_experiments.py                                   # auto devices
    python run_experiments.py --devices cuda:0                  # one GPU
    python run_experiments.py --author rule                     # mechanical ledgers
    python run_experiments.py --teacher qwen3-8b-think          # thinking-trace teacher
    python run_experiments.py --conditions base,standard,restated
    python run_experiments.py --plot-only                       # figure from cache
    python run_experiments.py --n-train 64 --n-test 64 --seeds 1   # smoke test
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

from gsm8k_data import DataConfig, get_problems
from traces import TraceConfig, load_traces, trace_summary, AUTHORS
from distill import (
    TEACHERS, FILLS, load_distill_traces, distill_summary, budget as distill_budget,
)
from train_eval import (
    ModelConfig, TrainConfig, EvalConfig, CONDITIONS,
    run_unit, cached_unit, train_adapter, load_for_eval, adapter_is_cached,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')


# ----------------------------------------------------------------------------
# the experiment
# ----------------------------------------------------------------------------

# 1,500 of the 7,473 GSM8K training problems, drawn once with seed 0 and shared
# by every fine-tuned condition, so the adapters differ only in the trace they
# were trained on. The test set is the full 1,319 problems.
DATA = DataConfig(n_train=1500, train_seed=0, n_test=None)

# The math task's adapter and optimiser settings, verbatim: r=32 doubles the
# adapter to ~2.4% of the model; batch 8 x 4 accumulation is the effective
# batch of 32 -- kept as 8 x 4 on larger GPUs too, so the loss is averaged
# exactly as in the math task.
MODEL = ModelConfig(lora_r=32, lora_alpha=64)
TRAIN = TrainConfig(lr=1e-4, batch_size=8, grad_accum=4)
PASSES = 3                      # sweeps over the 1,500-problem pool per adapter

# Greedy, batch 128: the 1.5B model's KV cache is small (2 KV heads, ~28 KB per
# token), so 128 sequences of ~1.2k tokens use ~4 GB and fit an H100 or B200
# with room to spare; `run_unit` halves the batch on OOM. Not a cache key.
EVAL = EvalConfig(max_new_tokens=1024, batch_size=128)

# Training seeds per fine-tuned condition (LoRA init and sample order; the
# problem subset is fixed). Three gives an error bar at the cost of three units
# per condition.
SEEDS = 3

# tick labels; 'std.' as in the math figures, 'ledger' for the restated
# condition (what its traces are), all short enough that four fit under a
# 3-inch axis at the tick font size without touching
COND_LABEL = {'base': 'base', 'standard': 'std.', 'restated': 'ledger', 'distill': 'distill'}


def train_cfg(seed, cond='standard', dcfg=DATA, trcfg=None):
    """
    The TrainConfig of one unit: `PASSES` sweeps over the training pool at
    `seed`. The distillation condition takes the teacher's sequence budget
    when it has one (`distill.budget`); every other setting is shared.
    """
    tcfg = replace(TRAIN, total_samples=dcfg.n_train * PASSES, seed=seed)
    if cond == 'distill' and trcfg is not None:
        max_seq_len, _ = distill_budget(trcfg)
        if max_seq_len is not None:
            tcfg = replace(tcfg, max_seq_len=max_seq_len)
    return tcfg


def eval_cfg(cond='standard', trcfg=None):
    """The EvalConfig of one unit: shared, except that the distillation
    condition takes the teacher's decoding budget when it has one."""
    if cond == 'distill' and trcfg is not None:
        _, max_new = distill_budget(trcfg)
        if max_new is not None:
            return replace(EVAL, max_new_tokens=max_new)
    return EVAL


def unit_list(seeds, conditions=CONDITIONS):
    """
    Every (condition, seed) to run, in queue order: the base model once, then
    each fine-tuned condition once per seed, condition by condition in
    `CONDITIONS` order -- which puts the distillation units last.
    """
    units = []
    for c in CONDITIONS:
        if c not in conditions:
            continue
        units += [('base', None)] if c == 'base' else [(c, s) for s in range(seeds)]
    return units


def _unit_cfg(cond, seed, dcfg, trcfg):
    """The (TrainConfig, EvalConfig) a unit is keyed by (seed 0 stands in for
    the base model, whose key ignores it)."""
    return (train_cfg(0 if seed is None else seed, cond, dcfg, trcfg),
            eval_cfg(cond, trcfg))


# ----------------------------------------------------------------------------
# plotting style (mirrors the math task figures)
# ----------------------------------------------------------------------------

LABEL_FS = 14
TICK_FS = 12
LEGEND_FS = 8

# The untuned model is the grey reference; the fine-tuned conditions take the
# first three colours of the categorical palette the other figures use.
COND_COLOR = {'base': '0.6', 'standard': '#2a78d6', 'restated': '#e87ba4',
              'distill': '#eda100'}


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

def _label(cond, seed):
    return f'[{cond}' + ('' if seed is None else f' s{seed}') + ']'


def _unit_jobs(units, dcfg, trcfg, device, force, progress_pos=0):
    """
    Run each (condition, seed) unit on `device`: train the adapter if needed,
    then decode the test set with the model still loaded. A unit that raises
    is recorded as ``{'cond', 'seed', 'error'}`` and the next one starts.
    Returns the results.
    """
    import torch
    out = []
    for cond, seed in units:
        TR, EV = _unit_cfg(cond, seed, dcfg, trcfg)
        r = None if force else cached_unit(cond, dcfg, MODEL, TR, EV, trcfg)
        if r is not None:
            print(f'{_label(cond, seed)} cached', flush=True)
            out.append(r)
            continue
        # the model is loaded lazily inside run_unit, so a unit whose decode is
        # cached but whose score is not never occupies the GPU
        _pack = {}

        def get_model(_cond=cond, _TR=TR):
            if 'p' not in _pack:
                adapter = None if _cond == 'base' else train_adapter(
                    _cond, dcfg, MODEL, _TR, trcfg, device=device, force=force,
                    log=lambda m: print(m, flush=True), progress_pos=progress_pos)
                _pack['p'] = load_for_eval(adapter, MODEL, _TR, device=device)
            return _pack['p']

        try:
            res = run_unit(cond, dcfg, MODEL, TR, EV, trcfg, device=device, force=force,
                           get_model=get_model, log=lambda m: print(m, flush=True),
                           progress_pos=progress_pos)
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


def _worker(queue, dcfg, trcfg, device, force, progress_pos):
    """Pull units off the shared queue until it is drained."""
    out = []
    while True:
        unit = queue.get()
        if unit == _STOP:
            break
        cond, seed = unit
        print(f'[{device}] starting {cond}' + ('' if seed is None else f' seed {seed}'),
              flush=True)
        out.extend(_unit_jobs([unit], dcfg, trcfg, device, force, progress_pos))
    print(f'[{device}] done', flush=True)
    return out


def run_all(units, dcfg, trcfg, devices, force):
    """
    Run every unit across `devices`.

    Work is handed out through a shared queue rather than pre-assigned, so a
    GPU that finishes early immediately picks up the next unit instead of
    idling; the queue holds the units in `unit_list` order. The problem sets
    are built here, in the parent, so the workers never race to download and
    cache the same split.
    """
    t0 = time.time()
    n_tr = len(get_problems(dcfg, 'train'))
    n_te = len(get_problems(dcfg, 'test'))
    print(f'datasets ready: {n_tr} train / {n_te} test ({time.time() - t0:.0f}s)', flush=True)

    if len(devices) == 1:
        return _unit_jobs(units, dcfg, trcfg, devices[0], force)

    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    queue = manager.Queue()
    for unit in units:
        queue.put(unit)
    for _ in devices:
        queue.put(_STOP)

    results = []
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as ex:
        futs = [ex.submit(_worker, queue, dcfg, trcfg, d, force, i)
                for i, d in enumerate(devices)]
        for f in futs:
            results.extend(f.result())
    return results


def pending_units(units, dcfg, trcfg, force):
    """The units that still have to train or decode."""
    return [(c, s) for c, s in units
            if force or cached_unit(c, dcfg, MODEL, *_unit_cfg(c, s, dcfg, trcfg), trcfg) is None]


def print_plan(units, dcfg, trcfg, devices, force):
    """
    What the run will actually do: which adapters must be trained, which
    evaluations are missing, and how the work lands on the GPUs. Printed
    before anything loads so the cost is visible up front.
    """
    todo = pending_units(units, dcfg, trcfg, force)
    to_train = [(c, s) for c, s in todo if c != 'base' and (
        force or not adapter_is_cached(c, dcfg, MODEL, _unit_cfg(c, s, dcfg, trcfg)[0], trcfg))]
    n_test = dcfg.n_test if dcfg.n_test is not None else len(get_problems(dcfg, 'test'))
    conds = [c for c in CONDITIONS if any(u == c for u, _ in units)]
    seeds = sorted({s for _, s in units if s is not None})
    print('-' * 66)
    print(f'conditions     : {", ".join(conds)}')
    if 'restated' in conds:
        s = trace_summary(trcfg, dcfg)
        print(f'restated traces: author={trcfg.author} -- ' + (
            f'{s["n"]} valid, {s["items_per_ledger"]:.1f} items per ledger, '
            f'{s["char_ratio"]:.2f}x the standard length' if s
            else 'MISSING OR INVALID (the run will stop when it loads them)'))
    if 'distill' in conds:
        d = distill_summary(trcfg, dcfg)
        seq, new = distill_budget(trcfg)
        bud = ('default budgets' if seq is None and new is None
               else f'budgets: {seq} train tokens, {new} new tokens')
        print(f'distill traces : teacher={trcfg.teacher} -- ' + (
            f'{d["n"]} valid ({d["n_filled"]} ground-truth fills), '
            f'{d["char_ratio"]:.2f}x the standard length; {bud}' if d
            else 'MISSING OR INVALID (the condition will be skipped)'))
    print(f'train subset   : {dcfg.n_train:,} GSM8K train problems (seed {dcfg.train_seed}) '
          f'x {PASSES} passes = {dcfg.n_train * PASSES:,} samples per adapter')
    print(f'test set       : {n_test:,} GSM8K test problems, greedy, '
          f'<= {EVAL.max_new_tokens} new tokens, batch {EVAL.batch_size}')
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
              f'(<={-(-len(todo) // len(devices))} per device); distill units last')
    print(f'  per-GPU train batch {TRAIN.batch_size} x {TRAIN.grad_accum} accum '
          f'= {TRAIN.batch_size * TRAIN.grad_accum} effective; eval batch {EVAL.batch_size}')
    print('rough cost     : on an H100 or B200 ~1-3 min per adapter and ~3-8 min per')
    print('                 evaluation of the full test set, so ~5-10 min per unit and')
    print('                 ~1-1.5 h for the 10 default units on one GPU (a 4090: ~3x that)')
    print('-' * 66, flush=True)


def load_cached(units, dcfg, trcfg):
    """Every cached unit, for --plot-only."""
    out = []
    for cond, seed in units:
        r = cached_unit(cond, dcfg, MODEL, *_unit_cfg(cond, seed, dcfg, trcfg), trcfg)
        if r is not None:
            out.append(r)
    return out


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------

def _binomial_sem(p, n):
    """Standard error of a proportion estimated from `n` independent trials."""
    return (p * (1.0 - p) / n) ** 0.5 if n else 0.0


def summarise(results):
    """
    Per condition: the per-seed accuracies, their mean, and a standard error.

    With more than one seed the SEM is over seeds -- the spread that matters
    when asking whether two fine-tunes differ. The base model has no seed
    (greedy decoding is deterministic), so its error is the binomial SEM over
    the test problems; a single-seed condition falls back to the same.
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
                         answered=float(np.mean([r['answered'] for r in rows])),
                         capped=float(np.mean([r['capped'] for r in rows])),
                         mean_tokens=float(np.mean([r['mean_tokens'] for r in rows])),
                         n_eval=rows[0]['n_eval'])
    return out


def plot_accuracy_bars(results, name='gsm8k_accuracy', subdir=''):
    """
    Test accuracy per condition as bars.

    Each bar is the mean over seeds, with the SEM of `summarise` as a black
    error bar and the individual seeds as small dots, so the reader sees both
    the estimate and how much a single training run moves it.
    """
    stats = summarise(results)
    conds = [c for c in CONDITIONS if c in stats]
    if not conds:
        return None
    fig, ax = plt.subplots(figsize=(3, 2.5))
    xs = np.arange(len(conds))
    for x, c in zip(xs, conds):
        s = stats[c]
        ax.bar(x, s['mean'], width=0.62, color=COND_COLOR[c], zorder=2)
        ax.errorbar(x, s['mean'], yerr=s['sem'], fmt='none', ecolor='black',
                    elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
        if len(s['accs']) > 1:
            jit = np.linspace(-0.13, 0.13, len(s['accs']))
            ax.plot(x + jit, s['accs'], ls='none', marker='o', ms=2.8, color='black',
                    alpha=0.75, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([COND_LABEL[c] for c in conds])
    ax.set_xlim(-0.6, len(conds) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Test Accuracy', fontsize=LABEL_FS)
    _style_axis(ax)
    return _save(fig, name, subdir)


def print_summary(results, trcfg):
    """The table behind the figure."""
    stats = summarise(results)
    if not stats:
        return
    print(f'\ntest accuracy (ledgers by {trcfg.author}; teacher {trcfg.teacher})')
    print('  sem       over seeds when there are several, else binomial over the test set')
    print('  answered  fraction of completions that wrote a #### line')
    print('  capped    fraction that hit the token cap without finishing (scored wrong)')
    print('  tokens    mean generated tokens per completion')
    print()
    print('  condition   seeds      acc      sem   answered   capped   tokens')
    for cond, s in stats.items():
        print(f'  {cond:<10} {len(s["accs"]):>6}   {s["mean"]:6.3f}   {s["sem"]:6.3f}   '
              f'{s["answered"]:8.3f}   {s["capped"]:6.3f}   {s["mean_tokens"]:6.0f}')
    for cond, s in stats.items():
        if len(s['accs']) > 1:
            print(f'  {cond} per seed: ' + '  '.join(
                f's{sd}={a:.3f}' for sd, a in zip(s['seeds'], s['accs'])))
    for a, b in (('restated', 'standard'), ('distill', 'standard'), ('restated', 'distill')):
        if a in stats and b in stats:
            d = stats[a]['mean'] - stats[b]['mean']
            e = (stats[a]['sem'] ** 2 + stats[b]['sem'] ** 2) ** 0.5
            print(f'  {a} - {b} = {d:+.3f} (+/- {e:.3f})')


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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--devices', default=None,
                    help='comma-separated torch devices, e.g. cuda:0,cuda:1,...')
    ap.add_argument('--seeds', type=int, default=SEEDS,
                    help='training seeds per fine-tuned condition')
    ap.add_argument('--conditions', default=','.join(CONDITIONS),
                    help=f'comma-separated subset of {", ".join(CONDITIONS)}')
    ap.add_argument('--author', choices=AUTHORS, default=TraceConfig.author,
                    help="who wrote the ledger lines: 'claude' (traces_ledger.txt) "
                         "or 'rule' (mechanical)")
    ap.add_argument('--teacher', choices=sorted(TEACHERS), default=TraceConfig.teacher,
                    help='whose solutions the distill condition trains on (distill.py)')
    ap.add_argument('--fill', choices=FILLS, default=TraceConfig.fill,
                    help='distill: problems without a usable teacher trace take the '
                         'ground-truth solution, or are dropped')
    ap.add_argument('--n-train', type=int, default=DATA.n_train)
    ap.add_argument('--n-test', type=int, default=DATA.n_test,
                    help='subsample the test set (default: all 1,319 problems)')
    ap.add_argument('--force', action='store_true', help='retrain and re-evaluate')
    ap.add_argument('--plot-only', action='store_true',
                    help='build the figure from cached results only')
    args = ap.parse_args()

    conditions = tuple(c.strip() for c in args.conditions.split(',') if c.strip())
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        ap.error(f'unknown condition(s) {unknown}; choose from {", ".join(CONDITIONS)}')
    devices = args.devices.split(',') if args.devices else _default_devices()
    dcfg = replace(DATA, n_train=args.n_train, n_test=args.n_test)
    trcfg = TraceConfig(author=args.author, teacher=args.teacher, fill=args.fill)
    units = unit_list(args.seeds, conditions)
    skipped, failed = [], []

    if args.plot_only:
        results = load_cached(units, dcfg, trcfg)
        if not results:
            print('no cached results; nothing to plot')
            return
    else:
        print_plan(units, dcfg, trcfg, devices, args.force)
        todo = pending_units(units, dcfg, trcfg, args.force)
        if any(c == 'restated' for c, _ in todo):
            # validated once, here, so a bad trace file fails before any worker
            # (and any GPU) is involved
            load_traces(trcfg, dcfg)
        if any(c == 'distill' for c, _ in todo):
            try:
                load_distill_traces(trcfg, dcfg)
            except (FileNotFoundError, ValueError) as e:
                print(f'\nWARNING: the distill condition is skipped this run -- {e}\n', flush=True)
                units = [(c, s) for c, s in units if c != 'distill']
                skipped.append('distill')
        t0 = time.time()
        results = run_all(units, dcfg, trcfg, devices, args.force)
        print(f'\nwall clock: {(time.time() - t0) / 60:.1f} min')

    failed = [r for r in results if r is not None and 'error' in r]
    results = [r for r in results if r is not None and 'error' not in r]
    path = plot_accuracy_bars(results, name=f'gsm8k_accuracy_{args.author}')
    if path:
        print(f'\nwrote {path}.pdf/.png')
    print_summary(results, trcfg)
    if failed:
        print('\nFAILED units (cached work is kept; rerun to retry them):')
        for r in failed:
            print(f'  {_label(r["cond"], r["seed"])} {r["error"]}')
    if skipped:
        print(f'\nskipped conditions: {", ".join(skipped)}')
    if failed or skipped:
        sys.exit(1)


if __name__ == '__main__':
    main()
