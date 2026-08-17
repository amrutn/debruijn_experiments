"""
Run the grid-navigation experiment and make the accuracy-vs-edges figure.

A decoder-only Transformer is trained from scratch on shortest-path traces of an
``m x m`` grid under two encodings (see `generate_data`): the standard trace
``s N E N g`` and the De Bruijn trace ``s N c1 E c2 N g`` that writes the current
cell before every move. Both are trained at the *same fixed budget* (number of
sampled traces and epochs) and evaluated on held-out prompts by greedy-decoding a
path and checking it is a shortest ``s -> g`` route (`train_eval`).

Figure (size (3, 2.5) pdf+png in figures/)
------
accuracy_vs_edges           Held-out shortest-path accuracy vs the exact number
                            of edges in the encoding's minimal DAG. Each grid
                            size is one task: its De Bruijn point (filled) and
                            standard point (hollow) are joined by a translucent
                            dashed line, as remapped/plain pairs are joined in the
                            synthetic experiment. The De Bruijn point sits to the
                            left (fewer edges, O(M) world model) of the standard
                            point (O(M^2) reasoning DAG); at a fixed budget it also
                            sits higher. Points are the seed mean, error bars SEM.

Caching
-------
Every (grid, encoding, seed) unit writes its point to cache/exp_results/<hash>.json
(see `train_eval.run_unit`), so an interrupted or repeated run reuses finished
units. Pass --force to recompute.

Usage
-----
    python run_experiments.py                       # auto profile + devices
    python run_experiments.py --profile full
    python run_experiments.py --profile laptop
    python run_experiments.py --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5
    python run_experiments.py --force
"""

import os
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

from scipy.optimize import least_squares

from train_eval import run_unit, cached_unit, ModelConfig, TrainConfig
from generate_data import NavGrid, max_tokens, mean_tokens


HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')

# grids shown in the figures (the sweep may train more; these read cleanest)
PLOT_GRIDS = (6, 7, 8, 9)


# ----------------------------------------------------------------------------
# plotting style (mirrors synthetic_experiment/run_experiments.py so the two
# sets of figures look consistent)
# ----------------------------------------------------------------------------

CURVE_COLORS = ['#2a78d6', '#008300', '#e87ba4', '#eda100']   # blue, green, magenta, gold
LABEL_FS = 14
TICK_FS = 12
LEGEND_FS = 7


def _log_tick_fmt(v, _):
    if v <= 0:
        return ''
    e = int(np.floor(np.log10(v) + 1e-9))
    m = v / 10 ** e
    if abs(m - 1) < 1e-6:
        return rf'$10^{{{e}}}$'
    if abs(m - round(m)) < 1e-6:
        return rf'${round(m):d}{{\times}}10^{{{e}}}$'
    return rf'${m:.1f}{{\times}}10^{{{e}}}$'


def _style_axis(ax):
    ax.tick_params(labelsize=TICK_FS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for axis in (ax.xaxis, ax.yaxis):
        if axis.get_scale() != 'log':
            continue
        lo, hi = axis.get_view_interval()
        if not (lo > 0 and hi > lo):
            continue
        emin, emax = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
        ticks = []
        for subs in ((1.0,), (1.0, 3.0), (1.0, 2.0, 5.0), (1.0, 2.0, 3.0, 5.0),
                     (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0)):
            cand = [k * 10.0 ** e for e in range(emin, emax + 1)
                    for k in subs if lo <= k * 10.0 ** e <= hi]
            if len(cand) > 7:
                continue
            if len(cand) >= 3:
                ticks = cand
                break
            if len(cand) > len(ticks):
                ticks = cand
        axis.set_major_locator(mticker.FixedLocator(ticks))
        axis.set_major_formatter(mticker.FuncFormatter(_log_tick_fmt))
        axis.set_minor_formatter(mticker.NullFormatter())


def _legend(ax, handles, loc='lower left', bbox=None):
    leg = ax.legend(handles=handles, fontsize=LEGEND_FS, frameon=False,
                    loc=loc, bbox_to_anchor=bbox, borderaxespad=0.15, handlelength=1.6,
                    labelspacing=0.25, handletextpad=0.5)
    leg.set_zorder(20)
    frame = leg.get_frame()
    frame.set_edgecolor('0.7')
    frame.set_facecolor('white')
    frame.set_alpha(1.0)
    frame.set_linewidth(0.7)
    return leg


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path + '.pdf', bbox_inches='tight')
    fig.savefig(path + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# profiles: the sweep. 'full' is the cluster sweep (6x4090); 'laptop' is a quick
# partial run to verify the pipeline on mps/cpu.
# ----------------------------------------------------------------------------

# A profile sets the training-set size with EITHER `num_train` (a fixed budget,
# identical for every grid) OR `samples_per_cell` (budget = samples_per_cell * M,
# scaling with grid size). Give exactly one (if both are set, `samples_per_cell`
# wins, per run_unit).
PROFILES = {
    'laptop': dict(
        grids=[3, 4, 5, 6],
        num_train=8000,             # fixed training budget (same for every grid)
        epochs=6,
        test_frac=0.2,              # fixed held-out fraction of prompts
        eval_cap=300,
        seeds=[0, 1],
        model=dict(d_model=96, num_heads=4, num_layers=2),
        train=dict(lr=2e-3, weight_decay=1e-2, batch_size=256),
    ),
    'full': dict(
        grids=[6, 7, 8, 9],
        num_train=16000,            # fixed budget: standard's O(M^2) DAG outgrows
                                    # it as the grid enlarges, De Bruijn's O(M) does not
        epochs=6,
        test_frac=0.2,
        eval_cap=2000,
        seeds=[0, 1, 2, 3, 4],
        model=dict(d_model=128, num_heads=4, num_layers=4),
        train=dict(lr=1e-3, weight_decay=1e-2, batch_size=512),
    ),
}


# ----------------------------------------------------------------------------
# running the units, one process pinned per device
# ----------------------------------------------------------------------------

def _intervals_for(m):
    """Emission conditions for an m x m grid: De Bruijn (1) .. every-m, then
    standard (None). k=1 sees the state every step; k=m every m steps; None never
    (only the start/goal bookends)."""
    return list(range(1, m + 1)) + [None]


def _run_device_jobs(jobs, device, profile, force, q=None):
    """Run a list of (m, interval, seed) units serially on one device, putting a
    token on `q` (if given) after each so a shared progress bar can advance."""
    mcfg = ModelConfig(**profile['model'])
    out = []
    for (m, interval, seed) in jobs:
        tcfg = TrainConfig(device=device, seed=seed, **profile['train'])
        r = run_unit(m=m, interval=interval, num_train=profile.get('num_train'),
                     samples_per_cell=profile.get('samples_per_cell'),
                     test_frac=profile['test_frac'], eval_cap=profile['eval_cap'],
                     epochs=profile['epochs'], seed=seed, mcfg=mcfg, tcfg=tcfg,
                     force=force)
        out.append(r)
        if q is not None:
            q.put(1)
    return out


def _cached_or_none(job, device, profile):
    """Cache hit for a (m, interval, seed) job on `device`, or None. Builds no
    model and touches no device (the cache key includes the device via tcfg, so
    the assigned device is passed to match how the unit was written)."""
    m, iv, seed = job
    mcfg = ModelConfig(**profile['model'])
    tcfg = TrainConfig(device=device, seed=seed, **profile['train'])
    return cached_unit(m=m, interval=iv, num_train=profile.get('num_train'),
                       samples_per_cell=profile.get('samples_per_cell'),
                       test_frac=profile['test_frac'], eval_cap=profile['eval_cap'],
                       epochs=profile['epochs'], seed=seed, mcfg=mcfg, tcfg=tcfg)


def run_all(profile, devices, force):
    """
    Run every (grid, interval, seed) unit, partitioned across `devices` (one
    process per device). The cache is checked up front (unless `force`): units
    already computed are loaded here and only the missing ones are dispatched, so
    a fully-cached sweep spawns no worker processes and initialises no GPU. A
    single progress bar accumulates the remaining points across all devices.
    Returns the flat list of per-unit result dicts.
    """
    jobs = [(m, iv, seed) for m in profile['grids']
            for iv in _intervals_for(m) for seed in profile['seeds']]
    # round-robin device assignment (kept identical for the cache check and the run)
    assigned = [(job, devices[i % len(devices)]) for i, job in enumerate(jobs)]

    # up-front cache check: collect hits, queue only the misses
    results, todo = [], []
    for job, dev in assigned:
        r = None if force else _cached_or_none(job, dev, profile)
        (results.append(r) if r is not None else todo.append((job, dev)))
    print(f'cache: {len(results)}/{len(jobs)} units cached, {len(todo)} to run')
    if not todo:
        return results

    buckets = {d: [] for d in devices}
    for job, dev in todo:
        buckets[dev].append(job)

    if len(devices) == 1:
        with tqdm(total=len(todo), desc='points') as bar:
            for job in buckets[devices[0]]:
                results.extend(_run_device_jobs([job], devices[0], profile, force))
                bar.update(1)
        return results

    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    q = manager.Queue()
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as ex, \
            tqdm(total=len(todo), desc='points') as bar:
        futs = [ex.submit(_run_device_jobs, buckets[d], d, profile, force, q)
                for d in devices if buckets[d]]
        for _ in range(len(todo)):                   # one token per completed unit
            q.get()
            bar.update(1)
        for f in futs:
            results.extend(f.result())
    return results


# ----------------------------------------------------------------------------
# aggregation + figure
# ----------------------------------------------------------------------------

def _agg(results, m, interval):
    """Seed mean/SEM of (edges, accuracy) for one (grid, interval)."""
    grp = [r for r in results if r['m'] == m and r['interval'] == interval]
    if not grp:
        return None
    edges = float(np.mean([r['edges'] for r in grp]))
    accs = np.array([r['accuracy'] for r in grp], float)
    n = len(accs)
    sem = float(accs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return edges, float(accs.mean()), sem, n


def fit_shared(results, restarts=120, seed=0):
    """
    Mechanistic fit ``accuracy = A * edges^(-p) + B * rho^length + c(grid)``:
    a *learnability* channel plus a *decode-reliability* channel, summed, with a
    per-grid baseline. ``A, p, B, rho`` are shared across grids; ``c`` is a per-grid
    additive intercept. ``length`` is the *mean* trace length over the prompt
    distribution (`generate_data.mean_tokens`).

    * ``A * edges^(-p)`` -- learnability: at a fixed training budget the fraction of
      the ``O(edges)`` transitions the model has acquired falls off with the DAG
      size (fitted ``p ~ 1/2``). Large at the De Bruijn end (few edges).
    * ``B * rho^length`` -- decode reliability: each of the ``length`` autoregressive
      tokens is produced correctly with probability ``rho``, so a whole trace
      survives with probability ``rho^length`` (fitted ``rho ~ 0.9``). Large at the
      standard end (shortest trace).

    The two channels are high at opposite ends of the interval sweep, so their sum
    dips in the middle and recovers -- capturing the U-shape. The reliability term
    must enter *additively*: as a multiplicative factor the model is log-linear in
    (log edges, length) and hence monotone along the connector (no U). The per-grid
    intercept absorbs the grid-size baseline. ``edges`` is centred in log and
    ``length`` in raw units for conditioning; ``p`` and ``rho`` are scale-invariant.

    Returns
    -------
    dict
        {'A', 'p', 'B', 'rho', 'c': {m: intercept}, 'r2', 'predict'} where
        ``predict(m, length, edges) -> accuracy``.
    """
    T = np.array([mean_tokens(NavGrid(r['m']), r['interval']) for r in results], float)
    E = np.array([r['edges'] for r in results], float)
    y = np.array([r['accuracy'] for r in results], float)
    ms = [r['m'] for r in results]
    grids = sorted(set(ms))
    gi = {g: i for i, g in enumerate(grids)}
    gidx = np.array([gi[m] for m in ms])
    lEm, Tm = np.log(E).mean(), T.mean()
    lEn, Tn = np.log(E) - lEm, T - Tm

    def resid(p):
        A, pw, B, r = p[:4]
        return A * np.exp(-pw * lEn) + B * np.exp(r * Tn) + p[4:][gidx] - y

    rng = np.random.default_rng(seed)
    lo = np.array([-50, -2, -50, -1] + [-50] * len(grids), float)
    hi = np.array([50, 4, 50, 0.2] + [50] * len(grids), float)
    best = None
    for _ in range(restarts):
        s = least_squares(resid, rng.uniform(lo, hi), bounds=(lo, hi), max_nfev=8000)
        if best is None or s.cost < best.cost:
            best = s
    A, pw, B, r = best.x[:4]
    cvec = best.x[4:]
    r2 = 1.0 - (best.fun ** 2).sum() / ((y - y.mean()) ** 2).sum()

    def predict(m, Tv, Ev):
        return A * np.exp(-pw * (np.log(Ev) - lEm)) + B * np.exp(r * (Tv - Tm)) + cvec[gi[m]]

    return dict(A=float(A), p=float(pw), B=float(B), rho=float(np.exp(r)),
                c={g: float(cvec[gi[g]]) for g in grids}, r2=float(r2),
                predict=predict)


# dark -> light fill for the emission interval: De Bruijn (k=1) darkest, larger k
# lighter, standard (no re-emission) hollow.
_INTERVAL_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'interval', ['#0a2340', '#9dc3ec'])


def plot_accuracy_vs_edges(results, name='accuracy_vs_edges'):
    """
    Held-out accuracy vs exact minimal-DAG edges (log2 x). Each grid is one task:
    its emission-interval conditions -- De Bruijn (state every step) through
    every-m steps, then standard (state only at the ends) -- lie along a
    translucent dashed connector, ordered by edge count. Marker fill is shaded by
    interval (De Bruijn darkest, standard hollow). The shared mechanistic fit
    (`fit_shared`, learnability in edges + reliability in mean trace length) is
    overlaid. Points are seed means, error bars +/- SEM.
    """
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    grids = sorted({r['m'] for r in results})
    edge_col = CURVE_COLORS[0]
    ivals = sorted({r['interval'] for r in results if r['interval'] is not None})
    kmax = max(ivals) if ivals else 1
    fit = fit_shared(results)

    def fill(interval):
        if interval is None:
            return 'none'
        frac = 0.0 if kmax <= 1 else (interval - 1) / (kmax - 1)
        return _INTERVAL_CMAP(frac)

    for m in grids:
        pts = []
        for iv in _intervals_for(m):
            a = _agg(results, m, iv)
            if a:
                pts.append((iv,) + a)                # (interval, edges, mean, sem, n)
        pts.sort(key=lambda p: p[1])                 # order by edge count
        if len(pts) >= 2:                            # connector through all points
            ax.plot([p[1] for p in pts], [p[2] for p in pts],
                    ls='--', lw=0.8, color=edge_col, alpha=0.4, zorder=1)
        fx = [p[1] for p in pts]                      # fitted curve (fit uses edges & mean length)
        fy = [fit['predict'](m, mean_tokens(NavGrid(m), p[0]), p[1]) for p in pts]
        if len(fx) >= 2:
            ax.plot(fx, fy, ls='-', lw=1.2, color=edge_col, alpha=1.0, zorder=2)
        for (iv, x, mean, sem, _n) in pts:
            ax.errorbar(x, mean, yerr=sem, fmt='o', ms=4.2, mew=1.0,
                        markerfacecolor=fill(iv), markeredgecolor=edge_col,
                        ecolor=edge_col, elinewidth=0.7, capsize=0, zorder=3)

    ax.set_xscale('log')                             # log10 x
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel('Edges in minimal DAG', fontsize=LABEL_FS)
    ax.set_ylabel('Test accuracy', fontsize=LABEL_FS)
    _style_axis(ax)                                  # places base-10 log ticks
    # fitted law with raw fitted numbers (c_m is per-grid, kept symbolic)
    ax.text(0.97, 0.97,
            rf'${fit["A"]:.2f}\,(\mathrm{{edges}})^{{-{fit["p"]:.2f}}}'
            rf'+{fit["B"]:.2f}{{\cdot}}{fit["rho"]:.2f}^{{\mathrm{{length}}}}+c_m$'
            + '\n' + rf'$R^2={fit["r2"]:.2f}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=LEGEND_FS,
            color=edge_col)

    # slim colorbar mapping fill shade -> emission interval k
    sm = plt.cm.ScalarMappable(cmap=_INTERVAL_CMAP,
                               norm=mcolors.Normalize(vmin=1, vmax=kmax))
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r'Interval $k$', fontsize=LEGEND_FS + 2)
    cb.ax.tick_params(labelsize=TICK_FS - 3)

    _legend(ax, [
        Line2D([], [], color=edge_col, marker='o', ls='', ms=4,
               mfc=_INTERVAL_CMAP(0.0), label=r'De Bruijn ($k{=}1$)'),
        Line2D([], [], color=edge_col, marker='o', ls='', ms=4, mfc='none',
               label='Standard'),
    ], loc='lower left')
    return _save(fig, name), fit


def plot_facets(results, name='accuracy_vs_edges_facets', ncols=4):
    """
    Small-multiples version: one panel per grid, so each grid's interval sweep
    (and its U-shape) fills its own axes without the 8 connectors overlapping in a
    single frame. Same shading (interval, De Bruijn darkest, standard hollow) and
    the shared mechanistic fit overlaid per panel. Returns (path, fit).
    """
    grids = sorted({r['m'] for r in results})
    nrows = (len(grids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.05 * ncols, 1.95 * nrows),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    edge_col = CURVE_COLORS[0]
    ivals = sorted({r['interval'] for r in results if r['interval'] is not None})
    kmax = max(ivals) if ivals else 1
    fit = fit_shared(results)

    def fill(iv):
        return 'none' if iv is None else _INTERVAL_CMAP(0.0 if kmax <= 1 else (iv - 1) / (kmax - 1))

    for ax, m in zip(axes, grids):
        pts = []
        for iv in _intervals_for(m):
            a = _agg(results, m, iv)
            if a:
                pts.append((iv,) + a)
        pts.sort(key=lambda p: p[1])
        fx = [p[1] for p in pts]
        fy = [fit['predict'](m, mean_tokens(NavGrid(m), p[0]), p[1]) for p in pts]
        if len(fx) >= 2:
            ax.plot(fx, fy, ls='-', lw=1.3, color=edge_col, alpha=0.8, zorder=2)
        for (iv, x, mean, sem, _n) in pts:
            ax.errorbar(x, mean, yerr=sem, fmt='o', ms=3.6, mew=0.8,
                        markerfacecolor=fill(iv), markeredgecolor=edge_col,
                        ecolor=edge_col, elinewidth=0.6, capsize=0, zorder=3)
        ax.set_xscale('log')
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(rf'$m={m}$', fontsize=TICK_FS)
        ax.tick_params(labelsize=TICK_FS - 3)
        _style_axis(ax)
    for ax in axes[len(grids):]:
        ax.axis('off')

    fig.supxlabel('Edges in minimal DAG', fontsize=LABEL_FS)
    fig.supylabel('Test accuracy', fontsize=LABEL_FS)
    sm = plt.cm.ScalarMappable(cmap=_INTERVAL_CMAP, norm=mcolors.Normalize(vmin=1, vmax=kmax))
    cb = fig.colorbar(sm, ax=axes.tolist(), pad=0.01, fraction=0.03)
    cb.set_label(r'Interval $k$', fontsize=LEGEND_FS + 2)
    cb.ax.tick_params(labelsize=TICK_FS - 3)
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png', dpi=300)
    plt.close(fig)
    return path, fit


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


def _default_profile(devices):
    return 'full' if any('cuda' in d for d in devices) else 'laptop'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', choices=list(PROFILES), default=None)
    ap.add_argument('--devices', default=None,
                    help='comma-separated torch devices, e.g. cuda:0,cuda:1,...')
    ap.add_argument('--force', action='store_true', help='recompute cached units')
    args = ap.parse_args()

    devices = args.devices.split(',') if args.devices else _default_devices()
    profile_name = args.profile or _default_profile(devices)
    profile = PROFILES[profile_name]
    print(f'profile={profile_name}  devices={devices}  grids={profile["grids"]}')

    results = run_all(profile, devices, args.force)
    plot_res = [r for r in results if r['m'] in PLOT_GRIDS] or results
    path, fit = plot_accuracy_vs_edges(plot_res)
    fpath, _ = plot_facets(plot_res)
    print('wrote', path + '.pdf/.png', 'and', fpath + '.pdf/.png')
    print(f'fit: acc = {fit["A"]:.3f}*edges^-{fit["p"]:.3f} + {fit["B"]:.3f}*{fit["rho"]:.3f}^len'
          f' + c(grid)   R^2={fit["r2"]:.3f}')
    print('  c_m: ' + '  '.join(f'm={m}:{c:+.3f}' for m, c in sorted(fit['c'].items())))
    # brief text summary: De Bruijn (k=1) and standard endpoints per grid
    for m in sorted({r['m'] for r in results}):
        row = []
        for iv, lbl in ((1, 'debruijn'), (None, 'standard')):
            a = _agg(results, m, iv)
            if a:
                row.append(f'{lbl}: edges={a[0]:.0f} acc={a[1]:.3f}')
        print(f'  m={m:2d}  ' + '   '.join(row))


if __name__ == '__main__':
    main()
