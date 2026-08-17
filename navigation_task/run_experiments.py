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

from train_eval import run_unit, ModelConfig, TrainConfig


HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')


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
        grids=[5, 6, 7, 8, 9, 10, 11, 12],
        num_train=8000,            # fixed budget: standard's O(M^2) DAG outgrows
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


def run_all(profile, devices, force):
    """
    Run every (grid, interval, seed) unit, partitioned across `devices` (one
    process per device). A single progress bar accumulates completed points
    across all devices. Returns the flat list of per-unit result dicts.
    """
    jobs = [(m, iv, seed) for m in profile['grids']
            for iv in _intervals_for(m) for seed in profile['seeds']]
    total = len(jobs)
    # partition round-robin so each device gets a balanced share
    buckets = {d: [] for d in devices}
    for i, job in enumerate(jobs):
        buckets[devices[i % len(devices)]].append(job)

    results = []
    if len(devices) == 1:
        with tqdm(total=total, desc='points') as bar:
            for (m, iv, seed) in buckets[devices[0]]:
                results.extend(_run_device_jobs([(m, iv, seed)], devices[0], profile, force))
                bar.update(1)
        return results

    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    q = manager.Queue()
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as ex, \
            tqdm(total=total, desc='points') as bar:
        futs = [ex.submit(_run_device_jobs, buckets[d], d, profile, force, q)
                for d in devices if buckets[d]]
        for _ in range(total):                       # one token per completed unit
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


# dark -> light fill for the emission interval: De Bruijn (k=1) darkest, larger k
# lighter, standard (no re-emission) hollow.
_INTERVAL_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'interval', ['#0a2340', '#9dc3ec'])


def plot_accuracy_vs_edges(results, name='accuracy_vs_edges'):
    """
    Held-out accuracy vs exact minimal-DAG edges. Each grid is one task: its
    emission-interval conditions -- De Bruijn (state every step) through every-m
    steps, then standard (state only at the ends) -- lie along a translucent
    dashed connector, ordered by edge count. Marker fill is shaded by interval
    (De Bruijn darkest, larger intervals lighter, standard hollow). Points are
    seed means, error bars +/- SEM.
    """
    fig, ax = plt.subplots(figsize=(3, 2.5))
    grids = sorted({r['m'] for r in results})
    edge_col = CURVE_COLORS[0]
    ivals = sorted({r['interval'] for r in results if r['interval'] is not None})
    kmax = max(ivals) if ivals else 1

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
        for (iv, x, mean, sem, _n) in pts:
            ax.errorbar(x, mean, yerr=sem, fmt='o', ms=4.2, mew=1.0,
                        markerfacecolor=fill(iv), markeredgecolor=edge_col,
                        ecolor=edge_col, elinewidth=0.7, capsize=0, zorder=3)

    ax.set_xscale('log')
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Edges in minimal DAG', fontsize=LABEL_FS)
    ax.set_ylabel('Test accuracy', fontsize=LABEL_FS)
    _style_axis(ax)

    # slim colorbar mapping fill shade -> emission interval k
    sm = plt.cm.ScalarMappable(cmap=_INTERVAL_CMAP,
                               norm=mcolors.Normalize(vmin=1, vmax=kmax))
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r'interval $k$', fontsize=LEGEND_FS + 2)
    cb.ax.tick_params(labelsize=TICK_FS - 3)

    _legend(ax, [
        Line2D([], [], color=edge_col, marker='o', ls='', ms=4,
               mfc=_INTERVAL_CMAP(0.0), label=r'De Bruijn ($k{=}1$)'),
        Line2D([], [], color=edge_col, marker='o', ls='', ms=4, mfc='none',
               label='Standard'),
    ], loc='lower left')
    return _save(fig, name)


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
    path = plot_accuracy_vs_edges(results)
    print('wrote', path + '.pdf/.png')
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
