"""
Run the synthetic De Bruijn training experiments and make the figures.

A decoder-only transformer is trained from scratch (one epoch) on paths sampled
from D_pi = F[B(V,n); pi, N, S]. There is no prompt: the model has to learn
which subset of the vocabulary is legal at each state, measured by `illegal_mass`
and `path_coverage` (see train_eval.py). Two criteria mark "has learned it":
illegal_mass < 0.05 and path_coverage > 0.95.

Figures (each a publication-size (3, 2.5) pdf+png in figures/)
-------
samples_vs_edges            Scaling of the number of training samples needed to
                            reach each criterion (illegal_mass < 0.05 in blue,
                            path_coverage > 0.95 in green) vs the number
                            of transitions the model must learn. pi is a random
                            permutation. Filled markers are plain De Bruijn DAGs
                            (x = number of edges). Hollow markers are the same
                            DAGs with the token identities scrambled by a
                            position-dependent permutation of period `bin_width`;
                            such a permutation multiplies the number of *distinct*
                            (edge, position-block) transitions, and the x value is
                            that larger count. If sample cost is set by the number
                            of distinct transitions, plain and permuted points fall
                            on the same fitted line y = a*x (drawn dotted).

length_vs_Lmax            Same criterion, but the model is trained only on paths
                            of at most L edges (`sample_length_limited`) and tested
                            on the full path distribution. y is the smallest L that
                            still reaches the criterion; x is L_max (the DAG's
                            longest path). A dotted y = x line marks "must train on
                            ~full-length paths". No permutation. (Points are means
                            over the seeds, plus a linear fit.)

samples_vs_edges_digit_sum  samples_vs_edges for the digit-sum ordering of pi.
length_vs_Lmax_digit_sum  length_vs_Lmax for the digit-sum ordering of pi.

Caching
-------
Every (DAG, setting, seed) unit writes its computed point to
cache/exp_results/<hash>.json, so an interrupted or repeated run reuses finished
units instead of retraining. train_eval additionally caches the model weights.
Pass --force to recompute.

Usage
-----
    python run_experiments.py                  # auto: 'full' on CUDA else 'laptop'
    python run_experiments.py --profile full   # the cluster sweep (< 30 min on 2x4090)
    python run_experiments.py --profile laptop # quick, partial, for a laptop (mps/cpu)
    python run_experiments.py --devices cuda:0,cuda:1
    python run_experiments.py --figures samples_vs_edges --force
"""

import os
import sys
import json
import math
import time
import hashlib
import argparse
from dataclasses import replace, asdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from tqdm.auto import tqdm
import torch

from generate_data import build_dag, ReasoningGenerator
from train_eval import (
	ModelConfig, TrainConfig, resolve_device, run_experiment_1,
	build_model, train_one_epoch, evaluate, training_visits, build_full_tokens,
	_DTYPES, DEFAULT_PATH_RATIO,
)


HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
UNIT_CACHE_DIR = os.path.join(HERE, 'cache', 'exp_results')

# criteria the model must reach
ILLEGAL_CRIT = 0.05        # illegal_mass must drop below this
PATH_COV_CRIT = 0.95       # path_coverage must rise above this


# ----------------------------------------------------------------------------
# plotting style (mirrors theory/theorem1_theorem2.py so the two sets of figures
# look consistent)
# ----------------------------------------------------------------------------

CURVE_COLORS = ['#2a78d6', '#008300', '#e87ba4', '#eda100']   # blue, green, magenta, gold
CURVE_MARKERS = ['o', 's', '^', 'D']
LABEL_FS = 14
TICK_FS = 12
LEGEND_FS = 8


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
	# label log axes at explicit "nice" values: plain decades once the span
	# exceeds ~1.3 decades, else 1-2-3-5 within the range. (LogLocator falls back
	# to crowded linear ticks under one decade, so we place the ticks ourselves.)
	for axis in (ax.xaxis, ax.yaxis):
		if axis.get_scale() != 'log':
			continue
		lo, hi = axis.get_view_interval()
		if not (lo > 0 and hi > lo):
			continue
		emin, emax = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
		# try tick densities from sparse (decades) to dense; take the first that
		# yields >= 3 labels (<= 7), else the densest that fits -- keeps both wide
		# and sub-decade ranges readable without crowding
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


def _legend(ax, handles, loc='lower right'):
	leg = ax.legend(handles=handles, fontsize=LEGEND_FS, frameon=True,
					loc=loc, borderaxespad=0.15, handlelength=1.6,
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
# profiles: the sweep sizes. 'full' is the cluster sweep (2x4090, ~a few hours
# with the enlarged length training); 'laptop' is a quick partial run to verify
# the pipeline on mps/cpu.
# ----------------------------------------------------------------------------

PROFILES = {
	'laptop': dict(
		# (V, n, S, N) De Bruijn graphs, spanning n=2 and n=3
		graphs=[(8, 2, 5, 8), (10, 2, 6, 10), (8, 3, 10, 20), (12, 3, 10, 20)],
		bin_widths=[None, 16, 8, 4],                # None == plain; else permutation bucket width
		num_train_grid=[316, 1000, 3162, 10000, 31623, 63246],
		L_grid=[2, 4, 6, 8, 12, 16, 20],
		length_num_train=16000,
		length_epochs=6,                            # short-path data is trained several passes
		num_test=2000,
		seeds=[0, 1, 2],                            # points are averaged over these runs (SEM bars)
		model=dict(d_model=96, num_heads=4, num_layers=2),
		train=dict(lr=2e-3, weight_decay=1e-2, batch_size=512),
	),
	'full': dict(
		# more conditions: n=2 (V 8-12), n=3 (V 8-12), n=4 (V 8-10)
		graphs=[(8, 2, 5, 8), (10, 2, 6, 10), (12, 2, 8, 12),
				(8, 3, 10, 20), (10, 3, 12, 24), (11, 3, 14, 28), (12, 3, 15, 30),
				(8, 4, 12, 24), (9, 4, 14, 28), (10, 4, 16, 32)],
		bin_widths=[None, 16, 8, 4],
		num_train_grid=[316, 1000, 3162, 10000, 31623, 100000, 316228],
		L_grid=[2, 3, 4, 6, 8, 12, 16, 20, 24],
		length_num_train=64000,
		length_epochs=12,
		num_test=8000,
		seeds=[0, 1, 2, 3, 4],                      # 5 runs averaged per point
		model=dict(d_model=128, num_heads=4, num_layers=4),
		train=dict(lr=1e-3, weight_decay=1e-2, batch_size=512),
	),
}


# ----------------------------------------------------------------------------
# problem-size quantities
# ----------------------------------------------------------------------------

def _node_depth_sets(dag):
	"""
	For every node value, the set of depths (edges from a source) at which it can
	appear on a source-to-sink path. Forward DP in topological order.
	"""
	M = dag.num_nodes_all
	indptr, indices = dag._succ_indptr, dag._succ_indices
	depths = [None] * M
	for s in dag.source_vals:
		depths[int(s)] = {0}
	for u in dag.pi:
		du = depths[int(u)]
		if not du:
			continue
		for j in range(indptr[u], indptr[u + 1]):
			w = int(indices[j])
			nxt = {d + 1 for d in du}
			depths[w] = nxt if depths[w] is None else (depths[w] | nxt)
	return depths


def distinct_edges(dag, bin_width):
	"""
	Number of distinct transitions the model must learn.

	Without a permutation (`bin_width` is None) that is just the number of edges.
	With a period-`bin_width` position permutation, an occurrence of an edge whose
	source node sits at depth d spans character positions [d, d + n]: the n-gram
	window (positions d .. d+n-1) that the model relabels, plus the predicted token
	(position d + n). Two occurrences are the *same* symbol only if that whole
	window+target falls into the same sequence of position blocks, so the count is
	the number of distinct block-signatures `((d+k)//bin_width for k in 0..n)` over
	the depths the source can occupy -- not just the target token's block.
	"""
	if bin_width is None:
		return int(dag.num_edges)
	n = dag.n
	depths = _node_depth_sets(dag)
	total = 0
	for u in dag.edges_u:
		du = depths[int(u)] or set()
		total += len({tuple((d + k) // bin_width for k in range(n + 1)) for d in du})
	return int(total)


# ----------------------------------------------------------------------------
# threshold interpolation
# ----------------------------------------------------------------------------

def _crossing(xs, ys, target, want_below):
	"""
	Smallest x (ascending `xs`) at which `ys` crosses `target`.

	want_below=True  -> first x with y < target (a decreasing metric, illegal_mass)
	want_below=False -> first x with y > target (an increasing metric)

	Linear interpolation in log10(x) between the bracketing grid points. Returns
	the first grid x if it already satisfies the criterion, or nan if the grid
	never reaches it.
	"""
	xs = np.asarray(xs, float)
	ys = np.asarray(ys, float)
	ok = ys < target if want_below else ys > target
	idx = np.nonzero(ok)[0]
	if len(idx) == 0:
		return float('nan')
	i = idx[0]
	if i == 0:
		return float(xs[0])
	x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
	if y1 == y0:
		return float(x1)
	frac = (target - y0) / (y1 - y0)
	logx = math.log10(x0) + frac * (math.log10(x1) - math.log10(x0))
	return float(10 ** logx)


# ----------------------------------------------------------------------------
# per-unit caching
# ----------------------------------------------------------------------------

def _unit_key(spec):
	blob = json.dumps(spec, sort_keys=True, default=str)
	return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _cached_unit(spec, compute, force):
	os.makedirs(UNIT_CACHE_DIR, exist_ok=True)
	path = os.path.join(UNIT_CACHE_DIR, _unit_key(spec) + '.json')
	if os.path.exists(path) and not force:
		with open(path) as f:
			return json.load(f), True
	result = compute()
	with open(path, 'w') as f:
		json.dump(result, f, indent=1)
	return result, False


# ----------------------------------------------------------------------------
# the two unit computations
# ----------------------------------------------------------------------------

def _mcfg(profile):
	return ModelConfig(**profile['model'])


def _tcfg(profile, device, seed):
	return TrainConfig(device=device, seed=seed, **profile['train'])


def coverage_only_samples(gen, criterion):
	"""
	No-learning reference: the number of training paths at which the expected
	fraction of test transitions never demonstrated in training falls to
	`criterion`. With edge e traversed by a random path with probability
	edge_p[e] and weighted by how often the test set uses it (also edge_p[e]),
	the unseen-rate after m paths is sum_e w_e (1-p_e)^m, monotone decreasing.
	"""
	p = np.asarray(gen.edge_p, float)
	p = p[p > 0]
	if len(p) == 0:
		return None
	w = p / p.sum()

	def unseen(m):
		return float((w * (1.0 - p) ** m).sum())

	if unseen(1.0) <= criterion:
		return 1.0
	lo, hi = 1.0, 2.0
	while unseen(hi) > criterion:
		hi *= 2.0
		if hi > 1e12:
			return None
	for _ in range(60):
		mid = 0.5 * (lo + hi)
		if unseen(mid) > criterion:
			lo = mid
		else:
			hi = mid
	return 0.5 * (lo + hi)


def samples_unit(profile, ordering, graph, bin_width, seed, device, force):
	"""
	One point of samples_vs_edges: for a fixed (graph, permutation bucket, seed),
	the number of training samples needed to drive illegal_mass below the
	criterion, the transition count, and the coverage-only reference.
	"""
	V, n, S, N = graph
	dag_seed = 0            # fixed per graph so the seeds are replicates on one DAG
	spec = dict(kind='samples', ordering=ordering, V=V, n=n, S=S, N=N,
				bin_width=bin_width, seed=seed, dag_seed=dag_seed,
				grid=profile['num_train_grid'], num_test=profile['num_test'],
				model=profile['model'], train=profile['train'])

	def compute():
		dag = build_dag(V, n, S, N, ordering=ordering, seed=dag_seed)
		gen = ReasoningGenerator(dag)
		mcfg = _mcfg(profile)
		tcfg = _tcfg(profile, device, seed)
		grid, illegal, path_cov = [], [], []
		for num_train in profile['num_train_grid']:
			metrics, _, _ = run_experiment_1(
				V, n, S, N, num_train=num_train, num_test=profile['num_test'],
				permute=(bin_width is not None), bin_width=(bin_width or 1),
				perm_seed=seed, ordering=ordering, dag_seed=dag_seed,
				data_seed=seed, mcfg=mcfg, tcfg=tcfg)
			grid.append(num_train)
			illegal.append(metrics['illegal_mass'])
			path_cov.append(metrics['path_coverage'])
			# stop only once BOTH criteria are met; path_coverage > 0.95 is reached
			# later than illegal < 0.05, so stopping on illegal alone would leave the
			# coverage crossing unbracketed (nan) and drop the green point
			if metrics['illegal_mass'] < ILLEGAL_CRIT and metrics['path_coverage'] > PATH_COV_CRIT:
				break
		return dict(
			V=V, n=n, seed=seed, bin_width=bin_width, ordering=ordering,
			num_edges=int(dag.num_edges),
			distinct_edges=distinct_edges(dag, bin_width),
			num_states=int(dag.num_nodes),
			grid=grid, illegal=illegal, path_coverage=path_cov,
			samples_illegal=_crossing(grid, illegal, ILLEGAL_CRIT, want_below=True),
			samples_coverage=_crossing(grid, path_cov, PATH_COV_CRIT, want_below=False),
			# coverage-only reference is a property of the plain graph's edge set
			coverage_only=(coverage_only_samples(gen, ILLEGAL_CRIT) if bin_width is None else None),
			# a point already below criterion at the smallest grid value, or still
			# above it at the largest, is a bound -- excluded from the slope fit
			censored_left=bool(illegal[0] <= ILLEGAL_CRIT),
			censored_right=bool(illegal[-1] > ILLEGAL_CRIT),
		)

	return _cached_unit(spec, compute, force)[0]


def _train_short_test_full(V, n, S, N, L, num_train, num_test, ordering,
						   dag_seed, seed, mcfg, tcfg, epochs):
	"""
	Train on paths of <= L edges (for `epochs` passes), evaluate on the full path
	distribution. Returns (metrics, dag). Uses train_eval's public building blocks
	so the train and test distributions can differ (run_experiment_2 uses the same
	distribution for both), and so the length-limited data can be trained for
	several epochs -- one pass over few short paths under-fits.
	"""
	device = resolve_device(tcfg.device)
	dtype = _DTYPES[tcfg.dtype]
	dag = build_dag(V, n, S, N, ordering=ordering, seed=dag_seed)
	gen = ReasoningGenerator(dag)

	tr_chars, tr_len = gen.sample_length_limited(num_train, L, np.random.default_rng([seed, 1]))
	te_chars, te_len = gen.sample(num_test, np.random.default_rng([seed, 2]))

	max_seq_len = gen.max_chars + 1                 # cover the (longer) full-path test set
	model = build_model(V + 1, max_seq_len, mcfg, device, dtype, tcfg.seed)
	tr_full = build_full_tokens(tr_chars, tr_len, V)
	for e in range(epochs):                          # reshuffle each epoch
		train_one_epoch(model, tr_full, tr_len, n, replace(tcfg, seed=tcfg.seed + e))

	seen_states = training_visits(tr_chars, tr_len, dag)
	metrics = evaluate(model, te_chars, te_len, dag, seen_states=seen_states,
					   true_branch_p=gen.branch_p, device=device, batch_size=tcfg.batch_size)
	return metrics, dag


def length_unit(profile, ordering, graph, seed, device, force):
	"""
	One point of length_vs_Lmax: the smallest training path length L that lets
	the model reach each criterion on the full path distribution. The L sweep runs
	up to the DAG's longest path (L_max), but stops early once path_coverage passes
	its threshold (the binding, later criterion), so we don't keep training past it.
	"""
	V, n, S, N = graph
	dag_seed = 0            # fixed per graph so the seeds are replicates on one DAG
	num_train = profile['length_num_train']
	epochs = profile['length_epochs']
	spec = dict(kind='length', ordering=ordering, V=V, n=n, S=S, N=N, seed=seed,
				dag_seed=dag_seed, L_grid=profile['L_grid'], num_train=num_train,
				epochs=epochs, num_test=profile['num_test'],
				model=profile['model'], train=profile['train'])

	def compute():
		mcfg = _mcfg(profile)
		tcfg = _tcfg(profile, device, seed)
		# extend the grid up to this DAG's longest path so coverage is reachable
		dag0 = build_dag(V, n, S, N, ordering=ordering, seed=dag_seed)
		gen0 = ReasoningGenerator(dag0)
		L_max = gen0.max_edges
		L95 = percentile_length(gen0, 95)          # 95th-pctile path length in edges
		num_states, num_edges = int(dag0.num_nodes), int(dag0.num_edges)
		L_values = sorted({L for L in profile['L_grid'] if L < L_max} | {L_max})

		Ls, illegal, path_cov = [], [], []
		for L in L_values:
			try:
				metrics, _ = _train_short_test_full(
					V, n, S, N, L, num_train, profile['num_test'], ordering,
					dag_seed, seed, mcfg, tcfg, epochs)
			except ValueError:
				# no source-to-sink path with <= L edges: nothing to train on at
				# this L, so skip it and move on to the next (larger) L
				continue
			Ls.append(L)
			illegal.append(metrics['illegal_mass'])
			path_cov.append(metrics['path_coverage'])
			# path_coverage is the binding criterion (reached last); once it passes,
			# the illegal crossing is already recorded, so stop training at larger L
			if metrics['path_coverage'] > PATH_COV_CRIT:
				break
		return dict(
			V=V, n=n, S=S, N=N, seed=seed, ordering=ordering, num_states=num_states,
			num_edges=num_edges, L_max=int(L_max), L95=L95,
			Ls=Ls, illegal=illegal, path_coverage=path_cov,
			minL_illegal=_crossing(Ls, illegal, ILLEGAL_CRIT, want_below=True),
			minL_coverage=_crossing(Ls, path_cov, PATH_COV_CRIT, want_below=False),
		)

	return _cached_unit(spec, compute, force)[0]


# ----------------------------------------------------------------------------
# dispatch (one worker per device; threads let the GPUs overlap, since the
# heavy work is in torch/numpy which release the GIL)
# ----------------------------------------------------------------------------

def _run_units(unit_fn, jobs, devices, force, desc='units'):
	"""Compute `jobs` (list of kwargs dicts) via `unit_fn`, one worker per device,
	showing a tqdm progress bar (thread-safe update/write across the workers)."""
	results = [None] * len(jobs)
	bar = tqdm(total=len(jobs), desc=desc, unit='unit')

	def work(i):
		device = devices[i % len(devices)]
		t0 = time.time()
		results[i] = unit_fn(device=device, force=force, **jobs[i])
		tag = {k: jobs[i][k] for k in ('graph', 'bin_width', 'seed') if k in jobs[i]}
		bar.write(f"  [{device}] {tag} done in {time.time() - t0:.1f}s")
		bar.update(1)

	if len(devices) == 1:
		for i in range(len(jobs)):
			work(i)
	else:
		with ThreadPoolExecutor(max_workers=len(devices)) as ex:
			list(ex.map(work, range(len(jobs))))
	bar.close()
	return results


# ----------------------------------------------------------------------------
# figure builders
# ----------------------------------------------------------------------------

def _loglog_fit(ax, xs, ys, col):
	"""
	Least-squares power-law fit y = c * x^b (a straight line on log-log axes).
	Draws the line and returns (b, c); None if there are < 2 points.
	"""
	xs, ys = np.asarray(xs, float), np.asarray(ys, float)
	m = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
	if m.sum() < 2:
		return None
	b, a = np.polyfit(np.log10(xs[m]), np.log10(ys[m]), 1)   # log10 y = b log10 x + a
	gx = np.array([xs[m].min() * 0.75, xs[m].max() * 1.3])
	ax.plot(gx, 10 ** a * gx ** b, ls=':', lw=1.1, color=col, zorder=0)
	return float(b), float(10 ** a)


def _linear_fit(ax, xs, ys, col):
	"""
	Least-squares linear fit y = m * x + c (both axes linear). Draws the line and
	returns (m, c); None if there are < 2 points.
	"""
	xs, ys = np.asarray(xs, float), np.asarray(ys, float)
	mask = np.isfinite(xs) & np.isfinite(ys)
	if mask.sum() < 2:
		return None
	m, c = np.polyfit(xs[mask], ys[mask], 1)
	gx = np.array([xs[mask].min(), xs[mask].max()])
	ax.plot(gx, m * gx + c, ls=':', lw=1.1, color=col, zorder=1)
	return float(m), float(c)


def _agg_over_seeds(group, key, drop_censored=False):
	"""
	Mean and standard error of the mean of `key` over the runs (seeds) in `group`,
	using only the finite (and, if requested, non-censored) values. Returns
	(mean, sem, n) or None if nothing qualifies.
	"""
	vals = [r[key] for r in group if np.isfinite(r[key])
			and not (drop_censored and (r.get('censored_left') or r.get('censored_right')))]
	if not vals:
		return None
	vals = np.asarray(vals, float)
	sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
	return float(vals.mean()), sem, len(vals)


def percentile_length(gen, pct=95, num=20000, seed=0):
	"""
	Mass-weighted `pct`-th percentile of the path length in edges: sample paths
	uniformly over source-to-sink paths and take the ordinary percentile of their
	edge counts. Deterministic given the graph (fixed rng seed).
	"""
	_, lengths = gen.sample(num, np.random.default_rng(seed))
	return float(np.percentile(lengths - gen.n, pct))


def plot_samples_vs_edges(results, name):
	"""
	x = edges (distinct transitions the learner must acquire), y = samples to
	reach a criterion, both averaged over the seeds (runs).
	Both criteria are shown: illegal (blue) and coverage (green). Plain De Bruijn
	DAGs are filled markers, position-permuted controls are hollow (their edge
	count is inflated by the bucket width). Within each graph a translucent line
	joins plain -> widest bucket -> ... -> narrowest bucket. The dotted
	least-squares power-law fit per criterion is fit to the plain points only.
	"""
	fig, ax = plt.subplots(figsize=(3, 2.5))
	# trajectory order: plain, then widest bucket down to narrowest (== increasing
	# edge count, since a narrower bucket splits each edge across more positions)
	bw_order = [None] + sorted({r['bin_width'] for r in results
								if r['bin_width'] is not None}, reverse=True)
	graphs = []
	for r in results:
		g = (r['V'], r['n'])
		if g not in graphs:
			graphs.append(g)

	any_pts = False
	fits = {}
	for key, col in [('samples_illegal', CURVE_COLORS[0]), ('samples_coverage', CURVE_COLORS[1])]:
		# one aggregated point (mean over seeds) per (graph, bucket)
		pts = {}
		for g in graphs:
			for bw in bw_order:
				group = [r for r in results if (r['V'], r['n']) == g and r['bin_width'] == bw]
				a = _agg_over_seeds(group, key, drop_censored=True) if group else None
				if a:
					pts[(g, bw)] = (group[0]['distinct_edges'],) + a
		# connecting trajectory per graph, in bucket order
		for g in graphs:
			traj = [pts[(g, bw)][:2] for bw in bw_order if (g, bw) in pts]
			if len(traj) >= 2:
				ax.plot([t[0] for t in traj], [t[1] for t in traj],
						ls='-', lw=0.8, color=col, alpha=0.3, zorder=1)
		# one point (seed mean) per (graph, bucket); fit over the plain points only
		fx, fy = [], []
		for (g, bw), (x, mean, sem, n) in pts.items():
			any_pts = True
			filled = bw is None
			ax.scatter(x, mean, s=14, marker='o', lw=1.0,
					   facecolors=col if filled else 'none', edgecolors=col,
					   alpha=0.75, zorder=3)
			if filled:
				fx.append(x)
				fy.append(mean)
		fits[key] = _loglog_fit(ax, fx, fy, col)

	if any_pts:
		ax.set_xscale('log')
		ax.set_yscale('log')
	ax.set_xlabel('Edges', fontsize=LABEL_FS)
	ax.set_ylabel('Samples', fontsize=LABEL_FS)
	_style_axis(ax)
	crit = _legend(ax, [
		Line2D([], [], color=CURVE_COLORS[0], marker='o', ls='', ms=3.5, label='illegal <0.05'),
		Line2D([], [], color=CURVE_COLORS[1], marker='o', ls='', ms=3.5, label='coverage >0.95'),
	], loc='upper left')
	ax.add_artist(crit)
	_legend(ax, [
		Line2D([], [], color='0.35', marker='o', ls='', ms=3.5, label='plain'),
		Line2D([], [], color='0.35', marker='o', ls='', ms=3.5, mfc='none', label='permuted'),
		Line2D([], [], color='0.35', ls=':', lw=1.1, label='fit'),
	], loc='lower right')
	# fit coefficients (plain points only) are recorded in the provenance, not here
	return _save(fig, name), fits


def plot_length_vs_Lmax(results, name):
	"""
	x = L_max (the DAG's longest path), y = smallest training length L reaching a
	criterion, averaged over the seeds (runs). Criteria: illegal (blue) and
	coverage (green), each with a dotted linear fit L = m * L_max + c. Grey 'x'
	markers are the 95th-percentile path length L95 of each graph (the effective
	max length: 95% of paths are shorter), with its own dotted fit -- comparing
	the training length needed against the length the model must reconstruct.
	"""
	fig, ax = plt.subplots(figsize=(3, 2.5))
	graphs = []
	for r in results:
		g = (r['V'], r['n'])
		if g not in graphs:
			graphs.append(g)

	fits = {}
	for key, col in [('minL_illegal', CURVE_COLORS[0]), ('minL_coverage', CURVE_COLORS[1])]:
		fx, fy = [], []
		for g in graphs:
			group = [r for r in results if (r['V'], r['n']) == g]
			a = _agg_over_seeds(group, key)
			if not a:
				continue
			x = float(np.mean([r['L_max'] for r in group]))
			ax.scatter(x, a[0], s=14, marker='o', color=col, alpha=0.75, zorder=3)
			fx.append(x)
			fy.append(a[0])
		fits[key] = _linear_fit(ax, fx, fy, col)

	# 95th-percentile path length per graph (same for all its seeds), as 'x' markers
	L95_COL = '0.4'
	gx, gy = [], []
	for g in graphs:
		group = [r for r in results if (r['V'], r['n']) == g and 'L95' in r]
		if not group:
			continue
		gx.append(float(np.mean([r['L_max'] for r in group])))
		gy.append(float(np.mean([r['L95'] for r in group])))
	if gx:
		ax.scatter(gx, gy, s=22, marker='x', color=L95_COL, lw=1.1, zorder=3)
		fits['L95'] = _linear_fit(ax, gx, gy, L95_COL)

	ax.set_xlabel(r'$L_{\max}$', fontsize=LABEL_FS)
	ax.set_ylabel('Training Length', fontsize=LABEL_FS)
	_style_axis(ax)
	_legend(ax, [
		Line2D([], [], color=CURVE_COLORS[0], marker='o', ls='', ms=3.5, label='illegal <0.05'),
		Line2D([], [], color=CURVE_COLORS[1], marker='o', ls='', ms=3.5, label='coverage >0.95'),
		Line2D([], [], color=L95_COL, marker='x', ls='', ms=4, label=r'$L_{95}$'),
		Line2D([], [], color='0.35', ls=':', lw=1.1, label='fit'),
	], loc='upper left')
	return _save(fig, name), fits


# ----------------------------------------------------------------------------
# top-level figure orchestration
# ----------------------------------------------------------------------------

def _write_provenance(name, kind, profile, ordering, results, fits=None):
	"""
	Everything needed to reproduce a figure: the exact conditions swept (graphs,
	buckets/lengths, seeds, grids), the shared model/optimiser settings, the seed
	rules, the criteria, the environment, and (for the samples figure) the fitted
	power-law coefficients. Written next to the results JSON.
	"""
	shared = dict(
		graphs=[list(g) for g in profile['graphs']],           # (V, n, S, N)
		ordering=ordering,
		num_test=profile['num_test'],
		seeds=profile['seeds'],
		model=asdict(ModelConfig(**profile['model'])),         # full effective ModelConfig
		train=asdict(TrainConfig(**profile['train'])),         # full effective TrainConfig
		device='per-unit resolve_device(...); default shown under environment',
		criteria={'illegal_mass': f'< {ILLEGAL_CRIT}', 'path_coverage': f'> {PATH_COV_CRIT}'},
		path_coverage_ratio=DEFAULT_PATH_RATIO,                # P_model(path) >= ratio * P_true(path)
		runs_per_point=len(profile['seeds']),                  # each plotted point averages this many seeds
		error_bars='none drawn; each point is the mean over the seeds',
		seed_rules={
			'dag_seed': 'fixed at 0 per graph, so the seeds are replicates on one DAG',
			'train_sampler_rng': '[seed, 1]', 'test_sampler_rng': '[seed, 2]',
			'perm_seed': 'seed', 'weight_init + shuffle seed': 'TrainConfig.seed = seed',
		},
	)
	if kind == 'samples':
		# power-law fits y = c * x^b (x = distinct edges, y = samples), least
		# squares in log10 over the plain, non-censored points only
		labels = {'samples_illegal': 'illegal_mass < %s' % ILLEGAL_CRIT,
				  'samples_coverage': 'path_coverage > %s' % PATH_COV_CRIT}
		fit_summary = {}
		for key, label in labels.items():
			f = (fits or {}).get(key)
			fit_summary[label] = (None if not f else dict(
				coefficient_c=f[1], exponent_b=f[0],
				equation='y = %.6g * x^%.6g' % (f[1], f[0]),
				fitted_over='plain (non-permuted, non-censored) points'))
		experiment = dict(
			description='experiment 1: uniform paths (ReasoningGenerator.sample); '
						'samples to reach a criterion vs distinct transitions',
			epochs=1,
			bin_widths=profile['bin_widths'],                  # None = plain; else permutation bucket width
			num_train_grid=profile['num_train_grid'],
			fits=fit_summary,
			units=[dict(V=r['V'], n=r['n'], bin_width=r['bin_width'], seed=r['seed'],
						num_edges=r['num_edges'], distinct_edges=r['distinct_edges'],
						grid_evaluated=r['grid'],
						samples_illegal=r['samples_illegal'], samples_coverage=r['samples_coverage'],
						censored=[r['censored_left'], r['censored_right']]) for r in results],
		)
	else:
		# linear fits L = m * L_max + c over the per-graph mean points
		labels = {'minL_illegal': 'illegal_mass < %s' % ILLEGAL_CRIT,
				  'minL_coverage': 'path_coverage > %s' % PATH_COV_CRIT}
		labels['L95'] = '95th-percentile path length (L95) vs L_max'
		fit_summary = {}
		for key, label in labels.items():
			f = (fits or {}).get(key)
			fit_summary[label] = (None if not f else dict(
				slope_L_per_L_max=f[0], intercept=f[1],
				equation='L = %.6g * L_max + %.6g' % (f[0], f[1]),
				fitted_over='per-graph mean points'))
		experiment = dict(
			description='experiment 2: train on <= L-edge paths '
						'(ReasoningGenerator.sample_length_limited), test on the full '
						'distribution; smallest L reaching a criterion vs L_max, '
						'with the 95th-percentile path length (L95) for reference',
			length_epochs=profile['length_epochs'],
			length_num_train=profile['length_num_train'],
			L_grid=profile['L_grid'],
			note='L sweep runs up to each DAG\'s L_max; stops early once path_coverage > threshold',
			fits=fit_summary,
			units=[dict(V=r['V'], n=r['n'], seed=r['seed'], num_states=r['num_states'],
						num_edges=r['num_edges'], L_max=r.get('L_max'), L95=r.get('L95'),
						L_evaluated=r['Ls'], minL_illegal=r['minL_illegal'],
						minL_coverage=r['minL_coverage'])
				   for r in results],
		)
	prov = dict(figure=name, experiment=experiment, shared=shared, environment=dict(
		python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
		default_device=str(resolve_device(None))))
	os.makedirs(FIG_DIR, exist_ok=True)
	with open(os.path.join(FIG_DIR, name + '_provenance.json'), 'w') as f:
		json.dump(prov, f, indent=2)


def build_samples_vs_edges(profile, ordering, devices, force, name):
	jobs = [dict(profile=profile, ordering=ordering, graph=tuple(g), bin_width=bw, seed=seed)
			for g in profile['graphs']
			for bw in profile['bin_widths']
			for seed in profile['seeds']]
	print(f"[{name}] {len(jobs)} units")
	results = _run_units(samples_unit, jobs, devices, force, desc=name)
	os.makedirs(FIG_DIR, exist_ok=True)
	with open(os.path.join(FIG_DIR, name + '.json'), 'w') as f:
		json.dump(results, f, indent=1)
	path, fits = plot_samples_vs_edges(results, name)
	_write_provenance(name, 'samples', profile, ordering, results, fits=fits)
	print(f"[{name}] saved {path}.pdf/.png (+ {name}.json, {name}_provenance.json)")


def build_length_vs_Lmax(profile, ordering, devices, force, name):
	jobs = [dict(profile=profile, ordering=ordering, graph=tuple(g), seed=seed)
			for g in profile['graphs']
			for seed in profile['seeds']]
	print(f"[{name}] {len(jobs)} units")
	results = _run_units(length_unit, jobs, devices, force, desc=name)
	os.makedirs(FIG_DIR, exist_ok=True)
	with open(os.path.join(FIG_DIR, name + '.json'), 'w') as f:
		json.dump(results, f, indent=1)
	path, fits = plot_length_vs_Lmax(results, name)
	_write_provenance(name, 'length', profile, ordering, results, fits=fits)
	print(f"[{name}] saved {path}.pdf/.png (+ {name}.json, {name}_provenance.json)")


FIGURES = {
	'samples_vs_edges': lambda p, d, f: build_samples_vs_edges(p, 'random', d, f, 'samples_vs_edges'),
	'length_vs_Lmax': lambda p, d, f: build_length_vs_Lmax(p, 'random', d, f, 'length_vs_Lmax'),
	'samples_vs_edges_digit_sum': lambda p, d, f: build_samples_vs_edges(p, 'digit_sum', d, f, 'samples_vs_edges_digit_sum'),
	'length_vs_Lmax_digit_sum': lambda p, d, f: build_length_vs_Lmax(p, 'digit_sum', d, f, 'length_vs_Lmax_digit_sum'),
}


def _default_devices():
	if torch.cuda.is_available():
		return [f'cuda:{i}' for i in range(torch.cuda.device_count())]
	return [str(resolve_device(None))]


def main():
	ap = argparse.ArgumentParser(description=__doc__,
								 formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--profile', choices=list(PROFILES), default=None,
					help="sweep size; default 'full' on CUDA else 'laptop'")
	ap.add_argument('--devices', default=None,
					help="comma-separated torch devices, e.g. cuda:0,cuda:1 (default: all)")
	ap.add_argument('--figures', default=None,
					help="comma-separated subset of: " + ', '.join(FIGURES))
	ap.add_argument('--digit-sum', action='store_true',
					help="also make the digit-sum-ordering figures (off by default)")
	ap.add_argument('--force', action='store_true', help="recompute, ignoring the unit cache")
	args = ap.parse_args()

	profile_name = args.profile or ('full' if torch.cuda.is_available() else 'laptop')
	profile = PROFILES[profile_name]
	devices = args.devices.split(',') if args.devices else _default_devices()
	if args.figures:                            # explicit subset overrides everything
		names = args.figures.split(',')
	else:                                       # random ordering only, unless --digit-sum
		names = [f for f in FIGURES if args.digit_sum or not f.endswith('_digit_sum')]

	print(f"profile={profile_name} devices={devices} figures={names}")
	t0 = time.time()
	for name in names:
		FIGURES[name](profile, devices, args.force)
	print(f"done in {time.time() - t0:.1f}s")


if __name__ == '__main__':
	main()
