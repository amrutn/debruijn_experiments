"""
Run the synthetic De Bruijn training experiments and make the figures.

A decoder-only transformer is trained from scratch on paths sampled
from D_pi = F[B(V,n); pi, N, S]. There is no prompt: the model has to learn
which subset of the vocabulary is legal at each state, measured by `illegal_mass`
and `path_coverage` (see train_eval.py). Two criteria mark "has learned it":
illegal_mass < 0.05 and path_coverage > 0.95.

Figures (each size (3, 2.5) pdf+png in figures/)
-------
samples_vs_edges            Scaling of the number of training samples needed to
                            reach each criterion (illegal_mass < 0.05 in blue,
                            path_coverage > 0.95 in green) vs the number
                            of edges the model must learn. The ordering pi is a random
                            permutation. Filled markers are plain De Bruijn DAGs
                            (x = number of edges). Hollow markers are the same
                            DAGs with the token identities scrambled by a
                            position-dependent permutation of period `bin_width`;
                            such a permutation multiplies the number of *distinct*
                            (edge, position-block) transitions, and the x value is
                            that larger count. If sample cost is set by the number
                            of distinct transitions, plain and permuted points fall
                            on the same fitted line y = a*x (drawn dotted).

length_vs_Lmax              Same criterion, but the model is trained only on paths
                            of at most L edges (`sample_length_limited`) and tested
                            on the full path distribution. y is the smallest L that
                            still reaches the criterion; x is L_max (the DAG's
                            longest path). 


Caching
-------
Every (DAG, setting, seed) unit writes its computed point to
cache/exp_results/<hash>.json, so an interrupted or repeated run reuses finished
units instead of retraining. train_eval additionally caches the model weights.
Pass --force to recompute.

Usage
-----
    python run_experiments.py                  # auto: 'full' on CUDA else 'laptop'
    python run_experiments.py --profile full   # the full run
    python run_experiments.py --profile laptop # quick, partial results for a laptop
    python run_experiments.py --devices cuda:0,cuda:1
    python run_experiments.py --figures samples_vs_edges --force # force rerun of a specific figure
"""

import os
import sys
import json
import math
import time
import hashlib
import argparse
from dataclasses import replace, asdict
import contextlib
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

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
	_load_or_train, _DTYPES, DEFAULT_PATH_RATIO,
)


HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
UNIT_CACHE_DIR = os.path.join(HERE, 'cache', 'exp_results')

# criteria the model must reach
ILLEGAL_CRIT = 0.05        # illegal_mass must drop below this
PATH_COV_CRIT = 0.95       # path_coverage must rise above this
HINT_MAX = 5               # Calculate both criteria with all hint
						   # levels up to HINT_MAX for cache.


# ----------------------------------------------------------------------------
# plotting style (mirrors theory/theorem1_theorem2.py so the two sets of figures
# look consistent)
# ----------------------------------------------------------------------------

CURVE_COLORS = ['#2a78d6', '#008300', '#e87ba4', '#eda100']   # blue, green, magenta, gold
CURVE_MARKERS = ['o', 's', '^', 'D']
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


def _legend(ax, handles, loc='lower right', bbox=None):
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
# profiles: the sweep sizes. 'full' is the cluster sweep (2x4090, ~a few hours
# with the enlarged length training); 'laptop' is a quick partial run to verify
# the pipeline on mps/cpu.
# ----------------------------------------------------------------------------

def _geomgrid(lo, hi, n):
	"""`n` integers geometrically spaced over [lo, hi] (deduped, sorted)."""
	return sorted({int(round(x)) for x in np.geomspace(lo, hi, n)})


PROFILES = {
	'laptop': dict(
		# (V, n, S, N) De Bruijn graphs, spanning n=2 and n=3
		graphs=[(8, 2, 5, 8), (10, 2, 6, 10), (8, 3, 10, 20), (12, 3, 10, 20)],
		bin_widths=[None, 16, 8, 4],                # None == plain; else permutation bucket width
		num_train_grid=_geomgrid(316, 63246, 15),   # dense geometric grid (~1.5x steps)
		L_grid=list(range(1, 40)),                  # every integer length (up to L_max)
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
		num_train_grid=_geomgrid(316, 316228, 22),  # dense geometric grid (~1.4x steps)
		L_grid=list(range(1, 40)),                  # every integer length (up to L_max)
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
	dag_seed = seed        # each seed draws a fresh DAG
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
			# stop once both criteria are met; path_coverage > 0.95 is reached
			# later than illegal < 0.05
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


# ----------------------------------------------------------------------------
# samples_vs_edges_minimal (--minimal): train on the optimal P_min edge covering
# ----------------------------------------------------------------------------

def _cover_flow(dag):
	"""The trace-minimal edge-cover flow for `dag`, via theory's min_cost_cover_flow
	(lambda large -> minimize the number of source-to-sink traces, i.e. P_min). Returns
	{(u, v): f >= 1}: an integer flow covering every edge at least once."""
	theory_dir = os.path.join(os.path.dirname(HERE), 'theory')
	if theory_dir not in sys.path:
		sys.path.insert(0, theory_dir)
	from reasoning_tradeoff import min_cost_cover_flow
	_, _, edge_flow = min_cost_cover_flow(dag, lam=int(dag.num_edges) + 10)
	return edge_flow


def _draw_optimal_covering(dag, edge_flow, T, rng):
	"""One random minimal edge covering as (chars, lengths) in the
	ReasoningGenerator char format. The cover flow is decomposed into source-to-sink
	traces by following a uniformly random out-edge with remaining flow at each step;
	the result is P_min traces (the flow value) that together traverse every edge at
	least once. Different `rng` draws give different minimal coverings of the same DAG.
	"""
	V, n = dag.V, dag.n
	Vp = V ** (n - 1)
	rem = defaultdict(dict)
	for (u, v), f in edge_flow.items():
		rem[u][v] = int(f)
	is_sink = dag.is_sink
	paths = []                                   # each is a node sequence u_1 -> ... -> u_{L+1}
	for s in dag.source_vals.tolist():
		for _ in range(int(sum(rem[s].values()))):
			nodes = [s]
			node = s
			while not is_sink[node]:
				choices = [w for w, fw in rem[node].items() if fw > 0]
				node = int(choices[rng.integers(len(choices))])
				nodes.append(node)
			for u, v in zip(nodes[:-1], nodes[1:]):
				rem[u][v] -= 1
			paths.append(nodes)
	num = len(paths)
	chars = np.full((num, T), -1, dtype=np.int16)
	lengths = np.full(num, 0, dtype=np.int32)
	for i, nodes in enumerate(paths):
		x = nodes[0]
		for k in range(n):                       # first n chars spell the start node
			chars[i, k] = x % V
			x //= V
		t = n
		for nd in nodes[1:]:                     # each edge appends one char (= nd // Vp)
			chars[i, t] = nd // Vp
			t += 1
		lengths[i] = t
	return chars, lengths


def samples_minimal_unit(profile, ordering, graph, seed, device, force):
	"""
	One point of samples_vs_edges_minimal: like samples_unit, but the training samples
	are drawn from the *optimal* P_min edge covering rather than uniformly over paths.
	Start from one random minimal covering (P_min traces covering every edge); if the
	criteria are not met, draw another random covering and append it, and so on, up to
	the same sample budget as samples_vs_edges. A fresh model is trained for one epoch
	on the accumulated samples at each step, and the crossing gives the samples needed.
	This is the control behind the claim that a minimal cover is *not* more sample
	efficient: it weights all edges nearly equally and uses short traces, so it
	under-represents the high-traffic transitions and long paths that dominate the test.
	"""
	V, n, S, N = graph
	dag_seed = seed        # each seed draws a fresh DAG
	sample_cap = int(profile['num_train_grid'][-1])     # same max budget as samples_vs_edges
	spec = dict(kind='samples_minimal', ordering=ordering, V=V, n=n, S=S, N=N,
				seed=seed, dag_seed=dag_seed, num_test=profile['num_test'],
				sample_cap=sample_cap, model=profile['model'], train=profile['train'])

	def compute():
		dag = build_dag(V, n, S, N, ordering=ordering, seed=dag_seed)
		gen = ReasoningGenerator(dag)
		mcfg = _mcfg(profile)
		tcfg = _tcfg(profile, device, seed)
		device_r = resolve_device(tcfg.device)
		dtype = _DTYPES[tcfg.dtype]
		T = gen.max_chars
		max_seq_len = gen.max_chars + 1
		edge_flow = _cover_flow(dag)
		test_chars, test_len = gen.sample(profile['num_test'], np.random.default_rng([seed, 2]))

		coverings = []                                  # accumulated random coverings
		def covering(i):
			while len(coverings) <= i:
				r = np.random.default_rng([seed, 10000 + len(coverings)])
				coverings.append(_draw_optimal_covering(dag, edge_flow, T, r))
			return coverings[i]
		P_min = int(covering(0)[0].shape[0])            # traces per covering (flow value)

		grid, illegal, path_cov = [], [], []
		k = 1
		while True:
			chs = np.concatenate([covering(i)[0] for i in range(k)])
			lns = np.concatenate([covering(i)[1] for i in range(k)])
			model = build_model(V + 1, max_seq_len, mcfg, device_r, dtype, tcfg.seed)
			train_one_epoch(model, build_full_tokens(chs, lns, V), lns, n, tcfg)
			seen_states = training_visits(chs, lns, dag)
			m = evaluate(model, test_chars, test_len, dag, seen_states=seen_states,
						 true_branch_p=gen.branch_p, device=device_r, batch_size=tcfg.batch_size)
			grid.append(int(chs.shape[0]))              # total samples used so far
			illegal.append(m['illegal_mass'])
			path_cov.append(m['path_coverage'])
			met = m['illegal_mass'] < ILLEGAL_CRIT and m['path_coverage'] > PATH_COV_CRIT
			if met or chs.shape[0] >= sample_cap:
				break
			k = max(k + 1, int(math.ceil(k * 1.5)))     # geometric growth in #coverings

		return dict(
			V=V, n=n, seed=seed, ordering=ordering, bin_width=None,
			num_edges=int(dag.num_edges), distinct_edges=int(dag.num_edges),
			num_states=int(dag.num_nodes), P_min=P_min,
			grid=grid, illegal=illegal, path_coverage=path_cov,
			samples_illegal=_crossing(grid, illegal, ILLEGAL_CRIT, want_below=True),
			samples_coverage=_crossing(grid, path_cov, PATH_COV_CRIT, want_below=False),
			censored_left=bool(illegal[0] <= ILLEGAL_CRIT),
			censored_right=bool(illegal[-1] > ILLEGAL_CRIT),
		)

	return _cached_unit(spec, compute, force)[0]


def _train_short_test_full(V, n, S, N, L, num_train, num_test, ordering,
						   dag_seed, seed, mcfg, tcfg, epochs, window=None, augment=False,
						   num_hints=0, force=False):
	"""
	Train on paths of <= L edges (for `epochs` passes), evaluate on the full path
	distribution. Returns (metrics, dag). If `window`, use sliding-window attention of that width.
	If `augment` is set, each short training sequence gets one random position gap
	(before its last n tokens) that shifts later tokens toward the deep positions.
	Evaluation is at natural positions. If `num_hints` > 0, the metrics dict
	also carries the hinted versions.
	"""
	device = resolve_device(tcfg.device)
	dtype = _DTYPES[tcfg.dtype]
	dag = build_dag(V, n, S, N, ordering=ordering, seed=dag_seed)
	gen = ReasoningGenerator(dag)

	train_chars, train_len = gen.sample_length_limited(num_train, L, np.random.default_rng([seed, 1]))
	test_chars, test_len = gen.sample(num_test, np.random.default_rng([seed, 2]))

	train_full = build_full_tokens(train_chars, train_len, V)
	# augmentation samples real-token positions in [0, pos_cap] (the deepest full-path
	# index) and lets padding continue past it; the RoPE cache must cover that overflow
	# (real max pos_cap + up to width padding slots), so size it to max_chars + width.
	pos_cap = gen.max_chars - 1 if augment else None
	max_seq_len = gen.max_chars + 1                 # cover the (longer) full-path test set
	if augment:
		max_seq_len = gen.max_chars + train_full.shape[1]

	def factory():
		return build_model(V + 1, max_seq_len, mcfg, device, dtype, tcfg.seed)

	def train_fn(model):
		for e in range(epochs):                      # reshuffle each epoch
			train_one_epoch(model, train_full, train_len, n, replace(tcfg, seed=tcfg.seed + e),
							window=window, augment=augment, pos_cap=pos_cap)
		return None

	# cache the trained weights keyed on everything that determines them (but not
	# the test set or the metrics), so re-plotting after a metric change reloads the
	# model and only re-evaluates -- no retraining. Device is excluded so the same
	# weights are reused across devices; the RoPE buffers are non-persistent and
	# recomputed by the factory, so max_seq_len need not be in the key.
	model_config = dict(
		kind='length_model', V=V, n=n, S=S, N=N, L=int(L), num_train=num_train,
		ordering=ordering, dag_seed=dag_seed, seed=seed, epochs=epochs,
		window=window, augment=bool(augment),
		model=asdict(mcfg),
		train={k: v for k, v in asdict(tcfg).items() if k != 'device'},
	)
	model, _, _ = _load_or_train(factory, train_fn, model_config, device, force)

	seen_states = training_visits(train_chars, train_len, dag)
	metrics = evaluate(model, test_chars, test_len, dag, seen_states=seen_states,
					   true_branch_p=gen.branch_p, device=device, batch_size=tcfg.batch_size,
					   window=window, num_hints=num_hints)
	return metrics, dag


def length_unit(profile, ordering, graph, seed, device, force, windowed=False,
				augmented=False):
	"""
	One point of length_vs_Lmax: the smallest training path length L that lets
	the model reach each criterion on the full path distribution. The L sweep
	stops early once path_coverage and illegal_mass pass their threshold.
	If `windowed` is set, training/evaluation use sliding-window attention of width
	n -- local attention that is position-invariant and so should generalise to
	longer paths. If `augmented` is set, the short training paths are trained with
	random non-contiguous RoPE position gaps (position-augmentation) instead.

	Every evaluation also carries the hint curve (metrics at 0..HINT_MAX free
	ground-truth hints per path, see evaluate), so `minL_illegal_by_hints[k]` /
	`minL_coverage_by_hints[k]` give the crossings under k hints without any
	retraining -- the `--hints k` plots just read slice k. Index 0 is the baseline.
	"""
	V, n, S, N = graph
	dag_seed = seed        # each seed draws a fresh DAG, so the spread over seeds
						   # reflects graph-to-graph variability, not just training noise
	num_train = profile['length_num_train']
	epochs = profile['length_epochs']
	window = n if windowed else None
	spec = dict(kind='length', ordering=ordering, V=V, n=n, S=S, N=N, seed=seed,
				dag_seed=dag_seed, L_grid=profile['L_grid'], num_train=num_train,
				epochs=epochs, num_test=profile['num_test'], hint_max=HINT_MAX,
				model=profile['model'], train=profile['train'],
				metrics_version=2)     # v2: reports max_covered (longest covered path) instead of L95
	if windowed:                        # keep the baseline spec (and its cache key)
		spec['window'] = window         # byte-identical to earlier runs
	if augmented:
		spec['augment'] = 'softgap'         # softened single-gap augmentation (bumped again,
											# so caches from earlier augment methods recompute)

	def compute():
		mcfg = _mcfg(profile)
		tcfg = _tcfg(profile, device, seed)
		# extend the grid up to this DAG's longest path so coverage is reachable
		dag0 = build_dag(V, n, S, N, ordering=ordering, seed=dag_seed)
		gen0 = ReasoningGenerator(dag0)
		L_max = gen0.max_edges
		num_states, num_edges = int(dag0.num_nodes), int(dag0.num_edges)
		L_values = sorted({L for L in profile['L_grid'] if L < L_max} | {L_max})

		Ls, illegal_bh, cov_bh = [], [], []        # per-L hint curves (len HINT_MAX+1)
		maxcov = []                                # per-L longest covered path (edges)
		for L in L_values:
			try:
				metrics, _ = _train_short_test_full(
					V, n, S, N, L, num_train, profile['num_test'], ordering,
					dag_seed, seed, mcfg, tcfg, epochs, window=window, augment=augmented,
					num_hints=HINT_MAX, force=force)
			except ValueError:
				# no source-to-sink path with <= L edges: nothing to train on at
				# this L, so skip it and move on to the next (larger) L
				continue
			Ls.append(L)
			illegal_bh.append(metrics['illegal_mass_by_hints'])
			cov_bh.append(metrics['path_coverage_by_hints'])
			maxcov.append(metrics['max_covered_edges'])
			# stop when criteria are met with 0 "hints".
			if metrics['path_coverage_by_hints'][0] > PATH_COV_CRIT:
				break

		il = np.array(illegal_bh, float).reshape(len(Ls), HINT_MAX + 1)
		cv = np.array(cov_bh, float).reshape(len(Ls), HINT_MAX + 1)
		minL_il = [_crossing(Ls, il[:, k], ILLEGAL_CRIT, want_below=True) for k in range(HINT_MAX + 1)]
		minL_cov = [_crossing(Ls, cv[:, k], PATH_COV_CRIT, want_below=False) for k in range(HINT_MAX + 1)]
		# longest path the model covers, taken from the same model that reaches the
		# coverage criterion: the smallest-L model with path_coverage > PATH_COV_CRIT
		# (the sweep stops right after this L). None if coverage is never reached.
		crossed = np.where(cv[:, 0] > PATH_COV_CRIT)[0]
		max_covered = int(maxcov[crossed[0]]) if len(crossed) else None
		return dict(
			V=V, n=n, S=S, N=N, seed=seed, ordering=ordering, num_states=num_states,
			num_edges=num_edges, L_max=int(L_max), max_covered=max_covered,
			window=window, augment=augmented,
			Ls=Ls, illegal=il[:, 0].tolist(), path_coverage=cv[:, 0].tolist(),
			minL_illegal=minL_il[0], minL_coverage=minL_cov[0],
			minL_illegal_by_hints=minL_il, minL_coverage_by_hints=minL_cov,
		)

	return _cached_unit(spec, compute, force)[0]


# ----------------------------------------------------------------------------
# dispatch (one worker per device; threads let the GPUs overlap, since the
# heavy work is in torch/numpy which release the GIL)
# ----------------------------------------------------------------------------

# Number of parallel worker *processes* for _run_units; set by main() from --workers.
# Processes (not threads) are used so the CPU-bound parts (path sampling, DAG
# building, the numpy eval/hint post-processing) run in parallel rather than
# serializing on the GIL. Already-cached units are read instantly.
# None -> default of len(devices).
_N_WORKERS = None


def _run_units(unit_fn, jobs, devices, force, desc='units'):
	"""Compute `jobs` (list of kwargs dicts) via `unit_fn` across a pool of worker
	processes (round-robin over `devices`), showing a tqdm progress bar. Each job is
	dispatched once -- no redundant recomputation -- and results are collected as
	they finish."""
	results = [None] * len(jobs)
	workers = min(_N_WORKERS or len(devices), len(jobs))
	bar = tqdm(total=len(jobs), desc=desc, unit='unit')

	if workers <= 1:                                   # serial (debug / single unit)
		for i, job in enumerate(jobs):
			results[i] = unit_fn(device=devices[i % len(devices)], force=force, **job)
			bar.update(1)
		bar.close()
		return results

	# One process pool per device (spawn, required for CUDA in children), so each
	# worker only ever touches its own GPU -- it holds a single CUDA context instead
	# of drifting across all devices. per_dev workers per device ~= workers total.
	ctx = mp.get_context('spawn')
	per_dev = max(1, -(-workers // len(devices)))      # ceil(workers / #devices)
	with contextlib.ExitStack() as stack:
		pools = {d: stack.enter_context(ProcessPoolExecutor(max_workers=per_dev, mp_context=ctx))
				 for d in devices}
		fut_to_i = {}
		for i, job in enumerate(jobs):
			d = devices[i % len(devices)]
			fut_to_i[pools[d].submit(unit_fn, device=d, force=force, **job)] = i
		for fut in as_completed(fut_to_i):
			i = fut_to_i[fut]
			results[i] = fut.result()                  # re-raises any error from the worker
			tag = {k: jobs[i][k] for k in ('graph', 'bin_width', 'seed') if k in jobs[i]}
			bar.write(f"  [{devices[i % len(devices)]}] {tag} done")
			bar.update(1)
	bar.close()
	return results


# ----------------------------------------------------------------------------
# figure builders
# ----------------------------------------------------------------------------

def _loglog_fit(xs, ys):
	"""
	Least-squares power-law fit y = c * x^b (a straight line on log-log axes).
	Returns (b, c), or None if there are < 2 usable points. Drawing is left to the
	caller so the line can span the full axis range once the limits are known.
	"""
	xs, ys = np.asarray(xs, float), np.asarray(ys, float)
	m = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
	if m.sum() < 2:
		return None
	b, a = np.polyfit(np.log10(xs[m]), np.log10(ys[m]), 1)   # log10 y = b log10 x + a
	return float(b), float(10 ** a)


def _linear_fit(xs, ys):
	"""
	Least-squares linear fit y = m * x + c (both axes linear). Returns (m, c), or
	None if there are < 2 finite points. Drawing is left to the caller so the line
	can span the full axis range once the limits are known.
	"""
	xs, ys = np.asarray(xs, float), np.asarray(ys, float)
	mask = np.isfinite(xs) & np.isfinite(ys)
	if mask.sum() < 2:
		return None
	m, c = np.polyfit(xs[mask], ys[mask], 1)
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


def plot_samples_vs_edges(results, name, fit_plain_only=False):
	"""
	x = edges (distinct transitions the learner must acquire), y = samples to
	reach a criterion, both averaged over the seeds (runs).
	Both criteria are shown: illegal (blue) and coverage (green). Plain De Bruijn
	DAGs are filled markers, position-permuted controls are hollow (their edge
	count is inflated by the bucket width); error bars are +/- SEM over the seeds.
	Within each graph a translucent line joins plain -> widest bucket -> ... ->
	narrowest bucket. The dotted least-squares power-law fit per criterion is fit to
	all points (plain and permuted), or to the plain points only when
	`fit_plain_only` is set.
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
	fit_lines = []          # (fit_params, color) to draw across the full x-range later
	for key, col in [('samples_illegal', CURVE_COLORS[0]), ('samples_coverage', CURVE_COLORS[1])]:
		# one aggregated point (mean over seeds) per (graph, bucket)
		pts = {}
		for g in graphs:
			for bw in bw_order:
				group = [r for r in results if (r['V'], r['n']) == g and r['bin_width'] == bw]
				a = _agg_over_seeds(group, key, drop_censored=True) if group else None
				if a:
					edges = float(np.mean([r['distinct_edges'] for r in group]))
					pts[(g, bw)] = (edges,) + a          # (edges, samples_mean, samples_sem, n)
		# connecting trajectory per graph, in bucket order
		for g in graphs:
			traj = [pts[(g, bw)][:2] for bw in bw_order if (g, bw) in pts]
			if len(traj) >= 2:
				ax.plot([t[0] for t in traj], [t[1] for t in traj],
						ls='--', lw=0.8, color=col, alpha=0.4, zorder=1)
		# one point (seed mean +/- SEM) per (graph, bucket); all points are plotted,
		# but the fit uses all points, or only the plain ones if fit_plain_only
		fx, fy = [], []
		for (g, bw), (x, mean, sem, n) in pts.items():
			any_pts = True
			filled = bw is None
			ax.errorbar(x, mean, yerr=sem, fmt='o', ms=3.8, mew=1.0,
						markerfacecolor=(col if filled else 'none'), markeredgecolor=col,
						ecolor=col, elinewidth=0.7, capsize=0, alpha=0.85, zorder=3)
			if filled or not fit_plain_only:
				fx.append(x)
				fy.append(mean)
		fits[key] = _loglog_fit(fx, fy)
		if fits[key]:
			fit_lines.append((fits[key], col))

	if any_pts:
		ax.set_xscale('log')
		ax.set_yscale('log')
	# draw each power-law fit across the full x-range, holding the data-driven limits
	# so the lines span the whole plot (clipped to the box) without rescaling it
	xlim, ylim = ax.get_xlim(), ax.get_ylim()
	gx_full = np.array(xlim)
	for (b, c), col in fit_lines:
		ax.plot(gx_full, c * gx_full ** b, ls='-', lw=1.6, color=col, zorder=0)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_xlabel('Edges', fontsize=LABEL_FS)
	ax.set_ylabel('Samples', fontsize=LABEL_FS)
	_style_axis(ax)
	crit = _legend(ax, [
		Line2D([], [], color=CURVE_COLORS[0], marker='o', ls='', ms=3.5, label=r'$P($invalid tokens$) < 0.05$'),
		Line2D([], [], color=CURVE_COLORS[1], marker='o', ls='', ms=3.5, label=r'path coverage $> 0.95$'),
	], loc='upper left')
	ax.add_artist(crit)
	_legend(ax, [
		Line2D([], [], color='0.35', marker='o', ls='', ms=3.5, label='plain'),
		Line2D([], [], color='0.35', marker='o', ls='', ms=3.5, mfc='none', label='remapped'),
		Line2D([], [], color='0.35', ls='-', lw=1.6, label='fit'),
	], loc='upper left', bbox=(0.0, 0.80))
	# fit coefficients (all points) are recorded in the provenance, not here
	return _save(fig, name), fits


def _min_cover_by_graph(results, profile, ordering):
	"""
	Mean and SEM (over the graph seeds) of the minimum number of source-to-sink
	paths needed to cover every edge of the DAG, per (V, n). Uses theory's exact
	min-flow `calc_min_cover`. S, N come from the profile (the samples results only
	store V, n); each seed rebuilds its own DAG (dag_seed == seed).
	"""
	import sys
	theory_dir = os.path.join(os.path.dirname(HERE), 'theory')
	if theory_dir not in sys.path:
		sys.path.insert(0, theory_dir)
	from theorem1_theorem2 import calc_min_cover

	sn = {(V, n): (S, N) for (V, n, S, N) in profile['graphs']}
	seeds_by_g = defaultdict(set)
	for r in results:
		seeds_by_g[(r['V'], r['n'])].add(r['seed'])

	out = {}
	for g, seeds in seeds_by_g.items():
		if g not in sn:
			continue
		V, n = g
		S, N = sn[g]
		vals = np.array([calc_min_cover(build_dag(V, n, S, N, ordering=ordering, seed=s))
						 for s in sorted(seeds)], float)
		sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
		out[g] = (float(vals.mean()), sem)
	return out


def plot_sample_efficiency_vs_edges(results, name, mincover):
	"""
	x = edges, y = sample efficiency = (min paths to cover all edges) / (samples to
	reach a criterion). The numerator is the per-(V,n) mean over seeds of the exact
	min edge-cover; the denominator is the seed-mean samples (as in samples_vs_edges).
	Both criteria are shown (illegal blue, coverage green), plain markers filled /
	permuted hollow, with a thin line joining each graph's plain -> permuted points
	and a thick line joining the plain points across graphs (per criterion). x is
	log-scaled, y linear. Error bars combine the two SEMs by error propagation for a
	ratio. No fits.
	"""
	fig, ax = plt.subplots(figsize=(3, 2.5))
	bw_order = [None] + sorted({r['bin_width'] for r in results
								if r['bin_width'] is not None}, reverse=True)
	graphs = []
	for r in results:
		g = (r['V'], r['n'])
		if g not in graphs:
			graphs.append(g)

	any_pts = False
	for key, col in [('samples_illegal', CURVE_COLORS[0]), ('samples_coverage', CURVE_COLORS[1])]:
		pts = {}
		for g in graphs:
			if g not in mincover:
				continue
			mc, mc_sem = mincover[g]
			for bw in bw_order:
				group = [r for r in results if (r['V'], r['n']) == g and r['bin_width'] == bw]
				a = _agg_over_seeds(group, key, drop_censored=True) if group else None
				if not a:
					continue
				s_mean, s_sem, _ = a
				if s_mean <= 0 or mc <= 0:
					continue
				edges = float(np.mean([r['distinct_edges'] for r in group]))
				eff = mc / s_mean
				# error propagation for the ratio N/D (treating N, D as independent)
				rel = np.sqrt((mc_sem / mc) ** 2 + (s_sem / s_mean) ** 2)
				pts[(g, bw)] = (edges, eff, eff * rel)
		# thin line joining plain -> widest bucket -> ... -> narrowest, per graph
		for g in graphs:
			traj = [pts[(g, bw)][:2] for bw in bw_order if (g, bw) in pts]
			if len(traj) >= 2:
				ax.plot([t[0] for t in traj], [t[1] for t in traj],
						ls='--', lw=0.8, color=col, alpha=0.4, zorder=1)
		# thick line joining the plain points across graphs (per criterion)
		plain = sorted(pts[(g, None)][:2] for g in graphs if (g, None) in pts)
		if len(plain) >= 2:
			ax.plot([p[0] for p in plain], [p[1] for p in plain],
					ls='-', lw=1.6, color=col, alpha=0.85, zorder=2)
		for (g, bw), (x, eff, err) in pts.items():
			any_pts = True
			filled = bw is None
			ax.errorbar(x, eff, yerr=err, fmt='o', ms=3.8, mew=1.0,
						markerfacecolor=(col if filled else 'none'), markeredgecolor=col,
						ecolor=col, elinewidth=0.7, capsize=0, alpha=0.85, zorder=3)

	if any_pts:
		ax.set_xscale('log')                           # x log, y linear
	ax.set_xlabel('Edges', fontsize=LABEL_FS)
	ax.set_ylabel(rf'$P_{{\text{min}}}/$Samples', fontsize=LABEL_FS)
	_style_axis(ax)
	# both legends stacked at the top-left (criteria on top, plain/permuted below)
	crit = _legend(ax, [
		Line2D([], [], color=CURVE_COLORS[0], marker='o', ls='', ms=3.5, label=r'$P($invalid tokens$) < 0.05$'),
		Line2D([], [], color=CURVE_COLORS[1], marker='o', ls='', ms=3.5, label=r'path coverage $>0.95$'),
	], loc='upper left')
	ax.add_artist(crit)
	_legend(ax, [
		Line2D([], [], color='0.35', marker='o', ls='', ms=3.5, label='plain'),
		Line2D([], [], color='0.35', marker='o', ls='', ms=3.5, mfc='none', label='remapped'),
	], loc='upper left', bbox=(0.0, 0.80))
	return _save(fig, name)


def plot_length_vs_Lmax(results, name):
	"""
	x = L_max (the DAG's longest path), y = smallest training length L reaching a
	criterion, averaged over the seeds (runs) with +/- SEM error bars. Criteria:
	illegal (blue) and coverage (green), each with a solid linear fit L = m * L_max
	+ c. Grey diamonds are the longest path the model covers (assigns probability
	>= PATH_RATIO * P_true), measured on the same model that reaches the coverage
	criterion (the smallest-L model with path_coverage > 0.95), with its own solid
	fit -- comparing the training length needed against how far that model's
	coverage actually reaches.
	"""
	fig, ax = plt.subplots(figsize=(3, 2.5))
	graphs = []
	for r in results:
		g = (r['V'], r['n'])
		if g not in graphs:
			graphs.append(g)

	fits = {}
	fit_lines = []          # (fit_params, color) to draw across the full x-range later
	for key, col in [('minL_illegal', CURVE_COLORS[0]), ('minL_coverage', CURVE_COLORS[1])]:
		fx, fy = [], []
		for g in graphs:
			group = [r for r in results if (r['V'], r['n']) == g]
			a = _agg_over_seeds(group, key)
			if not a:
				continue
			x = float(np.mean([r['L_max'] for r in group]))
			ax.errorbar(x, a[0], yerr=a[1], fmt='o', ms=3.8, color=col, alpha=0.85,
						elinewidth=0.7, capsize=0, zorder=3)
			fx.append(x)
			fy.append(a[0])
		fits[key] = _linear_fit(fx, fy)
		if fits[key]:
			fit_lines.append((fits[key], col))

	# longest path the model covers (P_model >= PATH_RATIO * P_true), measured on the
	# model at the coverage crossing, per graph, as grey diamonds -- one value per
	# seed (its own DAG), so a SEM like the other points. Seeds whose coverage never
	# reaches the criterion have no such model (max_covered is None) and are skipped.
	COV_COL = '0.4'
	gx, gy, gyerr = [], [], []
	for g in graphs:
		group = [r for r in results if (r['V'], r['n']) == g and r.get('max_covered') is not None]
		if not group:
			continue
		mc = np.array([r['max_covered'] for r in group], float)
		gx.append(float(np.mean([r['L_max'] for r in group])))
		gy.append(float(mc.mean()))
		gyerr.append(float(mc.std(ddof=1) / np.sqrt(len(mc))) if len(mc) > 1 else 0.0)
	if gx:
		ax.errorbar(gx, gy, yerr=gyerr, fmt='D', ms=3.5, mew=1.0, color=COV_COL,
					elinewidth=0.7, capsize=0, zorder=3)
		fits['max_covered'] = _linear_fit(gx, gy)
		if fits['max_covered']:
			fit_lines.append((fits['max_covered'], COV_COL))

	# draw each fit across the full x-range, holding the data-driven limits so the
	# lines span the whole plot without expanding the axes
	xlim = ax.get_xlim()
	gx_full = np.array(xlim)
	for (m, c), col in fit_lines:
		ax.plot(gx_full, m * gx_full + c, ls='-', lw=1.6, color=col, zorder=1)
	ax.set_xlim(xlim)

	ax.set_xlabel(r'$L_{\max}$', fontsize=LABEL_FS)
	ax.set_ylabel('Length', fontsize=LABEL_FS)
	_style_axis(ax)
	_legend(ax, [
		Line2D([], [], color=CURVE_COLORS[0], marker='o', ls='', ms=3.5, label=r'$L_{\text{train}}$: $P($invalid tokens$) < 0.05$'),
		Line2D([], [], color=CURVE_COLORS[1], marker='o', ls='', ms=3.5, label=r'$L_{\text{train}}$: path coverage $> 0.95$'),
		Line2D([], [], color=COV_COL, marker='D', ls='', ms=3.5, label='longest learned path'),
		Line2D([], [], color='0.35', ls='-', lw=1.6, label='fit'),
	], loc='upper left')
	return _save(fig, name), fits


# ----------------------------------------------------------------------------
# top-level figure orchestration
# ----------------------------------------------------------------------------

def write_provenance(name, kind, profile, ordering, results, fits=None):
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
			'dag_seed': 'equal to seed, so each seed draws a fresh DAG (spread reflects graph variability)',
			'train_sampler_rng': '[seed, 1]', 'test_sampler_rng': '[seed, 2]',
			'perm_seed': 'seed', 'weight_init + shuffle seed': 'TrainConfig.seed = seed',
		},
	)
	if kind == 'samples':
		# power-law fits y = c * x^b (x = distinct edges, y = samples), least squares
		# in log10 over the plain (non-permuted, non-censored) points only
		fit_scope = 'plain (non-permuted, non-censored) points'
		labels = {'samples_illegal': 'illegal_mass < %s' % ILLEGAL_CRIT,
				  'samples_coverage': 'path_coverage > %s' % PATH_COV_CRIT}
		fit_summary = {}
		for key, label in labels.items():
			f = (fits or {}).get(key)
			fit_summary[label] = (None if not f else dict(
				coefficient_c=f[1], exponent_b=f[0],
				equation='y = %.6g * x^%.6g' % (f[1], f[0]),
				fitted_over=fit_scope))
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
		labels['max_covered'] = 'longest covered path (P_model >= %s * P_true) vs L_max' % DEFAULT_PATH_RATIO
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
						'with the longest covered path (max_covered) for reference',
			length_epochs=profile['length_epochs'],
			length_num_train=profile['length_num_train'],
			L_grid=profile['L_grid'],
			note='L sweep runs up to each DAG\'s L_max; stops early once path_coverage > threshold',
			fits=fit_summary,
			units=[dict(V=r['V'], n=r['n'], seed=r['seed'], num_states=r['num_states'],
						num_edges=r['num_edges'], L_max=r.get('L_max'),
						max_covered=r.get('max_covered'),
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
	path, fits = plot_samples_vs_edges(results, name, fit_plain_only=False)
	write_provenance(name, 'samples', profile, ordering, results, fits=fits)
	print(f"[{name}] saved {path}.pdf/.png (+ {name}.json, {name}_provenance.json)")


def plot_samples_vs_edges_minimal(results, name):
	"""
	Like plot_samples_vs_edges but trained on the optimal P_min edge covering: one
	point per graph (mean over seeds), x = edges, y = samples to reach each criterion
	(illegal blue, coverage green), log-log with a solid power-law fit. Standalone
	(own labeled 'Samples' axis); there is no remapped control for this variant.
	"""
	fig, ax = plt.subplots(figsize=(3, 2.5))
	graphs = []
	for r in results:
		g = (r['V'], r['n'])
		if g not in graphs:
			graphs.append(g)

	any_pts = False
	fits = {}
	fit_lines = []
	for key, col in [('samples_illegal', CURVE_COLORS[0]), ('samples_coverage', CURVE_COLORS[1])]:
		fx, fy = [], []
		for g in graphs:
			group = [r for r in results if (r['V'], r['n']) == g]
			a = _agg_over_seeds(group, key, drop_censored=True) if group else None
			if not a:
				continue
			edges = float(np.mean([r['distinct_edges'] for r in group]))
			any_pts = True
			ax.errorbar(edges, a[0], yerr=a[1], fmt='o', ms=3.8, color=col,
						ecolor=col, elinewidth=0.7, capsize=0, alpha=0.85, zorder=3)
			fx.append(edges)
			fy.append(a[0])
		fits[key] = _loglog_fit(fx, fy)
		if fits[key]:
			fit_lines.append((fits[key], col))

	if any_pts:
		ax.set_xscale('log')
		ax.set_yscale('log')
	xlim = ax.get_xlim()
	gx_full = np.array(xlim)
	for (b, c), col in fit_lines:
		ax.plot(gx_full, c * gx_full ** b, ls='-', lw=1.6, color=col, zorder=0)
	ax.set_xlim(xlim)
	ax.set_xlabel('Edges', fontsize=LABEL_FS)
	ax.set_ylabel('Samples', fontsize=LABEL_FS)
	_style_axis(ax)
	_legend(ax, [
		Line2D([], [], color=CURVE_COLORS[0], marker='o', ls='', ms=3.5, label=r'$P($invalid tokens$) < 0.05$'),
		Line2D([], [], color=CURVE_COLORS[1], marker='o', ls='', ms=3.5, label=r'path coverage $> 0.95$'),
		Line2D([], [], color='0.35', ls='-', lw=1.6, label='fit'),
	], loc='upper left')
	return _save(fig, name), fits


def build_samples_vs_edges_minimal(profile, ordering, devices, force, name):
	"""Train on samples from the optimal P_min edge covering (samples_minimal_unit) and
	plot samples-to-criterion vs edges. The control behind the note that a minimal
	cover is not more sample efficient than uniform sampling."""
	jobs = [dict(profile=profile, ordering=ordering, graph=tuple(g), seed=seed)
			for g in profile['graphs']
			for seed in profile['seeds']]
	print(f"[{name}] {len(jobs)} units")
	results = _run_units(samples_minimal_unit, jobs, devices, force, desc=name)
	os.makedirs(FIG_DIR, exist_ok=True)
	with open(os.path.join(FIG_DIR, name + '.json'), 'w') as f:
		json.dump(results, f, indent=1)
	path, fits = plot_samples_vs_edges_minimal(results, name)
	write_provenance(name, 'samples', profile, ordering, results, fits=fits)
	print(f"[{name}] saved {path}.pdf/.png (+ {name}.json, {name}_provenance.json)")


def build_sample_efficiency_vs_edges(profile, ordering, devices, force, name):
	"""Reuse the cached samples_vs_edges results (no retraining) and plot sample
	efficiency = (min paths to cover all edges) / samples. Reads the samples figure
	JSON produced by build_samples_vs_edges for the same ordering."""
	samples_name = 'samples_vs_edges' + ('_digit_sum' if ordering == 'digit_sum' else '')
	src = os.path.join(FIG_DIR, samples_name + '.json')
	if not os.path.exists(src):
		raise FileNotFoundError(f"{src} not found -- build {samples_name} first "
								f"(this plot reuses its results, it does not retrain)")
	with open(src) as f:
		results = json.load(f)
	mincover = _min_cover_by_graph(results, profile, ordering)
	path = plot_sample_efficiency_vs_edges(results, name, mincover)
	print(f"[{name}] saved {path}.pdf/.png (from {samples_name}.json, no retraining)")


def build_length_vs_Lmax(profile, ordering, devices, force, name, windowed=False,
						 augmented=False, hint_index=0):
	jobs = [dict(profile=profile, ordering=ordering, graph=tuple(g), seed=seed,
				 windowed=windowed, augmented=augmented)
			for g in profile['graphs']
			for seed in profile['seeds']]
	print(f"[{name}] {len(jobs)} units")
	results = _run_units(length_unit, jobs, devices, force, desc=name)
	os.makedirs(FIG_DIR, exist_ok=True)
	with open(os.path.join(FIG_DIR, name + '.json'), 'w') as f:
		json.dump(results, f, indent=1)
	# hint_index k > 0 re-plots the same (shared) units with k free hints per path,
	# by swapping in the k-th slice of each unit's stored hint curve
	if hint_index:
		plot_results = [dict(r, minL_illegal=r['minL_illegal_by_hints'][hint_index],
							 minL_coverage=r['minL_coverage_by_hints'][hint_index]) for r in results]
	else:
		plot_results = results
	path, fits = plot_length_vs_Lmax(plot_results, name)
	write_provenance(name, 'length', profile, ordering, results, fits=fits)
	print(f"[{name}] saved {path}.pdf/.png (+ {name}.json, {name}_provenance.json)")


FIGURES = {
	'samples_vs_edges': lambda p, d, f: build_samples_vs_edges(p, 'random', d, f, 'samples_vs_edges'),
	# sample efficiency = min-edge-cover paths / samples; reuses samples_vs_edges results (no retrain)
	'sample_efficiency_vs_edges': lambda p, d, f: build_sample_efficiency_vs_edges(p, 'random', d, f, 'sample_efficiency_vs_edges'),
	'length_vs_Lmax': lambda p, d, f: build_length_vs_Lmax(p, 'random', d, f, 'length_vs_Lmax'),
	'samples_vs_edges_digit_sum': lambda p, d, f: build_samples_vs_edges(p, 'digit_sum', d, f, 'samples_vs_edges_digit_sum'),
	'sample_efficiency_vs_edges_digit_sum': lambda p, d, f: build_sample_efficiency_vs_edges(p, 'digit_sum', d, f, 'sample_efficiency_vs_edges_digit_sum'),
	'length_vs_Lmax_digit_sum': lambda p, d, f: build_length_vs_Lmax(p, 'digit_sum', d, f, 'length_vs_Lmax_digit_sum'),
	# windowed (local) attention: length generalisation with a sliding window of width n
	'length_vs_Lmax_windowed': lambda p, d, f: build_length_vs_Lmax(p, 'random', d, f, 'length_vs_Lmax_windowed', windowed=True),
	'length_vs_Lmax_windowed_digit_sum': lambda p, d, f: build_length_vs_Lmax(p, 'digit_sum', d, f, 'length_vs_Lmax_windowed_digit_sum', windowed=True),
	# position-augmented training: random non-contiguous RoPE position gaps on the short paths
	'length_vs_Lmax_augmented': lambda p, d, f: build_length_vs_Lmax(p, 'random', d, f, 'length_vs_Lmax_augmented', augmented=True),
	'length_vs_Lmax_augmented_digit_sum': lambda p, d, f: build_length_vs_Lmax(p, 'digit_sum', d, f, 'length_vs_Lmax_augmented_digit_sum', augmented=True),
	# --minimal: samples_vs_edges but trained on the optimal P_min edge covering (control:
	# a minimal cover is not more sample efficient than uniform sampling)
	'samples_vs_edges_minimal': lambda p, d, f: build_samples_vs_edges_minimal(p, 'random', d, f, 'samples_vs_edges_minimal'),
	'samples_vs_edges_minimal_digit_sum': lambda p, d, f: build_samples_vs_edges_minimal(p, 'digit_sum', d, f, 'samples_vs_edges_minimal_digit_sum'),
	# the k-hint length figure is produced via `--hints k` (build_length_vs_Lmax with
	# hint_index=k), reusing the shared length units -- no separate FIGURES entry.
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
					help="also make the digit-sum-ordering versions of all figures")
	ap.add_argument('--windowed-attention', action='store_true',
					help="also repeat length_vs_Lmax with sliding-window (local) "
						 "attention of width n (off by default)")
	ap.add_argument('--augmented', action='store_true',
					help="also repeat length_vs_Lmax with random RoPE position "
						 "augmentation on the short training paths (off by default)")
	ap.add_argument('--minimal', action='store_true',
					help="also make samples_vs_edges_minimal: samples_vs_edges trained on "
						 "the optimal P_min edge covering instead of uniform paths (off by default)")
	ap.add_argument('--hints', type=int, default=0, metavar='K',
					help=f"make length_vs_Lmax_Khints: the length figure with K free "
						 f"ground-truth hints per path (K <= {HINT_MAX}); reuses the shared "
						 f"length units, so it never retrains. Alone, only this figure is built")
	ap.add_argument('--workers', type=int, default=None, metavar='N',
					help="parallel worker processes for the unit sweep (default: one per "
						 "device). Since the work is CPU-bound and the models tiny, N > #devices "
						 "packs several units per GPU and speeds up the CPU parts; each unit is "
						 "still computed once, so there is no redundant recomputation")
	ap.add_argument('--force', action='store_true', help="recompute, ignoring the unit cache")
	args = ap.parse_args()

	if args.hints < 0 or args.hints > HINT_MAX:
		ap.error(f"--hints must be between 0 and HINT_MAX={HINT_MAX} "
				 f"(raise HINT_MAX in run_experiments.py for more)")
	if args.workers is not None and args.workers < 1:
		ap.error("--workers must be >= 1")

	global _N_WORKERS
	_N_WORKERS = args.workers

	profile_name = args.profile or ('full' if torch.cuda.is_available() else 'laptop')
	profile = PROFILES[profile_name]
	devices = args.devices.split(',') if args.devices else _default_devices()
	if args.figures:                            # explicit subset overrides everything
		names = args.figures.split(',')
	else:                                       # gate the optional families behind flags
		names = [f for f in FIGURES
				 if (args.digit_sum or not f.endswith('_digit_sum'))
				 and (args.windowed_attention or '_windowed' not in f)
				 and (args.augmented or '_augmented' not in f)
				 and (args.minimal or '_minimal' not in f)]

	# --hints K *adds* the K-hint length figure(s) on top of the selection above,
	# reusing the shared length units (no retrain). One per active ordering, so
	# --digit-sum also yields the digit-sum hint figure.
	hint_figs = []
	if args.hints:
		for ordering in ['random'] + (['digit_sum'] if args.digit_sum else []):
			suffix = '' if ordering == 'random' else '_digit_sum'
			hint_figs.append((ordering, f'length_vs_Lmax_{args.hints}hints{suffix}'))

	print(f"profile={profile_name} devices={devices} figures={names}"
		  + (f" +{[h[1] for h in hint_figs]}" if hint_figs else ""))
	t0 = time.time()
	for name in names:
		FIGURES[name](profile, devices, args.force)
	for ordering, name in hint_figs:
		build_length_vs_Lmax(profile, ordering, devices, args.force, name, hint_index=args.hints)
	print(f"done in {time.time() - t0:.1f}s")


if __name__ == '__main__':
	main()
