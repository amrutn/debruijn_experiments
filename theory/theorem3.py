"""
Branching structure of the reasoning prefix-tree.

Sample random-order permutations pi (using the DeBruijn_DAG data structure from
theorem1_theorem2.py), unroll the DAG D_pi into the prefix-tree of all
source-to-sink reasoning traces, and plot the average node out-degree (the number
of plausible next reasoning steps) against depth in that tree. One curve per
vocabulary size V at a fixed n-gram length n; averaged over random permutations.
Same S, N, formatting as theorem1_theorem2.py.

The prefix-tree is exponentially large, so we never materialise it. Instead we
propagate the distribution of where a random depth-d prefix ends: a depth-d node
of the tree that sits on DAG node v has out-degree outdeg(v), and the number of
such tree nodes is W_d(v) = #(length-d walks from a source to v). Hence the mean
out-degree at depth d is sum_v W_d(v) outdeg(v) / sum_v W_d(v). Writing q_d for the
normalised W_d (a distribution over nodes), the moments at depth d are just
q_d-weighted moments of outdeg, and q_{d+1} is proportional to q_d propagated along
edges -- no overflow, no enumeration.

Sink nodes (out-degree 0) are the leaves / completed traces; they are EXCLUDED from
the average, so each curve is the mean out-degree over the still-branching internal
nodes and stays >= 1 rather than tailing to 0.

The shaded band is +/- 1 std of the out-degree ACROSS prefix-tree nodes at that
depth (the spread of branching factors), pooled over the sampled permutations --
not the std of the per-depth mean across permutations, which is negligible (the
tree-averaged degree barely changes from one random ordering to the next, which is
why a cross-permutation band is invisible).
"""

import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from theorem1_theorem2 import (DeBruijn_DAG, CURVE_COLORS, CURVE_MARKERS,
							   LABEL_FS, TICK_FS, LEGEND_FS, _style_axis)


def degree_profile(graph):
	"""
	Per-depth first and second moments of the out-degree over NON-SINK prefix-tree
	nodes at each depth d (d = 0 is the sources), walk-weighted. Returns
	(means, second_moments) as depth-indexed lists, or None if the DAG has no paths.
	Sinks (out-degree 0, the completed-trace leaves) are excluded from the moments.
	"""
	M = graph.num_nodes_all
	if graph.num_edges == 0 or len(graph.source_vals) == 0:
		return None

	outdeg = np.bincount(graph.edges_u, minlength=M).astype(float)
	nonsink = (~graph.is_sink).astype(float)
	eu, ev = graph.edges_u, graph.edges_v

	# q = distribution over DAG nodes of where a random depth-d prefix ends
	q = np.zeros(M)
	q[graph.source_vals] = 1.0 / len(graph.source_vals)

	means, second = [], []
	while True:
		mass = float((q * nonsink).sum())        # walk mass on non-sink nodes
		if mass <= 0:                             # nothing left but leaves
			break
		w = (q * nonsink) / mass                  # distribution over non-sink nodes
		means.append(float((w * outdeg).sum()))
		second.append(float((w * outdeg ** 2).sum()))
		newq = np.zeros(M)
		np.add.at(newq, ev, q[eu])                # push mass along every edge
		s = newq.sum()
		if s <= 0:
			break
		q = newq / s
	return means, second


def sample_degree_profile(V, n, S, N, rng, max_retries=50):
	"""One random permutation's (means, second_moments) profile; retries empties."""
	for _ in range(max_retries):
		graph = DeBruijn_DAG(V, n, S, N, rng=rng)
		if graph.num_edges > 0 and len(graph.source_vals) > 0:
			prof = degree_profile(graph)
			if prof is not None:
				return prof
	return None


def run_experiments(n=4, Vs=(8, 12, 16, 20), n_trials=40, S=5, N=10, seed=0):
	"""
	At fixed n, for each vocabulary size V average the degree-vs-depth profile over
	n_trials random permutations. Per depth we record the mean out-degree, the std
	of out-degree across prefix-tree nodes at that depth (spread, pooled over
	samples as the mean within-sample variance), and the sample count.
	"""
	results = {}
	for V in Vs:
		prof_m, prof_sq = [], []
		for t in range(n_trials):
			rng = np.random.default_rng([seed, V, n, t])
			prof = sample_degree_profile(V, n, S, N, rng)
			if prof is not None:
				prof_m.append(prof[0])
				prof_sq.append(prof[1])
		max_depth = max(len(m) for m in prof_m)
		mean, std, count = [], [], []
		for d in range(max_depth):
			ms = [m[d] for m in prof_m if len(m) > d]
			var = [sq[d] - m[d] ** 2 for m, sq in zip(prof_m, prof_sq) if len(m) > d]
			mean.append(float(np.mean(ms)))
			std.append(float(np.sqrt(max(0.0, np.mean(var)))))
			count.append(len(ms))
		results[V] = {'n': n, 'mean': mean, 'std': std, 'count': count,
					  'n_trials': len(prof_m)}
		print(f"n={n} V={V}: {len(prof_m)} samples, max depth {max_depth}, "
			  f"deg[0]={mean[0]:.2f} +- {std[0]:.2f} -> "
			  f"deg[-1]={mean[-1]:.2f} +- {std[-1]:.2f}")
	return results


def make_plot(results, outdir, min_frac=0.25, xmax=20):
	"""Mean out-degree vs depth (log y), one colored curve per V, with a shaded
	band = +/- 1 std of out-degree across prefix-tree nodes at that depth. Depths
	reached by fewer than min_frac of the samples are dropped (noisy tail); the
	x-axis is capped at xmax."""
	keys = sorted(results.keys())
	ns = {r['n'] for r in results.values()}
	fig, ax = plt.subplots(figsize=(3, 2.5))
	for i, k in enumerate(keys):
		r = results[k]
		thr = max(3, min_frac * r['n_trials'])
		keep = np.array(r['count']) >= thr
		d = np.arange(len(r['mean']))[keep]
		mu = np.array(r['mean'])[keep]
		sd = np.array(r['std'])[keep]
		col = CURVE_COLORS[i]
		ax.fill_between(d, np.clip(mu - sd, 1e-9, None), mu + sd, color=col,
						alpha=0.18, lw=0)
		ax.plot(d, mu, color=col, marker=CURVE_MARKERS[i], ms=3.5, lw=1.4,
				markevery=3, label=f'$V={k}$')
	ax.set_xlim(0, xmax)
	ax.set_ylim(bottom=0)
	ax.set_xlabel('Prefix Tree Depth', fontsize=LABEL_FS)
	ax.set_ylabel('Out Degree', fontsize=LABEL_FS)
	_style_axis(ax)

	handles = [Line2D([], [], color=CURVE_COLORS[i], marker=CURVE_MARKERS[i],
					  ms=3.5, lw=1.4, label=f'$V={k:g}$') for i, k in enumerate(keys)]
	title = f'$n={ns.pop()}$' if len(ns) == 1 else None
	leg = ax.legend(handles=handles, fontsize=LEGEND_FS, frameon=False,
					ncol=2, loc='upper right', handlelength=1.4, columnspacing=0.8,
					labelspacing=0.2, handletextpad=0.4,
					title=title, title_fontsize=LEGEND_FS)
	leg._legend_box.align = 'left'

	figdir = os.path.join(outdir, 'figures')
	os.makedirs(figdir, exist_ok=True)
	outpath = os.path.join(figdir, 'theorem3_degree_depth')
	fig.savefig(outpath + '.pdf', bbox_inches='tight')
	fig.savefig(outpath + '.png', bbox_inches='tight', dpi=300)
	plt.close(fig)


def main():
	outdir = os.path.dirname(os.path.abspath(__file__))
	cachedir = os.path.join(outdir, 'cache')
	os.makedirs(cachedir, exist_ok=True)
	cache = os.path.join(cachedir, 'theorem3_data.json')

	if os.path.exists(cache) and '--force' not in sys.argv:
		with open(cache) as f:
			results = {int(V): v for V, v in json.load(f).items()}
		print(f"loaded cached results from {cache} (use --force to recompute)")
	else:
		results = run_experiments(n=4, Vs=(8, 12, 16, 20), n_trials=40, S=5, N=10)
		with open(cache, 'w') as f:
			json.dump({str(V): v for V, v in results.items()}, f, indent=1)

	make_plot(results, outdir)
	print("saved figures/theorem3_degree_depth.{pdf,png}")


if __name__ == '__main__':
	main()
