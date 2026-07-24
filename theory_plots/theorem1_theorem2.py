"""
Averaged over random permutations pi (with the digit-sum bound as reference):
1. Plot min paths to cover/total paths while increasing $V^n$ s.t. $\\log_2(V)/n=c$ is fixed.
	Plot curves for multiple values of $c$. Include the theoretical bound from theorem 1.
2. Plot ratio of min_length_to_cover/max_length while increasing $V^n$ s.t. $\\log_2(V)/n=c$ is fixed.
	Plot curves for multiple values of $c$. Include the theoretical bound from theorem 2.

Notes
-----
- Each n gets three curves vs V^n, all on a linear y-axis (both ratios are in
  [0, 1]): (i) the mean over n_trials random permutations pi (solid line + marker,
  shaded +/- 1 std band); (ii) the single deterministic digit-sum ordering used in
  the proofs (dashed); (iii) the digit-sum theorem bound (dotted). Color encodes n.
- The digit-sum ordering (nodes sorted by digit sum, ties broken by the reversed-
  string lexicographic comparison, which in our encoding is the node value
  descending) is the ordering the theorems are proven for, so its curve sits at or
  below the bound. A random ordering does not achieve it: for plot 1 the random
  average stays well above the near-zero digit-sum curve -- the gap between "a
  typical ordering" and "the optimal ordering".
- Each curve holds the n-gram length n fixed and sweeps integer V; the point at
  V is placed at V^n. Fixing n keeps the curves monotone (fixing log2(V)/n instead
  makes n jump in integer steps, so those curves sawtooth). Along a fixed-n curve
  the tightest reference m is constant, so the theory curve is smooth too.
- Total path counts explode combinatorially (far beyond float64 range), so all
  path counts are computed in log10-space.
- The minimum number of paths covering every edge is computed exactly as a
  minimum flow with a lower bound of 1 on every edge (two max-flow passes),
  NOT with the degree-imbalance formula sum_v max(0, out-in), which is only an
  upper bound (a cover path may traverse an edge that other cover paths also
  need, so edges can carry flow > 1). Max-flow runs on a numba-compiled Dinic
  implementation when numba is available (scipy 1.7's maximum_flow returns
  wrong values on some instances, and networkx is too slow for the dense
  digit-sum DAGs); networkx is kept as a fallback and as a cross-check in the
  self tests.
- We use constant S = 5 source and N = 10 sink nodes. The empirical average does
  not depend on m; m only sets the reference theory bound. For each (V, n) point m
  is the value giving the tightest theorem bounds: the smallest m with
  S, N < C(n+m, m), subject to 0 < m < (V-1)/2. Both bounds tighten as m shrinks,
  so the smallest feasible m is optimal for both. Points (V, n) that admit no such
  m (V too small) have no theory reference and are skipped.
"""

import os
import sys
import json
import math
from math import comb

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, dijkstra
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
	from numba import njit
	HAVE_NUMBA = True
except ImportError:
	HAVE_NUMBA = False


# def de bruijn dag class

class DeBruijn_DAG:
	"""
	A class that represents a DAG constructed from a De Bruijn graph. We provide a vocabulary size,
	n-gram length, node topological sort, source nodes, sink nodes and we would like to query
	1. in/out neighbors of each node
	2. in/out degrees of each node
	3. Source nodes
	4. Sink nodes

	Each node is represented by a list of n ints between 0, V-1. Each int represents
	a particular character.

	Use node_to_val and val_to_node to convert between the numerical representation and
	a n-gram node representation.

	The functions are:
	1. set_random_ordering : reset the topological sort to a random sample.
	2. node_to_val : convert a node list representation into a single number between 0, V^n.
	3. val_to_node : convert a number between 0, V^n into the node list representation.
	4. get_incoming_nodes : returns a list of the incoming nodes to the input
	5. get_outgoing_nodes : returns a list of the outgoing nodes to the input
	6. get_degree : returns a tuple of the in-degree and the out-degree of the input node

	Internally the pruned DAG is stored as flat edge arrays (edges_u, edges_v) over
	node values, which is what the metric functions below operate on.
	"""

	def __init__(self, V, n, S, N, pi=None, rng=None):
		"""
		Initialize the DAG.

		Params
		------
		V : int
			Vocabulary size.
		n : int
			n-gram length.
		S : int
			Number of source nodes.
		N : int
			Number of sink nodes
		pi : np array
			pi[i] is the value of the node at position i of the topological sort.
		rng : random number generator (numpy class)
		"""

		self.V = V
		self.n = n
		self.num_nodes_all = V ** n
		self.S = S
		self.N = N

		# setting the rng
		if rng is None:
			self.rng = np.random.default_rng(seed=42)
		else:
			self.rng = rng

		# generating a random topological sort if one is not provided
		if pi is None:
			self.pi_all = self.rng.permutation(self.num_nodes_all)
		else:
			self.pi_all = np.asarray(pi)

		self._build_dag()

	def set_random_ordering(self):
		"""
		Sets the topological sort to a random sample. Resets the source and sink nodes
		and rebuilds the pruned DAG.
		"""
		self.pi_all = self.rng.permutation(self.num_nodes_all)
		self._build_dag()

	def _build_dag(self):
		"""
		Applies the procedure F: keeps only forward-pointing De Bruijn edges
		(pi(u) < pi(v)), removes outgoing edges of sinks / incoming edges of
		sources, and prunes every node and edge not on a source-to-sink path.
		"""
		V, n, M = self.V, self.n, self.num_nodes_all

		# position of each node value in the ordering
		pos = np.empty(M, dtype=np.int64)
		pos[self.pi_all] = np.arange(M)
		self.pos = pos
		self.is_source = pos < self.S
		self.is_sink = pos >= M - self.N

		# all De Bruijn successors of node value v are v//V + b*V^(n-1), b in [0, V).
		# Scan one branch character b at a time to keep peak memory at O(M) rather
		# than O(M*V) -- the (M, V) form is prohibitive for large V^n (e.g. n=5).
		Vp = V ** (n - 1)
		base = np.arange(M) // V          # first n-1 chars shifted, shared by all b
		not_sink_u = ~self.is_sink
		is_source = self.is_source
		eu_parts, ev_parts = [], []
		for b in range(V):
			ev_b = base + b * Vp          # successor value when appending char b
			keep_b = not_sink_u & (pos < pos[ev_b]) & (~is_source[ev_b])
			idx = np.nonzero(keep_b)[0]
			eu_parts.append(idx)
			ev_parts.append(ev_b[idx])
		eu = np.concatenate(eu_parts)
		ev = np.concatenate(ev_parts)
		del base, eu_parts, ev_parts

		# prune nodes not on any source-to-sink path: a node survives iff it is
		# reachable from a source (forward BFS) AND it reaches a sink (backward BFS)
		s_star, t_star = M, M + 1
		src_vals = self.pi_all[:self.S]
		snk_vals = self.pi_all[M - self.N:]
		rows = np.concatenate([eu, np.full(self.S, s_star), snk_vals])
		cols = np.concatenate([ev, src_vals, np.full(self.N, t_star)])
		G = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
					   shape=(M + 2, M + 2))
		del rows, cols
		fwd = breadth_first_order(G, s_star, directed=True, return_predecessors=False)
		bwd = breadth_first_order(G.T.tocsr(), t_star, directed=True, return_predecessors=False)
		reach_f = np.zeros(M + 2, dtype=bool)
		reach_f[fwd] = True
		reach_b = np.zeros(M + 2, dtype=bool)
		reach_b[bwd] = True
		alive = reach_f[:M] & reach_b[:M]
		self.alive = alive

		# surviving edges: both endpoints alive (then source->u, edge, v->sink
		# concatenate into a valid source-to-sink path, so the edge survives too)
		mask = alive[eu] & alive[ev]
		self.edges_u = eu[mask]
		self.edges_v = ev[mask]
		self.num_edges = len(self.edges_u)

		# the pruned ordering; sources sit at the front, sinks at the back,
		# but note that fewer than S sources / N sinks may survive pruning
		self.pi = self.pi_all[alive[self.pi_all]]
		self.num_nodes = len(self.pi)
		self.source_vals = src_vals[alive[src_vals]]
		self.sink_vals = snk_vals[alive[snk_vals]]

		# successor adjacency (CSR over node values) for the DP passes
		A = csr_matrix((np.ones(self.num_edges, dtype=np.int8),
						(self.edges_u, self.edges_v)), shape=(M, M))
		self._succ_indptr = A.indptr
		self._succ_indices = A.indices
		self._dp_cache = None

	def node_to_val(self, node):
		"""
		Converts a node representation (list of ints) into a numerical value.
		node[0] is the first character of the n-gram (least significant digit).
		"""
		assert len(node) == self.n
		assert all([a < self.V and a >= 0 for a in node])

		return int(sum(a * self.V ** k for k, a in enumerate(node)))

	def val_to_node(self, val):
		"""
		Converts a numerical value back into a node representation
		"""
		assert val >= 0
		assert val < self.num_nodes_all

		node = []
		for k in range(self.n):
			node.append(int(val % self.V))
			val = val // self.V
		return node

	def get_incoming_nodes(self, node, use_all=False):
		"""
		Returns a list of the incoming nodes to a given node.

		Params
		------
		node : list
			A list of n integers between 0 and V-1
		use_all : bool
			If True, use the un-pruned DAG (before deleting nodes/edges that are
			not on source-to-sink paths); otherwise only pruned-DAG neighbors.

		Returns
		-------
		incoming : list
			A list of nodes that are incoming to the input
		"""
		val = self.node_to_val(node)

		# a source has no incoming edges
		if self.is_source[val]:
			return []
		if not use_all and not self.alive[val]:
			return []

		Vp = self.V ** (self.n - 1)
		incoming = []
		for c in range(self.V):
			# predecessor prepends character c and drops the last character
			pred = int(c + self.V * (val % Vp))
			if self.is_sink[pred]:
				continue
			if self.pos[pred] >= self.pos[val]:
				continue
			if not use_all and not self.alive[pred]:
				continue
			incoming.append(self.val_to_node(pred))
		return incoming

	def get_outgoing_nodes(self, node, use_all=False):
		"""
		Returns a list of the outgoing nodes from a given node.

		Params
		------
		node : list
			A list of n integers between 0 and V-1
		use_all : bool
			If True, use the un-pruned DAG; otherwise only pruned-DAG neighbors.

		Returns
		-------
		outgoing : list
			A list of nodes that are outgoing from the input
		"""
		val = self.node_to_val(node)

		# a sink has no outgoing edges
		if self.is_sink[val]:
			return []
		if not use_all and not self.alive[val]:
			return []

		Vp = self.V ** (self.n - 1)
		outgoing = []
		for b in range(self.V):
			# successor drops the first character and appends character b
			nxt = int(val // self.V + b * Vp)
			if self.is_source[nxt]:
				continue
			if self.pos[val] >= self.pos[nxt]:
				continue
			if not use_all and not self.alive[nxt]:
				continue
			outgoing.append(self.val_to_node(nxt))
		return outgoing

	def get_degree(self, node):
		"""
		Returns a tuple of the (in-degree, out-degree) of a node in the pruned DAG.

		Params
		------
		node : list
			A list of n integers between 0 and V-1
		Returns
		-------
		degrees : tuple
			Tuple of (in-degree, out-degree)
		"""
		return (len(self.get_incoming_nodes(node)), len(self.get_outgoing_nodes(node)))


def digit_sum_ordering(V, n):
	"""
	The node ordering used in the proofs of Theorems 1 and 2: sort by digit sum
	ascending; break ties by the reversed-string lexicographic comparison
	(if two nodes have equal digit sum and the reversed string of v is
	lexicographically smaller, v comes LATER). In our encoding node[0] is the
	least significant digit, so the reversed-string comparison is simply the
	integer value: ties are broken by node value descending.

	Returns pi such that pi[i] is the node value at position i.
	"""
	M = V ** n
	vals = np.arange(M, dtype=np.int64)
	digit_sum = np.zeros(M, dtype=np.int64)
	x = vals.copy()
	for _ in range(n):
		digit_sum += x % V
		x //= V
	# np.lexsort: last key is primary
	return np.lexsort((-vals, digit_sum))


# Define helper functions to:
# 1. calc_log10_num_paths : log10 of the total number of source-to-sink paths.
# 2. calc_max_length : length of the longest source-to-sink path.
# 3. calc_min_cover : minimum number of paths needed to cover every edge (exact, via min flow).
# 4. calc_min_length_cover : minimum path length L such that every edge lies on a
#    source-to-sink path of length <= L.

if HAVE_NUMBA:

	@njit(cache=True)
	def _topo_dp_kernel(order_rev, indptr, indices, is_sink):
		"""
		Reverse-topological sweep: for every surviving node, the log of the
		number of paths to a sink and the longest path to a sink.
		"""
		M = is_sink.shape[0]
		log_paths = np.zeros(M, dtype=np.float64)
		longest = np.zeros(M, dtype=np.int64)
		for idx in range(order_rev.shape[0]):
			v = order_rev[idx]
			if is_sink[v]:
				log_paths[v] = 0.0
				longest[v] = 0
				continue
			lo = indptr[v]
			hi = indptr[v + 1]
			mx = -1.0e300
			mg = 0
			for j in range(lo, hi):
				w = indices[j]
				if log_paths[w] > mx:
					mx = log_paths[w]
				if longest[w] > mg:
					mg = longest[w]
			acc = 0.0
			for j in range(lo, hi):
				acc += math.exp(log_paths[indices[j]] - mx)
			log_paths[v] = mx + math.log(acc)
			longest[v] = 1 + mg
		return log_paths, longest

	@njit(cache=True)
	def _build_adjacency(arc_from, head, nxt):
		"""Linked-list adjacency: head[u] = last arc out of u, nxt[a] = previous."""
		for a in range(arc_from.shape[0]):
			u = arc_from[a]
			nxt[a] = head[u]
			head[u] = a

	@njit(cache=True)
	def _dinic(head, nxt, arc_to, arc_cap, s, t):
		"""
		Dinic max flow on a linked-list graph where arc i's reverse is i ^ 1.
		Mutates arc_cap in place (residual capacities); returns the flow value.
		"""
		num_nodes = head.shape[0]
		level = np.empty(num_nodes, dtype=np.int64)
		it = np.empty(num_nodes, dtype=np.int64)
		queue = np.empty(num_nodes, dtype=np.int64)
		stack_node = np.empty(num_nodes + 1, dtype=np.int64)
		stack_arc = np.empty(num_nodes + 1, dtype=np.int64)
		total = 0
		while True:
			# BFS to build the level graph
			for i in range(num_nodes):
				level[i] = -1
			qh, qt = 0, 0
			queue[qt] = s
			qt += 1
			level[s] = 0
			while qh < qt:
				u = queue[qh]
				qh += 1
				e = head[u]
				while e != -1:
					v = arc_to[e]
					if arc_cap[e] > 0 and level[v] == -1:
						level[v] = level[u] + 1
						queue[qt] = v
						qt += 1
					e = nxt[e]
			if level[t] == -1:
				break
			# blocking flow with the current-arc heuristic
			for i in range(num_nodes):
				it[i] = head[i]
			top = 0
			stack_node[0] = s
			while top >= 0:
				u = stack_node[top]
				if u == t:
					aug = np.int64(1) << 62
					for i in range(top):
						if arc_cap[stack_arc[i]] < aug:
							aug = arc_cap[stack_arc[i]]
					for i in range(top):
						e = stack_arc[i]
						arc_cap[e] -= aug
						arc_cap[e ^ 1] += aug
					total += aug
					# retreat to just before the first saturated arc
					back = 0
					for i in range(top):
						if arc_cap[stack_arc[i]] == 0:
							back = i
							break
					top = back
					continue
				advanced = False
				e = it[u]
				while e != -1:
					v = arc_to[e]
					if arc_cap[e] > 0 and level[v] == level[u] + 1:
						it[u] = e
						stack_arc[top] = e
						top += 1
						stack_node[top] = v
						advanced = True
						break
					e = nxt[e]
				if not advanced:
					it[u] = -1
					level[u] = -1  # dead end: never revisit in this phase
					top -= 1
					if top >= 0:
						it[stack_node[top]] = nxt[stack_arc[top]]
		return total


def _topo_dp(graph):
	"""
	One reverse-topological sweep computing, for every surviving node,
	(i) the log of the number of paths to a sink and (ii) the longest path to a sink.

	Returns (log10 of total source-to-sink paths, longest source-to-sink path length).
	"""
	if graph._dp_cache is not None:
		return graph._dp_cache

	if HAVE_NUMBA:
		order_rev = np.ascontiguousarray(graph.pi[::-1])
		log_paths, longest = _topo_dp_kernel(
			order_rev, graph._succ_indptr, graph._succ_indices, graph.is_sink)
		log_paths = log_paths.tolist()
		longest = longest.tolist()
	else:
		# plain python lists: much faster than per-node numpy calls in this loop
		indptr = graph._succ_indptr.tolist()
		indices = graph._succ_indices.tolist()
		is_sink = graph.is_sink.tolist()
		log_paths = [0.0] * graph.num_nodes_all
		longest = [0] * graph.num_nodes_all
		exp, log = math.exp, math.log
		for v in reversed(graph.pi.tolist()):
			if is_sink[v]:
				log_paths[v] = 0.0
				longest[v] = 0
				continue
			succs = indices[indptr[v]:indptr[v + 1]]
			xs = [log_paths[w] for w in succs]
			mx = max(xs)
			log_paths[v] = mx + log(sum(exp(x - mx) for x in xs))
			longest[v] = 1 + max(longest[w] for w in succs)

	src = graph.source_vals.tolist()
	xs = [log_paths[v] for v in src]
	mx = max(xs)
	log10_total = (mx + math.log(sum(math.exp(x - mx) for x in xs))) / math.log(10)
	max_length = max(longest[v] for v in src)

	graph._dp_cache = (log10_total, max_length)
	return graph._dp_cache


def calc_log10_num_paths(graph):
	"""
	Compute log10 of the total number of source-to-sink paths in a DeBruijn_DAG.
	(Log-space because the raw count overflows float64 for even moderate graphs.)

	Params
	------
	graph : DeBruijn_DAG
		A DeBruijn graph with edges pruned to make a DAG.

	Returns
	-------
	log10_num_paths : float
		log10 of the number of source-to-sink paths
	"""
	return _topo_dp(graph)[0]


def calc_max_length(graph):
	"""
	Compute the length (in edges) of the longest source-to-sink path.
	"""
	return _topo_dp(graph)[1]


def _min_paths_to_cover_edges_numba(edges_u, edges_v, source_vals, sink_vals, M):
	"""
	Exact minimum number of source-to-sink paths covering every edge (minimum
	flow with lower bound 1 per real edge), via two Dinic max-flow phases:
	route the lower-bound excess sigma->tau WITHOUT the return edge t*->s*,
	then open the return edge and continue; the flow on the return edge is the
	minimum flow value.
	"""
	E = len(edges_u)
	if E == 0:
		return 0

	s_star, t_star, sigma, tau = M, M + 1, M + 2, M + 3
	INF = E + 5  # any flow value is bounded by the total lower bound E

	indeg = np.bincount(edges_v, minlength=M)
	outdeg = np.bincount(edges_u, minlength=M)
	vs = np.nonzero(indeg)[0]  # nodes with lower-bound excess in
	us = np.nonzero(outdeg)[0]  # nodes with lower-bound excess out

	# directed edge list; the return edge is last with capacity 0 for phase 1
	frm = np.concatenate([
		edges_u,
		np.full(len(source_vals), s_star),
		sink_vals,
		np.full(len(vs), sigma),
		us,
		[t_star],
	]).astype(np.int64)
	to = np.concatenate([
		edges_v,
		source_vals,
		np.full(len(sink_vals), t_star),
		vs,
		np.full(len(us), tau),
		[s_star],
	]).astype(np.int64)
	cap = np.concatenate([
		np.full(E, INF),
		np.full(len(source_vals), INF),
		np.full(len(sink_vals), INF),
		indeg[vs],
		outdeg[us],
		[0],
	]).astype(np.int64)

	T = len(frm)
	arc_to = np.empty(2 * T, dtype=np.int64)
	arc_to[0::2] = to
	arc_to[1::2] = frm
	arc_cap = np.empty(2 * T, dtype=np.int64)
	arc_cap[0::2] = cap
	arc_cap[1::2] = 0
	arc_from = np.empty(2 * T, dtype=np.int64)
	arc_from[0::2] = frm
	arc_from[1::2] = to
	head = np.full(M + 4, -1, dtype=np.int64)
	nxt = np.empty(2 * T, dtype=np.int64)
	_build_adjacency(arc_from, head, nxt)
	ts_arc = 2 * (T - 1)

	F1 = _dinic(head, nxt, arc_to, arc_cap, sigma, tau)
	arc_cap[ts_arc] = INF  # open the return edge and continue on the residual
	F2 = _dinic(head, nxt, arc_to, arc_cap, sigma, tau)

	# all lower-bound excess must be routable, otherwise no feasible cover exists
	assert F1 + F2 == E, \
		"min-flow infeasible: some edge is not on a source-to-sink path"

	p_min = int(INF - arc_cap[ts_arc])

	# sanity: the degree-imbalance formula upper-bounds the true minimum
	assert 1 <= p_min <= int(np.maximum(outdeg - indeg, 0).sum())
	return p_min


def _min_paths_to_cover_edges_nx(edges_u, edges_v, source_vals, sink_vals, M):
	"""
	Same computation with networkx (pure python; used as a fallback when numba
	is unavailable, and as a cross-check in the self tests). Reduction: find a
	feasible flow with the excess construction, then minimize it by pushing max
	flow from t* to s* in the residual network (backward residual capacity of a
	real edge is flow-1, respecting the lower bound).
	"""
	E = len(edges_u)
	if E == 0:
		return 0

	s_star, t_star, sigma, tau = 's*', 't*', 'sigma', 'tau'
	INF = E + 5

	indeg = np.bincount(edges_v, minlength=M)
	outdeg = np.bincount(edges_u, minlength=M)

	G = nx.DiGraph()
	for u, v in zip(edges_u.tolist(), edges_v.tolist()):
		G.add_edge(u, v, capacity=INF)  # transformed capacity: f - 1 >= 0
	for s in source_vals.tolist():
		G.add_edge(s_star, s, capacity=INF)
	for s in sink_vals.tolist():
		G.add_edge(s, t_star, capacity=INF)
	for v in np.nonzero(indeg)[0].tolist():
		G.add_edge(sigma, v, capacity=int(indeg[v]))
	for u in np.nonzero(outdeg)[0].tolist():
		G.add_edge(u, tau, capacity=int(outdeg[u]))
	G.add_edge(t_star, s_star, capacity=INF)

	flow_value, flow_dict = nx.maximum_flow(G, sigma, tau)
	assert flow_value == E, \
		"min-flow infeasible: some edge is not on a source-to-sink path"
	F = flow_dict[t_star][s_star]  # a feasible (not yet minimal) cover size

	# residual network of the feasible flow, in the ORIGINAL variables
	# (real edge flow = transformed flow + 1); no return edge
	R = nx.DiGraph()
	for u, v in zip(edges_u.tolist(), edges_v.tolist()):
		f = flow_dict[u][v] + 1
		R.add_edge(u, v, capacity=INF)  # forward: capacity infinite
		R.add_edge(v, u, capacity=f - 1)  # backward: can reduce down to the lower bound
	for s in source_vals.tolist():
		R.add_edge(s_star, s, capacity=INF)
		R.add_edge(s, s_star, capacity=flow_dict[s_star][s])
	for s in sink_vals.tolist():
		R.add_edge(s, t_star, capacity=INF)
		R.add_edge(t_star, s, capacity=flow_dict[s][t_star])

	reduction, _ = nx.maximum_flow(R, t_star, s_star)
	p_min = F - reduction

	# sanity: the degree-imbalance formula upper-bounds the true minimum
	assert 1 <= p_min <= int(np.maximum(outdeg - indeg, 0).sum())
	return p_min


def _min_paths_to_cover_edges(edges_u, edges_v, source_vals, sink_vals, M):
	if HAVE_NUMBA:
		return _min_paths_to_cover_edges_numba(edges_u, edges_v, source_vals, sink_vals, M)
	return _min_paths_to_cover_edges_nx(edges_u, edges_v, source_vals, sink_vals, M)


def calc_min_cover(graph):
	"""
	Compute the minimum number of source-to-sink paths required to cover every edge
	in a De Bruijn DAG object (exact, via minimum flow with lower bounds).

	Params
	------
	graph : DeBruijn_DAG
		A DeBruijn graph with edges pruned to make a DAG.

	Returns
	-------
	min_cover : int
		minimum number of source-to-sink paths that cover every edge
	"""
	return _min_paths_to_cover_edges(
		graph.edges_u, graph.edges_v,
		graph.source_vals, graph.sink_vals,
		graph.num_nodes_all)


def calc_min_length_cover(graph):
	"""
	Compute the minimum length L such that every edge of the DAG lies on a
	source-to-sink path of length <= L. A model that learns all paths of length
	up to L therefore covers every edge.

	For an edge u->v, the shortest source-to-sink path through it has length
	d_src(u) + 1 + d_sink(v) where d_src / d_sink are (BFS) shortest distances
	from any source / to any sink. L is the max of this over all edges.

	Params
	------
	graph : DeBruijn_DAG
		A DeBruijn graph with edges pruned to make a DAG.

	Returns
	-------
	min_length_cover : int
		minimum path length needed to cover every edge
	"""
	M = graph.num_nodes_all
	s_star, t_star = M, M + 1
	rows = np.concatenate([graph.edges_u,
						   np.full(len(graph.source_vals), s_star),
						   graph.sink_vals])
	cols = np.concatenate([graph.edges_v,
						   graph.source_vals,
						   np.full(len(graph.sink_vals), t_star)])
	G = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(M + 2, M + 2))

	d_from_src = dijkstra(G, directed=True, indices=s_star, unweighted=True)
	d_to_sink = dijkstra(G.T.tocsr(), directed=True, indices=t_star, unweighted=True)

	# subtract 1 for the auxiliary s*->source edge, add 1 for the edge itself,
	# subtract 1 for the auxiliary sink->t* edge
	lengths = d_from_src[graph.edges_u] + d_to_sink[graph.edges_v] - 1.0
	return int(lengths.max())


# Theoretical bounds (for the digit-sum permutation of Theorems 1 and 2)

def theory1_log10_ratio(V, n, m=1):
	"""
	Theorem 1: P_min <= V^n (V+1)/2 and P_tot >= sum_t C(V-2m, t)^n, so
	log10(P_min/P_tot) <= log10 of the ratio of the bounds.
	"""
	p_min_ub = V ** n * (V + 1) // 2
	p_tot_lb = sum(comb(V - 2 * m, t) ** n for t in range(V - 2 * m + 1))
	return math.log10(p_min_ub) - math.log10(p_tot_lb)


def theory2_ratio(V, n, m=1):
	"""
	Theorem 2: L_cover <= 2n+1 and L_max >= n(V-2m-1), so
	L_cover/L_max <= (2n+1) / (n(V-2m-1)).
	"""
	return (2 * n + 1) / (n * (V - 2 * m - 1))


# Self tests (fast; run before the experiments)

def _brute_force_reference(graph):
	"""
	Enumerate every source-to-sink path by DFS (small graphs only). Each edge is
	identified by its position in the successor CSR. Returns None if no paths.
	"""
	indptr = graph._succ_indptr
	indices = graph._succ_indices
	paths = []

	def dfs(v, edge_stack):
		if graph.is_sink[v]:
			paths.append(tuple(edge_stack))
			return
		for j in range(indptr[v], indptr[v + 1]):
			edge_stack.append(j)
			dfs(indices[j], edge_stack)
			edge_stack.pop()

	for s in graph.source_vals:
		dfs(int(s), [])
	if not paths:
		return None

	# min length of a covering path for each edge
	edge_min_len = {}
	for p in paths:
		for j in p:
			edge_min_len[j] = min(edge_min_len.get(j, len(p)), len(p))
	assert len(edge_min_len) == graph.num_edges

	return {
		'count': len(paths),
		'max_length': max(len(p) for p in paths),
		'min_length_cover': max(edge_min_len.values()),
		'paths': paths,
	}


def _brute_force_min_cover(paths, num_edges):
	"""Exact min path cover by subset enumeration (only for few paths)."""
	masks = []
	for p in paths:
		mask = 0
		for j in p:
			mask |= 1 << j
		masks.append(mask)
	full = (1 << num_edges) - 1
	best = len(paths)
	for sub in range(1, 1 << len(paths)):
		acc = 0
		for i, mk in enumerate(masks):
			if sub >> i & 1:
				acc |= mk
		if acc == full:
			best = min(best, bin(sub).count('1'))
	return best


def run_self_tests():
	# handcrafted case where the degree-imbalance formula (=3) overestimates the
	# true min cover (=2): s1->a, s2->a, a->b, b->t1, b->t2
	eu = np.array([0, 1, 2, 3, 3])
	ev = np.array([2, 2, 3, 4, 5])
	for backend in ([_min_paths_to_cover_edges_numba] if HAVE_NUMBA else []) \
			+ [_min_paths_to_cover_edges_nx]:
		p_min = backend(eu, ev, np.array([0, 1]), np.array([4, 5]), 6)
		assert p_min == 2, f"{backend.__name__} failed on the handcrafted case: got {p_min}"

	# random small De Bruijn DAGs against brute force
	for seed in range(6):
		g = DeBruijn_DAG(5, 2, 1, 1, rng=np.random.default_rng(seed))
		ref = _brute_force_reference(g)
		if ref is None:
			continue
		log10_tot = calc_log10_num_paths(g)
		assert abs(10 ** log10_tot - ref['count']) < 1e-6 * ref['count'] + 1e-9
		assert calc_max_length(g) == ref['max_length']
		assert calc_min_length_cover(g) == ref['min_length_cover']
		if len(ref['paths']) <= 14:
			assert calc_min_cover(g) == _brute_force_min_cover(ref['paths'], g.num_edges)

	# medium instances: numba Dinic against networkx (this size range is where
	# scipy 1.7's maximum_flow returned wrong values)
	if HAVE_NUMBA:
		for t in range(4):
			g = DeBruijn_DAG(10, 5, 5, 5, rng=np.random.default_rng([0, 2, 5, t]))
			a = _min_paths_to_cover_edges_numba(
				g.edges_u, g.edges_v, g.source_vals, g.sink_vals, g.num_nodes_all)
			b = _min_paths_to_cover_edges_nx(
				g.edges_u, g.edges_v, g.source_vals, g.sink_vals, g.num_nodes_all)
			assert a == b, f"Dinic ({a}) and networkx ({b}) disagree"

	# the theorem bounds hold for the digit-sum ordering at S=5, N=10
	for V, n in [(10, 2), (8, 3), (10, 4)]:
		m = tightest_m(V, n, 5, 10)
		assert m is not None
		verify_digit_sum_bounds(V, n, 5, 10, m)
	print("self-tests passed")


# Experiments

def tightest_m(V, n, S, N):
	"""
	The value of m that makes both theorem bounds tightest, subject to the
	theorem hypotheses S, N < C(n+m, m) and 0 < m < (V-1)/2.

	Both bounds tighten as m shrinks (the Theorem 1 P_tot lower bound
	sum_t C(V-2m, t)^n and the Theorem 2 L_max lower bound n(V-2m-1) both grow
	as V-2m grows), so the optimal m is the SMALLEST integer m >= 1 with
	C(n+m, m) > max(S, N). Returns None if that m is not < (V-1)/2, i.e. no
	feasible m exists for this (V, n) (a larger m only loosens the bound and
	needs an even larger V).
	"""
	target = max(S, N)
	m = 1
	while comb(n + m, m) <= target:
		m += 1
	return m if m < (V - 1) / 2 else None


def verify_digit_sum_bounds(V, n, S, N, m):
	"""
	Sanity check that the digit-sum ordering actually satisfies the Theorem 1/2
	bounds for these (V, n, S, N, m); used in the self tests. Raises on failure.
	"""
	pi = digit_sum_ordering(V, n)
	graph = DeBruijn_DAG(V, n, S, N, pi=pi)
	log10_tot = calc_log10_num_paths(graph)
	p_min = calc_min_cover(graph)
	l_max = calc_max_length(graph)
	l_cover = calc_min_length_cover(graph)
	p_tot_lb = sum(comb(V - 2 * m, t) ** n for t in range(V - 2 * m + 1))
	assert log10_tot >= math.log10(p_tot_lb) - 1e-9
	assert p_min <= V ** n * (V + 1) // 2
	assert l_max >= n * (V - 2 * m - 1)
	assert l_cover <= 2 * n + 1


def sample_random_metrics(V, n, S, N, rng, max_retries=50):
	"""
	Build D_pi for one random permutation pi and return the raw ratios
	(P_min/P_tot, L_cover/L_max). Retries with fresh permutations if pruning
	leaves no source-to-sink path; returns None if none is found. No theorem
	bounds are asserted here -- they hold only for the digit-sum ordering.
	"""
	for _ in range(max_retries):
		graph = DeBruijn_DAG(V, n, S, N, rng=rng)
		if graph.num_nodes > 0 and graph.num_edges > 0:
			break
	else:
		return None

	log10_tot = calc_log10_num_paths(graph)
	p_min = calc_min_cover(graph)
	l_max = calc_max_length(graph)
	l_cover = calc_min_length_cover(graph)
	return {
		'path_ratio': 10.0 ** (math.log10(p_min) - log10_tot),
		'len_ratio': l_cover / l_max,
	}


def digit_sum_metrics(V, n, S, N):
	"""
	Raw ratios (P_min/P_tot, L_cover/L_max) for the single deterministic
	digit-sum ordering used in the theorem proofs -- the ordering for which the
	dashed theory bounds are proven. Plotted alongside the random-permutation
	average to show what the optimal ordering actually achieves.
	"""
	graph = DeBruijn_DAG(V, n, S, N, pi=digit_sum_ordering(V, n))
	log10_tot = calc_log10_num_paths(graph)
	p_min = calc_min_cover(graph)
	l_max = calc_max_length(graph)
	l_cover = calc_min_length_cover(graph)
	return {
		'path_ratio': 10.0 ** (math.log10(p_min) - log10_tot),
		'len_ratio': l_cover / l_max,
	}


def run_experiments(n_to_vrange=((2, (10, 40)), (3, (8, 30)), (4, (6, 24))),
					n_trials=10, S=5, N=10, seed=0):
	"""
	One curve per fixed n-gram length n; within each, sweep integer V over its
	given inclusive range. Fixing n gives monotone curves (unlike fixing
	log2(V)/n, which makes n jump in integer steps and the curves sawtooth).
	Each point is the mean over n_trials random permutations of the raw ratios,
	with the std for the error band. S and N are constant; m is the tightest
	theorem reference bound for that point (constant along a fixed-n curve), and
	points with no feasible m are skipped. The V upper bounds keep every eval
	fast -- large V at small n is the slow case (the O(V^n * V) edge scan).
	"""
	results = {}
	for n, (v_lo, v_hi) in dict(n_to_vrange).items():
		rows = []
		for V in range(v_lo, v_hi + 1):
			m = tightest_m(V, n, S, N)
			if m is None:
				continue
			path_ratios, len_ratios = [], []
			for t in range(n_trials):
				rng = np.random.default_rng([seed, V, n, t])
				out = sample_random_metrics(V, n, S, N, rng)
				if out is None:
					continue
				path_ratios.append(out['path_ratio'])
				len_ratios.append(out['len_ratio'])
			if not path_ratios:
				continue
			ds = digit_sum_metrics(V, n, S, N)
			# store both arithmetic (mean +/- std -> lo/hi, for a linear y-axis) and
			# geometric (log-space, for a log y-axis) summaries of the path ratio, so
			# either axis scale can be plotted without recomputing.
			pr_mean, pr_std = float(np.mean(path_ratios)), float(np.std(path_ratios))
			logs = np.log10(np.clip(path_ratios, 1e-300, None))
			lm, lstd = float(np.mean(logs)), float(np.std(logs))
			lr_mean, lr_std = float(np.mean(len_ratios)), float(np.std(len_ratios))
			row = {
				'V': V, 'n': n, 'm': m, 'Vn': V ** n,
				'path_ratio_mean': pr_mean,
				'path_ratio_lo': pr_mean - pr_std,
				'path_ratio_hi': pr_mean + pr_std,
				'path_geo_mean': 10.0 ** lm,
				'path_geo_lo': 10.0 ** (lm - lstd),
				'path_geo_hi': 10.0 ** (lm + lstd),
				'len_ratio_mean': lr_mean,
				'len_ratio_lo': lr_mean - lr_std,
				'len_ratio_hi': lr_mean + lr_std,
				'ds_path_ratio': ds['path_ratio'],
				'ds_len_ratio': ds['len_ratio'],
				'theory1_ratio': 10.0 ** theory1_log10_ratio(V, n, m),
				'theory2_ratio': theory2_ratio(V, n, m),
				'num_trials': len(path_ratios),
			}
			rows.append(row)
			print(f"n={n} V={V} m={m} V^n={row['Vn']}: "
				  f"Pmin/Ptot rand={pr_mean:.3f} ds={row['ds_path_ratio']:.2e}, "
				  f"Lcover/Lmax rand={lr_mean:.3f} ds={row['ds_len_ratio']:.3f} "
				  f"({row['num_trials']} trials)")
		results[n] = rows
	return results


# Plotting

# first three slots of the validated categorical palette (blue, green, magenta);
# distinct markers double as a secondary (colorblind/print-safe) encoding
CURVE_COLORS = ['#2a78d6', '#008300', '#e87ba4', '#eda100']
CURVE_MARKERS = ['o', 's', '^', 'D']

LABEL_FS = 14
TICK_FS = 12
LEGEND_FS = 8


def _style_axis(ax):
	ax.tick_params(labelsize=TICK_FS)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)


# line styles: random-permutation mean (solid), digit-sum ordering (dashed),
# theory bound (dotted). Color encodes n; style encodes which quantity.
RANDOM_LS, DIGITSUM_LS, THEORY_LS = '-', '--', ':'


def _legend(ax, keys):
	# color key (one entry per n) then style key (random / digit-sum / theory),
	# laid out as two rows above the axes
	blank = lambda: Line2D([], [], alpha=0, label='')
	color_handles = [Line2D([], [], color=CURVE_COLORS[i], marker=CURVE_MARKERS[i],
							ms=3.5, lw=1.4, label=f'$n={k:g}$')
					 for i, k in enumerate(keys)]
	while len(color_handles) < 3:
		color_handles.append(blank())
	style_handles = [
		Line2D([], [], color='0.35', ls=RANDOM_LS, lw=1.4, label='random'),
		Line2D([], [], color='0.35', ls=DIGITSUM_LS, lw=1.3, label='digit-sum'),
		Line2D([], [], color='0.35', ls=THEORY_LS, lw=1.2, label='theory'),
	]
	# two vertical columns (column-major fill): left column = the n color key,
	# right column = the random / digit-sum / theory style key
	handles = color_handles + style_handles
	leg = ax.legend(handles=handles, fontsize=LEGEND_FS, frameon=False, ncol=2,
					loc='upper right', handlelength=1.4, columnspacing=0.8,
					labelspacing=0.2, handletextpad=0.4)
	leg._legend_box.align = 'left'


def _plot_ratio(results, keys, mean_key, lo_key, hi_key, ds_key, theory_key,
				ylabel, outpath, log_y=False, show_legend=True):
	"""
	Plot vs V^n (log x) of three quantities per n: the random-permutation mean
	(solid line + marker, with a shaded band from lo_key to hi_key), the digit-sum
	ordering value (dashed), and the digit-sum theory bound (dotted). Both ratios
	are in [0, 1]. With log_y=False the y-axis is linear and capped just above 1
	(the theory bound is vacuous, > 1, for small V^n, so its dotted line enters
	from the top where it becomes informative). With log_y=True the y-axis is
	logarithmic (use geometric summary keys so the band stays positive).
	"""
	fig, ax = plt.subplots(figsize=(3, 2.5))
	for i, k in enumerate(keys):
		rows = results[k]
		x = np.array([r['Vn'] for r in rows], dtype=float)
		mu = np.array([r[mean_key] for r in rows])
		lo = np.array([r[lo_key] for r in rows])
		hi = np.array([r[hi_key] for r in rows])
		ds = np.array([r[ds_key] for r in rows])
		th = np.array([r[theory_key] for r in rows])
		col = CURVE_COLORS[i]
		ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
		ax.plot(x, mu, color=col, ls=RANDOM_LS, marker=CURVE_MARKERS[i], ms=3.5, lw=1.4)
		ax.plot(x, ds, color=col, ls=DIGITSUM_LS, lw=1.3)
		ax.plot(x, th, color=col, ls=THEORY_LS, lw=1.2)
	ax.set_xscale('log')
	if log_y:
		ax.set_yscale('log')
	else:
		ax.set_ylim(0, 1.08)
	ax.set_xlabel('$V^n$', fontsize=LABEL_FS)
	ax.set_ylabel(ylabel, fontsize=LABEL_FS)
	_style_axis(ax)
	if show_legend:
		_legend(ax, keys)
	fig.savefig(outpath + '.pdf', bbox_inches='tight')
	fig.savefig(outpath + '.png', bbox_inches='tight', dpi=300)
	plt.close(fig)


def make_plots(results, outdir):
	keys = sorted(results.keys())
	figdir = os.path.join(outdir, 'figures')
	os.makedirs(figdir, exist_ok=True)
	# Plot 1: P_min / P_tot vs V^n (linear y), with the Theorem 1 bound
	_plot_ratio(results, keys, 'path_ratio_mean', 'path_ratio_lo', 'path_ratio_hi',
				'ds_path_ratio', 'theory1_ratio', r'$P_{\min}/P_{\mathrm{tot}}$',
				os.path.join(figdir, 'theorem1_path_ratio'), log_y=False)
	# Plot 2: L_cover / L_max vs V^n (linear y), with the Theorem 2 bound (no legend)
	_plot_ratio(results, keys, 'len_ratio_mean', 'len_ratio_lo', 'len_ratio_hi',
				'ds_len_ratio', 'theory2_ratio', r'$L_{\mathrm{cover}}/L_{\mathrm{max}}$',
				os.path.join(figdir, 'theorem2_length_ratio'), log_y=False,
				show_legend=False)


def main():
	outdir = os.path.dirname(os.path.abspath(__file__))
	cachedir = os.path.join(outdir, 'cache')
	os.makedirs(cachedir, exist_ok=True)
	cache = os.path.join(cachedir, 'theorem_data_random.json')

	if os.path.exists(cache) and '--force' not in sys.argv:
		with open(cache) as f:
			results = {int(n): v for n, v in json.load(f).items()}
		print(f"loaded cached results from {cache} (use --force to recompute)")
	else:
		run_self_tests()
		results = run_experiments(n_to_vrange=((2, (10, 40)), (3, (8, 30)), (4, (6, 24))),
								  n_trials=10, S=5, N=10)
		with open(cache, 'w') as f:
			json.dump({str(c): v for c, v in results.items()}, f, indent=1)

	make_plots(results, outdir)
	print("saved theorem1_path_ratio.{pdf,png} and theorem2_length_ratio.{pdf,png}")


if __name__ == '__main__':
	main()
