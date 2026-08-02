"""
Create data samples from a De Bruijn DAG of size (V,n) with
random topological sort, or digit-sum ordering.
"""

import numpy as np

# load the functions to create a de bruijn DAG from 
_THEORY_DIR = os.path.join(
	os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'theory')
if _THEORY_DIR not in sys.path:
	sys.path.insert(0, _THEORY_DIR)

from theorem1_theorem2 import DeBruijn_DAG, digit_sum_ordering, calc_max_length



def build_dag(V:int, n:int, S:int, N:int, ordering:str='random', seed:int=42, max_retries:int=50):
	"""
	Build D_pi = F[B(V,n); pi, N, S].

	Params
	------
	V : int
		Vocab size
	n : int
		n-gram memory length
	N : int
		Number of answer nodes
	S : int
		Number of start nodes
	ordering : {'random', 'digit_sum'}
		'random' draws pi uniformly. 'digit_sum' is the ordering of the
		Theorem 1/2 proofs.
	seed : int
		Seed for the random ordering; ignored when ordering='digit_sum'.
	max_retries : int
		Maximum number of tries to find a DAG with at least one path from start to answer.

	Returns
	-------
	dag : DeBruijn_DAG
		A pruned DAG with at least one start-to-answer path.
	"""
	if ordering == 'digit_sum':
		return DeBruijn_DAG(V, n, S, N, pi=digit_sum_ordering(V, n))
	if ordering != 'random':
		raise ValueError(f"unknown ordering {ordering!r}")
	for k in range(max_retries):
		rng = np.random.default_rng([seed, k])
		dag = DeBruijn_DAG(V, n, S, N, rng=rng)
		if dag.num_edges > 0 and len(dag.source_vals) > 0:
			return dag
	raise RuntimeError(f"Could not find a non-empty DAG for V={V} n={n} S={S} N={N} seed={seed}")


class ReasoningGenerator:
	"""
	Samples from a uniform distribution over start-to-answer paths of a De Bruijn DAG.

	A sample is the character string w_1...w_{n+L} generating the node path
	u_1 -> ... -> u_{L+1} (u_j = w_j...w_{j+n-1}); the first n characters name
	the start node and each subsequent character takes one out-edge. The
	terminal answer node emits the end-of-sequence symbol, which is treated as
	branch index V throughout.
	"""
	def __init__(self, dag):
		self.dag = dag
		self.V, self.n = dag.V, dag.n
		M = dag.num_nodes_all
		Vp = self.V ** (self.n - 1)
		self.rng = self.dag.rng

		# pre-compute quantities to efficiently sample paths uniformly
		# reverse-topological DP: log number of paths from each node to an answer
		log_paths = np.full(M, -np.inf)
		log_paths[dag.sink_vals] = 0.0
		indptr, indices = dag._succ_indptr, dag._succ_indices
		is_sink = dag.is_sink
		for v in dag.pi[::-1]:
			if is_sink[v]:
				continue
			succ = indices[indptr[v]:indptr[v + 1]]
			if len(succ) == 0:
				continue
			xs = log_paths[succ]
			mx = xs.max()
			log_paths[v] = mx + math.log(np.exp(xs - mx).sum())
		self.log_paths = log_paths

		# branching distribution and its support
		eu, ev = dag.edges_u, dag.edges_v
		eb = ev // Vp                              # character appended by the edge
		self.branch_p = np.zeros((M, self.V + 1))
		self.branch_p[eu, eb] = np.exp(log_paths[ev] - log_paths[eu])
		self.branch_p[dag.sink_vals, self.V] = 1.0
		self.branch_legal = np.zeros((M, self.V + 1), dtype=bool)
		self.branch_legal[eu, eb] = True
		self.branch_legal[dag.sink_vals, self.V] = True
		self._branch_cum = np.cumsum(self.branch_p, axis=1)

		# start distribution (proportional to the number of paths below each start)
		src = dag.source_vals
		xs = log_paths[src]
		mx = xs.max()
		self.log_total_paths = mx + math.log(np.exp(xs - mx).sum())
		self.start_vals = src
		self.start_p = np.exp(xs - self.log_total_paths)

		self.max_edges = calc_max_length(dag)       # L_max, longest path
		self.max_chars = self.n + self.max_edges    # characters in the longest sample

		# forward DP: probability that a path visits each node, and the induced
		# probability of traversing each edge (used for coverage weighting)
		occ = np.zeros(M)
		occ[src] = self.start_p
		for u in dag.pi:
			if occ[u] == 0.0 or is_sink[u]:
				continue
			succ = indices[indptr[u]:indptr[u + 1]]
			occ[succ] += occ[u] * np.exp(log_paths[succ] - log_paths[u])
		self.occupancy = occ
		self.edge_p = occ[eu] * np.exp(log_paths[ev] - log_paths[eu])

		# exact expected sequence length (in characters), for per-token quantities
		self.exp_num_edges = float(self.edge_p.sum())
		self.exp_num_chars = self.n + self.exp_num_edges


	def sample(self, num, rng, max_edges=None):
		"""
		Sample `num` paths exactly uniformly from the start-to-answer paths.

		Returns
		-------
		chars : (num, T) int16
			Character sequences, padded with -1. T = self.max_chars.
		lengths : (num,) int32
			Number of characters in each sample (n + path length).
		"""
		V, n = self.V, self.n
		Vp = V ** (n - 1)
		T = self.max_chars if max_edges is None else n + max_edges
		chars = np.full((num, T), -1, dtype=np.int16)
		lengths = np.full(num, n, dtype=np.int32)

		node = rng.choice(self.start_vals, size=num, p=self.start_p)
		# the first n characters spell out the start node (a_1 = least significant)
		x = node.copy()
		for k in range(n):
			chars[:, k] = x % V
			x //= V

		# T+1 iterations: the last one can only draw EOS (a path of L_max edges
		# fills every character slot and still has to be terminated)
		active = np.arange(num)
		for t in range(n, T + 1):
			cum = self._branch_cum[node[active]]
			u = rng.random(len(active))
			b = (cum < u[:, None]).sum(axis=1)
			b = np.minimum(b, V)                   # guard against fp round-off
			live = b != V
			idx = active[live]
			if len(idx) == 0:
				active = idx
				break
			assert t < T, "sampling did not terminate within L_max"
			bl = b[live]
			chars[idx, t] = bl
			node[idx] = node[idx] // V + bl * Vp
			lengths[idx] = t + 1
			active = idx
		assert len(active) == 0, "sampling did not terminate within L_max"
		return chars, lengths








