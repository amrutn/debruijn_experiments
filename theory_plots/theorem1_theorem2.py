"""
Averaged over different samples of permutations pi:
1. Plot mean of min paths to cover/total paths while increasing $V^n$ s.t. $V/n=k$ is fixed.
	Plot curves for multiple values of $k$. Plot theoretical bound.
2. Plot ratio of min_length_to_cover/max_length while increasing $V^n$ s.t. $V/n=k$ is fixed. 
	Plot curves for multiple values of $k$. Plot theoretical bound. 
"""

import os
import numpy as np
import matplotlib.pyplot as plt


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
	4. get_source nodes : returns a list of the source nodes
	5. get_sink_nodes : returns a list of the sink nodes
	6. get_incoming_nodes : returns a list of the incoming nodes to the input
	7. get_outgoing_nodes : returns a list of the outgoing nodes to the input
	8. get_degree : returns a tuple of the in-degree and the out-degree of the input node
	"""
	def __init__(self, V, n, S, N, pi=None, pi_inverse=None, rng=None):
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
			Sorts numerical values associated with each node from 0 to V^n.
		rng : random number generator (numpy class)

		"""

		self.V = V
		self.n = n
		self.S = S
		self.N = N
		self.pi = pi

		# setting the rng
		if rng is None:
			self.rng = np.random.default_rng(seed=42)
		else:
			self.rng = rng

		# generating a random topological sort if one is not provided
		if self.pi is None:
			DeBruijn_DAG.set_random_ordering(V,n, self.rng)

	def set_random_ordering(self,V,n,rng):
		"""
		Sets the topological sort to a random sample. Resets the source and sink nodes

		Params
		------
		V : int
			Vocabulary size.
		n : int
			n-gram length
		rng : random number generator (numpy class)
		"""
		self.pi = rng.shuffle(range(0, self.V**self.n))

	def node_to_val(self, node):
		"""
		Converts a node representation (list of ints) into a numerical value.
		"""
		assert len(node) == n
		assert all([a < V and a >= 0 for a in node])

		return sum([a**k for k, a in enumerate(node)])

	def val_to_node(self, val):
		"""
		Converts a numerical value back into a node representation
		"""
		assert val >= 0
		assert val < self.V**self.n

		node = []
		for k in range(n):
			node.append(val % V)
			val = val // V
		return node

	def get_source_nodes(self):
		"""
		Returns a list of the S source nodes.
		"""
		source_node_vals = self.pi[:S]
		return [val_to_node(a) for a in source_node_vals]

	def get_sink_nodes(self):
		"""
		Returns a list of the N sink nodes.
		"""
		sink_node_vals = self.pi[-N:]
		return [val_to_node(a) for a in sink_node_vals]

	def get_incoming_nodes(self, node):
		"""
		Returns a list of the incoming nodes from a given node. 

		Params
		------
		node : list
			A list of n integers between 0 and V-1
		Returns
		-------
		incoming : list
			A list of nodes that are incoming to the input
		"""
		assert len(node) == n
		assert all([a < V and a >= 0 for a in node])

		val = node_to_val(node) # numerical value of the node
		sort_val = np.where(self.pi == val)[0][0] # node position in topological sort

		# if node is a source return empty
		if sort_val < S:
			return []

		incoming = []
		for c in range(self.V):
			new_node = node.copy().insert(0, c).pop()
			new_val = node_to_val(new_node)
			new_sort_val = np.where(self.pi == new_val)[0][0]

			# if the new node is a sink, do not include it
			if new_sort_val >= V**n-N:
				continue

			# compare the positions of both nodes in the sort
			if sort_val > new_sort_val:
				incoming.append(new_node)

		return incoming

	def get_outgoing_nodes(self, node):
		"""
		Returns a list of the outgoing nodes from a given node. 

		Params
		------
		node : list
			A list of n integers between 0 and V-1
		Returns
		-------
		outgoing : list
			A list of nodes that are outgoing to the input
		"""
		assert len(node) == n
		assert all([a < V and a >= 0 for a in node])

		val = node_to_val(node) # numerical value of the node
		sort_val = np.where(self.pi == val)[0][0] # node position in topological sort

		# if node is a sink return empty
		if sort_val >= V**n-N:
			return []

		outgoing = []
		for c in range(self.V):
			new_node = node.copy().append(c).pop(0)
			new_val = node_to_val(new_node)
			new_sort_val = np.where(self.pi == new_val)[0][0]

			# if the new node is a source, it cannot be included
			if new_sort_val < S:
				continue

			# compare the positions of both nodes in the sort
			if sort_val < new_sort_val:
				outgoing.append(new_node)
				
		return outgoing


	def get_degree(self, node):
		"""
		Returns a tuple of the (in-degree, out-degree) of a node.

		Params
		------
		node : list
			A list of n integers between 0 and V-1
		Returns
		-------
		degrees : tuple
			Tuple of (in-degree, out-degree)
		"""
		return (len(get_incoming_nodes(node)), len(get_outgoing_nodes(node)))


# Define helper functions to:
# 1. calc_num_paths : calculate the total number of source-to-sink paths.
# 2. calc_min_cover : calculate the minimum number of paths needed to cover every edge.
# 3. calc_min_length_cover : calculate the minimum length of paths needed to cover every edge.

def calc_num_paths(graph):
	"""
	Compute the total number of source-to-sink paths in a DeBruijn_DAG object.

	Params
	------
	graph : DeBruijn DAG
		A DeBruijn graph with edges pruned to make a DAG.

	Returns
	-------
	num_paths : int
		Number of source-to-sink paths
	"""

	




