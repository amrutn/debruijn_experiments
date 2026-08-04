"""
Training and evaluation for the synthetic De Bruijn experiments.

There is no "correct" continuation to predict -- the model has to learn which
subset of the vocabulary is legal at each state. The metric is therefore a
support metric rather than accuracy:

illegal_mass      total probability the model puts on tokens that are not edges
                  of D_pi. This is the direct empirical analogue of "has it
                  learned the edge set", is well defined at every branch point
                  regardless of out-degree, and is reported split by whether the
                  state was visited during training.

path_coverage     What proportion of correct paths does the model assign non-trivial
				  probability to. A path counts as "covered" if the model assigns
				  P_model(path) >= PATH_RATIO*P_true(path)

Two experiments are provided:

experiment 1 (`run_experiment_1`)
    Paths are drawn from `ReasoningGenerator.sample`. Optionally the token
    identities are scrambled by a *position dependent* deterministic
    permutation: characters in the k-th block of `bin_width` positions are
    relabelled by a fixed permutation sigma_k. The scrambling is a bijection at
    every position, so the legal-edge structure is preserved (up to relabelling)
    and the metrics are computed after mapping the model's probabilities back to
    the original token identities.

experiment 2 (`run_experiment_2`)
    Paths are drawn from `ReasoningGenerator.sample_length_limited`, i.e. only
    source-to-sink paths of at most `L` edges.

A model is trained for a single epoch over the sampled data and cached under
`cache/` keyed by the full configuration, then evaluated on a fresh test set.
"""

import os
import json
import math
import hashlib
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn.functional as F

from model import TransformerLM
from generate_data import build_dag, ReasoningGenerator


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

# The default path ratio when computing the path_coverage metric
DEFAULT_PATH_RATIO = 0.1


@dataclass
class ModelConfig:
	"""
	Architecture of the decoder-only Transformer.

	`build_model` forwards these to `model.TransformerLM`; see that class's
	docstring for the precise meaning of each hyper-parameter. The remaining two
	arguments TransformerLM needs -- `vocab_size` and `max_seq_len` -- are not
	set here: they are derived from the data De Bruijn graph (vocab = V + 1,
	max_seq_len = T + 1).

	Fields
	------
	d_model : int
		Embedding dimension.
	num_heads : int
		Number of attention heads. 
	num_layers : int
		Number of stacked Transformer blocks.
	d_ff : int | None
		Feed-forward hidden dimension; None lets SwiGLU pick ~8/3 * d_model.
	theta : float
		Base wavelength for the RoPE positional embeddings.
	use_rope : bool
		Whether to apply RoPE inside attention.
	"""
	d_model: int = 64
	num_heads: int = 4
	num_layers: int = 2
	d_ff: int | None = None          # None -> SwiGLU picks ~8/3 * d_model
	theta: float = 10000.0
	use_rope: bool = True


@dataclass
class TrainConfig:
	"""
	Optimisation settings for the single-epoch training loop (`train_one_epoch`).

	Fields
	------
	lr : float
		AdamW learning rate.
	weight_decay : float
		AdamW weight decay.
	betas : tuple[float, float]
		AdamW (beta1, beta2).
	batch_size : int
		Sequences per optimisation step (also used as the eval batch size).
	grad_clip : float | None
		Max global gradient norm; None disables clipping.
	seed : int
		Seed for both weight initialisation and the training-data shuffle.
	device : str | None
		Torch device ('cpu', 'cuda', 'mps'). None auto-selects via
		`resolve_device` (cuda, then mps, then cpu).
	dtype : {'float32', 'float64', 'bfloat16'}
		Compute / parameter dtype (mapped to a torch.dtype by `_DTYPES`).
	"""
	lr: float = 1e-3
	weight_decay: float = 1e-2
	betas: tuple = (0.9, 0.98)
	batch_size: int = 256
	grad_clip: float | None = 1.0
	seed: int = 0                    # weight init + data-shuffle seed
	device: str | None = None        # None -> auto (cuda, then mps, then cpu)
	dtype: str = 'float32'


_DTYPES = {'float32': torch.float32, 'float64': torch.float64, 'bfloat16': torch.bfloat16}


def resolve_device(device):
	"""
	Turn a device spec into a concrete `torch.device`.

	Params
	------
	device : str | torch.device | None
		Explicit device to honour, or None to auto-select the fastest available
		backend: CUDA, then Apple MPS, then CPU.

	Returns
	-------
	torch.device
		The resolved device.
	"""
	if device is not None:
		return torch.device(device)
	if torch.cuda.is_available():
		return torch.device('cuda')
	if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
		return torch.device('mps')
	return torch.device('cpu')


# ----------------------------------------------------------------------------
# model construction
# ----------------------------------------------------------------------------

def build_model(vocab_size, max_seq_len, mcfg, device, dtype, seed):
	"""
	Instantiate a fresh, randomly initialised `TransformerLM`.

	The model owns its (random) initialisation; we only fix the seed so that the
	untrained baseline and the trained model start from identical weights. The
	init generator is created on the resolved device so trunc-normal init matches
	the on-device weight tensors.

	Params
	------
	vocab_size : int
		Number of token identities including EOS (i.e. V + 1 for a DAG over
		vocabulary V).
	max_seq_len : int
		Maximum sequence length for the RoPE cache; here T + 1 (longest sample in
		characters, plus the EOS slot).
	mcfg : ModelConfig
		Architecture hyper-parameters (the inputs to model.TransformerLM).
	device : str | torch.device | None
		Target device; None auto-selects via `resolve_device`.
	dtype : torch.dtype
		Parameter / compute dtype (e.g. torch.float32).
	seed : int
		Seed for weight initialisation.

	Returns
	-------
	model.TransformerLM
		The freshly initialised model, on `device`.
	"""
	device = resolve_device(device)
	gen = torch.Generator(device=device).manual_seed(int(seed))
	return TransformerLM(
		d_model=mcfg.d_model,
		d_ff=mcfg.d_ff,          # None -> SwiGLU picks ~8/3 * d_model
		num_heads=mcfg.num_heads,
		vocab_size=vocab_size,
		num_layers=mcfg.num_layers,
		theta=mcfg.theta,
		max_seq_len=max_seq_len,
		use_rope=mcfg.use_rope,
		rng=gen,
		dtype=dtype,
		device=device,
	)


# ----------------------------------------------------------------------------
# token-identity permutations (experiment 1)
# ----------------------------------------------------------------------------

def make_permutations(num_bins, V, seed):
	"""
	Deterministic per-block token permutations sigma_0, ..., sigma_{num_bins-1}.

	Params
	------
	num_bins : int
		Number of position blocks to generate a permutation for.
	V : int
		Vocabulary size; each permutation is a bijection of {0, ..., V-1}.
	seed : int
		Seed for the permutation draws, fixing the sigmas deterministically.

	Returns
	-------
	list[np.ndarray]
		`num_bins` arrays of shape (V,); perms[k][old] = new token identity for
		characters in the k-th block of positions.
	"""
	rng = np.random.default_rng(seed)
	return [rng.permutation(V) for _ in range(num_bins)]


def permute_chars(chars, perms, bin_width):
	"""
	Relabel token identities position-block-wise.

	The character at position t is mapped by sigma_{t // bin_width}, i.e. token
	value c becomes perms[t // bin_width][c]. The relabelling is a bijection at
	each position, so path structure is preserved up to renaming.

	Params
	------
	chars : (num, T) int array
		Character sequences with values in [0, V); padding is -1. EOS is added
		later (in `build_full_tokens`) and is never permuted.
	perms : list[np.ndarray] | None
		Per-block permutations from `make_permutations`. None returns a copy of
		`chars` unchanged.
	bin_width : int
		Number of consecutive positions sharing one permutation.

	Returns
	-------
	(num, T) int array
		The relabelled characters, with padding (-1) left untouched.
	"""
	if perms is None:
		return chars.copy()
	out = chars.copy()
	T = chars.shape[1]
	for t in range(T):
		P = perms[t // bin_width]
		col = chars[:, t]
		real = col >= 0
		out[real, t] = P[col[real]]
	return out


# ----------------------------------------------------------------------------
# packing samples into model input tensors
# ----------------------------------------------------------------------------

def build_full_tokens(input_chars, lengths, V):
	"""
	Turn (chars, lengths) into a padded token matrix with a trailing EOS.

	The alphabet is {0, ..., V-1}; the token `V` is used both as the
	end-of-sequence marker and as the padding token, so every position from
	`lengths[i]` onward is `V` (an EOS followed by `V`-padding). Generation is
	meant to stop the first time `V` is observed.

	Params
	------
	input_chars : (num, T) int array
		Character sequences with values in [0, V); the "no character" sentinel is
		-1. May be the original or the permuted characters.
	lengths : (num,) int array
		Number of characters in each sample (n + number of edges); the EOS lands
		at index `lengths[i]`.
	V : int
		Vocabulary size; `V` is the shared EOS / padding token id.

	Returns
	-------
	(num, T + 1) int64 array
		Token ids per sample: the characters in columns [0, lengths[i]), then
		`V` for every remaining column.
	"""
	num, T = input_chars.shape
	K = T + 1
	full = np.full((num, K), V, dtype=np.int64)   # EOS == padding == V
	r, c = np.where(input_chars >= 0)
	full[r, c] = input_chars[r, c]                 # overwrite with real characters
	return full


def _node_values(chars, n, V):
	"""
	De Bruijn node value of the n-gram window ending at every position.

	Params
	------
	chars : (num, T) int array
		Character sequences (values in [0, V); padding -1).
	n : int
		n-gram / window length.
	V : int
		Vocabulary size.

	Returns
	-------
	(num, T) int64 array
		`node_at[:, i]` is the node value of chars[:, i-n+1 : i+1], valid for
		i >= n-1 (earlier columns are -1). The encoding matches `generate_data`:
		the earliest character of the window is least significant.
	"""
	num, T = chars.shape
	powers = (V ** np.arange(n)).astype(np.int64)
	node_at = np.full((num, T), -1, dtype=np.int64)
	c = chars.astype(np.int64)
	for i in range(n - 1, T):
		node_at[:, i] = (c[:, i - n + 1:i + 1] * powers).sum(axis=1)
	return node_at


# ----------------------------------------------------------------------------
# DAG-derived legality and training coverage
# ----------------------------------------------------------------------------

def build_legal(dag):
	"""
	Table of legal next tokens for every possible node value.

	Params
	------
	dag : DeBruijn_DAG
		The pruned De Bruijn DAG (see theory.theorem1_theorem2.DeBruijn_DAG);
		provides V, n, the edge arrays and the sink node values.

	Returns
	-------
	(V^n, V+1) bool array
		legal[u, b] is True iff token b is a legal next token at node value u.
		Non-sink nodes allow their out-edge branch characters; sink nodes allow
		only EOS (token V).
	"""
	V, n = dag.V, dag.n
	M = dag.num_nodes_all
	Vp = V ** (n - 1)
	legal = np.zeros((M, V + 1), dtype=bool)
	eb = dag.edges_v // Vp                       # appended character of each edge
	legal[dag.edges_u, eb] = True
	legal[dag.sink_vals, V] = True
	return legal


def training_visits(chars, lengths, dag):
	"""
	Which states the training set actually visited.

	Params
	------
	chars : (num, T) int array
		Original (unpermuted) training character sequences.
	lengths : (num,) int array
		Number of characters in each training sample.
	dag : DeBruijn_DAG
		The DAG the samples were drawn from (provides V, n, num_nodes_all).

	Returns
	-------
	seen_states : (V^n,) bool
		seen_states[u] is True iff some training path visited node u.
	"""
	V, n = dag.V, dag.n
	M = dag.num_nodes_all
	seen_states = np.zeros(M, dtype=bool)

	num, T = chars.shape
	node_at = _node_values(chars, n, V)
	idx = np.arange(T)
	# prediction states are windows ending at i in [n-1, l-1]
	valid = (idx[None, :] >= n - 1) & (idx[None, :] <= (lengths[:, None] - 1))
	rows, cols = np.where(valid)
	seen_states[node_at[rows, cols]] = True
	return seen_states


# ----------------------------------------------------------------------------
# training (one epoch)
# ----------------------------------------------------------------------------

def train_one_epoch(model, full_tokens, lengths, n, tcfg):
	"""
	Train `model` for exactly one pass over the data.

	Loss is next-token cross-entropy restricted to genuine prediction states:
	input position i in [n-1, l-1] for each sample (predicting the branch/EOS at
	i+1). The first n-1 positions (which merely spell out the start node) and all
	padding are excluded.

	Params
	------
	model : model.TransformerLM
		Model to train in place.
	full_tokens : (num, K) int array
		Padded token matrix with EOS from `build_full_tokens`; K = T + 1.
	lengths : (num,) int array
		Number of characters per sample (used to build the loss mask).
	n : int
		n-gram length (determines the first predicted position, n-1).
	tcfg : TrainConfig
		Optimisation settings (lr, batch_size, device, seed, ...).

	Returns
	-------
	float
		Mean per-batch training loss over the epoch (nan if there were no
		batches).
	"""
	device = resolve_device(tcfg.device)
	model.train()
	opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
							weight_decay=tcfg.weight_decay, betas=tuple(tcfg.betas))

	ft = torch.from_numpy(full_tokens)
	lt = torch.from_numpy(np.asarray(lengths, dtype=np.int64))
	num, K = full_tokens.shape
	pos = torch.arange(K - 1)

	g = torch.Generator().manual_seed(int(tcfg.seed))
	order = torch.randperm(num, generator=g)

	losses = []
	for s in range(0, num, tcfg.batch_size):
		idx = order[s:s + tcfg.batch_size]
		batch = ft[idx].to(device)
		blen = lt[idx].to(device)

		inp = batch[:, :-1]
		tgt = batch[:, 1:]
		logits = model(inp)                      # (b, K-1, vocab)

		mask = (pos.to(device)[None, :] >= n - 1) & (pos.to(device)[None, :] <= blen[:, None] - 1)
		loss = F.cross_entropy(logits[mask], tgt[mask])

		opt.zero_grad()
		loss.backward()
		if tcfg.grad_clip:
			torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
		opt.step()
		losses.append(float(loss.item()))

	return float(np.mean(losses)) if losses else float('nan')


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, orig_chars, lengths, dag, perms=None, bin_width=None,
			 seen_states=None, true_branch_p=None, path_ratio=DEFAULT_PATH_RATIO,
			 device=None, batch_size=512):
	"""
	Compute illegal_mass (and optionally path_coverage) on a test set.

	`orig_chars`/`lengths` are the *unpermuted* samples (they define the states
	and legal transitions). If `perms` is given, the model is fed the permuted
	tokens and its output probabilities are mapped back to the original token
	identities before scoring -- the metrics are therefore invariant to the
	relabelling and comparable across experiments.

	When `seen_states` is supplied, illegal_mass is additionally split by whether
	the state was visited in training. When `true_branch_p` is supplied,
	path_coverage is also computed.

	Params
	------
	model : model.TransformerLM
		Trained (or untrained, for a baseline) model to score.
	orig_chars : (num, T) int array
		Original, unpermuted test character sequences, drawn from the true path
		distribution (`ReasoningGenerator.sample`).
	lengths : (num,) int array
		Number of characters in each test sample.
	dag : DeBruijn_DAG
		DAG defining legality (via `build_legal`).
	perms : list[np.ndarray] | None
		Per-block token permutations (experiment 1); None means no relabelling.
	bin_width : int | None
		Block width for `perms`; required when `perms` is given.
	seen_states : (V^n,) bool | None
		Training state coverage from `training_visits`; enables the seen/unseen
		split of illegal_mass.
	true_branch_p : (V^n, V+1) float | None
		The generator's true next-token distribution per node
		(`ReasoningGenerator.branch_p`). When given, path_coverage is reported.
	path_ratio : float
		A path counts as covered if the model assigns it at least this fraction
		of its true probability (see DEFAULT_PATH_RATIO).
	device : str | torch.device | None
		Device to run the model on; None auto-selects (see `resolve_device`).
	batch_size : int
		Number of sequences scored per forward pass.

	Returns
	-------
	dict
		Always present:
		- 'illegal_mass' : float
			Mean probability on illegal tokens, averaged over prediction states.
		- 'num_states' : int
			Number of prediction states scored.
		- 'num_legal' : int
			Number of legal-transition instances scored.
		Present only when `seen_states` is passed (nan for an empty group):
		- 'illegal_mass_seen' / 'illegal_mass_unseen' : float
			illegal_mass split by whether the state was seen in training.
		Present only when `true_branch_p` is passed:
		- 'path_coverage' : float
			Fraction of test paths whose model probability is at least
			`path_ratio` times their true probability -- i.e. the proportion of
			(true-distribution) paths the model still assigns non-trivial
			probability to.
	"""
	V, n = dag.V, dag.n
	device = resolve_device(device)
	model.eval()

	legal_mat = build_legal(dag)
	num, T = orig_chars.shape
	K = T + 1

	inp_chars = permute_chars(orig_chars, perms, bin_width) if perms is not None else orig_chars
	full = build_full_tokens(inp_chars, lengths, V)
	node_at = _node_values(orig_chars, n, V)
	lengths = np.asarray(lengths)
	pos = np.arange(K - 1)

	illegal_all, state_seen_all = [], []
	tot_legal = 0
	# per-path log-probability of the actual path, under the model and the truth
	logp_model = np.zeros(num) if true_branch_p is not None else None
	logp_true = np.zeros(num) if true_branch_p is not None else None

	for s in range(0, num, batch_size):
		e = min(num, s + batch_size)
		binp = torch.from_numpy(full[s:e, :K - 1]).to(device)
		logits = model(binp)
		probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

		blen = lengths[s:e]
		valid = (pos[None, :] >= n - 1) & (pos[None, :] <= (blen[:, None] - 1))
		rows, cols = np.where(valid)
		if len(rows) == 0:
			continue

		pv = probs[rows, cols, :]                # (P, V+1) in model/permuted token space
		u = node_at[s + rows, cols]
		legal = legal_mat[u]

		# map probabilities back to original token identities (branch targets only;
		# EOS is never permuted and illegal_mass is relabel-invariant there)
		op = pv.copy()
		if perms is not None:
			j = cols + 1
			binj = j // bin_width
			is_branch = j < blen[rows]
			for b in np.unique(binj[is_branch]):
				sel = np.where(is_branch & (binj == b))[0]
				op[np.ix_(sel, np.arange(V))] = pv[sel][:, perms[b]]

		illegal = (op * (~legal)).sum(axis=1)
		tot_legal += int(legal.sum())
		illegal_all.append(illegal)

		if seen_states is not None:
			state_seen_all.append(seen_states[u])

		if true_branch_p is not None:
			# the actual next token in original identities: the branch char, or
			# EOS (token V) at the terminal step
			gr = s + rows
			j = cols + 1
			correct = np.full(len(rows), V, dtype=np.int64)
			br = j < blen[rows]
			correct[br] = orig_chars[gr[br], j[br]]
			p_model = op[np.arange(len(rows)), correct]
			p_true = true_branch_p[u, correct]
			np.add.at(logp_model, gr, np.log(p_model + 1e-30))
			np.add.at(logp_true, gr, np.log(p_true + 1e-30))

	illegal_all = np.concatenate(illegal_all) if illegal_all else np.zeros(0)
	res = {
		'illegal_mass': float(illegal_all.mean()) if len(illegal_all) else float('nan'),
		'num_states': int(len(illegal_all)),
		'num_legal': int(tot_legal),
	}
	if seen_states is not None:
		ss = np.concatenate(state_seen_all) if state_seen_all else np.zeros(0, bool)
		res['illegal_mass_seen'] = float(illegal_all[ss].mean()) if ss.any() else float('nan')
		res['illegal_mass_unseen'] = float(illegal_all[~ss].mean()) if (~ss).any() else float('nan')
	if true_branch_p is not None:
		covered = (logp_model - logp_true) >= math.log(path_ratio)
		res['path_coverage'] = float(covered.mean()) if num else float('nan')
	return res


# ----------------------------------------------------------------------------
# caching
# ----------------------------------------------------------------------------

def _cache_key(config):
	"""
	Stable short hash of a JSON-serialisable config (the cache file stem).

	Params
	------
	config : dict
		Any JSON-serialisable configuration; keys are sorted before hashing.

	Returns
	-------
	str
		First 16 hex characters of the SHA-1 digest of the config.
	"""
	blob = json.dumps(config, sort_keys=True, default=str)
	return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _load_or_train(factory, train_fn, config, device, force=False):
	"""
	Return a trained model, loading a cached checkpoint when one exists.

	Params
	------
	factory : callable () -> TransformerLM
		Builds a fresh, untrained model.
	train_fn : callable (model) -> stats
		Trains the model in place and returns training stats (e.g. mean loss).
	config : dict
		Configuration hashed to key the checkpoint (see `_cache_key`).
	device : str | torch.device | None
		Device to map a loaded checkpoint onto.
	force : bool
		If True, retrain and overwrite any existing checkpoint.

	Returns
	-------
	model : TransformerLM
		The trained (or loaded) model.
	cached : bool
		True iff the model was loaded from cache rather than trained.
	stats : object | None
		Whatever `train_fn` returned, or None when loaded from cache.
	"""
	os.makedirs(CACHE_DIR, exist_ok=True)
	path = os.path.join(CACHE_DIR, _cache_key(config) + '.pt')
	model = factory()
	if os.path.exists(path) and not force:
		model.load_state_dict(torch.load(path, map_location=resolve_device(device)))
		return model, True, None
	stats = train_fn(model)
	torch.save(model.state_dict(), path)
	return model, False, stats


# ----------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------

def _run(sampler_name, dag, gen, train_data, test_data, perms, bin_width, mcfg, tcfg,
		 config, force, return_baseline):
	"""
	Shared training/eval driver for both experiments.

	Builds the model, trains one epoch (or loads a cached checkpoint), records
	training coverage, and evaluates on the test set (and optionally the
	untrained baseline).

	Params
	------
	sampler_name : str
		Name of the sampler that produced the data (recorded for clarity only).
	dag : DeBruijn_DAG
		The DAG the data was drawn from.
	gen : ReasoningGenerator
		The generator that produced the data (kept for symmetry / future use).
	train_data : tuple(chars, lengths)
		Original (unpermuted) training samples.
	test_data : tuple(chars, lengths)
		Original (unpermuted) test samples.
	perms : list[np.ndarray] | None
		Per-block token permutations, or None.
	bin_width : int | None
		Block width for `perms`.
	mcfg : ModelConfig
		Architecture config.
	tcfg : TrainConfig
		Optimisation config.
	config : dict
		Full configuration used as the cache key.
	force : bool
		Force retraining, ignoring any cached checkpoint.
	return_baseline : bool
		If True, also evaluate the untrained model and add 'untrained_*' keys.

	Returns
	-------
	metrics : dict
		Evaluation metrics (see `evaluate`) plus 'cached' (bool), 'train_loss'
		(float | None) and, when requested, 'untrained_*' baseline metrics.
	model : TransformerLM
		The trained model.
	dag : DeBruijn_DAG
		The DAG (passed through for downstream use).
	"""
	V, n = dag.V, dag.n
	vocab = V + 1
	tr_chars, tr_len = train_data
	te_chars, te_len = test_data
	T = tr_chars.shape[1]
	device = resolve_device(tcfg.device)
	dtype = _DTYPES[tcfg.dtype]

	tr_input = permute_chars(tr_chars, perms, bin_width) if perms is not None else tr_chars
	tr_full = build_full_tokens(tr_input, tr_len, V)
	max_seq_len = tr_full.shape[1]               # T + 1

	def factory():
		return build_model(vocab, max_seq_len, mcfg, device, dtype, tcfg.seed)

	def train_fn(model):
		return train_one_epoch(model, tr_full, tr_len, n, tcfg)

	seen_states = training_visits(tr_chars, tr_len, dag)

	out = {}
	if return_baseline:
		base = evaluate(factory(), te_chars, te_len, dag, perms=perms, bin_width=bin_width,
						seen_states=seen_states, true_branch_p=gen.branch_p,
						device=device, batch_size=tcfg.batch_size)
		out.update({f'untrained_{k}': v for k, v in base.items()})

	model, cached, train_loss = _load_or_train(factory, train_fn, config, device, force)

	metrics = evaluate(model, te_chars, te_len, dag, perms=perms, bin_width=bin_width,
					   seen_states=seen_states, true_branch_p=gen.branch_p,
					   device=device, batch_size=tcfg.batch_size)
	out.update(metrics)
	out['cached'] = cached
	out['train_loss'] = train_loss
	return out, model, dag


def run_experiment_1(V, n, S, N, num_train, num_test, bin_width=1, permute=False,
					 perm_seed=0, ordering='random', dag_seed=42, data_seed=0,
					 mcfg=None, tcfg=None, force=False, return_baseline=False):
	"""
	Experiment 1: uniform start-to-answer paths (`ReasoningGenerator.sample`).

	A model is trained for one epoch on paths drawn uniformly from the DAG and
	evaluated on a fresh test set. When `permute=True`, token identities are
	scrambled by fixed per-block permutations (block size `bin_width`, seeded by
	`perm_seed`) before training; the metrics are still computed in the original
	identity space.

	Params
	------
	V, n, S, N : int
		De Bruijn DAG parameters -- vocabulary, n-gram length, number of source
		and answer nodes (see `generate_data.build_dag`).
	num_train : int
		Number of training paths (one epoch is a single pass over these).
	num_test : int
		Number of test paths used for evaluation.
	bin_width : int
		Width of each position block sharing one token permutation (used only
		when `permute=True`).
	permute : bool
		Whether to apply the position-dependent token relabelling.
	perm_seed : int
		Seed for the deterministic permutations.
	ordering : {'random', 'digit_sum'}
		Topological ordering used to build the DAG (see `build_dag`).
	dag_seed : int
		Seed for the random DAG ordering.
	data_seed : int
		Seed for sampling paths; train and test use distinct sub-streams.
	mcfg : ModelConfig | None
		Architecture config; defaults to `ModelConfig()`.
	tcfg : TrainConfig | None
		Optimisation config; defaults to `TrainConfig()`.
	force : bool
		Retrain even if a cached checkpoint exists.
	return_baseline : bool
		Also evaluate the untrained model (adds 'untrained_*' keys).

	Returns
	-------
	metrics : dict
		Evaluation metrics (see `evaluate` / `_run`).
	model : TransformerLM
		The trained model.
	dag : DeBruijn_DAG
		The DAG used.
	"""
	mcfg = mcfg or ModelConfig()
	tcfg = tcfg or TrainConfig()
	dag = build_dag(V, n, S, N, ordering=ordering, seed=dag_seed)
	gen = ReasoningGenerator(dag)

	tr = gen.sample(num_train, np.random.default_rng([data_seed, 1]))
	te = gen.sample(num_test, np.random.default_rng([data_seed, 2]))

	perms = None
	if permute:
		T = tr[0].shape[1]
		perms = make_permutations(math.ceil(T / bin_width), V, perm_seed)

	config = dict(exp=1, V=V, n=n, S=S, N=N, ordering=ordering, dag_seed=dag_seed,
				  data_seed=data_seed, num_train=num_train, permute=bool(permute),
				  bin_width=bin_width, perm_seed=perm_seed,
				  model=asdict(mcfg), train=asdict(tcfg))
	return _run('sample', dag, gen, tr, te, perms, bin_width, mcfg, tcfg,
				config, force, return_baseline)


def run_experiment_2(V, n, S, N, L, num_train, num_test, ordering='random',
					 dag_seed=42, data_seed=0, mcfg=None, tcfg=None,
					 force=False, return_baseline=False):
	"""
	Experiment 2: length-limited paths
	(`ReasoningGenerator.sample_length_limited`), restricted to source-to-sink
	paths of at most `L` edges. A model is trained for multiple epochs and evaluated as
	in experiment 1, but without token permutation.

	Params
	------
	V, n, S, N : int
		De Bruijn DAG parameters (see `generate_data.build_dag`).
	L : int
		Maximum number of edges per sampled path.
	num_train : int
		Number of training paths.
	num_test : int
		Number of test paths used for evaluation.
	ordering : {'random', 'digit_sum'}
		Topological ordering used to build the DAG.
	dag_seed : int
		Seed for the random DAG ordering.
	data_seed : int
		Seed for sampling paths; train and test use distinct sub-streams.
	mcfg : ModelConfig | None
		Architecture config; defaults to `ModelConfig()`.
	tcfg : TrainConfig | None
		Optimisation config; defaults to `TrainConfig()`.
	force : bool
		Retrain even if a cached checkpoint exists.
	return_baseline : bool
		Also evaluate the untrained model (adds 'untrained_*' keys).

	Returns
	-------
	metrics : dict
		Evaluation metrics (see `evaluate` / `_run`).
	model : TransformerLM
		The trained model.
	dag : DeBruijn_DAG
		The DAG used.
	"""
	mcfg = mcfg or ModelConfig()
	tcfg = tcfg or TrainConfig()
	dag = build_dag(V, n, S, N, ordering=ordering, seed=dag_seed)
	gen = ReasoningGenerator(dag)

	tr = gen.sample_length_limited(num_train, L, np.random.default_rng([data_seed, 1]))
	te = gen.sample_length_limited(num_test, L, np.random.default_rng([data_seed, 2]))

	config = dict(exp=2, V=V, n=n, S=S, N=N, L=L, ordering=ordering, dag_seed=dag_seed,
				  data_seed=data_seed, num_train=num_train,
				  model=asdict(mcfg), train=asdict(tcfg))
	return _run('sample_length_limited', dag, gen, tr, te, None, None, mcfg, tcfg,
				config, force, return_baseline)
