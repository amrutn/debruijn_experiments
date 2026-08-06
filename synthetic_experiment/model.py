"""
Stores the model class that we are using to run the synthetic experiment:
how much data and what length of paths are required for a model to learn the
paths in a reasoning tree. 

From scratch implementation of a pre-norm Transformer model.

"""

import math
from collections import defaultdict

import torch
import torch.nn as nn
import einx


class Linear(nn.Module):
	"""
	Simple linear unit, predicts Wx for a vector x and
	weight matrix W.

	Initializes weights according to a Gaussian.
	"""

	def __init__(self,
				in_features: int,
				out_features: int,
				weights : torch.Tensor | None = None,
				device: torch.device | None = None,
				dtype: torch.dtype | None = None, 
				rng : torch.Generator | None = None
				):
		"""
		Params
		------
		in_features : int
			Dimension of the input.
		out_features : int
			Dimension of the output
		device : torch.device
			Device to store parameters on
		dtype : torch.dtype
			Data type of the parameters
		rng : torch.Generator
			Random number generator for the model
		"""
		super().__init__()

		self.din = in_features
		self.dout = out_features
		self.device = device
		self.dtype = dtype
		# random number generator
		if rng is None:
			self.rng = torch.Generator().manual_seed(42)
		else:
			self.rng=rng

		if weights is None:
			# sample weights according to a truncated Gaussian
			std = math.sqrt(2/(self.din + self.dout))
			weights = torch.empty(self.dout, self.din).to(device=device, dtype=dtype)
			nn.init.trunc_normal_(weights,
								mean=0,
								std=std, a=-3*std, b=3*std,
								generator=self.rng)

		self.weights = nn.Parameter(weights).to(device=device, dtype=dtype)
		assert self.weights.size() == (self.dout, self.din)

	def forward(self, x:torch.Tensor) -> torch.Tensor:
		"""
		Forward pass returning Wx

		Params
		------
		x : torch.Tensor (...,self.din)
			The input

		Returns
		-------
		y : torch.tensor (..., self.dout)
			The output
		"""
		x=x.to(dtype=self.dtype, device=self.device)
		return einx.dot('dout [din], ... [din] -> ... dout', self.weights, x)


class Embedding(nn.Module):
	"""
	Embedding layer which converts a sequence of token_IDs into 
	a sequence of vectors of dimension d_model. 

	Initializes weights according to a Gaussian. 
	"""

	def __init__(self,
				num_embeddings : int,
				embedding_dim : int,
				weights : torch.Tensor | None=None,
				device : torch.device | None = None,
				dtype : torch.dtype | None = None,
				rng : torch.Generator | None = None
				):
		"""
		Params
		------
		num_embeddings : int
			Vocabulary size
		embedding_dim : int
			d_model, the embedding dimension
		device : torch.device
			Device to store parameters
		dtype : torch.dtype
			Data type of the parameters
		rng : torch.Generator
			Random number generator for the model
		"""

		super().__init__()

		self.vocab_size = num_embeddings
		self.d_model = embedding_dim
		self.device = device
		self.dtype = dtype

		# random number generator
		if rng is None:
			self.rng = torch.Generator().manual_seed(42)
		else:
			self.rng=rng

		if weights is not None:
			self.weights = nn.Parameter(weights).to(device=device, dtype=dtype)
			assert weights.size() == (self.vocab_size, self.d_model)
		else:
			# initialize random weights
			weights = torch.empty(self.vocab_size, self.d_model).to(device=device, dtype=dtype)
			nn.init.trunc_normal_(weights,
								mean=0,
								std=1, a=-3, b=3,
								generator=self.rng)
			self.weights = nn.Parameter(weights)

	def forward(self, token_ids : torch.Tensor) -> torch.Tensor:
		"""
		Look up the embedding vectors for each token ID.

		Params
		------
		token_ids : torch.Tensor 
			The integer token ids in sequence
		"""
		token_ids = token_ids.to(dtype=torch.long, device=self.device)
		return einx.get_at("[vocab] d, ... seq -> ... seq d", self.weights, token_ids)


class RMSNorm(nn.Module):
	"""
	Implements the RMS pre-norm block for the Transformer
	"""
	def __init__(self,
				d_model : int,
				eps : float=1e-5,
				weights : torch.Tensor | None = None,
				device : torch.device | None = None,
				dtype : torch.dtype | None = None
				):

		"""
		Params
		------
		d_model : int
			Embedding dimension.
		eps : float
			small epsilon to prevent divide by zero errors
		weights : torch.Tensor [d_model]
			The activation gains for each input dim
		device : torch.device
		dtype : torch.dtype
		"""
		super().__init__()

		self.d_model = d_model
		self.eps = eps
		self.device = device
		self.dtype = dtype

		# learning gain param
		if weights is not None:
			self.weights = weights.to(dtype=self.dtype, device=self.device)
			assert weights.size() == (d_model,)
		else:
			self.weights = torch.ones(self.d_model, dtype=self.dtype, device=self.device)

	def forward(self, x : torch.Tensor) -> torch.Tensor:
		"""
		Normalize the input tensor x by RMS norm.

		Params
		------
		x : torch.Tensor
			Input tensor with final shape d_model
		"""

		assert x.size(-1) == self.d_model

		rms = torch.sqrt(einx.mean("... d_model -> ...",x**2) + self.eps)
		normalized = einx.divide("... d_model, ... -> ... d_model", x, rms)

		return einx.multiply("d_model, ... d_model -> ... d_model", self.weights, normalized)



class SwiGLU(nn.Module):
	"""
	Implements a feed-forward network with one hidden layer and the SwiGLU activation.
	W2(SiLU(W1 x) dot W3 x)
	"""

	def __init__(self,
				d_model : int,
				d_ff : int | None = None,
				w1_weight : torch.Tensor | None = None,
				w2_weight : torch.Tensor | None = None,
				w3_weight : torch.Tensor | None = None,
				dtype : torch.dtype | None = None, 
				device : torch.device | None = None,
				rng : torch.Generator | None = None
				):
		"""
		Params
		------
		d_model : int
			Embedding dimension
		d_ff : int
			Hidden layer dimension, if None then 8/3 * d_model
		w1_weight : torch.Tensor
			Weight inside the SILU activation, SILU(W1 x)
		w2_weight : torch.Tensor
			Final layer weight in the feedforward network
		w3_weight : torch.Tensor
			Weights for the gating W3 x
		dtype : torch.dtype
		device : torch.device
		rng : torch.Generator
			Random number generator for the model
		"""
		super().__init__()

		# random number generator
		if rng is None:
			self.rng = torch.Generator().manual_seed(42)
		else:
			self.rng=rng

		self.d_model = d_model
		if d_ff is None:
			d_ff = int((8/3 * d_model * 64)//64) # nearest dim to a multiple of 64
		self.d_ff = d_ff
		self.device = device
		self.dtype = dtype

		self.linear1 = Linear(in_features = d_model,
								out_features = d_ff,
								weights = w1_weight,
								rng=self.rng,
								dtype=dtype,
								device=device)
		self.linear2 = Linear(in_features = d_ff,
								out_features = d_model,
								weights = w2_weight,
								rng=self.rng,
								dtype=dtype,
								device=device)

		self.linear3 = Linear(in_features = d_model,
								out_features = d_ff,
								weights = w3_weight,
								rng=self.rng,
								dtype=dtype,
								device=device)


	def forward(self, x : torch.Tensor) -> torch.Tensor:
		"""
		Implements forward pass of the feedforward with swiglu

		Params
		------
		x : torch.Tensor
			Input of size d_model
		"""
		x = x.to(dtype=self.dtype, device=self.device)
		first = self.linear1(x)
		first_silu = first*torch.sigmoid(first)
		second = first_silu*self.linear3(x)
		third = self.linear2(second)

		return third


class RoPE(nn.Module):
	"""
	Implement the rope embeddings
	"""

	def __init__(self,
				theta : float,
				d_k : int, 
				max_seq_len : int,
				device : torch.device | None = None,
				dtype : torch.dtype | None = None
				):
		"""
		Params
		------
		theta : float
			theta value for the rope
		d_k : int
			dimension of key and query vectors
		max_seq_len : int
			maximum sequence length
		device : torch.device
		dtype : torch.dtype
		"""
		super().__init__()

		self.d_k = d_k
		self.max_seq_len = max_seq_len
		self.device = device
		self.dtype = dtype

		def angle_formula(i,k):
			return (i)/(theta**((2*k)/d_k))

		# a matrix of the theta values ordered by [sequence index, dimension]
		ein_angle_formula = einx.torch.adapt_with_vmap(angle_formula)
		i_idx = torch.arange(max_seq_len)
		k_idx = torch.arange(d_k // 2)

		thetas = ein_angle_formula("i, k -> i k", i_idx, k_idx)
		thetas = thetas.to(device=device, dtype=dtype)

		self.register_buffer('cosine_vals', torch.cos(thetas), persistent=False)
		self.register_buffer('sine_vals', torch.sin(thetas), persistent=False)


	def forward(self, x : torch.Tensor, token_positions : torch.Tensor) -> torch.Tensor:
		"""
		Params
		------
		x : torch.Tensor
			Input tensor of shape (..., seq_len, d_k)
		token_positions : torch.Tensor
			The positions associated with each token, of shape (..., seq_len)

		Returns
		-------
		out : torch.Tensor
			Output tensor with position encodings with the same shape as the input x.
		"""

		assert x.size(-1) == self.d_k
		x=x.to(device=self.device, dtype=self.dtype)

		# Creating a tensor of cos and sin values based on token_positions
		cos = einx.get_at("[max_seq_len] half_d, ... seq_len -> ... seq_len half_d", self.cosine_vals, token_positions)
		sin = einx.get_at("[max_seq_len] half_d, ... seq_len -> ... seq_len half_d", self.sine_vals, token_positions)

		# Use einx to split x into interleaved pairs
		x_pairs = einx.id("... (half_d c) -> ... half_d c", x, c=2)

		# create rotated pairs (this is to apply the sine part of the rotation matrix)
		x1 = x_pairs[...,0]
		x2 = x_pairs[...,1]
		x_rotated = torch.stack([-x2,x1], dim=-1)

		# Output cosines and sines
		out_cos = x_pairs * einx.id("... -> ... 1", cos)
		out_sin = x_rotated * einx.id("... -> ... 1", sin)

		out_pairs = out_cos + out_sin

		# arrange the pairs back into a single vector
		out = einx.id("... half_d c -> ... (half_d c)", out_pairs)

		return out

def scaled_attention(keys : torch.Tensor,
						 queries : torch.Tensor,
						  values : torch.Tensor,
						  causal_mask : torch.Tensor | None = None) -> torch.Tensor:
	"""
	Implement the attention operation, scaled by the square-root of the dimension.


	Params
	------
	keys : torch.Tensor
		Key matrix of size (batch, ..., seq_len, d_k)
	queries : torch.Tensor
		Query matrix of size (batch, ..., seq_len, d_k)
	values : torch.Tensor
		Value matrix of size (batch, ..., seq_len, d_v)
	causal_mask : torch.Tensor
		Masking for the attention operation (seq_len, seq_len).
		Use a standard causal mask if None.

	Returns
	-------
	out : torch.Tensor
		output matrix of size (batch, ..., seq_len, d_v)

	"""
	seq_len = keys.size()[-2]
	d_k = keys.size()[-1]
	if causal_mask is None:
		causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=keys.device))

	# Calculate QK^T
	prod = einx.dot("... queries_seq [d_k], ... keys_seq [d_k] -> ... queries_seq keys_seq", queries, keys)
	scaled_prod = prod/math.sqrt(d_k)
	# apply causal mask
	scaled_prod = scaled_prod.masked_fill(~causal_mask, float("-inf"))
	# apply softmax
	softmax_prod = torch.softmax(scaled_prod, dim=-1)
	# multiply by values
	out = einx.dot("... queries_seq [keys_seq], ... [keys_seq] d_v -> ... queries_seq d_v", softmax_prod, values)
	return out


class MultiHead_Attention(nn.Module):
	"""
	Implements the classic multi-head attention with a causal mask. 

	Each head has the same dimension and has the same set of positional embeddings.
	The key/query dimension d_k is just d_model//num_heads.
	"""

	def __init__(self,
			 d_model : int,
			 num_heads : int,
			 W_Q : torch.Tensor | None = None,
			 W_K : torch.Tensor | None = None, 
			 W_V : torch.Tensor | None = None,
			 W_O : torch.Tensor | None = None,
			 theta : float = 10000,
			 max_seq_len : int = 1000,
			 use_rope : bool = True,
			 rng : torch.Generator | None = None,
			 dtype : torch.dtype = torch.float32,
			 device : torch.device | None = None):
		"""
		Params
		------
		d_model : int
			Model dimension
		num_heads : int
			Number of heads in the attention
		W_Q : torch.Tensor
			Queries matrix, has size d_model x h*d_k
		W_K : torch.Tensor
			Keys matrix, has size d_model x h*d_k
		W_V : torch.Tensor
			Values matrix, has size d_model x h*d_k
		theta : float
			theta for the rope embeddings
		max_seq_len : int
			Maximum context length for the rope embeddings
		use_rope : bool
			Whether or not to use rope embeddings
		rng : torch.Generator
			Random number generator
		dtype : torch.dtype
		device : torch.device
		"""
		super().__init__()
		# random number generator
		if rng is None:
			self.rng = torch.Generator().manual_seed(42)
		else:
			self.rng=rng

		self.d_model = d_model
		self.num_heads = num_heads
		self.d_k = self.d_model//num_heads
		self.max_seq_len = max_seq_len
		self.dtype=dtype
		self.device=device
		if W_Q is None:
			std = math.sqrt(1/self.d_model)
			W_Q = torch.empty(self.d_k * self.num_heads,self.d_model).to(device=self.device, dtype=self.dtype)
			nn.init.trunc_normal_(W_Q,
								mean=0,
								std=std, a=-2*std, b=2*std,
								generator=self.rng)
		if W_K is None:
			std = math.sqrt(1/self.d_model)
			W_K = torch.empty(self.d_k * self.num_heads,self.d_model).to(device=self.device, dtype=self.dtype)
			nn.init.trunc_normal_(W_K,
								mean=0,
								std=std, a=-2*std, b=2*std,
								generator=self.rng)

		if W_V is None:
			std = math.sqrt(1/self.d_model)
			W_V = torch.empty(self.d_k * self.num_heads,self.d_model).to(device=self.device, dtype=self.dtype)
			nn.init.trunc_normal_(W_V,
								mean=0,
								std=std, a=-2*std, b=2*std,
								generator=self.rng)

		self.W_Q = nn.Parameter(W_Q).to(device=device, dtype=dtype)
		self.W_K = nn.Parameter(W_K).to(device=device, dtype=dtype)
		self.W_V = nn.Parameter(W_V).to(device=device, dtype=dtype)

		self.output_layer = Linear(self.d_k*self.num_heads,
									 self.d_model, W_O,
									 device=self.device,
									 dtype=self.dtype,
									 rng=self.rng)

		# define the rope embedding to use
		if use_rope:
			self.rope = RoPE(theta, self.d_k,
				self.max_seq_len, dtype=self.dtype,
				 device=self.device)
		else:
			self.rope = lambda x,y: x

	def forward(self, x : torch.Tensor, attn_mask : torch.Tensor | None = None,
				token_positions : torch.Tensor | None = None) -> torch.Tensor:
		"""
		Apply multi-head attention to an input tensor

		Params
		------
		x : torch.Tensor
			Input tensor of shape [batch, seq_len, d_model]
		attn_mask : torch.Tensor | None
			Optional boolean (seq_len, seq_len) mask (True = attend). If None, the
			usual causal mask is used. A sliding-window mask is passed here to
			restrict each query to a fixed number of recent positions.
		token_positions : torch.Tensor | None
			Optional RoPE positions, shape (seq_len,) or per-sample (batch, seq_len).
			None means the usual 0, 1, ..., seq_len-1. Non-contiguous positions are
			passed here to augment training with random inter-token gaps.
		Returns
		-------
		out : torch.Tensor
			Output of the same shape as input
		"""

		# first compute the key, query and value matrices - split across heads
		keys_combined = einx.dot("d_keys [d_model], batch seq_len [d_model] -> batch seq_len d_keys", self.W_K, x)
		keys = einx.id("batch seq_len (h d_k) -> batch h seq_len d_k", keys_combined, h=self.num_heads)

		queries_combined = einx.dot("d_keys [d_model], batch seq_len [d_model] -> batch seq_len d_keys", self.W_Q, x)
		queries = einx.id("batch seq_len (h d_k) -> batch h seq_len d_k", queries_combined, h=self.num_heads)

		values_combined = einx.dot("d_keys [d_model], batch seq_len [d_model] -> batch seq_len d_keys", self.W_V, x)
		values = einx.id("batch seq_len (h d_k) -> batch h seq_len d_k", values_combined, h=self.num_heads)

		# second apply rope. Positions default to 0..seq_len-1; a caller may pass
		# explicit positions (e.g. non-contiguous, for gap augmentation). Per-sample
		# positions (batch, seq_len) get a head axis so they broadcast over heads.
		if token_positions is None:
			token_positions = torch.arange(x.size(-2), dtype=torch.long, device=self.device)
		else:
			token_positions = token_positions.to(device=self.device, dtype=torch.long)
			if token_positions.dim() == 2:
				token_positions = token_positions.unsqueeze(1)   # (batch, 1, seq_len)
		embedded_keys = self.rope(keys, token_positions)
		embedded_queries = self.rope(queries, token_positions)


		# next compute the scaled dot product attention per head. A plain causal
		# mask is used unless a caller passes an explicit attn_mask (e.g. a
		# sliding-window mask).
		attention_out = scaled_attention(embedded_keys, embedded_queries, values, causal_mask=attn_mask)
		# reshape to combine the heads
		reshaped_attention = einx.id("batch h seq_len d_k -> batch seq_len (h d_k)", attention_out)

		out = self.output_layer(reshaped_attention)

		return out


class Transformer_Block(nn.Module):
	"""
	A single Transformer Block

	"""
	def __init__(self, d_model : int,
			 d_ff : int,
			 num_heads : int,
			 weights : dict[str, torch.Tensor|None] | None = None,
			 theta : float = 10000,
			 max_seq_len : int = 1000,
			 use_rope : bool = True,
			 rng : torch.Generator | None = None,
			 dtype : torch.dtype = torch.float32,
			 device : torch.device | None = None):
		"""
		Params
		------
		d_model : int
			Model dimension
		d_ff : int
			Feedforward dimension, typically 8/3 * model dimension
		num_heads : int
			Number of heads
		weights : dict[str, torch.Tensor|None] | None
			The weights dict. If any of them or None, or if no dict is passed,
			the weights without associated values will be initialized at random.
			The keys of this dictionary are:
			- `attn.q_proj.weight`
				The query projections for all `num_heads` attention heads.
				Shape is (d_model, d_model).
				Split across heads, W_Q = torch.cat([q_heads.0.weight, ..., q_heads.h.weight], dim=0)
			- `attn.k_proj.weight`
				The key projections for all `num_heads` attention heads.
				Shape is (d_model, d_model).
			- `attn.v_proj.weight`
				The value projections for all `num_heads` attention heads.
				Shape is (d_model, d_model).
			- `attn.output_proj.weight`
				Weight of the multi-head self-attention output projection
				Shape is (d_model, d_model).
			- `ln1.weight`
				Weights of affine transform for the first RMSNorm
				applied in the transformer block.
			    Shape is (d_model,).
			- `ffn.w1.weight`
				Weight of the first linear transformation in the FFN.
				Shape is (d_ff, d_model).
			- `ffn.w2.weight`
				Weight of the second linear transformation in the FFN.
				Shape is (d_model, d_ff).
			- `ffn.w3.weight`
				Weight of the third linear transformation in the FFN.
				Shape is (d_ff, d_model).
			- `ln2.weight`
				Weights of affine transform for the second RMSNorm
				applied in the transformer block.
				Shape is (d_model,).
		theta : float
			Theta value for the rope positional embedding
		max_seq_len : int
			Maximum context length for rope positional embedding
		use_rope : bool
			Whether or not to use the rope positional embedding
		rng : torch.Generator | None
			Random number generator
		dtype : torch.dtype
			dtype for the computations
		device : torch.device
			device for the computations
		"""
		super().__init__()
		self.d_model = d_model
		self.d_ff = d_ff
		self.num_heads = num_heads
		self.theta = theta
		self.max_seq_len = max_seq_len
		self.rng = rng
		self.dtype=dtype
		self.device=device

		if weights is None:
			# initialize dict with only None
			weights = defaultdict(lambda:None)

		# first pre-norm
		self.norm1 = RMSNorm(d_model, weights=weights['ln1.weight'], dtype=dtype, device=device)
		# attention module
		self.attn = MultiHead_Attention(d_model,
									 num_heads,
									  W_Q=weights['attn.q_proj.weight'],
									  W_K=weights['attn.k_proj.weight'],
									  W_V=weights['attn.v_proj.weight'],
									  W_O=weights['attn.output_proj.weight'],
									  theta=theta,
									  max_seq_len=max_seq_len,
									  use_rope=use_rope,
									  rng=rng,
									  dtype=dtype,
									  device=device)
		# second pre-norm
		self.norm2 = RMSNorm(d_model, weights=weights['ln2.weight'], dtype=dtype, device=device)
		# feedforward
		self.ffn = SwiGLU(d_model, d_ff,
							w1_weight=weights['ffn.w1.weight'],
							w2_weight=weights['ffn.w2.weight'],
							w3_weight=weights['ffn.w3.weight'],
							dtype=dtype, device=device, rng=rng)

	def forward(self, x : torch.Tensor, attn_mask : torch.Tensor | None = None,
				token_positions : torch.Tensor | None = None) -> torch.Tensor:
		"""
		Forward pass of the Transformer block
		Params
		------
		x : torch.Tensor
			Input tensor of shape (batch, seq_len, d_model)
		attn_mask : torch.Tensor | None
			Optional attention mask forwarded to the attention module.
		token_positions : torch.Tensor | None
			Optional RoPE positions forwarded to attention.
		Returns
		-------
		out : torch.Tensor
			Output tensor of the same shape as input
		"""

		first_normed = self.norm1(x)
		post_attn = self.attn(first_normed, attn_mask=attn_mask, token_positions=token_positions)

		# add residual
		first_residual = x + post_attn

		second_normed = self.norm2(first_residual)
		post_ffn = self.ffn(second_normed)

		# add residual
		second_residual = post_ffn + first_residual

		return second_residual

class TransformerLM(nn.Module):
	"""
	The entire model class
	"""
	def __init__(self, d_model : int,
			 d_ff : int,
			 num_heads : int,
			 vocab_size : int,
			 num_layers : int,
			 weights: dict[str, torch.Tensor|None]|None=None,
			 theta : float = 10000,
			 max_seq_len : int = 1000,
			 use_rope : bool = True,
			 rng : torch.Generator | None = None,
			 dtype : torch.dtype = torch.float32,
			 device : torch.device | None = None):
		"""
		Params
		------
		See above for parameter definitions

		vocab_size : int
			The vocabulary size
		num_layers : int
			Number of Transformer Blocks
		weights : dict
			Dictionary of weights for all layers. {num_layers} refers to an
			integer between `0` and `num_layers - 1` (the layer index).
			The keys of this dictionary are:
			- `token_embeddings.weight`
				Token embedding matrix. Shape is (vocab_size, d_model).
			- `layers.{num_layers}.attn.q_proj.weight`
				The query projections for all `num_heads` attention heads.
				Shape is (num_heads * (d_model / num_heads), d_model).
			- `layers.{num_layers}.attn.k_proj.weight`
				The key projections for all `num_heads` attention heads.
				Shape is (num_heads * (d_model / num_heads), d_model).
			- `layers.{num_layers}.attn.v_proj.weight`
				The value projections for all `num_heads` attention heads.
				Shape is (num_heads * (d_model / num_heads), d_model).
			- `layers.{num_layers}.attn.output_proj.weight`
				Weight of the multi-head self-attention output projection
				Shape is ((d_model / num_heads) * num_heads, d_model).
			- `layers.{num_layers}.ln1.weight`
				Weights of affine transform for the first RMSNorm
				applied in the transformer block.
			    Shape is (d_model,).
			- `layers.{num_layers}.ffn.w1.weight`
				Weight of the first linear transformation in the FFN.
				Shape is (d_ff, d_model).
			- `layers.{num_layers}.ffn.w2.weight`
				Weight of the second linear transformation in the FFN.
				Shape is (d_model, d_ff).
			- `layers.{num_layers}.ffn.w3.weight`
				Weight of the third linear transformation in the FFN.
				Shape is (d_ff, d_model).
			- `layers.{num_layers}.ln2.weight`
				Weights of affine transform for the second RMSNorm
				applied in the transformer block.
				Shape is (d_model,).
			- `ln_final.weight`
				Weights of affine transform for RMSNorm applied to the output of the final transformer block.
				Shape is (d_model, ).
			- `lm_head.weight`
				Weights of the language model output embedding.
				Shape is (vocab_size, d_model).

		"""
		super().__init__()
		if weights is None:
			weights = defaultdict(lambda:None)
			block_weights = [None]*num_layers
		else:
			# create weights dicts for transformer blocks
			block_weights = []
			for l in range(num_layers):
				layer_weights = {
					'attn.q_proj.weight' : weights[f"layers.{l}.attn.q_proj.weight"],
					'attn.k_proj.weight' : weights[f"layers.{l}.attn.k_proj.weight"],
					'attn.v_proj.weight' : weights[f"layers.{l}.attn.v_proj.weight"],
					'attn.output_proj.weight' : weights[f"layers.{l}.attn.output_proj.weight"],
					'ln1.weight' : weights[f"layers.{l}.ln1.weight"],
					'ln2.weight' : weights[f"layers.{l}.ln2.weight"],
					'ffn.w1.weight' : weights[f"layers.{l}.ffn.w1.weight"],
					'ffn.w2.weight' : weights[f"layers.{l}.ffn.w2.weight"],
					'ffn.w3.weight' : weights[f"layers.{l}.ffn.w3.weight"]
				}
				block_weights.append(layer_weights)



		self.embedding = Embedding(vocab_size,
									d_model,
									weights=weights['token_embeddings.weight'],
									device=device,
									dtype=dtype,
									rng=rng)

		self.transformer_blocks = nn.ModuleList([Transformer_Block(d_model,
			 d_ff,
			 num_heads,
			 weights=w,
			 theta=theta,
			 max_seq_len=max_seq_len,
			 use_rope=use_rope,
			 rng=rng,
			 dtype=dtype,
			 device=device) for w in block_weights])

		self.post_norm = RMSNorm(d_model,
								weights=weights['ln_final.weight'],
								device=device,
								dtype=dtype)

		self.out_embedding = Linear(d_model,
									vocab_size,
									weights=weights['lm_head.weight'],
									device=device,
									dtype=dtype, 
									rng=rng)

	def forward(self, x : torch.Tensor, attn_mask : torch.Tensor | None = None,
				token_positions : torch.Tensor | None = None) -> torch.Tensor:
		"""
		Forward pass of the model
		Params
		------
		x : torch.Tensor
			Sequence of integer token_ids of size (batch, seq_len)
		attn_mask : torch.Tensor | None
			Optional boolean (seq_len, seq_len) attention mask (True = attend),
			applied in every transformer block. If None, plain causal attention
			is used. A sliding-window mask is passed here for local attention.
		token_positions : torch.Tensor | None
			Optional RoPE positions, shape (seq_len,) or per-sample (batch, seq_len),
			applied in every block. None means the usual 0, 1, ..., seq_len-1;
			non-contiguous positions are passed here for gap augmentation.
		Returns
		-------
		out : torch.Tensor
			logits of the outputs of size (batch, vocab_size)
		"""
		# embedding
		embedded = self.embedding(x)
		# transformer blocks
		curr_res = embedded
		for block in self.transformer_blocks:
			curr_res = block(curr_res, attn_mask=attn_mask, token_positions=token_positions)
		# post norm
		post_normed = self.post_norm(curr_res)

		# out embedding
		out = self.out_embedding(post_normed)

		return out