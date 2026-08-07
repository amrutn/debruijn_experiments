"""
Measure and plot the token-level predictive entropy of a Qwen reasoning model as
a function of position in its own generated chain-of-thought ("entropy vs time"),
across several reasoning benchmarks, using only local NVIDIA GPUs.

What is measured
----------------
For a prompt, the model autoregressively generates a reasoning trace (thinking +
answer). At every generated position t the decoder induces a next-token
distribution p_t over the vocabulary; its Shannon entropy

    H_t = - sum_v p_t(v) log p_t(v)

quantifies how uncertain the model is at that step. We collect H_t for many
samples and problems and plot the mean H_t (with a shaded standard-error band)
against t. High early entropy that collapses over the trace, or entropy that
stays elevated on failures, are the kinds of effects this figure is meant to
expose.

vLLM only returns the top-`logprobs_k` logprobs per step (the tail is not
materialised), so H_t is the *truncated* top-k entropy -- a slight lower bound on
the true entropy. Because reasoning-model next-token distributions are strongly
peaked this is an accurate estimator for modest k (>= 20); raise `--logprobs-k`
(and the engine's `--max-logprobs`) if you need the tail. Entropy is reported in
bits (base-2) by default.

Correct vs incorrect
--------------------
Each generated trace is graded against the benchmark's gold answer (\\boxed{...}
extraction and normalisation for the math sets; final-letter matching for the
multiple-choice set). Besides the pooled figure, a per-(model, benchmark) figure
splits the mean entropy curve into correct and incorrect traces, so systematic
differences in the uncertainty dynamics of successful vs failed reasoning are
visible.

Models (all fit on 6x RTX 4090, 24 GB each, via vLLM tensor parallelism)
------------------------------------------------------------------------
    qwen3-8b        Qwen/Qwen3-8B            TP=1   bf16
    qwen3-14b       Qwen/Qwen3-14B           TP=2   bf16
    qwen3-32b       Qwen/Qwen3-32B           TP=4   bf16
    qwq-32b         Qwen/QwQ-32B             TP=4   bf16   (reasoning-tuned)
    qwen3-30b-a3b   Qwen/Qwen3-30B-A3B       TP=4   bf16   (MoE, 3B active)

Benchmarks
----------
    aime24          60-problem-class competition math (long traces)
    aime25          the 2025 set
    math500         MATH-500, mixed difficulty (difficulty gradient)
    gpqa_diamond    PhD-level science MCQ (non-math reasoning; gated dataset)
    baseline        non-reasoning open-ended prompts (write a poem/story/letter),
                    generated in non-thinking mode (enable_thinking=False): a
                    control with no gold answer whose entropy-vs-time curve shows
                    ordinary generation without a reasoning trace. (QwQ always
                    reasons and cannot disable thinking, so its baseline still
                    contains a think block.)

Caching
-------
Every (model, benchmark) unit writes its per-trace entropy arrays and correctness
flags to benchmarks/cache/entropy/<model>__<benchmark>.npz (the "benchmark
folder"), so plotting is decoupled from the (expensive) generation and can be
rerun freely. All figures are written to benchmarks/figures/. A per-(model,
benchmark) summary table -- accuracy, mean reasoning (chain-of-thought) length
and mean total generation length, ready to drop into a paper table -- is written
to benchmarks/results/summary.csv (and .json). Runs are seeded (--seed) for
reproducibility.

Re-running is incremental: a unit is loaded straight from cache when a cached
file exists whose generation parameters (n_samples, temperature, max_tokens,
seed, ...) match the current ones, and a model is not even loaded onto the GPU
if all of its requested benchmarks are already cached. A unit is (re)generated
only when it is missing, when those parameters differ from the cache, or under
--force. (Plot-only options such as bin width and axis caps are applied at plot
time and never invalidate a cache.)

Figures (each size (3, 2.5), pdf+png in figures/, style matching
../synthetic_experiment)
    entropy_vs_time__<benchmark>            models overlaid, all traces pooled
    entropy_correct_vs_incorrect__<model>__<benchmark>   split by correctness
    entropy_by_benchmark__<model>           benchmarks overlaid for one model,
                                            to compare reasoning vs the baseline

Usage
-----
    # generate + plot everything (run on the GPU box)
    python entropy_vs_time.py

    # pin specific GPUs (the 32B models shard across 4) and fix the seed
    python entropy_vs_time.py --models qwen3-32b --devices cuda:0 cuda:1 cuda:2 cuda:3 --seed 0

    # a quick smoke test: one small model, one benchmark, few problems
    python entropy_vs_time.py --models qwen3-8b --benchmarks math500 --limit 16 --devices cuda:0

    # re-draw figures from cache without any GPU work (works on a laptop)
    python entropy_vs_time.py --plot-only

Requirements (on the GPU box; not in the project conda env)
    pip install vllm datasets transformers
    # optional, for stricter math grading: pip install math-verify
"""

from __future__ import annotations

import os
import re
import json
import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
CACHE_DIR = os.path.join(HERE, 'cache', 'entropy')
RESULTS_DIR = os.path.join(HERE, 'results')


# ----------------------------------------------------------------------------
# plotting style (mirrors ../synthetic_experiment/run_experiments.py so the two
# sets of figures look consistent: size (3, 2.5), 14 pt axis labels, 12 pt ticks,
# no titles, shaded standard-error bands, top/right spines removed)
# ----------------------------------------------------------------------------

CURVE_COLORS = ['#2a78d6', '#008300', '#e87ba4', '#eda100', '#7b3fbf', '#c1443c']  # blue, green, magenta, gold, purple, red
CURVE_MARKERS = ['o', 's', '^', 'D', 'v', 'P']
CORRECT_COLOR = '#008300'    # green
INCORRECT_COLOR = '#c1443c'  # red
LABEL_FS = 14
TICK_FS = 12
LEGEND_FS = 7


def _style_axis(ax):
	"""
	Apply the shared axis cosmetics used throughout the project's figures: the
	tick label size and removal of the top and right spines.

	Params
	------
	ax : matplotlib.axes.Axes
		The axis to style, modified in place.
	"""
	ax.tick_params(labelsize=TICK_FS)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)


def _legend(ax, handles, loc='upper right'):
	"""
	Draw the compact, white-framed legend used across the project's figures.

	Params
	------
	ax : matplotlib.axes.Axes
		Axis to attach the legend to.
	handles : list[matplotlib.artist.Artist]
		Legend proxy handles, in the order to display them.
	loc : str
		Matplotlib legend location string.

	Returns
	-------
	leg : matplotlib.legend.Legend
		The created legend (already attached to `ax`).
	"""
	leg = ax.legend(handles=handles, fontsize=LEGEND_FS, frameon=True, loc=loc,
					borderaxespad=0.15, handlelength=1.6, labelspacing=0.25,
					handletextpad=0.5)
	leg.set_zorder(20)
	frame = leg.get_frame()
	frame.set_edgecolor('0.7')
	frame.set_facecolor('white')
	frame.set_alpha(1.0)
	frame.set_linewidth(0.7)
	return leg


def _save(fig, name):
	"""
	Save a figure to figures/<name> as both a vector pdf and a 300-dpi png, with
	tight bounding boxes, then close it.

	Params
	------
	fig : matplotlib.figure.Figure
		Figure to write out.
	name : str
		Basename (without extension) under the figures/ directory.

	Returns
	-------
	path : str
		The path stem (without extension) that was written.
	"""
	os.makedirs(FIG_DIR, exist_ok=True)
	path = os.path.join(FIG_DIR, name)
	fig.savefig(path + '.pdf', bbox_inches='tight')
	fig.savefig(path + '.png', bbox_inches='tight', dpi=300)
	plt.close(fig)
	return path


# ----------------------------------------------------------------------------
# model and benchmark specifications
# ----------------------------------------------------------------------------

@dataclass
class ModelSpec:
	"""
	The static configuration needed to serve one local model with vLLM.

	Params
	------
	key : str
		Short CLI-facing identifier (e.g. 'qwen3-8b').
	hf_id : str
		Hugging Face repository id passed to vLLM.
	tensor_parallel_size : int
		Number of GPUs to shard the model across (TP). Chosen so the bf16
		weights plus KV cache fit within the 24 GB-per-GPU budget.
	thinking_flag : bool
		Whether to pass `enable_thinking=True` to the chat template. True for the
		Qwen3 family (dense and MoE); False for QwQ-32B, which always reasons and
		does not accept the flag.
	max_model_len : int
		Context length (prompt + generation) to allocate KV cache for.
	"""
	key: str
	hf_id: str
	tensor_parallel_size: int
	thinking_flag: bool = True
	max_model_len: int = 20480


LOCAL_MODELS = {
	'qwen3-8b':      ModelSpec('qwen3-8b',      'Qwen/Qwen3-8B',      tensor_parallel_size=1),
	'qwen3-14b':     ModelSpec('qwen3-14b',     'Qwen/Qwen3-14B',     tensor_parallel_size=2),
	'qwen3-32b':     ModelSpec('qwen3-32b',     'Qwen/Qwen3-32B',     tensor_parallel_size=4),
	'qwq-32b':       ModelSpec('qwq-32b',       'Qwen/QwQ-32B',       tensor_parallel_size=4, thinking_flag=False),
	'qwen3-30b-a3b': ModelSpec('qwen3-30b-a3b', 'Qwen/Qwen3-30B-A3B', tensor_parallel_size=4),
}

# default set kept modest so a full run is affordable; add the 32B models on the CLI
DEFAULT_MODELS = ['qwen3-8b', 'qwen3-14b', 'qwen3-32b']


@dataclass
class BenchmarkSpec:
	"""
	How to load one benchmark and how to grade answers on it.

	Params
	------
	key : str
		Short CLI-facing identifier (e.g. 'math500').
	hf_id : str
		Hugging Face dataset repository id.
	kind : str
		'math' (free-form answer graded by normalised \\boxed{} match), 'mcq'
		(four-way multiple choice graded by final letter), or 'open' (open-ended
		baseline with no gold answer -- not graded, and loaded from a built-in
		prompt list rather than the Hub, so `hf_id`/`split`/`config` are ignored).
	split : str
		Dataset split to read.
	config : str | None
		Optional dataset configuration name (used by GPQA).
	thinking : bool | None
		Per-benchmark override for the chat template's thinking mode. None uses
		the model's default (thinking on for Qwen3); False forces a non-thinking
		template (used by the baseline control); True forces thinking on. Ignored
		for models whose template does not accept the flag (e.g. QwQ).
	question_keys : tuple[str, ...]
		Candidate column names for the problem statement (first present wins),
		tolerant of the differing capitalisation across these datasets.
	answer_keys : tuple[str, ...]
		Candidate column names for the gold answer.
	"""
	key: str
	hf_id: str
	kind: str
	split: str = 'test'
	config: str | None = None
	thinking: bool | None = None
	question_keys: tuple = ('problem', 'Problem', 'question', 'Question')
	answer_keys: tuple = ('answer', 'Answer', 'solution', 'Solution')


BENCHMARKS = {
	# NOTE: dataset ids/fields drift on the Hub; verify these resolve for your
	# `datasets` version and swap in a mirror if a load fails. GPQA is gated --
	# accept its terms and `huggingface-cli login` before using gpqa_diamond.
	'aime24':       BenchmarkSpec('aime24',   'Maxwell-Jia/AIME_2024',   kind='math', split='train'),
	'aime25':       BenchmarkSpec('aime25',   'yentinglin/aime_2025',    kind='math', split='train'),
	'math500':      BenchmarkSpec('math500',  'HuggingFaceH4/MATH-500',  kind='math', split='test'),
	'gpqa_diamond': BenchmarkSpec('gpqa_diamond', 'Idavidrein/gpqa',     kind='mcq',  split='train',
								   config='gpqa_diamond'),
	'baseline':     BenchmarkSpec('baseline',  '',                       kind='open',
								   thinking=False),
}

DEFAULT_BENCHMARKS = ['aime24', 'aime25', 'math500', 'baseline']

MCQ_LETTERS = ['A', 'B', 'C', 'D']
MATH_INSTRUCTION = 'Please reason step by step, and put your final answer within \\boxed{}.'
MCQ_INSTRUCTION = ('Please reason step by step, then give your final answer as the '
				   'single letter of the correct option within \\boxed{}.')

# Non-reasoning control prompts for the 'baseline' benchmark: open-ended creative
# and descriptive tasks with no correct answer, to contrast the entropy dynamics
# of free generation against those of the reasoning benchmarks.
BASELINE_PROMPTS = [
	'Write a poem about the ocean at dawn.',
	'Write a short poem about an old wooden chair.',
	'Write a haiku about falling snow.',
	'Write a limerick about a forgetful cat.',
	'Write a short bedtime story about a lighthouse keeper.',
	'Write the opening paragraph of a mystery novel set on a train.',
	'Write a short story about a robot who discovers gardening.',
	'Describe a bustling morning market using vivid sensory detail.',
	'Describe the feeling of walking into a warm house on a cold night.',
	'Describe an imaginary city floating among the clouds.',
	'Write a friendly letter to a pen pal who lives on the coast.',
	'Write a thank-you note to a teacher who inspired you.',
	'Write a postcard message from a seaside holiday.',
	'Write a short dialogue between the sun and the moon.',
	'Write a whimsical product description for a jar of bottled starlight.',
	'Write a toast for a friend celebrating a new job.',
	'Compose a lullaby about sleepy woodland animals.',
	'Write a short fairy tale about a kind-hearted dragon.',
	'Describe your ideal cozy reading nook.',
	'Write a playful ode to a cup of morning coffee.',
	'Write a brief travelogue entry about a quiet mountain village.',
	'Write a short monologue for a wandering street musician.',
	'Describe a garden as it changes through the four seasons.',
	'Write a gentle poem about growing older.',
]


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------

def _first_key(row, candidates):
	"""
	Return the value of the first present key from `candidates` in a dataset row.

	Params
	------
	row : dict
		A single dataset record.
	candidates : Iterable[str]
		Column names to try, in priority order.

	Returns
	-------
	value : Any
		The value under the first matching key.
	"""
	for k in candidates:
		if k in row and row[k] is not None:
			return row[k]
	raise KeyError(f'none of {tuple(candidates)} found in row with keys {list(row)}')


def load_benchmark(spec, limit=None, seed=0):
	"""
	Load a benchmark into a list of problem records with prompts and gold answers.

	Math problems are wrapped with a step-by-step \\boxed{} instruction. Multiple
	choice problems have their (correct + distractor) options shuffled with a
	fixed per-problem seed and rendered as a lettered list; the gold answer is the
	letter the correct option landed on.

	Params
	------
	spec : BenchmarkSpec
		The benchmark to load.
	limit : int | None
		If given, keep only the first `limit` problems (after loading).
	seed : int
		Base seed for the per-problem option shuffle (mcq only).

	Returns
	-------
	problems : list[dict]
		Records with keys: 'id' (int), 'question' (str, the full user turn), and
		'gold' (str, normalised for math or a letter for mcq; '' for the open
		baseline, which is not graded).
	"""
	if spec.kind == 'open':
		# open-ended control: built-in prompts, no Hub access, no gold answer
		prompts = BASELINE_PROMPTS if limit is None else BASELINE_PROMPTS[:limit]
		return [{'id': i, 'question': q, 'gold': ''} for i, q in enumerate(prompts)]

	from datasets import load_dataset

	ds = load_dataset(spec.hf_id, spec.config, split=spec.split) if spec.config \
		else load_dataset(spec.hf_id, split=spec.split)

	problems = []
	for i, row in enumerate(ds):
		if spec.kind == 'math':
			stem = str(_first_key(row, spec.question_keys)).strip()
			gold = _normalize_math_answer(str(_first_key(row, spec.answer_keys)))
			question = f'{stem}\n\n{MATH_INSTRUCTION}'
		else:  # mcq (GPQA-style: one correct + three incorrect answers)
			stem = str(_first_key(row, ('Question', 'question'))).strip()
			correct = str(_first_key(row, ('Correct Answer', 'correct_answer'))).strip()
			distractors = [str(row[k]).strip() for k in
						   ('Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3')
						   if k in row and row[k] is not None]
			options = [correct] + distractors
			rng = np.random.default_rng(seed + i)
			order = rng.permutation(len(options))
			shuffled = [options[j] for j in order]
			gold = MCQ_LETTERS[int(np.where(order == 0)[0][0])]
			rendered = '\n'.join(f'({MCQ_LETTERS[j]}) {opt}' for j, opt in enumerate(shuffled))
			question = f'{stem}\n\n{rendered}\n\n{MCQ_INSTRUCTION}'
		problems.append({'id': i, 'question': question, 'gold': gold})

	if limit is not None:
		problems = problems[:limit]
	return problems


def resolve_thinking(model_spec, benchmark_spec):
	"""
	Effective `enable_thinking` value to pass to the chat template for a
	(model, benchmark) pair.

	Params
	------
	model_spec : ModelSpec
		The model being served; `thinking_flag` is False for templates that do
		not accept the argument (e.g. QwQ, which always reasons).
	benchmark_spec : BenchmarkSpec
		The benchmark, whose `thinking` field may override the model default.

	Returns
	-------
	enable_thinking : bool | None
		True/False to force thinking on/off, or None to omit the argument (used
		when the model's template cannot toggle it). The benchmark override only
		takes effect for models that accept the flag; otherwise None is returned
		so, e.g., QwQ still reasons even on the baseline.
	"""
	if not model_spec.thinking_flag:
		return None
	return True if benchmark_spec.thinking is None else benchmark_spec.thinking


def build_prompts(tokenizer, problems, enable_thinking):
	"""
	Render each problem's user turn into a model-ready prompt string via the
	tokenizer's chat template (with the assistant generation prompt appended).

	Params
	------
	tokenizer : transformers.PreTrainedTokenizerBase
		The model's tokenizer, providing `apply_chat_template`.
	problems : list[dict]
		Records from `load_benchmark` (each with a 'question' field).
	enable_thinking : bool | None
		Value for the template's `enable_thinking` argument: True forces the
		thinking template, False forces a non-thinking (direct-answer) template
		(used by the baseline control), and None omits the argument entirely (the
		model's own default; for templates like QwQ's that lack the argument).

	Returns
	-------
	prompts : list[str]
		The formatted prompt strings, aligned with `problems`.
	"""
	kwargs = dict(tokenize=False, add_generation_prompt=True)
	if enable_thinking is not None:
		kwargs['enable_thinking'] = enable_thinking
	prompts = []
	for p in problems:
		messages = [{'role': 'user', 'content': p['question']}]
		try:
			prompts.append(tokenizer.apply_chat_template(messages, **kwargs))
		except TypeError:
			# template does not accept enable_thinking (e.g. QwQ) -- drop it
			kwargs.pop('enable_thinking', None)
			prompts.append(tokenizer.apply_chat_template(messages, **kwargs))
	return prompts


# ----------------------------------------------------------------------------
# answer grading
# ----------------------------------------------------------------------------

def _normalize_math_answer(ans):
	"""
	Canonicalise a math answer string for exact-match comparison: take the last
	\\boxed{...} payload if present, strip common LaTeX wrappers and whitespace,
	and lower-case. This is a heuristic; install `math-verify` for stricter,
	semantics-aware grading (used automatically by `grade` when available).

	Params
	------
	ans : str
		Raw answer or solution string.

	Returns
	-------
	norm : str
		The normalised answer key.
	"""
	s = str(ans)
	boxed = _extract_boxed(s)
	if boxed is not None:
		s = boxed
	s = s.strip()
	s = re.sub(r'\\left|\\right|\\!|\\,|\\;|\\ ', '', s)
	s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
	s = s.replace('\\%', '').replace('%', '')
	s = s.replace('$', '').replace(' ', '')
	s = s.rstrip('.')
	if s.startswith('{') and s.endswith('}'):
		s = s[1:-1]
	return s.lower()


def _extract_boxed(text):
	"""
	Return the contents of the last \\boxed{...} in `text`, matching balanced
	braces so nested groups (e.g. \\boxed{\\frac{1}{2}}) are captured whole.

	Params
	------
	text : str
		Text to search.

	Returns
	-------
	content : str | None
		The inner content of the final \\boxed{...}, or None if there is none.
	"""
	marker = '\\boxed'
	idx = text.rfind(marker)
	if idx == -1:
		return None
	i = idx + len(marker)
	while i < len(text) and text[i] != '{':
		i += 1
	if i >= len(text):
		return None
	depth = 0
	start = i
	for j in range(i, len(text)):
		if text[j] == '{':
			depth += 1
		elif text[j] == '}':
			depth -= 1
			if depth == 0:
				return text[start + 1:j]
	return None


def grade(kind, generated_text, gold):
	"""
	Decide whether a generated trace's final answer matches the gold answer.

	For math, uses `math_verify` when installed (semantics-aware), otherwise falls
	back to normalised \\boxed{} string match. For mcq, extracts the final answer
	letter and compares it to the gold letter.

	Params
	------
	kind : str
		'math' or 'mcq'.
	generated_text : str
		The model's full decoded output (thinking + answer).
	gold : str
		The benchmark's gold answer: a normalised string (math) or a letter (mcq).

	Returns
	-------
	correct : bool
		True iff the extracted answer matches the gold answer. Always False for
		the open baseline, whose correctness is undefined and unused.
	"""
	if kind == 'open':
		return False
	if kind == 'mcq':
		letter = _extract_answer_letter(generated_text)
		return letter is not None and letter == gold

	# math
	try:
		from math_verify import parse, verify
		pred = parse(generated_text)
		gld = parse(gold if '\\boxed' in gold else f'\\boxed{{{gold}}}')
		if pred and gld:
			return bool(verify(gld, pred))
	except Exception:
		pass  # fall through to the heuristic
	return _normalize_math_answer(generated_text) == _normalize_math_answer(gold)


def _extract_answer_letter(text):
	"""
	Pull the final multiple-choice letter (A-D) from a generation, preferring a
	\\boxed{...} payload and otherwise an "answer is (X)"-style phrase.

	Params
	------
	text : str
		The model's decoded output.

	Returns
	-------
	letter : str | None
		The upper-case option letter, or None if none could be found.
	"""
	boxed = _extract_boxed(text)
	if boxed:
		m = re.search(r'[A-D]', boxed.upper())
		if m:
			return m.group(0)
	m = re.search(r'answer\s*(?:is|:)?\s*\(?([A-D])\)?', text, re.IGNORECASE)
	if m:
		return m.group(1).upper()
	return None


# ----------------------------------------------------------------------------
# entropy computation
# ----------------------------------------------------------------------------

def token_entropy(logprob_dict, base=2.0, renormalize=False):
	"""
	Shannon entropy of one step's next-token distribution from vLLM's top-k
	logprobs.

	vLLM returns the natural-log probabilities of the top-k tokens at each step.
	The entropy over just those retained tokens,

	    H = - sum_i p_i log_base(p_i),

	is a (tight, for peaked distributions) lower bound on the true entropy because
	the unseen tail is dropped. With `renormalize=True` the retained probabilities
	are rescaled to sum to one first, giving the entropy of the top-k conditional
	distribution instead.

	Params
	------
	logprob_dict : dict[int, vllm.sequence.Logprob]
		Mapping token id -> object with a `.logprob` (natural log) attribute, as
		found in each entry of a vLLM output's `.logprobs` list.
	base : float
		Logarithm base for the entropy units (2.0 -> bits, math.e -> nats).
	renormalize : bool
		Whether to renormalise the top-k probabilities to sum to one.

	Returns
	-------
	H : float
		The (truncated) entropy of the step distribution, in units of `base`.
	"""
	lps = np.fromiter((lp.logprob for lp in logprob_dict.values()), dtype=np.float64)
	p = np.exp(lps)
	if renormalize:
		z = p.sum()
		p = p / z
		lps = np.log(p)
	H_nats = -np.sum(p * lps)
	return float(H_nats / np.log(base))


def entropy_trace(output, base=2.0, renormalize=False):
	"""
	Per-position entropy array for a single generated sequence.

	Params
	------
	output : vllm.outputs.CompletionOutput
		One completion (with `.logprobs`, a per-token list of top-k logprob dicts).
	base : float
		Entropy log base (2.0 -> bits).
	renormalize : bool
		Passed through to `token_entropy`.

	Returns
	-------
	trace : np.ndarray, shape (T,)
		Entropy H_t at each generated position t, in units of `base`.
	"""
	return np.array([token_entropy(d, base=base, renormalize=renormalize)
					 for d in output.logprobs], dtype=np.float32)


def thinking_length(tokenizer, output):
	"""
	Number of reasoning (chain-of-thought) tokens in one generated sequence.

	The thinking span is everything before the '</think>' marker, re-tokenised to
	count tokens. If the trace never closes the thought (truncated at max_tokens
	while still reasoning) the whole generation is counted; a non-thinking
	generation (no '<think>', e.g. the baseline control) counts as zero. This is
	the "reasoning length" reported in the summary table, distinct from the total
	generation length (which also includes the final answer).

	Params
	------
	tokenizer : transformers.PreTrainedTokenizerBase
		The model's tokenizer (used to count tokens in the thinking span).
	output : vllm.outputs.CompletionOutput
		One completion, with `.text` and `.token_ids`.

	Returns
	-------
	n_think : int
		The reasoning-span length in tokens.
	"""
	text = output.text
	if '</think>' in text:
		thinking = text.split('</think>', 1)[0]
		return len(tokenizer.encode(thinking, add_special_tokens=False))
	if '<think>' in text:
		return len(output.token_ids)   # ran out of budget before closing the thought
	return 0


# ----------------------------------------------------------------------------
# generation (vLLM) + caching
# ----------------------------------------------------------------------------

def cache_path(model_key, benchmark_key):
	"""
	Path of the .npz cache file for one (model, benchmark) unit.

	Params
	------
	model_key : str
		Model identifier (e.g. 'qwen3-8b').
	benchmark_key : str
		Benchmark identifier (e.g. 'math500').

	Returns
	-------
	path : str
		Absolute path under cache/entropy/.
	"""
	return os.path.join(CACHE_DIR, f'{model_key}__{benchmark_key}.npz')


def save_unit(model_key, benchmark_key, traces, correct, think_lengths, meta):
	"""
	Persist one (model, benchmark) unit's entropy traces, correctness flags and
	per-trace reasoning lengths.

	Variable-length traces are stored flat (concatenated values plus a lengths
	vector) to avoid ragged object arrays; `load_unit` reconstructs them.

	Params
	------
	model_key, benchmark_key : str
		Identifiers used to name the cache file.
	traces : list[np.ndarray]
		Per-sequence entropy arrays.
	correct : list[bool] | np.ndarray
		Correctness flag per sequence, aligned with `traces`.
	think_lengths : list[int] | np.ndarray
		Reasoning-span length (tokens) per sequence, aligned with `traces`.
	meta : dict
		Run parameters and summary stats stored as JSON for provenance.
	"""
	os.makedirs(CACHE_DIR, exist_ok=True)
	lengths = np.array([len(t) for t in traces], dtype=np.int64)
	flat = np.concatenate(traces).astype(np.float32) if traces else np.zeros(0, np.float32)
	np.savez_compressed(cache_path(model_key, benchmark_key),
						flat=flat, lengths=lengths,
						correct=np.asarray(correct, dtype=bool),
						think_lengths=np.asarray(think_lengths, dtype=np.int64),
						meta=json.dumps(meta))


def load_unit(model_key, benchmark_key):
	"""
	Load one (model, benchmark) unit from cache.

	Params
	------
	model_key, benchmark_key : str
		Identifiers naming the cache file.

	Returns
	-------
	traces : list[np.ndarray]
		Per-sequence entropy arrays.
	correct : np.ndarray of bool
		Correctness flag per sequence.
	think_lengths : np.ndarray | None
		Reasoning-span length (tokens) per sequence, or None for caches written
		before this field existed.
	meta : dict
		The stored run parameters.
	"""
	data = np.load(cache_path(model_key, benchmark_key), allow_pickle=False)
	flat, lengths = data['flat'], data['lengths']
	traces, off = [], 0
	for n in lengths:
		traces.append(flat[off:off + n])
		off += int(n)
	think = data['think_lengths'] if 'think_lengths' in data else None
	meta = json.loads(str(data['meta'])) if 'meta' in data else {}
	return traces, data['correct'], think, meta


def read_meta(model_key, benchmark_key):
	"""
	Read only the provenance `meta` dict from a unit's cache, without
	materialising the (large) entropy arrays. Used for the cheap cache-freshness
	check before deciding whether to load a model / regenerate.

	Params
	------
	model_key, benchmark_key : str
		Identifiers naming the cache file.

	Returns
	-------
	meta : dict
		The stored run parameters (empty dict if the field is absent).
	"""
	with np.load(cache_path(model_key, benchmark_key), allow_pickle=False) as data:
		return json.loads(str(data['meta'])) if 'meta' in data.files else {}


def _params_match(meta, args):
	"""
	Whether a cached unit's stored sampling parameters equal the current ones.

	Only generation-affecting parameters are compared (those in `_sampling_meta`:
	n_samples, temperature, top_p, max_tokens, logprobs_k, entropy_base,
	renormalize, limit, seed). Plot-only parameters (bin width, axis caps,
	min_sequences) are applied at aggregation time and are deliberately excluded,
	so tweaking them never invalidates a cache.

	Params
	------
	meta : dict
		The cached `meta` as written by `run_unit`.
	args : argparse.Namespace
		Current parsed options.

	Returns
	-------
	match : bool
		True iff every generation-affecting parameter matches.
	"""
	want = _sampling_meta(args)
	return all(meta.get(k) == v for k, v in want.items())


def unit_cached(model_key, benchmark_key, args):
	"""
	Whether a cached unit exists that matches the current sampling parameters, so
	it can be reused instead of regenerated.

	A file whose stored parameters differ from the current ones is treated as a
	miss (it belongs to a different configuration and would be regenerated),
	rather than being silently reused.

	Params
	------
	model_key, benchmark_key : str
		Identifiers for the unit.
	args : argparse.Namespace
		Current parsed options.

	Returns
	-------
	cached : bool
		True iff a matching cache is present and readable.
	"""
	if not os.path.exists(cache_path(model_key, benchmark_key)):
		return False
	try:
		return _params_match(read_meta(model_key, benchmark_key), args)
	except Exception:
		return False


def generate_unit(llm, model_spec, benchmark_spec, problems, args):
	"""
	Generate `n_samples` reasoning traces per problem with vLLM, compute each
	trace's entropy curve, and grade it against the gold answer.

	Params
	------
	llm : vllm.LLM
		An already-initialised engine for `model_spec` (reused across benchmarks).
	model_spec : ModelSpec
		Config for the loaded model (used for the chat template).
	benchmark_spec : BenchmarkSpec
		The benchmark being run (drives grading `kind`).
	problems : list[dict]
		Records from `load_benchmark`.
	args : argparse.Namespace
		Parsed CLI options (sampling and entropy settings).

	Returns
	-------
	traces : list[np.ndarray]
		Entropy curve per generated sequence (n_samples x len(problems) of them).
	correct : list[bool]
		Correctness flag per generated sequence, aligned with `traces`.
	think_lengths : list[int]
		Reasoning-span length (tokens) per generated sequence, aligned with
		`traces`.
	"""
	from vllm import SamplingParams

	tokenizer = llm.get_tokenizer()
	prompts = build_prompts(tokenizer, problems,
							resolve_thinking(model_spec, benchmark_spec))
	sampling = SamplingParams(
		n=args.n_samples,
		temperature=args.temperature,
		top_p=args.top_p,
		max_tokens=args.max_tokens,
		logprobs=args.logprobs_k,
		seed=args.seed,
	)
	request_outputs = llm.generate(prompts, sampling)

	gradable = benchmark_spec.kind in ('math', 'mcq')
	traces, correct, think_lengths = [], [], []
	desc = f'{model_spec.key}/{benchmark_spec.key} entropy+grade'
	for problem, req in tqdm(list(zip(problems, request_outputs)),
							 desc=desc, unit='prob'):
		for out in req.outputs:
			if not out.logprobs:
				continue
			traces.append(entropy_trace(out, base=args.entropy_base,
										renormalize=args.renormalize))
			correct.append(grade(benchmark_spec.kind, out.text, problem['gold'])
						   if gradable else False)
			think_lengths.append(thinking_length(tokenizer, out))
	return traces, correct, think_lengths


def run_unit(model_spec, benchmark_key, args, llm=None):
	"""
	Compute (or load from cache) one (model, benchmark) unit, honouring --force.

	When generation is needed and no engine is supplied, a vLLM engine is created
	for this model and torn down before returning; pass `llm` to reuse one engine
	across several benchmarks.

	Params
	------
	model_spec : ModelSpec
		The model to run.
	benchmark_key : str
		Which benchmark to run.
	args : argparse.Namespace
		Parsed CLI options.
	llm : vllm.LLM | None
		An existing engine to reuse, or None to build (and free) one here.

	Returns
	-------
	traces : list[np.ndarray]
		Per-sequence entropy curves.
	correct : np.ndarray of bool
		Per-sequence correctness flags.
	think_lengths : np.ndarray | None
		Per-sequence reasoning-span length in tokens (None only for a legacy
		cache written before this field existed).
	"""
	path = cache_path(model_spec.key, benchmark_key)
	if os.path.exists(path) and not args.force:
		traces, correct, think, meta = load_unit(model_spec.key, benchmark_key)
		if _params_match(meta, args):
			print(f'[cache] {model_spec.key}/{benchmark_key}: reused '
				  f'({len(traces)} traces)')
			return traces, correct, think
		print(f'[cache] {model_spec.key}/{benchmark_key}: parameters changed, '
			  f'regenerating')

	benchmark_spec = BENCHMARKS[benchmark_key]
	problems = load_benchmark(benchmark_spec, limit=args.limit, seed=args.seed)

	own_engine = llm is None
	if own_engine:
		llm = build_engine(model_spec, args)
	traces, correct, think = generate_unit(llm, model_spec, benchmark_spec, problems, args)
	if own_engine:
		_free_engine(llm)

	gradable = benchmark_spec.kind in ('math', 'mcq')
	total_len = [len(t) for t in traces]
	meta = dict(model=model_spec.hf_id, benchmark=benchmark_key,
				kind=benchmark_spec.kind,
				n_problems=len(problems), n_traces=len(traces),
				n_correct=int(np.sum(correct)),
				accuracy=(float(np.mean(correct)) if (gradable and traces) else None),
				mean_gen_length=(float(np.mean(total_len)) if traces else None),
				mean_think_length=(float(np.mean(think)) if think else None),
				**_sampling_meta(args))
	save_unit(model_spec.key, benchmark_key, traces, correct, think, meta)
	return traces, np.asarray(correct, dtype=bool), np.asarray(think, dtype=np.int64)


def build_engine(model_spec, args):
	"""
	Construct a vLLM engine for a model at the tensor-parallel size in its spec.

	Params
	------
	model_spec : ModelSpec
		The model to serve.
	args : argparse.Namespace
		Parsed CLI options (for gpu memory utilisation and top-k cap).

	Returns
	-------
	llm : vllm.LLM
		The initialised engine.
	"""
	from vllm import LLM
	return LLM(
		model=model_spec.hf_id,
		tensor_parallel_size=model_spec.tensor_parallel_size,
		dtype='bfloat16',
		max_model_len=model_spec.max_model_len,
		max_logprobs=args.logprobs_k,
		gpu_memory_utilization=args.gpu_mem_util,
		enforce_eager=False,
		trust_remote_code=True,
	)


def _free_engine(llm):
	"""
	Release a vLLM engine's GPU memory so the next model can be loaded.

	Params
	------
	llm : vllm.LLM
		Engine to tear down.
	"""
	import gc
	import torch
	del llm
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


def _sampling_meta(args):
	"""
	Collect the sampling/entropy parameters into a small dict for cache provenance.

	Params
	------
	args : argparse.Namespace
		Parsed CLI options.

	Returns
	-------
	meta : dict
		The subset of options that affect generated values.
	"""
	return dict(n_samples=args.n_samples, temperature=args.temperature,
				top_p=args.top_p, max_tokens=args.max_tokens,
				logprobs_k=args.logprobs_k, entropy_base=args.entropy_base,
				renormalize=args.renormalize, limit=args.limit, seed=args.seed)


# ----------------------------------------------------------------------------
# aggregation across sequences
# ----------------------------------------------------------------------------

def aggregate(traces, bin_width, max_position, min_sequences):
	"""
	Aggregate variable-length entropy curves into a mean curve with a standard
	error band, binning the position axis.

	The independent sampling unit is a full sequence, so within each position bin
	we first average each sequence's entropy over the positions it contributes,
	then take the mean and standard error of the mean *across sequences*. A bin is
	only reported if at least `min_sequences` sequences reach it, which trims the
	noisy tail where few long traces survive.

	Params
	------
	traces : list[np.ndarray]
		Per-sequence entropy curves (each shape (T_i,)).
	bin_width : int
		Number of token positions per bin (1 = no binning).
	max_position : int
		Ignore positions at or beyond this index (caps the x-axis).
	min_sequences : int
		Minimum contributing sequences for a bin to be plotted.

	Returns
	-------
	centers : np.ndarray
		Bin-center token positions.
	mean : np.ndarray
		Mean entropy per bin (across sequences).
	sem : np.ndarray
		Standard error of the mean per bin.
	count : np.ndarray
		Number of contributing sequences per bin.
	"""
	if not traces:
		empty = np.zeros(0)
		return empty, empty, empty, empty

	n_bins = int(np.ceil(max_position / bin_width))
	# per (sequence, bin) mean entropy, NaN where a sequence does not reach a bin
	per_seq_bin = np.full((len(traces), n_bins), np.nan, dtype=np.float64)
	for s, tr in enumerate(traces):
		t = tr[:max_position]
		if t.size == 0:
			continue
		pos = np.arange(t.size)
		b = pos // bin_width
		sums = np.bincount(b, weights=t, minlength=n_bins)
		counts = np.bincount(b, minlength=n_bins)
		nz = counts > 0
		per_seq_bin[s, nz] = sums[nz] / counts[nz]

	count = np.sum(~np.isnan(per_seq_bin), axis=0)
	# trailing bins that no sequence reaches are all-NaN columns; nanmean/nanstd
	# warn on those, so silence the expected warnings rather than clutter output
	import warnings
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', category=RuntimeWarning)
		mean = np.nanmean(per_seq_bin, axis=0)
		std = np.nanstd(per_seq_bin, axis=0, ddof=1)
	std = np.where(count >= 2, std, 0.0)          # SEM undefined for n<2 -> 0 band
	sem = std / np.sqrt(np.maximum(count, 1))

	keep = count >= min_sequences
	centers = (np.arange(n_bins) * bin_width + bin_width / 2.0)
	return centers[keep], mean[keep], sem[keep], count[keep]


# ----------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------

def _entropy_ylabel(base):
	"""
	Axis label for the entropy units implied by the log base.

	Params
	------
	base : float
		Entropy log base (2.0 -> bits, e -> nats).

	Returns
	-------
	label : str
		A matplotlib-ready y-axis label string.
	"""
	unit = 'bits' if abs(base - 2.0) < 1e-9 else 'nats'
	return f'Entropy ({unit})'


def plot_entropy_vs_time(units_by_model, benchmark_key, args):
	"""
	Overlay the pooled mean entropy-vs-position curve of each model for one
	benchmark, with shaded standard-error bands.

	Params
	------
	units_by_model : dict[str, tuple[list[np.ndarray], np.ndarray]]
		Maps model key -> (traces, correctness). Correctness is unused here (all
		traces are pooled).
	benchmark_key : str
		The benchmark whose figure is drawn (used in the filename).
	args : argparse.Namespace
		Parsed CLI options (aggregation and unit settings).

	Returns
	-------
	path : str | None
		The saved figure stem, or None if no model had plottable data.
	"""
	fig, ax = plt.subplots(figsize=(3, 2.5))
	handles, any_data = [], False
	for i, (model_key, (traces, _)) in enumerate(units_by_model.items()):
		centers, mean, sem, _ = aggregate(traces, args.bin_width,
										   args.max_position, args.min_sequences)
		if centers.size == 0:
			continue
		any_data = True
		col = CURVE_COLORS[i % len(CURVE_COLORS)]
		ax.plot(centers, mean, color=col, lw=1.3, zorder=3)
		ax.fill_between(centers, mean - sem, mean + sem, color=col, alpha=0.2,
						linewidth=0, zorder=2)
		handles.append(plt.Line2D([], [], color=col, lw=1.3, label=model_key))

	if not any_data:
		plt.close(fig)
		return None
	ax.set_xlabel('Token position', fontsize=LABEL_FS)
	ax.set_ylabel(_entropy_ylabel(args.entropy_base), fontsize=LABEL_FS)
	_style_axis(ax)
	_legend(ax, handles, loc='upper right')
	return _save(fig, f'entropy_vs_time__{benchmark_key}')


def plot_entropy_by_benchmark(model_key, units_by_benchmark, args):
	"""
	For one model, overlay the pooled mean entropy-vs-position curve of each
	benchmark, with shaded standard-error bands. This is the figure that places
	the non-reasoning `baseline` control directly against the reasoning
	benchmarks on a common axis.

	Params
	------
	model_key : str
		The model whose curves are drawn (used in the filename).
	units_by_benchmark : dict[str, tuple[list[np.ndarray], np.ndarray]]
		Maps benchmark key -> (traces, correctness). Correctness is unused here.
	args : argparse.Namespace
		Parsed CLI options (aggregation and unit settings).

	Returns
	-------
	path : str | None
		The saved figure stem, or None if no benchmark had plottable data.
	"""
	fig, ax = plt.subplots(figsize=(3, 2.5))
	handles, any_data = [], False
	for i, (benchmark_key, (traces, _)) in enumerate(units_by_benchmark.items()):
		centers, mean, sem, _ = aggregate(traces, args.bin_width,
										   args.max_position, args.min_sequences)
		if centers.size == 0:
			continue
		any_data = True
		col = CURVE_COLORS[i % len(CURVE_COLORS)]
		ax.plot(centers, mean, color=col, lw=1.3, zorder=3)
		ax.fill_between(centers, mean - sem, mean + sem, color=col, alpha=0.2,
						linewidth=0, zorder=2)
		handles.append(plt.Line2D([], [], color=col, lw=1.3, label=benchmark_key))

	if not any_data:
		plt.close(fig)
		return None
	ax.set_xlabel('Token position', fontsize=LABEL_FS)
	ax.set_ylabel(_entropy_ylabel(args.entropy_base), fontsize=LABEL_FS)
	_style_axis(ax)
	_legend(ax, handles, loc='upper right')
	return _save(fig, f'entropy_by_benchmark__{model_key}')


def plot_correct_vs_incorrect(model_key, benchmark_key, traces, correct, args):
	"""
	For one (model, benchmark), plot the mean entropy-vs-position curve split into
	correct (green) and incorrect (red) traces, each with a standard-error band.

	Params
	------
	model_key : str
		Model identifier (used in title-free filename).
	benchmark_key : str
		Benchmark identifier.
	traces : list[np.ndarray]
		Per-sequence entropy curves.
	correct : np.ndarray of bool
		Correctness flag per sequence, aligned with `traces`.
	args : argparse.Namespace
		Parsed CLI options.

	Returns
	-------
	path : str | None
		The saved figure stem, or None if neither subset had enough data.
	"""
	correct = np.asarray(correct, dtype=bool)
	subsets = [
		('correct', [t for t, c in zip(traces, correct) if c], CORRECT_COLOR),
		('incorrect', [t for t, c in zip(traces, correct) if not c], INCORRECT_COLOR),
	]

	fig, ax = plt.subplots(figsize=(3, 2.5))
	handles, any_data = [], False
	for label, subset, col in subsets:
		centers, mean, sem, _ = aggregate(subset, args.bin_width,
										   args.max_position, args.min_sequences)
		if centers.size == 0:
			continue
		any_data = True
		ax.plot(centers, mean, color=col, lw=1.3, zorder=3)
		ax.fill_between(centers, mean - sem, mean + sem, color=col, alpha=0.2,
						linewidth=0, zorder=2)
		handles.append(plt.Line2D([], [], color=col, lw=1.3,
								   label=f'{label} (n={len(subset)})'))

	if not any_data:
		plt.close(fig)
		return None
	ax.set_xlabel('Token position', fontsize=LABEL_FS)
	ax.set_ylabel(_entropy_ylabel(args.entropy_base), fontsize=LABEL_FS)
	_style_axis(ax)
	_legend(ax, handles, loc='upper right')
	return _save(fig, f'entropy_correct_vs_incorrect__{model_key}__{benchmark_key}')


# ----------------------------------------------------------------------------
# summary table (accuracy + reasoning length per model per benchmark)
# ----------------------------------------------------------------------------

SUMMARY_FIELDS = ['model', 'benchmark', 'kind', 'n_traces', 'n_correct', 'accuracy',
				  'mean_gen_length', 'sem_gen_length', 'mean_think_length',
				  'sem_think_length', 'median_gen_length', 'max_gen_length',
				  'frac_truncated']


def summarize_unit(model_key, benchmark_key, traces, correct, think_lengths, max_tokens):
	"""
	Compute the summary-table row for one (model, benchmark) unit: accuracy and
	reasoning-length statistics over its generated traces.

	Params
	------
	model_key, benchmark_key : str
		Identifiers for the unit.
	traces : list[np.ndarray]
		Per-sequence entropy curves; their lengths are the total generation
		lengths in tokens.
	correct : np.ndarray of bool
		Per-sequence correctness flags (unused for the open baseline).
	think_lengths : np.ndarray | None
		Per-sequence reasoning-span lengths in tokens, or None if unavailable.
	max_tokens : int
		The generation cap, used to report the fraction of truncated traces.

	Returns
	-------
	row : dict
		A mapping over `SUMMARY_FIELDS`; fields that do not apply (e.g. accuracy
		for the baseline) are None.
	"""
	kind = BENCHMARKS[benchmark_key].kind
	gradable = kind in ('math', 'mcq')
	n = len(traces)
	correct = np.asarray(correct, dtype=bool)
	gen_len = np.array([len(t) for t in traces], dtype=float) if n else np.zeros(0)
	think = (np.asarray(think_lengths, dtype=float)
			 if think_lengths is not None and len(think_lengths) == n else None)

	def _mean(a):
		return round(float(a.mean()), 1) if len(a) else None

	def _sem(a):
		return round(float(a.std(ddof=1) / np.sqrt(len(a))), 1) if len(a) > 1 else 0.0

	return {
		'model': model_key,
		'benchmark': benchmark_key,
		'kind': kind,
		'n_traces': n,
		'n_correct': (int(correct.sum()) if gradable else None),
		'accuracy': (round(float(correct.mean()), 4) if (gradable and n) else None),
		'mean_gen_length': _mean(gen_len),
		'sem_gen_length': (_sem(gen_len) if n else None),
		'mean_think_length': (_mean(think) if think is not None else None),
		'sem_think_length': (_sem(think) if think is not None else None),
		'median_gen_length': (int(np.median(gen_len)) if n else None),
		'max_gen_length': (int(gen_len.max()) if n else None),
		'frac_truncated': (round(float(np.mean(gen_len >= max_tokens)), 3) if n else None),
	}


def write_summary(rows, stem):
	"""
	Write the summary rows to `<stem>.csv` and `<stem>.json`.

	The CSV (columns = `SUMMARY_FIELDS`, one row per model x benchmark) is meant
	to be dropped straight into a paper table; the JSON preserves the same data
	for programmatic use.

	Params
	------
	rows : list[dict]
		Rows from `summarize_unit`.
	stem : str
		Output path without extension (its directory is created if needed).

	Returns
	-------
	stem : str
		The path stem written to.
	"""
	import csv
	os.makedirs(os.path.dirname(stem), exist_ok=True)
	with open(stem + '.csv', 'w', newline='') as f:
		w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
		w.writeheader()
		for r in rows:
			w.writerow({k: ('' if r.get(k) is None else r[k]) for k in SUMMARY_FIELDS})
	with open(stem + '.json', 'w') as f:
		json.dump(rows, f, indent=2)
	return stem


def print_summary(rows):
	"""
	Print a compact, aligned view of the key summary columns to stdout.

	Params
	------
	rows : list[dict]
		Rows from `summarize_unit`.
	"""
	cols = [('model', 15), ('benchmark', 14), ('n_traces', 10), ('accuracy', 10),
			('mean_think_length', 19), ('mean_gen_length', 16)]
	header = ''.join(f'{name:<{w}}' for name, w in cols)
	print(header)
	print('-' * len(header))
	for r in rows:
		line = ''
		for name, w in cols:
			v = r.get(name)
			line += f'{("" if v is None else v):<{w}}'
		print(line)


# ----------------------------------------------------------------------------
# CLI / orchestration
# ----------------------------------------------------------------------------

def configure_devices(devices):
	"""
	Restrict the process to specific CUDA devices by setting CUDA_VISIBLE_DEVICES.

	Must be called before any CUDA context is created (i.e. before `set_seed`,
	which probes `torch.cuda`, and before the vLLM engine is built), so the
	selection takes effect.

	Params
	------
	devices : list[str] | None
		Device specifiers such as 'cuda:0' (or bare indices like '0'). None (or an
		empty list) leaves the environment untouched, so all visible GPUs are used.

	Returns
	-------
	n_visible : int | None
		The number of selected devices, or None if `devices` was empty/None.
	"""
	if not devices:
		return None
	indices = []
	for d in devices:
		s = str(d).strip().lower()
		if s.startswith('cuda:'):
			s = s[len('cuda:'):]
		if not s.isdigit():
			raise ValueError(f"bad --devices entry {d!r}: expected e.g. 'cuda:0' or '0'")
		indices.append(s)
	os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(indices)
	return len(indices)


def set_seed(seed):
	"""
	Seed the Python, NumPy and Torch global RNGs so a run is reproducible.

	vLLM sampling is additionally seeded per request via `SamplingParams(seed=...)`
	and the MCQ option shuffle via a per-problem NumPy generator; this pins the
	global generators so any remaining randomness is deterministic too. Exact
	reproduction of generated text also requires the same model, vLLM version and
	device / tensor-parallel layout.

	Params
	------
	seed : int
		The master seed.
	"""
	import random
	random.seed(seed)
	np.random.seed(seed)
	try:
		import torch
		# Seed only the CPU generator here. torch.manual_seed registers CUDA
		# seeding lazily (applied when CUDA first initialises inside the vLLM
		# worker processes), so it does NOT create a CUDA context in this parent
		# process. Touching torch.cuda here (is_available / manual_seed_all)
		# would initialise CUDA in the parent and then break vLLM's forked
		# engine-core child ("Cannot re-initialize CUDA in forked subprocess").
		torch.manual_seed(seed)
	except Exception:
		pass  # torch absent (e.g. plot-only on a machine without it)


def parse_args(argv=None):
	"""
	Parse command-line options.

	Params
	------
	argv : list[str] | None
		Argument list (defaults to sys.argv).

	Returns
	-------
	args : argparse.Namespace
		The parsed options.
	"""
	p = argparse.ArgumentParser(description=__doc__,
								formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument('--models', nargs='+', default=DEFAULT_MODELS, choices=list(LOCAL_MODELS),
				   help='models to run (default: 8B/14B/32B)')
	p.add_argument('--benchmarks', nargs='+', default=DEFAULT_BENCHMARKS, choices=list(BENCHMARKS),
				   help='benchmarks to run')
	p.add_argument('--n-samples', type=int, default=8,
				   help='reasoning traces sampled per problem (for the SEM band)')
	p.add_argument('--temperature', type=float, default=0.6, help='sampling temperature')
	p.add_argument('--top-p', type=float, default=0.95, help='nucleus sampling top-p')
	p.add_argument('--max-tokens', type=int, default=8192, help='max generated tokens per trace')
	p.add_argument('--logprobs-k', type=int, default=20,
				   help='top-k logprobs kept per step for the truncated entropy')
	p.add_argument('--limit', type=int, default=None, help='cap problems per benchmark')
	p.add_argument('--bin-width', type=int, default=64, help='token positions per aggregation bin')
	p.add_argument('--max-position', type=int, default=8192, help='cap the x-axis token position')
	p.add_argument('--min-sequences', type=int, default=8,
				   help='min contributing sequences for a bin to be plotted')
	p.add_argument('--entropy-base', type=float, default=2.0, help='log base (2 -> bits, e -> nats)')
	p.add_argument('--renormalize', action='store_true',
				   help='renormalise top-k probs before computing entropy')
	p.add_argument('--gpu-mem-util', type=float, default=0.90, help='vLLM gpu_memory_utilization')
	p.add_argument('--devices', nargs='+', default=None, metavar='cuda:N',
				   help='GPUs to use, e.g. --devices cuda:0 cuda:1 cuda:2 cuda:3 '
						'(bare indices like 0 1 also accepted). Sets '
						'CUDA_VISIBLE_DEVICES; each model runs on the first '
						'tensor_parallel_size of these. Default: all visible GPUs.')
	p.add_argument('--seed', type=int, default=0,
				   help='master random seed (Python/NumPy/Torch + vLLM sampling + '
						'MCQ shuffle) so a run is reproducible on the same devices')
	p.add_argument('--force', action='store_true', help='regenerate cached units')
	p.add_argument('--plot-only', action='store_true',
				   help='only redraw figures from cache (no GPU work)')
	return p.parse_args(argv)


def main(argv=None):
	"""
	Compute the requested (model, benchmark) units (unless --plot-only), draw the
	entropy-vs-time figures and the correct-vs-incorrect figures, and write the
	accuracy / reasoning-length summary table.

	Generation loops models in the outer loop so each vLLM engine is loaded once
	and reused across all benchmarks before being freed.

	Params
	------
	argv : list[str] | None
		Argument list (defaults to sys.argv).
	"""
	args = parse_args(argv)
	# vLLM launches its engine-core in a child process; prefer 'spawn' so it is
	# unaffected by any CUDA/torch state in this parent (fork + CUDA is unsafe).
	# setdefault lets an explicit VLLM_WORKER_MULTIPROC_METHOD override this.
	os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
	# device selection must happen before any CUDA context is created
	n_visible = configure_devices(args.devices)
	set_seed(args.seed)

	# results[benchmark_key][model_key] = (traces, correct); reasoning lengths are
	# kept separately, keyed by (model, benchmark), for the summary table
	results = {b: {} for b in args.benchmarks}
	think_by_unit = {}

	if args.plot_only:
		units = [(m, b) for b in args.benchmarks for m in args.models
				 if os.path.exists(cache_path(m, b))]
		for m, b in tqdm(units, desc='loading cache', unit='unit'):
			traces, correct, think, _ = load_unit(m, b)
			results[b][m] = (traces, correct)
			think_by_unit[(m, b)] = think
	else:
		for model_key in args.models:
			spec = LOCAL_MODELS[model_key]
			if n_visible is not None and spec.tensor_parallel_size > n_visible:
				raise SystemExit(
					f"model '{model_key}' needs tensor_parallel_size="
					f"{spec.tensor_parallel_size} GPUs, but only {n_visible} were "
					f"given via --devices")
			# only spin up an engine if some benchmark for this model is missing,
			# stale (parameters changed), or forced -- otherwise everything is
			# served from cache and the model is never loaded
			to_gen = [b for b in args.benchmarks
					  if args.force or not unit_cached(model_key, b, args)]
			if to_gen:
				print(f'[run] {model_key}: generating {to_gen}; '
					  f'reusing cache for the rest')
			else:
				print(f'[run] {model_key}: all benchmarks cached, skipping model load')
			llm = build_engine(spec, args) if to_gen else None
			try:
				for b in args.benchmarks:
					traces, correct, think = run_unit(spec, b, args, llm=llm)
					results[b][model_key] = (traces, correct)
					think_by_unit[(model_key, b)] = think
			finally:
				if llm is not None:
					_free_engine(llm)

	# pooled entropy-vs-time, one figure per benchmark (models overlaid)
	for b in args.benchmarks:
		if results[b]:
			out = plot_entropy_vs_time(results[b], b, args)
			if out:
				print(f'wrote {out}.pdf/.png')

	# entropy-by-benchmark, one figure per model (benchmarks overlaid so the
	# non-reasoning baseline can be compared against the reasoning sets)
	for m in dict.fromkeys(m for b in args.benchmarks for m in results[b]):
		units_by_bench = {b: results[b][m] for b in args.benchmarks if m in results[b]}
		out = plot_entropy_by_benchmark(m, units_by_bench, args)
		if out:
			print(f'wrote {out}.pdf/.png')

	# correct-vs-incorrect, one figure per gradable (model, benchmark); the
	# open-ended baseline has no correctness, so it is skipped here
	for b in args.benchmarks:
		if BENCHMARKS[b].kind not in ('math', 'mcq'):
			continue
		for m, (traces, correct) in results[b].items():
			out = plot_correct_vs_incorrect(m, b, traces, correct, args)
			if out:
				print(f'wrote {out}.pdf/.png')

	# accuracy + reasoning-length summary table, one row per (model, benchmark)
	rows = []
	for m in args.models:
		for b in args.benchmarks:
			if m in results[b]:
				traces, correct = results[b][m]
				rows.append(summarize_unit(m, b, traces, correct,
										   think_by_unit.get((m, b)), args.max_tokens))
	if rows:
		stem = write_summary(rows, os.path.join(RESULTS_DIR, 'summary'))
		print(f'wrote {stem}.csv/.json')
		print_summary(rows)


if __name__ == '__main__':
	main()
