"""
GSM8K data for the restated-reasoning experiment: the training subset, the test
set, the two trace formats a model is fine-tuned on, and the grader.

Every GSM8K solution is a short list of natural-language steps, each carrying a
calculator annotation ``<<48/2=24>>``, followed by ``#### <answer>``:

    Natalia sold 48/2 = <<48/2=24>>24 clips in May.
    Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
    #### 72

Two trace formats are built from it:

standard    the ground-truth solution with the calculator annotations stripped
            (they are a tool-use artefact, not reasoning). The final ``####``
            line is kept, and is what the grader reads.

restated    the same step lines, verbatim, with a *ledger* of the problem's
            state written before the first step and again after every step:

                Known so far: April clips = 48; May clips = half of April.
                Natalia sold 48/2 = 24 clips in May.
                Known so far: April clips = 48; May clips = 24.
                Natalia sold 48+24 = 72 clips altogether in April and May.
                Known so far: April clips = 48; May clips = 24; total clips = 72.
                #### 72

            The ledger is the analogue of the math task's ``Current equation
            is`` report at k = 1: every quantity the problem gives, replaced by
            its value once a step computes it, plus every result computed so
            far, so that any step can be checked from the ledger above it
            rather than from the whole transcript. The ledger lines for the
            training problems were written by Claude (`traces.py`,
            ``traces_ledger.txt``); `restated_trace_rule` is a mechanical
            stand-in that quotes the problem's numeric sentences and the step
            lines instead. Either way the step lines are the ground truth, so
            the restated condition differs from the standard one by the ledger
            lines alone; `validate_restated` enforces exactly that.

A third trace source, teacher-written solutions for the distillation
condition, lives in `distill.py`; it shares the ``#### <answer>`` ending.

The grader (`is_correct`) reads the first number after the last ``####``, then
a ``\\boxed{}`` answer, then the last number in the text, so that a model that
ignores the prompt's format is scored on what it answered rather than on how
it was formatted.
"""

import os
import re
import json
import random
import hashlib
from dataclasses import dataclass, asdict

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
DATA_CACHE = os.path.join(CACHE, 'datasets')

DATASET = 'openai/gsm8k'
# The Hub commit the problems are read from, so that a later revision of the
# dataset cannot change the row order the training subset is drawn by.
DATASET_REVISION = '740312add88f781978c0658806c59bc2815b9866'
ANSWER_MARKER = '####'
LEDGER_PREFIX = 'Known so far:'

# The same prompt for every condition, base model included, so the only thing
# that differs between the models being compared is what they were tuned on.
SYSTEM_PROMPT = (
    'You solve grade-school math word problems. Reason step by step in plain '
    f'text, then give the final numeric answer on its own line after {ANSWER_MARKER}.'
)

# Bumped when the text either trace builder produces changes, so adapters built
# from the old text are re-made rather than silently reused. v2: the ledger
# format replaced per-step restatements.
TRACE_VERSION = '2'


# ----------------------------------------------------------------------------
# configs / cache keys
# ----------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Which GSM8K problems are trained and tested on."""
    n_train: int = 1500             # training problems drawn from the 7,473 train rows
    train_seed: int = 0             # seeds the draw; the subset is fixed across conditions
    n_test: int = None              # None = the whole 1,319-problem test set
    test_seed: int = 12345          # only used when n_test subsamples the test set


def _key(obj):
    """Short stable hash of a JSON-able config."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def train_spec(dcfg):
    """The training half of a DataConfig, for cache keys that must not depend on
    the test set (adapters, traces)."""
    return dict(n_train=dcfg.n_train, train_seed=dcfg.train_seed)


def test_spec(dcfg):
    """The test half of a DataConfig, for decode and eval keys."""
    return dict(n_test=dcfg.n_test, test_seed=dcfg.test_seed)


# ----------------------------------------------------------------------------
# problems (cached)
# ----------------------------------------------------------------------------

@dataclass
class Problem:
    """One GSM8K row. `answer` is the ground-truth solution verbatim, calculator
    annotations included; `idx` is the row index in its split."""
    idx: int
    question: str
    answer: str

    @property
    def gold(self):
        """The reference final answer as a plain string, e.g. '72' or '2.5'."""
        return gold_answer(self.answer)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(idx=int(d['idx']), question=d['question'], answer=d['answer'])


def load_split(split):
    """Every problem of a GSM8K split ('train' or 'test'), from the Hub."""
    from datasets import load_dataset
    ds = load_dataset(DATASET, 'main', split=split, revision=DATASET_REVISION)
    return [Problem(idx=i, question=ex['question'], answer=ex['answer'])
            for i, ex in enumerate(ds)]


def _select(n_rows, n, seed):
    """
    The first `n` rows of a seeded permutation, in row order.

    A prefix rather than an independent draw, so a larger `n` is a strict
    superset of a smaller one: the trace file written for the 1,500-problem
    subset then also covers any smaller smoke-test subset at the same seed.
    """
    order = list(range(n_rows))
    random.Random(seed).shuffle(order)
    return sorted(order[:n])


def get_problems(dcfg, split):
    """
    The `split` problem list for `dcfg`, drawn once and cached to ``cache/datasets``.

    The training subset is `_select`'ed from the train split; the test set is
    the whole test split unless `n_test` is set, in which case it is selected
    the same way.
    """
    if split == 'train':
        spec = dict(split='train', **train_spec(dcfg))
    else:
        spec = dict(split='test', **test_spec(dcfg))
    path = os.path.join(DATA_CACHE, _key(spec) + '.json')
    if os.path.exists(path):
        with open(path) as f:
            return [Problem.from_dict(d) for d in json.load(f)]

    rows = load_split(split)
    if split == 'train':
        probs = [rows[i] for i in _select(len(rows), dcfg.n_train, dcfg.train_seed)]
    elif dcfg.n_test is not None:
        probs = [rows[i] for i in _select(len(rows), dcfg.n_test, dcfg.test_seed)]
    else:
        probs = rows
    os.makedirs(DATA_CACHE, exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump([p.to_dict() for p in probs], f)
    os.replace(tmp, path)
    return probs


# ----------------------------------------------------------------------------
# parsing the ground-truth solution
# ----------------------------------------------------------------------------

# A number as it appears in prose or arithmetic: 1,000  2.5  .5  -3. The
# lookbehind stops '48-24' reading as 48 and -24, and keeps the fractional
# part of '2.5' from matching on its own.
_NUM = re.compile(r'(?<![\w.])-?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?![\w])')
_CALC = re.compile(r'<<([^<>]*)>>')

# Number words that turn up in the problems -- 'half as many' becomes '/2' in
# the solution -- so a sentence that states them counts as stating that
# number. 'half' and 'quarter' map to both the divisor and the fraction, since
# solutions write either.
_WORDS = {
    'one': (1,), 'two': (2,), 'three': (3,), 'four': (4,), 'five': (5,),
    'six': (6,), 'seven': (7,), 'eight': (8,), 'nine': (9,), 'ten': (10,),
    'eleven': (11,), 'twelve': (12,), 'twenty': (20,), 'thirty': (30,),
    'forty': (40,), 'fifty': (50,), 'hundred': (100,), 'thousand': (1000,),
    'half': (2, 0.5), 'twice': (2,), 'double': (2,), 'doubled': (2,),
    'triple': (3,), 'tripled': (3,), 'thrice': (3,), 'quarter': (4, 0.25),
    'dozen': (12,),
}


def to_number(s):
    """A number string ('1,000', '$2.50', '.5') as a float, or None."""
    try:
        return float(s.replace(',', '').replace('$', '').strip())
    except ValueError:
        return None


def numbers_in(text, words=False):
    """
    The numbers in `text`, as floats, in order of appearance.

    With `words` the number words of `_WORDS` count too, which is only wanted
    when reading the *problem* -- a solution's steps write their operands as
    digits.
    """
    out = [to_number(m.group(0)) for m in _NUM.finditer(text)]
    out = [x for x in out if x is not None]
    if words:
        for w in re.findall(r'[A-Za-z]+', text.lower()):
            out.extend(_WORDS.get(w, ()))
    return out


def gold_answer(answer):
    """The final answer of a ground-truth solution, commas removed."""
    return answer.split(ANSWER_MARKER)[-1].strip().replace(',', '')


def strip_calc(text):
    """Remove the ``<<expr=result>>`` calculator annotations."""
    return _CALC.sub('', text)


def parse_steps(answer):
    """
    The reasoning lines of a ground-truth solution, annotations still attached.

    Returns
    -------
    list[str]
        The non-empty lines before ``####``, stripped.
    """
    body = answer.split(ANSWER_MARKER)[0]
    return [ln.strip() for ln in body.split('\n') if ln.strip()]


def clean_step(raw):
    """A step with its annotations removed and whitespace normalised. This is
    the line as it appears in *both* trace formats, so they differ only by the
    ledger lines."""
    return re.sub(r'[ \t]+', ' ', strip_calc(raw)).strip()


def steps_of(problem):
    """The ground-truth step lines as both trace formats write them."""
    return [clean_step(s) for s in parse_steps(problem.answer)]


def step_numbers(raw):
    """
    What a step consumes and what it produces.

    From the calculator annotations when the step has any: the operands are the
    numbers left of ``=`` and the result is the number right of it. A step with
    no annotation ("So she has 5 left.") is read from its prose instead -- every
    number is treated as consumed and the last one as produced.

    Returns
    -------
    (set[float], set[float])
        ``(used, produced)``
    """
    used, produced = set(), set()
    for calc in _CALC.findall(raw):
        expr, _, result = calc.rpartition('=')
        used.update(numbers_in(expr))
        r = to_number(result)
        if r is not None:
            produced.add(r)
    if not used and not produced:
        nums = numbers_in(strip_calc(raw))
        used.update(nums)
        if nums:
            produced.add(nums[-1])
    return used, produced


# ----------------------------------------------------------------------------
# the two trace formats
# ----------------------------------------------------------------------------

def standard_trace(problem):
    """The ground-truth solution with annotations stripped; the standard condition."""
    return '\n'.join(steps_of(problem) + [f'{ANSWER_MARKER} {problem.gold}'])


def ledger_line(items):
    """
    A ledger line from its items -- either one string of ``; ``-separated items
    or a list of them. Normalised to ``Known so far: a; b; c.`` so that every
    author writes the ledger the same way.
    """
    if not isinstance(items, str):
        items = '; '.join(x.strip().rstrip('.') for x in items if x.strip())
    items = ' '.join(items.split()).strip()
    if items.endswith('.'):
        items = items[:-1].rstrip()
    return f'{LEDGER_PREFIX} {items or "nothing yet"}.'


def assemble_ledger_trace(problem, ledgers):
    """
    The restated trace of `problem` from its ledgers.

    `ledgers` holds one entry per state: the state before the first step, then
    the state after each step, so ``len(ledgers) == n_steps + 1``. The step
    lines are the ground truth (`steps_of`), so whoever writes the ledgers
    cannot alter the reasoning; only the state reports are theirs.
    """
    steps = steps_of(problem)
    if len(ledgers) != len(steps) + 1:
        raise ValueError(f'train row {problem.idx}: {len(ledgers)} ledger lines for '
                         f'{len(steps)} steps (need {len(steps) + 1})')
    lines = []
    for k, step in enumerate(steps):
        lines.append(ledger_line(ledgers[k]))
        lines.append(step)
    lines.append(ledger_line(ledgers[-1]))
    lines.append(f'{ANSWER_MARKER} {problem.gold}')
    return '\n'.join(lines)


# Abbreviations whose period does not end a sentence. Checked after the split
# rather than in the pattern because Python lookbehinds must be fixed-width.
_ABBREV = ('Mr.', 'Mrs.', 'Ms.', 'Dr.', 'St.', 'Jr.', 'Sr.', 'vs.', 'etc.',
           'No.', 'Mt.', 'a.m.', 'p.m.')


def split_sentences(text):
    """The sentences of a problem statement, whitespace-normalised."""
    text = ' '.join(text.split())
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9$"\'(])', text)
    out = []
    for p in parts:
        if out and out[-1].endswith(_ABBREV):
            out[-1] += ' ' + p            # the split fell after an abbreviation
        else:
            out.append(p)
    return out


def restated_trace_rule(problem):
    """
    A ledger trace built mechanically from the ground truth.

    The givens are the problem's sentences that state a number (number words
    included), quoted verbatim; each step's result is the step line itself,
    appended to the ledger after it. Nothing is ever replaced or dropped, so the
    ledger grows by one item per step -- a hand-written ledger instead replaces
    'half of April' by '24' once it is computed. It is the offline control for
    the *wording* of the hand-written ledgers: same structure, same step lines,
    no authoring.
    """
    givens = [s.rstrip('.!?') for s in split_sentences(problem.question)
              if numbers_in(s, words=True)]
    steps = [s.rstrip('.') for s in steps_of(problem)]
    ledgers = [givens + steps[:k] for k in range(len(steps) + 1)]
    return assemble_ledger_trace(problem, ledgers)


def _norm(s):
    return ' '.join(s.split()).strip()


def validate_restated(text, problem, max_ratio=None):
    """
    Is a restated trace the ground-truth reasoning plus ledgers, and nothing else?

    Checks, in order: the last line is ``#### <gold>``; no calculator
    annotations; the body alternates ledger and step lines, opening and closing
    with a ledger, so there are ``n_steps + 1`` non-empty ledgers; every step
    line equals the ground-truth step (whitespace-normalised) -- the restated
    condition must differ from the standard one by the ledger lines alone; and,
    when `max_ratio` is set, the text is at most that many times the standard
    trace.

    Returns
    -------
    (bool, str)
        ``(ok, reason)``; `reason` is '' when ok.
    """
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if not lines or not lines[-1].startswith(ANSWER_MARKER):
        return False, f'last line is not a {ANSWER_MARKER} line'
    pred = extract_answer(lines[-1])
    if pred is None or not _same_number(pred, problem.gold):
        return False, f'final answer {pred!r} != gold {problem.gold!r}'
    if '<<' in text or '>>' in text:
        return False, 'calculator annotation present'
    steps = steps_of(problem)
    body = lines[:-1]
    if len(body) != 2 * len(steps) + 1:
        return False, (f'{len(body)} body lines for {len(steps)} steps '
                       f'(need {2 * len(steps) + 1}: ledger, step, ..., ledger)')
    for i, line in enumerate(body):
        if i % 2 == 0:
            if not line.startswith(LEDGER_PREFIX):
                return False, f'line {i + 1} should be a ledger: {line[:60]!r}'
            if len(line) <= len(LEDGER_PREFIX) + 1:
                return False, f'empty ledger at line {i + 1}'
        elif _norm(line) != _norm(steps[i // 2]):
            return False, (f'step {i // 2 + 1} differs from the ground truth: '
                           f'{line[:60]!r} vs {steps[i // 2][:60]!r}')
    std_len = len(standard_trace(problem))
    if max_ratio is not None and len(text) > max_ratio * std_len + 400:
        return False, f'too long ({len(text)} chars vs {std_len} standard)'
    return True, ''


# Numbers a ledger may legitimately introduce without the problem or a step
# stating them: unit conversions and the odd 'per one'.
_LEDGER_CONSTANTS = {0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 24, 30, 52, 60, 100, 365, 1000}


def unknown_ledger_numbers(text, problem):
    """
    Numbers in the ledger lines that neither the problem (number words
    included), any ground-truth step, nor `_LEDGER_CONSTANTS` account for --
    the typo check for hand-written ledgers. Returns the offenders in order.
    """
    known = set(numbers_in(problem.question, words=True)) | _LEDGER_CONSTANTS
    for raw in parse_steps(problem.answer):
        known |= set(numbers_in(raw))
    out = []
    for line in text.split('\n'):
        if line.startswith(LEDGER_PREFIX):
            for x in numbers_in(line):
                if not any(abs(x - k) < 1e-6 for k in known):
                    out.append(x)
    return out


# ----------------------------------------------------------------------------
# grading a completion
# ----------------------------------------------------------------------------

_BOXED = re.compile(r'\\boxed\{([^{}]*)\}')


def extract_answer(text):
    """
    The number a completion claims as its answer, as a string, or None.

    The first number after the last ``####`` marker. Without a marker, the
    number inside the last ``\\boxed{}`` (the format teacher models and the
    untuned base model favour), and failing that the last number in the text,
    so a model that answers in another format is still scored on its answer.
    """
    if ANSWER_MARKER in text:
        tail = text.rsplit(ANSWER_MARKER, 1)[1]
        m = _NUM.search(tail)
        if m:
            return m.group(0)
    boxed = _BOXED.findall(text)
    if boxed:
        nums = _NUM.findall(boxed[-1])
        if nums:
            return nums[-1]
    nums = _NUM.findall(text)
    return nums[-1] if nums else None


def _same_number(a, b):
    x, y = to_number(a), to_number(b)
    if x is None or y is None:
        return a.strip() == b.strip()
    return abs(x - y) < 1e-6


def is_correct(text, gold):
    """True if the completion's answer equals `gold` numerically."""
    pred = extract_answer(text)
    return pred is not None and _same_number(pred, gold)
