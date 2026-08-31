"""
Equation-solving task: single-variable equations with step-by-step reasoning
traces, emitted at a controllable state interval k.

Set-up
------
Each problem is a single-variable equation in ``x`` whose solutions are all
integers (one answer, or several comma-separated). A reasoning trace is a
sequence of *operations* -- solution-preserving algebraic moves such as
``multiply both sides by 3.`` -- ending with the answer after a ``####`` marker.

Every ``k`` operations the trace also emits the *current state* (the equation as
it now stands), exactly as the navigation experiment re-emits the current cell:

    standard  (interval=None)
        divide both sides by 2. subtract 7 from both sides. divide both sides by 3. #### 1

    k = 1     (interval=1)
        divide both sides by 2. Current equation is 3*x + 7 = 10.
        subtract 7 from both sides. Current equation is 3*x = 3.
        divide both sides by 3. Current equation is x = 1. #### 1

Small k puts the current state in the model's local context, so each next step
depends only on the last few tokens; large k (and standard) forces the model to
carry the state implicitly across the whole trace.

How problems are built: nested parentheses
------------------------------------------
Trace length is the experiment's main knob, so problems are generated as an
*onion*: a core (``x``, or an expanded quadratic/cubic) wrapped in alternating
``k*(...)`` and ``... + c`` layers, e.g.

    2*(3*x + 7) - 5 = 17

Each layer costs exactly one solving operation -- peel the outermost one, undo
it on both sides -- so nesting depth sets the number of steps directly. The
equation is kept as an explicit tree rather than a sympy expression because
sympy automatically distributes a number over a sum (``3*(x + 7)`` becomes
``3*x + 21``), which would collapse every onion into at most ~3 operations.
Sympy is still used to render polynomial cores, to factor, and to verify.

Why the operations are solution-preserving
------------------------------------------
Every operation used here (multiply/divide both sides by a nonzero constant,
add/subtract the same quantity from both sides, factor) leaves the solution set
unchanged. So a *randomly injected* operation is a detour, not a corruption of
the answer: the equation changes but the target stays the same and a model that
actually tracks the algebraic state can recover. This mirrors the navigation
task, where a random injected move changes the cell but leaves the goal
reachable.
"""

import re
import random

import sympy as sp
from tqdm.auto import tqdm

X = sp.Symbol('x')

# Bump whenever the problem generator changes in a way that alters the data it
# produces. It is part of every cache key (datasets, adapters, evals), so stale
# caches from an older generator can never be silently reused.
GEN_VERSION = '5'

ANSWER_MARKER = '####'
STATE_PREFIX = 'Current equation is'

# Condition sentinel: emit no reasoning at all, just the answer. This is the
# task floor -- how much of the problem the model can do without any serial
# reasoning -- and it is what standard's accuracy under heavy injection has to be
# read against. It is not a value of k, so it never appears on the k axis.
NO_COT = 'nocot'

# Appended to every question so each prompt states the output convention itself,
# rather than relying only on the system turn.
ANSWER_INSTRUCTION = f'Output your answer after {ANSWER_MARKER}.'


# ----------------------------------------------------------------------------
# the equation tree
# ----------------------------------------------------------------------------
#
# A left-hand side is one of:
#   ('x',)                 the bare variable
#   ('poly', (r1,...), off)  the expanded monic polynomial with these integer
#                            roots, plus a constant offset
#   ('prod', (r1, r2), (s1, s2))  the *unexpanded* product (x-r1)*(x-r2), whose
#                            equation solves to roots s1, s2 -- so it must be
#                            expanded and re-factored rather than read off
#   ('mul', c, inner)      c * (inner)
#   ('div', c, inner)      (inner) / c
#   ('add', c, inner)      inner + c
#   ('addx', m, inner)     inner + m*x -- the matching m*x also sits on the value
#                          side, so cancelling it is one operation on both sides
#
# The right-hand side is a rational `v` (division layers can make it fractional
# even though every answer is an integer), plus an optional `m*x` term carried
# separately (see `xshift`). Peeling the outermost layer of the left-hand side is
# one solving operation.

def _poly_expr(roots, offset=0):
    """The expanded monic polynomial with the given integer roots, plus `offset`."""
    e = sp.Integer(1)
    for r in roots:
        e *= (X - r)
    return sp.expand(e + offset)


def _prod_expr(roots):
    """The unexpanded product ``(x - r1)*(x - r2)``."""
    e = sp.Integer(1)
    for r in roots:
        e *= (X - r)
    return e


def _needs_parens(node):
    """True if `node` renders as a sum and so must be bracketed under a product."""
    return node[0] in ('add', 'addx', 'poly')


def render_node(node):
    """Render a left-hand-side tree, e.g. ``2*(3*x + 7) - 5``."""
    kind = node[0]
    if kind == 'x':
        return 'x'
    if kind == 'poly':
        return sp.sstr(_poly_expr(node[1], node[2]))
    if kind == 'prod':
        return sp.sstr(_prod_expr(node[1]))
    if kind == 'div':
        _, c, inner = node
        body = render_node(inner)
        return f'({body})/{c}' if _needs_parens(inner) or inner[0] == 'prod' else f'{body}/{c}'
    if kind == 'mul':
        _, c, inner = node
        body = render_node(inner)
        if not _needs_parens(inner):
            return f'-{body}' if c == -1 else f'{c}*{body}'
        return f'-({body})' if c == -1 else f'{c}*({body})'
    if kind == 'add':
        _, c, inner = node
        body = render_node(inner)
        return f'{body} + {c}' if c > 0 else f'{body} - {-c}'
    if kind == 'addx':
        _, m, inner = node
        body = render_node(inner)
        term = 'x' if abs(m) == 1 else f'{abs(m)}*x'
        return f'{body} + {term}' if m > 0 else f'{body} - {term}'
    raise ValueError(f'unknown node {kind}')


def node_expr(node):
    """The sympy expression a tree denotes (used for verification / solving)."""
    kind = node[0]
    if kind == 'x':
        return X
    if kind == 'poly':
        return _poly_expr(node[1], node[2])
    if kind == 'prod':
        return sp.expand(_prod_expr(node[1]))
    if kind == 'div':
        return sp.expand(node_expr(node[2]) / node[1])
    if kind == 'mul':
        return sp.expand(node[1] * node_expr(node[2]))
    if kind == 'add':
        return sp.expand(node_expr(node[2]) + node[1])
    if kind == 'addx':
        return sp.expand(node_expr(node[2]) + node[1] * X)
    raise ValueError(f'unknown node {kind}')


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------

def _render_rhs(m, v):
    """Render the side ``m*x + v`` (m is usually 0; v may be a fraction)."""
    if m == 0:
        return sp.sstr(v)
    xt = 'x' if m == 1 else ('-x' if m == -1 else f'{m}*x')
    if v == 0:
        return xt
    return f'{xt} + {sp.sstr(v)}' if v > 0 else f'{xt} - {sp.sstr(-v)}'


def render_state(node, m, v, flipped=False):
    """
    Render a full equation state, e.g. ``2*(3*x + 7) - 5 = 4*x + 12``.

    `flipped` writes the value side first (``12 = 2*(3*x + 7)``), which is only a
    surface change -- the same operations solve it -- but keeps the model from
    assuming the unknown always sits on the left.
    """
    lhs, rhs = render_node(node), _render_rhs(m, v)
    return f'{rhs} = {lhs}' if flipped else f'{lhs} = {rhs}'


def render_answers(answers):
    """The ``#### a, b`` tail; answers are sorted and comma-separated."""
    return f'{ANSWER_MARKER} ' + ', '.join(str(a) for a in answers)


def _fmt_const_move(c):
    """Text for cancelling a constant ``c`` from both sides."""
    return f'subtract {c} from both sides.' if c > 0 else f'add {-c} to both sides.'


def _fmt_x_move(c):
    """Text for cancelling ``c*x`` from both sides."""
    term = 'x' if abs(c) == 1 else f'{abs(c)}*x'
    return f'subtract {term} from both sides.' if c > 0 else f'add {term} to both sides.'


# ----------------------------------------------------------------------------
# the canonical solver: peel the onion, then finish the core
# ----------------------------------------------------------------------------

def solve_steps(node, m, v, flipped=False):
    """
    The canonical operation sequence solving ``node = m*x + v``.

    Peels the left-hand side one layer at a time (each layer is one operation),
    then finishes according to the core: a bare ``x`` is already solved; an
    expanded polynomial has its constant moved across and is factored; an
    unexpanded product must first be expanded, since shifting it by a constant
    changes which factors it has.

    Params
    ------
    node : tuple
        The left-hand-side tree.
    m : int
        Coefficient of the ``x`` term carried on the value side.
    v : sympy.Rational
        The constant on the value side.
    flipped : bool
        Render states with the value side written first.

    Returns
    -------
    ops : list[str]
        Operation sentences.
    states : list[str]
        Rendered equation *after* each operation (same length as `ops`).
    answers : tuple[int, ...]
        Sorted integer solutions.
    """
    ops, states = [], []
    v = sp.Rational(v)

    def step(text, node_, m_, v_):
        ops.append(text)
        states.append(render_state(node_, m_, v_, flipped))

    # -- collect the x term that sits on both sides --------------------------
    if m != 0:
        assert node[0] == 'addx' and node[1] == m, \
            'an x term on the value side must be matched by one on the other side'
        node = node[2]                                # cancels from both sides
        step(_fmt_x_move(m), node, 0, v)
        m = 0

    # -- peel the wrapping layers --------------------------------------------
    while node[0] in ('mul', 'div', 'add'):
        kind, c, inner = node
        node = inner
        if kind == 'add':
            v -= c
            step(_fmt_const_move(c), node, 0, v)
        elif kind == 'mul':
            v /= c
            step(f'divide both sides by {c}.', node, 0, v)
        else:                                        # 'div'
            v *= c
            step(f'multiply both sides by {c}.', node, 0, v)

    # -- finish the core -----------------------------------------------------
    if node[0] == 'x':
        answers = (int(v),)
    elif node[0] == 'prod':
        shown, targets = node[1], node[2]
        expanded = ('poly', shown, 0)                # same polynomial, written out
        step('expand the left hand side.', expanded, 0, v)
        if v != 0:                                   # make the value side zero
            step(_fmt_const_move(v), ('poly', targets, 0), 0, 0)
        ops.append('factor the left hand side.')
        states.append(render_state_factored(targets, flipped))
        answers = tuple(sorted(set(int(r) for r in targets)))
    else:
        roots, offset = node[1], node[2]
        assert offset == v, 'core offset is built to equal the peeled value'
        if v != 0:                                   # make the value side zero
            step(_fmt_const_move(v), ('poly', roots, 0), 0, 0)
        ops.append('factor the left hand side.')
        states.append(render_state_factored(roots, flipped))
        answers = tuple(sorted(set(int(r) for r in roots)))

    return ops, states, answers


def render_state_factored(roots, flipped=False):
    """Render the factored state ``(x - r1)*(x - r2) = 0``."""
    lhs = sp.sstr(sp.factor(_poly_expr(roots)))
    return f'0 = {lhs}' if flipped else f'{lhs} = 0'


# ----------------------------------------------------------------------------
# problems
# ----------------------------------------------------------------------------

class Problem:
    """
    One equation-solving problem with its reference solution.

    Attributes
    ----------
    question : str
        The user-side prompt (either ``Solve for x: ...`` or a word problem).
    equation : str
        The rendered equation, for reference and de-duplication.
    answers : tuple[int, ...]
        Sorted integer solutions.
    ops : list[str]
        Canonical operation sentences.
    states : list[str]
        Rendered equation after each operation.
    family : str
        Which generator produced it, e.g. ``linear`` / ``product`` / ``word``.
    n_ops : int
        Number of operations: this problem's difficulty.
    """

    __slots__ = ('question', 'equation', 'answers', 'ops', 'states', 'family')

    def __init__(self, question, equation, answers, ops, states, family):
        self.question = question
        self.equation = equation
        self.answers = answers
        self.ops = ops
        self.states = states
        self.family = family

    @property
    def n_ops(self):
        return len(self.ops)

    def to_dict(self):
        return dict(question=self.question, equation=self.equation,
                    answers=list(self.answers), ops=self.ops, states=self.states,
                    family=self.family)

    @classmethod
    def from_dict(cls, d):
        return cls(d['question'], d['equation'], tuple(d['answers']),
                   d['ops'], d['states'], d['family'])


def _nz(rng, lo, hi):
    """A random nonzero int in [lo, hi]."""
    v = 0
    while v == 0:
        v = rng.randint(lo, hi)
    return v


def _story(chain, value):
    """
    A word-problem phrasing of a wrap `chain` applied to an unknown number.

    The onion is literally a recipe -- multiply by 3, add 7, divide by 2 -- so a
    linear problem reads off directly as a story, giving the same algebra a very
    different surface form.
    """
    acts = []
    for kind, c in chain:
        if kind == 'mul':
            acts.append(f'multiply it by {c}')
        elif kind == 'div':
            acts.append(f'divide it by {c}')
        else:
            acts.append(f'add {c}' if c > 0 else f'subtract {-c}')
    if not acts:
        recipe = 'keep it as it is'
    elif len(acts) == 1:
        recipe = acts[0]
    else:
        recipe = acts[0] + ', then ' + ', then '.join(acts[1:])
    return (f'I think of a number. I {recipe}. The result is {sp.sstr(value)}. '
            f'What is my number?\n{ANSWER_INSTRUCTION}')


def make_problem(rng, depth, core='linear', xshift=False, flipped=False,
                 word=False, max_rhs=10 ** 9):
    """
    Build one problem by wrapping a core in `depth` alternating layers.

    Layers alternate a scaling (``mul`` or ``div``) with an ``add``, so no two
    adjacent layers collapse into one and each costs exactly one solving
    operation. The value side is evaluated forward from the answer; ``div`` layers
    can make it a fraction, which is realistic and still peels back exactly.

    Params
    ------
    rng : random.Random
    depth : int
        Number of wrapping layers (the main difficulty knob).
    core : str
        ``linear`` (one answer), ``quadratic`` (two), ``cubic`` (three), or
        ``product`` (an unexpanded product that must be expanded and re-factored).
    xshift : bool
        Also carry an ``m*x`` term on the value side, costing one extra operation.
    flipped : bool
        Write the value side first (``17 = 2*(3*x + 7)``).
    word : bool
        Phrase it as a word problem; the trace then opens by writing the equation.
        Only supported for the ``linear`` core.
    max_rhs : int
        Reject problems whose value side grows past this.

    Returns
    -------
    Problem | None
        None if the draw was degenerate.
    """
    chain = []
    if core == 'linear':
        node, base_val = ('x',), sp.Rational(rng.randint(-12, 12))
    elif core == 'product':
        s1, s2 = rng.randint(-8, 8), rng.randint(-8, 8)
        d = _nz(rng, -5, 5)
        r1, r2 = s1 + d, s2 - d                       # same sum -> shifting by a
        if sorted((r1, r2)) == sorted((s1, s2)):      # constant re-factors cleanly
            return None
        node = ('prod', (r1, r2), (s1, s2))
        base_val = sp.Rational(r1 * r2 - s1 * s2)
    else:
        n_roots = 2 if core == 'quadratic' else 3
        span = 9 if core == 'quadratic' else 5
        roots = tuple(rng.randint(-span, span) for _ in range(n_roots))
        # peeling stops at ``core = base_val``; the core carries that same value as
        # its offset, so the constant move leaves prod(x - ri) = 0 exactly.
        base_val = sp.Rational(rng.choice([0, rng.randint(-12, 12), rng.randint(-12, 12)]))
        node = ('poly', roots, int(base_val))

    if word and core != 'linear':
        return None

    val = base_val
    for i in range(depth):
        if i % 2 == 0:                                # a scaling layer
            # magnitudes stay small (2-4): deep onions multiply the value side
            # once per scaling layer, so larger factors overflow `max_rhs` and
            # the problem gets rejected
            if rng.random() < 0.20:                         # divide instead
                c = rng.randint(2, 4)                       # keep divisors positive
                node, val = ('div', c, node), sp.Rational(val, c)
                chain.append(('div', c))
            else:
                c = rng.choice([-4, -3, -2, 2, 3, 4])
                node, val = ('mul', c, node), val * c
                chain.append(('mul', c))
        else:                                         # a shift layer
            c = _nz(rng, -15, 15)
            node, val = ('add', c, node), val + c
            chain.append(('add', c))
        if abs(val) > max_rhs:
            return None

    m = 0
    if xshift:                                        # same m*x on both sides
        m = _nz(rng, -6, 6)
        node = ('addx', m, node)
    ops, states, answers = solve_steps(node, m, val, flipped=flipped)
    if not ops or not answers:
        return None
    equation = render_state(node, m, val, flipped)

    if word:
        question = _story(chain, val)
        # the model must first turn the story into algebra
        ops = [f'write the equation {equation}.'] + ops
        states = [equation] + states
        family = 'word'
    else:
        question = f'Solve for x: {equation}\n{ANSWER_INSTRUCTION}'
        family = core

    # verify: the rendered equation really has exactly these integer solutions
    lhs, rhs = equation.split(' = ')
    got = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)), X)
    ints = sorted(set(int(g) for g in got if sp.simplify(g).is_Integer))
    if ints != sorted(set(answers)) or len(got) != len(set(answers)):
        return None
    return Problem(question, equation, answers, ops, states, family)


# The difficulty / diversity mix, as ``(core, (min_depth, max_depth), options)``.
# Depth sets trace length; the cores set how many answers a problem has; the
# options add surface variety (fractions from ``div`` layers are built into every
# scaling layer, an x term on the value side, flipped sides, word phrasing).
_MIX = [
    ('linear', (1, 11), dict(xshift=0.35, flipped=0.25, word=0.0)),
    ('linear', (1, 9), dict(xshift=0.0, flipped=0.0, word=1.0)),     # word problems
    ('quadratic', (1, 9), dict(xshift=0.30, flipped=0.25, word=0.0)),
    ('cubic', (1, 8), dict(xshift=0.25, flipped=0.25, word=0.0)),
    ('product', (0, 8), dict(xshift=0.25, flipped=0.25, word=0.0)),
]


def make_dataset(n, seed=0, min_ops=2, max_ops=9, mix=None, progress=True):
    """
    Sample `n` distinct problems spanning a range of difficulties and surface
    forms.

    Params
    ------
    n : int
        Number of problems.
    seed : int
        RNG seed.
    min_ops, max_ops : int
        Keep only problems whose canonical solution has this many operations.
    mix : list | None
        ``(core, (min_depth, max_depth), options)`` triples; None uses `_MIX`.
    progress : bool
        Show a progress bar. Generation is SymPy-bound (~40 problems/s, since
        every problem is solved and verified), so a 10k set takes minutes and
        should not run silently.

    Returns
    -------
    list[Problem]
    """
    rng = random.Random(seed)
    mix = mix or _MIX
    out, seen, guard = [], set(), 0
    bar = tqdm(total=n, desc=f'generating {n} problems', disable=not progress,
               dynamic_ncols=True, mininterval=1.0)
    while len(out) < n and guard < 500 * n + 20000:
        guard += 1
        core, (dlo, dhi), opt = mix[rng.randrange(len(mix))]
        p = make_problem(rng, depth=rng.randint(dlo, dhi), core=core,
                         xshift=rng.random() < opt['xshift'],
                         flipped=rng.random() < opt['flipped'],
                         word=rng.random() < opt['word'])
        if p is None or not (min_ops <= p.n_ops <= max_ops):
            continue
        if p.question in seen:
            continue
        seen.add(p.question)
        out.append(p)
        bar.update(1)
    bar.close()
    if len(out) < n:
        raise RuntimeError(f'only generated {len(out)}/{n} problems; loosen the filters')
    return out


# ----------------------------------------------------------------------------
# traces at a given state-emission interval
# ----------------------------------------------------------------------------

def encode_trace(problem, interval):
    """
    The reasoning trace for `problem` at state-emission interval `interval`.

    Lists the operation sentences in order and, every `interval` operations,
    inserts ``Current equation is <state>.``; closes with ``#### <answers>``.
    ``interval=None`` is the standard condition -- operations only, no state.

    Params
    ------
    problem : Problem
    interval : int | None
        Emit the state every `interval` operations; None = never (standard).

    Returns
    -------
    str
        The full assistant-side trace.
    """
    return _encode_ops(problem.ops, problem.states, problem.answers, interval)


def prompt_for(problem):
    """The user-side question for a problem."""
    return problem.question


SYSTEM_PROMPT = (
    'You solve single-variable equations. Work step by step, one algebraic '
    'operation per sentence, and give the integer solutions at the end, '
    'comma-separated if there is more than one.'
)


# ----------------------------------------------------------------------------
# parsing / grading a model's output
# ----------------------------------------------------------------------------

def parse_answers(text):
    """
    The integer answers a completion claims, as a sorted tuple.

    Reads whatever follows the final ``####`` marker, accepting comma- or
    whitespace-separated integers. Returns None if there is no parseable answer.
    """
    if ANSWER_MARKER not in text:
        return None
    tail = text.rsplit(ANSWER_MARKER, 1)[1]
    tail = tail.split('\n')[0].replace(',', ' ')
    vals = []
    for tok in tail.split():
        tok = tok.strip('.').strip()
        try:
            vals.append(int(tok))
        except ValueError:
            break                      # stop at the first non-integer token
    if not vals:
        return None
    return tuple(sorted(set(vals)))


def is_correct(problem, text):
    """True if `text` ends with exactly the problem's answer set."""
    got = parse_answers(text)
    return got is not None and got == tuple(sorted(set(problem.answers)))


# ----------------------------------------------------------------------------
# injected (corrupting) operations
# ----------------------------------------------------------------------------

def random_injection(rng):
    """
    A random solution-preserving operation sentence, for per-step injection at
    decode time.

    Multiply, divide, add and subtract by an integer constant are all used. Each
    preserves the solution set, so an injected step is a detour the model can
    recover from rather than a broken equation; division may leave fractional
    coefficients, which is a harder -- and more realistic -- perturbation than the
    integer-only operations.
    """
    kind = rng.choice(['mul', 'div', 'add', 'sub'])
    if kind == 'mul':
        return f'multiply both sides by {rng.randint(2, 5)}.'
    if kind == 'div':
        return f'divide both sides by {rng.randint(2, 5)}.'
    c = rng.randint(1, 12)
    return f'add {c} to both sides.' if kind == 'add' else f'subtract {c} from both sides.'


# ----------------------------------------------------------------------------
# replaying a trace symbolically (used by the diagnostics)
# ----------------------------------------------------------------------------

_OP_PATTERNS = [
    (re.compile(r'^multiply both sides by (-?\d+)\.?$'), 'mul'),
    (re.compile(r'^divide both sides by (-?\d+)\.?$'), 'div'),
    (re.compile(r'^add (\d+) to both sides\.?$'), 'add'),
    (re.compile(r'^subtract (\d+) from both sides\.?$'), 'sub'),
    (re.compile(r'^add (\d*)\*?x to both sides\.?$'), 'addx'),
    (re.compile(r'^subtract (\d*)\*?x from both sides\.?$'), 'subx'),
    (re.compile(r'^expand (?:the left hand side|the right hand side|both sides)\.?$'), 'expand'),
    (re.compile(r'^factor the left hand side\.?$'), 'factor'),
]


def parse_equation(text):
    """
    A rendered ``lhs = rhs`` string as a sympy Eq, or None if unparseable.

    Built with ``evaluate=False``: a decoded trace can contain a state with no
    variable left (``0 = 5``), and an evaluated ``Eq`` of two numbers collapses to
    ``BooleanFalse``, which has no ``.lhs``/``.rhs``. Keeping it unevaluated means
    every equation this module handles is a real relation.
    """
    if text.count('=') != 1:
        return None
    lhs, rhs = text.split('=')
    try:
        return sp.Eq(sp.sympify(lhs.strip()), sp.sympify(rhs.strip()), evaluate=False)
    except (sp.SympifyError, SyntaxError, TypeError, AttributeError, ValueError):
        return None


def apply_op(eq, op_text):
    """
    Apply one operation sentence to an equation, returning the new equation.

    Returns None if the sentence is not a recognised operation or cannot be
    applied (e.g. dividing by zero). Used to reconstruct the *true* state a model
    should be in at each point of a decoded trace, including after injections.
    """
    if eq is None or not hasattr(eq, 'lhs'):
        return None
    t = op_text.strip().lower()
    for pat, kind in _OP_PATTERNS:
        m = pat.match(t)
        if not m:
            continue
        try:
            if kind in ('expand', 'factor'):
                fn = sp.expand if kind == 'expand' else sp.factor
                return sp.Eq(fn(eq.lhs), fn(eq.rhs), evaluate=False)
            g = m.group(1)
            c = sp.Integer(int(g)) if g not in ('', None) else sp.Integer(1)
            if kind == 'mul':
                return sp.Eq(sp.expand(eq.lhs * c), sp.expand(eq.rhs * c), evaluate=False)
            if kind == 'div':
                if c == 0:
                    return None
                return sp.Eq(sp.expand(eq.lhs / c), sp.expand(eq.rhs / c), evaluate=False)
            if kind == 'add':
                return sp.Eq(sp.expand(eq.lhs + c), sp.expand(eq.rhs + c), evaluate=False)
            if kind == 'sub':
                return sp.Eq(sp.expand(eq.lhs - c), sp.expand(eq.rhs - c), evaluate=False)
            if kind == 'addx':
                return sp.Eq(sp.expand(eq.lhs + c * X), sp.expand(eq.rhs + c * X), evaluate=False)
            if kind == 'subx':
                return sp.Eq(sp.expand(eq.lhs - c * X), sp.expand(eq.rhs - c * X), evaluate=False)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def render_eq_obj(eq):
    """Render a sympy Eq back to the canonical ``lhs = rhs`` string."""
    if eq is None or not hasattr(eq, 'lhs'):
        return 'unparseable'
    return f'{sp.sstr(eq.lhs)} = {sp.sstr(eq.rhs)}'


def equations_equal(a, b):
    """
    True if two equations are the same relation (``lhs - rhs`` equal up to sign).

    Uses `expand` rather than `simplify`: these are polynomial equations, so
    expansion settles equality, and it is orders of magnitude faster. `simplify`
    is only tried as a fallback when expansion leaves something non-zero but
    small, which keeps a pathological model-generated expression from stalling
    the whole diagnostic.
    """
    # a decoded trace can yield a degenerate or unparseable equation; treat
    # anything that is not a real relation as "does not match"
    if a is None or b is None or not hasattr(a, 'lhs') or not hasattr(b, 'lhs'):
        return False
    try:
        da, db = a.lhs - a.rhs, b.lhs - b.rhs
        if len(str(da)) + len(str(db)) > 4000:      # runaway expression: give up
            return False
        for d in (sp.expand(da - db), sp.expand(da + db)):
            if d == 0:
                return True
            if d.is_number:                         # decided, and not zero
                continue
            if sp.simplify(d) == 0:
                return True
        return False
    except (TypeError, ValueError, AttributeError, RecursionError,
            OverflowError, ZeroDivisionError):
        return False


# ----------------------------------------------------------------------------
# derivation validity: was the *reasoning* right, not just the answer?
# ----------------------------------------------------------------------------

def solved_form_answers(eq):
    """
    The answers an equation exhibits if it is in *solved form*, else None.

    Solved form is either ``x = c`` (or ``c = x``), or a fully factored
    ``product = 0`` whose factors are all linear in x. Anything else -- an
    unreduced polynomial, a half-simplified expression -- is not solved, even
    though its solution set is unchanged.
    """
    if eq is None or not hasattr(eq, 'lhs'):
        return None
    for a, b in ((eq.lhs, eq.rhs), (eq.rhs, eq.lhs)):
        if a == X and b.is_number and sp.nsimplify(b).is_Integer:
            return (int(b),)
    for zero, expr in ((eq.lhs, eq.rhs), (eq.rhs, eq.lhs)):
        if zero != 0:
            continue
        factors = expr.args if isinstance(expr, sp.Mul) else (expr,)
        try:
            for f in factors:
                if not f.has(X):
                    continue                     # a numeric coefficient
                base, exp = (f.base, f.exp) if isinstance(f, sp.Pow) else (f, sp.Integer(1))
                # a repeated root is written (x - r)**n and is still fully factored
                if not exp.is_Integer or exp < 1 or sp.Poly(base, X).degree() > 1:
                    return None
            roots = sp.solve(sp.Eq(expr, 0, evaluate=False), X)
        except (sp.PolynomialError, sp.SympifyError, TypeError, ValueError):
            return None
        if not roots or any(not sp.simplify(r).is_Integer for r in roots):
            return None
        return tuple(sorted(set(int(r) for r in roots)))
    return None


def derivation_answers(problem, ops):
    """
    Replay a model's emitted operations and return the answers its derivation
    actually reaches, or None if the derivation is not sound and complete.

    Only the *operation* steps are used. The model's ``Current equation is ...``
    reports are ignored entirely: the equation is tracked here by replaying the
    operations from the problem, so the metric asks whether the sequence of steps
    is a derivation, independently of whether the model narrated the intermediate
    states correctly (that is D2's job). This also makes the metric defined for the
    standard condition, which reports no states at all.

    Returns None when any operation is unrecognised or inapplicable, or when the
    operations do not reduce the equation to a solved form (`solved_form_answers`).
    Because every legal operation preserves the solution set, this is not a test of
    whether the answer *could* be recovered -- it is a test of whether the written
    steps constitute a derivation of it.
    """
    eq = parse_equation(problem.equation)
    for text in ops:
        if text.lower().startswith('write the equation'):
            eq = parse_equation(problem.equation)      # word problems restate it
            continue
        eq = apply_op(eq, text)
        if eq is None:
            return None
    # complete iff the general solver has nothing left to do from here; it and
    # this metric therefore share one definition of "solved"
    remaining, _, answers = solve_equation(eq)
    return None if remaining else answers


def derivation_correct(problem, ops, claimed):
    """
    True if the emitted operations form a sound, complete derivation whose
    endpoint agrees with both the true answer and the answer the model claimed.

    Strictly harder than `is_correct`: a trace that states the right answer after
    steps that do not actually get there fails this.
    """
    got = derivation_answers(problem, ops)
    truth = tuple(sorted(set(problem.answers)))
    return got is not None and got == truth and claimed == truth


def replaced_ops(problem, ops, injected):
    """
    Repair a trace by putting the canonical operation back at each injected slot.

    Dropping the injected steps (the `ignore_injected` variant) is a confounded
    repair: injection replaces the operation the model was about to emit, so
    deleting it removes the model's own step too and leaves a hole in the plan. A
    trace can then fail for want of a step rather than for want of a plan. This
    variant closes the hole instead of leaving it, so what remains under judgement
    is only the operations the model actually chose.

    The substitution is positional: an injection at index `i` is replaced by
    ``problem.ops[i]``, the operation the canonical solution takes at that point,
    regardless of what the model wrote before it. Every problem has exactly one
    canonical ordering -- `solve_steps` peels the onion deterministically, and it is
    the ordering the model was trained on -- so the index alone identifies the step
    that was owed.

    Returns None when the trace is longer than the canonical solution, which counts
    as incorrect: a trace that runs long is not a repair candidate, and without that
    rule the substitutions would hand a rambling trace more free steps the longer it
    rambled.
    """
    if len(ops) > problem.n_ops:
        return None
    return [problem.ops[i] if was_injected else text
            for i, (text, was_injected) in enumerate(zip(ops, injected))]


def derivation_correct_replaced(problem, ops, injected, claimed):
    """`derivation_correct` on the positionally repaired sequence."""
    repaired = replaced_ops(problem, ops, injected)
    if repaired is None:
        return False
    return derivation_correct(problem, repaired, claimed)


# ----------------------------------------------------------------------------
# solving from an arbitrary equation (needed for injection-augmented traces)
# ----------------------------------------------------------------------------

def _denominator_lcm(*exprs):
    """LCM of the denominators of every polynomial coefficient in `exprs`."""
    dens = [1]
    for e in exprs:
        try:
            for c in sp.Poly(sp.together(e), X).all_coeffs():
                dens.append(int(sp.denom(sp.nsimplify(c))))
        except (sp.PolynomialError, sp.SympifyError, TypeError, ValueError):
            return 1
    return sp.ilcm(*dens) if len(dens) > 1 else 1


def solve_equation(eq, max_ops=40):
    """
    A canonical solution for an *arbitrary* equation, not just a generated onion.

    `solve_steps` peels the tree a problem was built from, which no longer exists
    once a random operation has been applied. This solves whatever equation it is
    given, in deterministic phases: clear denominators, expand, collect the x
    terms on the left, then either finish a linear equation (move the constant,
    divide) or reduce a higher-degree one (move the constant to zero, divide out
    the content, factor).

    Params
    ------
    eq : sympy.Eq
    max_ops : int
        Give up beyond this many operations (guards against a pathological state).

    Returns
    -------
    ops : list[str]
    states : list[str]
        Rendered equation after each operation.
    answers : tuple[int, ...] | None
        None if the equation cannot be solved to integer roots from here.
    """
    if eq is None or not hasattr(eq, 'lhs'):
        return [], [], None
    ops, states = [], []

    def emit(text, new_eq):
        nonlocal eq
        eq = new_eq
        ops.append(text)
        states.append(render_eq_obj(eq))

    try:
        roots = sp.solve(sp.Eq(sp.expand(eq.lhs - eq.rhs), 0), X)
        if not roots or any(not sp.simplify(r).is_Integer for r in roots):
            return [], [], None
        answers = tuple(sorted(set(int(r) for r in roots)))

        # 0. already solved: nothing left to do. This is also what makes "the
        # solver has no operations left" a usable definition of completeness,
        # and it stops a factored form cycling through expand -> ... -> factor.
        if solved_form_answers(eq) is not None:
            return [], [], answers

        # 1. clear denominators
        d = _denominator_lcm(eq.lhs, eq.rhs)
        if d != 1:
            emit(f'multiply both sides by {d}.',
                 sp.Eq(sp.expand(eq.lhs * d), sp.expand(eq.rhs * d), evaluate=False))
        # 2. expand any product form
        if sp.expand(eq.lhs) != eq.lhs or sp.expand(eq.rhs) != eq.rhs:
            emit('expand the left hand side.',
                 sp.Eq(sp.expand(eq.lhs), sp.expand(eq.rhs), evaluate=False))
        # 3. collect the x terms on the left
        rp = sp.Poly(eq.rhs, X)
        if rp.degree() >= 1:
            xpart = eq.rhs - rp.coeff_monomial(1)
            lead = sp.Poly(xpart, X).all_coeffs()[0]
            emit(_fmt_x_move(lead),
                 sp.Eq(sp.expand(eq.lhs - xpart), sp.expand(eq.rhs - xpart), evaluate=False))
        # 4. finish
        deg = sp.Poly(eq.lhs, X).degree()
        if deg == 1:
            a, b = sp.Poly(eq.lhs, X).all_coeffs()
            if b != 0:
                emit(_fmt_const_move(b),
                     sp.Eq(sp.expand(eq.lhs - b), sp.expand(eq.rhs - b), evaluate=False))
            a = sp.Poly(eq.lhs, X).all_coeffs()[0]
            if a != 1:
                emit(f'divide both sides by {a}.',
                     sp.Eq(sp.expand(eq.lhs / a), sp.expand(eq.rhs / a), evaluate=False))
        else:
            if eq.rhs != 0:
                c = eq.rhs
                emit(_fmt_const_move(c),
                     sp.Eq(sp.expand(eq.lhs - c), sp.Integer(0), evaluate=False))
            poly = sp.Poly(eq.lhs, X)
            content = poly.content() * sp.sign(poly.all_coeffs()[0])
            if content not in (0, 1):
                emit(f'divide both sides by {content}.',
                     sp.Eq(sp.expand(eq.lhs / content), sp.Integer(0), evaluate=False))
            emit('factor the left hand side.',
                 sp.Eq(sp.factor(eq.lhs), sp.Integer(0), evaluate=False))
        return (ops, states, answers) if len(ops) <= max_ops else ([], [], None)
    except (sp.PolynomialError, sp.SympifyError, TypeError, ValueError,
            ZeroDivisionError, NotImplementedError, IndexError):
        return [], [], None


def _encode_ops(ops, states, answers, interval):
    """Interleave state reports every `interval` operations and close with ####."""
    if interval == NO_COT:              # no reasoning at all: the answer alone
        return render_answers(answers)
    parts, since = [], 0
    for i, op in enumerate(ops):
        parts.append(op)
        since += 1
        if interval is not None and since == interval:
            parts.append(f'{STATE_PREFIX} {states[i]}.')
            since = 0
    parts.append(render_answers(answers))
    return ' '.join(parts)


def augmented_trace(problem, interval, rng, inject_p, cap_mult=3):
    """
    A training trace that contains random detours but is still a correct solution.

    At each step, with probability `inject_p`, a random solution-preserving
    operation is taken instead of the canonical next one. Training on these
    teaches recovery: the model sees operations it would not have chosen and
    learns to carry on correctly from wherever they land.

    Until a detour happens the trace follows the problem's own canonical solution
    exactly, so a trace that draws no injection is byte-identical to the clean
    one. Only after the onion structure has been destroyed does it fall back on
    `solve_equation`, which solves whatever equation it is handed. Without that
    split the augmented condition would differ from the baseline in *solving
    style* (expanding rather than peeling) as well as in injections, and the two
    effects could not be separated.

    Every state reported is truthful and the final answer is the true one, so an
    augmented trace is a valid demonstration -- only the path is unusual.

    Returns the encoded trace, or None if the detours failed to resolve or the
    trace grew past ``cap_mult`` times the canonical length.
    """
    eq = parse_equation(problem.equation)
    if eq is None:
        return None
    ops, states = [], []
    idx = 0                       # position in the canonical plan
    plan = plan_states = None     # general-solver plan, used only after a detour
    answers = tuple(sorted(set(problem.answers)))
    cap = cap_mult * max(problem.n_ops, 1) + 6

    while True:
        if plan is None:                              # still on the canonical path
            if idx >= len(problem.ops):
                break
            nxt_op, nxt_state = problem.ops[idx], problem.states[idx]
        else:                                         # re-solving after a detour
            if not plan:
                break
            nxt_op, nxt_state = plan[0], plan_states[0]

        # never inject in place of the word-problem set-up step: there is no
        # equation to operate on until it has been written down
        setup = nxt_op.lower().startswith('write the equation')
        if not setup and rng.random() < inject_p:
            op = random_injection(rng)
            eq = apply_op(eq, op)
            if eq is None:
                return None
            ops.append(op)
            states.append(render_eq_obj(eq))
            plan, plan_states, answers = solve_equation(eq)
            if answers is None:
                return None
        else:
            if setup:
                eq = parse_equation(problem.equation)
            else:
                eq = apply_op(eq, nxt_op)
                if eq is None:
                    return None
            ops.append(nxt_op)
            states.append(nxt_state)
            if plan is None:
                idx += 1
            else:
                plan, plan_states = plan[1:], plan_states[1:]
        if len(ops) > cap:
            return None

    if answers != tuple(sorted(set(problem.answers))):
        return None
    return _encode_ops(ops, states, answers, interval)
