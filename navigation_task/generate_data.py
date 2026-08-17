"""
Navigation task: square-grid shortest-path traces in two encodings, and the
exact size of the minimal DAG each encoding forces a next-token learner to
acquire.

Set-up
------
An ``m x m`` grid (``M = m*m`` cells, id ``r*m + c``, no wrap-around). A prompt
``s -> g`` asks for a shortest path from cell ``s`` to cell ``g``; moves are the
cardinal directions N, E, S, W. There are in general several shortest paths and a
training sample picks one uniformly at random. Both encodings bookend the trace
with the start and end cell labels:

Standard      prompt ``s -> g``  ->  ``s N E N g``            (start, moves, end)
De Bruijn     prompt ``s -> g``  ->  ``s N c1 E c2 N g``      (start, move, cell, ...)

e.g. ``10 -> 15`` gives ``10 N E N 15`` (standard) and ``10 N 2 E 13 N 15``
(De Bruijn): the De Bruijn trace writes the current cell before every move.

The minimal DAG (this experiment's x-axis)
------------------------------------------
Organise the traces on a minimal DAG. A node is labelled with *exactly as much of
the recent prefix as is required to determine the set of possible suffixes*, and
equal-label nodes merge; the prompt ``s -> g`` is external conditioning (it tells
the model how to navigate the DAG, so the goal is not part of a node). The number
of edges is the number of distinct context -> next-token transitions the model
must learn.

The node generalises across emission intervals to ``(last-emitted cell = anchor,
current cell)``:

* Standard (``interval=None``).  The current cell is never a token, so the anchor
  stays the start and the node is ``(start, current cell)`` -- ``O(M^2)`` edges,
  the "inefficient DAG" of the accompanying paragraph.

* De Bruijn (``interval=1``).  The current cell is re-emitted every step, so the
  anchor *is* the current cell and the node collapses to ``current cell``. Every
  route and goal through a cell shares it, so the DAG folds onto the world-model
  (the grid), with ``O(M)`` edges.

* Intermediate ``interval=k``.  The anchor resets every k moves, interpolating
  between the two.

`dag_edges(grid, prompts, interval)` computes these counts exactly for a prompt
set (including multi-token labels). At a fixed training budget the De Bruijn count
grows far slower (world model vs. reasoning DAG) than the standard one as the grid
enlarges, with the interval conditions filling in between.

Tokenisation
------------
Cell labels are base-``DIGIT_BASE`` numbers (``DIGIT_BASE = 94``), so the whole
vocabulary -- digits ``0..93`` plus ``N E S W``, the ``->`` separator and a shared
EOS/pad -- is exactly 100 tokens (see `Vocab`). A label needs
``ceil(log_base M)`` digit tokens: grids up to 9x9 (``M <= 81``) use one token per
cell, larger grids two. Growing the label length with the grid keeps the node
names within a <=100 vocabulary.
"""

import numpy as np


# ----------------------------------------------------------------------------
# vocabulary (fixed 100 tokens: 94 digits + N E S W + '->' + EOS/pad)
# ----------------------------------------------------------------------------

DIGIT_BASE = 94                       # cell-label digits are 0 .. DIGIT_BASE-1
DIRS = ('N', 'E', 'S', 'W')           # row-1, col+1, row+1, col-1
_DELTA = {'N': (-1, 0), 'E': (0, 1), 'S': (1, 0), 'W': (0, -1)}


class Vocab:
    """
    The fixed 100-token vocabulary shared by every grid size.

    Layout: token ids ``0 .. DIGIT_BASE-1`` are numeric digits used to spell cell
    labels; then N, E, S, W; then the ``->`` prompt separator; then a single EOS
    token that doubles as padding (matching `synthetic_experiment`'s convention
    that EOS == pad == the last id).

    Attributes
    ----------
    N, E, S, W : int
        Token id of each cardinal move.
    ARROW : int
        Token id of the ``->`` prompt separator.
    EOS : int
        End-of-sequence id, also used as the padding id.
    size : int
        Total vocabulary size (100).
    """

    def __init__(self, digit_base=DIGIT_BASE):
        self.digit_base = digit_base
        self.N, self.E, self.S, self.W = (digit_base + i for i in range(4))
        self.ARROW = digit_base + 4
        self.EOS = digit_base + 5
        self.size = digit_base + 6
        self.dir_token = {'N': self.N, 'E': self.E, 'S': self.S, 'W': self.W}


# ----------------------------------------------------------------------------
# the grid
# ----------------------------------------------------------------------------

class NavGrid:
    """
    An ``m x m`` grid with cardinal moves and L1 shortest paths.

    Params
    ------
    m : int
        Side length; cells are ``0 .. m*m - 1`` with id ``r*m + c``.
    digit_base : int
        Base for cell-label digits (see `Vocab`).

    Attributes
    ----------
    M : int
        Number of cells (``m*m``).
    vocab : Vocab
        The shared token vocabulary.
    label_len : int
        Number of digit tokens per cell label (``ceil(log_base M)``, >= 1).
    """

    def __init__(self, m, digit_base=DIGIT_BASE):
        self.m = m
        self.M = m * m
        self.vocab = Vocab(digit_base)
        b = digit_base
        n = 1
        while b ** n < self.M:
            n += 1
        self.label_len = n

    # -- geometry -----------------------------------------------------------

    def rc(self, cell):
        """(row, col) of a cell id."""
        return divmod(cell, self.m)

    def productive_dirs(self, cell, goal):
        """
        Moves from `cell` that lie on a shortest (L1) path to `goal`.

        Each axis that still differs contributes exactly one productive move, so
        the result has size 1 (cell shares a row or column with goal) or 2.
        """
        (r, c), (gr, gc) = self.rc(cell), self.rc(goal)
        out = []
        if gr < r:
            out.append('N')
        if gr > r:
            out.append('S')
        if gc > c:
            out.append('E')
        if gc < c:
            out.append('W')
        return out

    def valid_dirs(self, cell):
        """Moves from `cell` that stay on the grid (the world-model out-edges)."""
        r, c = self.rc(cell)
        out = []
        for d, (dr, dc) in _DELTA.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.m and 0 <= nc < self.m:
                out.append(d)
        return out

    def step(self, cell, d):
        """Neighbour of `cell` in direction `d` (assumes it stays on the grid)."""
        r, c = self.rc(cell)
        dr, dc = _DELTA[d]
        return (r + dr) * self.m + (c + dc)

    def dist(self, a, b):
        """L1 (grid) distance between two cells."""
        (ar, ac), (br, bc) = self.rc(a), self.rc(b)
        return abs(ar - br) + abs(ac - bc)

    # -- labels -------------------------------------------------------------

    def label_tokens(self, cell):
        """
        Digit tokens spelling a cell label, most-significant first, length
        `label_len`. E.g. cell 137 in base 94 -> [1, 43].
        """
        b = self.vocab.digit_base
        digits = []
        x = cell
        for _ in range(self.label_len):
            digits.append(x % b)
            x //= b
        return digits[::-1]

    def random_shortest_path(self, s, g, rng):
        """
        A random shortest path from `s` to `g`, as a list of moves.

        At each step a productive direction is drawn uniformly (this samples the
        several shortest paths of a prompt; the exact per-path weighting is not
        needed -- training only requires that every shortest edge is reachable).
        """
        moves, cell = [], s
        while cell != g:
            d = rng.choice(self.productive_dirs(cell, g))
            moves.append(d)
            cell = self.step(cell, d)
        return moves


# ----------------------------------------------------------------------------
# prompt sets (train / held-out test split)
# ----------------------------------------------------------------------------

def split_prompts(grid, test_frac=0.2, seed=0, min_dist=1):
    """
    Split all valid prompts into disjoint train / test sets.

    A fixed *fraction* ``test_frac`` of the prompts is held out as the (never
    trained) test set, the rest are training prompts. Holding out a fraction --
    rather than a fixed count -- keeps the held-out proportion constant across
    grid sizes, so small grids are not penalised with a disproportionately large
    test set.

    Params
    ------
    grid : NavGrid
        The grid to draw cells from.
    test_frac : float
        Fraction of prompts to hold out for the test set (in (0, 1)).
    seed : int
        Seed for the prompt shuffle.
    min_dist : int
        Minimum L1 distance between start and goal.

    Returns
    -------
    train, test : list[tuple[int, int]]
        Disjoint prompt lists; test is ``round(test_frac * n)`` prompts.
    """
    rng = np.random.default_rng(seed)
    pool = [(s, g) for s in range(grid.M) for g in range(grid.M)
            if s != g and grid.dist(s, g) >= min_dist]
    rng.shuffle(pool)
    n_test = max(1, int(round(test_frac * len(pool))))
    n_test = min(n_test, len(pool) - 1)          # keep at least one train prompt
    return pool[n_test:], pool[:n_test]


# ----------------------------------------------------------------------------
# encoders: a (prompt, chosen path) -> token sequence
# ----------------------------------------------------------------------------

def encode(grid, s, g, moves, interval):
    """
    Trace tokens for one prompt and chosen path under an emission `interval`.

    The answer opens with ``label(s)``, then one token per move; the *current
    cell* is re-emitted every `interval` moves, and the trace always closes with
    ``label(g)`` then EOS.

    * ``interval=1``     re-emits after every move -- the De Bruijn trace
      ``label(s) move label(c1) move ... label(g) EOS`` (the current cell is always
      an explicit recent token).
    * ``interval=None``  never emits an intermediate cell -- the standard trace
      ``label(s) move move ... label(g) EOS``.
    * intermediate `interval` values interpolate: the cell is seen every k steps.

    Returns (tokens, answer_start), where answer_start is the index of the first
    answer token (the start-cell label that opens the trace).
    """
    v = grid.vocab
    prompt = grid.label_tokens(s) + [v.ARROW] + grid.label_tokens(g)
    answer, cell, since = list(grid.label_tokens(s)), s, 0
    last = len(moves) - 1
    for i, d in enumerate(moves):
        answer.append(v.dir_token[d])
        cell = grid.step(cell, d)
        since += 1
        if i == last:
            answer += grid.label_tokens(g)                 # closing goal label
        elif interval is not None and since == interval:
            answer += grid.label_tokens(cell)              # intermediate cell
            since = 0
    answer.append(v.EOS)
    return prompt + answer, len(prompt)


def max_tokens(grid, interval):
    """
    Upper bound on sequence length for a grid at a given emission `interval`
    (longest path = 2*(m-1) moves; interval=None is the standard, shortest, trace).
    """
    L = 2 * (grid.m - 1)
    n_inter = 0 if interval is None else max(0, (L - 1) // interval)
    n_labels = n_inter + 2                                  # start + intermediates + goal
    return (2 * grid.label_len + 1) + n_labels * grid.label_len + L + 1


def mean_tokens(grid, interval, min_dist=1):
    """
    Mean trace length (total tokens) over the prompt distribution -- all ordered
    cell pairs with L1 distance >= `min_dist` -- at a given emission `interval`.
    A path of length ``L`` emits ``floor((L-1)/interval)`` intermediate cells (0
    for standard), so its trace is ``prompt + start-label + L moves + intermediate
    labels + goal-label + EOS``. This is the expected sequence length; used as the
    "length" variable in the reliability term of the fit (contrast `max_tokens`,
    the worst-case length).
    """
    m = grid.m
    rc = np.array([divmod(c, m) for c in range(grid.M)])
    dr = np.abs(rc[:, 0][:, None] - rc[:, 0][None, :])
    dc = np.abs(rc[:, 1][:, None] - rc[:, 1][None, :])
    L = (dr + dc).ravel().astype(float)
    L = L[L >= min_dist]
    n_inter = np.zeros_like(L) if interval is None else np.floor((L - 1) / interval)
    answer = grid.label_len * (2 + n_inter) + L + 1        # start+goal labels, moves, intermediates, EOS
    return float((2 * grid.label_len + 1) + answer.mean())


# ----------------------------------------------------------------------------
# sampling a padded token batch
# ----------------------------------------------------------------------------

def sample_batch(grid, prompts, num, interval, rng, T=None):
    """
    Draw `num` training/test sequences from a prompt set.

    Each sample picks a random prompt from `prompts` (with replacement) and a
    random shortest path for it, then encodes it at the given emission `interval`.
    Sequences are right-padded with EOS to width `T` (matching
    `synthetic_experiment`'s EOS == pad convention).

    Params
    ------
    grid : NavGrid
        The grid.
    prompts : list[tuple[int, int]]
        Prompts to sample from.
    num : int
        Number of sequences to draw.
    interval : int | None
        Cell-emission interval (1 = De Bruijn, None = standard); see `encode`.
    rng : numpy random generator
        Source of randomness (prompt choice and path choice).
    T : int | None
        Padded width; defaults to `max_tokens(grid, interval)`.

    Returns
    -------
    tokens : (num, T) int64
        Token ids, EOS-padded after each sequence's end.
    lengths : (num,) int64
        Number of real tokens per sequence (through and including the EOS).
    answer_start : (num,) int64
        Index of the first answer token in each sequence.
    """
    v = grid.vocab
    T = T or max_tokens(grid, interval)
    tokens = np.full((num, T), v.EOS, dtype=np.int64)
    lengths = np.empty(num, dtype=np.int64)
    answer_start = np.empty(num, dtype=np.int64)
    idx = rng.integers(0, len(prompts), size=num)
    for i, pi in enumerate(idx):
        s, g = prompts[pi]
        moves = grid.random_shortest_path(s, g, rng)
        seq, a0 = encode(grid, s, g, moves, interval)
        tokens[i, :len(seq)] = seq
        lengths[i] = len(seq)
        answer_start[i] = a0
    return tokens, lengths, answer_start


# ----------------------------------------------------------------------------
# exact minimal-DAG edge counts (the plot's x-axis)
# ----------------------------------------------------------------------------

def dag_edges(grid, prompts, interval):
    """
    Exact edge count of the minimal DAG over `prompts` at a given emission
    `interval` (the plot's x-axis).

    Enumerates every distinct ``(node, next-token)`` transition of the full trace
    (opening ``label(s)``, moves, cell re-emissions every `interval` moves,
    closing ``label(g)``, EOS). The prompt ``s -> g`` is external conditioning, so
    the goal is never in a node. The node is

        ``(last-emitted cell = anchor, current cell)``

    -- the current cell is recoverable from the anchor token plus the moves since
    it was last emitted, and the anchor resets each time a cell is re-emitted. So:

    * ``interval=1`` (De Bruijn): the anchor *is* the current cell (re-emitted
      every step), the node collapses to ``current cell``, and the DAG folds onto
      the world model -- ``O(M)`` edges.
    * ``interval=None`` (standard): no cell is re-emitted, the anchor stays the
      start, the node is ``(start, current cell)`` -- ``O(M^2)`` edges.
    * intermediate `interval`: the anchor resets every k moves, so move-prediction
      nodes ``(anchor, current)`` with ``dist < k`` interpolate between the two.

    Counted exactly, including multi-token labels: the opening/closing/interior
    cell labels are spelt digit by digit (first digit keyed by the emitting node,
    inner digits by the target cell); EOS is keyed by the just-emitted goal cell.
    """
    v = grid.vocab
    L = grid.label_len
    lab = {c: tuple(grid.label_tokens(c)) for c in range(grid.M)}
    k = 10 ** 9 if interval is None else interval
    starts = {s for s, _ in prompts}

    mv = set()            # (anchor, current, dir)  -- move predictions
    emit_cell = set()     # (anchor, cell)          -- intermediate cell emission
    emit_goal = set()     # (anchor, goal)          -- goal-label emission
    cells_emitted = set()
    eos_cells = set()
    # BFS over (anchor, current) states per prompt; the set dedups across prompts.
    for (s, g) in prompts:
        stack, seen = [(s, s)], set()
        while stack:
            anchor, current = stack.pop()
            if (anchor, current) in seen:
                continue
            seen.add((anchor, current))
            if current == g:
                emit_goal.add((anchor, g))
                cells_emitted.add(g)
                eos_cells.add(g)
                continue
            for d in grid.productive_dirs(current, g):
                mv.add((anchor, current, v.dir_token[d]))
                nxt = grid.step(current, d)
                if nxt == g:
                    stack.append((anchor, nxt))              # goal emitted when popped
                elif grid.dist(anchor, nxt) == k:
                    emit_cell.add((anchor, nxt))             # re-emit, anchor resets
                    cells_emitted.add(nxt)
                    stack.append((nxt, nxt))
                else:
                    stack.append((anchor, nxt))

    E = set()
    for s in starts:                                         # opening: spell start
        for j in range(L):
            E.add(('open', lab[s][:j], lab[s][j]))
    for (a, c, dt) in mv:                                    # move predictions
        E.add(('mv', a, c, dt))
    for (a, c) in emit_cell:                                 # intermediate emit (first digit)
        E.add(('e1', a, c, lab[c][0]))
    for (a, gg) in emit_goal:                                # goal emit (first digit)
        E.add(('g1', a, gg, lab[gg][0]))
    for c in cells_emitted:                                  # inner digits keyed by cell
        for j in range(1, L):
            E.add(('in', c, j, lab[c][j]))
    for c in eos_cells:                                      # EOS after a goal cell
        E.add(('eos', c))
    return len(E)
