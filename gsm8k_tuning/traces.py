"""
Restated (ledger) traces for the GSM8K training subset, and who wrote them.

The restated condition fine-tunes on the ground-truth steps interleaved with a
ledger of the problem's state (`gsm8k_data`). Two authors, chosen with
``--author``:

claude   ``traces_ledger.txt``, next to this file: the ledger lines for every
         problem of the 1,500-problem training subset, written by Claude
         (Claude Code, September 2026) from the problem and its reference
         solution. The step lines are the ground truth verbatim; only the
         ledger lines are authored, so the restated condition differs from the
         standard one by those lines alone. This is the default.
rule     `gsm8k_data.restated_trace_rule`: a mechanical ledger that quotes the
         problem's numeric sentences and the step lines so far. Free, offline,
         and a control for the *wording* of the hand-written ledgers, at the
         cost of ledgers that grow with every step rather than updating.

Both pass through `gsm8k_data.validate_restated` when loaded: same final
answer, ``n_steps + 1`` ledgers alternating with the ground-truth steps, no
calculator annotations. A trace file that fails stops the run before a GPU is
touched, with the offending rows named.

File format
-----------
One record per problem, in training-subset order:

    ### train row 0 q=3f2a9c1e
    Known so far: ...
    <step>
    ...
    Known so far: ...
    #### 72

The header carries the row index in the GSM8K train split and a short hash of
the question, which the loader checks against the dataset so that a change in
row order can never silently pair a trace with the wrong problem.

    python traces.py --check              # validate the file, print statistics
    python traces.py --show 3             # print examples (either author)
    python traces.py --show 3 --author rule
"""

import os
import re
import argparse
import hashlib
from dataclasses import dataclass

from gsm8k_data import (
    DataConfig, get_problems, standard_trace, restated_trace_rule,
    validate_restated, unknown_ledger_numbers, train_spec, TRACE_VERSION,
    LEDGER_PREFIX, HERE,
)

TRACE_FILE = os.path.join(HERE, 'traces_ledger.txt')
AUTHORS = ('claude', 'rule')

_HEADER = re.compile(r'^### train row (\d+) q=([0-9a-f]{8})$')


@dataclass
class TraceConfig:
    """
    Where the traces that are not the ground truth come from: the ledger
    author of the restated condition, and the teacher of the distillation
    condition (`distill.py`). Each condition's cache key reads only its own
    fields, so changing the teacher never invalidates a restated adapter and
    vice versa.
    """
    author: str = 'claude'          # restated: 'claude' | 'rule'
    teacher: str = 'qwen3.5-397b'   # distill: a key of `distill.TEACHERS`
    fill: str = 'standard'          # distill: problems without a usable teacher trace
    trace_version: str = TRACE_VERSION


def question_hash(question):
    """Eight hex characters identifying a problem by its text."""
    return hashlib.sha1(' '.join(question.split()).encode()).hexdigest()[:8]


def _file_sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()[:16]


def trace_spec(tcfg, dcfg):
    """
    The cache key of a restated trace set: the training subset and the author,
    plus a hash of the trace file for the claude author, so that editing the
    file re-trains the restated adapters rather than reusing ones trained on
    the old text. The teacher fields of `tcfg` are deliberately not read.
    """
    spec = dict(task='gsm8k-traces', author=tcfg.author, data=train_spec(dcfg),
                trace_version=tcfg.trace_version)
    if tcfg.author == 'claude':
        spec['file_sha'] = _file_sha(TRACE_FILE) if os.path.exists(TRACE_FILE) else None
    return spec


# ----------------------------------------------------------------------------
# the trace file
# ----------------------------------------------------------------------------

def format_trace_file(problems, traces):
    """The file text for `traces` (``{idx: trace}``), in `problems` order."""
    out = []
    for p in problems:
        out.append(f'### train row {p.idx} q={question_hash(p.question)}')
        out.append(traces[p.idx].strip())
        out.append('')
    return '\n'.join(out)


def parse_trace_file(text):
    """
    ``{idx: (question_hash, trace)}`` from the file text. Raises on a header
    that does not parse or a duplicated row.
    """
    out, cur, buf = {}, None, []

    def flush():
        if cur is not None:
            idx, qh = cur
            if idx in out:
                raise ValueError(f'train row {idx} appears twice')
            out[idx] = (qh, '\n'.join(buf).strip())

    for ln in text.split('\n'):
        if ln.startswith('### '):
            m = _HEADER.match(ln.strip())
            if not m:
                raise ValueError(f'bad record header: {ln!r}')
            flush()
            cur, buf = (int(m.group(1)), m.group(2)), []
        elif cur is not None:
            buf.append(ln.rstrip())
    flush()
    return out


def load_claude_traces(problems, path=TRACE_FILE):
    """
    The hand-written traces for `problems`, validated.

    Every problem must have a record whose question hash matches and whose
    trace passes `validate_restated`; otherwise a ValueError names the first
    few offenders, so a stale or edited file fails loudly before training.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} is missing: the claude author needs the '
                                'hand-written trace file (or use --author rule)')
    with open(path) as f:
        recs = parse_trace_file(f.read())
    traces, problems_ = {}, []
    for p in problems:
        if p.idx not in recs:
            problems_.append(f'row {p.idx}: no record')
            continue
        qh, trace = recs[p.idx]
        if qh != question_hash(p.question):
            problems_.append(f'row {p.idx}: question hash {qh} != {question_hash(p.question)} '
                             '(dataset row order changed?)')
            continue
        ok, why = validate_restated(trace, p)
        if not ok:
            problems_.append(f'row {p.idx}: {why}')
            continue
        traces[p.idx] = trace
    if problems_:
        more = f' (+{len(problems_) - 8} more)' if len(problems_) > 8 else ''
        raise ValueError(f'{len(problems_)} of {len(problems)} traces unusable in '
                         f'{os.path.basename(path)}:\n  ' + '\n  '.join(problems_[:8]) + more)
    return traces


# ----------------------------------------------------------------------------
# entry points
# ----------------------------------------------------------------------------

def load_traces(tcfg, dcfg):
    """The restated trace for every training problem, as ``{idx: text}``."""
    problems = get_problems(dcfg, 'train')
    if tcfg.author == 'claude':
        return load_claude_traces(problems)
    if tcfg.author == 'rule':
        out = {}
        for p in problems:
            t = restated_trace_rule(p)
            ok, why = validate_restated(t, p)
            if not ok:
                raise RuntimeError(f'rule-based trace for train row {p.idx} failed: {why}')
            out[p.idx] = t
        return out
    raise ValueError(f'unknown trace author {tcfg.author!r}; choose from {AUTHORS}')


def trace_summary(tcfg, dcfg):
    """
    Size statistics of a trace set for the run header, or None if the claude
    file is missing or invalid (the run reports that itself when it loads it).
    """
    try:
        traces = load_traces(tcfg, dcfg)
    except (FileNotFoundError, ValueError, RuntimeError):
        return None
    problems = get_problems(dcfg, 'train')
    n_items = n_ledgers = 0
    for t in traces.values():
        for ln in t.split('\n'):
            if ln.startswith(LEDGER_PREFIX):
                n_ledgers += 1
                n_items += ln.count(';') + 1
    chars = sum(len(t) for t in traces.values()) / max(len(traces), 1)
    std = sum(len(standard_trace(p)) for p in problems) / max(len(problems), 1)
    return dict(n=len(traces), items_per_ledger=n_items / max(n_ledgers, 1),
                chars=chars, char_ratio=chars / max(std, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--author', choices=AUTHORS, default='claude')
    ap.add_argument('--n-train', type=int, default=DataConfig.n_train)
    ap.add_argument('--train-seed', type=int, default=DataConfig.train_seed)
    ap.add_argument('--check', action='store_true',
                    help='validate every trace and print size statistics')
    ap.add_argument('--show', type=int, default=0, metavar='N',
                    help='print N problems as standard and restated traces')
    args = ap.parse_args()

    dcfg = DataConfig(n_train=args.n_train, train_seed=args.train_seed)
    tcfg = TraceConfig(author=args.author)
    problems = get_problems(dcfg, 'train')

    if args.show:
        traces = load_traces(tcfg, dcfg)
        for p in problems[:args.show]:
            print('=' * 72)
            print(p.question.strip())
            print('-' * 24 + ' standard ' + '-' * 24)
            print(standard_trace(p))
            print('-' * 24 + f' restated ({args.author}) ' + '-' * 18)
            print(traces[p.idx])
    if args.check or not args.show:
        traces = load_traces(tcfg, dcfg)             # raises with the offenders
        s = trace_summary(tcfg, dcfg)
        odd = {p.idx: unknown_ledger_numbers(traces[p.idx], p) for p in problems}
        odd = {k: v for k, v in odd.items() if v}
        print(f'{s["n"]} traces valid ({args.author}); {s["items_per_ledger"]:.1f} items '
              f'per ledger; {s["chars"]:.0f} chars per trace, {s["char_ratio"]:.2f}x standard')
        print(f'{len(odd)} traces mention a number that neither the problem nor a '
              'step states' + (': ' + ', '.join(
                  f'row {k} {v}' for k, v in list(odd.items())[:10]) if odd else ''))


if __name__ == '__main__':
    main()
