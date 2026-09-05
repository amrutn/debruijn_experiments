"""
Teacher-written solutions for the distillation condition.

The distillation condition fine-tunes on the same 1,500 training problems as
the other conditions, but on a solution written by a much larger model instead
of the ground truth or the ledger trace. The solutions come from a public Hub
dataset pinned to a commit. ``--build`` pairs them with the training subset,
verifies every final answer against GSM8K's, and writes the result next to
this file as ``traces_distill_<teacher>.txt`` in the format of
``traces_ledger.txt``, so that a run needs no network and the exact text every
adapter was trained on is in the repository.

Teachers (`TEACHERS`)
---------------------
qwen3.5-397b    (default) ``jerryjsjsj/gsm8k-qwen3.5-teacher-traces``:
                Qwen3.5-397B-A17B (Alibaba DashScope API) asked for a concise
                chain of thought and a bare final answer for every GSM8K train
                problem; the dataset keeps the 7,256 of 7,473 whose answer
                matches GSM8K's. The traces are as short as the ground truth
                (88 vs 91 assistant tokens on average over the subset), so this
                condition differs from 'standard' in who wrote the reasoning,
                not in how long it is, and it runs under the same sequence and
                decoding budgets as every other condition.
qwen3-8b-think  ``pingzhili/qwen3-8b-gsm8k``: Qwen3-8B in thinking mode, the
                whole ``<think>...</think>`` block followed by its answer, for
                every train problem (751 rows have no final response and are
                unusable). About twenty times the ground-truth length (median
                ~1,400 assistant tokens), so it carries its own budgets:
                training sequences up to 4,096 tokens (longer ones are dropped
                and counted) and 4,096 new tokens at evaluation. A different
                regime from the other conditions; offered for the
                thinking-trace flavour of distillation.

Every teacher trace is followed by the ``#### <answer>`` line the prompt asks
for, written with GSM8K's answer, which the trace's own answer was verified to
equal. A problem without a usable teacher trace (no record, or a wrong
answer) is filled with its ground-truth solution (``--fill standard``, the
default), so the condition sees the same 1,500 problems as the others, or
dropped (``--fill drop``). The fill count is reported, recorded in the file
header, and recovered by the loader.

    python distill.py --build                    # download, pair, verify, write
    python distill.py --check                    # validate the file, statistics
    python distill.py --show 3
    python distill.py --build --teacher qwen3-8b-think
"""

import os
import re
import json
import argparse
from datetime import datetime, timezone

from gsm8k_data import (
    DataConfig, get_problems, standard_trace, extract_answer, _same_number,
    is_correct, train_spec, TRACE_VERSION, ANSWER_MARKER, HERE,
)
from traces import (
    TraceConfig, question_hash, format_trace_file, parse_trace_file, _file_sha,
)

TEACHERS = {
    'qwen3.5-397b': dict(
        dataset='jerryjsjsj/gsm8k-qwen3.5-teacher-traces',
        revision='a07edc05646a97eba0b7f933553a15ee52821ecd',
        file='data/accepted.jsonl',
        teacher='qwen3.5-397b-a17b via the Alibaba DashScope API; concise '
                'reasoning plus a bare final answer, answer-matched to GSM8K',
        style='concise',
        max_seq_len=None,           # None: the run's default budget applies
        max_new_tokens=None,
    ),
    'qwen3-8b-think': dict(
        dataset='pingzhili/qwen3-8b-gsm8k',
        revision='1a60500f4cf4b35a94a511d0a4e9dfc2cd39c545',
        file='data/train-00000-of-00001.parquet',
        teacher='Qwen/Qwen3-8B in thinking mode; <think> block plus final response',
        style='thinking',
        max_seq_len=4096,
        max_new_tokens=4096,
    ),
}
FILLS = ('standard', 'drop')


def trace_file(teacher):
    """Where the built traces of `teacher` live: next to this file."""
    return os.path.join(HERE, f'traces_distill_{teacher}.txt')


def budget(tcfg):
    """
    ``(max_seq_len, max_new_tokens)`` the teacher's traces call for, each None
    where the run's default budget applies.
    """
    t = TEACHERS[tcfg.teacher]
    return t['max_seq_len'], t['max_new_tokens']


def distill_spec(tcfg, dcfg):
    """
    The cache key of a distillation trace set: teacher, source dataset and
    commit, fill policy, training subset, and a hash of the built file, so a
    rebuilt file re-trains the distillation adapters.
    """
    t = TEACHERS[tcfg.teacher]
    path = trace_file(tcfg.teacher)
    return dict(task='gsm8k-distill', teacher=tcfg.teacher, dataset=t['dataset'],
                revision=t['revision'], file=t['file'], fill=tcfg.fill,
                data=train_spec(dcfg), trace_version=TRACE_VERSION,
                file_sha=_file_sha(path) if os.path.exists(path) else None)


def _norm(s):
    return ' '.join(s.split()).strip()


# ----------------------------------------------------------------------------
# reading the Hub dataset
# ----------------------------------------------------------------------------

def teacher_rows(teacher):
    """
    The teacher's records from the pinned Hub file, as
    ``(train_row or None, question, trace_text, answer or None)`` tuples.
    Downloads the file (a few MB) into the Hub cache on first use.
    """
    from huggingface_hub import hf_hub_download
    t = TEACHERS[teacher]
    path = hf_hub_download(t['dataset'], t['file'], repo_type='dataset',
                           revision=t['revision'])
    rows = []
    if t['style'] == 'concise':
        with open(path) as f:
            for ln in f:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                rows.append((int(r['source_index']), r['question'],
                             r['teacher_reasoning'].strip(),
                             str(r['teacher_answer']).strip()))
    elif t['style'] == 'thinking':
        import pyarrow.parquet as pq
        for r in pq.read_table(path).to_pylist():
            think, resp = r.get('reasoning_response'), r.get('response')
            if not isinstance(think, str) or not isinstance(resp, str):
                continue                        # thinking never finished
            text = f'<think>\n{think.strip()}\n</think>\n\n{resp.strip()}'
            rows.append((None, r['prompt'], text, extract_answer(resp)))
    else:
        raise ValueError(f'unknown teacher style {t["style"]!r}')
    return rows


# ----------------------------------------------------------------------------
# building and loading the trace file
# ----------------------------------------------------------------------------

def build(tcfg, dcfg, log=print):
    """
    Pair the teacher's records with the training subset, verify the answers,
    apply the fill policy and write ``traces_distill_<teacher>.txt``.

    A record is paired by its recorded train-row index when the dataset has
    one (the question text must agree) and by exact question text otherwise;
    the first record per problem wins. A trace counts only if its answer
    equals GSM8K's; it is then closed with ``#### <answer>``.

    Returns
    -------
    dict
        The build statistics that also head the file.
    """
    if tcfg.fill not in FILLS:
        raise ValueError(f'fill must be one of {FILLS}, not {tcfg.fill!r}')
    t = TEACHERS[tcfg.teacher]
    problems = get_problems(dcfg, 'train')
    by_idx = {p.idx: p for p in problems}
    by_q = {_norm(p.question): p for p in problems}

    found, wrong = {}, {}
    for idx, q, text, ans in teacher_rows(tcfg.teacher):
        p = by_idx.get(idx) if idx is not None else by_q.get(_norm(q))
        if p is None or _norm(q) != _norm(p.question) or p.idx in found:
            continue
        if ans is not None and _same_number(ans, p.gold):
            found[p.idx] = f'{text}\n{ANSWER_MARKER} {p.gold}'
        else:
            wrong[p.idx] = ans

    traces, filled, dropped = {}, [], []
    for p in problems:
        if p.idx in found:
            traces[p.idx] = found[p.idx]
        elif tcfg.fill == 'standard':
            traces[p.idx] = standard_trace(p)
            filled.append(p.idx)
        else:
            dropped.append(p.idx)

    meta = dict(teacher=tcfg.teacher, dataset=t['dataset'], revision=t['revision'],
                file=t['file'], fill=tcfg.fill, n_problems=len(problems),
                n_teacher=len(found), n_wrong=len(wrong),
                n_missing=len(problems) - len(found) - len(wrong),
                n_filled=len(filled), n_dropped=len(dropped),
                built=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    head = [
        f'# distillation traces for the GSM8K training subset '
        f'(n_train={dcfg.n_train}, train_seed={dcfg.train_seed}), built {meta["built"]} '
        f'by distill.py --build --teacher {tcfg.teacher} --fill {tcfg.fill}',
        f'# teacher: {t["teacher"]}',
        f'# source: {t["dataset"]} @ {t["revision"]}, file {t["file"]}',
        f'# {len(found)} teacher traces; {len(wrong)} records with a wrong answer; '
        f'{meta["n_missing"]} problems without a record',
        f'# filled with the ground-truth solution: {len(filled)}'
        + (' (rows ' + ', '.join(map(str, filled)) + ')' if filled else ''),
        f'# dropped: {len(dropped)}'
        + (' (rows ' + ', '.join(map(str, dropped)) + ')' if dropped else ''),
        '# every record ends with "#### <GSM8K answer>", which the teacher\'s '
        'own answer was verified to equal',
        '',
    ]
    body = format_trace_file([p for p in problems if p.idx in traces], traces)
    path = trace_file(tcfg.teacher)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        f.write('\n'.join(head) + body)
    os.replace(tmp, path)
    log(f'wrote {path}: {len(traces)} records ({len(found)} teacher, {len(filled)} '
        f'filled, {len(dropped)} dropped; {len(wrong)} wrong-answer records)')
    return meta


def load_distill_traces(tcfg, dcfg, path=None):
    """
    The built traces for the training subset, validated, as ``{idx: text}``.

    Every problem needs a record whose question hash matches and whose last
    line is ``#### <GSM8K answer>``; otherwise a ValueError names the first
    offenders, so a stale file fails before anything is trained. Under
    ``fill='drop'`` a missing record is allowed, and a record that is the
    ground-truth solution (a fill) is skipped.
    """
    path = path or trace_file(tcfg.teacher)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{os.path.basename(path)} is missing: run `python distill.py --build '
            f'--teacher {tcfg.teacher}` (needs the network once) before the experiment')
    with open(path) as f:
        recs = parse_trace_file(f.read())
    problems = get_problems(dcfg, 'train')
    traces, bad = {}, []
    for p in problems:
        if p.idx not in recs:
            if tcfg.fill != 'drop':
                bad.append(f'row {p.idx}: no record')
            continue
        qh, trace = recs[p.idx]
        lines = [ln for ln in trace.split('\n') if ln.strip()]
        if qh != question_hash(p.question):
            bad.append(f'row {p.idx}: question hash {qh} != {question_hash(p.question)}')
        elif '<<' in trace or '>>' in trace:
            bad.append(f'row {p.idx}: calculator annotation present')
        elif len(lines) < 2 or not lines[-1].startswith(ANSWER_MARKER) \
                or not is_correct(lines[-1], p.gold):
            bad.append(f'row {p.idx}: does not end with "{ANSWER_MARKER} {p.gold}"')
        elif tcfg.fill == 'drop' and trace == standard_trace(p):
            continue
        else:
            traces[p.idx] = trace
    if bad:
        more = f' (+{len(bad) - 8} more)' if len(bad) > 8 else ''
        raise ValueError(f'{len(bad)} of {len(problems)} traces unusable in '
                         f'{os.path.basename(path)}:\n  ' + '\n  '.join(bad[:8]) + more)
    if not traces:
        raise ValueError(f'no usable traces in {os.path.basename(path)}')
    return traces


def distill_summary(tcfg, dcfg):
    """
    Size statistics of the built trace set for the run header, or None if the
    file is missing or invalid (the run reports that itself when it loads it).
    """
    try:
        traces = load_distill_traces(tcfg, dcfg)
    except (FileNotFoundError, ValueError, KeyError):
        return None
    by = {p.idx: p for p in get_problems(dcfg, 'train')}
    n_filled = sum(traces[i] == standard_trace(by[i]) for i in traces)
    chars = sum(len(t) for t in traces.values()) / len(traces)
    std = sum(len(standard_trace(by[i])) for i in traces) / len(traces)
    return dict(n=len(traces), n_filled=n_filled, chars=chars, char_ratio=chars / max(std, 1),
                teacher=TEACHERS[tcfg.teacher]['teacher'])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--teacher', choices=sorted(TEACHERS), default=TraceConfig.teacher)
    ap.add_argument('--fill', choices=FILLS, default=TraceConfig.fill)
    ap.add_argument('--n-train', type=int, default=DataConfig.n_train)
    ap.add_argument('--train-seed', type=int, default=DataConfig.train_seed)
    ap.add_argument('--build', action='store_true',
                    help='download the teacher records and (re)write the trace file')
    ap.add_argument('--check', action='store_true',
                    help='validate the trace file and print size statistics')
    ap.add_argument('--show', type=int, default=0, metavar='N',
                    help='print N problems as standard and teacher traces')
    args = ap.parse_args()

    dcfg = DataConfig(n_train=args.n_train, train_seed=args.train_seed)
    tcfg = TraceConfig(teacher=args.teacher, fill=args.fill)
    if args.build:
        build(tcfg, dcfg)
    problems = get_problems(dcfg, 'train')
    if args.show:
        traces = load_distill_traces(tcfg, dcfg)
        for p in [p for p in problems if p.idx in traces][:args.show]:
            print('=' * 72)
            print(p.question.strip())
            print('-' * 24 + ' standard ' + '-' * 24)
            print(standard_trace(p))
            print('-' * 24 + f' distill ({args.teacher}) ' + '-' * 16)
            print(traces[p.idx])
    if args.check or not (args.show or args.build):
        load_distill_traces(tcfg, dcfg)               # raises with the offenders
        s = distill_summary(tcfg, dcfg)
        print(f'{s["n"]} traces valid ({args.teacher}, fill={args.fill}); '
              f'{s["n_filled"]} are ground-truth fills; {s["chars"]:.0f} chars per '
              f'trace, {s["char_ratio"]:.2f}x standard')


if __name__ == '__main__':
    main()
