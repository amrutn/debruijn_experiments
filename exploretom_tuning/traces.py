"""
Training targets per condition, and the teacher-written traces of the
optional distillation condition.

Four conditions need no teacher: their targets are computed exactly from
the replayed story (`exploretom_data.TARGETS`):

direct          the answer field alone,
key_steps       the steps at which the answer to the question changes,
state_tracking  the story restated with the question's slice of the state
                after each block of `focus_stride` steps (every step by
                default), the asked beliefs written explicitly
                (`focus_beliefs`),
narration       the story restated with a local per-step note, no running
                state (the length-matched non-De-Bruijn control).

The fourth, `distill`, trains on a teacher model's own reasoning: the teacher
answers the training question under the student's prompt (plus a request for
brief plain prose), and the trace is kept only if its answer is the label
(one sampled retry, then a FAILED record). Teachers (``--teacher``):

hf:<hub id>[@<revision>]: a local Hugging Face chat model decoded in batches
          on ``--device``. The default is ``hf:Qwen/Qwen3-32B`` at the revision
          pinned in `HF_REVISIONS`; thinking mode is off unless ``--thinking``
          is passed, and any ``<think>...</think>`` block is stripped.
<anthropic model id>, e.g. ``claude-opus-5``: through the Anthropic API
          (``ANTHROPIC_API_KEY``, ``pip install anthropic``).

The traces live in ``traces_distill_<teacher>.txt`` next to this file, one
record per training row, written as the teacher answers (a build that stops
resumes where it was). The distill condition trains on the first `n_train`
rows of the training pool that have a usable trace: a row the teacher gets
wrong is replaced by the next row of the pool (`training_rows`, `build`),
so it also sees `n_train` rows, though not exactly the rows of the other
conditions. `run_experiments` builds the file itself when the condition is
requested and the file is short.

File format
-----------
    ### row 12 q=3f2a9c1e            (`q` hashes story, question and answer)
    <the teacher's reasoning>
    Answer: leather briefcase

A header ending in ``FAILED`` marks a row the teacher got wrong twice; its
body is the last attempt, and the loader treats it as absent.

    python traces.py --build --device cuda:0 --batch 16
    python traces.py --check
    python traces.py --show 2
"""

import os
import re
import sys
import time
import argparse
import hashlib
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from exploretom_data import (
    DataConfig, get_problems, train_pool, user_message, question_hash, answer_line,
    extract_answer, normalise, train_spec, TARGETS, state_tracking_target,
    SYSTEM_PROMPT, TRACE_VERSION, TARGET_VERSIONS, FOCUS_STRIDE, FOCUS_BELIEFS,
    cache_cond, HERE,
)

TEACHER = 'hf:Qwen/Qwen3-32B'
# The Hub commits the local teachers are read at when `--teacher` names no
# revision, so the traces are reproducible from the same weights.
HF_REVISIONS = {
    'Qwen/Qwen3-32B': '9216db5781bf21249d130ec9da846c4624c16137',
}
# Conditions whose targets are computed exactly from the replayed story
# (no teacher).
EXACT = ('direct', 'key_steps', 'state_tracking', 'narration')

_HEADER = re.compile(r'^### row (\d+) q=([0-9a-f]{8})( FAILED)?$')


@dataclass
class TraceConfig:
    """How the targets are written: the state-tracking stride and belief mode,
    and, for the distillation condition, who wrote the traces."""
    teacher: str = TEACHER
    focus_stride: int = FOCUS_STRIDE
    focus_beliefs: str = FOCUS_BELIEFS      # 'explicit' | 'departures'
    trace_version: str = TRACE_VERSION


def teacher_tag(teacher):
    """A file-name-safe tag for a teacher: 'hf-qwen3-32b', 'claude-opus-5'."""
    name = teacher[3:] if teacher.startswith('hf:') else teacher
    name = name.split('@')[0].split('/')[-1].lower()
    name = re.sub(r'[^a-z0-9.]+', '-', name).strip('-')
    return ('hf-' if teacher.startswith('hf:') else '') + name


def distill_file(tcfg):
    return os.path.join(HERE, f'traces_distill_{teacher_tag(tcfg.teacher)}.txt')


def _file_sha(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()[:16]


def target_spec(cond, tcfg, dcfg):
    """
    What identifies a condition's training targets in a cache key: the exact
    conditions by the target version and the training rows; the distill
    condition by the teacher and the digest of its trace file too.
    """
    # `cache_cond` keys a renamed condition under its old name (key_steps ->
    # chain); TARGET_VERSIONS is still read by the current name (same version).
    spec = dict(task='exploretom-targets', cond=cache_cond(cond), data=train_spec(dcfg),
                trace_version=TARGET_VERSIONS.get(cond, tcfg.trace_version))
    if cond == 'state_tracking':
        spec['stride'] = tcfg.focus_stride
        spec['beliefs'] = tcfg.focus_beliefs
    if cond == 'distill':
        spec.update(teacher=tcfg.teacher, distill_sha=_file_sha(distill_file(tcfg)))
    return spec


# ----------------------------------------------------------------------------
# the trace file
# ----------------------------------------------------------------------------

def parse_records(text):
    out, cur, buf = [], None, []
    for ln in text.split('\n'):
        if ln.startswith('### '):
            if cur is not None:
                out.append((cur, '\n'.join(buf).strip()))
            cur, buf = ln.strip(), []
        elif cur is not None:
            buf.append(ln.rstrip())
    if cur is not None:
        out.append((cur, '\n'.join(buf).strip()))
    return out


def read_distill_file(path):
    """``{row: (qhash, failed, text)}``; raises on a bad header or a duplicate."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        recs = parse_records(f.read())
    out = {}
    for header, body in recs:
        m = _HEADER.match(header)
        if not m:
            raise ValueError(f'{os.path.basename(path)}: bad record header {header!r}')
        idx = int(m.group(1))
        if idx in out:
            raise ValueError(f'{os.path.basename(path)}: row {idx} appears twice')
        out[idx] = (m.group(2), bool(m.group(3)), body)
    return out


_write_lock = threading.Lock()


def append_record(path, header, body):
    with _write_lock:
        with open(path, 'a') as f:
            f.write(f'### {header}\n{body.strip()}\n\n')
            f.flush()
            os.fsync(f.fileno())


def validate_distill(text, problem, max_chars=4000):
    """Usable if the last line is an answer field with the label, no header inside, not too long."""
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if not lines:
        return False, 'empty'
    pred, answered = extract_answer(lines[-1])
    if not answered:
        return False, f'last line {lines[-1][:40]!r} is not an answer field'
    if pred != normalise(problem.gold):
        return False, f'teacher answered {pred!r}, label {problem.gold!r}'
    if any(ln.startswith('###') for ln in lines):
        return False, 'contains a record header'
    if len(text) > max_chars:
        return False, f'{len(text)} chars'
    return True, ''


def load_distill(problems, tcfg):
    """The usable distill trace of every row that has one, as ``{idx: text}``,
    the answer line normalised to the label's."""
    recs = read_distill_file(distill_file(tcfg))
    out, bad = {}, []
    for p in problems:
        if p.idx not in recs:
            continue
        qh, failed, text = recs[p.idx]
        if qh != question_hash(p):
            bad.append(f'row {p.idx}: hash {qh} != {question_hash(p)} (data changed?)')
            continue
        if failed:
            continue
        ok, why = validate_distill(text, p)
        if not ok:
            bad.append(f'row {p.idx}: {why}')
            continue
        lines = [ln for ln in text.split('\n') if ln.strip()]
        lines[-1] = answer_line(p.gold)
        out[p.idx] = '\n'.join(lines)
    if bad:
        raise ValueError(f'{len(bad)} unusable records in {os.path.basename(distill_file(tcfg))}:'
                         '\n  ' + '\n  '.join(bad[:8]) + (f'\n  (+{len(bad) - 8} more)' if len(bad) > 8 else ''))
    return out


# ----------------------------------------------------------------------------
# training rows and targets
# ----------------------------------------------------------------------------

def _walk(dcfg, tcfg, target=None, retry_failed=False):
    """
    Walk the training pool in order and sort its rows into the first `target`
    (default `n_train`) rows with a usable distill trace, the candidates that
    could still get one (never asked, or FAILED with `retry_failed`) and the
    rows the teacher failed on.
    """
    target = dcfg.n_train if target is None else target
    pool = train_pool(dcfg)
    traces = load_distill(pool, tcfg)
    recs = read_distill_file(distill_file(tcfg))
    rows, candidates, failed = [], [], []
    for p in pool:
        if p.idx in traces:
            if len(rows) < target:
                rows.append(p)
            continue
        asked = p.idx in recs
        if asked:
            failed.append(p)
        if not asked or retry_failed:
            candidates.append(p)
    return dict(rows=rows, candidates=candidates, failed=failed, traces=traces,
                n_asked=sum(p.idx in recs for p in pool), n_failed=len(failed))


def coverage(dcfg, tcfg):
    w = _walk(dcfg, tcfg)
    return dict(n_rows=len(w['rows']), n_train=dcfg.n_train, n_asked=w['n_asked'],
                n_failed=w['n_failed'], n_candidates=len(w['candidates']))


def training_rows(dcfg, tcfg):
    """The distill condition's rows and traces: ``(problems, {idx: text})``."""
    w = _walk(dcfg, tcfg)
    keep = sorted(w['rows'], key=lambda p: p.idx)
    if not keep:
        raise FileNotFoundError(
            f'no training rows have a distill trace for teacher {tcfg.teacher!r}: build them '
            f'with `python traces.py --build --teacher {tcfg.teacher}`')
    return keep, {p.idx: w['traces'][p.idx] for p in keep}


def target_fn(cond, tcfg):
    """The ``problem -> assistant text`` of an exact condition under `tcfg`."""
    if cond == 'state_tracking':
        return lambda p: state_tracking_target(p, tcfg.focus_stride, tcfg.focus_beliefs)
    return TARGETS[cond]


def training_examples(cond, dcfg, tcfg):
    """
    The training examples of a condition as ``[(problem, target)]`` -- one per
    training row.
    """
    if cond in EXACT:
        fn = target_fn(cond, tcfg)
        return [(p, fn(p)) for p in get_problems(dcfg, 'train')]
    if cond == 'distill':
        keep, traces = training_rows(dcfg, tcfg)
        return [(p, traces[p.idx]) for p in keep]
    raise ValueError(f'condition {cond!r} is not trained')


def trace_summary(dcfg, tcfg):
    """Coverage and size of the distill traces, or None if unusable."""
    try:
        keep, traces = training_rows(dcfg, tcfg)
    except (FileNotFoundError, ValueError):
        return None
    words = sum(len(t.split()) for t in traces.values()) / max(len(traces), 1)
    return dict(coverage(dcfg, tcfg), distill_words=words)


# ----------------------------------------------------------------------------
# the teacher
# ----------------------------------------------------------------------------

DISTILL_SYSTEM = SYSTEM_PROMPT + (
    ' Reason in plain prose: a few sentences that follow who was where, who saw '
    'or heard what, and what that implies for the question. No markdown, lists or '
    'headings; at most 150 words. Then the answer field on its own last line.'
)

_THINK = re.compile(r'<think>.*?</think>\s*', re.S)


def strip_thinking(text):
    if '<think>' in text and '</think>' not in text:
        return ''
    return _THINK.sub('', text).strip()


class ClaudeTeacher:
    def __init__(self, model, workers=4, max_tokens=4096):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=5)
        self.model, self.workers, self.max_tokens = model, workers, max_tokens

    def _one(self, system, user):
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                        system=system, messages=[{'role': 'user', 'content': user}])
        return '\n'.join(b.text for b in r.content if b.type == 'text').strip()

    def generate(self, jobs, sample, on_result):
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self._one, s, u): i for i, (s, u) in enumerate(jobs)}
            for f in as_completed(futs):
                try:
                    on_result(futs[f], f.result())
                except Exception as e:                  # noqa: BLE001
                    print(f'  request failed: {e!r}', file=sys.stderr, flush=True)
                    on_result(futs[f], None)


class HFTeacher:
    def __init__(self, spec, device='cuda', batch_size=8, max_new_tokens=None, thinking=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name, _, rev = spec[3:].partition('@')
        rev = rev or HF_REVISIONS.get(name)
        self.tok = AutoTokenizer.from_pretrained(name, revision=rev)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = 'left'
        dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=dtype)
        self.model.to(device).eval()
        self.device, self.batch_size, self.thinking = device, batch_size, thinking
        self.max_new_tokens = max_new_tokens or (8192 if thinking else 1024)

    def generate(self, jobs, sample, on_result):
        import torch
        from tqdm.auto import tqdm
        prompts = [self.tok.apply_chat_template(
            [{'role': 'system', 'content': s}, {'role': 'user', 'content': u}],
            tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking)
            for s, u in jobs]
        order = sorted(range(len(jobs)), key=lambda i: -len(prompts[i]))
        for s0 in tqdm(range(0, len(order), self.batch_size), desc='teacher', leave=False):
            idx = order[s0:s0 + self.batch_size]
            enc = self.tok([prompts[i] for i in idx], return_tensors='pt', padding=True,
                           add_special_tokens=False).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                          do_sample=sample, temperature=0.7 if sample else None,
                                          top_p=0.9 if sample else None, top_k=None,
                                          pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc['input_ids'].shape[1]:]
            for j, i in enumerate(idx):
                on_result(i, strip_thinking(self.tok.decode(new[j], skip_special_tokens=True)))


def make_teacher(teacher, device='cuda', workers=4, batch_size=8, max_new_tokens=None,
                 thinking=False):
    if teacher.startswith('hf:'):
        return HFTeacher(teacher, device=device, batch_size=batch_size,
                         max_new_tokens=max_new_tokens, thinking=thinking)
    return ClaudeTeacher(teacher, workers=workers, max_tokens=max_new_tokens or 4096)


# ----------------------------------------------------------------------------
# building
# ----------------------------------------------------------------------------

def _ask(problems, tcfg, teacher, retry_failed, log):
    """Ask the teacher about `problems`; write usable replies, retry the rest once."""
    path = distill_file(tcfg)
    have = read_distill_file(path)
    items = {p.idx: p for p in problems if p.idx not in have or (retry_failed and have[p.idx][1])}
    header = lambda p, failed: f'row {p.idx} q={question_hash(p)}' + (' FAILED' if failed else '')  # noqa: E731
    stale = {'### ' + header(p, True) for k, p in items.items() if k in have and have[k][1]}
    if retry_failed and stale:
        with open(path) as f:
            keep = [(h, b) for h, b in parse_records(f.read()) if h not in stale]
        with open(path, 'w') as f:
            for h, b in keep:
                f.write(f'{h}\n{b}\n\n')
    log(f'[distill] {len(have)} records on disk, {len(items)} rows to ask {tcfg.teacher}')
    if not items:
        return
    keys = list(items)
    pending, n_ok, t0 = set(keys), 0, time.time()
    for attempt, sample in ((1, False), (2, True)):
        todo = [k for k in keys if k in pending]
        if not todo:
            break
        log(f'[distill] attempt {attempt}: {len(todo)} rows')

        def on_result(j, text, todo=todo):
            nonlocal n_ok
            k = todo[j]
            if text is None:
                return
            ok, why = validate_distill(text, items[k])
            if ok:
                append_record(path, header(items[k], False), text)
                pending.discard(k)
                n_ok += 1
                if n_ok % 100 == 0:
                    log(f'[distill] {n_ok} written, {len(pending)} pending ({time.time() - t0:.0f}s)')
            elif attempt == 2:
                append_record(path, header(items[k], True), text)
                pending.discard(k)

        teacher.generate([(DISTILL_SYSTEM, user_message(items[k])) for k in todo], sample, on_result)
    log(f'[distill] wrote {n_ok} usable records in {(time.time() - t0) / 60:.1f} min'
        + (f'; {len(pending)} rows got no reply (request errors)' if pending else ''))


def build(dcfg, tcfg, teacher, retry_failed=False, limit=None, max_rounds=50, log=print):
    """
    Extend the distill traces until `target` rows (`limit`, default `n_train`)
    of the pool have a usable trace, asking about the nominal training rows
    first and then, for every failed row, about the next rows of the pool.
    """
    target = limit or dcfg.n_train
    for round_ in range(max_rounds):
        retrying = retry_failed and round_ == 0
        w = _walk(dcfg, tcfg, target, retry_failed=retrying)
        need = target - len(w['rows'])
        retry = w['failed'] if retrying else []
        skip = {p.idx for p in retry}
        fresh = [p for p in w['candidates'] if p.idx not in skip][:max(need, 0)]
        if need <= 0 and not retry:
            log(f'{len(w["rows"])} rows have usable distill traces (target {target}, '
                f'{w["n_asked"]} asked, {w["n_failed"]} failed)')
            return
        if not fresh and not retry:
            log(f'pool exhausted: {len(w["rows"])} of {target} rows have usable traces '
                f'({w["n_asked"]} asked, {w["n_failed"]} failed)')
            return
        log(f'round {round_ + 1}: {len(w["rows"])}/{target} rows covered; asking about '
            f'{len(fresh)} new rows' + (f' and retrying {len(retry)} failed' if retry else ''))
        _ask(retry + fresh, tcfg, teacher, retrying, log)
    log(f'stopped after {max_rounds} rounds')


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--teacher', default=TEACHER,
                    help=f'hf:<hub id>[@revision] for a local model (default {TEACHER}), or an '
                         'Anthropic model id')
    ap.add_argument('--n-train', type=int, default=DataConfig.n_train)
    ap.add_argument('--seed', type=int, default=DataConfig.seed)
    ap.add_argument('--build', action='store_true', help='write the missing records')
    ap.add_argument('--retry-failed', action='store_true', help='retry the FAILED records too')
    ap.add_argument('--limit', type=int, default=None, help='target this many rows instead')
    ap.add_argument('--device', default='cuda', help='hf teacher: torch device')
    ap.add_argument('--batch', type=int, default=8, help='hf teacher: decode batch')
    ap.add_argument('--thinking', action='store_true', help="hf teacher: thinking mode on")
    ap.add_argument('--max-new-tokens', type=int, default=None)
    ap.add_argument('--workers', type=int, default=4, help='API teacher: concurrent requests')
    ap.add_argument('--check', action='store_true', help='validate the file, print statistics')
    ap.add_argument('--show', type=int, default=0, metavar='N', help='print N rows with all targets')
    args = ap.parse_args()

    dcfg = DataConfig(n_train=args.n_train, seed=args.seed)
    tcfg = TraceConfig(teacher=args.teacher)
    if args.build:
        teacher = make_teacher(args.teacher, device=args.device, workers=args.workers,
                               batch_size=args.batch, max_new_tokens=args.max_new_tokens,
                               thinking=args.thinking)
        build(dcfg, tcfg, teacher, retry_failed=args.retry_failed, limit=args.limit)
    if args.show:
        keep, traces = training_rows(dcfg, tcfg)
        for p in keep[:args.show]:
            print('=' * 72)
            print(user_message(p))
            for name in ('direct', 'key_steps', 'state_tracking', 'narration'):
                print('-' * 26 + f' {name} ' + '-' * 26)
                print(target_fn(name, tcfg)(p))
            print('-' * 26 + ' distill ' + '-' * 25)
            print(traces[p.idx])
    if args.check or not (args.show or args.build):
        s = trace_summary(dcfg, tcfg)
        if s is None:
            try:
                training_rows(dcfg, tcfg)
            except (FileNotFoundError, ValueError) as e:
                sys.exit(f'ERROR: {e}')
        print(f'{s["n_rows"]} of {s["n_train"]} training rows have a distill trace (teacher '
              f'{tcfg.teacher}): {s["n_asked"]} rows asked, {s["n_failed"]} unusable, '
              f'{s["n_candidates"]} more could be asked; {s["distill_words"]:.0f} words per trace')


if __name__ == '__main__':
    main()
