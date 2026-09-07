"""
Where does the state-tracking model go wrong? Compare the state lines a
fine-tuned model wrote at test time with the exact ones, from the cached
evaluations.

For every cached evaluation of a state-line condition ('state_tracking'; the
legacy 'focused'/'ledger' names still work on old caches) the script aligns
the model's ``State:`` lines with the gold lines of the same row (at the
stride and belief mode the run used), and reports:

* how many rows reproduce the final state line exactly, split into the
  world part and the beliefs part;
* for false-belief rows, whether the belief clauses the gold final state
  carries were *omitted* (the model wrote fewer clauses and nothing wrong),
  *wrong* (it wrote clauses the gold line does not have) or *matched*, and
  the accuracy in each bucket -- the test of the "omitted departure"
  diagnosis;
* at which state line the model first departs from the gold ledger;
* a few examples of omissions.

    python analyze_ledger.py                     # every state_tracking eval in the cache
    python analyze_ledger.py --stride 1 --examples 5
"""

import os
import re
import json
import glob
import argparse
import collections

from exploretom_data import all_problems, blocks, STATE_PREFIX, CACHE

EVAL_CACHE = os.path.join(CACHE, 'evals')


def state_lines(text):
    return [ln.strip() for ln in text.split('\n') if ln.strip().startswith(STATE_PREFIX)]


def split_state(line):
    """(world clauses, belief clauses) of a state line, as sets of clause strings."""
    body = line[len(STATE_PREFIX):].strip()
    if body.startswith('nothing relevant yet'):
        return set(), set()
    world, _, beliefs = body.partition(' Beliefs: ')
    clauses = lambda s: {c.strip().rstrip('.') for c in s.split(';') if c.strip().rstrip('.')}   # noqa: E731
    bel = set() if beliefs.strip().rstrip('.') in ('as stated', 'none yet', '') else clauses(beliefs)
    return clauses(world.rstrip('.')), bel


def gold_states(p, cond, beliefs, stride):
    if cond == 'ledger':                     # legacy full ledger
        states = p.states
    else:                                    # 'state_tracking' (or legacy 'focused')
        states = p.focused_explicit if beliefs == 'explicit' else p.focused_states
    return [states[e - 1] for e in blocks(len(p.steps), stride)]


def analyse(path, problems, stride, beliefs, n_examples):
    with open(path) as f:
        r = json.load(f)
    cond = r['cond']
    rows = r['samples']
    by_idx = {p.idx: p for p in problems}
    c = collections.Counter()
    first_div = collections.Counter()
    buckets = {True: collections.Counter(), False: collections.Counter()}
    correct = {True: collections.Counter(), False: collections.Counter()}
    examples = []
    for s in rows:
        p = by_idx[s['idx']]
        gold = gold_states(p, cond, beliefs, stride)
        gen = state_lines(s['text'])
        ok = s['pred'] is not None and s['pred'] == p.gold.lower()
        c['rows'] += 1
        c['correct'] += ok
        c['n_gold_lines'] += len(gold)
        c['n_gen_lines'] += len(gen)
        if not gen:
            c['no_state_line'] += 1
            buckets[p.false_belief]['no state'] += 1
            correct[p.false_belief]['no state'] += ok
            continue
        # first divergence, aligned by order
        div = next((i for i, (g, h) in enumerate(zip(gold, gen)) if g != h), None)
        if div is None and len(gen) != len(gold):
            div = min(len(gold), len(gen))
        first_div['none' if div is None else div + 1] += 1
        gw, gb = split_state(gold[-1])
        hw, hb = split_state(gen[-1])
        c['final_exact'] += gen[-1] == gold[-1]
        c['final_world'] += gw == hw
        c['final_beliefs'] += gb == hb
        if hb == gb:
            kind = 'match'
        elif hb < gb:
            kind = 'omitted'
        else:
            kind = 'wrong'
        buckets[p.false_belief][kind] += 1
        correct[p.false_belief][kind] += ok
        if kind == 'omitted' and p.false_belief and len(examples) < n_examples:
            examples.append((p, gold[-1], gen[-1], ok))
    n = max(c['rows'], 1)
    print(f'\n{cond} ({os.path.basename(path)}): {c["rows"]} rows, accuracy {c["correct"] / n:.3f}')
    print(f'  state lines per reply: gold {c["n_gold_lines"] / n:.2f}, written {c["n_gen_lines"] / n:.2f}; '
          f'replies without a state line: {c["no_state_line"]}')
    print(f'  final state exact {c["final_exact"] / n:.3f}; world part {c["final_world"] / n:.3f}; '
          f'beliefs part {c["final_beliefs"] / n:.3f}')
    print('  first state line that differs from gold: '
          + ', '.join(f'{k}: {v}' for k, v in sorted(first_div.items(), key=lambda kv: (kv[0] == "none", kv[0]))))
    for fb in (True, False):
        label = 'false-belief rows' if fb else 'true-belief / factual rows'
        parts = []
        for kind in ('match', 'omitted', 'wrong', 'no state'):
            if buckets[fb][kind]:
                parts.append(f'{kind} {buckets[fb][kind]} (acc {correct[fb][kind] / buckets[fb][kind]:.2f})')
        print(f'  {label} ({sum(buckets[fb].values())}): beliefs part ' + ', '.join(parts))
    for p, g, h, ok in examples:
        print(f'\n  example: {p.question}  [gold {p.gold}; model {"right" if ok else "wrong"}]')
        print(f'    gold : {g}')
        print(f'    model: {h}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stride', type=int, default=1, help='the state stride the run used')
    ap.add_argument('--beliefs', choices=('explicit', 'departures'), default='explicit',
                    help='the belief mode the run used (state_tracking default: explicit)')
    ap.add_argument('--conds', default='state_tracking')
    ap.add_argument('--examples', type=int, default=3)
    args = ap.parse_args()
    problems = all_problems()
    conds = set(args.conds.split(','))
    seen = 0
    for path in sorted(glob.glob(os.path.join(EVAL_CACHE, '*.json'))):
        with open(path) as f:
            head = f.read(200)
        m = re.search(r'"cond":\s*"(\w+)"', head)
        if m and m.group(1) in conds:
            analyse(path, problems, args.stride, args.beliefs, args.examples)
            seen += 1
    if not seen:
        print(f'no cached evaluation for {sorted(conds)} in {EVAL_CACHE}')


if __name__ == '__main__':
    main()
