"""
Why does a condition receive more injections than another at the same p?

Injections fire on *operation* steps only, so ``injections/trace = p x (operation
steps emitted)``. If two conditions differ in injections per trace at the same p,
it is because one of them is emitting more operations than the other -- and there
are two very different reasons that can happen:

  derailment      the model keeps operating on a corrupted equation, taking extra
                  steps before it reaches an answer, but still terminates
  non-termination the model never emits ``####`` and runs to the `max_steps` cap,
                  which inflates the operation count mechanically

Non-termination is graded as a wrong answer, so it silently inflates the error
rate: an accuracy of 0.012 mostly made of traces that never answered is a very
different result from one made of wrong answers. `cap/err` is the share of a
condition's errors that are the former.

This script reads the cached decodes and separates them: it reports operation
steps, state reports, how often the trace terminated with an answer, and how often
it hit the step cap, for every (interval, inject_p) already on disk. Nothing is
recomputed and no GPU is touched.

    python trace_stats.py                 # both budget modes
    python trace_stats.py --mode compute
"""

import os
import json
import argparse

from train_eval import _decode_path, cached_unit
import run_experiments as R


def summarise(records, ecfg):
    """Per-trace step counts and termination behaviour for one decode."""
    n = max(len(records), 1)
    ops = states = answered = capped = 0
    for rec in records:
        n_op = sum(1 for r in rec if r['kind'] == 'op')
        n_st = sum(1 for r in rec if r['kind'] == 'state')
        ops += n_op
        states += n_st
        has_ans = any(r['kind'] == 'answer' for r in rec)
        answered += has_ans
        # no answer and the step budget is exhausted -> ran to the cap
        capped += (not has_ans) and (len(rec) >= ecfg.max_steps - 1)
    return dict(ops=ops / n, states=states / n,
                answered=answered / n, capped=capped / n, n=n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=list(R.BUDGET_MODES), default=None)
    args = ap.parse_args()

    for mode in ([args.mode] if args.mode else list(R.BUDGET_MODES)):
        rows = []
        # data_cfg, not DATA: the recovery modes augment, which changes the
        # dataset half of the decode key. Using DATA silently found nothing there.
        dcfg = R.data_cfg(mode)
        for k in R.CONDITIONS:
            TR = R.train_cfg(k, mode)
            for p in R.inject_ps_for(k):
                path = _decode_path(k, p, dcfg, R.MODEL, TR, R.EVAL)
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    blob = json.load(f)
                st = summarise(blob['records'], R.EVAL)
                unit = cached_unit(k, p, dcfg, R.MODEL, TR, R.EVAL)
                st['accuracy'] = unit['accuracy'] if unit else None
                rows.append(((k, p), st))
        if not rows:
            print(f'[{mode}] no cached decodes')
            continue

        print(f'\n=== {mode} ===')
        print('  ops      : operation steps per trace  (injections = p x this)')
        print('  states   : state reports per trace')
        print('  answered : fraction of traces that emitted ####')
        print('  capped   : fraction that ran to the max_steps cap without answering')
        print('  stopped  : gave up early without an answer (not the cap)')
        print('  acc      : answer accuracy, for reference')
        print('  cap/err  : share of the errors that are non-termination rather')
        print('             than a wrong answer -- capped / (1 - accuracy)')
        print()
        print('  cond    p       ops   states   answered   capped  stopped'
              '      acc   cap/err')
        for (k, p), s in rows:
            a = s['accuracy']
            stopped = max(1.0 - s['answered'] - s['capped'], 0.0)
            share = (f'{s["capped"] / (1 - a):7.1%}'
                     if a is not None and a < 1 else '      -')
            acc = f'{a:7.3f}' if a is not None else '      -'
            print(f'  {R._cond_label(k):>4}  {p:<5g}  {s["ops"]:6.2f}  {s["states"]:6.2f}   '
                  f'{s["answered"]:8.3f}  {s["capped"]:7.3f}  {stopped:7.3f}  '
                  f'{acc}  {share}')


if __name__ == '__main__':
    main()
