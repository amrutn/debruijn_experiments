"""
Do the three derivation scorings accept the same traces?

Three metrics judge the same decoded trace, differing only in what they do with the
injected steps: `D` keeps them, `I` drops them, `R` replaces each with the correct
operation for that point. All three are subsets of the answer-correct set, since
`derivation_correct` requires `claimed == truth`, so their sizes alone bound the
overlap from below by |X n Y| >= X + Y - A. That bound is informative at low
injection probability and vacuous at high p -- exactly where the question matters,
since the standard condition scores 0.358 answer accuracy at p = 0.5 against 0.024
derivation and 0.062 ignored accuracy, and those two small sets could in principle
be disjoint.

The exclusive counts are what carry the meaning, because the three differ in how
they can fail:

  D but not I    the plan only works with the foreign steps counted -- the model
                 folded the injection in
  I but not D    the plan works once the injection is removed -- it was ignored
  in R, not I    the model's own steps were right but `I` was missing a step,
                 which is the hole `I` leaves and `R` closes
  in D, not R    almost always the length cap: the trace ran longer than the
                 canonical solution

This re-scores the cached decodes per problem instead of in aggregate, so it needs
no GPU and no retraining. Nothing here writes to the existing caches.

    python overlap.py                          # every mode, every (k, p)
    python overlap.py --mode samples --cond std --p 0.2 0.3 0.5
"""

import os
import json
import argparse

from generate_data import (derivation_correct, derivation_correct_replaced,
                           parse_answers, is_correct)
from train_eval import _decode_path, get_problems
import run_experiments as R


def score_unit(problems, records):
    """
    Per-problem membership in the three sets, plus the injection count.

    Returns (answer, derivation, ignored, replaced, n_inj) as parallel lists,
    scored exactly as `train_eval` and `run_experiments.derivation_results` score
    them so the totals reproduce the published tables.

    `n_inj` is what makes the conditional view possible. Injection is Bernoulli per
    operation step, so at moderate p a large share of traces receive no injection at
    all -- those are trivially in both D and I or in neither, and they dominate the
    unconditional overlap. Restricting to traces that actually took an injection is
    the comparison that carries information.
    """
    a, d, i, r, n_inj = [], [], [], [], []
    for prob, rec in zip(problems, records):
        steps = [x for x in rec if x['kind'] == 'op']
        ops_all = [x['text'] for x in steps]
        flags = [x['injected'] for x in steps]
        ops_own = [t for t, f in zip(ops_all, flags) if not f]
        ans_step = next((x['text'] for x in rec if x['kind'] == 'answer'), '')
        claimed = parse_answers(ans_step)
        a.append(bool(is_correct(prob, ans_step)))
        d.append(bool(derivation_correct(prob, ops_all, claimed)))
        i.append(bool(derivation_correct(prob, ops_own, claimed)))
        r.append(bool(derivation_correct_replaced(prob, ops_all, flags, claimed)))
        n_inj.append(len(ops_all) - len(ops_own))
    return a, d, i, r, n_inj


def calibrate(n=400, seed=21):
    """
    Score a *perfect* model: canonical steps, with only the injections corrupting.

    This is the baseline the real numbers have to be read against, and without it
    two of the three metrics are easy to misread. A flawless plan still scores near
    zero on D and I at a high injection probability -- D because the foreign
    operation is replayed, I because deleting it removes a step the derivation
    needed -- so a low D or I is not by itself evidence of anything about the
    model. It also shows that `I - D > 0` is the *normal* sign for a model that
    does not ignore anything, which is why only the negative gap is diagnostic.

    R is the metric that survives: it stays near 1.0 at every injection level,
    because closing the hole leaves nothing but the model's own operations to
    judge.
    """
    import random
    import generate_data as G

    pool = G.make_dataset(n, seed=seed, min_ops=3, max_ops=8, progress=False)
    rng = random.Random(7)
    print('\n=== calibration: a perfect model under injection ===')
    print('  the ceiling each metric can reach when the PLAN is flawless and only')
    print('  the injections corrupt the trace; read the real tables against this')
    print()
    print('     p       A       D       I       R   inj/trace')
    for p in R.INJECT_PS:
        recs = []
        for prob in pool:
            rec = []
            for op in prob.ops:
                inj = (not op.lower().startswith('write the equation')
                       and rng.random() < p)
                rec.append(dict(text=G.random_injection(rng) if inj else op,
                                kind='op', injected=inj))
            rec.append(dict(text=G.render_answers(prob.answers),
                            kind='answer', injected=False))
            recs.append(rec)
        a, d, i, r, n_inj = score_unit(pool, recs)
        m = len(pool)
        print(f'  {p:<5g}  {sum(a)/m:6.3f}  {sum(d)/m:6.3f}  {sum(i)/m:6.3f}  '
              f'{sum(r)/m:6.3f}  {sum(n_inj)/m:9.2f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calibrate', action='store_true',
                    help='score a perfect model to get each metric\'s ceiling')
    ap.add_argument('--mode', choices=list(R.BUDGET_MODES), default=None)
    ap.add_argument('--cond', nargs='*', default=None,
                    help="conditions to score, e.g. 1 5 std (default: all)")
    ap.add_argument('--p', nargs='*', type=float, default=None,
                    help='injection probabilities (default: all)')
    args = ap.parse_args()

    if args.calibrate:
        calibrate()

    def wanted(k):
        if args.cond is None:
            return True
        return ('std' if k is None else str(k)) in args.cond

    ks = [k for k in R.INTERVALS if wanted(k)]
    ps = args.p if args.p is not None else R.INJECT_PS

    for mode in ([args.mode] if args.mode else list(R.BUDGET_MODES)):
        problems = get_problems(R.data_cfg(mode), 'test')
        rows = []
        for k in ks:
            TR = R.train_cfg(k, mode)
            for p in ps:
                path = _decode_path(k, p, R.data_cfg(mode), R.MODEL, TR, R.EVAL)
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    records = json.load(f)['records']
                a, d, i, r, n_inj = score_unit(problems, records)
                for only_injected in (False, True):
                    sel = [j for j in range(len(a))
                           if n_inj[j] > 0 or not only_injected]
                    if not sel:
                        continue
                    aa = [a[j] for j in sel]
                    S = {'D': [d[j] for j in sel], 'I': [i[j] for j in sel],
                         'R': [r[j] for j in sel]}
                    n = len(sel)
                    row = dict(
                        cond='std' if k is None else str(k), p=p, n=n,
                        only_injected=only_injected,
                        # share of ALL traces that took no injection -- the
                        # population the unconditional numbers are diluted by
                        zero_frac=sum(x == 0 for x in n_inj) / max(len(a), 1),
                        A=sum(aa) / n,
                        union=sum(x or y or z for x, y, z in
                                  zip(S['D'], S['I'], S['R'])) / n,
                        # measured, not reconstructed: deriving it from the
                        # pairwise columns sums seven rounded terms and the error
                        # swamps a triple this small
                        tri=sum(x and y and z for x, y, z in
                                zip(S['D'], S['I'], S['R'])) / n,
                        # containment is a claim about the code, so check it rather
                        # than assume it: every sound trace must be answer-correct
                        leak=sum((x or y or z) and not w for x, y, z, w in
                                 zip(S['D'], S['I'], S['R'], aa)),
                    )
                    for name, v in S.items():
                        row[name] = sum(v) / n
                    for x, y in (('D', 'I'), ('D', 'R'), ('I', 'R')):
                        row[f'{x}\\{y}'] = sum(u and not v for u, v in
                                               zip(S[x], S[y])) / n
                        row[f'{y}\\{x}'] = sum(v and not u for u, v in
                                               zip(S[x], S[y])) / n
                    rows.append(row)
        if not rows:
            print(f'[{mode}] no cached decodes')
            continue

        print(f'\n=== {mode} ===')
        print('  A  answer accuracy      D  derivation: the sequence as decoded')
        print('  I  ignored: injected steps dropped   (leaves a hole in the plan)')
        print('  R  replaced: injected steps corrected (closes the hole; length-capped)')
        print('  DIR  |D n I n R|: traces every scoring accepts')
        print('  A-U  answer-correct but sound under none of the three')
        print('  X\\Y  in X and not Y; a near-zero column means X is contained in Y')

        for only_injected in (False, True):
            block = [r for r in rows if r['only_injected'] == only_injected]
            if not block:
                continue
            if only_injected:
                print('\n  --- traces with >=1 injected step only ---')
                print('  Traces that took no injection have D, I and R identical by')
                print('  construction, so they inflate the agreement above. This block')
                print('  drops them; n is what survives, and zero% is the share dropped.')
            else:
                print('\n  --- all traces ---')
            print()
            print('  cond    p       n   zero%       A       D       I       R'
                  '     DIR     A-U     D\\I     I\\D     D\\R     R\\D     I\\R     R\\I')
            for r in block:
                print(f"  {r['cond']:>4}  {r['p']:<5g}  {r['n']:>4}  {r['zero_frac']:5.1%}  "
                      f"{r['A']:6.3f}  {r['D']:6.3f}  {r['I']:6.3f}  {r['R']:6.3f}  "
                      f"{r['tri']:6.3f}  {r['A'] - r['union']:6.3f}  "
                      + '  '.join(f"{r[c]:6.3f}" for c in
                                  ('D\\I', 'I\\D', 'D\\R', 'R\\D', 'I\\R', 'R\\I')))
        bad = [r for r in rows if r['leak']]
        print('\n  containment check (every sound trace is answer-correct): '
              + ('OK' if not bad else f'VIOLATED in {len(bad)} unit(s)'))


if __name__ == '__main__':
    main()
