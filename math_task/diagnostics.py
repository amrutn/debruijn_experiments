"""
Two diagnostics that separate *why* a condition survives per-step injection.

The headline result is that clean accuracy and injected accuracy rank the
conditions oppositely: the models with the most state in context (small k) score
highest clean and lowest under injection. There are two very different readings
of that, and these diagnostics tell them apart.

D1 -- plan replay ("is it just ignoring the injection?")
--------------------------------------------------------
Every injected operation preserves the solution set, so a model that simply
*ignores* the injection and carries on with the plan it would have followed
anyway still lands on the correct answer. The standard condition never writes the
current equation, so nothing forces it to notice an injection at all: its apparent
robustness could be indifference rather than understanding.

D1 decodes each test problem twice -- once clean, once injected -- and measures
how much of the clean plan survives, as the longest-common-subsequence ratio
between the model's own operations (injected steps excluded) and the operations
it emitted on the clean run. A ratio near 1 means the model reproduced its clean
plan verbatim and therefore ignored the perturbation; a low ratio means it
genuinely re-planned. `answer_agree` is the companion: how often the injected run
ends on the same answer as the clean run.

D2 -- state tracking ("did it actually apply the injected operation?")
----------------------------------------------------------------------
For k >= 1 the model writes ``Current equation is ...`` as it goes. D2 replays
the decoded trace symbolically with sympy -- applying each operation, injected or
not, to the true equation -- and checks whether the model's stated equation
matches. `post_inj_acc` is restricted to the state report immediately after an
injected operation, which is the moment that actually tests recovery.

`state_tracking` scores each report against two ground truths replayed side by
side -- one that re-anchors on the model's claim, one that never does -- and
returns both. D2 itself reports only the anchored figure; the pair is used by
`run_experiments.state_results`, which measures them on the clean decodes as a
separate pass so that adding it does not disturb this cache.

D2 has no intermediate reports for the standard condition, which never writes its
state -- an asymmetry worth stating rather than hiding.

Results are cached per (interval, inject_p) under ``cache/diagnostics``.
"""

import os
import json
from dataclasses import asdict

from tqdm.auto import tqdm

from generate_data import (
    STATE_PREFIX, parse_equation, apply_op, equations_equal, parse_answers,
)
from train_eval import (
    CACHE, _key, _data_spec, _train_spec, get_problems, load_for_eval,
    train_adapter, get_or_decode,
)

DIAG_CACHE = os.path.join(CACHE, 'diagnostics')


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def _lcs(a, b):
    """Length of the longest common subsequence of two lists."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def plan_replay(clean_records, inj_records):
    """
    How much of the clean plan the model reproduced despite the injections.

    Returns (ratio, n_injected), where ratio is the LCS of the model's own
    operations against its clean-run operations, normalised by the clean length.
    Positions are compared as a subsequence rather than index-by-index, because an
    injected step displaces the model's own step and shifts everything after it.
    None if the injected run contained no injection.
    """
    n_inj = sum(1 for r in inj_records if r['injected'])
    if n_inj == 0:
        return None, 0
    clean_ops = [r['text'] for r in clean_records if r['kind'] == 'op']
    model_ops = [r['text'] for r in inj_records if r['kind'] == 'op' and not r['injected']]
    if not clean_ops:
        return None, n_inj
    return _lcs(model_ops, clean_ops) / len(clean_ops), n_inj


def state_tracking(problem, records):
    """
    Whether the model's stated equations match the true equation at each point.

    Two equations are replayed side by side from the problem, both applying the
    model's *own* operations, so both judge state computation rather than choice
    of operation. They differ only in what they treat as ground truth:

    anchored     after each report the replay adopts whatever the model claimed.
                 Each score is then "given where you said you were, did you apply
                 the next k operations correctly" -- local consistency, and one
                 early slip cannot mark every later report wrong.
    cumulative   the replay never adopts the model's claims, so ground truth is
                 the true consequence of the operations written so far. This asks
                 whether the model is still *on* the true trajectory.

    The gap between them is error propagation, and its *sign* is the diagnostic:

      cumulative > anchored   the model recovers. Anchoring charges a recovering
                              model twice for one slip -- once for the slip, and
                              again when it returns to the true state, since that
                              return disagrees with the wrong anchor it adopted.
                              Measured on synthetic traces carrying exactly one
                              misreported state, cumulative loses 1 report per
                              trace and anchored loses 2.
      cumulative < anchored   the model drifts: it stays locally self-consistent
                              while its stated trajectory leaves the true one.

    A report is only scorable when the relevant truth is known. If an operation is
    unrecognised or inapplicable `apply_op` returns None and from there that track
    is undetermined, so its reports are dropped from both numerator and denominator
    rather than counted wrong, which would be arbitrary. The anchored track resumes
    at the next report; the cumulative track cannot resume, by construction.

    The two verdicts are recorded as a *pair* per report, not just as two totals,
    because they are scored on the same reports: comparing them is a paired
    comparison, and only the reports where they disagree carry information. The
    discordant counts are what McNemar's test needs.

    Returns (n_states, n_correct, n_after_injection, n_correct_after_injection,
    n_unscorable, n_cumulative, n_cumulative_correct, n_cum_only, n_anch_only),
    where the last two count reports that only the cumulative scoring accepted and
    reports that only the anchored scoring accepted.
    """
    eq = parse_equation(problem.equation)               # anchored ground truth
    cum = parse_equation(problem.equation)              # cumulative ground truth
    total = correct = post_n = post_ok = skipped = 0
    cum_total = cum_correct = 0
    cum_only = anch_only = 0            # discordant pairs, for McNemar
    just_injected = False
    for r in records:
        if r['kind'] == 'op':
            if r['text'].lower().startswith('write the equation'):
                eq = parse_equation(problem.equation)   # word problems restate it
                cum = parse_equation(problem.equation)
            else:
                eq = apply_op(eq, r['text'])
                cum = apply_op(cum, r['text'])
            just_injected = r['injected']
        elif r['kind'] == 'state':
            stated = parse_equation(r['text'][len(STATE_PREFIX):].strip().rstrip('.'))
            if eq is None:                              # true state unknown
                skipped += 1
            else:
                ok = bool(equations_equal(stated, eq))
                total += 1
                correct += ok
                if just_injected:
                    post_n += 1
                    post_ok += ok
            if cum is not None:
                cum_ok = bool(equations_equal(stated, cum))
                cum_total += 1
                cum_correct += cum_ok
                if eq is not None:      # both scorable -> the pair is comparable
                    cum_only += cum_ok and not ok
                    anch_only += ok and not cum_ok
            if stated is not None:                      # re-anchor: per-step scoring
                eq = stated
            just_injected = False
    return (total, correct, post_n, post_ok, skipped, cum_total, cum_correct,
            cum_only, anch_only)


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def diag_is_cached(interval, inject_p, dcfg, mcfg, tcfg, ecfg):
    """True if this diagnostic has already been computed."""
    return os.path.exists(_diag_path(interval, inject_p, dcfg, mcfg, tcfg, ecfg))


def _diag_path(interval, inject_p, dcfg, mcfg, tcfg, ecfg):
    spec = dict(task='math-diag', interval=interval, inject_p=inject_p,
                data=_data_spec(dcfg), model=asdict(mcfg), train=_train_spec(tcfg),
                eval=asdict(ecfg))
    return os.path.join(DIAG_CACHE, _key(spec) + '.json')


def run_diagnostic(interval, inject_p, dcfg, mcfg, tcfg, ecfg, device='cuda',
                   force=False, model_pack=None, log=print, progress_pos=0,
                   get_model=None):
    """
    Run D1 and D2 for one (interval, inject_p) pair.

    Decodes the test set twice with the same model -- clean and injected -- and
    reports the plan-replay ratio, answer agreement, and (for k >= 1) state
    tracking overall and immediately after an injection.

    Returns
    -------
    dict
        {'interval', 'inject_p', 'replay_ratio', 'answer_agree', 'state_acc',
         'post_inj_state_acc', 'n_traces_injected', 'mean_injections', 'cached'}
    """
    path = _diag_path(interval, inject_p, dcfg, mcfg, tcfg, ecfg)
    if os.path.exists(path) and not force:
        with open(path) as f:
            r = json.load(f)
        r['cached'] = True
        return r

    problems = get_problems(dcfg, 'test')

    def _default_model():                            # only called on a cache miss
        adapter = train_adapter(interval, dcfg, mcfg, tcfg, device=device, force=False,
                                log=log, progress_pos=progress_pos)
        return model_pack or load_for_eval(adapter, mcfg, tcfg, device=device)

    gm = get_model or _default_model
    kw = dict(device=device, progress_pos=progress_pos, log=log)
    _, clean_rec = get_or_decode(gm, problems, interval, 0.0,
                                 dcfg, mcfg, tcfg, ecfg,
                                 desc=f'k={interval} clean', **kw)
    inj_txt, inj_rec = get_or_decode(gm, problems, interval, inject_p,
                                     dcfg, mcfg, tcfg, ecfg,
                                     desc=f'k={interval} p={inject_p}', **kw)
    # greedy decoding makes a second clean pass bit-identical, so the D1 baseline
    # is 1.0 by construction rather than measured
    clean_rec2 = None
    clean_txt = [' '.join(r['text'] for r in rec) for rec in clean_rec]
    clean_rec2 = clean_rec2 if clean_rec2 is not None else clean_rec
    clean_txt2 = [' '.join(r['text'] for r in rec) for rec in clean_rec2]

    ratios, agrees, n_inj_tot, n_inj_traces = [], [], 0, 0
    base_ratios, base_agrees = [], []
    st_tot = st_ok = post_tot = post_ok = 0
    unscorable = 0
    n_ops_tot = 0                            # operations the model emits, clean
    # the symbolic replay is the slow part -- two state tracks over two decodes
    # per problem, plus the LCS -- and it runs with no GPU to show progress on
    bar = tqdm(total=len(problems), desc=f'diag k={interval} p={inject_p}',
               leave=False, dynamic_ncols=True, mininterval=2.0,
               position=progress_pos)
    for prob, crec, c2rec, irec, ctxt, c2txt, itxt in zip(
            problems, clean_rec, clean_rec2, inj_rec, clean_txt, clean_txt2, inj_txt):
        bar.update(1)
        # baseline: clean vs clean, same comparison, no injections involved
        ops_a = [r['text'] for r in crec if r['kind'] == 'op']
        ops_b = [r['text'] for r in c2rec if r['kind'] == 'op']
        if ops_a:
            base_ratios.append(_lcs(ops_b, ops_a) / len(ops_a))
            base_agrees.append(parse_answers(c2txt) == parse_answers(ctxt))
        ratio, n_inj = plan_replay(crec, irec)
        n_inj_tot += n_inj
        if ratio is not None:
            n_inj_traces += 1
            ratios.append(ratio)
            agrees.append(parse_answers(itxt) == parse_answers(ctxt))
        if interval is not None:
            try:
                a, b, c, d, sk, _, _, _, _ = state_tracking(prob, irec)
            except Exception as exc:                    # a decoded trace is arbitrary
                print(f'  [diag] skipped a trace: {type(exc).__name__}: {exc}', flush=True)
                a = b = c = d = sk = 0
            st_tot += a
            st_ok += b
            post_tot += c
            post_ok += d
            unscorable += sk

    bar.close()
    r = dict(interval=interval, inject_p=inject_p,
             replay_ratio=(sum(ratios) / len(ratios)) if ratios else None,
             replay_baseline=(sum(base_ratios) / len(base_ratios)) if base_ratios else None,
             answer_agree=(sum(agrees) / len(agrees)) if agrees else None,
             answer_agree_baseline=(sum(base_agrees) / len(base_agrees)) if base_agrees else None,
             state_acc=(st_ok / st_tot) if st_tot else None,

             post_inj_state_acc=(post_ok / post_tot) if post_tot else None,
             # mean operations per clean trace: the number of operations the
             # standard condition composes before its single state commitment
             # (the answer), used to put every condition on a per-operation scale
             # reports dropped because an earlier unparseable operation left the
             # true equation undetermined; reported so the denominator is visible
             n_unscorable=unscorable,
             n_traces_injected=n_inj_traces, n_test=len(problems),
             mean_injections=n_inj_tot / max(len(problems), 1), cached=False)

    os.makedirs(DIAG_CACHE, exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(r, f, indent=1)
    os.replace(tmp, path)
    return r


def print_diagnostics(results):
    """Print D1 and D2 as one table, most-informative columns first."""
    def cell(v, w=9):
        return ('-' if v is None else f'{v:.3f}').rjust(w)

    rows = sorted(results, key=lambda r: (r['inject_p'],
                                          99 if r['interval'] is None else r['interval']))
    print('\ndiagnostics -- why a condition survives injection')
    print('  D1 replay : LCS(model ops under injection, clean ops) / clean ops.')
    print('              base = the same measure between two CLEAN decodes, i.e. the')
    print('              sampling noise floor. replay close to base => the injection')
    print('              barely changed the plan (the model ignored it).')
    print('  D1 agree  : injected run ends on the same answer as the clean run')
    print('              (base = agreement between two clean runs)')
    print('  D2 state  : model-reported equation matches the true one (k>=1 only)')
    print('  D2 post   : same, but only right after an injected operation')
    print()
    print('   p     cond   D1 replay  (base)   D1 agree  (base)   D2 state    D2 post   inj/trace')
    for r in rows:
        cond = 'std' if r['interval'] is None else str(r['interval'])
        print(f'  {r["inject_p"]:<4g}  {cond:>5}  {cell(r["replay_ratio"])} '
              f'{cell(r.get("replay_baseline"), 7)}  {cell(r["answer_agree"])} '
              f'{cell(r.get("answer_agree_baseline"), 7)}  {cell(r["state_acc"])}  '
              f'{cell(r["post_inj_state_acc"])}  {r["mean_injections"]:>9.2f}')
