"""
ExploreToM data for the state-tracking experiment: the stories, their exact
per-sentence world and belief state, the test and training rows, the prompt,
the exact training targets and the grader.

ExploreToM (Sclar et al., ICLR 2025, github.com/facebookresearch/ExploreToM)
generates theory-of-mind stories as programs over a small action language --
people enter and leave rooms, move objects into containers or other rooms,
tell each other things privately or out loud, and may witness an action in
secret or miss it while distracted. Its `belief_tracker` keeps the true
world state and every character's first- and second-order beliefs after each
action, and the questions and answers are read off that tracker. The
released sample (13,309 questions over 621 stories, adversarially selected
against Llama-3.1-70B) contains the *story structure* -- the templated
sentence per action -- and the question and answer, but not the states.

This module recovers the states exactly: each templated sentence is parsed
back into the action that produced it, the actions are replayed through the
authors' own tracker (fetched at a pinned commit), and the replayed
tracker's questions are regenerated and checked against the released
answers. A story is kept only if its script is reproduced verbatim and all
of its released questions come back with the released answer, so every
state used below is the state the answer key was computed from. Stories
with free-text object-state sentences (``Eric switched the walkie-talkie to
a high power mode.``) cannot be parsed and are left out (117 of 619 story
structures); ``memory_before_event`` questions, whose answer is not a
function of the final state, are left out too.

Prompt
------
Every condition sees the same chat prompt: `SYSTEM_PROMPT`, then a user turn
``Story: <story structure>\\n\\nQuestion: <question>``, and is asked to end
with the answer field ``Answer: <answer>``.

Training targets (the assistant turn a condition is fine-tuned on)
------------------------------------------------------------------
direct   the answer field alone: ``Answer: leather briefcase``.
chain    the sentences at which the answer to *this question* changes, each
         quoted with the answer as it stands after it, then the answer field
         -- a question-specific chain that skips the rest of the story:

             Tracking the answer to the question through the story.
             Kaylee moved the silver letter opener to the wooden desk drawer, ... -> answer now: wooden desk drawer
             Liam moved the silver letter opener to the leather briefcase, ... -> answer now: leather briefcase
             Answer: leather briefcase

state_tracking
         the story restated, with the *question's slice* of the state
         (`focus_of`: the people asked about, the object or topic) after
         every block of `FOCUS_STRIDE` (one) steps, then the answer field --
         the local, De Bruijn-structured format:

             Kaylee entered the hotel lobby.
             State: Kaylee is in the hotel lobby. Beliefs: none yet.
             Kaylee moved the silver letter opener to the wooden desk drawer, which is also located in the hotel lobby. While this action was happening, Liam witnessed this action in secret (and only this action).
             State: the silver letter opener is in the wooden desk drawer in the hotel lobby. Beliefs: Kaylee believes the opener is in the wooden desk drawer; Kaylee thinks Liam does not know where the opener is.
             ...
             Answer: leather briefcase

         The asked beliefs are written on every line whether or not they
         depart from the truth (`FOCUS_BELIEFS = 'explicit''`). A state line
         is a function of the previous state line and the steps since it
         (`render_state`): the true locations of the asked entities, then the
         asked people's beliefs about the asked object/topic and, for a
         second-order question, one person's view of the other. A structured
         reader (`answer_from_state`) answers every kept question from the
         final state alone, so the trace is sufficient by construction.

narration
         the length-matched control for state_tracking (`narration_note`):
         the same story restated, but each step is followed by a one-line
         ``Note:`` of that step's *local* event -- what happened and who was
         present -- carrying no running state forward:

             Kaylee entered the hotel lobby.
             Note: Kaylee is now in the hotel lobby; present there: Kaylee.
             Kaylee moved the silver letter opener to the wooden desk drawer, ...
             Note: Kaylee put the silver letter opener into the wooden desk drawer in the hotel lobby; present there: Kaylee.
             ...
             Answer: leather briefcase

         Each note is non-cumulative, so the final note is not a sufficient
         statistic and the question can only be answered by integrating the
         notes across the whole story. Same restating and roughly the same
         length as state_tracking, without its running-state (De Bruijn)
         structure: if state_tracking wins, the structure is what helps, not
         the length.

A sentence that adds secret witnesses or distracted people ("While this
action was happening, ...") modifies the action before it; the two sentences
form one *step* and share one state line (and one note).

Grader
------
`extract_answer` reads what follows the last ``Answer`` field, ignoring
case, markdown, brackets, a leading article or preposition and trailing
punctuation; a completion without the field is scored on its last line and
counted as unanswered. Exact match after that normalisation.
"""

import os
import re
import ast
import csv
import copy
import json
import random
import hashlib
import importlib.util
import urllib.request
from dataclasses import dataclass, asdict, field

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
DATA_CACHE = os.path.join(CACHE, 'datasets')
RAW_DIR = os.path.join(CACHE, 'raw')

# The released sample and the tracker, pinned. The tracker is the authors'
# code (CC BY-NC 4.0), fetched once and imported from the cache.
DATASET_REPO = 'facebook/ExploreToM'
DATASET_REVISION = 'a4ac6f257e0034945f829716047ae6306dc625a0'
DATASET_FILE = 'ExploreToM-data-sample.csv'
DATASET_SHA256 = 'f8462b1a2199819a2baef5ab321b7b9240249fe229b94c86d8d06c2c47fc448b'
TRACKER_REPO = 'facebookresearch/ExploreToM'
TRACKER_COMMIT = '6a9372870ddc1c7b9b9ce1f3ee1641dbbdb56760'
TRACKER_SHA256 = '0a00f85034735dc0b0b6da787152d643a4243db3624bfda09f75af20b2743b90'

ANSWER_MARKER = 'Answer:'
STATE_PREFIX = 'State:'
NOTE_PREFIX = 'Note:'
CHAIN_HEADER = 'Tracking the answer to the question through the story.'
CHAIN_ARROW = '-> answer now:'

SYSTEM_PROMPT = (
    'You read a short story and answer a question about it. You may reason '
    'first. Whatever else you write, the last line of your reply must be the '
    f'answer field: "{ANSWER_MARKER} <answer>". Answer a yes/no question with yes '
    'or no, a question that offers two phrases in brackets with one of those '
    'phrases, and any other question with the name of the container, room or '
    'object.'
)

# Bumped when the cached problem set's schema changes so it is rebuilt. v2:
# the focused ledger and state strides; v3: explicit belief clauses; v4: the
# per-step narration notes. Per-condition target text is versioned separately
# (`TARGET_VERSIONS`), so a bump here rebuilds the problem set but keeps the
# adapters whose target text did not change.
TRACE_VERSION = '4'
# The version of each condition's target text, for the adapter cache keys:
# a bump of TRACE_VERSION re-makes only the adapters whose text changed.
TARGET_VERSIONS = {'direct': '2', 'chain': '2', 'state_tracking': '3', 'narration': '2'}

# State-tracking stride: a state line after every block of this many steps
# (blocks as even as possible, the earlier ones taking the extra step; the
# last state is always written). Every step by default.
FOCUS_STRIDE = 1

# How the state-tracking trace writes beliefs: 'explicit' states the asked
# person's belief about the asked object or topic on every line (and, for a
# second-order question, their view of the other person), whether or not it
# departs from the truth; 'departures' lists only what departs. A model
# trained on departures learnt to write "as stated" and answered false-belief
# questions with the truth (`analyze_ledger.py`).
FOCUS_BELIEFS = 'explicit'
BELIEF_MODES = ('explicit', 'departures')


# ----------------------------------------------------------------------------
# configs / cache keys
# ----------------------------------------------------------------------------

@dataclass
class DataConfig:
    """
    Which questions are trained and tested on, and where the stories come from.

    ``dataset`` is 'sample' (the released ExploreToM sample) or 'generated'
    (larger stories built locally with `generate_stories`); the ``gen_*``
    fields size the generated stories and are ignored for the sample. The
    dataset choice and, when generating, the ``gen_*`` fields are part of the
    problem-set cache key, so a generated set is reproducible and kept apart
    from the sample.
    """
    n_train: int = 1500             # training rows (questions) from the training stories
    n_test: int = 250               # test rows, from stories disjoint from the training ones
    q_per_story: int = 4            # at most this many questions of one story, either split
    seed: int = 0                   # story order and the choice of questions within a split
    dataset: str = 'sample'         # 'sample' | 'generated'
    # generation (used only when dataset == 'generated'):
    gen_people: int = 6
    gen_rooms: int = 3
    gen_objects: int = 4
    gen_containers: int = 3
    gen_topics: int = 4
    gen_moves: int = 12             # actions after the initial placements
    gen_stories: int = 500          # distinct stories to generate
    gen_seed: int = 0
    gen_peek: float = 0.3           # chance a witnessable action adds a peeker/distracted person
    gen_comm: float = 0.35          # weight on communication vs physical actions
    gen_q_cap: int = 30             # most questions emitted per generated story (interesting first)
    gen_interesting_only: bool = False   # keep only questions whose answer depends on who is asked


DATASETS = ('sample', 'generated')


def _key(obj):
    """Short stable hash of a JSON-able config."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def dataset_spec(dcfg):
    """
    What identifies the universe of usable stories/questions (before the
    train/test split): the released sample at its pinned revision, or the
    generated set with its sizes and seed.
    """
    if dcfg.dataset == 'generated':
        return dict(kind='generated', gen_version=GEN_VERSION, trace_version=TRACE_VERSION,
                    people=dcfg.gen_people, rooms=dcfg.gen_rooms, objects=dcfg.gen_objects,
                    containers=dcfg.gen_containers, topics=dcfg.gen_topics, moves=dcfg.gen_moves,
                    stories=dcfg.gen_stories, gen_seed=dcfg.gen_seed, peek=dcfg.gen_peek,
                    comm=dcfg.gen_comm, q_cap=dcfg.gen_q_cap,
                    interesting_only=dcfg.gen_interesting_only)
    return dict(kind='sample', revision=DATASET_REVISION, trace_version=TRACE_VERSION)


def train_spec(dcfg):
    """
    The training half of a DataConfig, for cache keys that must not depend on
    the test set (adapters, traces). The dataset tag is added only for
    generated data, so the released-sample keys are exactly what they were
    before generation existed and its cached adapters/evals still hit.
    """
    spec = dict(n_train=dcfg.n_train, n_test=dcfg.n_test, q_per_story=dcfg.q_per_story,
                seed=dcfg.seed)
    if dcfg.dataset != 'sample':
        spec['data'] = dataset_spec(dcfg)
    return spec


def test_spec(dcfg):
    """The test half of a DataConfig, for decode and eval keys (dataset tag
    added only for generated data; see `train_spec`)."""
    spec = dict(n_test=dcfg.n_test, q_per_story=dcfg.q_per_story, seed=dcfg.seed)
    if dcfg.dataset != 'sample':
        spec['data'] = dataset_spec(dcfg)
    return spec


# ----------------------------------------------------------------------------
# fetching: the sample CSV and the belief tracker
# ----------------------------------------------------------------------------

def _download(url, path, sha256):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f'.tmp{os.getpid()}'
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, 'wb') as f:
            f.write(r.read())
        os.replace(tmp, path)
    with open(path, 'rb') as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != sha256:
        raise RuntimeError(f'{path} has sha256 {got}, expected {sha256}; delete it to re-download')
    return path


def fetch_csv():
    return _download(f'https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/'
                     f'{DATASET_FILE}', os.path.join(RAW_DIR, DATASET_FILE), DATASET_SHA256)


_tracker_module = None


def tracker():
    """The authors' `belief_tracker` module at the pinned commit, imported from the cache."""
    global _tracker_module
    if _tracker_module is None:
        path = _download(f'https://raw.githubusercontent.com/{TRACKER_REPO}/{TRACKER_COMMIT}/'
                         'belief_tracker.py', os.path.join(RAW_DIR, 'belief_tracker.py'),
                         TRACKER_SHA256)
        spec = importlib.util.spec_from_file_location('exploretom_belief_tracker', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _tracker_module = mod
    return _tracker_module


# ----------------------------------------------------------------------------
# parsing a story structure back into actions
# ----------------------------------------------------------------------------

_PATTERNS = [
    ('move_object_container', re.compile(
        r'^(?P<p>\S+) moved the (?P<o>.+?) to the (?P<c>.+?), which is also located in the (?P<r>.+)\.$')),
    ('move_object_room', re.compile(
        r'^(?P<p>\S+) moved the (?P<o>.+?) to the (?P<r>.+?), leaving the (?P<c>.+?) in its original location\.$')),
    ('move_object_room', re.compile(r'^(?P<p>\S+) moved the (?P<o>.+?) to the (?P<r>.+)\.$')),
    ('enter_room', re.compile(r'^(?P<e>\S+) entered the (?P<r>.+)\.$')),
    ('leave_room', re.compile(r'^(?P<e>\S+) left the (?P<r>.+)\.$')),
    ('private_about', re.compile(r'^(?P<a>\S+) told privately to (?P<b>\S+) about the (?P<t>.+)\.$')),
    ('private_that', re.compile(r'^(?P<a>\S+) told privately to (?P<b>\S+) that (?P<s>.+)\.$')),
    ('broadcast_about', re.compile(r'^(?P<a>\S+) told out loud about the (?P<t>.+)\.$')),
    ('broadcast_that', re.compile(r'^(?P<a>\S+) told out loud that (?P<s>.+)\.$')),
    ('location_declaration', re.compile(r'^(?P<e>\S+) is in the (?P<r>.+)\.$')),
]
_PEEK = re.compile(r'^While this action was happening, (?P<body>.+)$')
_PEEK_PART = re.compile(r'^(?P<names>.+?) witnessed this action in secret \(and only this action\)[.;]?$')
_DIST_PART = re.compile(r'^(?P<names>.+?) got distracted and did not realize what happened, without '
                        r'anyone noticing the brief lack of attention, and going back to paying '
                        r'attention immediately after the action was finished\.$')
_THAT = re.compile(r'^(?:the )?(?P<ent>.+?) is in the (?P<loc>.+)$')


def split_sentences(text):
    """The templated sentences of a story structure (each ends with a period)."""
    return [s for s in re.split(r'(?<=\.)\s+(?=[A-Z])', text.strip()) if s]


def _names(s):
    s = s.strip()
    if ', and ' in s:
        head, last = s.rsplit(', and ', 1)
        return [x.strip() for x in head.split(',')] + [last.strip()]
    if ' and ' in s:
        return [x.strip() for x in s.split(' and ')]
    return [s]


def _parse_peek(sentence):
    """``{'people_peeking', 'people_distracted'}`` of a witness-modifier
    sentence, None if the sentence is not one, 'BAD' if it is malformed."""
    m = _PEEK.match(sentence)
    if not m:
        return None
    body = m.group('body')
    parts = [p + ('.' if i == 0 and '; also, ' in body else '') for i, p in
             enumerate(body.split('; also, ', 1))]
    out = dict(people_peeking=[], people_distracted=[])
    for part in parts:
        mp, md = _PEEK_PART.match(part), _DIST_PART.match(part)
        if mp:
            out['people_peeking'] = _names(mp.group('names'))
        elif md:
            out['people_distracted'] = _names(md.group('names'))
        else:
            return 'BAD'
    return out


def parse_story(text):
    """
    The steps of a story structure: ``[(text, kind, groups, kwargs)]`` where
    `text` is the step's sentence(s) verbatim, `kind` the tracker action,
    `groups` its parsed arguments and `kwargs` the peeking/distracted people.
    Returns ``(None, reason)`` for a sentence that is not a known template.
    """
    steps = []
    for s in split_sentences(text):
        pk = _parse_peek(s)
        if pk == 'BAD':
            return None, f'malformed witness sentence: {s[:60]}'
        if pk is not None:
            if not steps:
                return None, 'witness sentence before any action'
            steps[-1][0] += ' ' + s
            steps[-1][3].update(pk)
            continue
        for kind, pat in _PATTERNS:
            m = pat.match(s)
            if m:
                steps.append([s, kind, m.groupdict(), {}])
                break
        else:
            return None, f'unparsed sentence: {s[:60]}'
    return steps, ''


def apply_step(bt, kind, g, kw):
    """Perform one parsed step on tracker `bt`; True if the tracker accepted it."""
    T = tracker().FullBeliefTracker
    if kind == 'enter_room':
        return bt.enter_room(g['e'], g['r'], **kw)
    if kind == 'leave_room':
        return bt.leave_room(g['e'], g['r'], **kw)
    if kind == 'location_declaration':
        return bt.location_declaration(g['e'], g['r'])
    if kind == 'move_object_container':
        return bt.move_object_container(g['p'], g['o'], g['c'], **kw)
    if kind == 'move_object_room':
        if kw:
            return False                      # the tracker does not support witnesses here
        return bt.move_object_room(g['p'], g['o'], g['r'])
    if kind in ('private_about', 'broadcast_about'):
        topic = g['t']
    else:
        m = _THAT.match(g['s'])
        if not m:
            return False
        ent, loc = m.group('ent'), m.group('loc')
        prop = T.CONTAINER_LOCATION if bt.world_state.get(ent, T.CONTAINER_LOCATION) == loc \
            else T.ROOM_LOCATION
        topic = (ent, prop, loc, True)
    if kind.startswith('private'):
        return bt.private_communication(g['a'], g['b'], topic, **kw)
    return bt.broadcast_communication(g['a'], topic, **kw)


def replay(steps):
    """
    Replay parsed steps on a fresh tracker. Returns ``(final tracker, [tracker
    after each step])`` or ``(None, reason)`` if the tracker refuses a step.
    """
    bt = tracker().FullBeliefTracker()
    states = []
    for _, kind, g, kw in steps:
        if not apply_step(bt, kind, g, kw):
            return None, f'tracker refused {kind} {g}'
        states.append(copy.deepcopy(bt))
    return bt, states


def regenerate_questions(bt, steps):
    """Every ``question -> answer`` the tracker generates for a replayed story
    (first- and second-order belief questions and the factual ones)."""
    qg = tracker().QuestionGenerator(bt)
    out = {}
    for order in (1, 2):
        for q, a, _, _ in qg.main(order):
            out[q] = a
    fl = [lambda b, k=k, g=g, kw=kw: apply_step(b, k, g, kw) for _, k, g, kw in steps]
    for q, a, _, _ in qg.generate_factual_questions(fl):
        out[q] = a
    return out


# ----------------------------------------------------------------------------
# the state after each step, and what a ledger line says about it
# ----------------------------------------------------------------------------

def _loc_pair(ws, thing, T):
    """(container, room) a WorldState holds for `thing`; None entries when unknown."""
    return (ws.get(thing, T.CONTAINER_LOCATION), ws.get(thing, T.ROOM_LOCATION))


def _known(v):
    return v is not None and not v.startswith('not(')


def _place(c, r):
    """'in the C in the R' / 'in the C' / 'in the R' from a (container, room) pair."""
    if _known(c) and _known(r):
        return f'in the {c} in the {r}'
    if _known(c):
        return f'in the {c}'
    if _known(r):
        return f'in the {r}'
    return None


def _order(bt, steps):
    """People, objects and containers in order of first mention."""
    text = ' '.join(s[0] for s in steps)
    pos = lambda name: text.find(name) if name in text else 10 ** 9       # noqa: E731
    return (sorted(bt.people, key=pos), sorted(bt.objects, key=pos), sorted(bt.containers, key=pos))


def focus_of(params, order, bt):
    """
    The entities a question depends on, for the focused ledger: the people
    asked about, the object or container the question is about, the topic.
    ``{'people', 'objects', 'containers', 'topics', 'pairs'}`` where `pairs`
    holds the ordered pair of a second-order question.
    """
    T = tracker().FullBeliefTracker
    entities, thing, rel = params
    people = list(entities or [])
    focus = dict(people=set(people), objects=set(), containers=set(), topics=set(),
                 pairs={(people[0], people[1])} if order == 2 and len(people) == 2 else set())
    base = rel.rsplit('-', 1)[0] if not rel.startswith(('memory', 'ground_truth')) else rel
    if base == T.KNOWLEDGE:
        focus['topics'].add(thing)
    elif thing in bt.containers:
        focus['containers'].add(thing)
    else:
        focus['objects'].add(thing)
    return focus


def state_facts(bt, steps, initial, focus=None):
    """
    The content of one ledger line, structured.

    ``world``: people -> room ('has left the R' as ``('left', R)``), objects
    -> (container, room), containers -> room, plus the object's *initial*
    container/room when it differs from the current one (`initial` is the
    ``{(obj, prop): first value}`` seen so far).
    ``first``: person -> {object: belief} for beliefs that differ from the
    truth: 'unknown' or a (container, room) pair; person -> topics heard.
    ``second``: (x, y) -> {object: belief} where x's view of y differs from
    x's own belief; (x, y) -> topics y is thought to lack / to have.

    With `focus` (`focus_of`) only the question's people, objects,
    containers, topics and ordered pair are kept: the slice of the state the
    question depends on, which is still a function of the previous slice and
    the current step.
    """
    T = tracker().FullBeliefTracker
    people, objects, containers = _order(bt, steps)
    if focus is not None:
        people = [p for p in people if p in focus['people']]
        objects = [o for o in objects if o in focus['objects']]
        containers = [c for c in containers if c in focus['containers']]
    W = bt.world_state
    world = dict(people={}, objects={}, containers={}, initial={})
    for p in people:
        r = W.get(p, T.ROOM_LOCATION)
        world['people'][p] = r
    for o in objects:
        c, r = _loc_pair(W, o, T)
        world['objects'][o] = (c, r)
        for prop, cur in ((T.CONTAINER_LOCATION, c), (T.ROOM_LOCATION, r)):
            first = initial.get((o, prop))
            if first is not None and first != cur:
                world['initial'][(o, prop)] = first
    for c in containers:
        world['containers'][c] = W.get(c, T.ROOM_LOCATION)

    def topics_of(ws):
        v = ws.get('', T.KNOWLEDGE)
        return frozenset(v) if v else frozenset()

    first, heard, second, sec_topics = {}, {}, {}, {}
    for p in people:
        own = bt.first_order_beliefs[p]
        heard[p] = topics_of(own)
        if focus is not None:
            heard[p] = heard[p] & focus['topics']
        diffs = {}
        for o in objects:
            truth = world['objects'][o]
            if truth == (None, None):
                continue
            bel = _loc_pair(own, o, T)
            if bel == truth:
                continue
            if _place(*bel) is None:
                if _place(*truth) is not None:
                    diffs[o] = 'unknown'
            else:
                diffs[o] = bel
        first[p] = diffs
        for q in people:
            if q == p or (focus is not None and (p, q) not in focus['pairs']):
                continue
            view = bt.second_order_beliefs[p][q]
            d2 = {}
            for o in objects:
                mine, theirs = _loc_pair(own, o, T), _loc_pair(view, o, T)
                if mine == theirs:
                    continue
                if _place(*theirs) is None:
                    if _place(*mine) is not None:
                        d2[o] = 'unknown'
                else:
                    d2[o] = theirs
            second[(p, q)] = d2
            vt = topics_of(view)
            if focus is not None:
                vt = vt & focus['topics']
            sec_topics[(p, q)] = (heard[p] - vt, vt - heard[p])
    topics = sorted(t for t in bt.topics if focus is None or t in focus['topics'])
    return dict(people=people, objects=objects, containers=containers, topics=topics,
                world=world, first=first, heard=heard, second=second, sec_topics=sec_topics)


def _belief_text(p, o, bel):
    return (f'{p} does not know where the {o} is' if bel == 'unknown' or _place(*bel) is None
            else f'{p} believes the {o} is {_place(*bel)}')


def _explicit_beliefs(facts):
    """
    Every belief the line is about, stated whether or not it departs from the
    truth: each person's belief about each object and topic, then, for each
    ordered pair kept, what the first thinks the second believes.
    """
    w = facts['world']
    out = []
    for p in facts['people']:
        for o in facts['objects']:
            truth = w['objects'][o]
            if _place(*truth) is None:
                continue                          # the object has not appeared yet
            out.append(_belief_text(p, o, facts['first'][p].get(o, truth)))
        for t in facts['topics']:
            out.append(f'{p} has {"" if t in facts["heard"][p] else "not "}heard about {t}')
    for (p, q), d2 in facts['second'].items():
        for o in facts['objects']:
            truth = w['objects'][o]
            if _place(*truth) is None:
                continue
            own = facts['first'][p].get(o, truth)
            view = d2.get(o, own)
            out.append(f'{p} thinks {q} ' + ('does not know where the ' + o + ' is'
                                            if view == 'unknown' or _place(*view) is None
                                            else f'believes the {o} is {_place(*view)}'))
        lack, extra = facts['sec_topics'][(p, q)]
        for t in facts['topics']:
            knows = (t in facts['heard'][p] and t not in lack) or t in extra
            out.append(f'{p} thinks {q} has {"" if knows else "not "}heard about {t}')
    return out


def render_state(facts, mode='departures'):
    """One ledger line from `state_facts`; `mode` is a `BELIEF_MODES` entry."""
    w = facts['world']
    clauses = []
    for p in facts['people']:
        r = w['people'][p]
        if r is None:
            clauses.append(f'{p} has not entered any room')
        elif r.startswith('not('):
            clauses.append(f'{p} has left the {r[4:-1]}')
        else:
            clauses.append(f'{p} is in the {r}')
    for o in facts['objects']:
        place = _place(*w['objects'][o])
        if place is None:
            continue
        s = f'the {o} is {place}'
        for prop in ('container_location', 'room_location'):
            if (o, prop) in w['initial']:
                s += f' (it started in the {w["initial"][(o, prop)]})'
        clauses.append(s)
    for c in facts['containers']:
        r = w['containers'][c]
        if _known(r):
            clauses.append(f'the {c} is in the {r}')
    beliefs = [] if mode != 'explicit' else None
    for p in (facts['people'] if beliefs is not None else []):
        if facts['heard'][p]:
            beliefs.append(f'{p} has heard about ' + ' and '.join(sorted(facts['heard'][p])))
        for o, bel in facts['first'][p].items():
            beliefs.append(f'{p} does not know where the {o} is' if bel == 'unknown'
                           else f'{p} thinks the {o} is {_place(*bel)}')
        for q in facts['people']:
            if q == p or (p, q) not in facts['second']:
                continue
            for o, bel in facts['second'][(p, q)].items():
                beliefs.append(f'{p} thinks {q} does not know where the {o} is' if bel == 'unknown'
                               else f'{p} thinks {q} believes the {o} is {_place(*bel)}')
            lack, extra = facts['sec_topics'][(p, q)]
            if lack:
                beliefs.append(f'{p} thinks {q} has not heard about ' + ' and '.join(sorted(lack)))
            if extra:
                beliefs.append(f'{p} thinks {q} has heard about ' + ' and '.join(sorted(extra)))
    if not clauses:
        return f'{STATE_PREFIX} nothing relevant yet.'
    if mode == 'explicit':
        stated = _explicit_beliefs(facts)
        return (f'{STATE_PREFIX} ' + '; '.join(clauses) + '. Beliefs: '
                + ('; '.join(stated) + '.' if stated else 'none yet.'))
    return (f'{STATE_PREFIX} ' + '; '.join(clauses) + '. Beliefs: '
            + ('; '.join(beliefs) + '.' if beliefs else 'as stated.'))


def answer_from_state(facts, params, order):
    """
    The answer to a question read from the ledger's final state alone, under
    the rendering's defaults (a belief not listed equals the truth; a view of
    another person not listed equals one's own belief). `params` is the
    dataset's ``(entities, thing, relation)`` and `order` its reasoning order.
    Returns None where the state does not determine an answer.
    """
    T = tracker().FullBeliefTracker
    entities, thing, rel = params
    base = rel.rsplit('-', 1)[0] if not rel.startswith(('memory', 'ground_truth')) else rel
    w = facts['world']

    def own(p, o):
        return facts['first'].get(p, {}).get(o, w['objects'].get(o))

    def view(p, q, o):
        return facts['second'].get((p, q), {}).get(o, own(p, o))

    def pick(bel, prop):
        if bel is None or bel == 'unknown':
            return None
        v = bel[0] if prop == T.CONTAINER_LOCATION else bel[1]
        return v if _known(v) else None

    if base.startswith('memory-') or base.startswith('ground_truth-'):
        prop = base.split('-', 1)[1]
        cur = w['objects'].get(thing, (None, None))
        cur = cur[0] if prop == T.CONTAINER_LOCATION else cur[1]
        if base.startswith('ground_truth'):
            return cur if _known(cur) else None
        return w['initial'].get((thing, prop), cur if _known(cur) else None)
    if base == T.KNOWLEDGE:
        if order == 1:
            return 'yes' if thing in facts['heard'].get(entities[0], ()) else 'no'
        x, y = entities
        lack, extra = facts['sec_topics'].get((x, y), ((), ()))
        knows = (thing in facts['heard'].get(x, ()) and thing not in lack) or thing in extra
        return 'knows about it' if knows else 'does not know about it'
    if base in (T.CONTAINER_LOCATION, T.ROOM_LOCATION):
        if thing in w['containers']:                      # containers never move
            return w['containers'][thing] if base == T.ROOM_LOCATION else None
        if order == 1:
            return pick(own(entities[0], thing), base)
        x, y = entities
        return pick(view(x, y, thing), base)
    return None


# ----------------------------------------------------------------------------
# problems
# ----------------------------------------------------------------------------

@dataclass
class Problem:
    """
    One (story, question) row. `story_id` indexes the kept stories, `steps`
    are the story's steps verbatim (a witness sentence joined to its action),
    `states` the full ledger line after each step, `focused_states` the
    question's focused ledger line after each step with beliefs as
    departures, `focused_explicit` the same with every asked belief stated
    (`FOCUS_BELIEFS`), `chain` the steps at
    which the answer to this question changes as ``[(step index, answer so
    far)]``, `qtype` a short label ('knowledge-2', 'container-1', 'room-2',
    'memory', ...), `false_belief` whether the asked belief differs from the
    truth.
    """
    idx: int
    story_id: int
    story: str
    steps: list
    states: list
    question: str
    gold: str
    order: int
    qtype: str
    false_belief: bool
    chain: list = field(default_factory=list)
    focused_states: list = field(default_factory=list)
    focused_explicit: list = field(default_factory=list)
    narration_notes: list = field(default_factory=list)
    num_people: int = 0
    num_rooms: int = 0
    story_type: str = ''

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d['chain'] = [tuple(c) for c in d['chain']]
        return cls(**d)


def user_message(problem):
    return f'Story: {problem.story}\n\nQuestion: {problem.question}'


def question_hash(problem):
    return hashlib.sha1((problem.story + '|' + problem.question + '|' + problem.gold).encode()).hexdigest()[:8]


def _qtype(params, order):
    entities, thing, rel = params
    if rel.startswith('memory_before_event'):
        return 'memory_before_event', False
    if rel.startswith('memory'):
        return 'memory', False
    if rel.startswith('ground_truth'):
        return 'ground_truth', False
    base = rel.rsplit('-', 1)[0]
    fb = rel.endswith('-False')
    name = {'<knowledge>': 'knowledge', 'container_location': 'container',
            'room_location': 'room'}.get(base, base)
    return f'{name}-{order}', fb


def _running_answers(question, order, params, states, steps):
    """
    The answer to `question` after each step -- ``[(step index, answer)]`` at
    the steps where it changes -- by regenerating the tracker's questions on
    every prefix (factual questions are read from the world state).
    """
    T = tracker().FullBeliefTracker
    entities, thing, rel = params
    out, prev = [], None
    for i, bt in enumerate(states):
        if rel.startswith(('memory', 'ground_truth')):
            prop = rel.split('-', 1)[1]
            cur = bt.world_state.get(thing, prop)
            if rel.startswith('memory'):
                a = prev if prev is not None else (cur if _known(cur) else None)
            else:
                a = cur if _known(cur) else None
        else:
            qg = tracker().QuestionGenerator(bt)
            a = next((ans for q, ans, _, _ in qg.main(order) if q == question), None)
        if a is not None and a != prev:
            out.append((i, str(a)))
            prev = a
    return out


def load_rows(dcfg):
    """The raw rows of the dataset `dcfg` selects: the released CSV, or the
    locally generated stories (`generate_stories`, imported lazily)."""
    if dcfg.dataset == 'generated':
        from generate_stories import generate_rows
        return generate_rows(dcfg)
    if dcfg.dataset != 'sample':
        raise ValueError(f'unknown dataset {dcfg.dataset!r}; choose from {DATASETS}')
    with open(fetch_csv(), newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def build_problems(rows, log=print):
    """
    Every usable (story, question) row: stories that parse, replay verbatim
    and reproduce all their given answers; questions other than
    ``memory_before_event`` whose answer the ledger's final state determines.
    Returns ``(problems, stats)``. `rows` are in the released CSV schema
    (`load_rows`), so generated stories are verified exactly like the sample.
    """
    T = tracker().FullBeliefTracker
    stories = {}
    for r in rows:
        stories.setdefault(r['story_structure'], []).append(r)
    stats = dict(rows=len(rows), stories=len(stories), unparsed=0, refused=0, mismatch=0,
                 not_reproduced=0, kept_stories=0, before_event=0, insufficient=0,
                 insufficient_focused=0, kept_rows=0)
    problems = []
    story_id = 0
    for text, qs in stories.items():
        steps, why = parse_story(text)
        if steps is None:
            stats['unparsed'] += 1
            continue
        bt, states = replay(steps)
        if bt is None:
            stats['refused'] += 1
            continue
        if ' '.join(bt.story_script) != text:
            stats['mismatch'] += 1
            continue
        qa = regenerate_questions(bt, steps)
        if any(qa.get(r['question']) is None or str(qa[r['question']]) != r['expected_answer']
               for r in qs):
            stats['not_reproduced'] += 1
            continue
        stats['kept_stories'] += 1
        # ledger lines and the initial locations they carry
        initial, lines, facts_list = {}, [], []
        for st in states:
            for o in st.objects:
                for prop in (T.CONTAINER_LOCATION, T.ROOM_LOCATION):
                    v = st.world_state.get(o, prop)
                    if (o, prop) not in initial and _known(v):
                        initial[(o, prop)] = v
            facts = state_facts(st, steps, initial)
            facts_list.append(facts)
            lines.append(render_state(facts))
        # per-step local notes for the narration condition (question-agnostic)
        notes = [narration_note(step[1], step[2], step[3], st)
                 for step, st in zip(steps, states)]
        for r in qs:
            params = ast.literal_eval(r['qprop=params'])
            order = int(r['qprop=nth_order'])
            qtype, fb = _qtype(params, order)
            if qtype == 'memory_before_event':
                stats['before_event'] += 1
                continue
            read = answer_from_state(facts_list[-1], params, order)
            if read is None or str(read) != r['expected_answer']:
                stats['insufficient'] += 1
                continue
            chain = _running_answers(r['question'], order, params, states, steps)
            if not chain or chain[-1][1] != r['expected_answer']:
                stats['insufficient'] += 1
                continue
            # the focused ledger of this question, checked the same way
            focus = focus_of(params, order, bt)
            focused, initial_f = [], {}
            for st in states:
                for o in st.objects:
                    for prop in (T.CONTAINER_LOCATION, T.ROOM_LOCATION):
                        v = st.world_state.get(o, prop)
                        if (o, prop) not in initial_f and _known(v):
                            initial_f[(o, prop)] = v
                focused.append(state_facts(st, steps, initial_f, focus=focus))
            read = answer_from_state(focused[-1], params, order)
            if read is None or str(read) != r['expected_answer']:
                stats['insufficient_focused'] += 1
                continue
            problems.append(Problem(
                idx=len(problems), story_id=story_id, story=text, steps=[s[0] for s in steps],
                states=lines, question=r['question'], gold=r['expected_answer'], order=order,
                qtype=qtype, false_belief=fb, chain=chain,
                focused_states=[render_state(f) for f in focused],
                focused_explicit=[render_state(f, 'explicit') for f in focused],
                narration_notes=notes,
                num_people=int(r['param=num_people']), num_rooms=int(r['param=num_rooms']),
                story_type=r['param=story_type']))
            stats['kept_rows'] += 1
        story_id += 1
    return problems, stats


def _all_path(dcfg):
    if dcfg.dataset == 'generated':
        return os.path.join(DATA_CACHE, f'all_v{TRACE_VERSION}_gen_{_key(dataset_spec(dcfg))}.json')
    return os.path.join(DATA_CACHE, 'all_v' + TRACE_VERSION + '.json')


def all_problems(dcfg):
    """Every usable row of the dataset `dcfg` selects, built once and cached
    (a distinct cache file per dataset / generation config)."""
    path = _all_path(dcfg)
    if os.path.exists(path):
        with open(path) as f:
            return [Problem.from_dict(d) for d in json.load(f)]
    problems, stats = build_problems(load_rows(dcfg))
    os.makedirs(DATA_CACHE, exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump([p.to_dict() for p in problems], f)
    os.replace(tmp, path)
    with open(path[:-5] + '_stats.json', 'w') as f:
        json.dump(stats, f, indent=1)
    return problems


# --- selection ----------------------------------------------------------------

def _split(problems, dcfg):
    """
    Test and training rows. Stories are put in a seeded order; the first ones
    supply the test rows (at most `q_per_story` seeded questions each) until
    `n_test` are taken; every later story supplies at most `q_per_story`
    questions to the training pool, in story order. Returns
    ``(test, train_pool)``; the training rows are the pool's first `n_train`.
    """
    rng = random.Random(dcfg.seed)
    by_story = {}
    for p in problems:
        by_story.setdefault(p.story_id, []).append(p)
    ids = sorted(by_story)
    rng.shuffle(ids)
    picks = {}
    for sid in ids:
        qs = list(by_story[sid])
        rng.shuffle(qs)
        picks[sid] = qs[:dcfg.q_per_story]
    test, pool, n = [], [], 0
    for sid in ids:
        if n < dcfg.n_test:
            test.extend(picks[sid])
            n += len(picks[sid])
        else:
            pool.extend(picks[sid])
    return sorted(test, key=lambda p: p.idx), pool


def train_pool(dcfg):
    """The training candidates in the order they become training rows (the
    first `n_train` are the nominal training set)."""
    return _split(all_problems(dcfg), dcfg)[1]


def get_problems(dcfg, split):
    """The 'train' (first `n_train` of the pool, in row order) or 'test' rows."""
    test, pool = _split(all_problems(dcfg), dcfg)
    if split == 'train':
        return sorted(pool[:dcfg.n_train], key=lambda p: p.idx)
    return test


# ----------------------------------------------------------------------------
# training targets
# ----------------------------------------------------------------------------

def answer_line(gold):
    return f'{ANSWER_MARKER} {gold}'


def direct_target(problem):
    return answer_line(problem.gold)


def chain_target(problem):
    """The steps at which the answer changes, each with the answer after it."""
    lines = [CHAIN_HEADER]
    for i, ans in problem.chain:
        lines.append(f'{problem.steps[i]} {CHAIN_ARROW} {ans}')
    lines.append(answer_line(problem.gold))
    return '\n'.join(lines)


def blocks(n, stride):
    """
    The end index (exclusive) of each block when `n` steps are cut into
    ``ceil(n / stride)`` blocks as even as possible, the earlier ones taking
    the extra step: ``blocks(7, 3) == [3, 5, 7]``, ``blocks(7, 1) == [1..7]``.
    """
    k = max(1, -(-n // max(stride, 1)))
    base, extra = divmod(n, k)
    out, end = [], 0
    for i in range(k):
        end += base + (1 if i < extra else 0)
        out.append(end)
    return out


def _ledger(problem, states, stride):
    """Steps restated, a state line after every block of `stride` steps, the answer."""
    lines, start = [], 0
    for end in blocks(len(problem.steps), stride):
        lines.extend(problem.steps[start:end])
        lines.append(states[end - 1])
        start = end
    lines.append(answer_line(problem.gold))
    return '\n'.join(lines)


def state_tracking_target(problem, stride=FOCUS_STRIDE, beliefs=FOCUS_BELIEFS):
    """
    The state-tracking trace: the story restated step by step, each step
    followed (every `stride` steps) by the question's slice of the world-and-
    belief state, its beliefs written per `beliefs` ('explicit' by default),
    then the answer. This is the local, De Bruijn-structured trace -- each
    state line is a function of the previous state line and the steps since.
    """
    if beliefs not in BELIEF_MODES:
        raise ValueError(f'unknown belief mode {beliefs!r}; choose from {BELIEF_MODES}')
    states = problem.focused_explicit if beliefs == 'explicit' else problem.focused_states
    return _ledger(problem, states, stride)


def narration_note(kind, g, kw, bt):
    """
    A one-line note describing the *local* event of a single step, from the
    parsed action `(kind, g, kw)` and the tracker `bt` just after it: what
    happened and, for a witnessable action, who was in the room. It is
    deliberately non-cumulative -- it carries no running world/belief state
    forward -- so, unlike a state-tracking line, the final note is not a
    sufficient statistic and the question can only be answered by integrating
    the notes across the whole story. This is what makes narration a
    length-matched control without the state-tracking (De Bruijn) structure.
    """
    T = tracker().FullBeliefTracker
    ROOM = T.ROOM_LOCATION
    occ = lambda room: [p for p in sorted(bt.people) if bt.world_state.get(p, ROOM) == room]  # noqa: E731
    enum = lambda xs: ', '.join(xs) if xs else 'no one'                                        # noqa: E731
    if kind in ('enter_room', 'location_declaration'):
        core = f"{g['e']} is now in the {g['r']}; present there: {enum(occ(g['r']))}"
    elif kind == 'leave_room':
        core = f"{g['e']} left the {g['r']}; still there: {enum(occ(g['r']))}"
    elif kind == 'move_object_container':
        room = bt.world_state.get(g['o'], ROOM)
        core = (f"{g['p']} put the {g['o']} into the {g['c']} in the {room}; "
                f"present there: {enum(occ(room))}")
    elif kind == 'move_object_room':
        core = f"{g['p']} carried the {g['o']} to the {g['r']}; present there: {enum(occ(g['r']))}"
    elif kind in ('private_about', 'private_that'):
        what = g['t'] if kind == 'private_about' else g['s']
        verb = 'about the' if kind == 'private_about' else 'that'
        core = f"{g['a']} spoke privately to {g['b']} {verb} {what}"
    elif kind in ('broadcast_about', 'broadcast_that'):
        what = g['t'] if kind == 'broadcast_about' else g['s']
        verb = 'about the' if kind == 'broadcast_about' else 'that'
        core = (f"{g['a']} announced to the room {verb} {what}; "
                f"present there: {enum(occ(bt.world_state.get(g['a'], ROOM)))}")
    else:
        core = 'an action occurred'
    extra = ''
    if kw.get('people_peeking'):
        extra += f"; secretly watched by {enum(kw['people_peeking'])}"
    if kw.get('people_distracted'):
        extra += f"; {enum(kw['people_distracted'])} were not paying attention"
    return f'{NOTE_PREFIX} {core}{extra}.'


def narration_target(problem):
    """The narration trace: every step restated, each followed by its local
    `Note:` line (`narration_note`), then the answer. Length-matched to the
    state-tracking trace but with no running state -- the non-De-Bruijn control."""
    lines = []
    for step, note in zip(problem.steps, problem.narration_notes):
        lines.append(step)
        lines.append(note)
    lines.append(answer_line(problem.gold))
    return '\n'.join(lines)


TARGETS = {'direct': direct_target, 'chain': chain_target,
           'state_tracking': state_tracking_target, 'narration': narration_target}

# Version of the local story generator (`generate_stories`); part of the
# generated dataset's cache key.
GEN_VERSION = '1'


# ----------------------------------------------------------------------------
# grading a completion
# ----------------------------------------------------------------------------

_FIELD = re.compile(r'\banswer\b\W{0,8}(.+)', re.I)


def normalise(s):
    """Lower-case, no markdown or brackets, no leading article/preposition, no
    trailing punctuation, single spaces."""
    s = s.strip().lower()
    s = re.sub(r'[*_`"\'\[\]()]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'^(?:in|at|inside|into|on)\s+', '', s)
    s = re.sub(r'^(?:the|a|an)\s+', '', s)
    s = re.sub(r'[.!?,;:]+$', '', s).strip()
    if s in ('yes.', 'yes'):
        return 'yes'
    if s in ('no.', 'no'):
        return 'no'
    return s


def extract_answer(text):
    """
    ``(answer, answered)``: the normalised text after the last answer field
    (first line only) with `answered` True; else the normalised last
    non-empty line with `answered` False; ``(None, False)`` for empty text.
    """
    fields = _FIELD.findall(text)
    if fields:
        return normalise(fields[-1].split('\n')[0]), True
    lines = [ln for ln in text.split('\n') if ln.strip()]
    if lines:
        return normalise(lines[-1]), False
    return None, False


def is_correct(text, gold):
    pred, _ = extract_answer(text)
    return pred is not None and pred == normalise(gold)


# ----------------------------------------------------------------------------
# entry point: build, validate, describe
# ----------------------------------------------------------------------------

def main():
    import argparse
    import collections
    import statistics
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--show', type=int, default=0, help='print N training rows with all targets')
    ap.add_argument('--rebuild', action='store_true', help='rebuild the cached problem set')
    ap.add_argument('--dataset', choices=DATASETS, default='sample')
    ap.add_argument('--gen-people', type=int, default=DataConfig.gen_people)
    ap.add_argument('--gen-rooms', type=int, default=DataConfig.gen_rooms)
    ap.add_argument('--gen-moves', type=int, default=DataConfig.gen_moves)
    ap.add_argument('--gen-stories', type=int, default=DataConfig.gen_stories)
    ap.add_argument('--gen-interesting-only', action='store_true')
    args = ap.parse_args()
    dcfg = DataConfig(dataset=args.dataset, gen_people=args.gen_people, gen_rooms=args.gen_rooms,
                      gen_moves=args.gen_moves, gen_stories=args.gen_stories,
                      gen_interesting_only=args.gen_interesting_only)
    if args.rebuild:
        for f in os.listdir(DATA_CACHE) if os.path.isdir(DATA_CACHE) else []:
            if f.startswith('all_v'):
                os.remove(os.path.join(DATA_CACHE, f))
    problems = all_problems(dcfg)
    stats_path = _all_path(dcfg)[:-5] + '_stats.json'
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            print('build:', json.load(f))
    tr, te = get_problems(dcfg, 'train'), get_problems(dcfg, 'test')
    pool = train_pool(dcfg)
    print(f'usable rows {len(problems)} in {len({p.story_id for p in problems})} stories; '
          f'test {len(te)} rows / {len({p.story_id for p in te})} stories; '
          f'train {len(tr)} rows / {len({p.story_id for p in tr})} stories (pool {len(pool)})')
    assert not ({p.story_id for p in te} & {p.story_id for p in pool}), 'story leak'
    for name, rows in (('test', te), ('train', tr)):
        c = collections.Counter(p.qtype + ('*' if p.false_belief else '') for p in rows)
        print(f'  {name} question types:', dict(sorted(c.items())))
        print(f'  {name} answers:', collections.Counter(p.gold for p in rows).most_common(6))
        print(f'  {name} people:', dict(collections.Counter(p.num_people for p in rows)),
              'rooms:', dict(collections.Counter(p.num_rooms for p in rows)))
    words = lambda f, rows: statistics.mean(len(f(p).split()) for p in rows)   # noqa: E731
    mx = lambda f, rows: max(len(f(p).split()) for p in rows)                  # noqa: E731
    for name, fn in TARGETS.items():
        print(f'  {name}: {words(fn, tr):.0f} words mean, {mx(fn, tr)} max (train)')
    print(f'  story: {words(lambda p: p.story, tr):.0f} words mean, {mx(lambda p: p.story, tr)} max; '
          f'steps {statistics.mean(len(p.steps) for p in tr):.1f} mean; state lines per '
          f'trace {statistics.mean(len(blocks(len(p.steps), FOCUS_STRIDE)) for p in tr):.1f} mean')
    for p in tr[:args.show]:
        print('=' * 72)
        print(user_message(p))
        for name, fn in TARGETS.items():
            print('-' * 26 + f' {name} ' + '-' * 26)
            print(fn(p))
    for t in ['Answer: yes', '**Answer:** The leather briefcase.', 'Answer: (does not know about it)',
              'answer: in the hotel lobby', 'I think it is the box']:
        print(repr(t), extract_answer(t))


if __name__ == '__main__':
    main()
