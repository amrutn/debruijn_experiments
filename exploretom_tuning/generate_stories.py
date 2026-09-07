"""
Generate larger ExploreToM stories locally, with no LLM.

The released sample tops out at 4 people, 4 moves and 2 rooms, and a
fine-tuned 1.5B model reaches ~0.91 on it, which is too saturated to separate
the reasoning formats. This module drives the authors' belief tracker
(`exploretom_data.tracker`) directly: it samples a random *valid* sequence of
tracker actions -- people entering and leaving rooms, moving objects into
containers or other rooms, telling each other things privately or out loud,
and peeking or being distracted -- of whatever size is asked for. Every action
is applied to the tracker, so only ones that pass the tracker's preconditions
are kept, and the resulting `story_script` is the templated story with an exact
world-and-belief state behind it.

For each generated story the tracker's `QuestionGenerator` produces the
first- and second-order location, container and knowledge questions with their
ground-truth answers. The stories are emitted as rows in exactly the released
CSV's schema (the columns `exploretom_data.build_problems` reads), so generated
data flows through the same parse -> replay -> verify -> ledger pipeline as the
released sample: the generated text is parsed back into actions, replayed, and
every emitted answer is re-derived and checked, so a bug in generation shows up
as a dropped story rather than a wrong label. Object-state updates and the
factual (`memory`/`ground_truth`) questions are not generated -- the former are
unparseable and excluded everywhere, the latter need a replayed action list the
verification rebuilds itself.

Difficulty is set by `DataConfig`'s ``gen_*`` fields (people, rooms, objects,
containers, topics, moves); a difficulty sweep varies them across runs. The
generation is fully determined by ``gen_seed`` and those sizes, which are part
of the dataset's cache key, so a generated problem set is reproducible.
"""

import random

from exploretom_data import tracker, GEN_VERSION  # noqa: F401 (re-exported for callers)

# Entity pools. Names are the given names the released stories use; the objects,
# containers and rooms are ordinary, unambiguous nouns so the templated
# sentences read cleanly. Topics are abstract things a person can "hear about".
NAMES = [
    'Alexander', 'Amelia', 'Andrew', 'Aubrey', 'Ava', 'Benjamin', 'Brooklyn', 'Caleb',
    'Charlotte', 'Chloe', 'Clayton', 'Cooper', 'Daniel', 'Dominic', 'Dylan', 'Elijah',
    'Emily', 'Eric', 'Gabriella', 'Hailey', 'Isabella', 'James', 'Julia', 'Kaylee',
    'Landon', 'Liam', 'Mia', 'Nicholas', 'Olivia', 'Owen', 'Paige', 'Samantha',
    'Sophia', 'Taylor', 'Tessa', 'Tristan', 'William', 'Wyatt', 'Zoe', 'Avery',
]
OBJECTS = [
    'silver letter opener', 'pocket watch', 'compass', 'harmonica', 'stuffed rabbit',
    'silver locket', 'walkie-talkie', 'tactical flashlight', 'leather notebook', 'brass key',
    'wooden spoon', 'snow globe', 'pocket knife', 'antique map', 'glass paperweight',
    'ceramic mug', 'wristwatch', 'fountain pen', 'chess piece', 'coin purse',
]
CONTAINERS = [
    'wooden chest', 'leather satchel', 'plastic storage bin', 'cardboard box', 'metal toolbox',
    'paper bag', 'wicker basket', 'desk drawer', 'glass display case', 'canvas backpack',
    'velvet pouch', 'steel locker',
]
ROOMS = [
    'hotel lobby', 'green room', 'main bookstore floor', 'visitor center', 'control room',
    'staff lounge', 'production office', 'artist studio', 'reading room', 'storage closet',
    'conference hall', 'workshop', 'gallery', 'back office', 'pantry', 'observation deck',
]
# bare nouns: the sentence template already prefixes "about the"/"that"
TOPICS = [
    'catering schedule', 'ticket refund policy', 'seating arrangement', 'fire drill plan',
    'guest list', 'maintenance backlog', 'budget shortfall', 'security code rotation',
    'travel itinerary', 'menu changes', 'volunteer roster', 'parking permits',
]


def _known_room(bt, entity):
    r = bt.world_state.get(entity, tracker().FullBeliefTracker.ROOM_LOCATION)
    return r if r is not None and not r.startswith('not(') else None


def _peek_kwargs(bt, actor, room, people, rng, p_peek):
    """Optionally make one outsider peek or one witness be distracted; the
    tracker's preconditions reject an impossible choice, so the caller retries."""
    if rng.random() >= p_peek:
        return {}
    ROOM = tracker().FullBeliefTracker.ROOM_LOCATION
    witnesses = [w for w in people if w != actor and bt.world_state.get(w, ROOM) == room]
    outsiders = [w for w in people if w != actor and bt.world_state.get(w, ROOM) != room]
    if outsiders and (not witnesses or rng.random() < 0.5):
        return {'people_peeking': [rng.choice(outsiders)]}
    if witnesses:
        return {'people_distracted': [rng.choice(witnesses)]}
    return {}


def _attempt(bt, kind, pools, rng, p_peek):
    """Try one action of `kind` with random valid-looking arguments; return
    True if the tracker accepted it (and thus appended to the story)."""
    BT = tracker().FullBeliefTracker
    ROOM, CONT = BT.ROOM_LOCATION, BT.CONTAINER_LOCATION
    people, rooms, objects, containers, topics = pools
    here = [p for p in people if _known_room(bt, p)]
    if kind == 'enter':
        p = rng.choice(people)
        return bt.enter_room(p, rng.choice(rooms))
    if kind == 'leave':
        if not here:
            return False
        p = rng.choice(here)
        return bt.leave_room(p, _known_room(bt, p))
    if kind == 'move_container':
        if not here:
            return False
        p = rng.choice(here)
        room = _known_room(bt, p)
        return bt.move_object_container(p, rng.choice(objects), rng.choice(containers),
                                       **_peek_kwargs(bt, p, room, people, rng, p_peek))
    if kind == 'move_room':
        movable = [(p, o) for p in here for o in objects
                   if bt.world_state.get(o, ROOM) == _known_room(bt, p)]
        if not movable:
            return False
        p, o = rng.choice(movable)
        dest = rng.choice([r for r in rooms if r != _known_room(bt, p)] or rooms)
        return bt.move_object_room(p, o, dest)
    if kind == 'private_abstract':
        if len(people) < 2:
            return False
        a, b = rng.sample(people, 2)
        room = _known_room(bt, a)
        return bt.private_communication(a, b, rng.choice(topics),
                                        **_peek_kwargs(bt, a, room, people, rng, p_peek) if room else {})
    if kind == 'private_world':
        speakers = [p for p in here
                    if any(_valid_loc(bt.first_order_beliefs[p].get(o, CONT)) for o in objects)]
        if not speakers:
            return False
        a = rng.choice(speakers)
        b = rng.choice([p for p in people if p != a])
        o = rng.choice([o for o in objects if _valid_loc(bt.first_order_beliefs[a].get(o, CONT))])
        loc = bt.first_order_beliefs[a].get(o, CONT)
        room = _known_room(bt, a)
        return bt.private_communication(a, b, (o, CONT, loc, True),
                                        **_peek_kwargs(bt, a, room, people, rng, p_peek) if room else {})
    if kind == 'broadcast':
        speakers = [p for p in here
                    if sum(1 for w in people if _known_room(bt, w) == _known_room(bt, p)) > 1]
        if not speakers:
            return False
        p = rng.choice(speakers)
        return bt.broadcast_communication(p, rng.choice(topics))
    return False


def _valid_loc(v):
    return v is not None and not v.startswith('not(')


_KINDS = ['move_container', 'move_room', 'enter', 'leave', 'private_abstract',
          'private_world', 'broadcast']


def sample_story(rng, dcfg):
    """A random valid story of roughly `dcfg.gen_moves` actions, as a tracker
    with its `story_script` filled in, or None if too few actions succeeded."""
    BT = tracker().FullBeliefTracker
    pool = lambda src, n: rng.sample(src, min(n, len(src)))          # noqa: E731
    pools = (pool(NAMES, dcfg.gen_people), pool(ROOMS, dcfg.gen_rooms),
             pool(OBJECTS, dcfg.gen_objects), pool(CONTAINERS, dcfg.gen_containers),
             pool(TOPICS, dcfg.gen_topics))
    people, rooms, objects, containers, _ = pools
    bt = BT()
    for p in people:                                    # everyone starts in a room
        bt.enter_room(p, rng.choice(rooms))
    for o in objects:                                   # place each object once
        here = [p for p in people if _known_room(bt, p)]
        if here:
            p = rng.choice(here)
            bt.move_object_container(p, o, rng.choice(containers))
    weights = {'move_container': 3, 'move_room': 2, 'enter': 2, 'leave': 1,
               'private_abstract': 2, 'private_world': 3, 'broadcast': 2}
    comm = {'private_abstract', 'private_world', 'broadcast'}
    target = len(bt.story_script) + dcfg.gen_moves
    tries = 0
    while len(bt.story_script) < target and tries < dcfg.gen_moves * 40:
        tries += 1
        w = [weights[k] * (dcfg.gen_comm if k in comm else (1 - dcfg.gen_comm)) for k in _KINDS]
        _attempt(bt, rng.choices(_KINDS, weights=w)[0], pools, rng, dcfg.gen_peek)
    return bt if len(bt.story_script) >= max(4, target // 2) else None


def story_rows(bt, dcfg, rng):
    """
    The first- and second-order questions of a generated story as rows in the
    released CSV schema (the columns `exploretom_data.build_problems` reads).
    Factual questions are not emitted (see the module docstring). At most
    `dcfg.gen_q_cap` questions are kept per story, interesting ones (answer
    depends on who is asked) first, so a large story does not swamp the
    build; a later `q_per_story` caps again per split.
    """
    story = ' '.join(bt.story_script)
    qg = tracker().QuestionGenerator(bt)
    cand = []
    for order in (1, 2):
        for q, a, cond, meta in qg.main(order, expand_relation_type_info=True):
            interesting = bool(meta[0]) if meta else False
            if dcfg.gen_interesting_only and not interesting:
                continue
            cand.append((interesting, order, q, str(a), repr(cond)))
    rng.shuffle(cand)
    cand.sort(key=lambda c: not c[0])           # interesting first, else stable-shuffled
    rows = []
    for interesting, order, q, a, params in cand[:dcfg.gen_q_cap]:
        rows.append({
            'story_structure': story, 'infilled_story': '', 'question': q,
            'expected_answer': a, 'qprop=params': params, 'qprop=nth_order': str(order),
            'param=num_people': str(len(bt.people)), 'param=num_rooms': str(len(bt.rooms)),
            'param=story_type': 'generated',
        })
    return rows


def generate_rows(dcfg):
    """
    Rows for `dcfg.gen_stories` distinct generated stories, in the released
    schema. Deterministic in `dcfg.gen_seed` and the sizes.
    """
    rng = random.Random(dcfg.gen_seed)
    rows, seen, made, guard = [], set(), 0, 0
    while made < dcfg.gen_stories and guard < dcfg.gen_stories * 20:
        guard += 1
        bt = sample_story(rng, dcfg)
        if bt is None:
            continue
        story = ' '.join(bt.story_script)
        if story in seen:
            continue
        r = story_rows(bt, dcfg, rng)
        if not r:
            continue
        seen.add(story)
        rows.extend(r)
        made += 1
    return rows


def main():
    import argparse
    import collections
    import statistics
    from exploretom_data import DataConfig
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--people', type=int, default=DataConfig.gen_people)
    ap.add_argument('--rooms', type=int, default=DataConfig.gen_rooms)
    ap.add_argument('--moves', type=int, default=DataConfig.gen_moves)
    ap.add_argument('--stories', type=int, default=20)
    ap.add_argument('--peek', type=float, default=DataConfig.gen_peek)
    ap.add_argument('--interesting-only', action='store_true')
    ap.add_argument('--show', type=int, default=1)
    args = ap.parse_args()
    dcfg = DataConfig(dataset='generated', gen_people=args.people, gen_rooms=args.rooms,
                      gen_moves=args.moves, gen_stories=args.stories, gen_peek=args.peek,
                      gen_interesting_only=args.interesting_only)
    rows = generate_rows(dcfg)
    stories = collections.OrderedDict()
    for r in rows:
        stories.setdefault(r['story_structure'], []).append(r)
    sizes = [len(s.split('. ')) for s in stories]
    print(f'{len(stories)} stories, {len(rows)} questions '
          f'({statistics.mean(len(v) for v in stories.values()):.1f} per story); '
          f'sentences mean/max {statistics.mean(sizes):.1f}/{max(sizes)}')
    print('question orders:', collections.Counter(r['qprop=nth_order'] for r in rows))
    for story, qs in list(stories.items())[:args.show]:
        print('=' * 72)
        print(story)
        for r in qs[:8]:
            print(f'  [{r["qprop=nth_order"]}] {r["question"]}  ->  {r["expected_answer"]}')


if __name__ == '__main__':
    main()
