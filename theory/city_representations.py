"""
Three schematic panels: the minimal DAG of a set of reasoning traces can be
larger than the simpler world-model the traces actually follow.

One underlying city map generates start-to-end navigation traces.

(i)   Underlying world-model -- the compact map: states = locations, streets =
                                absolute moves.  7 locations, 7 edges.
(ii)  Minimal DAG           -- the traces recorded as RELATIVE turns (F/L/R),
                               merged so each node is the minimal context that
                               fixes the possible suffixes.  Because relative
                               turns hide the heading, location 5 (reached with
                               two headings) splits into 5 and 5', giving MORE
                               edges (9) than the world-model.  This edge count
                               is what next-token prediction must learn.
(iii) De Bruijn DAG         -- the efficient encoding: each token carries
                               (location, move), so no hidden state remains and
                               the minimal DAG folds back onto the world-model
                               (7 edges); the split of 5 disappears.

Style (fonts, palette) matches debruijn_dag_figure.py.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, Circle

# --- palette / style (shared with debruijn_dag_figure.py) --------------------
START_COLOR = '#2a9d8f'
END_COLOR = '#e76f51'
MID_COLOR = '#f4f1ec'
LOC_COLOR = '#c9d6e5'        # neutral "a location" fill for the automaton map
EDGE_COLOR = '#3d5a80'
HL_COLOR = '#c1121f'         # highlight: the worked-example convergence
FONTSIZE = 12
R = 0.34                     # node radius (data units, equal-aspect axes)
FIG_W, FIG_H = 3.0, 2.5      # every panel is exactly this size (inches)
# All panels share one data window (aspect FIG_W:FIG_H) so the scale -- and
# therefore node sizes and every font -- is identical across the three.

mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'font.size': FONTSIZE})


# --- underlying world-model (finite automaton) -------------------------------
# H -> A -> {B,C} -> D -> {E,F}, with E, F the two destinations (sinks).
# Convergence at D: reach it going S from B or E from C (different moves).
AUT_EDGES = [
    ('H', 'A'), ('A', 'B'), ('A', 'C'),
    ('B', 'D'), ('C', 'D'),
    ('D', 'E'), ('D', 'F'),
]
# Direction tokens carried on the edges (a token = a direction of travel).
# These are consistent with the grid MAP_POS below (streets run N/S/E/W).
AUT_TOKENS = {
    ('H', 'A'): 'E', ('A', 'B'): 'E', ('A', 'C'): 'S',
    ('B', 'D'): 'S', ('C', 'D'): 'E',
    ('D', 'E'): 'E', ('D', 'F'): 'S',
}
# Map layout: intersections on a city grid, edges are one-way streets.
MAP_POS = {
    'H': (0.0, 1.8), 'A': (1.8, 1.8), 'B': (3.6, 1.8),
    'C': (1.8, 0.0), 'D': (3.6, 0.0), 'E': (5.4, 0.0),
    'F': (3.6, -1.8),
}
# DAG layout: left-to-right diamond, used by the De Bruijn panel.
AUT_POS = {
    'H': (0, 0), 'A': (1.6, 0),
    'B': (3.2, 1.15), 'C': (3.2, -1.15), 'D': (4.8, 0),
    'E': (6.4, 1.15), 'F': (6.4, -1.15),
}
START, ENDS = 'H', {'E', 'F'}

# nodes are shown as numbers so their labels never clash with the N/S/E/W
# direction tokens on the edges
NODE_LABEL = {'H': '1', 'A': '2', 'B': '3', 'C': '4', 'D': '5', 'E': '6',
              'F': '7'}


def role(loc):
    if loc == START:
        return 'start'
    if loc in ENDS:
        return 'end'
    return 'mid'


ROLE_FILL = {'start': START_COLOR, 'end': END_COLOR, 'mid': MID_COLOR}
ROLE_EDGE = {'start': '#1d6f65', 'end': '#b23a20', 'mid': '#8a8578'}
ROLE_TEXT = {'start': 'white', 'end': 'white', 'mid': '#3a372f'}


# --- generic drawing ---------------------------------------------------------

def draw_panel(pos, edges, fname, *, caption=None, role_of=None,
               tokens=None, hl_nodes=frozenset(), hl_edges=frozenset(),
               label_pos=0.5, map_mode=False, window=None):
    """pos: name->(x,y); edges: list of (u,v); role_of: name->role or None
    for the neutral 'location' style; label(name) strips a '#k' suffix.
    All panels share one data ``window`` so fonts/nodes match across them.
    map_mode draws a faint street grid and labels streets above (horizontal)
    or to the right (vertical)."""
    tokens = tokens or {}
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # faint street grid behind everything (map look)
    if map_mode:
        gx = sorted({round(x, 3) for x, _ in pos.values()})
        gy = sorted({round(y, 3) for _, y in pos.values()})
        for x in gx:
            ax.plot([x, x], [min(gy), max(gy)], color='#e3e7ee', lw=6,
                    solid_capstyle='round', zorder=0)
        for y in gy:
            ax.plot([min(gx), max(gx)], [y, y], color='#e3e7ee', lw=6,
                    solid_capstyle='round', zorder=0)

    # nodes (patches first so arrows can clip to their boundary)
    patch = {}
    for name, (x, y) in pos.items():
        if role_of is None:
            fc, ec, tc, lw = LOC_COLOR, EDGE_COLOR, '#22303f', 1.2
        else:
            r = role_of(name)
            fc, ec, tc, lw = ROLE_FILL[r], ROLE_EDGE[r], ROLE_TEXT[r], 1.3
        if name in hl_nodes:
            ec, lw = HL_COLOR, 2.4
        p = Circle((x, y), R, facecolor=fc, edgecolor=ec, lw=lw, zorder=3)
        ax.add_patch(p)
        patch[name] = p
        base = name.split('#')[0]
        ax.text(x, y, NODE_LABEL.get(base, base), ha='center', va='center',
                zorder=4, fontsize=FONTSIZE - 2, family='monospace', color=tc,
                weight='bold')

    # edges
    for u, v in edges:
        hl = (u, v) in hl_edges
        arrow = FancyArrowPatch(
            pos[u], pos[v], arrowstyle='-|>', mutation_scale=6,
            lw=2.2 if hl else 1.5, color=HL_COLOR if hl else EDGE_COLOR,
            alpha=0.95, shrinkA=1, shrinkB=1,
            patchA=patch[u], patchB=patch[v],
            zorder=2.5 if hl else 2)
        ax.add_patch(arrow)
        tok = tokens.get((u, v))
        if tok is not None:
            (xu, yu), (xv, yv) = pos[u], pos[v]
            bx, by = xu + (xv - xu) * label_pos, yu + (yv - yu) * label_pos
            dx, dy = xv - xu, yv - yu
            n = np.hypot(dx, dy) + 1e-9
            off = 0.36
            if map_mode:
                # horizontal street -> label above; vertical -> label right
                px, py = (0.0, 1.0) if abs(dx) >= abs(dy) else (1.0, 0.0)
            else:
                # perpendicular offset, pushed to the side away from the y=0
                # centerline so labels never fall between the branch nodes
                px, py = -dy / n, dx / n
                side = 1.0 if by >= -1e-9 else -1.0
                if py * side < 0:
                    px, py = -px, -py
            tx, ty = bx + px * off, by + py * off
            ax.text(tx, ty, tok, ha='center', va='center', zorder=5,
                    fontsize=FONTSIZE - 5, style='italic',
                    color=HL_COLOR if hl else '#5a564c')

    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    pad = R + 0.45
    cx = (min(xs) + max(xs)) / 2
    if caption:
        ax.text(cx, min(ys) - pad, caption, ha='center', va='top',
                fontsize=FONTSIZE, color='#22303f')
    # content block (nodes + pad, plus the caption line below)
    bx0, bx1 = min(xs) - pad, max(xs) + pad
    by0 = min(ys) - pad - (0.85 if caption else 0)
    by1 = max(ys) + pad
    W, H = window if window is not None else (bx1 - bx0, by1 - by0)
    ccx, ccy = (bx0 + bx1) / 2, (by0 + by1) / 2
    ax.set_xlim(ccx - W / 2, ccx + W / 2)
    ax.set_ylim(ccy - H / 2, ccy + H / 2)
    ax.axis('off')
    fig.set_size_inches(FIG_W, FIG_H)
    ax.set_position([0, 0, 1, 1])   # axes fill the figure exactly
    os.makedirs(os.path.dirname(fname) or '.', exist_ok=True)
    fig.savefig(fname + '.pdf')
    fig.savefig(fname + '.png', dpi=220)
    plt.close(fig)


# --- (i) underlying world-model ----------------------------------------------

def figure_world_model(outdir):
    # the compact city map: states = locations, streets = absolute moves.
    # location 5 is the convergence that will split in the minimal DAG.
    return dict(
        pos=MAP_POS, edges=AUT_EDGES,
        fname=os.path.join(outdir, 'city_world_model'),
        role_of=None, tokens=AUT_TOKENS,
        map_mode=True, hl_nodes={'D'}, hl_edges={('B', 'D'), ('C', 'D')})


# --- (ii) minimal DAG of the reasoning traces --------------------------------

def figure_minimal_dag(outdir):
    """Reasoning traces recorded as RELATIVE turns (F/L/R).  A node is the
    minimal past context that fixes the set of possible suffixes; equal-label
    nodes merge.  Location 5 is reached with two headings (via 3 heading S,
    via 4 heading E); afterwards the *same* turn leads to different goals, so
    the two contexts have different suffix-sets and cannot merge -- location 5
    splits into 5 and 5'.  Result: more edges than the world-model (9 vs 7)."""
    pos = {
        '1': (0.0, 0.0), '2': (1.5, 0.0),
        '3': (3.0, 1.15), '4': (3.0, -1.15),
        '5': (4.5, 1.15), "5'": (4.5, -1.15),
        '6': (6.0, 1.15), '7': (6.0, -1.15),
    }
    edges = [('1', '2'), ('2', '3'), ('2', '4'), ('3', '5'), ('4', "5'"),
             ('5', '6'), ('5', '7'), ("5'", '6'), ("5'", '7')]
    # relative-turn token on each edge (depends on the arrival heading)
    tokens = {('1', '2'): 'F', ('2', '3'): 'F', ('2', '4'): 'R',
              ('3', '5'): 'R', ('4', "5'"): 'L',
              ('5', '6'): 'L', ('5', '7'): 'F',
              ("5'", '6'): 'F', ("5'", '7'): 'R'}
    role_of = lambda n: ('start' if n.split('#')[0] == '1'
                         else 'end' if n.split('#')[0] in ('6', '7')
                         else 'mid')
    # the two 'F' edges land on different goals -> the ambiguity that forces
    # the split; highlight them and the two split nodes.
    return dict(pos=pos, edges=edges,
                fname=os.path.join(outdir, 'city_minimal_dag'),
                role_of=role_of, tokens=tokens, label_pos=0.34,
                hl_nodes={'5', "5'"}, hl_edges={('5', '7'), ("5'", '6')})


# --- (iii) De Bruijn DAG: the efficient encoding -----------------------------

def figure_debruijn(outdir):
    # Efficient encoding: every token carries (current location, next move),
    # so no hidden heading remains -- the minimal DAG folds back onto the
    # world-model (7 edges), and the split of location 5 disappears.
    tokens = {(u, v): f'{NODE_LABEL[u]},{AUT_TOKENS[(u, v)]}'
              for (u, v) in AUT_EDGES}
    return dict(
        pos=AUT_POS, edges=AUT_EDGES,
        fname=os.path.join(outdir, 'city_debruijn_dag'),
        role_of=role, tokens=tokens,
        hl_nodes={'D'}, hl_edges={('B', 'D'), ('C', 'D')})


def content_window(panels):
    """Smallest FIG_W:FIG_H data window containing every panel's content
    block, so all panels share one scale."""
    pad = R + 0.45
    need_w = need_h = 0.0
    for p in panels:
        xs = [x for x, _ in p['pos'].values()]
        ys = [y for _, y in p['pos'].values()]
        need_w = max(need_w, (max(xs) - min(xs)) + 2 * pad)
        need_h = max(need_h, (max(ys) - min(ys)) + 2 * pad
                     + (0.85 if p.get('caption') else 0))
    ar = FIG_W / FIG_H
    W = max(need_w, need_h * ar)
    return W, W / ar


if __name__ == '__main__':
    outdir = 'figures'
    panels = [figure_world_model(outdir), figure_minimal_dag(outdir),
              figure_debruijn(outdir)]
    window = content_window(panels)      # shared -> identical scale + fonts
    for p in panels:
        draw_panel(**p, window=window)
    print('wrote city_world_model, city_minimal_dag, city_debruijn_dag '
          f'(.pdf/.png, each {FIG_W}x{FIG_H} in) to', outdir)
