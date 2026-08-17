"""
Schematic of the grid-navigation experiment.

Left: an ``m x m`` grid (both sides labelled ``m``) with a shortest path drawn
from a start cell to a goal cell through intermediate nodes; edges carry the
cardinal move (N/E/S/W) and nodes carry their integer label.

Right: the three reasoning-trace encodings of that same path, drawn as token
chains --
  De Bruijn (k=1)  the current cell is re-emitted after every move,
  k = 2            the cell is re-emitted every other move,
  standard         only the start and goal cells bookend the moves.

Style (serif fonts, teal/orange/cream node palette, circular nodes with -|>
arrows) matches the De Bruijn graph figures in ../theory.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch

# --- palette / style (shared with theory/debruijn_dag_figure.py) -------------
START_COLOR = '#2a9d8f'
GOAL_COLOR = '#e76f51'
MID_COLOR = '#f4f1ec'
OFF_COLOR = '#dfe4ea'
EDGE_COLOR = '#3d5a80'
DIR_COLOR = '#8a6d3b'
ROLE = {
    'start': dict(fc=START_COLOR, ec='#1d6f65', tc='white'),
    'goal':  dict(fc=GOAL_COLOR,  ec='#b23a20', tc='white'),
    'mid':   dict(fc=MID_COLOR,   ec='#8a8578', tc='#3a372f'),
}
mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})

# --- the example path --------------------------------------------------------
M = 5                                   # drawn grid; the sides are labelled ``m``
DELTA = {'N': (-1, 0), 'E': (0, 1), 'S': (1, 0), 'W': (0, -1)}


def cell_pos(cell):
    """Drawing position: row 0 at the top, N points up, E points right."""
    r, c = divmod(cell, M)
    return (c, M - 1 - r)


def step(cell, d):
    r, c = divmod(cell, M)
    dr, dc = DELTA[d]
    return (r + dr) * M + (c + dc)


START = 16                              # (row 3, col 1)
MOVES = ['E', 'N', 'E', 'N']            # a staircase with turns
CELLS = [START]
for _d in MOVES:
    CELLS.append(step(CELLS[-1], _d))
GOAL = CELLS[-1]                        # CELLS = [16, 17, 12, 13, 8]


def role(i):
    return 'start' if i == 0 else ('goal' if i == len(CELLS) - 1 else 'mid')


def build_trace(interval):
    """Token list [('cell', id, role) | ('dir', letter)] for an emission
    interval (1 = De Bruijn, None = standard), matching generate_data.encode."""
    toks = [('cell', CELLS[0], 'start')]
    since, n = 0, len(MOVES)
    for i, d in enumerate(MOVES):
        toks.append(('dir', d))
        since += 1
        if i == n - 1:
            toks.append(('cell', CELLS[i + 1], 'goal'))
        elif interval is not None and since == interval:
            toks.append(('cell', CELLS[i + 1], 'mid'))
            since = 0
    return toks


# --- grid panel --------------------------------------------------------------

def draw_grid(ax):
    ax.set_aspect('equal')
    ax.axis('off')
    # faint lattice
    for r in range(M):
        for c in range(M):
            x, y = cell_pos(r * M + c)
            if c < M - 1:
                ax.plot([x, x + 1], [y, y], color=OFF_COLOR, lw=1.3, zorder=0)
            if r < M - 1:
                ax.plot([x, x], [y, y - 1], color=OFF_COLOR, lw=1.3, zorder=0)
    # off-path nodes as small dots
    path = set(CELLS)
    for cell in range(M * M):
        if cell not in path:
            x, y = cell_pos(cell)
            ax.add_patch(Circle((x, y), 0.075, facecolor='#c9d0d8',
                                edgecolor='none', zorder=1))
    # path edges with move labels
    for i, d in enumerate(MOVES):
        (x0, y0), (x1, y1) = cell_pos(CELLS[i]), cell_pos(CELLS[i + 1])
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                     mutation_scale=7, lw=1.4, color=EDGE_COLOR,
                     shrinkA=6, shrinkB=6, zorder=2))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        nx, ny = -dy, dx
        nn = np.hypot(nx, ny)
        ax.text(mx + nx / nn * 0.24, my + ny / nn * 0.24, d, fontsize=7,
                style='italic', ha='center', va='center', color=DIR_COLOR, zorder=5)
    # path nodes
    for i, cell in enumerate(CELLS):
        x, y = cell_pos(cell)
        st = ROLE[role(i)]
        ax.add_patch(Circle((x, y), 0.26, facecolor=st['fc'], edgecolor=st['ec'],
                            lw=1.2, zorder=3))
        ax.text(x, y, str(cell), fontsize=7, ha='center', va='center',
                color=st['tc'], weight='bold', zorder=4)
    # start / goal captions
    sx, sy = cell_pos(START)
    ax.text(sx, sy - 0.46, 'start', fontsize=6.5, ha='center', va='top', color='#1d6f65')
    gx, gy = cell_pos(GOAL)
    ax.text(gx, gy + 0.46, 'goal', fontsize=6.5, ha='center', va='bottom', color='#b23a20')
    ax.set_xlim(-0.6, M - 0.4)
    ax.set_ylim(-0.6, M - 0.4)


# --- trace panel -------------------------------------------------------------

def trace_str(toks):
    """Flat, space-separated trace string, e.g. '16 E 17 N 12 E 13 N 8'."""
    return ' '.join(str(t[1]) if t[0] == 'cell' else t[1] for t in toks)


def draw_traces(ax, center_y=0.5):
    """Three compact example boxes stacked at the side of the grid, each sized to
    hug its content (widths from the actual rendered text). Input and Output share
    one monospace font size. The stack is vertically centred on `center_y`."""
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig = ax.figure
    fig.canvas.draw()                              # ensure a renderer exists
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def wfrac(t):                                  # rendered text width in x-fraction
        bb = t.get_window_extent(renderer=rend)
        (x0, _), (x1, _) = inv.transform([(bb.x0, 0), (bb.x1, 0)])
        return x1 - x0

    rows = [('De Bruijn (k=1)', build_trace(1)),
            ('k=2', build_trace(2)),
            ('Standard', build_trace(None))]
    fs, title_fs = 7.0, 7.5
    box_x, tx, box_h, gap = 0.02, 0.05, 0.175, 0.04
    y = center_y + (3 * box_h + 2 * gap) / 2       # centre the stack on center_y
    placed, max_right = [], 0.0
    for title, toks in rows:                       # pass 1: place text, measure widths
        y0 = y - box_h
        top = y0 + box_h
        ax.text(tx, top - 0.044, title, fontsize=title_fs, weight='bold',
                va='center', ha='left', color='#1a1a1a')
        t_in = ax.text(tx, top - 0.093, f'Input:  {START} -> {GOAL}', fontsize=fs,
                       family='monospace', color='#8a8a8a', va='center', ha='left')
        t_ol = ax.text(tx, top - 0.142, 'Output:', fontsize=fs, family='monospace',
                       color='#2a6fd6', va='center', ha='left')
        x_tr = tx + wfrac(t_ol) + 0.015
        t_tr = ax.text(x_tr, top - 0.142, trace_str(toks), fontsize=fs,
                       family='monospace', color='#1a1a1a', va='center', ha='left')
        max_right = max(max_right, tx + wfrac(t_in), x_tr + wfrac(t_tr))
        placed.append(y0)
        y = y0 - gap
    t_ch = ax.text(0, 0, 'M', fontsize=fs, family='monospace')
    ch_w = wfrac(t_ch)                             # one monospace character
    t_ch.remove()
    box_w = (max_right - box_x) + 0.02 + ch_w      # hug the text, plus one character
    for y0 in placed:                              # pass 2: one box per row, same width
        ax.add_patch(FancyBboxPatch((box_x, y0), box_w, box_h,
                     boxstyle='round,pad=0.006,rounding_size=0.02',
                     fc='white', ec='#c2c2c2', lw=1.0, transform=ax.transAxes, zorder=1))


def main():
    fig = plt.figure(figsize=(4.5, 2.5))
    fig.subplots_adjust(left=0.03, right=0.985, top=0.98, bottom=0.02)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.04)
    axg = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1])
    draw_grid(axg)
    fig.canvas.draw()
    # vertical centre of the grid (its middle cell), in the traces-axes fraction
    disp = axg.transData.transform(((M - 1) / 2.0, (M - 1) / 2.0))
    cy = axt.transAxes.inverted().transform(disp)[1]
    draw_traces(axt, center_y=cy)
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'navigation_schematic')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    fig.savefig(path + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('wrote', path + '.pdf/.png')


if __name__ == '__main__':
    main()
