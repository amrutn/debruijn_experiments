"""
Visualize a DAG produced by procedure F from a De Bruijn graph B(2, n).

Procedure F:
  1. Order the nodes according to a random permutation pi.
  2. Designate the last N nodes as sinks and the first S nodes as sources.
     Delete outgoing edges from sinks and incoming edges into sources.
  3. Keep an edge u -> v iff pi(u) < pi(v).
  4. Delete every node/edge that does not lie on a source-to-sink path.

The figure places nodes horizontally by topological order (the permutation
order restricted to surviving nodes), highlights sources and sinks, and
shades each edge by the number of source-to-sink paths passing through it.
"""

import itertools
import os
import random

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, Circle
import networkx as nx

# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------

def de_bruijn(k, n):
    """De Bruijn graph B(k, n): nodes are k-ary strings of length n,
    edge u -> v iff u[1:] == v[:-1]."""
    G = nx.DiGraph()
    alphabet = [str(c) for c in range(k)]
    nodes = [''.join(s) for s in itertools.product(alphabet, repeat=n)]
    G.add_nodes_from(nodes)
    for u in nodes:
        for c in map(str, range(k)):
            G.add_edge(u, u[1:] + c)
    return G


def pi_order(B, seed):
    """Step 1: the node permutation pi."""
    rng = random.Random(seed)
    order = list(B.nodes())
    rng.shuffle(order)
    return order


def procedure_F(B, S, N, seed):
    """Apply procedure F; returns (DAG, ordered surviving nodes, sources, sinks)."""
    order = pi_order(B, seed)                # step 1: permutation pi
    pi = {v: i for i, v in enumerate(order)}
    sources = set(order[:S])                 # step 2
    sinks = set(order[-N:])

    D = nx.DiGraph()
    D.add_nodes_from(B.nodes())
    for u, v in B.edges():
        if u in sinks or v in sources:       # step 2: cut edges at boundary
            continue
        if pi[u] < pi[v]:                    # step 3: respect the ordering
            D.add_edge(u, v)

    # step 4: keep only nodes/edges on some source-to-sink path
    fwd = set(sources)
    for s in sources:
        fwd |= nx.descendants(D, s)
    bwd = set(sinks)
    for t in sinks:
        bwd |= nx.ancestors(D, t)
    alive = fwd & bwd
    D = D.subgraph(alive).copy()

    ordered = sorted(D.nodes(), key=lambda v: pi[v])
    return D, ordered, sources & alive, sinks & alive


def edge_path_counts(D, sources, sinks, ordered):
    """Number of source-to-sink paths through each edge:
    paths(source -> u) * paths(v -> sink)."""
    f = {v: (1 if v in sources else 0) for v in ordered}   # from sources
    for v in ordered:
        for u in D.predecessors(v):
            f[v] += f[u]
    g = {v: (1 if v in sinks else 0) for v in ordered}     # to sinks
    for v in reversed(ordered):
        for w in D.successors(v):
            g[v] += g[w]
    return {(u, v): f[u] * g[v] for u, v in D.edges()}


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------

SRC_COLOR = '#2a9d8f'
SNK_COLOR = '#e76f51'
MID_COLOR = '#f4f1ec'

def layered_positions(D, ordered, sources, sinks):
    """Column = longest-path layer from the sources (sinks pushed to the
    last column); within a column, order by barycenter of predecessors."""
    col = {}
    for v in ordered:                        # ordered is topological
        preds = list(D.predecessors(v))
        col[v] = 0 if not preds else max(col[u] for u in preds) + 1
    inter_max = max((col[v] for v in ordered if v not in sinks), default=0)
    snk_col = max(inter_max + 1, max(col[v] for v in sinks))
    for v in sinks:
        col[v] = snk_col

    columns = {}
    for v in ordered:
        columns.setdefault(col[v], []).append(v)

    pos = {}
    for c in sorted(columns):
        members = columns[c]
        if c > 0:
            def bary(v):
                ys = [pos[u][1] for u in D.predecessors(v) if u in pos]
                return sum(ys) / len(ys) if ys else 0.0
            members.sort(key=bary)
        k = len(members)
        for i, v in enumerate(members):
            pos[v] = (c, (i - (k - 1) / 2))
    return pos, snk_col


def draw(D, ordered, sources, sinks, counts, fname_base,
         figsize=(3.6, 3), fontsize=12):
    mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                         'font.size': fontsize})
    pos, ncols = layered_positions(D, ordered, sources, sinks)
    # Node size in points so circles stay round without an equal aspect.
    label_fs = fontsize - 2
    node_d = 2.9 * label_fs                   # circle diameter (points)

    cmap = plt.cm.viridis
    vmax = max(counts.values())
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)

    # Node radius in data units per axis (the axes are not equal-aspect),
    # estimated from the fraction of the canvas the plot occupies.
    y_hi = max(y for _, y in pos.values())
    y_lo = min(y for _, y in pos.values())
    rx = node_d / 2 / (figsize[0] * 72 * 0.70) * (ncols + 1)
    ry = node_d / 2 / (figsize[1] * 72 * 0.80) * (y_hi - y_lo + 1.6)

    # Column-skipping edges try curvatures from flat outward (preferring
    # the side away from the middle row) and keep the first whose curve
    # clears every node. arc3: positive rad bows the curve downward.
    import numpy as np
    def route(a, b, obstacles):
        (xu, yu), (xv, yv) = a, b
        side = -1 if yu + yv >= 0 else 1
        cands = [m * s for m in (0.15, 0.24, 0.35, 0.5) for s in (side, -side)]
        a, b = np.array(a, float), np.array(b, float)
        t = np.linspace(0.04, 0.96, 25)[:, None]
        best, best_clear = cands[0], -1.0
        for rad in cands:
            dx, dy = b - a
            c = (a + b) / 2 + rad * np.array([dy, -dx])
            curve = (1 - t) ** 2 * a + 2 * t * (1 - t) * c + t ** 2 * b
            clear = min(np.hypot((curve[:, 0] - ox) / rx,
                                 (curve[:, 1] - oy) / ry).min()
                        for ox, oy in obstacles)
            if clear > 2.4:                  # >1 diameter + margin
                return rad
            if clear > best_clear:
                best, best_clear = rad, clear
        return best

    shrink = node_d / 2 + 1.5
    for (u, v), c in sorted(counts.items(), key=lambda kv: kv[1]):
        (xu, yu), (xv, yv) = pos[u], pos[v]
        span = xv - xu
        if span == 1:
            rad = 0.0
        else:
            obstacles = [pos[w] for w in ordered if w not in (u, v)]
            rad = route(pos[u], pos[v], obstacles)
        arrow = FancyArrowPatch(
            (xu, yu), (xv, yv),
            connectionstyle=f'arc3,rad={rad}',
            arrowstyle='-|>', mutation_scale=label_fs,
            lw=1.3 + 2.4 * (c / vmax),
            color=cmap(norm(c)), alpha=0.92,
            shrinkA=shrink, shrinkB=shrink,
            zorder=1 + c / vmax,
        )
        ax.add_patch(arrow)

    # Nodes on top
    for v in ordered:
        if v in sources:
            fc, ec, lw = SRC_COLOR, '#1d6f65', 1.6
        elif v in sinks:
            fc, ec, lw = SNK_COLOR, '#b23a20', 1.6
        else:
            fc, ec, lw = MID_COLOR, '#8a8578', 1.2
        ax.scatter([pos[v][0]], [pos[v][1]], s=node_d ** 2, facecolor=fc,
                   edgecolor=ec, linewidth=lw, zorder=3)
        dark = v in sources or v in sinks
        ax.text(pos[v][0], pos[v][1], v, ha='center', va='center', zorder=4,
                fontsize=label_fs, family='monospace',
                color='white' if dark else '#3a372f', weight='bold')

    # Legend
    ms = 0.85 * fontsize
    handles = [
        mpl.lines.Line2D([], [], marker='o', ls='', ms=ms, mfc=SRC_COLOR,
                         mec='#1d6f65', label='source'),
        mpl.lines.Line2D([], [], marker='o', ls='', ms=ms, mfc=MID_COLOR,
                         mec='#8a8578', label='intermediate'),
        mpl.lines.Line2D([], [], marker='o', ls='', ms=ms, mfc=SNK_COLOR,
                         mec='#b23a20', label='sink'),
    ]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.0),
              ncol=3, frameon=False, fontsize=fontsize - 3,
              handletextpad=0.3, columnspacing=0.7, borderaxespad=0.0)

    # Colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02, aspect=15)
    cb.set_label('source-to-sink paths', fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize - 2)
    cb.set_ticks(range(0, vmax + 1))

    ax.set_xlim(-0.5, ncols + 0.5)
    ax.set_ylim(y_lo - 1.05, y_hi + 0.55)
    ax.axis('off')
    y_arrow = y_lo - 0.75
    ax.annotate('', xy=(ncols + 0.45, y_arrow), xytext=(-0.45, y_arrow),
                arrowprops=dict(arrowstyle='-|>', color='#8a8578', lw=1.2))
    ax.text(ncols / 2, y_arrow - 0.12, r'topological order $\pi$',
            ha='center', va='top', fontsize=fontsize, color='#5a564c')

    fig.tight_layout()
    os.makedirs(os.path.dirname(fname_base) or '.', exist_ok=True)
    fig.savefig(fname_base + '.pdf', bbox_inches='tight')
    fig.savefig(fname_base + '.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def draw_debruijn(B, D, sources, sinks, fname_base,
                  figsize=(3.05, 2.95), fontsize=12):
    """Companion panel: the full De Bruijn graph on a circle, with the
    designated sources/sinks highlighted and the edges kept by procedure F
    emphasized over the deleted ones."""
    import numpy as np
    mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                         'font.size': fontsize})
    KEPT_COLOR, DEAD_COLOR = '#3d5a80', '#cdc8bb'

    nodes = sorted(B.nodes())
    R = 0.24                                  # node radius (data units)

    # Force-directed layout spread over the full square, then push any
    # crowded pair apart until every node has clear space around it.
    G = nx.Graph((u, v) for u, v in B.edges() if u != v)
    pos = nx.kamada_kawai_layout(G)
    pts = np.array([pos[v] for v in nodes])
    for d in range(2):                        # fill the square axis by axis
        lo, hi = pts[:, d].min(), pts[:, d].max()
        pts[:, d] = (pts[:, d] - lo) / (hi - lo) * 3.2 - 1.6
    d_min = 2 * R + 0.26
    for _ in range(80):
        moved = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                delta = pts[j] - pts[i]
                d = np.hypot(*delta)
                if d < d_min:
                    shift = delta / (d + 1e-9) * (d_min - d) / 2
                    pts[i] -= shift
                    pts[j] += shift
                    moved = True
        if not moved:
            break
    pos = {v: pts[i] for i, v in enumerate(nodes)}

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    # Nodes first: attached patches give the arrows exact boundary clipping.
    node_patch = {}
    for v in nodes:
        alive = v in D
        if v in sources:
            fc, ec, lw, tc = SRC_COLOR, '#1d6f65', 1.4, 'white'
        elif v in sinks:
            fc, ec, lw, tc = SNK_COLOR, '#b23a20', 1.4, 'white'
        elif alive:
            fc, ec, lw, tc = MID_COLOR, '#8a8578', 1.0, '#3a372f'
        else:                                # deleted in step 4
            fc, ec, lw, tc = 'white', DEAD_COLOR, 1.0, '#b5afa1'
        p = Circle(pos[v], R, facecolor=fc, edgecolor=ec, lw=lw, zorder=3)
        ax.add_patch(p)
        node_patch[v] = p
        ax.text(*pos[v], v, ha='center', va='center', zorder=4,
                fontsize=fontsize - 4.5, family='monospace',
                color=tc, weight='bold')

    # Self-loops (never survive step 3): faded arcs with arrowheads, placed
    # on the side of the node facing away from its neighbours.
    loop_extents = []
    for v in nodes:
        if B.has_edge(v, v):
            away = pos[v] - np.mean([pos[u] for u in G[v]], axis=0)
            away /= np.hypot(*away) + 1e-9
            th = np.arctan2(away[1], away[0])
            p1, p2 = (pos[v] + (R + 0.01) * np.array([np.cos(a), np.sin(a)])
                      for a in (th + 0.55, th - 0.55))
            d = p2 - p1
            # rad sign whose control point bulges away from the node
            rad = max((1.9, -1.9), key=lambda r: np.hypot(
                *((p1 + p2) / 2 + r * np.array([d[1], -d[0]]) - pos[v])))
            ctrl = (p1 + p2) / 2 + rad * np.array([d[1], -d[0]])
            loop_extents.append(0.25 * p1 + 0.5 * ctrl + 0.25 * p2)
            ax.add_patch(FancyArrowPatch(
                p1, p2, connectionstyle=f'arc3,rad={rad}',
                arrowstyle='-|>', mutation_scale=7, lw=0.9,
                color=DEAD_COLOR, alpha=0.9, shrinkA=0, shrinkB=0, zorder=1))

    # Edge routing: try curvatures from flat outward and keep the first
    # whose quadratic Bezier stays clear of every other node (arc3's
    # control point is midpoint + rad * (B - A rotated clockwise)).
    def route(a, b, obstacles, candidates):
        best, best_clear = candidates[0], -1.0
        t = np.linspace(0, 1, 32)[:, None]
        for rad in candidates:
            dx, dy = b - a
            c = (a + b) / 2 + rad * np.array([dy, -dx])
            curve = (1 - t) ** 2 * a + 2 * t * (1 - t) * c + t ** 2 * b
            clear = min(np.hypot(*(curve - o).T).min() for o in obstacles)
            if clear > R + 0.05:
                return rad
            if clear > best_clear:
                best, best_clear = rad, clear
        return best

    edges = [(u, v) for u, v in B.edges() if u != v]
    for u, v in sorted(edges, key=lambda e: D.has_edge(*e)):
        kept = D.has_edge(u, v)
        obstacles = [pos[w] for w in nodes if w not in (u, v)]
        if B.has_edge(v, u):                 # 2-cycle: same-sign rads split
            candidates = [0.2, 0.32, 0.45]
        else:
            candidates = [0, 0.15, -0.15, 0.28, -0.28, 0.42, -0.42]
        rad = route(pos[u], pos[v], obstacles, candidates)
        arrow = FancyArrowPatch(
            pos[u], pos[v],
            connectionstyle=f'arc3,rad={rad}',
            arrowstyle='-|>', mutation_scale=8,
            lw=1.5 if kept else 0.8,
            color=KEPT_COLOR if kept else DEAD_COLOR,
            alpha=0.95 if kept else 0.85,
            shrinkA=1, shrinkB=1,
            patchA=node_patch[u], patchB=node_patch[v],
            zorder=2 if kept else 1.5,
        )
        ax.add_patch(arrow)

    handles = [
        mpl.lines.Line2D([], [], color=DEAD_COLOR, lw=1.4,
                         label='deleted'),
    ]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.0),
              frameon=False, fontsize=fontsize - 3,
              handletextpad=0.4, borderaxespad=0.0, handlelength=1.4)

    pad = R + 0.13
    ext = np.vstack([pts, np.array(loop_extents)])
    ax.set_xlim(ext[:, 0].min() - pad, ext[:, 0].max() + pad)
    ax.set_ylim(ext[:, 1].min() - pad, ext[:, 1].max() + pad)
    ax.axis('off')

    fig.tight_layout()
    os.makedirs(os.path.dirname(fname_base) or '.', exist_ok=True)
    fig.savefig(fname_base + '.pdf', bbox_inches='tight')
    fig.savefig(fname_base + '.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------

def stats(seed, n=4, S=3, N=3):
    B = de_bruijn(2, n)
    D, ordered, src, snk = procedure_F(B, S, N, seed)
    if not src or not snk:
        return None
    counts = edge_path_counts(D, src, snk, ordered)
    return D, ordered, src, snk, counts


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'search':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        S = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        N = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        for seed in range(60):
            r = stats(seed, n=n, S=S, N=N)
            if r is None:
                continue
            D, ordered, src, snk, counts = r
            print(f'seed={seed:2d}  nodes={len(ordered):2d}  edges={D.number_of_edges():2d}  '
                  f'src={len(src)}  snk={len(snk)}  max_paths={max(counts.values()):3d}  '
                  f'distinct_counts={len(set(counts.values()))}')
    else:
        seed = int(sys.argv[1]) if len(sys.argv) > 1 else 8
        n, S, N = 4, 3, 3
        D, ordered, src, snk, counts = stats(seed, n=n, S=S, N=N)
        draw(D, ordered, src, snk, counts, os.path.join('figures', 'debruijn_dag'))

        # Companion panel: the full De Bruijn graph with the step-2
        # designations (not just the survivors of step 4).
        B = de_bruijn(2, n)
        order = pi_order(B, seed)
        draw_debruijn(B, D, set(order[:S]), set(order[-N:]),
                      os.path.join('figures', 'debruijn_graph'))
        print(f'seed={seed}: {len(ordered)} nodes, {D.number_of_edges()} edges, '
              f'{sum(counts[e] for e in counts if e[0] in src)} total paths')
