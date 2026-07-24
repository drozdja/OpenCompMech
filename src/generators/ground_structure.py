"""Family D: ground-structure sampling — random flexure trusses.

Families E (rigid-body replacement) and B (FACT) construct TEXTBOOK
archetypes; their diversity is parametric. Family D samples the TOPOLOGY
itself: scatter nodes over the domain, take the Delaunay graph as the ground
structure, keep a random spanning tree plus a few loop-closing edges, and
render every kept member as a thick link with thin flexure necks at its ends.

Key structural rule: average node degree must stay ~2-3. A fully triangulated
truss carries load AXIALLY through the necks (axial stiffness ~w, not w^3),
so it is quasi-rigid and GA collapses; a tree plus 0-2 loops leaves bending
degrees of freedom = mechanism motion. Validity is decided downstream by the
standard gates (connectivity, hinge detector, |u_out| floor, GA >= 0.25 for
constructed samples) — the sampler only guarantees a well-formed graph and an
FEA-labeled output sense. No optimizer; ~1 s/sample like Families E/B.

v1 (production TO variant): optimize member areas for the mechanism
objective on the full ground structure. Deferred — sampling alone already
buys topology diversity none of the other families have.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .mech import MechProblem
from .seeds import (
    _create_mech_problem_from_linkage,
    _draw_thick_line,
    _draw_circle,
)
from .flexure_utils import _debridge, _unit, _draw_flexure_link


def _sample_nodes(nelx, nely, rng, n_internal, min_sep):
    """Anchors + internal nodes with rejection-sampled minimum separation."""
    pts = []
    for _ in range(400):
        if len(pts) >= n_internal:
            break
        p = np.array([rng.uniform(0.08, 0.92) * nelx,
                      rng.uniform(0.08, 0.92) * nely])
        if all(np.linalg.norm(p - q) >= min_sep for q in pts):
            pts.append(p)
    return pts


def construct_gs_truss(
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    k_in: float = None,
    k_out: float = None,
    max_attempts: int = 200,
) -> Optional[Tuple[np.ndarray, MechProblem, Dict]]:
    """Sample a random flexure truss from a Delaunay ground structure.

    Nodes: 2 anchors + input port + output port + 2-5 internal. Edges: random
    spanning tree of the Delaunay graph + 0-2 loop closers. Members: thick
    beam with flexure necks at both ends. Output direction labeled by FEA.
    """
    if k_in is None:
        k_in = 0.01
    if k_out is None:
        k_out = float(rng.uniform(0.01, 0.05))

    from scipy.spatial import Delaunay

    min_dim = min(nelx, nely)
    for _ in range(max_attempts):
        n_internal = int(rng.randint(2, 5))
        min_sep = 0.22 * min_dim
        nodes = _sample_nodes(nelx, nely, rng, 4 + n_internal, min_sep)
        if len(nodes) < 4 + n_internal:
            continue
        pts = np.array(nodes)
        n = len(pts)

        # anchors: random pair; ports are chosen AFTER the graph is built so
        # they can be restricted to degree>=2 nodes (see below).
        idx = rng.permutation(n)
        a1, a2 = idx[0], idx[1]

        # Delaunay edges
        try:
            tri = Delaunay(pts)
        except Exception:
            continue
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                e = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
                edges.add(e)
        edges = list(edges)

        # random spanning tree (randomized Kruskal) + 0-2 loop closers
        rng.shuffle(edges)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        tree, spare = [], []
        for (i, j) in edges:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                tree.append((i, j))
            else:
                spare.append((i, j))
        if len(tree) != n - 1:
            continue
        # >=1 loop: pure trees are floppy chains that dissipate the input
        # stroke before it reaches the output (v0 smoke: u_in 40-96, u_out ~3)
        n_loops = int(rng.randint(1, 3))
        keep = tree + spare[:n_loops]
        if len(keep) < n:                        # not enough spare edges
            continue

        # ports: degree>=2 nodes only, farthest-apart pair. A LEAF input just
        # bends its own neck locally (v0 smoke test: u_in ~90 = free spring
        # stroke, u_out ~0 — 8/12 failures were leaf ports); an interior node
        # transmits motion into the rest of the graph.
        deg = np.zeros(n, dtype=int)
        for (i, j) in keep:
            deg[i] += 1
            deg[j] += 1
        cand = [i for i in range(n) if i not in (a1, a2) and deg[i] >= 2]
        if len(cand) < 2:
            continue
        best, i_in, i_out = -1.0, None, None
        for ii in cand:
            for jj in cand:
                if jj <= ii:
                    continue
                d = np.linalg.norm(pts[ii] - pts[jj])
                if d > best:
                    best, i_in, i_out = d, ii, jj
        if best < 0.3 * min_dim:
            continue
        if rng.random() < 0.5:
            i_in, i_out = i_out, i_in

        link_hw = max(3, int(round(min_dim / rng.uniform(26.0, 36.0))))
        neck_hw = 1
        neck_len = float(min_dim) / rng.uniform(16.0, 24.0)
        pad_r = link_hw + 2

        density = np.zeros((nely, nelx), dtype=np.float64)
        for gp in (pts[a1], pts[a2]):
            _draw_circle(density, gp, radius=pad_r, value=1.0)
        for (i, j) in keep:
            _draw_flexure_link(density, pts[i], pts[j],
                               link_hw=link_hw, neck_hw=neck_hw,
                               neck_len=neck_len)
        for i in range(n):
            if i not in (a1, a2):
                _draw_circle(density, pts[i], radius=neck_hw + 1, value=1.0)
        _draw_circle(density, pts[i_in], radius=3, value=1.0)
        _draw_circle(density, pts[i_out], radius=3, value=1.0)
        if not _debridge(density):
            continue

        # input direction: perpendicular to the stiffest incident member
        # (bending actuation moves the structure; axial pushing just stretches
        # a neck). Use the longest incident edge as reference.
        inc = [e for e in keep if i_in in e]
        if not inc:
            continue
        i, j = max(inc, key=lambda e: np.linalg.norm(pts[e[0]] - pts[e[1]]))
        other = pts[j] if i == i_in else pts[i]
        axis = _unit(other - pts[i_in])
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        input_direction = (float(-sgn * axis[1]), float(sgn * axis[0]))
        # provisional output direction: random unit; FEA probe fixes it below
        phi = rng.uniform(0, 2 * np.pi)
        output_direction = (float(np.cos(phi)), float(np.sin(phi)))

        vf = float(density.mean())

        def _mk(out_dir):
            return _create_mech_problem_from_linkage(
                ground_pivots=[pts[a1], pts[a2]],
                input_joint=pts[i_in],
                output_joint=pts[i_out],
                input_direction=input_direction,
                output_direction=tuple(out_dir),
                nelx=nelx, nely=nely,
                volume_fraction=vf,
                k_in=k_in, k_out=k_out,
                bc_radius=max(2, pad_r - 1),
            )

        # FEA probe: find the ACTUAL motion direction of the output node under
        # the input load, then label the output along it. Unlike E/B there is
        # no kinematic prediction here — the probe IS the labeling step.
        from .mech import solve_mechanism_fea
        problem = _mk(output_direction)
        u, _, _ = solve_mechanism_fea(problem, density)
        no = problem.output_node
        move = np.array([u[2 * no], u[2 * no + 1]])
        if np.linalg.norm(move) < 1e-9:
            continue
        output_direction = tuple(_unit(move).tolist())
        problem = _mk(output_direction)

        meta = {
            'family': 'D',
            'generator': 'gs_truss',
            'n_nodes': int(n),
            'n_members': len(keep),
            'n_loops': int(n_loops),
            'link_hw': link_hw, 'neck_hw': neck_hw,
            'neck_len': float(neck_len),
            'constructed_vf': vf,
        }
        return density, problem, meta

    return None


# ---------------------------------------------------------------------------
# Family D v1: member-width OPTIMIZATION on the ground structure (gs_opt)
# ---------------------------------------------------------------------------
# The proper ground-structure method (Frecker et al. 1997 lineage): keep the
# FULL candidate edge set and optimize each member's WIDTH on a fast frame
# model. Width is the physical variable — a thin member IS a flexure
# (bending ~ w^3), a thick one IS a rigid link — so intermediate values are
# meaningful and no SIMP-style penalization is needed. The optimized frame is
# pruned, rasterized, and relabeled with the continuum FEA (ground truth).
# The graph (nodes, edges, widths) is stored per sample: Family D's native,
# GNN-ready representation.

def _frame_K_elems(nodes, edges, widths):
    """Element stiffness matrices + width-derivatives for 2D frames.

    Returns (K_es, dK_es): lists of (6,6) arrays in GLOBAL coords for each
    edge. E = 1, rectangular section: A = w, I = w^3/12.
    """
    K_es, dK_es = [], []
    for (i, j), w in zip(edges, widths):
        p, q = nodes[i], nodes[j]
        L = float(np.linalg.norm(q - p)) + 1e-12
        c, s = (q - p) / L
        A, dA = w, 1.0
        I, dI = w ** 3 / 12.0, w ** 2 / 4.0

        def k_local(A_, I_):
            a = A_ / L
            b = 12.0 * I_ / L ** 3
            cql = 6.0 * I_ / L ** 2
            d4 = 4.0 * I_ / L
            d2 = 2.0 * I_ / L
            return np.array([
                [ a,    0,    0,  -a,    0,    0],
                [ 0,    b,  cql,   0,   -b,  cql],
                [ 0,  cql,   d4,   0, -cql,   d2],
                [-a,    0,    0,   a,    0,    0],
                [ 0,   -b, -cql,   0,    b, -cql],
                [ 0,  cql,   d2,   0, -cql,   d4]])

        T = np.array([
            [ c, s, 0,  0, 0, 0],
            [-s, c, 0,  0, 0, 0],
            [ 0, 0, 1,  0, 0, 0],
            [ 0, 0, 0,  c, s, 0],
            [ 0, 0, 0, -s, c, 0],
            [ 0, 0, 0,  0, 0, 1]])
        K_es.append(T.T @ k_local(A, I) @ T)
        dK_es.append(T.T @ k_local(dA, dI) @ T)
    return K_es, dK_es


def _frame_assemble(n_nodes, edges, K_es, extra_diag):
    K = np.zeros((3 * n_nodes, 3 * n_nodes))
    for (i, j), K_e in zip(edges, K_es):
        dofs = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        K[np.ix_(dofs, dofs)] += K_e
    K[np.diag_indices_from(K)] += extra_diag
    return K


def _optimize_frame(nodes, edges, i_in, i_out, anchors, d_in, d_out,
                    k_in, k_out, rng, n_iters=200, alpha=0.35,
                    w_min=0.6, w_max=12.0):
    """Adam ascent on member widths for J = u_out - alpha * SE_in.

    Widths in PIXEL units. Returns (widths, u_out_history).
    """
    n = len(nodes)
    widths = rng.uniform(2.0, 5.0, size=len(edges))
    fixed = []
    for a in anchors:
        fixed += [3 * a, 3 * a + 1, 3 * a + 2]
    free = np.setdiff1d(np.arange(3 * n), fixed)

    # springs on port translational DOFs (k * d d^T)
    extra = np.zeros(3 * n)
    spring_blocks = []
    for idx, d, k in ((i_in, d_in, k_in), (i_out, d_out, k_out)):
        dofs = [3 * idx, 3 * idx + 1]
        blk = k * np.outer(d, d)
        spring_blocks.append((dofs, blk))

    F = np.zeros(3 * n)
    F[3 * i_in] = d_in[0]
    F[3 * i_in + 1] = d_in[1]
    L_vec = np.zeros(3 * n)
    L_vec[3 * i_out] = d_out[0]
    L_vec[3 * i_out + 1] = d_out[1]

    m = np.zeros_like(widths)
    v = np.zeros_like(widths)
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.25
    hist = []
    for it in range(n_iters):
        K_es, dK_es = _frame_K_elems(nodes, edges, widths)
        K = _frame_assemble(n, edges, K_es, extra)
        for dofs, blk in spring_blocks:
            K[np.ix_(dofs, dofs)] += blk
        Kf = K[np.ix_(free, free)]
        try:
            u_f = np.linalg.solve(Kf, F[free])
            lam_f = np.linalg.solve(Kf, L_vec[free])
        except np.linalg.LinAlgError:
            return None, []
        u = np.zeros(3 * n); u[free] = u_f
        lam = np.zeros(3 * n); lam[free] = lam_f
        u_out = float(L_vec @ u)
        se = float(F @ u)
        hist.append(u_out)

        grad = np.empty_like(widths)
        for e, ((i, j), dK_e) in enumerate(zip(edges, dK_es)):
            dofs = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
            u_e = u[dofs]; lam_e = lam[dofs]
            grad[e] = -lam_e @ dK_e @ u_e + alpha * (u_e @ dK_e @ u_e)

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        widths = widths + lr * (m / (1 - b1 ** (it + 1))) / (
            np.sqrt(v / (1 - b2 ** (it + 1))) + eps)
        widths = np.clip(widths, w_min, w_max)
    return widths, hist


def construct_gs_opt(
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    k_in: float = None,
    k_out: float = None,
    max_attempts: int = 60,
) -> Optional[Tuple[np.ndarray, MechProblem, Dict]]:
    """Family D v1: optimize member widths on the full Delaunay ground
    structure (frame surrogate), prune, rasterize, continuum-relabel."""
    if k_in is None:
        k_in = 0.01
    if k_out is None:
        k_out = float(rng.uniform(0.005, 0.02))

    from scipy.spatial import Delaunay
    min_dim = min(nelx, nely)
    for _ in range(max_attempts):
        n_internal = int(rng.randint(3, 7))
        pts_l = _sample_nodes(nelx, nely, rng, 4 + n_internal, 0.20 * min_dim)
        if len(pts_l) < 4 + n_internal:
            continue
        pts = np.array(pts_l)
        n = len(pts)
        idx = rng.permutation(n)
        a1, a2 = int(idx[0]), int(idx[1])
        rest = [i for i in range(n) if i not in (a1, a2)]
        rest.sort(key=lambda i: -np.linalg.norm(
            pts[i] - 0.5 * (pts[a1] + pts[a2])))
        i_in, i_out = rest[0], rest[1]
        if np.linalg.norm(pts[i_in] - pts[i_out]) < 0.3 * min_dim:
            continue
        try:
            tri = Delaunay(pts)
        except Exception:
            continue
        edges = sorted({tuple(sorted((s[i], s[(i + 1) % 3])))
                        for s in tri.simplices for i in range(3)})

        phi_i = rng.uniform(0, 2 * np.pi)
        d_in = np.array([np.cos(phi_i), np.sin(phi_i)])
        phi_o = rng.uniform(0, 2 * np.pi)
        d_out = np.array([np.cos(phi_o), np.sin(phi_o)])

        widths, hist = _optimize_frame(pts, edges, i_in, i_out, (a1, a2),
                                       d_in, d_out, k_in, k_out, rng)
        if widths is None or not hist or abs(hist[-1]) < 0.5:
            continue

        # prune near-void members; require port/anchor connectivity
        keep = [(e, w) for e, w in zip(edges, widths) if w > 1.2]
        if not keep:
            continue
        adj = {}
        for (i, j), _w in keep:
            adj.setdefault(i, set()).add(j)
            adj.setdefault(j, set()).add(i)
        seen = set()
        stack = [i_in]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj.get(x, ()))
        if not ({i_out, a1, a2} & seen >= {i_out}) or i_out not in seen \
                or not ({a1, a2} & seen):
            continue

        # rasterize kept members at their optimized widths
        density = np.zeros((nely, nelx), dtype=np.float64)
        pad_r = 7
        for (i, j), w in keep:
            _draw_thick_line(density, pts[i], pts[j],
                             width=max(1, int(round(w / 2.0))), value=1.0)
        for gp in (pts[a1], pts[a2]):
            _draw_circle(density, gp, radius=pad_r, value=1.0)
        for i in (i_in, i_out):
            _draw_circle(density, pts[i], radius=3, value=1.0)
        if not _debridge(density):
            continue

        vf = float(density.mean())

        def _mk(out_dir):
            return _create_mech_problem_from_linkage(
                ground_pivots=[pts[a1], pts[a2]],
                input_joint=pts[i_in], output_joint=pts[i_out],
                input_direction=tuple(d_in), output_direction=tuple(out_dir),
                nelx=nelx, nely=nely, volume_fraction=vf,
                k_in=k_in, k_out=k_out, bc_radius=max(2, pad_r - 1))

        problem = _mk(d_out)
        from .mech import solve_mechanism_fea
        _, _, u_probe = solve_mechanism_fea(problem, density)
        out_dir = tuple(d_out)
        if u_probe < 0:
            out_dir = (-d_out[0], -d_out[1])
            problem = _mk(out_dir)

        meta = {
            'family': 'D',
            'generator': 'gs_opt',
            'n_nodes': int(n),
            'n_candidate_members': len(edges),
            'n_members_kept': len(keep),
            'graph_nodes': [[float(x), float(y)] for x, y in pts],
            'graph_edges': [[int(i), int(j), float(w)] for (i, j), w in keep],
            'u_out_frame_final': float(hist[-1]),
            'opt_iters': len(hist),
            'constructed_vf': vf,
        }
        return density, problem, meta

    return None


GS_CONSTRUCTORS = {
    'gs_truss': construct_gs_truss,
    'gs_opt': construct_gs_opt,
}
