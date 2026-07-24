"""Combinatorial-complex (ETNN) lift of the mechanism graph.

The graph in :mod:`src.graph.mech_graph` is a rank-0/rank-1 object: junctions and
struts.  But a compliant mechanism is not only its struts — the *regions* matter:
the closed loops that make a flexure compliant in one direction and stiff in
another, and the solid anchor pads / plates where struts fuse.  A plain GNN can
only reach that information by many message-passing hops around a loop; a
topological network reads it in one, because the loop is an explicit cell.

This module lifts the graph to a **combinatorial complex** (CC) in the sense of
Hajij et al. and builds the tensors an **E(n)-Equivariant Topological Neural
Network** (ETNN, Battiloro et al. 2024, arXiv:2405.15429) consumes.

A CC is a triple ``(S, X, rk)``: ``S`` the ground set (here: node ids), ``X`` a
set of cells (non-empty subsets of ``S``), and a rank function monotone under
inclusion.  Unlike a simplicial or cell complex it imposes *no* topological
requirement on a cell, which is exactly what we need — a mechanism's anchor pad
is a blob, not a disk glued along a boundary sphere.

Ranks used here:

===== ============= ==============================================
rank  cell          built from
===== ============= ==============================================
0     junction      skeleton graph nodes
1     strut         skeleton graph edges (2-node sets)
2     hole / pad    bounded planar faces; fused wide-node blobs
===== ============= ==============================================

**Equivariance.**  Following ETNN, a cell carries *invariant* scalar features
only, and its **position is derived** as the mean of its member rank-0 positions.
So under a rotation R of the node positions every cell position transforms as
``R p`` automatically, and no cell feature changes.  There is nothing to get
wrong at training time.

Everything here is pure numpy — no scikit-image, no torch.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .mech_graph import MechCell, MechGraph, to_arrays

# Invariant rank-2 features (kept stable; append-only if extended).
CELL_SCALARS = ("area", "perimeter", "n_nodes", "is_hole", "is_pad", "circularity")


# --------------------------------------------------------------------------- #
# Planar faces (rank-2 "hole" cells)                                          #
# --------------------------------------------------------------------------- #
def _components(graph: MechGraph) -> dict:
    """Union-find component label per node id."""
    parent = {n.id: n.id for n in graph.nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in graph.edges:
        if e.u in parent and e.v in parent:
            ra, rb = find(e.u), find(e.v)
            if ra != rb:
                parent[ra] = rb
    return {i: find(i) for i in parent}


def _signed_area(pts) -> float:
    """Shoelace signed area of a closed polygon."""
    a = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return 0.5 * a


def _leave_angle(graph: MechGraph, pos: dict, v: int, ei: int) -> float:
    """Angle at which edge ``ei`` physically leaves node ``v``.

    Uses the skeleton polyline's local tangent when available, so curved struts
    and parallel edges (a "theta" shape: two distinct struts joining the same two
    nodes) get *distinct* angles.  Sorting by the straight chord instead would
    tie them and corrupt the face walk.
    """
    e = graph.edges[ei]
    x0, y0 = pos[v]
    pts = e.polyline
    if pts and len(pts) >= 2:
        # polyline runs u -> v; step inward from whichever end is this node
        d_start = (pts[0][1] - x0) ** 2 + (pts[0][0] - y0) ** 2
        d_end = (pts[-1][1] - x0) ** 2 + (pts[-1][0] - y0) ** 2
        k = min(3, len(pts) - 1)
        p = pts[k] if d_start <= d_end else pts[-1 - k]
        dx, dy = p[1] - x0, p[0] - y0
        if dx * dx + dy * dy > 1e-12:
            return math.atan2(dy, dx)
    w = e.v if e.u == v else e.u
    if w in pos:
        return math.atan2(pos[w][1] - y0, pos[w][0] - x0)
    return 0.0


def _planar_faces(graph: MechGraph):
    """All face traversals of the planar embedding.

    Walks directed half-edges, always taking the next neighbour *clockwise* from
    the arrival direction.  Each half-edge belongs to exactly one traversal, so
    this enumerates every face — including, for each connected component, its
    unbounded outer face, which the caller drops.
    """
    pos = {n.id: (n.x, n.y) for n in graph.nodes}
    adj = defaultdict(list)                       # v -> [(neighbour, edge_idx)]
    for idx, e in enumerate(graph.edges):
        if e.u in pos and e.v in pos and e.u != e.v:
            adj[e.u].append((e.v, idx))
            adj[e.v].append((e.u, idx))

    order, slot = {}, {}
    for v, lst in adj.items():
        lst = sorted(lst, key=lambda t: _leave_angle(graph, pos, v, t[1]))
        order[v] = lst
        # index by edge id, not neighbour id: robust to parallel edges
        slot[v] = {ei: i for i, (_, ei) in enumerate(lst)}

    seen = set()                                  # directed half-edges (from, edge)
    faces = []
    for v in order:
        for (w, ei) in order[v]:
            if (v, ei) in seen:
                continue
            cyc_n, cyc_e = [], []
            cu, cv, ce = v, w, ei
            while (cu, ce) not in seen:
                seen.add((cu, ce))
                cyc_n.append(cu)
                cyc_e.append(ce)
                lst = order[cv]
                j = (slot[cv][ce] - 1) % len(lst)   # clockwise-next
                nxt_v, nxt_e = lst[j]
                cu, cv, ce = cv, nxt_v, nxt_e
            if cyc_n:
                faces.append((cyc_n, cyc_e))
    return faces


def _hole_cells(graph: MechGraph, min_area: float, next_id: int):
    """Bounded faces of the planar skeleton -> rank-2 'hole' cells."""
    comp = _components(graph)
    pos = {n.id: (n.x, n.y) for n in graph.nodes}
    faces = _planar_faces(graph)

    # Per connected component the traversal with the largest |area| is the outer
    # face (it encloses the component); a pure tree yields one ~zero-area walk
    # which this also removes.
    by_comp = defaultdict(list)
    for k, (cyc_n, cyc_e) in enumerate(faces):
        by_comp[comp.get(cyc_n[0], -1)].append(k)
    outer = set()
    for _, idxs in by_comp.items():
        outer.add(max(idxs, key=lambda k: abs(_signed_area(
            [pos[i] for i in faces[k][0]]))))

    cells = []
    for k, (cyc_n, cyc_e) in enumerate(faces):
        if k in outer:
            continue
        area = abs(_signed_area([pos[i] for i in cyc_n]))
        if area < min_area:
            continue
        per = 0.0
        for ei in cyc_e:
            e = graph.edges[ei]
            per += e.curve_length if e.curve_length > 0 else e.length
        cells.append(MechCell(id=next_id + len(cells), kind="hole",
                              nodes=list(cyc_n), edges=sorted(set(cyc_e)),
                              area=float(area), perimeter=float(per)))
    return cells


# --------------------------------------------------------------------------- #
# Pads (rank-2 blob cells)                                                    #
# --------------------------------------------------------------------------- #
def _disk_union_area(members) -> float:
    """Exact-enough area of a union of disks, by local rasterization (numpy)."""
    if not members:
        return 0.0
    xs = [x for x, _, _ in members]
    ys = [y for _, y, _ in members]
    rs = [r for _, _, r in members]
    x0, x1 = min(x - r for x, _, r in members), max(x + r for x, _, r in members)
    y0, y1 = min(y - r for _, y, r in members), max(y + r for _, y, r in members)
    w = max(1, int(math.ceil(x1 - x0)) + 1)
    h = max(1, int(math.ceil(y1 - y0)) + 1)
    if w * h > 4_000_000:                          # pathological guard
        return float(sum(math.pi * r * r for r in rs))
    gx = np.arange(w, dtype=np.float32)[None, :] + x0
    gy = np.arange(h, dtype=np.float32)[:, None] + y0
    cover = np.zeros((h, w), bool)
    for x, y, r in members:
        cover |= ((gx - x) ** 2 + (gy - y) ** 2) <= r * r
    return float(cover.sum())


def _pad_cells(graph: MechGraph, pad_factor: float, min_r: float, next_id: int):
    """Fused wide-node blobs -> rank-2 'pad' cells.

    A node is pad-like when its medial-axis radius is much larger than the
    typical strut half-width (so it is a *region*, not a thick strut).  Pad-like
    nodes whose disks overlap are merged, and each group is extended by its
    directly attached neighbours: those are the struts entering the pad, so the
    cell spans the whole 2D region rather than a bare point.
    """
    if not graph.nodes:
        return []
    widths = [e.w for e in graph.edges if e.w > 0]
    typ = float(np.median(widths)) / 2.0 if widths else 0.0
    thr = max(min_r, pad_factor * typ)

    cand = [n for n in graph.nodes if n.r >= thr]
    if not cand:
        return []
    cand_ids = {n.id for n in cand}
    nb = graph.node_by_id()

    parent = {i: i for i in cand_ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    adj = defaultdict(list)
    for e in graph.edges:
        if e.u in nb and e.v in nb:
            adj[e.u].append(e.v)
            adj[e.v].append(e.u)
            if e.u in cand_ids and e.v in cand_ids:
                a, b = nb[e.u], nb[e.v]
                if math.hypot(a.x - b.x, a.y - b.y) <= a.r + b.r:   # disks overlap
                    ra, rb = find(e.u), find(e.v)
                    if ra != rb:
                        parent[ra] = rb

    groups = defaultdict(list)
    for i in cand_ids:
        groups[find(i)].append(i)

    cells = []
    for _, core in groups.items():
        members = set(core)
        for i in core:                             # attach the entering struts
            members.update(adj.get(i, []))
        if len(members) < 2:                       # a rank-2 cell must not
            continue                               # duplicate a rank-0 cell
        eidx = [k for k, e in enumerate(graph.edges)
                if e.u in members and e.v in members]
        area = _disk_union_area([(nb[i].x, nb[i].y, max(1.0, nb[i].r))
                                 for i in core])
        per = 2.0 * math.pi * math.sqrt(max(area, 1e-9) / math.pi)
        cells.append(MechCell(id=next_id + len(cells), kind="pad",
                              nodes=sorted(members), edges=sorted(eidx),
                              area=float(area), perimeter=float(per)))
    return cells


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def build_cells(graph: MechGraph, min_area: float = 12.0, pad_factor: float = 2.0,
                min_pad_r: float = 3.0, holes: bool = True, pads: bool = True):
    """Lift ``graph`` to a combinatorial complex, filling ``graph.faces``.

    Returns the list of rank-2 cells (also stored on the graph).  Idempotent:
    calling it again rebuilds the cells from scratch.
    """
    cells = []
    if holes:
        cells += _hole_cells(graph, min_area=min_area, next_id=0)
    if pads:
        cells += _pad_cells(graph, pad_factor=pad_factor, min_r=min_pad_r,
                            next_id=len(cells))
    graph.faces = cells
    return cells


def euler_check(graph: MechGraph) -> dict:
    """Validate the rank-2 hole cells against Euler's formula.

    For a planar graph with ``C`` connected components, ``V - E + F = 1 + C``
    where ``F`` counts the single unbounded face, so the number of *bounded*
    faces must be exactly ``E - V + C``.  Any mismatch means the face traversal
    or the outer-face removal is wrong — this is the correctness test for the
    rank-2 lift, independent of any threshold.
    """
    v = graph.n_nodes()
    pos = {n.id for n in graph.nodes}
    uniq = {(min(e.u, e.v), max(e.u, e.v))
            for e in graph.edges if e.u in pos and e.v in pos and e.u != e.v}
    e = len(uniq)
    c = len(set(_components(graph).values())) if graph.nodes else 0
    expected = e - v + c
    got = sum(1 for f in graph.faces if getattr(f, "kind", None) == "hole")
    return {"V": v, "E": e, "C": c, "expected_bounded_faces": expected,
            "hole_cells": got, "ok": got == expected}


def to_cc_arrays(graph: MechGraph, undirected: bool = True) -> dict:
    """Model-ready combinatorial-complex tensors for an ETNN.

    Per-rank features and *derived* positions, plus the incidence structures the
    network message-passes over.

    ==============  ==========  ============================================
    key             shape       contents
    ==============  ==========  ============================================
    ``pos_0``       (N, 2)      equivariant node positions (normalized)
    ``x_0``         (N, 6)      invariant node features (NODE_SCALARS)
    ``vec_0``       (N, 2)      equivariant load/motion directions
    ``pos_1``       (E, 2)      equivariant strut midpoints (derived)
    ``x_1``         (E, 3)      invariant strut features (EDGE_SCALARS)
    ``pos_2``       (F, 2)      equivariant cell centroids (derived)
    ``x_2``         (F, 6)      invariant cell features (CELL_SCALARS)
    ``inc_01``      (2, n01)    node -> strut incidence  [rank0_row; rank1_row]
    ``inc_12``      (2, n12)    strut -> cell incidence
    ``inc_02``      (2, n02)    node -> cell incidence
    ``adj_00``      (2, 2E)     node-node adjacency (the plain graph)
    ``adj_22``      (2, n22)    cells sharing at least one strut
    ``graph_y``     (G,)        graph-level goal/label scalars
    ==============  ==========  ============================================

    Positions for ranks 1 and 2 are means of member rank-0 positions, so they are
    equivariant by construction and never need to be stored or predicted.
    """
    base = to_arrays(graph, undirected=undirected)
    pos0 = base["pos"]
    scale = float(max(graph.shape)) if graph.shape else 1.0
    id2row = {nd.id: i for i, nd in enumerate(graph.nodes)}

    # ---- rank 1 ----
    e_rows, pos1 = [], []
    for k, e in enumerate(graph.edges):
        if e.u in id2row and e.v in id2row:
            e_rows.append(k)
            pos1.append(0.5 * (pos0[id2row[e.u]] + pos0[id2row[e.v]]))
    edge_row = {k: i for i, k in enumerate(e_rows)}
    pos1 = (np.asarray(pos1, np.float32) if pos1
            else np.zeros((0, 2), np.float32))
    # one row per undirected strut (base["edge_scalar"] is duplicated if undirected)
    x1 = np.asarray([[graph.edges[k].w / scale,
                      graph.edges[k].length / scale,
                      max(0.0, (graph.edges[k].curve_length
                                / max(graph.edges[k].length, 1e-6)) - 1.0)]
                     for k in e_rows], np.float32).reshape(len(e_rows), 3)

    inc01 = [(id2row[graph.edges[k].u], edge_row[k]) for k in e_rows] + \
            [(id2row[graph.edges[k].v], edge_row[k]) for k in e_rows]

    # ---- rank 2 ----
    cells = [c for c in graph.faces if isinstance(c, MechCell)]
    area_norm = scale * scale
    pos2, x2, inc12, inc02 = [], [], [], []
    for ci, c in enumerate(cells):
        rows = [id2row[i] for i in c.nodes if i in id2row]
        pos2.append(pos0[rows].mean(0) if rows else np.zeros(2, np.float32))
        circ = (4.0 * math.pi * c.area / (c.perimeter ** 2)
                if c.perimeter > 1e-6 else 0.0)
        x2.append([c.area / area_norm, c.perimeter / scale, len(c.nodes) / 8.0,
                   float(c.kind == "hole"), float(c.kind == "pad"),
                   min(1.0, circ)])
        for r in rows:
            inc02.append((r, ci))
        for k in c.edges:
            if k in edge_row:
                inc12.append((edge_row[k], ci))
    pos2 = (np.asarray(pos2, np.float32) if pos2 else np.zeros((0, 2), np.float32))
    x2 = (np.asarray(x2, np.float32) if x2
          else np.zeros((0, len(CELL_SCALARS)), np.float32))

    # cells that share a strut
    by_edge = defaultdict(list)
    for er, ci in inc12:
        by_edge[er].append(ci)
    adj22 = set()
    for _, cs in by_edge.items():
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                adj22.add((cs[i], cs[j]))
                adj22.add((cs[j], cs[i]))

    def _pairs(lst):
        return (np.asarray(sorted(lst), np.int64).T if lst
                else np.zeros((2, 0), np.int64))

    return {"pos_0": pos0, "x_0": base["node_scalar"], "vec_0": base["node_vec"],
            "pos_1": pos1, "x_1": x1,
            "pos_2": pos2, "x_2": x2,
            "inc_01": _pairs(inc01), "inc_12": _pairs(inc12),
            "inc_02": _pairs(inc02),
            "adj_00": base["edge_index"], "adj_22": _pairs(list(adj22)),
            "graph_y": base["graph_y"],
            "id_index": base["id_index"]}
