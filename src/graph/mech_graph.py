"""Canonical graph representation for 2D compliant mechanisms.

Generator-independent by design: the graph is derived from the *finished shape*
(the density that carries the physics and passes FEA), not from any
family-specific generator recipe.  One code path handles every family (SIMP
continuum, ground-structure truss, rigid-body linkage, morphable components,
flexure): threshold -> medial-axis skeleton -> typed geometric graph.

Representation choices (SOTA-informed, mid-2026):

* **EGNN / E(n)-equivariant ready.**  Node *positions* are kept as an
  equivariant channel; every other node/edge feature is a rotation/translation
  *invariant* scalar, except explicit *direction vectors* (applied load, output
  motion) which are equivariant.  A downstream EGNN builds messages from
  pairwise distances, so positions + invariant features are exactly what it
  needs (see Satorras et al. 2021, arXiv:2102.09844).
* **SE(2), not E(2).**  Reflection is deliberately *not* treated as a symmetry:
  a mirror-image mechanism inverts its output direction (chirality is physical).
* **Verification-first.**  ``to_raster`` round-trips the graph back to a density
  so any graph (including a *generated* one) can be re-checked by the FEA gate.
* **Wide-node pads (v1).**  2D anchor pads / plates are represented as
  high-radius nodes.  A ``faces`` field is reserved for a future combinatorial-
  complex (2-cell / ETNN-style, arXiv:2405.15429) extension and is empty here.

Scale convention: the raw schema stores geometry in *pixels* (faithful, so
reconstruction is exact).  ``to_arrays`` emits model-ready tensors normalized by
the domain size, matching the project's normalized-units ethos.

Only ``from_raster`` needs scikit-image + sknw + scipy; the schema, features,
serialization and PyG export work without them (e.g. to reload saved graphs).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Boundary-condition roles a node can carry (a node may have several, e.g. a
# joint that is also the input port).
ROLE_INPUT = "input"
ROLE_OUTPUT = "output"
ROLE_SUPPORT = "support"
ROLES = (ROLE_INPUT, ROLE_OUTPUT, ROLE_SUPPORT)


@dataclass
class MechNode:
    """A joint / junction / endpoint / port of the mechanism.

    x, y, r are in pixels.  ``roles`` is a subset of ROLES.  ``vec`` is an
    equivariant 2-vector: the applied-load direction for an input node, or the
    desired output-motion direction for an output node (None otherwise).
    """
    id: int
    x: float
    y: float
    r: float = 0.0
    degree: int = 0
    roles: list = field(default_factory=list)
    vec: Optional[list] = None
    fixed: bool = False


@dataclass
class MechEdge:
    """A structural member connecting two nodes.

    ``length`` is the invariant Euclidean node-to-node distance; ``curve_length``
    is the skeleton arc length (>= length, a curvature proxy).  ``polyline`` /
    ``radii`` are the faithful skeleton geometry used only for reconstruction and
    may be dropped for a lean export.
    """
    u: int
    v: int
    w: float = 0.0
    length: float = 0.0
    curve_length: float = 0.0
    polyline: Optional[list] = None
    radii: Optional[list] = None


@dataclass
class MechCell:
    """A rank-2 cell: a 2D region of the mechanism (ETNN / combinatorial complex).

    A combinatorial complex only requires a cell to be a non-empty set of rank-0
    cells with a monotone rank function, so a 2-cell is free to be *any* node set
    — it does not have to be a topological disk.  Two kinds are built:

    ``hole`` : a bounded face of the planar skeleton graph (a closed loop of
               struts).  ``nodes``/``edges`` are its boundary cycle, in order.
    ``pad``  : a merged 2D blob (anchor pad, plate, thick junction) where struts
               fuse into a region that is not strut-like.

    ``area``/``perimeter`` are in pixels and rotation-invariant.  A cell has no
    stored position: its position is *derived* from its member nodes so that it
    is equivariant by construction (see :func:`src.graph.complex.to_cc_arrays`).
    """
    id: int
    kind: str                                     # "hole" | "pad"
    nodes: list = field(default_factory=list)     # member rank-0 cell ids
    edges: list = field(default_factory=list)     # member rank-1 cell indices
    area: float = 0.0
    perimeter: float = 0.0
    rank: int = 2


@dataclass
class MechGraph:
    """A whole mechanism as a typed geometric graph + its BVP and physics label."""
    nodes: list
    edges: list
    shape: list                                   # [H, W] of the source raster
    globals: dict = field(default_factory=dict)   # goal-conditioning scalars
    label: dict = field(default_factory=dict)     # verified FEA response
    meta: dict = field(default_factory=dict)      # family, type, stem, hashes
    faces: list = field(default_factory=list)     # rank-2 cells (MechCell)

    # ---- basic queries ----
    def n_nodes(self) -> int:
        return len(self.nodes)

    def n_edges(self) -> int:
        return len(self.edges)

    def node_by_id(self) -> dict:
        return {n.id: n for n in self.nodes}

    # ---- serialization ----
    def to_dict(self) -> dict:
        def nd(n):
            return {"id": n.id, "x": n.x, "y": n.y, "r": n.r, "degree": n.degree,
                    "roles": list(n.roles), "vec": n.vec, "fixed": n.fixed}

        def ed(e):
            return {"u": e.u, "v": e.v, "w": e.w, "length": e.length,
                    "curve_length": e.curve_length, "polyline": e.polyline,
                    "radii": e.radii}

        def cd(c):
            return {"id": c.id, "kind": c.kind, "nodes": list(c.nodes),
                    "edges": list(c.edges), "area": c.area,
                    "perimeter": c.perimeter, "rank": c.rank}

        return {"format": "opencompmech.mech-graph.v2",
                "shape": list(self.shape),
                "nodes": [nd(n) for n in self.nodes],
                "edges": [ed(e) for e in self.edges],
                "globals": self.globals, "label": self.label,
                "meta": self.meta,
                "faces": [cd(c) if isinstance(c, MechCell) else c
                          for c in self.faces]}

    @staticmethod
    def from_dict(d: dict) -> "MechGraph":
        nodes = [MechNode(**n) for n in d["nodes"]]
        edges = [MechEdge(**e) for e in d["edges"]]
        # v1 wrote `faces: []` (reserved); v2 writes MechCell dicts.
        faces = [MechCell(**c) if isinstance(c, dict) else c
                 for c in d.get("faces", [])]
        return MechGraph(nodes=nodes, edges=edges, shape=list(d["shape"]),
                         globals=d.get("globals", {}), label=d.get("label", {}),
                         meta=d.get("meta", {}), faces=faces)

    def save(self, path: str, drop_geometry: bool = False) -> None:
        d = self.to_dict()
        if drop_geometry:                          # lean export for a GNN
            for e in d["edges"]:
                e["polyline"] = None
                e["radii"] = None
        with open(path, "w") as f:
            json.dump(d, f)

    @staticmethod
    def load(path: str) -> "MechGraph":
        with open(path) as f:
            return MechGraph.from_dict(json.load(f))


# --------------------------------------------------------------------------- #
# Model-ready tensors (EGNN-style: equivariant positions + invariant features) #
# --------------------------------------------------------------------------- #
# Documented feature layout (kept stable; append-only if extended).
NODE_SCALARS = ("r", "degree", "is_input", "is_output", "is_support", "vec_mag")
EDGE_SCALARS = ("w", "length", "curvature")


def to_arrays(graph: MechGraph, undirected: bool = True) -> dict:
    """Return normalized, model-ready arrays for a GNN.

    Keys:
      pos          (N,2) float   equivariant node positions (normalized to ~[0,1])
      node_scalar  (N,6) float   invariant node features  (see NODE_SCALARS)
      node_vec     (N,2) float   equivariant per-node direction (load/motion; 0 if none)
      edge_index   (2,E) int     (both directions if ``undirected``)
      edge_scalar  (E,3) float   invariant edge features  (see EDGE_SCALARS)
      graph_y      (G,)  float   graph-level goal/label scalars (from globals)
      id_index     (N,)  int     node ids in row order
    """
    n = graph.n_nodes()
    scale = float(max(graph.shape)) if graph.shape else 1.0
    id2row = {nd.id: i for i, nd in enumerate(graph.nodes)}

    pos = np.zeros((n, 2), np.float32)
    node_scalar = np.zeros((n, len(NODE_SCALARS)), np.float32)
    node_vec = np.zeros((n, 2), np.float32)
    for i, nd in enumerate(graph.nodes):
        pos[i] = (nd.x / scale, nd.y / scale)
        v = nd.vec if nd.vec is not None else (0.0, 0.0)
        vmag = math.hypot(v[0], v[1])
        node_scalar[i] = (nd.r / scale, nd.degree / 8.0,
                          float(ROLE_INPUT in nd.roles),
                          float(ROLE_OUTPUT in nd.roles),
                          float(ROLE_SUPPORT in nd.roles or nd.fixed),
                          vmag)
        if vmag > 1e-9:
            node_vec[i] = (v[0] / vmag, v[1] / vmag)

    ei, ea = [], []
    for e in graph.edges:
        if e.u not in id2row or e.v not in id2row:
            continue
        length = e.length / scale
        curv = (e.curve_length / max(e.length, 1e-6)) - 1.0
        feat = (e.w / scale, length, max(0.0, curv))
        ei.append((id2row[e.u], id2row[e.v]))
        ea.append(feat)
        if undirected:
            ei.append((id2row[e.v], id2row[e.u]))
            ea.append(feat)
    edge_index = (np.asarray(ei, np.int64).T if ei
                  else np.zeros((2, 0), np.int64))
    edge_scalar = (np.asarray(ea, np.float32) if ea
                   else np.zeros((0, len(EDGE_SCALARS)), np.float32))

    gy = graph.globals.get("scalar_vector")
    graph_y = (np.asarray(gy, np.float32) if gy is not None
               else np.zeros((0,), np.float32))

    return {"pos": pos, "node_scalar": node_scalar, "node_vec": node_vec,
            "edge_index": edge_index, "edge_scalar": edge_scalar,
            "graph_y": graph_y,
            "id_index": np.asarray([nd.id for nd in graph.nodes], np.int64)}


# --------------------------------------------------------------------------- #
# Raster  <->  graph                                                          #
# --------------------------------------------------------------------------- #
def _clean(binary, min_feature):
    import warnings
    from skimage.morphology import remove_small_holes, remove_small_objects
    with warnings.catch_warnings():          # skimage 0.26 param-rename FutureWarnings
        warnings.simplefilter("ignore", FutureWarning)
        return remove_small_objects(remove_small_holes(binary, min_feature),
                                    min_feature)


def _prune_spurs(g, min_len):
    for _ in range(3):
        drop = [(u, v) for u, v in list(g.edges())
                if (g.degree(u) == 1 or g.degree(v) == 1)
                and g[u][v].get("weight", len(g[u][v]["pts"])) < min_len]
        if not drop:
            break
        g.remove_edges_from(drop)
        g.remove_nodes_from([x for x in list(g.nodes()) if g.degree(x) == 0])
    return g


def _centroid_vec(mask, dir_x, dir_y):
    ys, xs = np.where(np.asarray(mask) > 0.5)
    if len(ys) == 0:
        return None, None
    c = (float(ys.mean()), float(xs.mean()))
    vec = None
    if dir_x is not None and dir_y is not None:
        vx = float(np.asarray(dir_x)[ys, xs].mean())
        vy = float(np.asarray(dir_y)[ys, xs].mean())
        if math.hypot(vx, vy) > 1e-6:
            vec = [vx, vy]
    return c, vec


def _nearest(nodes, cy, cx):
    return min(range(len(nodes)),
               key=lambda i: (nodes[i].y - cy) ** 2 + (nodes[i].x - cx) ** 2)


def _split_self_loops(graph: "MechGraph", dist=None):
    """Split every self-loop into a simple 3-cycle.

    A closed ring (the archetypal compliant flexure loop) has no junction, so the
    skeleton graph represents it as *one* node carrying a self-loop.  That is a
    poor representation twice over: the enclosed region is invisible to the
    rank-2 lift, and a GNN sees a single position instead of the ring's shape.

    Cutting the loop at 1/3 and 2/3 arc length yields a simple cycle — three
    distinct nodes, no parallel edges, identical pixel coverage.
    """
    loops = [k for k, e in enumerate(graph.edges) if e.u == e.v]
    if not loops:
        return graph
    keep = [e for k, e in enumerate(graph.edges) if k not in set(loops)]
    next_id = (max(n.id for n in graph.nodes) + 1) if graph.nodes else 0
    nb = graph.node_by_id()
    for k in loops:
        e = graph.edges[k]
        pts = e.polyline
        if not pts or len(pts) < 6:               # too short to be a real region
            continue
        rad = e.radii or [e.w / 2.0] * len(pts)
        n = len(pts)
        cuts = [0, n // 3, 2 * n // 3]
        ids = [e.u]
        for c in cuts[1:]:
            y, x = float(pts[c][0]), float(pts[c][1])
            r = float(rad[c]) if c < len(rad) else e.w / 2.0
            graph.nodes.append(MechNode(id=next_id, x=x, y=y, r=r, degree=2))
            ids.append(next_id)
            next_id += 1
        bounds = cuts + [n]
        for s in range(3):
            seg = pts[bounds[s]:bounds[s + 1] + 1] or pts[bounds[s]:]
            segr = rad[bounds[s]:bounds[s + 1] + 1] or rad[bounds[s]:]
            a_id, b_id = ids[s], ids[(s + 1) % 3]
            a = nb.get(a_id) or graph.nodes[-1]
            b = graph.node_by_id().get(b_id)
            keep.append(MechEdge(
                u=a_id, v=b_id,
                w=float(np.mean(segr)) * 2.0 if len(segr) else e.w,
                length=float(math.hypot(a.x - b.x, a.y - b.y)) if b else e.length,
                curve_length=float(len(seg)),
                polyline=[list(map(int, p)) for p in seg],
                radii=[round(float(r), 2) for r in segr]))
    graph.edges = keep
    deg = {n.id: 0 for n in graph.nodes}
    for e in graph.edges:
        deg[e.u] = deg.get(e.u, 0) + 1
        deg[e.v] = deg.get(e.v, 0) + 1
    for nd in graph.nodes:
        nd.degree = deg.get(nd.id, 0)
    return graph


def from_raster(density, cond=None, cond_channels=None, scalars=None,
                scalar_names=None, meta=None, threshold=0.0, min_feature=12,
                prune_len=4):
    """Convert a mechanism density raster to a :class:`MechGraph`.

    density       : 2D array (any family).
    cond          : optional (C,H,W) conditioning stack for BVP tagging.
    cond_channels : channel names for ``cond`` (needs in_blob/out_blob/fixed_mask
                    and, for directions, in_dir_x/y & out_dir_x/y).
    scalars       : optional goal/label scalar vector -> stored in globals.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize
    import sknw

    binary = _clean(np.asarray(density) > threshold, min_feature)
    graph = MechGraph(nodes=[], edges=[], shape=list(binary.shape),
                      meta=dict(meta or {}))
    if scalars is not None:
        graph.globals["scalar_vector"] = [float(s) for s in np.asarray(scalars).ravel()]
        if scalar_names is not None:
            graph.globals["scalar_names"] = list(scalar_names)
    if binary.sum() < min_feature:
        return graph

    dist = distance_transform_edt(binary)
    skel = skeletonize(binary)
    if skel.sum() < 3:
        return graph
    g = _prune_spurs(sknw.build_sknw(skel.astype(np.uint16), multi=False), prune_len)
    ids = {node: i for i, node in enumerate(g.nodes())}
    for node in g.nodes():
        y, x = float(g.nodes[node]["o"][0]), float(g.nodes[node]["o"][1])
        graph.nodes.append(MechNode(id=ids[node], x=x, y=y,
                                    r=float(dist[int(y), int(x)]),
                                    degree=int(g.degree(node))))
    for u, v in g.edges():
        pts = g[u][v]["pts"]
        radii = dist[pts[:, 0], pts[:, 1]]
        a, b = graph.nodes[ids[u]], graph.nodes[ids[v]]
        graph.edges.append(MechEdge(
            u=ids[u], v=ids[v], w=float(radii.mean() * 2.0),
            length=float(math.hypot(a.x - b.x, a.y - b.y)),
            curve_length=float(len(pts)),
            polyline=pts.astype(int).tolist(),
            radii=[round(float(r), 2) for r in radii]))

    _split_self_loops(graph)

    # boundary conditions from the conditioning raster
    if cond is not None and cond_channels is not None and graph.nodes:
        ch = {name: cond[i] for i, name in enumerate(cond_channels)}
        for blob, role, dx, dy in (
                ("in_blob", ROLE_INPUT, "in_dir_x", "in_dir_y"),
                ("out_blob", ROLE_OUTPUT, "out_dir_x", "out_dir_y")):
            if blob in ch:
                c, vec = _centroid_vec(ch[blob], ch.get(dx), ch.get(dy))
                if c is not None:
                    nd = graph.nodes[_nearest(graph.nodes, *c)]
                    if role not in nd.roles:
                        nd.roles.append(role)
                    if vec is not None:
                        nd.vec = vec
        if "fixed_mask" in ch:
            fm = np.asarray(ch["fixed_mask"])
            for nd in graph.nodes:
                if fm[int(nd.y), int(nd.x)] > 0.5:
                    nd.fixed = True
                    if ROLE_SUPPORT not in nd.roles:
                        nd.roles.append(ROLE_SUPPORT)
    return graph


def densify(graph: MechGraph, max_seg: float = 16.0) -> MechGraph:
    """Subdivide edges along their skeleton polylines.

    The lean tensor encoding a generative model works with keeps only nodes and
    straight struts — it cannot carry a polyline.  A long curved member then
    collapses to its chord, which both misplaces material and gives the whole
    member one averaged width.  Subdividing at most every ``max_seg`` pixels of
    arc length turns that member into several short straight struts, each with
    its *local* width, which is a far better approximation at the cost of a few
    more node slots.

    Returns a new graph; the original is untouched.  Edges without polyline
    geometry are copied through unchanged.
    """
    out = MechGraph(nodes=[MechNode(n.id, n.x, n.y, n.r, n.degree, list(n.roles),
                                    None if n.vec is None else list(n.vec),
                                    n.fixed) for n in graph.nodes],
                    edges=[], shape=list(graph.shape),
                    globals=dict(graph.globals), label=dict(graph.label),
                    meta=dict(graph.meta))
    next_id = (max((n.id for n in out.nodes), default=-1)) + 1
    for e in graph.edges:
        pts, rad = e.polyline, e.radii
        arc = e.curve_length if e.curve_length > 0 else e.length
        n_seg = int(max(1, math.ceil(arc / max(1e-6, max_seg))))
        if not pts or len(pts) < 3 or n_seg <= 1:
            out.edges.append(MechEdge(e.u, e.v, e.w, e.length, e.curve_length,
                                      e.polyline, e.radii))
            continue
        rad = rad or [e.w / 2.0] * len(pts)
        cuts = [round(k * (len(pts) - 1) / n_seg) for k in range(n_seg + 1)]
        chain = [e.u]
        for c in cuts[1:-1]:
            y, x = float(pts[c][0]), float(pts[c][1])
            r = float(rad[min(c, len(rad) - 1)])
            out.nodes.append(MechNode(id=next_id, x=x, y=y, r=r, degree=2))
            chain.append(next_id)
            next_id += 1
        chain.append(e.v)
        nb = out.node_by_id()
        for s in range(n_seg):
            seg = pts[cuts[s]:cuts[s + 1] + 1]
            segr = rad[cuts[s]:cuts[s + 1] + 1] or [e.w / 2.0]
            a, b = nb[chain[s]], nb[chain[s + 1]]
            out.edges.append(MechEdge(
                u=chain[s], v=chain[s + 1],
                w=float(np.mean(segr)) * 2.0,
                length=float(math.hypot(a.x - b.x, a.y - b.y)),
                curve_length=float(len(seg)),
                polyline=[list(map(int, p)) for p in seg],
                radii=[round(float(r), 2) for r in segr]))
    deg = {n.id: 0 for n in out.nodes}
    for e in out.edges:
        deg[e.u] = deg.get(e.u, 0) + 1
        deg[e.v] = deg.get(e.v, 0) + 1
    for n in out.nodes:
        n.degree = deg.get(n.id, 0)
    return out


def to_raster(graph: MechGraph, shape=None, width_scale: float = 1.0,
              min_radius: float = 1.0):
    """Reconstruct a binary density from the graph (for FEA re-verification).

    Uses faithful ``polyline``/``radii`` geometry when present, else falls back
    to straight edge + mean width (the discrete "truss" view).  ``width_scale``
    multiplies every member radius — see :func:`rasterize_to_volume`.

    ``min_radius`` floors every stamped disk.  The medial-axis radius pinches to
    ~1px where thin members meet, and the gate's hinge check rejects any solid
    that does not survive a 1px erosion; a floor of ~2px guarantees each member
    is manufacturable-width, which is the lever :func:`reconstruct_valid` uses.
    """
    from skimage.draw import disk as _disk, line as _line
    shape = tuple(shape or graph.shape)
    m = np.zeros(shape, bool)

    def stamp(y, x, r):
        yy, xx = _disk((int(round(y)), int(round(x))),
                       max(min_radius, r * width_scale), shape=shape)
        m[yy, xx] = True

    nb = graph.node_by_id()
    for e in graph.edges:
        if e.polyline:
            for (r, c), rad in zip(e.polyline, e.radii):
                stamp(r, c, rad)
        elif e.u in nb and e.v in nb:
            a, b = nb[e.u], nb[e.v]
            rr, cc = _line(int(a.y), int(a.x), int(b.y), int(b.x))
            for r, c in zip(rr, cc):
                stamp(r, c, e.w / 2.0)
    for nd in graph.nodes:
        stamp(nd.y, nd.x, nd.r)
    return m


def rasterize_to_volume(graph: MechGraph, target_vf: float, domain_mask=None,
                        shape=None, iters: int = 14, lo: float = 0.25,
                        hi: float = 4.0, vf_fn=None):
    """Rasterize the graph with member widths scaled to hit a target volume.

    The verification gate holds a design to its spec's volume fraction within a
    few percent.  A graph reconstruction reproduces the *topology* faithfully but
    its absolute volume drifts — the medial-axis radii are a good but not exact
    account of how much material the original spent — and that drift alone was
    enough to fail designs whose connectivity, feature size and hinge checks all
    passed.

    The raster pipeline solves this by projecting a candidate onto the target
    volume.  A graph can do it more naturally: scale every member width by a
    single factor.  That is the same mechanism with thicker or thinner limbs,
    not a pixel-level edit, so the topology being verified is untouched.
    Bisection on the factor converges in a handful of rasterizations.

    ``vf_fn`` measures the volume fraction of a candidate density.  Pass the
    verifier's own measurement whenever the projection has to satisfy a gate:
    the gate canonicalizes the density to the spec's resolution and uses the
    *problem's* domain mask, so projecting against a differently-defined volume
    lands in the wrong place — measured, that mistake made the ceiling worse,
    not better.

    Returns ``(density, width_scale, achieved_vf)``.
    """
    shape = tuple(shape or graph.shape)
    mask = None if domain_mask is None else (np.asarray(domain_mask) > 0.5)
    denom = float(mask.sum()) if mask is not None else float(shape[0] * shape[1])
    if denom <= 0:
        denom = float(shape[0] * shape[1])

    def vf_of(s):
        m = to_raster(graph, shape=shape, width_scale=s)
        if mask is not None:
            m = m & mask
        if vf_fn is not None:
            return m, float(vf_fn(m))
        return m, float(m.sum()) / denom

    m_hi, v_hi = vf_of(hi)
    if v_hi < target_vf:                       # cannot reach it even at max width
        return m_hi, hi, v_hi
    m_lo, v_lo = vf_of(lo)
    if v_lo > target_vf:
        return m_lo, lo, v_lo

    best = (m_lo, lo, v_lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m, v = vf_of(mid)
        if abs(v - target_vf) < abs(best[2] - target_vf):
            best = (m, mid, v)
        if v < target_vf:
            lo = mid
        else:
            hi = mid
    return best


def reconstruct_valid(graph: MechGraph, target_vf=None, domain_mask=None,
                      shape=None, min_radius: float = 2.0, densify_seg: float = 12.0,
                      repair: bool = True, vf_fn=None, iters: int = 14):
    """Erosion-robust reconstruction: a density that survives the FEA gate.

    The plain medial-axis reconstruction reproduces a *shape* (Dice ~0.96) but
    not a *working mechanism*: it pinches to ~1px where thin members meet, and
    the gate rejects any solid that does not survive a 1px erosion.  Measured,
    the naive path passes the gate ~0.22 of the time; this path passes ~0.63
    (see docs/GRAPH_MODEL.md), which is the difference between "the graph
    representation is the wrong fit" and "it is competitive".

    The recipe, each step measured to matter:

    1. **Densify** — subdivide curved members so straight-segment stamping does
       not shortcut across the domain.
    2. **Floor the width** (``min_radius``) — every member is at least
       manufacturable-width, so it survives the erosion the hinge check applies.
    3. **Project to the target volume** — bisect a single width scale (above the
       floor) so the design meets its spec's volume fraction.
    4. **Clip to the declared domain** — material outside it is never valid.
    5. **Repair single-pixel hinges** — widen any residual articulation pixel.
       Unlike a global morphological closing (which welds the compliant gaps
       shut and destroys the mechanism), this is local to the pinch points and
       leaves the load path's compliance intact.

    ``vf_fn`` measures a candidate's volume the way the *verifier* does; pass it
    whenever the result must satisfy the gate.  Returns a boolean density.
    """
    g = densify(graph, max_seg=densify_seg) if densify_seg else graph
    shape = tuple(shape or graph.shape)
    dom = None if domain_mask is None else (np.asarray(domain_mask) > 0.5)

    def render(scale):
        m = to_raster(g, shape=shape, width_scale=scale, min_radius=min_radius)
        return (m & dom) if dom is not None else m

    def vf_of(m):
        if vf_fn is not None:
            return float(vf_fn(m))
        denom = float(dom.sum()) if dom is not None else float(shape[0] * shape[1])
        return float(m.sum()) / max(denom, 1.0)

    if target_vf is None:
        m = render(1.0)
    else:
        lo, hi = 0.25, 4.0
        if vf_of(render(hi)) < target_vf:
            m = render(hi)
        elif vf_of(render(lo)) > target_vf:
            m = render(lo)
        else:
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                if vf_of(render(mid)) < target_vf:
                    lo = mid
                else:
                    hi = mid
            m = render(0.5 * (lo + hi))

    if repair:
        from src.validation.connectivity import repair_point_hinges
        rep, _ = repair_point_hinges(m.astype(float))
        m = rep > 0.5
        if dom is not None:
            m = m & dom
    return m


def roundtrip_dice(density, graph: MechGraph, threshold=0.0, min_feature=12):
    """Dice(original binary, graph reconstruction) — the fidelity of the graph."""
    a = to_raster(graph)
    b = _clean(np.asarray(density) > threshold, min_feature)
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else 1.0
