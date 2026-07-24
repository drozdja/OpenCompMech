"""Padded tensor encoding of a MechGraph, for generative graph models.

The raster model denoises a 128x128 image.  The graph model denoises a *set*:
a fixed budget of node slots (position, radius, existence) plus a dense
adjacency (strut present, strut width).  This module is the bridge between
:class:`src.graph.MechGraph` and those tensors, in both directions, so a
generated sample can be decoded and pushed straight through ``to_raster`` into
the same sparse-FEA gate the raster model is judged by.

Two design points worth stating:

**The boundary-value problem is given, not generated.**  The first
``N_ANCHOR`` slots are *anchor* nodes read off the conditioning rasters — the
input port, the output port, and up to two support regions.  Their positions
are known at sampling time for any spec, so they are supplied to the network
rather than denoised.  They still take part in message passing and may be
connected by generated struts, which is exactly the physical statement "the
mechanism must attach to its ports".

**Everything is centred.**  Positions are stored centred on the domain and
scaled to ~[-1, 1], so a rotation of the problem acts as a plain linear map on
the coordinate channels and the equivariance of the EGNN denoiser is exact.
"""
from __future__ import annotations

import math

import numpy as np

from src.graph import MechGraph, MechNode, MechEdge, ROLE_INPUT, ROLE_OUTPUT, ROLE_SUPPORT

N_ANCHOR = 4                     # input, output, support x2
N_MAX = 28                       # total node slots
N_FREE = N_MAX - N_ANCHOR        # slots the model generates

RAD_MAX_FRAC = 0.15              # radius / width normalisation, as a fraction of
WID_MAX_FRAC = 0.15              # the domain size (covers observed geometry)

NODE_CH = 4                      # pos_x, pos_y, radius, existence
EDGE_CH = 2                      # adjacency, width
ANCHOR_ROLES = ("input", "output", "support", "support")


def _centre(p, scale):
    """pixel -> centred, ~[-1, 1]."""
    return (np.asarray(p, np.float32) / scale - 0.5) * 2.0


def _uncentre(p, scale):
    return (np.asarray(p, np.float32) * 0.5 + 0.5) * scale


def _blob_centroid(mask, dir_x=None, dir_y=None):
    ys, xs = np.where(np.asarray(mask) > 0.5)
    if len(ys) == 0:
        return None, (0.0, 0.0)
    c = (float(xs.mean()), float(ys.mean()))          # (x, y)
    v = (0.0, 0.0)
    if dir_x is not None and dir_y is not None:
        vx = float(np.asarray(dir_x)[ys, xs].mean())
        vy = float(np.asarray(dir_y)[ys, xs].mean())
        n = math.hypot(vx, vy)
        if n > 1e-6:
            v = (vx / n, vy / n)
    return c, v


def _support_centroids(fixed_mask, k=2):
    """Centroids of the k largest fixed regions."""
    m = np.asarray(fixed_mask) > 0.5
    if not m.any():
        return []
    try:
        from scipy.ndimage import label
        lab, n = label(m)
    except Exception:
        lab, n = m.astype(np.int32), 1
    sizes = [(int((lab == i).sum()), i) for i in range(1, n + 1)]
    sizes.sort(reverse=True)
    out = []
    for _, i in sizes[:k]:
        ys, xs = np.where(lab == i)
        out.append((float(xs.mean()), float(ys.mean())))
    return out


def anchors_from_cond(cond, cond_channels, shape) -> dict:
    """Read the BVP anchors (ports + supports) from the conditioning stack.

    Identical at training and sampling time — there is no ground-truth-only
    information here, so nothing leaks and there is no train/test mismatch.
    """
    scale = float(max(shape))
    ch = {name: cond[i] for i, name in enumerate(cond_channels)}
    pos = np.zeros((N_ANCHOR, 2), np.float32)
    vec = np.zeros((N_ANCHOR, 2), np.float32)
    present = np.zeros((N_ANCHOR,), np.float32)

    c, v = _blob_centroid(ch.get("in_blob", np.zeros(shape)),
                          ch.get("in_dir_x"), ch.get("in_dir_y"))
    if c is not None:
        pos[0], vec[0], present[0] = _centre(c, scale), v, 1.0
    c, v = _blob_centroid(ch.get("out_blob", np.zeros(shape)),
                          ch.get("out_dir_x"), ch.get("out_dir_y"))
    if c is not None:
        pos[1], vec[1], present[1] = _centre(c, scale), v, 1.0
    for j, c in enumerate(_support_centroids(ch.get("fixed_mask",
                                                    np.zeros(shape)), k=2)):
        pos[2 + j], present[2 + j] = _centre(c, scale), 1.0

    roles = np.zeros((N_ANCHOR, 3), np.float32)      # in / out / support
    roles[0, 0] = roles[1, 1] = 1.0
    roles[2, 2] = roles[3, 2] = 1.0
    return {"anchor_pos": pos, "anchor_vec": vec, "anchor_present": present,
            "anchor_roles": roles}


# --------------------------------------------------------------------------- #
# encode / decode                                                             #
# --------------------------------------------------------------------------- #
def encode(graph: MechGraph, cond, cond_channels) -> dict:
    """MechGraph (+ conditioning) -> padded tensors.

    Free slots are filled with the graph's nodes in a canonical order (by y then
    x).  The network is permutation-equivariant, so a canonical assignment makes
    the regression target well defined; slot identity during sampling comes from
    each slot's own noise trajectory.
    """
    shape = graph.shape or [128, 128]
    scale = float(max(shape))
    out = anchors_from_cond(cond, cond_channels, shape)

    nodes = sorted(graph.nodes, key=lambda n: (round(n.y, 3), round(n.x, 3)))
    nodes = nodes[:N_FREE]
    row = {n.id: i for i, n in enumerate(nodes)}

    node_x = np.zeros((N_FREE, NODE_CH), np.float32)
    node_x[:, 3] = -1.0                               # existence: absent
    for i, n in enumerate(nodes):
        node_x[i, 0:2] = _centre((n.x, n.y), scale)
        node_x[i, 2] = np.clip(n.r / (RAD_MAX_FRAC * scale), 0.0, 1.0) * 2 - 1
        node_x[i, 3] = 1.0

    # Adjacency spans every slot, so struts may attach to the anchors.
    edge_x = np.zeros((N_MAX, N_MAX, EDGE_CH), np.float32)
    edge_x[:, :, 0] = -1.0
    for e in graph.edges:
        if e.u in row and e.v in row:
            a, b = N_ANCHOR + row[e.u], N_ANCHOR + row[e.v]
            w = np.clip(e.w / (WID_MAX_FRAC * scale), 0.0, 1.0) * 2 - 1
            edge_x[a, b, 0] = edge_x[b, a, 0] = 1.0
            edge_x[a, b, 1] = edge_x[b, a, 1] = w

    # A node tagged with a role is wired to the corresponding anchor: this is
    # what teaches the model to terminate struts on the ports.
    for n in graph.nodes:
        if n.id not in row:
            continue
        b = N_ANCHOR + row[n.id]
        for role, a in ((ROLE_INPUT, 0), (ROLE_OUTPUT, 1)):
            if role in n.roles and out["anchor_present"][a] > 0:
                edge_x[a, b, 0] = edge_x[b, a, 0] = 1.0
                edge_x[a, b, 1] = edge_x[b, a, 1] = node_x[row[n.id], 2]
        if ROLE_SUPPORT in n.roles or n.fixed:
            for a in (2, 3):
                if out["anchor_present"][a] > 0:
                    d = np.linalg.norm(out["anchor_pos"][a] - node_x[row[n.id], 0:2])
                    if d < 0.25:                       # attach to the near support
                        edge_x[a, b, 0] = edge_x[b, a, 0] = 1.0
                        edge_x[a, b, 1] = edge_x[b, a, 1] = node_x[row[n.id], 2]

    out["node_x"] = node_x
    out["edge_x"] = edge_x
    out["n_nodes"] = np.float32(len(nodes))
    return out


def decode(node_x, edge_x, anchor_pos, anchor_present, shape=(128, 128),
           thresh=0.0, connect_anchors=True) -> MechGraph:
    """Padded tensors -> MechGraph, ready for ``to_raster`` and the FEA gate.

    ``connect_anchors`` wires any present anchor that ended up isolated to its
    nearest body node.  This is not cosmetic: an anchor is a port or support, so
    a floating anchor disk is both a disconnected component (which the gate
    rejects outright) and a physically meaningless "port not attached to the
    mechanism".  ``encode`` stored ports as roles *on* body nodes and the padded
    layout splits them back into separate slots, so reconnecting restores what
    the encoding already knew; for a generated sample it is the same kind of
    decode-time constraint projection as scaling widths to hit a volume."""
    shape = tuple(int(s) for s in shape)
    scale = float(max(shape))
    node_x = np.asarray(node_x, np.float32)
    edge_x = np.asarray(edge_x, np.float32)
    anchor_pos = np.asarray(anchor_pos, np.float32)
    anchor_present = np.asarray(anchor_present, np.float32)

    g = MechGraph(nodes=[], edges=[], shape=list(shape))
    slot2id = {}
    for a in range(N_ANCHOR):
        if anchor_present[a] > 0.5:
            x, y = _uncentre(anchor_pos[a], scale)
            slot2id[a] = len(g.nodes)
            roles = [{0: ROLE_INPUT, 1: ROLE_OUTPUT}.get(a, ROLE_SUPPORT)]
            g.nodes.append(MechNode(id=slot2id[a], x=float(x), y=float(y),
                                    r=max(2.0, 0.02 * scale), roles=roles,
                                    fixed=(a >= 2)))
    for i in range(N_FREE):
        if node_x[i, 3] <= thresh:
            continue
        x, y = _uncentre(node_x[i, 0:2], scale)
        if not (0 <= x < shape[1] and 0 <= y < shape[0]):
            continue
        r = (np.clip(node_x[i, 2], -1, 1) * 0.5 + 0.5) * RAD_MAX_FRAC * scale
        slot2id[N_ANCHOR + i] = len(g.nodes)
        g.nodes.append(MechNode(id=slot2id[N_ANCHOR + i], x=float(x),
                                y=float(y), r=float(max(1.0, r))))

    for a in range(N_MAX):
        for b in range(a + 1, N_MAX):
            if a not in slot2id or b not in slot2id:
                continue
            if edge_x[a, b, 0] <= thresh:
                continue
            w = (np.clip(edge_x[a, b, 1], -1, 1) * 0.5 + 0.5) * WID_MAX_FRAC * scale
            u, v = slot2id[a], slot2id[b]
            nu, nv = g.nodes[u], g.nodes[v]
            g.edges.append(MechEdge(u=u, v=v, w=float(max(2.0, w)),
                                    length=float(math.hypot(nu.x - nv.x,
                                                            nu.y - nv.y))))
    deg = {n.id: 0 for n in g.nodes}
    for e in g.edges:
        deg[e.u] += 1
        deg[e.v] += 1

    if connect_anchors:
        anchor_ids = {slot2id[a] for a in range(N_ANCHOR) if a in slot2id}
        body = [n for n in g.nodes if n.id not in anchor_ids]
        for a in range(N_ANCHOR):
            aid = slot2id.get(a)
            if aid is None or deg.get(aid, 0) > 0:
                continue
            na = g.nodes[aid]
            # prefer a body node; fall back to any other node
            cands = body or [n for n in g.nodes if n.id != aid]
            if not cands:
                continue
            nb = min(cands, key=lambda n: (n.x - na.x) ** 2 + (n.y - na.y) ** 2)
            w = float(max(2.0, min(na.r, nb.r) * 2.0))
            g.edges.append(MechEdge(u=aid, v=nb.id, w=w,
                                    length=float(math.hypot(na.x - nb.x,
                                                            na.y - nb.y))))
            deg[aid] = deg.get(aid, 0) + 1
            deg[nb.id] = deg.get(nb.id, 0) + 1

    for n in g.nodes:
        n.degree = deg[n.id]
    return g
