"""Tests for the combinatorial-complex (ETNN) lift — src/graph/complex.py.

The load-bearing test is `test_euler_*`: face enumeration is validated against
Euler's formula (bounded faces == E - V + C), which is threshold-free and
independent of the implementation.
"""
import math
import os
import sys

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph import (MechGraph, MechNode, MechEdge, MechCell, build_cells,
                       to_cc_arrays, euler_check, CELL_SCALARS)


class _Skip(Exception):
    pass


def _require(*mods):
    for m in mods:
        try:
            __import__(m)
        except ImportError:
            if pytest is not None:
                pytest.skip(f"{m} not available")
            raise _Skip(f"{m} not available")


def _grid(nx, ny, step=20.0, r=2.0):
    """A regular nx*ny lattice: (nx-1)*(ny-1) square holes, known exactly."""
    nodes, edges, idx = [], [], {}
    k = 0
    for j in range(ny):
        for i in range(nx):
            idx[(i, j)] = k
            nodes.append(MechNode(k, 10.0 + i * step, 10.0 + j * step, r=r, degree=0))
            k += 1
    for j in range(ny):
        for i in range(nx):
            for di, dj in ((1, 0), (0, 1)):
                a, b = (i, j), (i + di, j + dj)
                if b in idx:
                    edges.append(MechEdge(idx[a], idx[b], w=2 * r, length=step,
                                          curve_length=step))
    for n in nodes:
        n.degree = sum(1 for e in edges if e.u == n.id or e.v == n.id)
    size = int(10 + max(nx, ny) * step + 10)
    return MechGraph(nodes, edges, shape=[size, size],
                     globals={"scalar_vector": [0.1] * 15})


def test_euler_single_square():
    g = _grid(2, 2)
    build_cells(g, pads=False)
    chk = euler_check(g)
    assert chk["ok"], chk
    assert chk["hole_cells"] == 1
    assert g.faces[0].kind == "hole"
    assert math.isclose(g.faces[0].area, 400.0, rel_tol=1e-6)   # 20x20


def test_euler_lattices():
    for nx, ny in [(2, 2), (3, 3), (4, 3), (5, 4)]:
        g = _grid(nx, ny)
        build_cells(g, pads=False)
        chk = euler_check(g)
        assert chk["ok"], (nx, ny, chk)
        assert chk["hole_cells"] == (nx - 1) * (ny - 1), (nx, ny, chk)


def test_tree_has_no_faces():
    """A spanning tree encloses nothing: zero rank-2 hole cells."""
    g = _grid(3, 3)
    keep, seen = [], set()
    for e in g.edges:                      # crude BFS spanning tree
        if e.u not in seen or e.v not in seen:
            keep.append(e)
            seen.add(e.u)
            seen.add(e.v)
    g.edges = keep
    build_cells(g, pads=False)
    assert euler_check(g)["ok"]
    assert len([f for f in g.faces if f.kind == "hole"]) == 0


def test_disconnected_components():
    """Two separate squares -> two holes; C=2 must be handled."""
    a, b = _grid(2, 2), _grid(2, 2)
    off = a.n_nodes()
    for n in b.nodes:
        n.id += off
        n.x += 200.0
    for e in b.edges:
        e.u += off
        e.v += off
    g = MechGraph(a.nodes + b.nodes, a.edges + b.edges, shape=[400, 400])
    build_cells(g, pads=False)
    chk = euler_check(g)
    assert chk["C"] == 2 and chk["ok"], chk
    assert chk["hole_cells"] == 2


def test_pad_cells():
    """A fat node fused with its neighbours becomes a rank-2 pad cell."""
    g = _grid(3, 3)
    g.nodes[4].r = 14.0                    # centre node is a blob
    build_cells(g, holes=False)
    pads = [f for f in g.faces if f.kind == "pad"]
    assert len(pads) == 1
    assert g.nodes[4].id in pads[0].nodes
    assert len(pads[0].nodes) >= 2         # never duplicates a rank-0 cell
    assert pads[0].area > 0


def test_cc_arrays_shapes_and_incidence():
    g = _grid(3, 3)
    build_cells(g)
    a = to_cc_arrays(g)
    n, e = g.n_nodes(), g.n_edges()
    f = len(g.faces)
    assert a["pos_0"].shape == (n, 2) and a["x_0"].shape == (n, 6)
    assert a["pos_1"].shape == (e, 2) and a["x_1"].shape == (e, 3)
    assert a["pos_2"].shape == (f, 2) and a["x_2"].shape == (f, len(CELL_SCALARS))
    assert a["inc_01"].shape == (2, 2 * e)          # each strut has 2 endpoints
    assert a["inc_12"].shape[0] == 2 and a["inc_02"].shape[0] == 2
    # every incidence index is in range
    assert a["inc_01"][0].max() < n and a["inc_01"][1].max() < e
    if a["inc_12"].shape[1]:
        assert a["inc_12"][0].max() < e and a["inc_12"][1].max() < f
    # rank-1 position is the strut midpoint
    e0 = g.edges[0]
    rows = {nd.id: i for i, nd in enumerate(g.nodes)}
    mid = 0.5 * (a["pos_0"][rows[e0.u]] + a["pos_0"][rows[e0.v]])
    assert np.allclose(a["pos_1"][0], mid, atol=1e-6)


def test_cc_se2_equivariance():
    """Rotate the mechanism: every rank's features invariant, positions rotate."""
    g = _grid(3, 3)
    g.nodes[4].r = 14.0
    build_cells(g)
    a0 = to_cc_arrays(g)

    theta = math.radians(37.0)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])
    c = max(g.shape) / 2.0
    for nd in g.nodes:
        x, y = nd.x - c, nd.y - c
        nd.x = float(R[0, 0] * x + R[0, 1] * y + c)
        nd.y = float(R[1, 0] * x + R[1, 1] * y + c)
    a1 = to_cc_arrays(g)                    # cells NOT rebuilt: same complex

    for k in ("x_0", "x_1", "x_2"):
        assert np.allclose(a0[k], a1[k], atol=1e-4), k
    for k in ("pos_0", "pos_1", "pos_2"):
        # positions rotate about the (normalized) centre
        p0 = (R @ (a0[k] - 0.5).T).T + 0.5
        assert np.allclose(p0, a1[k], atol=1e-4), k


def test_cells_serialize_roundtrip():
    g = _grid(3, 3)
    g.nodes[4].r = 14.0
    build_cells(g)
    g2 = MechGraph.from_dict(g.to_dict())
    assert len(g2.faces) == len(g.faces)
    assert all(isinstance(c, MechCell) for c in g2.faces)
    assert g2.faces[0].kind == g.faces[0].kind
    assert math.isclose(g2.faces[0].area, g.faces[0].area)
    a0, a1 = to_cc_arrays(g), to_cc_arrays(g2)
    assert np.allclose(a0["x_2"], a1["x_2"])


def test_euler_on_real_raster():
    """Euler must hold on a skeletonized raster, not just synthetic lattices."""
    _require("skimage", "sknw")
    from src.graph import from_raster
    img = np.zeros((96, 96), np.float32)
    img[20:28, 12:84] = 1.0                 # two horizontal bars
    img[68:76, 12:84] = 1.0
    img[20:76, 12:20] = 1.0                 # joined at both ends -> 1 loop
    img[20:76, 76:84] = 1.0
    g = from_raster(img)
    build_cells(g)
    assert euler_check(g)["ok"]
    assert len([f for f in g.faces if f.kind == "hole"]) >= 1


if __name__ == "__main__":
    ok = fail = skip = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
                ok += 1
            except _Skip as exc:
                print(f"SKIP {name}: {exc}")
                skip += 1
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {name}: {exc}")
                fail += 1
    print(f"\n{ok} passed, {fail} failed, {skip} skipped")
