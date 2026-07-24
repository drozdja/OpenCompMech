"""Tests for the standardized mechanism graph representation (src/graph).

Schema / feature / serialization / equivariance tests are self-contained (a toy
graph, no heavy deps).  The raster->graph smoke is skipped if scikit-image/sknw
are unavailable.
"""
import math
import os
import sys

import numpy as np

try:
    import pytest
except ImportError:                                # runnable without pytest
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph import (MechGraph, MechNode, MechEdge, to_arrays, to_tensors,
                       roundtrip_dice)


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


def _toy():
    d = math.hypot(10, 20)
    nodes = [MechNode(0, 10, 10, r=2, degree=2, roles=["input"], vec=[1.0, 0.0]),
             MechNode(1, 30, 10, r=3, degree=2),
             MechNode(2, 20, 30, r=2, degree=2, roles=["support"], fixed=True)]
    edges = [MechEdge(0, 1, w=4, length=20, curve_length=20),
             MechEdge(1, 2, w=3, length=d, curve_length=25),
             MechEdge(2, 0, w=3, length=d, curve_length=23)]
    return MechGraph(nodes, edges, shape=[64, 64],
                     globals={"scalar_vector": [0.1] * 15})


def test_serialization_roundtrip():
    g = _toy()
    g2 = MechGraph.from_dict(g.to_dict())
    assert g2.n_nodes() == 3 and g2.n_edges() == 3
    assert g2.nodes[0].roles == ["input"] and g2.nodes[0].vec == [1.0, 0.0]
    assert g2.nodes[2].fixed is True


def test_to_arrays_shapes():
    a = to_arrays(_toy())
    assert a["pos"].shape == (3, 2)
    assert a["node_scalar"].shape == (3, 6)
    assert a["node_vec"].shape == (3, 2)
    assert a["edge_index"].shape == (2, 6)     # undirected -> both directions
    assert a["edge_scalar"].shape == (6, 3)
    assert a["graph_y"].shape == (15,)
    # role one-hots land in the right node
    assert a["node_scalar"][0, 2] == 1.0       # is_input on node 0
    assert a["node_scalar"][2, 4] == 1.0       # is_support on node 2


def test_se2_equivariance():
    g = _toy()
    a0 = to_arrays(g)
    theta = math.radians(37.0)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])
    c = max(g.shape) / 2.0
    for n in g.nodes:
        x, y = n.x - c, n.y - c
        n.x, n.y = float(R[0, 0]*x + R[0, 1]*y + c), float(R[1, 0]*x + R[1, 1]*y + c)
        if n.vec is not None:
            vx, vy = n.vec
            n.vec = [float(R[0, 0]*vx + R[0, 1]*vy), float(R[1, 0]*vx + R[1, 1]*vy)]
    a1 = to_arrays(g)                          # (edge lengths are rotation-invariant)
    assert np.allclose(a0["node_scalar"], a1["node_scalar"], atol=1e-4)
    assert np.allclose(a0["edge_scalar"], a1["edge_scalar"], atol=1e-4)
    assert np.allclose((R @ a0["node_vec"].T).T, a1["node_vec"], atol=1e-4)


def test_to_tensors_torch_or_numpy():
    t = to_tensors(_toy())
    # torch tensors when torch is present, else numpy arrays with same keys
    for k in ["pos", "x", "node_vec", "edge_index", "edge_attr", "y"]:
        assert k in t
    assert t["x"].shape[0] == 3 and t["x"].shape[1] == 6


def test_from_raster_roundtrip_synthetic():
    _require("skimage", "sknw")
    from src.graph import from_raster
    img = np.zeros((64, 64), np.float32)
    img[28:36, 8:56] = 1.0     # horizontal bar
    img[8:56, 28:36] = 1.0     # vertical bar -> a "+" with a central junction
    g = from_raster(img)
    assert g.n_nodes() >= 4 and g.n_edges() >= 3
    assert roundtrip_dice(img, g) > 0.7


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except _Skip as exc:
                print(f"SKIP {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {name}: {exc}")
