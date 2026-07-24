"""Tests for the padded tensor encoding used by the graph generator.

Encode -> decode must preserve the mechanism well enough that a *generated*
sample can be trusted to mean what it says when it reaches the FEA gate.
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
from src.graph import MechGraph, MechNode, MechEdge, ROLE_INPUT, ROLE_OUTPUT
from src.ml.graph_tensors import (N_ANCHOR, N_FREE, N_MAX, NODE_CH, EDGE_CH,
                                  anchors_from_cond, encode, decode)
from src.ml.tensor_spec import COND_CHANNELS

SHAPE = (128, 128)


def _cond():
    """A conditioning stack with an input port, an output port and a support."""
    c = np.zeros((len(COND_CHANNELS), *SHAPE), np.float32)
    ix = {n: i for i, n in enumerate(COND_CHANNELS)}
    c[ix["in_blob"], 20:26, 10:16] = 1.0
    c[ix["in_dir_x"], 20:26, 10:16] = 1.0
    c[ix["out_blob"], 100:106, 110:116] = 1.0
    c[ix["out_dir_y"], 100:106, 110:116] = -1.0
    c[ix["fixed_mask"], 60:70, 4:10] = 1.0
    return c


def _graph():
    nodes = [MechNode(0, 13.0, 23.0, r=4.0, roles=[ROLE_INPUT], vec=[1.0, 0.0]),
             MechNode(1, 64.0, 64.0, r=5.0),
             MechNode(2, 113.0, 103.0, r=4.0, roles=[ROLE_OUTPUT], vec=[0.0, -1.0]),
             MechNode(3, 7.0, 65.0, r=6.0, fixed=True)]
    edges = [MechEdge(0, 1, w=8.0, length=math.hypot(51, 41)),
             MechEdge(1, 2, w=8.0, length=math.hypot(49, 39)),
             MechEdge(1, 3, w=9.0, length=math.hypot(57, 1))]
    for n in nodes:
        n.degree = sum(1 for e in edges if n.id in (e.u, e.v))
    return MechGraph(nodes, edges, shape=list(SHAPE),
                     globals={"scalar_vector": [0.1] * 15})


def test_anchor_extraction():
    a = anchors_from_cond(_cond(), COND_CHANNELS, SHAPE)
    assert a["anchor_pos"].shape == (N_ANCHOR, 2)
    assert a["anchor_present"][0] == 1.0 and a["anchor_present"][1] == 1.0
    assert a["anchor_present"][2] == 1.0            # one support region
    assert a["anchor_present"][3] == 0.0            # no second region
    # input anchor sits near the in_blob centroid (x~12.5, y~22.5), centred
    assert np.allclose(a["anchor_pos"][0], ((12.5 / 128 - 0.5) * 2,
                                            (22.5 / 128 - 0.5) * 2), atol=1e-3)
    assert np.allclose(a["anchor_vec"][0], (1.0, 0.0), atol=1e-6)


def test_encode_shapes_and_symmetry():
    e = encode(_graph(), _cond(), COND_CHANNELS)
    assert e["node_x"].shape == (N_FREE, NODE_CH)
    assert e["edge_x"].shape == (N_MAX, N_MAX, EDGE_CH)
    assert np.allclose(e["edge_x"], e["edge_x"].transpose(1, 0, 2))
    assert (e["node_x"][:, 3] > 0).sum() == 4       # four real nodes
    assert (e["node_x"][:, 3] < 0).sum() == N_FREE - 4
    # every channel stays inside the [-1, 1] the flow model operates on
    assert e["node_x"].min() >= -1.001 and e["node_x"].max() <= 1.001
    assert e["edge_x"].min() >= -1.001 and e["edge_x"].max() <= 1.001


def test_ports_wire_to_anchors():
    """A role-tagged node must be connected to its anchor: that is what teaches
    the model to terminate struts on the ports."""
    e = encode(_graph(), _cond(), COND_CHANNELS)
    adj = e["edge_x"][:, :, 0]
    assert adj[0, N_ANCHOR:].max() > 0              # input anchor is wired
    assert adj[1, N_ANCHOR:].max() > 0              # output anchor is wired


def test_encode_decode_roundtrip():
    g = _graph()
    c = _cond()
    e = encode(g, c, COND_CHANNELS)
    g2 = decode(e["node_x"], e["edge_x"], e["anchor_pos"], e["anchor_present"],
                shape=SHAPE)
    # 4 graph nodes + 3 present anchors
    assert g2.n_nodes() == 4 + int(e["anchor_present"].sum())
    assert g2.n_edges() >= g.n_edges()
    # positions survive the centred/normalised round trip
    got = sorted((round(n.x), round(n.y)) for n in g2.nodes)
    for n in g.nodes:
        assert (round(n.x), round(n.y)) in got, (n.x, n.y, got)


def test_decode_ignores_absent_slots():
    e = encode(_graph(), _cond(), COND_CHANNELS)
    node_x = e["node_x"].copy()
    node_x[:, 3] = -1.0                             # nothing exists
    g2 = decode(node_x, e["edge_x"], e["anchor_pos"], e["anchor_present"],
                shape=SHAPE, connect_anchors=False)
    assert g2.n_nodes() == int(e["anchor_present"].sum())
    assert g2.n_edges() == 0


def test_decode_connects_isolated_anchors():
    """A present anchor must never be left floating: it is a port/support, so a
    disconnected anchor disk is both an FEA-failing extra component and a
    physically meaningless unattached port."""
    e = encode(_graph(), _cond(), COND_CHANNELS)
    # strip every edge, so all anchors start isolated
    edge_x = np.full_like(e["edge_x"], -1.0)
    g_iso = decode(e["node_x"], edge_x, e["anchor_pos"], e["anchor_present"],
                   shape=SHAPE, connect_anchors=False)
    g_fix = decode(e["node_x"], edge_x, e["anchor_pos"], e["anchor_present"],
                   shape=SHAPE, connect_anchors=True)
    anchors = int(e["anchor_present"].sum())
    # every present anchor is isolated without the fix, connected with it
    iso_before = sum(1 for n in g_iso.nodes[:anchors] if n.degree == 0)
    iso_after = sum(1 for n in g_fix.nodes[:anchors] if n.degree == 0)
    assert iso_before == anchors and iso_after == 0


def test_decode_from_noise_is_safe():
    """A random sample must decode without raising: the eval loop cannot crash
    on a bad candidate, it must simply fail the gate."""
    rng = np.random.default_rng(0)
    a = anchors_from_cond(_cond(), COND_CHANNELS, SHAPE)
    for _ in range(20):
        node_x = rng.normal(size=(N_FREE, NODE_CH)).astype(np.float32)
        edge_x = rng.normal(size=(N_MAX, N_MAX, EDGE_CH)).astype(np.float32)
        edge_x = 0.5 * (edge_x + edge_x.transpose(1, 0, 2))
        g = decode(node_x, edge_x, a["anchor_pos"], a["anchor_present"],
                   shape=SHAPE)
        assert g.n_nodes() >= 0
        for n in g.nodes:
            assert 0 <= n.x < SHAPE[1] and 0 <= n.y < SHAPE[0]


if __name__ == "__main__":
    ok = fail = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {name}: {exc}")
                fail += 1
    print(f"\n{ok} passed, {fail} failed")
