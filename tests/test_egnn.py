"""Equivariance and shape tests for the EGNN graph denoiser.

The load-bearing test is `test_se2_equivariance`: rotating the whole problem —
node positions, anchor positions, and the load/motion direction vectors — must
rotate the predicted position velocity by the same rotation and leave every
invariant channel untouched.  If that fails, the model is not equivariant no
matter what the architecture claims, and the representation's main advantage
over a CNN is gone.

Requires torch; skipped otherwise.
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


class _Skip(Exception):
    pass


def _torch():
    try:
        import torch
        return torch
    except Exception as exc:                       # broken install counts as absent
        if pytest is not None:
            pytest.skip(f"torch unavailable: {exc}")
        raise _Skip(f"torch unavailable: {exc}")


def _setup(torch, b=3, seed=0):
    from src.ml.egnn import MechEGNN
    from src.ml.graph_tensors import N_ANCHOR, N_FREE, NODE_CH, EDGE_CH, N_MAX
    torch.manual_seed(seed)
    model = MechEGNN(n_anchor=N_ANCHOR, node_ch=NODE_CH, edge_ch=EDGE_CH,
                     scalar_dim=15, hidden=32, layers=3).double().eval()
    g = torch.Generator().manual_seed(seed)
    node_x = torch.randn(b, N_FREE, NODE_CH, generator=g, dtype=torch.float64)
    edge_x = torch.randn(b, N_MAX, N_MAX, EDGE_CH, generator=g, dtype=torch.float64)
    edge_x = 0.5 * (edge_x + edge_x.transpose(1, 2))
    t = torch.rand(b, generator=g, dtype=torch.float64) * 1000
    scal = torch.randn(b, 15, generator=g, dtype=torch.float64)
    apos = torch.randn(b, N_ANCHOR, 2, generator=g, dtype=torch.float64) * 0.4
    avec = torch.randn(b, N_ANCHOR, 2, generator=g, dtype=torch.float64)
    avec = avec / avec.norm(dim=-1, keepdim=True)
    apres = torch.ones(b, N_ANCHOR, dtype=torch.float64)
    aroles = torch.zeros(b, N_ANCHOR, 3, dtype=torch.float64)
    aroles[:, 0, 0] = aroles[:, 1, 1] = 1.0
    aroles[:, 2, 2] = aroles[:, 3, 2] = 1.0
    return model, (node_x, edge_x, t, scal, apos, avec, apres, aroles)


def test_output_shapes():
    torch = _torch()
    from src.ml.graph_tensors import N_FREE, NODE_CH, EDGE_CH, N_MAX
    model, args = _setup(torch)
    with torch.no_grad():
        vn, ve = model(*args)
    assert vn.shape == (3, N_FREE, NODE_CH)
    assert ve.shape == (3, N_MAX, N_MAX, EDGE_CH)


def test_edge_output_symmetric():
    """Struts are undirected: the edge velocity must be symmetric."""
    torch = _torch()
    model, args = _setup(torch)
    with torch.no_grad():
        _, ve = model(*args)
    assert torch.allclose(ve, ve.transpose(1, 2), atol=1e-10)


def test_se2_equivariance():
    torch = _torch()
    model, args = _setup(torch)
    node_x, edge_x, t, scal, apos, avec, apres, aroles = args

    th = math.radians(41.0)
    R = torch.tensor([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]], dtype=torch.float64)

    def rot(v):                                   # (B,N,2) @ R^T
        return v @ R.T

    with torch.no_grad():
        vn0, ve0 = model(node_x, edge_x, t, scal, apos, avec, apres, aroles)
        node_r = node_x.clone()
        node_r[:, :, 0:2] = rot(node_x[:, :, 0:2])
        vn1, ve1 = model(node_r, edge_x, t, scal, rot(apos), rot(avec),
                         apres, aroles)

    # positions: velocity rotates with the input
    assert torch.allclose(rot(vn0[:, :, 0:2]), vn1[:, :, 0:2], atol=1e-9), \
        (rot(vn0[:, :, 0:2]) - vn1[:, :, 0:2]).abs().max().item()
    # radius / existence: invariant
    assert torch.allclose(vn0[:, :, 2:], vn1[:, :, 2:], atol=1e-9)
    # adjacency / width: invariant
    assert torch.allclose(ve0, ve1, atol=1e-9)


def test_translation_equivariance():
    """Translating the problem must not change anything but the positions."""
    torch = _torch()
    model, args = _setup(torch)
    node_x, edge_x, t, scal, apos, avec, apres, aroles = args
    shift = torch.tensor([0.13, -0.07], dtype=torch.float64)
    with torch.no_grad():
        vn0, ve0 = model(node_x, edge_x, t, scal, apos, avec, apres, aroles)
        node_s = node_x.clone()
        node_s[:, :, 0:2] = node_x[:, :, 0:2] + shift
        vn1, ve1 = model(node_s, edge_x, t, scal, apos + shift, avec,
                         apres, aroles)
    # the velocity itself is a displacement: invariant under translation
    assert torch.allclose(vn0, vn1, atol=1e-9)
    assert torch.allclose(ve0, ve1, atol=1e-9)


def test_permutation_equivariance():
    """Relabelling the free slots permutes the output the same way."""
    torch = _torch()
    from src.ml.graph_tensors import N_ANCHOR, N_FREE
    model, args = _setup(torch)
    node_x, edge_x, t, scal, apos, avec, apres, aroles = args
    perm = torch.randperm(N_FREE, generator=torch.Generator().manual_seed(3))
    full = torch.cat([torch.arange(N_ANCHOR), perm + N_ANCHOR])
    with torch.no_grad():
        vn0, ve0 = model(node_x, edge_x, t, scal, apos, avec, apres, aroles)
        vn1, ve1 = model(node_x[:, perm], edge_x[:, full][:, :, full], t, scal,
                         apos, avec, apres, aroles)
    assert torch.allclose(vn0[:, perm], vn1, atol=1e-9)
    assert torch.allclose(ve0[:, full][:, :, full], ve1, atol=1e-9)


def test_absent_anchor_is_ignored():
    """A spec with no second support must not depend on that slot's position."""
    torch = _torch()
    model, args = _setup(torch)
    node_x, edge_x, t, scal, apos, avec, apres, aroles = args
    apres = apres.clone()
    apres[:, 3] = 0.0
    with torch.no_grad():
        vn0, ve0 = model(node_x, edge_x, t, scal, apos, avec, apres, aroles)
        apos2 = apos.clone()
        apos2[:, 3] = torch.tensor([0.9, -0.9], dtype=torch.float64)
        vn1, ve1 = model(node_x, edge_x, t, scal, apos2, avec, apres, aroles)
    assert torch.allclose(vn0, vn1, atol=1e-9)
    assert torch.allclose(ve0, ve1, atol=1e-9)


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
