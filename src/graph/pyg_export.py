"""Export a MechGraph to a PyTorch Geometric ``Data`` object (or a plain tensor
dict if torch_geometric / torch are unavailable), ready for an EGNN stack.

Field mapping (EGNN convention):
    pos        -> Data.pos        (N,2)  equivariant node coordinates
    node_scalar-> Data.x          (N,6)  invariant node features
    node_vec   -> Data.node_vec   (N,2)  equivariant per-node direction
    edge_index -> Data.edge_index (2,E)
    edge_scalar-> Data.edge_attr  (E,3)  invariant edge features
    graph_y    -> Data.y          (1,G)  graph-level goal/label scalars
"""
from __future__ import annotations

from .mech_graph import MechGraph, to_arrays


def to_tensors(graph: MechGraph, undirected: bool = True):
    """Model-ready tensors as a dict, keyed by the PyG field names above.

    Uses torch when it is importable, else numpy — the keys and shapes are the
    same either way, so downstream code does not branch on the backend.
    """
    import numpy as np

    arr = to_arrays(graph, undirected=undirected)
    try:
        # not only ImportError: a torch install can be present but unloadable
        # (e.g. a missing libgomp / ROCm shared object).
        import torch
        f32, i64 = torch.float32, torch.long
        as_t = lambda a, dtype: torch.as_tensor(a, dtype=dtype)
        expand = lambda t: t.unsqueeze(0)
    except Exception:
        f32, i64 = np.float32, np.int64
        as_t = lambda a, dtype: np.asarray(a, dtype=dtype)
        expand = lambda a: a[None, ...]

    return {
        "pos": as_t(arr["pos"], f32),
        "x": as_t(arr["node_scalar"], f32),
        "node_vec": as_t(arr["node_vec"], f32),
        "edge_index": as_t(arr["edge_index"], i64),
        "edge_attr": as_t(arr["edge_scalar"], f32),
        "y": expand(as_t(arr["graph_y"], f32)),
    }


def to_pyg(graph: MechGraph, undirected: bool = True):
    """Return a torch_geometric ``Data`` if available, else the tensor dict.

    The dict has identical field names, so callers can build ``Data(**d)``
    themselves once torch_geometric is installed.
    """
    t = to_tensors(graph, undirected=undirected)
    try:
        from torch_geometric.data import Data
    except Exception:
        return t
    data = Data(x=t["x"], pos=t["pos"], edge_index=t["edge_index"],
                edge_attr=t["edge_attr"], y=t["y"])
    data.node_vec = t["node_vec"]
    return data
