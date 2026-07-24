"""Canonical, reusable graph representation for 2D compliant mechanisms.

Derived from the finished shape (generator-independent), SE(2)-equivariant and
EGNN-ready, verification-first (round-trips to a raster for the FEA gate).  See
``mech_graph`` module docstring for the full design rationale.

Typical use::

    from src.graph import from_raster, to_pyg, roundtrip_dice
    g = from_raster(density, cond=cond, cond_channels=names, scalars=scalars)
    data = to_pyg(g)                      # PyG Data (or tensor dict)
    fidelity = roundtrip_dice(density, g) # graph -> raster fidelity for FEA re-check
"""
from .mech_graph import (
    MechGraph, MechNode, MechEdge, MechCell,
    from_raster, to_raster, to_arrays, roundtrip_dice, densify, rasterize_to_volume, reconstruct_valid,
    NODE_SCALARS, EDGE_SCALARS, ROLES,
    ROLE_INPUT, ROLE_OUTPUT, ROLE_SUPPORT,
)
from .complex import build_cells, to_cc_arrays, euler_check, CELL_SCALARS
from .pyg_export import to_pyg, to_tensors

__all__ = [
    "MechGraph", "MechNode", "MechEdge", "MechCell",
    "from_raster", "to_raster", "to_arrays", "roundtrip_dice", "densify", "rasterize_to_volume", "reconstruct_valid",
    "build_cells", "to_cc_arrays", "euler_check",
    "to_pyg", "to_tensors",
    "NODE_SCALARS", "EDGE_SCALARS", "CELL_SCALARS", "ROLES",
    "ROLE_INPUT", "ROLE_OUTPUT", "ROLE_SUPPORT",
]
