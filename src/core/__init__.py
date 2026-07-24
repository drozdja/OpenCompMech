"""
Core module - Problem definitions, mesh generation, material properties.
"""

from .mesh import Mesh2D, create_mesh, create_graph_adjacency
from .problem import (
    Problem,
    ProblemType,
    Material,
    BoundaryCondition,
    Load,
    create_cantilever_problem,
    create_bridge_problem,
    create_lbracket_problem,
)

__all__ = [
    "Mesh2D",
    "create_mesh",
    "create_graph_adjacency",
    "Problem",
    "ProblemType",
    "Material",
    "BoundaryCondition",
    "Load",
    "create_cantilever_problem",
    "create_bridge_problem",
    "create_lbracket_problem",
]

