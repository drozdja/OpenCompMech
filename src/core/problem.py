"""
Problem definition for 2D topology optimization.

Encapsulates:
- Domain geometry and masking
- Boundary conditions (fixed DOFs)
- Applied loads
- Material properties
- Volume constraint
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

from .mesh import Mesh2D, create_mesh


class ProblemType(Enum):
    """Type of optimization problem."""
    COMPLIANCE = "compliance"      # Minimize compliance (stiff)
    MECHANISM = "mechanism"        # Maximize output displacement (mech)


@dataclass
class Material:
    """Linear elastic isotropic material."""
    E: float = 1.0      # Young's modulus (normalized)
    nu: float = 0.3     # Poisson's ratio
    E_min: float = 1e-9  # Minimum stiffness for void (SIMP)
    
    def get_D_matrix(self, plane_stress: bool = True) -> np.ndarray:
        """
        Get constitutive matrix for plane stress/strain.
        
        Returns:
            D: (3, 3) constitutive matrix
        """
        E, nu = self.E, self.nu
        
        if plane_stress:
            factor = E / (1 - nu**2)
            D = factor * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu) / 2]
            ])
        else:  # Plane strain
            factor = E / ((1 + nu) * (1 - 2*nu))
            D = factor * np.array([
                [1 - nu, nu, 0],
                [nu, 1 - nu, 0],
                [0, 0, (1 - 2*nu) / 2]
            ])
        
        return D


@dataclass
class BoundaryCondition:
    """Fixed (Dirichlet) boundary condition."""
    node_indices: np.ndarray   # Which nodes are fixed
    directions: np.ndarray     # 0=x, 1=y, 2=both
    
    def get_fixed_dofs(self, mesh: Mesh2D) -> np.ndarray:
        """Get list of fixed DOF indices."""
        fixed_dofs = []
        for node, direction in zip(self.node_indices, self.directions):
            if direction == 0 or direction == 2:
                fixed_dofs.append(2 * node)      # x
            if direction == 1 or direction == 2:
                fixed_dofs.append(2 * node + 1)  # y
        return np.array(fixed_dofs, dtype=np.int32)


@dataclass
class Load:
    """Applied force."""
    node_index: int
    fx: float = 0.0
    fy: float = 0.0
    

@dataclass 
class Problem:
    """
    Complete problem definition for topology optimization.
    
    Attributes:
        mesh: FEM mesh
        material: Material properties
        bcs: List of boundary conditions
        loads: List of applied loads
        volume_fraction: Target volume fraction
        problem_type: COMPLIANCE or MECHANISM
        domain_mask: (nely, nelx) boolean mask for active elements
        filter_radius: Density filter radius (in elements)
    """
    mesh: Mesh2D
    material: Material
    bcs: List[BoundaryCondition]
    loads: List[Load]
    volume_fraction: float = 0.4
    problem_type: ProblemType = ProblemType.COMPLIANCE
    domain_mask: Optional[np.ndarray] = None
    filter_radius: float = 1.5
    
    # For mechanism problems
    output_node: Optional[int] = None
    output_direction: int = 0  # 0=x, 1=y
    k_spring: float = 0.0      # Output spring stiffness
    
    def __post_init__(self):
        """Initialize domain mask if not provided."""
        if self.domain_mask is None:
            self.domain_mask = np.ones((self.mesh.nely, self.mesh.nelx), dtype=bool)
    
    @property
    def n_active_elements(self) -> int:
        """Number of active (designable) elements."""
        return int(np.sum(self.domain_mask))
    
    def get_fixed_dofs(self) -> np.ndarray:
        """Get all fixed DOF indices from all BCs."""
        all_fixed = []
        for bc in self.bcs:
            all_fixed.extend(bc.get_fixed_dofs(self.mesh))
        return np.unique(np.array(all_fixed, dtype=np.int32))
    
    def get_free_dofs(self) -> np.ndarray:
        """Get all free (non-fixed) DOF indices."""
        all_dofs = np.arange(self.mesh.n_dofs)
        fixed = self.get_fixed_dofs()
        return np.setdiff1d(all_dofs, fixed)
    
    def get_force_vector(self) -> np.ndarray:
        """Assemble global force vector."""
        F = np.zeros(self.mesh.n_dofs)
        for load in self.loads:
            F[2 * load.node_index] = load.fx
            F[2 * load.node_index + 1] = load.fy
        return F
    
    def to_input_tensor(self) -> np.ndarray:
        """
        Convert problem to 8-channel input tensor for ML.
        
        Returns:
            tensor: (8, nely, nelx) float32 array
        """
        nely, nelx = self.mesh.nely, self.mesh.nelx
        tensor = np.zeros((8, nely, nelx), dtype=np.float32)
        
        # Channel 0: Domain mask
        tensor[0] = self.domain_mask.astype(np.float32)
        
        # Channels 1-2: Fixed DOFs (need to map nodes to elements)
        # For simplicity, mark elements adjacent to fixed nodes
        for bc in self.bcs:
            for node in bc.node_indices:
                # Find elements containing this node
                for ey in range(nely):
                    for ex in range(nelx):
                        elem = self.mesh.elements[ey * nelx + ex]
                        if node in elem:
                            for direction in ([0] if bc.directions[0] == 0 else 
                                            [1] if bc.directions[0] == 1 else [0, 1]):
                                tensor[1 + direction, ey, ex] = 1.0
        
        # Channels 3-4: Force location and direction
        for load in self.loads:
            # Find element containing load node
            node = load.node_index
            for ey in range(nely):
                for ex in range(nelx):
                    elem = self.mesh.elements[ey * nelx + ex]
                    if node in elem:
                        # Normalize force magnitude
                        f_mag = np.sqrt(load.fx**2 + load.fy**2)
                        if f_mag > 0:
                            tensor[3, ey, ex] = load.fx / f_mag
                            tensor[4, ey, ex] = load.fy / f_mag
        
        # Channels 5-6: Objective direction (for mechanisms)
        # For compliance problems, these stay zero
        
        # Channel 7: Output stiffness (normalized)
        tensor[7] = self.k_spring  # Scalar broadcast
        
        return tensor
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert problem definition to JSON-serializable metadata."""
        return {
            "resolution": self.mesh.nelx,
            "volume_fraction": self.volume_fraction,
            "problem_type": self.problem_type.value,
            "filter_radius": self.filter_radius,
            "n_fixed_nodes": sum(len(bc.node_indices) for bc in self.bcs),
            "n_loads": len(self.loads),
            "domain_type": "full" if np.all(self.domain_mask) else "masked",
        }


def create_cantilever_problem(nelx: int = 64, nely: int = 32,
                               volume_fraction: float = 0.4,
                               load_position: float = 1.0) -> Problem:
    """
    Create a classic cantilever beam problem.
    
    Fixed on left edge, point load on right edge.
    
    Args:
        nelx: Number of elements in X
        nely: Number of elements in Y  
        volume_fraction: Target volume
        load_position: Vertical position of load (0=bottom, 1=top)
    
    Returns:
        Problem instance
    """
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # Fix left edge (all nodes with x=0)
    left_nodes = np.array([i * (nelx + 1) for i in range(nely + 1)], dtype=np.int32)
    bc = BoundaryCondition(
        node_indices=left_nodes,
        directions=np.full(len(left_nodes), 2)  # Fix both x and y
    )
    
    # Point load on right edge
    load_node_y = int(load_position * nely)
    load_node = load_node_y * (nelx + 1) + nelx  # Right edge
    load = Load(node_index=load_node, fx=0.0, fy=-1.0)
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc],
        loads=[load],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE
    )


def create_bridge_problem(nelx: int = 64, nely: int = 32,
                          volume_fraction: float = 0.3) -> Problem:
    """
    Create a bridge problem (simply supported beam).
    
    Pinned supports at bottom corners, distributed or point load on top.
    """
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # Pin bottom-left (fix both)
    bc_left = BoundaryCondition(
        node_indices=np.array([0], dtype=np.int32),
        directions=np.array([2])  # Fix both
    )
    
    # Roller bottom-right (fix y only)
    bc_right = BoundaryCondition(
        node_indices=np.array([nelx], dtype=np.int32),
        directions=np.array([1])  # Fix y only
    )
    
    # Point load at top center
    top_center_node = nely * (nelx + 1) + nelx // 2
    load = Load(node_index=top_center_node, fx=0.0, fy=-1.0)
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc_left, bc_right],
        loads=[load],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE
    )


def create_lbracket_problem(nelx: int = 64, nely: int = 64,
                            volume_fraction: float = 0.4) -> Problem:
    """
    Create an L-bracket problem.
    
    L-shaped domain, fixed on top, load on right arm.
    """
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # L-shape mask: remove top-right quadrant
    mask = np.ones((nely, nelx), dtype=bool)
    mask[nely//2:, nelx//2:] = False
    
    # Fix top edge of vertical arm
    top_nodes = np.array([nely * (nelx + 1) + i for i in range(nelx // 2 + 1)], dtype=np.int32)
    bc = BoundaryCondition(
        node_indices=top_nodes,
        directions=np.full(len(top_nodes), 2)
    )
    
    # Load on right tip of horizontal arm
    load_node = (nely // 2) * (nelx + 1) + nelx
    load = Load(node_index=load_node, fx=0.0, fy=-1.0)
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc],
        loads=[load],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE,
        domain_mask=mask
    )
