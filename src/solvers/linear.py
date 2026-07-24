"""
Optimized Linear FEA solver for 2D topology optimization.

Highly optimized for EPYC batch generation:
- Vectorized assembly (no Python loops)
- Pre-computed DOF indices with LRU caching
- Memory-efficient sparse matrix construction

Uses direct solver (splu) for robustness with ill-conditioned SIMP matrices.
"""

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import splu
from dataclasses import dataclass
from typing import Tuple, Optional
from functools import lru_cache

from ..core.problem import Problem, Material
from ..core.mesh import Mesh2D


@dataclass
class FEAResult:
    """Results from FEA solve."""
    displacement: np.ndarray     # (n_dofs,) nodal displacements
    strain_energy: float         # Total strain energy
    compliance: float            # Compliance = F^T * u
    stress_vm: Optional[np.ndarray] = None  # (n_elements,) Von Mises stress


# =============================================================================
# Pre-computed element stiffness matrix (analytical for unit square Q4)
# =============================================================================

@lru_cache(maxsize=4)
def get_element_stiffness_cached(E: float = 1.0, nu: float = 0.3) -> np.ndarray:
    """
    Get Q4 element stiffness matrix for unit square element.
    
    Uses analytical integration (2x2 Gauss quadrature).
    Cached for repeated calls with same material.
    """
    # Constitutive matrix (plane stress)
    factor = E / (1 - nu**2)
    D = factor * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ], dtype=np.float64)
    
    # Gauss points
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = np.array([[-gp, -gp], [gp, -gp], [gp, gp], [-gp, gp]])
    
    ke = np.zeros((8, 8), dtype=np.float64)
    
    for xi, eta in gauss_pts:
        # Shape function derivatives
        dN_dxi = np.array([
            [-(1-eta)/4, (1-eta)/4, (1+eta)/4, -(1+eta)/4],
            [-(1-xi)/4, -(1+xi)/4, (1+xi)/4, (1-xi)/4]
        ])
        
        # For unit square: J = [[0.5, 0], [0, 0.5]], detJ = 0.25
        dN_dx = 2.0 * dN_dxi  # dN_dx = J^-1 @ dN_dxi
        
        # B matrix
        B = np.zeros((3, 8))
        B[0, 0::2] = dN_dx[0]  # du/dx
        B[1, 1::2] = dN_dx[1]  # dv/dy
        B[2, 0::2] = dN_dx[1]  # du/dy
        B[2, 1::2] = dN_dx[0]  # dv/dx
        
        # Integrate: weight=1, detJ=0.25, thickness=1
        ke += B.T @ D @ B * 0.25
    
    return ke


# =============================================================================
# Vectorized DOF index computation
# =============================================================================

def compute_element_dof_indices(nelx: int, nely: int) -> np.ndarray:
    """
    Pre-compute DOF indices for all elements (vectorized).
    
    Returns:
        edof: (n_elements, 8) DOF indices for each element
    """
    # Create element grid indices
    ey, ex = np.mgrid[0:nely, 0:nelx]
    ey = ey.flatten()
    ex = ex.flatten()
    
    # Node indices for each element
    n0 = ey * (nelx + 1) + ex       # bottom-left
    n1 = n0 + 1                      # bottom-right
    n2 = n1 + nelx + 1               # top-right
    n3 = n0 + nelx + 1               # top-left
    
    # DOF indices: [u0, v0, u1, v1, u2, v2, u3, v3]
    edof = np.column_stack([
        2*n0, 2*n0+1, 2*n1, 2*n1+1, 
        2*n2, 2*n2+1, 2*n3, 2*n3+1
    ]).astype(np.int32)
    
    return edof


@lru_cache(maxsize=16)
def get_cached_edof(nelx: int, nely: int) -> np.ndarray:
    """Cached DOF indices."""
    return compute_element_dof_indices(nelx, nely)


def compute_assembly_indices(edof: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute row/col indices for sparse matrix assembly.
    
    Returns:
        iK: (n_elements * 64,) row indices
        jK: (n_elements * 64,) col indices
    """
    # Broadcast to get all i,j pairs
    # For each element: outer product of edof with itself
    iK = np.repeat(edof, 8, axis=1).flatten()  # Each DOF repeated 8 times
    jK = np.tile(edof, (1, 8)).flatten()       # All DOFs tiled 8 times
    
    return iK.astype(np.int32), jK.astype(np.int32)


@lru_cache(maxsize=16)
def get_cached_assembly_indices(nelx: int, nely: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cached assembly indices."""
    edof = get_cached_edof(nelx, nely)
    return compute_assembly_indices(edof)


# =============================================================================
# Vectorized stiffness assembly
# =============================================================================

def assemble_stiffness_matrix_vectorized(
    nelx: int,
    nely: int, 
    n_dofs: int,
    density: np.ndarray,
    ke: np.ndarray,
    E: float = 1.0,
    E_min: float = 1e-9,
    penal: float = 3.0
) -> csr_matrix:
    """
    Vectorized global stiffness matrix assembly.
    
    No Python loops over elements - uses broadcasting.
    """
    # Get pre-computed indices
    iK, jK = get_cached_assembly_indices(nelx, nely)
    
    # SIMP material interpolation (vectorized)
    rho = density.flatten()
    Ee = E_min + np.power(rho, penal) * (E - E_min)  # (n_elem,)
    
    # Scale element stiffness by material
    # ke is (8,8), Ee is (n_elem,)
    # sK should be (n_elem * 64,)
    ke_flat = ke.flatten()  # (64,)
    sK = np.outer(Ee, ke_flat).flatten()  # (n_elem * 64,)
    
    # Assemble sparse matrix
    K = coo_matrix((sK, (iK, jK)), shape=(n_dofs, n_dofs))
    return K.tocsr()


# =============================================================================
# Vectorized sensitivity computation  
# =============================================================================

def compute_compliance_sensitivity_vectorized(
    nelx: int,
    nely: int,
    displacement: np.ndarray,
    density: np.ndarray,
    ke: np.ndarray,
    E: float = 1.0,
    E_min: float = 1e-9,
    penal: float = 3.0
) -> np.ndarray:
    """
    Vectorized compliance sensitivity computation.
    
    dc/drho_e = -p * rho_e^(p-1) * (E - E_min) * u_e^T * K0_e * u_e
    """
    edof = get_cached_edof(nelx, nely)  # (n_elem, 8)
    
    # Gather element displacements: u_e for all elements
    u_e = displacement[edof]  # (n_elem, 8)
    
    # Compute u^T K0 u for all elements (vectorized)
    # ke @ u_e.T gives (8, n_elem), then element-wise multiply and sum
    Ku = ke @ u_e.T  # (8, n_elem)
    uKu = np.sum(u_e.T * Ku, axis=0)  # (n_elem,)
    
    # SIMP derivative
    rho = density.flatten()
    # Handle rho=0 case to avoid 0^(p-1) issues
    rho_safe = np.maximum(rho, 1e-12)
    
    dc = -penal * np.power(rho_safe, penal - 1) * (E - E_min) * uKu
    
    return dc.reshape(nely, nelx)


def compute_von_mises_stress_vectorized(
    nelx: int,
    nely: int,
    displacement: np.ndarray,
    density: np.ndarray,
    E: float = 1.0,
    nu: float = 0.3,
    E_min: float = 1e-9,
    penal: float = 3.0
) -> np.ndarray:
    """
    Compute element-wise Von Mises stress (vectorized).
    
    Returns stress field on (nely, nelx) grid.
    """
    # Constitutive matrix (plane stress)
    factor = E / (1 - nu**2)
    D = factor * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])
    
    # B matrix at element center (xi=0, eta=0)
    B = np.zeros((3, 8))
    B[0, 0::2] = [-0.5, 0.5, 0.5, -0.5]
    B[1, 1::2] = [-0.5, -0.5, 0.5, 0.5]
    B[2, 0::2] = [-0.5, -0.5, 0.5, 0.5]
    B[2, 1::2] = [-0.5, 0.5, 0.5, -0.5]
    
    # Gather element displacements
    edof = get_cached_edof(nelx, nely)
    u_e = displacement[edof]  # (n_elem, 8)
    
    # Strain at all elements: (n_elem, 3)
    strain = (B @ u_e.T).T  # (n_elem, 3)
    
    # SIMP scaling
    rho = density.flatten()
    Ee = E_min + np.power(rho, penal) * (E - E_min)
    
    # Stress: (n_elem, 3) - scale by effective modulus
    # Note: D already has E in it, we scale relative
    stress = (Ee / E)[:, np.newaxis] * (D @ strain.T).T
    
    # Von Mises
    s11, s22, s12 = stress[:, 0], stress[:, 1], stress[:, 2]
    vm = np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2)
    
    return vm.reshape(nely, nelx)


# =============================================================================
# Main solver interface
# =============================================================================

def solve_fea_fast(
    problem: Problem,
    density: np.ndarray,
    penal: float = 3.0,
    compute_stress: bool = False
) -> FEAResult:
    """
    Fast FEA solve using vectorized assembly.
    
    Optimized for EPYC batch generation.
    """
    mesh = problem.mesh
    nelx, nely = mesh.nelx, mesh.nely
    
    # Get cached element stiffness
    E = problem.material.E
    nu = problem.material.nu
    E_min = problem.material.E_min
    ke = get_element_stiffness_cached(E, nu)
    
    # Vectorized assembly
    K = assemble_stiffness_matrix_vectorized(
        nelx, nely, mesh.n_dofs, density, ke, E, E_min, penal
    )
    
    # Force vector
    F = problem.get_force_vector()
    
    # BC elimination
    fixed_dofs = problem.get_fixed_dofs()
    free_dofs = problem.get_free_dofs()
    
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]
    
    # Direct solve
    lu = splu(K_ff.tocsc())
    u_f = lu.solve(F_f)
    
    # Reconstruct full displacement
    u = np.zeros(mesh.n_dofs)
    u[free_dofs] = u_f
    
    # Compliance
    compliance = float(F @ u)
    
    # Compute stress if requested
    stress_vm = None
    if compute_stress:
        stress_vm = compute_von_mises_stress_vectorized(
            nelx, nely, u, density, E, nu, E_min, penal
        )
    
    return FEAResult(
        displacement=u,
        strain_energy=0.5 * compliance,
        compliance=compliance,
        stress_vm=stress_vm
    )


def compute_sensitivity_fast(
    problem: Problem,
    density: np.ndarray,
    displacement: np.ndarray,
    penal: float = 3.0
) -> np.ndarray:
    """
    Fast compliance sensitivity using vectorized computation.
    """
    mesh = problem.mesh
    E = problem.material.E
    nu = problem.material.nu
    E_min = problem.material.E_min
    ke = get_element_stiffness_cached(E, nu)
    
    return compute_compliance_sensitivity_vectorized(
        mesh.nelx, mesh.nely, displacement, density,
        ke, E, E_min, penal
    )


# =============================================================================
# Legacy interface (for compatibility)
# =============================================================================

def get_unit_element_stiffness(material: Material = None) -> np.ndarray:
    """Get unit element stiffness matrix (legacy interface)."""
    if material is None:
        material = Material()
    return get_element_stiffness_cached(material.E, material.nu).copy()


def assemble_stiffness_matrix(
    problem: Problem,
    density: np.ndarray,
    penal: float = 3.0
) -> csr_matrix:
    """Legacy interface - calls vectorized version."""
    mesh = problem.mesh
    ke = get_unit_element_stiffness(problem.material)
    return assemble_stiffness_matrix_vectorized(
        mesh.nelx, mesh.nely, mesh.n_dofs, density, ke,
        problem.material.E, problem.material.E_min, penal
    )


def solve_fea(
    problem: Problem,
    density: np.ndarray,
    penal: float = 3.0,
    compute_stress: bool = False
) -> FEAResult:
    """Legacy interface - calls fast version."""
    return solve_fea_fast(problem, density, penal, compute_stress)


def compute_compliance_sensitivity(
    problem: Problem,
    density: np.ndarray,
    displacement: np.ndarray,
    penal: float = 3.0
) -> np.ndarray:
    """Legacy interface - calls vectorized version."""
    return compute_sensitivity_fast(problem, density, displacement, penal)


def compute_von_mises_stress(
    problem: Problem,
    density: np.ndarray,
    displacement: np.ndarray,
    penal: float = 3.0
) -> np.ndarray:
    """
    Compute element-wise Von Mises stress (vectorized).
    """
    mesh = problem.mesh
    nelx, nely = mesh.nelx, mesh.nely
    
    E = problem.material.E
    nu = problem.material.nu
    E_min = problem.material.E_min
    
    # Constitutive matrix
    factor = E / (1 - nu**2)
    D = factor * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])
    
    # B matrix at element center (xi=0, eta=0)
    B = np.zeros((3, 8))
    B[0, 0::2] = [-0.5, 0.5, 0.5, -0.5]
    B[1, 1::2] = [-0.5, -0.5, 0.5, 0.5]
    B[2, 0::2] = [-0.5, -0.5, 0.5, 0.5]
    B[2, 1::2] = [-0.5, 0.5, 0.5, -0.5]
    
    # Gather element displacements
    edof = get_cached_edof(nelx, nely)
    u_e = displacement[edof]  # (n_elem, 8)
    
    # Strain at all elements: (n_elem, 3)
    strain = (B @ u_e.T).T  # (n_elem, 3)
    
    # SIMP scaling
    rho = density.flatten()
    Ee = E_min + np.power(rho, penal) * (E - E_min)
    
    # Stress: (n_elem, 3)
    stress = Ee[:, np.newaxis] * (D @ strain.T).T
    
    # Von Mises
    s11, s22, s12 = stress[:, 0], stress[:, 1], stress[:, 2]
    vm = np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2)
    
    return vm.reshape(nely, nelx)
