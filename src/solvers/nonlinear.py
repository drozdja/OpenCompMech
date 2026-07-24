"""
Geometric Non-Linear FEA Solver using Newton-Raphson.

Implements Total Lagrangian formulation with Green-Lagrange strain
for capturing large rotations, buckling, and snap-through behavior.
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import splu, spsolve
from dataclasses import dataclass
from typing import Optional, Tuple
import signal
import warnings

from ..core.problem import Problem
from ..core.mesh import Mesh2D, create_mesh


# ============================================================================
# Configuration
# ============================================================================

TIMEOUT_SECONDS = 300        # 5 minutes max per sample
MAX_NR_ITERATIONS = 100      # At ~800ms each = 80s max for NR alone
NR_TOLERANCE = 1e-6          # Residual norm tolerance
LINE_SEARCH_MAX_ITER = 10    # Max line search iterations
LINE_SEARCH_ALPHA_MIN = 0.1  # Minimum step size


@dataclass
class NonlinearResult:
    """Result of nonlinear FEA solve."""
    displacement: np.ndarray      # (2, ny+1, nx+1)
    stress: np.ndarray            # (ny, nx) von Mises
    converged: bool
    nr_iterations: int
    final_residual: float
    failure_reason: Optional[str] = None


# ============================================================================
# Element Formulation (Q4 with Green-Lagrange Strain)
# ============================================================================

def gauss_points_2x2():
    """2x2 Gauss quadrature points and weights."""
    g = 1.0 / np.sqrt(3.0)
    points = np.array([[-g, -g], [g, -g], [g, g], [-g, g]])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    return points, weights


def shape_functions_and_derivatives(xi: float, eta: float):
    """
    Q4 shape functions and natural derivatives.
    
    Returns:
        N: (4,) shape functions
        dN: (4, 2) derivatives [dN/dxi, dN/deta]
    """
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ])
    
    dN = 0.25 * np.array([
        [-(1 - eta), -(1 - xi)],
        [ (1 - eta), -(1 + xi)],
        [ (1 + eta),  (1 + xi)],
        [-(1 + eta),  (1 - xi)]
    ])
    
    return N, dN


def compute_B_linear(dN_dx: np.ndarray) -> np.ndarray:
    """
    Linear strain-displacement matrix for Q4 element.
    
    Args:
        dN_dx: (4, 2) shape function derivatives in physical coords
        
    Returns:
        B: (3, 8) strain-displacement matrix
    """
    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2*i] = dN_dx[i, 0]       # ε_xx
        B[1, 2*i+1] = dN_dx[i, 1]     # ε_yy
        B[2, 2*i] = dN_dx[i, 1]       # γ_xy
        B[2, 2*i+1] = dN_dx[i, 0]
    return B


def compute_B_nonlinear(dN_dx: np.ndarray, u_elem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nonlinear strain-displacement matrix (Total Lagrangian).
    
    Green-Lagrange strain: E = 0.5*(F^T F - I) = ε + 0.5*(∇u)^T(∇u)
    
    Args:
        dN_dx: (4, 2) shape function derivatives
        u_elem: (8,) element displacement [u1, v1, u2, v2, ...]
        
    Returns:
        B_L: (3, 8) linear part
        B_NL: (3, 8) nonlinear part
    """
    # Displacement gradient
    grad_u = np.zeros((2, 2))  # [du/dx, du/dy; dv/dx, dv/dy]
    for i in range(4):
        grad_u[0, 0] += dN_dx[i, 0] * u_elem[2*i]      # du/dx
        grad_u[0, 1] += dN_dx[i, 1] * u_elem[2*i]      # du/dy
        grad_u[1, 0] += dN_dx[i, 0] * u_elem[2*i+1]    # dv/dx
        grad_u[1, 1] += dN_dx[i, 1] * u_elem[2*i+1]    # dv/dy
    
    # Linear B-matrix
    B_L = compute_B_linear(dN_dx)
    
    # Nonlinear B-matrix (depends on current displacement)
    B_NL = np.zeros((3, 8))
    for i in range(4):
        # ε_xx nonlinear: 0.5 * (du/dx)^2 + 0.5 * (dv/dx)^2
        B_NL[0, 2*i] = grad_u[0, 0] * dN_dx[i, 0]
        B_NL[0, 2*i+1] = grad_u[1, 0] * dN_dx[i, 0]
        
        # ε_yy nonlinear: 0.5 * (du/dy)^2 + 0.5 * (dv/dy)^2
        B_NL[1, 2*i] = grad_u[0, 1] * dN_dx[i, 1]
        B_NL[1, 2*i+1] = grad_u[1, 1] * dN_dx[i, 1]
        
        # γ_xy nonlinear
        B_NL[2, 2*i] = grad_u[0, 0] * dN_dx[i, 1] + grad_u[0, 1] * dN_dx[i, 0]
        B_NL[2, 2*i+1] = grad_u[1, 0] * dN_dx[i, 1] + grad_u[1, 1] * dN_dx[i, 0]
    
    return B_L, B_NL


def compute_geometric_stiffness_matrix(dN_dx: np.ndarray, stress: np.ndarray, detJ: float) -> np.ndarray:
    """
    Geometric stiffness matrix (stress stiffness).
    
    K_σ accounts for the effect of initial stress on stiffness.
    
    Args:
        dN_dx: (4, 2) shape function derivatives
        stress: (3,) Cauchy stress [σ_xx, σ_yy, τ_xy]
        detJ: Jacobian determinant
        
    Returns:
        K_sigma: (8, 8) geometric stiffness
    """
    # Stress matrix
    S = np.array([
        [stress[0], stress[2]],
        [stress[2], stress[1]]
    ])
    
    K_sigma = np.zeros((8, 8))
    for i in range(4):
        for j in range(4):
            val = dN_dx[i, :] @ S @ dN_dx[j, :]
            K_sigma[2*i, 2*j] = val
            K_sigma[2*i+1, 2*j+1] = val
    
    return K_sigma * detJ


# ============================================================================
# Newton-Raphson Solver
# ============================================================================

class TimeoutError(Exception):
    """Raised when Newton-Raphson exceeds time limit."""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Newton-Raphson exceeded time limit")


def newton_raphson(
    problem: Problem,
    density: np.ndarray,
    mesh: Optional[Mesh2D] = None,
    max_iter: int = MAX_NR_ITERATIONS,
    tol: float = NR_TOLERANCE,
    timeout: Optional[int] = None,
    force_non_convergence: bool = False,  # For testing timeout
) -> NonlinearResult:
    """
    Solve nonlinear FEA using Newton-Raphson iteration.
    
    Uses Total Lagrangian formulation with incremental loading.
    
    Args:
        problem: Problem definition (BCs, loads, material)
        density: (ny, nx) element densities
        mesh: Pre-created mesh (optional)
        max_iter: Maximum NR iterations
        tol: Convergence tolerance
        timeout: Timeout in seconds (None = no timeout)
        force_non_convergence: For testing - forces residual to stay high
        
    Returns:
        NonlinearResult with displacement, stress, and convergence info
    """
    ny, nx = density.shape
    
    # Create mesh if not provided
    if mesh is None:
        mesh = create_mesh(nx, ny)
    
    # Material stiffness matrix (plane stress)
    E = problem.material.E
    nu = problem.material.nu
    Emin = 1e-9  # Minimum stiffness for void
    
    D = (E / (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ])
    
    n_nodes = mesh.n_nodes
    n_dofs = 2 * n_nodes
    
    # Initialize displacement
    u = np.zeros(n_dofs)
    
    # Setup timeout if requested
    if timeout is not None:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
    
    try:
        # Get free DOFs (apply BCs) - use Problem's methods
        fixed_dofs_arr = problem.get_fixed_dofs()
        fixed_dofs = set(fixed_dofs_arr)
        
        all_dofs = set(range(n_dofs))
        free_dofs = np.array(sorted(all_dofs - fixed_dofs))
        
        if len(free_dofs) == 0:
            return NonlinearResult(
                displacement=np.zeros((2, ny+1, nx+1)),
                stress=np.zeros((ny, nx)),
                converged=False,
                nr_iterations=0,
                final_residual=0.0,
                failure_reason="All DOFs fixed"
            )
        
        # Build global force vector - use Problem's method
        F = problem.get_force_vector()
        
        # Gauss points
        gp, gw = gauss_points_2x2()
        
        # Newton-Raphson iterations
        for nr_iter in range(max_iter):
            # Assemble tangent stiffness and internal force
            K_T = lil_matrix((n_dofs, n_dofs))
            F_int = np.zeros(n_dofs)
            
            for ey in range(ny):
                for ex in range(nx):
                    elem_idx = ey * nx + ex
                    
                    # Element properties
                    rho = density[ey, ex]
                    E_elem = Emin + rho**3 * (E - Emin)  # SIMP penalization
                    D_elem = (E_elem / E) * D
                    
                    # Node indices (counter-clockwise from bottom-left)
                    n0 = ey * (nx + 1) + ex
                    n1 = n0 + 1
                    n2 = n1 + (nx + 1)
                    n3 = n0 + (nx + 1)
                    nodes = [n0, n1, n2, n3]
                    
                    # Element DOFs
                    elem_dofs = []
                    for n in nodes:
                        elem_dofs.extend([2*n, 2*n + 1])
                    
                    # Element displacement
                    u_elem = u[elem_dofs]
                    
                    # Use UNIT SQUARE element (same as linear solver)
                    # This ensures consistency with the linear FEA
                    # Unit square: corners at (0,0), (1,0), (1,1), (0,1)
                    # J = [[0.5, 0], [0, 0.5]], detJ = 0.25, J^-1 = 2*I
                    
                    # Element stiffness and internal force
                    Ke = np.zeros((8, 8))
                    Fe_int = np.zeros(8)
                    
                    for gp_idx, (xi, eta) in enumerate(gp):
                        N, dN = shape_functions_and_derivatives(xi, eta)
                        
                        # For unit square element: J^-1 = 2*I, detJ = 0.25
                        dN_dx = 2.0 * dN  # dN_dx = J^-1 @ dN.T, but J^-1 = 2*I
                        detJ = 0.25
                        
                        # B-matrices
                        B_L, B_NL = compute_B_nonlinear(dN_dx, u_elem)
                        B = B_L + 0.5 * B_NL  # Total B for Green-Lagrange
                        
                        # Strain (Green-Lagrange)
                        strain = B @ u_elem
                        
                        # Stress (2nd Piola-Kirchhoff)
                        stress = D_elem @ strain
                        
                        # Material stiffness
                        Ke += gw[gp_idx] * (B_L + B_NL).T @ D_elem @ (B_L + B_NL) * detJ
                        
                        # Geometric stiffness
                        Ke += compute_geometric_stiffness_matrix(dN_dx, stress, detJ) * gw[gp_idx]
                        
                        # Internal force
                        Fe_int += gw[gp_idx] * (B_L + B_NL).T @ stress * detJ
                    
                    # Assemble
                    for i, di in enumerate(elem_dofs):
                        F_int[di] += Fe_int[i]
                        for j, dj in enumerate(elem_dofs):
                            K_T[di, dj] += Ke[i, j]
            
            # Residual
            R = F - F_int
            R_free = R[free_dofs]
            
            # For testing timeout: force non-convergence
            if force_non_convergence:
                R_free = np.ones_like(R_free) * 100.0
            
            residual_norm = np.linalg.norm(R_free)
            
            # Check convergence
            if residual_norm < tol:
                if timeout is not None:
                    signal.alarm(0)  # Cancel timeout
                
                # Compute final stress field
                stress_field = _compute_stress_field(density, u, mesh, D, E, Emin)
                
                # Reshape displacement
                u_reshaped = np.zeros((2, ny+1, nx+1))
                for i in range(n_nodes):
                    row = i // (nx + 1)
                    col = i % (nx + 1)
                    u_reshaped[0, row, col] = u[2*i]
                    u_reshaped[1, row, col] = u[2*i + 1]
                
                return NonlinearResult(
                    displacement=u_reshaped,
                    stress=stress_field,
                    converged=True,
                    nr_iterations=nr_iter + 1,
                    final_residual=residual_norm
                )
            
            # Solve for displacement increment
            K_T_csr = csr_matrix(K_T)
            K_free = K_T_csr[np.ix_(free_dofs, free_dofs)]
            
            try:
                du_free = spsolve(K_free, R_free)
            except Exception as e:
                if timeout is not None:
                    signal.alarm(0)
                return NonlinearResult(
                    displacement=np.zeros((2, ny+1, nx+1)),
                    stress=np.zeros((ny, nx)),
                    converged=False,
                    nr_iterations=nr_iter + 1,
                    final_residual=residual_norm,
                    failure_reason=f"Linear solve failed: {str(e)}"
                )
            
            # Line search for robustness
            alpha = _line_search(
                u, du_free, free_dofs, F, density, mesh, D, E, Emin, 
                nx, ny, residual_norm
            )
            
            # Update displacement
            u[free_dofs] += alpha * du_free
        
        # Max iterations reached
        if timeout is not None:
            signal.alarm(0)
        
        stress_field = _compute_stress_field(density, u, mesh, D, E, Emin)
        u_reshaped = np.zeros((2, ny+1, nx+1))
        for i in range(n_nodes):
            row = i // (nx + 1)
            col = i % (nx + 1)
            u_reshaped[0, row, col] = u[2*i]
            u_reshaped[1, row, col] = u[2*i + 1]
        
        return NonlinearResult(
            displacement=u_reshaped,
            stress=stress_field,
            converged=False,
            nr_iterations=max_iter,
            final_residual=residual_norm,
            failure_reason="Max iterations reached"
        )
    
    except TimeoutError as e:
        signal.alarm(0)
        return NonlinearResult(
            displacement=np.zeros((2, ny+1, nx+1)),
            stress=np.zeros((ny, nx)),
            converged=False,
            nr_iterations=nr_iter if 'nr_iter' in dir() else 0,
            final_residual=float('inf'),
            failure_reason=str(e)
        )
    finally:
        # Always cancel alarm
        if timeout is not None:
            signal.alarm(0)


def _get_node_indices(region: str, nx: int, ny: int, mesh: Mesh2D) -> list:
    """Get node indices for a boundary region."""
    indices = []
    
    if region == 'left':
        for j in range(ny + 1):
            indices.append(j * (nx + 1))
    elif region == 'right':
        for j in range(ny + 1):
            indices.append(j * (nx + 1) + nx)
    elif region == 'bottom':
        for i in range(nx + 1):
            indices.append(i)
    elif region == 'top':
        for i in range(nx + 1):
            indices.append(ny * (nx + 1) + i)
    elif region == 'bottom_left':
        indices.append(0)
    elif region == 'bottom_right':
        indices.append(nx)
    elif region == 'top_left':
        indices.append(ny * (nx + 1))
    elif region == 'top_right':
        indices.append(ny * (nx + 1) + nx)
    elif region.startswith('point_'):
        # Format: point_x_y
        parts = region.split('_')
        px, py = float(parts[1]), float(parts[2])
        # Find nearest node
        dists = np.sqrt((mesh.nodes[:, 0] - px)**2 + 
                        (mesh.nodes[:, 1] - py)**2)
        indices.append(np.argmin(dists))
    
    return indices


def _line_search(
    u: np.ndarray,
    du: np.ndarray,
    free_dofs: np.ndarray,
    F: np.ndarray,
    density: np.ndarray,
    mesh: Mesh2D,
    D: np.ndarray,
    E: float,
    Emin: float,
    nx: int,
    ny: int,
    initial_residual: float
) -> float:
    """
    Simple backtracking line search.
    
    Returns step size alpha in [alpha_min, 1.0]
    """
    alpha = 1.0
    
    for _ in range(LINE_SEARCH_MAX_ITER):
        u_trial = u.copy()
        u_trial[free_dofs] += alpha * du
        
        # Compute internal force with trial displacement
        F_int_trial = _compute_internal_force(u_trial, density, mesh, D, E, Emin, nx, ny)
        R_trial = F - F_int_trial
        
        residual_trial = np.linalg.norm(R_trial[free_dofs])
        
        # Accept if residual decreased or alpha is minimum
        if residual_trial < initial_residual or alpha <= LINE_SEARCH_ALPHA_MIN:
            return alpha
        
        alpha *= 0.5
    
    return alpha


def _compute_internal_force(
    u: np.ndarray,
    density: np.ndarray,
    mesh: Mesh2D,
    D: np.ndarray,
    E: float,
    Emin: float,
    nx: int,
    ny: int
) -> np.ndarray:
    """Compute internal force vector for given displacement."""
    n_dofs = 2 * mesh.n_nodes
    F_int = np.zeros(n_dofs)
    
    gp, gw = gauss_points_2x2()
    
    for ey in range(ny):
        for ex in range(nx):
            rho = density[ey, ex]
            E_elem = Emin + rho**3 * (E - Emin)
            D_elem = (E_elem / E) * D
            
            n0 = ey * (nx + 1) + ex
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            nodes = [n0, n1, n2, n3]
            
            elem_dofs = []
            for n in nodes:
                elem_dofs.extend([2*n, 2*n + 1])
            
            u_elem = u[elem_dofs]
            
            # Use UNIT SQUARE element (same as linear solver)
            # Unit square: J^-1 = 2*I, detJ = 0.25
            
            for gp_idx, (xi, eta) in enumerate(gp):
                N, dN = shape_functions_and_derivatives(xi, eta)
                
                # For unit square element
                dN_dx = 2.0 * dN
                detJ = 0.25
                
                B_L, B_NL = compute_B_nonlinear(dN_dx, u_elem)
                B = B_L + 0.5 * B_NL
                
                strain = B @ u_elem
                stress = D_elem @ strain
                
                Fe_int = gw[gp_idx] * (B_L + B_NL).T @ stress * detJ
                
                for i, di in enumerate(elem_dofs):
                    F_int[di] += Fe_int[i]
    
    return F_int


def _compute_stress_field(
    density: np.ndarray,
    u: np.ndarray,
    mesh: Mesh2D,
    D: np.ndarray,
    E: float,
    Emin: float
) -> np.ndarray:
    """Compute von Mises stress field."""
    ny, nx = density.shape
    stress_field = np.zeros((ny, nx))
    
    gp, gw = gauss_points_2x2()
    
    for ey in range(ny):
        for ex in range(nx):
            rho = density[ey, ex]
            E_elem = Emin + rho**3 * (E - Emin)
            D_elem = (E_elem / E) * D
            
            n0 = ey * (nx + 1) + ex
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            nodes = [n0, n1, n2, n3]
            
            elem_dofs = []
            for n in nodes:
                elem_dofs.extend([2*n, 2*n + 1])
            
            u_elem = u[elem_dofs]
            
            # Use UNIT SQUARE element (same as linear solver)
            # Unit square: J^-1 = 2*I, detJ = 0.25
            
            # Average stress at element center
            stress_sum = np.zeros(3)
            weight_sum = 0
            
            for gp_idx, (xi, eta) in enumerate(gp):
                N, dN = shape_functions_and_derivatives(xi, eta)
                
                # For unit square element
                dN_dx = 2.0 * dN
                detJ = 0.25
                
                B_L, B_NL = compute_B_nonlinear(dN_dx, u_elem)
                B = B_L + 0.5 * B_NL
                
                strain = B @ u_elem
                stress = D_elem @ strain
                
                stress_sum += stress * gw[gp_idx]
                weight_sum += gw[gp_idx]
            
            if weight_sum > 0:
                avg_stress = stress_sum / weight_sum
                # Von Mises
                vm = np.sqrt(avg_stress[0]**2 - avg_stress[0]*avg_stress[1] + 
                            avg_stress[1]**2 + 3*avg_stress[2]**2)
                stress_field[ey, ex] = vm
    
    return stress_field


# ============================================================================
# Convenience wrapper with timeout
# ============================================================================

def solve_nonlinear_with_timeout(
    problem: Problem,
    density: np.ndarray,
    timeout_seconds: int = TIMEOUT_SECONDS,
    **kwargs
) -> NonlinearResult:
    """
    Wrapper for newton_raphson with timeout handling.
    
    Safe for use in production - catches TimeoutError and returns
    a failed result instead of crashing.
    """
    return newton_raphson(
        problem=problem,
        density=density,
        timeout=timeout_seconds,
        **kwargs
    )
