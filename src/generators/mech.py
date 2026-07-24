"""
SIMP topology optimization for compliant mechanisms.

Displacement maximization using MMA or OC update.
Key difference from stiff: maximize output displacement, not minimize compliance.
"""

import numpy as np
from scipy.ndimage import convolve, zoom
from scipy.sparse import csr_matrix, lil_matrix
from dataclasses import dataclass, field, replace
from typing import Tuple, Optional, Callable, List, Dict
import time
import os

from ..core.problem import Problem, ProblemType, Material, BoundaryCondition, Load
from ..core.mesh import Mesh2D, create_mesh
from ..solvers.linear import solve_fea, FEAResult, assemble_stiffness_matrix
from ..validation.connectivity import validate_sample

# Reuse utilities from stiff generator
from .stiff import (
    OptimizationConfig, OptimizationResult, PhysicsFields,
    create_density_filter, apply_density_filter, apply_sensitivity_filter,
    apply_heaviside_projection, heaviside_derivative,
    upscale_density, create_problem_at_resolution, compute_physics_fields
)

from scipy.optimize import brentq


def apply_heaviside_volume_preserving(
    density: np.ndarray,
    beta: float,
    target_vf: float,
    domain_mask: np.ndarray = None
) -> Tuple[np.ndarray, float]:
    """
    Apply Heaviside projection with eta chosen to preserve target volume.
    
    Standard Heaviside with eta=0.5 destroys volume fraction when most densities
    are below 0.5 (which happens at low VF like 0.25). This function uses bisection
    to find eta such that: mean(H(density, beta, eta)) = target_vf
    
    Reference: Wang et al. (2011), "Projection methods in topology optimization"
    
    Args:
        density: (nely, nelx) filtered densities
        beta: Sharpness parameter
        target_vf: Target volume fraction to preserve
        domain_mask: Optional mask for active design region
        
    Returns:
        projected: (nely, nelx) projected densities
        eta_opt: The optimized threshold value used
    """
    if domain_mask is None:
        domain_mask = np.ones_like(density, dtype=bool)
    
    rho_min = 1e-6
    
    def compute_volume_error(eta: float) -> float:
        """Compute VF error for given eta."""
        numerator = np.tanh(beta * eta) + np.tanh(beta * (density - eta))
        denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
        projected = np.clip(numerator / denominator, rho_min, 1.0)
        current_vf = np.mean(projected[domain_mask])
        return current_vf - target_vf
    
    # Find eta via bisection in (0.01, 0.99)
    try:
        eta_opt = brentq(compute_volume_error, 0.01, 0.99, xtol=1e-4)
    except ValueError:
        # If no root in range, use default
        eta_opt = 0.5
    
    # Apply with optimized eta
    numerator = np.tanh(beta * eta_opt) + np.tanh(beta * (density - eta_opt))
    denominator = np.tanh(beta * eta_opt) + np.tanh(beta * (1 - eta_opt))
    projected = np.clip(numerator / denominator, rho_min, 1.0)
    
    return projected, eta_opt


def heaviside_derivative_with_eta(
    density: np.ndarray,
    beta: float,
    eta: float
) -> np.ndarray:
    """
    Derivative of Heaviside projection for chain rule with custom eta.
    
    dH/dx = β * (1 - tanh²(β*(x-η))) / (tanh(β*η) + tanh(β*(1-η)))
    
    Args:
        density: (nely, nelx) filtered densities (before projection)
        beta: Sharpness parameter
        eta: Threshold value (from volume-preserving optimization)
        
    Returns:
        dH: (nely, nelx) Heaviside derivative values
    """
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    # Avoid division by tiny denominator at extreme beta
    denominator = np.maximum(denominator, 1e-10)
    
    # dH/dx = beta * sech²(beta*(x - eta)) / denominator
    tanh_term = np.tanh(beta * (density - eta))
    sech_squared = 1 - tanh_term ** 2

    return beta * sech_squared / denominator


def heaviside_project(
    density: np.ndarray,
    beta: float,
    eta: float,
    rho_min: float = 1e-6
) -> np.ndarray:
    """
    Fixed-threshold tanh Heaviside projection (no volume preservation).

    Used by the robust three-field formulation to build an eroded realization at
    a threshold offset above the nominal eta. A higher eta requires a higher
    filtered density to project to solid, producing a *thinner* (eroded) design.

    Args:
        density: (nely, nelx) filtered densities.
        beta: Sharpness parameter.
        eta: Fixed threshold.
        rho_min: Lower clip for the projected density.

    Returns:
        projected: (nely, nelx) projected densities.
    """
    num = np.tanh(beta * eta) + np.tanh(beta * (density - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    if den < 1e-10:
        den = 1e-10
    return np.clip(num / den, rho_min, 1.0)


@dataclass
class MechConfig(OptimizationConfig):
    """Configuration for mechanism optimization.
    
    Tuned for compliant mechanisms with thin flexure hinges.
    
    Key design choices:
    - alpha continuation: starts low (0.05) to discover topology,
      ramps to alpha_max (0.5) to stiffen and clean up.
    - penal continuation: standard 1→3 ramp.
    - filter_radius 1.5: thin flexure hinges (~3-element min feature).
    - beta ramp for Heaviside crispening.
    """
    # Mechanism-specific springs
    k_in: float = 0.01           # Input spring - soft actuator
    k_out: float = None          # Output spring - randomized per generator
    
    # Volume fraction
    volume_fraction: float = 0.30
    max_iterations: int = 400
    
    # Heaviside for crisp designs
    beta_max: float = 32.0
    beta_interval: int = 40
    beta_step: float = 2.0
    
    # Penalization — standard continuation 1 → 3
    penal_init: float = 1.0
    penal_interval: int = 25
    
    # Filter radius sets the minimum length scale (and prevents single-pixel
    # hinges). Must scale with resolution: ~res/25.6 (=> 2.5 @64, 5.0 @128).
    # The old 1.5 was below the hinge scale and produced point-flexure artifacts
    # (audit 2026-06-15). Production generators should set this from resolution.
    filter_radius: float = 2.5
    
    # SE-penalty weight (α in MSE - α·SE)
    # Uses continuation: alpha_init → alpha_max over alpha_ramp_iters.
    # Low initial α lets MSE dominate for topology discovery.
    # High final α stiffens structure and removes fragile connections.
    alpha_init: float = 0.05    # Low — let mechanism form first
    alpha_max: float = 0.5      # High — stiffen structure
    alpha_ramp_iters: int = 150 # Iterations to reach alpha_max
    alpha_max: float = 0.3      # Moderate — don't kill mechanism
    alpha_ramp_iters: int = 100 # Linear ramp duration
    
    # Legacy alias — set to initial value
    alpha: float = 0.05
    
    # Connectivity enforcement interval (0 = disabled)
    connectivity_interval: int = 0

    # Robust three-field formulation — eliminates single-node hinges by also
    # optimizing an eroded realization (Wang/Sigmund/Lazarov 2011). The eroded
    # design removes thin necks/point flexures, so requiring it to perform forces
    # finite-width hinges and a real minimum length scale. ON by default: validated
    # 2026-06-15 to convert hinge-artifact "gibberish" into real mechanisms
    # (amplifier 0%->100% hinge-free @128). Costs a 2nd FEA solve per iteration.
    robust: bool = True
    robust_eta_offset: float = 0.05   # eroded threshold = eta_nominal + offset
    #   tuned: 0.05 kills single-pixel hinges without over-eroding thin flexures
    #   (0.20 collapses mechanisms into rigid blobs; see audit 2026-06-15)
    robust_mode: str = "worst"        # "worst": min-max over {nominal, eroded} (best
    #   in audit 2026-06-15); "eroded": optimize eroded design consistently (collapses
    #   u_out at larger offsets, lower hinge-free yield -- kept for reference)

    # Multi-resolution warm-start (speedup). Phase 1 establishes the gross
    # topology cheaply on a coarse mesh (an FEA solve at 64 is ~8x cheaper than
    # at 128); phase 2 upscales and refines on the target mesh with the robust
    # formulation, which is what actually enforces the hinge-free min length
    # scale. Coarse solves carry the topology, the fine robust phase carries the
    # quality -- so this should preserve yield/GA while cutting wall time ~2x.
    # OFF by default until validated against single-res 128 (quality gate).
    multires: bool = False
    coarse_resolution: int = 64        # mesh for phase 1 (must divide target res)
    coarse_fraction: float = 0.6       # fraction of max_iterations spent coarse
    fine_warm_beta_frac: float = 0.25  # fine phase starts at beta_max*this (warm
    #   but not frozen, so the robust refine can still widen coarse thin necks)


@dataclass
class MechProblem:
    """
    Mechanism problem definition.
    
    Extends base Problem with input/output port definitions.
    Supports arbitrary angle directions via unit vectors.
    """
    base_problem: Problem
    input_node: int           # Node where input force is applied
    input_direction: tuple    # (dx, dy) unit vector OR int (0=x, 1=y for backwards compat)
    output_node: int          # Node where output displacement is measured
    output_direction: tuple   # (dx, dy) unit vector OR int (0=x, 1=y for backwards compat)
    k_in: float = 0.001       # Input spring stiffness
    k_out: float = 0.1        # Output spring stiffness
    k_perp: float = None      # Perpendicular spring at output (suppresses rotation)
    symmetry: str = None      # None | 'horizontal' | 'vertical': enforce mirror
                              # symmetry of the design about the domain centerline
                              # each iteration (keeps inverters balanced scissors).
    seed_density: np.ndarray = None  # Optional kinematic seed (nely, nelx) used by
                              # optimize_mechanism when the caller passes no
                              # initial_density. Contract: struts at 1.0 on a
                              # background at the target VF (NOT zeros).

    def __post_init__(self):
        """Convert legacy int directions to unit vectors."""
        # Handle backwards compatibility with int directions
        if isinstance(self.input_direction, (int, np.integer)):
            self.input_direction = (1.0, 0.0) if self.input_direction == 0 else (0.0, 1.0)
        if isinstance(self.output_direction, (int, np.integer)):
            self.output_direction = (1.0, 0.0) if self.output_direction == 0 else (0.0, 1.0)
        # Ensure tuples
        self.input_direction = tuple(self.input_direction)
        self.output_direction = tuple(self.output_direction)
        # Default perpendicular spring: 5× k_out suppresses rotation at output
        if self.k_perp is None:
            self.k_perp = 5.0 * self.k_out
    
    @property
    def mesh(self):
        return self.base_problem.mesh
    
    @property
    def material(self):
        return self.base_problem.material
    
    @property
    def bcs(self):
        return self.base_problem.bcs
    
    @property
    def volume_fraction(self):
        return self.base_problem.volume_fraction
    
    @property
    def domain_mask(self):
        return self.base_problem.domain_mask
    
    def get_input_dof(self) -> int:
        """Get the primary DOF index for input (for spring attachment).
        
        For arbitrary angles, we attach springs to both X and Y DOFs,
        but this returns the X DOF for compatibility.
        """
        return 2 * self.input_node  # X DOF
    
    def get_output_dof(self) -> int:
        """Get the primary DOF index for output (for spring attachment).
        
        For arbitrary angles, we attach springs to both X and Y DOFs,
        but this returns the X DOF for compatibility.
        """
        return 2 * self.output_node  # X DOF
    
    def get_input_dofs(self) -> tuple:
        """Get both DOF indices for input node."""
        return (2 * self.input_node, 2 * self.input_node + 1)  # (X, Y)
    
    def get_output_dofs(self) -> tuple:
        """Get both DOF indices for output node."""
        return (2 * self.output_node, 2 * self.output_node + 1)  # (X, Y)


def add_springs_to_stiffness(
    K: csr_matrix, 
    mech_problem: MechProblem
) -> csr_matrix:
    """
    Add input/output spring stiffnesses to global K matrix.
    
    CRITICAL: If you forget this:
    - k_in=0 → matrix may become singular at input DOF
    - k_out=0 → mechanism generates no force on workpiece (trivial solution)
    
    Args:
        K: Global stiffness matrix (n_dofs × n_dofs)
        mech_problem: Mechanism problem with spring definitions
        
    Returns:
        K_modified: Stiffness matrix with springs added to diagonal
    """
    K_lil = K.tolil()  # Convert to LIL for efficient modification

    def add_directional_spring(dofs, stiffness, direction):
        """Add k * d d^T, not two independent axis springs.

        A scalar spring acting along a 45-degree direction has off-diagonal
        terms.  Omitting them makes it spuriously resist perpendicular motion
        and corrupts every oblique-port FEA label/selectivity measurement.
        """
        if stiffness is None or stiffness <= 0:
            return
        d = np.asarray(direction, dtype=float)
        nrm = np.linalg.norm(d)
        if nrm <= 1e-12:
            return
        d /= nrm
        block = float(stiffness) * np.outer(d, d)
        for i, di in enumerate(dofs):
            for j, dj in enumerate(dofs):
                K_lil[di, dj] += block[i, j]

    in_dofs = mech_problem.get_input_dofs()
    out_dofs = mech_problem.get_output_dofs()
    add_directional_spring(in_dofs, mech_problem.k_in, mech_problem.input_direction)
    add_directional_spring(out_dofs, mech_problem.k_out, mech_problem.output_direction)

    # Perpendicular output restraint is another directional spring.
    dx_out, dy_out = mech_problem.output_direction
    add_directional_spring(out_dofs, mech_problem.k_perp, (-dy_out, dx_out))
    
    return K_lil.tocsr()


def compute_mechanism_objective(
    mech_problem: MechProblem,
    displacement: np.ndarray
) -> float:
    """
    Compute mechanism objective: output displacement in desired direction.
    
    Uses dot product of displacement vector with output direction.
    We want to MAXIMIZE this (or minimize negative).
    
    Args:
        mech_problem: Mechanism problem definition
        displacement: Full displacement vector
        
    Returns:
        u_out: Output displacement projected onto desired direction
    """
    out_dof_x, out_dof_y = mech_problem.get_output_dofs()
    dx, dy = mech_problem.output_direction
    
    # Dot product: displacement · direction
    u_out = displacement[out_dof_x] * dx + displacement[out_dof_y] * dy
    return u_out


def compute_mechanism_sensitivity(
    mech_problem: MechProblem,
    density: np.ndarray,
    K: csr_matrix,
    u: np.ndarray,
    penal: float = 3.0,
    alpha: float = 0.5,
    factor=None,
    free_dofs=None,
) -> np.ndarray:
    """
    Compute sensitivity of mechanism objective w.r.t. element densities.
    
    Uses Sigmund's multi-objective formulation:
        Objective = MSE - α * SE_input
                  = u_out - α * (F^T @ u)
    
    Where:
        MSE = Mutual Strain Energy (output displacement)
        SE_input = Strain Energy at input (compliance)
        α = small weight to prevent disconnected structures
    
    This formulation:
        - Maximizes output displacement (MSE term)
        - Penalizes high input compliance (α*SE term)
        - Forces structure to be connected to fixed BCs!
    
    Args:
        mech_problem: Mechanism problem
        density: Element densities (nely, nelx)
        K: Global stiffness matrix (with springs)
        u: Displacement solution
        penal: SIMP penalization
        alpha: Weight for compliance penalty (default 0.05)
        factor: optional precomputed factorization of K_free (from the forward
            solve). If given, the adjoint reuses it instead of refactorizing K.
        free_dofs: free-DOF index array matching ``factor`` (required if factor given)

    Returns:
        dc: (nely, nelx) combined sensitivity
    """
    from ..solvers.linear import get_cached_edof, get_element_stiffness_cached

    mesh = mech_problem.mesh
    nelx, nely = mesh.nelx, mesh.nely
    material = mech_problem.material

    # Get output DOFs and direction
    out_dof_x, out_dof_y = mech_problem.get_output_dofs()
    dx, dy = mech_problem.output_direction

    # Create adjoint load vector: L = output_direction at output DOFs
    L = np.zeros(mesh.n_dofs)
    L[out_dof_x] = dx
    L[out_dof_y] = dy

    # Get free DOFs (reuse the forward solve's set when a factor is supplied)
    if factor is None or free_dofs is None:
        fixed_dofs = mech_problem.base_problem.get_fixed_dofs()
        all_dofs = np.arange(mesh.n_dofs)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    L_free = L[free_dofs]

    # Solve adjoint equation: K @ λ = L. K_free is SPD and identical to the
    # forward system, so reuse the forward factorization when available; this
    # skips a full refactorization per iteration with identical numerics.
    if factor is not None:
        lambda_free = factor.solve(L_free)
    else:
        from ..core.sparse_factor import factorize
        K_free = K[np.ix_(free_dofs, free_dofs)]
        lambda_free = factorize(K_free).solve(L_free)
    
    # Reconstruct full adjoint vector
    lambda_full = np.zeros(mesh.n_dofs)
    lambda_full[free_dofs] = lambda_free
    
    # Get cached element DOF indices and element stiffness
    edof = get_cached_edof(nelx, nely)  # (n_elem, 8)
    ke = get_element_stiffness_cached(material.E, material.nu)  # (8, 8)
    
    # Gather element displacements and adjoint: (n_elem, 8)
    u_e = u[edof]
    lambda_e = lambda_full[edof]
    
    # Compute λ^T K_e u for MSE sensitivity (all elements, vectorized)
    Ku = ke @ u_e.T  # (8, n_elem)
    lambda_K_u = np.sum(lambda_e.T * Ku, axis=0)  # (n_elem,)
    
    # Compute u^T K_e u for compliance sensitivity (all elements, vectorized)
    u_K_u = np.sum(u_e.T * Ku, axis=0)  # (n_elem,)
    
    # SIMP derivative: p * ρ^(p-1) * (E - E_min)
    rho = density.flatten()
    rho_safe = np.maximum(rho, 1e-12)
    E = material.E
    E_min = material.E_min
    dE_drho = penal * np.power(rho_safe, penal - 1) * (E - E_min)
    
    # MSE sensitivity: d(u_out)/dρ = -dE/dρ * λ^T K_e u
    # For maximization, we want positive gradient where material helps
    dc_mse = -dE_drho * lambda_K_u
    
    # Compliance sensitivity: d(SE)/dρ = -dE/dρ * u^T K_e u
    # This is always negative (adding material reduces compliance)
    dc_compliance = -dE_drho * u_K_u
    
    # Combined objective: maximize (MSE - α * Compliance)
    # Sensitivity: dc_mse - α * dc_compliance
    # Note: dc_compliance is negative, so -α * dc_compliance is positive
    # This ADDS to the gradient where compliance is high (disconnected regions)
    dc = dc_mse - alpha * dc_compliance
    
    return dc.reshape(nely, nelx)


def oc_update_mechanism(
    density: np.ndarray,
    dc: np.ndarray,
    volume_fraction: float,
    move_limit: float = 0.2,
    domain_mask: np.ndarray = None,
    kernel: np.ndarray = None,
    beta: float = 1.0,
    eta: float = 0.5
) -> np.ndarray:
    """
    Optimality Criteria update for mechanism displacement MAXIMIZATION.
    
    For maximization of (MSE - α·SE), convert to minimization:
        min -(MSE - α·SE)  → sensitivity_min = -dc
    
    Standard OC: B = sqrt(-dc_min / λ) = sqrt(dc / λ)
    
    Volume checked on RAW density (smooth, stable for bisection).
    The VP Heaviside in the main loop handles projected-density VF.
    """
    if domain_mask is None:
        domain_mask = np.ones_like(density, dtype=bool)
    
    nely, nelx = density.shape
    total_elements = np.sum(domain_mask)
    target_volume = volume_fraction * total_elements
    
    rho_min = 1e-4
    eps = 1e-10
    
    # Maximization OC: B = sqrt(dc / λ)
    # dc > 0 where material helps objective → dc/λ > 0 → B > 1 → grows
    # dc < 0 where material hurts objective → clamped to eps → shrinks
    
    l1, l2 = 0.0, 1e9
    
    while (l2 - l1) / (l1 + l2 + eps) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        
        B = np.sqrt(np.maximum(dc / (lmid + eps), eps))
        
        new_density = np.maximum(
            np.maximum(density - move_limit, rho_min),
            np.minimum(
                np.minimum(density + move_limit, 1.0),
                density * B
            )
        )
        
        new_density = np.where(domain_mask, new_density, rho_min)
        
        # Check volume on RAW density (smooth, no Heaviside discontinuity)
        # The VP Heaviside in the main loop handles projected-density VF.
        current_vol = np.sum(new_density[domain_mask])
        
        if current_vol > target_volume:
            l1 = lmid
        else:
            l2 = lmid
        
        if abs(l2 - l1) < 1e-6:
            break
    
    # Final update
    lmid = 0.5 * (l1 + l2)
    B = np.sqrt(np.maximum(dc / (lmid + eps), eps))
    new_density = np.maximum(
        np.maximum(density - move_limit, rho_min),
        np.minimum(
            np.minimum(density + move_limit, 1.0),
            density * B
        )
    )
    new_density = np.where(domain_mask, new_density, rho_min)
    
    return new_density


def _solve_mech_state(
    mech_problem: MechProblem,
    density: np.ndarray,
    penal: float = 3.0
):
    """Core mechanism forward solve, returning the reusable factorization.

    The reduced stiffness K_free is SPD, so its factor (CHOLMOD, or SuperLU as a
    fallback) is also exactly what the adjoint solve K_free λ = L needs. Returning
    it lets the per-iteration sensitivity reuse this factor instead of
    refactorizing the same matrix from scratch (see compute_mechanism_sensitivity).

    Returns:
        u: full displacement vector
        K: stiffness matrix (with springs)
        u_out: output displacement
        factor: factorization of K_free with a .solve(b) method (reuse for adjoint)
        free_dofs: index array the factor is defined on
    """
    from ..core.sparse_factor import factorize

    base_problem = mech_problem.base_problem
    mesh = base_problem.mesh

    # Assemble stiffness matrix using the base problem
    K = assemble_stiffness_matrix(base_problem, density, penal)

    # Add springs
    K = add_springs_to_stiffness(K, mech_problem)

    # Get force vector (input force in arbitrary direction)
    F = np.zeros(mesh.n_dofs)
    in_dof_x, in_dof_y = mech_problem.get_input_dofs()
    dx, dy = mech_problem.input_direction
    F[in_dof_x] = dx  # Force component in X
    F[in_dof_y] = dy  # Force component in Y

    # Apply BCs
    fixed_dofs = base_problem.get_fixed_dofs()
    all_dofs = np.arange(mesh.n_dofs)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Solve reduced system
    K_free = K[np.ix_(free_dofs, free_dofs)]
    F_free = F[free_dofs]

    factor = factorize(K_free)
    u_free = factor.solve(F_free)

    # Reconstruct full displacement
    u = np.zeros(mesh.n_dofs)
    u[free_dofs] = u_free

    # Get output displacement
    u_out = compute_mechanism_objective(mech_problem, u)

    return u, K, u_out, factor, free_dofs


def solve_mechanism_fea(
    mech_problem: MechProblem,
    density: np.ndarray,
    penal: float = 3.0
) -> Tuple[np.ndarray, csr_matrix, float]:
    """
    Solve FEA for mechanism problem with springs.

    Thin wrapper over _solve_mech_state for callers that don't need the factor.

    Args:
        mech_problem: Mechanism problem
        density: Element densities
        penal: SIMP penalization

    Returns:
        u: Displacement vector
        K: Stiffness matrix (with springs)
        u_out: Output displacement
    """
    u, K, u_out, _factor, _free = _solve_mech_state(mech_problem, density, penal)
    return u, K, u_out


def compute_mechanism_response_fields(
    mech_problem: MechProblem,
    density: np.ndarray,
    penal: float = 3.0,
) -> Tuple[PhysicsFields, np.ndarray, float]:
    """Solve one *canonical* mechanism verification realization.

    Unlike :func:`compute_mech_physics_fields`, this helper does not threshold,
    resample, or refine the supplied density.  It is for verification of an
    already canonicalized proposal, so the displacement, stress, strain energy,
    and reported output displacement all describe the exact same density and
    spring-aware mechanism BVP.

    The historical ``compute_mech_physics_fields`` deliberately produces a
    binary refined rendering for dataset generation.  Reusing it for proposal
    verification silently compared a continuous functional solve with a
    thresholded stress solve.  Keep the two roles explicit.
    """
    from src.solvers.linear import compute_von_mises_stress_vectorized

    rho = np.asarray(density, dtype=float)
    if rho.shape != (mech_problem.mesh.nely, mech_problem.mesh.nelx):
        raise ValueError(
            f"density shape {rho.shape} does not match mechanism mesh "
            f"{(mech_problem.mesh.nely, mech_problem.mesh.nelx)}")
    rho = np.where(mech_problem.domain_mask, np.clip(rho, 0.0, 1.0), 0.0)
    u, K, u_out = solve_mechanism_fea(mech_problem, rho, penal)
    nely, nelx = rho.shape
    displacement = np.stack([
        u[0::2].reshape(nely + 1, nelx + 1),
        u[1::2].reshape(nely + 1, nelx + 1),
    ])
    mat = mech_problem.material
    stress_vm = compute_von_mises_stress_vectorized(
        nelx, nely, u, rho, E=mat.E, nu=mat.nu, E_min=mat.E_min,
        penal=penal)
    fields = PhysicsFields(
        displacement=displacement,
        stress_vm=stress_vm,
        strain_energy=float(0.5 * u @ K @ u),
    )
    return fields, u, float(u_out)


def compute_mech_physics_fields(
    mech_problem: MechProblem,
    density: np.ndarray,
    refinement_factor: int = 2,
    penal: float = 3.0
) -> Tuple[PhysicsFields, np.ndarray]:
    """
    Compute physics fields for mechanism problem on refined mesh.
    
    CRITICAL: This uses the mechanism FEA solver with proper input force
    and springs. Do NOT use compute_physics_fields() which uses base_problem
    with no loads!
    
    Args:
        mech_problem: Mechanism problem with input/output nodes
        density: (H, W) optimized density at design resolution
        refinement_factor: Upscale factor (default 2)
        penal: SIMP penalization
    
    Returns:
        physics: PhysicsFields with displacement and stress
        density_fine: Upscaled density used for FEA
    """
    from scipy.sparse.linalg import splu
    
    nely, nelx = density.shape
    fine_res = nelx * refinement_factor
    
    # Upscale density to fine mesh
    density_fine = upscale_density(density, factor=refinement_factor)
    
    # Create refined mechanism problem
    fine_problem = create_mech_problem_at_resolution(mech_problem, fine_res)
    
    # Solve mechanism FEA with input force and springs
    mesh = fine_problem.base_problem.mesh
    K = assemble_stiffness_matrix(fine_problem.base_problem, density_fine, penal)
    K = add_springs_to_stiffness(K, fine_problem)
    
    # Apply input force in arbitrary direction
    F = np.zeros(mesh.n_dofs)
    in_dof_x, in_dof_y = fine_problem.get_input_dofs()
    dx, dy = fine_problem.input_direction
    F[in_dof_x] = dx  # Force component in X
    F[in_dof_y] = dy  # Force component in Y
    
    # Apply BCs
    fixed_dofs = fine_problem.base_problem.get_fixed_dofs()
    all_dofs = np.arange(mesh.n_dofs)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    
    K_free = K[np.ix_(free_dofs, free_dofs)]
    F_free = F[free_dofs]
    
    lu = splu(K_free.tocsc())
    u_free = lu.solve(F_free)
    
    # Reconstruct full displacement
    u = np.zeros(mesh.n_dofs)
    u[free_dofs] = u_free
    
    # Reshape to grid
    n_nodes_per_dim = fine_res + 1
    u_x = u[0::2].reshape(n_nodes_per_dim, n_nodes_per_dim)
    u_y = u[1::2].reshape(n_nodes_per_dim, n_nodes_per_dim)
    displacement = np.stack([u_x, u_y], axis=0)  # (2, H+1, W+1)
    
    # Compute stress (von Mises at element centers)
    stress_vm = compute_von_mises_stress(fine_problem.base_problem, u, density_fine, penal)
    
    # Compute strain energy
    strain_energy = 0.5 * u @ K @ u
    
    physics = PhysicsFields(
        displacement=displacement,
        stress_vm=stress_vm,
        strain_energy=strain_energy
    )
    
    return physics, density_fine


def create_mech_problem_at_resolution(mech_problem: MechProblem, new_res: int) -> MechProblem:
    """
    Create mechanism problem at a different resolution.
    
    Scales node indices and positions proportionally.
    """
    old_nelx = mech_problem.mesh.nelx
    scale = new_res / old_nelx
    
    # Create new base problem at fine resolution
    base_fine = create_problem_at_resolution(mech_problem.base_problem, new_res)
    
    # Scale node indices
    def scale_node(node: int) -> int:
        old_nx = old_nelx + 1
        new_nx = new_res + 1
        x = node % old_nx
        y = node // old_nx
        new_x = int(round(x * scale))
        new_y = int(round(y * scale))
        return new_y * new_nx + new_x
    
    return MechProblem(
        base_problem=base_fine,
        input_node=scale_node(mech_problem.input_node),
        input_direction=mech_problem.input_direction,
        output_node=scale_node(mech_problem.output_node),
        output_direction=mech_problem.output_direction,
        k_in=mech_problem.k_in,
        k_out=mech_problem.k_out,
        k_perp=mech_problem.k_perp
        # domain_mask is a property that delegates to base_problem.domain_mask
    )


def compute_von_mises_stress(
    problem: Problem,
    displacement: np.ndarray,
    density: np.ndarray,
    penal: float
) -> np.ndarray:
    """
    Compute von Mises stress at element centers.
    """
    # Keep one source of truth for the Q4 B matrix.  The previous copy divided
    # the correct unit-square matrix by 0.5 and therefore reported exactly 2x
    # the canonical stress.  Dataset stress arrays made before this correction
    # remain legacy artifacts; fresh verification uses this canonical path.
    from src.solvers.linear import compute_von_mises_stress_vectorized

    mesh = problem.mesh
    return compute_von_mises_stress_vectorized(
        mesh.nelx, mesh.nely, np.asarray(displacement), np.asarray(density),
        E=problem.material.E, nu=problem.material.nu,
        E_min=problem.material.E_min, penal=penal)


def remove_small_fragments(density: np.ndarray, min_frac: float = 0.03) -> np.ndarray:
    """
    Post-processing: remove small disconnected fragments from binary density.
    
    Standard practice in topology optimization to clean up tiny numerical
    artifacts that cause connectivity validation to fail unnecessarily.
    
    Elements in fragments smaller than min_frac of total solid area
    are zeroed out. This typically removes 1-50 element islands while
    preserving the main structure (1000+ elements).
    
    Args:
        density: (nely, nelx) density field
        min_frac: Minimum fraction of total solid area to keep (default 3%)
    
    Returns:
        Cleaned density field
    """
    from scipy.ndimage import label
    
    density = density.copy()
    binary = density > 0.5
    total_solid = np.sum(binary)
    
    if total_solid == 0:
        return density
    
    # Label connected components (4-connectivity = Von Neumann)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, n_comp = label(binary, structure=structure)
    
    if n_comp <= 1:
        return density  # Already connected or empty
    
    min_size = max(5, int(total_solid * min_frac))  # At least 5 elements
    
    for comp_id in range(1, n_comp + 1):
        comp_mask = labeled == comp_id
        comp_size = np.sum(comp_mask)
        if comp_size < min_size:
            density[comp_mask] = 0.0
    
    return density


def keep_connected_to_ground(
    density: np.ndarray,
    mech_problem: 'MechProblem',
    load_path_only: bool = False,
) -> np.ndarray:
    """
    Keep ONLY the connected component(s) that touch fixed BC nodes.

    This is the correct post-processing for mechanisms:
    - If fixed nodes, input, and output are all in one component → great
    - If there are islands NOT connected to ground → zero them out
    - This removes equal-sized disconnected blobs that remove_small_fragments misses

    Args:
        density: (nely, nelx) density field (after Heaviside projection)
        mech_problem: MechProblem with BC/input/output definitions
        load_path_only: if True, keep ONLY the single component carrying the
            input→output signal (when it is grounded). This prunes fragments that
            merely touch a *secondary* support patch but lie off the load path —
            the dominant failure of the two-support gripper, where such a fragment
            survives the "ground-touching" rule and trips the single-component
            gate. MUST only be used as a FINAL post-process: applying it during
            optimization would kill a legitimately-forming second support arm
            before it has connected to the main body. Default False = keep all
            ground-touching components (safe during optimization).

    Returns:
        Cleaned density with only ground-connected material
    """
    from scipy.ndimage import label

    density = density.copy()
    nelx = density.shape[1]
    nely = density.shape[0]

    binary = density > 0.5
    if np.sum(binary) == 0:
        return density

    # 4-connectivity (Von Neumann) to MATCH the validity gate's check_connectivity:
    # a corner-only (diagonal) touch shares a single Q4 node => a point hinge, which
    # the gate rejects. Keeping the component under the same rule the gate measures
    # means what we keep is exactly what counts as connected.
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    labeled, n_comp = label(binary, structure=structure)

    if n_comp <= 1:
        return density

    def node_component(node_idx):
        """Component label of a node (check the up-to-4 adjacent elements)."""
        nx_pos = node_idx % (nelx + 1)
        ny_pos = node_idx // (nelx + 1)
        for ey, ex in [(ny_pos, nx_pos), (ny_pos-1, nx_pos),
                       (ny_pos, nx_pos-1), (ny_pos-1, nx_pos-1)]:
            if 0 <= ey < nely and 0 <= ex < nelx:
                lbl = labeled[ey, ex]
                if lbl > 0:
                    return lbl
        return 0

    # Find which component(s) contain the fixed BC nodes
    ground_labels = set()
    for bc in mech_problem.base_problem.bcs:
        for node in bc.node_indices:
            lbl = node_component(node)
            if lbl > 0:
                ground_labels.add(lbl)

    # Preferred (final post-process only): keep ONLY the load-path component — the
    # one carrying the input→output signal — provided it is grounded. Off-path
    # fragments (e.g. material touching a secondary support patch) are pruned.
    if load_path_only:
        in_lbl = node_component(mech_problem.input_node)
        out_lbl = node_component(mech_problem.output_node)
        if in_lbl > 0 and in_lbl == out_lbl and in_lbl in ground_labels:
            density[labeled != in_lbl] = 0.0
            return density
        # else: design is already broken (input/output split or ungrounded) — fall
        # through to the ground-touching rule so the connectivity gate rejects it.

    if not ground_labels:
        # No BC nodes touch any solid element — keep largest component
        comp_sizes = [(labeled == i).sum() for i in range(1, n_comp + 1)]
        ground_labels = {int(np.argmax(comp_sizes)) + 1}

    # Zero out everything NOT in the ground-connected component(s)
    keep_mask = np.zeros_like(binary)
    for lbl in ground_labels:
        keep_mask |= (labeled == lbl)

    density[~keep_mask] = 0.0
    return density


def reproject_on_load_path(
    density_filtered: np.ndarray,
    density_projected: np.ndarray,
    mech_problem: 'MechProblem',
    beta: float,
    target_vf: float,
    domain_mask: np.ndarray,
    grow_radius: int,
) -> np.ndarray:
    """
    Concentrate the full target volume onto the connected load-path body.

    The volume-preserving Heaviside spreads ``target_vf`` across the WHOLE domain,
    including off-path blobs anchored at a secondary support. Pruning those blobs
    (keep_connected_to_ground load_path_only) then drops VF below the relative-5%
    volume gate, and any per-pixel regrow re-introduces single-pixel hinges. Instead:
    identify the load-path component, confine the (smooth) filtered field to a
    ``grow_radius`` neighborhood of it, and RE-PROJECT volume-preserving to target.
    Because this is a smooth Heaviside re-threshold (lower eta => members thicken
    along the filtered contours) rather than greedy boundary pixel addition, it lands
    the whole budget on the connected body at the target VF *without* creating
    articulation points.

    Returns the re-projected near-binary density (single load-path component).
    """
    from scipy.ndimage import binary_dilation

    # Load-path component of the current projection (4-connected, grounded, input==output).
    kept = keep_connected_to_ground(density_projected, mech_problem, load_path_only=True)
    kept_mask = kept > 0.5
    if not kept_mask.any():
        return density_projected  # nothing connected — let the gate reject it

    # If pruning the off-path material costs little volume, the design is already a
    # clean single-component body at target VF — keep the validated simple-prune
    # behavior (no re-projection), so single-component types (e.g. the symmetric
    # scissor inverter) are untouched. Only the two-support-style designs that shed
    # a meaningful off-path blob (>3% of solid) get the confined re-projection.
    proj_vol = int((density_projected > 0.5).sum())
    kept_vol = int(kept_mask.sum())
    if proj_vol == 0 or (proj_vol - kept_vol) / proj_vol < 0.03:
        return kept

    # Allow re-grown material only within a few-pixel neighborhood of the load path,
    # never into forced-void domain. Off-path blobs farther away are excluded.
    region = binary_dilation(kept_mask, iterations=max(1, int(grow_radius))) & domain_mask
    confined = np.where(region, density_filtered, 0.0)

    density_final, _ = apply_heaviside_volume_preserving(
        confined, beta, target_vf, domain_mask
    )
    density_final = np.where(domain_mask, density_final, 0.0)
    # Re-projection is confined to a connected neighborhood, so this is ~a no-op; it
    # only guards against a thin secondary split introduced by the new threshold.
    density_final = remove_small_fragments(density_final, min_frac=0.01)
    density_final = keep_connected_to_ground(density_final, mech_problem, load_path_only=True)
    return density_final


def check_mechanism_connectivity(
    density: np.ndarray,
    mech_problem: 'MechProblem'
) -> dict:
    """
    Check if fixed BCs, input node, and output node are all in the SAME
    connected component. This is the minimum requirement for a functional
    compliant mechanism.
    
    Returns:
        dict with 'connected' (bool), 'n_components' (int),
        'input_connected' (bool), 'output_connected' (bool)
    """
    from scipy.ndimage import label
    
    nelx = density.shape[1]
    nely = density.shape[0]
    
    binary = density > 0.5
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, n_comp = label(binary, structure=structure)
    
    def node_component(node_idx):
        """Find which component a node belongs to (check adjacent elements)."""
        nx_pos = node_idx % (nelx + 1)
        ny_pos = node_idx // (nelx + 1)
        for ey, ex in [(ny_pos, nx_pos), (ny_pos-1, nx_pos),
                       (ny_pos, nx_pos-1), (ny_pos-1, nx_pos-1)]:
            if 0 <= ey < nely and 0 <= ex < nelx:
                lbl = labeled[ey, ex]
                if lbl > 0:
                    return lbl
        return 0  # Not in any component
    
    # Ground component(s)
    ground_labels = set()
    for bc in mech_problem.base_problem.bcs:
        for node in bc.node_indices:
            lbl = node_component(node)
            if lbl > 0:
                ground_labels.add(lbl)
    
    input_label = node_component(mech_problem.input_node)
    output_label = node_component(mech_problem.output_node)
    
    input_connected = input_label in ground_labels and input_label > 0
    output_connected = output_label in ground_labels and output_label > 0
    # Strict: input and output must be in the SAME grounded component. Two
    # *different* grounded components (input in one, output in another) is NOT a
    # functional mechanism — the input motion never reaches the output.
    all_same = (input_label > 0 and input_label == output_label
                and input_label in ground_labels)
    
    return {
        'connected': all_same,
        'n_components': n_comp,
        'input_connected': input_connected,
        'output_connected': output_connected,
        'ground_labels': ground_labels,
        'input_label': input_label,
        'output_label': output_label,
    }


def seed_connecting_path(
    density: np.ndarray,
    from_node: int,
    to_node: int,
    nelx: int,
    nely: int,
    width: int = 2,
    value: float = 0.8
) -> np.ndarray:
    """
    Seed a connecting path between two nodes using Bresenham's line algorithm.
    
    This is CRITICAL for mechanism connectivity - ensures there's an initial
    path of material connecting the fixed BCs to input to output.
    
    Args:
        density: Density field to modify
        from_node: Starting node index
        to_node: Ending node index
        nelx, nely: Mesh dimensions
        width: Half-width of path in elements
        value: Density value for seeded path
        
    Returns:
        Modified density field
    """
    density = density.copy()
    
    # Convert nodes to coordinates
    x0 = from_node % (nelx + 1)
    y0 = from_node // (nelx + 1)
    x1 = to_node % (nelx + 1)
    y1 = to_node // (nelx + 1)
    
    # Bresenham's line algorithm
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    
    while True:
        # Seed elements in width around current point
        for dw in range(-width, width + 1):
            for dh in range(-width, width + 1):
                ex = x + dw
                ey = y + dh
                if 0 <= ex < nelx and 0 <= ey < nely:
                    density[ey, ex] = max(density[ey, ex], value)
        
        if x == x1 and y == y1:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return density


def _cluster_nodes(nodes: list, nelx: int, threshold: int) -> list:
    """
    Cluster node indices into spatial groups.
    
    Nodes within `threshold` distance of any node in a cluster are merged.
    Returns list of lists (each sublist is a cluster of node indices).
    """
    if not nodes:
        return []
    
    # Get (x,y) for each node
    coords = [(n % (nelx + 1), n // (nelx + 1)) for n in nodes]
    
    # Simple greedy clustering
    assigned = [False] * len(nodes)
    clusters = []
    
    for i in range(len(nodes)):
        if assigned[i]:
            continue
        # Start new cluster
        cluster = [nodes[i]]
        assigned[i] = True
        queue = [i]
        
        while queue:
            idx = queue.pop(0)
            cx, cy = coords[idx]
            for j in range(len(nodes)):
                if assigned[j]:
                    continue
                jx, jy = coords[j]
                dist = abs(cx - jx) + abs(cy - jy)  # Manhattan distance
                if dist <= threshold:
                    assigned[j] = True
                    cluster.append(nodes[j])
                    queue.append(j)
        
        clusters.append(cluster)
    
    return clusters


def seed_density_at_ports(
    density: np.ndarray,
    mech_problem: 'MechProblem',
    nelx: int,
    nely: int,
    seed_radius: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Seed initial density and create passive solid mask near key locations.
    
    CRITICAL for mechanism connectivity:
    - Fixed BC nodes need solid connection to structure
    - Input/output ports need solid material to transmit force
    
    Returns both seeded density AND a passive mask that locks these elements
    to solid during optimization.
    
    Args:
        density: Initial density field
        mech_problem: Mechanism problem with BC and port info
        nelx, nely: Mesh dimensions
        seed_radius: Radius (in elements) to seed around key nodes
    
    Returns:
        Tuple of (seeded density, passive_solid_mask)
    """
    density = density.copy()
    passive_solid = np.zeros((nely, nelx), dtype=bool)
    
    # Get fixed nodes from boundary conditions
    fixed_nodes = []
    for bc in mech_problem.base_problem.bcs:
        fixed_nodes.extend(bc.node_indices.tolist())
    
    # Also add input and output nodes
    key_nodes = fixed_nodes + [mech_problem.input_node, mech_problem.output_node]
    
    # Convert nodes to element coordinates and seed
    # Use smaller radius for PASSIVE (locked) elements, larger for seed
    # Larger passive radius prevents detachment at low VF
    passive_radius = 2  # Moderate: doesn't consume too much VF budget
    
    for node_idx in key_nodes:
        node_x = node_idx % (nelx + 1)
        node_y = node_idx // (nelx + 1)
        
        # Seed elements in a radius around this node
        for dy in range(-seed_radius, seed_radius + 1):
            for dx in range(-seed_radius, seed_radius + 1):
                # Check within radius (circular)
                dist_sq = dx*dx + dy*dy
                    
                # Element coordinates (node is at corner of element)
                ex = node_x + dx
                ey = node_y + dy
                
                # Clamp to valid range
                if 0 <= ex < nelx and 0 <= ey < nely:
                    if dist_sq <= passive_radius * passive_radius:
                        # Lock to solid (passive element)
                        passive_solid[ey, ex] = True
                        density[ey, ex] = 1.0
                    elif dist_sq <= seed_radius * seed_radius:
                        # Seed with higher density
                        density[ey, ex] = max(density[ey, ex], 0.8)
    
    return density, passive_solid


def port_approach_void_corridor(mech_problem, nelx, nely, passive_solid,
                                half_width: int = 2, start_offset: int = 1):
    """Lock a thin VOID corridor from each port to the domain boundary along its
    external approach axis, so the optimizer cannot bury the port.

    The port-accessibility gate (src.validation.ports) marches a ray from each
    port along -input_direction (input: the actuator approaches against its push)
    and +output_direction (output: the workpiece sits where work is delivered),
    and rejects the design if that ray hits material before leaving the domain.
    Forcing that exact line to void — as a passive element, re-enforced every
    iteration like passive_solid — guarantees the ray stays clear by
    construction. The solid port host (passive_solid) always wins, so the port
    itself is never voided. Opt-in via env MECH_PORT_VOID=1.
    """
    void = np.zeros((nely, nelx), dtype=bool)
    n = nelx + 1
    ports = [
        (mech_problem.input_node,
         (-mech_problem.input_direction[0], -mech_problem.input_direction[1])),
        (mech_problem.output_node,
         (mech_problem.output_direction[0], mech_problem.output_direction[1])),
    ]
    max_len = int((nelx * nelx + nely * nely) ** 0.5) + 2
    for node, d in ports:
        px, py = node % n, node // n
        dx, dy = float(d[0]), float(d[1])
        nrm = (dx * dx + dy * dy) ** 0.5
        if nrm < 1e-9:
            continue
        dx, dy = dx / nrm, dy / nrm
        ox, oy = -dy, dx  # perpendicular, for corridor width
        for s in range(start_offset, max_len):
            cx, cy = px + dx * s, py + dy * s
            for w in range(-half_width, half_width + 1):
                ex = int(round(cx + ox * w))
                ey = int(round(cy + oy * w))
                if 0 <= ex < nelx and 0 <= ey < nely:
                    void[ey, ex] = True
    void &= ~passive_solid  # never void the locked port host
    if getattr(mech_problem, "domain_mask", None) is not None:
        void &= mech_problem.domain_mask  # stay inside the design envelope
    return void


def _optimize_mechanism_multires(
    mech_problem: MechProblem,
    config: MechConfig,
    initial_density: np.ndarray = None,
    seed_mask: np.ndarray = None,
    callback: Callable = None
) -> OptimizationResult:
    """
    Two-phase multi-resolution warm-start for mechanism optimization.

    Phase 1 (coarse): establish the gross topology cheaply on a coarse mesh
      (FEA at 64 is ~8x cheaper than at 128). Robust is kept on so the coarse
      design already avoids the worst point-flexures.
    Phase 2 (fine): upscale the coarse density and refine on the target mesh
      with the robust formulation. This phase carries the physical-validity
      guarantee (5px min length scale, hinge-free). It is warm-started but NOT
      soft-locked, so it can still widen any thin neck the coarse phase left.

    Quality-first: this only changes WHERE iterations are spent, not the robust
    fine-mesh enforcement. Gated behind config.multires and validated by audit.
    """
    fine_res = mech_problem.mesh.nelx
    coarse_res = config.coarse_resolution
    factor = fine_res // coarse_res

    total_iters = config.max_iterations
    coarse_iters = max(1, int(round(total_iters * config.coarse_fraction)))
    fine_iters = max(1, total_iters - coarse_iters)

    # --- Phase 1: coarse ---
    coarse_problem = create_mech_problem_at_resolution(mech_problem, coarse_res)
    coarse_init = None
    if initial_density is not None:
        # Downscale the provided seed to the coarse mesh with plain nearest-
        # neighbor zoom. Do NOT binarize: the seeding contract puts the
        # background at the target VF (not zeros) so the optimizer can grow
        # material off-seed; thresholding at 0.5 silently zeroed it.
        coarse_init = zoom(initial_density, coarse_res / fine_res, order=0)
    coarse_config = replace(
        config,
        multires=False,
        max_iterations=coarse_iters,
        # Filter scales with resolution to keep the same physical length scale.
        filter_radius=max(2.0, config.filter_radius / factor),
    )
    res_coarse = optimize_mechanism(
        coarse_problem, coarse_config, initial_density=coarse_init
    )

    # --- Phase 2: fine refine, warm-started, no soft-lock ---
    init_fine = upscale_density(res_coarse.density, factor=factor)
    warm_beta = max(config.beta_init, config.beta_max * config.fine_warm_beta_frac)
    fine_config = replace(
        config,
        multires=False,
        max_iterations=fine_iters,
        beta_init=warm_beta,            # warm but not frozen
        penal_init=config.penal_max,    # topology already formed -> full SIMP
        alpha_init=config.alpha_max,    # stiffening regime from the start
        alpha_ramp_iters=1,
    )
    # Explicit all-False seed mask disables the seeded-mode soft lock while still
    # using init_fine as the warm start (passive ports are re-seeded internally).
    no_lock = np.zeros((fine_res, fine_res), dtype=bool)
    res_fine = optimize_mechanism(
        mech_problem, fine_config,
        initial_density=init_fine, seed_mask=no_lock, callback=callback
    )

    # Combine bookkeeping so callers see the full run.
    res_fine.n_iterations = res_coarse.n_iterations + res_fine.n_iterations
    res_fine.time_seconds = res_coarse.time_seconds + res_fine.time_seconds
    res_fine.convergence_history = (
        list(res_coarse.convergence_history) + list(res_fine.convergence_history)
    )
    return res_fine


def optimize_mechanism(
    mech_problem: MechProblem,
    config: MechConfig = None,
    initial_density: np.ndarray = None,
    seed_mask: np.ndarray = None,
    callback: Callable = None
) -> OptimizationResult:
    """
    Run SIMP topology optimization for displacement maximization.
    
    Supports two modes:
      1. Unseeded (legacy): uniform density → port seeding → Bresenham paths
      2. Seeded: initial_density from linkage/literature generator with:
         - Soft lock on seed elements (gradual release over 20 iterations)
         - Volume ramp (if seed VF > target, ramp down over 30 iterations)
         - Background at VF_target (caller must prepare this — NOT zeros!)
    
    Args:
        mech_problem: Mechanism problem definition
        config: Optimization configuration
        initial_density: Optional starting density (seeded mode if provided)
        seed_mask: Boolean (nely, nelx) marking seed elements for soft lock.
                   Auto-detected as (initial_density > 0.5) if not provided.
        callback: Optional callback(iter, density, u_out)
    
    Returns:
        OptimizationResult with final density and statistics
    """
    if config is None:
        config = MechConfig()

    # A generator may ship its own kinematic seed with the problem (e.g. the
    # amplifier's grounded rhombus — without it the optimizer converges to the
    # degenerate FLOATING translator and never connects ground). An explicit
    # initial_density from the caller still takes precedence.
    if initial_density is None:
        initial_density = getattr(mech_problem, 'seed_density', None)

    # Use the problem's volume fraction if config still has the default 0.30
    # (literature generators specify VF per-template)
    if config.volume_fraction == 0.30 and hasattr(mech_problem, 'volume_fraction'):
        prob_vf = mech_problem.volume_fraction
        if prob_vf is not None and prob_vf > 0 and prob_vf != 0.30:
            from dataclasses import fields as dc_fields
            field_dict = {f.name: getattr(config, f.name) for f in dc_fields(config)}
            field_dict['volume_fraction'] = prob_vf
            config = MechConfig(**field_dict)

    # Multi-resolution warm-start: delegate to the two-phase orchestrator when
    # enabled and the target mesh is a clean multiple of the coarse mesh. The
    # orchestrator calls back into optimize_mechanism with multires=False for
    # each phase, so there is no recursion.
    if getattr(config, 'multires', False):
        _fine_res = mech_problem.mesh.nelx
        _cres = config.coarse_resolution
        if _fine_res > _cres and _fine_res % _cres == 0:
            return _optimize_mechanism_multires(
                mech_problem, config, initial_density, seed_mask, callback
            )

    mesh = mech_problem.mesh
    nelx, nely = mesh.nelx, mesh.nely
    
    # Detect seeded mode
    seeded_mode = initial_density is not None
    
    # Initialize density
    if seeded_mode:
        density = initial_density.copy()
        
        # CRITICAL: Even in seeded mode, we need passive_solid elements at
        # BCs, input, and output to anchor the structure. Without these, the
        # optimizer can freely remove material at supports, breaking the load path.
        _, passive_solid = seed_density_at_ports(
            density, mech_problem, nelx, nely, seed_radius=3
        )
        # Enforce passive_solid on the density
        density[passive_solid] = 1.0
        
        # Auto-detect seed mask if not provided
        if seed_mask is None:
            seed_mask = density > 0.5
        
        # Compute initial seed VF for volume ramp
        if mech_problem.domain_mask is not None:
            initial_seed_vf = float(np.mean(density[mech_problem.domain_mask]))
        else:
            initial_seed_vf = float(np.mean(density))
    else:
        density = np.ones((nely, nelx)) * config.volume_fraction
        # Seed density near fixed nodes and ports to ensure connectivity
        # Creates passive_solid mask that LOCKS these elements to 1.0
        density, passive_solid = seed_density_at_ports(
            density, mech_problem, nelx, nely, seed_radius=3
        )
        # NOTE: No Bresenham connecting paths. Starting from uniform density
        # lets the optimizer discover the optimal topology freely, instead of
        # biasing toward direct straight-line connections which produce blobs.
    
    # Optional port-approach void corridor (opt-in via MECH_PORT_VOID=1): keeps
    # each port's actuation axis clear to the boundary so the optimizer cannot
    # bury it and fail the port-accessibility gate. Enforced like passive_solid.
    passive_void = None
    if os.environ.get("MECH_PORT_VOID", "") == "1":
        passive_void = port_approach_void_corridor(
            mech_problem, nelx, nely, passive_solid)
        density = np.where(passive_void, 0.0, density)

    # Apply domain mask
    if mech_problem.domain_mask is not None:
        density = np.where(mech_problem.domain_mask, density, 0.0)
    
    # Create filter kernel
    kernel = create_density_filter(nelx, nely, config.filter_radius)
    
    # Optimization loop
    penal = config.penal_init
    beta = config.beta_init
    history = []
    start_time = time.time()
    
    u_out_old = 0.0
    converged = False
    
    for iteration in range(config.max_iterations):

        # Apply density filter
        density_filtered = apply_density_filter(density, kernel)
        
        # Volume ramp for seeded mode: gradually tighten VF from seed_vf to target
        if seeded_mode and initial_seed_vf > config.volume_fraction:
            ramp_factor = max(0.0, 1.0 - iteration / 30.0)
            effective_vf = config.volume_fraction + (initial_seed_vf - config.volume_fraction) * ramp_factor
        else:
            effective_vf = config.volume_fraction
        
        # Apply VOLUME-PRESERVING Heaviside projection for the NOMINAL field.
        # Standard Heaviside destroys VF at low volume fractions; volume-preserving
        # eta keeps the saved design at target VF.
        domain_mask = mech_problem.domain_mask if mech_problem.domain_mask is not None else np.ones_like(density, dtype=bool)
        density_nom, eta_nom = apply_heaviside_volume_preserving(
            density_filtered, beta, effective_vf, domain_mask
        )
        # CRITICAL: Enforce passive solid elements (BC and port regions)
        density_nom = np.where(passive_solid, 1.0, density_nom)
        if passive_void is not None:
            density_nom = np.where(passive_void, 0.0, density_nom)
        u_nom, K_nom, uout_nom, fac_nom, free_nom = _solve_mech_state(
            mech_problem, density_nom, penal)

        if config.robust:
            # ERODED realization: higher threshold -> thinner design. A single-pixel
            # hinge disappears here, collapsing u_out, so optimizing the worst of
            # {nominal, eroded} forbids point flexures and enforces a min length scale.
            eta_ero = min(eta_nom + config.robust_eta_offset, 0.98)
            density_ero = heaviside_project(density_filtered, beta, eta_ero)
            density_ero = np.where(passive_solid, 1.0, density_ero)
            if passive_void is not None:
                density_ero = np.where(passive_void, 0.0, density_ero)
            u_ero, K_ero, uout_ero, fac_ero, free_ero = _solve_mech_state(
                mech_problem, density_ero, penal)
            if config.robust_mode == "eroded":
                # Optimize the eroded design consistently. It defines the minimum
                # length scale, so a performant hinge-free eroded design guarantees a
                # hinge-free (thicker) nominal. Avoids the worst-case oscillation where,
                # once the eroded design works, optimization flips to the nominal and a
                # fresh single-pixel hinge appears.
                use_ero = True
            else:  # "worst": min-max subgradient over {nominal, eroded}
                use_ero = uout_ero <= uout_nom
            if use_ero:
                density_projected, u, K, eta_used, u_out, factor, free_dofs = (
                    density_ero, u_ero, K_ero, eta_ero, uout_ero, fac_ero, free_ero)
            else:
                density_projected, u, K, eta_used, u_out, factor, free_dofs = (
                    density_nom, u_nom, K_nom, eta_nom, uout_nom, fac_nom, free_nom)
        else:
            density_projected, u, K, eta_used, u_out, factor, free_dofs = (
                density_nom, u_nom, K_nom, eta_nom, uout_nom, fac_nom, free_nom)

        history.append(abs(u_out))  # Track worst-case magnitude
        
        # α continuation: ramp from alpha_init to alpha_max
        alpha_init = getattr(config, 'alpha_init', config.alpha)
        alpha_max = getattr(config, 'alpha_max', config.alpha)
        alpha_ramp = getattr(config, 'alpha_ramp_iters', 150)
        current_alpha = alpha_init + (alpha_max - alpha_init) * min(1.0, iteration / alpha_ramp)
        
        # Compute sensitivity with current α (reuse the forward factorization
        # of the chosen realization for the adjoint solve)
        dc = compute_mechanism_sensitivity(
            mech_problem, density_projected, K, u, penal, alpha=current_alpha,
            factor=factor, free_dofs=free_dofs
        )
        
        # Zero out sensitivity for passive elements (they don't change)
        dc = np.where(passive_solid, 0.0, dc)
        if passive_void is not None:
            dc = np.where(passive_void, 0.0, dc)
        
        # Soft lock for seeded mode: dampen sensitivity on seed elements
        # for first 20 iterations to preserve seed topology
        if seeded_mode and seed_mask is not None and iteration < 20:
            lock_weight = max(0.0, 1.0 - iteration / 20.0)
            # Scale down sensitivity on seed elements (keep 20% even at max lock)
            dc = np.where(seed_mask, dc * (1.0 - lock_weight * 0.8), dc)
        
        # Chain rule for Heaviside (use the eta of the realization we optimized)
        dH = heaviside_derivative_with_eta(density_filtered, beta, eta_used)
        dc = dc * dH
        
        # Filter sensitivity
        dc = apply_sensitivity_filter(dc, density_filtered, kernel)
        
        # CONNECTIVITY-AWARE SENSITIVITY MODIFICATION
        # MSE-αSE sensitivity is negative (= "add material") even for disconnected
        # islands. This redirects material from islands to connected structure by
        # flipping sensitivity for disconnected solid elements to positive (= "remove").
        # Only activates once topology is crisp enough to check (beta >= 4).
        _conn_active = (config.connectivity_interval > 0 and 
                        beta >= 4.0 and
                        iteration % config.connectivity_interval == 0)
        _disconnected_mask = None
        if _conn_active:
            # Use the already-computed projected density to check connectivity
            d_connected = keep_connected_to_ground(density_projected, mech_problem)
            # Elements that are solid (>0.5) in projection but NOT connected to ground
            _disconnected_mask = (density_projected > 0.5) & (d_connected < 0.01)
            n_disc = np.sum(_disconnected_mask)
            if n_disc > 0:
                # Flip sensitivity to NEGATIVE → OC will push these toward void (for maximization problem)
                # Scale by max sensitivity magnitude for proportional effect
                dc_max = np.abs(dc).max()
                if dc_max > 1e-12:
                    # Strength ramps with beta: gentle at β=4, full at β=32
                    strength = min(1.0, beta / config.beta_max)
                    dc[_disconnected_mask] = -dc_max * strength
        
        # OC update for mechanism (use effective_vf from volume ramp)
        density = oc_update_mechanism(
            density, dc, effective_vf,
            config.move_limit, mech_problem.domain_mask
        )

        # Enforce mirror symmetry of the design about the domain centerline.
        # Averaging the design with its mirror each iteration forces a balanced
        # structure (e.g. a symmetric scissor inverter) and removes the
        # asymmetric-blob / off-axis-pivot local minima that the non-convex
        # optimizer otherwise falls into. Ports/supports must be symmetric too.
        sym = getattr(mech_problem, 'symmetry', None)
        if sym == 'horizontal':
            density = 0.5 * (density + density[::-1, :])
        elif sym == 'vertical':
            density = 0.5 * (density + density[:, ::-1])

        # Enforce passive elements in raw density too
        density = np.where(passive_solid, 1.0, density)
        if passive_void is not None:
            density = np.where(passive_void, 0.0, density)
        
        # Callback
        if callback is not None:
            callback(iteration, density_projected, u_out)
        
        # Continuation
        if (iteration + 1) % config.penal_interval == 0:
            penal = min(penal + config.penal_step, config.penal_max)
        
        if (iteration + 1) % config.beta_interval == 0:
            beta = min(beta * config.beta_step, config.beta_max)
        
        # Convergence check
        if iteration > 10:
            change = abs(abs(u_out) - abs(u_out_old)) / (abs(u_out_old) + 1e-10)
            if change < config.convergence_tol and penal >= config.penal_max and beta >= config.beta_max:
                converged = True
                break
        
        u_out_old = u_out
    
    # Final: filter + VOLUME-PRESERVING project
    density_filtered = apply_density_filter(density, kernel)
    domain_mask = mech_problem.domain_mask if mech_problem.domain_mask is not None else np.ones_like(density, dtype=bool)
    density_final, _ = apply_heaviside_volume_preserving(
        density_filtered, beta, config.volume_fraction, domain_mask
    )
    
    if mech_problem.domain_mask is not None:
        density_final = np.where(mech_problem.domain_mask, density_final, 0.0)
    
    # Post-processing: threshold to near-binary
    # Just remove tiny numerical artifacts < 1% of solid area.
    density_final = remove_small_fragments(density_final, min_frac=0.01)
    # Concentrate the full target volume on the connected load-path body. Plain
    # pruning of off-path blobs (fragments touching a secondary support but off the
    # input→output path — the two-support gripper failure) drops VF below the volume
    # gate, and any per-pixel regrow re-introduces single-pixel hinges. A confined
    # smooth Heaviside re-projection lands the whole budget on the connected body AT
    # the target VF without creating articulation points. grow_radius ~ filter radius
    # so members may thicken by about one feature width. For the SEEDED linkage tier
    # (slider_crank/four_bar) this also strips the junk fragments grown off the seed.
    density_final = reproject_on_load_path(
        density_filtered, density_final, mech_problem, beta,
        config.volume_fraction, domain_mask,
        grow_radius=max(2, int(round(config.filter_radius))),
    )

    elapsed = time.time() - start_time
    
    return OptimizationResult(
        density=density_final,
        compliance=-abs(history[-1]) if history else 0.0,  # Negative for "minimization"
        volume_fraction=float(np.mean(density_final[mech_problem.domain_mask] 
                                       if mech_problem.domain_mask is not None 
                                       else density_final)),
        n_iterations=len(history),
        converged=converged,
        convergence_history=history,
        time_seconds=elapsed
    )


# =============================================================================
# Problem Generators (Random Mechanism Configurations)
# =============================================================================

def generate_inverter_problem(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.20,
    k_in: float = None,
    k_out: float = None,
    seed: int = None
) -> MechProblem:
    """
    Generate a CANONICAL symmetric force inverter (Sigmund 1997 / "L1" template).

    Layout (for the 'L' orientation; the other 3 are rotations for diversity):
    - Fixed: the TWO corners on the INPUT edge (left-top and left-bottom),
      fully pinned. These give the symmetric reaction that turns a push into
      an inversion.
    - Input: center of the input (left) edge, pushing INTO the domain (+x).
    - Output: center of the OPPOSITE (right) edge, rewarded for moving in the
      OPPOSITE direction (-x) = inversion.

    Why this beats the old corner-pivot lever: the structure is symmetric about
    the input/output axis, so the optimizer builds a balanced scissor/double-arm
    truss instead of a one-armed lever that pivots on a single point. That gives
    higher geometric advantage and far fewer point-hinge artifacts (the old
    corner-lever was the audit laggard at ~38% hinge-free, GA ~0.2).

    k_in is kept soft (0.01) because the springs are coupled to geometry; the
    canonical k_in=1.0 over-stiffens the input and suppresses u_out here too.
    """
    if seed is not None:
        np.random.seed(seed)

    if k_in is None:
        k_in = 0.01
    if k_out is None:
        k_out = np.random.uniform(0.01, 0.05)

    mesh = create_mesh(nelx, nely)
    material = Material()

    def nid(x, y):
        return y * (nelx + 1) + x

    # Diversity comes from symmetry-PRESERVING knobs only: support-patch size,
    # port depth, orientation, and k_out. The ports stay exactly on the
    # centerline so the enforced mirror symmetry is consistent with the BCs.
    corner_size = max(3, nelx // int(np.random.randint(8, 16)))
    inset = max(2, nelx // int(np.random.randint(24, 48)))  # port depth into domain

    # Randomize which edge is the INPUT edge (4 orientations) for dataset
    # diversity while keeping the canonical symmetric structure.
    orient = np.random.choice(['L', 'R', 'B', 'T'])
    fixed_nodes = []
    if orient == 'L':            # input on LEFT, output on RIGHT
        for yy in range(corner_size + 1):                       # bottom-left corner
            for xx in range(corner_size + 1):
                fixed_nodes.append(nid(xx, yy))
        for yy in range(nely - corner_size, nely + 1):          # top-left corner
            for xx in range(corner_size + 1):
                fixed_nodes.append(nid(xx, yy))
        input_x, input_y = inset, nely // 2
        input_direction = (1.0, 0.0)
        output_x, output_y = nelx - inset, nely // 2
        output_direction = (-1.0, 0.0)
    elif orient == 'R':          # input on RIGHT, output on LEFT
        for yy in range(corner_size + 1):
            for xx in range(nelx - corner_size, nelx + 1):
                fixed_nodes.append(nid(xx, yy))
        for yy in range(nely - corner_size, nely + 1):
            for xx in range(nelx - corner_size, nelx + 1):
                fixed_nodes.append(nid(xx, yy))
        input_x, input_y = nelx - inset, nely // 2
        input_direction = (-1.0, 0.0)
        output_x, output_y = inset, nely // 2
        output_direction = (1.0, 0.0)
    elif orient == 'B':          # input on BOTTOM, output on TOP
        for xx in range(corner_size + 1):
            for yy in range(corner_size + 1):
                fixed_nodes.append(nid(xx, yy))
        for xx in range(nelx - corner_size, nelx + 1):
            for yy in range(corner_size + 1):
                fixed_nodes.append(nid(xx, yy))
        input_x, input_y = nelx // 2, inset
        input_direction = (0.0, 1.0)
        output_x, output_y = nelx // 2, nely - inset
        output_direction = (0.0, -1.0)
    else:                        # 'T' input on TOP, output on BOTTOM
        for xx in range(corner_size + 1):
            for yy in range(nely - corner_size, nely + 1):
                fixed_nodes.append(nid(xx, yy))
        for xx in range(nelx - corner_size, nelx + 1):
            for yy in range(nely - corner_size, nely + 1):
                fixed_nodes.append(nid(xx, yy))
        input_x, input_y = nelx // 2, nely - inset
        input_direction = (0.0, -1.0)
        output_x, output_y = nelx // 2, inset
        output_direction = (0.0, 1.0)

    # NO per-port jitter: input and output stay exactly on the centerline that
    # the supports are symmetric about. Independent port jitter (the old code)
    # tilted the load axis and was the root cause of asymmetric-blob inverters
    # with off-axis input and GA~0.2. Symmetry is enforced in the optimizer.
    symmetry = 'horizontal' if orient in ('L', 'R') else 'vertical'

    bc = BoundaryCondition(
        node_indices=np.array(fixed_nodes, dtype=np.int32),
        directions=np.full(len(fixed_nodes), 2)
    )

    base_problem = Problem(
        mesh=mesh,
        material=material,
        bcs=[bc],
        loads=[],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.MECHANISM
    )

    return MechProblem(
        base_problem=base_problem,
        input_node=nid(input_x, input_y),
        input_direction=input_direction,
        output_node=nid(output_x, output_y),
        output_direction=output_direction,
        k_in=k_in,
        k_out=k_out,
        symmetry=symmetry
    )

def generate_gripper_problem(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.20,
    k_in: float = None,
    k_out: float = None,
    seed: int = None
) -> MechProblem:
    """
    Single-jaw compliant GRIPPER (distributed compliance), DIVERSIFIED.

    Distributed-compliance core (rebuilt 2026-06-28): N>=2 support patches on a
    mount edge with the input pushing in BETWEEN them, and a single jaw on the
    opposite edge rewarded for moving toward the centerline (perpendicular to the
    input = closing). The gap between supports + the offset jaw forces a
    distributed lever, not the rigid single-pivot blob the old one-support lever
    produced (gini ~0.8, "not a mechanism").

    DIVERSITY (v2, 2026-06-28): a fixed BVP (square domain, 2 supports, VF 0.20)
    made every sample the SAME wedge archetype — "the same thing with different
    cutouts." Compliance (k_out/alpha) only modulates member thickness within
    that archetype, so real topological variety is driven from the GEOMETRY here:
      - domain SHAPE: carve an active rectangle (square / wide / tall) via
        domain_mask, changing the mechanism envelope and proportions;
      - SUPPORTS: 2 or 3 patches, spread along the mount edge at varied positions;
      - PORTS: input placed in the largest support gap; jaw position varied along
        the opposite edge.
    The output direction is still DERIVED (perpendicular-to-input, toward the
    active-region center) so every config stays kinematically consistent (avoids
    the u_out=0 collapse that independent output dirs caused on other types).
    k_out is also widened (compliance diversity is cheap on top of geometry).

    Mirror symmetry is deliberately NOT enforced (the jaw output is asymmetric;
    symmetry-averaging zeroes the jaw motion). k_in stays soft (0.01) — a stiff
    input suppresses u_out (springs are geometry-coupled).
    """
    if seed is not None:
        np.random.seed(seed)

    if k_in is None:
        k_in = 0.01
    if k_out is None:
        # Safe range: high k_out (stiff output) collapses u_out toward 0 (compliance
        # sweep: k_out 0.10 -> u_out 2). Topological diversity comes from GEOMETRY
        # below, not from k_out (which only modulates cutouts), so keep k_out modest.
        k_out = float(np.random.uniform(0.01, 0.04))

    mesh = create_mesh(nelx, nely)
    material = Material()

    def nid(x, y):
        return y * (nelx + 1) + x

    def jit(c, lo, hi, amt):
        return int(np.clip(c + np.random.randint(-amt, amt + 1), lo, hi))

    # ---- domain SHAPE: active rectangle [ex0,ex1] x [ey0,ey1] (element coords) --
    # 'wide' trims top+bottom, 'tall' trims left+right; margins keep the active
    # dimension in [~1/2, ~3/4] of the mesh (aspect ratios up to ~2:1).
    shape = np.random.choice(['square', 'wide', 'tall'], p=[0.4, 0.3, 0.3])
    ex0, ey0, ex1, ey1 = 0, 0, nelx, nely
    if shape == 'wide':
        ey0 = np.random.randint(nely // 8, nely // 4 + 1)
        ey1 = nely - np.random.randint(nely // 8, nely // 4 + 1)
    elif shape == 'tall':
        ex0 = np.random.randint(nelx // 8, nelx // 4 + 1)
        ex1 = nelx - np.random.randint(nelx // 8, nelx // 4 + 1)
    domain_mask = np.zeros((nely, nelx), dtype=bool)
    domain_mask[ey0:ey1, ex0:ex1] = True

    W, H = ex1 - ex0, ey1 - ey0
    cx, cy = (ex0 + ex1) // 2, (ey0 + ey1) // 2

    def patch(px, py, r):
        """BC node disk, clamped to the active rectangle."""
        pts = []
        for yy in range(max(ey0, py - r), min(ey1, py + r) + 1):
            for xx in range(max(ex0, px - r), min(ex1, px + r) + 1):
                if (xx - px) ** 2 + (yy - py) ** 2 <= (r + 0.5) ** 2:
                    pts.append(nid(xx, yy))
        return pts

    # TWO supports only: 3 anchor points over-constrain the body into a rigid
    # pivot (lumped compliance -> blob, gini ~0.9, u_out~0 — measured 2026-06-28).
    # The distributed lever needs exactly two spread anchors with the input
    # between them. Position/spread jitter below supplies the support diversity.
    sup_fracs = [np.random.uniform(0.18, 0.32), np.random.uniform(0.68, 0.82)]
    high_jaw = np.random.random() < 0.5
    # Mount on the domain's LONG edge so the supports spread far apart — close
    # supports let the body rigidly pivot (lumped compliance -> blob, u_out~0).
    # 'wide' (W>H) => horizontal mount (B/T); 'tall' => vertical mount (L/R);
    # 'square' => any of the four.
    if shape == 'wide':
        orient = np.random.choice(['B', 'T'])
    elif shape == 'tall':
        orient = np.random.choice(['L', 'R'])
    else:
        orient = np.random.choice(['L', 'R', 'B', 'T'])
    vertical_mount = orient in ('L', 'R')

    if vertical_mount:                       # supports + input on a vertical edge
        pr = max(2, H // int(np.random.randint(12, 20)))
        mount_x = ex0 if orient == 'L' else ex1
        jaw_x = ex1 if orient == 'L' else ex0
        input_direction = (1.0, 0.0) if orient == 'L' else (-1.0, 0.0)
        sup = sorted(jit(int(ey0 + f * H), ey0 + pr + 1, ey1 - pr - 1, max(1, H // 14))
                     for f in sup_fracs)
        fixed_nodes = [n for s in sup for n in patch(mount_x, s, pr)]
        # input in the LARGEST gap between consecutive supports
        gap, mid = max((sup[i + 1] - sup[i], (sup[i] + sup[i + 1]) // 2)
                       for i in range(len(sup) - 1))
        in_y = jit(mid, ey0 + 1, ey1 - 1, max(1, gap // 4))
        input_node = nid(mount_x, in_y)
        jaw_y = jit(ey1 - H // 8 if high_jaw else ey0 + H // 8, ey0 + 2, ey1 - 2, max(1, H // 14))
        output_node = nid(jaw_x, jaw_y)
        output_direction = (0.0, -1.0) if jaw_y > cy else (0.0, 1.0)
    else:                                    # supports + input on a horizontal edge
        pr = max(2, W // int(np.random.randint(12, 20)))
        mount_y = ey0 if orient == 'B' else ey1
        jaw_y = ey1 if orient == 'B' else ey0
        input_direction = (0.0, 1.0) if orient == 'B' else (0.0, -1.0)
        sup = sorted(jit(int(ex0 + f * W), ex0 + pr + 1, ex1 - pr - 1, max(1, W // 14))
                     for f in sup_fracs)
        fixed_nodes = [n for s in sup for n in patch(s, mount_y, pr)]
        gap, mid = max((sup[i + 1] - sup[i], (sup[i] + sup[i + 1]) // 2)
                       for i in range(len(sup) - 1))
        in_x = jit(mid, ex0 + 1, ex1 - 1, max(1, gap // 4))
        input_node = nid(in_x, mount_y)
        jaw_x = jit(ex1 - W // 8 if high_jaw else ex0 + W // 8, ex0 + 2, ex1 - 2, max(1, W // 14))
        output_node = nid(jaw_x, jaw_y)
        output_direction = (-1.0, 0.0) if jaw_x > cx else (1.0, 0.0)

    fixed_nodes = sorted(set(fixed_nodes))
    bc = BoundaryCondition(
        node_indices=np.array(fixed_nodes, dtype=np.int32),
        directions=np.full(len(fixed_nodes), 2)
    )

    base_problem = Problem(
        mesh=mesh, material=material, bcs=[bc], loads=[],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.MECHANISM,
        domain_mask=domain_mask,
    )

    return MechProblem(
        base_problem=base_problem,
        input_node=input_node, input_direction=input_direction,
        output_node=output_node, output_direction=output_direction,
        k_in=k_in, k_out=k_out,
    )

def get_domain_boundary_nodes(
    domain_mask: np.ndarray,
    nelx: int,
    nely: int
) -> Dict[str, List[int]]:
    """
    Find boundary nodes of a masked domain.
    
    For a non-square domain (e.g., L-shape), this finds nodes on the actual
    boundary of the masked region, not just the outer square edges.
    
    Args:
        domain_mask: (nely, nelx) boolean mask where True = active region
        nelx, nely: Mesh dimensions
        
    Returns:
        Dictionary with edge keys ('left', 'right', 'top', 'bottom', 'interior')
        mapping to lists of boundary node indices
    """
    from scipy.ndimage import binary_erosion
    
    # Find boundary elements (active elements adjacent to void or edge)
    interior = binary_erosion(domain_mask, structure=np.ones((3,3)))
    boundary_elements = domain_mask & ~interior
    
    # Classify nodes by their position
    boundary_nodes = {'left': [], 'right': [], 'top': [], 'bottom': [], 'corner': []}
    
    # For each boundary element, check which edge it touches
    for ely in range(nely):
        for elx in range(nelx):
            if not boundary_elements[ely, elx]:
                continue
                
            # Get nodes of this element
            n1 = ely * (nelx + 1) + elx           # bottom-left
            n2 = n1 + 1                            # bottom-right
            n3 = (ely + 1) * (nelx + 1) + elx + 1  # top-right
            n4 = (ely + 1) * (nelx + 1) + elx      # top-left
            
            # Check which edges are on boundary
            is_left_boundary = (elx == 0 or (elx > 0 and not domain_mask[ely, elx-1]))
            is_right_boundary = (elx == nelx-1 or (elx < nelx-1 and not domain_mask[ely, elx+1]))
            is_bottom_boundary = (ely == 0 or (ely > 0 and not domain_mask[ely-1, elx]))
            is_top_boundary = (ely == nely-1 or (ely < nely-1 and not domain_mask[ely+1, elx]))
            
            if is_left_boundary:
                boundary_nodes['left'].extend([n1, n4])
            if is_right_boundary:
                boundary_nodes['right'].extend([n2, n3])
            if is_bottom_boundary:
                boundary_nodes['bottom'].extend([n1, n2])
            if is_top_boundary:
                boundary_nodes['top'].extend([n3, n4])
    
    # Remove duplicates and sort
    for edge in boundary_nodes:
        boundary_nodes[edge] = sorted(list(set(boundary_nodes[edge])))
    
    return boundary_nodes


def generate_random_mechanism(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.20,
    k_in: float = None,
    k_out: float = None,
    seed: int = None,
    domain_mask: np.ndarray = None
) -> MechProblem:
    """
    Randomized but KINEMATICALLY-CONSISTENT mechanism (rewritten 2026-06-16).

    The previous version chose the output direction from a random `mech_behavior`
    INDEPENDENT of the support geometry, so it constantly requested directions the
    support could not produce -> collapse / disconnection (12% yield, 2026-06-16
    audit). This version samples one of three proven archetypes and derives the
    output direction from the support lever, randomizing axis / corner / port
    positions / springs for diversity:

      - 'invert'   : two corner supports flanking one edge; input mid that edge,
                     output reverses on the opposite edge. Axis randomized over
                     all 4 edges -> horizontal AND vertical inverters (novelty
                     beyond the canonical horizontal-only `inverter`).
      - 'amplify'  : single corner pivot, collinear near/far ports, same
                     direction (delegates to generate_amplifier_problem).
      - 'redirect' : single corner pivot, 90-degree redirect (delegates to
                     generate_crusher_problem or generate_crank_slider_problem).
    """
    if seed is not None:
        np.random.seed(seed)

    if k_in is None:
        k_in = np.random.uniform(0.005, 0.02)
    if k_out is None:
        k_out = np.random.uniform(0.01, 0.05)

    # 'amplify' REMOVED 2026-07-15: it delegated to the amplifier, whose
    # floating-translator degeneracy is unsolved (0% yield — docs/DATASET_V1.md
    # §5.9). Overnight n=100 audit showed the amplify branch was pure waste
    # (~1/3 of random samples, all gate-rejected). Restore only when the
    # amplifier archetype works (likely via Family E, not SIMP).
    archetype = np.random.choice(['invert', 'redirect'])

    # Delegate the corner-pivot archetypes to the proven generators (they encode
    # consistent kinematics); seed=None -> they draw from the current RNG stream.
    if archetype == 'redirect':
        gen = generate_crusher_problem if np.random.random() < 0.5 else generate_crank_slider_problem
        return gen(nelx, nely, volume_fraction, k_in, k_out, seed=None)

    # 'invert': two corner supports flanking one edge; output reverses on the
    # opposite edge. Randomizing the edge yields vertical inverters too.
    mesh = create_mesh(nelx, nely)
    material = Material()
    patch_r = max(3, nelx // 12)
    jit = max(2, nelx // 8)

    def patch(cx, cy, r=patch_r):
        nodes = []
        for y in range(max(0, cy - r), min(nely, cy + r) + 1):
            for x in range(max(0, cx - r), min(nelx, cx + r) + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= (r + 0.5) ** 2:
                    nodes.append(y * (nelx + 1) + x)
        return nodes

    def clip(v, lo, hi):
        return int(np.clip(v, lo, hi))

    side = np.random.choice(['left', 'right', 'bottom', 'top'])
    if side in ('left', 'right'):
        sx = 0 if side == 'left' else nelx
        fixed = patch(sx, 0) + patch(sx, nely)        # two corners on this vertical edge
        iy = clip(nely // 2 + np.random.randint(-jit, jit + 1), 1, nely - 1)
        oy = clip(nely // 2 + np.random.randint(-jit, jit + 1), 1, nely - 1)
        input_node = iy * (nelx + 1) + sx
        output_node = oy * (nelx + 1) + (nelx - sx)
        sgn = 1.0 if side == 'left' else -1.0          # input pushes INTO domain
        input_direction = (sgn, 0.0)
        output_direction = (-sgn, 0.0)                 # reversal at the opposite edge
    else:
        sy = 0 if side == 'bottom' else nely
        fixed = patch(0, sy) + patch(nelx, sy)         # two corners on this horizontal edge
        ix = clip(nelx // 2 + np.random.randint(-jit, jit + 1), 1, nelx - 1)
        ox = clip(nelx // 2 + np.random.randint(-jit, jit + 1), 1, nelx - 1)
        input_node = sy * (nelx + 1) + ix
        output_node = (nely - sy) * (nelx + 1) + ox
        sgn = 1.0 if side == 'bottom' else -1.0
        input_direction = (0.0, sgn)
        output_direction = (0.0, -sgn)

    fixed = list(set(fixed))
    bc = BoundaryCondition(
        node_indices=np.array(fixed, dtype=np.int32),
        directions=np.full(len(fixed), 2)
    )

    base_problem = Problem(
        mesh=mesh, material=material, bcs=[bc], loads=[],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.MECHANISM,
        domain_mask=domain_mask
    )

    return MechProblem(
        base_problem=base_problem,
        input_node=input_node, input_direction=input_direction,
        output_node=output_node, output_direction=output_direction,
        k_in=k_in, k_out=k_out
    )


def generate_crusher_problem(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.20,
    k_in: float = None,
    k_out: float = None,
    seed: int = None
) -> MechProblem:
    """
    Side-jaw compliant CRUSHER (distributed compliance), reworked 2026-07-10.

    Replaces the old "diagonal inverter" (single corner-pivot patch + far-edge
    90-degree redirect). That layout IS the lumped failure mode: one anchor lets
    the whole body rotate rigidly about it, so the optimal solution was a blob
    on a point hinge — crusher was the audit laggard. The 90-degree-redirect
    IDENTITY is kept (input axis perpendicular to jaw axis, distinct from the
    inverter's anti-parallel on-axis ports and the gripper's opposite-edge jaw)
    but rebuilt on the anti-lump pattern proven by the gripper/inverter reworks:

      - TWO spread support patches on a mount edge (one anchor -> rigid pivot;
        three -> over-constrained blob, both measured 2026-06-28);
      - input on the mount edge in the largest support gap, pushing INTO the
        domain (the arch between the anchors is what distributes compliance);
      - jaw on an ADJACENT edge, placed 0.35-0.75 of the span away from the
        mount so the bell-crank arm around the near anchor stays long (a jaw
        hugging the anchor degenerates back to a point pivot);
      - jaw direction DERIVED, not sampled: the inward normal of the jaw edge
        (closing on a workpiece at the center). Independent output directions
        were the root cause of the u_out=0 collapses on other types.

    Diversity mirrors gripper v2: domain shape (square/wide/tall active
    rectangle via domain_mask, mount on the LONG edge), support/port jitter,
    jaw on either adjacent edge, modest k_out. Mirror symmetry is NOT enforced
    (the single jaw is off-axis; symmetry-averaging zeroes its motion).
    """
    if seed is not None:
        np.random.seed(seed)

    if k_in is None:
        k_in = 0.01
    if k_out is None:
        # Same safe range as the gripper: stiff outputs collapse u_out toward 0.
        k_out = float(np.random.uniform(0.01, 0.04))

    mesh = create_mesh(nelx, nely)
    material = Material()

    def nid(x, y):
        return y * (nelx + 1) + x

    def jit(c, lo, hi, amt):
        return int(np.clip(c + np.random.randint(-amt, amt + 1), lo, hi))

    # ---- domain SHAPE: active rectangle (same carving as gripper v2) ----------
    shape = np.random.choice(['square', 'wide', 'tall'], p=[0.4, 0.3, 0.3])
    ex0, ey0, ex1, ey1 = 0, 0, nelx, nely
    if shape == 'wide':
        ey0 = np.random.randint(nely // 8, nely // 4 + 1)
        ey1 = nely - np.random.randint(nely // 8, nely // 4 + 1)
    elif shape == 'tall':
        ex0 = np.random.randint(nelx // 8, nelx // 4 + 1)
        ex1 = nelx - np.random.randint(nelx // 8, nelx // 4 + 1)
    domain_mask = np.zeros((nely, nelx), dtype=bool)
    domain_mask[ey0:ey1, ex0:ex1] = True

    W, H = ex1 - ex0, ey1 - ey0

    def patch(px, py, r):
        """BC node disk, clamped to the active rectangle."""
        pts = []
        for yy in range(max(ey0, py - r), min(ey1, py + r) + 1):
            for xx in range(max(ex0, px - r), min(ex1, px + r) + 1):
                if (xx - px) ** 2 + (yy - py) ** 2 <= (r + 0.5) ** 2:
                    pts.append(nid(xx, yy))
        return pts

    sup_fracs = [np.random.uniform(0.18, 0.32), np.random.uniform(0.68, 0.82)]
    # Mount on the LONG edge (spread anchors need room), like gripper v2.
    if shape == 'wide':
        orient = np.random.choice(['B', 'T'])
    elif shape == 'tall':
        orient = np.random.choice(['L', 'R'])
    else:
        orient = np.random.choice(['L', 'R', 'B', 'T'])
    vertical_mount = orient in ('L', 'R')
    # Jaw depth: fraction of the perpendicular span, measured FROM the mount
    # edge. >=0.35 keeps the bell-crank arm long (anti-pivot), <=0.75 keeps the
    # jaw off the far corners where the filter under-smooths.
    jaw_frac = np.random.uniform(0.35, 0.75)

    if vertical_mount:                       # anchors + input on a vertical edge
        pr = max(2, H // int(np.random.randint(12, 20)))
        mount_x = ex0 if orient == 'L' else ex1
        input_direction = (1.0, 0.0) if orient == 'L' else (-1.0, 0.0)
        sup = sorted(jit(int(ey0 + f * H), ey0 + pr + 1, ey1 - pr - 1, max(1, H // 14))
                     for f in sup_fracs)
        fixed_nodes = [n for s in sup for n in patch(mount_x, s, pr)]
        gap, mid = max((sup[i + 1] - sup[i], (sup[i] + sup[i + 1]) // 2)
                       for i in range(len(sup) - 1))
        in_y = jit(mid, ey0 + 1, ey1 - 1, max(1, gap // 4))
        input_node = nid(mount_x, in_y)
        # jaw on an ADJACENT (horizontal) edge, jaw_frac of W away from the mount
        jaw_edge = np.random.choice(['B', 'T'])
        jaw_y = ey0 if jaw_edge == 'B' else ey1
        raw_x = ex0 + jaw_frac * W if orient == 'L' else ex1 - jaw_frac * W
        jaw_x = jit(int(raw_x), ex0 + 2, ex1 - 2, max(1, W // 14))
        output_node = nid(jaw_x, jaw_y)
        output_direction = (0.0, 1.0) if jaw_edge == 'B' else (0.0, -1.0)
    else:                                    # anchors + input on a horizontal edge
        pr = max(2, W // int(np.random.randint(12, 20)))
        mount_y = ey0 if orient == 'B' else ey1
        input_direction = (0.0, 1.0) if orient == 'B' else (0.0, -1.0)
        sup = sorted(jit(int(ex0 + f * W), ex0 + pr + 1, ex1 - pr - 1, max(1, W // 14))
                     for f in sup_fracs)
        fixed_nodes = [n for s in sup for n in patch(s, mount_y, pr)]
        gap, mid = max((sup[i + 1] - sup[i], (sup[i] + sup[i + 1]) // 2)
                       for i in range(len(sup) - 1))
        in_x = jit(mid, ex0 + 1, ex1 - 1, max(1, gap // 4))
        input_node = nid(in_x, mount_y)
        # jaw on an ADJACENT (vertical) edge, jaw_frac of H away from the mount
        jaw_edge = np.random.choice(['L', 'R'])
        jaw_x = ex0 if jaw_edge == 'L' else ex1
        raw_y = ey0 + jaw_frac * H if orient == 'B' else ey1 - jaw_frac * H
        jaw_y = jit(int(raw_y), ey0 + 2, ey1 - 2, max(1, H // 14))
        output_node = nid(jaw_x, jaw_y)
        output_direction = (1.0, 0.0) if jaw_edge == 'L' else (-1.0, 0.0)

    fixed_nodes = sorted(set(fixed_nodes))
    bc = BoundaryCondition(
        node_indices=np.array(fixed_nodes, dtype=np.int32),
        directions=np.full(len(fixed_nodes), 2)
    )

    base_problem = Problem(
        mesh=mesh, material=material, bcs=[bc], loads=[],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.MECHANISM,
        domain_mask=domain_mask,
    )

    return MechProblem(
        base_problem=base_problem,
        input_node=input_node, input_direction=input_direction,
        output_node=output_node, output_direction=output_direction,
        k_in=k_in, k_out=k_out,
    )


def generate_amplifier_problem(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.20,
    k_in: float = None,
    k_out: float = None,
    seed: int = None
) -> MechProblem:
    """
    Symmetric BRIDGE displacement amplifier (Sigmund 2001 "L5"), reworked
    2026-07-10.

    Replaces the old corner-pivot lever (input close to a single anchor, output
    far, lever ratio = amplification). One anchor is the lumped failure mode —
    the optimum was a rigid blob rotating on a point. The bridge layout is
    distributed by construction: BOTH short ends of a shallow rectangle are
    clamped, the input pushes into the middle of one long edge, and the output
    at the middle of the OPPOSITE long edge is rewarded for continuing in the
    SAME direction (the rising apex of the double-arch). The amplification
    comes from the shallow-arch geometry, not from a pivot.

    Kinematic identity: opposite edge + parallel same-sense output (vs the
    inverter's anti-parallel converge, the gripper's tangential jaw, and the
    crusher's adjacent-edge redirect).

    Anchoring follows the repo's validated L5 template: TWO corner patches at
    the ends of the INPUT (mount) edge, ports on the centerline between them.
    NOT clamped end-strips: fully building in both short ends makes the arch
    membrane-stiff in exactly the requested direction, and the optimizer then
    parks the whole volume budget against the clamps and disconnects the ports
    (measured 2026-07-10: u_in = free spring stroke, u_out = 0, 0/16 audit).

    Like the inverter, mirror symmetry about the input-output axis IS enforced
    and the ports sit exactly on the centerline (independent port jitter tilts
    the load axis -> asymmetric blobs; measured on the inverter 2026-06).
    The domain is a shallow active rectangle (aspect ~1.6-2.4, via domain_mask)
    like L5's 2:1.

    SPRINGS: the amplifier is the one type that NEEDS k_in >> k_out (L5 uses
    k_in=1.0). With the soft k_in=0.01 the other types use, the optimum is a
    FREE-FLOATING translator: push a rod, u_out = u_in, and any ground
    attachment only adds parasitic stiffness — so the optimizer deliberately
    skips the anchors and the connectivity gate zeroes the type (0/16,
    measured 2026-07-10). With k_in >> k_out, beating the floating rod
    requires levering against ground at ratio ~sqrt(k_in/k_out), which is the
    amplification we want. Callers must pass k_in=None and let the generator
    randomize it stiff.
    """
    if seed is not None:
        np.random.seed(seed)

    if k_in is None:
        k_in = float(np.random.uniform(0.3, 1.0))
    if k_out is None:
        k_out = float(np.random.uniform(0.005, 0.02))

    mesh = create_mesh(nelx, nely)
    material = Material()

    def nid(x, y):
        return y * (nelx + 1) + x

    # ---- shallow active rectangle: long axis horizontal or vertical ----------
    horizontal = bool(np.random.random() < 0.5)
    aspect = np.random.uniform(1.6, 2.4)
    ex0, ey0, ex1, ey1 = 0, 0, nelx, nely
    if horizontal:                            # trim top+bottom to H = W/aspect
        h = int(round(nely / aspect))
        trim = nely - h
        t0 = np.random.randint(0, trim + 1)
        ey0, ey1 = t0, t0 + h
    else:                                     # trim left+right to W = H/aspect
        w = int(round(nelx / aspect))
        trim = nelx - w
        t0 = np.random.randint(0, trim + 1)
        ex0, ex1 = t0, t0 + w
    domain_mask = np.zeros((nely, nelx), dtype=bool)
    domain_mask[ey0:ey1, ex0:ex1] = True
    W, H = ex1 - ex0, ey1 - ey0

    # ---- anchors: TWO corner patches at the ends of the INPUT (mount) edge ---
    # L5-style (validated template). NOT clamped end-strips — see docstring.
    # The two patches are mirror images about the port centerline.
    def patch(px, py, r):
        """BC node disk, clamped to the active rectangle."""
        pts = []
        for yy in range(max(ey0, py - r), min(ey1, py + r) + 1):
            for xx in range(max(ex0, px - r), min(ex1, px + r) + 1):
                if (xx - px) ** 2 + (yy - py) ** 2 <= (r + 0.5) ** 2:
                    pts.append(nid(xx, yy))
        return pts

    flip = bool(np.random.random() < 0.5)         # which long edge is the mount
    # ports exactly on the centerline, slightly inset like the inverter
    inset = max(2, (H if horizontal else W) // int(np.random.randint(12, 24)))
    if horizontal:
        pr = max(3, W // int(np.random.randint(8, 14)))
        mount_y = ey0 if not flip else ey1
        fixed_nodes = patch(ex0, mount_y, pr) + patch(ex1, mount_y, pr)
        cx = (ex0 + ex1) // 2
        in_y = (ey0 + inset) if not flip else (ey1 - inset)
        out_y = (ey1 - inset) if not flip else (ey0 + inset)
        input_node, output_node = nid(cx, in_y), nid(cx, out_y)
        d = (0.0, 1.0) if not flip else (0.0, -1.0)
        symmetry = 'vertical'                     # mirror about the port axis
    else:
        pr = max(3, H // int(np.random.randint(8, 14)))
        mount_x = ex0 if not flip else ex1
        fixed_nodes = patch(mount_x, ey0, pr) + patch(mount_x, ey1, pr)
        cy = (ey0 + ey1) // 2
        in_x = (ex0 + inset) if not flip else (ex1 - inset)
        out_x = (ex1 - inset) if not flip else (ex0 + inset)
        input_node, output_node = nid(in_x, cy), nid(out_x, cy)
        d = (1.0, 0.0) if not flip else (-1.0, 0.0)
        symmetry = 'horizontal'
    input_direction = d
    output_direction = d                          # SAME direction = amplification

    # ---- kinematic seed: grounded double bell-crank (rhombus) ---------------
    # input -> both anchors, both anchors -> output. Without it the optimizer
    # converges to the degenerate FLOATING translator basin (u_out =
    # F/(k_in+k_out), ground never attached; grounded lever scores ~3x higher
    # but is unreachable from the uniform soup — measured 2026-07-10, u_out
    # trajectory flat at the floating value). Struts 1.0 on a background at the
    # target VF per the seeding contract; the optimizer's soft lock + volume
    # ramp handle the rest.
    seed = np.where(domain_mask, volume_fraction, 0.0)
    if horizontal:
        a1, a2 = nid(ex0, mount_y), nid(ex1, mount_y)
    else:
        a1, a2 = nid(mount_x, ey0), nid(mount_x, ey1)
    for frm, to in ((input_node, a1), (input_node, a2),
                    (a1, output_node), (a2, output_node)):
        seed = seed_connecting_path(seed, frm, to, nelx, nely, width=1, value=1.0)
    seed = np.where(domain_mask, seed, 0.0)

    fixed_nodes = sorted(set(fixed_nodes))
    bc = BoundaryCondition(
        node_indices=np.array(fixed_nodes, dtype=np.int32),
        directions=np.full(len(fixed_nodes), 2)
    )

    base_problem = Problem(
        mesh=mesh, material=material, bcs=[bc], loads=[],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.MECHANISM,
        domain_mask=domain_mask,
    )

    return MechProblem(
        base_problem=base_problem,
        input_node=input_node, input_direction=input_direction,
        output_node=output_node, output_direction=output_direction,
        k_in=k_in, k_out=k_out,
        symmetry=symmetry,
        seed_density=seed,
    )

def generate_crank_slider_problem(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.20,
    k_in: float = None,
    k_out: float = None,
    seed: int = None
) -> MechProblem:
    """
    Distributed WALL-SLIDER (reworked 2026-07-10).

    Replaces the single-corner-pivot redirector (the lumped failure mode: one
    anchor -> rigid blob rotating on a point hinge). Rebuilt on the two-anchor
    pattern proven by the gripper/crusher reworks, keeping a distinct kinematic
    identity: the output is a follower on an ADJACENT wall that SLIDES ALONG
    that wall (tangential), in the same sense as the input — like the slider of
    a crank-slider linkage running in its guide.

      - TWO spread support patches on a mount edge, input in the largest gap
        pushing INTO the domain (the distributed arch);
      - follower port on an adjacent edge, 0.35-0.75 of the span away from the
        mount (a port hugging the near anchor degenerates to a point pivot);
      - output direction DERIVED = the input direction itself (tangential to
        the follower wall). The arch rising between the anchors drags the wall
        material along with it; requesting the natural sense avoids the
        u_out=0 collapse that independent output directions caused.

    Distinctness: crusher = adjacent edge, inward NORMAL (jaw closes);
    slider = adjacent edge, TANGENTIAL (follower translates along its wall).
    Diversity mirrors gripper v2: domain shape (square/wide/tall via
    domain_mask, mount on the LONG edge), support/port jitter, modest k_out.
    No enforced symmetry (the follower is off-axis).
    """
    if seed is not None:
        np.random.seed(seed)

    if k_in is None:
        k_in = 0.01
    if k_out is None:
        k_out = float(np.random.uniform(0.01, 0.04))

    mesh = create_mesh(nelx, nely)
    material = Material()

    def nid(x, y):
        return y * (nelx + 1) + x

    def jit(c, lo, hi, amt):
        return int(np.clip(c + np.random.randint(-amt, amt + 1), lo, hi))

    # ---- domain SHAPE: active rectangle (same carving as gripper v2) ----------
    shape = np.random.choice(['square', 'wide', 'tall'], p=[0.4, 0.3, 0.3])
    ex0, ey0, ex1, ey1 = 0, 0, nelx, nely
    if shape == 'wide':
        ey0 = np.random.randint(nely // 8, nely // 4 + 1)
        ey1 = nely - np.random.randint(nely // 8, nely // 4 + 1)
    elif shape == 'tall':
        ex0 = np.random.randint(nelx // 8, nelx // 4 + 1)
        ex1 = nelx - np.random.randint(nelx // 8, nelx // 4 + 1)
    domain_mask = np.zeros((nely, nelx), dtype=bool)
    domain_mask[ey0:ey1, ex0:ex1] = True
    W, H = ex1 - ex0, ey1 - ey0

    def patch(px, py, r):
        """BC node disk, clamped to the active rectangle."""
        pts = []
        for yy in range(max(ey0, py - r), min(ey1, py + r) + 1):
            for xx in range(max(ex0, px - r), min(ex1, px + r) + 1):
                if (xx - px) ** 2 + (yy - py) ** 2 <= (r + 0.5) ** 2:
                    pts.append(nid(xx, yy))
        return pts

    sup_fracs = [np.random.uniform(0.18, 0.32), np.random.uniform(0.68, 0.82)]
    if shape == 'wide':
        orient = np.random.choice(['B', 'T'])
    elif shape == 'tall':
        orient = np.random.choice(['L', 'R'])
    else:
        orient = np.random.choice(['L', 'R', 'B', 'T'])
    vertical_mount = orient in ('L', 'R')
    follower_frac = np.random.uniform(0.35, 0.75)   # distance from the mount edge

    if vertical_mount:                       # anchors + input on a vertical edge
        pr = max(2, H // int(np.random.randint(12, 20)))
        mount_x = ex0 if orient == 'L' else ex1
        input_direction = (1.0, 0.0) if orient == 'L' else (-1.0, 0.0)
        sup = sorted(jit(int(ey0 + f * H), ey0 + pr + 1, ey1 - pr - 1, max(1, H // 14))
                     for f in sup_fracs)
        fixed_nodes = [n for s in sup for n in patch(mount_x, s, pr)]
        gap, mid = max((sup[i + 1] - sup[i], (sup[i] + sup[i + 1]) // 2)
                       for i in range(len(sup) - 1))
        in_y = jit(mid, ey0 + 1, ey1 - 1, max(1, gap // 4))
        input_node = nid(mount_x, in_y)
        # follower on an ADJACENT (horizontal) wall, sliding ALONG it (= x sense
        # of the input)
        fol_edge = np.random.choice(['B', 'T'])
        fol_y = ey0 if fol_edge == 'B' else ey1
        raw_x = ex0 + follower_frac * W if orient == 'L' else ex1 - follower_frac * W
        fol_x = jit(int(raw_x), ex0 + 2, ex1 - 2, max(1, W // 14))
        output_node = nid(fol_x, fol_y)
    else:                                    # anchors + input on a horizontal edge
        pr = max(2, W // int(np.random.randint(12, 20)))
        mount_y = ey0 if orient == 'B' else ey1
        input_direction = (0.0, 1.0) if orient == 'B' else (0.0, -1.0)
        sup = sorted(jit(int(ex0 + f * W), ex0 + pr + 1, ex1 - pr - 1, max(1, W // 14))
                     for f in sup_fracs)
        fixed_nodes = [n for s in sup for n in patch(s, mount_y, pr)]
        gap, mid = max((sup[i + 1] - sup[i], (sup[i] + sup[i + 1]) // 2)
                       for i in range(len(sup) - 1))
        in_x = jit(mid, ex0 + 1, ex1 - 1, max(1, gap // 4))
        input_node = nid(in_x, mount_y)
        # follower on an ADJACENT (vertical) wall, sliding ALONG it (= y sense
        # of the input)
        fol_edge = np.random.choice(['L', 'R'])
        fol_x = ex0 if fol_edge == 'L' else ex1
        raw_y = ey0 + follower_frac * H if orient == 'B' else ey1 - follower_frac * H
        fol_y = jit(int(raw_y), ey0 + 2, ey1 - 2, max(1, H // 14))
        output_node = nid(fol_x, fol_y)
    output_direction = input_direction       # tangential slide, natural sense

    fixed_nodes = sorted(set(fixed_nodes))
    bc = BoundaryCondition(
        node_indices=np.array(fixed_nodes, dtype=np.int32),
        directions=np.full(len(fixed_nodes), 2)
    )

    base_problem = Problem(
        mesh=mesh, material=material, bcs=[bc], loads=[],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.MECHANISM,
        domain_mask=domain_mask,
    )

    return MechProblem(
        base_problem=base_problem,
        input_node=input_node, input_direction=input_direction,
        output_node=output_node, output_direction=output_direction,
        k_in=k_in, k_out=k_out,
    )

MECH_GENERATORS = {
    'inverter': generate_inverter_problem,
    'gripper': generate_gripper_problem,
    'random': generate_random_mechanism,
    'crusher': generate_crusher_problem,
    'amplifier': generate_amplifier_problem,
    'crank_slider': generate_crank_slider_problem,
}


def generate_mechanism_sample(
    sample_id: int,
    resolution: int = 64,
    config: MechConfig = None,
    compute_physics: bool = True,
    refinement_factor: int = 2
) -> Optional[Dict]:
    """
    Generate a complete mechanism sample with validation.
    
    Args:
        sample_id: Unique sample identifier
        resolution: Mesh resolution
        config: Optimization configuration
        compute_physics: Whether to compute physics fields
        refinement_factor: Upscale factor for physics
        
    Returns:
        Dictionary with density, physics, and metadata, or None if failed
    """
    if config is None:
        config = MechConfig()
    
    # Select problem type based on sample_id for variety
    problem_types = list(MECH_GENERATORS.keys())
    problem_idx = sample_id % len(problem_types)
    problem_name = problem_types[problem_idx]
    generator = MECH_GENERATORS[problem_name]
    
    # Random seed based on sample_id
    seed = sample_id * 12345 + 67890
    
    # Generate problem
    mech_problem = generator(
        nelx=resolution,
        nely=resolution,
        volume_fraction=config.volume_fraction,
        k_in=config.k_in,
        k_out=config.k_out,
        seed=seed
    )
    
    # Run optimization
    start_time = time.time()
    result = optimize_mechanism(mech_problem, config)
    opt_time = time.time() - start_time
    
    # Validate
    is_valid, validation_info = validate_sample(
        result.density,
        result.volume_fraction,
        config.volume_fraction
    )
    
    if not is_valid:
        return None
    
    # Compute physics if requested
    physics_data = None
    if compute_physics:
        try:
            physics, density_fine = compute_mech_physics_fields(
                mech_problem,
                result.density,
                refinement_factor=refinement_factor,
                penal=config.penal_max
            )
            physics_data = {
                'displacement': physics.displacement,
                'stress_vm': physics.stress_vm,
                'strain_energy': physics.strain_energy
            }
        except Exception as e:
            print(f"Physics computation failed for sample {sample_id}: {e}")
    
    # Build metadata
    metadata = {
        'sample_id': sample_id,
        'tier': 2,  # Mech tier
        'tier_name': 'mech',
        'problem_type': problem_name,
        'resolution': resolution,
        'volume_fraction_target': config.volume_fraction,
        'volume_fraction_actual': result.volume_fraction,
        'optimization': {
            'n_iterations': result.n_iterations,
            'converged': result.converged,
            'final_objective': float(result.compliance),
            'time_seconds': opt_time
        },
        'mechanism': {
            'input_node': int(mech_problem.input_node),
            'input_direction': list(mech_problem.input_direction),
            'output_node': int(mech_problem.output_node),
            'output_direction': list(mech_problem.output_direction),
            'k_in': float(mech_problem.k_in),
            'k_out': float(mech_problem.k_out),
            'k_perp': float(mech_problem.k_perp)
        },
        'validation': validation_info
    }
    
    return {
        'density': result.density,
        'physics': physics_data,
        'metadata': metadata
    }
