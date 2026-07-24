"""
SIMP topology optimization for stiff structures.

Compliance minimization using OC (Optimality Criteria) update.
"""

import numpy as np
from scipy.ndimage import convolve, zoom
from scipy.sparse import diags
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Tuple, Optional, Callable, List, Dict
import time

from ..core.problem import Problem, ProblemType, Material, BoundaryCondition, Load
from ..core.mesh import Mesh2D, create_mesh
from ..solvers.linear import solve_fea, compute_compliance_sensitivity, FEAResult
from ..validation.connectivity import validate_sample


@dataclass
class OptimizationConfig:
    """Configuration for SIMP optimization."""
    max_iterations: int = 400       # Increased for better convergence
    convergence_tol: float = 0.01
    move_limit: float = 0.2
    penal_init: float = 1.0
    penal_max: float = 3.0
    penal_step: float = 0.25
    penal_interval: int = 20        # Faster penalization ramp
    filter_radius: float = 2.5      # Larger for clean struts (no pixel dust)
    volume_fraction: float = 0.3    # Lower for cleaner trusses
    # Heaviside projection parameters
    beta_init: float = 1.0          # Initial sharpness
    beta_max: float = 32.0          # Good balance of sharpness and stability
    beta_step: float = 2.0          # Multiplicative step
    beta_interval: int = 30         # Iterations between increases
    eta: float = 0.5                # Threshold value

    

@dataclass
class OptimizationResult:
    """Result from optimization run."""
    density: np.ndarray
    compliance: float
    volume_fraction: float
    n_iterations: int
    converged: bool
    convergence_history: List[float] = field(default_factory=list)
    time_seconds: float = 0.0
    

@dataclass
class PhysicsFields:
    """Physics fields from fine-mesh FEA (ground truth)."""
    displacement: np.ndarray   # (2, H, W) - nodal displacements reshaped
    stress_vm: np.ndarray      # (H, W) - von Mises stress per element
    strain_energy: float       # Total strain energy
    

def upscale_density(
    density: np.ndarray,
    factor: int = 2,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Upscale density field with proper thresholding.
    
    Two-mesh approach: design on coarse mesh, FEA on fine mesh.
    
    Args:
        density: (H, W) coarse density
        factor: Upscale factor (e.g., 2 for 64→128)
        threshold: Threshold before upscaling for binary result
    
    Returns:
        density_fine: (H*factor, W*factor) upscaled density
    """
    # Threshold to binary for crisp upscaling
    density_binary = (density > threshold).astype(np.float64)
    
    # Use nearest-neighbor for binary (no interpolation artifacts)
    density_fine = zoom(density_binary, factor, order=0)
    
    return density_fine


def create_problem_at_resolution(
    original_problem: Problem,
    new_resolution: int
) -> Problem:
    """
    Create a problem at a different resolution, scaling BCs and loads.
    
    For edge-based BCs (like fixed left edge), we need to properly
    regenerate ALL nodes along that edge at the new resolution.
    
    Args:
        original_problem: Original problem definition
        new_resolution: New mesh resolution (nelx = nely = new_resolution)
    
    Returns:
        New Problem with scaled node indices
    """
    old_nelx = original_problem.mesh.nelx
    old_nely = original_problem.mesh.nely
    
    # Scale factor
    scale = new_resolution / old_nelx
    
    # Create new mesh
    new_mesh = create_mesh(new_resolution, new_resolution)
    new_nodes_per_row = new_resolution + 1
    
    # Scale boundary conditions
    new_bcs = []
    for bc in original_problem.bcs:
        old_nodes_per_row = old_nelx + 1
        
        # Detect if this is an edge BC by checking if nodes form a line
        old_indices = bc.node_indices
        if len(old_indices) > 2:
            # Check if all nodes share the same x or y coordinate
            old_xs = [idx % old_nodes_per_row for idx in old_indices]
            old_ys = [idx // old_nodes_per_row for idx in old_indices]
            
            if len(set(old_xs)) == 1:
                # Vertical edge (constant x)
                old_x = old_xs[0]
                new_x = int(round(old_x * scale))
                new_x = min(new_x, new_resolution)
                # Generate all nodes along this edge
                new_indices = [y * new_nodes_per_row + new_x 
                              for y in range(new_resolution + 1)]
            elif len(set(old_ys)) == 1:
                # Horizontal edge (constant y)
                old_y = old_ys[0]
                new_y = int(round(old_y * scale))
                new_y = min(new_y, new_resolution)
                # Generate all nodes along this edge
                new_indices = [new_y * new_nodes_per_row + x 
                              for x in range(new_resolution + 1)]
            else:
                # General case: scale each node individually
                new_indices = []
                for old_idx in old_indices:
                    old_y = old_idx // old_nodes_per_row
                    old_x = old_idx % old_nodes_per_row
                    new_x = min(int(round(old_x * scale)), new_resolution)
                    new_y = min(int(round(old_y * scale)), new_resolution)
                    new_indices.append(new_y * new_nodes_per_row + new_x)
        else:
            # Single or double node - scale each
            new_indices = []
            for old_idx in old_indices:
                old_y = old_idx // old_nodes_per_row
                old_x = old_idx % old_nodes_per_row
                new_x = min(int(round(old_x * scale)), new_resolution)
                new_y = min(int(round(old_y * scale)), new_resolution)
                new_indices.append(new_y * new_nodes_per_row + new_x)
        
        # Expand directions to match new node count
        if len(bc.directions) == 1:
            # Broadcast single direction to all nodes
            new_directions = np.full(len(new_indices), bc.directions[0])
        elif len(bc.directions) == len(bc.node_indices):
            # Need to match new number of nodes
            new_directions = np.full(len(new_indices), bc.directions[0])
        else:
            new_directions = bc.directions.copy()
        
        new_bcs.append(BoundaryCondition(
            node_indices=np.array(new_indices, dtype=np.int32),
            directions=new_directions
        ))
    
    # Scale loads
    new_loads = []
    for load in original_problem.loads:
        old_idx = load.node_index
        old_nodes_per_row = old_nelx + 1
        new_nodes_per_row = new_resolution + 1
        
        old_y = old_idx // old_nodes_per_row
        old_x = old_idx % old_nodes_per_row
        
        new_x = int(round(old_x * scale))
        new_y = int(round(old_y * scale))
        new_x = min(new_x, new_resolution)
        new_y = min(new_y, new_resolution)
        
        new_idx = new_y * new_nodes_per_row + new_x
        
        new_loads.append(Load(
            node_index=new_idx,
            fx=load.fx,
            fy=load.fy
        ))
    
    # Downsample a NON-TRIVIAL domain mask to the new resolution (nearest) so a
    # multi-resolution warm-start optimizes in the SAME carved domain at every
    # level. Dropping it (None) made the coarse stage fill the full square; the
    # fine-level mask then chopped the upscaled warm-start -> disconnected
    # mechanisms (the dominant collapse of the diversified gripper's wide/tall
    # domains). A full (all-True) mask stays None — unchanged for the common
    # full-domain case (mask is then set later from the upscaled density).
    new_domain_mask = None
    if original_problem.domain_mask is not None and not np.all(original_problem.domain_mask):
        mz = zoom(original_problem.domain_mask.astype(np.float64),
                  (new_resolution / old_nely, new_resolution / old_nelx), order=0)
        new_domain_mask = mz > 0.5

    return Problem(
        mesh=new_mesh,
        material=original_problem.material,
        bcs=new_bcs,
        loads=new_loads,
        volume_fraction=original_problem.volume_fraction,
        problem_type=original_problem.problem_type,
        domain_mask=new_domain_mask  # downsampled if non-trivial, else None
    )


def compute_physics_fields(
    problem: Problem,
    density: np.ndarray,
    refinement_factor: int = 2,
    penal: float = 3.0
) -> Tuple[PhysicsFields, np.ndarray]:
    """
    Run FEA on refined mesh to get ground-truth physics fields.
    
    Two-mesh principle:
    - Optimization runs on coarse mesh (e.g., 64×64)
    - Physics runs on fine mesh (e.g., 128×128) for accuracy
    
    Args:
        problem: Original problem at design resolution
        density: (H, W) optimized density at design resolution
        refinement_factor: Upscale factor (default 2)
        penal: SIMP penalization
    
    Returns:
        physics: PhysicsFields with displacement and stress
        density_fine: Upscaled density used for FEA
    """
    nely, nelx = density.shape
    fine_res = nelx * refinement_factor
    
    # Upscale density
    density_fine = upscale_density(density, factor=refinement_factor)
    
    # Create problem at fine resolution
    fine_problem = create_problem_at_resolution(problem, fine_res)
    
    # Run FEA on fine mesh with stress computation
    result = solve_fea(fine_problem, density_fine, penal, compute_stress=True)
    
    # Reshape displacement to (2, H+1, W+1) grid (nodal values)
    n_nodes_per_dim = fine_res + 1
    u_x = result.displacement[0::2].reshape(n_nodes_per_dim, n_nodes_per_dim)
    u_y = result.displacement[1::2].reshape(n_nodes_per_dim, n_nodes_per_dim)
    displacement = np.stack([u_x, u_y], axis=0)  # (2, H+1, W+1)
    
    # Stress is per-element, already (H, W) from solve
    if result.stress_vm is not None:
        stress_vm = result.stress_vm
    else:
        # Should not happen, but fallback
        stress_vm = np.zeros((fine_res, fine_res))
    
    physics = PhysicsFields(
        displacement=displacement,
        stress_vm=stress_vm,
        strain_energy=result.strain_energy
    )
    
    return physics, density_fine
    

def create_density_filter(nelx: int, nely: int, radius: float) -> np.ndarray:
    """
    Create convolution kernel for density filtering.
    
    Filter is a cone-shaped weight function that averages
    densities within the filter radius.
    
    Args:
        nelx: Number of elements in X
        nely: Number of elements in Y
        radius: Filter radius in elements
    
    Returns:
        kernel: (2*ceil(r)+1, 2*ceil(r)+1) convolution kernel
    """
    r = int(np.ceil(radius))
    size = 2 * r + 1
    
    kernel = np.zeros((size, size))
    center = r
    
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist <= radius:
                kernel[i, j] = radius - dist
    
    # Normalize
    kernel /= kernel.sum()
    
    return kernel


def apply_density_filter(
    density: np.ndarray,
    kernel: np.ndarray
) -> np.ndarray:
    """
    Apply density filter using convolution.
    
    Args:
        density: (nely, nelx) raw densities
        kernel: Convolution kernel from create_density_filter
    
    Returns:
        filtered: (nely, nelx) filtered densities
    """
    return convolve(density, kernel, mode='reflect')


def apply_sensitivity_filter(
    dc: np.ndarray,
    density: np.ndarray,
    kernel: np.ndarray
) -> np.ndarray:
    """
    Apply sensitivity filter (consistent with density filter).
    
    d̃c/dρ = (1/max(ε, ρ)) * H * (ρ * dc/dρ)
    
    Args:
        dc: (nely, nelx) raw sensitivity
        density: (nely, nelx) densities
        kernel: Convolution kernel
    
    Returns:
        dc_filtered: (nely, nelx) filtered sensitivity
    """
    # Regularization to avoid division by zero
    eps = 1e-8
    rho_safe = np.maximum(density, eps)
    
    # Weighted sensitivity
    numerator = convolve(density * dc, kernel, mode='reflect')
    denominator = convolve(density, kernel, mode='reflect')
    
    return numerator / np.maximum(denominator, eps)


def apply_heaviside_projection(
    density: np.ndarray,
    beta: float,
    eta: float = 0.5
) -> np.ndarray:
    """
    Apply smooth Heaviside projection to push densities toward 0 or 1.
    
    H(x) = (tanh(β*η) + tanh(β*(x-η))) / (tanh(β*η) + tanh(β*(1-η)))
    
    As β increases, the projection becomes sharper.
    
    Args:
        density: (nely, nelx) filtered densities
        beta: Sharpness parameter
        eta: Threshold value (typically 0.5)
    
    Returns:
        projected: (nely, nelx) projected densities, clamped to [rho_min, 1.0]
    """
    numerator = np.tanh(beta * eta) + np.tanh(beta * (density - eta))
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    projected = numerator / denominator
    # Clamp to prevent exactly-zero densities which cause singular stiffness matrix
    # NOTE: Use very small rho_min (1e-6) - direct solver handles this fine.
    # 1e-3 is too high and makes void act like "rubber foam", causing blobby designs.
    rho_min = 1e-6  # Very small - allows proper void while maintaining numerical stability
    return np.clip(projected, rho_min, 1.0)


def heaviside_derivative(
    density: np.ndarray,
    beta: float,
    eta: float = 0.5
) -> np.ndarray:
    """
    Derivative of Heaviside projection for chain rule.
    
    dH/dx = β * (1 - tanh²(β*(x-η))) / (tanh(β*η) + tanh(β*(1-η)))
    
    Args:
        density: (nely, nelx) filtered densities (before projection)
        beta: Sharpness parameter
        eta: Threshold value
    
    Returns:
        dH_dx: (nely, nelx) derivatives
    """
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    sech_sq = 1 - np.tanh(beta * (density - eta))**2
    return beta * sech_sq / denominator


def oc_update(
    density: np.ndarray,
    dc: np.ndarray,
    volume_target: float,
    move: float = 0.2,
    domain_mask: np.ndarray = None,
    kernel: np.ndarray = None,
    beta: float = 1.0,
    eta: float = 0.5
) -> np.ndarray:
    """
    Optimality Criteria (OC) update for compliance minimization.
    
    Finds density update that minimizes Lagrangian:
    L = c + λ(V - V_target)
    
    Uses bisection to find Lagrange multiplier λ.
    When kernel/beta are provided, volume is computed on projected density.
    
    Args:
        density: (nely, nelx) current design variables
        dc: (nely, nelx) sensitivity (dc/drho)
        volume_target: Target volume fraction
        move: Move limit
        domain_mask: Optional mask for active elements
        kernel: Optional filter kernel for volume check
        beta: Heaviside sharpness for volume check
        eta: Heaviside threshold
    
    Returns:
        new_density: (nely, nelx) updated densities
    """
    if domain_mask is None:
        domain_mask = np.ones_like(density, dtype=bool)
    
    nely, nelx = density.shape
    total_elements = np.sum(domain_mask)
    target_volume = volume_target * total_elements
    
    # Sensitivity must be negative for compliance minimization
    # OC formula: rho_new = rho * sqrt(-dc / lambda)
    # We need dc < 0 (which it should be for compliance)
    
    # Avoid division by zero
    eps = 1e-10
    
    # Bisection to find lambda
    l1, l2 = 0, 1e9
    
    while (l2 - l1) / (l1 + l2 + eps) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        
        # OC update formula
        # For compliance minimization: dc < 0, so -dc > 0
        # Clamp to avoid sqrt of negative (can happen at boundaries)
        B = np.sqrt(np.maximum(-dc / (lmid + eps), eps))
        
        # Apply update with move limits and bounds
        new_density = np.maximum(
            np.maximum(density - move, 0.0),
            np.minimum(
                np.minimum(density + move, 1.0),
                density * B
            )
        )
        
        # Apply domain mask
        new_density = np.where(domain_mask, new_density, 0.0)
        
        # Check volume constraint on projected density
        if kernel is not None:
            filtered = apply_density_filter(new_density, kernel)
            projected = apply_heaviside_projection(filtered, beta, eta)
            current_volume = np.sum(projected[domain_mask])
        else:
            current_volume = np.sum(new_density)
        
        if current_volume > target_volume:
            l1 = lmid
        else:
            l2 = lmid
    
    return new_density


def optimize_compliance(
    problem: Problem,
    config: OptimizationConfig = None,
    initial_density: np.ndarray = None,
    callback: Callable = None
) -> OptimizationResult:
    """
    Run SIMP topology optimization for compliance minimization.
    
    Args:
        problem: Problem definition
        config: Optimization configuration
        initial_density: Optional starting density
        callback: Optional callback(iter, density, compliance)
    
    Returns:
        OptimizationResult with final density and statistics
    """
    if config is None:
        config = OptimizationConfig()
    
    mesh = problem.mesh
    nelx, nely = mesh.nelx, mesh.nely
    
    # Initialize density
    if initial_density is not None:
        density = initial_density.copy()
    else:
        density = np.ones((nely, nelx)) * config.volume_fraction
    
    # Apply domain mask
    if problem.domain_mask is not None:
        density = np.where(problem.domain_mask, density, 0.0)
    
    # Create filter kernel
    kernel = create_density_filter(nelx, nely, config.filter_radius)
    
    # Optimization loop
    penal = config.penal_init
    beta = config.beta_init  # Heaviside sharpness
    history = []
    start_time = time.time()
    
    compliance_old = np.inf
    converged = False
    
    for iteration in range(config.max_iterations):
        # Apply density filter
        density_filtered = apply_density_filter(density, kernel)
        
        # Apply Heaviside projection
        density_projected = apply_heaviside_projection(
            density_filtered, beta, config.eta
        )
        
        # Solve FEA
        result = solve_fea(problem, density_projected, penal)
        compliance = result.compliance
        history.append(compliance)
        
        # Compute sensitivity
        dc = compute_compliance_sensitivity(
            problem, density_projected, result.displacement, penal
        )
        
        # Chain rule: dc/dx = dc/dρ_proj * dρ_proj/dρ_filt * dρ_filt/dx
        # Apply Heaviside derivative
        dH = heaviside_derivative(density_filtered, beta, config.eta)
        dc = dc * dH
        
        # Filter sensitivity
        dc = apply_sensitivity_filter(dc, density_filtered, kernel)
        
        # OC update (on design variables, with projected volume check)
        density = oc_update(
            density, dc, config.volume_fraction, 
            config.move_limit, problem.domain_mask,
            kernel=kernel, beta=beta, eta=config.eta
        )
        
        # Callback
        if callback is not None:
            callback(iteration, density_projected, compliance)
        
        # Continuation: increase penalization
        if (iteration + 1) % config.penal_interval == 0:
            penal = min(penal + config.penal_step, config.penal_max)
        
        # Heaviside continuation: increase beta
        if (iteration + 1) % config.beta_interval == 0:
            beta = min(beta * config.beta_step, config.beta_max)
        
        # Convergence check
        if iteration > 10:
            change = abs(compliance - compliance_old) / (compliance_old + 1e-10)
            if change < config.convergence_tol and penal >= config.penal_max and beta >= config.beta_max:
                converged = True
                break
        
        compliance_old = compliance
    
    # Final: filter + project
    density_filtered = apply_density_filter(density, kernel)
    density_final = apply_heaviside_projection(density_filtered, beta, config.eta)
    
    # Apply domain mask
    if problem.domain_mask is not None:
        density_final = np.where(problem.domain_mask, density_final, 0.0)
    
    elapsed = time.time() - start_time
    
    return OptimizationResult(
        density=density_final,
        compliance=history[-1] if history else np.inf,
        volume_fraction=float(np.mean(density_final[problem.domain_mask] 
                                       if problem.domain_mask is not None 
                                       else density_final)),
        n_iterations=len(history),
        converged=converged,
        convergence_history=history,
        time_seconds=elapsed
    )


def generate_random_cantilever(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.4,
    seed: int = None
) -> Problem:
    """
    Generate a random cantilever problem variant.
    
    Randomizes:
    - Fixed edge (left only - more stable)
    - Load position along free edge
    - Load direction (constrained to point into domain)
    
    Args:
        nelx: Elements in X (typically 64)
        nely: Elements in Y (now equal to nelx for square grid)
        volume_fraction: Target volume
        seed: Random seed
    
    Returns:
        Problem instance
    """
    if seed is not None:
        np.random.seed(seed)
    
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # Always fix left edge for cantilever (most stable configuration)
    fixed_nodes = np.array([i * (nelx + 1) for i in range(nely + 1)], dtype=np.int32)
    bc = BoundaryCondition(
        node_indices=fixed_nodes,
        directions=np.full(len(fixed_nodes), 2)  # Fix both x and y
    )
    
    # Load on right edge, random vertical position
    load_y = np.random.randint(1, nely)  # Avoid corners
    load_node = load_y * (nelx + 1) + nelx
    
    # Load direction: mostly leftward/downward (into the domain)
    # Avoid loads pointing away from fixed edge
    angle = np.random.uniform(-0.8 * np.pi, 0.8 * np.pi)  # -144° to +144°
    angle += np.pi  # Point toward left (fixed) edge
    fx = np.cos(angle)
    fy = np.sin(angle)
    load = Load(node_index=load_node, fx=fx, fy=fy)
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc],
        loads=[load],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE
    )


def generate_random_bridge(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.4,
    seed: int = None
) -> Problem:
    """
    Generate a random bridge problem variant.
    
    Args:
        nelx: Elements in X (typically 64)
        nely: Elements in Y (now equal to nelx for square grid)
        volume_fraction: Target volume
        seed: Random seed
    
    Returns:
        Problem instance
    """
    if seed is not None:
        np.random.seed(seed)
    
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # Support positions (both on bottom edge)
    left_support = np.random.randint(0, nelx // 4)
    right_support = np.random.randint(3 * nelx // 4, nelx + 1)
    
    # Left support: pin (fix both x and y)
    bc_left = BoundaryCondition(
        node_indices=np.array([left_support], dtype=np.int32),
        directions=np.array([2])
    )
    
    # Right support: roller (fix y only)
    bc_right = BoundaryCondition(
        node_indices=np.array([right_support], dtype=np.int32),
        directions=np.array([1])
    )
    
    # Load position (on top edge, between supports)
    load_x = np.random.randint(left_support, right_support + 1)
    load_node = nely * (nelx + 1) + load_x
    
    # Load direction (mostly downward)
    fy = -1.0
    fx = np.random.uniform(-0.3, 0.3)
    
    load = Load(node_index=load_node, fx=fx, fy=fy)
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc_left, bc_right],
        loads=[load],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE
    )


def generate_mbb_beam(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.4,
    seed: int = None
) -> Problem:
    """
    Generate MBB (Messerschmidt-Bölkow-Blohm) beam problem.
    
    Classic benchmark: symmetric half of a simply-supported beam.
    Left edge: roller (fix x), bottom-right corner: pin (fix y).
    Load at top-left corner.
    
    Args:
        nelx: Elements in X
        nely: Elements in Y
        volume_fraction: Target volume
        seed: Random seed
    
    Returns:
        Problem instance
    """
    if seed is not None:
        np.random.seed(seed)
    
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # Left edge: symmetry BC (fix x displacement)
    left_nodes = np.array([i * (nelx + 1) for i in range(nely + 1)], dtype=np.int32)
    bc_left = BoundaryCondition(
        node_indices=left_nodes,
        directions=np.full(len(left_nodes), 0)  # Fix x only
    )
    
    # Bottom-right corner: fix y
    bc_corner = BoundaryCondition(
        node_indices=np.array([nelx], dtype=np.int32),
        directions=np.array([1])  # Fix y only
    )
    
    # Load at top-left corner (pointing down)
    load_node = nely * (nelx + 1)  # Top-left
    
    # Randomize load slightly
    fy = -1.0
    fx = np.random.uniform(-0.1, 0.1)
    
    load = Load(node_index=load_node, fx=fx, fy=fy)
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc_left, bc_corner],
        loads=[load],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE
    )


def generate_l_bracket(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.4,
    seed: int = None
) -> Problem:
    """
    Generate L-bracket problem with passive (void) region.
    
    Upper-right quadrant is void. Top edge fixed, load on right edge.
    
    Args:
        nelx: Elements in X
        nely: Elements in Y  
        volume_fraction: Target volume
        seed: Random seed
    
    Returns:
        Problem instance with domain mask
    """
    if seed is not None:
        np.random.seed(seed)
    
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # Create L-shaped domain mask (upper-right is void)
    domain_mask = np.ones((nely, nelx), dtype=bool)
    cut_x = nelx // 2 + np.random.randint(-nelx//8, nelx//8)
    cut_y = nely // 2 + np.random.randint(-nely//8, nely//8)
    domain_mask[cut_y:, cut_x:] = False
    
    # Top edge of the vertical leg: fixed
    top_nodes = np.array([nely * (nelx + 1) + x for x in range(cut_x + 1)], dtype=np.int32)
    bc = BoundaryCondition(
        node_indices=top_nodes,
        directions=np.full(len(top_nodes), 2)  # Fix both
    )
    
    # Load on right edge of horizontal leg (bottom portion)
    load_y = np.random.randint(1, cut_y)
    load_node = load_y * (nelx + 1) + nelx
    
    # Load pointing left/down
    fx = np.random.uniform(-1.0, -0.5)
    fy = np.random.uniform(-0.5, 0.0)
    
    load = Load(node_index=load_node, fx=fx, fy=fy)
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc],
        loads=[load],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE,
        domain_mask=domain_mask
    )


def generate_double_clamped(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.4,
    seed: int = None
) -> Problem:
    """
    Generate double-clamped beam problem.
    
    Both left and right edges fixed, load in center.
    
    Args:
        nelx: Elements in X
        nely: Elements in Y
        volume_fraction: Target volume
        seed: Random seed
    
    Returns:
        Problem instance
    """
    if seed is not None:
        np.random.seed(seed)
    
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # Left edge fixed
    left_nodes = np.array([i * (nelx + 1) for i in range(nely + 1)], dtype=np.int32)
    bc_left = BoundaryCondition(
        node_indices=left_nodes,
        directions=np.full(len(left_nodes), 2)
    )
    
    # Right edge fixed
    right_nodes = np.array([i * (nelx + 1) + nelx for i in range(nely + 1)], dtype=np.int32)
    bc_right = BoundaryCondition(
        node_indices=right_nodes,
        directions=np.full(len(right_nodes), 2)
    )
    
    # Load at center, random position along vertical
    load_x = nelx // 2 + np.random.randint(-nelx//8, nelx//8)
    load_y = np.random.randint(nely//4, 3*nely//4)
    load_node = load_y * (nelx + 1) + load_x
    
    # Random load direction
    angle = np.random.uniform(0, 2 * np.pi)
    fx = np.cos(angle)
    fy = np.sin(angle)
    
    load = Load(node_index=load_node, fx=fx, fy=fy)
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc_left, bc_right],
        loads=[load],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE
    )


def generate_corner_load(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.4,
    seed: int = None
) -> Problem:
    """
    Generate corner load problem.
    
    One edge fixed, load at opposite corner.
    
    Args:
        nelx: Elements in X
        nely: Elements in Y
        volume_fraction: Target volume
        seed: Random seed
    
    Returns:
        Problem instance
    """
    if seed is not None:
        np.random.seed(seed)
    
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # Choose which edge to fix
    edge = np.random.randint(0, 4)
    
    if edge == 0:  # Left edge fixed
        fixed_nodes = np.array([i * (nelx + 1) for i in range(nely + 1)], dtype=np.int32)
        # Load at right corners
        corner = np.random.choice([nelx, nely * (nelx + 1) + nelx])
    elif edge == 1:  # Right edge fixed
        fixed_nodes = np.array([i * (nelx + 1) + nelx for i in range(nely + 1)], dtype=np.int32)
        # Load at left corners
        corner = np.random.choice([0, nely * (nelx + 1)])
    elif edge == 2:  # Bottom edge fixed
        fixed_nodes = np.array(list(range(nelx + 1)), dtype=np.int32)
        # Load at top corners
        corner = np.random.choice([nely * (nelx + 1), nely * (nelx + 1) + nelx])
    else:  # Top edge fixed
        fixed_nodes = np.array([nely * (nelx + 1) + x for x in range(nelx + 1)], dtype=np.int32)
        # Load at bottom corners
        corner = np.random.choice([0, nelx])
    
    bc = BoundaryCondition(
        node_indices=fixed_nodes,
        directions=np.full(len(fixed_nodes), 2)
    )
    
    # Random load direction (pointing toward center)
    angle = np.random.uniform(0, 2 * np.pi)
    fx = np.cos(angle)
    fy = np.sin(angle)
    
    load = Load(node_index=corner, fx=fx, fy=fy)
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc],
        loads=[load],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE
    )


def generate_multi_load(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.4,
    seed: int = None
) -> Problem:
    """
    Generate problem with multiple loads.
    
    One edge fixed, 2-3 loads at various positions.
    
    Args:
        nelx: Elements in X
        nely: Elements in Y
        volume_fraction: Target volume
        seed: Random seed
    
    Returns:
        Problem instance
    """
    if seed is not None:
        np.random.seed(seed)
    
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # Fix left edge
    fixed_nodes = np.array([i * (nelx + 1) for i in range(nely + 1)], dtype=np.int32)
    bc = BoundaryCondition(
        node_indices=fixed_nodes,
        directions=np.full(len(fixed_nodes), 2)
    )
    
    # Generate 2-3 loads
    n_loads = np.random.randint(2, 4)
    loads = []
    
    for _ in range(n_loads):
        # Random position on right or top edge
        if np.random.random() < 0.5:
            # Right edge
            load_y = np.random.randint(1, nely)
            load_node = load_y * (nelx + 1) + nelx
        else:
            # Top edge
            load_x = np.random.randint(nelx//2, nelx)
            load_node = nely * (nelx + 1) + load_x
        
        # Random direction
        angle = np.random.uniform(-np.pi, np.pi)
        fx = np.cos(angle)
        fy = np.sin(angle)
        
        loads.append(Load(node_index=load_node, fx=fx, fy=fy))
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc],
        loads=loads,
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE
    )


# =============================================================================
# CANONICAL PROBLEMS (No randomization - classic benchmarks)
# =============================================================================

def generate_canonical_cantilever(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.3,
    seed: int = None  # ignored, kept for API compatibility
) -> Problem:
    """
    Classic cantilever beam: fixed left edge, point load at mid-right.
    
    This is the most common topology optimization benchmark.
    Aspect ratio 2:1 is traditional but we use 1:1 for square grids.
    """
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # Fix entire left edge
    fixed_nodes = np.array([i * (nelx + 1) for i in range(nely + 1)], dtype=np.int32)
    bc = BoundaryCondition(
        node_indices=fixed_nodes,
        directions=np.full(len(fixed_nodes), 2)  # Fix both x and y
    )
    
    # Point load at mid-right, pointing down
    load_node = (nely // 2) * (nelx + 1) + nelx
    load = Load(node_index=load_node, fx=0.0, fy=-1.0)
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc],
        loads=[load],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE
    )


def generate_canonical_mbb(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.3,
    seed: int = None
) -> Problem:
    """
    Classic MBB beam: left edge roller (fix x), bottom-right support (fix y),
    load at top-left corner pointing down.
    
    This exploits symmetry - the full beam is mirrored about the left edge.
    Uses a small support region at bottom-right for numerical stability.
    """
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # Left edge: symmetry BC (fix x displacement only)
    left_nodes = np.array([i * (nelx + 1) for i in range(nely + 1)], dtype=np.int32)
    bc_left = BoundaryCondition(
        node_indices=left_nodes,
        directions=np.full(len(left_nodes), 0)  # Fix x only
    )
    
    # Bottom-right support region: fix y only for a few nodes
    support_width = max(2, nelx // 32)  # Small support region
    right_nodes = np.arange(nelx - support_width, nelx + 1, dtype=np.int32)
    bc_corner = BoundaryCondition(
        node_indices=right_nodes,
        directions=np.full(len(right_nodes), 1)  # Fix y only
    )
    
    # Load at top-left corner, pointing straight down
    load_node = nely * (nelx + 1)  # Top-left
    load = Load(node_index=load_node, fx=0.0, fy=-1.0)
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc_left, bc_corner],
        loads=[load],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE
    )


def generate_canonical_bridge(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.35,  # Higher VF for connected bridge
    seed: int = None
) -> Problem:
    """
    Classic bridge: supports at bottom corners (few nodes each),
    load at top center pointing down.
    
    More robust than single-node supports.
    """
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # Left support region: fix both x and y for a few nodes
    support_width = max(2, nelx // 16)
    left_nodes = np.arange(support_width + 1, dtype=np.int32)
    bc_left = BoundaryCondition(
        node_indices=left_nodes,
        directions=np.full(len(left_nodes), 2)  # Fix both
    )
    
    # Right support region: fix y only (roller)
    right_nodes = np.arange(nelx - support_width, nelx + 1, dtype=np.int32)
    bc_right = BoundaryCondition(
        node_indices=right_nodes,
        directions=np.full(len(right_nodes), 1)  # Fix y only
    )
    
    # Load at top center, pointing down
    load_node = nely * (nelx + 1) + nelx // 2
    load = Load(node_index=load_node, fx=0.0, fy=-1.0)
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc_left, bc_right],
        loads=[load],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE
    )


def generate_canonical_lbracket(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.3,
    seed: int = None
) -> Problem:
    """
    Classic L-bracket: fixed top edge, L-shaped domain (void in top-right),
    load at the tip of the horizontal arm (bottom-right of active domain).
    
    Domain layout (L rotated 180°):
      [FIXED]  [FIXED]  [FIXED]
      [solid]  [solid]  [void]
      [solid]  [solid]  [void]
      [solid]  [solid]  <-- LOAD HERE (tip of horizontal arm)
    """
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # Fixed top edge (only the solid part, left half + some margin)
    # Full top edge for simplicity (nodes over void are inactive anyway)
    top_nodes = np.array([nely * (nelx + 1) + i for i in range(nelx // 2 + 1)], dtype=np.int32)
    bc = BoundaryCondition(
        node_indices=top_nodes,
        directions=np.full(len(top_nodes), 2)  # Fix both
    )
    
    # L-bracket domain mask: void in top-right quadrant
    domain_mask = np.ones((nely, nelx), dtype=bool)
    domain_mask[nely//2:, nelx//2:] = False
    
    # Load at tip of horizontal arm: bottom-right corner of active domain
    # That's at row (nely//2 - 1), column (nelx - 1), which is at the bottom
    # of the lower-right solid region, pointing down
    # Node at bottom-right of active domain: row nely//2, column nelx//2
    load_row = nely // 2 - 1  # Just above the middle (bottom of upper solid)
    load_col = nelx - 1       # Right edge... but that's in the void
    # Actually the arm tip is at: row 0 to nely//2-1, col nelx//2 to nelx-1 is VOID
    # The L-shape has: bottom half is full width, top half is left half only
    # So the "arm tip" is at: row nely//2-1 (bottom of void), col nelx//2-1 (left of void)
    # Let's put load at the far right of the bottom row (which is solid)
    load_node = 0 * (nelx + 1) + (nelx // 2 - 1)  # Bottom row, at the inner corner
    # Better: load at the right edge of bottom strip, pointing left
    load_node = 0 * (nelx + 1) + nelx  # Bottom-right corner node
    load = Load(node_index=load_node, fx=0.0, fy=-1.0)
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc],
        loads=[load],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE,
        domain_mask=domain_mask
    )


def generate_canonical_michell(
    nelx: int = 64,
    nely: int = 64,
    volume_fraction: float = 0.3,
    seed: int = None
) -> Problem:
    """
    Classic Michell structure: fixed center region, loads on left/right edges.
    
    This produces the famous Michell truss pattern with radial struts.
    Two point loads on opposite sides, fixed central region.
    """
    mesh = create_mesh(nelx, nely)
    material = Material()
    
    # Fixed central region (small block in center)
    center_x, center_y = nelx // 2, nely // 2
    fixed_nodes = []
    support_radius = max(2, nelx // 16)
    
    for i in range(-support_radius, support_radius + 1):
        for j in range(-support_radius, support_radius + 1):
            nx = center_x + i
            ny = center_y + j
            if 0 <= nx <= nelx and 0 <= ny <= nely:
                node_idx = ny * (nelx + 1) + nx
                fixed_nodes.append(node_idx)
    
    bc = BoundaryCondition(
        node_indices=np.array(fixed_nodes, dtype=np.int32),
        directions=np.full(len(fixed_nodes), 2)  # Fix both x and y
    )
    
    # Two opposing horizontal loads on left and right edges at mid-height
    # Left edge: pull left (negative x)
    load_left = Load(
        node_index=(nely // 2) * (nelx + 1),  # Left edge, mid-height
        fx=-1.0, fy=0.0
    )
    # Right edge: pull right (positive x)
    load_right = Load(
        node_index=(nely // 2) * (nelx + 1) + nelx,  # Right edge, mid-height
        fx=1.0, fy=0.0
    )
    
    return Problem(
        mesh=mesh,
        material=material,
        bcs=[bc],
        loads=[load_left, load_right],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.COMPLIANCE
    )


# List of all problem generators - CANONICAL FIRST, then random variants
PROBLEM_GENERATORS = [
    # Canonical benchmarks (no randomization)
    ("cantilever", generate_canonical_cantilever),
    ("mbb_beam", generate_canonical_mbb),
    ("bridge", generate_canonical_bridge),
    ("l_bracket", generate_canonical_lbracket),
    ("michell", generate_canonical_michell),
    # Random variants for diversity
    ("cantilever_var", generate_random_cantilever),
    ("bridge_var", generate_random_bridge),
    ("double_clamped", generate_double_clamped),
    ("corner_load", generate_corner_load),
    ("multi_load", generate_multi_load),
]


def generate_stiff_sample(
    sample_id: int,
    resolution: int = 64,
    volume_fraction: float = 0.4,
    config: OptimizationConfig = None,
    warm_start: np.ndarray = None,
    max_retries: int = 3,
    compute_physics: bool = True,
    refinement_factor: int = 2
) -> Tuple[Optional[OptimizationResult], Optional[PhysicsFields], Optional[Problem], dict]:
    """
    Generate a single stiff structure sample with physics fields.
    
    Two-mesh approach:
    - Optimize on coarse mesh (resolution × resolution)
    - Run FEA on fine mesh (resolution*factor × resolution*factor)
    
    Args:
        sample_id: Sample identifier (used as seed)
        resolution: Mesh resolution (nelx = nely = resolution, square grid)
        volume_fraction: Target volume fraction
        config: Optimization config
        warm_start: Optional warm-start density
        max_retries: Max retries with different seeds if validation fails
        compute_physics: Whether to run fine-mesh FEA for physics fields
        refinement_factor: Fine mesh factor (2 means 64→128)
    
    Returns:
        result: OptimizationResult or None if failed
        physics: PhysicsFields with displacement/stress on fine mesh
        problem: Problem definition
        metadata: Sample metadata dict
    """
    if config is None:
        config = OptimizationConfig(volume_fraction=volume_fraction)
    else:
        # Override config volume_fraction with the passed value
        config = dataclass_replace(config, volume_fraction=volume_fraction)
    
    best_result = None
    best_physics = None
    best_problem = None
    best_metadata = None
    best_n_components = float('inf')
    
    # Number of problem types available
    n_problem_types = len(PROBLEM_GENERATORS)
    
    for attempt in range(max_retries):
        # Use different seed for each attempt
        current_seed = sample_id * 1000 + attempt
        
        # Cycle through problem types based on sample_id
        problem_idx = (sample_id + attempt) % n_problem_types
        problem_name, problem_generator = PROBLEM_GENERATORS[problem_idx]
        
        # Square grid: nelx = nely = resolution
        problem = problem_generator(
            nelx=resolution,
            nely=resolution,
            volume_fraction=volume_fraction,
            seed=current_seed
        )
        
        # Initialize from warm start or uniform
        if warm_start is not None:
            # Interpolate if resolution differs
            if warm_start.shape != (resolution, resolution):
                scale_y = resolution / warm_start.shape[0]
                scale_x = resolution / warm_start.shape[1]
                initial = zoom(warm_start, (scale_y, scale_x), order=1)
                initial = np.clip(initial, 0, 1)
            else:
                initial = warm_start
        else:
            initial = None
        
        # Run optimization
        try:
            result = optimize_compliance(problem, config, initial)
        except Exception as e:
            continue  # Try next seed
        
        # Validate result
        min_feature = 1 if resolution <= 64 else 2
        validation = validate_sample(
            result.density,
            target_vf=volume_fraction,
            domain_mask=problem.domain_mask,
            min_feature_size=min_feature
        )
        
        metadata = {
            "sample_id": sample_id,
            "attempt": attempt,
            "seed": current_seed,
            "problem_type": problem_name,
            "resolution": resolution,
            "fine_resolution": resolution * refinement_factor,
            "volume_fraction": result.volume_fraction,
            "compliance": result.compliance,
            "n_iterations": result.n_iterations,
            "converged": result.converged,
            "time_seconds": result.time_seconds,
            "validation": validation
        }
        
        # If validation passes, compute physics and return
        if validation["overall_passed"]:
            physics = None
            if compute_physics:
                try:
                    physics, density_fine = compute_physics_fields(
                        problem, result.density, 
                        refinement_factor=refinement_factor
                    )
                    metadata["physics_computed"] = True
                except Exception as e:
                    metadata["physics_computed"] = False
                    metadata["physics_error"] = str(e)
            
            return result, physics, problem, metadata
        
        # Track best attempt (fewest components)
        n_comp = validation.get("connectivity", {}).get("n_components", float('inf'))
        if n_comp < best_n_components:
            best_n_components = n_comp
            best_result = result
            best_problem = problem
            best_metadata = metadata
            
            # Also compute physics for best attempt
            if compute_physics:
                try:
                    best_physics, _ = compute_physics_fields(
                        problem, result.density,
                        refinement_factor=refinement_factor
                    )
                    best_metadata["physics_computed"] = True
                except Exception as e:
                    best_physics = None
                    best_metadata["physics_computed"] = False
    
    # Return best attempt even if not passing
    return best_result, best_physics, best_problem, best_metadata
