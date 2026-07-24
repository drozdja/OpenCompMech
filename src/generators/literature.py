"""
Literature-replicated mechanism problem generators.

Each function creates a MechProblem from a published SIMP mechanism benchmark.
These are PROVEN problem setups that produce recognizable compliant mechanisms.
Every template supports parametric sweeps (VF, k_out, filter_radius) for diversity.

IMPORTANT — Adaptation from original papers:
  - Input/output nodes are inset 2-3 elements from any fixed boundary
    (our solver constrains fixed-edge DOFs to zero, so I/O ON the edge
    would be dead).
  - Spring stiffnesses are calibrated for our MSE-αSE formulation:
    k_in ~ 0.01 (soft actuator), k_out ~ 0.01-0.05 (workpiece).
  - Patch BCs (≥20 nodes) are used instead of single-node supports
    for numerical stability.

References:
  L1-L3:  Sigmund (1997), "On the Design of Compliant Mechanisms..."
  L4-L5:  Sigmund (2001), "Design of multiphysics actuators..."
  L6:     Bruns & Tortorelli (2001), "Topology optimization of non-linear elastic structures..."
  L7:     Wang et al. (2011), "On projection methods, convergence and robust formulations..."
  L8:     Pedersen et al. (2001), "Topology synthesis of large-displacement mechanisms"
  L9:     Luo et al. (2005), "Compliant mechanism design using multi-objective..."
  L10:    Jonsmann et al. (1999), MEMS microgrip
  L11:    Frecker et al. (1997), "Topological synthesis of compliant mechanisms..."
  L12:    Yin & Ananthasuresh (2001), "Topology optimization of compliant mechanisms..."
  L13:    Sigmund (2001), actuator variant
  L14:    Bendsøe & Sigmund (2003), book example
  L15:    Sigmund (2001), compliant OR-gate
"""

import numpy as np
from typing import Optional, Dict, Tuple, List

from ..core.problem import Problem, ProblemType, Material, BoundaryCondition, Load
from ..core.mesh import Mesh2D, create_mesh
from .mech import MechProblem

# --- Calibrated spring stiffnesses for our MSE-αSE formulation ---
_K_IN = 0.01       # Soft actuator spring (proven: inverter u_out=22)
_K_OUT = 0.03      # Moderate workpiece resistance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_nodes(edge: str, nelx: int, nely: int) -> np.ndarray:
    """All nodes on an edge. 'left','right','top','bottom'."""
    nx = nelx + 1
    if edge == 'left':
        return np.array([y * nx for y in range(nely + 1)], dtype=int)
    elif edge == 'right':
        return np.array([y * nx + nelx for y in range(nely + 1)], dtype=int)
    elif edge == 'bottom':
        return np.arange(nx, dtype=int)
    elif edge == 'top':
        return np.arange(nx, dtype=int) + nely * nx
    else:
        raise ValueError(f"Unknown edge: {edge}")


def _edge_nodes_excluding(
    edge: str, nelx: int, nely: int, exclude_nodes: set
) -> np.ndarray:
    """All nodes on an edge, excluding specific nodes."""
    all_nodes = _edge_nodes(edge, nelx, nely)
    return np.array([n for n in all_nodes if n not in exclude_nodes], dtype=int)


def _node_at(x: int, y: int, nelx: int) -> int:
    """Node index at grid position (x, y)."""
    return y * (nelx + 1) + x


def _patch_nodes(cx: int, cy: int, radius: int, nelx: int, nely: int) -> np.ndarray:
    """Circular patch of nodes around (cx, cy)."""
    nodes = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                nx_coord, ny_coord = cx + dx, cy + dy
                if 0 <= nx_coord <= nelx and 0 <= ny_coord <= nely:
                    nodes.append(ny_coord * (nelx + 1) + nx_coord)
    return np.array(nodes, dtype=int) if nodes else np.array([], dtype=int)


def _band_nodes(
    edge: str, nelx: int, nely: int,
    start_frac: float, end_frac: float, width: int = 3
) -> np.ndarray:
    """Band of nodes along an edge between fractional positions.
    
    E.g., _band_nodes('left', 64, 64, 0.0, 0.3, width=3) gives
    nodes on x=0..2, y=0..19.
    """
    nodes = []
    nx = nelx + 1
    if edge in ('left', 'right'):
        base_x = 0 if edge == 'left' else nelx
        y_start = int(round(start_frac * nely))
        y_end = int(round(end_frac * nely))
        for y in range(y_start, y_end + 1):
            for dx in range(width):
                x = base_x + dx if edge == 'left' else base_x - dx
                if 0 <= x <= nelx:
                    nodes.append(y * nx + x)
    elif edge in ('bottom', 'top'):
        base_y = 0 if edge == 'bottom' else nely
        x_start = int(round(start_frac * nelx))
        x_end = int(round(end_frac * nelx))
        for x in range(x_start, x_end + 1):
            for dy in range(width):
                y = base_y + dy if edge == 'bottom' else base_y - dy
                if 0 <= y <= nely:
                    nodes.append(y * nx + x)
    return np.array(sorted(set(nodes)), dtype=int)


def _make_bc(nodes: np.ndarray, direction: int = 2) -> BoundaryCondition:
    """Create BC. direction: 0=x, 1=y, 2=both."""
    return BoundaryCondition(
        node_indices=nodes,
        directions=np.full(len(nodes), direction, dtype=int)
    )


def _make_problem(
    nelx: int, nely: int, bcs: list, 
    input_node: int, input_direction: tuple,
    output_node: int, output_direction: tuple,
    volume_fraction: float = 0.30,
    k_in: float = _K_IN, k_out: float = _K_OUT,
    domain_mask: np.ndarray = None
) -> MechProblem:
    """Assemble a MechProblem from components with I/O validation."""
    mesh = create_mesh(nelx, nely)
    material = Material(E=1.0, nu=0.3)
    
    base = Problem(
        mesh=mesh, material=material, bcs=bcs, loads=[],
        volume_fraction=volume_fraction,
        problem_type=ProblemType.MECHANISM,
        domain_mask=domain_mask
    )
    
    # Validate: I/O DOFs must NOT be in the fixed set
    fixed_dofs = set(base.get_fixed_dofs())
    in_dof_x = 2 * input_node
    in_dof_y = 2 * input_node + 1
    out_dof_x = 2 * output_node
    out_dof_y = 2 * output_node + 1
    
    dx_in, dy_in = input_direction
    if (abs(dx_in) > 0 and in_dof_x in fixed_dofs) or \
       (abs(dy_in) > 0 and in_dof_y in fixed_dofs):
        nx = nelx + 1
        ix, iy = input_node % nx, input_node // nx
        raise ValueError(
            f"Input node ({ix},{iy}) has constrained DOFs in the active "
            f"force direction {input_direction}. Move it away from fixed BCs."
        )
    
    dx_out, dy_out = output_direction
    if (abs(dx_out) > 0 and out_dof_x in fixed_dofs) or \
       (abs(dy_out) > 0 and out_dof_y in fixed_dofs):
        nx = nelx + 1
        ox, oy = output_node % nx, output_node // nx
        raise ValueError(
            f"Output node ({ox},{oy}) has constrained DOFs in the active "
            f"direction {output_direction}. Move it away from fixed BCs."
        )
    
    return MechProblem(
        base_problem=base,
        input_node=input_node,
        input_direction=input_direction,
        output_node=output_node,
        output_direction=output_direction,
        k_in=k_in,
        k_out=k_out
    )


def _scale_to_resolution(
    original_nelx: int, original_nely: int,
    target_nelx: int, target_nely: int,
    x: int, y: int
) -> Tuple[int, int]:
    """Scale a node position from original to target resolution."""
    sx = target_nelx / original_nelx
    sy = target_nely / original_nely
    return int(round(x * sx)), int(round(y * sy))


def _apply_sweep(
    base_vf: float, base_k_in: float, base_k_out: float,
    vf: float = None, k_in: float = None, k_out: float = None
) -> Tuple[float, float, float]:
    """Apply parametric sweep overrides."""
    return (
        vf if vf is not None else base_vf,
        k_in if k_in is not None else base_k_in,
        k_out if k_out is not None else base_k_out
    )


def _jitter(rng, value: int, amount: int, lo: int, hi: int) -> int:
    """Add random jitter to a position, clamped to [lo, hi]."""
    return int(np.clip(value + rng.randint(-amount, amount + 1), lo, hi))


# ---------------------------------------------------------------------------
# L1: Force Inverter (Sigmund 1997) — THE canonical benchmark
# ---------------------------------------------------------------------------

def generate_L1_force_inverter(
    nelx: int = 64, nely: int = 64,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Force inverter: two corner patches (TL + BL) as ground, input between
    them at mid-left, output at mid-right. Input → right, output → left.
    
    Patch BCs give the input node displacement freedom (critical for
    our MSE-αSE formulation). The gap between patches creates lever arm.
    
    Original: Sigmund (1997), 80×80 domain, VF=0.30
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.30, _K_IN, _K_OUT, vf, k_in, k_out)
    
    # Two corner patches on left side (TL + BL)
    patch_r = max(3, nelx // 12)
    tl_patch = _patch_nodes(0, 0, patch_r, nelx, nely)
    bl_patch = _patch_nodes(0, nely, patch_r, nelx, nely)
    bcs = [_make_bc(np.concatenate([tl_patch, bl_patch]))]
    
    # Input between patches, slightly inset from left edge
    inset = max(3, nelx // 16)
    in_x, in_y = inset, nely // 2
    out_x, out_y = nelx - inset, nely // 2
    
    if jitter_amount > 0:
        in_x = _jitter(rng, in_x, jitter_amount, 2, nelx // 4)
        in_y = _jitter(rng, in_y, jitter_amount, patch_r + 3, nely - patch_r - 3)
        out_y = _jitter(rng, out_y, jitter_amount, 2, nely - 2)
    
    in_dir = (1.0, 0.0)
    out_dir = (-1.0, 0.0)
    
    # Apply mirror transformation
    if mirror in ('h', 'hv'):
        tl_patch = _patch_nodes(nelx, 0, patch_r, nelx, nely)
        bl_patch = _patch_nodes(nelx, nely, patch_r, nelx, nely)
        bcs = [_make_bc(np.concatenate([tl_patch, bl_patch]))]
        in_x = nelx - in_x
        out_x = nelx - out_x
        in_dir = (-in_dir[0], in_dir[1])
        out_dir = (-out_dir[0], out_dir[1])
    if mirror in ('v', 'hv'):
        in_y = nely - in_y
        out_y = nely - out_y
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L1',
        'literature_name': 'Force Inverter (Sigmund 1997)',
        'literature_nl_original': False,
        'original_domain': '80x80',
        'original_params': {'VF': 0.30, 'k_in': 1.0, 'k_out': 0.1},
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
        'mirror': mirror,
        'jitter': jitter_amount
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# L2: Gripper / Jaw (Sigmund 1997)
# ---------------------------------------------------------------------------

def generate_L2_gripper(
    nelx: int = 64, nely: int = 64,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Gripper: two support patches on left edge, mid-left input →, 
    top-right output ↓ (jaw closes downward).
    Input is between support patches, at x=0.
    Supports are far enough that the input node is NOT constrained.
    
    Original: 80×80 half-symmetry, VF=0.30
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.30, _K_IN, 0.02, vf, k_in, k_out)
    
    # Two support patches on left edge (quarter and three-quarter heights)
    patch_r = max(2, nelx // 16)
    bc1 = _patch_nodes(0, nely // 4, patch_r, nelx, nely)
    bc2 = _patch_nodes(0, 3 * nely // 4, patch_r, nelx, nely)
    bcs = [_make_bc(np.concatenate([bc1, bc2]))]
    
    in_x, in_y = 0, nely // 2
    out_x, out_y = nelx, nely // 8  # Top-right jaw tip
    
    if jitter_amount > 0:
        in_y = _jitter(rng, in_y, jitter_amount, nely // 4 + patch_r + 2, 3 * nely // 4 - patch_r - 2)
        out_y = _jitter(rng, out_y, jitter_amount, 2, nely // 4 - 2)
    
    in_dir = (1.0, 0.0)
    out_dir = (0.0, 1.0)  # Jaw closes downward (in image coords, +y = down)
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L2',
        'literature_name': 'Gripper (Sigmund 1997)',
        'literature_nl_original': False,
        'original_domain': '80x80',
        'original_params': {'VF': 0.30, 'k_in': 1.0, 'k_out': 0.01},
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# L3: Crimper (Sigmund 1997)
# ---------------------------------------------------------------------------

def generate_L3_crimper(
    nelx: int = 64, nely: int = 64,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Crimper: two support patches on bottom, bottom-center input ↑, 
    top-center output ↓ (two-sided crimping).
    Input is between supports, not on the BC patch.
    
    Original: 80×80 half-symmetry, VF=0.30
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.30, _K_IN, 0.02, vf, k_in, k_out)
    
    patch_r = max(2, nelx // 16)
    bc1 = _patch_nodes(nelx // 4, 0, patch_r, nelx, nely)
    bc2 = _patch_nodes(3 * nelx // 4, 0, patch_r, nelx, nely)
    bcs = [_make_bc(np.concatenate([bc1, bc2]))]
    
    # Input between the two supports, at bottom edge
    in_x, in_y = nelx // 2, 0
    out_x, out_y = nelx // 2, nely
    
    if jitter_amount > 0:
        in_x = _jitter(rng, in_x, jitter_amount, nelx // 4 + patch_r + 2, 3 * nelx // 4 - patch_r - 2)
        out_x = _jitter(rng, out_x, jitter_amount, nelx // 4 + patch_r + 2, 3 * nelx // 4 - patch_r - 2)
    
    in_dir = (0.0, 1.0)     # Push downward in image (up physically) into domain
    out_dir = (0.0, -1.0)   # Want opposite at top = crimping
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L3',
        'literature_name': 'Crimper (Sigmund 1997)',
        'literature_nl_original': False,
        'original_domain': '80x80',
        'original_params': {'VF': 0.30, 'k_in': 1.0, 'k_out': 0.05},
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# L4: Upper-Jaw Gripper (Sigmund 2001) — asymmetric
# ---------------------------------------------------------------------------

def generate_L4_upper_jaw(
    nelx: int = 64, nely: int = 80,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Asymmetric gripper: BL + BR corner patches as ground.
    Input at bottom-center pushes up, output at top-right moves right (jaw).
    
    Vertical-to-horizontal force conversion creates jaw topology.
    
    Original: 80×100 domain, VF=0.25
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.25, _K_IN, 0.02, vf, k_in, k_out)
    
    # Two bottom corner patches
    patch_r = max(3, nelx // 12)
    bl_patch = _patch_nodes(0, nely, patch_r, nelx, nely)
    br_patch = _patch_nodes(nelx, nely, patch_r, nelx, nely)
    bcs = [_make_bc(np.concatenate([bl_patch, br_patch]))]
    
    # Input between patches at bottom, inset from edge
    inset = max(3, nely // 16)
    in_x, in_y = nelx // 2, nely - inset
    out_x, out_y = 3 * nelx // 4, nely // 4    # Top-right area
    
    if jitter_amount > 0:
        in_x = _jitter(rng, in_x, jitter_amount, patch_r + 3, nelx - patch_r - 3)
        out_y = _jitter(rng, out_y, jitter_amount, 2, nely // 2)
    
    in_dir = (0.0, -1.0)   # Push upward (into domain)
    out_dir = (1.0, 0.0)   # Want rightward motion at output (jaw closes)
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L4',
        'literature_name': 'Upper-Jaw Gripper (Sigmund 2001)',
        'literature_nl_original': False,
        'original_domain': '80x100',
        'original_params': {'VF': 0.25, 'k_in': 0.5, 'k_out': 0.02},
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# L5: Displacement Amplifier (Sigmund 2001)
# ---------------------------------------------------------------------------

def generate_L5_amplifier(
    nelx: int = 64, nely: int = 32,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Displacement amplifier: two large corner patches (BL, BR) as supports.
    Input at bottom-center pushes up, output at top-center wants up.
    
    The optimizer must create a mechanism that amplifies vertical
    displacement via lever arms between the supports.
    
    Original: 80×40 domain, VF=0.30
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.30, _K_IN, 0.02, vf, k_in, k_out)
    
    # Two bottom corner patches (large enough for stability)
    patch_r = max(3, nelx // 10)
    bc_bl = _patch_nodes(0, 0, patch_r, nelx, nely)
    bc_br = _patch_nodes(nelx, 0, patch_r, nelx, nely)
    bcs = [_make_bc(np.concatenate([bc_bl, bc_br]))]
    
    # Input at bottom-center, between the two supports
    in_x, in_y = nelx // 2, 0
    out_x, out_y = nelx // 2, nely   # Top-center
    
    if jitter_amount > 0:
        in_x = _jitter(rng, in_x, jitter_amount, patch_r + 3, nelx - patch_r - 3)
        out_x = _jitter(rng, out_x, jitter_amount, nelx // 4, 3 * nelx // 4)
    
    in_dir = (0.0, 1.0)    # Push down (into domain)
    out_dir = (0.0, 1.0)   # Want same direction = amplification
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L5',
        'literature_name': 'Displacement Amplifier (Sigmund 2001)',
        'literature_nl_original': False,
        'original_domain': '80x40',
        'original_params': {'VF': 0.30, 'k_in': 1.0, 'k_out': 0.05},
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# L6: Corner Inverter (Bruns & Tortorelli 2001)
# ---------------------------------------------------------------------------

def generate_L6_inverter_projection(
    nelx: int = 64, nely: int = 64,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Single-support diagonal inverter: BL corner patch only.
    Input at bottom-left area pushing right, output at top-right pushing left.
    
    Single ground attachment creates long diagonal lever arm with
    maximum topological freedom (only one constraint region).
    
    Original: inspired by Bruns & Tortorelli (2001), 100×100
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.30, _K_IN, _K_OUT, vf, k_in, k_out)
    
    # BL + TR diagonal patches — unique diagonal ground arrangement
    patch_r = max(4, nelx // 10)
    bl_patch = _patch_nodes(0, nely, patch_r, nelx, nely)
    tr_patch = _patch_nodes(nelx, 0, patch_r, nelx, nely)
    bcs = [_make_bc(np.concatenate([bl_patch, tr_patch]))]
    
    # Input near bottom-left (but above the BL patch), output near top-right
    in_x = max(3, nelx // 8)
    in_y = 3 * nely // 4
    out_x = 7 * nelx // 8
    out_y = nely // 4
    
    if jitter_amount > 0:
        in_x = _jitter(rng, in_x, jitter_amount, 3, nelx // 3)
        in_y = _jitter(rng, in_y, jitter_amount, nely // 2, nely - patch_r - 2)
        out_x = _jitter(rng, out_x, jitter_amount, 2 * nelx // 3, nelx - 2)
        out_y = _jitter(rng, out_y, jitter_amount, 2, nely // 2)
    
    in_dir = (1.0, 0.0)     # Push right
    out_dir = (-1.0, 0.0)   # Want left (inversion)
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L6',
        'literature_name': 'Corner Inverter (Bruns & Tortorelli 2001)',
        'literature_nl_original': False,
        'original_domain': '100x100',
        'original_params': {'VF': 0.30, 'k_in': 1.0, 'k_out': 0.1},
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# L7: Orthogonal Motion Converter (Wang et al. 2011)
# ---------------------------------------------------------------------------

def generate_L7_robust_inverter(
    nelx: int = 64, nely: int = 64,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Vertical-to-horizontal converter: BL + BR bottom patches,
    bottom-left input pushes up, top-right output wants rightward.
    
    90-degree force path conversion creates L-shaped mechanism topology.
    
    Inspired by Wang et al. (2011), robust formulation.
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.30, _K_IN, _K_OUT, vf, k_in, k_out)
    
    # Two bottom patches (BL + BR)
    patch_r = max(3, nelx // 12)
    bl_patch = _patch_nodes(0, nely, patch_r, nelx, nely)
    br_patch = _patch_nodes(nelx, nely, patch_r, nelx, nely)
    bcs = [_make_bc(np.concatenate([bl_patch, br_patch]))]
    
    # Input at bottom-left area pushing up, output at top-right pushing right
    in_x = nelx // 4
    in_y = nely - max(3, nely // 16)  # Inset from bottom
    out_x = 3 * nelx // 4
    out_y = nely // 4
    
    if jitter_amount > 0:
        in_x = _jitter(rng, in_x, jitter_amount, patch_r + 3, nelx // 2 - 2)
        in_y = _jitter(rng, in_y, jitter_amount, nely // 2, nely - patch_r - 2)
        out_x = _jitter(rng, out_x, jitter_amount, nelx // 2 + 2, nelx - 2)
        out_y = _jitter(rng, out_y, jitter_amount, 2, nely // 2 - 2)
    
    in_dir = (0.0, -1.0)    # Push upward
    out_dir = (1.0, 0.0)    # Want rightward
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L7',
        'literature_name': 'Orthogonal Converter (Wang et al. 2011)',
        'literature_nl_original': False,
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# L8: Cruncher (Pedersen et al. 2001)
# ---------------------------------------------------------------------------

def generate_L8_cruncher(
    nelx: int = 64, nely: int = 64,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Cruncher: four corner patches as supports, left-side input →,
    center output moves downward (squeezing).
    
    Creates a multi-hinge mechanism that redirects lateral force to 
    vertical crushing motion.
    
    Original: 80×80, VF=0.30
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.30, _K_IN, 0.03, vf, k_in, k_out)
    
    # Four corner patches
    patch_r = max(3, nelx // 12)
    corners = [(0, 0), (nelx, 0), (nelx, nely), (0, nely)]
    all_bc_nodes = []
    for cx, cy in corners:
        all_bc_nodes.append(_patch_nodes(cx, cy, patch_r, nelx, nely))
    bcs = [_make_bc(np.concatenate(all_bc_nodes))]
    
    # Input: left side, between corners
    in_x, in_y = 0, nely // 2
    # Output: center, want downward
    out_x, out_y = nelx // 2, nely // 2
    
    if jitter_amount > 0:
        in_y = _jitter(rng, in_y, jitter_amount, patch_r + 3, nely - patch_r - 3)
        out_y = _jitter(rng, out_y, jitter_amount, nelx // 4, 3 * nelx // 4)
    
    in_dir = (1.0, 0.0)     # Push inward from left
    out_dir = (0.0, 1.0)    # Center squeezes downward
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L8',
        'literature_name': 'Cruncher (Pedersen et al. 2001)',
        'literature_nl_original': False,
        'original_domain': '80x80',
        'original_params': {'VF': 0.30, 'k_in': 1.0, 'k_out': 0.05},
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# L9: Bi-directional Inverter (Luo et al. 2005)
# ---------------------------------------------------------------------------

def generate_L9_bidirectional(
    nelx: int = 64, nely: int = 64,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Bi-directional inverter: TL + BL corner patches as ground,
    mid-left input → right, top-right output moves upward.
    
    Converts horizontal input to vertical output (orthogonal redirect).
    Two left-side patches create hinge-like topology.
    
    Original: Luo et al. (2005), 80×80, VF=0.30
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.30, _K_IN, 0.03, vf, k_in, k_out)
    
    # Two left-side corner patches (TL + BL)
    patch_r = max(3, nelx // 12)
    tl_patch = _patch_nodes(0, 0, patch_r, nelx, nely)
    bl_patch = _patch_nodes(0, nely, patch_r, nelx, nely)
    bcs = [_make_bc(np.concatenate([tl_patch, bl_patch]))]
    
    # Input between patches, inset from left edge
    inset = max(3, nelx // 16)
    in_x, in_y = inset, nely // 2
    out_x, out_y = 3 * nelx // 4, nely // 4   # Upper-right INTERIOR (not on edge)
    
    if jitter_amount > 0:
        in_y = _jitter(rng, in_y, jitter_amount, patch_r + 3, nely - patch_r - 3)
        out_x = _jitter(rng, out_x, jitter_amount, nelx // 2 + 2, nelx - 2)
        out_y = _jitter(rng, out_y, jitter_amount, 2, nely // 2 - 2)
    
    in_dir = (1.0, 0.0)
    out_dir = (0.0, -1.0)  # Upward
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L9',
        'literature_name': 'Bi-directional Inverter (Luo et al. 2005)',
        'literature_nl_original': False,
        'original_domain': '80x80',
        'original_params': {'VF': 0.30, 'k_in': 1.0, 'k_out': 0.05},
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# L10: Microgrip (Jonsmann et al. 1999)
# ---------------------------------------------------------------------------

def generate_L10_microgrip(
    nelx: int = 64, nely: int = 64,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    MEMS microgrip: TL + TR top corner patches as ground.
    Top-center input pushes down, bottom-center output moves down (amplifier).
    
    Vertical amplifier: two top supports with I/O along center vertical axis.
    Creates symmetric lever arm topology.
    
    Original: Jonsmann et al. (1999), 60×60, VF=0.25
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.25, _K_IN, 0.02, vf, k_in, k_out)
    
    # Two top corner patches (TL + TR)
    patch_r = max(3, nelx // 12)
    tl_patch = _patch_nodes(0, 0, patch_r, nelx, nely)
    tr_patch = _patch_nodes(nelx, 0, patch_r, nelx, nely)
    bcs = [_make_bc(np.concatenate([tl_patch, tr_patch]))]
    
    # Input at top-center between patches, inset from edge
    inset = max(3, nely // 16)
    in_x, in_y = nelx // 2, inset
    out_x, out_y = nelx // 2, nely - inset   # Bottom center
    
    if jitter_amount > 0:
        in_x = _jitter(rng, in_x, jitter_amount, patch_r + 3, nelx - patch_r - 3)
        out_x = _jitter(rng, out_x, jitter_amount, nelx // 4, 3 * nelx // 4)
    
    in_dir = (0.0, 1.0)     # Push downward (into domain)
    out_dir = (0.0, 1.0)    # Same direction = vertical amplifier
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L10',
        'literature_name': 'Microgrip (Jonsmann et al. 1999)',
        'literature_nl_original': False,
        'original_domain': '60x60',
        'original_params': {'VF': 0.25, 'k_in': 0.5, 'k_out': 0.01},
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# L11: Compliant Plier (Frecker et al. 1997)
# ---------------------------------------------------------------------------

def generate_L11_plier(
    nelx: int = 64, nely: int = 32,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Compliant plier: BL + BR corner patches fixed, mid-left input →, mid-right output ←.
    
    Plier-like squeezing: push left jaw right, right jaw moves left (inverter).
    Two support patches form the fulcrum/pivot at the bottom.
    
    Original: 80×40, VF=0.30
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.30, _K_IN, 0.03, vf, k_in, k_out)
    
    patch_r = max(3, nelx // 10)
    bc_nodes_bl = _patch_nodes(0, 0, patch_r, nelx, nely)
    bc_nodes_br = _patch_nodes(nelx, 0, patch_r, nelx, nely)
    bcs = [_make_bc(bc_nodes_bl), _make_bc(bc_nodes_br)]
    
    in_x, in_y = 0, nely // 2        # Mid-left
    out_x, out_y = nelx, nely // 2    # Mid-right
    
    if jitter_amount > 0:
        in_y = _jitter(rng, in_y, jitter_amount, 2, nely - 2)
        out_y = _jitter(rng, out_y, jitter_amount, 2, nely - 2)
    
    in_dir = (1.0, 0.0)     # Push right (into domain)
    out_dir = (-1.0, 0.0)   # Output moves left (plier/inverter)
    
    # Mirror option
    if mirror == 'v':
        in_dir = (-1.0, 0.0)
        out_dir = (1.0, 0.0)
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L11',
        'literature_name': 'Compliant Plier (Frecker et al. 1997)',
        'literature_nl_original': False,
        'original_domain': '80x40',
        'original_params': {'VF': 0.30, 'k_in': 1.0, 'k_out': 0.1},
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# L12: Asymmetric Displacement Inverter (Yin & Ananthasuresh 2001)
# ---------------------------------------------------------------------------

def generate_L12_asymmetric_inverter(
    nelx: int = 64, nely: int = 64,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Asymmetric inverter: BL large patch + TR small roller.
    Left-area input → right, upper-right output → up (orthogonal).
    
    Diagonal force path across domain with orthogonal I/O.
    
    Original: Yin & Ananthasuresh (2001), 60×60, VF=0.30
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.30, _K_IN, 0.03, vf, k_in, k_out)
    
    # BL corner patch (main support) + TR roller
    patch_r = max(4, nelx // 10)
    roller_r = max(2, nelx // 16)
    bl_patch = _patch_nodes(0, nely, patch_r, nelx, nely)
    tr_patch = _patch_nodes(nelx, 0, roller_r, nelx, nely)
    
    bcs = [
        _make_bc(bl_patch, direction=2),     # BL: fully fixed
        _make_bc(tr_patch, direction=1),      # TR: fix y only (roller)
    ]
    
    in_x, in_y = nelx // 6, nely // 2   # Left area
    out_x, out_y = 5 * nelx // 6, nely // 4  # Upper-right area
    
    if jitter_amount > 0:
        in_x = _jitter(rng, in_x, jitter_amount, 3, nelx // 3)
        in_y = _jitter(rng, in_y, jitter_amount, patch_r + 3, nely - patch_r - 3)
        out_y = _jitter(rng, out_y, jitter_amount, 2, nely // 2)
    
    in_dir = (1.0, 0.0)     # Push right
    out_dir = (0.0, -1.0)   # Want upward at output
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L12',
        'literature_name': 'Asymmetric Inverter (Yin & Ananthasuresh 2001)',
        'literature_nl_original': False,
        'original_domain': '60x30',
        'original_params': {'VF': 0.30, 'k_in': 1.0, 'k_out': 0.05},
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# L13: Compliant Actuator (Sigmund 2001)
# ---------------------------------------------------------------------------

def generate_L13_actuator(
    nelx: int = 64, nely: int = 32,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Actuator/transmitter: TL + BL patches as ground, mid-left input → right,
    mid-right output → right (same direction = displacement transmission).
    
    Same-direction I/O creates linear actuator topology with
    thin efficient force paths through the domain.
    
    Original: Sigmund (2001), 80×40, VF=0.20
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.20, _K_IN, 0.02, vf, k_in, k_out)
    
    # Two left-side corner patches (TL + BL)
    patch_r = max(3, nelx // 12)
    tl_patch = _patch_nodes(0, 0, patch_r, nelx, nely)
    bl_patch = _patch_nodes(0, nely, patch_r, nelx, nely)
    bcs = [_make_bc(np.concatenate([tl_patch, bl_patch]))]
    
    # Input between patches, inset from left edge
    inset = max(3, nelx // 16)
    in_x, in_y = inset, nely // 2
    out_x, out_y = nelx - inset, nely // 2
    
    if jitter_amount > 0:
        in_y = _jitter(rng, in_y, jitter_amount, patch_r + 3, nely - patch_r - 3)
        out_y = _jitter(rng, out_y, jitter_amount, 3, nely - 3)
    
    in_dir = (1.0, 0.0)
    out_dir = (1.0, 0.0)    # SAME direction = actuator/transmitter
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L13',
        'literature_name': 'Compliant Actuator (Sigmund 2001)',
        'literature_nl_original': False,
        'original_domain': '80x40',
        'original_params': {'VF': 0.20, 'k_in': 0.5, 'k_out': 0.02},
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# L14: Force Inverter 3:1 Aspect (Bendsøe & Sigmund 2003)
# ---------------------------------------------------------------------------

def generate_L14_long_inverter(
    nelx: int = 96, nely: int = 32,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Elongated inverter: 3:1 aspect ratio, BL + BR bottom patches.
    Bottom-center input pushes up, top-center output moves down (inversion).
    
    Longer domain → longer force paths → more intermediate hinges.
    Vertical I/O in wide domain creates unique multi-hinge topologies.
    
    Original: Bendsøe & Sigmund (2003), 120×40, VF=0.30
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.30, _K_IN, _K_OUT, vf, k_in, k_out)
    
    # Two bottom corner patches (BL + BR)
    patch_r = max(3, nelx // 16)
    bl_patch = _patch_nodes(0, nely, patch_r, nelx, nely)
    br_patch = _patch_nodes(nelx, nely, patch_r, nelx, nely)
    bcs = [_make_bc(np.concatenate([bl_patch, br_patch]))]
    
    # Input at bottom center, output at top center
    inset = max(3, nely // 8)
    in_x, in_y = nelx // 2, nely - inset
    out_x, out_y = nelx // 2, inset
    
    if jitter_amount > 0:
        in_x = _jitter(rng, in_x, jitter_amount, nelx // 4, 3 * nelx // 4)
        out_x = _jitter(rng, out_x, jitter_amount, nelx // 4, 3 * nelx // 4)
    
    in_dir = (0.0, -1.0)    # Push upward
    out_dir = (0.0, 1.0)    # Want downward (inversion)
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L14',
        'literature_name': 'Force Inverter 3:1 (Bendsøe & Sigmund 2003)',
        'literature_nl_original': False,
        'original_domain': '120x40',
        'original_params': {'VF': 0.30, 'k_in': 1.0, 'k_out': 0.1},
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# L15: Compliant OR-Gate (Sigmund 2001)
# ---------------------------------------------------------------------------

def generate_L15_or_gate(
    nelx: int = 64, nely: int = 64,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Compliant OR-gate: two support patches on top edge,
    top-left input ↓ (one of two inputs), bottom-center output ↓.
    
    The mechanism redirects a top-edge force to a bottom-edge motion.
    
    Original: 80×80, VF=0.30
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.20, _K_IN, 0.02, vf, k_in, k_out)
    
    patch_r = max(2, nelx // 16)
    bc1 = _patch_nodes(nelx // 4, nely, patch_r, nelx, nely)
    bc2 = _patch_nodes(3 * nelx // 4, nely, patch_r, nelx, nely)
    bcs = [_make_bc(np.concatenate([bc1, bc2]))]
    
    in_x, in_y = nelx // 4, 0     # Top-left input
    # Output between the two supports but shifted down
    out_x, out_y = nelx // 2, 3 * nely // 4  # Lower area, away from patches
    
    if jitter_amount > 0:
        in_x = _jitter(rng, in_x, jitter_amount, 2, nelx // 2 - 2)
    
    in_dir = (0.0, 1.0)     # Push downward (into domain)
    out_dir = (0.0, 1.0)    # Want downward at output
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L15',
        'literature_name': 'Compliant OR-Gate (Sigmund 2001)',
        'literature_nl_original': False,
        'original_domain': '80x80',
        'original_params': {'VF': 0.30, 'k_in': 0.5, 'k_out': 0.02},
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


# ---------------------------------------------------------------------------
# Selective Compliance / Flexure-inspired generators
# These create mechanisms with clear kinematic behavior inspired by Koppen
# et al. 2022 and Hasse & Campanile 2009.
# ---------------------------------------------------------------------------

def generate_flexure_prismatic(
    nelx: int = 64, nely: int = 64,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Diagonal-support converter: TL + BR diagonal corner patches.
    Bottom-left input pushes up, top-right output wants right.
    
    Diagonal ground placement creates unique X-shaped force paths.
    The perpendicular I/O directions combined with diagonal support
    produce novel hinge topologies not seen in standard setups.
    
    Replaces prismatic flexure (same-dir I/O incompatible with MSE-αSE).
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.30, _K_IN, 0.03, vf, k_in, k_out)
    
    # Diagonal corner patches (TL + BR) — unique combination
    patch_r = max(3, nelx // 12)
    tl_patch = _patch_nodes(0, 0, patch_r, nelx, nely)
    br_patch = _patch_nodes(nelx, nely, patch_r, nelx, nely)
    bcs = [_make_bc(np.concatenate([tl_patch, br_patch]))]
    
    # Input at bottom-left area pushing up, output at top-right pushing right
    in_x = nelx // 4
    in_y = 3 * nely // 4
    out_x = 3 * nelx // 4
    out_y = nely // 4
    
    if jitter_amount > 0:
        in_x = _jitter(rng, in_x, jitter_amount, 3, nelx // 2 - 2)
        in_y = _jitter(rng, in_y, jitter_amount, nely // 2 + 2, nely - patch_r - 2)
        out_x = _jitter(rng, out_x, jitter_amount, nelx // 2 + 2, nelx - 2)
        out_y = _jitter(rng, out_y, jitter_amount, patch_r + 2, nely // 2 - 2)
    
    in_dir = (0.0, -1.0)    # Push upward
    out_dir = (1.0, 0.0)    # Want rightward
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'flexure_prismatic',
        'literature_name': 'Diagonal Converter (Koppen-inspired)',
        'literature_nl_original': False,
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta


def generate_flexure_revolute(
    nelx: int = 64, nely: int = 64,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Revolute joint: bottom edge fixed, top-left pushes down,
    top-right moves up → rotation about center.
    
    Inspired by Koppen et al. 2022, Fig 3c.
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.30, _K_IN, 0.03, vf, k_in, k_out)
    
    bc_nodes = _edge_nodes('bottom', nelx, nely)
    bcs = [_make_bc(bc_nodes)]
    
    in_x, in_y = nelx // 4, nely       # Top-left
    out_x, out_y = 3 * nelx // 4, nely  # Top-right
    
    if jitter_amount > 0:
        in_x = _jitter(rng, in_x, jitter_amount, 5, nelx // 2 - 5)
        out_x = _jitter(rng, out_x, jitter_amount, nelx // 2 + 5, nelx - 5)
    
    in_dir = (0.0, -1.0)    # Push up (into domain) at left
    out_dir = (0.0, 1.0)    # Want down at right = rotation about center
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'flexure_revolute',
        'literature_name': 'Revolute Flexure (Koppen-inspired)',
        'literature_nl_original': False,
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    
    return prob, meta



# ---------------------------------------------------------------------------
# L16: Large-Displacement Path Generator (Pedersen et al. 2001)
# ---------------------------------------------------------------------------

def generate_L16_path_generator(
    nelx: int = 64, nely: int = 64,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Path Generator (Pedersen et al. 2001).
    Corner-fixed domain. Input on side, output follows path.
    Linear approximation: Maximizes output displacement in average path direction.
    
    Setup:
      - Four corners fixed.
      - Input: Left edge, middle.
      - Output: Center or Right edge.
      
    Original paper uses NL solver. Here we provide the linear topology seed.
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.30, _K_IN, _K_OUT, vf, k_in, k_out)
    
    # Four corners fixed
    corner_r = max(2, nelx // 20)
    tl = _patch_nodes(0, 0, corner_r, nelx, nely)
    tr = _patch_nodes(nelx, 0, corner_r, nelx, nely)
    bl = _patch_nodes(0, nely, corner_r, nelx, nely)
    br = _patch_nodes(nelx, nely, corner_r, nelx, nely)
    bcs = [_make_bc(np.concatenate([tl, tr, bl, br]))]
    
    # Input: Left side
    in_x, in_y = max(3, nelx // 16), nely // 2
    
    # Output: Center-Right, moving vertically (path generation proxy)
    out_x, out_y = int(0.75 * nelx), nely // 2
    
    if jitter_amount > 0:
        in_y = _jitter(rng, in_y, jitter_amount, corner_r + 2, nely - corner_r - 2)
        out_x = _jitter(rng, out_x, jitter_amount, nelx // 2, nelx - corner_r - 2)
        out_y = _jitter(rng, out_y, jitter_amount, corner_r + 2, nely - corner_r - 2)

    in_dir = (1.0, 0.0)
    out_dir = (0.0, 1.0) # Vertical motion from horizontal input = curved path
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(out_x, out_y, nelx), out_dir,
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L16',
        'literature_name': 'Path Generator (Pedersen 2001)',
        'literature_nl_original': True,
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    return prob, meta


# ---------------------------------------------------------------------------
# L17: Bistable Switch (Bruns & Sigmund 2004)
# ---------------------------------------------------------------------------

def generate_L17_bistable_switch(
    nelx: int = 80, nely: int = 40,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Bistable Switch (Bruns & Sigmund 2004).
    Fixed ends, central load. Snap-through behavior.
    Linear approximation: Maximizes displacement at center (softens structure).
    
    Setup:
      - Left/Right edges fixed (or roller supported).
      - Input: Top Center, pushing down.
      - Output: Same point, displacement down.
    """
    rng = np.random.RandomState(seed)
    vf_use, k_in_use, k_out_use = _apply_sweep(0.35, _K_IN, _K_OUT, vf, k_in, k_out)
    
    # Both ends fixed (clamped)
    # Original paper likely uses clamped or roller. We'll use clamped patches.
    width = max(2, nelx // 16)
    left_edge = _band_nodes('left', nelx, nely, 0.0, 1.0, width=width)
    right_edge = _band_nodes('right', nelx, nely, 0.0, 1.0, width=width)
    bcs = [_make_bc(np.concatenate([left_edge, right_edge]))]
    
    # Input/Output at Top Center (or Center Center)
    # Paper: "Mid-top, F down"
    in_x, in_y = nelx // 2, 0 # Top edge
    # Inset slightly to avoid trivial boundary issues if using heavy filters
    in_y = max(2, nely // 10) 
    
    if jitter_amount > 0:
        in_x = _jitter(rng, in_x, jitter_amount, width + 5, nelx - width - 5)
    
    in_dir = (0.0, 1.0) # Down (y is positive down in our grid usually? Need to check. Assuming Y+ is down)
    out_dir = (0.0, 1.0) 
    
    prob = _make_problem(
        nelx, nely, bcs,
        _node_at(in_x, in_y, nelx), in_dir,
        _node_at(in_x, in_y, nelx), out_dir, # Maximize own displacement
        vf_use, k_in_use, k_out_use
    )
    
    meta = {
        'literature_id': 'L17',
        'literature_name': 'Bistable Switch (Bruns & Sigmund 2004)',
        'literature_nl_original': True,
        'variant_params': {'VF': vf_use, 'k_in': k_in_use, 'k_out': k_out_use},
    }
    return prob, meta


# ---------------------------------------------------------------------------
# L18: Large-Displacement Inverter (Buhl et al. 2000)
# ---------------------------------------------------------------------------

def generate_L18_large_disp_inverter(
    nelx: int = 64, nely: int = 64,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 0,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Large-Displacement Inverter (Buhl et al. 2000).
    Same topology setup as L1, but intended for NL validation.
    
    Setup:
       - Left edge fixed patches.
       - Input mid-left. Output mid-right.
       - Inverter logic included.
    """
    # Reuse L1 logic, but tag correctly
    prob, meta = generate_L1_force_inverter(
        nelx, nely, vf, k_in, k_out, seed, jitter_amount, mirror
    )
    
    meta['literature_id'] = 'L18'
    meta['literature_name'] = 'Large-Disp Inverter (Buhl et al. 2000)'
    meta['literature_nl_original'] = True
    
    return prob, meta


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LITERATURE_GENERATORS = {
    'L1': generate_L1_force_inverter,
    'L2': generate_L2_gripper,
    'L3': generate_L3_crimper,
    'L4': generate_L4_upper_jaw,
    'L5': generate_L5_amplifier,
    'L6': generate_L6_inverter_projection,
    'L7': generate_L7_robust_inverter,
    'L8': generate_L8_cruncher,
    'L9': generate_L9_bidirectional,
    'L10': generate_L10_microgrip,
    'L11': generate_L11_plier,
    'L12': generate_L12_asymmetric_inverter,
    'L13': generate_L13_actuator,
    'L14': generate_L14_long_inverter,
    'L15': generate_L15_or_gate,
    'L16': generate_L16_path_generator,
    'L17': generate_L17_bistable_switch,
    'L18': generate_L18_large_disp_inverter,
    'flexure_prismatic': generate_flexure_prismatic,
    'flexure_revolute': generate_flexure_revolute,
}


def generate_literature_problem(
    template_id: str,
    nelx: int = 64, nely: int = None,
    vf: float = None, k_in: float = None, k_out: float = None,
    seed: int = None, jitter_amount: int = 3,
    mirror: str = None
) -> Tuple[MechProblem, Dict]:
    """
    Generate a mechanism problem from a literature template.
    
    Args:
        template_id: L1-L15 or flexure_prismatic/flexure_revolute
        nelx, nely: Resolution (nely defaults from template aspect ratio)
        vf: Override volume fraction
        k_in, k_out: Override spring stiffnesses
        seed: Random seed for jitter
        jitter_amount: Position jitter in elements (0 = exact literature setup)
        mirror: None, 'h', 'v', 'hv' for geometric transformations
        
    Returns:
        (MechProblem, metadata_dict)
    """
    if template_id not in LITERATURE_GENERATORS:
        raise ValueError(f"Unknown template: {template_id}. Available: {list(LITERATURE_GENERATORS.keys())}")
    
    gen = LITERATURE_GENERATORS[template_id]
    
    # Default nely from template
    if nely is None:
        # Templates with non-square domains
        aspect_ratios = {
            'L4': (4, 5),      # 80:100
            'L5': (2, 1),      # 80:40
            'L11': (2, 1),     # 80:40
            'L13': (2, 1),     # 80:40
            'L14': (3, 1),     # 120:40
        }
        if template_id in aspect_ratios:
            ax, ay = aspect_ratios[template_id]
            nely = max(16, nelx * ay // ax)
        else:
            nely = nelx
    
    return gen(
        nelx=nelx, nely=nely,
        vf=vf, k_in=k_in, k_out=k_out,
        seed=seed, jitter_amount=jitter_amount,
        mirror=mirror
    )


def generate_random_literature_problem(
    nelx: int = 64,
    seed: int = None,
    vf_range: Tuple[float, float] = (0.15, 0.35),
    k_out_range: Tuple[float, float] = (0.005, 0.08),
) -> Tuple[MechProblem, Dict]:
    """
    Generate a random mechanism from a randomly selected literature template
    with random parametric sweep values.
    
    Uses calibrated spring stiffnesses for our MSE-αSE formulation.
    """
    rng = np.random.RandomState(seed)
    
    # Pick random template
    templates = list(LITERATURE_GENERATORS.keys())
    template_id = templates[rng.randint(len(templates))]
    
    # Random sweep parameters (calibrated for our formulation)
    vf = rng.uniform(*vf_range)
    k_out = rng.uniform(*k_out_range)
    k_in = _K_IN  # Always use our calibrated input spring
    
    # Random mirror
    mirror = rng.choice([None, 'h', 'v', 'hv'])
    
    # Random jitter
    jitter = rng.randint(0, max(3, nelx // 16))
    
    return generate_literature_problem(
        template_id=template_id,
        nelx=nelx,
        vf=vf, k_in=k_in, k_out=k_out,
        seed=seed, jitter_amount=jitter,
        mirror=mirror
    )
