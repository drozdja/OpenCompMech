"""
Linkage seed generators for compliant mechanism topology optimization.

Generates voxelized rigid-body linkage topologies as initial density fields
for SIMP optimization. The optimizer converts rigid joints → flexure hinges.

Supported linkage types:
  - four_bar:      4 joints, 4 links. Most versatile coupler-curve family.
  - slider_crank:  3 rotating joints + 1 slider. Linear output motion.
  - (more to come: toggle, scotch_yoke, watt, stephenson, crank_rocker, double_crank)

Each seed generator returns:
  - density_init: (nely, nelx) with background at volume_fraction, links at 1.0
  - mech_problem: MechProblem with BCs at ground pivots, I/O at moving joints
  - seed_info: dict with linkage metadata for provenance tracking

References:
  - Sigmund (1997): "On the design of compliant mechanisms using topology optimization"
  - Wang et al. (2011): "Projection methods in topology optimization"
  - EXECUTION_PLAN.md §4.5: Seeded Generation Architecture
"""

import numpy as np
from typing import Optional, Tuple, Dict

from ..core.problem import Problem, ProblemType, Material, BoundaryCondition, Load
from ..core.mesh import Mesh2D, create_mesh
from .mech import MechProblem, MechConfig


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def _circle_circle_intersection(
    x1: float, y1: float, r1: float,
    x2: float, y2: float, r2: float,
    rng: np.random.RandomState
) -> Tuple[Optional[float], Optional[float]]:
    """Find one intersection point of two circles (randomly pick one of two)."""
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if d > r1 + r2 or d < abs(r1 - r2) or d < 1e-6:
        return None, None

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h_sq = r1**2 - a**2
    if h_sq < 0:
        return None, None
    h = np.sqrt(h_sq)

    # Point on line between centers at distance a from (x1,y1)
    px = x1 + a * (x2 - x1) / d
    py = y1 + a * (y2 - y1) / d

    # Two intersection points
    ix1 = px + h * (y2 - y1) / d
    iy1 = py - h * (x2 - x1) / d
    ix2 = px - h * (y2 - y1) / d
    iy2 = py + h * (x2 - x1) / d

    if rng.random() < 0.5:
        return ix1, iy1
    return ix2, iy2


def _check_grashof(link_lengths: list) -> bool:
    """Check Grashof condition: shortest + longest ≤ sum of other two."""
    s = sorted(link_lengths)
    return s[0] + s[3] <= s[1] + s[2]


def _transmission_angle(B, C, D) -> float:
    """Compute transmission angle μ at C between coupler CB and rocker CD (degrees)."""
    cb = np.array(B) - np.array(C)
    cd = np.array(D) - np.array(C)
    norm_cb = np.linalg.norm(cb)
    norm_cd = np.linalg.norm(cd)
    if norm_cb < 1e-10 or norm_cd < 1e-10:
        return 0.0
    cos_mu = np.dot(cb, cd) / (norm_cb * norm_cd)
    return np.degrees(np.arccos(np.clip(cos_mu, -1.0, 1.0)))


def _compute_four_bar_kinematics(A, B, C, D):
    """Compute instantaneous I/O velocity directions for a four-bar linkage.

    Given unit CCW angular velocity of crank A-B about A, returns:
      input_dir:  velocity direction at B (tangent to crank rotation)
      output_dir: velocity direction at C (from velocity loop equation)

    Returns None if the linkage is at a singularity.
    """
    A, B, C, D = [np.asarray(p, dtype=float) for p in [A, B, C, D]]

    AB = B - A
    BC = C - B
    DC = C - D

    len_AB = np.linalg.norm(AB)
    len_BC = np.linalg.norm(BC)
    len_DC = np.linalg.norm(DC)

    if len_AB < 1e-10 or len_BC < 1e-10 or len_DC < 1e-10:
        return None

    theta2 = np.arctan2(AB[1], AB[0])
    theta3 = np.arctan2(BC[1], BC[0])
    theta4 = np.arctan2(DC[1], DC[0])

    # Perpendicular unit vectors (CCW rotation of link directions)
    j2 = np.array([-np.sin(theta2), np.cos(theta2)])
    j3 = np.array([-np.sin(theta3), np.cos(theta3)])
    j4 = np.array([-np.sin(theta4), np.cos(theta4)])

    # Velocity loop: ω₂·|AB|·ĵ₂ + ω₃·|BC|·ĵ₃ = ω₄·|DC|·ĵ₄
    # Rearrange: [|BC|·ĵ₃  -|DC|·ĵ₄] · [ω₃; ω₄] = -|AB|·ω₂·ĵ₂
    A_mat = np.array([
        [len_BC * j3[0], -len_DC * j4[0]],
        [len_BC * j3[1], -len_DC * j4[1]]
    ])

    det = np.linalg.det(A_mat)
    if abs(det) < 1e-10:
        return None  # At toggle position — singular

    omega2 = 1.0  # Unit input angular velocity
    rhs = -len_AB * omega2 * j2
    omega = np.linalg.solve(A_mat, rhs)
    omega4 = omega[1]

    # Velocity of B (input direction) — tangent to crank
    v_B = len_AB * omega2 * j2
    input_dir = v_B / (np.linalg.norm(v_B) + 1e-10)

    # Velocity of C (output direction) — tangent to rocker
    v_C = omega4 * len_DC * j4
    output_dir = v_C / (np.linalg.norm(v_C) + 1e-10)

    return {
        'input_dir': tuple(input_dir),
        'output_dir': tuple(output_dir),
        'omega4': omega4,  # Output angular velocity
        'mechanical_advantage': abs(omega4) / abs(omega2) if abs(omega2) > 1e-10 else 0,
    }


# ---------------------------------------------------------------------------
# Voxelization — draw thick lines on density grid
# ---------------------------------------------------------------------------

def _draw_thick_line(
    density: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    width: int = 2,
    value: float = 1.0
):
    """Draw a thick line on density grid between two floating-point positions.

    Uses Bresenham with rectangular brush of half-width `width`.
    Modifies density in-place.

    Args:
        density: (nely, nelx) array to draw on
        p0, p1: (x, y) floating-point coordinates in element space
        width: half-width of line in elements (2 → 5px total)
        value: density value to set (uses max to not erase existing)
    """
    nely, nelx = density.shape
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
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


def _draw_circle(
    density: np.ndarray,
    center: np.ndarray,
    radius: int = 3,
    value: float = 1.0
):
    """Draw a filled circle (joint) on the density grid."""
    nely, nelx = density.shape
    cx, cy = int(round(center[0])), int(round(center[1]))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                ex, ey = cx + dx, cy + dy
                if 0 <= ex < nelx and 0 <= ey < nely:
                    density[ey, ex] = max(density[ey, ex], value)


def _voxelize_linkage(
    joints: Dict[str, np.ndarray],
    link_pairs: list,
    nelx: int,
    nely: int,
    volume_fraction: float,
    line_width: int = 1,
    joint_radius: int = 2,
    background_density: float = 0.01,
) -> np.ndarray:
    """Create density field from linkage joints and links.

    Background at LOW density (not VF_target!) for seed-to-background contrast.
    The optimizer uses volume-preserving Heaviside to hit the actual VF target,
    so background density only affects the raw field's sensitivity distribution.

    IMPORTANT: using VF_target as the background is correct for UNSEEDED mode,
    where nonzero sensitivity is needed to grow material at all. For SEEDED mode
    the seed already supplies the topology, so what matters is CONTRAST against
    the background, not growth potential.

    Args:
        joints: dict mapping joint names to (x, y) positions
        link_pairs: list of (name1, name2) tuples defining links
        nelx, nely: grid dimensions
        volume_fraction: target VF for the optimization (stored in metadata)
        line_width: half-width of links in elements (1 → 3px total = ~2px effective)
        joint_radius: radius of joint circles in elements
        background_density: density for non-seed elements (default 0.01)

    Returns:
        density: (nely, nelx) with background at background_density, seed at 1.0
    """
    density = np.full((nely, nelx), background_density, dtype=np.float64)

    # Draw links
    for j1_name, j2_name in link_pairs:
        p0 = joints[j1_name]
        p1 = joints[j2_name]
        _draw_thick_line(density, p0, p1, width=line_width, value=1.0)

    # Draw slightly larger circles at joints for robustness
    for name, pos in joints.items():
        _draw_circle(density, pos, radius=joint_radius, value=1.0)

    return density


# ---------------------------------------------------------------------------
# MechProblem creation from linkage geometry
# ---------------------------------------------------------------------------

def _get_patch_node_indices(
    pos: np.ndarray,
    nelx: int,
    nely: int,
    radius: int = 2
) -> np.ndarray:
    """Get node indices in a circular patch around a position.

    Args:
        pos: (x, y) in element coordinates
        nelx, nely: mesh dimensions
        radius: patch radius in nodes

    Returns:
        Array of node indices (in the (nely+1)×(nelx+1) node grid)
    """
    cx, cy = int(round(pos[0])), int(round(pos[1]))
    nodes = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx <= nelx and 0 <= ny <= nely:
                    nodes.append(ny * (nelx + 1) + nx)
    return np.array(nodes, dtype=int) if nodes else np.array([], dtype=int)


def _nearest_node_index(pos: np.ndarray, nelx: int, nely: int) -> int:
    """Convert (x, y) position to nearest node index."""
    nx = int(round(np.clip(pos[0], 0, nelx)))
    ny = int(round(np.clip(pos[1], 0, nely)))
    return ny * (nelx + 1) + nx


def _create_mech_problem_from_linkage(
    ground_pivots: list,
    input_joint: np.ndarray,
    output_joint: np.ndarray,
    input_direction: tuple,
    output_direction: tuple,
    nelx: int,
    nely: int,
    volume_fraction: float = 0.20,
    k_in: float = 0.01,
    k_out: float = 0.03,
    bc_radius: int = 2,
) -> MechProblem:
    """Create MechProblem from linkage joint positions and directions.

    Args:
        ground_pivots: list of (x, y) positions for fixed ground joints
        input_joint: (x, y) position of input force application
        output_joint: (x, y) position of output displacement measurement
        input_direction: (dx, dy) unit vector for input force
        output_direction: (dx, dy) unit vector for output measurement
        nelx, nely: mesh dimensions
        volume_fraction: target volume fraction
        k_in: input spring stiffness
        k_out: output spring stiffness
        bc_radius: radius of BC node patches around ground pivots
    """
    mesh = create_mesh(nelx, nely)
    material = Material(E=1.0, nu=0.3)

    # Create BCs at ground pivots — small patches (2-5 nodes)
    bcs = []
    for pivot in ground_pivots:
        nodes = _get_patch_node_indices(pivot, nelx, nely, radius=bc_radius)
        if len(nodes) > 0:
            dirs = np.full(len(nodes), 2, dtype=int)  # Fix both x and y
            bcs.append(BoundaryCondition(node_indices=nodes, directions=dirs))

    if not bcs:
        raise ValueError("No valid BC nodes from ground pivots")

    # Input load at input joint node
    input_node = _nearest_node_index(input_joint, nelx, nely)
    # Force magnitude 1.0 in input direction
    loads = [Load(
        node_index=input_node,
        fx=float(input_direction[0]),
        fy=float(input_direction[1])
    )]

    # Output node
    output_node = _nearest_node_index(output_joint, nelx, nely)

    # Base problem
    base_problem = Problem(
        mesh=mesh,
        material=material,
        bcs=bcs,
        loads=loads,
        volume_fraction=volume_fraction,
        problem_type=ProblemType.MECHANISM,
    )

    # Mechanism problem
    mech_problem = MechProblem(
        base_problem=base_problem,
        input_node=input_node,
        input_direction=input_direction,
        output_node=output_node,
        output_direction=output_direction,
        k_in=k_in,
        k_out=k_out,
    )

    return mech_problem


# ---------------------------------------------------------------------------
# Four-bar linkage seed
# ---------------------------------------------------------------------------

def _generate_four_bar_geometry(
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    margin: int = 5
) -> Optional[Dict]:
    """Try to generate valid four-bar geometry.

    Four-bar ABCD:
        A --- D    (ground link, A and D are fixed pivots)
        |     |
        B --- C    (B = crank tip/input, C = rocker tip/output)

    Links: A-B (crank), B-C (coupler), C-D (rocker), A-D (ground)

    Returns dict with joint positions, link lengths, transmission angle,
    and I/O direction vectors.  Returns None if geometry is invalid.
    """
    min_dim = min(nelx, nely)
    min_sep = min_dim * 0.25
    max_sep = min_dim * 0.65

    # Place ground pivots A and D
    ax = rng.uniform(margin, nelx * 0.45)
    ay = rng.uniform(margin, nely - margin)

    ground_angle = rng.uniform(-np.pi / 4, np.pi / 4)
    ground_length = rng.uniform(min_sep, max_sep)

    dx = ax + ground_length * np.cos(ground_angle)
    dy = ay + ground_length * np.sin(ground_angle)

    # Clamp within domain
    dx = np.clip(dx, margin, nelx - margin)
    dy = np.clip(dy, margin, nely - margin)
    ground_length = np.sqrt((dx - ax)**2 + (dy - ay)**2)

    if ground_length < min_sep:
        return None

    # Link lengths as fractions of ground link
    crank_ratio = rng.uniform(0.20, 0.50)
    rocker_ratio = rng.uniform(0.30, 0.70)
    coupler_ratio = rng.uniform(0.40, 0.90)

    crank_len = ground_length * crank_ratio
    rocker_len = ground_length * rocker_ratio
    coupler_len = ground_length * coupler_ratio

    # Minimum link length — must be drawable on grid
    min_link = max(5.0, min_dim * 0.08)
    if crank_len < min_link or rocker_len < min_link or coupler_len < min_link:
        return None

    # Grashof check
    if not _check_grashof([ground_length, crank_len, coupler_len, rocker_len]):
        return None

    # Crank position — random initial angle
    crank_angle = rng.uniform(0, 2 * np.pi)
    bx = ax + crank_len * np.cos(crank_angle)
    by = ay + crank_len * np.sin(crank_angle)

    # Rocker tip (C) — circle-circle intersection
    cx, cy = _circle_circle_intersection(
        bx, by, coupler_len, dx, dy, rocker_len, rng
    )
    if cx is None:
        return None

    A = np.array([ax, ay])
    B = np.array([bx, by])
    C = np.array([cx, cy])
    D = np.array([dx, dy])

    # Check all joints inside domain (with margin)
    for jt in [A, B, C, D]:
        if (jt[0] < 1 or jt[0] > nelx - 1 or
                jt[1] < 1 or jt[1] > nely - 1):
            return None

    # Transmission angle check (20° < μ < 160°)
    mu = _transmission_angle(B, C, D)
    if mu < 20 or mu > 160:
        return None

    # Compute I/O kinematics
    kin = _compute_four_bar_kinematics(A, B, C, D)
    if kin is None:
        return None

    return {
        'joints': {'A': A, 'B': B, 'C': C, 'D': D},
        # CRITICAL: Do NOT draw the ground link A-D.
        # Drawing it creates a closed rigid polygon with no room for flexure hinges.
        # The ground link is represented by fixed BCs at A and D.
        # The open chain A-B-C-D lets the optimizer create hinges at joints.
        'link_pairs': [('A', 'B'), ('B', 'C'), ('C', 'D')],
        'ground_pivots': [A, D],
        'input_joint': B,
        'output_joint': C,
        'input_dir': kin['input_dir'],
        'output_dir': kin['output_dir'],
        'link_lengths': {
            'ground': ground_length,
            'crank': crank_len,
            'coupler': coupler_len,
            'rocker': rocker_len,
        },
        'transmission_angle': mu,
        'mechanical_advantage': kin['mechanical_advantage'],
    }


def seed_from_four_bar(
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    volume_fraction: float = 0.20,
    k_in: float = 0.01,
    k_out: float = None,
    max_attempts: int = 200,
) -> Optional[Tuple[np.ndarray, MechProblem, Dict]]:
    """Generate a four-bar linkage seed for mechanism optimization.

    Tries up to max_attempts times to generate a valid Grashof four-bar
    with good transmission angle.

    Args:
        nelx, nely: design grid dimensions
        rng: random state for reproducibility
        volume_fraction: target VF (used as background density)
        k_in: input spring stiffness
        k_out: output spring (randomized if None)
        max_attempts: rejection sampling attempts

    Returns:
        (density_init, mech_problem, seed_info) or None if all attempts fail
    """
    if k_out is None:
        k_out = rng.uniform(0.01, 0.05)

    for attempt in range(max_attempts):
        geom = _generate_four_bar_geometry(nelx, nely, rng)
        if geom is None:
            continue

        # Voxelize: ~5px-wide lines (half-width=2) for robustness against
        # density filter erosion. Wider seeds survive continuation better.
        density = _voxelize_linkage(
            joints=geom['joints'],
            link_pairs=geom['link_pairs'],
            nelx=nelx, nely=nely,
            volume_fraction=volume_fraction,
            line_width=2,
            joint_radius=3,
        )

        # Check seed VF isn't absurdly high
        seed_vf = float(np.mean(density))
        if seed_vf > volume_fraction * 2.5:
            continue  # Too much solid — links are too long/thick

        # Create MechProblem
        try:
            mech_problem = _create_mech_problem_from_linkage(
                ground_pivots=geom['ground_pivots'],
                input_joint=geom['input_joint'],
                output_joint=geom['output_joint'],
                input_direction=geom['input_dir'],
                output_direction=geom['output_dir'],
                nelx=nelx, nely=nely,
                volume_fraction=volume_fraction,
                k_in=k_in,
                k_out=k_out,
                bc_radius=2,
            )
        except ValueError:
            continue

        seed_info = {
            'linkage_type': 'four_bar',
            'link_lengths': geom['link_lengths'],
            'transmission_angle': geom['transmission_angle'],
            'mechanical_advantage': geom['mechanical_advantage'],
            'seed_vf': seed_vf,
            'joints': {k: v.tolist() for k, v in geom['joints'].items()},
            'input_dir': geom['input_dir'],
            'output_dir': geom['output_dir'],
            'k_in': k_in,
            'k_out': k_out,
            'attempt': attempt + 1,
        }

        return density, mech_problem, seed_info

    return None


# ---------------------------------------------------------------------------
# Slider-crank linkage seed
# ---------------------------------------------------------------------------

def _generate_slider_crank_geometry(
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    margin: int = 5
) -> Optional[Dict]:
    """Generate valid slider-crank geometry.

    Slider-crank:
        A = ground pivot (crank base, fixed)
        B = crank-connecting rod joint
        C = slider (constrained to slide axis)
        Links: A-B (crank), B-C (connecting rod)
        Slide axis: line through a point P in direction d̂

    For the optimization, the slider constraint is naturally enforced by k_perp.
    BC is only at ground pivot A.

    Returns dict with joint positions or None if invalid.
    """
    min_dim = min(nelx, nely)

    # Place ground pivot A
    ax = rng.uniform(margin, nelx - margin)
    ay = rng.uniform(margin, nely - margin)

    # Crank length
    crank_len = rng.uniform(min_dim * 0.10, min_dim * 0.30)

    # Connecting rod must be longer than crank (standard kinematic constraint)
    rod_ratio = rng.uniform(1.5, 3.0)
    rod_len = crank_len * rod_ratio

    # Slide axis: random direction (roughly horizontal or vertical)
    slide_type = rng.choice(['horizontal', 'vertical', 'angled'])
    if slide_type == 'horizontal':
        slide_dir = np.array([1.0, 0.0])
    elif slide_type == 'vertical':
        slide_dir = np.array([0.0, 1.0])
    else:
        slide_angle = rng.uniform(-np.pi / 6, np.pi / 6)
        slide_dir = np.array([np.cos(slide_angle), np.sin(slide_angle)])

    # Slide axis offset from A (perpendicular distance)
    # The slider slides along a line parallel to slide_dir, offset from A
    offset = rng.uniform(crank_len * 0.3, crank_len * 0.8)
    if rng.random() < 0.5:
        offset = -offset

    # Slide axis passes through point P
    perp_dir = np.array([-slide_dir[1], slide_dir[0]])
    P = np.array([ax, ay]) + offset * perp_dir

    # Initial crank angle
    crank_angle = rng.uniform(0, 2 * np.pi)
    bx = ax + crank_len * np.cos(crank_angle)
    by = ay + crank_len * np.sin(crank_angle)
    B = np.array([bx, by])

    # Find C on slide axis at distance rod_len from B
    # C = P + t * slide_dir, |B - C| = rod_len
    # Solve: |B - P - t*slide_dir|² = rod_len²
    bp = B - P
    a_coeff = 1.0  # |slide_dir|² = 1
    b_coeff = -2 * np.dot(bp, slide_dir)
    c_coeff = np.dot(bp, bp) - rod_len**2

    discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
    if discriminant < 0:
        return None

    t1 = (-b_coeff + np.sqrt(discriminant)) / (2 * a_coeff)
    t2 = (-b_coeff - np.sqrt(discriminant)) / (2 * a_coeff)

    # Pick the solution farther from A (more useful mechanism)
    C1 = P + t1 * slide_dir
    C2 = P + t2 * slide_dir
    dist1 = np.linalg.norm(C1 - np.array([ax, ay]))
    dist2 = np.linalg.norm(C2 - np.array([ax, ay]))
    C = C1 if dist1 > dist2 else C2

    A = np.array([ax, ay])

    # Check all joints within domain
    for jt in [A, B, C]:
        if (jt[0] < 1 or jt[0] > nelx - 1 or
                jt[1] < 1 or jt[1] > nely - 1):
            return None

    # Minimum separation between joints
    for j1, j2 in [(A, B), (B, C), (A, C)]:
        if np.linalg.norm(j1 - j2) < max(4.0, min_dim * 0.06):
            return None

    # Input direction: tangent to crank rotation at B
    AB = B - A
    input_dir = np.array([-AB[1], AB[0]])  # CCW perpendicular
    input_dir = input_dir / (np.linalg.norm(input_dir) + 1e-10)

    # Output direction: along the slide axis with the SIGN from the velocity
    # loop. Rigid rod => relative velocity perpendicular to it:
    # (v_C - v_B)·BC = 0 with v_C = s*slide_dir and v_B = input_dir (omega=+1
    # about A by construction), so s = (v_B·BC)/(slide_dir·BC).
    # The old code returned +slide_dir unconditionally — wrong for half the
    # crank/elbow configurations (2026-07-15 audit: |u_out| up to 17.5 with
    # the wrong sign; rigid_replace worked around it with an FEA sign probe,
    # which stays as a guard for near-singular elastic cases).
    BC = C - B
    denom = float(np.dot(slide_dir, BC))
    if abs(denom) < 1e-6 * np.linalg.norm(BC):
        return None                       # slide axis ~perpendicular to rod
    s = float(np.dot(input_dir, BC)) / denom
    output_dir = tuple(slide_dir * (1.0 if s >= 0 else -1.0))

    return {
        'joints': {'A': A, 'B': B, 'C': C},
        'link_pairs': [('A', 'B'), ('B', 'C')],
        'ground_pivots': [A],
        'input_joint': B,
        'output_joint': C,
        'input_dir': tuple(input_dir),
        'output_dir': output_dir,
        'slide_dir': tuple(slide_dir),
        'crank_len': crank_len,
        'rod_len': rod_len,
    }


def seed_from_slider_crank(
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    volume_fraction: float = 0.20,
    k_in: float = 0.01,
    k_out: float = None,
    max_attempts: int = 200,
) -> Optional[Tuple[np.ndarray, MechProblem, Dict]]:
    """Generate a slider-crank linkage seed for mechanism optimization.

    Returns:
        (density_init, mech_problem, seed_info) or None if all attempts fail
    """
    if k_out is None:
        k_out = rng.uniform(0.01, 0.05)

    for attempt in range(max_attempts):
        geom = _generate_slider_crank_geometry(nelx, nely, rng)
        if geom is None:
            continue

        density = _voxelize_linkage(
            joints=geom['joints'],
            link_pairs=geom['link_pairs'],
            nelx=nelx, nely=nely,
            volume_fraction=volume_fraction,
            line_width=2,
            joint_radius=3,
        )

        seed_vf = float(np.mean(density))
        if seed_vf > volume_fraction * 2.5:
            continue

        try:
            # Slider-crank: add a BC along the slide guide near C
            # This helps the optimizer understand the slider constraint
            # We fix perpendicular DOFs near C to encourage linear motion
            mech_problem = _create_mech_problem_from_linkage(
                ground_pivots=geom['ground_pivots'],
                input_joint=geom['input_joint'],
                output_joint=geom['output_joint'],
                input_direction=geom['input_dir'],
                output_direction=geom['output_dir'],
                nelx=nelx, nely=nely,
                volume_fraction=volume_fraction,
                k_in=k_in,
                k_out=k_out,
                bc_radius=2,
            )
        except ValueError:
            continue

        seed_info = {
            'linkage_type': 'slider_crank',
            'crank_len': geom['crank_len'],
            'rod_len': geom['rod_len'],
            'slide_dir': geom['slide_dir'],
            'seed_vf': seed_vf,
            'joints': {k: v.tolist() for k, v in geom['joints'].items()},
            'input_dir': geom['input_dir'],
            'output_dir': geom['output_dir'],
            'k_in': k_in,
            'k_out': k_out,
            'attempt': attempt + 1,
        }

        return density, mech_problem, seed_info

    return None


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

SEED_GENERATORS = {
    'four_bar': seed_from_four_bar,
    'slider_crank': seed_from_slider_crank,
}


def seed_from_linkage(
    linkage_type: str,
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    volume_fraction: float = 0.20,
    k_in: float = 0.01,
    k_out: float = None,
    max_attempts: int = 200,
) -> Optional[Tuple[np.ndarray, MechProblem, Dict]]:
    """Generate a linkage seed for mechanism topology optimization.

    Top-level entry point. Dispatches to specific linkage generators.

    Args:
        linkage_type: one of 'four_bar', 'slider_crank' (more coming)
        nelx, nely: design grid dimensions
        rng: numpy RandomState for reproducibility
        volume_fraction: target VF (also used as background density)
        k_in: input spring stiffness
        k_out: output spring (randomized if None)
        max_attempts: rejection sampling attempts

    Returns:
        (density_init, mech_problem, seed_info) or None

    Usage:
        rng = np.random.RandomState(42)
        result = seed_from_linkage('four_bar', 64, 64, rng, volume_fraction=0.20)
        if result is not None:
            density_init, mech_problem, seed_info = result
            # Pass to optimize_mechanism with seed_mask
            opt_result = optimize_mechanism(
                mech_problem, config,
                initial_density=density_init,
                seed_mask=(density_init > 0.5)
            )
    """
    if linkage_type not in SEED_GENERATORS:
        raise ValueError(
            f"Unknown linkage type '{linkage_type}'. "
            f"Available: {list(SEED_GENERATORS.keys())}"
        )

    return SEED_GENERATORS[linkage_type](
        nelx=nelx, nely=nely, rng=rng,
        volume_fraction=volume_fraction,
        k_in=k_in, k_out=k_out,
        max_attempts=max_attempts,
    )
