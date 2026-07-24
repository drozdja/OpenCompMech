"""
Validation functions for topology optimization results.

Critical: Uses 4-connectivity (Von Neumann neighborhood) - NO diagonals!
"""

import numpy as np
from scipy.ndimage import label, binary_erosion, binary_dilation, minimum_filter
from typing import Tuple, Dict, Any

# 4-connectivity (Von Neumann) structuring element used throughout.
_CROSS4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def check_connectivity(
    density: np.ndarray,
    threshold: float = 0.5
) -> Tuple[bool, int]:
    """
    Check if solid structure is a single connected component.
    
    Uses 4-connectivity (Von Neumann neighborhood):
    - Only horizontal and vertical neighbors count
    - Diagonal connections do NOT count
    
    This matches real-world manufacturability where diagonal-only
    connections would create stress singularities.
    
    Args:
        density: (nely, nelx) element densities [0, 1]
        threshold: Binarization threshold
    
    Returns:
        is_connected: True if single component
        n_components: Number of connected components
    """
    # Binarize
    binary = (density > threshold).astype(int)
    
    # Check if there's any solid material
    if np.sum(binary) == 0:
        return False, 0
    
    # 4-connectivity structure (Von Neumann)
    # NO diagonals - only up/down/left/right
    structure = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=int)
    
    labeled, n_components = label(binary, structure=structure)
    
    return n_components == 1, n_components


def check_bc_connectivity(
    density: np.ndarray,
    fixed_nodes: np.ndarray,
    nelx: int,
    nely: int,
    threshold: float = 0.5
) -> Tuple[bool, int]:
    """
    Check if solid structure connects to fixed boundary condition nodes.
    
    A valid mechanism MUST have a load path to the supports. If the solid
    structure floats disconnected from the fixed nodes, it's invalid.
    
    Args:
        density: (nely, nelx) element densities
        fixed_nodes: Array of fixed node indices
        nelx: Number of elements in x direction
        nely: Number of elements in y direction
        threshold: Binarization threshold
    
    Returns:
        is_connected_to_bc: True if solid connects to at least one fixed node
        n_connected_fixed: Number of fixed nodes connected to solid
    """
    binary = (density > threshold).astype(int)
    
    if np.sum(binary) == 0:
        return False, 0
    
    # Convert node indices to element indices
    # Each node (i, j) is at the corner of up to 4 elements
    # Node index = y * (nelx + 1) + x, so:
    # x = node_idx % (nelx + 1)
    # y = node_idx // (nelx + 1)
    
    # Find which elements are adjacent to fixed nodes
    bc_adjacent_elements = set()
    
    for node_idx in fixed_nodes:
        node_x = node_idx % (nelx + 1)
        node_y = node_idx // (nelx + 1)
        
        # Elements adjacent to this node (up to 4)
        # Element (ey, ex) has corners at nodes:
        #   (ex, ey), (ex+1, ey), (ex, ey+1), (ex+1, ey+1)
        
        # So node (nx, ny) touches elements:
        #   (nx-1, ny-1), (nx, ny-1), (nx-1, ny), (nx, ny)
        for dy in [-1, 0]:
            for dx in [-1, 0]:
                ex = node_x + dx
                ey = node_y + dy
                if 0 <= ex < nelx and 0 <= ey < nely:
                    bc_adjacent_elements.add((ey, ex))
    
    # Check if any BC-adjacent element is solid
    n_connected = 0
    for (ey, ex) in bc_adjacent_elements:
        if binary[ey, ex] > 0:
            n_connected += 1
    
    return n_connected > 0, n_connected


def _adjacent_element_labels(labels: np.ndarray, nodes, nelx: int, nely: int):
    """Connected-component labels on solid elements touching ``nodes``."""
    out = set()
    for node_idx in nodes:
        node_x = int(node_idx) % (nelx + 1)
        node_y = int(node_idx) // (nelx + 1)
        for dy in (-1, 0):
            for dx in (-1, 0):
                ex, ey = node_x + dx, node_y + dy
                if 0 <= ex < nelx and 0 <= ey < nely:
                    lab = int(labels[ey, ex])
                    if lab:
                        out.add(lab)
    return out


def check_mechanism_path_connectivity(
    density: np.ndarray,
    fixed_nodes,
    input_node: int,
    output_node: int,
    nelx: int,
    nely: int,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Require support, input port, and output port to share one solid body.

    A global "one component" check and separate port occupancy checks are not
    enough: a generated body can be internally connected while not touching a
    port, or a port can sit on a separate small body.  This is the strict gate
    used for both production data and generated proposals.
    """
    mask = np.asarray(density) > threshold
    labels, n_components = label(mask, structure=_CROSS4)
    groups = {
        "support": _adjacent_element_labels(labels, fixed_nodes, nelx, nely),
        "input": _adjacent_element_labels(labels, [input_node], nelx, nely),
        "output": _adjacent_element_labels(labels, [output_node], nelx, nely),
    }
    common = set.intersection(*groups.values()) if all(groups.values()) else set()
    return {
        "passed": bool(common),
        "n_components": int(n_components),
        "support_labels": sorted(groups["support"]),
        "input_labels": sorted(groups["input"]),
        "output_labels": sorted(groups["output"]),
        "shared_labels": sorted(common),
    }


def check_volume_fraction(
    density: np.ndarray,
    target: float,
    tolerance: float = 0.10,
    mask: np.ndarray = None
) -> Tuple[bool, float]:
    """
    Check if volume fraction is within tolerance of target.
    
    Args:
        density: (nely, nelx) element densities
        target: Target volume fraction
        tolerance: Allowed relative deviation (e.g., 0.10 = 10%)
        mask: Optional domain mask (only count masked elements)
    
    Returns:
        is_valid: True if within tolerance
        actual_vf: Actual volume fraction
    """
    if mask is not None:
        active_density = density[mask]
    else:
        active_density = density.flatten()
    
    actual_vf = float(np.mean(active_density))
    
    # Check relative deviation
    relative_error = abs(actual_vf - target) / target
    is_valid = relative_error <= tolerance
    
    return is_valid, actual_vf


def check_gray_fraction(
    density: np.ndarray,
    max_gray: float = 0.15,
    gray_range: Tuple[float, float] = (0.1, 0.9)
) -> Tuple[bool, float]:
    """
    Check if intermediate (gray) density fraction is acceptable.
    
    Gray elements indicate incomplete convergence and may cause
    issues with manufacturability.
    
    Args:
        density: (nely, nelx) element densities
        max_gray: Maximum allowed gray fraction
        gray_range: Range considered "gray" (not solid, not void)
    
    Returns:
        is_valid: True if gray fraction below threshold
        gray_fraction: Fraction of elements in gray range
    """
    low, high = gray_range
    gray_mask = (density > low) & (density < high)
    gray_fraction = float(np.mean(gray_mask))
    
    return gray_fraction <= max_gray, gray_fraction


def check_minimum_feature_size(
    density: np.ndarray,
    min_size: int = 2,
    threshold: float = 0.5
) -> Tuple[bool, int]:
    """
    Check if features meet minimum size requirement.
    
    Uses erosion + dilation to identify thin features.
    
    Args:
        density: (nely, nelx) element densities
        min_size: Minimum feature size in elements
        threshold: Binarization threshold
    
    Returns:
        is_valid: True if no sub-minimum features
        smallest_feature: Estimated smallest feature size
    """
    binary = (density > threshold).astype(bool)
    
    if np.sum(binary) == 0:
        return False, 0
    
    # Create circular structuring element
    struct = np.ones((min_size, min_size), dtype=bool)
    
    # Erode then dilate (opening)
    eroded = binary_erosion(binary, structure=struct)
    
    # If opening removes everything, features are too thin
    if np.sum(eroded) == 0:
        return False, min_size - 1
    
    # Estimate smallest feature by counting what was lost
    lost_fraction = 1 - np.sum(eroded) / np.sum(binary)
    
    # Heuristic: if less than 50% lost, features are likely >= min_size
    is_valid = lost_fraction < 0.5
    
    # Rough estimate of smallest feature
    smallest = min_size if is_valid else min_size - 1

    return is_valid, smallest


def detect_hinges(
    density: np.ndarray,
    threshold: float = 0.5,
    min_neck_px: int = 2,
) -> Dict[str, Any]:
    """
    Detect single-node hinges and thin necks that pass standard connectivity
    but are physically unsound.

    Motivation: a design made of stiff regions joined by a single-pixel "hinge"
    is a single 4-connected component (passes ``check_connectivity``) and yields
    a large output displacement under *linear* FEA — but the displacement is a
    point-flexure artifact that linear elasticity over-predicts and that is not
    manufacturable. ``check_minimum_feature_size`` is a soft global heuristic and
    misses a thin neck inside an otherwise chunky body. This function targets the
    neck/hinge geometry directly.

    Two complementary signals:

    1. **Articulation (bridge) pixels** — solid pixels whose removal increases the
       4-connected component count. A nonzero count means the structure is held
       together by at least one single-pixel hinge (a de-facto pin joint).
    2. **Erosion survival** — after eroding the solid by ``min_neck_px - 1`` pixels
       (4-connectivity), a physically sound member of width ``>= min_neck_px``
       remains connected; a sub-min neck pinches off and the component count jumps
       / a large fraction of material is lost.

    Args:
        density: (nely, nelx) element densities in [0, 1].
        threshold: Binarization threshold.
        min_neck_px: Minimum acceptable neck/member width in pixels. A member must
            survive ``min_neck_px - 1`` erosions to be considered manufacturable.

    Returns:
        Dict with:
            n_components: 4-connected components of the solid.
            n_bridge_pixels: count of single-pixel articulation points.
            bridge_mask: (nely, nelx) bool mask of those pixels.
            n_components_eroded: components after ``min_neck_px - 1`` erosions.
            frac_lost_eroded: fraction of solid removed by that erosion.
            has_point_hinge: True if any bridge pixel exists.
            survives_erosion: True if eroded solid is one component covering the body.
            passed: physical-validity verdict (no point hinge AND survives erosion).
    """
    binary = (density > threshold)
    n_solid = int(binary.sum())

    result: Dict[str, Any] = {
        "n_components": 0,
        "n_bridge_pixels": 0,
        "bridge_mask": np.zeros_like(binary, dtype=bool),
        "n_components_eroded": 0,
        "frac_lost_eroded": 1.0,
        "has_point_hinge": False,
        "survives_erosion": False,
        "passed": False,
    }
    if n_solid == 0:
        return result

    _, n0 = label(binary, structure=_CROSS4)
    result["n_components"] = int(n0)

    # --- Articulation / bridge pixels ---------------------------------------
    # Only pixels that are NOT fully surrounded can possibly be articulation
    # points; skip interior pixels (all 8 neighbours solid) to avoid an O(N)
    # relabel for every solid pixel in chunky regions.
    fully_surrounded = (minimum_filter(binary.astype(np.uint8), size=3,
                                       mode="constant", cval=0) == 1)
    candidates = binary & ~fully_surrounded

    bridge_mask = np.zeros_like(binary, dtype=bool)
    work = binary.copy()
    ys, xs = np.where(candidates)
    for y, x in zip(ys, xs):
        work[y, x] = False
        _, nc = label(work, structure=_CROSS4)
        work[y, x] = True
        if nc > n0:  # removing this pixel splits a component -> point hinge
            bridge_mask[y, x] = True
    n_bridge = int(bridge_mask.sum())
    result["n_bridge_pixels"] = n_bridge
    result["bridge_mask"] = bridge_mask
    result["has_point_hinge"] = n_bridge > 0

    # --- Erosion survival ----------------------------------------------------
    eroded = binary
    for _ in range(max(min_neck_px - 1, 1)):
        eroded = binary_erosion(eroded, structure=_CROSS4)
    n_eroded_solid = int(eroded.sum())
    _, n_eroded = label(eroded, structure=_CROSS4)
    frac_lost = 1.0 - n_eroded_solid / n_solid
    result["n_components_eroded"] = int(n_eroded)
    result["frac_lost_eroded"] = float(frac_lost)
    # Survives if a single dominant member remains after thinning.
    result["survives_erosion"] = bool(n_eroded_solid > 0 and n_eroded <= n0)

    result["passed"] = bool((not result["has_point_hinge"])
                            and result["survives_erosion"])
    return result


def repair_point_hinges(
    density: np.ndarray,
    threshold: float = 0.5,
    max_passes: int = 4,
) -> Tuple[np.ndarray, int]:
    """
    Widen single-pixel hinges to a finite-width neck by filling void neighbours
    of articulation pixels.

    Post-processing fallback for designs that are genuine distributed mechanisms
    but retain a few single-pixel pinch points the optimizer could not avoid at a
    given resolution. Each pass dilates the current bridge pixels into adjacent
    void elements (8-connectivity), turning a 1px neck into ~3px. Iterates until
    no point hinge remains or ``max_passes`` is reached.

    NOTE: this stiffens the repaired flexures slightly, so re-solve the FEA after
    repair and re-validate ``u_out``; reject if the mechanism no longer transmits.

    Args:
        density: (nely, nelx) element densities.
        threshold: Binarization threshold.
        max_passes: Maximum widening iterations.

    Returns:
        repaired: (nely, nelx) density with hinges widened (added elements = 1.0).
        n_added: number of elements turned solid.
    """
    out = density.astype(float).copy()
    n_added = 0
    full8 = np.ones((3, 3), dtype=int)
    for _ in range(max_passes):
        h = detect_hinges(out, threshold)
        if not h["has_point_hinge"]:
            break
        binary = out > threshold
        grow = binary_dilation(h["bridge_mask"], structure=full8) & ~binary
        if not grow.any():
            break
        out[grow] = 1.0
        n_added += int(grow.sum())
    return out, n_added


def check_no_nan_inf(density: np.ndarray) -> Tuple[bool, int]:
    """
    Check for NaN or Inf values.
    
    Args:
        density: Array to check
    
    Returns:
        is_valid: True if no NaN/Inf
        n_invalid: Number of invalid values
    """
    invalid_mask = ~np.isfinite(density)
    n_invalid = int(np.sum(invalid_mask))
    
    return n_invalid == 0, n_invalid


def check_boundary_conditions(
    density: np.ndarray,
    fixed_mask: np.ndarray,
    load_mask: np.ndarray,
    threshold: float = 0.5
) -> Tuple[bool, Dict[str, bool]]:
    """
    Check that solid material connects BCs to loads.
    
    Args:
        density: (nely, nelx) element densities
        fixed_mask: (nely, nelx) elements near fixed BCs
        load_mask: (nely, nelx) elements near loads
        threshold: Binarization threshold
    
    Returns:
        is_valid: True if BCs and loads are connected
        details: Dict with specific checks
    """
    binary = (density > threshold).astype(int)
    
    # Check if solid touches fixed BCs
    touches_fixed = np.any(binary & fixed_mask)
    
    # Check if solid touches loads
    touches_load = np.any(binary & load_mask)
    
    is_valid = touches_fixed and touches_load
    
    return is_valid, {
        "touches_fixed": bool(touches_fixed),
        "touches_load": bool(touches_load)
    }


def validate_sample(
    density: np.ndarray,
    target_vf: float = 0.4,
    domain_mask: np.ndarray = None,
    min_feature_size: int = 2,
    max_gray_fraction: float = 0.15,
    vf_tolerance: float = 0.10
) -> Dict[str, Any]:
    """
    Run all validation checks on a sample.
    
    Args:
        density: (nely, nelx) element densities
        target_vf: Target volume fraction
        domain_mask: Optional domain mask
        min_feature_size: Minimum feature size
        max_gray_fraction: Maximum allowed gray fraction
        vf_tolerance: Volume fraction tolerance
    
    Returns:
        Dict with validation results
    """
    results = {}
    
    # NaN/Inf check (must pass for other checks to work)
    is_finite, n_invalid = check_no_nan_inf(density)
    results["finite"] = {"passed": is_finite, "n_invalid": n_invalid}
    
    if not is_finite:
        # Can't proceed with other checks
        results["overall_passed"] = False
        return results
    
    # Connectivity
    is_connected, n_components = check_connectivity(density)
    results["connectivity"] = {
        "passed": is_connected,
        "n_components": n_components
    }
    
    # Volume fraction
    vf_valid, actual_vf = check_volume_fraction(
        density, target_vf, vf_tolerance, domain_mask
    )
    results["volume_fraction"] = {
        "passed": vf_valid,
        "target": target_vf,
        "actual": actual_vf,
        "tolerance": vf_tolerance
    }
    
    # Gray fraction
    gray_valid, gray_frac = check_gray_fraction(density, max_gray_fraction)
    results["gray_fraction"] = {
        "passed": gray_valid,
        "fraction": gray_frac,
        "max_allowed": max_gray_fraction
    }
    
    # Minimum feature size
    feature_valid, smallest = check_minimum_feature_size(density, min_feature_size)
    results["min_feature"] = {
        "passed": feature_valid,
        "min_required": min_feature_size,
        "estimated_smallest": smallest
    }
    
    # Overall verdict
    results["overall_passed"] = (
        is_finite and is_connected and vf_valid and gray_valid and feature_valid
    )
    
    return results


def validate_mechanism(
    density: np.ndarray,
    u_out: float,
    strain_energy: np.ndarray = None,
    min_transmission: float = 1.0,
    min_energy_localization: float = 0.5,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Mechanism-specific validation (beyond standard topology checks).
    
    Validates that the design actually behaves like a mechanism:
    - Meaningful motion transmission (not just soft everywhere)
    - Strain energy concentrated in flexures (not smeared)
    
    Args:
        density: (nely, nelx) element densities
        u_out: Output displacement magnitude
        strain_energy: (nely, nelx) element strain energies (optional)
        min_transmission: Minimum |u_out| to be considered valid
        min_energy_localization: Minimum localization ratio (0-1)
        threshold: Binarization threshold
    
    Returns:
        Dict with mechanism-specific validation results
    """
    results = {}
    
    # 1. Transmission metric: u_out must be meaningful
    # For linear FEA with unit force, expect |u_out| > 1.0 for good mechanisms
    results["transmission"] = {
        "passed": abs(u_out) >= min_transmission,
        "u_out": float(u_out),
        "min_required": min_transmission
    }
    
    # 2. Energy localization (if strain energy provided)
    if strain_energy is not None:
        binary = (density > threshold).astype(bool)
        solid_energy = strain_energy[binary]
        
        if len(solid_energy) > 0 and np.sum(solid_energy) > 1e-12:
            # Compute Gini coefficient as localization measure
            # High Gini = energy concentrated in few elements (flexures)
            # Low Gini = energy smeared everywhere (blob)
            sorted_energy = np.sort(solid_energy)
            n = len(sorted_energy)
            if n > 1:
                cumsum = np.cumsum(sorted_energy)
                gini = (2 * np.sum((np.arange(1, n+1)) * sorted_energy) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
                localization = float(gini)
            else:
                localization = 0.0
            
            results["energy_localization"] = {
                "passed": localization >= min_energy_localization,
                "gini_coefficient": localization,
                "min_required": min_energy_localization
            }
        else:
            results["energy_localization"] = {
                "passed": False,
                "gini_coefficient": 0.0,
                "min_required": min_energy_localization,
                "error": "No strain energy in solid regions"
            }
    
    # Overall mechanism validity
    transmission_ok = results["transmission"]["passed"]
    localization_ok = results.get("energy_localization", {}).get("passed", True)
    results["mechanism_valid"] = transmission_ok and localization_ok
    
    return results


def validate_batch(
    densities: np.ndarray,
    target_vf: float = 0.4,
    **kwargs
) -> Dict[str, Any]:
    """
    Validate a batch of samples.
    
    Args:
        densities: (batch, nely, nelx) density arrays
        target_vf: Target volume fraction
        **kwargs: Additional args for validate_sample
    
    Returns:
        Batch validation summary
    """
    n_samples = len(densities)
    n_passed = 0
    failures = {
        "connectivity": 0,
        "volume_fraction": 0,
        "gray_fraction": 0,
        "min_feature": 0,
        "finite": 0
    }
    
    for i, density in enumerate(densities):
        result = validate_sample(density, target_vf, **kwargs)
        
        if result["overall_passed"]:
            n_passed += 1
        else:
            for key in failures:
                if key in result and not result[key]["passed"]:
                    failures[key] += 1
    
    return {
        "n_samples": n_samples,
        "n_passed": n_passed,
        "pass_rate": n_passed / n_samples if n_samples > 0 else 0,
        "failures": failures
    }
