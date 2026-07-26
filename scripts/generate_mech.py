#!/usr/bin/env python3
"""
Generate mechanism samples for OpenCompMech.

Usage:
    # Quick test (10 samples, no physics)
    python scripts/generate_mech.py --n-samples 10 --output-dir data/test_mech --no-physics
    
    # With physics fields
    python scripts/generate_mech.py --n-samples 100 --output-dir data/shards/mech
    
    # Parallel generation
    python scripts/generate_mech.py --n-samples 1000 --workers 8 --output-dir data/shards/mech
"""

import argparse
import json
import os
import sys
import time
import tarfile
import io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generators.mech import (
    MechConfig, MechProblem,
    generate_inverter_problem, generate_gripper_problem,
    generate_random_mechanism, generate_crusher_problem,
    generate_amplifier_problem, generate_crank_slider_problem,
    optimize_mechanism, solve_mechanism_fea,
    compute_mech_physics_fields,  # Use mechanism-specific physics!
    MECH_GENERATORS
)
from src.validation.connectivity import (
    validate_sample, check_bc_connectivity, check_mechanism_path_connectivity,
    detect_hinges,
)
from src.io.shards import TarShardWriter


# --- Per-type mechanical-quality floor (beyond mere validity) ---------------
# The hinge/connectivity gate certifies a sample is VALID; it does not certify
# it is a GOOD mechanism. These per-type minimum geometric-advantage (GA =
# u_out/u_in) floors reject the weak tail. Levels are calibrated from the
# n=16/type distribution (data/audit_qual, 2026-06-17): each floor sits below
# that type's p10 so only clearly-degenerate samples are dropped. GA ceilings
# differ by mechanism physics (a symmetric inverter tops out ~0.5; an amplifier
# ~1.1), so the floor MUST be per-type — a single cross-type cut would wrongly
# reject good inverters or pass bad amplifiers.
GA_FLOOR = {
    'amplifier': 0.70,     # measured median 1.11, p10 0.95
    'crank_slider': 0.60,  # median 0.97, p10 0.90
    'crusher': 0.45,       # median 0.64, p10 0.61
    'gripper': 0.45,       # median 0.64, p10 0.55
    'inverter': 0.35,      # median 0.46 (reworked), p10 0.41 — low by physics
    'random': 0.45,        # median 0.82, wide spread (p10 0.39)
}
# Universal sanity cap on off-axis input (fraction of input motion perpendicular
# to the applied force). Lenient — lever mechanisms legitimately reach ~0.6 —
# so this only catches degenerate cases where the input barely moves on-axis.
OFF_AXIS_CAP = 0.70
# Strain-energy Gini cap: manual review of a validation batch (n=100-200/type)
# found that gate-passing samples with gini > ~0.75 are
# "blob with a floppy tail" pseudo-mechanisms (lumped compliance) — every
# flagged sample had gini 0.77-0.95 while genuinely articulated passers sit at
# ~0.4-0.7. Costs <=3.5 yield pts on most types (slider -10.5, honestly so).
GINI_CAP = 0.75
# A port which cannot be approached by an actuator/workpiece is not a usable
# mechanism interface even when the linear solve is well-conditioned.  Keep
# the gate aligned with eval_mechanism_gate.py.
MIN_PORT_CLEARANCE = 0.50
MIN_PORT_SELECTIVITY = 1.0

# --- Per-type density-filter scale (filter_radius = resolution / divisor) -----
# The filter sets the minimum length scale: a larger filter widens features and
# suppresses single-pixel hinges, but too large collapses the mechanism into a
# rigid blob. The default res/25.6 (= 5.0 @128) is validated across all types
# (2026-06-15). The rebuilt distributed two-support GRIPPER pinches at edge/port
# flexures that the filter under-smooths at the domain boundary, so it needs a
# wider filter: the root lever against point hinges (which thrive below ~res/25).
# Per-type because a global bump risks collapsing the other types — they stay at
# the validated default until each is re-tuned during rollout.
DEFAULT_FILTER_DIVISOR = 25.6
FILTER_DIVISOR = {
    # 2026-07-10 head-to-head @128 with the v2 diversity generator (n=16 each,
    # data/audit_gripper_f60_confirm vs _f725_div): 6.0 and 7.25 tie on yield
    # (7/16 new-gate; losses are non-square-domain collapse, not filter), but 6.0
    # gives slender members and a uniformly clean passing cohort (GA p10 0.64 vs
    # 0.45; 7.25 admits near-blob passers). Hinge flags land mostly on already-
    # dead samples and the gate rejects them either way.
    'gripper': 21.3,  # = filter 6.0 @128
}


def filter_radius_for(problem_name: str, resolution: int) -> float:
    """Per-type density-filter radius (see FILTER_DIVISOR)."""
    divisor = FILTER_DIVISOR.get(problem_name, DEFAULT_FILTER_DIVISOR)
    return max(2.5, resolution / divisor)


def encode_mask_rle(mask: np.ndarray) -> dict:
    """Compact, lossless domain-mask metadata for problem reconstruction."""
    flat = np.asarray(mask, dtype=np.uint8).ravel()
    if flat.size == 0:
        return {"shape": list(mask.shape), "starts_with": 0, "runs": []}
    changes = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.r_[0, changes, flat.size]
    return {
        "shape": list(mask.shape),
        "starts_with": int(flat[0]),
        "runs": [int(b - a) for a, b in zip(bounds[:-1], bounds[1:])],
    }


def compute_quality_metrics(problem, density):
    """Solve the mechanism FEA and return (ga, off_axis, u_in, u_out, gini).

    ga = u_out/u_in (geometric advantage, signed; <0 => wrong direction).
    off_axis = |input motion perpendicular to applied force| / |input motion|.
    gini = strain-energy localization over solid elements (healthy flexure
    band ~0.4-0.7; > GINI_CAP => lumped blob-with-tail, see GINI_CAP).
    One extra solve (~0.3% over the 300-iter optimization)."""
    u, _K, u_out = solve_mechanism_fea(problem, density)
    ix, iy = problem.get_input_dofs()
    dxi, dyi = problem.input_direction
    u_in = float(u[ix] * dxi + u[iy] * dyi)
    ui = np.array([u[ix], u[iy]])
    perp = np.array([-dyi, dxi])
    off_axis = float(abs(ui @ perp) / (np.linalg.norm(ui) + 1e-12))
    ga = u_out / u_in if abs(u_in) > 1e-12 else 0.0
    # Per-element SIMP strain energy -> Gini over solid elements (same
    # computation as scripts/audit_mech.py so audit and production agree).
    from src.solvers.linear import get_cached_edof, get_element_stiffness_cached
    nely, nelx = density.shape
    mat = problem.material
    edof = get_cached_edof(nelx, nely)
    ke = get_element_stiffness_cached(mat.E, mat.nu)
    u_e = u[edof]
    uku = np.einsum("ij,jk,ik->i", u_e, ke, u_e)
    rho = np.asarray(density, dtype=np.float64).flatten()
    E_interp = mat.E_min + np.power(np.maximum(rho, 0.0), 3.0) * (mat.E - mat.E_min)
    se = 0.5 * (E_interp / mat.E) * uku
    solid = rho > 0.5
    gini = 0.0
    if solid.sum() >= 3:
        e = np.sort(se[solid])
        n = len(e)
        tot = float(e.sum())
        if tot > 0:
            gini = float((2 * np.sum(np.arange(1, n + 1) * e) - (n + 1) * tot) / (n * tot))
    return float(ga), off_axis, u_in, float(u_out), gini


def compute_conditioning_field(problem, uniform_vf, resolution):
    """Per-element strain-energy raster of the uniform-density domain under
    the sample's exact boundary conditions, loads and springs.

    This is the "where do load paths want to run BEFORE any material layout
    exists" channel for the conditional generative model: it is computable
    from the problem spec alone (no design), so it is available at inference
    time. Uniform density = target VF inside the domain mask, 0 outside.
    One extra FEA solve (~0.3 s at 128²).
    """
    nely = nelx = resolution
    if problem.domain_mask is not None:
        uniform = np.where(problem.domain_mask, uniform_vf, 0.0).astype(np.float64)
    else:
        uniform = np.full((nely, nelx), uniform_vf, dtype=np.float64)
    u, _K, _u_out = solve_mechanism_fea(problem, uniform)
    from src.solvers.linear import get_cached_edof, get_element_stiffness_cached
    mat = problem.material
    edof = get_cached_edof(nelx, nely)
    ke = get_element_stiffness_cached(mat.E, mat.nu)
    u_e = u[edof]
    uku = np.einsum("ij,jk,ik->i", u_e, ke, u_e)
    rho = uniform.flatten()
    E_interp = mat.E_min + np.power(np.maximum(rho, 0.0), 3.0) * (mat.E - mat.E_min)
    se = 0.5 * (E_interp / mat.E) * uku
    return se.reshape(nely, nelx)


def generate_single_sample(
    sample_id: int,
    resolution: int,
    config: MechConfig,
    compute_physics: bool,
    refinement_factor: int = 2,
    problem_types: list = None
) -> dict:
    """Generate a single mechanism sample."""

    # Cycle through the requested problem types for diversity. Default excludes
    # 'amplifier' (floating-translator degeneracy, 0% yield — docs/DATASET.md
    # §5.9); pass problem_types explicitly to override.
    if problem_types is None:
        problem_types = ['inverter', 'gripper', 'random', 'crusher', 'crank_slider']
    problem_idx = sample_id % len(problem_types)
    problem_name = problem_types[problem_idx]

    # Per-type density-filter radius (the gripper needs a wider filter than the
    # default to suppress edge/port hinges — see FILTER_DIVISOR). Override the
    # shared config's filter_radius for this sample's type.
    type_filter = filter_radius_for(problem_name, resolution)
    if type_filter != config.filter_radius:
        config = replace(config, filter_radius=type_filter)

    # Seed based on sample_id
    # NumPy RandomState accepts only uint32 seeds.  Keep the historical mapping
    # for all existing IDs, but make sharded/resumed production safe beyond
    # ~348k samples instead of silently failing every worker.
    seed = (sample_id * 12345 + 67890) % (2 ** 32)

    # Family E (constructed, no optimizer) vs Family A (optimized) dispatch.
    from src.generators.rigid_replace import RR_CONSTRUCTORS
    is_constructed = problem_name in RR_CONSTRUCTORS
    rr_meta = None
    start_time = time.time()
    if is_constructed:
        rng = np.random.RandomState(seed)
        built = RR_CONSTRUCTORS[problem_name](
            nelx=resolution, nely=resolution, rng=rng)
        if built is None:
            return {'sample_id': sample_id, 'is_valid': False,
                    'error': 'construction_failed'}
        density_c, problem, rr_meta = built
        _, _, u_out_c = solve_mechanism_fea(problem, density_c)

        class _ConstructedResult:
            density = density_c
            compliance = -float(u_out_c)
            volume_fraction = float(rr_meta['constructed_vf'])
            n_iterations = 0
            converged = True
        result = _ConstructedResult()
        target_vf = float(rr_meta['constructed_vf'])   # its own VF is the target
    else:
        # Generate problem
        generator = MECH_GENERATORS.get(problem_name, generate_random_mechanism)
        # 'random' and 'amplifier' own their springs (k=None): the amplifier
        # NEEDS a stiff randomized k_in >> k_out, else the optimum is a
        # free-floating translator that skips ground and fails the
        # connectivity gate (2026-07-10).
        _gen_owns_springs = problem_name in ('random', 'amplifier')
        problem = generator(
            nelx=resolution,
            nely=resolution,
            volume_fraction=config.volume_fraction,
            k_in=None if _gen_owns_springs else config.k_in,
            k_out=None if _gen_owns_springs else config.k_out,
            seed=seed
        )
        result = optimize_mechanism(problem, config)
        target_vf = config.volume_fraction
    opt_time = time.time() - start_time

    # Validate with STRICT thresholds - don't hide optimizer failures
    # Gray >15% means Heaviside beta isn't high enough - fix optimizer, not threshold
    # Volume fraction check — 5% tolerance accounts for fragment removal post-processing
    validation_info = validate_sample(
        result.density,
        target_vf=target_vf,
        # measure VF over the ACTIVE region — generators may carve a non-square
        # domain (e.g. the diversified gripper's wide/tall shapes); a None mask
        # would dilute VF by the forced-void area and wrongly fail the gate.
        domain_mask=problem.domain_mask,
        vf_tolerance=0.05,  # 5% tolerance (fragment removal lowers VF slightly)
        max_gray_fraction=0.20  # 20% max - some gray is OK for mechanisms
    )
    
    # ALL checks must pass - mechanisms need continuous load path
    finite_ok = validation_info.get('finite', {}).get('passed', False)
    connected_ok = validation_info.get('connectivity', {}).get('passed', False)
    volume_ok = validation_info.get('volume_fraction', {}).get('passed', False)
    gray_ok = validation_info.get('gray_fraction', {}).get('passed', False)
    feature_ok = validation_info.get('min_feature', {}).get('passed', False)
    
    # CRITICAL: Check if solid structure connects to fixed boundary conditions
    # A mechanism floating in space is useless - needs load path to supports!
    fixed_nodes = (np.concatenate([bc.node_indices for bc in problem.base_problem.bcs])
                   if problem.base_problem.bcs else np.array([], dtype=int))
    bc_connected, n_bc_elements = check_bc_connectivity(
        result.density, 
        fixed_nodes,
        nelx=resolution,
        nely=resolution
    )
    validation_info['bc_connectivity'] = {
        'passed': bc_connected,
        'n_connected_elements': n_bc_elements
    }
    path_info = check_mechanism_path_connectivity(
        result.density, fixed_nodes, problem.input_node, problem.output_node,
        nelx=resolution, nely=resolution)
    validation_info['mechanism_path'] = path_info

    is_valid = (finite_ok and connected_ok and bc_connected and path_info['passed']
                and volume_ok and gray_ok and feature_ok)
    
    # Track which validation failed
    if not is_valid:
        if not finite_ok:
            validation_info['failure_reason'] = 'nan_or_inf'
        elif not connected_ok:
            validation_info['failure_reason'] = 'disconnected'
        elif not bc_connected:
            validation_info['failure_reason'] = 'not_connected_to_bc'
        elif not path_info['passed']:
            validation_info['failure_reason'] = 'ports_not_grounded'
        elif not volume_ok:
            validation_info['failure_reason'] = 'volume_constraint'
        elif not gray_ok:
            validation_info['failure_reason'] = 'gray_fraction'
        elif not feature_ok:
            validation_info['failure_reason'] = 'minimum_feature'
    
    # Also check if objective is reasonable (absolute value - direction doesn't matter).
    # 1.0 matches the audit's TRANSMISSION_MIN — production previously used 0.1,
    # which let sub-stroke constructed samples through that the audit rejects.
    u_out = -result.compliance
    if abs(u_out) < 1.0:
        is_valid = False
        validation_info['failure_reason'] = 'low_displacement'

    # Reject single-pixel point hinges: FEA-invalidating, unmanufacturable
    # point flexures that linear FEA over-rewards. Robust mode makes these rare;
    # this gate guarantees a hinge-free dataset (audit 2026-06-15).
    hinge_info = detect_hinges(result.density)
    validation_info['point_hinge'] = {
        'passed': hinge_info['passed'],
        'n_bridge_pixels': hinge_info['n_bridge_pixels'],
        'survives_erosion': hinge_info['survives_erosion'],
        'frac_lost_eroded': hinge_info['frac_lost_eroded'],
    }
    if is_valid and not hinge_info['passed']:
        is_valid = False
        validation_info['failure_reason'] = (
            'point_hinge' if hinge_info['has_point_hinge'] else 'erosion_failure')

    # Mechanical-quality floor: reject valid-but-weak mechanisms (low geometric
    # advantage, or input motion mostly off-axis). One extra FEA solve, only for
    # samples that are otherwise valid. Per-type GA floor (see GA_FLOOR).
    ga = off_axis = q_uin = q_uout = q_gini = None
    # Family E: flexure linkages localize strain energy in their DESIGNED necks
    # (high gini + high rigid-resid = correct piecewise-rigid behavior), so the
    # gini cap does not apply; floppy chains die on a GA floor instead. Same
    # per-family rule as the audit (audit_mech.py, kind='construct').
    ga_floor = 0.25 if is_constructed else GA_FLOOR.get(problem_name, 0.40)
    if is_valid:
        ga, off_axis, q_uin, q_uout, q_gini = compute_quality_metrics(problem, result.density)
        if ga < ga_floor:
            is_valid = False
            validation_info['failure_reason'] = 'low_ga'
        elif off_axis > OFF_AXIS_CAP:
            is_valid = False
            validation_info['failure_reason'] = 'off_axis_input'
        elif (not is_constructed) and q_gini > GINI_CAP:
            # Lumped blob-with-tail pseudo-mechanism.
            is_valid = False
            validation_info['failure_reason'] = 'lumped_compliance'
    validation_info['quality'] = {
        'passed': is_valid,
        'ga': ga, 'off_axis': off_axis, 'u_in': q_uin, 'u_out': q_uout,
        'gini': q_gini,
        'ga_floor': ga_floor,
        'gini_cap': None if is_constructed else GINI_CAP,
    }
    # Port exposure: clear external interface vs embedded in the structure.
    # Always RECORDED (clearance + approach_clear per port) so it can be a
    # conditioning axis / release-subset filter. It is NOT a hard reject by
    # default: as a hard gate it rejected 60-99% of whole archetypes (inverter,
    # rr_lever, rr_compound_lever, crusher) whose ports are interior BY DESIGN,
    # which contradicts ports.py's own "deliberately NOT a validity gate" intent
    # and destroys motion-class coverage. Opt in with MECH_STRICT_PORT_ACCESS=1
    # for a manufacturable-interface release subset.
    from src.validation.ports import problem_port_exposure
    validation_info['port_exposure'] = problem_port_exposure(
        result.density, problem)
    exposure = validation_info['port_exposure']
    interface_ok = bool(
        exposure.get('input', {}).get('approach_clear', False)
        and exposure.get('output', {}).get('approach_clear', False)
        and exposure.get('input', {}).get('clearance', 0.0) >= MIN_PORT_CLEARANCE
        and exposure.get('output', {}).get('clearance', 0.0) >= MIN_PORT_CLEARANCE)
    validation_info['port_interface'] = {
        'passed': interface_ok, 'min_clearance': MIN_PORT_CLEARANCE}
    if is_valid and not interface_ok and os.environ.get("MECH_STRICT_PORT_ACCESS", "") == "1":
        is_valid = False
        validation_info['failure_reason'] = 'port_not_accessible'
    # Footprint: solid bbox extent / domain side (tiny-mechanism indicator).
    # Reported, not gated.
    _fy, _fx = np.where(result.density > 0.5)
    validation_info['footprint'] = (
        float(max((_fx.max() - _fx.min()) / result.density.shape[1],
                  (_fy.max() - _fy.min()) / result.density.shape[0]))
        if _fx.size else 0.0)
    # Port compliance selectivity: the parasitic-compliance signal the single
    # working-load solve cannot see. One reduced-K
    # factorization reused for unit-load probes at both ports. Reported, NOT
    # gated (the dataset wants a RANGE of selectivity as a conditioning axis).
    if is_valid:
        try:
            from src.validation.compliance import port_selectivity
            validation_info['port_selectivity'] = port_selectivity(
                problem, result.density, penal=config.penal_max)
            selectivity = float(validation_info['port_selectivity'].get(
                'output', {}).get('selectivity', 0.0))
            validation_info['port_selectivity']['gate'] = {
                'passed': selectivity >= MIN_PORT_SELECTIVITY,
                'min_selectivity': MIN_PORT_SELECTIVITY,
            }
            if selectivity < MIN_PORT_SELECTIVITY:
                is_valid = False
                validation_info['failure_reason'] = 'low_selectivity'
        except Exception as e:
            validation_info['selectivity_error'] = str(e)
            is_valid = False
            validation_info['failure_reason'] = 'selectivity_error'
        # Motion-class: WHAT the mechanism measurably does (function axis —
        # the dataset's real diversity axis, user insight 2026-07-17).
        # Includes the blocked-force mechanical advantage (force-amp number).
        try:
            from src.validation.motion_class import motion_class
            validation_info['motion'] = motion_class(problem, result.density)
        except Exception as e:
            validation_info['motion_error'] = str(e)

    # Physics fields
    physics_data = None
    if compute_physics and is_valid:
        try:
            # CRITICAL: Use mechanism-specific physics that applies input force!
            physics, density_fine = compute_mech_physics_fields(
                problem,  # Use MechProblem, not base_problem!
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
            validation_info['physics_error'] = str(e)
            is_valid = False
            validation_info['failure_reason'] = 'physics_error'

    # Conditioning field: uniform-domain strain-energy raster.
    # Derivable from the problem spec alone => valid conditioning input at
    # inference time. Saved as <prefix>.cond_energy.npy for valid samples.
    conditioning_field = None
    if is_valid:
        try:
            conditioning_field = compute_conditioning_field(
                problem, target_vf, resolution).astype(np.float32)
        except Exception as e:
            validation_info['conditioning_error'] = str(e)
            is_valid = False
            validation_info['failure_reason'] = 'conditioning_error'
    
    # Build metadata
    metadata = {
        'sample_id': sample_id,
        'tier': 2,
        'tier_name': 'mech',
        'problem_type': problem_name,
        'family': (rr_meta.get('family', 'E') if is_constructed else 'A'),
        'rr_construction': rr_meta,   # construction provenance (None for Family A)
        'resolution': resolution,
        'volume_fraction_target': target_vf,
        'volume_fraction_actual': float(result.volume_fraction),
        'optimization': {
            'n_iterations': result.n_iterations,
            'converged': result.converged,
            'final_objective': float(u_out),
            'time_seconds': round(opt_time, 2)
        },
        'mechanism': {
            'input_node': int(problem.input_node),
            'input_direction': list(problem.input_direction),  # (dx, dy) unit vector
            'output_node': int(problem.output_node),
            'output_direction': list(problem.output_direction),  # (dx, dy) unit vector
            'k_in': float(problem.k_in),
            'k_out': float(problem.k_out),
            'k_perp': float(problem.k_perp or 0.0),
        },
        'conditioning': ({'cond_energy': True, 'uniform_vf': target_vf}
                         if conditioning_field is not None else None),
        # Re-derivability: BC node patches were the one part of the problem
        # spec not stored (ports/springs are under 'mechanism'). With these,
        # the FEA problem is fully reconstructible from the JSON alone.
        'boundary_conditions': [
            {'nodes': [int(n) for n in bc.node_indices],
             'directions': [int(d) for d in bc.directions]}
            for bc in problem.base_problem.bcs
        ],
        'domain_mask_rle': encode_mask_rle(problem.domain_mask),
        'provenance': {
            'generator': 'generate_mech.py',
            'seed': int(seed),
            'lineage_id': f"{problem_name}:{seed}",
            'spring_model': 'directional_outer_product_v1',
        },
        'validation': validation_info
    }
    validation_info['overall_passed'] = bool(is_valid)
    validation_info['quality']['passed'] = bool(is_valid)

    return {
        'sample_id': sample_id,
        'density': result.density,
        'physics': physics_data,
        'conditioning_field': conditioning_field,
        'metadata': metadata,
        'is_valid': is_valid
    }


def worker_generate(args):
    """Worker function for parallel generation."""
    sample_id, resolution, config_dict, compute_physics, problem_types = args

    # Reconstruct config from dict
    config = MechConfig(**config_dict)

    try:
        result = generate_single_sample(
            sample_id, resolution, config, compute_physics,
            problem_types=problem_types
        )
        return result
    except Exception as e:
        return {
            'sample_id': sample_id,
            'is_valid': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Generate mechanism samples for OpenCompMech'
    )
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Mesh resolution (default: 64)')
    parser.add_argument('--output-dir', type=str, default='data/test_mech',
                        help='Output directory for samples')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--no-physics', action='store_true',
                        help='Skip physics field computation')
    parser.add_argument('--volume-fraction', type=float, default=0.20,
                        help='Target volume fraction (default: 0.20)')
    parser.add_argument('--max-iterations', type=int, default=300,
                        help='Maximum optimization iterations (default: 300). '
                             '300 validated best; 400 over-sharpens single-pixel '
                             'hinges and halves yield on inverter/gripper (2026-06-16).')
    parser.add_argument('--no-multires', action='store_true',
                        help='Disable multi-resolution warm-start. Multires is ON by '
                             'default: ~2x faster with equal-or-better yield/GA '
                             '(validated 2026-06-16). Use this only for debugging.')
    parser.add_argument('--shard-size', type=int, default=1000,
                        help='Samples per tar shard')
    parser.add_argument('--start-id', type=int, default=0,
                        help='Starting sample ID (for resuming)')
    parser.add_argument('--types', nargs='+', default=None,
                        help='Problem types to cycle through (default: all '
                             'working types; amplifier excluded until its '
                             'floating-translator defect is fixed)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter radius sets the minimum length scale and must scale with resolution
    # (~res/25.6 => 2.5 @64, 5.0 @128). With robust mode (default on) this is what
    # eliminates single-pixel hinges. Validated 2026-06-15. This config value is the
    # DEFAULT; per_sample_generate overrides it per type via filter_radius_for()
    # (e.g. the gripper uses a wider filter — see FILTER_DIVISOR).
    filter_radius = max(2.5, args.resolution / DEFAULT_FILTER_DIVISOR)
    config = MechConfig(
        max_iterations=args.max_iterations,
        volume_fraction=args.volume_fraction,
        filter_radius=filter_radius,
        # robust=True by default (adds 2nd FEA solve/iter; kills point hinges)
        # multires: coarse establish -> fine robust refine. ~2x faster, equal-or-
        # better yield/GA (validated 2026-06-16). Coarse res 64 must divide target.
        multires=(not args.no_multires) and (args.resolution % 64 == 0) and (args.resolution > 64),
    )
    config_dict = asdict(config)
    _per_type_filters = ", ".join(
        f"{t}={filter_radius_for(t, args.resolution):.2f}" for t in sorted(FILTER_DIVISOR)
    )
    print(f"  Filter radius: {filter_radius:.1f} default"
          + (f" (per-type: {_per_type_filters})" if _per_type_filters else "")
          + f" | robust: {config.robust} "
          f"(mode={config.robust_mode}, eta_offset={config.robust_eta_offset})")
    print(f"  Multires: {config.multires}"
          + (f" (coarse {config.coarse_resolution} -> {args.resolution}, "
             f"coarse_frac {config.coarse_fraction})" if config.multires else ""))
    
    print(f"Generating {args.n_samples} mechanism samples")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Volume fraction: {args.volume_fraction}")
    print(f"  Workers: {args.workers}")
    print(f"  Physics: {'No' if args.no_physics else 'Yes'}")
    print(f"  Output: {output_dir}")
    print()
    
    # Generate samples
    compute_physics = not args.no_physics
    start_time = time.time()
    
    valid_count = 0
    total_count = 0
    
    # Prepare work items
    sample_ids = range(args.start_id, args.start_id + args.n_samples)
    work_items = [
        (sid, args.resolution, config_dict, compute_physics, args.types)
        for sid in sample_ids
    ]
    
    if args.workers > 1:
        # Parallel generation with NUMA-aware pinning: one worker per physical
        # core, single-threaded BLAS, local first-touch memory (see
        # src/core/cpu_affinity.py). Critical on NPS4 EPYC to avoid SMT
        # oversubscription and cross-node memory-bandwidth contention.
        from multiprocessing import Manager
        from src.core.cpu_affinity import make_affinity_initializer, physical_core_count
        _ncores = physical_core_count()
        _mgr = Manager()
        _pin_init, _pin_args = make_affinity_initializer(_mgr, _ncores)
        print(f"  Pinning {args.workers} workers across {_ncores} physical cores "
              f"(1 thread/core, NUMA-local)")
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_pin_init, initargs=_pin_args) as executor:
            futures = {executor.submit(worker_generate, item): item[0]
                       for item in work_items}
            
            for future in as_completed(futures):
                sample_id = futures[future]
                result = future.result()
                total_count += 1
                
                if result.get('is_valid', False):
                    valid_count += 1
                    # Save sample
                    save_sample(output_dir, result)
                    print(f"[{total_count}/{args.n_samples}] Sample {sample_id}: "
                          f"u_out={result['metadata']['optimization']['final_objective']:.2f} ✓")
                else:
                    reason = result.get('error', result.get('metadata', {}).get('validation', {}).get('failure_reason', 'unknown'))
                    print(f"[{total_count}/{args.n_samples}] Sample {sample_id}: FAILED ({reason})")
    else:
        # Sequential generation
        for item in work_items:
            result = worker_generate(item)
            total_count += 1
            
            if result.get('is_valid', False):
                valid_count += 1
                save_sample(output_dir, result)
                print(f"[{total_count}/{args.n_samples}] Sample {result['sample_id']}: "
                      f"u_out={result['metadata']['optimization']['final_objective']:.2f} ✓")
            else:
                reason = result.get('error', result.get('metadata', {}).get('validation', {}).get('failure_reason', 'unknown'))
                print(f"[{total_count}/{args.n_samples}] Sample {result['sample_id']}: FAILED ({reason})")
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 50)
    print(f"Generation complete!")
    print(f"  Valid samples: {valid_count}/{total_count} ({valid_count/total_count:.1%})")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Time per sample: {elapsed/total_count:.1f}s")
    print(f"  Throughput: {total_count/elapsed*3600:.0f} samples/hour")


def save_sample(output_dir: Path, result: dict):
    """Save a sample to individual files."""
    sample_id = result['sample_id']
    prefix = f"{sample_id:06d}"
    
    # Save density
    np.save(output_dir / f"{prefix}.density.npy", result['density'].astype(np.float32))
    
    # Save physics if available
    if result.get('physics'):
        if result['physics'].get('displacement') is not None:
            np.save(output_dir / f"{prefix}.displacement.npy",
                    result['physics']['displacement'].astype(np.float32))
        if result['physics'].get('stress_vm') is not None:
            np.save(output_dir / f"{prefix}.stress.npy",
                    result['physics']['stress_vm'].astype(np.float32))
    if result.get('conditioning_field') is not None:
        np.save(output_dir / f"{prefix}.cond_energy.npy",
                result['conditioning_field'])
    
    # Save metadata - convert numpy types to Python types
    metadata = json_safe(result['metadata'])
    with open(output_dir / f"{prefix}.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def json_safe(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == '__main__':
    main()
