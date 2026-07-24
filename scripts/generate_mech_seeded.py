#!/usr/bin/env python3
"""
Generate seeded mechanism (Tier 2) samples for COMP2D dataset.

Uses linkage seeds (four-bar, slider-crank) to initialize topology optimization.
This is the Phase B1 replacement for the old uniform-init generate_mech.py.

Usage:
    # Quick test (10 samples, with visualization)
    python scripts/generate_mech_seeded.py --n-samples 10 --output-dir data/test_seeded

    # Test specific linkage type
    python scripts/generate_mech_seeded.py --n-samples 20 --linkage-type four_bar --output-dir data/test_4bar

    # Production run (6 workers overnight on HX99G)  
    python scripts/generate_mech_seeded.py --n-samples 1000 --workers 6 --output-dir data/shards/mech_seeded

    # With physics fields for dataset
    python scripts/generate_mech_seeded.py --n-samples 100 --output-dir data/shards/mech --compute-physics
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generators.mech import (
    MechConfig, MechProblem,
    optimize_mechanism,
    compute_mech_physics_fields,
    check_mechanism_connectivity,
)
from src.generators.seeds import seed_from_linkage, SEED_GENERATORS
from src.validation.connectivity import (
    validate_sample, check_bc_connectivity, detect_hinges,
)


def generate_seeded_sample(
    sample_id: int,
    resolution: int,
    config: MechConfig,
    linkage_type: str = None,
    compute_physics: bool = False,
    refinement_factor: int = 2,
) -> dict:
    """Generate a single seeded mechanism sample.

    Args:
        sample_id: unique sample identifier (used as RNG seed)
        resolution: grid size (e.g. 64 for 64×64)
        config: MechConfig with optimization parameters
        linkage_type: 'four_bar', 'slider_crank', or None (random choice)
        compute_physics: whether to compute displacement/stress fields
        refinement_factor: FEA mesh refinement (2× default)

    Returns:
        dict with 'sample_id', 'density', 'physics', 'metadata', 'is_valid'
    """
    # Deterministic RNG from sample_id
    rng = np.random.RandomState(sample_id * 54321 + 13579)

    # Choose linkage type
    available_types = list(SEED_GENERATORS.keys())
    if linkage_type is None:
        chosen_type = rng.choice(available_types)
    else:
        chosen_type = linkage_type

    # Generate seed
    start_time = time.time()
    seed_result = seed_from_linkage(
        linkage_type=chosen_type,
        nelx=resolution,
        nely=resolution,
        rng=rng,
        volume_fraction=config.volume_fraction,
        k_in=config.k_in if config.k_in else 0.01,
        k_out=config.k_out,
        max_attempts=200,
    )

    if seed_result is None:
        return {
            'sample_id': sample_id,
            'density': None,
            'physics': None,
            'metadata': {
                'sample_id': sample_id,
                'failure_reason': f'seed_generation_failed_{chosen_type}',
            },
            'is_valid': False,
        }

    density_init, mech_problem, seed_info = seed_result

    # Create seed_mask for soft lock (elements that are solid in the seed)
    seed_mask = density_init > 0.5

    # Optimize with seeded density
    opt_start = time.time()
    result = optimize_mechanism(
        mech_problem,
        config,
        initial_density=density_init,
        seed_mask=seed_mask,
    )
    opt_time = time.time() - opt_start
    total_time = time.time() - start_time

    # Validate — use generous VF tolerance because post-processing
    # (keep_connected_to_ground + remove_small_fragments) naturally lowers VF
    validation_info = validate_sample(
        result.density,
        target_vf=config.volume_fraction,
        domain_mask=None,
        vf_tolerance=0.25,     # 25% tolerance (post-processing lowers VF)
        max_gray_fraction=0.20,
    )

    finite_ok = validation_info.get('finite', {}).get('passed', False)
    connected_ok = validation_info.get('connectivity', {}).get('passed', False)
    volume_ok = validation_info.get('volume_fraction', {}).get('passed', False)
    gray_ok = validation_info.get('gray_fraction', {}).get('passed', False)

    # BC connectivity
    fixed_nodes = []
    for bc in mech_problem.base_problem.bcs:
        fixed_nodes.extend(bc.node_indices.tolist())
    fixed_nodes = np.array(fixed_nodes, dtype=int)

    bc_connected, n_bc_elements = check_bc_connectivity(
        result.density, fixed_nodes, nelx=resolution, nely=resolution
    )
    validation_info['bc_connectivity'] = {
        'passed': bc_connected,
        'n_connected_elements': int(n_bc_elements),
    }

    # Mechanism connectivity (input + output in same component)
    mech_conn = check_mechanism_connectivity(result.density, mech_problem)
    validation_info['mechanism_connectivity'] = mech_conn

    is_valid = (
        finite_ok and connected_ok and bc_connected and volume_ok and gray_ok
        and mech_conn.get('connected', False)
    )

    # Displacement check
    u_out = -result.compliance  # compliance stored as -|u_out|
    if abs(u_out) < 0.05:
        is_valid = False
        validation_info['failure_reason'] = 'low_displacement'

    # Reject single-pixel point hinges (see audit 2026-06-15).
    hinge_info = detect_hinges(result.density)
    validation_info['point_hinge'] = {
        'passed': not hinge_info['has_point_hinge'],
        'n_bridge_pixels': hinge_info['n_bridge_pixels'],
    }
    if is_valid and hinge_info['has_point_hinge']:
        is_valid = False
        validation_info['failure_reason'] = 'point_hinge'

    if not is_valid and 'failure_reason' not in validation_info:
        if not finite_ok:
            validation_info['failure_reason'] = 'nan_or_inf'
        elif not connected_ok:
            validation_info['failure_reason'] = 'disconnected'
        elif not bc_connected:
            validation_info['failure_reason'] = 'not_connected_to_bc'
        elif not mech_conn.get('connected', False):
            validation_info['failure_reason'] = 'mechanism_disconnected'
        elif not volume_ok:
            validation_info['failure_reason'] = 'volume_constraint'
        elif not gray_ok:
            validation_info['failure_reason'] = 'gray_fraction'

    # Physics
    physics_data = None
    if compute_physics and is_valid:
        try:
            physics, _ = compute_mech_physics_fields(
                mech_problem, result.density,
                refinement_factor=refinement_factor,
                penal=config.penal_max,
            )
            physics_data = {
                'displacement': physics.displacement,
                'stress_vm': physics.stress_vm,
                'strain_energy': physics.strain_energy,
            }
        except Exception as e:
            validation_info['physics_error'] = str(e)

    # Seed quality metrics
    seed_to_final_hausdorff = _binary_hausdorff(
        density_init > 0.5, result.density > 0.5
    )

    # Metadata
    metadata = {
        'sample_id': sample_id,
        'tier': 2,
        'tier_name': 'mech',
        'problem_type': f'seeded_{chosen_type}',
        'seed_category': 'linkage',
        'resolution': resolution,
        'volume_fraction_target': config.volume_fraction,
        'volume_fraction_actual': float(result.volume_fraction),
        'optimization': {
            'n_iterations': result.n_iterations,
            'converged': result.converged,
            'final_objective': float(u_out),
            'time_seconds': round(opt_time, 2),
            'total_time_seconds': round(total_time, 2),
        },
        'mechanism': {
            'input_node': int(mech_problem.input_node),
            'input_direction': list(mech_problem.input_direction),
            'output_node': int(mech_problem.output_node),
            'output_direction': list(mech_problem.output_direction),
            'k_in': float(mech_problem.k_in),
            'k_out': float(mech_problem.k_out),
        },
        'seed_info': seed_info,
        'quality': {
            'seed_to_final_hausdorff': round(seed_to_final_hausdorff, 4),
            'seed_vf': seed_info.get('seed_vf', 0),
        },
        'validation': validation_info,
    }

    return {
        'sample_id': sample_id,
        'density': result.density,
        'density_init': density_init,
        'physics': physics_data,
        'metadata': metadata,
        'is_valid': is_valid,
    }


def _binary_hausdorff(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Approximate Hausdorff distance between two binary masks.

    Returns fraction of domain diagonal (0 = identical, 1 = maximally different).
    Uses one-sided max of minimum distances for speed.
    """
    from scipy.ndimage import distance_transform_edt

    if not mask_a.any() or not mask_b.any():
        return 1.0

    # Distance transform of complement
    dist_to_a = distance_transform_edt(~mask_a)
    dist_to_b = distance_transform_edt(~mask_b)

    # Max distance from B's boundary to nearest A boundary
    h_ab = dist_to_a[mask_b].max() if mask_b.any() else 0.0
    h_ba = dist_to_b[mask_a].max() if mask_a.any() else 0.0

    hausdorff = max(h_ab, h_ba)
    diagonal = np.sqrt(mask_a.shape[0]**2 + mask_a.shape[1]**2)
    return hausdorff / diagonal


def worker_fn(args):
    """Worker function for parallel generation."""
    sample_id, resolution, config_dict, linkage_type, compute_physics = args
    config = MechConfig(**config_dict)
    try:
        return generate_seeded_sample(
            sample_id, resolution, config,
            linkage_type=linkage_type,
            compute_physics=compute_physics,
        )
    except Exception as e:
        import traceback
        return {
            'sample_id': sample_id,
            'is_valid': False,
            'metadata': {'sample_id': sample_id, 'error': traceback.format_exc()},
        }


def save_sample(sample: dict, output_dir: str):
    """Save a single sample to disk."""
    sid = sample['sample_id']
    prefix = f"{sid:06d}"

    np.save(os.path.join(output_dir, f"{prefix}.density.npy"), sample['density'])

    if sample.get('density_init') is not None:
        np.save(os.path.join(output_dir, f"{prefix}.seed.npy"), sample['density_init'])

    if sample.get('physics') and sample['physics'].get('displacement') is not None:
        np.save(
            os.path.join(output_dir, f"{prefix}.displacement.npy"),
            sample['physics']['displacement'],
        )
    if sample.get('physics') and sample['physics'].get('stress_vm') is not None:
        np.save(
            os.path.join(output_dir, f"{prefix}.stress.npy"),
            sample['physics']['stress_vm'],
        )

    with open(os.path.join(output_dir, f"{prefix}.json"), 'w') as f:
        json.dump(sample['metadata'], f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description='Generate seeded mechanism samples (Phase B1)'
    )
    parser.add_argument('--n-samples', type=int, default=10)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--output-dir', type=str, default='data/test_seeded')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--linkage-type', type=str, default=None,
                        choices=list(SEED_GENERATORS.keys()),
                        help='Specific linkage type (default: random)')
    parser.add_argument('--volume-fraction', type=float, default=0.20)
    parser.add_argument('--max-iterations', type=int, default=400)
    parser.add_argument('--compute-physics', action='store_true')
    parser.add_argument('--start-id', type=int, default=0)
    parser.add_argument('--filter-radius', type=float, default=None,
                        help='Min length scale; default scales with resolution (~res/25.6)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Scale filter radius with resolution (2.5 @64, 5.0 @128) unless overridden.
    # With robust mode (default on) this is what eliminates single-pixel hinges.
    filter_radius = args.filter_radius if args.filter_radius is not None \
        else max(2.5, args.resolution / 25.6)
    config = MechConfig(
        volume_fraction=args.volume_fraction,
        max_iterations=args.max_iterations,
        filter_radius=filter_radius,
    )

    sample_ids = list(range(args.start_id, args.start_id + args.n_samples))

    print(f"=== Seeded Mechanism Generation ===")
    print(f"  Samples:    {args.n_samples}")
    print(f"  Resolution: {args.resolution}×{args.resolution}")
    print(f"  Linkage:    {args.linkage_type or 'mixed (random)'}")
    print(f"  VF:         {args.volume_fraction}")
    print(f"  Max iters:  {args.max_iterations}")
    print(f"  Workers:    {args.workers}")
    print(f"  Output:     {args.output_dir}")
    print()

    n_valid = 0
    n_failed = 0
    total_start = time.time()

    if args.workers <= 1:
        # Sequential
        for sid in sample_ids:
            t0 = time.time()
            sample = generate_seeded_sample(
                sid, args.resolution, config,
                linkage_type=args.linkage_type,
                compute_physics=args.compute_physics,
            )
            dt = time.time() - t0

            if sample['is_valid'] and sample['density'] is not None:
                save_sample(sample, args.output_dir)
                n_valid += 1
                obj = sample['metadata']['optimization']['final_objective']
                stype = sample['metadata']['problem_type']
                print(f"  [{sid:06d}] ✓ {stype:20s} obj={obj:.3f}  "
                      f"VF={sample['metadata']['volume_fraction_actual']:.3f}  "
                      f"{dt:.1f}s")
            else:
                n_failed += 1
                reason = sample['metadata'].get('failure_reason',
                         sample['metadata'].get('validation', {}).get('failure_reason', 'unknown'))
                print(f"  [{sid:06d}] ✗ {reason}")
    else:
        # Parallel
        config_dict = {
            'volume_fraction': config.volume_fraction,
            'max_iterations': config.max_iterations,
            'filter_radius': config.filter_radius,
            'k_in': config.k_in,
        }
        tasks = [
            (sid, args.resolution, config_dict, args.linkage_type, args.compute_physics)
            for sid in sample_ids
        ]

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(worker_fn, t): t[0] for t in tasks}
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    sample = future.result()
                    if sample['is_valid'] and sample.get('density') is not None:
                        save_sample(sample, args.output_dir)
                        n_valid += 1
                        obj = sample['metadata']['optimization']['final_objective']
                        print(f"  [{sid:06d}] ✓ obj={obj:.3f}")
                    else:
                        n_failed += 1
                        reason = sample.get('metadata', {}).get('failure_reason', 'unknown')
                        print(f"  [{sid:06d}] ✗ {reason}")
                except Exception as e:
                    n_failed += 1
                    print(f"  [{sid:06d}] ✗ exception: {e}")

    total_time = time.time() - total_start
    n_total = n_valid + n_failed
    rate = n_valid / n_total * 100 if n_total > 0 else 0

    print(f"\n=== Results ===")
    print(f"  Valid:   {n_valid}/{n_total} ({rate:.0f}%)")
    print(f"  Failed:  {n_failed}")
    print(f"  Time:    {total_time:.1f}s ({total_time/max(n_total,1):.1f}s/sample)")
    print(f"  Output:  {args.output_dir}")


if __name__ == '__main__':
    main()
