#!/usr/bin/env python3
"""Generate stiff (compliance-minimized) structures in the UNIFIED dataset
schema — identical per-field .npy + .json layout as the mechanism pipeline
(scripts/generate_mech.py), so stiff samples fold straight into the same
dataset, audit, and cache tooling.

Per sample:
    000042.density.npy       (res, res)              design density
    000042.displacement.npy  (2, fine+1, fine+1)     nodal displacements (fine mesh)
    000042.stress.npy        (fine, fine)            von Mises stress
    000042.cond_energy.npy   (res, res)              uniform-load strain-energy prior
    000042.json              unified metadata

Schema difference vs mechanisms: stiff structures have LOADS + supports, not
input/output ports, so the JSON carries a 'loads' block instead of 'mechanism'
(and tier_name='stiff', family='S'). Everything else — optimization block,
boundary_conditions (re-derivable), validation, conditioning — matches.

Usage:
    PYTHONPATH=. python scripts/generate_stiff.py \
        --n-samples 2000 --resolution 128 --workers 30 --output-dir data/stiff_v1
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.generators.stiff import generate_stiff_sample, OptimizationConfig  # noqa: E402


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def compute_stiff_conditioning(problem, uniform_vf, resolution):
    """Uniform-density strain-energy raster under the sample's own loads/BCs —
    the same 'where do load paths want to run' conditioning channel the mech
    pipeline stores, computed from the spec alone (available at inference)."""
    from src.solvers.linear import (solve_fea, get_cached_edof,
                                    get_element_stiffness_cached)
    nely = nelx = resolution
    if problem.domain_mask is not None:
        uniform = np.where(problem.domain_mask, uniform_vf, 0.0).astype(np.float64)
    else:
        uniform = np.full((nely, nelx), uniform_vf, dtype=np.float64)
    res = solve_fea(problem, uniform, 3.0)
    mat = problem.material
    edof = get_cached_edof(nelx, nely)
    ke = get_element_stiffness_cached(mat.E, mat.nu)
    u_e = res.displacement[edof]
    uku = np.einsum("ij,jk,ik->i", u_e, ke, u_e)
    rho = uniform.flatten()
    E_interp = mat.E_min + np.power(np.maximum(rho, 0.0), 3.0) * (mat.E - mat.E_min)
    se = 0.5 * (E_interp / mat.E) * uku
    return se.reshape(nely, nelx)


def build_metadata(core_meta, result, problem, resolution, refinement_factor,
                   have_cond):
    """Unified schema (mirrors generate_mech.build) with a 'loads' block."""
    return {
        "sample_id": core_meta["sample_id"],
        "tier": 1,
        "tier_name": "stiff",
        "problem_type": core_meta["problem_type"],
        "family": "S",
        "rr_construction": None,
        "resolution": resolution,
        "volume_fraction_target": float(problem.volume_fraction),
        "volume_fraction_actual": float(result.volume_fraction),
        "optimization": {
            "n_iterations": result.n_iterations,
            "converged": result.converged,
            "final_objective": float(result.compliance),  # compliance (minimized)
            "time_seconds": round(result.time_seconds, 2),
        },
        # stiff analogue of 'mechanism': applied loads (node + force vector)
        "loads": [{"node": int(ld.node_index), "fx": float(ld.fx),
                   "fy": float(ld.fy)} for ld in problem.loads],
        "conditioning": ({"cond_energy": True,
                          "uniform_vf": float(problem.volume_fraction)}
                         if have_cond else None),
        "boundary_conditions": [
            {"nodes": [int(n) for n in bc.node_indices],
             "directions": [int(d) for d in np.atleast_1d(bc.directions)]}
            for bc in problem.bcs
        ],
        "validation": core_meta["validation"],
    }


def worker(task):
    sample_id, resolution, vf, config_dict, compute_physics, refine = task
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "1"
    config = OptimizationConfig(volume_fraction=vf, **config_dict)
    try:
        result, physics, problem, core_meta = generate_stiff_sample(
            sample_id=sample_id, resolution=resolution, volume_fraction=vf,
            config=config, compute_physics=compute_physics,
            refinement_factor=refine, max_retries=2)
    except Exception as e:
        return {"sample_id": sample_id, "is_valid": False, "error": str(e)}

    if result is None or core_meta is None:
        return {"sample_id": sample_id, "is_valid": False, "error": "no result"}

    is_valid = bool(core_meta["validation"].get("overall_passed", False))
    cond = None
    if is_valid:
        try:
            cond = compute_stiff_conditioning(
                problem, float(problem.volume_fraction), resolution).astype(np.float32)
        except Exception as e:
            core_meta.setdefault("validation", {})["conditioning_error"] = str(e)

    meta = build_metadata(core_meta, result, problem, resolution, refine,
                          cond is not None)
    return {
        "sample_id": sample_id,
        "is_valid": is_valid,
        "density": result.density.astype(np.float32),
        "physics": ({"displacement": physics.displacement,
                     "stress_vm": physics.stress_vm} if physics else None),
        "conditioning_field": cond,
        "metadata": meta,
    }


def save_sample(output_dir: Path, r: dict):
    prefix = f"{int(r['sample_id']):06d}"
    np.save(output_dir / f"{prefix}.density.npy", r["density"].astype(np.float32))
    if r.get("physics"):
        if r["physics"].get("displacement") is not None:
            np.save(output_dir / f"{prefix}.displacement.npy",
                    r["physics"]["displacement"].astype(np.float32))
        if r["physics"].get("stress_vm") is not None:
            np.save(output_dir / f"{prefix}.stress.npy",
                    r["physics"]["stress_vm"].astype(np.float32))
    if r.get("conditioning_field") is not None:
        np.save(output_dir / f"{prefix}.cond_energy.npy", r["conditioning_field"])
    with open(output_dir / f"{prefix}.json", "w") as f:
        json.dump(json_safe(r["metadata"]), f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--volume-fraction", type=float, default=0.35)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output-dir", type=str, default="data/stiff_test")
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--max-iterations", type=int, default=200)
    ap.add_argument("--no-physics", action="store_true")
    ap.add_argument("--refinement-factor", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_dict = {"max_iterations": args.max_iterations}
    compute_physics = not args.no_physics
    tasks = [(sid, args.resolution, args.volume_fraction, config_dict,
              compute_physics, args.refinement_factor)
             for sid in range(args.start_id, args.start_id + args.n_samples)]

    n_valid = 0
    t0 = time.time()
    print(f"[stiff] {args.n_samples} samples @ {args.resolution}² "
          f"({args.workers} workers) -> {out}", flush=True)
    if args.workers == 1:
        for task in tqdm(tasks, desc="stiff"):
            r = worker(task)
            if r.get("is_valid"):
                save_sample(out, r); n_valid += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(worker, t): t for t in tasks}
            for fut in tqdm(as_completed(futs), total=len(tasks), desc="stiff"):
                r = fut.result()
                if r.get("is_valid"):
                    save_sample(out, r); n_valid += 1
    dt = time.time() - t0
    print(f"[stiff] done: {n_valid}/{args.n_samples} valid "
          f"({100*n_valid/args.n_samples:.0f}%) in {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
