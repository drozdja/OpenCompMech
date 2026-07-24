#!/usr/bin/env python3
"""Optimise multiple distinct layouts for one *fixed* mechanism specification.

The old corpus changed the BVP with the seed, so it taught one answer per
condition.  This tool deliberately fixes BCs, ports, springs, envelope and VF
from ``--meta`` and varies only the topology-optimisation initial field.  Each
candidate is independently FEA-gated before it is saved.  Run it for a balanced
set of useful specifications, then curate with curate_quality_diversity.py.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def initial_field(problem, vf, seed):
    rng = np.random.default_rng(seed)
    a = gaussian_filter(rng.standard_normal((problem.mesh.nely, problem.mesh.nelx)), 2.0)
    a = (a - a.mean()) / (a.std() + 1e-9)
    rho = np.clip(vf + 0.12 * a, 0.02, 0.98)
    return np.where(problem.domain_mask, rho, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True, help="JSON containing the fixed BVP")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iterations", type=int, default=300)
    args = ap.parse_args()
    from scripts.backfill_conditioning import problem_from_metadata
    from scripts.eval_mechanism_gate import gate, _json_default
    from scripts.generate_mech import compute_conditioning_field
    from src.generators.mech import MechConfig, optimize_mechanism

    with open(args.meta) as f:
        spec_meta = json.load(f)
    problem = problem_from_metadata(spec_meta)
    vf = float(spec_meta["volume_fraction_target"])
    config = MechConfig(volume_fraction=vf, max_iterations=args.max_iterations,
                        filter_radius=max(2.5, problem.mesh.nelx / 25.6), multires=False)
    os.makedirs(args.out_dir, exist_ok=True)
    # Same BVP => same energy prior.  Store a copy with every accepted design
    # so sample directories remain self-contained and hashable.
    cond_energy = compute_conditioning_field(problem, vf, problem.mesh.nelx).astype(np.float32)
    accepted = 0
    for offset in range(args.n):
        solution_id = args.start_id + offset
        result = optimize_mechanism(problem, config,
                                    initial_density=initial_field(problem, vf, args.seed + solution_id))
        report = gate(spec_meta, result.density, min_ga=0.25, min_selectivity=1.0,
                      min_exposure=0.5, vf_tol=0.10)
        if not report["passed"]:
            print(f"{solution_id}: reject")
            continue
        prefix = os.path.join(args.out_dir, f"{solution_id:06d}")
        meta = copy.deepcopy(spec_meta)
        meta["sample_id"] = solution_id
        meta["optimization"] = {"n_iterations": result.n_iterations,
                                "converged": bool(result.converged),
                                "time_seconds": result.time_seconds,
                                "multistart": True}
        meta["provenance"] = {"generator": "generate_multisolution_bank.py",
                              "seed": args.seed + solution_id,
                              # all alternatives intentionally share lineage
                              "lineage_id": f"fixed_bvp:{os.path.abspath(args.meta)}"}
        # Copy geometry into a FRESH dict (don't alias report["geometry"], then
        # store report under ["gate"] -> that closes a cycle json.dump rejects).
        val = dict(report.get("geometry", {}))
        val["mechanism_path"] = report.get("mechanism_path")
        val["motion"] = report.get("motion", {})
        val["port_selectivity"] = report.get("port_selectivity", {})
        val["port_exposure"] = report.get("port_exposure", {})
        # gate summary WITHOUT the geometry alias that caused the cycle
        val["gate"] = {k: v for k, v in report.items() if k != "geometry"}
        val["overall_passed"] = True
        meta["validation"] = val
        np.save(prefix + ".density.npy", result.density.astype(np.float32))
        np.save(prefix + ".cond_energy.npy", cond_energy)
        with open(prefix + ".json", "w") as f:
            json.dump(meta, f, indent=2, default=_json_default)
        accepted += 1
        print(f"{solution_id}: PASS ({accepted}/{offset+1})")
    print(f"accepted {accepted}/{args.n}")


if __name__ == "__main__":
    main()
