#!/usr/bin/env python3
"""Decisive test: can one FIXED spec yield topologically DISTINCT valid solutions?

Naive random-start multistart converges to one basin (dice ~0.93). This probes
whether varying the optimiser regime — filter radius (coarse<->fine length
scale) + structured initial fields — pushes the optimiser into DIFFERENT optima
for the same BVP. Prints pairwise Dice + a distinct-count. If Dice spreads down
to ~0.6, diverse-multistart is viable; if it stays ~0.9, SIMP is too convergent
for this spec and multimodality must come from other families (GS/MMC).
"""
import argparse, glob, itertools, json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def structured_init(problem, vf, seed, scale):
    """Coarser blobs at larger `scale` -> different topological basins."""
    rng = np.random.default_rng(seed)
    a = gaussian_filter(rng.standard_normal((problem.mesh.nely, problem.mesh.nelx)), scale)
    a = (a - a.mean()) / (a.std() + 1e-9)
    rho = np.clip(vf + 0.35 * a, 0.02, 0.98)          # bigger perturbation
    return np.where(problem.domain_mask, rho, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-iterations", type=int, default=200)
    args = ap.parse_args()
    from scripts.backfill_conditioning import problem_from_metadata
    from scripts.eval_mechanism_gate import gate
    from src.generators.mech import MechConfig, optimize_mechanism

    spec = json.load(open(args.meta))
    problem = problem_from_metadata(spec)
    vf = float(spec["volume_fraction_target"])
    nelx = problem.mesh.nelx
    # a small grid of regimes: (filter_radius divisor, blob scale)
    regimes = [(25.6, 2.0), (16.0, 2.0), (12.0, 3.0), (36.0, 4.0),
               (20.0, 5.0), (14.0, 2.5), (28.0, 3.5), (18.0, 6.0)]
    Ds, tags = [], []
    for k in range(args.n):
        fr_div, scale = regimes[k % len(regimes)]
        cfg = MechConfig(volume_fraction=vf, max_iterations=args.max_iterations,
                         filter_radius=max(2.5, nelx / fr_div), multires=False)
        res = optimize_mechanism(problem, cfg,
                                 initial_density=structured_init(problem, vf, 1000 + k, scale))
        rep = gate(spec, res.density, min_ga=0.25, min_selectivity=1.0,
                   min_exposure=0.5, vf_tol=0.10)
        ok = bool(rep["passed"])
        if ok:
            Ds.append(res.density > 0.5)
        tags.append(f"fr={nelx/fr_div:.1f},scale={scale}:{'PASS' if ok else 'rej'}")
        print(tags[-1], flush=True)
    print(f"\nvalid {len(Ds)}/{args.n}")
    if len(Ds) >= 2:
        def dice(a, b): return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9)
        ds = [dice(a, b) for a, b in itertools.combinations(Ds, 2)]
        print("pairwise dice: min=%.3f mean=%.3f max=%.3f" % (min(ds), sum(ds)/len(ds), max(ds)))
        distinct = 1 + sum(1 for i in range(1, len(Ds))
                           if all(dice(Ds[i], Ds[j]) < 0.85 for j in range(i)))
        print(f"distinct @dice<0.85: {distinct}/{len(Ds)}")


if __name__ == "__main__":
    main()
