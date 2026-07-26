#!/usr/bin/env python3
"""Backfill <prefix>.cond_energy.npy for samples generated before the
conditioning field was added.

The field is derivable from the problem spec alone, and the problem is
reconstructible from (problem_type, sample_id): production seeds every sample
with sample_id*12345+67890. We rebuild the problem the exact same way,
VERIFY the rebuilt ports/springs against the saved metadata (catching RNG
drift from generator code changes), and only then solve + save. Mismatches
are skipped and reported — never silently mislabeled.

Sign tolerance: output_direction may differ in SIGN from the saved metadata
(the seeds.py slider sign fix landed after pilot_v1 was launched). That is
acceptable for this field: the output spring enters as k*(u·d)^2, which is
sign-invariant, and the input load direction is verified strictly.

Usage:
    PYTHONPATH=. python3 scripts/backfill_conditioning.py \
        --dir data/pilot_v1/rr_lever --workers 30
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def problem_from_metadata(meta):
    """Reconstruct the MechProblem from STORED spec (BCs + ports + springs) —
    generator-independent, so it works even after the generator code changed
    (e.g. fact_rotation's cartwheel rework invalidated seed-rebuild for the
    pilot_v1 samples). Requires 'boundary_conditions' in the metadata.
    """
    import numpy as np
    from src.core.problem import (Problem, ProblemType, Material,
                                  BoundaryCondition, Load)
    from src.core.mesh import create_mesh
    from src.generators.mech import MechProblem
    from src.ml.tensor_spec import decode_mask_rle

    res = int(meta["resolution"])
    mech = meta["mechanism"]
    mesh = create_mesh(res, res)
    bcs = [BoundaryCondition(node_indices=np.asarray(bc["nodes"], dtype=int),
                             directions=np.asarray(bc["directions"], dtype=int))
           for bc in meta["boundary_conditions"]]
    idir = mech["input_direction"]
    loads = [Load(node_index=int(mech["input_node"]),
                  fx=float(idir[0]), fy=float(idir[1]))]
    base = Problem(mesh=mesh, material=Material(E=1.0, nu=0.3),
                   bcs=bcs, loads=loads,
                   volume_fraction=float(meta["volume_fraction_target"]),
                   problem_type=ProblemType.MECHANISM,
                   domain_mask=decode_mask_rle(meta.get("domain_mask_rle"), res).astype(bool))
    return MechProblem(
        base_problem=base,
        input_node=int(mech["input_node"]),
        input_direction=tuple(mech["input_direction"]),
        output_node=int(mech["output_node"]),
        output_direction=tuple(mech["output_direction"]),
        k_in=float(mech["k_in"]), k_out=float(mech["k_out"]),
        k_perp=float(mech.get("k_perp", 0.0)))


def backfill_one(json_path):
    from scripts.generate_mech import (
        compute_conditioning_field, MECH_GENERATORS)
    from src.generators.mech import MechConfig, generate_random_mechanism
    from src.generators.rigid_replace import RR_CONSTRUCTORS

    stem = json_path[:-5]
    out_path = stem + ".cond_energy.npy"
    with open(json_path) as f:
        meta = json.load(f)
    need_cond = not os.path.exists(out_path)
    need_sel = "port_selectivity" not in meta.get("validation", {})
    need_motion = "motion" not in meta.get("validation", {})
    if not need_cond and not need_sel and not need_motion:
        return "exists"
    ptype = meta["problem_type"]
    sample_id = int(meta["sample_id"])
    resolution = int(meta["resolution"])
    seed = (sample_id * 12345 + 67890) % (2 ** 32)
    mech_meta = meta["mechanism"]

    # Selectivity needs only the problem spec (BCs+ports+springs) + density —
    # not the generator. When only selectivity is missing and the metadata
    # carries boundary_conditions, take the generator-INDEPENDENT path so
    # generator code changes (e.g. fact_rotation cartwheel rework) don't block
    # the backfill.
    if (need_sel or need_motion) and not need_cond \
            and "boundary_conditions" in meta:
        try:
            problem = problem_from_metadata(meta)
            dens = np.load(stem + ".density.npy").astype(float)
            v = meta.setdefault("validation", {})
            if need_sel:
                from src.validation.compliance import port_selectivity
                v["port_selectivity"] = port_selectivity(problem, dens)
            if need_motion:
                from src.validation.motion_class import motion_class
                v["motion"] = motion_class(problem, dens)
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)
            return "ok_meta"
        except Exception as e:  # noqa: BLE001
            return f"error_meta:{type(e).__name__}"

    try:
        if ptype in RR_CONSTRUCTORS:
            rng = np.random.RandomState(seed)
            built = RR_CONSTRUCTORS[ptype](nelx=resolution, nely=resolution,
                                           rng=rng)
            if built is None:
                return "rebuild_failed"
            _d, problem, _m = built
        else:
            cfg = MechConfig()
            gen = MECH_GENERATORS.get(ptype, generate_random_mechanism)
            owns = ptype in ("random", "amplifier")
            problem = gen(nelx=resolution, nely=resolution,
                          volume_fraction=meta["volume_fraction_target"],
                          k_in=None if owns else cfg.k_in,
                          k_out=None if owns else cfg.k_out,
                          seed=seed)

        # verify the rebuild reproduces the saved sample's problem
        if problem.input_node != mech_meta["input_node"]:
            return "mismatch_input_node"
        if problem.output_node != mech_meta["output_node"]:
            return "mismatch_output_node"
        din = np.array(mech_meta["input_direction"], dtype=float)
        if np.dot(din, np.array(problem.input_direction)) < 0.999:
            return "mismatch_input_dir"
        dout = np.array(mech_meta["output_direction"], dtype=float)
        if abs(np.dot(dout, np.array(problem.output_direction))) < 0.999:
            return "mismatch_output_dir"
        for key in ("k_in", "k_out"):
            if abs(getattr(problem, key) - mech_meta[key]) > 1e-9:
                return f"mismatch_{key}"

        if need_cond:
            field = compute_conditioning_field(
                problem, float(meta["volume_fraction_target"]), resolution)
            np.save(out_path, field.astype(np.float32))
            meta["conditioning"] = {
                "cond_energy": True,
                "uniform_vf": float(meta["volume_fraction_target"]),
                "backfilled": True,
            }
        # port compliance selectivity (added 2026-07-17, after pilot_v1) —
        # needs the saved density; rebuild-verify already passed above so the
        # problem is the true one
        if need_sel or need_motion:
            dens = np.load(stem + ".density.npy").astype(float)
            v = meta.setdefault("validation", {})
            if need_sel:
                from src.validation.compliance import port_selectivity
                v["port_selectivity"] = port_selectivity(problem, dens)
            if need_motion:
                from src.validation.motion_class import motion_class
                v["motion"] = motion_class(problem, dens)
        # also backfill the BC patches (re-derivability; inline generation
        # stores these since 2026-07-17)
        if "boundary_conditions" not in meta:
            meta["boundary_conditions"] = [
                {"nodes": [int(n) for n in bc.node_indices],
                 "directions": [int(d) for d in bc.directions]}
                for bc in problem.base_problem.bcs
            ]
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"error:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="Sample directory (searched recursively)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "**", "*.json"),
                             recursive=True))
    files = [f for f in files if not f.endswith("run.log")]
    print(f"{len(files)} samples under {args.dir}")
    with Pool(args.workers) as pool:
        results = pool.map(backfill_one, files, chunksize=8)
    from collections import Counter
    counts = Counter(results)
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    ok = counts.get("ok", 0) + counts.get("ok_meta", 0)
    bad = sum(v for k, v in counts.items()
              if k not in ("ok", "ok_meta", "exists"))
    print(f"done: {ok} backfilled ({counts.get('ok_meta', 0)} via metadata), "
          f"{counts.get('exists', 0)} already present, {bad} skipped/failed")


if __name__ == "__main__":
    main()
