#!/usr/bin/env python3
"""Regression checks for the sparse mechanism verification path.

Run this against a frozen corpus before reporting a new evaluation table.  It
checks the bugs that historically made a convincing-looking hero figure
untrustworthy: sign-insensitive function passes, optional interface policy,
domain masking, and the duplicated 2x stress kernel.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backfill_conditioning import problem_from_metadata  # noqa: E402
from scripts.eval_mechanism_gate import canonicalize_density, gate, load_protocol  # noqa: E402
from src.generators.mech import (  # noqa: E402
    compute_mechanism_response_fields, compute_von_mises_stress, solve_mechanism_fea,
)
from src.solvers.linear import compute_von_mises_stress_vectorized  # noqa: E402


def find_record(manifest_path: Path, predicate):
    with manifest_path.open() as f:
        manifest = json.load(f)
    for rec in manifest["records"]:
        with open(rec["stem"] + ".json") as f:
            meta = json.load(f)
        if predicate(meta):
            return rec["stem"], meta
    return None, None


def check_stress_parity(meta: dict, rho: np.ndarray) -> tuple[bool, float]:
    problem = problem_from_metadata(meta)
    u, _K, _out = solve_mechanism_fea(problem, rho)
    old_public = compute_von_mises_stress(problem.base_problem, u, rho, 3.0)
    mat = problem.material
    canonical = compute_von_mises_stress_vectorized(
        problem.mesh.nelx, problem.mesh.nely, u, rho, E=mat.E, nu=mat.nu,
        E_min=mat.E_min, penal=3.0)
    rel = float(np.max(np.abs(old_public - canonical)) / (np.max(np.abs(canonical)) + 1e-12))
    fields, u2, out2 = compute_mechanism_response_fields(problem, rho)
    consistency = max(float(np.max(np.abs(u - u2))), abs(float(_out) - float(out2)),
                      float(np.max(np.abs(fields.stress_vm - canonical))))
    return rel < 1e-10 and consistency < 1e-10, max(rel, consistency)


def coarse_domain_coverage(domain: np.ndarray, source_resolution: int,
                           model_resolution: int = 64):
    """Return 64px source-area coverage when its block mapping is exact."""
    if source_resolution % model_resolution:
        return None
    factor = source_resolution // model_resolution
    if factor < 1:
        return None
    return np.asarray(domain, dtype=float).reshape(
        model_resolution, factor, model_resolution, factor).mean(axis=(1, 3))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--protocol", default=str(ROOT / "config" / "evaluation_protocol.v1.json"))
    args = ap.parse_args()
    manifest = Path(args.manifest)
    protocol = load_protocol(args.protocol)

    # A normal functional target verifies sparse solve/stress parity.
    stem, meta = find_record(manifest, lambda m: bool((m.get("validation", {}) or {}).get("overall_passed", False)))
    if stem is None:
        raise SystemExit("no valid source record")
    rho = np.load(stem + ".density.npy")
    canonical, _ = canonicalize_density(meta, rho)
    stress_ok, stress_error = check_stress_parity(meta, canonical)
    base = gate(meta, rho, protocol=protocol, require_port_access=False)

    # Spring stiffness is direction-sign invariant, so this isolates whether
    # the verifier actually checks the signed desired output response.
    flipped = copy.deepcopy(meta)
    flipped["mechanism"]["output_direction"] = [
        -float(v) for v in flipped["mechanism"]["output_direction"]]
    sign = gate(flipped, rho, protocol=protocol, require_port_access=False)
    sign_ok = not sign["functional_passed"] and (
        "positive_output" in sign.get("failure_reasons", [])
        or "signed_ga" in sign.get("failure_reasons", []))

    # An embedded source should be mechanically testable in broad mode, while
    # the strict interface policy remains a distinct result.
    embedded_stem, embedded_meta = find_record(
        manifest, lambda m: not bool((m.get("validation", {}) or {}).get("port_interface", {}).get("passed", False)))
    interface_ok = True
    if embedded_stem is not None:
        embedded_rho = np.load(embedded_stem + ".density.npy")
        broad = gate(embedded_meta, embedded_rho, protocol=protocol, require_port_access=False)
        strict = gate(embedded_meta, embedded_rho, protocol=protocol, require_port_access=True)
        interface_ok = broad["functional_passed"] and not strict["passed"]

    # If a source has a restricted domain, forbidden source-grid material must
    # be visible in the report and unable to help the solve after
    # canonicalization.
    masked_stem, masked_meta = find_record(
        manifest, lambda m: bool(np.any(problem_from_metadata(m).domain_mask == 0)))
    domain_ok = True
    if masked_stem is not None:
        masked_rho = np.load(masked_stem + ".density.npy")
        domain = problem_from_metadata(masked_meta).domain_mask
        proposal = np.asarray(masked_rho, dtype=float).copy()
        proposal[~domain] = 1.0
        report = gate(masked_meta, proposal, protocol=protocol, require_port_access=False)
        canonical, _ = canonicalize_density(masked_meta, proposal)
        source_grid_rejects = (
            report["density_transform"]["outside_domain_mean_before_mask"] > 0.99
            and "inside_domain" in report.get("failure_reasons", [])
            and np.all(canonical[~domain] == 0.0)
        )
        domain_ok = source_grid_rejects

    # For 64→128, distinguish mixed boundary alias from controllable leakage.
    # A coarse pixel that straddles the source mask cannot encode its void half;
    # it is recorded but does not fail.  A zero-coverage coarse cell is fully
    # controllable and must fail the domain check.
    mixed_stem, mixed_meta = find_record(
        manifest,
        lambda m: (
            (coverage := coarse_domain_coverage(
                problem_from_metadata(m).domain_mask, int(m["resolution"]))) is not None
            and bool(np.any((coverage > 0) & (coverage < 1)))
            and bool(np.any(coverage == 0))
        ))
    coarse_alias_ok = True
    if mixed_stem is not None:
        mixed_domain = problem_from_metadata(mixed_meta).domain_mask
        coverage = coarse_domain_coverage(mixed_domain, int(mixed_meta["resolution"]))
        partial_y, partial_x = np.argwhere((coverage > 0) & (coverage < 1))[0]
        void_y, void_x = np.argwhere(coverage == 0)[0]
        partial_proposal = np.zeros((64, 64), dtype=float)
        partial_proposal[partial_y, partial_x] = 1.0
        _rho_partial, partial_transform = canonicalize_density(mixed_meta, partial_proposal)
        void_proposal = np.zeros((64, 64), dtype=float)
        void_proposal[void_y, void_x] = 1.0
        _rho_void, void_transform = canonicalize_density(mixed_meta, void_proposal)
        void_gate = gate(mixed_meta, void_proposal, protocol=protocol,
                         require_port_access=False)
        coarse_alias_ok = (
            partial_transform["pre_mask_outside_policy"]
            == "reject_zero_coverage_coarse_cells_record_mixed_boundary_alias"
            and partial_transform["outside_domain_max_aliasable_before_mask"] > 0.99
            and partial_transform["outside_domain_max_avoidable_before_mask"] == 0.0
            and void_transform["outside_domain_max_avoidable_before_mask"] > 0.99
            and not bool(void_gate.get("functional", {}).get("checks", {}).get("inside_domain", True))
        )

    checks = {
        "stress_kernel_parity": stress_ok,
        "signed_output_rejection": sign_ok,
        "embedded_port_policy": interface_ok,
        "domain_mask_enforced": domain_ok,
        "coarse_domain_alias_partition": coarse_alias_ok,
        "base_target_has_sparse_result": "fea_error" not in base,
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    print(json.dumps({"checks": checks, "stress_max_relative_error": stress_error,
                      "base_stem": stem, "embedded_stem": embedded_stem,
                      "masked_stem": masked_stem, "mixed_domain_stem": mixed_stem,
                      "verdict": "PASS" if all(checks.values()) else "FAIL"}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
