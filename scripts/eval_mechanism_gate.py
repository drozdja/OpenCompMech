#!/usr/bin/env python3
"""Fresh full-resolution sparse-FEA verification for mechanism proposals.

This command reconstructs the boundary-value problem stored with a source
sample and re-evaluates a proposal at that source resolution.  It is separate
from the 64px differentiable guidance proxy, but it deliberately does *not*
claim an unrelated physics implementation: both use the same stated linear
elastic material model.  The useful claim is narrower and testable:

    fresh, spring-aware, full-resolution sparse-FEA re-evaluation

The protocol is versioned in ``config/evaluation_protocol.v1.json``.  Its
functional result and optional interface result are reported separately, since
an embedded port is a real interface limitation but is not evidence that the
underlying normalized mechanism BVP failed.

Stress is normalized diagnostic output only.  Nothing here establishes yield,
fatigue, buckling, contact, manufacturability, or flight qualification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_PROTOCOL_PATH = ROOT / "config" / "evaluation_protocol.v1.json"


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def score_gate_task(task: tuple[str, np.ndarray, str]) -> dict:
    """Spawn-safe CPU worker entry point used by ``eval_harness.py``.

    It deliberately lives in this light-weight module rather than the Torch
    evaluator, so 24-way sparse-FEA screening never forks a live ROCm/CUDA
    context or imports a model into every CPU worker.
    """
    stem, density, protocol_path = task
    try:
        with open(stem + ".json") as f:
            meta = json.load(f)
        return gate(meta, density, protocol_path=protocol_path,
                    require_port_access=False, include_stress=False)
    except Exception as exc:  # never drop a failed candidate from the denominator
        return {"passed": False, "functional_passed": False,
                "interface_passed": False, "failure_reasons": ["worker_error"],
                "worker_error": f"{type(exc).__name__}: {exc}"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_protocol(path: str | os.PathLike | None = None) -> dict:
    """Load a frozen verification contract and attach stable provenance."""
    protocol_path = Path(path) if path else DEFAULT_PROTOCOL_PATH
    with protocol_path.open() as f:
        protocol = json.load(f)
    if protocol.get("format") != "opencompmech.evaluation-protocol.v1":
        raise ValueError(f"unsupported evaluation protocol: {protocol.get('format')!r}")
    protocol = copy.deepcopy(protocol)
    protocol["_path"] = str(protocol_path.resolve())
    protocol["_sha256"] = _sha256_file(protocol_path)
    return protocol


def _two_dimensional_density(density: np.ndarray) -> np.ndarray:
    """Accept HxW, 1xHxW, or 1x1xHxW without silently reinterpreting batches."""
    a = np.asarray(density, dtype=float)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"expected a single square density, got shape {a.shape}")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"expected square density, got {a.shape}")
    return a


def density_at_spec_resolution(density: np.ndarray, res: int) -> np.ndarray:
    """Map one proposal to the source mesh without inventing gray material.

    Exact integer 64→128 mappings use block replication.  Generic mappings use
    nearest-neighbour interpolation.  Bilinear interpolation was inappropriate
    here: it changes volume/gray-fraction gates and made the functional and
    stress panels describe different realizations of the same proposal.
    """
    a = _two_dimensional_density(density)
    if a.shape == (res, res):
        return np.clip(a, 0.0, 1.0)
    if res % a.shape[0] == 0:
        factor = res // a.shape[0]
        return np.repeat(np.repeat(np.clip(a, 0.0, 1.0), factor, axis=0),
                         factor, axis=1)
    factor = res / float(a.shape[0])
    return np.clip(zoom(a, (factor, factor), order=0), 0.0, 1.0)


# Backward-compatible private name used by a few exploratory scripts.
_density_at_spec_resolution = density_at_spec_resolution


def _coarse_domain_partition(domain: np.ndarray, input_shape: tuple[int, int]):
    """Partition source-grid void cells for an integral coarse raster.

    A 64px proposal cannot distinguish the two source cells in a 64→128 block.
    If a block straddles the exact domain boundary, density in its source-void
    half is an unavoidable representation alias and is removed by the declared
    exact-domain realization.  In contrast, a coarse cell with zero source
    coverage is fully controllable and must remain void.  This routine makes
    that distinction explicit rather than either rejecting all coarse samples
    or silently accepting arbitrary exterior material.

    Non-integral resampling has no unambiguous block ownership, so it receives
    the conservative strict policy: every source-void cell is avoidable.
    """
    domain = np.asarray(domain, dtype=bool)
    h, w = (int(input_shape[0]), int(input_shape[1]))
    H, W = domain.shape
    outside = ~domain
    if h <= 0 or w <= 0 or H % h or W % w:
        return {
            "coverage": None,
            "aliasable": np.zeros_like(domain, dtype=bool),
            "avoidable": outside,
            "integral_mapping": False,
            "policy": "reject_all_source_void_nonintegral_mapping",
        }

    fy, fx = H // h, W // w
    coverage = domain.reshape(h, fy, w, fx).mean(axis=(1, 3))
    coarse_active = coverage > 0
    source_cells_represented = np.repeat(np.repeat(coarse_active, fy, axis=0), fx, axis=1)
    aliasable = outside & source_cells_represented
    return {
        "coverage": coverage,
        "aliasable": aliasable,
        "avoidable": outside & ~source_cells_represented,
        "integral_mapping": True,
        "policy": ("reject_all_source_void" if (h, w) == (H, W)
                   else "reject_zero_coverage_coarse_cells_record_mixed_boundary_alias"),
    }


def _masked_mean_and_max(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Stable JSON-friendly stats, including for an empty mask."""
    if not bool(np.any(mask)):
        return 0.0, 0.0
    selected = np.asarray(values)[mask]
    return float(selected.mean()), float(selected.max())


def canonicalize_density(meta: dict, density: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return the one density realization used by *all* verifier checks.

    Material outside the declared domain is measured and then zeroed before
    geometry, connectivity, FEA, stress, or visualisation.  A proposal cannot
    obtain a free load path through forbidden cells.  For an integral coarse
    proposal (the current 64px model rendered onto a 128px source mesh), the
    report separates fully forbidden coarse cells from mixed boundary cells:
    the former are avoidable contract violations, while the latter are an
    unavoidable alias recorded alongside the exact-masked physical result.
    """
    from scripts.backfill_conditioning import problem_from_metadata

    problem = problem_from_metadata(meta)
    input_shape = _two_dimensional_density(density).shape
    source_resolution = int(meta["resolution"])
    rho_raw = density_at_spec_resolution(density, source_resolution)
    domain = np.asarray(problem.domain_mask, dtype=bool)
    if domain.shape != rho_raw.shape:
        raise ValueError(f"domain shape {domain.shape} disagrees with density {rho_raw.shape}")
    outside = ~domain
    partition = _coarse_domain_partition(domain, input_shape)
    outside_mean, outside_max = _masked_mean_and_max(rho_raw, outside)
    alias_mean, alias_max = _masked_mean_and_max(rho_raw, partition["aliasable"])
    avoidable_mean, avoidable_max = _masked_mean_and_max(rho_raw, partition["avoidable"])
    coverage = partition["coverage"]
    rho = np.where(domain, rho_raw, 0.0)
    return rho, {
        "source_resolution": source_resolution,
        "input_resolution": int(input_shape[0]),
        "input_shape": [int(input_shape[0]), int(input_shape[1])],
        "resample": "nearest",
        "outside_domain_mean_before_mask": outside_mean,
        "outside_domain_max_before_mask": outside_max,
        "outside_domain_mean_aliasable_before_mask": alias_mean,
        "outside_domain_max_aliasable_before_mask": alias_max,
        "outside_domain_mean_avoidable_before_mask": avoidable_mean,
        "outside_domain_max_avoidable_before_mask": avoidable_max,
        "pre_mask_outside_policy": partition["policy"],
        "coarse_domain_alias": {
            "integral_mapping": bool(partition["integral_mapping"]),
            "fractional_coverage_input_cells": (int(((coverage > 0) & (coverage < 1)).sum())
                                                       if coverage is not None else None),
            "aliasable_source_void_cells": int(partition["aliasable"].sum()),
            "avoidable_source_void_cells": int(partition["avoidable"].sum()),
        },
    }


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size < 2 or values.sum() <= 1e-15:
        return 0.0
    ordered = np.sort(values)
    n = ordered.size
    return float((2.0 * np.dot(np.arange(1, n + 1), ordered)
                  - (n + 1) * ordered.sum()) / (n * ordered.sum()))


def resolve_gini_cap(meta: dict, family: str, simp_default_cap: float) -> tuple:
    """Resolve the energy-Gini acceptance policy for one design.

    The energy-Gini bound is a strain-energy localization / lumped-compliance
    heuristic (distinct from the separate ``no_point_hinge`` check).  It is only
    meaningful for SIMP density fields; constructed parametric families (FACT
    flexures, MMC bars, ground-structure trusses, rigid-replace levers)
    concentrate strain energy by design.  The generator recorded that exact
    decision in immutable metadata as ``validation.quality.gini_cap`` (``0.75``
    for SIMP family A, ``null`` for every constructed design).  Honour that
    recorded source contract rather than re-inferring it from a family label.
    Only when a design predates that field do we fall back to the protocol SIMP
    cap on family A.

    Returns ``(gini_required, gini_cap, source)``.
    """
    quality_meta = (meta.get("validation", {}) or {}).get("quality", {}) or {}
    if "gini_cap" in quality_meta:
        gini_cap = quality_meta["gini_cap"]
        source = "source_metadata_contract"
    else:
        gini_cap = float(simp_default_cap) if family == "A" else None
        source = "family_fallback_no_recorded_contract"
    return gini_cap is not None, gini_cap, source


def _quality_metrics(problem, rho: np.ndarray, u: np.ndarray, u_out: float,
                     penal: float = 3.0) -> dict:
    """Production-aligned response measurements from the same sparse solve."""
    from src.solvers.linear import get_cached_edof, get_element_stiffness_cached

    in_x, in_y = problem.get_input_dofs()
    in_dir = np.asarray(problem.input_direction, dtype=float)
    in_dir /= np.linalg.norm(in_dir) + 1e-30
    out_dir = np.asarray(problem.output_direction, dtype=float)
    out_dir /= np.linalg.norm(out_dir) + 1e-30
    u_input = np.asarray([u[in_x], u[in_y]], dtype=float)
    out_x, out_y = problem.get_output_dofs()
    u_output = np.asarray([u[out_x], u[out_y]], dtype=float)
    u_in_projected = float(u_input @ in_dir)
    output_norm = float(np.linalg.norm(u_output))
    output_alignment = (float(u_output @ out_dir) / output_norm
                        if output_norm > 1e-15 else 0.0)
    input_norm = float(np.linalg.norm(u_input))
    input_perp = np.asarray([-in_dir[1], in_dir[0]])
    off_axis_input = (float(abs(u_input @ input_perp)) / input_norm
                      if input_norm > 1e-15 else float("inf"))
    ga_signed = float(u_out / u_in_projected) if abs(u_in_projected) > 1e-15 else 0.0

    # Same per-element SIMP energy expression as the production quality audit.
    nely, nelx = rho.shape
    mat = problem.material
    edof = get_cached_edof(nelx, nely)
    ke = get_element_stiffness_cached(mat.E, mat.nu)
    u_e = u[edof]
    uku = np.einsum("ij,jk,ik->i", u_e, ke, u_e)
    e_eff = (mat.E_min
             + np.power(np.maximum(rho.reshape(-1), 0.0), penal)
             * (mat.E - mat.E_min))
    strain_energy = 0.5 * (e_eff / mat.E) * uku
    gini = _gini(strain_energy[rho.reshape(-1) > 0.5])
    return {
        "u_input_vector": u_input.tolist(),
        "u_output_vector": u_output.tolist(),
        "u_in_projected": u_in_projected,
        "u_out_projected": float(u_out),
        "ga_signed": ga_signed,
        "output_alignment": output_alignment,
        "off_axis_input": off_axis_input,
        "energy_gini": gini,
    }


def _source_ga_floor(meta: dict, protocol: dict, override: float | None) -> float:
    if override is not None:
        return float(override)
    function = protocol["function"]
    if function.get("use_source_ga_floor", False):
        quality = (meta.get("validation", {}) or {}).get("quality", {}) or {}
        floor = quality.get("ga_floor")
        if floor is not None:
            return float(floor)
    return float(function["default_min_signed_ga"])


def gate(meta: dict, density: np.ndarray, min_ga: float | None = None,
         min_selectivity: float | None = None, min_exposure: float | None = None,
         vf_tol: float | None = None, *, protocol: dict | None = None,
         protocol_path: str | os.PathLike | None = None,
         require_port_access: bool | None = None,
         include_stress: bool = False) -> dict:
    """Evaluate one proposal under the frozen sparse-FEA protocol.

    ``functional_passed`` never silently includes interface accessibility.
    ``interface_passed`` is the stricter result.  The legacy ``passed`` field
    follows ``require_port_access`` and exists only for command compatibility.
    """
    from scripts.backfill_conditioning import problem_from_metadata
    from src.generators.mech import compute_mechanism_response_fields, solve_mechanism_fea
    from src.validation.compliance import port_selectivity
    from src.validation.connectivity import (
        check_bc_connectivity, check_mechanism_path_connectivity, detect_hinges,
        validate_sample,
    )
    from src.validation.motion_class import motion_class
    from src.validation.ports import problem_port_exposure

    protocol = copy.deepcopy(protocol) if protocol is not None else load_protocol(protocol_path)
    # Callers which supply a plain dict should still get traceable protocol data.
    protocol.setdefault("_path", str(protocol_path or DEFAULT_PROTOCOL_PATH))
    if "_sha256" not in protocol and Path(protocol["_path"]).is_file():
        protocol["_sha256"] = _sha256_file(Path(protocol["_path"]))
    geometry_cfg = protocol["geometry"]
    function_cfg = protocol["function"]
    interface_cfg = protocol["interface"]
    vf_tol = float(geometry_cfg["vf_relative_tolerance"] if vf_tol is None else vf_tol)
    min_selectivity = float(function_cfg["min_output_selectivity"]
                            if min_selectivity is None else min_selectivity)
    min_exposure = float(interface_cfg["min_clearance"]
                         if min_exposure is None else min_exposure)
    require_port_access = (bool(interface_cfg["default_require_port_access"])
                           if require_port_access is None else bool(require_port_access))

    problem = problem_from_metadata(meta)
    rho, transform = canonicalize_density(meta, density)
    base = validate_sample(
        rho,
        target_vf=float(meta["volume_fraction_target"]),
        domain_mask=problem.domain_mask,
        vf_tolerance=vf_tol,
        max_gray_fraction=float(geometry_cfg["max_gray_fraction"]),
        min_feature_size=int(geometry_cfg["min_feature_size_px"]),
    )
    fixed_nodes = (np.concatenate([bc.node_indices for bc in problem.base_problem.bcs])
                   if problem.base_problem.bcs else np.asarray([], dtype=int))
    bc_connected, n_bc_elements = check_bc_connectivity(
        rho, fixed_nodes, problem.base_problem.mesh.nelx, problem.base_problem.mesh.nely)
    path = check_mechanism_path_connectivity(
        rho, fixed_nodes, problem.input_node, problem.output_node,
        problem.base_problem.mesh.nelx, problem.base_problem.mesh.nely)
    hinge = detect_hinges(rho, min_neck_px=int(geometry_cfg["min_feature_size_px"]))

    out: dict[str, Any] = {
        "format": "opencompmech.mechanism-gate.v2",
        "protocol": {
            "name": protocol.get("name"), "format": protocol.get("format"),
            "path": protocol.get("_path"), "sha256": protocol.get("_sha256"),
        },
        "density_transform": transform,
        "geometry": base,
        "bc_connectivity": {"passed": bool(bc_connected),
                              "n_connected_elements": int(n_bc_elements)},
        "mechanism_path": path,
        "point_hinge": {
            "passed": bool(hinge["passed"]),
            "n_bridge_pixels": int(hinge["n_bridge_pixels"]),
            "survives_erosion": bool(hinge["survives_erosion"]),
            "frac_lost_eroded": float(hinge["frac_lost_eroded"]),
        },
        "physical_units": "normalized_only; no yield/fatigue/buckling/contact/manufacturing claim",
    }
    try:
        u, _K, u_out = solve_mechanism_fea(problem, rho)
        response = _quality_metrics(problem, rho, u, u_out)
        selectivity_report = port_selectivity(problem, rho)
        selectivity = float(selectivity_report.get("output", {}).get("selectivity", 0.0))
        exposure = problem_port_exposure(rho, problem)
        motion = motion_class(problem, rho, u=u, u_out=u_out)
        ga_floor = _source_ga_floor(meta, protocol, min_ga)
        family = str(meta.get("family") or "")
        gini_required, gini_cap, gini_cap_source = resolve_gini_cap(
            meta, family, function_cfg["max_gini_for_simp"])
        max_outside = float(protocol["density_transform"].get("max_outside_domain_mean", 1e-6))
        # A mixed 64px boundary cell cannot encode the source-grid void half
        # independently.  Its occupancy is recorded but the exact-domain
        # masked realization is evaluated.  Material in a zero-coverage coarse
        # cell is controllable and remains a hard contract violation.
        pre_mask_domain_ok = (transform["outside_domain_mean_avoidable_before_mask"]
                              <= max_outside)
        access_ok = bool(
            exposure.get("input", {}).get("approach_clear", False)
            and exposure.get("output", {}).get("approach_clear", False)
            and exposure.get("input", {}).get("clearance", 0.0) >= min_exposure
            and exposure.get("output", {}).get("clearance", 0.0) >= min_exposure
        )
        checks = {
            "geometry": bool(base["overall_passed"]),
            "bc_connected": bool(bc_connected),
            "mechanism_path": bool(path["passed"]),
            "no_point_hinge": bool(hinge["passed"]),
            "inside_domain": bool(pre_mask_domain_ok),
            "positive_output": response["u_out_projected"] > 0.0,
            "minimum_output_stroke": response["u_out_projected"] >= float(function_cfg["min_output_stroke"]),
            "positive_input_response": response["u_in_projected"] > 0.0,
            "signed_ga": response["ga_signed"] >= ga_floor,
            "output_alignment": response["output_alignment"] >= float(function_cfg["min_output_alignment"]),
            "input_axis": response["off_axis_input"] <= float(function_cfg["max_input_off_axis"]),
            "selectivity": selectivity >= min_selectivity,
        }
        if gini_required:
            checks["energy_gini"] = response["energy_gini"] <= float(gini_cap)
        functional_passed = bool(all(checks.values()))
        failures = [name for name, passed in checks.items() if not passed]
        if not access_ok:
            failures.append("port_interface")
        out.update({
            "response": response,
            "port_selectivity": selectivity_report,
            "port_exposure": exposure,
            "motion": motion,
            "functional": {
                "checks": checks,
                "thresholds": {
                    "min_signed_ga": ga_floor,
                    "min_output_stroke": float(function_cfg["min_output_stroke"]),
                    "min_output_alignment": float(function_cfg["min_output_alignment"]),
                    "max_input_off_axis": float(function_cfg["max_input_off_axis"]),
                    "max_gini_for_simp": float(gini_cap) if gini_required else None,
                    "gini_cap_source": gini_cap_source,
                    "min_output_selectivity": min_selectivity,
                    "max_avoidable_outside_domain_mean": max_outside,
                },
                "passed": functional_passed,
            },
            "interface": {
                "passed": access_ok,
                "thresholds": {"min_clearance": min_exposure},
                "required_for_passed_alias": require_port_access,
            },
            "functional_passed": functional_passed,
            "interface_passed": bool(functional_passed and access_ok),
            "passed": bool(functional_passed and (access_ok if require_port_access else True)),
            "failure_reasons": failures,
        })
        if include_stress:
            fields, _u2, _uout2 = compute_mechanism_response_fields(problem, rho)
            out["normalized_stress"] = {
                "max_vm": float(np.nanmax(fields.stress_vm)),
                "p99_vm": float(np.nanpercentile(fields.stress_vm, 99)),
                "canonical_density": "same as functional sparse mechanism solve",
            }
    except Exception as exc:  # a failed solve is a failed verification, never an omission
        out["fea_error"] = f"{type(exc).__name__}: {exc}"
        out["functional"] = {"passed": False}
        out["interface"] = {"passed": False, "required_for_passed_alias": require_port_access}
        out["functional_passed"] = False
        out["interface_passed"] = False
        out["passed"] = False
        out["failure_reasons"] = ["fea_error"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True, help="source BVP JSON")
    ap.add_argument("--densities", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL_PATH))
    ap.add_argument("--min-ga", type=float, default=None)
    ap.add_argument("--min-selectivity", type=float, default=None)
    ap.add_argument("--min-exposure", type=float, default=None)
    ap.add_argument("--vf-tol", type=float, default=None)
    access = ap.add_mutually_exclusive_group()
    access.add_argument("--require-port-access", dest="require_port_access", action="store_true")
    access.add_argument("--no-require-port-access", dest="require_port_access", action="store_false")
    ap.set_defaults(require_port_access=None)
    ap.add_argument("--include-stress", action="store_true",
                    help="compute canonical stress diagnostics (slower; not needed for selection)")
    args = ap.parse_args()
    with open(args.meta) as f:
        meta = json.load(f)
    protocol = load_protocol(args.protocol)
    reports = []
    for path in args.densities:
        try:
            result = gate(meta, np.load(path), args.min_ga, args.min_selectivity,
                          args.min_exposure, args.vf_tol, protocol=protocol,
                          require_port_access=args.require_port_access,
                          include_stress=args.include_stress)
        except Exception as exc:  # malformed proposal is an explicit failed record
            result = {"passed": False, "functional_passed": False,
                      "interface_passed": False,
                      "error": f"{type(exc).__name__}: {exc}"}
        result["density"] = path
        reports.append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {path}")
    output = {
        "format": "opencompmech.mechanism-gate-report.v2",
        "meta": args.meta,
        "protocol": {"path": protocol["_path"], "sha256": protocol["_sha256"]},
        "n": len(reports),
        "functional_pass_rate": sum(r["functional_passed"] for r in reports) / len(reports),
        "interface_pass_rate": sum(r["interface_passed"] for r in reports) / len(reports),
        "pass_rate": sum(r["passed"] for r in reports) / len(reports),
        "reports": reports,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)


if __name__ == "__main__":
    main()
