#!/usr/bin/env python3
"""Render one reproducible, strictly interface-verified mechanism case.

The demonstrator consumes an already completed ``eval_harness.py`` result.  It
does not search by appearance: unless ``--spec-row`` is supplied, it takes the
first strict pass in the evaluator's predeclared spec order.  The figure uses
the same canonical density transform, sparse spring-aware mechanism BVP, and
protocol as the table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_conditioning import problem_from_metadata  # noqa: E402
from scripts.eval_mechanism_gate import (  # noqa: E402
    _json_default, canonicalize_density, gate, load_protocol,
)
from src.generators.mech import compute_mechanism_response_fields  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return None


def overlay(ax, rho: np.ndarray, meta: dict, title: str) -> None:
    """Plot FEM-coordinates faithfully: node +y is visually upward."""
    res = int(meta["resolution"])
    ax.imshow(1.0 - rho, cmap="gray", vmin=0, vmax=1, origin="lower",
              extent=(0, res, 0, res), interpolation="nearest")

    def xy(node: int):
        return node % (res + 1), node // (res + 1)

    for bc in meta.get("boundary_conditions", []):
        for node in bc.get("nodes", []):
            x, y = xy(int(node))
            ax.plot(x, y, "s", color="#2855c5", ms=4.5, zorder=5)
    mechanism = meta["mechanism"]
    arrow_len = max(8.0, res * 0.12)
    for node, direction, color, label in (
        (mechanism["input_node"], mechanism["input_direction"], "#0a9f30", "input"),
        (mechanism["output_node"], mechanism["output_direction"], "#c000c0", "output"),
    ):
        x, y = xy(int(node))
        d = np.asarray(direction, dtype=float)
        d /= np.linalg.norm(d) + 1e-12
        ax.plot(x, y, "o", color=color, ms=5.5, zorder=6)
        ax.annotate("", xy=(x + d[0] * arrow_len, y + d[1] * arrow_len), xytext=(x, y),
                    arrowprops=dict(color=color, width=1.4, headwidth=5), zorder=6)
        ax.text(x + 2, y + 2, label, color=color, fontsize=7, zorder=7)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, res); ax.set_ylim(0, res); ax.set_aspect("equal"); ax.axis("off")


def draw_response(ax, fields, rho: np.ndarray, meta: dict) -> float:
    res = int(meta["resolution"])
    ux, uy = fields.displacement
    magnitude = np.sqrt(ux ** 2 + uy ** 2)
    scale = 0.15 * res / (float(magnitude.max()) + 1e-12)
    xx, yy = np.meshgrid(np.arange(res + 1), np.arange(res + 1))
    # Cell stress is mapped through the same input-actuated displacement field.
    mesh = ax.pcolormesh(xx + scale * ux, yy + scale * uy, fields.stress_vm,
                         shading="flat", cmap="inferno")
    ax.pcolormesh(xx + scale * ux, yy + scale * uy, rho,
                  shading="flat", cmap="gray", alpha=0.14, vmin=0, vmax=1)
    ax.set_title("Generated design: input-actuated deformation + von Mises stress (normalized units)", fontsize=10)
    ax.set_xlim(-0.15 * res, 1.15 * res); ax.set_ylim(-0.15 * res, 1.15 * res)
    ax.set_aspect("equal"); ax.axis("off")
    cb = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    ax.text(0.02, 0.02, f"display magnification ×{scale:.2g}", transform=ax.transAxes,
            fontsize=7, color="white", bbox=dict(facecolor="black", alpha=0.55, pad=2))
    return float(scale)


def find_candidate(results: dict, eval_dir: Path, method: str, spec_row: int | None):
    specs = results["per_spec"]
    if spec_row is not None:
        specs = [s for s in specs if int(s["row"]) == int(spec_row)]
        if not specs:
            raise SystemExit(f"spec row {spec_row} is not in the evaluation result")
    for spec in specs:
        if not spec.get("reference_accessible", False):
            continue
        trial = spec["methods"].get(method)
        if not trial:
            continue
        selected = trial.get("selected_interface", {})
        index = selected.get("candidate_index")
        if index is None or not selected.get("interface_passed", False):
            continue
        candidate = next((c for c in trial["candidates"]
                          if int(c["candidate_index"]) == int(index)), None)
        if candidate and candidate.get("density_path"):
            path = eval_dir / candidate["density_path"]
            if path.exists():
                return spec, candidate, path
    suffix = f" for spec row {spec_row}" if spec_row is not None else ""
    raise SystemExit(f"no strict interface-passing {method} candidate with a saved density{suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-results", required=True,
                    help="runs/eval/results.json produced with --save-candidates")
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--method", choices=["neural", "retrieval"], default="neural")
    ap.add_argument("--spec-row", type=int, default=None,
                    help="explicit evaluated cache row; provenance labels this user-selected rather than automatic")
    ap.add_argument("--protocol", default=None,
                    help="must match the protocol hash embedded in evaluation results")
    args = ap.parse_args()
    results_path = Path(args.eval_results).resolve()
    eval_dir = results_path.parent
    with results_path.open() as f:
        results = json.load(f)
    protocol_path = args.protocol or results["protocol"]["path"]
    protocol = load_protocol(protocol_path)
    if protocol["_sha256"] != results["protocol"]["sha256"]:
        raise SystemExit("provided protocol does not match the evaluation result")
    spec, candidate, density_path = find_candidate(results, eval_dir, args.method, args.spec_row)
    with open(spec["stem"] + ".json") as f:
        meta = json.load(f)
    generated64 = np.load(density_path)
    verification = gate(meta, generated64, protocol=protocol, require_port_access=True,
                        include_stress=True)
    if not verification["passed"]:
        raise SystemExit("selected evaluation candidate no longer passes the frozen strict protocol")
    rho, transform = canonicalize_density(meta, generated64)
    problem = problem_from_metadata(meta)
    fields, _u, _uout = compute_mechanism_response_fields(problem, rho)
    reference = np.load(spec["stem"] + ".density.npy")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.15, 1]})
    overlay(axes[0], reference, meta, "Held-out specification + reference")
    overlay(axes[1], rho, meta, f"{args.method} candidate (fixed seed {candidate.get('seed')})")
    scale = draw_response(axes[2], fields, rho, meta)
    response = verification["response"]
    selectivity = ((verification.get("port_selectivity", {}) or {}).get("output", {}) or {}).get("selectivity")
    vf = verification["geometry"]["volume_fraction"]
    nearest = spec["methods"][args.method]["selected_interface"]
    text = (
        "FRESH SPARSE-FEA RE-EVALUATION\n\n"
        f"functional protocol: {'PASS' if verification['functional_passed'] else 'FAIL'}\n"
        f"strict interface: {'PASS' if verification['interface_passed'] else 'FAIL'}\n"
        f"u_out (declared direction): {response['u_out_projected']:.3g}\n"
        f"signed GA: {response['ga_signed']:.3g}\n"
        f"output alignment: {response['output_alignment']:.3f}\n"
        f"output selectivity: {selectivity:.3g}\n"
        f"active-domain VF: {vf['actual']:.3f} (target {vf['target']:.3f})\n"
        f"nearest train Dice: {nearest.get('nearest_train_dice', float('nan')):.3f}\n\n"
        "2D normalized linear-elastic analysis only.\n"
        "No material/thickness/yield/fatigue/buckling,\n"
        "contact, process, or flight qualification."
    )
    axes[3].axis("off")
    axes[3].text(0.0, 1.0, text, va="top", ha="left", fontsize=8.5, family="monospace")
    figure.suptitle("OpenCompMech verification-first demonstrator", fontsize=13)
    figure.tight_layout()
    figure.savefig(out / "demonstrator.png", dpi=160)
    plt.close(figure)

    np.save(out / "generated_64.npy", generated64.astype(np.float32))
    np.save(out / "generated_verifier_rho.npy", rho.astype(np.float32))
    with (out / "gate.json").open("w") as f:
        json.dump(verification, f, indent=2, default=_json_default)
    provenance = {
        "format": "opencompmech.demonstrator.v2",
        "evaluation_results": str(results_path), "evaluation_results_sha256": sha256_file(results_path),
        "protocol": {"path": protocol["_path"], "sha256": protocol["_sha256"]},
        "source_stem": spec["stem"], "source_metadata_sha256": sha256_file(Path(spec["stem"] + ".json")),
        "source_cache_row": spec["row"], "method": args.method,
        "candidate_index": candidate["candidate_index"], "candidate_seed": candidate.get("seed"),
        "candidate_density_from_eval": str(density_path), "density_transform": transform,
        "selection_rule": ("first strict-interface pass in predeclared evaluator order"
                           if args.spec_row is None else
                           "user-specified evaluated cache row; not an automatic first-pass demonstrator"),
        "display_deformation_magnification": scale, "git_head": git_head(),
    }
    with (out / "provenance.json").open("w") as f:
        json.dump(provenance, f, indent=2, default=_json_default)
    with (out / "demo.json").open("w") as f:
        json.dump({"source_row": spec["row"], "method": args.method,
                   "functional_passed": verification["functional_passed"],
                   "interface_passed": verification["interface_passed"],
                   "u_out_projected": response["u_out_projected"],
                   "ga_signed": response["ga_signed"],
                   "output_alignment": response["output_alignment"],
                   "output_selectivity": selectivity, "vf_active": vf["actual"]}, f, indent=2)
    print(f"[demo] wrote {out / 'demonstrator.png'} from row {spec['row']} ({args.method})")


if __name__ == "__main__":
    main()
