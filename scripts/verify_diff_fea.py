#!/usr/bin/env python3
"""Numerically verify DiffFEA against sparse NumPy FEA and finite differences.

The result is emitted as one machine-readable JSON document to stdout and,
optionally, to ``--out``.  A non-passing numerical check (or an unexpected
runtime error) returns a non-zero exit status, making this suitable for the
evidence bundle and unattended verification runs.

ROCm PyTorch exposes its accelerator through the ``cuda`` device name.  Use
``--device auto`` (the default) for a portable CPU/ROCm invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generators.stiff import generate_canonical_cantilever  # noqa: E402
from src.ml.diff_fea import DiffFEA, bcs_to_fixed_dofs  # noqa: E402
from src.solvers.linear import solve_fea  # noqa: E402


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def emit(payload: dict, out: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=json_default)
    print(text)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


def run_check(args: argparse.Namespace) -> dict:
    """Run the deterministic compliance and gradient parity checks."""
    res = int(args.res)
    if res < 6:
        raise ValueError("--res must be at least 6 for the fixed gradient probes")
    device = resolve_device(args.device)
    penal, emin = 3.0, 1e-9
    started = time.perf_counter()

    prob = generate_canonical_cantilever(res, res, volume_fraction=0.4)
    rng = np.random.default_rng(0)
    rho = np.clip(0.3 + 0.5 * rng.random((res, res)), 0.05, 1.0)

    c_np = float(solve_fea(prob, rho, penal).compliance)
    fea = DiffFEA(res, nu=prob.material.nu, E0=prob.material.E, Emin=emin,
                  penal=penal, device=device)
    fixed = bcs_to_fixed_dofs(prob.bcs, device=device)
    load = fea.load_vector([(int(item.node_index), float(item.fx), float(item.fy))
                            for item in prob.loads])
    rho_t = torch.tensor(rho, dtype=torch.float64, device=device, requires_grad=True)
    c_t, _ = fea.compliance(rho_t, fixed, load)
    c_t_value = float(c_t.item())
    compliance_rel = abs(c_t_value - c_np) / (abs(c_np) + 1e-12)

    c_t.backward()
    gradient = rho_t.grad.detach().cpu().numpy()
    epsilon = float(args.fd_epsilon)
    probes = [(res // 2, res // 2), (res // 3, res // 4), (2, 2),
              (res - 3, res - 3)]
    gradient_checks = []
    for row, col in probes:
        rho_plus = rho.copy()
        rho_minus = rho.copy()
        rho_plus[row, col] += epsilon
        rho_minus[row, col] -= epsilon
        with torch.no_grad():
            c_plus, _ = fea.compliance(
                torch.tensor(rho_plus, dtype=torch.float64, device=device), fixed, load)
            c_minus, _ = fea.compliance(
                torch.tensor(rho_minus, dtype=torch.float64, device=device), fixed, load)
        finite_difference = (float(c_plus.item()) - float(c_minus.item())) / (2.0 * epsilon)
        autograd = float(gradient[row, col])
        relative_error = abs(finite_difference - autograd) / max(abs(finite_difference), 1e-2)
        gradient_checks.append({
            "cell": [int(row), int(col)],
            "autograd": autograd,
            "finite_difference": finite_difference,
            "relative_error": relative_error,
        })
    gradient_max_rel = max(check["relative_error"] for check in gradient_checks)
    passed = (compliance_rel <= float(args.max_compliance_rel)
              and gradient_max_rel <= float(args.max_gradient_rel))
    accelerator = None
    if device.startswith("cuda") and torch.cuda.is_available():
        accelerator = torch.cuda.get_device_name(torch.device(device))
    return {
        "format": "opencompmech.diff-fea-verification.v1",
        "passed": bool(passed),
        "device": {"requested": args.device, "resolved": device,
                   "accelerator": accelerator, "torch_version": torch.__version__},
        "configuration": {"resolution": res, "dtype": "float64", "penal": penal,
                          "Emin": emin, "finite_difference_epsilon": epsilon,
                          "max_compliance_relative_error": float(args.max_compliance_rel),
                          "max_gradient_relative_error": float(args.max_gradient_rel)},
        "compliance": {"numpy_sparse": c_np, "torch_diff_fea": c_t_value,
                       "relative_error": compliance_rel},
        "gradient_checks": gradient_checks,
        "gradient_max_relative_error": gradient_max_rel,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    ap.add_argument("--out", default=None, help="optional JSON evidence output path")
    ap.add_argument("--fd-epsilon", type=float, default=1e-6)
    ap.add_argument("--max-compliance-rel", type=float, default=1e-4)
    ap.add_argument("--max-gradient-rel", type=float, default=1e-3)
    args = ap.parse_args()
    if args.fd_epsilon <= 0:
        ap.error("--fd-epsilon must be positive")
    if args.max_compliance_rel < 0 or args.max_gradient_rel < 0:
        ap.error("error thresholds must be non-negative")
    try:
        report = run_check(args)
    except Exception as exc:  # preserve a parseable evidence record on runtime failure
        report = {
            "format": "opencompmech.diff-fea-verification.v1",
            "passed": False,
            "device": {"requested": args.device, "resolved": resolve_device(args.device)},
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    emit(report, args.out)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
