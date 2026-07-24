"""
Solvers module - FEA solvers for structural analysis.

Modules:
    linear      - Direct sparse solver (scipy.sparse.linalg.splu)
    nonlinear   - Newton-Raphson for geometric non-linearity
    modal       - Eigenfrequency analysis
"""

from .linear import (
    FEAResult,
    solve_fea,
    assemble_stiffness_matrix,
    compute_compliance_sensitivity,
    compute_von_mises_stress,
)

from .nonlinear import (
    NonlinearResult,
    newton_raphson,
    solve_nonlinear_with_timeout,
    TimeoutError as NRTimeoutError,
)

__all__ = [
    "FEAResult",
    "solve_fea",
    "assemble_stiffness_matrix",
    "compute_compliance_sensitivity",
    "compute_von_mises_stress",
    "NonlinearResult",
    "newton_raphson",
    "solve_nonlinear_with_timeout",
    "NRTimeoutError",
]

