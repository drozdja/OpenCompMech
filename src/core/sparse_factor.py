"""Unified sparse SPD factorization: CHOLMOD if available, else SuperLU.

The mechanism global stiffness (linear elasticity + input/output springs with
fixed boundary conditions) is symmetric positive definite, so a sparse Cholesky
factorization (CHOLMOD via scikit-sparse) is ~2x faster than the SuperLU LU used
by scipy, and — crucially — a single factor can serve BOTH the forward solve
(K u = f) and the adjoint solve (K λ = L) used for the mechanism sensitivity,
eliminating a redundant per-iteration refactorization.

This module exposes one entry point, ``factorize(K)``, returning an object with a
uniform ``.solve(b)`` method. It transparently falls back to scipy SuperLU where
scikit-sparse / libcholmod isn't installed (e.g. the dev laptop), producing
numerically identical results (both are exact sparse direct solves).
"""

try:
    from sksparse.cholmod import cho_factor as _cho_factor
    from sksparse.cholmod import CholmodError as _CholmodError
    HAVE_CHOLMOD = True
except Exception:  # scikit-sparse or libcholmod absent
    HAVE_CHOLMOD = False
    _CholmodError = Exception


class _LUFactor:
    """Adapter giving scipy's splu object the same .solve(b) interface."""
    __slots__ = ("_lu",)

    def __init__(self, lu):
        self._lu = lu

    def solve(self, b):
        return self._lu.solve(b)


def factorize(K):
    """Factorize a symmetric positive-definite sparse matrix ``K``.

    Returns an opaque factor with a ``.solve(b)`` method that may be reused for
    multiple right-hand sides (forward + adjoint). Uses CHOLMOD when available,
    otherwise SuperLU. If CHOLMOD reports the matrix is not positive definite
    (shouldn't happen with the E_min floor + springs, but guard anyway), it
    falls back to SuperLU LU for that call.
    """
    Kc = K.tocsc()
    if HAVE_CHOLMOD:
        try:
            return _cho_factor(Kc)
        except _CholmodError:
            pass  # non-SPD edge case -> robust LU fallback below
    from scipy.sparse.linalg import splu
    return _LUFactor(splu(Kc))
