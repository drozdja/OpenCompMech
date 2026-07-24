"""Benchmark: CHOLMOD (sparse Cholesky, factor-reuse) vs SuperLU/spsolve.

Builds a REAL mechanism stiffness matrix (same assembly path the optimizer uses),
then times the per-iteration solve work three ways:

  current   : splu(K).solve(f)  + spsolve(K, L)        # forward LU + adjoint REFACTOR
  lu_reuse  : splu(K).solve(f)  + (same lu).solve(L)   # forward LU + adjoint REUSE
  cholmod   : cho_factor(K).solve(f) + (same).solve(L) # Cholesky + adjoint REUSE

All three must produce the SAME u and lambda (to round-off) -> zero quality risk.
"""
import sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve

sys.path.insert(0, ".")
from src.generators.mech import generate_inverter_problem
from src.solvers.linear import assemble_stiffness_matrix


def build_K_free(res, penal=3.0, vf=0.20, seed=0):
    prob = generate_inverter_problem(nelx=res, nely=res, volume_fraction=vf, seed=seed)
    mesh = prob.mesh
    # mid-optimization-like density: random-ish but valid range
    rng = np.random.default_rng(seed)
    density = np.clip(rng.uniform(0.2, 0.9, (mesh.nely, mesh.nelx)), 1e-3, 1.0)
    K = assemble_stiffness_matrix(prob.base_problem, density, penal)
    fixed = prob.base_problem.get_fixed_dofs()
    free = np.setdiff1d(np.arange(mesh.n_dofs), fixed)
    K_free = K[np.ix_(free, free)].tocsc()
    f = rng.standard_normal(K_free.shape[0])   # stand-in forward RHS
    L = rng.standard_normal(K_free.shape[0])   # stand-in adjoint RHS
    return K_free, f, L


def time_it(fn, n):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    try:
        from sksparse.cholmod import cho_factor
        have_chol = True
    except Exception as e:
        have_chol = False
        print("scikit-sparse not available:", e)

    for res in (64, 128, 192, 256):
        K, f, L = build_K_free(res)
        n = K.shape[0]
        reps = max(3, int(2_000_000 / n))

        # reference solution
        u_ref = splu(K).solve(f)

        def current():
            lu = splu(K)
            u = lu.solve(f)
            lam = spsolve(K, L)
            return u, lam

        def lu_reuse():
            lu = splu(K)
            u = lu.solve(f)
            lam = lu.solve(L)
            return u, lam

        t_cur = time_it(current, reps)
        t_lur = time_it(lu_reuse, reps)

        line = f"res={res:>3} dofs={n:>7} nnz={K.nnz:>9}  reps={reps:>3}  " \
               f"current={t_cur*1e3:8.1f}ms  lu_reuse={t_lur*1e3:8.1f}ms"

        if have_chol:
            def cholmod():
                fac = cho_factor(K)
                u = fac.solve(f)
                lam = fac.solve(L)
                return u, lam
            # verify identical
            u_c, _ = cholmod()
            err = np.linalg.norm(u_c - u_ref) / (np.linalg.norm(u_ref) + 1e-30)
            t_chol = time_it(cholmod, reps)
            line += f"  cholmod={t_chol*1e3:8.1f}ms  (rel_err={err:.1e}, " \
                    f"speedup vs current={t_cur/t_chol:.2f}x)"
        print(line)


if __name__ == "__main__":
    main()
