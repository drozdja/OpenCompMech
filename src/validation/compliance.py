"""Port compliance selectivity — the parasitic-compliance signal.

User question (2026-07-17): "many of those mechanisms exhibit large
compliances in parasitic directions, will it be captured during the FEA?"
Answer: the single working-load solve does NOT capture it — it is one column
of the mechanism's behavior. But the reduced stiffness K already contains the
full response; two extra unit-load probes at each port (reusing ONE
factorization) recover the 2x2 port compliance ellipse.

A precision single-DOF mechanism is COMPLIANT along its working direction and
STIFF against off-axis disturbance. Quantities per port:

    selectivity   c_work / c_perp — working compliance over perpendicular
                  compliance. >>1 = crisp single-DOF; ~1 = floppy every way.
    aniso         c_soft / c_stiff — the ellipse's max anisotropy (alignment-
                  agnostic; >= selectivity).
    soft_align    |cos| between the softest principal axis and the intended
                  working direction — is the easy motion the RIGHT motion?

Measured reference (constructed families, 2026-07-17): aniso 2.4-4.2,
soft_align 0.96-0.99 — they move the right way but are only mildly selective
(a precision flexure stage is 10-100x). Reported, NOT gated (the dataset
wants a RANGE of selectivity; it is a conditioning label, not a filter).
"""

import numpy as np


def _port_ellipse(factor, pos, node, work_dir):
    """2x2 compliance at one port from unit-load solves; None if the port DOFs
    are fixed (not in the free set)."""
    dx, dy = 2 * node, 2 * node + 1
    if dx not in pos or dy not in pos:
        return None
    n = len(pos)
    C = np.zeros((2, 2))
    for j, d in enumerate((dx, dy)):
        b = np.zeros(n)
        b[pos[d]] = 1.0
        u = factor.solve(b)
        C[0, j] = u[pos[dx]]
        C[1, j] = u[pos[dy]]
    C = 0.5 * (C + C.T)                       # symmetrize (Maxwell-Betti)
    w, V = np.linalg.eigh(C)                  # ascending eigenvalues
    c_stiff, c_soft = float(w[0]), float(w[1])
    wd = np.asarray(work_dir, dtype=float)
    wd = wd / (np.linalg.norm(wd) + 1e-30)
    perp = np.array([-wd[1], wd[0]])
    c_work = float(wd @ C @ wd)
    c_perp = float(perp @ C @ perp)
    return {
        "selectivity": c_work / max(c_perp, 1e-30),
        "aniso": c_soft / max(c_stiff, 1e-30),
        "soft_align": abs(float(V[:, 1] @ wd)),
        "c_work": c_work,
        "c_perp": c_perp,
    }


def port_selectivity(problem, density, penal: float = 3.0) -> dict:
    """Compliance-ellipse metrics for the input and output ports.

    One reduced-stiffness factorization (the same K_free the forward solve
    builds), reused for four unit-load solves. Returns {'output': {...},
    'input': {...}} with keys per _port_ellipse; a port whose DOFs are fixed
    is omitted.
    """
    from src.generators.mech import _solve_mech_state
    _, _K, _u_out, factor, free = _solve_mech_state(problem, density, penal)
    pos = {int(d): i for i, d in enumerate(free)}
    out = {}
    o = _port_ellipse(factor, pos, problem.output_node,
                      problem.output_direction)
    if o is not None:
        out["output"] = o
    i = _port_ellipse(factor, pos, problem.input_node,
                      problem.input_direction)
    if i is not None:
        out["input"] = i
    return out
