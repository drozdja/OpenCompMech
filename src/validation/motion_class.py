"""Motion-class labeling — WHAT a mechanism does, not HOW it was made.

User insight (2026-07-17): generation family is the wrong diversity axis —
five of nine constructed types turned out to be the same FUNCTION (an
amplifying lever). The dataset's useful diversity axis is the FUNCTION, so
every sample gets a measured motion-class label derived from the FEA
response (never from the generator's intent):

    transfer_class (input vs output motion sense, from the measured output
                    motion direction vs the input direction):
        'inverting'    output moves ~opposite the input (angle > 135 deg)
        'redirecting'  output ~perpendicular (45..135 deg)
        'forwarding'   output ~parallel (< 45 deg)
    magnitude_class (measured directly, not inferred from GA):
        'force_amp'              blocked-force mechanical advantage > 1.05
        'displacement_amp'       |GA| > 1.05 when not force-amplifying
        'displacement_reducer'   |GA| < 0.5 without measured force gain
        'transmitting'           everything between those bands
    motion_class = f"{transfer_class}_{magnitude_class}"  (12 cells)

Plus the raw quantities: transfer_angle_deg, ga_signed, and
mech_advantage (blocked-force ratio F_out/F_in with the output nearly
blocked by a stiff spring — the force-amplification number; one extra
solve reusing nothing, so it is computed here, not inline in production).

All measured, all backfillable from (density + problem spec) via
scripts/backfill_conditioning.py's problem_from_metadata path.
"""

import numpy as np


def motion_class(problem, density, u=None, u_out=None,
                 blocked_k: float = 50.0) -> dict:
    """Classify a mechanism's measured function. Returns a dict of labels.

    u, u_out: optional precomputed working solve (saves one FEA).
    blocked_k: stiff output spring for the blocked-force probe.
    """
    from src.generators.mech import solve_mechanism_fea, MechProblem

    if u is None or u_out is None:
        u, _, u_out = solve_mechanism_fea(problem, density)

    no = problem.output_node
    ni = problem.input_node
    move_out = np.array([u[2 * no], u[2 * no + 1]])
    d_in = np.asarray(problem.input_direction, dtype=float)
    u_in = float(u[2 * ni] * d_in[0] + u[2 * ni + 1] * d_in[1])

    n_out = float(np.linalg.norm(move_out))
    if n_out < 1e-9 or abs(u_in) < 1e-9:
        return {"motion_class": "degenerate", "transfer_class": "degenerate",
                "magnitude_class": "degenerate", "transfer_angle_deg": None,
                "ga_signed": 0.0, "mech_advantage": None}

    # transfer angle: measured output MOTION vs input direction, in the sense
    # the input actually moved (u_in sign)
    cosang = float(np.dot(move_out / n_out, d_in)) * np.sign(u_in)
    ang = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
    if ang > 135.0:
        transfer = "inverting"
    elif ang >= 45.0:
        transfer = "redirecting"
    else:
        transfer = "forwarding"

    ga_signed = float(u_out) / u_in
    ga = abs(ga_signed)

    # blocked-force probe: replace the output spring with a stiff one; the
    # transmitted force is k_blocked * u_out_blocked (input force = 1)
    from src.generators.mech import MechProblem as _MP
    blocked = _MP(base_problem=problem.base_problem,
                  input_node=problem.input_node,
                  input_direction=problem.input_direction,
                  output_node=problem.output_node,
                  output_direction=problem.output_direction,
                  k_in=problem.k_in, k_out=blocked_k,
                  k_perp=problem.k_perp)
    try:
        _, _, u_out_b = solve_mechanism_fea(blocked, density)
        mech_adv = float(blocked_k * u_out_b)      # F_out per unit F_in
    except Exception:  # noqa: BLE001
        mech_adv = None

    # Do not label a displacement reducer a force amplifier on reciprocity
    # grounds.  The output spring and finite compliance mean that implication
    # is not generally true; use the actual blocked-force solve instead.
    if mech_adv is not None and mech_adv > 1.05:
        magnitude = "force_amp"
    elif ga > 1.05:
        magnitude = "displacement_amp"
    elif ga < 0.5:
        magnitude = "displacement_reducer"
    else:
        magnitude = "transmitting"

    return {
        "motion_class": f"{transfer}_{magnitude}",
        "transfer_class": transfer,
        "magnitude_class": magnitude,
        "transfer_angle_deg": round(ang, 1),
        "ga_signed": float(ga_signed),
        "mech_advantage": mech_adv,
    }
