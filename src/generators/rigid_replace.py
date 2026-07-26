"""Rigid-body replacement with compliant flexure joints.

Instead of topology optimization, sample a rigid linkage with
known-good kinematics (reusing the samplers in seeds.py) and substitute every
revolute joint with a short thin flexure neck (pseudo-rigid-body model, Howell):

    rigid link  -> thick beam (bending-stiff)
    pin joint   -> short thin neck (bending-compliant, >=3 px wide so it is a
                   legitimate flexure, NOT a single-pixel point hinge)
    ground pin  -> anchor pad + neck to the link

No optimizer, no degenerate basins: one construction + one FEA labeling solve
per sample (~seconds vs ~minutes for SIMP). Kinematics (I/O directions,
transmission angle, Grashof) are inherited from the linkage sampler, so the
output direction is correct by construction. See docs/DATASET.md for generator
families and type distinctness.

Stiffness contrast is the design rule: bending stiffness ~ w^3, so 7 px links
vs 3 px necks gives ~13x contrast — links act rigid, necks act as pivots.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .mech import MechProblem
from .flexure_utils import (  # noqa: F401 — re-exported for compat
    _unit, _fit_points_to_domain, _debridge, _draw_flexure_link,
)
from .seeds import (
    _generate_four_bar_geometry,
    _generate_slider_crank_geometry,
    _create_mech_problem_from_linkage,
    _draw_thick_line,
    _draw_circle,
)






def construct_four_bar_flexure(
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    k_in: float = None,
    k_out: float = None,
    max_attempts: int = 200,
) -> Optional[Tuple[np.ndarray, MechProblem, Dict]]:
    """Construct a flexure four-bar (crank A-B, coupler B-C, rocker C-D).

    Ground link A-D is NOT drawn (it is the fixed frame: anchor pads at A, D).
    Input force at crank tip B along the sampled crank tangent; output measured
    at rocker tip C along the derived coupler-transmitted direction — both come
    from the validated kinematics in seeds._generate_four_bar_geometry.

    Returns (density, MechProblem, meta) or None if no valid geometry found.
    """
    if k_in is None:
        k_in = 0.01                              # soft actuator (recipe default)
    if k_out is None:
        k_out = float(rng.uniform(0.01, 0.05))

    for _ in range(max_attempts):
        geo = _generate_four_bar_geometry(nelx, nely, rng)
        if geo is None:
            continue

        # --- flexure sizing (resolution-scaled, jittered for diversity) ------
        # Links must be bending-STIFF relative to necks (w^3 contrast): 7-9 px
        # links vs 3 px necks ~ 13-27x. Thinner links flex along their whole
        # length -> mushy GA (measured v1: hw=2 gave GA 0.01-0.2).
        link_hw = max(3, int(round(nelx / rng.uniform(26.0, 36.0))))  # ~9-11px @128
        neck_hw = 1                                # 3 px: flexure, not a hinge
        neck_len = float(nelx) / rng.uniform(16.0, 24.0)              # ~5-8px @128
        pad_r = link_hw + 2

        # upscale tiny linkages to a healthy domain footprint (ratios and
        # directions are scale-invariant, so the sampled kinematics hold)
        J, fit_s = _fit_points_to_domain(
            geo['joints'], ('A', 'B', 'C', 'D'), nelx, nely, rng,
            margin=pad_r + 4.0)
        geo['ground_pivots'] = [J['A'], J['D']]
        geo['input_joint'] = J['B']
        geo['output_joint'] = J['C']
        density = np.zeros((nely, nelx), dtype=np.float64)

        # anchor pads at ground pivots (the "frame")
        for gp in geo['ground_pivots']:
            _draw_circle(density, gp, radius=pad_r, value=1.0)

        # links with necks at every joint (A and D necks = flexure-to-ground)
        for j1, j2 in geo['link_pairs']:
            _draw_flexure_link(density, J[j1], J[j2],
                               link_hw=link_hw, neck_hw=neck_hw,
                               neck_len=neck_len)

        # small discs at moving joints so meeting necks are 4-connected and the
        # port nodes at B / C sit on material
        for name in ('B', 'C'):
            _draw_circle(density, J[name], radius=neck_hw + 1, value=1.0)

        # De-bridge repair: where necks meet pads/discs at shallow angles the
        # union can pinch to single-pixel 4-connectivity contacts (articulation
        # points). Thicken 3x3 at each DETECTED bridge pixel until none remain.
        # This is targeted repair at measured defects — NOT the outlawed greedy
        # volume regrowth (which created hinges at threshold boundaries). A
        # global filter+re-threshold was tried instead and made things worse
        # (blurred necks into links, eroded tips, re-stamped discs re-pinched).
        if not _debridge(density):
            continue                             # rare: resample geometry

        vf = float(density.mean())
        problem = _create_mech_problem_from_linkage(
            ground_pivots=geo['ground_pivots'],
            input_joint=geo['input_joint'],
            output_joint=geo['output_joint'],
            input_direction=tuple(geo['input_dir']),
            output_direction=tuple(geo['output_dir']),
            nelx=nelx, nely=nely,
            volume_fraction=vf,                  # constructed VF is the target
            k_in=k_in, k_out=k_out,
            bc_radius=max(2, pad_r - 1),
        )

        meta = {
            'family': 'E',
            'generator': 'rr_four_bar',
            'fit_scale': fit_s,
            'link_lengths': {k: float(v * fit_s)
                             for k, v in geo['link_lengths'].items()},
            'transmission_angle': float(geo['transmission_angle']),
            'mechanical_advantage': float(geo['mechanical_advantage']),
            'link_hw': link_hw, 'neck_hw': neck_hw,
            'neck_len': float(neck_len),
            'constructed_vf': vf,
        }
        return density, problem, meta

    return None


def construct_slider_crank_flexure(
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    k_in: float = None,
    k_out: float = None,
    max_attempts: int = 200,
) -> Optional[Tuple[np.ndarray, MechProblem, Dict]]:
    """Construct a flexure slider-crank: crank A-B, rod B-C, and the prismatic
    joint at C replaced by a PARALLELOGRAM FLEXURE GUIDE — two parallel thin
    blades, perpendicular to the slide direction, from a small rigid carriage
    at C to anchor pads. The blades bend in unison so the carriage translates
    along slide_dir (the flexure equivalent of a slider in its guide).

    Ground: pad at crank pivot A + the two blade anchor pads.
    Input at crank tip B (tangent dir), output at carriage C along slide_dir.
    """
    if k_in is None:
        k_in = 0.01
    if k_out is None:
        k_out = float(rng.uniform(0.01, 0.05))

    for _ in range(max_attempts):
        geo = _generate_slider_crank_geometry(nelx, nely, rng)
        if geo is None:
            continue

        slide = _unit(np.asarray(geo['slide_dir'], dtype=float))
        normal = np.array([-slide[1], slide[0]])

        link_hw = max(3, int(round(nelx / rng.uniform(26.0, 36.0))))
        neck_hw = 1
        neck_len = float(nelx) / rng.uniform(16.0, 24.0)
        pad_r = link_hw + 2

        # upscale tiny linkages (same rationale as the four-bar); slide_dir
        # and port directions are scale-invariant
        J, fit_s = _fit_points_to_domain(
            geo['joints'], ('A', 'B', 'C'), nelx, nely, rng,
            margin=pad_r + 4.0)

        # carriage + parallelogram guide geometry
        C = np.asarray(J['C'], dtype=float)
        half_s = rng.uniform(7.0, 11.0) * nelx / 128.0   # half blade spacing
        blade_len = rng.uniform(18.0, 28.0) * nelx / 128.0
        g1 = C - slide * half_s
        g2 = C + slide * half_s
        # pick the guide side (±normal) that keeps blade anchors in-domain
        margin = pad_r + 2
        placed = False
        for sgn in rng.permutation([1.0, -1.0]):
            p1 = g1 + sgn * normal * blade_len
            p2 = g2 + sgn * normal * blade_len
            if all(margin <= q[0] <= nelx - margin and
                   margin <= q[1] <= nely - margin for q in (p1, p2)):
                placed = True
                break
        if not placed:
            continue

        density = np.zeros((nely, nelx), dtype=np.float64)

        # crank pivot pad + crank and rod with flexure necks
        _draw_circle(density, J['A'], radius=pad_r, value=1.0)
        _draw_flexure_link(density, J['A'], J['B'], link_hw, neck_hw, neck_len)
        _draw_flexure_link(density, J['B'], J['C'], link_hw, neck_hw, neck_len)

        # rigid carriage bar through C along the slide axis
        _draw_thick_line(density, g1, g2, width=max(2, link_hw - 1), value=1.0)
        # parallelogram blades: thin along their WHOLE length (distributed
        # flexures — this is what makes the carriage translate, not rotate)
        _draw_thick_line(density, g1, p1, width=neck_hw, value=1.0)
        _draw_thick_line(density, g2, p2, width=neck_hw, value=1.0)
        # blade anchor pads
        _draw_circle(density, p1, radius=pad_r, value=1.0)
        _draw_circle(density, p2, radius=pad_r, value=1.0)
        # joint discs so port nodes sit on material
        for name in ('B', 'C'):
            _draw_circle(density, J[name], radius=neck_hw + 1, value=1.0)

        if not _debridge(density):
            continue

        vf = float(density.mean())

        def _mk(out_dir):
            return _create_mech_problem_from_linkage(
                ground_pivots=[J['A'], p1, p2],
                input_joint=J['B'],
                output_joint=J['C'],
                input_direction=tuple(geo['input_dir']),
                output_direction=tuple(out_dir),
                nelx=nelx, nely=nely,
                volume_fraction=vf,
                k_in=k_in, k_out=k_out,
                bc_radius=max(2, pad_r - 1),
            )

        problem = _mk(geo['output_dir'])
        # Derive the output SIGN from the constructed mechanism's true response:
        # seeds.py's slider output_dir formula gets the sense wrong for some
        # crank/elbow configurations (v1 audit: |u_out| up to 17 with the wrong
        # sign). Under pure construction the real kinematics decide — solve
        # once and label the direction the carriage actually moves.
        from .mech import solve_mechanism_fea
        _, _, u_out_probe = solve_mechanism_fea(problem, density)
        if u_out_probe < 0:
            problem = _mk((-geo['output_dir'][0], -geo['output_dir'][1]))

        meta = {
            'family': 'E',
            'generator': 'rr_slider_crank',
            'fit_scale': fit_s,
            'crank_len': float(geo['crank_len'] * fit_s),
            'rod_len': float(geo['rod_len'] * fit_s),
            'blade_len': float(blade_len),
            'blade_spacing': float(2 * half_s),
            'link_hw': link_hw, 'neck_hw': neck_hw,
            'neck_len': float(neck_len),
            'constructed_vf': vf,
        }
        return density, problem, meta

    return None


def construct_lever_flexure(
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    k_in: float = None,
    k_out: float = None,
    max_attempts: int = 200,
) -> Optional[Tuple[np.ndarray, MechProblem, Dict]]:
    """Construct a flexure lever on a CROSS-BLADE PIVOT (cross-spring pivot):
    two thin blades whose axes intersect at the fulcrum F — translationally
    stiff in every direction, rotationally compliant about F. This attacks the
    GA>1 amplifier archetype directly: fact_rotation's long radial blades sag
    elastically (measured GA capped ~0.93 vs kinematic 2.5); a compact crossed
    pivot wastes far less input stroke on parasitic translation.

    Class-1 lever (fulcrum between ports, tangential directions OPPOSITE) or
    class-2/3 (ports on the same arm, directions same sense). Kinematic GA is
    r_out / r_in, sampled mostly > 1.
    """
    if k_in is None:
        k_in = 0.01
    # Softer output spring than the four-bar default: an amplifier must push
    # the output spring at the LONG arm, so a stiff k_out eats the lever ratio.
    if k_out is None:
        k_out = float(rng.uniform(0.005, 0.02))

    min_dim = min(nelx, nely)
    for _ in range(max_attempts):
        F = np.array([rng.uniform(0.30, 0.70) * nelx,
                      rng.uniform(0.30, 0.70) * nely])
        theta = rng.uniform(0, 2 * np.pi)
        u_l = np.array([np.cos(theta), np.sin(theta)])   # lever axis

        r_short = rng.uniform(0.10, 0.17) * min_dim
        ratio = rng.uniform(1.4, 3.5)
        r_long = min(ratio * r_short, 0.42 * min_dim)
        if r_long / r_short < 1.2:
            continue

        # 70% amplifier (input on short arm), 30% reducer — keep both senses
        # in the dataset so the conditional model sees GA above AND below 1.
        if rng.random() < 0.7:
            r_in, r_out = r_short, r_long
        else:
            r_in, r_out = r_long, r_short

        # class 1: ports on opposite sides of F; class 2/3: same side.
        # Pivot translation moves BOTH ports the same way, so class-1
        # (opposite port senses) pays the sag penalty twice — measured GA>1
        # only ever occurs in class-2 (smoke test: 1.2-1.3 vs 0.3-0.5).
        # Amplifiers therefore go class-2; class-1 is kept as the
        # direction-reversing (rocker/inverter-like) variant.
        amplifier = r_in < r_out
        if amplifier:
            lever_class = 2
        else:
            lever_class = 1 if rng.random() < 0.5 else 2
        if lever_class == 1:
            q_in = F - u_l * r_in
            q_out = F + u_l * r_out
            bar_a, bar_b = q_in, q_out
        else:
            q_in = F + u_l * r_in
            q_out = F + u_l * r_out
            # extend the bar slightly past F so the pivot blades attach ON it
            bar_a = F - u_l * (0.05 * min_dim)
            bar_b = F + u_l * max(r_in, r_out)

        # cross-blade pivot: two blades from the bar edge to ground pads, axes
        # through F, fanned about the lever normal so they cross at a healthy
        # angle.
        # GA sag budget (v0/v1 smoke tests): v0's thin bar bent as a
        # cantilever and bypassed the rotation (GA 0.1-0.4); v1's fat bar +
        # big pads swallowed the blades (free length ~3 px -> pivot rigid,
        # u_in ~2). The blade FREE length between bar edge and pad edge is the
        # controlling parameter and must be guaranteed explicitly.
        side = 1.0 if rng.random() < 0.5 else -1.0
        u_n = side * np.array([-u_l[1], u_l[0]])         # normal, chosen side
        half_ang = np.deg2rad(rng.uniform(35.0, 55.0))   # ~90deg crossing
        blade_free = rng.uniform(0.12, 0.20) * min_dim   # free flexure length
        bar_hw = max(4, int(round(min_dim / rng.uniform(16.0, 22.0))))
        blade_hw = 2                                     # 5 px: axially stiff
        pad_r = bar_hw + 2
        margin = pad_r + 2

        pads, blade_starts = [], []
        ok = True
        for s in (+1.0, -1.0):
            c, sn = np.cos(s * half_ang), np.sin(s * half_ang)
            d = np.array([c * u_n[0] - sn * u_n[1],
                          sn * u_n[0] + c * u_n[1]])
            p = F + d * (bar_hw + blade_free + pad_r)    # pad center
            if not (margin <= p[0] <= nelx - margin and
                    margin <= p[1] <= nely - margin):
                ok = False
                break
            pads.append(p)
            blade_starts.append(F + d * (0.6 * bar_hw))  # rooted inside bar
        if not ok:
            continue
        if not all(3 <= q[0] <= nelx - 3 and 3 <= q[1] <= nely - 3
                   for q in (q_in, q_out, bar_a, bar_b)):
            continue

        density = np.zeros((nely, nelx), dtype=np.float64)
        _draw_thick_line(density, bar_a, bar_b, width=bar_hw, value=1.0)
        for bs, p in zip(blade_starts, pads):
            _draw_thick_line(density, bs, p, width=blade_hw, value=1.0)
            _draw_circle(density, p, radius=pad_r, value=1.0)
        for q in (q_in, q_out):
            _draw_circle(density, q, radius=3, value=1.0)
        if not _debridge(density):
            continue

        # tangential port directions for one rotation sense about F
        rot = 1.0 if rng.random() < 0.5 else -1.0
        def _tangent(q):
            r = _unit(np.asarray(q) - F)
            return (float(-rot * r[1]), float(rot * r[0]))
        input_direction = _tangent(q_in)
        output_direction = _tangent(q_out)

        vf = float(density.mean())

        def _mk(out_dir):
            return _create_mech_problem_from_linkage(
                ground_pivots=list(pads),
                input_joint=q_in,
                output_joint=q_out,
                input_direction=input_direction,
                output_direction=tuple(out_dir),
                nelx=nelx, nely=nely,
                volume_fraction=vf,
                k_in=k_in, k_out=k_out,
                bc_radius=max(2, pad_r - 1),
            )

        problem = _mk(output_direction)
        # FEA sign probe (same rationale as the slider-crank): when elastic
        # sag competes with the rotation the measured output sense can flip
        # vs the ideal tangent — label the direction the port actually moves.
        from .mech import solve_mechanism_fea
        _, _, u_out_probe = solve_mechanism_fea(problem, density)
        if u_out_probe < 0:
            output_direction = (-output_direction[0], -output_direction[1])
            problem = _mk(output_direction)

        meta = {
            'family': 'E',
            'generator': 'rr_lever',
            'fulcrum': [float(F[0]), float(F[1])],
            'r_input': float(r_in), 'r_output': float(r_out),
            'ga_kinematic': float(r_out / r_in),
            'lever_class': lever_class,
            'pivot_half_angle_deg': float(np.rad2deg(half_ang)),
            'blade_free_len': float(blade_free),
            'bar_hw': bar_hw,
            'constructed_vf': vf,
        }
        return density, problem, meta

    return None


def construct_compound_lever_flexure(
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    k_in: float = None,
    k_out: float = None,
    max_attempts: int = 200,
) -> Optional[Tuple[np.ndarray, MechProblem, Dict]]:
    """Two class-2 cross-pivot levers in SERIES, joined by a flexure coupler:
    kinematic GA = ratio1 * ratio2 (sampled ~2.2-4.8).

    MEASURED OUTCOME (2026-07-17, keep expectations honest): GA 0.7-1.0,
    NOT >1 — rigid-fit diagnosis showed common-mode pivot translation
    (~3.5 px along the output axis) entering u_in and u_out equally, which
    pins GA near 1 regardless of the lever ratios; stiffening knobs moved
    u_in but never the ratio. Kept as a DIVERSITY type (distinct two-stage
    linkage motif, ~100% gate yield, 0 hinges) — the amplifier role belongs
    to the single rr_lever (GA_med 1.08). See docs/DATASET.md on the amplifier
    ceiling note.

    Layout (local frame, u = lever axis, v = coupler direction):
      lever 1: F1 at origin, input at r_in1*u, tip T1 at r_out1*u
      coupler: T1 -> P2 = T1 + Lc*v   (flexure link: axially stiff)
      lever 2: F2 = P2 - r_in2*u, output at F2 + r_out2*u
    Both levers class-2 (same-side ports) so every port motion is parallel
    to v — non-inverting amplifier. Pivot blades fan away from the coupler
    (lever 1: -v side, lever 2: +v side).
    """
    if k_in is None:
        k_in = 0.01
    # Cascading reflects k_out back to stage 1 as k_out*(ratio1*ratio2)^2 —
    # ~10x a single lever's burden — so the output spring must be SOFT or
    # stage-1 bar bending out-competes the rotation mode (v2 diagnosis:
    # deformed field showed bar-1 bending at the input port, u_out stuck at
    # ~0.9*u_in while u_in matched the ideal-lever prediction).
    if k_out is None:
        k_out = float(rng.uniform(0.002, 0.006))

    min_dim = min(nelx, nely)
    for _ in range(max_attempts):
        theta = rng.uniform(0, 2 * np.pi)
        u_l = np.array([np.cos(theta), np.sin(theta)])
        sigma = 1.0 if rng.random() < 0.5 else -1.0
        v = sigma * np.array([-u_l[1], u_l[0]])

        r_in1 = rng.uniform(0.10, 0.15) * min_dim
        ratio1 = rng.uniform(1.5, 2.2)
        r_out1 = min(ratio1 * r_in1, 0.33 * min_dim)
        r_in2 = rng.uniform(0.10, 0.14) * min_dim
        ratio2 = rng.uniform(1.5, 2.2)
        r_out2 = min(ratio2 * r_in2, 0.33 * min_dim)
        ratio1, ratio2 = r_out1 / r_in1, r_out2 / r_in2
        if ratio1 < 1.3 or ratio2 < 1.3:
            continue
        coupler_len = rng.uniform(0.32, 0.42) * min_dim

        F1 = np.array([rng.uniform(0.15, 0.85) * nelx,
                       rng.uniform(0.15, 0.85) * nely])
        q_in = F1 + u_l * r_in1
        T1 = F1 + u_l * r_out1
        P2 = T1 + v * coupler_len
        F2 = P2 - u_l * r_in2
        q_out = F2 + u_l * r_out2

        # bars must be FAT: stage-1 bending competes with rotation once the
        # cascaded load reflects back (bending stiffness ~ t^3)
        bar_hw = max(5, int(round(min_dim / rng.uniform(13.0, 17.0))))
        blade_free = rng.uniform(0.12, 0.18) * min_dim
        blade_hw = 2
        half_ang = np.deg2rad(rng.uniform(35.0, 55.0))
        pad_r = bar_hw + 2
        margin = pad_r + 2
        neck_len = float(min_dim) / rng.uniform(18.0, 24.0)
        # the coupler's free flexure length is what separates the two lever
        # bars — if they close up (v0: gap could shrink to ~2 px, port discs
        # bridged it), the stages weld into one rigid body (measured u_in ~3,
        # GA<=0.91 despite kinematic 2.4-4.3)
        if coupler_len - 2 * bar_hw - 6 < 2 * neck_len + 4:
            continue

        # pivot blades: fan BEHIND each fulcrum (-u side). v1 fanned them
        # about +/-v, which put one tilted blade + grounded pad right beside
        # the moving arm / output corridor — a grounded brake (GA<=0.9, all
        # seeds). The -u sector is structurally guaranteed clear: arms,
        # ports, and coupler all live on the +u / +v sides.
        pivots = []                                     # (start, pad) pairs
        ok = True
        for F in (F1, F2):
            base = -u_l
            for s in (+1.0, -1.0):
                c, sn = np.cos(s * half_ang), np.sin(s * half_ang)
                d = np.array([c * base[0] - sn * base[1],
                              sn * base[0] + c * base[1]])
                p = F + d * (bar_hw + blade_free + pad_r)
                if not (margin <= p[0] <= nelx - margin and
                        margin <= p[1] <= nely - margin):
                    ok = False
                    break
                pivots.append((F + d * (0.6 * bar_hw), p))
            if not ok:
                break
        if not ok:
            continue
        if not all(3 <= q[0] <= nelx - 3 and 3 <= q[1] <= nely - 3
                   for q in (q_in, q_out, T1, P2, F1, F2)):
            continue

        density = np.zeros((nely, nelx), dtype=np.float64)
        # lever bars (extend slightly past the fulcrum for blade rooting)
        _draw_thick_line(density, F1 - u_l * (0.05 * min_dim), T1,
                         width=bar_hw, value=1.0)
        _draw_thick_line(density, F2 - u_l * (0.05 * min_dim), q_out,
                         width=bar_hw, value=1.0)
        # coupler: flexure link (necks at both ends, axially stiff middle)
        _draw_flexure_link(density, T1, P2,
                           link_hw=max(2, bar_hw - 2), neck_hw=1,
                           neck_len=neck_len)
        for bs, p in pivots:
            _draw_thick_line(density, bs, p, width=blade_hw, value=1.0)
            _draw_circle(density, p, radius=pad_r, value=1.0)
        for q in (q_in, q_out):
            _draw_circle(density, q, radius=3, value=1.0)
        if not _debridge(density):
            continue

        direction = (float(v[0]), float(v[1]))
        vf = float(density.mean())
        pads = [p for _, p in pivots]

        def _mk(out_dir):
            return _create_mech_problem_from_linkage(
                ground_pivots=pads,
                input_joint=q_in,
                output_joint=q_out,
                input_direction=direction,
                output_direction=tuple(out_dir),
                nelx=nelx, nely=nely,
                volume_fraction=vf,
                k_in=k_in, k_out=k_out,
                bc_radius=max(2, pad_r - 1),
            )

        problem = _mk(direction)
        from .mech import solve_mechanism_fea
        _, _, u_out_probe = solve_mechanism_fea(problem, density)
        output_direction = direction
        if u_out_probe < 0:
            output_direction = (-direction[0], -direction[1])
            problem = _mk(output_direction)

        meta = {
            'family': 'E',
            'generator': 'rr_compound_lever',
            'ratio1': float(ratio1), 'ratio2': float(ratio2),
            'ga_kinematic': float(ratio1 * ratio2),
            'coupler_len': float(coupler_len),
            'bar_hw': bar_hw,
            'blade_free_len': float(blade_free),
            'constructed_vf': vf,
        }
        return density, problem, meta

    return None


def construct_bridge_amp_flexure(
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    k_in: float = None,
    k_out: float = None,
    max_attempts: int = 200,
) -> Optional[Tuple[np.ndarray, MechProblem, Dict]]:
    """Guided-knee (half-bridge) displacement amplifier: two shallow arms
    L-T-R with flexure necks; anchor at L; the driven corner R rides a
    PARALLELOGRAM GUIDE so it can only translate along the drive axis.
    Pushing R toward L straightens/buckles the knee: the apex T moves out
    along the output axis with kinematic ratio ~cot(arm angle)/2.

    Two failed predecessors, both diagnosed by measurement (2026-07-17):
      * serial compound lever — common-mode pivot translation enters u_in
        and u_out equally and pins GA near 1 (rigid-fit: both stages
        translated ~3.5 px);
      * full rhombus bridge — the drive direction is nearly collinear with
        the shallow arms, so input force went into axial arm compression and
        the driven corner fought its own rocker (u_in ~3, GA<=0.69).
    The guide fixes both: it reacts the transverse rocker force at R and
    leaves exactly one soft path — arm rotation about the corner necks.
    (This is also the archetype SIMP could never reach, docs/DATASET.md.)

    MEASURED OUTCOME (2026-07-17): GA 0.5-0.85, med 0.62 — better-behaved
    than the rhombus but still <1: knee axial-compliance loss scales as
    cot^2(phi), so shallower angles LOSE more than they promise; 5 px necks
    and softer k_out did not move it. Kept as a DIVERSITY type (knee/toggle
    motif). Amplifier ceiling at 128 px stands at rr_lever's ~1.4; deeper
    amplification needs finer resolution or true parameter optimization
    (Family C). See docs/DATASET.md on the amplifier ceiling.
    """
    if k_in is None:
        k_in = 0.01
    if k_out is None:
        k_out = float(rng.uniform(0.001, 0.004))

    min_dim = min(nelx, nely)
    for _ in range(max_attempts):
        theta = rng.uniform(0, 2 * np.pi)
        ex = np.array([np.cos(theta), np.sin(theta)])     # drive axis
        ey = np.array([-ex[1], ex[0]])                    # output axis
        if rng.random() < 0.5:
            ey = -ey                                      # knee side varies

        span = rng.uniform(0.45, 0.62) * min_dim          # L..R distance
        phi = np.deg2rad(rng.uniform(9.0, 17.0))          # arm angle
        apex_h = 0.5 * span * np.tan(phi)

        C = np.array([rng.uniform(0.40, 0.60) * nelx,
                      rng.uniform(0.40, 0.60) * nely])
        L = C - ex * (0.5 * span)
        R = C + ex * (0.5 * span)
        T = C + ey * apex_h                               # knee apex

        bar_hw = max(3, int(round(min_dim / rng.uniform(22.0, 30.0))))
        neck_len = float(min_dim) / rng.uniform(18.0, 26.0)
        pad_r = bar_hw + 3
        margin = pad_r + 2

        # parallelogram guide at R: carriage bar along the drive axis, two
        # blades perpendicular to it (same recipe as the slider carriage),
        # anchored on the side OPPOSITE the knee
        half_s = rng.uniform(7.0, 11.0) * min_dim / 128.0
        blade_len = rng.uniform(18.0, 26.0) * min_dim / 128.0
        g1 = R - ex * half_s
        g2 = R + ex * half_s
        p1 = g1 - ey * blade_len
        p2 = g2 - ey * blade_len

        pts = (L, R, T, p1, p2)
        if not all(margin <= p[0] <= nelx - margin and
                   margin <= p[1] <= nely - margin for p in pts):
            continue

        density = np.zeros((nely, nelx), dtype=np.float64)
        # knee arms — necks 5 px, NOT 3: the knee pays axial-compliance loss
        # ~cot^2(phi), so 3 px necks ate 25-35% of the apex motion; 5 px
        # necks are 1.7x axially stiffer while arm rotation stays soft
        _draw_flexure_link(density, L, T, link_hw=bar_hw, neck_hw=2,
                           neck_len=neck_len)
        _draw_flexure_link(density, T, R, link_hw=bar_hw, neck_hw=2,
                           neck_len=neck_len)
        # anchor at L
        _draw_circle(density, L, radius=pad_r, value=1.0)
        # guide: carriage + blades + pads
        _draw_thick_line(density, g1, g2, width=max(2, bar_hw - 1), value=1.0)
        _draw_thick_line(density, g1, p1, width=1, value=1.0)
        _draw_thick_line(density, g2, p2, width=1, value=1.0)
        _draw_circle(density, p1, radius=pad_r, value=1.0)
        _draw_circle(density, p2, radius=pad_r, value=1.0)
        for q in (R, T):
            _draw_circle(density, q, radius=3, value=1.0)
        if not _debridge(density):
            continue

        input_direction = (float(-ex[0]), float(-ex[1]))  # squeeze inward
        output_direction = (float(ey[0]), float(ey[1]))
        anchors = [L, p1, p2]

        vf = float(density.mean())

        def _mk(out_dir):
            return _create_mech_problem_from_linkage(
                ground_pivots=list(anchors),
                input_joint=R,
                output_joint=T,
                input_direction=input_direction,
                output_direction=tuple(out_dir),
                nelx=nelx, nely=nely,
                volume_fraction=vf,
                k_in=k_in, k_out=k_out,
                bc_radius=max(2, pad_r - 1),
            )

        problem = _mk(output_direction)
        from .mech import solve_mechanism_fea
        _, _, u_out_probe = solve_mechanism_fea(problem, density)
        if u_out_probe < 0:
            output_direction = (-output_direction[0], -output_direction[1])
            problem = _mk(output_direction)

        meta = {
            'family': 'E',
            'generator': 'rr_bridge_amp',
            'arm_angle_deg': float(np.rad2deg(phi)),
            'ga_kinematic': float(0.5 / np.tan(phi)),
            'span': float(span),
            'bar_hw': bar_hw,
            'constructed_vf': vf,
        }
        return density, problem, meta

    return None


# Registry for the audit harness / future production dispatcher.
RR_CONSTRUCTORS = {
    'rr_four_bar': construct_four_bar_flexure,
    'rr_slider_crank': construct_slider_crank_flexure,
    'rr_lever': construct_lever_flexure,
    'rr_compound_lever': construct_compound_lever_flexure,
    'rr_bridge_amp': construct_bridge_amp_flexure,
}

# Families B (planar FACT) and D (ground structure) share the
# constructed-sample plumbing (audit kind='construct', production dispatch).
# Registered here so every consumer sees one registry. Imports at bottom avoid
# a circular import (both modules import helpers from this one).
from .fact_planar import FACT_CONSTRUCTORS as _FACT
RR_CONSTRUCTORS.update(_FACT)
from .ground_structure import GS_CONSTRUCTORS as _GS
RR_CONSTRUCTORS.update(_GS)
from .mmc import MMC_CONSTRUCTORS as _MMC
RR_CONSTRUCTORS.update(_MMC)
