"""Family B: planar FACT-style flexure stages — valid by construction.

FACT (Freedom and Constraint Topologies, Hopkins): pick the desired freedom,
place flexure constraints in the complementary constraint space. In 2D the
canonical single-DOF cases are:

  * ROTATION about a (possibly remote) center P: every blade's AXIS passes
    through P. The rigid stage then rotates about P — a flexure lever with NO
    physical pivot. Ports at radii r1 (input) and r2 (output) give geometric
    advantage r2/r1 BY CONSTRUCTION — this is the parallel-motion / amplifier
    archetype that SIMP could not reach (floating-translator degeneracy, see
    the family notes in docs/DATASET.md).
  * TRANSLATION along a direction t: all blade axes PARALLEL (perpendicular
    to t) — the parallelogram guide (already used as the slider's prismatic
    joint in rigid_replace.py).

v0 implements the rotation stage ('fact_rotation'). Construction contract is
identical to Family E: thick rigid bodies + thin blade flexures, de-bridge
repair, one FEA labeling solve, gated by the audit's 'construct' rules.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .mech import MechProblem
from .seeds import (
    _create_mech_problem_from_linkage,
    _draw_thick_line,
    _draw_circle,
)
from .flexure_utils import _debridge, _unit


def construct_fact_rotation(
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    k_in: float = None,
    k_out: float = None,
    max_attempts: int = 200,
) -> Optional[Tuple[np.ndarray, MechProblem, Dict]]:
    """Remote-center rotation stage: rigid bar on 2-3 blades converging at P.

    Stage = thick bar along the local circumferential direction (so both ports
    sit on it at different radii from P). Blades run RADIALLY from the stage
    toward/away from P; their shared intersection point P is the virtual pivot.
    Input at radius r1 pushes tangentially; output at radius r2 moves
    tangentially with the same rotation sense: GA_kinematic = r2 / r1
    (sampled in ~[0.5, 2.5] — includes true amplifiers).
    """
    if k_in is None:
        k_in = 0.01
    if k_out is None:
        k_out = float(rng.uniform(0.01, 0.05))

    min_dim = min(nelx, nely)
    for _ in range(max_attempts):
        # Two pivot regimes (user reviews 2026-07-17):
        #   cartwheel — the pivot sits ON the rotor bar BETWEEN the ports;
        #     spoke blades cross at the hub. The seesaw gestalt (ports moving
        #     OPPOSITE ways) is what makes a mechanism read as rotational —
        #     stages with the pivot outside the bar span (both the old
        #     'proximal' and 'remote') look like wobbly translators no matter
        #     how accurately they rotate (rigid-fit COR confirmed rotation;
        #     the eye still refused).
        #   remote — offset-center stage (previous behavior); remote centers
        #     of compliance are a real precision-design archetype.
        cartwheel = rng.random() < 0.5
        if cartwheel:
            # ---- rotor bar THROUGH the hub P, blades crossing at P --------
            P = np.array([rng.uniform(0.35, 0.65) * nelx,
                          rng.uniform(0.35, 0.65) * nely])
            phi = rng.uniform(0, 2 * np.pi)
            u_b = np.array([np.cos(phi), np.sin(phi)])   # rotor axis
            L1 = rng.uniform(0.15, 0.24) * min_dim       # input arm
            L2 = rng.uniform(0.15, 0.24) * min_dim       # output arm
            q1 = P - u_b * L1
            q2 = P + u_b * L2
            r1, r2 = float(L1), float(L2)
            r_stage = 0.0
            # FAT rotor: unlike the remote stage (bar supported by blades
            # along its span), the cartwheel arms are free cantilevers from
            # the hub — v1 with hw 4-5 bent at the input (u_in 42-82 = near
            # free stroke, 17% yield). Same lesson as rr_lever's bar.
            bar_hw = max(5, int(round(min_dim / rng.uniform(15.0, 19.0))))
            blade_hw = 1
            pad_r = bar_hw + 2
            margin = pad_r + 2
            blade_free = rng.uniform(0.14, 0.22) * min_dim
            # 2-3 spoke blades, axes through P, angles kept away from the
            # rotor corridor and from each other (>=40 deg spread)
            n_blades = int(rng.randint(2, 4))
            angles = []
            tries = 0
            while len(angles) < n_blades and tries < 60:
                tries += 1
                b = rng.uniform(np.deg2rad(30), np.deg2rad(150))
                if rng.random() < 0.5:
                    b = -b                               # either side of bar
                if all(abs(np.angle(np.exp(1j * (b - a)))) > np.deg2rad(40)
                       for a in angles):
                    angles.append(b)
            if len(angles) < n_blades:
                continue
            attach, pads = [], []
            ok = True
            for b_ang in angles:
                c, sn = np.cos(b_ang), np.sin(b_ang)
                d = np.array([c * u_b[0] - sn * u_b[1],
                              sn * u_b[0] + c * u_b[1]])
                p = P + d * (bar_hw + blade_free + pad_r)
                if not (margin <= p[0] <= nelx - margin and
                        margin <= p[1] <= nely - margin):
                    ok = False
                    break
                attach.append(P + d * (0.6 * bar_hw))    # rooted in the rotor
                pads.append(p)
            if not ok:
                continue
            if not all(3 <= q[0] <= nelx - 3 and 3 <= q[1] <= nely - 3
                       for q in (q1, q2)):
                continue
            density = np.zeros((nely, nelx), dtype=np.float64)
            _draw_thick_line(density, q1, q2, width=bar_hw, value=1.0)
            for a, b in zip(attach, pads):
                _draw_thick_line(density, a, b, width=blade_hw, value=1.0)
                _draw_circle(density, b, radius=pad_r, value=1.0)
            for q in (q1, q2):
                _draw_circle(density, q, radius=2, value=1.0)
            if not _debridge(density):
                continue
            blade_len = float(blade_free)
        else:
            # ---- remote/offset-center stage (previous behavior) -----------
            P = np.array([rng.uniform(0.15, 0.85) * nelx,
                          rng.uniform(0.15, 0.85) * nely])
            phi = rng.uniform(0, 2 * np.pi)
            u_r = np.array([np.cos(phi), np.sin(phi)])   # radial unit
            u_t = np.array([-u_r[1], u_r[0]])            # tangential unit
            r_stage = rng.uniform(0.28, 0.45) * min_dim
            S = P + u_r * r_stage

            # Rigid stage: thick bar along the tangential direction through
            # S. Sizes bumped 2026-07-16 (footprint 0.43 -> 0.57).
            half_bar = rng.uniform(0.14, 0.22) * min_dim
            bar_hw = max(3, int(round(min_dim / rng.uniform(24.0, 32.0))))
            E1 = S - u_t * half_bar
            E2 = S + u_t * half_bar

            # Ports at different radii => lever ratio; kinked stage widens
            # the GA range.
            kink = rng.uniform(-0.08, 0.14) * min_dim
            E2 = E2 + u_r * kink
            q1, q2 = E1, E2
            r1 = float(np.linalg.norm(q1 - P))
            r2 = float(np.linalg.norm(q2 - P))
            if r1 < 0.12 * min_dim or r2 < 0.12 * min_dim:
                continue

            n_blades = int(rng.randint(2, 4))
            blade_len = rng.uniform(0.20, 0.32) * min_dim
            blade_hw = 1                                 # 3 px blade
            pad_r = bar_hw + 2
            margin = pad_r + 2

            attach_fracs = np.linspace(-0.8, 0.8, n_blades)
            attach_fracs += rng.uniform(-0.08, 0.08, size=n_blades)
            attach, pads = [], []
            ok = True
            for f in attach_fracs:
                a = (S + u_t * (f * half_bar)
                     + u_r * (0.5 * kink * (f + 0.8) / 1.6))
                ray = _unit(a - P)                       # axis through P
                b = a + ray * blade_len                  # outward anchor
                if not (margin <= b[0] <= nelx - margin and
                        margin <= b[1] <= nely - margin):
                    ok = False
                    break
                attach.append(a)
                pads.append(b)
            if not ok:
                continue
            if not all(2 <= q[0] <= nelx - 2 and 2 <= q[1] <= nely - 2
                       for q in (q1, q2)):
                continue

            density = np.zeros((nely, nelx), dtype=np.float64)
            _draw_thick_line(density, E1, E2, width=bar_hw, value=1.0)
            for a, b in zip(attach, pads):
                _draw_thick_line(density, a, b, width=blade_hw, value=1.0)
                _draw_circle(density, b, radius=pad_r, value=1.0)
            for q in (q1, q2):
                _draw_circle(density, q, radius=2, value=1.0)
            if not _debridge(density):
                continue

        # ---- derived port directions (rotation about P) ----------------------
        rot = 1.0 if rng.random() < 0.5 else -1.0
        def tangent(q):
            rvec = _unit(np.asarray(q) - P)
            return (float(-rot * rvec[1]), float(rot * rvec[0]))
        input_direction = tangent(q1)
        output_direction = tangent(q2)                   # same rotation sense

        vf = float(density.mean())
        problem = _create_mech_problem_from_linkage(
            ground_pivots=list(pads),
            input_joint=q1,
            output_joint=q2,
            input_direction=input_direction,
            output_direction=output_direction,
            nelx=nelx, nely=nely,
            volume_fraction=vf,
            k_in=k_in, k_out=k_out,
            bc_radius=max(2, pad_r - 1),
        )

        meta = {
            'family': 'B',
            'generator': 'fact_rotation',
            'pivot_regime': 'cartwheel' if cartwheel else 'remote',
            'virtual_pivot': [float(P[0]), float(P[1])],
            'r_stage': float(r_stage),
            'r_input': r1, 'r_output': r2,
            'ga_kinematic': r2 / r1,
            'n_blades': n_blades,
            'blade_len': float(blade_len),
            'bar_hw': bar_hw,
            'constructed_vf': vf,
        }
        return density, problem, meta

    return None


def construct_fact_translation(
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    k_in: float = None,
    k_out: float = None,
    max_attempts: int = 200,
) -> Optional[Tuple[np.ndarray, MechProblem, Dict]]:
    """Parallelogram translation stage: rigid platform bar guided by 2-3
    PARALLEL blades (axes perpendicular to the motion direction t) running to
    anchor pads. The FACT dual of the rotation stage: the single permitted
    freedom is straight-line translation along t.

    Input pushes one end of the platform along t; output is measured at the
    other end along t — a straight-line guide / transmission archetype
    (GA ~= 1 kinematically, slightly below after blade compliance). Distinct
    motion class from everything else in the dataset: no rotation anywhere.
    """
    if k_in is None:
        k_in = 0.01
    if k_out is None:
        k_out = float(rng.uniform(0.01, 0.05))

    min_dim = min(nelx, nely)
    for _ in range(max_attempts):
        theta = rng.uniform(0, 2 * np.pi)
        t = np.array([np.cos(theta), np.sin(theta)])     # motion direction
        n = np.array([-t[1], t[0]])                      # blade axis direction

        C = np.array([rng.uniform(0.35, 0.65) * nelx,
                      rng.uniform(0.35, 0.65) * nely])
        half_bar = rng.uniform(0.16, 0.26) * min_dim
        bar_hw = max(3, int(round(min_dim / rng.uniform(24.0, 32.0))))
        E1 = C - t * half_bar                            # input end
        E2 = C + t * half_bar                            # output end

        n_blades = int(rng.randint(2, 4))
        blade_len = rng.uniform(0.18, 0.30) * min_dim
        blade_hw = 1
        pad_r = bar_hw + 2
        margin = pad_r + 2

        # all blades on ONE side (true parallelogram); side with more room
        side = 1.0 if rng.random() < 0.5 else -1.0
        attach_fracs = np.linspace(-0.75, 0.75, n_blades)
        attach_fracs += rng.uniform(-0.06, 0.06, size=n_blades)

        placed = False
        for sgn in (side, -side):
            pads_try, attach_try = [], []
            ok = True
            for f in attach_fracs:
                a = C + t * (f * half_bar)
                b = a + sgn * n * blade_len
                if not (margin <= b[0] <= nelx - margin and
                        margin <= b[1] <= nely - margin):
                    ok = False
                    break
                attach_try.append(a)
                pads_try.append(b)
            if ok:
                attach, pads = attach_try, pads_try
                placed = True
                break
        if not placed:
            continue
        if not all(3 <= q[0] <= nelx - 3 and 3 <= q[1] <= nely - 3
                   for q in (E1, E2)):
            continue

        density = np.zeros((nely, nelx), dtype=np.float64)
        _draw_thick_line(density, E1, E2, width=bar_hw, value=1.0)
        for a, b in zip(attach, pads):
            _draw_thick_line(density, a, b, width=blade_hw, value=1.0)
            _draw_circle(density, b, radius=pad_r, value=1.0)
        for q in (E1, E2):
            _draw_circle(density, q, radius=2, value=1.0)
        if not _debridge(density):
            continue

        # both ports move along +t together (rigid platform)
        direction = (float(t[0]), float(t[1]))

        vf = float(density.mean())
        problem = _create_mech_problem_from_linkage(
            ground_pivots=list(pads),
            input_joint=E1,
            output_joint=E2,
            input_direction=direction,
            output_direction=direction,
            nelx=nelx, nely=nely,
            volume_fraction=vf,
            k_in=k_in, k_out=k_out,
            bc_radius=max(2, pad_r - 1),
        )

        meta = {
            'family': 'B',
            'generator': 'fact_translation',
            'motion_direction': [float(t[0]), float(t[1])],
            'ga_kinematic': 1.0,
            'n_blades': n_blades,
            'blade_len': float(blade_len),
            'bar_length': float(2 * half_bar),
            'bar_hw': bar_hw,
            'constructed_vf': vf,
        }
        return density, problem, meta

    return None


FACT_CONSTRUCTORS = {
    'fact_rotation': construct_fact_rotation,
    'fact_translation': construct_fact_translation,
}
