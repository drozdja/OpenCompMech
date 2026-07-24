"""Shared flexure-construction helpers (LEAF module — import nothing from
the generator family modules). Lives separately so fact_planar /
ground_structure / mmc / rigid_replace can all use these without circular
imports (rigid_replace imports the family registries at its bottom).
"""

import numpy as np
from typing import Tuple

from .seeds import _draw_thick_line


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _fit_points_to_domain(
    points: dict,
    keys,
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    min_frac: float = 0.55,
    target_range: Tuple[float, float] = (0.60, 0.85),
    margin: float = 12.0,
):
    """Uniformly upscale + re-place a joint set so its bbox spans a healthy
    fraction of the domain.

    User eyeball 2026-07-16: many rr_four_bar/rr_slider_crank samples were
    tiny linkages lost in an empty domain (footprint down to 0.36 of the
    domain side). Tiny mechanisms waste conditioning-raster resolution and
    make the dataset's scale distribution erratic. Uniform scaling preserves
    all link-length RATIOS, transmission angles, and direction vectors, so
    the sampled kinematics stay valid.

    Geometry already >= min_frac is returned unchanged (scale 1.0) to keep
    the sampler's native placement diversity.
    """
    P = np.array([points[k] for k in keys], dtype=float)
    lo, hi = P.min(axis=0), P.max(axis=0)
    ext = hi - lo
    cur = max(ext[0] / nelx, ext[1] / nely)
    if cur >= min_frac or cur < 1e-9:
        return points, 1.0
    s = rng.uniform(*target_range) / cur
    s = min(s,
            (nelx - 2 * margin) / max(ext[0], 1e-9),
            (nely - 2 * margin) / max(ext[1], 1e-9))
    c = 0.5 * (lo + hi)
    Q = (P - c) * s
    qlo, qhi = Q.min(axis=0), Q.max(axis=0)
    Q[:, 0] += rng.uniform(margin - qlo[0], nelx - margin - qhi[0])
    Q[:, 1] += rng.uniform(margin - qlo[1], nely - margin - qhi[1])
    return {k: Q[i] for i, k in enumerate(keys)}, float(s)


def _debridge(density: np.ndarray, max_iters: int = 5) -> bool:
    """Remove single-pixel articulation points by stamping 3x3 solid at each.

    Returns True when the design has zero bridge pixels (possibly after
    repair); False if repair did not converge in max_iters.
    """
    from src.validation.connectivity import detect_hinges
    nely, nelx = density.shape
    for _ in range(max_iters):
        h = detect_hinges(density)
        if h['n_bridge_pixels'] == 0:
            return True
        ys, xs = np.where(h['bridge_mask'])
        for y, x in zip(ys, xs):
            density[max(0, y-1):min(nely, y+2), max(0, x-1):min(nelx, x+2)] = 1.0
    return bool(detect_hinges(density)['n_bridge_pixels'] == 0)


def _draw_flexure_link(
    density: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    link_hw: int,
    neck_hw: int,
    neck_len: float,
    neck_at_p0: bool = True,
    neck_at_p1: bool = True,
):
    """Draw one link p0->p1 as [neck][thick beam][neck].

    Necks are drawn along the link axis so the flexure bends about the joint
    point. For links too short to fit both necks + a beam, the whole link is
    drawn at neck width (it acts as a long flexure — still valid).
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    u = _unit(p1 - p0)
    L = float(np.linalg.norm(p1 - p0))

    g0 = neck_len if neck_at_p0 else 0.0
    g1 = neck_len if neck_at_p1 else 0.0
    if L <= g0 + g1 + 4.0:                      # too short for a rigid middle
        _draw_thick_line(density, p0, p1, width=neck_hw, value=1.0)
        return

    a = p0 + u * g0                              # beam start
    b = p1 - u * g1                              # beam end
    if neck_at_p0:
        _draw_thick_line(density, p0, a, width=neck_hw, value=1.0)
    if neck_at_p1:
        _draw_thick_line(density, b, p1, width=neck_hw, value=1.0)
    _draw_thick_line(density, a, b, width=link_hw, value=1.0)

