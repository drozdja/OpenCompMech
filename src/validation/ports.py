"""Port exposure metrics — is a port a clear external interface?

User observation (2026-07-16, gs_truss review): ports sometimes land in the
MIDDLE of the structure. Not invalid — the FEA doesn't care — but real
actuators/workpieces need access, so "most of the time we want inputs and
outputs to be clear interfaces, not embedded in the structure."

This module quantifies that so it can be stored per sample (and filtered or
conditioned on later). It is deliberately NOT a validity gate.

Metrics (port = node position on the density grid):
  clearance      fraction of rays cast from the port that reach the domain
                 boundary without crossing material. 1.0 = fully exposed tip,
                 0.0 = fully embedded.
  local_solid    solid fraction in a small disc around the port (embeddedness
                 at the closest range, complements the ray view).
  approach_clear whether the single ray along a given direction is clear —
                 pass -input_direction for inputs (the actuator approaches
                 against its push direction) and +output_direction for
                 outputs (the workpiece sits where the motion delivers work).
"""

import numpy as np


def _ray_clear(solid: np.ndarray, px: float, py: float,
               dx: float, dy: float, max_body: int) -> bool:
    """March a unit-step ray from (px, py); True if it exits the domain
    without hitting material OTHER than the port's own host feature.

    The initial contiguous solid run (the member/disc the port sits on) is
    skipped — but only up to `max_body` steps: a run longer than that means
    the ray travels ALONG the host member, which is just as inaccessible as
    hitting a neighbor. After leaving the initial run, any solid pixel
    blocks. (A fixed skip failed in practice: rr_lever ports sit on ~9 px
    half-width bars, so skip=5 counted every ray as blocked -> exp 0.00.)"""
    nely, nelx = solid.shape
    x, y = float(px), float(py)
    in_body = True
    body_steps = 0
    while True:
        x += dx
        y += dy
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= nelx or iy >= nely:
            return True
        if solid[iy, ix]:
            if not in_body:
                return False
            body_steps += 1
            if body_steps > max_body:
                return False
        else:
            in_body = False


def port_exposure(density: np.ndarray, px: float, py: float,
                  direction=None, n_rays: int = 36,
                  max_body: int = 12) -> dict:
    """Compute exposure metrics for one port. See module docstring.

    max_body: longest initial solid run (px) still counted as the port's own
    host feature — a bit above the thickest member half-width (~10 @128)."""
    solid = density > 0.5
    nely, nelx = solid.shape

    angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    n_clear = sum(_ray_clear(solid, px, py, np.cos(a), np.sin(a), max_body)
                  for a in angles)

    r = 10
    x0, x1 = max(0, int(px - r)), min(nelx, int(px + r) + 1)
    y0, y1 = max(0, int(py - r)), min(nely, int(py + r) + 1)
    local_solid = 0.0
    if x1 > x0 and y1 > y0:
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = (xx + 0.5 - px) ** 2 + (yy + 0.5 - py) ** 2 <= r * r
        if disc.any():
            local_solid = float(solid[y0:y1, x0:x1][disc].mean())

    out = {
        "clearance": n_clear / float(n_rays),
        "local_solid": local_solid,
    }
    if direction is not None:
        d = np.asarray(direction, dtype=float)
        nrm = float(np.linalg.norm(d))
        if nrm > 1e-12:
            d = d / nrm
            out["approach_clear"] = bool(
                _ray_clear(solid, px, py, float(d[0]), float(d[1]), max_body))
        else:
            out["approach_clear"] = None
    return out


def problem_port_exposure(density: np.ndarray, problem) -> dict:
    """Exposure for a MechProblem's input and output ports.

    Node index -> grid position uses the (nelx+1)-node row convention shared
    by the generators and audit tooling.
    """
    nely, nelx = density.shape
    n = nelx + 1
    res = {}
    for tag, node, direction in (
            ("input", problem.input_node,
             (-problem.input_direction[0], -problem.input_direction[1])),
            ("output", problem.output_node,
             tuple(problem.output_direction))):
        px, py = node % n, node // n
        res[tag] = port_exposure(density, px, py, direction=direction)
    return res
