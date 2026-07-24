"""Family C: Moving Morphable Components — REAL gradient MMC (v1).

Guo et al. (2014): the design is a small set of explicit morphable
components; here each component is a capsule with 5 parameters
(xc, yc, half_len, theta, half_thick), all in NORMALIZED domain units so the
parameterization is resolution-independent. The mechanism objective
(u_out - alpha * input strain energy, identical to the SIMP formulation) is
differentiated w.r.t. the component parameters by the chain rule:

    dJ/dp = sum_e dJ/drho_e * drho_e/dp

dJ/drho_e comes from the EXISTING adjoint (compute_mechanism_sensitivity,
factor reuse); drho_e/dp is analytic through the smooth-union projection:

    d_k(x)   = dist(x, segment_k) - t_k          (capsule signed distance)
    phi_k    = sigmoid(-d_k / h)                 (smoothed indicator)
    rho      = 1 - prod_k (1 - phi_k)            (differentiable union)

The signed-distance gradients collapse to ONE branch-free formula set
(derived 2026-07-18; r = x - closest_point, s = clamped axial projection):

    d dist/d c     = -r_hat
    d dist/d a     = -(s/a) * (r_hat . d_axis)   [nonzero only when clamped]
    d dist/d theta = -s * (r_hat . d_axis_perp)
    d d_k /d t     = -1

Optimization: Adam ascent at 64x64 (parameters are resolution-free), ~200
iters, smoothing h annealed; final design rasterized at the target
resolution with a sharp threshold, de-bridged, FEA-relabeled. Problem specs
are sampled from generate_random_mechanism (invert/redirect archetypes with
randomized axes/ports/springs), so the optimizer faces varied objectives and
DISCOVERS varied component topologies — unlike v0 ('mmc_stage'), which was a
construction reusing rr_lever's cross-pivot and was rightly rejected as a
shortcut (user review 2026-07-17: "non-simp methods generate basically the
same results"). v0 is deleted; git history keeps it.

Every sample stores its final N x 6 component matrix
([xc, yc, half_len, theta, half_thick, active_flag], normalized units) —
the compact generation-friendly representation that is Family C's point.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .mech import (
    MechProblem,
    generate_random_mechanism,
    create_mech_problem_at_resolution,
    solve_mechanism_fea,
    compute_mechanism_sensitivity,
    _solve_mech_state,
)
from .flexure_utils import _debridge


# ---------------------------------------------------------------------------
# Differentiable geometry projection
# ---------------------------------------------------------------------------

def _project(params, res, h, fixed_phi=None):
    """Rasterize component params -> (rho, cache for gradients).

    params: (N,5) [xc,yc,a,theta,t] in NORMALIZED units (domain = 1.0).
    fixed_phi: optional (res*res,) non-designable material (port/pad discs).
    Returns rho (res,res) and a cache dict for _backprop.
    """
    N = params.shape[0]
    xs = (np.arange(res) + 0.5) / res
    X, Y = np.meshgrid(xs, xs)                    # element centers, normalized
    P = np.stack([X.ravel(), Y.ravel()], axis=1)  # (M,2)
    M = P.shape[0]

    phi = np.empty((N, M))
    cache = {"r_hat": np.empty((N, M, 2)), "s": np.empty((N, M)),
             "rd": np.empty((N, M)), "rdp": np.empty((N, M)),
             "d_axis": np.empty((N, 2))}
    for k in range(N):
        xc, yc, a, th, t = params[k]
        d = np.array([np.cos(th), np.sin(th)])
        rel = P - np.array([xc, yc])
        proj = rel @ d
        s = np.clip(proj, -a, a)
        r = rel - s[:, None] * d
        dist = np.linalg.norm(r, axis=1)
        r_hat = r / (dist[:, None] + 1e-12)
        sd = dist - t
        phi[k] = 1.0 / (1.0 + np.exp(np.clip(sd / h, -40, 40)))
        cache["r_hat"][k] = r_hat
        cache["s"][k] = s
        cache["rd"][k] = r_hat @ d
        cache["rdp"][k] = r_hat @ np.array([-d[1], d[0]])
        cache["d_axis"][k] = d

    one_minus = 1.0 - phi                          # (N,M)
    if fixed_phi is not None:
        prod_all = one_minus.prod(axis=0) * (1.0 - fixed_phi)
    else:
        prod_all = one_minus.prod(axis=0)
    rho = 1.0 - prod_all
    cache["phi"] = phi
    cache["one_minus"] = one_minus
    cache["prod_all"] = prod_all
    cache["h"] = h
    cache["params"] = params.copy()
    return rho.reshape(res, res), cache


def _backprop(dJ_drho, cache, res):
    """Chain rule: dJ/drho (res,res) -> dJ/dparams (N,5)."""
    g = dJ_drho.ravel()                            # (M,)
    phi, one_minus = cache["phi"], cache["one_minus"]
    prod_all, h = cache["prod_all"], cache["h"]
    params = cache["params"]
    N = phi.shape[0]
    grad = np.zeros((N, 5))
    for k in range(N):
        # d rho / d phi_k = prod_{j != k}(1-phi_j) (incl. fixed plane)
        drho_dphi = prod_all / np.maximum(one_minus[k], 1e-9)
        dphi_dsd = -phi[k] * (1.0 - phi[k]) / h    # d phi / d signed_dist
        w = g * drho_dphi * dphi_dsd               # (M,) dJ/d signed_dist_k
        r_hat = cache["r_hat"][k]
        s, rd, rdp = cache["s"][k], cache["rd"][k], cache["rdp"][k]
        a = max(params[k, 2], 1e-6)
        # signed-distance gradients (branch-free forms, see module docstring)
        grad[k, 0] = np.dot(w, -r_hat[:, 0])                 # xc
        grad[k, 1] = np.dot(w, -r_hat[:, 1])                 # yc
        grad[k, 2] = np.dot(w, -(s / a) * rd)                # a (caps only:
        #   s/a = ±1 there; interior s/a<1 but rd=0 so the term vanishes)
        grad[k, 3] = np.dot(w, -s * rdp)                     # theta
        grad[k, 4] = np.dot(w, -np.ones_like(s))             # t
    return grad


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def _init_components(problem, res, rng):
    """Connected initial web: anchors -> center -> ports, + random extras."""
    n = res + 1
    def node_xy(node):
        return np.array([node % n, node // n], dtype=float) / res

    pts = [node_xy(problem.input_node), node_xy(problem.output_node)]
    anchors = []
    for bc in problem.base_problem.bcs:
        ns = np.asarray(bc.node_indices)
        anchors.append(node_xy(int(ns[len(ns) // 2])))
    center = np.mean(pts + anchors, axis=0)

    comps = []
    def seg(p, q, t):
        c = 0.5 * (p + q)
        v = q - p
        L = np.linalg.norm(v) + 1e-9
        th = np.arctan2(v[1], v[0])
        comps.append([c[0], c[1], 0.5 * L, th, t])

    for p in pts + anchors:
        seg(np.asarray(p), center + rng.uniform(-0.06, 0.06, 2),
            rng.uniform(0.02, 0.035))
    for _ in range(int(rng.randint(3, 6))):        # free extras
        p = rng.uniform(0.15, 0.85, 2)
        q = np.clip(p + rng.uniform(-0.35, 0.35, 2), 0.05, 0.95)
        seg(p, q, rng.uniform(0.015, 0.03))
    return np.array(comps)


def _fixed_discs(problem, res):
    """Non-designable material at ports and anchor pads (phi plane)."""
    n = res + 1
    xs = (np.arange(res) + 0.5) / res
    X, Y = np.meshgrid(xs, xs)
    phi = np.zeros(res * res)
    spots = [(problem.input_node, 3.0), (problem.output_node, 3.0)]
    for bc in problem.base_problem.bcs:
        ns = np.asarray(bc.node_indices)
        spots.append((int(ns[len(ns) // 2]), 4.0))
    for node, r_px in spots:
        px, py = (node % n) / res, (node // n) / res
        r = r_px / res
        d = np.sqrt((X - px) ** 2 + (Y - py) ** 2).ravel()
        phi = np.maximum(phi, (d < r).astype(float))
    return phi


def optimize_mmc(
    problem_hi: MechProblem,
    rng: np.random.RandomState,
    opt_res: int = 64,
    n_iters: int = 240,
    alpha: float = 0.35,
    vf_cap: float = 0.22,
    vf_weight: float = 400.0,
):
    """Adam ascent on component params for the mechanism objective.

    Returns (params, history) — params in normalized units.
    """
    problem = create_mech_problem_at_resolution(problem_hi, opt_res)
    params = _init_components(problem, opt_res, rng)
    fixed_phi = _fixed_discs(problem, opt_res)

    # per-parameter step scales (normalized units / radians)
    scale = np.array([0.004, 0.004, 0.003, 0.02, 0.0015])
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for it in range(n_iters):
        # anneal smoothing DOWN TO the final raster sharpness — v1 stopped at
        # h=0.015 and the gray tendrils the optimizer relied on vanished at
        # the 0.5 threshold (outputs left floating in void)
        if it < n_iters // 3:
            h = 0.030
        elif it < 2 * n_iters // 3:
            h = 0.015
        else:
            h = 0.008
        rho, cache = _project(params, opt_res, h, fixed_phi)
        rho_c = np.clip(rho, 1e-4, 1.0)
        u, K, u_out, factor, free = _solve_mech_state(problem, rho_c)
        dc = compute_mechanism_sensitivity(problem, rho_c, K, u, alpha=alpha,
                                           factor=factor, free_dofs=free)
        # volume cap penalty (keeps the union from flooding the domain)
        excess = max(0.0, float(rho.mean()) - vf_cap)
        dJ_drho = dc - vf_weight * 2.0 * excess / rho.size
        grad = _backprop(dJ_drho, cache, opt_res)

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        mh = m / (1 - b1 ** (it + 1))
        vh = v / (1 - b2 ** (it + 1))
        params = params + scale * mh / (np.sqrt(vh) + eps)
        # box constraints
        params[:, 0:2] = np.clip(params[:, 0:2], 0.03, 0.97)
        params[:, 2] = np.clip(params[:, 2], 0.02, 0.45)
        # thickness box: floor above sub-pixel (survives the threshold),
        # ceiling low enough that the union can't flood into a blob
        params[:, 4] = np.clip(params[:, 4], 0.015, 0.04)
        hist.append(float(u_out))
    return params, hist


def construct_mmc_opt(
    nelx: int,
    nely: int,
    rng: np.random.RandomState,
    k_in: float = None,
    k_out: float = None,
    max_attempts: int = 3,
) -> Optional[Tuple[np.ndarray, MechProblem, Dict]]:
    """Family C sample: sample a problem spec, run gradient MMC, rasterize
    at full resolution, de-bridge, relabel. k_in/k_out are accepted for
    registry-signature compatibility but the problem sampler owns its springs
    (same contract as the 'random' SIMP type)."""
    del k_in, k_out
    res = nelx
    from .rigid_replace import _debridge       # lazy: avoids circular import
    for attempt in range(max_attempts):
        seed = int(rng.randint(0, 2**31 - 1))
        problem = generate_random_mechanism(nelx=nelx, nely=nely,
                                            volume_fraction=0.25, k_in=None,
                                            k_out=None, seed=seed)
        # MULTI-START: MMC is non-convex in component space (components
        # cannot split/merge, so basins are init-determined). Best-of-2
        # random inits roughly doubles the single-start yield for 2x cost —
        # still SIMP-class runtime.
        best = None
        fixed_hi = _fixed_discs(problem, res)
        for _restart in range(2):
            params, hist = optimize_mmc(problem, rng)
            rho, _ = _project(params, res, h=0.008, fixed_phi=fixed_hi)
            density = (rho > 0.5).astype(np.float64)
            if density.sum() < 40:
                continue
            if not _debridge(density):
                continue
            _, _, u_out = solve_mechanism_fea(problem, density)
            if best is None or abs(float(u_out)) > abs(best[2]):
                best = (params, density, float(u_out), hist)
        if best is None:
            continue
        params, density, u_out, hist = best
        if abs(u_out) < 0.5:                       # hopeless — resample spec
            continue
        # Direction relabel (same policy as every constructed family): if
        # the optimized mechanism moves opposite the sampled spec, the
        # MEASURED sense is the label.
        relabeled = False
        if u_out < 0:
            problem = MechProblem(
                base_problem=problem.base_problem,
                input_node=problem.input_node,
                input_direction=problem.input_direction,
                output_node=problem.output_node,
                output_direction=(-problem.output_direction[0],
                                  -problem.output_direction[1]),
                k_in=problem.k_in, k_out=problem.k_out)
            u_out = -u_out
            relabeled = True

        active = (params[:, 2] > 0.021) & (params[:, 4] > 0.0085)
        comp_matrix = [[float(x) for x in row] + [float(active[i])]
                       for i, row in enumerate(params)]
        meta = {
            'family': 'C',
            'generator': 'mmc_opt',
            'n_components': int(params.shape[0]),
            'n_active_components': int(active.sum()),
            'components': comp_matrix,
            'component_schema': ['xc', 'yc', 'half_len', 'theta',
                                 'half_thick', 'active'],
            'component_units': 'normalized (domain = 1.0)',
            'opt_iters': len(hist),
            'u_out_history_tail': [round(x, 3) for x in hist[-5:]],
            'problem_seed': seed,
            'output_dir_relabeled': relabeled,
            'constructed_vf': float(density.mean()),
        }
        return density, problem, meta

    return None


MMC_CONSTRUCTORS = {
    'mmc_opt': construct_mmc_opt,
}
