"""Physics-guided sampling: steer a flow model with DiffFEA gradients.

Reusable pieces for the directional-stiffness demo and the guidance-scale sweep.
Guidance is classifier-style on the predicted design x0 (FEA backward only, no
model backprop). An optional connectivity term (from physics.physics_loss) can
be mixed in to keep designs valid as the stiffness push gets stronger.
"""

import numpy as np
import torch

from src.ml.physics import physics_loss


def _coverage_weights_torch(x, domain):
    """Return a model-grid design-envelope coverage field for ``x``.

    ``tensor_spec.build_cond`` block-mean pools a native domain mask.  At
    64px, therefore, a boundary pixel can legitimately hold a fractional
    value: it is the fraction of the corresponding source-resolution cells
    that are in the design envelope.  Treating that value as ``bool`` makes a
    64→128 volume projection use the wrong physical area.
    """
    if domain is None:
        weights = torch.ones_like(x[:, :1])
    else:
        weights = domain.to(dtype=x.dtype, device=x.device).clamp(0, 1)
        if weights.ndim != 4 or weights.shape[1] != 1:
            raise ValueError(f"domain coverage must have shape (N,1,H,W), got {tuple(weights.shape)}")
        if weights.shape[-2:] != x.shape[-2:]:
            raise ValueError("domain coverage spatial shape disagrees with density")
        if weights.shape[0] == 1 and x.shape[0] != 1:
            weights = weights.expand(x.shape[0], -1, -1, -1)
        elif weights.shape[0] != x.shape[0]:
            raise ValueError("domain coverage batch size disagrees with density")
    if torch.any(weights.sum((1, 2, 3)) <= 0):
        raise ValueError("domain coverage contains an empty design envelope")
    return weights


def project_volume(x, target_vf, domain=None, iters=24):
    """Shift a sampled density field to the requested *native-area* VF.

    The old stiffness sweep could claim enormous compliance gains simply by
    adding material.  This monotone bisection projects every intermediate
    guided result back to the declared volume inside the design envelope.
    ``x`` stays in the model's [-1, 1] representation.

    ``domain`` may be binary, but at a coarser model resolution it is normally
    a fractional source-area coverage field.  Weighting the mean by that
    coverage exactly matches nearest-neighbour expansion followed by an exact
    source-domain mask for integer mappings such as 64→128.  Pixels with zero
    coverage are hard-set to void; mixed boundary pixels are retained and are
    subsequently clipped by the full-resolution verifier.
    """
    weights = _coverage_weights_torch(x, domain)
    active = weights > 0
    if target_vf is None:
        return torch.where(active, x.clamp(-1, 1), torch.full_like(x, -1.0))
    lo = torch.full((x.shape[0], 1, 1, 1), -4.0, device=x.device, dtype=x.dtype)
    hi = torch.full_like(lo, 4.0)
    target = torch.as_tensor(target_vf, device=x.device, dtype=x.dtype).reshape(-1, 1, 1, 1)
    if target.shape[0] == 1 and x.shape[0] > 1:
        target = target.expand(x.shape[0], -1, -1, -1)
    for _ in range(iters):
        mid = (lo + hi) * 0.5
        rho = ((x + mid + 1) * 0.5).clamp(0, 1)
        vf = ((rho * weights).sum((1, 2, 3), keepdim=True)
              / weights.sum((1, 2, 3), keepdim=True).clamp_min(1e-12))
        lo = torch.where(vf < target, mid, lo)
        hi = torch.where(vf < target, hi, mid)
    y = (x + (lo + hi) * 0.5).clamp(-1, 1)
    return torch.where(active, y, torch.full_like(y, -1.0))


def project_volume_numpy(rho, target_vf, domain=None, iters=24,
                         return_report=False):
    """Coverage-weighted counterpart of :func:`project_volume` for eval.

    Parameters use a single H×W density in ``[0, 1]``.  This is intentionally
    used for every 64px candidate baseline in the evaluation harness, so the
    neural method is not the only method receiving a volume/envelope
    post-process.  Native source designs are deliberately excluded: they are
    a full-resolution ceiling/check rather than a 64px representation method.
    """
    a = np.asarray(rho, dtype=np.float64)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"expected one HxW density, got shape {a.shape}")
    a = np.clip(a, 0.0, 1.0)
    if domain is None:
        weights = np.ones_like(a, dtype=np.float64)
    else:
        weights = np.asarray(domain, dtype=np.float64)
        while weights.ndim > 2 and weights.shape[0] == 1:
            weights = weights[0]
        if weights.shape != a.shape:
            raise ValueError(f"domain coverage shape {weights.shape} disagrees with density {a.shape}")
        weights = np.clip(weights, 0.0, 1.0)
    active = weights > 0
    denom = float(weights.sum())
    if denom <= 0:
        raise ValueError("domain coverage contains an empty design envelope")

    def weighted_vf(v):
        return float((v * weights).sum() / denom)

    before = weighted_vf(a)
    if target_vf is None:
        result = np.where(active, a, 0.0)
    else:
        target = float(target_vf)
        lo, hi = -4.0, 4.0
        for _ in range(int(iters)):
            mid = 0.5 * (lo + hi)
            candidate = np.clip(a + mid, 0.0, 1.0)
            if weighted_vf(candidate) < target:
                lo = mid
            else:
                hi = mid
        result = np.where(active, np.clip(a + 0.5 * (lo + hi), 0.0, 1.0), 0.0)
    report = {
        "format": "opencompmech.64px-envelope-projection.v1",
        "domain_semantics": "fractional native-area coverage",
        "target_vf": None if target_vf is None else float(target_vf),
        "weighted_vf_before": before,
        "weighted_vf_after": weighted_vf(result),
        "zero_coverage_pixels": int((~active).sum()),
        "fractional_coverage_pixels": int(((weights > 0) & (weights < 1)).sum()),
    }
    result = result.astype(np.float32, copy=False)
    return (result, report) if return_report else result


def remap_node(n, Rf, Rt):
    ix, iy = n % (Rf + 1), n // (Rf + 1)
    return min(round(iy * Rt / Rf), Rt) * (Rt + 1) + min(round(ix * Rt / Rf), Rt)


def fixed_dofs_from_meta(meta, Rf, Rt, device):
    dofs = []
    for bc in meta["boundary_conditions"]:
        nodes, dirs = bc["nodes"], bc["directions"]
        for k, n in enumerate(nodes):
            n2 = remap_node(int(n), Rf, Rt)
            d = dirs[k] if k < len(dirs) else dirs[0]
            if d == 0:
                dofs.append(2 * n2)
            elif d == 1:
                dofs.append(2 * n2 + 1)
            else:
                dofs += [2 * n2, 2 * n2 + 1]
    return torch.as_tensor(sorted(set(dofs)), dtype=torch.long, device=device)


def guided_sample(model, obj, cond, scal, fea, fixed, probe, direction, scale,
                  steps=40, device="cuda", guide_start=1 / 3, conn_weight=0.0,
                  cfg_scale=None, target_vf=None, domain=None):
    """Rectified-flow Euler sampling + directional-stiffness guidance on x0.

    direction: (dx,dy) to make stiff, or None for a plain (unguided) sample.
    scale: guidance step size (units of the normalized gradient) per guided step.
    conn_weight: optional weight on a connectivity/port guidance term that keeps
                 the design attached as stiffness guidance pushes off-manifold.
    cfg_scale: classifier-free guidance weight (requires a cfg-dropout-trained
               model). None -> plain conditional. w=0 -> unconditional (maximally
               free/funky). w=1 -> nominal conditional. 0<w<1 -> a real
               conditioning-STRENGTH dial: loosen the topology prior while the
               model stays on its valid manifold. w>1 -> sharpen conditioning.
    """
    # Sample at the conditioning grid resolution so the sampler follows the
    # cache/checkpoint (64px baseline, 128px experiment) instead of a fixed size.
    res = int(cond.shape[-1])
    x = torch.randn(1, 1, res, res, device=device)
    dt = 1.0 / steps
    # CFG at w=1 is the conditional field exactly.  Do not evaluate the
    # unconditional U-Net only to form u + (c - u): it doubles nominal-sampling
    # cost and introduces avoidable floating-point cancellation.  Non-nominal
    # CFG values deliberately retain the original two-branch computation.
    use_cfg = cfg_scale is not None and float(cfg_scale) != 1.0
    null_cond = torch.zeros_like(cond) if use_cfg else None
    null_scal = torch.zeros_like(scal) if use_cfg else None
    # Sampling precision. Default bf16 (fast, fine at 64px). gfx1201/RDNA4 has
    # broken half-precision conv kernels that NaN at 128px, so set
    # COMP2D_SAMPLE_PRECISION=fp32 there (fp32 => no autocast). See
    # scripts/train_pilot.py --precision for the same rationale on the train side.
    import os
    import contextlib
    _prec = os.environ.get("COMP2D_SAMPLE_PRECISION", "bf16")
    _fp32 = _prec == "fp32"
    _amp_dtype = torch.float16 if _prec == "fp16" else torch.bfloat16
    for i in range(steps):
        t = torch.full((1,), i * dt, device=device)
        _amp = (contextlib.nullcontext() if _fp32
                else torch.autocast(device, dtype=_amp_dtype))
        with torch.no_grad(), _amp:
            if not use_cfg:
                v = model(x, t * 1000.0, cond, scal)
            else:
                v_c = model(x, t * 1000.0, cond, scal)
                v_u = model(x, t * 1000.0, null_cond, null_scal)
                v = v_u + cfg_scale * (v_c - v_u)
        x = x + dt * v
        if direction is not None and i >= int(steps * guide_start):
            x0 = (x + (1 - t.view(-1, 1, 1, 1)) * v).float().detach()
            x0.requires_grad_(True)
            rho = ((x0[0, 0] + 1) * 0.5).clamp(0, 1).to(fea.dtype)
            C = fea.directional_compliance(rho, fixed, probe, direction)
            loss = C
            if conn_weight > 0:
                pl, _ = physics_loss(x0, cond)
                loss = C + conn_weight * pl.to(C.dtype)
            g = torch.autograd.grad(loss, x0)[0]
            x = x - scale * g / (g.norm() + 1e-8)
        # Project at each step: it keeps gradients from accumulating into an
        # unreported volume increase, while still leaving the flow trajectory
        # free to decide *where* to put the material.
        x = project_volume(x, target_vf, domain)
    return project_volume(x, target_vf, domain)


def measure_dir(design, fea, fixed, probe):
    """(C_x, C_y) directional compliances of a design (lower = stiffer)."""
    rho = ((design[0, 0] + 1) * 0.5).clamp(0, 1).to(fea.dtype)
    with torch.no_grad():
        cx = fea.directional_compliance(rho, fixed, probe, (1.0, 0.0)).item()
        cy = fea.directional_compliance(rho, fixed, probe, (0.0, 1.0)).item()
    return cx, cy


def to_gray(design):
    a = (design[0, 0].float().cpu().numpy() + 1) / 2
    v = (255 * (1 - np.clip(a, 0, 1))).astype(np.uint8)
    return np.stack([v, v, v], -1)
