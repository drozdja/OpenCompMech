"""Rectified Flow (flow matching) objective, drop-in alternative to DDPM.

Same UNet backbone; only the training target and sampler differ, so M(DDPM) vs
M(Flow) is a clean objective ablation. Straight OT path:
    x_t = (1-t) * noise + t * x0,   t ~ U(0,1)
    target velocity  v = x0 - noise
    x0_pred = x_t + (1-t) * v_pred        (for the physics-guided aux loss)
Sampling integrates dx/dt = v(x,t) from t=0 (noise) to t=1 (design), Euler.

The UNet's sinusoidal time embedding takes any float; we pass t*1000 so the
embedding magnitudes match the DDPM range (keeps the backbone identical).
"""

import torch
import torch.nn.functional as F


class RectifiedFlow:
    def __init__(self, device="cuda", **_):
        self.device = device

    def loss_and_x0(self, model, x0, cond, scal):
        b = x0.shape[0]
        t = torch.rand(b, device=x0.device)
        tt = t[:, None, None, None]
        noise = torch.randn_like(x0)
        xt = (1 - tt) * noise + tt * x0
        v_target = x0 - noise
        v_pred = model(xt, t * 1000.0, cond, scal)
        base = F.mse_loss(v_pred, v_target)
        x0_pred = (xt + (1 - tt) * v_pred).clamp(-1, 1)
        return base, x0_pred

    @torch.no_grad()
    def sample(self, model, cond, scal, steps=50, shape=None):
        b = cond.shape[0]
        if shape is None:
            shape = (b, 1, cond.shape[-2], cond.shape[-1])
        x = torch.randn(shape, device=self.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((b,), i * dt, device=self.device)
            v = model(x, t * 1000.0, cond, scal)
            x = x + dt * v
        return x.clamp(-1, 1)
