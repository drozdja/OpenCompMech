"""Gaussian diffusion (DDPM) with a cosine schedule, epsilon prediction.

Minimal but correct: q_sample for training, p_losses (MSE on epsilon), and a
DDIM sampler for fast eyeball grids during training. Target lives in [-1, 1].
"""

import torch
import torch.nn.functional as F


def cosine_betas(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    ac = f / f[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return betas.clamp(1e-8, 0.999).float()


class Diffusion:
    def __init__(self, T=1000, device="cuda"):
        self.T = T
        self.device = device
        betas = cosine_betas(T).to(device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.acp = torch.cumprod(self.alphas, dim=0)          # alpha_bar
        self.sqrt_acp = self.acp.sqrt()
        self.sqrt_om_acp = (1 - self.acp).sqrt()

    def q_sample(self, x0, t, noise):
        return (self.sqrt_acp[t][:, None, None, None] * x0
                + self.sqrt_om_acp[t][:, None, None, None] * noise)

    def p_losses(self, model, x0, cond, scalars):
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t, cond, scalars)
        return F.mse_loss(pred, noise)

    def loss_and_x0(self, model, x0, cond, scalars):
        """Common objective interface (mirrors flow.RectifiedFlow): returns
        (epsilon-MSE, predicted-x0 in [-1,1]) so the trainer can add a
        physics-guided aux loss on x0 regardless of objective."""
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps = model(xt, t, cond, scalars)
        base = F.mse_loss(eps, noise)
        ac = self.acp[t][:, None, None, None]
        x0_pred = ((xt - (1 - ac).sqrt() * eps) / ac.sqrt()).clamp(-1, 1)
        return base, x0_pred

    def sample(self, model, cond, scalars, steps=50, shape=None):
        return self.ddim_sample(model, cond, scalars, steps=steps, shape=shape)

    @torch.no_grad()
    def ddim_sample(self, model, cond, scalars, steps=50, eta=0.0, shape=None):
        """Deterministic (eta=0) DDIM. Returns x0 estimate in [-1, 1]."""
        b = cond.shape[0]
        if shape is None:
            shape = (b, 1, cond.shape[-2], cond.shape[-1])
        x = torch.randn(shape, device=self.device)
        ts = torch.linspace(self.T - 1, 0, steps, device=self.device).long()
        for i in range(steps):
            t = ts[i].expand(b)
            ac = self.acp[t][:, None, None, None]
            eps = model(x, t, cond, scalars)
            x0 = (x - (1 - ac).sqrt() * eps) / ac.sqrt()
            x0 = x0.clamp(-1, 1)
            if i == steps - 1:
                x = x0
                break
            t_next = ts[i + 1].expand(b)
            ac_next = self.acp[t_next][:, None, None, None]
            sigma = eta * ((1 - ac / ac_next) * (1 - ac_next) / (1 - ac)).sqrt()
            dir_xt = (1 - ac_next - sigma ** 2).clamp(min=0).sqrt() * eps
            x = ac_next.sqrt() * x0 + dir_xt + sigma * torch.randn_like(x)
        return x
