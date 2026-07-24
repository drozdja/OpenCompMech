"""Compact conditional UNet for the 64x64 pilot diffusion model.

Conditioning:
  * spatial: the COND_DIM raster channels are concatenated to the noisy image.
  * global : timestep + scalar vector -> shared embedding, injected into every
             ResBlock via FiLM (additive scale/shift).
Predicts epsilon. ~15M params at base=64 -- trivial for a 34GB R9700.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        a = t.float()[:, None] * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout * 2)          # FiLM scale+shift
        self.norm2 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        # Scores + softmax in fp32: under bf16 autocast the (q @ k) logits
        # overflow the bf16 range once the token count grows (e.g. the 128px
        # bottleneck), producing NaN. fp32 here is stable and near-free; the
        # value matmul stays in the autocast dtype.
        attn = torch.softmax((q.float() @ k.float()) / math.sqrt(c), dim=-1).to(v.dtype)
        out = (attn @ v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


class ConditionalUNet(nn.Module):
    def __init__(self, in_ch, cond_ch, scalar_dim, base=64,
                 mults=(1, 2, 4), emb_dim=256):
        super().__init__()
        self.time = nn.Sequential(
            SinusoidalEmb(base), nn.Linear(base, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim))
        self.scal = nn.Sequential(
            nn.Linear(scalar_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim))

        self.in_conv = nn.Conv2d(in_ch + cond_ch, base, 3, padding=1)
        chs = [base * m for m in mults]

        # Down (only ResBlock outputs are pushed as skips; see forward())
        self.downs = nn.ModuleList()
        c_prev = base
        for i, c in enumerate(chs):
            self.downs.append(ResBlock(c_prev, c, emb_dim))
            self.downs.append(ResBlock(c, c, emb_dim))
            if i < len(chs) - 1:
                self.downs.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
            c_prev = c

        # Mid
        self.mid1 = ResBlock(c_prev, c_prev, emb_dim)
        self.mid_attn = Attention(c_prev)
        self.mid2 = ResBlock(c_prev, c_prev, emb_dim)

        # Up
        self.ups = nn.ModuleList()
        for i, c in enumerate(reversed(chs)):
            self.ups.append(ResBlock(c_prev + c, c, emb_dim))
            self.ups.append(ResBlock(c + c, c, emb_dim))
            if i < len(chs) - 1:
                self.ups.append(nn.ConvTranspose2d(c, c, 4, stride=2, padding=1))
            c_prev = c

        self.out = nn.Sequential(
            nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1))

    def forward(self, x, t, cond, scalars):
        emb = self.time(t) + self.scal(scalars)
        h = self.in_conv(torch.cat([x, cond], dim=1))
        skips = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, emb); skips.append(h)
            else:  # downsample
                h = layer(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, skips.pop()], dim=1), emb)
            else:  # upsample
                h = layer(h)
        return self.out(h)
