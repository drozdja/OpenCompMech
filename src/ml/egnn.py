"""SE(2)-equivariant EGNN denoiser for generative mechanism graphs.

Architecture follows E(n)-Equivariant Graph Neural Networks (Satorras et al.,
arXiv:2102.09844) with the coordinate-output convention of equivariant diffusion
models: the network updates node coordinates through its layers and the
*displacement* it produced is the predicted velocity.  Because that displacement
is built only from ``(x_i - x_j)`` scaled by invariant weights, it rotates with
the input by construction — the equivariance is structural, not learned.

Channels are split by how they must transform:

* **equivariant** — node positions, and the static per-node direction vectors
  carrying the applied load / desired output motion.  Direction vectors enter
  messages only through the invariants ``<v_i, x_ij>``, ``<v_j, x_ij>`` and
  ``<v_i, v_j>``, which is how orientation information is used without breaking
  equivariance.
* **invariant** — radius, existence, strut presence, strut width, the goal
  scalars, and the flow time.

Message passing runs over the complete slot graph (N is small), so the model can
create a strut anywhere; adjacency is a generated channel, not a fixed input.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _mlp(i, h, o, act=nn.SiLU):
    return nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, o))


def timestep_embedding(t, dim):
    """Sinusoidal embedding; matches the raster model's time conditioning."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) *
                      torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


class EGNNLayer(nn.Module):
    """One equivariant message-passing round over a dense slot graph."""

    def __init__(self, h_dim, e_dim, hidden):
        super().__init__()
        self.edge_mlp = _mlp(2 * h_dim + e_dim + 4, hidden, hidden)
        self.coord_mlp = nn.Sequential(_mlp(hidden, hidden, 1), nn.Tanh())
        self.node_mlp = _mlp(h_dim + hidden, hidden, h_dim)
        self.edge_out = _mlp(hidden, hidden, e_dim)
        self.h_norm = nn.LayerNorm(h_dim)
        self.e_norm = nn.LayerNorm(e_dim)

    def forward(self, h, x, vec, e, mask, free):
        # h (B,N,H)  x (B,N,2)  vec (B,N,2)  e (B,N,N,E)  mask (B,N)  free (B,N)
        b, n, _ = h.shape
        dx = x.unsqueeze(2) - x.unsqueeze(1)                    # (B,N,N,2)
        d2 = (dx * dx).sum(-1, keepdim=True)
        vi, vj = vec.unsqueeze(2), vec.unsqueeze(1)
        inv = torch.cat([d2,
                         (vi * dx).sum(-1, keepdim=True),
                         (vj * dx).sum(-1, keepdim=True),
                         (vi * vj).sum(-1, keepdim=True)], dim=-1)
        hi = h.unsqueeze(2).expand(b, n, n, h.shape[-1])
        hj = h.unsqueeze(1).expand(b, n, n, h.shape[-1])
        m = self.edge_mlp(torch.cat([hi, hj, e, inv], dim=-1))

        pair = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(-1)
        eye = torch.eye(n, device=h.device, dtype=h.dtype).view(1, n, n, 1)
        pair = pair * (1.0 - eye)                               # no self-messages
        m = m * pair

        # Equivariant coordinate update, normalised for stability (EGNN eq. 4).
        # d2 is exactly 0 on the diagonal and sqrt has an INFINITE derivative
        # there, so a bare d2.sqrt() produces NaN gradients even though the
        # diagonal is masked out (0 * inf = NaN).  The epsilon is inside the
        # sqrt, which is the only place it fixes the backward pass.
        w = self.coord_mlp(m) * pair
        upd = (dx / ((d2 + 1e-8).sqrt() + 1.0) * w).sum(2) / max(n - 1, 1)
        x = x + upd * free.unsqueeze(-1)        # anchors are given: never moved

        h = h + self.node_mlp(torch.cat([h, m.sum(2)], dim=-1))
        h = self.h_norm(h) * mask.unsqueeze(-1)
        e = e + self.edge_out(m)
        e = self.e_norm(0.5 * (e + e.transpose(1, 2)))          # struts undirected
        return h, x, e


class MechEGNN(nn.Module):
    """Predicts the rectified-flow velocity for a padded mechanism graph.

    Inputs are the noisy node channels ``(pos, radius, existence)`` and edge
    channels ``(adjacency, width)``; outputs are velocities of the same shape.
    """

    def __init__(self, n_anchor=4, node_ch=4, edge_ch=2, scalar_dim=15,
                 hidden=128, layers=6, t_dim=64):
        super().__init__()
        self.n_anchor = n_anchor
        self.node_ch = node_ch
        self.edge_ch = edge_ch
        self.t_dim = t_dim
        # invariant node inputs: radius, existence, is_anchor, present, roles(3)
        node_in = (node_ch - 2) + 5
        self.node_embed = _mlp(node_in, hidden, hidden)
        self.edge_embed = _mlp(edge_ch, hidden, hidden)
        self.cond_embed = _mlp(t_dim + scalar_dim, hidden, hidden)
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden, hidden, hidden) for _ in range(layers)])
        self.node_out = _mlp(hidden, hidden, node_ch - 2)        # radius, existence
        self.edge_out = _mlp(hidden, hidden, edge_ch)

    def forward(self, node_x, edge_x, t, scalars, anchor_pos, anchor_vec,
                anchor_present, anchor_roles):
        """node_x (B,F,4)  edge_x (B,N,N,2)  t (B,)  scalars (B,S)."""
        b, f, _ = node_x.shape
        a = self.n_anchor
        n = a + f
        dev, dt = node_x.device, node_x.dtype

        # ---- assemble the full slot set: anchors (given) then free (denoised)
        pos = torch.cat([anchor_pos, node_x[:, :, 0:2]], dim=1)         # (B,N,2)
        vec = torch.cat([anchor_vec,
                         torch.zeros(b, f, 2, device=dev, dtype=dt)], dim=1)

        inv_free = node_x[:, :, 2:]                                    # rad, exist
        inv_anch = torch.zeros(b, a, self.node_ch - 2, device=dev, dtype=dt)
        inv = torch.cat([inv_anch, inv_free], dim=1)

        is_anchor = torch.cat([torch.ones(b, a, device=dev, dtype=dt),
                               torch.zeros(b, f, device=dev, dtype=dt)], dim=1)
        present = torch.cat([anchor_present,
                             torch.ones(b, f, device=dev, dtype=dt)], dim=1)
        roles = torch.cat([anchor_roles,
                           torch.zeros(b, f, 3, device=dev, dtype=dt)], dim=1)

        h = self.node_embed(torch.cat(
            [inv, is_anchor.unsqueeze(-1), present.unsqueeze(-1), roles], dim=-1))
        cond = self.cond_embed(torch.cat(
            [timestep_embedding(t, self.t_dim).to(dt), scalars], dim=-1))
        h = h + cond.unsqueeze(1)

        e = self.edge_embed(edge_x)
        mask = present
        free = torch.cat([torch.zeros(b, a, device=dev, dtype=dt),
                          torch.ones(b, f, device=dev, dtype=dt)], dim=1)

        pos0 = pos
        for layer in self.layers:
            h, pos, e = layer(h, pos, vec, e, mask, free)

        v_pos = (pos - pos0)[:, a:, :]                 # equivariant by construction
        v_inv = self.node_out(h[:, a:, :])             # invariant channels
        v_node = torch.cat([v_pos, v_inv], dim=-1)
        v_edge = self.edge_out(e)
        v_edge = 0.5 * (v_edge + v_edge.transpose(1, 2))
        return v_node, v_edge
