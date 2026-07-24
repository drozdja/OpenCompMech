"""Rectified-flow objective and sampler for padded mechanism graphs.

Deliberately the *same* objective as the raster model (``src.ml.flow``): straight
optimal-transport path, velocity target, Euler integration, classifier-free
guidance.  Holding the objective, the data, the split and the FEA gate fixed
means a CNN-on-rasters vs GNN-on-graphs comparison isolates representation and
architecture instead of confounding them with the training recipe.

The discrete channels (node existence, strut presence) are carried as relaxed
+/-1 variables and thresholded at decode time.  A discrete flow-matching
treatment (DeFoG, arXiv:2410.04263) is the natural upgrade and is left as an
explicit next step rather than silently assumed to be better.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_loss(model, node_x0, edge_x0, scalars, anchor_pos, anchor_vec,
              anchor_present, anchor_roles, node_w=1.0, edge_w=1.0):
    """Rectified-flow MSE on the joint (node, edge) state."""
    b = node_x0.shape[0]
    t = torch.rand(b, device=node_x0.device)
    tn = t[:, None, None]
    te = t[:, None, None, None]

    n_noise = torch.randn_like(node_x0)
    e_noise = torch.randn_like(edge_x0)
    e_noise = 0.5 * (e_noise + e_noise.transpose(1, 2))     # struts undirected
    node_t = (1 - tn) * n_noise + tn * node_x0
    edge_t = (1 - te) * e_noise + te * edge_x0

    v_node, v_edge = model(node_t, edge_t, t * 1000.0, scalars,
                           anchor_pos, anchor_vec, anchor_present, anchor_roles)
    loss = (node_w * F.mse_loss(v_node, node_x0 - n_noise)
            + edge_w * F.mse_loss(v_edge, edge_x0 - e_noise))
    return loss


@torch.no_grad()
def sample(model, scalars, anchor_pos, anchor_vec, anchor_present, anchor_roles,
           n_free, n_max, node_ch, edge_ch, steps=50, cfg=1.0, generator=None):
    """Integrate dx/dt = v from noise (t=0) to a design (t=1).

    ``cfg`` > 1 applies classifier-free guidance against the null conditioning
    the model was trained on (zeroed scalars and absent anchors).
    """
    dev = scalars.device
    b = scalars.shape[0]
    node = torch.randn(b, n_free, node_ch, device=dev, generator=generator)
    edge = torch.randn(b, n_max, n_max, edge_ch, device=dev, generator=generator)
    edge = 0.5 * (edge + edge.transpose(1, 2))

    null_scal = torch.zeros_like(scalars)
    null_pres = torch.zeros_like(anchor_present)
    null_pos = torch.zeros_like(anchor_pos)
    null_vec = torch.zeros_like(anchor_vec)

    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((b,), i * dt, device=dev)
        vn, ve = model(node, edge, t * 1000.0, scalars,
                       anchor_pos, anchor_vec, anchor_present, anchor_roles)
        if cfg != 1.0:
            un, ue = model(node, edge, t * 1000.0, null_scal,
                           null_pos, null_vec, null_pres, anchor_roles)
            vn = un + cfg * (vn - un)
            ve = ue + cfg * (ve - ue)
        node = node + dt * vn
        edge = edge + dt * ve
        edge = 0.5 * (edge + edge.transpose(1, 2))
    return node.clamp(-1, 1), edge.clamp(-1, 1)
