"""Physics-guided auxiliary loss + validity metrics for the mechanism pilot.

The generate->verify->polish loop's verifier rejects designs for two dominant
reasons: (1) material not attached at the load/output ports, (2) material not
CONNECTED to the supports/ports (floating blobs, broken load path). Both are
differentiable to approximate, so we can guide the generator toward validity
WITHOUT running FEA in the training loop.

physics_loss() operates on the predicted design x0 in [-1,1]; it reads the port
blobs and fixed-support mask straight from the conditioning stack (channels are
fixed by tensor_spec: 1=in_blob, 4=out_blob, 7=fixed_mask).

For evaluation, validity_metrics() computes HARD (thresholded) analogues:
connected-component fraction, port-material hit, and floating-material fraction.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .tensor_spec import COND_CHANNELS

# conditioning channel indices (must match src/ml/tensor_spec.COND_CHANNELS)
CH_IN_BLOB = COND_CHANNELS.index("in_blob")
CH_OUT_BLOB = COND_CHANNELS.index("out_blob")
CH_FIXED = COND_CHANNELS.index("fixed_mask")
CH_DOMAIN = COND_CHANNELS.index("domain_mask")


def _dens01(x0):
    """[-1,1] design -> [0,1] density."""
    return ((x0 + 1) * 0.5).clamp(0, 1)


def soft_reachability(dens, seeds, iters=24):
    """Differentiable flood-fill: how much each pixel is 'connected through
    material' to a seed. reach stays 1 at seeds; a material pixel adjacent to a
    reachable pixel inherits reach~=dens; void (dens~0) blocks propagation."""
    reach = seeds.clamp(0, 1)
    for _ in range(iters):
        grown = F.max_pool2d(reach, kernel_size=3, stride=1, padding=1)
        reach = torch.maximum(seeds, torch.minimum(dens, grown))
    return reach


def physics_loss(x0, cond, iters=24, sample_mask=None):
    """Return (total, terms) for genuinely conditioned examples only.

    CFG null-token rows have deliberately zero port/support rasters.  Applying
    a port/connectivity objective to them silently teaches the unconditional
    branch to produce empty material.  ``sample_mask`` excludes those rows
    while retaining a vectorized loss for the rest of the batch.
    """
    if sample_mask is not None:
        keep = sample_mask.to(dtype=torch.bool, device=x0.device)
        if not bool(keep.any()):
            zero = x0.sum() * 0.0
            return zero, {"port": 0.0, "float": 0.0, "n": 0}
        x0, cond = x0[keep], cond[keep]
    dens = _dens01(x0)
    in_blob = cond[:, CH_IN_BLOB:CH_IN_BLOB + 1]
    out_blob = cond[:, CH_OUT_BLOB:CH_OUT_BLOB + 1]
    fixed = cond[:, CH_FIXED:CH_FIXED + 1]

    # (1) ports must carry material
    port_loss = (in_blob * (1 - dens)).mean() + (out_blob * (1 - dens)).mean()

    # (2) material must be reachable from ports/supports
    seeds = (in_blob + out_blob + fixed).clamp(0, 1)
    reach = soft_reachability(dens, seeds, iters)
    float_loss = F.relu(dens - reach).mean()

    total = port_loss + float_loss
    return total, {"port": float(port_loss.detach()),
                   "float": float(float_loss.detach()), "n": int(x0.shape[0])}


# ---------------------------------------------------------------- eval (hard)
def _label_cc(mask):
    """Connected components (4-conn). Uses scipy if present, else a numpy BFS."""
    try:
        from scipy.ndimage import label
        lab, n = label(mask)
        return lab, n
    except Exception:
        lab = np.zeros(mask.shape, np.int32)
        cur = 0
        H, W = mask.shape
        for i in range(H):
            for j in range(W):
                if mask[i, j] and lab[i, j] == 0:
                    cur += 1
                    stack = [(i, j)]
                    lab[i, j] = cur
                    while stack:
                        y, x = stack.pop()
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] \
                                    and lab[ny, nx] == 0:
                                lab[ny, nx] = cur
                                stack.append((ny, nx))
        return lab, cur


def validity_metrics(x0_np, cond_np, thr=0.5):
    """Hard validity proxies for one generated design.
    x0_np:(1,H,W) in [-1,1]; cond_np:(C,H,W). Returns dict of scalars."""
    dens = (x0_np[0] + 1) * 0.5
    # The 64px domain channel is source-area coverage after mean pooling, not
    # a binary mask.  A mixed boundary cell is usable (and exact-masked later),
    # while its contribution to material fractions must be coverage-weighted.
    coverage = (np.clip(cond_np[CH_DOMAIN], 0, 1)
                if cond_np.shape[0] > CH_DOMAIN else np.ones_like(dens, float))
    domain = coverage > 0
    mask = (dens > thr) & domain
    total = int(mask.sum())
    weighted_total = float((mask * coverage).sum())
    if total == 0:
        return {"port_in": 0.0, "port_out": 0.0, "connected_frac": 0.0,
                "floating_frac": 1.0, "n_components": 0, "material_frac": 0.0}

    lab, n = _label_cc(mask)
    # component that best covers the seed pixels (ports + supports)
    seeds = (cond_np[CH_IN_BLOB] + cond_np[CH_OUT_BLOB] + cond_np[CH_FIXED]) > 0.3
    seed_labels = lab[seeds & mask]
    if seed_labels.size:
        main = np.bincount(seed_labels).argmax()
        main_size = int((lab == main).sum())
        main_weight = float(((lab == main) * coverage).sum())
    else:  # nothing seeded on material -> take largest CC
        sizes = np.bincount(lab.ravel())[1:]
        main_size = int(sizes.max()) if sizes.size else 0
        if main_size:
            main = int(np.argmax(sizes)) + 1
            main_weight = float(((lab == main) * coverage).sum())
        else:
            main_weight = 0.0

    def port_hit(ch):
        w = cond_np[ch]
        pk = np.unravel_index(w.argmax(), w.shape)
        r0, r1 = max(0, pk[0] - 1), pk[0] + 2
        c0, c1 = max(0, pk[1] - 1), pk[1] + 2
        return float(mask[r0:r1, c0:c1].any())

    def seed_label(ch):
        w = cond_np[ch]
        pk = np.unravel_index(w.argmax(), w.shape)
        r0, r1 = max(0, pk[0] - 1), pk[0] + 2
        c0, c1 = max(0, pk[1] - 1), pk[1] + 2
        vals = lab[r0:r1, c0:c1]
        vals = vals[vals > 0]
        return int(np.bincount(vals).argmax()) if vals.size else 0

    in_label, out_label = seed_label(CH_IN_BLOB), seed_label(CH_OUT_BLOB)
    support_labels = lab[(cond_np[CH_FIXED] > 0.3) & mask]
    support_labels = support_labels[support_labels > 0]
    grounded = bool(in_label and out_label and in_label == out_label
                    and np.any(support_labels == in_label))

    return {
        "port_in": port_hit(CH_IN_BLOB),
        "port_out": port_hit(CH_OUT_BLOB),
        "connected_frac": main_weight / max(weighted_total, 1e-12),
        "floating_frac": 1.0 - main_weight / max(weighted_total, 1e-12),
        "n_components": int(n),
        "material_frac": weighted_total / max(float(coverage.sum()), 1e-12),
        "common_load_path": float(grounded),
    }
