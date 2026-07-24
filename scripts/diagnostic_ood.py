#!/usr/bin/env python3
"""(C) Out-of-distribution diagnostic: how much NOVELTY does the pilot support?

Two probes, both on a trained checkpoint:
  * INTERPOLATION  — blend the conditioning of two dissimilar held-out specs and
    sweep alpha 0->1. Endpoints ~reconstruct; the middle columns are conditions
    the model NEVER saw. Coherent novel intermediates => the conditioning
    manifold is smooth and novelty is reachable by moving through it. Mush =>
    the model only works near training points.
  * VARIETY — sample the SAME spec many times (independent noise). Near-identical
    rows => the spec nearly determines the design (low per-spec novelty); varied
    rows => real diversity per spec.

Outputs ood_interp.png and ood_variety.png next to --out-dir.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml.dataset import PilotCache
from src.ml.unet import ConditionalUNet
from src.ml.diffusion import Diffusion
from src.ml.flow import RectifiedFlow
from src.ml.tensor_spec import COND_DIM, SCALAR_DIM


def gray(x01):
    v = (255 * (1 - np.clip(x01, 0, 1))).astype(np.uint8)
    return np.stack([v, v, v], -1)


def montage(rows_of_imgs, path, scale, gap=3, gapcol=(210, 210, 210)):
    from PIL import Image
    R = rows_of_imgs[0][0].shape[0] * scale
    ncol = max(len(r) for r in rows_of_imgs)
    H = len(rows_of_imgs) * (R + gap) - gap
    W = ncol * (R + gap) - gap
    canvas = np.full((H, W, 3), 255, np.uint8)
    for ri, row in enumerate(rows_of_imgs):
        for ci, im in enumerate(row):
            up = np.asarray(Image.fromarray(im).resize((R, R), Image.NEAREST))
            y, x = ri * (R + gap), ci * (R + gap)
            canvas[y:y + R, x:x + R] = up
            if ci > 0:  # thin separator to read columns
                canvas[y:y + R, x - gap:x] = gapcol
    Image.fromarray(canvas).save(path)
    return canvas.shape


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    ca = ck.get("args", {})
    m = ConditionalUNet(1, COND_DIM, SCALAR_DIM, base=ca.get("base", 64)).to(device)
    m.load_state_dict(ck["ema"]); m.eval()
    obj = (RectifiedFlow(device=device) if ca.get("objective") == "flow"
           else Diffusion(T=ca.get("timesteps", 1000), device=device))
    return m, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--alpha-steps", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    m, obj = load_model(args.ckpt, args.device)
    val = PilotCache(args.cache, split="val")

    # types per val row (for picking dissimilar interpolation endpoints)
    man = {e["row"]: e for e in val.manifest["index"]}
    vtype = [man.get(int(r), {}).get("type", "?") for r in val.rows]

    # --- INTERPOLATION: 4 pairs of DISSIMILAR (different-type) specs ---
    import itertools
    by_type = {}
    for i, t in enumerate(vtype):
        by_type.setdefault(t, []).append(i)
    types = [t for t in by_type if by_type[t]]
    pairs = [(by_type[ta][0], by_type[tb][0])
             for ta, tb in itertools.combinations(types, 2)][:4]
    if not pairs:  # single type available -> just use distinct indices
        pairs = [(2 * k, 2 * k + 1) for k in range(4) if 2 * k + 1 < len(val)]
    print(f"[ood] interpolation pairs (val idx / type): "
          + ", ".join(f"{a}:{vtype[a]}->{b}:{vtype[b]}" for a, b in pairs), flush=True)

    alphas = np.linspace(0, 1, args.alpha_steps)
    interp_rows = []
    for (a, b) in pairs:
        cA, sA = val[a]["cond"], val[a]["scalars"]
        cB, sB = val[b]["cond"], val[b]["scalars"]
        conds = torch.stack([(1 - al) * cA + al * cB for al in alphas]).to(args.device)
        scals = torch.stack([(1 - al) * sA + al * sB for al in alphas]).to(args.device)
        with torch.autocast(args.device, dtype=torch.bfloat16):
            g = obj.sample(m, conds, scals, steps=args.steps).float().cpu().numpy()
        interp_rows.append([gray((g[k, 0] + 1) / 2) for k in range(len(alphas))])
    shp1 = montage(interp_rows, os.path.join(args.out_dir, "ood_interp.png"), scale=3)

    # --- VARIETY: 4 specs, N independent samples each ---
    N = args.alpha_steps
    spec_ids = [i for i in range(min(4, len(val)))]
    var_rows = []
    for i in spec_ids:
        c = val[i]["cond"].repeat(N, 1, 1, 1).to(args.device)
        s = val[i]["scalars"].repeat(N, 1).to(args.device)
        with torch.autocast(args.device, dtype=torch.bfloat16):
            g = obj.sample(m, c, s, steps=args.steps).float().cpu().numpy()
        # pixel std across the N samples = how much they differ
        std = float(np.std([(g[k, 0] + 1) / 2 for k in range(N)], axis=0).mean())
        var_rows.append([gray((g[k, 0] + 1) / 2) for k in range(N)])
        print(f"[ood] variety spec {i}: mean pixel std across {N} samples = {std:.4f}",
              flush=True)
    shp2 = montage(var_rows, os.path.join(args.out_dir, "ood_variety.png"), scale=3)
    print(f"[ood] wrote ood_interp.png {shp1} and ood_variety.png {shp2}", flush=True)


if __name__ == "__main__":
    main()
