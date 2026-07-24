#!/usr/bin/env python3
"""Evaluate a trained pilot checkpoint on HELD-OUT (novel) conditions and emit
quantitative metrics, so methods can be compared apples-to-apples.

Metrics (mean over N_eval val samples, EMA weights, sampler = the ckpt's own
objective):
  * dice        : overlap of generated vs ground-truth design (fidelity)
  * port_in/out : fraction of designs with material at the load/output port
  * connected   : fraction of material on the single seed-connected body (1=clean)
  * floating    : material off the load path (lower=better)
  * n_comp      : mean number of disconnected components (1=ideal)

The val split is deterministic (PilotCache seed=0), so every method sees the
SAME novel conditions.

Usage: python scripts/eval_pilot.py --cache <dir> --ckpt <run>/ckpt_final.pt \
           --out <run>/eval --n-eval 128 --sample-steps 50
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
from src.ml.physics import validity_metrics
from src.ml.tensor_spec import COND_DIM, SCALAR_DIM


def save_grid(path, targets, gens):
    def to255(a):
        return (np.clip((a + 1) / 2, 0, 1) * 255).astype(np.uint8)
    K, R = targets.shape[0], targets.shape[-1]
    pad = 2
    W = K * (R + pad) - pad
    img = np.full((2 * R + pad, W), 255, np.uint8)
    for k in range(K):
        x = k * (R + pad)
        img[:R, x:x + R] = to255(targets[k])
        img[R + pad:, x:x + R] = to255(gens[k])
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except Exception:
        np.save(path.replace(".png", ".npy"), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cargs = ck.get("args", {})
    objective = cargs.get("objective", "ddpm")
    base = cargs.get("base", 64)
    model = ConditionalUNet(1, COND_DIM, SCALAR_DIM, base=base).to(args.device)
    model.load_state_dict(ck["ema"])
    model.eval()
    obj = (RectifiedFlow(device=args.device) if objective == "flow"
           else Diffusion(T=cargs.get("timesteps", 1000), device=args.device))

    val = PilotCache(args.cache, split="val")
    n = min(args.n_eval, len(val))
    rows = list(range(n))

    all_m = []
    tgt_keep, gen_keep = [], []
    for s in range(0, n, args.batch):
        idx = rows[s:s + args.batch]
        cond = torch.stack([val[i]["cond"] for i in idx]).to(args.device)
        scal = torch.stack([val[i]["scalars"] for i in idx]).to(args.device)
        tgt = torch.stack([val[i]["target"] for i in idx]).numpy()
        with torch.autocast(args.device, dtype=torch.bfloat16):
            gen = obj.sample(model, cond, scal, steps=args.sample_steps)
        gen = gen.float().cpu().numpy()
        cond_np = cond.float().cpu().numpy()
        for k in range(len(idx)):
            m = validity_metrics(gen[k], cond_np[k])
            # fidelity vs ground truth
            a = gen[k, 0] > 0.0
            b = tgt[k, 0] > 0.0
            inter = np.logical_and(a, b).sum()
            m["dice"] = float(2 * inter / (a.sum() + b.sum() + 1e-9))
            all_m.append(m)
        if s == 0:
            tgt_keep = tgt[:8, 0]
            gen_keep = gen[:8, 0]

    keys = ["dice", "port_in", "port_out", "connected_frac", "floating_frac",
            "n_components", "material_frac"]
    agg = {k: float(np.mean([m[k] for m in all_m])) for k in keys}
    agg["port_both"] = float(np.mean(
        [(m["port_in"] > 0.5 and m["port_out"] > 0.5) for m in all_m]))
    agg["n_eval"] = n
    agg["objective"] = objective
    agg["physics_weight"] = cargs.get("physics_weight", 0.0)
    agg["ckpt"] = args.ckpt

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(agg, f, indent=2)
    save_grid(os.path.join(args.out, "grid.png"),
              np.array(tgt_keep), np.array(gen_keep))
    print("[eval] " + json.dumps(agg), flush=True)


if __name__ == "__main__":
    main()
