#!/usr/bin/env python3
"""Render a rich results montage from a trained pilot checkpoint.

For K held-out (novel) conditions, show per row:
  [ annotated spec | ground-truth | gen#1 | gen#2 | gen#3 | gen#4 ]
Multiple generations per spec (independent noise) demonstrate that the model
produces DIVERSE valid designs, not one memorised answer.

Ports/supports are drawn on the spec+GT columns: input=red, output=blue (with a
short arrow for the load/motion direction), fixed supports=green.
"""

import argparse
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

CH_IN, CH_INX, CH_INY, CH_OUT, CH_OUTX, CH_OUTY, CH_FIX = 1, 2, 3, 4, 5, 6, 7


def _gray(x01):
    """(H,W) in [0,1] -> (H,W,3) uint8 (dark structure on light bg)."""
    v = (255 * (1 - x01)).clip(0, 255).astype(np.uint8)  # material=black
    return np.stack([v, v, v], -1)


def _peak(ch):
    return np.unravel_index(ch.argmax(), ch.shape)  # (row, col)


def annotate(base_rgb, cond, S):
    """Upscale base by S and draw ports/supports/direction arrows."""
    from PIL import Image, ImageDraw
    H, W = base_rgb.shape[:2]
    img = Image.fromarray(base_rgb).resize((W * S, H * S), Image.NEAREST)
    d = ImageDraw.Draw(img)
    # fixed supports (green) — draw every masked pixel as a small square
    fix = cond[CH_FIX]
    ys, xs = np.where(fix > 0.5)
    for y, x in zip(ys, xs):
        d.rectangle([x * S, y * S, x * S + S - 1, y * S + S - 1],
                    fill=(0, 170, 0))
    for (blob, dx_ch, dy_ch, col) in [
            (CH_IN, CH_INX, CH_INY, (220, 30, 30)),      # input = red
            (CH_OUT, CH_OUTX, CH_OUTY, (30, 90, 230))]:  # output = blue
        r, c = _peak(cond[blob])
        cx, cy = c * S + S // 2, r * S + S // 2
        rad = S + 1
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=col)
        dx, dy = float(cond[dx_ch][r, c]), float(cond[dy_ch][r, c])
        n = (dx * dx + dy * dy) ** 0.5 + 1e-9
        L = 7 * S
        d.line([cx, cy, cx + dx / n * L, cy + dy / n * L], fill=col,
               width=max(2, S // 2))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--gens", type=int, default=4)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from PIL import Image

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    ca = ck.get("args", {})
    model = ConditionalUNet(1, COND_DIM, SCALAR_DIM, base=ca.get("base", 64)).to(args.device)
    model.load_state_dict(ck["ema"]); model.eval()
    obj = (RectifiedFlow(device=args.device) if ca.get("objective") == "flow"
           else Diffusion(T=ca.get("timesteps", 1000), device=args.device))

    val = PilotCache(args.cache, split="val")
    K, N, S = args.rows, args.gens, args.scale
    idx = list(range(K))
    cond = torch.stack([val[i]["cond"] for i in idx])
    scal = torch.stack([val[i]["scalars"] for i in idx])
    tgt = torch.stack([val[i]["target"] for i in idx]).numpy()

    # N independent generations per condition (one big batched sample call)
    cB = cond.repeat_interleave(N, 0).to(args.device)
    sB = scal.repeat_interleave(N, 0).to(args.device)
    with torch.autocast(args.device, dtype=torch.bfloat16):
        gen = obj.sample(model, cB, sB, steps=args.steps)
    gen = gen.float().cpu().numpy().reshape(K, N, 1, gen.shape[-2], gen.shape[-1])

    cond_np = cond.numpy()
    pad, R = 4, tgt.shape[-1] * S
    ncol = 2 + N
    canvas = np.full((K * (R + pad) - pad, ncol * (R + pad) - pad, 3), 255, np.uint8)

    def place(row, col, rgb):
        y, x = row * (R + pad), col * (R + pad)
        canvas[y:y + R, x:x + R] = rgb

    for r in range(K):
        d01 = (tgt[r, 0] + 1) / 2
        place(r, 0, annotate(_gray(d01), cond_np[r], S))            # spec
        gt = np.asarray(Image.fromarray(_gray(d01)).resize((R, R), Image.NEAREST))
        place(r, 1, gt)                                             # ground truth
        for g in range(N):
            gg = (gen[r, g, 0] + 1) / 2
            up = np.asarray(Image.fromarray(_gray(gg)).resize((R, R), Image.NEAREST))
            place(r, 2 + g, up)                                     # generations

    Image.fromarray(canvas).save(args.out)
    print(f"[viz] wrote {args.out}  ({canvas.shape[1]}x{canvas.shape[0]})  "
          f"cols=[spec, GT, gen x{N}]", flush=True)


if __name__ == "__main__":
    main()
