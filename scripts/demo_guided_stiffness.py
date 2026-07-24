#!/usr/bin/env python3
"""Physics-GUIDED directional-stiffness generation (prototype demo).

Takes the trained flow model and a held-out spec, then generates three designs:
  * baseline  — plain conditional sample
  * stiff-X   — sampling steered by DiffFEA toward LOW compliance under a
                horizontal probe load (= stiff in x)
  * stiff-Y   — same, vertical probe load (= stiff in y)
Guidance = classifier-style gradient of directional compliance on the predicted
design (FEA backward only, no model backprop -> fast). We then MEASURE C_x and
C_y for each design; success = stiff-X has the lowest C_x, stiff-Y the lowest C_y.
This is novelty-on-demand: designs pushed off the data manifold toward a
requested physical property, not remixed from training.
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
from src.ml.flow import RectifiedFlow
from src.ml.diff_fea import DiffFEA
from src.ml.tensor_spec import COND_DIM, SCALAR_DIM


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


def guided_sample(model, obj, cond, scal, fea, fixed, probe, direction,
                  scale, steps, device):
    """Rectified-flow Euler sampling + directional-stiffness guidance on the
    predicted design x0 (guidance applied over the last 2/3 of the trajectory)."""
    x = torch.randn(1, 1, 64, 64, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((1,), i * dt, device=device)
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            v = model(x, t * 1000.0, cond, scal)
        x = x + dt * v
        if direction is not None and i >= steps // 3:
            x0 = (x + (1 - t.view(-1, 1, 1, 1)) * v).float().detach()
            x0.requires_grad_(True)
            rho = ((x0[0, 0] + 1) * 0.5).clamp(0, 1).to(fea.dtype)
            C = fea.directional_compliance(rho, fixed, probe, direction)
            g = torch.autograd.grad(C, x0)[0]
            x = x - scale * g / (g.norm() + 1e-8)
    return x.clamp(-1, 1)


def measure(design, fea, fixed, probe):
    rho = ((design[0, 0] + 1) * 0.5).clamp(0, 1).to(fea.dtype)
    with torch.no_grad():
        cx = fea.directional_compliance(rho, fixed, probe, (1.0, 0.0)).item()
        cy = fea.directional_compliance(rho, fixed, probe, (0.0, 1.0)).item()
    return cx, cy


def gray(x01):
    v = (255 * (1 - np.clip(x01, 0, 1))).astype(np.uint8)
    return np.stack([v, v, v], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--scale", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    ca = ck.get("args", {})
    model = ConditionalUNet(1, COND_DIM, SCALAR_DIM, base=ca.get("base", 64)).to(args.device)
    model.load_state_dict(ck["ema"]); model.eval()
    obj = RectifiedFlow(device=args.device)

    val = PilotCache(args.cache, split="val")
    row = val.rows[args.row]
    stem = next(e["stem"] for e in val.manifest["index"] if e["row"] == int(row))
    meta = json.load(open(stem + ".json"))   # stem is a repo-relative path
    Rf = int(meta["resolution"])
    fea = DiffFEA(64, Emin=1e-9, penal=3.0, device=args.device, dtype=torch.float32)
    fixed = fixed_dofs_from_meta(meta, Rf, 64, args.device)
    probe = remap_node(int(meta["mechanism"]["output_node"]), Rf, 64)

    cond = val[args.row]["cond"].unsqueeze(0).to(args.device)
    scal = val[args.row]["scalars"].unsqueeze(0).to(args.device)

    designs = {}
    designs["baseline"] = guided_sample(model, obj, cond, scal, fea, fixed, probe,
                                        None, args.scale, args.steps, args.device)
    designs["stiff-X"] = guided_sample(model, obj, cond, scal, fea, fixed, probe,
                                       (1.0, 0.0), args.scale, args.steps, args.device)
    designs["stiff-Y"] = guided_sample(model, obj, cond, scal, fea, fixed, probe,
                                       (0.0, 1.0), args.scale, args.steps, args.device)

    print(f"[demo] spec row {args.row} ({stem}) probe@64={probe}")
    print(f"{'design':10s}  {'C_x':>10s}  {'C_y':>10s}  (lower = stiffer)")
    imgs = []
    for name in ["baseline", "stiff-X", "stiff-Y"]:
        cx, cy = measure(designs[name], fea, fixed, probe)
        print(f"{name:10s}  {cx:10.3f}  {cy:10.3f}")
        imgs.append(gray((designs[name][0, 0].float().cpu().numpy() + 1) / 2))

    from PIL import Image
    S, pad = 4, 4
    R = 64 * S
    canvas = np.full((R, 3 * (R + pad) - pad, 3), 255, np.uint8)
    for k, im in enumerate(imgs):
        up = np.asarray(Image.fromarray(im).resize((R, R), Image.NEAREST))
        canvas[:, k * (R + pad):k * (R + pad) + R] = up
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Image.fromarray(canvas).save(args.out)
    print(f"[demo] wrote {args.out}  [baseline | stiff-X | stiff-Y]")


if __name__ == "__main__":
    main()
