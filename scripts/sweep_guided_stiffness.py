#!/usr/bin/env python3
"""Guidance-scale sweep: find the funky-but-VALID regime for physics-guided
directional stiffness.

For a fixed spec + target direction, sweep the guidance strength and, at each
scale, measure the whole tradeoff:
  * C_target  : compliance in the guided direction (lower = stiffer)  -> effect
  * connected : fraction of material on the seed-connected body        -> validity
  * floating  : material off the load path                             -> validity
  * ports     : both ports carry material                              -> validity
  * dice_base : overlap with the unguided design (lower = more novel)  -> novelty

The sweet spot maximises stiffness + novelty while validity stays high. Renders
a strip (increasing guidance L->R) and prints the table.
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
from src.ml.physics import validity_metrics
from src.ml.guided_sample import (fixed_dofs_from_meta, remap_node,
                                   guided_sample, measure_dir, to_gray)
from src.ml.tensor_spec import COND_DIM, SCALAR_DIM, COND_CHANNELS


def dice(a, b):
    A, B = a > 0.5, b > 0.5
    return float(2 * np.logical_and(A, B).sum() / (A.sum() + B.sum() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.0, 0.15, 0.3, 0.5, 0.8, 1.2])
    ap.add_argument("--conn-weight", type=float, default=0.0)
    ap.add_argument("--guide-start", type=float, default=1 / 3,
                    help="fraction of trajectory before guidance kicks in "
                         "(0.0 = guide from the very start)")
    ap.add_argument("--drop-channels", type=int, nargs="*", default=[],
                    help="cond channel indices to ZERO at sampling (e.g. 0 = "
                         "cond_energy) to loosen the shape prior")
    ap.add_argument("--cond-scale", type=float, default=1.0,
                    help="multiply all conditioning (channels + scalars) to "
                         "weaken it; <1 lets guidance explore more")
    ap.add_argument("--cfg-scale", type=float, default=None,
                    help="classifier-free guidance weight (needs a cfg-dropout "
                         "model). 0=unconditional/funky, 1=nominal, 0<w<1=real "
                         "conditioning-strength dial, >1=sharpen")
    ap.add_argument("--seed", type=int, default=0,
                    help="reset for every scale so the sweep is paired")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    ca = ck.get("args", {})
    model = ConditionalUNet(1, COND_DIM, SCALAR_DIM, base=ca.get("base", 64)).to(args.device)
    model.load_state_dict(ck["ema"]); model.eval()
    obj = RectifiedFlow(device=args.device)

    val = PilotCache(args.cache, split="val")
    row = val.rows[args.row]
    stem = next(e["stem"] for e in val.manifest["index"] if e["row"] == int(row))
    meta = json.load(open(stem + ".json"))
    Rf = int(meta["resolution"])
    fea = DiffFEA(64, Emin=1e-9, penal=3.0, device=args.device, dtype=torch.float32)
    fixed = fixed_dofs_from_meta(meta, Rf, 64, args.device)
    probe = remap_node(int(meta["mechanism"]["output_node"]), Rf, 64)
    cond = val[args.row]["cond"].unsqueeze(0).to(args.device)
    scal = val[args.row]["scalars"].unsqueeze(0).to(args.device)
    cond_np = cond[0].float().cpu().numpy()   # TRUE conditioning, for validity metrics
    target_vf = float(scal[0, 0].item())
    domain = cond[:, COND_CHANNELS.index("domain_mask"):COND_CHANNELS.index("domain_mask") + 1]

    # loosen the conditioning prior fed to the MODEL (metrics still use the true ports)
    cond_use, scal_use = cond.clone(), scal.clone()
    for c in args.drop_channels:
        cond_use[:, c] = 0.0
    cond_use *= args.cond_scale
    scal_use *= args.cond_scale
    if args.drop_channels or args.cond_scale != 1.0 or args.guide_start != 1 / 3:
        print(f"[sweep] loosened cond: drop={args.drop_channels} "
              f"cond_scale={args.cond_scale} guide_start={args.guide_start}")

    def gsample(direction, s):
        # Every row must start from identical noise; otherwise scale and a
        # random latent draw are confounded.
        torch.manual_seed(args.seed)
        return guided_sample(model, obj, cond_use, scal_use, fea, fixed, probe,
                             direction, s, args.steps, args.device,
                             guide_start=args.guide_start, conn_weight=args.conn_weight,
                             cfg_scale=args.cfg_scale, target_vf=target_vf,
                             domain=domain)

    # baseline first (scale 0 -> unguided), to pick the weaker direction + dice ref
    base = gsample(None, 0.0)
    bcx, bcy = measure_dir(base, fea, fixed, probe)
    tgt = (1.0, 0.0) if bcx > bcy else (0.0, 1.0)   # weaker (higher C) direction
    tname = "X" if tgt == (1.0, 0.0) else "Y"
    base_np = (base[0, 0].float().cpu().numpy() + 1) / 2
    print(f"[sweep] spec row {args.row} ({stem}) probe@64={probe}  "
          f"baseline Cx={bcx:.2f} Cy={bcy:.2f} -> guiding stiff-{tname}")
    print(f"{'scale':>6s} {'C_'+tname:>9s} {'connect':>8s} {'floating':>8s} "
          f"{'ports':>6s} {'dice_base':>9s}")

    imgs, rows = [], []
    for s in args.scales:
        d = base if s == 0.0 else gsample(tgt, s)
        cx, cy = measure_dir(d, fea, fixed, probe)
        ct = cx if tname == "X" else cy
        dnp = (d[0, 0].float().cpu().numpy() + 1) / 2
        vm = validity_metrics(d[0].float().cpu().numpy(), cond_np)
        db = dice(dnp, base_np)
        print(f"{s:6.2f} {ct:9.3f} {vm['connected_frac']:8.3f} "
              f"{vm['floating_frac']:8.3f} "
              f"{('yes' if vm['port_in']>0.5 and vm['port_out']>0.5 else 'no'):>6s} "
              f"{db:9.3f}", flush=True)
        imgs.append(to_gray(d))
        rows.append({"scale": s, "Cx": cx, "Cy": cy, "C_target": ct,
                     "dice_base_binary": db, "validity_proxy": vm,
                     "volume_fraction": float((dnp * (domain[0, 0].cpu().numpy() > .5)).sum() /
                                              max(1, int((domain[0, 0].cpu().numpy() > .5).sum())))})
        np.save(f"{args.out}.scale_{s:g}.density.npy", dnp.astype(np.float32))

    from PIL import Image
    S, pad = 4, 4
    R = 64 * S
    canvas = np.full((R, len(imgs) * (R + pad) - pad, 3), 255, np.uint8)
    for k, im in enumerate(imgs):
        up = np.asarray(Image.fromarray(im).resize((R, R), Image.NEAREST))
        canvas[:, k * (R + pad):k * (R + pad) + R] = up
    Image.fromarray(canvas).save(args.out)
    with open(f"{args.out}.metrics.json", "w") as f:
        json.dump({"seed": args.seed, "row": args.row, "stem": stem,
                   "target_vf": target_vf, "guide_direction": tname,
                   "scales": rows}, f, indent=2)
    print(f"[sweep] wrote {args.out}  (L->R scales {args.scales})")


if __name__ == "__main__":
    main()
