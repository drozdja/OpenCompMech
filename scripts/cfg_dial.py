#!/usr/bin/env python3
"""Probe the TRAINED classifier-free-guidance conditioning dial (cfg_scale).

Needs a cfg-dropout-trained model (train_pilot.py --cfg-dropout). For a fixed
spec, sweep cfg_scale with physics guidance OFF and, at each strength, measure:
  * dice_nom  : overlap with the nominal (cfg=1.0) design -> LOWER = funkier form
  * connected : fraction of material on the seed-connected body   -> validity
  * floating  : material off the load path                        -> validity
  * ports     : both ports carry material                         -> validity
  * Cx, Cy    : directional compliances (context)

This is the clean rebuttal to the naive cond-scale experiment: because the model
LEARNED p(x) via cond-dropout, low cfg_scale should loosen the topology while
staying on a VALID manifold (unlike EXP3, where cond-scale 0.4 broke the model).

Two strips are rendered:
  <out>_dial.png    : cfg_scale swept L->R at a fixed seed (the dial's effect)
  <out>_variety.png : N unconditional (cfg=0) samples at different seeds
                      -> "funky, not slightly-adjusted copies" evidence.
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


def strip(imgs, path):
    from PIL import Image
    S, pad = 4, 4
    R = 64 * S
    canvas = np.full((R, len(imgs) * (R + pad) - pad, 3), 255, np.uint8)
    for k, im in enumerate(imgs):
        up = np.asarray(Image.fromarray(im).resize((R, R), Image.NEAREST))
        canvas[:, k * (R + pad):k * (R + pad) + R] = up
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(canvas).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="path stem; _dial.png/_variety.png appended")
    ap.add_argument("--row", type=int, default=1)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg-scales", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    ap.add_argument("--variety-n", type=int, default=6)
    ap.add_argument("--variety-cfg", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    ca = ck.get("args", {})
    model = ConditionalUNet(1, COND_DIM, SCALAR_DIM, base=ca.get("base", 64)).to(args.device)
    model.load_state_dict(ck["ema"]); model.eval()
    obj = RectifiedFlow(device=args.device)
    print(f"[cfg] ckpt trained with cfg_dropout={ca.get('cfg_dropout')} "
          f"objective={ca.get('objective')} physics_weight={ca.get('physics_weight')}")

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
    cond_np = cond[0].float().cpu().numpy()   # TRUE conditioning for validity
    target_vf = float(scal[0, 0].item())
    domain = cond[:, COND_CHANNELS.index("domain_mask"):COND_CHANNELS.index("domain_mask") + 1]

    def gen(w, seed):
        torch.manual_seed(seed)
        return guided_sample(model, obj, cond, scal, fea, fixed, probe,
                             None, 0.0, args.steps, args.device, cfg_scale=w,
                             target_vf=target_vf, domain=domain)

    # nominal reference (cfg=1.0) for the novelty (dice) metric
    nom = gen(1.0, args.seed)
    nom_np = (nom[0, 0].float().cpu().numpy() + 1) / 2

    print(f"[cfg] spec row {args.row} ({stem}) probe@64={probe}")
    print(f"{'cfg':>5s} {'dice_nom':>8s} {'connect':>8s} {'floating':>8s} "
          f"{'ports':>6s} {'Cx':>9s} {'Cy':>9s}")
    dial_imgs, dial_metrics = [], []
    for w in args.cfg_scales:
        d = nom if w == 1.0 else gen(w, args.seed)
        dnp = (d[0, 0].float().cpu().numpy() + 1) / 2
        vm = validity_metrics(d[0].float().cpu().numpy(), cond_np)
        cx, cy = measure_dir(d, fea, fixed, probe)
        ports = "yes" if vm["port_in"] > 0.5 and vm["port_out"] > 0.5 else "no"
        print(f"{w:5.2f} {dice(dnp, nom_np):8.3f} {vm['connected_frac']:8.3f} "
              f"{vm['floating_frac']:8.3f} {ports:>6s} {cx:9.2f} {cy:9.2f}",
              flush=True)
        dial_imgs.append(to_gray(d))
        np.save(f"{args.out}.cfg_{w:g}.density.npy", dnp.astype(np.float32))
        dial_metrics.append({"cfg_scale": w, "dice_nom_binary": dice(dnp, nom_np),
                             "Cx": cx, "Cy": cy, "validity_proxy": vm})
    strip(dial_imgs, args.out + "_dial.png")
    print(f"[cfg] wrote {args.out}_dial.png  (L->R cfg {args.cfg_scales})")

    # unconditional variety: same spec, N seeds, cfg=variety_cfg
    print(f"\n[cfg] variety @ cfg={args.variety_cfg}, {args.variety_n} seeds:")
    print(f"{'seed':>5s} {'connect':>8s} {'floating':>8s} {'ports':>6s} "
          f"{'dice_nom':>8s}")
    var_imgs, variety_metrics = [], []
    for s in range(args.variety_n):
        d = gen(args.variety_cfg, 100 + s)
        dnp = (d[0, 0].float().cpu().numpy() + 1) / 2
        vm = validity_metrics(d[0].float().cpu().numpy(), cond_np)
        ports = "yes" if vm["port_in"] > 0.5 and vm["port_out"] > 0.5 else "no"
        print(f"{100+s:5d} {vm['connected_frac']:8.3f} {vm['floating_frac']:8.3f} "
              f"{ports:>6s} {dice(dnp, nom_np):8.3f}", flush=True)
        var_imgs.append(to_gray(d))
        np.save(f"{args.out}.variety_{100+s}.density.npy", dnp.astype(np.float32))
        variety_metrics.append({"seed": 100+s, "dice_nom_binary": dice(dnp, nom_np),
                                "validity_proxy": vm})
    strip(var_imgs, args.out + "_variety.png")
    with open(args.out + ".metrics.json", "w") as f:
        json.dump({"seed": args.seed, "row": args.row, "stem": stem,
                   "target_vf": target_vf, "dial": dial_metrics,
                   "variety": variety_metrics}, f, indent=2)
    print(f"[cfg] wrote {args.out}_variety.png")


if __name__ == "__main__":
    main()
