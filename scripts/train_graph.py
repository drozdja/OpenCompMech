#!/usr/bin/env python3
"""Train the SE(2)-equivariant EGNN graph generator.

Mirrors ``scripts/train_pilot.py`` deliberately — same rectified-flow objective,
same lineage split, same classifier-free-guidance dropout, same NaN guards — so
that comparing it against the raster model measures the representation and the
architecture rather than the training recipe.

    python scripts/train_graph.py --graphs data/v1_graph_128 \
        --cache data/v1_broad_cache_128 --out runs/v1graph_128 \
        --steps 60000 --batch 64 --precision fp32
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.dataset import PilotCache                                  # noqa: E402
from src.ml.egnn import MechEGNN                                       # noqa: E402
from src.ml.graph_flow import flow_loss, sample                        # noqa: E402
from src.ml.optim_guard import check_optimizer, sanitize_optimizer     # noqa: E402
from src.ml.graph_tensors import (EDGE_CH, N_ANCHOR, N_FREE, N_MAX,    # noqa: E402
                                  NODE_CH, decode)
from src.ml.tensor_spec import SCALAR_DIM                              # noqa: E402


class GraphSet(Dataset):
    """Padded graphs, restricted to the rows of a cache split."""

    def __init__(self, graphs_dir, cache_dir, split, **split_kwargs):
        z = np.load(os.path.join(graphs_dir, "graphs.npz"))
        with open(os.path.join(graphs_dir, "meta.json")) as f:
            self.meta = json.load(f)
        rows = z["rows"]
        cache = PilotCache(cache_dir, split=split, **split_kwargs)
        keep_rows = set(int(r) for r in cache.rows)
        sel = np.asarray([i for i, r in enumerate(rows) if int(r) in keep_rows],
                         dtype=np.int64)
        if sel.size == 0:
            raise SystemExit(f"no graphs in the {split} split")
        self.split_info = dict(cache.split_info)
        self.rows = rows[sel]
        self.node_x = z["node_x"][sel]
        self.edge_x = z["edge_x"][sel]
        self.anchor_pos = z["anchor_pos"][sel]
        self.anchor_vec = z["anchor_vec"][sel]
        self.anchor_present = z["anchor_present"][sel]
        self.scalars = z["scalars"][sel]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return {"node_x": torch.from_numpy(self.node_x[i].astype(np.float32)),
                "edge_x": torch.from_numpy(self.edge_x[i].astype(np.float32)),
                "anchor_pos": torch.from_numpy(self.anchor_pos[i]),
                "anchor_vec": torch.from_numpy(self.anchor_vec[i]),
                "anchor_present": torch.from_numpy(self.anchor_present[i]),
                "scalars": torch.from_numpy(self.scalars[i])}


def anchor_roles_tensor(b, device):
    r = torch.zeros(N_ANCHOR, 3)
    r[0, 0] = r[1, 1] = 1.0
    r[2, 2] = r[3, 2] = 1.0
    return r.unsqueeze(0).expand(b, -1, -1).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True, help="dir from build_graph_dataset.py")
    ap.add_argument("--cache", required=True, help="source raster cache (for the split)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="training/model/data-order seed; recorded in checkpoints")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--cfg-dropout", type=float, default=0.15)
    ap.add_argument("--edge-weight", type=float, default=1.0)
    ap.add_argument("--node-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp32",
                    help="fp32 by default: gfx1201/RDNA4 has broken half-precision "
                         "kernels (see docs/HARDWARE_NOTES_RDNA4.md)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--opt-check-every", type=int, default=200,
                    help="check+repair corrupted Adam state this often (0 = off)")
    ap.add_argument("--nan-consec", type=int, default=50)
    ap.add_argument("--nan-window", type=int, default=2000)
    ap.add_argument("--nan-window-max", type=int, default=200)
    ap.add_argument("--split-mode", default="lineage")
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--holdout-value", default=None)
    ap.add_argument("--split-plan", default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    stop_file = os.path.join(args.out, "STOP")

    split_kwargs = dict(split_mode=args.split_mode, val_frac=args.val_frac,
                        seed=args.split_seed, holdout_value=args.holdout_value,
                        split_plan=args.split_plan)
    train = GraphSet(args.graphs, args.cache, "train", **split_kwargs)
    val = GraphSet(args.graphs, args.cache, "val", **split_kwargs)
    print(f"[graph] {len(train)} train / {len(val)} val graphs "
          f"split={train.split_info}", flush=True)
    dl = DataLoader(train, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=args.num_workers, pin_memory=True,
                    persistent_workers=(args.num_workers > 0))

    model = MechEGNN(n_anchor=N_ANCHOR, node_ch=NODE_CH, edge_ch=EDGE_CH,
                     scalar_dim=SCALAR_DIM, hidden=args.hidden,
                     layers=args.layers).to(args.device)
    print(f"[graph] model params: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    def autocast_ctx():
        if args.precision == "fp32":
            return contextlib.nullcontext()
        dt = torch.float16 if args.precision == "fp16" else torch.bfloat16
        return torch.autocast(args.device, dtype=dt)

    step, loss_ema = 0, None
    nan_skips = nan_consec = nan_window = opt_repairs = 0
    window_start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            chk = check_optimizer(opt)
            if not chk["ok"]:
                print(f"[graph] corrupt optimizer state: {chk}", flush=True)
                print(f"[graph] repaired: {sanitize_optimizer(opt)}", flush=True)
        step = window_start = int(ck.get("step", 0))
        print(f"[graph] resumed @ step {step}", flush=True)

    t0 = time.time()
    it = iter(dl)
    while step < args.steps:
        if os.path.exists(stop_file):
            print("[graph] STOP file -> checkpoint and exit", flush=True)
            break
        if args.warmup > 0:
            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)

        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            b = next(it)
        dev = args.device
        node_x = b["node_x"].to(dev, non_blocking=True)
        edge_x = b["edge_x"].to(dev, non_blocking=True)
        scal = b["scalars"].to(dev, non_blocking=True)
        apos = b["anchor_pos"].to(dev, non_blocking=True)
        avec = b["anchor_vec"].to(dev, non_blocking=True)
        apres = b["anchor_present"].to(dev, non_blocking=True)
        aroles = anchor_roles_tensor(node_x.shape[0], dev)

        if args.cfg_dropout > 0:
            keep = (torch.rand(node_x.shape[0], device=dev)
                    >= args.cfg_dropout).float()
            scal = scal * keep[:, None]
            apres = apres * keep[:, None]
            apos = apos * keep[:, None, None]
            avec = avec * keep[:, None, None]

        opt.zero_grad(set_to_none=True)
        with autocast_ctx():
            loss = flow_loss(model, node_x, edge_x, scal, apos, avec, apres,
                             aroles, node_w=args.node_weight,
                             edge_w=args.edge_weight)

        bad = not torch.isfinite(loss)
        if not bad:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            bad = not torch.isfinite(gnorm)
        if bad:
            # A non-finite loss OR gradient must never reach opt.step(): the
            # grad-norm clip would rescale every gradient by NaN and poison the
            # model permanently.
            opt.zero_grad(set_to_none=True)
            nan_skips += 1
            nan_consec += 1
            nan_window += 1
            if nan_consec >= args.nan_consec:
                raise SystemExit(f"[graph] aborting: {nan_consec} consecutive "
                                 f"non-finite steps at step {step}")
            if nan_window > args.nan_window_max:
                raise SystemExit(f"[graph] aborting: {nan_window} non-finite "
                                 f"steps within {step - window_start} steps")
            step += 1
            continue
        nan_consec = 0
        opt.step()

        # gfx1201 sporadically corrupts the fused optimizer kernels, leaving a
        # negative exp_avg_sq that turns into NaN weights on the next step.
        if args.opt_check_every and step % args.opt_check_every == 0:
            rep = sanitize_optimizer(opt)
            if rep["repaired"]:
                opt_repairs += 1
                print(f"[graph] repaired optimizer @ step {step}: {rep}",
                      flush=True)

        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(args.ema).add_(pm, alpha=1 - args.ema)

        lv = float(loss.detach())
        loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
        step += 1
        if step - window_start >= args.nan_window:
            window_start, nan_window = step, 0

        if step % 100 == 0:
            print(f"[graph] step {step}/{args.steps} loss {loss_ema:.4f} "
                  f"({step/(time.time()-t0):.1f} it/s)"
                  + (f" [nan={nan_skips}]" if nan_skips else "")
                  + (f" [optfix={opt_repairs}]" if opt_repairs else ""), flush=True)

        if step % args.sample_every == 0 or step == args.steps:
            ema.eval()
            k = min(8, len(val))
            vb = [val[i] for i in range(k)]
            with autocast_ctx():
                nx, ex = sample(
                    ema,
                    torch.stack([v["scalars"] for v in vb]).to(dev),
                    torch.stack([v["anchor_pos"] for v in vb]).to(dev),
                    torch.stack([v["anchor_vec"] for v in vb]).to(dev),
                    torch.stack([v["anchor_present"] for v in vb]).to(dev),
                    anchor_roles_tensor(k, dev),
                    N_FREE, N_MAX, NODE_CH, EDGE_CH, steps=50, cfg=1.0)
            nx = nx.float().cpu().numpy()
            ex = ex.float().cpu().numpy()
            nn_, ne_ = [], []
            for j in range(k):
                g = decode(nx[j], ex[j], vb[j]["anchor_pos"].numpy(),
                           vb[j]["anchor_present"].numpy())
                nn_.append(g.n_nodes())
                ne_.append(g.n_edges())
            print(f"[graph] sample @ {step}: nodes {np.mean(nn_):.1f} "
                  f"edges {np.mean(ne_):.1f}", flush=True)
            ema.train()

        if step % args.ckpt_every == 0 or step == args.steps:
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise SystemExit(f"[graph] aborting at step {step}: non-finite "
                                 f"weights; keeping the previous checkpoint")
            torch.save({"step": step, "model": model.state_dict(),
                        "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "args": vars(args), "split": train.split_info,
                        "graph_meta": train.meta},
                       os.path.join(args.out, "ckpt.pt"))

    torch.save({"step": step, "model": model.state_dict(),
                "ema": ema.state_dict(), "args": vars(args),
                "split": train.split_info, "graph_meta": train.meta},
               os.path.join(args.out, "ckpt_final.pt"))
    print(f"[graph] done at step {step} in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
