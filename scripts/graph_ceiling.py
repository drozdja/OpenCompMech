#!/usr/bin/env python3
"""Measure the graph representation's ceiling against the sparse-FEA gate.

No model is involved.  This takes *ground-truth* designs, pushes them through
the graph encoder and back, and asks the verifier whether the result is still a
working mechanism.  Whatever this reports is an upper bound on any graph
generator trained on the representation — a generator cannot produce something
the representation cannot express.

Measuring it first is the lesson of the 64px raster experiment, where the
achievability ceiling — not the model — turned out to be the bottleneck.

Variants:

``native``           the source design (protocol self-check; must be 1.00)
``polyline``         graph -> raster using the faithful skeleton geometry
``polyline_vf``      the same, with member widths scaled to the spec's volume
``encoded``          the padded tensor encode->decode path a MODEL produces
                     (straight struts, one width per member)
``encoded_dense``    the same, after subdividing polylines into short segments

    python scripts/graph_ceiling.py --cache data/v1_broad_cache_128 \
        --n 40 --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_mechanism_gate import (canonicalize_density,        # noqa: E402
                                         score_gate_task)
from src.graph import (densify, from_raster, rasterize_to_volume,     # noqa: E402
                       reconstruct_valid, to_raster)
from src.ml.dataset import PilotCache                                 # noqa: E402
from src.ml.graph_tensors import decode, encode                       # noqa: E402
from src.ml.tensor_spec import COND_CHANNELS                          # noqa: E402
from src.validation.connectivity import check_volume_fraction         # noqa: E402


def gate_vf_fn(meta):
    """Volume fraction as the VERIFIER measures it.

    The gate canonicalizes to the spec's resolution and uses the problem's own
    domain mask, so projecting against any other definition of volume lands in
    the wrong place.
    """
    from scripts.backfill_conditioning import problem_from_metadata
    dom = np.asarray(problem_from_metadata(meta).domain_mask, bool)
    target = float(meta["volume_fraction_target"])

    def f(mask_img):
        rho, _ = canonicalize_density(meta, mask_img.astype(np.float32))
        _, vf = check_volume_fraction(rho, target, 0.05, dom)
        return vf
    return f


def build_variants(d, c, meta, shape, max_seg):
    g = from_raster(d, cond=c, cond_channels=COND_CHANNELS)
    vf = float(meta["volume_fraction_target"])
    dom = np.asarray(c[COND_CHANNELS.index("domain_mask")]) > 0.5

    def project(graph):
        # material outside the declared domain is never valid, so clip inside
        # the volume loop: the width scale then targets the volume of the part
        # the verifier will actually keep.
        vfn = gate_vf_fn(meta)
        img, _, _ = rasterize_to_volume(graph, vf, shape=shape,
                                        vf_fn=lambda m: vfn(m & dom))
        return (img & dom).astype(np.float32)

    vfn = gate_vf_fn(meta)
    out = {"native": (d + 1.0) * 0.5, "polyline": to_raster(g, shape=shape)}
    out["polyline_vf"] = project(g)
    # erosion-robust reconstruction of the ground-truth graph
    out["polyline_valid"] = reconstruct_valid(
        g, target_vf=vf, domain_mask=dom, shape=shape,
        vf_fn=lambda m: vfn(m & dom)).astype(np.float32)

    enc = encode(g, c, COND_CHANNELS)
    ge = decode(enc["node_x"], enc["edge_x"], enc["anchor_pos"],
                enc["anchor_present"], shape=shape)
    out["encoded"] = project(ge)

    gd = densify(g, max_seg=max_seg)
    encd = encode(gd, c, COND_CHANNELS)
    gde = decode(encd["node_x"], encd["edge_x"], encd["anchor_pos"],
                 encd["anchor_present"], shape=shape)
    out["encoded_dense"] = project(gde)
    # erosion-robust reconstruction of the encode->decode graph (the model path)
    out["encoded_valid"] = reconstruct_valid(
        gde, target_vf=vf, domain_mask=dom, shape=shape, densify_seg=0.0,
        vf_fn=lambda m: vfn(m & dom)).astype(np.float32)
    return out, g.n_nodes(), gd.n_nodes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--split", default="val")
    ap.add_argument("--split-mode", default="lineage")
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--max-seg", type=float, default=14.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--protocol", default=None)
    args = ap.parse_args()

    protocol = args.protocol or str(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "config", "evaluation_protocol.v1.json"))
    cache = PilotCache(args.cache, split=args.split, split_mode=args.split_mode,
                       val_frac=args.val_frac, seed=args.split_seed)
    target = np.load(os.path.join(args.cache, "target.f16"), mmap_mode="r")
    cond = np.load(os.path.join(args.cache, "cond.f16"), mmap_mode="r")
    shape = target.shape[-2:]

    rows = [int(r) for r in cache.rows[:args.n]]
    tasks, keys, nodes, nodes_d = [], [], [], []
    for row in rows:
        stem = cache.stem_for_row(row)
        with open(stem + ".json") as f:
            meta = json.load(f)
        d = np.asarray(target[row, 0], np.float32)
        c = np.asarray(cond[row], np.float32)
        variants, nn, nd = build_variants(d, c, meta, shape, args.max_seg)
        nodes.append(nn)
        nodes_d.append(nd)
        for tag, img in variants.items():
            tasks.append((stem, np.asarray(img, np.float32), protocol))
            keys.append((row, tag))
        print(f"[ceiling] built {len(nodes)}/{len(rows)}", flush=True)

    print(f"[ceiling] gating {len(tasks)} candidates on {args.workers} workers...",
          flush=True)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            reports = list(pool.map(score_gate_task, tasks, chunksize=1))
    else:
        reports = [score_gate_task(t) for t in tasks]

    passed = defaultdict(list)
    reasons = defaultdict(Counter)
    for (row, tag), rep in zip(keys, reports):
        passed[tag].append(bool(rep.get("passed", False)))
        for r in rep.get("failure_reasons", []):
            reasons[tag][r] += 1

    order = ["native", "polyline", "polyline_vf", "polyline_valid",
             "encoded", "encoded_dense", "encoded_valid"]
    print(f"\n=== graph representation ceiling (n={len(rows)}, {args.split}) ===")
    for tag in order:
        if tag in passed:
            print(f"  {tag:<16} pass@1 {np.mean(passed[tag]):.3f}   "
                  f"top failures: {reasons[tag].most_common(3)}")
    print(f"\n  nodes: skeleton mean {np.mean(nodes):.1f} max {max(nodes)} | "
          f"densified mean {np.mean(nodes_d):.1f} max {max(nodes_d)}")
    if not all(passed["native"]):
        print("\n!! protocol self-check FAILED — the other rows are not "
              "interpretable", flush=True)

    if args.out:
        payload = {"args": vars(args), "n": len(rows),
                   "pass_at_1": {k: float(np.mean(v)) for k, v in passed.items()},
                   "failure_reasons": {k: dict(v) for k, v in reasons.items()},
                   "nodes_mean": float(np.mean(nodes)),
                   "nodes_densified_mean": float(np.mean(nodes_d))}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[ceiling] wrote {args.out}")


if __name__ == "__main__":
    main()
