#!/usr/bin/env python3
"""Build the padded-graph training set from a raster cache.

Runs the standardized raster->graph converter over every design in the cache and
stores the padded tensors a graph generator trains on.  Row indices from the
source cache are kept, so the *same* lineage split used by the raster model can
be reproduced exactly and the CNN-vs-GNN comparison stays like-for-like.

    python scripts/build_graph_dataset.py --cache data/v1_broad_cache_128 \
        --out data/v1_graph_128 --workers 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import from_raster, roundtrip_dice                  # noqa: E402
from src.ml.graph_tensors import N_FREE, N_MAX, NODE_CH, EDGE_CH, encode  # noqa: E402
from src.ml.tensor_spec import COND_CHANNELS                       # noqa: E402

_CACHE = {}


def _open(cache_dir):
    if cache_dir not in _CACHE:
        _CACHE[cache_dir] = (
            np.load(os.path.join(cache_dir, "target.f16"), mmap_mode="r"),
            np.load(os.path.join(cache_dir, "cond.f16"), mmap_mode="r"),
            np.load(os.path.join(cache_dir, "scalars.f32"), mmap_mode="r"))
    return _CACHE[cache_dir]


def _one(task):
    cache_dir, i = task
    target, cond, scalars = _open(cache_dir)
    d = np.asarray(target[i, 0], np.float32)
    c = np.asarray(cond[i], np.float32)
    try:
        g = from_raster(d, cond=c, cond_channels=COND_CHANNELS)
        if g.n_nodes() == 0:
            return None
        enc = encode(g, c, COND_CHANNELS)
        return (i, enc["node_x"], enc["edge_x"], enc["anchor_pos"],
                enc["anchor_vec"], enc["anchor_present"],
                np.asarray(scalars[i], np.float32),
                np.float32(roundtrip_dice(d, g)), np.float32(g.n_nodes()))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.cache, "index.json")) as f:
        manifest = json.load(f)
    n = manifest["n"] if not args.limit else min(args.limit, manifest["n"])
    os.makedirs(args.out, exist_ok=True)

    tasks = [(args.cache, i) for i in range(n)]
    t0 = time.time()
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, r in enumerate(pool.map(_one, tasks, chunksize=16)):
                if r is not None:
                    results.append(r)
                if (k + 1) % 2000 == 0:
                    print(f"  {k+1}/{n} ({time.time()-t0:.0f}s, "
                          f"{len(results)} kept)", flush=True)
    else:
        for k, t in enumerate(tasks):
            r = _one(t)
            if r is not None:
                results.append(r)

    if not results:
        raise SystemExit("no designs converted")
    results.sort(key=lambda r: r[0])
    rows = np.asarray([r[0] for r in results], np.int64)
    pack = lambda k, dt: np.stack([r[k] for r in results]).astype(dt)

    np.savez_compressed(
        os.path.join(args.out, "graphs.npz"),
        rows=rows,
        node_x=pack(1, np.float16), edge_x=pack(2, np.float16),
        anchor_pos=pack(3, np.float32), anchor_vec=pack(4, np.float32),
        anchor_present=pack(5, np.float32), scalars=pack(6, np.float32),
        dice=pack(7, np.float32), n_nodes=pack(8, np.float32))

    dice = pack(7, np.float32)
    nn_ = pack(8, np.float32)
    meta = {"format": "opencompmech.graph-dataset.v1",
            "source_cache": os.path.abspath(args.cache),
            "source_index_n": manifest["n"],
            "n": int(len(results)),
            "N_MAX": N_MAX, "N_FREE": N_FREE,
            "NODE_CH": NODE_CH, "EDGE_CH": EDGE_CH,
            "cond_channels": list(COND_CHANNELS),
            "roundtrip_dice_mean": float(dice.mean()),
            "roundtrip_dice_p10": float(np.percentile(dice, 10)),
            "nodes_mean": float(nn_.mean()), "nodes_max": float(nn_.max()),
            "truncated_frac": float((nn_ > N_FREE).mean())}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"[graphds] {len(results)}/{n} designs in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
