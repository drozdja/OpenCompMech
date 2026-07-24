#!/usr/bin/env python3
"""Audit exact and near-topology leakage before publishing an evaluation table.

The report is deliberately separate from model inference.  It checks the
frozen cache/split provenance, exact density and binary-topology overlap, then
computes each evaluation design's nearest train binary Dice and full
conditioning-vector distance.  A unique spec/lineage hash does not prove
topology independence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_rank(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def automatic_rows(records, n: int, mode: str, frac: float, seed: int, holdout: str | None):
    if mode == "iid":
        rng = np.random.default_rng(seed)
        val = set(rng.permutation(n)[:max(1, int(round(frac * n)))].tolist())
    else:
        field = {"lineage": "lineage_id", "spec": "spec_hash", "family": "family", "type": "type"}[mode]
        groups = defaultdict(list)
        for row, rec in enumerate(records):
            groups[str(rec.get(field) or rec.get("stem"))].append(row)
        if holdout is not None:
            val = set(groups[str(holdout)])
        else:
            val = set()
            goal = max(1, int(round(frac * n)))
            ranked = sorted(groups, key=lambda g: stable_rank(seed, g))
            for group in ranked:
                if val and len(val) + len(groups[group]) > goal:
                    continue
                val.update(groups[group])
                if len(val) >= goal:
                    break
            if not val:
                val.update(groups[ranked[0]])
    train = sorted(set(range(n)) - val)
    return train, sorted(val)


def signature_batch(cond: np.ndarray, scalars: np.ndarray, grid: int = 8) -> np.ndarray:
    a = np.asarray(cond, dtype=np.float32)
    n, channels, res, _ = a.shape
    b = res // grid
    pooled = a.reshape(n, channels, grid, b, grid, b).mean(axis=(3, 5))
    return np.concatenate([pooled.reshape(n, -1), np.asarray(scalars, dtype=np.float32)], axis=1)


def binary_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="little").tobytes()).hexdigest()


def nearest_metrics(target, train_rows, eval_rows, cond, scalars, max_eval: int):
    if max_eval and len(eval_rows) > max_eval:
        eval_rows = eval_rows[:max_eval]
    train_mask = np.asarray(target[train_rows, 0] > 0, dtype=np.uint8)
    train_packed = np.packbits(train_mask.reshape(len(train_mask), -1), axis=1,
                               bitorder="little")
    train_counts = train_mask.reshape(len(train_mask), -1).sum(axis=1)
    lut = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)

    chunks = []
    for start in range(0, len(train_rows), 256):
        rows = train_rows[start:start + 256]
        chunks.append(signature_batch(cond[rows], scalars[rows]))
    train_sig = np.concatenate(chunks)
    mean = train_sig.mean(axis=0)
    std = train_sig.std(axis=0)
    std[std < 1e-5] = 1.0
    train_sig = (train_sig - mean) / std

    out = []
    for row in eval_rows:
        mask = np.asarray(target[row, 0] > 0, dtype=np.uint8)
        packed = np.packbits(mask.reshape(-1), bitorder="little")
        n_solid = int(mask.sum())
        inter = lut[np.bitwise_and(train_packed, packed)].sum(axis=1)
        dice = 2.0 * inter / np.maximum(train_counts + n_solid, 1)
        nearest = int(np.argmax(dice))
        q = signature_batch(cond[row:row + 1], scalars[row:row + 1])[0]
        condition_dist = np.mean((train_sig - (q - mean) / std) ** 2, axis=1)
        near_condition = int(np.argmin(condition_dist))
        out.append({"eval_row": int(row), "nearest_train_row": int(train_rows[nearest]),
                    "nearest_train_dice": float(dice[nearest]),
                    "nearest_train_topology_distance": float(1.0 - dice[nearest]),
                    "nearest_conditioning_train_row": int(train_rows[near_condition]),
                    "nearest_conditioning_distance": float(condition_dist[near_condition])})
    return out


def quantiles(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {}
    return {str(q): float(np.quantile(a, q)) for q in (0.0, 0.05, 0.5, 0.95, 1.0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-plan", default=None)
    ap.add_argument("--split-mode", choices=["iid", "lineage", "spec", "family", "type"], default="lineage")
    ap.add_argument("--holdout-value", default=None)
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0,
                    help="0 audits every evaluation row; otherwise deterministic prefix")
    args = ap.parse_args()
    cache = Path(args.cache).resolve()
    index_path = cache / "index.json"
    with index_path.open() as f:
        index = json.load(f)
    records = index["index"]
    n = int(index["n"])
    if args.split_plan:
        with open(args.split_plan) as f:
            plan = json.load(f)
        if plan.get("cache_index_sha256") != sha256_file(index_path):
            raise SystemExit("split plan/cache-index hash mismatch")
        train_rows = [int(r) for r in plan["train_rows"]]
        eval_rows = [int(r) for r in plan.get("test_rows", plan.get("val_rows", []))]
        split_note = f"frozen plan: {plan.get('scheme')}"
    else:
        train_rows, eval_rows = automatic_rows(records, n, args.split_mode, args.val_frac,
                                               args.seed, args.holdout_value)
        split_note = f"automatic {args.split_mode}; not a frozen test plan"
    target = np.load(cache / "target.f16", mmap_mode="r")
    cond = np.load(cache / "cond.f16", mmap_mode="r")
    scalars = np.load(cache / "scalars.f32", mmap_mode="r")

    def set_for(rows, field):
        return {str(records[r].get(field)) for r in rows}
    density_overlap = set_for(train_rows, "density_hash") & set_for(eval_rows, "density_hash")
    binary_train = {binary_hash(target[r, 0] > 0) for r in train_rows}
    binary_eval = {binary_hash(target[r, 0] > 0) for r in eval_rows}
    binary_overlap = binary_train & binary_eval
    nearest = nearest_metrics(target, train_rows, eval_rows, cond, scalars, args.max_eval)
    dice = [r["nearest_train_dice"] for r in nearest]
    condition_distance = [r["nearest_conditioning_distance"] for r in nearest]
    closest = sorted(nearest, key=lambda r: (-r["nearest_train_dice"], r["eval_row"]))[:20]
    report = {
        "format": "opencompmech.evaluation-split-audit.v1",
        "cache": str(cache), "cache_index_sha256": sha256_file(index_path),
        "split": {"note": split_note, "n_train": len(train_rows), "n_evaluation": len(eval_rows),
                  "split_mode": args.split_mode, "seed": args.seed},
        "overlap": {
            "exact_density_hash_count": len(density_overlap),
            "binary_64_topology_hash_count": len(binary_overlap),
            "spec_hash_count": len(set_for(train_rows, "spec_hash") & set_for(eval_rows, "spec_hash")),
            "lineage_id_count": len(set_for(train_rows, "lineage_id") & set_for(eval_rows, "lineage_id")),
        },
        "nearest_topology": {"n_evaluated": len(nearest), "dice_quantiles": quantiles(dice),
                             "conditioning_distance_quantiles": quantiles(condition_distance),
                             "closest_pairs": closest},
        "per_eval": nearest,
        "verdict": {
            "exact_topology_leakage_free": not density_overlap and not binary_overlap,
            "note": "Exact overlap-free does not establish topology independence; inspect nearest-Dice distribution.",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"out": str(out), "overlap": report["overlap"],
                      "nearest_dice": report["nearest_topology"]["dice_quantiles"],
                      "verdict": report["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
