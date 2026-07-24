#!/usr/bin/env python3
"""Freeze a claim-bearing evaluation protocol for a fair CNN vs GNN comparison.

Everything downstream (the comparable eval, the tuning sweeps, the final test)
reads the artifact this produces, so the two models are scored on the *identical*
specs, and the tuning set is provably disjoint from the untouched final test set.

Three properties it guarantees, each by rule rather than by hand:

1. **General eligibility, not manual removal.** A spec is eligible iff its own
   native reference density passes the frozen functional gate *and* it has a
   converted graph (so the GNN can be scored on it too). The native-reference
   self-check is the same one the harness already runs; here it becomes a
   filter, so a borderline ground-truth design (e.g. row 7872) is excluded by
   the same rule for both models instead of being deleted by hand.

2. **Disjoint tuning / final-test sets.** Eligible specs are split, stratified by
   mechanism type, into a `tuning` set (for inference sweeps + the small
   hyperparameter budget) and an untouched `test` set. Anything beyond the two
   quotas is kept as `reserve`. The partition is a stable hash of the seed and
   the design stem, so it is reproducible and never leaks across the boundary.

3. **Full provenance.** Cache tensor hashes, protocol hash, implementation
   hashes, the split parameters, the eligibility rule text, and every spec's
   native functional/interface verdict are written into the artifact.

CPU only (Torch-free gate workers) — safe to run beside a live GPU training job.

    python scripts/freeze_eval_protocol.py --cache data/v1_broad_cache_128 \
        --graphs data/v1_graph_128 --test-n 200 --tuning-n 150 --workers 8 \
        --out data/eval_protocol/v1_128_frozen.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_harness import (cache_file_hashes, implementation_hashes,   # noqa: E402
                                  source_accessible, stable_key)
from scripts.eval_mechanism_gate import load_protocol, score_gate_task         # noqa: E402
from src.ml.dataset import PilotCache                                          # noqa: E402


def _stratified_split(rows_by_type, quotas, seed):
    """Assign eligible rows to labels, stratified by type, disjointly.

    ``quotas`` is an ordered list of (label, n); rows are dealt round-robin
    across types into each label in turn until its quota is met, then the rest
    go to the final label. Ordering within a type is a stable hash, so the
    partition is reproducible and boundary-stable.
    """
    order = {t: sorted(rs, key=lambda r: stable_key(seed, str(r)))
             for t, rs in rows_by_type.items()}
    types = sorted(order, key=lambda t: (-len(order[t]), t))
    assigned = {}
    for label, n in quotas:
        taken = 0
        while taken < n and any(order[t] for t in types):
            for t in types:
                if taken >= n:
                    break
                if order[t]:
                    assigned[order[t].pop(0)] = label
                    taken += 1
    leftover_label = quotas[-1][0] if quotas else "reserve"
    for t in types:
        for r in order[t]:
            assigned.setdefault(r, "reserve")
    return assigned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--graphs", required=True,
                    help="graph dataset dir; specs are intersected with it so the "
                         "GNN can be scored on the identical set")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--split-mode", default="lineage")
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--test-n", type=int, default=200)
    ap.add_argument("--tuning-n", type=int, default=150)
    ap.add_argument("--partition-seed", type=int, default=12345)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    protocol_path = args.protocol or str(
        Path(__file__).resolve().parent.parent / "config"
        / "evaluation_protocol.v1.json")
    protocol = load_protocol(protocol_path)

    cache = PilotCache(args.cache, split=args.split, split_mode=args.split_mode,
                       val_frac=args.val_frac, seed=args.split_seed)
    pool = [int(r) for r in cache.rows]

    graph_rows = set(int(r) for r in
                     np.load(os.path.join(args.graphs, "graphs.npz"))["rows"])
    with_graph = [r for r in pool if r in graph_rows]
    print(f"[freeze] held-out pool {len(pool)} | with graph {len(with_graph)}",
          flush=True)

    # ---- gate every native reference (the eligibility rule) ----
    target = np.load(os.path.join(args.cache, "target.f16"), mmap_mode="r")
    tasks, meta_by_row = [], {}
    for r in with_graph:
        stem = cache.stem_for_row(r)
        with open(stem + ".json") as f:
            meta_by_row[r] = json.load(f)
        native01 = (np.asarray(target[r, 0], np.float32) + 1.0) * 0.5
        tasks.append((stem, native01, protocol_path))

    print(f"[freeze] gating {len(tasks)} native references on {args.workers} "
          f"workers...", flush=True)
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn")) as pool_ex:
            reports = list(pool_ex.map(score_gate_task, tasks, chunksize=1))
    else:
        reports = [score_gate_task(t) for t in tasks]
    print(f"[freeze] gated in {time.time()-t0:.0f}s", flush=True)

    specs, eligible_by_type = [], defaultdict(list)
    ineligible = []
    for r, rep in zip(with_graph, reports):
        meta = meta_by_row[r]
        func = bool(rep.get("functional_passed", False))
        rec = {"row": r, "stem": cache.stem_for_row(r),
               "type": str(cache.record_for_row(r).get("type")),
               "family": str(cache.record_for_row(r).get("family")),
               "source_interface_accessible": source_accessible(meta),
               "native_functional": func,
               "native_interface": bool(rep.get("interface_passed", False))}
        specs.append(rec)
        if func:
            eligible_by_type[rec["type"]].append(r)
        else:
            ineligible.append({"row": r, "type": rec["type"],
                               "failure_reasons": rep.get("failure_reasons", [])})

    n_elig = sum(len(v) for v in eligible_by_type.values())
    quotas = [("test", args.test_n), ("tuning", args.tuning_n), ("reserve", 0)]
    assigned = _stratified_split(eligible_by_type, quotas, args.partition_seed)
    for rec in specs:
        rec["split"] = assigned.get(rec["row"], "ineligible")

    counts = Counter(rec["split"] for rec in specs)
    payload = {
        "format": "opencompmech.eval-protocol-freeze.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eligibility_rule": (
            "A spec is eligible iff its native reference density passes the "
            "frozen functional gate at native resolution AND a converted graph "
            "exists for it. Eligible specs are stratified by type into disjoint "
            "test/tuning sets; the remainder is reserve. No spec is removed by "
            "hand."),
        "held_out_split": {"split": args.split, "mode": args.split_mode,
                           "val_frac": args.val_frac, "seed": args.split_seed},
        "partition_seed": args.partition_seed,
        "protocol_path": protocol.get("_path"),
        "protocol_sha256": protocol.get("_sha256"),
        "cache_index_sha256": cache.index_sha256,
        "cache_file_sha256": cache_file_hashes(args.cache),
        "implementation_sha256": implementation_hashes(),
        "counts": {"pool": len(pool), "with_graph": len(with_graph),
                   "native_eligible": n_elig,
                   "native_ineligible": len(ineligible),
                   "test": counts.get("test", 0),
                   "tuning": counts.get("tuning", 0),
                   "reserve": counts.get("reserve", 0)},
        "native_ineligible": ineligible,
        "specs": sorted(specs, key=lambda s: s["row"]),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n=== frozen protocol: {args.out} ===")
    for k, v in payload["counts"].items():
        print(f"  {k:<18} {v}")
    print(f"  native pass rate    {n_elig}/{len(with_graph)} = "
          f"{n_elig/max(len(with_graph),1):.3f}")
    if ineligible:
        print(f"  excluded by rule    rows {[x['row'] for x in ineligible][:12]}"
              + (" ..." if len(ineligible) > 12 else ""))
    # per-type coverage of the test set
    tt = Counter(s["type"] for s in specs if s["split"] == "test")
    print("  test types:", dict(sorted(tt.items())))


if __name__ == "__main__":
    main()
