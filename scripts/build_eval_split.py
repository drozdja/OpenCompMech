#!/usr/bin/env python3
"""Freeze reproducible train/validation/test rows for a claim-bearing run.

The current ``v1_broad_cache`` lineage IDs are unique per row, so its legacy
``lineage`` split is only an IID-ish holdout.  This tool makes that fact
explicit and also supports genuine generator-type/family exclusion for a new
OOD retrain.  It does *not* call an exact-topology grouping a near-topology
split; the latter requires the separate leakage audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_rank(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def binary_hashes(cache: Path, n: int) -> list[str]:
    target = np.load(cache / "target.f16", mmap_mode="r")
    hashes = []
    for row in range(n):
        mask = np.asarray(target[row, 0] > 0, dtype=np.uint8)
        hashes.append(hashlib.sha256(np.packbits(mask, bitorder="little").tobytes()).hexdigest())
    return hashes


def allocate_groups(groups: dict[str, list[int]], frac: float, seed: int) -> set[int]:
    n = sum(len(rows) for rows in groups.values())
    goal = min(n, max(1, int(round(frac * n))))
    chosen: set[int] = set()
    for key in sorted(groups, key=lambda k: stable_rank(seed, k)):
        rows = groups[key]
        if chosen and len(chosen) + len(rows) > goal:
            continue
        chosen.update(rows)
        if len(chosen) >= goal:
            break
    # A group can be larger than the desired fraction.  Selecting it whole is
    # safer than leaking it across partitions.
    if not chosen:
        first = min(groups, key=lambda k: stable_rank(seed, k))
        chosen.update(groups[first])
    return chosen


def rows_summary(records, rows):
    counter = Counter()
    for row in rows:
        rec = records[row]
        counter[(rec.get("type"), rec.get("family"), rec.get("motion_class"))] += 1
    return [{"type": k[0], "family": k[1], "motion_class": k[2], "n": v}
            for k, v in sorted(counter.items(), key=lambda x: tuple(str(a) for a in x[0]))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scheme", choices=["iid", "lineage", "spec", "binary_topology_exact",
                                            "type_holdout", "family_holdout"], required=True)
    ap.add_argument("--holdout-value", default=None,
                    help="required for type_holdout/family_holdout; comma-separated values allowed")
    ap.add_argument("--test-frac", type=float, default=0.04)
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=20260721)
    args = ap.parse_args()
    if not (0 < args.test_frac < 1 and 0 < args.val_frac < 1 and args.test_frac + args.val_frac < 1):
        ap.error("test/val fractions must be positive and sum to less than one")

    cache = Path(args.cache).resolve()
    index_path = cache / "index.json"
    with index_path.open() as f:
        index = json.load(f)
    records = index.get("index", [])
    n = int(index["n"])
    if len(records) != n:
        raise SystemExit("cache needs a complete v1 record index")

    scheme = args.scheme
    all_rows = set(range(n))
    notes = []
    if scheme in ("type_holdout", "family_holdout"):
        if not args.holdout_value:
            ap.error(f"--holdout-value is required for {scheme}")
        field = "type" if scheme == "type_holdout" else "family"
        held = {v.strip() for v in args.holdout_value.split(",") if v.strip()}
        test_rows = {i for i, rec in enumerate(records) if str(rec.get(field)) in held}
        if not test_rows:
            ap.error(f"no rows match {field} values {sorted(held)}")
        notes.append(f"all {field} values {sorted(held)} are test-only")
        remaining = sorted(all_rows - test_rows)
        groups = defaultdict(list)
        # Keep exact binary duplicate topologies out of train/val splits too.
        for row in remaining:
            groups[str(records[row].get("density_hash") or records[row]["stem"])].append(row)
        val_rows = allocate_groups(groups, args.val_frac / (1.0 - len(test_rows) / n), args.seed + 1)
    else:
        if scheme == "iid":
            groups = {str(i): [i] for i in range(n)}
            notes.append("IID rows; no topology/template leakage protection")
        elif scheme in ("lineage", "spec"):
            field = "lineage_id" if scheme == "lineage" else "spec_hash"
            groups = defaultdict(list)
            for i, rec in enumerate(records):
                groups[str(rec.get(field) or rec["stem"])].append(i)
            notes.append(f"groups are cache {field} values")
        else:  # binary_topology_exact
            hashes = binary_hashes(cache, n)
            groups = defaultdict(list)
            for i, h in enumerate(hashes):
                groups[h].append(i)
            notes.append("exact binary 64px topology groups only; near-neighbour audit still required")
        test_rows = allocate_groups(groups, args.test_frac, args.seed)
        remaining_groups = {k: [r for r in rows if r not in test_rows]
                            for k, rows in groups.items()}
        remaining_groups = {k: rows for k, rows in remaining_groups.items() if rows}
        val_rows = allocate_groups(remaining_groups, args.val_frac / (1.0 - len(test_rows) / n), args.seed + 1)

    train_rows = sorted(all_rows - set(test_rows) - set(val_rows))
    val_rows = sorted(val_rows)
    test_rows = sorted(test_rows)
    if not train_rows or not val_rows or not test_rows:
        raise SystemExit("split generated an empty partition")
    plan = {
        "format": "opencompmech.split-plan.v1",
        "cache": str(cache),
        "cache_index_sha256": sha256_file(index_path),
        "scheme": scheme,
        "seed": args.seed,
        "test_frac_requested": args.test_frac,
        "val_frac_requested": args.val_frac,
        "holdout_value": args.holdout_value,
        "notes": notes,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "counts": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "strata": {
            "train": rows_summary(records, train_rows),
            "val": rows_summary(records, val_rows),
            "test": rows_summary(records, test_rows),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(plan, f, indent=2)
    print(json.dumps({"out": str(out), "scheme": scheme, "counts": plan["counts"],
                      "notes": notes}, indent=2))


if __name__ == "__main__":
    main()
