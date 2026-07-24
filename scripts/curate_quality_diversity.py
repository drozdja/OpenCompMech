#!/usr/bin/env python3
"""Curate a release candidate by measured function and topology diversity.

This is a *selector*, not a generator: retain raw optimisation output for
research, then make a training/release manifest only from designs which pass
the strict gate.  Within each measured motion class it removes mirrored copies
of the same BVP and greedily retains designs furthest from the selected set.
That turns "many seeds" into useful coverage rather than a count of near
duplicates.
"""

import argparse
import hashlib
import json
import os

import numpy as np


def binary_density(stem):
    return np.load(stem + ".density.npy") > 0.5


def canonical_hash(mask):
    """Topology hash invariant to the rectangular dihedral symmetries."""
    choices = (mask, np.fliplr(mask), np.flipud(mask), np.rot90(mask, 2))
    return min(hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
               for x in choices)


def dice_distance(a, b):
    return 1.0 - (2.0 * np.logical_and(a, b).sum() / (a.sum() + b.sum() + 1e-9))


def quality(meta):
    v = meta.get("validation", {})
    motion = v.get("motion", {}) or {}
    sel = (v.get("port_selectivity", {}) or {}).get("output", {}) or {}
    exp = v.get("port_exposure", {}) or {}
    ga = abs(float(motion.get("ga_signed", v.get("quality", {}).get("ga", 0))))
    selectivity = float(sel.get("selectivity", 0.0))
    access = float(bool(exp.get("input", {}).get("approach_clear"))) + float(bool(exp.get("output", {}).get("approach_clear")))
    return ga + 0.15 * min(selectivity, 10.0) + 0.2 * access


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="input dataset-manifest.v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-motion", type=int, default=500)
    ap.add_argument("--min-distance", type=float, default=0.15,
                    help="minimum binary Dice distance within a motion class")
    args = ap.parse_args()
    with open(args.manifest) as f:
        source = json.load(f)
    if source.get("format") != "opencompmech.dataset-manifest.v1":
        raise ValueError("expected dataset-manifest.v1")

    cells = {}
    rejected, seen_symmetry = {}, set()
    for rec in source["records"]:
        try:
            with open(rec["stem"] + ".json") as f:
                meta = json.load(f)
            motion = (meta.get("validation", {}).get("motion", {}) or {}).get("motion_class")
            if not motion or motion == "degenerate":
                rejected["missing_motion"] = rejected.get("missing_motion", 0) + 1
                continue
            mask = binary_density(rec["stem"])
            # Symmetry is only a duplicate for an identical declared BVP.
            skey = (rec.get("spec_hash"), canonical_hash(mask))
            if skey in seen_symmetry:
                rejected["same_spec_symmetry_duplicate"] = rejected.get("same_spec_symmetry_duplicate", 0) + 1
                continue
            seen_symmetry.add(skey)
            item = dict(rec)
            item["motion_class"] = motion
            item["symmetry_hash"] = skey[1]
            item["quality_score"] = quality(meta)
            item["_mask"] = mask
            cells.setdefault(motion, []).append(item)
        except Exception:
            rejected["unreadable"] = rejected.get("unreadable", 0) + 1

    selected = []
    for motion, candidates in sorted(cells.items()):
        candidates.sort(key=lambda x: x["quality_score"], reverse=True)
        kept = []
        while candidates and len(kept) < args.per_motion:
            if not kept:
                best_i = 0
            else:
                # Quality remains a tie-breaker, but each next design must
                # contribute a visibly distinct topology to this function bin.
                scores = [min(dice_distance(c["_mask"], k["_mask"]) for k in kept)
                          for c in candidates]
                best_i = int(np.argmax(np.asarray(scores)))
                if scores[best_i] < args.min_distance:
                    break
            kept.append(candidates.pop(best_i))
        for item in kept:
            item.pop("_mask", None)
        selected.extend(kept)
        print(f"{motion}: {len(kept)}/{len(cells[motion])}")

    out = {"format": "opencompmech.curated-manifest.v1",
           "source_manifest": args.manifest, "selection": {"per_motion": args.per_motion,
           "min_distance": args.min_distance}, "n": len(selected),
           "records": selected, "rejected": rejected}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"selected {len(selected)}")


if __name__ == "__main__":
    main()
