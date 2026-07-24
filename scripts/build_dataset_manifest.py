#!/usr/bin/env python3
"""Create a reproducible, de-duplicated manifest for mechanism/stiff samples.

The old production directories are append-only and several batches reused the
same deterministic sample IDs.  A directory count therefore is *not* a design
count.  This tool makes the unit of a dataset an explicit manifest record:

* ``density_hash`` identifies byte-identical designs;
* ``spec_hash`` groups repeated declared boundary-value specifications;
* ``lineage_id`` is a stable conservative grouping key for train/val/test;
* all required fields are checked before a sample becomes eligible.

The manifest is intentionally JSON (rather than a database) so it can be
versioned, hashed, inspected, and consumed by the tensor-cache builder.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


REQUIRED_MECH = (".density.npy", ".json", ".cond_energy.npy")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_json(value):
    """Normalise numpy-free metadata into a stable JSON representation."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _spec_payload(meta: dict) -> dict:
    """Fields known before geometry optimisation, excluding measured labels."""
    payload = {
        "tier": meta.get("tier"),
        "tier_name": meta.get("tier_name"),
        "problem_type": meta.get("problem_type"),
        "family": meta.get("family"),
        "resolution": meta.get("resolution"),
        "volume_fraction_target": meta.get("volume_fraction_target"),
        "boundary_conditions": meta.get("boundary_conditions"),
        "domain_mask_rle": meta.get("domain_mask_rle"),
    }
    # Mechanisms and stiff structures deliberately have different problem
    # descriptions.  Keep all declared actuation values, including springs.
    if "mechanism" in meta:
        payload["mechanism"] = meta["mechanism"]
    if "loads" in meta:
        payload["loads"] = meta["loads"]
    return payload


def _lineage_payload(meta: dict, spec_hash: str) -> dict:
    """Conservative group for leakage-safe splitting.

    Existing historic rows do not all contain explicit generator provenance, so
    this deliberately groups at least a whole family/type/spec together.  New
    generators should additionally write ``provenance.lineage_id``.
    """
    provenance = meta.get("provenance") or {}
    explicit = provenance.get("lineage_id")
    if explicit:
        return {"explicit": explicit}
    return {
        "family": meta.get("family"),
        "problem_type": meta.get("problem_type"),
        "spec_hash": spec_hash,
        "construction_kind": (meta.get("rr_construction") or {}).get("kind"),
    }


def _iter_jsons(dirs: list[str]):
    for directory in dirs:
        if not os.path.isdir(directory):
            continue
        for jp in sorted(glob.glob(os.path.join(directory, "*.json"))):
            yield jp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="Sample directories. Earlier directories win exact duplicates.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-physics", action="store_true",
                    help="Require displacement and stress arrays too.")
    ap.add_argument("--include-invalid", action="store_true",
                    help="Keep records not marked overall_passed.")
    args = ap.parse_args()

    seen_density: dict[str, str] = {}
    records = []
    rejected = Counter()
    duplicates = []

    for jp in _iter_jsons(args.dirs):
        stem = jp[:-5]
        missing = [suffix for suffix in REQUIRED_MECH
                   if not os.path.exists(stem + suffix)]
        if args.require_physics:
            missing += [suffix for suffix in (".displacement.npy", ".stress.npy")
                        if not os.path.exists(stem + suffix)]
        if missing:
            rejected["missing_fields"] += 1
            continue
        try:
            with open(jp) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            rejected["bad_json"] += 1
            continue
        if not args.include_invalid and not meta.get("validation", {}).get("overall_passed", False):
            rejected["invalid"] += 1
            continue

        density_hash = _sha256_file(stem + ".density.npy")
        if density_hash in seen_density:
            duplicates.append({"stem": os.path.relpath(stem),
                               "canonical_stem": seen_density[density_hash],
                               "density_hash": density_hash})
            rejected["exact_duplicate"] += 1
            continue

        spec_hash = hashlib.sha256(_canonical_json(_spec_payload(meta)).encode()).hexdigest()
        lineage_id = hashlib.sha256(
            _canonical_json(_lineage_payload(meta, spec_hash)).encode()).hexdigest()
        record = {
            "row": len(records),
            "stem": os.path.relpath(stem),
            "density_hash": density_hash,
            "spec_hash": spec_hash,
            "lineage_id": lineage_id,
            "type": meta.get("problem_type"),
            "family": meta.get("family"),
            "tier": meta.get("tier_name"),
            "sample_id": meta.get("sample_id"),
            "motion_class": (meta.get("validation", {}).get("motion", {}) or {}).get("motion_class"),
        }
        seen_density[density_hash] = record["stem"]
        records.append(record)

    manifest = {
        "format": "opencompmech.dataset-manifest.v1",
        "input_dirs": args.dirs,
        "require_physics": args.require_physics,
        "n_unique": len(records),
        "n_duplicate_extras": len(duplicates),
        "rejected": dict(rejected),
        "records": records,
        "duplicates": duplicates,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: manifest[k] for k in ("n_unique", "n_duplicate_extras", "rejected")},
                     indent=2))


if __name__ == "__main__":
    main()
