#!/usr/bin/env python3
"""Derive a reproducible port-interface-accessible corpus manifest.

The broad v1 corpus deliberately keeps mechanically functional mechanisms with
embedded ports.  This utility creates a *separate* manifest whose membership is
defined only by the source metadata field
``validation.port_interface.passed is True``.  It does not rerun FEA, alter a
density, or turn the selected rows into manufacture-ready designs.

The output stays compatible with ``build_tensor_cache.py``: it retains the
base ``opencompmech.dataset-manifest.v1`` format and reindexes selected records
for a future, explicitly separate cache.  Its ``derivation`` block binds the
result to both the exact source manifest and the exact selected metadata files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


SUPPORTED_FORMATS = {
    "opencompmech.dataset-manifest.v1",
    "opencompmech.curated-manifest.v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def metadata_path_for_record(stem: str, source_manifest: Path) -> Path:
    """Resolve a manifest stem without changing its portable stored value.

    Canonical manifests store stems relative to the repository working
    directory (for example ``data/v1_pool_20260720/inverter/000001``).  The
    ancestor search also lets a user invoke this script from outside that
    directory tree.  A corpus-local fallback covers manifests which already
    store paths relative to their own directory.
    """
    raw = Path(stem)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(raw)
        candidates.extend(parent / raw for parent in source_manifest.parents)
        candidates.append(source_manifest.parent / raw)

        # The frozen broad manifest lives in the corpus root.  If its stored
        # stem includes that root (as v1 does), resolve the suffix below it.
        corpus_name = source_manifest.parent.name
        if corpus_name in raw.parts:
            suffix = Path(*raw.parts[raw.parts.index(corpus_name) + 1:])
            candidates.append(source_manifest.parent / suffix)

    seen: set[Path] = set()
    for candidate in candidates:
        json_path = candidate.with_suffix(".json")
        if json_path in seen:
            continue
        seen.add(json_path)
        if json_path.is_file():
            return json_path
    raise FileNotFoundError(
        f"cannot locate metadata for manifest stem {stem!r}; tried paths relative "
        f"to the current directory and {source_manifest}")


def selected_metadata_digest(entries: list[dict[str, Any]]) -> str:
    """Hash ordered source row/stem/metadata-hash triples without ambiguity."""
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(canonical_json({
            "source_row": entry["source_row"],
            "stem": entry["stem"],
            "metadata_sha256": entry["metadata_sha256"],
        }))
        digest.update(b"\n")
    return digest.hexdigest()


def source_row(record: dict[str, Any], ordinal: int) -> int:
    """Use the frozen source row when present; old compatible manifests use order."""
    value = record.get("row", ordinal)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid source record row {value!r}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-manifest", required=True,
                    help="frozen broad/curated manifest to filter")
    ap.add_argument("--out", required=True,
                    help="derived interface-accessible manifest JSON")
    args = ap.parse_args()

    source_path = Path(args.source_manifest).resolve()
    out_path = Path(args.out).resolve()
    if out_path == source_path:
        ap.error("--out must differ from --source-manifest; refusing to overwrite the frozen broad manifest")
    with source_path.open() as f:
        source = json.load(f)
    if source.get("format") not in SUPPORTED_FORMATS:
        raise SystemExit(f"unsupported source manifest format: {source.get('format')!r}")
    records = source.get("records")
    if not isinstance(records, list) or not records:
        raise SystemExit("source manifest has no records")
    declared_n = source.get("n_unique")
    if declared_n is not None and int(declared_n) != len(records):
        raise SystemExit("source manifest n_unique does not match its record count")

    candidates = sorted(
        ((source_row(record, ordinal), ordinal, record)
         for ordinal, record in enumerate(records)),
        key=lambda value: (value[0], value[1]),
    )
    selected: list[dict[str, Any]] = []
    rejected = 0
    for frozen_row, _ordinal, record in candidates:
        stem = record.get("stem")
        if not isinstance(stem, str) or not stem:
            raise SystemExit(f"source row {frozen_row} has no usable stem")
        metadata_path = metadata_path_for_record(stem, source_path)
        try:
            with metadata_path.open() as f:
                metadata = json.load(f)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid metadata JSON for source row {frozen_row}: {metadata_path}") from exc
        passed = ((metadata.get("validation", {}) or {})
                  .get("port_interface", {}) or {}).get("passed") is True
        if not passed:
            rejected += 1
            continue
        derived = copy.deepcopy(record)
        derived["source_row"] = frozen_row
        derived["port_interface_passed"] = True
        derived["metadata_sha256"] = sha256_file(metadata_path)
        selected.append(derived)

    if not selected:
        raise SystemExit("no records passed validation.port_interface.passed is True")
    for row, record in enumerate(selected):
        record["row"] = row

    metadata_hash = selected_metadata_digest(selected)
    output = {
        # Retain the base format so build_tensor_cache.py can consume this
        # derived manifest without a special code path.
        "format": "opencompmech.dataset-manifest.v1",
        "input_dirs": source.get("input_dirs", []),
        "require_physics": bool(source.get("require_physics", False)),
        "n_unique": len(selected),
        "n_duplicate_extras": 0,
        "rejected": {"source_not_interface_accessible": rejected},
        "records": selected,
        "duplicates": [],
        "derivation": {
            "kind": "interface_accessible_filter.v1",
            "selection_rule": "metadata.validation.port_interface.passed is exactly true",
            # Keep the caller's spelling here rather than an absolute host path
            # so repeated repository-root invocations produce byte-identical
            # manifests across machines.
            "source_manifest": args.source_manifest,
            "source_manifest_sha256": sha256_file(source_path),
            "source_manifest_format": source.get("format"),
            "source_record_count": len(records),
            "selected_record_count": len(selected),
            "selected_metadata_sha256": metadata_hash,
            "selected_metadata_sha256_algorithm": (
                "sha256 of newline-delimited canonical JSON triples "
                "{source_row,stem,metadata_sha256}, ordered by source_row then source order"),
            "record_fields_added": ["source_row", "port_interface_passed", "metadata_sha256"],
            "notes": (
                "Filtering preserves source densities and mechanical labels; it is an "
                "interface metadata subset, not a manufacturing qualification."),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(json.dumps({
        "out": str(out_path),
        "source_records": len(records),
        "interface_accessible_records": len(selected),
        "source_manifest_sha256": output["derivation"]["source_manifest_sha256"],
        "selected_metadata_sha256": metadata_hash,
    }, indent=2))


if __name__ == "__main__":
    main()
