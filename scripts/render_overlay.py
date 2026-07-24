#!/usr/bin/env python3
"""Render annotated mechanism montages.

The preferred mode is ``--manifest``.  It renders only the declared records of
a frozen corpus manifest, rather than globbing whatever happens to be on disk.
That mode writes ``render_provenance.json`` next to the PNGs, binding the
figures to the exact source-manifest hash and the selected metadata files.

Legacy positional ``ROOT OUT`` mode remains available for exploratory folders,
but it is intentionally labelled ``legacy_glob`` in its sidecar and should not
be used for case-study evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_LEGACY_ROOT = "data/v1_pool_20260720"
DEFAULT_OUT = "runs/corpus_viz_annotated"
COLS, ROWS = 5, 4          # 20 designs/group
CELL, PAD, LAB = 128, 4, 14


class ManifestRenderError(RuntimeError):
    """A frozen-manifest figure cannot be produced faithfully."""


@dataclass(frozen=True)
class RenderEntry:
    source_row: int
    source_ordinal: int
    stem: str
    base_path: Path
    metadata: dict[str, Any]
    metadata_sha256: str
    problem_type: str | None
    family: str | None
    interface_accessible: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def source_row(record: dict[str, Any], ordinal: int) -> int:
    try:
        return int(record.get("row", ordinal))
    except (TypeError, ValueError) as exc:
        raise ManifestRenderError(f"invalid manifest row {record.get('row')!r}") from exc


def density_path(base: Path) -> Path:
    return base.with_suffix(".density.npy")


def metadata_path(base: Path) -> Path:
    return base.with_suffix(".json")


def manifest_stem_candidates(stem: str, manifest_path: Path, root: Path) -> list[Path]:
    """Return deterministic candidate bases for portable manifest stems."""
    raw = Path(stem)
    if raw.is_absolute():
        return [raw]
    candidates = [root / raw, raw]
    candidates.extend(parent / raw for parent in manifest_path.parents)

    # v1 stores repository-relative stems but places manifest.json in the
    # corpus root.  Resolve a matching suffix for callers that use that root.
    corpus_name = manifest_path.parent.name
    if corpus_name in raw.parts:
        suffix = Path(*raw.parts[raw.parts.index(corpus_name) + 1:])
        candidates.append(manifest_path.parent / suffix)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def resolve_manifest_stem(stem: str, manifest_path: Path, root: Path) -> Path:
    """Resolve a stem, retaining a useful expected path for failure reports."""
    candidates = manifest_stem_candidates(stem, manifest_path, root)
    partial = None
    for candidate in candidates:
        json_file = metadata_path(candidate)
        density_file = density_path(candidate)
        if json_file.is_file() and density_file.is_file():
            return candidate
        if partial is None and (json_file.exists() or density_file.exists()):
            partial = candidate
    return partial or candidates[0]


def metadata_group(metadata: dict[str, Any], record: dict[str, Any]) -> tuple[str | None, str | None]:
    """Group by source metadata type/family, never by a directory name."""
    problem_type = clean_text(metadata.get("problem_type") or record.get("type")
                              or record.get("problem_type"))
    family = clean_text(metadata.get("family") or record.get("family"))
    if problem_type is None and family is None:
        raise ManifestRenderError("metadata has neither problem_type nor family")
    return problem_type, family


def entry_rank(entry: RenderEntry, seed: int) -> bytes:
    text = f"{seed}:{entry.source_row}:{entry.source_ordinal}:{entry.stem}"
    return hashlib.sha256(text.encode("utf-8")).digest()


def select_group_entries(entries: list[RenderEntry], max_cells: int, seed: int) -> list[RenderEntry]:
    """Deterministically prefer a balanced accessible/embedded montage."""
    accessible = sorted((entry for entry in entries if entry.interface_accessible),
                        key=lambda entry: (entry_rank(entry, seed), entry.source_row, entry.stem))
    embedded = sorted((entry for entry in entries if not entry.interface_accessible),
                      key=lambda entry: (entry_rank(entry, seed), entry.source_row, entry.stem))
    accessible_budget = max_cells // 2
    embedded_budget = max_cells - accessible_budget
    selected = accessible[:accessible_budget] + embedded[:embedded_budget]
    remaining = accessible[accessible_budget:] + embedded[embedded_budget:]
    remaining.sort(key=lambda entry: (entry_rank(entry, seed), entry.source_row, entry.stem))
    selected.extend(remaining[:max_cells - len(selected)])
    return selected


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return name or "unnamed"


def group_file_names(groups: dict[tuple[str | None, str | None], list[RenderEntry]]) -> dict[tuple[str | None, str | None], str]:
    """Keep familiar type filenames unless family is needed to disambiguate."""
    base_to_groups: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
    for key in groups:
        problem_type, family = key
        base_to_groups[safe_name(problem_type or family or "unnamed")].append(key)
    names: dict[tuple[str | None, str | None], str] = {}
    for base, keys in base_to_groups.items():
        if len(keys) == 1:
            names[keys[0]] = base
            continue
        for problem_type, family in keys:
            suffix = safe_name(family or problem_type or "unnamed")
            names[(problem_type, family)] = f"{base}__{suffix}"
    if len(set(names.values())) != len(names):
        # A sanitization collision should not silently overwrite a group.
        raise ManifestRenderError(f"group output-name collision: {names}")
    return names


def npix(node: int, res: int, cell: int) -> tuple[int, int]:
    ix, iy = node % (res + 1), node // (res + 1)
    return (min(int(ix * (cell - 1) / res), cell - 1),
            min(int(iy * (cell - 1) / res), cell - 1))


def draw_cell(entry: RenderEntry, cell: int = CELL, label_height: int = LAB) -> tuple[Image.Image, bool]:
    """Draw one declared sample; missing/corrupt files are fatal upstream."""
    density = np.load(density_path(entry.base_path))
    if density.ndim != 2:
        raise ManifestRenderError(f"{entry.stem}: expected 2D density, got {density.shape}")
    if density.shape[0] != cell or density.shape[1] != cell:
        density = np.asarray(Image.fromarray((np.clip(density, 0, 1) * 255).astype(np.uint8))
                           .resize((cell, cell), Image.Resampling.NEAREST)) / 255.0
    grayscale = (255 * (1 - np.clip(density, 0, 1))).astype(np.uint8)
    image = Image.fromarray(np.stack([grayscale, grayscale, grayscale], -1))
    drawing = ImageDraw.Draw(image)
    metadata = entry.metadata
    res = int(metadata["resolution"])
    mechanism = metadata["mechanism"]
    motion = ((metadata.get("validation", {}) or {}).get("motion", {}) or {})

    for bc in metadata.get("boundary_conditions", []):
        for node in bc.get("nodes", []):
            x, y = npix(int(node), res, cell)
            drawing.rectangle([x - 2, y - 2, x + 2, y + 2], fill=(40, 90, 255))

    def arrow(node: int, direction: Any, color: tuple[int, int, int]) -> None:
        x, y = npix(int(node), res, cell)
        vector = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(vector) + 1e-9
        dx, dy = vector / norm * 20
        drawing.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
        drawing.line([x, y, x + dx, y + dy], fill=color, width=2)
        drawing.line([x + dx, y + dy, x + dx - 0.3 * dx - 0.3 * dy,
                      y + dy - 0.3 * dy + 0.3 * dx], fill=color, width=2)

    arrow(int(mechanism["input_node"]), mechanism["input_direction"], (0, 200, 0))
    arrow(int(mechanism["output_node"]), mechanism["output_direction"], (230, 0, 230))

    out = Image.new("RGB", (cell, cell + label_height), (255, 255, 255))
    out.paste(image, (0, label_height))
    label = ImageDraw.Draw(out)
    magnitude = (motion.get("magnitude_class") or "?")[:5]
    transfer = (motion.get("transfer_class") or "?")[:3]
    ga = motion.get("ga_signed", ((metadata.get("validation", {}) or {}).get("quality", {}) or {}).get("ga", 0.0))
    tag = "ACC" if entry.interface_accessible else "EMB"
    label.text((2, 2), f"{magnitude}/{transfer} ga{float(ga):+.1f} {tag}", fill=(0, 0, 0))
    border = (0, 170, 0) if entry.interface_accessible else (220, 0, 0)
    label.rectangle([0, label_height, cell - 1, cell + label_height - 1], outline=border, width=2)
    return out, entry.interface_accessible


def load_manifest_entries(manifest_path: Path, root: Path) -> tuple[dict[str, Any], list[RenderEntry]]:
    try:
        with manifest_path.open() as f:
            manifest = json.load(f)
    except FileNotFoundError as exc:
        raise ManifestRenderError(f"manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestRenderError(f"invalid manifest JSON: {manifest_path}: {exc}") from exc
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ManifestRenderError("manifest has no records")
    declared_n = manifest.get("n_unique")
    if declared_n is not None and int(declared_n) != len(records):
        raise ManifestRenderError("manifest n_unique does not match records")

    entries: list[RenderEntry] = []
    failures: list[str] = []
    for ordinal, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append(f"record ordinal {ordinal}: not an object")
            continue
        try:
            row = source_row(record, ordinal)
            stem = record.get("stem")
            if not isinstance(stem, str) or not stem:
                raise ManifestRenderError("missing stem")
            base = resolve_manifest_stem(stem, manifest_path, root)
            missing = [str(path) for path in (metadata_path(base), density_path(base)) if not path.is_file()]
            if missing:
                raise ManifestRenderError("missing required file(s): " + ", ".join(missing))
            try:
                with metadata_path(base).open() as f:
                    metadata = json.load(f)
            except json.JSONDecodeError as exc:
                raise ManifestRenderError(f"invalid metadata JSON {metadata_path(base)}: {exc}") from exc
            problem_type, family = metadata_group(metadata, record)
            accessible = ((metadata.get("validation", {}) or {})
                          .get("port_interface", {}) or {}).get("passed") is True
            entries.append(RenderEntry(
                source_row=row, source_ordinal=ordinal, stem=stem, base_path=base,
                metadata=metadata, metadata_sha256=sha256_file(metadata_path(base)),
                problem_type=problem_type, family=family,
                interface_accessible=accessible))
        except ManifestRenderError as exc:
            failures.append(f"record row {record.get('row', ordinal)!r} ({record.get('stem')!r}): {exc}")
    if failures:
        shown = "\n".join(f"  - {failure}" for failure in failures[:12])
        more = "" if len(failures) <= 12 else f"\n  - ... and {len(failures) - 12} more"
        raise ManifestRenderError(
            f"manifest-driven rendering aborted: {len(failures)} declared record(s) are missing or invalid:\n{shown}{more}")
    entries.sort(key=lambda entry: (entry.source_row, entry.source_ordinal, entry.stem))
    return manifest, entries


def load_legacy_entries(root: Path) -> list[RenderEntry]:
    """Best-effort historical scan, retained only for exploratory use."""
    entries: list[RenderEntry] = []
    ordinal = 0
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        for density_file in sorted(directory.glob("*.density.npy")):
            base = Path(str(density_file)[:-len(".density.npy")])
            json_file = metadata_path(base)
            if not json_file.is_file():
                continue
            try:
                with json_file.open() as f:
                    metadata = json.load(f)
                problem_type, family = metadata_group(metadata, {"type": directory.name})
            except (OSError, json.JSONDecodeError, ManifestRenderError):
                continue
            accessible = ((metadata.get("validation", {}) or {})
                          .get("port_interface", {}) or {}).get("passed") is True
            entries.append(RenderEntry(
                source_row=ordinal, source_ordinal=ordinal, stem=str(base), base_path=base,
                metadata=metadata, metadata_sha256=sha256_file(json_file),
                problem_type=problem_type, family=family,
                interface_accessible=accessible))
            ordinal += 1
    return entries


def render(entries: list[RenderEntry], out: Path, seed: int, *, mode: str,
           manifest_arg: str | None = None,
           manifest_root_arg: str | None = None,
           source_manifest_sha256: str | None = None,
           source_record_count: int | None = None) -> dict[str, Any]:
    if not entries:
        raise ManifestRenderError("no renderable entries")
    groups: dict[tuple[str | None, str | None], list[RenderEntry]] = defaultdict(list)
    for entry in entries:
        groups[(entry.problem_type, entry.family)].append(entry)
    names = group_file_names(groups)
    out.mkdir(parents=True, exist_ok=True)
    group_reports = []
    for group in sorted(groups, key=lambda key: ((key[0] or ""), (key[1] or ""))):
        all_entries = groups[group]
        picked = select_group_entries(all_entries, COLS * ROWS, seed)
        width = COLS * (CELL + PAD) - PAD
        height = ROWS * (CELL + LAB + PAD) - PAD
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        selected_accessible = 0
        for index, entry in enumerate(picked):
            cell, accessible = draw_cell(entry)
            selected_accessible += int(accessible)
            row, column = divmod(index, COLS)
            canvas.paste(cell, (column * (CELL + PAD), row * (CELL + LAB + PAD)))
        filename = f"{names[group]}.png"
        canvas.save(out / filename)
        group_reports.append({
            "problem_type": group[0], "family": group[1], "output": filename,
            "declared_record_count": len(all_entries),
            "source_interface_accessible_count": sum(entry.interface_accessible for entry in all_entries),
            "source_interface_embedded_count": sum(not entry.interface_accessible for entry in all_entries),
            "selected_record_count": len(picked),
            "selected_source_interface_accessible_count": selected_accessible,
            "selected_source_interface_embedded_count": len(picked) - selected_accessible,
            "selected_records": [
                {"source_row": entry.source_row, "stem": entry.stem,
                 "metadata_sha256": entry.metadata_sha256,
                 "port_interface_passed": entry.interface_accessible}
                for entry in picked
            ],
        })
        print(f"{filename}: {len(all_entries)} declared, "
              f"{sum(entry.interface_accessible for entry in all_entries)} accessible / "
              f"{sum(not entry.interface_accessible for entry in all_entries)} embedded; "
              f"shows {selected_accessible} accessible / {len(picked) - selected_accessible} embedded",
              flush=True)

    provenance = {
        "format": "opencompmech.annotated-corpus-render.v1",
        "mode": mode,
        "source_manifest": (None if manifest_arg is None else {
            "path": manifest_arg,
            "sha256": source_manifest_sha256,
            "record_count": source_record_count,
            "stem_resolution_root": manifest_root_arg,
        }),
        "selection": {
            "seed": int(seed),
            "algorithm": (
                "rank source records by sha256(seed:source_row:source_ordinal:stem); "
                "select up to half accessible and half embedded, then fill by the same rank"),
            "max_per_group": COLS * ROWS,
        },
        "render_config": {"columns": COLS, "rows": ROWS, "cell_px": CELL,
                          "padding_px": PAD, "label_px": LAB},
        "rendered_group_count": len(group_reports),
        "rendered_selected_record_count": sum(group["selected_record_count"] for group in group_reports),
        "groups": group_reports,
    }
    with (out / "render_provenance.json").open("w") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")
    return provenance


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("legacy_root", nargs="?", help="legacy glob-scan corpus root")
    ap.add_argument("legacy_out", nargs="?", help="legacy glob-scan output directory")
    ap.add_argument("--manifest", default=None,
                    help="frozen manifest; enables provenance-bound rendering")
    ap.add_argument("--root", default=None,
                    help="repository/data root for relative manifest stems (default: current directory)")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--seed", type=int, default=20260721,
                    help="stable selection seed recorded in render_provenance.json")
    args = ap.parse_args()
    if args.manifest and (args.legacy_root or args.legacy_out):
        ap.error("use --root/--out with --manifest; positional ROOT OUT is legacy glob mode only")
    if args.root and args.legacy_root:
        ap.error("pass either --root or positional legacy ROOT, not both")
    if args.out and args.legacy_out:
        ap.error("pass either --out or positional legacy OUT, not both")
    return args


def main() -> None:
    args = parse_args()
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        root = Path(args.root or ".").resolve()
        out = Path(args.out or DEFAULT_OUT)
        try:
            try:
                manifest_sha256 = sha256_file(manifest_path)
            except FileNotFoundError as exc:
                raise ManifestRenderError(f"manifest does not exist: {manifest_path}") from exc
            manifest, entries = load_manifest_entries(manifest_path, root)
            render(entries, out, args.seed, mode="frozen_manifest",
                   manifest_arg=args.manifest,
                   manifest_root_arg=args.root or ".",
                   source_manifest_sha256=manifest_sha256,
                   source_record_count=len(manifest["records"]))
        except ManifestRenderError as exc:
            raise SystemExit(str(exc)) from exc
        return

    root = Path(args.root or args.legacy_root or DEFAULT_LEGACY_ROOT)
    out = Path(args.out or args.legacy_out or DEFAULT_OUT)
    if not root.is_dir():
        raise SystemExit(f"legacy glob root does not exist: {root}")
    entries = load_legacy_entries(root)
    try:
        render(entries, out, args.seed, mode="legacy_glob")
    except ManifestRenderError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
