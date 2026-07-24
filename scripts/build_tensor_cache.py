#!/usr/bin/env python3
"""Build the Phase-J pilot tensor cache (memmap) from generated samples.

Reads per-field .npy + .json samples, projects each to the fixed pilot tensor
layout (src/ml/tensor_spec.py), and writes compact fp16 memmaps that the
trainer mmaps directly -- so training does ZERO json parsing / FEA / disk
churn against the running generation.

Runs in the generation venv (numpy only, no torch). Keep it thread-capped and
niced so it never competes with the CPU generation:

    OMP_NUM_THREADS=1 nice -n 19 ionice -c3 \\
      venv/bin/python scripts/build_tensor_cache.py \\
        --dirs data/pool_v1/* data/pilot_v1/* data/famupgrade/* \\
        --out data/pilot_cache_v1

Outputs in --out:
    cond.f16      (N, COND_DIM, R, R) fp16   conditioning rasters
    target.f16    (N, 1, R, R)        fp16   density target in [-1,1]
    scalars.f32   (N, SCALAR_DIM)     fp32   global conditioning
    index.json    provenance + labels per row, and the layout manifest
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import hashlib
import json
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml.tensor_spec import (  # noqa: E402
    OUT_RES, COND_DIM, COND_CHANNELS, SCALAR_DIM, SCALAR_NAMES,
    build_cond, build_target, build_scalars,
)


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _iter_samples(dirs, deduplicate=True):
    """Yield (stem, meta) for every well-formed, complete sample. Skips
    half-written or corrupt samples silently (generation may be writing)."""
    seen_density = set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for jp in sorted(glob.glob(os.path.join(d, "*.json"))):
            stem = jp[:-5]
            if not (os.path.exists(stem + ".density.npy")
                    and os.path.exists(stem + ".cond_energy.npy")):
                continue
            try:
                with open(jp) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if "mechanism" not in meta or "resolution" not in meta:
                continue
            if not meta.get("validation", {}).get("overall_passed", True):
                continue
            density_hash = None
            if deduplicate:
                try:
                    density_hash = _file_hash(stem + ".density.npy")
                except OSError:
                    continue
                if density_hash in seen_density:
                    continue
                seen_density.add(density_hash)
            yield stem, meta, {"density_hash": density_hash}


def _iter_manifest(path):
    """Yield samples from build_dataset_manifest.py without re-scanning dirs."""
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get("format") not in ("opencompmech.dataset-manifest.v1",
                                       "opencompmech.curated-manifest.v1"):
        raise ValueError(f"unsupported manifest format: {manifest.get('format')}")
    for rec in manifest.get("records", []):
        stem = rec["stem"]
        try:
            with open(stem + ".json") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not (os.path.exists(stem + ".density.npy")
                and os.path.exists(stem + ".cond_energy.npy")):
            continue
        if not meta.get("validation", {}).get("overall_passed", True):
            continue
        yield stem, meta, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=[],
                    help="Sample directories; exact duplicate densities are skipped.")
    ap.add_argument("--manifest", default=None,
                    help="De-duplicated manifest from build_dataset_manifest.py. "
                         "Preferred for any training/release cache.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-res", type=int, default=OUT_RES)
    ap.add_argument("--limit", type=int, default=0, help="cap N (0 = all)")
    args = ap.parse_args()
    if not args.dirs and not args.manifest:
        ap.error("pass --manifest or at least one --dirs directory")
    if args.dirs and args.manifest:
        ap.error("--dirs and --manifest are mutually exclusive")
    R = args.out_res
    os.makedirs(args.out, exist_ok=True)

    # Pass 1: enumerate valid samples (cheap json stat/parse).
    print("[cache] scanning ...", flush=True)
    stems = []
    source = _iter_manifest(args.manifest) if args.manifest else _iter_samples(args.dirs)
    for stem, meta, provenance in source:
        stems.append((stem, meta, provenance))
        if args.limit and len(stems) >= args.limit:
            break
    N = len(stems)
    if N == 0:
        print("[cache] no samples found; abort.")
        sys.exit(1)
    print(f"[cache] {N} valid samples -> {args.out}", flush=True)

    # Allocate memmaps (written straight to disk; builder RAM stays flat).
    cond = np.lib.format.open_memmap(
        os.path.join(args.out, "cond.f16"), mode="w+",
        dtype=np.float16, shape=(N, COND_DIM, R, R))
    target = np.lib.format.open_memmap(
        os.path.join(args.out, "target.f16"), mode="w+",
        dtype=np.float16, shape=(N, 1, R, R))
    scal = np.lib.format.open_memmap(
        os.path.join(args.out, "scalars.f32"), mode="w+",
        dtype=np.float32, shape=(N, SCALAR_DIM))

    index = []
    t0 = time.time()
    written = 0
    for stem, meta, provenance in stems:
        try:
            rho = np.load(stem + ".density.npy")
            ce = np.load(stem + ".cond_energy.npy")
            cond[written] = build_cond(meta, ce, R).astype(np.float16)
            target[written] = build_target(rho, R).astype(np.float16)
            scal[written] = build_scalars(meta)
        except (OSError, ValueError, KeyError) as e:
            print(f"[cache] skip {stem}: {e}", flush=True)
            continue
        motion = meta.get("validation", {}).get("motion", {}) or {}
        index.append({
            "row": written,
            "stem": os.path.relpath(stem),
            "type": meta.get("problem_type"),
            "family": meta.get("family"),
            "motion_class": motion.get("motion_class"),
            "magnitude_class": motion.get("magnitude_class"),
            "transfer_class": motion.get("transfer_class"),
            "density_hash": provenance.get("density_hash"),
            "spec_hash": provenance.get("spec_hash"),
            "lineage_id": provenance.get("lineage_id"),
        })
        written += 1
        if written % 2000 == 0:
            dt = time.time() - t0
            print(f"[cache] {written}/{N}  ({written/dt:.0f}/s)", flush=True)

    # Memmaps stay at capacity N; the trainer reads only the first `written`
    # rows via manifest["n"] (rare mid-fill skips just leave unused tail slots).
    cond.flush(); target.flush(); scal.flush()

    manifest = {
        "n": written,
        "out_res": R,
        "cond_dim": COND_DIM,
        "cond_channels": COND_CHANNELS,
        "scalar_dim": SCALAR_DIM,
        "scalar_names": SCALAR_NAMES,
        "target_range": [-1.0, 1.0],
        "dirs": args.dirs,
        "manifest": args.manifest,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "index": index,
    }
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(manifest, f)
    print(f"[cache] DONE: {written} rows in {time.time()-t0:.0f}s -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
