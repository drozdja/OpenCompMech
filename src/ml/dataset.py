"""Dataset over the Phase-J pilot memmap cache (built by build_tensor_cache.py).

Reads the fp16 memmaps lazily (mmap_mode='r') so the OS page cache -- not the
process heap -- holds the data; a handful of dataloader workers then serve
batches with near-zero CPU. Keep num_workers small (2): the box's cores belong
to the CPU generation.
"""

import json
import os
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class PilotCache(Dataset):
    def __init__(self, cache_dir, split="train", val_frac=0.04, seed=0,
                 split_mode="lineage", holdout_value=None, split_plan=None):
        self.dir = cache_dir
        with open(os.path.join(cache_dir, "index.json")) as f:
            self.manifest = json.load(f)
        n = self.manifest["n"]
        self.index_path = os.path.join(cache_dir, "index.json")
        self.index_sha256 = _sha256_file(self.index_path)
        self.cond = np.load(os.path.join(cache_dir, "cond.f16"), mmap_mode="r")
        self.target = np.load(os.path.join(cache_dir, "target.f16"), mmap_mode="r")
        self.scalars = np.load(os.path.join(cache_dir, "scalars.f32"), mmap_mode="r")

        records = self.manifest.get("index", [])
        if records and len(records) != n:
            raise ValueError("cache index sample count disagrees with memmaps")
        self.split_info = {
            "split": split, "mode": split_mode, "seed": int(seed),
            "val_frac": float(val_frac), "holdout_value": holdout_value,
            "index_sha256": self.index_sha256,
        }
        if split_plan is not None:
            plan_path = Path(split_plan)
            with plan_path.open() as f:
                plan = json.load(f)
            if plan.get("format") != "opencompmech.split-plan.v1":
                raise ValueError(f"unsupported split plan: {plan.get('format')!r}")
            expected = plan.get("cache_index_sha256")
            if expected and expected != self.index_sha256:
                raise ValueError("split plan is for a different cache index")
            partitions = {}
            for label in ("train", "val", "test"):
                values = plan.get(f"{label}_rows")
                if values is None:
                    raise ValueError(f"split plan has no {label}_rows")
                parsed = [int(r) for r in values]
                if len(parsed) != len(set(parsed)):
                    raise ValueError(f"split plan has duplicate {label} rows")
                rows_set = set(parsed)
                if any(r < 0 or r >= n for r in rows_set):
                    raise ValueError(f"split plan has out-of-range {label} rows")
                partitions[label] = rows_set
            labels = tuple(partitions)
            for i, left in enumerate(labels):
                for right in labels[i + 1:]:
                    if partitions[left] & partitions[right]:
                        raise ValueError(f"split plan overlaps {left}/{right} rows")
            if set().union(*partitions.values()) != set(range(n)):
                raise ValueError("split plan does not partition every cache row")
            plan_provenance = {
                "mode": "split_plan",
                "plan": str(plan_path.resolve()),
                "plan_sha256": _sha256_file(plan_path),
                "plan_format": plan.get("format"),
                "plan_scheme": plan.get("scheme"),
                "plan_holdout_value": plan.get("holdout_value"),
                "plan_cache_index_sha256": plan.get("cache_index_sha256"),
            }
            key = {"train": "train_rows", "val": "val_rows", "test": "test_rows"}.get(split)
            if split == "all":
                self.rows = np.arange(n)
                self.split_info.update(plan_provenance)
                return
            if key is None:
                raise ValueError(f"unknown split {split!r}")
            rows = plan.get(key)
            if rows is None:
                raise ValueError(f"split plan has no {key}")
            self.rows = np.asarray(sorted(partitions[split]), dtype=np.int64)
            if self.rows.size == 0:
                raise ValueError(f"split plan's {split} split is empty")
            # The path is useful for humans, but it is not provenance: two
            # different plans can have the same filename.  Persist the content
            # hash so a checkpoint can later reject a subtly different split.
            self.split_info.update(plan_provenance)
            return

        if split_mode == "iid":
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
            n_val = min(n, max(1, int(round(val_frac * n))))
            val_idx = set(perm[:n_val].tolist())
        else:
            # A design family/spec/optimization lineage must never appear in
            # both sides.  Hashing group IDs makes the split stable regardless
            # of cache ordering or a future append operation.
            field = {"lineage": "lineage_id", "spec": "spec_hash",
                     "family": "family", "type": "type"}.get(split_mode)
            if field is None:
                raise ValueError(f"unknown split_mode {split_mode!r}")
            if not records:
                raise ValueError("group splits require a v1 cache index with sample records")
            groups = {}
            for i, rec in enumerate(records):
                key = rec.get(field) or rec.get("spec_hash") or rec.get("stem")
                groups.setdefault(str(key), []).append(i)
            if len(groups) < 2:
                raise ValueError(f"{split_mode} split has fewer than two groups; "
                                 "add coverage or use --split-mode iid only for legacy diagnostics")
            ranked = []
            if holdout_value is not None:
                holdout_key = str(holdout_value)
                if holdout_key not in groups:
                    available = ", ".join(sorted(groups)[:16])
                    raise ValueError(f"unknown {split_mode} holdout {holdout_key!r}; "
                                     f"available includes: {available}")
                val_idx = set(groups[holdout_key])
                self.split_info["holdout_value"] = holdout_key
            else:
                ranked = sorted(groups, key=lambda g: hashlib.sha256(
                    f"{seed}:{g}".encode()).digest())
                target = min(n, max(1, int(round(val_frac * n))))
                val_idx = set()
                for key in ranked:
                    # Do not turn a tiny corpus into an empty training set.
                    if len(val_idx) and len(val_idx) + len(groups[key]) > target:
                        continue
                    val_idx.update(groups[key])
                    if len(val_idx) >= target:
                        break
            if not val_idx and ranked:
                val_idx.update(groups[ranked[0]])
            if len(val_idx) >= n:
                raise ValueError(f"{split_mode} split would leave no training rows")
            perm = np.arange(n)
        if split == "val":
            self.rows = np.asarray(sorted(val_idx), dtype=np.int64)
        elif split == "train":
            self.rows = np.asarray([i for i in perm if i not in val_idx], dtype=np.int64)
        elif split == "all":
            self.rows = np.arange(n)
        else:
            raise ValueError(f"split {split!r} requires a frozen split plan")

    def record_for_row(self, row):
        """Return immutable cache provenance for a global row."""
        row = int(row)
        records = self.manifest.get("index", [])
        if not records:
            raise ValueError("cache has no record index")
        if row < 0 or row >= len(records):
            raise IndexError(row)
        return records[row]

    def stem_for_row(self, row):
        """Resolve a cache record's stem relative to the repository root.

        Cache manifests deliberately store relative paths.  This resolver makes
        scripts robust when they are launched outside the repository root.
        """
        stem = self.record_for_row(row)["stem"]
        p = Path(stem)
        if p.is_absolute():
            return str(p)
        cache = Path(self.dir).resolve()
        candidates = [Path.cwd() / p, cache.parent.parent / p]
        for candidate in candidates:
            if (candidate.parent / (candidate.name + ".json")).exists():
                return str(candidate)
        return str(candidates[-1])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        # copy=True: source is a read-only memmap; give torch a writable array.
        cond = torch.from_numpy(np.array(self.cond[r], dtype=np.float32))
        target = torch.from_numpy(np.array(self.target[r], dtype=np.float32))
        scal = torch.from_numpy(np.array(self.scalars[r], dtype=np.float32))
        return {"cond": cond, "scalars": scal, "target": target}
