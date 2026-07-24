#!/usr/bin/env python3
"""Verification-first evaluation for conditional mechanism generation.

This is deliberately an engineering workflow evaluation, not a single-number
image benchmark.  Every method receives the same predeclared candidate budget
``K`` and each proposal is freshly re-evaluated with the versioned, sparse,
full-resolution mechanism protocol:

* ``neural``: K fixed latent seeds from the conditional flow model;
* ``retrieval``: K nearest train designs under the full conditioning vector;
* ``reference_64_contract_projection``: held-out corpus design through the
  same 64px raster, exact-envelope, and coverage-weighted volume contract used
  for neural and retrieval candidates;
* ``reference_native``: the original source design (a ceiling/check, not a
  budget-matched baseline).

For neural and retrieval, ``pass@K`` means that at least one of the K fixed
candidates passes.  Selection is also predeclared: first passing candidate in
candidate order, otherwise the lowest protocol-violation candidate.  This is a
generate→verify workflow measurement, never an oracle target-similarity score.

The legacy v1flow checkpoint has a unique-lineage 4% validation split.  It is
reported as an IID-ish held-out goal-conditioned audit only; it is not a
family/type/topology OOD result.  A type/family-holdout claim requires a
checkpoint trained with the same frozen split plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

# Avoid every FEA worker expanding BLAS pools.  Set before NumPy/SciPy imports.
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_mechanism_gate import (  # noqa: E402
    _json_default, density_at_spec_resolution, load_protocol, score_gate_task,
)
from src.ml.dataset import PilotCache  # noqa: E402
from src.ml.flow import RectifiedFlow  # noqa: E402
from src.ml.guided_sample import guided_sample, project_volume_numpy  # noqa: E402
from src.ml.tensor_spec import COND_CHANNELS, COND_DIM, SCALAR_DIM, build_target  # noqa: E402
from src.ml.unet import ConditionalUNet  # noqa: E402


# ``reference_native`` is deliberately not a 64px method: it is the
# full-resolution ceiling/check.  Every remaining method below receives the
# exact same deterministic 64px envelope + coverage-weighted volume contract
# before its fresh sparse-FEA screen.
METHOD_ORDER = (
    "reference_native",
    "reference_64_contract_projection",
    "retrieval",
    "neural",
)


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return None


def git_status() -> str | None:
    """Record whether `git_head` alone identifies the executed source."""
    try:
        return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                       text=True).strip() or None
    except Exception:
        return None


IMPLEMENTATION_FILES = (
    "scripts/eval_harness.py",
    "scripts/eval_mechanism_gate.py",
    "scripts/backfill_conditioning.py",
    "scripts/run_verification_docker.sh",
    "scripts/verify_diff_fea.py",
    "src/generators/mech.py",
    "src/solvers/linear.py",
    "src/validation/compliance.py",
    "src/validation/connectivity.py",
    "src/validation/motion_class.py",
    "src/validation/ports.py",
    "src/ml/dataset.py",
    "src/ml/diff_fea.py",
    "src/ml/flow.py",
    "src/ml/guided_sample.py",
    "src/ml/tensor_spec.py",
    "src/ml/unet.py",
    "config/evaluation_protocol.v1.json",
)


def implementation_hashes() -> dict[str, str]:
    """Hash the code that defines the reported evaluation, even on a dirty host."""
    return {rel: sha256_file(ROOT / rel) for rel in IMPLEMENTATION_FILES
            if (ROOT / rel).is_file()}


def cache_file_hashes(cache_dir: str | os.PathLike) -> dict[str, str]:
    """Bind results to the actual conditioning/target tensors, not just rows."""
    cache = Path(cache_dir)
    names = ("index.json", "cond.f16", "target.f16", "scalars.f32")
    return {name: sha256_file(cache / name) for name in names if (cache / name).is_file()}


def stable_key(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def density01(value: Any) -> np.ndarray:
    """Convert one cache/model tensor in [-1,1] to a strict HxW [0,1] array."""
    if isinstance(value, torch.Tensor):
        a = value.detach().float().cpu().numpy()
    else:
        a = np.asarray(value, dtype=np.float32)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"expected one density tensor, got {a.shape}")
    return np.clip((a + 1.0) * 0.5, 0.0, 1.0)


def downsample_signature(cond: np.ndarray, scalars: np.ndarray, grid: int = 8) -> np.ndarray:
    """Inference-only full-conditioning signature for deterministic retrieval.

    Ports/directions/supports/domain and the uniform-load energy field are all
    retained, unlike the former scalar-only retrieval.  Each spatial channel is
    average-pooled, then every feature is standardized on the train split.
    """
    c = np.asarray(cond, dtype=np.float32)
    if c.ndim == 3:
        c = c[None]
    if c.ndim != 4 or c.shape[-1] != c.shape[-2]:
        raise ValueError(f"invalid conditioning shape {c.shape}")
    n, channels, res, _ = c.shape
    if res % grid:
        raise ValueError(f"conditioning resolution {res} is not divisible by grid {grid}")
    block = res // grid
    pooled = c.reshape(n, channels, grid, block, grid, block).mean(axis=(3, 5))
    s = np.asarray(scalars, dtype=np.float32)
    if s.ndim == 1:
        s = s[None]
    if s.shape[0] != n:
        raise ValueError("conditioning/scalar batch sizes disagree")
    return np.concatenate([pooled.reshape(n, -1), s], axis=1)


class ConditioningRetriever:
    """All-train, spatially aware top-K retrieval over the model input vector.

    The vector includes BVP fields *and* requested performance/motion labels.
    In this corpus those labels were measured on the reference topology, so the
    result is a within-distribution goal-conditioned audit—not a BVP-only
    inverse-design claim.
    """

    def __init__(self, cache: PilotCache, rows: np.ndarray, grid: int = 8):
        self.cache = cache
        self.rows = np.asarray(rows, dtype=np.int64)
        self.grid = grid
        features = []
        for start in range(0, len(self.rows), 256):
            ids = self.rows[start:start + 256]
            features.append(downsample_signature(cache.cond[ids], cache.scalars[ids], grid))
        self.features = np.concatenate(features, axis=0)
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0)
        self.std[self.std < 1e-5] = 1.0
        self.normalized = (self.features - self.mean) / self.std

    def nearest_rows(self, cond: np.ndarray, scalars: np.ndarray, k: int) -> list[int]:
        q = downsample_signature(cond, scalars, self.grid)[0]
        d = np.mean((self.normalized - (q - self.mean) / self.std) ** 2, axis=1)
        k = min(int(k), len(d))
        # Full lexsort avoids nondeterministic ties at an argpartition boundary.
        order = np.lexsort((self.rows, d))[:k]
        return [int(self.rows[i]) for i in order]


class TopologyIndex:
    """Exact nearest binary-Dice search against every training density."""

    def __init__(self, cache: PilotCache, rows: np.ndarray):
        target = np.asarray(cache.target[rows, 0] > 0, dtype=np.uint8)
        # Index/query at the cache's own grid (64px baseline, 128px experiment)
        # so the packed train masks and the query mask always agree in size.
        self.res = int(target.shape[-1])
        self.packed = np.packbits(target.reshape(target.shape[0], -1), axis=1,
                                  bitorder="little")
        self.counts = target.reshape(target.shape[0], -1).sum(axis=1).astype(np.int32)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.lut = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)

    def nearest(self, density_64: np.ndarray) -> tuple[float, int]:
        rho = np.asarray(density_64, dtype=float)
        r = self.res
        if rho.shape != (r, r):
            # Cache targets were generated by block-mean pooling.  Use that
            # same representation rather than a nearest-neighbour downsample
            # that would alter the Dice metric.
            if rho.ndim == 2 and rho.shape[0] == rho.shape[1] and rho.shape[0] % r == 0:
                rho = (build_target(rho, r)[0] + 1.0) * 0.5
            else:
                rho = density_at_spec_resolution(rho, r)
        mask = np.asarray(rho > 0.5, dtype=np.uint8)
        packed = np.packbits(mask.reshape(-1), bitorder="little")
        count = int(mask.sum())
        intersection = self.lut[np.bitwise_and(self.packed, packed)].sum(axis=1)
        dice = 2.0 * intersection / np.maximum(self.counts + count, 1)
        idx = int(np.argmax(dice))
        return float(dice[idx]), int(self.rows[idx])


def source_accessible(meta: dict) -> bool:
    return bool((meta.get("validation", {}) or {}).get("port_interface", {}).get("passed", False))


def select_eval_rows(cache: PilotCache, rows: np.ndarray, n: int, seed: int) -> tuple[list[int], dict]:
    """Stable round-robin stratification by type and source interface status.

    This intentionally gives small strata visibility; it is a diagnostic
    subset, not a proportional population sample.  Its allocation is recorded
    alongside every result so confidence intervals cannot be misread as a
    corpus-wide rate.
    """
    groups: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for row in rows:
        rec = cache.record_for_row(int(row))
        stem = cache.stem_for_row(int(row))
        with open(stem + ".json") as f:
            meta = json.load(f)
        groups[(str(rec.get("type")), source_accessible(meta))].append(int(row))
    for key, values in groups.items():
        values.sort(key=lambda r: stable_key(seed, cache.record_for_row(r)["stem"]))
    eligible_counts = {key: len(values) for key, values in groups.items()}
    selected_counts: Counter = Counter()
    chosen: list[int] = []
    keys = sorted(groups, key=lambda k: (str(k[0]), not k[1]))
    while len(chosen) < n and any(groups.values()):
        for key in keys:
            if groups[key] and len(chosen) < n:
                chosen.append(groups[key].pop(0))
                selected_counts[key] += 1
    strata = []
    for key in sorted(eligible_counts, key=lambda k: (str(k[0]), not k[1])):
        strata.append({"type": key[0], "source_interface_accessible": bool(key[1]),
                       "eligible_n": int(eligible_counts[key]),
                       "selected_n": int(selected_counts[key])})
    return chosen, {
        "kind": ("exhaustive held-out evaluation" if len(chosen) == len(rows)
                 else "deterministic balanced type×source-interface stratified diagnostic subset"),
        "population_weighted": False,
        "eligible_n": int(len(rows)), "selected_n": int(len(chosen)),
        "seed": int(seed), "strata": strata,
    }


def candidate_seed(seed_base: int, row: int, candidate_index: int) -> int:
    return int((seed_base + 1_000_003 * int(row) + 97 * int(candidate_index)) % (2 ** 31 - 1))


def protocol_violation(candidate: dict, strict: bool = False) -> tuple:
    """Predeclared fallback selection score; it never reads target topology."""
    report = candidate["gate"]
    functional = report.get("functional")
    checks = functional.get("checks") if isinstance(functional, dict) else None
    # A failed worker/solver has no meaningful normalized violation vector.  It
    # must rank behind an evaluated, non-passing proposal; otherwise an empty
    # ``checks`` mapping would misleadingly count as zero failed checks and win
    # fallback selection whenever no candidate passes.
    if (report.get("worker_error") or report.get("fea_error")
            or not isinstance(checks, dict) or not checks):
        return (1_000_000, float("inf"), float("inf"), float("inf"),
                int(candidate.get("candidate_index", 0)))
    failed = sum(not bool(v) for v in checks.values())
    if strict and not bool(report.get("interface", {}).get("passed", False)):
        failed += 1
    response = report.get("response", {}) or {}
    thresholds = (report.get("functional", {}) or {}).get("thresholds", {}) or {}
    stroke_deficit = max(0.0, float(thresholds.get("min_output_stroke", 1.0))
                         - float(response.get("u_out_projected", -1e9)))
    ga_deficit = max(0.0, float(thresholds.get("min_signed_ga", 0.25))
                     - float(response.get("ga_signed", -1e9)))
    vf = (report.get("geometry", {}) or {}).get("volume_fraction", {}) or {}
    vf_error = abs(float(vf.get("actual", 1e9)) - float(vf.get("target", 0.0)))
    return failed, stroke_deficit, ga_deficit, vf_error, int(candidate["candidate_index"])


def select_candidate(candidates: list[dict], strict: bool = False) -> dict:
    pass_key = "interface_passed" if strict else "functional_passed"
    for candidate in candidates:  # fixed candidate order is the selection rule
        if bool(candidate["gate"].get(pass_key, False)):
            return candidate
    return min(candidates, key=lambda c: protocol_violation(c, strict))


def observed_target_match(observed: Any, target: Any) -> bool | None:
    """Return an evaluable class-match only when both labels actually exist."""
    if observed is None or target is None:
        return None
    return bool(observed == target)


def candidate_diversity(candidates: list[dict], pass_key: str) -> dict:
    """Within-condition binary topology diversity for the fixed K candidates.

    The primary form is deliberately restricted to FEA-passing candidates: it
    answers whether the model supplied multiple *usable* topology options, not
    merely multiple invalid raster variations.  Candidate densities have
    already received the common 64px envelope/volume contract at this point.
    """
    passing = [c for c in candidates if bool(c.get("gate", {}).get(pass_key, False))]
    masks = [np.asarray(c["density64"] > 0.5, dtype=np.uint8) for c in passing]
    packed = [np.packbits(mask.reshape(-1), bitorder="little").tobytes() for mask in masks]
    pairwise_dice = []
    for i, left in enumerate(masks):
        for right in masks[i + 1:]:
            denom = int(left.sum()) + int(right.sum())
            pairwise_dice.append(float(2 * np.logical_and(left, right).sum() / max(denom, 1)))
    return {
        "candidate_budget": len(candidates),
        "passing_candidate_count": len(masks),
        "passing_binary_unique_topologies": len(set(packed)),
        "passing_pairwise_dice_mean": float(np.mean(pairwise_dice)) if pairwise_dice else None,
        "passing_pairwise_topology_distance": (float(1.0 - np.mean(pairwise_dice))
                                                if pairwise_dice else None),
        "pairwise_n": len(pairwise_dice),
        "definition": "threshold 0.5, pairwise Dice among verifier-passing candidates only",
    }


def enrich_selected(selected: dict, meta: dict, topology: TopologyIndex) -> dict:
    report = selected["gate"]
    nearest_dice, nearest_row = topology.nearest(selected["density64"])
    source_motion = ((meta.get("validation", {}) or {}).get("motion", {}) or {})
    observed_motion = report.get("motion", {}) or {}
    response = report.get("response", {}) or {}
    vf = (report.get("geometry", {}) or {}).get("volume_fraction", {}) or {}
    return {
        "candidate_index": int(selected["candidate_index"]),
        "seed": selected.get("seed"),
        "source_train_row": selected.get("source_train_row"),
        "density_path": selected.get("density_path"),
        "functional_passed": bool(report.get("functional_passed", False)),
        "interface_passed": bool(report.get("interface_passed", False)),
        "passed_alias": bool(report.get("passed", False)),
        "failure_reasons": report.get("failure_reasons", []),
        "bc_connected": bool((report.get("bc_connectivity", {}) or {}).get("passed", False)),
        "mechanism_path_connected": bool((report.get("mechanism_path", {}) or {}).get("passed", False)),
        "u_out_projected": response.get("u_out_projected"),
        "ga_signed": response.get("ga_signed"),
        "output_alignment": response.get("output_alignment"),
        "output_selectivity": ((report.get("port_selectivity", {}) or {}).get("output", {}) or {}).get("selectivity"),
        "vf_active": vf.get("actual"),
        "vf_abs_error": (abs(float(vf.get("actual", np.nan)) - float(vf.get("target", np.nan)))
                         if vf.get("actual") is not None and vf.get("target") is not None else None),
        "requested_motion_class": source_motion.get("motion_class"),
        "observed_motion_class": observed_motion.get("motion_class"),
        "motion_class_match": observed_target_match(observed_motion.get("motion_class"),
                                                      source_motion.get("motion_class")),
        "transfer_class_match": observed_target_match(observed_motion.get("transfer_class"),
                                                        source_motion.get("transfer_class")),
        "magnitude_class_match": observed_target_match(observed_motion.get("magnitude_class"),
                                                         source_motion.get("magnitude_class")),
        "target_ga_signed": source_motion.get("ga_signed"),
        "ga_abs_error": (abs(float(response.get("ga_signed")) - float(source_motion.get("ga_signed")))
                         if response.get("ga_signed") is not None and source_motion.get("ga_signed") is not None else None),
        "nearest_train_dice": nearest_dice,
        "nearest_train_topology_distance": 1.0 - nearest_dice,
        "nearest_train_row": nearest_row,
    }


def bootstrap(values: list[float | bool | None], replicates: int, seed: int) -> dict:
    a = np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    # Resample whole specs; callers pass one aggregate value per spec.
    draws = rng.integers(0, len(a), size=(replicates, len(a)))
    means = a[draws].mean(axis=1)
    return {"mean": float(a.mean()), "lo": float(np.quantile(means, 0.025)),
            "hi": float(np.quantile(means, 0.975)), "n": int(len(a))}


def paired_bootstrap(a_values: list[float | bool | None], b_values: list[float | bool | None],
                     replicates: int, seed: int) -> dict:
    a = np.asarray([np.nan if v is None else float(v) for v in a_values], dtype=float)
    b = np.asarray([np.nan if v is None else float(v) for v in b_values], dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    if not len(d):
        return {"mean_difference": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(d), size=(replicates, len(d)))
    means = d[draws].mean(axis=1)
    return {"mean_difference": float(d.mean()), "lo": float(np.quantile(means, 0.025)),
            "hi": float(np.quantile(means, 0.975)), "n": int(len(d))}


def method_summary(spec_rows: list[dict], method: str, strict: bool,
                   bootstrap_reps: int, seed: int) -> dict:
    items = []
    failure_counter = Counter()
    for row in spec_rows:
        trial = row["methods"][method]
        candidates = trial["candidates"]
        selected = trial["selected_interface" if strict else "selected_functional"]
        pass_key = "interface_passed" if strict else "functional_passed"
        diversity = trial["diversity"]["interface" if strict else "functional"]
        selected_pass = bool(selected.get(pass_key, False))
        functional_candidate_pass_rate = np.mean([
            bool(c["gate"].get("functional_passed", False)) for c in candidates
        ])
        interface_candidate_pass_rate = np.mean([
            bool(c["gate"].get("interface_passed", False)) for c in candidates
        ])
        for candidate in candidates:
            if not candidate["gate"].get(pass_key, False):
                failure_counter.update(candidate["gate"].get("failure_reasons", []))
        items.append({
            # Keep both pass definitions in every cohort.  The strict cohort
            # changes which selected candidate supplies † quality metrics, not
            # the meaning of the pass@K columns themselves.
            "functional_candidate_pass_rate": functional_candidate_pass_rate,
            "functional_pass_at_k": any(
                bool(c["gate"].get("functional_passed", False)) for c in candidates),
            "interface_candidate_pass_rate": interface_candidate_pass_rate,
            "interface_pass_at_k": any(
                bool(c["gate"].get("interface_passed", False)) for c in candidates),
            # Compatibility aliases for downstream diagnostic readers.  New
            # reports use the explicit functional/interface keys above.
            "candidate_pass_rate": np.mean([bool(c["gate"].get(pass_key, False)) for c in candidates]),
            "pass_at_k": any(bool(c["gate"].get(pass_key, False)) for c in candidates),
            "selected_pass": selected_pass,
            # Functional quality and novelty are meaningful only for a design
            # that passed the corresponding verifier.  Keeping failed fallback
            # geometry out of these denominators prevents a pretty-but-invalid
            # proposal from inflating a headline metric.
            "motion_class_match": selected["motion_class_match"] if selected_pass else None,
            "vf_abs_error": selected["vf_abs_error"] if selected_pass else None,
            "ga_abs_error": selected["ga_abs_error"] if selected_pass else None,
            "u_out_projected": selected["u_out_projected"] if selected_pass else None,
            "ga_signed": selected["ga_signed"] if selected_pass else None,
            "output_alignment": selected["output_alignment"] if selected_pass else None,
            "output_selectivity": selected["output_selectivity"] if selected_pass else None,
            "bc_connected": selected["bc_connected"] if selected_pass else None,
            "mechanism_path_connected": (selected["mechanism_path_connected"]
                                           if selected_pass else None),
            "nearest_train_dice": selected["nearest_train_dice"] if selected_pass else None,
            "nearest_train_topology_distance": (selected["nearest_train_topology_distance"]
                                                 if selected_pass else None),
            "passing_candidate_count": diversity["passing_candidate_count"],
            "passing_binary_unique_topologies": (diversity["passing_binary_unique_topologies"]
                                                   if diversity["passing_candidate_count"] else None),
            "passing_pairwise_topology_distance": diversity["passing_pairwise_topology_distance"],
        })
    metrics = {name: bootstrap([item[name] for item in items], bootstrap_reps, seed + k)
               for k, name in enumerate(items[0])} if items else {}
    return {"n_specs": len(items), "K": len(spec_rows[0]["methods"][method]["candidates"]) if spec_rows else 0,
            "strict_interface_selection": strict, "metrics": metrics,
            "failure_taxonomy": dict(failure_counter.most_common())}


def split_assessment(cache: PilotCache, train_rows: np.ndarray, eval_rows: np.ndarray,
                     split_mode: str, checkpoint: dict, legacy: bool,
                     plan_scheme: str | None = None,
                     checkpoint_split_verified: bool = False,
                     claim_bearing_test: bool = False,
                     evaluation_partition: str = "val",
                     legacy_partition_verified: bool = False) -> dict:
    records = cache.manifest.get("index", [])
    train = [records[int(r)] for r in train_rows]
    evaluation = [records[int(r)] for r in eval_rows]
    def overlap(field):
        return sorted(set(str(r.get(field)) for r in train) & set(str(r.get(field)) for r in evaluation))
    lineage_unique = len({r.get("lineage_id") for r in records}) == len(records)
    spec_unique = len({r.get("spec_hash") for r in records}) == len(records)
    assessment = {
        "split_mode": split_mode,
        "plan_scheme": plan_scheme,
        "legacy_heldout_audit": legacy,
        "legacy_partition_provenance_verified": legacy_partition_verified,
        "checkpoint_split_provenance_verified": checkpoint_split_verified,
        "claim_bearing_test_partition": claim_bearing_test,
        "evaluation_partition": evaluation_partition,
        "n_train": len(train), "n_evaluation": len(evaluation),
        "exact_density_hash_overlap": len(overlap("density_hash")),
        "spec_hash_overlap": len(overlap("spec_hash")),
        "lineage_overlap": len(overlap("lineage_id")),
        "type_overlap": overlap("type"),
        "family_overlap": overlap("family"),
        "cache_lineage_unique_per_row": lineage_unique,
        "cache_spec_unique_per_row": spec_unique,
        "claim": "unclassified",
    }
    # A frozen plan calls these `*_holdout`; normalize that vocabulary only for
    # the claim, while preserving its original scheme in provenance.
    claim_mode = {"type_holdout": "type", "family_holdout": "family"}.get(plan_scheme,
                                                                                split_mode)
    if legacy and not legacy_partition_verified:
        assessment["claim"] = (
            "post-hoc split diagnostic only; legacy checkpoint partition does "
            "not match the requested evaluation rows")
    elif not claim_bearing_test and not legacy:
        if checkpoint_split_verified:
            assessment["claim"] = (
                "frozen validation-partition diagnostic; use the frozen test "
                "partition for a claim-bearing held-out result")
        else:
            assessment["claim"] = (
                "post-hoc split diagnostic only; checkpoint was not trained on the "
                "frozen evaluation partition")
    elif not legacy and claim_mode == "type" and not assessment["type_overlap"]:
        assessment["claim"] = "generator-type holdout (requires checkpoint split provenance)"
    elif not legacy and claim_mode == "family" and not assessment["family_overlap"]:
        assessment["claim"] = "coarse-family holdout (requires checkpoint split provenance)"
    elif legacy or (lineage_unique and split_mode in ("lineage", "spec")):
        assessment["claim"] = "IID-ish held-out goal-conditioned audit; not topology/template/family OOD"
    return assessment


def fmt_ci(item: dict, digits: int = 3, show_n: bool = False) -> str:
    if item.get("mean") is None:
        return "—"
    value = f"{item['mean']:.{digits}f} [{item['lo']:.{digits}f}, {item['hi']:.{digits}f}]"
    return f"{value}; n={item.get('n', 0)}" if show_n else value


def write_markdown(out: Path, payload: dict) -> None:
    split = payload["split_assessment"]
    lines = [
        "# Verification-first evaluation",
        "",
        f"Protocol: `{payload['protocol']['name']}` (`{payload['protocol']['sha256'][:12]}`).",
        "",
        f"Split claim: **{split['claim']}**",
        "",
        "Per-method intervals are descriptive spec-level bootstrap 95% CIs for the predeclared stratified subset; neural–retrieval differences use paired-by-spec bootstrap. `pass@K` is a fixed-K, fresh sparse-FEA-screened generate→verify workflow metric; it is not an unscreened model success rate.",
        "",
        f"Sampling: {payload['evaluation_sampling']['kind']} (selected "
        f"n={payload['evaluation_sampling']['selected_n']} from eligible "
        f"n={payload['evaluation_sampling']['eligible_n']}; not population-weighted).",
        "",
    ]
    for cohort, result in payload["summary"].items():
        cohort_n = next((int(item.get("n_specs", 0)) for item in result.values()), 0)
        selection_label = "strict-interface" if result and next(iter(result.values())).get("strict_interface_selection") else "functional"
        lines.extend([f"## Cohort: {cohort} (n={cohort_n})", "",
                      "| Method | K | functional candidate pass rate | functional pass@K | strict-interface pass@K | selected motion-class match† | selected VF abs. error† | nearest-train Dice†‡ |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for method in METHOD_ORDER:
            if method not in result:
                continue
            metrics = result[method]["metrics"]
            lines.append(
                f"| {method} | {result[method]['K']} | {fmt_ci(metrics['functional_candidate_pass_rate'])} | "
                f"{fmt_ci(metrics['functional_pass_at_k'])} | {fmt_ci(metrics['interface_pass_at_k'])} | "
                f"{fmt_ci(metrics['motion_class_match'], show_n=True)} | "
                f"{fmt_ci(metrics['vf_abs_error'], show_n=True)} | {fmt_ci(metrics['nearest_train_dice'], show_n=True)} |")
        lines.append("")
        lines.extend([f"### Selected {selection_label}-passing response quality", "",
                      "| Method | signed GA† | output alignment† | output selectivity† | output stroke† | BC/path connected† |",
                      "|---|---:|---:|---:|---:|---:|"])
        for method in METHOD_ORDER:
            if method not in result:
                continue
            metrics = result[method]["metrics"]
            # Both connectivity quantities must hold for a functional pass;
            # retaining their observed aggregate makes that gate explicit.
            lines.append(
                f"| {method} | {fmt_ci(metrics['ga_signed'], show_n=True)} | "
                f"{fmt_ci(metrics['output_alignment'], show_n=True)} | "
                f"{fmt_ci(metrics['output_selectivity'], show_n=True)} | "
                f"{fmt_ci(metrics['u_out_projected'], show_n=True)} | "
                f"{fmt_ci(metrics['bc_connected'], show_n=True)}/{fmt_ci(metrics['mechanism_path_connected'], show_n=True)} |")
        lines.append("")
        lines.extend(["### Usable within-condition topology diversity", "",
                      "| Method | mean verifier-passing candidates | distinct passing binary topologies† | pairwise topology distance among passing candidates† |",
                      "|---|---:|---:|---:|"])
        for method in ("retrieval", "neural"):
            if method not in result:
                continue
            metrics = result[method]["metrics"]
            lines.append(
                f"| {method} | {fmt_ci(metrics['passing_candidate_count'])} | "
                f"{fmt_ci(metrics['passing_binary_unique_topologies'], show_n=True)} | "
                f"{fmt_ci(metrics['passing_pairwise_topology_distance'], show_n=True)} |")
        lines.append("")
    if payload.get("paired_differences"):
        lines.extend(["## Neural minus retrieval", "",
                      "| Cohort | metric | mean difference [95% CI] |", "|---|---|---:|"])
        for cohort, metrics in payload["paired_differences"].items():
            for name, value in metrics.items():
                if value["mean_difference"] is not None:
                    lines.append(f"| {cohort} | {name} | {value['mean_difference']:.3f} "
                                 f"[{value['lo']:.3f}, {value['hi']:.3f}]; n={value['n']} |")
        lines.append("")
    lines.extend([
        "## Interpretation limits",
        "",
        "- The conditioning vector includes requested motion/performance labels. In this corpus they are measured from the reference solution, so this is a within-distribution goal-conditioned audit, not a BVP-only inverse-design result.",
        "- † Selected quality and novelty metrics are conditioned on a passing selected design; their `n` can be smaller than the cohort's pass@K denominator.",
        "- Paired quality differences exclude specifications where either selected method did not pass; their printed `n` is the joint-passing denominator. Paired pass@K uses every cohort specification.",
        "- ‡ Retrieval draws a training topology, so its nearest-train Dice is 1 by construction. It is a memorization reference, not independent novelty evidence.",
        "- Every 64px method is projected with the same fractional-domain-coverage volume rule before FEA. `reference_native` is intentionally not projected: it is a full-resolution ceiling/check. The 64px reference method quantifies that representation contract rather than native-reference fidelity.",
        "- Fresh full-resolution sparse-FEA re-evaluation is separate from the differentiable 64px guidance proxy, but it uses the same declared normalized linear-elastic model; it is not an independent material/physics certification.",
        "- Interface pass requires the generated topology itself to have clear input and output approaches. Broad functional pass does not imply a usable actuator/workpiece interface.",
        "- No dimensional material, thickness, yield, fatigue, buckling, contact, friction, process, or physical-test qualification is claimed.",
    ])
    (out / "results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--protocol", default=str(ROOT / "config" / "evaluation_protocol.v1.json"))
    ap.add_argument("--split-plan", default=None)
    ap.add_argument("--split", choices=["val", "test"], default=None)
    ap.add_argument("--split-mode", choices=["checkpoint", "iid", "lineage", "spec", "family", "type"],
                    default="checkpoint")
    ap.add_argument("--holdout-value", default=None)
    ap.add_argument("--val-frac", type=float, default=None,
                    help="override checkpoint split fraction (default: inherit checkpoint, else 0.04)")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="override checkpoint split seed (default: inherit checkpoint, else 0)")
    ap.add_argument("--allow-legacy-heldout-audit", action="store_true",
                    help="acknowledge that a no-plan unique-lineage split is IID-ish, not OOD")
    ap.add_argument("--allow-unmatched-split", action="store_true",
                    help="only for diagnostics; otherwise a split plan must match checkpoint provenance")
    ap.add_argument("--allow-reference-protocol-mismatch", action="store_true",
                    help="write a diagnostic result even if native corpus references fail the frozen gate")
    ap.add_argument("--n-specs", type=int, default=60)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed-base", type=int, default=20260721)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1,
                    help="CPU sparse-FEA workers; launch process with NUMA pinning externally")
    ap.add_argument("--save-candidates", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.K < 1 or args.n_specs < 1:
        ap.error("--K and --n-specs must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.save_candidates:
        (out / "candidates").mkdir(exist_ok=True)

    checkpoint = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    ck_args = checkpoint.get("args", {}) or {}
    checkpoint_split = checkpoint.get("split") or {}
    objective = ck_args.get("objective")
    if objective != "flow":
        raise SystemExit("eval_harness currently implements only rectified-flow sampling; "
                         f"checkpoint objective is {objective!r}")
    cfg_dropout = float(ck_args.get("cfg_dropout", 0.0) or 0.0)
    if args.cfg != 1.0 and cfg_dropout <= 0.0:
        raise SystemExit("--cfg other than 1 requires a checkpoint trained with --cfg-dropout > 0")
    inherit_checkpoint_split = args.split_mode == "checkpoint"
    split_mode = ck_args.get("split_mode", "lineage") if inherit_checkpoint_split else args.split_mode
    resolved_holdout = (args.holdout_value if args.holdout_value is not None
                        else (ck_args.get("holdout_value") if inherit_checkpoint_split else None))
    resolved_val_frac = (args.val_frac if args.val_frac is not None
                         else ck_args.get("val_frac", 0.04) if inherit_checkpoint_split else 0.04)
    resolved_split_seed = (args.split_seed if args.split_seed is not None
                           else ck_args.get("split_seed", 0) if inherit_checkpoint_split else 0)
    split_name = args.split or ("test" if args.split_plan else "val")
    legacy = args.split_plan is None
    if legacy and split_name == "test":
        raise SystemExit("--split test requires a frozen split plan")
    split_provenance_problem = None
    legacy_partition_verified = False
    if legacy:
        expected_mode = ck_args.get("split_mode", "lineage")
        expected_frac = float(ck_args.get("val_frac", 0.04))
        expected_seed = int(ck_args.get("split_seed", 0))
        expected_holdout = ck_args.get("holdout_value")
        legacy_partition_verified = (
            split_mode == expected_mode
            and float(resolved_val_frac) == expected_frac
            and int(resolved_split_seed) == expected_seed
            and resolved_holdout == expected_holdout
        )
        if not legacy_partition_verified:
            split_provenance_problem = (
                "requested legacy split does not match checkpoint split mode/seed/fraction/holdout")
            if not args.allow_unmatched_split:
                raise SystemExit(f"{split_provenance_problem}; use --allow-unmatched-split only for diagnostics")
    if legacy and not args.allow_legacy_heldout_audit:
        raise SystemExit("No frozen split plan: pass --allow-legacy-heldout-audit to label this honestly as an IID-ish audit.")
    plan_scheme = None
    plan_sha256 = None
    if args.split_plan:
        plan_path = Path(args.split_plan).resolve()
        with plan_path.open() as f:
            plan_data = json.load(f)
        if plan_data.get("format") != "opencompmech.split-plan.v1":
            raise SystemExit("unsupported frozen split plan format")
        plan_scheme = plan_data.get("scheme")
        plan_sha256 = sha256_file(plan_path)
        if plan_scheme == "type_holdout":
            split_mode = "type"
        elif plan_scheme == "family_holdout":
            split_mode = "family"
        elif plan_scheme in {"iid", "lineage", "spec"}:
            split_mode = plan_scheme
    if split_mode in {"type", "family"} and not args.split_plan:
        raise SystemExit("type/family OOD evaluation requires the frozen split plan used for training")
    checkpoint_split_verified = False
    if args.split_plan:
        if checkpoint_split.get("mode") != "split_plan":
            split_provenance_problem = "checkpoint was not trained with a frozen split plan"
        elif checkpoint_split.get("index_sha256") and checkpoint_split["index_sha256"] != sha256_file(Path(args.cache) / "index.json"):
            split_provenance_problem = "checkpoint/cache split provenance mismatch"
        elif not checkpoint_split.get("plan_sha256"):
            split_provenance_problem = "checkpoint split plan lacks a content hash"
        elif checkpoint_split["plan_sha256"] != plan_sha256:
            split_provenance_problem = "checkpoint was trained with different split-plan contents"
        elif checkpoint_split.get("plan_scheme") and checkpoint_split["plan_scheme"] != plan_scheme:
            split_provenance_problem = "checkpoint split-plan scheme disagrees with evaluation plan"
        else:
            checkpoint_split_verified = True
        if split_provenance_problem and not args.allow_unmatched_split:
            raise SystemExit(f"{split_provenance_problem}; use --allow-unmatched-split only for diagnostics")
    claim_bearing_test = bool(checkpoint_split_verified and split_name == "test")

    split_kwargs = dict(split_mode=split_mode, val_frac=resolved_val_frac, seed=resolved_split_seed,
                        holdout_value=resolved_holdout, split_plan=args.split_plan)
    evaluation = PilotCache(args.cache, split=split_name, **split_kwargs)
    train = PilotCache(args.cache, split="train", **split_kwargs)
    selected_rows, evaluation_sampling = select_eval_rows(
        evaluation, evaluation.rows, min(args.n_specs, len(evaluation)), args.seed_base)
    protocol = load_protocol(args.protocol)
    print(f"[eval] {len(selected_rows)} specs from {split_name}; train={len(train)}; "
          f"split={split_mode}; K={args.K}; workers={args.workers}", flush=True)

    model = ConditionalUNet(1, COND_DIM, SCALAR_DIM, base=ck_args.get("base", 64)).to(args.device)
    model.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"])
    model.eval()
    obj = RectifiedFlow(device=args.device)
    retriever = ConditioningRetriever(train, train.rows)
    topology = TopologyIndex(train, train.rows)

    specs: list[dict] = []
    all_candidates: list[dict] = []
    for ordinal, row in enumerate(selected_rows):
        stem = evaluation.stem_for_row(row)
        meta_path = Path(stem + ".json")
        density_path = Path(stem + ".density.npy")
        with meta_path.open() as f:
            meta = json.load(f)
        local_index = int(np.where(evaluation.rows == row)[0][0])
        item = evaluation[local_index]
        cond = item["cond"].unsqueeze(0).to(args.device)
        scalars = item["scalars"].unsqueeze(0).to(args.device)
        domain = cond[:, COND_CHANNELS.index("domain_mask"):COND_CHANNELS.index("domain_mask") + 1]
        target64 = density01(item["target"])
        source_native = np.load(density_path).astype(np.float32)
        domain_coverage64 = item["cond"][COND_CHANNELS.index("domain_mask")].numpy().astype(
            np.float32, copy=False)
        target_vf = float(item["scalars"][0].item())

        def contract_project(rho64: np.ndarray) -> tuple[np.ndarray, dict]:
            """The one fair post-process for all 64px evaluation methods."""
            return project_volume_numpy(rho64, target_vf, domain_coverage64,
                                        return_report=True)

        reference_64, reference_64_projection = contract_project(target64)
        spec = {
            "row": int(row), "stem": stem, "type": meta.get("problem_type"),
            "family": meta.get("family"), "reference_accessible": source_accessible(meta),
            "source_metadata_sha256": sha256_file(meta_path),
            "source_density_sha256": sha256_file(density_path),
            "reference_motion": ((meta.get("validation", {}) or {}).get("motion", {}) or {}),
            "methods": {},
        }
        proposed = {
            # Full-resolution ceiling/check: never apply the 64px contract.
            "reference_native": [(source_native, None, None, None)],
            # A representation-contract check, not an unprojected round trip.
            "reference_64_contract_projection": [(reference_64, None, None,
                                                   reference_64_projection)],
            "retrieval": [],
            "neural": [],
        }
        nearest_rows = retriever.nearest_rows(item["cond"].numpy(), item["scalars"].numpy(), args.K)
        for index, train_row in enumerate(nearest_rows):
            retrieved, projection = contract_project(density01(train.target[train_row]))
            proposed["retrieval"].append((retrieved, None, int(train_row), projection))
        for index in range(args.K):
            seed = candidate_seed(args.seed_base, row, index)
            torch.manual_seed(seed)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                generated = guided_sample(model, obj, cond, scalars, None, None, None, None, 0.0,
                                          args.steps, args.device, cfg_scale=args.cfg,
                                          target_vf=target_vf, domain=domain)
            # Repeat the same NumPy contract used by the two non-neural 64px
            # methods.  The torch sampler already projects, so this is normally
            # a numerical no-op; keeping it makes the recorded FEA input exact.
            neural, projection = contract_project(density01(generated))
            proposed["neural"].append((neural, seed, None, projection))

        for method, candidates in proposed.items():
            records = []
            for candidate_index, (rho64, seed, source_train_row, projection) in enumerate(candidates):
                path = None
                if args.save_candidates:
                    filename = f"row_{row:06d}.{method}.{candidate_index:02d}.density.npy"
                    path = out / "candidates" / filename
                    np.save(path, rho64.astype(np.float32))
                    path = str(path.relative_to(out))
                record = {
                    "method": method, "candidate_index": candidate_index,
                    "seed": seed, "source_train_row": source_train_row,
                    # Kept as ``density64`` for the existing topology-index
                    # call-site.  It is native-resolution only for the explicit
                    # reference ceiling; all other methods are 64px.
                    "density64": rho64, "density_path": path, "stem": stem,
                    "density_resolution": int(rho64.shape[0]),
                    "representation_contract": (
                        "native_source_ceiling_unprojected" if projection is None
                        else "64px_fractional_coverage_weighted_volume_projection"
                    ),
                    "volume_projection": projection,
                }
                records.append(record)
                all_candidates.append(record)
            spec["methods"][method] = {"candidates": records}
        specs.append(spec)
        print(f"[eval] sampled {ordinal + 1}/{len(selected_rows)} row={row} type={meta.get('problem_type')}", flush=True)

    tasks = [(c["stem"], c["density64"], str(args.protocol)) for c in all_candidates]
    if args.workers > 1:
        # Forking after ROCm/CUDA model sampling is unsafe.  The worker target
        # is deliberately Torch-free and uses a fresh spawned interpreter.
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            reports = list(pool.map(score_gate_task, tasks, chunksize=1))
    else:
        reports = [score_gate_task(task) for task in tasks]
    for candidate, report in zip(all_candidates, reports):
        candidate["gate"] = report

    native_reports = [
        spec["methods"]["reference_native"]["candidates"][0]["gate"]
        for spec in specs
    ]
    native_functional = [bool(report.get("functional_passed", False)) for report in native_reports]
    native_accessible = [
        bool(report.get("interface_passed", False))
        for spec, report in zip(specs, native_reports) if spec["reference_accessible"]
    ]
    protocol_self_check = {
        "functional_native_reference_pass_rate": float(np.mean(native_functional)) if native_functional else None,
        "functional_native_reference_n": len(native_functional),
        "accessible_native_reference_strict_pass_rate": (float(np.mean(native_accessible))
                                                        if native_accessible else None),
        "accessible_native_reference_strict_n": len(native_accessible),
        "passed": bool(native_functional and all(native_functional)),
        "requirement": "all selected native references must pass functional protocol before model comparisons are interpreted",
    }
    if not protocol_self_check["passed"]:
        with (out / "protocol_self_check.json").open("w") as f:
            json.dump(protocol_self_check, f, indent=2, default=_json_default)
        if not args.allow_reference_protocol_mismatch:
            raise SystemExit("native corpus references failed the frozen functional protocol; "
                             "see protocol_self_check.json and resolve the contract mismatch before reporting model metrics")

    for spec in specs:
        for method, trial in spec["methods"].items():
            candidates = trial["candidates"]
            trial["diversity"] = {
                "functional": candidate_diversity(candidates, "functional_passed"),
                "interface": candidate_diversity(candidates, "interface_passed"),
            }
            trial["selected_functional"] = enrich_selected(select_candidate(candidates, strict=False),
                                                            json.load(open(spec["stem"] + ".json")), topology)
            trial["selected_interface"] = enrich_selected(select_candidate(candidates, strict=True),
                                                           json.load(open(spec["stem"] + ".json")), topology)
            # Density arrays are retained only until the selected metrics above.
            for candidate in candidates:
                candidate.pop("density64", None)

    cohorts = {
        "predeclared_stratified_heldout_subset": specs,
        "predeclared_stratified_reference_interface_accessible_subset": [
            s for s in specs if s["reference_accessible"]],
    }
    summary = {}
    paired = {}
    for cohort, rows in cohorts.items():
        strict = cohort == "predeclared_stratified_reference_interface_accessible_subset"
        summary[cohort] = {method: method_summary(rows, method, strict, args.bootstrap,
                                                   args.seed_base + 101 * idx)
                           for idx, method in enumerate(METHOD_ORDER)} if rows else {}
        if rows:
            paired[cohort] = {}
            # Do not form a neural-minus-retrieval novelty contrast: retrieval
            # is literally sampled from the train set, hence Dice=1 by design.
            for key in ("functional_pass_at_k", "interface_pass_at_k",
                        "motion_class_match", "vf_abs_error", "ga_signed",
                        "output_alignment", "output_selectivity",
                        "passing_binary_unique_topologies",
                        "passing_pairwise_topology_distance"):
                a = []
                b = []
                for row in rows:
                    for method, bucket in (("neural", a), ("retrieval", b)):
                        trial = row["methods"][method]
                        candidates = trial["candidates"]
                        selected = trial["selected_interface" if strict else "selected_functional"]
                        if key in ("functional_pass_at_k", "interface_pass_at_k"):
                            pass_key = ("functional_passed" if key == "functional_pass_at_k"
                                        else "interface_passed")
                            bucket.append(any(c["gate"].get(pass_key, False) for c in candidates))
                        else:
                            if key.startswith("passing_"):
                                diversity = trial["diversity"]["interface" if strict else "functional"]
                                bucket.append(diversity.get(key))
                            else:
                                pass_key = "interface_passed" if strict else "functional_passed"
                                bucket.append(selected.get(key) if selected.get(pass_key, False) else None)
                paired[cohort][key] = paired_bootstrap(a, b, args.bootstrap, args.seed_base + 700)

    provenance = {
        "checkpoint": str(Path(args.ckpt).resolve()), "checkpoint_sha256": sha256_file(args.ckpt),
        "checkpoint_step": checkpoint.get("step"), "checkpoint_args": ck_args,
        "cache": str(Path(args.cache).resolve()), "cache_index_sha256": train.index_sha256,
        "cache_file_sha256": cache_file_hashes(args.cache),
        "git_head": git_head(), "git_status_porcelain": git_status(),
        "implementation_sha256": implementation_hashes(), "device": args.device,
        "candidate_seed_rule": "seed_base + 1000003*cache_row + 97*candidate_index modulo 2^31-1",
        "candidate_seed_base": args.seed_base, "cfg_scale": args.cfg, "flow_steps": args.steps,
        "64px_representation_contract": {
            "methods": ["reference_64_contract_projection", "retrieval", "neural"],
            "domain": "cond[domain_mask] is fractional source-area coverage at 64px",
            "postprocess": "zero 0-coverage cells; bisection-project weighted active-domain VF before sparse FEA",
            "native_reference": "reference_native is unprojected full-resolution ceiling/check",
        },
        "within_condition_diversity": {
            "definition": "binary threshold 0.5; pairwise Dice/topology distance among FEA-passing fixed-K candidates only",
            "purpose": "reports usable option diversity separately from nearest-train novelty",
        },
        "selection_rule": "first protocol-passing candidate in fixed order; otherwise minimum predeclared protocol violation",
        "split_resolution": {"mode": split_mode, "holdout_value": resolved_holdout,
                             "val_frac": resolved_val_frac, "seed": resolved_split_seed,
                             "plan": str(Path(args.split_plan).resolve()) if args.split_plan else None,
                             "plan_sha256": plan_sha256, "plan_scheme": plan_scheme,
                             "checkpoint_split_verified": checkpoint_split_verified,
                             "claim_bearing_test_partition": claim_bearing_test,
                             "diagnostic_provenance_problem": split_provenance_problem},
        "retrieval": {"kind": "all-train full conditioning-vector signature", "grid": 8,
                      "features": "11 pooled conditioning channels + 15 standardized scalar conditions; scalar targets include observed-reference motion/performance labels"},
    }
    payload = {
        "format": "opencompmech.verification-evaluation.v2",
        "protocol": {"name": protocol["name"], "path": protocol["_path"], "sha256": protocol["_sha256"]},
        "protocol_self_check": protocol_self_check,
        "provenance": provenance,
        "split_assessment": split_assessment(train, train.rows, np.asarray(selected_rows),
                                              split_mode, checkpoint, legacy, plan_scheme,
                                              checkpoint_split_verified, claim_bearing_test,
                                              split_name, legacy_partition_verified),
        "evaluation_sampling": evaluation_sampling,
        "n_specs": len(specs), "K": args.K,
        "summary": summary, "paired_differences": paired, "per_spec": specs,
    }
    with (out / "results.json").open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    write_markdown(out, payload)
    print(f"[eval] wrote {out / 'results.json'} and {out / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
