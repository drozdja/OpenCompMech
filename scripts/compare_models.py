#!/usr/bin/env python3
"""Score the raster (CNN) and graph (GNN) models under identical conditions.

This is the comparable evaluation: one script drives both models so that the
specifications, candidate count, seed schedule, FEA gate and reporting are the
same by construction rather than by two scripts agreeing on convention.

What is held identical
----------------------
* **Specifications** — the frozen protocol's `test` split
  (`scripts/freeze_eval_protocol.py`). Eligibility is a rule, not a hand-edit.
* **Seed schedule** — candidate *j* of row *r* uses `candidate_seed(base, r, j)`
  for both models, so neither gets a luckier noise draw.
* **Candidate count** — the same K; pass@1 and pass@8 are both reported.
* **Gate** — the same Torch-free sparse-FEA workers.

Decoder repair is decomposed, not hidden
----------------------------------------
Each pipeline is scored twice: once raw, once with its deterministic
post-processing. This separates what the *model* learned from what the *decoder*
repairs, which a single end-to-end number cannot.

    raster_raw        sample -> threshold
    raster_projected  sample -> volume projection (the raster pipeline's standard)
    graph_raw         sample -> decode -> plain rasterization
    graph_repaired    sample -> decode -> reconstruct_valid (anchors, volume,
                      domain clip, local hinge repair)

`reference_native` is carried through as the protocol self-check.

With ``--graph-null-ablation``, the same graph noise is also sampled through
the model's trained unconditional branch.  The resulting free graph is decoded
against the *target* specification's anchors, domain and volume exactly like the
conditioned sample.  This isolates learned specification-following from the
validity supplied by the representation and deterministic reconstruction.

    python scripts/compare_models.py \
        --freeze data/eval_protocol/v1_128_frozen.json \
        --cache data/v1_broad_cache_128 --graphs data/v1_graph_128 \
        --cnn-ckpt runs/v1flow_128_base96/ckpt_final.pt \
        --gnn-ckpt runs/v1graph_128_matched/ckpt_final.pt \
        --out runs/compare_v1 --K 8 --workers 8
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
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_harness import (cache_file_hashes, candidate_seed,      # noqa: E402
                                  implementation_hashes)
from scripts.eval_mechanism_gate import (canonicalize_density,            # noqa: E402
                                         load_protocol, score_gate_task)
from src.graph import reconstruct_valid, to_raster                        # noqa: E402
from src.ml.dataset import PilotCache                                     # noqa: E402
from src.ml.egnn import MechEGNN                                          # noqa: E402
from src.ml.flow import RectifiedFlow                                     # noqa: E402
from src.ml.graph_flow import sample as graph_sample                      # noqa: E402
from src.ml.graph_tensors import (EDGE_CH, N_ANCHOR, N_FREE, N_MAX,       # noqa: E402
                                  NODE_CH, decode)
from src.ml.guided_sample import guided_sample, project_volume_numpy      # noqa: E402
from src.ml.tensor_spec import COND_CHANNELS, COND_DIM, SCALAR_DIM        # noqa: E402
from src.ml.unet import ConditionalUNet                                   # noqa: E402
from src.validation.connectivity import check_volume_fraction             # noqa: E402

METHODS = ["reference_native", "raster_raw", "raster_projected",
           "graph_raw", "graph_repaired", "graph_null_raw",
           "graph_null_repaired"]


def anchor_roles_tensor(b, device):
    r = torch.zeros(N_ANCHOR, 3)
    r[0, 0] = r[1, 1] = 1.0
    r[2, 2] = r[3, 2] = 1.0
    return r.unsqueeze(0).expand(b, -1, -1).to(device)


def gate_vf_fn(meta):
    """Volume fraction measured the way the verifier measures it."""
    from scripts.backfill_conditioning import problem_from_metadata
    dom = np.asarray(problem_from_metadata(meta).domain_mask, bool)
    t = float(meta["volume_fraction_target"])

    def f(mask_img):
        rho, _ = canonicalize_density(meta, np.asarray(mask_img, np.float32))
        return check_volume_fraction(rho, t, 0.05, dom)[1]
    return f


def bootstrap_ci(flags, reps=2000, seed=0):
    """Percentile bootstrap CI over specs (the unit of independence)."""
    a = np.asarray(flags, dtype=float)
    if a.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(reps, a.size))
    means = a[idx].mean(axis=1)
    return {"mean": float(a.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True, help="frozen protocol artifact")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--cnn-ckpt", required=True)
    ap.add_argument("--gnn-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["test", "tuning"])
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--protocol", default=None)
    ap.add_argument(
        "--graph-null-ablation", action="store_true",
        help=("also sample the GNN's unconditional branch with identical noise; "
              "decode it using the target spec's anchors/domain/volume"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.protocol or str(
        Path(__file__).resolve().parent.parent / "config"
        / "evaluation_protocol.v1.json")
    protocol = load_protocol(protocol_path)

    with open(args.freeze) as f:
        freeze = json.load(f)
    if freeze.get("protocol_sha256") != protocol.get("_sha256"):
        raise SystemExit("frozen protocol was built against a different gate protocol")
    rows = [s["row"] for s in freeze["specs"] if s["split"] == args.split]
    if args.limit and args.limit < len(rows):
        # even stride, not a prefix: rows are sorted by index, so a prefix is
        # dominated by whichever families occupy the low indices.
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, args.limit).astype(int)]
    print(f"[cmp] {len(rows)} specs from frozen '{args.split}' split, K={args.K}",
          flush=True)

    cache = PilotCache(args.cache, split="all")
    target = np.load(os.path.join(args.cache, "target.f16"), mmap_mode="r")
    cond_mm = np.load(os.path.join(args.cache, "cond.f16"), mmap_mode="r")
    scal_mm = np.load(os.path.join(args.cache, "scalars.f32"), mmap_mode="r")
    shape = target.shape[-2:]
    dom_ch = COND_CHANNELS.index("domain_mask")

    z = np.load(os.path.join(args.graphs, "graphs.npz"))
    grow = {int(r): i for i, r in enumerate(z["rows"])}

    # ---- models ----
    ck_c = torch.load(args.cnn_ckpt, map_location=args.device, weights_only=False)
    cnn = ConditionalUNet(1, COND_DIM, SCALAR_DIM,
                          base=int(ck_c.get("args", {}).get("base", 64))).to(args.device)
    cnn.load_state_dict(ck_c["ema"])
    cnn.eval()
    obj = RectifiedFlow(device=args.device)

    ck_g = torch.load(args.gnn_ckpt, map_location=args.device, weights_only=False)
    ga = ck_g.get("args", {})
    gnn = MechEGNN(n_anchor=N_ANCHOR, node_ch=NODE_CH, edge_ch=EDGE_CH,
                   scalar_dim=SCALAR_DIM, hidden=int(ga.get("hidden", 128)),
                   layers=int(ga.get("layers", 6))).to(args.device)
    gnn.load_state_dict(ck_g["ema"])
    gnn.eval()

    exposure = {
        "cnn": {"steps": ck_c.get("args", {}).get("steps"),
                "batch": ck_c.get("args", {}).get("batch"),
                "params_M": sum(p.numel() for p in cnn.parameters()) / 1e6},
        "gnn": {"steps": ga.get("steps"), "batch": ga.get("batch"),
                "params_M": sum(p.numel() for p in gnn.parameters()) / 1e6},
    }
    for m in exposure.values():
        if m["steps"] and m["batch"]:
            m["examples_M"] = m["steps"] * m["batch"] / 1e6
    print(f"[cmp] exposure: {json.dumps(exposure)}", flush=True)

    # ---- generate ----
    tasks, keys, specs = [], [], []
    t0 = time.time()
    for ordinal, row in enumerate(rows):
        stem = cache.stem_for_row(row)
        with open(stem + ".json") as f:
            meta = json.load(f)
        cond = torch.from_numpy(np.array(cond_mm[row], np.float32))[None].to(args.device)
        scal = torch.from_numpy(np.array(scal_mm[row], np.float32))[None].to(args.device)
        dom = np.array(cond_mm[row, dom_ch], np.float32) > 0.5
        # guided_sample projects volume on-device and requires (N,1,H,W) coverage
        dom_t = torch.from_numpy(dom.astype(np.float32))[None, None].to(args.device)
        target_vf = float(scal_mm[row][0])
        vfn = gate_vf_fn(meta)
        rec = {"row": row, "stem": stem,
               "type": str(cache.record_for_row(row).get("type")),
               "family": str(cache.record_for_row(row).get("family"))}
        specs.append(rec)

        def push(method, k, dens):
            tasks.append((stem, np.asarray(dens, np.float32), protocol_path))
            keys.append((row, method, k))

        push("reference_native", 0, (np.asarray(target[row, 0], np.float32) + 1.0) * 0.5)

        gi = grow.get(row)
        for k in range(args.K):
            seed = candidate_seed(args.seed_base, row, k)

            # --- raster ---
            torch.manual_seed(seed)
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                g_raw = guided_sample(cnn, obj, cond, scal, None, None, None, None,
                                      0.0, args.steps, args.device, cfg_scale=args.cfg,
                                      target_vf=None, domain=None)
            g_raw_f = g_raw.float().cpu().numpy()[0, 0]
            r_raw = ((np.asarray(g_raw_f) + 1.0) * 0.5) > 0.5
            push("raster_raw", k, r_raw & dom)

            # Same sample, post-processed: isolates the projection as a decoder
            # step, exactly parallel to the graph pipeline's reconstruct_valid.
            rho = (np.asarray(g_raw_f) + 1.0) * 0.5
            push("raster_projected", k,
                 project_volume_numpy(rho, target_vf, dom.astype(np.float32)) > 0.5)

            # --- graph (same seed) ---
            if gi is None:
                continue
            gen = torch.Generator(device=args.device).manual_seed(seed)
            with torch.no_grad():
                nx, ex = graph_sample(
                    gnn,
                    torch.from_numpy(np.array(z["scalars"][gi])[None]).to(args.device),
                    torch.from_numpy(np.array(z["anchor_pos"][gi])[None]).to(args.device),
                    torch.from_numpy(np.array(z["anchor_vec"][gi])[None]).to(args.device),
                    torch.from_numpy(np.array(z["anchor_present"][gi])[None]).to(args.device),
                    anchor_roles_tensor(1, args.device),
                    N_FREE, N_MAX, NODE_CH, EDGE_CH,
                    steps=args.steps, cfg=args.cfg, generator=gen)
            gk = decode(nx.float().cpu().numpy()[0], ex.float().cpu().numpy()[0],
                        z["anchor_pos"][gi], z["anchor_present"][gi], shape=shape)
            push("graph_raw", k, to_raster(gk, shape=shape) & dom)
            push("graph_repaired", k,
                 reconstruct_valid(gk, target_vf=target_vf, domain_mask=dom,
                                   shape=shape, vf_fn=lambda m: vfn(m & dom)))

            if args.graph_null_ablation:
                # Reset to the same seed so conditioned and null samples start
                # from identical node/edge noise. Null conditioning is the
                # branch explicitly trained by classifier-free dropout.
                null_gen = torch.Generator(device=args.device).manual_seed(seed)
                graph_scal = torch.from_numpy(
                    np.array(z["scalars"][gi])[None]).to(args.device)
                graph_apos = torch.from_numpy(
                    np.array(z["anchor_pos"][gi])[None]).to(args.device)
                graph_avec = torch.from_numpy(
                    np.array(z["anchor_vec"][gi])[None]).to(args.device)
                graph_apres = torch.from_numpy(
                    np.array(z["anchor_present"][gi])[None]).to(args.device)
                with torch.no_grad():
                    nnx, nex = graph_sample(
                        gnn, torch.zeros_like(graph_scal),
                        torch.zeros_like(graph_apos), torch.zeros_like(graph_avec),
                        torch.zeros_like(graph_apres),
                        anchor_roles_tensor(1, args.device),
                        N_FREE, N_MAX, NODE_CH, EDGE_CH,
                        steps=args.steps, cfg=1.0, generator=null_gen)
                ng = decode(nnx.float().cpu().numpy()[0],
                            nex.float().cpu().numpy()[0],
                            z["anchor_pos"][gi], z["anchor_present"][gi],
                            shape=shape)
                push("graph_null_raw", k, to_raster(ng, shape=shape) & dom)
                push("graph_null_repaired", k,
                     reconstruct_valid(
                         ng, target_vf=target_vf, domain_mask=dom, shape=shape,
                         vf_fn=lambda m: vfn(m & dom)))

        if (ordinal + 1) % 20 == 0:
            print(f"[cmp] generated {ordinal+1}/{len(rows)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- gate ----
    print(f"[cmp] gating {len(tasks)} candidates on {args.workers} workers...",
          flush=True)
    tg = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            reports = list(pool.map(score_gate_task, tasks, chunksize=1))
    else:
        reports = [score_gate_task(t) for t in tasks]
    print(f"[cmp] gated in {(time.time()-tg)/60:.1f} min", flush=True)

    # ---- aggregate ----
    passed = defaultdict(dict)          # method -> row -> [bool per k]
    reasons = defaultdict(Counter)
    for (row, method, k), rep in zip(keys, reports):
        passed[method].setdefault(row, []).append(bool(rep.get("passed", False)))
        if not rep.get("passed", False):
            for r in rep.get("failure_reasons", []):
                reasons[method][r] += 1

    by_type = {s["row"]: s["type"] for s in specs}
    summary = {}
    for method in METHODS:
        if method not in passed:
            continue
        rws = sorted(passed[method])
        at1 = [passed[method][r][0] for r in rws]
        at8 = [any(passed[method][r]) for r in rws]
        cand = [v for r in rws for v in passed[method][r]]
        per_type = {}
        for t in sorted(set(by_type[r] for r in rws)):
            sel = [any(passed[method][r]) for r in rws if by_type[r] == t]
            per_type[t] = {"pass_at_k": float(np.mean(sel)), "n": len(sel)}
        summary[method] = {
            "pass_at_1": bootstrap_ci(at1, seed=args.seed_base),
            f"pass_at_{args.K}": bootstrap_ci(at8, seed=args.seed_base),
            "candidate_pass_rate": float(np.mean(cand)) if cand else None,
            "n_specs": len(rws), "per_type": per_type,
            "top_failures": reasons[method].most_common(5),
        }

    payload = {
        "format": "opencompmech.model-comparison.v1",
        "args": vars(args),
        "frozen_protocol": {"path": os.path.abspath(args.freeze),
                            "split": args.split,
                            "eligibility_rule": freeze.get("eligibility_rule"),
                            "cache_index_sha256": freeze.get("cache_index_sha256")},
        "protocol_sha256": protocol.get("_sha256"),
        "cache_file_sha256": cache_file_hashes(args.cache),
        "implementation_sha256": implementation_hashes(),
        "training_exposure": exposure,
        "summary": summary,
        "specs": specs,
        "outcomes": {
            str(row): {method: [bool(v) for v in values]
                       for method, rows_by_method in passed.items()
                       if (values := rows_by_method.get(row)) is not None}
            for row in sorted({s["row"] for s in specs})
        },
    }
    with (out / "comparison.json").open("w") as f:
        json.dump(payload, f, indent=2)

    # ---- report ----
    nat = summary.get("reference_native", {}).get("pass_at_1", {}).get("mean")
    print(f"\n=== model comparison (n={len(rows)} specs, K={args.K}, "
          f"frozen '{args.split}') ===")
    print(f"{'method':<20}{'pass@1':>22}{f'pass@{args.K}':>22}{'cand rate':>11}")
    for m in METHODS:
        if m not in summary:
            continue
        s = summary[m]
        a1, a8 = s["pass_at_1"], s[f"pass_at_{args.K}"]
        f1 = f"{a1['mean']:.3f} [{a1['lo']:.3f},{a1['hi']:.3f}]"
        f8 = f"{a8['mean']:.3f} [{a8['lo']:.3f},{a8['hi']:.3f}]"
        print(f"{m:<20}{f1:>22}{f8:>22}{s['candidate_pass_rate']:>11.3f}")
    print(f"\n  95% CIs are percentile bootstrap over specs.")
    print(f"  protocol self-check (native pass@1): {nat:.3f}"
          if nat is not None else "")
    if nat is not None and nat < 1.0:
        print("  NOTE: natives below 1.000 — the frozen protocol should have "
              "excluded these; investigate before quoting.")
    print(f"\n  wrote {out/'comparison.json'}")


if __name__ == "__main__":
    main()
