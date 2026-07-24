#!/usr/bin/env python3
"""pass@K for the graph generator, through the same sparse-FEA gate.

Verification-first, exactly like ``scripts/eval_harness.py``: a candidate counts
only if a *fresh* FEA solve says the mechanism works.  Nothing here compares a
sample to the target raster.

Three methods are reported, and the middle one matters as much as the model:

``reference_native``
    the source design, straight through the gate.  If these do not all pass, the
    protocol is broken and the numbers below are meaningless — the run aborts.
``reference_graph_roundtrip``
    the *ground-truth* design converted to a graph and rebuilt with
    ``to_raster``.  This is the **representation ceiling**: no graph generator
    can beat it, because it measures what survives the encoding itself.  The
    raster experiment showed a ceiling like this was the real bottleneck at
    64px, so it is measured here before the model is judged.
``neural``
    K graphs sampled per spec, decoded, rasterized, gated.

    python scripts/eval_graph.py --graphs data/v1_graph_128 \
        --cache data/v1_broad_cache_128 --ckpt runs/v1graph_128/ckpt_final.pt \
        --out runs/v1graph_128_eval --n-specs 60 --K 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_mechanism_gate import (_json_default, canonicalize_density,  # noqa: E402
                                         load_protocol, score_gate_task)
from scripts.eval_harness import select_eval_rows                       # noqa: E402
from src.graph import reconstruct_valid                                 # noqa: E402
from src.ml.dataset import PilotCache                                   # noqa: E402
from src.ml.egnn import MechEGNN                                        # noqa: E402
from src.ml.graph_flow import sample                                    # noqa: E402
from src.ml.graph_tensors import (EDGE_CH, N_ANCHOR, N_FREE, N_MAX,     # noqa: E402
                                  NODE_CH, decode)
from src.ml.tensor_spec import COND_CHANNELS, SCALAR_DIM                # noqa: E402


def anchor_roles_tensor(b, device):
    r = torch.zeros(N_ANCHOR, 3)
    r[0, 0] = r[1, 1] = 1.0
    r[2, 2] = r[3, 2] = 1.0
    return r.unsqueeze(0).expand(b, -1, -1).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-specs", type=int, default=60)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--split-mode", default="lineage")
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--allow-legacy-heldout-audit", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    protocol_path = args.protocol or str(
        Path(__file__).resolve().parent.parent / "config"
        / "evaluation_protocol.v1.json")
    load_protocol(protocol_path)

    cache = PilotCache(args.cache, split=args.split, split_mode=args.split_mode,
                       val_frac=args.val_frac, seed=args.split_seed)
    rows, alloc = select_eval_rows(cache, cache.rows, args.n_specs, args.seed)

    z = np.load(os.path.join(args.graphs, "graphs.npz"))
    row2i = {int(r): i for i, r in enumerate(z["rows"])}
    rows = [r for r in rows if int(r) in row2i]
    if not rows:
        raise SystemExit("no eval rows have a converted graph")
    print(f"[evalg] {len(rows)} specs, K={args.K}", flush=True)

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cargs = ck.get("args", {})
    model = MechEGNN(n_anchor=N_ANCHOR, node_ch=NODE_CH, edge_ch=EDGE_CH,
                     scalar_dim=SCALAR_DIM,
                     hidden=int(cargs.get("hidden", 128)),
                     layers=int(cargs.get("layers", 6))).to(args.device)
    model.load_state_dict(ck["ema"])
    model.eval()

    target = np.load(os.path.join(args.cache, "target.f16"), mmap_mode="r")
    cond_mm = np.load(os.path.join(args.cache, "cond.f16"), mmap_mode="r")
    shape = target.shape[-2:]
    dom_ch = COND_CHANNELS.index("domain_mask")

    def _gate_vf(meta):
        from scripts.backfill_conditioning import problem_from_metadata
        from src.validation.connectivity import check_volume_fraction
        dp = np.asarray(problem_from_metadata(meta).domain_mask, bool)
        t = float(meta["volume_fraction_target"])
        return lambda m: check_volume_fraction(
            canonicalize_density(meta, m.astype(np.float32))[0], t, 0.05, dp)[1]

    def rasterize(g, row, densify_seg=12.0):
        """Graph -> an erosion-robust density that can survive the FEA gate.

        The plain reconstruction pinches to ~1px at junctions and is rejected by
        the hinge check; reconstruct_valid floors member widths, projects to the
        spec's volume, clips to the domain and repairs single-pixel hinges. This
        is the ground-truth ceiling's own decoder, so the model is scored on the
        same footing as its ceiling (see docs/GNN_APPROACH.md).
        """
        with open(cache.stem_for_row(int(row)) + ".json") as f:
            meta = json.load(f)
        vf = float(meta["volume_fraction_target"])
        dom = np.asarray(cond_mm[int(row), dom_ch], np.float32) > 0.5
        vfn = _gate_vf(meta)
        return reconstruct_valid(g, target_vf=vf, domain_mask=dom, shape=shape,
                                 densify_seg=densify_seg,
                                 vf_fn=lambda m: vfn(m & dom)).astype(np.float32)

    all_c, specs = [], []
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    for ordinal, row in enumerate(rows):
        i = row2i[int(row)]
        stem = cache.stem_for_row(int(row))
        spec = {"row": int(row), "stem": stem, "methods": {}}

        # 1. native reference (protocol self-check)
        native = np.asarray(target[int(row), 0], np.float32)
        native01 = (native + 1.0) * 0.5
        rec = {"method": "reference_native", "candidate_index": 0,
               "stem": stem, "density": native01}
        spec["methods"]["reference_native"] = {"candidates": [rec]}
        all_c.append(rec)

        # 2. representation ceiling: GT graph -> raster
        g = decode(z["node_x"][i].astype(np.float32),
                   z["edge_x"][i].astype(np.float32),
                   z["anchor_pos"][i], z["anchor_present"][i], shape=shape)
        rec = {"method": "reference_graph_roundtrip", "candidate_index": 0,
               "stem": stem, "density": rasterize(g, row)}
        spec["methods"]["reference_graph_roundtrip"] = {"candidates": [rec]}
        all_c.append(rec)

        # 3. neural: K samples
        b = args.K
        scal = torch.from_numpy(np.repeat(z["scalars"][i][None], b, 0)).to(args.device)
        apos = torch.from_numpy(np.repeat(z["anchor_pos"][i][None], b, 0)).to(args.device)
        avec = torch.from_numpy(np.repeat(z["anchor_vec"][i][None], b, 0)).to(args.device)
        apres = torch.from_numpy(np.repeat(z["anchor_present"][i][None], b, 0)).to(args.device)
        with torch.no_grad():
            nx, ex = sample(model, scal, apos, avec, apres,
                            anchor_roles_tensor(b, args.device),
                            N_FREE, N_MAX, NODE_CH, EDGE_CH,
                            steps=args.steps, cfg=args.cfg, generator=gen)
        nx = nx.float().cpu().numpy()
        ex = ex.float().cpu().numpy()
        cands = []
        for k in range(b):
            gk = decode(nx[k], ex[k], z["anchor_pos"][i], z["anchor_present"][i],
                        shape=shape)
            rec = {"method": "neural", "candidate_index": k, "stem": stem,
                   "density": rasterize(gk, row),
                   "n_nodes": gk.n_nodes(), "n_edges": gk.n_edges()}
            cands.append(rec)
            all_c.append(rec)
        spec["methods"]["neural"] = {"candidates": cands}
        specs.append(spec)
        if (ordinal + 1) % 10 == 0:
            print(f"[evalg] sampled {ordinal+1}/{len(rows)}", flush=True)

    tasks = [(c["stem"], c["density"], protocol_path) for c in all_c]
    print(f"[evalg] gating {len(tasks)} candidates...", flush=True)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            reports = list(pool.map(score_gate_task, tasks, chunksize=1))
    else:
        reports = [score_gate_task(t) for t in tasks]
    for c, r in zip(all_c, reports):
        c["gate"] = r
        c.pop("density", None)

    native_ok = [bool(s["methods"]["reference_native"]["candidates"][0]["gate"]
                      .get("functional_passed", False)) for s in specs]
    self_check = {"native_reference_pass_rate": float(np.mean(native_ok)),
                  "n": len(native_ok), "passed": bool(all(native_ok))}

    summary = {}
    for method in ("reference_native", "reference_graph_roundtrip", "neural"):
        hits = []
        for s in specs:
            cs = s["methods"][method]["candidates"]
            hits.append(any(bool(c["gate"].get("passed", False)) for c in cs))
        summary[method] = {"pass_at_k": float(np.mean(hits)), "n": len(hits),
                           "K": args.K if method == "neural" else 1}
    nn_ = [c["n_nodes"] for c in all_c if c["method"] == "neural"]
    ne_ = [c["n_edges"] for c in all_c if c["method"] == "neural"]
    summary["neural"]["mean_nodes"] = float(np.mean(nn_)) if nn_ else 0.0
    summary["neural"]["mean_edges"] = float(np.mean(ne_)) if ne_ else 0.0

    payload = {"args": vars(args), "allocation": alloc,
               "protocol_self_check": self_check, "summary": summary,
               "specs": specs}
    with (out / "results.json").open("w") as f:
        json.dump(payload, f, default=_json_default, indent=2)

    print("\n=== graph pass@K ===")
    for m, v in summary.items():
        extra = (f"  nodes {v['mean_nodes']:.1f} edges {v['mean_edges']:.1f}"
                 if m == "neural" else "")
        print(f"  {m:<28} {v['pass_at_k']:.3f}  (n={v['n']}, K={v['K']}){extra}")
    if not self_check["passed"]:
        print(f"\n!! protocol self-check FAILED "
              f"({self_check['native_reference_pass_rate']:.3f}) — "
              f"model numbers are not interpretable", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
