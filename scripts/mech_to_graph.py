#!/usr/bin/env python3
"""CLI for the standardized mechanism graph representation (``src/graph``).

Runs the uniform raster->graph converter over N designs of each family from a
tensor cache and reports round-trip Dice, graph size, and (with --arrays) the
EGNN-ready tensor shapes -- demonstrating one schema for every family.

    python scripts/mech_to_graph.py --cache data/v1_broad_cache_128 \
        --per-family 25 --dump runs/graph_export --arrays

Requires: scikit-image, sknw, scipy, numpy (torch optional, for --arrays tensors).
"""
import argparse
import collections
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph import from_raster, roundtrip_dice, to_arrays, MechGraph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/v1_broad_cache_128")
    ap.add_argument("--per-family", type=int, default=25)
    ap.add_argument("--dump", default=None,
                    help="write one sample graph JSON per family here")
    ap.add_argument("--arrays", action="store_true",
                    help="also report EGNN-ready tensor shapes + serialization check")
    args = ap.parse_args()

    from src.ml.dataset import PilotCache
    ds = PilotCache(args.cache, split="all", split_mode="lineage",
                    val_frac=0.04, seed=0)
    recs = ds.manifest["index"]
    chn = ds.manifest["cond_channels"]
    scal_names = ds.manifest.get("scalar_names")

    byfam = collections.defaultdict(list)
    for i, r in enumerate(recs):
        byfam[r["family"]].append(i)
    rng = random.Random(0)

    print(f"{'family':7} {'n':>4} {'roundtrip_dice':>15} {'nodes':>6} {'edges':>6} "
          f"{'ports_ok':>9}", flush=True)
    for fam in sorted(byfam):
        rows = byfam[fam][:]
        rng.shuffle(rows)
        rows = rows[:args.per_family]
        dices, nn, ne, ports = [], [], [], []
        first = None
        for i in rows:
            item = ds[i]
            dens = item["target"][0].numpy()
            g = from_raster(dens, cond=item["cond"].numpy(), cond_channels=chn,
                            scalars=item["scalars"].numpy(), scalar_names=scal_names,
                            meta={"family": recs[i]["family"], "type": recs[i]["type"],
                                  "stem": recs[i]["stem"]})
            if not g.nodes:
                continue
            dices.append(roundtrip_dice(dens, g))
            nn.append(g.n_nodes())
            ne.append(g.n_edges())
            has_ports = any("input" in n.roles for n in g.nodes) and \
                any("output" in n.roles for n in g.nodes)
            ports.append(float(has_ports))
            if first is None:
                first = (recs[i]["type"], dens, g)
        print(f"{fam:7} {len(dices):4d} {np.mean(dices):15.3f} "
              f"{np.mean(nn):6.1f} {np.mean(ne):6.1f} {np.mean(ports):9.2f}",
              flush=True)
        if args.dump and first:
            os.makedirs(args.dump, exist_ok=True)
            typ, _, g = first
            g.save(os.path.join(args.dump, f"graph_{fam}_{typ}.json"))
        if args.arrays and first:
            typ, dens, g = first
            a = to_arrays(g)
            # serialization round-trip check
            g2 = MechGraph.from_dict(g.to_dict())
            ok = (g2.n_nodes() == g.n_nodes() and g2.n_edges() == g.n_edges())
            print(f"        [{fam}] arrays: pos{a['pos'].shape} "
                  f"node_scalar{a['node_scalar'].shape} node_vec{a['node_vec'].shape} "
                  f"edge_index{a['edge_index'].shape} edge_scalar{a['edge_scalar'].shape} "
                  f"graph_y{a['graph_y'].shape}  serialize_roundtrip={ok}", flush=True)


if __name__ == "__main__":
    main()
