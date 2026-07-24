#!/usr/bin/env python3
"""Render evaluation montages from saved pilot samples (no re-solving).

Loads density (design res) + displacement (fine node grid) + metadata from a
pilot output directory and draws the same deformed-overlay panels as
audit_mech.py, one montage PNG per problem type. Panels are randomly sampled
per type (seeded, reproducible).

Usage:
    python scripts/pilot_montages.py --dir data/pilot_v0 --per-type 24
"""

import argparse
import json
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw_panel(ax, density, ux, uy, meta):
    nely, nelx = density.shape
    ax.imshow(density, cmap="Greys", origin="lower",
              extent=[0, nelx, 0, nely], alpha=0.18, vmin=0, vmax=1)

    solid = density > 0.5
    cx, cy = np.meshgrid(np.arange(nelx) + 0.5, np.arange(nely) + 0.5)
    uxc = 0.25 * (ux[:-1, :-1] + ux[1:, :-1] + ux[:-1, 1:] + ux[1:, 1:])
    uyc = 0.25 * (uy[:-1, :-1] + uy[1:, :-1] + uy[:-1, 1:] + uy[1:, 1:])
    mag = max(np.abs(ux).max(), np.abs(uy).max()) + 1e-12
    scale = 0.20 * max(nelx, nely) / mag
    ax.scatter((cx + scale * uxc)[solid], (cy + scale * uyc)[solid],
               s=2, c="steelblue", linewidths=0)
    # true deformation magnification (panels are normalized to ~20% domain)
    ax.text(0.02, 0.02,
            f"×{scale:.1f}" if scale < 1.5 else f"×{scale:.0f}",
            transform=ax.transAxes,
            fontsize=7, color="dimgray")

    mech = meta["mechanism"]
    n = nelx + 1
    # fixtures (user request 2026-07-16): BC node patches, stored in
    # metadata since 2026-07-17 (older samples: backfill_conditioning.py)
    for bc in meta.get("boundary_conditions") or []:
        nodes = np.asarray(bc["nodes"], dtype=int)
        ax.scatter(nodes % n, nodes // n, s=10, c="black", marker="^",
                   zorder=6, linewidths=0)
    for node, dvec, color, marker in (
            (mech["input_node"], mech["input_direction"], "red", "o"),
            (mech["output_node"], mech["output_direction"], "green", "s")):
        px, py = node % n, node // n
        ax.annotate("", xy=(px + 4 * dvec[0], py + 4 * dvec[1]),
                    xytext=(px, py),
                    arrowprops=dict(color=color, width=1.5, headwidth=6))
        ax.scatter([px], [py], c=color, s=25, marker=marker, zorder=7)

    q = meta["validation"]["quality"]
    ax.set_title(
        f"#{meta['sample_id']}  u_out={meta['optimization']['final_objective']:.1f} "
        f"GA={q['ga']:.2f}\ngini={q['gini']:.2f} offax={q['off_axis']:.2f}",
        fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, nelx); ax.set_ylim(0, nely)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/pilot_v0")
    ap.add_argument("--per-type", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    by_type = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        if f.endswith("run.log"):
            continue
        meta = json.load(open(f))
        by_type[meta["problem_type"]].append((f, meta))

    for ptype, items in sorted(by_type.items()):
        picks = [items[i] for i in
                 rng.choice(len(items), size=min(args.per_type, len(items)),
                            replace=False)]
        ncols = len(picks)
        fig, axes = plt.subplots(1, ncols, figsize=(2.2 * ncols, 2.6))
        if ncols == 1:
            axes = [axes]
        for ax, (f, meta) in zip(axes, picks):
            stem = f[:-5]
            density = np.load(stem + ".density.npy").astype(np.float64)
            u = np.load(stem + ".displacement.npy").astype(np.float64)
            # displacement is on the FINE node grid (2x refinement):
            # subsample every 2nd node back to the design-resolution node grid.
            step = (u.shape[1] - 1) // density.shape[1]
            ux, uy = u[0][::step, ::step], u[1][::step, ::step]
            draw_panel(ax, density, ux, uy, meta)
        run_name = os.path.basename(os.path.normpath(args.dir)) or "pilot"
        fig.suptitle(f"{run_name} — {ptype} ({len(items)} valid, showing {ncols})",
                     fontsize=10)
        out = os.path.join(args.dir, f"montage_{ptype}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
