#!/usr/bin/env python3
"""Self-contained tour of the graph representation. No corpus, no GPU, no model.

Builds a synthetic mechanism-like shape, then exercises the pieces that are
reusable on their own:

    raster -> canonical graph -> ETNN rank-2 lift -> back to a raster

and checks the rank-2 lift against Euler's formula, which is threshold-free and
independent of the implementation.

    python examples/quickstart.py [--save-figure docs/img/quickstart.png]

Requires numpy, scipy, scikit-image and sknw (see requirements.txt).
Matplotlib is needed only for --save-figure.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import (build_cells, euler_check, from_raster,      # noqa: E402
                       reconstruct_valid, roundtrip_dice, to_arrays, to_raster)


def synthetic_mechanism(n: int = 128) -> np.ndarray:
    """A lever-like shape containing one closed loop, drawn directly as pixels.

    Deliberately crude: the point is that the converter takes *any* density and
    does not care how it was produced. The loop is what the rank-2 lift detects.
    """
    img = np.zeros((n, n), np.float32)

    def bar(y0, y1, x0, x1):
        img[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 1.0

    # A rectangular ring: four walls enclosing a large void.
    bar(24, 34, 24, 104)          # top wall
    bar(74, 84, 24, 104)          # bottom wall
    bar(24, 84, 24, 34)           # left wall
    bar(24, 84, 94, 104)          # right wall
    # Legs hanging off the ring (open branches, not part of the loop).
    bar(84, 112, 26, 36)          # output leg
    bar(84, 112, 92, 102)         # support leg
    bar(48, 58, 104, 120)         # input stub
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--save-figure", default=None)
    args = ap.parse_args()

    img = synthetic_mechanism(args.size)
    print(f"input raster: {img.shape}, material fraction {img.mean():.3f}")

    # 1. raster -> canonical graph (same code path for every generator family)
    g = from_raster(img)
    print(f"\ngraph: {g.n_nodes()} nodes, {g.n_edges()} edges")
    for nd in g.nodes[:5]:
        print(f"   node {nd.id}: x={nd.x:6.1f} y={nd.y:6.1f} r={nd.r:4.1f} "
              f"degree={nd.degree}")
    if g.n_nodes() > 5:
        print(f"   ... {g.n_nodes() - 5} more")

    # 2. rank-2 lift (ETNN combinatorial complex): holes and pads
    cells = build_cells(g)
    holes = [c for c in cells if c.kind == "hole"]
    pads = [c for c in cells if c.kind == "pad"]
    print(f"\nrank-2 cells: {len(holes)} hole(s), {len(pads)} pad(s)")

    chk = euler_check(g)
    print(f"Euler check: V={chk['V']} E={chk['E']} C={chk['C']} -> expected "
          f"{chk['expected_bounded_faces']} bounded faces, found "
          f"{chk['hole_cells']}  ->  {'OK' if chk['ok'] else 'MISMATCH'}")
    if not chk["ok"]:
        raise SystemExit("Euler check failed; the rank-2 lift is wrong")

    # 3. model-ready tensors (EGNN convention)
    a = to_arrays(g)
    print(f"\ntensors: pos {a['pos'].shape}  node_scalar {a['node_scalar'].shape}"
          f"  edge_index {a['edge_index'].shape}  edge_scalar {a['edge_scalar'].shape}")

    # 4. back to a density, two ways
    plain = to_raster(g)
    robust = reconstruct_valid(g, target_vf=float((img > 0).mean()),
                               shape=img.shape)
    print(f"\nreconstruction Dice (plain)  {roundtrip_dice(img, g):.3f}")
    print(f"material fraction: source {(img > 0).mean():.3f} | "
          f"plain {plain.mean():.3f} | erosion-robust {robust.mean():.3f}")
    print("\nNote: high Dice does not imply the design passes FEA. Physical "
          "validity is decided\nonly by scripts/eval_mechanism_gate.py, which "
          "needs a full boundary-value problem.")

    if args.save_figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise SystemExit("--save-figure needs matplotlib")
        def edge_xy(e):
            """Actual skeleton path, not the straight chord between endpoints."""
            if e.polyline:
                p = np.asarray(e.polyline)
                return p[:, 1], p[:, 0]         # polyline is (row, col)
            return ([nb[e.u].x, nb[e.v].x], [nb[e.u].y, nb[e.v].y])

        nb = g.node_by_id()
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
        ax[0].imshow(img, cmap="gray_r")
        ax[0].set_title("1. input raster")

        ax[1].imshow(img, cmap="gray_r", alpha=0.2)
        for e in g.edges:
            if e.u in nb and e.v in nb:
                xs, ys = edge_xy(e)
                ax[1].plot(xs, ys, "-", lw=1.8, color="tab:blue")
        ax[1].scatter([n.x for n in g.nodes], [n.y for n in g.nodes],
                      s=[10 + 5 * n.r for n in g.nodes], color="tab:red", zorder=3)
        ax[1].set_title(f"2. graph ({g.n_nodes()} nodes / {g.n_edges()} edges)")

        ax[2].imshow(img, cmap="gray_r", alpha=0.2)
        for c in cells:
            # Trace the cell boundary along its member edges' real geometry,
            # then order the points about the centroid to close the region.
            pts = []
            for k in c.edges:
                if 0 <= k < len(g.edges):
                    xs, ys = edge_xy(g.edges[k])
                    pts.extend(zip(np.asarray(xs), np.asarray(ys)))
            if len(pts) < 3:
                continue
            p = np.asarray(pts, float)
            cx, cy = p[:, 0].mean(), p[:, 1].mean()
            p = p[np.argsort(np.arctan2(p[:, 1] - cy, p[:, 0] - cx))]
            ax[2].fill(p[:, 0], p[:, 1], alpha=0.45,
                       color="tab:green" if c.kind == "hole" else "tab:orange")
        ax[2].set_title(f"3. rank-2 cells ({len(holes)} hole / {len(pads)} pad)")

        ax[3].imshow(robust, cmap="gray_r")
        ax[3].set_title("4. reconstruction")

        for a_ in (ax[1], ax[2]):
            a_.set_xlim(0, img.shape[1])
            a_.set_ylim(img.shape[0], 0)
            a_.set_aspect("equal")
        for a_ in ax:
            a_.set_xticks([])
            a_.set_yticks([])
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        os.makedirs(os.path.dirname(args.save_figure) or ".", exist_ok=True)
        fig.savefig(args.save_figure, dpi=130)
        print(f"\nwrote {args.save_figure}")


if __name__ == "__main__":
    main()
