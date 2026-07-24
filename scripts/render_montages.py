#!/usr/bin/env python3
"""Render per-family montages of the v1 corpus so designs can be eyeballed.

For each family: a grid of random valid designs (material = black). Also a
cross-family overview. Read-only. PNGs land in <out-dir>.
"""
import glob, os, random, sys
import numpy as np
from PIL import Image

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/v1_pool_20260720"
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/corpus_viz"
GRID = 6          # 6x6 = 36 designs per family montage
CELL = 128
PAD = 3
os.makedirs(OUT, exist_ok=True)


def render(dfile):
    a = np.load(dfile)
    if a.shape[0] != CELL:
        img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
        a = np.asarray(img.resize((CELL, CELL), Image.NEAREST)) / 255.0
    v = (255 * (1 - np.clip(a, 0, 1))).astype(np.uint8)   # material dark
    return v


def montage(files, cols, rows, title=""):
    W = cols * (CELL + PAD) - PAD
    H = rows * (CELL + PAD) - PAD
    canvas = np.full((H, W), 245, np.uint8)
    for k, f in enumerate(files[:cols * rows]):
        r, c = divmod(k, cols)
        canvas[r * (CELL + PAD):r * (CELL + PAD) + CELL,
               c * (CELL + PAD):c * (CELL + PAD) + CELL] = render(f)
    return Image.fromarray(canvas)


fam_dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d))
overview_rows = []
for d in fam_dirs:
    fam = os.path.basename(d)
    files = glob.glob(os.path.join(d, "*.density.npy"))
    if not files:
        continue
    random.seed(1); random.shuffle(files)
    montage(files, GRID, GRID).save(os.path.join(OUT, f"{fam}.png"))
    print(f"{fam}: {len(files)} designs -> {fam}.png", flush=True)
    overview_rows.append((fam, files[:8]))

# cross-family overview: one row of 8 per family, stacked, family label strip
if overview_rows:
    rowW = 8 * (CELL + PAD) - PAD
    labelH = 22
    tiles = []
    for fam, files in overview_rows:
        strip = np.full((CELL + labelH, rowW), 245, np.uint8)
        for k, f in enumerate(files):
            strip[labelH:labelH + CELL, k * (CELL + PAD):k * (CELL + PAD) + CELL] = render(f)
        img = Image.fromarray(strip)
        tiles.append(img)
    from PIL import ImageDraw
    total = Image.new("L", (rowW, sum(t.height + PAD for t in tiles)), 245)
    y = 0
    draw = ImageDraw.Draw(total)
    for (fam, _), t in zip(overview_rows, tiles):
        total.paste(t, (0, y))
        draw.text((4, y + 4), fam, fill=0)
        y += t.height + PAD
    total.save(os.path.join(OUT, "_overview.png"))
    print("overview -> _overview.png", flush=True)
