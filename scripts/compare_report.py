#!/usr/bin/env python3
"""Collate eval metrics.json from several runs into one comparison table."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="run dirs, each containing eval/metrics.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for d in args.dirs:
        mp = os.path.join(d, "eval", "metrics.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        rows.append((os.path.basename(d.rstrip("/")), m))

    cols = [("dice", "dice↑"), ("port_both", "ports↑"),
            ("connected_frac", "conn↑"), ("floating_frac", "float↓"),
            ("n_components", "ncomp↓"), ("material_frac", "matl")]
    header = "| method | obj | phys | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "---|" * (3 + len(cols))
    lines = [header, sep]
    for name, m in rows:
        cells = [f"{m.get(k, float('nan')):.3f}" for k, _ in cols]
        lines.append(f"| {name} | {m.get('objective','?')} | "
                     f"{m.get('physics_weight',0)} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
