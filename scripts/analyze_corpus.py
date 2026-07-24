#!/usr/bin/env python3
"""Snapshot analysis of the v1 corpus: per-family counts, motion-class balance,
topological diversity (nearest-neighbour Dice), and port-accessibility split.
Read-only; safe to run while generation continues. Bounded per-family sampling
so it stays light next to the FEA workers.
"""
import glob, json, os, sys, random
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/v1_pool_20260720"
NN_SAMPLE = 80  # densities/family for the diversity estimate


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9)


fam_dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d))
total = 0
mag = {}; trn = {}; access = {"accessible": 0, "embedded": 0, "unknown": 0}
print(f"{'family':<18}{'valid':>7}{'nnDice(med)':>12}{'access%':>9}")
print("-" * 46)
for d in fam_dirs:
    fam = os.path.basename(d)
    jsons = glob.glob(os.path.join(d, "*.json"))
    n = len(jsons)
    if n == 0:
        continue
    total += n
    acc_ok = 0; acc_known = 0
    for jf in jsons:
        try:
            m = json.load(open(jf))
        except Exception:
            continue
        val = m.get("validation", {})
        mo = val.get("motion", {}) or {}
        mc = mo.get("magnitude_class"); tc = mo.get("transfer_class")
        if mc: mag[mc] = mag.get(mc, 0) + 1
        if tc: trn[tc] = trn.get(tc, 0) + 1
        pi = val.get("port_interface", {})
        if "passed" in pi:
            acc_known += 1
            if pi["passed"]:
                acc_ok += 1
                access["accessible"] += 1
            else:
                access["embedded"] += 1
        else:
            access["unknown"] += 1
    # diversity: median nearest-neighbour Dice over a bounded sample
    dfiles = glob.glob(os.path.join(d, "*.density.npy"))
    random.seed(0); random.shuffle(dfiles); dfiles = dfiles[:NN_SAMPLE]
    nn = float("nan")
    if len(dfiles) >= 3:
        D = [np.load(f) > 0.5 for f in dfiles]
        nns = []
        for i in range(len(D)):
            best = max(dice(D[i], D[j]) for j in range(len(D)) if j != i)
            nns.append(best)
        nn = float(np.median(nns))
    acc_pct = (100.0 * acc_ok / acc_known) if acc_known else float("nan")
    print(f"{fam:<18}{n:>7}{nn:>12.3f}{acc_pct:>9.1f}")

print("-" * 46)
print(f"{'TOTAL':<18}{total:>7}")
print("\nmagnitude_class:", dict(sorted(mag.items(), key=lambda x: -x[1])))
print("transfer_class :", dict(sorted(trn.items(), key=lambda x: -x[1])))
print("port access    :", access,
      f"({100*access['accessible']/max(1,access['accessible']+access['embedded']):.0f}% accessible of known)")
