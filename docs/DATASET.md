# Dataset pipeline

How a mechanism corpus is built and what a design must satisfy to enter it. The
corpus itself is not distributed with this code.

## Acceptance criteria

Every accepted mechanism carries:

- an explicit design envelope, all support degrees of freedom, port definitions
  and all spring values;
- a common 4-connected support/input/output material path, plus feature-size,
  erosion and volume checks;
- independent sparse-FEA measurement of transfer, selectivity and interface
  access;
- a lineage/specification group that never straddles a validation split.

`scripts/eval_mechanism_gate.py` is the required final gate, for generated
samples as well as corpus designs.

## Pipeline

```bash
# 1. Collect source directories into a traceable manifest.
#    Earlier directories win byte-identical duplicates.
python scripts/build_dataset_manifest.py \
    --dirs data/raw/* --out data/raw_manifest.json

# 2. For a fixed boundary-value problem, produce alternatives from
#    different optimizer starts.
python scripts/generate_multisolution_bank.py \
    --meta data/specs/example.json --out-dir data/raw/example \
    --n 64 --start-id 0

# 3. Apply the full sparse-FEA gates, balance across motion classes and
#    drop near-duplicate variants.
python scripts/curate_quality_diversity.py \
    --manifest data/raw_manifest.json --out data/curated_manifest.json \
    --per-motion 500 --min-distance 0.15

# 4. Pack a tensor cache carrying the BVP contract:
#    domain + per-DOF supports + springs.
python scripts/build_tensor_cache.py \
    --manifest data/curated_manifest.json --out data/cache

# 5. Train.
python scripts/train_pilot.py --cache data/cache --split-mode lineage
```

## Splits

Group-aware splitting is available by lineage, specification, family or type
(`--split-mode`). Use `scripts/build_eval_split.py` to freeze a plan for a
claim-bearing run, and `scripts/audit_eval_split.py` to check for exact and
near-topology leakage across the boundary.

**Important caveat.** Because nearly every specification in a generated corpus
is unique, a `lineage` split is effectively a random row split — an IID-like
holdout, not a generalization test. A held-out mechanism *type* or *family* is
the meaningful out-of-distribution experiment.

## Composition is uneven by construction

Generator families do not contribute equally, for three compounding reasons:

- **Throughput.** Parametric families (linkage replacement, flexure) produce a
  valid design cheaply; optimization-based families (SIMP, MMC, ground
  structure) require a full topology optimization per design.
- **Gate yield.** Families fail the validity gate at different rates.
- **De-duplication.** Optimization families converge toward one solution basin
  per boundary-value problem, so additional seeds mostly yield near-duplicates
  that de-duplication removes; parametric families produce genuinely distinct
  topologies.

Any absolute rate measured on such a corpus is therefore weighted toward the
larger families. Report per-family results when the distinction matters.

Note also that unique *records* and unique *binarised topologies* are not the
same quantity: a small number of distinct designs collide once binarised at a
given resolution.

## Non-claims

The gate reports normalized stress. It cannot certify yield, fatigue, buckling,
contact, finite stroke, manufacturability or safety without a dimensional use
case (material, thickness, load, process and allowable limits). Do not make
those claims from this dataset.

A rendered image or a proxy connectivity score is not a validity result; only
the independent FEA gate is.
