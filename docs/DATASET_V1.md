# Dataset v1: release gate, not a sample count

The pre-existing mechanism folders are retained as `v0` exploratory output.
They are useful for debugging and pretraining, but are not a release-ready
engineering corpus: some rows are duplicate deterministic restarts, old
oblique spring labels used a diagonal approximation, and their IID split can
leak near-identical topology into validation.

v1 starts with a traceable manifest and only trains from a curated manifest.

```bash
# Earlier directories win byte-identical duplicates.
python scripts/build_dataset_manifest.py --dirs data/new_mech/* --out data/v1/raw_manifest.json

# For each deliberately fixed BVP, create alternatives from different starts.
python scripts/generate_multisolution_bank.py --meta data/specs/example.json \
  --out-dir data/v1/raw/example --n 64 --start-id 0

# Apply full sparse-FEA gates, then balance functions and remove close variants.
python scripts/curate_quality_diversity.py --manifest data/v1/raw_manifest.json \
  --out data/v1/curated_manifest.json --per-motion 500 --min-distance 0.15

# Tensor cache has the v1 BVP contract: domain + per-DOF supports + springs.
python scripts/build_tensor_cache.py --manifest data/v1/curated_manifest.json \
  --out data/v1/cache
python scripts/train_pilot.py --cache data/v1/cache --split-mode lineage ...
```

Every accepted v1 mechanism must have:

- an explicit design envelope, all support DOFs, ports, and all spring values;
- a common 4-connected support/input/output material path, feature/erosion and
  volume checks;
- independent sparse-FEA measurement of transfer, selectivity and interface
  access; and
- a lineage/spec group that never straddles a validation split.

`scripts/eval_mechanism_gate.py` is the required last gate for neural samples.
It reports normalized stress, but cannot certify yield, fatigue, buckling,
contact, finite stroke, manufacturability or safety without a dimensional use
case (material, thickness, load, process and allowable limits). Do not make
those claims from this dataset.

For guidance sweeps, the script now fixes the noise seed across scales,
projects each result to target volume, emits raw densities/JSON metrics, and
requires the independent gate above. A pleasing PNG or a proxy connectivity
score is not a validity result.
