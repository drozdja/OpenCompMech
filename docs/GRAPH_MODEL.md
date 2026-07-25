# The graph generative model

An SE(2)-equivariant EGNN that generates mechanism graphs, as an alternative to
the raster diffusion model. Code: [`src/ml/egnn.py`](../src/ml/egnn.py),
[`src/ml/graph_tensors.py`](../src/ml/graph_tensors.py),
[`src/ml/graph_flow.py`](../src/ml/graph_flow.py).

> **Result:** at matched training exposure the raster model substantially
> outperforms this one. See
> [Result](#result-the-raster-model-outperforms-the-graph-model).

## Motivation (not borne out)

Three properties motivated building a graph model. The measurement below
contradicts all three as decisive advantages. They are recorded because the
reasoning is worth understanding, not because it was validated.

**Compactness.** A padded graph is roughly an order of magnitude fewer values
per design than a dense raster, and far smaller on disk because the adjacency is
mostly empty. A mechanism is a sparse object; a dense image spends most of its
capacity describing background.

**Explicit connectivity.** Whether two members connect is the property the FEA
gate is most sensitive to — a floating region fails immediately. In a raster
that must be learned as a statistical property of neighbouring pixels; in a
graph it is an edge.

**Built-in orientation handling.** An EGNN is equivariant by construction, so a
rotated problem is not a separate thing to learn.
[`tests/test_egnn.py`](../tests/test_egnn.py) verifies this numerically:
rotating the problem rotates the predicted position velocity by the same
rotation to 1e-9 in float64, leaving invariant channels bit-identical.

**Counterweight.** The graph passes through a lossy encoder, so it has a
representation ceiling that no generator trained on it can exceed. That ceiling
is measured before any model is judged.

## Representation

The model generates a fixed budget of node slots plus a dense adjacency:

- `node_x` — per slot: position (2), radius, existence
- `edge_x` — per slot pair: strut presence, strut width

The boundary-value problem is **given, not generated**: the first slots are
anchor nodes read from the conditioning rasters (input port, output port, up to
two support regions). They are known for any specification at sampling time, so
they are supplied rather than denoised, and generated struts may terminate on
them. Training wires each role-tagged node to its anchor, so port attachment is
taught directly rather than hoped for.

Discrete channels (existence, adjacency) are carried as relaxed ±1 variables and
thresholded at decode. This matches the raster model's continuous treatment and
keeps the training objective identical. Discrete flow matching
([DeFoG, arXiv:2410.04263](https://arxiv.org/abs/2410.04263)) is the principled
alternative and is not yet implemented.

## Architecture

`MechEGNN` follows
[Satorras et al. (arXiv:2102.09844)](https://arxiv.org/abs/2102.09844) with the
coordinate-output convention of equivariant diffusion models. Node coordinates
are updated through the layers and the resulting displacement is the predicted
velocity. Because that displacement is built only from `(x_i - x_j)` scaled by
invariant weights, it rotates with the input by construction.

Load and motion directions are equivariant vectors and enter messages only
through the invariants `<v_i, x_ij>`, `<v_j, x_ij>` and `<v_i, v_j>`, so
orientation information is used without breaking equivariance.

Message passing runs over the complete slot graph, so a strut may be created
anywhere; adjacency is an output, not a fixed input. All message-passing layers
share one hidden width; only the input embeddings and output heads are shaped by
the problem.

## Decoding to a verifiable density

A generated graph must become a density before it can be verified.
`reconstruct_valid` in [`src/graph`](../src/graph) performs:

1. **Densify** — subdivide curved members so straight-segment stamping does not
   shortcut across the domain.
2. **Floor member width** — guarantee each member survives the erosion the hinge
   check applies.
3. **Project to target volume** — bisect a single width scale to meet the
   specification's volume fraction.
4. **Clip to the declared domain.**
5. **Repair single-pixel hinges locally** — a global morphological closing welds
   the compliant gaps shut and destroys the mechanism; local repair of
   articulation points does not.

This is deterministic post-processing, not learned behaviour. It is why the
current comparison is pipeline-vs-pipeline rather than
representation-vs-representation (see below).

## Representation ceiling

Before judging a generator, [`scripts/graph_ceiling.py`](../scripts/graph_ceiling.py)
measures what the representation can express at all: ground-truth designs are
pushed through the encoder and back, then gated. No model is involved and the
measurement is deterministic.

Reported variants:

| variant | meaning |
|---|---|
| `native` | the source design (protocol self-check) |
| `polyline` | graph to raster using faithful skeleton geometry |
| `polyline_valid` | the same, through `reconstruct_valid` |
| `encoded` | the padded encode/decode path a model actually emits |
| `encoded_valid` | the same, through `reconstruct_valid` |

Two findings from this tooling are worth recording, both established by
measurement:

- **Shape similarity does not predict physical validity.** A reconstruction can
  reach high Dice overlap with the source design and still fail the gate most of
  the time.
- **The naive medial-axis reconstruction, not the graph itself, was the binding
  constraint.** A skeleton with per-node radii pinches to roughly one pixel where
  thin members meet, and the gate rejects any solid that does not survive a
  one-pixel erosion. An erosion-robust decoder recovers most of that loss.

## Result: the raster model outperforms the graph model

Measured with `scripts/compare_models.py` on the frozen protocol's test split.
One driver runs both models, so specifications, candidate count, seed schedule
and FEA gate are identical by construction. **Training exposure is matched at
2.56M examples** (CNN 80k x 32, GNN 40k x 64) and neither model has been tuned,
so both received the same (zero) tuning budget.

n = 200 specs, K = 8, 95% percentile-bootstrap CIs over specs.

| method | pass@1 | pass@8 | candidate rate |
|---|---:|---:|---:|
| `reference_native` (self-check) | 1.000 | 1.000 | 1.000 |
| `raster_raw` | 0.720 [0.655, 0.775] | **0.905** [0.860, 0.945] | 0.719 |
| `raster_projected` | 0.720 [0.655, 0.775] | **0.905** [0.860, 0.945] | 0.723 |
| `graph_raw` | 0.005 [0.000, 0.015] | 0.040 [0.015, 0.070] | 0.005 |
| `graph_repaired` | 0.110 [0.065, 0.150] | **0.335** [0.270, 0.400] | 0.099 |

**The raster model wins decisively** — 0.905 vs 0.335 pass@8, CIs nowhere near
overlapping — at equal data exposure, despite having no equivariance and no
explicit connectivity. The motivating arguments for the graph representation did
not survive measurement.

### Almost all of the graph pipeline's performance is decoder repair

`graph_raw` 0.040 to `graph_repaired` 0.335: roughly **88% of what the graph
pipeline achieves comes from deterministic post-processing**, not from the
learned model. The raster pipeline is the opposite — `raster_raw` and
`raster_projected` are identical to three decimals, so it needs no repair at all.

A single end-to-end number would have hidden this. It is visible only because
each pipeline was scored twice, raw and repaired.

### Where the graph representation does work

| type | raster pass@8 | graph pass@8 |
|---|---:|---:|
| rr_compound_lever | 1.00 | **1.00** |
| rr_lever | 1.00 | 0.73 |
| fact_translation | 1.00 | 0.67 |
| gs_truss | 0.67 | 0.53 |
| gs_opt | 0.47 | **0.47** |
| rr_four_bar | 1.00 | 0.33 |
| rr_bridge_amp | 1.00 | 0.27 |
| fact_rotation | 1.00 | 0.20 |
| rr_slider_crank | 0.87 | 0.20 |
| crusher | 1.00 | 0.07 |
| crank_slider | 0.20 | 0.00 |
| gripper / inverter / mmc_opt | 1.00 | **0.00** |

The pattern follows what the representation is: a medial-axis strut graph. It
matches the raster model on **linkages and ground-structure trusses**
(`rr_compound_lever`, `gs_opt`), stays competitive on `rr_lever`, `gs_truss` and
`fact_translation`, and collapses to zero on **SIMP continuum families**
(`gripper`, `inverter`, `mmc_opt`, `crusher`), where the mechanism is a
distributed compliant body rather than a network of struts.

That is a representation result, not a model result: skeletonizing a continuum
into struts discards the thing that makes it work.

### How it fails

`graph_repaired`'s dominant failures are functional, not geometric:
`signed_ga` (1351), `minimum_output_stroke` (1061), `output_alignment` (764),
`positive_output` (635). The graph model produces structures that are
geometrically plausible and connected but move the wrong way, or not far enough.

### A withdrawn claim and a corrected measurement

An earlier version of this document reported the opposite conclusion — that the
graph model beat the raster model at 128px, 0.300 against 0.05. **Both numbers
were wrong.**

- The 0.05 raster figure was an **evaluation artifact**. `guided_sample`'s
  on-device volume projection collapses the density at 128px: for a 20% volume
  target only ~0.08% of pixels survive thresholding, so every candidate failed
  `bc_connected` and `mechanism_path`. `scripts/eval_harness.py` uses that path,
  with a comment asserting the follow-up projection is "normally a numerical
  no-op" — true only because the field was already destroyed. The comparison
  driver samples raw and projects explicitly instead.
- The graph figure came from a different, non-comparable spec selection, scored
  before the frozen protocol existed.

The same document also divided a pass@8 model number by a pass@1 decoder ceiling
and called the ratio "fraction of ceiling reached". That is invalid twice over:
different K, and a pass@1 ceiling does not bound pass@8, since a generator with
K attempts may find a passing design the ground-truth graph does not yield.

### What this supports, and what it does not

Supported: at matched data exposure and equal (zero) tuning, on this corpus, the
raster model is substantially better overall, and the graph representation is
viable only for strut-like mechanism families.

Not yet supported — a general claim about CNNs versus GNNs. Outstanding:

- neither model is tuned; an equal tuning budget could move both
- single training seed and single sampling-seed schedule
- the holdout is IID-like, so this measures in-distribution performance
- the base-96 raster checkpoint was resumed mid-run with a reset optimizer, so
  it is a capacity probe rather than a clean architecture ablation

### Reproduce

```bash
COMP2D_SAMPLE_PRECISION=fp32 python scripts/eval_graph.py \
    --graphs data/v1_graph_128 --cache data/v1_broad_cache_128 \
    --ckpt runs/v1graph_128/ckpt_final.pt --out runs/v1graph_128_eval \
    --n-specs 60 --K 8 --steps 50 --cfg 1.0 --workers 8
```

## Usage

```bash
# 1. Convert a raster cache to graphs.
python scripts/build_graph_dataset.py \
    --cache data/cache --out data/graphs --workers 12

# 2. Train. fp32 is required on gfx1201; see HARDWARE_NOTES_RDNA4.md.
python scripts/train_graph.py \
    --graphs data/graphs --cache data/cache --out runs/graph \
    --steps 60000 --batch 64 --hidden 128 --layers 6 --precision fp32

# 3. Measure the representation ceiling (no model involved).
python scripts/graph_ceiling.py --cache data/cache --n 40 --workers 8

# 4. Score through the same FEA gate as the raster model.
python scripts/eval_graph.py \
    --graphs data/graphs --cache data/cache \
    --ckpt runs/graph/ckpt_final.pt --out runs/graph_eval --K 8 --workers 6
```

## Limitations

- **Rank-2 cells are not used.** `src/graph/complex.py` builds and validates the
  ETNN combinatorial-complex lift, but this generator uses the rank-0/1 graph
  only.
- **Discrete channels are relaxed, not discrete.**
- **Node ordering is canonical (by position), not matched.** The network is
  permutation-equivariant so the target is well defined, but Hungarian matching
  would remove the dependence on that choice.
- **Decoding uses straight struts.** The converter can store faithful polylines;
  the generator does not predict them, so curved members are approximated by
  straight segments of the predicted width.
