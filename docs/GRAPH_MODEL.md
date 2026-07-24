# The graph generative model

An SE(2)-equivariant EGNN that generates mechanism graphs, as an alternative to
the raster diffusion model. Code: [`src/ml/egnn.py`](../src/ml/egnn.py),
[`src/ml/graph_tensors.py`](../src/ml/graph_tensors.py),
[`src/ml/graph_flow.py`](../src/ml/graph_flow.py).

> **No comparative result is reported here.** See
> [Evaluation status](#evaluation-status).

## Motivation

Three properties motivate a graph model over a raster model. They are reasons to
run the experiment, not findings.

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

## Evaluation status

A first generator has been trained and scored end to end. **Those numbers are
not reported**, because a pre-publication audit found the run does not support a
claim:

1. **The evaluation failed its own protocol self-check.** The harness requires
   every native reference to pass the gate before model numbers may be
   interpreted; one did not, and the code says so.
2. **Not a clean representation ablation.** The graph pipeline is scored through
   the deterministic decoder above, which the raster pipeline has no equivalent
   of.
3. **Budgets were not matched.** The models saw different numbers of training
   examples and neither was meaningfully tuned.
4. **The holdout is IID-like.** See [DATASET.md](DATASET.md).

A withdrawn claim: an earlier write-up divided a pass@8 model number by a pass@1
decoder ceiling and reported the ratio as "fraction of ceiling reached". That is
invalid — different K, and a pass@1 ceiling is not an upper bound on pass@8,
since a generator with K attempts can find a passing design the ground-truth
graph does not yield.

[`scripts/freeze_eval_protocol.py`](../scripts/freeze_eval_protocol.py)
addresses (1) and the sample-size problem by rule: a specification is eligible
only if its own native reference passes the gate and it has a converted graph,
and eligible specifications are partitioned — stratified by mechanism type —
into disjoint tuning and untouched test sets with full hash provenance.

Remaining work before any comparative number is published:

- decompose learned performance from decoder repair (raw and repaired, both
  pipelines)
- match training exposure in examples seen, and report parameters and compute
- give both models an equal tuning budget, then repeat the best configurations
  across several sampling seeds
- add a held-out mechanism type or family experiment

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
