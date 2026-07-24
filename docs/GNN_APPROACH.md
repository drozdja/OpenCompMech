# The GNN approach: an SE(2)-equivariant graph generator

*Status: method and tooling implemented and tested. Comparative results are
NOT yet claim-bearing — see "Status: preliminary" below. Code:
[`src/ml/egnn.py`](../src/ml/egnn.py),
[`src/ml/graph_tensors.py`](../src/ml/graph_tensors.py),
[`src/ml/graph_flow.py`](../src/ml/graph_flow.py).*

---

## Why bother, when a UNet on rasters already works

Three concrete claims, and what each is actually worth here:

**1. The graph is a far smaller object.** Measured on the same 20 429 designs at
128px:

| representation | size on disk | values per design |
|---|---:|---:|
| raster target (`target.f16`) | 669 MB | 16 384 |
| graph tensors (`graphs.npz`) | **3.0 MB** | 1 664 |

That is ~10x fewer values per design, and ~220x smaller on disk because the
adjacency is overwhelmingly empty. A mechanism is a sparse object; storing it as
a dense image spends almost all of its capacity describing background.

**2. Connectivity is explicit, not inferred.** Whether a strut *connects* is the
single property the FEA gate cares about most — a floating blob fails
immediately. In a raster the model must learn connectivity as a statistical
property of neighbouring pixels. In a graph it is an edge: exactly represented,
and impossible to get half-right.

**3. Orientation is built in, not learned from data.** A CNN must see rotated
examples to handle rotated problems. An EGNN is equivariant *by construction*, so
a mechanism rotated 40° is not a new thing to learn. `tests/test_egnn.py` verifies
this numerically — rotating the whole problem rotates the predicted position
velocity by the same rotation, to **1e-9 in float64**, and leaves every invariant
channel bit-identical.

**The honest counterweight:** the graph passes through a lossy encoder, so it has
a representation ceiling that no generator trained on it can exceed. Earlier work
here showed such a ceiling — not the model — was the binding constraint, so the
ceiling is measured *before* any model is judged. All three claims above are
motivations to test, not results.

## Making it a fair comparison, not a demo

Everything except the two things under test is held fixed:

| held fixed | raster model | graph model |
|---|---|---|
| objective | rectified flow | rectified flow |
| conditioning | CFG, dropout 0.15 | CFG, dropout 0.15 |
| split | lineage, val_frac 0.04, seed 0 | *the same rows* |
| verification | fresh sparse-FEA gate | the same gate |
| **representation** | 128x128 raster | mechanism graph |
| **architecture** | conditional UNet | EGNN |

The graph trainer derives its split by loading `PilotCache` and intersecting —
it reported `19612 train / 817 val`, identical to the raster run, so the
comparison is genuinely like-for-like rather than approximately so.

## What the model generates

A fixed budget of node slots plus a dense adjacency:

* `node_x` (24, 4) — position (2), radius, existence
* `edge_x` (28, 28, 2) — strut present, strut width

The budget is not arbitrary: the converted dataset has **max 21 nodes, mean 6.9**,
so **0% of designs are truncated** by the 24 free slots.

**The boundary-value problem is given, not generated.** The first four slots are
*anchor* nodes read off the conditioning rasters — input port, output port, and
up to two support regions. They are known for any spec at sampling time, so they
are supplied rather than denoised, and generated struts may terminate on them.
This is the physical statement "the mechanism must attach to its ports", encoded
structurally instead of hoped for. Training also wires each role-tagged node to
its anchor, so port attachment is something the model is taught directly.

Discrete channels (existence, adjacency) are carried as relaxed ±1 variables and
thresholded at decode. This mirrors the raster model's continuous treatment and
keeps the objective identical; a proper discrete flow-matching treatment
([DeFoG, arXiv:2410.04263](https://arxiv.org/abs/2410.04263)) is the obvious
upgrade and is named as a next step rather than quietly assumed to be better.

## Architecture

`MechEGNN` follows [Satorras et al. (arXiv:2102.09844)](https://arxiv.org/abs/2102.09844)
with the coordinate-output convention of equivariant diffusion models. Node
coordinates are updated through the layers and the *displacement* produced is the
predicted velocity. Because that displacement is built only from `(x_i - x_j)`
scaled by invariant weights, it rotates with the input by construction.

Load and motion directions are equivariant vectors, so they enter messages only
through the invariants `<v_i, x_ij>`, `<v_j, x_ij>`, `<v_i, v_j>` — orientation
information used without breaking equivariance. Message passing runs over the
complete slot graph (N = 28 is small), so a strut can be created anywhere;
adjacency is an output, not a fixed input.

## Verification-first evaluation

`scripts/eval_graph.py` reports three methods through the **same** sparse-FEA
gate as the raster harness. Nothing is compared to the target raster.

| method | what it measures |
|---|---|
| `reference_native` | the source design. **Must pass 100%** or the run aborts — otherwise the protocol is broken and no other number means anything. |
| `reference_graph_roundtrip` | the ground-truth design encoded to a graph and rebuilt. **The representation ceiling** — no generator can beat it. |
| `neural` | K graphs per spec, decoded, rasterized, gated. |

The middle row is the one to read first. Encoder fidelity is already measured at
**round-trip Dice 0.969 (p10 0.944)** across all 20 429 designs, but Dice is not
the gate — only the FEA result is.

## Measured: the ceiling is the problem, and the model is not ready to be judged

`scripts/graph_ceiling.py`, n = 40 held-out designs at 128px, no model involved.
The `encoded` rows are the path a generator actually emits.

| variant | pass@1 (v1) | pass@1 (after Rung 1) | dominant fatal reason now |
|---|---:|---:|---|
| `native` — the source design | 1.000 | **1.000** | protocol valid |
| `polyline` — faithful skeleton geometry | 0.225 | **0.225** | point hinges |
| `polyline_vf` — projected to the spec's volume | 0.100 | 0.100 | point hinges (33) |
| `encoded` — encode/decode, straight struts | **0.000** | **0.075** | point hinges |
| `encoded_dense` — encode/decode, subdivided | **0.000** | **0.175** | point hinges (23) |

*(`port_interface` is reported for the native designs too, which still pass, so
it is a non-fatal strict-mode note, not a cause of failure.)*

**Rung 1 (the decoder) closed the connectivity gap.** The `encoded` path was
failing the gate's connectivity check 20/20; the cause was isolated *anchor*
nodes — ports and supports that `decode` stamped as separate disks but never
wired to the body. `decode(connect_anchors=True)` wires each isolated anchor to
its nearest body node, and clipping to the domain removed out-of-domain material.

**Rung 2 (the reconstruction) lifted the decoder ceiling substantially.** The
remaining wall was `no_point_hinge`: a medial-axis skeleton pinches to ~1px where
thin members meet, and the gate rejects any solid that does not survive a 1px
erosion. `reconstruct_valid` (in `src/graph`) fixes it — densify, **floor every
member to manufacturable width**, project to the spec's volume, clip to the
domain, and **repair single-pixel hinges locally** (unlike a global closing,
which welds the compliant gaps shut). The recipe was built one measured step at a
time. These are **decoder measurements, not model results**: no network is
involved, the input is the ground-truth design, and `scripts/graph_ceiling.py`
reproduces them deterministically.

| variant | pass@1 | what it bounds |
|---|---:|---|
| `native` — source design | 1.000 | protocol valid on this subset |
| `polyline` — naive medial-axis | 0.225 | the old (wrong) ceiling |
| `polyline_valid` — GT graph, robust reconstruction | **0.600** | best case for the representation |
| `encoded_dense` — model path, naive reconstruction | 0.175 | the old model-path ceiling |
| `encoded_valid` — **model path, robust reconstruction** | **0.375** | the ceiling a generator could approach |

*(n = 40 held-out designs, before the frozen protocol existed; these predate
`freeze_eval_protocol.py` and should be re-measured on the frozen test set
before being quoted anywhere.)*

**What this establishes** is narrow but real: the medial-axis graph was never
inherently unable to express a working compliant mechanism — the naive
reconstruction was throwing that away, and a better decoder recovers most of it.
It says nothing yet about how a *generator* performs, and it is not a comparison
against the raster model; the two have not yet been evaluated under matched
conditions.

Two dead ends ruled out by measurement, not argument:

- **Dice is not a proxy for the gate.** The naive polyline path is Dice 0.958 and
  passes 0.225. Shape similarity and physical validity are nearly unrelated here.
- **Global closing does not work; local hinge-repair does.** Morphological closing
  welds the compliant structure solid (`mechanism_path`/`signed_ga` fail 20/20);
  repairing only the single-pixel articulation points leaves the load path's
  compliance intact and `minimum_output_stroke` still passes.

### What is left in the ceiling (encoded_valid 0.375 < polyline_valid 0.600)

The gap between the model-path ceiling and the best case is the straight-strut,
single-width encoding (its residual failures are `energy_gini` — strain energy
too concentrated — not hinges anymore). Predicting per-segment widths or storing a
coarse polyline in the tensor layout would recover some of it; that is the next
representation refinement, not a blocker for training.

## Status: preliminary, NOT claim-bearing

A first EGNN generator has been trained (1.09M params, 60k steps) and scored
end-to-end. **Those numbers are deliberately not reported here**, because a
pre-publication audit found the run does not support a claim:

1. **The evaluation failed its own protocol self-check.** The harness requires
   every native reference to pass the frozen gate before model numbers may be
   interpreted; one held-out ground-truth design did not, and the code prints
   exactly that. A run in that state is not quotable.
2. **It is not yet a clean representation ablation.** The graph pipeline is
   scored through a deterministic decoder (`reconstruct_valid`: anchor
   connection, volume projection, hinge repair) that the raster pipeline has no
   equivalent of. That is legitimate engineering, but it makes the comparison
   end-to-end pipeline vs pipeline, not representation vs representation.
3. **Training and sampling budgets were not matched.** The two models saw
   different numbers of training examples and were tuned to different (near-zero)
   degrees.
4. **The holdout is IID-like.** Almost every spec/lineage in the corpus is
   unique, so the `lineage` split is effectively a random row split, not a
   generalization test. See [V1_CORPUS_SNAPSHOT.md](V1_CORPUS_SNAPSHOT.md).

An earlier draft of this document also compared a pass@8 model number against a
pass@1 decoder ceiling and reported the ratio as "fraction of ceiling reached".
That is invalid twice over: the two are different K, and a pass@1 ceiling is not
an upper bound on pass@8, since a generator with K tries can find a passing
design that the ground-truth graph does not yield. That claim is withdrawn.

### What is being done about it

`scripts/freeze_eval_protocol.py` produces a frozen protocol artifact that fixes
(1) and the set-size problem by rule rather than by hand: a spec is eligible iff
its own native reference passes the gate *and* it has a converted graph, and the
eligible specs are partitioned, stratified by mechanism type, into disjoint
**tuning** and **untouched test** sets with full hash provenance.

The remaining work before any comparative number is published:

- decompose learned performance from decoder repair (raw vs repaired, both sides)
- match training exposure (examples seen), and report parameters and compute
- give both models an equal tuning budget, then repeat the best configs over
  several sampling seeds
- add a genuine held-out mechanism type/family experiment

Until that is complete, this repository documents the *method*, not a result.

### Reproduce

```bash
COMP2D_SAMPLE_PRECISION=fp32 python scripts/eval_graph.py \
    --graphs data/v1_graph_128 --cache data/v1_broad_cache_128 \
    --ckpt runs/v1graph_128/ckpt_final.pt --out runs/v1graph_128_eval \
    --n-specs 60 --K 8 --steps 50 --cfg 1.0 --workers 8
```

## Running it

```bash
# 1. convert the raster cache to graphs (~1 min on 12 cores)
python scripts/build_graph_dataset.py --cache data/v1_broad_cache_128 \
    --out data/v1_graph_128 --workers 12

# 2. train (fp32: gfx1201 — see docs/HARDWARE_NOTES_RDNA4.md)
python scripts/train_graph.py --graphs data/v1_graph_128 \
    --cache data/v1_broad_cache_128 --out runs/v1graph_128 \
    --steps 60000 --batch 64 --hidden 128 --layers 6 --precision fp32

# 3. pass@K through the same FEA gate
python scripts/eval_graph.py --graphs data/v1_graph_128 \
    --cache data/v1_broad_cache_128 --ckpt runs/v1graph_128/ckpt_final.pt \
    --out runs/v1graph_128_eval --n-specs 60 --K 8 --workers 6
```

## How to read the result when it lands

- If `reference_graph_roundtrip` is **low**, the encoder is the bottleneck and
  the model number says little — fix the representation first. This is exactly
  what the 64px raster experiment taught.
- If the ceiling is high but `neural` is low, the graph generator is genuinely
  worse than the UNet, and the representation argument does not survive contact
  with the measurement.
- Raster reference points exist from earlier runs, but they were measured before
  the frozen protocol and under unmatched budgets, so they are not quoted here.
  Both models need re-scoring on the frozen test set before any comparison.

## Known limitations

- **The rank-2 cells are not wired in yet.** `src/graph/complex.py` builds and
  validates them, but this generator uses the rank-0/1 graph only.
- **Discrete channels are relaxed, not discrete.** DeFoG-style discrete flow
  matching is the principled treatment.
- **Node ordering is canonical (by y then x), not matched.** The network is
  permutation-equivariant so the target is well defined, but Hungarian matching
  would remove the dependence on that choice.
- **Decode uses straight struts.** The converter can store faithful polylines;
  the generator does not predict them, so curved members are approximated by
  straight ones of the predicted width.
