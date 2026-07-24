# OpenCompMech

Research code for **generating 2D compliant mechanisms and verifying them with
finite-element analysis**. The organising principle is that a generated design
counts for nothing until a fresh FEA solve says it actually works — so every
number here comes from a solver, never from similarity to a training example.

> **Status: work in progress.** The generation pipeline, the FEA verifier, the
> graph representation and the evaluation protocol are implemented and tested.
> **Comparative model results are not yet claim-bearing** and are deliberately
> not reported — see [Honest status](#honest-status) below. This repository
> currently documents a *method*, not a benchmark result.

## What this is

A pipeline for building a corpus of 2D compliant mechanisms across five
generator families, and for asking whether a conditional generative model can
propose candidate topologies that survive an independent physics check.

Everything is **normalized 2D linear elasticity**. It is not a manufacturing
dataset, not a validated engineering tool, and makes no claim about any physical
part.

## Why it might be interesting

**The verification gate is the product.** Most generative-design work reports
similarity to a reference. Here a candidate must pass a full-resolution sparse
FEA screen: connectivity, minimum feature size, no single-pixel hinges, volume
fraction, port attachment, and functional response (does it actually move the
right way, with the right sign and enough stroke). See
[docs/VERIFICATION_PROTOCOL.md](docs/VERIFICATION_PROTOCOL.md).

**A reusable, generator-independent graph representation.** `src/graph/` derives
one canonical graph from the *finished shape* of any mechanism, so all five
families share a single schema. It is SE(2)-equivariant, round-trips back to a
raster for re-verification, and includes an ETNN-style combinatorial-complex
lift (rank-2 cells for holes and anchor pads) validated against Euler's formula.
See [docs/GRAPH_REPRESENTATION.md](docs/GRAPH_REPRESENTATION.md).

**Measured ceilings before model claims.** Before judging any generator, the
pipeline measures what its *representation* can express at all, by pushing
ground-truth designs through the encoder and back and gating the result
(`scripts/graph_ceiling.py`). On this project that ceiling — not the model — was
repeatedly the binding constraint.

**Hard-won hardware notes.** Two silent-corruption failure modes on RDNA4
(gfx1201), including fused AdamW producing negative second moments, with fixes:
[docs/HARDWARE_NOTES_RDNA4.md](docs/HARDWARE_NOTES_RDNA4.md).

## Repository map

```text
src/core/          mesh, problem definition, sparse Cholesky factorization
src/generators/    mechanism generators (5 families, 15 types)
src/solvers/       linear and nonlinear FEA
src/validation/    geometry, connectivity, hinge, port and motion checks
src/ml/            raster model (UNet, rectified flow) + graph model (EGNN)
src/graph/         canonical mechanism graph, ETNN complex lift, PyG export
scripts/           generation, training, evaluation, auditing
tests/             unit tests (graph schema, equivariance, gate policy, harness)
config/            frozen evaluation protocol and split plans
```

## Getting started

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # tests run without a GPU
```

The graph library is usable standalone and has no hard GNN-framework dependency:

```python
from src.graph import from_raster, to_pyg, build_cells, euler_check

g = from_raster(density, cond=cond, cond_channels=names)   # any family
build_cells(g)                                             # rank-2 ETNN lift
assert euler_check(g)["ok"]                                # bounded faces = E-V+C
data = to_pyg(g)          # torch_geometric Data, or a tensor dict without it
```

## Honest status

A pre-publication audit of the first CNN-vs-GNN comparison found it **not
publication-grade**. The specific problems, all of which are being fixed rather
than papered over:

1. **The evaluation failed its own protocol self-check.** The harness requires
   every native reference design to pass the gate before model numbers may be
   interpreted. One did not, and the code says so explicitly. Numbers from a run
   in that state are not quotable.
2. **It was not a clean representation ablation.** The graph pipeline is scored
   through a deterministic decoder (anchor connection, volume projection, hinge
   repair) with no raster equivalent — a pipeline-vs-pipeline comparison, not
   representation-vs-representation.
3. **Budgets were not matched.** The two models saw different numbers of
   training examples, and neither was meaningfully tuned.
4. **The holdout is IID-like.** Nearly every spec in the corpus is unique, so
   the "lineage" split is effectively a random row split, not a generalization
   test.
5. **A withdrawn claim.** An earlier write-up divided a pass@8 model number by a
   pass@1 decoder ceiling and called the ratio "fraction of ceiling reached".
   That is invalid — different K, and a pass@1 ceiling is not an upper bound on
   pass@8. It has been removed.

`scripts/freeze_eval_protocol.py` addresses (1) and the sample-size problem by
rule rather than by hand: a spec is eligible only if its own native reference
passes the gate and it has a converted graph, and eligible specs are partitioned
— stratified by mechanism type — into disjoint **tuning** and **untouched test**
sets with full hash provenance. Remaining work is listed in
[docs/GNN_APPROACH.md](docs/GNN_APPROACH.md).

## Limitations

- **Normalized 2D linear elasticity only.** No material calibration, thickness,
  yield, fatigue, buckling, contact, friction, manufacturing constraints, or
  physical testing. Those are separate validation layers, not footnotes.
- **Synthetic corpus.** The corpus is generated, not measured from real parts.
- **Uneven family coverage.** Parametric families produce far more designs than
  optimization-based ones (throughput, gate yield, and de-duplication of
  near-identical optimizer solutions all differ per family). Absolute rates are
  weighted toward the larger families.
- **One solution per specification.** Per-spec multimodality is unsolved, so the
  corpus cannot currently teach a model that a problem has several good answers.
- **Goal-conditioned, not pure inverse design.** The conditioning vector
  includes performance labels measured from a reference design.
- **The corpus is not distributed here.** Design counts quoted in the docs refer
  to a corpus generated by this code; note that unique *records* and unique
  *binarised topologies* are not identical (a small number of designs collide
  after binarisation).

## License

MIT for the source code — see [LICENSE](LICENSE). The generated corpus is not
included in this repository and would carry its own license.
