# The mechanism graph representation (`src/graph`)

A single, reusable, generator-independent graph representation for 2D compliant
mechanisms — designed to be the substrate a graph model (GNN) trains on, and to
interoperate with the existing raster pipeline through a shared boundary-value
problem and the FEA verifier.

*Status: 2026-07-23. Code: [`src/graph/`](../src/graph/). CLI:
[`scripts/mech_to_graph.py`](../scripts/mech_to_graph.py). Tests:
[`tests/test_mech_graph.py`](../tests/test_mech_graph.py).*

---

## Why this exists

The generators store *family-specific* parametric recipes (truss node counts,
linkage link-lengths, MMC component lists), and SIMP stores none. That is
provenance, not a usable representation. The one thing every family shares is the
**finished shape** — the density that carries the physics and passes FEA. This
library derives **one canonical graph from that shape, the same way for every
family**, so SIMP is not a special case and there is a single schema to train on.

Design principles (SOTA-informed, mid-2026):

- **EGNN / E(n)-equivariant ready** ([Satorras 2021](https://arxiv.org/abs/2102.09844)):
  node *positions* are an equivariant channel; all other features are rotation/
  translation **invariant** scalars, except explicit **direction vectors** (load,
  output motion) which are equivariant. That is exactly what an EGNN consumes.
- **SE(2), not E(2):** reflection is *not* a symmetry — a mirror-image mechanism
  inverts its output direction (chirality is physical).
- **Verification-first:** every graph round-trips to a raster (`to_raster`), so a
  *generated* graph can be re-checked by the same FEA gate as the raster model.
- **Wide-node pads (v1):** 2D anchor pads become high-radius nodes; a `faces`
  field is reserved for a future combinatorial-complex (2-cell / ETNN-style,
  [arXiv:2405.15429](https://arxiv.org/abs/2405.15429)) extension.

## Schema

- **Node** (`MechNode`): `id`, position `x, y`, half-width `r`, `degree`, `roles`
  (subset of `input`/`output`/`support`), equivariant direction `vec` (applied
  load or desired output motion), `fixed`.
- **Edge** (`MechEdge`): endpoints `u, v`, width `w`, invariant `length`
  (Euclidean), `curve_length` (arc length; a curvature proxy), and optional
  `polyline`/`radii` (faithful skeleton geometry, used only for reconstruction —
  drop them for a lean GNN export via `save(..., drop_geometry=True)`).
- **Graph** (`MechGraph`): `nodes`, `edges`, `shape`, `globals` (goal-conditioning
  scalars, incl. the 15-vector), `label` (verified FEA response), `meta` (family,
  type, hashes), `faces` (reserved).

Geometry is stored in **pixels** (faithful). `to_arrays` emits model-ready
tensors **normalized** by the domain size (matches the project's normalized units).

## Model-ready tensors (`to_arrays` / `to_pyg`)

| tensor | shape | kind | contents |
|---|---|---|---|
| `pos` | (N, 2) | **equivariant** | node coordinates (normalized) |
| `node_scalar` | (N, 6) | invariant | `r, degree, is_input, is_output, is_support, vec_mag` |
| `node_vec` | (N, 2) | **equivariant** | unit load/motion direction (0 if none) |
| `edge_index` | (2, E) | — | connectivity (both directions if undirected) |
| `edge_scalar` | (E, 3) | invariant | `w, length, curvature` |
| `graph_y` | (G,) | — | graph-level goal/label scalars |

`to_pyg` returns a PyTorch Geometric `Data` if `torch_geometric` is installed
(`pos`, `x`, `edge_index`, `edge_attr`, `y`, `node_vec`), else the same fields as
a torch/numpy tensor dict — so it drops straight into an EGNN or a discrete
flow-matching graph generator ([DeFoG](https://arxiv.org/pdf/2410.04263)).

## Usage

```python
from src.graph import from_raster, to_pyg, roundtrip_dice, MechGraph

# any density (any family) -> canonical graph, with BVP tagged from conditioning
g = from_raster(density, cond=cond, cond_channels=names,
                scalars=scalars, scalar_names=scalar_names, meta=meta)

data = to_pyg(g)                        # PyG Data (or tensor dict)
fidelity = roundtrip_dice(density, g)   # graph -> raster fidelity (for FEA re-check)

g.save("mech.json")                     # lossless JSON
g2 = MechGraph.load("mech.json")
```

CLI (uniform sweep + tensor shapes + serialization check):

```bash
python scripts/mech_to_graph.py --cache data/v1_broad_cache_128 \
    --per-family 25 --dump runs/graph_export --arrays
```

## Validation (measured)

Over 25 designs per family at 128px, one code path:

| family | round-trip Dice | nodes | edges | ports tagged |
|---|---:|---:|---:|---:|
| A — SIMP (continuum) | 0.956 | 8.0 | 8.8 | 100% |
| B — flexure | 0.974 | 6.3 | 5.3 | 100% |
| C — MMC | 0.974 | 8.0 | 8.1 | 100% |
| D — truss | 0.949 | 9.0 | 10.2 | 100% |
| E — linkage | 0.965 | 5.6 | 4.9 | 100% |

Plus (unit tests): **SE(2)-equivariance verified** (invariant features unchanged
under rotation; `node_vec` rotates with R), **serialization lossless**, tensor
shapes correct, PyG/torch export works. See
[figure](figures_128px/unified_graph_converter.png) for the schema across families.

## Dependencies

Converter (`from_raster`/`to_raster`): `scikit-image`, `sknw`, `scipy`, `numpy`.
Schema, features, serialization, PyG export work without them. PyG `Data` output
additionally needs `torch` + `torch_geometric` (both optional; falls back to a
tensor dict).

---

# The rank-2 lift: a combinatorial complex (ETNN)

*Code: [`src/graph/complex.py`](../src/graph/complex.py). Tests:
[`tests/test_mech_complex.py`](../tests/test_mech_complex.py).*

A mechanism is not only its struts. The **regions** carry physics too: the closed
loop that makes a flexure compliant one way and stiff another, and the solid pad
where struts fuse into an anchor. A plain GNN can only reach that information by
walking a loop hop by hop; a topological network reads it in one step, because
the loop is an explicit cell.

`build_cells` lifts the graph to a **combinatorial complex** and `to_cc_arrays`
emits the tensors an **E(n)-Equivariant Topological Neural Network** consumes
([ETNN, arXiv:2405.15429](https://arxiv.org/abs/2405.15429)).

## Why a combinatorial complex, not a simplicial or cell complex

A CC is a triple `(S, X, rk)`: a ground set, a set of cells that are non-empty
subsets of it, and a rank function monotone under inclusion. That is *all* it
requires — no gluing condition, no requirement that a cell be a topological disk.
That freedom is the point here: an anchor pad is a **blob**, and forcing it into
a cell complex would misrepresent it. A simplicial complex would be worse still,
since it would demand every face be a simplex.

| rank | cell | built from |
|---|---|---|
| 0 | junction | skeleton nodes |
| 1 | strut | skeleton edges (2-node sets) |
| 2 | **hole** | a bounded face of the planar skeleton — a closed loop of struts |
| 2 | **pad** | fused wide nodes: a 2D region where struts merge |

## Equivariance is structural, not learned

Following ETNN, a rank-2 cell stores **invariant scalars only**
(`area, perimeter, n_nodes, is_hole, is_pad, circularity`) and its **position is
derived** as the mean of its member rank-0 positions. Rotate the mechanism and
every cell position transforms as `R p` automatically while no cell feature
changes. There is nothing to get wrong at training time, and the unit tests
check exactly this across all three ranks.

## Correctness: Euler's formula, not eyeballing

Face enumeration walks directed half-edges of the planar embedding, always
turning clockwise, and drops each connected component's outer face. Rather than
trusting that, the tests check the threshold-free invariant

> a planar graph with `V` nodes, `E` struts and `C` components has exactly
> `E - V + C` bounded faces.

`euler_check(graph)` reports it for any graph. Measured over **200 real designs,
40 per family, at 128px: 100% agreement**.

| family | nodes | edges | holes | pads | round-trip Dice | Euler ok |
|---|---:|---:|---:|---:|---:|---:|
| A — SIMP (continuum) | 8.4 | 9.4 | 1.95 | 0.53 | 0.925 | 100% |
| B — flexure | 6.0 | 5.0 | 0.00 | 0.00 | 0.977 | 100% |
| C — MMC | 7.7 | 7.8 | 1.15 | 0.55 | 0.970 | 100% |
| D — truss | 9.6 | 11.2 | 2.65 | 0.12 | 0.972 | 100% |
| E — linkage | 4.1 | 3.3 | 0.20 | 0.10 | 0.926 | 100% |

Read honestly: the rank-2 lift adds a lot for **trusses and SIMP continua**
(1.9–2.7 holes each) and **nothing at all for family B**, whose flexures come out
as exact trees (`E = V - 1`). The lift is worth its complexity only for the
loop-bearing families, and that is a measurement, not an assumption.

### A representation bug this exposed

A closed ring — the archetypal compliant flexure loop — has *no junction*, so the
skeleton graph represented it as **one node with a self-loop**. The enclosed
region was invisible and a GNN would have seen a single point instead of a ring.
Self-loops are now cut at 1/3 and 2/3 arc length into a simple 3-cycle with
identical pixel coverage. Face traversal also orders half-edges by the skeleton's
local **tangent** rather than the straight chord, so parallel struts (a "theta"
shape) and curved struts order correctly.

## Tensors (`to_cc_arrays`)

| key | shape | kind | contents |
|---|---|---|---|
| `pos_0` / `x_0` / `vec_0` | (N,2)/(N,6)/(N,2) | equi / inv / equi | nodes |
| `pos_1` / `x_1` | (E,2)/(E,3) | equi / inv | struts (position = midpoint) |
| `pos_2` / `x_2` | (F,2)/(F,6) | equi / inv | cells (position = member mean) |
| `inc_01`, `inc_12`, `inc_02` | (2,·) | — | incidence between ranks |
| `adj_00`, `adj_22` | (2,·) | — | within-rank adjacency |

## Next step (deliberately not done yet)

The rank-2 cells are **built and validated but not yet consumed by a model** — the
generator below uses the rank-0/1 graph. Wiring `inc_12` into the message passing
is the natural follow-up, and it should be judged by whether pass@K improves on
the loop-bearing families, where the table above says the information actually
exists.
