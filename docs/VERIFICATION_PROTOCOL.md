# Verification protocol and reporting contract

`config/evaluation_protocol.v1.json` is the frozen protocol for the current
case-study evaluation. It maps a proposal to the source mesh with nearest
neighbour replication, reconstructs the stored mechanism boundary-value
problem, and re-evaluates it with the sparse spring-aware solver.

The reported broad result is `functional_passed`; it does not require exposed
input/output approaches. `interface_passed` adds the separately recorded port
accessibility rule and is the only suitable result for an accessible
actuator/workpiece demonstration. Neither result is a claim of manufactured,
material-calibrated, or flight-qualified hardware.

The current model emits 64×64 densities while some source BVPs use 128×128
meshes. The verifier realizes those proposals on the source mesh, then exactly
masks forbidden cells before every geometry, connectivity, FEA, stress, and
visualization calculation. A 64×64 cell may straddle a curved or restricted
domain boundary; therefore pre-mask material in such mixed cells is recorded
in `density_transform` but is not a functional failure. Source-resolution
proposals with material outside their domain are still rejected. Fully
forbidden coarse cells are also rejected; this avoids treating an avoidable
leak as a harmless representation artifact.

At the 64×64 model grid, the `domain_mask` conditioning channel is fractional
source-area coverage rather than a Boolean mask. Before FEA, neural, retrieval,
and the 64px reference-contract baseline all receive the same deterministic
projection: zero cells with zero coverage and match active-domain volume using
coverage weights. The native source reference is deliberately not projected;
it is a full-resolution ceiling/check. This makes the 64px representation loss
visible instead of giving the neural method a private post-process advantage.

All verification is normalized 2D linear elasticity. It excludes dimensional
material selection, thickness, yield, fatigue, buckling, contact, friction,
manufacturing process constraints, and physical testing. Historical saved
stress arrays produced before the canonical vectorized stress-kernel correction
are legacy visualization data; case-study figures recompute stress through the
current verifier path.

Before a GPU evaluation, `scripts/verify_diff_fea.py` emits a versioned JSON
parity report for the differentiable guidance proxy: sparse-NumPy compliance
and finite-difference/autograd gradients on a deterministic canonical problem.
It fails closed in the evaluation launcher. This validates the proxy
implementation; it does not make the proxy an independent material or
nonlinear-physics certification.

The evaluation is goal-conditioned, not BVP-only inverse design: scalar inputs
include motion/performance values measured from each corpus reference. Results
must therefore be described as a within-distribution goal-conditioned audit.
