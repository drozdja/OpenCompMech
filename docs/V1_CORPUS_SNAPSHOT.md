# CORRECTION (audit 2026-07-21, authoritative; supersedes the historical run log below)

> The remainder of this file preserves time-stamped generator diagnostics,
> including values recorded while generation was still live. It is useful for
> the debugging narrative but not as the current corpus fact sheet. Cite the
> correction below and the frozen manifest instead.

The port-access change **relaxed the acceptance policy; it did NOT fix the
generator**. Embedded-port designs are not more usable — data was recovered, not
interfaces. Consequences from a direct manifest audit:

- The four "recovered" families are almost entirely INACCESSIBLE: inverter
  0/1711, crusher 0/1180, rr_compound_lever 0/1994, rr_lever 13/1964. Restored
  inverting/amplifier coverage is real for a MATHEMATICAL corpus, not for the
  real-world-usable objective.
- Uniqueness by BINARY topology: 20,422 at 128², 20,415 at 64² (vs 20,429 float
  rows). Accessible 8,857 / embedded 11,572.
- Motion coverage is 11/12 joint classes (no `inverting_force_amp`); only 6
  `inverting_displacement_amp`. force_amp 368 (52 accessible); displacement_amp
  2,042 (243 accessible).
- The per-family counts below are off by ~7 (analyzer raced live generation;
  mmc_opt is 688, not 681). Trust the manifest, not this table.
- The "lineage-safe" split is effectively IID (every spec/lineage unique) — it
  does NOT guard against template/topology leakage. Split by binary-topology
  clusters at 64² instead.
- Physical usability untested (no dims/material/thickness/load/process/yield/
  buckling/contact/nonlinear stroke); ~69% have linear-FEA input displacement
  >10% of the domain under the normalized load.

**Correct characterization: a corrected SYNTHETIC mechanism corpus — not yet a
diverse, practically-usable mechanism dataset.** Recommended packaging: keep all
20,429 as `v1_broad` (pretraining/research); publish the 8,857 accessible as a
distinct `v1_interface_accessible`; balance by joint motion class + generator;
port-placement redesign and identical-spec cross-family solutions are the real
open work.

---

# v1 corpus — FINAL 2026-07-21 ~17:00 (numbers below are the raw run; see correction above)

`data/v1_pool_20260720`, generated all day on the corrected code (k·ddᵀ springs,
strict grounded gate, port-access as metadata). Manifest:
`data/v1_pool_20260720/manifest.json`.

- **20,429 unique valid designs, 0 byte-identical duplicates** (15 families).
  The offset sample-id strategy (start-ids 0 / 2M / 4M / keep-alive rotations)
  produced zero dups across all passes.
- **All four dead families recovered** by the port-access fix: inverter 1711
  (was 0), rr_lever 1964 (0), rr_compound_lever 1994 (0), crusher 1180 (0).
- **Motion-class coverage transformed** vs the 6k mid-run snapshot:
  inverting 954→**4905** (5×), displacement_amp 173→**2042** (12×). Final
  magnitude: transmitting 11865, displacement_reducer 6147, displacement_amp
  2042, force_amp 368. transfer: forwarding 9056, redirecting 6461, inverting
  4905.
- **Diversity (nnDice median):** most families 0.43–0.67 (healthy cross-topology
  spread); only inverter stays convergent (0.907) — the per-spec multimodality
  gap. `unique_spec_hashes == records == 20429`: every design is its own spec
  (one solution per spec), so multimodality must still come from cross-family or
  QD, not this corpus.
- **Port accessibility:** 43% accessible (8856) / 57% embedded (11566), now a
  RECORDED axis. A manufacturable-interface release subset = filter
  `port_interface.passed` (≈8.9k designs) or regenerate with
  `MECH_STRICT_PORT_ACCESS=1`.

Per-family (valid | nnDice | access%): crank_slider 154|0.43|100 ·
crusher 1180|0.67|0 · fact_rotation 2766|0.55|45 · fact_translation 800|0.50|100
· gripper 1764|0.72|86 · gs_opt 2658|0.66|59 · gs_truss 1028|0.55|34 ·
inverter 1711|0.91|0 · mmc_opt 681|0.76|36 · random 52|0.40|100 ·
rr_bridge_amp 441|0.58|100 · rr_compound_lever 1994|0.66|0 · rr_four_bar
2765|0.50|73 · rr_lever 1964|0.54|1 · rr_slider_crank 464|0.51|100.

---

# v1 corpus snapshot — 2026-07-21 ~08:2x (mid-run)

Snapshot of `data/v1_pool_20260720` while the comprehensive run is still
generating (scripts/analyze_corpus.py). Numbers grow through the day.

**6137 valid** across 12 families so far. The comprehensive run's remaining
stages (rr_lever, crusher, rr_compound_lever, gripper, gs_truss, ...) had not
landed yet at snapshot time.

| family | valid | nnDice(med) | access% |
|---|---:|---:|---:|
| rr_four_bar | 1113 | 0.506 | 100 |
| gs_opt | 1061 | 0.612 | 100 |
| gripper | 819 | 0.709 | 100 |
| fact_translation | 800 | 0.495 | 100 |
| fact_rotation | 638 | 0.533 | 100 |
| rr_slider_crank | 464 | 0.509 | 100 |
| rr_bridge_amp | 441 | 0.579 | 100 |
| **inverter** | **270** | 0.929 | **0** |
| gs_truss | 189 | 0.545 | 100 |
| crank_slider | 154 | 0.432 | 100 |
| mmc_opt | 136 | 0.840 | 100 |
| random | 52 | 0.403 | 100 |

- `nnDice(med)`: median nearest-neighbour Dice within the family (lower = more
  topological diversity). random / crank_slider / fact / rr families are
  healthiest (0.40–0.55); gs_opt 0.61; **inverter 0.93 and mmc 0.84 are
  near-convergent** — matches the SIMP-multistart finding (one basin per BVP).
- `access%`: fraction whose ports are externally accessible. **inverter is 0%
  accessible — its ports are embedded BY DESIGN**, which is exactly why the hard
  port-access gate drove it to 0% yield. Recovered by demoting that gate to
  recorded metadata (commit 246c257).

**Motion-class balance (needs work):**
- magnitude: transmitting 4113 (67%), displacement_reducer 1817 (30%),
  displacement_amp 173 (3%), force_amp 34 (0.6%)
- transfer: forwarding 2668, redirecting 2515, inverting 954

Transmitting-heavy; amplifier classes scarce. The upcoming rr_lever / crusher /
mmc / rr_compound_lever stages target displacement_amp + grasping, so balance
improves through the run; final PACKAGING should subsample to even the classes
(the review's point).

**Read this way:** the corpus now has real cross-family topological diversity and
all motion classes represented; the two honest weak spots are (a) intra-family
convergence for SIMP families (inverter/mmc) — the per-spec multimodality
problem, still open; and (b) class imbalance — a packaging/subsampling fix.
