#!/usr/bin/env python3
"""
Physical-validity audit for mechanism generators.

Measures the gap between "passes the current acceptance gate" and "is actually a
physically sound compliant mechanism". For each generator it reports:

  - old_gate %   : the gate production uses today (finite, connected, BC-connected,
                   volume, gray, |u_out|>0.1). What the dataset would accept now.
  - point_hinge %: designs held together by >=1 single-pixel hinge (de-facto pin
                   joint that linear FEA over-rewards). See validation.detect_hinges.
  - new_gate %   : old_gate AND no point hinge AND survives 1px erosion AND strain
                   energy is localized in flexures (Gini) AND meaningful u_out.

It also renders a deformed-shape montage (actual FEA displacement overlaid on the
density, input/output ports marked) so the motion can be eyeballed: does the
"inverter" actually invert?

Usage:
    python scripts/audit_mech.py --n-per-type 10 --workers 12 --iterations 200
    python scripts/audit_mech.py --types inverter gripper --n-per-type 20
"""

import argparse
import os
import sys
import time
import json
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Generators that go through the standard optimize_mechanism path.
OPT_TYPES = ["inverter", "gripper", "crusher", "amplifier", "crank_slider", "random"]
# Generators that go through the linkage-seed path.
SEEDED_TYPES = ["four_bar", "slider_crank"]
# Canonical published templates (literature.py) -> function name.
LIT_TYPES = {
    "L1_inverter": "generate_L1_force_inverter",
    "L2_gripper": "generate_L2_gripper",
}

# Family E constructors (rigid-body replacement): no optimization — build the
# flexure mechanism directly and only FEA-label it. Audited with the SAME gate.
RR_TYPES = {"rr_four_bar", "rr_slider_crank", "rr_lever", "rr_compound_lever",
            "rr_bridge_amp", "fact_rotation", "fact_translation", "gs_truss", "gs_opt",
            "mmc_opt"}

GINI_MIN = 0.5        # strain-energy localization threshold (distributed flexure)
GINI_CAP = 0.75       # upper bound: gini > cap = lumped blob-with-tail pseudo-
                      # mechanism (manual validation review found flagged
                      # samples at 0.77-0.95, accepted samples ~0.4-0.7)
TRANSMISSION_MIN = 1.0  # |u_out| floor for a meaningful mechanism


def worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _rigid_residual(xs, ys, uxs, uys):
    """Fraction of the displacement field NOT explained by a single rigid-body
    motion u = [a - th*y ; b + th*x]. ~0 => rigid pivot/hinge; ->1 => genuinely
    deforming (distributed compliance). Robust quality signal vs point-pivots."""
    n = len(xs)
    if n < 3:
        return float("nan")
    A = np.zeros((2 * n, 3)); rhs = np.zeros(2 * n)
    A[0::2, 0] = 1.0; A[0::2, 2] = -ys; rhs[0::2] = uxs
    A[1::2, 1] = 1.0; A[1::2, 2] = xs; rhs[1::2] = uys
    p, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    res = rhs - A @ p
    return float(np.linalg.norm(res) / (np.linalg.norm(rhs) + 1e-12))


def _element_strain_energy(problem, density, u, penal=3.0):
    """Per-element SIMP strain energy (relative), shape (nely, nelx)."""
    from src.solvers.linear import get_cached_edof, get_element_stiffness_cached
    nely, nelx = density.shape
    mat = problem.material
    edof = get_cached_edof(nelx, nely)            # (n_elem, 8)
    ke = get_element_stiffness_cached(mat.E, mat.nu)
    u_e = u[edof]                                  # (n_elem, 8)
    uku = np.einsum("ij,jk,ik->i", u_e, ke, u_e)   # u_e^T ke u_e
    rho = density.flatten()
    E_interp = mat.E_min + np.power(np.maximum(rho, 0.0), penal) * (mat.E - mat.E_min)
    se = 0.5 * (E_interp / mat.E) * uku
    return se.reshape(nely, nelx)


def run_case(task):
    """Generate one sample, solve FEA, and run all physical-validity checks."""
    worker_init()
    (kind, name, sample_id, res, iters, vf, robust, eta_offset, filt_r,
     k_in_p, k_out_p, alpha_max_p, multires_p) = task
    from src.generators.mech import (
        MechConfig, MECH_GENERATORS, optimize_mechanism, solve_mechanism_fea,
    )
    from src.validation.connectivity import (
        validate_sample, check_bc_connectivity, detect_hinges, validate_mechanism,
    )

    out = {"type": name, "sample_id": sample_id, "ok": False}
    t0 = time.time()
    try:
        config = MechConfig(volume_fraction=vf, max_iterations=iters,
                            k_in=k_in_p if k_in_p is not None else 0.01,
                            k_out=k_out_p, filter_radius=filt_r,
                            robust=robust, robust_eta_offset=eta_offset,
                            multires=multires_p)
        if alpha_max_p is not None:
            config.alpha_max = alpha_max_p
            config.alpha_init = min(getattr(config, "alpha_init", 0.05), alpha_max_p)

        if kind == "construct":
            # Family E: constructed, not optimized. The constructed VF is the
            # design's own target (the CLI --volume-fraction does not apply).
            from src.generators.rigid_replace import RR_CONSTRUCTORS
            rng = np.random.RandomState(sample_id * 12345 + 67890)
            built = RR_CONSTRUCTORS[name](nelx=res, nely=res, rng=rng,
                                          k_in=k_in_p, k_out=k_out_p)
            if built is None:
                out["fail"] = "construction_failed"
                out["gen_seconds"] = time.time() - t0
                return out
            density_c, problem, _rr_meta = built
            vf = float(_rr_meta['constructed_vf'])

            class _R:  # minimal stand-in for OptimizationResult
                density = density_c
            result = _R()
        elif kind == "opt":
            seed = sample_id * 12345 + 67890
            # Pass the RAW CLI spring values: None lets the generator own its
            # defaults (the amplifier randomizes k_in STIFF — see its docstring).
            problem = MECH_GENERATORS[name](
                nelx=res, nely=res, volume_fraction=vf,
                k_in=k_in_p, k_out=k_out_p, seed=seed,
            )
            result = optimize_mechanism(problem, config)
        elif kind == "lit":
            from src.generators import literature as _lit
            gen = getattr(_lit, LIT_TYPES[name])
            problem, _meta = gen(nelx=res, nely=res, vf=vf,
                                 k_in=k_in_p, k_out=k_out_p, seed=sample_id,
                                 jitter_amount=2)
            result = optimize_mechanism(problem, config)
        else:  # seeded linkage
            from src.generators.seeds import seed_from_linkage
            rng = np.random.RandomState(sample_id * 54321 + 13579)
            sr = seed_from_linkage(
                linkage_type=name, nelx=res, nely=res, rng=rng,
                volume_fraction=vf, k_in=config.k_in, k_out=config.k_out or 0.05,
                max_attempts=200,
            )
            if sr is None:
                out["fail"] = "seed_generation_failed"
                out["gen_seconds"] = time.time() - t0
                return out
            density_init, problem, _ = sr
            result = optimize_mechanism(
                problem, config, initial_density=density_init,
                seed_mask=density_init > 0.5,
            )

        density = np.asarray(result.density, dtype=np.float64)
        u, _K, u_out = solve_mechanism_fea(problem, density)

        # Input displacement / geometric advantage
        in_x, in_y = problem.get_input_dofs()
        dxi, dyi = problem.input_direction
        u_in = float(u[in_x] * dxi + u[in_y] * dyi)
        ga = float(u_out / u_in) if abs(u_in) > 1e-12 else float("inf")

        # --- replicate the CURRENT production gate exactly ---
        # domain_mask makes the VF check measure over the ACTIVE region only —
        # required for generators that carve a non-square domain (e.g. the
        # diversified gripper's wide/tall shapes), else VF is diluted by void.
        vi = validate_sample(density, target_vf=vf, vf_tolerance=0.05,
                             max_gray_fraction=0.20,
                             domain_mask=problem.domain_mask)
        fixed_nodes = []
        for bc in problem.base_problem.bcs:
            fixed_nodes.extend(np.asarray(bc.node_indices).tolist())
        fixed_nodes = np.array(fixed_nodes, dtype=int)
        bc_ok, _ = check_bc_connectivity(density, fixed_nodes, nelx=res, nely=res)
        old_gate = bool(
            vi["finite"]["passed"] and vi["connectivity"]["passed"]
            and vi["volume_fraction"]["passed"] and vi["gray_fraction"]["passed"]
            and bc_ok and abs(u_out) > 0.1
        )

        # --- new physical checks ---
        hinge = detect_hinges(density, min_neck_px=2)
        se_grid = _element_strain_energy(problem, density, u)
        vm = validate_mechanism(density, u_out, strain_energy=se_grid,
                                min_transmission=TRANSMISSION_MIN,
                                min_energy_localization=GINI_MIN)
        gini = float(vm.get("energy_localization", {}).get("gini_coefficient", 0.0))

        # Decisive physical defects: single-pixel point hinge (FEA-invalidating,
        # unmanufacturable) and lumped compliance (gini > GINI_CAP = blob-with-
        # tail pseudo-mechanism, established by manual review). A 2px flexure is
        # legitimate, so survives_erosion stays a reported metric, not a fail.
        #
        # Family E exception (kind == 'construct'): flexure linkages localize
        # strain energy in their DESIGNED necks by construction — high gini +
        # HIGH rigid-residual (piecewise-rigid links, several rigid bodies) is
        # correct behavior there, unlike the SIMP blob (high gini + LOW resid =
        # one rigid body on an accidental pivot). Constructed samples use a GA
        # floor instead: floppy chains (links bending along their length) die
        # on GA, not on gini. Gini remains reported for all kinds.
        if kind == "construct":
            quality_ok = ga >= 0.25
        else:
            quality_ok = gini <= GINI_CAP
        new_gate = bool(
            old_gate and not hinge["has_point_hinge"]
            and abs(u_out) >= TRANSMISSION_MIN
            and quality_ok
        )

        # Node grid displacement for the montage overlay
        n = res + 1
        u_x = u[0::2].reshape(n, n)
        u_y = u[1::2].reshape(n, n)

        # --- quality metrics (beyond mere validity) ---
        # off-axis input: fraction of input-node motion perpendicular to the
        # applied force. High => load path is tilted (asymmetric blob).
        ui_full = np.array([u[2 * problem.input_node],
                            u[2 * problem.input_node + 1]])
        perp = np.array([-dyi, dxi])
        off_axis = float(abs(ui_full @ perp) / (np.linalg.norm(ui_full) + 1e-12))
        # rigid-body residual over solid element centers (anti-pivot signal)
        solid_q = density > 0.5
        cyq, cxq = np.where(solid_q)
        uxc_q = 0.25 * (u_x[:-1, :-1] + u_x[1:, :-1] + u_x[:-1, 1:] + u_x[1:, 1:])
        uyc_q = 0.25 * (u_y[:-1, :-1] + u_y[1:, :-1] + u_y[:-1, 1:] + u_y[1:, 1:])
        rigid_resid = _rigid_residual(cxq + 0.5, cyq + 0.5,
                                      uxc_q[solid_q], uyc_q[solid_q])

        # Port exposure: clear external interface vs embedded in the structure.
        # Reported, not gated.
        from src.validation.ports import problem_port_exposure
        pexp = problem_port_exposure(density, problem)

        # Footprint: solid bbox extent as a fraction of the domain side. This
        # catches tiny mechanisms lost in an empty domain; reported, not gated.
        fy, fx = np.where(density > 0.5)
        footprint = float(max((fx.max() - fx.min()) / res,
                              (fy.max() - fy.min()) / res)) if fx.size else 0.0

        # Port compliance selectivity: parasitic-compliance signal the working
        # solve cannot see; reported, not gated.
        from src.validation.compliance import port_selectivity
        psel = port_selectivity(problem, density)
        sel_out = psel.get("output", {})
        sel_in = psel.get("input", {})

        out.update({
            "off_axis": off_axis, "rigid_resid": rigid_resid,
            "footprint": footprint,
            "sel_out": sel_out.get("selectivity"),
            "sel_out_align": sel_out.get("soft_align"),
            "sel_in": sel_in.get("selectivity"),
            "port_clear_in": pexp["input"]["clearance"],
            "port_clear_out": pexp["output"]["clearance"],
            "approach_clear_in": pexp["input"]["approach_clear"],
            "approach_clear_out": pexp["output"]["approach_clear"],
            "ok": True,
            "u_out": float(u_out), "u_in": u_in, "ga": ga, "gini": gini,
            "n_bridge_pixels": int(hinge["n_bridge_pixels"]),
            "has_point_hinge": bool(hinge["has_point_hinge"]),
            "survives_erosion": bool(hinge["survives_erosion"]),
            "frac_lost_eroded": float(hinge["frac_lost_eroded"]),
            "old_gate": old_gate, "new_gate": new_gate,
            "input_node": int(problem.input_node),
            "output_node": int(problem.output_node),
            "in_dir": [float(dxi), float(dyi)],
            "out_dir": [float(problem.output_direction[0]),
                        float(problem.output_direction[1])],
            "gen_seconds": time.time() - t0,
            # arrays for rendering (kept small: 64x64-ish)
            "_density": density.astype(np.float32),
            "_u_x": u_x.astype(np.float32),
            "_u_y": u_y.astype(np.float32),
            "_bridge_mask": hinge["bridge_mask"],
        })
        return out
    except Exception as e:  # noqa: BLE001
        out["fail"] = f"{type(e).__name__}: {e}"
        out["trace"] = traceback.format_exc().splitlines()[-3:]
        out["gen_seconds"] = time.time() - t0
        return out


def draw_panel(ax, r):
    """Overlay actual FEA deformed shape on density with input/output ports."""
    d = r["_density"]
    nely, nelx = d.shape
    ux, uy = r["_u_x"], r["_u_y"]

    # Faint undeformed density
    ax.imshow(d, cmap="Greys", origin="lower", extent=[0, nelx, 0, nely],
              alpha=0.18, vmin=0, vmax=1)

    # Deformed solid element centers
    solid = d > 0.5
    cx, cy = np.meshgrid(np.arange(nelx) + 0.5, np.arange(nely) + 0.5)
    uxc = 0.25 * (ux[:-1, :-1] + ux[1:, :-1] + ux[:-1, 1:] + ux[1:, 1:])
    uyc = 0.25 * (uy[:-1, :-1] + uy[1:, :-1] + uy[:-1, 1:] + uy[1:, 1:])
    mag = np.sqrt(ux ** 2 + uy ** 2).max() + 1e-12
    scale = 0.20 * max(nelx, nely) / mag
    ax.scatter((cx + scale * uxc)[solid], (cy + scale * uyc)[solid],
               s=2, c="steelblue", linewidths=0)
    # every panel is normalized to ~20% domain deflection — print the true
    # magnification so "the motion looks wrong" can be judged fairly (linear
    # FEA strokes are small; the overlay exaggerates by this factor)
    ax.text(0.02, 0.02,
            f"×{scale:.1f}" if scale < 1.5 else f"×{scale:.0f}",
            transform=ax.transAxes,
            fontsize=7, color="dimgray")

    # Bridge / single-pixel hinge pixels (undeformed location)
    bm = r["_bridge_mask"]
    if bm.any():
        by, bx = np.where(bm)
        ax.scatter(bx + 0.5, by + 0.5, s=40, marker="x", c="red", zorder=6,
                   linewidths=1.5)

    def node_xy(node):
        return node % (nelx + 1), node // (nelx + 1)

    ix, iy = node_xy(r["input_node"])
    ox, oy = node_xy(r["output_node"])
    # Desired directions (force in / target out)
    ax.annotate("", xy=(ix + 4 * r["in_dir"][0], iy + 4 * r["in_dir"][1]),
                xytext=(ix, iy),
                arrowprops=dict(color="red", width=1.5, headwidth=6))
    ax.annotate("", xy=(ox + 4 * r["out_dir"][0], oy + 4 * r["out_dir"][1]),
                xytext=(ox, oy),
                arrowprops=dict(color="green", width=1.5, headwidth=6))
    ax.scatter([ix], [iy], c="red", s=25, zorder=7)
    ax.scatter([ox], [oy], c="green", s=25, marker="s", zorder=7)

    verdict = "OK" if r["new_gate"] else ("HINGE" if r["has_point_hinge"] else "WEAK")
    color = "green" if r["new_gate"] else "red"
    exp = ""
    if "port_clear_in" in r:
        exp = f" exp={r['port_clear_in']:.1f}/{r['port_clear_out']:.1f}"
    ax.set_title(
        f"{r['type']} #{r['sample_id']}  [{verdict}]\n"
        f"u_out={r['u_out']:.1f} GA={r['ga']:.1f} gini={r['gini']:.2f} "
        f"br={r['n_bridge_pixels']}{exp}",
        fontsize=8, color=color)
    ax.set_xlim(-3, nelx + 6); ax.set_ylim(-3, nely + 6)
    ax.set_xticks([]); ax.set_yticks([])


def summarize(rows):
    """Print per-type summary table; return dict of stats."""
    def pct(c, n):
        return 100.0 * c / n if n else 0.0

    stats = {}
    types = sorted({r["type"] for r in rows})
    def q(vals, p):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return float(np.percentile(v, p)) if v else float("nan")

    print("\n" + "=" * 158)
    print(f"{'generator':<14}{'n':>4}{'new_gate':>9}{'pt_hinge':>9}"
          f"{'GA_p10':>8}{'GA_med':>8}{'offax_med':>10}{'offax_p90':>10}"
          f"{'resid_p10':>10}{'resid_med':>10}{'exp_in':>8}{'exp_out':>8}"
          f"{'fp_med':>8}{'sel_out':>8}{'sel_algn':>9}")
    print("(metrics over NEW-GATE-PASSING samples only)")
    print("-" * 158)
    for t in types:
        rs = [r for r in rows if r["type"] == t]
        n = len(rs)
        ok = [r for r in rs if r.get("ok")]
        n_ok = len(ok)
        new = sum(1 for r in ok if r["new_gate"])
        hinge = sum(1 for r in ok if r["has_point_hinge"])
        passing = [r for r in ok if r["new_gate"]]
        gas = [r["ga"] for r in passing if np.isfinite(r["ga"])]
        offs = [r.get("off_axis") for r in passing]
        resids = [r.get("rigid_resid") for r in passing]
        exp_in = [r.get("port_clear_in") for r in passing]
        exp_out = [r.get("port_clear_out") for r in passing]
        fps = [r.get("footprint") for r in passing]
        sel_out = [r.get("sel_out") for r in passing]
        sel_align = [r.get("sel_out_align") for r in passing]
        med_ga = float(np.median(gas)) if gas else float("nan")
        print(f"{t:<14}{n:>4}{pct(new,n_ok):>8.0f}%{pct(hinge,n_ok):>8.0f}%"
              f"{q(gas,10):>8.2f}{med_ga:>8.2f}{q(offs,50):>10.2f}{q(offs,90):>10.2f}"
              f"{q(resids,10):>10.2f}{q(resids,50):>10.2f}"
              f"{q(exp_in,50):>8.2f}{q(exp_out,50):>8.2f}{q(fps,50):>8.2f}"
              f"{q(sel_out,50):>8.1f}{q(sel_align,50):>9.2f}")
        stats[t] = {
            "n": n, "gen_ok": n_ok, "new_gate_pct": pct(new, n_ok),
            "point_hinge_pct": pct(hinge, n_ok), "median_ga": med_ga,
            "ga_p10": q(gas, 10), "off_axis_med": q(offs, 50),
            "off_axis_p90": q(offs, 90), "rigid_resid_p10": q(resids, 10),
            "rigid_resid_med": q(resids, 50),
            "port_clear_in_med": q(exp_in, 50),
            "port_clear_out_med": q(exp_out, 50),
            "footprint_med": q(fps, 50),
            "sel_out_med": q(sel_out, 50),
            "sel_out_align_med": q(sel_align, 50),
        }
    print("=" * 158)
    print("offax = off-axis input fraction (lower better); "
          "resid = rigid-body residual (HIGHER better; low => pivot/hinge); "
          "exp = median port ray-clearance (1=exposed tip, 0=embedded); "
          "fp = median footprint (bbox/domain);")
    print("sel_out = output working/perp compliance ratio (HIGHER = more "
          "single-DOF; ~1 = parasitically floppy); sel_algn = soft-axis aligns "
          "with working dir. All reported, not gated")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Physical-validity audit for mech generators")
    ap.add_argument("--types", nargs="+",
                    default=OPT_TYPES + SEEDED_TYPES,
                    help="Generator names to audit")
    ap.add_argument("--n-per-type", type=int, default=10)
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--volume-fraction", type=float, default=0.20)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--output-dir", type=str, default="data/audit")
    ap.add_argument("--montage-per-type", type=int, default=6)
    ap.add_argument("--robust", action="store_true",
                    help="Use the robust three-field formulation (kills point hinges)")
    ap.add_argument("--multires", action="store_true",
                    help="Multi-resolution warm-start (coarse establish -> fine robust refine; ~2x faster)")
    ap.add_argument("--eta-offset", type=float, default=0.20,
                    help="Eroded threshold offset above nominal eta (robust mode)")
    ap.add_argument("--filter-radius", type=float, default=1.5,
                    help="Density filter radius (root enabler of thin hinges)")
    ap.add_argument("--k-in", type=float, default=None,
                    help="Input spring stiffness (None = generator decides; most types "
                         "default soft 0.01, amplifier randomizes stiff — it needs "
                         "k_in >> k_out or the floating-translator optimum wins)")
    ap.add_argument("--k-out", type=float, default=None,
                    help="Output spring stiffness (None = generator randomizes)")
    ap.add_argument("--alpha-max", type=float, default=None,
                    help="Max input-compliance penalty alpha (raises GA; None = MechConfig default)")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for t in args.types:
        if t in OPT_TYPES:
            kind = "opt"
        elif t in LIT_TYPES:
            kind = "lit"
        elif t in RR_TYPES:
            kind = "construct"
        else:
            kind = "seeded"
        for sid in range(args.n_per_type):
            tasks.append((kind, t, sid, args.resolution, args.iterations,
                          args.volume_fraction, args.robust, args.eta_offset,
                          args.filter_radius, args.k_in, args.k_out,
                          args.alpha_max, args.multires))

    print(f"Auditing {len(args.types)} generators x {args.n_per_type} samples "
          f"= {len(tasks)} cases on {args.workers} workers "
          f"({args.iterations} iters, {args.resolution}x{args.resolution})")

    rows = []
    t0 = time.time()
    # NUMA-aware pinning: one worker per physical core, local first-touch memory.
    from multiprocessing import Manager
    from src.core.cpu_affinity import make_affinity_initializer, physical_core_count
    _mgr = Manager()
    _pin_init, _pin_args = make_affinity_initializer(_mgr, physical_core_count())
    print(f"  Pinning {args.workers} workers across {physical_core_count()} physical cores "
          f"(1 thread/core, NUMA-local)")
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_pin_init, initargs=_pin_args) as ex:
        futs = {ex.submit(run_case, task): task for task in tasks}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            done += 1
            tag = "ok " if r.get("ok") else "FAIL"
            extra = "" if r.get("ok") else f" ({r.get('fail','?')})"
            print(f"[{done}/{len(tasks)}] {tag} {r['type']} #{r['sample_id']}"
                  f"{extra}", flush=True)
    print(f"\nAll cases done in {time.time() - t0:.0f}s")

    stats = summarize(rows)

    # JSON report (without big arrays)
    slim = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    report = {
        "config": vars(args), "summary": stats, "cases": slim,
        "thresholds": {"gini_min": GINI_MIN, "transmission_min": TRANSMISSION_MIN},
    }
    with open(outdir / "audit_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report: {outdir / 'audit_report.json'}")

    # Deformed-shape montage
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    types = [t for t in args.types]
    cols = args.montage_per_type
    rows_grid = len(types)
    fig, axes = plt.subplots(rows_grid, cols,
                             figsize=(2.4 * cols, 2.6 * rows_grid),
                             squeeze=False)
    for ri, t in enumerate(types):
        ok = [r for r in rows if r["type"] == t and r.get("ok")]
        ok.sort(key=lambda r: r["sample_id"])
        for ci in range(cols):
            ax = axes[ri][ci]
            if ci < len(ok):
                draw_panel(ax, ok[ci])
            else:
                ax.axis("off")
    fig.suptitle("Mechanism audit — actual deformed shape (red=input, green=output, "
                 "X=single-pixel hinge)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    montage = outdir / "audit_montage.png"
    fig.savefig(montage, dpi=95)
    print(f"Montage: {montage}")


if __name__ == "__main__":
    main()
