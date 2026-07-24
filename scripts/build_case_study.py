#!/usr/bin/env python3
"""Build an honest static case-study page and concise technical brief.

The builder consumes actual evaluation JSON and refuses to invent performance
numbers.  It is intentionally suitable for a portfolio/application artifact:
method, verification evidence, reproducibility provenance, and limitations are
all adjacent rather than hidden in a notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_facts(broad_manifest: Path | None, interface_manifest: Path | None) -> dict | None:
    """Read only explicit frozen manifests; never infer corpus counts from figures."""
    if broad_manifest is None or not broad_manifest.is_file():
        return None
    broad = load_json(broad_manifest)
    records = broad.get("records", [])
    broad_n = int(broad.get("n_unique", len(records)))
    if broad_n != len(records):
        raise ValueError("broad corpus manifest n_unique does not match records")
    types = sorted({str(record.get("type") or record.get("problem_type") or record.get("family"))
                    for record in records
                    if record.get("type") or record.get("problem_type") or record.get("family")})
    facts = {
        "broad_manifest": str(broad_manifest.resolve()),
        "broad_manifest_sha256": sha256_file(broad_manifest),
        "broad_records": broad_n,
        "generator_type_count": len(types),
        "generator_types": types,
    }
    if interface_manifest is not None and interface_manifest.is_file():
        interface = load_json(interface_manifest)
        interface_records = interface.get("records", [])
        accessible_n = int(interface.get("n_unique", len(interface_records)))
        if accessible_n != len(interface_records):
            raise ValueError("interface corpus manifest n_unique does not match records")
        derivation = interface.get("derivation", {}) or {}
        facts.update({
            "interface_manifest": str(interface_manifest.resolve()),
            "interface_manifest_sha256": sha256_file(interface_manifest),
            "source_interface_accessible_records": accessible_n,
            "source_interface_embedded_records": broad_n - accessible_n,
            "interface_derivation_matches_broad": (
                derivation.get("source_manifest_sha256") == facts["broad_manifest_sha256"]),
            "interface_selection_rule": derivation.get("selection_rule"),
            "interface_selected_metadata_sha256": derivation.get("selected_metadata_sha256"),
        })
    return facts


METHOD_ORDER = (
    "reference_native",
    "reference_64_contract_projection",
    "retrieval",
    "neural",
)


def fmt(metric, digits=3, show_n=False):
    if not metric or metric.get("mean") is None:
        return "—"
    value = f"{metric['mean']:.{digits}f} [{metric['lo']:.{digits}f}, {metric['hi']:.{digits}f}]"
    return f"{value}; n={metric.get('n', 0)}" if show_n else value


def table_html(summary: dict) -> str:
    rows = []
    for method in METHOD_ORDER:
        result = summary.get(method)
        if not result:
            continue
        metrics = result["metrics"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(method)}</td><td>{result['K']}</td>"
            f"<td>{fmt(metrics.get('functional_candidate_pass_rate', metrics.get('candidate_pass_rate')))}</td>"
            f"<td>{fmt(metrics.get('functional_pass_at_k', metrics.get('pass_at_k')))}</td>"
            f"<td>{fmt(metrics.get('interface_pass_at_k', metrics.get('pass_at_k')))}</td>"
            f"<td>{fmt(metrics.get('motion_class_match'), show_n=True)}</td>"
            f"<td>{fmt(metrics.get('vf_abs_error'), show_n=True)}</td>"
            f"<td>{fmt(metrics.get('nearest_train_dice'), show_n=True)}</td>"
            "</tr>")
    return ("<table><thead><tr><th>Method</th><th>K</th><th>functional candidate pass rate</th>"
            "<th>functional pass@K</th><th>strict-interface pass@K</th>"
            "<th>motion-class match†</th><th>VF abs. error†</th>"
            "<th>nearest-train Dice†‡</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def cohort_n(summary: dict) -> int:
    """All methods in a cohort share its selected-spec count."""
    for result in summary.values():
        return int(result.get("n_specs", 0))
    return 0


def copy_evidence(evaluation_path: Path, demo_dir: Path | None, evidence: Path,
                  protocol_path: Path | None) -> list[str]:
    """Package the exact lightweight evidence that the static page describes."""
    evidence.mkdir(parents=True, exist_ok=True)
    copied = []

    def copy_if_exists(source: Path, name: str | None = None):
        if source.is_file():
            target = evidence / (name or source.name)
            shutil.copy2(source, target)
            copied.append(str(target.name))

    copy_if_exists(evaluation_path)
    copy_if_exists(evaluation_path.with_name("results.md"))
    copy_if_exists(evaluation_path.with_name("split_audit.json"))
    copy_if_exists(evaluation_path.with_name("diff_fea_verification.json"))
    # Normally a native-reference mismatch aborts before a report is built,
    # but retain this diagnostic if an explicitly allowed audit produced one.
    copy_if_exists(evaluation_path.with_name("protocol_self_check.json"))
    # Preserve execution logs too: numerical warnings are evidence, not noise
    # to hide from an engineering-facing case study.
    for name in ("eval.log", "diff_fea_verification.log", "split_audit.log",
                 "split_audit.stderr", "verifier_regression.json",
                 "verifier_warnings.log"):
        copy_if_exists(evaluation_path.with_name(name))
    candidates = evaluation_path.parent / "candidates"
    if candidates.is_dir():
        # results.json stores paths relative to itself (`candidates/...`), so
        # retain the complete screened proposal set under the same layout.
        shutil.copytree(candidates, evidence / "candidates", dirs_exist_ok=True)
        copied.append("candidates/")
    if protocol_path:
        copy_if_exists(protocol_path, "evaluation_protocol.v1.json")
    if demo_dir:
        for name in ("gate.json", "provenance.json", "demo.json",
                     "generated_64.npy", "generated_verifier_rho.npy"):
            copy_if_exists(demo_dir / name)
    return copied


def table_markdown(summary: dict) -> list[str]:
    lines = ["| Method | K | functional candidate pass rate | functional pass@K | strict-interface pass@K | motion-class match† | VF abs. error† | nearest-train Dice†‡ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHOD_ORDER:
        result = summary.get(method)
        if not result:
            continue
        m = result["metrics"]
        lines.append(f"| {method} | {result['K']} | "
                     f"{fmt(m.get('functional_candidate_pass_rate', m.get('candidate_pass_rate')))} | "
                     f"{fmt(m.get('functional_pass_at_k', m.get('pass_at_k')))} | "
                     f"{fmt(m.get('interface_pass_at_k', m.get('pass_at_k')))} | "
                     f"{fmt(m.get('motion_class_match'), show_n=True)} | "
                     f"{fmt(m.get('vf_abs_error'), show_n=True)} | {fmt(m.get('nearest_train_dice'), show_n=True)} |")
    return lines


def response_html(summary: dict, strict_selection: bool) -> str:
    """Expose response quantities behind a pass@K headline.

    Quality values remain conditional on the selected passing candidate, which
    is why every cell carries its own valid n.  Reporting this separately
    avoids implying that a high pass@K alone proves target-function fidelity.
    """
    rows = []
    for method in METHOD_ORDER:
        result = summary.get(method)
        if not result:
            continue
        m = result["metrics"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(method)}</td>"
            f"<td>{fmt(m.get('ga_signed'), show_n=True)}</td>"
            f"<td>{fmt(m.get('output_alignment'), show_n=True)}</td>"
            f"<td>{fmt(m.get('output_selectivity'), show_n=True)}</td>"
            f"<td>{fmt(m.get('u_out_projected'), show_n=True)}</td>"
            f"<td>{fmt(m.get('bc_connected'), show_n=True)}/{fmt(m.get('mechanism_path_connected'), show_n=True)}</td>"
            "</tr>")
    label = "strict-interface" if strict_selection else "functional"
    return (f"<p>Values are measured on the selected {label}-passing candidate; "
            "BC/path connectivity is shown explicitly even though both are required by the functional gate.</p>"
            "<table><thead><tr><th>Method</th><th>signed GA†</th><th>output alignment†</th>"
            "<th>output selectivity†</th><th>output stroke†</th><th>BC/path connected†</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def response_markdown(summary: dict, strict_selection: bool) -> list[str]:
    label = "strict-interface" if strict_selection else "functional"
    lines = [f"Values are measured on the selected {label}-passing candidate; BC/path connectivity is included explicitly although both are functional-gate requirements.", "",
             "| Method | signed GA† | output alignment† | output selectivity† | output stroke† | BC/path connected† |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in METHOD_ORDER:
        result = summary.get(method)
        if not result:
            continue
        m = result["metrics"]
        lines.append(f"| {method} | {fmt(m.get('ga_signed'), show_n=True)} | "
                     f"{fmt(m.get('output_alignment'), show_n=True)} | "
                     f"{fmt(m.get('output_selectivity'), show_n=True)} | "
                     f"{fmt(m.get('u_out_projected'), show_n=True)} | "
                     f"{fmt(m.get('bc_connected'), show_n=True)}/{fmt(m.get('mechanism_path_connected'), show_n=True)} |")
    return lines


def paired_html(paired: dict) -> str:
    rows = []
    for cohort, metrics in paired.items():
        for name, metric in metrics.items():
            if metric.get("mean_difference") is None:
                continue
            rows.append("<tr>"
                        f"<td>{html.escape(cohort)}</td><td>{html.escape(name)}</td>"
                        f"<td>{metric['mean_difference']:.3f} [{metric['lo']:.3f}, {metric['hi']:.3f}]; n={metric['n']}</td>"
                        "</tr>")
    if not rows:
        return "<p>No paired comparison has an evaluable joint-passing denominator.</p>"
    return ("<table><thead><tr><th>Cohort</th><th>neural − retrieval metric</th>"
            "<th>mean difference [95% CI]; n</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def paired_markdown(paired: dict) -> list[str]:
    lines = ["| Cohort | neural − retrieval metric | mean difference [95% CI]; n |",
             "|---|---|---:|"]
    for cohort, metrics in paired.items():
        for name, metric in metrics.items():
            if metric.get("mean_difference") is not None:
                lines.append(f"| {cohort} | {name} | {metric['mean_difference']:.3f} "
                             f"[{metric['lo']:.3f}, {metric['hi']:.3f}]; n={metric['n']} |")
    return lines


def diversity_html(summary: dict) -> str:
    rows = []
    for method in ("retrieval", "neural"):
        result = summary.get(method)
        if not result:
            continue
        metrics = result["metrics"]
        rows.append("<tr>"
                    f"<td>{method}</td>"
                    f"<td>{fmt(metrics.get('passing_candidate_count'))}</td>"
                    f"<td>{fmt(metrics.get('passing_binary_unique_topologies'), show_n=True)}</td>"
                    f"<td>{fmt(metrics.get('passing_pairwise_topology_distance'), show_n=True)}</td>"
                    "</tr>")
    return ("<table><thead><tr><th>Method</th><th>mean verifier-passing candidates</th>"
            "<th>distinct passing binary topologies†</th>"
            "<th>pairwise topology distance among passing candidates†</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def diversity_markdown(summary: dict) -> list[str]:
    lines = ["| Method | mean verifier-passing candidates | distinct passing binary topologies† | pairwise topology distance among passing candidates† |",
             "|---|---:|---:|---:|"]
    for method in ("retrieval", "neural"):
        result = summary.get(method)
        if result:
            metrics = result["metrics"]
            lines.append(f"| {method} | {fmt(metrics.get('passing_candidate_count'))} | "
                         f"{fmt(metrics.get('passing_binary_unique_topologies'), show_n=True)} | "
                         f"{fmt(metrics.get('passing_pairwise_topology_distance'), show_n=True)} |")
    return lines


def _ood_pieces(ood: dict) -> dict:
    """Pull the honest framing numbers for a held-out generator-type eval."""
    summary = ood["summary"].get("predeclared_stratified_heldout_subset", {})
    self_check = ood.get("protocol_self_check", {})
    split = ood.get("split_assessment", {})
    holdout = ood["provenance"]["split_resolution"].get("holdout_value")
    if not holdout:
        # A --split-plan run records the held-out type in the plan, not in
        # split_resolution; recover it from the evaluated specs themselves.
        types = sorted({str(s.get("type")) for s in ood.get("per_spec", []) if s.get("type")})
        holdout = ", ".join(types) if types else "held-out type"
    ref64 = summary.get("reference_64_contract_projection", {}).get("metrics", {})
    ref64_pass = ref64.get("functional_pass_at_k", ref64.get("pass_at_k", {}))
    native = self_check.get("functional_native_reference_pass_rate")
    return {
        "summary": summary,
        "holdout": holdout,
        "n": cohort_n(summary),
        "ref64_functional_passk": fmt(ref64_pass),
        "native_self_check": ("not recorded" if native is None
                              else f"{native:.3f} (n={self_check.get('functional_native_reference_n', 0)})"),
        "verified": bool(split.get("checkpoint_split_provenance_verified")
                         and split.get("claim_bearing_test_partition")),
    }


def ood_section_html(ood: dict) -> str:
    p = _ood_pieces(ood)
    verified = ("verified generator-type holdout (checkpoint split provenance confirmed)"
                if p["verified"] else "generator-type holdout (provenance unconfirmed)")
    return (
        f"<section class='card'><h2>Out-of-distribution: held-out generator type "
        f"(<code>{html.escape(str(p['holdout']))}</code>, n={p['n']})</h2>"
        f"<p>A separate model was trained with a frozen split plan that excludes every "
        f"<code>{html.escape(str(p['holdout']))}</code> design; that type is the reserved test partition. "
        f"This is a {verified}, not the IID audit above. Native reference functional self-check on the "
        f"held-out type: {p['native_self_check']} — the held-out specifications are themselves valid. "
        f"These references all have embedded ports, so the strict-interface accessible cohort is empty by "
        f"construction and functional pass@K is the generalization signal.</p>"
        f"<p><b>Reading the result honestly:</b> the shared 64px representation contract passes only "
        f"<b>{p['ref64_functional_passk']}</b> functional pass@K for held-out references — the coarse raster "
        f"is itself a severe bottleneck for a fine-featured family — so a model score here compounds "
        f"representation loss with genuine type-novelty. Under that compounded bar, both the neural model "
        f"and the conditioning-retrieval baseline reach the values below. Separating the representation "
        f"bottleneck from true out-of-family synthesis (higher-resolution contract; a family curriculum) is "
        f"the clear next milestone.</p>{table_html(p['summary'])}</section>")


def ood_section_markdown(ood: dict) -> list[str]:
    p = _ood_pieces(ood)
    verified = ("verified generator-type holdout (checkpoint split provenance confirmed)"
                if p["verified"] else "generator-type holdout (provenance unconfirmed)")
    lines = ["", f"## Out-of-distribution — held-out generator type ({p['holdout']}, n={p['n']})", "",
             (f"A separate model was trained with a frozen split plan that excludes every "
              f"`{p['holdout']}` design; that type is the reserved test partition. This is a {verified}, "
              f"not the IID audit above. Native reference functional self-check on the held-out type: "
              f"{p['native_self_check']} (the specifications are valid). Those references all have embedded "
              f"ports, so the strict-interface accessible cohort is empty by construction and functional "
              f"pass@K is the generalization signal."),
             "",
             (f"Reading it honestly: the shared 64px representation contract passes only "
              f"**{p['ref64_functional_passk']}** functional pass@K for held-out references, so a model score "
              f"here compounds representation loss with genuine type-novelty. Separating the two "
              f"(higher-resolution contract; a family curriculum) is the clear next milestone."),
             ""]
    lines.extend(table_markdown(p["summary"]))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-results", required=True)
    ap.add_argument("--ood-eval-results", nargs="*", default=None,
                    help="optional held-out generator-type eval(s); adds one OOD section each")
    ap.add_argument("--demo", default=None, help="optional runs/demo directory")
    ap.add_argument("--corpus-viz", default="corpus_viz")
    ap.add_argument("--corpus-manifest", default="data/v1_pool_20260720/manifest.json",
                    help="optional frozen broad manifest for stated corpus facts")
    ap.add_argument("--interface-manifest", default="data/v1_interface_accessible/manifest.json",
                    help="optional derived source-interface manifest for stated accessibility facts")
    ap.add_argument("--out", default="site")
    ap.add_argument("--brief", default="docs/ESA_CASE_STUDY.md")
    args = ap.parse_args()
    evaluation_path = Path(args.eval_results).resolve()
    evaluation = load_json(evaluation_path)
    ood_evals = [load_json(Path(p).resolve()) for p in (args.ood_eval_results or [])]
    ood_html_block = "".join(ood_section_html(o) for o in ood_evals)
    out = Path(args.out).resolve()
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    evidence_dir = out / "evidence"
    demo_dir = Path(args.demo).resolve() if args.demo else None
    broad_manifest = Path(args.corpus_manifest).resolve() if args.corpus_manifest else None
    interface_manifest = Path(args.interface_manifest).resolve() if args.interface_manifest else None
    corpus = corpus_facts(broad_manifest, interface_manifest)
    diff_fea_path = evaluation_path.with_name("diff_fea_verification.json")
    diff_fea = load_json(diff_fea_path) if diff_fea_path.is_file() else None
    protocol_path = Path(evaluation["protocol"]["path"])
    hero = None
    hero_selection_rule = None
    if demo_dir and (demo_dir / "demonstrator.png").exists():
        hero = "assets/demonstrator.png"
        shutil.copy2(demo_dir / "demonstrator.png", out / hero)
        provenance_path = demo_dir / "provenance.json"
        if provenance_path.is_file():
            hero_selection_rule = load_json(provenance_path).get("selection_rule")
    evidence_files = copy_evidence(evaluation_path, demo_dir, evidence_dir, protocol_path)
    if corpus is not None:
        provenance_path = evidence_dir / "corpus_provenance.json"
        provenance_path.write_text(json.dumps(corpus, indent=2) + "\n")
        evidence_files.append(provenance_path.name)

    # Use the annotated renderer's family montages rather than bare topology
    # grids.  Their overlays expose supports/ports/directions and their border
    # colour exposes the source port-interface metadata.
    figure_names = ["gripper.png", "fact_translation.png", "gs_truss.png",
                    "rr_four_bar.png", "inverter.png", "crusher.png"]
    copied = []
    viz = Path(args.corpus_viz)
    for name in figure_names:
        source = viz / name
        if source.exists():
            target = assets / name
            shutil.copy2(source, target)
            copied.append((name, f"assets/{name}"))
    viz_provenance = viz / "render_provenance.json"
    if viz_provenance.is_file():
        viz_record = load_json(viz_provenance)
        if corpus is not None:
            source = viz_record.get("source_manifest", {}) or {}
            if (viz_record.get("mode") != "frozen_manifest"
                    or source.get("sha256") != corpus["broad_manifest_sha256"]
                    or int(source.get("record_count", -1)) != corpus["broad_records"]):
                raise ValueError(
                    "annotated corpus figures do not bind to the supplied frozen broad manifest")
        shutil.copy2(viz_provenance, evidence_dir / "corpus_viz_provenance.json")
        evidence_files.append("corpus_viz_provenance.json")
    elif copied and corpus is not None:
        raise ValueError(
            "refusing unprovenanced corpus figures when a frozen corpus manifest is supplied; "
            "run scripts/render_overlay.py --manifest first")

    split = evaluation["split_assessment"]
    protocol_self_check = evaluation.get("protocol_self_check", {})
    native_self_check = protocol_self_check.get("functional_native_reference_pass_rate")
    native_self_check_text = ("not recorded" if native_self_check is None else
                              f"{native_self_check:.3f} (n={protocol_self_check.get('functional_native_reference_n', 0)})")
    corpus_text = "an unspecified number of frozen records (corpus manifest not supplied)"
    if corpus is not None:
        corpus_text = (f"{corpus['broad_records']:,} byte-unique broad records across "
                       f"{corpus['generator_type_count']} generator types")
        if "source_interface_accessible_records" in corpus:
            corpus_text += (f"; {corpus['source_interface_accessible_records']:,} source-interface accessible "
                            f"and {corpus['source_interface_embedded_records']:,} embedded")
    diff_fea_text = "not recorded"
    diff_fea_html = "not recorded"
    if diff_fea is not None:
        verdict = "PASS" if diff_fea.get("passed") else "FAIL"
        comp = ((diff_fea.get("compliance") or {}).get("relative_error"))
        grad = diff_fea.get("gradient_max_relative_error")
        details = []
        if comp is not None:
            details.append(f"compliance relative error {float(comp):.2e}")
        if grad is not None:
            details.append(f"max gradient relative error {float(grad):.2e}")
        diff_fea_text = verdict + (f" ({'; '.join(details)})" if details else "")
        diff_fea_html = (f"<a href='evidence/diff_fea_verification.json'>{html.escape(diff_fea_text)}</a>")
    all_summary = evaluation["summary"].get("predeclared_stratified_heldout_subset",
                                              evaluation["summary"].get("all_heldout_specs", {}))
    interface_summary = evaluation["summary"].get(
        "predeclared_stratified_reference_interface_accessible_subset",
        evaluation["summary"].get("reference_interface_accessible_specs", {}))
    all_n = cohort_n(all_summary)
    interface_n = cohort_n(interface_summary)
    figures = "".join(
        f"<figure><img src='{path}' alt='{html.escape(name)}'><figcaption>{html.escape(name.rsplit('.', 1)[0].replace('_', ' '))}</figcaption></figure>"
        for name, path in copied)
    hero_section = (f"<img class='hero' src='{hero}' alt='Sparse-FEA verified generated mechanism demonstrator'>"
                    if hero else "<div class='pending'>Hero figure pending: run demonstrator.py after evaluation.</div>")
    automatic_hero = not hero_selection_rule or "first strict-interface pass" in str(hero_selection_rule)
    hero_selection_note = ("It was not selected by visual appearance. " if automatic_hero
                           else "This is a user-specified evaluated row, not an automatic first-pass hero. ")
    hero_evidence = ("<p>One illustrative strict-interface pass. Selection rule: "
                     f"<code>{html.escape(str(hero_selection_rule or 'recorded in provenance.json'))}</code>. "
                     f"{hero_selection_note}The exact "
                     "<a href='evidence/gate.json'>gate report</a>, "
                     "<a href='evidence/provenance.json'>provenance</a>, and generated density arrays are packaged in "
                     "<a href='evidence/results.json'>the evidence bundle</a>.</p>"
                     if hero else
                     "<p>No illustrative result is shown until a strict-interface passing candidate is available.</p>")
    page = f"""<!doctype html>
<html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>OpenCompMech — verification-first inverse mechanism design</title>
<style>
body{{font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;max-width:1120px;margin:0 auto;padding:2rem;color:#15202b;background:#f8fafc}} h1,h2{{line-height:1.16}} .lead{{font-size:1.18rem;max-width:850px}} .card{{background:white;border:1px solid #dce3ea;border-radius:12px;padding:1.2rem 1.4rem;margin:1rem 0}} table{{border-collapse:collapse;width:100%;font-size:.88rem}} th,td{{border-bottom:1px solid #dce3ea;padding:.45rem;text-align:left}} th{{background:#f1f5f9}} figure{{display:inline-block;width:31%;vertical-align:top;margin:1%;background:white;padding:.45rem;box-sizing:border-box}} figure img{{width:100%;display:block}} figcaption{{font-size:.8rem;color:#475569;padding:.35rem}} .hero{{width:100%;border-radius:8px}} .pending{{padding:3rem;text-align:center;background:#fff7ed;border:1px dashed #f97316;border-radius:8px}} code{{background:#eaf0f6;padding:.1rem .25rem;border-radius:3px}} .warning{{border-left:5px solid #e11d48;padding-left:1rem}} </style>
<body>
<h1>OpenCompMech</h1>
<p class='lead'>A verification-first case study in conditional generation of 2D compliant mechanisms. The contribution is not a claim of manufacture-ready or flight-qualified hardware: it is a reproducible workflow that generates goal-conditioned candidates and screens them with fresh full-resolution sparse FEA.</p>
<section class='card'><h2>What was tested</h2><p><b>Corpus:</b> {html.escape(corpus_text)}<br><b>Split:</b> {html.escape(split['claim'])}<br><b>Protocol:</b> <code>{html.escape(evaluation['protocol']['name'])}</code> ({evaluation['protocol']['sha256'][:12]})<br><b>Native-reference functional self-check:</b> {native_self_check_text}<br><b>Differentiable-proxy parity:</b> {diff_fea_html}<br><b>Evaluation:</b> {evaluation['n_specs']} held-out specifications, fixed candidate budget K={evaluation['K']}; descriptive subset bootstrap intervals and paired neural–retrieval differences.</p></section>
<section class='card'><h2>Fresh sparse-FEA demonstrator</h2>{hero_section}{hero_evidence}</section>
<section class='card'><h2>Predeclared stratified held-out goal-conditioned subset (n={all_n})</h2><p>These descriptive intervals summarize the deterministic balanced type×source-access subset, not a population-weighted corpus estimate. Functional pass includes geometry, BC/path connectivity, hinge, exact-domain, and signed-response checks. Strict-interface pass additionally requires clear approaches to both generated ports. † metrics are conditional on a passing selected candidate and show their own valid n. ‡ Retrieval is a train-set memorization baseline, so its nearest-train Dice is 1 by construction.</p>{table_html(all_summary)}<h3>Verified selected-response quality</h3>{response_html(all_summary, False)}</section>
<section class='card'><h2>Usable within-condition topology diversity</h2><p>For each specification, this measures pairwise binary topology distance only among verifier-passing candidates in the fixed K budget. It directly separates multiple usable options from merely diverse invalid rasters.</p>{diversity_html(all_summary)}</section>
<section class='card'><h2>Strict interface subset (n={interface_n})</h2><p>This subset starts from source specifications whose reference topology has accessible ports; the generated topology must independently pass the same interface check.</p>{table_html(interface_summary)}<h3>Verified selected-response quality</h3>{response_html(interface_summary, True)}</section>
<section class='card'><h2>Paired neural − retrieval differences</h2><p>Pass@K uses every specification in its cohort. Other paired quality metrics use only specifications where both selected candidates passed; each cell prints that joint-passing n.</p>{paired_html(evaluation.get('paired_differences', {}))}</section>
<section class='card'><h2>Corpus coverage snapshots</h2><div>{figures}</div><p>These annotated synthetic-corpus montages are not physical prototypes: blue squares are supports, green is the input port/direction, magenta is the output port/direction, and a green/red border denotes source port-interface accessible/embedded respectively. Broad functional coverage and interface accessibility are reported as distinct axes.</p></section>
{ood_html_block}<section class='card warning'><h2>What this does not establish</h2><p>2D normalized linear-elastic FEA is not material calibration, thickness design, yield, fatigue, buckling, contact, friction, manufacturing-process validation, or flight qualification. The conditioning vector includes motion/performance labels measured from each reference solution; treating those as desired targets makes this a within-distribution goal-conditioned audit, not BVP-only inverse design. The current corrected corpus also has one source design per specification; useful per-spec multimodality remains an open research problem.</p></section>
<section class='card'><h2>Reproducibility capsule</h2><p>Checkpoint SHA-256: <code>{evaluation['provenance']['checkpoint_sha256'][:16]}</code><br>Cache-index SHA-256: <code>{evaluation['provenance']['cache_index_sha256'][:16]}</code><br>Candidate seeds: <code>{html.escape(evaluation['provenance']['candidate_seed_rule'])}</code><br>Evaluator source hashes: recorded in <a href='evidence/results.json'>results.json</a>{(' (working tree was dirty)' if evaluation['provenance'].get('git_status_porcelain') else '')}.</p><p>Evidence bundle: {html.escape(', '.join(evidence_files) or 'no files copied')}. Run <code>scripts/audit_eval_split.py</code>, then <code>scripts/eval_harness.py</code>, then <code>scripts/demonstrator.py</code> to regenerate it.</p></section>
</body></html>"""
    (out / "index.html").write_text(page)

    brief = ["# OpenCompMech: verification-first conditional mechanism generation", "",
             "## Purpose", "",
             "This case study investigates whether a conditional generative model can propose 2D compliant-mechanism topologies for a stored boundary-value problem, then be screened with a reproducible full-resolution sparse-FEA protocol. It is an engineering workflow demonstration, not a claim of certified hardware.",
             "", "## Data and model", "",
             f"The corrected synthetic v1 corpus contains {corpus_text}. It carries boundary conditions, ports, directional springs, motion labels, and port-interface metadata. The 64×64 flow model is conditioned on the BVP raster stack and scalar function targets. Those targets are measured from the source topology in this benchmark, so the result is within-distribution goal conditioning rather than BVP-only inverse design. Broad-corpus and interface-accessible interpretations remain separate.",
             "", "## Verification protocol", "",
             f"Protocol `{evaluation['protocol']['name']}` ({evaluation['protocol']['sha256'][:12]}) reconstructs each source BVP, maps a candidate to source resolution with nearest-neighbour replication, masks forbidden-domain material, and evaluates geometry/path/hinge checks plus spring-aware signed mechanism response. Every 64px method receives the same fractional-domain-coverage volume projection before FEA; the native reference is intentionally left unprojected as a ceiling/check. Port accessibility is separately reported and required only for the strict interface result. Native-reference functional self-check: {native_self_check_text}. Differentiable-proxy parity report: {diff_fea_text}.",
             "", "## Split and interpretation", "",
             f"**{split['claim']}**", "",
             "The table reports fixed-K candidate budgets. Neural and conditioning-vector retrieval both receive K fresh FEA screens; native references are ceilings/checks, not matched competitors. Intervals are descriptive for the predeclared balanced subset, not corpus-population estimates. Paired quality differences are evaluated only on the joint passing subset and report their n.", "",
             f"## Results — predeclared stratified held-out goal-conditioned subset (n={all_n})", "",
             "† Quality and novelty metrics are conditional on a verifier-passing selected candidate and include their valid n. ‡ Retrieval's nearest-train Dice is 1 by construction because it returns a training topology.", ""]
    brief.extend(table_markdown(all_summary))
    brief.extend(["", "## Verified selected functional-response quality", ""])
    brief.extend(response_markdown(all_summary, False))
    brief.extend(["", "## Usable within-condition topology diversity", "",
                  "Pairwise binary topology distance is measured only among verifier-passing candidates in the fixed K budget; it does not count diverse invalid rasters.", ""])
    brief.extend(diversity_markdown(all_summary))
    brief.extend(["", f"## Results — strict interface subset (n={interface_n})", ""])
    brief.extend(table_markdown(interface_summary))
    brief.extend(["", "## Verified selected strict-interface response quality", ""])
    brief.extend(response_markdown(interface_summary, True))
    brief.extend(["", "## Paired neural − retrieval differences", "",
                  "Pass@K uses every specification in its cohort. Other paired quality metrics use only the joint-passing denominator printed below.", ""])
    brief.extend(paired_markdown(evaluation.get("paired_differences", {})))
    for ood_eval in ood_evals:
        brief.extend(ood_section_markdown(ood_eval))
    brief.extend(["", "## Limitations and next steps", "",
                  "The sparse verifier is distinct from the differentiable sampling proxy but uses the same normalized linear-elastic model; it is not an unrelated material solver. No material/thickness/yield/fatigue/buckling/contact/process/physical-test claim is made. The generator-type held-out retrain (frozen split plan) is now reported above and gives a faithful negative result; the strongest next milestone is to separate the 64px representation bottleneck from genuine out-of-family synthesis (a higher-resolution contract and a family curriculum), then add cross-family solutions for identical specifications and a quality-diversity objective.",
                  "", "## Reproducibility", "",
                  f"Evaluation JSON: `{evaluation_path}`", f"Checkpoint SHA-256: `{evaluation['provenance']['checkpoint_sha256']}`", f"Cache index SHA-256: `{evaluation['provenance']['cache_index_sha256']}`", f"Packaged evidence: `{evidence_dir}`"])
    brief_path = Path(args.brief)
    brief_path.parent.mkdir(parents=True, exist_ok=True)
    brief_path.write_text("\n".join(brief) + "\n")
    print(json.dumps({"site": str(out / "index.html"), "brief": str(brief_path),
                      "hero_included": bool(hero), "corpus_figures": len(copied)}, indent=2))


if __name__ == "__main__":
    main()
