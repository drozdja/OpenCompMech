"""Regression coverage for explicit functional/interface reporting fields."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from scripts.build_case_study import corpus_facts, response_html, sha256_file, table_html
from scripts.eval_harness import method_summary


def _candidate(functional: bool, interface: bool):
    return {
        "gate": {
            "functional_passed": functional,
            "interface_passed": interface,
            "failure_reasons": [] if functional else ["geometry"],
        }
    }


def _selected(functional: bool = True, interface: bool = True):
    return {
        "functional_passed": functional,
        "interface_passed": interface,
        "motion_class_match": True,
        "vf_abs_error": 0.03,
        "ga_abs_error": 0.07,
        "u_out_projected": 1.4,
        "ga_signed": 0.8,
        "output_alignment": 0.9,
        "output_selectivity": 1.6,
        "bc_connected": True,
        "mechanism_path_connected": True,
        "nearest_train_dice": 0.7,
        "nearest_train_topology_distance": 0.3,
    }


def _trial():
    diversity = {
        "candidate_budget": 2,
        "passing_candidate_count": 1,
        "passing_binary_unique_topologies": 1,
        "passing_pairwise_topology_distance": None,
    }
    return {
        "candidates": [_candidate(True, True), _candidate(False, False)],
        "selected_functional": _selected(),
        "selected_interface": _selected(),
        "diversity": {"functional": diversity, "interface": diversity},
    }


class ExplicitPassReportingTest(unittest.TestCase):
    def test_method_summary_keeps_functional_and_interface_passes_separate(self):
        row = {"methods": {"neural": _trial()}}
        result = method_summary([row], "neural", strict=False, bootstrap_reps=20, seed=0)
        metrics = result["metrics"]
        self.assertEqual(metrics["functional_candidate_pass_rate"]["mean"], 0.5)
        self.assertEqual(metrics["functional_pass_at_k"]["mean"], 1.0)
        self.assertEqual(metrics["interface_candidate_pass_rate"]["mean"], 0.5)
        self.assertEqual(metrics["interface_pass_at_k"]["mean"], 1.0)
        self.assertEqual(metrics["ga_signed"]["mean"], 0.8)
        self.assertEqual(metrics["output_selectivity"]["mean"], 1.6)

    def test_case_study_tables_expose_the_new_metrics(self):
        result = method_summary([{"methods": {"neural": _trial()}}], "neural",
                                strict=False, bootstrap_reps=20, seed=0)
        html_table = table_html({"neural": result})
        html_response = response_html({"neural": result}, strict_selection=False)
        self.assertIn("functional pass@K", html_table)
        self.assertIn("strict-interface pass@K", html_table)
        self.assertIn("signed GA", html_response)
        self.assertIn("output selectivity", html_response)

    def test_corpus_facts_are_manifest_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broad = root / "broad.json"
            broad.write_text(json.dumps({
                "format": "opencompmech.dataset-manifest.v1", "n_unique": 2,
                "records": [{"row": 0, "type": "gripper"},
                            {"row": 1, "type": "inverter"}],
            }))
            interface = root / "interface.json"
            interface.write_text(json.dumps({
                "format": "opencompmech.dataset-manifest.v1", "n_unique": 1,
                "records": [{"row": 0, "type": "gripper"}],
                "derivation": {"source_manifest_sha256": sha256_file(broad),
                               "selection_rule": "metadata.validation.port_interface.passed is exactly true"},
            }))
            facts = corpus_facts(broad, interface)
            self.assertEqual(facts["broad_records"], 2)
            self.assertEqual(facts["generator_type_count"], 2)
            self.assertEqual(facts["source_interface_accessible_records"], 1)
            self.assertEqual(facts["source_interface_embedded_records"], 1)
            self.assertTrue(facts["interface_derivation_matches_broad"])


if __name__ == "__main__":
    unittest.main()
