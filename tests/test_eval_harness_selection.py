"""Focused regression tests for evaluator fallback selection."""

import unittest

from scripts.eval_harness import protocol_violation


def _ordinary_failed(candidate_index=0):
    return {
        "candidate_index": candidate_index,
        "gate": {
            "functional": {"checks": {"geometry": False}},
            "response": {"u_out_projected": 1.0, "ga_signed": 1.0},
            "geometry": {"volume_fraction": {"actual": 0.2, "target": 0.2}},
        },
    }


class ProtocolViolationTest(unittest.TestCase):
    def test_solver_and_malformed_reports_rank_after_evaluated_failure(self):
        ordinary = protocol_violation(_ordinary_failed())
        reports = [
            {"candidate_index": 1, "gate": {"worker_error": "broken"}},
            {"candidate_index": 2, "gate": {"fea_error": "singular"}},
            {"candidate_index": 3, "gate": {"functional": {}}},
            {"candidate_index": 4, "gate": {"functional": "malformed"}},
        ]
        for candidate in reports:
            with self.subTest(candidate=candidate["candidate_index"]):
                self.assertGreater(protocol_violation(candidate), ordinary)


if __name__ == "__main__":
    unittest.main()
