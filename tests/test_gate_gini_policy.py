"""Regression coverage for the energy-Gini acceptance policy.

The gate must honour the source acceptance contract recorded by the generator
in ``validation.quality.gini_cap`` (``0.75`` for SIMP family A, ``null`` for
constructed families), not re-infer it from a family label.  A mis-scoped Gini
check previously failed 10/60 native references and aborted the evaluation.
"""

from __future__ import annotations

import unittest

from scripts.eval_mechanism_gate import resolve_gini_cap


def _meta(gini_cap, family):
    meta = {"family": family}
    if gini_cap is not ...:  # sentinel: ... means "field absent"
        meta["validation"] = {"quality": {"gini_cap": gini_cap}}
    return meta


class GiniPolicyTest(unittest.TestCase):
    def test_recorded_null_exempts_constructed_family(self):
        for family in ("B", "C", "D", "E"):
            required, cap, source = resolve_gini_cap(_meta(None, family), family, 0.75)
            self.assertFalse(required, family)
            self.assertIsNone(cap, family)
            self.assertEqual(source, "source_metadata_contract")

    def test_recorded_numeric_enforces_on_simp(self):
        required, cap, source = resolve_gini_cap(_meta(0.75, "A"), "A", 0.75)
        self.assertTrue(required)
        self.assertEqual(cap, 0.75)
        self.assertEqual(source, "source_metadata_contract")

    def test_recorded_contract_overrides_family_label(self):
        # A hypothetical SIMP-labelled record explicitly exempted stays exempt;
        # a constructed-labelled record explicitly capped stays capped.  The
        # recorded contract wins over the family heuristic in both directions.
        required, cap, source = resolve_gini_cap(_meta(None, "A"), "A", 0.75)
        self.assertFalse(required)
        self.assertEqual(source, "source_metadata_contract")
        required, cap, _ = resolve_gini_cap(_meta(0.5, "D"), "D", 0.75)
        self.assertTrue(required)
        self.assertEqual(cap, 0.5)

    def test_missing_field_falls_back_to_family(self):
        required, cap, source = resolve_gini_cap(_meta(..., "A"), "A", 0.75)
        self.assertTrue(required)
        self.assertEqual(cap, 0.75)
        self.assertEqual(source, "family_fallback_no_recorded_contract")
        required, cap, source = resolve_gini_cap(_meta(..., "B"), "B", 0.75)
        self.assertFalse(required)
        self.assertIsNone(cap)
        self.assertEqual(source, "family_fallback_no_recorded_contract")


if __name__ == "__main__":
    unittest.main()
