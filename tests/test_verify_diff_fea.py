"""CLI contract tests for the DiffFEA numerical verification evidence."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_diff_fea.py"


class VerifyDiffFEACliTest(unittest.TestCase):
    def test_cpu_pass_writes_machine_readable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "diff_fea.json"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--res", "8", "--device", "cpu", "--out", str(out)],
                cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            stdout = json.loads(result.stdout)
            saved = json.loads(out.read_text())
            self.assertTrue(stdout["passed"])
            self.assertEqual(stdout, saved)
            self.assertEqual(stdout["device"]["resolved"], "cpu")
            self.assertEqual(len(stdout["gradient_checks"]), 4)

    def test_invalid_resolution_is_json_failure_and_nonzero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--res", "4", "--device", "cpu"],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        report = json.loads(result.stdout)
        self.assertFalse(report["passed"])
        self.assertEqual(report["error"]["type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
