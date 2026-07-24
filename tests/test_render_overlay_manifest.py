"""Focused CLI coverage for provenance-bound annotated corpus rendering."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_overlay.py"


def write_sample(corpus: Path, name: str, problem_type: str, accessible: bool) -> dict:
    density = np.zeros((8, 8), dtype=np.float32)
    density[2:6, 1:7] = 1.0
    np.save(corpus / f"{name}.density.npy", density)
    metadata = {
        "resolution": 8,
        "problem_type": problem_type,
        "mechanism": {
            "input_node": 9,
            "output_node": 71,
            "input_direction": [1.0, 0.0],
            "output_direction": [0.0, 1.0],
        },
        "boundary_conditions": [{"nodes": [0], "directions": [0, 1]}],
        "validation": {
            "port_interface": {"passed": accessible},
            "motion": {"magnitude_class": "transmitting", "transfer_class": "forwarding", "ga_signed": 1.0},
        },
    }
    (corpus / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"stem": f"corpus/{name}", "type": problem_type}


class ManifestOverlayCliTest(unittest.TestCase):
    def test_manifest_mode_uses_only_declared_records_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            gripper_accessible = write_sample(corpus, "gripper_accessible", "gripper", True)
            gripper_embedded = write_sample(corpus, "gripper_embedded", "gripper", False)
            inverter = write_sample(corpus, "inverter", "inverter", True)
            # This valid on-disk sample must never enter the manifest-driven figures.
            write_sample(corpus, "undeclared", "undeclared", True)
            manifest = {
                "format": "opencompmech.dataset-manifest.v1",
                "n_unique": 3,
                "records": [
                    {"row": 10, **gripper_accessible},
                    {"row": 2, **gripper_embedded},
                    {"row": 7, **inverter},
                ],
            }
            source = corpus / "manifest.json"
            source.write_text(json.dumps(manifest, indent=2) + "\n")

            base = [sys.executable, str(SCRIPT), "--manifest", "corpus/manifest.json",
                    "--root", ".", "--seed", "17"]
            first = subprocess.run(base + ["--out", "render_a"], cwd=root,
                                   capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run(base + ["--out", "render_b"], cwd=root,
                                    capture_output=True, text=True, check=False)
            self.assertEqual(second.returncode, 0, second.stderr)

            provenance = json.loads((root / "render_a" / "render_provenance.json").read_text())
            self.assertEqual(provenance["format"], "opencompmech.annotated-corpus-render.v1")
            self.assertEqual(provenance["mode"], "frozen_manifest")
            self.assertEqual(provenance["source_manifest"]["path"], "corpus/manifest.json")
            self.assertEqual(provenance["source_manifest"]["record_count"], 3)
            self.assertEqual(provenance["source_manifest"]["sha256"],
                             hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(provenance["selection"]["seed"], 17)
            selected_stems = {record["stem"] for group in provenance["groups"]
                              for record in group["selected_records"]}
            self.assertEqual(selected_stems, {
                "corpus/gripper_accessible", "corpus/gripper_embedded", "corpus/inverter"})
            self.assertNotIn("corpus/undeclared", selected_stems)
            self.assertTrue((root / "render_a" / "gripper.png").is_file())
            self.assertTrue((root / "render_a" / "inverter.png").is_file())
            self.assertFalse((root / "render_a" / "undeclared.png").exists())
            self.assertEqual((root / "render_a" / "render_provenance.json").read_bytes(),
                             (root / "render_b" / "render_provenance.json").read_bytes())
            for name in ("gripper.png", "inverter.png"):
                self.assertEqual((root / "render_a" / name).read_bytes(),
                                 (root / "render_b" / name).read_bytes())

    def test_manifest_mode_fails_before_rendering_on_missing_declared_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            manifest = {
                "format": "opencompmech.dataset-manifest.v1",
                "n_unique": 1,
                "records": [{"row": 0, "stem": "corpus/missing", "type": "gripper"}],
            }
            (corpus / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--manifest", "corpus/manifest.json",
                 "--root", ".", "--out", "rendered"],
                cwd=root, capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest-driven rendering aborted", result.stderr)
            self.assertIn("corpus/missing", result.stderr)
            self.assertFalse((root / "rendered" / "render_provenance.json").exists())

    def test_legacy_positional_scan_remains_available_for_exploration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            family = root / "legacy" / "gripper"
            family.mkdir(parents=True)
            write_sample(family, "sample", "gripper", True)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "legacy", "rendered", "--seed", "3"],
                cwd=root, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            provenance = json.loads((root / "rendered" / "render_provenance.json").read_text())
            self.assertEqual(provenance["mode"], "legacy_glob")
            self.assertIsNone(provenance["source_manifest"])
            self.assertTrue((root / "rendered" / "gripper.png").is_file())


if __name__ == "__main__":
    unittest.main()
