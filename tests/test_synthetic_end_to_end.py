import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aerial_stereo_pitch.config import ConfigError, safe_relative_path
from aerial_stereo_pitch.synthetic import run_experiment

ROOT = Path(__file__).resolve().parents[1]


class SyntheticEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "examples" / "example_config.json").read_text())

    def test_known_truth_noise_and_refusal_contract(self):
        first = run_experiment(self.config)
        second = run_experiment(self.config)
        self.assertEqual(first, second)
        self.assertEqual(first["counts"], {"scored": 6, "refused": 1, "failed": 0, "missing": 0})
        self.assertLessEqual(first["max_noiseless_absolute_error_rise_per_12"], 1e-8)
        noisy = next(scene for scene in first["scenes"] if scene["id"].startswith("seeded-noise"))
        self.assertLess(noisy["absolute_error_rise_per_12"], 0.05)
        refused = next(scene for scene in first["scenes"] if scene["status"] == "refused")
        self.assertIn("fewer than three", refused["reason"])

    def test_malformed_config_refuses(self):
        with self.assertRaisesRegex(ValueError, "schema_version"):
            run_experiment({"schema_version": 2, "scenes": []})
        with self.assertRaises(ConfigError):
            safe_relative_path("/private/example.tif", "test")
        with self.assertRaises(ConfigError):
            safe_relative_path("../escape.tif", "test")

    def test_fresh_checkout_script_matches_expected_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_example.py"),
                    "--output-dir",
                    str(output),
                    "--check",
                    str(ROOT / "examples" / "expected_summary.json"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_existing_example_output_refuses_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            marker = output / "marker.txt"
            marker.write_text("preserve-me")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_example.py"),
                    "--output-dir",
                    str(output),
                    "--check",
                    str(ROOT / "examples" / "expected_summary.json"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(marker.read_text(), "preserve-me")
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["marker.txt"])

    def test_failed_golden_check_publishes_no_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = root / "wrong.json"
            wrong.write_text("{}\n")
            output = root / "should-not-exist"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_example.py"),
                    "--output-dir",
                    str(output),
                    "--check",
                    str(wrong),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertFalse(output.exists())

    def test_ontario_template_fails_closed_without_rights_acknowledgement(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "prepare_dataset.py"),
                "--config",
                str(ROOT / "examples" / "ontario-scoop-example.json"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("rights_acknowledged must be true", completed.stderr)


if __name__ == "__main__":
    unittest.main()
