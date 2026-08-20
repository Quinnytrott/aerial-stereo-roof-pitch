import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aerial_stereo_pitch.config import ConfigError, safe_relative_path
from aerial_stereo_pitch.synthetic import run_experiment, run_scene_with_trace

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

        noisy_config = next(
            scene for scene in self.config["scenes"] if scene["id"].startswith("seeded-noise")
        )
        traced_result, trace = run_scene_with_trace(noisy_config, self.config["seed"])
        self.assertEqual(traced_result.detail["recovered_rise_per_12"], 6.002269925)
        self.assertIsNotNone(trace)
        self.assertEqual({key: len(value) for key, value in trace.pair_clouds.items()}, {
            "pair-a": 73,
            "pair-b": 74,
        })
        self.assertEqual(trace.shared_fit.inlier_count, 146)

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
            self.assertIn("Pair A", completed.stdout)
            self.assertIn("6.002270", completed.stdout)
            self.assertIn("refused: fewer than three", completed.stdout)

    def test_no_argument_example_uses_cwd_relative_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "run_example.py")],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = Path(temporary) / "outputs" / "example"
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "pitch-profile.svg").is_file())

    def test_example_reports_malformed_config_without_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            malformed = root / "malformed.json"
            malformed.write_text("{not json}")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_example.py"),
                    "--config",
                    str(malformed),
                    "--output-dir",
                    str(root / "output"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertTrue(completed.stderr.startswith("REFUSED:"))
            self.assertNotIn("Traceback", completed.stderr)
            self.assertFalse((root / "output").exists())

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
