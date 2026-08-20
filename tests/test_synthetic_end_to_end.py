import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aerial_stereo_pitch.config import ConfigError, safe_relative_path, stable_json_text
from aerial_stereo_pitch.synthetic import (
    canonical_rounded_float,
    run_experiment,
    run_scene_with_trace,
)

ROOT = Path(__file__).resolve().parents[1]


class SyntheticEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "examples" / "example_config.json").read_text())

    def test_known_truth_noise_and_refusal_contract(self):
        first = run_experiment(self.config)
        second = run_experiment(self.config)
        self.assertEqual(first, second)
        self.assertEqual(stable_json_text(first), stable_json_text(second))
        self.assertNotIn("-0.0", stable_json_text(first))
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

    def test_public_rounding_canonicalizes_only_rounded_zero(self):
        for value in (-0.0, -0.0000000004, 0.0, 0.0000000004):
            rounded = canonical_rounded_float(value)
            self.assertEqual(rounded, 0.0)
            self.assertEqual(math.copysign(1.0, rounded), 1.0)
        self.assertEqual(canonical_rounded_float(1.2345678916), 1.234567892)
        self.assertEqual(canonical_rounded_float(-1.2345678916), -1.234567892)

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

    def test_failed_golden_check_reports_signed_zero_path_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong_summary = run_experiment(self.config)
            wrong_summary["scenes"][0]["normal_up"][1] = -0.0
            wrong = root / "wrong.json"
            wrong.write_text(stable_json_text(wrong_summary))
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
            self.assertIn(
                "$.scenes[0].normal_up[1]: expected -0.0, actual 0.0",
                completed.stderr,
            )
            self.assertFalse(output.exists())

    def test_failed_golden_check_hides_file_path_unexpected_key_and_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private_directory = (
                root / "private-address-123-main-st" / "account-998877"
            )
            private_directory.mkdir(parents=True)
            wrong_summary = run_experiment(self.config)
            secret_key = "customer-account-id-ACCT-987654321"
            secret_value = 9876543210123456
            wrong_summary["conventions"][secret_key] = secret_value
            wrong = private_directory / "signed-download-token.json"
            wrong.write_text(stable_json_text(wrong_summary))
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
            self.assertIn("summary differs from expected file", completed.stderr)
            self.assertIn("$.conventions[<expected-only-key#1,len=", completed.stderr)
            self.assertIn("expected type=number, actual missing", completed.stderr)
            for private_text in (
                str(wrong),
                "private-address-123-main-st",
                "account-998877",
                "signed-download-token.json",
                secret_key,
                str(secret_value),
            ):
                self.assertNotIn(private_text, completed.stderr)
            self.assertFalse(output.exists())

    def test_failed_golden_check_redacts_strings_and_caps_differences(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong_summary = run_experiment(self.config)
            secrets = [f"PRIVATE-VALUE-{index}" for index in range(6)]
            wrong_summary["conventions"]["camera_axes"] = secrets[0]
            wrong_summary["conventions"]["pixel_origin"] = secrets[1]
            wrong_summary["conventions"]["plane"] = secrets[2]
            wrong_summary["conventions"]["world_units"] = secrets[3]
            wrong_summary["experiment"] = secrets[4]
            wrong_summary["interpretation"] = secrets[5]
            wrong = root / "wrong.json"
            wrong.write_text(stable_json_text(wrong_summary))
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
            self.assertIn("$.conventions.camera_axes: expected string(len=15)", completed.stderr)
            self.assertIn("... 3 more differences", completed.stderr)
            self.assertNotIn("PRIVATE-VALUE", completed.stderr)
            self.assertNotIn("x right", completed.stderr)
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
