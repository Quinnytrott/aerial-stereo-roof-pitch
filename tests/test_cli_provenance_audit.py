import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aerial_stereo_pitch.synthetic import (
    analytic_roof_points,
    project_correspondences,
    synthetic_camera_pairs,
)

ROOT = Path(__file__).resolve().parents[1]


def write_pair_inputs(root: Path, pair_id: str = "pair-a") -> tuple[Path, Path]:
    left, right = synthetic_camera_pairs()["pair-a"]
    config = {
        "pair": {
            "pair_id": pair_id,
            "roof_face_scope_id": "synthetic-roof-face",
            "left": left.to_dict(),
            "right": right.to_dict(),
        },
        "quality": {
            "max_ray_miss": 0.05,
            "min_intersection_angle_deg": 1.0,
            "max_reprojection_error_px": 1.0,
        },
    }
    config_path = root / "pair.json"
    config_path.write_text(json.dumps(config))
    observations = project_correspondences(
        "pair-a",
        (left, right),
        analytic_roof_points(6.0)[:12],
        rng=np.random.default_rng(1),
        noise_px=0.0,
        outlier_fraction=0.0,
    )
    csv_path = root / "correspondences.csv"
    fields = [
        "id",
        "x_left",
        "y_left",
        "x_right",
        "y_right",
        "pair_id",
        "roof_face_scope_id",
        "world_frame_id",
        "world_units",
        "left_camera_id",
        "left_source_id",
        "left_frame_id",
        "left_image_width",
        "left_image_height",
        "left_pixel_convention",
        "left_distortion_corrected",
        "right_camera_id",
        "right_source_id",
        "right_frame_id",
        "right_image_width",
        "right_image_height",
        "right_pixel_convention",
        "right_distortion_corrected",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point_id, left_pixel, right_pixel in zip(
            observations.ids, observations.left_pixels, observations.right_pixels
        ):
            writer.writerow(
                {
                    "id": point_id,
                    "x_left": left_pixel[0],
                    "y_left": left_pixel[1],
                    "x_right": right_pixel[0],
                    "y_right": right_pixel[1],
                    "pair_id": observations.lineage.pair_id,
                    "roof_face_scope_id": observations.lineage.roof_face_scope_id,
                    "world_frame_id": observations.lineage.world_frame_id,
                    "world_units": observations.lineage.world_units,
                    "left_camera_id": observations.lineage.left_camera_id,
                    "left_source_id": observations.lineage.left_source_id,
                    "left_frame_id": observations.left_frame.frame_id,
                    "left_image_width": observations.left_frame.image_size[0],
                    "left_image_height": observations.left_frame.image_size[1],
                    "left_pixel_convention": observations.left_frame.pixel_convention,
                    "left_distortion_corrected": "true",
                    "right_camera_id": observations.lineage.right_camera_id,
                    "right_source_id": observations.lineage.right_source_id,
                    "right_frame_id": observations.right_frame.frame_id,
                    "right_image_width": observations.right_frame.image_size[0],
                    "right_image_height": observations.right_frame.image_size[1],
                    "right_pixel_convention": observations.right_frame.pixel_convention,
                    "right_distortion_corrected": "true",
                }
            )
    return config_path, csv_path


def write_xyz(
    path: Path,
    pair_id: str,
    *,
    world="synthetic-local-world",
    scope="synthetic-roof-face",
    reuse_source=None,
):
    x, y = np.meshgrid(np.linspace(-4, 4, 7), np.linspace(-3, 3, 7))
    z = 0.5 * x + 0.1 * y
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    prefix = pair_id
    provenance = {
        "pair_id": pair_id,
        "roof_face_scope_id": scope,
        "world_frame_id": world,
        "world_units": "metres",
        "left_camera_id": f"{prefix}-left-camera",
        "left_source_id": reuse_source or f"{prefix}-left-source",
        "right_camera_id": f"{prefix}-right-camera",
        "right_source_id": f"{prefix}-right-source",
        "left_frame_id": f"{prefix}-left-frame",
        "left_image_width": "2000",
        "left_image_height": "2000",
        "left_pixel_convention": "top-left origin; x right; y down",
        "left_distortion_corrected": "true",
        "right_frame_id": f"{prefix}-right-frame",
        "right_image_width": "2000",
        "right_image_height": "2000",
        "right_pixel_convention": "top-left origin; x right; y down",
        "right_distortion_corrected": "true",
    }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", *provenance, "x", "y", "z"])
        writer.writeheader()
        for index, point in enumerate(points):
            writer.writerow({"id": index, **provenance, "x": point[0], "y": point[1], "z": point[2]})


class CliProvenanceTests(unittest.TestCase):
    def test_stereo_cli_refuses_missing_or_equal_camera_ids(self):
        for mode in ("missing", "equal"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config_path, correspondences = write_pair_inputs(root)
                config = json.loads(config_path.read_text())
                if mode == "missing":
                    del config["pair"]["left"]["camera_id"]
                else:
                    config["pair"]["right"]["camera_id"] = config["pair"]["left"]["camera_id"]
                config_path.write_text(json.dumps(config))
                output = root / "output"
                completed = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/run_stereo_pair.py"), "--config", str(config_path),
                     "--correspondences", str(correspondences), "--output-dir", str(output)],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(output.exists())

    def test_existing_stereo_output_refuses_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, correspondences = write_pair_inputs(root)
            output = root / "existing"
            output.mkdir()
            marker = output / "marker.txt"
            marker.write_text("preserve-me")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_stereo_pair.py"), "--config", str(config),
                 "--correspondences", str(correspondences), "--output-dir", str(output)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(marker.read_text(), "preserve-me")
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["marker.txt"])

    def test_stereo_cli_refuses_frame_contract_mismatches_and_bounds(self):
        mutations = {
            "frame": ("left_frame_id", "wrong-frame"),
            "size": ("left_image_width", "1999"),
            "distortion": ("left_distortion_corrected", "false"),
            "bounds": ("x_left", "2000"),
        }
        for name, (field, value) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, correspondences = write_pair_inputs(root)
                with correspondences.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                    fieldnames = list(rows[0])
                if name == "bounds":
                    rows[0][field] = value
                else:
                    for row in rows:
                        row[field] = value
                with correspondences.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                output = root / "output"
                completed = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/run_stereo_pair.py"), "--config", str(config),
                     "--correspondences", str(correspondences), "--output-dir", str(output)],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(output.exists())

    def test_stereo_cli_writes_point_audit_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, correspondences = write_pair_inputs(root)
            with correspondences.open(newline="") as handle:
                input_rows = list(csv.DictReader(handle))
                fieldnames = list(input_rows[0])
            input_rows[-1]["x_right"] = str(float(input_rows[-1]["x_right"]) + 80.0)
            with correspondences.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(input_rows)
            output = root / "output"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_stereo_pair.py"), "--config", str(config),
                 "--correspondences", str(correspondences), "--output-dir", str(output)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (output / "pair-a-point-audit.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(row["status"] == "accepted" for row in rows), 11)
            refused = next(row for row in rows if row["status"] == "refused")
            self.assertTrue(refused["rejection_reasons"])
            self.assertTrue(all(row["left_frame_id"] and row["ray_miss"] for row in rows))
            with (output / "pair-a-xyz.csv").open(newline="") as handle:
                xyz_rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["world_frame_id"] == "synthetic-local-world" for row in xyz_rows))
            self.assertTrue(all(row["left_frame_id"] for row in xyz_rows))
            self.assertTrue(all(row["roof_face_scope_id"] == "synthetic-roof-face" for row in xyz_rows))
            diagnostics = json.loads((output / "pair-a-diagnostics.json").read_text())
            self.assertEqual(diagnostics["quality"]["max_ray_miss"], 0.05)
            self.assertNotIn("failed", diagnostics)

    def test_pair_id_traversal_refuses_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, correspondences = write_pair_inputs(root, "../escape")
            output = root / "output"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_stereo_pair.py"), "--config", str(config),
                 "--correspondences", str(correspondences), "--output-dir", str(output)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertFalse(output.exists())

    def test_fit_cli_preserves_world_and_pair_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair_a, pair_b, output = root / "a.csv", root / "b.csv", root / "fit.json"
            write_xyz(pair_a, "pair-a")
            write_xyz(pair_b, "pair-b")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts/fit_roof_plane.py"), "--pair-a", str(pair_a),
                 "--pair-b", str(pair_b), "--output", str(output)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["world_frame_id"], "synthetic-local-world")
            self.assertEqual(result["roof_face_scope_id"], "synthetic-roof-face")
            self.assertEqual([item["pair_id"] for item in result["pair_lineage"]], ["pair-a", "pair-b"])
            self.assertEqual(
                result["pair_lineage"][0]["left_observation_frame"]["frame_id"],
                "pair-a-left-frame",
            )
            shared = result["shared_equal_total_pair_weight"]
            self.assertIn("shared_pair_diagnostics", shared)
            self.assertEqual(
                shared["pairwise_convex_footprint_overlap_fractions"]["pair-a|pair-b"], 1.0
            )

    def test_fit_cli_persists_and_applies_non_default_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair_a, pair_b, output = root / "a.csv", root / "b.csv", root / "fit.json"
            write_xyz(pair_a, "pair-a")
            write_xyz(pair_b, "pair-b")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/fit_roof_plane.py"),
                    "--pair-a",
                    str(pair_a),
                    "--pair-b",
                    str(pair_b),
                    "--output",
                    str(output),
                    "--threshold",
                    "0.04",
                    "--stability-thresholds",
                    "0.06,0.02,0.06",
                    "--seed",
                    "13",
                    "--iterations",
                    "77",
                    "--min-support-fraction",
                    "0.9",
                    "--min-xy-spread",
                    "1.25",
                    "--min-footprint-overlap-fraction",
                    "0.8",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(
                result["fit_configuration"],
                {
                    "primary_residual_threshold": 0.04,
                    "stability_residual_thresholds": [0.02, 0.06],
                    "seed": 13,
                    "iterations": 77,
                    "min_support_fraction": 0.9,
                    "min_xy_spread": 1.25,
                    "min_convex_footprint_overlap_fraction": 0.8,
                },
            )
            shared = result["shared_equal_total_pair_weight"]
            self.assertEqual(shared["residual_threshold"], 0.04)
            self.assertEqual(shared["seed"], 13)
            self.assertEqual(result["pair_fits"]["pair-a"]["seed"], 13)
            self.assertEqual(result["pair_fits"]["pair-b"]["seed"], 14)
            self.assertEqual(
                [item["residual_threshold"] for item in result["threshold_stability"]],
                [0.02, 0.06],
            )
            self.assertTrue(all(item["seed"] == 13 for item in result["threshold_stability"]))

    def test_fit_cli_refuses_mismatched_world_or_reused_evidence(self):
        for mode in ("world", "source", "scope", "frame-change"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                pair_a, pair_b, output = root / "a.csv", root / "b.csv", root / "fit.json"
                write_xyz(pair_a, "pair-a")
                if mode == "world":
                    write_xyz(pair_b, "pair-b", world="different-world")
                elif mode == "scope":
                    write_xyz(pair_b, "pair-b", scope="different-roof-face")
                else:
                    write_xyz(
                        pair_b,
                        "pair-b",
                        reuse_source="pair-a-left-source" if mode == "source" else None,
                    )
                if mode == "frame-change":
                    with pair_b.open(newline="") as handle:
                        rows = list(csv.DictReader(handle))
                        fieldnames = list(rows[0])
                    rows[-1]["left_frame_id"] = "changed-within-artifact"
                    with pair_b.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                completed = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/fit_roof_plane.py"), "--pair-a", str(pair_a),
                     "--pair-b", str(pair_b), "--output", str(output)],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(output.exists())

    def test_fit_cli_refuses_missing_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad = root / "bad.csv"
            bad.write_text("x,y,z\n0,0,0\n1,0,1\n0,1,1\n")
            good = root / "good.csv"
            write_xyz(good, "pair-b")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts/fit_roof_plane.py"), "--pair-a", str(bad),
                 "--pair-b", str(good), "--output", str(root / "fit.json")],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 2)


if __name__ == "__main__":
    unittest.main()
