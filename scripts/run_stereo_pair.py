#!/usr/bin/env python3
"""Triangulate a declared camera pair from a correspondence CSV."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerial_stereo_pitch.config import load_json, write_stable_json  # noqa: E402
from aerial_stereo_pitch.correspondence import read_correspondence_csv  # noqa: E402
from aerial_stereo_pitch.point_cloud import QualityThresholds, filter_triangulated_points  # noqa: E402
from aerial_stereo_pitch.stereo_pairs import StereoPair  # noqa: E402
from aerial_stereo_pitch.triangulation import TriangulationRefusal, triangulate_one  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON object containing pair and quality")
    parser.add_argument("--correspondences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_json(args.config)
        pair = StereoPair.from_dict(config["pair"])
        observations = read_correspondence_csv(args.correspondences)
        observations.require_compatible_cameras(pair)
        quality = QualityThresholds(**config.get("quality", {}))
        results = []
        early_refusals: Counter[str] = Counter()
        early_reasons: dict[str, tuple[str, ...]] = {}
        for point_id, left_pixel, right_pixel in zip(
            observations.ids, observations.left_pixels, observations.right_pixels
        ):
            try:
                results.append(triangulate_one(pair.left, pair.right, left_pixel, right_pixel, point_id))
            except TriangulationRefusal:
                early_refusals["degenerate_rays"] += 1
                early_reasons[point_id] = ("degenerate_rays",)
        cloud = filter_triangulated_points(results, quality)
        if len(cloud.xyz) < 3:
            raise ValueError("fewer than three accepted XYZ points; refusing output")
    except (KeyError, TypeError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    rejection_counts = Counter(cloud.rejection_counts)
    rejection_counts.update(early_refusals)
    accepted = {point.point_id: point for point in cloud.accepted_points}
    triangulated = {point.point_id: point for point in results}
    lineage = observations.lineage
    if args.output_dir.exists():
        print("REFUSED: --output-dir must be a new path", file=sys.stderr)
        return 2
    output_parent = args.output_dir.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".stereo-stage-", dir=output_parent))
    try:
        xyz_path = stage / f"{pair.pair_id}-xyz.csv"
        with xyz_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "id",
                    "pair_id",
                    "roof_face_scope_id",
                    "world_frame_id",
                    "world_units",
                    "left_camera_id",
                    "left_source_id",
                    "right_camera_id",
                    "right_source_id",
                    "left_frame_id",
                    "left_image_width",
                    "left_image_height",
                    "left_pixel_convention",
                    "left_distortion_corrected",
                    "right_frame_id",
                    "right_image_width",
                    "right_image_height",
                    "right_pixel_convention",
                    "right_distortion_corrected",
                    "x",
                    "y",
                    "z",
                ]
            )
            for point_id in observations.ids:
                if point_id not in accepted:
                    continue
                xyz = accepted[point_id].xyz
                writer.writerow(
                    [
                        point_id,
                        lineage.pair_id,
                        lineage.roof_face_scope_id,
                        lineage.world_frame_id,
                        lineage.world_units,
                        lineage.left_camera_id,
                        lineage.left_source_id,
                        lineage.right_camera_id,
                        lineage.right_source_id,
                        observations.left_frame.frame_id,
                        observations.left_frame.image_size[0],
                        observations.left_frame.image_size[1],
                        observations.left_frame.pixel_convention,
                        "true",
                        observations.right_frame.frame_id,
                        observations.right_frame.image_size[0],
                        observations.right_frame.image_size[1],
                        observations.right_frame.pixel_convention,
                        "true",
                        *(f"{value:.12g}" for value in xyz),
                    ]
                )

        audit_path = stage / f"{pair.pair_id}-point-audit.csv"
        with audit_path.open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "id",
                "status",
                "rejection_reasons",
                "left_frame_id",
                "right_frame_id",
                "x_left",
                "y_left",
                "x_right",
                "y_right",
                "x",
                "y",
                "z",
                "ray_miss",
                "intersection_angle_deg",
                "depth_left",
                "depth_right",
                "reprojection_left_px",
                "reprojection_right_px",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for point_id, left_pixel, right_pixel in zip(
                observations.ids, observations.left_pixels, observations.right_pixels
            ):
                point = triangulated.get(point_id)
                reasons = early_reasons.get(point_id, cloud.rejection_reasons.get(point_id, ()))
                row = {
                    "id": point_id,
                    "status": "accepted" if point_id in accepted else "refused",
                    "rejection_reasons": ";".join(reasons),
                    "left_frame_id": observations.left_frame.frame_id,
                    "right_frame_id": observations.right_frame.frame_id,
                    "x_left": f"{left_pixel[0]:.12g}",
                    "y_left": f"{left_pixel[1]:.12g}",
                    "x_right": f"{right_pixel[0]:.12g}",
                    "y_right": f"{right_pixel[1]:.12g}",
                }
                if point is not None:
                    row.update(
                        {
                            "x": f"{point.xyz[0]:.12g}",
                            "y": f"{point.xyz[1]:.12g}",
                            "z": f"{point.xyz[2]:.12g}",
                            "ray_miss": f"{point.ray_miss:.12g}",
                            "intersection_angle_deg": f"{point.intersection_angle_deg:.12g}",
                            "depth_left": f"{point.depth_left:.12g}",
                            "depth_right": f"{point.depth_right:.12g}",
                            "reprojection_left_px": f"{point.reprojection_left_px:.12g}",
                            "reprojection_right_px": f"{point.reprojection_right_px:.12g}",
                        }
                    )
                writer.writerow(row)

        diagnostics_path = stage / f"{pair.pair_id}-diagnostics.json"
        write_stable_json(
            diagnostics_path,
            {
                "pair_id": pair.pair_id,
                "roof_face_scope_id": pair.roof_face_scope_id,
                "world_frame_id": pair.world_frame_id,
                "world_units": pair.world_units,
                "lineage": {
                    "left_camera_id": pair.left.camera_id,
                    "left_source_id": pair.left.source_id,
                    "right_camera_id": pair.right.camera_id,
                    "right_source_id": pair.right.source_id,
                },
                "observation_frames": {
                    "left": observations.left_frame.to_dict(),
                    "right": observations.right_frame.to_dict(),
                },
                "quality": {
                    "max_ray_miss": quality.max_ray_miss,
                    "min_intersection_angle_deg": quality.min_intersection_angle_deg,
                    "max_reprojection_error_px": quality.max_reprojection_error_px,
                },
                "input": len(observations.ids),
                "accepted": len(cloud.xyz),
                "refused": len(observations.ids) - len(cloud.xyz),
                "rejection_counts": dict(sorted(rejection_counts.items())),
            },
        )

        output_root = args.output_dir.resolve()
        for filename in (xyz_path.name, audit_path.name, diagnostics_path.name):
            destination = (output_root / filename).resolve()
            if destination.parent != output_root:
                raise ValueError("resolved output would escape output directory")
        os.replace(stage, args.output_dir)
        stage = None
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
    print(f"accepted {len(cloud.xyz)}/{len(observations.ids)} points for {pair.pair_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
