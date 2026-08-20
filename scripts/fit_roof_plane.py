#!/usr/bin/env python3
"""Fit separately processed Pair A/B planes, then an equal-weight shared plane."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerial_stereo_pitch.config import write_stable_json  # noqa: E402
from aerial_stereo_pitch.frames import PixelFrame  # noqa: E402
from aerial_stereo_pitch.pitch import pitch_from_plane  # noqa: E402
from aerial_stereo_pitch.plane_fit import PlaneFitRefusal, balanced_shared_fit, threshold_stability  # noqa: E402


@dataclass(frozen=True)
class XYZCloud:
    xyz: np.ndarray
    pair_id: str
    roof_face_scope_id: str
    world_frame_id: str
    world_units: str
    left_camera_id: str
    left_source_id: str
    right_camera_id: str
    right_source_id: str
    left_frame: PixelFrame
    right_frame: PixelFrame


PROVENANCE_COLUMNS = {
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
}


def strict_bool(value: str, name: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{name} must be 'true' or 'false'")


def read_xyz(path: Path) -> XYZCloud:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"x", "y", "z"} | PROVENANCE_COLUMNS
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("XYZ CSV requires x,y,z and complete provenance columns")
            rows = list(reader)
            if not rows:
                raise ValueError("XYZ CSV must contain points")
            first = rows[0]
            provenance = {key: first[key] for key in PROVENANCE_COLUMNS}
            if any(not value.strip() for value in provenance.values()):
                raise ValueError("XYZ provenance values must be non-empty")
            if any(any(row[key] != provenance[key] for key in PROVENANCE_COLUMNS) for row in rows[1:]):
                raise ValueError("XYZ provenance changes within one cloud")
            points = np.array(
                [[float(row["x"]), float(row["y"]), float(row["z"])] for row in rows]
            )
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    left_frame = PixelFrame(
        provenance["left_frame_id"],
        (int(provenance["left_image_width"]), int(provenance["left_image_height"])),
        provenance["left_pixel_convention"],
        strict_bool(provenance["left_distortion_corrected"], "left_distortion_corrected"),
    )
    right_frame = PixelFrame(
        provenance["right_frame_id"],
        (int(provenance["right_image_width"]), int(provenance["right_image_height"])),
        provenance["right_pixel_convention"],
        strict_bool(provenance["right_distortion_corrected"], "right_distortion_corrected"),
    )
    return XYZCloud(
        xyz=points,
        pair_id=provenance["pair_id"],
        roof_face_scope_id=provenance["roof_face_scope_id"],
        world_frame_id=provenance["world_frame_id"],
        world_units=provenance["world_units"],
        left_camera_id=provenance["left_camera_id"],
        left_source_id=provenance["left_source_id"],
        right_camera_id=provenance["right_camera_id"],
        right_source_id=provenance["right_source_id"],
        left_frame=left_frame,
        right_frame=right_frame,
    )


def validate_distinct_evidence(pair_a: XYZCloud, pair_b: XYZCloud) -> None:
    for cloud in (pair_a, pair_b):
        if cloud.left_camera_id == cloud.right_camera_id or cloud.left_source_id == cloud.right_source_id:
            raise ValueError("each stereo pair must use distinct left/right cameras and source images")
        if cloud.left_frame.frame_id == cloud.right_frame.frame_id:
            raise ValueError("each stereo pair must use distinct left/right observation frames")
    if pair_a.pair_id == pair_b.pair_id:
        raise ValueError("Pair A and Pair B must have distinct pair_id values")
    if (
        pair_a.world_frame_id != pair_b.world_frame_id
        or pair_a.world_units != pair_b.world_units
    ):
        raise ValueError("Pair A and Pair B must use the same world frame and units")
    if pair_a.roof_face_scope_id != pair_b.roof_face_scope_id:
        raise ValueError("Pair A and Pair B must use the same roof_face_scope_id")
    if pair_a.world_units != "metres":
        raise ValueError("world_units must be 'metres'")
    cameras_a = {pair_a.left_camera_id, pair_a.right_camera_id}
    cameras_b = {pair_b.left_camera_id, pair_b.right_camera_id}
    sources_a = {pair_a.left_source_id, pair_a.right_source_id}
    sources_b = {pair_b.left_source_id, pair_b.right_source_id}
    if cameras_a & cameras_b or sources_a & sources_b:
        raise ValueError("Pair A and Pair B must not reuse camera or source-image evidence")


def fit_record(result: object) -> dict[str, object]:
    pitch = pitch_from_plane(result.model)
    record = {
        "rise_per_12": round(pitch.rise_per_12, 9),
        "angle_deg": round(pitch.angle_deg, 9),
        "normal_up": [round(value, 9) for value in pitch.normal_up],
        "support": round(result.support_fraction, 9),
        "inliers": result.inlier_count,
        "input": result.input_count,
        "rms_z": round(result.rms_z, 9),
        "xy_spread_minor": round(result.xy_spread_minor, 9),
        "residual_threshold": result.residual_threshold,
        "seed": result.seed,
    }
    if result.shared_pair_diagnostics is not None:
        record["shared_pair_diagnostics"] = {
            pair_id: {
                "input": diagnostics.input_count,
                "inliers": diagnostics.inlier_count,
                "support": round(diagnostics.support_fraction, 9),
                "rms_z": round(diagnostics.rms_z, 9),
                "xy_spread_minor": round(diagnostics.xy_spread_minor, 9),
            }
            for pair_id, diagnostics in sorted(result.shared_pair_diagnostics.items())
        }
    if result.pairwise_overlap_fractions is not None:
        record["pairwise_convex_footprint_overlap_fractions"] = {
            pair_names: round(fraction, 9)
            for pair_names, fraction in sorted(result.pairwise_overlap_fractions.items())
        }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-a", type=Path, required=True)
    parser.add_argument("--pair-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--stability-thresholds", default="0.02,0.03,0.05")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--min-support-fraction", type=float, default=0.5)
    parser.add_argument("--min-xy-spread", type=float, default=0.25)
    parser.add_argument("--min-footprint-overlap-fraction", type=float, default=0.1)
    args = parser.parse_args()
    try:
        pair_a = read_xyz(args.pair_a)
        pair_b = read_xyz(args.pair_b)
        validate_distinct_evidence(pair_a, pair_b)
        clouds = {pair_a.pair_id: pair_a.xyz, pair_b.pair_id: pair_b.xyz}
        thresholds = sorted({float(value) for value in args.stability_thresholds.split(",")})
        shared, independent = balanced_shared_fit(
            clouds,
            args.threshold,
            seed=args.seed,
            iterations=args.iterations,
            min_support_fraction=args.min_support_fraction,
            min_xy_spread=args.min_xy_spread,
            min_footprint_overlap_fraction=args.min_footprint_overlap_fraction,
        )
        stability = threshold_stability(
            clouds,
            thresholds,
            seed=args.seed,
            iterations=args.iterations,
            min_support_fraction=args.min_support_fraction,
            min_xy_spread=args.min_xy_spread,
            min_footprint_overlap_fraction=args.min_footprint_overlap_fraction,
        )
    except (ValueError, PlaneFitRefusal) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    write_stable_json(
        args.output,
        {
            "fit_configuration": {
                "primary_residual_threshold": args.threshold,
                "stability_residual_thresholds": thresholds,
                "seed": args.seed,
                "iterations": args.iterations,
                "min_support_fraction": args.min_support_fraction,
                "min_xy_spread": args.min_xy_spread,
                "min_convex_footprint_overlap_fraction": args.min_footprint_overlap_fraction,
            },
            "pair_fits": {key: fit_record(value) for key, value in sorted(independent.items())},
            "shared_equal_total_pair_weight": fit_record(shared),
            "threshold_stability": [
                {"threshold": threshold, **fit_record(result)} for threshold, result in stability
            ],
            "interpretation": "internal geometric consistency; not independent measurement accuracy",
            "world_frame_id": pair_a.world_frame_id,
            "world_units": pair_a.world_units,
            "roof_face_scope_id": pair_a.roof_face_scope_id,
            "pair_lineage": [
                {
                    "pair_id": value.pair_id,
                    "left_camera_id": value.left_camera_id,
                    "left_source_id": value.left_source_id,
                    "right_camera_id": value.right_camera_id,
                    "right_source_id": value.right_source_id,
                    "left_observation_frame": value.left_frame.to_dict(),
                    "right_observation_frame": value.right_frame.to_dict(),
                }
                for value in (pair_a, pair_b)
            ],
        },
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
