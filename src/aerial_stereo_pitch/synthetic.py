"""Deterministic analytic scenes for pipeline correctness validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import PinholeCamera
from .correspondence import CorrespondenceSet
from .pitch import pitch_from_plane
from .plane_fit import PlaneFitRefusal, balanced_shared_fit
from .point_cloud import QualityThresholds, filter_triangulated_points
from .triangulation import TriangulationRefusal, triangulate_correspondences


def look_at_camera(camera_id: str, center: np.ndarray, target: np.ndarray) -> PinholeCamera:
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    intrinsic = np.array([[1800.0, 0.0, 1000.0], [0.0, 1800.0, 1000.0], [0.0, 0.0, 1.0]])
    return PinholeCamera(
        intrinsic,
        rotation,
        center,
        (2000, 2000),
        camera_id,
        f"{camera_id}-distortion-corrected-pixels",
        source_id=f"{camera_id}-source",
        world_frame_id="synthetic-local-world",
        world_units="metres",
    )


def synthetic_camera_pairs(degenerate: bool = False) -> dict[str, tuple[PinholeCamera, PinholeCamera]]:
    target = np.zeros(3)
    if degenerate:
        centers = [np.array([0.0, -18.0, 30.0]), np.array([0.00001, -18.0, 30.0])]
        return {
            "pair-a": (
                look_at_camera("degenerate-left", centers[0], target),
                look_at_camera("degenerate-right", centers[1], target),
            )
        }
    return {
        "pair-a": (
            look_at_camera("a-left", np.array([-12.0, -18.0, 30.0]), target),
            look_at_camera("a-right", np.array([12.0, -18.0, 30.0]), target),
        ),
        "pair-b": (
            look_at_camera("b-left", np.array([-18.0, -12.0, 32.0]), target),
            look_at_camera("b-right", np.array([-18.0, 12.0, 32.0]), target),
        ),
    }


def analytic_roof_points(rise_per_12: float, grid_size: int = 9) -> np.ndarray:
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    coordinates = np.linspace(-4.0, 4.0, grid_size)
    x, y = np.meshgrid(coordinates, coordinates)
    z = (rise_per_12 / 12.0) * x
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def project_correspondences(
    pair_id: str,
    pair: tuple[PinholeCamera, PinholeCamera],
    xyz: np.ndarray,
    *,
    rng: np.random.Generator,
    noise_px: float,
    outlier_fraction: float,
    roof_face_scope_id: str = "synthetic-roof-face",
) -> CorrespondenceSet:
    left, right = pair
    left_pixels, left_depth = left.project(xyz)
    right_pixels, right_depth = right.project(xyz)
    if np.any(left_depth <= 0) or np.any(right_depth <= 0):
        raise ValueError("synthetic roof is behind a camera")
    if noise_px:
        left_pixels = left_pixels + rng.normal(0.0, noise_px, left_pixels.shape)
        right_pixels = right_pixels + rng.normal(0.0, noise_px, right_pixels.shape)
    outlier_count = int(round(len(xyz) * outlier_fraction))
    if outlier_count:
        indices = rng.choice(len(xyz), size=outlier_count, replace=False)
        right_pixels[indices] += rng.normal(0.0, 35.0, (outlier_count, 2))
    ids = tuple(f"p-{index:03d}" for index in range(len(xyz)))
    from .correspondence import CorrespondenceLineage

    return CorrespondenceSet(
        left_pixels,
        right_pixels,
        ids,
        left.observation_frame,
        right.observation_frame,
        CorrespondenceLineage(
            pair_id,
            roof_face_scope_id,
            left.world_frame_id,
            left.world_units,
            left.camera_id,
            left.source_id,
            right.camera_id,
            right.source_id,
        ),
    )


@dataclass(frozen=True)
class SyntheticSceneResult:
    scene_id: str
    truth_rise_per_12: float | None
    status: str
    detail: dict[str, object]


def run_scene(scene: dict[str, object], base_seed: int) -> SyntheticSceneResult:
    scene_id = str(scene.get("id", "unnamed"))
    kind = str(scene.get("kind", "scored"))
    truth = None if kind == "degenerate" else float(scene["rise_per_12"])
    noise_px = float(scene.get("noise_px", 0.0))
    outlier_fraction = float(scene.get("outlier_fraction", 0.0))
    if noise_px < 0 or not 0 <= outlier_fraction < 0.5:
        raise ValueError("noise_px must be non-negative and outlier_fraction in [0, 0.5)")
    seed = base_seed + int(scene.get("seed_offset", 0))
    rng = np.random.default_rng(seed)
    points = analytic_roof_points(6.0 if truth is None else truth)
    pair_cameras = synthetic_camera_pairs(degenerate=(kind == "degenerate"))
    thresholds = QualityThresholds(
        max_ray_miss=float(scene.get("max_ray_miss", 0.05)),
        min_intersection_angle_deg=float(scene.get("min_intersection_angle_deg", 1.0)),
        max_reprojection_error_px=float(scene.get("max_reprojection_error_px", 1.0)),
    )
    pair_clouds: dict[str, np.ndarray] = {}
    pair_diagnostics: dict[str, object] = {}
    try:
        for pair_id, pair in sorted(pair_cameras.items()):
            correspondences = project_correspondences(
                pair_id,
                pair,
                points,
                rng=rng,
                noise_px=noise_px,
                outlier_fraction=outlier_fraction,
            )
            triangulated = triangulate_correspondences(pair[0], pair[1], correspondences)
            cloud = filter_triangulated_points(triangulated, thresholds)
            if len(cloud.xyz) < 3:
                reasons = ", ".join(f"{key}={value}" for key, value in cloud.rejection_counts.items())
                raise PlaneFitRefusal(f"fewer than three accepted XYZ points ({reasons})")
            pair_clouds[pair_id] = cloud.xyz
            pair_diagnostics[pair_id] = {
                "input": cloud.input_count,
                "accepted": len(cloud.xyz),
                "rejection_counts": cloud.rejection_counts,
            }
        shared, pair_fits = balanced_shared_fit(
            pair_clouds,
            float(scene.get("plane_residual_threshold", 0.03)),
            seed=seed,
            iterations=int(scene.get("ransac_iterations", 300)),
            min_support_fraction=float(scene.get("min_support_fraction", 0.6)),
            min_xy_spread=float(scene.get("min_xy_spread", 0.5)),
        )
    except (PlaneFitRefusal, TriangulationRefusal, ValueError) as exc:
        status = "refused" if kind == "degenerate" else "failed"
        return SyntheticSceneResult(scene_id, truth, status, {"reason": str(exc)})
    if kind == "degenerate":
        return SyntheticSceneResult(
            scene_id, None, "failed", {"reason": "degenerate geometry was unexpectedly scored"}
        )
    shared_pitch = pitch_from_plane(shared.model)
    pair_pitch = {
        pair_id: round(pitch_from_plane(result.model).rise_per_12, 9)
        for pair_id, result in sorted(pair_fits.items())
    }
    return SyntheticSceneResult(
        scene_id,
        truth,
        "scored",
        {
            "truth_rise_per_12": truth,
            "recovered_rise_per_12": round(shared_pitch.rise_per_12, 9),
            "absolute_error_rise_per_12": round(abs(shared_pitch.rise_per_12 - truth), 9),
            "angle_deg": round(shared_pitch.angle_deg, 9),
            "normal_up": [round(value, 9) for value in shared_pitch.normal_up],
            "pair_rise_per_12": pair_pitch,
            "pair_diagnostics": pair_diagnostics,
            "shared_inliers": shared.inlier_count,
            "shared_rms_z": round(shared.rms_z, 9),
            "shared_xy_spread_minor": round(shared.xy_spread_minor, 9),
        },
    )


def run_experiment(config: dict[str, object]) -> dict[str, object]:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("synthetic config schema_version must be 1")
    scenes = config.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("synthetic config requires a non-empty scenes list")
    seed = int(config.get("seed", 0))
    results: list[SyntheticSceneResult] = []
    for raw_scene in scenes:
        if not isinstance(raw_scene, dict):
            raise ValueError("each synthetic scene must be an object")
        try:
            results.append(run_scene(raw_scene, seed))
        except (KeyError, TypeError, ValueError) as exc:
            results.append(
                SyntheticSceneResult(
                    str(raw_scene.get("id", "unnamed")), None, "failed", {"reason": str(exc)}
                )
            )
    counts = {name: sum(result.status == name for result in results) for name in ("scored", "refused", "failed")}
    counts["missing"] = 0
    noiseless_errors = [
        float(result.detail["absolute_error_rise_per_12"])
        for result, raw in zip(results, scenes)
        if result.status == "scored" and float(raw.get("noise_px", 0.0)) == 0.0
    ]
    return {
        "schema_version": 1,
        "experiment": "analytic calibrated stereo roof-plane recovery",
        "interpretation": "synthetic pipeline correctness; not independent real-world accuracy",
        "seed": seed,
        "conventions": {
            "world_units": "metres",
            "pixel_origin": "top-left; x right, y down",
            "camera_axes": "x right, y down, z forward",
            "plane": "Z=a(X-Xref)+b(Y-Yref)+Zref",
        },
        "unit_of_analysis": "synthetic scene/stereo pair; aggregated by scene",
        "counts": counts,
        "max_noiseless_absolute_error_rise_per_12": round(max(noiseless_errors, default=0.0), 9),
        "scenes": [
            {
                "id": result.scene_id,
                "status": result.status,
                **result.detail,
            }
            for result in results
        ],
    }
