"""Quality-filtered XYZ clouds with missing/refused reason accounting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .triangulation import TriangulatedPoint


@dataclass(frozen=True)
class QualityThresholds:
    max_ray_miss: float = 0.05
    min_intersection_angle_deg: float = 1.0
    max_reprojection_error_px: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.max_ray_miss, self.min_intersection_angle_deg, self.max_reprojection_error_px],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("quality thresholds must be positive")


@dataclass(frozen=True)
class FilteredPointCloud:
    xyz: np.ndarray
    accepted_ids: tuple[str, ...]
    rejection_counts: dict[str, int]
    input_count: int
    accepted_points: tuple[TriangulatedPoint, ...]
    rejection_reasons: dict[str, tuple[str, ...]]


def filter_triangulated_points(
    points: list[TriangulatedPoint], thresholds: QualityThresholds
) -> FilteredPointCloud:
    accepted: list[TriangulatedPoint] = []
    rejected: Counter[str] = Counter()
    rejection_reasons: dict[str, tuple[str, ...]] = {}
    for point in points:
        reasons: list[str] = []
        if point.depth_left <= 0 or point.depth_right <= 0:
            reasons.append("non_positive_depth")
        if point.intersection_angle_deg < thresholds.min_intersection_angle_deg:
            reasons.append("intersection_angle_too_small")
        if point.ray_miss > thresholds.max_ray_miss:
            reasons.append("ray_miss_too_large")
        if max(point.reprojection_left_px, point.reprojection_right_px) > thresholds.max_reprojection_error_px:
            reasons.append("reprojection_error_too_large")
        if reasons:
            rejected.update(reasons)
            rejection_reasons[point.point_id] = tuple(reasons)
        else:
            accepted.append(point)
    xyz = np.array([point.xyz for point in accepted], dtype=float).reshape((-1, 3))
    return FilteredPointCloud(
        xyz=xyz,
        accepted_ids=tuple(point.point_id for point in accepted),
        rejection_counts=dict(sorted(rejected.items())),
        input_count=len(points),
        accepted_points=tuple(accepted),
        rejection_reasons=dict(sorted(rejection_reasons.items())),
    )
