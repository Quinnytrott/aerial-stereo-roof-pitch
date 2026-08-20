"""Closest-ray triangulation with explicit geometric diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import PinholeCamera
from .correspondence import CorrespondenceSet


class TriangulationRefusal(ValueError):
    pass


@dataclass(frozen=True)
class TriangulatedPoint:
    point_id: str
    xyz: np.ndarray
    ray_miss: float
    intersection_angle_deg: float
    depth_left: float
    depth_right: float
    reprojection_left_px: float
    reprojection_right_px: float


def triangulate_one(
    left: PinholeCamera,
    right: PinholeCamera,
    left_pixel: np.ndarray,
    right_pixel: np.ndarray,
    point_id: str = "point",
) -> TriangulatedPoint:
    origin_left, direction_left = left.backproject_ray(left_pixel)
    origin_right, direction_right = right.backproject_ray(right_pixel)
    dot = float(np.clip(direction_left @ direction_right, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(abs(dot))))
    system = np.column_stack((direction_left, -direction_right))
    if np.linalg.matrix_rank(system, tol=1e-12) < 2:
        raise TriangulationRefusal("rays are parallel or numerically degenerate")
    parameters, _, _, _ = np.linalg.lstsq(system, origin_right - origin_left, rcond=None)
    closest_left = origin_left + parameters[0] * direction_left
    closest_right = origin_right + parameters[1] * direction_right
    xyz = (closest_left + closest_right) / 2.0
    ray_miss = float(np.linalg.norm(closest_left - closest_right))
    projected_left, depth_left = left.project(xyz)
    projected_right, depth_right = right.project(xyz)
    return TriangulatedPoint(
        point_id=point_id,
        xyz=xyz,
        ray_miss=ray_miss,
        intersection_angle_deg=angle,
        depth_left=float(depth_left),
        depth_right=float(depth_right),
        reprojection_left_px=float(np.linalg.norm(projected_left - left_pixel)),
        reprojection_right_px=float(np.linalg.norm(projected_right - right_pixel)),
    )


def triangulate_correspondences(
    left: PinholeCamera, right: PinholeCamera, correspondences: CorrespondenceSet
) -> list[TriangulatedPoint]:
    results = []
    for point_id, left_pixel, right_pixel in zip(
        correspondences.ids, correspondences.left_pixels, correspondences.right_pixels
    ):
        results.append(triangulate_one(left, right, left_pixel, right_pixel, point_id))
    return results
