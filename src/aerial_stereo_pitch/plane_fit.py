"""Robust explicit roof-plane fitting and pair-balanced shared refits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


class PlaneFitRefusal(ValueError):
    pass


@dataclass(frozen=True)
class PlaneModel:
    a: float
    b: float
    x_ref: float
    y_ref: float
    z_ref: float

    def __post_init__(self) -> None:
        if not np.all(np.isfinite([self.a, self.b, self.x_ref, self.y_ref, self.z_ref])):
            raise PlaneFitRefusal("plane coefficients and reference must be finite")

    def predict(self, xy: np.ndarray) -> np.ndarray:
        values = np.asarray(xy, dtype=float)
        return self.a * (values[..., 0] - self.x_ref) + self.b * (
            values[..., 1] - self.y_ref
        ) + self.z_ref

    def residuals(self, xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(xyz, dtype=float)
        return points[:, 2] - self.predict(points[:, :2])

    @property
    def normal_up(self) -> np.ndarray:
        normal = np.array([-self.a, -self.b, 1.0], dtype=float)
        return normal / np.linalg.norm(normal)


@dataclass(frozen=True)
class PlaneFitResult:
    model: PlaneModel
    input_count: int
    inlier_count: int
    support_fraction: float
    rms_z: float
    xy_spread_minor: float
    residual_threshold: float
    seed: int
    shared_pair_diagnostics: dict[str, "SharedPairDiagnostics"] | None = None
    pairwise_overlap_fractions: dict[str, float] | None = None


@dataclass(frozen=True)
class SharedPairDiagnostics:
    input_count: int
    inlier_count: int
    support_fraction: float
    rms_z: float
    xy_spread_minor: float


def _validated_points(xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise PlaneFitRefusal("at least three XYZ points are required")
    if not np.all(np.isfinite(points)):
        raise PlaneFitRefusal("XYZ points must be finite")
    return points


def fit_weighted_plane(xyz: np.ndarray, weights: np.ndarray | None = None) -> PlaneModel:
    points = _validated_points(xyz)
    if weights is None:
        values = np.full(len(points), 1.0 / len(points))
    else:
        values = np.asarray(weights, dtype=float)
        if values.shape != (len(points),) or np.any(values <= 0) or not np.all(np.isfinite(values)):
            raise PlaneFitRefusal("weights must be finite, positive, and match XYZ points")
        values = values / values.sum()
    x_ref = float(np.sum(values * points[:, 0]))
    y_ref = float(np.sum(values * points[:, 1]))
    design = np.column_stack(
        (points[:, 0] - x_ref, points[:, 1] - y_ref, np.ones(len(points)))
    )
    weighted_design = design * np.sqrt(values)[:, None]
    weighted_z = points[:, 2] * np.sqrt(values)
    if np.linalg.matrix_rank(weighted_design) < 3:
        raise PlaneFitRefusal("points do not span a two-dimensional roof plane")
    coefficients, _, _, _ = np.linalg.lstsq(weighted_design, weighted_z, rcond=None)
    return PlaneModel(
        a=float(coefficients[0]),
        b=float(coefficients[1]),
        x_ref=x_ref,
        y_ref=y_ref,
        z_ref=float(coefficients[2]),
    )


def _xy_spread_minor(xy: np.ndarray, weights: np.ndarray | None = None) -> float:
    if len(xy) < 3:
        return 0.0
    if weights is None:
        centered = xy - np.mean(xy, axis=0)
        covariance = centered.T @ centered / len(xy)
    else:
        normalized = weights / weights.sum()
        centered = xy - np.sum(normalized[:, None] * xy, axis=0)
        covariance = (centered * normalized[:, None]).T @ centered
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(np.sqrt(max(0.0, eigenvalues[0])))


def _cross_2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _convex_hull(xy: np.ndarray) -> np.ndarray:
    points = np.unique(np.asarray(xy, dtype=float), axis=0)
    if len(points) < 3 or not np.all(np.isfinite(points)):
        raise PlaneFitRefusal("shared support has a degenerate convex footprint")
    ordered = points[np.lexsort((points[:, 1], points[:, 0]))]

    def half(values: np.ndarray) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for point in values:
            while len(result) >= 2 and _cross_2d(
                result[-1] - result[-2], point - result[-1]
            ) <= 1e-12:
                result.pop()
            result.append(point)
        return result

    lower = half(ordered)
    upper = half(ordered[::-1])
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3 or _polygon_area(hull) <= 1e-12:
        raise PlaneFitRefusal("shared support has a degenerate convex footprint")
    return hull


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        float(
            np.sum(
                polygon[:, 0] * np.roll(polygon[:, 1], -1)
                - polygon[:, 1] * np.roll(polygon[:, 0], -1)
            )
        )
    ) / 2.0


def _line_intersection(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    clip_start: np.ndarray,
    clip_end: np.ndarray,
) -> np.ndarray:
    segment = segment_end - segment_start
    clip = clip_end - clip_start
    denominator = _cross_2d(segment, clip)
    if abs(denominator) <= 1e-15:
        raise PlaneFitRefusal("convex footprint intersection is numerically degenerate")
    parameter = _cross_2d(clip_start - segment_start, clip) / denominator
    return segment_start + parameter * segment


def _convex_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = [point.copy() for point in subject]
    for index, clip_start in enumerate(clip):
        clip_end = clip[(index + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        start = input_points[-1]
        start_inside = _cross_2d(clip_end - clip_start, start - clip_start) >= -1e-12
        for end in input_points:
            end_inside = _cross_2d(clip_end - clip_start, end - clip_start) >= -1e-12
            if end_inside:
                if not start_inside:
                    output.append(_line_intersection(start, end, clip_start, clip_end))
                output.append(end.copy())
            elif start_inside:
                output.append(_line_intersection(start, end, clip_start, clip_end))
            start = end
            start_inside = end_inside
    return np.asarray(output, dtype=float).reshape((-1, 2))


def _validate_fit_controls(
    residual_threshold: float,
    seed: int,
    iterations: int,
    min_support_fraction: float,
    min_xy_spread: float,
) -> None:
    numeric = np.asarray(
        [residual_threshold, min_support_fraction, min_xy_spread], dtype=float
    )
    if not np.all(np.isfinite(numeric)) or residual_threshold <= 0:
        raise ValueError("fit thresholds and support controls must be finite")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, (int, np.integer))
        or iterations <= 0
    ):
        raise ValueError("iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if not 0 < min_support_fraction <= 1 or min_xy_spread < 0:
        raise ValueError("invalid support or XY-spread requirement")


def _refine_mask_to_stability(
    points: np.ndarray,
    initial_mask: np.ndarray,
    residual_threshold: float,
    *,
    max_refits: int = 25,
) -> tuple[PlaneModel, np.ndarray]:
    mask = np.asarray(initial_mask, dtype=bool)
    seen: set[bytes] = set()
    for _ in range(max_refits):
        if int(mask.sum()) < 3:
            raise PlaneFitRefusal("refit left fewer than three inliers")
        key = mask.tobytes()
        if key in seen:
            raise PlaneFitRefusal("plane inlier refinement did not converge")
        seen.add(key)
        model = fit_weighted_plane(points[mask])
        updated = np.abs(model.residuals(points)) <= residual_threshold
        if np.array_equal(updated, mask):
            return model, updated
        mask = updated
    raise PlaneFitRefusal("plane inlier refinement exceeded its bounded iterations")


def robust_plane_fit(
    xyz: np.ndarray,
    residual_threshold: float,
    *,
    seed: int = 0,
    iterations: int = 300,
    min_support_fraction: float = 0.5,
    min_xy_spread: float = 0.25,
) -> PlaneFitResult:
    points = _validated_points(xyz)
    _validate_fit_controls(
        residual_threshold, seed, iterations, min_support_fraction, min_xy_spread
    )
    rng = np.random.default_rng(seed)
    best: tuple[int, float, np.ndarray] | None = None
    for _ in range(iterations):
        sample_indices = rng.choice(len(points), size=3, replace=False)
        try:
            hypothesis = fit_weighted_plane(points[sample_indices])
        except PlaneFitRefusal:
            continue
        residuals = np.abs(hypothesis.residuals(points))
        inliers = residuals <= residual_threshold
        count = int(inliers.sum())
        if count < 3:
            continue
        rms = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
        candidate = (count, -rms, inliers)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise PlaneFitRefusal("no non-degenerate plane hypothesis")
    model, inliers = _refine_mask_to_stability(points, best[2], residual_threshold)
    support = float(np.mean(inliers))
    if support < min_support_fraction:
        raise PlaneFitRefusal(
            f"insufficient plane support: {support:.3f} < {min_support_fraction:.3f}"
        )
    spread = _xy_spread_minor(points[inliers, :2])
    if spread < min_xy_spread:
        raise PlaneFitRefusal(f"insufficient XY spread: {spread:.6f} < {min_xy_spread:.6f}")
    residuals = model.residuals(points[inliers])
    return PlaneFitResult(
        model=model,
        input_count=len(points),
        inlier_count=int(inliers.sum()),
        support_fraction=float(np.mean(inliers)),
        rms_z=float(np.sqrt(np.mean(residuals**2))),
        xy_spread_minor=spread,
        residual_threshold=float(residual_threshold),
        seed=int(seed),
    )


def balanced_shared_fit(
    pair_clouds: dict[str, np.ndarray],
    residual_threshold: float,
    *,
    seed: int = 0,
    iterations: int = 300,
    min_support_fraction: float = 0.5,
    min_xy_spread: float = 0.25,
    min_footprint_overlap_fraction: float = 0.1,
) -> tuple[PlaneFitResult, dict[str, PlaneFitResult]]:
    _validate_fit_controls(
        residual_threshold, seed, iterations, min_support_fraction, min_xy_spread
    )
    if (
        not np.isfinite(min_footprint_overlap_fraction)
        or not 0 < min_footprint_overlap_fraction <= 1
    ):
        raise ValueError("min_footprint_overlap_fraction must be finite in (0, 1]")
    if len(pair_clouds) < 2:
        raise PlaneFitRefusal("a shared fit requires at least two distinct pair clouds")
    pair_results: dict[str, PlaneFitResult] = {}
    clouds: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    pair_count = len(pair_clouds)
    for offset, (pair_id, raw_cloud) in enumerate(sorted(pair_clouds.items())):
        # Exact repeated samples contain no new evidence. Removing them also makes
        # the equal-pair contract invariant to duplicating an entire cloud.
        cloud = np.unique(_validated_points(raw_cloud), axis=0)
        result = robust_plane_fit(
            cloud,
            residual_threshold,
            seed=seed + offset,
            iterations=iterations,
            min_support_fraction=min_support_fraction,
            min_xy_spread=min_xy_spread,
        )
        inliers = np.abs(result.model.residuals(cloud)) <= residual_threshold
        pair_results[pair_id] = result
        clouds[pair_id] = cloud
        masks[pair_id] = inliers

    def refit(current_masks: dict[str, np.ndarray]) -> PlaneModel:
        selected = [clouds[key][current_masks[key]] for key in sorted(clouds)]
        weights = [
            np.full(len(points), 1.0 / (pair_count * len(points))) for points in selected
        ]
        return fit_weighted_plane(np.vstack(selected), np.concatenate(weights))

    model = refit(masks)
    seen: set[tuple[bytes, ...]] = set()
    for _ in range(25):
        updated: dict[str, np.ndarray] = {}
        for pair_id, cloud in sorted(clouds.items()):
            mask = np.abs(model.residuals(cloud)) <= residual_threshold
            support = float(np.mean(mask))
            if int(mask.sum()) < 3 or support < min_support_fraction:
                raise PlaneFitRefusal(
                    f"shared plane has insufficient support in {pair_id}: "
                    f"{support:.3f} < {min_support_fraction:.3f}"
                )
            spread = _xy_spread_minor(cloud[mask, :2])
            if spread < min_xy_spread:
                raise PlaneFitRefusal(
                    f"shared plane has insufficient XY spread in {pair_id}: "
                    f"{spread:.6f} < {min_xy_spread:.6f}"
                )
            updated[pair_id] = mask
        key = tuple(updated[pair_id].tobytes() for pair_id in sorted(updated))
        refined = refit(updated)
        if all(np.array_equal(updated[pair_id], masks[pair_id]) for pair_id in updated):
            model = refined
            masks = updated
            break
        if key in seen:
            raise PlaneFitRefusal("shared plane inlier refinement did not converge")
        seen.add(key)
        model = refined
        masks = updated
    else:
        raise PlaneFitRefusal("shared plane refinement exceeded its bounded iterations")

    pair_ids = sorted(clouds)
    shared_clouds = [clouds[key][masks[key]] for key in pair_ids]
    hulls = {pair_id: _convex_hull(clouds[pair_id][masks[pair_id], :2]) for pair_id in pair_ids}
    hull_areas = {pair_id: _polygon_area(hull) for pair_id, hull in hulls.items()}
    overlap_fractions: dict[str, float] = {}
    for index, pair_a in enumerate(pair_ids):
        for pair_b in pair_ids[index + 1 :]:
            intersection = _convex_intersection(hulls[pair_a], hulls[pair_b])
            intersection_area = _polygon_area(intersection)
            fraction = intersection_area / min(hull_areas[pair_a], hull_areas[pair_b])
            overlap_fractions[f"{pair_a}|{pair_b}"] = fraction
            if fraction < min_footprint_overlap_fraction:
                raise PlaneFitRefusal(
                    "shared plane pair footprints do not overlap sufficiently: "
                    f"{fraction:.3f} < {min_footprint_overlap_fraction:.3f}"
                )

    combined = np.vstack(shared_clouds)
    shared_weights = np.concatenate(
        [np.full(len(points), 1.0 / (pair_count * len(points))) for points in shared_clouds]
    )
    residuals = model.residuals(combined)
    weighted_rms = float(np.sqrt(np.sum(shared_weights * residuals**2)))
    spread = _xy_spread_minor(combined[:, :2], shared_weights)
    if spread < min_xy_spread:
        raise PlaneFitRefusal("balanced inliers do not provide sufficient XY spread")
    shared_pair_diagnostics: dict[str, SharedPairDiagnostics] = {}
    for pair_id in pair_ids:
        pair_points = clouds[pair_id]
        pair_inliers = masks[pair_id]
        pair_residuals = model.residuals(pair_points[pair_inliers])
        if np.any(np.abs(pair_residuals) > residual_threshold):
            raise PlaneFitRefusal("shared-model inliers exceed the declared residual threshold")
        shared_pair_diagnostics[pair_id] = SharedPairDiagnostics(
            input_count=len(pair_points),
            inlier_count=int(pair_inliers.sum()),
            support_fraction=float(np.mean(pair_inliers)),
            rms_z=float(np.sqrt(np.mean(pair_residuals**2))),
            xy_spread_minor=_xy_spread_minor(pair_points[pair_inliers, :2]),
        )
    shared = PlaneFitResult(
        model=model,
        input_count=sum(len(value) for value in clouds.values()),
        inlier_count=len(combined),
        support_fraction=float(np.mean([np.mean(masks[key]) for key in sorted(masks)])),
        rms_z=weighted_rms,
        xy_spread_minor=spread,
        residual_threshold=float(residual_threshold),
        seed=int(seed),
        shared_pair_diagnostics=shared_pair_diagnostics,
        pairwise_overlap_fractions=overlap_fractions,
    )
    return shared, pair_results


def threshold_stability(
    pair_clouds: dict[str, np.ndarray],
    thresholds: Iterable[float],
    *,
    seed: int = 0,
    iterations: int = 300,
    min_support_fraction: float = 0.5,
    min_xy_spread: float = 0.25,
    min_footprint_overlap_fraction: float = 0.1,
) -> list[tuple[float, PlaneFitResult]]:
    values = sorted({float(value) for value in thresholds})
    if not values or not np.all(np.isfinite(values)) or values[0] <= 0:
        raise ValueError("at least one positive threshold is required")
    return [
        (
            threshold,
            balanced_shared_fit(
                pair_clouds,
                threshold,
                seed=seed,
                iterations=iterations,
                min_support_fraction=min_support_fraction,
                min_xy_spread=min_xy_spread,
                min_footprint_overlap_fraction=min_footprint_overlap_fraction,
            )[0],
        )
        for threshold in values
    ]
