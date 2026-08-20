import unittest

import numpy as np

from aerial_stereo_pitch.pitch import pitch_from_plane, pitch_from_slopes, rise_per_12_from_angle
from aerial_stereo_pitch.plane_fit import (
    PlaneFitRefusal,
    PlaneModel,
    balanced_shared_fit,
    robust_plane_fit,
    threshold_stability,
)


def plane_cloud(slope, offset=0.0):
    x, y = np.meshgrid(np.linspace(-4, 4, 9), np.linspace(-3, 3, 7))
    z = slope * x + 0.1 * y + offset
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


class PlanePitchTests(unittest.TestCase):
    def test_known_pitch_conversions(self):
        for rise in (0.0, 4.0, 6.0, 8.0, 12.0):
            with self.subTest(rise=rise):
                pitch = pitch_from_slopes(rise / 12.0, 0.0)
                self.assertAlmostEqual(pitch.rise_per_12, rise, places=12)
                self.assertAlmostEqual(rise_per_12_from_angle(pitch.angle_deg), rise, places=12)
                self.assertGreater(pitch.normal_up[2], 0.0)

    def test_ransac_rejects_outliers(self):
        cloud = plane_cloud(0.5)
        outliers = np.array([[x, y, 10 + index] for index, (x, y) in enumerate(cloud[:12, :2])])
        result = robust_plane_fit(np.vstack((cloud, outliers)), 0.02, seed=17)
        self.assertAlmostEqual(result.model.a, 0.5, places=10)
        self.assertAlmostEqual(result.model.b, 0.1, places=10)
        self.assertEqual(result.inlier_count, len(cloud))
        self.assertLess(result.rms_z, 1e-10)

    def test_collinear_points_refuse(self):
        line = np.array([[value, value, value] for value in range(5)], dtype=float)
        with self.assertRaises(PlaneFitRefusal):
            robust_plane_fit(line, 0.1)

    def test_balanced_fit_is_invariant_to_pair_duplication(self):
        rng = np.random.default_rng(44)
        pair_a = plane_cloud(0.5)
        pair_a[:, 2] += rng.normal(0.0, 0.002, len(pair_a))
        pair_b = plane_cloud(0.5)
        pair_b[:, 2] += rng.normal(0.0, 0.002, len(pair_b))
        first, _ = balanced_shared_fit({"pair-a": pair_a, "pair-b": pair_b}, 0.03, seed=4)
        duplicate_a, _ = balanced_shared_fit(
            {"pair-a": np.vstack((pair_a, pair_a, pair_a)), "pair-b": pair_b}, 0.03, seed=4
        )
        duplicate_b, _ = balanced_shared_fit(
            {"pair-a": pair_a, "pair-b": np.vstack((pair_b, pair_b))}, 0.03, seed=4
        )
        self.assertTrue(np.allclose(first.model.normal_up, duplicate_a.model.normal_up, atol=1e-12))
        self.assertTrue(np.allclose(first.model.normal_up, duplicate_b.model.normal_up, atol=1e-12))
        self.assertAlmostEqual(pitch_from_plane(first.model).rise_per_12, 12 * np.hypot(0.5, 0.1), places=2)

    def test_shared_fit_refuses_incompatible_slopes(self):
        with self.assertRaisesRegex(PlaneFitRefusal, "shared plane has insufficient support"):
            balanced_shared_fit(
                {"pair-a": plane_cloud(0.5), "pair-b": plane_cloud(0.65)}, 0.03, seed=4
            )

    def test_shared_fit_refuses_incompatible_vertical_offsets(self):
        with self.assertRaisesRegex(PlaneFitRefusal, "shared plane has insufficient support"):
            balanced_shared_fit(
                {"pair-a": plane_cloud(0.5), "pair-b": plane_cloud(0.5, offset=0.2)},
                0.03,
                seed=4,
            )

    def test_final_support_is_enforced_and_reported(self):
        cloud = plane_cloud(0.5)
        outliers = cloud[:8].copy()
        outliers[:, 2] += 2.0
        result = robust_plane_fit(
            np.vstack((cloud, outliers)), 0.02, seed=3, min_support_fraction=0.8
        )
        self.assertGreaterEqual(result.support_fraction, 0.8)
        self.assertEqual(result.support_fraction, result.inlier_count / result.input_count)

    def test_shared_metrics_use_shared_model_inliers(self):
        base = plane_cloud(0.5)
        high = base[:6].copy()
        high[:, 2] += 2.0
        low = base[:6].copy()
        low[:, 2] -= 2.0
        shared, _ = balanced_shared_fit(
            {"pair-a": np.vstack((base, high)), "pair-b": np.vstack((base, low))},
            0.02,
            seed=7,
            min_support_fraction=0.8,
        )
        self.assertEqual(shared.input_count, 138)
        self.assertEqual(shared.inlier_count, 126)
        self.assertAlmostEqual(shared.support_fraction, 63 / 69)
        self.assertLess(shared.rms_z, 1e-10)
        self.assertEqual(shared.shared_pair_diagnostics["pair-a"].inlier_count, 63)
        self.assertEqual(shared.shared_pair_diagnostics["pair-b"].inlier_count, 63)
        self.assertLess(shared.shared_pair_diagnostics["pair-a"].rms_z, 1e-10)
        self.assertAlmostEqual(shared.pairwise_overlap_fractions["pair-a|pair-b"], 1.0)

    def test_shared_fit_refuses_non_overlapping_footprints(self):
        pair_a = plane_cloud(0.5)
        pair_b = pair_a.copy()
        pair_b[:, 0] += 20.0
        pair_b[:, 2] += 10.0
        with self.assertRaisesRegex(PlaneFitRefusal, "footprints do not overlap"):
            balanced_shared_fit({"pair-a": pair_a, "pair-b": pair_b}, 0.02, seed=2)

    def test_convex_overlap_refuses_disjoint_triangles_with_same_aabb(self):
        grid = np.array(
            [(x, y) for x in np.linspace(0, 4, 9) for y in np.linspace(0, 4, 9)]
        )
        lower = grid[grid[:, 0] + grid[:, 1] <= 4.0 + 1e-12]
        upper = grid[grid[:, 0] + grid[:, 1] >= 4.0 - 1e-12]

        def xyz(xy):
            return np.column_stack((xy, 0.5 * xy[:, 0] + 0.1 * xy[:, 1]))

        with self.assertRaisesRegex(PlaneFitRefusal, "footprints do not overlap"):
            balanced_shared_fit({"pair-a": xyz(lower), "pair-b": xyz(upper)}, 0.02, seed=2)

    def test_convex_overlap_reports_compatible_partial_overlap(self):
        pair_a = plane_cloud(0.5)
        pair_b = pair_a.copy()
        pair_b[:, 0] += 4.0
        pair_b[:, 2] += 2.0
        shared, _ = balanced_shared_fit(
            {"pair-a": pair_a, "pair-b": pair_b},
            0.02,
            seed=2,
            min_footprint_overlap_fraction=0.4,
        )
        self.assertAlmostEqual(shared.pairwise_overlap_fractions["pair-a|pair-b"], 0.5)

    def test_threshold_stability_is_explicit(self):
        clouds = {"pair-a": plane_cloud(0.5), "pair-b": plane_cloud(0.5)}
        values = threshold_stability(clouds, [0.05, 0.01, 0.03], seed=9)
        self.assertEqual([item[0] for item in values], [0.01, 0.03, 0.05])
        self.assertTrue(all(abs(item[1].model.a - 0.5) < 1e-10 for item in values))

    def test_pitch_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            pitch_from_slopes(float("nan"), 0.0)

    def test_fit_controls_refuse_non_finite_or_non_integer_values(self):
        cloud = plane_cloud(0.5)
        for threshold in (float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                robust_plane_fit(cloud, threshold)
        with self.assertRaises(ValueError):
            robust_plane_fit(cloud, 0.02, iterations=2.5)
        with self.assertRaises(ValueError):
            robust_plane_fit(cloud, 0.02, min_support_fraction=float("nan"))
        with self.assertRaises(ValueError):
            robust_plane_fit(cloud, 0.02, min_xy_spread=float("inf"))
        with self.assertRaises(ValueError):
            threshold_stability({"pair-a": cloud, "pair-b": cloud}, [0.02, float("nan")])


if __name__ == "__main__":
    unittest.main()
