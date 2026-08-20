import unittest

import numpy as np

from aerial_stereo_pitch.camera import PinholeCamera
from aerial_stereo_pitch.point_cloud import QualityThresholds, filter_triangulated_points
from aerial_stereo_pitch.synthetic import analytic_roof_points, synthetic_camera_pairs
from aerial_stereo_pitch.triangulation import TriangulationRefusal, triangulate_one


class TriangulationTests(unittest.TestCase):
    def test_known_point_reconstruction(self):
        left, right = synthetic_camera_pairs()["pair-a"]
        point = analytic_roof_points(6.0)[17]
        left_pixel, _ = left.project(point)
        right_pixel, _ = right.project(point)
        result = triangulate_one(left, right, left_pixel, right_pixel)
        self.assertTrue(np.allclose(result.xyz, point, atol=1e-10))
        self.assertLess(result.ray_miss, 1e-10)
        self.assertGreater(result.intersection_angle_deg, 1.0)
        self.assertLess(result.reprojection_left_px, 1e-9)

    def test_parallel_rays_refuse(self):
        camera = PinholeCamera(np.eye(3), np.eye(3), np.zeros(3), (10, 10))
        with self.assertRaises(TriangulationRefusal):
            triangulate_one(camera, camera, np.zeros(2), np.zeros(2))

    def test_behind_camera_point_is_rejected(self):
        left = PinholeCamera(np.eye(3), np.eye(3), np.array([-1.0, 0.0, 0.0]), (10, 10))
        right = PinholeCamera(np.eye(3), np.eye(3), np.array([1.0, 0.0, 0.0]), (10, 10))
        result = triangulate_one(left, right, np.array([-0.2, 0.0]), np.array([0.2, 0.0]))
        cloud = filter_triangulated_points([result], QualityThresholds())
        self.assertEqual(len(cloud.xyz), 0)
        self.assertEqual(cloud.rejection_counts["non_positive_depth"], 1)

    def test_near_zero_baseline_rejected_by_angle(self):
        left, right = synthetic_camera_pairs(degenerate=True)["pair-a"]
        point = analytic_roof_points(6.0)[20]
        lp, _ = left.project(point)
        rp, _ = right.project(point)
        result = triangulate_one(left, right, lp, rp)
        cloud = filter_triangulated_points([result], QualityThresholds(min_intersection_angle_deg=1.0))
        self.assertEqual(cloud.rejection_counts["intersection_angle_too_small"], 1)

    def test_quality_thresholds_refuse_non_finite(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                QualityThresholds(max_ray_miss=value)


if __name__ == "__main__":
    unittest.main()
