import unittest
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from aerial_stereo_pitch.camera import PinholeCamera
from aerial_stereo_pitch.correspondence import CorrespondenceLineage, CorrespondenceSet
from aerial_stereo_pitch.frames import PIXEL_CONVENTION, CropTransform, PixelFrame, apply_homography
from aerial_stereo_pitch.rectify import (
    calibrated_rectification_maps,
    fundamental_matrix,
    require_opencv,
    sampson_error,
)
from aerial_stereo_pitch.stereo_pairs import StereoPair
from aerial_stereo_pitch.synthetic import analytic_roof_points, synthetic_camera_pairs


class CameraFrameTests(unittest.TestCase):
    def test_project_and_backproject_are_consistent(self):
        camera = PinholeCamera(
            np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]),
            np.eye(3),
            np.zeros(3),
            (100, 80),
        )
        point = np.array([1.0, 2.0, 10.0])
        pixel, depth = camera.project(point)
        origin, ray = camera.backproject_ray(pixel)
        self.assertAlmostEqual(depth, 10.0)
        self.assertTrue(np.allclose(pixel, [60.0, 60.0]))
        self.assertTrue(np.allclose(np.cross(point - origin, ray), 0.0, atol=1e-12))

    def test_invalid_rotation_refuses(self):
        with self.assertRaisesRegex(ValueError, "rotation"):
            PinholeCamera(np.eye(3), np.zeros((3, 3)), np.zeros(3), (10, 10))
        with self.assertRaises(ValueError):
            PinholeCamera(np.eye(3), np.eye(3), np.zeros(3), (10.5, 10))

    def test_crop_round_trip_and_provenance(self):
        frame = CropTransform((1000, 800), (100.0, 200.0), (400, 200), (800, 200), "source-a")
        source = np.array([[100.0, 200.0], [500.0, 400.0]])
        output = frame.source_to_output(source)
        self.assertTrue(np.allclose(output, [[0.0, 0.0], [800.0, 200.0]]))
        self.assertTrue(np.allclose(frame.output_to_source(output), source))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            frame.require_compatible_source("source-b", (1000, 800))

    def test_homography_refuses_point_at_infinity(self):
        with self.assertRaisesRegex(ValueError, "infinity"):
            apply_homography(np.array([1.0, 2.0]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]))

    def test_crop_and_homography_refuse_non_finite_values(self):
        with self.assertRaises(ValueError):
            CropTransform((100, 100), (float("nan"), 0.0), (20, 20), (20, 20), "source")
        with self.assertRaises(ValueError):
            CropTransform((100, 100), (0.0, 0.0), (float("inf"), 20), (20, 20), "source")
        with self.assertRaises(ValueError):
            apply_homography(np.array([1.0, 2.0]), np.full((3, 3), float("nan")))

    def test_calibrated_epipolar_relation(self):
        left, right = synthetic_camera_pairs()["pair-a"]
        xyz = analytic_roof_points(6.0)
        left_pixels, _ = left.project(xyz)
        right_pixels, _ = right.project(xyz)
        errors = sampson_error(left_pixels, right_pixels, fundamental_matrix(left, right))
        self.assertLess(float(errors.max()), 1e-20)

    def test_optional_opencv_refuses_with_guidance_when_absent(self):
        try:
            require_opencv()
        except RuntimeError as exc:
            message = str(exc)
            if "unsupported" in message:
                self.assertIn("OpenCV major 4", message)
                return
            self.assertIn("optional image extra", message)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "optional OpenCV is not installed")
    def test_optional_rectification_adapter_when_available(self):
        left, right = synthetic_camera_pairs()["pair-a"]
        try:
            left_maps, right_maps = calibrated_rectification_maps(left, right)
        except RuntimeError as exc:
            if "unsupported" in str(exc):
                self.skipTest(str(exc))
            raise
        self.assertEqual(left_maps[0].shape, (2000, 2000))
        self.assertEqual(right_maps[0].shape, (2000, 2000))

    def test_opencv_major_five_refuses_with_guidance(self):
        with patch.dict(sys.modules, {"cv2": SimpleNamespace(__version__="5.0.0")}):
            with self.assertRaisesRegex(RuntimeError, "OpenCV major 4"):
                require_opencv()

    def test_correspondence_frames_bind_to_cameras(self):
        left, right = synthetic_camera_pairs()["pair-a"]
        pair = StereoPair("pair-a", left, right)
        pixels = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]])
        lineage = CorrespondenceLineage(
            "pair-a",
            pair.roof_face_scope_id,
            pair.world_frame_id,
            pair.world_units,
            left.camera_id,
            left.source_id,
            right.camera_id,
            right.source_id,
        )
        observations = CorrespondenceSet(
            pixels, pixels, ("a", "b", "c"), left.observation_frame, right.observation_frame, lineage
        )
        observations.require_compatible_cameras(pair)
        wrong_size = PixelFrame(left.frame_id, (1999, 2000))
        mismatched = CorrespondenceSet(
            pixels, pixels, ("a", "b", "c"), wrong_size, right.observation_frame, lineage
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            mismatched.require_compatible_cameras(pair)
        wrong_id = CorrespondenceSet(
            pixels,
            pixels,
            ("a", "b", "c"),
            PixelFrame("different-frame", left.image_size),
            right.observation_frame,
            lineage,
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            wrong_id.require_compatible_cameras(pair)

    def test_distortion_and_out_of_bounds_pixels_refuse(self):
        with self.assertRaisesRegex(ValueError, "distortion-corrected"):
            PixelFrame("raw-frame", (100, 100), PIXEL_CONVENTION, False)
        with self.assertRaisesRegex(ValueError, "pixel_convention"):
            PixelFrame("wrong-convention", (100, 100), "bottom-left")
        frame = PixelFrame("frame", (100, 100))
        lineage = CorrespondenceLineage(
            "pair-a", "roof-face", "world", "metres", "lc", "ls", "rc", "rs"
        )
        with self.assertRaisesRegex(ValueError, "outside frame"):
            CorrespondenceSet(
                np.array([[100.0, 5.0]]),
                np.array([[5.0, 5.0]]),
                ("p",),
                frame,
                PixelFrame("right-frame", (100, 100)),
                lineage,
            )


if __name__ == "__main__":
    unittest.main()
