"""Calibrated epipolar checks and an optional OpenCV rectification adapter."""

from __future__ import annotations

import numpy as np

from .camera import PinholeCamera


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def fundamental_matrix(left: PinholeCamera, right: PinholeCamera) -> np.ndarray:
    """Return F such that ``x_right.T @ F @ x_left == 0`` ideally."""
    relative_rotation = right.rotation_world_to_camera @ left.rotation_world_to_camera.T
    relative_translation = right.rotation_world_to_camera @ (
        left.center_world - right.center_world
    )
    essential = _skew(relative_translation) @ relative_rotation
    fundamental = (
        np.linalg.inv(right.intrinsic).T @ essential @ np.linalg.inv(left.intrinsic)
    )
    norm = np.linalg.norm(fundamental)
    if norm < 1e-15:
        raise ValueError("camera geometry does not define a usable epipolar relation")
    return fundamental / norm


def sampson_error(
    left_pixels: np.ndarray, right_pixels: np.ndarray, fundamental: np.ndarray
) -> np.ndarray:
    left = np.atleast_2d(np.asarray(left_pixels, dtype=float))
    right = np.atleast_2d(np.asarray(right_pixels, dtype=float))
    if left.shape != right.shape or left.shape[1] != 2:
        raise ValueError("left and right pixels must have matching shape (N, 2)")
    x1 = np.column_stack((left, np.ones(len(left))))
    x2 = np.column_stack((right, np.ones(len(right))))
    fx1 = (fundamental @ x1.T).T
    ftx2 = (fundamental.T @ x2.T).T
    numerator = np.sum(x2 * fx1, axis=1) ** 2
    denominator = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    return np.divide(numerator, denominator, out=np.full(len(left), np.inf), where=denominator > 0)


def require_opencv() -> object:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Image rectification requires the optional image extra: "
            "pip install 'aerial-stereo-roof-pitch[images]'"
        ) from exc
    version = str(getattr(cv2, "__version__", ""))
    try:
        major = int(version.split(".", 1)[0])
    except ValueError as exc:
        raise RuntimeError("Could not determine the installed OpenCV major version") from exc
    if major != 4:
        raise RuntimeError(
            f"OpenCV {version or 'unknown'} is unsupported; diagnostic rectification requires "
            "OpenCV major 4 (install 'opencv-python-headless>=4.10,<5')"
        )
    return cv2


def calibrated_rectification_maps(
    left: PinholeCamera, right: PinholeCamera
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build diagnostic/image-preparation remaps from calibrated pinhole cameras.

    This optional image operation uses zero distortion because the public core
    contract requires distortion-corrected pixels. Camera/image sizes must match.
    The maps do not define triangulation cameras. Rectified observations require
    separately derived explicit camera and frame models, which are not implemented.
    """
    if left.image_size != right.image_size:
        raise ValueError("calibrated rectification requires equal image dimensions")
    cv2 = require_opencv()
    relative_rotation = right.rotation_world_to_camera @ left.rotation_world_to_camera.T
    relative_translation = right.rotation_world_to_camera @ (
        left.center_world - right.center_world
    )
    distortion = np.zeros(5, dtype=float)
    rect_left, rect_right, proj_left, proj_right, _, _, _ = cv2.stereoRectify(
        left.intrinsic,
        distortion,
        right.intrinsic,
        distortion,
        left.image_size,
        relative_rotation,
        relative_translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    left_maps = cv2.initUndistortRectifyMap(
        left.intrinsic, distortion, rect_left, proj_left, left.image_size, cv2.CV_32FC1
    )
    right_maps = cv2.initUndistortRectifyMap(
        right.intrinsic, distortion, rect_right, proj_right, right.image_size, cv2.CV_32FC1
    )
    return left_maps, right_maps
