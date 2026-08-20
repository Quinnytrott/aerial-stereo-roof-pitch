"""Pinhole camera model with explicit world and camera frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .frames import PIXEL_CONVENTION, PixelFrame, _positive_integer_pair


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


@dataclass(frozen=True)
class PinholeCamera:
    """A camera using ``X_cam = R_world_to_camera @ (X_world - C)``.

    Camera axes are x right, y down, z forward. Pixels have a top-left origin.
    Distortion must be removed before using this core model.
    """

    intrinsic: np.ndarray
    rotation_world_to_camera: np.ndarray
    center_world: np.ndarray
    image_size: tuple[int, int]
    camera_id: str = "camera"
    frame_id: str = "camera-pixels"
    pixel_convention: str = PIXEL_CONVENTION
    distortion_corrected: bool = True
    source_id: str = "source"
    world_frame_id: str = "local-world"
    world_units: str = "metres"

    def __post_init__(self) -> None:
        k = _array(self.intrinsic, (3, 3), "intrinsic")
        r = _array(self.rotation_world_to_camera, (3, 3), "rotation")
        c = _array(self.center_world, (3,), "center_world")
        if abs(np.linalg.det(k)) < 1e-12:
            raise ValueError("intrinsic matrix must be invertible")
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-8) or not np.isclose(
            np.linalg.det(r), 1.0, atol=1e-8
        ):
            raise ValueError("rotation must be a right-handed orthonormal matrix")
        image_size = _positive_integer_pair(self.image_size, "image_size")
        frame = PixelFrame(
            frame_id=self.frame_id,
            image_size=image_size,
            pixel_convention=self.pixel_convention,
            distortion_corrected=self.distortion_corrected,
        )
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("camera_id must be a non-empty string")
        for name, value in (
            ("source_id", self.source_id),
            ("world_frame_id", self.world_frame_id),
            ("world_units", self.world_units),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.world_units != "metres":
            raise ValueError("world_units must be 'metres' for this pipeline")
        object.__setattr__(self, "intrinsic", k)
        object.__setattr__(self, "rotation_world_to_camera", r)
        object.__setattr__(self, "center_world", c)
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "frame_id", frame.frame_id)

    @property
    def observation_frame(self) -> PixelFrame:
        return PixelFrame(
            self.frame_id, self.image_size, self.pixel_convention, self.distortion_corrected
        )

    @property
    def projection_matrix(self) -> np.ndarray:
        extrinsic = np.column_stack(
            (self.rotation_world_to_camera, -self.rotation_world_to_camera @ self.center_world)
        )
        return self.intrinsic @ extrinsic

    def project(self, xyz_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(xyz_world, dtype=float)
        one = points.ndim == 1
        points = np.atleast_2d(points)
        if points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("xyz_world must have shape (N, 3) and be finite")
        camera = (self.rotation_world_to_camera @ (points - self.center_world).T).T
        depth = camera[:, 2]
        if np.any(np.abs(depth) < 1e-12):
            raise ValueError("point projects at zero camera depth")
        homogeneous = (self.intrinsic @ camera.T).T
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
        return (pixels[0], depth[0]) if one else (pixels, depth)

    def backproject_ray(self, pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        uv = _array(pixel, (2,), "pixel")
        direction_camera = np.linalg.solve(
            self.intrinsic, np.array([uv[0], uv[1], 1.0], dtype=float)
        )
        direction_world = self.rotation_world_to_camera.T @ direction_camera
        direction_world /= np.linalg.norm(direction_world)
        return self.center_world.copy(), direction_world

    def depth(self, xyz_world: np.ndarray) -> float:
        point = _array(xyz_world, (3,), "xyz_world")
        return float((self.rotation_world_to_camera @ (point - self.center_world))[2])

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PinholeCamera":
        required = {
            "camera_id",
            "intrinsic",
            "rotation_world_to_camera",
            "center_world",
            "observation_frame",
            "source_id",
            "world_frame_id",
            "world_units",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"camera is missing: {', '.join(missing)}")
        frame = PixelFrame.from_dict(value["observation_frame"])
        if "image_size" in value and tuple(value["image_size"]) != frame.image_size:
            raise ValueError("legacy image_size conflicts with observation_frame.image_size")
        return cls(
            intrinsic=value["intrinsic"],
            rotation_world_to_camera=value["rotation_world_to_camera"],
            center_world=value["center_world"],
            image_size=frame.image_size,
            camera_id=str(value["camera_id"]),
            frame_id=frame.frame_id,
            pixel_convention=frame.pixel_convention,
            distortion_corrected=frame.distortion_corrected,
            source_id=value["source_id"],
            world_frame_id=value["world_frame_id"],
            world_units=value["world_units"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "intrinsic": self.intrinsic.tolist(),
            "rotation_world_to_camera": self.rotation_world_to_camera.tolist(),
            "center_world": self.center_world.tolist(),
            "image_size": list(self.image_size),
            "observation_frame": self.observation_frame.to_dict(),
            "source_id": self.source_id,
            "world_frame_id": self.world_frame_id,
            "world_units": self.world_units,
        }
