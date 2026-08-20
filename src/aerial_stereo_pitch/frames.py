"""Pixel-frame provenance and explicit crop/resize transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PIXEL_CONVENTION = "top-left origin; x right; y down"


def _positive_integer_pair(value: tuple[int, int], name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain width and height")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)) or int(item) <= 0:
            raise ValueError(f"{name} must contain finite positive integers")
        result.append(int(item))
    return result[0], result[1]


@dataclass(frozen=True)
class PixelFrame:
    frame_id: str
    image_size: tuple[int, int]
    pixel_convention: str = PIXEL_CONVENTION
    distortion_corrected: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("frame_id must be a non-empty string")
        object.__setattr__(self, "image_size", _positive_integer_pair(self.image_size, "image_size"))
        if self.pixel_convention != PIXEL_CONVENTION:
            raise ValueError(f"pixel_convention must be exactly '{PIXEL_CONVENTION}'")
        if self.distortion_corrected is not True:
            raise ValueError("observations must be explicitly distortion-corrected")

    def require_compatible(self, expected: "PixelFrame") -> None:
        if self != expected:
            raise ValueError(
                f"observation frame {self.frame_id!r} is incompatible with expected "
                f"frame {expected.frame_id!r}"
            )

    def validate_pixels(self, pixels: np.ndarray, side: str) -> None:
        values = np.asarray(pixels, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"{side} pixels must be finite with shape (N, 2)")
        width, height = self.image_size
        outside = (
            (values[:, 0] < 0)
            | (values[:, 0] > width - 1)
            | (values[:, 1] < 0)
            | (values[:, 1] > height - 1)
        )
        if np.any(outside):
            index = int(np.flatnonzero(outside)[0])
            raise ValueError(
                f"{side} pixel at row {index} is outside frame {self.frame_id!r} "
                f"with size {self.image_size}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "image_size": list(self.image_size),
            "pixel_convention": self.pixel_convention,
            "distortion_corrected": self.distortion_corrected,
        }

    @classmethod
    def from_dict(cls, value: object) -> "PixelFrame":
        if not isinstance(value, dict):
            raise ValueError("observation_frame must be an object")
        required = {"frame_id", "image_size", "pixel_convention", "distortion_corrected"}
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"observation_frame is missing: {', '.join(missing)}")
        return cls(
            frame_id=value["frame_id"],
            image_size=tuple(value["image_size"]),
            pixel_convention=value["pixel_convention"],
            distortion_corrected=value["distortion_corrected"],
        )


@dataclass(frozen=True)
class CropTransform:
    source_size: tuple[int, int]
    crop_origin: tuple[float, float]
    crop_size: tuple[int, int]
    output_size: tuple[int, int]
    source_id: str

    def __post_init__(self) -> None:
        source_size = _positive_integer_pair(self.source_size, "source_size")
        crop_size = _positive_integer_pair(self.crop_size, "crop_size")
        output_size = _positive_integer_pair(self.output_size, "output_size")
        sw, sh = source_size
        cw, ch = crop_size
        ow, oh = output_size
        x0, y0 = self.crop_origin
        if not np.all(np.isfinite(np.asarray([x0, y0], dtype=float))):
            raise ValueError("crop_origin must be finite")
        if x0 < 0 or y0 < 0 or x0 + cw > sw or y0 + ch > sh:
            raise ValueError("crop must lie inside the source image")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id is required for provenance")
        object.__setattr__(self, "source_size", source_size)
        object.__setattr__(self, "crop_size", crop_size)
        object.__setattr__(self, "output_size", output_size)
        object.__setattr__(self, "crop_origin", (float(x0), float(y0)))

    @property
    def source_to_output_homography(self) -> np.ndarray:
        sx = self.output_size[0] / self.crop_size[0]
        sy = self.output_size[1] / self.crop_size[1]
        x0, y0 = self.crop_origin
        return np.array([[sx, 0.0, -sx * x0], [0.0, sy, -sy * y0], [0.0, 0.0, 1.0]])

    def source_to_output(self, pixels: np.ndarray) -> np.ndarray:
        return apply_homography(pixels, self.source_to_output_homography)

    def output_to_source(self, pixels: np.ndarray) -> np.ndarray:
        return apply_homography(pixels, np.linalg.inv(self.source_to_output_homography))

    def require_compatible_source(self, source_id: str, source_size: tuple[int, int]) -> None:
        if source_id != self.source_id or tuple(source_size) != tuple(self.source_size):
            raise ValueError("pixel frame is incompatible with declared source provenance")


def apply_homography(pixels: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(pixels, dtype=float)
    one = points.ndim == 1
    points = np.atleast_2d(points)
    h = np.asarray(homography, dtype=float)
    if (
        points.shape[1] != 2
        or h.shape != (3, 3)
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(h))
    ):
        raise ValueError("pixels and 3x3 homography must be finite")
    mapped = (h @ np.column_stack((points, np.ones(len(points)))).T).T
    if np.any(np.abs(mapped[:, 2]) < 1e-12):
        raise ValueError("homography maps a point to infinity")
    result = mapped[:, :2] / mapped[:, 2:3]
    if not np.all(np.isfinite(result)):
        raise ValueError("homography produced non-finite pixels")
    return result[0] if one else result
