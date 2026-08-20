"""Correspondence records and transparent CSV interchange."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import PinholeCamera
from .frames import PIXEL_CONVENTION, PixelFrame


FRAME_COLUMNS = {
    "pair_id",
    "roof_face_scope_id",
    "world_frame_id",
    "world_units",
    "left_camera_id",
    "left_source_id",
    "left_frame_id",
    "left_image_width",
    "left_image_height",
    "left_pixel_convention",
    "left_distortion_corrected",
    "right_frame_id",
    "right_image_width",
    "right_image_height",
    "right_pixel_convention",
    "right_distortion_corrected",
    "right_camera_id",
    "right_source_id",
}


@dataclass(frozen=True)
class CorrespondenceLineage:
    pair_id: str
    roof_face_scope_id: str
    world_frame_id: str
    world_units: str
    left_camera_id: str
    left_source_id: str
    right_camera_id: str
    right_source_id: str

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.world_units != "metres":
            raise ValueError("world_units must be 'metres'")


@dataclass(frozen=True)
class CorrespondenceSet:
    left_pixels: np.ndarray
    right_pixels: np.ndarray
    ids: tuple[str, ...]
    left_frame: PixelFrame
    right_frame: PixelFrame
    lineage: CorrespondenceLineage

    def __post_init__(self) -> None:
        left = np.asarray(self.left_pixels, dtype=float)
        right = np.asarray(self.right_pixels, dtype=float)
        if left.ndim != 2 or left.shape[1] != 2 or right.shape != left.shape:
            raise ValueError("correspondence pixels must have matching shape (N, 2)")
        if len(left) == 0 or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError("correspondences must be non-empty and finite")
        if len(self.ids) != len(left) or len(set(self.ids)) != len(self.ids):
            raise ValueError("correspondence ids must be unique and match the observations")
        self.left_frame.validate_pixels(left, "left")
        self.right_frame.validate_pixels(right, "right")
        object.__setattr__(self, "left_pixels", left)
        object.__setattr__(self, "right_pixels", right)

    def require_compatible_cameras(
        self, pair: object
    ) -> None:
        left_camera = pair.left
        right_camera = pair.right
        self.left_frame.require_compatible(left_camera.observation_frame)
        self.right_frame.require_compatible(right_camera.observation_frame)
        expected = CorrespondenceLineage(
            pair.pair_id,
            pair.roof_face_scope_id,
            pair.world_frame_id,
            pair.world_units,
            left_camera.camera_id,
            left_camera.source_id,
            right_camera.camera_id,
            right_camera.source_id,
        )
        if self.lineage != expected:
            raise ValueError("correspondence lineage is incompatible with configured stereo pair")


def _strict_bool(value: str, name: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{name} must be 'true' or 'false'")


def _frame_from_row(row: dict[str, str], side: str) -> PixelFrame:
    return PixelFrame(
        frame_id=row[f"{side}_frame_id"],
        image_size=(int(row[f"{side}_image_width"]), int(row[f"{side}_image_height"])),
        pixel_convention=row[f"{side}_pixel_convention"],
        distortion_corrected=_strict_bool(
            row[f"{side}_distortion_corrected"], f"{side}_distortion_corrected"
        ),
    )


def _lineage_from_row(row: dict[str, str]) -> CorrespondenceLineage:
    return CorrespondenceLineage(
        pair_id=row["pair_id"],
        roof_face_scope_id=row["roof_face_scope_id"],
        world_frame_id=row["world_frame_id"],
        world_units=row["world_units"],
        left_camera_id=row["left_camera_id"],
        left_source_id=row["left_source_id"],
        right_camera_id=row["right_camera_id"],
        right_source_id=row["right_source_id"],
    )


def read_correspondence_csv(path: str | Path) -> CorrespondenceSet:
    required = {"id", "x_left", "y_left", "x_right", "y_right"} | FRAME_COLUMNS
    rows: list[dict[str, str]] = []
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"CSV must include columns: {', '.join(sorted(required))}")
            rows.extend(reader)
        ids = tuple(row["id"] for row in rows)
        if not rows:
            raise ValueError("CSV must contain at least one correspondence")
        left = np.array([[float(row["x_left"]), float(row["y_left"])] for row in rows])
        right = np.array([[float(row["x_right"]), float(row["y_right"])] for row in rows])
        left_frame = _frame_from_row(rows[0], "left")
        right_frame = _frame_from_row(rows[0], "right")
        lineage = _lineage_from_row(rows[0])
        for index, row in enumerate(rows[1:], start=1):
            if (
                _frame_from_row(row, "left") != left_frame
                or _frame_from_row(row, "right") != right_frame
                or _lineage_from_row(row) != lineage
            ):
                raise ValueError(f"frame metadata changes at CSV row {index}")
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"Could not read correspondence CSV {path}: {exc}") from exc
    return CorrespondenceSet(left, right, ids, left_frame, right_frame, lineage)
