"""Stereo-pair identity and camera configuration."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .camera import PinholeCamera


PAIR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class StereoPair:
    pair_id: str
    left: PinholeCamera
    right: PinholeCamera
    roof_face_scope_id: str = "roof-face"

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not PAIR_ID_PATTERN.fullmatch(self.pair_id):
            raise ValueError("pair_id must use filename-safe letters, digits, dot, underscore, or hyphen")
        if not isinstance(self.roof_face_scope_id, str) or not self.roof_face_scope_id.strip():
            raise ValueError("roof_face_scope_id must be a non-empty string")
        baseline = float(((self.right.center_world - self.left.center_world) ** 2).sum() ** 0.5)
        if baseline <= 0.0:
            raise ValueError("stereo cameras must have distinct centers")
        if self.left.frame_id == self.right.frame_id:
            raise ValueError("stereo cameras must reference distinct observation frames")
        if self.left.camera_id == self.right.camera_id:
            raise ValueError("stereo cameras must reference distinct camera IDs")
        if self.left.source_id == self.right.source_id:
            raise ValueError("stereo cameras must reference distinct source images")
        if (
            self.left.world_frame_id != self.right.world_frame_id
            or self.left.world_units != self.right.world_units
        ):
            raise ValueError("stereo cameras must use the same world frame and units")

    @property
    def world_frame_id(self) -> str:
        return self.left.world_frame_id

    @property
    def world_units(self) -> str:
        return self.left.world_units

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "StereoPair":
        try:
            return cls(
                pair_id=str(value["pair_id"]),
                left=PinholeCamera.from_dict(value["left"]),  # type: ignore[arg-type]
                right=PinholeCamera.from_dict(value["right"]),  # type: ignore[arg-type]
                roof_face_scope_id=str(value["roof_face_scope_id"]),
            )
        except KeyError as exc:
            raise ValueError(f"stereo pair is missing {exc.args[0]}") from exc
