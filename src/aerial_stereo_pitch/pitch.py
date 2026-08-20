"""Unambiguous plane-normal, slope, angle, and x/12 conversions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .plane_fit import PlaneModel


@dataclass(frozen=True)
class Pitch:
    slope: float
    angle_deg: float
    rise_per_12: float
    normal_up: tuple[float, float, float]


def pitch_from_slopes(a: float, b: float) -> Pitch:
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("plane slopes must be finite")
    slope = math.hypot(a, b)
    normal = np.array([-a, -b, 1.0], dtype=float)
    normal /= np.linalg.norm(normal)
    return Pitch(
        slope=slope,
        angle_deg=math.degrees(math.atan(slope)),
        rise_per_12=12.0 * slope,
        normal_up=(float(normal[0]), float(normal[1]), float(normal[2])),
    )


def pitch_from_plane(model: PlaneModel) -> Pitch:
    return pitch_from_slopes(model.a, model.b)


def slope_from_rise_per_12(rise_per_12: float) -> float:
    if not math.isfinite(rise_per_12) or rise_per_12 < 0:
        raise ValueError("rise_per_12 must be finite and non-negative")
    return rise_per_12 / 12.0


def rise_per_12_from_angle(angle_deg: float) -> float:
    if not math.isfinite(angle_deg) or angle_deg < 0 or angle_deg >= 90:
        raise ValueError("angle must be finite in [0, 90) degrees")
    return 12.0 * math.tan(math.radians(angle_deg))
