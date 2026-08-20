"""Transparent calibrated-stereo roof-plane research primitives."""

from .camera import PinholeCamera
from .plane_fit import PlaneModel, PlaneFitResult
from .pitch import pitch_from_plane

__all__ = ["PinholeCamera", "PlaneModel", "PlaneFitResult", "pitch_from_plane"]
__version__ = "0.1.0"
