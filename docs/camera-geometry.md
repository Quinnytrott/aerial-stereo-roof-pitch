# Camera geometry

The camera convention is explicit:

```text
X_camera = R_world_to_camera (X_world - C_world)
p ~ K X_camera
```

Camera x points right, y down, and z forward. Pixels have a top-left origin.
`K` is an invertible 3×3 intrinsic matrix; `R` must be right-handed and
orthonormal; `C` is expressed in the declared world coordinates. The core model
assumes distortion-corrected observations.

The calibrated epipolar diagnostic computes relative rotation and translation,
`E=[t]_x R`, and `F=K_right^-T E K_left^-1`. Sampson error diagnoses whether
correspondences respect the calibrated relation. It does not itself recover
correspondences or prove calibration accuracy.

When the optional OpenCV extra is installed, `calibrated_rectification_maps`
uses the same declared camera geometry to produce explicit left/right remap
arrays for diagnostic/image preparation only. Only OpenCV major 4 is supported.
Those maps do not define triangulation cameras. Rectified pixels must not enter
the current stereo CLI unless the operator supplies a separately derived,
explicit rectified camera and `PixelFrame` model; that derivation is not
implemented. The core synthetic and CSV path does not need OpenCV.

Ray intersection becomes unreliable as intersection angle approaches zero.
Parallel or numerically degenerate rays refuse; the quality layer also rejects
angles below the experiment's declared setting.
