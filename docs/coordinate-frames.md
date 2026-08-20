# Coordinate frames and provenance

World XYZ uses metres in the synthetic example, Z up. Camera coordinates use x
right, y down, z forward. Source and output pixels use top-left origin, x right,
y down.

A crop is not just an array slice. `CropTransform` records source dimensions,
source identity, crop origin and dimensions, output dimensions, and the exact
homography between source and output pixels. A source point maps as:

```text
x_out = (x_source - crop_x) * output_width / crop_width
y_out = (y_source - crop_y) * output_height / crop_height
```

The implementation rejects a mismatched source identity or size. Orientation,
undistortion, padding, and provider transforms would require additional explicit
transforms; callers must not silently reinterpret those pixels.

Executable correspondence CSVs repeat an immutable observation contract for
both sides: frame ID, image width/height, the exact top-left/x-right/y-down
convention, and `distortion_corrected=true`. They also carry pair ID, camera IDs,
source-image IDs, world frame ID, and world units. These values must exactly
match the configured cameras before backprojection, and every pixel must lie
inside its declared frame. A required `roof_face_scope_id` binds both stereo
pairs to the same declared roof face; shared fitting refuses a scope mismatch.

The configured observation frame is the output frame of any `CropTransform`.
Its frame ID and size therefore bind cropped pixels back to declared source
provenance without exposing a real source path. XYZ output preserves pair,
roof-face scope, camera, source, observation-frame, world-frame, and unit
lineage; shared fitting refuses missing or incompatible lineage and reuse of the
same camera/source evidence across pairs.

The current CLI persists that safe frame contract but does not derive camera
intrinsics from a crop. The supplied `K` must already be expressed in the
declared observation frame. Numeric private crop-transform and calibration
provenance remains external to the public artifact. Rectified pixels likewise
need a separately derived explicit camera/frame model and otherwise refuse.
