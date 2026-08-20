# Limitations

- Internal pair agreement cannot reveal errors shared by calibration,
  orientations, datum, matching, or manual roof-face scope.
- Occlusion, vegetation, shadows, low texture, repeated shingles, specular
  surfaces, and resampling can create missing or ambiguous correspondence.
- A building can contain several physical planes; pooling faces biases the fit.
- Closest-ray midpoint is a simple transparent triangulator, not a full bundle
  adjustment or uncertainty model.
- RANSAC settings are synthetic experiment settings, not universal cutoffs.
- Point residuals are correlated within imagery and are not independent accuracy
  samples; evaluation aggregates by scene.
- Optional OpenCV support is only an adapter boundary. The public repository does
  not claim automatic dense matching on Ontario data. Its rectification maps are
  diagnostic/image-preparation outputs and cannot be used for triangulation
  without a separately derived explicit rectified camera/frame model.
- Camera intrinsics are not automatically transformed for a crop; supplied
  intrinsics must already match the declared observation frame.
- Artifact-producing CLIs require a unique, non-existing output directory and
  atomically install the complete staged directory rather than merging outputs.
- Successful stereo runs write point-level audits. Preconditions that fail before
  an output directory is accepted report to stderr but do not yet create a
  durable machine-readable refusal record.
- No independent real-roof pitch truth, population-level error distribution, or
  operational suitability has been established.
