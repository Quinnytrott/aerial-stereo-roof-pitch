# Aerial Stereo Roof Pitch

This standalone experimental photogrammetry project asks a deliberately narrow
question: **can calibrated overlapping aerial imagery reconstruct a physical
roof plane and estimate its pitch with measurable internal geometric
consistency?** It implements explicit cameras, provenance-aware image frames,
ray triangulation, quality filtering, robust plane fitting, pair-balanced
cross-flight fitting, and pitch conversion. Its only accuracy ground truth is
analytic synthetic truth.

> Agreement between stereo pairs demonstrates internal geometric consistency,
> not independent measurement accuracy.

This is research software, not production measurement software. It does not
depend on RoofBench or private application code.

## Why I Built This

A single nadir image can support two-dimensional roof tracing, but roof pitch is
a physical three-dimensional orientation. Overlapping calibrated images provide
parallax: matched image observations define rays, ray geometry reconstructs XYZ
points, and points on one roof face constrain a plane. Making every frame,
filter, rejection, and conversion visible is essential to reviewing whether the
result is geometrically coherent.

## Pipeline

```mermaid
flowchart TD
    A[Ontario source acquired by user\nor analytic synthetic scene] --> B[Calibration and exterior orientation]
    B --> C[Pair selection]
    C --> D[Source/crop provenance]
    D --> E[Diagnostic rectification/image preparation or epipolar check]
    E --> F[Correspondences]
    F --> G[Closest-ray triangulation]
    G --> H[Positive depth, ray miss, angle, reprojection filters]
    H --> I[Separately processed distinct-pair XYZ clouds]
    I --> J[Robust per-pair roof-plane fits]
    J --> K[Equal-total-weight shared refit]
    K --> L[Normal, angle, and x/12 pitch]
    L --> M[Diagnostics and threshold stability]
```

## Run the synthetic known-truth example

Python 3.11+ and NumPy are sufficient. From a fresh checkout, without installing
the package. Each command that writes artifacts requires a new, non-existing
output directory:

```bash
PYTHONPATH=src python3 scripts/run_example.py --output-dir outputs/example
PYTHONPATH=src python3 -m unittest discover -s tests -v
cmp outputs/example/summary.json examples/expected_summary.json
```

The deterministic experiment includes noiseless planes at 0/12, 4/12, 6/12,
8/12, and 12/12; a declared seeded noise/outlier scene; and a deliberately
degenerate stereo scene that must refuse. The tracked summary records scored,
refused, failed, and missing scenes separately. Synthetic truth checks pipeline
correctness under the declared camera/noise model; it does not validate accuracy
on real aerial imagery.

## Method in brief

Each pinhole camera maps world XYZ through a declared world-to-camera rotation,
camera center, and intrinsic matrix. A correspondence is back-projected into two
world rays. Their closest points produce an XYZ midpoint and a ray-miss distance;
positive depth, intersection angle, and reprojection errors are reported and
filtered. Pair A and Pair B are separately processed distinct stereo evidence
before any shared fit; they are not assumed statistically independent.

The fitted roof face is

```text
Z = a(X - Xref) + b(Y - Yref) + Zref
```

with upward unit normal `n = normalize([-a, -b, 1])`. The slope magnitude is
`sqrt(a²+b²)`, angle is `atan(slope)`, and pitch is `12*slope` inches of rise per
12 inches of run. RANSAC exposes threshold, inliers, support, RMS residual, XY
spread, and rejection reasons. A shared refit gives each accepted pair equal
total weight, so duplicating all observations in one pair cannot give that pair
more influence. The common model must retain support and XY spread in every pair,
and actual convex support footprints must overlap. Per-pair common-model metrics
and pairwise overlap fractions remain visible. See
[methodology](docs/methodology.md).

## Historical Ontario context (not ground truth)

A sanitized historical derivative was verified during preparation, but it is
unapproved and intentionally excluded from this public-shaped repository. Any
future historical result requires privacy/safety and data/image review plus a
separately recorded manual publication approval. No independent pitch ground
truth exists for that historical roof. Publication readiness remains **READY
AFTER DATA/IMAGE REVIEW**, not approved or accuracy validated.

## Ontario data

The acquisition lane is the **South Central Ontario Orthophotography Project
(SCOOP) 2023** from the Ontario Ministry of Natural Resources / Land Information
Ontario. This repository never downloads, copies, or redistributes it. Users must
obtain imagery and aerial-triangulation/orientation material through the official
record and document their rights. Required attribution is: “Contains information
licensed under the Open Government Licence – Ontario.” See
[Ontario data and rights](docs/ontario-data.md) and the safe
[`ontario-scoop-example.json`](examples/ontario-scoop-example.json) template.

## Validation philosophy

Pair agreement, low residuals, low ray miss, and threshold stability can expose
contradictions and establish internal consistency within the same calibration
and imagery system. They cannot reveal shared systematic errors in calibration,
orientation, matching, roof-face selection, or datum. Independent accuracy needs
an independently acquired, traceable roof-pitch reference with declared
uncertainty and a multi-roof evaluation design. The unit of analysis here is a
synthetic scene/stereo pair and results are aggregated by scene—not by correlated
points.

## Boundaries and limitations

- Crop/source transforms are explicit; incompatible frames refuse rather than
  silently reinterpret pixels.
- Real dense image matching and plots are optional (`images` and `plots` extras)
  and imported lazily. The core CSV/synthetic path uses only NumPy.
- Rectification maps are diagnostic image-preparation outputs, not triangulation
  cameras. Rectified pixels require a separately derived explicit camera/frame
  model, which this repository does not implement.
- Occlusion, vegetation, shadows, low texture, repeated shingles, orientation and
  calibration error, disparity ambiguity, and mixed roof faces remain hazards.
- The Ontario path is operator-supplied and not claimed to be fully automatic.
- Source imagery, derived target imagery, exact orientations, calibration
  reports, coordinates, hashes, and private results are excluded.

Code is MIT licensed. `examples/example_config.json`,
`examples/expected_summary.json`, `examples/synthetic/*`, and
`results/synthetic/*` are project-created synthetic material dedicated to the
public domain under CC0 terms described in `examples/synthetic/README.md`.
Ontario source data retains the Open Government Licence – Ontario; third-party
calibration documents are not redistributed.

## Relationship to MeasureAgent

This research originated while investigating imagery-derived roof measurements
for [MeasureAgent](https://measureagent.ca); the commercial application remains
private and is not required by this repository.

Author profile: [github.com/Quinnytrott](https://github.com/Quinnytrott)
