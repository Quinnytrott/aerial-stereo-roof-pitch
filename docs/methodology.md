# Methodology

The pipeline processes Pair A and Pair B separately through correspondence,
triangulation, filtering, and robust plane fitting. They are distinct stereo
evidence, not assumed statistically independent. Only accepted per-pair plane
support enters the shared fit.

For pixel `u=[x,y,1]`, a ray is `C + lambda R^T K^-1 u`, normalized in the
world frame. For two rays, least squares finds their closest points. Their
midpoint is the XYZ estimate; their separation is ray miss. The implementation
also reports camera depth, acute intersection angle, and reprojection error.
Non-positive depth, weak intersection geometry, excessive ray miss, or excessive
reprojection error has a named refusal reason.

RANSAC samples three points and fits

```text
Z = a(X - Xref) + b(Y - Yref) + Zref.
```

Hypotheses are ranked by inlier count and then inlier RMS. The accepted support
is refit by least squares and must satisfy declared support and two-dimensional
XY-spread requirements. The report exposes threshold, input and inlier counts,
support fraction, RMS Z residual, XY minor spread, normal, angle, and pitch.

For the shared refit, each pair has total weight `1 / number_of_pairs`; each
accepted point within a pair has an equal share of that pair's weight. Exact
duplicate observations are removed. Thus duplicating one whole cloud does not
make it dominate. The refined shared model must itself meet minimum support and
XY spread in every pair, and the accepted pair footprints must overlap. Inlier
masks are refit to bounded convergence; incompatible slopes or vertical offsets
refuse instead of being averaged. Multiple residual thresholds are rerun as a
stability analysis, never treated as a magic accuracy threshold.

Footprint compatibility uses each pair's two-dimensional convex hull and the
actual convex-polygon intersection area. Every pair combination must satisfy
`intersection_area / min(hull_area_a, hull_area_b)` at the declared minimum;
degenerate hulls refuse. The shared result reports, for each pair, common-model
input count, inliers, support, RMS Z residual, and XY spread, plus named pairwise
overlap fractions. RMS is a diagnostic on shared-model inliers; every reported
inlier is also bounded by the declared residual threshold.

`fit_roof_plane.py` exposes and records the primary and stability residual
thresholds, seed, RANSAC iterations, minimum support, minimum XY spread, and
minimum convex-footprint overlap. The same effective controls are applied to the
primary shared fit and every threshold-stability rerun; each fit record also
retains its residual threshold and seed.

The upward normal is `normalize([-a,-b,1])`, slope is `sqrt(a^2+b^2)`, angle is
`atan(slope)`, and `x/12 = 12*slope`.
