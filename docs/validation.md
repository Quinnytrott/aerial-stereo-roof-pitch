# Validation and experiment card

**Decision:** is a safe standalone methodology repository technically runnable?

**Falsifiable hypothesis:** noiseless analytic scenes at 0/12, 4/12, 6/12,
8/12, and 12/12 recover pitch within numerical tolerance; seeded pixel noise
and mismatches remain bounded under the tracked synthetic settings; an
equal-total-weight shared fit is invariant to duplicating either pair; and
invalid or ambiguous geometry refuses.

**Population and unit:** deterministic analytic synthetic scenes. The unit of
analysis is a scene; its stereo pairs and reconstructed points are correlated
within-scene evidence, not independent evaluation samples.

**Ground truth:** the analytic plane used to generate synthetic XYZ and projected
observations. This is the repository's only accuracy ground truth.

**Baseline:** no standalone public project. Earlier private R&D was partial and
not independently reproducible end to end. The Ontario work covers one roof and
has no independent pitch ground truth.

**Go/no-go:** all five noiseless cases must meet numerical tolerance; the tracked
noisy case must remain below its declared 0.05/12 bound; duplication invariance
must hold; the degenerate scene must refuse; and summary bytes must match the
tracked expected file. Missing, refused, failed, and scored scene counts remain
separate.

The deterministic tracked run scores six scenes, refuses one degenerate scene,
fails zero, and marks zero missing. Its noiseless maximum absolute x/12 error is
0.0 at the recorded precision. The noisy 6/12 case recovers 6.002269925/12
(absolute error 0.002269925/12) under the exact public settings in
`examples/example_config.json`.

Canonical summary bytes omit runtime-version strings to improve portability.
Exact byte reproduction is verified only on CPython 3.12.13 with NumPy 2.3.5,
recorded in `requirements-lock.txt` rather than mixed into the scientific
payload. Python 3.11+ remains the declared package support range, not a claim of
byte verification on every such runtime.

This validates implementation behavior under an analytic model. Independent
real-world accuracy requires traceable pitch measurements acquired independently
of the imagery/calibration system, declared reference uncertainty, multiple roof
types and conditions, and scene-level error and refusal reporting.

Three evidence levels must remain separate:

1. **Synthetic known-truth validation:** known cameras, known plane, and known
   pitch verify mathematical and software behavior under the declared fixture.
2. **Internal real-world consistency:** distinct stereo evidence may recover a
   similar physical plane, but shared calibration, imagery, scope, or datum errors
   can agree. A private Ontario case exists at this level and is not included.
3. **Independent ground truth:** separately measured, traceable roof pitch is
   required before any real-world accuracy claim.

The tracked noisy case is one fixed seed, one analytic camera arrangement, and
one planar roof face. It provides neither a population error distribution nor an
uncertainty interval or generalization evidence.
