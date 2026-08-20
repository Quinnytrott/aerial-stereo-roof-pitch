# Publication safety

This repository is a sanitized, review-pending public derivative. Publication
readiness is **READY AFTER DATA/IMAGE REVIEW**, which explicitly includes
privacy/safety review and a separately recorded manual publication approval. It
is not currently approved or accuracy validated.

Allowed tracked material is general mathematical code, public methodology,
analytic synthetic fixtures/results, public dataset citation, and safe relative
path templates. Code is MIT licensed. CC0 covers only the project-created
`examples/example_config.json`, `examples/expected_summary.json`,
`examples/synthetic/*`, and `results/synthetic/*`. Ontario data retains the Open
Government Licence – Ontario. Third-party calibration reports are excluded.

Do not track source or target-derived imagery, addresses, coordinates, exact
orientations, private crops, elevations, hashes, dense outputs, calibration
reports, credentials, signed URLs, account IDs, private filesystem paths,
customer records, or private application/benchmark artifacts. Generated outputs
should remain untracked until rights and privacy are manually reviewed.

Before publication, inspect all files and symlinks; scan for secrets, token-bearing
URLs, private paths and identifiers; verify data attribution and third-party
rights; rerun tests and deterministic summary comparison; and record manual
approval. An unknown item blocks publication.

A sanitized historical Ontario derivative was verified during preparation but
is deliberately absent: it remains unapproved until that same review and manual
approval gate is completed.
