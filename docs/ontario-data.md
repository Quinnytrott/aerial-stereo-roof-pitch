# Ontario acquisition and reproduction lane

Dataset: **South Central Ontario Orthophotography Project (SCOOP) 2023**.

- Owner/publisher: Ontario Ministry of Natural Resources / Land Information Ontario.
- Licence: Open Government Licence – Ontario.
- Attribution: “Contains information licensed under the Open Government Licence – Ontario.”
- Official record and ordering: [Ontario GeoHub dataset record](https://geohub.lio.gov.on.ca/maps/a17b2922c68a4ee2ac3f3a1103b4ca69/about).
- Licence text: [Open Government Licence – Ontario](https://www.ontario.ca/page/open-government-licence-ontario).

The licence permits use, adaptation, and distribution with attribution, subject
to its terms, but does not grant rights in third-party material. An inspected
Vexcel calibration report states that it may not be reproduced without written
consent, so no such report is included.

This repository excludes all real imagery and target-derived imagery despite the
Ontario licence, both for target privacy/publication policy and repository size.
It also excludes exact orientations, coordinates, crops, hashes, disparity,
point clouds, and calibration documents. Users must acquire source imagery and
aerial-triangulation/orientation material themselves through the official record,
confirm their applicable rights, and keep it outside version control.

Copy `examples/ontario-scoop-example.json` to a private untracked location, set
`rights_acknowledged` only after review, and use relative private-data paths:

```bash
python3 scripts/prepare_dataset.py --config path/to/private-config.json
python3 scripts/run_stereo_pair.py --config path/to/pair-a.json --correspondences path/to/pair-a.csv --output-dir outputs/pair-a
python3 scripts/run_stereo_pair.py --config path/to/pair-b.json --correspondences path/to/pair-b.csv --output-dir outputs/pair-b
python3 scripts/fit_roof_plane.py --pair-a outputs/pair-a/pair-a-xyz.csv --pair-b outputs/pair-b/pair-b-xyz.csv --output outputs/fit.json
```

`prepare_dataset.py` validates only; it never downloads or copies files. The CSV
path assumes correspondences and calibrated camera models have already been
prepared. Supplied intrinsics must already be expressed in each declared
observation frame; private numeric crop/calibration derivation stays external.
Use a new output directory for every artifact-producing command. No claim of
fully automatic Ontario reproduction is made.
