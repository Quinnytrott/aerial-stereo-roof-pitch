# Synthetic fixture rights

To the extent possible under law, the project author has waived all copyright
and related or neighboring rights under CC0 1.0 only for these project-created
synthetic materials:

- `examples/example_config.json`
- `examples/expected_summary.json`
- `examples/synthetic/*`
- `results/synthetic/*`
- `docs/assets/synthetic-roof-plane-reconstruction.png`

No real place, property, source image, or private dataset is represented.

The PNG is generated from the tracked seeded synthetic 6/12 scene through the
same `run_scene_with_trace` execution used by the golden summary:

```bash
python scripts/render_synthetic_example.py \
  --output docs/assets/synthetic-roof-plane-reconstruction.png
```

The output path must not already exist. Cross-platform PNG bytes are not part of
the scientific golden contract; tests verify the public scene semantics,
dimensions, safe metadata, and successful regeneration. The renderer code is
MIT licensed with the rest of the software.
