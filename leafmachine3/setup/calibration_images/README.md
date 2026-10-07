# Packaged calibration images

Six downscaled herbarium sheets used by `lm3-setup --calibrate` to MEASURE per-worker VRAM.

- Source: the first six files of `<repo>/examples/images` (Python `sorted()` order), copied — the
  originals are deliberately kept in full resolution and are not touched.
- Transform: long side capped at 3200 px (the default `ingest.max_working_dim`), JPEG quality 90.
- Why they live here: `leafmachine3.setup.calibrate` resolves them with `importlib.resources`, so
  calibration works from a wheel or container image instead of only from a source checkout.

Replace or extend the set only if the replacements still exercise every GPU stage (rulers, labels,
leaves, petioles); a set that skips a stage silently leaves that stage on its heuristic estimate.
