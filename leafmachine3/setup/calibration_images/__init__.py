"""Bundled herbarium sheets the VRAM calibration run measures against.

These are PACKAGE DATA, not example inputs: :mod:`leafmachine3.setup.calibrate` resolves them
through ``importlib.resources`` (:func:`leafmachine3.core.paths.calibration_images_dir`) so a
calibration pass works from an installed wheel or a container image and can never depend on the
directory the user happened to launch from (plan section 3.1, precedence-table row 7).

They are downscaled copies of the first six sheets of ``examples/images`` -- the exact set
``_stage_images`` already picked, since it takes ``sorted(...)[:6]`` -- resized so the long side
is 3200 px, which is ``ingest.max_working_dim``'s default, and re-encoded at JPEG q90. The
originals stay in ``examples/`` untouched; downscaling here is what makes a 212 MB tree into a
7 MB wheel payload without changing what the pipeline sees, because every detector resizes to its
own fixed model input anyway.

The six were chosen for coverage rather than beauty: between them they carry rulers, printed and
handwritten labels, whole leaves and petioles, so every GPU stage does real work and the measured
per-worker VRAM is a peak the allocator can trust.
"""
