"""Fail-fast validation that the exported ML artifacts the YAML names exist."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("leafmachine3")

_MODEL_STAGES = ("mp_conversion_factor", "archival_detector", "plant_detector", "specimen_segmenter",
                 "leaf_segmenter", "landmark_detector")


def validate_ml_artifacts(cfg) -> None:
    """Raise FileNotFoundError if an enabled stage's model artifact is missing.

    Skipped entirely in mock mode (``compute.mock: true``), which uses synthetic models.
    """
    if bool(getattr(cfg.compute, "mock", False)):
        log.info("mock mode -- skipping ML artifact validation")
        return

    missing: list[tuple[str, str]] = []
    for key in _MODEL_STAGES:
        if not cfg.is_enabled(key):
            continue
        blk = cfg.stage(key)
        model = getattr(blk, "model", None)
        path = getattr(model, "path", None) if model is not None else None
        if path and not Path(cfg.resolve_path(path)).exists():
            missing.append((key, str(path)))

    if cfg.is_enabled("ruler_classifier"):
        d = getattr(cfg.stage("ruler_classifier"), "models_dir", None)
        if d and not Path(cfg.resolve_path(d)).exists():
            missing.append(("ruler_classifier", str(d)))

    for key, path in missing:
        log.error("missing model artifact for %s: %s", key, path)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} model artifact(s) missing. Place exports under models/, fix "
            "LM3_settings.yaml paths, or set compute.mock: true to run without models."
        )
