"""Inference factory — the single entry point stages use to obtain a backend.

Each loader returns a **mock** backend when ``cfg.compute.mock`` is set, otherwise the
**real** backend. Real backends lazy-import their heavy dependency (onnxruntime) so importing this module never requires them; only actually loading a real
model does.

Return contracts
----------------
* ``load_detector``      -> object with ``.predict(image)->list[Detection]``.
* ``load_segmenter``     -> object with ``.predict(image)->list[Instance]`` (crop coords).
* ``load_ruler_ensemble``-> object with ``.predict(image)->{"ensemble": str, ...}``.
* ``load_landmark_pose`` -> object with ``.predict(crop)->list[{kpt_name:(x,y,conf)}]`` (crop coords).
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("leafmachine3.inference.factory")


def _get(section: Any, key: str, default: Any = None) -> Any:
    """Tolerant read from a Section / dict / attr-bearing object."""
    if section is None:
        return default
    if hasattr(section, "get"):
        try:
            val = section.get(key, default)
            return default if val is None else val
        except Exception:  # noqa: BLE001
            pass
    val = getattr(section, key, default)
    return default if val is None else val


def _is_mock(cfg: Any) -> bool:
    return bool(_get(cfg.compute, "mock", False))


def _model_path(cfg: Any, stage_cfg: Any) -> str:
    raw = _get(_get(stage_cfg, "model", None), "path", None)
    if raw is None:
        raise ValueError("stage config is missing model.path")
    return cfg.resolve_path(raw)


def load_mp_conversion_factor(cfg: Any, device: Any = None):
    """Return the resolution->CF linear predictor (loads ``model.json``; baked fallback on error).

    Not gated by mock mode -- the artifact is a tiny 2-coefficient JSON, loaded identically in
    mock and real runs so the prediction is deterministic.
    """
    from leafmachine3.inference.mp_conversion_factor import MpConversionFactorModel, load_model

    # Tolerant of a cfg-less caller. The stage now asks the model which FRAME its form is defined
    # on before it can even collect items, so this is reached from more places than the warm-build,
    # and the fallback coefficients are a valid answer everywhere -- the same reasoning that already
    # makes an unreadable artifact a warning rather than a crash.
    try:
        stage_cfg = cfg.stage("mp_conversion_factor")
        raw = _get(_get(stage_cfg, "model", None), "path", None) or "models/mp_conversion_factor/model.json"
        return load_model(cfg.resolve_path(raw))
    except Exception:  # noqa: BLE001
        return MpConversionFactorModel()


def load_detector(cfg: Any, stage_key: str, device: Any):
    """Return a detector backend for the given detector stage (``archival``/``plant``)."""
    stage_cfg = cfg.stage(stage_key)
    classes = list(_get(stage_cfg, "classes", []) or [])
    if _is_mock(cfg):
        from leafmachine3.inference.mock import MockDetector

        return MockDetector(classes)

    from leafmachine3.inference.detector import YoloDetector

    return YoloDetector(
        _model_path(cfg, stage_cfg),
        class_names=classes or None,
        conf=float(_get(stage_cfg, "conf", 0.25)),
        iou=float(_get(stage_cfg, "iou", 0.45)),
        imgsz=_get(stage_cfg, "imgsz", None),
        device=device.torch_str,
        providers=device.ort_providers(),
        max_det=_get(stage_cfg, "max_det", None),
    )


def load_segmenter(cfg: Any, device: Any):
    """Return an instance-segmentation backend for the leaf segmenter stage."""
    stage_cfg = cfg.stage("leaf_segmenter")
    if _is_mock(cfg):
        from leafmachine3.inference.mock import MockSegmenter

        return MockSegmenter()

    from leafmachine3.inference.leaf_segmenter import YoloSegmenter

    mask_storage = _get(stage_cfg, "mask_storage", {})
    min_area_px = int(mask_storage.get("min_area_px", 64)) if isinstance(mask_storage, dict) else 64
    return YoloSegmenter(
        _model_path(cfg, stage_cfg),
        imgsz=_get(stage_cfg, "imgsz", None),
        conf=float(_get(stage_cfg, "conf", 0.30)),
        iou=float(_get(stage_cfg, "iou", 0.50)),
        retina_masks=bool(_get(stage_cfg, "retina_masks", True)),
        device=device.torch_str,
        providers=device.ort_providers(),
        max_det=_get(stage_cfg, "max_det", None),
        min_area_px=min_area_px,
    )


def load_specimen_segmenter(cfg: Any, device: Any):
    """Return the whole-specimen segmentation backend (UNet++ ONNX + paperclean)."""
    stage_cfg = cfg.stage("specimen_segmenter")
    if _is_mock(cfg):
        from leafmachine3.inference.mock import MockSpecimenSegmenter

        return MockSpecimenSegmenter()

    from leafmachine3.inference.specimen_segmenter import OnnxSpecimenSegmenter

    path = _model_path(cfg, stage_cfg)
    return OnnxSpecimenSegmenter(
        path,
        providers=device.ort_providers(),
        imgsz=int(_get(stage_cfg, "imgsz", 1024)),
        conf=float(_get(stage_cfg, "conf", 0.5)),
        paperclean=bool(_get(stage_cfg, "paperclean", True)),
        model_name=os.path.splitext(os.path.basename(path))[0],
    )


def load_ruler_ensemble(cfg: Any, device: Any):
    """Return the ruler unit-type classifier ensemble backend."""
    if _is_mock(cfg):
        from leafmachine3.inference.mock import MockEnsemble

        return MockEnsemble()

    from leafmachine3.inference.ruler_ensemble import RulerEnsemble

    stage_cfg = cfg.stage("ruler_classifier")
    models_dir = cfg.resolve_path(_get(stage_cfg, "models_dir", "models/ruler_classifier"))
    # squarify is mandatory (training-time preprocessing); only its shape is tunable
    sq = _get(stage_cfg, "squarify", None) or {}
    _sq = (lambda k, d: _get(sq, k, d)) if sq else (lambda k, d: d)
    return RulerEnsemble(
        models_dir,
        providers=device.ort_providers(),
        members=_get(stage_cfg, "ensemble_members", None),
        squarify_sz=int(_sq("sz", 720)),
        squarify_method=str(_sq("method", "tile_four")),
    )


def load_landmark_pose(cfg: Any, device: Any):
    """Return the leaf-landmark pose backend (31-kpt mid15_pet5)."""
    stage_cfg = cfg.stage("landmark_detector")
    if _is_mock(cfg):
        from leafmachine3.inference.mock import MockLandmarkPose

        return MockLandmarkPose()

    from leafmachine3.inference.landmark_pose import LeafLandmarkPose

    return LeafLandmarkPose(
        _model_path(cfg, stage_cfg),
        conf=float(_get(stage_cfg, "conf", 0.25)),
        iou=float(_get(stage_cfg, "iou", 0.45)),
        imgsz=_get(stage_cfg, "imgsz", 640),
        device=device.torch_str,
        providers=device.ort_providers(),
        white_pad=float(_get(stage_cfg, "white_pad", 0.10)),
    )
