"""Ruler unit-type classifier: a frozen DINOv3 ViT-S/16 with an MLP head, one self-contained ONNX graph.

Every crop is SQUARIFIED first (LM2 ``squarify_tile_four_versions``, 896x896), because that is what
the model was trained on -- see ``ruler_squarify`` for the full rationale. Skipping it feeds the model
a sliver of a long ruler strip. The size and method are properties of the model, not settings: serving
any other input is a train/serve mismatch.

Only ``onnxruntime``, ``numpy`` and ``cv2`` are needed (no torch). ``label_map.json`` beside the graph
names the classes.

``predict(image)`` returns::

    {"label": cls, "conf": p, "top3": [[cls, p], ...]}

This backend is **best-effort**: if the graph, the label map or ``onnxruntime`` are unavailable it
degrades gracefully and every ``predict`` returns ``{"label": "UNKNOWN"}`` instead of raising, so a run
without the ruler model still completes (the ruler CF stage then has no class to work from).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger("leafmachine3.inference.ruler_classifier")

# The training-time input (LM2 tile_four at sz=448 is a collage of 2*sz).
SQUARIFY_SZ = 448
SQUARIFY_METHOD = "tile_four"

# ImageNet statistics, the normalization the DINOv3 backbone was trained with.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

UNKNOWN = {"label": "UNKNOWN"}


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)


def _load_label_map(model_path: str) -> list[str]:
    """The ordered class list from ``label_map.json`` beside the graph."""
    path = os.path.join(os.path.dirname(model_path), "label_map.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    classes = data.get("classes") or data.get("idx_to_name")
    if not classes:
        raise ValueError(f"no classes in {path}")
    return [str(c) for c in classes]


class RulerClassifierModel:
    """The ONNX classifier plus its mandatory squarify.

    Construction never raises on missing artifacts: it logs a warning and enters a degraded state
    where ``predict`` returns ``{"label": "UNKNOWN"}``.
    """

    def __init__(self, model_path, providers: Optional[Sequence] = None) -> None:
        from leafmachine3.inference.ruler_squarify import RulerSquarifier

        self.model_path = str(model_path)
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        # augment=False keeps it deterministic and pixel-identical to the LM2 oracle.
        self._squarifier = RulerSquarifier(sz=SQUARIFY_SZ, method=SQUARIFY_METHOD, augment=False)
        self.classes: list[str] = []
        self.session = None
        self.ok = False
        try:
            self._load()
            self.ok = True
        except Exception as exc:  # noqa: BLE001 - best-effort backend
            log.warning("ruler classifier unavailable (%s); ruler classification -> UNKNOWN", exc)

    def _load(self) -> None:
        import onnxruntime as ort

        from leafmachine3.inference import coreml
        from leafmachine3.inference.providers import session_options

        self.classes = _load_label_map(self.model_path)
        # ViTs run from ONNX on a Mac's CPU; there is no Core ML package of this model.
        self.session = ort.InferenceSession(self.model_path, session_options(),
                                            providers=coreml.for_model(self.providers, False))
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        # The class names come from the label map, the scores from the graph: if their lengths
        # differ, every index past the change is silently misnamed. Refuse to load instead.
        n_out = self.session.get_outputs()[0].shape[-1]
        if isinstance(n_out, int) and n_out != len(self.classes):
            raise ValueError(f"{self.model_path} outputs {n_out} classes but the label map has "
                             f"{len(self.classes)}; install the model and label_map.json from the same lock")

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """Squarified BGR uint8 -> NCHW float32: RGB, [0, 1], ImageNet mean/std."""
        rgb = bgr[..., ::-1].astype(np.float32) / 255.0
        chw = np.transpose((rgb - _IMAGENET_MEAN) / _IMAGENET_STD, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(chw, dtype=np.float32)

    def predict(self, image) -> dict:
        """Classify one ruler crop (path | BGR ndarray | PIL)."""
        if not self.ok:
            return dict(UNKNOWN)
        try:
            batch = self._preprocess(self.squarify_tile(image))
            logits = self.session.run([self._output_name], {self._input_name: batch})[0]
        except Exception as exc:  # noqa: BLE001 - never crash the pipeline on one crop
            log.warning("ruler classifier predict failed (%s) -> UNKNOWN", exc)
            return dict(UNKNOWN)
        probs = _softmax(logits)[0]
        top = np.argsort(-probs)[:3]
        return {"label": self.classes[int(top[0])], "conf": float(probs[top[0]]),
                "top3": [[self.classes[int(i)], round(float(probs[i]), 6)] for i in top]}

    def squarify_tile(self, image) -> np.ndarray:
        """The four-tile collage the classifier sees, BGR ndarray (also the pre-made QC tile)."""
        import cv2

        src = image
        if not isinstance(src, (str, os.PathLike, np.ndarray)):
            # PIL -> BGR ndarray (transform() only accepts a path or an ndarray)
            src = cv2.cvtColor(np.asarray(src.convert("RGB")), cv2.COLOR_RGB2BGR)
        return self._squarifier.transform(src)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RulerClassifierModel(ok={self.ok}, model={self.model_path})"
