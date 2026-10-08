"""Ruler unit-type classifier ENSEMBLE — a self-contained 3-model ONNX majority vote.

Every crop is SQUARIFIED first (LM2 ``squarify_tile_four_versions``, 1440x1440), because
that is what all three members were trained on -- see ``ruler_squarify`` for the full
rationale. Skipping it feeds the models ~11% of a long ruler strip and the votes collapse.

Ports the portable ensemble from ``LM3_Ruler_Classifier`` into a single dependency-light
module. Each member is a self-contained ``<member>/exported/model.onnx`` graph
(image-tensor -> logits) with the family-appropriate preprocessing baked in on the
*Python* side here (so no torch / ultralytics import is required — only ``onnxruntime``,
``numpy`` and ``cv2``). All members share ``splits/label_map.json`` so a class index means
the same class everywhere.

``predict(image)`` returns::

    {"ensemble": cls, "x224": cls, "n224": cls, "dinov2mlp": cls}

where ``ensemble`` is the class holding >= 2 of the 3 votes, falling back to the ``x224``
vote (highest macro-F1) on a three-way disagreement.

This backend is **best-effort**: if the model directory, ONNX graphs, or ``onnxruntime``
are unavailable it degrades gracefully and every ``predict`` returns
``{"ensemble": "UNKNOWN"}`` instead of raising, so a run without the ruler models still
completes (the ruler CF stage is disabled anyway).
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger("leafmachine3.inference.ruler_ensemble")

# ImageNet statistics used by the DINOv2 (HF BitImageProcessor-parity) transform.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default ensemble member directories (output-dict key -> directory name). ``x224`` is
# the tie-break fallback so it is listed first.
_DEFAULT_MEMBERS: tuple[tuple[str, str], ...] = (
    ("x224", "yolo26x_cls_224"),
    ("n224", "yolo26n_cls_224"),
    ("dinov2mlp", "dinov2_frozen_mlp"),
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)


def _renormalize(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, None)
    s = probs.sum(axis=-1, keepdims=True)
    return np.divide(probs, s, out=np.zeros_like(probs), where=s > 0)


def resolve_members(members: Optional[Sequence[str]]) -> list[tuple[str, str]]:
    """``[(vote key, member dir name), ...]`` in the fixed tie-break order (x224 first).

    Maps optional config member-dir names onto that ordering. The ONE implementation: the ensemble
    loads exactly these, and the hardware profile fingerprints exactly these (ensemble_files).
    """
    if not members:
        return list(_DEFAULT_MEMBERS)
    provided = {str(m): str(m) for m in members}
    resolved: list[tuple[str, str]] = []
    for key, default_dir in _DEFAULT_MEMBERS:
        match = next(
            (d for d in provided if d == default_dir
             or (key.startswith("dino") and "dino" in d.lower())
             or (key == "x224" and "26x" in d.lower())
             or (key == "n224" and "26n" in d.lower())),
            default_dir,
        )
        resolved.append((key, match))
    return resolved


def ensemble_files(models_dir: str, members: Optional[Sequence[str]] = None) -> list[str]:
    """Every file the ensemble reads: each member's exported/model.onnx + metadata.json, and the
    shared label map. What a change to the ensemble means, for the hardware profile."""
    files = []
    for _key, dir_name in resolve_members(members):
        member = os.path.join(models_dir, dir_name)
        files += [os.path.join(member, "exported", "model.onnx"), os.path.join(member, "metadata.json")]
    for candidate in _label_map_candidates(models_dir):
        if os.path.isfile(candidate):
            files.append(candidate)
            break
    return files


def _label_map_candidates(models_dir: str) -> list[str]:
    """Where the shared class list lives, in priority order: the installed layout
    (``label_map.json`` beside the members, written by ``lm3 models install``), then the training
    tree's ``splits/label_map.json``. The installed copy comes from the same Hub revisions as the
    members, so it wins when both exist."""
    return [os.path.join(models_dir, "label_map.json"),
            os.path.join(models_dir, "splits", "label_map.json")]


def _load_label_map(models_dir: str) -> list[str]:
    """Load the shared ordered class list from the first label_map.json we can find."""
    for path in _label_map_candidates(models_dir):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            classes = data.get("classes") or data.get("idx_to_name")
            if classes:
                return [str(c) for c in classes]
    raise FileNotFoundError(f"no label_map.json under {models_dir}")


def _as_rgb(image) -> np.ndarray:
    """Coerce a path / BGR ndarray / PIL image to an HWC uint8 RGB array."""
    import cv2

    if isinstance(image, str):
        bgr = cv2.imread(image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cannot read ruler crop: {image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        # Callers hand us BGR (OpenCV) arrays; convert to RGB for the model.
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # PIL image
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _resize_shortest(rgb: np.ndarray, size: int, interp) -> np.ndarray:
    import cv2

    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return rgb
    scale = size / float(min(h, w))
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=interp)


def _center_crop(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    crop = rgb[top:top + size, left:left + size]
    # Pad if the source was smaller than the crop window.
    if crop.shape[0] != size or crop.shape[1] != size:
        padded = np.zeros((size, size, 3), dtype=crop.dtype)
        padded[: crop.shape[0], : crop.shape[1]] = crop
        crop = padded
    return crop


class _Member:
    """One ONNX classifier member with family-appropriate preprocessing."""

    def __init__(self, model_dir: str, providers: Sequence, classes: list[str]) -> None:
        import onnxruntime as ort

        self.model_dir = model_dir
        self.family, self.imgsz = self._read_metadata(model_dir)
        onnx_path = os.path.join(model_dir, "exported", "model.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"exported ONNX not found: {onnx_path}")
        from leafmachine3.inference.providers import session_options

        self.session = ort.InferenceSession(onnx_path, session_options(), providers=list(providers))
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        # The class names come from the shared label map, the scores from this model: if their
        # lengths differ, every index past the change is silently misnamed (a 19-class model read
        # with an 18-class map shifted 9 classes). Refuse to load instead.
        n_out = self.session.get_outputs()[0].shape[-1]
        if isinstance(n_out, int) and n_out != len(classes):
            raise ValueError(
                f"{onnx_path} outputs {n_out} classes but the label map has {len(classes)}; "
                "install the models and label_map.json from the same lock"
            )
        self.classes = classes
        # YOLO's Classify head softmaxes internally at export; DINOv2 emits raw logits.
        self._output_is_probs = self.family == "yolo"

    @staticmethod
    def _read_metadata(model_dir: str) -> tuple[str, int]:
        meta_path = os.path.join(model_dir, "metadata.json")
        family, imgsz = None, 224
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            family = meta.get("family")
            imgsz = int(meta.get("imgsz", 224))
        if family is None:  # infer from the directory name as a fallback
            name = os.path.basename(model_dir).lower()
            family = "dinov2" if "dino" in name else "yolo"
        return family, imgsz

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        if self.family == "dinov2":
            resize_size, crop_size, interp = 256, 224, cv2.INTER_CUBIC
        else:  # yolo classify transform
            resize_size = crop_size = self.imgsz
            interp = cv2.INTER_LINEAR
        img = _resize_shortest(rgb, resize_size, interp)
        img = _center_crop(img, crop_size)
        arr = img.astype(np.float32) / 255.0
        if self.family == "dinov2":
            arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        chw = np.transpose(arr, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(chw, dtype=np.float32)

    def predict_label(self, rgb: np.ndarray) -> str:
        batch = self._preprocess(rgb)
        raw = self.session.run([self._output_name], {self._input_name: batch})[0]
        raw = np.asarray(raw, dtype=np.float64)
        probs = _renormalize(raw) if self._output_is_probs else _softmax(raw)
        idx = int(np.argmax(probs[0]))
        return self.classes[idx] if 0 <= idx < len(self.classes) else str(idx)


class RulerEnsemble:
    """3-model majority-vote ensemble over the portable ONNX ruler classifiers.

    Construction never raises on missing artifacts: it logs a warning and enters a
    degraded state where ``predict`` returns ``{"ensemble": "UNKNOWN"}``.
    """

    def __init__(
        self,
        models_dir,
        providers: Optional[Sequence] = None,
        members: Optional[Sequence[str]] = None,
        squarify_sz: int = 720,
        squarify_method: str = "tile_four",
    ) -> None:
        self.models_dir = str(models_dir)
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        # Training-time preprocessing. NOT optional: the members were fitted on
        # squarified 1440x1440 inputs, so serving anything else is a train/serve
        # mismatch. Measured on 28 real ruler crops, skipping it gave 12/28 "FP"
        # and 15 three-way member splits; with it, all 3 members agree on 28/28.
        # augment=False keeps it deterministic and pixel-identical to the LM2 oracle.
        from leafmachine3.inference.ruler_squarify import RulerSquarifier

        self._squarifier = RulerSquarifier(
            sz=int(squarify_sz), method=str(squarify_method), augment=False
        )
        self._members: dict[str, _Member] = {}
        self.classes: list[str] = []
        self.ok = False
        try:
            self._load(members)
            self.ok = True
        except Exception as exc:  # noqa: BLE001 - best-effort backend
            log.warning(
                "RulerEnsemble unavailable (%s); ruler classification -> UNKNOWN", exc
            )

    def _load(self, members: Optional[Sequence[str]]) -> None:
        self.classes = _load_label_map(self.models_dir)
        for key, dir_name in resolve_members(members):
            self._members[key] = _Member(
                os.path.join(self.models_dir, dir_name), self.providers, self.classes
            )

    @staticmethod
    def _vote(x224: str, n224: str, dinov2mlp: str) -> str:
        """Majority of the 3 votes; a three-way tie falls back to x224 (best macro-F1)."""
        cls, votes = Counter((x224, n224, dinov2mlp)).most_common(1)[0]
        return cls if votes >= 2 else x224

    def predict(self, image) -> dict:
        """Classify one ruler crop (path | BGR ndarray | PIL) -> the ensemble dict."""
        if not self.ok:
            return {"ensemble": "UNKNOWN"}
        try:
            rgb = _as_rgb(self._squarify(image))
            x = self._members["x224"].predict_label(rgb)
            n = self._members["n224"].predict_label(rgb)
            d = self._members["dinov2mlp"].predict_label(rgb)
        except Exception as exc:  # noqa: BLE001 - never crash the pipeline on one crop
            log.warning("ruler ensemble predict failed (%s) -> UNKNOWN", exc)
            return {"ensemble": "UNKNOWN"}
        return {"ensemble": self._vote(x, n, d), "x224": x, "n224": n, "dinov2mlp": d}

    def squarify_tile(self, image):
        """The four-tile collage the classifier saw, BGR ndarray (for the pre-made QC tile)."""
        return self._squarify(image)

    def _squarify(self, image):
        """Apply the mandatory training-time squarify -> BGR ndarray."""
        import os

        import cv2

        src = image
        if not isinstance(src, (str, os.PathLike, np.ndarray)):
            # PIL -> BGR ndarray (transform() only accepts a path or an ndarray)
            src = cv2.cvtColor(np.asarray(src.convert("RGB")), cv2.COLOR_RGB2BGR)
        return self._squarifier.transform(src)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"RulerEnsemble(ok={self.ok}, members={list(self._members)}, "
                f"squarify={self._squarifier.method}@{self._squarifier.sz}, "
                f"dir={self.models_dir})")
