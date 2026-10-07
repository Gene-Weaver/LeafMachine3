"""Real specimen-segmentation backends: one per input workflow, chosen by model KEY.

The settings name the model (``modules.specimen_segmenter.model.key``); the factory resolves it to
a :class:`~leafmachine3.inference.specimen_models.SpecimenModelSpec` and calls
:func:`build_specimen_segmenter`, which picks the backend from ``spec.workflow``. Nothing here
inspects a file to decide what it is -- a file that cannot be the named model raises
:class:`SpecimenModelMismatch` naming both, and that is the only use made of its tensors.

* :class:`BinaryOnnxSpecimenSegmenter` -- single-logit ONNX graphs (UNet++, BiRefNet). The spec's
  workflow decides the image preparation: ``letterbox_imagenet`` or ``stretch_imagenet``.
* :class:`YoloSpecimenSegmenter` -- a YOLO26-seg end2end export run through
  ``inference/ultra_replacements.py``; the foreground is the union of the instance masks.

All of them finish identically: the informed HSV **paperclean** follow-up step, then a
:class:`~leafmachine3.core.records.SpecimenMaskResult` whose ``model_name`` is the model KEY.
Dependency-light: onnxruntime + numpy + cv2 only (no torch at runtime). See ``specimen_models.py``
for each workflow's training reference.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import cv2
import numpy as np

from leafmachine3.core.paper_removal import paperclean
from leafmachine3.core.records import SpecimenMaskResult
from leafmachine3.inference.specimen_models import (
    SPECIMEN_MODELS, WORKFLOW_LETTERBOX_IMAGENET, WORKFLOW_STRETCH_IMAGENET, WORKFLOW_ULTRALYTICS_SEG,
    DEFAULT_SPECIMEN_MODEL_KEY, SpecimenModelSpec,
)

log = logging.getLogger("leafmachine3.inference.specimen_segmenter")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # RGB
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)    # RGB


class SpecimenModelMismatch(ValueError):
    """The file at ``model.path`` cannot be the model ``model.key`` names."""


# --------------------------------------------------------------------------- #
# workflow: letterbox_imagenet (UNet++)
# --------------------------------------------------------------------------- #
def _letterbox(img_rgb: np.ndarray, imgsz: int):
    """Aspect-preserving INTER_AREA resize onto a centered white ``imgsz`` square.

    Byte-identical to the UNet++ trainer's ``letterbox_pair`` (image half) and ``seg_infer.letterbox``.
    """
    H, W = img_rgb.shape[:2]
    scale = imgsz / max(H, W)
    h2, w2 = max(1, round(H * scale)), max(1, round(W * scale))
    r = cv2.resize(img_rgb, (w2, h2), interpolation=cv2.INTER_AREA)
    top, left = (imgsz - h2) // 2, (imgsz - w2) // 2
    canvas = np.full((imgsz, imgsz, 3), 255, np.uint8)
    canvas[top:top + h2, left:left + w2] = r
    return canvas, (top, left, h2, w2)


def _unletterbox(mask_sq: np.ndarray, meta, H: int, W: int) -> np.ndarray:
    top, left, h2, w2 = meta
    crop = mask_sq[top:top + h2, left:left + w2].astype(np.uint8)
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_NEAREST)


def _normalize_imagenet(rgb_u8: np.ndarray) -> np.ndarray:
    x = (rgb_u8.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def prep_letterbox_imagenet(img_bgr: np.ndarray, imgsz: int):
    """``(tensor, meta)`` for the UNet++ workflow."""
    sq, meta = _letterbox(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), imgsz)
    return _normalize_imagenet(sq), meta


def post_letterbox_imagenet(prob_sq: np.ndarray, meta, H: int, W: int, thr: float) -> np.ndarray:
    """Threshold at the square, drop the pad, nearest-resize to ``(H, W)`` -> {0,1} uint8."""
    return _unletterbox((prob_sq > thr).astype(np.uint8), meta, H, W)


# --------------------------------------------------------------------------- #
# workflow: stretch_imagenet (BiRefNet)
# --------------------------------------------------------------------------- #
def prep_stretch_imagenet(img_bgr: np.ndarray, imgsz: int):
    """``(tensor, None)`` for the BiRefNet workflow: BGR INTER_LINEAR stretch, then RGB.

    Same order and filter as BiRefNet's training loader (``utils.path_to_image``: ``cv2.resize``
    on the BGR image, then ``cvtColor`` to RGB), so the uint8 pixels the model sees are identical.
    """
    r = cv2.resize(img_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    return _normalize_imagenet(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)), None


def post_stretch_imagenet(prob_sq: np.ndarray, H: int, W: int, thr: float) -> np.ndarray:
    """Bilinear-resize the probabilities to ``(H, W)``, then threshold -> {0,1} uint8."""
    prob = cv2.resize(prob_sq.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return (prob > thr).astype(np.uint8)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z.astype(np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- #
# shared tail
# --------------------------------------------------------------------------- #
def _finish(img_bgr: np.ndarray, mask: np.ndarray, do_paperclean: bool, model_name: str) -> SpecimenMaskResult:
    H, W = img_bgr.shape[:2]
    if do_paperclean:
        final, removed, centers = paperclean(img_bgr, mask)
    else:
        final, removed, centers = mask, np.zeros_like(mask), []
    return _pack(final, removed, centers, H, W, model_name)


def _pack(final, removed, centers, H, W, model_name) -> SpecimenMaskResult:
    final_png = cv2.imencode(".png", (final > 0).astype(np.uint8) * 255)[1].tobytes()
    removed_png = cv2.imencode(".png", (removed > 0).astype(np.uint8) * 255)[1].tobytes()
    area_frac = float((final > 0).sum()) / float(final.size) if final.size else 0.0
    return SpecimenMaskResult(
        final_png=final_png, removed_png=removed_png,
        frame_width=int(W), frame_height=int(H),
        centers=[(int(x), int(y)) for (x, y) in centers],
        area_frac=area_frac, model_name=str(model_name),
    )


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class BinaryOnnxSpecimenSegmenter:
    """Single-logit whole-sheet segmenter (UNet++ / BiRefNet ONNX) + paperclean."""

    def __init__(self, model_path, spec: SpecimenModelSpec, providers: Optional[Sequence] = None,
                 conf: float = 0.5, paperclean: bool = True, session=None) -> None:
        if spec.workflow not in (WORKFLOW_LETTERBOX_IMAGENET, WORKFLOW_STRETCH_IMAGENET):
            raise ValueError(f"{spec.key} uses workflow {spec.workflow!r}, not a single-logit workflow")
        self.model_path = str(model_path)
        self.spec = spec
        self.thr = float(conf)
        self.paperclean = bool(paperclean)
        self.model_name = spec.key
        if session is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"specimen segmenter model artifact not found: {self.model_path}")
            import onnxruntime as ort
            session = ort.InferenceSession(
                self.model_path, providers=list(providers) if providers else ["CPUExecutionProvider"])
        self.session = session
        self._check_contract()
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        log.info("specimen segmenter loaded: key=%s (%s, workflow=%s, imgsz=%d, thr=%.2f, paperclean=%s, EP=%s) %s",
                 spec.key, spec.label, spec.workflow, spec.imgsz, self.thr, self.paperclean,
                 self.session.get_providers()[0], os.path.basename(self.model_path))

    def _check_contract(self) -> None:
        """Raise :class:`SpecimenModelMismatch` if the file cannot be this single-logit model."""
        ins, outs = self.session.get_inputs(), self.session.get_outputs()
        name = os.path.basename(self.model_path)
        problems = []
        if len(ins) != 1 or len(outs) != 1:
            problems.append(f"{len(ins)} input(s) / {len(outs)} output(s), expected 1 / 1")
        else:
            shape = list(ins[0].shape)
            if len(shape) != 4 or shape[1] != 3:
                problems.append(f"input shape {shape}, expected [1, 3, {self.spec.imgsz}, {self.spec.imgsz}]")
            else:
                for d in shape[2:]:
                    if isinstance(d, int) and d != self.spec.imgsz:
                        problems.append(f"input size {shape[2:]} but {self.spec.key} is trained at {self.spec.imgsz}")
                        break
        if problems:
            raise SpecimenModelMismatch(
                f"modules.specimen_segmenter.model.key={self.spec.key!r} ({self.spec.label}) does not match "
                f"{name}: {'; '.join(problems)}. Point model.path at the {self.spec.key} export, or set model.key "
                f"to the model that file is (known: {', '.join(SPECIMEN_MODELS)}).")

    def mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Native-resolution {0,1} uint8 foreground mask, BEFORE paperclean."""
        img_bgr = np.ascontiguousarray(img_bgr)
        H, W = img_bgr.shape[:2]
        if self.spec.workflow == WORKFLOW_LETTERBOX_IMAGENET:
            x, meta = prep_letterbox_imagenet(img_bgr, self.spec.imgsz)
            prob = _sigmoid(self.session.run([self._output_name], {self._input_name: x})[0][0, 0])
            return post_letterbox_imagenet(prob, meta, H, W, self.thr)
        x, _ = prep_stretch_imagenet(img_bgr, self.spec.imgsz)
        prob = _sigmoid(self.session.run([self._output_name], {self._input_name: x})[0][0, 0])
        return post_stretch_imagenet(prob, H, W, self.thr)

    def predict(self, img_bgr: np.ndarray) -> SpecimenMaskResult:
        img_bgr = np.ascontiguousarray(img_bgr)
        return _finish(img_bgr, self.mask(img_bgr), self.paperclean, self.model_name)


class YoloSpecimenSegmenter:
    """YOLO26-seg end2end export; foreground = union of instance masks at ``conf`` + paperclean."""

    def __init__(self, model_path, spec: SpecimenModelSpec, providers: Optional[Sequence] = None,
                 conf: float = 0.25, iou: float = 0.5, max_det: int = 300, paperclean: bool = True,
                 model=None) -> None:
        if spec.workflow != WORKFLOW_ULTRALYTICS_SEG:
            raise ValueError(f"{spec.key} uses workflow {spec.workflow!r}, not {WORKFLOW_ULTRALYTICS_SEG!r}")
        self.model_path = str(model_path)
        self.spec = spec
        self.conf, self.iou, self.max_det = float(conf), float(iou), int(max_det)
        self.paperclean = bool(paperclean)
        self.model_name = spec.key
        if model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"specimen segmenter model artifact not found: {self.model_path}")
            from leafmachine3.inference import ultra_replacements as ultra_rep
            try:
                model = ultra_rep.YOLO(self.model_path, task="segment", providers=providers)
            except ValueError as exc:   # not an end2end segment export
                raise SpecimenModelMismatch(
                    f"modules.specimen_segmenter.model.key={spec.key!r} ({spec.label}) expects a YOLO26-seg end2end "
                    f"export, but {os.path.basename(self.model_path)} is not one ({exc}). Point model.path at the "
                    f"{spec.key} export, or set model.key to the model that file is (known: {', '.join(SPECIMEN_MODELS)}).") from exc
        self.model = model
        log.info("specimen segmenter loaded: key=%s (%s, workflow=%s, imgsz=%d, conf=%.2f, max_det=%d, paperclean=%s) %s",
                 spec.key, spec.label, spec.workflow, spec.imgsz, self.conf, self.max_det, self.paperclean,
                 os.path.basename(self.model_path))

    def mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Union of the instance masks at native resolution, {0,1} uint8, BEFORE paperclean."""
        img_bgr = np.ascontiguousarray(img_bgr)
        H, W = img_bgr.shape[:2]
        res = self.model.predict(img_bgr, conf=self.conf, iou=self.iou, imgsz=self.spec.imgsz,
                                 max_det=self.max_det, retina_masks=True, verbose=False)[0]
        union = np.zeros((H, W), np.uint8)
        if res.masks is None or len(res.masks) == 0:
            return union
        for m in np.asarray(res.masks.data):
            m = (m > 0).astype(np.uint8)
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            union |= m
        return union

    def predict(self, img_bgr: np.ndarray) -> SpecimenMaskResult:
        img_bgr = np.ascontiguousarray(img_bgr)
        return _finish(img_bgr, self.mask(img_bgr), self.paperclean, self.model_name)


def build_specimen_segmenter(spec: SpecimenModelSpec, model_path, providers=None, *, conf: float = 0.5,
                             yolo: Optional[dict] = None, paperclean: bool = True):
    """Construct the backend for ``spec`` (the name decides; the file only has to match it)."""
    if spec.workflow in (WORKFLOW_LETTERBOX_IMAGENET, WORKFLOW_STRETCH_IMAGENET):
        return BinaryOnnxSpecimenSegmenter(model_path, spec, providers=providers, conf=conf, paperclean=paperclean)
    if spec.workflow == WORKFLOW_ULTRALYTICS_SEG:
        y = dict(yolo or {})
        return YoloSpecimenSegmenter(model_path, spec, providers=providers, conf=float(y.get("conf", 0.25)),
                                     iou=float(y.get("iou", 0.5)), max_det=int(y.get("max_det", 300)),
                                     paperclean=paperclean)
    raise ValueError(f"{spec.key}: no backend for workflow {spec.workflow!r}")


class OnnxSpecimenSegmenter(BinaryOnnxSpecimenSegmenter):
    """Back-compat constructor (pre-registry signature): always the UNet++ letterbox workflow."""

    def __init__(self, model_path, providers=None, imgsz: int = 1024, conf: float = 0.5,
                 paperclean: bool = True, model_name: str = "") -> None:
        super().__init__(model_path, SPECIMEN_MODELS[DEFAULT_SPECIMEN_MODEL_KEY], providers=providers,
                         conf=conf, paperclean=paperclean)

    def _unet_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        return self.mask(img_bgr)
