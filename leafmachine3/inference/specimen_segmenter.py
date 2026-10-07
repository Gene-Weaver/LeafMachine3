"""Real specimen-segmentation backend — an exported UNet++ (efficientnet-b7) via onnxruntime.

``OnnxSpecimenSegmenter`` runs the whole-sheet plant-vs-background segmenter: letterbox the
working image to the model's 1024 square (white pad), normalize with ImageNet stats (RGB), run
the ONNX graph (logits), sigmoid + threshold, un-letterbox back to the native working frame, then
apply the informed HSV **paperclean** follow-up step. ``predict`` returns a
:class:`~leafmachine3.core.records.SpecimenMaskResult` carrying the final + removed masks
(PNG-encoded) and the paper-sampling box centers for the QC overlay.

Dependency-light like the ruler ensemble: only ``onnxruntime`` + ``numpy`` + ``cv2`` (no torch /
segmentation_models_pytorch at runtime). The preprocessing here matches the training project's
``common/seg_infer.py`` byte-for-byte so the ONNX output equals the source checkpoint.
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from leafmachine3.core.paper_removal import paperclean
from leafmachine3.core.records import SpecimenMaskResult

log = logging.getLogger("leafmachine3.inference.specimen_segmenter")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # RGB
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)    # RGB


def _letterbox(img_rgb: np.ndarray, imgsz: int):
    """Aspect-preserving resize onto a white ``imgsz`` square (matches seg_infer.letterbox)."""
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


class OnnxSpecimenSegmenter:
    """Whole-specimen binary segmenter (UNet++ ONNX) + paperclean follow-up."""

    def __init__(
        self,
        model_path,
        providers=None,
        imgsz: int = 1024,
        conf: float = 0.5,
        paperclean: bool = True,
        model_name: str = "",
    ) -> None:
        import onnxruntime as ort

        self.model_path = str(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"specimen segmenter model artifact not found: {self.model_path}")
        self.imgsz = int(imgsz)
        self.thr = float(conf)
        self.paperclean = bool(paperclean)
        self.model_name = model_name or os.path.splitext(os.path.basename(self.model_path))[0]
        self.session = ort.InferenceSession(
            self.model_path, providers=list(providers) if providers else ["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self._input_name = inp.name
        self._output_name = self.session.get_outputs()[0].name
        # The exported graph has a STATIC 1x3x<imgsz>x<imgsz> input; fail fast at LOAD if the
        # configured imgsz disagrees, rather than deep in session.run on the first specimen.
        graph_dim = inp.shape[-1] if inp.shape and isinstance(inp.shape[-1], int) else None
        if graph_dim is not None and graph_dim != self.imgsz:
            raise ValueError(
                f"specimen_segmenter imgsz={self.imgsz} does not match the model input size "
                f"{graph_dim} ({os.path.basename(self.model_path)}); set "
                f"modules.specimen_segmenter.imgsz to {graph_dim} (or re-export at {self.imgsz})."
            )
        log.info(
            "OnnxSpecimenSegmenter loaded: %s (imgsz=%d, thr=%.2f, paperclean=%s, EP=%s)",
            os.path.basename(self.model_path), self.imgsz, self.thr, self.paperclean,
            self.session.get_providers()[0],
        )

    def _unet_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Native-resolution {0,1} uint8 UNet foreground mask for a BGR image."""
        H, W = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        sq, meta = _letterbox(rgb, self.imgsz)
        x = (sq.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)
        logit = self.session.run([self._output_name], {self._input_name: x})[0]
        # overflow-safe sigmoid: clip before exp (saturates far below +-30, so the mask is identical
        # to the exact sigmoid but no RuntimeWarning is emitted on confidently-background logits).
        z = np.clip(logit[0, 0].astype(np.float32), -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-z))
        return _unletterbox((prob > self.thr).astype(np.uint8), meta, H, W)

    def predict(self, img_bgr: np.ndarray) -> SpecimenMaskResult:
        """Segment one working image (BGR ndarray) -> :class:`SpecimenMaskResult`."""
        img_bgr = np.ascontiguousarray(img_bgr)
        H, W = img_bgr.shape[:2]
        unet = self._unet_mask(img_bgr)
        if self.paperclean:
            final, removed, centers = paperclean(img_bgr, unet)
        else:
            final, removed, centers = unet, np.zeros_like(unet), []
        return _pack(final, removed, centers, H, W, self.model_name)


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
