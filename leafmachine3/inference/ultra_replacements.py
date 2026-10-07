# Ports code from Ultralytics 8.4.107 (https://github.com/ultralytics/ultralytics),
# Copyright Ultralytics Inc., AGPL-3.0 License (https://ultralytics.com/license).
# This file remains under the AGPL-3.0; see NOTICE.
"""Pure-onnxruntime replacement for the Ultralytics ``YOLO(...).predict`` path LM3 uses at inference.

Imported as ``ultra_rep``. Every exported LM3 YOLO26 model (plant / archival detectors, leaf
segmenter, landmark pose) is an **end2end** ONNX graph: the network already emits its final, score-
sorted detections (no NMS). What Ultralytics still did for us on top of ``onnxruntime`` was

1. load the image (``cv2.imdecode`` of the raw bytes),
2. letterbox it (``LetterBox(auto=True)`` because the graph has dynamic H/W: long side -> ``imgsz``,
   short side padded up to a multiple of the stride, gray 114, centered),
3. BGR->RGB, HWC->CHW, uint8 -> float32 / 255,
4. ``session.run``,
5. confidence filter + ``max_det`` slice (``non_max_suppression`` end2end branch),
6. undo the letterbox on boxes / keypoints (``scale_boxes`` / ``scale_coords`` + clip),
7. for segmentation: decode ``sigmoid(coeffs @ proto) > 0.5`` at native resolution, crop to the box,
   drop detections whose mask is empty (``process_mask_native`` with ``retina_masks=True``).

Each step below is a line-for-line port of the Ultralytics 8.4.107 code it replaces (named in the
docstrings) with ``numpy`` / ``cv2`` standing in for ``torch``. Rounding conventions (``round(x - 0.1)``
pads, ``np.mod`` stride padding, float32 arithmetic) are preserved so boxes and keypoints match the
Ultralytics path bit-for-bit on the same execution provider; masks match up to last-ulp differences
in bilinear upsampling (see ``_bilinear_resize``).

**Division convention.** Ultralytics runs its three scalar divisions (``im /= 255``, ``boxes /= gain``,
``coords /= gain``) as torch tensor ops. On a CUDA device torch's ``div`` kernel turns a CPU-scalar
divisor into ``x * (1 / d)`` (float32 reciprocal, then multiply), which differs from true division by
one ulp for 126 of the 256 pixel values; on a CPU device torch divides exactly. LM3's production
reference (every evaluation and every run to date) is the CUDA path, so this module uses the
reciprocal-multiply form everywhere (``_scale``). Measured on the 22 example sheets: with true
division the plant detector's boxes drift by up to 0.4 px and confidences by up to 0.02 and one
near-threshold detection flips; with the reciprocal form the outputs are identical to the
Ultralytics-on-CUDA baseline. Do not "simplify" these back to ``/``.

The return objects mimic the small slice of ``ultralytics.engine.results.Results`` the LM3 wrappers
read -- ``boxes.xyxy / conf / cls``, ``masks.data``, ``keypoints.data``, ``orig_shape``, ``names`` --
but hold plain ``numpy`` arrays (no ``.cpu()``).

No torch. No ultralytics. No runtime ``pip install`` side effects. No ``~/.config`` writes.
"""
from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger("leafmachine3.inference.ultra_rep")

PAD_VALUE = 114          # LetterBox(padding_value=114)
DEFAULT_IMGSZ = 640      # ultralytics cfg/default.yaml ``imgsz: 640`` (used when a caller passes none)
DEFAULT_MAX_DET = 300    # ultralytics cfg/default.yaml ``max_det: 300``

__all__ = ["YOLO", "Result", "Boxes", "Masks", "Keypoints", "letterbox", "imread"]


# --------------------------------------------------------------------------------------------------
# Result containers (the subset of ultralytics.engine.results that LM3 reads)
# --------------------------------------------------------------------------------------------------
@dataclass
class Boxes:
    """``Results.boxes``: ``xyxy`` (n,4) float32 in ORIGINAL image pixels, ``conf`` (n,), ``cls`` (n,)."""
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


@dataclass
class Masks:
    """``Results.masks``: ``data`` (n, H, W) uint8 {0,1} raster masks at ORIGINAL resolution."""
    data: np.ndarray

    def __len__(self) -> int:
        return int(self.data.shape[0])


@dataclass
class Keypoints:
    """``Results.keypoints``: ``data`` (n, K, 3) float32 = x, y (original pixels), conf."""
    data: np.ndarray

    def __len__(self) -> int:
        return int(self.data.shape[0])


@dataclass
class Result:
    """One image's predictions. ``orig_shape`` is ``(h, w)`` of the ORIGINAL (un-letterboxed) image."""
    orig_shape: tuple[int, int]
    names: dict[int, str]
    boxes: Boxes
    masks: Optional[Masks] = None
    keypoints: Optional[Keypoints] = None
    speed: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.boxes)


# --------------------------------------------------------------------------------------------------
# Image I/O and preprocessing
# --------------------------------------------------------------------------------------------------
def imread(path) -> Optional[np.ndarray]:
    """``ultralytics.utils.patches.imread``: decode the raw bytes with ``cv2.IMREAD_COLOR`` (BGR uint8).

    Reading the bytes with ``np.fromfile`` (rather than ``cv2.imread``) is what makes non-ASCII paths
    work on Windows; the decode itself is identical.
    """
    try:
        buf = np.fromfile(str(path), np.uint8)
    except (FileNotFoundError, OSError):
        return None
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if im is None:
        return None
    if im.ndim == 2:                                   # grayscale -> 3-channel so the net gets RGB
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im


def check_imgsz(imgsz, stride: int = 32) -> tuple[int, int]:
    """``ultralytics.utils.checks.check_imgsz(imgsz, stride, min_dim=2)``: (h, w), each a multiple of stride."""
    if imgsz is None:
        imgsz = DEFAULT_IMGSZ
    if isinstance(imgsz, (int, float, np.integer)):
        sizes = [int(imgsz)]
    else:
        sizes = [int(x) for x in imgsz]
    if len(sizes) > 2:
        raise ValueError(f"imgsz={imgsz} must be an int or a (h, w) pair")
    sz = [max(int(np.ceil(x / stride) * stride), 0) for x in sizes]
    if sz != sizes:
        log.warning("imgsz=%s must be multiple of max stride %d, updating to %s", sizes, stride, sz)
    return (sz[0], sz[0]) if len(sz) == 1 else (sz[0], sz[1])


def letterbox(img: np.ndarray, new_shape: tuple[int, int], stride: int = 32, auto: bool = True,
              center: bool = True) -> np.ndarray:
    """``ultralytics.data.augment.LetterBox.__call__`` for the predict path.

    ``auto=True`` is what the Ultralytics predictor uses for a dynamic-shape ONNX export
    (``rect=True`` and ``model.dynamic``): the image is scaled so its long side is ``new_shape`` and
    the short side is padded only up to the next multiple of ``stride``. ``scaleup=True``,
    ``scale_fill=False``, ``center=True``, ``padding_value=114``, ``cv2.INTER_LINEAR`` are the defaults.
    """
    shape = img.shape[:2]                                  # (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = round(shape[1] * r), round(shape[0] * r)   # (w, h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    if center:
        dw /= 2
        dh /= 2
    top, bottom = (round(dh - 0.1) if center else 0), round(dh + 0.1)
    left, right = (round(dw - 0.1) if center else 0), round(dw + 0.1)
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right),
                              cv2.BORDER_CONSTANT, value=(PAD_VALUE, PAD_VALUE, PAD_VALUE))


def _scale(x: np.ndarray, divisor: float) -> np.ndarray:
    """In-place ``x /= divisor`` the way torch's CUDA kernel does it: ``x *= float32(1) / float32(d)``.

    See the module docstring ("Division convention"). ``x`` must be float32.
    """
    x *= np.float32(1.0) / np.float32(divisor)
    return x


def preprocess(img_bgr: np.ndarray, imgsz: tuple[int, int], stride: int, auto: bool) -> np.ndarray:
    """``BasePredictor.preprocess``: letterbox -> BGR->RGB -> BCHW -> float32 -> ``im /= 255`` (CUDA form)."""
    im = letterbox(img_bgr, imgsz, stride=stride, auto=auto)[None]   # (1, H, W, 3)
    im = im[..., ::-1].transpose((0, 3, 1, 2))                       # BGR->RGB, BHWC->BCHW
    im = np.ascontiguousarray(im).astype(np.float32)
    return _scale(im, 255)


# --------------------------------------------------------------------------------------------------
# Postprocessing (ultralytics.utils.ops / utils.nms, end2end branch only)
# --------------------------------------------------------------------------------------------------
def _letterbox_gain_pad(img1_shape, img0_shape) -> tuple[float, int, int]:
    """Shared by ``scale_boxes`` / ``scale_coords``: gain and the (x, y) pad the letterbox added."""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1)
    pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1)
    return gain, pad_x, pad_y


def scale_boxes(img1_shape, boxes: np.ndarray, img0_shape) -> np.ndarray:
    """``ops.scale_boxes(img1_shape, boxes, img0_shape)`` (xyxy, padding=True) + ``clip_boxes``. In place."""
    gain, pad_x, pad_y = _letterbox_gain_pad(img1_shape, img0_shape)
    boxes[..., 0] -= pad_x
    boxes[..., 1] -= pad_y
    boxes[..., 2] -= pad_x
    boxes[..., 3] -= pad_y
    _scale(boxes[..., :4], gain)
    h, w = img0_shape[:2]
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)
    return boxes


def scale_coords(img1_shape, coords: np.ndarray, img0_shape) -> np.ndarray:
    """``ops.scale_coords`` (padding=True, normalize=False) + ``clip_coords``. ``coords`` is (n, K, >=2). In place."""
    gain, pad_x, pad_y = _letterbox_gain_pad(img1_shape, img0_shape)
    coords[..., 0] -= pad_x
    coords[..., 1] -= pad_y
    _scale(coords[..., 0], gain)
    _scale(coords[..., 1], gain)
    h, w = img0_shape[:2]
    coords[..., 0] = coords[..., 0].clip(0, w)
    coords[..., 1] = coords[..., 1].clip(0, h)
    return coords


def end2end_filter(pred: np.ndarray, conf: float, max_det: int) -> np.ndarray:
    """``nms.non_max_suppression`` end2end branch: ``pred[pred[:, 4] > conf][:max_det]`` (order kept)."""
    keep = pred[:, 4] > conf
    return pred[keep][:max_det]


def _bilinear_resize(m: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """``F.interpolate(mode="bilinear", align_corners=False)`` for one (h, w) float32 map.

    ``cv2.resize(INTER_LINEAR)`` uses the same half-pixel-center sampling and the same two-tap
    weights as torch's bilinear with ``align_corners=False``; results differ only in float32
    summation order (last ulp), which matters only for pixels whose logit is within ~1e-6 of 0.
    """
    h, w = shape
    if m.shape[0] == h and m.shape[1] == w:
        return m
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)


def process_mask_native(proto: np.ndarray, coeffs: np.ndarray, boxes_xyxy: np.ndarray,
                        shape: tuple[int, int]) -> np.ndarray:
    """``ops.process_mask_native(protos, masks_in, bboxes, shape)`` -> (n, H, W) uint8 {0,1}.

    ``scale_masks`` first strips the letterbox pad from the prototype-resolution logits, then
    upsamples to the original ``shape``; ``gt_(0.0)`` is the sigmoid-0.5 threshold; ``crop_mask`` zeroes
    everything outside each box (``x1 <= col < x2``, ``y1 <= row < y2``).
    """
    c, mh, mw = proto.shape
    h, w = shape
    n = coeffs.shape[0]
    if n == 0:
        return np.zeros((0, h, w), np.uint8)
    logits = (coeffs.astype(np.float32) @ proto.reshape(c, -1).astype(np.float32)).reshape(n, mh, mw)

    # scale_masks: pad geometry of the PROTO relative to the original image
    gain = min(mh / h, mw / w)
    pad_w, pad_h = (mw - round(w * gain)) / 2, (mh - round(h * gain)) / 2
    top, left = round(pad_h - 0.1), round(pad_w - 0.1)
    bottom, right = mh - round(pad_h + 0.1), mw - round(pad_w + 0.1)

    cols = np.arange(w, dtype=np.float32)[None, :]
    rows = np.arange(h, dtype=np.float32)[:, None]
    out = np.empty((n, h, w), np.uint8)
    for i in range(n):
        up = _bilinear_resize(np.ascontiguousarray(logits[i, top:bottom, left:right]), (h, w))
        x1, y1, x2, y2 = boxes_xyxy[i, :4]
        keep = ((cols >= x1) & (cols < x2)) & ((rows >= y1) & (rows < y2))   # crop_mask
        out[i] = ((up > 0.0) & keep).astype(np.uint8)
    return out


# --------------------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------------------
class YOLO:
    """Drop-in for ``ultralytics.YOLO(path, task=...)`` restricted to LM3's exported end2end ONNX graphs.

    Parameters
    ----------
    model_path : str
        An ``.onnx`` export produced by ``ultralytics`` with ``nms=False`` on a YOLO26 (end2end) model.
    task : {"detect", "segment", "pose"} or None
        Checked against the export's metadata ``task``; ``None`` takes the metadata value.
    providers : sequence or None
        onnxruntime execution providers (e.g. ``Device.ort_providers()``). Defaults to CPU.
    """

    def __init__(self, model_path, task: Optional[str] = None, providers: Optional[Sequence] = None) -> None:
        import onnxruntime as ort   # deferred so importing this module never needs ORT

        self.model_path = str(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model artifact not found: {self.model_path}")
        if not self.model_path.lower().endswith(".onnx"):
            raise ValueError(
                f"ultra_rep.YOLO handles .onnx exports only, got {os.path.basename(self.model_path)}; "
                "re-export with `format=onnx` (dynamic=True, nms=False)."
            )
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        from leafmachine3.inference.providers import session_options

        self.session = ort.InferenceSession(self.model_path, session_options(), providers=self.providers)
        self.bound_provider = self.session.get_providers()[0]
        requested = self.providers[0][0] if isinstance(self.providers[0], (tuple, list)) else self.providers[0]
        if requested != self.bound_provider:
            log.warning("ultra_rep.YOLO: requested %s but onnxruntime bound %s for %s",
                        requested, self.bound_provider, os.path.basename(self.model_path))

        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.fp16 = "float16" in inputs[0].type
        self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)

        meta = self._parse_metadata(dict(self.session.get_modelmeta().custom_metadata_map or {}))
        self.metadata = meta
        self.task = str(meta.get("task") or task or "detect")
        if task is not None and task != self.task:
            raise ValueError(f"{os.path.basename(self.model_path)} is a {self.task} export, not {task}")
        if not meta.get("end2end", False):
            raise ValueError(
                f"{os.path.basename(self.model_path)} is not an end2end export (metadata end2end=False); "
                "ultra_rep has no NMS. Re-export the YOLO26 model with nms=False, or keep ultralytics."
            )
        self.stride = int(meta.get("stride", 32))
        self.names: dict[int, str] = {int(k): str(v) for k, v in (meta.get("names") or {}).items()}
        self.kpt_shape = tuple(meta.get("kpt_shape") or ()) if self.task == "pose" else ()
        if self.task == "pose" and len(self.kpt_shape) != 2:
            raise ValueError(f"pose export {os.path.basename(self.model_path)} has no kpt_shape metadata")
        # auto (minimum-rectangle) letterbox iff the graph is dynamic, as the ultralytics predictor does
        self.auto_letterbox = bool(meta.get("dynamic", self.dynamic))
        log.info("ultra_rep.YOLO loaded %s task=%s EP=%s dynamic=%s names=%d",
                 os.path.basename(self.model_path), self.task, self.bound_provider, self.dynamic, len(self.names))

    @staticmethod
    def _parse_metadata(meta: dict) -> dict:
        """``BaseBackend.apply_metadata``: literal_eval the stringified fields, derive end2end / dynamic."""
        out: dict[str, Any] = {}
        for k, v in meta.items():
            if k in {"stride", "batch", "channels"}:
                out[k] = int(v)
            elif k in {"imgsz", "names", "kpt_shape", "kpt_names", "args", "end2end"} and isinstance(v, str):
                try:
                    out[k] = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    out[k] = v
            else:
                out[k] = v
        args = out.get("args") if isinstance(out.get("args"), dict) else {}
        out["end2end"] = bool(out.get("end2end", False)) or bool(args.get("nms", False))
        if "dynamic" in args:
            out["dynamic"] = bool(args["dynamic"])
        return out

    # ---- inference ---------------------------------------------------------------------------------
    @staticmethod
    def _load_source(source) -> np.ndarray:
        if isinstance(source, np.ndarray):
            img = source
        elif isinstance(source, (str, os.PathLike)):
            img = imread(source)
            if img is None:
                raise FileNotFoundError(f"cannot read image: {source}")
        else:   # PIL.Image or anything array-like: ultralytics treats PIL as RGB
            arr = np.asarray(source)
            img = arr[..., ::-1] if arr.ndim == 3 and arr.shape[2] == 3 else arr
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return np.ascontiguousarray(img)

    def predict(self, source, conf: float = 0.25, iou: float = 0.7, imgsz=None, max_det: Optional[int] = None,
                retina_masks: bool = False, classes=None, device=None, verbose: bool = False,
                **_ignored) -> list[Result]:
        """Run one image. Returns ``[Result]`` (a 1-list, like ultralytics).

        ``iou`` / ``device`` / ``verbose`` / ``half`` are accepted for signature compatibility; ``iou``
        has no effect because end2end graphs do no NMS (ultralytics ignores it on this path too).
        """
        img = self._load_source(source)
        orig_shape = (int(img.shape[0]), int(img.shape[1]))
        in_hw = check_imgsz(imgsz, self.stride)
        im = preprocess(img, in_hw, self.stride, self.auto_letterbox)
        if self.fp16:
            im = im.astype(np.float16)
        outs = self.session.run(self.output_names, {self.input_name: im})
        max_det = DEFAULT_MAX_DET if max_det is None else int(max_det)
        img1_shape = im.shape[2:]

        pred = np.asarray(outs[0][0], dtype=np.float32).copy()        # (300, 6 | 38 | 99)
        pred = end2end_filter(pred, float(conf), max_det)
        if classes is not None:
            pred = pred[np.isin(pred[:, 5], np.asarray(classes, dtype=np.float32))]

        masks = keypoints = None
        if self.task == "segment":
            proto = np.asarray(outs[1][0], dtype=np.float32)            # (32, mh, mw)
            if pred.shape[0] == 0:
                masks = None
            elif retina_masks:
                scale_boxes(img1_shape, pred[:, :4], orig_shape)
                data = process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_shape)
                masks = Masks(data)
            else:
                raise NotImplementedError(
                    "ultra_rep supports retina_masks=True only (LM3 always sets it); the low-res "
                    "process_mask(upsample=True) path was never used by LM3."
                )
            if masks is not None:                                        # drop empty-mask detections
                keep = masks.data.reshape(masks.data.shape[0], -1).max(axis=1) > 0
                if not keep.all():
                    pred, masks = pred[keep], Masks(masks.data[keep])
        else:
            scale_boxes(img1_shape, pred[:, :4], orig_shape)
            if self.task == "pose":
                k, d = self.kpt_shape
                kp = pred[:, 6:].reshape(pred.shape[0], k, d).copy()
                scale_coords(img1_shape, kp, orig_shape)
                keypoints = Keypoints(kp)

        boxes = Boxes(xyxy=pred[:, :4].copy(), conf=pred[:, 4].copy(), cls=pred[:, 5].copy())
        return [Result(orig_shape=orig_shape, names=self.names, boxes=boxes, masks=masks, keypoints=keypoints)]

    __call__ = predict

    def __repr__(self) -> str:   # pragma: no cover - debugging aid
        return f"ultra_rep.YOLO({os.path.basename(self.model_path)}, task={self.task}, EP={self.bound_provider})"
