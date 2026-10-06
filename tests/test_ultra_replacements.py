"""``leafmachine3.inference.ultra_replacements`` (``ultra_rep``): the pure-onnxruntime stand-in for
``ultralytics.YOLO(...).predict`` on LM3's end2end ONNX exports.

Three layers, cheapest first:

1. **Geometry, no models, no ORT** -- letterbox / scale_boxes / scale_coords / end2end_filter /
   process_mask_native on synthetic arrays, checked against values computed by hand from the
   Ultralytics formulas they port.
2. **Reference parity for the mask decoder** -- ``process_mask_native`` against
   ``ultralytics.utils.ops.process_mask_native`` on random prototypes (skipped when ultralytics is not
   importable). This is the only function whose port is not pure integer/float bookkeeping.
3. **Static guard** -- nothing under ``leafmachine3/inference`` (or the pipeline) imports ultralytics,
   and ``torch`` appears only behind an ImportError guard. The A/B on the real models (bitwise-equal
   boxes / keypoints, mask IoU >= 0.9999 on 1,001 detections) was run against the shipped exports on
   both CPU and CUDA; it needs the models and is not a CI test.

WARNING for anyone extending layer 2 to drive ``ultralytics.YOLO`` itself: on a CPU device its ONNX
backend calls ``check_requirements("onnxruntime")`` which ``pip install``s the CPU build OVER
``onnxruntime-gpu`` in the active environment. Monkeypatch
``ultralytics.nn.backends.onnx.check_requirements`` to a no-op first.
"""
from __future__ import annotations

import importlib
import pathlib
import re

import numpy as np
import pytest

from leafmachine3.inference import ultra_replacements as ultra_rep

# ---------------------------------------------------------------------------------------------------
# 1. geometry
# ---------------------------------------------------------------------------------------------------

def test_check_imgsz_matches_ultralytics_rounding():
    assert ultra_rep.check_imgsz(1280) == (1280, 1280)
    assert ultra_rep.check_imgsz(None) == (640, 640)          # ultralytics default when a caller passes none
    assert ultra_rep.check_imgsz(1000) == (1024, 1024)        # ceil to the stride
    assert ultra_rep.check_imgsz([640, 480]) == (640, 480)


@pytest.mark.parametrize("hw", [(3000, 2000), (2000, 3000), (640, 640), (517, 1333), (100, 50)])
def test_letterbox_auto_pads_short_side_to_stride_multiple(hw):
    img = np.full((*hw, 3), 7, np.uint8)
    out = ultra_rep.letterbox(img, (1024, 1024), stride=32, auto=True)
    r = min(1024 / hw[0], 1024 / hw[1])
    new_w, new_h = round(hw[1] * r), round(hw[0] * r)
    assert max(out.shape[:2]) == 1024
    assert out.shape[0] % 32 == 0 and out.shape[1] % 32 == 0
    assert out.shape[0] >= new_h and out.shape[1] >= new_w
    # the pad is gray 114 and the content survives
    assert (out == 7).any() and ((out == 114) | (out == 7)).all()


def test_letterbox_square_when_auto_false():
    out = ultra_rep.letterbox(np.zeros((300, 100, 3), np.uint8), (640, 640), auto=False)
    assert out.shape == (640, 640, 3)


def test_scale_boxes_undoes_letterbox_exactly():
    # a 3000x2000 image letterboxed (auto) to long side 1280: gain = 1280/3000, pad_x = (864-853)/2 -> round(5.5-0.1)=5
    img1, img0 = (1280, 864), (3000, 2000)
    gain = min(1280 / 3000, 864 / 2000)
    pad_x = round((864 - round(2000 * gain)) / 2 - 0.1)
    boxes = np.array([[pad_x + 10 * gain, 20 * gain, pad_x + 1000 * gain, 1500 * gain]], np.float32)
    out = ultra_rep.scale_boxes(img1, boxes.copy(), img0)
    assert np.allclose(out, [[10, 20, 1000, 1500]], atol=1e-2)
    # clipping to the original frame
    out = ultra_rep.scale_boxes(img1, np.array([[-50, -50, 5000, 5000]], np.float32), img0)
    assert out.tolist() == [[0, 0, 2000, 3000]]


def test_scale_coords_matches_scale_boxes_on_xy_and_keeps_conf():
    img1, img0 = (640, 480), (1000, 700)
    kp = np.array([[[100.0, 200.0, 0.9], [-5.0, 999.0, 0.1]]], np.float32)
    out = ultra_rep.scale_coords(img1, kp.copy(), img0)
    bx = ultra_rep.scale_boxes(img1, np.array([[100.0, 200.0, 100.0, 200.0]], np.float32), img0)
    assert np.allclose(out[0, 0, :2], bx[0, :2])
    assert out[0, 0, 2] == np.float32(0.9) and out[0, 1, 2] == np.float32(0.1)
    assert out[0, 1, 0] == 0 and out[0, 1, 1] == 1000         # clipped to (h=1000, w=700), not dropped


def test_division_uses_the_cuda_reciprocal_convention():
    """``im /= 255`` and ``boxes /= gain`` must be ``x * (float32(1) / float32(d))``, not true division.

    That is what torch's CUDA div kernel does for a CPU-scalar divisor, i.e. what every LM3 GPU run
    and evaluation to date actually fed the network. The two differ by one ulp for 126 of the 256
    pixel values and that alone moves plant-detector confidences by up to 0.02 (module docstring).
    """
    x = np.arange(256, dtype=np.uint8)
    im = ultra_rep.preprocess(np.repeat(x, 3).reshape(1, 256, 3).repeat(32, 0), (32, 256), 32, auto=False)
    expected = x.astype(np.float32) * (np.float32(1) / np.float32(255))
    assert np.array_equal(im[0, 0, 0, :], expected)
    assert not np.array_equal(im[0, 0, 0, :], x.astype(np.float32) / 255)     # the convention is load-bearing
    boxes = np.array([[7.0, 7.0, 7.0, 7.0]], np.float32)
    ultra_rep.scale_boxes((1000, 1000), boxes, (3000, 3000))                   # gain = 1/3, no pad
    assert boxes[0, 0] == np.float32(7.0) * (np.float32(1) / np.float32(1 / 3))


def test_end2end_filter_keeps_model_order_and_slices_max_det():
    pred = np.array([[0, 0, 1, 1, 0.9, 0], [0, 0, 1, 1, 0.2, 1], [0, 0, 1, 1, 0.5, 2], [0, 0, 1, 1, 0.4, 3]], np.float32)
    out = ultra_rep.end2end_filter(pred, 0.25, max_det=2)
    assert out[:, 4].tolist() == [np.float32(0.9), np.float32(0.5)]
    assert ultra_rep.end2end_filter(pred, 0.95, 300).shape == (0, 6)


def test_process_mask_native_thresholds_crops_and_upsamples():
    rng = np.random.default_rng(0)
    proto = rng.standard_normal((32, 64, 64)).astype(np.float32)
    coeffs = rng.standard_normal((2, 32)).astype(np.float32)
    h, w = 128, 128                                               # same aspect as proto: no pad crop
    boxes = np.array([[0, 0, 64, 128], [64, 0, 128, 128]], np.float32)
    out = ultra_rep.process_mask_native(proto, coeffs, boxes, (h, w))
    assert out.shape == (2, h, w) and out.dtype == np.uint8
    assert out[0][:, 64:].max() == 0 and out[1][:, :64].max() == 0  # crop_mask zeroes outside the box
    # reference: full-res logits via the same bilinear, thresholded at 0
    import cv2
    logits = (coeffs @ proto.reshape(32, -1)).reshape(2, 64, 64)
    ref = np.stack([cv2.resize(l, (w, h), interpolation=cv2.INTER_LINEAR) > 0 for l in logits]).astype(np.uint8)
    ref[0][:, 64:] = 0
    ref[1][:, :64] = 0
    assert np.array_equal(out, ref)
    assert ultra_rep.process_mask_native(proto, coeffs[:0], boxes[:0], (h, w)).shape == (0, h, w)


# ---------------------------------------------------------------------------------------------------
# 2. reference parity with ultralytics' own mask decoder (optional dependency)
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("hw", [(400, 300), (300, 400), (97, 211), (1024, 1024)])
def test_process_mask_native_matches_ultralytics_reference(hw):
    ops = pytest.importorskip("ultralytics.utils.ops")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    proto = rng.standard_normal((32, 256, 256)).astype(np.float32)
    coeffs = (rng.standard_normal((5, 32)) * 0.3).astype(np.float32)
    h, w = hw
    boxes = np.stack([rng.uniform(0, w * 0.5, 5), rng.uniform(0, h * 0.5, 5),
                      rng.uniform(w * 0.5, w, 5), rng.uniform(h * 0.5, h, 5)], 1).astype(np.float32)
    ours = ultra_rep.process_mask_native(proto, coeffs, boxes, (h, w)).astype(bool)
    ref = ops.process_mask_native(torch.from_numpy(proto), torch.from_numpy(coeffs),
                                  torch.from_numpy(boxes), (h, w)).cpu().numpy().astype(bool)
    assert ours.shape == ref.shape
    diff = (ours ^ ref).sum()
    # last-ulp bilinear differences only: a vanishing fraction of the foreground
    assert diff <= max(2, 1e-5 * ref.sum()), f"{diff} px differ of {ref.sum()} foreground"


# ---------------------------------------------------------------------------------------------------
# 3. static guard: inference never imports ultralytics; torch only behind ImportError
# ---------------------------------------------------------------------------------------------------

def test_no_ultralytics_import_anywhere_in_runtime():
    root = pathlib.Path(ultra_rep.__file__).resolve().parents[1]
    offenders = []
    for f in root.rglob("*.py"):
        if "experiments" in f.parts:
            continue
        for line in f.read_text(errors="ignore").splitlines():
            if re.match(r"\s*(from|import)\s+ultralytics\b", line):
                offenders.append(f"{f.relative_to(root)}: {line.strip()}")
    assert not offenders, offenders


def test_ultra_rep_module_itself_imports_neither_torch_nor_ultralytics():
    src = pathlib.Path(ultra_rep.__file__).read_text()
    assert not re.search(r"^\s*(from|import)\s+(torch|ultralytics)\b", src, re.M)
    importlib.reload(ultra_rep)
