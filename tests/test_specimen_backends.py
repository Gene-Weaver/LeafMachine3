"""Specimen segmenter: chosen by NAME (model.key), each model fed the input it was trained on.

No real weights: ONNX sessions and the YOLO model are faked. The input-preparation tests compare
against the TRAINING references (reproduced inline, line for line, from the training projects):

* UNet++  -- LM3_Specimen_Segmentation/common/seg_data.py ``letterbox_pair`` (image half)
* BiRefNet -- BiRefNet ``utils.path_to_image`` (cv2 INTER_LINEAR stretch) + ``ToTensor`` + ``Normalize``
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from leafmachine3.inference import specimen_segmenter as SS
from leafmachine3.inference.specimen_models import (
    DEFAULT_SPECIMEN_MODEL_KEY, SPECIMEN_MODELS, UnknownSpecimenModel, resolve_specimen_model,
)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
REPO = Path(__file__).resolve().parents[1]


def _sheet(h=1601, w=1033, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), min(h, w) // 3, (30, 120, 40), -1)
    return img


# ----------------------------------------------------------------- the name ---
def test_missing_key_resolves_to_the_unet_default_with_a_warning():
    spec, warns = resolve_specimen_model({"model": {"path": "x.onnx"}})
    assert spec.key == DEFAULT_SPECIMEN_MODEL_KEY == "unetpp_effb7_1024"
    assert any("model.key is not set" in w for w in warns)


def test_unknown_key_raises_with_the_known_list():
    with pytest.raises(UnknownSpecimenModel, match="yolo26x_seg_1280"):
        resolve_specimen_model({"model": {"key": "segformer_b5"}})


@pytest.mark.parametrize("key", sorted(SPECIMEN_MODELS))
def test_each_key_resolves_and_matching_legacy_imgsz_is_silent(key):
    spec, warns = resolve_specimen_model({"model": {"key": key}, "imgsz": SPECIMEN_MODELS[key].imgsz})
    assert spec.key == key and warns == []


def test_a_different_legacy_imgsz_is_ignored_with_a_warning():
    spec, warns = resolve_specimen_model({"model": {"key": "yolo26x_seg_1280"}, "imgsz": 1024})
    assert spec.imgsz == 1280
    assert any("imgsz=1024 is ignored" in w for w in warns)


def test_registry_gui_and_lock_agree_on_the_names():
    meta = json.loads((REPO / "leafmachine3/server/ui/settings_meta.json").read_text())
    assert set(meta["modules.specimen_segmenter.model.key"]["enum"]) == set(SPECIMEN_MODELS)
    lock = yaml.safe_load((REPO / "leafmachine3/modelhub/models.lock.yaml").read_text())
    default_unit = lock["actions"]["specimen_segmenter"]["units"][0]
    assert default_unit["model_key"] == DEFAULT_SPECIMEN_MODEL_KEY
    assert default_unit["repo_id"] == SPECIMEN_MODELS[DEFAULT_SPECIMEN_MODEL_KEY].hub_repo
    alts = lock["alternates"]["specimen_segmenter"]
    assert set(alts) | {DEFAULT_SPECIMEN_MODEL_KEY} == set(SPECIMEN_MODELS)
    for k, spec in alts.items():
        assert spec["units"][0]["repo_id"] == SPECIMEN_MODELS[k].hub_repo


def test_config_validate_rejects_an_unknown_key(mock_config_path):
    from leafmachine3.core.config import Config
    data = yaml.safe_load(mock_config_path.read_text())
    data.setdefault("modules", {}).setdefault("specimen_segmenter", {})["enabled"] = True
    data["modules"]["specimen_segmenter"]["model"] = {"key": "not_a_model", "path": "m.onnx", "format": "onnx"}
    mock_config_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="not a known specimen segmenter"):
        Config.load(mock_config_path).validate()


# ------------------------------------------------------ input = training input ---
def _ref_unet_input(img_bgr, imgsz):
    """letterbox_pair (image half), verbatim logic, then /255 + ImageNet norm."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    s = imgsz / max(H, W)
    h2, w2 = max(1, round(H * s)), max(1, round(W * s))
    ri = cv2.resize(img_rgb, (w2, h2), interpolation=cv2.INTER_AREA)
    ci = np.full((imgsz, imgsz, 3), 255, np.uint8)
    top, left = (imgsz - h2) // 2, (imgsz - w2) // 2
    ci[top:top + h2, left:left + w2] = ri
    return ((ci.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]


def _ref_birefnet_input(img_bgr, imgsz):
    """path_to_image: cv2.resize(BGR, (w, h), INTER_LINEAR) -> RGB; ToTensor (/255); Normalize."""
    image = cv2.resize(img_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]


@pytest.mark.parametrize("hw", [(1601, 1033), (900, 1500), (1024, 1024)])
def test_unet_input_is_byte_identical_to_its_training_letterbox(hw):
    img = _sheet(*hw)
    x, _ = SS.prep_letterbox_imagenet(img, 1024)
    assert x.shape == (1, 3, 1024, 1024) and np.array_equal(x, _ref_unet_input(img, 1024))


@pytest.mark.parametrize("hw", [(1601, 1033), (900, 1500)])
def test_birefnet_input_is_byte_identical_to_its_training_loader(hw):
    img = _sheet(*hw)
    x, _ = SS.prep_stretch_imagenet(img, 1024)
    assert x.shape == (1, 3, 1024, 1024) and np.array_equal(x, _ref_birefnet_input(img, 1024))


def test_unet_and_birefnet_inputs_differ_for_a_non_square_sheet():
    img = _sheet(1601, 1033)
    assert not np.array_equal(SS.prep_letterbox_imagenet(img, 1024)[0], SS.prep_stretch_imagenet(img, 1024)[0])


def test_letterbox_output_drops_the_pad_and_returns_native_size():
    img = _sheet(1600, 800)
    _, meta = SS.prep_letterbox_imagenet(img, 1024)
    prob = np.zeros((1024, 1024), np.float32)
    top, left, h2, w2 = meta
    prob[top:top + h2, left:left + w2] = 0.9            # everything inside the image is plant
    m = SS.post_letterbox_imagenet(prob, meta, 1600, 800, 0.5)
    assert m.shape == (1600, 800) and m.all()


def test_stretch_output_thresholds_after_the_bilinear_resize():
    prob = np.tile(np.linspace(0, 1, 1024, dtype=np.float32), (1024, 1))
    m = SS.post_stretch_imagenet(prob, 300, 2000, 0.5)
    ref = cv2.resize(prob, (2000, 300), interpolation=cv2.INTER_LINEAR) > 0.5
    assert m.shape == (300, 2000) and np.array_equal(m.astype(bool), ref)


# ---------------------------------------------------------------- backends ---
class FakeSession:
    def __init__(self, in_shape=(1, 3, 1024, 1024), n_out=1, n_in=1, logit=4.0):
        self._ins = [SimpleNamespace(name=f"input{i}", shape=list(in_shape)) for i in range(n_in)]
        self._outs = [SimpleNamespace(name=f"out{i}", shape=[1, 1, 1024, 1024]) for i in range(n_out)]
        self.logit, self.seen = logit, []

    def get_inputs(self):
        return self._ins

    def get_outputs(self):
        return self._outs

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, feed):
        x = next(iter(feed.values()))
        self.seen.append(x)
        return [np.full((1, 1, x.shape[2], x.shape[3]), self.logit, np.float32)]


@pytest.mark.parametrize("key,ref", [("unetpp_effb7_1024", _ref_unet_input), ("birefnet_hr_swinl_1024", _ref_birefnet_input)])
def test_binary_backend_feeds_each_model_its_own_training_input(key, ref):
    sess = FakeSession()
    seg = SS.BinaryOnnxSpecimenSegmenter("m.onnx", SPECIMEN_MODELS[key], session=sess, paperclean=False)
    img = _sheet(1601, 1033)
    res = seg.predict(img)
    assert np.array_equal(sess.seen[0], ref(img, 1024))
    assert res.model_name == key                          # provenance is the NAME, not a file stem
    assert (res.frame_width, res.frame_height) == (1033, 1601) and res.area_frac == pytest.approx(1.0)


@pytest.mark.parametrize("sess,why", [
    (FakeSession(in_shape=("batch", 3, "height", "width"), n_out=2), "output"),     # a YOLO export
    (FakeSession(in_shape=(1, 3, 512, 512)), "trained at 1024"),                     # wrong size
])
def test_binary_backend_refuses_a_file_that_cannot_be_the_named_model(sess, why):
    with pytest.raises(SS.SpecimenModelMismatch, match=why) as e:
        SS.BinaryOnnxSpecimenSegmenter("other.onnx", SPECIMEN_MODELS["unetpp_effb7_1024"], session=sess)
    assert "unetpp_effb7_1024" in str(e.value) and "other.onnx" in str(e.value)


class FakeYolo:
    """Stands in for ultra_rep.YOLO: returns ultra_rep.Result objects built from ``masks(h, w)``."""

    def __init__(self, masks):
        self.masks, self.calls = masks, []

    def predict(self, img, **kw):
        from leafmachine3.inference.ultra_replacements import Boxes, Masks, Result
        self.calls.append(kw)
        h, w = img.shape[:2]
        data = None if self.masks is None else self.masks(h, w)
        n = 0 if data is None else len(data)
        boxes = Boxes(xyxy=np.zeros((n, 4), np.float32), conf=np.ones(n, np.float32), cls=np.zeros(n, np.float32))
        return [Result(orig_shape=(h, w), names={0: "plant"}, boxes=boxes, masks=None if data is None else Masks(data=data))]


def test_yolo_backend_unions_instances_at_its_training_size_and_names_itself():
    def two(h, w):
        a = np.zeros((2, h, w), np.uint8)
        a[0, :h // 2] = 1
        a[1, :, :w // 3] = 1
        return a
    fake = FakeYolo(two)
    seg = SS.YoloSpecimenSegmenter("y.onnx", SPECIMEN_MODELS["yolo26x_seg_1280"], model=fake, conf=0.3, paperclean=False)
    img = _sheet(900, 600)
    m = seg.mask(img)
    assert m.shape == (900, 600) and m[:450].all() and m[:, :200].all() and not m[450:, 200:].any()
    kw = fake.calls[0]
    assert kw["imgsz"] == 1280 and kw["conf"] == 0.3 and kw["retina_masks"] is True
    assert seg.predict(img).model_name == "yolo26x_seg_1280"


def test_yolo_backend_no_instances_is_an_empty_mask():
    seg = SS.YoloSpecimenSegmenter("y.onnx", SPECIMEN_MODELS["yolo26x_seg_1280"], model=FakeYolo(None), paperclean=False)
    assert not seg.mask(_sheet(300, 200)).any()


def test_build_dispatches_on_the_name(monkeypatch):
    built = []
    monkeypatch.setattr(SS, "BinaryOnnxSpecimenSegmenter", lambda *a, **k: built.append(("binary", a[1].key)) or "b")
    monkeypatch.setattr(SS, "YoloSpecimenSegmenter", lambda *a, **k: built.append(("yolo", a[1].key, k["conf"])) or "y")
    for key in SPECIMEN_MODELS:
        SS.build_specimen_segmenter(SPECIMEN_MODELS[key], "f.onnx", yolo={"conf": 0.4})
    assert built == [("binary", "unetpp_effb7_1024"), ("binary", "birefnet_hr_swinl_1024"), ("yolo", "yolo26x_seg_1280", 0.4)]


def test_builtin_default_key_is_the_registry_default():
    from leafmachine3.core.config import builtin_defaults
    block = builtin_defaults()["modules"]["specimen_segmenter"]
    assert block["model"]["key"] == DEFAULT_SPECIMEN_MODEL_KEY
    assert set(block["yolo"]) == {"conf", "iou", "max_det"}
