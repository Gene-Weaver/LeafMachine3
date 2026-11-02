"""The release tools that keep a repo's ``coreml_fp32`` packages true to its ONNX model."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
ONNX = "onnx/model.onnx"
PACKAGE = "coreml_fp32/model.json"


def _tool(rel: str):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def uploader():
    return _tool("tools/release/hf_upload_verify.py")


# --------------------------------------------------------------------------------------------------
# hf_upload_verify.py: what an upload may do to the packages
# --------------------------------------------------------------------------------------------------
def test_a_card_edit_may_not_delete_valid_packages(uploader):
    with pytest.raises(SystemExit, match="would delete valid packages"):
        uploader.check_coreml_fp32("r", {ONNX: "a"}, {ONNX: "a", PACKAGE: None}, None, drop=False)


def test_the_override_deletes_them_and_says_so(uploader):
    note = uploader.check_coreml_fp32("r", {ONNX: "a"}, {ONNX: "a", PACKAGE: None}, None, drop=True)
    assert "removes coreml_fp32/" in note


def test_new_weights_remove_the_old_packages(uploader):
    note = uploader.check_coreml_fp32("r", {ONNX: "new"}, {ONNX: "old", PACKAGE: None}, None, drop=False)
    assert "removes coreml_fp32/" in note


def test_an_unknown_remote_onnx_hash_counts_as_unchanged(uploader):
    with pytest.raises(SystemExit, match="would delete valid packages"):
        uploader.check_coreml_fp32("r", {ONNX: "a"}, {ONNX: None, PACKAGE: None}, None, drop=False)


def test_a_staged_package_must_match_the_staged_onnx(uploader):
    with pytest.raises(SystemExit, match="made from ONNX"):
        uploader.check_coreml_fp32("r", {ONNX: "new"}, {}, {"onnx": {"sha256": "old"}}, drop=False)
    assert uploader.check_coreml_fp32("r", {ONNX: "new"}, {}, {"onnx": {"sha256": "new"}}, drop=False) is None


def test_a_repo_without_packages_uploads_freely(uploader):
    assert uploader.check_coreml_fp32("r", {ONNX: "a"}, {ONNX: "b"}, None, drop=False) is None
    assert uploader.check_coreml_fp32("r", {ONNX: "a"}, {}, None, drop=False) is None


# --------------------------------------------------------------------------------------------------
# build_lock.py: a lock that stops pinning a package says so
# --------------------------------------------------------------------------------------------------
def _lock(**actions):
    return {"actions": {a: {"units": [{"repo_id": repo, "files": [{"format": f} for f in formats]}
                                      for repo, formats in units.items()]} for a, units in actions.items()}}


def test_the_lock_names_models_that_lost_their_package():
    lost = _tool("tools/modelhub/build_lock.py").lost_coreml_packages
    old = _lock(plant={"p": ["onnx", "coreml_fp32"]}, leaf={"l": ["onnx", "coreml_fp32"]}, ruler={"r": ["onnx"]})
    assert lost(old, _lock(plant={"p": ["onnx"]}, leaf={"l": ["onnx", "coreml_fp32"]}, ruler={"r": ["onnx"]})) == ["p"]
    assert lost(old, _lock(plant={"new_repo": ["onnx"]}, leaf={"l": ["coreml_fp32"]}, ruler={})) == ["p"]
    assert lost(old, old) == []
    assert lost(None, old) == []


# --------------------------------------------------------------------------------------------------
# The rest of the release flow, on a repo staged without the old coreml/ folder
# --------------------------------------------------------------------------------------------------
CARD = """---
lm3:
  formats: [pytorch, onnx, torchscript, openvino]
---

## Files

| Folder | File | Use |
|---|---|---|
| `onnx/` | `model.onnx` | onnxruntime |
| `torchscript/` | `model.torchscript` | libtorch |

## Use in LeafMachine3

Stage `plant_detector`.

## Citation
"""


def test_publishing_adds_the_package_to_a_card_without_the_old_coreml_row():
    pub = _tool("tools/modelhub/publish_coreml_fp32.py")
    card = pub.edit_card(CARD)
    assert "formats: [pytorch, onnx, torchscript, openvino, coreml_fp32]" in card
    assert f"| `onnx/` | `model.onnx` | onnxruntime |\n{pub.FILES_ROW}\n" in card
    assert f"Stage `plant_detector`.\n\n{pub.USE_LINE}\n\n## Citation" in card
    assert pub.add_format(["pytorch", "onnx"]) == ["pytorch", "onnx", "coreml_fp32"]


def test_the_exporter_skips_models_with_no_core_ml_conversion(monkeypatch, tmp_path):
    from leafmachine3.modelhub import registry

    def unit(key):
        files = [SimpleNamespace(format="onnx", dest=f"{key}/model.onnx"),
                 SimpleNamespace(format="pytorch", dest=f"{key}/best.pt")]
        return SimpleNamespace(model_key=key, files=files)

    keys = ["yolo26x_det_1280", "dinov2_frozen_mlp", "dinov3_vits16_tile_four_896_mlp",
            "birefnet_hr_swinl_1024", "unetpp_effb7_1024"]
    lock = SimpleNamespace(actions={"stage": SimpleNamespace(units=[unit(k) for k in keys])})
    monkeypatch.setattr(registry, "load_lock", lambda: lock)
    jobs = _tool("tools/modelhub/export_coreml_fp32.py").lock_jobs(tmp_path, tmp_path, tmp_path)
    assert [(j[1], j[2]) for j in jobs] == [("yolo26x_det_1280", "yolo"), ("unetpp_effb7_1024", "unetpp")]
