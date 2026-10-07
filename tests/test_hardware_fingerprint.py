"""The hardware profile's model fingerprint: content, not size+mtime (core/digest.py, hardware_setup).

Found 2026-10-07 in an install test: every fresh install printed "hardware / driver / models changed
since the last LM3_Setup", because the models were re-downloaded (byte-identical, new mtime) and the
profile recorded size+mtime. The same profile also hid a real change it could not name (the leaf
segmenter had moved from a 142 MB .pt to a 251 MB ONNX export) and signed the ruler ensemble by its
directory inode.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from leafmachine3.core import digest
from leafmachine3.setup import hardware_setup as hs


def _fp(**over) -> hs.Fingerprint:
    base = dict(os="Linux-6.8.0", cpu="x86_64", cpu_cores=64, ram_gb=503,
                gpus=[["NVIDIA RTX 6000 Ada Generation", 49140]], driver="560.35.05",
                ort_version="1.20.2", ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                model_hashes={}, lm3_version="3.0.0")
    base.update(over)
    return hs.Fingerprint(**base)


class _Cfg:
    """The two Config methods the fingerprint uses."""

    def __init__(self, artifacts: dict[str, list[Path]], stage_blocks: dict | None = None):
        self._a, self._s = artifacts, stage_blocks or {}

    def _stage_artifacts(self, key):
        return [str(p) for p in self._a.get(key, [])]

    def stage(self, key):
        return self._s.get(key, {})

    def bind_hardware(self, profile):
        self.bound = profile


# -------------------------------------------------------------------------------------------- digests

def test_digests_are_cached_by_stat_and_invalidated_by_a_change(tmp_path, monkeypatch):
    f = tmp_path / "model.onnx"
    f.write_bytes(b"weights v1")
    cache = tmp_path / "cache.json"
    calls = []
    real = digest.sha256_file
    monkeypatch.setattr(digest, "sha256_file", lambda p: calls.append(p) or real(p))
    first = digest.file_digests([f], cache)[str(f)]
    again = digest.file_digests([f], cache)[str(f)]
    assert first == again and len(calls) == 1                      # second call served from cache
    f.write_bytes(b"weights v2 -- a different size")
    changed = digest.file_digests([f], cache)[str(f)]
    assert changed != first and len(calls) == 2


def test_a_redownload_with_a_new_mtime_has_the_same_identity(tmp_path):
    """The false alarm: identical bytes, new mtime. Content identity is unaffected."""
    a, b = tmp_path / "a" / "model.onnx", tmp_path / "b" / "model.onnx"
    for p in (a, b):
        p.parent.mkdir()
        p.write_bytes(b"identical weights")
    import os
    os.utime(b, (1_800_000_000, 1_800_000_000))
    cache = tmp_path / "cache.json"
    assert digest.combined_digest([a], cache) == digest.combined_digest([b], cache)


def test_a_multi_file_stage_hashes_every_file(tmp_path):
    files = []
    for name in ("m1.onnx", "m2.onnx", "label_map.json"):
        p = tmp_path / name
        p.write_text(name)
        files.append(p)
    cache = tmp_path / "cache.json"
    whole = digest.combined_digest(files, cache)
    files[0].write_text("m1 retrained")
    assert digest.combined_digest(files, cache) != whole           # the FIRST file counts too
    assert digest.combined_digest([tmp_path / "missing"], cache) is None


def test_the_ruler_ensemble_is_fingerprinted_by_the_files_it_loads(tmp_path):
    models = tmp_path / "ruler_classifier"
    for member in ("yolo26x_cls_224", "yolo26n_cls_224", "dinov2_frozen_mlp"):
        (models / member / "exported").mkdir(parents=True)
        (models / member / "exported" / "model.onnx").write_text(member)
        (models / member / "metadata.json").write_text("{}")
    (models / "label_map.json").write_text("[]")
    (models / "unrelated_checkpoint.pt").write_text("not loaded")
    cfg = _Cfg({"ruler_classifier": [models]})
    files = {p.relative_to(models).as_posix() for p in hs._stage_model_files(cfg, "ruler_classifier")}
    assert files == {f"{m}/exported/model.onnx" for m in ("yolo26x_cls_224", "yolo26n_cls_224", "dinov2_frozen_mlp")} \
        | {f"{m}/metadata.json" for m in ("yolo26x_cls_224", "yolo26n_cls_224", "dinov2_frozen_mlp")} \
        | {"label_map.json"}


# -------------------------------------------------------------------------------------------- comparison

def test_identical_fingerprints_have_no_differences():
    fp = _fp(model_hashes={"plant_detector": "sha256:aa"})
    assert hs.compare_fingerprints(fp, replace(fp)) == ([], {})


def test_differences_are_named():
    stored = _fp(model_hashes={"plant_detector": "sha256:aa", "ect_only": "sha256:cc"})
    now = _fp(driver="575.10", model_hashes={"plant_detector": "sha256:bb", "landmark_detector": "sha256:dd"})
    diffs, upgrades = hs.compare_fingerprints(stored, now)
    assert "driver: 560.35.05 -> 575.10" in diffs
    assert "model plant_detector: changed (content differs)" in diffs
    assert "model landmark_detector: added" in diffs and "model ect_only: removed" in diffs
    assert upgrades == {}


def test_legacy_entries_upgrade_when_the_size_matches_and_stay_visible_when_it_does_not(tmp_path):
    same, grown = tmp_path / "plant.onnx", tmp_path / "leaf.onnx"
    same.write_bytes(b"x" * 100)
    grown.write_bytes(b"y" * 250)
    cfg = _Cfg({"plant_detector": [same], "leaf_segmenter": [grown]})
    stored = _fp(model_hashes={"plant_detector": "sig:100:1784221993", "leaf_segmenter": "sig:142:1782764923"})
    now = _fp(model_hashes={"plant_detector": "sha256:p", "leaf_segmenter": "sha256:l"})
    diffs, upgrades = hs.compare_fingerprints(stored, now, cfg)
    assert upgrades == {"plant_detector": "sha256:p"}
    assert diffs == ["model leaf_segmenter: changed (size 142 -> 250 bytes)"]


def test_the_run_time_check_upgrades_the_profile_and_names_only_the_real_change(tmp_path, monkeypatch, caplog):
    """The 2026-10-07 install, replayed: six legacy entries that match by size, one real change."""
    plant, leaf = tmp_path / "plant.onnx", tmp_path / "leaf.onnx"
    plant.write_bytes(b"p" * 100)
    leaf.write_bytes(b"l" * 250)
    cfg = _Cfg({"plant_detector": [plant], "leaf_segmenter": [leaf]})
    profile_path = tmp_path / "hardware_settings.yaml"
    stored = hs.HardwareSettings(fingerprint=_fp(model_hashes={"plant_detector": "sig:100:1",
                                                               "leaf_segmenter": "sig:142:1"}),
                                 provider="CUDAExecutionProvider", precision="fp16", gpus=[], cpu_cores=64,
                                 ram_gb=503, tmp_dir="", io_workers=1, stages={"plant_detector": {"x": 1}},
                                 generated_at="then", lm3_version="3.0.0")
    hs._write(profile_path, stored)
    monkeypatch.setattr(hs, "hardware_profile_path", lambda cfg=None: profile_path)
    monkeypatch.setattr(hs, "migrate_legacy_profile", lambda cfg=None: None)
    monkeypatch.setattr(hs, "_fingerprint", lambda cfg: _fp(model_hashes={"plant_detector": "sha256:p",
                                                                          "leaf_segmenter": "sha256:l"}))
    with caplog.at_level(logging.INFO):
        hs.ensure_hardware_profile(cfg)
    rewritten = hs._load(profile_path)
    assert rewritten.fingerprint.model_hashes == {"plant_detector": "sha256:p", "leaf_segmenter": "sig:142:1"}
    assert rewritten.stages == {"plant_detector": {"x": 1}}               # the tuning itself is untouched
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "model leaf_segmenter: changed (size 142 -> 250 bytes)" in warnings[0]
    assert "plant_detector" not in warnings[0]
