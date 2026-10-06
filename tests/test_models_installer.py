"""The model installer: status states, the backup/rollback protocol and crash recovery.

Everything runs against a tmp models root with a tiny synthetic lock and a fake downloader, so no
network and no real weights. The download step is the seam ``installer.install(downloader=...)``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from leafmachine3.modelhub import installer, registry


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A lock with two actions (one single-file, one 'ensemble' of two repos) + a fake Hub."""
    hub = {  # (repo_id, src) -> bytes
        ("org/det", "onnx/model.onnx"): b"DET-ONNX-v2",
        ("org/det", "training_metadata.json"): b'{"k": "det"}',
        ("org/ens_a", "onnx/model.onnx"): b"ENS-A",
        ("org/ens_a", "label_map.json"): b'["x","y"]',
        ("org/ens_b", "onnx/model.onnx"): b"ENS-B",
    }
    def f(repo, src, dest, fmt, optional=False):
        return {"src": src, "dest": dest, "format": fmt, "sha256": _sha(hub[(repo, src)]),
                "bytes": len(hub[(repo, src)]), "optional": optional}
    lock = {
        "schema_version": 1, "lm3_version": "3.0.0", "default_formats": ["onnx"],
        "actions": {
            "det": {"required": True, "units": [{"repo_id": "org/det", "revision": "rev2", "model_key": "x", "files": [
                f("org/det", "onnx/model.onnx", "det/model.onnx", "onnx"),
                f("org/det", "training_metadata.json", "det/training_metadata.json", "meta", optional=True)]}]},
            "ens": {"required": True, "units": [
                {"repo_id": "org/ens_a", "revision": "a1", "model_key": "a", "files": [
                    f("org/ens_a", "onnx/model.onnx", "ens/a/exported/model.onnx", "onnx"),
                    f("org/ens_a", "label_map.json", "ens/label_map.json", "meta")]},
                {"repo_id": "org/ens_b", "revision": "b1", "model_key": "b", "files": [
                    f("org/ens_b", "onnx/model.onnx", "ens/b/exported/model.onnx", "onnx")]}]},
            "ph": {"required": True, "placeholder": True, "units": [{"repo_id": "org/ph", "revision": None, "model_key": None,
                   "files": [{"src": "model.json", "dest": "ph/model.json", "format": "meta", "sha256": None, "bytes": None}]}]},
        },
    }
    lock_path = tmp_path / "lock.yaml"
    lock_path.write_text(yaml.safe_dump(lock))
    root = tmp_path / "models"
    calls: list[tuple[str, str]] = []

    def downloader(unit, lf, scratch, progress):
        calls.append((unit.repo_id, lf.src))
        if (unit.repo_id, lf.src) in failing:
            raise installer.InstallError("boom")
        p = Path(scratch) / lf.src
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(hub[(unit.repo_id, lf.src)])
        return p

    failing: set[tuple[str, str]] = set()
    monkeypatch.delenv(installer.ENV_ROOT, raising=False)

    class W:
        pass
    w = W()
    w.root, w.lock, w.hub, w.calls, w.failing, w.downloader = root, registry.load_lock(lock_path), hub, calls, failing, downloader
    w.lock_path = lock_path
    return w


def test_fresh_root_is_missing_and_install_fills_it(world):
    st = installer.status(world.root, lock=world.lock)
    assert st["actions"]["det"]["state"] == "missing"
    assert st["actions"]["ens"]["state"] == "missing"
    assert st["actions"]["ph"]["state"] == "unavailable"
    assert st["summary"]["button_label"] == "Install Models from Hugging Face"
    assert st["summary"]["unavailable"] == ["ph"] and st["summary"]["tab_attention"] is True

    events: list[dict] = []
    st = installer.install(world.root, lock=world.lock, downloader=world.downloader, progress=events.append)
    assert (world.root / "det/model.onnx").read_bytes() == b"DET-ONNX-v2"
    assert (world.root / "ens/a/exported/model.onnx").read_bytes() == b"ENS-A"
    assert (world.root / "ens/label_map.json").read_bytes() == b'["x","y"]'
    assert st["actions"]["det"]["state"] == "current"
    assert st["actions"]["ens"]["state"] == "current"
    assert st["actions"]["ph"]["state"] == "unavailable"      # placeholder is never fetched
    assert any(e["type"] == "skip" and e["action"] == "ph" for e in events)
    assert st["summary"]["needs_attention"] is False      # nothing the install button can fix...
    assert st["summary"]["tab_attention"] is True         # ...but an unpublished required model still flags the tab
    rec = json.loads((world.root / "installed.json").read_text())
    assert rec["actions"]["det"]["revisions"] == {"org/det": "rev2"}
    assert not list(world.root.rglob("*.backup"))
    assert not (world.root / installer.DOWNLOADS_DIR).exists()


def test_current_install_is_a_noop(world):
    installer.install(world.root, lock=world.lock, downloader=world.downloader)
    n = len(world.calls)
    installer.install(world.root, lock=world.lock, downloader=world.downloader)
    assert len(world.calls) == n, "nothing should be downloaded when everything is current"


def test_optional_sidecar_absence_is_not_missing(world):
    installer.install(world.root, lock=world.lock, downloader=world.downloader)
    (world.root / "det/training_metadata.json").unlink()
    st = installer.status(world.root, lock=world.lock)
    assert st["actions"]["det"]["state"] == "current"


def test_hand_placed_matching_file_is_adopted_as_current(world):
    p = world.root / "det/model.onnx"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"DET-ONNX-v2")
    st = installer.status(world.root, lock=world.lock)
    assert st["actions"]["det"]["state"] == "current"
    rec = json.loads((world.root / "installed.json").read_text())
    assert rec["actions"]["det"]["files"]["det/model.onnx"]["sha256"] == _sha(b"DET-ONNX-v2")


def test_outdated_file_is_replaced_and_backup_removed(world):
    p = world.root / "det/model.onnx"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"DET-ONNX-v1")                       # an older model
    st = installer.status(world.root, lock=world.lock)
    assert st["actions"]["det"]["state"] == "outdated"
    assert st["summary"]["button_label"] == "Install Models from Hugging Face"   # ens still missing
    installer.install(world.root, lock=world.lock, downloader=world.downloader, actions=["ens"])
    st = installer.status(world.root, lock=world.lock)
    assert st["summary"]["button_label"] == "Newer Models are Available"       # only det outdated now
    installer.install(world.root, lock=world.lock, downloader=world.downloader)
    assert p.read_bytes() == b"DET-ONNX-v2"
    assert not p.with_name("model.onnx.backup").exists()


def test_failed_download_rolls_back_the_whole_action(world):
    installer.install(world.root, lock=world.lock, downloader=world.downloader)
    # make ens outdated: both members changed on disk
    (world.root / "ens/a/exported/model.onnx").write_bytes(b"OLD-A")
    (world.root / "ens/b/exported/model.onnx").write_bytes(b"OLD-B")
    world.failing.add(("org/ens_b", "onnx/model.onnx"))
    with pytest.raises(installer.InstallError):
        installer.install(world.root, lock=world.lock, downloader=world.downloader, actions=["ens"])
    # nothing moved: the download phase failed before any swap
    assert (world.root / "ens/a/exported/model.onnx").read_bytes() == b"OLD-A"
    assert (world.root / "ens/b/exported/model.onnx").read_bytes() == b"OLD-B"
    assert not list(world.root.rglob("*.backup"))
    assert not (world.root / installer.DOWNLOADS_DIR).exists()


def test_failure_during_swap_restores_backups(world, monkeypatch):
    installer.install(world.root, lock=world.lock, downloader=world.downloader)
    a = world.root / "ens/a/exported/model.onnx"
    b = world.root / "ens/b/exported/model.onnx"
    a.write_bytes(b"OLD-A")
    b.write_bytes(b"OLD-B")
    # the second move blows up AFTER the first file was already swapped in
    real_move = installer._move
    n = {"i": 0}
    def flaky_move(src, dest):
        n["i"] += 1
        if n["i"] == 2:
            raise OSError("disk full")
        real_move(src, dest)
    monkeypatch.setattr(installer, "_move", flaky_move)
    with pytest.raises(installer.InstallError):
        installer.install(world.root, lock=world.lock, downloader=world.downloader, actions=["ens"])
    assert a.read_bytes() == b"OLD-A", "the already-swapped file must be restored from its .backup"
    assert b.read_bytes() == b"OLD-B"
    assert not list(world.root.rglob("*.backup"))


def test_crash_leaves_backup_and_repair_restores_it(world):
    """Simulate a process death between 'rename dest -> .backup' and 'move new file in'."""
    installer.install(world.root, lock=world.lock, downloader=world.downloader)
    p = world.root / "det/model.onnx"
    os.replace(p, p.with_name("model.onnx.backup"))          # dest gone, backup stranded
    st = installer.status(world.root, lock=world.lock)
    assert st["repaired"] == [{"file": "det/model.onnx", "action": "restored_backup"}]
    assert p.read_bytes() == b"DET-ONNX-v2"
    assert st["actions"]["det"]["state"] == "current"


def test_repair_discards_backup_when_dest_is_already_pinned(world):
    installer.install(world.root, lock=world.lock, downloader=world.downloader)
    p = world.root / "det/model.onnx"
    p.with_name("model.onnx.backup").write_bytes(b"DET-ONNX-v1")   # died after the move, before cleanup
    st = installer.status(world.root, lock=world.lock)
    assert st["repaired"] == [{"file": "det/model.onnx", "action": "discarded_backup"}]
    assert p.read_bytes() == b"DET-ONNX-v2"


def test_hash_mismatch_is_rejected_and_nothing_changes(world):
    world.hub[("org/det", "onnx/model.onnx")] = b"TAMPERED"
    with pytest.raises(installer.InstallError, match="hash mismatch"):
        installer.install(world.root, lock=world.lock, downloader=world.downloader, actions=["det"])
    assert not (world.root / "det/model.onnx").exists()


def test_models_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(installer.ENV_ROOT, str(tmp_path / "elsewhere"))
    assert installer.models_root(tmp_path / "LM3_settings.yaml") == tmp_path / "elsewhere"
    monkeypatch.delenv(installer.ENV_ROOT)
    assert installer.models_root(tmp_path / "LM3_settings.yaml") == tmp_path / "models"


def test_config_resolve_model_path_reroots_models_prefix(tmp_path, monkeypatch):
    from leafmachine3.core.config import Config
    settings = tmp_path / "LM3_settings.yaml"
    settings.write_text("project: {name: t}\n")
    cfg = Config.load(settings)
    monkeypatch.delenv("LM3_MODELS_DIR", raising=False)
    assert cfg.resolve_model_path("models/plant_detector/model.onnx") == str(tmp_path / "models/plant_detector/model.onnx")
    monkeypatch.setenv("LM3_MODELS_DIR", str(tmp_path / "shared"))
    assert cfg.resolve_model_path("models/plant_detector/model.onnx") == str(tmp_path / "shared/plant_detector/model.onnx")
    assert cfg.resolve_model_path("other/x.onnx") == str(tmp_path / "other/x.onnx")   # only the models/ prefix re-roots
    assert cfg.resolve_model_path("/abs/x.onnx") == "/abs/x.onnx"


def test_shipped_lock_parses_and_covers_every_default_stage():
    lock = registry.load_lock()
    assert set(lock.actions) >= {"archival_detector", "plant_detector", "landmark_detector", "leaf_segmenter",
                                 "ruler_classifier", "specimen_segmenter", "mp_conversion_factor"}
    for a in lock.actions.values():
        for u in a.units:
            assert u.repo_id.startswith("phyloforfun/lm3_")
            if not a.placeholder:
                assert u.revision and len(u.revision) == 40, f"{u.repo_id} must pin a full commit"
                for f in u.files:
                    assert f.sha256 and f.bytes


def test_cli_status_and_install(world, monkeypatch, capsys):
    from leafmachine3.modelhub import cli
    monkeypatch.setattr(installer, "_download", world.downloader)
    assert cli.main(["--dest", str(world.root), "--lock", str(world.lock_path), "status"]) == 0
    out = capsys.readouterr().out
    assert "MISSING" in out and "Install Models from Hugging Face" in out
    assert cli.main(["--dest", str(world.root), "--lock", str(world.lock_path), "install", "--yes"]) == 0
    assert cli.main(["--dest", str(world.root), "--lock", str(world.lock_path), "verify"]) == 0
    assert "Models are up to date" in capsys.readouterr().out
