"""The Models tab's data: ``installer.catalog`` / ``activation_plan`` and ``/v1/models/{catalog,activate}``.

The lock here pins one stage with a default and an alternate, each in two formats (one of them a
multi-file format), a locked stage, and a "json" stage, against a fake Hub -- the same style as
test_models_installer.py. Nothing names a real model: the tab is driven by the lock alone.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from leafmachine3.modelhub import installer, registry


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hub = {
        ("org/det", "onnx/model.onnx"): b"DET-ONNX",
        ("org/det", "coreml/model.mlpackage/Manifest.json"): b"{}",
        ("org/det", "coreml/model.mlpackage/Data/w.bin"): b"WEIGHTS",
        ("org/det_n", "onnx/model.onnx"): b"DETN-ONNX",
        ("org/det_n", "coreml/model.mlpackage/Manifest.json"): b"{}",
        ("org/fit", "json/model.json"): b'{"k": 1}',
    }

    def f(repo, src, dest, fmt, optional=False):
        return {"src": src, "dest": dest, "format": fmt, "sha256": _sha(hub[(repo, src)]),
                "bytes": len(hub[(repo, src)]), "optional": optional}

    lock = {
        "schema_version": 1, "lm3_version": "3.0.0", "default_formats": ["onnx", "json"],
        "actions": {
            "det": {"required": True, "activatable": True, "settings": {"imgsz": 1280},
                    "units": [{"repo_id": "org/det", "revision": "r1", "model_key": "x_1280", "files": [
                        f("org/det", "onnx/model.onnx", "det/model.onnx", "onnx"),
                        f("org/det", "coreml/model.mlpackage/Manifest.json", "det/coreml/model.mlpackage/Manifest.json", "coreml"),
                        f("org/det", "coreml/model.mlpackage/Data/w.bin", "det/coreml/model.mlpackage/Data/w.bin", "coreml")]}]},
            "fit": {"required": True, "activatable": False,
                    "units": [{"repo_id": "org/fit", "revision": "f1", "model_key": "sqrt", "files": [
                        f("org/fit", "json/model.json", "fit/model.json", "json")]}]},
        },
        "alternates": {"det": {"n_640": {"settings": {"imgsz": 640},
                                          "units": [{"repo_id": "org/det_n", "revision": "r9", "model_key": "n_640", "files": [
                                              f("org/det_n", "onnx/model.onnx", "det/n_640/model.onnx", "onnx"),
                                              f("org/det_n", "coreml/model.mlpackage/Manifest.json", "det/n_640/coreml/model.mlpackage/Manifest.json", "coreml")]}]}}},
    }
    lock_path = tmp_path / "lock.yaml"
    lock_path.write_text(yaml.safe_dump(lock))
    root = tmp_path / "models"

    def downloader(unit, lf, scratch, progress):
        p = Path(scratch) / lf.src
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(hub[(unit.repo_id, lf.src)])
        return p

    monkeypatch.delenv(installer.ENV_ROOT, raising=False)

    class W:
        pass
    w = W()
    w.root, w.lock, w.hub, w.downloader, w.lock_path = root, registry.load_lock(lock_path), hub, downloader, lock_path
    return w


def _settings(path="models/det/model.onnx", key="x_1280"):
    return {"modules": {"det": {"model": {"key": key, "path": path, "format": "onnx"}, "imgsz": 1280},
                        "fit": {"model": {"path": "models/fit/model.json", "format": "json"}}}}


def _fmt(cat, stage, repo, fmt):
    st = next(s for s in cat["stages"] if s["stage"] == stage)
    unit = next(u for v in st["variants"] for u in v["units"] if u["repo_id"] == repo)
    return unit["formats"][fmt]


def test_catalog_lists_every_variant_and_format_with_states(world):
    installer.install(world.root, lock=world.lock, downloader=world.downloader)        # defaults in default formats
    cat = installer.catalog(world.root, lock=world.lock, settings_values=_settings())
    assert cat["runnable_formats"] == ["onnx", "json"]
    det = next(s for s in cat["stages"] if s["stage"] == "det")
    assert det["activatable"] is True and det["state"] == "current"
    assert [v["model_key"] for v in det["variants"]] == ["x_1280", "n_640"]
    assert [v["default"] for v in det["variants"]] == [True, False]
    assert det["variants"][1]["settings"] == {"imgsz": 640}
    # runnable formats lead; the default's onnx is on disk and active, its coreml is not
    assert list(_fmt(cat, "det", "org/det", "onnx").keys()) >= ["state"]
    assert list(det["variants"][0]["units"][0]["formats"]) == ["onnx", "coreml"]
    onnx = _fmt(cat, "det", "org/det", "onnx")
    assert (onnx["state"], onnx["active"], onnx["runnable"], onnx["runtime_file"]) == ("current", True, True, "det/model.onnx")
    coreml = _fmt(cat, "det", "org/det", "coreml")
    assert (coreml["state"], coreml["files"], coreml["bytes"], coreml["runnable"]) == ("missing", 2, 2 + 7, False)
    assert _fmt(cat, "det", "org/det_n", "onnx")["state"] == "missing"
    # the locked stage: not activatable, but its json is current and matched
    fit = next(s for s in cat["stages"] if s["stage"] == "fit")
    assert fit["activatable"] is False and fit["active"]["matched"] is True
    assert det["active"]["model_key"] == "x_1280" and det["active"]["matched_format"] == "onnx"


def test_catalog_reports_a_partial_multi_file_format(world):
    installer.install(world.root, lock=world.lock, actions=["det"], formats=["coreml"], downloader=world.downloader)
    (world.root / "det/coreml/model.mlpackage/Data/w.bin").unlink()
    cat = installer.catalog(world.root, lock=world.lock, settings_values=_settings())
    assert _fmt(cat, "det", "org/det", "coreml")["state"] == "partial"


def test_catalog_marks_an_outdated_format(world):
    installer.install(world.root, lock=world.lock, downloader=world.downloader)
    (world.root / "det/model.onnx").write_bytes(b"OLD-ONNX")             # a previous revision, hand-placed
    cat = installer.catalog(world.root, lock=world.lock, settings_values=_settings())
    assert _fmt(cat, "det", "org/det", "onnx")["state"] == "outdated"
    det = next(s for s in cat["stages"] if s["stage"] == "det")
    assert det["state"] == "outdated" and det["active_state"] == "outdated"
    # the module's own state follows the ACTIVE model: pointing the settings at the (current) alternate
    # turns it green while the default still has its update waiting
    installer.install(world.root, lock=world.lock, models=[("det", "n_640")], downloader=world.downloader)
    cat2 = installer.catalog(world.root, lock=world.lock, settings_values=_settings(path="models/det/n_640/model.onnx", key="n_640"))
    det2 = next(s for s in cat2["stages"] if s["stage"] == "det")
    assert det2["state"] == "outdated" and det2["active_state"] == "current"
    # the updater: installing the default's onnx again replaces it with the pinned bytes
    installer.install(world.root, lock=world.lock, actions=["det"], formats=["onnx"], downloader=world.downloader)
    assert (world.root / "det/model.onnx").read_bytes() == b"DET-ONNX"
    assert _fmt(installer.catalog(world.root, lock=world.lock, settings_values=_settings()), "det", "org/det", "onnx")["state"] == "current"


def test_catalog_shows_a_custom_settings_path_as_unmatched(world):
    installer.install(world.root, lock=world.lock, downloader=world.downloader)
    cat = installer.catalog(world.root, lock=world.lock, settings_values=_settings(path="/elsewhere/my.onnx"))
    det = next(s for s in cat["stages"] if s["stage"] == "det")
    assert det["active"]["matched"] is False and det["active"]["path"] == "/elsewhere/my.onnx"
    assert not any(f["active"] for v in det["variants"] for u in v["units"] for f in u["formats"].values())


def test_install_one_format_of_an_alternate_touches_nothing_else(world):
    installer.install(world.root, lock=world.lock, downloader=world.downloader)
    before = (world.root / "det/model.onnx").stat().st_mtime_ns
    installer.install(world.root, lock=world.lock, models=[("det", "n_640")], formats=["coreml"], downloader=world.downloader)
    assert (world.root / "det/n_640/coreml/model.mlpackage/Manifest.json").is_file()
    assert not (world.root / "det/n_640/model.onnx").exists()
    assert (world.root / "det/model.onnx").stat().st_mtime_ns == before
    cat = installer.catalog(world.root, lock=world.lock, settings_values=_settings())
    assert _fmt(cat, "det", "org/det_n", "coreml")["state"] == "current"
    assert _fmt(cat, "det", "org/det_n", "onnx")["state"] == "missing"


def test_activation_plan_requires_an_installed_runnable_file_and_an_unlocked_stage(world):
    installer.install(world.root, lock=world.lock, downloader=world.downloader)
    plan = installer.activation_plan(world.lock, "det", "x_1280", "onnx", world.root)
    assert plan["model"] == {"key": "x_1280", "path": "models/det/model.onnx", "format": "onnx"}
    assert plan["set"] == {"imgsz": 1280} and plan["restore"] == {} and plan["unset"] == []
    with pytest.raises(installer.InstallError, match="not installed"):
        installer.activation_plan(world.lock, "det", "n_640", "onnx", world.root)
    with pytest.raises(installer.InstallError, match="not a format LM3 can run"):
        installer.activation_plan(world.lock, "det", "x_1280", "coreml", world.root)
    with pytest.raises(installer.InstallError, match="always runs its default"):
        installer.activation_plan(world.lock, "fit", "sqrt", "json", world.root)
    installer.install(world.root, lock=world.lock, models=[("det", "n_640")], downloader=world.downloader)
    plan = installer.activation_plan(world.lock, "det", "n_640", "onnx", world.root)
    assert plan["model"]["path"] == "models/det/n_640/model.onnx" and plan["set"] == {"imgsz": 640}


def test_repair_leaves_the_scratch_of_a_running_install_alone(world, monkeypatch):
    """status()/catalog() are polled by every open GUI while a download runs; the sweep used to
    delete the half-written file from under the downloader."""
    seen = {}

    def slow_downloader(unit, lf, scratch, progress):
        p = world.downloader(unit, lf, scratch, progress)
        installer.repair(world.root, world.lock)          # a concurrent status() call
        seen[lf.src] = p.is_file()
        return p

    installer.install(world.root, lock=world.lock, actions=["det"], formats=["coreml"], downloader=slow_downloader)
    assert all(seen.values()) and len(seen) == 2
    assert (world.root / "det/coreml/model.mlpackage/Data/w.bin").read_bytes() == b"WEIGHTS"
    # ...but a stale scratch from a dead process is still swept
    stale = world.root / installer.DOWNLOADS_DIR / "x"
    stale.mkdir(parents=True)
    (stale / "f").write_bytes(b"x")
    import os
    old = 0
    os.utime(stale / "f", (old, old))
    os.utime(stale, (old, old))
    installer.repair(world.root, world.lock)
    assert not (world.root / installer.DOWNLOADS_DIR).exists()


def test_adopting_a_matching_sidecar_keeps_the_recorded_older_revision(world):
    """The record holds one revision per repo. A hand-placed metadata file that matches the lock must
    not relabel an OLDER model file as the pinned revision: the update dialog shows that revision."""
    installer.install(world.root, lock=world.lock, downloader=world.downloader)
    (world.root / "det/model.onnx").write_bytes(b"OLD-ONNX")
    rec = installer.read_record(world.root)
    rec["actions"]["det"]["revisions"]["org/det"] = "r0"
    rec["actions"]["det"]["files"]["det/model.onnx"]["sha256"] = "0" * 64
    installer.write_record(world.root, rec)
    # a sidecar the record does not know yet, byte-identical to the pinned one -> adopted
    rec["actions"]["det"]["files"].pop("det/coreml/model.mlpackage/Manifest.json", None)
    installer.write_record(world.root, rec)
    (world.root / "det/coreml/model.mlpackage").mkdir(parents=True, exist_ok=True)
    (world.root / "det/coreml/model.mlpackage/Manifest.json").write_bytes(b"{}")
    cat = installer.catalog(world.root, lock=world.lock, settings_values=_settings())
    unit = next(s for s in cat["stages"] if s["stage"] == "det")["variants"][0]["units"][0]
    assert unit["formats"]["onnx"]["state"] == "outdated"
    assert unit["installed_revision"] == "r0" and unit["revision"] == "r1"
    assert installer.read_record(world.root)["actions"]["det"]["revisions"]["org/det"] == "r0"


def test_concurrent_status_calls_do_not_lose_the_record(world):
    """Every open GUI polls status()/catalog(); the record writes they trigger must not race."""
    import threading
    installer.install(world.root, lock=world.lock, downloader=world.downloader)
    errors: list[BaseException] = []

    def poll():
        try:
            for _ in range(25):
                rec = installer.read_record(world.root)
                installer.write_record(world.root, rec)
                installer.status(world.root, lock=world.lock)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=poll) for _ in range(6)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, errors
    assert installer.read_record(world.root)["actions"]["det"]["revisions"]["org/det"] == "r1"
    assert not list(world.root.glob(".installed.*.tmp"))
