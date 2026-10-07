"""Desktop commands must reject drift and run npm/scripts using uv's pinned Node, even on a host
with an old Node, Conda, or an Electron terminal environment.
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from leafmachine3 import desktop, doctor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    for filename in ("app/package.json", "app/package-lock.json", "app/desktop-contract.json", "leafmachine3/_env_contract.json"):
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, target)
    contract = json.loads((tmp_path / "leafmachine3/_env_contract.json").read_text())
    installed = {row["name"]: row["version"] for row in contract["extras"]["cpu"]}
    installed.update({"nodejs-wheel": contract["desktop"]["node"],
                      "nodejs-wheel-binaries": contract["desktop"]["node"], "uv": contract["uv"]})
    env = doctor.Env(python_version=contract["python"], executable=str(tmp_path / ".venv/bin/python"),
                     prefix=str(tmp_path / ".venv"), base_prefix="/home/u/.local/share/uv/python/cpython-3.11.17",
                     sys_platform="linux", machine="x86_64", installed=installed, environ={}, contract=contract)
    return tmp_path, env


@pytest.mark.parametrize("changes", [
    {"prefix": "/legacy/.venv_LM3"}, {"python_version": "3.11.0"}, {"base_prefix": "/usr"},
    {"installed": {}},
])
def test_desktop_refuses_other_interpreters_and_unlocked_packages(project, changes):
    root, env = project
    with pytest.raises(desktop.DesktopEnvironmentError):
        desktop.verify_project(root, replace(env, **changes))


def test_desktop_refuses_a_stale_js_lock_and_missing_tool_group(project):
    root, env = project
    assert desktop.verify_project(root, env) == env.contract
    packages = {k: v for k, v in env.installed.items() if k != "uv"}
    with pytest.raises(desktop.DesktopEnvironmentError, match="uv=="):
        desktop.verify_project(root, replace(env, installed=packages))
    (root / "app/package-lock.json").write_text("{}")
    with pytest.raises(desktop.DesktopEnvironmentError, match="package-lock.json"):
        desktop.verify_project(root, env)


def test_install_uses_ci_and_puts_locked_node_ahead_of_host_path(project, monkeypatch):
    root, env = project
    seen = []

    def npm(args, **kwargs):
        seen.append((args, kwargs))
        (root / "app/node_modules").mkdir(exist_ok=True)
        for name, key in (("electron", "electron"), ("electron-builder", "electron_builder")):
            package = root / "app/node_modules" / name / "package.json"
            package.parent.mkdir(exist_ok=True)
            package.write_text(json.dumps({"version": env.contract["desktop"][key]}))
        runtime = root / "app/node_modules/electron/dist/version"
        runtime.parent.mkdir(exist_ok=True)
        runtime.write_text(env.contract["desktop"]["electron"])
        return 0

    monkeypatch.setattr(doctor.Env, "current", lambda: env)
    monkeypatch.setitem(sys.modules, "nodejs_wheel", SimpleNamespace(npm=npm))
    monkeypatch.setitem(sys.modules, "nodejs_wheel.executable", SimpleNamespace(ROOT_DIR="/locked/node"))
    monkeypatch.setenv("LM3_ROOT", str(root))
    monkeypatch.setenv("PATH", "/host/node")
    monkeypatch.setenv("ELECTRON_RUN_AS_NODE", "1")
    monkeypatch.setenv("LM3_STARTUP_GATE", "0")
    assert desktop.main(["install"]) == 0
    args, kwargs = seen[0]
    assert args == ["ci"]
    assert seen[1][0] == ["run", "install:electron"]
    assert kwargs["env"]["PATH"].split(":")[:2] == [str(root / ".venv/bin"), "/locked/node/bin"]
    assert kwargs["env"]["LM3_STARTUP_GATE"] == "1"
    assert kwargs["env"]["LM3_EXTRA"] == "cpu"
    assert "ELECTRON_RUN_AS_NODE" not in kwargs["env"]
    receipt = root / "app/node_modules/.lm3-lock.json"
    assert receipt.is_file()
    # A changed lock invalidates an installation instead of testing/packaging stale node_modules.
    contract = json.loads(json.dumps(env.contract))
    contract["desktop"]["package_lock_sha256"] = "changed"
    with pytest.raises(desktop.DesktopEnvironmentError, match="need installation"):
        desktop.verify_install(root / "app", contract)


def test_failed_npm_install_cannot_authorize_later_commands(project, monkeypatch):
    root, env = project
    monkeypatch.setattr(doctor.Env, "current", lambda: env)
    monkeypatch.setitem(sys.modules, "nodejs_wheel", SimpleNamespace(npm=lambda *a, **kw: 7))
    monkeypatch.setitem(sys.modules, "nodejs_wheel.executable", SimpleNamespace(ROOT_DIR="/locked/node"))
    monkeypatch.setenv("LM3_ROOT", str(root))
    assert desktop.main(["install"]) == 7
    assert not (root / "app/node_modules/.lm3-lock.json").exists()
