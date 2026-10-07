"""The release's pinned identities agree everywhere they are written down (PACKAGING_PLAN.md 2.3).

tools/release/versions.env is the single source. It is restated, by necessity, in places tools read
on their own: uv reads ``[tool.uv] required-version`` and ``.python-version``; ``lm3 doctor`` reads
``leafmachine3/_env_contract.json``. A bump that edits one and forgets another would make the doctor
judge an install against the wrong release, so each restatement is checked here.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "release"))
import write_env_contract  # noqa: E402

ENV = write_env_contract.read_versions_env(ROOT / "tools" / "release" / "versions.env")
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
CONTRACT = json.loads((ROOT / "leafmachine3" / "_env_contract.json").read_text())


def test_python_pin_agrees():
    assert (ROOT / ".python-version").read_text().strip() == ENV["PYTHON_VERSION"]
    assert CONTRACT["python"] == ENV["PYTHON_VERSION"]
    minor = ".".join(ENV["PYTHON_VERSION"].split(".")[:2])
    lo, hi = minor, f"{minor.split('.')[0]}.{int(minor.split('.')[1]) + 1}"
    assert PYPROJECT["project"]["requires-python"] == f">={lo},<{hi}"


def test_uv_pin_agrees():
    assert PYPROJECT["tool"]["uv"]["required-version"] == f"=={ENV['UV_VERSION']}"
    assert CONTRACT["uv"] == ENV["UV_VERSION"]


def test_desktop_pins_and_locks_agree():
    """A desktop upgrade must update uv, npm and the shipped runtime identity together."""
    import hashlib

    desktop = json.loads((ROOT / "app" / "desktop-contract.json").read_text())
    package = json.loads((ROOT / "app" / "package.json").read_text())
    lock = json.loads((ROOT / "app" / "package-lock.json").read_text())
    assert PYPROJECT["dependency-groups"]["desktop"] == [
        f"nodejs-wheel=={ENV['NODE_VERSION']}", f"uv=={ENV['UV_VERSION']}"]
    assert desktop["node"] == ENV["NODE_VERSION"] == package["engines"]["node"]
    assert desktop["uv"] == ENV["UV_VERSION"]
    assert desktop["python"] == ENV["PYTHON_VERSION"]
    assert lock["packages"][""]["devDependencies"] == package["devDependencies"]
    for key, name in (("electron", "electron"), ("electron_builder", "electron-builder")):
        assert desktop[key] == package["devDependencies"][name] == lock["packages"][f"node_modules/{name}"]["version"]
    for key, path in (("backend_contract_sha256", "leafmachine3/_env_contract.json"),
                      ("package_lock_sha256", "app/package-lock.json"),
                      ("package_json_sha256", "app/package.json")):
        assert desktop[key] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    # Every registry package (including optional native builder helpers) needs an integrity pin.
    for name, row in lock["packages"].items():
        if name:
            assert row.get("version") and row.get("integrity"), name


def test_driver_minimums_agree():
    assert CONTRACT["cuda_min_driver"] == {"linux": ENV["CUDA_MIN_DRIVER_LINUX"],
                                           "win32": ENV["CUDA_MIN_DRIVER_WINDOWS"]}


def test_lm3_version_agrees():
    assert CONTRACT["lm3_version"] == PYPROJECT["project"]["version"]


def test_the_production_lock_holds_no_development_packages():
    """torch / ultralytics live only in the `full` dependency group, never in a hardware extra."""
    for extra, rows in CONTRACT["extras"].items():
        leaked = {r["name"] for r in rows} & set(CONTRACT["dev_only"])
        assert not leaked, f"{extra} would ship {leaked}"
    deps = " ".join(PYPROJECT["project"]["dependencies"]
                    + [d for e in ("gpu", "cpu", "macos") for d in PYPROJECT["project"]["optional-dependencies"][e]])
    for name in ("torch", "torchvision", "ultralytics"):
        assert f"{name}=" not in deps


def test_every_direct_dependency_is_pinned_exactly():
    for spec in PYPROJECT["project"]["dependencies"] + [
            d for e in ("gpu", "cpu", "macos") for d in PYPROJECT["project"]["optional-dependencies"][e]]:
        assert "==" in spec.split(";")[0], f"not an exact pin: {spec}"


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_lock_and_contract_are_current():
    """`uv lock --check` and the contract generator agree with pyproject.toml as committed."""
    lock = subprocess.run(["uv", "lock", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert lock.returncode == 0, lock.stderr
    assert write_env_contract.main(["--check"]) == 0


@pytest.mark.parametrize("filename", ["backend.json", "desktop.json"])
def test_contract_check_rejects_changed_line_endings(tmp_path, monkeypatch, filename):
    """The desktop validates raw file hashes; newline normalization must not hide drift."""
    monkeypatch.setattr(write_env_contract, "ROOT", tmp_path)
    monkeypatch.setattr(write_env_contract, "CONTRACT", tmp_path / "backend.json")
    monkeypatch.setattr(write_env_contract, "DESKTOP_CONTRACT", tmp_path / "desktop.json")
    monkeypatch.setattr(write_env_contract, "build", lambda: CONTRACT)
    assert write_env_contract.main([]) == 0
    assert write_env_contract.main(["--check"]) == 0
    path = tmp_path / filename
    raw = path.read_bytes()
    assert b"\r" not in raw
    path.write_bytes(raw.replace(b"\n", b"\r\n"))
    assert write_env_contract.main(["--check"]) == 1


def test_readme_setup_section_names_the_pinned_versions():
    """README's "Setup with Verification Steps" restates the uv and Python pins; keep them in step."""
    import re

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    uv_urls = re.findall(r"astral\.sh/uv/([0-9.]+)/install", readme)
    assert uv_urls and set(uv_urls) == {ENV["UV_VERSION"]}
    assert re.findall(r"uv --version\s+# must print ([0-9.]+)", readme) == [ENV["UV_VERSION"]]
    pythons = set(re.findall(r"Python (3\.\d+\.\d+)", readme))
    assert pythons <= {ENV["PYTHON_VERSION"]}, f"README names {pythons}, versions.env pins {ENV['PYTHON_VERSION']}"
