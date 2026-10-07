"""LM3 must have exactly one version, and a wheel must contain the whole application.

Two defects this pins, both found by audit rather than by anything failing:

1. Three disagreeing versions. ``pyproject.toml`` said 0.1.0 while the server, ``/healthz``, the
   hardware profile and ``app/package.json`` all said 3.0.0 -- and the runtime launch manifest reads
   installed metadata, so it recorded 0.1.0 for a run whose ``/healthz`` advertised 3.0.0. Plan Step
   5b compares exactly these values across a process boundary.

2. ``packages.find`` used ``include = ["leafmachine3*"]``, which also globs ``leafmachine3_backup*``
   (13 packages, 13 MB of superseded code), while ``leafmachine3/server/ui/`` has no ``__init__.py``
   and so was invisible to the package finder AND undeclared as package data. An installed wheel
   would have shipped old backup code and no browser UI -- which is the entire interface for the
   loopback and SSH-tunnelled cluster deployments.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

try:                                   # tomllib is stdlib from 3.11; LM3 supports 3.10 and CI
    import tomllib                     # tests it, so this module must import there too.
except ModuleNotFoundError:            # pragma: no cover - only taken on 3.10
    import tomli as tomllib

import pytest

REPO = Path(__file__).resolve().parent.parent


def declared_scripts() -> dict[str, str]:
    return pyproject()["project"]["scripts"]


def pyproject() -> dict:
    with (REPO / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


# --- one version -------------------------------------------------------------------------------- #

def test_every_version_source_agrees() -> None:
    import leafmachine3
    from leafmachine3.core.runtime.config_io import lm3_version
    from leafmachine3.setup.hardware_setup import LM3_VERSION

    declared = pyproject()["project"]["version"]
    electron = json.loads((REPO / "app" / "package.json").read_text(encoding="utf-8"))["version"]

    sources = {
        "pyproject.toml": declared,
        "leafmachine3.__version__": leafmachine3.__version__,
        "hardware_setup.LM3_VERSION": LM3_VERSION,
        "config_io.lm3_version()": lm3_version(),
        "app/package.json": electron,
    }
    assert len(set(sources.values())) == 1, f"LM3 versions disagree: {sources}"


def test_the_checkout_fallback_matches_the_declared_version() -> None:
    """A bare checkout has no distribution metadata; its fallback must not drift from pyproject."""
    import leafmachine3

    assert leafmachine3._FALLBACK_VERSION == pyproject()["project"]["version"]


def test_no_module_hardcodes_a_version_string_any_more() -> None:
    """The server and the hardware profile must DERIVE the version, not restate it."""
    declared = pyproject()["project"]["version"]
    pattern = re.compile(r'["\']' + re.escape(declared) + r'["\']')
    for rel in ("leafmachine3/server/app.py", "leafmachine3/setup/hardware_setup.py"):
        source = (REPO / rel).read_text(encoding="utf-8")
        offenders = [ln.strip() for ln in source.splitlines()
                     if pattern.search(ln) and not ln.lstrip().startswith("#")]
        assert not offenders, f"{rel} hardcodes the version instead of importing it: {offenders}"


# --- the wheel contains the application ---------------------------------------------------------- #

def test_the_package_finder_excludes_the_backup_tree_and_the_config_declares_the_ui() -> None:
    """Cheap declarative check, so the expensive build test below is not the only guard."""
    from setuptools import find_packages

    cfg = pyproject()["tool"]["setuptools"]
    found = find_packages(where=str(REPO),
                          include=cfg["packages"]["find"]["include"],
                          exclude=cfg["packages"]["find"].get("exclude", []))
    assert not [p for p in found if p.startswith("leafmachine3_backup")], (
        "the superseded leafmachine3_backup tree would ship in the wheel")
    assert "leafmachine3.core" in found, "the package finder lost the real package"

    declared = cfg["package-data"]["leafmachine3"]
    for needed in ("server/ui/*", "server/ui/js/*", "server/ui/css/*", "server/ui/js/tabs/*"):
        assert needed in declared, f"package-data does not declare {needed}; the wheel omits the UI"
    webview = cfg["package-data"]["leafmachine3.postprocessing"]
    for needed in ("stl_webview/*.html", "stl_webview/src/*",
                   "stl_webview/assets/*", "stl_webview/assets/fonts/*"):
        assert needed in webview, f"package-data does not declare {needed}; the STL page ships broken"


@pytest.mark.slow
def test_an_installed_wheel_is_a_working_application(tmp_path: Path) -> None:
    """Build a wheel, INSTALL it into a throwaway venv, and exercise it from an unrelated CWD.

    Reading a wheel's zip index proves what was packaged. It does not prove the result imports, that
    its resources resolve through ``importlib.resources`` rather than a checkout-relative path, that
    the static mount can be constructed, or that the console scripts exist. Those are the properties
    a cluster or desktop install actually depends on, so they are checked in an installed
    interpreter, with the working directory somewhere that has no LM3 checkout in sight.
    """
    try:
        import build.__main__  # noqa: F401 - proves ``python -m build`` is executable
    except ImportError:
        pytest.skip("`build` is not installed; CI installs the `test` extra, which provides it")

    dist = tmp_path / "dist"
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist), str(REPO)],
        capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, f"wheel build failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"

    # --- what is in the archive ---------------------------------------------------------------- #
    names = zipfile.ZipFile(wheels[0]).namelist()

    def present(fragment: str) -> bool:
        return any(fragment in n for n in names)

    assert present("leafmachine3/server/ui/index.html"), "wheel omits the browser UI entry point"
    assert present("leafmachine3/server/ui/js/"), "wheel omits the UI javascript"
    assert present("leafmachine3/server/ui/css/"), "wheel omits the UI stylesheets"
    assert present("leafmachine3/server/ui/settings_meta.json"), "wheel omits the settings metadata"
    assert present("leafmachine3/core/schema.sql"), "wheel omits the database schema"
    assert any(n.startswith("leafmachine3/setup/calibration_images/") and n.endswith(".jpg")
               for n in names), "wheel omits the calibration images"
    assert not any(n.startswith("leafmachine3_backup/") for n in names), (
        "wheel ships the superseded leafmachine3_backup tree")
    # The STL deliverable ships WHOLE or not at all. Before this it was `_build.py` and nothing
    # else -- a build script with no inputs. `electron_app_plan.md` and closeout finding F-017 both
    # call for it to be packaged, so it is; and "whole" is asserted against the tree rather than a
    # hand-listed set, so adding a font or a source module cannot silently fall out of the wheel.
    webview = REPO / "leafmachine3" / "postprocessing" / "stl_webview"
    on_disk = {str(f.relative_to(REPO)) for f in webview.rglob("*") if f.is_file()}
    assert on_disk - set(names) == set(), (
        f"stl_webview is packaged but incomplete; missing: {sorted(on_disk - set(names))}")
    assert any(n.endswith("stl_webview/generate_3d_file.html") for n in names)
    assert sum(1 for n in names if n.endswith(".woff2")) == 10, "the inlined fonts are missing"

    # --- install it and use it ----------------------------------------------------------------- #
    venv_dir = tmp_path / "venv"
    assert subprocess.run([sys.executable, "-m", "venv", str(venv_dir)],
                          capture_output=True, timeout=600).returncode == 0
    bindir = "Scripts" if sys.platform == "win32" else "bin"
    py = venv_dir / bindir / ("python.exe" if sys.platform == "win32" else "python")
    install = subprocess.run([str(py), "-m", "pip", "install", "--no-deps", "-q", str(wheels[0])],
                             capture_output=True, text=True, timeout=1800)
    assert install.returncode == 0, f"wheel install failed:\n{install.stdout}\n{install.stderr}"

    probe = r"""
import json, sys
from pathlib import Path
import leafmachine3
from importlib.resources import files
out = {"version": leafmachine3.__version__}
root = Path(leafmachine3.__file__).resolve().parent
out["inside_site_packages"] = "site-packages" in str(root)
ui = root / "server" / "ui"
out["ui_index"] = (ui / "index.html").is_file()
out["ui_meta"] = (ui / "settings_meta.json").is_file()
out["ui_js"] = (ui / "js").is_dir()
out["schema"] = (root / "core" / "schema.sql").is_file()
cal = files("leafmachine3.setup.calibration_images")
out["calibration_jpgs"] = sum(1 for p in cal.iterdir() if p.name.endswith(".jpg"))
out["calibration_inside_install"] = str(root) in str(cal)
out["backup_importable"] = True
try:
    import leafmachine3_backup  # noqa: F401
except ImportError:
    out["backup_importable"] = False
wv = root / "postprocessing" / "stl_webview"
out["stl_webview_page"] = (wv / "generate_3d_file.html").is_file()
out["stl_webview_fonts"] = len(list((wv / "assets" / "fonts").glob("*.woff2"))) if wv.exists() else 0
out["ui_dir"] = str(ui)
print(json.dumps(out))
"""
    elsewhere = tmp_path / "unrelated_cwd"
    elsewhere.mkdir()
    run = subprocess.run([str(py), "-c", probe], capture_output=True, text=True,
                         cwd=str(elsewhere), timeout=600)
    assert run.returncode == 0, f"installed package probe failed:\n{run.stdout}\n{run.stderr}"
    got = json.loads(run.stdout.strip().splitlines()[-1])

    declared = pyproject()["project"]["version"]
    assert got["version"] == declared, f"installed version {got['version']} != {declared}"
    assert got["inside_site_packages"], "the probe imported the checkout, not the installed wheel"
    assert got["ui_index"] and got["ui_meta"] and got["ui_js"], f"UI missing from the install: {got}"
    assert got["schema"], "schema.sql missing from the install"
    assert got["calibration_jpgs"] > 0, "no calibration images in the install"
    assert got["calibration_inside_install"], "calibration images resolved OUTSIDE the installation"
    assert not got["backup_importable"], "leafmachine3_backup is importable from the install"
    assert got["stl_webview_page"], "the STL deliverable page is not in the installation"
    assert got["stl_webview_fonts"] == 10, (
        f"the STL page's inlined fonts did not install: {got['stl_webview_fonts']}/10")
    # The mount is constructed HERE, against the path inside the installation. The venv is built
    # with --no-deps on purpose -- so the wheel is proven coherent on its own, offline -- which
    # means starlette is not importable in it. Constructing from this interpreter against the
    # installed directory tests the same property: is what the wheel laid down a valid mount target?
    from starlette.staticfiles import StaticFiles

    installed_ui = Path(got["ui_dir"])
    assert installed_ui.is_dir() and str(installed_ui) != str(REPO / "leafmachine3" / "server" / "ui")
    StaticFiles(directory=str(installed_ui), html=True)

    # --- console scripts ------------------------------------------------------------------------ #
    # Existence plus the declared target. The scripts are NOT executed: the venv is --no-deps by
    # design, so `lm3-serve --help` would die importing pyyaml -- a missing dependency, not a
    # packaging defect. What a no-deps install can prove is that the entry point was declared and
    # generated pointing at the right callable, which is exactly what was missing before (`lm3-serve`
    # did not exist at all, so an installed wheel had no documented way to start the server).
    # `lm3` is the canonical command with a `serve` subcommand, which is what the plan specifies
    # (section 3.1's settings contract, Step 5's exit gate) and what `lm3 connect` will extend.
    # A separate `lm3-serve` script would be a second public spelling of one command.
    # `machine3` and `lm3-setup` stay their own entry points: `machine3` is the long-standing
    # pipeline command, and section 2.13 requires `lm3-setup` to be spawnable as its own subprocess.
    expected = {
        "machine3": ("leafmachine3.machine3", "main"),
        "lm3-setup": ("leafmachine3.setup.hardware_setup", "main"),
        "lm3": ("leafmachine3.cli", "main"),
    }
    assert "lm3-serve" not in declared_scripts(), (
        "lm3-serve is a redundant second spelling of `lm3 serve`; the plan specifies the subcommand")
    declared = declared_scripts()
    for script, (module, func) in expected.items():
        assert declared.get(script) == f"{module}:{func}", (
            f"pyproject declares {script!r} as {declared.get(script)!r}, expected {module}:{func}")
        exe = venv_dir / bindir / (f"{script}.exe" if sys.platform == "win32" else script)
        assert exe.exists(), f"console script {script!r} is missing from the install"
        if sys.platform != "win32":
            body = exe.read_text(encoding="utf-8", errors="replace")
            assert module in body and func in body, (
                f"{script} does not dispatch to {module}:{func}; it contains:\n{body[:400]}")

    # `lm3 --help` must work in an install with NO server extra: the dispatcher imports FastAPI only
    # when `serve` is actually requested. This runs the installed script for real.
    helped = subprocess.run([str(venv_dir / bindir / "lm3"), "--help"],
                            capture_output=True, text=True, cwd=str(elsewhere), timeout=300)
    assert helped.returncode == 0, f"lm3 --help failed:\n{helped.stdout}\n{helped.stderr}"
    assert "serve" in helped.stdout, f"`lm3 --help` does not advertise the serve subcommand:\n{helped.stdout}"
