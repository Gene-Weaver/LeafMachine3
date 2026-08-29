"""The Settings tab's advisory warnings must name the SAME folders the run will use.

Plan section 3.1, precedence-table preamble: "**No path falls back to the current working
directory.**" -- restated as mandatory regression gate 44 ("No resolved path falls back to the
current working directory"). ``settings_api._collect_warnings`` used to absolutize
``project.input.dirs``, ``project.output.dir`` and ``project.output.tmp_dir`` with
``Config.resolve_path``, which joins the SERVER's CWD (``core/config.py:407-412``), while
``progress_api``/``metrics_api``/``core.runtime.config_io`` all absolutize the same fields against
the settings file through :func:`leafmachine3.core.paths.resolve_project_output_dir`. With the
canonical settings file at ``<user-config>/lm3/<deployment>/LM3_settings.yaml`` and a server
launched from anywhere else, the Settings tab warned about a different ``runs`` folder than the one
the run writes to.

Every test here deliberately runs with the CWD pointed somewhere unrelated to the settings file,
because that is the only condition under which the two rules disagree.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

from leafmachine3.core.paths import resolve_project_output_dir
from leafmachine3.server import settings_api


def _write_settings(directory: Path, values: dict) -> Path:
    """Drop a settings file in ``directory`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "LM3_settings.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _values(**project_output: object) -> dict:
    """A minimal, mock-mode settings tree whose project paths are all RELATIVE.

    ``compute.mock`` keeps the model-artifact checks (which still use ``Config.resolve_path``, and
    are deliberately out of scope here) from firing, so the assertions stay about project paths.
    """
    return {
        "compute": {"mock": True},
        "project": {
            "run_name": "run",
            "input": {"dirs": ["input_images"]},
            "output": dict({"dir": "runs", "tmp_dir": "auto"}, **project_output),
        },
    }


def _warning(result: dict, code: str) -> dict:
    for w in result["warnings"]:
        if w["code"] == code:
            return w
    raise AssertionError(f"no {code!r} warning in {[w['code'] for w in result['warnings']]}")


def _unwritable(path: Path) -> bool:
    """Make ``path`` read-only, reporting whether the platform actually honored it."""
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o500)
    return not os.access(str(path), os.W_OK | os.X_OK)


def test_missing_input_names_the_settings_dir_not_the_cwd(tmp_path, monkeypatch):
    """``missing_input`` must point at ``<settings dir>/input_images`` (gate 44)."""
    settings = _write_settings(tmp_path / "cfg", _values())
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = settings_api.validate_values(_values(), settings_file=str(settings))
    msg = _warning(result, "missing_input")["msg"]

    expected = resolve_project_output_dir(settings, "input_images")
    assert str(expected) in msg
    assert str(elsewhere) not in msg


def test_output_unwritable_names_the_settings_dir_not_the_cwd(tmp_path, monkeypatch):
    """``output_unwritable`` must name ``<settings dir>/runs`` -- one answer with progress_api."""
    settings = _write_settings(tmp_path / "cfg", _values())
    if not _unwritable(settings.parent / "runs"):
        pytest.skip("filesystem or user (root?) ignores the read-only bit")
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = settings_api.validate_values(_values(), settings_file=str(settings))
    msg = _warning(result, "output_unwritable")["msg"]

    # The exact string progress_api.py and metrics_api.py resolve for the same file.
    assert str(resolve_project_output_dir(settings, "runs")) in msg
    assert str(elsewhere / "runs") not in msg


def test_tmp_unwritable_names_the_settings_dir_not_the_cwd(tmp_path, monkeypatch):
    """``project.output.tmp_dir`` follows the same rule; ``auto`` still short-circuits."""
    values = _values(tmp_dir="scratch")
    settings = _write_settings(tmp_path / "cfg", values)
    if not _unwritable(settings.parent / "scratch"):
        pytest.skip("filesystem or user (root?) ignores the read-only bit")
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = settings_api.validate_values(values, settings_file=str(settings))
    assert str(settings.parent / "scratch") in _warning(result, "tmp_unwritable")["msg"]

    # ``auto`` means "let LM3 choose": no resolution, therefore no warning.
    auto = settings_api.validate_values(_values(), settings_file=str(settings))
    assert [w for w in auto["warnings"] if w["code"] == "tmp_unwritable"] == []


def test_no_explicit_path_uses_the_canonical_chain_not_the_cwd(tmp_path, monkeypatch):
    """With no caller-supplied path, row 2 (``LM3_SETTINGS``) anchors the warnings."""
    settings = _write_settings(tmp_path / "cfg", _values())
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.setenv("LM3_SETTINGS", str(settings))
    monkeypatch.chdir(elsewhere)

    result = settings_api.validate_values(_values())
    msg = _warning(result, "missing_input")["msg"]

    assert str(settings.parent / "input_images") in msg
    assert str(elsewhere) not in msg


def test_write_settings_anchors_warnings_on_the_file_it_writes(tmp_path, monkeypatch):
    """``write_settings`` already knows the destination -- its warnings must use it."""
    target = tmp_path / "cfg" / "LM3_settings.yaml"
    target.parent.mkdir(parents=True)
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    out = settings_api.write_settings(_values(), path=str(target))
    codes = {w["code"]: w["msg"] for w in out["warnings"]}

    assert str(target.parent / "input_images") in codes["missing_input"]
    assert str(elsewhere) not in codes["missing_input"]


def test_unresolvable_settings_path_downgrades_instead_of_raising(tmp_path, monkeypatch):
    """Validation is advisory: no settings path means fewer warnings, never an exception.

    ``resolve_project_output_dir`` raises ``PathsError`` for a relative value with no anchor, so
    the guarded call sites must swallow it -- a split-brain environment should not turn "is this
    tree valid?" into a 500.
    """
    def _boom(explicit=None):
        raise settings_api.SettingsError("no canonical settings path in this environment")

    monkeypatch.setattr(settings_api, "settings_path", _boom)
    monkeypatch.chdir(tmp_path)

    result = settings_api.validate_values(_values())

    assert result["ok"] is True
    codes = [w["code"] for w in result["warnings"]]
    # The relative project paths are simply not previewed -- and above all, not CWD-joined.
    assert "missing_input" not in codes
    assert all(str(tmp_path) not in w["msg"] for w in result["warnings"])


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX absolute-path spelling")
def test_absolute_project_paths_are_untouched(tmp_path, monkeypatch):
    """An absolute configured path passes through both rules identically."""
    absolute = tmp_path / "elsewhere_inputs"
    values = _values()
    values["project"]["input"]["dirs"] = [str(absolute)]
    settings = _write_settings(tmp_path / "cfg", values)
    monkeypatch.chdir(tmp_path)

    result = settings_api.validate_values(values, settings_file=str(settings))
    assert str(absolute) in _warning(result, "missing_input")["msg"]
