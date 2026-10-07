"""Plan section 3.5 -- the relative-path contract, end to end.

These tests deliberately drive the REAL ``Config.load()`` and the REAL ``build_dirs()`` rather than
the section 2.7 resolver in isolation. The defect they exist to prevent was precisely that those two
agreed in unit tests and disagreed in a process: ``Config.resolve_path()`` joined ``Path.cwd()``
while ``runtime.config_io.resolve_run_paths()`` joined the settings file's directory, and the
shipped default ``output.dir`` was the relative string ``runs`` -- so it was the DEFAULT path, not an
edge case. Every test here therefore runs from a CWD that is not the settings directory.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from leafmachine3.core import paths
from leafmachine3.core.config import Config
from leafmachine3.core.dirs import build_dirs
from leafmachine3.core.runtime.config_io import resolve_run_paths, storage_roles
from leafmachine3.machine3 import _cli_overrides

REPO = Path(__file__).resolve().parent.parent


def write_cfg(directory: Path, **output) -> Path:
    """A minimal but REAL settings file: Config.load() must accept it unmodified."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "in").mkdir(exist_ok=True)
    cfg = {
        "version": 3,
        "project": {
            "run_name": "probe",
            "input": {"dirs": ["in"]},
            "output": {"dir": "runs", **output},
        },
    }
    p = directory / "LM3_settings.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return p


@pytest.fixture
def elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Chdir to a directory that is neither the checkout nor the settings directory."""
    cwd = tmp_path / "some_other_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


# --- 1. one settings file, two launcher CWDs ------------------------------------------------- #

def test_the_same_settings_file_resolves_identically_from_two_different_cwds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = write_cfg(tmp_path / "cfgdir")
    a, b = tmp_path / "cwd_a", tmp_path / "cwd_b"
    a.mkdir()
    b.mkdir()

    seen = []
    for cwd in (a, b, REPO):
        monkeypatch.chdir(cwd)
        cfg = Config.load(cfg_path)
        seen.append((build_dirs(cfg).root, build_dirs(cfg).db_path, resolve_run_paths(cfg).run_dir))

    assert len(set(seen)) == 1, f"launcher CWD changed the resolved run: {seen}"
    assert seen[0][0] == tmp_path / "cfgdir" / "runs" / "probe"


# --- 2, 3. relative output.dir and tmp_dir ---------------------------------------------------- #

def test_a_relative_yaml_output_dir_resolves_against_the_settings_file(
    tmp_path: Path, elsewhere: Path
) -> None:
    cfg_path = write_cfg(tmp_path / "cfgdir")
    dirs = build_dirs(Config.load(cfg_path))
    assert dirs.root == tmp_path / "cfgdir" / "runs" / "probe"
    assert Path(os.getcwd()) not in dirs.root.parents, "resolved against the CWD"


def test_a_relative_yaml_tmp_dir_resolves_against_the_settings_file(
    tmp_path: Path, elsewhere: Path
) -> None:
    cfg_path = write_cfg(tmp_path / "cfgdir", tmp_dir="scratch")
    dirs = build_dirs(Config.load(cfg_path))
    assert dirs.tmp == tmp_path / "cfgdir" / "scratch" / "probe" / "_tmp_original"


# --- 4. relative input and model paths -------------------------------------------------------- #

def test_relative_input_and_model_paths_resolve_against_the_settings_file(
    tmp_path: Path, elsewhere: Path
) -> None:
    d = tmp_path / "cfgdir"
    cfg_path = write_cfg(d)
    raw = yaml.safe_load(cfg_path.read_text())
    raw["modules"] = {"plant_detector": {"model": {"path": "models/plant.onnx"},
                                         "models_dir": "models/rulers"}}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = Config.load(cfg_path)
    assert Path(cfg.resolve_path("in")) == d / "in"
    assert Path(cfg.resolve_path("models/plant.onnx")) == d / "models" / "plant.onnx"
    assert Path(cfg.resolve_path("models/rulers")) == d / "models" / "rulers"


# --- 5, 6. CLI and direct-call overrides use the CALLER's CWD --------------------------------- #

def test_relative_cli_overrides_are_absolutized_against_the_callers_cwd(elsewhere: Path) -> None:
    ov = _cli_overrides("in_here", "out_there", None)
    assert ov["project"]["input"]["dirs"] == [str(elsewhere / "in_here")]
    assert ov["project"]["output"]["dir"] == str(elsewhere / "out_there")


def test_a_relative_override_beats_the_yaml_and_keeps_the_callers_base(
    tmp_path: Path, elsewhere: Path
) -> None:
    """The rule that makes rules 2/3 differ from rule 1: same string, two different bases."""
    cfg_path = write_cfg(tmp_path / "cfgdir")
    cfg = Config.load(cfg_path, overrides=_cli_overrides(None, "runs", None))
    # YAML "runs" would be <cfgdir>/runs; a CLI "runs" is <caller cwd>/runs.
    assert build_dirs(cfg).root == elsewhere / "runs" / "probe"


# --- 7. absolute paths are untouched ---------------------------------------------------------- #

def test_absolute_paths_pass_through_unchanged(tmp_path: Path, elsewhere: Path) -> None:
    out = tmp_path / "absolute_out"
    cfg_path = write_cfg(tmp_path / "cfgdir")
    raw = yaml.safe_load(cfg_path.read_text())
    raw["project"]["output"]["dir"] = str(out)
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    cfg = Config.load(cfg_path)
    assert build_dirs(cfg).root == out / "probe"
    assert Path(cfg.resolve_path(str(out))) == out


# --- 8. THE invariant ------------------------------------------------------------------------- #

@pytest.mark.parametrize("output_dir", ["runs", "nested/deep/runs", None])
def test_build_dirs_and_the_early_resolver_agree(
    tmp_path: Path, elsewhere: Path, output_dir: str | None
) -> None:
    d = tmp_path / "cfgdir"
    cfg_path = write_cfg(d)
    if output_dir is None:                      # absolute case
        output_dir = str(tmp_path / "abs_out")
    raw = yaml.safe_load(cfg_path.read_text())
    raw["project"]["output"]["dir"] = output_dir
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = Config.load(cfg_path)
    built, early = build_dirs(cfg), resolve_run_paths(cfg)
    assert built.root == early.run_dir
    assert built.db_path == early.active_db_path
    assert built.logs == early.log_path.parent


def test_the_builtin_default_output_dir_is_not_a_hidden_config_directory(
    tmp_path: Path, elsewhere: Path
) -> None:
    """`auto` must never land under <user-config>/lm3 (plan section 3.5)."""
    d = tmp_path / "cfgdir"
    d.mkdir()
    (d / "in").mkdir()
    (d / "LM3_settings.yaml").write_text(yaml.safe_dump(
        {"version": 3, "project": {"run_name": "probe", "input": {"dirs": ["in"]}}}), encoding="utf-8")
    cfg = Config.load(d / "LM3_settings.yaml")
    root = build_dirs(cfg).root
    assert root.is_absolute()
    assert paths.user_config_dir() not in root.parents, f"default output landed in the config dir: {root}"
    assert root == paths.default_output_dir() / "probe"


# --- 9, 10. the repository's own configurations ----------------------------------------------- #

@pytest.mark.parametrize("name", sorted(p.name for p in (REPO / "examples").glob("*.yaml")))
def test_migrated_example_configs_still_point_at_their_repository_files(
    name: str, elsewhere: Path
) -> None:
    """After migration each example must resolve to the SAME repo files as before, from any CWD."""
    cfg = Config.load(REPO / "examples" / name)
    for raw in cfg.project.input.get("dirs", []):
        resolved = Path(cfg.resolve_path(raw))
        assert resolved.is_absolute()
        if Path(str(raw)).is_absolute():
            # Absolute inputs were never touched by the migration, and some of them deliberately
            # name generated artifacts under examples_out/ that need not exist in a clean checkout.
            assert resolved == Path(str(raw))
            continue
        assert resolved.exists(), f"{name}: migrated input {raw} -> {resolved} does not exist"
        assert REPO / "examples" / "images" == resolved
    for mod in cfg.modules.values():
        model = mod.get("model") if isinstance(mod, dict) else None
        if isinstance(model, dict) and model.get("path"):
            resolved = Path(cfg.resolve_path(model["path"]))
            assert resolved.exists(), f"{name}: model {model['path']} -> {resolved}"
            assert REPO / "models" in resolved.parents
        if isinstance(mod, dict) and mod.get("models_dir"):
            assert Path(cfg.resolve_path(mod["models_dir"])) == REPO / "models" / "ruler_classifier"


def test_an_entirely_absolute_config_is_unaffected(tmp_path: Path, elsewhere: Path) -> None:
    """R5: a configuration whose paths are all absolute cannot be moved by rule 1.

    This used to load LM3_settings_global_greening.yaml, the real absolute-path config of a batch
    run. That file is an operational artifact and is no longer tracked, so the same shape is built
    here: absolute input dirs and output dir, loaded from a settings file in a third directory.
    """
    abs_in, abs_out = tmp_path / "absolute_in", tmp_path / "absolute_out"
    abs_in.mkdir()
    settings = write_cfg(tmp_path / "settings_dir")
    raw = yaml.safe_load(settings.read_text(encoding="utf-8"))
    raw["project"]["input"]["dirs"] = [str(abs_in)]
    raw["project"]["output"]["dir"] = str(abs_out)
    settings.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    cfg = Config.load(settings)
    assert Path(str(cfg.project.output.dir)).is_absolute()
    for raw in cfg.project.input.get("dirs", []):
        assert Path(str(raw)).is_absolute()
        assert Path(cfg.resolve_path(raw)) == Path(str(raw))
    assert build_dirs(cfg).root == resolve_run_paths(cfg).run_dir


# --- 11, 12. records are absolute; nothing re-derives a base later ----------------------------- #

def test_every_published_runtime_path_is_absolute(tmp_path: Path, elsewhere: Path) -> None:
    cfg = Config.load(write_cfg(tmp_path / "cfgdir"))
    early = resolve_run_paths(cfg)
    for field in ("run_dir", "artifact_dir", "active_state_dir", "active_db_path", "log_path"):
        value = getattr(early, field)
        assert value is not None and Path(value).is_absolute(), f"{field} is not absolute: {value}"
    for role, value in storage_roles(early).items():
        assert value is None or Path(value).is_absolute(), f"{role} is not absolute: {value}"


def test_resolution_does_not_depend_on_the_cwd_once_the_config_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load once, then move the process. Nothing may re-derive a base from the new CWD."""
    cfg_path = write_cfg(tmp_path / "cfgdir", tmp_dir="scratch")
    monkeypatch.chdir(tmp_path)
    cfg = Config.load(cfg_path)
    first = (build_dirs(cfg).root, build_dirs(cfg).tmp, cfg.resolve_path("in"),
             cfg.resolve_path("models/x.onnx"), resolve_run_paths(cfg).active_db_path)

    moved = tmp_path / "moved_away"
    moved.mkdir()
    monkeypatch.chdir(moved)
    second = (build_dirs(cfg).root, build_dirs(cfg).tmp, cfg.resolve_path("in"),
              cfg.resolve_path("models/x.onnx"), resolve_run_paths(cfg).active_db_path)
    assert first == second


def test_a_relative_path_with_no_settings_file_refuses_rather_than_guessing() -> None:
    """Rule 6: never invent a base. A Config with no source_path must raise, not join the CWD."""
    with pytest.raises(ValueError, match="source_path"):
        Config({"project": {}}).resolve_path("runs")
