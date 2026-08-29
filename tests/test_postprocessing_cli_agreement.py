"""F-009 -- the GUI and the two standalone postprocessing CLIs must read ONE file.

`postprocessing/config.py` was rewired to the section 3.1 row-3 resolver, but both CLIs kept
defaulting `--config` to the bare string "postprocessing_settings.yaml", so the canonical branch was
unreachable from a CLI and the loader's ``None`` path was dead code for them. The GUI read
``<user-config>/lm3/<deployment>/postprocessing.yaml`` while a CLI read whatever happened to sit in
the launcher's working directory -- the same class of split section 3.1 exists to remove, pointing
the other way.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
from pathlib import Path

import pytest
import yaml

from leafmachine3.core import paths
from leafmachine3.postprocessing.config import load_settings, module_settings
from tests._contract_helpers import capture_lm3_logs

CLI_MODULES = ["generate_stl_from_mask", "generate_leaf_collage"]


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_the_cli_config_default_is_not_a_cwd_relative_string(module_name: str) -> None:
    """The defect itself: a bare filename default is resolved against the launcher's CWD."""
    source = Path(f"leafmachine3/postprocessing/{module_name}.py").read_text(encoding="utf-8")
    assert 'default="postprocessing_settings.yaml"' not in source, (
        f"{module_name} still defaults --config to a CWD-relative filename")
    assert '"--config", default=None' in source.replace("\n", " ").replace("  ", " ") or \
           '"--config",\n        default=None' in source, (
        f"{module_name} should default --config to None so the canonical resolver runs")


def test_the_loader_with_no_path_reads_the_canonical_deployment_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_settings(None) is row 3, and it is what the CLIs now reach."""
    canonical = tmp_path / "pp.yaml"
    canonical.write_text(yaml.safe_dump({"generate_stl_from_mask": {"length_mm": 42.0}}), encoding="utf-8")
    monkeypatch.setenv(paths.ENV_POSTPROCESS_SETTINGS, str(canonical))

    assert module_settings(load_settings(None), "generate_stl_from_mask") == {"length_mm": 42.0}


def test_gui_and_cli_resolve_the_same_file_from_any_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agreement this finding is about, measured from three different working directories."""
    from leafmachine3.server import app

    canonical = tmp_path / "shared_postprocessing.yaml"
    canonical.write_text(yaml.safe_dump({"generate_leaf_collage": {"marker": "canonical"}}), encoding="utf-8")
    monkeypatch.setenv(paths.ENV_POSTPROCESS_SETTINGS, str(canonical))

    # a decoy in each CWD: the old default would have found these instead
    seen = []
    for name in ("cwd_a", "cwd_b", "cwd_c"):
        cwd = tmp_path / name
        cwd.mkdir()
        (cwd / paths.LEGACY_POSTPROCESS_FILENAME).write_text(
            yaml.safe_dump({"generate_leaf_collage": {"marker": name}}), encoding="utf-8")
        monkeypatch.chdir(cwd)
        seen.append((module_settings(load_settings(None), "generate_leaf_collage"),
                     app.canonical_postprocess_settings_path()))

    markers = {tuple(sorted(s[0].items())) for s in seen}
    assert markers == {(("marker", "canonical"),)}, f"a CLI read a decoy: {seen}"
    assert len({s[1] for s in seen}) == 1
    assert seen[0][1] == canonical, "the GUI resolver disagrees with the CLI loader"


def test_the_checkout_level_file_is_adopted_by_copy_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Migration behavior for the existing checkout-level postprocessing_settings.yaml."""
    legacy = tmp_path / "checkout" / paths.LEGACY_POSTPROCESS_FILENAME
    legacy.parent.mkdir(parents=True)
    legacy.write_text(yaml.safe_dump({"generate_stl_from_mask": {"marker": "legacy"}}), encoding="utf-8")
    canonical = tmp_path / "deployment" / "postprocessing.yaml"
    monkeypatch.setenv(paths.ENV_POSTPROCESS_SETTINGS, str(canonical))

    assert not canonical.exists()
    adopted = paths.migrate_legacy_postprocessing_settings(source=legacy)

    assert adopted == canonical
    assert module_settings(load_settings(None), "generate_stl_from_mask") == {"marker": "legacy"}
    assert legacy.is_file(), "adoption COPIES; an older LM3 in that checkout keeps working"
    # one-time, not a sync
    assert paths.migrate_legacy_postprocessing_settings(source=legacy) is None
    canonical.write_text(yaml.safe_dump({"generate_stl_from_mask": {"marker": "mine"}}), encoding="utf-8")
    assert paths.migrate_legacy_postprocessing_settings(source=legacy) is None
    assert module_settings(load_settings(None), "generate_stl_from_mask") == {"marker": "mine"}


def test_resolving_the_postprocessing_path_never_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same rule as the hardware profile: resolution is pure, migration is explicit."""
    legacy = tmp_path / "checkout" / paths.LEGACY_POSTPROCESS_FILENAME
    legacy.parent.mkdir(parents=True)
    legacy.write_text("a: 1\n", encoding="utf-8")
    canonical = tmp_path / "deployment" / "postprocessing.yaml"
    monkeypatch.setenv(paths.ENV_POSTPROCESS_SETTINGS, str(canonical))

    for _ in range(5):
        paths.postprocessing_settings_path()
        load_settings(None)
    assert not canonical.exists(), "resolving or loading the postprocessing settings wrote a file"


def test_an_explicit_config_argument_still_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(yaml.safe_dump({"generate_stl_from_mask": {"marker": "explicit"}}), encoding="utf-8")
    other = tmp_path / "canonical.yaml"
    other.write_text(yaml.safe_dump({"generate_stl_from_mask": {"marker": "canonical"}}), encoding="utf-8")
    monkeypatch.setenv(paths.ENV_POSTPROCESS_SETTINGS, str(other))

    assert module_settings(load_settings(str(explicit)), "generate_stl_from_mask") == {"marker": "explicit"}


# --------------------------------------------------------------------------------------------- #
# The real CLI path: parse real argv, run the real main(), stub only the work
# --------------------------------------------------------------------------------------------- #
# Reading source text proves the default changed; it does NOT prove the parser and main() reach the
# canonical resolver. These run each CLI's actual main() from a CWD holding a decoy settings file --
# exactly what the old default would have picked up -- and capture the settings dict main() built.

@pytest.mark.parametrize(
    ("module_name", "argv"),
    [("generate_stl_from_mask", ["--paths", "x.png"]),
     ("generate_leaf_collage", ["--run-dir", "."])],
)
def test_each_cli_main_reads_the_canonical_file_and_ignores_a_cwd_decoy(
    module_name: str, argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module(f"leafmachine3.postprocessing.{module_name}")

    canonical = tmp_path / "canonical.yaml"
    canonical.write_text(yaml.safe_dump({module_name: {"marker": "canonical"}}), encoding="utf-8")
    monkeypatch.setenv(paths.ENV_POSTPROCESS_SETTINGS, str(canonical))

    cwd = tmp_path / "launch_dir"
    cwd.mkdir()
    (cwd / paths.LEGACY_POSTPROCESS_FILENAME).write_text(
        yaml.safe_dump({module_name: {"marker": "decoy"}}), encoding="utf-8")
    monkeypatch.chdir(cwd)

    captured: dict = {}

    def fake_run(settings=None, *a, **kw):
        captured["settings"] = dict(settings or {})
        return []

    monkeypatch.setattr(mod, "run", fake_run)

    assert mod.main(argv) == 0
    assert captured, "main() never reached run(); the stub did not fire"
    assert captured["settings"].get("marker") == "canonical", (
        f"{module_name}'s main() read the CWD decoy instead of the canonical file: "
        f"{captured['settings'].get('marker')!r}")


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_each_cli_main_still_honors_an_explicit_config(
    module_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canonical default must not swallow a path the user typed."""
    mod = importlib.import_module(f"leafmachine3.postprocessing.{module_name}")

    canonical = tmp_path / "canonical.yaml"
    canonical.write_text(yaml.safe_dump({module_name: {"marker": "canonical"}}), encoding="utf-8")
    monkeypatch.setenv(paths.ENV_POSTPROCESS_SETTINGS, str(canonical))
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(yaml.safe_dump({module_name: {"marker": "explicit"}}), encoding="utf-8")

    captured: dict = {}
    monkeypatch.setattr(mod, "run", lambda settings=None, *a, **kw: captured.setdefault(
        "settings", dict(settings or {})) and [] or [])

    argv = ["--config", str(explicit)]
    argv += ["--paths", "x.png"] if module_name == "generate_stl_from_mask" else ["--run-dir", "."]
    assert mod.main(argv) == 0
    assert captured["settings"].get("marker") == "explicit"


# --------------------------------------------------------------------------------------------- #
# The adopt must never clobber, even under a real race
# --------------------------------------------------------------------------------------------- #

def _adopt_worker(legacy: str, canonical: str, barrier, results) -> None:
    os.environ[paths.ENV_POSTPROCESS_SETTINGS] = canonical
    barrier.wait()                                   # release all workers at the same instant
    try:
        results.append(paths.migrate_legacy_postprocessing_settings(source=Path(legacy)) is not None)
    except Exception:                                # noqa: BLE001 - a crash is a failed adopt
        results.append(False)


def test_concurrent_adopts_produce_exactly_one_winner_and_no_clobber(tmp_path: Path) -> None:
    """CLI and GUI can both adopt before Step 5b exists. Only one may create the file.

    The old sequence was `if not target.exists(): _write_atomic(target, ...)`, and `_write_atomic`
    ends in `os.replace`, which overwrites unconditionally -- so a loser could stamp on the winner's
    file. This drives 8 real processes through the adopt simultaneously.
    """
    legacy = tmp_path / "checkout" / paths.LEGACY_POSTPROCESS_FILENAME
    legacy.parent.mkdir(parents=True)
    legacy.write_text(yaml.safe_dump({"generate_stl_from_mask": {"marker": "legacy"}}), encoding="utf-8")
    canonical = tmp_path / "deployment" / "postprocessing.yaml"

    ctx = mp.get_context("spawn")
    n = 8
    with ctx.Manager() as manager:
        results = manager.list()
        barrier = manager.Barrier(n)
        procs = [ctx.Process(target=_adopt_worker, args=(str(legacy), str(canonical), barrier, results))
                 for _ in range(n)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
        winners = list(results)

    assert sum(1 for w in winners if w) == 1, f"expected exactly one winner, got {winners}"
    assert yaml.safe_load(canonical.read_text(encoding="utf-8")) == {
        "generate_stl_from_mask": {"marker": "legacy"}}
    assert not list(canonical.parent.glob(".*tmp")), "an adopt left a temp file behind"


def test_an_adopt_never_overwrites_a_file_the_user_already_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The race's real consequence: a user save landing between the check and the write."""
    legacy = tmp_path / "checkout" / paths.LEGACY_POSTPROCESS_FILENAME
    legacy.parent.mkdir(parents=True)
    legacy.write_text(yaml.safe_dump({"generate_stl_from_mask": {"marker": "legacy"}}), encoding="utf-8")
    canonical = tmp_path / "deployment" / "postprocessing.yaml"
    monkeypatch.setenv(paths.ENV_POSTPROCESS_SETTINGS, str(canonical))

    # Simulate the interleaving directly: the file appears after the existence check, before the
    # write. _install_if_absent must refuse rather than replace.
    real_read = Path.read_text

    def racing_read(self, *a, **kw):
        if self == legacy and not canonical.exists():
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_text(
                yaml.safe_dump({"generate_stl_from_mask": {"marker": "user-saved"}}), encoding="utf-8")
        return real_read(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", racing_read)
    assert paths.migrate_legacy_postprocessing_settings(source=legacy) is None
    monkeypatch.undo()

    assert yaml.safe_load(canonical.read_text(encoding="utf-8")) == {
        "generate_stl_from_mask": {"marker": "user-saved"}}, "the adopt clobbered a user save"


def test_the_hardware_profile_adopt_has_the_same_no_clobber_guarantee(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same check-then-write shape, same fix -- pinned so only one of the two can regress."""
    settings = tmp_path / "checkout" / "LM3_settings.yaml"
    settings.parent.mkdir(parents=True)
    settings.write_text("project: {}\n", encoding="utf-8")
    (settings.parent / paths.LEGACY_HARDWARE_FILENAME).write_text("marker: legacy\n", encoding="utf-8")

    target = tmp_path / "hw.yaml"
    monkeypatch.setenv(paths.ENV_HARDWARE, str(target))
    target.write_text("marker: mine\n", encoding="utf-8")

    assert paths.migrate_legacy_hardware_profile(settings_file=settings) is None
    assert target.read_text(encoding="utf-8") == "marker: mine\n"


def test_an_adopt_skips_loudly_when_hardlinks_are_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No half-written destination on a filesystem without hardlinks.

    ``O_CREAT|O_EXCL`` would settle the create decision race-free, but the file becomes visible at
    creation, so a concurrent reader could parse truncated YAML as the deployment's settings.
    Migration is best-effort and the caller degrades to packaged defaults, so skipping is safer than
    publishing a partial file.
    """
    legacy = tmp_path / "checkout" / paths.LEGACY_POSTPROCESS_FILENAME
    legacy.parent.mkdir(parents=True)
    legacy.write_text(yaml.safe_dump({"generate_stl_from_mask": {"marker": "legacy"}}), encoding="utf-8")
    canonical = tmp_path / "deployment" / "postprocessing.yaml"
    monkeypatch.setenv(paths.ENV_POSTPROCESS_SETTINGS, str(canonical))

    def no_hardlinks(*a, **kw):
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(paths.os, "link", no_hardlinks)

    # NOT caplog: leafmachine3.core.logging_setup sets root.propagate = False, so once any earlier
    # test has started a pipeline caplog captures nothing and this assertion would be order-dependent.
    with capture_lm3_logs() as messages:
        assert paths.migrate_legacy_postprocessing_settings(source=legacy) is None

    assert not canonical.exists(), "a partial destination was published"
    assert not list(canonical.parent.glob(".*tmp")) if canonical.parent.exists() else True
    assert any("could not atomically install" in m for m in messages), (
        "the skip must be loud; a silent no-op looks like a successful migration")


def test_the_hardware_resolver_exposes_no_migration_parameter() -> None:
    """One migration path, not two. A disabled-by-default adopt still exposes the race."""
    import inspect

    params = inspect.signature(paths.hardware_profile_path).parameters
    assert "adopt_legacy" not in params
    assert "settings_file" not in params
    source = Path("leafmachine3/core/paths.py").read_text(encoding="utf-8")
    body_start = source.index("def hardware_profile_path(")
    body = source[body_start:source.index("def migrate_legacy_hardware_profile(", body_start)]
    assert "_write_atomic" not in body and "_install_if_absent" not in body, (
        "hardware_profile_path must not write; migration belongs to migrate_legacy_hardware_profile")
