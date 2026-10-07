"""Positive tests for Step 1: every server module resolves the SAME canonical paths.

The characterization suite next door records what the split used to look like. This module asserts
what replaced it -- the normative §3.1 precedence table and §4 Step 1's exit gate, "from any
supported CWD every subsystem reports the same canonical settings path; conflicting legacy
variables stop startup".

Three properties are proven here, each of which the pre-refactor tree failed:

1. **Agreement.** Six accessors across six modules -- ``settings_api.settings_path``,
   ``metrics_api.default_config_path``, ``progress_api._settings_path``,
   ``postprocess_api._lm3_settings_path``, ``results_api._settings_file`` and
   ``app.canonical_settings_path`` -- return one path, and the same is true of the hardware
   profile and the jobs root.
2. **CWD independence.** The same environment answers identically from two unrelated working
   directories, each seeded with a decoy file of the right name (§3.1: "No path falls back to the
   current working directory").
3. **The one-release legacy migration** (§4 Step 1): ``LM3_SETTINGS_PATH`` alone is honored with a
   deprecation warning; agreeing with ``LM3_SETTINGS`` warns ONCE PER PROCESS and is used;
   disagreeing stops server startup with an error naming both paths. Same for
   ``LM3_HARDWARE_SETTINGS`` vs ``LM3_HARDWARE``.

Nothing here binds a port or starts a run: ``create_app`` is exercised through
``fastapi.testclient.TestClient`` in-process, and the only endpoint touched is ``/healthz``.
"""
from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

import pytest
import yaml

from leafmachine3.core import paths

#: Every variable that could otherwise decide an answer from a developer's shell. ``LM3_RUNTIME_DIR``
#: and ``LM3_DEPLOYMENT_ID`` are NOT cleared: ``tests/conftest.py`` sets them at import time to keep
#: the suite out of the real deployment.
_PATH_ENV_VARS = (
    "LM3_SETTINGS", "LM3_SETTINGS_PATH", "LM3_HARDWARE", "LM3_HARDWARE_SETTINGS",
    "LM3_POSTPROCESS_SETTINGS", "LM3_SERVER_JOBS", "LM3_RUNS_ROOTS", "LM3_STATUS_ROOTS",
    "LM3_POSTPROCESS_ROOTS",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from leafmachine3.server import app as server_app

    for name in _PATH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Per-test <user-config> / <user-state>: a legacy adopt or a seeded settings file must land in
    # tmp_path, never in the developer's ~/.config/lm3.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    # The deprecation warnings are memoized per process (that IS the "warn once" mechanism), so the
    # memo has to be dropped between tests or the second test to use a given environment sees none.
    server_app._LEGACY_CACHE.clear()
    server_app._LEGACY_CONFLICTS.clear()


@contextlib.contextmanager
def collect_records(logger_name: str, level: int = logging.WARNING):
    """Collect one logger's records by attaching to it directly.

    Deliberately not ``caplog``: ``leafmachine3.core.logging_setup.start_logging`` sets
    ``propagate = False`` on the ``leafmachine3`` logger, so once any pipeline test has run in the
    same session, records never reach the root handler ``caplog`` installs and the assertion
    silently inverts. Attaching to the logger itself is immune to that ordering.
    """
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger(logger_name)
    handler = _Collector(level=level)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def write_settings(path: Path, run_name: str = "unified", output_dir: str = "runs") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"version": 3,
                        "project": {"run_name": run_name, "output": {"dir": output_dir}}},
                       sort_keys=False),
        encoding="utf-8")
    return path


def settings_answers() -> dict[str, Path | None]:
    """What each module thinks the next-run settings file is, keyed by module name."""
    from leafmachine3.server import (
        app, metrics_api, postprocess_api, progress_api, results_api, settings_api,
    )

    return {
        "app": app.canonical_settings_path(),
        "settings_api": settings_api.settings_path(),
        "metrics_api": metrics_api.default_config_path(),
        "progress_api": progress_api._settings_path(),
        "postprocess_api": postprocess_api._lm3_settings_path(),
        "results_api": results_api._settings_file(),
    }


# --------------------------------------------------------------------------- #
# Agreement, and independence from the launch directory
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("launch_from", ["one", "two"])
def test_every_module_resolves_the_same_settings_file_from_any_cwd(
    launch_from: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4 Step 1 exit gate: "from any supported CWD every subsystem reports the same canonical
    settings path"."""
    chosen = write_settings(tmp_path / "chosen" / "LM3_settings.yaml")
    work = tmp_path / launch_from
    work.mkdir()
    write_settings(work / "LM3_settings.yaml", run_name="decoy")     # the old CWD fallback's prize
    monkeypatch.chdir(work)
    monkeypatch.setenv("LM3_SETTINGS", str(chosen))

    answers = settings_answers()

    assert set(answers.values()) == {chosen}, answers


def test_every_module_agrees_with_no_environment_variable_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.1 row 1 step 4: the deployment's own file, from every module, with nothing exported.

    This is the case §4 Step 1 says a rename would not fix -- "three fallbacks reachable from
    three CWDs". The decoy in the CWD is what each of those fallbacks used to return.
    """
    deployment = paths.deployment_config_dir() / paths.SETTINGS_FILENAME
    write_settings(deployment, run_name="deployment_copy")
    work = tmp_path / "elsewhere"
    work.mkdir()
    write_settings(work / "LM3_settings.yaml", run_name="decoy")
    monkeypatch.chdir(work)

    answers = settings_answers()

    assert set(answers.values()) == {deployment}, answers


def test_every_module_resolves_the_same_hardware_profile(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """§3.1 row 2: one deployment-scoped, machine-keyed profile -- server side and setup side.

    ``hardware_setup`` is included on purpose: the GUI tuning panel and the profiler that writes
    the file have to name the same path, or the panel reports "run LM3_Setup" forever.
    """
    from leafmachine3.server import app, metrics_api
    from leafmachine3.setup import hardware_setup

    # A settings file with NO profile beside it, so the one-release legacy adopt (covered by its
    # own test below) does not fill the deployment path in before the miss is observed.
    monkeypatch.setenv("LM3_SETTINGS", str(write_settings(tmp_path / "cfg" / "LM3_settings.yaml")))
    canonical = app.canonical_hardware_path()

    assert canonical == hardware_setup.hardware_profile_path()
    assert canonical == paths.hardware_profile_path()
    assert canonical.is_absolute()
    assert metrics_api.hardware_path() is None            # nothing written yet -> honest miss
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text(yaml.safe_dump({"marker": "profile"}), encoding="utf-8")
    assert metrics_api.hardware_path() == canonical
    assert app.read_hardware_settings() == {"marker": "profile"}


def test_every_module_resolves_the_same_jobs_root(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """§3.1 row 4. WAS: one variable, four readers, two absolutization rules -- so the server could
    create job dirs under the CWD while writing the record describing them under the checkout."""
    from leafmachine3.server import app, metrics_api, progress_api, results_api

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setenv("LM3_SERVER_JOBS", str(jobs))

    assert app.server_jobs_root() == jobs
    assert metrics_api._state_path() == jobs / "active_run.json"
    assert progress_api._jobs_root() == jobs
    assert results_api._jobs_root() == jobs.resolve()
    assert app.JobManager().root == jobs


def test_the_jobs_root_defaults_to_user_state_not_a_relative_runs_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.1 row 4 step 2: ``<user-state>/lm3/<deployment>/jobs``.

    WAS: ``Path(os.environ.get("LM3_SERVER_JOBS", "runs/_server_jobs"))``, evaluated at IMPORT
    time and left relative -- so merely constructing the app created ``runs/_server_jobs`` in
    whatever directory the caller happened to be in.
    """
    from leafmachine3.server import app

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    root = app.server_jobs_root()

    assert root.is_absolute()
    assert root == paths.deployment_state_dir() / "jobs"
    assert not (work / "runs").exists()


def test_postprocessing_settings_resolve_to_the_deployment_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.1 row 3. WAS: ``postprocessing_settings.yaml`` off the server's CWD."""
    from leafmachine3.server import app, postprocess_api

    work = tmp_path / "work"
    work.mkdir()
    (work / "postprocessing_settings.yaml").write_text("generate_stl: {}\n", encoding="utf-8")
    monkeypatch.chdir(work)

    resolved = postprocess_api._settings_path()

    assert resolved == app.canonical_postprocess_settings_path()
    assert resolved == paths.deployment_config_dir() / paths.POSTPROCESS_FILENAME
    assert resolved != work / "postprocessing_settings.yaml"


def test_run_history_roots_no_longer_include_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.1 row 5 + the governing rule. WAS: ``cwd``, ``cwd/runs`` and ``cwd/examples_out`` were
    pushed unconditionally, and with ``LM3_RUNS_ROOTS`` unset they were the ONLY roots -- so the
    Results tab listed whatever happened to sit beside the launcher."""
    from leafmachine3.server import results_api

    work = tmp_path / "work"
    (work / "runs").mkdir(parents=True)
    (work / "examples_out").mkdir()
    monkeypatch.chdir(work)
    chosen = write_settings(tmp_path / "cfg" / "LM3_settings.yaml", output_dir="out")
    (chosen.parent / "out").mkdir()
    monkeypatch.setenv("LM3_SETTINGS", str(chosen))
    results_api._RUNS_CACHE.invalidate()

    roots = results_api.run_roots()

    assert (chosen.parent / "out").resolve() in roots
    assert work.resolve() not in roots
    assert (work / "runs").resolve() not in roots
    assert (work / "examples_out").resolve() not in roots


def test_the_postprocess_sandbox_no_longer_authorizes_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.1's governing rule applied to a WRITE sandbox.

    ``allowed_roots()`` seeded itself with ``Path.cwd()``, which authorized whatever tree the
    server was started in. The settings file's own directory replaces it -- the directory the
    project is actually described from.
    """
    from leafmachine3.server import postprocess_api

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    chosen = write_settings(tmp_path / "cfg" / "LM3_settings.yaml")
    monkeypatch.setenv("LM3_SETTINGS", str(chosen))

    roots = postprocess_api.allowed_roots()

    assert chosen.parent.resolve() in roots
    assert work.resolve() not in roots
    # A relative request path lands under the settings file, not under the launch directory --
    # the same rule project.output.dir follows.
    assert postprocess_api.resolve_path("nested/out.stl", must_exist=False) == \
        chosen.parent.resolve() / "nested" / "out.stl"
    with pytest.raises(postprocess_api.ParamError):
        postprocess_api.resolve_path(str(work / "escape.stl"), must_exist=False)


def test_a_profile_beside_the_settings_file_is_adopted_by_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.1 row 2 step 3: "legacy adopt (one release, with a warning): **copy** a
    beside-the-settings profile into path 2 and use the copy".

    Copy, not move: an older LM3 launched from that checkout must keep working, and the tuning the
    user already paid for must not be thrown away by an upgrade.
    """
    from leafmachine3.server import app, metrics_api

    chosen = write_settings(tmp_path / "checkout" / "LM3_settings.yaml")
    legacy = chosen.parent / "hardware_settings.yaml"
    legacy.write_text(yaml.safe_dump({"marker": "legacy"}), encoding="utf-8")
    monkeypatch.setenv("LM3_SETTINGS", str(chosen))

    # Resolving must NOT adopt: canonical_hardware_path() is on the 2 Hz status path.
    app.reset_path_caches()
    resolved = app.canonical_hardware_path()
    assert not resolved.exists(), "a read path copied a file"

    adopted = paths.migrate_legacy_hardware_profile(
        env=app.server_env(), settings_file=chosen)

    assert adopted == resolved
    assert adopted == paths.deployment_config_dir() / adopted.name
    assert yaml.safe_load(adopted.read_text(encoding="utf-8")) == {"marker": "legacy"}
    assert legacy.is_file(), "adoption COPIES; the old file stays put for an older LM3"
    assert metrics_api.hardware_path() == adopted


# --------------------------------------------------------------------------- #
# The one-release legacy migration (§4 Step 1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("canonical", "legacy", "filename"),
    [("LM3_SETTINGS", "LM3_SETTINGS_PATH", "LM3_settings.yaml"),
     ("LM3_HARDWARE", "LM3_HARDWARE_SETTINGS", "hardware_settings.yaml")],
)
def test_a_legacy_variable_alone_is_honored_with_a_deprecation_warning(
    canonical: str, legacy: str, filename: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4 Step 1: "honor ``LM3_SETTINGS_PATH`` alone with a deprecation warning"."""
    from leafmachine3.server import app

    target = write_settings(tmp_path / "legacy" / filename)
    monkeypatch.setenv(legacy, str(target))

    with collect_records("leafmachine3.paths") as records:
        folded = app.server_env()

    assert folded[canonical] == str(target)
    assert legacy not in folded
    assert any(legacy in r.getMessage() and "deprecated" in r.getMessage() for r in records)


@pytest.mark.parametrize(
    ("canonical", "legacy", "filename"),
    [("LM3_SETTINGS", "LM3_SETTINGS_PATH", "LM3_settings.yaml"),
     ("LM3_HARDWARE", "LM3_HARDWARE_SETTINGS", "hardware_settings.yaml")],
)
def test_agreeing_variables_are_used_and_warn_exactly_once_per_process(
    canonical: str, legacy: str, filename: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4 Step 1: "if it and ``LM3_SETTINGS`` resolve to the same file, use it and warn once".

    "Once" means once per process, not once per request: the resolution below runs twenty times,
    which is what a handful of status frames costs, and the log must still carry one line.
    """
    from leafmachine3.server import app

    target = write_settings(tmp_path / "same" / filename)
    monkeypatch.setenv(canonical, str(target))
    monkeypatch.setenv(legacy, str(target))

    with collect_records("leafmachine3.paths") as records:
        for _ in range(20):
            folded = app.server_env()

    assert folded[canonical] == str(target)
    assert legacy not in folded
    deprecations = [r for r in records if legacy in r.getMessage()]
    assert len(deprecations) == 1, [r.getMessage() for r in deprecations]


@pytest.mark.parametrize(
    ("canonical", "legacy", "filename"),
    [("LM3_SETTINGS", "LM3_SETTINGS_PATH", "LM3_settings.yaml"),
     ("LM3_HARDWARE", "LM3_HARDWARE_SETTINGS", "hardware_settings.yaml")],
)
def test_disagreeing_variables_are_a_startup_error_naming_both_paths(
    canonical: str, legacy: str, filename: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4 Step 1: "if they resolve differently, fail server startup with a precise split-brain
    error". The message has to name BOTH files, or the operator cannot tell which to delete."""
    from leafmachine3.server import app

    mine = write_settings(tmp_path / "mine" / filename)
    theirs = write_settings(tmp_path / "theirs" / filename)
    monkeypatch.setenv(canonical, str(mine))
    monkeypatch.setenv(legacy, str(theirs))

    with pytest.raises(paths.LegacyEnvConflictError) as first:
        app.check_legacy_env()

    message = str(first.value)
    assert str(mine) in message and str(theirs) in message
    assert canonical in message and legacy in message

    # It must keep failing. The memo that makes the WARNING fire once must not make the ERROR fire
    # once and then let the second caller through with a silently chosen winner.
    with pytest.raises(paths.LegacyEnvConflictError):
        app.check_legacy_env()


def test_a_split_brain_environment_stops_server_startup(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """§4 Step 1 exit gate: "conflicting legacy variables stop startup".

    ``create_app`` mounts each router inside a ``try/except Exception`` that logs and continues, so
    the check has to run BEFORE that loop -- otherwise a conflict produces a silently half-mounted
    API instead of a refusal.
    """
    pytest.importorskip("fastapi")
    from leafmachine3.server import app

    monkeypatch.setenv("LM3_SETTINGS", str(write_settings(tmp_path / "a" / "LM3_settings.yaml")))
    monkeypatch.setenv("LM3_SETTINGS_PATH",
                       str(write_settings(tmp_path / "b" / "LM3_settings.yaml")))

    with pytest.raises(paths.LegacyEnvConflictError):
        app.create_app()


# --------------------------------------------------------------------------- #
# Startup logs, /healthz diagnostics, and the explicit `lm3 serve` pin
# --------------------------------------------------------------------------- #
def test_healthz_reports_the_resolved_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """§4 Step 1: "Expose resolved paths in startup logs and ``/healthz`` diagnostics".

    This is what makes the exit gate checkable from OUTSIDE the process: start the server from
    anywhere, ask ``/healthz``, and the answer names the file it is really on.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from leafmachine3.server import app

    chosen = write_settings(tmp_path / "chosen" / "LM3_settings.yaml")
    monkeypatch.setenv("LM3_SETTINGS", str(chosen))
    monkeypatch.setenv("LM3_SERVER_JOBS", str(tmp_path / "jobs"))
    monkeypatch.setenv("LM3_SERVER_TOKEN", "unification-test-token")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    with TestClient(app.create_app()) as client:
        body = client.get("/healthz").json()

    resolved = body["paths"]
    assert resolved["settings"] == str(chosen)
    assert resolved["server_jobs_root"] == str(tmp_path / "jobs")
    assert resolved["hardware_profile"] == str(app.canonical_hardware_path())
    assert resolved["deployment_key"] == paths.deployment_key()
    # Diagnostics only -- §4 Step 5b owns the shape of the rest of this body.
    assert body["status"] == "ok" and body["pid"] > 0


def test_startup_logs_name_every_resolved_path(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """§4 Step 1's last bullet, log half: a support log answers "which config?" without a repro."""
    from leafmachine3.server import app

    chosen = write_settings(tmp_path / "chosen" / "LM3_settings.yaml")
    monkeypatch.setenv("LM3_SETTINGS", str(chosen))

    with collect_records("leafmachine3.server", logging.INFO) as records:
        app.log_resolved_paths("test")

    line = "\n".join(r.getMessage() for r in records)
    assert str(chosen) in line
    assert "deployment_key=" in line and "hardware_profile=" in line


def test_lm3_serve_pins_the_canonical_settings_path_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.1: "``lm3 serve`` and Electron always pass the canonical settings path explicitly rather
    than relying on any fallback".

    Pinning it into the environment is what makes the guarantee reach the ROUTERS and the spawned
    child, not just the process that parsed the flag.
    """
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    from leafmachine3.server import app

    wanted = write_settings(tmp_path / "wanted" / "LM3_settings.yaml", run_name="wanted")
    ignored = write_settings(tmp_path / "ignored" / "LM3_settings.yaml", run_name="ignored")
    # setenv (not a raw assignment) so pytest restores whatever serve() writes over it.
    monkeypatch.setenv("LM3_SETTINGS", str(ignored))
    monkeypatch.setenv("LM3_SETTINGS_PATH", str(ignored))
    monkeypatch.setenv("LM3_SERVER_JOBS", str(tmp_path / "jobs"))
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)

    app.serve(jobs_root=tmp_path / "jobs", settings=wanted)

    assert os.environ["LM3_SETTINGS"] == str(wanted)
    assert "LM3_SETTINGS_PATH" not in os.environ, "the deprecated alias is folded in, not left to rot"
    assert set(settings_answers().values()) == {wanted}


def test_create_app_seeds_a_first_run_settings_file(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """§3.1 precedence table row 1, on-miss column: seed 4 from the packaged template and continue.

    ``lm3 serve`` is NOT the only server entry point -- the Electron shell spawns
    ``uvicorn leafmachine3.server.app:create_app --factory`` -- so leaving the seed to ``serve()``
    means a first-run GUI install meets ``metrics_api``'s 400 "no LM3_settings.yaml found" instead
    of the seeded file the plan promises ("so a first-run install works").

    ``dev_checkout_root`` is stubbed to ``None`` so row 5 (this repo's own ``LM3_settings.yaml``)
    cannot mask the miss the way it would for a developer running the suite in a checkout.
    """
    pytest.importorskip("fastapi")
    from leafmachine3.server import app, metrics_api

    monkeypatch.setattr(paths, "dev_checkout_root", lambda *a, **k: None)
    monkeypatch.setenv("LM3_SERVER_JOBS", str(tmp_path / "jobs"))
    expected = paths.deployment_config_dir() / paths.SETTINGS_FILENAME
    assert not expected.exists(), "the clean_env fixture must hand this test an empty <user-config>"

    app.create_app()

    assert expected.is_file(), "create_app must seed row 4 from the packaged template"
    assert metrics_api.default_config_path() == expected
