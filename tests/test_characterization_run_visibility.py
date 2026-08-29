"""Characterization tests for PRE-refactor run visibility (consensus plan Step 1).

These pin PRE-REFACTOR behavior, including behavior the consensus plan classifies as a defect.
They are EXPECTED TO CHANGE in Step 1; when they fail, update them deliberately and record which
plan clause caused the change.

Two entangled behaviors are recorded here:

* **Configured-project precedence.** ``progress_api.resolve_run`` asks the settings file which
  project it is describing *before* it looks at what is actually running on disk, and returns the
  configured ``<project.output.dir>/<project.run_name>`` even when that directory has never
  existed. §2.1 / §3.1 replace this with a deployment-scoped activity record, and the §3.1
  workspace-pointer table is explicit that the settings selection "**Never** declares, implies, or
  influences the **active run**".
* **CLI invisibility.** ``metrics_api`` keeps the active run in a private process-local
  ``_RUN`` (Appendix A: "Server private run state", ``metrics_api.py:427-429``) that is written
  only when the server itself spawns the child. A ``machine3()`` run started from a shell is
  therefore reported as ``state: "idle"`` by ``/v1/run/active`` while it is pinning every GPU, and
  ``/v1/status`` describes the configured project instead. §1's invariants and §2.2's activity
  hierarchy exist to make exactly this impossible.

No server is bound to a real port anywhere in this module: the plain functions are called
directly, and the one endpoint-level assertion uses ``fastapi.testclient.TestClient`` against a
router built in-process.
"""
from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

import pytest
import yaml

# The ``characterization`` marker is not registered in a pytest ini section (this repo has none),
# and Step 1 owns pyproject.toml, so the marker is silenced locally instead of registered
# globally. Narrowed to this one mark name so a genuine typo in another module still warns.
warnings.filterwarnings(
    "ignore",
    message=r".*Unknown pytest\.mark\.characterization.*",
    category=pytest.PytestUnknownMarkWarning,
)

pytestmark = pytest.mark.characterization

_ENV_VARS = (
    "LM3_SETTINGS_PATH",
    "LM3_SETTINGS",
    "LM3_HARDWARE_SETTINGS",
    "LM3_HARDWARE",
    "LM3_SERVER_JOBS",
    "LM3_STATUS_ROOTS",
)


@pytest.fixture(autouse=True)
def _isolated_server_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Neutralize every ambient input to the two modules under test, then restore module state.

    ``progress_api`` and ``metrics_api`` both hold PROCESS-GLOBAL run state -- the whole point of
    the CLI-invisibility defect -- so a test that sets it would leak into the next one. The state
    is reset on the way in and on the way out.
    """
    from leafmachine3.server import metrics_api, progress_api

    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # A CWD with nothing in it, so no relative fallback in either module finds a real file.
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    # Keep the server-jobs root (the adoption record's home, metrics_api._state_path) inside tmp.
    jobs = tmp_path / "server_jobs"
    jobs.mkdir()
    monkeypatch.setenv("LM3_SERVER_JOBS", str(jobs))

    def reset() -> None:
        progress_api.clear_run()
        progress_api.bind_job_source(None)
        progress_api._SNAPSHOT_CACHE.invalidate()
        progress_api._SCAN_CACHE.invalidate()
        progress_api._RUNS_CACHE.invalidate()
        progress_api._YAML_CACHE.clear()

    reset()
    monkeypatch.setattr(metrics_api, "_RUN", None, raising=False)
    monkeypatch.setattr(metrics_api, "_ADOPT_TRIED", False, raising=False)
    yield
    reset()


def _write_settings(path: Path, *, run_name: str, output_dir: str) -> Path:
    """A minimal but real settings document -- only the two keys resolution actually reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"project": {"run_name": run_name, "output": {"dir": output_dir}}},
                       sort_keys=False),
        encoding="utf-8",
    )
    return path


def _make_cli_run(output_dir: Path, name: str) -> Path:
    """A run directory as ``core.dirs`` lays one out: ``<root>/<name>/<name>.sqlite``.

    This is exactly what a ``machine3()`` invocation from a shell leaves behind, and it is all
    ``progress_api._is_run_dir`` inspects, so it is a faithful stand-in for a live CLI run.
    """
    root = output_dir / name
    (root / "logs").mkdir(parents=True, exist_ok=True)
    sqlite3.connect(root / f"{name}.sqlite").close()
    return root


# --------------------------------------------------------------------------- #
# Configured-project precedence
# --------------------------------------------------------------------------- #
def test_the_configured_project_wins_over_a_run_that_actually_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """YAML says ``configured_project``; the disk says ``cli_run``. Today YAML wins.

    The configured directory has never been created, and it is still preferred over a real
    ledger sitting in the very same output root.
    """
    from leafmachine3.server import progress_api

    output_dir = tmp_path / "runs"
    cli_run = _make_cli_run(output_dir, "cli_run")
    monkeypatch.setenv(
        "LM3_SETTINGS",
        str(_write_settings(tmp_path / "LM3_settings.yaml",
                            run_name="configured_project", output_dir=str(output_dir))),
    )

    ref = progress_api.resolve_run()

    assert ref is not None
    assert ref.run_name == "configured_project"
    assert ref.source == "settings"
    assert not ref.root.exists()
    assert cli_run.is_dir()          # the run that IS on disk was never considered


def test_a_relative_configured_output_dir_hangs_off_the_settings_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative ``project.output.dir`` is resolved against the settings file, not the CWD.

    One of the few places in the pre-refactor code that already refuses the CWD -- worth pinning
    so Step 1 does not lose it while removing the CWD fallbacks around it.
    """
    from leafmachine3.server import progress_api

    settings = _write_settings(tmp_path / "cfgdir" / "LM3_settings.yaml",
                               run_name="configured_project", output_dir="runs")
    monkeypatch.setenv("LM3_SETTINGS", str(settings))

    ref = progress_api.resolve_run()

    assert ref is not None
    assert ref.root == settings.parent / "runs" / "configured_project"


def test_settings_precedence_is_skipped_when_the_project_is_unnamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no configured project name, discovery gets its turn and DOES see the CLI run.

    ``_from_settings``'s docstring calls this out as "what keeps a run started outside the app
    (from the machine3 CLI) findable" -- i.e. today a CLI run is visible in the status API only
    when the settings file names no project at all.
    """
    from leafmachine3.server import progress_api

    output_dir = tmp_path / "runs"
    cli_run = _make_cli_run(output_dir, "cli_run")
    monkeypatch.setenv("LM3_SETTINGS",
                       str(_write_settings(tmp_path / "LM3_settings.yaml",
                                           run_name="", output_dir=str(output_dir))))
    monkeypatch.setenv("LM3_STATUS_ROOTS", str(output_dir))

    ref = progress_api.resolve_run()

    assert ref is not None
    assert ref.root == cli_run
    assert ref.source == "discovered"


def test_a_bound_run_beats_the_configured_project(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """``bind_run`` is the hook the SERVER gets and the CLI does not (``metrics_api._bind_status``).

    This is the whole asymmetry in one test: the identical run directory is authoritative when
    the server announces it and invisible when nobody does.
    """
    from leafmachine3.server import progress_api

    output_dir = tmp_path / "runs"
    cli_run = _make_cli_run(output_dir, "cli_run")
    monkeypatch.setenv("LM3_SETTINGS",
                       str(_write_settings(tmp_path / "LM3_settings.yaml",
                                           run_name="configured_project",
                                           output_dir=str(output_dir))))

    progress_api.bind_run(root=cli_run, run_name="cli_run")
    ref = progress_api.resolve_run()

    assert ref is not None
    assert ref.root == cli_run
    assert ref.source == "bound"


def test_a_bound_run_whose_directory_is_gone_falls_back_to_the_configured_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from leafmachine3.server import progress_api

    output_dir = tmp_path / "runs"
    output_dir.mkdir()
    monkeypatch.setenv("LM3_SETTINGS",
                       str(_write_settings(tmp_path / "LM3_settings.yaml",
                                           run_name="configured_project",
                                           output_dir=str(output_dir))))

    progress_api.bind_run(root=tmp_path / "never_created", run_name="never_created")
    ref = progress_api.resolve_run()

    assert ref is not None
    assert ref.run_name == "configured_project"
    assert ref.source == "settings"


def test_the_job_source_beats_a_bound_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full pre-refactor chain: query > job source > bound > settings > discovered."""
    from leafmachine3.server import progress_api

    output_dir = tmp_path / "runs"
    job_run = _make_cli_run(output_dir, "job_run")
    bound_run = _make_cli_run(output_dir, "bound_run")
    monkeypatch.setenv("LM3_SETTINGS",
                       str(_write_settings(tmp_path / "LM3_settings.yaml",
                                           run_name="configured_project",
                                           output_dir=str(output_dir))))

    progress_api.bind_run(root=bound_run, run_name="bound_run")
    progress_api.bind_job_source(lambda: {"root": str(job_run), "job_id": "j1", "state": "running"})

    ref = progress_api.resolve_run()
    assert ref is not None
    assert ref.root == job_run
    assert ref.source == "job"
    assert ref.job_id == "j1"

    # ... and an explicit query beats even that.
    queried = progress_api.resolve_run(db=str(bound_run / "bound_run.sqlite"))
    assert queried is not None
    assert queried.root == bound_run
    assert queried.source == "query"


# --------------------------------------------------------------------------- #
# CLI invisibility
# --------------------------------------------------------------------------- #
def test_a_cli_run_is_invisible_to_the_active_run_record(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """``metrics_api.active()`` reports "idle" while a run started outside the server is going.

    Nothing about this depends on the run being finished: the record is process-local server
    bookkeeping, and a CLI run never writes it.
    """
    from leafmachine3.server import metrics_api

    _make_cli_run(tmp_path / "runs", "cli_run")

    record = metrics_api.active()

    assert record["active"] is False
    assert record["state"] == "idle"
    assert record["run_name"] is None
    assert record["pid"] is None
    assert record["run_dir"] is None
    assert metrics_api.is_active() is False
    assert metrics_api.active_db_path() is None
    assert metrics_api.active_log_path() is None


def test_the_adoption_record_is_the_only_way_a_run_becomes_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adoption reads ``<LM3_SERVER_JOBS>/active_run.json``, which only the server ever writes.

    Pinned because §2.2's shared lease replaces this private file, and the replacement has to keep
    working for a server restart that re-attaches to its own child.
    """
    from leafmachine3.server import metrics_api

    state = metrics_api._state_path()

    assert state.parent == Path(str(tmp_path / "server_jobs"))
    assert state.name == "active_run.json"
    assert not state.exists()          # a CLI run leaves nothing here, so adoption finds nothing


def test_the_status_snapshot_describes_the_configured_project_not_the_cli_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/v1/status``'s payload, built in-process: the live CLI run does not appear in it."""
    from leafmachine3.server import progress_api

    output_dir = tmp_path / "runs"
    _make_cli_run(output_dir, "cli_run")
    monkeypatch.setenv("LM3_SETTINGS",
                       str(_write_settings(tmp_path / "LM3_settings.yaml",
                                           run_name="configured_project",
                                           output_dir=str(output_dir))))

    snapshot = progress_api.status()

    assert snapshot["run_name"] == "configured_project"
    assert snapshot["source"] == "settings"
    assert snapshot["ready"] is False


def test_v1_run_active_reports_idle_over_http_while_a_cli_run_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same invisibility at the endpoint boundary, through the real router.

    ``TestClient`` speaks ASGI in-process -- no socket is bound and no port is taken.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from leafmachine3.server import metrics_api

    _make_cli_run(tmp_path / "runs", "cli_run")

    app = FastAPI()
    app.include_router(metrics_api.router())          # no auth guard: this router is not mounted
    with TestClient(app) as client:                   # on a real app here
        payload = client.get("/v1/run/active").json()

    assert payload["active"] is False
    assert payload["state"] == "idle"
    assert payload["run_name"] is None


def test_the_active_record_shape_is_invariant_when_idle() -> None:
    """Every key the populated record carries is present (and null) when idle.

    The front end tests values, never key existence, so Step 1's replacement record must keep the
    key set -- this pins it.
    """
    from leafmachine3.server import metrics_api

    record = metrics_api.active()

    assert set(record) == set(metrics_api._IDLE_RECORD)
    assert record["input_dirs"] == [] and record["restart"] == [] and record["argv"] == []
    # Fresh containers, not aliases of the template (a caller must not be able to mutate it).
    assert record["input_dirs"] is not metrics_api._IDLE_RECORD["input_dirs"]
