"""Step 4: status, logs and postprocessing follow the RUNTIME REGISTRY, not a mutable guess.

Plan clauses under test:

* **invariant 7** -- "In follow-active mode, which is the default, status, logs, results, and
  postprocessing targets follow the active record's paths. An explicit ``run=``/``db=`` selector
  deliberately overrides this for that client request."
* **Step 4** -- ``resolve_run()``'s precedence becomes, exactly: explicit ``db=``/``run=``; the
  active runtime record; an explicitly selected historical run; the canonical next-run project for
  an idle prepared project; filesystem discovery last, labeled ``source: discovered`` and never
  able to authorize control. Plus: ``run_id`` in the snapshot cache key, follow the active
  ``run_id`` for logs, follow ``last.json`` on completion, and delete the duplicate
  progress-router ``GET /v1/runs``.
* **section 2.6** -- a ``hardware_setup`` root must NOT switch project history to
  ``_lm3_calibration``.
* **section 2.8** -- postprocessing refuses a target that is the active pipeline's run directory,
  allows a different completed run, and serializes two ``read_write`` tools on one completed run
  through a per-run advisory lock in the LOCAL deployment runtime registry, keyed by a hash of the
  resolved ``artifact_dir`` -- never a lock file inside the output directory.
* **Step 4 exit gate** -- changing ``project.run_name`` during an active run does not move status,
  logs, results, or postprocessing away from the active DB. That is
  ``TestTheExitGate``, and it is the reason the rest of this file exists.

Everything is exercised in-process against real artifacts: a real POSIX lease (so the section 2.9
reader classifies the record ``live`` off the OS lock rather than off the JSON), real
``active.json`` / ``last.json`` written through :class:`RecordStore`, real SQLite ledgers, and a
real advisory lock. No server is bound to a port and no pipeline is started.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from leafmachine3.core.runtime import (
    Activity,
    ActivityRole,
    ArchiveMode,
    ArchiveStatus,
    ConfigRef,
    HardwareBlock,
    Launcher,
    ProjectBlock,
    RunState,
    RuntimeRecord,
)
from leafmachine3.core.runtime import lease as lease_mod
from leafmachine3.core.runtime import records as records_mod
from leafmachine3.server import postprocess_api, progress_api

FLAG = "LM3_RUNTIME_V2"


# --------------------------------------------------------------------------- #
# Fixtures and builders
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A private deployment: its own runtime dir, jobs root, settings file and empty CWD.

    ``LM3_RUNTIME_V2`` is set explicitly on the way in rather than inherited: a developer with the
    flag exported would otherwise run this suite in the opposite mode from CI, and every assertion
    below would pass or fail for the wrong reason. ``tests/conftest.py`` does not isolate it yet.
    """
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    monkeypatch.setenv("LM3_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("LM3_DEPLOYMENT_ID", "test-step4")
    jobs = tmp_path / "server_jobs"
    jobs.mkdir()
    monkeypatch.setenv("LM3_SERVER_JOBS", str(jobs))
    monkeypatch.setenv("LM3_POSTPROCESS_ROOTS", str(tmp_path))
    monkeypatch.delenv("LM3_STATUS_ROOTS", raising=False)
    monkeypatch.setenv(FLAG, "1")
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    write_settings(tmp_path / "LM3_settings.yaml", run_name="", output_dir=str(tmp_path / "runs"))
    monkeypatch.setenv("LM3_SETTINGS", str(tmp_path / "LM3_settings.yaml"))

    reset_module_state()
    yield deployment_dir()
    reset_module_state()


def reset_module_state() -> None:
    progress_api.clear_run()
    progress_api.bind_job_source(None)
    progress_api.invalidate_run_caches()
    progress_api._YAML_CACHE.clear()
    postprocess_api._LOCAL_LOCKS.clear()


def deployment_dir() -> Path:
    from leafmachine3.core import paths as core_paths

    path = core_paths.deployment_runtime_dir(create=True, check_filesystem=False)
    return path


def write_settings(path: Path, *, run_name: str, output_dir: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"project": {"run_name": run_name, "output": {"dir": output_dir}}},
                       sort_keys=False),
        encoding="utf-8",
    )
    progress_api._YAML_CACHE.clear()
    progress_api.invalidate_run_caches()
    return path


def make_run(output_dir: Path, name: str) -> Path:
    """A run directory exactly as ``core.dirs`` lays one out: ``<root>/<name>/<name>.sqlite``."""
    root = output_dir / name
    (root / "logs").mkdir(parents=True, exist_ok=True)
    sqlite3.connect(root / f"{name}.sqlite").close()
    (root / "logs" / "lm3.log").write_text("12:00:00 INFO lm3: hello\n", encoding="utf-8")
    return root


def make_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "used_at_launch.yaml"
    cfg.write_text("version: 3\n", encoding="utf-8")
    return cfg


def pipeline_record(run_dir: Path, *, run_id: str, config_path: Path,
                    state: RunState = RunState.RUNNING) -> RuntimeRecord:
    """A valid in-place ``pipeline`` record for ``run_dir`` (plan section 3.2)."""
    now = records_mod.utc_now()
    db = run_dir / f"{run_dir.name}.sqlite"
    project = ProjectBlock(
        run_name=run_dir.name,
        input_dirs=(str(run_dir.parent),),
        artifact_dir=str(run_dir),
        active_state_dir=str(run_dir),
        active_db_path=str(db),
        archive_mode=ArchiveMode.IN_PLACE,
        archive_status=ArchiveStatus.NOT_APPLICABLE,
        archive_pointer_path=None,
        archived_db_path=str(db),
        run_dir=str(run_dir),
        log_path=str(run_dir / "logs" / "lm3.log"),
    )
    return RuntimeRecord(
        run_id=run_id,
        activity=Activity.PIPELINE,
        activity_role=ActivityRole.ROOT,
        state=state,
        launcher=Launcher.CLI,
        pid=os.getpid(),
        process_started_at=records_mod.process_start_time(),
        started_at=now,
        updated_at=now,
        finished_at=now if state in {RunState.DONE, RunState.ERROR} else None,
        deployment=records_mod.build_deployment_info(),
        config=ConfigRef(path=str(config_path), sha256="0" * 64),
        project=project,
    )


def hardware_record(destination: Path, *, run_id: str, config_path: Path) -> RuntimeRecord:
    """A ``hardware_setup`` root: config + hardware, and deliberately NO project (invariant 6)."""
    now = records_mod.utc_now()
    return RuntimeRecord(
        run_id=run_id,
        activity=Activity.HARDWARE_SETUP,
        activity_role=ActivityRole.ROOT,
        state=RunState.RUNNING,
        launcher=Launcher.CLI,
        pid=os.getpid(),
        process_started_at=records_mod.process_start_time(),
        started_at=now,
        updated_at=now,
        deployment=records_mod.build_deployment_info(),
        config=ConfigRef(path=str(config_path), sha256="0" * 64),
        hardware=HardwareBlock(destination_path=str(destination)),
    )


@contextlib.contextmanager
def holding_the_lease(record: RuntimeRecord) -> Iterator[records_mod.RecordStore]:
    """Publish ``record`` as ``active.json`` while genuinely holding the deployment lease.

    The lease is real, not a stub: :func:`records.read_runtime` decides ``active`` from the OS lock
    and NEVER from the JSON (section 2.9), so a record written without one classifies ``abandoned``
    and must not pin any view. Two tests below depend on exactly that distinction.
    """
    lease = lease_mod.RuntimeLease()
    lease.acquire()
    store = records_mod.RecordStore(deployment_dir(), run_id=record.run_id)
    try:
        store.write_active(record)
        progress_api.invalidate_run_caches()
        yield store
    finally:
        with contextlib.suppress(Exception):
            lease.release()
        progress_api.invalidate_run_caches()


def finish(store: records_mod.RecordStore, record: RuntimeRecord) -> None:
    """Move ``record`` to ``done`` and publish ``last.json`` the way section 3.3 orders it."""
    now = records_mod.utc_now()
    store.finalize(dataclasses.replace(record, state=RunState.DONE, finished_at=now, updated_at=now))
    progress_api.invalidate_run_caches()


# --------------------------------------------------------------------------- #
# Precedence (Step 4, invariant 7)
# --------------------------------------------------------------------------- #
class TestPrecedence:
    def test_the_active_record_outranks_the_configured_project(self, tmp_path: Path) -> None:
        """Tier 2 beats tier 4. The settings file names a project that EXISTS, and still loses."""
        runs = tmp_path / "runs"
        live = make_run(runs, "the_live_run")
        make_run(runs, "the_configured_one")
        write_settings(tmp_path / "LM3_settings.yaml",
                       run_name="the_configured_one", output_dir=str(runs))
        record = pipeline_record(live, run_id="rid-live", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            ref = progress_api.resolve_run()
            assert ref is not None
            assert ref.root == live
            assert ref.source == "runtime"
            assert ref.run_id == "rid-live"

    def test_the_active_record_outranks_an_explicit_selection(self, tmp_path: Path) -> None:
        """Tier 2 beats tier 3: a hook someone called for a finished run must not hide a live one."""
        runs = tmp_path / "runs"
        live = make_run(runs, "live")
        selected = make_run(runs, "selected_earlier")
        progress_api.bind_run(root=selected, run_name="selected_earlier")
        record = pipeline_record(live, run_id="rid-a", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            ref = progress_api.resolve_run()
            assert ref is not None and ref.root == live and ref.source == "runtime"

    def test_an_explicit_query_still_overrides_the_active_record(self, tmp_path: Path) -> None:
        """Tier 1. Invariant 7's deliberate override, scoped to one client request."""
        runs = tmp_path / "runs"
        live = make_run(runs, "live")
        other = make_run(runs, "an_old_run")
        record = pipeline_record(live, run_id="rid-b", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            ref = progress_api.resolve_run(db=str(other / "an_old_run.sqlite"))
            assert ref is not None and ref.root == other and ref.source == "query"
            named = progress_api.resolve_run(run="an_old_run")
            assert named is not None and named.root == other and named.source == "query"

    def test_an_explicit_selection_outranks_settings_and_discovery(self, tmp_path: Path) -> None:
        """Tier 3 beats tiers 4 and 5 when nothing holds the lease."""
        runs = tmp_path / "runs"
        configured = make_run(runs, "configured")
        selected = make_run(runs, "selected")
        write_settings(tmp_path / "LM3_settings.yaml", run_name="configured", output_dir=str(runs))
        progress_api.bind_run(root=selected, run_name="selected")
        ref = progress_api.resolve_run()
        assert ref is not None and ref.root == selected and ref.source == "bound"
        assert configured.is_dir()

    def test_the_prepared_project_is_tier_four_and_discovery_is_tier_five(
        self, tmp_path: Path
    ) -> None:
        runs = tmp_path / "runs"
        make_run(runs, "somebody_elses_run")
        prepared = make_run(runs, "prepared")
        write_settings(tmp_path / "LM3_settings.yaml", run_name="prepared", output_dir=str(runs))
        ref = progress_api.resolve_run()
        assert ref is not None and ref.root == prepared and ref.source == "settings"

        write_settings(tmp_path / "LM3_settings.yaml", run_name="", output_dir=str(runs))
        fallback = progress_api.resolve_run()
        assert fallback is not None and fallback.source == "discovered"

    def test_discovery_can_never_authorize_control(self, tmp_path: Path) -> None:
        """Step 4: discovery is "labeled ``source: discovered`` and never able to authorize control".

        The label is not decoration -- it is the flag every control-adjacent caller reads. A recency
        guess over directories nobody claimed must not be usable as permission to stop or overwrite.
        """
        runs = tmp_path / "runs"
        make_run(runs, "found_by_scanning")
        write_settings(tmp_path / "LM3_settings.yaml", run_name="", output_dir=str(runs))
        guessed = progress_api.resolve_run()
        assert guessed is not None and guessed.source == "discovered"
        assert guessed.may_authorize_control is False
        assert progress_api.active_runtime_ref() is None, (
            "the registry accessor must not fall back to discovery -- that is what makes it usable "
            "as an authorization input"
        )

        record = pipeline_record(make_run(runs, "claimed"), run_id="rid-c",
                                config_path=make_config(tmp_path))
        with holding_the_lease(record):
            claimed = progress_api.resolve_run()
            assert claimed is not None and claimed.may_authorize_control is True

    def test_an_abandoned_record_does_not_pin_the_view(self, tmp_path: Path) -> None:
        """A record saying ``running`` with the lease FREE is a crashed writer, not an active run.

        Pinning every view to a ledger nobody is advancing is the exact failure ``_discover_run``
        already learned to avoid; the registry tier must not reintroduce it.
        """
        runs = tmp_path / "runs"
        crashed = make_run(runs, "crashed")
        prepared = make_run(runs, "prepared")
        write_settings(tmp_path / "LM3_settings.yaml", run_name="prepared", output_dir=str(runs))
        records_mod.atomic_write_json(
            deployment_dir() / "active.json",
            records_mod.record_to_dict(pipeline_record(crashed, run_id="rid-dead",
                                                       config_path=make_config(tmp_path))),
        )
        progress_api.invalidate_run_caches()
        ref = progress_api.resolve_run()
        assert ref is not None
        assert ref.root == prepared and ref.source == "settings"

    def test_a_hardware_setup_root_does_not_switch_project_history(self, tmp_path: Path) -> None:
        """Section 2.6: show a tuning state in the Machine panel, do NOT switch project history.

        The record carries no project block at all (invariant 6), so there is nothing to switch to
        -- and this must not be "fixed" by descending into the calibration child, whose run name is
        ``_lm3_calibration``.
        """
        runs = tmp_path / "runs"
        prepared = make_run(runs, "prepared")
        write_settings(tmp_path / "LM3_settings.yaml", run_name="prepared", output_dir=str(runs))
        record = hardware_record(tmp_path / "hardware_settings.yaml", run_id="rid-hw",
                                 config_path=make_config(tmp_path))
        with holding_the_lease(record):
            ref = progress_api.resolve_run()
            assert ref is not None
            assert ref.root == prepared and ref.source == "settings"
            assert progress_api.active_runtime_ref() is None

    def test_with_the_flag_off_the_registry_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Step 3's flag gates Step 4 too: flag off must be byte-identical to the old precedence."""
        runs = tmp_path / "runs"
        live = make_run(runs, "live")
        prepared = make_run(runs, "prepared")
        write_settings(tmp_path / "LM3_settings.yaml", run_name="prepared", output_dir=str(runs))
        record = pipeline_record(live, run_id="rid-off", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            monkeypatch.setenv(FLAG, "0")
            progress_api.invalidate_run_caches()
            ref = progress_api.resolve_run()
            assert ref is not None
            assert ref.root == prepared and ref.source == "settings"
            assert progress_api.active_runtime_ref() is None


# --------------------------------------------------------------------------- #
# run_id: identity that a settings edit cannot move
# --------------------------------------------------------------------------- #
class TestRunIdentity:
    def test_the_snapshot_cache_key_carries_the_run_id(self, tmp_path: Path) -> None:
        """Step 4: "Use ``run_id`` in snapshot cache keys".

        ``db_path`` is not an identity: rename the project and the same run resolves to a different
        path, rerun it and two different runs resolve to the same one. Either direction serves one
        run's cached snapshot as another's.
        """
        runs = tmp_path / "runs"
        live = make_run(runs, "live")
        record = pipeline_record(live, run_id="rid-key", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            progress_api._SNAPSHOT_CACHE.invalidate()
            progress_api.status()
            keys = list(progress_api._SNAPSHOT_CACHE._entries)
            assert len(keys) == 1
            assert "rid-key" in keys[0], f"run_id missing from the snapshot cache key {keys[0]!r}"

    def test_the_log_follow_key_changes_when_the_run_does_but_not_when_the_name_does(
        self, tmp_path: Path
    ) -> None:
        """Step 4: "follow the active ``run_id`` for logs".

        The console's re-resolve loop switches files when ``(run_id, log_path)`` changes. Both
        halves matter: a rename must NOT look like a new run, and a rerun that truncates the same
        log file in place MUST -- a path comparison gets both cases backwards.
        """
        runs = tmp_path / "runs"
        live = make_run(runs, "reused_name")
        first = pipeline_record(live, run_id="rid-first", config_path=make_config(tmp_path))
        with holding_the_lease(first) as store:
            before = progress_api.resolve_run()
            assert before is not None
            write_settings(tmp_path / "LM3_settings.yaml", run_name="something_else",
                           output_dir=str(runs))
            after = progress_api.resolve_run()
            assert after is not None
            assert (after.run_id, after.log_path) == (before.run_id, before.log_path)
            finish(store, first)

        second = pipeline_record(live, run_id="rid-second", config_path=make_config(tmp_path))
        with holding_the_lease(second):
            rerun = progress_api.resolve_run()
            assert rerun is not None
            assert rerun.log_path == before.log_path, "the same project reuses the same log file"
            assert (rerun.run_id, rerun.log_path) != (before.run_id, before.log_path)

    def test_the_record_supplies_the_starting_state_the_ledger_cannot(self, tmp_path: Path) -> None:
        """A run between lease acquisition and its first ledger commit is running, not idle."""
        runs = tmp_path / "runs"
        live = runs / "not_built_yet"
        (live / "logs").mkdir(parents=True)
        record = pipeline_record(live, run_id="rid-start", config_path=make_config(tmp_path),
                                 state=RunState.STARTING)
        with holding_the_lease(record):
            snapshot = progress_api.status()
            assert snapshot["state"] == "running"
            assert snapshot["ready"] is False
            assert snapshot["run_name"] == "not_built_yet"


# --------------------------------------------------------------------------- #
# last.json on completion
# --------------------------------------------------------------------------- #
class TestCompletion:
    def test_on_completion_the_view_follows_last_json_not_a_newer_unrelated_run(
        self, tmp_path: Path
    ) -> None:
        """Step 4: "on completion follow ``last.json`` rather than the most recently modified
        unrelated run"."""
        runs = tmp_path / "runs"
        mine = make_run(runs, "mine")
        record = pipeline_record(mine, run_id="rid-done", config_path=make_config(tmp_path))
        with holding_the_lease(record) as store:
            finish(store, record)

        # Somebody else's run is newer on disk, and the settings now name a project that has never
        # been created -- which is what a mid-run rename leaves behind the moment the run ends.
        stranger = make_run(runs, "stranger")
        os.utime(stranger / "stranger.sqlite", (time.time() + 60, time.time() + 60))
        write_settings(tmp_path / "LM3_settings.yaml", run_name="renamed_mid_run",
                       output_dir=str(runs))
        progress_api.invalidate_run_caches()

        ref = progress_api.resolve_run()
        assert ref is not None
        assert ref.root == mine, "the run that just finished, not the newest directory"
        assert ref.source == "last"
        assert ref.run_id == "rid-done"

    def test_last_json_does_not_outrank_a_prepared_project_that_exists(
        self, tmp_path: Path
    ) -> None:
        """An idle deployment whose configured project EXISTS is still tier 4: the user pointed
        the app at it, and history does not get to override a live selection."""
        runs = tmp_path / "runs"
        mine = make_run(runs, "mine")
        record = pipeline_record(mine, run_id="rid-hist", config_path=make_config(tmp_path))
        with holding_the_lease(record) as store:
            finish(store, record)
        prepared = make_run(runs, "prepared")
        write_settings(tmp_path / "LM3_settings.yaml", run_name="prepared", output_dir=str(runs))
        ref = progress_api.resolve_run()
        assert ref is not None and ref.root == prepared and ref.source == "settings"


# --------------------------------------------------------------------------- #
# Section 2.10 GUI database resolution
# --------------------------------------------------------------------------- #
def staged_run(tmp_path: Path, name: str, *, run_id: str) -> tuple[Path, Path, Path]:
    """Lay out a CLUSTER-profile run on disk: persistent artifacts, node-local scratch.

    Section 2.10's role table, verbatim: ``artifact_dir`` is persistent project storage,
    ``active_state_dir`` is node-local scratch, ``active_db_path`` lives in the scratch, and
    ``archived_db_path`` is the generation named by ``artifact_dir/archive.current.json``. One
    generation has committed, so the pointer resolves and ``archive_status`` is ``ready``.
    """
    artifact_dir = tmp_path / "shared" / name
    (artifact_dir / "logs").mkdir(parents=True, exist_ok=True)
    (artifact_dir / "logs" / "lm3.log").write_text("12:00:00 INFO lm3: hello\n", encoding="utf-8")
    scratch = tmp_path / "scratch" / name
    scratch.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(scratch / f"{name}.sqlite").close()
    generation = artifact_dir / f"archive.{run_id}.1.sqlite"
    sqlite3.connect(generation).close()
    (artifact_dir / "archive.current.json").write_text(
        json.dumps({"schema_version": 1, "run_id": run_id, "generation": f"{run_id}.1",
                    "archived_db_path": str(generation), "committed_at": time.time()}),
        encoding="utf-8",
    )
    return artifact_dir, scratch, generation


def staged_record(artifact_dir: Path, scratch: Path, generation: Path, *, run_id: str,
                  config_path: Path, state: RunState = RunState.RUNNING) -> RuntimeRecord:
    """A valid STAGED ``pipeline`` record for the layout :func:`staged_run` built."""
    now = records_mod.utc_now()
    project = ProjectBlock(
        run_name=artifact_dir.name,
        input_dirs=(str(artifact_dir.parent),),
        artifact_dir=str(artifact_dir),
        active_state_dir=str(scratch),
        active_db_path=str(scratch / f"{artifact_dir.name}.sqlite"),
        archive_mode=ArchiveMode.STAGED,
        archive_status=ArchiveStatus.READY,
        archive_pointer_path=str(artifact_dir / "archive.current.json"),
        archived_db_path=str(generation),
        run_dir=str(artifact_dir),
        log_path=str(artifact_dir / "logs" / "lm3.log"),
    )
    return RuntimeRecord(
        run_id=run_id,
        activity=Activity.PIPELINE,
        activity_role=ActivityRole.ROOT,
        state=state,
        launcher=Launcher.CLI,
        pid=os.getpid(),
        process_started_at=records_mod.process_start_time(),
        started_at=now,
        updated_at=now,
        finished_at=now if state in {RunState.DONE, RunState.ERROR} else None,
        deployment=records_mod.build_deployment_info(),
        config=ConfigRef(path=str(config_path), sha256="0" * 64),
        project=project,
    )


class TestGuiDatabaseResolution:
    """Section 2.10: "while the run is active -> ``active_db_path``; after terminal finalization ->
    ``archived_db_path``; ``last.json`` must **never** leave the historical GUI pointing at deleted
    node-local scratch" -- gates 26 and 59."""

    def test_a_live_staged_run_still_reads_the_node_local_active_db(self, tmp_path: Path) -> None:
        """The active half of the rule: while the allocation holds, the scratch ledger IS the run."""
        artifact_dir, scratch, generation = staged_run(tmp_path, "staged_live", run_id="rid-live-st")
        record = staged_record(artifact_dir, scratch, generation, run_id="rid-live-st",
                               config_path=make_config(tmp_path))
        with holding_the_lease(record):
            ref = progress_api.resolve_run()
            assert ref is not None and ref.source == "runtime"
            assert ref.db_path == scratch / "staged_live.sqlite", (
                "a running staged job writes node-local scratch; the archive lags it by one "
                "checkpoint interval"
            )

    def test_a_finalized_staged_run_reads_the_archive_not_the_dead_scratch(
        self, tmp_path: Path
    ) -> None:
        """Gate 26. The allocation ends and takes ``active_state_dir`` with it; the historical GUI
        must land on the generation in persistent storage, not on a path that no longer exists."""
        artifact_dir, scratch, generation = staged_run(tmp_path, "staged_done", run_id="rid-st")
        record = staged_record(artifact_dir, scratch, generation, run_id="rid-st",
                               config_path=make_config(tmp_path))
        with holding_the_lease(record) as store:
            finish(store, record)

        shutil.rmtree(scratch)                       # what Slurm does when the allocation ends
        progress_api.invalidate_run_caches()

        ref = progress_api.resolve_run()
        assert ref is not None, "the run directory survives, so history is still resolvable"
        assert ref.source == "last" and ref.run_id == "rid-st"
        assert ref.root == artifact_dir
        assert ref.db_path == generation, (
            "last.json must never leave the historical GUI pointing at deleted node-local scratch"
        )
        assert ref.db_path.exists(), "a ref whose ledger does not exist renders 'database missing'"

    def test_the_in_place_desktop_ref_is_unchanged(self, tmp_path: Path) -> None:
        """The fix is INERT on the desktop: in-place mode has ``archived_db_path ==
        active_db_path`` (section 2.10's mode table), so both tiers produce the same ref."""
        runs = tmp_path / "runs"
        mine = make_run(runs, "desktop")
        record = pipeline_record(mine, run_id="rid-inplace", config_path=make_config(tmp_path))
        assert record.project is not None
        assert record.project.archived_db_path == record.project.active_db_path
        with holding_the_lease(record) as store:
            live = progress_api.resolve_run()
            assert live is not None and live.db_path == mine / "desktop.sqlite"
            finish(store, record)
        progress_api.invalidate_run_caches()

        historical = progress_api.resolve_run()
        assert historical is not None and historical.source == "last"
        assert historical.db_path == mine / "desktop.sqlite"


# --------------------------------------------------------------------------- #
# The duplicate GET /v1/runs
# --------------------------------------------------------------------------- #
def test_the_progress_router_no_longer_serves_v1_runs() -> None:
    """Step 4: "Remove the duplicate progress-router ``GET /v1/runs``, keeping the ``results_api``
    route". Two routers answering one path meant the winner was decided by registration ORDER."""
    pytest.importorskip("fastapi")
    api = progress_api.router()
    paths = {getattr(route, "path", None) for route in api.routes}
    assert "/v1/runs" not in paths
    assert {"/v1/status", "/v1/logs", "/v1/status/stream", "/v1/logs/stream"} <= paths
    assert callable(progress_api.list_runs), "the function stays; only the duplicate ROUTE goes"


# --------------------------------------------------------------------------- #
# Section 2.8 postprocessing guards
# --------------------------------------------------------------------------- #
def make_tool(tool_id: str, *, access: str = "read_write",
              runner: Any = None) -> postprocess_api.Tool:
    return postprocess_api.Tool(
        id=tool_id,
        name=tool_id,
        description="test tool",
        inputs=(postprocess_api.ToolInput(key="run_dir", label="Run folder", type="path",
                                          accepts="dir", must_exist=True, required=True),),
        outputs_description="nothing",
        runner=runner or (lambda params, ctx: {"ok": True}),
        settings_key=tool_id,
        module="tests.test_progress_registry",
        cli="none",
        access=access,
        resource="cpu",
        target_keys=("run_dir",),
    )


class TestPostprocessingGuards:
    def test_every_shipped_tool_declares_access_and_resource(self) -> None:
        """Section 2.8: "Every tool declares ``read_only``/``read_write`` and ``cpu``/``gpu``"."""
        for described in postprocess_api.list_tools():
            assert described["access"] in {"read_only", "read_write"}
            assert described["resource"] in {"cpu", "gpu"}, (
                "a 'gpu' postprocessor may not be added without explicit device coordination"
            )
            assert described["resource"] == "cpu", (
                "the shipped tools import no torch/onnxruntime; change this test WITH the "
                "coordination, never before it"
            )
        for tool in postprocess_api._TOOLS.values():
            assert tool.target_keys, f"{tool.id} declares no target -- the guard would never fire"

    def test_a_tool_aimed_at_the_active_run_is_refused(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        live = make_run(runs, "live")
        (live / "reports").mkdir()
        tool = make_tool("t_active")
        record = pipeline_record(live, run_id="rid-pp", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            with pytest.raises(postprocess_api.TargetActive) as excinfo:
                postprocess_api.check_target_allowed(tool, {"run_dir": live})
            assert excinfo.value.run_id == "rid-pp"
            assert "running right now" in str(excinfo.value)

    def test_a_file_inside_the_active_run_is_refused_too(self, tmp_path: Path) -> None:
        """The guard resolves a target to its ENCLOSING run dir: masks are named, not run dirs."""
        runs = tmp_path / "runs"
        live = make_run(runs, "live")
        mask = live / "reports" / "Leaf_Original" / "Lamina_Mask" / "a__og-SEG-lamina.png"
        mask.parent.mkdir(parents=True)
        mask.write_bytes(b"")
        record = pipeline_record(live, run_id="rid-pp2", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            with pytest.raises(postprocess_api.TargetActive):
                postprocess_api.check_target_allowed(make_tool("t_mask"), {"run_dir": mask})

    def test_a_different_completed_run_is_allowed_while_a_pipeline_is_active(
        self, tmp_path: Path
    ) -> None:
        """Section 2.8: "CPU-only tools may run against a different completed run while a pipeline
        is active"."""
        runs = tmp_path / "runs"
        live = make_run(runs, "live")
        done = make_run(runs, "finished_yesterday")
        (done / "reports").mkdir()
        record = pipeline_record(live, run_id="rid-pp3", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            targets = postprocess_api.check_target_allowed(make_tool("t_other"), {"run_dir": done})
        assert targets == [done]

    def test_a_symlinked_spelling_of_the_active_run_is_still_refused(self, tmp_path: Path) -> None:
        """Both sides of the comparison must be resolved, or the refusal is trivially evaded.

        Every TARGET reaches the guard through ``resolve_path``, which realpaths. The record
        carries whatever spelling the pipeline was launched with -- and on a cluster the output
        root is very often a symlink. Comparing the two spellings unresolved lets a tool write
        into the live run through the other name.
        """
        real = tmp_path / "real_runs"
        live = make_run(real, "live")
        link = tmp_path / "runs_link"
        try:
            link.symlink_to(real, target_is_directory=True)
        except (OSError, NotImplementedError):          # pragma: no cover - unprivileged Windows
            pytest.skip("this filesystem/user cannot create symlinks")
        record = pipeline_record(link / "live", run_id="rid-link", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            assert postprocess_api.active_run_target()["artifact_dir"] == live
            with pytest.raises(postprocess_api.TargetActive):
                postprocess_api.check_target_allowed(make_tool("t_link"), {"run_dir": live})

    def test_two_read_write_tools_on_one_completed_run_are_serialized(
        self, tmp_path: Path
    ) -> None:
        runs = tmp_path / "runs"
        done = make_run(runs, "done")
        (done / "reports").mkdir()
        first = make_tool("t_writer_a")
        second = make_tool("t_writer_b")
        held = postprocess_api._acquire_artifact_locks(first, [done])
        assert held
        try:
            with pytest.raises(postprocess_api.TargetLocked):
                postprocess_api._acquire_artifact_locks(second, [done])
            other = make_run(runs, "unrelated")
            elsewhere = postprocess_api._acquire_artifact_locks(second, [other])
            assert elsewhere, "a DIFFERENT run is never blocked by this lock"
            for lock in elsewhere:
                lock.release()
        finally:
            for lock in held:
                lock.release()
        again = postprocess_api._acquire_artifact_locks(second, [done])
        assert again, "the lock is released when the first tool finishes"
        for lock in again:
            lock.release()

    def test_a_read_only_tool_takes_no_lock(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        done = make_run(runs, "done")
        writer = postprocess_api._acquire_artifact_locks(make_tool("t_w"), [done])
        try:
            reader = postprocess_api._acquire_artifact_locks(
                make_tool("t_r", access="read_only"), [done])
            assert reader == []
        finally:
            for lock in writer:
                lock.release()

    def test_the_lock_lives_in_the_runtime_registry_never_in_the_output_directory(
        self, tmp_path: Path
    ) -> None:
        """Section 2.8: "never as a lock file inside the output directory itself. Placing it beside
        the artifacts would put an advisory lock on exactly the network storage section 3.1
        declares unreliable for locking"."""
        import hashlib

        runs = tmp_path / "runs"
        done = make_run(runs, "done")
        before = sorted(p.name for p in done.iterdir())
        path = postprocess_api.artifact_lock_path(done)
        assert path is not None
        assert path.parent == deployment_dir() / "postprocess"
        digest = hashlib.sha256(str(done).encode("utf-8", "surrogateescape")).hexdigest()
        assert path.name == f"{digest}.lock", "keyed by a hash of the RESOLVED artifact_dir"
        held = postprocess_api._acquire_artifact_locks(make_tool("t_place"), [done])
        try:
            assert sorted(p.name for p in done.iterdir()) == before, (
                "nothing may be written into the run directory to take this lock"
            )
        finally:
            for lock in held:
                lock.release()

    def test_start_tool_refuses_the_active_run_end_to_end(self, tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
        """The HTTP path: ``start_tool`` -> validate -> guard, with no task registered on refusal."""
        runs = tmp_path / "runs"
        live = make_run(runs, "live")
        (live / "reports").mkdir()
        tool = make_tool("t_e2e")
        monkeypatch.setitem(postprocess_api._TOOLS, tool.id, tool)
        record = pipeline_record(live, run_id="rid-e2e", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            with pytest.raises(postprocess_api.TargetActive):
                postprocess_api.start_tool(tool.id, {"run_dir": str(live)})
        assert tool.id not in postprocess_api.registry().active()

    def test_a_running_tool_holds_the_lock_until_it_finishes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The lock is taken in ``start`` and released by the worker thread, so it spans the run."""
        runs = tmp_path / "runs"
        done = make_run(runs, "done")
        (done / "reports").mkdir()
        release = threading.Event()

        def slow(params: dict, ctx: Any) -> dict:
            assert release.wait(timeout=20.0)
            return {"ok": True}

        first = make_tool("t_slow", runner=slow)
        second = make_tool("t_second")
        monkeypatch.setitem(postprocess_api._TOOLS, first.id, first)
        monkeypatch.setitem(postprocess_api._TOOLS, second.id, second)
        task = postprocess_api.start_tool(first.id, {"run_dir": str(done)})
        try:
            deadline = time.time() + 10.0
            while time.time() < deadline and not postprocess_api.registry().active():
                time.sleep(0.02)
            with pytest.raises(postprocess_api.TargetLocked):
                postprocess_api.start_tool(second.id, {"run_dir": str(done)})
        finally:
            release.set()
        deadline = time.time() + 20.0
        while time.time() < deadline and postprocess_api.registry().get(task.id).state == "running":
            time.sleep(0.02)
        assert postprocess_api.registry().get(task.id).state == "done"
        later = postprocess_api.start_tool(second.id, {"run_dir": str(done)})
        deadline = time.time() + 20.0
        while time.time() < deadline and postprocess_api.registry().get(later.id).state == "running":
            time.sleep(0.02)
        assert postprocess_api.registry().get(later.id).state == "done"

    def test_with_the_flag_off_the_guards_are_inert(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runs = tmp_path / "runs"
        live = make_run(runs, "live")
        record = pipeline_record(live, run_id="rid-off2", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            monkeypatch.setenv(FLAG, "0")
            progress_api.invalidate_run_caches()
            assert postprocess_api.active_run_target() is None
            assert postprocess_api.check_target_allowed(make_tool("t_off"), {"run_dir": live}) \
                == [live]
            assert postprocess_api._acquire_artifact_locks(make_tool("t_off"), [live]) == []

    def test_the_context_payload_names_the_active_run(self, tmp_path: Path) -> None:
        """So the tab can gray a target out instead of learning about it through a 409."""
        runs = tmp_path / "runs"
        live = make_run(runs, "live")
        record = pipeline_record(live, run_id="rid-ctx", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            payload = postprocess_api.context()
        assert payload["active_run"] == {"artifact_dir": str(live), "run_name": "live",
                                         "run_id": "rid-ctx"}
        assert postprocess_api.context()["active_run"] is None


# --------------------------------------------------------------------------- #
# THE EXIT GATE
# --------------------------------------------------------------------------- #
class TestTheExitGate:
    """Step 4's exit gate, verbatim: "changing ``project.run_name`` during an active run does not
    move status, logs, results, or postprocessing away from the active DB"."""

    @staticmethod
    def _rename(tmp_path: Path, runs: Path) -> None:
        """What the user does: types a new project name into the settings strip mid-run.

        It is written straight to the settings file because that is what ``PUT /v1/settings`` does,
        and because the whole point is that the file is NOT the authority on what is running.
        """
        write_settings(tmp_path / "LM3_settings.yaml", run_name="a_totally_different_project",
                       output_dir=str(runs))

    def test_status_and_logs_stay_on_the_active_run(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        live = make_run(runs, "the_real_run")
        write_settings(tmp_path / "LM3_settings.yaml", run_name="the_real_run",
                       output_dir=str(runs))
        record = pipeline_record(live, run_id="rid-gate", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            before = progress_api.status()
            before_log = progress_api.log_tail()
            self._rename(tmp_path, runs)

            after = progress_api.status()
            assert after["db_path"] == before["db_path"] == str(live / "the_real_run.sqlite")
            assert after["run_name"] == "the_real_run"
            assert after["source"] == "runtime"
            assert after["state"] == "running"

            after_log = progress_api.log_tail()
            assert after_log["path"] == before_log["path"] == str(live / "logs" / "lm3.log")
            assert after_log["ready"] is True

    def test_the_run_listing_still_marks_the_active_run(self, tmp_path: Path) -> None:
        """The results side of the gate that this module owns: ``list_runs()``'s ``active`` marker
        (``results_api.run_roots`` is a sibling change in another module)."""
        runs = tmp_path / "runs"
        live = make_run(runs, "the_real_run")
        make_run(runs, "an_older_run")
        record = pipeline_record(live, run_id="rid-gate2", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            self._rename(tmp_path, runs)
            listing = progress_api.list_runs()
            assert listing["active"] == str(live)
            marked = [row for row in listing["runs"] if row["active"]]
            assert [row["run_name"] for row in marked] == ["the_real_run"]

    def test_postprocessing_still_refuses_the_active_run(self, tmp_path: Path) -> None:
        """The rename must not make the live run look like a completed one to section 2.8."""
        runs = tmp_path / "runs"
        live = make_run(runs, "the_real_run")
        (live / "reports").mkdir()
        record = pipeline_record(live, run_id="rid-gate3", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            self._rename(tmp_path, runs)
            assert postprocess_api.active_run_target()["artifact_dir"] == live
            with pytest.raises(postprocess_api.TargetActive):
                postprocess_api.check_target_allowed(make_tool("t_gate"), {"run_dir": live})

    def test_an_explicit_selection_is_the_only_thing_that_moves_the_view(
        self, tmp_path: Path
    ) -> None:
        """Invariant 7's other half: the override exists, it is explicit, and it is per-request."""
        runs = tmp_path / "runs"
        live = make_run(runs, "the_real_run")
        other = make_run(runs, "an_older_run")
        record = pipeline_record(live, run_id="rid-gate4", config_path=make_config(tmp_path))
        with holding_the_lease(record):
            self._rename(tmp_path, runs)
            chosen = progress_api.status(run="an_older_run")
            assert chosen["run_path"] == str(other)
            assert chosen["source"] == "query"
            assert progress_api.status()["run_path"] == str(live), (
                "the override is scoped to the request that asked for it"
            )
