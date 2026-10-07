"""The two inputs the Postprocess tab's guards must actually observe: run history and the lease.

Plan clauses under test:

* **section 3.1 row 5** -- "runs roots (history): 1. ``LM3_RUNS_ROOTS`` - 2. active/last runtime
  output roots - 3. ``project.output.dir`` from the resolved settings". ``allowed_roots()`` states
  in its own comments that it covers "every run-history root the Results tab can list, so a run
  visible in the app is a run the Postprocess tab may read". That coupling is two-way, and it was
  defeated by a memo whose key observes only the settings file, ``LM3_POSTPROCESS_ROOTS``, the
  ``project`` block and each tool's ``output_dir`` -- none of which move when a run starts or ends.
* **section 2.8** -- "It is **refused** when its target is the currently active pipeline's run
  directory", and "CPU-only tools may run against a **different completed run** while a pipeline is
  active". The refusal is an authorization decision, so it may not be answered out of
  ``progress_api``'s 0.25-2.0 s registry memo, which no run-lifecycle event invalidates.

Everything runs in-process against real artifacts: a real POSIX/Windows lease (so the section 2.9
reader classifies off the OS lock and never off the JSON), a real ``active.json`` written through
:class:`RecordStore`, and a private deployment under ``tmp_path``. No server is bound to a port and
no pipeline is started.
"""
from __future__ import annotations

import contextlib
import os
import sqlite3
from pathlib import Path
from typing import Iterator

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
from leafmachine3.server import postprocess_api, progress_api, results_api

FLAG = "LM3_RUNTIME_V2"


# --------------------------------------------------------------------------- #
# Fixtures and builders
# --------------------------------------------------------------------------- #
@pytest.fixture
def deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A private deployment whose settings-derived roots deliberately EXCLUDE ``scratch_output``.

    That exclusion is the whole point of this module: a run written outside every configured root
    is exactly the CLI case section 3.1 row 5 slot 2 exists to serve, so ``LM3_POSTPROCESS_ROOTS``
    is left empty here rather than pointed at ``tmp_path``.

    The flag is set explicitly on the way in rather than inherited: a developer with
    ``LM3_RUNTIME_V2`` exported would otherwise run this file in the opposite mode from CI.
    """
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    monkeypatch.setenv("LM3_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("LM3_DEPLOYMENT_ID", "test-postprocess-guards")
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setenv("LM3_SERVER_JOBS", str(jobs))
    monkeypatch.setenv("LM3_POSTPROCESS_ROOTS", "")
    monkeypatch.delenv("LM3_RUNS_ROOTS", raising=False)
    monkeypatch.delenv("LM3_STATUS_ROOTS", raising=False)
    monkeypatch.setenv(FLAG, "1")

    cfg_dir = tmp_path / "cfg"
    (cfg_dir / "runs").mkdir(parents=True)
    settings = cfg_dir / "LM3_settings.yaml"
    settings.write_text(
        yaml.safe_dump({"project": {"run_name": "", "output": {"dir": str(cfg_dir / "runs")}}},
                       sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("LM3_SETTINGS", str(settings))
    postprocess_settings = cfg_dir / "postprocessing.yaml"
    postprocess_settings.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("LM3_POSTPROCESS_SETTINGS", str(postprocess_settings))

    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)                      # nothing may resolve against it (section 3.1)

    reset_module_state()
    yield deployment_dir()
    reset_module_state()


def reset_module_state() -> None:
    """Every module-level memo these two functions read, so tests cannot leak into each other."""
    postprocess_api._roots_cache.update({"key": None, "roots": []})
    postprocess_api._yaml_cache.update({"path": None, "mtime": None, "data": {}})
    postprocess_api._LOCAL_LOCKS.clear()
    results_api._EXTRA_ROOTS.clear()
    progress_api.invalidate_run_caches()


def deployment_dir() -> Path:
    from leafmachine3.core import paths as core_paths

    return core_paths.deployment_runtime_dir(create=True, check_filesystem=False)


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


def pipeline_record(run_dir: Path, *, run_id: str, config_path: Path) -> RuntimeRecord:
    """A valid in-place ``running`` ``pipeline`` root record for ``run_dir`` (section 3.2)."""
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
        state=RunState.RUNNING,
        launcher=Launcher.CLI,
        pid=os.getpid(),
        process_started_at=records_mod.process_start_time(),
        started_at=now,
        updated_at=now,
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


class Published:
    """A record published as ``active.json`` under a genuinely held lease, retractable mid-test.

    Not a context manager on purpose: several tests below have to observe the state BEFORE the
    record exists and AFTER it stops existing, in one process, with no cache invalidation in
    between -- which is exactly the coupling under test.
    """

    def __init__(self, record: RuntimeRecord) -> None:
        self.record = record
        self.lease = lease_mod.RuntimeLease()
        self.lease.acquire()
        self.store = records_mod.RecordStore(deployment_dir(), run_id=record.run_id)
        self.store.write_active(record)
        self.path = deployment_dir() / records_mod.ACTIVE_RECORD_FILENAME

    def retract(self) -> None:
        """Release the lease and remove the record, the way a cleanup/quarantine leaves things.

        Deliberately NOT ``finalize()``: ``last.json`` is itself a row-5 slot-2 contribution, so a
        finished run keeps its root legitimately. The stale-widening case is the one where NO
        record names the root any more.
        """
        with contextlib.suppress(Exception):
            self.lease.release()
        self.path.unlink(missing_ok=True)


def make_tool(tool_id: str, *, access: str = "read_write") -> postprocess_api.Tool:
    return postprocess_api.Tool(
        id=tool_id,
        name=tool_id,
        description="test tool",
        inputs=(postprocess_api.ToolInput(key="run_dir", label="Run folder", type="path",
                                          accepts="dir", must_exist=True, required=True),),
        outputs_description="nothing",
        runner=lambda params, ctx: {"ok": True},
        settings_key=tool_id,
        module="tests.test_postprocess_api",
        cli="none",
        access=access,
        resource="cpu",
        target_keys=("run_dir",),
    )


def can_read(run_dir: Path) -> bool:
    """Does the path sandbox admit ``run_dir``? The one question the Postprocess tab asks."""
    try:
        postprocess_api.resolve_path(str(run_dir), must_exist=True, kind="dir")
    except postprocess_api.ParamError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Section 3.1 row 5: the sandbox follows run history, in BOTH directions
# --------------------------------------------------------------------------- #
class TestAllowedRootsFollowRunHistory:
    @pytest.mark.parametrize("prewarm", [False, True], ids=["cold-memo", "warm-memo"])
    def test_a_run_outside_every_configured_root_becomes_readable(
        self, tmp_path: Path, deployment: Path, prewarm: bool
    ) -> None:
        """The false refusal: a CLI run with ``--output /scratch`` that the Results tab lists.

        ``warm-memo`` is the regression: ``allowed_roots()`` is computed BEFORE the run exists, and
        nothing in the old memo key changes when the record appears -- so the tab listed a run the
        Postprocess tab then refused with a 422 naming roots that did not include it.
        """
        scratch = tmp_path / "scratch_output"
        run_dir = make_run(scratch, "cli_run")
        # The static half only, so ``cold-memo`` really is cold: asking the composed
        # ``allowed_roots()`` here would itself be the pre-warm the other parameter supplies.
        assert scratch not in postprocess_api._static_allowed_roots(), (
            "the fixture must place the run outside every settings-derived root"
        )
        if prewarm:
            assert scratch not in postprocess_api.allowed_roots()
            assert not can_read(run_dir)

        published = Published(pipeline_record(run_dir, run_id="rid-scratch",
                                              config_path=make_config(tmp_path)))
        try:
            assert scratch in results_api.run_roots(), "row 5 slot 2: the Results tab can list it"
            assert scratch in postprocess_api.allowed_roots(), (
                "a run visible in the app is a run the Postprocess tab may read (section 3.1 row 5)"
            )
            assert can_read(run_dir)
        finally:
            published.retract()

    def test_a_root_no_record_names_any_more_stops_being_readable(
        self, tmp_path: Path, deployment: Path
    ) -> None:
        """The stale-widening half: the sandbox must SHRINK with run history, not only grow.

        A memoized union kept the scratch root writable for the life of the server after the record
        that contributed it was gone -- a path-sandbox entry backed by no current input.
        """
        scratch = tmp_path / "scratch_output"
        run_dir = make_run(scratch, "cli_run")
        published = Published(pipeline_record(run_dir, run_id="rid-gone",
                                              config_path=make_config(tmp_path)))
        try:
            assert scratch in postprocess_api.allowed_roots()   # warm the memo WHILE it is live
            assert can_read(run_dir)
        finally:
            published.retract()

        assert scratch not in results_api.run_roots()
        assert scratch not in postprocess_api.allowed_roots()
        assert not can_read(run_dir)
        with pytest.raises(postprocess_api.ParamError, match="outside the allowed roots"):
            postprocess_api.resolve_path(str(run_dir), must_exist=True, kind="dir")

    def test_the_flag_off_case_is_unchanged(
        self, tmp_path: Path, deployment: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With ``LM3_RUNTIME_V2`` off nothing reads a record, so the sandbox is settings-only."""
        monkeypatch.setenv(FLAG, "0")
        scratch = tmp_path / "scratch_output"
        run_dir = make_run(scratch, "cli_run")
        published = Published(pipeline_record(run_dir, run_id="rid-off",
                                              config_path=make_config(tmp_path)))
        try:
            assert scratch not in results_api.run_roots()
            assert scratch not in postprocess_api.allowed_roots()
            assert not can_read(run_dir)
        finally:
            published.retract()

    def test_the_configured_roots_still_hold_and_stay_memoized(
        self, tmp_path: Path, deployment: Path
    ) -> None:
        """Taking the runtime union out of the memo must not take the settings half with it."""
        cfg_dir = tmp_path / "cfg"
        run_dir = make_run(cfg_dir / "runs", "configured")
        assert can_read(run_dir)
        roots = postprocess_api.allowed_roots()
        assert cfg_dir in roots and (cfg_dir / "runs") in roots
        assert (tmp_path / "jobs") in roots, "the staged job dirs are row 4, not row 5"
        assert postprocess_api._roots_cache["key"] is not None, "the static half is still memoized"
        assert postprocess_api.allowed_roots() == roots
        assert len(roots) == len(set(roots)), "roots are de-duplicated across both halves"


# --------------------------------------------------------------------------- #
# Section 2.8: the refusal reads the registry, not a memo
# --------------------------------------------------------------------------- #
class TestTheActiveRunGuardReadsFresh:
    def test_a_run_that_starts_after_a_status_read_is_still_refused(
        self, tmp_path: Path, deployment: Path
    ) -> None:
        """The stale-cache window, with NO manual invalidation anywhere.

        ``progress_api``'s registry read is memoized behind a 0.25-2.0 s TTL that nothing in the
        run-start path invalidates, so a guard reading through it answers "nothing is running" for
        one TTL after ``active.json`` is published -- and admits a ``read_write`` tool into a run
        the pipeline is writing. The guard therefore reads the record itself.
        """
        runs = tmp_path / "cfg" / "runs"
        live = make_run(runs, "live")
        progress_api.active_runtime_ref()                    # a 2 Hz status poll, pre-launch

        published = Published(pipeline_record(live, run_id="rid-race",
                                              config_path=make_config(tmp_path)))
        try:
            # Re-read through the memo immediately before the guard: with the pre-launch answer
            # still cached this is ``None``, which is exactly the input the old guard trusted. The
            # assertion below holds either way, so the test survives progress_api gaining a
            # lifecycle invalidation later.
            progress_api.active_runtime_ref()
            target = postprocess_api.active_run_target()
            assert target is not None and target["artifact_dir"] == live
            assert target["run_id"] == "rid-race"
            with pytest.raises(postprocess_api.TargetActive):
                postprocess_api.check_target_allowed(make_tool("t_race"), {"run_dir": live})
        finally:
            published.retract()

    def test_a_run_that_ended_stops_being_refused_immediately(
        self, tmp_path: Path, deployment: Path
    ) -> None:
        """The same staleness in the other direction: a finished run must not be refused work.

        Section 2.8's whole point is that history is fair game; a cached "it is live" answer that
        outlives the lease refuses legitimate work on a run nobody is writing.
        """
        runs = tmp_path / "cfg" / "runs"
        done = make_run(runs, "done")
        published = Published(pipeline_record(done, run_id="rid-ended",
                                              config_path=make_config(tmp_path)))
        assert progress_api.active_runtime_ref() is not None  # warm the memo WHILE it is live
        published.retract()

        assert postprocess_api.active_run_target() is None
        assert postprocess_api.check_target_allowed(make_tool("t_ended"), {"run_dir": done}) == [done]

    def test_a_different_completed_run_is_allowed_while_one_is_live(
        self, tmp_path: Path, deployment: Path
    ) -> None:
        """Section 2.8, third bullet -- the fresh read must not turn into a blanket refusal."""
        runs = tmp_path / "cfg" / "runs"
        live = make_run(runs, "live")
        other = make_run(runs, "other")
        published = Published(pipeline_record(live, run_id="rid-both",
                                              config_path=make_config(tmp_path)))
        try:
            assert postprocess_api.check_target_allowed(make_tool("t_other"),
                                                        {"run_dir": other}) == [other]
        finally:
            published.retract()

    def test_a_hardware_setup_root_names_no_run_directory(
        self, tmp_path: Path, deployment: Path
    ) -> None:
        """Invariant 6: a tuning root has no project block, so it refuses nothing (section 2.6)."""
        runs = tmp_path / "cfg" / "runs"
        done = make_run(runs, "done")
        published = Published(hardware_record(tmp_path / "hardware_settings.yaml",
                                              run_id="rid-hw", config_path=make_config(tmp_path)))
        try:
            assert postprocess_api.active_run_target() is None
            assert postprocess_api.check_target_allowed(make_tool("t_hw"),
                                                        {"run_dir": done}) == [done]
        finally:
            published.retract()

    def test_the_guard_is_inert_with_the_flag_off(
        self, tmp_path: Path, deployment: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The explicit flag-off compatibility path keeps the postprocess guard disabled."""
        monkeypatch.setenv(FLAG, "0")
        runs = tmp_path / "cfg" / "runs"
        live = make_run(runs, "live")
        published = Published(pipeline_record(live, run_id="rid-flagoff",
                                              config_path=make_config(tmp_path)))
        try:
            assert postprocess_api.active_run_target() is None
            assert postprocess_api.check_target_allowed(make_tool("t_flagoff"),
                                                        {"run_dir": live}) == [live]
        finally:
            published.retract()
