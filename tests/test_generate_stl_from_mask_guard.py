"""Plan section 2.8 over the STANDALONE CLI surface of ``generate_stl_from_mask``.

Plan clauses under test:

* **section 2.8, bullet 2** -- "A postprocessor ... is **refused** when its target is the currently
  active pipeline's run directory." The bullet says *a postprocessor*, not *the HTTP API*.
* **section 2.8, bullet 5** -- two ``read_write`` tools on one completed run are serialized by the
  per-run advisory lock in the local deployment runtime registry.
* **section 2.8, last bullet** -- "Standalone CLI tools use the same guard as the HTTP API, not a
  parallel one." That one is not testable by observing a refusal (a copied guard refuses too), so
  it is tested by SUBSTITUTION: monkeypatching ``postprocess_api.check_target_allowed`` changes
  what the CLI does. A future parallel copy fails these tests.
* **section 4, Step 3's flag rule** -- with ``LM3_RUNTIME_V2`` off the CLI behaves exactly as it
  did before the guard existed.

The lease and the runtime record are REAL (``records.read_runtime`` decides "active" from the OS
lock, never from the JSON), and the advisory lock is the real one in the deployment runtime
directory. Only ``generate_stl_from_mask.run`` is stubbed: what is under test is which targets the
CLI is allowed to write, not the geometry, and stubbing it keeps these tests independent of the
optional trimesh/shapely stack that ``tests/test_generate_stl_from_mask.py`` skips on.
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
    Launcher,
    ProjectBlock,
    RunState,
    RuntimeRecord,
)
from leafmachine3.core.runtime import lease as lease_mod
from leafmachine3.core.runtime import records as records_mod
from leafmachine3.postprocessing import generate_stl_from_mask as stl_cli
from leafmachine3.server import postprocess_api, progress_api

FLAG = "LM3_RUNTIME_V2"


# --------------------------------------------------------------------------- #
# Fixtures and builders
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A private deployment: its own runtime dir and an empty working directory.

    ``LM3_RUNTIME_V2`` is set explicitly rather than inherited -- a developer with the flag
    exported would otherwise run this module in the opposite mode from CI, and every assertion
    here would pass or fail for the wrong reason.
    """
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    monkeypatch.setenv("LM3_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("LM3_DEPLOYMENT_ID", "test-stl-guard")
    monkeypatch.setenv(FLAG, "1")
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    reset_module_state()
    yield deployment_dir()
    reset_module_state()


def reset_module_state() -> None:
    progress_api.invalidate_run_caches()
    postprocess_api._LOCAL_LOCKS.clear()


def deployment_dir() -> Path:
    from leafmachine3.core import paths as core_paths

    return core_paths.deployment_runtime_dir(create=True, check_filesystem=False)


def make_run(output_dir: Path, name: str) -> Path:
    """A run directory exactly as ``core.dirs`` lays one out: ``<root>/<name>/<name>.sqlite``."""
    root = output_dir / name
    (root / "logs").mkdir(parents=True, exist_ok=True)
    sqlite3.connect(root / f"{name}.sqlite").close()
    return root


def make_config(tmp_path: Path, **block) -> Path:
    """A postprocessing settings YAML with this tool's block (``--config`` never guesses)."""
    cfg = tmp_path / "postprocessing.yaml"
    cfg.write_text(yaml.safe_dump({"generate_stl_from_mask": dict(block)}, sort_keys=False),
                   encoding="utf-8")
    return cfg


def mask_in(run_dir: Path, name: str = "a__og-SEG-lamina.png") -> Path:
    """A mask path the way the Reporter writes one, deep inside a run's reports tree."""
    mask = run_dir / "reports" / "Specimen_Masks" / "Leaf_Original" / name
    mask.parent.mkdir(parents=True, exist_ok=True)
    mask.write_bytes(b"")
    return mask


def pipeline_record(run_dir: Path, *, run_id: str, config_path: Path) -> RuntimeRecord:
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
        state=RunState.RUNNING,
        launcher=Launcher.CLI,
        pid=os.getpid(),
        process_started_at=records_mod.process_start_time(),
        started_at=now,
        updated_at=now,
        finished_at=None,
        deployment=records_mod.build_deployment_info(),
        config=ConfigRef(path=str(config_path), sha256="0" * 64),
        project=project,
    )


@contextlib.contextmanager
def holding_the_lease(record: RuntimeRecord) -> Iterator[records_mod.RecordStore]:
    """Publish ``record`` as ``active.json`` while genuinely holding the deployment lease."""
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


class RunSpy:
    """Stand-in for ``generate_stl_from_mask.run``: records the write it WOULD have performed."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, settings=None, paths=None, output_dir=None) -> list[dict]:
        self.calls.append({"settings": settings, "paths": paths, "output_dir": output_dir})
        first = (paths[0] if isinstance(paths, (list, tuple)) else paths) or "x.png"
        return [{"stl": str(Path(first).with_suffix(".stl")), "size_mm": [1.0, 1.0, 1.0],
                 "n_parts": 1, "watertight": True}]

    @property
    def ran(self) -> bool:
        return bool(self.calls)


@pytest.fixture()
def spy(monkeypatch: pytest.MonkeyPatch) -> RunSpy:
    stub = RunSpy()
    monkeypatch.setattr(stl_cli, "run", stub)
    return stub


# --------------------------------------------------------------------------- #
# Rule 1: never write into the run a pipeline owns right now
# --------------------------------------------------------------------------- #
class TestTheActiveRunIsRefused:
    def test_a_mask_inside_the_active_run_is_refused_and_nothing_is_written(
        self, tmp_path: Path, spy: RunSpy, capsys: pytest.CaptureFixture
    ) -> None:
        """Section 2.8 bullet 2, over the CLI: the refusal, and no write at all."""
        live = make_run(tmp_path / "runs", "live")
        mask = mask_in(live)
        cfg = make_config(tmp_path)
        record = pipeline_record(live, run_id="rid-cli-1", config_path=cfg)
        with holding_the_lease(record):
            code = stl_cli.main(["--config", str(cfg), "--paths", str(mask)])
        assert code == stl_cli.EXIT_REFUSED
        assert not spy.ran, "the CLI must refuse BEFORE it writes anything"
        assert "running right now" in capsys.readouterr().err

    def test_an_output_dir_inside_the_active_run_is_refused_too(
        self, tmp_path: Path, spy: RunSpy
    ) -> None:
        """``output_dir`` is one of the tool's ``target_keys``: redirecting the write INTO the
        live run is the same violation as reading out of it."""
        live = make_run(tmp_path / "runs", "live")
        done = make_run(tmp_path / "runs", "done")
        mask = mask_in(done)
        cfg = make_config(tmp_path)
        record = pipeline_record(live, run_id="rid-cli-2", config_path=cfg)
        with holding_the_lease(record):
            code = stl_cli.main(["--config", str(cfg), "--paths", str(mask),
                                 "--output-dir", str(live / "reports" / "STL")])
        assert code == stl_cli.EXIT_REFUSED
        assert not spy.ran

    def test_paths_configured_in_the_yaml_are_guarded_as_well(
        self, tmp_path: Path, spy: RunSpy
    ) -> None:
        """The guard sees the EFFECTIVE targets. A yaml-configured ``paths`` writes into a run
        just as surely as ``--paths`` does, so guarding only the flags would leave a hole."""
        live = make_run(tmp_path / "runs", "live")
        mask = mask_in(live)
        cfg = make_config(tmp_path, paths=[str(mask)])
        record = pipeline_record(live, run_id="rid-cli-3", config_path=cfg)
        with holding_the_lease(record):
            code = stl_cli.main(["--config", str(cfg)])
        assert code == stl_cli.EXIT_REFUSED
        assert not spy.ran

    def test_a_different_completed_run_is_allowed_while_a_pipeline_is_active(
        self, tmp_path: Path, spy: RunSpy
    ) -> None:
        """Section 2.8 bullet 3: this tool is CPU-only, so an unrelated finished run is fair game."""
        runs = tmp_path / "runs"
        live = make_run(runs, "live")
        done = make_run(runs, "finished_yesterday")
        mask = mask_in(done)
        cfg = make_config(tmp_path)
        record = pipeline_record(live, run_id="rid-cli-4", config_path=cfg)
        with holding_the_lease(record):
            code = stl_cli.main(["--config", str(cfg), "--paths", str(mask)])
        assert code == 0
        assert spy.calls and spy.calls[0]["paths"] == [str(mask)]

    def test_a_symlinked_spelling_of_the_active_run_is_still_refused(
        self, tmp_path: Path, spy: RunSpy
    ) -> None:
        """Both sides of the comparison must be resolved or the refusal is trivially evaded: the
        record carries whatever spelling the pipeline was launched with, and a CLI path arrives
        exactly as typed."""
        real = tmp_path / "real_runs"
        live = make_run(real, "live")
        mask = mask_in(live)
        link = tmp_path / "runs_link"
        try:
            link.symlink_to(real, target_is_directory=True)
        except (OSError, NotImplementedError):      # pragma: no cover - unprivileged Windows
            pytest.skip("this filesystem/user cannot create symlinks")
        cfg = make_config(tmp_path)
        record = pipeline_record(live, run_id="rid-cli-5", config_path=cfg)
        with holding_the_lease(record):
            code = stl_cli.main(["--config", str(cfg),
                                 "--paths", str(link / "live" / mask.relative_to(live))])
        assert code == stl_cli.EXIT_REFUSED
        assert not spy.ran


# --------------------------------------------------------------------------- #
# Rule 2: two read_write tools on one completed run serialize
# --------------------------------------------------------------------------- #
class TestTheArtifactLock:
    def test_a_cli_is_refused_while_another_writer_holds_the_run_lock(
        self, tmp_path: Path, spy: RunSpy, capsys: pytest.CaptureFixture
    ) -> None:
        """Section 2.8 bullet 5 over the CLI: the second writer refuses, it does not interleave."""
        done = make_run(tmp_path / "runs", "done")
        mask = mask_in(done)
        cfg = make_config(tmp_path)
        other = postprocess_api.get_tool("generate_leaf_collage")
        held = postprocess_api._acquire_artifact_locks(other, [done])
        assert held, "the fixture must actually hold the lock, or this test proves nothing"
        try:
            code = stl_cli.main(["--config", str(cfg), "--paths", str(mask)])
        finally:
            for lock in held:
                lock.release()
        assert code == stl_cli.EXIT_REFUSED
        assert not spy.ran
        assert "already working on" in capsys.readouterr().err

    def test_the_cli_releases_the_lock_when_it_finishes(self, tmp_path: Path, spy: RunSpy) -> None:
        """A held-forever lock would be worse than no lock: the next run of the tool would refuse."""
        done = make_run(tmp_path / "runs", "done")
        mask = mask_in(done)
        cfg = make_config(tmp_path)
        assert stl_cli.main(["--config", str(cfg), "--paths", str(mask)]) == 0
        after = postprocess_api._acquire_artifact_locks(
            postprocess_api.get_tool("generate_leaf_collage"), [done])
        assert after, "the CLI leaked the artifact lock"
        for lock in after:
            lock.release()

    def test_the_lock_is_released_when_the_tool_itself_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        done = make_run(tmp_path / "runs", "done")
        mask = mask_in(done)
        cfg = make_config(tmp_path)

        def boom(*a, **k):
            raise ValueError("no usable foreground polygon")

        monkeypatch.setattr(stl_cli, "run", boom)
        with pytest.raises(ValueError):
            stl_cli.main(["--config", str(cfg), "--paths", str(mask)])
        after = postprocess_api._acquire_artifact_locks(
            postprocess_api.get_tool("generate_leaf_collage"), [done])
        assert after, "a failing tool must not strand the run's advisory lock"
        for lock in after:
            lock.release()

    def test_the_lock_is_taken_while_the_tool_runs(self, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
        """Held DURING the write, not merely acquired and dropped before it -- otherwise the two
        writers the lock exists to separate would still overlap."""
        done = make_run(tmp_path / "runs", "done")
        mask = mask_in(done)
        cfg = make_config(tmp_path)
        seen: list[bool] = []

        def observe(settings=None, paths=None, output_dir=None):
            with pytest.raises(postprocess_api.TargetLocked):
                postprocess_api._acquire_artifact_locks(
                    postprocess_api.get_tool("generate_leaf_collage"), [done])
            seen.append(True)
            return []

        monkeypatch.setattr(stl_cli, "run", observe)
        assert stl_cli.main(["--config", str(cfg), "--paths", str(mask)]) == 0
        assert seen == [True]


# --------------------------------------------------------------------------- #
# The last bullet: the SAME guard, not a parallel one
# --------------------------------------------------------------------------- #
class TestTheGuardIsShared:
    def test_the_cli_calls_check_target_allowed_itself(
        self, tmp_path: Path, spy: RunSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Substitution test for section 2.8's last bullet. A CLI that grew its own copy of the
        policy would sail past this monkeypatch and fail here."""
        done = make_run(tmp_path / "runs", "done")
        mask = mask_in(done)
        cfg = make_config(tmp_path)
        seen: list[tuple] = []

        def fake(tool, params):
            seen.append((tool.id, dict(params)))
            raise postprocess_api.TargetActive(done, "someone_elses_run", "rid-x")

        monkeypatch.setattr(postprocess_api, "check_target_allowed", fake)
        code = stl_cli.main(["--config", str(cfg), "--paths", str(mask)])
        assert code == stl_cli.EXIT_REFUSED
        assert not spy.ran
        assert seen and seen[0][0] == "generate_stl_from_mask"

    def test_the_params_come_from_the_registry_s_target_keys(
        self, tmp_path: Path, spy: RunSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every key the registry declares as a target is handed to the guard -- resolved, because
        the guard compares against a resolved artifact dir."""
        done = make_run(tmp_path / "runs", "done")
        mask = mask_in(done)
        outdir = tmp_path / "stl_out"
        cfg = make_config(tmp_path)
        tool = postprocess_api.get_tool("generate_stl_from_mask")
        seen: dict = {}
        real = postprocess_api.check_target_allowed

        def spying(t, params):
            seen.update(params)
            return real(t, params)

        monkeypatch.setattr(postprocess_api, "check_target_allowed", spying)
        assert stl_cli.main(["--config", str(cfg), "--paths", str(mask),
                             "--output-dir", str(outdir)]) == 0
        assert set(seen) == set(tool.target_keys) == {"paths", "output_dir"}
        assert seen["paths"] == [os.path.realpath(mask)]
        assert seen["output_dir"] == os.path.realpath(outdir)

    def test_the_cli_takes_its_locks_through_the_shared_helper(
        self, tmp_path: Path, spy: RunSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rule 2 is reached through postprocess_api as well -- not a second lock implementation
        living in the postprocessing package (section 2.8, last bullet)."""
        done = make_run(tmp_path / "runs", "done")
        mask = mask_in(done)
        cfg = make_config(tmp_path)
        calls: list[tuple] = []
        real = postprocess_api._acquire_artifact_locks

        def spying(tool, targets):
            calls.append((tool.id, list(targets)))
            return real(tool, targets)

        monkeypatch.setattr(postprocess_api, "_acquire_artifact_locks", spying)
        assert stl_cli.main(["--config", str(cfg), "--paths", str(mask)]) == 0
        assert calls == [("generate_stl_from_mask", [done])]

    def test_a_registry_target_the_cli_cannot_supply_is_a_loud_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the registry grows a third target key, the guard would cover only part of the write.
        That must break the CLI, not silently narrow the guard."""
        import dataclasses

        tool = postprocess_api.get_tool("generate_stl_from_mask")
        grown = dataclasses.replace(tool, target_keys=("paths", "output_dir", "report_dir"))
        monkeypatch.setattr(postprocess_api, "get_tool", lambda tool_id: grown)
        with pytest.raises(RuntimeError, match="report_dir"):
            with stl_cli.guarded_targets([str(tmp_path / "a.png")], None):
                pass


# --------------------------------------------------------------------------- #
# Step 3's flag rule
# --------------------------------------------------------------------------- #
class TestTheFlagIsOff:
    def test_with_the_flag_off_the_cli_behaves_exactly_as_before(
        self, tmp_path: Path, spy: RunSpy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Step 3: production wiring is inert until the flag is on. Rule 1 could not fire anyway
        (no record is published), and rule 2's lock is a behavior change, so it stays off too."""
        monkeypatch.setenv(FLAG, "0")
        live = make_run(tmp_path / "runs", "live")
        mask = mask_in(live)
        cfg = make_config(tmp_path)
        record = pipeline_record(live, run_id="rid-cli-off", config_path=cfg)
        with holding_the_lease(record):
            code = stl_cli.main(["--config", str(cfg), "--paths", str(mask)])
        assert code == 0
        assert spy.ran, "with the flag off the tool runs exactly as it did before the guard"

    def test_no_lock_is_taken_with_the_flag_off(self, tmp_path: Path, spy: RunSpy,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FLAG, "0")
        done = make_run(tmp_path / "runs", "done")
        mask = mask_in(done)
        cfg = make_config(tmp_path)
        with stl_cli.guarded_targets([str(mask)], None):
            # Nothing is held, so a would-be second writer is not blocked.
            held = postprocess_api._acquire_artifact_locks(
                postprocess_api.get_tool("generate_leaf_collage"), [done])
        assert held == []
        assert stl_cli.main(["--config", str(cfg), "--paths", str(mask)]) == 0


def test_the_refusal_code_is_distinct_from_failure_and_usage() -> None:
    """A script has to tell "refused, retry later" from "this input is broken" (0 ok, 1 raised
    failure, 2 argparse usage)."""
    assert stl_cli.EXIT_REFUSED not in (0, 1, 2)
