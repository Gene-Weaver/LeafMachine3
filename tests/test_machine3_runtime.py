"""``leafmachine3.machine3`` under the Step 3 runtime wiring (plan sections 3.3, 3.4, 2.4, 3.5).

``machine3()`` is the PUBLIC entry point, so the activity is owned there and not in ``main()``:
a lease taken only by the CLI would leave every in-process caller -- a notebook, a test, the
server's legacy job worker -- free to start a second concurrent run on the same GPUs. These tests
drive the real function with the expensive middle stubbed out (hardware profiling, artifact
validation, orphan reaping, ingest and the pipeline itself), so what is exercised is the WIRING:
the order of the section 3.3 insertion point, the records published, the section 3.4 manifest, the
section 2.4 status line, exit code 75, and ``--run-name``.

What each group pins:

* flag off  -- explicit ``LM3_RUNTIME_V2=0`` means the compatibility behavior: no lease, records, or manifest,
  the same calls in the same order, and the same output tree bar nothing at all.
* flag on   -- ``starting`` at acquisition, ``running`` once the DB and log exist, a terminal
  record on the normal path, on an exception and on ``KeyboardInterrupt``.
* contention -- a second root is refused with :class:`RuntimeBusyError`, the loser creates no
  directories (invariant 14), the CLI exits 75 naming the winner, and the status line says ``busy``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import pytest
import yaml

from leafmachine3 import machine3 as m3
from leafmachine3.core import paths
from leafmachine3.core.runtime import execution as ex
from leafmachine3.core.runtime._types import (
    ACTIVE_RECORD_FILENAME,
    LAST_RECORD_FILENAME,
    Activity,
    Launcher,
    RunState,
    RuntimeBusyError,
)

#: Every call the stubs record, in the order the un-wired ``machine3()`` made them. The wiring adds
#: no step and moves none: it only brackets them.
EXPECTED_CALLS = ("hardware", "validate", "reap", "ingest", "build_pipeline", "run_pipeline")


# --------------------------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------------------------- #

@dataclass
class Harness:
    """A private deployment plus a stubbed pipeline body, so one ``machine3()`` costs milliseconds."""

    tmp_path: Path
    monkeypatch: pytest.MonkeyPatch
    calls: list[str] = field(default_factory=list)
    #: name -> callback, so a test can observe the world at one exact point in the run.
    hooks: dict[str, Callable[[], None]] = field(default_factory=dict)

    def settings(self, *, run_name: str = "acer_rubrum", name: str = "LM3_settings.yaml",
                 output: str | None = None) -> Path:
        """A minimal but realistic config. Absolute paths only: nothing here depends on the CWD."""
        data = {
            "project": {
                "run_name": run_name,
                "input": {"dirs": [str(self.tmp_path / "input")]},
                "output": {"dir": output or str(self.tmp_path / "output"), "tmp_dir": "auto"},
            },
            "compute": {"mock": True},
        }
        path = self.tmp_path / name
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    def enable(self) -> None:
        """Turn ``LM3_RUNTIME_V2`` on for this test only."""
        self.monkeypatch.setenv(ex.ENV_RUNTIME_V2, "1")

    # -- observation ----------------------------------------------------------------------------- #

    @property
    def deployment_dir(self) -> Path:
        return paths.deployment_runtime_dir(env=os.environ)

    def active(self) -> dict:
        return json.loads((self.deployment_dir / ACTIVE_RECORD_FILENAME).read_text(encoding="utf-8"))

    def last(self) -> dict:
        return json.loads((self.deployment_dir / LAST_RECORD_FILENAME).read_text(encoding="utf-8"))

    def manifest(self, run_dir: Path) -> dict:
        return json.loads((run_dir / "logs" / "run_manifest.json").read_text(encoding="utf-8"))

    def fire(self, name: str) -> None:
        self.calls.append(name)
        hook = self.hooks.get(name)
        if hook is not None:
            hook()


@pytest.fixture()
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    """Stub the expensive middle of ``machine3()`` and give the test its own deployment.

    ``build_dirs``, ``start_logging``, ``ProjectDB`` and ``Project`` stay REAL -- they are cheap and
    they are exactly what the record's paths and the ``starting -> running`` transition are about.
    """
    h = Harness(tmp_path=tmp_path, monkeypatch=monkeypatch)

    # A deployment of this test's own, under tmp_path. tests/conftest.py has already moved
    # LM3_RUNTIME_DIR out of the developer's real state dir; this narrows it further so two tests
    # can never see each other's active.json.
    monkeypatch.setenv("LM3_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv(ex.ENV_RUNTIME_V2, "0")   # pin the OLD path; unset now means the current one
    for name in (ex.ENV_STATUS_FD, ex.ENV_STATUS_HANDLE):
        monkeypatch.delenv(name, raising=False)

    class _Ingestor:
        def __init__(self, cfg, db) -> None:
            self.cfg, self.db = cfg, db

        def run(self, *, restart: bool = False) -> None:
            h.fire("ingest")

    monkeypatch.setattr(m3, "ensure_hardware_profile", lambda cfg: h.fire("hardware"))
    monkeypatch.setattr(m3, "validate_ml_artifacts", lambda cfg: h.fire("validate"))
    monkeypatch.setattr(m3, "reap_orphaned_workers", lambda: h.fire("reap"))
    monkeypatch.setattr(m3, "ImageIngestor", _Ingestor)
    monkeypatch.setattr(m3, "build_pipeline", lambda cfg: h.fire("build_pipeline") or [])
    monkeypatch.setattr(m3, "run_pipeline", lambda *a, **k: h.fire("run_pipeline"))
    # main() re-execs this interpreter when the CUDA lib path is unset -- which under pytest would
    # restart the test session, not LM3.
    monkeypatch.setattr(m3, "_exec_with_cuda_libpath", lambda: None)

    yield h

    # start_logging() attaches a FileHandler inside tmp_path; drop it so the session does not
    # accumulate open descriptors into directories pytest is about to delete.
    import logging

    for handler in list(logging.getLogger("leafmachine3").handlers):
        logging.getLogger("leafmachine3").removeHandler(handler)
        handler.close()


def _winner(cfg_path: Path):
    """A root activity holding this deployment, resolved from the environment exactly as ``machine3``
    does -- so the contention under test is the real one, not one arranged by passing a path."""
    from leafmachine3.core.config import Config

    return ex.root_activity(Activity.PIPELINE, cfg=Config.load(cfg_path), enabled=True,
                            announce=False, launcher=Launcher.PYTHON)


# --------------------------------------------------------------------------------------------- #
# Explicit flag OFF -- the one-release compatibility behavior, exactly
# --------------------------------------------------------------------------------------------- #

def test_flag_off_runs_the_same_steps_in_the_same_order(harness: Harness) -> None:
    cfg_path = harness.settings()

    project = m3.machine3(cfg_path)

    assert not ex.runtime_v2_enabled()
    assert tuple(harness.calls) == EXPECTED_CALLS
    assert project.dirs.root == harness.tmp_path / "output" / "acer_rubrum"
    assert project.dirs.db_path.exists()


def test_flag_off_acquires_nothing_and_publishes_nothing(harness: Harness) -> None:
    """No lease, no record store, no deployment directory: the wiring is inert, not merely quiet."""
    def refuse(*a, **k):                       # pragma: no cover - the assertion is that it is unused
        raise AssertionError("the lease was attempted with LM3_RUNTIME_V2 off")

    harness.monkeypatch.setattr(ex, "acquire_root_lease", refuse)
    harness.monkeypatch.setattr(ex, "RecordStore", refuse)

    m3.machine3(harness.settings())

    assert not harness.deployment_dir.exists()


def test_flag_off_writes_no_launch_manifest(harness: Harness) -> None:
    project = m3.machine3(harness.settings())
    assert not (project.dirs.logs / "run_manifest.json").exists()


def test_the_flag_adds_exactly_one_file_to_the_run_directory(harness: Harness) -> None:
    """The whole observable difference in the OUTPUT tree is the section 3.4 manifest."""
    off = m3.machine3(harness.settings(run_name="off"))
    harness.enable()
    on = m3.machine3(harness.settings(run_name="on"))

    def tree(root: Path) -> set[str]:
        # The sqlite file is named after the run, so the run name is normalized away: what is
        # under comparison is the SHAPE of the tree, not the two names that had to differ for the
        # two runs to be independent.
        return {str(p.relative_to(root)).replace(f"{root.name}.sqlite", "<run>.sqlite")
                for p in root.rglob("*")}

    assert tree(on.dirs.root) - tree(off.dirs.root) == {"logs/run_manifest.json"}
    assert tree(off.dirs.root) - tree(on.dirs.root) == set()


def test_a_direct_python_caller_still_gets_the_project_back(harness: Harness) -> None:
    """The return value survives the ``with``: six existing tests call ``machine3()`` for it."""
    harness.enable()
    project = m3.machine3(harness.settings())
    assert project.dirs.db_path.exists()
    assert project.cfg.project.run_name == "acer_rubrum"


# --------------------------------------------------------------------------------------------- #
# Flag ON -- the section 3.3 lifecycle
# --------------------------------------------------------------------------------------------- #

def test_starting_is_published_before_hardware_profiling(harness: Harness) -> None:
    """Section 3.3's ordering, observed from inside: the lease is held and the record exists by the
    time the first potentially-expensive step runs."""
    seen: dict[str, object] = {}
    harness.hooks["hardware"] = lambda: seen.update(harness.active())
    harness.enable()

    m3.machine3(harness.settings())

    assert seen["state"] == RunState.STARTING.value
    assert seen["activity"] == Activity.PIPELINE.value
    assert seen["activity_role"] == "root"
    assert seen["launcher"] == Launcher.PYTHON.value
    assert seen["pid"] == os.getpid()


def test_running_is_published_once_the_db_and_log_exist(harness: Harness, caplog) -> None:
    at_pipeline: dict[str, object] = {}
    harness.hooks["run_pipeline"] = lambda: at_pipeline.update(harness.active())
    harness.enable()

    with caplog.at_level("WARNING", logger="leafmachine3.core.runtime.execution"):
        project = m3.machine3(harness.settings())

    assert at_pipeline["state"] == RunState.RUNNING.value
    block = at_pipeline["project"]
    assert isinstance(block, dict)
    # The section 3.5 invariant, made observable one level up: the record names the database and log
    # this run actually created, not a second resolution of the same config.
    assert Path(block["active_db_path"]) == project.dirs.db_path
    assert Path(block["log_path"]) == project.dirs.logs / "lm3.log"
    assert "do(es) not exist yet" not in caplog.text


def test_running_comes_after_ingest_never_before_the_database(harness: Harness) -> None:
    """``starting`` while the DB is still absent, ``running`` only once it is there."""
    states: list[str] = []
    harness.hooks["reap"] = lambda: states.append(harness.active()["state"])
    harness.hooks["ingest"] = lambda: states.append(harness.active()["state"])
    harness.enable()

    m3.machine3(harness.settings())

    assert states == [RunState.STARTING.value, RunState.RUNNING.value]


def test_a_normal_return_finalizes_done_and_frees_the_deployment(harness: Harness) -> None:
    harness.enable()
    m3.machine3(harness.settings())

    assert not (harness.deployment_dir / ACTIVE_RECORD_FILENAME).exists()
    last = harness.last()
    assert last["state"] == RunState.DONE.value
    assert last["returncode"] == 0
    assert last["finished_at"]

    # The lease really was released: a second run succeeds rather than colliding with the first.
    m3.machine3(harness.settings(run_name="second"))
    assert harness.last()["project"]["run_name"] == "second"


def test_an_exception_finalizes_error_and_still_releases_the_lease(harness: Harness) -> None:
    def boom() -> None:
        raise RuntimeError("stage exploded")

    harness.hooks["run_pipeline"] = boom
    harness.enable()

    with pytest.raises(RuntimeError, match="stage exploded"):
        m3.machine3(harness.settings())

    last = harness.last()
    assert last["state"] == RunState.ERROR.value
    assert "stage exploded" in (last["error"] or "")
    assert not (harness.deployment_dir / ACTIVE_RECORD_FILENAME).exists()

    harness.hooks.pop("run_pipeline")
    m3.machine3(harness.settings(run_name="after_error"))     # the deployment is free again


def test_keyboard_interrupt_finalizes_interrupted(harness: Harness) -> None:
    def interrupt() -> None:
        raise KeyboardInterrupt

    harness.hooks["run_pipeline"] = interrupt
    harness.enable()

    with pytest.raises(KeyboardInterrupt):
        m3.machine3(harness.settings())

    assert harness.last()["state"] == RunState.INTERRUPTED.value
    assert not (harness.deployment_dir / ACTIVE_RECORD_FILENAME).exists()


# --------------------------------------------------------------------------------------------- #
# The section 3.4 launch manifest
# --------------------------------------------------------------------------------------------- #

def test_the_manifest_is_written_beside_the_run_log(harness: Harness) -> None:
    harness.enable()
    cfg_path = harness.settings()
    project = m3.machine3(cfg_path, output_dir=str(harness.tmp_path / "elsewhere"))

    manifest = harness.manifest(project.dirs.root)
    assert manifest["run_id"] == harness.last()["run_id"]
    assert manifest["parent_run_id"] is None
    assert manifest["config"]["path"] == str(cfg_path.resolve())
    assert manifest["config"]["sha256"]
    assert manifest["launcher"] == Launcher.PYTHON.value
    assert manifest["project"]["run_name"] == "acer_rubrum"
    # tmp_dir appears HERE and only here (section 2.7): it is knowable only after build_dirs()
    # has actually created it, which is why the manifest is written after that call.
    assert Path(manifest["project"]["tmp_dir"]) == project.dirs.tmp
    assert manifest["effective_config"]["project"]["run_name"] == "acer_rubrum"


def test_the_manifest_records_the_explicit_overrides_as_overrides(harness: Harness) -> None:
    """Section 3.4 wants the overrides kept separate: after the deep merge the provenance is gone."""
    harness.enable()
    project = m3.machine3(harness.settings(), run_name="from_the_flag", restart="all")

    manifest = harness.manifest(project.dirs.root)
    assert manifest["overrides"] == {
        "project": {"run_mode": {"restart": "all"}, "run_name": "from_the_flag"}
    }


def test_a_run_with_no_overrides_records_an_empty_override_tree(harness: Harness) -> None:
    harness.enable()
    project = m3.machine3(harness.settings())
    assert harness.manifest(project.dirs.root)["overrides"] == {}


def test_the_manifest_is_written_once_even_though_build_dirs_runs_again(harness: Harness) -> None:
    """It is immutable: ``build_dirs()`` is called three times in a real run and must not rewrite it."""
    harness.enable()
    project = m3.machine3(harness.settings())
    path = project.dirs.logs / "run_manifest.json"
    before = path.read_bytes()

    m3.build_dirs(project.cfg)
    assert path.read_bytes() == before


# --------------------------------------------------------------------------------------------- #
# Contention: RuntimeBusyError, exit 75, and the losing launch's silence
# --------------------------------------------------------------------------------------------- #

def test_a_second_root_activity_is_refused(harness: Harness) -> None:
    harness.enable()
    winner_cfg = harness.settings(run_name="winner", name="winner.yaml")
    loser_cfg = harness.settings(run_name="loser", name="loser.yaml")

    with _winner(winner_cfg) as winner:
        with pytest.raises(RuntimeBusyError) as excinfo:
            m3.machine3(loser_cfg)

    busy = excinfo.value
    assert busy.exit_code == 75
    assert busy.active is not None and busy.active.run_id == winner.run_id
    assert winner.run_id in str(busy)                       # the message names the winner


def test_the_losing_launch_creates_nothing_under_the_output_dir(harness: Harness) -> None:
    """Invariant 14 from the CLI side: the loser must not have started the run it lost."""
    harness.enable()
    winner_cfg = harness.settings(run_name="winner", name="winner.yaml")
    loser_cfg = harness.settings(run_name="loser", name="loser.yaml")

    with _winner(winner_cfg):
        with pytest.raises(RuntimeBusyError):
            m3.machine3(loser_cfg)

    assert not (harness.tmp_path / "output" / "loser").exists()
    assert harness.calls == []                              # not even hardware profiling ran


def test_main_maps_busy_to_exit_75_and_names_the_winner(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    harness.enable()
    winner_cfg = harness.settings(run_name="winner", name="winner.yaml")
    loser_cfg = harness.settings(run_name="loser", name="loser.yaml")

    with _winner(winner_cfg) as winner:
        code = m3.main(["--config", str(loser_cfg)])

    assert code == 75
    err = capsys.readouterr().err
    assert winner.run_id in err and "busy" in err


def test_main_returns_zero_on_a_normal_run(harness: Harness) -> None:
    harness.enable()
    assert m3.main(["--config", str(harness.settings())]) == 0
    assert harness.last()["launcher"] == Launcher.CLI.value


# --------------------------------------------------------------------------------------------- #
# The section 2.4 status line
# --------------------------------------------------------------------------------------------- #

def _read_line(read_fd: int) -> dict:
    """Drain the pipe to EOF and parse the one line the protocol allows."""
    with os.fdopen(read_fd, "rb") as fh:
        raw = fh.read()
    assert raw.endswith(b"\n"), raw
    return json.loads(raw.decode("utf-8").strip())


def test_acquired_is_announced_when_a_status_channel_is_present(harness: Harness) -> None:
    read_fd, write_fd = os.pipe()
    harness.monkeypatch.setenv(ex.ENV_STATUS_FD, str(write_fd))
    harness.enable()

    m3.machine3(harness.settings())

    message = _read_line(read_fd)                # the channel was closed by the activity -> EOF
    assert message["status"] == "acquired"
    assert message["run_id"] == harness.last()["run_id"]
    assert message["project"]["run_name"] == "acer_rubrum"


def test_busy_is_announced_and_the_cli_exits_75(harness: Harness) -> None:
    """Both halves of section 2.4 step 2's second bullet: the line, then the exit code."""
    harness.enable()
    winner_cfg = harness.settings(run_name="winner", name="winner.yaml")
    loser_cfg = harness.settings(run_name="loser", name="loser.yaml")
    read_fd, write_fd = os.pipe()
    harness.monkeypatch.setenv(ex.ENV_STATUS_FD, str(write_fd))

    with _winner(winner_cfg) as winner:
        code = m3.main(["--config", str(loser_cfg)])

    assert code == 75
    message = _read_line(read_fd)
    assert message["status"] == "busy"
    assert message["active"]["run_id"] == winner.run_id


def test_no_status_channel_is_needed(harness: Harness) -> None:
    """A plain CLI run has no launcher listening; the handshake is skipped, not faked."""
    harness.enable()
    assert m3.main(["--config", str(harness.settings())]) == 0
    assert harness.last()["state"] == RunState.DONE.value


# --------------------------------------------------------------------------------------------- #
# --run-name (Step 3) and the section 3.5 rules it must NOT be subject to
# --------------------------------------------------------------------------------------------- #

def test_run_name_flag_overrides_the_yaml(harness: Harness) -> None:
    cfg_path = harness.settings(run_name="from_yaml")
    assert m3.main(["--config", str(cfg_path), "--run-name", "from_flag"]) == 0
    assert (harness.tmp_path / "output" / "from_flag").is_dir()
    assert not (harness.tmp_path / "output" / "from_yaml").exists()


def test_run_name_is_a_name_not_a_path(harness: Harness) -> None:
    """Rules 2 and 3 absolutize a relative --input/--output against the caller's CWD. A run name is
    joined UNDER output.dir, so the same treatment would silently relocate the run."""
    assert m3._cli_overrides(None, None, None, "acer_rubrum") == {
        "project": {"run_name": "acer_rubrum"}
    }
    for bad in ("../escape", "a/b", "/absolute", "..", ".", "   "):
        with pytest.raises(ValueError, match="single directory name"):
            m3._cli_overrides(None, None, None, bad)


def test_a_path_shaped_run_name_fails_before_anything_is_created(harness: Harness) -> None:
    harness.enable()
    with pytest.raises(ValueError, match="single directory name"):
        m3.machine3(harness.settings(), run_name="../escape")

    assert not harness.deployment_dir.exists()   # refused before the lease, not after it
    assert harness.calls == []


def test_path_overrides_are_still_absolutized_against_the_callers_cwd(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule --run-name is the exception to, kept green here so the two cannot drift apart."""
    elsewhere = harness.tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    overrides = m3._cli_overrides("in_here", "out_there", None, "named")
    assert overrides["project"]["input"]["dirs"] == [str(elsewhere / "in_here")]
    assert overrides["project"]["output"]["dir"] == str(elsewhere / "out_there")
    assert overrides["project"]["run_name"] == "named"


def test_the_parser_accepts_run_name(harness: Harness) -> None:
    cfg_path = harness.settings(run_name="from_yaml")
    seen: dict[str, object] = {}
    harness.monkeypatch.setattr(m3, "machine3", lambda *a, **k: seen.update(k) or None)

    assert m3.main(["--config", str(cfg_path), "--run-name", "renamed"]) == 0
    assert seen["run_name"] == "renamed"
    assert seen["launcher"] is Launcher.CLI


# --------------------------------------------------------------------------------------------- #
# End to end: the Step 3 exit gate for the CLI entry point
# --------------------------------------------------------------------------------------------- #

def test_a_real_cli_subprocess_exits_75_against_a_busy_deployment(harness: Harness) -> None:
    """``python -m leafmachine3.machine3`` -- no stubs, no monkeypatching, a real process.

    This is the CLI half of Step 3's exit gate. The loser exits before ``ensure_hardware_profile``,
    so the whole run costs one interpreter start.
    """
    import subprocess
    import sys

    harness.enable()
    winner_cfg = harness.settings(run_name="winner", name="winner.yaml")
    loser_cfg = harness.settings(run_name="loser", name="loser.yaml")

    env = dict(os.environ)
    env["LM3_CUDA_LIBPATH_SET"] = "1"                # do not re-exec: there is nothing to gain here
    with _winner(winner_cfg) as winner:
        proc = subprocess.run(
            [sys.executable, "-m", "leafmachine3.machine3", "--config", str(loser_cfg)],
            capture_output=True, text=True, env=env, timeout=180,
            cwd=str(Path(__file__).resolve().parent.parent),
        )

    assert proc.returncode == 75, proc.stderr
    assert winner.run_id in proc.stderr
    assert not (harness.tmp_path / "output" / "loser").exists()


def test_the_flag_is_never_re_spelled_in_this_module() -> None:
    """One reader, per the Step 3 brief: ``machine3`` asks ``runtime_v2_enabled()`` or nothing.

    A scattered ``os.environ["LM3_RUNTIME_V2"]`` is how a flag ends up defaulted differently in two
    places and missed in one when it is finally removed.
    """
    lines = Path(m3.__file__).read_text(encoding="utf-8").splitlines()
    mentions = [ln for ln in lines if ex.ENV_RUNTIME_V2 in ln]
    assert mentions, "the flag should still be NAMED here, in the comment explaining the wiring"
    assert all(ln.lstrip().startswith("#") for ln in mentions), mentions


# --------------------------------------------------------------------------------------------- #
# SIGTERM (section 3.3: a server Stop must not leave a record saying "running")
# --------------------------------------------------------------------------------------------- #

def test_sigterm_finalizes_stopped_when_the_caller_opted_in(harness: Harness) -> None:
    import signal

    harness.hooks["run_pipeline"] = lambda: os.kill(os.getpid(), signal.SIGTERM)
    harness.enable()

    with pytest.raises(KeyboardInterrupt):
        m3.machine3(harness.settings(), handle_sigterm=True)

    # ``stopped``, not ``interrupted``: a Stop and a Ctrl-C are different events and the history
    # should say which one happened.
    assert harness.last()["state"] == RunState.STOPPED.value
    assert not (harness.deployment_dir / ACTIVE_RECORD_FILENAME).exists()
    assert signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, signal.default_int_handler)


def test_the_cli_opts_into_sigterm_handling_and_a_library_caller_does_not(harness: Harness) -> None:
    """The asymmetry is the point: ``main()`` owns its process, a direct caller does not."""
    import inspect

    assert inspect.signature(m3.machine3).parameters["handle_sigterm"].default is False

    seen: dict[str, object] = {}
    harness.monkeypatch.setattr(m3, "machine3", lambda *a, **k: seen.update(k) or None)
    m3.main(["--config", str(harness.settings())])
    assert seen["handle_sigterm"] is True
