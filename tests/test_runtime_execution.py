"""``leafmachine3.core.runtime.execution`` -- the Step 3 activity context managers.

Plan sections 2.2 (roots and inherited subactivities), 2.4 (the launch handshake), 2.7 (what a
``starting`` record may publish), 3.3 (lifecycle and finalization ordering) and 3.4 (the immutable
launch manifest), plus Step 3's entry tasks and exit gate.

Several of the properties here are about PROCESSES and cannot be simulated in one: a second root
refused with exit 75, a child that inherits a lease reference and then disarms it, a worker that
inherits nothing, a handshake line that survives a child flooding its stdout first. Those run real
subprocesses driven by ``CHILD_SCRIPT`` below, written into ``tmp_path`` rather than into the repo.

Which gates land here:

* gate 5  -- a root never removes ``active.json`` under a live approved child
* gate 7  -- an executor-style worker never holds the lease reference
* gate 32 -- the approved child disarms inheritance and clears ``LM3_LEASE_*`` before any work
* the Step 3 exit gate -- a second root activity is refused, and the loser exits 75
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from leafmachine3.core.config import Config
from leafmachine3.core.runtime import execution as ex
from leafmachine3.core.runtime import records as rec
from leafmachine3.core.runtime._types import (
    ACTIVE_RECORD_FILENAME,
    CALIBRATION_RUN_NAME,
    ENV_LEASE_CAPABILITY,
    ENV_STATUS_FD,
    ENV_STATUS_HANDLE,
    EXIT_CODE_BUSY,
    LAST_RECORD_FILENAME,
    LEASE_ENV_VARS,
    Activity,
    ArchiveMode,
    Launcher,
    LeaseInheritanceError,
    RunState,
    RuntimeBusyError,
)
from leafmachine3.core.runtime.lease import probe_deployment_occupied
from leafmachine3.core.runtime.launch import LaunchComposeError, LaunchContribution

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX descriptor plumbing")

REPO_ROOT = Path(__file__).resolve().parent.parent
KEY = "pytest-execution"
#: Generous enough for a cold interpreter on a loaded box, short enough that a hang fails the test
#: rather than the session.
SPAWN_TIMEOUT_S = 90.0


# --------------------------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------------------------- #

def write_settings(tmp_path: Path, *, run_name: str = "acer_rubrum", name: str = "LM3_settings.yaml",
                   **output: object) -> Path:
    """A minimal but realistic settings file. Absolute paths, so nothing depends on the CWD."""
    data = {
        "project": {
            "run_name": run_name,
            "input": {"dirs": [str(tmp_path / "input")]},
            "output": {"dir": str(tmp_path / "output"), "tmp_dir": "auto", **output},
        },
        "compute": {"mock": True},
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture()
def deployment(tmp_path: Path) -> Path:
    """A private deployment runtime directory. Never the developer's -- see tests/conftest.py."""
    return tmp_path / "runtime" / KEY


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    return Config.load(write_settings(tmp_path))


def root(deployment_dir: Path, config: Config, activity: Activity = Activity.PIPELINE, **kwargs):
    """A root activity scope with the flag forced on and no handshake, for in-process tests."""
    kwargs.setdefault("enabled", True)
    kwargs.setdefault("announce", False)
    if activity is Activity.HARDWARE_SETUP:
        kwargs.setdefault("hardware_destination", deployment_dir.parent / "hardware.yaml")
    return ex.root_activity(activity, cfg=config, deployment_dir=deployment_dir,
                            deployment_key=KEY, **kwargs)


def active_json(deployment_dir: Path) -> dict:
    return json.loads((deployment_dir / ACTIVE_RECORD_FILENAME).read_text(encoding="utf-8"))


def last_json(deployment_dir: Path) -> dict:
    return json.loads((deployment_dir / LAST_RECORD_FILENAME).read_text(encoding="utf-8"))


def wait_for(path: Path, timeout: float = SPAWN_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"{path} never appeared within {timeout}s")


# --------------------------------------------------------------------------------------------- #
# The subprocess helper. Written into tmp_path, because this file owns no helper module.
# --------------------------------------------------------------------------------------------- #

CHILD_SCRIPT = '''\
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ["LM3_TEST_REPO"])

from leafmachine3.core.config import Config
from leafmachine3.core.runtime import execution as ex
from leafmachine3.core.runtime._types import (
    EXIT_CODE_BUSY,
    LEASE_ENV_VARS,
    Activity,
    RuntimeBusyError,
)

MODE = sys.argv[1]
DEP = Path(os.environ["LM3_TEST_DIR"])
KEY = os.environ["LM3_TEST_KEY"]
REPORT = Path(os.environ["LM3_TEST_REPORT"])


def cfg():
    return Config.load(os.environ["LM3_TEST_SETTINGS"])


def hold():
    """Idle until the test says stop, so the parent can observe a live process."""
    ready = os.environ.get("LM3_TEST_READY")
    if ready:
        Path(ready).write_text("ready", encoding="utf-8")
    stop = Path(os.environ["LM3_TEST_STOP"])
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline and not stop.exists():
        time.sleep(0.01)


if MODE == "root":
    flood = int(os.environ.get("LM3_TEST_FLOOD", "0"))
    if flood:
        # Unbounded startup chatter BEFORE the status line: section 2.4's separate control pipe is
        # what makes this safe, and this is the case the plan says to test exactly.
        sys.stdout.write("x" * flood + "\\n")
        sys.stdout.flush()
    try:
        with ex.root_activity(Activity.PIPELINE, cfg=cfg(), deployment_dir=DEP,
                              deployment_key=KEY, enabled=True,
                              launcher="cli") as handle:
            REPORT.write_text(json.dumps({"result": "acquired", "run_id": handle.run_id}),
                              encoding="utf-8")
            if os.environ.get("LM3_TEST_STOP"):
                hold()
    except RuntimeBusyError as busy:
        REPORT.write_text(json.dumps({
            "result": "busy",
            "winner": busy.active.run_id if busy.active is not None else None,
            "run_dir_created": Path(os.environ["LM3_TEST_RUN_DIR"]).exists(),
        }), encoding="utf-8")
        sys.exit(EXIT_CODE_BUSY)
    sys.exit(0)

if MODE == "child-bind":
    fd = int(os.environ.get("LM3_LEASE_FD", "-1"))
    bound = ex.bind_child_lease(deployment_dir=DEP, deployment_key=KEY)
    data = {
        "run_id": bound.run_id,
        "parent_run_id": bound.parent_run_id,
        "activity": bound.activity.value,
        "fd": fd,
        "inheritable": os.get_inheritable(fd) if fd >= 0 else None,
        "lease_env": {name: os.environ.get(name) for name in LEASE_ENV_VARS},
        "child_env": {name: os.environ.get(name) for name in ex.CHILD_ENV_VARS},
        "grant_consumed": (DEP / "children" / (bound.run_id + ".grant.consumed.json")).exists(),
        "grant_present": (DEP / "children" / (bound.run_id + ".grant.json")).exists(),
    }
    # An ordinary executor-style worker: spawned with no allowlist, so it must inherit nothing.
    worker_report = Path(os.environ["LM3_TEST_WORKER_REPORT"])
    worker_code = (
        "import json,os,sys\\n"
        "fd=int(sys.argv[1]); out={}\\n"
        "try:\\n"
        "    os.fstat(fd); out['fd_open']=True\\n"
        "except OSError:\\n"
        "    out['fd_open']=False\\n"
        "out['env']={n: os.environ.get(n) for n in " + repr(list(LEASE_ENV_VARS)) + "}\\n"
        "open(sys.argv[2],'w').write(json.dumps(out))\\n"
    )
    subprocess.run([sys.executable, "-c", worker_code, str(fd), str(worker_report)], check=True)
    data["worker"] = json.loads(worker_report.read_text(encoding="utf-8"))
    REPORT.write_text(json.dumps(data), encoding="utf-8")
    bound.lease.release()
    sys.exit(0)

if MODE == "child-hold":
    try:
        with ex.child_activity(Activity.CALIBRATION_PIPELINE, cfg=cfg(), deployment_dir=DEP,
                               deployment_key=KEY, enabled=True, announce=False,
                               handle_sigterm=os.environ.get("LM3_TEST_SIGTERM") != "0") as handle:
            handle.mark_running()
            REPORT.write_text(json.dumps({"run_id": handle.run_id}), encoding="utf-8")
            hold()
    except KeyboardInterrupt:
        sys.exit(130)
    sys.exit(0)

raise SystemExit("unknown mode " + MODE)
'''


@pytest.fixture()
def child_script(tmp_path: Path) -> Path:
    path = tmp_path / "exec_child.py"
    path.write_text(CHILD_SCRIPT, encoding="utf-8")
    return path


def child_env(deployment_dir: Path, settings: Path, report: Path, **extra: object) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "LM3_TEST_REPO": str(REPO_ROOT),
        "LM3_TEST_DIR": str(deployment_dir),
        "LM3_TEST_KEY": KEY,
        "LM3_TEST_SETTINGS": str(settings),
        "LM3_TEST_REPORT": str(report),
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONUNBUFFERED": "1",
        ex.ENV_RUNTIME_V2: "1",
    })
    for name, value in extra.items():
        env[name] = str(value)
    return env


# --------------------------------------------------------------------------------------------- #
#: An environment that explicitly pins the pre-Step-3 path. Tests say this rather than passing
#: ``{}``: an empty mapping used to mean "off" only because that was the DEFAULT, so every such test
#: silently changed meaning at the cutover. State the value you are testing.
OFF: dict[str, str] = {"LM3_RUNTIME_V2": "0"}


# --------------------------------------------------------------------------------------------- #
# 1. The feature flag -- one reader, default ON since the cutover (plan section 4, Step 3)
# --------------------------------------------------------------------------------------------- #

@pytest.mark.parametrize("value,expected", [
    # Unset and blank mean "the current runtime". The flag shipped OFF while Steps 3-7 landed so
    # the wiring could arrive without changing behavior; the cutover flipped it once the Step 3
    # exit gate closed. Only an explicit falsey value pins the pre-Step-3 path now.
    (None, True), ("", True), ("   ", True),
    ("0", False), ("no", False), ("off", False), ("false", False), ("FALSE", False),
    ("maybe", False),                       # unrecognized is treated as "the operator meant off"
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), (" on ", True),
])
def test_runtime_v2_enabled_reads_one_variable_and_defaults_on(value, expected) -> None:
    env = {} if value is None else {ex.ENV_RUNTIME_V2: value}
    assert ex.runtime_v2_enabled(env) is expected


def test_runtime_v2_is_on_in_a_clean_ambient_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default ON means the SHIPPED default, not just the default of an empty dict.

    ``tests/conftest.py`` pins the suite's ambient value to ``0`` so no test inherits a default that
    can move underneath it -- so this deletes the variable to ask what a real user's shell gets.
    """
    monkeypatch.delenv(ex.ENV_RUNTIME_V2, raising=False)
    assert ex.runtime_v2_enabled() is True


def test_the_suite_pins_the_flag_rather_than_inheriting_it() -> None:
    """The ambient value every other test runs under is explicit, and it is the OLD path.

    Flag-off tests therefore keep meaning what they say after a default change, and the enabled
    path is covered by the tests that set ``1`` themselves plus gate 60's both-paths comparison.
    """
    assert os.environ.get(ex.ENV_RUNTIME_V2) == "0"


def test_server_surfaces_fail_closed_when_the_runtime_helper_breaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken unified runtime must never be reinterpreted as an operator selecting the old path.

    Before the cutover, the four lazy server readers caught every import/runtime exception and
    returned ``False``. With runtime v2 now the safety default that is the dangerous answer: Start
    could lose its lease while Results resumed mtime discovery and postprocessing lost its active-
    run guard. The compatibility path remains available, but only through explicit
    ``LM3_RUNTIME_V2=0``.
    """
    from leafmachine3.server import metrics_api, postprocess_api, progress_api, results_api

    def broken(*_args, **_kwargs):
        raise RuntimeError("synthetic broken runtime")

    monkeypatch.setattr(ex, "runtime_v2_enabled", broken)
    for reader in (
        metrics_api.runtime_v2,
        postprocess_api._runtime_v2_enabled,
        progress_api._runtime_v2_enabled,
        results_api._runtime_v2,
    ):
        with pytest.raises(RuntimeError, match="synthetic broken runtime"):
            reader()


def test_the_flag_is_read_through_the_helper_and_nowhere_else() -> None:
    """One reader, or the flag is spelled four ways and removed in three of them."""
    package = REPO_ROOT / "leafmachine3"
    offenders = []
    for path in package.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if ex.ENV_RUNTIME_V2 not in text:
            continue
        if path == Path(ex.__file__):
            continue
        for line in text.splitlines():
            if ex.ENV_RUNTIME_V2 in line and ("environ" in line or "getenv" in line):
                offenders.append(f"{path}: {line.strip()}")
    assert not offenders, "read the flag through runtime_v2_enabled(): " + "; ".join(offenders)


def test_with_the_flag_off_a_root_activity_owns_nothing(deployment: Path, cfg: Config,
                                                        tmp_path: Path) -> None:
    """Flag off is byte-identical to today: no lease, no records, no directories, no manifest."""
    ran = False
    with ex.root_activity(Activity.PIPELINE, cfg=cfg, deployment_dir=deployment,
                          deployment_key=KEY, env=OFF, announce=False) as handle:
        ran = True
        assert handle.enabled is False
        assert isinstance(handle, ex.DisabledActivity)
        # Every method is callable and does nothing at all.
        handle.mark_running()
        handle.set_result(RunState.STOPPED)
        assert handle.write_manifest(tmp_dir=tmp_path / "tmp") is None
    assert ran
    assert not deployment.exists(), "a disabled activity created a registry directory"
    assert probe_deployment_occupied(deployment, deployment_key=KEY) is False


def test_with_the_flag_off_a_child_activity_owns_nothing(deployment: Path, cfg: Config) -> None:
    with ex.child_activity(Activity.CALIBRATION_PIPELINE, cfg=cfg, deployment_dir=deployment,
                           deployment_key=KEY, env=OFF, announce=False) as handle:
        assert handle.enabled is False
    assert not deployment.exists()


def test_execution_activity_dispatches_on_the_capability(deployment: Path, cfg: Config) -> None:
    """``machine3`` is both the CLI entry point and the calibration child; the env decides which."""
    with ex.execution_activity(cfg=cfg, deployment_dir=deployment, deployment_key=KEY,
                               env=OFF, announce=False) as handle:
        assert handle.enabled is False          # flag off, root path
    assert ex.is_approved_child({}) is False
    assert ex.is_approved_child({ENV_LEASE_CAPABILITY: "abc"}) is True


# --------------------------------------------------------------------------------------------- #
# 2. Root lifecycle: starting -> running -> each terminal state (plan section 3.3)
# --------------------------------------------------------------------------------------------- #

def test_starting_is_published_at_acquisition_and_running_after_the_paths_exist(
    deployment: Path, cfg: Config, tmp_path: Path
) -> None:
    with root(deployment, cfg) as handle:
        published = active_json(deployment)
        assert published["state"] == RunState.STARTING.value
        assert published["activity"] == "pipeline"
        assert published["activity_role"] == "root"
        assert published["run_id"] == handle.run_id
        assert published["project"]["run_name"] == "acer_rubrum"
        assert published["config"]["path"] == str(cfg.source_path)

        # Section 3.3: `running` once the DB and log paths exist.
        db = Path(published["project"]["active_db_path"])
        log = Path(published["project"]["log_path"])
        db.parent.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        db.write_bytes(b"")
        log.write_text("", encoding="utf-8")
        handle.mark_running()
        assert active_json(deployment)["state"] == RunState.RUNNING.value

    assert not (deployment / ACTIVE_RECORD_FILENAME).exists()
    finished = last_json(deployment)
    assert finished["state"] == RunState.DONE.value
    assert finished["returncode"] == 0
    assert finished["finished_at"]


def test_the_starting_record_never_carries_a_tmp_dir(deployment: Path, cfg: Config) -> None:
    """Section 2.7: ``_ensure_tmp`` can fall back, so no tmp path is knowable before build_dirs."""
    with root(deployment, cfg):
        text = (deployment / ACTIVE_RECORD_FILENAME).read_text(encoding="utf-8")
        payload = json.loads(text)
        assert not [k for k in payload["project"] if "tmp" in k.lower()]
        assert "tmp_dir" not in text


def test_normal_return_finalizes_done_and_releases_the_lease(deployment: Path, cfg: Config) -> None:
    with root(deployment, cfg) as handle:
        assert probe_deployment_occupied(deployment, deployment_key=KEY) is True
        assert handle.enabled is True
    assert last_json(deployment)["state"] == RunState.DONE.value
    assert probe_deployment_occupied(deployment, deployment_key=KEY) is False


def test_an_exception_finalizes_error_and_still_releases_the_lease(deployment: Path,
                                                                   cfg: Config) -> None:
    with pytest.raises(ValueError, match="the pipeline exploded"):
        with root(deployment, cfg):
            raise ValueError("the pipeline exploded")
    record = last_json(deployment)
    assert record["state"] == RunState.ERROR.value
    assert "the pipeline exploded" in record["error"]
    assert record["finished_at"]
    assert probe_deployment_occupied(deployment, deployment_key=KEY) is False


def test_keyboardinterrupt_finalizes_interrupted(deployment: Path, cfg: Config) -> None:
    """A ``KeyboardInterrupt`` is a BaseException: without an explicit clause it escapes uncaught."""
    with pytest.raises(KeyboardInterrupt):
        with root(deployment, cfg):
            raise KeyboardInterrupt
    assert last_json(deployment)["state"] == RunState.INTERRUPTED.value
    assert probe_deployment_occupied(deployment, deployment_key=KEY) is False


def test_a_caller_may_declare_stopped(deployment: Path, cfg: Config) -> None:
    with root(deployment, cfg) as handle:
        handle.set_result(RunState.STOPPED, returncode=143)
    record = last_json(deployment)
    assert record["state"] == RunState.STOPPED.value
    assert record["returncode"] == 143


def test_set_result_refuses_a_non_terminal_state(deployment: Path, cfg: Config) -> None:
    with root(deployment, cfg) as handle:
        with pytest.raises(ValueError, match="not a terminal state"):
            handle.set_result(RunState.RUNNING)


def test_an_exception_outranks_a_declared_outcome(deployment: Path, cfg: Config) -> None:
    """What happened beats what was planned: a failing run must not finalize as ``done``."""
    with pytest.raises(RuntimeError):
        with root(deployment, cfg) as handle:
            handle.set_result(RunState.DONE)
            raise RuntimeError("boom")
    assert last_json(deployment)["state"] == RunState.ERROR.value


def test_a_hardware_setup_root_carries_no_project(deployment: Path, cfg: Config,
                                                  tmp_path: Path) -> None:
    """Invariant 6: tuning the machine is not work on anybody's specimens."""
    destination = tmp_path / "profiles" / "hardware.yaml"
    with root(deployment, cfg, Activity.HARDWARE_SETUP, hardware_destination=destination):
        payload = active_json(deployment)
        assert payload["activity"] == "hardware_setup"
        assert "project" not in payload
        assert payload["hardware"]["destination_path"] == str(destination)
        assert payload["config"]["path"] == str(cfg.source_path)


def test_a_child_activity_may_not_be_started_as_a_root(deployment: Path, cfg: Config) -> None:
    with pytest.raises(ValueError, match="not a root activity"):
        with root(deployment, cfg, Activity.CALIBRATION_PIPELINE):
            pass


def test_a_second_root_in_this_process_is_refused_with_the_winner(deployment: Path,
                                                                  cfg: Config) -> None:
    with root(deployment, cfg) as first:
        with pytest.raises(RuntimeBusyError) as caught:
            with root(deployment, cfg):
                pass
    busy = caught.value
    assert busy.exit_code == EXIT_CODE_BUSY
    assert busy.active is not None and busy.active.run_id == first.run_id
    # And the loser wrote nothing: last.json belongs to the winner's normal finalization only.
    assert last_json(deployment)["run_id"] == first.run_id


# --------------------------------------------------------------------------------------------- #
# 3. The section 3.4 launch manifest
# --------------------------------------------------------------------------------------------- #

def test_the_manifest_lands_beside_the_log_and_is_the_only_place_tmp_dir_appears(
    deployment: Path, cfg: Config, tmp_path: Path
) -> None:
    scratch = tmp_path / "scratch"
    overrides = {"project": {"output": {"dir": str(tmp_path / "output")}}}
    with root(deployment, cfg) as handle:
        path = handle.write_manifest(tmp_dir=scratch, overrides=overrides)
    assert path == tmp_path / "output" / "acer_rubrum" / "logs" / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == handle.run_id
    assert manifest["parent_run_id"] is None
    assert manifest["config"]["path"] == str(cfg.source_path)
    assert manifest["config"]["sha256"]
    assert manifest["overrides"] == overrides
    assert manifest["project"]["tmp_dir"] == str(scratch)
    assert manifest["project"]["archive_mode"] == ArchiveMode.IN_PLACE.value
    assert manifest["effective_config"]["project"]["run_name"] == "acer_rubrum"
    assert manifest["launcher"] == Launcher.PYTHON.value
    assert manifest["versions"]["lm3"] and manifest["started_at"]


def test_a_hardware_setup_root_has_no_manifest(deployment: Path, cfg: Config,
                                               tmp_path: Path) -> None:
    with root(deployment, cfg, Activity.HARDWARE_SETUP) as handle:
        with pytest.raises(ValueError, match="no resolved run paths"):
            handle.write_manifest(tmp_dir=tmp_path)


# --------------------------------------------------------------------------------------------- #
# 4. The section 2.4 handshake -- the child half
# --------------------------------------------------------------------------------------------- #

@posix_only
def test_status_channel_writes_exactly_one_line() -> None:
    read_fd, write_fd = os.pipe()
    try:
        channel = ex.StatusChannel(write_fd, close_fd=False)
        assert channel.send_acquired(run_id="abc") is True
        assert channel.sent is True
        # A second line is REFUSED, never queued: the protocol is exactly one line.
        assert channel.send_busy(None) is False
        os.close(write_fd)
        payload = os.read(read_fd, 65536).decode("utf-8")
    finally:
        os.close(read_fd)
    assert payload.count("\n") == 1
    assert json.loads(payload) == {"status": "acquired", "run_id": "abc"}


@posix_only
def test_status_channel_refuses_a_standard_stream(caplog: pytest.LogCaptureFixture) -> None:
    """Section 2.4: the child's stdout/stderr are never the handshake channel."""
    for fd in (0, 1, 2):
        assert ex.status_channel({ENV_STATUS_FD: str(fd)}) is None
    assert ex.status_channel({ENV_STATUS_FD: "not-a-number"}) is None
    assert ex.status_channel({}) is None


@posix_only
def test_status_channel_ignores_a_descriptor_this_process_does_not_have() -> None:
    """A stale ``LM3_STATUS_FD`` inherited through an unfiltered env must not be written into."""
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    os.close(write_fd)
    assert ex.status_channel({ENV_STATUS_FD: str(write_fd)}) is None


@posix_only
def test_the_windows_branch_opens_the_inherited_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Windows transport is a handle in ``LM3_STATUS_HANDLE``, exercised here with a fake."""
    read_fd, write_fd = os.pipe()
    seen: list[tuple[int, int]] = []

    def fake_open_osfhandle(handle: int, flags: int) -> int:
        seen.append((handle, flags))
        return write_fd

    channel = ex.status_channel({ENV_STATUS_HANDLE: "4242"}, platform_name="win32",
                                open_osfhandle=fake_open_osfhandle)
    assert channel is not None and seen == [(4242, 0)]
    assert channel.send_busy(None) is True
    channel.close()
    payload = os.read(read_fd, 65536).decode("utf-8")
    os.close(read_fd)
    assert json.loads(payload) == {"status": "busy"}


@posix_only
def test_an_oversized_handshake_degrades_to_the_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    read_fd, write_fd = os.pipe()
    try:
        channel = ex.StatusChannel(write_fd, close_fd=False)
        huge = {"status": "acquired", "run_id": "abc", "project": {"run_name": "z" * 200_000}}
        channel._write(huge)  # noqa: SLF001 - the bound is the unit under test
        os.close(write_fd)
        payload = os.read(read_fd, 1 << 20).decode("utf-8")
    finally:
        os.close(read_fd)
    assert json.loads(payload) == {"status": "acquired", "run_id": "abc"}


@posix_only
def test_closing_the_channel_unpublishes_the_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    """fd numbers are recycled: a child that still saw the variable would write into a stranger."""
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv(ENV_STATUS_FD, str(write_fd))
    channel = ex.status_channel()
    assert channel is not None
    channel.send_acquired(run_id="abc")
    channel.close()
    assert ENV_STATUS_FD not in os.environ
    assert os.read(read_fd, 65536)          # the reader still sees the line, then EOF
    assert os.read(read_fd, 65536) == b""
    os.close(read_fd)


def _read_status_line(read_fd: int, timeout: float = SPAWN_TIMEOUT_S) -> dict:
    """Read one newline-terminated JSON line from the control pipe, or fail the test."""
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        chunk = os.read(read_fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    payload = b"".join(chunks)
    assert payload, "the child never wrote a status line"
    return json.loads(payload.splitlines()[0].decode("utf-8"))


@posix_only
@pytest.mark.parametrize("flood", [0, 300_000],
                         ids=["quiet-child", "child-floods-stdout-first"])
def test_the_child_reports_acquired_over_the_control_pipe(tmp_path: Path, deployment: Path,
                                                          child_script: Path, flood: int) -> None:
    """Section 2.4: one JSON line on a dedicated pipe, even after unbounded startup chatter."""
    settings = write_settings(tmp_path)
    report = tmp_path / "child.json"
    stop = tmp_path / "stop"
    console = tmp_path / "console.log"
    read_fd, write_fd = os.pipe()
    env = child_env(deployment, settings, report, LM3_TEST_STOP=stop, LM3_TEST_FLOOD=flood,
                    LM3_STATUS_FD=write_fd, LM3_TEST_RUN_DIR=tmp_path / "output" / "acer_rubrum")
    with console.open("wb") as handle:
        proc = subprocess.Popen(                                      # noqa: S603 - fixed argv
            [sys.executable, str(child_script), "root"], env=env, cwd=str(tmp_path),
            stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            pass_fds=(write_fd,), close_fds=True,
        )
    os.close(write_fd)
    try:
        message = _read_status_line(read_fd)
        assert message["status"] == "acquired"
        assert message["project"]["run_name"] == "acer_rubrum"
        assert message["run_id"]
        # The child is genuinely holding the deployment at this point.
        assert probe_deployment_occupied(deployment, deployment_key=KEY) is True
        assert active_json(deployment)["run_id"] == message["run_id"]
        stop.write_text("stop", encoding="utf-8")
        assert proc.wait(timeout=SPAWN_TIMEOUT_S) == 0
    finally:
        os.close(read_fd)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=SPAWN_TIMEOUT_S)
    assert console.stat().st_size >= flood
    assert last_json(deployment)["state"] == RunState.DONE.value


@posix_only
def test_a_busy_child_reports_the_winner_and_exits_75(tmp_path: Path, deployment: Path,
                                                      child_script: Path, cfg: Config) -> None:
    """The Step 3 exit gate, from the loser's side: 409-worthy identity, exit 75, nothing written."""
    settings = write_settings(tmp_path)
    report = tmp_path / "child.json"
    run_dir = tmp_path / "output" / "acer_rubrum"
    read_fd, write_fd = os.pipe()
    with root(deployment, cfg) as winner:
        env = child_env(deployment, settings, report, LM3_STATUS_FD=write_fd,
                        LM3_TEST_RUN_DIR=run_dir)
        proc = subprocess.Popen(                                      # noqa: S603 - fixed argv
            [sys.executable, str(child_script), "root"], env=env, cwd=str(tmp_path),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            pass_fds=(write_fd,), close_fds=True,
        )
        os.close(write_fd)
        try:
            message = _read_status_line(read_fd)
            returncode = proc.wait(timeout=SPAWN_TIMEOUT_S)
        finally:
            os.close(read_fd)
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=SPAWN_TIMEOUT_S)
        assert message["status"] == "busy"
        assert message["active"]["run_id"] == winner.run_id
        assert message["active"]["activity"] == "pipeline"
        assert returncode == EXIT_CODE_BUSY
        # Invariant 14: the losing launch mutated nothing execution-owned.
        assert json.loads(report.read_text(encoding="utf-8"))["run_dir_created"] is False
        assert not run_dir.exists()
        assert active_json(deployment)["run_id"] == winner.run_id


# --------------------------------------------------------------------------------------------- #
# 5. Subactivities: the parent composes one launch, the child disarms (plan section 2.2)
# --------------------------------------------------------------------------------------------- #

@posix_only
def test_the_launch_is_composed_once_and_carries_every_descriptor(deployment: Path,
                                                                  cfg: Config) -> None:
    """Both the lease handoff and a second contributor inject into ONE exhaustive ``pass_fds``."""
    read_fd, write_fd = os.pipe()
    try:
        extra = LaunchContribution(name="status", env={ENV_STATUS_FD: str(write_fd)},
                                   popen_kwargs={"pass_fds": (write_fd,)})
        with root(deployment, cfg, Activity.HARDWARE_SETUP) as handle:
            with handle.launch_subactivity(run_name=CALIBRATION_RUN_NAME,
                                           contributions=[extra]) as launch:
                fds = launch.popen_kwargs["pass_fds"]
                assert write_fd in fds
                assert len(fds) == 2, "the lease descriptor and the status pipe must both survive"
                assert launch.popen_kwargs["close_fds"] is True
                assert launch.env[ENV_LEASE_CAPABILITY] == launch.capability
                assert launch.env[ex.ENV_CHILD_RUN_ID] == launch.run_id
                assert launch.env[ex.ENV_PARENT_RUN_ID] == handle.run_id
                assert launch.env[ex.ENV_CHILD_ACTIVITY] == "calibration_pipeline"
                assert (deployment / "children" / f"{launch.run_id}.grant.json").exists()
                assert active_json(deployment)["current_child"]["run_id"] == launch.run_id
                launch.completed(returncode=0)
            assert active_json(deployment)["last_child"]["run_id"] == launch.run_id
            assert active_json(deployment).get("current_child") is None
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_the_child_environment_is_filtered_not_copied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Copying ``os.environ`` hands a child an fd NUMBER it does not own (section 2.4)."""
    monkeypatch.setenv(ENV_STATUS_FD, "7")
    monkeypatch.setenv(ENV_STATUS_HANDLE, "77")
    monkeypatch.setenv(ENV_LEASE_CAPABILITY, "secret")
    monkeypatch.setenv(ex.ENV_CHILD_RUN_ID, "stale")
    monkeypatch.setenv("LM3_KEEP_ME", "yes")
    filtered = ex.child_base_env()
    for name in (ENV_STATUS_FD, ENV_STATUS_HANDLE, *LEASE_ENV_VARS, *ex.CHILD_ENV_VARS):
        assert name not in filtered
    assert filtered["LM3_KEEP_ME"] == "yes"
    assert ENV_STATUS_FD in ex.child_base_env(keep_status=True)


@posix_only
def test_a_subactivity_does_not_reach_a_pipeline_root(deployment: Path, cfg: Config) -> None:
    """``CHILD_PARENT_ACTIVITY``: calibration runs under hardware setup and nothing else."""
    with root(deployment, cfg) as handle:
        with pytest.raises(ValueError, match="runs under a hardware_setup root"):
            with handle.launch_subactivity():
                pass


@posix_only
def test_the_child_validates_claims_disarms_and_clears_its_environment(
    tmp_path: Path, deployment: Path, child_script: Path, cfg: Config
) -> None:
    """Section 2.2 step 4 and invariant 3, in one real process tree (gates 22, 32 and 7)."""
    settings = write_settings(tmp_path, run_name=CALIBRATION_RUN_NAME, name="calibration.yaml")
    report = tmp_path / "child.json"
    worker_report = tmp_path / "worker.json"
    with root(deployment, cfg, Activity.HARDWARE_SETUP) as handle:
        with handle.launch_subactivity(run_name=CALIBRATION_RUN_NAME) as launch:
            env = child_env(deployment, settings, report,
                            LM3_TEST_WORKER_REPORT=worker_report)
            env.update(launch.env)
            proc = subprocess.Popen(                                  # noqa: S603 - fixed argv
                [sys.executable, str(child_script), "child-bind"], env=env, cwd=str(tmp_path),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **launch.popen_kwargs,
            )
            launch.attach(proc)
            out, err = proc.communicate(timeout=SPAWN_TIMEOUT_S)
            launch.completed(returncode=proc.returncode)
        assert proc.returncode == 0, f"child failed: {err}"

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["parent_run_id"] == handle.run_id
    assert data["activity"] == "calibration_pipeline"
    # Step 4: the reference is disarmed and the environment is cleared BEFORE any expensive work.
    assert data["inheritable"] is False
    assert data["lease_env"] == {name: None for name in LEASE_ENV_VARS}
    assert data["child_env"] == {name: None for name in ex.CHILD_ENV_VARS}
    # Step 2: consumption is the rename, and a replay finds no source file.
    assert data["grant_consumed"] is True
    assert data["grant_present"] is False
    # Invariant 3: an ordinary worker spawned by the child inherits neither the descriptor nor the
    # environment that would let it find one.
    assert data["worker"]["fd_open"] is False
    assert data["worker"]["env"] == {name: None for name in LEASE_ENV_VARS}


@posix_only
def test_a_subactivity_launch_never_forwards_the_parents_status_descriptor(
    deployment: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hazard from section 2.4: an fd NUMBER the child does not own is worse than no channel."""
    monkeypatch.setenv(ENV_STATUS_FD, "9")
    monkeypatch.setenv(ENV_STATUS_HANDLE, "99")
    with root(deployment, cfg, Activity.HARDWARE_SETUP) as handle:
        with handle.launch_subactivity(run_name=CALIBRATION_RUN_NAME) as launch:
            assert ENV_STATUS_FD not in launch.env
            assert ENV_STATUS_HANDLE not in launch.env
            launch.completed(returncode=0)


@posix_only
def test_a_spawn_that_failed_is_erased_not_promoted_to_last_child(deployment: Path,
                                                                 cfg: Config) -> None:
    """Section 3.2 step 4 promotes ``current_child`` "on normal child completion". A ``Popen``
    that raised is not a completion of any kind, and reporting it as ``done`` would advertise a
    calibration that never measured anything -- the exact opposite of section 2.2's "calibration
    failure must be loud"."""
    with root(deployment, cfg, Activity.HARDWARE_SETUP) as handle:
        with pytest.raises(OSError, match="no such executable"):
            with handle.launch_subactivity(run_name=CALIBRATION_RUN_NAME) as launch:
                child_run_id = launch.run_id
                # Section 3.2 step 1: the summary IS published before the launch, on purpose.
                assert active_json(deployment)["current_child"]["run_id"] == child_run_id
                raise OSError("no such executable")
        assert active_json(deployment).get("current_child") is None
        assert active_json(deployment).get("last_child") is None
    assert handle.finalize_error is None
    assert last_json(deployment)["state"] == RunState.DONE.value
    assert last_json(deployment).get("current_child") is None
    assert last_json(deployment).get("last_child") is None
    # Retention (section 3.2): the one-use grant nothing will ever claim is pruned by the root.
    assert not (deployment / "children" / f"{child_run_id}.grant.json").exists()


@posix_only
def test_a_launch_that_never_composed_leaves_no_phantom_starting_child(
    deployment: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure BETWEEN registration and the body -- a Windows handle duplication that raised, a
    compose conflict -- must not strand a summary in ``starting``. Section 3.3's lifecycle has no
    such resting state, and the root would otherwise write it into its terminal ``last.json``."""
    def refuse(*args: object, **kwargs: object) -> None:
        raise LaunchComposeError("conflicting contributions")

    monkeypatch.setattr(ex, "composed_launch", refuse)
    with root(deployment, cfg, Activity.HARDWARE_SETUP) as handle:
        with pytest.raises(LaunchComposeError):
            with handle.launch_subactivity(run_name=CALIBRATION_RUN_NAME):
                pytest.fail("the body must never run when the launch could not be composed")
        assert active_json(deployment).get("current_child") is None
    assert handle.finalize_error is None
    assert last_json(deployment)["state"] == RunState.DONE.value
    assert last_json(deployment).get("current_child") is None
    assert last_json(deployment).get("last_child") is None
    assert list((deployment / "children").glob("*.grant.json")) == []


@posix_only
def test_a_child_that_wrote_its_own_record_is_promoted_without_a_handle(deployment: Path,
                                                                       cfg: Config) -> None:
    """The discriminator is EXISTENCE of ``children/<run_id>.json``, not its liveness.

    A caller that blocks on the child with ``subprocess.run`` never calls ``attach``, so the root
    holds no handle -- yet that child really ran and finalized its own record, which is no longer
    live. Only the presence of the file separates it from a spawn that never happened."""
    with root(deployment, cfg, Activity.HARDWARE_SETUP) as handle:
        with handle.launch_subactivity(run_name=CALIBRATION_RUN_NAME) as launch:
            (deployment / "children" / f"{launch.run_id}.json").write_text(
                json.dumps({"schema_version": 1, "run_id": launch.run_id, "state": "done"}),
                encoding="utf-8",
            )
        assert active_json(deployment)["last_child"]["run_id"] == launch.run_id
        assert active_json(deployment)["last_child"]["state"] == RunState.DONE.value
        assert active_json(deployment).get("current_child") is None


def test_with_the_flag_off_a_subactivity_launch_is_inert(cfg: Config, deployment: Path) -> None:
    """The caller's Popen still runs; it simply carries no grant, no capability and no lease."""
    with ex.root_activity(Activity.HARDWARE_SETUP, cfg=cfg, deployment_dir=deployment,
                          deployment_key=KEY, env=OFF, announce=False) as handle:
        with ex.launch_subactivity(handle, env={"PATH": "/usr/bin"}) as launch:
            assert launch.run_id == ""
            assert launch.capability == ""
            assert launch.popen_kwargs == {}
            assert launch.env == {"PATH": "/usr/bin"}
            launch.attach(None)
            launch.completed(returncode=0)
    assert not deployment.exists()


def test_bind_child_lease_refuses_a_process_that_is_not_an_approved_child(deployment: Path) -> None:
    with pytest.raises(LeaseInheritanceError, match=ENV_LEASE_CAPABILITY):
        ex.bind_child_lease(deployment_dir=deployment, deployment_key=KEY, env=OFF)
    with pytest.raises(LeaseInheritanceError, match=ex.ENV_CHILD_RUN_ID):
        ex.bind_child_lease(deployment_dir=deployment, deployment_key=KEY,
                            env={ENV_LEASE_CAPABILITY: "x"})


# --------------------------------------------------------------------------------------------- #
# 6. Finalization must not outrun the children (plan section 3.3, gate 5)
# --------------------------------------------------------------------------------------------- #

@posix_only
def test_a_root_refuses_to_finalize_under_a_live_child(tmp_path: Path, deployment: Path,
                                                       child_script: Path, cfg: Config) -> None:
    """``active.json`` is never removed under a live child: the deployment would be occupied but
    unidentifiable, and the GUI would show idle while the GPUs are pinned."""
    settings = write_settings(tmp_path, run_name=CALIBRATION_RUN_NAME, name="calibration.yaml")
    report = tmp_path / "child.json"
    ready = tmp_path / "ready"
    stop = tmp_path / "stop"
    proc = None
    try:
        with root(deployment, cfg, Activity.HARDWARE_SETUP) as handle:
            with handle.launch_subactivity(run_name=CALIBRATION_RUN_NAME) as launch:
                env = child_env(deployment, settings, report, LM3_TEST_READY=ready,
                                LM3_TEST_STOP=stop, LM3_TEST_SIGTERM="0")
                env.update(launch.env)
                proc = subprocess.Popen(                              # noqa: S603 - fixed argv
                    [sys.executable, str(child_script), "child-hold"], env=env, cwd=str(tmp_path),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    **launch.popen_kwargs,
                )
                # Deliberately NOT attached: this is the case where the root has no handle to join,
                # which is exactly when it must refuse to finalize rather than guess.
                wait_for(ready)
            child_run_id = launch.run_id

        assert handle.finalize_error is not None
        assert "subactivit" in str(handle.finalize_error)
        assert (deployment / ACTIVE_RECORD_FILENAME).exists(), "gate 5: the record must stay"
        assert not (deployment / LAST_RECORD_FILENAME).exists()
        assert active_json(deployment)["current_child"]["run_id"] == child_run_id
        # The root released ITS reference, but the child still holds the inherited one, so the
        # deployment stays occupied (invariant 2).
        assert probe_deployment_occupied(deployment, deployment_key=KEY) is True
        child_record = json.loads(
            (deployment / "children" / f"{child_run_id}.json").read_text(encoding="utf-8"))
        assert child_record["state"] == RunState.RUNNING.value
        assert child_record["activity_role"] == "child"
        assert child_record["parent_run_id"] == handle.run_id
    finally:
        stop.write_text("stop", encoding="utf-8")
        if proc is not None:
            proc.wait(timeout=SPAWN_TIMEOUT_S)
    # Once the last reference goes, the deployment is free and the stale root record is left for
    # the next reader to classify as abandoned.
    assert probe_deployment_occupied(deployment, deployment_key=KEY) is False
    snapshot = rec.read_runtime(deployment)
    assert snapshot.classification.value == "abandoned"


@posix_only
def test_a_root_terminates_a_child_it_still_holds_before_finalizing(
    tmp_path: Path, deployment: Path, child_script: Path, cfg: Config
) -> None:
    """With a retained handle the root joins or terminates, then finalizes normally."""
    settings = write_settings(tmp_path, run_name=CALIBRATION_RUN_NAME, name="calibration.yaml")
    report = tmp_path / "child.json"
    ready = tmp_path / "ready"
    stop = tmp_path / "never"
    # A short join timeout: this child is deliberately never told to stop, so the point of the test
    # is the escalation, not the wait.
    with root(deployment, cfg, Activity.HARDWARE_SETUP, children_join_timeout=1.0) as handle:
        with handle.launch_subactivity(run_name=CALIBRATION_RUN_NAME) as launch:
            env = child_env(deployment, settings, report, LM3_TEST_READY=ready, LM3_TEST_STOP=stop)
            env.update(launch.env)
            proc = subprocess.Popen(                                  # noqa: S603 - fixed argv
                [sys.executable, str(child_script), "child-hold"], env=env, cwd=str(tmp_path),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **launch.popen_kwargs,
            )
            launch.attach(proc)
            wait_for(ready)
    assert handle.finalize_error is None
    assert proc.poll() is not None, "the root must not leave its subactivity running"
    assert last_json(deployment)["state"] == RunState.DONE.value
    assert probe_deployment_occupied(deployment, deployment_key=KEY) is False
    # The terminated child finalized its OWN record, so history keeps it as ``last_child``.
    child_record = json.loads(
        (deployment / "children" / f"{launch.run_id}.json").read_text(encoding="utf-8"))
    assert child_record["state"] == RunState.STOPPED.value
    assert last_json(deployment)["last_child"]["run_id"] == launch.run_id


# --------------------------------------------------------------------------------------------- #
# The recovery writer: sections 2.9 and 3.2, gates 50 and 51
# --------------------------------------------------------------------------------------------- #

def test_a_newer_schema_record_is_quarantined_before_a_new_root_publishes(
    deployment: Path, cfg: Config
) -> None:
    """Section 2.9's writer half: the lock is FREE, so preserve the stale record, then start.

    A record from a newer build is evidence we may not interpret. Overwriting it -- which is what
    ``write_active`` alone does, because a foreign ``run_id`` and an incompatible schema both read
    back as "no prior state" -- destroys the only description of what that build was doing.
    """
    deployment.mkdir(parents=True, exist_ok=True)
    seeded = json.dumps(
        {
            "schema_version": rec.SCHEMA_VERSION + 1,
            "run_id": rec.new_run_id(),
            "activity": "pipeline",
            "activity_role": "root",
            "state": "running",
            "a_field_this_build_has_never_heard_of": {"nested": [1, 2, 3]},
        },
        indent=2,
    )
    (deployment / ACTIVE_RECORD_FILENAME).write_text(seeded, encoding="utf-8")

    with root(deployment, cfg) as handle:
        # The new root started, and `active.json` is now ITS record...
        assert active_json(deployment)["run_id"] == handle.run_id
        # ...while the newer build's bytes survive beside it, byte for byte. The real name carries
        # an 8-hex collision suffix after the stamp, hence the glob.
        quarantined = sorted(deployment.glob("active.incompatible.*.json"))
        assert len(quarantined) == 1, f"expected exactly one quarantine file, got {quarantined}"
        assert quarantined[0].read_text(encoding="utf-8") == seeded


def test_a_stale_root_record_is_finalized_as_abandoned_before_the_next_root_starts(
    deployment: Path, cfg: Config
) -> None:
    """Section 3.3: a record found while the lease is acquirable is abandoned, and reaches history.

    Without the recovery transaction the hard-killed run is simply overwritten and vanishes from
    ``last.json`` -- the deployment then claims its last run was the NEW one.
    """
    with root(deployment, cfg) as first:
        stale = json.loads((deployment / ACTIVE_RECORD_FILENAME).read_text(encoding="utf-8"))
        assert stale["run_id"] == first.run_id

    # Replay that record as a foreign run that died without finalizing: same shape, another run_id,
    # a non-terminal state, and no live process behind it.
    stale["run_id"] = rec.new_run_id()
    stale["state"] = RunState.RUNNING.value
    (deployment / ACTIVE_RECORD_FILENAME).write_text(json.dumps(stale), encoding="utf-8")

    with root(deployment, cfg) as second:
        assert second.run_id != stale["run_id"]
        assert active_json(deployment)["run_id"] == second.run_id
        history = last_json(deployment)
        assert history["run_id"] == stale["run_id"], "the abandoned run never reached history"
        assert history["state"] == RunState.INTERRUPTED.value
        assert "abandoned" in (history["error"] or "")
        assert history["finished_at"]
