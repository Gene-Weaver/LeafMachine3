"""The SERVER half of the section 2.4 launch handshake, and section 2.13's setup subprocess.

Plan reference: ``UNIFIED_RUNTIME_IMPLEMENTATION_PLAN_CONSENSUS.md`` -- section 2.4 (launch
handshake and the staging boundary), section 2.5 (control authority is handle ownership), section
2.13 (hardware setup requires a resolved config, and must not run inside the server), section 3.3
(lifecycle and finalization ordering) and section 4, Step 3.

What is being pinned, in the plan's own words:

* "Server reads with a bounded timeout. ``acquired`` -> 200 with the runtime identity; ``busy`` ->
  409 with the winner's record."
* "On timeout, malformed status, oversized status, or EOF the server must not simply return 500 and
  walk away" -- it terminates the retained tree, escalates, reconciles any record the child already
  published, and returns 500 only once no managed child remains.
* "Only after ``acquired`` may execution-owned directories be created", stated testably as
  invariant 14: a losing launch may write under the server-private jobs root and must NOT create
  anything under ``<output.dir>/<run_name>/``.
* "The child's stdout/stderr go to a server-private request log under the jobs root from the moment
  of ``Popen``" -- so a child may emit unbounded startup chatter before its status line without
  filling a pipe the server is not draining. That case is tested with a REAL subprocess, because a
  fake cannot deadlock and therefore cannot prove the absence of one.
* Section 2.13: GUI hardware setup runs as an ``lm3-setup`` subprocess with a retained handle, its
  progress replayable from an append-only JSONL log, and Stop kills that tree and leaves the server
  serving.

Everything is behind ``LM3_RUNTIME_V2``. The first test in the file is the flag-OFF regression: with
the flag explicitly set to ``0`` the launch path must be exactly what it was before Step 3.

Nothing here binds a port, loads a model, touches a GPU, or reads the developer's real runtime
state. The handful of tests that use a real subprocess run a generated 20-line Python script.
"""
from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from tests._contract_helpers import (
    Sandbox,
    bearer,
    isolate_server_paths,
    reset_server_module_state,
)

#: Short enough that the timeout branch is a fast test, long enough that a loaded CI box does not
#: trip it on the happy path.
FAST_TIMEOUT_S = "0.75"


# --------------------------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------------------------- #
@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Sandbox]:
    """A tmp-dir world plus a clean slate of the server's module globals, before AND after."""
    reset_server_module_state()
    box = isolate_server_paths(tmp_path, monkeypatch)
    try:
        yield box
    finally:
        reset_server_module_state()


@pytest.fixture
def quiet_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sampler is a process-wide daemon thread that reads /proc. No handshake test wants it."""
    from leafmachine3.server import metrics_api

    monkeypatch.setattr(metrics_api.metrics, "reset_baseline", lambda: {}, raising=True)
    monkeypatch.setattr(metrics_api.metrics, "set_worker_root", lambda pid: None, raising=True)


@pytest.fixture
def v2(monkeypatch: pytest.MonkeyPatch, quiet_metrics: None) -> None:
    """Turn Step 3's wiring on for this test, and bound the handshake so failures are quick."""
    monkeypatch.setenv("LM3_RUNTIME_V2", "1")
    monkeypatch.setenv("LM3_HANDSHAKE_TIMEOUT_S", FAST_TIMEOUT_S)


@pytest.fixture
def client(sandbox: Sandbox) -> Iterator[TestClient]:
    """A TestClient over a freshly built app whose JobManager lives inside the sandbox.

    Deliberately NOT entered as a context manager: ``__enter__`` runs the lifespan, which starts the
    job worker and the owner watchdog. These tests want the routing table and the handlers.
    """
    from leafmachine3.server.app import JobManager, create_app

    app = create_app(JobManager(sandbox.jobs_root / "managed"))
    yield TestClient(app)


def start_a_run(client: TestClient, sandbox: Sandbox, **body: Any) -> Any:
    payload: dict[str, Any] = {"config_path": str(sandbox.settings_path)}
    payload.update(body)
    return client.post("/v1/run/start", json=payload, headers=bearer())


# --------------------------------------------------------------------------------------------- #
# A fake machine3 that answers on the status channel
# --------------------------------------------------------------------------------------------- #
class _FakeChild:
    """``Popen``'s shape, minus the process: alive until :meth:`release`."""

    #: Well outside ``/proc/sys/kernel/pid_max`` on Linux, so nothing real can ever wear it.
    FAKE_PID = 2 ** 31 - 11

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.argv = list(argv)
        self.kwargs = dict(kwargs)
        self.pid = self.FAKE_PID
        self.returncode: int | None = None
        self._done = threading.Event()

    def poll(self) -> int | None:
        return self.returncode if self._done.is_set() else None

    def wait(self, timeout: float | None = None) -> int:
        self._done.wait(timeout)
        return self.returncode if self.returncode is not None else 0

    def release(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self._done.set()


class _FakeSpawn:
    """The slice of :mod:`subprocess` the managed launch uses, wired to a scripted handshake.

    ``reply`` is the exact bytes the "child" writes to the status descriptor it was passed --
    ``None`` means it writes nothing, which is what makes the EOF and timeout branches reachable.
    ``hold`` keeps a duplicate of the write end open, so "wrote nothing AND is still running" is
    distinguishable from "exited"; without it every silent child would be an instant EOF.
    """

    STDOUT = -2
    DEVNULL = -3

    def __init__(self, *, reply: bytes | None = None, hold: bool = False,
                 chatter: int = 0) -> None:
        self.reply = reply
        self.hold = hold
        self.chatter = chatter
        self.calls: list[_FakeChild] = []
        self.held_fds: list[int] = []

    def Popen(self, argv: list[str], **kwargs: Any) -> _FakeChild:   # noqa: N802 - mirrors the API
        fds = kwargs.get("pass_fds") or ()
        stdout = kwargs.get("stdout")
        if self.chatter and hasattr(stdout, "write"):
            stdout.write("startup chatter\n" * self.chatter)
            stdout.flush()
        for fd in fds:
            # A dup, because that is what a real child gets: the parent closes ITS copy the moment
            # the launch is composed, and the write must survive that.
            dup = os.dup(fd)
            if self.reply is not None:
                # From a THREAD, not inline: an oversized line is bigger than the pipe buffer and
                # the parent has not started reading yet, so a synchronous write here would wedge
                # the launch -- exactly as it would inside a real child with no reader.
                threading.Thread(target=self._emit, args=(dup,), daemon=True).start()
            elif self.hold:
                self.held_fds.append(dup)                 # alive, and deliberately silent
            else:
                os.close(dup)                             # exited without writing -> EOF
        child = _FakeChild(argv, **kwargs)
        self.calls.append(child)
        return child

    def _emit(self, fd: int) -> None:
        try:
            os.write(fd, self.reply or b"")
        except OSError:                                   # the reader gave up (oversized/timeout)
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def release_all(self, returncode: int = 0) -> None:
        for fd in self.held_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.held_fds.clear()
        for child in self.calls:
            child.release(returncode)


@pytest.fixture
def fake_spawn(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Install a scripted ``subprocess`` for metrics_api, and make termination instantaneous.

    ``_ManagedChild.signal_group`` is stubbed for the same reason the contract tests stub
    ``_signal_group``: the stand-in child has a synthetic PID no signal can reach, so the real
    escalation would sit out its full SIGTERM grace. Releasing the child from inside the signal is
    what a real, well-behaved LM3 does.
    """
    from leafmachine3.server import metrics_api

    box: dict[str, Any] = {}

    def install(**kwargs: Any) -> _FakeSpawn:
        fake = _FakeSpawn(**kwargs)
        box["fake"] = fake
        monkeypatch.setattr(metrics_api, "subprocess", fake)
        return fake

    signals: list[int] = []

    def _fake_signal(self: Any, sig: int) -> bool:
        signals.append(int(sig))
        fake = box.get("fake")
        if fake is not None:
            fake.release_all(returncode=-int(sig))
        return True

    monkeypatch.setattr(metrics_api._ManagedChild, "signal_group", _fake_signal, raising=True)
    install.signals = signals                                        # type: ignore[attr-defined]
    install.box = box                                                # type: ignore[attr-defined]
    try:
        yield install
    finally:
        fake = box.get("fake")
        if fake is not None:
            fake.release_all()
        for thread in threading.enumerate():
            if thread.name in ("lm3-run-reaper", "lm3-launch-handshake"):
                thread.join(5.0)


def acquired_line(run_id: str = "11111111-1111-4111-8111-111111111111") -> bytes:
    return (json.dumps({"status": "acquired", "run_id": run_id,
                        "project": {"run_name": "contract_run"}}) + "\n").encode()


def busy_line(run_id: str = "99999999-9999-4999-8999-999999999999") -> bytes:
    winner = {"schema_version": 1, "run_id": run_id, "activity": "pipeline",
              "state": "running", "project": {"run_name": "somebody_elses_run"}}
    return (json.dumps({"status": "busy", "active": winner}) + "\n").encode()


# --------------------------------------------------------------------------------------------- #
# A REAL child, for the properties a fake cannot prove
# --------------------------------------------------------------------------------------------- #
_CHILD_SCRIPT = '''#!{python}
"""A stand-in for machine3 that speaks the section 2.4 child protocol."""
import json, os, sys

CHATTER_LINES = {chatter}
STATUS = {status!r}
EXIT = {exit_code}

# Unbounded startup chatter FIRST: this is the case section 2.4 says to test, and it deadlocks
# instantly if the server ever wires stdout to a pipe it is not draining.
for _ in range(CHATTER_LINES):
    sys.stdout.write("loading models, please wait ................................\\n")
sys.stdout.flush()

fd = os.environ.get("LM3_STATUS_FD")
if fd and STATUS:
    payload = {{"status": STATUS, "run_id": "22222222-2222-4222-8222-222222222222"}}
    if STATUS == "busy":
        payload["active"] = {{"schema_version": 1, "run_id": "abc", "state": "running"}}
    os.write(int(fd), (json.dumps(payload) + "\\n").encode())
    os.close(int(fd))
sys.exit(EXIT)
'''


def write_child_script(path: Path, *, status: str = "acquired", chatter: int = 0,
                       exit_code: int = 0) -> Path:
    path.write_text(_CHILD_SCRIPT.format(python=sys.executable, chatter=chatter,
                                         status=status, exit_code=exit_code), encoding="utf-8")
    path.chmod(0o755)
    return path


# --------------------------------------------------------------------------------------------- #
# 1. Explicit flag-off compatibility remains unchanged
# --------------------------------------------------------------------------------------------- #
def test_with_the_flag_off_the_launch_path_is_exactly_what_it_was(
    client: TestClient, sandbox: Sandbox, fake_spawn: Any, monkeypatch: pytest.MonkeyPatch,
    quiet_metrics: None,
) -> None:
    """Explicit ``LM3_RUNTIME_V2=0`` means no status pipe and the pre-Step-3 directories.

    This is the regression that matters most while the flag exists -- if the off path drifts, the
    "we can turn it off" safety net is not one.
    """
    monkeypatch.setenv("LM3_RUNTIME_V2", "0")
    fake = fake_spawn()

    assert start_a_run(client, sandbox).status_code == 200

    kwargs = fake.calls[0].kwargs
    assert "pass_fds" not in kwargs, "the handshake pipe must not exist with the flag off"
    assert "LM3_STATUS_FD" not in kwargs["env"]
    assert (sandbox.run_dir / "logs" / "console.log").is_file(), "the old console log location"


# --------------------------------------------------------------------------------------------- #
# 2. acquired -> 200 with the runtime identity
# --------------------------------------------------------------------------------------------- #
def test_acquired_answers_200_with_the_runtime_identity(
    client: TestClient, sandbox: Sandbox, fake_spawn: Any, v2: None,
) -> None:
    """Section 2.4 step 3: "``acquired`` -> 200 with the runtime identity"."""
    fake = fake_spawn(reply=acquired_line())

    resp = start_a_run(client, sandbox)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["run_id"] == "11111111-1111-4111-8111-111111111111"
    assert body["active"] is True
    assert body["run_name"] == sandbox.run_name
    # The identity is added to the START response only: GET /v1/run/active's key set is a published
    # contract its own docstring promises never changes, and the registry view is Step 4's.
    assert "run_id" not in client.get("/v1/run/active", headers=bearer()).json()
    assert fake.calls, "a child must actually have been launched"


def test_the_child_is_given_a_status_descriptor_and_a_filtered_environment(
    client: TestClient, sandbox: Sandbox, fake_spawn: Any, v2: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The POSIX half of section 2.4's transport table, plus the environment filtering it needs.

    ``LM3_STATUS_FD`` names a descriptor NUMBER. Copying ``os.environ`` wholesale would hand the
    child the SERVER's number, which in the child belongs to somebody else -- so the child env is
    built by ``child_base_env`` and the only status variable in it is the one this launch created.
    """
    monkeypatch.setenv("LM3_STATUS_FD", "3")            # the server's own, which must NOT travel
    monkeypatch.setenv("LM3_LEASE_CAPABILITY", "not-the-childs")
    fake = fake_spawn(reply=acquired_line())

    assert start_a_run(client, sandbox).status_code == 200

    kwargs = fake.calls[0].kwargs
    env = kwargs["env"]
    assert kwargs["pass_fds"], "the write end must be in the exhaustive pass_fds allowlist"
    assert env["LM3_STATUS_FD"] == str(kwargs["pass_fds"][0]) != "3"
    assert "LM3_LEASE_CAPABILITY" not in env, "lease variables come from a handoff or not at all"
    assert kwargs["close_fds"] is True                  # invariant 3: nothing else is inherited
    assert kwargs["start_new_session"] is True          # section 3.3: Stop targets the group
    assert "LM3_SERVER_TOKEN" not in env
    assert "LM3_CUDA_LIBPATH_SET" not in env


def test_the_child_console_goes_to_a_server_private_request_log(
    client: TestClient, sandbox: Sandbox, fake_spawn: Any, v2: None,
) -> None:
    """Section 2.4: stdout/stderr to "a server-private request log under the jobs root" from Popen.

    Not to the run directory (that is execution-owned, and this launch may yet lose) and not to a
    pipe (which the server is not draining while it waits on the handshake).
    """
    fake = fake_spawn(reply=acquired_line())

    body = start_a_run(client, sandbox).json()

    console = Path(body["console_log"])
    assert console.is_file()
    assert sandbox.jobs_root in console.parents, f"{console} is not under the jobs root"
    assert sandbox.run_dir not in console.parents
    assert Path(fake.calls[0].kwargs["stdout"].name) == console
    assert fake.calls[0].kwargs["stderr"] == _FakeSpawn.STDOUT


# --------------------------------------------------------------------------------------------- #
# 3. busy -> 409 with the winner, and no execution-owned mutation
# --------------------------------------------------------------------------------------------- #
def test_busy_answers_409_with_the_winners_record(
    client: TestClient, sandbox: Sandbox, fake_spawn: Any, v2: None,
) -> None:
    """Section 2.4 step 3: "``busy`` -> 409 with the winner's record".

    Today this is unimplementable: ``start_run`` returns before the child has said anything, so an
    exit 75 cannot influence the response that was already sent.
    """
    fake_spawn(reply=busy_line())

    resp = start_a_run(client, sandbox)

    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["reason"] == "busy"
    assert detail["active"]["run_id"] == "99999999-9999-4999-8999-999999999999"
    assert detail["active"]["project"]["run_name"] == "somebody_elses_run"
    assert "another root activity" in detail["message"]


def test_a_losing_launch_creates_nothing_under_the_run_directory(
    client: TestClient, sandbox: Sandbox, fake_spawn: Any, v2: None,
) -> None:
    """Invariant 14, in section 2.4's own terms.

    Prohibited for a losing launch: "anything under ``<output.dir>/<run_name>/``, the project DB,
    the run's ``logs/``". Permitted: writes under the server-private jobs root -- which is where
    the request log for this very refusal lives, and it is still there afterwards.
    """
    from leafmachine3.server.metrics_api import execution_owned_paths

    prohibited = execution_owned_paths(sandbox.run_dir, sandbox.run_name)
    assert not any(p.exists() for p in prohibited)
    fake_spawn(reply=busy_line())

    resp = start_a_run(client, sandbox)

    assert resp.status_code == 409
    existing = [str(p) for p in prohibited if p.exists()]
    assert existing == [], f"a losing launch created execution-owned paths: {existing}"
    request_log = Path(resp.json()["detail"]["request_log"])
    assert request_log.is_file() and sandbox.jobs_root in request_log.parents


# --------------------------------------------------------------------------------------------- #
# 4. Every failure branch terminates the retained child before it answers 500
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "kwargs", "fragment"),
    [
        ("timeout", {"reply": None, "hold": True}, "timeout"),
        ("eof", {"reply": None}, "eof"),
        ("malformed", {"reply": b"this is not json\n"}, "malformed"),
        ("oversized", {"reply": b"x" * (64 * 1024 + 64) + b"\n"}, "oversized"),
    ],
)
def test_a_failed_handshake_terminates_the_child_before_returning_500(
    client: TestClient, sandbox: Sandbox, fake_spawn: Any, v2: None,
    name: str, kwargs: dict, fragment: str,
) -> None:
    """Section 2.4 step 3: the server "must not simply return 500 and walk away".

    A slow child could acquire the lease and run on after the API reported failure, leaving a run
    nobody launched -- so the retained tree is terminated and escalated FIRST, and 500 is returned
    only once no managed child remains. The ``timeout`` case is the plan's named regression test:
    a child that holds the channel open and never writes.
    """
    fake = fake_spawn(**kwargs)

    resp = start_a_run(client, sandbox)

    assert resp.status_code == 500, resp.text
    assert fragment in resp.json()["detail"]
    assert fake.calls, "the child was launched"
    assert fake.calls[0].poll() is not None, "the managed child must be gone before the 500"
    assert fake_spawn.signals, "a failed handshake must signal the retained process group"
    # And the refusal is not the start of a run: nothing is left active.
    assert client.get("/v1/run/active", headers=bearer()).json()["active"] is False


def test_a_timed_out_child_that_already_published_a_record_is_reconciled(
    client: TestClient, sandbox: Sandbox, fake_spawn: Any, v2: None, tmp_path: Path,
) -> None:
    """Section 2.4 step 3 (iii): "reconciles any runtime record the child may already have published".

    The named regression: a child that DOES acquire the lease but delays its status line past the
    timeout. Its ``active.json`` would otherwise sit there forever making the deployment look
    occupied; the server takes the section 3.3 cleanup lock (which it can only take because the
    child is really gone) and lets ``recover_abandoned`` finalize it.
    """
    from leafmachine3.core import paths
    from leafmachine3.core.runtime import _types as T
    from leafmachine3.core.runtime import records as R

    deployment_dir = paths.deployment_runtime_dir()
    (deployment_dir / T.CHILDREN_DIRNAME).mkdir(parents=True, exist_ok=True)
    run_id = "33333333-3333-4333-8333-333333333333"
    record = T.RuntimeRecord(
        run_id=run_id, activity=T.Activity.PIPELINE, activity_role=T.ActivityRole.ROOT,
        state=T.RunState.RUNNING, launcher=T.Launcher.SERVER, pid=4321,
        process_started_at=1787932800.25,
        started_at="2026-08-28T12:00:00Z", updated_at="2026-08-28T12:00:00Z",
        deployment=T.DeploymentInfo(id=paths.deployment_key()),
        config=T.ConfigRef(path=str(sandbox.settings_path), sha256="a" * 64),
        project=T.ProjectBlock(
            run_name=sandbox.run_name,
            input_dirs=(str(sandbox.input_dir),),
            artifact_dir=str(sandbox.run_dir),
            active_state_dir=str(sandbox.run_dir),
            active_db_path=str(sandbox.run_dir / f"{sandbox.run_name}.sqlite"),
            archive_mode=T.ArchiveMode.IN_PLACE,
            archive_status=T.ArchiveStatus.NOT_APPLICABLE,
            archive_pointer_path=None,
            archived_db_path=str(sandbox.run_dir / f"{sandbox.run_name}.sqlite"),
            run_dir=str(sandbox.run_dir),
            log_path=str(sandbox.run_dir / "logs" / "lm3.log"),
        ),
    )
    R.RecordStore(deployment_dir, run_id=run_id).write_active(record)
    assert (deployment_dir / T.ACTIVE_RECORD_FILENAME).is_file()
    fake_spawn(reply=None, hold=True)

    resp = start_a_run(client, sandbox)

    assert resp.status_code == 500
    assert not (deployment_dir / T.ACTIVE_RECORD_FILENAME).exists(), (
        "the abandoned record must be reconciled, not left describing a run that is gone")
    last = json.loads((deployment_dir / T.LAST_RECORD_FILENAME).read_text(encoding="utf-8"))
    assert last["run_id"] == run_id
    assert last["state"] in {s.value for s in T.RunState if s.value != "running"}


# --------------------------------------------------------------------------------------------- #
# 5. The transport itself: composition, and the Windows half
# --------------------------------------------------------------------------------------------- #
def test_the_status_pipe_and_a_lease_handoff_compose_into_one_allowlist() -> None:
    """Step 3 entry task 3: ``pass_fds`` is EXHAUSTIVE, so both contributors must be composed.

    Supplied independently, the second silently replaces the first and the child starts without a
    reference it believes it holds.
    """
    from leafmachine3.core.runtime.launch import LaunchContribution, compose_launch
    from leafmachine3.server.metrics_api import _LaunchStatusPipe

    lease_r, lease_w = os.pipe()
    pipe = _LaunchStatusPipe()
    try:
        lease = LaunchContribution(name="lease", env={"LM3_LEASE_FD": str(lease_w)},
                                   popen_kwargs={"pass_fds": (lease_w,)})
        composed = compose_launch([lease, pipe.contribution()], base_env={})

        assert set(composed.popen_kwargs["pass_fds"]) == {lease_w, pipe.child_value}
        assert composed.env["LM3_STATUS_FD"] == str(pipe.child_value)
        assert composed.env["LM3_LEASE_FD"] == str(lease_w)
    finally:
        pipe.close()
        for fd in (lease_r, lease_w):
            try:
                os.close(fd)
            except OSError:
                pass


def test_the_windows_transport_names_the_handle_and_the_startupinfo_allowlist() -> None:
    """The Windows half of section 2.4's table, exercised on Linux through the injected factory.

    ``pass_fds`` does not exist on Windows; the write end travels as an inheritable handle in the
    ``STARTUPINFOEX`` allowlist, and its VALUE goes into ``LM3_STATUS_HANDLE``. The platform half
    that is never run is the platform half that is wrong, so it is run here.
    """
    from leafmachine3.core.runtime import _win32
    from leafmachine3.server.metrics_api import _LaunchStatusPipe

    read_fd, write_fd = os.pipe()
    fake_handle = 0xF00D
    closed: list[bool] = []

    def factory() -> tuple:
        return (read_fd, fake_handle,
                # No platform_name: off Windows the helper hands back HandleAllowlist, whose one
                # attribute mirrors subprocess.STARTUPINFO exactly, so this assertion is the
                # assertion that holds against the real object over there.
                _win32.handle_list_popen_kwargs([fake_handle]),
                lambda: closed.append(True))

    pipe = _LaunchStatusPipe(platform_name="win32", factory=factory)
    try:
        contribution = pipe.contribution()
        assert contribution.env == {"LM3_STATUS_HANDLE": str(fake_handle)}
        handles = contribution.popen_kwargs["startupinfo"].lpAttributeList["handle_list"]
        assert list(handles) == [fake_handle]

        os.write(write_fd, acquired_line())
        os.close(write_fd)
        assert pipe.read(2.0).outcome == "acquired"
        assert closed, "the child's handle must be released in the parent after the launch"
    finally:
        pipe.close()


# --------------------------------------------------------------------------------------------- #
# 6. A real child: unbounded chatter, and a real exit 75
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("chatter", [0, 20000])
def test_a_real_child_completes_the_handshake_after_unbounded_startup_chatter(
    client: TestClient, sandbox: Sandbox, tmp_path: Path, v2: None,
    monkeypatch: pytest.MonkeyPatch, chatter: int,
) -> None:
    """Section 2.4: "a child can emit unbounded startup chatter before it sends its status line
    without ever filling a pipe. Test exactly that case."

    ~1.2 MB of stdout is far past any pipe buffer, so if the console were ever wired to a pipe the
    server is not draining, this hangs. A fake cannot deadlock, so this one is a real process --
    and it also proves the descriptor is genuinely inherited across ``execve``.
    """
    script = write_child_script(tmp_path / "fake_machine3.py", status="acquired", chatter=chatter)
    monkeypatch.setenv("LM3_MACHINE3_BIN", str(script))

    resp = start_a_run(client, sandbox)

    assert resp.status_code == 200, resp.text
    assert resp.json()["run_id"] == "22222222-2222-4222-8222-222222222222"
    console = Path(resp.json()["console_log"])
    deadline = time.time() + 10.0
    while time.time() < deadline and console.stat().st_size < 16:
        time.sleep(0.05)
    assert console.stat().st_size >= 16, "the chatter must have landed in the request log"


def test_a_real_child_that_exits_75_is_a_409(
    client: TestClient, sandbox: Sandbox, tmp_path: Path, v2: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole loop, for real: the child writes ``busy``, exits 75, and the API answers 409.

    Section 2.3/3.3 fix the exit code at 75 so the shell wrapper can tell "another root holds the
    deployment" from a genuine failure; the server's equivalent is this status code.
    """
    script = write_child_script(tmp_path / "busy_machine3.py", status="busy", exit_code=75)
    monkeypatch.setenv("LM3_MACHINE3_BIN", str(script))

    resp = start_a_run(client, sandbox)

    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["reason"] == "busy"
    assert not (sandbox.run_dir / "logs").exists(), "invariant 14: the loser creates no run dirs"


# --------------------------------------------------------------------------------------------- #
# 7. Section 2.13 -- hardware setup leaves the server process
# --------------------------------------------------------------------------------------------- #
# Three DISTINCT lines on purpose. With one line, a drain() that emits the first line and then
# aborts (and re-emits it every poll) is indistinguishable from a correct one.
_SETUP_CONSOLE_LINES = ["sweep: gpu 0", "sweep: gpu 1", "profile written"]

_SETUP_SCRIPT = '''#!{python}
import sys, time
for line in {lines!r}:
    sys.stdout.write(line + "\\n")
    sys.stdout.flush()
{grandchild}
time.sleep({sleep})
sys.exit({exit_code})
'''

#: Opt-in body that gives the fake setup root a real DESCENDANT -- the tree shape gate 6 is about.
#: ``Popen`` WITHOUT ``start_new_session`` leaves the grandchild in the root's process group, so only
#: a ``killpg`` reaches it; ``terminate()`` aimed at the retained handle alone would orphan it. The
#: pid is published by rename, so a reader can never see a half-written file.
_GRANDCHILD_BODY = '''
import os, subprocess
_gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
_tmp = {pidfile!r} + ".tmp"
with open(_tmp, "w") as _fh:
    _fh.write(str(_gc.pid))
    _fh.flush()
    os.fsync(_fh.fileno())
os.replace(_tmp, {pidfile!r})
'''


def write_setup_script(path: Path, *, sleep: float = 0.0, exit_code: int = 0,
                       grandchild_pidfile: Path | None = None) -> Path:
    '''Write the fake ``lm3-setup``. With ``grandchild_pidfile`` it also forks a long sleeper.'''
    grandchild = ("" if grandchild_pidfile is None
                  else _GRANDCHILD_BODY.format(pidfile=str(grandchild_pidfile)))
    path.write_text(_SETUP_SCRIPT.format(python=sys.executable, sleep=sleep,
                                         exit_code=exit_code, lines=_SETUP_CONSOLE_LINES,
                                         grandchild=grandchild),
                    encoding="utf-8")
    path.chmod(0o755)
    return path


def wait_for_pid_file(pidfile: Path, timeout: float = 20.0) -> int:
    '''Block until the fake setup root has published its grandchild's pid, and return it.'''
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            return int(pidfile.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            time.sleep(0.05)
    raise AssertionError(f"the fake setup root never published a grandchild pid at {pidfile}")


def wait_until_gone(pid: int, timeout: float = 20.0) -> bool:
    '''True once ``pid`` is gone. Signal 0 only PROBES -- it delivers nothing to a live process.'''
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:                        # the pid belongs to someone else now
            return True
        time.sleep(0.05)
    return False


def reap(pid: int) -> None:
    '''Best-effort cleanup, so a failed assertion cannot leak a 600-second sleeper onto the box.'''
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:                                    # already gone, or never ours
        pass


def wait_for_setup_state(client: TestClient, jid: str, states: set[str],
                         timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    body: dict = {}
    while time.time() < deadline:
        body = client.get("/v1/setup/events", params={"jid": jid}, headers=bearer()).json()
        if body.get("state") in states:
            return body
        time.sleep(0.1)
    return body


def test_gui_hardware_setup_runs_out_of_process_and_reports_through_a_jsonl_log(
    client: TestClient, sandbox: Sandbox, tmp_path: Path, v2: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Section 2.13: "GUI hardware setup runs as a dedicated ``lm3-setup`` subprocess".

    Today it is ``run_in_threadpool(run_setup, ...)`` INSIDE the server -- no Popen, no process
    group -- which makes "terminate the setup root's process group" mean "kill the server", and
    which is the single exception invariant 13 had to carry. Progress arrives through a
    server-private append-only JSONL log, replayable after a disconnect.
    """
    monkeypatch.setenv("LM3_SETUP_BIN", str(write_setup_script(tmp_path / "lm3-setup")))

    started = client.post("/v1/setup", headers=bearer())

    assert started.status_code == 200, started.text
    jid = started.json()["job_id"]
    body = wait_for_setup_state(client, jid, {"done", "error"})
    assert body["state"] == "done", body
    kinds = [e.get("type") for e in body["events"]]
    assert kinds[0] == "started"
    assert "state" in kinds
    # EXACTLY once each, in order. ``any(...)`` cannot see the bug this pins: drain() used to mix
    # the file-iterator protocol with tell(), which raises OSError("telling position disabled by
    # next() call"), so the offset never advanced and the FIRST console line was re-appended on
    # every 0.5 s poll while every later line was lost.
    lines = [e["line"] for e in body["events"] if e.get("type") == "log"]
    assert lines == _SETUP_CONSOLE_LINES, body["events"]
    # Replayable because it is a FILE, not a buffer: a second reader sees the same history.
    again = client.get("/v1/setup/events", params={"jid": jid}, headers=bearer()).json()
    assert again["events"] == body["events"]


def test_setup_stop_kills_the_setup_tree_and_leaves_the_server_serving(
    client: TestClient, sandbox: Sandbox, tmp_path: Path, v2: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Section 2.13: "Stop terminates the setup/calibration tree and leaves the server running"."""
    monkeypatch.setenv("LM3_SETUP_BIN", str(write_setup_script(tmp_path / "lm3-setup", sleep=120)))
    jid = client.post("/v1/setup", headers=bearer()).json()["job_id"]

    stopped = client.post("/v1/setup/stop", headers=bearer())

    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["stopped"] == [jid]
    body = wait_for_setup_state(client, jid, {"error", "done"})
    assert body["state"] == "error"
    assert any(e.get("state") == "stopped" for e in body["events"]), body["events"]
    assert client.get("/healthz").status_code == 200, "the server keeps serving"


@pytest.mark.skipif(os.name == "nt",
                    reason="POSIX process groups; the Windows job object is gate 17 / Step 5b")
def test_setup_stop_kills_a_grandchild_that_inherited_the_setup_process_group(
    client: TestClient, sandbox: Sandbox, tmp_path: Path, v2: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate 6: "Server Stop of a calibration tree stops the child, not just the root".

    Section 3.3: "Server Stop targets the retained root process group (POSIX) or job object
    (Windows), so a calibration tree stops as one unit rather than orphaning its child." The
    one-level Stop test above cannot tell that apart from ``proc.terminate()`` on the handle --
    both kill a lone sleeper. Only a DESCENDANT that inherited the root's process group can, so the
    fake setup root forks one and this test observes it die.

    The grandchild stands in for the calibration child a real ``lm3-setup`` spawns; it deliberately
    holds no lease, because what gate 6 pins is the REACH of the signal. The lease half of the tree
    is gates 4 and 5, tested with real lease-holding subprocesses elsewhere.
    """
    pidfile = tmp_path / "grandchild.pid"
    monkeypatch.setenv("LM3_SETUP_BIN", str(write_setup_script(
        tmp_path / "lm3-setup", sleep=120, grandchild_pidfile=pidfile)))
    jid = client.post("/v1/setup", headers=bearer()).json()["job_id"]
    grandchild_pid = wait_for_pid_file(pidfile)

    try:
        stopped = client.post("/v1/setup/stop", headers=bearer())

        assert stopped.status_code == 200, stopped.text
        assert stopped.json()["stopped"] == [jid]
        body = wait_for_setup_state(client, jid, {"error", "done"})
        assert body["state"] == "error", body
        assert wait_until_gone(grandchild_pid), (
            f"grandchild {grandchild_pid} outlived Stop: the tree was not signaled as a group")
        assert client.get("/healthz").status_code == 200, "the server keeps serving"
    finally:
        reap(grandchild_pid)


def test_setup_without_a_canonical_config_names_the_missing_file(
    client: TestClient, sandbox: Sandbox, v2: None,
) -> None:
    """Section 2.13: "the setup job fails with a precise message naming the missing path".

    ``run_setup`` dereferences ``cfg`` at ``_fingerprint(cfg)`` before any guard, so passing the
    ``None`` the server passes today raises an ``AttributeError`` the GUI shows as an opaque job
    error. The config is required, and its absence is a 400 that says which file.
    """
    sandbox.settings_path.unlink()

    resp = client.post("/v1/setup", headers=bearer())

    assert resp.status_code == 400, resp.text
    assert str(sandbox.settings_path) in resp.json()["detail"]


def test_setup_stop_refuses_when_there_is_no_managed_child(
    client: TestClient, sandbox: Sandbox, v2: None,
) -> None:
    """Section 2.5: control needs a retained handle. No handle, no stop -- and no synthesized one."""
    resp = client.post("/v1/setup/stop", headers=bearer())

    assert resp.status_code == 409
    assert "no managed hardware setup" in resp.json()["detail"]


# --------------------------------------------------------------------------------------------- #
# 8. Legacy /v1/jobs must not make the SERVER a lease holder (invariant 13)
# --------------------------------------------------------------------------------------------- #
def test_a_legacy_job_goes_out_of_process_under_the_flag(
    sandbox: Sandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant 13: "the server process is NEVER a lease holder", without exception.

    ``app._worker`` calls ``machine3()`` in-process. Once Step 3 wraps ``machine3()`` in a lease,
    that line makes the server one -- so under the flag a legacy job takes the same managed launch
    the Run button takes, and inherits its 409 for free (Step 3's exit gate names "legacy jobs" as
    one of the four entry points that must refuse a second root activity).
    """
    from leafmachine3.server import app as app_mod
    from leafmachine3.server import metrics_api

    job = app_mod.Job(id="j1", root=tmp_path / "job", cfg_path=sandbox.settings_path)
    seen: list[str] = []
    states = iter([True, True, False])

    monkeypatch.setattr(metrics_api, "start_run",
                        lambda **kw: seen.append(kw["config_path"]) or {}, raising=True)
    monkeypatch.setattr(metrics_api, "is_active", lambda: next(states, False), raising=True)
    monkeypatch.setattr(metrics_api, "active", lambda: {"state": "done"}, raising=True)

    app_mod._run_job_as_subprocess(job)

    assert seen == [str(sandbox.settings_path)], "the job must be launched, not called in-process"


def test_a_legacy_job_surfaces_a_failed_run_as_an_error(
    sandbox: Sandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The out-of-process job still fails the JOB when the run fails -- exit code, not exception."""
    from leafmachine3.server import app as app_mod
    from leafmachine3.server import metrics_api

    job = app_mod.Job(id="j2", root=tmp_path / "job", cfg_path=sandbox.settings_path)
    monkeypatch.setattr(metrics_api, "start_run", lambda **kw: {}, raising=True)
    monkeypatch.setattr(metrics_api, "is_active", lambda: False, raising=True)
    monkeypatch.setattr(metrics_api, "active",
                        lambda: {"state": "error", "error": "machine3 exited with code 3"},
                        raising=True)

    with pytest.raises(RuntimeError, match="exited with code 3"):
        app_mod._run_job_as_subprocess(job)


# --------------------------------------------------------------------------------------------- #
# 9. Unit-level: the bounded reader, and the timeout knob
# --------------------------------------------------------------------------------------------- #
def test_the_reader_is_bounded_in_bytes_as_well_as_time() -> None:
    """Section 2.4 names "oversized status" as its own failure branch, so it needs its own cap."""
    from leafmachine3.core.runtime._types import MAX_HANDSHAKE_BYTES
    from leafmachine3.server.metrics_api import _LaunchStatusPipe

    pipe = _LaunchStatusPipe()
    writer = os.dup(pipe.child_value)
    try:
        thread = threading.Thread(
            target=lambda: os.write(writer, b"{" + b"x" * (MAX_HANDSHAKE_BYTES + 1024)),
            daemon=True)
        thread.start()
        outcome = pipe.read(5.0)
        thread.join(5.0)
        assert outcome.outcome == "oversized"
        assert not outcome.ok
    finally:
        try:
            os.close(writer)
        except OSError:
            pass
        pipe.close()


def test_the_handshake_timeout_is_bounded_and_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Section 2.4: "Size it accordingly (order of seconds ...) and make it configurable"."""
    from leafmachine3.core.runtime._types import DEFAULT_HANDSHAKE_TIMEOUT_S
    from leafmachine3.server.metrics_api import _handshake_timeout_s

    monkeypatch.delenv("LM3_HANDSHAKE_TIMEOUT_S", raising=False)
    assert _handshake_timeout_s() == float(DEFAULT_HANDSHAKE_TIMEOUT_S)
    assert DEFAULT_HANDSHAKE_TIMEOUT_S <= 60.0, "bounded by config load, not by a cold torch import"

    monkeypatch.setenv("LM3_HANDSHAKE_TIMEOUT_S", "2.5")
    assert _handshake_timeout_s() == 2.5
    for bad in ("nonsense", "-1", "0"):
        monkeypatch.setenv("LM3_HANDSHAKE_TIMEOUT_S", bad)
        assert _handshake_timeout_s() == float(DEFAULT_HANDSHAKE_TIMEOUT_S)


def test_execution_owned_paths_is_the_prohibited_list_section_2_4_names(tmp_path: Path) -> None:
    """Invariant 14 has to be testable, which means the prohibited set has ONE definition."""
    from leafmachine3.server.metrics_api import execution_owned_paths

    run_dir = tmp_path / "out" / "acer"
    names = {p.name for p in execution_owned_paths(run_dir, "acer")}

    assert {"acer", "logs", "lm3.log", "console.log", "acer.sqlite", "_tmp_original"} == names


# --------------------------------------------------------------------------------------------- #
# 10. Process isolation is ONE platform-aware source, and the Windows tree is a job object
# --------------------------------------------------------------------------------------------- #
class _FakeJobSurface:
    """The five Win32 calls ``_JobObject`` makes, recorded instead of made.

    Section 1.1 rule 2: a Windows branch proven only against a fake is a test of our MODEL of the
    object manager, not of the object manager -- so everything asserted here is "implemented but not
    yet natively validated". It is still worth far more than an unexecuted branch, which is what the
    job object was before: ``start_new_session`` is discarded by CPython on Windows, so nothing gave
    the child a tree of its own there at all.
    """

    def __init__(self, *, fail_on: str = "") -> None:
        self.fail_on = fail_on
        self.calls: list[tuple] = []
        self.job_handle = 0x00C0FFEE
        self.open_handles: set[int] = set()

    def _check(self, name: str) -> None:
        if self.fail_on == name:
            raise OSError(5, f"{name} refused")

    def create_job(self) -> int:
        self._check("create_job")
        self.calls.append(("create_job",))
        self.open_handles.add(self.job_handle)
        return self.job_handle

    def set_kill_on_close(self, job: int) -> None:
        self._check("set_kill_on_close")
        self.calls.append(("set_kill_on_close", job))

    def assign_process(self, job: int, process_handle: int) -> None:
        self._check("assign_process")
        self.calls.append(("assign_process", job, process_handle))

    def terminate_job(self, job: int, exit_code: int) -> None:
        self._check("terminate_job")
        self.calls.append(("terminate_job", job, exit_code))

    def open_process(self, pid: int) -> int:
        self._check("open_process")
        self.calls.append(("open_process", pid))
        return 0x00BEEF01

    def close_handle(self, handle: int) -> None:
        self.calls.append(("close_handle", handle))
        self.open_handles.discard(handle)


class _WindowsProc:
    """``Popen``'s Windows shape: a process handle, and no session anybody can killpg."""

    def __init__(self) -> None:
        self.pid = 4242
        self._handle = 0x00BEEF01
        self.returncode: int | None = None
        self.terminated = 0
        self.killed = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1

    def kill(self) -> None:
        self.killed += 1


def _windows_child(surface: _FakeJobSurface) -> Any:
    """A ``_ManagedChild`` shaped exactly as one is on Windows: a job object and no pgid."""
    from leafmachine3.server import metrics_api

    child = metrics_api._ManagedChild(_WindowsProc(), kind="pipeline", platform_name="nt",
                                      job_surface=surface)
    # What ``__init__`` leaves there over here: ``os.getpgid`` does not exist on Windows, so the
    # POSIX branch of ``signal_group`` is unreachable. On Linux the same call succeeds against this
    # test's fake pid, so it is set back to the Windows value explicitly.
    child.pgid = None
    return child


def test_both_managed_launches_take_their_isolation_keywords_from_one_helper() -> None:
    """Finding: ``start_new_session`` is a SILENT no-op on Windows, and it was hardcoded twice.

    CPython's Windows ``_execute_child`` receives it as ``unused_start_new_session``, so the child
    stayed in the server's own process group with no ``CREATE_NEW_PROCESS_GROUP`` and no job object
    -- and section 2.4's "child termination ... identical on both platforms" was false. One helper,
    delegating to the ``setup_popen_kwargs`` that already had the ``nt`` branch.
    """
    from leafmachine3.server.metrics_api import process_isolation_kwargs
    from leafmachine3.setup.hardware_setup import setup_popen_kwargs

    assert process_isolation_kwargs() == {"start_new_session": True}
    assert process_isolation_kwargs() == setup_popen_kwargs(), "one source, not two spellings"


def test_on_windows_the_isolation_keywords_are_a_new_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``nt`` branch, exercised on Linux: a group, and no keyword the platform discards.

    ``os.name`` is patched INSIDE ``hardware_setup`` rather than on the ``os`` module itself:
    ``pathlib`` reads ``os.name`` too, and flipping it process-wide makes every ``Path(...)`` in the
    interpreter try to build a ``WindowsPath``.
    """
    from types import SimpleNamespace

    from leafmachine3.server.metrics_api import process_isolation_kwargs
    from leafmachine3.setup import hardware_setup

    monkeypatch.setattr(hardware_setup, "os", SimpleNamespace(name="nt"), raising=True)
    kwargs = process_isolation_kwargs()

    # 0x00000200 is CREATE_NEW_PROCESS_GROUP; ``subprocess`` does not define the name off Windows.
    assert kwargs == {"creationflags": 0x00000200}
    assert "start_new_session" not in kwargs, "CPython discards it on Windows -- it must not be sent"


def test_the_managed_launch_passes_the_helpers_keywords_through_to_popen(
    client: TestClient, sandbox: Sandbox, fake_spawn: Any, v2: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not just "the POSIX keyword is there": the keywords come FROM the shared helper."""
    from leafmachine3.server import metrics_api

    monkeypatch.setattr(metrics_api, "process_isolation_kwargs",
                        lambda: {"start_new_session": True, "creationflags": 0x200}, raising=True)
    fake = fake_spawn(reply=acquired_line())

    assert start_a_run(client, sandbox).status_code == 200

    kwargs = fake.calls[0].kwargs
    assert kwargs["start_new_session"] is True
    assert kwargs["creationflags"] == 0x200, "composed in, not dropped by the section 2.4 allowlist"


def test_on_posix_there_is_no_job_object_because_the_session_is_the_tree() -> None:
    """Section 3.3's two spellings of one idea. POSIX already has the tree; Windows must build it."""
    from leafmachine3.server.metrics_api import _attach_job_object

    assert _attach_job_object(_WindowsProc(), platform_name="posix") is None


def test_a_windows_child_is_assigned_to_a_kill_on_close_job_object() -> None:
    """Section 3.3: "Stop targets the retained root process group (POSIX) or job object (Windows)".

    ``CREATE_NEW_PROCESS_GROUP`` alone is not that: it only enables ``GenerateConsoleCtrlEvent``,
    while ``TerminateProcess`` still reaches one process -- so ``calibrate._run_pipeline``'s
    grandchild and every executor spawn worker would survive a Stop.
    """
    surface = _FakeJobSurface()

    child = _windows_child(surface)

    assert child.job is not None, "a Windows managed child must have a tree of its own"
    assert ("create_job",) in surface.calls
    assert ("set_kill_on_close", surface.job_handle) in surface.calls
    assert ("assign_process", surface.job_handle, 0x00BEEF01) in surface.calls


def test_stopping_a_windows_child_terminates_the_JOB_not_just_the_process() -> None:
    """The whole point: one call that reaches the grandchildren, and the ladder keeps its shape."""
    surface = _FakeJobSurface()
    child = _windows_child(surface)

    assert child.signal_group(15) is True

    assert [c for c in surface.calls if c[0] == "terminate_job"] == [
        ("terminate_job", surface.job_handle, 15)]
    assert child.proc.terminated == 0, "TerminateProcess would have reached only the direct child"


def test_a_windows_child_whose_job_call_fails_falls_back_to_the_handle() -> None:
    """A degraded Stop still beats no Stop -- and section 2.5's authority is still the handle."""
    surface = _FakeJobSurface()
    child = _windows_child(surface)
    surface.fail_on = "terminate_job"

    assert child.signal_group(15) is True

    assert child.proc.terminated == 1


def test_a_launch_whose_job_object_cannot_be_created_still_starts() -> None:
    """Loud, not fatal: the log line is what tells an operator which kind of Stop they have."""
    from leafmachine3.server import metrics_api

    surface = _FakeJobSurface(fail_on="create_job")

    child = metrics_api._ManagedChild(_WindowsProc(), kind="pipeline", platform_name="nt",
                                      job_surface=surface)

    assert child.job is None
    assert child.pid == 4242


def test_a_job_object_that_cannot_be_configured_does_not_leak_its_handle() -> None:
    """A half-built job is still a kernel handle, and the caller never sees the object to close it."""
    from leafmachine3.server import metrics_api

    surface = _FakeJobSurface(fail_on="set_kill_on_close")

    child = metrics_api._ManagedChild(_WindowsProc(), kind="pipeline", platform_name="nt",
                                      job_surface=surface)

    assert child.job is None
    assert surface.open_handles == set(), "CreateJobObject succeeded, so its handle must be closed"


def test_the_job_handle_is_released_once_the_tree_is_gone() -> None:
    """KILL_ON_JOB_CLOSE makes releasing the handle the tree's last rites, so it happens ONCE."""
    surface = _FakeJobSurface()
    child = _windows_child(surface)

    child.release_job()
    child.release_job()

    assert child.job is None
    assert [c for c in surface.calls if c[0] == "close_handle"] == [
        ("close_handle", surface.job_handle)]
    assert surface.open_handles == set(), "the job handle must not leak per launch"


@pytest.mark.skipif(sys.platform != "win32", reason="requires the real Windows job-object kernel")
def test_the_real_windows_job_object_stops_root_and_grandchild(tmp_path: Path) -> None:
    """Gate 17's native half: TerminateJobObject reaches a descendant on the real object manager.

    The root waits until after :class:`_ManagedChild` assigned it to the job before it creates the
    grandchild, removing the otherwise-real race where a very fast fixture could fork before job
    assignment. Handles are opened before Stop and waited directly, so PID reuse cannot turn this
    into a false pass.
    """
    import ctypes
    from ctypes import wintypes

    from leafmachine3.server.metrics_api import _ManagedChild, process_isolation_kwargs

    ready = tmp_path / "root.ready"
    go = tmp_path / "spawn-child"
    grandchild_pidfile = tmp_path / "grandchild.pid"
    script = tmp_path / "windows_job_tree.py"
    script.write_text(
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "ready, go, pidfile = map(Path, sys.argv[1:4])\n"
        "ready.write_text(str(os.getpid()), encoding='utf-8')\n"
        "deadline = time.monotonic() + 30\n"
        "while not go.exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
        "if not go.exists(): raise SystemExit(2)\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "tmp = pidfile.with_suffix('.tmp')\n"
        "tmp.write_text(str(child.pid), encoding='utf-8')\n"
        "os.replace(tmp, pidfile)\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )

    proc: subprocess.Popen[bytes] | None = None
    managed = None
    grandchild_handle = None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    try:
        proc = subprocess.Popen(                                      # noqa: S603 - fixed interpreter/script
            [sys.executable, str(script), str(ready), str(go), str(grandchild_pidfile)],
            **process_isolation_kwargs(),
        )
        wait_for_pid_file(ready, timeout=20.0)  # numeric content makes publication atomic enough here
        managed = _ManagedChild(proc, kind="pipeline", platform_name="nt")
        assert managed.job is not None, "the real process was not assigned to a Windows job object"

        go.write_text("spawn", encoding="utf-8")
        grandchild_pid = wait_for_pid_file(grandchild_pidfile, timeout=20.0)
        # SYNCHRONIZE is enough to wait for termination; no signaling authority is reconstructed
        # from this PID. The managed job handle remains the only control authority.
        grandchild_handle = kernel32.OpenProcess(0x00100000, False, grandchild_pid)
        assert grandchild_handle, (
            f"could not open grandchild {grandchild_pid} for a wait: {ctypes.get_last_error()}")

        assert managed.signal_group(signal.SIGTERM) is True
        proc.wait(timeout=20.0)
        assert kernel32.WaitForSingleObject(grandchild_handle, 20_000) == 0, (
            "the root exited but its grandchild survived TerminateJobObject")
    finally:
        if grandchild_handle:
            kernel32.CloseHandle(grandchild_handle)
        if managed is not None:
            managed.release_job()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10.0)


# --------------------------------------------------------------------------------------------- #
# 11. The handshake must not freeze the server while it waits
# --------------------------------------------------------------------------------------------- #
def test_a_launch_in_flight_leaves_the_status_lock_free(
    client: TestClient, sandbox: Sandbox, fake_spawn: Any, v2: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The section 2.4 read is bounded at 30 s by default; ``_RUN_LOCK`` must not be held across it.

    The plan's named regression case -- "a child that DOES acquire the lease but deliberately delays
    its status line past the timeout" -- is exactly the slow path, and it is followed by a SIGTERM
    grace and a SIGKILL wait. With the lock held across all of that, ``_current()`` (so
    ``GET /v1/run/active``, ``GET /v1/run/console``, ``POST /v1/run/stop`` and the SSE stream)
    blocked for the duration.
    """
    from leafmachine3.server import metrics_api

    monkeypatch.setenv("LM3_HANDSHAKE_TIMEOUT_S", "5")
    fake_spawn(reply=None, hold=True)
    outcome: dict[str, Any] = {}

    def launch() -> None:
        try:
            outcome["record"] = metrics_api.start_run(config_path=str(sandbox.settings_path))
        except BaseException as exc:                      # noqa: BLE001 - reported below
            outcome["error"] = exc

    worker = threading.Thread(target=launch, name="lm3-test-launch", daemon=True)
    worker.start()
    deadline = time.time() + 5.0
    while time.time() < deadline and not fake_spawn.box.get("fake", None).calls:
        time.sleep(0.01)                                  # the child is launched: the read is on

    started = time.time()
    record = metrics_api.active()
    elapsed = time.time() - started

    assert elapsed < 0.5, f"GET /v1/run/active waited {elapsed:.1f}s on the handshake"
    assert record["active"] is True and record["state"] == "starting", record
    assert record["run_name"] == sandbox.run_name
    assert set(record) == set(metrics_api._IDLE_RECORD), "the published key set never changes"
    assert metrics_api.is_active() is True
    # A second start is still refused, which is the reason the lock was held in the first place.
    with pytest.raises(metrics_api.RunError) as busy:
        metrics_api.start_run(config_path=str(sandbox.settings_path))
    assert busy.value.status == 409
    # And so is a Stop, which has no managed child to signal yet (section 2.5).
    with pytest.raises(metrics_api.RunError) as stop:
        metrics_api.stop_run()
    assert stop.value.status == 409

    worker.join(30.0)
    assert not worker.is_alive()
    assert isinstance(outcome.get("error"), metrics_api.RunError), outcome
    assert metrics_api.active()["state"] != "starting", "the sentinel must be cleared on failure"
    assert metrics_api.active()["active"] is False


def test_the_status_route_answers_off_the_event_loop(
    sandbox: Sandbox, quiet_metrics: None,
) -> None:
    """``GET /v1/run/active`` was the one run route calling a lock-taking function on the loop.

    A ``threading`` lock awaited from a coroutine blocks the LOOP THREAD, not just its own request,
    so one held lock stalls every route, every SSE stream and ``/healthz`` with it. Measured here
    the way it was measured on the real router: hold the lock, and count heartbeats.
    """
    import asyncio

    from leafmachine3.server import metrics_api

    route = next(r for r in metrics_api.router().routes
                 if getattr(r, "path", "") == "/v1/run/active")
    ticks: list[float] = []

    async def drive() -> dict:
        async def heartbeat() -> None:
            while True:
                ticks.append(time.time())
                await asyncio.sleep(0.02)

        beat = asyncio.ensure_future(heartbeat())
        await asyncio.sleep(0.05)
        holder = threading.Thread(target=_hold_run_lock, args=(1.0,), daemon=True)
        holder.start()
        await asyncio.sleep(0.05)                          # let the holder actually take it
        try:
            return await route.endpoint()
        finally:
            beat.cancel()
            holder.join(5.0)

    record = asyncio.run(drive())

    assert record["state"] == "idle"
    assert len(ticks) > 10, (
        f"the event loop got {len(ticks)} ticks while _RUN_LOCK was held for 1 s -- the route is "
        "calling active() on the loop thread instead of run_in_threadpool")


def _hold_run_lock(seconds: float) -> None:
    from leafmachine3.server import metrics_api

    with metrics_api._RUN_LOCK:
        time.sleep(seconds)


# --------------------------------------------------------------------------------------------- #
# 12. The status pipe never frees a descriptor a thread is still reading
# --------------------------------------------------------------------------------------------- #
def test_close_does_not_free_a_descriptor_the_reader_is_blocked_on() -> None:
    """``close()``'s own docstring: "never out from under a blocked reader thread".

    It joined for at most 5 s and then closed ``read_fd`` anyway. On Linux ``close()`` does not wake
    a blocked ``read()``, so the pump stayed parked on a NUMBER that the next ``open`` in the server
    would be handed -- and it would then eat that file's bytes. The survivor is not hypothetical:
    the module's own 500 branch says "the child ... survived SIGKILL".
    """
    from leafmachine3.server.metrics_api import _LaunchStatusPipe

    pipe = _LaunchStatusPipe()
    survivor = os.dup(pipe.child_value)                    # a grandchild that inherited the write end
    try:
        assert pipe.read(0.2).outcome == "timeout"
        pipe.close(join_s=0.1)

        assert pipe._thread is not None and pipe._thread.is_alive(), "the reader is still blocked"
        os.fstat(pipe.read_fd)                             # raises EBADF if close() freed it

        os.close(survivor)                                 # the survivor finally dies -> EOF
        survivor = -1
        pipe._thread.join(5.0)
        assert not pipe._thread.is_alive()
        with pytest.raises(OSError):
            os.fstat(pipe.read_fd)                         # the pump closed its own descriptor
    finally:
        if survivor >= 0:
            with contextlib.suppress(OSError):
                os.close(survivor)
        pipe.close(join_s=1.0)


def test_a_completed_read_still_closes_the_descriptor_exactly_once() -> None:
    """Handing the fd to the pump must not leak it on the ordinary path, or double-close it."""
    from leafmachine3.server.metrics_api import _LaunchStatusPipe

    pipe = _LaunchStatusPipe()
    writer = os.dup(pipe.child_value)
    os.write(writer, acquired_line())
    os.close(writer)

    assert pipe.read(5.0).outcome == "acquired"
    pipe.close()
    pipe.close()                                           # idempotent: no second os.close(fd)

    with pytest.raises(OSError):
        os.fstat(pipe.read_fd)
