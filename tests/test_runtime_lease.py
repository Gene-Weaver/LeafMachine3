"""The deployment activity lease: the POSIX adapter, ``RuntimeLease``, and the real-process gates.

Plan section 2.2, with section 3.3 for the lifecycle and section 8 for the gates. The Windows
adapter has its own file (``test_runtime_lease_windows.py``) because gate 27 requires the Windows
behavior to be "verified against the Windows adapter itself, not inferred from the POSIX result" --
the two implementations share no mechanism, so one file proving both would be proving neither.

Which gates land here:

* gate 4  -- killing the root while an inherited child runs keeps a second root blocked
* gate 7  -- an executor worker never holds the lease reference, proved rather than assumed
* gate 9  -- a bogus inherited descriptor is rejected, including one on the RIGHT inode
* gate 22 -- a child that fails validation renames nothing (here: it never reaches the grant)
* gate 39 -- default deployments contend, resolved from the environment
* gate 51 -- a killed lone owner releases the lock and is left to be classified abandoned
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

import pytest

from leafmachine3.core import paths as P
from leafmachine3.core.runtime import lease as lease_mod
from leafmachine3.core.runtime._types import (
    ACTIVE_RECORD_FILENAME,
    ACTIVITY_LOCK_FILENAME,
    ENV_LEASE_CAPABILITY,
    ENV_LEASE_EVENT_HANDLE,
    ENV_LEASE_FD,
    EXIT_CODE_BUSY,
    Activity,
    LeaseError,
    LeaseInheritanceError,
    LeaseNotHeldError,
    RuntimeBusyError,
)
from leafmachine3.core.runtime.lease import (
    PosixLeaseAdapter,
    RuntimeLease,
    WindowsLeaseAdapter,
    acquire_root_lease,
    cleanup_lease,
    clear_lease_env,
    inherit_lease,
    lease_adapter,
    probe_deployment_occupied,
    process_creation_lock,
    real_win32_surface,
    sid_hash,
    windows_lease_event_name,
)

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="the POSIX adapter is flock-based")

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPER = REPO_ROOT / "tests" / "helpers" / "lease_subprocess.py"
KEY = "pytest-lease"
#: Generous enough for a cold interpreter start on a loaded box, short enough that a real hang fails
#: the test instead of the session.
SPAWN_TIMEOUT_S = 60.0


# --------------------------------------------------------------------------------------------- #
# Fixtures and small utilities
# --------------------------------------------------------------------------------------------- #

@pytest.fixture()
def deployment(tmp_path: Path) -> Path:
    """A private deployment runtime directory. Never the developer's -- see tests/conftest.py."""
    return tmp_path / "runtime" / KEY


def adapter_for(deployment_dir: Path) -> PosixLeaseAdapter:
    return PosixLeaseAdapter(deployment_dir, deployment_key=KEY)


def helper_env(deployment_dir: Path, **extra: object) -> dict[str, str]:
    env = dict(os.environ)
    env["LM3_TEST_DIR"] = str(deployment_dir)
    env["LM3_TEST_KEY"] = KEY
    env["PYTHONPATH"] = str(REPO_ROOT)
    for name, value in extra.items():
        env[name] = str(value)
    return env


def default_helper_env(base: Path, **extra: object) -> dict[str, str]:
    """A child environment that states the runtime BASE and nothing at all about the identity.

    ``tests/conftest.py`` exports ``LM3_DEPLOYMENT_ID`` at import time -- unconditionally, so the
    suite can never take the developer's lease -- which means popping it here is what actually makes
    the child a DEFAULT deployment, gate 39's subject. ``LM3_RUNTIME_DIR`` still points inside
    ``tmp_path``, so the ``<base>/<key>`` the child resolves for itself lands in the sandbox rather
    than in the real default runtime directory (gate 46).
    """
    env = dict(os.environ)
    env["LM3_RUNTIME_DIR"] = str(base)
    env["PYTHONPATH"] = str(REPO_ROOT)
    for name in ("LM3_DEPLOYMENT_ID", "LM3_TEST_DIR", "LM3_TEST_KEY"):
        env.pop(name, None)
    for name, value in extra.items():
        env[name] = str(value)
    return env


#: The gate 39 child, written into ``tmp_path`` by the test rather than living in
#: ``tests/helpers/lease_subprocess.py``, because every mode of that helper is handed both its
#: deployment directory and its deployment key -- which is exactly the resolution step this gate has
#: to exercise. This child is told only ``LM3_RUNTIME_DIR``; section 2.1's "unset ``LM3_DEPLOYMENT_ID``
#: means the raw literal ``default``. Nothing else" and section 3.1's ``<base>/<deployment_id>`` are
#: what have to put two independent processes on one lock.
DEFAULT_DEPLOYMENT_CHILD = """\
import json
import os
import sys
import time
from pathlib import Path

from leafmachine3.core.runtime._types import EXIT_CODE_BUSY, RuntimeBusyError
from leafmachine3.core.runtime.lease import RuntimeLease

mode = sys.argv[1]
report = Path(os.environ["LM3_TEST_REPORT"])
lease = RuntimeLease(env=os.environ)   # no deployment_dir, no deployment_key: resolve both
identity = {"deployment_key": lease.deployment_key,
            "deployment_dir": str(lease.deployment_dir),
            "cwd": os.getcwd()}
try:
    lease.acquire()
except RuntimeBusyError as busy:
    report.write_text(json.dumps({"result": "busy", "busy_key": busy.deployment_key, **identity}),
                      encoding="utf-8")
    sys.exit(EXIT_CODE_BUSY)
report.write_text(json.dumps({"result": "acquired", "pid": os.getpid(), **identity}),
                  encoding="utf-8")
if mode == "root-hold":
    Path(os.environ["LM3_TEST_READY"]).write_text("ready", encoding="utf-8")
    stop = Path(os.environ["LM3_TEST_STOP"])
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline and not stop.exists():
        time.sleep(0.01)
lease.release()
"""


def spawn(mode: str, env: dict[str, str], *, cwd: str | None = None,
          script: Path | None = None) -> subprocess.Popen[str]:
    """Run one helper mode. ``cwd`` is explicit because section 3.1 requires path resolution to be
    "independent of the checkout and CWD", and a default is a value no test ever varies."""
    return subprocess.Popen(                                          # noqa: S603 - fixed argv
        [sys.executable, str(script or HELPER), mode], env=env, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def wait_for(path: Path, timeout: float = SPAWN_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"{path} never appeared within {timeout}s")


def wait_until(predicate, timeout: float = SPAWN_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"condition never became true within {timeout}s")


def report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------- #
# Naming and environment plumbing
# --------------------------------------------------------------------------------------------- #

def test_sid_hash_is_the_bounded_sha256_of_the_canonical_sid() -> None:
    """Plan section 2.2: sha256 of the ConvertSidToStringSidW form, UTF-8, 16 hex chars (gate 30)."""
    from hashlib import sha256

    sid = "S-1-5-21-1004336348-1177238915-682003330-512"
    assert sid_hash(sid) == sha256(sid.encode("utf-8")).hexdigest()[:16]
    assert len(sid_hash(sid)) == 16
    assert sid_hash(sid) != sid_hash("S-1-5-21-1004336348-1177238915-682003330-513")


def test_lease_event_name_is_global_and_names_both_discriminators() -> None:
    """``Global\\``, never ``Local\\`` -- a Local event is one logon session wide (gates 28, 29)."""
    name = windows_lease_event_name("my-deployment", sid_hash_value=sid_hash("S-1-5-21-7"))
    assert name.startswith("Global\\lm3-lease-")
    assert "Local\\" not in name
    assert name.endswith("-my-deployment")
    assert sid_hash("S-1-5-21-7") in name


def test_clear_lease_env_removes_every_variable_regardless_of_platform() -> None:
    """A POSIX child clears the Windows variable too: a stale one is a false claim of inheritance."""
    env = {ENV_LEASE_FD: "7", ENV_LEASE_EVENT_HANDLE: "812", ENV_LEASE_CAPABILITY: "abc", "KEEP": "1"}
    clear_lease_env(env)
    assert env == {"KEEP": "1"}
    clear_lease_env(env)          # idempotent
    assert env == {"KEEP": "1"}


def test_process_creation_lock_is_reentrant_and_shared_across_callers() -> None:
    """Plan section 2.2 step 6: ONE process-creation mutex, shared by every kind of spawn."""
    entered = threading.Event()
    blocked = threading.Event()

    with process_creation_lock():
        with process_creation_lock():        # re-entrant within one thread, or a nested launch deadlocks
            def contender() -> None:
                entered.set()
                with process_creation_lock():
                    blocked.set()

            thread = threading.Thread(target=contender, daemon=True)
            thread.start()
            entered.wait(timeout=5)
            assert not blocked.wait(timeout=0.2), "an unrelated spawn entered the serialized window"
    thread.join(timeout=5)
    assert blocked.is_set()


# --------------------------------------------------------------------------------------------- #
# POSIX adapter -- acquisition, release, probing
# --------------------------------------------------------------------------------------------- #

@posix_only
def test_acquire_creates_a_user_only_registry_directory_and_lock(deployment: Path) -> None:
    adapter = adapter_for(deployment)
    adapter.acquire()
    try:
        assert deployment.is_dir()
        assert (deployment.stat().st_mode & 0o777) == 0o700
        lock = deployment / ACTIVITY_LOCK_FILENAME
        assert lock.is_file()
        assert (lock.stat().st_mode & 0o777) == 0o600
    finally:
        adapter.release()


@posix_only
def test_the_owner_descriptor_is_never_inheritable(deployment: Path) -> None:
    """Gate 7, in-process half: proved, not trusted to PEP 446 (the plan says exactly that)."""
    adapter = adapter_for(deployment)
    adapter.acquire()
    try:
        assert adapter.fd is not None
        assert os.get_inheritable(adapter.fd) is False
    finally:
        adapter.release()


@posix_only
def test_a_second_adapter_in_the_same_process_is_refused(deployment: Path) -> None:
    """flock treats two open file descriptions independently, even inside one process."""
    first = adapter_for(deployment)
    first.acquire()
    second = adapter_for(deployment)
    try:
        with pytest.raises(RuntimeBusyError) as excinfo:
            second.acquire()
        assert excinfo.value.deployment_key == KEY
        assert excinfo.value.exit_code == EXIT_CODE_BUSY
        assert second.fd is None, "the loser kept a descriptor it never got the lock on"
    finally:
        first.release()


@posix_only
def test_double_acquire_is_a_lease_error_not_a_silent_no_op(deployment: Path) -> None:
    adapter = adapter_for(deployment)
    adapter.acquire()
    try:
        with pytest.raises(LeaseError):
            adapter.acquire()
    finally:
        adapter.release()


@posix_only
def test_release_is_idempotent_and_never_unlinks_the_lock(deployment: Path) -> None:
    """The inode is the identity a child validates against; recreating it would fork the truth."""
    adapter = adapter_for(deployment)
    adapter.acquire()
    lock = deployment / ACTIVITY_LOCK_FILENAME
    inode = lock.stat().st_ino
    adapter.release()
    adapter.release()
    assert adapter.is_held() is False
    assert lock.is_file()
    again = adapter_for(deployment)
    again.acquire()
    try:
        assert lock.stat().st_ino == inode
    finally:
        again.release()


@posix_only
def test_probe_is_non_destructive_and_answers_about_our_own_lease(deployment: Path) -> None:
    adapter = adapter_for(deployment)
    assert adapter.probe_occupied() is False        # no lock file yet: nobody has ever held it
    adapter.acquire()
    try:
        assert adapter.probe_occupied() is True
        assert adapter.probe_occupied() is True     # probing twice does not consume anything
        assert adapter.is_held() is True
    finally:
        adapter.release()
    assert adapter.probe_occupied() is False
    # A probe that had accidentally taken the lock would make this acquire fail.
    adapter_for(deployment).acquire()


@posix_only
def test_probe_deployment_occupied_matches_the_adapter(deployment: Path) -> None:
    assert probe_deployment_occupied(deployment, deployment_key=KEY) is False
    adapter = adapter_for(deployment)
    adapter.acquire()
    try:
        assert probe_deployment_occupied(deployment, deployment_key=KEY) is True
    finally:
        adapter.release()
    assert probe_deployment_occupied(deployment, deployment_key=KEY) is False


# --------------------------------------------------------------------------------------------- #
# The probe is an OBSERVER -- invariant 13, and the two ways it used to stop being one
# --------------------------------------------------------------------------------------------- #

@posix_only
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="counting descriptors needs /proc")
def test_probing_an_occupied_deployment_leaks_no_descriptor(deployment: Path, tmp_path: Path) -> None:
    """The OCCUPIED branch is the one a live run takes on every status frame (plan section 2.6).

    A real holder in a real subprocess, because that is the only shape in which the occupied branch
    is reached honestly. One leaked descriptor per poll would kill the long-lived server with
    EMFILE inside the first hour of any run, so the count is the assertion.
    """
    ready, stop = tmp_path / "ready", tmp_path / "stop"
    holder = spawn("root-hold", helper_env(deployment, LM3_TEST_READY=ready, LM3_TEST_STOP=stop))
    try:
        wait_for(ready)
        observer = adapter_for(deployment)
        assert observer.probe_occupied() is True      # warm the interpreter before counting
        before = len(os.listdir("/proc/self/fd"))
        for _ in range(200):
            assert observer.probe_occupied() is True
        assert len(os.listdir("/proc/self/fd")) == before, "the occupied branch leaked a descriptor"
    finally:
        stop.write_text("stop", encoding="utf-8")
        holder.wait(timeout=SPAWN_TIMEOUT_S)
    # The free branch has always closed its descriptor; assert it here so the pair cannot drift.
    free_before = len(os.listdir("/proc/self/fd"))
    for _ in range(200):
        assert adapter_for(deployment).probe_occupied() is False
    assert len(os.listdir("/proc/self/fd")) == free_before


def _seed_lock_file(deployment_dir: Path) -> Path:
    """Create ``activity.lock`` and leave the deployment FREE, as a finished run does."""
    seed = adapter_for(deployment_dir)
    seed.acquire()
    seed.release()
    return deployment_dir / ACTIVITY_LOCK_FILENAME


@posix_only
def test_a_contender_waits_out_an_observers_momentary_lock(deployment: Path) -> None:
    """Invariant 13 from the CONTENDER's side: an observer must not be able to refuse a root.

    Linux has no query-only ``flock``, so a probe holds something for a microsecond. Parking a
    probe-shaped lock and dropping it inside the budget stands in for that window: the acquire must
    win, not exit 75 (plan section 3.3) against a deployment where nothing is running.
    """
    import fcntl                          # POSIX-only, so imported inside the POSIX-only test

    lock_path = _seed_lock_file(deployment)
    parked = os.open(lock_path, os.O_RDWR)
    fcntl.flock(parked, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def unpark() -> None:
        time.sleep(0.05)                  # well inside lease_mod._ACQUIRE_CONTENTION_BUDGET_S
        fcntl.flock(parked, fcntl.LOCK_UN)

    releaser = threading.Thread(target=unpark, daemon=True)
    releaser.start()
    contender = adapter_for(deployment)
    try:
        contender.acquire()               # a bare LOCK_EX|LOCK_NB here would raise RuntimeBusyError
        assert contender.is_held() is True
    finally:
        releaser.join(timeout=5)
        contender.release()
        os.close(parked)


@posix_only
def test_an_observer_never_makes_a_contender_exit_busy(deployment: Path) -> None:
    """One observer polling flat out beside one acquire/release loop: zero spurious refusals."""
    _seed_lock_file(deployment)
    stop = threading.Event()
    observed: list[bool] = []

    def observe() -> None:
        while not stop.is_set():
            observed.append(adapter_for(deployment).probe_occupied())

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    spurious = 0
    try:
        for _ in range(200):
            contender = adapter_for(deployment)
            try:
                contender.acquire()
            except RuntimeBusyError:
                spurious += 1             # nobody else acquires here, so any refusal is spurious
            else:
                contender.release()
    finally:
        stop.set()
        observer.join(timeout=10)
    assert spurious == 0, f"{spurious} of 200 legitimate acquisitions were refused by an observer"
    assert observed, "the observer thread never ran, so this proved nothing"


@posix_only
def test_two_observers_do_not_invent_an_occupied_deployment(deployment: Path) -> None:
    """Two concurrent readers -- the section 2.6 GUI poll plus any other ``read_runtime`` caller.

    An exclusive probe would have each refuse the other and report ``active: true`` on an idle
    machine, which disables Start (gate 51 classification) while nothing is running.
    """
    _seed_lock_file(deployment)
    phantom: list[int] = []

    def observe() -> None:
        for _ in range(2000):
            if adapter_for(deployment).probe_occupied():
                phantom.append(1)

    threads = [threading.Thread(target=observe, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not phantom, f"{len(phantom)} probes reported an occupied deployment with no holder"


@posix_only
def test_a_real_holder_is_still_refused_within_the_bounded_budget(deployment: Path, monkeypatch) -> None:
    """Gate 36: a busy start is an immediate refusal carrying the winner, never a late failure.

    The budget is a module constant precisely so a test can shrink it; the shipped value is
    asserted small here as well, because an unbounded one would turn every busy start into a hang.
    """
    monkeypatch.setattr(lease_mod, "_ACQUIRE_CONTENTION_BUDGET_S", 0.05)
    holder = adapter_for(deployment)
    holder.acquire()
    try:
        loser = adapter_for(deployment)
        started = time.monotonic()
        with pytest.raises(RuntimeBusyError) as excinfo:
            loser.acquire()
        elapsed = time.monotonic() - started
        assert excinfo.value.deployment_key == KEY
        assert excinfo.value.exit_code == EXIT_CODE_BUSY
        assert loser.fd is None, "the loser kept a descriptor it never got the lock on"
        assert 0.05 <= elapsed < 5.0, f"the refusal took {elapsed}s, which is not the bounded budget"
    finally:
        holder.release()


def test_the_shipped_contention_budget_is_short_enough_to_stay_an_immediate_refusal() -> None:
    """Gate 36 again, against the value that actually ships rather than a test's shrunken one."""
    assert 0 < lease_mod._ACQUIRE_CONTENTION_BUDGET_S <= 0.5
    assert 0 < lease_mod._ACQUIRE_RETRY_SLEEP_S < lease_mod._ACQUIRE_CONTENTION_BUDGET_S


# --------------------------------------------------------------------------------------------- #
# Handing the reference to an approved subactivity
# --------------------------------------------------------------------------------------------- #

@posix_only
def test_child_handoff_carries_exactly_the_descriptor_and_the_capability(deployment: Path) -> None:
    adapter = adapter_for(deployment)
    adapter.acquire()
    try:
        with adapter.child_handoff("cap-value") as handoff:
            assert handoff.capability == "cap-value"
            assert handoff.lease_fd == adapter.fd
            assert handoff.env == {ENV_LEASE_FD: str(adapter.fd), ENV_LEASE_CAPABILITY: "cap-value"}
            assert handoff.popen_kwargs == {"pass_fds": (adapter.fd,)}
            assert handoff.lease_handle is None
        # The POSIX handoff must NOT close anything: the descriptor is the root's own and the root
        # holds it for the whole activity.
        assert adapter.is_held() is True
        assert os.get_inheritable(adapter.fd) is False
    finally:
        adapter.release()


@posix_only
def test_child_handoff_without_the_lease_is_refused(deployment: Path) -> None:
    adapter = adapter_for(deployment)
    with pytest.raises(LeaseNotHeldError):
        with adapter.child_handoff("cap"):
            pass


# --------------------------------------------------------------------------------------------- #
# Child-side validation (gate 9)
# --------------------------------------------------------------------------------------------- #

@posix_only
def test_a_shared_open_file_description_validates(deployment: Path) -> None:
    """``os.dup`` models inheritance exactly: a new descriptor onto the SAME description."""
    owner = adapter_for(deployment)
    owner.acquire()
    inherited_fd = os.dup(owner.fd)                      # type: ignore[arg-type]
    try:
        child = PosixLeaseAdapter.from_inherited(
            deployment, deployment_key=KEY, env={ENV_LEASE_FD: str(inherited_fd)}
        )
        child.validate_inherited()                        # must not raise
        assert child.inherited is True
    finally:
        os.close(inherited_fd)
        owner.release()


@posix_only
def test_an_independent_descriptor_on_the_right_inode_is_rejected(deployment: Path) -> None:
    """Gate 9's hard half: the inode matches, so only the non-blocking re-assert can tell."""
    owner = adapter_for(deployment)
    owner.acquire()
    forged = os.open(deployment / ACTIVITY_LOCK_FILENAME, os.O_RDWR)
    try:
        child = PosixLeaseAdapter.from_inherited(
            deployment, deployment_key=KEY, env={ENV_LEASE_FD: str(forged)}
        )
        with pytest.raises(LeaseInheritanceError, match="open file description"):
            child.validate_inherited()
    finally:
        os.close(forged)
        owner.release()


@posix_only
def test_a_descriptor_on_another_file_is_rejected_by_inode(deployment: Path, tmp_path: Path) -> None:
    owner = adapter_for(deployment)
    owner.acquire()
    import fcntl                    # POSIX-only, so imported inside the POSIX-only test

    decoy = tmp_path / "decoy"
    decoy.write_text("not the lease", encoding="utf-8")
    fd = os.open(decoy, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)   # even locked, it is the wrong inode
        child = PosixLeaseAdapter.from_inherited(
            deployment, deployment_key=KEY, env={ENV_LEASE_FD: str(fd)}
        )
        with pytest.raises(LeaseInheritanceError, match="activity lock"):
            child.validate_inherited()
    finally:
        os.close(fd)
        owner.release()


@posix_only
def test_a_closed_descriptor_is_rejected(deployment: Path) -> None:
    owner = adapter_for(deployment)
    owner.acquire()
    try:
        spare = os.open(deployment / ACTIVITY_LOCK_FILENAME, os.O_RDWR)
        os.close(spare)
        child = PosixLeaseAdapter.from_inherited(
            deployment, deployment_key=KEY, env={ENV_LEASE_FD: str(spare)}
        )
        with pytest.raises(LeaseInheritanceError):
            child.validate_inherited()
    finally:
        owner.release()


@pytest.mark.parametrize("value", [None, "", "   ", "not-a-number", "-3"])
def test_from_inherited_rejects_a_missing_or_malformed_reference(deployment: Path, value) -> None:
    env = {} if value is None else {ENV_LEASE_FD: value}
    with pytest.raises(LeaseInheritanceError):
        PosixLeaseAdapter.from_inherited(deployment, deployment_key=KEY, env=env)


@posix_only
def test_validate_inherited_without_any_reference_is_rejected(deployment: Path) -> None:
    adapter = adapter_for(deployment)
    with pytest.raises(LeaseInheritanceError):
        adapter.validate_inherited()


@posix_only
def test_disarm_and_clear_disarms_the_reference_and_then_the_environment(deployment: Path) -> None:
    """Plan section 2.2 step 4, in that order: the reference stops propagating, then the variables go."""
    owner = adapter_for(deployment)
    owner.acquire()
    inherited_fd = os.dup(owner.fd)                       # type: ignore[arg-type]
    os.set_inheritable(inherited_fd, True)                # as subprocess leaves it in a real child
    env = {ENV_LEASE_FD: str(inherited_fd), ENV_LEASE_CAPABILITY: "cap", "OTHER": "kept"}
    try:
        lease = inherit_lease(deployment_dir=deployment, deployment_key=KEY, env=env)
        lease.validate_inherited()
        assert lease.held is True
        lease.disarm_and_clear(env)
        assert os.get_inheritable(inherited_fd) is False
        assert env == {"OTHER": "kept"}
    finally:
        os.close(inherited_fd)
        owner.release()


# --------------------------------------------------------------------------------------------- #
# RuntimeLease
# --------------------------------------------------------------------------------------------- #

@posix_only
def test_runtime_lease_inherited_is_the_same_binding_as_inherit_lease(deployment: Path) -> None:
    """One binding path, two spellings -- a second implementation would be a second answer."""
    owner = adapter_for(deployment)
    owner.acquire()
    inherited_fd = os.dup(owner.fd)                       # type: ignore[arg-type]
    try:
        env = {ENV_LEASE_FD: str(inherited_fd), ENV_LEASE_CAPABILITY: "cap"}
        lease = RuntimeLease.inherited(deployment_dir=deployment, deployment_key=KEY, env=env)
        assert isinstance(lease.adapter, PosixLeaseAdapter)
        assert lease.adapter.fd == inherited_fd
        assert lease.held is True
        lease.validate_inherited()
    finally:
        os.close(inherited_fd)
        owner.release()


@posix_only
def test_runtime_lease_is_a_context_manager(deployment: Path) -> None:
    with RuntimeLease(deployment_dir=deployment, deployment_key=KEY) as lease:
        assert lease.held is True
        assert lease.probe_occupied() is True
    assert probe_deployment_occupied(deployment, deployment_key=KEY) is False


@posix_only
def test_busy_error_names_the_deployment_even_with_no_readable_record(deployment: Path) -> None:
    """Exclusivity comes from the lock: a missing record does not make a deployment free."""
    with RuntimeLease(deployment_dir=deployment, deployment_key=KEY):
        loser = RuntimeLease(deployment_dir=deployment, deployment_key=KEY)
        with pytest.raises(RuntimeBusyError) as excinfo:
            loser.acquire()
    assert excinfo.value.deployment_key == KEY
    assert excinfo.value.active is None
    assert loser.held is False


@posix_only
def test_busy_error_carries_the_sanitized_winner_when_a_record_exists() -> None:
    """Plan section 3.3 and gate 36: the loser reports the WINNER's identity, not a bare refusal.

    ``records`` is a sibling module owned elsewhere, so this is skipped rather than failed if it is
    mid-edit -- the contract between the two is what is being checked, not that module's internals.
    """
    records = pytest.importorskip("leafmachine3.core.runtime.records")
    from leafmachine3.core.runtime._types import (
        Activity, ActivityRole, ArchiveMode, ArchiveStatus, ConfigRef, DeploymentInfo, Launcher,
        ProjectBlock, RunState, RuntimeRecord,
    )

    import tempfile

    root = Path(tempfile.mkdtemp())
    deployment_dir = root / "runtime" / KEY
    run_dir = root / "run"
    record = RuntimeRecord(
        run_id=records.new_run_id(), activity=Activity.PIPELINE, activity_role=ActivityRole.ROOT,
        state=RunState.RUNNING, launcher=Launcher.CLI, pid=os.getpid(),
        process_started_at=records.process_start_time(), started_at=records.utc_now(),
        updated_at=records.utc_now(), deployment=DeploymentInfo(id=KEY),
        config=ConfigRef(path=str(root / "LM3_settings.yaml"), sha256="0" * 64),
        project=ProjectBlock(
            run_name="winner", input_dirs=[str(root / "in")], artifact_dir=str(run_dir),
            active_state_dir=str(run_dir), active_db_path=str(run_dir / "winner.sqlite"),
            archive_mode=ArchiveMode.IN_PLACE, archive_status=ArchiveStatus.NOT_APPLICABLE,
            archive_pointer_path=None, archived_db_path=str(run_dir / "winner.sqlite"),
            run_dir=str(run_dir), log_path=str(run_dir / "logs" / "run.log"),
        ),
    )
    owner = RuntimeLease(deployment_dir=deployment_dir, deployment_key=KEY)
    owner.acquire()
    try:
        records.RecordStore(deployment_dir, run_id=record.run_id).write_active(record)
        loser = RuntimeLease(deployment_dir=deployment_dir, deployment_key=KEY)
        with pytest.raises(RuntimeBusyError) as excinfo:
            loser.acquire()
    finally:
        owner.release()
    busy = excinfo.value
    assert busy.active is not None
    assert busy.active.run_id == record.run_id
    assert record.run_id in str(busy)
    assert "pipeline" in str(busy)


@posix_only
def test_lease_adapter_dispatches_on_the_injected_platform(deployment: Path) -> None:
    assert isinstance(lease_adapter(deployment, deployment_key=KEY, platform_name="linux"),
                      PosixLeaseAdapter)
    assert isinstance(lease_adapter(deployment, deployment_key=KEY, platform_name="darwin"),
                      PosixLeaseAdapter)
    windows = lease_adapter(deployment, deployment_key=KEY, platform_name="win32", win32=object())
    assert isinstance(windows, WindowsLeaseAdapter)


def test_lease_imports_cleanly_on_a_platform_without_fcntl() -> None:
    """Windows has no ``fcntl``, and ``lease.py`` still has to import there -- the Windows adapter
    lives in it. Run in a subprocess so blocking the module cannot leak into this session.
    """
    program = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'fcntl':\n"
        "            raise ImportError('no fcntl on this platform')\n"
        "        return None\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'fcntl':\n"
        "            raise ImportError('no fcntl on this platform')\n"
        "        return None\n"
        "sys.modules.pop('fcntl', None)\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from leafmachine3.core.runtime import lease\n"
        "assert lease.fcntl is None\n"
        "assert lease.WindowsLeaseAdapter is not None\n"
        "print('ok')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    result = subprocess.run(                                          # noqa: S603 - fixed argv
        [sys.executable, "-c", program], env=env, capture_output=True, text=True,
        timeout=SPAWN_TIMEOUT_S,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_real_win32_surface_refuses_to_pretend_off_windows() -> None:
    """A stub surface would let the Windows lease 'succeed' where there is no lease at all."""
    if sys.platform == "win32":                                       # pragma: no cover - Windows CI
        assert real_win32_surface() is not None
        return
    with pytest.raises(LeaseError):
        real_win32_surface()


# --------------------------------------------------------------------------------------------- #
# Acquisition ordering (plan section 3.3) and abandoned-record cleanup
# --------------------------------------------------------------------------------------------- #

@posix_only
@pytest.mark.parametrize("activity", [Activity.CALIBRATION_PIPELINE, Activity.BATCH_ITEM_PIPELINE])
def test_a_subactivity_may_never_acquire_a_lease(deployment: Path, activity: Activity) -> None:
    """Invariant 2: a child INHERITS. Two leases would mean killing the parent frees one of them."""
    with pytest.raises(LeaseError, match="subactivity"):
        acquire_root_lease(activity, deployment_dir=deployment, deployment_key=KEY, env={})
    assert probe_deployment_occupied(deployment, deployment_key=KEY) is False


@posix_only
@pytest.mark.parametrize("variable", [ENV_LEASE_FD, ENV_LEASE_EVENT_HANDLE])
def test_a_process_holding_an_inherited_reference_may_not_acquire(deployment: Path, variable: str) -> None:
    with pytest.raises(LeaseError, match="inherited a lease reference"):
        acquire_root_lease(
            Activity.PIPELINE, deployment_dir=deployment, deployment_key=KEY, env={variable: "9"}
        )
    assert probe_deployment_occupied(deployment, deployment_key=KEY) is False


@contextlib.contextmanager
def captured_lease_warnings():
    """Capture the lease logger directly.

    Not ``caplog``: its handler sits on the ROOT logger, so any module in this large suite that
    turns off propagation somewhere in the ``leafmachine3`` chain silently empties it. Attaching to
    the logger under test makes the assertion independent of what else has been imported.
    """
    import logging

    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("leafmachine3.runtime.lease")
    handler = Collector(level=logging.WARNING)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


@posix_only
def test_acquire_root_lease_reports_a_cuda_context_that_preceded_it(deployment: Path) -> None:
    """Plan section 3.3 orders the lease BEFORE model loading and GPU work. Reported, never fatal."""
    with captured_lease_warnings() as records:
        lease = acquire_root_lease(
            Activity.PIPELINE, deployment_dir=deployment, deployment_key=KEY, env={},
            cuda_initialized=lambda: True,
        )
        try:
            assert lease.held is True
            assert any("CUDA context" in record.getMessage() for record in records)
        finally:
            lease.release()


@posix_only
def test_acquire_root_lease_is_quiet_when_the_ordering_is_right(deployment: Path) -> None:
    with captured_lease_warnings() as records:
        lease = acquire_root_lease(
            Activity.PIPELINE, deployment_dir=deployment, deployment_key=KEY, env={},
            cuda_initialized=lambda: False,
        )
        lease.release()
    assert [record for record in records if "CUDA" in record.getMessage()] == []


@posix_only
def test_cleanup_lease_yields_the_held_lease_and_releases_it(deployment: Path) -> None:
    with cleanup_lease(deployment, deployment_key=KEY) as lease:
        assert lease is not None
        assert lease.held is True
    assert probe_deployment_occupied(deployment, deployment_key=KEY) is False


@posix_only
def test_cleanup_lease_yields_none_while_the_deployment_is_live(deployment: Path) -> None:
    """A record under a HELD lease is live, not abandoned -- there is nothing to clean up."""
    with RuntimeLease(deployment_dir=deployment, deployment_key=KEY):
        with cleanup_lease(deployment, deployment_key=KEY) as lease:
            assert lease is None


def test_the_lease_module_never_signals_a_process() -> None:
    """Invariant 12 / section 3.3: a PID in a JSON file is never grounds to signal anything.

    A source assertion rather than a behavioral one, because the property is an ABSENCE: no call
    path can be exercised to prove that a call which does not exist was not made.
    """
    import ast

    tree = ast.parse(Path(lease_mod.__file__).read_text(encoding="utf-8"))
    # The AST, not the text: prose that DISCUSSES os.kill is exactly what this file should contain.
    called: set[str] = set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            called.add(node.attr)
        elif isinstance(node, ast.Name):
            called.add(node.id)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    for forbidden in ("kill", "killpg", "terminate", "send_signal", "pthread_kill"):
        assert forbidden not in called, f"lease.py must never signal a process, found {forbidden!r}"
    assert "signal" not in imported


# --------------------------------------------------------------------------------------------- #
# Real processes -- the properties only the kernel can settle
# --------------------------------------------------------------------------------------------- #

@posix_only
def test_two_processes_contend_and_exactly_one_wins(tmp_path: Path) -> None:
    """Gate 39 in its simplest form, and the exit code section 3.3 assigns to a busy start."""
    deployment = tmp_path / "runtime" / KEY
    ready, stop = tmp_path / "ready", tmp_path / "stop"
    winner_report, loser_report = tmp_path / "winner.json", tmp_path / "loser.json"
    winner = spawn("root-hold", helper_env(
        deployment, LM3_TEST_READY=ready, LM3_TEST_STOP=stop, LM3_TEST_REPORT=winner_report))
    try:
        wait_for(ready)
        loser = spawn("contend", helper_env(deployment, LM3_TEST_REPORT=loser_report))
        assert loser.wait(timeout=SPAWN_TIMEOUT_S) == EXIT_CODE_BUSY
        assert report(loser_report)["result"] == "busy"
        assert report(loser_report)["deployment_key"] == KEY
    finally:
        stop.write_text("go", encoding="utf-8")
        assert winner.wait(timeout=SPAWN_TIMEOUT_S) == 0
    assert report(winner_report)["result"] == "acquired"
    # And the deployment is genuinely free afterwards.
    second_report = tmp_path / "second.json"
    later = spawn("contend", helper_env(
        deployment, LM3_TEST_REPORT=second_report, LM3_TEST_STOP=stop))
    assert later.wait(timeout=SPAWN_TIMEOUT_S) == 0
    assert report(second_report)["result"] == "acquired"


@posix_only
def test_two_default_deployments_contend(tmp_path: Path) -> None:
    """Gate 39 as stated: neither process is told its deployment, and section 2.1's "unset means
    the raw literal default" plus section 3.1's ``<base>/<deployment_id>`` are what put them on one
    lock -- resolved independently in each child, from different working directories.

    The test above proves the ADAPTER excludes when both processes are handed the same directory and
    the same literal key. That is a strictly weaker claim: a resolver that derived a different key
    per process would still pass it, and gate 39 is about the DEFAULT identity in particular
    ("two checkouts under the default deployment coordinate. That is intentional."). So this one
    passes neither ``deployment_dir`` nor ``deployment_key`` and lets ``RuntimeLease`` ask
    ``paths``, which is the branch the rest of the suite never reaches.
    """
    base = tmp_path / "runtime"
    script = tmp_path / "default_deployment_child.py"
    script.write_text(DEFAULT_DEPLOYMENT_CHILD, encoding="utf-8")
    ready, stop = tmp_path / "ready", tmp_path / "stop"
    winner_report, loser_report = tmp_path / "winner.json", tmp_path / "loser.json"
    winner = spawn("root-hold", default_helper_env(
        base, LM3_TEST_READY=ready, LM3_TEST_STOP=stop, LM3_TEST_REPORT=winner_report),
        cwd=str(REPO_ROOT), script=script)
    try:
        wait_for(ready)
        # A DIFFERENT working directory on purpose: section 3.1's "keep it independent of the
        # checkout and CWD" is the half a test that hands over an absolute directory cannot see.
        loser = spawn("contend", default_helper_env(base, LM3_TEST_REPORT=loser_report),
                      cwd=str(tmp_path), script=script)
        assert loser.wait(timeout=SPAWN_TIMEOUT_S) == EXIT_CODE_BUSY, loser.communicate()
    finally:
        stop.write_text("go", encoding="utf-8")
        assert winner.wait(timeout=SPAWN_TIMEOUT_S) == 0, winner.communicate()

    held, refused = report(winner_report), report(loser_report)
    assert held["result"] == "acquired" and refused["result"] == "busy"
    # The canonical key, not the raw literal: section 2.1 makes the hash the authority, and
    # ``deployment_key({})`` is the same derivation Electron's golden vectors pin.
    assert refused["deployment_key"] == P.canonical_deployment_key("default") == P.deployment_key({})
    assert refused["busy_key"] == refused["deployment_key"], "the busy error named another deployment"
    # One deployment directory, derived twice, from two working directories.
    assert held["deployment_dir"] == str(base / P.deployment_key({})) == refused["deployment_dir"]
    assert held["cwd"] != refused["cwd"]
    # And it landed under the tmp base -- nowhere near the developer's real default runtime dir.
    assert (base / P.deployment_key({}) / ACTIVITY_LOCK_FILENAME).exists()


@posix_only
def test_an_approved_child_inherits_the_lease_and_a_worker_does_not(tmp_path: Path) -> None:
    """Invariants 2 and 3 in one real tree (gate 7): the child validates, the worker sees nothing."""
    deployment = tmp_path / "runtime" / KEY
    ready, child_ready, stop = tmp_path / "ready", tmp_path / "child-ready", tmp_path / "stop"
    root_report = tmp_path / "root.json"
    child_report, worker_report = tmp_path / "child.json", tmp_path / "worker.json"
    root = spawn("root-with-child", helper_env(
        deployment, LM3_TEST_READY=ready, LM3_TEST_STOP=stop, LM3_TEST_REPORT=root_report,
        LM3_TEST_CHILD_READY=child_ready, LM3_TEST_CHILD_REPORT=child_report,
        LM3_TEST_WORKER_REPORT=worker_report))
    try:
        wait_for(child_ready)
        wait_for(ready)
    finally:
        stop.write_text("go", encoding="utf-8")
        assert root.wait(timeout=SPAWN_TIMEOUT_S) == 0, root.communicate()

    child = report(child_report)
    assert child["result"] == "validated"
    lock_stat = (deployment / ACTIVITY_LOCK_FILENAME).stat()
    assert child["fd_identity"] == {"open": True, "dev": lock_stat.st_dev, "ino": lock_stat.st_ino}
    # Step 4: the reference stops propagating and the variables are gone, BEFORE any worker spawn.
    assert child["inheritable"] is False
    assert child["lease_env_after_disarm"] == {
        ENV_LEASE_FD: None, ENV_LEASE_EVENT_HANDLE: None, ENV_LEASE_CAPABILITY: None}

    worker = child["worker"]
    assert worker is not None and worker["result"] == "worker"
    assert worker["fd_identity"] != child["fd_identity"], "an executor worker inherited the lease"
    assert set(worker["lease_env"].values()) == {None}


@posix_only
def test_a_worker_spawned_by_the_root_never_inherits_the_lease(tmp_path: Path) -> None:
    """The case a child-side disarm cannot help with: the ROOT spawns the worker (gate 7, gate 33's
    POSIX analogue). POSIX is safe by construction -- ``pass_fds`` is a per-spawn opt-in -- and this
    is the test the plan asks for instead of trusting that."""
    deployment = tmp_path / "runtime" / KEY
    ready, stop = tmp_path / "ready", tmp_path / "stop"
    root_report, worker_report = tmp_path / "root.json", tmp_path / "worker.json"
    root = spawn("root-with-worker", helper_env(
        deployment, LM3_TEST_READY=ready, LM3_TEST_STOP=stop, LM3_TEST_REPORT=root_report,
        LM3_TEST_WORKER_REPORT=worker_report))
    try:
        wait_for(ready)
    finally:
        stop.write_text("go", encoding="utf-8")
        assert root.wait(timeout=SPAWN_TIMEOUT_S) == 0, root.communicate()
    result = report(root_report)
    assert result["owner_fd_inheritable"] is False
    lock_stat = (deployment / ACTIVITY_LOCK_FILENAME).stat()
    worker_identity = result["worker"]["fd_identity"]
    assert worker_identity != {"open": True, "dev": lock_stat.st_dev, "ino": lock_stat.st_ino}
    assert set(result["worker"]["lease_env"].values()) == {None}


@posix_only
def test_killing_the_root_keeps_the_deployment_occupied_while_the_child_lives(tmp_path: Path) -> None:
    """Gate 4, and the whole reason the child inherits the reference rather than re-checking it.

    Revision 1 had the child merely validate that the root's lock was held. Kill the root after that
    and the OS releases the lock while the child runs on -- two overlapping LM3 runs.
    """
    deployment = tmp_path / "runtime" / KEY
    ready, child_ready, stop = tmp_path / "ready", tmp_path / "child-ready", tmp_path / "stop"
    root_report, child_report = tmp_path / "root.json", tmp_path / "child.json"
    root = spawn("root-with-child", helper_env(
        deployment, LM3_TEST_READY=ready, LM3_TEST_STOP=stop, LM3_TEST_REPORT=root_report,
        LM3_TEST_CHILD_READY=child_ready, LM3_TEST_CHILD_REPORT=child_report))
    wait_for(child_ready)
    wait_for(ready)

    root.send_signal(signal.SIGKILL)
    root.wait(timeout=SPAWN_TIMEOUT_S)
    assert root.returncode == -signal.SIGKILL

    # The root is gone. The child still holds the inherited open file description, so the deployment
    # is still occupied and a second root must stay out.
    contender_report = tmp_path / "contender.json"
    contender = spawn("contend", helper_env(deployment, LM3_TEST_REPORT=contender_report))
    assert contender.wait(timeout=SPAWN_TIMEOUT_S) == EXIT_CODE_BUSY
    assert report(contender_report)["result"] == "busy"

    # Only when the last reference goes does the deployment become free.
    stop.write_text("go", encoding="utf-8")
    wait_until(lambda: not probe_deployment_occupied(deployment, deployment_key=KEY))
    after_report = tmp_path / "after.json"
    after = spawn("contend", helper_env(deployment, LM3_TEST_REPORT=after_report, LM3_TEST_STOP=stop))
    assert after.wait(timeout=SPAWN_TIMEOUT_S) == 0
    assert report(after_report)["result"] == "acquired"


@posix_only
def test_a_bogus_inherited_descriptor_is_rejected_in_a_real_child(tmp_path: Path) -> None:
    """Gate 9 across a process boundary, and gate 22's precondition: a child that fails validation
    has done nothing -- it never reaches the grant, so the legitimate child's grant is untouched."""
    deployment = tmp_path / "runtime" / KEY
    ready, stop = tmp_path / "ready", tmp_path / "stop"
    root_report, bogus_report = tmp_path / "root.json", tmp_path / "bogus.json"
    root = spawn("root-hold", helper_env(
        deployment, LM3_TEST_READY=ready, LM3_TEST_STOP=stop, LM3_TEST_REPORT=root_report))
    try:
        wait_for(ready)
        bogus = spawn("bogus-child", helper_env(deployment, LM3_TEST_REPORT=bogus_report))
        assert bogus.wait(timeout=SPAWN_TIMEOUT_S) == 0, bogus.communicate()
        result = report(bogus_report)
        assert result["result"] == "rejected"
        assert result["error"] == "LeaseInheritanceError"
        children = deployment / "children"
        assert not children.exists() or list(children.iterdir()) == []
    finally:
        stop.write_text("go", encoding="utf-8")
        root.wait(timeout=SPAWN_TIMEOUT_S)


@posix_only
def test_a_killed_lone_owner_releases_the_lock_and_leaves_its_record_behind(tmp_path: Path) -> None:
    """Gate 51 and section 3.3's classification rule.

    The kernel releases the lock when the last descriptor closes, which for a hard-killed lone owner
    is immediately -- but ``active.json`` survives, because a dead process cannot finalize. That
    combination (lock free, record present) is what the next reader classifies as abandoned, under
    the cleanup lock, WITHOUT signaling the recorded PID.
    """
    deployment = tmp_path / "runtime" / KEY
    ready, stop = tmp_path / "ready", tmp_path / "stop"
    root_report = tmp_path / "root.json"
    root = spawn("root-hold", helper_env(
        deployment, LM3_TEST_READY=ready, LM3_TEST_STOP=stop, LM3_TEST_REPORT=root_report))
    wait_for(ready)
    # Stand in for the record the root would have written; records.py owns the real writer.
    active = deployment / ACTIVE_RECORD_FILENAME
    active.write_text(json.dumps({"run_id": "abandoned-run", "pid": root.pid}), encoding="utf-8")

    root.send_signal(signal.SIGKILL)
    root.wait(timeout=SPAWN_TIMEOUT_S)

    wait_until(lambda: not probe_deployment_occupied(deployment, deployment_key=KEY))
    assert active.is_file(), "the stale root record is the only description of what was running"
    with cleanup_lease(deployment, deployment_key=KEY) as lease:
        assert lease is not None, "a free lease must be acquirable for classification"
        assert json.loads(active.read_text(encoding="utf-8"))["run_id"] == "abandoned-run"
    assert probe_deployment_occupied(deployment, deployment_key=KEY) is False
