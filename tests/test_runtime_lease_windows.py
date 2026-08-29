"""The WINDOWS lease adapter, exercised on this Linux host against a fake Win32 surface.

Gate 27 is explicit: the Windows behavior must be "verified against the Windows adapter itself, not
inferred from the POSIX result". The two adapters share no mechanism -- POSIX exclusion is a
``flock`` on an open file description, Windows exclusion is the EXISTENCE of a named kernel event --
so a POSIX pass says nothing about Windows. But this host is Linux, where ``CreateEventW`` does not
exist, which is why every Win32 call in the adapter goes through the injected
:class:`~leafmachine3.core.runtime._types.Win32Surface`.

:class:`FakeKernel` below is a small model of the four parts of the Win32 object manager the lease
depends on, and it is the load-bearing piece of this file:

* a named object exists for exactly as long as one handle to it is open -- not while its creator
  lives. That is the whole reason the Windows lease is an event and not ``LockFileEx``;
* names beginning ``Global\\`` live in a MACHINE-WIDE namespace, names beginning ``Local\\`` in a
  per-logon-session one -- which is what gate 28 turns on;
* a handle is a per-process table entry, so an inherited handle is a SECOND reference to one object;
* ``CreateProcess(bInheritHandles=TRUE)`` copies exactly the inheritable handles, narrowed to the
  ``STARTUPINFOEX`` allowlist when one is given -- which is what gates 32 and 33 turn on.

The one thing a fake cannot prove is that the real ``CreateEventW`` behaves like the model; the
``skipif`` tests at the bottom are the Windows-CI half of that, and gate 17 covers the launch
handshake separately.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from leafmachine3.core.runtime import _win32
from leafmachine3.core.runtime._types import (
    ACTIVITY_LOCK_FILENAME,
    ENV_LEASE_CAPABILITY,
    ENV_LEASE_EVENT_HANDLE,
    ENV_LEASE_FD,
    ERROR_ALREADY_EXISTS,
    EVENT_ALL_ACCESS,
    HANDLE_FLAG_INHERIT,
    LeaseError,
    LeaseInheritanceError,
    LeaseNotHeldError,
    RuntimeBusyError,
)
from leafmachine3.core.runtime.lease import (
    RuntimeLease,
    WindowsLeaseAdapter,
    process_creation_lock,
    sid_hash,
    windows_lease_event_name,
)

KEY = "pytest-lease"
SID_A = "S-1-5-21-1004336348-1177238915-682003330-1001"
SID_B = "S-1-5-21-1004336348-1177238915-682003330-1002"
GLOBAL_PREFIX = "Global" + "\\"
LOCAL_PREFIX = "Local" + "\\"


# --------------------------------------------------------------------------------------------- #
# The fake Win32 object manager
# --------------------------------------------------------------------------------------------- #

class FakeHandle:
    """One entry in one process's handle table. ``inheritable`` is ``HANDLE_FLAG_INHERIT``."""

    def __init__(self, value: int, name: str, *, pid: int, session: int, inheritable: bool) -> None:
        self.value = value
        self.name = name
        self.pid = pid
        self.session = session
        self.inheritable = inheritable


class FakeKernel:
    """A machine. Every :class:`FakeWin32` is one process's view of it."""

    def __init__(self) -> None:
        #: (pid, handle value) -> entry. Handle values are unique at allocation and are duplicated
        #: into a child's table by inheritance, exactly as Windows does.
        self.refs: dict[tuple[int, int], FakeHandle] = {}
        self._next_handle = 0x100
        self._next_pid = 1
        #: Every name ever passed to CreateEventW, for the "no Local\\ fallback" assertions.
        self.seen_names: list[str] = []

    # -- namespaces ----------------------------------------------------------------------------- #

    @staticmethod
    def namespace_key(name: str, session: int) -> tuple[int, str]:
        """Machine-wide for ``Global``, per-logon-session otherwise."""
        return (0, name) if name.startswith(GLOBAL_PREFIX) else (session, name)

    def exists(self, name: str, session: int) -> bool:
        """True while ANY process still holds a handle to that name in that namespace."""
        wanted = self.namespace_key(name, session)
        return any(
            self.namespace_key(entry.name, entry.session) == wanted for entry in self.refs.values()
        )

    # -- handles -------------------------------------------------------------------------------- #

    def new_pid(self) -> int:
        self._next_pid += 1
        return self._next_pid

    def allocate(self, name: str, *, pid: int, session: int, inheritable: bool) -> int:
        value = self._next_handle
        self._next_handle += 1
        self.refs[(pid, value)] = FakeHandle(
            value, name, pid=pid, session=session, inheritable=inheritable
        )
        self.seen_names.append(name)
        return value

    def close(self, value: int, pid: int) -> None:
        if self.refs.pop((pid, value), None) is None:
            raise OSError(6, f"invalid handle {value} in process {pid}")

    def entry(self, value: int, pid: int) -> FakeHandle:
        return self.refs[(pid, value)]

    def handles_for(self, name: str) -> list[FakeHandle]:
        return [entry for entry in self.refs.values() if entry.name == name]

    def inheritable_handles(self, pid: int | None = None) -> list[FakeHandle]:
        return [
            entry for entry in self.refs.values()
            if entry.inheritable and (pid is None or entry.pid == pid)
        ]

    def create_process(self, *, parent_pid: int, child_pid: int, session: int,
                       handle_list: list[int] | None = None) -> list[int]:
        """``CreateProcess(bInheritHandles=TRUE)``.

        Without a ``STARTUPINFOEX`` allowlist -- an ordinary executor worker -- every inheritable
        handle in the parent is copied. With one, only the allowlisted handles are, and they must
        still be inheritable. The returned list is what the child ends up holding.
        """
        inherited: list[int] = []
        for (pid, value), entry in list(self.refs.items()):
            if pid != parent_pid or not entry.inheritable:
                continue
            if handle_list is not None and value not in handle_list:
                continue
            self.refs[(child_pid, value)] = FakeHandle(
                value, entry.name, pid=child_pid, session=session, inheritable=entry.inheritable
            )
            inherited.append(value)
        return inherited


class FakeSecurityAttributes:
    """What ``current_user_security_attributes`` hands ``CreateEventW``: one user, EVENT_ALL_ACCESS."""

    def __init__(self, sid: str) -> None:
        self.sid = sid
        self.access = EVENT_ALL_ACCESS
        self.inherit_handle = False


class FakeWin32:
    """One process's view of a :class:`FakeKernel`. Records every call, in order."""

    def __init__(self, kernel: FakeKernel, *, sid: str = SID_A, session: int = 1,
                 pid: int | None = None) -> None:
        self.kernel = kernel
        self.sid = sid
        self.session = session
        self.pid = kernel.new_pid() if pid is None else pid
        self.calls: list[tuple] = []
        self.last_error = 0
        #: Set to an OSError to make the next create_event / open_event fail.
        self.create_error: OSError | None = None
        self.open_error: OSError | None = None

    # -- events --------------------------------------------------------------------------------- #

    def create_event(self, name: str, *, security_attributes: Any, manual_reset: bool = False,
                     initial_state: bool = False, inheritable: bool = False) -> int:
        self.calls.append(("create_event", name, inheritable, security_attributes))
        if self.create_error is not None:
            raise self.create_error
        existed = self.kernel.exists(name, self.session)
        handle = self.kernel.allocate(
            name, pid=self.pid, session=self.session, inheritable=inheritable
        )
        # The atomic test-and-create: a valid handle to the EXISTING object, plus this error code.
        self.last_error = ERROR_ALREADY_EXISTS if existed else 0
        return handle

    def open_event(self, name: str, *, desired_access: int = EVENT_ALL_ACCESS,
                   inheritable: bool = False) -> int:
        self.calls.append(("open_event", name, desired_access, inheritable))
        if self.open_error is not None:
            raise self.open_error
        if not self.kernel.exists(name, self.session):
            self.last_error = 2
            raise OSError(2, f"no such object {name}")
        self.last_error = 0
        return self.kernel.allocate(
            name, pid=self.pid, session=self.session, inheritable=inheritable
        )

    # -- handles -------------------------------------------------------------------------------- #

    def close_handle(self, handle: int) -> None:
        self.calls.append(("close_handle", handle))
        self.kernel.close(handle, self.pid)

    def duplicate_handle(self, handle: int, *, inheritable: bool) -> int:
        self.calls.append(("duplicate_handle", handle, inheritable))
        source = self.kernel.entry(handle, self.pid)
        return self.kernel.allocate(
            source.name, pid=self.pid, session=self.session, inheritable=inheritable
        )

    def compare_object_handles(self, first: int, second: int) -> bool:
        self.calls.append(("compare_object_handles", first, second))
        return self.kernel.entry(first, self.pid).name == self.kernel.entry(second, self.pid).name

    def set_handle_information(self, handle: int, mask: int, flags: int) -> None:
        self.calls.append(("set_handle_information", handle, mask, flags))
        if mask & HANDLE_FLAG_INHERIT:
            self.kernel.entry(handle, self.pid).inheritable = bool(flags & HANDLE_FLAG_INHERIT)

    def get_last_error(self) -> int:
        return self.last_error

    # -- identity ------------------------------------------------------------------------------- #

    def current_user_sid(self) -> str:
        return self.sid

    def current_user_security_attributes(self) -> Any:
        self.calls.append(("current_user_security_attributes", self.sid))
        return FakeSecurityAttributes(self.sid)

    # -- assertion helpers ------------------------------------------------------------------------ #

    def names(self, method: str) -> list[str]:
        return [call[1] for call in self.calls if call[0] == method]

    def kinds(self) -> list[str]:
        return [call[0] for call in self.calls]


# --------------------------------------------------------------------------------------------- #
# Fixtures and process modeling
# --------------------------------------------------------------------------------------------- #

@pytest.fixture()
def kernel() -> FakeKernel:
    return FakeKernel()


@pytest.fixture()
def deployment(tmp_path: Path) -> Path:
    return tmp_path / "runtime" / KEY


def adapter(deployment: Path, kernel: FakeKernel, *, sid: str = SID_A, session: int = 1,
            key: str = KEY) -> tuple[WindowsLeaseAdapter, FakeWin32]:
    """One process running the Windows lease adapter."""
    surface = FakeWin32(kernel, sid=sid, session=session)
    return WindowsLeaseAdapter(deployment, deployment_key=key, win32=surface), surface


def event_name(key: str = KEY, sid: str = SID_A) -> str:
    return windows_lease_event_name(key, sid_hash_value=sid_hash(sid))


def spawn_approved_child(deployment: Path, kernel: FakeKernel, handoff, parent: FakeWin32, *,
                         session: int = 1, sid: str = SID_A,
                         extra_env: dict[str, str] | None = None
                         ) -> tuple[WindowsLeaseAdapter, FakeWin32]:
    """Model the real launch: ``CreateProcess`` with the STARTUPINFOEX allowlist, then the child
    binding the handle its environment names."""
    surface = FakeWin32(kernel, sid=sid, session=session)
    allowlist = handoff.popen_kwargs["startupinfo"].lpAttributeList["handle_list"]
    inherited = kernel.create_process(
        parent_pid=parent.pid, child_pid=surface.pid, session=session, handle_list=allowlist
    )
    assert inherited == allowlist, "the child did not receive exactly the allowlisted handles"
    env = dict(handoff.env)
    env.update(extra_env or {})
    child = WindowsLeaseAdapter.from_inherited(
        deployment, deployment_key=KEY, env=env, win32=surface
    )
    return child, surface


# --------------------------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------------------------- #

def test_the_lease_is_one_global_event_with_a_current_user_only_acl(deployment: Path, kernel) -> None:
    """Plan section 2.2: ``Global\\lm3-lease-<sid hash>-<key>``, current-user DACL, EVENT_ALL_ACCESS."""
    lease, win32 = adapter(deployment, kernel)
    lease.acquire()
    try:
        assert lease.event_name == event_name()
        created = [call for call in win32.calls if call[0] == "create_event"]
        assert len(created) == 1
        _, name, inheritable, security = created[0]
        assert name.startswith(GLOBAL_PREFIX)
        assert inheritable is False
        assert isinstance(security, FakeSecurityAttributes)
        assert security.sid == SID_A
        # EVENT_ALL_ACCESS and not less: CreateEventW opening an EXISTING named event requests
        # exactly that, so a stingier DACL would lock our own next process out of its own lease.
        assert security.access == EVENT_ALL_ACCESS
    finally:
        lease.release()


def test_activity_lock_is_a_passive_artifact_on_windows(deployment: Path, kernel) -> None:
    """It exists so the registry directory has the same shape everywhere -- and is never consulted.

    Half the proof is structural: the surface exposes no locking call at all, so an adapter that
    tried to lock the file could not even be written against it. What is left to assert is that the
    file is created (shape) and that nothing is ever written into or read out of it.
    """
    lease, win32 = adapter(deployment, kernel)
    lease.acquire()
    try:
        assert (deployment / ACTIVITY_LOCK_FILENAME).is_file()
        assert lease.lock_path.read_bytes() == b""
        assert "lock" not in "".join(call[0] for call in win32.calls)
    finally:
        lease.release()


def test_a_second_root_is_refused_and_the_loser_closes_its_handle_first(deployment: Path, kernel) -> None:
    """Gate 31. ``CreateEventW`` hands the LOSER a valid handle to the existing object; a loser that
    kept it would pin the deployment occupied after the winner exited -- and a direct Python caller
    that catches ``RuntimeBusyError`` and carries on is not hypothetical."""
    winner, _ = adapter(deployment, kernel)
    winner.acquire()
    loser, loser_win32 = adapter(deployment, kernel)
    with pytest.raises(RuntimeBusyError) as excinfo:
        loser.acquire()
    assert excinfo.value.deployment_key == KEY
    assert loser.handle is None

    # The close happened BEFORE the raise, on the handle CreateEventW had just returned.
    assert loser_win32.kinds()[-2:] == ["create_event", "close_handle"]
    assert len(kernel.handles_for(event_name())) == 1, "the loser kept a reference to the lease"

    # And the deployment really is free once the winner goes, loser process still alive or not.
    winner.release()
    assert kernel.handles_for(event_name()) == []
    third, _ = adapter(deployment, kernel)
    third.acquire()
    third.release()


def test_a_failure_to_create_the_global_event_is_a_startup_error(deployment: Path, kernel) -> None:
    """Gate 29: no silent fall back to ``Local``, which would weaken the invariant exactly when the
    environment is unusual."""
    lease, win32 = adapter(deployment, kernel)
    win32.create_error = OSError(5, "access denied")
    with pytest.raises(LeaseError, match="Local"):
        lease.acquire()
    assert lease.handle is None
    assert all(name.startswith(GLOBAL_PREFIX) for name in win32.names("create_event"))
    assert not any(name.startswith(LOCAL_PREFIX) for name in kernel.seen_names)


def test_two_users_on_one_machine_do_not_block_each_other(deployment: Path, kernel) -> None:
    """Gate 30: the SID hash in the name is what keeps the machine-wide namespace per-user."""
    first, _ = adapter(deployment, kernel, sid=SID_A)
    second, _ = adapter(deployment, kernel, sid=SID_B)
    first.acquire()
    second.acquire()          # must NOT raise
    try:
        assert first.event_name != second.event_name
        assert sid_hash(SID_A) in first.event_name
        assert sid_hash(SID_B) in second.event_name
    finally:
        first.release()
        second.release()


def test_two_deployments_of_one_user_do_not_contend(deployment: Path, kernel) -> None:
    """Gate 40's Windows half: the canonical deployment key is the other half of the event name."""
    first, _ = adapter(deployment, kernel, key="default")
    second, _ = adapter(deployment, kernel, key="greening-batch-1f0a2b3c")
    first.acquire()
    second.acquire()
    first.release()
    second.release()


def test_probe_is_observation_and_leaks_nothing(deployment: Path, kernel) -> None:
    lease, _ = adapter(deployment, kernel)
    observer, _ = adapter(deployment, kernel)
    assert observer.probe_occupied() is False
    lease.acquire()
    try:
        assert observer.probe_occupied() is True
        assert len(kernel.handles_for(event_name())) == 1, "the probe kept a handle open"
    finally:
        lease.release()
    assert observer.probe_occupied() is False


def test_double_acquire_is_refused(deployment: Path, kernel) -> None:
    lease, _ = adapter(deployment, kernel)
    lease.acquire()
    try:
        with pytest.raises(LeaseError):
            lease.acquire()
    finally:
        lease.release()


# --------------------------------------------------------------------------------------------- #
# Subactivity launch -- the duplication dance
# --------------------------------------------------------------------------------------------- #

def test_the_owner_handle_is_never_inheritable(deployment: Path, kernel) -> None:
    """Revision 12's K1, "the whole ballgame" (gate 33).

    If the root held an INHERITABLE handle, every ordinary executor worker it spawned with
    ``bInheritHandles=TRUE`` would inherit the lease -- and a disarm inside an approved CHILD does
    nothing about that, because the workers are spawned by the ROOT from the root's own handle.
    """
    lease, win32 = adapter(deployment, kernel)
    lease.acquire()
    try:
        assert kernel.entry(lease.handle, win32.pid).inheritable is False
        # What an ordinary worker spawned at this instant would actually receive:
        worker_pid = kernel.new_pid()
        assert kernel.create_process(parent_pid=win32.pid, child_pid=worker_pid, session=1) == []
    finally:
        lease.release()


def test_child_handoff_duplicates_inheritably_and_closes_the_duplicate(deployment: Path, kernel) -> None:
    lease, win32 = adapter(deployment, kernel)
    lease.acquire()
    try:
        with lease.child_handoff("cap-value") as handoff:
            duplicate = handoff.lease_handle
            assert duplicate is not None and duplicate != lease.handle
            assert ("duplicate_handle", lease.handle, True) in win32.calls
            assert kernel.entry(duplicate, win32.pid).inheritable is True
            assert kernel.entry(lease.handle, win32.pid).inheritable is False, "the OWNER was touched"
            assert handoff.env == {
                ENV_LEASE_EVENT_HANDLE: str(duplicate), ENV_LEASE_CAPABILITY: "cap-value"}
            assert handoff.lease_fd is None
            # The STARTUPINFOEX allowlist carries the duplicate and nothing else (step 3).
            allowlist = handoff.popen_kwargs["startupinfo"].lpAttributeList["handle_list"]
            assert allowlist == [duplicate]
            assert handoff.popen_kwargs["close_fds"] is True
            # A worker spawned INSIDE this window would inherit it -- which is why step 6 serializes.
            assert [entry.value for entry in kernel.inheritable_handles(win32.pid)] == [duplicate]
        # Step 4: closed on the way out, so nothing inheritable survives the launch.
        assert (win32.pid, duplicate) not in kernel.refs
        assert kernel.inheritable_handles() == []
    finally:
        lease.release()


def test_the_duplicate_is_closed_even_when_the_spawn_fails(deployment: Path, kernel) -> None:
    """Gate 34: "on every path, including every failure path" -- a failed spawn otherwise leaks a
    lease reference that nothing will ever release, and the deployment never becomes free again."""
    lease, win32 = adapter(deployment, kernel)
    lease.acquire()
    duplicate: int | None = None
    with pytest.raises(RuntimeError, match="CreateProcess failed"):
        with lease.child_handoff("cap") as handoff:
            duplicate = handoff.lease_handle
            raise RuntimeError("CreateProcess failed")
    assert duplicate is not None
    assert (win32.pid, duplicate) not in kernel.refs
    assert kernel.inheritable_handles() == []
    # The lease itself is untouched: a failed subactivity launch does not end the root activity.
    assert (win32.pid, lease.handle) in kernel.refs
    lease.release()


def test_closing_the_duplicate_twice_is_harmless(deployment: Path, kernel) -> None:
    """The launcher closes as soon as ``CreateProcess`` returns; the context manager then closes
    again on the way out. A non-idempotent close would turn correct launcher code into an
    invalid-handle error."""
    lease, win32 = adapter(deployment, kernel)
    lease.acquire()
    with lease.child_handoff("cap") as handoff:
        duplicate = handoff.lease_handle
        handoff.close()
        handoff.close()
    assert win32.calls.count(("close_handle", duplicate)) == 1
    lease.release()


def test_the_spawn_window_is_serialized_against_every_other_spawn(deployment: Path, kernel) -> None:
    """Plan section 2.2 step 6: between the duplication and the close an inheritable lease handle
    exists in the root, so a concurrent ``CreateProcess`` from another thread could inherit it."""
    lease, _ = adapter(deployment, kernel)
    lease.acquire()
    entered = threading.Event()
    got_lock = threading.Event()

    def worker_spawn() -> None:
        entered.set()
        with process_creation_lock():
            got_lock.set()

    try:
        with lease.child_handoff("cap"):
            thread = threading.Thread(target=worker_spawn, daemon=True)
            thread.start()
            entered.wait(timeout=5)
            assert not got_lock.wait(timeout=0.2), "a worker spawn raced the inheritable duplicate"
        thread.join(timeout=5)
        assert got_lock.is_set()
    finally:
        lease.release()


def test_child_handoff_requires_the_lease(deployment: Path, kernel) -> None:
    lease, _ = adapter(deployment, kernel)
    with pytest.raises(LeaseNotHeldError):
        with lease.child_handoff("cap"):
            pass


# --------------------------------------------------------------------------------------------- #
# Child-side validation and disarming
# --------------------------------------------------------------------------------------------- #

def test_the_child_validates_its_inherited_handle(deployment: Path, kernel) -> None:
    lease, win32 = adapter(deployment, kernel)
    lease.acquire()
    with lease.child_handoff("cap") as handoff:
        child, child_win32 = spawn_approved_child(deployment, kernel, handoff, win32)
        assert child.handle == handoff.lease_handle
        assert child.inherited is True
        before = len(kernel.refs)
        child.validate_inherited()
        # OpenEventW the canonical name, CompareObjectHandles, and close the comparison handle --
        # a leaked one would pin the lease object alive.
        assert child_win32.kinds() == ["open_event", "compare_object_handles", "close_handle"]
        assert child_win32.names("open_event") == [event_name()]
        assert len(kernel.refs) == before
    lease.release()
    # The child alone keeps the deployment occupied now.
    assert len(kernel.handles_for(event_name())) == 1
    child.release()


def test_a_handle_for_a_different_object_is_rejected(deployment: Path, kernel) -> None:
    """Gate 9's Windows half: ``CompareObjectHandles`` says they are different kernel objects."""
    lease, _ = adapter(deployment, kernel)
    lease.acquire()
    child_win32 = FakeWin32(kernel)
    decoy = child_win32.create_event(
        GLOBAL_PREFIX + "something-else", security_attributes=None, inheritable=True)
    child = WindowsLeaseAdapter.from_inherited(
        deployment, deployment_key=KEY,
        env={ENV_LEASE_EVENT_HANDLE: str(decoy), ENV_LEASE_CAPABILITY: "cap"}, win32=child_win32,
    )
    before = len(kernel.refs)
    with pytest.raises(LeaseInheritanceError, match="different kernel objects"):
        child.validate_inherited()
    assert len(kernel.refs) == before, "the comparison handle leaked on the rejection path"
    child_win32.close_handle(decoy)
    lease.release()


def test_validation_fails_when_the_canonical_event_cannot_be_opened(deployment: Path, kernel) -> None:
    """No openable lease event, no validation: the inherited handle is compared against nothing."""
    child_win32 = FakeWin32(kernel)
    orphan = child_win32.create_event(event_name(), security_attributes=None, inheritable=True)
    child = WindowsLeaseAdapter.from_inherited(
        deployment, deployment_key=KEY, env={ENV_LEASE_EVENT_HANDLE: str(orphan)}, win32=child_win32,
    )
    child_win32.open_error = OSError(5, "access denied")
    with pytest.raises(LeaseInheritanceError, match="could not be opened"):
        child.validate_inherited()
    child_win32.open_error = None
    child_win32.close_handle(orphan)


@pytest.mark.parametrize("value", [None, "", "  ", "not-a-handle", "0", "-1"])
def test_from_inherited_rejects_a_missing_or_malformed_handle(deployment: Path, kernel, value) -> None:
    env = {} if value is None else {ENV_LEASE_EVENT_HANDLE: value}
    with pytest.raises(LeaseInheritanceError):
        WindowsLeaseAdapter.from_inherited(
            deployment, deployment_key=KEY, env=env, win32=FakeWin32(kernel))


def test_the_child_disarms_inheritance_and_clears_the_environment(deployment: Path, kernel) -> None:
    """Gate 32: after validating, the child stops the handle propagating and clears all three
    variables, BEFORE it spawns any executor worker."""
    lease, win32 = adapter(deployment, kernel)
    lease.acquire()
    with lease.child_handoff("cap") as handoff:
        child, child_win32 = spawn_approved_child(
            deployment, kernel, handoff, win32,
            # A POSIX leftover must be cleared too: a stale variable is a false claim of inheritance.
            extra_env={ENV_LEASE_FD: "9", "UNRELATED": "kept"},
        )
        child.validate_inherited()
        env = dict(handoff.env) | {ENV_LEASE_FD: "9", "UNRELATED": "kept"}
        child_lease = RuntimeLease(deployment_dir=deployment, deployment_key=KEY, adapter=child)
        child_lease.disarm_and_clear(env)
        assert ("set_handle_information", child.handle, HANDLE_FLAG_INHERIT, 0) in child_win32.calls
        assert kernel.entry(child.handle, child_win32.pid).inheritable is False
        assert env == {"UNRELATED": "kept"}
        # A worker spawned by the CHILD from here inherits nothing.
        worker_pid = kernel.new_pid()
        assert kernel.create_process(
            parent_pid=child_win32.pid, child_pid=worker_pid, session=1) == []
    lease.release()
    child.release()


# --------------------------------------------------------------------------------------------- #
# Lifetime -- the reason Windows uses an event at all
# --------------------------------------------------------------------------------------------- #

def test_killing_the_root_leaves_the_deployment_occupied_until_the_child_exits(
    deployment: Path, kernel
) -> None:
    """Gate 27, against the Windows adapter itself.

    ``LockFileEx`` could not do this: Windows file locks belong to the process and the OS releases
    them when it terminates, so killing the root would free the lease while the subactivity ran on.
    The event object survives because the child still holds a handle to it.
    """
    root, root_win32 = adapter(deployment, kernel)
    root.acquire()
    with root.child_handoff("cap") as handoff:
        child, _ = spawn_approved_child(deployment, kernel, handoff, root_win32)
        child.validate_inherited()

    root.release()                      # the root is hard-killed: the OS closes its handles
    assert kernel.handles_for(event_name()), "the lease died with its creator"

    contender, _ = adapter(deployment, kernel)
    with pytest.raises(RuntimeBusyError):
        contender.acquire()

    # Only when the LAST handle in the activity tree closes does the deployment become free.
    child.release()
    assert kernel.handles_for(event_name()) == []
    successor, _ = adapter(deployment, kernel)
    successor.acquire()
    successor.release()


def test_that_exclusion_holds_across_logon_sessions(deployment: Path, kernel) -> None:
    """Gate 28. The contender is in a SECOND logon session of the same user.

    A ``Local`` event would be scoped to one session and would let this contender straight in --
    which is why the name is ``Global`` and why there is no fallback.
    """
    root, root_win32 = adapter(deployment, kernel, session=1)
    root.acquire()
    with root.child_handoff("cap") as handoff:
        child, _ = spawn_approved_child(deployment, kernel, handoff, root_win32, session=1)
        child.validate_inherited()
    root.release()                                   # the root is killed; the child lives on

    other_session, other_win32 = adapter(deployment, kernel, session=2)
    with pytest.raises(RuntimeBusyError):
        other_session.acquire()
    assert other_win32.names("create_event") == [event_name()]
    assert event_name().startswith(GLOBAL_PREFIX)

    # Contrast, to show the model really does have a session boundary to cross: a Local\ name in
    # session 2 does NOT see session 1's object. That is the hole the plan refuses to open.
    session_one = FakeWin32(kernel, session=1)
    session_one.create_event(LOCAL_PREFIX + "lm3-lease-demo", security_attributes=None)
    assert kernel.exists(LOCAL_PREFIX + "lm3-lease-demo", 1) is True
    assert kernel.exists(LOCAL_PREFIX + "lm3-lease-demo", 2) is False

    child.release()


def test_a_worker_spawned_by_the_root_holds_nothing_after_the_tree_exits(
    deployment: Path, kernel
) -> None:
    """Gate 33's final clause: with an ordinary worker still alive after the root and its approved
    child have exited, the event disappears and another root acquires successfully."""
    root, root_win32 = adapter(deployment, kernel)
    root.acquire()
    worker_pid = kernel.new_pid()
    assert kernel.create_process(parent_pid=root_win32.pid, child_pid=worker_pid, session=1) == []
    with root.child_handoff("cap") as handoff:
        child, child_win32 = spawn_approved_child(deployment, kernel, handoff, root_win32)
        child.validate_inherited()
        child_lease = RuntimeLease(deployment_dir=deployment, deployment_key=KEY, adapter=child)
        child_lease.disarm_and_clear({})
    # A worker spawned at ANY point outside the handoff window inherits nothing either.
    late_worker = kernel.new_pid()
    assert kernel.create_process(parent_pid=root_win32.pid, child_pid=late_worker, session=1) == []

    root.release()
    child.release()
    assert kernel.handles_for(event_name()) == [], "a worker is still holding the lease event"
    successor, _ = adapter(deployment, kernel)
    successor.acquire()
    successor.release()


def test_runtime_lease_drives_the_windows_adapter_end_to_end(deployment: Path, kernel) -> None:
    """The object the rest of LM3 uses, with the Windows primitive underneath it."""
    win_adapter, _ = adapter(deployment, kernel)
    with RuntimeLease(deployment_dir=deployment, deployment_key=KEY, adapter=win_adapter) as lease:
        assert lease.held is True
        assert lease.probe_occupied() is True
        loser_adapter, _ = adapter(deployment, kernel)
        loser = RuntimeLease(deployment_dir=deployment, deployment_key=KEY, adapter=loser_adapter)
        with pytest.raises(RuntimeBusyError) as excinfo:
            loser.acquire()
        assert excinfo.value.deployment_key == KEY
    assert kernel.handles_for(event_name()) == []


# --------------------------------------------------------------------------------------------- #
# The pieces only a real Windows host can answer
# --------------------------------------------------------------------------------------------- #

def test_the_handle_allowlist_has_the_same_shape_on_both_platforms() -> None:
    """The Linux stand-in exposes ``lpAttributeList`` exactly as ``subprocess.STARTUPINFO`` does, so
    the assertions above are the assertions that hold on Windows."""
    kwargs = _win32.handle_list_popen_kwargs([0x1F4], platform_name="linux")
    assert kwargs["startupinfo"].lpAttributeList == {"handle_list": [0x1F4]}
    # close_fds is not decoration: subprocess REQUIRES it whenever handle_list is non-empty.
    assert kwargs["close_fds"] is True


@pytest.mark.skipif(sys.platform != "win32", reason="a real STARTUPINFO only exists on Windows")
def test_the_real_allowlist_is_a_startupinfo() -> None:                # pragma: no cover - Windows CI
    kwargs = _win32.handle_list_popen_kwargs([])
    assert isinstance(kwargs["startupinfo"], subprocess.STARTUPINFO)


@pytest.mark.skipif(sys.platform != "win32", reason="requires the real Win32 object manager")
def test_the_real_surface_acquires_and_releases(tmp_path: Path) -> None:  # pragma: no cover - Win CI
    """The one thing the fake cannot prove: that the real ``CreateEventW`` behaves like the model."""
    from leafmachine3.core.runtime.lease import real_win32_surface

    surface = real_win32_surface()
    assert surface.current_user_sid().startswith("S-1-")
    deployment_dir = tmp_path / "runtime" / KEY
    first = WindowsLeaseAdapter(deployment_dir, deployment_key=KEY, win32=surface)
    first.acquire()
    try:
        second = WindowsLeaseAdapter(deployment_dir, deployment_key=KEY, win32=surface)
        with pytest.raises(RuntimeBusyError):
            second.acquire()
        assert first.probe_occupied() is True
    finally:
        first.release()
    assert WindowsLeaseAdapter(
        deployment_dir, deployment_key=KEY, win32=surface).probe_occupied() is False
