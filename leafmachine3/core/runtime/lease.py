"""The deployment activity lease -- both platform adapters and the inheritance plumbing.

Plan sections 2.2 (the mechanism), 3.1 (where the lock file lives) and 3.3 (when it is acquired,
and what a reader may conclude when it is free). Invariants 1-3 are the acceptance test:

1. at most one ROOT activity (``pipeline`` / ``hardware_setup``) per deployment;
2. an approved SUBACTIVITY runs under its parent's lease by INHERITING the lease reference, so the
   deployment stays occupied for as long as the root *or any live subactivity* holds it -- killing
   the parent does not free the lease;
3. ordinary executor workers never hold that reference.

Two mechanisms, sharing nothing
-------------------------------
POSIX uses ``flock`` on ``<deployment runtime dir>/activity.lock``. ``flock`` locks belong to the
OPEN FILE DESCRIPTION, not to the descriptor and not to the process, so a child that inherits the
descriptor shares the lock, and the lock survives until the LAST descriptor referring to that
description closes. That is invariant 2, for free, from the kernel.

Windows CANNOT use that: ``LockFileEx`` locks belong to the PROCESS ("the child process is not
granted access to the locked region"), and the OS unlocks them when the holder terminates -- so
killing the root would free the lease while its subactivity ran on. The Windows lease is therefore
a single named kernel event, ``Global\\lm3-lease-<sid hash>-<deployment key>``. Nobody acquires it;
its EXISTENCE is the lease, ``CreateEventW`` reporting ``ERROR_ALREADY_EXISTS`` is the atomic
test-and-create, and the object lives until its last handle closes -- the close analogue of the
POSIX open-file-description lifetime. ``activity.lock`` still exists on Windows so the registry
directory has the same shape everywhere, but it is a PASSIVE artifact: never locked, never
consulted. A second mechanism is a second source of truth waiting to be believed.

Because the two share no mechanism, every lifetime property is tested against BOTH adapters (gate
27 says the Windows behavior must be "verified against the Windows adapter itself, not inferred
from the POSIX result"). Every Win32 call goes through the injected
:class:`~leafmachine3.core.runtime._types.Win32Surface`, so the Windows ORDERING, handle lifetime,
close-on-every-failure-path and rejection logic are unit-tested on this Linux host against a fake
that records the call sequence.

What this module deliberately does NOT do
-----------------------------------------
It never signals a process. A reader that finds ``active.json`` while the lease is acquirable
classifies the record as abandoned under the cleanup lock (plan section 3.3, gate 51) -- it does
not kill the recorded PID, because a PID in a JSON file is not proof of anything (invariant 12).
There is no ``os.kill`` and no ``signal`` import in this file, and a test asserts that.
"""
from __future__ import annotations

import errno
import logging
import os
import stat
import sys
import threading
import time
from contextlib import AbstractContextManager, contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, MutableMapping

from .. import paths
from . import _win32
from ._types import (
    ACTIVITY_LOCK_FILENAME,
    CHILD_ACTIVITIES,
    ENV_LEASE_CAPABILITY,
    ENV_LEASE_EVENT_HANDLE,
    ENV_LEASE_FD,
    ERROR_ALREADY_EXISTS,
    EVENT_ALL_ACCESS,
    HANDLE_FLAG_INHERIT,
    LEASE_ENV_VARS,
    SID_HASH_LENGTH,
    WINDOWS_LEASE_EVENT_TEMPLATE,
    Activity,
    ChildHandoff,
    LeaseAdapter,
    LeaseError,
    LeaseInheritanceError,
    LeaseNotHeldError,
    RecordClassification,
    RuntimeBusyError,
)

try:
    # POSIX only, and imported at module scope on purpose: the POSIX adapter needs it on every
    # acquire, and a per-call import would hide a broken interpreter behind a lock failure. On
    # Windows it simply does not exist -- and must not, because this module has to import cleanly
    # there so the event adapter can be used.
    import fcntl
except ImportError:                                                   # pragma: no cover - Windows
    fcntl = None                                                      # type: ignore[assignment]

log = logging.getLogger("leafmachine3.runtime.lease")

__all__ = [
    "PosixLeaseAdapter",
    "RuntimeLease",
    "WindowsLeaseAdapter",
    "acquire_root_lease",
    "cleanup_lease",
    "clear_lease_env",
    "inherit_lease",
    "lease_adapter",
    "probe_deployment_occupied",
    "process_creation_lock",
    "real_win32_surface",
    "sid_hash",
    "windows_lease_event_name",
]

#: ``0o700`` on the deployment directory, ``0o600`` on the lock file: the registry describes what
#: this user is running and is nobody else's business (plan section 3.1).
_DIR_MODE = 0o700
_LOCK_MODE = 0o600

# How long a contender keeps re-trying a refused ``flock`` before it believes the refusal, and how
# long it pauses between attempts. A GENUINE holder keeps the lease for its whole root activity, so
# a refusal that clears inside this budget was never a holder at all -- it was an observer's
# microsecond shared test-and-release in ``probe_occupied``, which is the only way Linux lets us ask
# the occupancy question at all (there is no query-only primitive for ``flock``: ``F_OFD_GETLK``
# reports on POSIX record locks, a separate lock space that is blind to ``flock``, and plan section
# 2.2 fixes ``flock`` as the mechanism). The retry is therefore what keeps invariant 13 ("the
# server observes") true from the CONTENDER's point of view: no observer can refuse a legitimate
# root with a spurious RuntimeBusyError / exit 75. Bounded so a real busy start is still an
# immediate 409 and not a late failure (gate 36). Module level so a test can shrink them.
_ACQUIRE_CONTENTION_BUDGET_S = 0.25
_ACQUIRE_RETRY_SLEEP_S = 0.001


# --------------------------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------------------------- #

def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _is_windows(platform_name: str | None = None) -> bool:
    return (platform_name or sys.platform).startswith("win")


def sid_hash(sid: str) -> str:
    """``sha256`` of a canonical ``ConvertSidToStringSidW`` SID string, truncated (plan section 2.2).

    The Windows event name is machine-wide, so it embeds a per-user discriminator: two users on one
    machine must get distinct leases and must not block each other (gate 30). Same bounded-hash
    convention as the deployment key (section 2.1) and the machine key (section 3.1).
    """
    return sha256(sid.encode("utf-8")).hexdigest()[:SID_HASH_LENGTH]


def windows_lease_event_name(deployment_key: str, *, sid_hash_value: str) -> str:
    """The one canonical lease-event name. ``Global\\``, never ``Local\\``.

    ``SeCreateGlobalPrivilege`` gates only file-mapping and symbolic-link objects, so an event needs
    no privilege; a ``Local\\`` event is scoped to one logon session and would leave the
    dead-root-with-live-child window unguarded against a contender in another session of the same
    user (gates 28, 29). If the global event cannot be created, that is a startup error.
    """
    return WINDOWS_LEASE_EVENT_TEMPLATE.format(sid_hash=sid_hash_value, deployment_key=deployment_key)


def real_win32_surface() -> Any:
    """Build the real ``ctypes`` Win32 surface, at CALL time, behind a platform check.

    Raises :class:`LeaseError` off Windows rather than returning a stub: a stub would let the
    Windows lease "succeed" where there is no lease at all.
    """
    return _win32.build_real_surface()


def clear_lease_env(env: MutableMapping[str, str] | None = None) -> None:
    """Remove every lease variable from ``env`` (defaults to ``os.environ``).

    Unconditional and platform-independent on purpose: a POSIX child clears
    ``LM3_LEASE_EVENT_HANDLE`` too, and a Windows child clears ``LM3_LEASE_FD``. The variables are
    inherited by everything the process spawns afterwards, and a stale one is a false claim of
    inheritance for the next reader to trip over (plan section 2.2 step 4, gate 32).
    """
    target = os.environ if env is None else env
    for name in LEASE_ENV_VARS:
        target.pop(name, None)


# The process-creation mutex of plan section 2.2 step 6. Between duplicating the lease handle
# inheritably and closing that duplicate, an inheritable lease handle exists in the root -- and a
# CONCURRENT CreateProcess from another thread (an executor worker starting at that instant) would
# inherit it. Subactivity launches and worker launches take this one lock so the window is never
# open during an unrelated spawn. It is module level, and re-entrant so a launcher that already
# holds it can nest.
_PROCESS_CREATION_LOCK = threading.RLock()


@contextmanager
def process_creation_lock() -> Iterator[None]:
    """Serialize the spawn window (plan section 2.2, step 6). Re-entrant within one thread.

    POSIX does not strictly need it -- ``pass_fds`` is a per-spawn opt-in, so an unrelated worker
    cannot pick anything up -- but the launcher takes it on both platforms so there is one spawn
    discipline rather than two.
    """
    with _PROCESS_CREATION_LOCK:
        yield None


def _ensure_deployment_dir(deployment_dir: Path) -> None:
    """Create the deployment runtime directory, user-only. Registry-owned, never execution-owned.

    Invariant 14 forbids a BUSY loser from creating execution-owned state (a run directory, a
    project DB, a CUDA context). The registry directory is none of those -- it is where the answer
    "who holds the lease" lives, and it has to exist before the question can be asked.
    """
    try:
        deployment_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            os.chmod(deployment_dir, _DIR_MODE)
    except OSError as exc:
        raise LeaseError(f"could not prepare the deployment runtime directory {deployment_dir}: {exc}") from exc


# --------------------------------------------------------------------------------------------- #
# POSIX adapter -- flock on the open file description
# --------------------------------------------------------------------------------------------- #

class PosixLeaseAdapter:
    """``flock(LOCK_EX|LOCK_NB)`` on ``<deployment runtime dir>/activity.lock``.

    The descriptor is the lease reference. It is handed to approved subactivities with
    ``Popen(..., pass_fds=(fd,))`` and to nothing else.
    """

    def __init__(self, deployment_dir: Path, *, deployment_key: str) -> None:
        self.deployment_dir = Path(deployment_dir)
        self.lock_path = self.deployment_dir / ACTIVITY_LOCK_FILENAME
        self.deployment_key = deployment_key
        self.fd: int | None = None
        #: True when the descriptor came from a parent rather than from our own ``acquire``.
        self.inherited = False

    # -- acquisition --------------------------------------------------------------------------- #

    def acquire(self) -> None:
        if self.fd is not None:
            raise LeaseError(
                f"this lease adapter already holds {self.lock_path}; a second acquire would hide a "
                f"double-start bug rather than fix one"
            )
        if fcntl is None:                                             # pragma: no cover - Windows
            raise LeaseError(
                "the POSIX lease adapter needs flock, which this platform has no fcntl for. "
                "Windows uses the named-event adapter; there is no third option."
            )
        _ensure_deployment_dir(self.deployment_dir)
        # os.open returns a NON-inheritable descriptor (PEP 446), which is the posture invariant 3
        # needs: nothing this process spawns sees the lock unless it is named in pass_fds.
        try:
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, _LOCK_MODE)
        except OSError as exc:
            raise LeaseError(f"could not open the activity lock {self.lock_path}: {exc}") from exc
        # Retry a refusal on THIS descriptor for a bounded moment before believing it. See
        # _ACQUIRE_CONTENTION_BUDGET_S: a live root holds for its whole activity, an observer's
        # probe holds for microseconds, so anything that clears inside the budget was contention
        # with an observer and refusing it would break invariant 13 for the contender.
        deadline = time.monotonic() + _ACQUIRE_CONTENTION_BUDGET_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                    os.close(fd)
                    raise LeaseError(f"could not lock {self.lock_path}: {exc}") from exc
                if time.monotonic() >= deadline:
                    os.close(fd)
                    # Held by a live root, or by a subactivity whose root has already died. Both are
                    # "occupied"; RuntimeLease decorates this with the winner's sanitized record.
                    raise RuntimeBusyError(self.deployment_key) from None
                time.sleep(_ACQUIRE_RETRY_SLEEP_S)
        self.fd = fd

    def release(self) -> None:
        """Drop THIS process's reference. Idempotent.

        Deliberately a plain ``close`` and never ``flock(LOCK_UN)``: the unlock would apply to the
        shared open file description, so a subactivity calling ``release`` would free the ROOT's
        lease and let a second root in while both were still running. Closing merely drops one
        reference, and the kernel releases the lock when the last one goes.

        It also never unlinks ``activity.lock``. The file's inode is the identity a child validates
        against; a delete-and-recreate cycle would let two processes lock two different inodes at
        the same path and both believe they hold the deployment.
        """
        fd, self.fd = self.fd, None
        if fd is None:
            return
        try:
            os.close(fd)
        except OSError as exc:                                        # pragma: no cover - defensive
            log.debug("closing the activity-lock descriptor failed: %s", exc)

    def is_held(self) -> bool:
        return self.fd is not None

    def probe_occupied(self) -> bool:
        """Occupancy question for observers such as ``GET /v1/runtime``. Retains nothing.

        Opens its own private descriptor, tries a SHARED lock on it, and lets go immediately.
        ``flock`` treats two descriptions of one file independently even inside a single process,
        so this answers truthfully about our own held lease too.

        Shared, not exclusive, and that is the whole point. Linux offers no query-only primitive for
        ``flock`` -- ``F_OFD_GETLK`` reports on POSIX record locks, a separate lock space that is
        blind to ``flock`` -- and plan section 2.2 fixes ``flock`` as the mechanism, so a momentary
        test-and-release is the only honest question available. An EXCLUSIVE test would make every
        observer briefly exclude every other observer, so two concurrent readers (the section 2.6
        GUI poll plus any other ``read_runtime`` caller) would each invent an occupied deployment on
        an idle machine, disable Start, and turn an abandoned record into a live one (section 3.3).
        A shared test is refused only by the holder's ``LOCK_EX``, so ``True`` here means a real
        lease. The one residual overlap -- this shared lock momentarily refusing a contender's
        ``LOCK_EX`` -- is absorbed by the bounded retry in :meth:`acquire`, never by stalling the
        observer, which polls far more often than anyone acquires.
        """
        if fcntl is None or not self.lock_path.exists():
            return False        # no lock file, therefore nobody has ever held this deployment
        try:
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CLOEXEC)
        except OSError:
            return False
        # One finally for the whole body: the OCCUPIED branch is taken on every poll while a run is
        # live, so an early return that skipped the close would exhaust the observer's descriptors
        # (section 2.6 polls this on each status frame) long before the run ended.
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError:
                return True
            # We just took a lock we do not want. Release this probe description explicitly -- it is
            # a description of our own, so LOCK_UN here cannot touch anybody else's lease.
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)

    # -- handing the reference to one approved subactivity ------------------------------------- #

    @contextmanager
    def _handoff(self, capability: str) -> Iterator[ChildHandoff]:
        """Yield the launch keywords for ONE approved subactivity.

        ``popen_kwargs`` describes the lease and nothing else. A launcher that also passes the
        section 2.4 status pipe MERGES its descriptor into this ``pass_fds`` tuple rather than
        replacing it -- passing two separate ``pass_fds`` would silently drop one of them.
        """
        if self.fd is None:
            raise LeaseNotHeldError(
                "cannot hand a lease reference to a subactivity without holding the lease"
            )
        fd = self.fd
        with process_creation_lock():
            # ``pass_fds`` is a per-spawn opt-in: subprocess keeps exactly these descriptors open in
            # the child (at the same numbers) and closes everything else, so an ordinary worker
            # spawned without it inherits nothing. Nothing here makes the descriptor globally
            # inheritable -- that is what gate 7 proves.
            yield ChildHandoff(
                capability=capability,
                env={ENV_LEASE_FD: str(fd), ENV_LEASE_CAPABILITY: capability},
                popen_kwargs={"pass_fds": (fd,)},
                close=_noop_close,
                lease_fd=fd,
            )

    def child_handoff(self, capability: str) -> AbstractContextManager[ChildHandoff]:
        return self._handoff(capability)

    # -- child side ---------------------------------------------------------------------------- #

    def validate_inherited(self) -> None:
        """Prove the inherited descriptor really is this deployment's lease (gate 9).

        Two checks, exactly as plan section 2.2 step 1 specifies:

        * ``fstat`` it and require the same ``(st_dev, st_ino)`` as ``activity.lock`` -- a descriptor
          pointing at some other file is not the lease however plausible its number looks;
        * re-assert ``LOCK_EX|LOCK_NB`` on the inherited description. On the real inherited
          description this is a no-op that succeeds, because the lock already belongs to it. On an
          ordinary independently opened descriptor it FAILS while the parent's lock is held, which
          is exactly the forgery gate 9 describes.
        """
        fd = self.fd
        if fd is None:
            raise LeaseInheritanceError(
                f"no inherited lease descriptor: {ENV_LEASE_FD} was absent or unusable"
            )
        try:
            inherited = os.fstat(fd)
        except OSError as exc:
            raise LeaseInheritanceError(f"inherited descriptor {fd} is not open: {exc}") from exc
        if not stat.S_ISREG(inherited.st_mode):
            raise LeaseInheritanceError(f"inherited descriptor {fd} is not a regular file")
        try:
            lock_file = os.stat(self.lock_path)
        except OSError as exc:
            raise LeaseInheritanceError(
                f"the deployment's activity lock {self.lock_path} does not exist, so the inherited "
                f"descriptor cannot be it: {exc}"
            ) from exc
        if (inherited.st_dev, inherited.st_ino) != (lock_file.st_dev, lock_file.st_ino):
            raise LeaseInheritanceError(
                f"inherited descriptor {fd} names inode {inherited.st_dev}:{inherited.st_ino}, not "
                f"the deployment's activity lock {self.lock_path} "
                f"({lock_file.st_dev}:{lock_file.st_ino})"
            )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise LeaseInheritanceError(
                f"inherited descriptor {fd} does not share the locked open file description of "
                f"{self.lock_path}: re-asserting the lease lock on it failed ({exc}). It is an "
                f"ordinary independently opened handle, not the parent's lease."
            ) from exc

    def disarm_inheritance(self) -> None:
        """Stop the descriptor propagating any further, before any spawn (invariant 3)."""
        if self.fd is None:
            return
        os.set_inheritable(self.fd, False)

    @classmethod
    def from_inherited(
        cls,
        deployment_dir: Path,
        *,
        deployment_key: str,
        env: Mapping[str, str] | None = None,
    ) -> "PosixLeaseAdapter":
        """Bind the descriptor named by ``LM3_LEASE_FD``. Does NOT validate -- the caller does that.

        Keeping binding and validation apart is what lets ``grant.redeem_grant`` run the plan's
        ordered procedure, where the lease check happens in step 1 and nothing is renamed before it
        passes (gate 22).
        """
        values = _env(env)
        raw = values.get(ENV_LEASE_FD)
        if raw is None or not raw.strip():
            raise LeaseInheritanceError(
                f"{ENV_LEASE_FD} is not set: this process was not launched as an approved "
                f"subactivity of a lease holder"
            )
        try:
            fd = int(raw)
        except ValueError as exc:
            raise LeaseInheritanceError(f"{ENV_LEASE_FD}={raw!r} is not a descriptor number") from exc
        if fd < 0:
            raise LeaseInheritanceError(f"{ENV_LEASE_FD}={raw!r} is not a descriptor number")
        adapter = cls(deployment_dir, deployment_key=deployment_key)
        adapter.fd = fd
        adapter.inherited = True
        return adapter


def _noop_close() -> None:
    """POSIX handoff close: there is nothing to close.

    The descriptor handed to the child is the ROOT's own, still held for the whole root activity --
    closing it after the spawn would drop the root's reference. Windows is the opposite case: there
    the handoff creates a temporary duplicate that MUST be closed on every path (gate 34).
    """


# --------------------------------------------------------------------------------------------- #
# Windows adapter -- one named Global\ kernel event
# --------------------------------------------------------------------------------------------- #

class WindowsLeaseAdapter:
    """The Windows lease: existence of ``Global\\lm3-lease-<sid hash>-<deployment key>``.

    ``activity.lock`` is created for directory shape and then ignored completely -- never locked,
    never consulted (plan section 2.2, revision 11 J3).
    """

    def __init__(
        self,
        deployment_dir: Path,
        *,
        deployment_key: str,
        win32: Any | None = None,
        sid: str | None = None,
    ) -> None:
        self.deployment_dir = Path(deployment_dir)
        self.lock_path = self.deployment_dir / ACTIVITY_LOCK_FILENAME
        self.deployment_key = deployment_key
        self.handle: int | None = None
        self.inherited = False
        self._win32 = win32
        self._sid = sid
        self._event_name: str | None = None

    # -- lazily resolved Win32 plumbing -------------------------------------------------------- #

    @property
    def win32(self) -> Any:
        """The injected surface, or the real ctypes one built on first use.

        Lazy on purpose: constructing the adapter must not reach ``ctypes.WinDLL``, so that
        ``lease_adapter(..., platform_name="win32", win32=fake)`` builds and exercises this class on
        Linux.
        """
        if self._win32 is None:
            self._win32 = real_win32_surface()
        return self._win32

    @property
    def sid(self) -> str:
        if self._sid is None:
            self._sid = self.win32.current_user_sid()
        return self._sid

    @property
    def event_name(self) -> str:
        if self._event_name is None:
            self._event_name = windows_lease_event_name(
                self.deployment_key, sid_hash_value=sid_hash(self.sid)
            )
        return self._event_name

    # -- acquisition --------------------------------------------------------------------------- #

    def _touch_passive_artifact(self) -> None:
        """Create ``activity.lock`` so the registry directory looks the same on every platform.

        It is never locked and never read. A best-effort failure here must not fail the lease: the
        event is the sole authority, and a missing shape file is cosmetic.
        """
        _ensure_deployment_dir(self.deployment_dir)
        try:
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, _LOCK_MODE)
        except OSError as exc:                                        # pragma: no cover - defensive
            log.debug("could not create the passive activity.lock artifact: %s", exc)
            return
        os.close(fd)

    def acquire(self) -> None:
        if self.handle is not None:
            raise LeaseError(
                f"this lease adapter already holds {self.event_name}; a second acquire would hide a "
                f"double-start bug rather than fix one"
            )
        self._touch_passive_artifact()
        win32 = self.win32
        security = win32.current_user_security_attributes()
        try:
            handle = win32.create_event(
                self.event_name,
                security_attributes=security,
                manual_reset=False,
                initial_state=False,
                inheritable=False,      # the owner handle is NEVER inheritable -- gate 33
            )
        except OSError as exc:
            # No Local\ fallback. A global event needs no privilege, so a failure here means the
            # environment is broken in a way that would silently weaken the invariant.
            raise LeaseError(
                f"could not create the global lease event {self.event_name}: {exc}. LM3 does not "
                f"fall back to a Local\\ event, which would be scoped to one logon session."
            ) from exc
        if win32.get_last_error() == ERROR_ALREADY_EXISTS:
            # CreateEventW returned a VALID handle to the existing object. Close it BEFORE raising:
            # the object lives until its last handle closes, so a loser that keeps it pins the
            # deployment occupied after the real run exits (gate 31).
            try:
                win32.close_handle(handle)
            except OSError as exc:                                    # pragma: no cover - defensive
                log.warning("losing contender could not close its lease-event handle: %s", exc)
            raise RuntimeBusyError(self.deployment_key)
        self.handle = handle

    def release(self) -> None:
        """Close this process's handle. Idempotent. The object dies with its LAST handle."""
        handle, self.handle = self.handle, None
        if handle is None:
            return
        try:
            self.win32.close_handle(handle)
        except OSError as exc:                                        # pragma: no cover - defensive
            log.debug("closing the lease-event handle failed: %s", exc)

    def is_held(self) -> bool:
        return self.handle is not None

    def probe_occupied(self) -> bool:
        """``OpenEventW`` and close immediately -- observation, never acquisition."""
        try:
            handle = self.win32.open_event(self.event_name)
        except OSError:
            return False
        try:
            self.win32.close_handle(handle)
        except OSError:                                               # pragma: no cover - defensive
            pass
        return True

    # -- handing the reference to one approved subactivity ------------------------------------- #

    @contextmanager
    def _handoff(self, capability: str) -> Iterator[ChildHandoff]:
        """Yield the launch keywords for ONE approved subactivity.

        ``popen_kwargs`` carries a ``STARTUPINFOEX`` allowlist naming the lease duplicate only. A
        launcher that also passes the section 2.4 status-pipe handle MERGES both values into one
        allowlist: a second ``startupinfo`` would replace this one, and a ``handle_list`` is
        exhaustive -- whatever is left out is not inherited.
        """
        if self.handle is None:
            raise LeaseNotHeldError(
                "cannot hand a lease reference to a subactivity without holding the lease"
            )
        # Steps 2-6 of plan section 2.2's subactivity launch, in one indivisible window: the owner
        # handle stays untouched and non-inheritable, a temporary INHERITABLE duplicate exists only
        # inside this block, and the process-creation mutex keeps any concurrent CreateProcess out
        # of the window so no unrelated worker can inherit the duplicate.
        with process_creation_lock():
            duplicate = self.win32.duplicate_handle(self.handle, inheritable=True)
            closed = False

            def close() -> None:
                # Idempotent: the launcher is encouraged to close as soon as CreateProcess returns,
                # and the context manager closes again on the way out.
                nonlocal closed
                if closed:
                    return
                closed = True
                try:
                    self.win32.close_handle(duplicate)
                except OSError as exc:                                # pragma: no cover - defensive
                    log.warning("could not close the subactivity's inheritable duplicate: %s", exc)

            try:
                yield ChildHandoff(
                    capability=capability,
                    env={
                        ENV_LEASE_EVENT_HANDLE: str(duplicate),
                        ENV_LEASE_CAPABILITY: capability,
                    },
                    popen_kwargs=_win32.handle_list_popen_kwargs(
                        [duplicate], platform_name=self._platform_hint()
                    ),
                    close=close,
                    lease_handle=duplicate,
                )
            finally:
                # EVERY path, including every failure path: a spawn that raised would otherwise leak
                # a lease reference that nothing will ever release (gate 34).
                close()

    def _platform_hint(self) -> str:
        """The REAL platform, never the simulated one -- deliberately.

        ``subprocess.STARTUPINFO`` exists only on Windows, so a Linux test driving this adapter
        against a fake surface has to receive the off-Windows stand-in. It carries the same
        ``lpAttributeList`` mapping, so the assertion the test writes about the handle allowlist is
        the assertion that holds against the real object on Windows.
        """
        return sys.platform

    def child_handoff(self, capability: str) -> AbstractContextManager[ChildHandoff]:
        return self._handoff(capability)

    # -- child side ---------------------------------------------------------------------------- #

    def validate_inherited(self) -> None:
        """``OpenEventW`` the canonical name and ``CompareObjectHandles`` against the inherited one.

        The comparison handle is closed on BOTH branches -- a leaked one would pin the lease object
        alive in a process that was about to be rejected.
        """
        if self.handle is None:
            raise LeaseInheritanceError(
                f"no inherited lease handle: {ENV_LEASE_EVENT_HANDLE} was absent or unusable"
            )
        try:
            canonical = self.win32.open_event(self.event_name, desired_access=EVENT_ALL_ACCESS)
        except OSError as exc:
            raise LeaseInheritanceError(
                f"the canonical lease event {self.event_name} could not be opened, so the inherited "
                f"handle cannot be validated against it: {exc}"
            ) from exc
        try:
            same = self.win32.compare_object_handles(canonical, self.handle)
        finally:
            try:
                self.win32.close_handle(canonical)
            except OSError as exc:                                    # pragma: no cover - defensive
                log.debug("closing the comparison handle failed: %s", exc)
        if not same:
            raise LeaseInheritanceError(
                f"the inherited handle {self.handle} is not the deployment's lease event "
                f"{self.event_name}: CompareObjectHandles says they are different kernel objects"
            )

    def disarm_inheritance(self) -> None:
        """``SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0)`` (plan section 2.2, gate 32)."""
        if self.handle is None:
            return
        self.win32.set_handle_information(self.handle, HANDLE_FLAG_INHERIT, 0)

    @classmethod
    def from_inherited(
        cls,
        deployment_dir: Path,
        *,
        deployment_key: str,
        env: Mapping[str, str] | None = None,
        win32: Any | None = None,
    ) -> "WindowsLeaseAdapter":
        """Bind the handle named by ``LM3_LEASE_EVENT_HANDLE``. Does NOT validate."""
        values = _env(env)
        raw = values.get(ENV_LEASE_EVENT_HANDLE)
        if raw is None or not raw.strip():
            raise LeaseInheritanceError(
                f"{ENV_LEASE_EVENT_HANDLE} is not set: this process was not launched as an approved "
                f"subactivity of a lease holder"
            )
        try:
            handle = int(raw)
        except ValueError as exc:
            raise LeaseInheritanceError(
                f"{ENV_LEASE_EVENT_HANDLE}={raw!r} is not a handle value"
            ) from exc
        if handle <= 0:
            raise LeaseInheritanceError(f"{ENV_LEASE_EVENT_HANDLE}={raw!r} is not a handle value")
        adapter = cls(deployment_dir, deployment_key=deployment_key, win32=win32)
        adapter.handle = handle
        adapter.inherited = True
        return adapter


# --------------------------------------------------------------------------------------------- #
# Adapter selection and the lease object
# --------------------------------------------------------------------------------------------- #

def lease_adapter(
    deployment_dir: Path,
    *,
    deployment_key: str,
    platform_name: str | None = None,
    win32: Any | None = None,
) -> LeaseAdapter:
    """Pick the platform primitive. ``platform_name`` is injectable so Linux can drive both."""
    if _is_windows(platform_name):
        return WindowsLeaseAdapter(deployment_dir, deployment_key=deployment_key, win32=win32)
    return PosixLeaseAdapter(deployment_dir, deployment_key=deployment_key)


def inherit_lease(
    *,
    deployment_dir: Path | None = None,
    deployment_key: str | None = None,
    env: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    win32: Any | None = None,
) -> "RuntimeLease":
    """Bind the lease reference this process inherited from its parent.

    Binding only. Validation is the caller's, because ``grant.redeem_grant`` owns the order in which
    the checks happen and nothing may be renamed before the lease check passes (gate 22).
    """
    values = _env(env)
    key = deployment_key or paths.deployment_key(values)
    directory = (
        Path(deployment_dir)
        if deployment_dir is not None
        else paths.deployment_runtime_dir(env=values, deployment=key)
    )
    if _is_windows(platform_name):
        adapter: LeaseAdapter = WindowsLeaseAdapter.from_inherited(
            directory, deployment_key=key, env=values, win32=win32
        )
    else:
        adapter = PosixLeaseAdapter.from_inherited(directory, deployment_key=key, env=values)
    return RuntimeLease(deployment_dir=directory, deployment_key=key, env=values, adapter=adapter)


class RuntimeLease:
    """One deployment's activity lease: acquire it, hold it, hand it to approved subactivities.

    The lease is the ONLY thing that decides whether a deployment is occupied. Records describe what
    is running; they never grant or withhold exclusivity, which is why a corrupt or newer-schema
    ``active.json`` still leaves the deployment busy (plan section 2.9).
    """

    def __init__(
        self,
        *,
        deployment_dir: Path | None = None,
        deployment_key: str | None = None,
        env: Mapping[str, str] | None = None,
        adapter: LeaseAdapter | None = None,
        platform_name: str | None = None,
    ) -> None:
        values = _env(env)
        self.deployment_key = deployment_key or paths.deployment_key(values)
        self.deployment_dir = (
            Path(deployment_dir)
            if deployment_dir is not None
            else paths.deployment_runtime_dir(env=values, deployment=self.deployment_key)
        )
        self.adapter: LeaseAdapter = adapter or lease_adapter(
            self.deployment_dir, deployment_key=self.deployment_key, platform_name=platform_name
        )

    # -- state ---------------------------------------------------------------------------------- #

    @property
    def held(self) -> bool:
        """True when THIS process holds or has inherited the lease reference.

        A property rather than a stored flag so it cannot drift from the adapter, which is the only
        thing that actually knows.
        """
        return self.adapter.is_held()

    def probe_occupied(self) -> bool:
        return self.adapter.probe_occupied()

    # -- lifecycle ------------------------------------------------------------------------------ #

    def acquire(self) -> None:
        """Take the lease, or raise :class:`RuntimeBusyError` carrying the sanitized winner.

        Plan section 3.3 fixes WHEN this is called: after cheap config loading and validation have
        resolved project identity, and BEFORE hardware profiling, orphan-worker cleanup, directory
        creation, database writes, model loading, or GPU work. :func:`acquire_root_lease` is the
        entry point that states and checks that ordering.
        """
        try:
            self.adapter.acquire()
        except RuntimeBusyError as busy:
            raise self._busy_error(busy) from None

    def release(self) -> None:
        self.adapter.release()

    def __enter__(self) -> "RuntimeLease":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    # -- subactivity plumbing -------------------------------------------------------------------- #

    def child_handoff(self, capability: str) -> AbstractContextManager[ChildHandoff]:
        return self.adapter.child_handoff(capability)

    def validate_inherited(self) -> None:
        self.adapter.validate_inherited()

    def disarm_and_clear(self, env: MutableMapping[str, str] | None = None) -> None:
        """The one call an approved child makes before it spawns anything (plan section 2.2 step 4).

        Disarm first, then clear: if the order were reversed, a spawn racing between the two would
        find no environment variables but would still inherit a live reference -- and a worker
        holding the lease with nothing in its environment to explain why is the worst version of
        invariant 3's failure, because nothing can even diagnose it.
        """
        self.adapter.disarm_inheritance()
        clear_lease_env(env)

    @classmethod
    def inherited(cls, **kwargs: Any) -> "RuntimeLease":
        """Alias of :func:`inherit_lease`, for callers that already have the class in hand."""
        return inherit_lease(**kwargs)

    # -- busy reporting -------------------------------------------------------------------------- #

    def _busy_error(self, busy: RuntimeBusyError) -> RuntimeBusyError:
        """Decorate a bare adapter busy error with the winner's SANITIZED record.

        ``records`` is imported here rather than at module scope: it reads the lease through this
        module, and a top-level import would be a cycle. A failure to read or sanitize the record is
        deliberately NOT fatal -- exclusivity comes from the lock, and
        :class:`RuntimeBusyError` documents ``active=None`` as the honest answer when no readable
        record exists.
        """
        active = None
        classification = RecordClassification.LIVE
        try:
            from . import records                                     # noqa: PLC0415 - cycle-avoidance

            snapshot = records.read_runtime(self.deployment_dir, lease_probe=lambda: True)
            classification = snapshot.classification
            if snapshot.record is not None:
                active = records.sanitize_record(snapshot.record)
        except Exception as exc:                                      # noqa: BLE001 - never mask BUSY
            log.debug("could not describe the lease winner for %s: %s", self.deployment_key, exc)
        # No ``message=``: RuntimeBusyError describes itself from the record, and a message built by
        # the adapter (which had no record) would hide the winner's identity the API needs for 409.
        return RuntimeBusyError(self.deployment_key, active=active, classification=classification)


# --------------------------------------------------------------------------------------------- #
# Acquisition ordering (plan section 3.3) and abandoned-record cleanup
# --------------------------------------------------------------------------------------------- #

def acquire_root_lease(
    activity: Activity,
    *,
    deployment_dir: Path | None = None,
    deployment_key: str | None = None,
    env: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    adapter: LeaseAdapter | None = None,
    cuda_initialized: Callable[[], bool] | None = None,
) -> RuntimeLease:
    """Acquire the lease for a ROOT activity, with the section 3.3 ordering rules enforced.

    Three things it refuses or reports, all of which are ordering bugs that would otherwise show up
    as a mysterious second run:

    * a CHILD activity may never acquire. ``calibration_pipeline``
      INHERIT their parent's reference (invariant 2); acquiring would mean the parent's lease and
      the child's lease are two different things, and killing the parent would free one of them.
    * a process that inherited a lease reference may never acquire a second one. The environment
      still naming ``LM3_LEASE_FD`` / ``LM3_LEASE_EVENT_HANDLE`` means this process is somebody's
      approved child, and a child that acquires has skipped ``disarm_and_clear``.
    * GPU work must not precede the lease. A CUDA context already initialized proves the caller ran
      model or device work first, which is exactly what section 3.3 forbids and what gate 37 tests
      from the loser's side. It is reported, never raised: a long-lived Python session that
      legitimately holds a context must still be able to start a run, and refusing would trade a
      correctness warning for an outage.

    ``cuda_initialized`` is injected so this is testable without importing torch -- the default
    probe answers False whenever torch has not already been imported by somebody else.
    """
    if activity in CHILD_ACTIVITIES:
        raise LeaseError(
            f"{activity.value} is a subactivity: it inherits its parent's lease reference and must "
            f"never acquire a lease of its own (invariant 2)"
        )
    values = _env(env)
    inherited = [name for name in (ENV_LEASE_FD, ENV_LEASE_EVENT_HANDLE) if values.get(name)]
    if inherited:
        raise LeaseError(
            f"this process inherited a lease reference ({', '.join(inherited)}) and must not acquire "
            f"a second one. An approved subactivity calls RuntimeLease.disarm_and_clear() instead."
        )
    probe = cuda_initialized or _cuda_context_exists
    try:
        if probe():
            log.warning(
                "the deployment lease is being acquired AFTER a CUDA context already existed. Plan "
                "section 3.3 requires the lease before hardware profiling, directory creation, "
                "database writes, model loading or GPU work, so that a busy loser mutates nothing."
            )
    except Exception as exc:                                          # noqa: BLE001 - diagnostics only
        log.debug("could not check for an existing CUDA context: %s", exc)
    lease = RuntimeLease(
        deployment_dir=deployment_dir, deployment_key=deployment_key, env=values,
        adapter=adapter, platform_name=platform_name,
    )
    lease.acquire()
    return lease


def _cuda_context_exists() -> bool:
    """True only if torch is ALREADY imported and its CUDA runtime is already initialized.

    Never imports torch: importing a multi-hundred-megabyte framework to check whether it was
    imported would itself be the expensive work this check exists to keep behind the lease.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    cuda = getattr(torch, "cuda", None)
    is_initialized = getattr(cuda, "is_initialized", None)
    return bool(is_initialized()) if callable(is_initialized) else False


def probe_deployment_occupied(
    deployment_dir: Path | None = None,
    *,
    deployment_key: str | None = None,
    env: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    win32: Any | None = None,
) -> bool:
    """Is this deployment occupied? An observer's question -- it retains no lease reference.

    POSIX answers it with a bounded shared test-and-release on the observer's own private open file
    description (there is no query-only ``flock``), Windows with a pure ``OpenEventW`` existence
    test. Neither ever becomes a lease holder, and :meth:`PosixLeaseAdapter.acquire` carries a
    bounded retry precisely so this question can never refuse a legitimate root (invariant 13).

    This is the question that turns a leftover ``active.json`` into a classification: a record while
    the lease is FREE is abandoned, and a record while the lease is HELD is live (plan section 3.3).
    """
    values = _env(env)
    key = deployment_key or paths.deployment_key(values)
    directory = (
        Path(deployment_dir)
        if deployment_dir is not None
        else paths.deployment_runtime_dir(env=values, deployment=key)
    )
    adapter = lease_adapter(directory, deployment_key=key, platform_name=platform_name, win32=win32)
    return adapter.probe_occupied()


@contextmanager
def cleanup_lease(
    deployment_dir: Path | None = None,
    *,
    deployment_key: str | None = None,
    env: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    adapter: LeaseAdapter | None = None,
) -> Iterator[RuntimeLease | None]:
    """Hold the lease for the sole purpose of classifying and recovering an abandoned record.

    Plan section 3.3: "a reader finding ``active.json`` while the lease is acquirable classifies the
    record as stale/abandoned UNDER THE CLEANUP LOCK, and must never signal the recorded PID." The
    cleanup lock is not a second lock -- it is this same activity lease, which is what makes the
    classification safe: whoever holds it is the only process that can be rewriting the record, and
    if the lease could not be taken then the deployment is genuinely live and there is nothing
    abandoned to clean up.

    Yields the held lease, or ``None`` when the deployment is busy. ``records.recover_abandoned``
    takes that lease and does the writing; this module never interprets a record and never signals
    the PID inside one.
    """
    lease = RuntimeLease(
        deployment_dir=deployment_dir, deployment_key=deployment_key, env=env,
        adapter=adapter, platform_name=platform_name,
    )
    try:
        lease.acquire()
    except RuntimeBusyError:
        yield None
        return
    try:
        yield lease
    finally:
        lease.release()
