"""The context manager every LM3 execution entry point uses to own a runtime activity.

Plan sections 2.2 (roots and inherited subactivities), 2.4 (the launch handshake and the staging
boundary), 2.7 (what a ``starting`` record may publish), 3.3 (the lifecycle and its ordering) and
3.4 (the immutable launch manifest). Step 3 is what turns the Step 2 primitives -- the lease, the
record store, the one-use grants, the launch composer -- into something production code calls, and
this module is the single seam through which it does that. ``machine3()``, standalone and GUI
hardware setup, ``calibrate._run_pipeline`` and the server's ``POST /v1/run/start`` child all reach
the lease through here (invariant 4), so there is exactly one implementation of the ordering rules
rather than four that drift.

What it does NOT do, deliberately:

* it never decides control authority. Section 2.5 makes that handle ownership, which lives in the
  server, and nothing in a record grants it.
* it never signals a PID read out of a JSON file (invariant 12). The only processes it terminates
  are subactivities it launched and still holds a handle for.
* it never creates the run directory, opens the database, probes hardware or touches CUDA. Those
  are the caller's, and section 3.3 puts every one of them AFTER the lease so a busy loser mutates
  nothing (invariant 14).

The ordering this module enforces, once, for everybody::

    cheap config load + validate (the caller's)
      -> resolve project identity            <- pure; no directory exists yet
      -> acquire the lease                   <- RuntimeBusyError here means exit 75, nothing written
      -> recover any abandoned record        <- sections 2.9/3.2: finalize or quarantine, never clobber
      -> publish `starting`                  <- run_dir, active_db_path, log_path; NEVER tmp_dir
      -> write the section 2.4 status line   <- exactly one JSON line, then continue
      -> [caller: hardware profile, orphan reaping, build_dirs, DB, models, GPU]
      -> publish `running`                   <- once the DB and log paths exist
      -> [caller: the actual work]
      -> join or terminate every subactivity <- BEFORE finalizing (gate 5)
      -> finalize done | error | stopped | interrupted
      -> release the lease reference         <- last, in a finally, on every path

The compatibility flag
----------------------
Every production call site is gated by ``LM3_RUNTIME_V2``, default ON since the cutover, and
:func:`runtime_v2_enabled` is the ONLY reader of it. With the flag explicitly set to ``0`` the
context managers still run the caller's body but acquire nothing, publish nothing and create
nothing: the handle they yield is
:class:`DisabledActivity`, whose every method is a no-op. That is what makes "flag off means
byte-identical behavior" a property of one module instead of a promise repeated at four call sites.

A known limitation, stated rather than hidden
---------------------------------------------
The paths published here come from :func:`~leafmachine3.core.runtime.config_io.resolve_run_paths`,
which honors ``project.output.active_state_dir`` while ``core.dirs.build_dirs`` does not. For a
desktop run the two agree, and ``tests/test_relative_path_contract.py`` pins that. For a section
2.10 STAGED run they do not, and the record would name a database the run never creates -- so a
caller in that mode must pass an authoritative ``early=`` rather than let this module guess.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import signal
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence

from .. import paths
from . import config_io
from ._types import (
    CHILD_ACTIVITIES,
    CHILD_PARENT_ACTIVITY,
    DEFAULT_GRANT_TTL_S,
    ENV_LEASE_CAPABILITY,
    ENV_STATUS_FD,
    ENV_STATUS_HANDLE,
    LEASE_ENV_VARS,
    LIVE_STATES,
    MAX_ERROR_CHARS,
    MAX_HANDSHAKE_BYTES,
    ROOT_ACTIVITIES,
    TERMINAL_STATES,
    Activity,
    ActivityRole,
    ChildSummary,
    ConfigRef,
    ControlBlock,
    ControlMode,
    HandshakeMessage,
    GrantRecord,
    HandshakeStatus,
    HardwareBlock,
    Launcher,
    LeaseInheritanceError,
    ProjectBlock,
    RunState,
    RuntimeBusyError,
    RuntimeRecord,
)
from .grant import grant_path, issue_grant, redeem_grant, write_grant
from .launch import LaunchContribution, composed_launch
from .lease import RuntimeLease, acquire_root_lease, inherit_lease
from .records import (
    RecordStore,
    atomic_write_json,
    build_deployment_info,
    new_run_id,
    process_start_time,
    recover_abandoned,
    record_to_dict,
    sanitize_record,
    utc_now,
)

log = logging.getLogger(__name__)

__all__ = [
    "CHILD_ENV_VARS",
    "DEFAULT_CHILD_JOIN_TIMEOUT_S",
    "DEFAULT_CHILD_KILL_TIMEOUT_S",
    "ENV_CHILD_ACTIVITY",
    "ENV_CHILD_RUN_ID",
    "ENV_PARENT_RUN_ID",
    "ENV_RUNTIME_V2",
    "ActivityHandle",
    "ChildActivity",
    "DisabledActivity",
    "InheritedLease",
    "RootActivity",
    "StatusChannel",
    "SubactivityLaunch",
    "bind_child_lease",
    "child_activity",
    "child_base_env",
    "execution_activity",
    "is_approved_child",
    "launch_subactivity",
    "root_activity",
    "runtime_v2_enabled",
    "status_channel",
    "write_launch_manifest",
]


# --------------------------------------------------------------------------------------------- #
# The feature flag -- ONE reader (plan section 4, Step 3)
# --------------------------------------------------------------------------------------------- #

#: Step 3's one-release compatibility switch. Unset is ON; explicit false values select the old path.
ENV_RUNTIME_V2 = "LM3_RUNTIME_V2"

# Borrowed rather than re-spelled: ``core.paths`` already decided what a true-ish environment value
# looks like ("1", "true", "yes", "on"), and a second table here would eventually disagree with it
# over a value a user typed.
_TRUTHY = paths._TRUTHY


def runtime_v2_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Is the Step 3 runtime wiring turned on? THE only reader of ``LM3_RUNTIME_V2``.

    Every other module asks this function rather than reading the environment, because a flag read
    in four places is four places that can be spelled differently, defaulted differently, or missed
    when the flag is finally removed.

    **Default is ON.** It shipped OFF through Steps 3-7 so the wiring could land without changing
    behavior, and the Step 3 exit gate has since closed: the lease refuses a second root from every
    entry point, calibration runs under its parent's lease, the handshake returns 409 with the
    winner, and the batch wrapper records exit 75 as retryable. Leaving it off after that point
    would mean an ordinary launch keeps the blindness the whole refactor exists to remove -- a GUI
    opened during a CLI run reporting ``idle`` -- which is a default nobody would choose on purpose.

    Setting ``LM3_RUNTIME_V2=0`` pins the pre-Step-3 path. That escape hatch is deliberate and is
    what the flag-off characterization tests exercise; it goes away when the compatibility window
    documented in ``docs/DEPRECATIONS.md`` closes.
    """
    values = os.environ if env is None else env
    raw = values.get(ENV_RUNTIME_V2)
    if raw is None or not raw.strip():
        return True                               # unset means "the current runtime"
    return raw.strip().lower() in _TRUTHY


# --------------------------------------------------------------------------------------------- #
# Environment plumbing owned by this module
# --------------------------------------------------------------------------------------------- #
# The lease reference and the capability have their names in ``_types`` because ``lease`` and
# ``grant`` both read them. These three are the launch-side identity a child needs in order to find
# ITS OWN grant file, and only this module writes or reads them, so they live here.

ENV_CHILD_RUN_ID = "LM3_CHILD_RUN_ID"        # the child's own run_id: names children/<id>.grant.json
ENV_PARENT_RUN_ID = "LM3_PARENT_RUN_ID"      # the root that issued the grant
ENV_CHILD_ACTIVITY = "LM3_CHILD_ACTIVITY"    # the declared purpose, validated against the grant

CHILD_ENV_VARS: tuple[str, ...] = (ENV_CHILD_RUN_ID, ENV_PARENT_RUN_ID, ENV_CHILD_ACTIVITY)

#: How long a normally-completing root waits for a subactivity before it escalates. Generous: the
#: normal path has already joined the child, so reaching this timeout means something is wrong.
DEFAULT_CHILD_JOIN_TIMEOUT_S = 30.0
#: How long a terminating subactivity gets between SIGTERM and SIGKILL.
DEFAULT_CHILD_KILL_TIMEOUT_S = 10.0


def is_approved_child(env: Mapping[str, str] | None = None) -> bool:
    """True when this process was launched as an approved subactivity (section 2.2).

    The capability is the marker rather than the descriptor, because the descriptor number is
    platform-specific and the capability is not. A process for which this is true must NEVER
    acquire a lease -- :func:`~leafmachine3.core.runtime.lease.acquire_root_lease` refuses outright
    -- so ``machine3()`` uses this to choose between :func:`root_activity` and
    :func:`child_activity`.
    """
    values = os.environ if env is None else env
    return bool(values.get(ENV_LEASE_CAPABILITY))


def child_base_env(
    env: Mapping[str, str] | None = None,
    *,
    keep_status: bool = False,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """A filtered copy of this process's environment, safe to hand to a child.

    Copying ``os.environ`` wholesale is the bug this exists to prevent: it hands the child our
    ``LM3_STATUS_FD`` -- a descriptor NUMBER that means something different in the child, where the
    low numbers belong to somebody else -- and the child would write its handshake JSON into an
    unrelated file. The lease variables are stripped for the same reason: they come from the
    :class:`~leafmachine3.core.runtime._types.ChildHandoff`, freshly, or not at all.

    ``keep_status=True`` is for a child that IS the handshake child and is being given the same
    channel deliberately; it still only works if the descriptor is in the launch's allowlist.
    """
    values = dict(os.environ if env is None else env)
    for name in LEASE_ENV_VARS:
        values.pop(name, None)
    for name in CHILD_ENV_VARS:
        values.pop(name, None)
    if not keep_status:
        values.pop(ENV_STATUS_FD, None)
        values.pop(ENV_STATUS_HANDLE, None)
    if extra:
        values.update({str(k): str(v) for k, v in extra.items()})
    return values


# --------------------------------------------------------------------------------------------- #
# The launch handshake -- the CHILD half (plan section 2.4)
# --------------------------------------------------------------------------------------------- #

class StatusChannel:
    """The dedicated control pipe a launched child answers on. Exactly one JSON line, then EOF.

    Three properties, all of which are the reason it is a separate channel and not stdout:

    * it is SMALL and bounded, so the server can read it with a timeout and a byte cap;
    * the child's stdout/stderr go somewhere else entirely (a server-private request log), so a
      child may emit unbounded startup chatter before its status line without ever filling a pipe
      the server is not draining;
    * closing it is the EOF the server needs in order to stop waiting.

    A write failure never propagates. A run that has already acquired the lease must not die
    because the launcher went away -- the server's timeout branch (section 2.4, step 3) is what
    handles that side, by terminating the child it still holds.
    """

    def __init__(self, fd: int, *, close_fd: bool = True) -> None:
        self._fd = int(fd)
        self._close_fd = bool(close_fd)
        self._sent = False
        self._closed = False

    @property
    def fd(self) -> int:
        return self._fd

    @property
    def sent(self) -> bool:
        """True once a line has been attempted. A second send is refused, never queued."""
        return self._sent

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        platform_name: str | None = None,
        open_osfhandle: Callable[[int, int], int] | None = None,
    ) -> "StatusChannel | None":
        """Build the channel this process was launched with, or ``None`` when there is none.

        ``open_osfhandle`` is injectable so the Windows branch -- ``msvcrt.open_osfhandle`` on the
        inherited ``STARTUPINFOEX`` handle -- is exercised by unit tests on Linux instead of being
        written once and never run.
        """
        values = os.environ if env is None else env
        name = platform_name if platform_name is not None else sys.platform
        if name.startswith("win"):
            raw = values.get(ENV_STATUS_HANDLE)
            if not raw:
                return None
            try:
                handle = int(raw.strip())
            except ValueError:
                log.warning("%s=%r is not an integer handle; no launch handshake", ENV_STATUS_HANDLE, raw)
                return None
            opener = open_osfhandle or _msvcrt_open_osfhandle
            try:
                fd = opener(handle, 0)
            except OSError as exc:
                log.warning("%s=%s could not be opened as a descriptor: %s", ENV_STATUS_HANDLE, handle, exc)
                return None
            return cls(fd)

        raw = values.get(ENV_STATUS_FD)
        if not raw:
            return None
        try:
            fd = int(raw.strip())
        except ValueError:
            log.warning("%s=%r is not an integer descriptor; no launch handshake", ENV_STATUS_FD, raw)
            return None
        if fd <= 2:
            # stdin/stdout/stderr are NEVER the handshake channel (section 2.4). A launcher that
            # named one is a launcher whose child would interleave a status line with its startup
            # chatter, which is the deadlock-and-corruption case the separate pipe exists to avoid.
            log.error("%s=%d names a standard stream; refusing to use it as the status channel",
                      ENV_STATUS_FD, fd)
            return None
        try:
            os.fstat(fd)
        except OSError as exc:
            log.warning("%s=%d is not an open descriptor in this process: %s", ENV_STATUS_FD, fd, exc)
            return None
        return cls(fd)

    # -- the two messages ----------------------------------------------------------------------- #

    def send_acquired(self, *, run_id: str, project: ProjectBlock | Mapping[str, Any] | None = None) -> bool:
        """``{"status":"acquired","run_id":...,"project":{...}}`` -- then the child continues."""
        message: HandshakeMessage = {"status": HandshakeStatus.ACQUIRED.value, "run_id": str(run_id)}
        block = _project_dict(project)
        if block is not None:
            message["project"] = block  # type: ignore[typeddict-item]
        return self._write(message)

    def send_busy(self, active: RuntimeRecord | None = None) -> bool:
        """``{"status":"busy","active":{...sanitized winner...}}`` -- then the child exits 75.

        The winner's record is sanitized (redacted, field-bounded) before it leaves this process:
        it is about to become an HTTP 409 body, and section 3.2's "no bearer tokens, no secrets"
        rule does not stop at the process boundary.
        """
        message: HandshakeMessage = {"status": HandshakeStatus.BUSY.value}
        if active is not None:
            with contextlib.suppress(Exception):
                message["active"] = record_to_dict(sanitize_record(active))
        return self._write(message)

    # -- internals ------------------------------------------------------------------------------ #

    def _write(self, message: HandshakeMessage) -> bool:
        if self._sent:
            log.error("a second handshake line was suppressed: the protocol is exactly one line")
            return False
        if self._closed:
            log.warning("the status channel is already closed; dropping the handshake line")
            return False
        data = _handshake_bytes(message)
        self._sent = True
        try:
            written = 0
            while written < len(data):
                written += os.write(self._fd, data[written:])
        except OSError as exc:
            # The launcher went away. Its own timeout branch owns the consequence; ours is to keep
            # running the run that already holds the lease.
            log.warning("could not write the launch handshake: %s", exc)
            return False
        return True

    def close(self) -> None:
        """Close the write end -- the EOF the reader waits on -- and unpublish the descriptor.

        Removing ``LM3_STATUS_FD`` / ``LM3_STATUS_HANDLE`` from ``os.environ`` is not tidiness: fd
        numbers are recycled, so a child spawned later that still saw the variable would write a
        status line into whatever now occupies that number.
        """
        if self._closed:
            return
        self._closed = True
        for name in (ENV_STATUS_FD, ENV_STATUS_HANDLE):
            os.environ.pop(name, None)
        if self._close_fd:
            with contextlib.suppress(OSError):
                os.close(self._fd)


def status_channel(
    env: Mapping[str, str] | None = None,
    *,
    fd: int | None = None,
    platform_name: str | None = None,
    open_osfhandle: Callable[[int, int], int] | None = None,
) -> StatusChannel | None:
    """The channel for this process: an explicit ``fd`` wins, else the environment, else ``None``."""
    if fd is not None:
        return StatusChannel(int(fd))
    return StatusChannel.from_env(env, platform_name=platform_name, open_osfhandle=open_osfhandle)


def _msvcrt_open_osfhandle(handle: int, flags: int) -> int:
    """``msvcrt.open_osfhandle``, imported at CALL time so this module imports cleanly on Linux."""
    import msvcrt  # noqa: PLC0415 - Windows-only, deliberately not an import-time dependency

    return msvcrt.open_osfhandle(handle, flags)


def _handshake_bytes(message: Mapping[str, Any]) -> bytes:
    """One line, UTF-8, bounded by ``MAX_HANDSHAKE_BYTES`` (section 2.4's "oversized status")."""
    data = (json.dumps(message, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    if len(data) <= MAX_HANDSHAKE_BYTES:
        return data
    # Shed the detail rather than the answer: the status and the run_id are what the server acts
    # on, and an oversized line would be rejected wholesale.
    minimal = {"status": message.get("status")}
    if message.get("run_id"):
        minimal["run_id"] = message["run_id"]
    log.warning("the handshake line exceeded %d bytes; sending the identity only", MAX_HANDSHAKE_BYTES)
    return (json.dumps(minimal, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _project_dict(project: ProjectBlock | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if project is None:
        return None
    if isinstance(project, ProjectBlock):
        # Round-trip through a throwaway record rather than re-spelling the twelve keys here: one
        # serializer, so the handshake's project block and the record's cannot diverge.
        return {
            "run_name": project.run_name,
            "input_dirs": list(project.input_dirs),
            "artifact_dir": project.artifact_dir,
            "active_state_dir": project.active_state_dir,
            "active_db_path": project.active_db_path,
            "archive_mode": project.archive_mode.value,
            "archive_status": project.archive_status.value,
            "archive_pointer_path": project.archive_pointer_path,
            "archived_db_path": project.archived_db_path,
            "run_dir": project.run_dir,
            "log_path": project.log_path,
            "archive_error": project.archive_error,
        }
    return dict(project)


# --------------------------------------------------------------------------------------------- #
# The section 3.4 launch manifest
# --------------------------------------------------------------------------------------------- #

def write_launch_manifest(
    cfg: Any,
    *,
    run_id: str,
    launcher: Launcher | str,
    early: paths.EarlyRunPaths,
    tmp_dir: str | os.PathLike[str],
    parent_run_id: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    config: ConfigRef | None = None,
    settings_file: str | os.PathLike[str] | None = None,
    started_at: str | None = None,
    clock: Callable[[], float] | None = None,
) -> Path:
    """Atomically write ``<run>/logs/run_manifest.json`` and return its path (section 3.4).

    Call it AFTER ``build_dirs()`` -- ``tmp_dir`` is a required argument precisely so a caller who
    has not settled the scratch location cannot produce a manifest by accident. That is also why
    the manifest, and not the ``starting`` record, is where ``tmp_dir`` appears (section 2.7).

    It must not be written from inside ``build_dirs()``: that function runs three times in a normal
    run (``machine3.py``, ``core/ingest.py``, ``modules/ruler_conversion_factor.py``), and a
    manifest rewritten mid-run is not immutable.
    """
    payload = config_io.build_launch_manifest(
        cfg,
        run_id=run_id,
        launcher=launcher,
        early=early,
        tmp_dir=tmp_dir,
        parent_run_id=parent_run_id,
        overrides=overrides,
        config=config,
        settings_file=settings_file,
        started_at=started_at,
        clock=clock,
    )
    path = config_io.launch_manifest_path(early)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


# --------------------------------------------------------------------------------------------- #
# Handles
# --------------------------------------------------------------------------------------------- #

class ActivityHandle:
    """What the context managers yield. The base exists so a caller can be written once.

    ``machine3()`` does not care whether it is a root ``pipeline`` or a ``calibration_pipeline``
    child, and with the flag off it does not care that nothing is being recorded at all -- the same
    four calls (:meth:`mark_running`, :meth:`write_manifest`, :meth:`set_result`,
    :meth:`launch_subactivity`) are valid on every handle.
    """

    #: False on :class:`DisabledActivity` -- the flag-off handle, which owns nothing.
    enabled: bool = True

    run_id: str = ""
    activity: Activity = Activity.PIPELINE
    launcher: Launcher = Launcher.PYTHON
    parent_run_id: str | None = None

    def mark_running(self, *, project: ProjectBlock | None = None) -> None:
        raise NotImplementedError

    def write_manifest(self, *, tmp_dir: str | os.PathLike[str], **kwargs: Any) -> Path | None:
        raise NotImplementedError

    def set_result(
        self,
        state: RunState | None = None,
        *,
        returncode: int | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        raise NotImplementedError

    def launch_subactivity(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class DisabledActivity(ActivityHandle):
    """The handle yielded when ``LM3_RUNTIME_V2`` is off: every method is a no-op.

    Not ``None``, because a caller written against ``None`` grows an ``if`` around every call and
    those are exactly the branches that rot once the flag is removed.
    """

    enabled = False

    def __init__(self, *, activity: Activity, launcher: Launcher) -> None:
        self.activity = activity
        self.launcher = launcher
        self.run_id = ""
        self.parent_run_id = None
        self.lease = None
        self.store = None
        self.record = None

    def mark_running(self, *, project: ProjectBlock | None = None) -> None:
        return None

    def write_manifest(self, *, tmp_dir: str | os.PathLike[str], **kwargs: Any) -> Path | None:
        return None

    def set_result(
        self,
        state: RunState | None = None,
        *,
        returncode: int | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        return None

    @contextmanager
    def launch_subactivity(self, *args: Any, **kwargs: Any) -> Iterator["SubactivityLaunch"]:
        """Yields an inert launch: the caller's ``Popen`` still runs, with a plain environment."""
        env = kwargs.get("env")
        yield SubactivityLaunch(
            run_id="",
            capability="",
            env=dict(os.environ if env is None else env),
            popen_kwargs=dict(kwargs.get("base_kwargs") or {}),
        )


@dataclass
class _LiveChild:
    """One approved subactivity the root launched and is responsible for joining."""

    summary: ChildSummary
    proc: Any = None
    completed: bool = False
    returncode: int | None = None


class _RecordedActivity(ActivityHandle):
    """Shared machinery for the root and child handles: publish, advance, finalize."""

    def __init__(
        self,
        *,
        run_id: str,
        activity: Activity,
        launcher: Launcher,
        lease: RuntimeLease,
        store: RecordStore,
        record: RuntimeRecord,
        deployment_dir: Path,
        deployment_key: str,
        cfg: Any = None,
        early: paths.EarlyRunPaths | None = None,
        config: ConfigRef | None = None,
        settings_file: str | os.PathLike[str] | None = None,
        channel: StatusChannel | None = None,
    ) -> None:
        self.run_id = run_id
        self.activity = activity
        self.launcher = launcher
        self.parent_run_id = record.parent_run_id
        self.lease = lease
        self.store = store
        self.record = record
        self.deployment_dir = Path(deployment_dir)
        self.deployment_key = deployment_key
        self.early = early
        self.config = config
        self.manifest_path: Path | None = None
        #: Set when finalization could not complete -- most often gate 5, a live child. The record
        #: is deliberately left in place when that happens; see :meth:`_close`.
        self.finalize_error: BaseException | None = None
        self._cfg = cfg
        self._settings_file = settings_file
        self._channel = channel
        self._outcome: RunState | None = None
        self._requested: RunState | None = None
        self._returncode: int | None = None
        self._error: BaseException | str | None = None
        self._closed = False

    # -- publishing ----------------------------------------------------------------------------- #

    def _publish(self, record: RuntimeRecord) -> None:
        raise NotImplementedError

    def mark_running(self, *, project: ProjectBlock | None = None) -> None:
        """Move ``starting -> running``. Section 3.3: once the DB and log paths exist.

        The existence check is a warning, not a refusal: the caller knows when its database is
        open, and a record that lags reality by a few milliseconds is a far smaller problem than a
        run that refuses to start because a path was created one line later than expected.

        The LOG path is only checked when file logging is actually enabled. With
        ``project.logging.to_file: false`` -- which the deterministic mock baseline uses, and which
        any embedding caller may choose -- ``lm3.log`` is never created, so warning about its
        absence reports a configuration as a fault. A warning that fires on a supported setting is
        noise, and noise is what stops people reading warnings that matter.
        """
        if self._closed:
            raise RuntimeError(f"run {self.run_id} has already been finalized")
        block = project if project is not None else self.record.project
        if block is not None:
            expected = [block.active_db_path]
            if self._file_logging_enabled():
                expected.append(block.log_path)
            missing = [p for p in expected if p and not Path(p).exists()]
            if missing:
                log.warning("publishing 'running' for %s while %s do(es) not exist yet",
                            self.run_id, ", ".join(missing))
        updated = dataclasses.replace(
            self.record, state=RunState.RUNNING, updated_at=utc_now(),
            project=block if block is not None else self.record.project,
        )
        self._publish(updated)

    def _file_logging_enabled(self) -> bool:
        """Does this run write ``lm3.log`` at all? ``project.logging.to_file``, defaulting True.

        Read defensively: this is diagnostics, and a config shape we did not anticipate must not be
        able to turn a warning into a crash on the way to publishing ``running``.
        """
        try:
            logging_cfg = getattr(getattr(self._cfg, "project", None), "logging", None)
            if logging_cfg is None:
                return True
            getter = getattr(logging_cfg, "get", None)
            value = getter("to_file", True) if callable(getter) else getattr(logging_cfg, "to_file", True)
            return bool(value)
        except Exception:                       # noqa: BLE001 - diagnostics must never break a run
            return True

    def write_manifest(
        self,
        *,
        tmp_dir: str | os.PathLike[str],
        overrides: Mapping[str, Any] | None = None,
        early: paths.EarlyRunPaths | None = None,
        cfg: Any = None,
        **kwargs: Any,
    ) -> Path | None:
        """Write the section 3.4 launch manifest for this activity. See :func:`write_launch_manifest`."""
        resolved_early = early if early is not None else self.early
        if resolved_early is None:
            raise ValueError(
                f"{self.activity.value} has no resolved run paths, so it has no launch manifest; "
                f"pass early= if you have them"
            )
        self.manifest_path = write_launch_manifest(
            cfg if cfg is not None else self._cfg,
            run_id=self.run_id,
            launcher=self.launcher,
            early=resolved_early,
            tmp_dir=tmp_dir,
            parent_run_id=self.parent_run_id,
            overrides=overrides,
            config=self.config,
            settings_file=self._settings_file,
            started_at=self.record.started_at,
            **kwargs,
        )
        return self.manifest_path

    def set_result(
        self,
        state: RunState | None = None,
        *,
        returncode: int | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        """Declare the terminal outcome the caller wants, e.g. ``stopped`` for an external Stop.

        Only the *requested* state is remembered: an exception escaping the body still wins, because
        what actually happened outranks what was planned.
        """
        if state is not None:
            if state not in TERMINAL_STATES:
                raise ValueError(f"{state.value!r} is not a terminal state")
            self._requested = state
        if returncode is not None:
            self._returncode = int(returncode)
        if error is not None:
            self._error = error

    # -- outcome and teardown -------------------------------------------------------------------- #

    def _record_outcome(self, state: RunState | None, *, error: BaseException | None = None) -> None:
        self._outcome = state or self._requested or RunState.DONE
        if error is not None and self._error is None:
            self._error = error

    def _terminal_record(self) -> RuntimeRecord:
        state = self._outcome or RunState.DONE
        now = utc_now()
        returncode = self._returncode
        if returncode is None:
            returncode = 0 if state is RunState.DONE else None
        return dataclasses.replace(
            self.record, state=state, updated_at=now, finished_at=now,
            error=_error_text(self._error), returncode=returncode,
        )

    def _finalize(self) -> None:
        raise NotImplementedError

    def _close(self) -> None:
        """Finalize, then release the lease reference -- last, in a finally, on every path.

        Nothing in here is allowed to raise. This runs in the context manager's ``finally``, where
        a new exception would REPLACE whatever the run actually failed with, and the lease release
        below it would be skipped. Failures are logged and recorded on :attr:`finalize_error`
        instead, which is also the observable for gate 5: a root that could not join a live child
        leaves ``active.json`` in place, because that record is then the only description of what
        is still holding the deployment.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._finalize()
        except BaseException as exc:  # noqa: BLE001 - see the docstring; never mask the real error
            self.finalize_error = exc
            log.exception("could not finalize %s %s; leaving its record in place",
                          self.activity.value, self.run_id)
        finally:
            if self._channel is not None:
                with contextlib.suppress(Exception):
                    self._channel.close()
            try:
                self.lease.release()
            except Exception:  # noqa: BLE001 - a release failure must not hide the run's outcome
                log.exception("could not release the lease reference for %s", self.run_id)


class RootActivity(_RecordedActivity):
    """A root activity's handle: it owns ``active.json``, ``last.json`` and its subactivities."""

    def __init__(self, *, children_join_timeout: float = DEFAULT_CHILD_JOIN_TIMEOUT_S,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._children: dict[str, _LiveChild] = {}
        self._current_child: ChildSummary | None = None
        self._last_child: ChildSummary | None = None
        self._join_timeout = float(children_join_timeout)

    def _publish(self, record: RuntimeRecord) -> None:
        record = dataclasses.replace(record, current_child=self._current_child,
                                     last_child=self._last_child)
        self.store.write_active(record)
        self.record = record

    # -- subactivities --------------------------------------------------------------------------- #

    def launch_subactivity(self, *args: Any, **kwargs: Any) -> Any:
        """See :func:`launch_subactivity`; this is the same thing bound to ``self``."""
        return launch_subactivity(self, *args, **kwargs)

    def live_children(self) -> tuple[str, ...]:
        """Run ids whose process handle says they are still alive. The authority for gate 5."""
        alive = []
        for run_id, child in self._children.items():
            proc = child.proc
            if proc is not None and _proc_alive(proc):
                alive.append(run_id)
        return tuple(alive)

    def _register_child(self, summary: ChildSummary) -> None:
        """Record the approved child BEFORE it is launched (section 3.2, sequence step 3).

        Before, not after, so a hard kill inside the launch window still leaves the deployment
        describable: the summary names what may be holding the inherited lease reference.
        """
        self._children[summary.run_id] = _LiveChild(summary=summary)
        self._current_child = summary
        self.store.set_current_child(summary)
        self.record = dataclasses.replace(self.record, current_child=summary)

    def _settle_child(self, launch: "SubactivityLaunch") -> None:
        """Called when a subactivity launch scope exits: promote, forget, or deliberately keep."""
        child = self._children.get(launch.run_id)
        if child is None:
            return
        if child.proc is not None and _proc_alive(child.proc):
            # Still running. It stays ``current_child`` on purpose -- the root will terminate it in
            # _close(), and until then the record must keep naming it.
            return
        if self._child_never_started(child):
            # The spawn itself failed: there is nothing to promote, and section 3.2 step 4 only
            # moves current_child -> last_child "on normal child completion".
            self._forget_child(launch.run_id)
            return
        self._retire_child(child)

    def _child_never_started(self, child: _LiveChild) -> bool:
        """True when no process was ever created for this summary. Three independent negatives.

        Each one alone is ambiguous, which is why all three are required:

        * ``proc is None`` -- the caller never called :meth:`SubactivityLaunch.attach`. On its own
          that is legal: a caller that blocks on the child with ``subprocess.run`` never attaches.
        * ``completed`` is False -- nobody declared an exit through
          :meth:`SubactivityLaunch.completed` either.
        * ``children/<run_id>.json`` does not EXIST -- existence, deliberately, not liveness. A
          child that ran under ``subprocess.run`` and finalized has a record that is no longer
          live, so ``_child_record_is_live`` cannot tell it apart from a child that never ran;
          the mere presence of the file can.
        """
        if child.proc is not None or child.completed:
            return False
        try:
            return not self.store.child_record_path(child.summary.run_id).exists()
        except Exception:  # noqa: BLE001 - an unreadable children/ is not evidence of a failed spawn
            return False

    def _forget_child(self, run_id: str) -> None:
        """Erase a summary whose spawn failed, instead of retiring it (section 3.2, step 4).

        Step 4 moves ``current_child`` to ``last_child`` "on normal child completion", and a spawn
        that produced no process is not a completion of any kind. ``RecordStore.set_current_child``
        states the counterpart directly: "Pass ``None`` when a spawn fails". Retiring instead would
        publish a calibration that reached ``done`` without ever measuring anything, and section
        2.2 requires calibration failure to be loud, not silently successful.

        Idempotent and self-guarding, because two paths call it: the ``except`` in
        :func:`launch_subactivity` (a failure between registration and the body -- a Windows handle
        duplication that raised, a launch-compose conflict) and :meth:`_settle_child`. It refuses to
        touch a child that really did start; dropping a live child here would throw away the only
        handle the root has for terminating it before finalizing (gate 5).
        """
        child = self._children.get(run_id)
        if child is None or not self._child_never_started(child):
            return
        log.warning("subactivity %s never started: no process handle, no declared exit and no "
                    "record of its own; clearing the pointer rather than reporting a completion",
                    run_id)
        self._children.pop(run_id, None)
        if self._current_child is not None and self._current_child.run_id == run_id:
            self._current_child = None
            self.record = dataclasses.replace(self.record, current_child=None)
            try:
                self.store.set_current_child(None)
            except Exception:  # noqa: BLE001 - this runs on a failure path; never mask that error
                log.exception("could not clear current_child for the failed spawn %s", run_id)
        # The one-use grant was minted for a child that will never claim it. It already fails
        # closed after DEFAULT_GRANT_TTL_S, but section 3.2's retention rule makes pruning the
        # root's job rather than letting children/ collect one orphan per failed spawn.
        with contextlib.suppress(Exception):
            grant_path(self.deployment_dir, run_id).unlink(missing_ok=True)

    def _retire_child(self, child: _LiveChild) -> None:
        """Move a finished child out of ``current_child``, once it is provably finished.

        Three cases, and they are not the same:

        * no handle and its own record still says live -- we cannot prove it is gone, so the
          summary STAYS and finalize refuses. ``active.json`` is never removed under a live child
          (section 3.3, gate 5).
        * a handle that says the process is gone, but a live record -- it was killed before it
          could finalize its own record. We hold the handle, so "gone" is a fact and the record is
          merely stale: drop the pointer so the root can finalize, and say so in the log. Keeping
          it would wedge the root forever behind a child that no longer exists.
        * the ordinary case -- promote it to ``last_child``, which is what the GUI shows until the
          next child appears.
        """
        run_id = child.summary.run_id
        stale = self._child_record_is_live(run_id)
        if stale and child.proc is None:
            log.warning("subactivity %s has no process handle and its record is still live", run_id)
            return
        is_current = self._current_child is not None and self._current_child.run_id == run_id
        if stale:
            log.warning(
                "subactivity %s exited without finalizing its own record; dropping the stale "
                "pointer so the root can finalize", run_id,
            )
            if is_current:
                self.store.set_current_child(None)
                self._current_child = None
                self.record = dataclasses.replace(self.record, current_child=None)
        elif is_current:
            state = RunState.DONE if (child.returncode or 0) == 0 else RunState.ERROR
            finished = dataclasses.replace(child.summary, state=state)
            self._current_child = finished
            self.store.set_current_child(finished)
            self.store.promote_current_child()
            self._last_child = finished
            self._current_child = None
            self.record = dataclasses.replace(self.record, current_child=None, last_child=finished)
        self._children.pop(run_id, None)

    def _child_record_is_live(self, run_id: str) -> bool:
        try:
            record = self.store.read_child(run_id)
        except Exception:  # noqa: BLE001 - an unreadable child record is not evidence of life
            return False
        return record is not None and record.state in LIVE_STATES

    def _stop_children(self, *, graceful: bool) -> None:
        """Join or terminate every subactivity, BEFORE finalizing (section 3.3).

        ``graceful`` on the normal path: the child is expected to be finishing, so it gets the join
        timeout before any signal. On the exception and interrupt paths it is terminated at once --
        the root is going away, and a calibration tree must stop as one unit rather than orphaning
        a child that would keep the deployment occupied.
        """
        for child in list(self._children.values()):
            proc = child.proc
            if proc is None or not _proc_alive(proc):
                continue
            run_id = child.summary.run_id
            try:
                if graceful:
                    log.info("waiting for subactivity %s to finish before finalizing", run_id)
                    with contextlib.suppress(Exception):
                        child.returncode = proc.wait(timeout=self._join_timeout)
                if _proc_alive(proc):
                    log.warning("terminating subactivity %s so the root can finalize", run_id)
                    proc.terminate()
                    with contextlib.suppress(Exception):
                        child.returncode = proc.wait(timeout=DEFAULT_CHILD_KILL_TIMEOUT_S)
                if _proc_alive(proc):
                    log.error("subactivity %s ignored terminate; killing it", run_id)
                    proc.kill()
                    with contextlib.suppress(Exception):
                        child.returncode = proc.wait(timeout=DEFAULT_CHILD_KILL_TIMEOUT_S)
            except Exception:  # noqa: BLE001 - a stubborn child must not hide the run's outcome
                log.exception("could not stop subactivity %s", run_id)

    def _finalize(self) -> None:
        self._stop_children(graceful=self._outcome is RunState.DONE)
        for child in list(self._children.values()):
            if child.proc is not None and not _proc_alive(child.proc):
                self._retire_child(child)
        record = dataclasses.replace(
            self._terminal_record(), current_child=self._current_child, last_child=self._last_child
        )
        self.store.finalize(record, live_children=self.live_children())
        self.record = record


class ChildActivity(_RecordedActivity):
    """An approved subactivity's handle: it writes ``children/<run_id>.json`` and nothing else.

    It may finish its own record after its root has died, but it must never modify or remove the
    root's (section 3.2). It also never removes ``active.json`` -- that file belongs to the root
    even when the root is gone, because it is the only description of what still holds the lease.
    """

    def _publish(self, record: RuntimeRecord) -> None:
        self.store.write_child(record)
        self.record = record

    def _finalize(self) -> None:
        record = self._terminal_record()
        self.store.write_child(record)
        self.record = record


# --------------------------------------------------------------------------------------------- #
# Root activities (plan section 3.3)
# --------------------------------------------------------------------------------------------- #

@contextmanager
def root_activity(
    activity: Activity,
    *,
    cfg: Any,
    config_path: str | os.PathLike[str] | None = None,
    launcher: Launcher | str = Launcher.PYTHON,
    run_id: str | None = None,
    env: Mapping[str, str] | None = None,
    status_fd: int | None = None,
    announce: bool = True,
    deployment_dir: Path | None = None,
    deployment_key: str | None = None,
    settings_file: str | os.PathLike[str] | None = None,
    early: paths.EarlyRunPaths | None = None,
    project: ProjectBlock | None = None,
    hardware_destination: str | os.PathLike[str] | None = None,
    control: ControlBlock | None = None,
    enabled: bool | None = None,
    handle_sigterm: bool = False,
    children_join_timeout: float = DEFAULT_CHILD_JOIN_TIMEOUT_S,
    platform_name: str | None = None,
    adapter: Any = None,
) -> Iterator[ActivityHandle]:
    """Own one ROOT activity for the duration of the body, in section 3.3's order.

    Call it from the ONE place in each entry point where cheap config loading and validation have
    finished and nothing expensive has started -- in ``machine3()`` that is between ``cfg.validate()``
    and ``ensure_hardware_profile(cfg)``.

    On contention it raises :class:`~leafmachine3.core.runtime._types.RuntimeBusyError` carrying the
    sanitized winner, after writing the section 2.4 ``busy`` status line if this process was
    launched with a status channel. The CLI turns that into exit code
    :data:`~leafmachine3.core.runtime._types.EXIT_CODE_BUSY` (75); the server turns the status line
    into a 409 with the winner's identity.

    ``handle_sigterm`` is opt-in because installing a signal handler is a process-wide act. When it
    is on, a ``SIGTERM`` becomes a ``KeyboardInterrupt`` inside the body and the run finalizes as
    ``stopped`` rather than ``interrupted`` -- which is what a server Stop means.
    """
    launcher = _as_launcher(launcher)
    if enabled is None:
        enabled = runtime_v2_enabled(env)
    if not enabled:
        # Flag off: acquire nothing, publish nothing, create nothing. The body still runs.
        yield DisabledActivity(activity=activity, launcher=launcher)
        return
    if activity not in ROOT_ACTIVITIES:
        raise ValueError(
            f"{activity.value!r} is not a root activity; an inherited subactivity uses "
            f"child_activity() and never acquires a lease of its own (invariant 2)"
        )

    values = dict(os.environ if env is None else env)
    key = deployment_key or paths.deployment_key(values)
    dep_dir = Path(deployment_dir) if deployment_dir is not None else paths.deployment_runtime_dir(
        env=values, deployment=key
    )
    channel = status_channel(values, fd=status_fd) if announce else None

    # -- cheap identity, resolved from the ALREADY-VALIDATED config. Pure: nothing is created. ---- #
    try:
        config_ref = config_io.config_ref(cfg, path=config_path or settings_file)
        project_block, early_paths, hardware = _identity_blocks(
            activity, cfg, project=project, early=early, settings_file=settings_file or config_path,
            hardware_destination=hardware_destination, env=values, deployment=key,
        )
    except BaseException:
        if channel is not None:
            channel.close()
        raise

    # -- the lease. AFTER validation, BEFORE hardware profiling, orphan reaping, directory
    #    creation, database writes, model loading or GPU work (section 3.3). ---------------------- #
    try:
        lease = acquire_root_lease(
            activity, deployment_dir=dep_dir, deployment_key=key, env=values,
            platform_name=platform_name, adapter=adapter,
        )
    except RuntimeBusyError as busy:
        if channel is not None:
            channel.send_busy(busy.active)
            channel.close()
        raise
    except BaseException:
        if channel is not None:
            channel.close()
        raise

    # -- the recovery transaction (sections 2.9 and 3.2). The lease is now HELD, which is the whole
    #    proof that any `active.json` still sitting here belongs to a root that is gone: no live root
    #    -- and no child holding an inherited reference -- could have let us acquire. So before this
    #    process begins its OWN root activity we temporarily become the recovery writer, exactly as
    #    section 3.2's "one documented exception" allows: a stale non-terminal record is finalized
    #    into `last.json` as abandoned (so a hard-killed run reaches history instead of vanishing),
    #    and a record we may not interpret -- a newer schema_version, or malformed JSON -- is
    #    QUARANTINED beside itself rather than overwritten. Without this, `write_active` below would
    #    silently destroy another build's evidence, which is precisely what section 2.9 forbids
    #    (`_check_transition` cannot see it: a foreign run_id and an incompatible schema both read
    #    back as "no prior state").
    #
    #    We do NOT open a second `cleanup_lease`: the root already holds the same activity lock, and
    #    that is the ownership `recover_abandoned` checks. Failure is fatal by design -- section 3.2
    #    makes completing the cleanup transaction a precondition of starting -- so we hand the
    #    deployment straight back rather than starting on top of a record we could not resolve.
    try:
        recover_abandoned(dep_dir, lease=lease)
    except BaseException:
        if channel is not None:
            channel.close()
        with contextlib.suppress(Exception):
            lease.release()
        raise

    run_id = run_id or new_run_id()
    now = utc_now()
    record = RuntimeRecord(
        run_id=run_id,
        activity=activity,
        activity_role=ActivityRole.ROOT,
        state=RunState.STARTING,
        launcher=launcher,
        pid=os.getpid(),
        process_started_at=process_start_time(),
        started_at=now,
        updated_at=now,
        deployment=build_deployment_info(values, deployment_key=key),
        config=config_ref,
        project=project_block,
        hardware=hardware,
        control=control or ControlBlock(mode=ControlMode.UNMANAGED),
    )
    store = RecordStore(dep_dir, run_id=run_id, role=ActivityRole.ROOT)
    handle = RootActivity(
        run_id=run_id, activity=activity, launcher=launcher, lease=lease, store=store,
        record=record, deployment_dir=dep_dir, deployment_key=key, cfg=cfg, early=early_paths,
        config=config_ref, settings_file=settings_file or config_path, channel=channel,
        children_join_timeout=children_join_timeout,
    )
    try:
        handle._publish(record)
    except BaseException:
        # Nothing has been announced and nothing else has been created, so the honest thing is to
        # give the deployment straight back.
        if channel is not None:
            channel.close()
        with contextlib.suppress(Exception):
            lease.release()
        raise

    if channel is not None:
        channel.send_acquired(run_id=run_id, project=project_block)
        channel.close()

    trap = _SigtermTrap() if handle_sigterm else None
    try:
        if trap is not None:
            trap.install()
        yield handle
    except KeyboardInterrupt:
        handle._record_outcome(RunState.STOPPED if (trap and trap.fired) else RunState.INTERRUPTED)
        raise
    except BaseException as exc:
        handle._record_outcome(RunState.ERROR, error=exc)
        raise
    else:
        handle._record_outcome(None)
    finally:
        if trap is not None:
            trap.restore()
        handle._close()


def _identity_blocks(
    activity: Activity,
    cfg: Any,
    *,
    project: ProjectBlock | None,
    early: paths.EarlyRunPaths | None,
    settings_file: str | os.PathLike[str] | None,
    hardware_destination: str | os.PathLike[str] | None,
    env: Mapping[str, str],
    deployment: str,
) -> tuple[ProjectBlock | None, paths.EarlyRunPaths | None, HardwareBlock | None]:
    """The per-activity blocks section 3.2's table requires, resolved without touching the disk.

    A ``hardware_setup`` root deliberately carries NO project (invariant 6): tuning the machine is
    not work on anybody's specimens, and a project block there would make the GUI switch project
    history to ``_lm3_calibration``.
    """
    if activity is Activity.HARDWARE_SETUP:
        destination = hardware_destination or paths.hardware_profile_path(env=env, deployment=deployment)
        return None, None, HardwareBlock(destination_path=str(Path(destination).expanduser().resolve()))
    early_paths = early if early is not None else config_io.resolve_run_paths(
        cfg, settings_file=settings_file
    )
    block = project if project is not None else config_io.project_block(cfg, settings_file=settings_file)
    return block, early_paths, None


# --------------------------------------------------------------------------------------------- #
# Subactivity launch (plan section 2.2) -- the PARENT half
# --------------------------------------------------------------------------------------------- #

@dataclass
class SubactivityLaunch:
    """Everything one approved child launch needs, already composed. Pass it to ONE ``Popen``.

    ``env`` and ``popen_kwargs`` come from :func:`~leafmachine3.core.runtime.launch.compose_launch`
    and must be used together and unmodified: the lease reference and the status pipe both inject
    into ``pass_fds`` / the Windows handle allowlist, both of which are exhaustive, so anything that
    rebuilds them by hand silently drops one.
    """

    run_id: str
    capability: str
    env: dict[str, str]
    popen_kwargs: dict[str, Any]
    inherited: tuple[int, ...] = ()
    _root: "RootActivity | None" = field(default=None, repr=False, compare=False)
    _child: _LiveChild | None = field(default=None, repr=False, compare=False)

    def attach(self, proc: Any) -> Any:
        """Hand the root the live process handle, so it can join or terminate this child.

        Optional only for a caller that blocks on the child itself (``subprocess.run``). Without it
        the root has nothing to join, and a child that outlives the body will keep the deployment
        occupied with no way for the root to stop it.
        """
        if self._child is not None:
            self._child.proc = proc
        return proc

    def completed(self, *, returncode: int | None = None) -> None:
        """Declare that the child has exited, so the root can promote it to ``last_child``."""
        if self._child is not None:
            self._child.completed = True
            self._child.returncode = returncode


@contextmanager
def launch_subactivity(
    root: ActivityHandle,
    activity: Activity = Activity.CALIBRATION_PIPELINE,
    *,
    run_id: str | None = None,
    run_name: str | None = None,
    run_dir: str | os.PathLike[str] | None = None,
    ttl_s: float = DEFAULT_GRANT_TTL_S,
    env: Mapping[str, str] | None = None,
    base_kwargs: Mapping[str, Any] | None = None,
    contributions: Sequence[LaunchContribution] = (),
    clock: Callable[[], float] | None = None,
) -> Iterator[SubactivityLaunch]:
    """Mint the grant, open the lease handoff, and compose ONE child launch (section 2.2).

    Usage, and the ``attach`` is not optional if the caller does not block::

        with root.launch_subactivity(run_name=CALIBRATION_RUN_NAME) as launch:
            proc = subprocess.Popen(argv, env=launch.env, **launch.popen_kwargs)
            launch.attach(proc)
            launch.completed(returncode=proc.wait())

    The handoff is a context manager because its ``close`` must run on every path including every
    failure path: on Windows a failed spawn otherwise leaks an inheritable duplicate that nothing
    will ever release, and the deployment stays occupied forever.
    """
    if not root.enabled:
        with root.launch_subactivity(env=env, base_kwargs=base_kwargs) as inert:  # type: ignore[union-attr]
            yield inert
        return
    if not isinstance(root, RootActivity):
        raise TypeError("only a root activity may launch a subactivity")
    if activity not in CHILD_ACTIVITIES:
        raise ValueError(f"{activity.value!r} is not an inherited subactivity")
    expected_parent = CHILD_PARENT_ACTIVITY[activity]
    if root.activity is not expected_parent:
        raise ValueError(
            f"{activity.value} runs under a {expected_parent.value} root, not a {root.activity.value}"
        )

    child_run_id = run_id or new_run_id()
    grant, capability = issue_grant(
        root.deployment_dir,
        child_run_id=child_run_id,
        parent_run_id=root.run_id,
        deployment_id=root.record.deployment.id,
        purpose=activity,
        ttl_s=ttl_s,
        clock=clock,
    )
    write_grant(root.deployment_dir, grant)

    summary = ChildSummary(
        run_id=child_run_id,
        activity=activity,
        state=RunState.STARTING,
        started_at=utc_now(),
        run_name=run_name,
        run_dir=str(Path(run_dir).resolve()) if run_dir is not None else None,
    )
    root._register_child(summary)

    identity = LaunchContribution(
        name="subactivity",
        env={
            ENV_CHILD_RUN_ID: child_run_id,
            ENV_PARENT_RUN_ID: root.run_id,
            ENV_CHILD_ACTIVITY: activity.value,
        },
    )
    base_env = child_base_env(env)
    try:
        with root.lease.child_handoff(capability) as handoff:
            lease_contribution = LaunchContribution.from_handoff(handoff)
            with composed_launch(
                [lease_contribution, identity, *contributions],
                base_env=base_env,
                base_kwargs=base_kwargs,
            ) as composed:
                launch = SubactivityLaunch(
                    run_id=child_run_id,
                    capability=capability,
                    env=composed.env,
                    popen_kwargs=composed.popen_kwargs,
                    inherited=composed.inherited,
                    _root=root,
                    _child=root._children.get(child_run_id),
                )
                try:
                    yield launch
                finally:
                    root._settle_child(launch)
    except BaseException:
        # Anything that raised BEFORE the body ran -- a handle duplication that failed on Windows,
        # a launch-compose conflict -- would otherwise strand the summary registered above in
        # ``starting`` forever, and section 3.3 has no such resting state: the root then finalizes
        # and writes a phantom calibration into last.json that the GUI shows as permanently
        # starting. _forget_child is self-guarding, so the ordinary case (the caller's body raised,
        # _settle_child already ran, or the child is genuinely alive) passes through untouched.
        root._forget_child(child_run_id)
        raise


# --------------------------------------------------------------------------------------------- #
# Subactivity execution (plan section 2.2) -- the CHILD half
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class InheritedLease:
    """A validated, claimed, DISARMED lease reference plus the grant that authorized it."""

    lease: RuntimeLease
    grant: GrantRecord
    run_id: str
    parent_run_id: str
    activity: Activity
    deployment_dir: Path
    deployment_key: str


def bind_child_lease(
    *,
    activity: Activity | None = None,
    env: MutableMapping[str, str] | None = None,
    deployment_dir: Path | None = None,
    deployment_key: str | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    clock: Callable[[], float] | None = None,
    platform_name: str | None = None,
    win32: Any | None = None,
) -> InheritedLease:
    """Bind, validate, claim, revalidate and DISARM the lease reference this process inherited.

    The order is section 2.2's and no step may forward-reference a later one:

    1. validate everything -- the inherited reference itself included -- before touching the grant,
       so an invalid child leaves the legitimate child's grant exactly where it found it;
    2. claim by atomic rename, which is what decides the single winner;
    3. re-read the claimed file and revalidate, because the claim itself can be delayed;
    4. disarm inheritance (``os.set_inheritable(fd, False)`` / ``SetHandleInformation(h, ..., 0)``)
       and clear ``LM3_LEASE_FD`` / ``LM3_LEASE_EVENT_HANDLE`` / ``LM3_LEASE_CAPABILITY`` from the
       environment BEFORE any expensive work, so no later fork can carry the reference and no
       executor worker can hold the deployment open (invariant 3, gates 7 and 32).

    Steps 1b-4 are :func:`~leafmachine3.core.runtime.grant.redeem_grant`, which owns that ordering.
    """
    values = env if env is not None else os.environ
    capability = (values.get(ENV_LEASE_CAPABILITY) or "").strip()
    if not capability:
        raise LeaseInheritanceError(
            f"{ENV_LEASE_CAPABILITY} is not set: this process was not launched as an approved "
            f"subactivity, so it has no lease reference to bind"
        )
    child_run_id = run_id or (values.get(ENV_CHILD_RUN_ID) or "").strip()
    parent = parent_run_id or (values.get(ENV_PARENT_RUN_ID) or "").strip()
    if not child_run_id or not parent:
        raise LeaseInheritanceError(
            f"an approved subactivity needs {ENV_CHILD_RUN_ID} and {ENV_PARENT_RUN_ID} to find its "
            f"own grant; got {child_run_id!r} and {parent!r}"
        )
    purpose = activity
    if purpose is None:
        declared = (values.get(ENV_CHILD_ACTIVITY) or "").strip()
        purpose = Activity(declared) if declared else Activity.CALIBRATION_PIPELINE
    key = deployment_key or paths.deployment_key(values)
    dep_dir = Path(deployment_dir) if deployment_dir is not None else paths.deployment_runtime_dir(
        env=values, deployment=key
    )
    lease = inherit_lease(deployment_dir=dep_dir, deployment_key=key, env=values,
                          platform_name=platform_name, win32=win32)
    grant = redeem_grant(
        dep_dir,
        child_run_id=child_run_id,
        parent_run_id=parent,
        deployment_id=key,
        purpose=purpose,
        capability=capability,
        lease=lease,
        clock=clock,
        env=values,
    )
    # The identity variables are ours, and they have done their job. Clearing them keeps a nested
    # spawn from looking like an approved child of a grant that no longer exists.
    for name in CHILD_ENV_VARS:
        values.pop(name, None)
    return InheritedLease(
        lease=lease, grant=grant, run_id=child_run_id, parent_run_id=parent, activity=purpose,
        deployment_dir=dep_dir, deployment_key=key,
    )


@contextmanager
def child_activity(
    activity: Activity = Activity.CALIBRATION_PIPELINE,
    *,
    cfg: Any,
    config_path: str | os.PathLike[str] | None = None,
    launcher: Launcher | str = Launcher.CLI,
    env: MutableMapping[str, str] | None = None,
    status_fd: int | None = None,
    announce: bool = True,
    deployment_dir: Path | None = None,
    deployment_key: str | None = None,
    settings_file: str | os.PathLike[str] | None = None,
    early: paths.EarlyRunPaths | None = None,
    project: ProjectBlock | None = None,
    enabled: bool | None = None,
    handle_sigterm: bool = True,
    inherited: InheritedLease | None = None,
    platform_name: str | None = None,
) -> Iterator[ActivityHandle]:
    """Own one INHERITED subactivity for the duration of the body.

    It never acquires: it binds its parent's reference, and the deployment stays occupied for as
    long as this process lives even if the parent is killed (invariant 2). It writes only
    ``children/<run_id>.json`` -- never ``active.json``, which belongs to the root even after the
    root has died, because it is the only description of what still holds the lease.

    ``handle_sigterm`` defaults to True here and to False for a root, and the asymmetry is
    deliberate: a subactivity is normally ended by its parent terminating it (section 3.3, "Stop
    targets the retained root process group"), and a child that dies without finalizing its own
    record leaves a record that says ``running`` forever -- which is then the thing blocking the
    root's own finalization. Handling SIGTERM is how the common stop path stays clean.
    """
    launcher = _as_launcher(launcher)
    if enabled is None:
        enabled = runtime_v2_enabled(env)
    if not enabled:
        yield DisabledActivity(activity=activity, launcher=launcher)
        return

    values = env if env is not None else os.environ
    channel = status_channel(values, fd=status_fd) if announce else None
    try:
        bound = inherited or bind_child_lease(
            activity=activity, env=values, deployment_dir=deployment_dir,
            deployment_key=deployment_key, platform_name=platform_name,
        )
        config_ref = config_io.config_ref(cfg, path=config_path or settings_file)
        early_paths = early if early is not None else config_io.resolve_run_paths(
            cfg, settings_file=settings_file or config_path
        )
        block = project if project is not None else config_io.project_block(
            cfg, settings_file=settings_file or config_path
        )
    except BaseException:
        if channel is not None:
            channel.close()
        raise

    now = utc_now()
    record = RuntimeRecord(
        run_id=bound.run_id,
        activity=bound.activity,
        activity_role=ActivityRole.CHILD,
        state=RunState.STARTING,
        launcher=launcher,
        pid=os.getpid(),
        process_started_at=process_start_time(),
        started_at=now,
        updated_at=now,
        deployment=build_deployment_info(values, deployment_key=bound.deployment_key),
        parent_run_id=bound.parent_run_id,
        config=config_ref,
        project=block,
    )
    store = RecordStore(bound.deployment_dir, run_id=bound.run_id, role=ActivityRole.CHILD)
    handle = ChildActivity(
        run_id=bound.run_id, activity=bound.activity, launcher=launcher, lease=bound.lease,
        store=store, record=record, deployment_dir=bound.deployment_dir,
        deployment_key=bound.deployment_key, cfg=cfg, early=early_paths, config=config_ref,
        settings_file=settings_file or config_path, channel=channel,
    )
    try:
        handle._publish(record)
    except BaseException:
        if channel is not None:
            channel.close()
        with contextlib.suppress(Exception):
            bound.lease.release()
        raise

    if channel is not None:
        channel.send_acquired(run_id=bound.run_id, project=block)
        channel.close()

    trap = _SigtermTrap() if handle_sigterm else None
    try:
        if trap is not None:
            trap.install()
        yield handle
    except KeyboardInterrupt:
        handle._record_outcome(RunState.STOPPED if (trap and trap.fired) else RunState.INTERRUPTED)
        raise
    except BaseException as exc:
        handle._record_outcome(RunState.ERROR, error=exc)
        raise
    else:
        handle._record_outcome(None)
    finally:
        if trap is not None:
            trap.restore()
        handle._close()


# --------------------------------------------------------------------------------------------- #
# The one entry point an execution path actually calls
# --------------------------------------------------------------------------------------------- #

@contextmanager
def execution_activity(
    root: Activity = Activity.PIPELINE,
    *,
    child: Activity = Activity.CALIBRATION_PIPELINE,
    env: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> Iterator[ActivityHandle]:
    """Own a root activity, or -- when this process is an approved child -- an inherited one.

    This is what ``machine3()`` calls, because ``python -m leafmachine3.machine3`` is BOTH the
    ordinary CLI entry point and the calibration child. The choice is made by
    :func:`is_approved_child`, i.e. by the presence of a capability in the environment, and never by
    an argument a caller could get wrong.
    """
    values = os.environ if env is None else env
    if is_approved_child(values):
        with child_activity(child, env=values, **kwargs) as handle:  # type: ignore[arg-type]
            yield handle
        return
    with root_activity(root, env=values, **kwargs) as handle:
        yield handle


# --------------------------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------------------------- #

class _SigtermTrap:
    """Turn ``SIGTERM`` into ``KeyboardInterrupt`` for the duration of an activity.

    Opt-in, because installing a handler is process-wide and only works in the main thread of the
    main interpreter. When it fires the run finalizes as ``stopped`` rather than ``interrupted``:
    a server Stop and a user's Ctrl-C are different events and history should say which happened.
    """

    def __init__(self) -> None:
        self.fired = False
        self._previous: Any = None
        self._installed = False

    def install(self) -> None:
        def _handler(signum: int, frame: Any) -> None:  # noqa: ARG001 - signal handler signature
            self.fired = True
            raise KeyboardInterrupt(f"terminated by signal {signum}")

        try:
            self._previous = signal.signal(signal.SIGTERM, _handler)
            self._installed = True
        except (ValueError, OSError, AttributeError) as exc:
            # Not the main thread, or a platform without SIGTERM. Not fatal: the run simply keeps
            # the default disposition.
            log.debug("could not install a SIGTERM handler: %s", exc)

    def restore(self) -> None:
        if not self._installed:
            return
        with contextlib.suppress(ValueError, OSError, TypeError):
            signal.signal(signal.SIGTERM, self._previous)
        self._installed = False


def _proc_alive(proc: Any) -> bool:
    """Is this child process handle still live? Handle ownership only -- never a PID lookup."""
    try:
        return proc.poll() is None
    except Exception:  # noqa: BLE001 - a handle we cannot poll is not evidence of life
        return False


def _error_text(error: BaseException | str | None) -> str | None:
    if error is None:
        return None
    text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return text[:MAX_ERROR_CHARS]


def _as_launcher(value: Launcher | str) -> Launcher:
    return value if isinstance(value, Launcher) else Launcher(str(value))
