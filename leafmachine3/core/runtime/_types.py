"""Shared vocabulary for the LM3 runtime registry -- types only, no behavior.

Everything in this module is a *contract*: the record fields written to ``active.json`` /
``children/<id>.json`` / ``last.json``, the taxonomy of activities and states, the two enums that
carry real archive semantics, the grant schema, the lease adapter surface, and the exception
hierarchy. The four implementation modules (``lease``, ``records``, ``grant``, ``config_io``) may
add helpers, but they may not redefine any name here: a second spelling of ``archive_status`` or a
second ``RuntimeBusyError`` is exactly the class of drift the unified runtime exists to remove.

Three decisions worth reading before touching anything:

* ``abandoned`` is a CLASSIFICATION, never a written ``state`` (plan section 3.3). A root that is
  hard-killed leaves a ``running`` record behind and the next reader labels it -- see
  :class:`RecordClassification`. ``RunState`` deliberately has no ``abandoned`` member so a writer
  cannot persist one, and :data:`_ABANDONED_IS_NOT_A_STATE` pins that at import.
* ``archive_status`` uses ``"n/a"`` -- with a slash. Revision 8 (G3) standardized on it and both
  the Python readers and the GUI compare the literal string, so ``"n-a"`` is a bug, not a synonym.
* Every Windows call goes through :class:`Win32Surface`, which is injected. That is what makes the
  Windows lease adapter unit-testable on Linux (working rules; plan section 7 wants both adapters
  covered, and CI cannot run the real ``CreateEventW`` from a Linux runner).

The wire/disk shapes are given twice on purpose: as frozen dataclasses for typed in-process use,
and as ``TypedDict``s naming the exact JSON keys. ``records.record_to_dict`` /
``records.record_from_dict`` are the only sanctioned bridge between the two.
"""
from __future__ import annotations

import enum
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, TypedDict, TypeVar

from .. import paths

# --------------------------------------------------------------------------------------------- #
# Schema and process-level constants
# --------------------------------------------------------------------------------------------- #

#: Version stamped into every record this build writes (plan section 3.2). A reader finding a
#: HIGHER value must not interpret unknown fields; see :class:`RecordClassification.INCOMPATIBLE`
#: and section 2.9.
SCHEMA_VERSION = 1

#: The stable busy exit code (plan sections 2.3 and 3.3). A losing root exits with exactly this,
#: and the shell wrapper distinguish it from an ordinary pipeline failure.
EXIT_CODE_BUSY = 75

#: The HTTP equivalent of :data:`EXIT_CODE_BUSY` (plan section 2.3): the launch handshake turns a
#: child's ``busy`` status line into this response code.
HTTP_STATUS_BUSY = 409

#: Child records and consumed grants would otherwise accumulate forever (plan section 3.2,
#: "Retention"). Default: keep the last 50 children per deployment.
DEFAULT_CHILD_RETENTION = 50

#: Grant lifetime. The plan's example grant is issued at ``1787932800.25`` and expires at
#: ``1787932860.25`` -- exactly 60 seconds (section 2.2).
DEFAULT_GRANT_TTL_S = 60.0

#: The reserved run name of a calibration child (plan section 3.2 table). The GUI keys off it so
#: hardware setup never switches project history to it (section 2.6).
CALIBRATION_RUN_NAME = "_lm3_calibration"

# -- registry filenames ------------------------------------------------------------------------ #
# Re-exported from core.paths rather than re-spelled: one definition, so a rename cannot leave the
# runtime package reading a file the resolver stopped writing.
ACTIVITY_LOCK_FILENAME = paths.ACTIVITY_LOCK_FILENAME
ACTIVE_RECORD_FILENAME = paths.ACTIVE_RECORD_FILENAME
LAST_RECORD_FILENAME = paths.LAST_RECORD_FILENAME
CHILDREN_DIRNAME = paths.CHILDREN_DIRNAME
CONNECTION_PRIVATE_FILENAME = paths.CONNECTION_PRIVATE_FILENAME
CONNECTION_PUBLIC_FILENAME = paths.CONNECTION_PUBLIC_FILENAME

#: ``children/<child_run_id>.json`` -- the full child lifecycle record, written only by that child.
CHILD_RECORD_SUFFIX = ".json"
#: ``children/<child_run_id>.grant.json`` -- written by the ROOT, never edited by the child.
GRANT_SUFFIX = ".grant.json"
#: ``children/<child_run_id>.grant.consumed.json`` -- the claim is the RENAME onto this name
#: (plan section 2.2). Exactly one claimant can win it; a replay finds no source and fails closed.
GRANT_CONSUMED_SUFFIX = ".grant.consumed.json"

# -- archive artifacts (plan section 2.10) ------------------------------------------------------ #

#: Staged mode only. Lives in ``artifact_dir`` so it survives the allocation.
ARCHIVE_POINTER_FILENAME = "archive.current.json"
#: ``<run_id>.<sequence>`` -- keyed on the run's UUID, not its name, so a RESUMED invocation of the
#: same project cannot reuse a generation filename and overwrite an earlier run's archive.
ARCHIVE_GENERATION_TEMPLATE = "archive.{run_id}.{sequence}.sqlite"
ARCHIVE_TMP_SUFFIX = ".tmp"

# -- environment plumbing ----------------------------------------------------------------------- #
# The lease reference and the one-use capability travel in the approved child's environment. The
# child clears all three BEFORE it spawns any executor worker (invariant 3, plan section 2.2).

ENV_LEASE_FD = "LM3_LEASE_FD"                        # POSIX: the inherited lock descriptor number
ENV_LEASE_EVENT_HANDLE = "LM3_LEASE_EVENT_HANDLE"    # Windows: the inherited lease-event handle
ENV_LEASE_CAPABILITY = "LM3_LEASE_CAPABILITY"        # the raw one-use capability value

#: Everything a child must delete from its environment once it has validated and disarmed its lease
#: reference. Clearing is unconditional on both platforms: a POSIX child that finds a stray
#: ``LM3_LEASE_EVENT_HANDLE`` should still drop it rather than pass it on.
LEASE_ENV_VARS: tuple[str, ...] = (ENV_LEASE_FD, ENV_LEASE_EVENT_HANDLE, ENV_LEASE_CAPABILITY)

# The launch handshake (plan section 2.4). Named here because the lease attempt and the status line
# are two halves of one protocol; the server side is wired in Step 3.
ENV_STATUS_FD = "LM3_STATUS_FD"                      # POSIX: write end of the anonymous pipe
ENV_STATUS_HANDLE = "LM3_STATUS_HANDLE"              # Windows: inheritable pipe handle value
#: Bounded, because "oversized status" is an explicit failure branch the server must handle.
MAX_HANDSHAKE_BYTES = 64 * 1024
#: Order of seconds -- the lease attempt happens before any cold torch import, so this is bounded by
#: config load, not by model loading. Configurable; this is only the default.
DEFAULT_HANDSHAKE_TIMEOUT_S = 30.0

# -- record bounds and redaction (plan section 3.2 rules; section 7 "record redaction and field
#    bounds") ------------------------------------------------------------------------------------ #

MAX_RECORD_BYTES = 64 * 1024        # a record is a bounded description, never a log
MAX_STRING_FIELD_CHARS = 4096       # any single string value
MAX_ERROR_CHARS = 2048              # ``error`` / ``archive_error``
MAX_INPUT_DIRS = 64                 # ``project.input_dirs``
REDACTION_PLACEHOLDER = "[redacted]"

#: Substring match, lowercased, on both keys and dotted paths. "No bearer tokens, environment
#: dumps, or other secrets" (section 3.2) is only testable if the forbidden shape is named.
SECRET_KEY_SUBSTRINGS: frozenset[str] = frozenset(
    {"token", "secret", "password", "passwd", "authorization", "bearer", "api_key", "apikey",
     "credential", "cookie", "session_key", "private_key"}
)

# -- Windows lease naming (plan section 2.2) ---------------------------------------------------- #

#: ``Global\``, never ``Local\``: a Local event is scoped to one logon session, which would leave
#: the dead-root-with-live-child window unguarded against a contender in another session.
WINDOWS_LEASE_EVENT_TEMPLATE = r"Global\lm3-lease-{sid_hash}-{deployment_key}"
#: sha256 of the SID in canonical ``ConvertSidToStringSidW`` form (``S-1-5-21-...``), UTF-8, cut to
#: 16 hex characters -- the same bounded-hash convention as the deployment and machine keys.
SID_HASH_LENGTH = 16

# Win32 constants the adapter needs. Defined here so the fake surface and the real ctypes surface
# agree on the numbers, and so Linux can import this module without touching ``ctypes.windll``.
ERROR_ALREADY_EXISTS = 183
EVENT_ALL_ACCESS = 0x1F0003
EVENT_MODIFY_STATE = 0x0002
HANDLE_FLAG_INHERIT = 0x00000001


# --------------------------------------------------------------------------------------------- #
# Activity taxonomy (plan sections 2.2 and 3.2)
# --------------------------------------------------------------------------------------------- #

class Activity(str, enum.Enum):
    """What is running. Two roots, one inherited subactivity -- no others exist.

    There was a third root, ``batch``, with a ``batch_item_pipeline`` child. Plan revision 14
    removed both: the Global Greening wrapper is a shell loop that invokes ordinary LM3 runs, so
    each species takes an ordinary ``pipeline`` lease and the wrapper owns only sequencing. Lease
    inheritance is therefore exercised by calibration alone -- which genuinely needs it, because a
    calibration child runs nested LM3 work while ``hardware_setup`` owns the deployment.
    """

    PIPELINE = "pipeline"
    HARDWARE_SETUP = "hardware_setup"
    CALIBRATION_PIPELINE = "calibration_pipeline"


class ActivityRole(str, enum.Enum):
    """Only ``ROOT`` records correspond to a lock acquisition (plan section 3.2).

    A ``CHILD`` record describes a process holding an INHERITED lease reference: it never acquires,
    replaces, or unlocks the lease (invariant 2).
    """

    ROOT = "root"
    CHILD = "child"


ROOT_ACTIVITIES: frozenset[Activity] = frozenset(
    {Activity.PIPELINE, Activity.HARDWARE_SETUP}
)
CHILD_ACTIVITIES: frozenset[Activity] = frozenset({Activity.CALIBRATION_PIPELINE})

#: The role an activity is REQUIRED to declare. ``activity_role`` is not free-form: a
#: ``calibration_pipeline`` claiming ``root`` is a schema error, because it would imply a lock
#: acquisition that never happened.
ACTIVITY_ROLE: dict[Activity, ActivityRole] = {
    Activity.PIPELINE: ActivityRole.ROOT,
    Activity.HARDWARE_SETUP: ActivityRole.ROOT,
    Activity.CALIBRATION_PIPELINE: ActivityRole.CHILD,
}

#: The parent activity each subactivity may legitimately run under. A ``calibration_pipeline`` whose
#: grant names any other parent is forged or mis-ordered, and fails closed.
CHILD_PARENT_ACTIVITY: dict[Activity, Activity] = {
    Activity.CALIBRATION_PIPELINE: Activity.HARDWARE_SETUP,
}


class Launcher(str, enum.Enum):
    """Descriptive only -- NEVER an authorization decision (plan section 3.2).

    Control authority comes from section 2.5's five-way match, not from a string in a JSON file.
    """

    CLI = "cli"
    PYTHON = "python"
    SERVER = "server"
    LEGACY_JOB = "legacy-job"


# --------------------------------------------------------------------------------------------- #
# Lifecycle (plan section 3.3)
# --------------------------------------------------------------------------------------------- #

class RunState(str, enum.Enum):
    """``starting -> running -> done | error | stopped | interrupted``.

    There is deliberately no ``abandoned`` member: abandonment is what a READER concludes about a
    record whose lease has become acquirable, and a writer must never be able to persist it.
    """

    STARTING = "starting"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"


TERMINAL_STATES: frozenset[RunState] = frozenset(
    {RunState.DONE, RunState.ERROR, RunState.STOPPED, RunState.INTERRUPTED}
)
LIVE_STATES: frozenset[RunState] = frozenset({RunState.STARTING, RunState.RUNNING})

#: The only legal moves. ``starting`` may go terminal directly: a run can fail during config load,
#: and an interrupt can arrive before the first stage.
STATE_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.STARTING: frozenset({RunState.RUNNING}) | TERMINAL_STATES,
    RunState.RUNNING: frozenset(TERMINAL_STATES),
    RunState.DONE: frozenset(),
    RunState.ERROR: frozenset(),
    RunState.STOPPED: frozenset(),
    RunState.INTERRUPTED: frozenset(),
}


class RecordClassification(str, enum.Enum):
    """What a reader concludes about a record it found on disk.

    This is the ONLY place ``abandoned`` appears, and it is never written into ``state``.
    """

    #: Lease held, record readable and non-terminal: this describes what is actually running.
    LIVE = "live"
    #: Lease acquirable but ``active.json`` still holds a non-terminal record -- the writer died
    #: without finalizing. Classified under the cleanup lock; the recorded PID is NEVER signaled.
    ABANDONED = "abandoned"
    #: A terminal record (``last.json``, or an ``active.json`` caught between finalize and remove).
    FINALIZED = "finalized"
    #: ``schema_version`` is higher than this build's (section 2.9). Exclusivity still comes from the
    #: lock; unknown fields are not interpreted; Start and all control actions are disabled.
    INCOMPATIBLE = "incompatible"
    #: Missing, truncated, or malformed JSON. Report diagnostics, use lock state for exclusivity,
    #: never guess a PID.
    UNREADABLE = "unreadable"
    #: No record at all.
    ABSENT = "absent"


# Pinned at import: if somebody ever adds an ``ABANDONED`` member to ``RunState``, this fails loudly
# here rather than silently letting a writer persist a classification as a state.
_ABANDONED_IS_NOT_A_STATE = RecordClassification.ABANDONED.value not in {s.value for s in RunState}
assert _ABANDONED_IS_NOT_A_STATE, "abandoned is a classification, not a writable state (section 3.3)"


class HandshakeStatus(str, enum.Enum):
    """The two status lines a launched child may write (plan section 2.4). Exactly one JSON line."""

    ACQUIRED = "acquired"
    BUSY = "busy"


class ControlMode(str, enum.Enum):
    """``control.mode`` -- capability metadata, not authority (plan section 3.2).

    The plan names the field and its two keys but does not enumerate the values, so the pair below
    is the minimum section 2.5 implies: ``MANAGED`` means some server retains a live ``Popen``
    handle for this run and recorded its own ``instance_id`` as ``owner_instance_id``; ``UNMANAGED``
    means nobody does. Neither value grants anything on its own -- a stop still requires the full
    five-way match (live handle, instance_id, run_id, pid, process creation time).
    """

    MANAGED = "managed"
    UNMANAGED = "unmanaged"


# --------------------------------------------------------------------------------------------- #
# Storage roles and archive semantics (plan section 2.10)
# --------------------------------------------------------------------------------------------- #

class StorageRole(str, enum.Enum):
    """The four roles the bounded early resolver emits (plan section 4, Step 2).

    ``archived_db_path`` is deliberately NOT one of them: in-place mode it equals
    ``active_db_path``, and in staged mode it is whatever the pointer currently names -- a resolved
    value, not an emitted one, and unknowable before the first committed snapshot.
    """

    ARTIFACT_DIR = "artifact_dir"                  # desktop: run_dir; cluster: persistent storage
    ACTIVE_STATE_DIR = "active_state_dir"          # desktop: run_dir; cluster: node-local scratch
    ACTIVE_DB_PATH = "active_db_path"              # <active_state_dir>/<run>.sqlite
    ARCHIVE_POINTER_PATH = "archive_pointer_path"  # null in-place; artifact_dir/archive.current.json


class ArchiveMode(str, enum.Enum):
    """Exactly two modes, explicit in the record.

    ``IN_PLACE`` when ``artifact_dir == active_state_dir`` (desktop and any single-filesystem run):
    null pointer, ``archived_db_path == active_db_path``, no copying, no generations. ``STAGED``
    otherwise: versioned snapshots and a MANDATORY pointer.
    """

    IN_PLACE = "in-place"
    STAGED = "staged"


class ArchiveStatus(str, enum.Enum):
    """Whether a staged run has a durable snapshot yet, and how it ended.

    ``NOT_APPLICABLE`` serializes as ``"n/a"`` -- with a slash. Revision 8 (G3) standardized on that
    literal and the GUI compares it verbatim; ``"n-a"`` is a bug.

    ``PENDING -> READY`` only when step 6's directory fsync returns, so the record never advertises
    a snapshot that is not durable.
    """

    PENDING = "pending"           # staged, no snapshot committed yet; archived_db_path is null
    READY = "ready"               # at least one snapshot committed; an unreadable pointer is an error
    NOT_APPLICABLE = "n/a"        # in-place mode; archived_db_path equals active_db_path
    STALE = "stale"               # terminal: an earlier snapshot committed, the final one failed
    FAILED = "failed"             # terminal: no snapshot ever committed; archived_db_path is null


#: Terminal archive outcomes. ``last.json`` must carry a resolved ``archived_db_path`` under
#: ``STALE`` (the last good generation) and may carry ``null`` ONLY under ``FAILED`` (gate 14).
TERMINAL_ARCHIVE_STATUSES: frozenset[ArchiveStatus] = frozenset(
    {ArchiveStatus.STALE, ArchiveStatus.FAILED}
)


class CheckpointTrigger(str, enum.Enum):
    """Why a staged backup was requested (plan section 2.10, coalescing table).

    The asymmetry is the whole point: ``PERIODIC`` may coalesce into a running snapshot, while
    ``SIGNAL`` and ``FINAL`` must queue exactly one follow-up -- merging them would silently drop
    every write made after the in-flight snapshot began.
    """

    PERIODIC = "periodic"
    SIGNAL = "signal"
    FINAL = "final"


COALESCEABLE_TRIGGERS: frozenset[CheckpointTrigger] = frozenset({CheckpointTrigger.PERIODIC})


# --------------------------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------------------------- #

class RuntimeRegistryError(RuntimeError):
    """Base for every error raised by ``leafmachine3.core.runtime``."""


class RuntimeBusyError(RuntimeRegistryError):
    """The deployment's lease is already held by a root activity.

    Carries the SANITIZED current record (redacted, field-bounded) so the loser can report the
    winner's identity over the status pipe and the API can turn it into a 409 body. ``active`` is
    ``None`` when the lease is held but no readable record exists -- exclusivity comes from the
    lock, never from the JSON, so a missing or corrupt record does not make the deployment free.
    """

    #: Named constant, documented value (plan sections 2.3 and 3.3). The CLI exits with this.
    exit_code = EXIT_CODE_BUSY

    def __init__(
        self,
        deployment_key: str,
        *,
        active: Optional["RuntimeRecord"] = None,
        classification: RecordClassification = RecordClassification.LIVE,
        message: str | None = None,
    ) -> None:
        self.deployment_key = deployment_key
        self.active = active
        self.classification = classification
        detail = message or self._describe(deployment_key, active, classification)
        super().__init__(detail)

    @staticmethod
    def _describe(
        deployment_key: str,
        active: Optional["RuntimeRecord"],
        classification: RecordClassification,
    ) -> str:
        if active is None:
            return (
                f"LM3 deployment {deployment_key!r} is busy: its activity lease is held "
                f"({classification.value} record). Exit code {EXIT_CODE_BUSY}."
            )
        return (
            f"LM3 deployment {deployment_key!r} is busy: {active.activity.value} "
            f"{active.run_id} (pid {active.pid}, state {active.state.value}) holds the lease. "
            f"Exit code {EXIT_CODE_BUSY}."
        )


class LeaseError(RuntimeRegistryError):
    """A lease primitive failed for a reason other than contention.

    Includes the Windows case the plan calls out explicitly: if the ``Global\\`` event cannot be
    created, that is a STARTUP ERROR, never a silent fall back to ``Local\\``.
    """


class LeaseNotHeldError(LeaseError):
    """An operation that requires the lease was attempted without holding it."""


class LeaseInheritanceError(LeaseError):
    """The inherited lease reference is not the deployment's lease (gate 9).

    POSIX: the descriptor does not name the same inode as ``activity.lock``, or a non-blocking
    re-assert shows it is an ordinary independently opened handle. Windows:
    ``CompareObjectHandles`` says the inherited handle and ``OpenEventW`` of the canonical name are
    different kernel objects.
    """


class RecordError(RuntimeRegistryError):
    """Base for record IO and schema problems."""


class RecordSchemaError(RecordError):
    """A record is missing a required field, has a bad enum value, or violates a field bound."""


class StateTransitionError(RecordError):
    """A record was published in a state its predecessor may not move to (plan section 3.3).

    ``STATE_TRANSITIONS`` described the lifecycle from the day it was written and was consulted by
    nothing, so the machine was decorative: a root could publish ``done`` and then ``running``, and
    every reader downstream would believe the second. Step 3 is what begins publishing states from
    ``machine3()``, hardware setup and the launch handshake, so enforcement lands FIRST -- before
    the writers exist, rather than after they have all been written against an unenforced machine.
    """


class RecordCorruptError(RecordError):
    """A record file is missing, truncated, or not parseable JSON."""


class IncompatibleSchemaError(RecordError):
    """The record's ``schema_version`` is newer than this build's (plan section 2.9).

    Callers must not interpret unknown fields. The lock -- not this record -- still decides whether
    the deployment is occupied.
    """

    def __init__(self, schema_version: int, *, path: Path | None = None) -> None:
        self.schema_version = schema_version
        self.path = path
        where = f" at {path}" if path is not None else ""
        super().__init__(
            f"runtime record{where} declares schema_version {schema_version}, newer than this "
            f"build's {SCHEMA_VERSION}. This runtime was created by a newer LM3."
        )


class WriterOwnershipError(RecordError):
    """A process tried to write a file it does not own (plan section 3.2, writer ownership table).

    ``active.json`` and ``last.json`` belong to the root; ``children/<id>.json`` belongs to that
    child; the grant belongs to the root and is CLAIMED by rename, never edited. The one exception
    is the recovery writer, which holds the activity lock.
    """


class GrantError(RuntimeRegistryError):
    """Base for capability-grant failures. Every one of them fails closed."""


class GrantInvalidError(GrantError):
    """Capability hash, deployment, parent, or purpose mismatch -- or a malformed grant file."""


class GrantExpiredError(GrantError):
    """``expires_at`` has passed, at validation time or at the post-claim re-check."""


class GrantAlreadyConsumedError(GrantError):
    """The claim rename found no source file: already consumed, never issued, or replayed."""


class ArchivePointerError(RuntimeRegistryError):
    """``archive.current.json`` is missing, malformed, or stale while ``archive_status`` is ready.

    A precise error, never a guessed filename (plan section 2.10, gate 14).
    """


# --------------------------------------------------------------------------------------------- #
# Record blocks -- typed form
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DeploymentInfo:
    """Descriptive scheduler/host context. These fields NEVER authorize signaling across a host or
    container boundary (plan section 3.2)."""

    id: str                              # the canonical deployment key
    scheduler: str | None = None         # "slurm", "pbs", ... or None for a desktop deployment
    job_id: str | None = None
    step_id: str | None = None
    node: str | None = None
    container_id: str | None = None


@dataclass(frozen=True)
class ConfigRef:
    """The exact config used at launch. ``sha256`` fingerprints the BYTES actually loaded, so an
    edit mid-run is detectable and the record stays pinned to the launch manifest."""

    path: str                            # absolute, resolved
    sha256: str


@dataclass(frozen=True)
class ProjectBlock:
    """The effective project identity, where the activity has one.

    A ``hardware_setup`` root deliberately carries NO project (invariant 6): tuning the machine is
    not work on anybody's specimens.

    ``tmp_dir`` is deliberately absent (plan section 2.7): ``dirs._ensure_tmp`` falls back to
    ``<root>/_tmp_original`` on ``OSError``, so a configured scratch path is not knowable before
    ``build_dirs()`` succeeds and must not appear in a ``starting`` record.
    """

    run_name: str
    input_dirs: tuple[str, ...]
    artifact_dir: str                    # storage role: persistent project storage
    active_state_dir: str                # storage role: node-local scratch on a cluster
    active_db_path: str                  # storage role: <active_state_dir>/<run>.sqlite
    archive_mode: ArchiveMode
    archive_status: ArchiveStatus
    archive_pointer_path: str | None     # storage role: null in in-place mode
    archived_db_path: str | None         # derived; null only under pending/failed
    run_dir: str
    log_path: str
    #: Set when a checkpoint failed. Required alongside ``stale`` and ``failed`` (section 2.10).
    archive_error: str | None = None


@dataclass(frozen=True)
class HardwareBlock:
    """Where a ``hardware_setup`` root will write its profile.

    ``config`` is REQUIRED alongside this block: optimized or calibrated setup needs a valid
    canonical config, because model hashes, enabled stages, ``compute.devices`` and the scratch
    location all come from configuration (plan section 2.13).
    """

    destination_path: str


@dataclass(frozen=True)
class ControlBlock:
    """Capability metadata. A server may stop a run only under section 2.5's five-way match."""

    mode: ControlMode
    owner_instance_id: str | None = None


@dataclass(frozen=True)
class ChildSummary:
    """The bounded child summary embedded in the root's ``active.json`` -- exactly six keys, so the
    root record stays bounded however many children a root launches over its life (section 3.2)."""

    run_id: str
    activity: Activity
    state: RunState
    started_at: str
    run_name: str | None
    run_dir: str | None


@dataclass(frozen=True)
class RuntimeRecord:
    """One activity's registry record.

    The common fields are section 3.2 verbatim, in the order the plan lists them. The optional
    blocks below them are gated per activity by :data:`REQUIRED_BLOCKS`.

    Frozen on purpose: a record is published by atomic replace, so a mutation is always a NEW
    record. Use ``dataclasses.replace`` to advance ``state``/``updated_at``.
    """

    run_id: str
    activity: Activity
    activity_role: ActivityRole
    state: RunState
    launcher: Launcher
    pid: int
    process_started_at: float            # OS process creation time; with pid, defeats PID reuse
    started_at: str                      # ISO-8601 UTC, "...Z"
    updated_at: str                      # ISO-8601 UTC, "...Z"
    deployment: DeploymentInfo
    schema_version: int = SCHEMA_VERSION
    parent_run_id: str | None = None     # required for children; must match a live root
    error: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    # -- per-activity blocks (plan section 3.2 table) ------------------------------------------- #
    config: ConfigRef | None = None
    project: ProjectBlock | None = None
    hardware: HardwareBlock | None = None
    control: ControlBlock | None = None
    # -- root-only, bounded child tracking ------------------------------------------------------ #
    current_child: ChildSummary | None = None
    last_child: ChildSummary | None = None


#: What each activity's record MUST carry, beyond the common fields (plan section 3.2 table).
#: ``parent_run_id`` is listed for children because it is a common field that is optional for roots
#: and mandatory for them.
REQUIRED_BLOCKS: dict[Activity, tuple[str, ...]] = {
    Activity.PIPELINE: ("config", "project"),
    Activity.HARDWARE_SETUP: ("config", "hardware"),
    Activity.CALIBRATION_PIPELINE: ("parent_run_id", "project"),
}

#: Blocks that must be ABSENT. A ``hardware_setup`` root with a ``project`` block would contradict
#: invariant 6, and the GUI would show one species as though it were the whole run.
FORBIDDEN_BLOCKS: dict[Activity, tuple[str, ...]] = {
    Activity.HARDWARE_SETUP: ("project",),
    Activity.PIPELINE: ("hardware",),
    Activity.CALIBRATION_PIPELINE: ("current_child", "last_child"),
}


@dataclass(frozen=True)
class GrantRecord:
    """``children/<child_run_id>.grant.json`` -- plan section 2.2, verbatim.

    Written by the ROOT before it launches the child. The child never edits it: consumption is the
    atomic rename onto ``.grant.consumed.json``, because a ``consumed`` field would need a second
    writer on a file the root owns, which section 3.2 forbids and which the shared lease cannot
    serialize.

    ``consumed`` therefore stays in the schema as a readable marker the CLAIMANT writes INTO the
    renamed file -- never as the mechanism.

    Scope: this is COORDINATION protection. It stops an accidental or mis-ordered invocation of the
    internal child path from bypassing the lease. It is NOT a security boundary against hostile code
    running as the same user, which can read the environment and the runtime directory anyway.
    """

    child_run_id: str
    parent_run_id: str
    deployment_id: str                   # the canonical deployment key
    purpose: Activity                    # a CHILD activity; see CHILD_ACTIVITIES
    capability_sha256: str
    issued_at: float                     # unix epoch seconds
    expires_at: float
    schema_version: int = SCHEMA_VERSION
    consumed: bool = False


@dataclass(frozen=True)
class RuntimeSnapshot:
    """What the compatibility reader returns (plan section 2.9).

    ``active`` comes from the LOCK, never from the record: a held lease with an unreadable or
    newer-schema record is still ``active: True``. ``compatible`` is False for
    :class:`RecordClassification.INCOMPATIBLE`, and the UI then disables Start and every control
    action, shows "This runtime was created by a newer LM3", and interprets no unknown fields.
    """

    active: bool
    compatible: bool
    classification: RecordClassification
    record: RuntimeRecord | None = None
    #: Populated only when the record could not be parsed into :class:`RuntimeRecord` -- an
    #: incompatible or malformed payload. Never interpreted, only reported.
    raw: Mapping[str, Any] | None = None
    schema_version: int | None = None
    children: tuple[RuntimeRecord, ...] = ()
    #: Human-readable diagnostic for the UI; None when nothing is wrong.
    message: str | None = None
    #: Where the record came from, for diagnostics.
    path: Path | None = None


@dataclass(frozen=True)
class ChildHandoff:
    """Everything one approved subactivity launch needs, and nothing that outlives it.

    ``env`` carries the lease reference and the raw capability; ``popen_kwargs`` carries the
    platform's inheritance opt-in (``pass_fds`` on POSIX, the ``STARTUPINFOEX`` handle allowlist on
    Windows). ``close`` MUST be called on every path including every failure path -- on Windows a
    failed spawn otherwise leaks an inheritable duplicate that nothing will ever release (gate 34).

    Obtain it from :meth:`LeaseAdapter.child_handoff`, which is a context manager precisely so the
    close cannot be forgotten.
    """

    capability: str
    env: Mapping[str, str]
    popen_kwargs: Mapping[str, Any]
    close: Callable[[], None]
    #: POSIX only: the descriptor number handed to the child. None on Windows.
    lease_fd: int | None = None
    #: Windows only: the inheritable duplicate's handle value. None on POSIX.
    lease_handle: int | None = None


# --------------------------------------------------------------------------------------------- #
# On-disk / on-wire shapes
# --------------------------------------------------------------------------------------------- #
# Written as required-base + total=False subclass rather than typing.NotRequired, which is 3.11+.
# Readers tolerate MISSING optional keys as well as explicit null (plan section 2.9).

class DeploymentDict(TypedDict, total=False):
    id: str
    scheduler: str | None
    job_id: str | None
    step_id: str | None
    node: str | None
    container_id: str | None


class ConfigDict(TypedDict):
    path: str
    sha256: str


class ProjectDict(TypedDict, total=False):
    run_name: str
    input_dirs: list[str]
    artifact_dir: str
    active_state_dir: str
    active_db_path: str
    archive_mode: str
    archive_status: str
    archive_pointer_path: str | None
    archived_db_path: str | None
    run_dir: str
    log_path: str
    archive_error: str | None


class HardwareDict(TypedDict):
    destination_path: str


class ControlDict(TypedDict, total=False):
    mode: str
    owner_instance_id: str | None


class ChildSummaryDict(TypedDict, total=False):
    run_id: str
    activity: str
    state: str
    started_at: str
    run_name: str | None
    run_dir: str | None


class _RuntimeRecordRequired(TypedDict):
    schema_version: int
    run_id: str
    activity: str
    activity_role: str
    state: str
    launcher: str
    pid: int
    process_started_at: float
    started_at: str
    updated_at: str
    deployment: DeploymentDict


class RuntimeRecordDict(_RuntimeRecordRequired, total=False):
    parent_run_id: str | None
    error: str | None
    finished_at: str | None
    returncode: int | None
    config: ConfigDict | None
    project: ProjectDict | None
    hardware: HardwareDict | None
    control: ControlDict | None
    current_child: ChildSummaryDict | None
    last_child: ChildSummaryDict | None


class _GrantRequired(TypedDict):
    schema_version: int
    child_run_id: str
    parent_run_id: str
    deployment_id: str
    purpose: str
    capability_sha256: str
    issued_at: float
    expires_at: float


class GrantRecordDict(_GrantRequired, total=False):
    consumed: bool


class ArchivePointerDict(TypedDict, total=False):
    """``archive.current.json`` -- staged mode only, resolved through, never guessed."""

    schema_version: int
    run_id: str
    generation: str                      # "<run_id>.<sequence>"
    archived_db_path: str
    committed_at: float


class HandshakeMessage(TypedDict, total=False):
    """The single JSON line a launched child writes to the status channel (plan section 2.4)."""

    status: str                          # HandshakeStatus
    run_id: str
    project: ProjectDict
    active: RuntimeRecordDict            # present only when status == "busy"


# --------------------------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------------------------- #

class Win32Surface(Protocol):
    """The injectable Win32 calls the Windows lease adapter is allowed to make.

    Nothing here imports ``ctypes``, ``msvcrt``, or ``winreg`` at module scope, so this file -- and
    every module that type-checks against it -- imports cleanly on Linux. The real surface is built
    at call time by ``lease.real_win32_surface()`` behind a ``sys.platform`` check; the tests pass a
    fake and exercise the SAME adapter logic on Linux (working rules; plan section 7).

    Handles are plain ``int``s. Failures raise ``OSError``; ``ERROR_ALREADY_EXISTS`` is NOT a
    failure -- it is the atomic test-and-create result, reported through :meth:`get_last_error`.
    """

    def create_event(
        self, name: str, *, security_attributes: Any, manual_reset: bool = False,
        initial_state: bool = False, inheritable: bool = False,
    ) -> int:
        """``CreateEventW``. Returns a handle; check :meth:`get_last_error` for
        ``ERROR_ALREADY_EXISTS``, which means the object already existed and the deployment is
        occupied. The returned handle is valid either way and MUST be closed by a loser."""

    def open_event(self, name: str, *, desired_access: int = EVENT_ALL_ACCESS,
                   inheritable: bool = False) -> int:
        """``OpenEventW`` on the canonical global name, for handle validation."""

    def close_handle(self, handle: int) -> None:
        """``CloseHandle``. The lease object lives until its LAST handle closes."""

    def duplicate_handle(self, handle: int, *, inheritable: bool) -> int:
        """``DuplicateHandle`` into a temporary inheritable copy, for one subactivity spawn only."""

    def compare_object_handles(self, first: int, second: int) -> bool:
        """``CompareObjectHandles`` (Windows 10 / Server 2016 -- LM3's normative Windows floor).
        Decides whether two handles refer to the same underlying kernel object."""

    def set_handle_information(self, handle: int, mask: int, flags: int) -> None:
        """``SetHandleInformation``; used with ``HANDLE_FLAG_INHERIT, 0`` to disarm inheritance."""

    def get_last_error(self) -> int:
        """``GetLastError`` for the immediately preceding call."""

    def current_user_sid(self) -> str:
        """The current user's SID in canonical ``ConvertSidToStringSidW`` form (``S-1-5-21-...``)."""

    def current_user_security_attributes(self) -> Any:
        """``SECURITY_ATTRIBUTES`` with a DACL granting the current user ``EVENT_ALL_ACCESS`` only.

        ``EVENT_ALL_ACCESS`` and not less: opening an EXISTING named event requests exactly that, so
        a stingier DACL would lock our own next process out of its own lease.
        """


class LeaseAdapter(Protocol):
    """The platform primitive behind :class:`RuntimeLease`. Implemented by both
    ``PosixLeaseAdapter`` (``flock`` on the open file description) and ``WindowsLeaseAdapter`` (a
    named ``Global\\`` kernel event). The two share NO mechanism, which is why every lifetime test
    runs on both platforms rather than being extrapolated from the POSIX result.
    """

    #: The canonical deployment key this lease coordinates.
    deployment_key: str
    #: ``<deployment runtime dir>/activity.lock``. On Windows this is a PASSIVE artifact only --
    #: never locked, never consulted; the event is the sole authority there.
    lock_path: Path

    def acquire(self) -> None:
        """Take the lease. Raises :class:`RuntimeBusyError` if the deployment is occupied -- by a
        live root, or by a subactivity whose root has already died. A Windows loser MUST close the
        handle ``CreateEventW`` returned before raising, or it pins the deployment occupied."""

    def release(self) -> None:
        """Drop this process's reference. Idempotent. The deployment is free only when the LAST
        reference in the activity tree goes away."""

    def is_held(self) -> bool:
        """True if THIS adapter instance holds (or inherited) the lease."""

    def probe_occupied(self) -> bool:
        """Non-destructive: is the deployment occupied by anyone? Used by observers such as
        ``GET /v1/runtime``, which must never acquire anything."""

    def child_handoff(self, capability: str) -> AbstractContextManager[ChildHandoff]:
        """Context manager yielding one approved subactivity's :class:`ChildHandoff`, closing the
        parent's inheritable duplicate on EVERY exit path. On Windows it also holds the
        process-creation mutex, so no unrelated spawn can inherit the temporarily inheritable
        handle (plan section 2.2, step 6)."""

    def validate_inherited(self) -> None:
        """Prove the inherited reference really is this deployment's lease, or raise
        :class:`LeaseInheritanceError`. POSIX: ``fstat`` inode match against ``activity.lock`` plus a
        non-blocking re-assert on the inherited open file description, rejecting an ordinary
        independently opened handle. Windows: ``OpenEventW`` the canonical name and
        ``CompareObjectHandles``, closing the comparison handle either way."""

    def disarm_inheritance(self) -> None:
        """Stop the reference propagating any further, BEFORE any further spawn: POSIX
        ``os.set_inheritable(fd, False)``, Windows ``SetHandleInformation(h, HANDLE_FLAG_INHERIT,
        0)``. Ordinary executor workers must never hold the deployment open (invariant 3)."""


_E = TypeVar("_E", bound=enum.Enum)


def parse_enum(enum_cls: type[_E], value: object, *, field_name: str) -> _E:
    """Coerce a JSON string into ``enum_cls`` or raise :class:`RecordSchemaError`.

    Shared because ``records`` and ``grant`` both decode the same taxonomy, and a permissive
    ``getattr`` fallback in either one would let an unknown activity through as ``None``.
    """
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)  # type: ignore[call-arg]
    except (ValueError, KeyError) as exc:
        allowed = ", ".join(sorted(str(member.value) for member in enum_cls))
        raise RecordSchemaError(
            f"{field_name}: {value!r} is not one of {{{allowed}}}"
        ) from exc


__all__ = [
    # constants
    "ACTIVE_RECORD_FILENAME", "ACTIVITY_LOCK_FILENAME", "ARCHIVE_GENERATION_TEMPLATE",
    "ARCHIVE_POINTER_FILENAME", "ARCHIVE_TMP_SUFFIX", "CALIBRATION_RUN_NAME",
    "CHILDREN_DIRNAME", "CHILD_RECORD_SUFFIX", "CONNECTION_PRIVATE_FILENAME",
    "CONNECTION_PUBLIC_FILENAME", "DEFAULT_CHILD_RETENTION", "DEFAULT_GRANT_TTL_S",
    "DEFAULT_HANDSHAKE_TIMEOUT_S", "ENV_LEASE_CAPABILITY", "ENV_LEASE_EVENT_HANDLE", "ENV_LEASE_FD",
    "ENV_STATUS_FD", "ENV_STATUS_HANDLE", "ERROR_ALREADY_EXISTS", "EVENT_ALL_ACCESS",
    "EVENT_MODIFY_STATE", "EXIT_CODE_BUSY", "GRANT_CONSUMED_SUFFIX", "GRANT_SUFFIX",
    "HANDLE_FLAG_INHERIT", "HTTP_STATUS_BUSY", "LAST_RECORD_FILENAME", "LEASE_ENV_VARS",
    "MAX_ERROR_CHARS", "MAX_HANDSHAKE_BYTES", "MAX_INPUT_DIRS", "MAX_RECORD_BYTES",
    "MAX_STRING_FIELD_CHARS", "REDACTION_PLACEHOLDER", "SCHEMA_VERSION", "SECRET_KEY_SUBSTRINGS",
    "SID_HASH_LENGTH", "WINDOWS_LEASE_EVENT_TEMPLATE",
    # taxonomy
    "ACTIVITY_ROLE", "Activity", "ActivityRole", "CHILD_ACTIVITIES", "CHILD_PARENT_ACTIVITY",
    "ControlMode", "FORBIDDEN_BLOCKS", "HandshakeStatus", "Launcher", "REQUIRED_BLOCKS",
    "ROOT_ACTIVITIES",
    # lifecycle
    "LIVE_STATES", "RecordClassification", "RunState", "STATE_TRANSITIONS", "TERMINAL_STATES",
    # storage / archive
    "ArchiveMode", "ArchiveStatus", "COALESCEABLE_TRIGGERS", "CheckpointTrigger", "StorageRole",
    "TERMINAL_ARCHIVE_STATUSES",
    # errors
    "ArchivePointerError", "GrantAlreadyConsumedError", "GrantError", "GrantExpiredError",
    "GrantInvalidError", "IncompatibleSchemaError", "LeaseError", "LeaseInheritanceError",
    "LeaseNotHeldError", "RecordCorruptError", "RecordError", "RecordSchemaError", "StateTransitionError",
    "RuntimeBusyError", "RuntimeRegistryError", "WriterOwnershipError",
    # typed records
    "ChildHandoff", "ChildSummary", "ConfigRef", "ControlBlock", "DeploymentInfo",
    "GrantRecord", "HardwareBlock", "ProjectBlock", "RuntimeRecord", "RuntimeSnapshot",
    # wire shapes
    "ArchivePointerDict", "ChildSummaryDict", "ConfigDict", "ControlDict",
    "DeploymentDict", "GrantRecordDict", "HandshakeMessage", "HardwareDict", "ProjectDict",
    "RuntimeRecordDict",
    # protocols and shared helpers
    "LeaseAdapter", "Win32Surface", "parse_enum",
]
