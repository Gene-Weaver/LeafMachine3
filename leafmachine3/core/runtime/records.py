"""Registry record IO: atomic writes, per-activity validation, ownership, retention, recovery.

This module owns the *record* half of the unified runtime (plan sections 3.2, 3.3, 2.9 and the
2.10 archive contract). The lease decides who may run; these records describe *what* is running,
and they are what the API and the GUI read. Nothing here acquires, inherits, or releases a lease.

Five ideas carry the whole file:

* **Atomic publication.** Every write is a temp sibling + ``fsync`` + :func:`os.replace`, so a
  reader never sees a half-written record and a crash mid-write leaves the previous one intact.
* **Static writer ownership.** A parent and its approved child share the same open file
  description, so the activity lock does *not* serialize their JSON writes -- two atomic
  replacements can still clobber each other. Exclusivity and mutual exclusion are different
  problems, and the lease only solves the first. Ownership is therefore assigned by construction:
  :class:`RecordStore` built with ``role=ROOT`` can write ``active.json``/``last.json`` and nothing
  else; built with ``role=CHILD`` it can write ``children/<its own run_id>.json`` and nothing else.
  Calling the wrong writer raises :class:`WriterOwnershipError` before it touches the filesystem.
* **One documented exception.** :func:`recover_abandoned` writes ``last.json`` and ``active.json``
  without being the root -- legal only while holding the activity lock, which is what makes it
  unable to race a live root (section 3.2, "the recovery writer").
* **Bounded records.** A record is a description, never a log. Strings, error text, and
  ``input_dirs`` are capped, secret-shaped keys are refused outright, secret-shaped *values* are
  redacted, and the serialized result must fit :data:`MAX_RECORD_BYTES`. ``active.json`` carries
  only bounded child *summaries* precisely so it cannot grow with the number of children.
* **Skew fails safe** (section 2.9). A higher ``schema_version`` is never interpreted: the OS lock
  still decides occupancy, the snapshot reports ``active: true, compatible: false``, and the raw
  payload is carried through uninterpreted for display only.

Deliberately NOT here: the section 2.10 backup/checkpoint procedure itself (Step 8). This module
implements the record side of that contract -- the ``archive_mode``/``archive_status``/
``archived_db_path`` pairings and their validation -- not the copying.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import re
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from ._types import (
    ACTIVE_RECORD_FILENAME,
    ACTIVITY_ROLE,
    CALIBRATION_RUN_NAME,
    CHILD_RECORD_SUFFIX,
    CHILDREN_DIRNAME,
    DEFAULT_CHILD_RETENTION,
    FORBIDDEN_BLOCKS,
    GRANT_CONSUMED_SUFFIX,
    GRANT_SUFFIX,
    LAST_RECORD_FILENAME,
    LIVE_STATES,
    MAX_ERROR_CHARS,
    MAX_INPUT_DIRS,
    MAX_RECORD_BYTES,
    MAX_STRING_FIELD_CHARS,
    REDACTION_PLACEHOLDER,
    REQUIRED_BLOCKS,
    SCHEMA_VERSION,
    SECRET_KEY_SUBSTRINGS,
    TERMINAL_STATES,
    Activity,
    ActivityRole,
    ArchiveMode,
    ArchiveStatus,
    ChildSummary,
    ChildSummaryDict,
    ConfigRef,
    ControlBlock,
    ControlMode,
    DeploymentInfo,
    HardwareBlock,
    IncompatibleSchemaError,
    Launcher,
    ProjectBlock,
    RecordClassification,
    RecordCorruptError,
    RecordError,
    RecordSchemaError,
    RunState,
    RuntimeRecord,
    RuntimeRecordDict,
    RuntimeSnapshot,
    WriterOwnershipError,
    parse_enum,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; importing lease here would be a cycle
    from .lease import RuntimeLease

__all__ = [
    "RecordStore",
    "atomic_write_json",
    "classify_record",
    "new_run_id",
    "process_start_time",
    "prune_children",
    "read_json_file",
    "read_runtime",
    "record_from_dict",
    "record_to_dict",
    "recover_abandoned",
    "sanitize_record",
    "utc_now",
    "validate_record",
]

# --------------------------------------------------------------------------------------------- #
# Module constants
# --------------------------------------------------------------------------------------------- #

#: The exact wire form of ``started_at`` / ``updated_at`` / ``finished_at`` (plan section 3.2's
#: ``"2026-08-28T12:00:00Z"``). Second resolution, UTC, literal trailing ``Z`` -- not
#: ``datetime.isoformat()``, which emits ``+00:00`` and microseconds.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

#: Filename-safe stamp for quarantined records. ``active.json``'s replacement cannot use
#: :data:`TIMESTAMP_FORMAT` because Windows forbids ``:`` in filenames.
_FILE_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: How much of a record file we are willing to read before refusing it. We write at most
#: :data:`MAX_RECORD_BYTES`; the headroom is for a *newer* build's larger record, which we must be
#: able to read far enough to see its ``schema_version`` (section 2.9). Beyond this the file is a
#: log or an attack, not a record, and is reported as unreadable instead of loaded into memory.
MAX_READ_BYTES = MAX_RECORD_BYTES * 4

#: Appended to any truncated string so a reader can tell a bound from a value.
TRUNCATION_MARKER = "...[truncated]"

#: Progressive shrink ladder used by :func:`sanitize_record`. The per-field bounds alone do not
#: guarantee the record fits: 64 ``input_dirs`` at 4096 characters each is 256 KiB, four times
#: :data:`MAX_RECORD_BYTES`. Each rung is (string cap, input_dirs cap); the first rung that
#: serializes small enough wins, and exhausting the ladder is a hard error rather than a silently
#: oversized record.
_SHRINK_LADDER: tuple[tuple[int, int], ...] = (
    (MAX_STRING_FIELD_CHARS, MAX_INPUT_DIRS),
    (1024, 16),
    (512, 8),
    (256, 4),
    (256, 1),
)

#: Value-shaped secrets. Keys are handled separately (:data:`SECRET_KEY_SUBSTRINGS`); these catch a
#: bearer header or a ``token=...`` pasted into an ``error`` string by an exception's ``repr``.
#: Deliberately narrow -- they require the assignment/header punctuation, so ordinary prose that
#: merely contains the word "token" is not mangled.
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\bauthorization\s*[:=]\s*\S+"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|token|secret|password|passwd|credential)"
        r"\s*[:=]\s*\S+"
    ),
)

#: Error-text fields get the tighter :data:`MAX_ERROR_CHARS` bound.
_ERROR_FIELDS = frozenset({"error", "archive_error"})

# Known keys, per block. Anything else in a record claiming *our* schema version is either a
# forward-dated field (which would have arrived with a higher schema_version) or exactly the
# "environment dump" section 3.2 forbids -- so it is refused rather than ignored.
_RECORD_KEYS = frozenset(
    {
        "schema_version", "run_id", "activity", "activity_role", "parent_run_id", "state",
        "launcher", "pid", "process_started_at", "started_at", "updated_at", "deployment",
        "error", "finished_at", "returncode", "config", "project", "hardware",
        "control", "current_child", "last_child",
    }
)
_DEPLOYMENT_KEYS = frozenset({"id", "scheduler", "job_id", "step_id", "node", "container_id"})
_CONFIG_KEYS = frozenset({"path", "sha256"})
_PROJECT_KEYS = frozenset(
    {
        "run_name", "input_dirs", "artifact_dir", "active_state_dir", "active_db_path",
        "archive_mode", "archive_status", "archive_pointer_path", "archived_db_path", "run_dir",
        "log_path", "archive_error",
    }
)
_HARDWARE_KEYS = frozenset({"destination_path"})
_CONTROL_KEYS = frozenset({"mode", "owner_instance_id"})
_CHILD_SUMMARY_KEYS = frozenset({"run_id", "activity", "state", "started_at", "run_name", "run_dir"})

#: Project fields that must be absolute, resolved paths (section 3.2, "Paths are absolute and
#: resolved"). ``archive_pointer_path`` and ``archived_db_path`` are checked only when non-null.
_PROJECT_PATH_FIELDS = (
    "artifact_dir", "active_state_dir", "active_db_path", "run_dir", "log_path",
    "archive_pointer_path", "archived_db_path",
)


# --------------------------------------------------------------------------------------------- #
# Small primitives
# --------------------------------------------------------------------------------------------- #

def utc_now(clock: Callable[[], float] | None = None) -> str:
    """Return the current UTC time in the record's exact wire format."""
    now = (clock or time.time)()
    return datetime.fromtimestamp(now, tz=timezone.utc).strftime(TIMESTAMP_FORMAT)


def new_run_id() -> str:
    """A fresh ``run_id``: one *invocation*, even when it resumes the same project DB (section 3.2)."""
    return str(uuid.uuid4())


def process_start_time(pid: int | None = None) -> float:
    """OS process creation time in epoch seconds, or ``0.0`` when it cannot be determined.

    Paired with ``pid`` this is what defeats PID reuse in section 2.5's five-way stop match: a
    recycled PID belongs to a process created later, so the times disagree. ``psutil`` is optional
    throughout LM3, hence the guarded import and the ``/proc`` fallback; ``0.0`` means "unknown",
    and a caller comparing two unknowns must treat the match as failed rather than satisfied.
    """
    target = os.getpid() if pid is None else int(pid)
    try:
        import psutil  # noqa: PLC0415 - optional dependency, probed at call time
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        pass
    else:
        try:
            return float(psutil.Process(target).create_time())
        except Exception:  # noqa: BLE001 - dead process, permissions, platform quirk
            return 0.0
    if sys.platform.startswith("linux"):
        try:
            return _linux_process_start_time(target)
        except (OSError, ValueError, IndexError):
            return 0.0
    return 0.0


def _linux_process_start_time(pid: int) -> float:
    """``/proc/<pid>/stat`` field 22 (ticks since boot) plus ``/proc/stat``'s ``btime``.

    The comm field can itself contain spaces and parentheses, so the split is anchored on the LAST
    ``)`` rather than on whitespace.
    """
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    fields = raw[raw.rindex(")") + 1:].split()
    starttime_ticks = float(fields[19])  # field 22 overall; 2 consumed by pid and comm
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    boot_time = 0.0
    for line in Path("/proc/stat").read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("btime "):
            boot_time = float(line.split()[1])
            break
    return boot_time + starttime_ticks / float(ticks_per_second)


def _json_text(payload: Mapping[str, Any]) -> str:
    """Canonical serialization: sorted keys, indented, newline-terminated, UTF-8 safe."""
    return json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _fsync_directory(directory: Path) -> None:
    """fsync a directory so a rename survives a power loss. A no-op where unsupported (Windows)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    """Publish ``payload`` at ``path`` atomically: temp sibling, fsync, replace, fsync parent.

    The temp file is created in the *same* directory so :func:`os.replace` is a rename within one
    filesystem, which is the only form that is atomic. The parent fsync is what makes the rename
    itself durable -- without it the record and the rename can both still be in page cache when a
    node dies (the same reasoning section 2.10 applies to archive pointers).
    """
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    text = _json_text(payload)
    fd, tmp_name = tempfile.mkstemp(dir=str(directory), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    _fsync_directory(directory)


def read_json_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read one JSON object. Returns ``(payload, error)``; never raises.

    ``(None, None)`` means the file is simply absent -- which classifies as
    :attr:`RecordClassification.ABSENT`, not as a fault. Any other failure returns a human-readable
    diagnostic in the second slot: readers report it, and take exclusivity from the lock regardless.
    """
    path = Path(path)
    # The whole read stays inside one try chain so a mid-read OSError is *reported*, never raised:
    # the docstring's "never raises" is a guarantee every caller (status polls included) leans on.
    try:
        with open(path, "rb") as handle:
            # fstat the open descriptor, not the path, so the size describes the same file the
            # handle refers to, and refuse an oversized record BEFORE any payload byte is
            # allocated -- otherwise the "bound" is a post-hoc rejection of a file we already
            # loaded, and a grown or corrupted record costs its full size on every status poll.
            size = os.fstat(handle.fileno()).st_size
            if size > MAX_READ_BYTES:
                return None, (
                    f"oversized: {size} bytes exceeds the {MAX_READ_BYTES}-byte reader bound"
                )
            # Not redundant with the fstat gate: the file may have grown between the two calls,
            # and a pipe-like or otherwise non-regular path reports st_size 0 while streaming
            # unbounded bytes. Reading one byte past the bound is what detects both.
            raw = handle.read(MAX_READ_BYTES + 1)
    except FileNotFoundError:
        return None, None
    except IsADirectoryError:
        return None, f"{path} is a directory, not a record"
    except OSError as exc:
        return None, f"unreadable: {exc}"
    if len(raw) > MAX_READ_BYTES:
        return None, (
            f"oversized: more than {MAX_READ_BYTES} bytes exceeds the {MAX_READ_BYTES}-byte "
            f"reader bound"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        return None, f"malformed: not UTF-8 ({exc})"
    except json.JSONDecodeError as exc:
        return None, f"malformed JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"malformed: top level is {type(payload).__name__}, not an object"
    return payload, None


# --------------------------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------------------------- #

def _summary_to_dict(summary: ChildSummary) -> ChildSummaryDict:
    return {
        "run_id": summary.run_id,
        "activity": summary.activity.value,
        "state": summary.state.value,
        "started_at": summary.started_at,
        "run_name": summary.run_name,
        "run_dir": summary.run_dir,
    }


def record_to_dict(record: RuntimeRecord) -> RuntimeRecordDict:
    """Typed record -> the exact JSON shape of section 3.2.

    Common optional scalars are written explicitly as ``null`` (the plan's example shows them that
    way); absent *blocks* are omitted entirely. Both spellings mean the same thing to
    :func:`record_from_dict`, which tolerates missing keys and explicit null alike (section 2.9).
    """
    payload: dict[str, Any] = {
        "schema_version": int(record.schema_version),
        "run_id": record.run_id,
        "activity": record.activity.value,
        "activity_role": record.activity_role.value,
        "parent_run_id": record.parent_run_id,
        "state": record.state.value,
        "launcher": record.launcher.value,
        "pid": int(record.pid),
        "process_started_at": float(record.process_started_at),
        "started_at": record.started_at,
        "updated_at": record.updated_at,
        "deployment": {
            "id": record.deployment.id,
            "scheduler": record.deployment.scheduler,
            "job_id": record.deployment.job_id,
            "step_id": record.deployment.step_id,
            "node": record.deployment.node,
            "container_id": record.deployment.container_id,
        },
        "error": record.error,
        "finished_at": record.finished_at,
        "returncode": record.returncode,
    }
    if record.config is not None:
        payload["config"] = {"path": record.config.path, "sha256": record.config.sha256}
    if record.project is not None:
        project = record.project
        payload["project"] = {
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
    if record.hardware is not None:
        payload["hardware"] = {"destination_path": record.hardware.destination_path}
    if record.control is not None:
        payload["control"] = {
            "mode": record.control.mode.value,
            "owner_instance_id": record.control.owner_instance_id,
        }
    if record.current_child is not None:
        payload["current_child"] = _summary_to_dict(record.current_child)
    if record.last_child is not None:
        payload["last_child"] = _summary_to_dict(record.last_child)
    return payload  # type: ignore[return-value]


def _optional(payload: Mapping[str, Any], key: str) -> Any:
    """Missing key and explicit ``null`` are the same thing (section 2.9)."""
    return payload.get(key, None)


def _require(payload: Mapping[str, Any], key: str, *, field: str) -> Any:
    value = payload.get(key, None)
    if value is None:
        raise RecordSchemaError(f"{field}: required field is missing or null")
    return value


def _as_str(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise RecordSchemaError(f"{field}: expected a string, got {type(value).__name__}")
    return value


def _as_opt_str(value: Any, *, field: str) -> str | None:
    return None if value is None else _as_str(value, field=field)


def _as_int(value: Any, *, field: str) -> int:
    # bool is an int subclass; a boolean pid is a schema error, not a zero.
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordSchemaError(f"{field}: expected an integer, got {type(value).__name__}")
    return value


def _as_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordSchemaError(f"{field}: expected a number, got {type(value).__name__}")
    return float(value)


def _as_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordSchemaError(f"{field}: expected an object, got {type(value).__name__}")
    return value


def _reject_unknown_keys(payload: Mapping[str, Any], known: frozenset[str], *, field: str) -> None:
    """An unknown key at our own schema version is refused, never ignored.

    A genuinely new field would have arrived with a higher ``schema_version``, which
    :func:`record_from_dict` routes to :class:`IncompatibleSchemaError` long before this. What is
    left is a hand-edited record or the "environment dump" section 3.2 forbids, and silently
    dropping it would let a record the GUI renders carry whatever the writer felt like attaching.
    """
    unknown = sorted(set(payload) - known)
    if unknown:
        where = f"{field}." if field else ""
        raise RecordSchemaError(
            f"unknown field(s) at schema_version {SCHEMA_VERSION}: "
            + ", ".join(f"{where}{name}" for name in unknown)
        )


def _reject_secret_keys(payload: Any, *, path: str = "") -> None:
    """Refuse any key -- at any depth -- whose name looks like a credential (section 3.2).

    Matching on the dotted path as well as the leaf name means a nested ``{"headers": {"bearer":
    ...}}`` is caught even if some future block legitimately owns a generic key name.
    """
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            dotted = f"{path}.{key}" if path else str(key)
            haystack = dotted.lower()
            for needle in SECRET_KEY_SUBSTRINGS:
                if needle in haystack:
                    raise RecordSchemaError(
                        f"refusing a record carrying a secret-shaped field {dotted!r}: records hold "
                        f"no bearer tokens, environment dumps, or other secrets (plan section 3.2)"
                    )
            _reject_secret_keys(value, path=dotted)
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _reject_secret_keys(value, path=f"{path}[{index}]")


def _deployment_from_dict(payload: Mapping[str, Any]) -> DeploymentInfo:
    _reject_unknown_keys(payload, _DEPLOYMENT_KEYS, field="deployment")
    return DeploymentInfo(
        id=_as_str(_require(payload, "id", field="deployment.id"), field="deployment.id"),
        scheduler=_as_opt_str(_optional(payload, "scheduler"), field="deployment.scheduler"),
        job_id=_as_opt_str(_optional(payload, "job_id"), field="deployment.job_id"),
        step_id=_as_opt_str(_optional(payload, "step_id"), field="deployment.step_id"),
        node=_as_opt_str(_optional(payload, "node"), field="deployment.node"),
        container_id=_as_opt_str(_optional(payload, "container_id"), field="deployment.container_id"),
    )


def _config_from_dict(payload: Mapping[str, Any]) -> ConfigRef:
    _reject_unknown_keys(payload, _CONFIG_KEYS, field="config")
    return ConfigRef(
        path=_as_str(_require(payload, "path", field="config.path"), field="config.path"),
        sha256=_as_str(_require(payload, "sha256", field="config.sha256"), field="config.sha256"),
    )


def _project_from_dict(payload: Mapping[str, Any]) -> ProjectBlock:
    _reject_unknown_keys(payload, _PROJECT_KEYS, field="project")
    raw_dirs = _require(payload, "input_dirs", field="project.input_dirs")
    if not isinstance(raw_dirs, (list, tuple)):
        raise RecordSchemaError("project.input_dirs: expected a list of strings")
    dirs = tuple(_as_str(item, field="project.input_dirs[]") for item in raw_dirs)
    return ProjectBlock(
        run_name=_as_str(_require(payload, "run_name", field="project.run_name"),
                         field="project.run_name"),
        input_dirs=dirs,
        artifact_dir=_as_str(_require(payload, "artifact_dir", field="project.artifact_dir"),
                             field="project.artifact_dir"),
        active_state_dir=_as_str(_require(payload, "active_state_dir",
                                          field="project.active_state_dir"),
                                 field="project.active_state_dir"),
        active_db_path=_as_str(_require(payload, "active_db_path", field="project.active_db_path"),
                               field="project.active_db_path"),
        archive_mode=parse_enum(ArchiveMode, _require(payload, "archive_mode",
                                                      field="project.archive_mode"),
                                field_name="project.archive_mode"),
        archive_status=parse_enum(ArchiveStatus, _require(payload, "archive_status",
                                                          field="project.archive_status"),
                                  field_name="project.archive_status"),
        archive_pointer_path=_as_opt_str(_optional(payload, "archive_pointer_path"),
                                         field="project.archive_pointer_path"),
        archived_db_path=_as_opt_str(_optional(payload, "archived_db_path"),
                                     field="project.archived_db_path"),
        run_dir=_as_str(_require(payload, "run_dir", field="project.run_dir"),
                        field="project.run_dir"),
        log_path=_as_str(_require(payload, "log_path", field="project.log_path"),
                         field="project.log_path"),
        archive_error=_as_opt_str(_optional(payload, "archive_error"),
                                  field="project.archive_error"),
    )


def _hardware_from_dict(payload: Mapping[str, Any]) -> HardwareBlock:
    _reject_unknown_keys(payload, _HARDWARE_KEYS, field="hardware")
    return HardwareBlock(
        destination_path=_as_str(
            _require(payload, "destination_path", field="hardware.destination_path"),
            field="hardware.destination_path",
        )
    )


def _control_from_dict(payload: Mapping[str, Any]) -> ControlBlock:
    _reject_unknown_keys(payload, _CONTROL_KEYS, field="control")
    return ControlBlock(
        mode=parse_enum(ControlMode, _require(payload, "mode", field="control.mode"),
                        field_name="control.mode"),
        owner_instance_id=_as_opt_str(_optional(payload, "owner_instance_id"),
                                      field="control.owner_instance_id"),
    )


def _summary_from_dict(payload: Mapping[str, Any], *, field: str) -> ChildSummary:
    _reject_unknown_keys(payload, _CHILD_SUMMARY_KEYS, field=field)
    return ChildSummary(
        run_id=_as_str(_require(payload, "run_id", field=f"{field}.run_id"),
                       field=f"{field}.run_id"),
        activity=parse_enum(Activity, _require(payload, "activity", field=f"{field}.activity"),
                            field_name=f"{field}.activity"),
        state=parse_enum(RunState, _require(payload, "state", field=f"{field}.state"),
                         field_name=f"{field}.state"),
        started_at=_as_str(_require(payload, "started_at", field=f"{field}.started_at"),
                           field=f"{field}.started_at"),
        run_name=_as_opt_str(_optional(payload, "run_name"), field=f"{field}.run_name"),
        run_dir=_as_opt_str(_optional(payload, "run_dir"), field=f"{field}.run_dir"),
    )


def record_from_dict(payload: Mapping[str, Any], *, path: Path | None = None) -> RuntimeRecord:
    """JSON -> typed record, or raise.

    :class:`IncompatibleSchemaError` when ``schema_version`` is newer than this build's -- the
    caller must then not interpret a single field (section 2.9). Everything else is a
    :class:`RecordSchemaError`. Missing optional keys and explicit ``null`` are equivalent; unknown
    keys and secret-shaped keys are refused.
    """
    payload = _as_mapping(payload, field="record")
    schema_version = _as_int(_require(payload, "schema_version", field="schema_version"),
                             field="schema_version")
    if schema_version > SCHEMA_VERSION:
        raise IncompatibleSchemaError(schema_version, path=path)
    _reject_secret_keys(payload)
    _reject_unknown_keys(payload, _RECORD_KEYS, field="")

    activity = parse_enum(Activity, _require(payload, "activity", field="activity"),
                          field_name="activity")
    deployment_payload = _as_mapping(_require(payload, "deployment", field="deployment"),
                                     field="deployment")
    config_payload = _optional(payload, "config")
    project_payload = _optional(payload, "project")
    hardware_payload = _optional(payload, "hardware")
    control_payload = _optional(payload, "control")
    current_child = _optional(payload, "current_child")
    last_child = _optional(payload, "last_child")

    return RuntimeRecord(
        run_id=_as_str(_require(payload, "run_id", field="run_id"), field="run_id"),
        activity=activity,
        activity_role=parse_enum(ActivityRole, _require(payload, "activity_role",
                                                        field="activity_role"),
                                 field_name="activity_role"),
        state=parse_enum(RunState, _require(payload, "state", field="state"), field_name="state"),
        launcher=parse_enum(Launcher, _require(payload, "launcher", field="launcher"),
                            field_name="launcher"),
        pid=_as_int(_require(payload, "pid", field="pid"), field="pid"),
        process_started_at=_as_float(
            _require(payload, "process_started_at", field="process_started_at"),
            field="process_started_at",
        ),
        started_at=_as_str(_require(payload, "started_at", field="started_at"), field="started_at"),
        updated_at=_as_str(_require(payload, "updated_at", field="updated_at"), field="updated_at"),
        deployment=_deployment_from_dict(deployment_payload),
        schema_version=schema_version,
        parent_run_id=_as_opt_str(_optional(payload, "parent_run_id"), field="parent_run_id"),
        error=_as_opt_str(_optional(payload, "error"), field="error"),
        finished_at=_as_opt_str(_optional(payload, "finished_at"), field="finished_at"),
        returncode=(None if _optional(payload, "returncode") is None
                    else _as_int(payload["returncode"], field="returncode")),
        config=(None if config_payload is None
                else _config_from_dict(_as_mapping(config_payload, field="config"))),
        project=(None if project_payload is None
                 else _project_from_dict(_as_mapping(project_payload, field="project"))),
        hardware=(None if hardware_payload is None
                  else _hardware_from_dict(_as_mapping(hardware_payload, field="hardware"))),
        control=(None if control_payload is None
                 else _control_from_dict(_as_mapping(control_payload, field="control"))),
        current_child=(None if current_child is None
                       else _summary_from_dict(_as_mapping(current_child, field="current_child"),
                                               field="current_child")),
        last_child=(None if last_child is None
                    else _summary_from_dict(_as_mapping(last_child, field="last_child"),
                                            field="last_child")),
    )


# --------------------------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------------------------- #

def _check_timestamp(value: str | None, *, field: str, required: bool) -> None:
    if value is None:
        if required:
            raise RecordSchemaError(f"{field}: required timestamp is missing")
        return
    if not _TIMESTAMP_RE.match(value):
        raise RecordSchemaError(f"{field}: {value!r} is not an ISO-8601 UTC timestamp ('...Z')")


def _check_absolute(value: str | None, *, field: str) -> None:
    if value is None:
        return
    if not value or not Path(value).is_absolute():
        raise RecordSchemaError(f"{field}: {value!r} is not an absolute path (plan section 3.2)")


def _check_common(record: RuntimeRecord) -> None:
    if not record.run_id:
        raise RecordSchemaError("run_id: must be a non-empty string")
    if record.schema_version != SCHEMA_VERSION:
        raise RecordSchemaError(
            f"schema_version: this build writes {SCHEMA_VERSION}, got {record.schema_version}"
        )
    if record.pid < 0:
        raise RecordSchemaError(f"pid: {record.pid} is negative")
    if record.process_started_at < 0:
        raise RecordSchemaError(f"process_started_at: {record.process_started_at} is negative")
    _check_timestamp(record.started_at, field="started_at", required=True)
    _check_timestamp(record.updated_at, field="updated_at", required=True)
    # A terminal record with no finish time is unusable as history: last.json's whole job is to say
    # when and how the activity ended.
    _check_timestamp(record.finished_at, field="finished_at",
                     required=record.state in TERMINAL_STATES)
    if record.state not in TERMINAL_STATES and record.finished_at is not None:
        raise RecordSchemaError(
            f"finished_at: set while state is {record.state.value!r}, which is not terminal"
        )
    if not record.deployment.id:
        raise RecordSchemaError("deployment.id: must be a non-empty canonical deployment key")


def _check_role_and_blocks(record: RuntimeRecord) -> None:
    expected_role = ACTIVITY_ROLE[record.activity]
    if record.activity_role is not expected_role:
        raise RecordSchemaError(
            f"activity_role: {record.activity.value!r} is always {expected_role.value!r}, not "
            f"{record.activity_role.value!r} -- only a root record corresponds to a lock acquisition"
        )
    for name in REQUIRED_BLOCKS[record.activity]:
        if getattr(record, name) is None:
            raise RecordSchemaError(
                f"{record.activity.value}: {name!r} is required (plan section 3.2 table)"
            )
    for name in FORBIDDEN_BLOCKS.get(record.activity, ()):  # type: ignore[arg-type]
        if getattr(record, name) is not None:
            raise RecordSchemaError(
                f"{record.activity.value}: {name!r} must be absent (plan section 3.2 table)"
            )
    if record.activity_role is ActivityRole.ROOT and record.parent_run_id is not None:
        raise RecordSchemaError(
            "parent_run_id: a root record has no parent -- a parent would imply an inherited lease"
        )
    if record.activity is Activity.CALIBRATION_PIPELINE:
        assert record.project is not None  # guaranteed by REQUIRED_BLOCKS above
        if record.project.run_name != CALIBRATION_RUN_NAME:
            raise RecordSchemaError(
                f"calibration_pipeline: project.run_name must be {CALIBRATION_RUN_NAME!r}, got "
                f"{record.project.run_name!r}"
            )

def _check_paths(record: RuntimeRecord) -> None:
    if record.config is not None:
        _check_absolute(record.config.path, field="config.path")
        if not record.config.sha256:
            raise RecordSchemaError("config.sha256: must fingerprint the bytes actually loaded")
    if record.hardware is not None:
        _check_absolute(record.hardware.destination_path, field="hardware.destination_path")
    if record.project is not None:
        project = record.project
        if not project.run_name:
            raise RecordSchemaError("project.run_name: must be a non-empty string")
        for field in _PROJECT_PATH_FIELDS:
            _check_absolute(getattr(project, field), field=f"project.{field}")
        if len(project.input_dirs) > MAX_INPUT_DIRS:
            raise RecordSchemaError(
                f"project.input_dirs: {len(project.input_dirs)} entries exceeds the "
                f"{MAX_INPUT_DIRS}-entry bound"
            )
        for index, value in enumerate(project.input_dirs):
            _check_absolute(value, field=f"project.input_dirs[{index}]")
    for name in ("current_child", "last_child"):
        summary: ChildSummary | None = getattr(record, name)
        if summary is None:
            continue
        if not summary.run_id:
            raise RecordSchemaError(f"{name}.run_id: must be a non-empty string")
        if ACTIVITY_ROLE[summary.activity] is not ActivityRole.CHILD:
            raise RecordSchemaError(
                f"{name}.activity: {summary.activity.value!r} is a root activity, not a subactivity"
            )
        _check_timestamp(summary.started_at, field=f"{name}.started_at", required=True)
        _check_absolute(summary.run_dir, field=f"{name}.run_dir")


def _check_archive(record: RuntimeRecord) -> None:
    """The section 2.10 ``archive_mode`` / ``archive_status`` / ``archived_db_path`` table, verbatim.

    Every pairing below is a row of that table. Enforcing them here is what stops a record from
    advertising an archive that does not exist, or a desktop run from acquiring pointer indirection
    it has no use for.
    """
    project = record.project
    if project is None:
        return
    mode = project.archive_mode
    status = project.archive_status
    archived = project.archived_db_path
    if mode is ArchiveMode.IN_PLACE:
        if status is not ArchiveStatus.NOT_APPLICABLE:
            raise RecordSchemaError(
                f"project: archive_mode 'in-place' requires archive_status 'n/a', got "
                f"{status.value!r}"
            )
        if project.archive_pointer_path is not None:
            raise RecordSchemaError(
                "project: in-place mode has no pointer -- archive_pointer_path must be null"
            )
        if archived != project.active_db_path:
            raise RecordSchemaError(
                "project: in-place mode copies nothing, so archived_db_path must equal "
                f"active_db_path ({archived!r} != {project.active_db_path!r})"
            )
        if project.archive_error is not None:
            raise RecordSchemaError(
                "project: in-place mode runs no archiving, so archive_error must be null"
            )
        return
    # -- staged -------------------------------------------------------------------------------- #
    if status is ArchiveStatus.NOT_APPLICABLE:
        raise RecordSchemaError("project: 'n/a' is the in-place status; a staged run is never n/a")
    if project.archive_pointer_path is None:
        raise RecordSchemaError(
            "project: staged mode requires a pointer -- archive_pointer_path must not be null"
        )
    if status is ArchiveStatus.PENDING:
        if archived is not None:
            raise RecordSchemaError(
                "project: archive_status 'pending' means no snapshot has committed, so "
                "archived_db_path must be null"
            )
        if project.archive_error is not None:
            raise RecordSchemaError(
                "project: 'pending' is the expected pre-first-snapshot window, not a failure -- "
                "archive_error must be null"
            )
    elif status is ArchiveStatus.READY:
        if archived is None:
            raise RecordSchemaError(
                "project: archive_status 'ready' must name the generation the pointer resolves to"
            )
    elif status is ArchiveStatus.STALE:
        if archived is None:
            raise RecordSchemaError(
                "project: archive_status 'stale' must name the last good generation"
            )
        if not project.archive_error:
            raise RecordSchemaError(
                "project: archive_status 'stale' must record archive_error for the failed final "
                "checkpoint"
            )
    elif status is ArchiveStatus.FAILED:
        if archived is not None:
            raise RecordSchemaError(
                "project: archive_status 'failed' means no snapshot ever committed, so "
                "archived_db_path must be null"
            )
        if not project.archive_error:
            raise RecordSchemaError("project: archive_status 'failed' must record archive_error")


def _check_secrets_and_bounds(record: RuntimeRecord) -> None:
    payload = record_to_dict(record)
    _reject_secret_keys(payload)
    _walk_strings(payload, _reject_secret_value)
    _walk_strings(payload, _reject_oversized_string)
    size = len(_json_text(payload).encode("utf-8"))
    if size > MAX_RECORD_BYTES:
        raise RecordSchemaError(
            f"record: {size} bytes exceeds the {MAX_RECORD_BYTES}-byte bound -- a record is a "
            f"bounded description, never a log (sanitize_record shrinks one that overflows)"
        )


def _walk_strings(payload: Any, visit: Callable[[str, str], None], *, path: str = "") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            _walk_strings(value, visit, path=f"{path}.{key}" if path else str(key))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _walk_strings(value, visit, path=f"{path}[{index}]")
    elif isinstance(payload, str):
        visit(path, payload)


def _reject_secret_value(field: str, value: str) -> None:
    for pattern in _SECRET_VALUE_PATTERNS:
        if pattern.search(value):
            raise RecordSchemaError(
                f"{field}: value looks like a credential; records carry no bearer tokens or "
                f"secrets (plan section 3.2). sanitize_record() redacts it."
            )


def _reject_oversized_string(field: str, value: str) -> None:
    leaf = field.rsplit(".", 1)[-1]
    cap = MAX_ERROR_CHARS if leaf in _ERROR_FIELDS else MAX_STRING_FIELD_CHARS
    if len(value) > cap:
        raise RecordSchemaError(
            f"{field}: {len(value)} characters exceeds the {cap}-character bound"
        )


def validate_record(record: RuntimeRecord) -> None:
    """Raise :class:`RecordSchemaError` unless ``record`` satisfies section 3.2 in full.

    Checked here, in order: common-field shape and timestamps; the activity/role and
    required/forbidden block tables; absolute paths; the section 2.10 archive pairings; and the
    secret/bounds rules. :class:`RecordStore` runs :func:`sanitize_record` *first* and validates the
    sanitized result, so a credential that leaked into an ``error`` string is redacted rather than
    crashing a finalization -- while a caller validating a raw record still gets a loud rejection.
    """
    _check_common(record)
    _check_role_and_blocks(record)
    _check_paths(record)
    _check_archive(record)
    _check_secrets_and_bounds(record)


# --------------------------------------------------------------------------------------------- #
# Sanitization
# --------------------------------------------------------------------------------------------- #

def _redact(value: str) -> str:
    for pattern in _SECRET_VALUE_PATTERNS:
        value = pattern.sub(REDACTION_PLACEHOLDER, value)
    return value


def _truncate(value: str, cap: int) -> str:
    if len(value) <= cap:
        return value
    if cap > len(TRUNCATION_MARKER):
        return value[: cap - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER
    return value[:cap]


def _bound_value(key: str, value: Any, *, string_cap: int, error_cap: int, dirs_cap: int) -> Any:
    if isinstance(value, Mapping):
        return {
            inner_key: _bound_value(str(inner_key), inner_value, string_cap=string_cap,
                                    error_cap=error_cap, dirs_cap=dirs_cap)
            for inner_key, inner_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = list(value)
        if key == "input_dirs":
            items = items[:dirs_cap]
        return [
            _bound_value(key, item, string_cap=string_cap, error_cap=error_cap, dirs_cap=dirs_cap)
            for item in items
        ]
    if isinstance(value, str):
        cap = min(error_cap, string_cap) if key in _ERROR_FIELDS else string_cap
        return _truncate(_redact(value), cap)
    return value


def sanitize_record(record: RuntimeRecord) -> RuntimeRecord:
    """Return a redacted, field-bounded copy that is guaranteed to serialize under the record cap.

    Two jobs. First, defense: secret-shaped values are replaced with
    :data:`REDACTION_PLACEHOLDER`, so a bearer token that reached an exception message never lands
    in a file the GUI renders or in a :class:`RuntimeBusyError` shown to the loser of a race.
    Second, bounds: the per-field caps alone do not bound the *record* (64 ``input_dirs`` at 4096
    characters each is four times :data:`MAX_RECORD_BYTES`), so the shrink ladder tightens the caps
    until the whole thing fits. Exhausting the ladder raises rather than writing an oversized record.
    """
    payload = record_to_dict(record)
    for string_cap, dirs_cap in _SHRINK_LADDER:
        bounded = {
            key: _bound_value(key, value, string_cap=string_cap,
                              error_cap=min(MAX_ERROR_CHARS, string_cap), dirs_cap=dirs_cap)
            for key, value in payload.items()
        }
        if len(_json_text(bounded).encode("utf-8")) <= MAX_RECORD_BYTES:
            return record_from_dict(bounded)
    raise RecordSchemaError(
        f"record for run {record.run_id} cannot be reduced under {MAX_RECORD_BYTES} bytes; it is "
        f"carrying data that does not belong in a registry record"
    )


# --------------------------------------------------------------------------------------------- #
# Classification and the section 2.9 compatibility reader
# --------------------------------------------------------------------------------------------- #

def classify_record(
    record: RuntimeRecord | None,
    *,
    lease_occupied: bool,
    parse_error: str | None = None,
    schema_version: int | None = None,
) -> RecordClassification:
    """Decide what a record on disk means. The lock -- never the JSON -- decides *occupancy*.

    ``INCOMPATIBLE`` outranks everything: a newer record cannot be parsed, so its parse failure is
    a symptom, not the diagnosis. ``ABANDONED`` is only ever reached with a free lease, which is why
    a stale record under a live child is still reported as ``LIVE``: the child holds the inherited
    reference, and that record is the only description of what is running (gate 4).
    """
    if schema_version is not None and schema_version > SCHEMA_VERSION:
        return RecordClassification.INCOMPATIBLE
    if parse_error is not None:
        return RecordClassification.UNREADABLE
    if record is None:
        return RecordClassification.ABSENT
    if record.state in TERMINAL_STATES:
        return RecordClassification.FINALIZED
    return RecordClassification.LIVE if lease_occupied else RecordClassification.ABANDONED


def _default_lease_probe(deployment_dir: Path) -> Callable[[], bool]:
    """A probe-only lease adapter for the deployment. Acquires nothing (invariant: observers observe).

    ``lease`` is imported lazily here so this module has no import-time dependency on it -- the two
    are written independently and ``lease`` imports ``records`` for the busy-record lookup.
    """
    def probe() -> bool:
        from . import lease as lease_module

        adapter = lease_module.lease_adapter(deployment_dir, deployment_key=deployment_dir.name)
        return adapter.probe_occupied()

    return probe


def _child_record_path(deployment_dir: Path, child_run_id: str) -> Path:
    return Path(deployment_dir) / CHILDREN_DIRNAME / f"{child_run_id}{CHILD_RECORD_SUFFIX}"


def _load_child(deployment_dir: Path, child_run_id: str) -> RuntimeRecord | None:
    """Best-effort read of one child record. A malformed child never breaks the root's snapshot."""
    payload, error = read_json_file(_child_record_path(deployment_dir, child_run_id))
    if payload is None or error is not None:
        return None
    try:
        return sanitize_record(record_from_dict(payload))
    except (RecordError, IncompatibleSchemaError):
        return None


def read_runtime(
    deployment_dir: Path,
    *,
    lease_probe: Callable[[], bool] | None = None,
    include_children: bool = True,
) -> RuntimeSnapshot:
    """The section 2.9 compatibility reader: what is this deployment doing, and can we act on it?

    ``active`` comes from ``lease_probe`` -- the OS lock -- and never from the JSON, so a held lease
    with a missing, corrupt, or newer-schema record is still ``active: True``. A higher
    ``schema_version`` yields ``compatible: False`` with the raw payload carried through
    uninterpreted and a message the UI shows verbatim; the caller then disables Start and every
    control action.

    ``children`` is deliberately bounded to the records the root *references*
    (``current_child`` then ``last_child``), which is how the API merges "the latest child record"
    without the response growing with the number of children (section 3.2).
    """
    deployment_dir = Path(deployment_dir)
    active_path = deployment_dir / ACTIVE_RECORD_FILENAME
    probe = lease_probe or _default_lease_probe(deployment_dir)
    occupied = bool(probe())

    payload, parse_error = read_json_file(active_path)
    schema_version: int | None = None
    if payload is not None:
        raw_version = payload.get("schema_version")
        if isinstance(raw_version, int) and not isinstance(raw_version, bool):
            schema_version = raw_version

    if schema_version is not None and schema_version > SCHEMA_VERSION:
        return RuntimeSnapshot(
            active=occupied,
            compatible=False,
            classification=RecordClassification.INCOMPATIBLE,
            record=None,
            raw=payload,
            schema_version=schema_version,
            children=(),
            message=(
                "This runtime was created by a newer LM3 "
                f"(record schema_version {schema_version}, this build reads {SCHEMA_VERSION})."
            ),
            path=active_path,
        )

    record: RuntimeRecord | None = None
    message = parse_error
    if payload is not None and parse_error is None:
        try:
            record = sanitize_record(record_from_dict(payload, path=active_path))
            validate_record(record)
        except IncompatibleSchemaError as exc:  # schema_version was absent or non-integer above
            return RuntimeSnapshot(
                active=occupied,
                compatible=False,
                classification=RecordClassification.INCOMPATIBLE,
                raw=payload,
                schema_version=exc.schema_version,
                message="This runtime was created by a newer LM3.",
                path=active_path,
            )
        except RecordError as exc:
            record = None
            parse_error = str(exc)
            message = parse_error

    classification = classify_record(
        record, lease_occupied=occupied, parse_error=parse_error, schema_version=schema_version
    )
    if message is None and classification is RecordClassification.ABSENT and occupied:
        # Occupied but unidentifiable: the lease is held and nothing says by what. Section 3.3
        # calls this out by name; the GUI must not render it as idle.
        message = "the deployment lease is held but no active record describes it"

    children: tuple[RuntimeRecord, ...] = ()
    if include_children and record is not None:
        seen: list[RuntimeRecord] = []
        seen_ids: set[str] = set()
        for summary in (record.current_child, record.last_child):
            if summary is None or summary.run_id in seen_ids:
                continue
            seen_ids.add(summary.run_id)
            child = _load_child(deployment_dir, summary.run_id)
            if child is not None:
                seen.append(child)
        children = tuple(seen)

    return RuntimeSnapshot(
        active=occupied,
        compatible=True,
        classification=classification,
        record=record,
        raw=payload if record is None else None,
        schema_version=schema_version,
        children=children,
        message=message,
        path=active_path,
    )


# --------------------------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------------------------- #

def _referenced_run_ids(deployment_dir: Path) -> set[str]:
    """Run ids that ``active.json`` or ``last.json`` still point at, plus their own run ids.

    Read defensively: a malformed or newer-schema root record must not cause its children to be
    swept, so anything unparseable contributes whatever ids can be scraped from the raw payload.
    """
    referenced: set[str] = set()
    for name in (ACTIVE_RECORD_FILENAME, LAST_RECORD_FILENAME):
        payload, error = read_json_file(Path(deployment_dir) / name)
        if payload is None or error is not None:
            continue
        for key in ("run_id",):
            value = payload.get(key)
            if isinstance(value, str) and value:
                referenced.add(value)
        for key in ("current_child", "last_child"):
            summary = payload.get(key)
            if isinstance(summary, Mapping):
                value = summary.get("run_id")
                if isinstance(value, str) and value:
                    referenced.add(value)
    return referenced


def _grant_is_expired(path: Path, now: float) -> bool:
    payload, error = read_json_file(path)
    if payload is None or error is not None:
        # An unreadable grant can never be redeemed (grant.py validates every field), so it is
        # garbage by definition.
        return True
    expires_at = payload.get("expires_at")
    if not isinstance(expires_at, (int, float)) or isinstance(expires_at, bool):
        return True
    return float(expires_at) <= now


def prune_children(
    deployment_dir: Path,
    *,
    retention: int = DEFAULT_CHILD_RETENTION,
    keep_run_ids: Iterable[str] = (),
    clock: Callable[[], float] | None = None,
) -> int:
    """Sweep ``children/``; return how many files were removed.

    The rules are section 3.2's "Retention", in order:

    * anything referenced by ``active.json`` or ``last.json`` is preserved, as is anything in
      ``keep_run_ids`` (the run the caller is finalizing right now, whose files it still needs);
    * durable per-species history lives in each species' own DB and the wrapper's TSV, never here, so nothing in this
      directory is history and pruning it loses nothing;
    * an **unexpired** grant is preserved even though nothing references it yet -- it belongs to a
      child that has been approved but has not started;
    * expired grants go immediately; consumed grants and child records go once they fall outside
      the ``retention`` newest, by modification time.
    """
    deployment_dir = Path(deployment_dir)
    children_dir = deployment_dir / CHILDREN_DIRNAME
    if not children_dir.is_dir():
        return 0
    now = (clock or time.time)()
    preserved = set(keep_run_ids) | _referenced_run_ids(deployment_dir)

    records: list[tuple[float, str, Path]] = []
    grants: list[Path] = []
    consumed: list[tuple[float, str, Path]] = []
    removed = 0

    for entry in sorted(children_dir.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if name.endswith(GRANT_CONSUMED_SUFFIX):
            consumed.append((mtime, name[: -len(GRANT_CONSUMED_SUFFIX)], entry))
        elif name.endswith(GRANT_SUFFIX):
            grants.append(entry)
        elif name.endswith(CHILD_RECORD_SUFFIX):
            records.append((mtime, name[: -len(CHILD_RECORD_SUFFIX)], entry))

    for path in grants:
        run_id = path.name[: -len(GRANT_SUFFIX)]
        if run_id in preserved:
            continue
        if _grant_is_expired(path, now):
            removed += int(_unlink(path))

    # Newest first, so "beyond the retention limit" means "older than the newest N".
    for group in (records, consumed):
        group.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for index, (_mtime, run_id, path) in enumerate(group):
            if run_id in preserved or index < max(0, retention):
                continue
            removed += int(_unlink(path))
    return removed


def _unlink(path: Path) -> bool:
    try:
        path.unlink()
    except OSError:
        return False
    return True


# --------------------------------------------------------------------------------------------- #
# Terminal-record rules for last.json
# --------------------------------------------------------------------------------------------- #

def _check_last_record(record: RuntimeRecord) -> None:
    """Extra rules that apply only to the record published as ``last.json`` (gates 14 and 26).

    Two things that are legal in ``active.json`` are not legal in history:

    * ``archive_status: pending`` -- a finalized staged run either committed a snapshot (``ready``
      or ``stale``) or never did (``failed``). "Pending forever" would offer the GUI a link to an
      archive that will never exist;
    * an ``archived_db_path`` under ``active_state_dir`` -- that storage dies with the allocation,
      and a historical GUI following it produces a confusing "database missing" instead of an
      honest "this run has no archive" (section 2.10, gate 26).
    """
    if record.activity_role is not ActivityRole.ROOT:
        raise WriterOwnershipError(
            f"last.json holds roots only (plan section 3.2); refusing a "
            f"{record.activity_role.value} record for run {record.run_id}"
        )
    if record.state not in TERMINAL_STATES:
        raise RecordSchemaError(
            f"last.json holds the terminal record; state {record.state.value!r} is not terminal"
        )
    project = record.project
    if project is None or project.archive_mode is ArchiveMode.IN_PLACE:
        return
    if project.archive_status is ArchiveStatus.PENDING:
        raise RecordSchemaError(
            "last.json: a finalized staged run is 'ready', 'stale', or 'failed' -- 'pending' would "
            "promise an archive that will never be written (plan section 2.10)"
        )
    archived = project.archived_db_path
    if project.archive_status in (ArchiveStatus.READY, ArchiveStatus.STALE) and not archived:
        raise RecordSchemaError(
            f"last.json: archive_status {project.archive_status.value!r} requires a resolved "
            f"archived_db_path (gate 14)"
        )
    if archived and _is_within(archived, project.active_state_dir):
        raise RecordSchemaError(
            f"last.json: archived_db_path {archived!r} is inside the node-local active_state_dir "
            f"{project.active_state_dir!r}, which is gone once the allocation ends (gate 26)"
        )


def _is_within(candidate: str, parent: str) -> bool:
    try:
        return Path(candidate) == Path(parent) or Path(parent) in Path(candidate).parents
    except (OSError, ValueError):  # pragma: no cover - defensive on malformed paths
        return False


def _terminalize_archive(project: ProjectBlock, reason: str) -> ProjectBlock:
    """Move a project block's archive state to a terminal value (section 2.10, gate 25).

    ``ready`` -> ``stale`` keeping the last good generation; ``pending`` -> ``failed`` with a null
    path, because no snapshot ever committed and the GUI must say recovery data is unavailable
    rather than offer a broken link. In-place and already-terminal blocks are returned unchanged.
    """
    if project.archive_mode is ArchiveMode.IN_PLACE:
        return project
    if project.archive_status is ArchiveStatus.READY:
        return dataclasses.replace(project, archive_status=ArchiveStatus.STALE,
                                   archive_error=_truncate(reason, MAX_ERROR_CHARS))
    if project.archive_status is ArchiveStatus.PENDING:
        return dataclasses.replace(project, archive_status=ArchiveStatus.FAILED,
                                   archived_db_path=None,
                                   archive_error=_truncate(reason, MAX_ERROR_CHARS))
    return project


# --------------------------------------------------------------------------------------------- #
# The recovery writer (the one documented exception to single-writer)
# --------------------------------------------------------------------------------------------- #

def _lease_holds(lease: "RuntimeLease") -> bool:
    held = getattr(lease, "held", None)
    if isinstance(held, bool):
        return held
    adapter = getattr(lease, "adapter", None)
    if adapter is not None and hasattr(adapter, "is_held"):
        return bool(adapter.is_held())
    if hasattr(lease, "is_held"):
        return bool(lease.is_held())
    return False


def _quarantine(active_path: Path, kind: str, clock: Callable[[], float] | None) -> Path:
    """Move an uninterpretable ``active.json`` aside so a new root may start (section 2.9, gate 50).

    Preserved rather than deleted: it is the only evidence of what the other build was doing, and
    deleting a record we admit we cannot read would be exactly the silent guess section 2.9 forbids.
    """
    stamp = datetime.fromtimestamp((clock or time.time)(), tz=timezone.utc).strftime(
        _FILE_STAMP_FORMAT
    )
    target = active_path.with_name(f"{active_path.stem}.{kind}.{stamp}.{uuid.uuid4().hex[:8]}.json")
    os.replace(active_path, target)
    _fsync_directory(active_path.parent)
    return target


def recover_abandoned(
    deployment_dir: Path,
    *,
    lease: "RuntimeLease",
    clock: Callable[[], float] | None = None,
) -> RuntimeRecord | None:
    """Finalize a stale ``active.json`` on behalf of a root that is gone. Requires the lease.

    This is the *only* place a non-root writes ``active.json`` or ``last.json`` (section 3.2). Its
    safety comes entirely from the lock: holding the activity lease proves no root -- and no child
    holding an inherited reference -- is alive, so there is nobody to race. That is also why the
    ownership check below is not a formality: called without the lease it would be a second writer
    on a live root's file, and it refuses.

    Returns the terminal record it published, or ``None`` when there was nothing to recover (no
    record) or nothing it may interpret (a newer schema, or malformed JSON -- both quarantined).
    The recorded PID is never signaled: it may have been recycled, and a record is not authority.
    """
    deployment_dir = Path(deployment_dir)
    if not _lease_holds(lease):
        raise WriterOwnershipError(
            "recover_abandoned requires the activity lease: without it this would be a second "
            "writer on a live root's active.json (plan section 3.2, the recovery writer)"
        )
    lease_dir = getattr(lease, "deployment_dir", None)
    if lease_dir is not None and Path(lease_dir) != deployment_dir:
        raise WriterOwnershipError(
            f"lease covers {lease_dir}, not {deployment_dir}; a lease authorizes recovery of its "
            f"own deployment only"
        )

    active_path = deployment_dir / ACTIVE_RECORD_FILENAME
    payload, parse_error = read_json_file(active_path)
    if payload is None and parse_error is None:
        prune_children(deployment_dir, clock=clock)
        return None
    if parse_error is not None:
        _quarantine(active_path, "unreadable", clock)
        prune_children(deployment_dir, clock=clock)
        return None

    try:
        record = record_from_dict(payload or {}, path=active_path)
    except IncompatibleSchemaError:
        _quarantine(active_path, "incompatible", clock)
        prune_children(deployment_dir, clock=clock)
        return None
    except RecordError:
        _quarantine(active_path, "invalid", clock)
        prune_children(deployment_dir, clock=clock)
        return None

    now = utc_now(clock)
    if record.state in TERMINAL_STATES:
        # Caught between "wrote last.json" and "removed active.json": republish and clear.
        terminal = record
    else:
        reason = (
            f"abandoned: the {record.activity.value} process (pid {record.pid}) exited without "
            f"finalizing; recovered under the activity lock"
        )
        project = record.project
        terminal = dataclasses.replace(
            record,
            state=RunState.INTERRUPTED,
            error=_truncate(record.error or reason, MAX_ERROR_CHARS),
            finished_at=now,
            updated_at=now,
            project=None if project is None else _terminalize_archive(project, reason),
            current_child=None,
            last_child=record.last_child or record.current_child,
        )

    terminal = sanitize_record(terminal)
    validate_record(terminal)
    _check_last_record(terminal)

    # last.json first, then active.json: a crash between them leaves a duplicate description, which
    # the next reader resolves, rather than no description at all.
    atomic_write_json(deployment_dir / LAST_RECORD_FILENAME, record_to_dict(terminal))
    with contextlib.suppress(FileNotFoundError):
        active_path.unlink()
    _fsync_directory(deployment_dir)
    prune_children(deployment_dir, keep_run_ids={terminal.run_id}, clock=clock)
    return terminal


# --------------------------------------------------------------------------------------------- #
# RecordStore -- ownership enforced by construction
# --------------------------------------------------------------------------------------------- #

class RecordStore:
    """One process's view of the registry, scoped to what that process is allowed to write.

    The role and ``run_id`` are fixed at construction, so the wrong writer is a
    :class:`WriterOwnershipError` raised before any filesystem access rather than a clobbered file
    discovered later:

    ================================ ============================================================
    File                             Writer
    ================================ ============================================================
    ``active.json``                  the ROOT (``write_active``, ``set_current_child``,
                                     ``promote_current_child``, ``finalize``)
    ``children/<child_run_id>.json`` that CHILD, and only for its own ``run_id``
    ``last.json``                    the ROOT, at finalization
    ================================ ============================================================

    ``children/<id>.grant.json`` belongs to the root as well, but it is ``grant.py``'s file: the
    child claims it by atomic rename, never by editing it, which is why no method here writes it.
    """

    def __init__(
        self,
        deployment_dir: Path,
        *,
        run_id: str,
        role: ActivityRole = ActivityRole.ROOT,
    ) -> None:
        if not run_id:
            raise ValueError("run_id is required: it is what scopes this store's write permissions")
        self.deployment_dir = Path(deployment_dir)
        self.run_id = run_id
        self.role = role
        self.active_path = self.deployment_dir / ACTIVE_RECORD_FILENAME
        self.last_path = self.deployment_dir / LAST_RECORD_FILENAME
        self.children_dir = self.deployment_dir / CHILDREN_DIRNAME

    # -- internals ------------------------------------------------------------------------------ #

    def _require_root(self, what: str) -> None:
        if self.role is not ActivityRole.ROOT:
            raise WriterOwnershipError(
                f"{what} belongs to the root process; this store is a "
                f"{self.role.value} for run {self.run_id} (plan section 3.2)"
            )

    def _prepare(self, record: RuntimeRecord) -> RuntimeRecordDict:
        """Sanitize, then validate, then serialize. Order matters -- see :func:`validate_record`."""
        clean = sanitize_record(record)
        validate_record(clean)
        return record_to_dict(clean)

    def _read_record(self, path: Path) -> RuntimeRecord | None:
        payload, error = read_json_file(path)
        if payload is None:
            if error is None:
                return None
            raise RecordCorruptError(f"{path}: {error}")
        return record_from_dict(payload, path=path)

    def _read_own_active(self) -> RuntimeRecord:
        record = self._read_record(self.active_path)
        if record is None:
            raise RecordError(
                f"{self.active_path} does not exist; write_active() publishes it before any child "
                f"is launched"
            )
        if record.run_id != self.run_id:
            raise WriterOwnershipError(
                f"active.json describes run {record.run_id}, not {self.run_id}: this process is not "
                f"its writer"
            )
        return record

    # -- writers -------------------------------------------------------------------------------- #

    def write_active(self, record: RuntimeRecord) -> None:
        """Publish the root record. Root only, and only this store's own ``run_id``."""
        self._require_root("active.json")
        if record.run_id != self.run_id:
            raise WriterOwnershipError(
                f"this store writes run {self.run_id}; refusing a record for {record.run_id}"
            )
        if record.activity_role is not ActivityRole.ROOT:
            raise WriterOwnershipError(
                f"active.json holds root records; {record.activity.value} is a subactivity and "
                f"writes children/{record.run_id}.json instead"
            )
        atomic_write_json(self.active_path, self._prepare(record))

    def write_child(self, record: RuntimeRecord) -> None:
        """Publish this child's own record -- the only file a child ever writes.

        A child writes it for its whole life, including after its root dies: it may finish its own
        record, but it must never modify or remove the root's (section 3.2, sequence step 5).
        """
        if self.role is not ActivityRole.CHILD:
            raise WriterOwnershipError(
                f"children/{record.run_id}.json belongs to that child; this store is a "
                f"{self.role.value} for run {self.run_id}"
            )
        if record.run_id != self.run_id:
            raise WriterOwnershipError(
                f"a child writes only its own record: this store is run {self.run_id}, the record "
                f"is run {record.run_id}"
            )
        if record.activity_role is not ActivityRole.CHILD:
            raise WriterOwnershipError(
                f"children/ holds child records; {record.activity.value} is a root activity"
            )
        atomic_write_json(_child_record_path(self.deployment_dir, self.run_id),
                          self._prepare(record))

    def set_current_child(self, summary: ChildSummary | None) -> None:
        """Record (or clear) the approved child in ``active.json``. Root only.

        Called **before** the child is launched, so that a hard kill in the launch window still
        leaves the deployment describable: the summary names what may be holding the inherited
        lease. Pass ``None`` when a spawn fails, or the root will refuse to finalize waiting for a
        child that never started.
        """
        self._require_root("active.json's current_child")
        record = self._read_own_active()
        updated = dataclasses.replace(record, current_child=summary, updated_at=utc_now())
        atomic_write_json(self.active_path, self._prepare(updated))

    def promote_current_child(self) -> None:
        """Move ``current_child`` to ``last_child`` on normal child completion (section 3.2, step 4)."""
        self._require_root("active.json's child summaries")
        record = self._read_own_active()
        if record.current_child is None:
            return
        updated = dataclasses.replace(
            record, current_child=None, last_child=record.current_child, updated_at=utc_now()
        )
        atomic_write_json(self.active_path, self._prepare(updated))

    def finalize(self, record: RuntimeRecord, *, live_children: Sequence[str] = ()) -> None:
        """Publish the terminal record and clear ``active.json``. Root only. Refuses under a child.

        Ordering is section 3.3's, and it is not interchangeable: ``last.json`` first, then
        ``active.json`` is removed, then (by the caller, in its ``finally``) the lease is released.
        A crash between the first two steps leaves a duplicate description, which the next reader
        resolves; the reverse order would leave a window with no description at all.

        The refusal is gate 5. ``active.json`` is never removed while an approved child is alive,
        because the child still holds the inherited lease: the deployment would be *occupied but
        unidentifiable*, and the GUI would show idle while the GPUs are pinned. A child counts as
        alive when the caller says so in ``live_children`` -- the authority for the window between
        spawn and the child's first write -- or when its on-disk record is in a live state.
        """
        self._require_root("last.json")
        if record.run_id != self.run_id:
            raise WriterOwnershipError(
                f"this store finalizes run {self.run_id}; refusing a record for {record.run_id}"
            )
        alive = self._alive_children(record, live_children)
        if alive:
            raise RecordError(
                "refusing to finalize while approved subactivit"
                + ("y " if len(alive) == 1 else "ies ")
                + ", ".join(sorted(alive))
                + " remain(s) alive: a root must join every subactivity first, and active.json is "
                "never removed under a live child (plan section 3.3, gate 5)"
            )
        clean = sanitize_record(record)
        validate_record(clean)
        _check_last_record(clean)
        atomic_write_json(self.last_path, record_to_dict(clean))
        with contextlib.suppress(FileNotFoundError):
            self.active_path.unlink()
        _fsync_directory(self.deployment_dir)
        prune_children(self.deployment_dir, keep_run_ids={self.run_id})

    def _alive_children(self, record: RuntimeRecord, live_children: Sequence[str]) -> set[str]:
        alive = {run_id for run_id in live_children if run_id}
        candidates: set[str] = set(alive)
        for source in (record, self._read_active_quietly()):
            if source is None:
                continue
            for summary in (source.current_child, source.last_child):
                if summary is not None:
                    candidates.add(summary.run_id)
        for run_id in candidates:
            child = _load_child(self.deployment_dir, run_id)
            # A missing child record is NOT treated as alive: a spawn that failed would otherwise
            # block finalization forever. The caller's live_children list is the authority for the
            # window between spawn and the child's first write.
            if child is not None and child.state in LIVE_STATES:
                alive.add(run_id)
        return alive

    def _read_active_quietly(self) -> RuntimeRecord | None:
        try:
            return self._read_record(self.active_path)
        except (RecordError, IncompatibleSchemaError):
            return None

    # -- readers -------------------------------------------------------------------------------- #

    def read_active(self) -> RuntimeSnapshot:
        """The section 2.9 snapshot for this deployment, including the referenced child records."""
        return read_runtime(self.deployment_dir)

    def read_child(self, child_run_id: str) -> RuntimeRecord | None:
        """One child's record, or ``None`` when it is absent, malformed, or from a newer build."""
        return _load_child(self.deployment_dir, child_run_id)

    def read_last(self) -> RuntimeRecord | None:
        """The most recent terminal root record, or ``None``.

        Projected through :func:`sanitize_record` for the same reason ``read_runtime`` and
        ``_load_child`` are: the bytes on disk may predate this build's caps or have been edited by
        hand, so a reader must never hand a caller a record that violates section 3.2's "no bearer
        tokens ... no secrets" rule or the bounded-description cap. The sanitize call stays inside
        the ``try`` because :class:`RecordSchemaError` subclasses :class:`RecordError`: an
        unshrinkable record yields ``None`` here, exactly as a corrupt one does.
        """
        try:
            record = self._read_record(self.last_path)
            # _read_record returns None for an absent file, and sanitize_record would raise
            # AttributeError on it -- which this except clause does not catch.
            return None if record is None else sanitize_record(record)
        except (RecordError, IncompatibleSchemaError):
            return None
