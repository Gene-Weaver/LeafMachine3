"""``leafmachine3.core.runtime`` -- the deployment activity registry (plan section 3.1/3.2/3.3).

One named LM3 deployment holds at most one root activity lease. This package owns the primitives
that make that true: the lease adapters, the record schemas and their atomic IO, the one-use
capability grants that let an approved subactivity inherit the lease reference, and the bounded
pure path resolver that emits the four storage roles.

Import path
-----------
The plan says "Add ``leafmachine3/core/runtime.py``". It is a PACKAGE instead, so three
implementers can work on disjoint files, and this ``__init__`` re-exports the whole surface so
every documented import in the plan works exactly as written::

    from leafmachine3.core.runtime import RuntimeLease, RuntimeRecord, RuntimeBusyError

Step 2 scope
------------
NOTHING here is imported by ``machine3.py``, the server, or setup. Step 2's exit gate is "the
primitives pass in isolation on all three platforms; no production entry point has changed". The
wiring is Step 3.

Layout
------
``_types``      every shared dataclass, enum, TypedDict, protocol, exception, and constant. It is
                the contract: the other four modules consume it and may not redefine any of it.
``lease``       ``RuntimeLease`` + the POSIX and Windows adapters + inheritance plumbing.
``records``     atomic record IO, per-activity validation, writer ownership, retention, recovery
                writer, and the section 2.9 compatibility reader.
``grant``       capability mint / validate / claim-by-rename / revalidate.
``config_io``   ``Config`` serialization and the section 2.7 bounded pure resolver.

Deployment- and machine-key derivation live in :mod:`leafmachine3.core.paths` (landed in Step 1)
and are re-exported below, because Electron reproduces the deployment key from the same golden
vectors and a second implementation here would be a second answer.

The four implementation modules do not exist yet. Their names are resolved lazily through the
module-level ``__getattr__`` below, so importing this package today succeeds and a reference to a
not-yet-written symbol fails with a message naming the module that owes it -- rather than an
``ImportError`` at the top of this file that would break the tree for everyone.
"""
from __future__ import annotations

from typing import Any

from ..paths import (
    canonical_deployment_key,
    deployment_key,
    deployment_runtime_dir,
    machine_key,
)
from ._types import (
    ACTIVE_RECORD_FILENAME,
    ACTIVITY_LOCK_FILENAME,
    ACTIVITY_ROLE,
    ARCHIVE_GENERATION_TEMPLATE,
    ARCHIVE_POINTER_FILENAME,
    ARCHIVE_TMP_SUFFIX,
    CALIBRATION_RUN_NAME,
    CHILD_ACTIVITIES,
    CHILD_PARENT_ACTIVITY,
    CHILD_RECORD_SUFFIX,
    CHILDREN_DIRNAME,
    COALESCEABLE_TRIGGERS,
    CONNECTION_PRIVATE_FILENAME,
    CONNECTION_PUBLIC_FILENAME,
    DEFAULT_CHILD_RETENTION,
    DEFAULT_GRANT_TTL_S,
    DEFAULT_HANDSHAKE_TIMEOUT_S,
    ENV_LEASE_CAPABILITY,
    ENV_LEASE_EVENT_HANDLE,
    ENV_LEASE_FD,
    ENV_STATUS_FD,
    ENV_STATUS_HANDLE,
    ERROR_ALREADY_EXISTS,
    EVENT_ALL_ACCESS,
    EVENT_MODIFY_STATE,
    EXIT_CODE_BUSY,
    FORBIDDEN_BLOCKS,
    GRANT_CONSUMED_SUFFIX,
    GRANT_SUFFIX,
    HANDLE_FLAG_INHERIT,
    HTTP_STATUS_BUSY,
    LAST_RECORD_FILENAME,
    LEASE_ENV_VARS,
    LIVE_STATES,
    MAX_ERROR_CHARS,
    MAX_HANDSHAKE_BYTES,
    MAX_INPUT_DIRS,
    MAX_RECORD_BYTES,
    MAX_STRING_FIELD_CHARS,
    REDACTION_PLACEHOLDER,
    REQUIRED_BLOCKS,
    ROOT_ACTIVITIES,
    SCHEMA_VERSION,
    SECRET_KEY_SUBSTRINGS,
    SID_HASH_LENGTH,
    STATE_TRANSITIONS,
    TERMINAL_ARCHIVE_STATUSES,
    TERMINAL_STATES,
    WINDOWS_LEASE_EVENT_TEMPLATE,
    Activity,
    ActivityRole,
    ArchiveMode,
    ArchivePointerDict,
    ArchivePointerError,
    ArchiveStatus,
    CheckpointTrigger,
    ChildHandoff,
    ChildSummary,
    ChildSummaryDict,
    ConfigDict,
    ConfigRef,
    ControlBlock,
    ControlDict,
    ControlMode,
    DeploymentDict,
    DeploymentInfo,
    GrantAlreadyConsumedError,
    GrantError,
    GrantExpiredError,
    GrantInvalidError,
    GrantRecord,
    GrantRecordDict,
    HandshakeMessage,
    HandshakeStatus,
    HardwareBlock,
    HardwareDict,
    IncompatibleSchemaError,
    Launcher,
    LeaseAdapter,
    LeaseError,
    LeaseInheritanceError,
    LeaseNotHeldError,
    ProjectBlock,
    ProjectDict,
    RecordClassification,
    RecordCorruptError,
    RecordError,
    RecordSchemaError,
    StateTransitionError,
    RunState,
    RuntimeBusyError,
    RuntimeRecord,
    RuntimeRecordDict,
    RuntimeRegistryError,
    RuntimeSnapshot,
    StorageRole,
    Win32Surface,
    WriterOwnershipError,
    parse_enum,
)

# --------------------------------------------------------------------------------------------- #
# Lazily resolved surface -- the binding contract for the four implementation modules.
# --------------------------------------------------------------------------------------------- #
# These names ARE the public API. Publishing them here before the modules exist is deliberate: the
# signatures are fixed by the spec handed to each implementer, and four agents writing against one
# published surface is the only way the pieces meet. Each entry maps a public name to the submodule
# that owes it.

_LAZY: dict[str, str] = {
    "LaunchComposeError": "launch",
    "ComposedLaunch": "launch",
    "LaunchContribution": "launch",
    "composed_launch": "launch",
    "compose_launch": "launch",
    "build_deployment_info": "records",
    # A) runtime/lease.py -- lease adapters and inheritance plumbing (sections 2.2, 3.1, 3.3)
    "RuntimeLease": "lease",
    "PosixLeaseAdapter": "lease",
    "WindowsLeaseAdapter": "lease",
    "lease_adapter": "lease",
    "inherit_lease": "lease",
    "clear_lease_env": "lease",
    "windows_lease_event_name": "lease",
    "sid_hash": "lease",
    "real_win32_surface": "lease",
    "process_creation_lock": "lease",
    # Section 3.3's acquisition-ordering entry point plus its cleanup lock. These are named here
    # rather than left to a private submodule import because Step 3 calls them from machine3.py --
    # across the package boundary -- between cfg.validate() and ensure_hardware_profile(cfg).
    "acquire_root_lease": "lease",
    "probe_deployment_occupied": "lease",
    "cleanup_lease": "lease",
    # B) runtime/records.py -- atomic IO, validation, ownership, retention, compatibility reader
    "RecordStore": "records",
    "atomic_write_json": "records",
    "read_json_file": "records",
    "record_to_dict": "records",
    "record_from_dict": "records",
    "validate_record": "records",
    "sanitize_record": "records",
    "classify_record": "records",
    "read_runtime": "records",
    "recover_abandoned": "records",
    "prune_children": "records",
    "new_run_id": "records",
    "utc_now": "records",
    "process_start_time": "records",
    # C) runtime/grant.py -- one-use capability grants (section 2.2)
    "mint_capability": "grant",
    "capability_digest": "grant",
    "grant_path": "grant",
    "consumed_grant_path": "grant",
    "issue_grant": "grant",
    "write_grant": "grant",
    "validate_grant": "grant",
    "claim_grant": "grant",
    "revalidate_claimed_grant": "grant",
    "redeem_grant": "grant",
    "prune_grants": "grant",
    "grant_to_dict": "grant",
    "grant_from_dict": "grant",
    # D) runtime/config_io.py -- Config serialization + the section 2.7 bounded pure resolver
    "config_to_dict": "config_io",
    "config_sha256": "config_io",
    "config_ref": "config_io",
    "resolve_run_paths": "config_io",
    "storage_roles": "config_io",
    "project_block": "config_io",
    "resolve_archived_db_path": "config_io",
    "read_archive_pointer": "config_io",
    "canonical_json": "config_io",
    "config_json": "config_io",
    "archive_mode": "config_io",
    "resolved_paths": "config_io",
    "db_path": "config_io",
    "CLUSTER_STATE_DIR_KEY": "config_io",
    # Section 3.4's launch-manifest half. The builder is pure, but the mapping it returns is written
    # by Step 3 from machine3.py once build_dirs() has settled tmp_dir, so the builder, the path
    # helper, the filename, and the version stamp all have to be reachable from outside the package.
    "LAUNCH_MANIFEST_FILENAME": "config_io",
    "launch_manifest_path": "config_io",
    "build_launch_manifest": "config_io",
    "lm3_version": "config_io",
    # E) runtime/execution.py -- the Step 3 activity context managers (sections 2.2, 2.4, 3.3, 3.4)
    # THE feature flag reader, the root/child/subactivity context managers, and the child half of
    # the launch handshake. Every execution entry point reaches the lease through these, which is
    # what makes invariant 4 one implementation instead of four.
    "ENV_RUNTIME_V2": "execution",
    "ENV_CHILD_RUN_ID": "execution",
    "ENV_PARENT_RUN_ID": "execution",
    "ENV_CHILD_ACTIVITY": "execution",
    "CHILD_ENV_VARS": "execution",
    "DEFAULT_CHILD_JOIN_TIMEOUT_S": "execution",
    "DEFAULT_CHILD_KILL_TIMEOUT_S": "execution",
    "runtime_v2_enabled": "execution",
    "is_approved_child": "execution",
    "child_base_env": "execution",
    "StatusChannel": "execution",
    "status_channel": "execution",
    "write_launch_manifest": "execution",
    "ActivityHandle": "execution",
    "RootActivity": "execution",
    "ChildActivity": "execution",
    "DisabledActivity": "execution",
    "SubactivityLaunch": "execution",
    "InheritedLease": "execution",
    "root_activity": "execution",
    "child_activity": "execution",
    "execution_activity": "execution",
    "launch_subactivity": "execution",
    "bind_child_lease": "execution",
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy resolution for the four implementation modules.

    Deliberately raises ``AttributeError`` (which ``from ... import X`` reports as an ImportError
    naming this package) with the owing module spelled out, so a reference landing before its
    module does is a one-line diagnosis rather than a mystery.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    try:
        module = import_module(f".{module_name}", __name__)
    except ModuleNotFoundError as exc:
        # Only mask the absence of OUR submodule; a genuine missing third-party import inside it
        # must still surface as itself.
        if exc.name not in {f"{__name__}.{module_name}", module_name}:
            raise
        raise AttributeError(
            f"{name!r} is part of the published runtime API but "
            f"leafmachine3/core/runtime/{module_name}.py has not been written yet (plan section 4, "
            f"Step 2). Its signature is fixed by the Step 2 skeleton spec."
        ) from exc
    value = getattr(module, name)
    globals()[name] = value  # cache: a second lookup skips import_module entirely
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    # -- schema / process constants ------------------------------------------------------------ #
    "SCHEMA_VERSION", "EXIT_CODE_BUSY", "HTTP_STATUS_BUSY", "DEFAULT_CHILD_RETENTION",
    "DEFAULT_GRANT_TTL_S", "DEFAULT_HANDSHAKE_TIMEOUT_S", "CALIBRATION_RUN_NAME",
    # -- registry filenames -------------------------------------------------------------------- #
    "ACTIVITY_LOCK_FILENAME", "ACTIVE_RECORD_FILENAME", "LAST_RECORD_FILENAME", "CHILDREN_DIRNAME",
    "CHILD_RECORD_SUFFIX", "GRANT_SUFFIX", "GRANT_CONSUMED_SUFFIX",
    "CONNECTION_PRIVATE_FILENAME", "CONNECTION_PUBLIC_FILENAME",
    "ARCHIVE_POINTER_FILENAME", "ARCHIVE_GENERATION_TEMPLATE", "ARCHIVE_TMP_SUFFIX",
    # -- environment plumbing ------------------------------------------------------------------ #
    "ENV_LEASE_FD", "ENV_LEASE_EVENT_HANDLE", "ENV_LEASE_CAPABILITY", "LEASE_ENV_VARS",
    "ENV_STATUS_FD", "ENV_STATUS_HANDLE", "MAX_HANDSHAKE_BYTES",
    # -- record bounds / redaction ------------------------------------------------------------- #
    "MAX_RECORD_BYTES", "MAX_STRING_FIELD_CHARS", "MAX_ERROR_CHARS", "MAX_INPUT_DIRS",
    "REDACTION_PLACEHOLDER", "SECRET_KEY_SUBSTRINGS",
    # -- Windows lease naming and Win32 constants ---------------------------------------------- #
    "WINDOWS_LEASE_EVENT_TEMPLATE", "SID_HASH_LENGTH", "ERROR_ALREADY_EXISTS", "EVENT_ALL_ACCESS",
    "EVENT_MODIFY_STATE", "HANDLE_FLAG_INHERIT",
    # -- taxonomy ------------------------------------------------------------------------------ #
    "Activity", "ActivityRole", "Launcher", "ControlMode", "HandshakeStatus",
    "ROOT_ACTIVITIES", "CHILD_ACTIVITIES", "ACTIVITY_ROLE", "CHILD_PARENT_ACTIVITY",
    "REQUIRED_BLOCKS", "FORBIDDEN_BLOCKS",
    # -- lifecycle ----------------------------------------------------------------------------- #
    "RunState", "RecordClassification", "TERMINAL_STATES", "LIVE_STATES", "STATE_TRANSITIONS",
    # -- storage / archive --------------------------------------------------------------------- #
    "StorageRole", "ArchiveMode", "ArchiveStatus", "TERMINAL_ARCHIVE_STATUSES",
    "CheckpointTrigger", "COALESCEABLE_TRIGGERS",
    # -- errors -------------------------------------------------------------------------------- #
    "RuntimeRegistryError", "RuntimeBusyError", "LeaseError", "LeaseNotHeldError",
    "LeaseInheritanceError", "RecordError", "RecordSchemaError", "RecordCorruptError",
    "StateTransitionError",
    "IncompatibleSchemaError", "WriterOwnershipError", "GrantError", "GrantInvalidError",
    "GrantExpiredError", "GrantAlreadyConsumedError", "ArchivePointerError",
    # -- typed records ------------------------------------------------------------------------- #
    "RuntimeRecord", "RuntimeSnapshot", "DeploymentInfo", "build_deployment_info",
    "compose_launch", "composed_launch", "LaunchContribution", "ComposedLaunch",
    "LaunchComposeError", "ConfigRef", "ProjectBlock",
    "HardwareBlock", "ControlBlock", "ChildSummary", "ChildHandoff", "GrantRecord",
    # -- wire shapes --------------------------------------------------------------------------- #
    "RuntimeRecordDict", "DeploymentDict", "ConfigDict", "ProjectDict",
    "HardwareDict", "ControlDict", "ChildSummaryDict", "GrantRecordDict", "ArchivePointerDict",
    "HandshakeMessage",
    # -- protocols and shared helpers ---------------------------------------------------------- #
    "LeaseAdapter", "Win32Surface", "parse_enum",
    # -- deployment identity, re-exported from core.paths (Step 1) ----------------------------- #
    "canonical_deployment_key", "deployment_key", "machine_key", "deployment_runtime_dir",
    # -- A) lease.py --------------------------------------------------------------------------- #
    "RuntimeLease", "PosixLeaseAdapter", "WindowsLeaseAdapter", "lease_adapter", "inherit_lease",
    "clear_lease_env", "windows_lease_event_name", "sid_hash", "real_win32_surface",
    "process_creation_lock", "acquire_root_lease", "probe_deployment_occupied", "cleanup_lease",
    # -- B) records.py ------------------------------------------------------------------------- #
    "RecordStore", "atomic_write_json", "read_json_file", "record_to_dict", "record_from_dict",
    "validate_record", "sanitize_record", "classify_record", "read_runtime", "recover_abandoned",
    "prune_children", "new_run_id", "utc_now", "process_start_time",
    # -- C) grant.py --------------------------------------------------------------------------- #
    "mint_capability", "capability_digest", "grant_path", "consumed_grant_path", "issue_grant",
    "write_grant", "validate_grant", "claim_grant", "revalidate_claimed_grant", "redeem_grant",
    "prune_grants", "grant_to_dict", "grant_from_dict",
    # -- D) config_io.py ----------------------------------------------------------------------- #
    "config_to_dict", "config_sha256", "config_ref", "resolve_run_paths", "storage_roles",
    "project_block", "resolve_archived_db_path", "read_archive_pointer", "canonical_json",
    "config_json", "archive_mode", "resolved_paths", "db_path", "CLUSTER_STATE_DIR_KEY",
    "LAUNCH_MANIFEST_FILENAME", "launch_manifest_path", "build_launch_manifest", "lm3_version",
    # -- E) execution.py ----------------------------------------------------------------------- #
    "ENV_RUNTIME_V2", "ENV_CHILD_RUN_ID", "ENV_PARENT_RUN_ID", "ENV_CHILD_ACTIVITY",
    "CHILD_ENV_VARS", "DEFAULT_CHILD_JOIN_TIMEOUT_S", "DEFAULT_CHILD_KILL_TIMEOUT_S",
    "runtime_v2_enabled", "is_approved_child", "child_base_env", "StatusChannel", "status_channel",
    "write_launch_manifest", "ActivityHandle", "RootActivity", "ChildActivity", "DisabledActivity",
    "SubactivityLaunch", "InheritedLease", "root_activity", "child_activity", "execution_activity",
    "launch_subactivity", "bind_child_lease",
]
