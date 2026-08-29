"""One-use capability grants for approved subactivities (plan section 2.2).

A root activity holds the deployment lease. When it launches an *approved* subactivity --
``calibration_pipeline`` under ``hardware_setup``, ``batch_item_pipeline`` under ``batch`` -- that
child inherits the lease reference itself rather than acquiring its own. Something has to decide
which processes are entitled to that inheritance, and "the child passed a capability string" is
meaningless unless the string is validated *against* something. It is validated against a grant
record the root writes to ``children/<child_run_id>.grant.json`` before the spawn.

Scope, stated plainly so nobody later builds a feature on top of it that assumes more
-----------------------------------------------------------------------------------
This is **coordination** protection. Its whole job is to stop an accidental or mis-ordered
invocation of the internal child path -- a stray CLI invocation, a resumed script, a replayed
command line -- from slipping past the lease and running a second activity inside one deployment.

It is **not** a security boundary. Hostile code running as the same user can read the child's
environment and the deployment runtime directory directly, so it can obtain everything a legitimate
child has. Nothing here defends against that, nothing here is intended to, and no future feature may
assume otherwise. If a real trust boundary is ever needed it has to come from the operating system
(a different user, a container, a sandbox), not from this file.

Consumption is a rename, not a field update
-------------------------------------------
The grant file belongs to the root (section 3.2's writer-ownership table). A ``consumed`` flag
flipped by the child would make the child a second writer on a file the root owns -- which section
3.2 forbids, and which the shared lease cannot serialize, because parent and child hold the *same*
open file description. So the claim is::

    children/<id>.grant.json   ->   children/<id>.grant.consumed.json

Exactly one claimant can win that rename; a replay finds no source file and fails closed. The
``consumed`` field survives in the schema as a readable marker the winner writes *into the renamed
file*, never as the mechanism.

The order of operations is itself a requirement
-----------------------------------------------
:func:`redeem_grant` performs section 2.2's four steps, and no step may forward-reference a later
one:

1. validate **everything**, including the inherited lease reference, before touching the grant file;
2. claim by atomic rename -- and on failure exit immediately;
3. re-read the claimed file and revalidate expiry and identity, because the claim itself can be
   delayed;
4. disarm inheritance and clear the lease environment variables, before config loading or any other
   expensive work, so a later fork cannot carry the reference.

Gate 22 is the reason step 1 comes first: "a child that fails descriptor validation never renames
the grant, leaving it claimable by the legitimate child". Claiming before checking would let an
invalid or mis-ordered child eat the legitimate child's grant and lock it out of a run it was
entitled to. The ordering is encoded, not merely documented: :func:`revalidate_claimed_grant`
refuses a path that is not already a *claimed* file, so step 3 cannot run before step 2, and
:func:`redeem_grant` is the only entry point a child path is meant to call.

The raw capability value travels only in the child's environment (``LM3_LEASE_CAPABILITY``). Any
mismatch at any step fails closed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import os
import re
import secrets
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping

from . import _types as t
from ._types import (
    CHILD_ACTIVITIES,
    CHILD_PARENT_ACTIVITY,
    CHILDREN_DIRNAME,
    DEFAULT_CHILD_RETENTION,
    DEFAULT_GRANT_TTL_S,
    GRANT_CONSUMED_SUFFIX,
    GRANT_SUFFIX,
    MAX_STRING_FIELD_CHARS,
    SCHEMA_VERSION,
    Activity,
    GrantAlreadyConsumedError,
    GrantExpiredError,
    GrantInvalidError,
    GrantRecord,
    GrantRecordDict,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; importing lease here would be a cycle
    from .lease import RuntimeLease

__all__ = [
    "mint_capability",
    "capability_digest",
    "grant_path",
    "consumed_grant_path",
    "grant_to_dict",
    "grant_from_dict",
    "issue_grant",
    "write_grant",
    "validate_grant",
    "claim_grant",
    "revalidate_claimed_grant",
    "redeem_grant",
    "prune_grants",
]

#: ``secrets.token_hex(32)`` is 32 random bytes rendered as 64 hex characters. Named so the
#: injectable RNG in :func:`mint_capability` and the default agree by construction.
CAPABILITY_BYTES = 32

#: Run ids are UUIDs, but this module turns one into a FILENAME, so the charset is checked rather
#: than assumed. A separator or a ``..`` in a "run id" would otherwise write outside ``children/``.
_SAFE_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


# --------------------------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------------------------- #

def _records() -> Any:
    """Import ``records`` lazily.

    ``records`` owns atomic JSON IO for the whole package; re-implementing it here would give the
    registry two answers about what "atomically written" means. The import is deferred to keep the
    module graph acyclic (``records`` reads grants when it prunes) and to keep import cost off the
    child's startup path until it actually writes something.
    """
    from . import records

    return records


def _check_id(value: object, field_name: str) -> str:
    """Reject anything that cannot safely be a run-id component of a filename."""
    if not isinstance(value, str) or not _SAFE_ID.match(value):
        raise GrantInvalidError(
            f"{field_name}: {value!r} is not a valid run id "
            "(expected 1-128 characters of A-Z a-z 0-9 . _ -, not starting with . _ or -)"
        )
    return value


def _check_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise GrantInvalidError(f"grant field {key!r}: expected a non-empty string, got {value!r}")
    if len(value) > MAX_STRING_FIELD_CHARS:
        raise GrantInvalidError(
            f"grant field {key!r}: {len(value)} characters exceeds the {MAX_STRING_FIELD_CHARS} bound"
        )
    return value


def _check_time(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    # bool is an int subclass; a JSON ``true`` here is a malformed grant, not a timestamp of 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GrantInvalidError(f"grant field {key!r}: expected a unix timestamp, got {value!r}")
    return float(value)


def _now(clock: Callable[[], float] | None) -> float:
    return (clock or time.time)()


# --------------------------------------------------------------------------------------------- #
# Capability values
# --------------------------------------------------------------------------------------------- #

def mint_capability(*, rng: Callable[[int], bytes] | None = None) -> str:
    """Return a fresh raw capability value -- ``secrets.token_hex(32)`` by default.

    Only its SHA-256 is ever written to disk; the raw value reaches the child solely through
    ``LM3_LEASE_CAPABILITY`` in that child's environment.
    """
    source = rng or secrets.token_bytes
    raw = source(CAPABILITY_BYTES)
    if not isinstance(raw, (bytes, bytearray)) or len(raw) != CAPABILITY_BYTES:
        raise GrantInvalidError(
            f"capability RNG returned {len(raw) if hasattr(raw, '__len__') else raw!r} bytes; "
            f"expected exactly {CAPABILITY_BYTES}"
        )
    return bytes(raw).hex()


def capability_digest(raw: str) -> str:
    """SHA-256 hexdigest of the UTF-8 encoding of ``raw``."""
    if not isinstance(raw, str) or not raw:
        raise GrantInvalidError("capability: expected a non-empty string")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------------------------- #

def _children_dir(deployment_dir: Path) -> Path:
    return Path(deployment_dir) / CHILDREN_DIRNAME


def grant_path(deployment_dir: Path, child_run_id: str) -> Path:
    """``<deployment>/children/<child_run_id>.grant.json`` -- written by the ROOT."""
    return _children_dir(deployment_dir) / f"{_check_id(child_run_id, 'child_run_id')}{GRANT_SUFFIX}"


def consumed_grant_path(deployment_dir: Path, child_run_id: str) -> Path:
    """``<deployment>/children/<child_run_id>.grant.consumed.json`` -- the rename target.

    Reaching this name is what "consumed" means. The claimant writes ``consumed: true`` into the
    file afterwards as a readable marker; the *rename* is the mechanism.
    """
    return (
        _children_dir(deployment_dir)
        / f"{_check_id(child_run_id, 'child_run_id')}{GRANT_CONSUMED_SUFFIX}"
    )


# --------------------------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------------------------- #

def grant_to_dict(grant: GrantRecord) -> GrantRecordDict:
    """The section 2.2 JSON shape, in the plan's field order."""
    payload: GrantRecordDict = {
        "schema_version": int(grant.schema_version),
        "child_run_id": grant.child_run_id,
        "parent_run_id": grant.parent_run_id,
        "deployment_id": grant.deployment_id,
        "purpose": grant.purpose.value,
        "capability_sha256": grant.capability_sha256,
        "issued_at": float(grant.issued_at),
        "expires_at": float(grant.expires_at),
        "consumed": bool(grant.consumed),
    }
    return payload


def grant_from_dict(payload: Mapping[str, Any]) -> GrantRecord:
    """Decode a grant file. **Any** fault is :class:`GrantInvalidError` -- grants fail closed.

    A grant is not a runtime record: there is no "read it anyway, the lock decides" path (section
    2.9). An unreadable or newer-schema grant authorizes nothing, so it is simply invalid.
    """
    if not isinstance(payload, Mapping):
        raise GrantInvalidError(f"grant: expected a JSON object, got {type(payload).__name__}")

    schema_version = payload.get("schema_version", None)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise GrantInvalidError(f"grant field 'schema_version': expected an int, got {schema_version!r}")
    if schema_version > SCHEMA_VERSION:
        raise GrantInvalidError(
            f"grant declares schema_version {schema_version}, newer than this build's "
            f"{SCHEMA_VERSION}; it authorizes nothing"
        )

    child_run_id = _check_id(_check_str(payload, "child_run_id"), "child_run_id")
    parent_run_id = _check_id(_check_str(payload, "parent_run_id"), "parent_run_id")
    deployment_id = _check_str(payload, "deployment_id")
    capability_sha256 = _check_str(payload, "capability_sha256")

    try:
        purpose = t.parse_enum(Activity, payload.get("purpose"), field_name="grant field 'purpose'")
    except t.RecordSchemaError as exc:
        # The taxonomy decoder is shared with ``records``; a grant translates its failure into the
        # grant family so a caller never has to catch two unrelated error trees.
        raise GrantInvalidError(str(exc)) from exc
    if purpose not in CHILD_ACTIVITIES:
        # A grant exists only to authorize an INHERITED subactivity. One naming a root activity
        # describes something that would have to acquire its own lease, so it is malformed, not
        # merely unauthorized.
        allowed = ", ".join(sorted(a.value for a in CHILD_ACTIVITIES))
        raise GrantInvalidError(
            f"grant field 'purpose': {purpose.value!r} is a root activity, not one of {{{allowed}}}"
        )

    issued_at = _check_time(payload, "issued_at")
    expires_at = _check_time(payload, "expires_at")
    if expires_at <= issued_at:
        raise GrantInvalidError(
            f"grant expires_at ({expires_at}) is not after issued_at ({issued_at})"
        )

    consumed = payload.get("consumed", False)
    if not isinstance(consumed, bool):
        raise GrantInvalidError(f"grant field 'consumed': expected a bool, got {consumed!r}")

    return GrantRecord(
        child_run_id=child_run_id,
        parent_run_id=parent_run_id,
        deployment_id=deployment_id,
        purpose=purpose,
        capability_sha256=capability_sha256,
        issued_at=issued_at,
        expires_at=expires_at,
        schema_version=schema_version,
        consumed=consumed,
    )


# --------------------------------------------------------------------------------------------- #
# The root side: mint, issue, write
# --------------------------------------------------------------------------------------------- #

def issue_grant(
    deployment_dir: Path,
    *,
    child_run_id: str,
    parent_run_id: str,
    deployment_id: str,
    purpose: Activity,
    ttl_s: float = DEFAULT_GRANT_TTL_S,
    clock: Callable[[], float] | None = None,
    rng: Callable[[int], bytes] | None = None,
) -> tuple[GrantRecord, str]:
    """Mint a capability and build the grant that will authorize it. ROOT only.

    Returns ``(grant, raw_capability)``. The raw value is returned rather than stored: it goes into
    the child's environment as ``LM3_LEASE_CAPABILITY`` and nowhere else. Only its digest is
    written, by :func:`write_grant`.

    ``deployment_dir`` is accepted (and validated against the child id) so a caller cannot mint a
    grant for one deployment and write it into another by accident.
    """
    _check_id(child_run_id, "child_run_id")
    _check_id(parent_run_id, "parent_run_id")
    if child_run_id == parent_run_id:
        raise GrantInvalidError("child_run_id and parent_run_id must differ")
    if not isinstance(deployment_id, str) or not deployment_id:
        raise GrantInvalidError("deployment_id: expected a non-empty canonical deployment key")
    if purpose not in CHILD_ACTIVITIES:
        allowed = ", ".join(sorted(a.value for a in CHILD_ACTIVITIES))
        raise GrantInvalidError(
            f"purpose {getattr(purpose, 'value', purpose)!r} is not an inherited subactivity "
            f"({allowed}); only those may run under a parent's lease"
        )
    if not isinstance(ttl_s, (int, float)) or isinstance(ttl_s, bool) or ttl_s <= 0:
        raise GrantInvalidError(f"ttl_s must be a positive number of seconds, got {ttl_s!r}")

    # ``deployment_dir`` participates only as a sanity check on the id charset today; it is in the
    # signature because the root always has it and a future audit hook will want it.
    grant_path(deployment_dir, child_run_id)

    issued_at = _now(clock)
    raw = mint_capability(rng=rng)
    grant = GrantRecord(
        child_run_id=child_run_id,
        parent_run_id=parent_run_id,
        deployment_id=deployment_id,
        purpose=purpose,
        capability_sha256=capability_digest(raw),
        issued_at=issued_at,
        expires_at=issued_at + float(ttl_s),
        consumed=False,
    )
    return grant, raw


def write_grant(deployment_dir: Path, grant: GrantRecord) -> Path:
    """Publish the grant the root just issued. Root-owned file, mode ``0o600``.

    Refuses to re-arm an id whose consumed marker already exists: a run id is one invocation, so a
    second grant under a claimed id means either a bug or a replay, and silently overwriting would
    hand a fresh capability to a path that has already been used once.
    """
    target = grant_path(deployment_dir, grant.child_run_id)
    consumed = consumed_grant_path(deployment_dir, grant.child_run_id)
    if consumed.exists():
        raise GrantInvalidError(
            f"a grant for child_run_id {grant.child_run_id!r} has already been consumed "
            f"({consumed}); run ids are single-use"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    _records().atomic_write_json(target, grant_to_dict(grant), mode=0o600)
    return target


# --------------------------------------------------------------------------------------------- #
# Step 1: validation. Nothing here touches the filesystem.
# --------------------------------------------------------------------------------------------- #

def validate_grant(
    grant: GrantRecord,
    *,
    capability: str,
    child_run_id: str,
    parent_run_id: str,
    deployment_id: str,
    purpose: Activity,
    now: float,
) -> None:
    """Section 2.2 step 1, minus the lease reference (which only the lease adapter can check).

    Pure and side-effect free by design: this is the check that must complete *before* the grant
    file is renamed (gate 22), so it must be impossible for it to have renamed anything.

    Identity is checked before the capability digest, and expiry last, because an identity mismatch
    is a mis-wiring the developer wants named precisely, while an expiry is an ordinary timing
    outcome the caller may want to distinguish.
    """
    if grant.schema_version > SCHEMA_VERSION:
        raise GrantInvalidError(
            f"grant declares schema_version {grant.schema_version}, newer than this build's "
            f"{SCHEMA_VERSION}"
        )
    if grant.child_run_id != child_run_id:
        raise GrantInvalidError(
            f"grant names child_run_id {grant.child_run_id!r}, not {child_run_id!r}"
        )
    if grant.deployment_id != deployment_id:
        raise GrantInvalidError(
            f"grant names deployment {grant.deployment_id!r}, not {deployment_id!r}"
        )
    if grant.parent_run_id != parent_run_id:
        raise GrantInvalidError(
            f"grant names parent_run_id {grant.parent_run_id!r}, not {parent_run_id!r}"
        )
    if purpose not in CHILD_ACTIVITIES or purpose not in CHILD_PARENT_ACTIVITY:
        allowed = ", ".join(sorted(a.value for a in CHILD_ACTIVITIES))
        raise GrantInvalidError(
            f"declared purpose {getattr(purpose, 'value', purpose)!r} is not an inherited "
            f"subactivity ({allowed})"
        )
    if grant.purpose is not purpose:
        raise GrantInvalidError(
            f"grant authorizes {grant.purpose.value!r}, but this process declares "
            f"{purpose.value!r}"
        )

    # Constant-time only as hygiene: both values are already readable by anyone who can read the
    # runtime directory (see the scope statement). It costs nothing and removes a distraction.
    if not isinstance(capability, str) or not capability:
        raise GrantInvalidError("capability: expected a non-empty string")
    if not hmac.compare_digest(capability_digest(capability), grant.capability_sha256):
        raise GrantInvalidError("capability does not match the grant's capability_sha256")

    if now >= grant.expires_at:
        raise GrantExpiredError(
            f"grant for {child_run_id!r} expired at {grant.expires_at} (now {now})"
        )


# --------------------------------------------------------------------------------------------- #
# Step 2: the claim. The rename IS the single-winner mechanism.
# --------------------------------------------------------------------------------------------- #

def claim_grant(deployment_dir: Path, *, child_run_id: str) -> Path:
    """Rename ``<id>.grant.json`` onto ``<id>.grant.consumed.json`` and return the new path.

    ``os.rename`` and deliberately not ``os.replace``: on POSIX exactly one of two concurrent
    renames of the same source succeeds and the loser sees ``ENOENT``; on Windows an existing
    target raises ``FileExistsError`` rather than clobbering a marker that records an earlier
    claim. Both losses mean the same thing -- someone else already has it -- so both become
    :class:`GrantAlreadyConsumedError`.

    A replay finds no source file and lands here too. That is the fail-closed path, not an
    exceptional one.
    """
    source = grant_path(deployment_dir, child_run_id)
    target = consumed_grant_path(deployment_dir, child_run_id)
    try:
        os.rename(source, target)
    except FileNotFoundError as exc:
        raise GrantAlreadyConsumedError(
            f"no claimable grant at {source}: already consumed, never issued, or replayed"
        ) from exc
    except FileExistsError as exc:
        raise GrantAlreadyConsumedError(
            f"grant for {child_run_id!r} was already claimed ({target} exists)"
        ) from exc
    except IsADirectoryError as exc:  # pragma: no cover - only a corrupted registry gets here
        raise GrantAlreadyConsumedError(f"grant path {source} is not a claimable file") from exc
    return target


# --------------------------------------------------------------------------------------------- #
# Step 3: revalidate the claimed file, then mark it.
# --------------------------------------------------------------------------------------------- #

def revalidate_claimed_grant(
    path: Path,
    *,
    capability: str,
    child_run_id: str,
    parent_run_id: str,
    deployment_id: str,
    purpose: Activity,
    clock: Callable[[], float] | None = None,
) -> GrantRecord:
    """Re-read the CLAIMED file, re-check expiry and identity, and mark it consumed.

    The claim can be delayed -- a slow spawn, a descheduled process, a stalled filesystem -- so the
    grant that was valid at step 1 may not be valid now. Re-reading also means the file that is
    about to be marked is the file that was actually renamed, not the one that was read earlier.

    ``path`` must already be a claimed file. That is the structural half of "no step may
    forward-reference a later one": step 3 cannot be run before step 2, because the only path it
    accepts is the one step 2 produces.
    """
    path = Path(path)
    if not path.name.endswith(GRANT_CONSUMED_SUFFIX):
        raise GrantInvalidError(
            f"{path.name} is not a claimed grant (expected a *{GRANT_CONSUMED_SUFFIX} path); "
            "revalidation runs on the renamed file, after the claim, never before it"
        )

    payload, error = _records().read_json_file(path)
    if error is not None or payload is None:
        raise GrantInvalidError(f"claimed grant at {path} is unreadable: {error}")
    grant = grant_from_dict(payload)
    if grant.consumed:
        # Only reachable by calling this directly on an already-marked file: the rename cannot be
        # won twice. Fail closed rather than re-bless a spent grant.
        raise GrantAlreadyConsumedError(f"claimed grant at {path} is already marked consumed")

    validate_grant(
        grant,
        capability=capability,
        child_run_id=child_run_id,
        parent_run_id=parent_run_id,
        deployment_id=deployment_id,
        purpose=purpose,
        now=_now(clock),
    )

    # The readable marker, written by the CLAIMANT into the file it now owns. The root never writes
    # this path, so there is still exactly one writer per file (section 3.2).
    marked = dataclasses.replace(grant, consumed=True)
    _records().atomic_write_json(path, grant_to_dict(marked), mode=0o600)
    return marked


# --------------------------------------------------------------------------------------------- #
# The whole ordered procedure -- the only entry point a child path should call.
# --------------------------------------------------------------------------------------------- #

def redeem_grant(
    deployment_dir: Path,
    *,
    child_run_id: str,
    parent_run_id: str,
    deployment_id: str,
    purpose: Activity,
    capability: str,
    lease: "RuntimeLease",
    clock: Callable[[], float] | None = None,
    env: MutableMapping[str, str] | None = None,
) -> GrantRecord:
    """Section 2.2's four steps, in order, with no step forward-referencing a later one.

    1. Validate everything -- the inherited lease reference **and** the grant record -- before the
       grant file is touched in any way that could consume it.
    2. Claim by atomic rename; a failure here exits immediately.
    3. Re-read the claimed file and revalidate.
    4. Disarm inheritance and clear ``LM3_LEASE_*`` from the environment, before config loading or
       any other expensive work, so a later fork cannot carry the reference.

    The lease reference is validated first because it is the check gate 22 is written about: a
    process whose inherited descriptor or handle is not really the deployment lease must leave the
    grant file exactly where it found it, still claimable by the legitimate child.
    """
    # -- step 1a: the lease reference itself. lease.py owns the platform mechanics (POSIX fstat +
    # non-blocking re-assert, Windows OpenEventW + CompareObjectHandles); reimplementing either
    # here would give the runtime two opinions about what a valid reference is.
    lease.validate_inherited()

    # -- step 1b: the grant record. Reading is not consuming; nothing below is reachable with an
    # unvalidated record.
    source = grant_path(deployment_dir, child_run_id)
    payload, error = _records().read_json_file(source)
    if error is not None or payload is None:
        if not source.exists():
            raise GrantAlreadyConsumedError(
                f"no grant at {source}: already consumed, never issued, or replayed"
            )
        raise GrantInvalidError(f"grant at {source} is unreadable: {error}")
    grant = grant_from_dict(payload)
    if grant.consumed:
        raise GrantAlreadyConsumedError(f"grant at {source} is already marked consumed")
    validate_grant(
        grant,
        capability=capability,
        child_run_id=child_run_id,
        parent_run_id=parent_run_id,
        deployment_id=deployment_id,
        purpose=purpose,
        now=_now(clock),
    )

    # -- step 2: the claim. From here the grant is ours or nobody's.
    claimed = claim_grant(deployment_dir, child_run_id=child_run_id)

    # -- step 3: revalidate the file we actually won, since the claim can be delayed.
    redeemed = revalidate_claimed_grant(
        claimed,
        capability=capability,
        child_run_id=child_run_id,
        parent_run_id=parent_run_id,
        deployment_id=deployment_id,
        purpose=purpose,
        clock=clock,
    )

    # -- step 4: stop the reference propagating BEFORE any expensive work. One call, because
    # disarming without clearing the environment (or the reverse) is a half fix: invariant 3 says
    # an ordinary executor worker must never hold the deployment open.
    lease.disarm_and_clear(env)
    return redeemed


# --------------------------------------------------------------------------------------------- #
# Retention (section 3.2, "Retention")
# --------------------------------------------------------------------------------------------- #

def prune_grants(
    deployment_dir: Path,
    *,
    clock: Callable[[], float] | None = None,
    retention: int = DEFAULT_CHILD_RETENTION,
) -> int:
    """Remove expired grants and consumed markers beyond ``retention``. Returns the count removed.

    Two rules, both from section 3.2:

    * an **expired** unclaimed grant can never be redeemed again, so it is removed outright. So is
      a malformed one -- it authorizes nothing by construction, and leaving unreadable files in the
      registry only makes the next diagnostic noisier.
    * **consumed** markers are history; the newest ``retention`` are kept, by modification time,
      and older ones are removed.

    Deletions race harmlessly with a concurrent claim: a file that disappears underneath us was
    already someone else's, so ``FileNotFoundError`` is ignored rather than raised.
    """
    children = _children_dir(deployment_dir)
    if not children.is_dir():
        return 0
    if not isinstance(retention, int) or isinstance(retention, bool) or retention < 0:
        raise GrantInvalidError(f"retention must be a non-negative int, got {retention!r}")

    now = _now(clock)
    removed = 0

    for candidate in sorted(children.glob(f"*{GRANT_SUFFIX}")):
        # ``*.grant.json`` also matches nothing else: consumed markers end in ``.grant.consumed.json``.
        payload, error = _records().read_json_file(candidate)
        expired = True
        if error is None and payload is not None:
            try:
                expired = now >= grant_from_dict(payload).expires_at
            except t.GrantError:
                expired = True  # unusable, therefore prunable
        if expired:
            removed += _unlink(candidate)

    consumed = sorted(
        (p for p in children.glob(f"*{GRANT_CONSUMED_SUFFIX}") if p.is_file()),
        key=lambda p: (_mtime(p), p.name),
        reverse=True,
    )
    for stale in consumed[retention:]:
        removed += _unlink(stale)

    return removed


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:  # pragma: no cover - vanished between glob and stat
        return 0.0


def _unlink(path: Path) -> int:
    try:
        path.unlink()
    except FileNotFoundError:
        return 0
    except OSError:  # pragma: no cover - a permission problem is a diagnostic, not a failure here
        return 0
    return 1
