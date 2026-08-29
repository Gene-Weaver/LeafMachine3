"""``Config`` serialization, the section 2.7 bounded pure resolver, and the section 2.10 roles.

Three jobs, in the order the plan hands them out:

* **section 3.4** -- ``Config.to_dict()`` support plus the immutable launch manifest BUILDER. The
  builder is pure: it turns already-resolved values into the manifest mapping. Writing that mapping
  is Step 3's job, after ``build_dirs()`` has settled ``tmp_dir``.
* **section 2.7** -- :func:`resolve_run_paths`, which publishes ONLY what is deterministic from
  config: ``run_dir``, ``active_db_path``, ``log_path`` and the storage roles. It creates nothing,
  probes nothing, and is safe to call before a lease is even attempted.
* **section 2.10** -- the four storage roles on every platform, with the archive MODE derived from
  ``artifact_dir == active_state_dir`` rather than guessed or configured.

Why ``tmp_dir`` is absent, stated once so nobody re-adds it: ``dirs._ensure_tmp`` falls back to
``<root>/_tmp_original`` when the configured scratch directory cannot be created, so the final tmp
location is a function of whether a ``mkdir`` worked, not of the config. Section 3.2 therefore says
``tmp_dir`` "is deliberately absent" from a ``starting`` record. This module makes that impossible
to get wrong by never returning a tmp field at all -- :func:`resolved_paths` has no tmp key, and the
only place ``tmp_dir`` appears is the launch manifest, where the caller must PASS the settled value
it got back from ``build_dirs()``.

Gate 18 is the shape of the whole design: a desktop run resolves ``archive_mode: in-place``, a null
pointer, and performs no archive copying. Modes exist so today's desktop behavior stays
byte-for-byte identical; a resolver that introduced pointer indirection into a run with nothing to
stage would fail that gate no matter how correct its cluster half was.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform as platform_module
import sys
import time
import warnings
from collections.abc import Mapping as MappingABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .. import paths
from ..config import jsonable
from . import _types as t

#: ``<run>/logs/run_manifest.json`` (plan section 3.4). It hangs off ``log_path``'s directory, which
#: is under ``artifact_dir`` -- so on a cluster the manifest survives the allocation even though
#: ``active_state_dir`` is node-local, exactly as section 3.4 requires.
LAUNCH_MANIFEST_FILENAME = "run_manifest.json"

#: ``project.output.<this>`` -- the node-local scratch ROOT of a staged (cluster) run, parallel with
#: ``project.output.dir``. Absent, blank, or ``auto`` means desktop: state lives beside the
#: artifacts and the run is in-place. It is deliberately NOT in ``builtin_defaults()``: a default
#: value here would put a second answer for "where does the DB live" into every settings file, and
#: the desktop must keep resolving in-place from a config that has never heard of clusters.
CLUSTER_STATE_DIR_KEY = "active_state_dir"

#: Values of :data:`CLUSTER_STATE_DIR_KEY` that mean "no separate state dir", matching the ``auto``
#: spelling ``project.output.tmp_dir`` already uses.
_UNSET_PATH_VALUES: frozenset[str] = frozenset({"", "auto", "none", "null"})


# --------------------------------------------------------------------------------------------- #
# Config serialization (plan section 3.4)
# --------------------------------------------------------------------------------------------- #

def config_to_dict(cfg: Any) -> dict[str, Any]:
    """A JSON-safe, deterministic deep copy of the merged effective config.

    Prefers ``cfg.to_dict()`` -- the one implementation, in :mod:`leafmachine3.core.config` -- and
    falls back to normalizing a bare ``Section``/mapping tree so callers holding a raw config node
    (the settings API, a test fixture) get the same answer. There is no second serializer here on
    purpose: two normalizations means two fingerprints for one config.
    """
    to_dict = getattr(cfg, "to_dict", None)
    if callable(to_dict):
        out = to_dict()
    else:
        out = jsonable(getattr(cfg, "_raw", cfg))
    if not isinstance(out, dict):
        raise t.RecordSchemaError(
            f"config_to_dict expected a mapping-shaped config, got {type(out).__name__}"
        )
    return out


def canonical_json(payload: Mapping[str, Any]) -> str:
    """The canonical serialization: sorted keys, no incidental whitespace, stable bytes.

    Used for the config fingerprint and the launch manifest so "the same config" and "the same
    manifest" are byte-level claims a test can make.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_json(cfg: Any) -> str:
    """:func:`canonical_json` of :func:`config_to_dict` -- the effective config as stable bytes."""
    return canonical_json(config_to_dict(cfg))


def config_sha256(path: str | os.PathLike[str]) -> str:
    """SHA-256 of the config file's BYTES as they were loaded (plan section 3.2).

    The bytes, not the merged tree: that is what makes a mid-run edit of the YAML detectable
    against the launch manifest. Streamed, because a settings file is not guaranteed small.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_ref(cfg: Any = None, *, path: str | os.PathLike[str] | None = None) -> t.ConfigRef:
    """The record's ``config`` block: absolute resolved path plus the loaded bytes' SHA-256.

    ``path`` defaults to ``cfg.source_path``, which ``Config.load`` already stamps with a resolved
    absolute path. A config with neither is a programming error, not a fallback: guessing a path
    here would let the record pin the run to a file it never read.
    """
    source = path if path is not None else getattr(cfg, "source_path", None)
    if source is None:
        raise t.RecordSchemaError(
            "config_ref needs the config file it was loaded from; pass path= or use a Config built "
            "by Config.load(), which records source_path"
        )
    resolved = Path(source).expanduser().resolve()
    return t.ConfigRef(path=str(resolved), sha256=config_sha256(resolved))


# --------------------------------------------------------------------------------------------- #
# The bounded pure resolver (plan section 2.7)
# --------------------------------------------------------------------------------------------- #

def _node(source: Any, key: str) -> Any:
    """Read ``key`` from a ``Section``, a plain mapping, or an attribute-bearing object."""
    if isinstance(source, MappingABC):
        return source.get(key)
    return getattr(source, key, None)


def _optional_path_value(raw: Any) -> str | None:
    """``None`` for absent/blank/``auto`` config path values, the stripped string otherwise."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in _UNSET_PATH_VALUES:
        return None
    return text


def resolve_run_paths(
    cfg: Any,
    *,
    settings_file: str | os.PathLike[str] | None = None,
    active_state_dir: str | os.PathLike[str] | None = None,
) -> paths.EarlyRunPaths:
    """The section 2.7 resolver. Pure: creates nothing, probes nothing, publishes no ``tmp_dir``.

    ``project.output.dir`` is absolutized through :func:`paths.resolve_project_output_dir`, which
    resolves a relative value against the SETTINGS FILE's parent and refuses to join the current
    working directory (gate 44). ``settings_file`` defaults to ``cfg.source_path``.

    The staged/in-place decision comes from the state root and nothing else: an explicit
    ``active_state_dir`` argument wins, else ``project.output.active_state_dir``, else nothing --
    and "nothing" is a desktop run, resolved in-place with a null pointer (gate 18).
    """
    project = _node(cfg, "project")
    if project is None:
        raise t.RecordSchemaError("config has no 'project' block; cannot resolve run paths")
    output = _node(project, "output")
    run_name = _optional_path_value(_node(project, "run_name"))
    if not run_name:
        raise t.RecordSchemaError("config has no project.run_name; cannot resolve run paths")

    if settings_file is None:
        settings_file = getattr(cfg, "source_path", None)

    output_dir = paths.resolve_project_output_dir(settings_file, _optional_path_value(_node(output, "dir")))
    if output_dir is None:
        raise t.RecordSchemaError("config has no project.output.dir; cannot resolve run paths")

    state_root_raw = active_state_dir if active_state_dir is not None else _node(output, CLUSTER_STATE_DIR_KEY)
    # Same relative-path rule as the output dir, reused rather than re-decided: one function owns
    # "relative to the settings file, never the CWD" for every configured root.
    state_root = paths.resolve_project_output_dir(settings_file, _optional_path_value(state_root_raw))

    return paths.early_run_paths(
        output_dir=output_dir,
        run_name=run_name,
        active_state_dir=state_root,
    )


# --------------------------------------------------------------------------------------------- #
# The four storage roles (plan section 2.10)
# --------------------------------------------------------------------------------------------- #

def storage_roles(early: paths.EarlyRunPaths) -> dict[t.StorageRole, str | None]:
    """Exactly the four section 2.10 roles, as absolute strings.

    ``archive_pointer_path`` is ``None`` in in-place mode -- the null of the plan's table, not an
    empty string. ``archived_db_path`` is not here because it is DERIVED, not emitted: see
    :func:`resolve_archived_db_path` and :class:`~leafmachine3.core.runtime._types.StorageRole`.
    """
    return {
        t.StorageRole.ARTIFACT_DIR: str(early.artifact_dir),
        t.StorageRole.ACTIVE_STATE_DIR: str(early.active_state_dir),
        t.StorageRole.ACTIVE_DB_PATH: str(early.active_db_path),
        t.StorageRole.ARCHIVE_POINTER_PATH: (
            str(early.archive_pointer_path) if early.archive_pointer_path is not None else None
        ),
    }


def archive_mode(early: paths.EarlyRunPaths) -> t.ArchiveMode:
    """The mode, DERIVED from the roles: in-place iff ``artifact_dir == active_state_dir``.

    ``paths.early_run_paths`` already made that comparison; this only turns its string into the
    enum, so the two can never disagree.
    """
    return t.parse_enum(t.ArchiveMode, early.archive_mode, field_name="archive_mode")


def resolved_paths(early: paths.EarlyRunPaths) -> dict[str, str | None]:
    """Everything the early resolver publishes, flat, for records/manifests/diagnostics.

    ``run_dir``, ``log_path`` and the four storage roles -- and NO tmp key of any spelling. A test
    asserts that absence directly, because the failure it guards against is a later contributor
    "helpfully" adding one back (section 2.7).
    """
    out: dict[str, str | None] = {
        "run_dir": str(early.run_dir),
        "log_path": str(early.log_path),
    }
    out.update({role.value: value for role, value in storage_roles(early).items()})
    return out


def db_path(source: paths.EarlyRunPaths | t.ProjectBlock, *, warn: bool = True) -> str:
    """DEPRECATED projection of ``active_db_path`` (plan section 2.10), for one transition release.

    A projection, never a second source of truth: it reads ``active_db_path`` off the object it is
    given and cannot drift from it. Callers still spelling ``dirs.db_path`` get a
    ``DeprecationWarning`` pointing at the role that replaced it; pass ``warn=False`` when the
    projection is what is under test.
    """
    if warn:
        warnings.warn(
            "db_path is a deprecated projection of the active_db_path storage role (plan section "
            "2.10) and will be removed after one transition release; read active_db_path instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    return str(source.active_db_path)


def project_block(
    cfg: Any,
    *,
    settings_file: str | os.PathLike[str] | None = None,
    active_state_dir: str | os.PathLike[str] | None = None,
    input_dirs: Sequence[str] | None = None,
    archive_status: t.ArchiveStatus | None = None,
    archived_db_path: str | None = None,
    archive_error: str | None = None,
) -> t.ProjectBlock:
    """The record's ``project`` block for one pipeline activity (plan section 3.2).

    The archive defaults are the section 2.10 table, not a choice: in-place gets ``n/a`` with
    ``archived_db_path`` EQUAL to ``active_db_path`` (there is nothing to stage, so the active file
    is the archive); staged gets ``pending`` with a null path, because a staged run genuinely has no
    archive until its first checkpoint commits and calling that window an error would make every
    cluster run start in a failure state (gate 21).
    """
    early = resolve_run_paths(cfg, settings_file=settings_file, active_state_dir=active_state_dir)
    mode = archive_mode(early)
    in_place = mode is t.ArchiveMode.IN_PLACE
    roles = storage_roles(early)

    if archive_status is None:
        archive_status = t.ArchiveStatus.NOT_APPLICABLE if in_place else t.ArchiveStatus.PENDING
    if archived_db_path is None and in_place:
        archived_db_path = roles[t.StorageRole.ACTIVE_DB_PATH]

    if input_dirs is None:
        if settings_file is None:
            settings_file = getattr(cfg, "source_path", None)
        input_dirs = _resolve_input_dirs(cfg, settings_file)

    return t.ProjectBlock(
        run_name=early.run_name,
        input_dirs=tuple(str(d) for d in input_dirs),
        artifact_dir=str(roles[t.StorageRole.ARTIFACT_DIR]),
        active_state_dir=str(roles[t.StorageRole.ACTIVE_STATE_DIR]),
        active_db_path=str(roles[t.StorageRole.ACTIVE_DB_PATH]),
        archive_mode=mode,
        archive_status=archive_status,
        archive_pointer_path=roles[t.StorageRole.ARCHIVE_POINTER_PATH],
        archived_db_path=archived_db_path,
        run_dir=str(early.run_dir),
        log_path=str(early.log_path),
        archive_error=archive_error,
    )


def _resolve_input_dirs(cfg: Any, settings_file: str | os.PathLike[str] | None) -> list[str]:
    """``project.input.dirs``, absolutized by the same settings-relative rule as the output dir.

    ``Config.resolve_path`` would join the current working directory, which gate 44 forbids for a
    recorded path: the same record is read by a server, a CLI and a spawned child, none of which
    share a CWD.
    """
    raw = _node(_node(cfg, "project"), "input")
    dirs = _node(raw, "dirs") or []
    out: list[str] = []
    for entry in dirs:
        value = _optional_path_value(entry)
        if value is None:
            continue
        resolved = paths.resolve_project_output_dir(settings_file, value)
        if resolved is not None:
            out.append(str(resolved))
    return out


# --------------------------------------------------------------------------------------------- #
# Archive pointer resolution (plan section 2.10, gate 14)
# --------------------------------------------------------------------------------------------- #

def read_archive_pointer(
    path: Path,
    *,
    exists: Callable[[Path], bool] | None = None,
) -> t.ArchivePointerDict:
    """Read ``archive.current.json``. Missing, malformed, or stale all fail precisely (gate 14).

    "Stale" is concrete: the pointer parses but the generation it names is gone. That is the one
    case a reader could paper over by rebuilding a filename from the run name, which section 2.10
    forbids in as many words -- an archive is "never a fixed filename" in staged mode, so an
    unresolvable pointer is an error and never a guess.
    """
    probe = exists if exists is not None else (lambda p: p.exists())
    pointer_path = Path(path)
    try:
        text = pointer_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise t.ArchivePointerError(f"archive pointer is missing: {pointer_path}") from exc
    except OSError as exc:
        raise t.ArchivePointerError(f"archive pointer is unreadable: {pointer_path}: {exc}") from exc

    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise t.ArchivePointerError(f"archive pointer is malformed JSON: {pointer_path}: {exc}") from exc
    if not isinstance(payload, MappingABC):
        raise t.ArchivePointerError(
            f"archive pointer is malformed: {pointer_path} did not contain a JSON object"
        )

    named = payload.get("archived_db_path")
    if not isinstance(named, str) or not named:
        raise t.ArchivePointerError(
            f"archive pointer is malformed: {pointer_path} names no archived_db_path"
        )
    if not probe(Path(named)):
        raise t.ArchivePointerError(
            f"archive pointer is stale: {pointer_path} names {named}, which no longer exists"
        )
    return dict(payload)  # type: ignore[return-value]


def resolve_archived_db_path(
    project: t.ProjectBlock,
    *,
    pointer_reader: Callable[[Path], t.ArchivePointerDict] | None = None,
) -> str | None:
    """Where the GUI should look for this run's database after finalization (section 2.10).

    * in-place -> ``active_db_path``; there is no pointer and no copying (gate 18);
    * staged + ``pending``/``failed`` -> ``None``: no snapshot has ever committed, and offering a
      broken link is worse than reporting honestly that recovery data is unavailable;
    * staged + ``ready`` -> whatever the pointer currently names, resolved THROUGH the pointer;
    * staged + ``stale`` -> the last good generation already recorded on the block. The run has
      ended and its final checkpoint failed; re-reading the pointer would not improve the answer.
    """
    if project.archive_mode is t.ArchiveMode.IN_PLACE:
        return project.active_db_path
    if project.archive_status in (t.ArchiveStatus.PENDING, t.ArchiveStatus.FAILED):
        return None
    if project.archive_status is t.ArchiveStatus.STALE:
        return project.archived_db_path
    if project.archive_pointer_path is None:
        raise t.ArchivePointerError(
            "staged run reports archive_status 'ready' but carries no archive_pointer_path; the "
            "pointer is mandatory in staged mode"
        )
    read = pointer_reader if pointer_reader is not None else read_archive_pointer
    pointer = read(Path(project.archive_pointer_path))
    return str(pointer["archived_db_path"])


# --------------------------------------------------------------------------------------------- #
# The immutable launch manifest (plan section 3.4) -- the pure builder half
# --------------------------------------------------------------------------------------------- #

def launch_manifest_path(early: paths.EarlyRunPaths) -> Path:
    """``<run>/logs/run_manifest.json``, derived from ``log_path``'s directory.

    Deriving it from ``log_path`` rather than re-joining ``run_dir`` keeps the manifest wherever the
    log lives, which is under ``artifact_dir`` -- section 3.4's cluster requirement, satisfied by
    construction rather than by a second rule.
    """
    return Path(early.log_path).parent / LAUNCH_MANIFEST_FILENAME


def lm3_version() -> str:
    """The installed LM3 version, or ``"unknown"`` when running from a checkout without metadata."""
    try:
        from importlib.metadata import version  # noqa: PLC0415 - only paid on a real launch

        return version("leafmachine3")
    except Exception:  # noqa: BLE001 - a missing dist must not break a launch
        return "unknown"


def _utc_timestamp(clock: Callable[[], float] | None = None) -> str:
    """ISO-8601 UTC, ``...Z``, identical to ``records.utc_now``.

    Delegates to ``records`` when that sibling module is importable so there is one formatter, and
    falls back to the same ``strftime`` when it is not -- Step 2's modules land independently and
    this one must not become unimportable because another has not been written yet.
    """
    try:
        from .records import utc_now  # noqa: PLC0415 - sibling module, deliberately lazy
    except ImportError:
        pass
    else:
        return utc_now(clock)
    now = (clock or time.time)()
    return datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_launch_manifest(
    cfg: Any,
    *,
    run_id: str,
    launcher: t.Launcher | str,
    early: paths.EarlyRunPaths,
    tmp_dir: str | os.PathLike[str],
    parent_run_id: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    config: t.ConfigRef | None = None,
    input_dirs: Sequence[str] | None = None,
    settings_file: str | os.PathLike[str] | None = None,
    started_at: str | None = None,
    versions: Mapping[str, str] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Build the section 3.4 manifest mapping. Pure: nothing is written and nothing is created.

    ``tmp_dir`` is a REQUIRED argument and has no default, which is the ordering rule expressed as a
    signature: the manifest is written after ``build_dirs()`` has settled the tmp location, so a
    caller that has not run ``build_dirs()`` yet cannot produce a manifest by accident. The early
    resolver still publishes no tmp of any kind (section 2.7).

    Wiring this into ``build_dirs()`` is Step 3; Step 2 delivers the builder alone.
    """
    ref = config if config is not None else config_ref(cfg, path=settings_file)
    project_paths = resolved_paths(early)
    project_paths["tmp_dir"] = str(Path(tmp_dir))
    if input_dirs is None:
        input_dirs = _resolve_input_dirs(cfg, settings_file if settings_file is not None
                                         else getattr(cfg, "source_path", None))

    return {
        "schema_version": t.SCHEMA_VERSION,
        "run_id": str(run_id),
        "parent_run_id": str(parent_run_id) if parent_run_id is not None else None,
        "config": {"path": ref.path, "sha256": ref.sha256},
        "effective_config": config_to_dict(cfg),
        "overrides": jsonable(dict(overrides or {})),
        "project": {
            "run_name": early.run_name,
            "input_dirs": [str(d) for d in input_dirs],
            "archive_mode": archive_mode(early).value,
            **project_paths,
        },
        "launcher": launcher.value if isinstance(launcher, t.Launcher) else str(launcher),
        "versions": dict(versions) if versions is not None else {
            "lm3": lm3_version(),
            "python": platform_module.python_version(),
            "platform": f"{sys.platform}-{platform_module.machine()}",
        },
        "started_at": started_at if started_at is not None else _utc_timestamp(clock),
    }


__all__ = [
    "CLUSTER_STATE_DIR_KEY",
    "LAUNCH_MANIFEST_FILENAME",
    "archive_mode",
    "build_launch_manifest",
    "canonical_json",
    "config_json",
    "config_ref",
    "config_sha256",
    "config_to_dict",
    "db_path",
    "launch_manifest_path",
    "lm3_version",
    "project_block",
    "read_archive_pointer",
    "resolve_archived_db_path",
    "resolve_run_paths",
    "resolved_paths",
    "storage_roles",
]
