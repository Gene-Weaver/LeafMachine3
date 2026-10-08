"""Download, verify, back up, replace, roll back.

Layout on disk (``root`` = :func:`models_root`)::

    <root>/<dest files from the lock>          e.g. archival_detector/model.onnx
    <root>/installed.json                      what was installed, from which repo/commit, with sha256
    <root>/<dest>.backup                       ONLY while an update is in flight (see below)
    <root>/.downloads/<action>/                scratch for in-flight downloads, removed afterwards

Update protocol, per ACTION (all of an ensemble's files move together):

1. every file is downloaded into ``.downloads/<action>/`` and its sha256 checked against the lock;
2. only then is each existing destination renamed to ``<dest>.backup`` and the new file moved in;
3. once every file of the action is in place the backups are deleted and ``installed.json`` written.

Any exception between 2 and 3 restores every ``.backup`` over its destination before re-raising, so
the folder is never left half-updated. If the process dies instead, the stranded ``.backup`` files are
found by :func:`repair`, which :func:`status` and :func:`install` both call first: a backup whose
destination is missing or fails its hash is restored; a backup whose destination is already the
pinned file is simply removed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from leafmachine3.modelhub.registry import META_FORMAT, Action, Lock, LockFile, Unit, load_lock

log = logging.getLogger("leafmachine3.modelhub")

BACKUP_SUFFIX = ".backup"
RECORD_NAME = "installed.json"
DOWNLOADS_DIR = ".downloads"
SCRATCH_STALE_S = 3600          # a .downloads dir untouched this long belongs to a dead install
_INSTALLING = threading.Event()  # set while install() runs in this process: repair() leaves scratch alone
ENV_ROOT = "LM3_MODELS_DIR"

Progress = Callable[[dict[str, Any]], None]


class InstallError(RuntimeError):
    """A download or verification failed; the models folder was rolled back."""


# --------------------------------------------------------------------------- #
# where the models live
# --------------------------------------------------------------------------- #
def models_root(settings_path: str | os.PathLike[str] | None = None,
                env: Optional[dict[str, str]] = None) -> Path:
    """``$LM3_MODELS_DIR`` if set, else ``models/`` beside the settings file (the checkout's)."""
    e = os.environ if env is None else env
    raw = e.get(ENV_ROOT)
    if raw:
        return Path(raw).expanduser()
    if settings_path is None:
        from leafmachine3.core.paths import settings_path as _resolve  # noqa: PLC0415

        settings_path = _resolve(seed=False)
    return Path(settings_path).expanduser().resolve().parent / "models"


# --------------------------------------------------------------------------- #
# the installed record
# --------------------------------------------------------------------------- #
def _record_path(root: Path) -> Path:
    return root / RECORD_NAME


def read_record(root: Path) -> dict[str, Any]:
    p = _record_path(root)
    if not p.is_file():
        return {"schema_version": 1, "actions": {}}
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("not a mapping")
        data.setdefault("actions", {})
        return data
    except Exception as exc:  # noqa: BLE001 - a corrupt record is just "nothing recorded"
        log.warning("models: unreadable %s (%s) -- treating as empty", p, exc)
        return {"schema_version": 1, "actions": {}}


_RECORD_LOCK = threading.Lock()


def write_record(root: Path, data: dict[str, Any]) -> None:
    """Atomic replace through a temp file that is unique per call.

    status()/catalog() may adopt files and write the record from several server threads at once
    (every open GUI polls). A fixed ``installed.json.tmp`` let one call's os.replace consume the
    temp file the other had just written, and the second replace died with ENOENT -- which the GUI
    showed as "could not read model status". The lock serializes writers in this process; the
    unique name keeps a second process (the CLI) from tripping over them.
    """
    root.mkdir(parents=True, exist_ok=True)
    with _RECORD_LOCK:
        fd, tmp = tempfile.mkstemp(dir=str(root), prefix=".installed.", suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
            os.replace(tmp, _record_path(root))
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# crash recovery
# --------------------------------------------------------------------------- #
def repair(root: Path, lock: Lock | None = None) -> list[dict[str, str]]:
    """Resolve every stranded ``*.backup`` under ``root``; returns what was done."""
    root = Path(root)
    if not root.is_dir():
        return []
    lock = lock or load_lock()
    expected = {f.dest: f.sha256 for a in lock.all_actions() for _u, f in a.files(_all_formats(lock))}
    done: list[dict[str, str]] = []
    for backup in sorted(root.rglob(f"*{BACKUP_SUFFIX}")):
        if not backup.is_file():
            continue
        dest = backup.with_name(backup.name[: -len(BACKUP_SUFFIX)])
        rel = dest.relative_to(root).as_posix()
        want = expected.get(rel)
        dest_ok = dest.is_file() and (want is None or sha256_of(dest) == want)
        if dest_ok:
            backup.unlink()
            done.append({"file": rel, "action": "discarded_backup"})
        else:
            if dest.exists():
                dest.unlink()
            os.replace(backup, dest)
            done.append({"file": rel, "action": "restored_backup"})
            log.warning("models: restored %s from its backup (an update did not finish)", rel)
    # scratch left by a dead process. NOT the scratch of an install in flight: status()/catalog()
    # are polled by every open GUI while a download runs, and sweeping here used to delete the
    # half-written file from under hf_hub_download. In-process installs are flagged; another
    # process's scratch is only swept once it has gone quiet for SCRATCH_STALE_S.
    scratch = root / DOWNLOADS_DIR
    if scratch.is_dir() and not _INSTALLING.is_set():
        try:
            newest = max((p.stat().st_mtime for p in scratch.rglob("*") if p.is_file()), default=scratch.stat().st_mtime)
        except OSError:
            newest = 0.0
        if time.time() - newest > SCRATCH_STALE_S:
            shutil.rmtree(scratch, ignore_errors=True)
    return done


def _all_formats(lock: Lock) -> set[str]:
    return {f.format for a in lock.all_actions() for u in a.units for f in u.files}


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
STATE_MISSING = "missing"          # at least one required file absent
STATE_CURRENT = "current"          # every file present and matches the lock
STATE_OUTDATED = "outdated"        # present, but the lock pins a different revision/sha
STATE_PENDING = "pending"          # placeholder action: present locally, not on the Hub yet
STATE_UNAVAILABLE = "unavailable"  # placeholder action and not present locally either


@dataclass
class FileStatus:
    dest: str
    format: str
    present: bool
    bytes: int | None
    expected_bytes: int | None
    sha_ok: bool | None          # None = not checked / unknown
    optional: bool = False


@dataclass
class ActionStatus:
    action: str
    state: str
    required: bool
    placeholder: bool
    repos: list[str]
    installed_revision: dict[str, str | None] = field(default_factory=dict)
    lock_revision: dict[str, str | None] = field(default_factory=dict)
    files: list[FileStatus] = field(default_factory=list)
    detail: str = ""


def _stat_sig(p: Path) -> dict[str, Any]:
    st = p.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _record_matches(rec: dict[str, Any] | None, unit: Unit, f: LockFile, p: Path | None = None) -> bool:
    """True when the record says ``f`` is the pinned file AND the file on disk is still the one recorded."""
    if not rec:
        return False
    files = rec.get("files") or {}
    got = files.get(f.dest) or {}
    if got.get("sha256") != f.sha256:
        return False
    revs = rec.get("revisions") or {}
    if revs.get(unit.repo_id) != unit.revision:
        return False
    if p is not None:
        sig = got.get("stat")
        if not sig or sig != _stat_sig(p):
            return False
    return True


def _action_status(root: Path, action: Action, rec: dict[str, Any] | None, formats: Sequence[str],
                   verify_hashes: bool) -> tuple[ActionStatus, dict[str, Any] | None, bool]:
    """One action's status; returns ``(status, record_entry, record_changed)``."""
    files: list[FileStatus] = []
    any_missing = any_stale = dirty = False
    for unit, f in action.files(formats):
        p = root / f.dest
        present = p.is_file()
        fs = FileStatus(dest=f.dest, format=f.format, present=present, bytes=p.stat().st_size if present else None,
                        expected_bytes=f.bytes, sha_ok=None, optional=f.optional)
        if not present:
            any_missing = any_missing or not f.optional
            files.append(fs)
            continue
        if action.placeholder:
            fs.sha_ok = (sha256_of(p) == f.sha256) if (verify_hashes and f.sha256) else None
        elif _record_matches(rec, unit, f, p) and not verify_hashes:
            fs.sha_ok = True
        else:
            ok = sha256_of(p) == f.sha256
            fs.sha_ok = ok
            if ok and not _record_matches(rec, unit, f, p):
                # adopt a hand-placed (or re-stat'd) file that is byte-identical to the pinned one.
                # The unit's recorded REVISION is left alone when one exists: the record keeps one
                # revision per repo, and a matching sidecar must not relabel an older model file as
                # the pinned revision (that is what the update dialog shows as "installed").
                rec = rec or {"revisions": {}, "files": {}}
                rec.setdefault("revisions", {}).setdefault(unit.repo_id, unit.revision)
                rec.setdefault("files", {})[f.dest] = {"sha256": f.sha256, "bytes": f.bytes, "src": f.src,
                                                       "repo_id": unit.repo_id, "stat": _stat_sig(p)}
                rec["installed_at"] = rec.get("installed_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z")
                dirty = True
            if not ok:
                any_stale = True
        files.append(fs)
    if action.placeholder:
        state = STATE_UNAVAILABLE if any_missing else STATE_PENDING
        detail = "not published on the Hub yet" + ("" if any_missing else "; using the local copy")
    elif any_missing:
        state = STATE_MISSING
        detail = f"{sum(1 for x in files if not x.present and not x.optional)} of {sum(1 for x in files if not x.optional)} files missing"
    elif any_stale:
        state, detail = STATE_OUTDATED, "the lock pins a newer revision"
    else:
        state, detail = STATE_CURRENT, ""
    st = ActionStatus(
        action=action.key, state=state, required=action.required, placeholder=action.placeholder,
        repos=[u.repo_id for u in action.units],
        installed_revision={u.repo_id: ((rec or {}).get("revisions") or {}).get(u.repo_id) for u in action.units},
        lock_revision={u.repo_id: u.revision for u in action.units}, files=files, detail=detail)
    return st, rec, dirty


def status(root: Path | None = None, *, lock: Lock | None = None, formats: Sequence[str] | None = None,
           verify_hashes: bool = False) -> dict[str, Any]:
    """Per-action state plus a summary the GUI/CLI switch on.

    Without ``verify_hashes`` a file counts as current when ``installed.json`` records the pinned
    sha for it; a file with no record (installed by hand, or before the installer existed) is hashed
    once and, if it matches, recorded so the next call is free.
    """
    lock = lock or load_lock()
    root = Path(root) if root is not None else models_root()
    formats = tuple(formats or lock.default_formats)
    repaired = repair(root, lock)
    record = read_record(root)
    rec_actions: dict[str, Any] = record.get("actions") or {}
    out: dict[str, ActionStatus] = {}
    record_dirty = False

    for key, action in lock.actions.items():
        st, rec, dirty = _action_status(root, action, rec_actions.get(key), formats, verify_hashes)
        out[key] = st
        if dirty:
            rec_actions[key] = rec
            record_dirty = True
    # alternates: reported only once something of theirs is on disk; never part of the summary
    alt_out: dict[str, ActionStatus] = {}
    for action in (a for ks in lock.alternates.values() for a in ks.values()):
        if not any((root / f.dest).is_file() for _u, f in action.files(formats)):
            continue
        st, rec, dirty = _action_status(root, action, rec_actions.get(action.key), formats, verify_hashes)
        alt_out[action.key] = st
        if dirty:
            rec_actions[action.key] = rec
            record_dirty = True

    if record_dirty:
        record["actions"] = rec_actions
        write_record(root, record)

    states = [a.state for a in out.values() if a.required]
    missing = [k for k, a in out.items() if a.required and a.state == STATE_MISSING]
    outdated = [k for k, a in out.items() if a.required and a.state == STATE_OUTDATED]
    unavailable = [k for k, a in out.items() if a.required and a.state == STATE_UNAVAILABLE]
    if missing:
        label = "Install Models from Hugging Face"
    elif outdated:
        label = "Newer Models are Available"
    else:
        label = "Models are up to date"
    return {
        "root": str(root), "lock_path": lock.path, "lm3_version": lock.lm3_version, "formats": list(formats),
        "actions": {k: asdict(v) for k, v in out.items()},
        "alternates": {k: asdict(v) for k, v in alt_out.items()},
        "summary": {"missing": missing, "outdated": outdated, "unavailable": unavailable,
                    # needs_attention = something the install button can fix; tab_attention also counts
                    # a required model that is not published yet and has no local copy.
                    "needs_attention": bool(missing or outdated),
                    "tab_attention": bool(missing or outdated or unavailable),
                    "all_present": not missing and all(s != STATE_UNAVAILABLE for s in states),
                    "button_label": label},
        "repaired": repaired,
    }


def verify(root: Path | None = None, **kw: Any) -> dict[str, Any]:
    """:func:`status` with every present file re-hashed against the lock."""
    return status(root, verify_hashes=True, **kw)


# --------------------------------------------------------------------------- #
# install
# --------------------------------------------------------------------------- #
def _download(unit: Unit, f: LockFile, scratch: Path, progress: Progress) -> Path:
    """Fetch one file from the Hub into ``scratch``; patched in tests."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError  # noqa: PLC0415

    try:
        got = hf_hub_download(repo_id=unit.repo_id, filename=f.src, revision=unit.revision,
                              local_dir=str(scratch), force_download=True)
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise InstallError(
            f"{unit.repo_id} is not accessible ({type(exc).__name__}). The LM3 model repos are private for "
            f"now: log in with `hf auth login` (or set HF_TOKEN) using an account that can read them.") from exc
    except Exception as exc:  # noqa: BLE001 - network, disk, proxy...
        raise InstallError(f"download failed for {unit.repo_id}:{f.src}: {exc}") from exc
    return Path(got)


def _emit(progress: Optional[Progress], **ev: Any) -> None:
    if progress:
        try:
            progress(ev)
        except Exception:  # noqa: BLE001 - a UI callback must never break the install
            log.exception("models: progress callback raised")


def install(root: Path | None = None, *, lock: Lock | None = None, actions: Iterable[str] | None = None,
            formats: Sequence[str] | None = None, force: bool = False,
            progress: Optional[Progress] = None, downloader: Callable[..., Path] | None = None,
            models: Iterable[tuple[str, str]] | None = None) -> dict[str, Any]:
    """Bring ``root`` up to the lock. Returns the post-install :func:`status`.

    ``force`` re-downloads actions that are already current. ``downloader`` replaces the Hub fetch
    (tests). Placeholder actions are always skipped with a ``skip`` event. ``models`` installs
    alternates instead, as ``(stage, model_key)`` pairs (e.g. ``("specimen_segmenter",
    "yolo26x_seg_1280")``); when it is given, the defaults are left alone unless ``actions`` also is.
    """
    lock = lock or load_lock()
    root = Path(root) if root is not None else models_root()
    formats = tuple(formats or lock.default_formats)
    fetch = downloader or _download
    root.mkdir(parents=True, exist_ok=True)

    before = status(root, lock=lock, formats=formats)
    models = list(models or [])
    wanted = list(actions) if actions else ([] if models else list(lock.actions))
    todo: list[Action] = []
    for stage, model_key in models:
        alt = lock.alternate(stage, model_key)
        st = (before.get("alternates") or {}).get(alt.key, {}).get("state")
        if st == STATE_CURRENT and not force:
            _emit(progress, type="skip", action=alt.key, reason="current")
            continue
        todo.append(alt)
    for key in wanted:
        action = lock.action(key)
        st = before["actions"][key]["state"]
        if action.placeholder:
            _emit(progress, type="skip", action=key, reason="placeholder: not on the Hub yet")
            continue
        if st == STATE_CURRENT and not force:
            _emit(progress, type="skip", action=key, reason="current")
            continue
        todo.append(action)

    total_bytes = sum((f.bytes or 0) for a in todo for _u, f in a.files(formats))
    _emit(progress, type="start", root=str(root), actions=[a.key for a in todo], total_bytes=total_bytes)
    record = read_record(root)

    _INSTALLING.set()
    try:
        for action in todo:
            _install_action(root, action, formats, fetch, progress, record)
    finally:
        _INSTALLING.clear()
        shutil.rmtree(root / DOWNLOADS_DIR, ignore_errors=True)     # scratch never outlives an install

    after = status(root, lock=lock, formats=formats)
    _emit(progress, type="done", summary=after["summary"])
    return after


def _install_action(root: Path, action: Action, formats: Sequence[str], fetch: Callable[..., Path],
                    progress: Optional[Progress], record: dict[str, Any]) -> None:
    scratch = root / DOWNLOADS_DIR / action.key
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    pairs = list(action.files(formats))
    staged: list[tuple[Unit, LockFile, Path]] = []
    try:
        # 1. download + verify everything first; nothing in place is touched yet
        for idx, (unit, f) in enumerate(pairs):
            _emit(progress, type="file", action=action.key, file=f.dest, repo_id=unit.repo_id, bytes=f.bytes, phase="download")
            # One scratch dir PER FILE: ensemble members are separate repos that all ship e.g.
            # ``onnx/model.onnx``, and hf_hub_download(local_dir=...) writes the repo path verbatim.
            unit_scratch = scratch / f"f{idx:02d}"
            unit_scratch.mkdir(parents=True, exist_ok=True)
            got = fetch(unit, f, unit_scratch, progress)
            if f.sha256:
                digest = sha256_of(got)
                if digest != f.sha256:
                    raise InstallError(f"{unit.repo_id}:{f.src} hash mismatch (expected {f.sha256[:12]}, got {digest[:12]}); "
                                       f"the download was discarded")
            staged.append((unit, f, got))
            _emit(progress, type="file_done", action=action.key, file=f.dest, phase="download")
        # 2. swap in, keeping a .backup of whatever was there
        replaced: list[tuple[Path, Path]] = []
        try:
            for unit, f, got in staged:
                dest = root / f.dest
                dest.parent.mkdir(parents=True, exist_ok=True)
                backup = dest.with_name(dest.name + BACKUP_SUFFIX)
                if dest.exists():
                    if backup.exists():
                        backup.unlink()
                    os.replace(dest, backup)
                    replaced.append((dest, backup))
                _move(got, dest)
                _emit(progress, type="file_done", action=action.key, file=f.dest, phase="install")
        except BaseException:
            for dest, backup in replaced:            # roll back in place, newest first
                try:
                    if dest.exists():
                        dest.unlink()
                    os.replace(backup, dest)
                except OSError:
                    log.exception("models: could not restore %s from %s", dest, backup)
            raise
        # 3. commit: drop backups, record what is installed
        for _dest, backup in replaced:
            try:
                backup.unlink()
            except OSError:
                pass
        rec = record.setdefault("actions", {}).setdefault(action.key, {})
        rec["revisions"] = {u.repo_id: u.revision for u in action.units}
        rec.setdefault("files", {})
        for unit, f, _got in staged:
            rec["files"][f.dest] = {"sha256": f.sha256, "bytes": f.bytes, "src": f.src, "repo_id": unit.repo_id,
                                    "stat": _stat_sig(root / f.dest)}
        rec["installed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_record(root, record)
        _emit(progress, type="action_done", action=action.key)
    except InstallError:
        _emit(progress, type="error", action=action.key, message="install failed; previous files restored")
        raise
    except Exception as exc:
        _emit(progress, type="error", action=action.key, message=str(exc))
        raise InstallError(f"{action.key}: {exc}") from exc
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _move(src: Path, dest: Path) -> None:
    try:
        os.replace(src, dest)
    except OSError:                       # scratch on another filesystem
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
        src.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# catalog: every stage, every variant, every format -- what the Models tab renders
# --------------------------------------------------------------------------- #
#: Formats the LM3 runtime can execute. The lock may pin more (CoreML, OpenVINO, TorchScript,
#: PyTorch) for users who run the models elsewhere; only these can be made the ACTIVE model.
RUNNABLE_FORMATS: tuple[str, ...] = ("onnx", "json")

FMT_MISSING = "missing"      # none of the format's files is on disk
FMT_PARTIAL = "partial"      # some are (a multi-file format half-installed, or a crash mid-swap)
FMT_OUTDATED = "outdated"    # on disk, but not the pinned revision/sha
FMT_CURRENT = "current"      # on disk and matches the lock


def _stage_order() -> list[str]:
    """Pipeline order for the catalog; the lock's own order is the fallback."""
    try:
        from leafmachine3.core.config import CANONICAL_STAGE_KEYS  # noqa: PLC0415

        return list(CANONICAL_STAGE_KEYS)
    except Exception:  # noqa: BLE001 - the catalog must not depend on the pipeline importing
        return []


def _settings_model(stage: str, settings_values: Mapping[str, Any] | None) -> dict[str, Any]:
    """``modules.<stage>.model`` (plus ``models_dir`` for the ensemble) as the settings tree has it."""
    try:
        block = ((settings_values or {}).get("modules") or {}).get(stage) or {}
        model = block.get("model") if isinstance(block, Mapping) else None
        out = dict(model) if isinstance(model, Mapping) else {}
        if isinstance(block, Mapping) and block.get("models_dir"):
            out.setdefault("models_dir", block["models_dir"])
        return out
    except Exception:  # noqa: BLE001
        return {}


def _norm_model_path(root: Path, p: str | None) -> str | None:
    """A settings model path as a dest relative to the models root, or None if it points elsewhere."""
    if not p:
        return None
    path = Path(str(p)).expanduser()
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return None
    parts = path.parts
    if parts and parts[0] == "models":
        return Path(*parts[1:]).as_posix()
    return path.as_posix()


def _variant_entry(root: Path, action: Action, rec: dict[str, Any] | None, *, default: bool,
                   active_dest: str | None) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    """One variant (a default action or an alternate): per-unit, per-format install state."""
    units: list[dict[str, Any]] = []
    dirty = False
    by_dest = {f.dest: u for u in action.units for f in u.files}
    # runnable formats first (onnx is what LM3 runs), the rest alphabetically
    formats = sorted({f.format for u in action.units for f in u.files if f.format != META_FORMAT},
                     key=lambda f: (f not in RUNNABLE_FORMATS, f))
    # one _action_status per format reuses the record/adoption logic, so a hand-placed file that
    # matches the lock is recorded here exactly as status() would record it
    per_format: dict[str, list[FileStatus]] = {}
    for fmt in formats:
        st, rec, d = _action_status(root, action, rec, (fmt,), False)
        dirty = dirty or d
        per_format[fmt] = [fs for fs in st.files if fs.format == fmt]
    rec_revs = ((rec or {}).get("revisions") or {})
    for u in action.units:
        fmts: dict[str, Any] = {}
        for fmt in formats:
            mine = [fs for fs in per_format[fmt] if by_dest.get(fs.dest) is u]
            if not mine:
                continue
            required = [fs for fs in mine if not fs.optional]
            present = [fs for fs in required if fs.present]
            if not present:
                state = FMT_MISSING
            elif len(present) < len(required):
                state = FMT_PARTIAL
            elif any(fs.sha_ok is False for fs in present):
                state = FMT_OUTDATED
            else:
                state = FMT_CURRENT
            # the file the settings point at when this (unit, format) is active: the single
            # non-optional file of a runnable format
            runtime_file = required[0].dest if (fmt in RUNNABLE_FORMATS and len(required) == 1) else None
            fmts[fmt] = {
                "state": state,
                "bytes": sum((fs.expected_bytes or 0) for fs in required),
                "present_bytes": sum((fs.bytes or 0) for fs in present),
                "files": len(required),
                "runtime_file": runtime_file,
                "runnable": fmt in RUNNABLE_FORMATS,
                "active": bool(runtime_file and active_dest and runtime_file == active_dest),
            }
        # an "installed" revision equal to the pinned one while a format is outdated is a stale
        # record (an older install adopted by sha, or hand-placed files): report it as unknown
        installed = rec_revs.get(u.repo_id)
        if installed == u.revision and any(f["state"] == FMT_OUTDATED for f in fmts.values()):
            installed = None
        units.append({"repo_id": u.repo_id, "revision": u.revision, "model_key": u.model_key,
                      "installed_revision": installed, "formats": fmts})
    entry = {"model_key": action.model_key or (action.units[0].model_key if action.units else ""),
             "default": default, "action": action.key, "settings": dict(action.settings), "units": units}
    return entry, rec, dirty


def catalog(root: Path | None = None, *, lock: Lock | None = None,
            settings_values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Everything the Models tab shows: per stage, every variant the lock knows, every format of
    every unit with its install state, which (variant, format) the settings make active, and
    whether the stage's model may be swapped at all.

    Nothing here is stage-specific: a new stage, variant or format appears by regenerating the
    lock. ``settings_values`` is the raw settings tree (``modules.<stage>.model`` is the truth about
    what is active); without it the active column is empty.
    """
    lock = lock or load_lock()
    root = Path(root) if root is not None else models_root()
    repair(root, lock)
    record = read_record(root)
    rec_actions: dict[str, Any] = record.get("actions") or {}
    dirty = False
    # the roll-up state of the required default, as status() reports it (same vocabulary as the banner)
    summary = status(root, lock=lock)
    order = _stage_order()
    stages_in_lock = list(lock.actions)
    ordered = [k for k in order if k in lock.actions] + [k for k in stages_in_lock if k not in order]
    stages: list[dict[str, Any]] = []
    for stage in ordered:
        action = lock.action(stage)
        model = _settings_model(stage, settings_values)
        active_dest = _norm_model_path(root, model.get("path"))
        variants: list[dict[str, Any]] = []
        for default, a in [(True, action)] + [(False, alt) for alt in (lock.alternates.get(stage) or {}).values()]:
            entry, rec, d = _variant_entry(root, a, rec_actions.get(a.key), default=default, active_dest=active_dest)
            if d:
                rec_actions[a.key] = rec
                dirty = True
            variants.append(entry)
        matched = next(((v["model_key"], u["repo_id"], fmt, f["state"]) for v in variants for u in v["units"]
                        for fmt, f in u["formats"].items() if f["active"]), None)
        st = summary["actions"].get(stage) or {}
        # The module's own state follows the model it RUNS: an up-to-date alternate makes the module
        # green even while its default has an update waiting (the header line still says so).
        # A settings path the lock does not know, or a locked stage, falls back to the default's state.
        active_state = matched[3] if (matched and action.activatable) else st.get("state")
        stages.append({
            "stage": stage,
            "activatable": bool(action.activatable),
            "state": st.get("state"),
            "active_state": active_state,
            "detail": st.get("detail", ""),
            "active": {"key": model.get("key"), "path": model.get("path"), "format": model.get("format"),
                       "model_key": matched[0] if matched else None, "repo_id": matched[1] if matched else None,
                       "matched_format": matched[2] if matched else None,
                       # the ensemble has no single path: its folder is the thing
                       "models_dir": model.get("models_dir"),
                       "matched": bool(matched) or (not action.activatable and not model.get("path"))},
            "variants": variants,
        })
    if dirty:
        record["actions"] = rec_actions
        write_record(root, record)
    return {"root": str(root), "lock_path": lock.path, "lm3_version": lock.lm3_version,
            "default_formats": list(lock.default_formats), "runnable_formats": list(RUNNABLE_FORMATS),
            "summary": summary["summary"], "stages": stages}


def activation_plan(lock: Lock, stage: str, model_key: str, fmt: str, root: Path,
                    defaults: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What activating ``(stage, model_key, fmt)`` writes into ``modules.<stage>``.

    Returns ``{"model": {key, path, format}, "set": {...}, "restore": {...}, "unset": [...]}``:
    ``set`` are the chosen variant's stage settings (an input size, say); ``restore`` the default
    variant's values for keys some other variant sets but this one does not; ``unset`` the keys
    nobody has a value for, which the caller returns to the built-in default or removes.
    Raises ``InstallError`` when the stage is locked, the variant/format is unknown, the format is
    not runnable, or the file is not installed -- the settings file is never pointed at a file that
    is not there.
    """
    action = lock.action(stage)
    if not action.activatable:
        raise InstallError(f"{stage} always runs its default model; it cannot be switched")
    chosen: Action | None = None
    if action.model_key == model_key or (action.units and action.units[0].model_key == model_key):
        chosen = action
    else:
        try:
            chosen = lock.alternate(stage, model_key)
        except KeyError as exc:
            raise InstallError(str(exc)) from None
    if fmt not in RUNNABLE_FORMATS:
        raise InstallError(f"{fmt} is not a format LM3 can run (runnable: {', '.join(RUNNABLE_FORMATS)})")
    files = [f for _u, f in chosen.files((fmt,)) if f.format == fmt and not f.optional]
    if len(files) != 1:
        raise InstallError(f"{stage}={model_key} has no single {fmt} runtime file in the lock")
    dest = files[0].dest
    if not (root / dest).is_file():
        raise InstallError(f"{stage}={model_key} ({fmt}) is not installed; install it first")
    # keys any alternate of this stage may set; the chosen variant sets its own, the rest go back
    # to the built-in default so switching back to the default model really is the default
    all_keys = {k for alt in (lock.alternates.get(stage) or {}).values() for k in alt.settings} | set(action.settings)
    set_: dict[str, Any] = dict(chosen.settings)
    restore = {k: v for k, v in action.settings.items() if k not in set_}
    unset = sorted(k for k in all_keys if k not in set_ and k not in restore)
    return {"model": {"key": model_key, "path": f"models/{dest}", "format": fmt}, "set": set_, "restore": restore,
            "unset": unset}
