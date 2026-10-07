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
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from leafmachine3.modelhub.registry import Action, Lock, LockFile, Unit, load_lock

log = logging.getLogger("leafmachine3.modelhub")

BACKUP_SUFFIX = ".backup"
RECORD_NAME = "installed.json"
DOWNLOADS_DIR = ".downloads"
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


def write_record(root: Path, data: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = _record_path(root).with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, _record_path(root))


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
    # scratch left by a dead process
    scratch = root / DOWNLOADS_DIR
    if scratch.is_dir():
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
                # adopt a hand-placed (or re-stat'd) file that is byte-identical to the pinned one
                rec = rec or {"revisions": {}, "files": {}}
                rec.setdefault("revisions", {})[unit.repo_id] = unit.revision
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

    try:
        for action in todo:
            _install_action(root, action, formats, fetch, progress, record)
    finally:
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
