"""leafmachine3.server.settings_api -- read / validate / write ``LM3_settings.yaml`` over HTTP.

The GUI's Settings tab edits the SAME file the CLI reads, so every write has to leave a file
that :class:`leafmachine3.core.config.Config` can load and validate. The contract here is
therefore strict in one direction only:

  * a write is validated FIRST, against a throwaway copy, and is refused outright when
    ``cfg.validate()`` raises -- an invalid ``LM3_settings.yaml`` is never written to disk;
  * every write rotates a timestamped backup beside the file (last 5 kept) plus a one-time
    pristine ``<name>.orig.yaml``, because a plain-YAML round-trip discards the comments in
    the shipped file and a GUI user must never be able to lose a working config.

``fastapi`` / ``pydantic`` are OPTIONAL extras, so -- exactly like
:func:`leafmachine3.server.app.create_app` and :func:`leafmachine3.server.metrics.router` --
every heavy import is deferred. Importing this module on a base install is free; the router is
built lazily by :func:`router` / :func:`create_router`.

The plain functions (``read_settings``, ``validate_values``, ``write_settings``, ``browse``,
the preset helpers) are the real implementation and are usable without a web server at all.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat as stat_mod
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

log = logging.getLogger("leafmachine3.server.settings")

# Where the settings file lives. app.py resolves ``LM3_settings.yaml`` relative to the CWD the
# server was started in; we honor the same default plus an explicit env override so a packaged
# Electron shell can point at a user-data copy without a code change.
_SETTINGS_ENV = "LM3_SETTINGS_PATH"
_META_ENV = "LM3_SETTINGS_META"
DEFAULT_SETTINGS_NAME = "LM3_settings.yaml"

#: UI asset directory -- ``settings_meta.json`` is served from here by the static mount too.
UI_DIR = Path(__file__).resolve().parent / "ui"
META_NAME = "settings_meta.json"

#: How many rotating backups to keep beside the settings file.
BACKUP_KEEP = 5

#: Raw YAML text is echoed back for the `.yamlview` preview pane; refuse absurd files.
MAX_TEXT_BYTES = 1 << 19          # 512 KB

# --- folder-browser guard rails ------------------------------------------------------------- #
_BROWSE_MAX_ENTRIES = 500         # directories returned per listing
_BROWSE_SCAN_CAP = 8000           # dirents examined per directory before we call it truncated
_BROWSE_COUNT_CHILDREN = 300      # above this many children we stop counting images per child
_RECURSIVE_WALK_CAP = 40000       # dirents examined when probing a recursive input dir

_SAFE_PRESET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,63}$")

# Config.validate() bullets conveniently START with the offending dotted path, which lets the UI
# paint `.invalid` + `.err` on the exact `.treerow` instead of dumping a blob at the top.
_PATH_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+")
_HAS_NO = re.compile(r"has no ([A-Za-z_][A-Za-z0-9_.]*)")


# pathlib's exists()/is_dir() only swallow ENOENT/ENOTDIR/EBADF/ELOOP -- EACCES propagates, so a
# user pointing a path field at something like /proc/1/root would otherwise take down the whole
# request. Every predicate applied to a user-supplied path goes through these.
def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except OSError:
        return False


def _is_dir(p: Path) -> bool:
    try:
        return p.is_dir()
    except OSError:
        return False


def _is_file(p: Path) -> bool:
    try:
        return p.is_file()
    except OSError:
        return False


class SettingsError(Exception):
    """A write was refused. ``errors`` carries the human bullets, ``detail`` the full payload."""

    def __init__(self, message: str, errors: Optional[list] = None, detail: Optional[dict] = None):
        super().__init__(message)
        self.errors = errors or [message]
        self.detail = detail or {}


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
#: A settings file is a YAML file. Enforced because ``yaml_path`` is caller-supplied over HTTP.
_SETTINGS_SUFFIXES = frozenset({".yaml", ".yml"})


def settings_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the settings file: explicit argument > ``$LM3_SETTINGS_PATH`` > ``./LM3_settings.yaml``.

    Always returned absolute so the UI can display it and so backups/presets land in a
    predictable place regardless of the server's CWD.

    The suffix check is a SECURITY boundary, not tidiness. Every ``yaml_path`` the routes accept
    funnels through here, and the endpoints downstream both read the target's raw bytes back to
    the caller (``read_settings`` echoes them in ``text``) and write to it (``write_settings``,
    ``save_preset``). Without it, ``GET /v1/settings?yaml_path=/etc/passwd`` is an arbitrary-file
    reader and ``PUT /v1/settings`` an arbitrary-file writer for anyone holding the token.
    """
    raw = str(explicit) if explicit else os.environ.get(_SETTINGS_ENV) or DEFAULT_SETTINGS_NAME
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    # normpath (not resolve) so a symlinked settings file is written through, not replaced.
    path = Path(os.path.normpath(str(path)))
    if path.suffix.lower() not in _SETTINGS_SUFFIXES:
        raise SettingsError(
            f"a settings path must name a {' or '.join(sorted(_SETTINGS_SUFFIXES))} file; "
            f"got {raw!r}"
        )
    return path


def meta_path() -> Path:
    """Resolve ``settings_meta.json`` (env override > the UI asset directory)."""
    raw = os.environ.get(_META_ENV)
    return Path(raw).expanduser() if raw else UI_DIR / META_NAME


def presets_dir(path: str | os.PathLike[str] | None = None) -> Path:
    """``<settings dir>/presets`` -- named copies of a whole settings file."""
    return settings_path(path).parent / "presets"


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
def _defaults() -> dict:
    from leafmachine3.core.config import builtin_defaults

    return builtin_defaults()


def _effective(values: dict) -> dict:
    """Defaults deep-merged with the file -- what LM3 will ACTUALLY run.

    Reuses ``config._deep_merge`` so the UI's "effective value" can never drift from the merge
    ``Config.load`` performs. Falls back to a local copy if that private helper ever moves.
    """
    try:
        from leafmachine3.core.config import _deep_merge
    except Exception:  # noqa: BLE001 - keep the endpoint alive if the private helper moves
        def _deep_merge(base: dict, override: Any) -> dict:  # type: ignore[misc]
            out = dict(base)
            for k, v in override.items():
                cur = out.get(k)
                out[k] = _deep_merge(dict(cur), v) if isinstance(cur, dict) and isinstance(v, dict) else v
            return out
    return _deep_merge(_defaults(), values or {})


def load_yaml(path: Path) -> dict:
    """Parse a YAML file to a plain dict. Raises :class:`SettingsError` on garbage."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise SettingsError(f"could not parse {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SettingsError(f"{path} did not parse to a mapping (got {type(data).__name__})")
    return data


def read_settings(path: str | os.PathLike[str] | None = None, *, with_text: bool = True) -> dict:
    """Return the settings payload for ``GET /v1/settings``.

    ``values`` is the file as written (``{}`` when it does not exist yet), ``defaults`` is
    ``builtin_defaults()`` and ``effective`` is the two merged -- the UI needs the merge to render
    rows for keys that only exist in the defaults (the whole ``timing`` block, for instance).
    """
    p = settings_path(path)
    out: dict[str, Any] = {
        "yaml_path": str(p),
        "dir": str(p.parent),
        "exists": _is_file(p),
        "mtime": None,
        "size": None,
        "values": {},
        "defaults": _defaults(),
        "effective": {},
        "text": None,
        "text_truncated": False,
        "readonly": False,
        "error": None,
    }
    if _is_file(p):
        try:
            st = p.stat()
            out["mtime"] = round(st.st_mtime, 6)
            out["size"] = st.st_size
        except OSError as exc:
            out["error"] = str(exc)
        # A read-only file must disable Save in the UI rather than fail at write time.
        out["readonly"] = not os.access(p, os.W_OK)
        try:
            out["values"] = load_yaml(p)
        except SettingsError as exc:
            out["error"] = str(exc)
        if with_text:
            try:
                raw = p.read_bytes()
                out["text_truncated"] = len(raw) > MAX_TEXT_BYTES
                out["text"] = raw[:MAX_TEXT_BYTES].decode("utf-8", "replace")
            except OSError as exc:
                out["error"] = out["error"] or str(exc)
    else:
        # A brand-new install: the parent dir must be writable or Save will never succeed.
        out["readonly"] = not _dir_writable(p.parent)
    out["effective"] = _effective(out["values"])
    out["backups"] = list_backups(p)
    return out


def read_meta() -> dict:
    """Return the parsed ``ui/settings_meta.json`` -- ``{}`` when absent (never raises).

    A corrupt file returns ``{"_error": ...}``: keys starting with ``_`` are inert to the
    renderer (it looks meta up BY dotted path), so the UI can toast the problem and still fall
    back to generic rows instead of silently dropping every setting.
    """
    p = meta_path()
    if not _is_file(p):
        log.warning("settings meta not found: %s", p)
        return {}
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.error("settings meta unreadable (%s): %s", p, exc)
        return {"_error": f"{p.name} is unreadable: {exc}"}
    if not isinstance(data, dict):
        return {"_error": f"{p.name} did not parse to an object"}
    return data


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def dump_yaml(values: dict) -> str:
    """Serialize settings to YAML text.

    ``sort_keys=False`` preserves the order the client sent (JSON objects keep insertion order,
    so that is the order the file was read in). ``default_flow_style=None`` keeps leaf
    collections inline -- ``dirs: [examples/images]``, ``hole_rgb_color: [10, 10, 10]``,
    ``logging: {level: INFO, ...}`` -- which is exactly how the shipped file is written and keeps
    it hand-editable after a GUI round-trip.
    """
    return yaml.safe_dump(
        values,
        sort_keys=False,
        default_flow_style=None,
        allow_unicode=True,
        width=100,
        indent=2,
    )


def _split_errors(message: str) -> list:
    """Explode ``Config.validate``'s multi-line ValueError into one string per bullet."""
    text = str(message)
    if "\n  - " in text:
        _head, _sep, rest = text.partition("\n  - ")
        return [line.strip() for line in rest.split("\n  - ") if line.strip()]
    return [text.strip()]


def _has_path(tree: Any, dotted: str) -> bool:
    node = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


_META_CACHE: dict = {"mtime": None, "paths": frozenset()}


def _meta_paths() -> frozenset:
    """Dotted paths known to ``settings_meta.json``, cached on the file's mtime.

    Used only for error anchoring: a bullet about a key that is MISSING from the YAML still has
    a row in the UI (the renderer draws rows from the metadata too), so anchoring to a known
    metadata path beats falling back to the parent block.
    """
    p = meta_path()
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return frozenset()
    if _META_CACHE["mtime"] != mtime:
        meta = read_meta()
        _META_CACHE["paths"] = frozenset(k for k in meta if not k.startswith("_"))
        _META_CACHE["mtime"] = mtime
    return _META_CACHE["paths"]


def _anchor(message: str, tree: dict, known: Optional[frozenset] = None) -> Optional[str]:
    """Best-effort: map an error bullet onto the dotted setting path it is about."""
    tokens = _PATH_TOKEN.findall(message)
    if not tokens:
        return None
    head = tokens[0]
    candidates: list[str] = []
    # "modules.X is enabled but has no model.path" / "... has no models_dir" -> point at the
    # missing leaf, not just at the module block.
    missing = _HAS_NO.search(message)
    if missing:
        candidates.append(f"{head}.{missing.group(1)}")
    candidates.extend(f"{head}.{tok}" for tok in tokens[1:])
    candidates.append(head)
    for cand in candidates:
        if _has_path(tree, cand) or (known and cand in known):
            return cand
    # Nothing resolved (the key is absent -- often the point of the error): walk up to the
    # nearest ancestor that does exist so the UI can at least open the right section.
    parts = head.split(".")
    while len(parts) > 1:
        parts.pop()
        cand = ".".join(parts)
        if _has_path(tree, cand):
            return cand
    return head


def _dir_writable(p: Path) -> bool:
    """True if ``p`` -- or the nearest existing ancestor -- can be written to."""
    probe = p
    for _ in range(64):
        if _exists(probe):
            return os.access(str(probe), os.W_OK | os.X_OK)
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return False


def _image_exts(values: Optional[dict] = None) -> frozenset:
    """The extensions LM3 itself would ingest (from the config, falling back to the defaults)."""
    exts: Any = None
    if isinstance(values, dict):
        try:
            exts = values["project"]["input"]["image_extensions"]
        except (KeyError, TypeError):
            exts = None
    if not exts:
        exts = _defaults()["project"]["input"]["image_extensions"]
    out = set()
    for e in exts or []:
        e = str(e).lower()
        out.add(e if e.startswith(".") else f".{e}")
    return frozenset(out)


def _count_images(d: Path, exts: frozenset, *, cap: int = _BROWSE_SCAN_CAP) -> tuple:
    """(n_images, n_dirs, truncated, readable) for one directory -- never raises."""
    n_img = n_dir = 0
    seen = 0
    try:
        with os.scandir(str(d)) as it:
            for ent in it:
                seen += 1
                if seen > cap:
                    return n_img, n_dir, True, True
                try:
                    # follow_symlinks=True on purpose: LM3's own _working/ tree is symlinks, and a
                    # symlinked folder of originals is a perfectly normal thing to point LM3 at.
                    if ent.is_dir():
                        n_dir += 1
                        continue
                except OSError:
                    continue
                if os.path.splitext(ent.name)[1].lower() in exts:
                    n_img += 1
    except PermissionError:
        return 0, 0, False, False
    except OSError:
        return 0, 0, False, False
    return n_img, n_dir, False, True


def _count_images_recursive(root: Path, exts: frozenset, *, cap: int = _RECURSIVE_WALK_CAP) -> tuple:
    """(n_images, truncated) over a tree -- capped, used only to answer "is this dir empty?"."""
    n = seen = 0
    for dirpath, dirnames, filenames in os.walk(str(root), onerror=lambda _e: None):
        for name in filenames:
            seen += 1
            if os.path.splitext(name)[1].lower() in exts:
                n += 1
        if seen > cap:
            return n, True
        # do not descend into LM3's own derived trees when sizing an input folder
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "_tmp", "_working"))]
    return n, False


def _collect_warnings(cfg: Any, values: dict) -> list:
    """Non-fatal problems worth surfacing before a run starts.

    ``Config.validate()`` deliberately checks structure only (artifact existence is
    ``core.validate.validate_ml_artifacts``' job, and it only runs at run start). A GUI that
    waits until then to mention a missing model folder is a GUI that wastes the user's time.
    """
    out: list = []

    def warn(code: str, path: Optional[str], msg: str) -> None:
        out.append({"code": code, "path": path, "msg": msg})

    try:
        mock = bool(cfg.compute.get("mock", False))
    except Exception:  # noqa: BLE001 - a hostile tree should not break validation reporting
        mock = False

    if mock:
        warn("mock", "compute.mock",
             "Mock mode is on: LM3 runs synthetic models and produces placeholder results.")

    # -- run mode ------------------------------------------------------------ #
    try:
        if cfg.overwrite:
            warn("overwrite", "project.run_mode.overwrite",
                 "Overwrite is on: the working set is rebuilt and prior results for this run name "
                 "are discarded.")
        restart = cfg.restart
        if restart == "all":
            warn("restart", "project.run_mode.restart",
                 "Restart is 'all': every module re-runs from scratch.")
        elif isinstance(restart, list) and restart:
            warn("restart", "project.run_mode.restart",
                 f"Restart will discard and re-run: {', '.join(restart)}.")
        if not str(cfg.project.get("run_name", "") or "").strip():
            warn("run_name", "project.run_name",
                 "Run name is empty: outputs would land directly in the output folder.")
    except Exception as exc:  # noqa: BLE001
        log.debug("run-mode warnings skipped: %s", exc)

    # -- input folders -------------------------------------------------------- #
    exts = _image_exts(values)
    recursive = True
    input_dirs: Iterable = []
    try:
        inp = cfg.project.get("input", {}) or {}
        recursive = bool(inp.get("recursive", True))
        input_dirs = inp.get("dirs", []) or []
    except Exception:  # noqa: BLE001
        input_dirs = []

    resolved_inputs: list[str] = []
    for d in input_dirs:
        try:
            p = Path(cfg.resolve_path(d))
        except Exception:  # noqa: BLE001
            warn("bad_input", "project.input.dirs", f"Input folder is not a usable path: {d!r}")
            continue
        resolved_inputs.append(os.path.normpath(str(p)))
        if not _exists(p):
            warn("missing_input", "project.input.dirs", f"Input folder does not exist: {p}")
            continue
        if not _is_dir(p):
            warn("missing_input", "project.input.dirs", f"Input path is not a folder: {p}")
            continue
        n_img, _n_dir, truncated, readable = _count_images(p, exts)
        if not readable:
            warn("unreadable_input", "project.input.dirs", f"Input folder cannot be read: {p}")
            continue
        if n_img == 0 and not truncated:
            if recursive:
                n_deep, _t = _count_images_recursive(p, exts)
                if n_deep == 0:
                    warn("empty_input", "project.input.dirs",
                         f"No images ({', '.join(sorted(exts))}) found under {p}.")
            else:
                warn("empty_input", "project.input.dirs",
                     f"No images directly in {p} and recursive search is off.")

    # -- output / tmp --------------------------------------------------------- #
    try:
        outp = cfg.project.get("output", {}) or {}
        out_dir = Path(cfg.resolve_path(outp.get("dir", "runs")))
        if not _dir_writable(out_dir):
            warn("output_unwritable", "project.output.dir", f"Output folder is not writable: {out_dir}")
        if os.path.normpath(str(out_dir)) in resolved_inputs:
            warn("output_is_input", "project.output.dir",
                 "Output folder is also an input folder: a second run would ingest its own results.")
        tmp = outp.get("tmp_dir", "auto")
        if isinstance(tmp, str) and tmp.strip().lower() != "auto":
            tmp_dir = Path(cfg.resolve_path(tmp))
            if not _dir_writable(tmp_dir):
                warn("tmp_unwritable", "project.output.tmp_dir",
                     f"Temp folder is not writable (LM3 falls back to the run folder): {tmp_dir}")
    except Exception as exc:  # noqa: BLE001
        log.debug("output warnings skipped: %s", exc)

    # -- model artifacts ------------------------------------------------------ #
    if not mock:
        from leafmachine3.core.config import CANONICAL_STAGE_KEYS

        model_stages = ("mp_conversion_factor", "archival_detector", "plant_detector",
                        "specimen_segmenter", "leaf_segmenter", "landmark_detector")
        for key in model_stages:
            try:
                if not cfg.is_enabled(key):
                    continue
                model = cfg.stage(key).get("model") or {}
                mpath = model.get("path") if isinstance(model, dict) else None
            except Exception:  # noqa: BLE001
                continue
            if mpath and not _exists(Path(cfg.resolve_path(mpath))):
                warn("missing_model", f"modules.{key}.model.path",
                     f"Model file not found: {cfg.resolve_path(mpath)}")
        try:
            if cfg.is_enabled("ruler_classifier"):
                mdir = cfg.stage("ruler_classifier").get("models_dir")
                if mdir and not _is_dir(Path(cfg.resolve_path(mdir))):
                    warn("missing_model", "modules.ruler_classifier.models_dir",
                         f"Ruler classifier models folder not found: {cfg.resolve_path(mdir)}")
        except Exception:  # noqa: BLE001
            pass

        # A typo'd module key is silently ignored by the merge -- the real module keeps running
        # with its default settings, which is the most confusing failure mode there is.
        mods = values.get("modules") if isinstance(values.get("modules"), dict) else {}
        for key in mods or {}:
            if key not in CANONICAL_STAGE_KEYS:
                warn("unknown_module", f"modules.{key}",
                     f"'{key}' is not an LM3 module and will be ignored "
                     f"(valid keys: {', '.join(CANONICAL_STAGE_KEYS)}).")

    return out


def validate_values(values: Any, *, warnings: bool = True) -> dict:
    """Validate a settings tree WITHOUT touching the real file.

    Writes the tree to a throwaway file, runs the real ``Config.load`` + ``Config.validate``
    against it, and returns the outcome instead of raising::

        {"ok": bool, "errors": [str], "field_errors": {path: [str]},
         "warnings": [{code, path, msg}], "preview": "<yaml text>"}
    """
    result: dict[str, Any] = {"ok": False, "errors": [], "field_errors": {},
                              "warnings": [], "preview": None}

    if not isinstance(values, dict):
        result["errors"] = [f"settings must be a mapping, got {type(values).__name__}"]
        return result

    try:
        text = dump_yaml(values)
    except yaml.YAMLError as exc:
        result["errors"] = [f"settings could not be serialized to YAML: {exc}"]
        return result
    result["preview"] = text

    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="lm3_settings_", suffix=".yaml")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)

        from leafmachine3.core.config import Config

        try:
            cfg = Config.load(tmp_path)
        except Exception as exc:  # noqa: BLE001 - malformed tree, unreadable temp, ...
            result["errors"] = [f"settings could not be loaded: {exc}"]
            return result

        try:
            cfg.validate()
            result["ok"] = True
        except ValueError as exc:
            result["errors"] = _split_errors(str(exc))
        except Exception as exc:  # noqa: BLE001 - validate() should only raise ValueError
            result["errors"] = [f"validation failed: {exc}"]

        if warnings:
            try:
                result["warnings"] = _collect_warnings(cfg, values)
            except Exception as exc:  # noqa: BLE001 - advisory only, never fatal
                log.debug("warning collection failed: %s", exc)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if result["errors"]:
        merged = _effective(values)
        known = _meta_paths()
        anchored: dict[str, list] = {}
        for msg in result["errors"]:
            path = _anchor(msg, merged, known)
            if path:
                anchored.setdefault(path, []).append(msg)
        result["field_errors"] = anchored
    return result


# --------------------------------------------------------------------------- #
# Backups
# --------------------------------------------------------------------------- #
def _backup_glob(path: Path) -> str:
    return f"{path.stem}.bak.*.yaml"


def list_backups(path: str | os.PathLike[str] | None = None) -> list:
    """Timestamped backups beside the settings file, newest first."""
    p = settings_path(path)
    out: list = []
    try:
        for b in p.parent.glob(_backup_glob(p)):
            try:
                st = b.stat()
            except OSError:
                continue
            out.append({"name": b.name, "path": str(b), "mtime": round(st.st_mtime, 6),
                        "size": st.st_size})
    except OSError:
        return []
    pristine = p.with_name(f"{p.stem}.orig.yaml")
    if _is_file(pristine):
        try:
            st = pristine.stat()
            out.append({"name": pristine.name, "path": str(pristine), "mtime": round(st.st_mtime, 6),
                        "size": st.st_size, "pristine": True})
        except OSError:
            pass
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def _rotate_backups(path: Path, keep: int = BACKUP_KEEP) -> Optional[str]:
    """Copy the current file aside and prune to the newest ``keep`` backups.

    Also lays down a one-time ``<name>.orig.yaml`` that is NEVER rotated: the shipped
    ``LM3_settings.yaml`` is heavily commented, a plain-YAML round-trip drops those comments,
    and after five GUI saves the last commented copy would otherwise have aged out.
    """
    if not _is_file(path):
        return None

    pristine = path.with_name(f"{path.stem}.orig.yaml")
    if not _exists(pristine):
        try:
            shutil.copy2(str(path), str(pristine))
            log.info("kept a pristine copy of the original settings at %s", pristine)
        except OSError as exc:
            log.warning("could not write pristine backup %s: %s", pristine, exc)

    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = path.with_name(f"{path.stem}.bak.{ts}.yaml")
    n = 1
    while _exists(dest):                      # two saves inside the same second
        dest = path.with_name(f"{path.stem}.bak.{ts}-{n}.yaml")
        n += 1
    try:
        shutil.copy2(str(path), str(dest))
    except OSError as exc:
        raise SettingsError(f"could not back up {path}: {exc}") from exc

    try:
        olds = sorted(path.parent.glob(_backup_glob(path)), key=lambda p: p.stat().st_mtime)
        for stale in olds[:-keep] if keep > 0 else olds:
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass
    return str(dest)


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically, preserving the original file mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode: Optional[int] = None
    if _exists(path):
        try:
            mode = stat_mod.S_IMODE(path.stat().st_mode)
        except OSError:
            mode = None
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode if mode is not None else 0o644)
        os.replace(tmp, str(path))            # atomic: readers never see a half-written config
        tmp = ""
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #
def write_settings(
    values: Any,
    *,
    path: str | os.PathLike[str] | None = None,
    if_mtime: Optional[float] = None,
    backup: bool = True,
) -> dict:
    """Validate then write ``values``. Raises :class:`SettingsError` if it would be invalid.

    ``if_mtime`` is an optional optimistic-concurrency guard: pass the ``mtime`` from the
    ``GET`` that seeded the form and the write is refused when the file changed underneath
    (a second window, or a hand edit). Omit it to write unconditionally.
    """
    p = settings_path(path)
    check = validate_values(values)
    if not check["ok"]:
        raise SettingsError(
            "settings are invalid and were NOT written",
            errors=check["errors"],
            detail={"field_errors": check["field_errors"], "warnings": check["warnings"]},
        )

    if if_mtime is not None and _is_file(p):
        try:
            current = p.stat().st_mtime
        except OSError:
            current = None
        # 1 ms slack: JSON round-trips the float and some filesystems coarsen the timestamp.
        if current is not None and abs(current - float(if_mtime)) > 1e-3:
            raise SettingsError(
                f"{p.name} changed on disk since it was loaded -- reload before saving",
                errors=[f"{p.name} was modified by someone else (expected mtime {if_mtime}, "
                        f"found {round(current, 6)})"],
                detail={"conflict": True, "mtime": round(current, 6)},
            )

    backup_path = _rotate_backups(p) if backup else None
    try:
        _atomic_write(p, check["preview"] or dump_yaml(values))
    except OSError as exc:
        raise SettingsError(f"could not write {p}: {exc}") from exc

    log.info("wrote %s (backup: %s)", p, backup_path or "none")
    out = read_settings(p)
    out["ok"] = True
    out["backup"] = backup_path
    out["warnings"] = check["warnings"]
    out["errors"] = []
    return out


def restore_backup(name: str, *, path: str | os.PathLike[str] | None = None) -> dict:
    """Validate a backup and put it back in place (the current file is itself backed up first)."""
    p = settings_path(path)
    safe = Path(str(name)).name                  # no traversal: basename only
    src = p.parent / safe
    allowed = {b["name"] for b in list_backups(p)}
    if safe not in allowed or not _is_file(src):
        raise SettingsError(f"unknown backup: {name}")
    return write_settings(load_yaml(src), path=p)


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
def _preset_file(name: str, path: str | os.PathLike[str] | None = None) -> Path:
    raw = str(name).strip()
    if raw.lower().endswith(".yaml"):
        raw = raw[:-5]
    if not _SAFE_PRESET.match(raw):
        raise SettingsError(
            "preset name must be 1-64 chars of letters, digits, space, dot, dash or underscore "
            f"and start alphanumeric; got {name!r}"
        )
    d = presets_dir(path)
    target = d / f"{raw}.yaml"
    # Belt and braces: the regex already bans separators, but re-check the normalized parent so a
    # future loosening of the pattern cannot turn into a path traversal.
    if os.path.normpath(str(target.parent)) != os.path.normpath(str(d)):
        raise SettingsError(f"illegal preset name: {name!r}")
    return target


def list_presets(path: str | os.PathLike[str] | None = None) -> dict:
    """Named settings copies under ``<settings dir>/presets``."""
    d = presets_dir(path)
    items: list = []
    if _is_dir(d):
        try:
            for f in sorted(d.glob("*.yaml"), key=lambda p: p.name.lower()):
                try:
                    st = f.stat()
                except OSError:
                    continue
                note = None
                try:
                    data = load_yaml(f)
                    note = (data.get("_preset") or {}).get("note") if isinstance(data, dict) else None
                except SettingsError:
                    note = "(unreadable)"
                items.append({"name": f.stem, "path": str(f), "mtime": round(st.st_mtime, 6),
                              "size": st.st_size, "note": note})
        except OSError as exc:
            log.warning("could not list presets in %s: %s", d, exc)
    return {"dir": str(d), "presets": items}


def read_preset(name: str, path: str | os.PathLike[str] | None = None) -> dict:
    """Load a preset's values (does NOT write them to the settings file)."""
    f = _preset_file(name, path)
    if not _is_file(f):
        raise SettingsError(f"preset not found: {name}")
    values = load_yaml(f)
    meta = values.pop("_preset", None)           # stored provenance is not a real setting
    try:
        mtime = round(f.stat().st_mtime, 6)
    except OSError:
        mtime = None
    return {"name": f.stem, "path": str(f), "mtime": mtime, "values": values, "preset": meta}


def save_preset(
    name: str,
    values: Any = None,
    *,
    path: str | os.PathLike[str] | None = None,
    note: Optional[str] = None,
) -> dict:
    """Save ``values`` (or a snapshot of the current settings file) as a named preset.

    Presets are validated too -- a preset that cannot be applied is worse than no preset.
    """
    p = settings_path(path)
    if values is None:
        if not _is_file(p):
            raise SettingsError(f"no settings file to snapshot: {p}")
        values = load_yaml(p)
    check = validate_values(values, warnings=False)
    if not check["ok"]:
        raise SettingsError("preset settings are invalid and were NOT saved",
                            errors=check["errors"],
                            detail={"field_errors": check["field_errors"]})

    f = _preset_file(name, p)
    f.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(values)
    payload["_preset"] = {"name": f.stem, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "note": note or "", "source": str(p)}
    _atomic_write(f, dump_yaml(payload))
    log.info("saved settings preset %s", f)
    try:
        mtime = round(f.stat().st_mtime, 6)
    except OSError:
        mtime = None
    return {"ok": True, "name": f.stem, "path": str(f), "mtime": mtime}


def apply_preset(name: str, *, path: str | os.PathLike[str] | None = None) -> dict:
    """Load a preset and write it to the live settings file (validated + backed up)."""
    preset = read_preset(name, path)
    out = write_settings(preset["values"], path=path)
    out["applied_preset"] = preset["name"]
    return out


def delete_preset(name: str, path: str | os.PathLike[str] | None = None) -> dict:
    f = _preset_file(name, path)
    if not _is_file(f):
        raise SettingsError(f"preset not found: {name}")
    try:
        f.unlink()
    except OSError as exc:
        raise SettingsError(f"could not delete preset {name}: {exc}") from exc
    return {"ok": True, "name": f.stem}


# --------------------------------------------------------------------------- #
# Folder browser (for the `path`-typed settings)
# --------------------------------------------------------------------------- #
def _roots() -> list:
    """Sensible starting points for the picker."""
    out: list = []
    seen: set = set()

    def add(label: str, p: Path) -> None:
        s = os.path.normpath(str(p))
        if s not in seen and os.path.isdir(s):
            seen.add(s)
            out.append({"label": label, "path": s})

    add("Home", Path.home())
    add("Working folder", Path.cwd())
    try:
        add("Settings folder", settings_path().parent)
    except SettingsError:            # a misconfigured $LM3_SETTINGS_PATH must not break the picker
        pass
    for extra in ("/data", "/datac", "/mnt", "/media", "/scratch"):
        add(extra, Path(extra))
    add("Filesystem root", Path(os.path.abspath(os.sep)))
    return out


def browse(
    path: Optional[str] = None,
    *,
    show_hidden: bool = False,
    count_images: bool = True,
    extensions: Optional[list] = None,
    max_entries: int = _BROWSE_MAX_ENTRIES,
) -> dict:
    """List the sub-folders of ``path`` (plus how many images each holds).

    Containment: the requested path is expanded, absolutized against the CWD and normalized, so
    ``..`` can never climb past the filesystem root -- and only directories are ever listed, so
    this cannot be used to read file contents. Unreadable folders come back with
    ``readable: false`` instead of raising.
    """
    requested = path
    if path is None or not str(path).strip():
        target = Path.cwd()
    else:
        target = Path(str(path).strip()).expanduser()
        if not target.is_absolute():
            target = Path.cwd() / target
    # normpath collapses ".."/"." textually; realpath then resolves symlinks. Both stop at "/",
    # which IS the containment guarantee on POSIX -- there is nothing above the root to escape to.
    try:
        resolved = Path(os.path.realpath(os.path.normpath(str(target))))
    except OSError:
        resolved = Path(os.path.normpath(str(target)))

    out: dict[str, Any] = {
        "requested": str(requested) if requested is not None else None,
        "path": str(resolved),
        "parent": None,
        "exists": _exists(resolved),
        "is_dir": _is_dir(resolved),
        "readable": True,
        "fell_back": False,
        "entries": [],
        "n_dirs": 0,
        "n_images": 0,
        "truncated": False,
        "entries_truncated": False,
        "extensions": [],
        "roots": _roots(),
        "error": None,
    }

    exts = frozenset(f".{e.lstrip('.').lower()}" for e in extensions) if extensions else None
    if exts is None:
        try:
            exts = _image_exts(read_settings(with_text=False)["values"])
        except Exception:  # noqa: BLE001 - a broken settings file must not break the picker
            exts = _image_exts(None)
    out["extensions"] = sorted(exts)

    # Typing a path that does not exist yet is normal in a picker: fall back to the nearest
    # existing ancestor so the UI still has something to show, and say so.
    if not _is_dir(resolved):
        probe = resolved
        for _ in range(64):
            parent = probe.parent
            if parent == probe:
                break
            probe = parent
            if _is_dir(probe):
                break
        if _is_dir(probe) and probe != resolved:
            out["fell_back"] = True
            out["path"] = str(probe)
            resolved = probe
        else:
            out["error"] = f"not a folder: {resolved}"
            return out

    if resolved.parent != resolved:
        out["parent"] = str(resolved.parent)

    dirs: list = []
    n_img = seen = 0
    try:
        with os.scandir(str(resolved)) as it:
            for ent in it:
                seen += 1
                if seen > _BROWSE_SCAN_CAP:
                    out["truncated"] = True
                    break
                if not show_hidden and ent.name.startswith("."):
                    continue
                try:
                    is_dir = ent.is_dir()
                except OSError:
                    continue
                if is_dir:
                    dirs.append(ent)
                elif os.path.splitext(ent.name)[1].lower() in exts:
                    n_img += 1
    except PermissionError as exc:
        out["readable"] = False
        out["error"] = f"permission denied: {resolved}"
        log.debug("browse denied %s: %s", resolved, exc)
        return out
    except OSError as exc:
        out["readable"] = False
        out["error"] = str(exc)
        return out

    out["n_images"] = n_img
    out["n_dirs"] = len(dirs)
    dirs.sort(key=lambda e: e.name.lower())
    if len(dirs) > max_entries:
        out["entries_truncated"] = True
        dirs = dirs[:max_entries]

    # Counting images inside every child means one scandir per child: fine for a normal folder,
    # wasteful for one with thousands of children, so it is capped rather than conditional.
    do_counts = count_images and len(dirs) <= _BROWSE_COUNT_CHILDREN
    for ent in dirs:
        child = os.path.join(str(resolved), ent.name)
        row: dict[str, Any] = {"name": ent.name, "path": child, "n_images": None,
                               "n_dirs": None, "readable": True, "truncated": False}
        if do_counts:
            c_img, c_dir, trunc, readable = _count_images(Path(child), exts)
            row.update({"n_images": c_img, "n_dirs": c_dir, "truncated": trunc,
                        "readable": readable})
        out["entries"].append(row)
    return out


def make_dir(path: str) -> dict:
    """Create a folder from the picker (``mkdir -p``). Used by the output/tmp path fields."""
    p = Path(str(path).strip()).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = Path(os.path.normpath(str(p)))
    try:
        p.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise SettingsError(f"permission denied creating {p}: {exc}") from exc
    except OSError as exc:
        raise SettingsError(f"could not create {p}: {exc}") from exc
    return {"ok": True, "path": str(p), "created": True}


# --------------------------------------------------------------------------- #
# Router (lazy fastapi import -- mirrors metrics.router / app.create_app)
# --------------------------------------------------------------------------- #
_MODELS: dict = {}


def _request_models() -> dict:
    """Build the pydantic request bodies once, and publish them in this module's globals.

    The publishing is deliberate, not a hack: ``from __future__ import annotations`` turns every
    route annotation into a STRING, and FastAPI resolves those strings against the route
    function's ``__globals__`` (module scope) -- never its enclosing locals. A model defined
    inside ``create_router`` would therefore stay an unresolved ForwardRef and blow up the moment
    anything built the OpenAPI schema (``/docs``). Importing pydantic here rather than at module
    scope keeps a base install -- no ``server`` extra -- importable.
    """
    if _MODELS:
        return _MODELS
    from pydantic import BaseModel, Field

    class SettingsBody(BaseModel):
        values: dict = Field(..., description="the full settings tree to write")
        yaml_path: Optional[str] = Field(None, description="override the target file")
        if_mtime: Optional[float] = Field(None, description="mtime the client loaded (409 on drift)")
        backup: bool = Field(True, description="rotate a timestamped backup first")

    class ValidateBody(BaseModel):
        values: dict
        yaml_path: Optional[str] = None

    class BrowseBody(BaseModel):
        path: Optional[str] = None
        show_hidden: bool = False
        count_images: bool = True
        extensions: Optional[list] = None
        max_entries: int = Field(_BROWSE_MAX_ENTRIES, ge=1, le=5000)

    class MkdirBody(BaseModel):
        path: str

    class PresetBody(BaseModel):
        values: Optional[dict] = None
        yaml_path: Optional[str] = None
        note: Optional[str] = None

    class RestoreBody(BaseModel):
        name: str
        yaml_path: Optional[str] = None

    _MODELS.update(
        SettingsBody=SettingsBody, ValidateBody=ValidateBody, BrowseBody=BrowseBody,
        MkdirBody=MkdirBody, PresetBody=PresetBody, RestoreBody=RestoreBody,
    )
    globals().update(_MODELS)
    return _MODELS


def create_router(require_token: Any = None, dependencies: Optional[list] = None) -> Any:
    """Build the ``/v1/settings`` APIRouter.

    Pass EITHER ``dependencies=[Depends(require_token)]`` (what ``metrics.router`` takes, and the
    recommended wiring) OR the bare ``require_token`` callable, which is wrapped in ``Depends``
    here. ``app.create_app`` defines ``require_token`` as a closure, so it cannot be imported --
    the integrator hands it over at mount time.
    """
    from fastapi import APIRouter, Body, Depends, HTTPException, Path as PathParam, Query

    # Local aliases for readability -- these are the very objects the (stringified) route
    # annotations below resolve to, since _request_models() also published them module-wide.
    models = _request_models()
    SettingsBody = models["SettingsBody"]
    ValidateBody = models["ValidateBody"]
    BrowseBody = models["BrowseBody"]
    MkdirBody = models["MkdirBody"]
    PresetBody = models["PresetBody"]
    RestoreBody = models["RestoreBody"]

    deps = list(dependencies or [])
    if require_token is not None:
        deps.append(Depends(require_token))

    api = APIRouter(prefix="/v1/settings", tags=["settings"], dependencies=deps)

    # Routes are plain ``def`` (not ``async def``), for the same reason ``results_api.router``
    # gives: every one of them does blocking disk work -- parsing YAML, an fsync'd atomic write,
    # backup rotation, and for /browse a scandir of up to _BROWSE_SCAN_CAP dirents plus one more
    # per child folder. Starlette runs a sync endpoint in its threadpool; an ``async def`` here
    # would stall the event loop that is simultaneously streaming status, logs and metrics to
    # every connected client.
    def _fail(exc: SettingsError, status: int = 400) -> "HTTPException":
        detail = {"ok": False, "message": str(exc), "errors": exc.errors}
        detail.update(exc.detail)
        if exc.detail.get("conflict"):
            status = 409
        return HTTPException(status_code=status, detail=detail)

    # -- core ---------------------------------------------------------------- #
    @api.get("")
    def get_settings(yaml_path: Optional[str] = Query(None)) -> dict:
        try:
            return read_settings(yaml_path)
        except SettingsError as exc:                 # a rejected yaml_path is a 400, not a 500
            raise _fail(exc)

    @api.put("")
    def put_settings(body: SettingsBody) -> dict:
        try:
            return write_settings(body.values, path=body.yaml_path, if_mtime=body.if_mtime,
                                  backup=body.backup)
        except SettingsError as exc:
            raise _fail(exc)

    @api.post("/validate")
    def post_validate(body: ValidateBody) -> dict:
        return validate_values(body.values)

    @api.get("/meta")
    def get_meta() -> dict:
        return read_meta()

    @api.get("/defaults")
    def get_defaults() -> dict:
        return {"defaults": _defaults()}

    # -- backups ------------------------------------------------------------- #
    @api.get("/backups")
    def get_backups(yaml_path: Optional[str] = Query(None)) -> dict:
        try:
            p = settings_path(yaml_path)
        except SettingsError as exc:
            raise _fail(exc)
        return {"yaml_path": str(p), "keep": BACKUP_KEEP, "backups": list_backups(p)}

    @api.post("/restore")
    def post_restore(body: RestoreBody) -> dict:
        try:
            return restore_backup(body.name, path=body.yaml_path)
        except SettingsError as exc:
            raise _fail(exc)

    # -- presets ------------------------------------------------------------- #
    @api.get("/presets")
    def get_presets(yaml_path: Optional[str] = Query(None)) -> dict:
        try:
            return list_presets(yaml_path)
        except SettingsError as exc:
            raise _fail(exc)

    @api.get("/presets/{name}")
    def get_preset(name: str = PathParam(...), yaml_path: Optional[str] = Query(None)) -> dict:
        try:
            return read_preset(name, yaml_path)
        except SettingsError as exc:
            raise _fail(exc, status=404 if "not found" in str(exc) else 400)

    @api.post("/presets/{name}")
    def post_preset(name: str = PathParam(...), body: Optional[PresetBody] = Body(None)) -> dict:
        body = body or PresetBody()
        try:
            return save_preset(name, body.values, path=body.yaml_path, note=body.note)
        except SettingsError as exc:
            raise _fail(exc)

    @api.post("/presets/{name}/apply")
    def post_preset_apply(name: str = PathParam(...),
                                yaml_path: Optional[str] = Query(None)) -> dict:
        try:
            return apply_preset(name, path=yaml_path)
        except SettingsError as exc:
            raise _fail(exc, status=404 if "not found" in str(exc) else 400)

    @api.delete("/presets/{name}")
    def delete_preset_route(name: str = PathParam(...),
                                  yaml_path: Optional[str] = Query(None)) -> dict:
        try:
            return delete_preset(name, yaml_path)
        except SettingsError as exc:
            raise _fail(exc, status=404 if "not found" in str(exc) else 400)

    # -- folder picker ------------------------------------------------------- #
    @api.post("/browse")
    def post_browse(body: Optional[BrowseBody] = Body(None)) -> dict:
        body = body or BrowseBody()
        return browse(body.path, show_hidden=body.show_hidden, count_images=body.count_images,
                      extensions=body.extensions, max_entries=body.max_entries)

    @api.post("/mkdir")
    def post_mkdir(body: MkdirBody) -> dict:
        try:
            return make_dir(body.path)
        except SettingsError as exc:
            raise _fail(exc)

    return api


def router(dependencies: Optional[list] = None) -> Any:
    """Alias matching :func:`leafmachine3.server.metrics.router`'s signature."""
    return create_router(dependencies=dependencies)


__all__ = [
    "SettingsError", "settings_path", "meta_path", "presets_dir",
    "read_settings", "read_meta", "validate_values", "write_settings", "dump_yaml", "load_yaml",
    "list_backups", "restore_backup",
    "list_presets", "read_preset", "save_preset", "apply_preset", "delete_preset",
    "browse", "make_dir", "create_router", "router",
]
