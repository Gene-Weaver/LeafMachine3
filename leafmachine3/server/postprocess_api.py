"""leafmachine3.server.postprocess_api -- host and run the LM3 POSTPROCESSING tools over HTTP.

Postprocessing tools are standalone: they run AFTER a project finishes, are configured from
``postprocessing_settings.yaml`` (deliberately separate from ``LM3_settings.yaml``), and each one
exposes a ``run(...)`` API plus a ``python -m`` CLI. This module puts a THIN, SELF-DESCRIBING HTTP
layer over them so the UI can render a form for a tool it has never heard of:

    GET  /v1/postprocess/tools            the registry -- inputs, types, defaults, help
    POST /v1/postprocess/run              start one tool in a background thread -> {task_id}
    GET  /v1/postprocess/tasks            recent tasks (newest first)
    GET  /v1/postprocess/tasks/{id}       state / progress / outputs / log tail
    GET  /v1/postprocess/tasks/{id}/events  SSE: hello -> log|progress -> done
    POST /v1/postprocess/pick-masks       candidate input masks from a finished LM3 run
    GET  /v1/postprocess/context          settings path, allowed roots, discovered runs

Adding a second tool is ONE :class:`Tool` entry in :func:`_build_registry` plus its runner
function -- the tab hosts tools PLURAL and needs no frontend change to pick a new one up.

``fastapi`` is imported lazily inside :func:`router` (exactly like ``leafmachine3.server.app`` and
``leafmachine3.server.metrics``), so a base install without the ``server`` extra still imports this
module. The heavy tool dependencies (trimesh / shapely / cv2) are only ever imported by the worker
thread, so starting the server costs nothing.

Security: every user-supplied path is resolved (symlinks included) and must land inside
:func:`allowed_roots` -- the LM3 project dir, the configured output / tmp / input dirs, and the
server jobs root. A caller cannot read or write outside the install.

Concurrency (plan section 2.8): a tool aimed at the run a pipeline is writing right now is
REFUSED, a different completed run is always allowed, and two ``read_write`` tools on one
completed run are serialized by a per-run advisory lock held in the local deployment runtime
registry -- never a lock file inside the output directory, which on a cluster is the network
storage section 3.1 declares unreliable for locking. Both rules live in one place,
:func:`check_target_allowed` and :func:`_acquire_artifact_locks`, so a standalone CLI uses the
same guard as this HTTP layer instead of a parallel copy that drifts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from leafmachine3.core.paths import PathsError

log = logging.getLogger("leafmachine3.server.postprocess")

# -- limits (bounded by construction: a long batch must never grow memory without end) --------
MAX_TASKS = 64                 # retained task records; oldest FINISHED ones are evicted first
MAX_LOG_LINES = 5000           # per-task ring of captured log/stdout lines
HELLO_LOG_TAIL = 400           # lines replayed in the SSE hello frame
SSE_POLL_S = 0.4               # task polling cadence for the events stream
SSE_PING_S = 15.0              # keep-alive frame interval
SSE_MAX_S = 6 * 3600.0         # hard cap on one stream

#: Canonical names only (plan section 3.1). ``LM3_POSTPROCESS_SETTINGS`` is read by the resolver;
#: ``LM3_POSTPROCESS_ROOTS`` is NOT a path-resolution variable -- it extends the write sandbox
#: below and has no row in the section 3.1 table.
_SETTINGS_ENV = "LM3_POSTPROCESS_SETTINGS"
_ROOTS_ENV = "LM3_POSTPROCESS_ROOTS"

# Directory names skipped when scanning for run directories (noise, not results).
_SKIP_DIRS = {".git", ".hg", "__pycache__", "node_modules", ".idea", ".vscode", ".ipynb_checkpoints"}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ParamError(ValueError):
    """A user-supplied parameter is missing, the wrong type, or points outside the allowed roots.

    Surfaces as HTTP 422 with the message verbatim, so every message must be actionable on its own.
    """


class ToolBusy(RuntimeError):
    """The requested tool already has a task running (surfaces as HTTP 409)."""

    def __init__(self, tool_id: str, task_id: str) -> None:
        super().__init__(f"tool {tool_id!r} is already running (task {task_id})")
        self.tool_id = tool_id
        self.task_id = task_id


class TargetActive(RuntimeError):
    """The tool's target is the run a pipeline is writing RIGHT NOW (plan section 2.8, HTTP 409).

    Postprocessing reads a finished run's reports and DB and writes new artifacts beside them. Do
    that to a live run and the two disagree about what exists: the Reporter is still creating the
    files being globbed, the ledger is mid-transaction, and any output lands in a directory the run
    may still rewrite. A DIFFERENT completed run is always allowed -- these tools are CPU-only, so
    there is no device to contend for.
    """

    def __init__(self, artifact_dir: Path, run_name: Optional[str] = None,
                 run_id: Optional[str] = None) -> None:
        who = f"{run_name!r}" if run_name else "the active run"
        super().__init__(
            f"refused: {artifact_dir} belongs to {who}, which is running right now. Postprocessing "
            f"a live run would read half-written reports and write into a directory the pipeline "
            f"still owns. Wait for it to finish, or pick a completed run."
        )
        self.artifact_dir = artifact_dir
        self.run_name = run_name
        self.run_id = run_id


class TargetLocked(RuntimeError):
    """Another read_write tool already holds this run's advisory lock (plan section 2.8, HTTP 409).

    The lock is per RUN, not per tool: two different tools writing the same run's ``reports`` tree
    concurrently is the race the per-tool gate in :class:`TaskRegistry` does not cover.
    """

    def __init__(self, artifact_dir: Path) -> None:
        super().__init__(
            f"refused: another read/write postprocessing tool is already working on {artifact_dir}. "
            f"Two writers on one run's artifacts are serialized; try again when it finishes."
        )
        self.artifact_dir = artifact_dir


# --------------------------------------------------------------------------- #
# Registry types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolInput:
    """One form field of a tool, described well enough for the UI to render it blind."""

    key: str
    label: str
    type: str                                   # bool | int | float | string | path | list | enum
    #                                             | color | color_or_none ("transparent" allowed)
    default: Any = None
    help: str = ""
    required: bool = False
    accepts: Optional[str] = None               # mask_png | dir | file | None
    multi: bool = False                         # accepts a LIST of values (paths)
    must_exist: bool = False                    # a path input that has to be there already
    item_type: Optional[str] = None             # for type == "list": what each item is
    enum: Optional[tuple] = None                # for type == "enum": the allowed values
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    placeholder: Optional[str] = None
    group: str = ""                             # sub-block heading inside the tool card
    important: bool = False                     # the knobs a user actually adjusts (green rows)

    def describe(self, default: Any) -> dict:
        """Wire form. EVERY key is always present -- the UI branches on null, never on presence."""
        return {
            "key": self.key,
            "label": self.label,
            "type": self.type,
            "default": default,
            "help": self.help,
            "required": self.required,
            "accepts": self.accepts,
            "multi": self.multi,
            "must_exist": self.must_exist,
            "item_type": self.item_type,
            "enum": list(self.enum) if self.enum else None,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "placeholder": self.placeholder,
            "group": self.group,
            "important": self.important,
        }


@dataclass(frozen=True)
class Tool:
    """One postprocessing tool: its form, its runner, and where its defaults come from."""

    id: str
    name: str
    description: str
    inputs: tuple[ToolInput, ...]
    outputs_description: str
    runner: Callable[[dict, "TaskContext"], dict]
    settings_key: str                           # top-level block in postprocessing_settings.yaml
    module: str                                 # importable module implementing the tool
    cli: str                                    # the equivalent command line, shown in the UI
    icon: str = "*"
    #: plan section 2.8: "Every tool declares read_only / read_write and cpu / gpu resource
    #: metadata." ``access`` decides whether the per-run advisory lock is taken; ``resource``
    #: exists so the first GPU postprocessor cannot be added without confronting the question --
    #: nothing today may declare "gpu" and skip explicit device coordination.
    access: str = "read_write"                  # read_only | read_write
    resource: str = "cpu"                       # cpu | gpu
    #: Parameter keys whose value names the run (or a file inside it) this tool touches. The guard
    #: resolves each to its enclosing run directory; that directory IS the ``artifact_dir`` the
    #: refusal and the advisory lock are keyed on.
    target_keys: tuple[str, ...] = ()

    def input(self, key: str) -> Optional[ToolInput]:
        return next((i for i in self.inputs if i.key == key), None)

    def describe(self) -> dict:
        """Registry entry, with defaults refreshed from postprocessing_settings.yaml."""
        block = _yaml_block(self.settings_key)
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "module": self.module,
            "settings_key": self.settings_key,
            "settings_path": str(_settings_path()),
            "cli": self.cli,
            "access": self.access,
            "resource": self.resource,
            "inputs": [i.describe(_default_for(i, block)) for i in self.inputs],
            "outputs_description": self.outputs_description,
        }


def _default_for(spec: ToolInput, block: dict) -> Any:
    """A field's default: the value in postprocessing_settings.yaml when set, else the built-in.

    The tab must show what the tool would ACTUALLY do if run right now, and the CLI reads that same
    yaml -- so an edited yaml has to be reflected here or the form lies about the tool.
    """
    value = block.get(spec.key, None) if isinstance(block, dict) else None
    if value is None:
        value = spec.default
    if spec.multi and value is None:
        return []
    if spec.multi and not isinstance(value, (list, tuple)):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    return value


# --------------------------------------------------------------------------- #
# postprocessing_settings.yaml  (cached on mtime -- the CLI and the UI share this file)
# --------------------------------------------------------------------------- #
_yaml_cache: dict[str, Any] = {"path": None, "mtime": None, "data": {}}
_yaml_lock = threading.Lock()


def _settings_path() -> Path:
    """Section 3.1 row 3: ``LM3_POSTPROCESS_SETTINGS``, else the deployment's ``postprocessing.yaml``.

    Previously ``postprocessing_settings.yaml`` off the server's CWD, which meant the Postprocess
    tab configured a different file than the standalone CLIs unless both were launched from the
    same directory.
    """
    from leafmachine3.server.app import canonical_postprocess_settings_path

    try:
        return canonical_postprocess_settings_path()
    except PathsError as exc:
        log.debug("cannot resolve the postprocessing settings: %s", exc)
        return Path(os.devnull)


def _load_settings() -> dict:
    """Parsed ``postprocessing_settings.yaml`` ({} when absent or unreadable)."""
    path = _settings_path()
    try:
        mtime = path.stat().st_mtime if path.is_file() else None
    except OSError:
        mtime = None
    with _yaml_lock:
        if _yaml_cache["path"] == str(path) and _yaml_cache["mtime"] == mtime:
            return _yaml_cache["data"]
    data: dict = {}
    if mtime is not None:
        try:
            from leafmachine3.postprocessing.config import load_settings

            data = load_settings(str(path)) or {}
        except Exception as exc:  # noqa: BLE001 - a broken yaml must not take the tab down
            log.warning("cannot read %s: %s", path, exc)
            data = {}
    with _yaml_lock:
        _yaml_cache.update({"path": str(path), "mtime": mtime, "data": data})
    return data


def _yaml_block(key: str) -> dict:
    block = _load_settings().get(key)
    return dict(block) if isinstance(block, dict) else {}


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #
_roots_cache: dict[str, Any] = {"key": None, "roots": []}
_roots_lock = threading.Lock()


def _lm3_settings_path() -> Optional[Path]:
    """The canonical settings file (section 3.1 row 1), or ``None`` when it cannot be resolved."""
    from leafmachine3.server.app import canonical_settings_path

    try:
        return canonical_settings_path()
    except PathsError as exc:
        log.debug("cannot resolve the LM3 settings file: %s", exc)
        return None


def _lm3_settings() -> dict:
    """Best-effort parse of LM3_settings.yaml (for the configured output / tmp / input dirs)."""
    path = _lm3_settings_path()
    if path is None or not path.is_file():
        return {}
    try:
        import yaml

        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        log.debug("cannot read %s: %s", path, exc)
        return {}


def _normalize_root(raw: Any, settings_file: Optional[Path]) -> Optional[Path]:
    """One raw root -> an absolute, symlink-resolved path, or ``None`` when it is unusable.

    Relative entries are resolved against the settings FILE, exactly like ``project.output.dir`` --
    never against the CWD (section 3.1) -- and are dropped outright when there is no settings file
    to resolve them against.
    """
    if not raw or not isinstance(raw, (str, os.PathLike)):
        return None
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        if settings_file is None:
            return None
        p = settings_file.parent / p
    try:
        return Path(os.path.realpath(p))
    except OSError:
        return None


def _static_allowed_roots() -> list[Path]:
    """The half of :func:`allowed_roots` that only a settings/env edit can move -- memoized.

    Every input here is observable in the cache key below: the settings file, the
    ``LM3_POSTPROCESS_ROOTS`` override, the ``project`` block, and each tool's configured
    ``output_dir``. The RUN-HISTORY half is not observable in any of them, so it deliberately does
    not live here (see :func:`allowed_roots`).
    """
    settings_file = _lm3_settings_path()
    cfg = _lm3_settings()
    # The tool output_dirs below come from postprocessing_settings.yaml, so that file has to be part
    # of the cache key -- otherwise editing it updates the FORM (Tool.describe re-reads on every GET)
    # while the roots stay stale, and the tool's own configured output is rejected as outside them.
    key = json.dumps([str(settings_file), os.environ.get(_ROOTS_ENV, ""), cfg.get("project", {}),
                      {t.settings_key: _yaml_block(t.settings_key).get("output_dir")
                       for t in _TOOLS.values()}],
                     default=str, sort_keys=True)
    with _roots_lock:
        if _roots_cache["key"] == key:
            return list(_roots_cache["roots"])

    out: list[Path] = []

    def add(raw: Any) -> None:
        real = _normalize_root(raw, settings_file)
        if real is not None and real not in out:
            out.append(real)

    if settings_file is not None:
        add(settings_file.parent)
    for extra in os.environ.get(_ROOTS_ENV, "").split(os.pathsep):
        add(extra.strip())

    project = cfg.get("project", {}) if isinstance(cfg.get("project"), dict) else {}
    output = project.get("output", {}) if isinstance(project.get("output"), dict) else {}
    add(output.get("dir"))
    tmp = output.get("tmp_dir")
    if isinstance(tmp, str) and tmp.strip().lower() != "auto":
        add(tmp)
    inp = project.get("input", {}) if isinstance(project.get("input"), dict) else {}
    for d in inp.get("dirs") or []:
        add(d)

    # the server's staged job dirs (section 3.1 row 4)
    try:
        from leafmachine3.server.app import server_jobs_root

        add(server_jobs_root())
    except Exception:  # noqa: BLE001 - app.py is optional here
        pass

    # anything a tool is already configured to write to
    for tool in _TOOLS.values():
        block = _yaml_block(tool.settings_key)
        add(block.get("output_dir"))

    with _roots_lock:
        _roots_cache.update({"key": key, "roots": list(out)})
    return list(out)


def allowed_roots() -> list[Path]:
    """Directories a request is allowed to read from / write into.

    Loopback binding plus a Bearer token already gates WHO can call; this gates WHERE. The set is
    intentionally the same ground the CLI covers (the project dir and its configured data dirs) so
    nothing the user can do from the terminal is blocked in the app -- and nothing beyond it works.

    The server's CWD is NOT a root any more (plan section 3.1: "No path falls back to the current
    working directory"). It was never a statement about the user's data -- it was a statement about
    where someone happened to type ``lm3 serve`` -- and as a WRITE sandbox that is the wrong shape:
    a server started from ``/`` or from a home directory authorized the lot. The settings file's own
    directory replaces it, which is the directory the project is actually described from.

    Composed from two halves on purpose: a memoized settings/env half, and the run-history half,
    recomputed on EVERY call because nothing in that memo key observes it.
    """
    settings_file = _lm3_settings_path()
    out = _static_allowed_roots()

    # every run-history root the Results tab can list, so a run visible in the app is a run the
    # Postprocess tab may read (section 3.1 row 5).
    #
    # This union sits OUTSIDE the memo above, and that placement is the whole point: ``run_roots()``
    # re-reads active.json / last.json on every call (row 5 slot 2), and NOTHING in the memo key
    # moves when a run starts or finishes. Inside the memo the sandbox was computed from an input it
    # never observed, and it failed in both directions -- a CLI run writing outside every configured
    # root was refused here while the Results tab was listing it, and a root that entered the set
    # while a record named it stayed writable for the life of the server long after no record named
    # it. The active/last run_id is no use as a key either: section 2.10's staged mode contributes
    # run_dir, artifact_dir and active_state_dir separately and a hardware_setup record contributes
    # none, so record -> roots is not run_id-determined. The read is no more expensive than the
    # per-tool YAML reads the key already performs.
    try:
        from leafmachine3.server.results_api import run_roots

        for root in run_roots():
            real = _normalize_root(root, settings_file)
            if real is not None and real not in out:
                out.append(real)
    except Exception:  # noqa: BLE001 - results_api is optional here
        log.debug("could not read the run-history roots", exc_info=True)

    return out


def _within_roots(real: Path, roots: list[Path]) -> bool:
    return any(real == r or real.is_relative_to(r) for r in roots)


def resolve_path(raw: Any, *, must_exist: bool, kind: str = "any", label: str = "path") -> Path:
    """Resolve one user path and assert it stays inside :func:`allowed_roots`.

    ``os.path.realpath`` is used (not ``Path.resolve(strict=True)``) because an OUTPUT directory
    legitimately does not exist yet -- but it still resolves symlinks in the existing prefix, which
    is what keeps a symlinked shortcut from escaping the roots.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise ParamError(f"{label}: empty path")
    if not isinstance(raw, (str, os.PathLike)):
        raise ParamError(f"{label}: expected a path string, got {type(raw).__name__}")
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        # A relative path is resolved against the SETTINGS FILE's directory -- the same rule
        # project.output.dir follows (core.paths.resolve_project_output_dir) and the same base
        # allowed_roots() uses. It used to be joined onto the server's CWD, so one request meant
        # different files depending on where the server was launched (section 3.1).
        base = _lm3_settings_path()
        if base is None:
            raise ParamError(
                f"{label}: {raw!r} is relative and there is no resolved settings file to resolve "
                f"it against; give an absolute path"
            )
        p = base.parent / p
    try:
        real = Path(os.path.realpath(p))
    except (OSError, ValueError) as exc:      # ValueError = embedded NUL byte; still a bad PARAM
        raise ParamError(f"{label}: cannot resolve {raw!r} ({exc})") from exc

    roots = allowed_roots()
    if not _within_roots(real, roots):
        shown = ", ".join(str(r) for r in roots[:6])
        raise ParamError(
            f"{label}: {real} is outside the allowed roots ({shown}). "
            f"Set {_ROOTS_ENV} to add another location."
        )
    if must_exist and not real.exists():
        raise ParamError(f"{label}: no such path: {real}")
    if real.exists():
        if kind == "file" and not real.is_file():
            raise ParamError(f"{label}: expected a file, got a directory: {real}")
        if kind == "dir" and not real.is_dir():
            raise ParamError(f"{label}: expected a directory, got a file: {real}")
    return real


# --------------------------------------------------------------------------- #
# Parameter validation / coercion
# --------------------------------------------------------------------------- #
_TRUE = {"1", "true", "t", "yes", "y", "on"}
_FALSE = {"0", "false", "f", "no", "n", "off"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_NAMED_COLORS = {"white", "black"}


def _as_bool(v: Any, label: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    raise ParamError(f"{label}: expected true or false, got {v!r}")


def _as_number(v: Any, label: str, *, integer: bool) -> Any:
    if isinstance(v, bool):
        raise ParamError(f"{label}: expected a number, got a boolean")
    if isinstance(v, str):
        s = v.strip()
        if not s:
            raise ParamError(f"{label}: expected a number, got an empty string")
        try:
            v = float(s)
        except ValueError as exc:
            raise ParamError(f"{label}: expected a number, got {v!r}") from exc
    if not isinstance(v, (int, float)):
        raise ParamError(f"{label}: expected a number, got {type(v).__name__}")
    if integer:
        if float(v) != int(v):
            raise ParamError(f"{label}: expected a whole number, got {v}")
        return int(v)
    return float(v)


def _check_bounds(value: float, spec: ToolInput, label: str) -> None:
    if spec.min is not None and value < spec.min:
        raise ParamError(f"{label}: {value} is below the minimum {spec.min}")
    if spec.max is not None and value > spec.max:
        raise ParamError(f"{label}: {value} is above the maximum {spec.max}")


def _normalize_color(v: Any, label: str) -> Any:
    """Accept the shapes a UI can produce and hand back what the tool understands.

    ``generate_stl_from_mask`` takes a name ("white"/"black") or an ``[R, G, B(, A)]`` list; a text
    or swatch control naturally produces ``"#ffffff"`` or ``"255,255,255"``, so both are folded to
    an RGB triple here rather than being rejected.
    """
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in _NAMED_COLORS:
            return s.lower()
        if s.startswith("#"):
            h = s[1:]
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            if len(h) not in (6, 8):
                raise ParamError(f"{label}: {v!r} is not a #rrggbb color")
            try:
                return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]
            except ValueError as exc:
                raise ParamError(f"{label}: {v!r} is not a #rrggbb color") from exc
        parts = [p for p in re.split(r"[,\s]+", s) if p]
        if len(parts) in (3, 4):
            try:
                rgb = [int(round(float(p))) for p in parts[:3]]
            except ValueError as exc:
                raise ParamError(f"{label}: {v!r} is not a color name or R,G,B triple") from exc
            return _clamp_rgb(rgb, label)
        raise ParamError(
            f"{label}: {v!r} is not a color -- use 'white', 'black', '#rrggbb' or 'R,G,B'"
        )
    if isinstance(v, (list, tuple)):
        if len(v) < 3:
            raise ParamError(f"{label}: a color needs at least R, G, B (got {list(v)!r})")
        try:
            rgb = [int(round(float(c))) for c in list(v)[:3]]
        except (TypeError, ValueError) as exc:
            raise ParamError(f"{label}: {list(v)!r} is not an [R, G, B] color") from exc
        return _clamp_rgb(rgb, label)
    raise ParamError(f"{label}: expected a color name or [R, G, B], got {type(v).__name__}")


def _clamp_rgb(rgb: list[int], label: str) -> list[int]:
    for c in rgb:
        if c < 0 or c > 255:
            raise ParamError(f"{label}: color channels must be 0-255, got {rgb!r}")
    return rgb


def _coerce(spec: ToolInput, value: Any) -> Any:
    """Coerce one submitted value to what the tool's Python API expects."""
    label = spec.label or spec.key

    if spec.multi:
        if value is None:
            items: list = []
        elif isinstance(value, (list, tuple)):
            items = list(value)
        elif isinstance(value, str):
            # a textarea / comma list is the natural fallback when no picker was used
            items = [s.strip() for s in re.split(r"[\n,]+", value) if s.strip()]
        else:
            items = [value]
        out = []
        for i, item in enumerate(items):
            out.append(_coerce_scalar(spec, item, f"{label}[{i}]"))
        return out

    if spec.type == "list":
        if value is None:
            items = []
        elif isinstance(value, (list, tuple)):
            items = list(value)
        elif isinstance(value, str):
            items = [s.strip() for s in re.split(r"[\n,;]+", value) if s.strip()] \
                if spec.item_type != "color" else _split_colors(value)
        else:
            items = [value]
        if spec.item_type == "color":
            return [_normalize_color(c, f"{label}[{i}]") for i, c in enumerate(items)]
        return items

    return _coerce_scalar(spec, value, label)


def _split_colors(text: str) -> list:
    """Split a free-text color list. ``"white, 255 0 0"`` -> ``["white", "255 0 0"]``."""
    return [s.strip() for s in re.split(r"[\n;]+|,(?=\s*[A-Za-z#])", text) if s.strip()]


def _coerce_scalar(spec: ToolInput, value: Any, label: str) -> Any:
    t = spec.type
    if t == "bool":
        return _as_bool(value, label)
    if t == "int":
        v = _as_number(value, label, integer=True)
        _check_bounds(v, spec, label)
        return v
    if t == "float":
        v = _as_number(value, label, integer=False)
        _check_bounds(v, spec, label)
        return v
    if t == "enum":
        if spec.enum and value not in spec.enum:
            raise ParamError(f"{label}: {value!r} is not one of {list(spec.enum)}")
        return value
    if t == "path":
        kind = "dir" if spec.accepts == "dir" else "file" if spec.accepts in ("file", "mask_png") else "any"
        p = resolve_path(value, must_exist=spec.must_exist, kind=kind, label=label)
        if spec.accepts == "mask_png" and p.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ParamError(
                f"{label}: {p.name} is not an image ({', '.join(sorted(_IMAGE_SUFFIXES))})"
            )
        return str(p)
    if t == "color":
        return _normalize_color(value, label)
    if t == "color_or_none":
        # A collage background is a color OR the absence of one, and "transparent" is not a color
        # the STL builder's palette may accept -- so the opt-in lives in the TYPE, not in the parser.
        if isinstance(value, str) and value.strip().lower() in ("transparent", "none"):
            return "transparent"
        return _normalize_color(value, label)
    # "string" and anything unknown: pass through as text
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def validate_params(tool: Tool, params: Any) -> dict:
    """Validate a submitted parameter dict against ``tool``'s registry entry.

    Every field falls back to the tool's CURRENT default (built-in overridden by
    postprocessing_settings.yaml), so a caller may submit only what it wants to change.
    """
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ParamError(f"params must be an object, got {type(params).__name__}")

    known = {i.key for i in tool.inputs}
    unknown = [k for k in params if k not in known]
    if unknown:
        raise ParamError(
            f"unknown parameter(s) for {tool.id}: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )

    block = _yaml_block(tool.settings_key)
    out: dict[str, Any] = {}
    for spec in tool.inputs:
        supplied = spec.key in params
        raw = params[spec.key] if supplied else _default_for(spec, block)
        empty = raw is None or (isinstance(raw, str) and not raw.strip()) or \
            (spec.multi and isinstance(raw, (list, tuple)) and len(raw) == 0)
        if empty:
            if spec.required:
                raise ParamError(f"{spec.label or spec.key}: required (no value supplied)")
            # an optional empty stays None/[] -- NOT coerced to 0 or "" (a null output_dir means
            # "write beside the source mask", which is a different behavior from any real path)
            out[spec.key] = [] if spec.multi else None
            continue
        out[spec.key] = _coerce(spec, raw)
    return out


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    """One tool invocation and everything the UI needs to render it live."""

    id: str
    tool_id: str
    tool_name: str
    params: dict
    state: str = "running"                      # running | done | error
    progress: float = 0.0                       # 0.0 .. 1.0
    message: str = ""
    n_done: int = 0
    n_total: int = 0
    outputs: list = field(default_factory=list)
    result: Optional[dict] = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    log_seq: int = 0
    lines: deque = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- writers (worker thread) ------------------------------------------- #
    def append(self, msg: str, level: str = "INFO", src: str = "") -> None:
        for part in str(msg).splitlines() or [""]:
            with self.lock:
                self.log_seq += 1
                self.lines.append({
                    "seq": self.log_seq,
                    "t": round(time.time(), 3),
                    "level": (level or "INFO").upper(),
                    "src": src or self.tool_id,
                    "msg": part,
                })

    def set_progress(self, frac: Optional[float], message: Optional[str] = None,
                     *, n_done: Optional[int] = None, n_total: Optional[int] = None) -> None:
        with self.lock:
            if frac is not None:
                self.progress = max(0.0, min(1.0, float(frac)))
            if message is not None:
                self.message = str(message)
            if n_done is not None:
                self.n_done = int(n_done)
            if n_total is not None:
                self.n_total = int(n_total)

    def add_output(self, path: Any) -> None:
        with self.lock:
            p = str(path)
            if p not in self.outputs:
                self.outputs.append(p)

    def finish(self, result: Optional[dict], error: Optional[str]) -> None:
        with self.lock:
            self.state = "error" if error else "done"
            self.error = error
            self.result = result
            self.finished_at = time.time()
            if not error:
                self.progress = 1.0

    # -- readers (event loop) ---------------------------------------------- #
    def snapshot(self, *, since: Optional[int] = None, log_limit: int = HELLO_LOG_TAIL) -> dict:
        """Wire form. ``since`` returns only log lines with ``seq > since`` (incremental polling)."""
        with self.lock:
            if since is None:
                tail = list(self.lines)[-log_limit:] if log_limit else []
            else:
                tail = [ln for ln in self.lines if ln["seq"] > since][:log_limit or None]
            elapsed = (self.finished_at or time.time()) - self.started_at
            return {
                "task_id": self.id,
                "tool_id": self.tool_id,
                "tool_name": self.tool_name,
                "state": self.state,
                "progress": round(self.progress, 4),
                "pct": round(self.progress * 100.0, 1),
                "message": self.message,
                "n_done": self.n_done,
                "n_total": self.n_total,
                "outputs": list(self.outputs),
                "n_outputs": len(self.outputs),
                "result": self.result,
                "error": self.error,
                "started_at": round(self.started_at, 3),
                "finished_at": round(self.finished_at, 3) if self.finished_at else None,
                "elapsed_s": round(elapsed, 3),
                "params": self.params,
                "log_seq": self.log_seq,
                "log": tail,
            }


class TaskContext:
    """The handle a runner uses to report progress -- the only surface a new tool has to learn."""

    def __init__(self, task: Task, tool: Tool) -> None:
        self._task = task
        self._tool = tool

    @property
    def task(self) -> Task:
        return self._task

    def log(self, msg: str, level: str = "INFO", src: str = "") -> None:
        self._task.append(msg, level, src or self._tool.id)

    def progress(self, frac: Optional[float], message: Optional[str] = None,
                 *, n_done: Optional[int] = None, n_total: Optional[int] = None) -> None:
        self._task.set_progress(frac, message, n_done=n_done, n_total=n_total)

    def add_output(self, path: Any) -> None:
        self._task.add_output(path)


# --------------------------------------------------------------------------- #
# stdout / stderr / logging capture (thread-scoped -- a concurrent LM3 run must be untouched)
# --------------------------------------------------------------------------- #
_thread_tasks: dict[int, Task] = {}
_capture_lock = threading.Lock()
_capture_refs = 0
_orig_stdout: Any = None
_orig_stderr: Any = None
_log_handler: Optional[logging.Handler] = None
_prev_level: Optional[int] = None
_CAPTURE_LOGGERS = ("leafmachine3.postprocessing", "")


class _ThreadTee:
    """A ``sys.stdout`` stand-in that routes writes by THREAD.

    ``contextlib.redirect_stdout`` swaps the stream process-wide, which would steal the console
    output of a pipeline run happening at the same time. Routing on the thread ident means a tool's
    prints land in its own task log and everyone else's keep going to the real terminal.
    """

    def __init__(self, original: Any, level: str) -> None:
        self._original = original
        self._level = level
        self._buffers: dict[int, str] = {}

    def write(self, text: str) -> int:
        task = _thread_tasks.get(threading.get_ident())
        if task is None:
            return self._original.write(text) if self._original else len(text)
        buf = self._buffers.get(threading.get_ident(), "") + text
        *lines, rest = buf.split("\n")
        self._buffers[threading.get_ident()] = rest
        for line in lines:
            task.append(line, self._level, "stdout" if self._level == "INFO" else "stderr")
        return len(text)

    def flush(self) -> None:
        ident = threading.get_ident()
        task = _thread_tasks.get(ident)
        rest = self._buffers.pop(ident, "")
        if task is not None and rest:
            task.append(rest, self._level, "stdout" if self._level == "INFO" else "stderr")
        elif self._original:
            self._original.flush()

    def isatty(self) -> bool:                    # tqdm and friends ask; a captured stream is not
        return False

    def __getattr__(self, name: str) -> Any:     # encoding, fileno, ... delegate to the real stream
        return getattr(self._original, name)


class _TaskLogHandler(logging.Handler):
    """Route ``logging`` records emitted on a task's thread into that task's log."""

    def emit(self, record: logging.LogRecord) -> None:
        task = _thread_tasks.get(threading.get_ident())
        if task is None:
            return
        # This handler sits on BOTH "leafmachine3.postprocessing" and root, and propagation hands
        # the SAME record object to each ancestor -- without this guard every tool warning would be
        # logged twice.
        if getattr(record, "_lm3_pp_seen", False):
            return
        record._lm3_pp_seen = True                   # type: ignore[attr-defined]
        try:
            task.append(record.getMessage(), record.levelname, record.name.split(".")[-1])
        except Exception:  # noqa: BLE001 - logging must never raise into the tool
            pass


def _capture_acquire(task: Task) -> None:
    """Register this thread for capture, installing the tee/handler on the first task."""
    global _capture_refs, _orig_stdout, _orig_stderr, _log_handler, _prev_level
    with _capture_lock:
        _thread_tasks[threading.get_ident()] = task
        _capture_refs += 1
        if _capture_refs > 1:
            return
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _ThreadTee(_orig_stdout, "INFO")
        sys.stderr = _ThreadTee(_orig_stderr, "ERROR")
        _log_handler = _TaskLogHandler(level=logging.DEBUG)
        # Attach at "leafmachine3.postprocessing" (NOT "leafmachine3"): core.logging_setup wipes the
        # handlers off the "leafmachine3" logger when a pipeline run starts, and this has to survive
        # that. The root logger is also covered so third-party chatter (trimesh/shapely) emitted on
        # the same thread is captured -- its level is left alone, since raising root's level would
        # change logging for the whole server.
        for name in _CAPTURE_LOGGERS:
            logging.getLogger(name).addHandler(_log_handler)
        pp = logging.getLogger(_CAPTURE_LOGGERS[0])
        _prev_level = pp.level
        if pp.level == logging.NOTSET or pp.level > logging.INFO:
            pp.setLevel(logging.INFO)


def _capture_release() -> None:
    """Unregister this thread, restoring the real streams once the last task is gone."""
    global _capture_refs, _orig_stdout, _orig_stderr, _log_handler, _prev_level
    with _capture_lock:
        _thread_tasks.pop(threading.get_ident(), None)
        _capture_refs = max(0, _capture_refs - 1)
        if _capture_refs or _log_handler is None:
            return
        for name in _CAPTURE_LOGGERS:
            logging.getLogger(name).removeHandler(_log_handler)
        _log_handler = None
        if _prev_level is not None:
            logging.getLogger(_CAPTURE_LOGGERS[0]).setLevel(_prev_level)
            _prev_level = None
        # only restore if nobody else swapped the stream underneath us
        if isinstance(sys.stdout, _ThreadTee):
            sys.stdout = _orig_stdout
        if isinstance(sys.stderr, _ThreadTee):
            sys.stderr = _orig_stderr


# --------------------------------------------------------------------------- #
# Concurrency policy (plan section 2.8)
# --------------------------------------------------------------------------- #
# Two rules, and one place that enforces them for BOTH the HTTP API and any standalone CLI:
#
#   1. a postprocessor whose target is the ACTIVE pipeline's run directory is refused; a different
#      COMPLETED run is allowed, because these tools are CPU-only and contend for nothing;
#   2. two ``read_write`` tools aimed at the same completed run are serialized by a per-run
#      advisory lock that lives in the LOCAL deployment runtime registry, keyed by a hash of the
#      resolved ``artifact_dir``.
#
# The lock deliberately does NOT live beside the artifacts. On a cluster the output directory is
# network storage, which section 3.1 declares unreliable for locking -- an advisory lock there is a
# lock that silently does not lock. Keeping it node-local is honest, and it states the limit out
# loud: CROSS-HOST concurrent postprocessing of one run is out of scope for the single-user cluster
# profile, not quietly half-supported.
_MAX_RUN_DIR_WALK = 8            # bounded parent walk; a run dir is never that far above a report
_LOCK_DIRNAME = "postprocess"    # <deployment runtime dir>/postprocess/<sha256(artifact_dir)>.lock


def _runtime_v2_enabled() -> bool:
    """The one place this module reads ``LM3_RUNTIME_V2`` (default ON since the cutover).

    With explicit ``LM3_RUNTIME_V2=0`` no runtime record exists, so rule 1 cannot fire and rule 2's
    lock stays off with it. Import failures propagate: silently returning false here would disable
    the active-run write guard, which is the unsafe direction to fail.
    """
    from leafmachine3.core.runtime.execution import runtime_v2_enabled

    return bool(runtime_v2_enabled())


def _enclosing_run_dir(path: Path) -> Path:
    """The run directory ``path`` belongs to, or the nearest directory when it belongs to none.

    A tool is handed a mask file deep inside ``<run>/reports/...`` or an output folder that does
    not exist yet; both have to collapse to the one identity the guard and the lock are keyed on.
    """
    start = path if path.is_dir() else path.parent
    candidate = start
    for _ in range(_MAX_RUN_DIR_WALK):
        if _is_run_dir(candidate):
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return start


def target_artifact_dirs(tool: Tool, params: dict) -> list[Path]:
    """Every run directory ``tool`` would touch with ``params``, de-duplicated, in order.

    ``params`` must already have been through :func:`validate_params`: that is what turns each
    path field into a resolved, sandbox-checked :class:`~pathlib.Path`.
    """
    out: list[Path] = []
    for key in tool.target_keys:
        value = params.get(key)
        for raw in (value if isinstance(value, (list, tuple)) else [value]):
            if raw in (None, ""):
                continue
            try:
                resolved = _enclosing_run_dir(Path(str(raw)))
            except OSError:                        # a vanished or unreadable path is not a target
                continue
            if resolved not in out:
                out.append(resolved)
    return out


def _overlaps(target: Path, active: Path) -> bool:
    """True when the two paths are the same run, or one contains the other.

    Containment matters in both directions: an output folder INSIDE the live run is a write into
    it, and a target ABOVE it (someone passing the whole output root) sweeps it up.
    """
    if target == active:
        return True
    return active in target.parents or target in active.parents


def active_run_target() -> Optional[dict]:
    """The run holding the deployment lease right now, as ``{artifact_dir, run_name, run_id}``.

    The registry and ONLY the registry: never the settings file, never a filesystem recency guess.
    That matters here more than anywhere else -- a refusal built on a guess would block legitimate
    work on a finished run, and section 2.5 is explicit that a guess authorizes nothing.

    The record is read STRAIGHT OFF DISK on every call rather than through
    ``progress_api.active_runtime_ref()``, whose registry read is memoized behind a 0.25-2.0 s TTL
    for the 2 Hz status endpoints. That memo has no lifecycle invalidation from the run-start path,
    so for one TTL after a run publishes ``active.json`` the cached answer is still the pre-launch
    "nothing is running" -- and this function is the input to a NORMATIVE refusal ("it is refused
    when its target is the currently active pipeline's run directory", section 2.8). An
    authorization decision may not be answered out of a cache nobody invalidates: a stale hit here
    admits a ``read_write`` tool into a run the pipeline is writing, and the tool then runs for
    minutes. The read itself is one lock probe plus one small JSON parse, and it happens once per
    tool start (and once per Postprocess tab render), not at 2 Hz.

    Only a LIVE record qualifies, exactly as ``progress_api._from_runtime`` decides it: ABANDONED is
    a crashed writer holding nothing, and INCOMPATIBLE yields no record at all so a newer-schema
    runtime is never half-interpreted here (section 2.9). A ``hardware_setup`` root has no project
    block by invariant 6, so it names no run directory and refuses nothing.
    """
    if not _runtime_v2_enabled():
        return None
    try:
        from leafmachine3.core import paths as core_paths
        from leafmachine3.core.runtime import RecordClassification
        from leafmachine3.core.runtime import records as runtime_records

        # check_filesystem=False: the network-filesystem refusal is for a WRITER taking a lock. A
        # reader that declined to look would report "nothing is running" on exactly the cluster
        # setup section 3.1 warns about -- which here would mean ADMITTING a tool into a live run.
        deployment = core_paths.deployment_runtime_dir(check_filesystem=False)
        if not deployment.is_dir():
            return None
        # include_children=False: the refusal is keyed on the ROOT's run directory. A
        # calibration child (the only child activity there is) writes under its own root's
        # tree, so reading the two child records would cost two file reads and decide nothing.
        snapshot = runtime_records.read_runtime(deployment, include_children=False)
    except Exception:  # noqa: BLE001 - an unreadable registry must not take the tools down
        log.debug("could not read the active runtime record", exc_info=True)
        return None
    if snapshot is None or snapshot.classification is not RecordClassification.LIVE:
        return None
    project = getattr(getattr(snapshot, "record", None), "project", None)
    if project is None:
        return None
    try:
        run_dir = Path(str(project.run_dir))
        run_name = str(project.run_name)
    except (AttributeError, TypeError, ValueError):
        log.debug("runtime record carries an unusable project block", exc_info=True)
        return None
    # realpath, because every TARGET has already been through ``resolve_path`` (which realpaths).
    # Comparing a resolved target against an unresolved record path is how a run reached through a
    # symlinked output root slips past the refusal -- the two spellings never meet.
    try:
        artifact_dir = Path(os.path.realpath(run_dir))
    except (OSError, ValueError):
        artifact_dir = run_dir
    run_id = str(getattr(snapshot.record, "run_id", "")) or None
    return {"artifact_dir": artifact_dir, "run_name": run_name, "run_id": run_id}


def check_target_allowed(tool: Tool, params: dict) -> list[Path]:
    """Rule 1. Return the tool's target run dirs, or raise :class:`TargetActive`.

    THE one guard: the HTTP route reaches it through :func:`start_tool`, and a standalone CLI must
    call it rather than growing a parallel copy that drifts (section 2.8, last bullet).
    """
    targets = target_artifact_dirs(tool, params)
    active = active_run_target()
    if active is None:
        return targets
    for target in targets:
        if _overlaps(target, active["artifact_dir"]):
            raise TargetActive(target, active["run_name"], active["run_id"])
    return targets


def artifact_lock_path(artifact_dir: Path) -> Optional[Path]:
    """``<deployment runtime dir>/postprocess/<sha256(artifact_dir)>.lock``, or ``None``.

    The hash, rather than a mangled path, keeps the name bounded and filesystem-safe for artifact
    directories of any depth, and it is stable across processes because the input is the RESOLVED
    directory.
    """
    try:
        from leafmachine3.core import paths as core_paths

        base = core_paths.deployment_runtime_dir(create=True, check_filesystem=False) / _LOCK_DIRNAME
        base.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(base, 0o700)                  # the registry is user-private; keep it so
        except OSError:
            pass
    except Exception:  # noqa: BLE001 - no registry directory means no cross-process lock
        log.debug("could not resolve the postprocess lock directory", exc_info=True)
        return None
    digest = hashlib.sha256(str(artifact_dir).encode("utf-8", "surrogateescape")).hexdigest()
    return base / f"{digest}.lock"


#: In-process half of the advisory lock, one entry per artifact key. The OS file lock covers other
#: PROCESSES; this covers the server's own tool threads, whose behavior when two handles in one
#: process lock the same file is not identical on POSIX and Windows. Holding both makes the
#: guarantee the same everywhere, which is the only kind worth documenting.
_LOCAL_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


class _ArtifactLock:
    """A non-blocking advisory lock over one run's artifacts. Never waits; refuses instead."""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.path = artifact_lock_path(artifact_dir)
        self._local: Optional[threading.Lock] = None
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        key = str(self.artifact_dir)
        with _LOCAL_LOCKS_GUARD:
            local = _LOCAL_LOCKS.setdefault(key, threading.Lock())
        if not local.acquire(blocking=False):
            return False
        self._local = local
        if self.path is None:
            return True
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            log.debug("could not open the artifact lock file %s", self.path, exc_info=True)
            return True                            # in-process serialization still holds
        try:
            if not _try_lock_fd(fd):
                os.close(fd)
                self.release()
                return False
        except OSError:
            os.close(fd)
            log.debug("advisory locking is unavailable on %s", self.path, exc_info=True)
            return True
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            try:
                _unlock_fd(self._fd)
            except OSError:
                log.debug("could not release the artifact lock %s", self.path, exc_info=True)
            finally:
                os.close(self._fd)
                self._fd = None
        if self._local is not None:
            self._local.release()
            self._local = None


def _try_lock_fd(fd: int) -> bool:
    """Take an exclusive, non-blocking OS lock on ``fd``. False means somebody else holds it.

    ``fcntl`` / ``msvcrt`` are imported HERE, never at module scope: this file must import cleanly
    on both platforms, and only one of the two exists on each.
    """
    if os.name == "nt":                            # pragma: no cover - exercised on Windows only
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock_fd(fd: int) -> None:
    if os.name == "nt":                            # pragma: no cover - exercised on Windows only
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


def _acquire_artifact_locks(tool: Tool, targets: list[Path]) -> list["_ArtifactLock"]:
    """Rule 2. Lock every target for a ``read_write`` tool, or raise :class:`TargetLocked`.

    All-or-nothing: a partial acquisition is released before the refusal, so a caller that retries
    is not slowly starving itself behind locks nobody is using.
    """
    if not _runtime_v2_enabled() or tool.access != "read_write":
        return []
    held: list[_ArtifactLock] = []
    for target in targets:
        lock = _ArtifactLock(target)
        if not lock.acquire():
            for done in held:
                done.release()
            raise TargetLocked(target)
        held.append(lock)
    return held


# --------------------------------------------------------------------------- #
# Task registry
# --------------------------------------------------------------------------- #
class TaskRegistry:
    """Bounded registry of tool invocations, with a one-task-at-a-time gate per tool."""

    def __init__(self, max_tasks: int = MAX_TASKS) -> None:
        self._tasks: "OrderedDict[str, Task]" = OrderedDict()
        self._active: dict[str, str] = {}        # tool_id -> task_id
        self._lock = threading.Lock()
        self._max = max_tasks

    def start(self, tool: Tool, params: dict) -> Task:
        """Validate, guard, claim the tool, and launch the runner on a daemon thread.

        Order matters. Validation first, because the guard needs RESOLVED paths to know what the
        tool would touch. Then section 2.8's two rules -- refuse a live target, serialize two
        writers on one completed run -- and only then the per-tool claim, so a refused request
        leaves no trace in the registry. The locks are handed to the worker thread, which is what
        releases them; taking them here and releasing them there is deliberate, because the lock
        has to outlive this call for exactly as long as the tool runs.
        """
        clean = validate_params(tool, params)
        targets = check_target_allowed(tool, clean)
        locks = _acquire_artifact_locks(tool, targets)
        task = Task(id=uuid.uuid4().hex[:12], tool_id=tool.id, tool_name=tool.name, params=clean)
        try:
            with self._lock:
                active = self._active.get(tool.id)
                if active and active in self._tasks and self._tasks[active].state == "running":
                    raise ToolBusy(tool.id, active)
                self._active[tool.id] = task.id
                self._tasks[task.id] = task
                self._evict_locked()
        except BaseException:
            for lock in locks:
                lock.release()
            raise
        task.append(f"{tool.name}: starting", "INFO", tool.id)
        threading.Thread(target=self._run, args=(tool, task, locks), name=f"lm3-pp-{tool.id}",
                         daemon=True).start()
        return task

    def _run(self, tool: Tool, task: Task, locks: Optional[list] = None) -> None:
        _capture_acquire(task)
        try:
            result = tool.runner(task.params, TaskContext(task, tool))
            task.finish(result if isinstance(result, dict) else {"result": result}, None)
            task.append(f"{tool.name}: done ({len(task.outputs)} output(s))", "SUCCESS", tool.id)
        except ParamError as exc:
            task.finish(None, str(exc))
            task.append(str(exc), "ERROR", tool.id)
        except Exception as exc:  # noqa: BLE001 - surface to the task, never kill the server
            log.exception("postprocess tool %s failed", tool.id)
            task.finish(None, f"{type(exc).__name__}: {exc}")
            task.append(f"{type(exc).__name__}: {exc}", "ERROR", tool.id)
        finally:
            for stream in (sys.stdout, sys.stderr):      # drain any half-written line
                if isinstance(stream, _ThreadTee):
                    stream.flush()
            _capture_release()
            for lock in (locks or ()):             # released HERE, not in start(): see its docstring
                lock.release()
            with self._lock:
                if self._active.get(tool.id) == task.id:
                    self._active.pop(tool.id, None)

    def get(self, task_id: str) -> Task:
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(task_id)
        return task

    def list(self, limit: int = 25) -> list[dict]:
        with self._lock:
            tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: t.started_at, reverse=True)
        return [t.snapshot(log_limit=0) for t in tasks[:limit]]

    def active(self) -> dict[str, str]:
        with self._lock:
            return {k: v for k, v in self._active.items()}

    def _evict_locked(self) -> None:
        """Drop the oldest FINISHED tasks once the registry is full (never a running one)."""
        while len(self._tasks) > self._max:
            for tid, t in list(self._tasks.items()):
                if t.state != "running":
                    self._tasks.pop(tid, None)
                    break
            else:
                return


_registry = TaskRegistry()


def registry() -> TaskRegistry:
    return _registry


# --------------------------------------------------------------------------- #
# Tool: generate_stl_from_mask
# --------------------------------------------------------------------------- #
_STL_INPUTS: tuple[ToolInput, ...] = (
    ToolInput(
        key="paths", label="Mask images", type="path", default=[], multi=True, required=True,
        accepts="mask_png", must_exist=True, group="Input", important=True,
        placeholder="reports/Leaf_Original/Lamina_Mask/<specimen>__og-SEG-lamina__x_y_x_y.png",
        help="Binary-mask PNG(s) to extrude. Pick them from a finished LM3 run, or paste paths. "
             "Each mask becomes one .stl.",
    ),
    ToolInput(
        key="output_dir", label="Output folder", type="path", default=None, accepts="dir",
        must_exist=False, group="Input", important=True,
        placeholder="leave empty to write beside each mask",
        help="Where the .stl files are written. Empty writes <mask name>.stl next to its source "
             "mask. The folder is created if it does not exist.",
    ),
    ToolInput(
        key="length_mm", label="Longest dimension", type="float", default=150.0,
        min=0.1, max=5000.0, step=1.0, group="Geometry", important=True,
        help="The mask's LONGEST side is scaled to this many mm -- this sets the overall printed "
             "size. The other side follows the mask's aspect ratio.",
    ),
    ToolInput(
        key="thickness_mm", label="Thickness (z)", type="float", default=2.0,
        min=0.01, max=500.0, step=0.1, group="Geometry", important=True,
        help="Extrusion height in mm. The flat 2D shape is pushed to this depth: a circle mask "
             "becomes a cylinder, a leaf mask becomes a flat leaf slab.",
    ),
    ToolInput(
        key="fill_holes", label="Fill internal holes", type="bool", default=True,
        group="Shape cleanup", important=True,
        help="Fill fully enclosed background holes so the printed solid has no through-gaps. Turn "
             "off to keep insect damage and other perforations open.",
    ),
    ToolInput(
        key="colors", label="Foreground color(s)", type="list", item_type="color",
        default=["white"], group="Foreground selection", important=True,
        placeholder="white",
        help="Which color(s) in the PNG become the model. LM3 binary masks are white on black, so "
             "'white' is right for Binary_Masks and Lamina_Mask outputs. Accepts 'white', 'black', "
             "'#rrggbb' or 'R,G,B'; several colors are unioned into one shape.",
    ),
    ToolInput(
        key="color_tolerance", label="Color tolerance", type="int", default=0,
        min=0, max=255, step=1, group="Foreground selection",
        help="Per-channel match slack, 0-255. 0 is an exact match (correct for a clean binary "
             "mask); raise it for a JPEG-compressed or anti-aliased mask.",
    ),
    ToolInput(
        key="simplify_tolerance_px", label="Boundary smoothing", type="float", default=1.5,
        min=0.0, max=100.0, step=0.1, group="Shape cleanup",
        help="Smooths the pixel-staircase outline, in pixels, before extrusion (0 = keep every "
             "pixel step). Higher values shrink the .stl considerably and round fine serrations.",
    ),
    ToolInput(
        key="min_area_px", label="Minimum blob area", type="float", default=4.0,
        min=0.0, max=1e9, step=1.0, group="Shape cleanup",
        help="Drop foreground blobs smaller than this many square pixels -- removes speckle that "
             "would otherwise print as loose crumbs.",
    ),
)


def _run_generate_stl(params: dict, ctx: TaskContext) -> dict:
    """Drive ``generate_stl_from_mask`` one mask at a time so the tab gets live progress.

    The module's own ``run()`` loops internally with no callback, so the per-file loop is repeated
    here -- that is the only way to report N-of-M progress and stream each output path as it lands.
    ``_out_path`` is reused rather than re-derived so the naming stays identical to the CLI.
    """
    from leafmachine3.postprocessing.generate_stl_from_mask import _out_path, generate_stl

    paths: list[str] = list(params.get("paths") or [])
    outdir = params.get("output_dir")
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    queue: list[str] = []
    for p in paths:                                  # de-dupe repeated paths, like run() does
        key = str(Path(p))
        if key not in seen:
            seen.add(key)
            queue.append(p)

    # Plan the destinations UP FRONT so same-named masks from different folders cannot silently
    # overwrite each other. The picker offers e.g. Leaf_Original/Lamina_Mask AND
    # Leaf_Oriented/Lamina_Mask, whose files share a filename -- writing both into one output
    # folder would hand back fewer .stl files than the user selected, with no warning.
    planned: list[tuple[str, Path]] = []
    claimed: set[str] = set()
    for mask in queue:
        out = Path(_out_path(mask, outdir))
        if str(out) in claimed:
            tag = Path(mask).parent.name or "dup"
            cand = out.with_name(f"{out.stem}__{tag}{out.suffix}")
            n = 2
            while str(cand) in claimed:
                cand = out.with_name(f"{out.stem}__{tag}_{n}{out.suffix}")
                n += 1
            ctx.log(f"name clash on {out.name}: {Path(mask).name} will be written as {cand.name}",
                    "WARNING")
            out = cand
        claimed.add(str(out))
        planned.append((mask, out))

    total = len(planned)
    ctx.progress(0.0, f"0 / {total} masks", n_done=0, n_total=total)
    ctx.log(f"{total} mask(s) -> {outdir or 'beside each mask'}")
    ctx.log(f"length_mm={params['length_mm']}  thickness_mm={params['thickness_mm']}  "
            f"fill_holes={params['fill_holes']}  colors={params['colors']}  "
            f"tolerance={params['color_tolerance']}  simplify_px={params['simplify_tolerance_px']}  "
            f"min_area_px={params['min_area_px']}")

    results: list[dict] = []
    failures: list[dict] = []
    for i, (mask, out) in enumerate(planned):
        ctx.log(f"[{i + 1}/{total}] {Path(mask).name}")
        # the destination is derived, not submitted -- re-check it so a mask reached through a
        # symlink cannot steer the .stl outside the allowed roots
        out = resolve_path(out, must_exist=False, kind="file", label="output file")
        try:
            r = generate_stl(
                mask, out,
                colors=params["colors"],
                fill_holes=bool(params["fill_holes"]),
                color_tolerance=int(params["color_tolerance"]),
                length_mm=float(params["length_mm"]),
                thickness_mm=float(params["thickness_mm"]),
                simplify_tolerance_px=float(params["simplify_tolerance_px"]),
                min_area_px=float(params["min_area_px"]),
            )
        except Exception as exc:  # noqa: BLE001 - one bad mask must not abandon the batch
            failures.append({"mask": str(mask), "error": f"{type(exc).__name__}: {exc}"})
            ctx.log(f"    FAILED: {type(exc).__name__}: {exc}", "ERROR")
        else:
            results.append(r)
            ctx.add_output(r["stl"])
            size = " x ".join(f"{v:g}" for v in r["size_mm"])
            ctx.log(f"    {Path(r['stl']).name}  {size} mm  parts={r['n_parts']}  "
                    f"watertight={r['watertight']}")
        ctx.progress((i + 1) / total if total else 1.0,
                     f"{i + 1} / {total} masks", n_done=i + 1, n_total=total)

    if failures and not results:
        first = failures[0]["error"]
        raise RuntimeError(f"all {len(failures)} mask(s) failed -- first error: {first}")
    if failures:
        ctx.log(f"{len(failures)} of {total} mask(s) failed; {len(results)} .stl written", "WARNING")

    return {
        "n_written": len(results),
        "n_failed": len(failures),
        "output_dir": str(outdir) if outdir else None,
        "results": results,
        "failures": failures,
    }


# --------------------------------------------------------------------------- #
# Tool: generate_leaf_collage
# --------------------------------------------------------------------------- #
_COLLAGE_INPUTS: tuple[ToolInput, ...] = (
    ToolInput(
        key="run_dir", label="Run folder", type="path", default=None, accepts="dir",
        must_exist=True, required=True, group="Input", important=True,
        placeholder="examples_out/<run>  (the folder holding <run>.sqlite and reports/)",
        help="The finished LM3 run to read. Its project database supplies the archetype scores and "
             "the veto flags; its reports/ tree supplies the leaf masks.",
    ),
    ToolInput(
        key="primary_mask", label="Primary mask", type="path", default=None, accepts="mask_png",
        must_exist=True, required=True, group="Input", important=True,
        placeholder="the mask whose outline the collage fills",
        help="The mask whose white area becomes the collage's overall shape. Pick from the run's "
             "own high-scoring leaves for a leaf built out of leaves, or paste any mask PNG.",
    ),
    ToolInput(
        key="output_dir", label="Output folder", type="path", default=None, accepts="dir",
        must_exist=False, group="Input",
        placeholder="leave empty for <run>/reports/Collage",
        help="Where the collage PNG lands. Empty writes it to the run's reports/Collage folder, "
             "which is created on first use.",
    ),
    ToolInput(
        key="name", label="File name", type="string", default=None, group="Input",
        placeholder="collage__<layout>__<primary mask name>",
        help="Output file stem, without the .png. Empty names it after the layout and the primary "
             "mask so repeated runs do not overwrite each other.",
    ),

    ToolInput(
        key="min_archetype_score", label="Minimum archetype score", type="float", default=0.8,
        min=0.0, max=1.0, step=0.01, group="Which leaves", important=True,
        help="Keep leaves scoring STRICTLY above this (0-1). Leaves that failed a structural veto "
             "are excluded no matter how they scored, so this only ever narrows a clean set.",
    ),
    ToolInput(
        key="max_leaves", label="Leaf cap", type="int", default=0,
        min=0, max=5000, step=10, group="Which leaves", important=True,
        help="At most this many leaves, highest score first. 0 uses every leaf that passes.",
    ),
    ToolInput(
        key="tree", label="Leaf orientation", type="enum", default="Leaf_Oriented",
        enum=("Leaf_Oriented", "Leaf_Original"), group="Which leaves",
        help="Leaf_Oriented is rotated tip-up and reads as a coherent collage; Leaf_Original keeps "
             "each leaf as it sat on the sheet. Oriented masks exist only where the orientation "
             "stage succeeded -- any leaf without one is skipped and counted.",
    ),
    ToolInput(
        key="mask_variant", label="Mask version", type="enum", default="lamina_mask",
        enum=("lamina_mask", "lamina_holes_mask", "lamina_petiole_mask", "lamina_petiole_holes_mask"),
        group="Which leaves", important=True,
        help="Which per-leaf product to tile. lamina_mask is the lamina with its holes CUT OUT; "
             "lamina_holes_mask is the solid silhouette with holes filled; the petiole variants "
             "keep the stalk. Only the first three have an RGB cutout sibling.",
    ),

    ToolInput(
        key="style", label="Leaf rendering", type="enum", default="mask",
        enum=("mask", "rgb"), group="Appearance", important=True,
        help="mask recolors each binary mask to the color below. rgb uses the matching RGB cutout, "
             "so the collage keeps every leaf's real pixels.",
    ),
    ToolInput(
        key="color", label="Leaf color", type="color", default=[255, 255, 255],
        group="Appearance", important=True, placeholder="white, #ff0000 or 255,0,0",
        help="The color solid leaves are drawn in. Used by the mask style only -- the rgb style "
             "keeps each leaf's own pixels.",
    ),
    ToolInput(
        key="background", label="Background", type="color_or_none", default="transparent",
        group="Appearance", important=True, placeholder="transparent, black or 255,255,255",
        help="Transparent drops the collage onto any slide or poster. Pick a color instead to "
             "flatten it onto a solid sheet.",
    ),
    ToolInput(
        key="max_dim_px", label="Longest side", type="int", default=10000,
        min=256, max=30000, step=100, group="Appearance", important=True,
        help="Longest side of the output PNG in pixels; every tile is scaled to suit. A 10000 px "
             "canvas is roughly 400 MB in memory while it is being drawn.",
    ),
    ToolInput(
        key="tile_scale", label="Tile fill", type="float", default=0.98,
        min=0.1, max=1.0, step=0.01, group="Appearance",
        help="How much of its cell each leaf fills. Below 1.0 opens a little air between leaves; "
             "1.0 lets them touch their cell edges.",
    ),

    ToolInput(
        key="layout", label="Arrangement", type="enum", default="grid",
        enum=("grid", "mosaic", "organic", "puzzle"), group="Arrangement", important=True,
        help="grid solves one square cell per leaf. mosaic is a quadtree -- big tiles inside, "
             "subdivided toward the edge for a crisp outline. organic packs each leaf into the "
             "largest remaining pocket at its own scale and rotation. "
             "puzzle nests the real leaf outlines into each other so they interlock, "
             "every leaf at the same size.",
    ),
    ToolInput(
        key="ranking", label="Score placement", type="enum", default="center",
        enum=("center", "reading", "random"), group="Arrangement", important=True,
        help="center puts the best leaves deepest inside the silhouette (and, in the mosaic, on "
             "the biggest tiles). reading fills left-to-right, top-to-bottom. random shuffles.",
    ),
    ToolInput(
        key="random_seed", label="Random seed", type="int", default=0,
        min=0, max=100000, step=1, group="Arrangement",
        help="Same seed, same collage. Change it to re-roll the organic packing and the shuffle "
             "below without changing any other setting.",
    ),
    ToolInput(
        key="shuffle_top", label="Shuffle the top N", type="int", default=0,
        min=0, max=512, step=1, group="Arrangement",
        help="Shuffle the N highest-scoring leaves among themselves before placing them, so a "
             "different archetype gets the biggest, most central tile each seed. 0 turns it off.",
    ),
    ToolInput(
        key="layout_px", label="Layout resolution", type="int", default=2048,
        min=256, max=8192, step=64, group="Arrangement",
        help="Resolution the arrangement is solved at, NOT the output size. Higher tracks the "
             "silhouette's edge more finely and costs more time in the organic layout.",
    ),

    ToolInput(
        key="primary_colors", label="Primary foreground", type="list", item_type="color",
        default=["white"], group="Primary mask",
        help="Which color(s) in the primary mask count as its shape. LM3 masks are white on black.",
    ),
    ToolInput(
        key="primary_color_tolerance", label="Primary color tolerance", type="int", default=0,
        min=0, max=255, step=1, group="Primary mask",
        help="Per-channel match slack, 0-255. 0 is exact, which is right for a clean binary mask.",
    ),
    ToolInput(
        key="primary_fill_holes", label="Fill the primary mask's holes", type="bool", default=True,
        group="Primary mask",
        help="Fill the outline mask's own holes so the collage silhouette is solid. Turn off to "
             "leave insect damage open and have the leaves flow around it.",
    ),

    ToolInput(
        key="mosaic_min_cell_px", label="Mosaic smallest tile", type="float", default=24.0,
        min=2.0, max=512.0, step=1.0, group="Layout tuning",
        help="The quadtree stops subdividing below this tile size, in output pixels.",
    ),
    ToolInput(
        key="organic_rotate", label="Organic rotation", type="bool", default=True,
        group="Layout tuning",
        help="Give each packed leaf a random rotation. Off keeps every leaf upright.",
    ),
    ToolInput(
        key="organic_fill", label="Organic pocket fill", type="float", default=0.9,
        min=0.1, max=1.5, step=0.05, group="Layout tuning",
        help="Leaf size as a fraction of its pocket's inscribed circle. Above 1.0 the leaves start "
             "to overlap.",
    ),
    ToolInput(
        key="organic_max_scale", label="Organic size spread", type="float", default=3.0,
        min=1.0, max=12.0, step=0.5, group="Layout tuning",
        help="How much bigger the largest leaf may be than the average one. Without a cap the "
             "first leaf would claim the whole silhouette.",
    ),
    ToolInput(
        key="organic_gap_px", label="Organic clearance", type="float", default=2.0,
        min=0.0, max=64.0, step=1.0, group="Layout tuning",
        help="Minimum space kept between packed leaves, in output pixels.",
    ),
    ToolInput(
        key="organic_min_tile_px", label="Organic smallest tile", type="float", default=12.0,
        min=1.0, max=512.0, step=1.0, group="Layout tuning",
        help="Stop packing once the biggest free pocket falls below this, in output pixels.",
    ),
    ToolInput(
        key="leaf_order", label="Leaf order", type="enum", default="score",
        enum=("score", "random"), group="Arrangement",
        help="score gives the best-scoring leaves the most prominent spots in every layout. "
             "random removes any order from the placement, so nothing about how the leaves were "
             "gathered shows up as a pattern in the picture.",
    ),
    ToolInput(
        key="puzzle_leaf_px", label="Puzzle leaf size", type="float", default=0.0,
        min=0.0, max=4096.0, step=1.0, group="Layout tuning",
        help="Longest dimension of every leaf, in output pixels. It is measured across the white "
             "region and does not change when the leaf is rotated. 0 solves it from Puzzle fill.",
    ),
    ToolInput(
        key="puzzle_fill", label="Puzzle fill", type="float", default=0.6,
        min=0.1, max=1.2, step=0.05, group="Layout tuning",
        help="How much of the shape to cover in leaf, which is what sets the leaf size. Around "
             "0.6 every leaf still fits at one size; higher packs denser but shrinks a growing "
             "share of the leaves through the backfill.",
    ),
    ToolInput(
        key="puzzle_gap_px", label="Puzzle seam", type="float", default=2.0,
        min=0.0, max=64.0, step=1.0, group="Layout tuning",
        help="Space kept between nested leaves, in output pixels.",
    ),
    ToolInput(
        key="puzzle_angles", label="Puzzle rotations", type="int", default=8,
        min=1, max=64, step=1, group="Layout tuning",
        help="How many orientations to try per leaf. Each leaf starts from its own random angle "
             "(re-rolled by Random seed) and the nester keeps whichever one interlocks best.",
    ),
    ToolInput(
        key="puzzle_coarse", label="Puzzle search downscale", type="int", default=6,
        min=1, max=16, step=1, group="Layout tuning",
        help="The nest is searched on a map this many times smaller. Lower nests tighter and "
             "costs roughly four times as much per step.",
    ),
    ToolInput(
        key="puzzle_refine", label="Puzzle candidates", type="int", default=3,
        min=1, max=16, step=1, group="Layout tuning",
        help="How many of the best coarse spots are re-checked exactly before a leaf is placed.",
    ),
    ToolInput(
        key="puzzle_nest_px", label="Puzzle nest resolution", type="int", default=3072,
        min=256, max=8192, step=256, group="Layout tuning",
        help="Resolution the nest is solved at, independent of Layout resolution.",
    ),
    ToolInput(
        key="puzzle_overhang", label="Puzzle overhang", type="float", default=0.0,
        min=0.0, max=1.0, step=0.05, group="Layout tuning",
        help="Let leaves spill past the outline by this fraction of a leaf, for a softer edge.",
    ),
    ToolInput(
        key="puzzle_backfill_ratio", label="Puzzle backfill", type="float", default=0.72,
        min=0.0, max=0.95, step=0.02, group="Layout tuning",
        help="A leaf that no longer fits is retried at this fraction of its size, so smaller "
             "leaves close the leftover gaps. 0 keeps every leaf at exactly one size.",
    ),
    ToolInput(
        key="puzzle_min_leaf_px", label="Puzzle smallest leaf", type="float", default=12.0,
        min=1.0, max=512.0, step=1.0, group="Layout tuning",
        help="Never shrink a backfilled leaf below this, in output pixels.",
    ),
    ToolInput(
        key="puzzle_max_shrink", label="Puzzle shrink steps", type="int", default=3,
        min=0, max=8, step=1, group="Layout tuning",
        help="Cap on how many times the backfill may shrink a leaf, so none becomes a speck.",
    ),
    ToolInput(
        key="tmp_dir", label="Scratch folder", type="path", default=None, accepts="dir",
        must_exist=False, group="Layout tuning",
        placeholder="leave empty to stage beside the run",
        help="Where the collage is staged while it encodes, in a _leaf_collage subfolder. Empty "
             "stages beside the run; a read-only run falls back to the output folder.",
    ),
    ToolInput(
        key="write_manifest", label="Write the leaf manifest", type="bool", default=True,
        group="Layout tuning",
        help="Also write a .json next to the collage listing every placed leaf, its score, and "
             "where it landed.",
    ),
)


def _run_generate_collage(params: dict, ctx: TaskContext) -> dict:
    """Drive ``generate_leaf_collage`` once, forwarding its progress callback to the tab.

    Unlike the STL builder there is no per-file loop to re-implement here: a collage is ONE output
    built from N inputs, so the module reports its own phases (database, layout, render, write)
    through ``on_progress`` and this just relays them.
    """
    from leafmachine3.postprocessing.generate_leaf_collage import (
        _DEFAULTS as _COLLAGE_DEFAULTS, _out_path, generate_collage,
    )

    def p(key):
        """A cleared form field arrives as None; fall back to the tool's documented default.

        ``validate_params`` deliberately keeps an emptied optional as None (a null output_dir MEANS
        something), so the runner -- not the validator -- is where a blank number box has to become
        the default again. Without this, clearing any numeric field died in ``float(None)``.
        """
        value = params.get(key)
        return _COLLAGE_DEFAULTS[key] if value is None else value

    run_dir = params["run_dir"]
    primary = params["primary_mask"]
    outdir = params.get("output_dir")
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)

    out = _out_path(run_dir, primary, outdir, params.get("name"), str(p("layout")))
    # the destination is DERIVED (from the run dir or the settings), not submitted -- re-check it so
    # a run reached through a symlink cannot steer the collage outside the allowed roots
    out = resolve_path(out, must_exist=False, kind="file", label="output file")

    ctx.log(f"run={Path(run_dir).name}  primary={Path(primary).name}")
    ctx.log(f"layout={p('layout')}  ranking={p('ranking')}  style={p('style')}  "
            f"tree={p('tree')}  variant={p('mask_variant')}")
    ctx.log(f"score>{p('min_archetype_score')}  cap={p('max_leaves') or 'none'}  "
            f"max_dim={p('max_dim_px')}px  color={p('color')}  bg={p('background')}")

    r = generate_collage(
        run_dir, primary, out,
        min_archetype_score=float(p("min_archetype_score")),
        max_leaves=int(p("max_leaves") or 0),
        tree=str(p("tree")), mask_variant=str(p("mask_variant")),
        style=str(p("style")), layout=str(p("layout")), ranking=str(p("ranking")),
        max_dim_px=int(p("max_dim_px")), color=p("color"), background=p("background"),
        primary_colors=p("primary_colors"),
        primary_color_tolerance=int(p("primary_color_tolerance")),
        primary_fill_holes=bool(p("primary_fill_holes")),
        tile_scale=float(p("tile_scale")), random_seed=int(p("random_seed") or 0),
        shuffle_top=int(p("shuffle_top") or 0), leaf_order=str(p("leaf_order")),
        mosaic_min_cell_px=float(p("mosaic_min_cell_px")),
        organic_rotate=bool(p("organic_rotate")), organic_fill=float(p("organic_fill")),
        organic_gap_px=float(p("organic_gap_px")),
        organic_min_tile_px=float(p("organic_min_tile_px")),
        organic_max_scale=float(p("organic_max_scale")),
        puzzle_leaf_px=float(p("puzzle_leaf_px") or 0), puzzle_fill=float(p("puzzle_fill")),
        puzzle_gap_px=float(p("puzzle_gap_px")), puzzle_angles=int(p("puzzle_angles")),
        puzzle_coarse=int(p("puzzle_coarse")), puzzle_refine=int(p("puzzle_refine")),
        puzzle_nest_px=int(p("puzzle_nest_px")), puzzle_overhang=float(p("puzzle_overhang")),
        puzzle_backfill_ratio=float(p("puzzle_backfill_ratio")),
        puzzle_min_leaf_px=float(p("puzzle_min_leaf_px")),
        puzzle_max_shrink=int(p("puzzle_max_shrink")),
        layout_px=int(p("layout_px")), tmp_dir=params.get("tmp_dir"),
        write_manifest=bool(p("write_manifest")),
        on_progress=lambda frac, msg, done, total: ctx.progress(
            frac, msg, n_done=done or None, n_total=total or None),
    )

    ctx.add_output(r["collage"])
    if r.get("manifest"):
        ctx.add_output(r["manifest"])
    ctx.log(f"{r['n_placed']} leaf/leaves placed of {r['n_passing']} that passed "
            f"({r['size_px'][0]} x {r['size_px'][1]} px, scores "
            f"{r['score_range'][0]}-{r['score_range'][1]})")
    if r["n_missing_files"]:
        ctx.log(f"{r['n_missing_files']} qualifying leaf/leaves had no {p('tree')} file and "
                f"were skipped", "WARNING")
    if r["n_merged_detections"]:
        ctx.log(f"{r['n_merged_detections']} leaf/leaves shared a mask file with another leaf "
                f"(multi-instance detection) and were collapsed to one tile", "WARNING")
    return {"n_written": 1, "n_failed": 0, "output_dir": str(Path(r["collage"]).parent),
            "results": [r], "failures": []}


def _build_registry() -> "OrderedDict[str, Tool]":
    """The tool registry. ADD A TOOL HERE -- one entry plus its runner, no frontend change."""
    tools = OrderedDict()
    tools["generate_stl_from_mask"] = Tool(
        id="generate_stl_from_mask",
        name="STL 3D-file builder",
        icon="cube",
        description=(
            "Turn a binary-mask PNG into a 3D-printable .stl by extruding the 2D shape to a fixed "
            "z thickness. A leaf mask becomes a flat leaf slab; the model is scaled so the mask's "
            "longest dimension equals the requested length. Watertight output via trimesh."
        ),
        inputs=_STL_INPUTS,
        outputs_description=(
            "One watertight .stl per input mask, named <mask name>.stl -- written to the output "
            "folder, or beside each source mask when that is left empty. Each result reports its "
            "bounding-box size in mm, the number of solid parts, and whether the mesh is watertight."
        ),
        runner=_run_generate_stl,
        settings_key="generate_stl_from_mask",
        access="read_write",                 # writes .stl beside the masks unless redirected
        resource="cpu",                      # trimesh + numpy; no torch, no onnxruntime
        target_keys=("paths", "output_dir"),
        module="leafmachine3.postprocessing.generate_stl_from_mask",
        cli=(".venv_LM3/bin/python -m leafmachine3.postprocessing.generate_stl_from_mask "
             "--config postprocessing_settings.yaml --paths <mask.png>"),
    )
    tools["generate_leaf_collage"] = Tool(
        id="generate_leaf_collage",
        name="Leaf collage builder",
        icon="leaf",
        description=(
            "Tile a finished run's best leaf masks into the shape of one primary mask -- a leaf "
            "built out of leaves. Leaves are ranked by their bilateral-symmetry archetype score "
            "and the vetoed ones are never used, so only clean, well-traced leaves appear. Three "
            "arrangements: a solved grid, a quadtree mosaic, or organic packing."
        ),
        inputs=_COLLAGE_INPUTS,
        outputs_description=(
            "One PNG (transparent or on a solid background) whose outline is the primary mask and "
            "whose substance is the run's high-scoring leaves -- written to reports/Collage unless "
            "an output folder is set. A matching .json lists every placed leaf, its archetype "
            "score, and where it landed, so any collage can be traced back to its specimens."
        ),
        runner=_run_generate_collage,
        settings_key="generate_leaf_collage",
        access="read_write",                 # writes reports/Collage into the run it reads
        resource="cpu",                      # PIL + numpy over finished masks
        target_keys=("run_dir", "output_dir"),
        module="leafmachine3.postprocessing.generate_leaf_collage",
        cli=(".venv_LM3/bin/python -m leafmachine3.postprocessing.generate_leaf_collage "
             "--config postprocessing_settings.yaml --run-dir <run> --primary-mask <mask.png>"),
    )
    return tools


_TOOLS: "OrderedDict[str, Tool]" = _build_registry()


def list_tools() -> list[dict]:
    """The self-describing registry (defaults refreshed from postprocessing_settings.yaml)."""
    return [t.describe() for t in _TOOLS.values()]


def get_tool(tool_id: str) -> Tool:
    tool = _TOOLS.get(str(tool_id))
    if tool is None:
        raise KeyError(tool_id)
    return tool


def start_tool(tool_id: str, params: Any) -> Task:
    """Validate, guard, and launch ``tool_id``.

    Raises ``KeyError`` / :class:`ParamError` / :class:`ToolBusy` / :class:`TargetActive` /
    :class:`TargetLocked`. The last two are section 2.8's concurrency policy and are the reason a
    standalone CLI must come through here (or through :func:`check_target_allowed`) rather than
    calling a runner directly.
    """
    return _registry.start(get_tool(tool_id), params or {})


# --------------------------------------------------------------------------- #
# Run discovery + mask picking
# --------------------------------------------------------------------------- #
# Where the Reporter writes BINARY masks (see leafmachine3/modules/reporter.py and
# leafmachine3/reporting/leaf_products.py). "Leaf_Original"/"Leaf_Oriented" is the current tree;
# "Original"/"Oriented" is the earlier layout still present in older runs.
_MASK_PARENTS: tuple[tuple[str, str], ...] = (
    ("Specimen_Masks", "sheet"),        # current: BOTH mask trees share this parent
    ("Binary_Masks", "sheet"),          # older runs: Binary_Masks_Full_Image__<Cls>, Binary_Masks__<Cls>
    ("Leaf_Original", "leaf"),          # Lamina_Mask, LaminaPetiole_Mask, Lamina_Holes_Mask, ...
    ("Leaf_Oriented", "leaf"),
    ("Original", "leaf"),
    ("Oriented", "leaf"),
)
_RGB_PARENTS: tuple[str, ...] = ("RGB_Masks",)
_MASK_SUFFIXES = {".png", ".tif", ".tiff", ".bmp"}
# The tag is 2+ hyphen-joined words, not exactly 2: leaf products carry a tree tag in front of the
# prefix (og-SEG-lamina) and the ECT visuals qualify theirs (ECT-radial-overlay).
_FNAME_RE = re.compile(
    r"^(?P<stem>.+?)__(?P<tag>[A-Za-z]+(?:-[A-Za-z]+)+)(?:__(?P<box>\d+_\d+_\d+_\d+))?$"
)


def _is_run_dir(p: Path) -> bool:
    """A run directory is one that holds a project ledger or a reports tree."""
    try:
        if not p.is_dir():
            return False
        return ((p / "reports").is_dir() or (p / f"{p.name}.sqlite").is_file()
                or (p / "run.sqlite").is_file())
    except OSError:
        return False


def _iter_child_dirs(p: Path) -> Iterator[Path]:
    try:
        for child in sorted(p.iterdir()):
            if child.name.startswith(".") or child.name in _SKIP_DIRS:
                continue
            try:
                if child.is_dir():
                    yield child
            except OSError:
                continue
    except (OSError, PermissionError):
        return


def discover_runs(max_runs: int = 500) -> list[dict]:
    """Find LM3 run directories under the allowed roots (depth 2 -- roots hold output dirs)."""
    found: "OrderedDict[str, dict]" = OrderedDict()

    def note(p: Path) -> None:
        real = Path(os.path.realpath(p))
        key = str(real)
        if key in found:
            return
        try:
            mtime = real.stat().st_mtime
        except OSError:
            mtime = 0.0
        found[key] = {
            "name": real.name,
            "path": key,
            "has_reports": (real / "reports").is_dir(),
            "mtime": round(mtime, 3),
        }

    for root in allowed_roots():
        if len(found) >= max_runs:
            break
        if _is_run_dir(root):
            note(root)
        for child in _iter_child_dirs(root):
            if len(found) >= max_runs:
                break
            if _is_run_dir(child):
                note(child)
                continue
            for grand in _iter_child_dirs(child):     # <output dir>/<run> and <jobs>/<id>/run
                if len(found) >= max_runs:
                    break
                if _is_run_dir(grand):
                    note(grand)

    runs = list(found.values())
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def resolve_run(run: Any) -> Path:
    """Resolve a run NAME (or a path to one) to its directory, inside the allowed roots."""
    if run is None or not str(run).strip():
        raise ParamError("run: required -- pass a run name or a run directory")
    raw = str(run).strip()
    if os.sep in raw or raw.startswith("~") or (os.altsep and os.altsep in raw):
        p = resolve_path(raw, must_exist=True, kind="dir", label="run")
        if not _is_run_dir(p):
            raise ParamError(f"run: {p} is not an LM3 run directory (no reports/ or project DB)")
        return p
    matches = [r for r in discover_runs() if r["name"] == raw]
    if not matches:
        known = ", ".join(r["name"] for r in discover_runs()[:12]) or "none found"
        raise ParamError(f"run: no run named {raw!r} under the allowed roots. Known runs: {known}")
    return Path(matches[0]["path"])


def _pretty(name: str) -> str:
    """``Leaf_Original/Lamina_Mask`` -> ``Leaf Original / Lamina Mask`` for a group heading."""
    return " / ".join(part.replace("_", " ").strip() for part in name.split("/"))


def _describe_mask(path: Path) -> dict:
    """One candidate mask: enough for a picker row without opening the image."""
    try:
        st = path.stat()
        size, mtime = st.st_size, round(st.st_mtime, 3)
    except OSError:
        size, mtime = None, None
    m = _FNAME_RE.match(path.stem)
    box = None
    if m and m.group("box"):
        box = [int(v) for v in m.group("box").split("_")]
    return {
        "name": path.name,
        "path": str(path),
        "specimen": m.group("stem") if m else path.stem,
        "tag": m.group("tag") if m else None,       # e.g. og-SEG-lamina / MaskFull-leaf
        "box": box,                                  # [x1, y1, x2, y2] in working-image pixels
        "size_bytes": size,
        "mtime": mtime,
    }


def _archetype_index(root: Path, min_score: float) -> tuple[Optional[dict], str]:
    """``({(stem, box): score}, "")`` for a run's non-vetoed leaves, or ``(None, why-not)``.

    A missing ``bilateral_symmetry`` table is NOT an error here -- the picker degrades to listing
    every mask with an explanatory note, which is far more useful than a 422 in a modal.
    """
    try:
        from leafmachine3.postprocessing.generate_leaf_collage import leaf_scores

        return leaf_scores(root, min_score=float(min_score)), ""
    except Exception as exc:  # noqa: BLE001 - no scores is a degraded list, not a failed request
        log.info("no archetype scores for %s: %s", root, exc)
        return None, (f"no archetype scores in this run ({type(exc).__name__}) -- showing every "
                      f"mask. Re-run the project with modules.bilateral_symmetry.enabled: true.")


def pick_masks(run: Any = None, *, query: str = "", include_rgb: bool = False,
               per_group: int = 200, limit: int = 2000,
               min_archetype_score: Optional[float] = None) -> dict:
    """List candidate input masks inside a finished run's ``reports`` tree.

    Grouped by the folder that produced them (``Specimen_Masks/Binary_Masks_Specimen__Leaf``,
    ``Leaf_Original/Lamina_Mask``, ...) so a user picks "the fitted lamina masks" rather than
    hunting a path. RGB cutouts are excluded by default: the STL builder selects foreground BY
    COLOR, and a cutout's background is the same black as its holes.

    ``min_archetype_score`` narrows the list to leaves that cleared every structural veto AND
    scored above it, annotates each with its score, and orders them best-first -- which is how the
    collage builder's primary-mask picker offers "one of the good ones" instead of all of them.
    Non-leaf folders (whole-sheet masks) hold nothing scoreable and drop out when it is set.
    """
    runs = discover_runs()
    if run is None or not str(run).strip():
        return {
            "ready": False, "run": None, "root": None, "reports_dir": None,
            "groups": [], "n_groups": 0, "n_masks": 0, "truncated": False,
            "runs": runs, "roots": [str(r) for r in allowed_roots()],
            "message": "pick a run to list its masks",
        }

    root = resolve_run(run)
    reports = root / "reports"
    per_group = max(1, min(int(per_group or 200), 2000))
    limit = max(1, min(int(limit or 2000), 20000))
    needle = str(query or "").strip().lower()

    scores: Optional[dict] = None
    score_note = ""
    if min_archetype_score is not None:
        scores, score_note = _archetype_index(root, float(min_archetype_score))

    parents = list(_MASK_PARENTS) + ([(p, "rgb") for p in _RGB_PARENTS] if include_rgb else [])
    groups: list[dict] = []
    n_masks = 0
    truncated = False

    if reports.is_dir():
        for parent_name, kind in parents:
            if scores is not None and kind != "leaf":
                continue                        # whole-sheet masks are not per-leaf, so unscoreable
            parent = reports / parent_name
            if not parent.is_dir():
                continue
            for sub in _iter_child_dirs(parent):
                # leaf-product trees also hold RGB folders; keep only the mask ones
                if kind == "leaf" and not sub.name.endswith("_Mask"):
                    continue
                # Specimen_Masks holds the RGB tree alongside the binary one, so the
                # binary-vs-RGB split that used to be one of directory NAME is now one of
                # child name. Without this, RGB cutouts would be offered as masks whenever
                # img_ext is a lossless suffix, and the STL builder keys on color.
                if kind == "sheet" and not include_rgb and sub.name.startswith("RGB_"):
                    continue
                candidates: list[dict] = []
                for f in sorted(sub.iterdir()):
                    try:
                        if not f.is_file() or f.suffix.lower() not in _MASK_SUFFIXES:
                            continue
                    except OSError:
                        continue
                    if needle and needle not in f.name.lower():
                        continue
                    described = _describe_mask(f)
                    if scores is not None:
                        score = scores.get((described["specimen"], tuple(described["box"] or ())))
                        if score is None:
                            continue            # vetoed, below the threshold, or not a leaf at all
                        described["score"] = round(float(score), 4)
                    candidates.append(described)
                total_here = len(candidates)
                if not total_here:
                    continue
                # Rank BEFORE truncating, so a capped page is the best of the folder rather than
                # the alphabetically-first slice of it.
                if scores is not None:
                    candidates.sort(key=lambda m: -float(m.get("score") or 0.0))
                room = min(per_group, max(0, limit - n_masks))
                files = candidates[:room]
                n_masks += len(files)
                if len(files) < total_here:
                    truncated = True
                key = f"{parent_name}/{sub.name}"
                groups.append({
                    "key": key,
                    "label": _pretty(key),
                    "kind": kind,
                    "dir": str(sub),
                    "n": total_here,
                    "n_listed": len(files),
                    "masks": files,
                })

    groups.sort(key=lambda g: (g["kind"] != "leaf", g["key"]))
    return {
        "ready": True,
        "run": root.name,
        "root": str(root),
        "reports_dir": str(reports) if reports.is_dir() else None,
        "groups": groups,
        "n_groups": len(groups),
        "n_masks": n_masks,
        "truncated": truncated,
        "query": query or "",
        "include_rgb": bool(include_rgb),
        "min_archetype_score": float(min_archetype_score) if min_archetype_score is not None else None,
        "scored": scores is not None,
        "n_scored_leaves": len(scores) if scores is not None else None,
        "runs": runs,
        "roots": [str(r) for r in allowed_roots()],
        "message": score_note or ("" if groups else (
            f"no mask folders under {reports} -- run the Reporter module with report.masks or "
            f"report.leaf_products enabled"
        )),
    }


def context() -> dict:
    """Everything the tab needs to render its header without a second round trip."""
    return {
        "settings_path": str(_settings_path()),
        "settings_exists": _settings_path().is_file(),
        "roots": [str(r) for r in allowed_roots()],
        "cwd": str(Path.cwd()),
        "tools": [t.id for t in _TOOLS.values()],
        "active": _registry.active(),
        # The run a pipeline is writing right now, so the tab can gray out its own targets instead
        # of finding out through a 409 (section 2.8). ``null`` when nothing holds the deployment.
        "active_run": _describe_active_run(),
        "runs": discover_runs(),
    }


def _describe_active_run() -> Optional[dict]:
    target = active_run_target()
    if target is None:
        return None
    return {"artifact_dir": str(target["artifact_dir"]), "run_name": target["run_name"],
            "run_id": target["run_id"]}


# --------------------------------------------------------------------------- #
# SSE
# --------------------------------------------------------------------------- #
def _frame(kind: str, **payload: Any) -> str:
    return "data: " + json.dumps({"type": kind, **payload}, default=str) + "\n\n"


def _watch_state(snap: dict) -> tuple:
    """The fields whose change is worth a progress frame (log lines ride their own frame)."""
    return (snap["state"], snap["progress"], snap["message"], snap["n_done"], snap["n_outputs"])


def sse_frames(task_id: str, *, poll_s: float = SSE_POLL_S,
               max_seconds: float = SSE_MAX_S) -> Iterator[str]:
    """Blocking generator of SSE frames for one task (the async route mirrors this).

    Frame envelope, switch on ``type``:
      {"type":"hello",    "task": <snapshot with a log tail>}     first frame only
      {"type":"log",      "lines":[{seq,t,level,src,msg}, ...]}   new output since the last frame
      {"type":"progress", "task": <snapshot, log omitted>}        state/progress/outputs moved
      {"type":"ping",     "t": <unix s>}                          keep-alive
      {"type":"done",     "task": <final snapshot>}               terminal; the stream then ends
    """
    task = _registry.get(task_id)
    snap = task.snapshot(since=None, log_limit=HELLO_LOG_TAIL)
    yield _frame("hello", task=snap)
    cursor = snap["log_seq"]
    watched = _watch_state(snap)
    deadline = time.time() + max_seconds
    last_ping = time.time()

    while time.time() < deadline:
        snap = task.snapshot(since=cursor, log_limit=500)
        if snap["log"]:
            cursor = snap["log"][-1]["seq"]
            yield _frame("log", lines=snap["log"])
        now = _watch_state(snap)
        if now != watched:
            watched = now
            yield _frame("progress", task={k: v for k, v in snap.items() if k != "log"})
        if snap["state"] != "running":
            final = task.snapshot(since=cursor, log_limit=500)
            if final["log"]:
                yield _frame("log", lines=final["log"])
            yield _frame("done", task={k: v for k, v in final.items() if k != "log"})
            return
        if time.time() - last_ping >= SSE_PING_S:
            last_ping = time.time()
            yield _frame("ping", t=round(time.time(), 3))
        time.sleep(poll_s)

    yield _frame("done", task={k: v for k, v in task.snapshot(log_limit=0).items() if k != "log"})


# --------------------------------------------------------------------------- #
# FastAPI router (built lazily; requires the `server` extra)
# --------------------------------------------------------------------------- #
def _expected_token() -> Optional[str]:
    """The Bearer secret the app is using, or None when the server never minted one."""
    try:
        from leafmachine3.server.app import _server_token

        return _server_token()
    except Exception:  # noqa: BLE001 - app.py optional / not yet initialized
        return os.environ.get("LM3_SERVER_TOKEN") or None


def router(dependencies: Optional[list] = None) -> Any:
    """Build the ``/v1/postprocess`` APIRouter.

    ``fastapi`` is imported lazily, exactly like ``leafmachine3.server.app.create_app``, so a base
    install without the ``server`` extra can still import this module. Pass the app's auth
    dependency through ``dependencies`` (``[Depends(require_token)]``) to protect the routes.

    The SSE route does NOT use that dependency: ``EventSource`` cannot set an Authorization header,
    so it takes the same secret as ``?token=`` (or the header, when a client can send one). It is
    only mounted with auth when ``dependencies`` is non-empty, so the protection level matches.
    """
    import asyncio

    from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query
    from fastapi.responses import StreamingResponse

    deps = list(dependencies or [])
    api = APIRouter(prefix="/v1/postprocess", tags=["postprocess"])

    async def stream_token(
        token: Optional[str] = Query(default=None, description="Bearer secret, for EventSource"),
        authorization: str = Header(default=""),
    ) -> None:
        """Auth for the SSE route: ``?token=`` OR the Authorization header."""
        if not deps:
            return
        expected = _expected_token()
        if not expected:
            # FAIL CLOSED -- see the same note in progress_api._token_ok. An unauthenticated
            # task stream would leak tool stdout, which includes filesystem paths.
            raise HTTPException(status_code=401,
                                detail="this LM3 server has no token configured; the stream is refused")
        if token and secrets.compare_digest(str(token), expected):
            return
        if authorization and secrets.compare_digest(authorization, f"Bearer {expected}"):
            return
        raise HTTPException(status_code=401, detail="invalid or missing token")

    # -- registry ----------------------------------------------------------- #
    # Plain ``def``: describe() refreshes each tool's defaults from postprocessing_settings.yaml,
    # so this reads and parses a file. Starlette runs it in its threadpool.
    @api.get("/tools", dependencies=deps)
    def get_tools() -> list:
        return list_tools()

    @api.get("/context", dependencies=deps)
    async def get_context() -> dict:
        from fastapi.concurrency import run_in_threadpool

        return await run_in_threadpool(context)          # discover_runs() touches the filesystem

    # -- running ------------------------------------------------------------ #
    @api.post("/run", dependencies=deps)
    async def run_tool(payload: dict = Body(default={})) -> dict:
        from fastapi.concurrency import run_in_threadpool

        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="body must be a JSON object")
        tool_id = payload.get("tool_id") or payload.get("tool") or payload.get("id")
        if not tool_id:
            raise HTTPException(status_code=422, detail="tool_id is required")
        params = payload.get("params", {})
        try:
            # validate_params() realpath+stats EVERY supplied path, and a multi-path input can
            # legitimately carry thousands (pick_masks lists up to 2000) -- that is filesystem
            # work, so it goes to a thread rather than onto the event loop.
            task = await run_in_threadpool(start_tool, str(tool_id), params)
        except KeyError:
            known = ", ".join(_TOOLS)
            raise HTTPException(status_code=404, detail=f"unknown tool {tool_id!r}. Known: {known}")
        except ToolBusy as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": str(exc), "tool_id": exc.tool_id, "task_id": exc.task_id},
            )
        except TargetActive as exc:
            # 409, not 403: this is a temporal conflict, and the same request succeeds unchanged
            # once the pipeline finishes. ``reason`` lets the tab say WHY without parsing prose.
            raise HTTPException(
                status_code=409,
                detail={"message": str(exc), "reason": "target_active",
                        "artifact_dir": str(exc.artifact_dir), "run_name": exc.run_name,
                        "run_id": exc.run_id},
            )
        except TargetLocked as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": str(exc), "reason": "target_locked",
                        "artifact_dir": str(exc.artifact_dir)},
            )
        except ParamError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"task_id": task.id, "tool_id": task.tool_id, "state": task.state,
                "started_at": round(task.started_at, 3)}

    @api.get("/tasks", dependencies=deps)
    async def get_tasks(limit: int = 25) -> dict:
        return {"tasks": _registry.list(max(1, min(int(limit), MAX_TASKS))),
                "active": _registry.active()}

    @api.get("/tasks/{task_id}", dependencies=deps)
    async def get_task(task_id: str, since: Optional[int] = None, log_limit: int = 400) -> dict:
        try:
            task = _registry.get(task_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown task")
        return task.snapshot(since=since, log_limit=max(0, min(int(log_limit), MAX_LOG_LINES)))

    # `response_class=` + a plain `Any` return annotation, NOT `-> "StreamingResponse"`: with
    # `from __future__ import annotations` every annotation is a string, and FastAPI would try to
    # build a response MODEL out of that unresolvable forward ref -- which makes app.openapi()
    # (and therefore /docs and /openapi.json) raise PydanticUserError for the whole server.
    @api.get("/tasks/{task_id}/events", dependencies=[Depends(stream_token)],
             response_class=StreamingResponse)
    async def task_events(task_id: str) -> Any:
        try:
            task = _registry.get(task_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown task")

        async def gen() -> Any:
            # Same frame sequence as sse_frames(), paced with asyncio.sleep so the event loop keeps
            # serving: every read here is an in-memory snapshot copy, never a blocking call.
            snap = task.snapshot(since=None, log_limit=HELLO_LOG_TAIL)
            yield _frame("hello", task=snap)
            cursor = snap["log_seq"]
            watched = _watch_state(snap)
            deadline = time.time() + SSE_MAX_S
            last_ping = time.time()
            while time.time() < deadline:
                snap = task.snapshot(since=cursor, log_limit=500)
                if snap["log"]:
                    cursor = snap["log"][-1]["seq"]
                    yield _frame("log", lines=snap["log"])
                now = _watch_state(snap)
                if now != watched:
                    watched = now
                    yield _frame("progress", task={k: v for k, v in snap.items() if k != "log"})
                if snap["state"] != "running":
                    final = task.snapshot(since=cursor, log_limit=500)
                    if final["log"]:
                        yield _frame("log", lines=final["log"])
                    yield _frame("done", task={k: v for k, v in final.items() if k != "log"})
                    return
                if time.time() - last_ping >= SSE_PING_S:
                    last_ping = time.time()
                    yield _frame("ping", t=round(time.time(), 3))
                await asyncio.sleep(SSE_POLL_S)
            yield _frame("done", task={k: v for k, v in task.snapshot(log_limit=0).items()
                                       if k != "log"})

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # -- input picking ------------------------------------------------------ #
    @api.post("/pick-masks", dependencies=deps)
    async def post_pick_masks(payload: dict = Body(default={})) -> dict:
        from fastapi.concurrency import run_in_threadpool

        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="body must be a JSON object")
        raw_min = payload.get("min_archetype_score")
        try:
            return await run_in_threadpool(
                pick_masks,
                payload.get("run"),
                query=str(payload.get("query") or ""),
                include_rgb=bool(payload.get("include_rgb", False)),
                per_group=int(payload.get("per_group") or 200),
                limit=int(payload.get("limit") or 2000),
                min_archetype_score=None if raw_min is None else float(raw_min),
            )
        except ParamError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    return api


__all__ = [
    "ParamError", "ToolBusy", "TargetActive", "TargetLocked", "Tool", "ToolInput", "Task",
    "TaskContext", "TaskRegistry",
    "allowed_roots", "resolve_path", "validate_params", "list_tools", "get_tool", "start_tool",
    "registry", "discover_runs", "resolve_run", "pick_masks", "context", "sse_frames", "router",
    "active_run_target", "check_target_allowed", "target_artifact_dirs", "artifact_lock_path",
]
