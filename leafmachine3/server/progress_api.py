"""leafmachine3.server.progress_api -- live LM3 run status + console logs for the app.

This is the data source behind the app's PRIMARY tab (top half = what LM3 is doing right
now, bottom half = the console) and behind the global stage bar pinned above the tabs.

WHERE THE TRUTH LIVES. LM3 keeps no in-memory progress object an HTTP handler could read:
the run happens inside :func:`leafmachine3.machine3.machine3`, and the only two things it
publishes as it goes are

  1. the project SQLite ledger  ``<output>/<run_name>/<run_name>.sqlite``
     - ``project_status``  one row per canonical stage: state / n_total / n_done / timestamps
     - ``image_status``    one row per (specimen x stage) checkpoint, written in the SAME
       transaction as the stage's own rows, so it is exact and crash-consistent
  2. the run log ``<output>/<run_name>/logs/lm3.log``

so this module READS BOTH, read-only, and never touches the running pipeline. The DB is
opened ``file:...?mode=ro`` over WAL, which is why a 2 Hz poll cannot block the single
writer. The log is TAILED from the file rather than through a logging handler because
``core.logging_setup.start_logging`` strips every handler off the ``leafmachine3`` logger at
run start and sets ``propagate = False`` -- a handler installed by the server beforehand is
silently discarded, the file is not.

WHAT IS MEASURED VS WHAT IS DERIVED. Every payload carries its own provenance so the UI
never has to guess and nobody is tempted to "fix" an inference later:

  MEASURED   per-module done/error counts, no_work counts, stage timestamps, this session's
             completion count and therefore the rate, the image total, and (via
             ``server.metrics``) the real worker PROCESSES with their CPU / RSS / VRAM.
  DERIVED    the true per-module denominator (``project_status.n_total`` is the PENDING count
             at stage start, not the image count -- see ``_module_totals``), exec mode and
             worker count for thread-pooled stages (no child processes exist to observe, so
             these come from ``hardware_settings.yaml`` + the executor's own selection rules),
             how many workers are busy (the executor submits every item up front, so
             ``busy = min(workers, outstanding)`` is a property of the code, not a guess),
             the weighted overall percentage, and the ETAs.
  UNKNOWABLE which worker is holding which specimen. Nothing anywhere records it. Worker
             rows therefore report ``item_label: null`` forever; do not invent one.

Public surface (all plain Python, importable without the ``server`` extra):
    status(...)          -> the full live snapshot (the shape the front end codes against)
    list_runs()          -> every run found under the configured output dir, newest first
    log_tail(...)        -> the last N parsed console lines
    status_frames(...)   -> synchronous SSE generator of status frames
    log_frames(...)      -> synchronous SSE generator of console frames
    bind_run(...)        -> point the API at a specific run (the integrator calls this)
    bind_job_source(...) -> hand the API a callable returning the live JobManager job
    active_run()         -> the run everything above resolved to (results_api can reuse it)
    router(...)          -> the FastAPI APIRouter the integrator mounts on app.py
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import yaml

log = logging.getLogger("leafmachine3.server.progress")

# Cadence. The status stream pushes at 2 Hz while a run is live and backs off to a slow
# heartbeat when nothing is happening, so an idle app costs ~nothing.
STATUS_INTERVAL_S = float(os.environ.get("LM3_STATUS_INTERVAL_S", "0.5"))
STATUS_IDLE_INTERVAL_S = float(os.environ.get("LM3_STATUS_IDLE_INTERVAL_S", "2.0"))
LOG_INTERVAL_S = float(os.environ.get("LM3_LOG_INTERVAL_S", "0.35"))
HEARTBEAT_S = 15.0                      # SSE comment frames keep intermediaries from timing out

# A run whose ledger says "running" but which has produced nothing for this long is reported
# with ``stale: true``. LM3 only repairs a crashed 'running' row on the NEXT start
# (``ProjectDB.reclaim_running``), so without this a killed run reads as live forever.
STALE_AFTER_S = float(os.environ.get("LM3_STATUS_STALE_AFTER_S", "180"))

# Two module starts belong to the same session when the second begins within this long of the
# first one FINISHING. Measured from finished_at (not started_at) so an hours-long module can
# never split a session; the gap only has to cover the inter-stage barrier (GPU cleanup +
# collect_items). The log's "LM3 start" marker wins whenever the log file is readable.
SESSION_GAP_S = float(os.environ.get("LM3_STATUS_SESSION_GAP_S", "900"))

DEFAULT_LOG_BACKFILL = 400              # console lines replayed when a client connects
_MAX_PARTIAL_LINE = 1 << 20             # flush an unterminated line past 1 MiB rather than grow
_LOG_BACKFILL_BYTES = 512 * 1024        # how far back to seek for the backfill block


# --------------------------------------------------------------------------- #
# The canonical module table
# --------------------------------------------------------------------------- #
# Mirrors ``leafmachine3.pipeline.STAGE_ORDER``. The live classes are the source of truth and
# are read whenever they import; this literal is the fallback for a base install where a
# module's heavy dependency is missing, exactly like ``core.db._CANONICAL_STAGE_KEYS``.
# ``serial`` marks a thread-pooled CPU stage that nonetheless runs SERIALLY because it warm-
# loads a model -- see the executor rule quoted in :func:`_plan_exec`.
_STATIC_MODULES: tuple[dict, ...] = (
    {"key": "mp_conversion_factor",  "name": "MP Conversion Factor",    "device_kind": "cpu",
     "cpu_parallel": "thread",  "serial": True,  "fanout": False, "depends_on": ()},
    {"key": "archival_detector",     "name": "Archival Detector",       "device_kind": "cuda",
     "cpu_parallel": "thread",  "serial": False, "fanout": False, "depends_on": ()},
    {"key": "plant_detector",        "name": "Plant Detector",          "device_kind": "cuda",
     "cpu_parallel": "thread",  "serial": False, "fanout": False, "depends_on": ()},
    {"key": "specimen_segmenter",    "name": "Specimen Segmenter",      "device_kind": "cuda",
     "cpu_parallel": "thread",  "serial": False, "fanout": False, "depends_on": ()},
    {"key": "phenology_detector",    "name": "Phenology Detector",      "device_kind": "cpu",
     "cpu_parallel": "thread",  "serial": False, "fanout": False, "depends_on": ("plant_detector",)},
    {"key": "ruler_classifier",      "name": "Ruler Classifier",        "device_kind": "cuda",
     "cpu_parallel": "thread",  "serial": False, "fanout": False, "depends_on": ("archival_detector",)},
    {"key": "ruler_cf",              "name": "Ruler Conversion Factor", "device_kind": "cpu",
     "cpu_parallel": "process", "serial": False, "fanout": False,
     "depends_on": ("archival_detector", "ruler_classifier")},
    {"key": "leaf_segmenter",        "name": "Leaf Segmenter",          "device_kind": "cuda",
     "cpu_parallel": "thread",  "serial": False, "fanout": False, "depends_on": ("plant_detector",)},
    {"key": "morphology",            "name": "Morphology",              "device_kind": "cpu",
     "cpu_parallel": "thread",  "serial": False, "fanout": False, "depends_on": ("leaf_segmenter",)},
    {"key": "landmark_detector",     "name": "Landmark Detector",       "device_kind": "cuda",
     "cpu_parallel": "thread",  "serial": False, "fanout": False, "depends_on": ("plant_detector",)},
    {"key": "landmark_measurements", "name": "Landmark Measurements",   "device_kind": "cpu",
     "cpu_parallel": "thread",  "serial": False, "fanout": False, "depends_on": ("landmark_detector",)},
    {"key": "leaf_orientation",      "name": "Leaf Orientation",        "device_kind": "cpu",
     "cpu_parallel": "thread",  "serial": False, "fanout": False,
     "depends_on": ("morphology", "landmark_detector")},
    {"key": "petiole_width",         "name": "Petiole Width",           "device_kind": "cpu",
     "cpu_parallel": "thread",  "serial": False, "fanout": False,
     "depends_on": ("leaf_segmenter", "landmark_detector")},
    {"key": "bilateral_symmetry",    "name": "Bilateral Symmetry",      "device_kind": "cpu",
     "cpu_parallel": "process", "serial": False, "fanout": True,
     "depends_on": ("leaf_segmenter", "landmark_detector", "morphology", "leaf_orientation")},
    {"key": "metric_grounding",      "name": "Metric Grounding",        "device_kind": "cpu",
     "cpu_parallel": "thread",  "serial": False, "fanout": False,
     "depends_on": ("ruler_cf", "leaf_segmenter", "petiole_width")},
    {"key": "reporter",              "name": "Reporter",                "device_kind": "cpu",
     "cpu_parallel": "thread",  "serial": False, "fanout": False,
     "depends_on": ("archival_detector", "plant_detector", "specimen_segmenter",
                    "phenology_detector", "ruler_classifier", "ruler_cf", "leaf_segmenter",
                    "morphology", "landmark_detector", "landmark_measurements",
                    "leaf_orientation", "petiole_width", "metric_grounding")},
    {"key": "ect",                   "name": "ECT",                     "device_kind": "cpu",
     "cpu_parallel": "process", "serial": False, "fanout": True,  "depends_on": ("reporter",)},
)

# Coarse per-image cost prior, seconds, used ONLY to weight a module that has never run in
# this project (so it has no observed duration to weight by) and to extend the run ETA past
# the current module. Measured on this workstation from a 19-image reference run; the numbers
# are calibrated per-run against whatever HAS been observed (see ``_calibration``), and any
# module that has actually run uses its own measured duration instead. Relative magnitude is
# what matters here, not absolute accuracy.
_PRIOR_S_PER_IMAGE: dict[str, float] = {
    "mp_conversion_factor": 0.01, "archival_detector": 0.32, "plant_detector": 0.32,
    "specimen_segmenter": 0.37, "phenology_detector": 0.02, "ruler_classifier": 0.37,
    "ruler_cf": 0.80, "leaf_segmenter": 0.32, "morphology": 0.03,
    "landmark_detector": 0.53, "landmark_measurements": 0.03, "leaf_orientation": 0.02,
    "petiole_width": 0.06, "metric_grounding": 0.01, "reporter": 0.68, "ect": 0.31,
}

_MODULES_CACHE: Optional[tuple[dict, ...]] = None
_MODULES_FROM_LIVE = False
_MODULES_LOCK = threading.Lock()


def module_table() -> tuple[dict, ...]:
    """The 17 canonical modules in STAGE_ORDER, each with the facts the UI needs.

    Read off the live ``PipelineStage`` subclasses when ``leafmachine3.pipeline`` is ALREADY
    in ``sys.modules`` (so a new or edited stage is picked up for free during and after a run),
    and off :data:`_STATIC_MODULES` otherwise.

    That condition is not fastidiousness -- it is the whole point. Importing the pipeline pulls
    in torch, OpenCV, ultralytics and matplotlib, and their OpenBLAS/OpenMP runtimes start ~126
    native threads on a 64-core box which then IDLE-SPIN at roughly 40% of a core, forever.
    Measured here. Doing that from a status endpoint would mean merely opening the app's
    primary tab permanently burns a core and hundreds of MB, on a server that may never run a
    job. The static table carries the same seven facts, and the cache re-checks on every call
    so the first status frame after a run starts upgrades itself to the live classes.
    """
    global _MODULES_CACHE, _MODULES_FROM_LIVE
    live_available = "leafmachine3.pipeline" in sys.modules
    with _MODULES_LOCK:
        if _MODULES_CACHE is not None and (_MODULES_FROM_LIVE or not live_available):
            return _MODULES_CACHE
        table, from_live = _STATIC_MODULES, False
        try:
            if not live_available:
                raise ImportError("pipeline not imported in this process")
            from leafmachine3.core.stage import PipelineStage
            from leafmachine3.pipeline import STAGE_ORDER

            live: list[dict] = []
            for cls in STAGE_ORDER:
                # A CPU stage that overrides build_model warm-loads something, and the executor
                # then refuses to thread it (see _run_inprocess: n_threads stays 1 unless the
                # model is None). That is why mp_conversion_factor runs serially despite a
                # 62-worker plan -- detect it from the class instead of hard-coding a list.
                warm_loads = cls.build_model is not PipelineStage.build_model
                live.append({
                    "key": cls.key,
                    "name": cls.name or cls.key,
                    "device_kind": getattr(cls, "device_kind", "cpu"),
                    "cpu_parallel": getattr(cls, "cpu_parallel", "thread"),
                    "serial": bool(warm_loads and getattr(cls, "device_kind", "cpu") == "cpu"
                                   and getattr(cls, "cpu_parallel", "thread") == "thread"),
                    "fanout": bool(getattr(cls, "fanout", False)),
                    "depends_on": tuple(getattr(cls, "depends_on", ())),
                })
            if live:
                table, from_live = tuple(live), True
        except Exception:  # noqa: BLE001 - a base install may not import every module
            log.debug("pipeline classes unavailable; using the static module table", exc_info=True)
        _MODULES_CACHE = tuple(
            {**m, "order": i + 1, "device": "gpu" if m["device_kind"] == "cuda" else "cpu"}
            for i, m in enumerate(table)
        )
        _MODULES_FROM_LIVE = from_live
        return _MODULES_CACHE


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _parse_db_ts(value: Any) -> Optional[float]:
    """Parse a ledger timestamp to unix seconds.

    Every timestamp LM3 writes comes from SQLite ``datetime('now')``, which is UTC with no
    zone suffix. Parsing it as local time is the classic four-hour bug, so pin UTC here.
    """
    if not value:
        return None
    text = str(value).strip().replace("T", " ").rstrip("Z")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def _iso(ts: Optional[float]) -> Optional[str]:
    """Unix seconds -> ``2026-07-31T20:03:08Z``; unambiguous for ``new Date()`` in the UI."""
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _r(value: Any, ndigits: int = 1) -> Optional[float]:
    """Round for the wire, preserving ``None`` (null means "not knowable", never zero)."""
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _pct(done: float, total: float) -> float:
    if not total or total <= 0:
        return 0.0
    return round(max(0.0, min(100.0, 100.0 * float(done) / float(total))), 1)


class _TimedCache:
    """Memoize a read with a TTL that scales with how expensive the read turned out to be.

    The status endpoints run at 2 Hz forever, and the ledger queries are counts over
    ``image_status`` -- microseconds on a 20-image project and tens of milliseconds on a
    100k-image one. Rather than pick one TTL that is either wastefully slow or ruinously
    fast, each entry's TTL is ``cost x factor`` clamped to [min, max]: a cheap project stays
    live at 2 Hz, a huge one automatically backs off to a few reads per second and the server
    never spends more than ~1/factor of a core on the ledger.
    """

    def __init__(self, min_ttl: float, max_ttl: float, factor: float = 12.0) -> None:
        self.min_ttl, self.max_ttl, self.factor = min_ttl, max_ttl, factor
        self._entries: dict[Any, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: Any, produce: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None and now < hit[1]:
                return hit[0]
        started = time.monotonic()
        value = produce()
        cost = time.monotonic() - started
        ttl = min(self.max_ttl, max(self.min_ttl, cost * self.factor))
        with self._lock:
            self._entries[key] = (value, time.monotonic() + ttl)
            if len(self._entries) > 64:                       # bounded; drop the stalest half
                for stale in sorted(self._entries, key=lambda k: self._entries[k][1])[:32]:
                    self._entries.pop(stale, None)
        return value

    def invalidate(self) -> None:
        with self._lock:
            self._entries.clear()


_SNAPSHOT_CACHE = _TimedCache(min_ttl=0.30, max_ttl=5.0)
_RUNS_CACHE = _TimedCache(min_ttl=2.0, max_ttl=20.0)
_SCAN_CACHE = _TimedCache(min_ttl=3.0, max_ttl=30.0)
_YAML_CACHE: dict[str, tuple[float, dict]] = {}
_YAML_LOCK = threading.Lock()


def _read_yaml(path: Path) -> dict:
    """Load a YAML file, re-reading only when its mtime changes (this runs at 2 Hz)."""
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = str(path)
    with _YAML_LOCK:
        cached = _YAML_CACHE.get(key)
        if cached is not None and cached[0] == stat.st_mtime:
            return cached[1]
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:  # noqa: BLE001 - a half-written YAML must not break the status endpoint
        data = {}
    if not isinstance(data, dict):
        data = {}
    with _YAML_LOCK:
        _YAML_CACHE[key] = (stat.st_mtime, data)
    return data


def _settings_path() -> Path:
    """The LM3_settings.yaml the server is driving (``LM3_SETTINGS`` overrides the cwd copy)."""
    return Path(os.environ.get("LM3_SETTINGS", "LM3_settings.yaml"))


def _hardware() -> dict:
    """``hardware_settings.yaml`` as a plain dict.

    Same file ``app.read_hardware_settings`` serves, read locally so the status path never
    imports the app module (that import would be circular once app.py mounts this router) and
    so it can be mtime-cached -- this is read on every status frame.
    """
    return _read_yaml(Path(os.environ.get("LM3_HARDWARE", "hardware_settings.yaml")))


def _dig(node: Any, *keys: str, default: Any = None) -> Any:
    """Walk a nested mapping tolerantly; any missing link yields ``default``."""
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


# --------------------------------------------------------------------------- #
# Which run are we looking at?
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunRef:
    """A resolved run directory and the artifacts inside it."""

    run_name: str
    root: Path                 # <output.dir>/<run_name>
    db_path: Path              # <root>/<run_name>.sqlite
    log_path: Path             # <root>/logs/lm3.log
    job_id: Optional[str] = None
    source: str = "discovered"  # bound | job | query | settings | discovered

    def as_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "run_path": str(self.root),
            "db_path": str(self.db_path),
            "log_path": str(self.log_path),
            "job_id": self.job_id,
            "source": self.source,
        }


def _run_ref(root: Path, *, job_id: Optional[str] = None, source: str = "discovered") -> RunRef:
    root = Path(root)
    return RunRef(
        run_name=root.name,
        root=root,
        db_path=root / f"{root.name}.sqlite",
        log_path=root / "logs" / "lm3.log",
        job_id=job_id,
        source=source,
    )


# The integrator's hooks. app.py knows things this module cannot discover -- which job the
# JobManager is running right now, and its queued/running/done/error state -- so it can push
# them here instead of this module guessing from the filesystem.
_BOUND: Optional[RunRef] = None
_BOUND_STATE: Optional[str] = None
_JOB_SOURCE: Optional[Callable[[], Optional[dict]]] = None
_BIND_LOCK = threading.Lock()


def bind_run(db_path: Any = None, *, run_name: Optional[str] = None, root: Any = None,
             job_id: Optional[str] = None, state: Optional[str] = None) -> dict:
    """Point the status API at one specific run and return the resolved reference.

    Call it from ``app.py``'s job worker::

        progress_api.bind_run(job.db_path, job_id=job.id, state="running")

    ``db_path`` may be the sqlite file or the run directory; either identifies the run. Pass
    ``state`` to have the run-level state reported from the JobManager (which knows about
    ``queued`` and about a crash) rather than inferred from the ledger.
    """
    global _BOUND, _BOUND_STATE
    if root is not None:
        run_root = Path(root)
    elif db_path is not None:
        candidate = Path(db_path)
        run_root = candidate.parent if candidate.suffix == ".sqlite" else candidate
    else:
        raise ValueError("bind_run needs a db_path or a root")
    ref = _run_ref(run_root, job_id=job_id, source="bound")
    if run_name:
        ref = RunRef(run_name, ref.root, ref.db_path, ref.log_path, job_id, "bound")
    with _BIND_LOCK:
        _BOUND, _BOUND_STATE = ref, state
    _SNAPSHOT_CACHE.invalidate()
    return ref.as_dict()


def clear_run() -> None:
    """Forget the bound run and go back to discovering the newest one on disk."""
    global _BOUND, _BOUND_STATE
    with _BIND_LOCK:
        _BOUND, _BOUND_STATE = None, None
    _SNAPSHOT_CACHE.invalidate()


def bind_job_source(fn: Optional[Callable[[], Optional[dict]]]) -> None:
    """Register a callable returning the live job as ``{db_path|root, job_id, state}``.

    Preferred over :func:`bind_run` when the app runs jobs back to back: it is consulted on
    every request, so the status tab follows the JobManager without any further wiring::

        progress_api.bind_job_source(lambda: _current_job_dict())
    """
    global _JOB_SOURCE
    _JOB_SOURCE = fn
    _SNAPSHOT_CACHE.invalidate()


def _from_job_source() -> tuple[Optional[RunRef], Optional[str]]:
    if _JOB_SOURCE is None:
        return None, None
    try:
        info = _JOB_SOURCE()
    except Exception:  # noqa: BLE001 - never let a host hook break the status endpoint
        log.debug("job source raised", exc_info=True)
        return None, None
    if not info:
        return None, None
    raw = info.get("db_path") or info.get("root")
    if not raw:
        return None, None
    candidate = Path(str(raw))
    root = candidate.parent if candidate.suffix == ".sqlite" else candidate
    ref = _run_ref(root, job_id=info.get("job_id"), source="job")
    return ref, info.get("state")


def _search_roots() -> list[Path]:
    """Directories that may contain run dirs, most specific first.

    ``project.output.dir`` from the settings file is the one users configure; the server's
    staged-job root holds runs submitted through ``POST /v1/jobs``; ``LM3_STATUS_ROOTS``
    (os.pathsep separated) lets an operator add more without editing anything.
    """
    roots: list[Path] = []

    def add(value: Any) -> None:
        if not value:
            return
        path = Path(str(value)).expanduser()
        if path not in roots:
            roots.append(path)

    add(_dig(_read_yaml(_settings_path()), "project", "output", "dir"))
    add(os.environ.get("LM3_SERVER_JOBS", "runs/_server_jobs"))
    for extra in (os.environ.get("LM3_STATUS_ROOTS") or "").split(os.pathsep):
        add(extra.strip())
    return roots


def _is_run_dir(path: Path) -> bool:
    """A run dir is the one place ``<name>/<name>.sqlite`` exists (see ``core.dirs``).

    Server jobs satisfy it too: ``JobManager`` names the run "run", so the ledger is
    ``<job>/run/run.sqlite``.
    """
    try:
        return path.is_dir() and (path / f"{path.name}.sqlite").is_file()
    except OSError:
        return False


def _scan_runs() -> list[Path]:
    """Every run dir under the search roots, scanned two levels deep and TTL-cached.

    Depth 2 is what reaches a server job's ledger (``<jobs_root>/<job_id>/run``) while still
    being a bounded number of ``stat`` calls.
    """
    def produce() -> list[Path]:
        found: list[Path] = []
        seen: set[Path] = set()
        for root in _search_roots():
            if not root.is_dir():
                continue
            if _is_run_dir(root) and root not in seen:
                seen.add(root)
                found.append(root)
            try:
                children = sorted(root.iterdir())[:400]        # bounded: never walk a huge tree
            except OSError:
                continue
            for child in children:
                if not child.is_dir():
                    continue
                if _is_run_dir(child) and child not in seen:
                    seen.add(child)
                    found.append(child)
                    continue
                try:
                    grandchildren = sorted(child.iterdir())[:64]
                except OSError:
                    continue
                for grandchild in grandchildren:
                    if _is_run_dir(grandchild) and grandchild not in seen:
                        seen.add(grandchild)
                        found.append(grandchild)
        return found

    return _SCAN_CACHE.get("scan", produce)


def resolve_run(run: Optional[str] = None, db: Optional[str] = None) -> Optional[RunRef]:
    """Decide which run the status endpoints describe.

    Precedence: an explicit ``db=`` path, then an explicit ``run=`` name, then the live job
    handed over by :func:`bind_job_source`, then whatever :func:`bind_run` pinned, then the
    project the SETTINGS name (:func:`_from_settings`), and only failing all of those,
    discovery -- which prefers a genuinely live run and otherwise the most recently written one.

    The settings step is the important one for a GUI that is sitting idle: <output folder>/<project
    name> is what the app is pointed at, so the status stream, the console and the stage bar all
    describe that project instead of whatever scanning the disk happens to turn up.
    """
    if db:
        candidate = Path(db).expanduser()
        root = candidate.parent if candidate.suffix == ".sqlite" else candidate
        return _run_ref(root, source="query")
    if run:
        for path in _scan_runs():
            if path.name == run:
                return _run_ref(path, source="query")
        for root in _search_roots():                          # not scanned yet but may exist
            if _is_run_dir(root / run):
                return _run_ref(root / run, source="query")
        return None

    ref, _ = _from_job_source()
    if ref is not None and (ref.db_path.exists() or ref.root.exists()):
        return ref
    with _BIND_LOCK:
        bound = _BOUND
    if bound is not None and (bound.db_path.exists() or bound.root.exists()):
        return bound
    configured = _from_settings()
    if configured is not None:
        return configured
    return _SCAN_CACHE.get("discover", _discover_run)


def _from_settings() -> Optional[RunRef]:
    """The run the SETTINGS name: ``<project.output.dir>/<project.run_name>``.

    That path IS the project -- it is what the main settings strip edits, what ``start_run``
    launches, and what ``core.dirs`` lays out -- so deriving the ledger from it keeps every live
    view describing the project the app is set up for.

    Returned even when the directory does not exist yet. "The project you are pointed at has not
    run" is the honest answer for a freshly typed name, and a far better one than letting
    discovery wander off and describe an unrelated run as though it were yours. Discovery still
    gets its turn when the settings name no project at all, which is what keeps a run started
    outside the app (from the machine3 CLI) findable.
    """
    cfg = _read_yaml(_settings_path())
    name = str(_dig(cfg, "project", "run_name", default="") or "").strip()
    out = str(_dig(cfg, "project", "output", "dir", default="") or "").strip()
    if not name or not out:
        return None
    root = Path(out).expanduser()
    if not root.is_absolute():
        # Relative output dirs (the default is just "runs") hang off the settings file itself,
        # which is also how metrics_api.start_run resolves them.
        try:
            root = _settings_path().resolve().parent / root
        except OSError:
            return None
    return _run_ref(root / name, source="settings")


def _last_write(root: Path, db_path: Path) -> float:
    """Newest write to a run's OWN files -- the "is this thing alive" clock.

    Deliberately not the whole ``db*`` set: SQLite touches ``-shm`` whenever anyone merely OPENS
    the database, including the read-only handles this module opens twice a second, so a dead
    run's ``-shm`` is permanently a few seconds old. The WAL is what a live writer actually
    advances (in WAL mode the main file's mtime can sit still for a long stretch), and the log
    covers a stage that is busy but not yet committing rows.
    """
    newest = 0.0
    for candidate in (db_path, db_path.with_name(db_path.name + "-wal"), root / "logs" / "lm3.log"):
        try:
            newest = max(newest, candidate.stat().st_mtime)
        except OSError:
            continue
    return newest


def _discover_run() -> Optional[RunRef]:
    """Pick the interesting run off the filesystem: a live one, else the newest one.

    Opens one read-only handle per candidate, which is why the result is TTL-cached -- the
    status stream asks for it twice a second.
    """
    now = time.time()
    newest_running: Optional[tuple[float, Path]] = None
    newest_any: Optional[tuple[float, Path]] = None
    for path in _scan_runs():
        db_path = path / f"{path.name}.sqlite"
        try:
            mtime = db_path.stat().st_mtime
        except OSError:
            continue
        if newest_any is None or mtime > newest_any[0]:
            newest_any = (mtime, path)
        rows = _stage_rows(db_path)
        if any(r.get("state") == "running" for r in rows):
            # A ``running`` row only means "live" if something wrote recently. LM3 repairs a
            # crashed run's row on that PROJECT's next start (ProjectDB.reclaim_running), so a
            # killed run advertises ``running`` forever -- and preferring that unconditionally
            # lets ONE abandoned run outrank every real run from then on, which is exactly how
            # the status stream and console end up pinned to a ledger nobody is writing.
            #
            # Demoting it loses nothing: it still competes as newest_any, so the genuinely most
            # recent run wins on recency instead.
            if now - _last_write(path, db_path) > STALE_AFTER_S:
                continue
            started = max((_parse_db_ts(r.get("started_at")) or 0.0) for r in rows)
            if newest_running is None or started > newest_running[0]:
                newest_running = (started, path)
    chosen = newest_running or newest_any
    return _run_ref(chosen[1], source="discovered") if chosen else None


def active_run() -> Optional[dict]:
    """The currently resolved run as a plain dict (``None`` when no run exists yet).

    Other server modules (results / postprocess) can call this to stay on the same run the
    Status tab is showing, instead of re-deriving it.
    """
    ref = resolve_run()
    return ref.as_dict() if ref else None


# --------------------------------------------------------------------------- #
# Read-only ledger access
# --------------------------------------------------------------------------- #
def _connect_ro(db_path: Path) -> Optional[sqlite3.Connection]:
    """Open the project DB READ-ONLY. Never blocks a live LM3 run: the ledger is WAL, and an
    ``mode=ro`` handle takes no write lock, so a 2 Hz poll is invisible to the collector."""
    if not Path(db_path).is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def _stage_rows(db_path: Path) -> list[dict]:
    """``project_status`` as dicts in stage order (empty when the DB is absent/half-built)."""
    conn = _connect_ro(db_path)
    if conn is None:
        return []
    try:
        return [dict(r) for r in conn.execute(
            "SELECT stage_key, stage_order, state, n_total, n_done, settings_hash, "
            "started_at, finished_at, error_msg FROM project_status ORDER BY stage_order")]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _read_ledger(db_path: Path) -> dict:
    """Every ledger number one snapshot needs, in as few scans as SQLite allows.

    The per-(stage, state) counts come out of ``ix_imgstatus (stage_key, state)`` as an
    index-only scan; the ``no_work`` and session-completion tallies need the rows themselves,
    so they are folded into one extra pass rather than a query per module.
    """
    conn = _connect_ro(db_path)
    if conn is None:
        return {"ready": False}
    out: dict[str, Any] = {"ready": True, "stages": [], "images_total": 0,
                           "counts": {}, "no_work": {}, "since": {}, "last_update": None}
    try:
        out["stages"] = [dict(r) for r in conn.execute(
            "SELECT stage_key, stage_order, state, n_total, n_done, settings_hash, "
            "started_at, finished_at, error_msg FROM project_status ORDER BY stage_order")]
        row = conn.execute("SELECT COUNT(*) FROM specimen").fetchone()
        out["images_total"] = int(row[0]) if row and row[0] is not None else 0

        # Every tally below is aggregated by SQLite, never by iterating rows in Python: on a
        # 100k-image project image_status holds 1.6M rows, and pulling those across the driver
        # twice a second would cost more than the run.
        counts: dict[str, dict[str, int]] = {}
        for stage_key, state, n in conn.execute(
                "SELECT stage_key, state, COUNT(*) FROM image_status GROUP BY stage_key, state"):
            counts.setdefault(str(stage_key), {})[str(state)] = int(n)
        out["counts"] = counts

        out["no_work"] = {
            str(k): int(n) for k, n in conn.execute(
                "SELECT stage_key, COUNT(*) FROM image_status WHERE no_work = 1 GROUP BY stage_key")
        }
        # Checkpointed during the stage's CURRENT attempt. This is what makes a rate honest on
        # a resumed stage, where n_done is pre-seeded with the previous session's work and
        # would otherwise report a throughput of thousands of images per second in the first
        # frame. Kept split by state: recovering the eligible set has to net off exactly what
        # mark_stage_running seeded n_done with, and that is DONE rows only (done_ids ignores
        # errors), whereas the throughput rate should count an errored image as work performed.
        # Timestamps are 'YYYY-MM-DD HH:MM:SS' UTC on both sides, so a string compare is a
        # chronological compare.
        since: dict[str, dict[str, int]] = {}
        for stage_key, state, n in conn.execute(
                """
                SELECT i.stage_key, i.state, COUNT(*)
                  FROM image_status i
                  JOIN project_status p ON p.stage_key = i.stage_key
                 WHERE p.started_at IS NOT NULL AND i.updated_at >= p.started_at
                 GROUP BY i.stage_key, i.state
                """):
            since.setdefault(str(stage_key), {})[str(state)] = int(n)
        out["since"] = since

        row = conn.execute("SELECT MAX(updated_at) FROM image_status").fetchone()
        out["last_update"] = row[0] if row else None
    except sqlite3.Error as exc:
        log.debug("ledger read failed for %s: %s", db_path, exc)
        out["ready"] = bool(out["stages"])
    finally:
        conn.close()
    return out


def _recent_rows(db_path: Path, stage_key: Optional[str], limit: int) -> list[dict]:
    """The newest per-image checkpoints, for the live ticker.

    Scoped to one stage so ``ix_imgstatus`` does the work; ``updated_at`` has only second
    resolution, so ties are broken by insertion order (rowid).
    """
    if limit <= 0:
        return []
    conn = _connect_ro(db_path)
    if conn is None:
        return []
    try:
        params: tuple = ()
        where = ""
        if stage_key:
            where, params = "WHERE i.stage_key = ?", (stage_key,)
        rows = conn.execute(
            f"""
            SELECT i.specimen_id, i.stage_key, i.state, i.no_work, i.updated_at, i.error_msg,
                   s.image_name
              FROM image_status i
              LEFT JOIN specimen s ON s.specimen_id = i.specimen_id
              {where}
             ORDER BY i.updated_at DESC, i.rowid DESC
             LIMIT ?
            """,
            (*params, int(limit)),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [
        {
            "specimen_id": int(r["specimen_id"]),
            "image_name": r["image_name"],
            "stage_key": r["stage_key"],
            "state": r["state"],
            "no_work": bool(r["no_work"]),
            "updated_at": _iso(_parse_db_ts(r["updated_at"])),
            "error_msg": r["error_msg"],
        }
        for r in rows
    ]


# --------------------------------------------------------------------------- #
# Execution plan: how a module runs, and with how many workers
# --------------------------------------------------------------------------- #
def _planned_workers(meta: dict, hardware: dict, n_gpus: int) -> int:
    """The worker count LM3 would use for this module, from ``hardware_settings.yaml``.

    Mirrors ``core.executor.DeviceManager.plan``: GPU stages get ``workers_per_gpu`` slots on
    every bound GPU, CPU stages get the stage's tuned ``workers`` and fall back to the
    profile's ``io_workers``.
    """
    stage_cfg = _dig(hardware, "stages", meta["key"], default={}) or {}
    if meta["device_kind"] == "cuda":
        per_gpu = int(stage_cfg.get("workers_per_gpu") or 1)
        return max(1, per_gpu * max(1, n_gpus))
    workers = stage_cfg.get("workers")
    if workers:
        return max(1, int(workers))
    io_workers = hardware.get("io_workers")
    if io_workers:
        return max(1, int(io_workers))
    return max(1, (os.cpu_count() or 2) - 2)


def _plan_exec(meta: dict, hardware: dict, n_gpus: int, n_items: int) -> dict:
    """Reproduce the executor's own mode selection for one module.

    Straight out of ``StageExecutor._use_inprocess`` / ``_run_inprocess`` / ``_run_pool``:

      * cuda stage with GPUs bound            -> ``gpu``     process pool, workers_per_gpu x GPUs
      * cuda stage with none                  -> ``serial``  (it warm-loads a model in-process)
      * cpu + cpu_parallel="process"          -> ``process`` when the batch clears
                                                 ``min_pool_items`` and >1 worker is planned,
                                                 else ``serial``
      * cpu + thread, warm-loads a model      -> ``serial``  (threads give it nothing)
      * cpu + thread                          -> ``thread``  in-process pool
    The pool is never wider than the batch (``n = min(len(devices), len(todo))``).
    """
    stage_cfg = _dig(hardware, "stages", meta["key"], default={}) or {}
    planned = _planned_workers(meta, hardware, n_gpus)
    cap = max(1, n_items) if n_items > 0 else planned

    if meta["device_kind"] == "cuda":
        if n_gpus <= 0:
            return {"exec_mode": "serial", "workers": 1, "planned_workers": planned,
                    "reason": "no GPU is bound, so the stage warm-loads its model in-process"}
        return {"exec_mode": "gpu", "workers": min(planned, cap), "planned_workers": planned,
                "reason": None}

    if meta["cpu_parallel"] == "process":
        floor = int(stage_cfg.get("min_pool_items")
                    or hardware.get("process_pool_min_items") or 8)
        if planned > 1 and n_items >= floor:
            return {"exec_mode": "process", "workers": min(planned, cap),
                    "planned_workers": planned, "reason": None}
        return {"exec_mode": "serial", "workers": 1, "planned_workers": planned,
                "reason": (f"batch of {n_items} is under the {floor}-item threshold, so the spawn "
                           "pool would cost more than it saves")}

    if meta.get("serial"):
        return {"exec_mode": "serial", "workers": 1, "planned_workers": planned,
                "reason": "warm-loads a model and its per-item work is light, so it runs serially"}
    if planned > 1:
        return {"exec_mode": "thread", "workers": min(planned, cap), "planned_workers": planned,
                "reason": None}
    return {"exec_mode": "serial", "workers": 1, "planned_workers": planned,
            "reason": "only one CPU worker was allocated"}


def _bound_gpu_count(settings: dict, hardware: dict) -> int:
    """How many GPUs this run binds: ``compute.devices`` when explicit, else the profile's."""
    devices = _dig(settings, "compute", "devices", default="auto")
    if isinstance(devices, (list, tuple)):
        return len(devices)
    if isinstance(devices, str) and devices.strip().lower() == "cpu":
        return 0
    if _dig(settings, "compute", "mock", default=False):
        return 0
    gpus = hardware.get("gpus") or []
    return len(gpus) if isinstance(gpus, (list, tuple)) else 0


# --------------------------------------------------------------------------- #
# The snapshot
# --------------------------------------------------------------------------- #
def _module_totals(stage: dict, meta: dict, ledger: dict, images_total: int) -> dict:
    """Recover a module's TRUE denominator and completed count from the ledger.

    ``project_status`` cannot be read literally. ``mark_stage_running`` sets
    ``n_total = len(pending)`` -- the work OUTSTANDING at stage start, not the image count --
    and seeds ``n_done`` with what was already finished, then bumps it per image. On a resumed
    run ``n_done`` therefore starts above zero and can exceed ``n_total``.

    ``image_status`` is exact, so:

        done        rows in state 'done'  (no_work rows included: a sheet with no ruler
                                           legitimately retires the ruler modules)
        errors      rows in state 'error'
        since_done  'done' rows checkpointed during the CURRENT attempt
        eligible    n_total + done - since_done   ==   n_total + n_done_at_start

    which is the eligible set the executor saw, capped by the image count (no module can have
    more eligible specimens than there are specimens) and floored by done+errors so the bar is
    monotone and always reaches 100%.
    """
    key = meta["key"]
    counts = ledger["counts"].get(key, {})
    done = int(counts.get("done", 0))
    errors = int(counts.get("error", 0))
    since = ledger["since"].get(key, {})
    since_done = int(since.get("done", 0))
    since_any = sum(int(v) for v in since.values())
    n_total_raw = int(stage.get("n_total") or 0)

    if stage.get("started_at"):
        eligible = n_total_raw + done - since_done
        if images_total:
            eligible = min(eligible, images_total)
        total = max(done + errors, eligible, 0)
        source = "ledger"
    else:
        # Never started: the eligible set is unknown until depends_on resolves, so show the
        # image count as a ceiling. The bar is at 0 either way, and flagging the source keeps
        # the UI honest about it.
        total = images_total
        source = "ceiling"

    return {
        "n_done": done,
        "n_error": errors,
        "n_no_work": int(ledger["no_work"].get(key, 0)),
        "n_total": total,
        "n_total_source": source,
        "n_done_this_attempt": since_any,
        "db_n_total": n_total_raw,
        "db_n_done": int(stage.get("n_done") or 0),
    }


def _session_start(modules: list[dict], log_anchor: Optional[float]) -> Optional[float]:
    """When the CURRENT run session began.

    The log's "LM3 start" marker is authoritative when the log is readable. Without it, walk
    the modules backwards from the newest start and keep absorbing the previous one while it
    FINISHED within ``SESSION_GAP_S`` of the next one starting. Measuring from finished_at is
    what makes this safe: modules are strictly sequential, so an eight-hour Leaf Segmenter
    still hands straight over to Morphology and cannot split a session, whereas a genuine gap
    between sessions shows up as a module that started long after the previous one ended.
    """
    started = sorted(
        (m for m in modules if m["_started_ts"] is not None),
        key=lambda m: m["_started_ts"],
    )
    if not started:
        return None
    if log_anchor is not None:
        # Trust the log, but only if the ledger agrees a module started at/after it.
        in_session = [m["_started_ts"] for m in started if m["_started_ts"] >= log_anchor - 5.0]
        if in_session:
            return min(in_session)
    anchor = started[-1]["_started_ts"]
    for previous in reversed(started[:-1]):
        end = previous["_finished_ts"] if previous["_finished_ts"] is not None else previous["_started_ts"]
        if anchor - end > SESSION_GAP_S:
            break
        anchor = previous["_started_ts"]
    return anchor


def _calibration(modules: list[dict], images_total: int) -> float:
    """How much slower/faster this machine+project is than the built-in per-image priors.

    Fitted from whatever HAS been measured in this project. The clamp is wide on purpose: the
    priors describe the SHAPE of a run (the Reporter costs many times what Morphology does),
    not its scale, and the scale legitimately moves by an order of magnitude between a laptop
    and a two-GPU workstation. It only exists to stop one pathological module from running
    away with the ETA. Returns 1.0 until there is something to fit.
    """
    observed = predicted = 0.0
    for mod in modules:
        duration = mod.get("elapsed_s")
        if mod["state"] != "done" or not duration or duration <= 0.2:
            continue
        prior = _PRIOR_S_PER_IMAGE.get(mod["key"], 0.1) * max(1, mod["n_total"] or images_total)
        if prior <= 0:
            continue
        observed += duration
        predicted += prior
    if predicted <= 0 or observed <= 0:
        return 1.0
    return max(0.05, min(20.0, observed / predicted))


def _weights(modules: list[dict], images_total: int, calib: float) -> None:
    """Attach a duration WEIGHT to every module, in place.

    The overall bar weights modules by how long they take, not 1/16 each -- a run is not half
    done when the eighth of sixteen modules starts, because the Reporter alone can outweigh
    five cheap CPU modules. Preference order per module: its own measured duration; for the
    running one, the duration its current rate projects; otherwise the calibrated prior.
    Skipped modules weigh nothing -- they never run.
    """
    for mod in modules:
        if mod["state"] == "skipped" or mod["enabled"] is False:
            mod["weight_s"] = 0.0
            mod["weight_source"] = "skipped"
            continue
        prior = _PRIOR_S_PER_IMAGE.get(mod["key"], 0.1) * max(1, mod["n_total"] or images_total)
        prior *= calib
        elapsed = mod.get("elapsed_s")
        if mod["state"] == "done" and elapsed and elapsed > 0:
            mod["weight_s"], mod["weight_source"] = round(float(elapsed), 2), "measured"
        elif mod["state"] == "running" and elapsed and elapsed > 1.0 and mod["pct"] > 5.0:
            mod["weight_s"] = round(max(float(elapsed), float(elapsed) * 100.0 / mod["pct"]), 2)
            mod["weight_source"] = "projected"
        else:
            mod["weight_s"], mod["weight_source"] = round(max(0.01, prior), 2), "prior"


def _derive_workers(active: Optional[dict], live: dict) -> tuple[list[dict], dict]:
    """Build the per-worker rows for the status tab.

    Three sources, in descending order of how real they are:

      1. ``metrics.workers()`` -- actual spawn children of the server process. GPU modules and
         ``cpu_parallel="process"`` modules run a real process pool, so these rows are pure
         observation: pid, CPU%, RSS, attributed VRAM, which GPU, and how long the process has
         been alive.
      2. the executor's own scheduling rule for how many of those workers hold an item. Both
         pools are fed with every outstanding item up front (the thread pool submits them all;
         the process feeder blocks on a bounded queue), so ``busy = min(workers, outstanding)``
         is a property of the code rather than a guess.
      3. ``hardware_settings.yaml`` for thread-pooled modules. Eleven of the sixteen modules
         run THREADS inside the server process -- there are no child processes to see, and
         ``metrics.workers()`` correctly returns []. Those rows are marked ``observed: false``.

    ``item_label`` is ALWAYS null. Nothing in LM3 records which worker took which specimen, so
    there is nothing to report; ``pct`` carries the active module's overall progress instead,
    which is what the per-worker bars were asked to show.
    """
    processes = list(live.get("workers") or [])
    meta: dict[str, Any] = {
        "module": active["key"] if active else None,
        "exec_mode": active["exec_mode"] if active else None,
        "planned": active["planned_workers"] if active else 0,
        "observed": len(processes),
        "outstanding": 0,
        "n_busy": 0,
        "pooled": False,
        "pct_is": "module_overall",
        "item_label_available": False,
        "root_pid": live.get("root_pid"),
        "host": live.get("host"),
        "basis": "idle",
    }
    if active is None:
        # Nothing running: still surface any lingering child processes so a wedged worker is
        # visible, but claim nothing about what they are doing.
        rows = [_worker_row(f"w{i}", p, None, None, "idle") for i, p in enumerate(processes)]
        meta["basis"] = "no module is running; rows are live child processes only"
        return rows, meta

    pct = active["pct"]
    outstanding = max(0, int(active["n_total"]) - int(active["n_done"]))
    meta["outstanding"] = outstanding
    # Only a POOL module's workers are child processes. A thread or serial module runs inside
    # this very process, so any child processes visible during one are not its workers (a
    # straggler from the previous module's pool, or something else entirely) and attributing
    # them to it would be a fabrication.
    pooled = active["exec_mode"] in ("gpu", "process")
    processes = processes if pooled else []
    meta["pooled"] = pooled
    slots = len(processes) if processes else int(active["workers"] or 1)
    n_busy = min(slots, outstanding) if outstanding else 0
    meta["n_busy"] = n_busy

    rows: list[dict] = []
    if processes:
        meta["basis"] = ("live spawn children of the server process; busy is per-process CPU, "
                         "falling back to min(workers, outstanding)")
        # Stable, position-based ids (metrics sorts by start time) so a bar does not jump rows
        # between frames. CPU% is the primary busy signal; a GPU worker blocked on the device
        # can idle its core, so the outstanding-item rule backstops it.
        for i, proc in enumerate(processes):
            cpu = proc.get("cpu_pct")
            busy = bool((cpu is not None and cpu >= 5.0) or (i < n_busy))
            rows.append(_worker_row(f"w{i}", proc, active, pct, "busy" if busy else "idle"))
    else:
        # No child processes to look at. For a thread-pooled or serial module that is the
        # normal, permanent state -- the work happens on threads inside this very process. For
        # a pool module it means the pool has not spawned yet (or psutil/NVML is unavailable),
        # which is a different statement and must not be reported as the first one.
        if active["exec_mode"] in ("gpu", "process"):
            meta["basis"] = (f"{active['exec_mode']} pool with no worker processes visible yet; "
                             "slots come from hardware_settings.yaml and "
                             "busy = min(workers, outstanding)")
        else:
            meta["basis"] = (f"{active['exec_mode']} module: it runs inside the server process, "
                             "so no child processes exist; slots come from "
                             "hardware_settings.yaml and busy = min(workers, outstanding)")
        for i in range(max(1, slots)):
            rows.append({
                "id": f"w{i}",
                "state": "busy" if i < max(n_busy, 1 if outstanding else 0) else "idle",
                "item_label": None,
                "started_at": active["started_at"],
                "started_ts": active["started_ts"],
                "pct": pct,
                "observed": False,
                "module": active["key"],
                "device": active["device"],
                "exec_mode": active["exec_mode"],
                "pid": None, "gpu_index": None, "cpu_pct": None, "rss_mb": None,
                "vram_mb": None, "age_s": None, "proc_state": None,
            })
    return rows, meta


def _worker_row(wid: str, proc: dict, active: Optional[dict], pct: Optional[float],
                state: str) -> dict:
    started_ts = proc.get("started_at")
    return {
        "id": wid,
        "state": state,
        "item_label": None,                      # never knowable -- see _derive_workers
        "started_at": _iso(started_ts),
        "started_ts": _r(started_ts, 3),
        "pct": pct,
        "observed": True,
        "module": active["key"] if active else None,
        "device": (f"cuda:{proc['gpu_index']}" if proc.get("gpu_index") is not None
                   else (active["device"] if active else "cpu")),
        "exec_mode": active["exec_mode"] if active else None,
        "pid": proc.get("pid"),
        "gpu_index": proc.get("gpu_index"),
        "cpu_pct": proc.get("cpu_pct"),
        "rss_mb": proc.get("rss_mb"),
        "vram_mb": proc.get("vram_mb"),
        "age_s": proc.get("age_s"),
        "proc_state": proc.get("status"),
    }


def _live_workers() -> dict:
    """``metrics.workers()`` with every failure mode flattened to "nothing observed".

    Reuses the sampler that already runs for the perf monitor -- this module never scans
    processes itself.
    """
    try:
        from leafmachine3.server import metrics

        return metrics.workers() or {}
    except Exception:  # noqa: BLE001 - psutil is optional; the status tab still works
        return {}


_ANCHOR_CACHE: dict[str, tuple[tuple[float, int], Optional[float]]] = {}
_ANCHOR_LOCK = threading.Lock()


def _log_anchor(log_path: Path) -> Optional[float]:
    """Unix seconds of the last ``LM3 start`` line -- the authoritative session boundary.

    ``machine3`` logs it once per invocation, before anything else happens, so it beats any
    inference from the ledger. Keyed on (mtime, size) so an unchanged log is never re-read;
    the status snapshot needs this on every frame.
    """
    try:
        stat = log_path.stat()
    except OSError:
        return None
    key, stamp = str(log_path), (stat.st_mtime, stat.st_size)
    with _ANCHOR_LOCK:
        hit = _ANCHOR_CACHE.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    anchor: Optional[float] = None
    try:
        with log_path.open("rb") as fh:
            fh.seek(max(0, stat.st_size - _LOG_BACKFILL_BYTES))
            text = fh.read().decode("utf-8", "replace")
        for raw in reversed(text.split("\n")):
            if " LM3 start " not in raw:
                continue
            entry = parse_log_line(raw, anchor=stat.st_mtime)
            if entry.get("kind") == "run_start":
                anchor = entry.get("ts")
                break
    except OSError:
        anchor = None
    with _ANCHOR_LOCK:
        if len(_ANCHOR_CACHE) > 32:
            _ANCHOR_CACHE.clear()
        _ANCHOR_CACHE[key] = (stamp, anchor)
    return anchor


def status(run: Optional[str] = None, db: Optional[str] = None, *,
           recent: Optional[int] = None) -> dict:
    """The full live snapshot. See the module docstring for what is measured vs derived."""
    ref = resolve_run(run, db)
    if ref is None:
        return _empty_status()
    # The reference is part of the key, not just the DB path: the same ledger reached through
    # ``?run=`` and through discovery produces snapshots that differ in ``source`` / ``job_id``,
    # and keying on the path alone would serve one request's provenance to the other.
    key = (str(ref.db_path), ref.source, ref.job_id, int(recent) if recent is not None else -1)
    return _SNAPSHOT_CACHE.get(key, lambda: _build_status(ref, recent))


def _empty_status() -> dict:
    """The no-run-yet snapshot.

    Every key the populated snapshot carries is present here too, nulled. The front end must
    never have to test for a key's existence -- only for its value -- so the shape is
    invariant whether or not a run has ever been started.
    """
    return {
        "ready": False, "t": _r(time.time(), 3), "run_name": None, "run_path": None,
        "db_path": None, "log_path": None, "job_id": None, "source": None,
        "state": "idle", "stale": False, "stale_for_s": None,
        "started_at": None, "started_ts": None, "finished_at": None,
        "elapsed_s": None, "eta_s": None, "calibration": 1.0,
        "images_total": 0, "images_done": 0,
        "modules": [], "active": None, "next": None, "workers": [],
        "workers_meta": {"module": None, "exec_mode": None, "planned": 0, "observed": 0,
                         "outstanding": 0, "n_busy": 0, "pooled": False,
                         "pct_is": "module_overall", "item_label_available": False,
                         "root_pid": None, "host": None, "basis": "no run found"},
        "totals": {"images": 0, "images_done": 0, "modules_done": 0, "modules_total": 0,
                   "modules_enabled": 0, "modules_skipped": 0, "modules_complete": 0,
                   "overall_pct": 0.0, "session_pct": 0.0},
        "recent": [],
        "provenance": _PROVENANCE,
    }


# Shipped verbatim in every snapshot so a reader of the JSON alone can tell which numbers are
# observations and which are inferences, without going back to this file.
_PROVENANCE = {
    "measured": ["modules[].n_done", "modules[].n_error", "modules[].n_no_work",
                 "modules[].started_at", "modules[].finished_at", "modules[].elapsed_s",
                 "totals.images", "workers[] where observed=true"],
    "derived": ["modules[].n_total (project_status.n_total is the PENDING count at stage "
                "start; the eligible set is recovered as n_total + n_done - n_done_this_attempt, "
                "capped by the image count)",
                "modules[].state=='skipped' (a disabled module is written to the ledger as "
                "done/no_work with started_at == finished_at)",
                "modules[].exec_mode and modules[].workers for thread-pooled modules "
                "(no child processes exist to observe; taken from hardware_settings.yaml and "
                "the executor's own selection rules)",
                "workers[].state where observed=false (the executor submits every outstanding "
                "item up front, so busy = min(workers, outstanding))",
                "totals.overall_pct (modules weighted by measured or projected duration)",
                "eta_s"],
    "unknowable": ["workers[].item_label -- nothing in LM3 records which worker holds which "
                   "specimen, so it is always null"],
}


def _build_status(ref: RunRef, recent: Optional[int]) -> dict:
    now = time.time()
    ledger = _read_ledger(ref.db_path)
    settings = _run_settings(ref)
    hardware = _hardware()
    n_gpus = _bound_gpu_count(settings, hardware)
    images_total = int(ledger.get("images_total") or 0)
    by_key = {str(s["stage_key"]): s for s in ledger.get("stages", [])}

    modules: list[dict] = []
    for meta in module_table():
        stage = by_key.get(meta["key"], {})
        enabled = _module_enabled(settings, meta["key"])
        totals = _module_totals(stage, meta, ledger, images_total) if ledger.get("ready") else {
            "n_done": 0, "n_error": 0, "n_no_work": 0, "n_total": images_total,
            "n_total_source": "ceiling", "n_done_this_attempt": 0,
            "db_n_total": 0, "db_n_done": 0,
        }
        started_ts = _parse_db_ts(stage.get("started_at"))
        finished_ts = _parse_db_ts(stage.get("finished_at"))
        state = str(stage.get("state") or "pending")

        # A DISABLED module is written to the ledger by mark_stage_complete_no_work as done,
        # n_total = n_done = COUNT(specimen), started_at == finished_at, and every image_status
        # row flagged no_work. Rendering that as "done" would let the global bar count modules
        # that never ran, so it becomes its own state. The settings flag is the primary signal
        # (free and exact); the ledger signature is the fallback when the run's settings file
        # is not on disk -- and it insists on the no_work tally, so a genuinely instantaneous
        # module like Metric Grounding is not mistaken for a skipped one.
        skipped_by_ledger = (
            state == "done"
            and started_ts is not None and finished_ts is not None and finished_ts == started_ts
            and images_total > 0
            and totals["n_no_work"] >= images_total
            and totals["n_done"] >= images_total
        )
        if state == "done" and (enabled is False or (enabled is None and skipped_by_ledger)):
            state = "skipped"

        if state == "running" and started_ts is not None:
            elapsed = max(0.0, now - started_ts)
        elif started_ts is not None and finished_ts is not None:
            elapsed = max(0.0, finished_ts - started_ts)
        else:
            elapsed = None

        n_items = totals["db_n_total"] or totals["n_total"]
        plan = _plan_exec(meta, hardware, n_gpus, n_items)
        pct = 100.0 if state in ("done", "skipped") else _pct(totals["n_done"], totals["n_total"])

        rate = None
        if state == "running" and elapsed and elapsed > 0.5 and totals["n_done_this_attempt"] > 0:
            rate = totals["n_done_this_attempt"] / elapsed
        eta = None
        if rate and rate > 0:
            eta = max(0.0, (totals["n_total"] - totals["n_done"]) / rate)

        modules.append({
            "key": meta["key"],
            "name": meta["name"],
            "order": meta["order"],
            "state": state,
            "enabled": enabled,
            "n_total": totals["n_total"],
            "n_done": totals["n_done"],
            "n_error": totals["n_error"],
            "n_no_work": totals["n_no_work"],
            "n_done_this_attempt": totals["n_done_this_attempt"],
            "n_total_source": totals["n_total_source"],
            "db_n_total": totals["db_n_total"],
            "db_n_done": totals["db_n_done"],
            "pct": pct,
            "started_at": _iso(started_ts),
            "started_ts": _r(started_ts, 3),
            "finished_at": _iso(finished_ts),
            "finished_ts": _r(finished_ts, 3),
            "elapsed_s": _r(elapsed, 2),
            "rate_per_s": _r(rate, 3),
            "eta_s": _r(eta, 1),
            "exec_mode": plan["exec_mode"],
            "workers": plan["workers"],
            "planned_workers": plan["planned_workers"],
            "exec_note": plan["reason"],
            "device": meta["device"],
            "fanout": meta["fanout"],
            "depends_on": list(meta["depends_on"]),
            "error_msg": stage.get("error_msg"),
            "_started_ts": started_ts,
            "_finished_ts": finished_ts,
        })

    session_ts = _session_start(modules, _log_anchor(ref.log_path))
    for mod in modules:
        mod["session"] = (
            "current" if mod["_started_ts"] is not None and session_ts is not None
            and mod["_started_ts"] >= session_ts - 1.0
            else ("prior" if mod["_started_ts"] is not None else "none")
        )
        mod.pop("_started_ts", None)
        mod.pop("_finished_ts", None)

    calib = _calibration(modules, images_total)
    _weights(modules, images_total, calib)

    active = next((m for m in modules if m["state"] == "running"), None)
    # A module the settings switch off never runs, so it is neither "up next" nor part of the
    # remaining cost -- even before ``run_pipeline`` has had a chance to write its skip marker
    # into the ledger (which is what turns it into state 'skipped').
    upcoming = [m for m in modules if m["state"] == "pending" and m["enabled"] is not False]
    errored = [m for m in modules if m["state"] == "error"]

    last_activity = _parse_db_ts(ledger.get("last_update")) or _log_mtime(ref.log_path)
    run_state = _run_state(active, errored, modules, ledger, last_activity, now)
    stale_for = (now - last_activity) if (run_state == "running" and last_activity) else None
    stale = bool(stale_for is not None and stale_for > STALE_AFTER_S)

    finished_ts = None
    if run_state in ("done", "error"):
        # Prefer this session's own last finish. A run where every module was already up to
        # date finishes without writing one, so fall back to the project's newest.
        session_ends = [m["finished_ts"] for m in modules
                        if m["finished_ts"] is not None and m["session"] == "current"]
        all_ends = [m["finished_ts"] for m in modules if m["finished_ts"] is not None]
        finished_ts = max(session_ends or all_ends) if (session_ends or all_ends) else None
    elapsed_s = None
    if session_ts is not None:
        elapsed_s = max(0.0, (finished_ts or now) - session_ts)

    total_weight = sum(m["weight_s"] for m in modules)
    if total_weight > 0:
        overall_pct = round(sum(m["weight_s"] * m["pct"] for m in modules) / total_weight, 1)
    else:
        # Every module weighs nothing -- they are all disabled or skipped. There is no work to
        # be part-way through, so the run is either complete or has not been touched.
        overall_pct = 100.0 if any(m["state"] in ("done", "skipped") for m in modules) else 0.0
    session_mods = [m for m in modules if m["session"] == "current" or m["state"] in
                    ("running", "pending")]
    session_weight = sum(m["weight_s"] for m in session_mods) or 1.0
    session_pct = round(sum(m["weight_s"] * m["pct"] for m in session_mods) / session_weight, 1)

    # Run ETA: finish the current module at its measured rate, then add the calibrated cost of
    # everything still pending. Null until the current module has produced enough to measure.
    eta_s = None
    if active is not None and active["eta_s"] is not None:
        eta_s = float(active["eta_s"]) + sum(m["weight_s"] for m in upcoming)
    elif active is None and upcoming and run_state == "running":
        eta_s = sum(m["weight_s"] for m in upcoming)

    n_recent = 12 if recent is None else max(0, int(recent))
    if recent is None and images_total > 50000:
        n_recent = 0                                   # the ticker is a nicety; do not pay for it
    recent_rows = _recent_rows(ref.db_path, active["key"] if active else None, n_recent)

    # Only pay for a process scan when it could show something: a pooled module has real worker
    # processes, and a live run may have a wedged straggler worth surfacing. An idle app must
    # not be scanning the process table twice a second.
    observe = run_state == "running" or (
        active is not None and active["exec_mode"] in ("gpu", "process"))
    workers, workers_meta = _derive_workers(active, _live_workers() if observe else {})

    # "Images finished end to end" is the Reporter's per-image count, since it is the last
    # module every specimen passes through. With the Reporter switched off, fall back to the
    # last module that did run so the figure still means something.
    counts = ledger.get("counts", {})
    images_done = int(counts.get("reporter", {}).get("done", 0))
    if not images_done:
        for mod in reversed(modules):
            if mod["state"] in ("done", "running") and mod["n_done"]:
                images_done = mod["n_done"]
                break
    done_states = ("done", "skipped")
    return {
        "ready": bool(ledger.get("ready")),
        "t": _r(now, 3),
        **ref.as_dict(),
        "state": run_state,
        "stale": stale,
        "stale_for_s": _r(stale_for, 1),
        "started_at": _iso(session_ts),
        "started_ts": _r(session_ts, 3),
        "finished_at": _iso(finished_ts),
        "elapsed_s": _r(elapsed_s, 1),
        "eta_s": _r(eta_s, 1),
        "calibration": _r(calib, 3),
        "images_total": images_total,
        "images_done": images_done,
        "modules": modules,
        "active": _active_view(active),
        "next": ({"key": upcoming[0]["key"], "name": upcoming[0]["name"],
                  "order": upcoming[0]["order"]} if upcoming else None),
        "workers": workers,
        "workers_meta": workers_meta,
        "totals": {
            "images": images_total,
            "images_done": images_done,
            "modules_done": sum(1 for m in modules if m["state"] == "done"),
            "modules_total": len(modules),
            "modules_enabled": sum(1 for m in modules if m["state"] != "skipped"),
            "modules_skipped": sum(1 for m in modules if m["state"] == "skipped"),
            "modules_complete": sum(1 for m in modules if m["state"] in done_states),
            "overall_pct": overall_pct,
            "session_pct": session_pct,
        },
        "recent": recent_rows,
        "provenance": _PROVENANCE,
    }


def _active_view(active: Optional[dict]) -> Optional[dict]:
    """The compact "what LM3 is doing right now" card; ``None`` when nothing is running."""
    if active is None:
        return None
    return {
        "key": active["key"], "name": active["name"], "order": active["order"],
        "pct": active["pct"], "n_done": active["n_done"], "n_total": active["n_total"],
        "n_error": active["n_error"], "rate_per_s": active["rate_per_s"],
        "eta_s": active["eta_s"], "elapsed_s": active["elapsed_s"],
        "started_at": active["started_at"], "started_ts": active["started_ts"],
        "exec_mode": active["exec_mode"], "workers": active["workers"],
        "planned_workers": active["planned_workers"], "device": active["device"],
        "fanout": active["fanout"], "exec_note": active["exec_note"],
    }


def _run_state(active: Optional[dict], errored: list[dict], modules: list[dict],
               ledger: dict, last_activity: Optional[float], now: float) -> str:
    """Run-level state: ``idle`` | ``running`` | ``done`` | ``error``.

    The JobManager's own state wins when the integrator bound one -- it is the only thing that
    knows about ``queued`` and about a job that died before touching the ledger.
    """
    with _BIND_LOCK:
        bound_state = _BOUND_STATE
    _, job_state = _from_job_source()
    override = job_state or bound_state
    if override in ("running", "done", "error"):
        # A bound 'running' job whose ledger already shows an error is an error.
        if override == "running" and errored:
            return "error"
        return override
    if override == "queued":
        return "running"

    if errored:
        return "error"
    if active is not None:
        return "running"
    if not ledger.get("ready"):
        return "idle"
    touched = [m for m in modules if m["started_at"] is not None]
    if not touched:
        return "idle"
    if all(m["state"] in ("done", "skipped") for m in modules):
        return "done"
    # Between modules there is a real window with nothing in state 'running': the barrier runs
    # GPU cleanup and then the next stage's collect_items, which on a large project is seconds.
    # Reporting 'idle' there would strobe the whole app, so recent ledger or log activity keeps
    # the run live -- and STALE_AFTER_S of silence is exactly what says it is not.
    if last_activity is not None and (now - last_activity) <= STALE_AFTER_S:
        return "running"
    return "idle"


def _run_settings(ref: RunRef) -> dict:
    """The settings that drove this run.

    A server job writes its own ``LM3_settings.yaml`` one level above the run dir, which is
    the copy that actually applies to it; anything else falls back to the settings file the
    server itself was started with. Only used for the ``enabled`` flags, and the ledger
    signature backstops it, so a miss degrades rather than lies.
    """
    local = ref.root.parent / "LM3_settings.yaml"
    if local.is_file():
        return _read_yaml(local)
    return _read_yaml(_settings_path())


def _module_enabled(settings: dict, key: str) -> Optional[bool]:
    """``modules.<key>.enabled``; ``None`` when the settings file says nothing about it."""
    node = _dig(settings, "modules", key)
    if not isinstance(node, dict) or "enabled" not in node:
        return None
    return bool(node.get("enabled"))


def _log_mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Run listing
# --------------------------------------------------------------------------- #
def list_runs(limit: int = 60) -> dict:
    """Every run found under the configured output dirs, newest first."""
    def produce() -> list[dict]:
        out: list[dict] = []
        for path in _scan_runs():
            ref = _run_ref(path)
            rows = _stage_rows(ref.db_path)
            starts = [t for t in (_parse_db_ts(r.get("started_at")) for r in rows) if t]
            ends = [t for t in (_parse_db_ts(r.get("finished_at")) for r in rows) if t]
            states = {str(r.get("state")) for r in rows}
            if "error" in states:
                state = "error"
            elif "running" in states:
                state = "running"
            elif rows and states <= {"done"}:
                state = "done"
            elif starts:
                state = "idle"
            else:
                state = "idle"
            conn = _connect_ro(ref.db_path)
            n_images = 0
            if conn is not None:
                try:
                    row = conn.execute("SELECT COUNT(*) FROM specimen").fetchone()
                    n_images = int(row[0]) if row and row[0] is not None else 0
                except sqlite3.Error:
                    n_images = 0
                finally:
                    conn.close()
            try:
                stat = ref.db_path.stat()
                mtime, size = stat.st_mtime, stat.st_size
            except OSError:
                mtime, size = 0.0, 0
            out.append({
                "run_name": ref.run_name,
                "path": str(ref.root),
                "db_path": str(ref.db_path),
                "log_path": str(ref.log_path) if ref.log_path.is_file() else None,
                "started": _iso(min(starts)) if starts else None,
                "finished": _iso(max(ends)) if (ends and state == "done") else None,
                "n_images": n_images,
                "state": state,
                "modules_done": sum(1 for r in rows if r.get("state") == "done"),
                "modules_total": len(rows) or len(module_table()),
                "mtime": _r(mtime, 3),
                "size_bytes": size,
                "has_reports": (ref.root / "reports").is_dir(),
            })
        out.sort(key=lambda r: (r["mtime"] or 0.0), reverse=True)
        return out

    runs = _RUNS_CACHE.get("runs", produce)
    current = resolve_run()
    current_path = str(current.root) if current else None
    trimmed = runs[: max(1, int(limit))]
    for entry in trimmed:
        entry["active"] = entry["path"] == current_path
    return {"t": _r(time.time(), 3), "n": len(runs), "active": current_path,
            "roots": [str(p) for p in _search_roots()], "runs": trimmed}


# --------------------------------------------------------------------------- #
# Console log tailing
# --------------------------------------------------------------------------- #
# ``core.logging_setup`` formats every line as
#     %(asctime)s %(levelname)-7s %(name)s: %(message)s   with datefmt %H:%M:%S
# Logger names never contain whitespace, so a non-greedy run up to the first ": " separates
# the source from a message that may itself be full of colons.
_LINE_RE = re.compile(
    r"^(?P<t>\d{2}:\d{2}:\d{2})\s+(?P<level>[A-Z]+)\s+(?P<src>\S+?):\s?(?P<msg>.*)$"
)
_LEVELS = {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL", "SUCCESS", "TRACE"}


def parse_log_line(raw: str, *, seq: int = 0, anchor: Optional[float] = None,
                   previous_level: str = "INFO") -> dict:
    """Parse one log line into the frame shape the console renders.

    Emits both ``src`` and ``logger`` for the same value: the CSS console column is ``.src``
    while the progress contract names it ``logger``, and one duplicated short string per line
    is cheaper than making either side translate.

    A line that does not match the format is a CONTINUATION -- a traceback body, or a
    multi-line message such as the end-of-run time report. It inherits the previous line's
    level and is flagged ``trace`` so the console can indent it instead of pretending it is a
    fresh record.
    """
    text = raw.rstrip("\n").rstrip("\r")
    match = _LINE_RE.match(text)
    if not match or match.group("level") not in _LEVELS:
        return {"t": None, "ts": None, "level": previous_level, "src": None, "logger": None,
                "msg": text, "trace": True, "mark": False, "kind": None, "seq": seq}
    level = match.group("level")
    level = {"WARN": "WARNING", "FATAL": "CRITICAL"}.get(level, level)
    msg = match.group("msg")
    kind = None
    if msg.startswith("LM3 start"):
        kind = "run_start"
    elif msg.startswith("LM3 complete"):
        kind = "run_end"
    elif msg.startswith("start "):
        kind = "module_start"
    elif msg.startswith("skip "):
        kind = "module_skip"
    return {
        "t": match.group("t"),
        "ts": _resolve_clock(match.group("t"), anchor),
        "level": level,
        "src": match.group("src"),
        "logger": match.group("src"),
        "msg": msg,
        "trace": False,
        "mark": kind in ("run_start", "run_end"),
        "kind": kind,
        "seq": seq,
    }


def _resolve_clock(clock: str, anchor: Optional[float]) -> Optional[float]:
    """Turn a bare ``HH:MM:SS`` into unix seconds by anchoring it to the log file's own date.

    The formatter drops the date, so the only way back is to pin the clock to a day. The log's
    mtime is that day (it is the file those lines were written to); a time that lands in that
    day's future belongs to the day before, which is what makes a run spanning midnight come
    out in order. Local time throughout -- ``logging`` formats asctime in local time, unlike
    the ledger's UTC timestamps.
    """
    if anchor is None:
        return None
    try:
        hour, minute, second = (int(p) for p in clock.split(":"))
    except ValueError:
        return None
    day = datetime.fromtimestamp(anchor).replace(
        hour=hour, minute=minute, second=second, microsecond=0)
    if day.timestamp() > anchor + 60.0:
        day -= timedelta(days=1)
    return round(day.timestamp(), 3)


def _read_tail_lines(path: Path, *, max_lines: int) -> list[dict]:
    """Parse the last ``max_lines`` complete lines of a log file (empty when absent)."""
    try:
        stat = path.stat()
        with path.open("rb") as fh:
            start = max(0, stat.st_size - _LOG_BACKFILL_BYTES)
            fh.seek(start)
            blob = fh.read()
    except OSError:
        return []
    text = blob.decode("utf-8", "replace")
    if start:
        text = text.split("\n", 1)[-1]                 # drop the partial first line
    raw_lines = [ln for ln in text.split("\n") if ln.strip()]
    raw_lines = raw_lines[-max(1, int(max_lines)):]
    out: list[dict] = []
    level = "INFO"
    for i, raw in enumerate(raw_lines):
        entry = parse_log_line(raw, seq=i, anchor=stat.st_mtime, previous_level=level)
        level = entry["level"]
        out.append(entry)
    return out


def log_tail(run: Optional[str] = None, db: Optional[str] = None, *,
             lines: int = DEFAULT_LOG_BACKFILL) -> dict:
    """The last ``lines`` parsed console records for a run (used for the console's first paint)."""
    ref = resolve_run(run, db)
    if ref is None:
        return {"ready": False, "run_name": None, "path": None, "lines": []}
    parsed = _read_tail_lines(ref.log_path, max_lines=lines)
    return {
        "ready": ref.log_path.is_file(),
        "run_name": ref.run_name,
        "path": str(ref.log_path),
        "n": len(parsed),
        "lines": parsed,
    }


class LogTailer:
    """Follow a log file the way ``tail -F`` does, tolerating everything LM3 can do to it.

    The file may not exist yet (the run dir is built before the first record), it may be
    replaced wholesale by a new run, and a read can land mid-line while the handler is
    flushing. Each ``poll`` therefore re-stats the path, reopens on an inode change or a
    shrink (truncation / rotation), and holds any unterminated tail back until its newline
    arrives, so a half-written record is never rendered.
    """

    def __init__(self, path: Path, *, backfill: int = DEFAULT_LOG_BACKFILL) -> None:
        self.path = Path(path)
        self.backfill = int(backfill)
        self._fh: Any = None
        self._inode: Optional[int] = None
        self._pos = 0
        self._buf = ""
        self._seq = 0
        self._level = "INFO"
        self._anchor: Optional[float] = None
        self.state = "waiting"                          # waiting | tailing | rotated | missing

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
        self._fh = None

    def _open(self, stat: os.stat_result, *, first: bool) -> list[dict]:
        """(Re)open the file. The first open replays a backfill; a rotation starts at byte 0."""
        self.close()
        self._fh = self.path.open("r", encoding="utf-8", errors="replace")
        self._inode = stat.st_ino
        self._buf = ""
        self._anchor = stat.st_mtime
        replay: list[dict] = []
        if first and self.backfill > 0:
            replay = _read_tail_lines(self.path, max_lines=self.backfill)
            for entry in replay:
                entry["seq"] = self._seq
                self._seq += 1
            if replay:
                self._level = replay[-1]["level"]
            self._fh.seek(0, os.SEEK_END)
        else:
            self._fh.seek(0)
        self._pos = self._fh.tell()
        return replay

    def poll(self) -> list[dict]:
        """Return the log records that appeared since the last call (possibly empty).

        ``state`` is a one-poll pulse: it reads ``rotated`` for exactly the call in which the
        file was replaced or truncated, then falls back to ``tailing``. The console draws a
        divider off that pulse, so it must not latch.
        """
        if self.state == "rotated":
            self.state = "tailing"
        try:
            stat = self.path.stat()
        except OSError:
            if self._fh is not None:
                self.close()
            self.state = "missing" if self.state == "tailing" else "waiting"
            return []

        out: list[dict] = []
        if self._fh is None:
            out.extend(self._open(stat, first=True))
            self.state = "tailing"
        elif stat.st_ino != self._inode or stat.st_size < self._pos:
            # A new run replaced the file, or a handler truncated it: start over from the top
            # rather than seeking past content that is no longer there.
            out.extend(self._open(stat, first=False))
            self.state = "rotated"
        self._anchor = stat.st_mtime

        try:
            chunk = self._fh.read()
        except (OSError, ValueError):
            self.close()
            return out
        self._pos = self._fh.tell()
        if not chunk:
            return out

        self._buf += chunk
        chunk_lines = self._buf.split("\n")
        if len(self._buf) > _MAX_PARTIAL_LINE:
            # A megabyte with no newline in it is not a log line any more (a binary blob got
            # into the file, or a writer died mid-record). Emit what we have rather than let
            # the buffer grow without bound.
            self._buf = ""
        else:
            self._buf = chunk_lines.pop()               # the tail is partial until its newline

        for raw in chunk_lines:
            if not raw.strip():
                continue
            entry = parse_log_line(raw, seq=self._seq, anchor=self._anchor,
                                   previous_level=self._level)
            self._level = entry["level"]
            self._seq += 1
            out.append(entry)
        return out


# --------------------------------------------------------------------------- #
# Server-Sent Events (synchronous generators; the router runs the same sequence
# on the event loop so it never blocks it)
# --------------------------------------------------------------------------- #
def _frame(kind: str, **payload: Any) -> str:
    """One typed SSE ``data:`` frame -- the same envelope convention ``metrics.py`` uses, so
    both streams are consumed by a single ``switch (frame.type)``."""
    return "data: " + json.dumps({"type": kind, **payload}) + "\n\n"


def _heartbeat() -> str:
    """An SSE comment. Keeps the connection warm without firing an onmessage handler."""
    return ": ping\n\n"


def _status_fingerprint(snapshot: dict) -> tuple:
    """What must change before a status frame is worth resending."""
    return (
        snapshot.get("state"), snapshot.get("run_name"), snapshot.get("stale"),
        snapshot.get("totals", {}).get("overall_pct"),
        tuple((m["key"], m["state"], m["n_done"], m["n_error"]) for m in snapshot.get("modules", [])),
        tuple((w["id"], w["state"], w.get("cpu_pct")) for w in snapshot.get("workers", [])),
    )


def status_frames(run: Optional[str] = None, db: Optional[str] = None, *,
                  interval: Optional[float] = None, max_seconds: float = 86400.0) -> Iterator[str]:
    """Blocking generator of status SSE frames (the sync form; handy for tests and curl).

    Frames go out when something actually changed, plus a forced one every 2 s so the client's
    elapsed clock and "last heard from" stay honest. That floor doubles as the keep-alive, so
    this stream needs no heartbeat comments -- unlike the console, which can legitimately be
    silent for minutes.
    """
    period = float(interval or STATUS_INTERVAL_S)
    deadline = time.time() + max_seconds
    last_key: Any = object()
    last_sent = 0.0
    while time.time() < deadline:
        snapshot = status(run, db)
        key = _status_fingerprint(snapshot)
        now = time.time()
        if key != last_key or now - last_sent >= 2.0:
            yield _frame("status", snapshot=snapshot)
            last_key, last_sent = key, now
        time.sleep(period if snapshot.get("state") == "running" else STATUS_IDLE_INTERVAL_S)


def log_frames(run: Optional[str] = None, db: Optional[str] = None, *,
               backfill: int = DEFAULT_LOG_BACKFILL, interval: Optional[float] = None,
               max_seconds: float = 86400.0) -> Iterator[str]:
    """Blocking generator of console SSE frames (the sync form)."""
    period = float(interval or LOG_INTERVAL_S)
    deadline = time.time() + max_seconds
    ref = resolve_run(run, db)
    tailer = LogTailer(ref.log_path, backfill=backfill) if ref else None
    yield _frame("logmeta", state=("waiting" if tailer is None else tailer.state),
                 run=(ref.run_name if ref else None),
                 path=(str(ref.log_path) if ref else None), reset=True)
    last_beat = time.time()
    while time.time() < deadline:
        if tailer is not None:
            lines = tailer.poll()
            if tailer.state == "rotated":
                yield _frame("logmeta", state="rotated", run=(ref.run_name if ref else None),
                             path=str(tailer.path), reset=True)
            if lines:
                yield _frame("log", lines=lines, run=ref.run_name if ref else None)
                last_beat = time.time()
        if time.time() - last_beat >= HEARTBEAT_S:
            yield _heartbeat()
            last_beat = time.time()
        time.sleep(period)


# --------------------------------------------------------------------------- #
# Auth for the SSE routes
# --------------------------------------------------------------------------- #
def _expected_token() -> Optional[str]:
    """The server's Bearer secret. ``app._server_token`` mints it into the environment at
    startup, so the env is the cheap read; the import is only the cold-start fallback."""
    token = os.environ.get("LM3_SERVER_TOKEN")
    if token:
        return token
    try:
        from leafmachine3.server.app import _server_token

        return _server_token()
    except Exception:  # noqa: BLE001
        return None


def _token_ok(header: str, query: Optional[str]) -> bool:
    """Accept the shared secret from the ``Authorization`` header OR a ``?token=`` query param.

    The query form exists because ``EventSource`` cannot set request headers; loopback-only
    binding is what makes putting the secret in a URL acceptable. Constant-time compares on
    both paths. A server with no secret configured at all has nothing to check -- that only
    happens when this router is mounted outside ``app.py``, which mints one at startup.
    """
    expected = _expected_token()
    if not expected:
        # FAIL CLOSED. These streams carry the run log and the project ledger, so "no secret
        # configured" must mean "nobody gets in", not "everybody does". app.py always mints a
        # token, so this only bites a router mounted standalone -- which is exactly the case
        # where an open stream would be unnoticed.
        return False
    if header and secrets.compare_digest(header, f"Bearer {expected}"):
        return True
    return bool(query and secrets.compare_digest(str(query), expected))


# --------------------------------------------------------------------------- #
# Optional FastAPI router (the integrator mounts this; app.py is not edited here)
# --------------------------------------------------------------------------- #
def router(dependencies: Optional[list] = None) -> Any:
    """Build the status / logs / runs APIRouter.

    Mount it from ``app.py`` exactly like the metrics one::

        from leafmachine3.server import progress_api
        app.include_router(progress_api.router(dependencies=[Depends(require_token)]))

    ``dependencies`` is applied PER ROUTE rather than to the router, because the two SSE
    routes cannot use a header-only guard: ``EventSource`` is unable to set an
    ``Authorization`` header, so they accept ``?token=`` as well and check it themselves
    against the same secret. Loopback-only binding is what keeps that safe.

    ``fastapi`` is imported lazily, exactly like ``app.create_app`` and ``metrics.router``,
    so a base install without the ``server`` extra can still import this module.
    """
    import asyncio

    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import StreamingResponse

    # ``from __future__ import annotations`` (top of this file, per house style) turns every
    # annotation into a STRING, and FastAPI resolves those strings with typing.get_type_hints
    # against the handler's MODULE globals -- where a name imported inside this function does
    # not exist. Without this, ``request: Request`` is unresolvable and FastAPI falls back to
    # treating it as a query parameter, so every SSE connection 422s with "field required:
    # request". Publishing the two names it must resolve is what lets the lazy import (needed
    # so a base install can still import this module) coexist with postponed annotations.
    globals().setdefault("Request", Request)
    globals().setdefault("StreamingResponse", StreamingResponse)

    guards = list(dependencies or [])
    api = APIRouter(tags=["progress"])
    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                   "Connection": "keep-alive"}

    # These three are plain ``def``, not ``async def``: each opens the project ledger and runs
    # GROUP BY counts over ``image_status`` (1.6M rows on a 100k-image project), scans the run
    # roots, or reads the tail of the log file -- all blocking. Starlette runs a sync endpoint in
    # its threadpool, so the event loop stays free for the SSE routes below (which push the same
    # reads to a thread with ``asyncio.to_thread`` for exactly the same reason).
    @api.get("/v1/status", dependencies=guards)
    def get_status(run: Optional[str] = None, db: Optional[str] = None,
                   recent: Optional[int] = None) -> dict:
        return status(run, db, recent=recent)

    @api.get("/v1/runs", dependencies=guards)
    def get_runs(limit: int = 60) -> dict:
        return list_runs(limit=limit)

    @api.get("/v1/logs", dependencies=guards)
    def get_logs(run: Optional[str] = None, db: Optional[str] = None,
                 lines: int = DEFAULT_LOG_BACKFILL) -> dict:
        return log_tail(run, db, lines=lines)

    @api.get("/v1/status/stream")
    async def stream_status(request: Request, run: Optional[str] = None,
                            db: Optional[str] = None, interval: Optional[float] = None,
                            token: Optional[str] = Query(default=None)) -> "StreamingResponse":
        if not _token_ok(request.headers.get("authorization", ""), token):
            raise HTTPException(status_code=401, detail="invalid or missing token")
        period = float(interval or STATUS_INTERVAL_S)

        async def gen() -> Any:
            # Same sequence as status_frames, paced with asyncio.sleep and gated on the client
            # still being there. Every read is either cached or an indexed count, so this
            # streams without ever blocking the loop, and an idle run costs one query per 2 s.
            last_key: Any = object()
            last_sent = 0.0
            while not await request.is_disconnected():
                snapshot = await asyncio.to_thread(status, run, db)
                key = _status_fingerprint(snapshot)
                now = time.time()
                if key != last_key or now - last_sent >= 2.0:
                    yield _frame("status", snapshot=snapshot)
                    last_key, last_sent = key, now
                await asyncio.sleep(period if snapshot.get("state") == "running"
                                    else STATUS_IDLE_INTERVAL_S)

        return StreamingResponse(gen(), media_type="text/event-stream", headers=sse_headers)

    @api.get("/v1/logs/stream")
    async def stream_logs(request: Request, run: Optional[str] = None, db: Optional[str] = None,
                          backfill: int = DEFAULT_LOG_BACKFILL,
                          interval: Optional[float] = None,
                          token: Optional[str] = Query(default=None)) -> "StreamingResponse":
        if not _token_ok(request.headers.get("authorization", ""), token):
            raise HTTPException(status_code=401, detail="invalid or missing token")
        period = float(interval or LOG_INTERVAL_S)

        async def gen() -> Any:
            ref = resolve_run(run, db)
            tailer = LogTailer(ref.log_path, backfill=backfill) if ref else None
            yield _frame("logmeta", state=(tailer.state if tailer else "waiting"),
                         run=(ref.run_name if ref else None),
                         path=(str(ref.log_path) if ref else None), reset=True)
            last_beat = time.time()
            last_check = 0.0
            # A fresh LogTailer reports "waiting" because it does not open the file until its
            # first poll. That opening frame is the only logmeta the console used to receive, so
            # it sat on "waiting for log" for the whole run while lines streamed in underneath.
            last_meta = tailer.state if tailer is not None else None
            try:
                while not await request.is_disconnected():
                    # Follow the ACTIVE run, not just the file we opened with: when the user
                    # starts a new run the console has to switch files and replay its head.
                    if run is None and db is None and time.time() - last_check > 2.0:
                        last_check = time.time()
                        current = await asyncio.to_thread(resolve_run, None, None)
                        if current is not None and (tailer is None
                                                    or current.log_path != tailer.path):
                            if tailer is not None:
                                tailer.close()
                            ref, tailer = current, LogTailer(current.log_path, backfill=backfill)
                            last_meta = tailer.state
                            yield _frame("logmeta", state=tailer.state, run=current.run_name,
                                         path=str(current.log_path), reset=True)
                    if tailer is not None:
                        lines = await asyncio.to_thread(tailer.poll)
                        if tailer.state == "rotated":
                            # The file was replaced or truncated under us -- a new run took
                            # over this path. Tell the console to clear rather than let two
                            # runs' output interleave in one buffer.
                            yield _frame("logmeta", state="rotated",
                                         run=(ref.run_name if ref else None),
                                         path=str(tailer.path), reset=True)
                            last_meta = tailer.state
                        elif tailer.state != last_meta:
                            # waiting -> tailing (or -> missing). No `reset`: the file is the same
                            # one, so the console must keep the scrollback it has already drawn.
                            last_meta = tailer.state
                            yield _frame("logmeta", state=tailer.state,
                                         run=(ref.run_name if ref else None),
                                         path=str(tailer.path))
                        if lines:
                            yield _frame("log", lines=lines,
                                         run=(ref.run_name if ref else None))
                            last_beat = time.time()
                    if time.time() - last_beat >= HEARTBEAT_S:
                        yield _heartbeat()
                        last_beat = time.time()
                    await asyncio.sleep(period)
            finally:
                if tailer is not None:
                    tailer.close()

        return StreamingResponse(gen(), media_type="text/event-stream", headers=sse_headers)

    return api


__all__ = [
    "status", "list_runs", "log_tail", "status_frames", "log_frames",
    "bind_run", "clear_run", "bind_job_source", "active_run", "resolve_run",
    "module_table", "parse_log_line", "LogTailer", "RunRef", "router",
]
