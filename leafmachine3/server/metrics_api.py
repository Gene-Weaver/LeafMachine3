"""leafmachine3.server.metrics_api -- HTTP surface for the machine monitor + LM3 run control.

Two jobs, one router:

1. **Machine monitor.** Publishes the ring buffer kept by :mod:`leafmachine3.server.metrics` --
   a current point, the rolling window that fills the plots on first paint, and an SSE stream that
   keeps them live. Nothing here samples hardware; every read is a ring-buffer copy (microseconds),
   so a browser at 2 Hz costs the same as a browser at 0.1 Hz.

2. **Run control.** Starts / stops / reports the LM3 run behind the app's Run button.

   THE RUN IS A SUBPROCESS, AND THAT IS NOT NEGOTIABLE. ``leafmachine3.machine3.main`` opens with
   :func:`~leafmachine3.machine3._exec_with_cuda_libpath`, which puts the venv's bundled
   ``nvidia-*`` libs on ``LD_LIBRARY_PATH`` and RE-EXECS the interpreter, because the dynamic loader
   reads that variable only at PROCESS START. onnxruntime-gpu dlopen's
   ``libonnxruntime_providers_cuda.so`` (which links libcublasLt / cuDNN 9 from those wheels); if the
   variable is not already set when the process starts, ORT silently falls back to
   ``CPUExecutionProvider`` and every ONNX module runs on the CPU -- no error, just a run that is
   ~40x slower. Calling :func:`leafmachine3.machine3.machine3` in-process cannot fix that (machine3()
   deliberately does NOT re-exec: it would restart the server instead of LM3). So the app launches
   the CLI, in its own session, and reads its progress from the project SQLite ledger.

Public Python surface (other server modules import these rather than re-deriving them):
    hardware_profile()      -> parsed + annotated hardware_settings.yaml
    active()                -> the most recent run record (active or finished)
    is_active()             -> bool
    active_db_path()        -> Path | None    (the project SQLite the progress module tails)
    active_log_path()       -> Path | None    (<run>/logs/lm3.log)
    active_console_path()   -> Path | None    (stdout+stderr capture, catches pre-logging crashes)
    start_run(...) / stop_run(...)            -> raise RunError (carries .status) on refusal
    router(dependencies=[Depends(require_token)])  -> the APIRouter the integrator mounts

MOUNTING: this router SUPERSEDES :func:`leafmachine3.server.metrics.router` -- both serve
``/v1/metrics``, and mounting both would leave the first-registered one shadowing the other. Mount
this one only.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

from leafmachine3.server import metrics

log = logging.getLogger("leafmachine3.server.metrics_api")

# Where the profiler writes its tuned profile. hardware_setup.HW_PATH is RELATIVE ("beside
# LM3_settings.yaml"), so it resolves against the CWD -- we search the same places a run would.
HW_FILENAME = "hardware_settings.yaml"
CFG_FILENAME = "LM3_settings.yaml"

# Grace between SIGTERM and SIGKILL when stopping a run. LM3 checkpoints every image into the
# project DB and reclaims 'running' stages on the next start, so a hard stop costs at most the
# images currently in flight -- 10 s is plenty for the executor's pools to unwind on their own.
DEFAULT_STOP_GRACE_S = 10.0

# Mirrors leafmachine3.pipeline.STAGE_ORDER (key, display name, device kind, cpu parallelism).
# Duplicated on purpose: importing pipeline.STAGE_ORDER would drag in every stage module (torch,
# cv2, ultralytics) just to label a settings panel. leafmachine3.core.config.CANONICAL_STAGE_KEYS is
# the cheap cross-check, and _stage_rows() logs if the two ever drift.
_STAGE_META: tuple[tuple[str, str, str, str], ...] = (
    ("mp_conversion_factor", "MP Conversion Factor", "cpu", "thread"),
    ("archival_detector", "Archival Detector", "gpu", ""),
    ("plant_detector", "Plant Detector", "gpu", ""),
    ("specimen_segmenter", "Specimen Segmenter", "gpu", ""),
    ("phenology_detector", "Phenology Detector", "cpu", "thread"),
    ("ruler_classifier", "Ruler Classifier", "gpu", ""),
    ("ruler_cf", "Ruler Conversion Factor", "cpu", "process"),
    ("leaf_segmenter", "Leaf Segmenter", "gpu", ""),
    ("morphology", "Morphology", "cpu", "thread"),
    ("landmark_detector", "Landmark Detector", "gpu", ""),
    ("landmark_measurements", "Landmark Measurements", "cpu", "thread"),
    ("leaf_orientation", "Leaf Orientation", "cpu", "thread"),
    ("petiole_width", "Petiole Width", "cpu", "thread"),
    ("metric_grounding", "Metric Grounding", "cpu", "thread"),
    ("reporter", "Reporter", "cpu", "thread"),
    ("ect", "ECT", "cpu", "process"),
)


class RunError(RuntimeError):
    """A refusal the HTTP layer turns into a status code (409 busy, 400 bad request, ...)."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    """The LM3 checkout root (the directory that holds the ``leafmachine3`` package)."""
    return Path(__file__).resolve().parents[2]


def _search_roots() -> list[Path]:
    """Candidate working directories, most-specific first: CWD, then the checkout root."""
    roots: list[Path] = []
    try:
        roots.append(Path.cwd())
    except OSError:                                   # CWD was deleted underneath us
        pass
    root = _repo_root()
    if root not in roots:
        roots.append(root)
    return roots


def _find_file(filename: str, env_var: str) -> Optional[Path]:
    """Locate ``filename``: an explicit env override wins, else the first search root that has it."""
    override = os.environ.get(env_var)
    if override:
        path = Path(override).expanduser()
        return path if path.is_file() else None
    for root in _search_roots():
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def hardware_path() -> Optional[Path]:
    return _find_file(HW_FILENAME, "LM3_HARDWARE_SETTINGS")


def default_config_path() -> Optional[Path]:
    return _find_file(CFG_FILENAME, "LM3_SETTINGS")


def _free_gb(path: Path) -> Optional[float]:
    """Free space on the filesystem holding ``path`` (walking up to the nearest existing parent)."""
    probe = path
    for _ in range(6):
        if probe.exists():
            try:
                return round(shutil.disk_usage(probe).free / (1024.0 ** 3), 1)
            except OSError:
                return None
        if probe.parent == probe:
            break
        probe = probe.parent
    return None


def _iso(ts: Optional[float]) -> Optional[str]:
    if not ts:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


# --------------------------------------------------------------------------- #
# Hardware profile
# --------------------------------------------------------------------------- #
def _stage_rows(stages: dict, n_gpus: int, io_workers: Optional[int]) -> list[dict]:
    """One row per canonical module, annotated with the PLANNED worker count.

    The planned count is the number the status tab draws per-worker bars for: GPU modules run
    ``workers_per_gpu`` spawn workers on EACH visible GPU, CPU modules run ``workers`` (threads or
    spawn processes depending on ``cpu_parallel``). A module missing from the profile was disabled
    when LM3_Setup last ran -- it is still listed, marked ``tuned: false``, so the panel shows the
    full pipeline instead of a hole.
    """
    known = {key for key, _, _, _ in _STAGE_META}
    extra = [k for k in stages if k not in known]
    if extra:                                          # profile written by a newer LM3 than this UI
        log.debug("hardware profile has unknown stage keys: %s", extra)

    rows: list[dict] = []
    order = 0
    for key, name, device, parallel in _STAGE_META:
        cfg = stages.get(key) or {}
        order += 1
        if device == "gpu":
            per_gpu = cfg.get("workers_per_gpu")
            planned = int(per_gpu) * max(1, n_gpus) if per_gpu else None
        else:
            workers = cfg.get("workers", io_workers)
            planned = int(workers) if workers else None
        rows.append({
            "key": key,
            "name": name,
            "order": order,
            "device": device,
            "cpu_parallel": str(cfg.get("cpu_parallel", parallel)) or None,
            "tuned": key in stages,
            "planned_workers": planned,
            "workers": cfg.get("workers"),
            "workers_per_gpu": cfg.get("workers_per_gpu"),
            "batch": cfg.get("batch"),
            "queue_size": cfg.get("queue_size"),
            "peak_vram_mb": cfg.get("peak_vram_mb"),
            "io_bound": bool(cfg.get("io_bound", False)),
            "disk_write_mbps": cfg.get("disk_write_mbps"),
            "min_pool_items": cfg.get("min_pool_items"),
            "spawn_overhead_s": cfg.get("spawn_overhead_s"),
        })
    for key in extra:
        cfg = stages.get(key) or {}
        order += 1
        rows.append({"key": key, "name": key.replace("_", " ").title(), "order": order,
                     "device": "cpu", "cpu_parallel": cfg.get("cpu_parallel"), "tuned": True,
                     "planned_workers": cfg.get("workers"), "workers": cfg.get("workers"),
                     "workers_per_gpu": cfg.get("workers_per_gpu"), "batch": cfg.get("batch"),
                     "queue_size": cfg.get("queue_size"), "peak_vram_mb": cfg.get("peak_vram_mb"),
                     "io_bound": bool(cfg.get("io_bound", False)),
                     "disk_write_mbps": cfg.get("disk_write_mbps"),
                     "min_pool_items": cfg.get("min_pool_items"),
                     "spawn_overhead_s": cfg.get("spawn_overhead_s")})
    return rows


def _drift(fingerprint: dict) -> dict:
    """Compare the profile's fingerprint against the LIVE machine.

    hardware_setup does the authoritative check (it also hashes the model exports), but that needs a
    loaded Config. This is the cheap version -- core count, RAM, GPU names, driver -- enough for the
    app to say "your profile predates this hardware, rerun LM3_Setup" without touching disk.
    """
    reasons: list[str] = []
    try:
        machine = metrics.describe_machine()
    except Exception:                                  # noqa: BLE001 - a hint must never 500
        return {"checked": False, "stale": False, "reasons": []}

    cores = machine.get("cpu", {}).get("cores_logical")
    want_cores = fingerprint.get("cpu_cores")
    if cores and want_cores and int(cores) != int(want_cores):
        reasons.append(f"CPU cores {want_cores} -> {cores}")

    ram_mb = machine.get("memory", {}).get("ram_total_mb")
    want_ram = fingerprint.get("ram_gb")
    if ram_mb and want_ram and abs(ram_mb / 1024.0 - float(want_ram)) > 2.0:
        reasons.append(f"RAM {want_ram} GB -> {round(ram_mb / 1024.0)} GB")

    live_gpus = [g.get("name") for g in machine.get("gpus", [])]
    want_gpus = [g[0] if isinstance(g, (list, tuple)) and g else None
                 for g in (fingerprint.get("gpus") or [])]
    if live_gpus != want_gpus:
        reasons.append(f"GPUs {want_gpus or 'none'} -> {live_gpus or 'none'}")

    driver, want_driver = machine.get("gpu_driver"), fingerprint.get("driver")
    if driver and want_driver and str(driver) != str(want_driver):
        reasons.append(f"driver {want_driver} -> {driver}")

    return {"checked": True, "stale": bool(reasons), "reasons": reasons}


def hardware_profile() -> dict:
    """``hardware_settings.yaml`` parsed, annotated, and safe to render.

    ``raw`` is the file verbatim (so the app can show it as YAML); everything above it is the
    derived view the tuning panel actually draws: per-module planned workers, GPU sizing, the
    measured spawn overhead and the disk-write knee.
    """
    path = hardware_path()
    if path is None:
        return {"available": False, "path": None, "reason": f"no {HW_FILENAME} found -- run LM3_Setup",
                "raw": {}, "stages": _stage_rows({}, 0, None), "gpus": [], "n_gpus": 0,
                "tuning": {}, "fingerprint": {}, "drift": {"checked": False, "stale": False,
                                                           "reasons": []}}
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {"available": False, "path": str(path), "reason": f"could not read {path.name}: {exc}",
                "raw": {}, "stages": _stage_rows({}, 0, None), "gpus": [], "n_gpus": 0,
                "tuning": {}, "fingerprint": {}, "drift": {"checked": False, "stale": False,
                                                           "reasons": []}}

    gpus = list(raw.get("gpus") or [])
    stages = dict(raw.get("stages") or {})
    io_workers = raw.get("io_workers")
    tmp_dir = raw.get("tmp_dir")
    rows = _stage_rows(stages, len(gpus), io_workers)

    # The two measured constants worth surfacing on their own: the spawn-pool startup cost (which
    # sets each process module's min_pool_items) and the disk-write knee (which caps the Reporter).
    spawn = next((r["spawn_overhead_s"] for r in rows if r.get("spawn_overhead_s")), None)
    reporter = next((r for r in rows if r["key"] == "reporter"), {})
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None

    return {
        "available": True,
        "path": str(path),
        "mtime": mtime,
        "age_s": round(time.time() - mtime, 1) if mtime else None,
        "generated_at": raw.get("generated_at"),
        "lm3_version": raw.get("lm3_version"),
        "provider": raw.get("provider"),
        "precision": raw.get("precision"),
        "cpu_cores": raw.get("cpu_cores"),
        "ram_gb": raw.get("ram_gb"),
        "io_workers": io_workers,
        "tmp_dir": tmp_dir,
        "tmp_free_gb": _free_gb(Path(str(tmp_dir))) if tmp_dir else None,
        "gpus": gpus,
        "n_gpus": len(gpus),
        "stages": rows,
        "tuning": {
            "io_workers": io_workers,
            "spawn_overhead_s": spawn,
            "disk_write_mbps": reporter.get("disk_write_mbps"),
            "disk_writers": reporter.get("workers"),
            "tmp_dir": tmp_dir,
            "provider": raw.get("provider"),
            "precision": raw.get("precision"),
            "n_gpus": len(gpus),
            "gpu_workers": sum(r["planned_workers"] or 0 for r in rows if r["device"] == "gpu"),
        },
        "fingerprint": dict(raw.get("fingerprint") or {}),
        "drift": _drift(dict(raw.get("fingerprint") or {})),
        "raw": raw,
    }


# --------------------------------------------------------------------------- #
# Run control
# --------------------------------------------------------------------------- #
class _Run:
    """One launched (or adopted) LM3 process and everything the app needs to describe it."""

    def __init__(self, **fields: Any) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.log_fh: Any = None
        self.pid: int = 0
        self.pgid: Optional[int] = None
        self.create_time: Optional[float] = None       # psutil start time -> guards PID reuse
        self.adopted = False
        self.state = "running"                         # running | stopping | done | error
        self.returncode: Optional[int] = None
        self.stopped_by_user = False
        self.error: Optional[str] = None
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.run_name = ""
        self.config_path = ""
        self.cwd = ""
        self.output_dir = ""
        self.run_dir = ""
        self.db_path = ""
        self.log_path = ""
        self.console_log = ""
        self.tmp_dir = ""
        self.input_dirs: list[str] = []
        self.restart: list[str] = []
        self.argv: list[str] = []
        for key, value in fields.items():
            setattr(self, key, value)

    @property
    def alive(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        return _pid_alive(self.pid, self.create_time)

    def record(self) -> dict:
        """The wire shape of ``GET /v1/run/active`` (also used for the SSE ``run`` frame)."""
        active = self.state in ("running", "stopping")
        end = self.finished_at or time.time()
        return {
            "active": active,
            "state": self.state,
            "run_name": self.run_name,
            "pid": self.pid,
            "pgid": self.pgid,
            "started_at": self.started_at,
            "started_iso": _iso(self.started_at),
            "finished_at": self.finished_at,
            "finished_iso": _iso(self.finished_at),
            "elapsed_s": round(end - self.started_at, 1) if self.started_at else None,
            "returncode": self.returncode,
            "stopped_by_user": self.stopped_by_user,
            "adopted": self.adopted,
            "error": self.error,
            "config_path": self.config_path,
            "cwd": self.cwd,
            "output_dir": self.output_dir,
            "run_dir": self.run_dir,
            "db_path": self.db_path,
            "log_path": self.log_path,
            "console_log": self.console_log,
            "tmp_dir": self.tmp_dir,
            "input_dirs": list(self.input_dirs),
            "restart": list(self.restart),
            "argv": list(self.argv),
        }

    def persist(self) -> None:
        """Write the record so a RESTARTED server can re-adopt a still-running LM3 (see _adopt)."""
        path = _state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if self.state in ("running", "stopping"):
                payload = self.record()
                payload["create_time"] = self.create_time
                path.write_text(json.dumps(payload), encoding="utf-8")
            elif path.exists():
                path.unlink()
        except OSError:                                # bookkeeping only -- never fail a run on it
            log.debug("could not persist run state to %s", path, exc_info=True)


_IDLE_RECORD: dict = {
    "active": False, "state": "idle", "run_name": None, "pid": None, "pgid": None,
    "started_at": None, "started_iso": None, "finished_at": None, "finished_iso": None,
    "elapsed_s": None, "returncode": None, "stopped_by_user": False, "adopted": False,
    "error": None, "config_path": None, "cwd": None, "output_dir": None, "run_dir": None,
    "db_path": None, "log_path": None, "console_log": None, "tmp_dir": None,
    "input_dirs": [], "restart": [], "argv": [],
}

_RUN: Optional[_Run] = None
_RUN_LOCK = threading.RLock()
_ADOPT_TRIED = False


def _state_path() -> Path:
    """Where the active-run record is cached (beside the server's job dirs)."""
    root = Path(os.environ.get("LM3_SERVER_JOBS", "runs/_server_jobs"))
    if not root.is_absolute():
        root = _repo_root() / root
    return root / "active_run.json"


def _pid_alive(pid: int, create_time: Optional[float] = None) -> bool:
    """Is ``pid`` still the SAME process? ``create_time`` guards against PID reuse."""
    if not pid:
        return False
    try:
        import psutil                                  # type: ignore

        proc = psutil.Process(pid)
        if create_time is not None and abs(proc.create_time() - create_time) > 1.0:
            return False
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except Exception:                                  # noqa: BLE001 - psutil absent or pid gone
        try:
            os.kill(pid, 0)                            # signal 0 == "does this pid exist"
            return True
        except OSError:
            return False


def _adopt() -> None:
    """Re-attach to a run that outlived the server process.

    Runs are launched in their own session, so an LM3 run survives a server restart (a reload
    during development, or the desktop shell being relaunched). Without this the app would report
    "idle" while the GPUs are pinned. Any failure here is silently ignored -- adoption is a
    convenience, not a correctness requirement.
    """
    global _RUN, _ADOPT_TRIED
    _ADOPT_TRIED = True
    path = _state_path()
    if not path.is_file():
        return
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    pid = int(saved.get("pid") or 0)
    create_time = saved.get("create_time")
    if not _pid_alive(pid, create_time):
        try:
            path.unlink()
        except OSError:
            pass
        return

    run = _Run(**{k: v for k, v in saved.items()
                  if k in ("run_name", "config_path", "cwd", "output_dir", "run_dir", "db_path",
                           "log_path", "console_log", "tmp_dir", "input_dirs", "restart", "argv")})
    run.pid = pid
    run.create_time = create_time
    run.started_at = float(saved.get("started_at") or time.time())
    run.adopted = True
    run.state = "running"
    try:
        run.pgid = os.getpgid(pid)
    except OSError:
        run.pgid = None
    _RUN = run
    _bind_status(run)
    metrics.set_worker_root(pid)                       # per-worker bars must follow the RUN, not us
    threading.Thread(target=_reap, args=(run,), name="lm3-run-reaper", daemon=True).start()
    log.info("adopted running LM3 process pid=%s run=%s", pid, run.run_name)


def _bind_status(run: "_Run") -> None:
    """Tell the status side which run this is.

    Without it, progress_api has to GUESS which ledger to describe -- from the configured
    <output dir>/<project name>, or by scanning the disk -- and a run whose output lands outside
    the configured output dir (an example, a one-off --output) is invisible to both. The status
    stream, the stage bar AND the console all resolve through it, so the console can sit on a
    dead ledger reporting "waiting for log" while a run is going.

    No ``state`` is passed on purpose: that would override the ledger's own reading for as long
    as the binding lasts, freezing the run at "running" after it finished. The ledger already
    knows how it ended.

    The binding is NOT cleared when the run ends -- "the run you just did, and how it finished"
    is exactly what the app should still be showing. It lives only in this process, so the next
    server start goes back to the configured project.
    """
    try:
        from leafmachine3.server import progress_api

        progress_api.bind_run(run.db_path, run_name=run.run_name)
    except Exception:                                  # noqa: BLE001 - never break a launch over this
        log.debug("could not bind the status stream to run %r", run.run_name, exc_info=True)


def _current() -> Optional[_Run]:
    """The most recent run, adopting an orphan on first call."""
    global _RUN
    with _RUN_LOCK:
        if _RUN is None and not _ADOPT_TRIED:
            try:
                _adopt()
            except Exception:                          # noqa: BLE001
                log.debug("run adoption failed", exc_info=True)
        return _RUN


def active() -> dict:
    """The most recent run record -- ``active`` says whether it is still going.

    The key set never changes: before the first run every field is null and ``state`` is "idle";
    after a run ends the record STAYS (with ``active: false``) so the app can reveal the results of
    the run that just finished without asking a second question.
    """
    run = _current()
    if run is not None:
        return run.record()
    idle = dict(_IDLE_RECORD)                          # fresh lists: callers must not alias the template
    idle["input_dirs"], idle["restart"], idle["argv"] = [], [], []
    return idle


def is_active() -> bool:
    run = _current()
    return bool(run and run.state in ("running", "stopping") and run.alive)


def _path_or_none(value: str) -> Optional[Path]:
    run = _current()
    if run is None:
        return None
    raw = getattr(run, value, "")
    return Path(raw) if raw else None


def active_db_path() -> Optional[Path]:
    """The project SQLite of the current/last run (what the progress module tails)."""
    return _path_or_none("db_path")


def active_log_path() -> Optional[Path]:
    """``<run>/logs/lm3.log`` -- LM3's own log file, the reliable live log source."""
    return _path_or_none("log_path")


def active_console_path() -> Optional[Path]:
    """The captured stdout+stderr. Holds tracebacks that die BEFORE logging is configured."""
    return _path_or_none("console_log")


def _reap(run: _Run) -> None:
    """Wait for the run to end, then record how it ended and hand the workers table back."""
    if run.proc is not None:
        try:
            run.returncode = run.proc.wait()
        except Exception:                              # noqa: BLE001
            run.returncode = None
    else:                                              # adopted: not our child, so poll instead
        while _pid_alive(run.pid, run.create_time):
            time.sleep(1.0)

    with _RUN_LOCK:
        run.finished_at = time.time()
        if run.stopped_by_user:
            run.state = "done"
        elif run.returncode in (0, None):
            run.state = "done"
        else:
            run.state = "error"
            run.error = f"machine3 exited with code {run.returncode}"
        if run.log_fh is not None:
            try:                                       # the child is gone, so appending is safe
                run.log_fh.write(
                    f"\n[LM3 app] run finished at {_iso(run.finished_at)} "
                    f"(exit code {run.returncode}, "
                    f"{round(run.finished_at - run.started_at, 1)} s)\n")
                run.log_fh.flush()
                run.log_fh.close()
            except Exception:                          # noqa: BLE001
                pass
            run.log_fh = None
        run.persist()
    metrics.set_worker_root(os.getpid())               # stop scanning a pid that no longer exists
    log.info("LM3 run %s finished: state=%s rc=%s", run.run_name, run.state, run.returncode)


def _machine3_argv() -> list[str]:
    """The command that launches LM3.

    Prefer the console script from THIS venv (``<venv>/bin/machine3``) so the run uses the same
    interpreter and the same installed leafmachine3 as the server. ``python -m
    leafmachine3.machine3`` is the fallback; both route through ``machine3.main()``, which is the
    part that matters -- see this module's docstring for why an in-process call is wrong.
    """
    override = os.environ.get("LM3_MACHINE3_BIN")
    if override:
        return [override]
    exe = Path(sys.executable).with_name("machine3")
    if exe.is_file() and os.access(exe, os.X_OK):
        return [str(exe)]
    found = shutil.which("machine3")
    if found:
        return [found]
    return [sys.executable, "-m", "leafmachine3.machine3"]


def _normalize_restart(restart: Any) -> list[str]:
    """``None`` / ``""`` -> resume; ``"all"`` -> full rebuild; a key or list of keys -> those."""
    if restart is None or restart == "" or restart is False:
        return []
    if isinstance(restart, str):
        return ["all"] if restart.strip().lower() == "all" else [restart.strip()]
    if isinstance(restart, (list, tuple)):
        keys = [str(k).strip() for k in restart if str(k).strip()]
        return ["all"] if any(k.lower() == "all" for k in keys) else keys
    raise RunError(f"restart must be a stage key, a list of keys, or 'all' (got {restart!r})")


def start_run(config_path: Optional[str] = None, input_dir: Optional[str] = None,
              output_dir: Optional[str] = None, restart: Any = None) -> dict:
    """Launch LM3 as a subprocess and return its run record.

    ``input_dir`` / ``output_dir`` map to ``machine3 --input/--output`` (the same overrides the CLI
    takes). Everything else -- run name, temp-file location, module toggles -- lives in the YAML;
    save it through the settings module first, then start.
    """
    global _RUN
    from leafmachine3.core.config import CANONICAL_STAGE_KEYS, Config

    with _RUN_LOCK:
        current = _current()
        if current is not None and current.state in ("running", "stopping"):
            if current.alive:
                raise RunError(
                    f"a run is already active (pid {current.pid}, run '{current.run_name}') -- "
                    "stop it before starting another", status=409)
            _reap_stale(current)

        cfg_path = Path(config_path).expanduser() if config_path else default_config_path()
        if cfg_path is None:
            raise RunError(f"no {CFG_FILENAME} found -- pass config_path", status=400)
        cfg_path = cfg_path.resolve()
        if not cfg_path.is_file():
            raise RunError(f"config file not found: {cfg_path}", status=400)

        keys = _normalize_restart(restart)
        bad = [k for k in keys if k != "all" and k not in CANONICAL_STAGE_KEYS]
        if bad:
            raise RunError(f"unknown restart stage key(s): {bad}; valid keys are "
                           f"{list(CANONICAL_STAGE_KEYS)} or 'all'", status=400)

        # Resolve the run exactly the way machine3 will, so the record points at the real files
        # before the process has created any of them. Same override tree as machine3._cli_overrides.
        overrides: dict = {"project": {}}
        if input_dir:
            overrides["project"]["input"] = {"dirs": [str(input_dir)]}
        if output_dir:
            overrides["project"]["output"] = {"dir": str(output_dir)}
        if keys:
            overrides["project"]["run_mode"] = {"restart": "all" if keys == ["all"] else keys}
        try:
            cfg = Config.load(cfg_path, overrides=overrides)
            cfg.validate()                             # fail here with a 400, not in the child
        except (ValueError, FileNotFoundError) as exc:
            raise RunError(str(exc), status=400) from exc

        # The run's CWD decides where hardware_settings.yaml and every relative model path resolve
        # (hardware_setup.HW_PATH is relative), so prefer the directory that actually holds the
        # profile: the config's own directory, then the server's CWD, then the checkout root.
        cwd = next((d for d in [cfg_path.parent, *_search_roots()] if (d / HW_FILENAME).is_file()),
                   cfg_path.parent)

        run_name = str(cfg.project.run_name)
        out_dir = Path(str(cfg.project.output.dir)).expanduser()
        if not out_dir.is_absolute():
            out_dir = cwd / out_dir
        run_dir = out_dir / run_name
        logs_dir = run_dir / "logs"
        tmp_cfg = str(getattr(cfg.project.output, "tmp_dir", "auto") or "auto")
        tmp_dir = (run_dir / "_tmp_original" if tmp_cfg.lower() == "auto"
                   else Path(tmp_cfg).expanduser() / run_name / "_tmp_original")
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)   # build_dirs is exist_ok, so this is safe
        except OSError as exc:
            raise RunError(f"cannot create the run log directory {logs_dir}: {exc}", status=400) from exc

        argv = _machine3_argv() + ["--config", str(cfg_path)]
        if input_dir:
            argv += ["--input", str(input_dir)]
        if output_dir:
            argv += ["--output", str(output_dir)]
        for key in keys:
            argv += ["--restart", key]

        console = logs_dir / "console.log"
        try:
            fh = console.open("a", encoding="utf-8", errors="replace")
            fh.write(f"\n[LM3 app] {_iso(time.time())} launching: {' '.join(argv)}\n"
                     f"[LM3 app] cwd={cwd}\n")
            fh.flush()
        except OSError as exc:
            raise RunError(f"cannot open the console log {console}: {exc}", status=400) from exc

        env = os.environ.copy()
        # MUST NOT be inherited: machine3.main() skips the LD_LIBRARY_PATH re-exec when it is set,
        # which is exactly the failure mode this whole subprocess dance exists to avoid.
        env.pop("LM3_CUDA_LIBPATH_SET", None)
        env.pop("LM3_SERVER_TOKEN", None)              # the run has no business holding the secret
        env["PYTHONUNBUFFERED"] = "1"                  # so console.log tails live, not in 4 KB blocks

        try:
            proc = subprocess.Popen(
                argv, cwd=str(cwd), env=env,
                stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                close_fds=True,
                # Own session/process group: one killpg then reaches the executor's spawn workers
                # too, and the run survives a server restart instead of dying with it.
                start_new_session=True,
            )
        except OSError as exc:
            fh.close()
            raise RunError(f"could not launch {argv[0]}: {exc}", status=500) from exc

        run = _Run(run_name=run_name, config_path=str(cfg_path), cwd=str(cwd),
                   output_dir=str(out_dir), run_dir=str(run_dir),
                   db_path=str(run_dir / f"{run_name}.sqlite"),
                   log_path=str(logs_dir / "lm3.log"), console_log=str(console),
                   tmp_dir=str(tmp_dir),
                   input_dirs=[str(d) for d in (cfg.project.input.dirs or [])],
                   restart=keys, argv=argv)
        run.proc = proc
        run.log_fh = fh
        run.pid = proc.pid
        try:
            run.pgid = os.getpgid(proc.pid)
        except OSError:
            run.pgid = proc.pid
        try:
            import psutil                              # type: ignore

            run.create_time = psutil.Process(proc.pid).create_time()
        except Exception:                              # noqa: BLE001 - only used for PID-reuse checks
            run.create_time = None

        _RUN = run
        run.persist()
        _bind_status(run)
        # Every ram_delta_mb / vram_delta_mb from here reads as "what THIS run added", and the
        # worker table follows the run's spawn children instead of the server's (metrics gotcha #1).
        metrics.reset_baseline()
        metrics.set_worker_root(proc.pid)
        threading.Thread(target=_reap, args=(run,), name="lm3-run-reaper", daemon=True).start()
        log.info("started LM3 run '%s' pid=%s cwd=%s", run_name, proc.pid, cwd)
        return run.record()


def _reap_stale(run: _Run) -> None:
    """Finalize a record whose process vanished without the reaper noticing (adopted runs)."""
    run.state = "done" if run.returncode in (0, None) else "error"
    run.finished_at = run.finished_at or time.time()
    run.persist()


def _signal_group(run: _Run, sig: int) -> bool:
    """Signal the whole run: its process group first (that reaches the spawn workers)."""
    pgid = run.pgid
    # Refuse to signal our OWN group -- if start_new_session somehow failed, killpg here would take
    # the server down with the run.
    if pgid and hasattr(os, "killpg") and pgid != os.getpgrp():
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            log.debug("killpg(%s, %s) failed; falling back to the pid", pgid, sig, exc_info=True)
    try:
        os.kill(run.pid, sig)
        return True
    except OSError:
        return False


def stop_run(grace_s: float = DEFAULT_STOP_GRACE_S) -> dict:
    """Stop the active run: SIGTERM, then SIGKILL after ``grace_s``.

    LM3 is resumable -- every image is checkpointed in the project ledger and ``reclaim_running()``
    turns interrupted stages back to 'pending' on the next start -- so a stopped run restarts from
    where it was, minus the images that were mid-flight.
    """
    with _RUN_LOCK:
        run = _current()
        if run is None or run.state not in ("running", "stopping"):
            raise RunError("no active run to stop", status=409)
        if not run.alive:
            _reap_stale(run)
            return run.record()
        run.state = "stopping"
        run.stopped_by_user = True
        _signal_group(run, signal.SIGTERM)
        log.info("SIGTERM sent to LM3 run '%s' (pgid %s)", run.run_name, run.pgid)

    deadline = time.time() + max(0.0, float(grace_s))
    while time.time() < deadline:
        if not run.alive:
            break
        time.sleep(0.2)

    if run.alive:                                      # ignored SIGTERM (a wedged CUDA call, say)
        log.warning("LM3 run '%s' ignored SIGTERM after %.1fs -- sending SIGKILL", run.run_name, grace_s)
        _signal_group(run, signal.SIGKILL)
        hard_deadline = time.time() + 5.0
        while time.time() < hard_deadline and run.alive:
            time.sleep(0.1)

    # The reaper thread sets the terminal state; give it a moment so the response is not stale.
    for _ in range(20):
        if run.state not in ("running", "stopping"):
            break
        time.sleep(0.1)
    with _RUN_LOCK:
        if run.state == "stopping" and not run.alive:
            _reap_stale(run)
        return run.record()


def console_tail(offset: int = -65536, max_bytes: int = 262144) -> dict:
    """Byte-range read of the captured stdout+stderr.

    A negative ``offset`` means "the last N bytes" (the first call), and ``next_offset`` feeds the
    next poll, so the console view appends instead of refetching. This is the ONLY place a crash
    before ``start_logging()`` shows up -- config errors, a missing model export, an OOM kill.
    """
    path = active_console_path()
    if path is None or not path.is_file():
        return {"path": str(path) if path else None, "size": 0, "offset": 0,
                "next_offset": 0, "text": "", "eof": True}
    try:
        size = path.stat().st_size
        start = max(0, size + offset) if offset < 0 else min(int(offset), size)
        with path.open("rb") as fh:
            fh.seek(start)
            chunk = fh.read(max(0, int(max_bytes)))
        return {"path": str(path), "size": size, "offset": start,
                "next_offset": start + len(chunk),
                "text": chunk.decode("utf-8", "replace"),
                "eof": start + len(chunk) >= size}
    except OSError as exc:
        return {"path": str(path), "size": 0, "offset": 0, "next_offset": 0,
                "text": f"[LM3 app] could not read the console log: {exc}", "eof": True}


# --------------------------------------------------------------------------- #
# SSE
# --------------------------------------------------------------------------- #
def _frame(kind: str, **payload: Any) -> str:
    """One typed SSE envelope -- the panel switches on ``type`` (same contract as metrics.py)."""
    return "data: " + json.dumps({"type": kind, **payload}) + "\n\n"


def stream_frames(interval_s: Optional[float] = None, *, workers_every: int = 4,
                  max_seconds: float = 86400.0) -> Iterator[str]:
    """Synchronous form of the metrics stream (tests, non-async callers).

    Frames: ``hello`` (machine + the whole rolling window + the run record) once, then ``metrics``
    per new sample, ``workers`` every ``workers_every`` metrics frames, and ``run`` whenever the
    run's state or pid changes.
    """
    sampler = metrics.get_sampler()
    period = float(interval_s or sampler.interval)
    yield _frame("hello", machine=sampler.describe_machine(), history=sampler.history(), run=active())
    deadline = time.time() + max_seconds
    last_seq, tick, last_run = -1, 0, _run_key()
    while time.time() < deadline:
        point = sampler.snapshot()
        if point.get("seq") != last_seq:
            last_seq = point.get("seq")
            yield _frame("metrics", snapshot=point)
            tick += 1
            if workers_every and tick % workers_every == 0:
                yield _frame("workers", workers=sampler.workers())
        key = _run_key()
        if key != last_run:
            last_run = key
            yield _frame("run", run=active())
        time.sleep(period)


def _run_key() -> tuple:
    """Cheap change-detector for the run record (state + identity), so `run` frames are rare."""
    run = _current()
    return (run.pid, run.state, run.returncode) if run is not None else (0, "idle", None)


# --------------------------------------------------------------------------- #
# FastAPI router (the integrator mounts this; app.py is not edited here)
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
    except Exception:  # noqa: BLE001 - app.py optional / not yet initialized
        return None


def router(dependencies: Optional[list] = None) -> Any:
    """Build the ``/v1`` router for metrics + run control.

    ``fastapi`` is imported lazily, exactly like ``leafmachine3.server.app.create_app``, so a base
    install without the ``server`` extra can still import this module. Pass the app's auth
    dependency through ``dependencies`` (``[Depends(require_token)]``).

    ``dependencies`` is applied PER ROUTE rather than to the router, because ``/metrics/stream``
    cannot use a header-only guard: ``EventSource`` is unable to set an ``Authorization`` header,
    so it accepts ``?token=`` as well and checks it itself against the same secret -- exactly the
    arrangement ``progress_api.router`` and ``postprocess_api.router`` already use for their SSE
    routes. Loopback-only binding is what keeps a secret in a URL acceptable.
    """
    import asyncio

    from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import StreamingResponse

    guards = list(dependencies or [])
    api = APIRouter(prefix="/v1", tags=["metrics"])

    async def stream_token(
        token: Optional[str] = Query(default=None, description="Bearer secret, for EventSource"),
        authorization: str = Header(default=""),
    ) -> None:
        """Auth for the SSE route: ``?token=`` OR the Authorization header."""
        if not guards:                                 # mounted without auth -> nothing to check
            return
        expected = _expected_token()
        if not expected:
            return
        if token and secrets.compare_digest(str(token), expected):
            return
        if authorization and secrets.compare_digest(authorization, f"Bearer {expected}"):
            return
        raise HTTPException(status_code=401, detail="invalid or missing token")

    def _fail(exc: RunError) -> Any:
        return HTTPException(status_code=exc.status, detail=str(exc))

    # -- metrics ------------------------------------------------------------ #
    @api.get("/metrics", dependencies=guards)
    async def get_metrics(workers: bool = False) -> dict:
        """Current point + static machine facts. ``seq`` is monotonic -- use it to skip redraws."""
        point = metrics.snapshot()
        out = {"snapshot": point, "machine": metrics.describe_machine(),
               "t": point.get("t"), "seq": point.get("seq"), "ready": point.get("ready", False)}
        if workers:
            out["workers"] = metrics.workers()
        return out

    @api.get("/metrics/machine", dependencies=guards)
    async def get_machine() -> dict:
        return metrics.describe_machine()

    @api.get("/metrics/history", dependencies=guards)
    async def get_history(since: Optional[float] = None, max_points: Optional[int] = None) -> dict:
        """The rolling window, columnar. Pass ``since=t_last`` to append instead of refetching."""
        return metrics.history(since=since, max_points=max_points)

    @api.get("/metrics/workers", dependencies=guards)
    async def get_workers() -> dict:
        return metrics.workers()

    @api.get("/metrics/stream", dependencies=[Depends(stream_token)])
    async def metrics_stream(interval: Optional[float] = None, workers_every: int = 4,
                             token: Optional[str] = None) -> Any:
        """SSE at the sampler's rate (~2 Hz). One connection fills the plots AND keeps them live.

        ``token`` is for ``EventSource``, which cannot set an Authorization header. The
        ``stream_token`` dependency above is what checks it -- the app's own ``require_token``
        reads the HEADER only, so guarding this route with it would 401 every EventSource.
        """
        sampler = metrics.get_sampler()
        period = max(0.1, float(interval or sampler.interval))

        async def gen() -> Any:
            # Every read is a ring-buffer copy, so this streams without ever blocking the loop.
            #
            # STOPPING ON DISCONNECT: Starlette drives this generator and awaits ``send`` for each
            # frame; once the client is gone that send raises (ClientDisconnect / CancelledError at
            # shutdown), the exception lands here at the ``yield``, and the ``finally`` runs. The
            # heartbeat below is what bounds the detection latency -- a stream that yielded nothing
            # would never learn the socket had closed, so we always write SOMETHING every 15 s even
            # if the sampler has stalled.
            last_hb = time.monotonic()
            try:
                yield _frame("hello", machine=sampler.describe_machine(),
                             history=sampler.history(), run=active())
                last_seq, tick, last_run = -1, 0, _run_key()
                while True:
                    point = sampler.snapshot()
                    if point.get("seq") != last_seq:
                        last_seq = point.get("seq")
                        yield _frame("metrics", snapshot=point)
                        tick += 1
                        if workers_every and tick % workers_every == 0:
                            yield _frame("workers", workers=sampler.workers())
                    key = _run_key()
                    if key != last_run:
                        last_run = key
                        yield _frame("run", run=active())
                    now = time.monotonic()
                    if now - last_hb > 15.0:
                        last_hb = now
                        yield ": hb\n\n"               # comment frame: keeps idle proxies honest
                    await asyncio.sleep(period)
            finally:
                # Reached on a client disconnect, on shutdown cancellation, and on a clean return.
                # Nothing to release (the sampler is shared and stays running) -- this is the proof
                # the generator really did stop rather than leaking a task per reload.
                log.debug("metrics stream closed")

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                          "X-Accel-Buffering": "no"})

    # -- hardware ----------------------------------------------------------- #
    @api.get("/hardware/profile", dependencies=guards)
    async def get_hardware_profile() -> dict:
        """The tuned ``hardware_settings.yaml``: per-module workers, GPU sizing, spawn overhead,
        disk knee -- plus a cheap staleness check against the live machine."""
        return await run_in_threadpool(hardware_profile)

    # -- run control -------------------------------------------------------- #
    @api.post("/run/start", dependencies=guards)
    async def run_start(payload: Optional[dict] = Body(default=None)) -> dict:
        body = payload or {}
        unknown = set(body) - {"config_path", "input_dir", "output_dir", "restart"}
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown field(s): {sorted(unknown)}")
        try:
            return await run_in_threadpool(
                start_run,
                config_path=body.get("config_path"),
                input_dir=body.get("input_dir"),
                output_dir=body.get("output_dir"),
                restart=body.get("restart"),
            )
        except RunError as exc:
            raise _fail(exc) from exc

    @api.post("/run/stop", dependencies=guards)
    async def run_stop(payload: Optional[dict] = Body(default=None)) -> dict:
        grace = float((payload or {}).get("grace_s", DEFAULT_STOP_GRACE_S))
        try:
            return await run_in_threadpool(stop_run, grace)
        except RunError as exc:
            raise _fail(exc) from exc

    @api.get("/run/active", dependencies=guards)
    async def run_active() -> dict:
        return active()

    @api.get("/run/console", dependencies=guards)
    async def run_console(offset: int = -65536, max_bytes: int = 262144) -> dict:
        return await run_in_threadpool(console_tail, offset, max_bytes)

    return api


__all__ = ["RunError", "router", "hardware_profile", "hardware_path", "default_config_path",
           "active", "is_active", "active_db_path", "active_log_path", "active_console_path",
           "start_run", "stop_run", "console_tail", "stream_frames"]
