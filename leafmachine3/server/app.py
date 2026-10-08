"""leafmachine3.server.app -- the optional local LM3 server (``lm3 serve``).

A thin FastAPI shell over the SAME :func:`leafmachine3.machine3.machine3` pipeline, so
it inherits resume for free: each submitted job is one project directory with its own
SQLite ledger, and status / events / results are read straight from that ledger.

``fastapi`` / ``uvicorn`` are OPTIONAL extras. Every heavy import is deferred, so this
module imports cleanly on a base install (mock or CLI only); the FastAPI application is
built lazily by :func:`create_app` and only then requires the ``server`` extra. Jobs run
ONE at a time off the event loop (the pipeline saturates the hardware itself), while the
event loop keeps streaming progress from the project DB.

Security: bind loopback on a caller-chosen port and require a Bearer shared secret
(``LM3_SERVER_TOKEN``, auto-generated if unset) so no other local process can drive it.
"""
from __future__ import annotations

import collections
import contextlib
import hashlib
import ipaddress
import json
import logging
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

import yaml

from leafmachine3 import __version__
from leafmachine3.core import paths

log = logging.getLogger("leafmachine3.server")

_TOKEN_ENV = "LM3_SERVER_TOKEN"


# --------------------------------------------------------------------------- #
# Canonical path resolution (plan section 3.1) -- the ONE seam every server module uses
# --------------------------------------------------------------------------- #
# Every settings / hardware / postprocessing / jobs path in leafmachine3.server resolves through
# the helpers below, which are thin wrappers over leafmachine3.core.paths. They live HERE, in the
# module that already owns server startup, because the one-release legacy-variable migration has
# to warn ONCE PER PROCESS rather than once per request: that requires shared state, and app.py is
# the only module the other five may import without a cycle (it imports none of them at module
# scope -- create_app imports them lazily, long after this module is fully initialized).
#
# Nothing here is resolved at IMPORT time. The old ``DEFAULT_JOBS_ROOT`` constant froze the answer
# before a spawned child had even read LM3_DEPLOYMENT_ID; every helper below reads the environment
# at call time instead.

#: Memoized results of the legacy-alias migration, keyed by the exact (canonical, new, old) triple
#: so a deprecation warning is emitted once per process per distinct environment -- and so a
#: split-brain conflict keeps failing on every later call instead of only the first.
_LEGACY_CACHE: dict[tuple[str, str | None, str | None], str | None] = {}
_LEGACY_CONFLICTS: dict[tuple[str, str | None, str | None], str] = {}
_LEGACY_LOCK = threading.Lock()


def _legacy_value(source: Any, canonical: str, legacy: str) -> str | None:
    """Resolve one canonical/legacy pair, warning at most once per process.

    :func:`leafmachine3.core.paths.resolve_legacy_env` does the deciding (and the warning); this
    only calls it the first time a given environment is seen. A conflict is cached as its message
    and re-raised every time, because "the two variables disagree" is a startup-fatal condition,
    not a one-off notice.
    """
    new = source.get(canonical)
    old = source.get(legacy)
    key = (canonical, new, old)
    with _LEGACY_LOCK:
        if key in _LEGACY_CONFLICTS:
            raise paths.LegacyEnvConflictError(_LEGACY_CONFLICTS[key])
        if key in _LEGACY_CACHE:
            return _LEGACY_CACHE[key]
    try:
        value = paths.resolve_legacy_env(source, canonical, legacy)
    except paths.LegacyEnvConflictError as exc:
        with _LEGACY_LOCK:
            _LEGACY_CONFLICTS[key] = str(exc)
        raise
    with _LEGACY_LOCK:
        _LEGACY_CACHE[key] = value
    return value


def server_env(env: Any = None) -> Any:
    """The environment with the deprecated aliases folded into their canonical names.

    ``LM3_SETTINGS_PATH`` -> ``LM3_SETTINGS`` and ``LM3_HARDWARE_SETTINGS`` -> ``LM3_HARDWARE``,
    for one release (plan section 4, Step 1). Legacy alone is honored with a deprecation warning;
    legacy plus canonical naming the same file warns once and uses it; legacy plus canonical
    naming DIFFERENT files raises :class:`~leafmachine3.core.paths.LegacyEnvConflictError`, which
    stops server startup rather than silently picking a winner.

    Returned as a plain mapping handed to every ``paths`` call, so the resolver never sees the
    deprecated spellings at all and never re-warns from inside a per-request code path.
    """
    source = os.environ if env is None else env
    folded: dict[str, str] | None = None
    for canonical, legacy in paths.LEGACY_ENV_ALIASES.items():
        if source.get(legacy) is None:
            continue
        value = _legacy_value(source, canonical, legacy)
        if folded is None:
            folded = dict(source)
        folded.pop(legacy, None)
        if value is None:
            folded.pop(canonical, None)
        else:
            folded[canonical] = value
    return source if folded is None else folded


def check_legacy_env() -> None:
    """Fail fast on conflicting legacy variables. Called before anything else at startup.

    Raising from inside a router factory would be swallowed by the per-router ``try/except`` in
    :func:`create_app`, leaving a half-mounted API; Step 1's exit gate says conflicting legacy
    variables must STOP startup, so the check happens up front.
    """
    server_env()


def canonical_settings_path(explicit: Any = None, *, seed: bool = False) -> Path:
    """Row 1 of the section 3.1 precedence table -- the next-run ``LM3_settings.yaml``.

    ``seed`` is off by default: resolving a path must never create a file as a side effect of a
    status poll. Seeding happens once per process at startup -- in :func:`create_app` for any
    server entry point, and additionally in :func:`serve`, which also pins the result into
    ``LM3_SETTINGS``.
    """
    return paths.settings_path(explicit, env=server_env(), seed=seed)


#: Memoized canonical hardware-profile path, keyed by the resolved deployment key. The status
#: stream asks for this at 2 Hz; the answer depends only on the deployment and the machine, neither
#: of which changes inside a process.
_HARDWARE_PATH_CACHE: dict[str, Path] = {}


def canonical_hardware_path() -> Path:
    """Row 2 -- the deployment-scoped, machine-keyed hardware profile. PURE and memoized.

    This used to resolve the settings file on every call in order to perform the one-release legacy
    adopt, which made a READ endpoint copy a file: that is how two legacy profiles were written into
    a real developer's ``~/.config/lm3`` during a test run. Adoption now happens exactly once, in
    :func:`create_app`, through ``paths.migrate_legacy_hardware_profile``. Resolution writes nothing.
    """
    env = server_env()
    # The key must name every input the answer depends on -- deployment, config root and the
    # LM3_HARDWARE override -- so a deliberate environment change is never masked by the memo.
    key = "\x00".join((
        paths.deployment_key(env),
        str(paths.user_config_dir(env)),
        env.get(paths.ENV_HARDWARE, ""),
    ))
    cached = _HARDWARE_PATH_CACHE.get(key)
    if cached is None:
        cached = paths.hardware_profile_path(env=env)
        _HARDWARE_PATH_CACHE[key] = cached
    return cached


def reset_path_caches() -> None:
    """Drop the per-process path memos. For tests, and after a deliberate deployment change."""
    _HARDWARE_PATH_CACHE.clear()
    paths.reset_machine_key_cache()


def canonical_postprocess_settings_path() -> Path:
    """Row 3 -- ``<user-config>/lm3/<deployment>/postprocessing.yaml`` unless overridden."""
    return paths.postprocessing_settings_path(env=server_env())


def server_jobs_root(*, create: bool = False) -> Path:
    """Row 4 -- ``LM3_SERVER_JOBS``, else ``<user-state>/lm3/<deployment>/jobs``.

    Replaces the import-time ``DEFAULT_JOBS_ROOT`` constant. One reader, one answer: the old
    module-level constant and ``metrics_api``'s independent env read absolutized the SAME variable
    two different ways, so the server could write job dirs to one place and its active-run record
    to another.
    """
    return paths.server_jobs_root(env=server_env(), create=create)


def path_diagnostics(explicit_settings: Any = None) -> dict:
    """Flat, log-safe summary of every canonical path (startup logs and ``/healthz``).

    Never raises -- a diagnostics view that dies on a bad environment tells the user nothing about
    why (plan section 4, Step 1: "Expose resolved paths in startup logs and ``/healthz``
    diagnostics").
    """
    try:
        env = server_env()
    except paths.PathsError as exc:                  # a split-brain env must still be REPORTABLE
        return {"error": redact_token(str(exc))}
    resolved = paths.describe_resolved_paths(env=env, explicit_settings=explicit_settings, seed=False)
    # Gate 13: "redact tokens from error messages, tracebacks, and /healthz diagnostics". A path
    # here should never contain the secret -- but this block is served UNAUTHENTICATED, and a
    # resolver error string quoting an environment value is exactly how one would arrive.
    return {key: redact_token(value) for key, value in resolved.items()}


def log_resolved_paths(where: str = "startup") -> dict:
    """Log every canonical path once, at INFO, so a support log answers "which config?" outright."""
    resolved = path_diagnostics()
    log.info("LM3 %s path resolution: %s", where,
             ", ".join(f"{k}={v}" for k, v in resolved.items()))
    return resolved


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    """One submitted pipeline run: a self-contained project dir + its config path."""

    id: str
    root: Path
    cfg_path: Path
    kind: str = "pipeline"          # "pipeline" | "setup"
    state: str = "queued"           # queued | running | done | error
    error: str | None = None
    events: list = field(default_factory=list)

    @property
    def db_path(self) -> Path:
        return self.root / "run.sqlite"

    def emit(self, *args: Any, **kwargs: Any) -> None:
        """Progress sink handed to LM3_Setup (records sweep steps for ``/v1/setup/events``)."""
        self.events.append({"args": list(args), "kwargs": dict(kwargs), "t": time.time()})


class JobManager:
    """Filesystem-backed registry of jobs (one project dir each)."""

    def __init__(self, root: Path | None = None) -> None:
        # Resolved HERE, not in the signature: a default argument is evaluated at import time, so
        # the old ``root: Path = DEFAULT_JOBS_ROOT`` froze the jobs root before a spawned child had
        # read LM3_DEPLOYMENT_ID (plan section 3.1, "server jobs root").
        self.root = Path(root) if root is not None else server_jobs_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}

    # -- creation ----------------------------------------------------------- #
    def create(
        self,
        files: Optional[Iterable[tuple[str, bytes]]] = None,
        input_dir: str | None = None,
        settings: Optional[dict] = None,
    ) -> Job:
        """Stage inputs into a fresh job dir and write its ``LM3_settings.yaml``.

        ``files`` is an iterable of ``(filename, content)`` pairs (uploaded images);
        ``input_dir`` points the job at an existing folder of originals instead.
        """
        job_id = uuid.uuid4().hex[:12]
        root = self.root / job_id
        images_dir = root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        if files:
            for name, content in files:
                safe = Path(name).name or f"upload_{uuid.uuid4().hex[:6]}.jpg"
                (images_dir / safe).write_bytes(content)
            in_dirs = [str(images_dir)]
        elif input_dir:
            in_dirs = [str(Path(input_dir))]
        else:
            in_dirs = [str(images_dir)]

        cfg = self._base_settings(settings)
        cfg.setdefault("project", {})
        cfg["project"]["run_name"] = "run"
        cfg["project"].setdefault("input", {})["dirs"] = in_dirs
        cfg["project"].setdefault("output", {})["dir"] = str(root)

        cfg_path = root / "LM3_settings.yaml"
        with cfg_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)

        job = Job(id=job_id, root=root / "run", cfg_path=cfg_path)
        self._jobs[job_id] = job
        return job

    def create_setup(self) -> Job:
        """Create a lightweight job that reruns the hardware profiler."""
        job_id = uuid.uuid4().hex[:12]
        root = self.root / job_id
        root.mkdir(parents=True, exist_ok=True)
        # The canonical settings path, not a bare relative name: with a CWD-relative cfg_path the
        # profiler was handed ``cfg=None`` (app.py's ``job.cfg_path.is_file()`` check) whenever the
        # server was started from anywhere but the checkout (plan section 3.1 row 1).
        job = Job(id=job_id, root=root, cfg_path=canonical_settings_path(), kind="setup")
        self._jobs[job_id] = job
        return job

    def _base_settings(self, settings: Optional[dict]) -> dict:
        """Start from the user's ``LM3_settings.yaml`` if present, else built-in defaults."""
        if settings:
            return json.loads(json.dumps(settings))     # deep copy of a plain dict
        base = canonical_settings_path()
        if base.is_file():
            with base.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        from leafmachine3.core.config import builtin_defaults

        return builtin_defaults()

    # -- lookup ------------------------------------------------------------- #
    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def db_path(self, job_id: str) -> Path:
        return self.get(job_id).db_path

    def jobs_of_kind(self, kind: str) -> list[Job]:
        """Every registered job of one kind ("pipeline" / "setup"), newest last."""
        return [j for j in self._jobs.values() if j.kind == kind]

    def mark_running(self, job_id: str) -> None:
        self.get(job_id).state = "running"

    def mark_done(self, job_id: str) -> None:
        self.get(job_id).state = "done"

    def mark_error(self, job_id: str, msg: str) -> None:
        # Redacted HERE rather than at each of the six call sites (gate 13: "redact tokens from
        # error messages"). A job error is echoed back over /v1/jobs/{id}, replayed into the setup
        # event log and shown in the UI, and the messages are built from exception text that has
        # passed through argv and environment handling -- so one chokepoint is the only version of
        # this that stays true as call sites are added.
        job = self.get(job_id)
        job.state = "error"
        job.error = redact_token(msg)

    # -- reads over the project ledger -------------------------------------- #
    def status(self, job_id: str) -> dict:
        """Job status merged with per-stage / per-image counters from the project DB."""
        job = self.get(job_id)
        out: dict[str, Any] = {"job_id": job_id, "state": job.state, "kind": job.kind}
        if job.error:
            out["error"] = job.error
        rows = self._read(job, "SELECT stage_key, state, n_total, n_done FROM project_status "
                               "ORDER BY stage_order")
        if rows is not None:
            out["stages"] = [dict(r) for r in rows]
            current = next((r["stage_key"] for r in rows if r["state"] == "running"), None)
            out["stage"] = current
            done = self._scalar(job, "SELECT COUNT(*) FROM image_status WHERE stage_key='reporter' "
                                     "AND state='done'")
            total = self._scalar(job, "SELECT COUNT(*) FROM specimen")
            out["images_done"] = done or 0
            out["images_total"] = total or 0
        return out

    def results(self, job_id: str) -> dict:
        """Detections, CF, mask refs and grounded areas for a finished job."""
        job = self.get(job_id)
        specimens = self._read(job, "SELECT * FROM specimen")
        if specimens is None:
            return {"job_id": job_id, "state": job.state, "specimens": []}
        out_specimens: list[dict] = []
        for spec in specimens:
            sid = spec["specimen_id"]
            out_specimens.append(
                {
                    "specimen_id": sid,
                    "cf_px_per_cm": spec["cf_px_per_cm"] if "cf_px_per_cm" in spec.keys() else None,
                    "cf_source": spec["cf_source"] if "cf_source" in spec.keys() else None,
                    "archival": self._rows(job, "archival_detection", sid),
                    "plant": self._rows(job, "plant_detection", sid),
                    "leaves": self._rows(job, "leaf_segmentation", sid),
                }
            )
        return {"job_id": job_id, "state": job.state, "specimens": out_specimens}

    # -- low-level sqlite helpers (read-only) ------------------------------- #
    def _connect(self, job: Job) -> Optional[sqlite3.Connection]:
        if not job.db_path.exists():
            return None
        conn = sqlite3.connect(f"file:{job.db_path}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _read(self, job: Job, sql: str, params: tuple = ()) -> Optional[list[sqlite3.Row]]:
        conn = self._connect(job)
        if conn is None:
            return None
        try:
            return list(conn.execute(sql, params))
        except sqlite3.Error as exc:
            log.debug("read failed (%s): %s", sql, exc)
            return None
        finally:
            conn.close()

    def _scalar(self, job: Job, sql: str, params: tuple = ()) -> Optional[int]:
        rows = self._read(job, sql, params)
        if not rows:
            return None
        value = rows[0][0]
        return int(value) if value is not None else None

    def _rows(self, job: Job, table: str, sid: int) -> list[dict]:
        # detection tables carry same-class duplicate suppression -> show only the kept boxes
        keep = " AND suppressed = 0" if table in ("archival_detection", "plant_detection") else ""
        rows = self._read(job, f"SELECT * FROM {table} WHERE specimen_id = ?{keep}", (sid,))
        return [dict(r) for r in rows] if rows else []


# --------------------------------------------------------------------------- #
# Hardware setup as a subprocess (plan section 2.13) -- runtime v2 only
# --------------------------------------------------------------------------- #
# GUI hardware setup runs INSIDE the server today: ``run_in_threadpool(run_setup, ...)``, no Popen,
# no child, no process group of its own. That is irreconcilable with section 3.3, which requires
# Stop to terminate the retained root process group so a calibration tree dies as one unit -- if
# the setup root IS the server, "terminate the root process group" means killing the server along
# with every unrelated request it is serving. It is also the one exception invariant 13 had to
# carry ("the server process is never a lease holder"), and an invariant with an exception erodes.
#
# So: a dedicated ``lm3-setup`` child, a retained handle, and progress through a server-private
# append-only JSONL log -- chosen over a live channel because it survives server and UI
# disconnects, cannot fill a pipe and stall the subprocess, and can be replayed on reconnect.

#: Append-only JSONL progress, under the job dir (server-private jobs root = permitted staging).
SETUP_EVENT_LOG = "setup_events.jsonl"
#: The setup child's stdout+stderr, from the moment of Popen. Never a pipe the server may not drain.
SETUP_CONSOLE_LOG = "setup_console.log"
#: A setup child that exits with this lost the deployment lease race (section 2.3 / 3.3).
EXIT_CODE_BUSY = 75


def _lm3_setup_argv() -> list[str]:
    """The command that runs hardware setup out of process.

    Prefer the console script from THIS venv so the child uses the same interpreter and the same
    installed leafmachine3 as the server; ``python -m leafmachine3.setup.hardware_setup`` is the
    fallback. Both route through ``hardware_setup.main()``, which is the part that matters.
    """
    # At call time, like every other heavy-ish import in this module: app.py must keep importing
    # cleanly on a base install, and this function runs once per setup request.
    import shutil
    import sys

    override = os.environ.get("LM3_SETUP_BIN")
    if override:
        return [override]
    exe = Path(sys.executable).with_name("lm3-setup")
    if exe.is_file() and os.access(exe, os.X_OK):
        return [str(exe)]
    found = shutil.which("lm3-setup")
    if found:
        return [found]
    return [sys.executable, "-m", "leafmachine3.setup.hardware_setup"]


def setup_event_log_path(job: Job) -> Path:
    """Where one setup job's JSONL progress lives."""
    return Path(job.root) / SETUP_EVENT_LOG


def append_setup_event(path: Path, event: dict) -> None:
    """Append one JSON line. Append-only on purpose: a replayable log, never a mutable snapshot."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": time.time(), **event}, default=str) + "\n")
            fh.flush()
    except OSError:                                   # progress is diagnostics: never fail a run on it
        log.debug("could not append a setup event to %s", path, exc_info=True)


def read_setup_events(path: Path, *, limit: int = 5000) -> list[dict]:
    """Replay the JSONL log. A malformed line is skipped, never fatal -- the rest still describes."""
    if not path.is_file():
        return []
    out: list[dict] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            # deque(maxlen=limit) bounds the READ: only the tail is held and parsed. Trimming with
            # ``out[-limit:]`` after parsing bounded the reply but not the work, and this endpoint
            # is polled once a second for the whole of a multi-minute calibration.
            tail = collections.deque(fh, maxlen=limit)
    except OSError:
        return out
    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def launch_setup_subprocess(job: Job, *, optimize: bool = True, quick: bool = False,
                            force: bool = False, calibrate: bool = False) -> Any:
    """Start ``lm3-setup`` for ``job`` and return the retained control handle.

    Raises :class:`FileNotFoundError` when the canonical config does not exist. Section 2.13:
    "optimized or calibrated setup requires a valid canonical config ... if it cannot, the setup job
    fails with a precise message naming the missing path rather than an ``AttributeError``" --
    which is what ``run_setup(None, ...)`` produces today, because it dereferences ``cfg`` at
    ``_fingerprint(cfg)`` before any guard.
    """
    import subprocess

    from leafmachine3.core.runtime.execution import child_base_env
    from leafmachine3.server.metrics_api import _ManagedChild

    cfg_path = Path(job.cfg_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"hardware setup needs the canonical settings file, and there is none at {cfg_path}. "
            f"Save settings first, or set LM3_SETTINGS to the file you want profiled.")

    argv = _lm3_setup_argv() + ["--config", str(cfg_path)]
    if optimize:
        argv.append("--optimize")
    if quick:
        argv.append("--quick")
    if force:
        argv.append("--force")
    if calibrate:
        argv.append("--calibrate")

    root = Path(job.root)
    root.mkdir(parents=True, exist_ok=True)
    console = root / SETUP_CONSOLE_LOG
    events = setup_event_log_path(job)
    fh = console.open("a", encoding="utf-8", errors="replace")

    # Filtered, never a wholesale copy: os.environ carries this server's own LM3_STATUS_FD and any
    # lease variables, and a descriptor NUMBER means something different in the child.
    env = child_base_env(extra={"PYTHONUNBUFFERED": "1"})
    env.pop("LM3_CUDA_LIBPATH_SET", None)
    env.pop("LM3_SERVER_TOKEN", None)

    try:
        proc = subprocess.Popen(
            argv, cwd=str(cfg_path.parent), env=env,
            stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, close_fds=True,
            # Its own group, so Stop kills the setup AND the calibration child that inherited its
            # lease -- as one unit -- while the server keeps serving (section 2.13).
            start_new_session=True,
        )
    except OSError:
        fh.close()
        raise

    child = _ManagedChild(proc, kind="hardware_setup", console_path=console, log_fh=fh)
    append_setup_event(events, {"type": "started", "pid": child.pid, "pgid": child.pgid,
                                "argv": argv, "config": str(cfg_path),
                                "console": str(console)})
    return child


def watch_setup_subprocess(job: Job, child: Any, manager: "JobManager") -> None:
    """Tail the child's console into the JSONL log, then record how it ended.

    Runs on a daemon thread. It reads the console FILE rather than a pipe, so the child can never
    block on a reader that went away -- which is the same reason section 2.4 keeps the run's console
    off a pipe.
    """
    events = setup_event_log_path(job)
    console = child.console_path
    offset = 0

    def drain() -> None:
        nonlocal offset
        if console is None or not Path(console).is_file():
            return
        try:
            with Path(console).open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(offset)
                # readline(), NOT ``for line in fh``: iterating a TextIOWrapper sets its read-ahead
                # flag, after which tell() raises OSError("telling position disabled by next()
                # call"). That exception is swallowed by the handler below, so the iterator form
                # silently drained ONE line and then re-appended it every pass forever.
                while True:
                    line = fh.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):       # a partial line: leave it for the next pass
                        break
                    text = line.rstrip("\n")
                    if text.strip():
                        append_setup_event(events, {"type": "log", "line": text})
                    # Strictly AFTER the append, and only for a newline-terminated line, so a torn
                    # write is re-read next pass rather than lost. tell() is legal here because
                    # readline() never engages the iterator read-ahead.
                    offset = fh.tell()
        except OSError:
            log.debug("could not drain the setup console %s", console, exc_info=True)

    while child.alive:
        drain()
        time.sleep(0.5)
    with contextlib.suppress(Exception):
        child.proc.wait(timeout=5)
    drain()

    rc = child.returncode
    child.close_log(f"[LM3 app] hardware setup finished (exit code {rc})")
    if getattr(child, "stopped_by_user", False):
        append_setup_event(events, {"type": "state", "state": "stopped", "returncode": rc})
        manager.mark_error(job.id, "hardware setup was stopped")
        return
    if rc == 0:
        append_setup_event(events, {"type": "state", "state": "done", "returncode": rc})
        manager.mark_done(job.id)
        return
    if rc == EXIT_CODE_BUSY:
        message = ("another root activity holds this deployment, so hardware setup did not run "
                   f"(exit {EXIT_CODE_BUSY})")
    else:
        message = f"hardware setup exited with code {rc}"
    append_setup_event(events, {"type": "state", "state": "error", "returncode": rc,
                                "error": message})
    manager.mark_error(job.id, message)


#: One entry per legacy ``/v1/jobs`` route that has actually been called in this process.
#:
#: THE AUDIT INSTRUMENT for section 4 Step 7. The step asks for the actual EXTERNAL consumers of
#: ``/v1/jobs``, and a source grep cannot answer that -- it can only prove what the tree itself
#: calls, which is nothing (Appendix B C2: ``getJob``, ``getJobResults`` and ``streamJobEvents``
#: are DEFINED in ``ui/js/api.js`` and called from nowhere; no Python, test, script or Electron
#: file references the routes). Whether some operator's script does is unknowable from here, so the
#: server records the evidence instead of guessing: the first use of each route in a process logs a
#: warning naming the caller, and that line in a user's log is what turns "probably nobody" into a
#: fact before anything is deleted. See docs/DEPRECATIONS.md.
_LEGACY_JOBS_SEEN: set = set()


def note_legacy_jobs_use(route: str, user_agent: str = "") -> bool:
    """Record one use of a legacy ``/v1/jobs`` route. ``True`` when this was the first.

    Once per route per process: these are polled endpoints (``/events`` is an SSE stream and
    ``/{jid}`` is what a poller hits every second), so warning on every call would bury the log it
    is meant to inform. The User-Agent is included because it is the only thing that distinguishes
    an operator's ``curl`` from a browser, and it is NEVER a secret -- unlike the query string,
    which on the SSE routes can carry ``?token=`` and is deliberately not logged.
    """
    if route in _LEGACY_JOBS_SEEN:
        return False
    _LEGACY_JOBS_SEEN.add(route)
    log.warning(
        "legacy %s was called by %s. This upload-and-queue API is a removal CANDIDATE (see "
        "docs/DEPRECATIONS.md): it has no in-tree consumer, and POST /v1/run/start plus "
        "GET /v1/runtime cover everything except multipart upload. If you depend on it, say so "
        "before it is scheduled for removal.",
        route, user_agent.strip() or "an unidentified client")
    return True


def _run_job_as_subprocess(job: Job) -> None:
    """Run a legacy ``/v1/jobs`` submission through the SAME managed launch the Run button uses.

    ``app.py``'s worker calls ``machine3()`` in-process. The moment Step 3 wraps ``machine3()`` in
    a lease, that line makes the SERVER a lease holder -- which invariant 13 forbids without
    exception, and which would let a queued job take the deployment out from under a CLI run with
    no handshake and no refusal. Step 3's exit gate names "legacy jobs" as one of the four entry
    points that must refuse a second root activity, so under the flag the job goes out of process
    and inherits the section 2.4 handshake, the 409, and the retained handle for free.
    """
    from leafmachine3.server import metrics_api

    metrics_api.start_run(config_path=str(job.cfg_path))   # RunError(409) when the deployment is busy
    while metrics_api.is_active():
        time.sleep(0.5)
    record = metrics_api.active()
    if record.get("state") == "error":
        raise RuntimeError(record.get("error") or "the job process failed")


# --------------------------------------------------------------------------- #
# Progress tailing (used by the SSE endpoint)
# --------------------------------------------------------------------------- #
TAIL_POLL_S = 1.0                      # seconds between job-progress snapshots
TAIL_MAX_POLLS = 3600                  # ~1 h, then the stream ends and the client reconnects


def tail_progress(db_path: Path, *, poll_s: float = TAIL_POLL_S,
                  max_polls: int = TAIL_MAX_POLLS) -> Iterator[str]:
    """Yield JSON progress snapshots from a job's project DB until it drains or stalls.

    Emits Server-Sent-Events ``data:`` frames. Terminates when the reporter stage is
    ``done`` (or errored) or after ``max_polls`` idle iterations.

    SYNCHRONOUS by design -- for tests, curl-alikes and non-async callers. It blocks on
    ``time.sleep`` and on SQLite, so it must NEVER be driven from a coroutine: doing that runs
    the sleep ON the event loop and freezes every other client. The ``/v1/jobs/{jid}/events``
    route runs the same sequence with ``asyncio.sleep`` and the read pushed to a thread.
    """
    db_path = Path(db_path)
    for _ in range(max_polls):
        snapshot = _progress_snapshot(db_path)
        yield f"data: {json.dumps(snapshot)}\n\n"
        state = snapshot.get("reporter_state")
        if state in {"done", "error"}:
            break
        time.sleep(poll_s)


def _progress_snapshot(db_path: Path) -> dict:
    if not db_path.exists():
        return {"ready": False}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return {"ready": False}
    try:
        stages = [dict(r) for r in conn.execute(
            "SELECT stage_key, state, n_total, n_done FROM project_status ORDER BY stage_order")]
        reporter = next((s for s in stages if s["stage_key"] == "reporter"), {})
        return {"ready": True, "stages": stages, "reporter_state": reporter.get("state")}
    except sqlite3.Error:
        return {"ready": False}
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def _server_token() -> str:
    """Return the shared Bearer secret, generating and exporting one if unset.

    **The raw token is never logged, at any level** (plan section 2.11, gate 13). It used to be
    printed at WARNING, which on a cluster lands the bearer token in Slurm job output and retained
    container logs, where it long outlives the allocation. What gets logged instead is WHERE to
    find it, which is all a user actually needs.
    """
    token = os.environ.get(_TOKEN_ENV)
    if not token:
        token = secrets.token_urlsafe(24)
        os.environ[_TOKEN_ENV] = token
        try:
            where = f"it will be written to {connection_private_path(server_env())}"
        except (paths.PathsError, OSError):
            where = "set it explicitly to choose your own"
        log.warning(
            "generated an LM3 server token for this process; %s. Set %s to override. "
            "The token itself is deliberately not logged.", where, _TOKEN_ENV)
    return token


def redact_token(text: str) -> str:
    """Replace the live bearer token wherever it appears in ``text``.

    A last line of defense for error messages, tracebacks and diagnostics: gate 13 says the raw
    token appears in NO log, and a string that merely passes near an exception handler is exactly
    how a secret escapes.
    """
    token = os.environ.get(_TOKEN_ENV)
    if not token or not text:
        return text
    return text.replace(token, "***redacted***")


#: A ``token=`` query parameter inside a request line. Redaction goes by PATTERN as well as by the
#: literal secret because a user-supplied ``LM3_SERVER_TOKEN`` may contain characters the client
#: percent-encodes on the way out (``api.js`` builds the SSE URL through ``URLSearchParams``), and
#: what lands in the log is then an ENCODING of the secret that a literal replace sails past.
_ACCESS_LOG_TOKEN_QUERY = re.compile(r"(?i)([?&]token=)[^&\s]*")


class _AccessLogTokenRedactor(logging.Filter):
    """Scrub ``?token=`` out of uvicorn's access log (plan section 2.11, gate 13).

    Two clients cannot send an ``Authorization`` header and so must carry the secret on the query
    string: the ``EventSource`` streams (``api.js``; ``progress_api.py`` accepts ``token`` as a
    ``Query`` for exactly that reason) and Electron's first ``win.loadURL(.../?token=...)``.
    uvicorn's ``AccessFormatter`` builds the request line from ``get_path_with_query_string()``,
    which appends the RAW query, so without this filter every SSE connect -- one per stream, again
    on every reconnect -- and every window load writes the bearer token to the server's stdout.
    Under ``lm3 serve`` in an allocation that stdout IS the Slurm job output file section 2.11
    names; under Electron it is the stderr pipe ``app/main.js`` tails into an error dialog. Gate 13
    is absolute: the raw token appears in NO log.

    It rewrites ``record.args``, never ``record.msg``: uvicorn logs the tuple
    ``(client_addr, method, full_path, http_version, status)`` against a fixed format string and the
    formatter rebuilds the request line from those args, so patching the message would change
    nothing. Every step is guarded -- a filter that raises breaks logging itself, so an unexpected
    arg shape from a future uvicorn must degrade to a no-op, not to an exception on every request.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            args = record.args
            # uvicorn's access record is always a 5-tuple with the path third; anything else is a
            # shape we do not understand and therefore must not rewrite.
            if not isinstance(args, tuple) or len(args) < 3 or not isinstance(args[2], str):
                return True
            scrubbed = redact_token(_ACCESS_LOG_TOKEN_QUERY.sub(r"\1***redacted***", args[2]))
            if scrubbed != args[2]:
                record.args = args[:2] + (scrubbed,) + args[3:]
        except Exception:  # noqa: BLE001 - logging must never raise out of a filter
            return True
        return True


def install_access_log_redaction(logger_name: str = "uvicorn.access") -> bool:
    """Attach :class:`_AccessLogTokenRedactor` to uvicorn's access logger. Idempotent.

    This belongs in :func:`create_app`, NOT in :func:`serve`. Electron spawns the production server
    as ``python -m uvicorn leafmachine3.server.app:create_app --factory`` (``app/main.js``), which
    never enters ``serve()``, so a fix confined there would cover only ``lm3 serve``. The ordering
    works out on both entry points: uvicorn calls ``Config.configure_logging()`` from
    ``Config.__init__``, before ``Config.load()`` imports the app factory, and ``dictConfig`` removes
    a logger's HANDLERS but leaves its FILTERS in place -- so a filter installed here survives
    uvicorn's own logging configuration whichever order the two happen in.

    Returns True when it installed the filter, False when one was already present (create_app is
    called more than once per process by the tests, and duplicate filters would each rewrite).
    """
    logger = logging.getLogger(logger_name)
    if any(isinstance(existing, _AccessLogTokenRedactor) for existing in logger.filters):
        return False
    logger.addFilter(_AccessLogTokenRedactor())
    return True


#: Host names that mean "this machine" for the purposes of handing over the shared secret.
_LOOPBACK_HOST_NAMES = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def _client_is_loopback(client_host: str | None) -> bool:
    """True when the peer address is a loopback address (127.0.0.0/8 or ``::1``).

    ``lm3 serve --host`` will happily bind a routable interface; this is what keeps the token
    out of a response to anyone who is not on this machine.
    """
    try:
        return ipaddress.ip_address((client_host or "").strip()).is_loopback
    except ValueError:
        return False


def _host_header_is_loopback(header: str | None) -> bool:
    """True when the ``Host:`` header names loopback.

    This is the DNS-rebinding guard. A page served from ``evil.test`` that has rebound that
    name to 127.0.0.1 is same-origin with us as far as the browser is concerned, so it can read
    our responses -- but it still sends ``Host: evil.test``, so it never gets the secret.
    """
    raw = (header or "").strip().lower()
    if not raw:
        return False
    if raw.startswith("["):                              # bracketed IPv6, e.g. [::1]:8765
        name = raw.split("]", 1)[0] + "]"
    else:                                                # host[:port]; bare IPv6 has many colons
        name = raw.rsplit(":", 1)[0] if raw.count(":") == 1 else raw
    return name in _LOOPBACK_HOST_NAMES


# --------------------------------------------------------------------------- #
# FastAPI application (built lazily; requires the `server` extra)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Shutdown
# --------------------------------------------------------------------------- #
#: Set by a parent that owns this server's lifetime (the desktop shell). See _start_owner_watchdog.
_OWNER_ENV = "LM3_OWNER_PID"


def _pid_alive(pid: int) -> bool:
    """True when a process with this pid is still RUNNING.

    A zombie does not count. Signal 0 alone cannot tell the difference -- a process that has exited
    but has not been reaped by its parent still accepts it -- and "the owner exited, its parent has
    not gotten around to reaping it" is exactly the moment this needs to report the owner gone.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                  # it exists, it just is not ours to signal
        return True
    except OSError:
        return False
    try:                                     # Linux: field 3 of /proc/<pid>/stat is the state
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        return stat.rsplit(") ", 1)[1].split(" ", 1)[0] != "Z"
    except (OSError, IndexError):
        return True                          # no procfs (or it vanished mid-read): trust signal 0


def _hard_exit_soon(delay_s: float = 0.25, code: int = 0) -> None:
    """Exit the whole process shortly, deliberately bypassing uvicorn's graceful shutdown.

    Graceful shutdown waits for open connections to drain, and this UI holds THREE SSE streams
    open (status, logs, metrics) for as long as a page is alive -- so a plain SIGTERM can leave
    the server parked in "waiting for connections to close" indefinitely. That is precisely how
    an orphaned server outlives the desktop shell, keeps the port, keeps answering /healthz, and
    then gets silently adopted by the next launch (running whatever code it started with). So the
    exit here is unconditional: answer the request, then go.

    Skipping cleanup costs nothing. Job state is filesystem-backed, each run's ledger is written
    by the RUN's own process, and a run is started with ``start_new_session`` precisely so it
    outlives this one.
    """
    def _bye() -> None:
        time.sleep(max(0.0, delay_s))        # long enough for the response to reach the client
        # os._exit skips the lifespan shutdown, so the ONE piece of cleanup that matters happens
        # here: a connection descriptor left behind advertises a token and a port that are about to
        # stop existing, and the next client would authenticate against a dead server. It is
        # removed only when the instance ID is still ours (section 2.12).
        with contextlib.suppress(Exception):
            remove_connection_descriptor()
        os._exit(code)

    threading.Thread(target=_bye, name="lm3-shutdown", daemon=True).start()


def _start_owner_watchdog() -> None:
    """Exit when the process that started us dies (opt-in, via ``LM3_OWNER_PID``).

    The desktop shell asks for a clean shutdown when it quits, but it cannot ask for anything if
    it is SIGKILLed or crashes -- and the server it left behind is exactly the orphan described in
    :func:`_hard_exit_soon`. This is the backstop for that path.

    Opt-in by design: only a parent that means to own our lifetime sets the variable, so a
    hand-started ``lm3 serve`` (or one launched with nohup/setsid for browser testing) is never
    touched by this.
    """
    raw = (os.environ.get(_OWNER_ENV) or "").strip()
    if not raw:
        return
    try:
        owner = int(raw)
    except ValueError:
        log.warning("ignoring a non-numeric %s=%r", _OWNER_ENV, raw)
        return
    if owner <= 1:                           # 1 is init -- it outlives everything, so nothing to watch
        return

    if os.getppid() == owner:
        # We are the owner's direct child, so the kernel reparents us the instant it dies. Watching
        # for that is immune to the pid reuse that a bare "does pid N still exist" poll can trip on.
        def gone() -> bool:
            return os.getppid() != owner
    else:
        def gone() -> bool:
            return not _pid_alive(owner)

    def _watch() -> None:
        while True:
            time.sleep(2.0)
            if gone():
                log.warning("owner pid %s is gone -- stopping the LM3 server", owner)
                with contextlib.suppress(Exception):
                    remove_connection_descriptor()
                os._exit(0)

    threading.Thread(target=_watch, name="lm3-owner-watchdog", daemon=True).start()


# --------------------------------------------------------------------------- #
# Server identity and the connection descriptor (sections 2.11, 2.12; Step 5b)
# --------------------------------------------------------------------------- #
#: What this service calls itself on the wire. A client that finds ANY other value (or no
#: ``service`` key at all) on the configured port is talking to an unrelated service and must
#: refuse to attach -- section 2.11 makes that a NAMED error, distinct from "a valid LM3 server
#: for a different deployment is here".
SERVICE_NAME = "leafmachine3"

#: Wire version of the identity fields in ``/healthz`` and of ``connection.private.json``. Bumped
#: only when an existing field changes MEANING; adding an optional field does not bump it. It is
#: deliberately separate from ``__version__`` (which moves with releases) and from the runtime
#: record's schema version (which moves with the on-disk registry).
HEALTH_PROTOCOL_VERSION = 1
CONNECTION_SCHEMA_VERSION = 1

#: A parent that SPAWNS this server states the instance ID it is going to verify. Section 2.11's
#: "expected-instance-ID handshake for spawned servers" exists because a shell that loses a
#: startup race otherwise cannot tell its own server from one that already owned the port -- both
#: answer 200. With this set, the shell attaches only to the instance it named.
ENV_INSTANCE_ID = "LM3_INSTANCE_ID"

#: The address uvicorn was actually told to bind. The application object never sees the command
#: line (Electron runs ``uvicorn ... --host --port`` directly), so the launcher states it here and
#: :func:`serve` sets it for ``lm3 serve``. Only ``connection.private.json`` consumes it.
ENV_BIND_HOST = "LM3_BIND_HOST"
ENV_BIND_PORT = "LM3_BIND_PORT"

#: Accepted shape of an externally supplied instance ID. Narrow on purpose: the value reaches a
#: JSON body and a log line, and a caller that can put arbitrary text there can forge either.
_INSTANCE_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{8,64}\Z")

#: Used only when the ``server`` extra is absent, so ``metrics_api`` cannot be imported.
_FALLBACK_INSTANCE_ID = uuid.uuid4().hex
_INSTANCE_ID: str | None = None
_INSTANCE_LOCK = threading.Lock()


def server_instance_id() -> str:
    """This server process's instance ID. Stable for the life of the process, never persisted.

    Three sources, in this order, and the order is the whole point:

    1. ``LM3_INSTANCE_ID`` from the parent that spawned us (section 2.11). The shell mints the
       value, passes it down, and then refuses to attach to anything on the port that answers with
       a different one -- which is how "my server came up" is distinguished from "someone else's
       server already had the port".
    2. ``metrics_api.SERVER_INSTANCE_ID``, so the identity ``/healthz`` publishes is the SAME value
       control authority is decided against (section 2.5). Two IDs for one process would let a
       client verify one thing while the server enforced another.
    3. A local uuid4, for a base install where the ``server`` extra (and therefore ``metrics_api``)
       is not importable at all.

    It is an IDENTITY, not a capability: knowing it authorizes nothing (invariant 12).
    """
    global _INSTANCE_ID
    with _INSTANCE_LOCK:
        if _INSTANCE_ID is not None:
            return _INSTANCE_ID
        supplied = (os.environ.get(ENV_INSTANCE_ID) or "").strip()
        if supplied:
            if _INSTANCE_ID_RE.match(supplied):
                _INSTANCE_ID = supplied
                return _INSTANCE_ID
            log.warning("ignoring a malformed %s (expected 8-64 of [A-Za-z0-9_-])", ENV_INSTANCE_ID)
        try:
            from leafmachine3.server.metrics_api import SERVER_INSTANCE_ID  # noqa: PLC0415
        except Exception:  # noqa: BLE001 - a base install has no server extra
            _INSTANCE_ID = _FALLBACK_INSTANCE_ID
        else:
            _INSTANCE_ID = str(SERVER_INSTANCE_ID) or _FALLBACK_INSTANCE_ID
        return _INSTANCE_ID


def reset_instance_id_cache() -> None:
    """Forget the memoized instance ID. For tests only.

    The ID is deliberately minted once per PROCESS -- an identity that changed under a client would
    defeat the section 2.11 handshake it exists for -- so a test that wants to observe a DIFFERENT
    server (a restart, a supplied ``LM3_INSTANCE_ID``) has to say so explicitly rather than get it
    by accident.
    """
    global _INSTANCE_ID
    with _INSTANCE_LOCK:
        _INSTANCE_ID = None


def ownership_mode() -> str:
    """``owned`` / ``orphaned`` / ``independent`` -- who claims this server's lifetime.

    DESCRIPTIVE, never an authorization. Invariant 12 forbids deriving control from a response
    body, and Electron stops a server only through its own retained child handle. The field exists
    so a person (and a support log) can see whether the thing on this port belongs to a desktop
    shell or was started by hand with ``lm3 serve`` -- which is exactly the question a user asks
    when a window will not attach.

    ``orphaned`` is reported rather than hidden: a server whose owner has died is still serving,
    and its own watchdog is about to stop it, so a client that sees this should expect the port to
    go quiet shortly rather than treat the server as durable.
    """
    raw = (os.environ.get(_OWNER_ENV) or "").strip()
    if not raw:
        return "independent"
    try:
        owner = int(raw)
    except ValueError:
        return "independent"
    if owner <= 1:                              # 1 is init; it outlives everything
        return "independent"
    return "owned" if _pid_alive(owner) else "orphaned"


def deployment_identity(env: "Mapping[str, str] | None" = None) -> dict:
    """``{"deployment_id": raw, "deployment_key": canonical, "error": str}``. Never raises.

    The canonical key is the value every client verifies (section 2.1), so a resolution failure has
    to be REPORTABLE rather than fatal -- an empty key with an error string tells a shell "this
    server cannot name its deployment", which is a refusal it can act on. A silently absent field
    would look like an old server instead.
    """
    out = {"deployment_id": "", "deployment_key": "", "error": ""}
    try:
        environ = dict(os.environ) if env is None else dict(env)
        out["deployment_id"] = paths.raw_deployment_id(environ)
        out["deployment_key"] = paths.deployment_key(environ)
    except paths.PathsError as exc:
        out["error"] = str(exc)
    return out


def bind_address(env: "Mapping[str, str] | None" = None) -> tuple[str, int]:
    """The address this server was told to bind, for ``connection.private.json`` only.

    ``LM3_BIND_HOST`` / ``LM3_BIND_PORT`` are what the launcher states; the port otherwise comes
    from :func:`paths.resolve_port`, which is section 2.1's rule (8765 for the default deployment,
    an explicit ``LM3_PORT`` for any named one). A named deployment with no port is not fatal HERE
    -- refusing to publish a descriptor because a port could not be inferred would be a worse
    failure than publishing the default -- so the fallback is taken and logged.
    """
    environ = dict(os.environ) if env is None else dict(env)
    host = (environ.get(ENV_BIND_HOST) or environ.get("LM3_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    raw = (environ.get(ENV_BIND_PORT) or "").strip()
    if raw:
        try:
            return host, int(raw)
        except ValueError:
            log.warning("ignoring a non-numeric %s=%r", ENV_BIND_PORT, raw)
    try:
        return host, paths.resolve_port(environ)
    except paths.PathsError as exc:
        log.debug("falling back to port %s for the connection descriptor: %s", paths.DEFAULT_PORT, exc)
        return host, paths.DEFAULT_PORT


def connection_private_path(env: "Mapping[str, str] | None" = None, *, create: bool = False) -> Path:
    """``<deployment runtime dir>/connection.private.json`` (section 2.12). May raise PathsError."""
    environ = dict(os.environ) if env is None else dict(env)
    directory = paths.deployment_runtime_dir(env=environ, create=create)
    return directory / paths.CONNECTION_PRIVATE_FILENAME


def _restrict_to_current_user(path: Path, *, os_name: str | None = None,
                              runner: "Any" = None) -> bool:
    """Windows: replace an inherited ACL with one granting the current user only.

    Section 2.12 says so explicitly -- "POSIX mode bits are not a substitute there". The 0o600 the
    temporary file was created with is largely cosmetic on Windows, so the ACL is reset before the
    file is renamed into place. ``icacls`` is used rather than ``win32security`` because pywin32 is
    not a dependency of LM3 and a missing optional import must never be what decides whether a
    bearer token is world-readable.

    IMPLEMENTED BUT NOT NATIVELY VALIDATED (section 1.1): the qualification target is Linux, and
    the Windows branch is exercised on Linux through the injected ``os_name``/``runner`` seams.
    Returns True when nothing needed doing.
    """
    if (os_name or os.name) != "nt":
        return True
    user = (os.environ.get("USERNAME") or "").strip()
    if not user:
        log.warning("USERNAME is unset, so %s keeps its inherited ACL", path.name)
        return False
    domain = (os.environ.get("USERDOMAIN") or "").strip()
    principal = f"{domain}\\{user}" if domain else user
    cmd = ["icacls", str(path), "/inheritance:r", "/grant:r", f"{principal}:(F)"]
    try:
        if runner is not None:
            code = int(runner(cmd))
        else:
            import subprocess  # noqa: PLC0415 - Windows-only, kept out of the import path

            code = subprocess.run(cmd, capture_output=True, timeout=20, check=False).returncode
    except Exception as exc:  # noqa: BLE001 - never let ACL tightening break startup
        log.warning("could not restrict %s to the current user: %s", path.name, exc)
        return False
    if code != 0:
        log.warning("icacls refused to restrict %s (exit %s)", path.name, code)
        return False
    return True


def _fsync_dir(directory: Path) -> None:
    """fsync a directory so a rename into it is durable. A no-op where the platform refuses."""
    try:
        fd = os.open(str(directory), getattr(os, "O_DIRECTORY", os.O_RDONLY))
    except OSError:
        return                                   # Windows has no directory fsync; nothing to do
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_connection_descriptor(*, host: str | None = None, port: int | None = None,
                                token: str | None = None,
                                env: "Mapping[str, str] | None" = None,
                                _acl: "Any" = None) -> Path | None:
    """Publish this server's ``connection.private.json``. Returns the path, or None on failure.

    Section 2.12's procedure, exactly, and each step is load-bearing:

    1. a UNIQUE temporary sibling, ``O_CREAT|O_EXCL|O_WRONLY`` at 0o600 -- user-only from its first
       byte. Writing it broadly and ``chmod``-ing afterward leaves a window in which the token is
       world-readable, and a window is all an attacker needs;
    2. write and ``fsync`` it;
    3. ``os.replace`` onto the final name -- exclusive-create AT the final name would work exactly
       once, and a restarted server could then never publish its new token and instance ID;
    4. ``fsync`` the directory, so the rename itself survives a node dying;
    5. deletion is :func:`remove_connection_descriptor`, which checks the instance ID first.

    NEVER raises: a server that cannot publish a descriptor still serves, and the browser bootstrap
    of section 2.11 is a complete authentication path on its own.
    """
    try:
        directory = connection_private_path(env, create=True).parent
    except (paths.PathsError, OSError) as exc:
        log.warning("could not create the deployment runtime directory for %s: %s",
                    paths.CONNECTION_PRIVATE_FILENAME, exc)
        return None

    resolved_host, resolved_port = bind_address(env)
    identity = deployment_identity(env)
    payload = {
        "schema_version": CONNECTION_SCHEMA_VERSION,
        "service": SERVICE_NAME,
        "protocol_version": HEALTH_PROTOCOL_VERSION,
        "version": __version__,
        "instance_id": server_instance_id(),
        "deployment_id": identity["deployment_id"],
        "deployment_key": identity["deployment_key"],
        "host": host if host is not None else resolved_host,
        "port": int(port) if port is not None else resolved_port,
        "pid": os.getpid(),
        "token": token if token is not None else _server_token(),
        "created_at": time.time(),
    }
    payload["base_url"] = f"http://{payload['host']}:{payload['port']}"

    final = directory / paths.CONNECTION_PRIVATE_FILENAME
    tmp = directory / f".{paths.CONNECTION_PRIVATE_FILENAME}.{uuid.uuid4().hex}.tmp"
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        _restrict_to_current_user(tmp, runner=_acl)
        os.replace(str(tmp), str(final))
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.unlink(str(tmp))
        log.warning("could not publish %s: %s", final, exc)
        return None
    _fsync_dir(directory)
    # The PATH, never the secret (gate 13). This one line is what a user needs in order to find
    # their token; the token itself is deliberately absent from every log LM3 writes.
    log.info("published the LM3 connection descriptor at %s (instance %s)",
             final, payload["instance_id"])
    return final


def read_connection_descriptor(path: "Path | None" = None, *,
                               env: "Mapping[str, str] | None" = None) -> dict | None:
    """Read this deployment's private descriptor. Returns None when absent or unreadable."""
    try:
        target = Path(path) if path is not None else connection_private_path(env)
    except (paths.PathsError, OSError):
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def remove_connection_descriptor(*, instance_id: str | None = None,
                                 env: "Mapping[str, str] | None" = None) -> bool:
    """Delete the descriptor ONLY when its instance ID is still ours (section 2.12, step 5).

    The check is the whole point. Shutdown is not instantaneous and a user restarting the server is
    the common case, so an unconditional unlink lets a dying predecessor delete the descriptor its
    SUCCESSOR just published -- leaving a live server no client can authenticate against. A stale
    descriptor from a crashed server is the lesser problem: the next start replaces it atomically.
    """
    try:
        target = connection_private_path(env)
    except (paths.PathsError, OSError):
        return False
    payload = read_connection_descriptor(target)
    if payload is None:
        return False
    mine = instance_id if instance_id is not None else server_instance_id()
    if str(payload.get("instance_id") or "") != mine:
        log.debug("leaving %s alone: it belongs to instance %r, not %r",
                  target.name, payload.get("instance_id"), mine)
        return False
    try:
        os.unlink(str(target))
    except OSError as exc:
        log.debug("could not remove %s: %s", target, exc)
        return False
    _fsync_dir(target.parent)
    return True


def create_app(jobs: JobManager | None = None) -> Any:
    """Build and return the FastAPI application.

    Imported lazily so the base install (no ``fastapi``) can still import this module.
    """
    # FIRST, before anything can serve a request: keep the bearer token out of uvicorn's access
    # log (gate 13). Installed here rather than in serve() because the Electron shell spawns
    # ``uvicorn ...:create_app --factory`` and never reaches serve().
    install_access_log_redaction()

    import asyncio
    from contextlib import asynccontextmanager

    from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import StreamingResponse

    # ``from __future__ import annotations`` (top of this file) stringifies every annotation, and
    # FastAPI resolves those strings against the handler's MODULE globals -- where a name imported
    # inside create_app does not exist. Publishing UploadFile is what lets the lazy import (needed
    # so a base install can still import this module) coexist with ``files: list[UploadFile]``;
    # without it app.openapi() raises PydanticUserError and /openapi.json + /docs 500 for the
    # WHOLE app. ``Request`` gets the same treatment further down, beside the UI route.
    globals().setdefault("UploadFile", UploadFile)
    globals().setdefault("Request", Request)          # same trick, for /v1/shutdown below

    # BEFORE anything else: conflicting legacy variables must stop startup (plan section 4,
    # Step 1's exit gate). It cannot live in a router factory -- the per-router try/except below
    # would swallow it into a silently half-mounted API.
    check_legacy_env()

    # section 3.1 precedence table row 1, on-miss column: "seed 4 from the packaged template, log
    # the path loudly, continue", and the bullet "Missing settings seed from the packaged template
    # ... so a first-run install works". serve() is not the only entry point -- the Electron shell
    # spawns ``uvicorn leafmachine3.server.app:create_app --factory`` (app/main.js:162-165) and
    # never reaches it, so a first-run GUI install would otherwise meet a 400 "no LM3_settings.yaml
    # found" instead of a seeded file. Once per process at startup is not "a file created as a side
    # effect of a status poll" -- the seed=False default above stays. It runs AFTER the legacy check
    # so a split-brain environment still stops startup before anything is written, and BEFORE the
    # log line so the startup diagnostics name a file that now exists.
    try:
        canonical_settings_path(seed=True)
    except (paths.PathsError, OSError) as exc:       # startup must not die on this
        # An explicit LM3_SETTINGS naming a missing file is row 1's hard error, but it belongs to
        # the request that needs the file: a half-mounted or dead API says less than a late 400.
        log.warning("could not resolve or seed the LM3 settings file at startup: %s", exc)

    # The one-release legacy hardware-profile adopt, performed EXACTLY here: once, at startup, in a
    # controlled path. It is deliberately not part of hardware_profile_path() any more -- resolving
    # a path must never mutate the filesystem, or a 2 Hz status poll becomes a writer.
    try:
        adopted = paths.migrate_legacy_hardware_profile(
            env=server_env(), settings_file=canonical_settings_path(seed=False))
        if adopted is not None:
            log.info("adopted a legacy hardware profile into %s", adopted)
    except (paths.PathsError, OSError) as exc:
        log.warning("legacy hardware-profile migration skipped: %s", exc)

    # Same one-release adopt for the postprocessing settings, so the Postprocess tab and the two
    # standalone CLIs converge on one file instead of the CWD-relative one they used to disagree over.
    try:
        adopted = paths.migrate_legacy_postprocessing_settings(env=server_env())
        if adopted is not None:
            log.info("adopted legacy postprocessing settings into %s", adopted)
    except (paths.PathsError, OSError) as exc:
        log.warning("legacy postprocessing-settings migration skipped: %s", exc)

    log_resolved_paths("server")

    manager = jobs or JobManager()
    jobs_q: asyncio.Queue = asyncio.Queue(maxsize=64)
    token = _server_token()

    async def require_token(authorization: str = Header(default="")) -> None:
        expected = f"Bearer {token}"
        if not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    async def _worker() -> None:
        """Run queued pipeline jobs ONE at a time, off the event loop."""
        from leafmachine3.machine3 import machine3
        from leafmachine3.server.metrics_api import runtime_v2

        while True:
            job_id = await jobs_q.get()
            manager.mark_running(job_id)
            job = manager.get(job_id)
            try:
                if runtime_v2():
                    # Out of process: the server is never a lease holder (invariant 13), and the
                    # job inherits the section 2.4 handshake -- so a legacy job against a busy
                    # deployment is REFUSED rather than started beside the run that holds it.
                    await run_in_threadpool(_run_job_as_subprocess, job)
                else:
                    await run_in_threadpool(machine3, str(job.cfg_path))
                manager.mark_done(job_id)
            except Exception as exc:  # noqa: BLE001 - surface to the job, keep the worker alive
                log.exception("job %s failed", job_id)
                manager.mark_error(job_id, str(exc))
            finally:
                jobs_q.task_done()

    @asynccontextmanager
    async def lifespan(_app: "FastAPI"):
        worker = asyncio.create_task(_worker())
        _start_owner_watchdog()
        # Section 2.12: the local-client authentication flow. Electron reads this file rather than
        # inventing a token and hoping an existing server accepts it (section 2.11), so it has to
        # exist before the first window opens. It is published here, not in create_app(), because
        # create_app() also runs in-process in tests and in `import`-only contexts, and a bare
        # import must not write a secret to disk.
        write_connection_descriptor()
        try:
            yield
        finally:
            worker.cancel()
            with contextlib.suppress(Exception):
                remove_connection_descriptor()

    app = FastAPI(title="LeafMachine3", version=__version__, lifespan=lifespan)

    # -- legacy /v1/jobs ---------------------------------------------------- #
    # Section 4 Step 7 audits these four routes rather than deleting them: "removal is a decision
    # about external users, not in-tree call sites". The in-tree evidence is that there are NO
    # consumers (see :func:`note_legacy_jobs_use`), and the one capability they still hold alone is
    # multipart UPLOAD -- ``POST /v1/run/start`` runs a settings file that is already on the
    # server's filesystem. So they stay, instrumented, until a release names their removal.
    @app.post("/v1/jobs", dependencies=[Depends(require_token)])
    async def create_job(files: list[UploadFile] | None = None,
                         user_agent: str = Header(default="")) -> dict:
        note_legacy_jobs_use("POST /v1/jobs", user_agent)
        payload = [(f.filename or "upload.jpg", await f.read()) for f in (files or [])]
        job = manager.create(files=payload)
        await jobs_q.put(job.id)
        return {"job_id": job.id}

    # Plain ``def``, not ``async def``: these read the project ledger over SQLite, which blocks.
    # Starlette runs a sync endpoint in its threadpool, so the event loop stays free to keep the
    # status / logs / metrics streams flowing for every other client.
    @app.get("/v1/jobs/{jid}", dependencies=[Depends(require_token)])
    def status(jid: str, user_agent: str = Header(default="")) -> dict:
        note_legacy_jobs_use("GET /v1/jobs/{jid}", user_agent)
        try:
            return manager.status(jid)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown job")

    # ``response_class=`` plus a plain ``Any`` return annotation, NOT ``-> "StreamingResponse"``:
    # this module uses ``from __future__ import annotations``, so every annotation reaches FastAPI
    # as a STRING resolved against MODULE globals -- where a name imported inside create_app does
    # not exist. FastAPI then tries to build a response model out of the unresolvable forward ref
    # and app.openapi() raises PydanticUserError, taking /openapi.json and /docs down for the
    # WHOLE app. (Same trap postprocess_api.task_events documents.)
    @app.get("/v1/jobs/{jid}/events", dependencies=[Depends(require_token)],
             response_class=StreamingResponse)
    async def events(jid: str, user_agent: str = Header(default="")) -> Any:
        note_legacy_jobs_use("GET /v1/jobs/{jid}/events", user_agent)
        try:
            db_path = manager.db_path(jid)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown job")

        async def stream() -> "Any":
            # The same frame sequence as tail_progress, but the SQLite read goes to a worker
            # thread and the pacing uses asyncio.sleep. Iterating the synchronous generator here
            # would run its time.sleep(1 s) on the event loop -- measured at ~6 s of added
            # latency on EVERY other request for as long as one client held this stream open.
            for _ in range(TAIL_MAX_POLLS):
                snapshot = await asyncio.to_thread(_progress_snapshot, db_path)
                yield f"data: {json.dumps(snapshot)}\n\n"
                if snapshot.get("reporter_state") in {"done", "error"}:
                    break
                await asyncio.sleep(TAIL_POLL_S)

        return StreamingResponse(
            stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/v1/jobs/{jid}/results", dependencies=[Depends(require_token)])
    def results(jid: str, user_agent: str = Header(default="")) -> dict:
        note_legacy_jobs_use("GET /v1/jobs/{jid}/results", user_agent)
        try:
            return manager.results(jid)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown job")

    @app.get("/healthz/doctor", dependencies=[Depends(require_token)])
    async def healthz_doctor(full: bool = False) -> dict:
        """`lm3 doctor --production` for this server's environment, plus its desktop release identity.

        Quick (checks 1-4) by default; ``?full=1`` adds the driver and the accelerator probe, which
        starts a GPU child process. Authenticated, unlike /healthz, for exactly that reason.
        """
        from leafmachine3.doctor import Env, diagnose  # noqa: PLC0415

        def inspect_environment() -> dict:
            env = Env.current()
            result = diagnose(env, quick=not full, production=True).to_json()
            from importlib import resources  # noqa: PLC0415

            result["environment"] = {
                "python": env.python_version,
                "venv_uv": next((line.split("=", 1)[1].strip() for line in
                                 (Path(env.prefix) / "pyvenv.cfg").read_text().splitlines()
                                 if line.startswith("uv =")), None)
                if (Path(env.prefix) / "pyvenv.cfg").is_file() else None,
                "contract_sha256": hashlib.sha256(resources.files("leafmachine3").joinpath(
                    "_env_contract.json").read_bytes()).hexdigest(),
            }
            return result

        return await asyncio.to_thread(inspect_environment)

    @app.get("/healthz")
    async def healthz() -> dict:
        # UNAUTHENTICATED by design: this is the readiness probe every client polls before it
        # holds the secret, so nothing here may be a capability. It carries IDENTITY (who am I,
        # which deployment, which instance) and DIAGNOSTICS, and no field of it authorizes an
        # action.
        #
        # `pid` IS DIAGNOSTIC ONLY. It used to be documented here as "what lets a client that
        # ATTACHED to an already-running server still stop it" -- its only escalation path. That is
        # the behavior section 2.5 deletes and invariant 12 forbids: a PID read out of a response
        # body authorizes nothing, and a client that did not spawn this server has no control
        # authority over it however wedged it looks. Step 5b has now removed the last consumer (the
        # desktop shell no longer scrapes it), so the field survives purely for support logs and
        # for `ps`, and says so on the wire. Step 7 may drop it.
        # `paths` is DIAGNOSTIC ONLY too (plan section 4, Step 1: "Expose resolved paths in startup
        # logs and /healthz diagnostics"). It is what makes the Step 1 exit gate -- "from any
        # supported CWD every subsystem reports the same canonical settings path" -- checkable
        # from outside the process.
        identity = deployment_identity()
        return {
            # Section 2.11 / Step 5b: service, protocol version, instance ID, deployment key and
            # ownership mode. Together these are what let a client distinguish the three cases it
            # actually faces on a port -- MY LM3 server, an LM3 server for a DIFFERENT deployment,
            # and an unrelated service -- which used to be one undifferentiated "200 OK".
            "service": SERVICE_NAME,
            "status": "ok",
            "version": __version__,
            "protocol_version": HEALTH_PROTOCOL_VERSION,
            "instance_id": server_instance_id(),
            "deployment_id": identity["deployment_id"],
            "deployment_key": identity["deployment_key"],
            "ownership_mode": ownership_mode(),
            "provider": _current_provider(),
            "pid": os.getpid(),
            # Stated on the wire, not only in a comment: a reader of this body is exactly who used
            # to be tempted to signal that pid, and a field that says "diagnostic only" is harder
            # to misread than a field that merely stopped being documented.
            "pid_is_diagnostic_only": True,
            "paths": path_diagnostics(),
        }

    @app.post("/v1/shutdown", dependencies=[Depends(require_token)])
    async def shutdown(request: Request) -> dict:
        """Stop this server. Answers first, then exits hard -- see :func:`_hard_exit_soon`.

        A running LM3 job is NOT touched: it is a separate session, it is checkpointed, and it is
        meant to survive a server restart. The UI stops the job itself before it asks for this.
        """
        # Loopback as well as the token: this is the one request whose entire effect is destructive,
        # so it must not travel even when someone binds a routable interface with `lm3 serve --host`.
        client = request.client.host if request.client else None
        if not _client_is_loopback(client):
            log.warning("refusing a shutdown request from %r", client)
            raise HTTPException(status_code=403, detail="shutdown is loopback-only")
        log.info("shutdown requested by %s -- exiting", client)
        _hard_exit_soon()
        return {"status": "stopping", "pid": os.getpid()}

    # The profiler benchmarks the real hardware and rewrites hardware_settings.yaml, so exactly
    # one may be in flight: two at once measure each other's load and race on the same file.
    setup_tasks: set = set()
    #: job id -> the retained control handle for its ``lm3-setup`` child (section 2.5). Populated
    #: only under runtime v2; it is what makes Stop able to kill the setup/calibration tree.
    setup_children: dict = {}

    @app.post("/v1/setup", dependencies=[Depends(require_token)])
    async def setup(optimize: bool = True, force: bool = False, calibrate: bool = False) -> dict:
        from leafmachine3.core.config import Config
        from leafmachine3.server.metrics_api import runtime_v2
        from leafmachine3.setup.hardware_setup import run_setup

        running = next((j for j in manager.jobs_of_kind("setup")
                        if j.state in ("queued", "running")), None)
        if running is not None:
            raise HTTPException(status_code=409,
                                detail=f"a hardware profile is already running (job {running.id})")

        if runtime_v2():
            # Section 2.13 requires a RESOLVED config, so the resolver's refusal is the answer:
            # it names the file it looked for, where an unhandled PathsError would be a 500 and
            # ``run_setup(None, ...)`` an AttributeError surfaced as an opaque job error.
            try:
                job = manager.create_setup()
            except paths.PathsError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"hardware setup needs a canonical settings file -- {exc}") from exc

            # Section 2.13: a dedicated lm3-setup subprocess with a retained handle and its own
            # process group. The config is RESOLVED here, not passed as None -- run_setup
            # dereferences cfg at _fingerprint() before any guard, so None is an AttributeError
            # surfaced as an opaque job error rather than a message naming the missing file.
            try:
                child = launch_setup_subprocess(job, optimize=optimize, force=force,
                                                calibrate=calibrate)
            except FileNotFoundError as exc:
                manager.mark_error(job.id, str(exc))
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except OSError as exc:
                manager.mark_error(job.id, str(exc))
                raise HTTPException(status_code=500,
                                    detail=f"could not start lm3-setup: {exc}") from exc
            manager.mark_running(job.id)
            setup_children[job.id] = child
            threading.Thread(target=watch_setup_subprocess, args=(job, child, manager),
                             name="lm3-setup-watch", daemon=True).start()
            return {"job_id": job.id}

        job = manager.create_setup()

        async def _run() -> None:
            manager.mark_running(job.id)
            try:
                cfg = Config.load(str(job.cfg_path)) if job.cfg_path.is_file() else None
                await run_in_threadpool(
                    run_setup, cfg, optimize=optimize, force=force,
                    calibrate=calibrate, on_progress=job.emit
                )
                manager.mark_done(job.id)
            except Exception as exc:  # noqa: BLE001
                manager.mark_error(job.id, str(exc))

        # Hold a strong reference: a bare create_task() is only weakly held by the loop, so the
        # profiler can be garbage-collected mid-run (see asyncio.create_task's own warning).
        task = asyncio.create_task(_run())
        setup_tasks.add(task)
        task.add_done_callback(setup_tasks.discard)
        return {"job_id": job.id}

    @app.get("/v1/setup/events", dependencies=[Depends(require_token)])
    async def setup_events(jid: str) -> dict:
        from leafmachine3.server.metrics_api import runtime_v2

        try:
            job = manager.get(jid)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown setup job")
        if runtime_v2():
            # Section 2.13: the append-only JSONL log, replayed. It survives a UI disconnect and a
            # server restart, and it cannot fill a pipe and stall the subprocess -- which is why it
            # was chosen over a live channel.
            # Off the loop: it is a synchronous file read, and the UI polls it throughout a
            # multi-minute calibration.
            replayed = await run_in_threadpool(read_setup_events, setup_event_log_path(job))
            return {"job_id": jid, "state": job.state, "events": replayed}
        return {"job_id": jid, "state": job.state, "events": job.events}

    @app.post("/v1/setup/stop", dependencies=[Depends(require_token)])
    async def setup_stop(jid: Optional[str] = None) -> dict:
        """Terminate the setup/calibration tree, and leave the server serving (section 2.13).

        Authority is the retained handle and nothing else (section 2.5): a setup this process did
        not launch -- one from a previous server, say -- is observable and is not stoppable here.
        """
        from leafmachine3.server.metrics_api import runtime_v2

        if not runtime_v2():
            raise HTTPException(
                status_code=409,
                detail="hardware setup runs inside the server process; there is no child to stop")
        live = [(job_id, child) for job_id, child in setup_children.items()
                if child.alive and (jid is None or job_id == jid)]
        if not live:
            raise HTTPException(status_code=409, detail="no managed hardware setup to stop")
        stopped: list[str] = []
        for job_id, child in live:
            child.stopped_by_user = True
            await run_in_threadpool(child.terminate)
            stopped.append(job_id)
        return {"stopped": stopped}

    @app.get("/v1/hardware", dependencies=[Depends(require_token)])
    def hardware() -> dict:                          # reads + parses a YAML file: keep it off the loop
        return read_hardware_settings()

    # ---- LM3 desktop app: feature routers + the static UI ------------------ #
    # Each router is built by its own module (settings / status+logs / results / postprocessing /
    # metrics+run-control) and shares this app's Bearer dependency. Imported HERE rather than at
    # module scope so a base install without the ``server`` extra still imports app.py cleanly.
    _auth = [Depends(require_token)]
    from leafmachine3.server import (            # noqa: WPS433 - lazy by design
        metrics_api, models_api, postprocess_api, progress_api, results_api, settings_api,
    )
    # ORDER CARRIES NO MEANING. It used to: results_api and progress_api both defined
    # ``GET /v1/runs``, whoever was registered first shadowed the other, and a comment here asked
    # the next reader not to sort the tuple -- so an innocuous tidy-up would have handed the Results
    # tab a listing with no run ids and nothing would have failed at the point of the change. That
    # is the "registration-order workaround" section 4 Step 4 says to delete, and it is deleted at
    # the source: section 5 consolidates the path onto ``results_api``, and progress_api no longer
    # defines it. Two routers must never answer one path again; if one ever does, fix the duplicate
    # rather than reintroducing a load-bearing tuple order.
    for _mod in (settings_api, results_api, progress_api, postprocess_api, metrics_api, models_api):
        try:                                     # every router module exposes router(dependencies=)
            app.include_router(_mod.router(dependencies=_auth))
        except Exception as exc:                 # one broken router must not sink the whole app
            log.exception("could not mount %s: %s", _mod.__name__, exc)

    # The UI is a plain static bundle (no build step) served at "/". EventSource cannot set headers,
    # so the SSE endpoints also accept ?token=; index.html receives the token via a <meta> tag.
    _ui = Path(__file__).resolve().parent / "ui"
    if _ui.is_dir():
        from fastapi import Request
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles

        # ``from __future__ import annotations`` (top of this file) stringifies every annotation,
        # and FastAPI resolves those strings against the handler's MODULE globals -- where a name
        # imported inside create_app does not exist. Publishing it is what lets the lazy import
        # (needed so a base install can still import this module) coexist with `request: Request`.
        globals().setdefault("Request", Request)

        @app.get("/", response_class=HTMLResponse)
        async def _index(request: Request) -> Any:
            # This response carries the shared secret, so it is the one place the Bearer check
            # cannot protect -- the whole point is to bootstrap a client that has no token yet.
            # Hand it over ONLY to a loopback peer that also addressed us as loopback: that
            # covers the Electron shell and a local browser, and refuses both a remote client
            # (when someone binds a routable interface) and a DNS-rebinding page. Everyone else
            # gets the same UI with an empty token, so every /v1 call it makes 401s.
            #
            # RESIDUAL RISK, stated plainly: a loopback peer is any process on this host running
            # as this user, so embedding still hands the secret to a local process that asks. Set
            # LM3_EMBED_TOKEN=0 to switch that off entirely -- the Electron shell passes ?token=
            # itself and keeps working, while a plain browser then needs the token supplied by hand.
            html = (_ui / "index.html").read_text(encoding="utf-8")
            client = request.client.host if request.client else None
            host = request.headers.get("host", "")
            embed = os.environ.get("LM3_EMBED_TOKEN", "1").strip().lower() not in {"0", "false", "no"}
            # ``no-store``, NOT ``no-cache`` (gate 13). The distinction is documented below and it
            # matters here more than anywhere: ``no-cache`` STORES the response and merely
            # revalidates it, so a token-bearing page would be written to the browser cache on
            # disk. ``no-store`` is the only directive that keeps it out. Applied to the bootstrap
            # document unconditionally -- it is one small file, and whether it carries a token
            # depends on request details that are the wrong thing to hang a security header on.
            no_store = {"Cache-Control": "no-store"}
            if embed and _client_is_loopback(client) and _host_header_is_loopback(host):
                return HTMLResponse(html.replace("{{LM3_TOKEN}}", _server_token()), headers=no_store)
            log.warning("refusing to embed the LM3 server token (client=%r, Host=%r)", client, host)
            return HTMLResponse(html.replace("{{LM3_TOKEN}}", ""), headers=no_store)

        # The UI bundle is unversioned -- no build step, no content hashes, just files on disk that
        # a `git pull` replaces in place. StaticFiles sends Last-Modified and ETag but no
        # Cache-Control, and a browser fed no freshness directive at all falls back to HEURISTIC
        # caching: it invents a lifetime from how old the file is, so long-untouched files get
        # cached for DAYS without revalidating. Electron does the same, which is how the desktop
        # app ends up running edited-hours-ago JS and edits appear to have no effect.
        #
        # "no-cache" is not "do not cache" -- it stores as usual but must revalidate, so the ETag
        # still turns the common case into a 304 with no body. Over loopback that costs nothing.
        @app.middleware("http")
        async def _revalidate_ui(request, call_next):
            response = await call_next(request)
            if not request.url.path.startswith("/v1/"):
                response.headers.setdefault("Cache-Control", "no-cache")
            return response

        app.mount("/", StaticFiles(directory=str(_ui), html=True), name="ui")

    return app


def read_hardware_settings() -> dict:
    """Return the current hardware profile as a plain dict (empty if absent).

    The profile is deployment-scoped and machine-keyed (section 3.1 row 2), resolved through the
    same helper ``metrics_api`` uses -- previously this read a bare ``hardware_settings.yaml``
    from the server's CWD and could therefore describe a different file than the tuning panel.
    """
    path = canonical_hardware_path()
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _current_provider() -> str:
    """Report the execution provider that would bind (never raises)."""
    try:
        import onnxruntime as ort  # type: ignore

        available = ort.get_available_providers()
        return available[0] if available else "CPUExecutionProvider"
    except Exception:  # noqa: BLE001 - onnxruntime optional
        return "CPUExecutionProvider"


# --------------------------------------------------------------------------- #
# ``lm3 serve`` entry point
# --------------------------------------------------------------------------- #
def serve(host: str = "127.0.0.1", port: int = 8765, *, jobs_root: Path | None = None,
          settings: str | os.PathLike[str] | None = None) -> None:
    """Run the server on loopback (requires the ``server`` extra).

    ``lm3 serve`` PINS the canonical settings path explicitly (plan section 3.1: "``lm3 serve``
    and Electron always pass the canonical settings path explicitly rather than relying on any
    fallback"). It is resolved once here -- seeding a first-run file from the packaged template if
    nothing exists yet -- and exported as ``LM3_SETTINGS``, so every router, every request and
    every child process the server spawns reads the SAME file no matter where it was launched
    from. :func:`create_app` seeds too, for the entry points that never reach here; what is unique
    to ``lm3 serve`` is the explicit pin and honoring ``--config``. Resolution inside a request
    handler must still never create a file as a side effect.
    """
    import uvicorn

    check_legacy_env()
    resolved = canonical_settings_path(settings, seed=True)
    os.environ[paths.ENV_SETTINGS] = str(resolved)
    os.environ.pop(paths.LEGACY_ENV_ALIASES[paths.ENV_SETTINGS], None)   # folded in above
    # State the bound address for connection.private.json (section 2.12). The application object
    # never sees these arguments -- uvicorn does -- so without this the descriptor would have to
    # GUESS the port, and a client reading it would be told to talk to the wrong server.
    os.environ[ENV_BIND_HOST] = str(host)
    os.environ[ENV_BIND_PORT] = str(port)
    log.info("lm3 serve: settings %s, binding %s:%s", resolved, host, port)

    app = create_app(JobManager(jobs_root) if jobs_root else None)
    _server_token()                                  # ensure a token is minted + logged before start
    uvicorn.run(app, host=host, port=port)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI for ``lm3 serve`` and ``python -m leafmachine3.server``."""
    import argparse

    parser = argparse.ArgumentParser(prog="lm3 serve", description="Run the local LeafMachine3 server.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (loopback by default)")
    parser.add_argument("--port", type=int, default=None,
                        help="port (default: this deployment's port -- 8765 for the default "
                             "deployment; a NAMED deployment must set LM3_PORT, section 2.1)")
    parser.add_argument("--jobs-root", default=None, help="directory for staged job dirs")
    parser.add_argument("--config", default=None,
                        help="LM3_settings.yaml to serve (default: the canonical resolved path)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Section 2.1's port rule, enforced rather than defaulted: "starting a named non-default
    # deployment without one is a startup error, not a silent collision on 8765". Two deployments
    # that quietly share a port is the failure /healthz deployment verification exists to catch as
    # a LAST defense -- this is the first one.
    try:
        port = args.port if args.port is not None else paths.resolve_port()
    except paths.PathsError as exc:
        print(f"lm3 serve: {exc}", file=sys.stderr)
        return 2

    from leafmachine3.doctor import run_startup_gate  # noqa: PLC0415

    refused = run_startup_gate("lm3 serve")           # `lm3 doctor` checks 1-4, before binding a port
    if refused is not None:
        return refused

    serve(args.host, port, jobs_root=Path(args.jobs_root) if args.jobs_root else None,
          settings=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
