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

import ipaddress
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

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
        return {"error": str(exc)}
    return paths.describe_resolved_paths(env=env, explicit_settings=explicit_settings, seed=False)


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
        job = self.get(job_id)
        job.state = "error"
        job.error = msg

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
            descriptor = (paths.deployment_runtime_dir(env=server_env())
                          / paths.CONNECTION_PRIVATE_FILENAME)
            where = f"it will be written to {descriptor}"
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
                os._exit(0)

    threading.Thread(target=_watch, name="lm3-owner-watchdog", daemon=True).start()


def create_app(jobs: JobManager | None = None) -> Any:
    """Build and return the FastAPI application.

    Imported lazily so the base install (no ``fastapi``) can still import this module.
    """
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

        while True:
            job_id = await jobs_q.get()
            manager.mark_running(job_id)
            job = manager.get(job_id)
            try:
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
        try:
            yield
        finally:
            worker.cancel()

    app = FastAPI(title="LeafMachine3", version=__version__, lifespan=lifespan)

    @app.post("/v1/jobs", dependencies=[Depends(require_token)])
    async def create_job(files: list[UploadFile] | None = None) -> dict:
        payload = [(f.filename or "upload.jpg", await f.read()) for f in (files or [])]
        job = manager.create(files=payload)
        await jobs_q.put(job.id)
        return {"job_id": job.id}

    # Plain ``def``, not ``async def``: these read the project ledger over SQLite, which blocks.
    # Starlette runs a sync endpoint in its threadpool, so the event loop stays free to keep the
    # status / logs / metrics streams flowing for every other client.
    @app.get("/v1/jobs/{jid}", dependencies=[Depends(require_token)])
    def status(jid: str) -> dict:
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
    async def events(jid: str) -> Any:
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
    def results(jid: str) -> dict:
        try:
            return manager.results(jid)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown job")

    @app.get("/healthz")
    async def healthz() -> dict:
        # `pid` is what lets a client that ATTACHED to an already-running server still stop it: with
        # no process handle of its own, that pid is its only escalation path when the server is too
        # wedged to honor POST /v1/shutdown.
        # `paths` is DIAGNOSTIC ONLY (plan section 4, Step 1: "Expose resolved paths in startup
        # logs and /healthz diagnostics"). It is what makes the Step 1 exit gate -- "from any
        # supported CWD every subsystem reports the same canonical settings path" -- checkable
        # from outside the process. The rest of this body is untouched; section 4 Step 5b owns it.
        return {"status": "ok", "version": __version__, "provider": _current_provider(),
                "pid": os.getpid(), "paths": path_diagnostics()}

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

    @app.post("/v1/setup", dependencies=[Depends(require_token)])
    async def setup(optimize: bool = True, force: bool = False, calibrate: bool = False) -> dict:
        from leafmachine3.core.config import Config
        from leafmachine3.setup.hardware_setup import run_setup

        running = next((j for j in manager.jobs_of_kind("setup")
                        if j.state in ("queued", "running")), None)
        if running is not None:
            raise HTTPException(status_code=409,
                                detail=f"a hardware profile is already running (job {running.id})")

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
        try:
            job = manager.get(jid)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown setup job")
        return {"job_id": jid, "state": job.state, "events": job.events}

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
    # ORDER MATTERS: results_api and progress_api both define GET /v1/runs. results_api's listing is
    # the richer one (it indexes every run folder and assigns the stable `id` that all its other
    # /v1/runs/{id}/... routes key off), so it must be registered FIRST or the Results tab receives
    # entries with no id and can never select a run.
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
    log.info("lm3 serve: settings %s", resolved)

    app = create_app(JobManager(jobs_root) if jobs_root else None)
    _server_token()                                  # ensure a token is minted + logged before start
    uvicorn.run(app, host=host, port=port)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI for ``lm3 serve`` and ``python -m leafmachine3.server``."""
    import argparse

    parser = argparse.ArgumentParser(prog="lm3 serve", description="Run the local LeafMachine3 server.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (loopback by default)")
    parser.add_argument("--port", type=int, default=8765, help="port")
    parser.add_argument("--jobs-root", default=None, help="directory for staged job dirs")
    parser.add_argument("--config", default=None,
                        help="LM3_settings.yaml to serve (default: the canonical resolved path)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    serve(args.host, args.port, jobs_root=Path(args.jobs_root) if args.jobs_root else None,
          settings=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
