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

import json
import logging
import os
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import yaml

log = logging.getLogger("leafmachine3.server")

DEFAULT_JOBS_ROOT = Path(os.environ.get("LM3_SERVER_JOBS", "runs/_server_jobs"))
_TOKEN_ENV = "LM3_SERVER_TOKEN"


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

    def __init__(self, root: Path = DEFAULT_JOBS_ROOT) -> None:
        self.root = Path(root)
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
        job = Job(id=job_id, root=root, cfg_path=Path("LM3_settings.yaml"), kind="setup")
        self._jobs[job_id] = job
        return job

    def _base_settings(self, settings: Optional[dict]) -> dict:
        """Start from the user's ``LM3_settings.yaml`` if present, else built-in defaults."""
        if settings:
            return json.loads(json.dumps(settings))     # deep copy of a plain dict
        base = Path("LM3_settings.yaml")
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
        rows = self._read(job, f"SELECT * FROM {table} WHERE specimen_id = ?", (sid,))
        return [dict(r) for r in rows] if rows else []


# --------------------------------------------------------------------------- #
# Progress tailing (used by the SSE endpoint)
# --------------------------------------------------------------------------- #
def tail_progress(db_path: Path, *, poll_s: float = 1.0, max_polls: int = 3600) -> Iterator[str]:
    """Yield JSON progress snapshots from a job's project DB until it drains or stalls.

    Emits Server-Sent-Events ``data:`` frames. Terminates when the reporter stage is
    ``done`` (or errored) or after ``max_polls`` idle iterations.
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
    """Return the shared Bearer secret, generating and exporting one if unset."""
    token = os.environ.get(_TOKEN_ENV)
    if not token:
        token = secrets.token_urlsafe(24)
        os.environ[_TOKEN_ENV] = token
        log.warning("LM3 server token (set %s to override): %s", _TOKEN_ENV, token)
    return token


# --------------------------------------------------------------------------- #
# FastAPI application (built lazily; requires the `server` extra)
# --------------------------------------------------------------------------- #
def create_app(jobs: JobManager | None = None) -> Any:
    """Build and return the FastAPI application.

    Imported lazily so the base install (no ``fastapi``) can still import this module.
    """
    import asyncio
    from contextlib import asynccontextmanager

    from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import StreamingResponse

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
        try:
            yield
        finally:
            worker.cancel()

    app = FastAPI(title="LeafMachine3", version="3.0.0", lifespan=lifespan)

    @app.post("/v1/jobs", dependencies=[Depends(require_token)])
    async def create_job(files: list[UploadFile] | None = None) -> dict:
        payload = [(f.filename or "upload.jpg", await f.read()) for f in (files or [])]
        job = manager.create(files=payload)
        await jobs_q.put(job.id)
        return {"job_id": job.id}

    @app.get("/v1/jobs/{jid}", dependencies=[Depends(require_token)])
    async def status(jid: str) -> dict:
        try:
            return manager.status(jid)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown job")

    @app.get("/v1/jobs/{jid}/events", dependencies=[Depends(require_token)])
    async def events(jid: str) -> "StreamingResponse":
        try:
            db_path = manager.db_path(jid)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown job")

        async def stream() -> "Any":
            for frame in tail_progress(db_path):
                yield frame
                await asyncio.sleep(0)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/v1/jobs/{jid}/results", dependencies=[Depends(require_token)])
    async def results(jid: str) -> dict:
        try:
            return manager.results(jid)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown job")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "version": "3.0.0", "provider": _current_provider()}

    @app.post("/v1/setup", dependencies=[Depends(require_token)])
    async def setup(optimize: bool = True, force: bool = False) -> dict:
        from leafmachine3.core.config import Config
        from leafmachine3.setup.hardware_setup import run_setup

        job = manager.create_setup()

        async def _run() -> None:
            manager.mark_running(job.id)
            try:
                cfg = Config.load(str(job.cfg_path)) if job.cfg_path.is_file() else None
                await run_in_threadpool(
                    run_setup, cfg, optimize=optimize, force=force, on_progress=job.emit
                )
                manager.mark_done(job.id)
            except Exception as exc:  # noqa: BLE001
                manager.mark_error(job.id, str(exc))

        asyncio.create_task(_run())
        return {"job_id": job.id}

    @app.get("/v1/setup/events", dependencies=[Depends(require_token)])
    async def setup_events(jid: str) -> dict:
        try:
            job = manager.get(jid)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown setup job")
        return {"job_id": jid, "state": job.state, "events": job.events}

    @app.get("/v1/hardware", dependencies=[Depends(require_token)])
    async def hardware() -> dict:
        return read_hardware_settings()

    return app


def read_hardware_settings() -> dict:
    """Return the current ``hardware_settings.yaml`` as a plain dict (empty if absent)."""
    path = Path("hardware_settings.yaml")
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
def serve(host: str = "127.0.0.1", port: int = 8765, *, jobs_root: Path | None = None) -> None:
    """Run the server on loopback (requires the ``server`` extra)."""
    import uvicorn

    app = create_app(JobManager(jobs_root) if jobs_root else None)
    _server_token()                                  # ensure a token is minted + logged before start
    uvicorn.run(app, host=host, port=port)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI for ``python -m leafmachine3.server`` / ``lm3 serve``."""
    import argparse

    parser = argparse.ArgumentParser(prog="lm3-serve", description="Run the local LeafMachine3 server.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (loopback by default)")
    parser.add_argument("--port", type=int, default=8765, help="port")
    parser.add_argument("--jobs-root", default=None, help="directory for staged job dirs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    serve(args.host, args.port, jobs_root=Path(args.jobs_root) if args.jobs_root else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
