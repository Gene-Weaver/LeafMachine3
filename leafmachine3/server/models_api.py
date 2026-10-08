"""``/v1/models``: the model installer behind the GUI's "Install Models from Hugging Face" button.

    GET  /v1/models/status                 -> installer.status() + the resolved models folder
    GET  /v1/models/catalog                -> installer.catalog(): every stage / variant / format + the active one
    POST /v1/models/activate               -> {stage, model_key, format}: point modules.<stage>.model at an installed file
    POST /v1/models/install                -> start an install in a worker thread; {task_id}
                                              body: {actions?, models? [[stage, key], ...], formats?, force?}
    GET  /v1/models/install/{task_id}      -> snapshot {state, events[], error}
    GET  /v1/models/install/{task_id}/events  (SSE) -> the same events as they happen, then "done"

One install at a time: a second POST while one is running returns the running task. The installer
itself does the backup / verify / rollback work (see ``leafmachine3.modelhub.installer``); this module
only moves it off the event loop and streams its progress events.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
import uuid
from collections import OrderedDict

from leafmachine3 import __version__
from typing import Any, Iterator, Optional

log = logging.getLogger("leafmachine3.server.models")

MAX_TASKS = 8
SSE_POLL_S = 0.25
SSE_MAX_S = 6 * 3600


class _Task:
    def __init__(self, task_id: str, body: dict[str, Any]) -> None:
        self.id = task_id
        self.body = body
        self.state = "running"
        self.error: str | None = None
        self.events: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None
        self._lock = threading.Lock()

    def push(self, ev: dict[str, Any]) -> None:
        with self._lock:
            ev = {"seq": len(self.events) + 1, "t": time.time(), **ev}
            self.events.append(ev)

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        with self._lock:
            return {"task_id": self.id, "state": self.state, "error": self.error, "started_at": self.started_at,
                    "finished_at": self.finished_at, "events": [e for e in self.events if e["seq"] > since],
                    "summary": (self.result or {}).get("summary") if self.result else None}


_tasks: "OrderedDict[str, _Task]" = OrderedDict()
_tasks_lock = threading.Lock()


def _running() -> Optional[_Task]:
    with _tasks_lock:
        return next((t for t in _tasks.values() if t.state == "running"), None)


def _run(task: _Task) -> None:
    from leafmachine3.modelhub import installer  # noqa: PLC0415

    body = task.body
    try:
        root = body.get("dest") or None
        models = [tuple(m) for m in (body.get("models") or [])] or None
        result = installer.install(root, actions=body.get("actions") or None, formats=body.get("formats") or None,
                                   force=bool(body.get("force")), progress=task.push, models=models)
        task.result = result
        task.state = "done"
    except installer.InstallError as exc:
        task.error = str(exc)
        task.state = "failed"
        task.push({"type": "failed", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        log.exception("models install crashed")
        task.error = f"{type(exc).__name__}: {exc}"
        task.state = "failed"
        task.push({"type": "failed", "message": task.error})
    finally:
        task.finished_at = time.time()


def start_install(body: dict[str, Any]) -> _Task:
    with _tasks_lock:
        running = next((t for t in _tasks.values() if t.state == "running"), None)
        if running:
            return running
        task = _Task(uuid.uuid4().hex[:12], body)
        _tasks[task.id] = task
        while len(_tasks) > MAX_TASKS:
            _tasks.popitem(last=False)
    threading.Thread(target=_run, args=(task,), name=f"lm3-models-install-{task.id}", daemon=True).start()
    return task


def _frame(event: str, **data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def sse_frames(task: _Task, *, poll_s: float = SSE_POLL_S, max_seconds: float = SSE_MAX_S) -> Iterator[str]:
    cursor = 0
    deadline = time.time() + max_seconds
    last_ping = time.time()
    while time.time() < deadline:
        snap = task.snapshot(since=cursor)
        for ev in snap["events"]:
            cursor = ev["seq"]
            yield _frame("progress", **ev)
        if snap["state"] != "running":
            yield _frame("done", state=snap["state"], error=snap["error"], summary=snap["summary"])
            return
        if time.time() - last_ping > 15:
            last_ping = time.time()
            yield _frame("ping", t=last_ping)
        time.sleep(poll_s)
    yield _frame("done", state="timeout", error="stream timed out", summary=None)


def router(dependencies: Optional[list[Any]] = None):
    from fastapi import APIRouter, Depends, Header, HTTPException, Query  # noqa: PLC0415
    from fastapi.responses import StreamingResponse  # noqa: PLC0415

    # Guards are applied PER ROUTE, not to the router: the SSE route cannot use the header-only
    # guard (EventSource is unable to set an Authorization header), so it accepts ``?token=`` as
    # well and checks it against the same secret -- the pattern progress_api / postprocess_api use.
    # With a router-level guard the stream answered 401, the browser never saw "done", and the
    # install button sat on "Installing..." forever.
    deps = list(dependencies or [])
    api = APIRouter(prefix="/v1/models", tags=["models"])

    async def stream_token(
        token: Optional[str] = Query(default=None, description="Bearer secret, for EventSource"),
        authorization: str = Header(default=""),
    ) -> None:
        if not deps:
            return
        try:                                    # the same secret app.py mints/uses for every route
            from leafmachine3.server.app import _server_token  # noqa: PLC0415

            expected = _server_token() or ""
        except Exception:  # noqa: BLE001 - router mounted standalone
            expected = os.environ.get("LM3_SERVER_TOKEN") or ""
        if not expected:            # fail closed: no secret configured means nobody gets the stream
            raise HTTPException(status_code=401, detail="this LM3 server has no token configured; the stream is refused")
        if token and secrets.compare_digest(str(token), expected):
            return
        if authorization and secrets.compare_digest(authorization, f"Bearer {expected}"):
            return
        raise HTTPException(status_code=401, detail="invalid or missing token")

    @api.get("/status", dependencies=deps)
    def get_status(verify: bool = False) -> dict[str, Any]:
        from leafmachine3.modelhub import installer  # noqa: PLC0415

        try:
            return installer.status(verify_hashes=verify)
        except Exception as exc:  # noqa: BLE001 - a broken lock/folder must surface as JSON, not a 500 page
            raise HTTPException(status_code=500, detail=f"models status failed: {exc}") from exc

    @api.get("/catalog", dependencies=deps)
    def get_catalog() -> dict[str, Any]:
        from leafmachine3.modelhub import installer  # noqa: PLC0415
        from leafmachine3.server import settings_api  # noqa: PLC0415

        try:
            settings = settings_api.read_settings(with_text=False)
            out = installer.catalog(settings_values=settings.get("values") or {})
            out["settings"] = {"yaml_path": settings.get("yaml_path"), "mtime": settings.get("mtime")}
            out["app_version"] = __version__            # the running LM3, for the update dialogs
            return out
        except Exception as exc:  # noqa: BLE001 - a broken lock/folder/settings must surface as JSON
            raise HTTPException(status_code=500, detail=f"models catalog failed: {exc}") from exc

    @api.post("/activate", dependencies=deps)
    def post_activate(body: dict[str, Any]) -> dict[str, Any]:
        """Make ``(stage, model_key, format)`` the active model by writing the settings file.

        The YAML stays the truth: this writes ``modules.<stage>.model`` (and the alternate's extra
        keys, restoring the built-in default for keys the chosen variant does not set) through the
        same validated, backed-up write the Settings tab uses, then returns the fresh catalog.
        """
        from leafmachine3.modelhub import installer, registry  # noqa: PLC0415
        from leafmachine3.server import settings_api  # noqa: PLC0415

        stage, model_key, fmt = (str(body.get(k) or "") for k in ("stage", "model_key", "format"))
        if not (stage and model_key and fmt):
            raise HTTPException(status_code=400, detail="activate needs stage, model_key and format")
        lock = registry.load_lock()
        root = installer.models_root()
        try:
            plan = installer.activation_plan(lock, stage, model_key, fmt, root)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except installer.InstallError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        current = settings_api.read_settings(with_text=False)
        if current.get("readonly"):
            raise HTTPException(status_code=409, detail="the settings file is read-only")
        values = current.get("values") or {}
        block = values.setdefault("modules", {}).setdefault(stage, {})
        if not isinstance(block, dict):
            raise HTTPException(status_code=500, detail=f"modules.{stage} is not a mapping in the settings file")
        block["model"] = dict(plan["model"])
        # Only knobs the stage actually has: a key the yaml already carries, or one the settings
        # schema (settings_meta.json) lists for this stage. The lock knows an input size for every
        # model, but a stage that ignores it (the specimen segmenter runs each model at its own
        # size) must not have one written into its block.
        meta = settings_api.read_meta()
        def live(k: str) -> bool:          # listed for this stage and not retired
            entry = meta.get(f"modules.{stage}.{k}")
            return isinstance(entry, dict) and not entry.get("retired")
        known = {k for k in block if live(k) or f"modules.{stage}.{k}" not in meta} \
            | {k for k in {**plan["restore"], **plan["set"]} if live(k)}
        for k, v in {**plan["restore"], **plan["set"]}.items():
            if k in known:
                block[k] = v
        defaults = (settings_api._defaults().get("modules") or {}).get(stage) or {}
        for k in plan["unset"]:
            if k in defaults:
                block[k] = defaults[k]
            else:
                block.pop(k, None)
        try:
            written = settings_api.write_settings(values, if_mtime=current.get("mtime"), backup=True)
        except settings_api.SettingsError as exc:
            status = 409 if exc.detail.get("conflict") else 400
            raise HTTPException(status_code=status, detail={"message": str(exc), "errors": exc.errors}) from exc
        out = installer.catalog(settings_values=written.get("values") or values)
        out["settings"] = {"yaml_path": written.get("yaml_path"), "mtime": written.get("mtime"),
                           "warnings": written.get("warnings") or []}
        out["app_version"] = __version__
        out["activated"] = plan["model"]
        return out

    @api.post("/install", dependencies=deps)
    def post_install(body: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        task = start_install(body or {})
        return {"task_id": task.id, "state": task.state, "already_running": task.body is not (body or {}) and task.state == "running"
                and len(task.events) > 0}

    @api.get("/install/{task_id}", dependencies=deps)
    def get_install(task_id: str, since: int = 0) -> dict[str, Any]:
        task = _tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="unknown install task")
        return task.snapshot(since=since)

    @api.get("/install/{task_id}/events", dependencies=[Depends(stream_token)], response_class=StreamingResponse)
    def get_install_events(task_id: str) -> Any:
        task = _tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="unknown install task")
        return StreamingResponse(sse_frames(task), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return api
