"""``/v1/models``: status, install task lifecycle and the SSE stream, against a faked installer."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from leafmachine3.modelhub import installer
from leafmachine3.server import models_api

from tests._contract_helpers import TEST_TOKEN, bearer, isolate_server_paths, reset_server_module_state


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reset_server_module_state()
    isolate_server_paths(tmp_path, monkeypatch)
    # a one-action lock, a models root under tmp, and a downloader that writes known bytes
    data = b"ONNX"
    import hashlib
    lock = {"schema_version": 1, "lm3_version": "3.0.0", "default_formats": ["onnx"], "actions": {
        "det": {"required": True, "units": [{"repo_id": "org/det", "revision": "r1", "model_key": "k", "files": [
            {"src": "onnx/model.onnx", "dest": "det/model.onnx", "format": "onnx", "sha256": hashlib.sha256(data).hexdigest(), "bytes": 4}]}]}}}
    lock_path = tmp_path / "lock.yaml"
    lock_path.write_text(yaml.safe_dump(lock))
    monkeypatch.setenv("LM3_MODELS_LOCK", str(lock_path))
    monkeypatch.setenv(installer.ENV_ROOT, str(tmp_path / "models"))

    def fake_download(unit, lf, scratch, progress):
        p = Path(scratch) / lf.src
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p
    monkeypatch.setattr(installer, "_download", fake_download)
    models_api._tasks.clear()

    from leafmachine3.server.app import JobManager, create_app
    app = create_app(JobManager(tmp_path / "jobs" / "managed"))
    try:
        yield TestClient(app)
    finally:
        models_api._tasks.clear()
        reset_server_module_state()


def _auth(client: TestClient) -> dict[str, str]:
    return bearer()


def _wait(client: TestClient, headers: dict[str, str], task_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = client.get(f"/v1/models/install/{task_id}", headers=headers).json()
        if snap["state"] != "running":
            return snap
        time.sleep(0.05)
    raise AssertionError("install task did not finish")


def test_status_then_install_then_status(client: TestClient):
    headers = _auth(client)
    st = client.get("/v1/models/status", headers=headers)
    assert st.status_code == 200, st.text
    body = st.json()
    assert body["actions"]["det"]["state"] == "missing"
    assert body["summary"]["button_label"] == "Install Models from Hugging Face"
    assert body["root"].endswith("models")

    started = client.post("/v1/models/install", json={}, headers=headers)
    assert started.status_code == 200, started.text
    task_id = started.json()["task_id"]
    snap = _wait(client, headers, task_id)
    assert snap["state"] == "done", snap
    types = [e["type"] for e in snap["events"]]
    assert "start" in types and "action_done" in types and types[-1] == "done"
    assert snap["summary"]["needs_attention"] is False

    assert client.get("/v1/models/status", headers=headers).json()["actions"]["det"]["state"] == "current"
    assert client.get("/v1/models/install/nope", headers=headers).status_code == 404


def test_events_stream_ends_with_done(client: TestClient):
    headers = _auth(client)
    task_id = client.post("/v1/models/install", json={}, headers=headers).json()["task_id"]
    _wait(client, headers, task_id)
    with client.stream("GET", f"/v1/models/install/{task_id}/events", headers=headers) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())
    frames = [f for f in text.split("\n\n") if f.strip()]
    assert frames[-1].startswith("event: done")
    last = json.loads(frames[-1].split("data: ", 1)[1])
    assert last["state"] == "done"
    assert any(f.startswith("event: progress") for f in frames)


def test_routes_require_the_token(client: TestClient):
    assert client.get("/v1/models/status").status_code in (401, 403)


def test_events_stream_accepts_the_query_token_because_eventsource_cannot_send_headers(client: TestClient):
    """Regression: with a header-only guard the browser's EventSource got 401, never saw "done",
    and the install button stayed on "Installing..." forever."""
    headers = _auth(client)
    task_id = client.post("/v1/models/install", json={}, headers=headers).json()["task_id"]
    _wait(client, headers, task_id)
    with client.stream("GET", f"/v1/models/install/{task_id}/events", params={"token": TEST_TOKEN}) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())
    assert "event: done" in text
    assert client.get(f"/v1/models/install/{task_id}/events").status_code == 401
    assert client.get(f"/v1/models/install/{task_id}/events", params={"token": "wrong"}).status_code == 401


def test_catalog_and_activate_write_the_settings_file(client: TestClient):
    headers = _auth(client)
    task_id = client.post("/v1/models/install", json={}, headers=headers).json()["task_id"]
    _wait(client, headers, task_id)
    cat = client.get("/v1/models/catalog", headers=headers).json()
    det = next(s for s in cat["stages"] if s["stage"] == "det")
    assert det["activatable"] is True and det["variants"][0]["units"][0]["formats"]["onnx"]["state"] == "current"
    assert cat["settings"]["yaml_path"]
    # activate: the yaml gains modules.det.model pointing at the installed file, and the catalog
    # reports it active; the response is the fresh catalog
    res = client.post("/v1/models/activate", json={"stage": "det", "model_key": "k", "format": "onnx"}, headers=headers)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["activated"] == {"key": "k", "path": "models/det/model.onnx", "format": "onnx"}
    import os
    written = yaml.safe_load(Path(os.environ["LM3_SETTINGS"]).read_text())
    assert written["modules"]["det"]["model"] == {"key": "k", "path": "models/det/model.onnx", "format": "onnx"}
    det = next(s for s in body["stages"] if s["stage"] == "det")
    assert det["active"]["matched"] is True and det["variants"][0]["units"][0]["formats"]["onnx"]["active"] is True
    # a format LM3 cannot run, or a model that is not installed, is refused and nothing is written
    before = Path(os.environ["LM3_SETTINGS"]).stat().st_mtime_ns
    assert client.post("/v1/models/activate", json={"stage": "det", "model_key": "k", "format": "coreml"}, headers=headers).status_code == 409
    assert client.post("/v1/models/activate", json={"stage": "det", "model_key": "nope", "format": "onnx"}, headers=headers).status_code in (404, 409)
    assert client.post("/v1/models/activate", json={"stage": "det"}, headers=headers).status_code == 400
    assert Path(os.environ["LM3_SETTINGS"]).stat().st_mtime_ns == before
