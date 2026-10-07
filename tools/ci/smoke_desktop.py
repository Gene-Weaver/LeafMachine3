"""Linux CI smoke: real packaged Electron, frozen uv backend, UI load and ownership-safe shutdown.

Runs under Xvfb; --no-sandbox is confined to this headless CI test.
Logs are retained in a temporary directory when a check fails.
"""

import json
import os
import secrets
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[2]
package = root / "app/dist/linux-unpacked/leafmachine3"
contract = json.loads((root / "app/desktop-contract.json").read_text())
scratch = Path(tempfile.mkdtemp(prefix="lm3-uv-gui-smoke-"))
env = {k: v for k, v in os.environ.items() if not k.startswith(("LM3_", "ELECTRON_", "UV_"))}
for directory in ("config", "cache", "data", "runtime"):
    (scratch / directory).mkdir()
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
env.update(
    LM3_ROOT=str(root),
    LM3_EXTRA="cpu",
    LM3_PORT=str(port),
    LM3_DEPLOYMENT_ID=scratch.name,
    LM3_RUNTIME_DIR=str(scratch / "runtime"),
    LM3_SERVER_JOBS=str(scratch / "jobs"),
    LM3_SETTINGS=str(root / "LM3_settings.yaml"),
    LM3_HARDWARE=str(scratch / "hardware.yaml"),
    LM3_MODELS_DIR=str(scratch / "models"),
    LM3_SERVER_TOKEN=secrets.token_hex(32),
    XDG_CONFIG_HOME=str(scratch / "config"),
    XDG_CACHE_HOME=str(scratch / "cache"),
    LM3_STARTUP_GATE="0",
)
url = f"http://127.0.0.1:{port}"


def get(endpoint):
    request = urllib.request.Request(
        url + endpoint, headers={"Authorization": f"Bearer {env['LM3_SERVER_TOKEN']}"}
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return json.load(response)


def until(predicate, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = predicate()
            if result:
                return result
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"smoke timed out; logs in {scratch}")


def quiet():
    try:
        get("/healthz")
        return False
    except Exception:
        return True


def launch(label):
    log = (scratch / f"{label}.log").open("w")
    return subprocess.Popen(
        ["xvfb-run", "-a", str(package), "--no-sandbox"],
        cwd=root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def stop(process):
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


backend = desktop = None
try:
    # Launch from a bad active environment. The desktop must disregard it and enable the gate.
    env.update(
        VIRTUAL_ENV="/unrelated",
        CONDA_PREFIX="/unrelated",
        UV_PROJECT_ENVIRONMENT="/unrelated",
        UV_NO_SYNC="1",
        PYTHONPATH="/unrelated",
    )
    desktop = launch("owned")
    health = until(lambda: get("/healthz"))
    report = get("/healthz/doctor")
    assert report["ready"] and all(
        c["status"] == "ok" for c in report["checks"] if c["num"] in (1, 2, 3, 4)
    ), report
    assert report["environment"]["python"] == contract["python"], report
    assert report["environment"]["venv_uv"] == contract["uv"], report
    assert report["environment"]["contract_sha256"] == contract["backend_contract_sha256"], report
    assert str(root / ".venv") in report["checks"][0]["detail"], report
    until(
        lambda: "server spawned via uv.lock" in (scratch / "owned.log").read_text()
        and 'GET /v1/settings HTTP/1.1" 200' in (scratch / "owned.log").read_text()
    )
    print(
        "PASS: packaged desktop spawned uv-locked Python/backend, verified it and loaded the UI", flush=True
    )
    stop(desktop)
    desktop = None
    until(quiet, timeout=20)
    print("PASS: closing the desktop stopped its own backend", flush=True)
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX", "UV_PROJECT_ENVIRONMENT", "UV_NO_SYNC", "PYTHONPATH"):
        env.pop(key, None)
    env["LM3_STARTUP_GATE"] = "1"
    log = (scratch / "independent.log").open("w")
    backend = subprocess.Popen(
        [
            str(root / ".venv/bin/uv"),
            "run",
            "--frozen",
            "--no-sync",
            "lm3",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    until(lambda: get("/healthz"))
    desktop = launch("attached")
    until(
        lambda: "server attached" in (scratch / "attached.log").read_text()
        and 'GET /v1/settings HTTP/1.1" 200' in (scratch / "independent.log").read_text()
    )
    stop(desktop)
    desktop = None
    assert backend.poll() is None and get("/healthz")["service"] == "leafmachine3"
    print("PASS: closing an attached desktop left the independent uv server running", flush=True)
    print("Logs:", scratch, flush=True)
finally:
    stop(desktop)
    stop(backend)
