"""Step 5b: server identity on ``/healthz``, and the private connection descriptor.

Three things are pinned here, and each one has a failure that is otherwise invisible:

* **The deployment key on the wire.** ``/healthz`` is how every client verifies it is talking to
  its OWN deployment (plan section 2.11). If the field is missing, wrong, or not the canonical key,
  a desktop shell attaches to a server for a different deployment and the two share one lease
  without ever saying so.
* **Python and JavaScript agreeing on that key.** ``leafmachine3.core.paths`` and ``app/main.js``
  derive it independently, and a disagreement sends the two halves of the desktop app to two
  runtime directories -- a mismatch that happens BEFORE any later test could observe it. The golden
  vectors are the contract; this file runs the JavaScript half of them.
* **``connection.private.json``** (section 2.12): user-only from its first byte, replaced rather
  than corrected on a restart, and deleted on shutdown ONLY by the instance that owns it -- so a
  dying predecessor can never delete the descriptor its successor just published.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from leafmachine3.core import paths
from tests._contract_helpers import (
    Sandbox, capture_lm3_logs, isolate_server_paths, reset_server_module_state,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VECTORS = REPO_ROOT / "tests" / "golden" / "deployment_key_vectors.json"


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Sandbox]:
    """An isolated deployment with its own runtime directory and a clean identity cache."""
    from leafmachine3.server import app as app_mod

    reset_server_module_state()
    box = isolate_server_paths(tmp_path, monkeypatch)
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(paths.ENV_RUNTIME_DIR, str(runtime))
    monkeypatch.delenv(app_mod.ENV_INSTANCE_ID, raising=False)
    monkeypatch.delenv(app_mod.ENV_BIND_HOST, raising=False)
    monkeypatch.delenv(app_mod.ENV_BIND_PORT, raising=False)
    app_mod.reset_instance_id_cache()
    app_mod.reset_path_caches()
    try:
        yield box
    finally:
        app_mod.reset_instance_id_cache()
        reset_server_module_state()


@pytest.fixture
def client(sandbox: Sandbox) -> Iterator[TestClient]:
    from leafmachine3.server.app import JobManager, create_app

    # A CONTEXT MANAGER on purpose: the lifespan is what publishes and retracts the connection
    # descriptor, and a bare TestClient never runs it.
    with TestClient(create_app(JobManager(sandbox.jobs_root / "managed"))) as c:
        yield c


def descriptor_path() -> Path:
    from leafmachine3.server import app as app_mod

    return app_mod.connection_private_path(dict(os.environ))


# --------------------------------------------------------------------------- #
# /healthz identity (section 2.11, Step 5b)
# --------------------------------------------------------------------------- #
def test_healthz_names_the_service_protocol_instance_deployment_and_ownership(
    client: TestClient
) -> None:
    """The five fields Step 5b requires, plus the identity a client verifies against."""
    from leafmachine3.server import app as app_mod

    body = client.get("/healthz").json()
    assert body["service"] == app_mod.SERVICE_NAME == "leafmachine3"
    assert body["protocol_version"] == app_mod.HEALTH_PROTOCOL_VERSION
    assert body["instance_id"] == app_mod.server_instance_id()
    assert body["deployment_key"] == paths.deployment_key(dict(os.environ))
    assert body["deployment_id"] == paths.raw_deployment_id(dict(os.environ))
    assert body["ownership_mode"] in {"owned", "orphaned", "independent"}
    # Kept, and demoted in the body itself rather than only in a comment.
    assert body["pid"] == os.getpid()
    assert body["pid_is_diagnostic_only"] is True


def test_the_advertised_deployment_key_is_the_canonical_one_for_a_named_deployment(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The RAW value must never be what a client compares -- it can contain a path separator."""
    from leafmachine3.server import app as app_mod
    from leafmachine3.server.app import JobManager, create_app

    monkeypatch.setenv(paths.ENV_DEPLOYMENT_ID, "Lab Bench 2")
    monkeypatch.setenv(paths.ENV_PORT, "8801")
    app_mod.reset_path_caches()
    with TestClient(create_app(JobManager(sandbox.jobs_root / "named"))) as c:
        body = c.get("/healthz").json()
    assert body["deployment_id"] == "Lab Bench 2"
    assert body["deployment_key"] == "lab-bench-2-7791367d"     # the golden vector, verbatim
    assert "/" not in body["deployment_key"] and " " not in body["deployment_key"]


def test_a_broken_deployment_id_is_reported_rather_than_crashing_the_probe(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """A readiness probe that 500s tells a client nothing about WHY it cannot attach."""
    from leafmachine3.server import app as app_mod

    monkeypatch.setenv(paths.ENV_DEPLOYMENT_ID, "   ")
    identity = app_mod.deployment_identity()
    assert identity["deployment_key"] == ""
    assert "empty or whitespace-only" in identity["error"]


def test_a_spawning_parent_can_pin_the_instance_id_and_a_forged_one_is_ignored(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 2.11's expected-instance-ID handshake, from the server's side."""
    from leafmachine3.server import app as app_mod

    monkeypatch.setenv(app_mod.ENV_INSTANCE_ID, "shell-minted-0001")
    app_mod.reset_instance_id_cache()
    assert app_mod.server_instance_id() == "shell-minted-0001"

    # Narrow on purpose: the value reaches a JSON body and a log line.
    monkeypatch.setenv(app_mod.ENV_INSTANCE_ID, "no spaces or ; allowed")
    app_mod.reset_instance_id_cache()
    supplied = app_mod.server_instance_id()
    assert supplied != "no spaces or ; allowed"
    assert len(supplied) >= 8


def test_the_instance_id_is_stable_and_is_the_one_control_authority_uses(
    sandbox: Sandbox
) -> None:
    """Two IDs for one process would let a client verify one thing and the server enforce another."""
    from leafmachine3.server import app as app_mod
    from leafmachine3.server import metrics_api

    app_mod.reset_instance_id_cache()
    first = app_mod.server_instance_id()
    assert app_mod.server_instance_id() == first, "the identity must not move under a client"
    assert first == metrics_api.SERVER_INSTANCE_ID


def test_ownership_mode_reports_who_claims_this_server(monkeypatch: pytest.MonkeyPatch) -> None:
    from leafmachine3.server import app as app_mod

    monkeypatch.delenv(app_mod._OWNER_ENV, raising=False)
    assert app_mod.ownership_mode() == "independent"
    monkeypatch.setenv(app_mod._OWNER_ENV, str(os.getpid()))
    assert app_mod.ownership_mode() == "owned"
    monkeypatch.setenv(app_mod._OWNER_ENV, "999999999")        # long dead, or never existed
    assert app_mod.ownership_mode() == "orphaned"
    monkeypatch.setenv(app_mod._OWNER_ENV, "not-a-pid")
    assert app_mod.ownership_mode() == "independent"


def test_healthz_carries_no_token_anywhere_in_its_body(client: TestClient) -> None:
    """Gate 13: it is UNAUTHENTICATED, so a secret in it is a secret given away."""
    from tests._contract_helpers import TEST_TOKEN

    raw = client.get("/healthz").text
    assert TEST_TOKEN not in raw


# --------------------------------------------------------------------------- #
# Python <-> JavaScript agreement (section 2.1)
# --------------------------------------------------------------------------- #
def test_the_golden_vectors_pin_the_python_side() -> None:
    payload = json.loads(VECTORS.read_text(encoding="utf-8"))
    for vec in payload["vectors"]:
        raw = vec["effective_raw"]
        assert paths.ascii_slug(raw) == vec["slug"], vec["name"]
        assert paths.canonical_deployment_key(raw) == vec["canonical"], vec["name"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_electron_shell_reproduces_every_golden_vector() -> None:
    """The JavaScript half of the contract, run from the Python suite.

    Without this, the two implementations are only ever checked in two places that never meet, and
    the failure they guard against -- a key that differs between the shell and the server -- has no
    symptom until a user has two runtime directories and no explanation.
    """
    proc = subprocess.run(
        ["node", "--test", "test/deployment_key.test.js"],
        cwd=str(REPO_ROOT / "app"), capture_output=True, text=True, timeout=180, check=False,
    )
    assert proc.returncode == 0, f"the Electron deployment key disagrees:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_electron_ownership_rules_hold() -> None:
    """Invariants 9/10/12 as the shell implements them: no GUI, no LM3 server, no real port."""
    proc = subprocess.run(
        ["node", "--test", "test/ownership.test.js"],
        cwd=str(REPO_ROOT / "app"), capture_output=True, text=True, timeout=300, check=False,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


# --------------------------------------------------------------------------- #
# connection.private.json (section 2.12)
# --------------------------------------------------------------------------- #
def test_the_descriptor_is_published_on_startup_and_retracted_on_shutdown(
    sandbox: Sandbox
) -> None:
    from leafmachine3.server import app as app_mod
    from leafmachine3.server.app import JobManager, create_app
    from tests._contract_helpers import TEST_TOKEN

    target = descriptor_path()
    assert not target.exists()
    with TestClient(create_app(JobManager(sandbox.jobs_root / "d"))) as c:
        c.get("/healthz")
        assert target.is_file(), "Electron cannot authenticate without this file"
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["service"] == "leafmachine3"
        assert payload["schema_version"] == app_mod.CONNECTION_SCHEMA_VERSION
        assert payload["token"] == TEST_TOKEN
        assert payload["instance_id"] == app_mod.server_instance_id()
        assert payload["deployment_key"] == paths.deployment_key(dict(os.environ))
        assert payload["pid"] == os.getpid()
        assert payload["host"] and isinstance(payload["port"], int)
    assert not target.exists(), "a stopping server must retract its own descriptor"


def test_the_descriptor_is_user_only_from_its_first_byte(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Written broadly and chmod'ed afterward leaves a window, and a window is all it takes."""
    from leafmachine3.server import app as app_mod

    modes: list[int] = []
    real_open = os.open

    def recording(path: str, flags: int, mode: int = 0o777, **kwargs: object) -> int:
        if str(path).endswith(".tmp") and (flags & os.O_CREAT):
            modes.append(mode)
            assert flags & os.O_EXCL, "a predictable temp name is a symlink attack"
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(os, "open", recording)
    written = app_mod.write_connection_descriptor()
    assert written is not None
    assert modes == [0o600], f"the temp file was created with {[oct(m) for m in modes]}"
    assert stat.S_IMODE(written.stat().st_mode) == 0o600
    # And nothing is left behind for the next start to trip over.
    assert [p.name for p in written.parent.glob("*.tmp")] == []


def test_a_restart_rotates_the_token_and_the_instance_id(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exclusive-create AT the final path would work exactly once; a restart must republish."""
    from leafmachine3.server import app as app_mod

    monkeypatch.setenv("LM3_SERVER_TOKEN", "token-of-the-first-server")
    monkeypatch.setenv(app_mod.ENV_INSTANCE_ID, "instance-first-0001")
    app_mod.reset_instance_id_cache()
    first = app_mod.write_connection_descriptor()
    before = json.loads(first.read_text(encoding="utf-8"))

    monkeypatch.setenv("LM3_SERVER_TOKEN", "token-of-the-second-server")
    monkeypatch.setenv(app_mod.ENV_INSTANCE_ID, "instance-second-002")
    app_mod.reset_instance_id_cache()
    app_mod.write_connection_descriptor()
    after = json.loads(first.read_text(encoding="utf-8"))

    assert before["token"] != after["token"]
    assert after["token"] == "token-of-the-second-server"
    assert after["instance_id"] == "instance-second-002"
    assert stat.S_IMODE(first.stat().st_mode) == 0o600


def test_a_predecessors_late_cleanup_never_deletes_its_successors_descriptor(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 2.12 step 5, and the reason it is stated: shutdown is not instantaneous."""
    from leafmachine3.server import app as app_mod

    monkeypatch.setenv(app_mod.ENV_INSTANCE_ID, "instance-successor")
    app_mod.reset_instance_id_cache()
    target = app_mod.write_connection_descriptor()
    assert target.is_file()

    # The predecessor, finally getting around to its cleanup:
    assert app_mod.remove_connection_descriptor(instance_id="instance-predecessor") is False
    assert target.is_file(), "the live server's descriptor was deleted by a dead one"

    assert app_mod.remove_connection_descriptor() is True
    assert not target.exists()


def test_a_stale_descriptor_from_a_crashed_server_is_replaced_not_refused(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    from leafmachine3.server import app as app_mod

    target = descriptor_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"instance_id": "the-crashed-one", "token": "stale"}),
                      encoding="utf-8")
    monkeypatch.setenv("LM3_SERVER_TOKEN", "fresh-token")
    monkeypatch.setenv(app_mod.ENV_INSTANCE_ID, "instance-fresh-01")
    app_mod.reset_instance_id_cache()

    assert app_mod.write_connection_descriptor() == target
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["token"] == "fresh-token"
    assert payload["instance_id"] == "instance-fresh-01"


def test_publishing_a_descriptor_never_raises_and_never_logs_the_token(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A server that cannot publish a descriptor still serves -- the browser bootstrap is complete
    authentication on its own (section 2.11) -- and gate 13 says the path is logged, not the secret.
    """
    from leafmachine3.server import app as app_mod

    monkeypatch.setenv("LM3_SERVER_TOKEN", "supersecret-token-value")
    app_mod.reset_instance_id_cache()
    with capture_lm3_logs() as records:
        written = app_mod.write_connection_descriptor()
    assert written is not None
    joined = "\n".join(records)
    assert "supersecret-token-value" not in joined
    assert str(written) in joined, "a user must still be told WHERE their token is"

    # An unusable runtime directory is a warning, not a crash.
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv(paths.ENV_RUNTIME_DIR, str(blocker))
    assert app_mod.write_connection_descriptor() is None


def test_the_windows_acl_branch_is_implemented_and_driven_from_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IMPLEMENTED BUT NOT NATIVELY VALIDATED (section 1.1): this asserts the CALL, not the ACL.

    POSIX mode bits are not a substitute on Windows (section 2.12), so the branch has to exist and
    has to be reachable; whether the object manager honors it is a Windows qualification item.
    """
    from leafmachine3.server import app as app_mod

    target = tmp_path / "connection.private.json.tmp"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("USERNAME", "labuser")
    monkeypatch.setenv("USERDOMAIN", "LAB")
    seen: list[list[str]] = []

    assert app_mod._restrict_to_current_user(
        target, os_name="nt", runner=lambda cmd: (seen.append(cmd), 0)[1]) is True
    assert seen[0][0] == "icacls"
    assert "/inheritance:r" in seen[0]
    assert "LAB\\labuser:(F)" in seen[0]

    # A refusal is reported, never silently treated as success.
    assert app_mod._restrict_to_current_user(target, os_name="nt", runner=lambda cmd: 5) is False
    # On POSIX there is nothing to do and the mode bits already did it.
    assert app_mod._restrict_to_current_user(target, os_name="posix") is True


# --------------------------------------------------------------------------- #
# Gate 13's redaction half now has a production call site
# --------------------------------------------------------------------------- #
def test_a_job_error_message_carrying_the_token_is_redacted_before_anyone_sees_it(
    sandbox: Sandbox
) -> None:
    """``redact_token`` used to be defined and never called. This is the chokepoint it needed."""
    from leafmachine3.server.app import JobManager
    from tests._contract_helpers import TEST_TOKEN

    manager = JobManager(sandbox.jobs_root / "redact")
    job = manager.create(files=[])
    manager.mark_error(job.id, f"child failed: LM3_SERVER_TOKEN={TEST_TOKEN} was rejected")
    reported = manager.status(job.id)["error"]
    assert TEST_TOKEN not in reported
    assert "***redacted***" in reported


def test_the_healthz_path_diagnostics_are_redacted_too(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate 13 names ``/healthz`` diagnostics explicitly, and the block is unauthenticated."""
    from leafmachine3.server import app as app_mod
    from tests._contract_helpers import TEST_TOKEN

    monkeypatch.setattr(
        paths, "describe_resolved_paths",
        lambda **kwargs: {"settings": f"/tmp/{TEST_TOKEN}/LM3_settings.yaml"})
    block = app_mod.path_diagnostics()
    assert TEST_TOKEN not in json.dumps(block)
    assert "***redacted***" in block["settings"]


# --------------------------------------------------------------------------- #
# Section 2.1's port rule, from the entry point that has to enforce it
# --------------------------------------------------------------------------- #
def test_a_named_deployment_without_a_port_refuses_to_serve(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Gate 41. Silently defaulting to 8765 is the collision the rule exists to prevent."""
    from leafmachine3.server import app as app_mod

    monkeypatch.setenv(paths.ENV_DEPLOYMENT_ID, "gpu1")
    monkeypatch.delenv(paths.ENV_PORT, raising=False)
    calls: list[tuple] = []
    monkeypatch.setattr(app_mod, "serve", lambda *a, **k: calls.append((a, k)))

    assert app_mod.main(["--config", str(Path(sys.executable))]) == 2
    assert calls == [], "it must not start on a colliding port"
    assert "must set LM3_PORT" in capsys.readouterr().err
