"""Plan gate 13 -- the raw bearer token appears in no log, and the bootstrap is ``no-store``.

Gate 13 was stated in section 2.11 and section 8 but owned by no step in the implementation
sequence, so it was implemented nowhere. Revision 13 assigns it to Step 5b; these tests are its
evidence.

Why it matters beyond tidiness: ``_server_token()`` used to print the generated secret at WARNING.
On a cluster that lands the bearer token in Slurm job output and retained container logs, where it
long outlives the allocation.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from tests._contract_helpers import (
    Sandbox, capture_lm3_logs, isolate_server_paths, reset_server_module_state,
)

TOKEN_ENV = "LM3_SERVER_TOKEN"


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Sandbox]:
    reset_server_module_state()
    box = isolate_server_paths(tmp_path, monkeypatch)
    try:
        yield box
    finally:
        reset_server_module_state()


def test_a_generated_token_is_never_written_to_a_log(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    from leafmachine3.server import app

    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with capture_lm3_logs() as messages:
        token = app._server_token()

    assert token, "a token should have been generated"
    captured = "\n".join(messages)
    assert captured, "captured nothing -- the assertion below would pass vacuously"
    assert token not in captured, "the raw bearer token was logged"
    assert TOKEN_ENV in captured, "the log should still tell the user how to set it"


def test_the_whole_server_startup_never_logs_the_token(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not just _server_token(): nothing reached during create_app may print it either."""
    from leafmachine3.server.app import JobManager, create_app

    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with capture_lm3_logs() as messages:
        app = create_app(JobManager(sandbox.jobs_root / "managed"))
        client = TestClient(app)
        for ep in ("/healthz", "/v1/status", "/v1/runs", "/v1/settings", "/"):
            client.get(ep, headers={"Authorization": f"Bearer {os.environ[TOKEN_ENV]}"})

    token = os.environ[TOKEN_ENV]
    captured = "\n".join(messages)
    assert captured, "captured nothing -- the assertion below would pass vacuously"
    assert token not in captured, "the bearer token reached a log during startup or a request"


def test_the_token_bearing_bootstrap_is_served_no_store(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``no-cache`` STORES the response and revalidates; only ``no-store`` keeps it off disk."""
    from leafmachine3.server.app import JobManager, create_app

    client = TestClient(create_app(JobManager(sandbox.jobs_root / "managed")))
    response = client.get("/")
    if response.status_code == 404:
        pytest.skip("no UI bundle in this checkout")

    cache_control = response.headers.get("cache-control", "")
    assert "no-store" in cache_control, f"bootstrap Cache-Control was {cache_control!r}"


def test_static_assets_may_still_revalidate(sandbox: Sandbox) -> None:
    """The gate is about the token-bearing document, not about making every asset uncacheable."""
    from leafmachine3.server.app import JobManager, create_app

    client = TestClient(create_app(JobManager(sandbox.jobs_root / "managed")))
    response = client.get("/js/api.js")
    if response.status_code == 404:
        pytest.skip("no UI bundle in this checkout")
    assert "no-store" not in response.headers.get("cache-control", "")


def test_redact_token_scrubs_the_live_secret(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    from leafmachine3.server import app

    monkeypatch.setenv(TOKEN_ENV, "s3cret-value")
    message = "connection failed with Authorization: Bearer s3cret-value"
    assert "s3cret-value" not in app.redact_token(message)
    assert "***redacted***" in app.redact_token(message)


def test_no_source_line_logs_the_token_variable_directly() -> None:
    """A static backstop: a future edit that re-adds `log.*(..., token)` fails here."""
    source = Path("leafmachine3/server/app.py").read_text(encoding="utf-8")
    # Only an ARGUMENT counts: a message that merely says the word "token" is fine, and one of
    # them is the deliberate "refusing to embed the LM3 server token" warning.
    offenders = [line.strip() for line in source.splitlines()
                 if re.search(r'log\.\w+\([^)]*",\s*[^)]*\btoken\b', line)
                 and "redact" not in line]
    assert not offenders, f"a log call takes the token as an argument: {offenders}"
