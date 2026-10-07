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


# --------------------------------------------------------------------------- #
# The access log -- the one gate-13 channel no test used to watch
# --------------------------------------------------------------------------- #
# The token has to ride the query string for the two clients that cannot set a header (the
# EventSource streams in api.js, and Electron's first `win.loadURL(.../?token=...)`), and uvicorn's
# AccessFormatter rebuilds the request line from the RAW query string. Every SSE connect and every
# window load therefore used to write the bearer token to the server's stdout -- which under
# `lm3 serve` in an allocation IS the Slurm job output file section 2.11 names. TestClient never
# goes through uvicorn's HTTP protocol, so nothing above observes this; these drive the real
# formatter with the real record shape instead. No socket is opened.
UVICORN_ACCESS_FMT = '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'


def _emit_access_record(path: str) -> str:
    """Log one uvicorn access record for ``path`` and return what the access formatter produced.

    The format string and the argument tuple are copied from uvicorn's own protocol implementations
    (``h11_impl``/``httptools_impl``): a rewrite that patched ``record.msg`` instead of
    ``record.args`` would pass a laxer test and still leak here.
    """
    import io
    import logging

    from uvicorn.logging import AccessFormatter

    logger = logging.getLogger("uvicorn.access")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(AccessFormatter(UVICORN_ACCESS_FMT, use_colors=False))
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        logger.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:52814", "GET", path, "1.1", 200)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    return stream.getvalue()


@pytest.fixture
def access_logger_clean() -> Iterator[None]:
    """Leave ``uvicorn.access`` exactly as it was found -- filters are process-global state."""
    import logging

    logger = logging.getLogger("uvicorn.access")
    before = list(logger.filters)
    try:
        yield
    finally:
        logger.filters = before


def test_the_access_log_never_shows_a_token_on_the_query_string(
    sandbox: Sandbox, access_logger_clean: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from leafmachine3.server.app import JobManager, create_app

    pytest.importorskip("uvicorn")
    sentinel = "SENTINEL-bearer-value-do-not-log"
    monkeypatch.setenv(TOKEN_ENV, sentinel)
    create_app(JobManager(sandbox.jobs_root / "managed"))     # installs the filter

    for path in (f"/v1/status/stream?token={sentinel}",       # api.js EventSource
                 f"/?token={sentinel}",                       # app/main.js win.loadURL
                 f"/v1/logs/stream?run_id=abc&token={sentinel}"):
        line = _emit_access_record(path)
        assert line.strip(), "captured nothing -- the assertion below would pass vacuously"
        assert sentinel not in line, f"the bearer token reached uvicorn's access log: {line!r}"
        assert "***redacted***" in line, f"nothing was redacted out of {line!r}"

    # ...and the log is still useful: the route, method and status must survive.
    line = _emit_access_record(f"/v1/status/stream?token={sentinel}")
    assert "/v1/status/stream" in line and "GET" in line and "200" in line, line


def test_the_access_log_redacts_a_token_the_client_url_encoded(
    sandbox: Sandbox, access_logger_clean: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A literal string replace is not enough: a user-set token may reach the log percent-encoded."""
    from leafmachine3.server.app import JobManager, create_app

    pytest.importorskip("uvicorn")
    monkeypatch.setenv(TOKEN_ENV, "pa ss/word+value")
    create_app(JobManager(sandbox.jobs_root / "managed"))

    line = _emit_access_record("/v1/status/stream?token=pa%20ss%2Fword%2Bvalue")
    assert line.strip()
    assert "ss%2Fword" not in line, f"an encoded token survived the access-log filter: {line!r}"
    assert "***redacted***" in line


def test_ordinary_access_records_are_left_alone(
    sandbox: Sandbox, access_logger_clean: None
) -> None:
    """The filter is a scrubber, not a rewriter: a token-free request line is untouched."""
    from leafmachine3.server.app import JobManager, create_app

    pytest.importorskip("uvicorn")
    create_app(JobManager(sandbox.jobs_root / "managed"))
    line = _emit_access_record("/v1/runs?limit=25")
    assert "/v1/runs?limit=25" in line
    assert "***redacted***" not in line


def test_the_access_log_filter_degrades_to_a_no_op_on_an_unknown_record_shape(
    sandbox: Sandbox, access_logger_clean: None
) -> None:
    """A future uvicorn arg-shape change must not raise inside logging on every request."""
    import logging

    from leafmachine3.server.app import JobManager, create_app

    create_app(JobManager(sandbox.jobs_root / "managed"))
    logger = logging.getLogger("uvicorn.access")
    shapes = ((), ("only-one",), ("a", "b", None, "1.1", 200), ["list", "style"], "a-bare-string",
              {"mapping": "style"})
    for args in shapes:
        record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, "%s", None, None)
        record.args = args                 # set after construction: LogRecord unwraps mappings
        for flt in logger.filters:
            assert flt.filter(record) is True
        assert record.args == args, "an unrecognized arg shape must be left untouched"


def test_installing_the_access_log_filter_twice_installs_one_filter(
    sandbox: Sandbox, access_logger_clean: None
) -> None:
    import logging

    from leafmachine3.server import app

    logging.getLogger("uvicorn.access").filters = []
    assert app.install_access_log_redaction() is True
    assert app.install_access_log_redaction() is False
    installed = [f for f in logging.getLogger("uvicorn.access").filters
                 if isinstance(f, app._AccessLogTokenRedactor)]
    assert len(installed) == 1


def test_the_real_uvicorn_request_line_builder_is_what_gets_scrubbed(
    sandbox: Sandbox, access_logger_clean: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close the last gap to production: build the path argument with uvicorn's OWN code.

    ``get_path_with_query_string()`` percent-quotes the path and then appends the query string
    verbatim -- which is exactly why the secret reaches the record unencoded and why the filter has
    to exist. The tests above hand-build that string, so they would keep passing if uvicorn stopped
    doing it; this one asserts the leak is still real before asserting it is closed.
    """
    pytest.importorskip("uvicorn")
    from uvicorn.protocols.utils import get_path_with_query_string

    from leafmachine3.server.app import JobManager, create_app

    sentinel = "SENTINEL-from-the-asgi-scope"
    monkeypatch.setenv(TOKEN_ENV, sentinel)
    create_app(JobManager(sandbox.jobs_root / "managed"))

    # The scope an EventSource connect produces (api.js puts the token on every stream URL, and
    # re-puts it on every automatic reconnect).
    request_path = get_path_with_query_string(
        {"path": "/v1/status/stream", "query_string": f"token={sentinel}&run_id=abc".encode("ascii")})
    assert sentinel in request_path, "uvicorn no longer inlines the raw query string -- retune this"

    line = _emit_access_record(request_path)
    assert line.strip(), "captured nothing -- the assertions below would pass vacuously"
    assert sentinel not in line, f"the bearer token reached uvicorn's access log: {line!r}"
    assert "?token=***redacted***&run_id=abc" in line, line
