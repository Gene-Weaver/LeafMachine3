"""Endpoint-contract tests for the LM3 server HTTP surface.

Plan reference: ``UNIFIED_RUNTIME_IMPLEMENTATION_PLAN_CONSENSUS.md`` §4 Step 1 --
"Endpoint-contract tests for ``/v1/run/active``, ``/v1/run/start``, ``/v1/run/stop``,
``/v1/status``, ``/v1/runs``, ``/v1/settings``."

UNLIKE the characterization tests, these are meant to SURVIVE the unified-runtime refactor. They
pin the request/response contract that the Electron renderer and any other client depend on, so
Step 4 ("Registry-backed server and status") can prove it kept faith. The file is therefore split
in two:

* ``TestPreservedContract`` -- shapes and behaviors the plan does NOT change. A failure here after
  the refactor is a regression, full stop.
* ``TestContractThePlanChanges`` -- shapes the plan explicitly rewrites. Every test in it names the
  clause that will change it, so the person doing that step knows to edit the assertion instead of
  wondering whether they broke something.

Nothing here binds a port, spawns a pipeline, touches a GPU, loads a model, or reads the user's
real runtime state: ``fastapi.testclient.TestClient`` drives the ASGI app in-process, and
``tests/_contract_helpers.py`` redirects every path seam into ``tmp_path`` and stubs the one route
that would otherwise fork a real ``machine3``.
"""
from __future__ import annotations

import signal
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from tests._contract_helpers import (
    BOOL,
    HEALTHZ_SPEC,
    INT,
    RUN_RECORD_SPEC,
    RUN_SUMMARY_SPEC,
    RUNS_ENVELOPE_SPEC,
    SETTINGS_SPEC,
    STATUS_SPEC,
    STR,
    TEST_TOKEN,
    Sandbox,
    assert_every_item_shape,
    assert_json_shape,
    bearer,
    drain_reapers,
    flattened_app_routes,
    install_fake_spawn,
    isolate_server_paths,
    make_run_dir,
    reset_server_module_state,
)

#: Every route this file is the contract for, as (method, path). Used by the auth matrix so a new
#: route cannot be added to the contract without also being covered for authentication.
CONTRACT_ROUTES: tuple[tuple[str, str], ...] = (
    ("GET", "/v1/run/active"),
    ("POST", "/v1/run/start"),
    ("POST", "/v1/run/stop"),
    ("GET", "/v1/status"),
    ("GET", "/v1/runs"),
    ("GET", "/v1/settings"),
)

#: ``GET /healthz`` AFTER §4 Step 5b, which added "service, protocol version, instance ID,
#: deployment key and ownership mode" to the probe body and demoted ``pid`` on the wire (§2.5).
#: Built by SPREADING the shared ``HEALTHZ_SPEC`` rather than editing it: that helper is imported by
#: eight suites this file does not own, and spreading keeps the two in step either way -- if the
#: helper later grows the same keys, the same names resolve to the same types and this literal is a
#: no-op. The exact key set is the assertion that matters: /healthz is UNAUTHENTICATED, so a key
#: arriving here undeclared is a field nobody vetted for being a capability.
HEALTHZ_POST_5B_SPEC: dict[str, tuple] = {
    **HEALTHZ_SPEC,
    "service": STR,
    "protocol_version": INT,
    "instance_id": STR,
    "deployment_id": STR,
    "deployment_key": STR,
    "ownership_mode": STR,
    "pid_is_diagnostic_only": BOOL,
}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Sandbox]:
    """A tmp-dir world plus a clean slate of server module globals, before AND after."""
    reset_server_module_state()
    box = isolate_server_paths(tmp_path, monkeypatch)
    try:
        yield box
    finally:
        reset_server_module_state()


@pytest.fixture
def client(sandbox: Sandbox) -> Iterator[TestClient]:
    """A TestClient over a freshly built app whose JobManager lives inside the sandbox.

    The TestClient is NOT entered as a context manager on purpose: ``__enter__`` runs the app's
    lifespan, which starts the job worker task and the owner watchdog thread. A contract test wants
    the routing table and the handlers, not a background loop.
    """
    from leafmachine3.server.app import JobManager, create_app

    app = create_app(JobManager(sandbox.jobs_root / "managed"))
    yield TestClient(app)


@pytest.fixture
def fake_spawn(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Stub the process seam ``POST /v1/run/start`` would launch a real pipeline through.

    Also stubs ``_signal_group``: the stand-in child has a synthetic PID that no signal can reach,
    so without this ``stop_run`` would sit through its full SIGTERM grace and SIGKILL escalation.
    Releasing the child from inside the signal is what a real, well-behaved LM3 does.
    """
    from leafmachine3.server import metrics_api

    fake = install_fake_spawn(monkeypatch)
    sent: list[int] = []

    def _fake_signal_group(run: Any, sig: int) -> bool:
        sent.append(int(sig))
        for child in fake.calls:
            child.release(returncode=0)
        return True

    monkeypatch.setattr(metrics_api, "_signal_group", _fake_signal_group)
    fake.signals_sent = sent
    try:
        yield fake
    finally:
        drain_reapers(fake)


def start_a_run(client: TestClient, sandbox: Sandbox, **body: Any) -> Any:
    """POST /v1/run/start against the sandbox config, with the spawn already stubbed."""
    payload = {"config_path": str(sandbox.settings_path)}
    payload.update(body)
    return client.post("/v1/run/start", json=payload, headers=bearer())


# --------------------------------------------------------------------------- #
# The contract the plan preserves
# --------------------------------------------------------------------------- #
class TestPreservedContract:
    """Shapes and behaviors Step 4 must keep. A failure here after the refactor is a regression."""

    # -- auth (§2.5: control requires provenance; the token is the outer gate) ------------- #
    @pytest.mark.parametrize(("method", "path"), CONTRACT_ROUTES)
    def test_an_unauthenticated_request_is_rejected(self, client: TestClient,
                                                    method: str, path: str) -> None:
        """No credential at all -> 401, on every contract route, before any work happens."""
        resp = client.request(method, path)
        assert resp.status_code == 401, f"{method} {path} answered {resp.status_code} with no token"
        assert resp.json() == {"detail": "invalid or missing bearer token"}

    @pytest.mark.parametrize(("method", "path"), CONTRACT_ROUTES)
    def test_a_wrong_token_is_rejected(self, client: TestClient, method: str, path: str) -> None:
        resp = client.request(method, path, headers={"Authorization": "Bearer not-the-secret"})
        assert resp.status_code == 401

    @pytest.mark.parametrize(("method", "path"), CONTRACT_ROUTES)
    def test_the_token_must_be_a_bearer_header_not_a_query_parameter(
        self, client: TestClient, method: str, path: str
    ) -> None:
        """``?token=`` is for the SSE / media routes only.

        ``EventSource`` and ``<img src>`` cannot set a header, so those routes accept the secret on
        the query string; the six contract routes deliberately do not. Pinning that keeps a future
        refactor from widening the query-string exception to routes that never needed it.
        """
        resp = client.request(method, f"{path}?token={TEST_TOKEN}")
        assert resp.status_code == 401

    def test_a_bearer_header_is_how_a_client_authenticates_today(self, client: TestClient) -> None:
        """The positive half of the auth contract: the exact header form that works."""
        resp = client.get("/v1/run/active", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert resp.status_code == 200

    def test_healthz_is_deliberately_unauthenticated(self, client: TestClient) -> None:
        """It is the Electron shell's readiness probe, polled before it holds the secret."""
        resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == (Path(__file__).resolve().parents[1] / "VERSION").read_text().strip()

    def test_healthz_carries_the_step_1_resolved_path_diagnostics(
        self, client: TestClient, sandbox: Sandbox
    ) -> None:
        """§4 Step 1: "Expose resolved paths in startup logs and ``/healthz`` diagnostics".

        This is the outside-the-process half of the Step 1 exit gate -- "from any supported CWD
        every subsystem reports the same canonical settings path" -- so the block must name the
        settings file the sandbox actually configured, not a checkout-relative fallback. It is also
        the only place the contract states what the block may contain: paths and deployment
        identity, never the server token (§2.11's redaction rule), which is why an unauthenticated
        probe is allowed to carry it at all.
        """
        block = client.get("/healthz").json()["paths"]
        assert block["settings"] == str(sandbox.settings_path)
        # Flat and log-safe: a nested structure would break the startup-log line this shares with.
        assert all(isinstance(v, str) for v in block.values()), block
        assert TEST_TOKEN not in " ".join(block.values())

    # -- GET /v1/run/active --------------------------------------------------------------- #
    def test_run_active_idle_record_has_the_exact_documented_key_set(
        self, client: TestClient
    ) -> None:
        """"The key set never changes" (``metrics_api.active`` docstring), starting from idle.

        §5 keeps this route as a compatibility projection, so the projection has to reproduce every
        one of these keys.
        """
        resp = client.get("/v1/run/active", headers=bearer())
        assert resp.status_code == 200
        body = resp.json()
        assert_json_shape(body, RUN_RECORD_SPEC, where="GET /v1/run/active (idle)")
        assert body["active"] is False
        assert body["state"] == "idle"
        assert body["run_name"] is None
        assert body["input_dirs"] == [] and body["restart"] == [] and body["argv"] == []

    def test_run_active_key_set_is_identical_once_a_run_exists(
        self, client: TestClient, sandbox: Sandbox, fake_spawn: Any
    ) -> None:
        """The idle template and a live record must not drift apart: same keys, different values."""
        assert start_a_run(client, sandbox).status_code == 200
        body = client.get("/v1/run/active", headers=bearer()).json()
        assert_json_shape(body, RUN_RECORD_SPEC, where="GET /v1/run/active (running)")
        assert body["active"] is True
        assert body["state"] == "running"
        assert body["run_name"] == sandbox.run_name

    # -- POST /v1/run/start --------------------------------------------------------------- #
    def test_run_start_returns_the_run_record_and_launches_exactly_one_child(
        self, client: TestClient, sandbox: Sandbox, fake_spawn: Any
    ) -> None:
        resp = start_a_run(client, sandbox)
        assert resp.status_code == 200
        body = resp.json()
        assert_json_shape(body, RUN_RECORD_SPEC, where="POST /v1/run/start")

        assert body["state"] == "running" and body["active"] is True
        assert body["run_name"] == sandbox.run_name
        assert Path(body["config_path"]) == sandbox.settings_path
        assert Path(body["output_dir"]) == sandbox.output_dir
        assert Path(body["run_dir"]) == sandbox.run_dir
        assert Path(body["db_path"]) == sandbox.run_dir / f"{sandbox.run_name}.sqlite"
        assert Path(body["log_path"]) == sandbox.run_dir / "logs" / "lm3.log"
        assert Path(body["console_log"]) == sandbox.run_dir / "logs" / "console.log"
        assert body["input_dirs"] == [str(sandbox.input_dir)]
        assert body["restart"] == []
        assert body["argv"][-2:] == ["--config", str(sandbox.settings_path)]
        assert body["returncode"] is None and body["error"] is None
        assert body["stopped_by_user"] is False and body["adopted"] is False

        assert len(fake_spawn.calls) == 1, "exactly one child per accepted start"
        child = fake_spawn.calls[0]
        assert child.argv == body["argv"]
        assert child.kwargs["start_new_session"] is True, (
            "the run gets its own session so one killpg reaches the executor's spawn workers"
        )
        # The child must not inherit the server's shared secret (metrics_api.start_run).
        assert "LM3_SERVER_TOKEN" not in child.kwargs["env"]

    def test_run_start_maps_input_and_output_overrides_onto_the_cli(
        self, client: TestClient, sandbox: Sandbox, fake_spawn: Any, tmp_path: Path
    ) -> None:
        """``input_dir`` / ``output_dir`` are the same overrides ``machine3 --input/--output`` takes."""
        other_in, other_out = tmp_path / "other_in", tmp_path / "other_out"
        other_in.mkdir()
        resp = start_a_run(client, sandbox, input_dir=str(other_in), output_dir=str(other_out))
        assert resp.status_code == 200
        body = resp.json()
        assert body["input_dirs"] == [str(other_in)]
        assert Path(body["output_dir"]) == other_out
        assert body["argv"][-6:] == ["--config", str(sandbox.settings_path),
                                     "--input", str(other_in), "--output", str(other_out)]

    def test_run_start_rejects_an_unknown_body_field_with_400(self, client: TestClient) -> None:
        resp = client.post("/v1/run/start", json={"nope": 1, "also_nope": 2}, headers=bearer())
        assert resp.status_code == 400
        assert resp.json()["detail"] == "unknown field(s): ['also_nope', 'nope']"

    def test_run_start_rejects_an_unknown_restart_key_with_400(
        self, client: TestClient, sandbox: Sandbox, fake_spawn: Any
    ) -> None:
        resp = start_a_run(client, sandbox, restart=["not_a_stage"])
        assert resp.status_code == 400
        assert "unknown restart stage key(s)" in resp.json()["detail"]
        assert not fake_spawn.calls, "a rejected start must not launch anything"

    def test_run_start_rejects_a_missing_config_with_400(
        self, client: TestClient, sandbox: Sandbox, fake_spawn: Any
    ) -> None:
        resp = start_a_run(client, sandbox, config_path=str(sandbox.root / "absent.yaml"))
        assert resp.status_code == 400
        assert "config file not found" in resp.json()["detail"]
        assert not fake_spawn.calls

    def test_run_start_is_409_while_a_run_is_already_active(
        self, client: TestClient, sandbox: Sandbox, fake_spawn: Any
    ) -> None:
        """The busy refusal clients already code against -- §5 keeps ``409 on busy``."""
        assert start_a_run(client, sandbox).status_code == 200
        resp = start_a_run(client, sandbox)
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "a run is already active" in detail
        assert sandbox.run_name in detail
        assert len(fake_spawn.calls) == 1, "the losing start must not launch a second child"

    # -- POST /v1/run/stop ---------------------------------------------------------------- #
    def test_run_stop_without_an_active_run_is_409(self, client: TestClient) -> None:
        resp = client.post("/v1/run/stop", json={}, headers=bearer())
        assert resp.status_code == 409
        assert resp.json()["detail"] == "no active run to stop"

    def test_run_stop_signals_the_run_and_returns_the_finalized_record(
        self, client: TestClient, sandbox: Sandbox, fake_spawn: Any
    ) -> None:
        assert start_a_run(client, sandbox).status_code == 200
        resp = client.post("/v1/run/stop", json={"grace_s": 0}, headers=bearer())
        assert resp.status_code == 200
        body = resp.json()
        assert_json_shape(body, RUN_RECORD_SPEC, where="POST /v1/run/stop")
        assert fake_spawn.signals_sent[:1] == [int(signal.SIGTERM)], "SIGTERM first, always"
        assert body["stopped_by_user"] is True
        assert body["state"] in ("stopping", "done")
        assert body["run_name"] == sandbox.run_name

    # -- GET /v1/status ------------------------------------------------------------------- #
    def test_status_has_the_exact_key_set_for_the_configured_project(
        self, client: TestClient, sandbox: Sandbox
    ) -> None:
        """An idle app still describes the project the settings name (``_from_settings``)."""
        resp = client.get("/v1/status", headers=bearer())
        assert resp.status_code == 200
        body = resp.json()
        assert_json_shape(body, STATUS_SPEC, where="GET /v1/status")
        assert body["source"] == "settings"
        assert body["run_name"] == sandbox.run_name
        assert Path(body["run_path"]) == sandbox.run_dir
        assert body["ready"] is False, "no ledger has been written yet"
        assert set(body["provenance"]) == {"measured", "derived", "unknowable"}

    def test_status_key_set_is_the_same_when_no_run_can_be_resolved(
        self, client: TestClient, sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_empty_status`` and ``_build_status`` must stay key-for-key identical.

        The front end "must never have to test for a key's existence -- only for its value"
        (``progress_api._empty_status`` docstring), so the nulled snapshot is part of the contract.
        """
        blank = sandbox.root / "blank_settings.yaml"
        blank.write_text("version: 3\n", encoding="utf-8")
        monkeypatch.setenv("LM3_SETTINGS", str(blank))
        reset_server_module_state()

        body = client.get("/v1/status", headers=bearer()).json()
        assert_json_shape(body, STATUS_SPEC, where="GET /v1/status (nothing resolvable)")
        assert body["state"] == "idle"
        assert body["run_name"] is None and body["source"] is None
        assert body["modules"] == [] and body["workers"] == [] and body["recent"] == []

    def test_status_accepts_an_explicit_db_selector(
        self, client: TestClient, sandbox: Sandbox
    ) -> None:
        """``db=`` is the explicit override §1 invariant 7 preserves for a single request."""
        run_dir = make_run_dir(sandbox.output_dir, "picked_by_hand")
        db = run_dir / "picked_by_hand.sqlite"
        body = client.get(f"/v1/status?db={db}", headers=bearer()).json()
        assert_json_shape(body, STATUS_SPEC, where="GET /v1/status?db=")
        assert body["source"] == "query"
        assert body["run_name"] == "picked_by_hand"
        assert Path(body["db_path"]) == db

    # -- GET /v1/runs --------------------------------------------------------------------- #
    def test_runs_envelope_and_row_shape(self, client: TestClient, sandbox: Sandbox) -> None:
        make_run_dir(sandbox.output_dir, "run_alpha")
        make_run_dir(sandbox.output_dir, "run_beta")
        resp = client.get("/v1/runs", headers=bearer())
        assert resp.status_code == 200
        body = resp.json()
        assert_json_shape(body, RUNS_ENVELOPE_SPEC, where="GET /v1/runs")
        assert body["n"] == len(body["runs"]) == 2
        assert all(isinstance(r, str) for r in body["roots"])
        assert_every_item_shape(body["runs"], RUN_SUMMARY_SPEC, where="GET /v1/runs -> runs")
        assert {r["name"] for r in body["runs"]} == {"run_alpha", "run_beta"}
        # The stable `id` every other /v1/runs/{id}/... route keys off (`results_api.run_ref`).
        assert all(r["id"] for r in body["runs"])
        assert len({r["id"] for r in body["runs"]}) == 2

    def test_runs_refresh_query_parameter_is_accepted(
        self, client: TestClient, sandbox: Sandbox
    ) -> None:
        """``refresh=true`` bypasses the discovery TTL -- the Results tab's reload button."""
        assert client.get("/v1/runs", headers=bearer()).json()["n"] == 0
        make_run_dir(sandbox.output_dir, "appeared_later")
        assert client.get("/v1/runs?refresh=true", headers=bearer()).json()["n"] == 1

    # -- GET /v1/settings ----------------------------------------------------------------- #
    def test_settings_has_the_exact_key_set(self, client: TestClient, sandbox: Sandbox) -> None:
        resp = client.get("/v1/settings", headers=bearer())
        assert resp.status_code == 200
        body = resp.json()
        assert_json_shape(body, SETTINGS_SPEC, where="GET /v1/settings")
        assert Path(body["yaml_path"]) == sandbox.settings_path
        assert Path(body["dir"]) == sandbox.settings_path.parent
        assert body["exists"] is True
        assert body["error"] is None
        assert body["text_truncated"] is False

    def test_settings_reports_values_defaults_and_the_merge_of_the_two(
        self, client: TestClient, sandbox: Sandbox
    ) -> None:
        """``effective`` is defaults deep-merged with the file -- what LM3 will ACTUALLY run."""
        body = client.get("/v1/settings", headers=bearer()).json()
        assert body["values"]["project"]["run_name"] == sandbox.run_name
        assert body["effective"]["project"]["run_name"] == sandbox.run_name
        # A key that exists only in the defaults still reaches the UI through `effective`.
        defaults_only = set(body["defaults"]) - set(body["values"])
        assert defaults_only, "the sandbox config is deliberately partial"
        assert defaults_only <= set(body["effective"])
        assert body["text"].startswith("version: 3")

    def test_settings_rejects_a_non_yaml_path_with_400(self, client: TestClient,
                                                       sandbox: Sandbox) -> None:
        """The suffix check is a security boundary, not tidiness (``settings_api.settings_path``)."""
        resp = client.get("/v1/settings?yaml_path=/etc/passwd", headers=bearer())
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["ok"] is False
        assert "must name a" in detail["message"]

    def test_settings_reports_a_file_that_does_not_exist_yet_without_failing(
        self, client: TestClient, sandbox: Sandbox
    ) -> None:
        """A brand-new install must render, not 500: ``exists: false`` with defaults filled in."""
        absent = sandbox.root / "never_written.yaml"
        body = client.get(f"/v1/settings?yaml_path={absent}", headers=bearer()).json()
        assert_json_shape(body, SETTINGS_SPEC, where="GET /v1/settings (absent file)")
        assert body["exists"] is False
        assert body["values"] == {} and body["text"] is None
        assert body["mtime"] is None and body["size"] is None
        assert body["effective"] == body["defaults"]


# --------------------------------------------------------------------------- #
# The contract the plan deliberately changes
# --------------------------------------------------------------------------- #
class TestContractThePlanChanges:
    """Today's behavior, each test naming the plan clause that will rewrite it.

    When one of these fails during the refactor, the fix is to update the assertion to the new
    contract named in the docstring -- NOT to restore the old behavior.
    """

    def test_v1_runs_is_registered_once_and_owned_by_results_api(self, client: TestClient) -> None:
        """SATISFIED AT §4 Step 4: "Remove the duplicate progress-router ``GET /v1/runs``, keeping
        the ``results_api`` route, and delete the registration-order workaround at
        ``app.py:613-615``." (That citation is quoted verbatim from the plan; the workaround it
        names is the comment block above the ``include_router`` loop in ``app.create_app``, which
        now reads "ORDER CARRIES NO MEANING".)

        Also §5: "Consolidate: one ``GET /v1/runs`` owned by ``results_api``."

        Both halves have landed: ``progress_api`` no longer defines the path, and the router mount
        order in ``create_app`` is therefore free of meaning. So the assertion inverts -- a SECOND
        registration is now the defect, because a duplicate would silently restore an order-decided
        winner that no comment is guarding any more.
        """
        routes = [
            r for r in flattened_app_routes(client.app)
            if getattr(r, "path", None) == "/v1/runs"
        ]
        assert len(routes) == 1, (
            "the progress_api duplicate is deleted; two routers must not answer one path"
        )
        assert routes[0].endpoint.__module__.rsplit(".", 1)[-1] == "results_api", (
            "§5: the surviving GET /v1/runs is owned by results_api"
        )

    def test_v1_runs_body_is_the_results_api_shape_not_the_progress_api_one(
        self, client: TestClient, sandbox: Sandbox
    ) -> None:
        """CHANGES AT §4 Step 4 (duplicate ``GET /v1/runs`` removal).

        The behavioral half of the test above, so the duplicate cannot be resolved in the losing
        router's favor by accident. ``progress_api.list_runs`` returns a top-level ``active`` key
        and honors ``?limit=``; ``results_api.list_runs`` does neither.
        """
        for name in ("run_a", "run_b", "run_c"):
            make_run_dir(sandbox.output_dir, name)
        body = client.get("/v1/runs?limit=1", headers=bearer()).json()
        assert "active" not in body, "a top-level `active` key would mean progress_api answered"
        assert body["n"] == 3 and len(body["runs"]) == 3, "?limit= is progress_api's parameter"
        assert "id" in body["runs"][0], "results_api rows carry the stable id"
        assert "run_name" not in body["runs"][0], "`run_name` is progress_api's row shape"

    def test_v1_runtime_does_not_exist_yet(self, client: TestClient) -> None:
        """CHANGES AT §5 / §4 Step 4: "Add ``GET /v1/runtime`` -- canonical runtime, bounded
        activity tree, last-run, next-settings, diagnostics", returning
        ``{active, last, next_run_settings, server, diagnostics}``.

        Pinned as absent so the route's arrival is a deliberate, visible change.
        """
        assert client.get("/v1/runtime", headers=bearer()).status_code == 404

    def test_run_active_carries_no_deprecation_metadata_today(self, client: TestClient) -> None:
        """CHANGES AT §5: ``GET /v1/run/active`` is kept "temporarily" as a "compatibility
        projection, deprecation header", and §4 Step 4 "Keep ``GET /v1/run/active`` as a
        compatibility projection with deprecation metadata"."""
        resp = client.get("/v1/run/active", headers=bearer())
        assert resp.status_code == 200
        assert "deprecation" not in {k.lower() for k in resp.headers}
        assert "sunset" not in {k.lower() for k in resp.headers}

    def test_run_active_overloads_a_single_field_named_active(
        self, client: TestClient, sandbox: Sandbox, fake_spawn: Any
    ) -> None:
        """CHANGES AT §5: "Do not overload a single field named ``active`` to mean both 'the process
        is live' and 'the historical run selected in the UI'. Use ``runtime.active`` and
        ``view.selected``."

        Today one boolean carries the whole meaning, and there is no ``runtime`` / ``view``
        namespace to disambiguate it.
        """
        idle = client.get("/v1/run/active", headers=bearer()).json()
        assert idle["active"] is False and "runtime" not in idle and "view" not in idle
        assert start_a_run(client, sandbox).status_code == 200
        live = client.get("/v1/run/active", headers=bearer()).json()
        assert live["active"] is True and "runtime" not in live and "view" not in live

    def test_run_active_has_no_can_stop_field(self, client: TestClient) -> None:
        """CHANGES AT §2.5 / §4 Step 4: a restarted server "observes but cannot signal; it reports
        ``can_stop: false`` with a reason", and ``can_stop`` derives only from the §2.5 five-way
        match (live child handle, instance_id, run_id, PID, process creation time).

        Today the record instead exposes ``adopted``, and ``POST /v1/run/stop`` will happily
        synthesize a process group from a persisted record -- exactly what §2.5 removes.
        """
        body = client.get("/v1/run/active", headers=bearer()).json()
        assert "can_stop" not in body
        assert "adopted" in body
        assert "run_id" not in body, "§3.2 introduces the run_id this record has no notion of"

    def test_run_start_answers_before_the_child_has_reported_anything(
        self, client: TestClient, sandbox: Sandbox, fake_spawn: Any
    ) -> None:
        """CHANGES AT §2.4: today ``POST /v1/run/start`` "returns at :787 without waiting", so
        "a child that exits 75 ... has already been sent" the response. §2.4 replaces this with a
        launch handshake: the child writes one JSON status line over a control pipe, and the server
        answers 200 on ``acquired`` / 409 on ``busy``.

        Pinned by the stand-in child: it never writes a status line, never exits, and never gets a
        control descriptor -- and the endpoint still returns 200.
        """
        resp = start_a_run(client, sandbox)
        assert resp.status_code == 200
        child = fake_spawn.calls[0]
        assert "pass_fds" not in child.kwargs, "the POSIX half of the §2.4 handshake"
        assert "LM3_STATUS_FD" not in child.kwargs["env"]
        assert "LM3_STATUS_HANDLE" not in child.kwargs["env"]

    def test_run_start_creates_execution_owned_directories_before_acquisition(
        self, client: TestClient, sandbox: Sandbox, fake_spawn: Any
    ) -> None:
        """CHANGES AT §2.4 / §1 invariant 14: "Only after ``acquired`` may execution-owned
        directories be created", and "A losing (busy) launch performs no execution-owned mutation:
        no run directory, no project DB, ...". Prohibited for a losing launch is "anything under
        ``<output.dir>/<run_name>/`` ... the run's ``logs/``".

        Today the server creates ``<run_dir>/logs`` and opens ``logs/console.log`` BEFORE the child
        has said anything -- the mutation §2.4 moves behind the handshake.
        """
        logs_dir = sandbox.run_dir / "logs"
        assert not logs_dir.exists()
        assert start_a_run(client, sandbox).status_code == 200
        assert logs_dir.is_dir(), "the run's logs/ is created by the SERVER, pre-handshake"
        assert (logs_dir / "console.log").is_file()
        # And the child's stdout goes straight into it, rather than to a server-private request
        # log under the jobs root as §2.4 requires.
        console_fh = fake_spawn.calls[0].kwargs["stdout"]
        assert Path(console_fh.name) == logs_dir / "console.log"

    def test_healthz_demotes_the_server_pid_and_publishes_the_deployment_identity(
        self, client: TestClient
    ) -> None:
        """SATISFIED AT §2.5: ``/healthz``'s ``pid`` field "must be removed or explicitly demoted to
        diagnostic-only", because ``app.py:532-538`` documented it as how an attached client kills a
        server it does not own -- behavior §1 invariant 12 forbids. §4 Step 5b: "``/healthz``
        returning service, protocol version, instance ID, **deployment key**, and ownership mode".

        Step 5b has landed, so the assertion inverts. ``pid`` survives, but as a field that SAYS it
        authorizes nothing, and the identity fields are now required rather than forbidden.
        ``tests/test_healthz_identity.py`` pins their VALUES; what this pins is the exact key set,
        because the probe is unauthenticated -- an undeclared key here is a field nobody checked for
        being a capability.
        """
        body = client.get("/healthz").json()
        assert_json_shape(body, HEALTHZ_POST_5B_SPEC, where="GET /healthz")
        assert body["pid"] > 0
        assert body["pid_is_diagnostic_only"] is True, "§2.5: the demotion is stated ON THE WIRE"
        assert body["deployment_key"], "§4 Step 5b: the probe names its deployment"
        assert body["instance_id"], "§2.5 control authority is keyed to this instance"

    def test_a_disagreeing_legacy_settings_alias_is_refused_and_a_coherent_one_agrees(
        self, client: TestClient, sandbox: Sandbox, fake_spawn: Any,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REWRITTEN BY §4 Step 1, which has landed.

        The clause: §4 Step 1's migration bullet -- "honor ``LM3_SETTINGS_PATH`` alone with a
        deprecation warning; if it and ``LM3_SETTINGS`` resolve to the same file, use it and warn
        once; if they resolve differently, fail server startup with a precise split-brain error" --
        together with Step 1's exit gate "conflicting legacy variables stop startup", §3.1 row 1
        (one chain for the settings path), and §5's "one settings resolver for all settings routes".

        It used to assert the split: ``settings_api`` read ``$LM3_SETTINGS_PATH`` while
        ``metrics_api`` read ``$LM3_SETTINGS``, and the two endpoints happily disagreed about which
        config the app was on.
        """
        other = sandbox.root / "other_settings.yaml"
        other.write_text("version: 3\nproject:\n  run_name: from_settings_path\n", encoding="utf-8")
        monkeypatch.setenv("LM3_SETTINGS_PATH", str(other))     # now disagrees with $LM3_SETTINGS
        reset_server_module_state()

        # Half one: the deprecated alias names a different file, so the resolver refuses to guess.
        resp = client.get("/v1/settings", headers=bearer())
        assert resp.status_code == 400, f"split-brain settings answered {resp.status_code}"
        detail = resp.json()["detail"]
        assert detail["ok"] is False
        # "precise" is the plan's word: the error has to name both variables and both files. The
        # exact sentence is settings_api's to word, so it is deliberately not asserted here.
        for fragment in ("LM3_SETTINGS", "LM3_SETTINGS_PATH", str(sandbox.settings_path), str(other)):
            assert fragment in detail["message"], f"split-brain error does not name {fragment}"
        assert "yaml_path" not in resp.json(), "a refused resolution must not still name a file"

        started = start_a_run(client, sandbox, config_path=None)
        assert started.status_code == 400 and len(fake_spawn.calls) == 0, (
            "run/start must refuse too, rather than resolving through a second, disagreeing chain"
        )
        # Its message is vaguer than /v1/settings': metrics_api.default_config_path swallows the
        # PathsError into a warning (metrics_api.py:130-144), so start_run answers the generic
        # "no LM3_settings.yaml found" (metrics_api.py:692). Recorded, not asserted.

        # Half two: drop the alias and the two routes name the SAME file -- what §5's single
        # resolver buys, and what the old split made impossible.
        monkeypatch.delenv("LM3_SETTINGS_PATH")
        reset_server_module_state()

        shown = client.get("/v1/settings", headers=bearer()).json()
        assert Path(shown["yaml_path"]) == sandbox.settings_path

        started = start_a_run(client, sandbox, config_path=None)
        assert started.status_code == 200
        assert Path(started.json()["config_path"]) == sandbox.settings_path
