"""Compatibility cleanup -- plan section 4, Step 7, and the section 5 migration contract.

Plan reference: ``UNIFIED_RUNTIME_IMPLEMENTATION_PLAN_CONSENSUS.md`` -- Step 7 (L1773-1782),
section 5 (L1785-1806) and section 2.10 (the ``db_path`` -> ``active_db_path`` storage roles).

Step 7 is four sentences, and each one is a property this file pins:

* "Retain ``/v1/run/active`` for one declared transition release, with a **named release owning its
  removal**." -- a deprecation with no owner never happens, so the release name has to be reachable
  by a CLIENT (a header), by a MAINTAINER (``docs/DEPRECATIONS.md``) and by CI (one constant that
  both of those must agree with).
* "Audit actual external ``/v1/jobs`` consumers. Helper *definitions* in ``api.js`` are not
  consumers." -- so this file re-derives the in-tree evidence instead of trusting the register's
  prose, and pins the instrumentation that answers the half a grep cannot: whether anyone outside
  the tree calls them.
* "Keep ``LM3_settings_gg_watch.yaml`` as a manual operational artifact ... **no migration code
  deletes it**." -- pinned as an absence: no source file may so much as name it.
* "Retire the deprecated ``db_path`` projection in favor of ``active_db_path``." -- honoring the
  one-transition-release rule, which means the wire key must still be THERE and must still be
  correct, while nothing inside the server resolves through the legacy spelling any more.

Plus the section 5 consolidations that are testable from the server side: one settings resolver for
all settings routes, one ``/v1/runs``, and the rule that ``active`` means "the process is live" and
never "the run selected in the UI".

Nothing here binds a port, launches a process, loads a model or touches a GPU. Records are written
into a ``tmp_path`` deployment directory and the deployment lock -- when a test needs it held -- is
taken by the real POSIX adapter on that same tmp directory.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from leafmachine3.core.runtime import _types as T
from leafmachine3.core.runtime import records as R
from tests._contract_helpers import (
    RUN_RECORD_SPEC,
    Sandbox,
    assert_json_shape,
    bearer,
    capture_lm3_logs,
    flattened_app_routes,
    isolate_server_paths,
    reset_server_module_state,
)

REPO = Path(__file__).resolve().parents[1]
REGISTER = REPO / "docs" / "DEPRECATIONS.md"

NOW = "2026-08-28T12:00:00Z"
RUN_ID = "33333333-3333-4333-8333-333333333333"


# --------------------------------------------------------------------------------------------- #
# Fixtures and builders
# --------------------------------------------------------------------------------------------- #
@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Sandbox]:
    reset_server_module_state()
    box = isolate_server_paths(tmp_path, monkeypatch)
    monkeypatch.setenv("LM3_RUNTIME_DIR", str(tmp_path / "runtime"))
    # Pin the one-release compatibility path explicitly: the production default is now ON, while
    # these regressions deliberately exercise the old projection.
    monkeypatch.setenv("LM3_RUNTIME_V2", "0")
    from leafmachine3.server import app as app_mod
    from leafmachine3.server import metrics_api

    app_mod._LEGACY_JOBS_SEEN.clear()          # the audit latch is process-wide, like the log line
    metrics_api.invalidate_runtime_cache()
    try:
        yield box
    finally:
        app_mod._LEGACY_JOBS_SEEN.clear()
        metrics_api.invalidate_runtime_cache()
        reset_server_module_state()


@pytest.fixture
def v2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM3_RUNTIME_V2", "1")
    from leafmachine3.server import metrics_api

    metrics_api.invalidate_runtime_cache()


@pytest.fixture
def client(sandbox: Sandbox) -> Iterator[TestClient]:
    """A TestClient over a freshly built app.

    NOT entered as a context manager: ``__enter__`` runs the lifespan, which starts the job worker
    and the owner watchdog, and no test here wants either.
    """
    from leafmachine3.server.app import JobManager, create_app

    app = create_app(JobManager(sandbox.jobs_root / "managed"))
    yield TestClient(app)


@pytest.fixture
def deployment(sandbox: Sandbox) -> Path:
    from leafmachine3.server import metrics_api

    directory = metrics_api.deployment_dir()
    (directory / T.CHILDREN_DIRNAME).mkdir(parents=True, exist_ok=True)
    return directory


def _record(box: Sandbox) -> T.RuntimeRecord:
    run_dir = box.run_dir
    db = run_dir / f"{box.run_name}.sqlite"
    project = T.ProjectBlock(
        run_name=box.run_name,
        input_dirs=(str(box.input_dir),),
        artifact_dir=str(run_dir),
        active_state_dir=str(run_dir),
        active_db_path=str(db),
        archive_mode=T.ArchiveMode.IN_PLACE,
        archive_status=T.ArchiveStatus.NOT_APPLICABLE,
        archive_pointer_path=None,
        archived_db_path=str(db),
        run_dir=str(run_dir),
        log_path=str(run_dir / "logs" / "lm3.log"),
    )
    return T.RuntimeRecord(
        run_id=RUN_ID,
        activity=T.Activity.PIPELINE,
        activity_role=T.ActivityRole.ROOT,
        state=T.RunState.RUNNING,
        launcher=T.Launcher.CLI,
        pid=4321,
        process_started_at=1787932800.25,
        started_at=NOW,
        updated_at=NOW,
        deployment=T.DeploymentInfo(id="default", node="gpu042"),
        config=T.ConfigRef(path=str(box.settings_path), sha256="a" * 64),
        project=project,
    )


def publish_active(deployment_dir: Path, record: T.RuntimeRecord) -> Path:
    path = deployment_dir / T.ACTIVE_RECORD_FILENAME
    R.atomic_write_json(path, R.record_to_dict(record))
    return path


def hold_the_lease(deployment_dir: Path) -> Any:
    """Take the REAL deployment lock. Section 2.9: the OS lock decides occupancy, never the JSON."""
    from leafmachine3.core.runtime import lease as lease_module

    adapter = lease_module.lease_adapter(deployment_dir, deployment_key=deployment_dir.name)
    adapter.acquire()
    return adapter


# --------------------------------------------------------------------------------------------- #
# 1. The named release (Step 7: "a named release owning its removal")
# --------------------------------------------------------------------------------------------- #
def test_the_removal_release_is_one_constant_shared_by_code_and_the_register() -> None:
    """The release name is a constant, not a comment, because three places must agree on it.

    A header a client reads, a document a maintainer reads and a test CI reads all naming the same
    release is what makes the deprecation real. Spelling it three times is how a schedule quietly
    becomes three schedules, one of which is a promise nobody kept.
    """
    from leafmachine3.server.metrics_api import COMPAT_REMOVAL_RELEASE

    assert re.fullmatch(r"\d+\.\d+\.\d+", COMPAT_REMOVAL_RELEASE), "name a release, not 'later'"
    assert REGISTER.is_file(), "Step 7 requires docs/DEPRECATIONS.md to exist"
    text = REGISTER.read_text(encoding="utf-8")
    assert COMPAT_REMOVAL_RELEASE in text, (
        f"docs/DEPRECATIONS.md must name the release the code schedules ({COMPAT_REMOVAL_RELEASE})"
    )
    assert f"Scheduled for removal in {COMPAT_REMOVAL_RELEASE}" in text


def test_the_removal_release_is_ahead_of_the_release_that_announces_it() -> None:
    """One transition release means the removal cannot be the release the client is running.

    Deprecating and removing in the same version gives an integrator no window at all -- the point
    of the rule is that a client which upgrades once sees the warning, and only a client which
    upgrades twice sees the removal.
    """
    from leafmachine3 import __version__
    from leafmachine3.server.metrics_api import COMPAT_REMOVAL_RELEASE

    current = tuple(int(p) for p in __version__.split(".")[:3])
    removal = tuple(int(p) for p in COMPAT_REMOVAL_RELEASE.split(".")[:3])
    assert removal > current, (
        f"the removal release {COMPAT_REMOVAL_RELEASE} must be LATER than the release announcing "
        f"it ({__version__}); a same-release removal is not a transition release"
    )


def test_every_scheduled_entry_in_the_register_names_a_replacement() -> None:
    """A register entry with no replacement is an outage notice, not a deprecation."""
    text = REGISTER.read_text(encoding="utf-8")
    scheduled = text.split("## Scheduled for removal", 1)[1].split("## Under audit", 1)[0]
    entries = [block for block in re.split(r"\n### ", scheduled)[1:]]
    assert len(entries) >= 6, "the register lost entries; is this file still describing the tree?"
    for entry in entries:
        title = entry.splitlines()[0]
        assert re.search(r"Replaced by|replaced by|→|->", entry), (
            f"register entry {title!r} schedules a removal without naming what replaces it"
        )


def test_the_header_builder_names_the_release_and_sends_no_sunset_date() -> None:
    """RFC 8594's ``Sunset`` is an HTTP-DATE. This project ships releases, not dates.

    Inventing a date to satisfy a header would be a promise the server cannot keep, so the schedule
    rides in ``Warning`` (which clients surface to humans) and in ``X-LM3-Removed-In`` (which a
    migration script can read).
    """
    from leafmachine3.server.metrics_api import COMPAT_REMOVAL_RELEASE, deprecation_headers

    headers = deprecation_headers(what="GET /v1/thing", replacement="GET /v1/other")
    assert headers["Deprecation"] == "true"
    assert headers["X-LM3-Removed-In"] == COMPAT_REMOVAL_RELEASE
    assert COMPAT_REMOVAL_RELEASE in headers["Warning"]
    assert 'rel="successor-version"' in headers["Link"]
    assert "Sunset" not in headers
    assert "X-LM3-Deprecated-Fields" not in headers, "the whole route is going; no field list"

    fielded = deprecation_headers(what="the db_path field", replacement="active_db_path",
                                  successor=None, fields="db_path")
    assert fielded["X-LM3-Deprecated-Fields"] == "db_path"
    assert "Link" not in fielded


# --------------------------------------------------------------------------------------------- #
# 2. GET /v1/run/active is retained for exactly one transition release
# --------------------------------------------------------------------------------------------- #
def test_the_route_is_still_served(client: TestClient, v2: None) -> None:
    """Retained, not removed. Step 7 keeps it for one release; deleting it now would strand the
    desktop shell, which falls back to this route when ``GET /v1/runtime`` answers 404."""
    resp = client.get("/v1/run/active", headers=bearer())
    assert resp.status_code == 200
    assert_json_shape(resp.json(), RUN_RECORD_SPEC, where="GET /v1/run/active")


def test_the_projection_announces_its_removal_release_in_headers(
    client: TestClient, v2: None
) -> None:
    from leafmachine3.server.metrics_api import COMPAT_REMOVAL_RELEASE

    resp = client.get("/v1/run/active", headers=bearer())
    assert resp.headers["Deprecation"] == "true"
    assert resp.headers["X-LM3-Removed-In"] == COMPAT_REMOVAL_RELEASE
    assert "/v1/runtime" in resp.headers["Link"]
    assert COMPAT_REMOVAL_RELEASE in resp.headers["Warning"]
    # The metadata rides in headers ONLY: the body's key set is pinned key-for-key, so a
    # ``deprecated: true`` field would break the very clients the projection exists to serve.
    assert "deprecated" not in resp.json()


def test_the_flag_off_response_carries_no_deprecation_metadata(client: TestClient) -> None:
    """With ``LM3_RUNTIME_V2`` off there is no ``/v1/runtime`` to migrate TO, so announcing a
    removal would tell a client to move to a route that answers 404."""
    resp = client.get("/v1/run/active", headers=bearer())
    assert resp.status_code == 200
    for header in ("Deprecation", "X-LM3-Removed-In", "Warning", "X-LM3-Deprecated-Fields"):
        assert header not in resp.headers


# --------------------------------------------------------------------------------------------- #
# 3. Retiring the db_path projection in favor of active_db_path (section 2.10)
# --------------------------------------------------------------------------------------------- #
def test_the_wire_field_is_still_there_and_still_correct(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    """The one-transition-release rule cuts BOTH ways: retiring the name must not break the field.

    Dropping a key out of a compatibility projection is a harder break than the route removal it is
    meant to prepare for, because a client that reads ``db_path`` gets ``None`` instead of a 404 and
    silently tails nothing.
    """
    record = _record(sandbox)
    publish_active(deployment, record)
    adapter = hold_the_lease(deployment)
    try:
        body = client.get("/v1/run/active", headers=bearer()).json()
    finally:
        adapter.release()
    assert_json_shape(body, RUN_RECORD_SPEC, where="GET /v1/run/active")
    assert body["db_path"] == record.project.active_db_path, (
        "db_path is a PROJECTION of active_db_path; it may not drift from the role it projects"
    )


def test_the_server_resolves_the_storage_role_not_the_legacy_name(
    sandbox: Sandbox, deployment: Path, v2: None
) -> None:
    """``active_db_path()`` asks the registry for the ROLE, so the day the projection is deleted
    this function does not move. The legacy spelling survives in exactly one mapping."""
    from leafmachine3.server import metrics_api

    record = _record(sandbox)
    publish_active(deployment, record)
    adapter = hold_the_lease(deployment)
    try:
        metrics_api.invalidate_runtime_cache()
        resolved = metrics_api.active_db_path()
    finally:
        adapter.release()
    assert resolved is not None and str(resolved) == record.project.active_db_path
    # The registry reference publishes the ROLE name, never the projection.
    assert "active_db_path" in metrics_api._RUN_ATTR_FOR_ROLE
    assert metrics_api._RUN_ATTR_FOR_ROLE["active_db_path"] == "db_path"


def test_the_canonical_project_reference_uses_the_role_name(
    sandbox: Sandbox, deployment: Path, v2: None
) -> None:
    """Section 5's "one project-reference shape" must be spelled in the section 2.10 vocabulary.

    ``active_run_ref()`` is the reference every other server module is meant to resolve through, so
    it is the one shape that must not carry the deprecated projection forward: a new surface that
    copies it inherits the role name, not the name being retired.
    """
    from leafmachine3.server import metrics_api

    record = _record(sandbox)
    publish_active(deployment, record)
    adapter = hold_the_lease(deployment)
    try:
        metrics_api.invalidate_runtime_cache()
        ref = metrics_api.active_run_ref()
    finally:
        adapter.release()
    assert ref is not None
    assert ref["active_db_path"] == record.project.active_db_path
    assert "db_path" not in ref, "the canonical reference carries the role, not the projection"
    assert {"run_id", "run_name", "run_dir", "activity", "state"} <= set(ref)


def test_the_run_control_routes_survive_but_flag_the_field(
    client: TestClient, v2: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 5 KEEPS ``/run/start`` and ``/run/stop``; only ``db_path`` inside their body goes.

    So they answer with ``X-LM3-Deprecated-Fields`` rather than a bare ``Deprecation: true`` -- a
    client must not read "a field is going" as "stop calling this endpoint".
    """
    from leafmachine3.server import metrics_api

    monkeypatch.setattr(metrics_api, "start_run", lambda **kw: dict(metrics_api._IDLE_RECORD))
    monkeypatch.setattr(metrics_api, "stop_run", lambda grace: dict(metrics_api._IDLE_RECORD))

    for method, route in (("post", "/v1/run/start"), ("post", "/v1/run/stop")):
        resp = getattr(client, method)(route, headers=bearer(), json={})
        assert resp.status_code == 200, resp.text
        assert resp.headers["X-LM3-Deprecated-Fields"] == "db_path"
        assert "/v1/runtime" not in resp.headers.get("Link", ""), (
            "these routes are KEPT; pointing them at a successor would read as a route removal"
        )


def test_the_core_projection_warns_and_cannot_drift(sandbox: Sandbox) -> None:
    """``config_io.db_path()`` reads ``active_db_path`` off the object it is handed, so the two
    cannot disagree, and it says so to anyone still calling it."""
    from leafmachine3.core.runtime import config_io

    project = _record(sandbox).project
    with pytest.warns(DeprecationWarning, match="active_db_path"):
        assert config_io.db_path(project) == project.active_db_path
    assert config_io.db_path(project, warn=False) == project.active_db_path


# --------------------------------------------------------------------------------------------- #
# 4. The /v1/jobs audit (Step 7: a decision about external users, not in-tree call sites)
# --------------------------------------------------------------------------------------------- #
#: Files allowed to mention ``/v1/jobs`` at all. Everything here is a DEFINITION, a comment or a
#: test of the worker -- never a call. A new entry means a new consumer, which means the register's
#: recommendation has to be revisited before anything is scheduled.
_JOBS_MENTIONS_ALLOWED = {
    "leafmachine3/server/app.py",                 # the route definitions themselves
    "leafmachine3/server/ui/js/api.js",           # helper DEFINITIONS (Appendix B, C2)
    "leafmachine3/server/progress_api.py",        # a docstring naming the staged-job root
    "leafmachine3/server/results_api.py",         # a comment naming the staged-job root
    "tests/test_server_handshake.py",             # invariant 13, through the worker not the route
    "tests/test_compat_cleanup.py",               # this file
}


def _tracked_sources() -> list[Path]:
    roots = [REPO / "leafmachine3", REPO / "tests", REPO / "app"]
    files: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".js", ".mjs", ".html"} or not path.is_file():
                continue
            if "node_modules" in path.parts:
                continue
            files.append(path)
    return files


def test_the_legacy_jobs_api_still_has_no_in_tree_consumer() -> None:
    """Appendix B C2 re-derived, not quoted. Helper DEFINITIONS are not consumers.

    ``api.js`` defines ``getJob``, ``getJobResults`` and ``streamJobEvents`` and nothing calls them;
    the Run button uses ``POST /v1/run/start``. This test fails the moment that stops being true,
    which is the signal that the register's recommendation needs rewriting BEFORE a removal is
    scheduled -- not after.
    """
    offenders = []
    for path in _tracked_sources():
        rel = path.relative_to(REPO).as_posix()
        if "/v1/jobs" in path.read_text(encoding="utf-8", errors="ignore"):
            if rel not in _JOBS_MENTIONS_ALLOWED:
                offenders.append(rel)
    assert not offenders, (
        f"new /v1/jobs references in {offenders}: update docs/DEPRECATIONS.md's audit before "
        f"scheduling anything"
    )

    # And the three helpers are still definitions only.
    ui = REPO / "leafmachine3" / "server" / "ui" / "js"
    for helper in ("getJob", "getJobResults", "streamJobEvents"):
        callers = [p.relative_to(REPO).as_posix() for p in ui.rglob("*.js")
                   if p.name != "api.js" and f"{helper}(" in p.read_text(encoding="utf-8")]
        assert not callers, f"{helper} is now called from {callers}; it has a consumer"


def test_the_routes_are_not_deleted(client: TestClient) -> None:
    """Step 7 says removal is a DECISION about external users. This step does not make it."""
    paths = {getattr(r, "path", "") for r in client.app.routes}
    for route in ("/v1/jobs", "/v1/jobs/{jid}", "/v1/jobs/{jid}/events", "/v1/jobs/{jid}/results"):
        assert route in paths, f"{route} was deleted without an owner deciding to delete it"


def test_using_a_legacy_route_records_the_evidence_a_grep_cannot_produce(
    client: TestClient
) -> None:
    """The audit's unknowable half: whether an OPERATOR's script calls these.

    A source scan can only prove what the tree calls. The server therefore logs the first use of
    each legacy route, naming the User-Agent -- that line in a real user's log is what turns
    "probably nobody" into a fact before anything is deleted.
    """
    with capture_lm3_logs() as messages:
        first = client.get("/v1/jobs/nope", headers={**bearer(), "User-Agent": "curl/8.5.0"})
        second = client.get("/v1/jobs/nope", headers={**bearer(), "User-Agent": "curl/8.5.0"})
    assert first.status_code == 404 and second.status_code == 404, "unknown job is still a 404"
    noted = [m for m in messages if "legacy GET /v1/jobs/{jid}" in m]
    assert len(noted) == 1, (
        "once per route per process: /events is an SSE stream and /{jid} is polled, so a line per "
        f"call would bury the log it exists to inform -- got {len(noted)}"
    )
    assert "curl/8.5.0" in noted[0], "the User-Agent is the only thing identifying the caller"
    assert "DEPRECATIONS" in noted[0], "the log has to say where the decision is recorded"


def test_the_audit_log_never_carries_the_token(client: TestClient, sandbox: Sandbox) -> None:
    """Gate 13: the SSE routes accept ``?token=``, so the query string must never be logged."""
    from tests._contract_helpers import TEST_TOKEN

    with capture_lm3_logs() as messages:
        client.get(f"/v1/jobs/nope/results?token={TEST_TOKEN}", headers=bearer())
    assert any("legacy GET /v1/jobs/{jid}/results" in m for m in messages)
    assert not any(TEST_TOKEN in m for m in messages), "the audit line leaked the bearer secret"


# --------------------------------------------------------------------------------------------- #
# 5. LM3_settings_gg_watch.yaml survives untouched (Step 7: "no migration code deletes it")
# --------------------------------------------------------------------------------------------- #
def test_no_source_file_so_much_as_names_the_gg_watch_settings_file() -> None:
    """It is an operator's file, not the app's.

    ``LM3_settings_gg_watch.yaml`` is a settings file with an empty ``project.run_name`` -- an
    operator's workaround that forces ``_from_settings()`` to decline so the GUI follows whatever
    run is actually going. Step 7 keeps it until follow-active is VALIDATED, and the strongest
    guarantee that no migration deletes it is that no shipped code knows the name at all.
    """
    named = [p.relative_to(REPO).as_posix() for p in _tracked_sources()
             if "gg_watch" in p.read_text(encoding="utf-8", errors="ignore")
             and p.name != "test_compat_cleanup.py"]
    assert not named, (
        f"{named} references LM3_settings_gg_watch.yaml. Step 7: it is a MANUAL operational "
        f"artifact and no migration code deletes it."
    )
    assert "LM3_settings_gg_watch.yaml" in REGISTER.read_text(encoding="utf-8"), (
        "the register must say WHY it is kept, or the next cleanup deletes it"
    )


# --------------------------------------------------------------------------------------------- #
# 6. Section 5 consolidation: ONE settings resolver for all settings routes
# --------------------------------------------------------------------------------------------- #
#: ``(method, path, kwargs)`` for every settings route that names a file. ``/meta`` and
#: ``/defaults`` are excluded because they name no settings file at all, and ``/mkdir`` because it
#: creates a folder a user picked, which is not a settings path.
_SETTINGS_ROUTES = [
    ("get", "/v1/settings", {}),
    ("get", "/v1/settings/backups", {}),
    ("get", "/v1/settings/presets", {}),
    ("get", "/v1/settings/presets/nope", {}),
    ("post", "/v1/settings/validate", {"json": {"values": {}}}),
    ("post", "/v1/settings/restore", {"json": {"name": "nope.bak.yaml"}}),
    ("post", "/v1/settings/presets/nope", {"json": {"values": {}}}),
    ("post", "/v1/settings/presets/nope/apply", {}),
    ("post", "/v1/settings/browse", {"json": {}}),
    ("delete", "/v1/settings/presets/nope", {}),
]


@pytest.mark.parametrize("method,path,kwargs", _SETTINGS_ROUTES,
                         ids=[f"{m.upper()} {p}" for m, p, _ in _SETTINGS_ROUTES])
def test_every_settings_route_resolves_through_the_one_resolver(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, method: str, path: str, kwargs: dict
) -> None:
    """Section 5: "one settings resolver for all settings routes".

    Proven by REPLACING that resolver and watching each route follow it, which is the only way to
    catch a route that quietly grows its own resolution -- the split that once let the Settings tab
    edit one file while the run launcher started another from the same directory.

    The response status is deliberately not asserted: a missing preset is a 404 and a missing backup
    a 400, and both are fine. What matters is that the path was resolved in one place.
    """
    from leafmachine3.server import settings_api

    original = settings_api.settings_path
    seen: list[Any] = []

    def spy(explicit: Any = None) -> Path:
        seen.append(explicit)
        return original(explicit)

    monkeypatch.setattr(settings_api, "settings_path", spy)
    getattr(client, method)(path, headers=bearer(), **kwargs)
    assert seen, f"{method.upper()} {path} resolved a settings path without the one resolver"


def test_the_one_resolver_agrees_with_the_canonical_chain(sandbox: Sandbox) -> None:
    """It is a LAYER over ``paths.settings_path``, not a second answer.

    ``tests/test_settings_path_unification.py`` proves the five modules share one chain; this pins
    that the settings ROUTES sit on the same chain rather than beside it.
    """
    from leafmachine3.server import settings_api
    from leafmachine3.server.app import canonical_settings_path

    assert settings_api.settings_path() == canonical_settings_path()


# --------------------------------------------------------------------------------------------- #
# 7. Section 5 naming: `active` is liveness, never a UI selection
# --------------------------------------------------------------------------------------------- #
def test_exactly_one_route_answers_v1_runs_and_results_api_owns_it(client: TestClient) -> None:
    """Section 5: "one ``GET /v1/runs`` owned by ``results_api``". The progress duplicate is gone,
    so registration order no longer decides which shape the Results tab receives."""
    routes = [
        r for r in flattened_app_routes(client.app)
        if getattr(r, "path", None) == "/v1/runs"
    ]
    assert len(routes) == 1, f"{len(routes)} routes answer /v1/runs"
    assert routes[0].endpoint.__module__.rsplit(".", 1)[-1] == "results_api"


def test_liveness_and_selection_are_separate_fields(client: TestClient) -> None:
    """Section 5: "do not overload a single field named ``active``".

    ``active`` is "a process is live"; the run a user picked is ``selected_default`` on the wire and
    ``view.selected`` in the renderer. One field doing both jobs is how a GUI ends up offering Stop
    for a run that finished last week.
    """
    body = client.get("/v1/runs/-/selector", headers=bearer()).json()
    assert "active" in body and "selected_default" in body
    assert body["active"] is None, "nothing is running in this sandbox"
    # The kept listing carries no `active` at all: it is history, and history has no liveness.
    assert "active" not in client.get("/v1/runs", headers=bearer()).json()


def test_the_runtime_view_reports_liveness_from_the_lock(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    """The canonical surface answers the liveness question the projection cannot ask properly."""
    publish_active(deployment, _record(sandbox))
    adapter = hold_the_lease(deployment)
    try:
        from leafmachine3.server import metrics_api

        metrics_api.invalidate_runtime_cache()
        body = client.get("/v1/runtime", headers=bearer()).json()
    finally:
        adapter.release()
    assert body["active"]["occupied"] is True
    assert body["active"]["run_id"] == RUN_ID
    assert body["next_run_settings"]["applies_to"] == "next run", (
        "settings describe the NEXT run; conflating them with the running one is the same "
        "overloading section 5 forbids for `active`"
    )
