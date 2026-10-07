"""``GET /v1/runtime`` and control authority -- plan section 4, Step 4.

Plan reference: ``UNIFIED_RUNTIME_IMPLEMENTATION_PLAN_CONSENSUS.md`` -- section 2.5 (control
authority is handle ownership, as amended by revision 15), section 2.6 (activity-aware UI),
section 2.9 (schema skew fails safe), section 3.2 (record schemas, the bounded child summary,
writer ownership) and section 5 (the API migration contract).

What is pinned here, in the plan's own words:

* "Add ``GET /v1/runtime`` returning ``{active, last, next_run_settings, server, diagnostics}``
  with the bounded activity tree."
* "Keep ``GET /v1/run/active`` as a compatibility projection with deprecation metadata."
* "``can_stop`` derives only from section 2.5: a retained live child handle for this ``run_id``,
  launched by this ``instance_id``. No PID comparison, no creation-time comparison."
* "``POST /v1/run/stop`` rejects observer-only runs with 409/403 and never synthesizes a process
  group from registry JSON."
* "**Delete the re-adoption machinery** that handle ownership makes dead: ``_adopt()``,
  ``_pid_alive()`` and its ``psutil`` / ``os.kill(pid, 0)`` fallback, and ``_Run.persist()``'s
  ``create_time`` plus the on-disk state file."
* Section 2.9: "if held -> report ``active: true, compatible: false``; disable Start and all control
  actions; do not interpret unknown fields."

Everything is behind ``LM3_RUNTIME_V2`` (default ON since cutover), and the first section is the
explicit-flag-off regression: with ``LM3_RUNTIME_V2=0`` there is no ``/v1/runtime`` route, no
deprecation header, and ``GET /v1/run/active`` answers exactly what it answered before.

Nothing here binds a port, launches a process, loads a model or touches a GPU. Registry records are
written directly into a ``tmp_path`` deployment directory and the deployment lock is taken -- when a
test needs it held -- by the real POSIX adapter on that same tmp directory.
"""
from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest
from fastapi.testclient import TestClient

from leafmachine3.core.runtime import _types as T
from leafmachine3.core.runtime import records as R
from tests._contract_helpers import (
    RUN_RECORD_SPEC,
    Sandbox,
    assert_json_shape,
    bearer,
    isolate_server_paths,
    reset_server_module_state,
)

NOW = "2026-08-28T12:00:00Z"
ROOT_RUN_ID = "11111111-1111-4111-8111-111111111111"
CHILD_RUN_ID = "22222222-2222-4222-8222-222222222222"


# --------------------------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------------------------- #
@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Sandbox]:
    """The shared tmp-dir world, plus a per-test deployment runtime directory.

    ``LM3_RUNTIME_DIR`` names the BASE (section 3.1), so the deployment directory this test writes
    records into is ``<base>/<canonical deployment key>`` -- exactly what
    ``metrics_api.deployment_dir()`` resolves. Pointing the base at ``tmp_path`` is what keeps the
    suite off the developer's real registry even though conftest has already isolated it once.
    """
    reset_server_module_state()
    box = isolate_server_paths(tmp_path, monkeypatch)
    monkeypatch.setenv("LM3_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("LM3_RUNTIME_V2", "0")
    from leafmachine3.server import metrics_api

    metrics_api.invalidate_runtime_cache()
    try:
        yield box
    finally:
        metrics_api.invalidate_runtime_cache()
        reset_server_module_state()


@pytest.fixture
def v2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM3_RUNTIME_V2", "1")
    from leafmachine3.server import metrics_api

    metrics_api.invalidate_runtime_cache()


@pytest.fixture
def client(sandbox: Sandbox) -> Iterator[TestClient]:
    """A TestClient over a freshly built app. Not entered as a context manager: ``__enter__`` runs
    the lifespan, which starts the job worker and the owner watchdog, and no test here wants them.
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


# --------------------------------------------------------------------------------------------- #
# Record builders (the same shapes tests/test_runtime_records.py uses)
# --------------------------------------------------------------------------------------------- #
def _project(tmp_path: Path, run_name: str = "contract_run") -> T.ProjectBlock:
    run_dir = tmp_path / "out" / run_name
    db = run_dir / f"{run_name}.sqlite"
    return T.ProjectBlock(
        run_name=run_name,
        input_dirs=(str(tmp_path / "in"),),
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


def _root(tmp_path: Path, **overrides: Any) -> T.RuntimeRecord:
    base: dict = dict(
        run_id=ROOT_RUN_ID,
        activity=T.Activity.PIPELINE,
        activity_role=T.ActivityRole.ROOT,
        state=T.RunState.RUNNING,
        launcher=T.Launcher.CLI,
        pid=4321,
        process_started_at=1787932800.25,
        started_at=NOW,
        updated_at=NOW,
        deployment=T.DeploymentInfo(id="default", node="gpu042"),
        config=T.ConfigRef(path=str(tmp_path / "cwd" / "LM3_settings.yaml"), sha256="a" * 64),
        project=_project(tmp_path),
    )
    base.update(overrides)
    return T.RuntimeRecord(**base)


def _child(tmp_path: Path, **overrides: Any) -> T.RuntimeRecord:
    base: dict = dict(
        run_id=CHILD_RUN_ID,
        activity=T.Activity.CALIBRATION_PIPELINE,
        activity_role=T.ActivityRole.CHILD,
        state=T.RunState.RUNNING,
        launcher=T.Launcher.PYTHON,
        pid=4322,
        process_started_at=1787932801.25,
        started_at=NOW,
        updated_at=NOW,
        parent_run_id=ROOT_RUN_ID,
        deployment=T.DeploymentInfo(id="default", node="gpu042"),
        project=_project(tmp_path, run_name=T.CALIBRATION_RUN_NAME),
    )
    base.update(overrides)
    return T.RuntimeRecord(**base)


def publish_active(deployment_dir: Path, record: T.RuntimeRecord) -> Path:
    path = deployment_dir / T.ACTIVE_RECORD_FILENAME
    R.atomic_write_json(path, R.record_to_dict(record))
    return path


def publish_child(deployment_dir: Path, record: T.RuntimeRecord) -> Path:
    children = deployment_dir / T.CHILDREN_DIRNAME
    children.mkdir(parents=True, exist_ok=True)
    path = children / f"{record.run_id}{T.CHILD_RECORD_SUFFIX}"
    R.atomic_write_json(path, R.record_to_dict(record))
    return path


def publish_last(deployment_dir: Path, record: T.RuntimeRecord) -> Path:
    path = deployment_dir / T.LAST_RECORD_FILENAME
    R.atomic_write_json(path, R.record_to_dict(record))
    return path


def hold_the_lease(deployment_dir: Path) -> Any:
    """Take the REAL deployment lock, so ``read_runtime`` reports the deployment as occupied.

    The lock is what section 2.9 says decides occupancy -- "the OS lock still decides whether the
    deployment is occupied" -- so a test that only wrote JSON would be testing the wrong half.
    """
    from leafmachine3.core.runtime import lease as lease_module

    adapter = lease_module.lease_adapter(deployment_dir, deployment_key=deployment_dir.name)
    adapter.acquire()
    return adapter


class _StubHandle:
    """A stand-in ``_ManagedChild``: the only thing section 2.5 lets authorize a stop."""

    def __init__(self, *, run_id: str, instance_id: str, alive: bool = True,
                 kind: str = "pipeline") -> None:
        self.run_id = run_id
        self.instance_id = instance_id
        self.kind = kind
        self._alive = alive
        self.signals: list[int] = []
        self.pid = 2 ** 31 - 21
        self.pgid = self.pid

    @property
    def alive(self) -> bool:
        return self._alive

    def signal_group(self, sig: int) -> bool:
        self.signals.append(int(sig))
        self._alive = False
        return True

    def release_job(self) -> None:
        pass


def install_launched_run(run_id: str = ROOT_RUN_ID, *, alive: bool = True,
                         instance_id: Optional[str] = None,
                         state: str = "running") -> Any:
    """Pretend this server launched ``run_id`` and still holds the handle."""
    from leafmachine3.server import metrics_api

    handle = _StubHandle(run_id=run_id,
                         instance_id=instance_id or metrics_api.SERVER_INSTANCE_ID,
                         alive=alive)
    run = metrics_api._Run(run_name="contract_run", run_dir="", db_path="", log_path="")
    run.child = handle
    run.run_id = run_id
    run.pid = handle.pid
    run.pgid = handle.pgid
    run.state = state
    metrics_api._RUN = run
    return handle


# --------------------------------------------------------------------------------------------- #
# 1. Flag off means unchanged
# --------------------------------------------------------------------------------------------- #
def test_with_the_flag_off_the_runtime_route_answers_404(client: TestClient) -> None:
    """Step 4 ships behind ``LM3_RUNTIME_V2`` like Step 3 did.

    A 404 and an empty 200 are different claims: 404 says "this build cannot tell you what is
    running", and with nothing publishing records that is the only true answer. An empty 200 would
    say "nothing is running" while a CLI run was going.
    """
    assert client.get("/v1/runtime", headers=bearer()).status_code == 404


def test_the_flag_is_read_per_request_not_at_router_build_time(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One flag, one reader, one answer -- on THIS app object, without rebuilding it.

    The route used to be registered inside ``if runtime_v2():`` at router-build time, so whether
    ``/v1/runtime`` existed depended on what the environment looked like at the instant ``create_app``
    ran, while every other reader in the module (``active``, ``_can_stop``, ``start_run``) consulted
    the flag per request. A server built a moment too early then served the v2 record from
    ``/v1/run/active`` and 404'd on ``/v1/runtime`` -- one feature, two answers.
    """
    from leafmachine3.server import metrics_api

    assert client.get("/v1/runtime", headers=bearer()).status_code == 404
    monkeypatch.setenv("LM3_RUNTIME_V2", "1")
    metrics_api.invalidate_runtime_cache()
    assert client.get("/v1/runtime", headers=bearer()).status_code == 200


def test_with_the_flag_off_run_active_is_byte_for_byte_the_old_contract(client: TestClient) -> None:
    resp = client.get("/v1/run/active", headers=bearer())
    assert resp.status_code == 200
    assert_json_shape(resp.json(), RUN_RECORD_SPEC, where="GET /v1/run/active")
    assert "deprecation" not in {k.lower() for k in resp.headers}
    assert "link" not in {k.lower() for k in resp.headers}


def test_with_the_flag_off_a_published_record_is_not_projected(
    client: TestClient, sandbox: Sandbox, deployment: Path
) -> None:
    """The registry is not consulted at all with the flag off -- there is nothing there to consult,
    and reading it would be the behavior change the flag exists to prevent."""
    publish_active(deployment, _root(sandbox.root))
    assert client.get("/v1/run/active", headers=bearer()).json()["active"] is False


# --------------------------------------------------------------------------------------------- #
# 2. The re-adoption machinery is GONE, not bypassed
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["_adopt", "_pid_alive", "_ADOPT_TRIED"])
def test_the_re_adoption_machinery_no_longer_exists(name: str) -> None:
    """Step 4: "Delete the re-adoption machinery that handle ownership (section 2.5) makes dead".

    Asserted as ABSENCE rather than as behavior, because the hazard is that somebody calls it
    again: ``_pid_alive`` reconstructed liveness from a PID, and its ``psutil`` branch compared a
    process creation time that is ``0.0`` on any host without psutil -- a check that looks rigorous
    and is vacuous, which revision 15 calls worse than no check because it is trusted.

    The SOURCE is what is asserted on, not only ``hasattr``: ``tests/_contract_helpers.py`` still
    assigns ``metrics_api._ADOPT_TRIED = True`` in its shared reset fixture (a belt-and-braces guard
    from when adoption existed), which CREATES the attribute on the module object in any session
    that imports the helper. Deleting that line belongs to the owner of that file; meanwhile the
    thing this test is actually defending -- that no code in this module defines or calls the
    machinery -- is a property of the module's text, so it is read from the module's text.
    """
    import re

    from leafmachine3.server import metrics_api

    source = Path(metrics_api.__file__).read_text(encoding="utf-8")
    defined = re.search(rf"^(?:def {re.escape(name)}\(|{re.escape(name)}(?:: [^=]+)? *=)",
                        source, re.MULTILINE)
    assert defined is None, f"{name} is defined again in metrics_api.py"
    # A reference in prose ("the machinery that used to do it") is fine and deliberate; a CALL is
    # not, and a call is what a bare name followed by "(" looks like outside a comment.
    called = [line for line in source.splitlines()
              if f"{name}(" in line and not line.lstrip().startswith("#")
              and "``" not in line]
    assert not called, f"{name} is called again: {called}"
    if name != "_ADOPT_TRIED":                         # see the docstring
        assert not hasattr(metrics_api, name)


def test_a_run_record_no_longer_carries_a_process_creation_time() -> None:
    from leafmachine3.server import metrics_api

    run = metrics_api._Run()
    assert not hasattr(run, "create_time")
    assert not hasattr(run, "persist")


def test_the_legacy_adoption_state_file_is_never_written_and_is_purged(
    client: TestClient, sandbox: Sandbox
) -> None:
    """The on-disk state file "whose only stated purpose is 'so a RESTARTED server can re-adopt a
    still-running LM3'" is deleted -- and a leftover from an older build is removed on sight, so
    the absence of the mechanism is visible on the filesystem too.
    """
    from leafmachine3.server import metrics_api

    stale = metrics_api._state_path()
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps({"pid": 12345, "create_time": 1.0, "state": "running"}),
                     encoding="utf-8")
    metrics_api._LEGACY_STATE_PURGED = False

    assert client.get("/v1/run/active", headers=bearer()).json()["active"] is False
    assert not stale.exists(), "a PID in a JSON file must not survive to be re-read"


# --------------------------------------------------------------------------------------------- #
# 3. GET /v1/runtime -- shape, and the bounded activity tree
# --------------------------------------------------------------------------------------------- #
def test_runtime_has_exactly_the_five_top_level_keys(client: TestClient, v2: None) -> None:
    body = client.get("/v1/runtime", headers=bearer()).json()
    assert set(body) == {"active", "last", "next_run_settings", "server", "diagnostics"}


def test_an_idle_deployment_reports_active_null(client: TestClient, v2: None,
                                                deployment: Path) -> None:
    body = client.get("/v1/runtime", headers=bearer()).json()
    assert body["active"] is None
    assert body["diagnostics"]["lease_occupied"] is False
    assert body["diagnostics"]["readable"] is True


def test_a_live_root_is_reported_from_the_registry_not_from_server_state(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    """The whole point of Step 4: a run this server did not launch is visible anyway.

    Nothing installs a ``_Run`` here. The server has launched nothing, holds nothing, and still
    describes the run -- because observation comes from the registry (section 2.5).
    """
    publish_active(deployment, _root(sandbox.root))
    adapter = hold_the_lease(deployment)
    try:
        body = client.get("/v1/runtime", headers=bearer()).json()
    finally:
        adapter.release()
    active = body["active"]
    assert active["occupied"] is True and active["compatible"] is True
    assert active["run_id"] == ROOT_RUN_ID
    assert active["activity"] == "pipeline"
    assert active["state"] == "running"
    assert active["record"]["project"]["run_name"] == "contract_run"
    assert active["can_stop"] is False
    assert "did not launch" in active["can_stop_reason"]


def test_the_activity_tree_is_bounded_to_two_child_objects(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    """Section 3.2: ``active.json`` carries a ``current_child`` and a ``last_child`` SUMMARY, "so
    ``active.json`` stays bounded however many children a root launches over its life". The API
    inherits that: two objects, never a list that grows with the number of children.
    """
    child = _child(sandbox.root)
    summary = T.ChildSummary(run_id=CHILD_RUN_ID, activity=T.Activity.CALIBRATION_PIPELINE,
                             state=T.RunState.STARTING, started_at=NOW,
                             run_name=T.CALIBRATION_RUN_NAME,
                             run_dir=str(sandbox.root / "out" / T.CALIBRATION_RUN_NAME))
    publish_child(deployment, child)
    publish_active(deployment, _root(sandbox.root, current_child=summary))
    adapter = hold_the_lease(deployment)
    try:
        active = client.get("/v1/runtime", headers=bearer()).json()["active"]
    finally:
        adapter.release()

    assert isinstance(active["current_child"], dict)
    assert active["last_child"] is None
    assert not any(isinstance(v, list) for v in active.values()), (
        "the activity tree must never grow a list of children")
    # Merged: the ROOT wrote "starting" before the launch, the CHILD then wrote "running" into its
    # own file, and section 3.2 says the API "follows the root's current_child.run_id and merges
    # the latest child record".
    assert active["current_child"]["run_id"] == CHILD_RUN_ID
    assert active["current_child"]["state"] == "running"
    assert active["current_child"]["record_available"] is True
    assert set(active["current_child"]) == {
        "run_id", "activity", "state", "started_at", "run_name", "run_dir",
        "updated_at", "finished_at", "returncode", "error", "record_available"}


def test_a_child_summary_with_no_readable_record_still_reports_the_summary(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    """Section 3.2 sequence step 1: the root publishes ``current_child`` BEFORE launching. Between
    that write and the child's first write there is a summary and no record, and the API must show
    the summary rather than pretending no child exists."""
    summary = T.ChildSummary(run_id=CHILD_RUN_ID, activity=T.Activity.CALIBRATION_PIPELINE,
                             state=T.RunState.STARTING, started_at=NOW,
                             run_name=T.CALIBRATION_RUN_NAME, run_dir=None)
    publish_active(deployment, _root(sandbox.root, current_child=summary))
    adapter = hold_the_lease(deployment)
    try:
        active = client.get("/v1/runtime", headers=bearer()).json()["active"]
    finally:
        adapter.release()
    assert active["current_child"]["state"] == "starting"
    assert active["current_child"]["record_available"] is False


def test_last_comes_from_last_json(client: TestClient, v2: None, sandbox: Sandbox,
                                   deployment: Path) -> None:
    publish_last(deployment, _root(sandbox.root, state=T.RunState.DONE, finished_at=NOW,
                                   returncode=0))
    body = client.get("/v1/runtime", headers=bearer()).json()
    assert body["last"]["run_id"] == ROOT_RUN_ID
    assert body["last"]["state"] == "done"
    assert body["active"] is None


def test_next_run_settings_are_labeled_as_next_run(client: TestClient, v2: None,
                                                   sandbox: Sandbox) -> None:
    """Section 2.6: "Editing YAML affects only the next run". The renderer cannot say that today
    (Step 6), and it cannot start saying it unless the payload distinguishes the two."""
    body = client.get("/v1/runtime", headers=bearer()).json()
    nxt = body["next_run_settings"]
    assert nxt["applies_to"] == "next run"
    assert Path(nxt["settings_path"]) == sandbox.settings_path
    assert nxt["run_name"] == sandbox.run_name
    assert Path(nxt["run_dir"]) == sandbox.run_dir
    assert nxt["error"] is None


def test_the_settings_parse_is_memoized_on_the_FILE_and_not_on_a_clock(
    v2: None, sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/v1/runtime`` is polled, and its settings load sits outside the registry TTL -- an ~9 KB
    YAML through PyYAML's pure-Python loader measured ~17 ms, i.e. the entire cost of the endpoint.

    The key is the file's mtime and size, never an expiry, because section 2.6 promises "editing
    YAML affects only the next run" -- and the corollary is that an edit shows up on the very next
    read, including one made by ``PUT /v1/settings`` a millisecond ago.
    """
    from leafmachine3.core import config as config_module
    from leafmachine3.server import metrics_api

    calls: list[Path] = []
    real_load = config_module.Config.load.__func__

    def counting_load(cls: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(Path(path))
        return real_load(cls, path, *args, **kwargs)

    monkeypatch.setattr(config_module.Config, "load", classmethod(counting_load))
    metrics_api.invalidate_runtime_cache()

    first = metrics_api.next_run_settings()
    second = metrics_api.next_run_settings()
    assert first["run_name"] == second["run_name"] == "contract_run"
    assert len(calls) == 1, "the settings YAML is parsed once, not once per poll"

    sandbox.write_settings(run_name="edited_run")
    assert metrics_api.next_run_settings()["run_name"] == "edited_run"
    assert len(calls) == 2, "an edit must be visible immediately, so the key is the file itself"


def test_the_server_block_identifies_this_instance_and_never_carries_a_token(
    client: TestClient, v2: None
) -> None:
    from leafmachine3.server import metrics_api

    body = client.get("/v1/runtime", headers=bearer()).json()
    assert body["server"]["instance_id"] == metrics_api.SERVER_INSTANCE_ID
    assert body["server"]["runtime_v2"] is True
    assert body["server"]["pid_is_diagnostic_only"] is True
    blob = json.dumps(body)
    assert os.environ["LM3_SERVER_TOKEN"] not in blob, "gate 13: no bearer token in any payload"


def test_a_settings_file_that_will_not_parse_is_reported_not_raised(
    client: TestClient, v2: None, sandbox: Sandbox
) -> None:
    sandbox.settings_path.write_text("project: [this is not a mapping\n", encoding="utf-8")
    body = client.get("/v1/runtime", headers=bearer()).json()
    assert body["next_run_settings"]["error"]
    assert body["server"]["runtime_v2"] is True         # the rest of the view still answers


# --------------------------------------------------------------------------------------------- #
# 4. Section 2.9 -- schema skew fails safe
# --------------------------------------------------------------------------------------------- #
def test_a_newer_schema_is_occupied_incompatible_and_uninterpreted(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    """Section 2.9, every clause: "the OS lock still decides whether the deployment is occupied;
    if held -> report ``active: true, compatible: false``; disable Start and all control actions;
    do not interpret unknown fields; display 'This runtime was created by a newer LM3'".
    """
    payload = R.record_to_dict(_root(sandbox.root))
    payload["schema_version"] = T.SCHEMA_VERSION + 5
    payload["something_from_the_future"] = {"a": 1}
    R.atomic_write_json(deployment / T.ACTIVE_RECORD_FILENAME, payload)

    adapter = hold_the_lease(deployment)
    try:
        active = client.get("/v1/runtime", headers=bearer()).json()["active"]
    finally:
        adapter.release()

    assert active["occupied"] is True
    assert active["compatible"] is False
    assert active["record"] is None, "an unreadable schema must not be half-interpreted"
    assert active["run_id"] is None
    assert active["state"] == "unknown"
    assert active["schema_version"] == T.SCHEMA_VERSION + 5
    assert "newer LM3" in active["message"]
    assert active["can_stop"] is False and "newer LM3" in active["can_stop_reason"]
    assert "something_from_the_future" not in json.dumps(active)


def test_a_newer_schema_never_reaches_the_compatibility_projection(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    """Projecting an uninterpretable record into the legacy shape would BE interpreting it."""
    payload = R.record_to_dict(_root(sandbox.root))
    payload["schema_version"] = T.SCHEMA_VERSION + 5
    R.atomic_write_json(deployment / T.ACTIVE_RECORD_FILENAME, payload)
    adapter = hold_the_lease(deployment)
    try:
        body = client.get("/v1/run/active", headers=bearer()).json()
    finally:
        adapter.release()
    assert body["active"] is False
    assert_json_shape(body, RUN_RECORD_SPEC, where="GET /v1/run/active")


def test_a_held_lease_with_no_record_at_all_is_still_occupied(
    client: TestClient, v2: None, deployment: Path
) -> None:
    """Section 3.3's occupied-but-unidentifiable case. The GUI must not render it as idle."""
    adapter = hold_the_lease(deployment)
    try:
        active = client.get("/v1/runtime", headers=bearer()).json()["active"]
    finally:
        adapter.release()
    assert active["occupied"] is True
    assert active["run_id"] is None
    assert active["can_stop"] is False
    assert "lease is held" in (active["message"] or "")


def test_a_malformed_record_reports_diagnostics_and_never_guesses(
    client: TestClient, v2: None, deployment: Path
) -> None:
    (deployment / T.ACTIVE_RECORD_FILENAME).write_text("{not json", encoding="utf-8")
    adapter = hold_the_lease(deployment)
    try:
        body = client.get("/v1/runtime", headers=bearer()).json()
    finally:
        adapter.release()
    assert body["active"]["occupied"] is True
    assert body["active"]["record"] is None
    assert body["active"]["can_stop"] is False


# --------------------------------------------------------------------------------------------- #
# 5. can_stop -- handle ownership ALONE (section 2.5, revision 15)
# --------------------------------------------------------------------------------------------- #
def test_can_stop_is_true_only_with_a_live_handle_for_this_run_id(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    publish_active(deployment, _root(sandbox.root))
    install_launched_run(ROOT_RUN_ID)
    adapter = hold_the_lease(deployment)
    try:
        active = client.get("/v1/runtime", headers=bearer()).json()["active"]
    finally:
        adapter.release()
    assert active["can_stop"] is True
    assert active["can_stop_reason"] is None


def test_a_handle_for_a_different_run_id_does_not_authorize_control(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    """Ownership is per RUN, not per server: holding a handle for yesterday's run says nothing
    about the one holding the deployment now."""
    publish_active(deployment, _root(sandbox.root))
    install_launched_run("33333333-3333-4333-8333-333333333333")
    adapter = hold_the_lease(deployment)
    try:
        active = client.get("/v1/runtime", headers=bearer()).json()["active"]
    finally:
        adapter.release()
    assert active["can_stop"] is False
    assert ROOT_RUN_ID in active["can_stop_reason"]


def test_a_handle_from_another_server_instance_does_not_authorize_control(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    """Section 2.5: "launched by this server ``instance_id``". A restarted server observes."""
    publish_active(deployment, _root(sandbox.root))
    install_launched_run(ROOT_RUN_ID, instance_id="a-previous-server-instance")
    adapter = hold_the_lease(deployment)
    try:
        active = client.get("/v1/runtime", headers=bearer()).json()["active"]
    finally:
        adapter.release()
    assert active["can_stop"] is False
    assert "different server instance" in active["can_stop_reason"]


def test_a_dead_handle_does_not_authorize_control(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    publish_active(deployment, _root(sandbox.root))
    install_launched_run(ROOT_RUN_ID, alive=False)
    adapter = hold_the_lease(deployment)
    try:
        active = client.get("/v1/runtime", headers=bearer()).json()["active"]
    finally:
        adapter.release()
    assert active["can_stop"] is False


def test_can_stop_consults_neither_the_pid_nor_a_creation_time(
    v2: None, sandbox: Sandbox, deployment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Revision 15 removed both extra checks. Pinned by making them impossible to perform: if
    anything reached for ``psutil`` or ``os.kill(pid, 0)`` to answer this question, it would raise.

    The record's ``pid`` is deliberately a value no live process can wear, and it still does not
    change the answer -- the handle does.
    """
    from leafmachine3.server import metrics_api

    def _forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("control authority must not probe a PID (invariant 12)")

    monkeypatch.setattr(os, "kill", _forbidden)
    publish_active(deployment, _root(sandbox.root, pid=2 ** 31 - 33))
    install_launched_run(ROOT_RUN_ID)
    can_stop, reason = metrics_api._can_stop(ROOT_RUN_ID, compatible=True)
    assert can_stop is True and reason is None


# --------------------------------------------------------------------------------------------- #
# 6. POST /v1/run/stop -- observer-only refusals
# --------------------------------------------------------------------------------------------- #
def test_stop_refuses_a_run_this_server_did_not_launch(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    """Step 4: "``POST /v1/run/stop`` rejects observer-only runs with 409/403 and never synthesizes
    a process group from registry JSON." 409, because the same request succeeds from the process
    that owns the run -- it is a state conflict, not an authorization failure of the caller."""
    publish_active(deployment, _root(sandbox.root))
    adapter = hold_the_lease(deployment)
    try:
        resp = client.post("/v1/run/stop", json={}, headers=bearer())
    finally:
        adapter.release()
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["reason"] == "observer-only"
    assert detail["can_stop"] is False
    assert detail["active"]["run_id"] == ROOT_RUN_ID
    assert "did not launch" in detail["message"]


def test_stop_never_synthesizes_a_process_group_from_the_record(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant 12, stated as a trap: the record names a PID and a plausible pgid, and neither
    ``killpg`` nor ``kill`` may be reached."""
    fired: list[tuple] = []
    monkeypatch.setattr(os, "killpg", lambda *a: fired.append(a), raising=False)
    monkeypatch.setattr(os, "kill", lambda *a: fired.append(a))
    publish_active(deployment, _root(sandbox.root, pid=4321))
    adapter = hold_the_lease(deployment)
    try:
        assert client.post("/v1/run/stop", json={}, headers=bearer()).status_code == 409
    finally:
        adapter.release()
    assert fired == []


def test_stop_refuses_when_the_handle_belongs_to_a_previous_instance(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    publish_active(deployment, _root(sandbox.root))
    handle = install_launched_run(ROOT_RUN_ID, instance_id="a-previous-server-instance")
    adapter = hold_the_lease(deployment)
    try:
        resp = client.post("/v1/run/stop", json={}, headers=bearer())
    finally:
        adapter.release()
    assert resp.status_code == 409
    assert handle.signals == [], "no signal may be sent to a run we do not own"


def test_stop_signals_the_tree_when_the_handle_is_ours(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    publish_active(deployment, _root(sandbox.root))
    handle = install_launched_run(ROOT_RUN_ID)
    adapter = hold_the_lease(deployment)
    try:
        resp = client.post("/v1/run/stop", json={"grace_s": 0.2}, headers=bearer())
    finally:
        adapter.release()
    assert resp.status_code == 200
    assert handle.signals[:1] == [int(signal.SIGTERM)]


def test_an_idle_deployment_still_says_no_active_run_to_stop(client: TestClient, v2: None,
                                                             deployment: Path) -> None:
    resp = client.post("/v1/run/stop", json={}, headers=bearer())
    assert resp.status_code == 409
    assert resp.json()["detail"] == "no active run to stop"


# --------------------------------------------------------------------------------------------- #
# 7. GET /v1/run/active as a compatibility projection (section 5)
# --------------------------------------------------------------------------------------------- #
def test_the_projection_shows_a_run_this_server_did_not_launch(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    """Section 5 keeps the route "temporarily" as a projection. Making it project the REGISTRY is
    what lets an un-migrated client (and step 6's top bar, until it moves) see a CLI run instead of
    "idle" -- while the key set stays exactly what ``_Run.record``'s docstring promises."""
    publish_active(deployment, _root(sandbox.root))
    adapter = hold_the_lease(deployment)
    try:
        resp = client.get("/v1/run/active", headers=bearer())
    finally:
        adapter.release()
    body = resp.json()
    assert_json_shape(body, RUN_RECORD_SPEC, where="GET /v1/run/active")
    assert body["active"] is True and body["state"] == "running"
    assert body["run_name"] == "contract_run"
    assert Path(body["db_path"]) == sandbox.root / "out" / "contract_run" / "contract_run.sqlite"
    assert body["adopted"] is False, "the projection observes; it does not adopt"


def test_the_projection_carries_deprecation_metadata_in_headers_only(
    client: TestClient, v2: None
) -> None:
    """Metadata in HEADERS, never in the body: the body's key set is pinned key-for-key by
    ``tests/_contract_helpers.RUN_RECORD_SPEC``, so a ``deprecated: true`` field would break the
    very clients the projection exists to keep working."""
    resp = client.get("/v1/run/active", headers=bearer())
    assert resp.headers["Deprecation"] == "true"
    assert "/v1/runtime" in resp.headers["Link"]
    assert_json_shape(resp.json(), RUN_RECORD_SPEC, where="GET /v1/run/active")
    assert "deprecated" not in resp.json()


def test_an_abandoned_record_is_never_projected_as_a_live_run(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    """Section 2.9: "the OS lock still decides whether the deployment is occupied" -- so a record
    that says ``running`` while the lease is ACQUIRABLE is section 3.3's ``abandoned``, not a run.

    This is the SIGKILLed-lone-root case of section 6: the kernel released the lock and nothing
    rewrote ``active.json``, so an unfiltered projection would answer ``active: true`` forever --
    the top bar disables Start on that flag while ``POST /v1/run/stop`` refuses (the lock is free),
    which is a GUI wedged until somebody deletes the file by hand.

    The fact is not erased, only reclassified: ``GET /v1/runtime`` still describes the record, which
    is exactly the split section 5 asks for ("do not overload a single field named ``active``").
    """
    from leafmachine3.core.runtime import RecordClassification
    from leafmachine3.server import metrics_api

    publish_active(deployment, _root(sandbox.root))     # ... and NOBODY takes the lease
    metrics_api.invalidate_runtime_cache()
    snapshot, _diag = metrics_api.runtime_snapshot()
    assert snapshot.classification is RecordClassification.ABANDONED
    assert snapshot.active is False, "the lease is free: this is the whole premise of the test"

    body = client.get("/v1/run/active", headers=bearer()).json()
    assert_json_shape(body, RUN_RECORD_SPEC, where="GET /v1/run/active")
    assert body["active"] is False, "a crashed writer's stale JSON is not a live run"
    assert body["state"] == "idle"
    assert body["pid"] is None, "section 3.3: a reader must never even publish that PID as live"

    active_view = client.get("/v1/runtime", headers=bearer()).json()["active"]
    assert active_view["occupied"] is False
    assert active_view["classification"] == "abandoned"
    assert active_view["state"] == "running", "the record is reported verbatim, just not as active"
    assert active_view["can_stop"] is False


def test_an_abandoned_record_does_not_mask_this_server_s_own_crash(
    client: TestClient, v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    """``active()`` consults the projection BEFORE its own ``_RUN``, so an unfiltered projection
    would hide a run this server launched and watched die.

    A SIGKILLed child leaves ``_RUN.state == "error"`` with the exit code in ``error``; the stale
    ``active.json`` it left behind must not outrank that. Section 2.9 again: the lock is free, so
    there is nothing live to outrank it WITH.
    """
    from leafmachine3.server import metrics_api

    install_launched_run(alive=False, state="error")
    metrics_api._RUN.error = "machine3 exited with code -9"
    metrics_api._RUN.returncode = -9
    publish_active(deployment, _root(sandbox.root))     # the record its own crash left behind
    metrics_api.invalidate_runtime_cache()

    body = client.get("/v1/run/active", headers=bearer()).json()
    assert body["active"] is False
    assert body["state"] == "error", "the crash must be visible, not overwritten by stale JSON"
    assert body["error"] == "machine3 exited with code -9"
    assert body["returncode"] == -9


# --------------------------------------------------------------------------------------------- #
# 8. active_run_ref -- the registry reader that replaces the private binding
# --------------------------------------------------------------------------------------------- #
def test_active_run_ref_names_the_ledger_of_the_run_holding_the_lease(
    sandbox: Sandbox, v2: None, deployment: Path
) -> None:
    from leafmachine3.server import metrics_api

    publish_active(deployment, _root(sandbox.root))
    adapter = hold_the_lease(deployment)
    try:
        ref = metrics_api.active_run_ref()
    finally:
        adapter.release()
    assert ref["run_id"] == ROOT_RUN_ID
    assert ref["source"] == "runtime"
    assert Path(ref["active_db_path"]).name == "contract_run.sqlite"
    assert metrics_api.active_db_path() == Path(ref["active_db_path"])


def test_a_hardware_setup_root_does_not_move_project_history(
    client: TestClient, sandbox: Sandbox, v2: None, deployment: Path
) -> None:
    """Section 2.6: a ``hardware_setup`` root shows "a tuning state in the Machine panel" and must
    NOT switch project history to ``_lm3_calibration``. It carries no ``project`` block at all
    (invariant 6), so the ledger reference is ``None`` -- there is nothing to follow."""
    from leafmachine3.server import metrics_api

    setup = _root(sandbox.root, activity=T.Activity.HARDWARE_SETUP, project=None,
                  hardware=T.HardwareBlock(destination_path=str(sandbox.hardware_path)))
    publish_active(deployment, setup)
    adapter = hold_the_lease(deployment)
    try:
        assert metrics_api.active_run_ref() is None
        active = client.get("/v1/runtime", headers=bearer()).json()["active"]
    finally:
        adapter.release()
    assert active["activity"] == "hardware_setup"
    assert active["record"]["hardware"]["destination_path"] == str(sandbox.hardware_path)
    assert "project" not in active["record"]


def test_is_active_is_true_for_a_run_this_server_did_not_launch(
    sandbox: Sandbox, v2: None, deployment: Path
) -> None:
    """The question callers ask ``is_active()`` is "may I start?", and the answer must not depend
    on who launched the incumbent."""
    from leafmachine3.server import metrics_api

    adapter = hold_the_lease(deployment)
    try:
        assert metrics_api.is_active() is True
    finally:
        adapter.release()
    metrics_api.invalidate_runtime_cache()
    assert metrics_api.is_active() is False


# --------------------------------------------------------------------------------------------- #
# 9. Failure modes of the reader itself
# --------------------------------------------------------------------------------------------- #
def test_an_unresolvable_deployment_is_reported_not_raised(
    client: TestClient, v2: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"Registry directory unavailable: fail before starting work; never run unregistered" applies
    to the LAUNCHER. A read-only observer answering 500 would take the whole GUI down with it."""
    from leafmachine3.server import metrics_api

    def _boom() -> Path:
        raise RuntimeError("no runtime base could be resolved")

    monkeypatch.setattr(metrics_api, "deployment_dir", _boom)
    metrics_api.invalidate_runtime_cache()
    body = client.get("/v1/runtime", headers=bearer()).json()
    assert body["active"] is None
    assert "no runtime base" in body["diagnostics"]["error"]
    assert body["diagnostics"]["readable"] is False


def test_the_registry_read_is_cached_and_invalidatable(
    v2: None, sandbox: Sandbox, deployment: Path
) -> None:
    """The TTL bounds the lock-probe rate under a polling GUI; the explicit invalidation is what
    replaces ``bind_run``'s cache drop, so a state change is visible immediately rather than up to
    a TTL later."""
    from leafmachine3.server import metrics_api

    assert metrics_api.runtime_snapshot()[0].record is None
    publish_active(deployment, _root(sandbox.root))
    assert metrics_api.runtime_snapshot()[0].record is None, "still serving the cached read"
    metrics_api.invalidate_runtime_cache()
    assert metrics_api.runtime_snapshot()[0].record.run_id == ROOT_RUN_ID
