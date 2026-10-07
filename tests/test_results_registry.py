"""``results_api`` against the runtime registry -- plan section 4 Step 4 and section 3.1 row 5.

Three questions, and they are separable:

  ROOTS    Row 5 fixes the run-history search order as ``LM3_RUNS_ROOTS`` -> the ACTIVE and LAST
           runtime output roots -> ``project.output.dir`` from the resolved settings, and an EMPTY
           list on a miss. The middle slot is what makes a run started from the CLI (or by another
           GUI, or by a Slurm step) visible in the Results tab at all: nothing else in the tab's
           world knows where that run writes.

  SELECTOR ``GET /v1/runs`` has a frozen envelope (``tests/_contract_helpers.RUNS_ENVELOPE_SPEC``
           is an EXACT key set, asserted by ``TestPreservedContract``), so "which of these rows is
           the run happening right now" cannot be answered by adding a key to it. It is answered by
           ``GET /v1/runs/-/selector``, which is what the Step 6 renderer selects a run by.

  FLAG     Every one of those behaviors is gated on ``LM3_RUNTIME_V2``. With the flag off this
           module must read no record and change no root -- the registry is not written yet.

Everything runs inside ``tmp_path``: a private ``LM3_RUNTIME_DIR``, a private deployment id, a
private settings file and a private jobs root. No test acquires a lease through the production
path; the one test that needs the deployment to look OCCUPIED takes a real ``flock`` on the real
lock file, which is what ``probe_occupied`` answers about (it opens its own descriptor and asks for
a SHARED lock, so it sees this process's exclusive one).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from leafmachine3.core import paths
from leafmachine3.core.runtime import _types as T
from leafmachine3.core.runtime import records as R
from leafmachine3.server import results_api

NOW = "2026-08-28T12:00:00Z"
LATER = "2026-08-28T12:30:00Z"


# --------------------------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A private deployment: runtime dir, settings file, jobs root -- and explicit compatibility mode.

    ``monkeypatch.chdir`` into an empty directory as well: the point of row 5 is that nothing beside
    the launcher is a run root, and a test left in the checkout could only prove that by accident.
    """
    cwd = tmp_path / "cwd"
    runtime_dir = tmp_path / "runtime"
    jobs = tmp_path / "jobs"
    settings_dir = tmp_path / "cfg"
    output_dir = tmp_path / "out"
    for directory in (cwd, runtime_dir, jobs, settings_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(cwd)

    settings = settings_dir / "LM3_settings.yaml"
    settings.write_text(
        yaml.safe_dump({"project": {"output": {"dir": str(output_dir)}, "input": {"dirs": []}}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("LM3_RUNTIME_V2", "0")
    monkeypatch.delenv("LM3_RUNS_ROOTS", raising=False)
    monkeypatch.setenv("LM3_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("LM3_DEPLOYMENT_ID", "results-registry-test")
    monkeypatch.setenv("LM3_SETTINGS", str(settings))
    monkeypatch.setenv("LM3_SERVER_JOBS", str(jobs))

    deployment_dir = paths.deployment_runtime_dir(
        env=os.environ, create=False, check_filesystem=False
    )
    (deployment_dir / T.CHILDREN_DIRNAME).mkdir(parents=True, exist_ok=True)

    _reset()
    yield {
        "tmp": tmp_path, "cwd": cwd, "settings": settings, "output_dir": output_dir,
        "jobs": jobs, "deployment_dir": deployment_dir,
    }
    _reset()


def _reset() -> None:
    results_api._EXTRA_ROOTS.clear()
    results_api._RUNS_CACHE.invalidate()
    results_api._MEDIA_CACHE.clear()


def flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM3_RUNTIME_V2", "1")


# --------------------------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------------------------- #
def make_run_dir(parent: Path, name: str) -> Path:
    """A directory ``_looks_like_run`` recognizes: ``<dir>/<dir>.sqlite`` plus a reports tree."""
    run = parent / name
    (run / "reports").mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(parents=True, exist_ok=True)
    (run / f"{name}.sqlite").write_bytes(b"")
    return run


def in_place_project(run_dir: Path, *, run_name: str | None = None) -> T.ProjectBlock:
    name = run_name or run_dir.name
    db = run_dir / f"{name}.sqlite"
    return T.ProjectBlock(
        run_name=name,
        input_dirs=(str(run_dir.parent / "input"),),
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


def staged_project(artifact_dir: Path, scratch: Path) -> T.ProjectBlock:
    name = artifact_dir.name
    return T.ProjectBlock(
        run_name=name,
        input_dirs=(str(artifact_dir.parent / "input"),),
        artifact_dir=str(artifact_dir),
        active_state_dir=str(scratch),
        active_db_path=str(scratch / f"{name}.sqlite"),
        archive_mode=T.ArchiveMode.STAGED,
        archive_status=T.ArchiveStatus.PENDING,
        archive_pointer_path=str(artifact_dir / T.ARCHIVE_POINTER_FILENAME),
        archived_db_path=None,
        run_dir=str(artifact_dir),
        log_path=str(artifact_dir / "logs" / "lm3.log"),
    )


def root_record(settings: Path, project: T.ProjectBlock | None, **overrides: Any) -> T.RuntimeRecord:
    base: dict[str, Any] = dict(
        run_id="11111111-1111-4111-8111-111111111111",
        activity=T.Activity.PIPELINE,
        activity_role=T.ActivityRole.ROOT,
        state=T.RunState.RUNNING,
        launcher=T.Launcher.CLI,
        pid=4321,
        process_started_at=1787932800.25,
        started_at=NOW,
        updated_at=NOW,
        deployment=T.DeploymentInfo(id=paths.deployment_key(os.environ), node="testnode"),
        config=T.ConfigRef(path=str(settings), sha256="a" * 64),
        project=project,
    )
    base.update(overrides)
    return T.RuntimeRecord(**base)


def publish(deployment_dir: Path, record: T.RuntimeRecord, *, filename: str) -> None:
    """Write a record straight to disk.

    Not through ``RecordStore``: the store is scoped to the ``run_id`` whose writes it is allowed to
    make and enforces the state machine, and these tests are about what a READER concludes from
    bytes that are already there.
    """
    R.validate_record(record)
    R.atomic_write_json(deployment_dir / filename, R.record_to_dict(record))


def hold_lease(deployment_dir: Path):
    """Take a real exclusive ``flock`` on the deployment's activity lock. POSIX only."""
    import fcntl

    lock_path = deployment_dir / paths.ACTIVITY_LOCK_FILENAME
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


# --------------------------------------------------------------------------------------------- #
# section 3.1 row 5 -- the roots
# --------------------------------------------------------------------------------------------- #
class TestRunRoots:
    def test_active_run_output_root_is_searched(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Row 5 slot 2. The CLI run writes somewhere the settings file has never heard of; without
        the registry the Results tab cannot see the run that is happening right now."""
        flag_on(monkeypatch)
        elsewhere = sandbox["tmp"] / "elsewhere"
        run = make_run_dir(elsewhere, "acer_rubrum")
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], in_place_project(run)),
                filename=paths.ACTIVE_RECORD_FILENAME)
        _reset()

        assert elsewhere.resolve() in results_api.run_roots()
        assert "acer_rubrum" in {r.name for r in results_api.discover_runs(refresh=True)}

    def test_last_run_output_root_is_searched(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Row 5 says active AND last: when the CLI run finishes, its output must not vanish from
        the tab the moment ``active.json`` is removed."""
        flag_on(monkeypatch)
        elsewhere = sandbox["tmp"] / "finished"
        run = make_run_dir(elsewhere, "quercus_alba")
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], in_place_project(run),
                            state=T.RunState.DONE, finished_at=LATER, updated_at=LATER,
                            returncode=0),
                filename=paths.LAST_RECORD_FILENAME)
        _reset()

        assert elsewhere.resolve() in results_api.run_roots()

    def test_staged_mode_contributes_both_the_artifact_and_the_scratch_root(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Section 2.10 gives a run three storage roles, and in staged mode the live ledger is under
        node-local scratch while the browsable tree is under persistent storage. Searching only one
        of them loses half the run."""
        flag_on(monkeypatch)
        artifact = make_run_dir(sandbox["tmp"] / "persistent", "betula")
        scratch = make_run_dir(sandbox["tmp"] / "scratch", "betula")
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], staged_project(artifact, scratch)),
                filename=paths.ACTIVE_RECORD_FILENAME)
        _reset()

        roots = results_api.run_roots()
        assert (sandbox["tmp"] / "persistent").resolve() in roots
        assert (sandbox["tmp"] / "scratch").resolve() in roots

    def test_row_5_order_is_env_then_runtime_then_settings(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The order is normative, not incidental: it decides which root a run is REPORTED under
        (``Run.root``/``rel``) when two roots contain it."""
        flag_on(monkeypatch)
        explicit = sandbox["tmp"] / "explicit"
        explicit.mkdir()
        monkeypatch.setenv("LM3_RUNS_ROOTS", str(explicit))
        runtime_root = sandbox["tmp"] / "runtime-output"
        run = make_run_dir(runtime_root, "salix")
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], in_place_project(run)),
                filename=paths.ACTIVE_RECORD_FILENAME)
        _reset()

        roots = [str(p) for p in results_api.run_roots()]
        assert roots.index(str(explicit.resolve())) < roots.index(str(runtime_root.resolve()))
        assert roots.index(str(runtime_root.resolve())) < roots.index(
            str(sandbox["output_dir"].resolve()))

    def test_an_abandoned_record_still_contributes_history(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A record whose writer died is not LIVE (no lease is held here), but the run it names
        produced files and the history view should keep finding them."""
        flag_on(monkeypatch)
        elsewhere = sandbox["tmp"] / "abandoned"
        run = make_run_dir(elsewhere, "tilia")
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], in_place_project(run)),
                filename=paths.ACTIVE_RECORD_FILENAME)
        _reset()

        view = results_api.runtime_view()
        assert view.record is not None and view.active is None      # published, not live
        assert view.classification == T.RecordClassification.ABANDONED.value
        assert elsewhere.resolve() in results_api.run_roots()

    def test_a_hardware_setup_root_contributes_no_root(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invariant 6: tuning the machine is not work on anybody's specimens, so a
        ``hardware_setup`` record carries no project -- and therefore no output root."""
        flag_on(monkeypatch)
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], None,
                            activity=T.Activity.HARDWARE_SETUP,
                            hardware=T.HardwareBlock(
                                destination_path=str(sandbox["tmp"] / "hardware_settings.yaml"))),
                filename=paths.ACTIVE_RECORD_FILENAME)
        _reset()

        assert results_api.runtime_roots() == []

    def test_a_calibration_child_never_becomes_a_run_root(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Section 2.2: the UI must never mistake ``_lm3_calibration`` for the user's project. The
        reader does not follow ``current_child``/``last_child`` at all."""
        flag_on(monkeypatch)
        child_dir = make_run_dir(sandbox["tmp"] / "calibration-scratch", T.CALIBRATION_RUN_NAME)
        summary = T.ChildSummary(
            run_id="22222222-2222-4222-8222-222222222222",
            activity=T.Activity.CALIBRATION_PIPELINE,
            run_name=T.CALIBRATION_RUN_NAME,
            state=T.RunState.RUNNING,
            started_at=NOW,
            run_dir=str(child_dir),
        )
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], None,
                            activity=T.Activity.HARDWARE_SETUP,
                            hardware=T.HardwareBlock(
                                destination_path=str(sandbox["tmp"] / "hardware_settings.yaml")),
                            current_child=summary),
                filename=paths.ACTIVE_RECORD_FILENAME)
        _reset()

        roots = results_api.run_roots()
        assert (sandbox["tmp"] / "calibration-scratch").resolve() not in roots

    def test_the_flag_off_reads_nothing_and_moves_no_root(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit flag OFF preserves the compatibility path even with records already on disk."""
        elsewhere = sandbox["tmp"] / "elsewhere"
        run = make_run_dir(elsewhere, "acer_rubrum")
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], in_place_project(run)),
                filename=paths.ACTIVE_RECORD_FILENAME)
        _reset()

        assert results_api.runtime_view().enabled is False
        assert results_api.runtime_roots() == []
        assert elsewhere.resolve() not in results_api.run_roots()

    def test_on_a_total_miss_the_root_list_is_empty(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Row 5's "on miss: an EMPTY list". An empty Results tab is honest; a tab full of whatever
        happened to be beside the launcher is not."""
        flag_on(monkeypatch)
        for name in ("runs", "examples_out", "output"):
            (sandbox["cwd"] / name).mkdir()
        sandbox["settings"].write_text(
            yaml.safe_dump({"project": {"output": {"dir": str(sandbox["tmp"] / "gone")}}}),
            encoding="utf-8")
        monkeypatch.setenv("LM3_SERVER_JOBS", str(sandbox["tmp"] / "no-such-jobs-root"))
        _reset()

        assert results_api.run_roots() == []

    def test_a_record_naming_the_filesystem_root_is_refused(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``/`` is never a run root: a five-level scan of the whole machine is not a Results tab."""
        assert results_api._output_root("/run") is None
        assert results_api._output_root("relative/path") is None
        assert results_api._output_root(None) is None
        assert results_api._output_root("/out/run") == Path("/out")

    def test_an_unreadable_registry_never_breaks_discovery(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A corrupt ``active.json`` costs the runtime roots and nothing else."""
        flag_on(monkeypatch)
        (sandbox["deployment_dir"] / paths.ACTIVE_RECORD_FILENAME).write_text(
            "{ this is not json", encoding="utf-8")
        make_run_dir(sandbox["output_dir"], "from_settings")
        _reset()

        assert results_api.runtime_roots() == []
        assert sandbox["output_dir"].resolve() in results_api.run_roots()
        assert "from_settings" in {r.name for r in results_api.discover_runs(refresh=True)}


# --------------------------------------------------------------------------------------------- #
# the selector -- what the Step 6 renderer selects a run BY
# --------------------------------------------------------------------------------------------- #
class TestRunSelector:
    def test_the_active_run_is_named_and_joined_to_its_history_row(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flag_on(monkeypatch)
        run = make_run_dir(sandbox["tmp"] / "elsewhere", "acer_rubrum")
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], in_place_project(run)),
                filename=paths.ACTIVE_RECORD_FILENAME)
        _reset()

        payload = results_api.run_selector(refresh=True, lease_probe=lambda: True)

        active = payload["active"]
        assert active is not None
        assert active["run_name"] == "acer_rubrum"
        assert active["run_id"] == "11111111-1111-4111-8111-111111111111"
        assert active["state"] == "running" and active["activity"] == "pipeline"
        assert active["live"] is True
        assert Path(active["db_path"]) == run / "acer_rubrum.sqlite"
        # joined to the discovered run, so the renderer can link straight at /v1/runs/{id}
        assert active["id"] and active["id"] in {r["id"] for r in payload["runs"]}
        row = next(r for r in payload["runs"] if r["id"] == active["id"])
        assert row["live"] is True and row["source"] == "runtime-active"
        assert payload["follow"] == "active"
        assert payload["selected_default"] == active["id"]

    def test_a_run_with_no_directory_yet_is_still_named(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run publishes ``starting`` before it has a ledger. ``id: null`` is the honest answer --
        the renderer can say what is running without being handed a link that would 404."""
        flag_on(monkeypatch)
        future = sandbox["tmp"] / "elsewhere" / "not_created_yet"
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], in_place_project(future),
                            state=T.RunState.STARTING),
                filename=paths.ACTIVE_RECORD_FILENAME)
        _reset()

        payload = results_api.run_selector(refresh=True, lease_probe=lambda: True)
        assert payload["active"]["id"] is None
        assert payload["active"]["run_name"] == "not_created_yet"
        assert payload["active"]["state"] == "starting"
        assert payload["selected_default"] is None

    def test_the_last_run_is_offered_when_nothing_is_live(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flag_on(monkeypatch)
        run = make_run_dir(sandbox["output_dir"], "quercus_alba")
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], in_place_project(run),
                            state=T.RunState.DONE, finished_at=LATER, updated_at=LATER,
                            returncode=0),
                filename=paths.LAST_RECORD_FILENAME)
        _reset()

        payload = results_api.run_selector(refresh=True, lease_probe=lambda: False)
        assert payload["active"] is None
        assert payload["last"]["run_name"] == "quercus_alba"
        assert payload["last"]["live"] is False
        assert payload["last"]["state"] == "done"
        assert payload["follow"] == "last"
        assert payload["selected_default"] == payload["last"]["id"]

    def test_an_abandoned_record_is_not_offered_as_live(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Section 2.5/3.3: the LOCK decides whether anything is running. A non-terminal record with
        no lease behind it is an abandoned run, and the renderer must not offer to follow it."""
        flag_on(monkeypatch)
        run = make_run_dir(sandbox["output_dir"], "tilia")
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], in_place_project(run)),
                filename=paths.ACTIVE_RECORD_FILENAME)
        _reset()

        payload = results_api.run_selector(refresh=True, lease_probe=lambda: False)
        assert payload["active"] is None
        assert payload["runtime"]["classification"] == "abandoned"
        assert payload["runtime"]["lease_held"] is False
        assert all(row["live"] is False for row in payload["runs"])

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="flock is the POSIX lease primitive")
    def test_a_real_held_lease_is_what_makes_a_run_live(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The DEFAULT probe, not an injected one: ``probe_occupied`` opens its own descriptor and
        asks for a shared lock, so it answers truthfully about this process's exclusive one."""
        flag_on(monkeypatch)
        run = make_run_dir(sandbox["output_dir"], "acer_rubrum")
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], in_place_project(run)),
                filename=paths.ACTIVE_RECORD_FILENAME)
        _reset()

        assert results_api.run_selector(refresh=True)["active"] is None   # no lease yet
        fd = hold_lease(sandbox["deployment_dir"])
        try:
            _reset()
            payload = results_api.run_selector(refresh=True)
            assert payload["runtime"]["lease_held"] is True
            assert payload["active"] is not None and payload["active"]["live"] is True
        finally:
            os.close(fd)

    def test_a_newer_schema_disables_interpretation_but_not_the_list(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Section 2.9: a record from a newer LM3 is reported, never interpreted. ``active: null``
        then means UNKNOWN, and ``runtime.compatible`` is how the renderer knows the difference."""
        flag_on(monkeypatch)
        make_run_dir(sandbox["output_dir"], "from_settings")
        (sandbox["deployment_dir"] / paths.ACTIVE_RECORD_FILENAME).write_text(
            json.dumps({"schema_version": T.SCHEMA_VERSION + 1, "run_id": "x"}), encoding="utf-8")
        _reset()

        payload = results_api.run_selector(refresh=True, lease_probe=lambda: True)
        assert payload["runtime"]["compatible"] is False
        assert payload["active"] is None
        assert payload["runtime"]["message"]
        assert "from_settings" in {row["run_name"] for row in payload["runs"]}

    def test_with_the_flag_off_the_selector_degrades_to_the_plain_list(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        make_run_dir(sandbox["output_dir"], "from_settings")
        _reset()

        payload = results_api.run_selector(refresh=True)
        assert payload["runtime"]["enabled"] is False
        assert payload["active"] is None and payload["last"] is None
        assert payload["follow"] is None
        assert [row["run_name"] for row in payload["runs"]] == ["from_settings"]
        assert payload["selected_default"] == payload["runs"][0]["id"]

    def test_every_row_carries_the_one_reference_shape(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One shape for a run reference, whatever it came from -- that is what lets the Status,
        Results and Postprocess tabs agree on which run they are showing."""
        flag_on(monkeypatch)
        run = make_run_dir(sandbox["output_dir"], "acer_rubrum")
        make_run_dir(sandbox["output_dir"], "quercus_alba")
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], in_place_project(run)),
                filename=paths.ACTIVE_RECORD_FILENAME)
        _reset()

        payload = results_api.run_selector(refresh=True, lease_probe=lambda: True)
        expected = {"schema_version", "source", "id", "run_id", "run_name", "run_dir", "db_path",
                    "activity", "state", "live"}
        for ref in [payload["active"], *payload["runs"]]:
            assert set(ref) == expected
            assert ref["schema_version"] == results_api.RUN_REF_SCHEMA_VERSION
            assert ref["source"] in {"history", "runtime-active", "runtime-last"}

    def test_limit_truncates_the_newest_first_list(
        self, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("a", "b", "c"):
            make_run_dir(sandbox["output_dir"], name)
        _reset()

        assert results_api.run_selector(refresh=True, limit=2)["n"] == 2
        assert results_api.run_selector(refresh=True)["n"] == 3
        assert results_api.list_runs(refresh=True)["n"] == 3


# --------------------------------------------------------------------------------------------- #
# the HTTP surface -- one /v1/runs listing, plus the selector beside it
# --------------------------------------------------------------------------------------------- #
@pytest.fixture()
def client(sandbox: dict):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = fastapi.FastAPI()
    app.include_router(results_api.router())
    return TestClient(app)


class TestRoutes:
    def test_the_runs_envelope_is_unchanged(self, client, sandbox: dict) -> None:
        """The Step 6 additions live at ``/-/selector`` precisely so this key set can stay frozen."""
        make_run_dir(sandbox["output_dir"], "from_settings")
        _reset()

        body = client.get("/v1/runs").json()
        assert set(body) == {"t", "n", "roots", "runs"}
        assert body["n"] == len(body["runs"]) == 1

    def test_the_selector_route_answers(
        self, client, sandbox: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flag_on(monkeypatch)
        run = make_run_dir(sandbox["output_dir"], "acer_rubrum")
        publish(sandbox["deployment_dir"],
                root_record(sandbox["settings"], in_place_project(run)),
                filename=paths.ACTIVE_RECORD_FILENAME)
        _reset()

        body = client.get("/v1/runs/-/selector?refresh=true").json()
        assert set(body) == {"t", "schema_version", "n", "roots", "runs", "active", "last",
                             "follow", "selected_default", "runtime"}
        assert body["runs"][0]["run_name"] == "acer_rubrum"

    def test_the_selector_literal_cannot_shadow_a_run(self, client, sandbox: dict) -> None:
        """``/-/selector`` is two segments; ``/{run}`` matches one. A run may be named anything, so a
        one-segment literal would quietly make a run of that name unreachable by name."""
        make_run_dir(sandbox["output_dir"], "selector")
        make_run_dir(sandbox["output_dir"], "-")
        _reset()

        assert client.get("/v1/runs/selector").json()["name"] == "selector"
        assert client.get("/v1/runs/-").json()["name"] == "-"
        assert client.get("/v1/runs/-/selector").status_code == 200

    def test_limit_is_honored_by_the_selector_and_ignored_by_the_listing(
        self, client, sandbox: dict
    ) -> None:
        """``?limit=`` moved to the selector rather than onto ``GET /v1/runs``: the duplicate route
        being deleted in this same step is told apart from the surviving one by exactly this
        parameter, and its contract test would break for an unrelated reason."""
        for name in ("a", "b", "c"):
            make_run_dir(sandbox["output_dir"], name)
        _reset()

        assert client.get("/v1/runs?limit=2").json()["n"] == 3
        assert client.get("/v1/runs/-/selector?limit=2").json()["n"] == 2
