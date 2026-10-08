"""The renderer's half of the unified runtime -- plan section 4, Step 6.

Plan reference: ``UNIFIED_RUNTIME_IMPLEMENTATION_PLAN_CONSENSUS.md`` -- section 2.6 (UI behavior is
activity-aware), section 2.9 (schema skew fails safe), section 5 (the API migration contract),
section 9 (definition of done), invariants 5, 7, 8 and 9.

What Step 6 asks for, in the plan's own words:

* "Replace top-bar state with ``runtime.active``, ``runtime.last``, ``view.mode``, ``view.runRef``,
  ``settings``, ``busy``."
* "Remove ``openFresh()``, sticky ``state.newRun`` suppression, run-name blanking without writing
  YAML, deriving run activity from a status snapshot, and Close/New paths that stop a job."
* Section 2.6's table: ``pipeline`` -> follow its project DB and logs; ``hardware_setup`` -> show a
  tuning state in the Machine panel and **do not** switch project history to ``_lm3_calibration``;
  unknown/future -> disable Start, refuse control, show a compatibility warning.
* "Historical selection pauses follow-active explicitly, with a visible action to return."
* "Editing YAML affects only the next run." / "Close never stops a pipeline."
* Exit gate: "opening the GUI during any CLI run immediately shows that run, while editable fields
  continue to show -- and clearly label -- next-run settings."

Three kinds of test live here, and the difference matters:

1. **Source contract** -- plain text assertions over the shipped JavaScript. A deletion is only
   real if it cannot come back, and these are the tripwires for the five things Step 6 deletes.
   They are cheap, they run everywhere, and they are the only thing standing between a future
   edit and a resurrected ``state.newRun``.
2. **Pure-logic** -- ``deriveView()`` executed by node, imported from the real ``topbar.js``, not
   a copy. This is section 2.6's decision table, driven with the exact payload shapes the server
   emits.
3. **Agreement (section 9)** -- a real registry record, a real deployment lock, the real FastAPI
   app, and the real renderer logic, checked to agree on one run.

WHAT THIS FILE CANNOT COVER, stated plainly because a green suite must not be read as more than it
is:

* There is no browser and no DOM here. Nothing below renders a pixel, so "the Start button is
  visibly disabled" is verified as ``deriveView(...).canStart is False`` -- the decision, not its
  painting. A regression that computes the right answer and then forgets to apply it to
  ``refs.startBtn.disabled`` would pass tests 1 and 2 and be caught only by the source contract's
  narrow check that the assignment exists at all.
* No Electron, no window, no second instance, no SSE. The status/log streams are asserted by the
  parameters the renderer would open them with, not by bytes arriving.
* No real pipeline runs. The active record is published directly and the lease is taken by the real
  POSIX adapter; nothing here starts a process, loads a model or touches a GPU.
* The tabs' own rendering (Results grid, Console scrollback, Postprocessing form) is exercised only
  as far as its source contract. Their end-to-end behavior needs a headed browser.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest
from fastapi.testclient import TestClient

from leafmachine3.core.runtime import _types as T
from leafmachine3.core.runtime import records as R
from tests._contract_helpers import (
    Sandbox,
    bearer,
    isolate_server_paths,
    reset_server_module_state,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_JS = REPO_ROOT / "leafmachine3" / "server" / "ui" / "js"
TOPBAR = UI_JS / "topbar.js"
API_JS = UI_JS / "api.js"
APP_JS = UI_JS / "app.js"
STATUS_JS = UI_JS / "tabs" / "status.js"
RESULTS_JS = UI_JS / "tabs" / "results.js"
POSTPROCESS_JS = UI_JS / "tabs" / "postprocess.js"
SETTINGS_JS = UI_JS / "tabs" / "settings.js"
INDEX_HTML = REPO_ROOT / "leafmachine3" / "server" / "ui" / "index.html"

NOW = "2026-08-28T12:00:00Z"
CLI_RUN_ID = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
SETUP_RUN_ID = "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"
CALIB_RUN_ID = "cccccccc-3333-4333-8333-cccccccccccc"

#: The run name the CLI run uses. Deliberately NOT the settings file's ``run_name``: the exit gate
#: is that identity comes from the lease record and never from mutable YAML (invariant 5), and the
#: two have to differ for that to be provable rather than accidentally true.
CLI_RUN_NAME = "cli_started_run"

requires_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_comments(source: str) -> str:
    """Source with block and line comments removed.

    Every deletion below is documented IN a comment that names the thing deleted -- that is
    deliberate, so the next reader learns why it is gone. Asserting on raw text would therefore
    fail on its own explanation, so the comments come out before the check.
    """
    out = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"^\s*//.*$", "", out, flags=re.M)


# ============================================================================================== #
# 1. SOURCE CONTRACT -- the five deletions Step 6 names, and what replaced them
# ============================================================================================== #
class TestTheDeletions:
    """Step 6: "Remove openFresh(), sticky state.newRun suppression, run-name blanking without
    writing YAML, deriving run activity from a status snapshot, and Close/New paths that stop a
    job." Each is checked where it lived."""

    def test_open_fresh_is_gone(self) -> None:
        """Opening the app was implemented as "start a new run": it blanked the project name.

        That is the single line the exit gate contradicted -- during a CLI run the GUI opened,
        blanked the name box, and broadcast a reset that wiped three tabs.
        """
        body = strip_comments(read(TOPBAR))
        assert "openFresh" not in body
        assert "resetForNewRun" not in body, "the client-side reset went with it"

    def test_the_sticky_new_run_suppression_is_gone_from_both_copies(self) -> None:
        """There were two: topbar's ``state.newRun`` and status.js's module-scope ``newRunPending``.

        Both dropped every frame that was not "running and not stale", which is exactly why a
        ``done`` / ``error`` / stalled CLI run had no representation in this renderer at all.
        """
        assert "state.newRun" not in strip_comments(read(TOPBAR))
        assert "newRunPending" not in strip_comments(read(STATUS_JS))

    def test_nothing_blanks_the_run_name_without_writing_it(self) -> None:
        """The old reset emptied the field in the GUI and deliberately did NOT write the empty
        value, so the yaml still named the previous project while the box showed nothing."""
        body = strip_comments(read(TOPBAR))
        assert 'setFieldValue(nameField, "")' not in body
        assert '"unnamed"' not in body, "the chip's placeholder for a blanked name is gone too"

    def test_close_never_stops_a_run(self) -> None:
        """Invariant 9. Closing the window and killing the job were the same gesture."""
        body = strip_comments(read(TOPBAR))
        close = body[body.index("async function closeApp"):]
        close = close[:close.index("\n  }\n")]
        assert "stopRun" not in close and "/v1/run/stop" not in close
        assert "lm3desktop" in close, "it still quits the shell -- it just does not stop anything"

    def test_new_run_became_prepare_next_run_and_stops_nothing(self) -> None:
        body = read(TOPBAR)
        assert "Prepare next run" in body
        assert "startNewRun" not in strip_comments(body)
        prep = body[body.index("function prepareNextRun"):]
        prep = prep[:prep.index("\n  }\n")]
        assert "stopRun" not in prep and "/v1/run/stop" not in prep
        assert 'dispatch("lm3:prepare-next-run"' in prep

    def test_the_pid_no_longer_appears_as_evidence_of_ownership(self) -> None:
        """Section 2.5 removes the PID from ownership; "Running (adopted)" and ``pid 1234`` in the
        state badge were the user-visible claim that it meant something."""
        body = strip_comments(read(TOPBAR))
        assert "adopted" not in body
        assert "pid ${" not in body and "r.pid" not in body

    def test_run_activity_is_no_longer_derived_from_the_status_snapshot(self) -> None:
        """A ledger reading is an observation, never an authority (section 2.5).

        ``renderRunControls`` decides Start/Stop; it may read the snapshot for ``stale`` (a
        presentation detail) but the activity, the occupancy and the stop right come from the
        record.
        """
        body = read(TOPBAR)
        fn = body[body.index("function renderRunControls()"):]
        fn = fn[:fn.index("\n  }\n")]
        assert "state.view" in fn and "state.runtime.active" in fn
        assert 's.state === "running"' not in fn
        assert "v.canStart" in fn and "v.canStop" in fn


class TestTheReplacements:
    """"Replace top-bar state with runtime.active, runtime.last, view.mode, view.runRef, settings,
    busy, fed by GET /v1/runtime.\""""

    def test_the_state_object_is_the_one_step_6_specifies(self) -> None:
        body = read(TOPBAR)
        block = body[body.index("  const state = {"):body.index("  const refs = {};")]
        for key in ("runtime:", "view:", "selected:", "settings:", "busy:"):
            assert key in block, f"{key} missing from the top-bar state"
        assert "run:" not in block, "the /v1/run/active projection is not state any more"

    def test_the_runtime_route_is_what_feeds_it(self) -> None:
        assert '"/v1/runtime"' in read(API_JS)
        assert "api.getRuntime()" in read(TOPBAR)

    def test_the_legacy_route_survives_only_as_a_named_fallback(self) -> None:
        """Section 5 keeps ``/v1/run/active`` for one transition release. The renderer may use it
        when ``/v1/runtime`` 404s, but it must SAY that the answer is narrower rather than
        reporting "idle" through somebody's CLI run."""
        body = read(TOPBAR)
        assert "legacyRuntime" in body
        assert "supported: false" in body
        assert "reports only runs it started itself" in body

    def test_the_controller_is_no_longer_discarded(self) -> None:
        """``initTopBar()``'s return value was thrown away at the call site, which made
        ``refresh()`` -- documented in its own source as the escape hatch for a CLI-started run --
        unreachable for the life of the app."""
        assert "topbar = initTopBar(" in read(APP_JS)

    def test_every_tab_that_shows_a_run_follows_the_same_reference(self) -> None:
        """Invariant 7. Before this, the top bar, the Status tab and the Console each resolved a
        run independently, so they could describe three different ones."""
        # The Settings tab is not in this list: it stopped describing a run when its next-run
        # card was removed (2026-10-08), so it has nothing to follow.
        for path in (STATUS_JS, RESULTS_JS, POSTPROCESS_JS):
            assert 'addEventListener("lm3:runtime"' in read(path), path.name
        assert 'addEventListener("lm3:newrun"' not in read(STATUS_JS)
        assert 'addEventListener("lm3:newrun"' not in read(RESULTS_JS)
        assert 'addEventListener("lm3:newrun"' not in read(POSTPROCESS_JS)

    def test_the_status_and_log_streams_are_pinned_to_that_reference(self) -> None:
        """Section 2.6's ``pipeline`` row: "follow its project DB and logs"."""
        body = read(STATUS_JS)
        assert "function streamPin()" in body
        assert "params: streamPin()" in body
        assert "...(streamPin() || {})" in body

    def test_repinning_the_status_stream_drops_the_previous_runs_snapshot(self) -> None:
        """Section 9: the run the bar NAMES and the "active stage and progress" it paints must
        agree. ``state.snapshot`` is the previous run's progress detail; if re-pinning the stream
        to a different ledger leaves it in place, the bar paints the old run's counters, ETA and
        module timeline under the new run's name until the first frame arrives -- the Status tab
        already clears on this exact edge (``resetForNewRun``). The clear must be CONDITIONAL on
        the pin having moved: unconditional, it would also fire on the boot fall-through.
        """
        body = read(TOPBAR)
        start = body.index("  function connectStatus(ref) {")
        block = body[start:body.index('    setConn("wait");', start)]
        assert "const moved = pin !== state.statusPin;" in block, \
            "the pin move must be captured before state.statusPin is overwritten"
        assert "state.snapshot = null;" in block
        assert "state.snapshotAt = 0;" in block
        # Cleared only on the move, never on every reconnect.
        assert re.search(r"if \(moved\) \{", block)

    # The next-run labels (invariant 8) were removed on 2026-10-08: the top strip's line and the
    # Settings tab's card both went, at Will's direction, so their source contracts went with them.

    def test_the_results_run_selector_can_name_the_live_row_and_return_to_it(self) -> None:
        """Step 6: "Add a run selector over the existing results run list." The picker already
        existed; what it could not do was say which row is live or hand the choice back."""
        body = read(RESULTS_JS)
        assert "api.runSelector(" in body
        assert "running now" in body
        assert "Return to the active run" in body
        assert 'CustomEvent("lm3:select-run"' in body

    def test_the_start_and_stop_buttons_are_actually_wired_to_the_decision(self) -> None:
        """The narrow structural check the pure-logic tests cannot make: the computed answer is
        applied to the DOM. It cannot prove the button LOOKS disabled; it can prove nothing else
        assigns those two properties."""
        body = read(TOPBAR)
        assert "refs.startBtn.disabled = !v.canStart;" in body
        assert "refs.stopBtn.disabled = !v.canStop;" in body
        assert len(re.findall(r"refs\.startBtn\.disabled\s*=", body)) == 1
        assert len(re.findall(r"refs\.stopBtn\.disabled\s*=", body)) == 1


# ============================================================================================== #
# 2. PURE LOGIC -- section 2.6's decision table, run by node against the real topbar.js
# ============================================================================================== #
HARNESS = """
import { deriveView, normalizeRuntime, legacyRuntime, VIEW_MODE }
  from "%(topbar)s";

const cases = JSON.parse(process.argv[2]);
const out = cases.map((c) => {
  const runtime = c.legacy ? legacyRuntime(c.payload) : normalizeRuntime(c.payload);
  const view = deriveView(runtime, c.opts || {});
  return { name: c.name, runtime, view, modes: VIEW_MODE };
});
process.stdout.write(JSON.stringify(out));
"""


def run_cases(tmp_path: Path, cases: list[dict]) -> dict[str, dict]:
    """Execute ``deriveView`` in node, importing the REAL renderer module.

    Importing the shipped file is the whole point: a copy of the logic in the test would pass
    forever while the renderer drifted. ``api.js`` (which topbar imports) is guarded so it loads
    without a DOM precisely so this is possible.
    """
    script = tmp_path / "derive_harness.mjs"
    script.write_text(HARNESS % {"topbar": TOPBAR.as_uri()}, encoding="utf-8")
    proc = subprocess.run(
        ["node", str(script), json.dumps(cases)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    return {row["name"]: row for row in json.loads(proc.stdout)}


def pipeline_record(run_name: str = CLI_RUN_NAME, run_id: str = CLI_RUN_ID,
                    state: str = "running", root: str = "/out") -> dict:
    """A ``pipeline`` record in exactly the shape ``records.record_to_dict`` emits."""
    run_dir = f"{root}/{run_name}"
    return {
        "schema_version": T.SCHEMA_VERSION, "run_id": run_id, "activity": "pipeline",
        "activity_role": "root", "parent_run_id": None, "state": state, "launcher": "cli",
        "pid": 4321, "process_started_at": 1787932800.25, "started_at": NOW, "updated_at": NOW,
        "deployment": {"id": "default", "scheduler": None, "job_id": None, "step_id": None,
                       "node": "gpu042", "container_id": None},
        "error": None, "finished_at": None, "returncode": None,
        "config": {"path": "/cfg/LM3_settings.yaml", "sha256": "a" * 64},
        "project": {
            "run_name": run_name, "input_dirs": ["/in"], "artifact_dir": run_dir,
            "active_state_dir": run_dir, "active_db_path": f"{run_dir}/{run_name}.sqlite",
            "archive_mode": "in_place", "archive_status": "not_applicable",
            "archive_pointer_path": None, "archived_db_path": f"{run_dir}/{run_name}.sqlite",
            "run_dir": run_dir, "log_path": f"{run_dir}/logs/lm3.log", "archive_error": None,
        },
    }


def active_block(record: Optional[dict], **overrides: Any) -> dict:
    """``GET /v1/runtime``'s ``active`` object around a record."""
    block = {
        "occupied": True, "compatible": True,
        "classification": "live", "schema_version": T.SCHEMA_VERSION, "message": None,
        "run_id": record["run_id"] if record else None,
        "activity": record["activity"] if record else None,
        "state": record["state"] if record else "unknown",
        "record": record, "current_child": None, "last_child": None,
        "can_stop": False, "can_stop_reason": "this server did not launch that run",
    }
    block.update(overrides)
    return block


@requires_node
class TestSection26:
    """The activity table, the compatibility row, and the follow/pause rule."""

    def test_a_pipeline_root_is_followed_by_default(self, tmp_path: Path) -> None:
        rows = run_cases(tmp_path, [{
            "name": "cli",
            "payload": {"active": active_block(pipeline_record()), "last": None},
        }])
        view = rows["cli"]["view"]
        assert view["mode"] == "follow-active"
        assert view["runRef"]["run_name"] == CLI_RUN_NAME
        assert view["runRef"]["db_path"] == f"/out/{CLI_RUN_NAME}/{CLI_RUN_NAME}.sqlite"
        assert view["runRef"]["log_path"] == f"/out/{CLI_RUN_NAME}/logs/lm3.log"
        assert view["live"] is True and view["occupied"] is True

    def test_a_running_run_reads_the_active_db_and_a_finished_one_the_archived(
        self, tmp_path: Path
    ) -> None:
        """Section 6 item 59: "the GUI reads ``active_db_path`` while running and
        ``archived_db_path`` after finalization, never a deleted scratch path"."""
        done = pipeline_record(state="done")
        done["project"]["active_db_path"] = "/scratch/gone.sqlite"
        done["project"]["archived_db_path"] = f"/out/{CLI_RUN_NAME}/{CLI_RUN_NAME}.sqlite"
        rows = run_cases(tmp_path, [
            {"name": "live", "payload": {"active": active_block(pipeline_record()), "last": None}},
            {"name": "finished", "payload": {"active": None, "last": done}},
        ])
        assert rows["live"]["view"]["runRef"]["db_path"].endswith(f"{CLI_RUN_NAME}.sqlite")
        assert "/scratch/" not in rows["finished"]["view"]["runRef"]["db_path"]

    def test_a_hardware_setup_root_shows_a_tuning_state_and_keeps_project_history(
        self, tmp_path: Path
    ) -> None:
        """Section 2.6, row 2 -- both halves.

        The record carries no project block at all (invariant 6), and its calibration child runs
        under the name ``_lm3_calibration``. The previous run must stay in the project views.
        """
        setup = {
            "schema_version": T.SCHEMA_VERSION, "run_id": SETUP_RUN_ID,
            "activity": "hardware_setup", "activity_role": "root", "state": "running",
            "launcher": "cli", "pid": 99, "process_started_at": 1.0,
            "started_at": NOW, "updated_at": NOW,
            "deployment": {"id": "default"},
            "config": {"path": "/cfg/LM3_settings.yaml", "sha256": "b" * 64},
            "hardware": {"destination_path": "/cfg/hardware_settings.yaml"},
        }
        block = active_block(setup, current_child={
            "run_id": CALIB_RUN_ID, "activity": "calibration_pipeline", "state": "running",
            "started_at": NOW, "run_name": T.CALIBRATION_RUN_NAME,
            "run_dir": f"/tmp/x/{T.CALIBRATION_RUN_NAME}", "record_available": True,
            "updated_at": NOW, "finished_at": None, "returncode": None, "error": None,
        })
        rows = run_cases(tmp_path, [{
            "name": "tuning",
            "payload": {"active": block, "last": pipeline_record(state="done")},
        }])
        view = rows["tuning"]["view"]
        assert view["machine"]["tuning"] is True
        assert view["machine"]["child"]["activity"] == "calibration_pipeline"
        assert view["runRef"]["run_name"] == CLI_RUN_NAME, "project history must not move"
        assert T.CALIBRATION_RUN_NAME not in json.dumps(view["runRef"])
        assert view["canStart"] is False, "the deployment is occupied by the tuning root"

    def test_a_calibration_scratch_project_is_never_offered_as_a_project(
        self, tmp_path: Path
    ) -> None:
        """Belt and braces beside the activity test: even a record that names ``_lm3_calibration``
        in its project block must not reach the run picker under that name (section 2.2)."""
        rec = pipeline_record(run_name=T.CALIBRATION_RUN_NAME)
        rows = run_cases(tmp_path, [{
            "name": "calib", "payload": {"active": active_block(rec), "last": None},
        }])
        assert rows["calib"]["view"]["runRef"] is None

    def test_an_unknown_activity_disables_start_and_refuses_control(self, tmp_path: Path) -> None:
        """Section 2.6's last row: an unknown/future activity refuses CONTROL, not just Start.

        The payload is deliberately adversarial. ``known`` is the renderer's OWN test -- no server
        computes it -- so a newer server that understands ``quantum_reticulation`` reports it as
        perfectly compatible and stoppable. The refusal has to come from this side."""
        rec = pipeline_record()
        rec["activity"] = "quantum_reticulation"
        rows = run_cases(tmp_path, [{
            "name": "future",
            "payload": {"active": active_block(rec, activity="quantum_reticulation",
                                               compatible=True, can_stop=True,
                                               can_stop_reason=None),
                        "last": None},
        }])
        view = rows["future"]["view"]
        assert view["mode"] == "incompatible"
        assert view["canStart"] is False and view["canStop"] is False
        assert "quantum_reticulation" in view["warning"]
        # And it must say WHY it refused -- not the ownership fallback, which would be a lie here.
        assert "quantum_reticulation" in view["stopReason"]
        assert "not launched by this server" not in view["stopReason"]

    def test_a_newer_schema_disables_everything_and_says_why(self, tmp_path: Path) -> None:
        """Section 2.9: the lock still decides occupancy, nothing unknown is interpreted, every
        control action is off, and the message is shown."""
        rows = run_cases(tmp_path, [{
            "name": "skew",
            "payload": {"active": active_block(
                None, compatible=False, classification="incompatible", schema_version=99,
                message="This runtime was created by a newer LM3", can_stop=True,
                can_stop_reason=None), "last": None},
        }])
        view = rows["skew"]["view"]
        assert view["mode"] == "incompatible"
        assert view["occupied"] is True, "the OS lock decides occupancy, not the JSON"
        # `can_stop: True` is the adversarial half: "every control action is off" is a READER
        # obligation, so satisfying it only because the writer said no does not satisfy it.
        assert view["canStart"] is False and view["canStop"] is False
        assert "newer LM3" in view["warning"]
        assert "newer LM3" in view["stopReason"]

    def test_a_historical_selection_pauses_follow_active_and_can_be_undone(
        self, tmp_path: Path
    ) -> None:
        selected = {"source": "history", "live": False, "run_id": "old", "run_name": "last_march",
                    "run_dir": "/out/last_march", "db_path": "/out/last_march/last_march.sqlite"}
        rows = run_cases(tmp_path, [
            {"name": "paused",
             "payload": {"active": active_block(pipeline_record()), "last": None},
             "opts": {"selected": selected}},
            {"name": "resumed",
             "payload": {"active": active_block(pipeline_record()), "last": None},
             "opts": {"selected": None}},
        ])
        paused = rows["paused"]["view"]
        assert paused["mode"] == "history" and paused["followPaused"] is True
        assert paused["runRef"]["run_name"] == "last_march"
        # Pausing the VIEW never pauses the truth: the deployment is still occupied.
        assert paused["occupied"] is True and paused["canStart"] is False
        assert rows["resumed"]["view"]["followPaused"] is False
        assert rows["resumed"]["view"]["runRef"]["run_name"] == CLI_RUN_NAME


@requires_node
class TestControlAuthority:
    """Section 2.5 as the renderer must present it: the server decides, the UI quotes it."""

    def test_stop_is_disabled_for_a_run_this_server_did_not_launch_and_says_why(
        self, tmp_path: Path
    ) -> None:
        rows = run_cases(tmp_path, [{
            "name": "observer",
            "payload": {"active": active_block(
                pipeline_record(),
                can_stop=False,
                can_stop_reason="this server did not launch that run"), "last": None},
        }])
        view = rows["observer"]["view"]
        assert view["canStop"] is False
        assert view["stopReason"] == "this server did not launch that run"

    def test_stop_is_enabled_only_when_the_server_says_can_stop(self, tmp_path: Path) -> None:
        rows = run_cases(tmp_path, [{
            "name": "ours",
            "payload": {"active": active_block(pipeline_record(), can_stop=True,
                                               can_stop_reason=None), "last": None},
        }])
        assert rows["ours"]["view"]["canStop"] is True

    def test_start_is_disabled_whenever_the_lease_is_occupied(self, tmp_path: Path) -> None:
        """The visible symptom Step 6 exists to fix: during a CLI run the old GUI left Start
        ENABLED, and pressing it launched a second LM3 on the same GPUs."""
        rows = run_cases(tmp_path, [
            {"name": "busy", "payload": {"active": active_block(pipeline_record()), "last": None}},
            {"name": "idle", "payload": {"active": None, "last": pipeline_record(state="done")}},
        ])
        assert rows["busy"]["view"]["canStart"] is False
        assert "already has a run in progress" in rows["busy"]["view"]["startReason"]
        assert rows["idle"]["view"]["canStart"] is True

    def test_the_legacy_fallback_is_narrower_and_says_so(self, tmp_path: Path) -> None:
        rows = run_cases(tmp_path, [{
            "name": "legacy", "legacy": True,
            "payload": {"active": False, "state": "idle", "run_name": None},
        }])
        assert rows["legacy"]["runtime"]["supported"] is False
        assert "started itself" in rows["legacy"]["view"]["warning"]


# ============================================================================================== #
# 3. AGREEMENT -- section 9, as far as a headless environment reaches
# ============================================================================================== #
@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Sandbox]:
    reset_server_module_state()
    box = isolate_server_paths(tmp_path, monkeypatch)
    monkeypatch.setenv("LM3_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("LM3_RUNTIME_V2", "1")
    from leafmachine3.server import metrics_api, progress_api

    metrics_api.invalidate_runtime_cache()
    progress_api.invalidate_run_caches()
    try:
        yield box
    finally:
        metrics_api.invalidate_runtime_cache()
        progress_api.invalidate_run_caches()
        reset_server_module_state()


@pytest.fixture
def client(sandbox: Sandbox) -> Iterator[TestClient]:
    from leafmachine3.server.app import JobManager, create_app

    yield TestClient(create_app(JobManager(sandbox.jobs_root / "managed")))


def publish_cli_run(sandbox: Sandbox) -> tuple[Path, Any]:
    """Publish a LIVE ``pipeline`` record and hold the real deployment lock.

    This is the closest a headless test gets to "a CLI run is in progress": the record the CLI
    would have written, and the lock it would be holding. Nothing is executed.
    """
    from leafmachine3.core.runtime import lease as lease_module
    from leafmachine3.server import metrics_api

    directory = metrics_api.deployment_dir()
    (directory / T.CHILDREN_DIRNAME).mkdir(parents=True, exist_ok=True)

    run_dir = sandbox.output_dir / CLI_RUN_NAME
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    db = run_dir / f"{CLI_RUN_NAME}.sqlite"
    db.write_bytes(b"")
    (run_dir / "logs" / "lm3.log").write_text("started\n", encoding="utf-8")

    record = T.RuntimeRecord(
        run_id=CLI_RUN_ID,
        activity=T.Activity.PIPELINE,
        activity_role=T.ActivityRole.ROOT,
        state=T.RunState.RUNNING,
        launcher=T.Launcher.CLI,
        pid=4321,
        process_started_at=1787932800.25,
        started_at=NOW,
        updated_at=NOW,
        deployment=T.DeploymentInfo(id="default"),
        config=T.ConfigRef(path=str(sandbox.settings_path), sha256="a" * 64),
        project=T.ProjectBlock(
            run_name=CLI_RUN_NAME,
            input_dirs=(str(sandbox.input_dir),),
            artifact_dir=str(run_dir),
            active_state_dir=str(run_dir),
            active_db_path=str(db),
            archive_mode=T.ArchiveMode.IN_PLACE,
            archive_status=T.ArchiveStatus.NOT_APPLICABLE,
            archive_pointer_path=None,
            archived_db_path=str(db),
            run_dir=str(run_dir),
            log_path=str(run_dir / "logs" / "lm3.log"),
        ),
    )
    R.atomic_write_json(directory / T.ACTIVE_RECORD_FILENAME, R.record_to_dict(record))

    adapter = lease_module.lease_adapter(directory, deployment_key=directory.name)
    adapter.acquire()

    metrics_api.invalidate_runtime_cache()
    from leafmachine3.server import progress_api

    progress_api.invalidate_run_caches()
    return run_dir, adapter


@pytest.mark.skipif(os.name == "nt", reason="the POSIX lease adapter is the one exercised here")
class TestTheExitGate:
    """Section 9: open the GUI during a CLI run and have everything agree.

    The settings file names a DIFFERENT project (``contract_run``) throughout. That is the point:
    invariant 5 says identity comes from the lease record and never from mutable YAML, and the two
    must disagree for the assertion to mean anything.
    """

    def test_the_settings_file_names_a_different_project(self, sandbox: Sandbox) -> None:
        assert sandbox.run_name != CLI_RUN_NAME, "the premise of every test below"

    def test_the_runtime_route_reports_the_cli_run(
        self, sandbox: Sandbox, client: TestClient
    ) -> None:
        run_dir, adapter = publish_cli_run(sandbox)
        try:
            body = client.get("/v1/runtime", headers=bearer()).json()
        finally:
            adapter.release()
        active = body["active"]
        assert active is not None and active["occupied"] is True
        assert active["run_id"] == CLI_RUN_ID
        assert active["activity"] == "pipeline"
        assert active["record"]["project"]["run_name"] == CLI_RUN_NAME
        assert active["record"]["project"]["run_dir"] == str(run_dir)
        # Section 2.5: this server launched nothing, so it may not stop it, and it says why.
        assert active["can_stop"] is False
        assert active["can_stop_reason"]
        # Invariant 8: the editable settings are published as a SEPARATE, labeled concern.
        assert body["next_run_settings"]["applies_to"] == "next run"
        assert body["next_run_settings"]["run_name"] == sandbox.run_name

    def test_status_logs_results_and_postprocessing_all_name_the_same_run(
        self, sandbox: Sandbox, client: TestClient
    ) -> None:
        """Invariant 7, across the four surfaces that have to agree."""
        run_dir, adapter = publish_cli_run(sandbox)
        try:
            runtime = client.get("/v1/runtime", headers=bearer()).json()
            status = client.get("/v1/status", headers=bearer()).json()
            logs = client.get("/v1/logs", headers=bearer(), params={"limit": 5}).json()
            selector = client.get("/v1/runs/-/selector", headers=bearer()).json()
            context = client.get("/v1/postprocess/context", headers=bearer()).json()
        finally:
            adapter.release()

        project = runtime["active"]["record"]["project"]
        assert status["run_name"] == CLI_RUN_NAME
        assert status["db_path"] == project["active_db_path"]
        assert status["source"] == "runtime", "identity came from the record, not from discovery"
        assert logs["run_name"] == CLI_RUN_NAME
        assert selector["active"] is not None
        assert selector["active"]["run_name"] == CLI_RUN_NAME
        assert selector["active"]["run_id"] == CLI_RUN_ID
        assert selector["follow"] == "active"
        assert context["active_run"] is not None
        assert context["active_run"]["run_name"] == CLI_RUN_NAME
        assert Path(context["active_run"]["artifact_dir"]).resolve() == run_dir.resolve()

    def test_editing_the_settings_file_mid_run_moves_nothing(
        self, sandbox: Sandbox, client: TestClient
    ) -> None:
        """Invariant 8. This is what ``PUT /v1/settings`` does, and it used to move the view."""
        run_dir, adapter = publish_cli_run(sandbox)
        try:
            sandbox.write_settings(run_name="renamed_mid_run")
            from leafmachine3.server import progress_api

            progress_api.invalidate_run_caches()
            status = client.get("/v1/status", headers=bearer()).json()
            runtime = client.get("/v1/runtime", headers=bearer()).json()
        finally:
            adapter.release()
        assert status["run_name"] == CLI_RUN_NAME
        assert status["db_path"] == str(run_dir / f"{CLI_RUN_NAME}.sqlite")
        assert runtime["active"]["record"]["project"]["run_name"] == CLI_RUN_NAME
        # ...and the NEXT run is the thing that moved, which is what the label promises.
        assert runtime["next_run_settings"]["run_name"] == "renamed_mid_run"

    @requires_node
    def test_the_renderer_agrees_with_the_server_on_all_of_it(
        self, sandbox: Sandbox, client: TestClient, tmp_path: Path
    ) -> None:
        """The join: the REAL ``GET /v1/runtime`` payload through the REAL ``deriveView``.

        This is as close to "open the GUI during a CLI run" as a headless environment reaches. It
        proves the renderer's decision, not its painting -- see the module docstring.
        """
        run_dir, adapter = publish_cli_run(sandbox)
        try:
            runtime = client.get("/v1/runtime", headers=bearer()).json()
            status = client.get("/v1/status", headers=bearer()).json()
        finally:
            adapter.release()

        rows = run_cases(tmp_path, [{"name": "gui", "payload": runtime}])
        view = rows["gui"]["view"]

        assert view["mode"] == "follow-active"
        assert view["runRef"]["run_name"] == CLI_RUN_NAME == status["run_name"]
        assert view["runRef"]["run_id"] == CLI_RUN_ID
        assert view["runRef"]["run_dir"] == str(run_dir)
        assert view["runRef"]["db_path"] == status["db_path"]
        assert view["runRef"]["log_path"] == str(run_dir / "logs" / "lm3.log")
        assert view["runRef"]["config_path"] == str(sandbox.settings_path)
        assert view["live"] is True
        # Start disabled BECAUSE the lease is occupied; Stop refused with the server's own reason.
        assert view["canStart"] is False
        assert view["canStop"] is False
        assert view["stopReason"] == runtime["active"]["can_stop_reason"]

    @requires_node
    def test_the_renderer_reports_stop_as_available_when_the_server_grants_it(
        self, sandbox: Sandbox, client: TestClient, tmp_path: Path
    ) -> None:
        """The other half of "Stop capability accurately reflecting ownership": with a retained
        handle for this run id, the server says yes and the renderer follows it."""
        from leafmachine3.server import metrics_api

        run_dir, adapter = publish_cli_run(sandbox)
        try:
            metrics_api._RUN = _launched_run(CLI_RUN_ID, run_dir)
            metrics_api.invalidate_runtime_cache()
            runtime = client.get("/v1/runtime", headers=bearer()).json()
        finally:
            metrics_api._RUN = None
            adapter.release()
        assert runtime["active"]["can_stop"] is True
        rows = run_cases(tmp_path, [{"name": "ours", "payload": runtime}])
        assert rows["ours"]["view"]["canStop"] is True


class _Handle:
    """A retained live child handle -- the ONLY thing section 2.5 lets authorize a stop."""

    def __init__(self, run_id: str, instance_id: str) -> None:
        self.run_id = run_id
        self.instance_id = instance_id
        self.kind = "pipeline"
        self.alive = True
        self.pid = 2 ** 31 - 33
        self.pgid = self.pid

    def signal_group(self, sig: int) -> bool:
        return True

    def release_job(self) -> None:
        pass


def _launched_run(run_id: str, run_dir: Path) -> Any:
    from leafmachine3.server import metrics_api

    run = metrics_api._Run(
        run_name=CLI_RUN_NAME, run_dir=str(run_dir),
        db_path=str(run_dir / f"{CLI_RUN_NAME}.sqlite"),
        log_path=str(run_dir / "logs" / "lm3.log"),
    )
    run.child = _Handle(run_id, metrics_api.SERVER_INSTANCE_ID)
    run.run_id = run_id
    run.state = "running"
    return run
# ============================================================================================== #
# 4. THE SETTINGS TAB'S NEXT-RUN CARD -- invariant 8 stated about the RIGHT run
# ============================================================================================== #
