"""Who is allowed to pause follow-active -- plan section 2.6, invariant 7, the Step 6 exit gate.

Plan reference: ``UNIFIED_RUNTIME_IMPLEMENTATION_PLAN_CONSENSUS.md``

* Invariant 7: "In follow-active mode, which is the default, status, logs, results, and
  postprocessing targets follow the active record's paths. An explicit ``run=``/``db=`` selector
  deliberately overrides this for that client request and visibly pauses follow-active."
* Section 2.6: "Default mode is follow-active. ... Historical selection pauses follow-active
  explicitly, with a visible action to return."
* Step 6 exit gate: "opening the GUI during any CLI run immediately shows that run".

The word doing the work is EXPLICIT. The top bar treats any non-null ``lm3:select-run`` reference
as a user's historical choice and pauses the whole window on it -- status SSE, console and results
together -- and ``applyRuntime`` un-pins only a selection that IS the newly active run. So a
reference announced by the Results tab for a selection the tab made ITSELF (its mount-time default,
or its echo of a runtime frame) pins the window onto a finished run and no later CLI run can take
it back. The exit gate is then broken by the renderer, in the gate's own literal ordering: the app
restores the last tab from ``localStorage["lm3.tab"]``, and Results is a normal choice.

``tests/test_renderer_contract.py`` drives ``deriveView`` with no selection, so it cannot see this;
these tests are the missing half. Two kinds, matching that file's conventions:

1. **Source contract** over the shipped ``results.js`` -- which call sites may announce.
2. **Pure logic** -- the real ``topbar.js`` ``deriveView`` under node, showing what a non-user
   announcement would cost and what the gate requires instead.

There is no DOM here, so "the tab did not dispatch the event" is asserted as "the tab's only
dispatcher is gated on ``explicit``", not by catching an event.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import pytest

from leafmachine3.core.runtime import _types as T

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_JS = REPO_ROOT / "leafmachine3" / "server" / "ui" / "js"
TOPBAR = UI_JS / "topbar.js"
RESULTS_JS = UI_JS / "tabs" / "results.js"

requires_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

CLI_RUN_ID = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
CLI_RUN_NAME = "cli_started_run"
NOW = "2026-08-28T12:00:00Z"


def strip_comments(source: str) -> str:
    """Source with block and line comments removed -- the reasons are written IN comments."""
    out = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"^\s*//.*$", "", out, flags=re.M)


# ============================================================================================== #
# 1. SOURCE CONTRACT -- only an explicit action announces a selection
# ============================================================================================== #
class TestOnlyAnExplicitActionAnnounces:
    """Invariant 7: an explicit selector pauses follow-active. Nothing else may."""

    def test_select_run_takes_an_explicit_flag_and_announces_only_under_it(self) -> None:
        src = strip_comments(RESULTS_JS.read_text(encoding="utf-8"))
        assert "async function selectRun(runId, { explicit = false } = {}) {" in src, \
            "selectRun must distinguish a user's pick from the tab's own default"
        assert "if (explicit) announceSelection(next);" in src, \
            "selectRun may announce only for an explicit choice (section 2.6)"
        # Exactly three mentions: the definition, the gated call, and the RESUME form
        # (``ref: null``), which can never pause anything. A fourth would be a new way to pin the
        # window, so the count is the tripwire.
        assert src.count("announceSelection(") == 3, \
            "unexpected announceSelection call site; announcing pauses the WHOLE window"
        assert "announceSelection(null)" in src

    def test_the_user_facing_call_sites_are_the_explicit_ones(self) -> None:
        src = strip_comments(RESULTS_JS.read_text(encoding="utf-8"))
        # The <select> onchange: the only true "pick a run out of history" gesture.
        assert "onchange: (e) => selectRun(e.target.value, { explicit: true })" in src
        # "Return to the active run": explicit too -- it RESUMES, which is the visible action
        # section 2.6 requires.
        assert "void selectRun(target.id, { explicit: true });" in src

    def test_the_tabs_own_default_selection_is_not_announced(self) -> None:
        """``initResults`` auto-selects a row at mount. That is a default, not a decision."""
        src = strip_comments(RESULTS_JS.read_text(encoding="utf-8"))
        assert "if (S.run) selectRun(S.run.id);" in src, \
            "the mount-time default must not pass explicit:true"

    def test_the_runtime_listener_does_not_echo_the_top_bars_decision_back(self) -> None:
        """The listener follows ``view.runRef``; announcing it back is at best a no-op."""
        src = strip_comments(RESULTS_JS.read_text(encoding="utf-8"))
        assert "if (match && (!S.run || S.run.id !== match.id)) void selectRun(match.id);" in src

    def test_a_starting_run_is_still_recognized_as_the_active_one(self) -> None:
        """``_record_ref`` publishes ``id: null`` until the run directory is discoverable.

        An id-only comparison therefore calls the live run historical for the whole ``starting``
        window -- which is exactly when the exit gate's "opening the GUI during any CLI run" lands.
        ``run_id`` and ``run_name`` are always published, so the identity rule uses all three.
        """
        src = strip_comments(RESULTS_JS.read_text(encoding="utf-8"))
        assert "function isActiveRun(run) {" in src
        assert "if (active.run_id && run.run_id === active.run_id) return true;" in src
        assert 'return Boolean(active.run_name && run.name === active.run_name);' in src
        # And it is the single rule: no id-only survivors in the three places that ask the question.
        assert "const isActive = isActiveRun(run);" in src           # announceSelection
        assert "const following = isActiveRun(S.run);" in src        # renderFollowState
        assert "if (isActiveRun(r)) bits.push(\"running now\");" in src   # fillRunSelect
        assert "S.activeRef && S.activeRef.id" not in src, \
            "an id-only active test is what mislabels a starting run"


# ============================================================================================== #
# 2. PURE LOGIC -- what a non-user announcement would cost, run against the real topbar.js
# ============================================================================================== #
HARNESS = """
import { deriveView, normalizeRuntime, VIEW_MODE } from "%(topbar)s";

const cases = JSON.parse(process.argv[2]);
const out = cases.map((c) => {
  const runtime = normalizeRuntime(c.payload);
  const view = deriveView(runtime, c.opts || {});
  return { name: c.name, view, modes: VIEW_MODE };
});
process.stdout.write(JSON.stringify(out));
"""


def run_cases(tmp_path: Path, cases: list[dict]) -> dict[str, dict]:
    """Execute ``deriveView`` in node, importing the REAL topbar module (never a copy)."""
    script = tmp_path / "select_harness.mjs"
    script.write_text(HARNESS % {"topbar": TOPBAR.as_uri()}, encoding="utf-8")
    proc = subprocess.run(
        ["node", str(script), json.dumps(cases)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    return {row["name"]: row for row in json.loads(proc.stdout)}


def pipeline_record(run_name: str = CLI_RUN_NAME, run_id: str = CLI_RUN_ID,
                    state: str = "running", root: str = "/out") -> dict:
    """A ``pipeline`` record in the shape ``records.record_to_dict`` emits."""
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
        "occupied": record is not None, "compatible": True,
        "classification": "live" if record else "none",
        "schema_version": T.SCHEMA_VERSION, "message": None,
        "run_id": record["run_id"] if record else None,
        "activity": record["activity"] if record else None,
        "state": record["state"] if record else "unknown",
        "record": record, "current_child": None, "last_child": None,
        "can_stop": False, "can_stop_reason": "this server did not launch that run",
    }
    block.update(overrides)
    return block


OLD_RUN = {"source": "history", "live": False, "run_id": "old", "run_name": "last_march",
           "run_dir": "/out/last_march", "db_path": "/out/last_march/last_march.sqlite"}


@requires_node
class TestTheExitGateWithASelectionInPlay:
    """Step 6's gate, driven the way a real window reaches it: a tab may already hold a selection.

    ``tests/test_renderer_contract.py`` drives the gate with no selection at all, so it proves the
    server-to-renderer half and nothing about who may create one.
    """

    def test_a_cli_run_is_shown_when_no_one_picked_a_run(self, tmp_path: Path) -> None:
        """The gate itself: default mode is follow-active, so the live run wins the window."""
        rows = run_cases(tmp_path, [{
            "name": "gate",
            "payload": {"active": active_block(pipeline_record()), "last": None},
            "opts": {"selected": None},
        }])
        view = rows["gate"]["view"]
        assert view["mode"] == rows["gate"]["modes"]["FOLLOW"]
        assert view["followPaused"] is False
        assert view["runRef"]["run_name"] == CLI_RUN_NAME
        assert view["runRef"]["live"] is True

    def test_a_selection_survives_a_later_cli_run_which_is_why_only_a_person_may_make_one(
        self, tmp_path: Path
    ) -> None:
        """The cost of announcing a non-user selection, stated as an assertion.

        ``applyRuntime`` clears ``state.selected`` only when it IS the newly active run, so a
        selection for an unrelated finished run outlives every later CLI run. This is why
        ``results.js`` must not announce its own default -- the pin has no expiry.
        """
        rows = run_cases(tmp_path, [
            {"name": "idle_pinned",
             "payload": {"active": active_block(None, occupied=False), "last": None},
             "opts": {"selected": OLD_RUN}},
            {"name": "cli_started_pinned",
             "payload": {"active": active_block(pipeline_record()), "last": None},
             "opts": {"selected": OLD_RUN}},
        ])
        modes = rows["idle_pinned"]["modes"]
        for name in ("idle_pinned", "cli_started_pinned"):
            view = rows[name]["view"]
            assert view["mode"] == modes["HISTORY"], name
            assert view["followPaused"] is True, name
            assert view["runRef"]["run_name"] == "last_march", name
            assert view["runRef"]["live"] is False, name
        # The deployment is still occupied while the window looks elsewhere -- the pin hides the
        # live run without making the machine idle.
        assert rows["cli_started_pinned"]["view"]["occupied"] is True
