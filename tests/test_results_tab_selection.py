"""The Results run picker as a WINDOW-wide control (section 2.6), executed by node.

Section 2.6: "Historical selection pauses follow-active explicitly, with a visible action to
return." Step 6 makes that pause window-wide -- one place decides (`view.mode` on the top bar) and
every tab follows -- so the Results tab's picker is not allowed to change which run it is showing
without saying so on ``lm3:select-run``.

The gap this file closes is the one case where the tab changes the selection on the user's behalf:
a rescan in which the picked run has DISAPPEARED (its directory deleted, moved or renamed). Before
the fix ``loadRuns()`` quietly fell back to the active run and repainted its own badge, leaving the
top bar paused on a run that no longer exists -- "Following the active run" in one strip and
"Viewing a finished run" in the other, with the Status stream still pinned to the deleted ledger.

Method: the REAL ``results.js`` is imported into plain node behind a DOM small enough to run it,
and driven only through what the module exports (``initResults``, ``refresh``, ``currentRun``) plus
the picker's own ``onchange`` handler. Importing the shipped file is the point -- a copy of the
logic in the test would pass forever while the renderer drifted.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_JS = REPO_ROOT / "leafmachine3" / "server" / "ui" / "js"
API_JS = UI_JS / "api.js"
RESULTS_JS = UI_JS / "tabs" / "results.js"

requires_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

LIVE_RUN = "cli_started_run"
OLD_RUN = "old_batch"


#: A DOM small enough to run the Results tab, the two picker calls stubbed, and a fixed script:
#: mount -> a person picks a run out of the selector -> two rescans. Each step records the run the
#: tab is showing and every ``lm3:select-run`` emitted since the previous step, which is the whole
#: observable contract of "the window is told when the selection moves".
HARNESS = r"""
class FakeNode {
  constructor(tag) {
    this.tagName = String(tag || "div").toUpperCase();
    this.children = []; this.style = {}; this.dataset = {}; this.attrs = {}; this.listeners = {};
    this.className = ""; this.id = ""; this.value = ""; this.disabled = false;
    this.textContent = ""; this.innerHTML = "";
    this.classList = { add: () => {}, remove: () => {}, toggle: () => {}, contains: () => false };
  }
  get firstChild() { return this.children[0] || null; }
  appendChild(c) { this.children.push(c); c.parentNode = this; return c; }
  removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); return c; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); }
  removeEventListener() {}
  setAttribute(k, v) { this.attrs[k] = v; }
  getAttribute(k) { return this.attrs[k]; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  focus() {} scrollTo() {} scrollIntoView() {}
  getBoundingClientRect() { return { top: 0, left: 0, width: 0, height: 0 }; }
}
class FakeText extends FakeNode {
  constructor(t) { super("#text"); this.textContent = String(t); }
}
globalThis.Node = FakeNode;                       // api.js `append()` tests `instanceof Node`
const bus = new EventTarget();
globalThis.document = {
  createElement: (t) => new FakeNode(t),
  createTextNode: (t) => new FakeText(t),
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener: bus.addEventListener.bind(bus),
  removeEventListener: bus.removeEventListener.bind(bus),
  dispatchEvent: bus.dispatchEvent.bind(bus),
  body: new FakeNode("body"),
};
globalThis.window = globalThis;
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.IntersectionObserver = class { observe() {} unobserve() {} disconnect() {} };
// Nothing in this test may reach the network: a render path that tries lands in the tab's own
// failure panel, which is a valid outcome here and must not fail the harness.
globalThis.fetch = async () => { throw new Error("the harness makes no network calls"); };
process.on("unhandledRejection", () => {});

const CASE = JSON.parse(process.argv[2]);
const { api } = await import("%(api)s");
const results = await import("%(results)s");

const responses = CASE.responses.slice();
let served = 0;
const nextPayload = async () => {
  const p = responses[Math.min(served, responses.length - 1)];
  served += 1;
  return JSON.parse(JSON.stringify(p));      // each call gets its own copy, as a fetch would
};
api.runSelector = nextPayload;
api.listRuns = nextPayload;
let fetched = [];
api.getResults = async (id) => { fetched.push(id); return { run: null, groups: [], categories: [] }; };

const events = [];
document.addEventListener("lm3:select-run", (ev) => events.push(ev.detail));
const settle = async () => { for (let i = 0; i < 30; i += 1) await Promise.resolve(); };

function findSelect(node) {
  if (node.tagName === "SELECT") return node;
  for (const c of node.children) { const hit = findSelect(c); if (hit) return hit; }
  return null;
}

const trace = [];
const mark = (label) => {
  const run = results.currentRun();
  const seen = fetched.slice(); fetched = [];
  trace.push({ label, run: run ? run.id : null, fetched: seen,
               events: events.splice(0).map((d) => d.ref) });
};

const root = new FakeNode("div");
results.initResults(root);
await settle();
mark("mount");

// A PERSON picks a run: the selector's real onchange handler, not a private function.
const sel = findSelect(root);
for (const fn of sel.listeners.change || []) fn({ target: { value: CASE.pick } });
await settle();
mark("picked");

await results.refresh();
await settle();
mark("rescan_1");

await results.refresh();
await settle();
mark("rescan_2");

process.stdout.write(JSON.stringify(trace));
"""


def _run(name: str, run_id: str, state: str, **extra: object) -> dict:
    """One row of ``GET /v1/runs/-/selector``'s ``runs`` list."""
    row = {
        "id": name, "name": name, "run_id": run_id, "state": state,
        "path": f"/out/{name}", "has_db": True, "db_path": f"/out/{name}/{name}.sqlite",
    }
    row.update(extra)
    return row


def selector_payload(rows: list[dict]) -> dict:
    """A selector payload whose live row -- and whose ``selected_default`` -- is the CLI run.

    ``selected_default`` really is the active run: ``run_selector`` (results_api.py) prefers the
    active reference over the last finished one, so the fallback this test exercises is the one
    production takes.
    """
    return {
        "runtime": {"enabled": True, "compatible": True, "message": None},
        "active": {
            "id": LIVE_RUN, "run_id": "aaaa-1111", "run_name": LIVE_RUN, "source": "runtime",
            "live": True, "state": "running", "activity": "pipeline",
            "run_dir": f"/out/{LIVE_RUN}", "db_path": f"/out/{LIVE_RUN}/{LIVE_RUN}.sqlite",
        },
        "last": None,
        "selected_default": LIVE_RUN,
        "runs": rows,
    }


LIVE_ROW = _run(LIVE_RUN, "aaaa-1111", "running")
OLD_ROW = _run(OLD_RUN, "bbbb-2222", "done")

#: The rescan in which the picked run has vanished from disk.
BOTH = selector_payload([LIVE_ROW, OLD_ROW])
LIVE_ONLY = selector_payload([LIVE_ROW])


def drive(tmp_path: Path, responses: list[dict], pick: str = OLD_RUN) -> dict[str, dict]:
    """Run the real Results module under node against `responses`, keyed by step label."""
    script = tmp_path / "results_selection_harness.mjs"
    script.write_text(
        HARNESS % {"api": API_JS.as_uri(), "results": RESULTS_JS.as_uri()}, encoding="utf-8")
    proc = subprocess.run(
        ["node", str(script), json.dumps({"pick": pick, "responses": responses})],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    return {row["label"]: row for row in json.loads(proc.stdout)}


@requires_node
class TestTheSelectionIsAnnouncedWheneverItMoves:
    """Section 2.6: the pause is window-wide, so the window hears about every move of it."""

    def test_a_vanished_selection_resumes_follow_active(self, tmp_path: Path) -> None:
        """The regression: the picked run disappears between rescans.

        ``loadRuns()`` falls back to ``selected_default`` -- the ACTIVE run -- so the tab starts
        describing a different run. Without an announcement the top bar keeps ``view.mode ==
        "history"`` and ``view.runRef`` pointing at the deleted run, and the Status tab keeps
        streaming that run's ledger, while this tab's own strip reads "Following the active run".
        ``ref: null`` is the one value that ends the pause instead of moving it.
        """
        steps = drive(tmp_path, [BOTH, LIVE_ONLY, LIVE_ONLY])
        assert steps["picked"]["run"] == OLD_RUN
        assert [e["run_name"] for e in steps["picked"]["events"]] == [OLD_RUN], \
            "an explicit pick is announced, and pauses follow-active"

        assert steps["rescan_1"]["run"] == LIVE_RUN, "the gone run cannot stay selected"
        assert steps["rescan_1"]["events"] == [None], \
            "exactly one lm3:select-run, with ref null: follow-active resumes"

    def test_the_replacement_run_replaces_the_body_too(self, tmp_path: Path) -> None:
        """Announcing is half of it: the media/tables state of the vanished run has to go with it.

        Re-selecting through ``selectRun()`` -- rather than repainting the picker in place -- is
        what drops ``S.media``/``S.tables``/paging, so the body is re-fetched for the run whose
        name the header now shows instead of keeping the old run's contents under it.
        """
        steps = drive(tmp_path, [BOTH, LIVE_ONLY, LIVE_ONLY])
        assert steps["rescan_1"]["fetched"] == [LIVE_RUN, LIVE_RUN], \
            "the body is rebuilt for the new run (re-selected, then refresh()'s own reload)"

    def test_a_surviving_selection_is_not_re_announced(self, tmp_path: Path) -> None:
        """The other half: silence when nothing moved.

        A rescan happens on every Refresh, every ``lm3:runtime`` frame and (before any run exists)
        every 20 seconds. Re-announcing an unchanged historical ref on each of those would be
        noise at best, and a periodic re-pin of the whole window at worst.
        """
        steps = drive(tmp_path, [BOTH, BOTH, BOTH])
        assert steps["rescan_1"]["run"] == OLD_RUN and steps["rescan_1"]["events"] == []
        assert steps["rescan_2"]["run"] == OLD_RUN and steps["rescan_2"]["events"] == []

    def test_the_mount_default_stays_silent(self, tmp_path: Path) -> None:
        """Mounting the tab is a default, not a decision.

        ``announceSelection`` is reserved for explicit user action precisely so a mount cannot pin
        the window onto whatever row the tab happens to open on -- the Step 6 exit gate inverted.
        The fix above must not have turned the mount into an announcement.
        """
        steps = drive(tmp_path, [BOTH, BOTH, BOTH])
        assert steps["mount"]["run"] == LIVE_RUN
        assert steps["mount"]["events"] == []
        assert steps["mount"]["fetched"] == [LIVE_RUN], \
            "and it renders the body exactly once, not twice"
