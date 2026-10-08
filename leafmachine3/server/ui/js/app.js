/* LeafMachine3 desktop app -- bootstrap + tab router.
 *
 * Each tab is lazily initialized ONCE, the first time it is shown, so opening the app does not pay
 * for the results grid or the settings tree until they are actually looked at. The top bar and the
 * performance monitor live OUTSIDE the tabs and start immediately -- they must keep updating no
 * matter which tab is in front.
 */
import { api, el, qsa } from "./api.js";
import { initTopBar } from "./topbar.js";
import { initPerfMon } from "./perfmon.js";
import { initStatus, initConsole, focusModule } from "./tabs/status.js";
import { initSettings } from "./tabs/settings.js";
import { initModelsTab } from "./tabs/models.js";
import { refreshModelsStatus } from "./models.js";
import { initResults } from "./tabs/results.js";
import { initPostprocess } from "./tabs/postprocess.js";

const TABS = {
  status: { pane: "pane-status", init: initStatus },
  console: { pane: "pane-console", init: initConsole },
  settings: { pane: "pane-settings", init: initSettings },
  models: { pane: "pane-models", init: initModelsTab },
  results: { pane: "pane-results", init: initResults },
  postprocess: { pane: "pane-postprocess", init: initPostprocess },
};
// Tabs that manage their own full-height layout: the shell must not add page padding or scrolling
// around them, or their content stops short of the performance panel instead of meeting it.
// Settings is one so its toolbar stays put while the rail and the form scroll independently.
const FILL_TABS = new Set(["status", "console", "settings"]);
const started = new Set();
/* Whatever each tab's init() returned. The settings controller is the one that
   matters: it exposes focusPath/showSection, which is how a click on the stage
   bar reaches the right settings pane. */
const controllers = {};
/* The top-bar controller. It used to be DISCARDED at the call site, which made
   `topbar.refresh()` -- documented in its own source as "the escape hatch for a run
   started OUTSIDE the app (from the machine3 CLI)" -- unreachable for the life of the
   app. Kept now, so the shell can re-read the runtime record after anything that could
   have changed it. */
let topbar = null;

function show(name) {
  if (!TABS[name]) name = "status";
  qsa(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  qsa(".tabpane").forEach((p) => p.classList.toggle("active", p.id === TABS[name].pane));
  const body = document.getElementById("tabbody");
  body.classList.toggle("nopad", FILL_TABS.has(name));
  body.classList.toggle("noscroll", FILL_TABS.has(name));
  if (!started.has(name)) {                       // lazy first-paint
    started.add(name);
    try {
      controllers[name] = TABS[name].init(document.getElementById(TABS[name].pane));
    } catch (err) {
      console.error(`LM3: ${name} tab failed to initialize`, err);
      // Built as nodes, not markup: the message can carry a server string (a
      // path, a rejected value) and innerHTML would execute anything in it.
      const pane = document.getElementById(TABS[name].pane);
      pane.replaceChildren(el("div.card",
        el("h4", "This tab failed to load"),
        el("p.mono.dim", String(err))));
    }
  }
  if (location.hash.slice(1) !== name) history.replaceState(null, "", `#${name}`);
  try { localStorage.setItem("lm3.tab", name); } catch (_) {}
}

function wireTabs() {
  qsa(".tab").forEach((btn) => btn.addEventListener("click", () => show(btn.dataset.tab)));
  window.addEventListener("hashchange", () => show(location.hash.slice(1)));
}

/**
 * Cross-tab navigation: `lm3:navigate {tab, module?, path?}`.
 *
 * topbar.js has dispatched this event since the stage bar was written, and until
 * now NOTHING listened for it -- so "click the run chip to edit it in Settings"
 * silently did nothing. The settings tab may not have been initialized yet when
 * the event lands; show() constructs it here and its controller queues the
 * request until its first read of the YAML finishes.
 */
function wireNavigation() {
  document.addEventListener("lm3:navigate", (ev) => {
    const d = (ev && ev.detail) || {};
    const tab = d.tab || "settings";
    if (!TABS[tab]) return;
    show(tab);
    const c = controllers[tab];
    if (!c) return;
    if (tab === "settings") {
      if (d.path && typeof c.focusPath === "function") c.focusPath(d.path);
      else if (d.module && typeof c.showSection === "function") c.showSection(d.module);
    } else if (tab === "status" && d.module) {
      focusModule(d.module);
    }
  });
}

/** Reflect server reachability in the tab bar, so a dead server is obvious rather than silent. */
async function watchConnection() {
  const el = document.getElementById("connstate");
  const tick = async () => {
    try {
      await api.health();
      el.textContent = "connected";
      el.className = "conn ok";
    } catch (_) {
      el.textContent = "server unreachable";
      el.className = "conn bad";
    }
  };
  await tick();
  setInterval(tick, 5000);
}

/** The Results tab only makes sense once a run has produced output. */
async function gateResultsTab() {
  const btn = document.getElementById("tab-results");
  try {
    const runs = await api.listRuns();
    const has = (runs && runs.n) > 0;
    btn.disabled = !has;
    btn.title = has ? "" : "No LM3 runs found yet -- finish a run to browse its output";
  } catch (_) { /* leave enabled; the tab shows its own empty state */ }
}

/**
 * Re-read the runtime record when the window comes back to the front.
 *
 * A CLI run can start, finish, or be started by a second window while this one is hidden, and the
 * idle poll is deliberately slow. This is the cheap way to make "open the GUI during a CLI run and
 * it shows that run" true for "come back to the GUI" as well.
 */
function wakeOnFocus() {
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && topbar) void topbar.refresh();
  });
}

function boot() {
  wireTabs();
  wireNavigation();
  // topbar.js resolves its own two mounts (.topbar for the stage bar, .primary for the settings
  // strip) from the app container, so hand it the whole shell rather than individual nodes.
  // Start LM3 must run what the Settings tab is SHOWING, so the top bar needs a way to commit
  // that tab's unsaved edits. Looked up at call time, not bound here: tabs are initialized lazily,
  // so `controllers.settings` usually does not exist yet at boot.
  topbar = initTopBar(document.querySelector(".app"), {
    focusModule,
    flushSettings: () => (controllers.settings && controllers.settings.flush
      ? controllers.settings.flush()
      : true),
  });
  initPerfMon(document.getElementById("perfpanel"));
  const initial = location.hash.slice(1) || (() => {
    try { return localStorage.getItem("lm3.tab"); } catch (_) { return null; }
  })() || "status";
  show(initial);
  refreshModelsStatus();          // colors the Models tab (and feeds its panels) from boot, whichever tab opens first
  watchConnection();
  wakeOnFocus();
  gateResultsTab();
  setInterval(gateResultsTab, 20000);
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
else boot();
