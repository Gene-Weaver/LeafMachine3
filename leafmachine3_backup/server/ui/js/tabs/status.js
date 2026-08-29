/* ==========================================================================
   LM3 — Status tab (the primary tab)
   --------------------------------------------------------------------------
   TOP HALF     what LeafMachine3 is doing right now: the active module, the
                worker fleet, the 16-module strip, and the live counters.
   BOTTOM HALF  the console, tailing <run>/logs/lm3.log.

   Both halves are fed by SSE (GET /v1/status/stream, GET /v1/logs/stream).
   A poll of GET /v1/status backstops the stream if EventSource cannot hold a
   connection, so the pane is never silently stale.

   HONESTY RULES THIS FILE OBEYS — every one of them is a real property of the
   backend, not a style preference:
     * `workers[].item_label` is permanently null. Nothing in LM3 records which
       worker holds which specimen, so no tile ever claims to know. What the
       per-worker bars show is the ACTIVE MODULE's overall progress (which is
       what `workers_meta.pct_is` says `pct` is), with each worker's own live
       CPU / RSS / VRAM as the texture that proves the tile is alive.
     * `observed:false` rows are planned slots, not observed processes — 11 of
       the 16 modules run threads inside the server process, so there are no
       children to see. Those tiles say "planned" and never show fake CPU.
     * "skipped" is a fifth module state (a disabled module is written to the
       ledger as done). It is rendered as skipped, never as done.
     * `session:"prior"` means the module was already up to date and did not run
       this time. Rendered as done-but-muted so the strip does not imply work
       that this run did not do.
   ========================================================================== */

import {
  api, el, clear, esc,
  fmtNum, fmtDuration, fmtTime, fmtMB, fmtPath,
} from "../api.js";


/* ================================================================ constants */

/** Console memory bound. Older lines are dropped from the front. */
const LOG_CAP = 5000;

/** How many lines the log stream replays when we attach. */
const LOG_BACKFILL = 600;

/** Consecutive stream drops before we fall back to polling /v1/status. */
const POLL_AFTER_DROPS = 3;
const POLL_INTERVAL_MS = 3000;

const LS = {
  split: "lm3.status.split",
  level: "lm3.status.level",
  follow: "lm3.status.follow",
};

/** Python logging levels, ordered. The console filter is a MINIMUM level. */
const LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"];
const LEVEL_RANK = { DEBUG: 0, INFO: 1, SUCCESS: 1, WARNING: 2, ERROR: 3, CRITICAL: 4 };

/**
 * The 16 LM3 modules in canonical order, with a one-line description of what
 * each actually does (written from the module docstrings, not invented).
 *
 * The live names/order always come from the snapshot; this table supplies the
 * descriptions the snapshot has no field for, AND the strip skeleton shown when
 * no run exists yet (an idle snapshot carries `modules: []`).
 */
const MODULE_INFO = [
  ["mp_conversion_factor", "MP Conversion Factor",
   "Predicts a pixels-per-cm scale for every sheet from its image resolution alone, before any model runs."],
  ["archival_detector", "Archival Detector",
   "Finds archival sheet elements — rulers, barcodes, color cards, labels — and saves a crop of each."],
  ["plant_detector", "Plant Detector",
   "Finds plant organs — whole and partial leaves, flowers, buds — and saves the leaf crops later modules consume."],
  ["specimen_segmenter", "Specimen Segmenter",
   "Separates plant from paper across the whole sheet, producing one binary specimen mask per image."],
  ["phenology_detector", "Phenology Detector",
   "Rolls the plant detections up to per-specimen presence of leaves, flowers and fruits. No model, pure aggregation."],
  ["ruler_classifier", "Ruler Classifier",
   "Classifies every ruler crop to its measurement unit type with a three-model ensemble vote."],
  ["ruler_cf", "Ruler Conversion Factor",
   "Measures the tick lattice of each ruler to fix one pixels-per-cm factor per sheet, publishing only confident readings."],
  ["leaf_segmenter", "Leaf Segmenter",
   "Segments each leaf crop into lamina, petiole and hole instances, re-based onto the parent sheet."],
  ["morphology", "Morphology",
   "Computes shape metrics per leaf mask — area, perimeter, convexity, circularity — plus the rotated bounding box."],
  ["landmark_detector", "Landmark Detector",
   "Predicts the 31-keypoint leaf skeleton — tip, base, midvein, petiole and width points — on every leaf crop."],
  ["landmark_measurements", "Landmark Measurements",
   "Turns those keypoints into lamina trace length, leaf width, apex and base angles, petiole length and curvature."],
  ["leaf_orientation", "Leaf Orientation",
   "Works out the rotation that stands each leaf tip-up, from its keypoints, for the oriented leaf products."],
  ["petiole_width", "Petiole Width",
   "Measures petiole thickness perpendicular to the landmark centerline and reports the median."],
  ["metric_grounding", "Metric Grounding",
   "Applies each sheet's conversion factor to the stored pixel measurements, turning them into cm and cm²."],
  ["reporter", "Reporter",
   "Renders every requested output — overlays, crops, masks and leaf products — from the stored records and the originals."],
  ["ect", "ECT",
   "Computes the Euler Characteristic Transform of each oriented leaf and writes its radial and Cartesian views."],
];

const INFO_BY_KEY = new Map(MODULE_INFO.map(([k, name, blurb]) => [k, { key: k, name, blurb }]));

/** Module state -> the `.pill` modifier that matches the design system. */
const STATE_PILL = {
  done: "m", running: "s", error: "e", skipped: "x", pending: "x",
};

const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)");


/* ============================================================ scoped styles */
/* app.css belongs to the design-system agent, so the handful of classes that
   are genuinely this tab's own (the module strip, the completion flourish, the
   figure rows) ship here instead. Everything is namespaced `.st-` so it cannot
   collide, and every value is a design token — no forked hexes. */

const STYLE_ID = "lm3-status-style";
const STYLE = `
.st-live{display:flex;flex-direction:column;gap:16px}
.st-sec{display:flex;flex-direction:column;gap:9px}

/* --- active-module card interior ------------------------------------------ */
.st-nowhd{display:flex;align-items:flex-start;gap:14px}
.st-nowtitle{min-width:0;flex:1 1 auto}
.st-nowtags{display:flex;align-items:center;gap:6px;flex-wrap:wrap;justify-content:flex-end;flex:0 0 auto}
.st-fig{display:flex;align-items:baseline;gap:6px;min-width:0}
.st-fig > .v{font:650 14px/1.2 var(--mono);font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}
.st-fig.big > .v{font-size:18px;letter-spacing:-.01em}
.st-fig > .k{font:650 11.5px/1.2 var(--sans);letter-spacing:.05em;text-transform:uppercase;color:var(--dim);white-space:nowrap}
.st-fig > .v > .sub{color:var(--dim);font-size:.74em;font-weight:600}
.st-nowbar{margin:12px 0 12px}

/* --- the 16-module strip --------------------------------------------------- */
.st-strip{display:flex;flex-wrap:wrap;gap:7px}
.st-mod{
  position:relative;overflow:hidden;appearance:none;cursor:pointer;text-align:left;
  display:flex;align-items:center;gap:7px;padding:6px 10px 8px;
  background:var(--panel);border:1px solid var(--line);border-radius:var(--r-sm);
  color:var(--mute);font:12px/1.25 var(--sans);
  transition:background .15s ease,border-color .15s ease,color .15s ease,box-shadow .15s ease;
}
.st-mod:hover{background:var(--hover);color:var(--ink);border-color:#3a3d45}
.st-mod:focus-visible{outline:2px solid var(--acc2);outline-offset:2px}
.st-mod > .ord{
  flex:0 0 auto;font:650 10px/1 var(--mono);color:var(--dim);
  background:var(--panel2);border-radius:var(--r-xs);padding:3px 5px;
}
.st-mod > .nm{font-weight:650;white-space:nowrap}
.st-mod > .el{font:10.5px/1 var(--mono);color:var(--dim);font-variant-numeric:tabular-nums}
.st-mod > .trk{position:absolute;left:0;right:0;bottom:0;height:2px;background:rgba(255,255,255,.05)}
.st-mod > .trk > i{display:block;height:100%;width:0;background:var(--dim);
  transition:width .35s cubic-bezier(.4,0,.2,1)}

.st-mod.done{color:var(--ink);border-color:rgba(74,222,128,.4)}
.st-mod.done > .ord{color:var(--acc3)}
.st-mod.done > .trk > i{background:var(--acc3)}
.st-mod.prior{opacity:.66}
.st-mod.running{color:var(--ink);border-color:var(--acc2);background:rgba(56,189,248,.10);
  animation:st-modpulse 1.9s ease-in-out infinite}
.st-mod.running > .ord{color:var(--acc2);background:rgba(56,189,248,.18)}
.st-mod.running > .trk > i{background:linear-gradient(90deg,#0ea5e9,var(--acc2))}
.st-mod.error{color:var(--bad);border-color:var(--bad);background:rgba(248,113,113,.08)}
.st-mod.error > .ord{color:var(--bad)}
.st-mod.error > .trk > i{background:var(--bad)}
.st-mod.skipped{opacity:.5}
.st-mod.skipped > .nm{text-decoration:line-through;text-decoration-color:var(--dim)}
.st-mod.sel{border-color:var(--acc2);box-shadow:0 0 0 2px rgba(56,189,248,.35)}
@keyframes st-modpulse{0%,100%{box-shadow:0 0 0 0 rgba(56,189,248,0)}
                       50%   {box-shadow:0 0 0 3px rgba(56,189,248,.15)}}
/* the flourish when a module finishes: one sweep, then it is gone */
@keyframes st-sweep{from{transform:translateX(-120%)}to{transform:translateX(220%)}}
.st-flash::before{
  content:"";position:absolute;inset:0 auto 0 0;width:62%;pointer-events:none;
  background:linear-gradient(100deg,transparent,rgba(74,222,128,.4),transparent);
  animation:st-sweep .9s cubic-bezier(.4,0,.2,1);
}

/* --- module detail --------------------------------------------------------- */
.st-detail{display:none}
.st-detail.open{display:block;animation:lm3-fadein .18s ease}
.st-dethd{display:flex;align-items:center;gap:9px;margin:0 0 9px}
.st-dethd > .nm{font-size:.98rem;font-weight:650;letter-spacing:-.01em;color:var(--ink)}
.st-dethd > .btn{margin-left:auto}

/* --- worker fleet ---------------------------------------------------------- */
/* a thread module can plan dozens of slots; past two dozen the tiles shrink so
   the fleet still reads as a fleet instead of a page of scrolling */
.workers.st-compact{grid-template-columns:repeat(auto-fill,minmax(158px,1fr))}
.workers.st-compact .worker{padding:8px 9px}
.workers.st-compact .worker-ft{font-size:10px}
.st-wbar{margin:2px 0 0}
.st-wnote{font-size:12px;color:var(--dim);line-height:1.5;margin:0}

/* --- console --------------------------------------------------------------- */
.st-conspane{position:relative}
.st-jump{position:absolute;right:18px;bottom:16px;z-index:5;box-shadow:0 8px 22px rgba(0,0,0,.6)}
.st-lc{font:11px/1.4 var(--mono);font-variant-numeric:tabular-nums;color:var(--dim)}
.st-hd-run{font:11.5px/1.4 var(--mono);color:var(--dim);text-transform:none;letter-spacing:0;
  max-width:32ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* --- idle ------------------------------------------------------------------ */
.st-idle{display:flex;flex-direction:column;gap:14px}
.st-idle .st-mod{cursor:default;opacity:.6}
.st-idle .st-mod:hover{background:var(--panel);color:var(--mute);border-color:var(--line)}

/* the denominator riding under a counter's value */
.st-live .stat .v > .sub{color:var(--dim);font-size:.68em;font-weight:600}
`;

function injectStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const tag = document.createElement("style");
  tag.id = STYLE_ID;
  tag.textContent = STYLE;
  document.head.appendChild(tag);
}


/* ================================================================== helpers */

const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const clampPct = (v) => Math.max(0, Math.min(100, num(v) ?? 0));

/** Durations at the ledger's 1-second granularity: 0.0 means "under a second". */
function shortDur(s) {
  const v = num(s);
  if (v === null) return "–";
  if (v < 1) return "<1s";
  return fmtDuration(v);
}

/** A rate in items/second, phrased the way the module is actually counted. */
function rateLabel(r) {
  const v = num(r);
  if (v === null || v <= 0) return "–";
  if (v >= 1) return `${fmtNum(v, { digits: 2 })}/s`;
  return `${fmtNum(60 * v, { digits: 1 })}/min`;
}

/** How a module is running, in the words the user reads on the card. */
function execLabel(m) {
  const n = num(m.workers) ?? num(m.planned_workers) ?? 1;
  const plural = n === 1 ? "" : "s";
  switch (m.exec_mode) {
    case "gpu": return `GPU · ${n} worker${plural}`;
    case "process": return `CPU · ${n} process${n === 1 ? "" : "es"}`;
    case "thread": return `CPU · ${n} thread${plural}`;
    case "serial": return "Serial";
    default: return m.device === "gpu" ? "GPU" : "CPU";
  }
}

/** One worker slot's label: "GPU 0 · w2", "CPU proc 7", "CPU thread 3". */
function workerLabel(w, i) {
  const n = String(w.id ?? `w${i}`).replace(/^w/, "");
  switch (w.exec_mode) {
    case "gpu":
      return num(w.gpu_index) !== null ? `GPU ${w.gpu_index} · w${n}` : `GPU · w${n}`;
    case "process": return `CPU proc ${n}`;
    case "thread": return `CPU thread ${n}`;
    default: return `Worker ${n}`;
  }
}

/** Modules the run reports, backfilled with descriptions from MODULE_INFO. */
function moduleName(key) {
  const info = INFO_BY_KEY.get(key);
  return info ? info.name : String(key ?? "–");
}
function moduleBlurb(key) {
  const info = INFO_BY_KEY.get(key);
  return info ? info.blurb : "";
}

/**
 * The class a module chip wears. `session:"prior"` is layered on top of `done`
 * so an already-up-to-date module reads as complete but visibly not run now.
 */
function modClass(m) {
  const base = ["st-mod", m.state || "pending"];
  if (m.state === "done" && m.session === "prior") base.push("prior");
  return base.join(" ");
}

/**
 * Animate a number to a new value. Cheap, cancelable, and a no-op when the OS
 * asks for reduced motion or when this is the element's first paint (there is
 * nothing to count up FROM on the first frame).
 */
function setNum(node, value, fmt = (v) => fmtNum(v)) {
  if (!node) return;
  const to = num(value);
  if (node._raf) { cancelAnimationFrame(node._raf); node._raf = 0; }
  if (to === null) { node.textContent = "–"; node._n = null; return; }

  const from = num(node._n);
  node._n = to;
  if (from === null || from === to || REDUCED.matches) { node.textContent = fmt(to); return; }

  const t0 = performance.now();
  const dur = 420;
  const step = (now) => {
    const p = Math.min(1, (now - t0) / dur);
    const eased = 1 - Math.pow(1 - p, 3);              // easeOutCubic
    node.textContent = fmt(from + (to - from) * eased);
    node._raf = p < 1 ? requestAnimationFrame(step) : 0;
  };
  node._raf = requestAnimationFrame(step);
}

/** Set a `.bar`'s fill width plus its running/idle modifiers in one call. */
function setBar(bar, pct, { running = false, tone = null, label = null } = {}) {
  if (!bar) return;
  const fill = bar.querySelector(".fill");
  if (fill) fill.style.width = `${clampPct(pct)}%`;
  bar.classList.toggle("running", !!running);
  for (const t of ["acc", "acc2", "ok", "warn", "bad", "vio", "done", "idle"]) {
    bar.classList.toggle(t, tone === t);
  }
  const lbl = bar.querySelector(".lbl");
  if (lbl && label !== null) lbl.textContent = label;
}

/** Escape a log message and wrap query hits in <mark> (the design system styles it). */
function highlight(msg, query) {
  const text = String(msg ?? "");
  if (!query) return esc(text);
  const q = query.toLowerCase();
  const hay = text.toLowerCase();
  let out = "";
  let i = 0;
  for (;;) {
    const at = hay.indexOf(q, i);
    if (at < 0) { out += esc(text.slice(i)); break; }
    out += esc(text.slice(i, at)) + "<mark>" + esc(text.slice(at, at + q.length)) + "</mark>";
    i = at + q.length;
  }
  return out;
}

const store = {
  get(k, dflt) { try { const v = localStorage.getItem(k); return v === null ? dflt : v; } catch { return dflt; } },
  set(k, v) { try { localStorage.setItem(k, String(v)); } catch { /* private mode */ } },
};


/* =================================================================== state */
/* One tab, one instance. The module-level handle is what lets the exported
   focusModule() work when another part of the app (the top bar's stage
   segments) calls it without holding a reference to the controller. */

let S = null;
// Deliberately OUTSIDE S: "the GUI was reset, do not adopt the run the server is reporting" has to
// survive S being null. The app restores whichever tab you were last on, so a launch reset usually
// fires while Live Status has never been mounted -- storing this in S dropped it on the floor, and
// the tab then rendered the previous run's "Run complete" cards as if nothing had been reset.
let newRunPending = false;
let pendingFocus = null;


/* ============================================================ initStatus() */

/**
 * Build the Status tab into `root` and start streaming.
 * @param {HTMLElement} root  the `.tabpane` for this tab
 * @returns {{destroy:Function, focusModule:Function, refresh:Function}}
 */
export function initStatus(root) {
  if (!root) throw new Error("initStatus(root): root element is required");
  if (S && S.root === root) return S.ctl;       // idempotent re-init
  if (S) S.ctl.destroy();

  injectStyle();
  clear(root);
  root.classList.add("fill");

  const ui = {};
  S = {
    root, ui,
    snap: null,
    skew: 0,                    // server clock - browser clock, seconds
    selected: null,
    prevState: new Map(),       // module key -> last seen state, for the flourish
    lines: [],                  // console model (capped at LOG_CAP)
    uid: 0,                     // our own monotonic line id (backend seq resets per run)
    filter: { min: store.get(LS.level, "INFO"), q: "" },
    follow: store.get(LS.follow, "1") === "1",
    drops: 0,
    pollTimer: 0,
    tickTimer: 0,
    closeStatus: null,
    closeLogs: null,
    ctl: null,
  };
  if (!LEVELS.includes(S.filter.min)) S.filter.min = "INFO";

  /* ------------------------------------------------------------- structure
     The console is its OWN tab, so this tab is the live view at full height. The console DOM is
     still built here (it owns the log stream, filters and follow state, which must keep collecting
     whether or not its tab has been opened) and is simply parented into the Console tab by
     initConsole() when that tab first paints. */
  const pane = el("div.split-pane");
  const scroll = el("div.pane-scroll");
  pane.appendChild(scroll);
  root.appendChild(pane);

  ui.scroll = scroll;

  ui.consoleHost = el("div.split-pane.st-conspane");   // detached until the Console tab mounts it
  buildTop(scroll, ui);
  buildConsole(ui.consoleHost, ui);

  /* ----------------------------------------------------------- controller */
  S.ctl = {
    destroy: () => destroy(),
    focusModule: (k) => focusModule(k),
    refresh: () => pollOnce(),
  };

  connect();
  pollOnce();                                   // paint immediately, do not wait for SSE
  startTicker();

  if (pendingFocus) { const k = pendingFocus; pendingFocus = null; focusModule(k); }
  return S.ctl;
}


/**
 * Select a module and reveal its detail. Exported so the top bar's stage
 * segments can deep-link into this tab. Safe to call before initStatus():
 * the request is held and applied as soon as the tab builds.
 */
export function focusModule(key) {
  if (!S) { pendingFocus = key; return; }
  if (!key) return;
  S.selected = String(key);
  renderStrip(S.snap);
  renderDetail(S.snap);
  const chip = S.ui.strip && S.ui.strip.querySelector(`.st-mod[data-key="${CSS.escape(S.selected)}"]`);
  if (chip) chip.scrollIntoView({ block: "nearest", inline: "nearest" });
  if (S.ui.detail && S.ui.detail.classList.contains("open")) {
    S.ui.detail.scrollIntoView({ block: "nearest" });
  }
}


/* ============================================================ TOP: markup */

function buildTop(scroll, ui) {
  /* --- idle panel (shown when no run has produced a ledger yet) ----------- */
  ui.idle = el("div.st-idle", { hidden: true });
  scroll.appendChild(ui.idle);

  /* --- live panel --------------------------------------------------------- */
  ui.live = el("div.st-live");
  scroll.appendChild(ui.live);

  /* active module card */
  ui.nowName = el("h3.modname", "—");
  ui.nowSub = el("p.modsub", "");
  ui.nowTags = el("div.st-nowtags");
  ui.nowBar = el("div.bar.xl.st-nowbar", el("span.fill"), el("span.lbl", "0%"));
  ui.nowRow = el("div.modrow");

  ui.now = el("div.nowcard",
    el("div.st-nowhd",
      el("div.st-nowtitle", ui.nowName, ui.nowSub),
      ui.nowTags),
    ui.nowBar,
    ui.nowRow);
  ui.live.appendChild(ui.now);

  /* live counters */
  ui.counters = el("div.grid.auto220");
  ui.counterCells = {};
  for (const c of counterSpecs()) {
    const v = el("div.v", el("span.n", "–"), el("span.sub", ""));
    const cell = el(`div.stat.${c.tone}`, { title: c.title }, v, el("div.l", c.label));
    ui.counterCells[c.id] = { cell, n: v.querySelector(".n"), sub: v.querySelector(".sub") };
    ui.counters.appendChild(cell);
  }
  ui.live.appendChild(ui.counters);

  /* worker fleet */
  ui.workersBadge = el("span.badge", "0 slots");
  ui.workers = el("div.workers");
  ui.workerNote = el("p.st-wnote", "");
  ui.live.appendChild(el("div.st-sec",
    el("div.sechd", "Worker fleet", ui.workersBadge),
    ui.workerNote,
    ui.workers));

  /* the 16 modules */
  ui.strip = el("div.st-strip");
  ui.stripBadge = el("span.badge", "");
  ui.detail = el("div.card.info.st-detail");
  ui.live.appendChild(el("div.st-sec",
    el("div.sechd", "LM3 modules", ui.stripBadge),
    ui.strip,
    ui.detail));
}

/**
 * The live counters. Each one is derived from a real ledger count — the label
 * says exactly what was counted, because "leaves segmented" and "specimens the
 * Leaf Segmenter finished" are different numbers and only the second exists.
 */
function counterSpecs() {
  return [
    { id: "images", tone: "plain", label: "Specimens finished end to end",
      title: "Specimens that have cleared the Reporter, the last module every specimen passes through." },
    { id: "modules", tone: "plain", label: "Modules complete",
      title: "Modules finished this run. Skipped (disabled) modules are not counted." },
    { id: "segmented", tone: "plain", label: "Specimens segmented",
      title: "Specimens the Leaf Segmenter has finished. LM3 records progress per specimen, not per leaf." },
    { id: "rulers", tone: "plain", label: "Rulers measured",
      title: "Specimens the Ruler Conversion Factor module measured a ruler on — sheets with no ruler are excluded." },
    { id: "ect", tone: "plain", label: "ECT computed",
      title: "Specimens whose leaves have been through the ECT module." },
    { id: "errors", tone: "plain", label: "Item errors",
      title: "Per-specimen errors across all modules. LM3 keeps going and records them." },
  ];
}


/* ============================================================ TOP: render */

function renderSnapshot(snap) {
  if (!snap || !S) return;
  // After "New run" the server still reports the PREVIOUS run (it discovers the
  // newest one off disk when no job is active). Stay blank until something is
  // genuinely running, or the reset would undo itself on the next frame.
  //
  // `stale` has to be part of "genuinely running": a killed run's ledger still says `running`,
  // and without this that frame walks straight past the guard. See the matching test in topbar.js.
  if (newRunPending) {
    if (snap.state !== "running" || snap.stale) {
      // Show the idle pane and HIDE the finished-run pane. Returning without this
      // leaves the previous run's "Run complete" cards on screen: the live/idle
      // swap below is the only thing that hides them, and we never reach it.
      S.ui.live.hidden = true;
      S.ui.idle.hidden = false;
      renderIdle();
      return;
    }
    newRunPending = false;
  }
  S.snap = snap;
  const t = num(snap.t);
  if (t !== null) S.skew = Date.now() / 1000 - t;    // neutralize server/browser clock drift

  const live = !!snap.ready;
  S.ui.live.hidden = !live;
  S.ui.idle.hidden = live;

  if (!live) { renderIdle(); return; }

  renderNowCard(snap);
  renderCounters(snap);
  renderWorkers(snap);
  renderStrip(snap);
  renderDetail(snap);
  setConsoleRun(snap.run_name);
}


/* ---------------------------------------------------------- active module */

function renderNowCard(snap) {
  const ui = S.ui;
  const a = snap.active;
  clear(ui.nowTags);
  clear(ui.nowRow);

  if (a) {
    ui.nowName.textContent = a.name || moduleName(a.key);
    ui.nowSub.textContent = moduleBlurb(a.key);

    ui.nowTags.appendChild(el(`span.badge.${a.device === "gpu" ? "info" : "ok"}`, execLabel(a)));
    if (a.exec_mode === "serial" && a.exec_note) {
      ui.nowTags.appendChild(el("span.serialflag", { title: a.exec_note }, "serial"));
    }
    if (snap.stale) {
      ui.nowTags.appendChild(el("span.badge.warn", {
        title: "The ledger and the log have both been silent. The run may have been killed; "
             + "LM3 only repairs an interrupted module at the next start.",
      }, `silent ${shortDur(snap.stale_for_s)}`));
    }

    setBar(ui.nowBar, a.pct, { running: !snap.stale, tone: null, label: `${fmtNum(a.pct, { digits: 1 })}%` });

    const total = num(a.n_total);
    figures(ui.nowRow, [
      { big: true, k: "done", v: `${fmtNum(a.n_done)}<span class="sub"> / ${total === null ? "?" : fmtNum(total)}</span>` },
      { k: "rate", v: rateLabel(a.rate_per_s) },
      { k: "elapsed", v: liveElapsed(a.started_ts, a.elapsed_s), live: "elapsed" },
      { k: "eta", v: num(a.eta_s) === null ? "–" : fmtDuration(a.eta_s) },
      num(a.n_error) ? { k: "errors", v: fmtNum(a.n_error), bad: true } : null,
      snap.next ? { k: "next", v: esc(snap.next.name || moduleName(snap.next.key)) } : null,
    ]);
    return;
  }

  /* Nothing is running. That is three different situations and the card says
     which: the run is finished, it failed, or it is between modules (a real
     window — the executor tears one pool down before it builds the next). */
  const errored = (snap.modules || []).find((m) => m.state === "error");
  if (snap.state === "error" || errored) {
    ui.nowName.textContent = "Run stopped on an error";
    ui.nowSub.textContent = errored
      ? `${errored.name || moduleName(errored.key)}: ${errored.error_msg || "see the console below"}`
      : "See the console below.";
    ui.nowTags.appendChild(el("span.badge.bad", "error"));
    setBar(ui.nowBar, snap.totals.overall_pct, { tone: "bad", label: `${fmtNum(snap.totals.overall_pct, { digits: 1 })}%` });
  } else if (snap.state === "done") {
    ui.nowName.textContent = "Run complete";
    ui.nowSub.textContent = `${moduleCountPhrase(snap)} · ${fmtNum(snap.images_total)} specimens`;
    ui.nowTags.appendChild(el("span.badge.ok", "done"));
    setBar(ui.nowBar, 100, { tone: "done", label: "100%" });
  } else if (snap.stale) {
    /* The ledger and the log have both gone quiet with no module active: the run was stopped or
       killed, it is NOT "between modules". Saying "working" here is the stale status to avoid --
       show it as stopped, and say plainly that it is resumable, because it is. */
    const done = snap.totals.modules_complete ?? snap.totals.modules_done;
    ui.nowName.textContent = "Run stopped";
    ui.nowSub.textContent = `LM3 is no longer running (silent ${shortDur(snap.stale_for_s)}). `
      + `${fmtNum(done)} of ${fmtNum(snap.totals.modules_total)} modules finished — press Start LM3 to resume where it left off.`;
    ui.nowTags.appendChild(el("span.badge.warn", "stopped"));
    setBar(ui.nowBar, snap.totals.overall_pct, {
      running: false, tone: "warn", label: `${fmtNum(snap.totals.overall_pct, { digits: 1 })}%`,
    });
  } else if (snap.state === "running") {
    /* Genuinely mid-run with no module active: the executor tears one worker pool down before it
       builds the next, which is a real window of a few seconds. */
    ui.nowName.textContent = snap.next
      ? `Starting ${snap.next.name || moduleName(snap.next.key)}`
      : "Between modules";
    ui.nowSub.textContent = snap.next
      ? moduleBlurb(snap.next.key)
      : "LM3 is tearing down one worker pool and building the next.";
    ui.nowTags.appendChild(el("span.badge.info", "working"));
    setBar(ui.nowBar, snap.totals.overall_pct, {
      running: true, label: `${fmtNum(snap.totals.overall_pct, { digits: 1 })}%`,
    });
  } else {
    /* NOT running. A `next` module here is where LM3 would RESUME, not something starting now --
       calling that "working" with a moving bar is the stale status this card must never show. */
    const partial = (snap.totals.modules_complete ?? snap.totals.modules_done) > 0;
    ui.nowName.textContent = partial ? "Run paused" : "No LM3 run in flight";
    ui.nowSub.textContent = snap.next
      ? `Not running. Press Start LM3 to resume at ${snap.next.name || moduleName(snap.next.key)}.`
      : "Set the folders above and press Start LM3.";
    ui.nowTags.appendChild(el("span.badge.warn", partial ? "paused" : "idle"));
    setBar(ui.nowBar, snap.totals.overall_pct, {
      running: false, tone: partial ? "warn" : null,
      label: `${fmtNum(snap.totals.overall_pct, { digits: 1 })}%`,
    });
  }

  figures(ui.nowRow, [
    { big: true, k: "overall", v: `${fmtNum(snap.totals.overall_pct, { digits: 1 })}%` },
    { k: "specimens", v: `${fmtNum(snap.images_done)}<span class="sub"> / ${fmtNum(snap.images_total)}</span>` },
    { k: "elapsed", v: liveElapsed(snap.started_ts, snap.elapsed_s), live: "elapsed" },
    snap.finished_at ? { k: "finished", v: fmtTime(Date.parse(snap.finished_at) / 1000) } : null,
  ]);
}

function moduleCountPhrase(snap) {
  const t = snap.totals;
  const bits = [`${fmtNum(t.modules_complete)} of ${fmtNum(t.modules_total)} modules`];
  if (t.modules_skipped) bits.push(`${fmtNum(t.modules_skipped)} skipped`);
  return bits.join(" · ");
}

/** Render a `.modrow` of figure/label pairs. `live:"elapsed"` marks the ticker. */
function figures(row, specs) {
  clear(row);
  for (const f of specs) {
    if (!f) continue;
    const v = el("div.v", { html: f.v });
    if (f.bad) v.style.color = "var(--bad)";
    const node = el(`div.st-fig${f.big ? ".big" : ""}`, v, el("div.k", f.k));
    if (f.live) node.dataset.live = f.live;
    row.appendChild(node);
  }
}

/**
 * Elapsed time, ticked locally between snapshots. `started_ts` is a real
 * timestamp from the ledger, so counting up from it is measurement, not
 * invention — it just keeps the clock from looking frozen when a frame is late.
 */
function liveElapsed(startedTs, fallback) {
  const st = num(startedTs);
  if (st === null) return shortDur(fallback);
  const now = Date.now() / 1000 - S.skew;
  return fmtDuration(Math.max(0, now - st), { clock: true });
}

/** 1 Hz local clock for the elapsed figures. Only runs while something runs. */
function startTicker() {
  stopTicker();
  S.tickTimer = setInterval(() => {
    const snap = S.snap;
    if (!snap || !snap.ready || snap.state !== "running" || snap.stale) return;
    const node = S.ui.nowRow.querySelector('[data-live="elapsed"] > .v');
    if (!node) return;
    const st = snap.active ? snap.active.started_ts : snap.started_ts;
    node.textContent = liveElapsed(st, null);
  }, 1000);
}
function stopTicker() { if (S.tickTimer) { clearInterval(S.tickTimer); S.tickTimer = 0; } }


/* ---------------------------------------------------------------- counters */

function renderCounters(snap) {
  const cells = S.ui.counterCells;
  const by = new Map((snap.modules || []).map((m) => [m.key, m]));
  const t = snap.totals || {};

  const put = (id, value, sub, tone) => {
    const c = cells[id];
    if (!c) return;
    setNum(c.n, value, (v) => fmtNum(Math.round(v), { compact: true }));
    c.sub.textContent = sub ? ` ${sub}` : "";
    for (const k of ["good", "warn", "bad", "plain", "vio"]) c.cell.classList.toggle(k, tone === k);
  };

  put("images", snap.images_done, `/ ${fmtNum(snap.images_total, { compact: true })}`,
      snap.images_total && snap.images_done >= snap.images_total ? "good" : "plain");
  put("modules", t.modules_complete, `/ ${fmtNum(t.modules_total)}`,
      t.modules_complete >= t.modules_total && t.modules_total ? "good" : "plain");

  const seg = by.get("leaf_segmenter");
  put("segmented", seg ? seg.n_done : null, seg ? `/ ${fmtNum(seg.n_total, { compact: true })}` : "", "plain");

  /* A no-work row is a specimen with no ruler at all, so it is a completion but
     not a measurement. Subtracting it is what makes this figure honest. */
  const cf = by.get("ruler_cf");
  const measured = cf ? Math.max(0, (num(cf.n_done) ?? 0) - (num(cf.n_no_work) ?? 0)) : null;
  put("rulers", measured, cf && cf.n_no_work ? `· ${fmtNum(cf.n_no_work)} no ruler` : "", "plain");

  const ect = by.get("ect");
  put("ect", ect ? ect.n_done : null, ect ? `/ ${fmtNum(ect.n_total, { compact: true })}` : "", "plain");

  const errors = (snap.modules || []).reduce((a, m) => a + (num(m.n_error) ?? 0), 0);
  put("errors", errors, "", errors > 0 ? "bad" : "plain");
}


/* ----------------------------------------------------------- worker fleet */

function renderWorkers(snap) {
  const ui = S.ui;
  const rows = snap.workers || [];
  const meta = snap.workers_meta || {};

  ui.workersBadge.textContent = rows.length
    ? `${fmtNum(meta.n_busy ?? 0)} busy · ${fmtNum(rows.length)} slot${rows.length === 1 ? "" : "s"}`
    : "no slots";
  ui.workersBadge.className = `badge${meta.n_busy ? " info" : ""}`;
  ui.workerNote.textContent = meta.basis || "";
  ui.workers.classList.toggle("st-compact", rows.length > 24);

  if (!rows.length) {
    if (ui.workerIdsRendered !== "none") {
      clear(ui.workers);
      ui.workers.appendChild(el("div.empty",
        el("div.ic", "◍"),
        el("div.t", "No worker slots"),
        el("div.s", meta.basis || "Nothing is running, so LM3 has allocated no workers.")));
      ui.workerIdsRendered = "none";
    }
    return;
  }

  /* Rebuild only when the slot set changes; otherwise mutate in place so the
     bars animate instead of flashing. */
  const sig = rows.map((w) => `${w.id}:${w.module}:${w.observed ? 1 : 0}`).join("|");
  if (sig !== ui.workerIdsRendered) {
    clear(ui.workers);
    ui.workerNodes = rows.map((w, i) => {
      const node = workerTile(w, i);
      ui.workers.appendChild(node);
      return node;
    });
    ui.workerIdsRendered = sig;
  }
  rows.forEach((w, i) => updateWorkerTile(ui.workerNodes[i], w, snap));
}

function workerTile(w, i) {
  const dev = w.exec_mode === "gpu" ? "gpu" : "cpu";
  const tile = el(`div.worker.${dev}`);
  tile.append(
    el("div.worker-hd",
      el("span.wid", workerLabel(w, i)),
      el("span.wmod", moduleName(w.module)),
      el("span.wdev.badge", "")),
    /* The fill is the ACTIVE MODULE's overall progress — the only per-worker
       percentage that exists (workers_meta.pct_is === "module_overall"). */
    el("div.bar.sm", el("span.fill")),
    /* The worker's OWN live CPU load. Real per-worker texture, observed rows
       only; a planned slot has nothing to measure and shows nothing. */
    el("div.mini.st-wbar", el("span")),
    el("div.worker-ft", el("span.det", ""), el("span.pct", "0%")));
  return tile;
}

function updateWorkerTile(tile, w, snap) {
  if (!tile) return;
  const busy = w.state === "busy";
  tile.classList.toggle("busy", busy);
  tile.classList.toggle("idle", !busy);

  const badge = tile.querySelector(".wdev");
  badge.textContent = w.observed ? (w.device || w.exec_mode || "cpu") : "planned";
  badge.className = `wdev badge${w.observed ? (busy ? " info" : "") : " vio"}`;
  badge.title = w.observed
    ? `Live process, pid ${w.pid ?? "?"}${num(w.age_s) !== null ? `, up ${shortDur(w.age_s)}` : ""}`
    : (snap.workers_meta || {}).basis || "A planned slot from hardware_settings.yaml, not an observed process.";

  tile.querySelector(".wmod").textContent = moduleName(w.module) || "idle";

  const pct = clampPct(w.pct);
  setBar(tile.querySelector(".bar"), pct, { running: busy && !snap.stale, tone: busy ? null : "idle" });
  tile.querySelector(".pct").textContent = `${fmtNum(pct, { digits: 1 })}%`;

  /* CPU meter: 100 means one full core, and a multithreaded worker can exceed
     it, so the bar clamps while the footer prints the true figure. */
  const cpu = num(w.cpu_pct);
  const mini = tile.querySelector(".mini");
  mini.hidden = cpu === null;
  if (cpu !== null) mini.firstChild.style.width = `${Math.max(0, Math.min(100, cpu))}%`;

  const bits = [];
  if (cpu !== null) bits.push(`cpu ${fmtNum(cpu, { digits: 0 })}%`);
  if (num(w.rss_mb) !== null) bits.push(fmtMB(w.rss_mb));
  if (num(w.vram_mb) !== null) bits.push(`${fmtMB(w.vram_mb)} vram`);
  if (!bits.length) bits.push(w.observed ? "no counters" : "planned slot");
  tile.querySelector(".det").textContent = bits.join(" · ");
}


/* ------------------------------------------------------------ module strip */

function renderStrip(snap) {
  const ui = S.ui;
  const mods = (snap && snap.modules && snap.modules.length)
    ? snap.modules
    : MODULE_INFO.map(([key, name], i) => ({ key, name, order: i + 1, state: "pending", pct: 0 }));

  const sig = mods.map((m) => m.key).join("|");
  if (sig !== ui.stripSig) {
    clear(ui.strip);
    ui.stripNodes = new Map();
    for (const m of mods) {
      const chip = el("button.st-mod", {
        type: "button",
        dataset: { key: m.key },
        onclick: () => focusModule(m.key),
      },
        el("span.ord", String(m.order ?? "")),
        el("span.nm", m.name || moduleName(m.key)),
        el("span.el", ""),
        el("span.trk", el("i")));
      ui.strip.appendChild(chip);
      ui.stripNodes.set(m.key, chip);
    }
    ui.stripSig = sig;
  }

  for (const m of mods) {
    const chip = ui.stripNodes.get(m.key);
    if (!chip) continue;
    const cls = modClass(m) + (S.selected === m.key ? " sel" : "");
    if (chip.className.replace(/\s*st-flash\s*/, " ").trim() !== cls) {
      const flashing = chip.classList.contains("st-flash");
      chip.className = cls + (flashing ? " st-flash" : "");
    }
    chip.querySelector(".trk > i").style.width = `${clampPct(m.pct)}%`;
    chip.querySelector(".el").textContent =
      m.state === "done" || m.state === "error" ? shortDur(m.elapsed_s)
      : m.state === "running" ? `${fmtNum(m.pct, { digits: 0 })}%`
      : m.state === "skipped" ? "skipped" : "";
    chip.title = chipTitle(m);

    /* the flourish: one sweep the moment a module finishes */
    const was = S.prevState.get(m.key);
    if (was && was !== m.state && m.state === "done" && !REDUCED.matches) {
      chip.classList.add("st-flash");
      setTimeout(() => chip.classList.remove("st-flash"), 950);
    }
    S.prevState.set(m.key, m.state);
  }

  if (snap && snap.totals) {
    ui.stripBadge.textContent = moduleCountPhrase(snap);
    ui.stripBadge.className = "badge";
  }
}

function chipTitle(m) {
  const lines = [`${m.name || moduleName(m.key)} — ${m.state}`];
  const blurb = moduleBlurb(m.key);
  if (blurb) lines.push(blurb);
  if (m.state === "skipped") lines.push("Disabled for this run, so it did no work.");
  if (m.session === "prior") lines.push("Already up to date — skipped this session.");
  if (m.error_msg) lines.push(m.error_msg);
  return lines.join("\n");
}


/* ----------------------------------------------------------- module detail */

function renderDetail(snap) {
  const ui = S.ui;
  const key = S.selected;
  if (!key) { ui.detail.classList.remove("open"); return; }

  const m = (snap && snap.modules || []).find((x) => x.key === key)
    || { key, name: moduleName(key), state: "pending" };

  clear(ui.detail);
  ui.detail.classList.add("open");
  ui.detail.className = `card st-detail open ${
    m.state === "error" ? "bad" : m.state === "running" ? "info" : m.state === "done" ? "good" : "why"}`;

  const rows = [
    ["State", el("span", el(`span.pill.${STATE_PILL[m.state] || "x"}`, m.state || "pending"),
      m.session === "prior" ? el("span.dim", "  already up to date") : null)],
    ["Progress", `${fmtNum(m.pct, { digits: 1 })}%  (${fmtNum(m.n_done)} / ${fmtNum(m.n_total)})`],
    ["Errors", num(m.n_error) ? String(m.n_error) : "none"],
    ["No work", num(m.n_no_work) ? `${fmtNum(m.n_no_work)} specimens had nothing for it to do` : "none"],
    ["Running as", m.exec_mode ? execLabel(m) : "–"],
    ["Device", m.device || "–"],
    ["Elapsed", shortDur(m.elapsed_s)],
    ["Rate", rateLabel(m.rate_per_s)],
    ["ETA", num(m.eta_s) === null ? "–" : fmtDuration(m.eta_s)],
    ["Started", m.started_at ? fmtTime(Date.parse(m.started_at) / 1000, { withDate: true }) : "–"],
    ["Finished", m.finished_at ? fmtTime(Date.parse(m.finished_at) / 1000, { withDate: true }) : "–"],
    ["Depends on", (m.depends_on || []).length
      ? (m.depends_on || []).map(moduleName).join(", ") : "nothing"],
    ["Enabled", m.enabled === null || m.enabled === undefined ? "unknown" : (m.enabled ? "yes" : "no")],
  ];
  if (m.fanout) rows.push(["Fan-out", "one work item per leaf; the ledger still counts specimens"]);
  if (m.exec_note) rows.push(["Why serial", m.exec_note]);
  if (m.error_msg) rows.push(["Error", m.error_msg]);

  ui.detail.append(
    el("div.st-dethd",
      el("span.nm", m.name || moduleName(key)),
      el(`span.pill.${STATE_PILL[m.state] || "x"}`, m.state || "pending"),
      el("button.btn.sm.ghost", { type: "button", onclick: () => { S.selected = null; renderStrip(S.snap); renderDetail(S.snap); } }, "Close")),
    moduleBlurb(key) ? el("p", moduleBlurb(key)) : null,
    el("ul.kv", rows.map(([k, v]) =>
      el("li", el("span.k", k), el("span.v", v instanceof Node ? v : String(v))))));
}


/* -------------------------------------------------------------- idle panel */

function renderIdle() {
  const ui = S.ui;
  if (ui.idleBuilt) return;                     // static until a run appears
  ui.lastRun = el("div.card.why", el("span.dim", "Looking for a previous run…"));
  clear(ui.idle);
  ui.idle.append(
    el("div.empty",
      el("div.ic", "🌿"),
      el("div.t", "No LeafMachine3 run in flight"),
      el("div.s", "Set the input and output folders above and start a run. "
                + "The live view fills in the moment the first module begins.")),
    el("div.st-sec",
      el("div.sechd", "Last run"),
      ui.lastRun),
    el("div.st-sec",
      el("div.sechd", "What LM3 will do"),
      /* Its OWN strip — the live strip belongs to .st-live, and appending a node
         here would MOVE it out of the panel that needs it once a run starts. */
      staticStrip()));
  ui.idleBuilt = true;
  loadLastRun();
}

/** The 16 modules as inert, self-describing chips, for the idle pane. */
function staticStrip() {
  return el("div.st-strip", MODULE_INFO.map(([key, name, blurb], i) =>
    el("div.st-mod.pending", { title: `${name}\n${blurb}` },
      el("span.ord", String(i + 1)),
      el("span.nm", name),
      el("span.trk", el("i")))));
}

/**
 * Summarize the most recent run so the idle pane says something true instead of
 * sitting dead.
 *
 * DEFENSIVE ON PURPOSE: two backend modules both publish GET /v1/runs
 * (progress_api's run list and results_api's), and whichever router the
 * integrator mounts first wins. The two payloads name their fields differently
 * (`run_name` vs `name`/`id`), so this reads either shape rather than betting on
 * the mount order.
 */
async function loadLastRun() {
  const box = S.ui.lastRun;
  if (!box) return;
  try {
    const data = await api.listRuns();
    const runs = (data && data.runs) || [];
    if (!runs.length) {
      clear(box);
      box.appendChild(el("span.dim", "No previous runs found under the configured output folder."));
      return;
    }
    const r = runs[0];
    const name = r.run_name || r.name || r.id || "run";
    const started = r.started || r.started_at;
    const finished = r.finished || r.finished_at;
    clear(box);
    box.append(
      el("h4", name),
      el("ul.kv",
        el("li", el("span.k", "State"), el("span.v",
          el(`span.pill.${r.state === "done" ? "m" : r.state === "error" ? "e" : "x"}`, r.state || "unknown"))),
        el("li", el("span.k", "Specimens"), el("span.v", fmtNum(r.n_images))),
        el("li", el("span.k", "Modules"), el("span.v",
          `${fmtNum(r.modules_done)} / ${fmtNum(r.modules_total)}`)),
        el("li", el("span.k", "Started"), el("span.v", started || "–")),
        el("li", el("span.k", "Finished"), el("span.v", finished || "–")),
        el("li", el("span.k", "Folder"), el("span.v.mono", { title: r.path || "" }, fmtPath(r.path, 3)))));
  } catch (err) {
    clear(box);
    box.appendChild(el("span.dim", err && err.isAuth
      ? "Cannot read the run list — the server token was rejected."
      : "Cannot reach the LM3 server to list previous runs."));
  }
}


/* =========================================================== BOTTOM: console */

function buildConsole(pane, ui) {
  /* --- header ------------------------------------------------------------- */
  ui.connBadge = el("span.badge.dotd", "connecting");
  ui.runLabel = el("span.st-hd-run", "");
  ui.lineCount = el("span.st-lc", "0 lines");

  ui.levelBtns = LEVELS.map((lv) =>
    el(`button.btn.sm.ghost${lv === S.filter.min ? ".on" : ""}`, {
      type: "button",
      title: `Show ${lv} and above`,
      onclick: () => setLevel(lv),
    }, lv === "WARNING" ? "WARN" : lv));

  ui.search = el("input", {
    type: "search", placeholder: "Filter lines…", "aria-label": "Filter console lines",
    oninput: () => { S.filter.q = ui.search.value.trim(); rerenderConsole(); },
  });

  ui.followBtn = el(`button.btn.sm.ghost${S.follow ? ".on" : ""}`, {
    type: "button", title: "Keep the newest line in view",
    onclick: () => setFollow(!S.follow, true),
  }, "Follow");

  ui.copyBtn = el("button.btn.sm.ghost", {
    type: "button", title: "Copy the lines currently shown", onclick: copyVisible,
  }, "Copy");

  const clearBtn = el("button.btn.sm.ghost", {
    type: "button", title: "Clear the console view (the log file is untouched)",
    onclick: () => { S.lines = []; rerenderConsole(); },
  }, "Clear");

  // No collapse control: the console owns its own tab now, so folding it away inside that tab
  // would leave an empty page rather than free up space for anything.
  ui.consoleHd = el("div.console-hd",
    el("span.sechd-inline", "Console"),
    ui.connBadge,
    ui.runLabel,
    el("div.tools",
      el("div.btngroup", ui.levelBtns),
      el("div.searchbox", ui.search),
      ui.followBtn, ui.copyBtn, clearBtn,
      ui.lineCount));

  /* --- body --------------------------------------------------------------- */
  ui.console = el("div.console", { onscroll: onConsoleScroll });
  ui.jump = el("button.btn.sm.accent.st-jump", {
    type: "button", hidden: true, onclick: () => setFollow(true, true),
  }, "Jump to newest ↓");

  pane.append(ui.consoleHd, ui.console, ui.jump);
  rerenderConsole();
}

function setLevel(lv) {
  S.filter.min = lv;
  store.set(LS.level, lv);
  S.ui.levelBtns.forEach((b, i) => b.classList.toggle("on", LEVELS[i] === lv));
  rerenderConsole();
}

function setFollow(on, scroll) {
  S.follow = !!on;
  store.set(LS.follow, S.follow ? "1" : "0");
  S.ui.followBtn.classList.toggle("on", S.follow);
  S.ui.jump.hidden = S.follow;
  if (S.follow && scroll) scrollConsoleToEnd();
}

function scrollConsoleToEnd() {
  const c = S.ui.console;
  c.scrollTop = c.scrollHeight;
}

/** Pause following the moment the user scrolls away; resume when they come back. */
function onConsoleScroll() {
  const c = S.ui.console;
  const atEnd = c.scrollHeight - c.scrollTop - c.clientHeight <= 24;
  if (atEnd === S.follow) { S.ui.jump.hidden = S.follow; return; }
  setFollow(atEnd, false);
}

function setConsoleRun(name) {
  const label = name ? `run: ${name}` : "";
  if (S.ui.runLabel.textContent !== label) S.ui.runLabel.textContent = label;

  /* A DIFFERENT run means everything on screen belongs to the old one. Starting a new run without
     restarting the app must not leave the previous run's log lines, counters or per-module tiles
     behind -- that is the stale state the app would otherwise show for the whole next run. */
  if (name && S.consoleRunName && name !== S.consoleRunName) resetForNewRun(name);
  S.consoleRunName = name || S.consoleRunName;
}

/** Drop everything tied to the previous run so the new one starts from a clean slate. */
function resetForNewRun(name) {
  S.lines = [];
  S.snap = null;
  S.follow = true;
  S.selected = null;                  // drop the pinned module from the old run
  rerenderConsole();
  if (S.ui.jump) S.ui.jump.hidden = true;
  if (S.ui.search) { S.ui.search.value = ""; S.filter.q = ""; }
  // these repaint from the next snapshot; clear them so nothing from the old run lingers
  if (S.ui.strip) clear(S.ui.strip);
  if (S.ui.workers) clear(S.ui.workers);
  if (S.ui.nowRow) clear(S.ui.nowRow);
  console.info(`LM3: cleared the live view for run ${name}`);
}


function setConn(state, note) {
  const b = S.ui.connBadge;
  if (!b) return;
  const map = { up: ["ok", "live"], wait: ["warn", "connecting"], down: ["bad", "offline"] };
  const [tone, text] = map[state] || map.wait;
  b.className = `badge dotd ${tone}`;
  b.textContent = note || text;
}

/** Does a record pass the current filter? */
function passes(rec) {
  const rank = LEVEL_RANK[rec.level] ?? 1;
  if (rank < (LEVEL_RANK[S.filter.min] ?? 1)) return false;
  if (S.filter.q && !rec._hay.includes(S.filter.q.toLowerCase())) return false;
  return true;
}

function lineNode(rec) {
  const div = document.createElement("div");
  div.className = `line ${rec.level || "INFO"}${rec.trace ? " trace" : ""}${rec.mark ? " mark" : ""}`;
  div._uid = rec.uid;

  const t = document.createElement("span");
  t.className = "t";
  t.textContent = rec.t || "";

  const src = document.createElement("span");
  src.className = "src";
  src.textContent = rec.src || "";
  src.title = rec.logger || rec.src || "";

  const msg = document.createElement("span");
  msg.className = "msg";
  if (S.filter.q) msg.innerHTML = highlight(rec.msg, S.filter.q);
  else msg.textContent = rec.msg ?? "";

  div.append(t, src, msg);
  return div;
}

/** Ingest a batch of parsed log lines from the stream. */
function ingestLines(lines) {
  if (!lines || !lines.length) return;
  const c = S.ui.console;
  const wasFollowing = S.follow;
  const frag = document.createDocumentFragment();
  let appended = 0;

  for (const raw of lines) {
    const rec = {
      uid: ++S.uid,
      t: raw.t, level: raw.level || "INFO", src: raw.src, logger: raw.logger,
      msg: raw.msg ?? "", trace: !!raw.trace, mark: !!raw.mark, kind: raw.kind || null,
    };
    rec._hay = `${rec.msg} ${rec.src || ""}`.toLowerCase();
    S.lines.push(rec);
    if (passes(rec)) { frag.appendChild(lineNode(rec)); appended += 1; }
  }

  const empty = c.querySelector(".console-empty");
  if (empty && appended) empty.remove();
  if (appended) c.appendChild(frag);

  trimConsole();
  S.ui.lineCount.textContent = `${fmtNum(S.lines.length)} line${S.lines.length === 1 ? "" : "s"}`;
  if (wasFollowing) scrollConsoleToEnd();
}

/**
 * Hold the console to LOG_CAP lines, dropping from the front. The DOM is
 * trimmed by comparing our own monotonic uid against the model's oldest, which
 * survives the backend's per-run `seq` reset.
 */
function trimConsole() {
  if (S.lines.length <= LOG_CAP) return;
  S.lines.splice(0, S.lines.length - LOG_CAP);
  const oldest = S.lines[0].uid;
  const c = S.ui.console;
  while (c.firstChild && c.firstChild._uid !== undefined && c.firstChild._uid < oldest) {
    c.removeChild(c.firstChild);
  }
}

/** Full repaint — only on a filter change or a run switch, never per frame. */
function rerenderConsole() {
  const c = S.ui.console;
  clear(c);
  const frag = document.createDocumentFragment();
  let n = 0;
  for (const rec of S.lines) {
    if (!passes(rec)) continue;
    frag.appendChild(lineNode(rec));
    n += 1;
  }
  if (n) {
    c.appendChild(frag);
  } else {
    c.appendChild(el("div.console-empty",
      S.lines.length
        ? `No lines match ${S.filter.q ? `"${S.filter.q}" at ` : ""}${S.filter.min} and above.`
        : "Waiting for LM3 to write to the run log…"));
  }
  S.ui.lineCount.textContent = `${fmtNum(S.lines.length)} line${S.lines.length === 1 ? "" : "s"}`;
  if (S.follow) scrollConsoleToEnd();
}

async function copyVisible() {
  const text = Array.from(S.ui.console.querySelectorAll(".line")).map((n) => {
    const [t, src, msg] = n.children;
    return `${t.textContent} ${(src.textContent || "").padEnd(28)} ${msg.textContent}`;
  }).join("\n");

  const done = (ok) => {
    const btn = S.ui.copyBtn;
    btn.textContent = ok ? "Copied ✓" : "Copy failed";
    setTimeout(() => { btn.textContent = "Copy"; }, 1400);
  };
  try {
    await navigator.clipboard.writeText(text);
    done(true);
  } catch {
    /* Clipboard API blocked (some Electron configurations). Fall back to the
       selection-based path, which works without a permission prompt. */
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.cssText = "position:fixed;left:-9999px;top:0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      done(ok);
    } catch { done(false); }
  }
}


/* ================================================================= streams */

function connect() {
  S.closeStatus = api.streamStatus({
    onOpen: () => { S.drops = 0; stopPolling(); setConn("up"); },
    onMessage: (frame) => {
      if (!frame || frame.type !== "status") return;
      S.drops = 0;
      stopPolling();
      setConn("up");
      renderSnapshot(frame.snapshot);
    },
    onError: (err, retries) => {
      S.drops = retries;
      setConn("down");
      if (retries >= POLL_AFTER_DROPS) startPolling();
    },
  });

  S.closeLogs = api.streamLogs({
    params: { backfill: LOG_BACKFILL },
    onMessage: (frame) => {
      if (!frame) return;
      if (frame.type === "log") {
        ingestLines(frame.lines);
      } else if (frame.type === "logmeta") {
        /* `reset` means a different file: a new run, or the log rotated. Keeping
           the old lines would splice two runs into one scrollback. */
        if (frame.reset) { S.lines = []; S.uid = 0; rerenderConsole(); }
        if (frame.state === "missing") setConn("wait", "no log yet");
        else if (frame.state === "waiting") setConn("wait", "waiting for log");
        else if (frame.state === "rotated") setConn("up", "log rotated");
        // "tailing" needs a branch of its own: without it nothing ever clears the "waiting for
        // log" set above, so a console happily streaming a live run still claims to be waiting.
        else if (frame.state === "tailing") setConn("up");
      }
    },
  });
}

/** Poll /v1/status when the stream cannot hold. Stops as soon as a frame lands. */
function startPolling() {
  if (S.pollTimer) return;
  setConn("wait", "polling");
  S.pollTimer = setInterval(pollOnce, POLL_INTERVAL_MS);
}
function stopPolling() {
  if (!S.pollTimer) return;
  clearInterval(S.pollTimer);
  S.pollTimer = 0;
}

async function pollOnce() {
  try {
    const snap = await api.getStatus();
    renderSnapshot(snap);
    if (!S.pollTimer) setConn("up");
  } catch (err) {
    setConn("down", err && err.isAuth ? "unauthorized" : "offline");
  }
}


/* ============================================================ split gutter */

function applySplit(split, ratio) {
  const r = Math.max(0.15, Math.min(0.85, ratio));
  split.style.setProperty("--split-top", `${r}fr`);
  split.style.setProperty("--split-bot", `${1 - r}fr`);
  return r;
}

function wireSplit(split, gutter) {
  let dragging = false;

  const ratioFromY = (clientY) => {
    const box = split.getBoundingClientRect();
    if (!box.height) return 0.5;
    return (clientY - box.top) / box.height;
  };

  gutter.addEventListener("pointerdown", (ev) => {
    dragging = true;
    gutter.classList.add("dragging");
    gutter.setPointerCapture(ev.pointerId);
    ev.preventDefault();
  });

  gutter.addEventListener("pointermove", (ev) => {
    if (!dragging) return;
    applySplit(split, ratioFromY(ev.clientY));
  });

  const end = (ev) => {
    if (!dragging) return;
    dragging = false;
    gutter.classList.remove("dragging");
    try { gutter.releasePointerCapture(ev.pointerId); } catch { /* already released */ }
    store.set(LS.split, applySplit(split, ratioFromY(ev.clientY)));
    if (S.follow) scrollConsoleToEnd();
  };
  gutter.addEventListener("pointerup", end);
  gutter.addEventListener("pointercancel", end);

  gutter.addEventListener("dblclick", () => {
    store.set(LS.split, applySplit(split, 0.5));
  });

  /* keyboard: the gutter is a focusable separator, so it must be drivable */
  gutter.addEventListener("keydown", (ev) => {
    const cur = parseFloat(split.style.getPropertyValue("--split-top")) || 0.5;
    let next = null;
    if (ev.key === "ArrowUp") next = cur - 0.03;
    else if (ev.key === "ArrowDown") next = cur + 0.03;
    else if (ev.key === "Home") next = 0.2;
    else if (ev.key === "End") next = 0.8;
    if (next === null) return;
    ev.preventDefault();
    store.set(LS.split, applySplit(split, next));
  });
}


/* ================================================================= teardown */

function destroy() {
  if (!S) return;
  try { if (S.closeStatus) S.closeStatus(); } catch { /* already closed */ }
  try { if (S.closeLogs) S.closeLogs(); } catch { /* already closed */ }
  stopPolling();
  stopTicker();
  clear(S.root);
  S = null;
}

/**
 * Mount the LM3 console into its own tab.
 *
 * The console is built by initStatus (it owns the log SSE, the level filter and the follow
 * behavior), so this only re-parents the finished node -- which keeps every line collected while
 * the tab was never opened. Live Status is initialized first by the shell, so the node exists;
 * if it somehow does not, initStatus is run headlessly to create it.
 */
export function initConsole(root) {
  // S is null until initStatus runs, which is NOT guaranteed: the app can deep-link straight to
  // #console on load. Build the status state headlessly in that case so the console has its stream.
  if (!S || !S.ui || !S.ui.consoleHost) {
    const stash = document.createElement("div");
    initStatus(stash);
  }
  root.classList.add("fill", "st-consoletab");
  root.appendChild(S.ui.consoleHost);
  if (S.follow) scrollConsoleToEnd();
}

/**
 * The top bar's "New run" button clears the GUI down to "nothing has run yet".
 * Registered at module scope so it works even if this tab was never opened --
 * the console buffer accumulates from the moment the app starts.
 */
document.addEventListener("lm3:newrun", () => {
  newRunPending = true;              // set FIRST: the rest needs S, this must not depend on it
  if (!S) return;                    // never mounted yet -- the flag is applied by its first render
  S.consoleRunName = null;
  resetForNewRun(null);
  renderIdle();
});

export default initStatus;
