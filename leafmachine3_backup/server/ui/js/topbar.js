/* ==========================================================================
   LM3 — top bar: everything ABOVE the tabs
   --------------------------------------------------------------------------
   Two stacked pieces, both pinned outside every tab so they never scroll away:

     (A) THE GLOBAL STAGE PROGRESS BAR
         One segment per LM3 module in STAGE_ORDER, each segment sized by that
         module's share of the expected wall-clock time (progress_api hands us
         `weight_s`), filling as the module progresses. Under it, the single
         overall bar, plus the active module, its counts, elapsed and ETA.

     (B) THE PRIMARY SETTINGS STRIP
         Input folder, output folder, temp-file location, the settings file in
         use, the target device, and Start / Stop LM3.

   WHY the two live in one module: they are one control surface. The strip
   decides what a run will do; the bar reports what that run IS doing; Start /
   Stop moves between the two. Splitting them would mean two copies of the run
   record, the settings tree and the status stream.

   Usage (integrator, index.html):
       import { initTopBar } from "./js/topbar.js";
       const topbar = initTopBar(document.querySelector(".app"));

   `root` may be the .app grid, an existing .topbar element, or any container;
   the mounts are found if present and created in the right order if not.

   Events dispatched on `document` (all bubbling CustomEvents):
       lm3:topbar-ready   detail {topbar, primary}
       lm3:status         detail = the /v1/status snapshot, on every frame
       lm3:run            detail = the /v1/run/active record, only when it moves
       lm3:navigate       detail {tab:"status"|"settings", module?, path?}
       lm3:focus-module   detail {key, name, order, module}
       lm3:settings-saved detail {path, value, settings}
   ========================================================================== */

import {
  api, el, append, clear, debounce,
  fmtDuration, fmtNum, fmtPath,
} from "./api.js";
/* The 17-module table used to live here. It now lives in ./modules.js because the
   settings rail renders the same list in its own order, and two hand-maintained
   copies is exactly how Bilateral Symmetry went missing from the settings tree.
   The stage bar wants RUN order; the rail wants reading order. One table, two
   sorts. */
import { MODULES_BY_RUN_ORDER as MODULES } from "./modules.js";

/* Human wording for the run states metrics_api reports. */
const RUN_STATE_TEXT = {
  idle: "Idle", running: "Running", stopping: "Stopping",
  done: "Complete", error: "Failed",
};
const RUN_STATE_BADGE = {
  idle: "", running: "info", stopping: "warn", done: "ok", error: "bad",
};

/* Polling cadence for the run record. The status SSE carries the fine-grained
   progress; this record only supplies pid / exit code / launcher errors, which
   move rarely — so we ask often while something is in flight and rarely when
   nothing is. */
const RUN_POLL_ACTIVE_MS = 4000;
const RUN_POLL_IDLE_MS = 15000;


/* ==========================================================================
   SMALL UTILITIES
   ========================================================================== */

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/** Structured-clone if available, JSON round-trip otherwise (settings trees
    are plain YAML scalars/maps/lists, so both are lossless here). */
function cloneTree(obj) {
  if (typeof structuredClone === "function") {
    try { return structuredClone(obj); } catch { /* fall through */ }
  }
  return JSON.parse(JSON.stringify(obj ?? {}));
}

/** deepGet(values, "project.output.dir") */
function deepGet(obj, path, fallback = undefined) {
  let cur = obj;
  for (const part of String(path).split(".")) {
    if (cur === null || cur === undefined || typeof cur !== "object") return fallback;
    cur = cur[part];
  }
  return cur === undefined ? fallback : cur;
}

/** deepSet(values, "project.output.dir", "/data/out") — creates missing maps. */
function deepSet(obj, path, value) {
  const parts = String(path).split(".");
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const k = parts[i];
    if (cur[k] === null || cur[k] === undefined || typeof cur[k] !== "object" || Array.isArray(cur[k])) {
      cur[k] = {};
    }
    cur = cur[k];
  }
  cur[parts[parts.length - 1]] = value;
  return obj;
}

/** "1,204" but "–" for null — the topstats want a dash, not a zero. */
function num(v) {
  return (v === null || v === undefined) ? "–" : fmtNum(v, { digits: 0 });
}

function pctText(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "–";
  return `${Number(v).toFixed(digits)}%`;
}

/** Pull the human message out of an ApiError, whatever shape the server used. */
function errText(err) {
  const d = err && err.body && err.body.detail;
  if (d && typeof d === "object") {
    if (Array.isArray(d.errors) && d.errors.length) return d.errors.slice(0, 2).join(" · ");
    if (d.message) return d.message;
  }
  if (typeof d === "string" && d) return d;
  if (err && err.isOffline) return "the LM3 server is not reachable";
  return (err && err.message) || "unknown error";
}


/* ==========================================================================
   TOASTS
   The design system ships .toasts / .toast but nothing owns creating the
   container, so we adopt an existing one and only mint it when absent. That
   keeps every module's toasts in ONE stack instead of overlapping columns.
   ========================================================================== */
function toastHost() {
  let host = document.querySelector(".toasts");
  if (!host) {
    host = el("div.toasts", { role: "status", "aria-live": "polite" });
    document.body.appendChild(host);
  }
  return host;
}

/**
 * @param {string} title
 * @param {string} [body]
 * @param {{kind?:"ok"|"warn"|"bad", ms?:number, action?:{label:string,onClick:Function}}} [opts]
 */
function toast(title, body, opts = {}) {
  const { kind = "", ms = 5200, action = null } = opts;
  const node = el(`div.toast${kind ? `.${kind}` : ""}`,
    el("span.t", title),
    body ? document.createTextNode(body) : null,
  );
  let timer = null;
  const dismiss = () => {
    if (timer) clearTimeout(timer);
    node.classList.add("leaving");
    setTimeout(() => node.remove(), 200);
  };
  if (action) {
    node.appendChild(el("div", { style: { marginTop: "8px", display: "flex", gap: "6px" } },
      el("button.btn.sm.accent", { type: "button", onclick: () => { dismiss(); action.onClick(); } }, action.label),
      el("button.btn.sm.ghost", { type: "button", onclick: dismiss }, "Dismiss"),
    ));
  }
  toastHost().appendChild(node);
  if (ms > 0) timer = setTimeout(dismiss, action ? Math.max(ms, 12000) : ms);
  return dismiss;
}


/* ==========================================================================
   MODAL — a promise-returning overlay built from .backdrop / .modal
   Used for the Stop confirmation and the folder browser. window.confirm()
   would block the SSE handlers and looks nothing like the rest of LM3.
   ========================================================================== */
function openModal({ title, build, width = 560, onClose = null }) {
  const backdrop = el("div.backdrop");
  const panel = el("div.modal", { style: { maxWidth: `min(${width}px, 100%)`, width: "100%" } });
  const head = el("div.panel-hd", el("span.t", title));
  const closeBtn = el("button.btn.sm.ghost", { type: "button", "aria-label": "Close" }, "✕");
  head.appendChild(el("span.tools", closeBtn));
  panel.appendChild(head);
  backdrop.appendChild(panel);

  let done = false;
  const close = (result) => {
    if (done) return;
    done = true;
    document.removeEventListener("keydown", onKey, true);
    backdrop.remove();
    if (onClose) onClose(result);
  };
  const onKey = (ev) => {
    if (ev.key === "Escape") { ev.stopPropagation(); close(undefined); }
  };
  closeBtn.addEventListener("click", () => close(undefined));
  backdrop.addEventListener("mousedown", (ev) => { if (ev.target === backdrop) close(undefined); });
  document.addEventListener("keydown", onKey, true);

  build(panel, close);
  document.body.appendChild(backdrop);
  return close;
}

/** A styled replacement for window.confirm(). Resolves true/false. */
function confirmModal({ title, message, detail = null, confirmLabel = "Confirm", danger = false }) {
  return new Promise((resolve) => {
    openModal({
      title,
      width: 460,
      onClose: (r) => resolve(r === true),
      build: (panel, close) => {
        panel.appendChild(el("div.panel-bd",
          el("p", { style: { margin: "0 0 8px", color: "var(--ink)" } }, message),
          detail ? el("p", { style: { margin: 0, fontSize: "12.5px", color: "var(--mute)", lineHeight: "1.55" } }, detail) : null,
        ));
        const ok = el(`button.btn.sm${danger ? ".danger" : ".accent"}`, { type: "button" }, confirmLabel);
        ok.addEventListener("click", () => close(true));
        panel.appendChild(el("div.panel-ft", { style: { display: "flex", gap: "8px", justifyContent: "flex-end" } },
          el("button.btn.sm.ghost", { type: "button", onclick: () => close(false) }, "Cancel"),
          ok,
        ));
        setTimeout(() => ok.focus(), 0);
      },
    });
  });
}


/* ==========================================================================
   FOLDER BROWSER — POST /v1/settings/browse
   The server never 500s on an unreadable or nonexistent path: it answers 200
   with readable:false, or falls back to the nearest existing ancestor and sets
   fell_back:true. Both are rendered as affordances, not errors.
   ========================================================================== */
function browseFolder({ title, start, countImages }) {
  return new Promise((resolve) => {
    let showHidden = false;
    let data = null;
    let busy = false;

    openModal({
      title,
      width: 780,
      onClose: (r) => resolve(typeof r === "string" ? r : null),
      build: (panel, close) => {
        const pathInput = el("input.mono", {
          type: "text", value: start || "", spellcheck: "false",
          placeholder: "/path/to/folder", style: { flex: "1 1 auto", minWidth: "0" },
        });
        const goBtn = el("button.btn.sm", { type: "button" }, "Go");
        const upBtn = el("button.btn.sm.ghost", { type: "button", title: "Parent folder" }, "↑ Up");
        const hiddenBtn = el("button.btn.sm.ghost", { type: "button" }, "Show hidden");
        const rootsRow = el("div", { style: { display: "flex", gap: "6px", flexWrap: "wrap", margin: "0 0 10px" } });
        const noteRow = el("div", { style: { margin: "0 0 10px" } });
        const crumbs = el("div.crumbs", { style: { margin: "0 0 10px" } });
        const listWrap = el("div.tblwrap.tall", { style: { margin: "0" } });
        const summary = el("div", { style: { fontSize: "11.5px", color: "var(--dim)", fontFamily: "var(--mono)" } });

        const useBtn = el("button.btn.sm.accent", { type: "button" }, "Use this folder");
        useBtn.addEventListener("click", () => { if (data && data.path) close(data.path); });

        panel.appendChild(el("div.panel-bd",
          el("div.toolbar", { style: { padding: "0 0 10px" } }, pathInput, goBtn, upBtn, hiddenBtn),
          rootsRow, noteRow, crumbs, listWrap,
        ));
        panel.appendChild(el("div.panel-ft", { style: { display: "flex", gap: "8px", alignItems: "center" } },
          summary,
          el("span.spacer"),
          el("button.btn.sm.ghost", { type: "button", onclick: () => close(null) }, "Cancel"),
          useBtn,
        ));

        async function load(path) {
          if (busy) return;
          busy = true;
          clear(listWrap).appendChild(el("div.empty", el("span.spinner.lg"), el("span.s", "Reading folder…")));
          try {
            data = await api.post("/v1/settings/browse", {
              path: path || null,
              show_hidden: showHidden,
              count_images: countImages,
              max_entries: 500,
            });
          } catch (err) {
            busy = false;
            clear(listWrap).appendChild(el("div.empty",
              el("span.ic", "⚠"), el("span.t", "Cannot list that folder"), el("span.s", errText(err))));
            return;
          }
          busy = false;
          draw();
        }

        function draw() {
          if (!data) return;
          pathInput.value = data.path || data.requested || "";

          /* quick jumps the server offers (cwd, output dir, input dirs, home…) */
          clear(rootsRow);
          for (const r of data.roots || []) {
            const chip = el("span.chip", { title: r.path }, r.label);
            chip.addEventListener("click", () => load(r.path));
            rootsRow.appendChild(chip);
          }

          /* fell_back means the typed path does not exist and we are showing
             its nearest existing ancestor — that is the cue to offer mkdir */
          clear(noteRow);
          if (data.error) {
            noteRow.appendChild(el("div.card.bad", el("p", data.error)));
          } else if (data.fell_back && data.requested) {
            const mk = el("button.btn.sm.accent", { type: "button" }, "Create it");
            mk.addEventListener("click", async () => {
              mk.disabled = true;
              try {
                const res = await api.post("/v1/settings/mkdir", { path: data.requested });
                toast("Folder created", res.path, { kind: "ok" });
                await load(res.path);
              } catch (err) {
                toast("Could not create that folder", errText(err), { kind: "bad" });
                mk.disabled = false;
              }
            });
            noteRow.appendChild(el("div.card.warn",
              el("p", "That folder does not exist yet — showing the nearest existing parent."),
              el("p", { style: { display: "flex", gap: "8px", alignItems: "center", marginTop: "6px" } },
                el("code.mono", { style: { color: "var(--warn)" } }, data.requested), mk),
            ));
          } else if (data.readable === false) {
            noteRow.appendChild(el("div.card.warn", el("p", "This folder exists but cannot be read with the server's permissions.")));
          }

          /* breadcrumbs from the absolute path */
          clear(crumbs);
          const abs = String(data.path || "");
          const segs = abs.split("/").filter(Boolean);
          const rootCrumb = el("a", "/");
          rootCrumb.addEventListener("click", () => load("/"));
          crumbs.appendChild(rootCrumb);
          let acc = "";
          segs.forEach((s, i) => {
            acc += `/${s}`;
            const target = acc;
            if (i) crumbs.appendChild(el("span.sep", "/"));
            if (i === segs.length - 1) {
              crumbs.appendChild(el("span.cur", s));
            } else {
              const a = el("a", s);
              a.addEventListener("click", () => load(target));
              crumbs.appendChild(a);
            }
          });

          /* entries */
          const entries = data.entries || [];
          clear(listWrap);
          if (!entries.length) {
            listWrap.appendChild(el("div.empty",
              el("span.ic", "📂"), el("span.t", "No subfolders here"),
              el("span.s", "Use this folder, or type a path above.")));
          } else {
            const tbody = el("tbody");
            for (const e of entries) {
              const row = el("tr", { style: { cursor: e.readable === false ? "not-allowed" : "pointer" } },
                el("td.mono", { style: { color: e.readable === false ? "var(--dim)" : "var(--ink)" } },
                  `${e.readable === false ? "🔒 " : "📁 "}${e.name}`),
                el("td.num", e.n_images === null || e.n_images === undefined ? "–" : fmtNum(e.n_images, { digits: 0 })),
                el("td.num", e.n_dirs === null || e.n_dirs === undefined ? "–" : fmtNum(e.n_dirs, { digits: 0 })),
              );
              if (e.readable !== false) row.addEventListener("click", () => load(e.path));
              tbody.appendChild(row);
            }
            listWrap.appendChild(el("table.datatable.dense",
              el("thead", el("tr",
                el("th.l", "Folder"),
                el("th", { title: "Images directly inside that folder" }, "Images"),
                el("th", "Subfolders"))),
              tbody,
            ));
          }

          const bits = [];
          if (data.n_images !== null && data.n_images !== undefined) bits.push(`${fmtNum(data.n_images, { digits: 0 })} images here`);
          if (data.n_dirs !== null && data.n_dirs !== undefined) bits.push(`${fmtNum(data.n_dirs, { digits: 0 })} subfolders`);
          if (data.entries_truncated || data.truncated) bits.push("list truncated");
          clear(summary).appendChild(document.createTextNode(bits.join(" · ")));
          /* is_dir describes the path the user TYPED, so after a fall-back it
             is false while data.path is a perfectly good folder — gate on the
             folder we are actually LISTING, not on the typed one */
          useBtn.disabled = !data.path || data.readable === false
                            || (!data.fell_back && data.is_dir === false);
        }

        goBtn.addEventListener("click", () => load(pathInput.value.trim()));
        upBtn.addEventListener("click", () => { if (data && data.parent) load(data.parent); });
        hiddenBtn.addEventListener("click", () => {
          showHidden = !showHidden;
          hiddenBtn.classList.toggle("on", showHidden);
          load(pathInput.value.trim());
        });
        pathInput.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter") { ev.preventDefault(); load(pathInput.value.trim()); }
        });

        setTimeout(() => pathInput.focus(), 0);
        load(start || null);
      },
    });
  });
}


/* ==========================================================================
   TOOLTIP — one floating node, moved and refilled on hover
   Styled inline from the design tokens rather than with new CSS classes,
   because css/app.css belongs to the design-system module and a hover card is
   not part of its published vocabulary. Every value below is a var(), so it
   still tracks the palette.
   ========================================================================== */
function makeTip() {
  const node = el("div.lm3-tip", {
    role: "tooltip",
    style: {
      position: "fixed", zIndex: "400", left: "0", top: "0",
      minWidth: "236px", maxWidth: "340px", padding: "11px 13px",
      background: "var(--panel)", border: "1px solid var(--line)",
      borderRadius: "var(--r)", boxShadow: "0 14px 40px rgba(0,0,0,.6)",
      color: "var(--mute)", fontSize: "12.3px", lineHeight: "1.5",
      pointerEvents: "none", opacity: "0", visibility: "hidden",
      transition: "opacity .12s ease",
    },
  });
  document.body.appendChild(node);

  return {
    node,
    show(anchor, content) {
      clear(node);
      append(node, content);
      node.style.visibility = "hidden";
      node.style.opacity = "0";
      /* measure after fill so the clamp uses the real height */
      const r = anchor.getBoundingClientRect();
      const w = node.offsetWidth, h = node.offsetHeight;
      let left = r.left + r.width / 2 - w / 2;
      left = clamp(left, 8, Math.max(8, window.innerWidth - w - 8));
      let top = r.bottom + 8;
      if (top + h > window.innerHeight - 8) top = Math.max(8, r.top - h - 8);
      node.style.left = `${Math.round(left)}px`;
      node.style.top = `${Math.round(top)}px`;
      node.style.visibility = "visible";
      node.style.opacity = "1";
    },
    hide() {
      node.style.opacity = "0";
      node.style.visibility = "hidden";
    },
    destroy() { node.remove(); },
  };
}


/* ==========================================================================
   initTopBar
   ========================================================================== */

/**
 * Mount the global stage bar and the primary settings strip.
 *
 * @param {Element|string} [root]  .app grid, an existing .topbar, or any host
 * @param {{onModuleFocus?:Function, onNavigate?:Function}} [opts]
 * @returns {object} controller — see the bottom of this function
 */
export function initTopBar(root, opts = {}) {
  const { bar, strip } = resolveMounts(root);

  /* ---------------------------------------------------------------- state */
  const state = {
    snapshot: null,           // last /v1/status snapshot
    snapshotAt: 0,            // performance.now() when it arrived (for local ticking)
    run: null,                // last /v1/run/active record
    runApi: true,             // false once /v1/run/* answers 404 (module not mounted)
    settings: null,           // last GET /v1/settings (values + effective + mtime)
    settingsApi: true,
    hardware: null,           // gpus for the device picker
    conn: "wait",             // up | wait | down
    busy: null,               // "starting" | "stopping" | null
    collapsed: false,         // main-settings strip folded away
    userPinnedOpen: false,    // user expanded it by hand -> do not auto-collapse over them
    wasRunning: false,        // for the idle -> running edge that triggers the auto-collapse
    trackKeys: "",            // signature of the rendered segment list
    closeStatus: null,        // SSE closer
    statusDb: null,           // db path the stream is pinned to
    runTimer: null,
    tickTimer: null,
    destroyed: false,
  };

  const refs = {};
  const tip = makeTip();

  /* ---------------------------------------------------------------- build */
  buildTopBar();
  buildPrimary();

  /* ------------------------------------------------------------ boot data */
  loadHealth();
  // On open, the paths in the strip already point somewhere -- at the project the last run left
  // behind. openFresh() clears the name so this opens ready for a NEW run instead.
  loadSettings().then(() => { renderTrack(); openFresh(); });
  loadHardware();
  connectStatus(null);
  pollRun();
  state.tickTimer = setInterval(tickClocks, 1000);
  window.addEventListener("resize", onResize, { passive: true });

  dispatch("lm3:topbar-ready", { topbar: bar, primary: strip });


  /* ======================================================================
     (A) GLOBAL STAGE PROGRESS
     ====================================================================== */

  function buildTopBar() {
    clear(bar);

    refs.ver = el("span.ver", "v3");
    const brand = el("div.brand", el("span.mark"), "LeafMachine3", refs.ver);

    refs.nowWhat = el("span.what", "Idle");
    refs.nowWho = el("span.who", "No run yet");
    refs.nowCount = el("span.pill.x", "–");
    const stagenow = el("div.stagenow", { style: { cursor: "pointer" }, title: "Open the live status" },
      refs.nowWhat, refs.nowWho, refs.nowCount);
    stagenow.addEventListener("click", () => go("status"));

    refs.statImages = statNode("Images");
    refs.statOverall = statNode("Overall");
    refs.statElapsed = statNode("Elapsed");
    refs.statEta = statNode("ETA");
    refs.conn = el("span.conn.wait", "connecting");
    const topstats = el("div.topstats",
      refs.statImages.node, refs.statOverall.node, refs.statElapsed.node, refs.statEta.node, refs.conn);

    refs.track = el("div.stagetrack", { role: "group", "aria-label": "LM3 module progress" });

    refs.fill = el("span.fill");
    refs.fillLabel = el("span.lbl", "0%");
    refs.bar = el("div.bar.lg.idle", { role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": "0" },
      refs.fill, refs.fillLabel);
    refs.metaLeft = el("span", "No LM3 run yet");
    refs.metaRight = el("span.r", "");
    /* The blue fill bar is built and kept up to date, but deliberately NOT mounted: the module
       timeline above already shows progress graphically, so the second bar was redundant. Only its
       meta line ("N of M modules done", run name, rate) stays visible. Keeping the node live rather
       than deleting it means the update path below needs no null guards and re-mounting it is a
       one-line change. The percentage it used to print is still shown by the "Overall" top stat. */
    const globalbar = el("div.globalbar.nobar", { style: { cursor: "pointer" } },
      el("div.meta", refs.metaLeft, refs.metaRight));
    globalbar.addEventListener("click", () => go("status"));

    append(bar, [el("div.topbar-main", brand, stagenow, topstats), refs.track, globalbar]);
    renderTrack();
  }

  function statNode(label) {
    const v = el("span.v", "–");
    const node = el("div.topstat", v, el("span.l", label));
    return { node, v, set(text, cls) {
      v.textContent = text;
      node.className = `topstat${cls ? ` ${cls}` : ""}`;
    } };
  }

  /**
   * Merge the live snapshot over the static module table.
   * Anything the snapshot does not cover falls back to what the settings tree
   * says about that module, so an idle app still shows which modules are
   * switched off instead of a uniformly gray strip.
   */
  function moduleList() {
    const snap = state.snapshot;
    const live = new Map();
    for (const m of (snap && snap.modules) || []) live.set(m.key, m);

    const out = [];
    for (const base of MODULES) {
      const m = live.get(base.key);
      live.delete(base.key);
      if (m) {
        out.push({ ...base, ...m, short: base.short });
      } else {
        const enabled = deepGet(effective(), `modules.${base.key}.enabled`, null);
        out.push({
          ...base, state: enabled === false ? "skipped" : "pending", enabled,
          n_done: 0, n_total: null, n_error: 0, pct: 0, weight_s: null,
          elapsed_s: null, eta_s: null, rate_per_s: null, error_msg: null,
          exec_mode: null, workers: null, planned_workers: null, session: null,
        });
      }
    }
    /* a module the server knows about and we do not — never silently drop it */
    for (const [key, m] of live) {
      out.push({ key, name: m.name || key, short: m.name || key, order: m.order ?? 999, depends: m.depends_on || [], ...m });
    }
    out.sort((a, b) => (a.order ?? 999) - (b.order ?? 999));
    return out;
  }

  function renderTrack() {
    const list = moduleList();
    const sig = list.map((m) => m.key).join("|");
    if (sig !== state.trackKeys) {
      state.trackKeys = sig;
      clear(refs.track);
      refs.segs = new Map();
      for (const m of list) {
        const subfill = el("span.subfill");
        const nm = el("span.nm", m.short || m.name);
        // The bar is the map of the pipeline people actually read, so it is also
        // the shortest route to a module's settings: the body goes to Live
        // Status as it always has, the cog goes to the settings pane for exactly
        // this module. Without it, "Petiole looks wrong" means translating a
        // stage name into a settings section by hand.
        const cog = el("button.segcog", {
          type: "button", tabindex: "-1",
          title: `${m.name} settings`,
          "aria-label": `${m.name} settings`,
        }, "⚙");
        cog.addEventListener("click", (ev) => {
          ev.stopPropagation();
          go("settings", { module: m.key });
        });
        const seg = el("div.stageseg.pending", {
          role: "button", tabindex: "0",
          style: { cursor: "pointer" },
          "aria-label": m.name,
        }, subfill, nm, cog);
        seg.addEventListener("pointerenter", () => tip.show(seg, tipContent(currentModule(m.key) || m)));
        seg.addEventListener("pointerleave", () => tip.hide());
        seg.addEventListener("click", () => focusModule(m.key));
        seg.addEventListener("keydown", (ev) => {
          if (ev.key !== "Enter" && ev.key !== " ") return;
          ev.preventDefault();
          // Shift is the keyboard route to the same place the cog goes.
          if (ev.shiftKey) go("settings", { module: m.key });
          else focusModule(m.key);
        });
        refs.segs.set(m.key, { seg, subfill, nm, cog });
        refs.track.appendChild(seg);
      }
    }

    /* Segment width = share of expected time. `weight_s` is progress_api's
       duration prior for THIS machine (measured where it can be, projected
       otherwise); without it every module gets an equal slice. Clamped so a
       3-second module is still hoverable and a dominant one cannot swallow
       the strip. */
    const active = list.filter((m) => m.state !== "skipped");
    let totalW = 0;
    for (const m of active) if (Number(m.weight_s) > 0) totalW += Number(m.weight_s);
    const nActive = Math.max(1, active.length);

    for (const m of list) {
      const r = refs.segs.get(m.key);
      if (!r) continue;
      const st = m.state || "pending";
      r.seg.className = `stageseg ${st}`;
      r.nm.textContent = m.short || m.name;

      let grow;
      if (st === "skipped") {
        grow = 0.42;
      } else if (totalW > 0 && Number(m.weight_s) > 0) {
        grow = clamp((Number(m.weight_s) / totalW) * nActive, 0.42, 3.6);
      } else {
        grow = 1;
      }
      if (st === "running") grow = Math.max(grow * 1.5, 1.9);
      r.seg.style.flexGrow = String(Math.round(grow * 100) / 100);

      const pct = clamp(Number(m.pct) || 0, 0, 100);
      if (st === "running") {
        r.subfill.style.width = `${pct}%`;
        r.subfill.style.background = "";
      } else if (st === "error") {
        r.subfill.style.width = `${pct}%`;
        r.subfill.style.background = "rgba(248,113,113,.30)";
      } else if (st === "pending" && pct > 0) {
        /* work carried over from an earlier session: shown, but muted */
        r.subfill.style.width = `${pct}%`;
        r.subfill.style.background = "rgba(156,163,175,.18)";
      } else {
        r.subfill.style.width = "0";
        r.subfill.style.background = "";
      }
      r.seg.setAttribute("aria-label", `${m.name} — ${st}${m.n_total ? `, ${m.n_done || 0} of ${m.n_total}` : ""}`);
    }
  }

  function currentModule(key) {
    return moduleList().find((m) => m.key === key) || null;
  }

  function tipContent(m) {
    const rows = [];
    const kv = (k, v) => rows.push(el("li", el("span.k", k), el("span.v", v)));

    if (m.n_total) {
      kv("Progress", `${num(m.n_done)} / ${num(m.n_total)}  (${pctText(m.pct)})`);
    } else if (m.state === "skipped") {
      kv("Progress", m.enabled === false ? "module disabled" : "nothing to do");
    } else {
      kv("Progress", m.state === "pending" ? "not started" : pctText(m.pct));
    }
    if (m.n_error) kv("Errors", num(m.n_error));
    if (m.elapsed_s !== null && m.elapsed_s !== undefined) kv("Elapsed", fmtDuration(m.elapsed_s));
    if (m.eta_s !== null && m.eta_s !== undefined) kv("ETA", fmtDuration(m.eta_s));
    if (m.rate_per_s) kv("Rate", `${fmtNum(m.rate_per_s, { digits: 2 })} / s`);

    const execBits = [];
    if (m.exec_mode) execBits.push(m.exec_mode);
    else if (m.device) execBits.push(m.device === "gpu" ? "gpu" : `cpu · ${m.exec}`);
    if (m.workers) execBits.push(`${m.workers} worker${m.workers === 1 ? "" : "s"}`);
    else if (m.planned_workers) execBits.push(`${m.planned_workers} planned`);
    if (execBits.length) kv("Execution", execBits.join(" · "));

    if (m.weight_s) kv("Time share", `${fmtDuration(m.weight_s)} (${m.weight_source || "projected"})`);
    if (m.session === "prior") kv("Session", "up to date from an earlier run");
    // An empty array is truthy, so `m.depends || m.depends_on` always took the
    // static table's value and never consulted the live one the server sends as
    // `depends_on`. Prefer whichever actually has entries.
    const dep = (m.depends_on && m.depends_on.length) ? m.depends_on : (m.depends || []);
    if (dep.length) {
      kv("Depends on", dep.map((k) => shortOf(k)).join(", "));
    }

    const stateCls = { done: "m", running: "s", error: "e", skipped: "x", pending: "x" }[m.state] || "x";
    return [
      el("div", { style: { display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" } },
        el("span", { style: { color: "var(--ink)", fontWeight: "650", fontSize: "13px" } }, m.name),
        el("span.spacer"),
        el(`span.pill.${stateCls}`, String(m.state || "pending").toUpperCase()),
      ),
      el("ul.kv", rows),
      m.exec_note ? el("div", { style: { marginTop: "8px", color: "var(--warn)", fontSize: "11.5px" } }, m.exec_note) : null,
      m.error_msg ? el("div", { style: { marginTop: "8px", color: "var(--bad)", fontSize: "11.5px", wordBreak: "break-word" } }, m.error_msg) : null,
      el("div", { style: { marginTop: "9px", color: "var(--dim)", fontSize: "11px", letterSpacing: ".04em", textTransform: "uppercase" } },
        "Click to open in Status"),
    ];
  }

  function shortOf(key) {
    const m = MODULES.find((x) => x.key === key);
    return m ? m.short : key;
  }

  function focusModule(key) {
    const m = currentModule(key);
    tip.hide();
    if (opts.onModuleFocus) { try { opts.onModuleFocus(key, m); } catch (err) { console.error("[lm3] onModuleFocus threw", err); } }
    dispatch("lm3:focus-module", { key, name: m ? m.name : key, order: m ? m.order : null, module: m });
    go("status", { module: key });
  }

  /** Repaint everything that depends on the status snapshot. */
  function renderStatus() {
    const s = state.snapshot;
    renderTrack();

    const running = s && s.state === "running";
    const starting = state.busy === "starting" || (!!(state.run && state.run.active) && (!s || !s.ready));

    /* ---- the "what is LM3 doing" line ---- */
    let what = "Idle", who = "No run yet", whoDim = true, count = null, countCls = "x";
    if (starting && !(s && s.active)) {
      what = "Starting"; who = (state.run && state.run.run_name) || "LM3"; whoDim = false;
    } else if (s && s.active) {
      what = s.stale ? "Stalled" : "Running";
      who = s.active.name;
      whoDim = false;
      count = `${num(s.active.n_done)} / ${num(s.active.n_total)}`;
      countCls = s.stale ? "w" : "s";
    } else if (s && s.state === "running" && s.next) {
      what = "Starting"; who = s.next.name; whoDim = false;
    } else if (s && s.state === "done") {
      what = "Complete"; who = s.run_name || "run"; whoDim = false; count = `${num(s.images_done)} images`; countCls = "m";
    } else if (s && s.state === "error") {
      what = "Failed"; who = s.run_name || "run"; whoDim = false; countCls = "e"; count = "see console";
    } else if (s && s.run_name) {
      what = "Last run"; who = s.run_name; whoDim = false;
    }
    refs.nowWhat.textContent = what;
    refs.nowWho.textContent = who;
    refs.nowWho.style.color = whoDim ? "var(--dim)" : "";
    refs.nowCount.className = `pill ${countCls}`;
    refs.nowCount.textContent = count || "";
    refs.nowCount.style.display = count ? "" : "none";

    /* ---- top stats ---- */
    const t = (s && s.totals) || {};
    refs.statImages.set(s && s.images_total ? `${num(s.images_done)} / ${num(s.images_total)}` : "–");
    const overall = t.overall_pct;
    refs.statOverall.set(pctText(overall), overall >= 100 ? "good" : "");
    tickClocks();

    /* ---- the one global bar ---- */
    const pct = clamp(Number(overall) || 0, 0, 100);
    let color = "idle";
    let stripe = false;
    if (s && s.state === "error") color = "bad";
    else if (s && s.state === "done") color = "ok";
    else if (running || starting) { color = "acc2"; stripe = !s || !s.stale; }
    refs.bar.className = `bar lg ${color}${stripe ? " running" : ""}${starting && pct < 0.5 ? " indeterminate" : ""}`;
    refs.fill.style.width = `${pct}%`;
    refs.fillLabel.textContent = starting && pct < 0.5 ? "starting…" : pctText(pct);
    refs.bar.setAttribute("aria-valuenow", String(Math.round(pct)));

    /* ---- the meta line under the bar ---- */
    const left = [];
    if (t.modules_total) {
      left.push(`${num(t.modules_done)} of ${num(t.modules_enabled ?? t.modules_total)} modules done`);
      if (t.modules_skipped) left.push(`${num(t.modules_skipped)} skipped`);
      if (t.session_pct !== null && t.session_pct !== undefined && Math.abs(t.session_pct - (overall || 0)) > 0.5) {
        left.push(`this session ${pctText(t.session_pct, 0)}`);
      }
    } else {
      left.push(state.runApi ? "No LM3 run yet — set the folders below and press Start" : "No LM3 run yet");
    }
    refs.metaLeft.textContent = left.join(" · ");

    const right = [];
    if (s && s.run_name) right.push(s.run_name);
    if (s && s.started_ts) right.push(`started ${new Date(s.started_ts * 1000).toLocaleTimeString("en-US", { hour12: false })}`);
    if (s && s.active && s.active.rate_per_s) right.push(`${fmtNum(s.active.rate_per_s, { digits: 2 })} img/s`);
    if (s && s.stale) right.push(`silent for ${fmtDuration(s.stale_for_s)}`);
    refs.metaRight.textContent = right.join("  ·  ");

    renderRunControls();
  }

  /**
   * Elapsed / ETA advance between frames. The snapshot is authoritative; we
   * only add the wall-clock time since it arrived, so the readout never drifts
   * away from the server's own accounting.
   */
  function tickClocks() {
    const s = state.snapshot;
    if (!s || !s.ready) {
      refs.statElapsed.set("–");
      refs.statEta.set("–");
      return;
    }
    const drift = s.state === "running" && !s.stale
      ? Math.max(0, (performance.now() - state.snapshotAt) / 1000) : 0;
    const elapsed = s.elapsed_s === null || s.elapsed_s === undefined ? null : s.elapsed_s + drift;
    refs.statElapsed.set(elapsed === null ? "–" : fmtDuration(elapsed));

    const eta = s.eta_s === null || s.eta_s === undefined ? null : Math.max(0, s.eta_s - drift);
    if (s.state === "done") refs.statEta.set("done", "good");
    else if (s.state === "error") refs.statEta.set("failed", "badv");
    else refs.statEta.set(eta === null ? "–" : fmtDuration(eta), eta !== null && eta < 60 ? "good" : "");
  }

  function onResize() { tip.hide(); }


  /* ======================================================================
     (B) PRIMARY SETTINGS STRIP
     ====================================================================== */

  /** Effective tree (defaults deep-merged with the file) for DISPLAY. */
  function effective() {
    return (state.settings && (state.settings.effective || state.settings.values)) || {};
  }

  function buildPrimary() {
    clear(strip);

    refs.fields = {};
    refs.fields.input = pathField({
      label: "Input images",
      path: "project.input.dirs",
      isList: true,
      countImages: true,
      placeholder: "folder of original specimen images",
      title: "Folder(s) of ORIGINAL images. LM3 reads project.input.dirs.",
    });
    refs.fields.runName = textField({
      label: "Project name",
      path: "project.run_name",
      placeholder: "run name",
      title: "Names this run's output folder (<output folder>/<project name>/) and its project database.",
    });
    refs.fields.output = pathField({
      label: "Output folder",
      path: "project.output.dir",
      countImages: false,
      placeholder: "where <run name>/ is written",
      title: "<output>/<run name>/ holds the project database, crops, reports and logs.",
    });
    refs.fields.tmp = pathField({
      label: "Temp files",
      path: "project.output.tmp_dir",
      countImages: false,
      allowAuto: true,
      placeholder: "auto",
      title: "Where ingest writes _tmp_original (the RGB/downscaled working copies). "
           + "\"auto\" puts them beside the run; a path puts them on a faster or larger disk.",
    });

    /* the settings file in use — read-only, but reloadable */
    refs.cfgInput = el("input.mono", { type: "text", readonly: true, value: "…", style: { minWidth: "0" } });
    const reloadBtn = el("button.btn.sm.ghost", { type: "button", title: "Re-read LM3_settings.yaml from disk" }, "↻");
    reloadBtn.addEventListener("click", async () => {
      reloadBtn.disabled = true;
      await loadSettings();
      reloadBtn.disabled = false;
      toast("Settings reloaded", state.settings ? state.settings.yaml_path : "", { kind: "ok", ms: 2600 });
    });
    refs.cfgBadge = el("span.badge", "LM3_settings.yaml");
    const cfgField = el("div.pfield", { style: { flex: "0 1 260px", minWidth: "180px" } },
      el("label", "Settings file"),
      el("div.withbtn", refs.cfgInput, reloadBtn),
      statusRow(refs.cfgBadge),
    );

    /* target device */
    refs.deviceSel = el("select", { style: { minWidth: "0" } });
    refs.deviceSel.addEventListener("change", onDeviceChange);
    refs.deviceBadge = el("span.badge", "compute.devices");
    const devField = el("div.pfield", { style: { flex: "0 1 230px", minWidth: "170px" } },
      el("label", "Device"), refs.deviceSel, statusRow(refs.deviceBadge));

    /* run controls */
    refs.runChip = el("span.chip", { title: "Run name — click to edit it in Settings" }, "demo");
    refs.runChip.addEventListener("click", () => go("settings", { path: "project.run_name" }));
    refs.stateBadge = el("span.badge", "Idle");
    refs.startBtn = el("button.btn.primary", { type: "button" }, "▶  Start LM3");
    refs.stopBtn = el("button.btn.danger", { type: "button", disabled: true }, "■  Stop");
    refs.startBtn.addEventListener("click", startRun);
    refs.stopBtn.addEventListener("click", stopRun);
    // The run controls do NOT live in this strip: they are docked into the tab bar (see
    // mountRunControls) so they stay reachable when the strip is collapsed during a run.
    refs.fieldsWrap = el("div.pfields",
      refs.fields.input.node, refs.fields.runName.node, refs.fields.output.node, refs.fields.tmp.node,
      cfgField, devField);
    append(strip, [refs.fieldsWrap]);
    mountRunControls();
    applyCollapsed(loadCollapsed());
  }

  /* --------------------------------------------------------------------------
     COLLAPSE
     The folder/settings fields matter when SETTING UP a run and are dead weight while one is in
     flight, so the strip folds into a single "Main settings" disclosure and gives the couple of
     hundred pixels back to the live view. It auto-collapses when a run starts (unless the user has
     explicitly pinned it open) and the choice is remembered.
     ----------------------------------------------------------------------- */
  const COLLAPSE_KEY = "lm3.primaryCollapsed";

  function loadCollapsed() {
    try { return localStorage.getItem(COLLAPSE_KEY) === "1"; } catch (_) { return false; }
  }

  function applyCollapsed(on) {
    state.collapsed = !!on;
    strip.classList.toggle("collapsed", state.collapsed);
    if (refs.collapseBtn) {
      refs.collapseBtn.classList.toggle("on", !state.collapsed);
      refs.collapseBtn.setAttribute("aria-expanded", String(!state.collapsed));
      refs.collapseBtn.title = state.collapsed
        ? "Show the main settings (input, project name, output, device)"
        : "Hide the main settings";
    }
    try { localStorage.setItem(COLLAPSE_KEY, state.collapsed ? "1" : "0"); } catch (_) { /* private mode */ }
  }

  function toggleCollapsed() {
    state.userPinnedOpen = state.collapsed;    // expanding by hand pins it open for this run
    applyCollapsed(!state.collapsed);
  }

  /** Dock "Main settings", the run chip and Start/Stop/Close into the tab bar. */
  function mountRunControls() {
    const tabbar = document.querySelector(".tabbar");
    if (!tabbar || tabbar.querySelector(".runctl")) return;

    refs.collapseBtn = el("button.btn.sm.ghost.disc.on", {
      type: "button", "aria-expanded": "true",
      onclick: toggleCollapsed,
    }, el("span.caret", "▾"), el("span", "Main settings"));

    refs.newRunBtn = el("button.btn.sm.ghost", {
      type: "button",
      title: "Clear the GUI and set up a new run",
      onclick: startNewRun,
    }, "✚  New run");

    refs.closeBtn = el("button.btn.sm.danger.ghost", {
      type: "button",
      title: "Stop any running LM3 job and close the app",
      onclick: closeApp,
    }, "✕  Close");

    const ctl = el("div.runctl",
      refs.collapseBtn,
      el("span.sep"),
      refs.runChip, refs.stateBadge, refs.startBtn, refs.stopBtn);

    // sits at the far right of the tab bar, beside the connection indicator
    const conn = tabbar.querySelector("#connstate");
    if (conn) tabbar.insertBefore(ctl, conn);
    else tabbar.appendChild(ctl);

    // Close lives in the very top-right corner, after the live/connection dot -- the
    // conventional place to quit a window, and clear of the run controls it must not be
    // mistaken for.
    const topstats = document.querySelector(".topstats");
    if (topstats) append(topstats, [refs.newRunBtn, refs.closeBtn]);
  }

  /**
   * Clear the GUI down to "nothing has run yet" and set up for a new run.
   *
   * The reset is CLIENT-SIDE and deliberately sticky. When no job is active the
   * server discovers the newest run off disk and keeps reporting it, so simply
   * blanking the bar would be undone by the next status frame ~4s later. Instead
   * `state.newRun` detaches the top bar from the feed until a snapshot actually
   * says "running" -- which is the moment a NEW run exists.
   *
   * Settings are deliberately untouched: a new run is nearly always the same
   * pipeline configuration pointed at new input.
   */
  async function startNewRun() {
    const running = state.snapshot && state.snapshot.state === "running";
    const ok = await confirmModal({
      title: "Start a new run",
      message: running
        ? "Stop the running LM3 job and set up a new run?"
        : "Clear the GUI and set up a new run?",
      detail: running
        ? "The current run is checkpointed before it stops, so it stays resumable — retype its "
          + "name and press Start LM3 to continue it. Your settings are not changed."
        : "The stage bar, console, results view and postprocessing history are cleared, and the "
          + "project name is emptied so the new run cannot resume the old one. Your settings are "
          + "not changed.",
      confirmLabel: running ? "Stop and start new" : "Start new",
      danger: running,
    });
    if (!ok) return;

    if (running) {
      refs.newRunBtn.disabled = true;
      refs.newRunBtn.textContent = "Stopping…";
      try {
        await api.stopRun();
        for (let i = 0; i < 20; i += 1) {
          const st = await api.getStatus().catch(() => null);
          if (!st || st.state !== "running") break;
          await new Promise((r) => setTimeout(r, 500));
        }
      } catch (err) {
        toast("Could not stop the run", String((err && err.message) || err), { kind: "bad" });
      } finally {
        refs.newRunBtn.disabled = false;
        refs.newRunBtn.textContent = "";
        append(refs.newRunBtn, ["✚  New run"]);
      }
    }

    resetForNewRun({ stopped: running });
  }

  /**
   * The reset itself. Shared by the "New run" button and by opening the app (see openFresh), so
   * the two cannot drift apart -- "as if you had pressed New run" has to keep meaning that.
   *
   * Settings are deliberately untouched, and the emptied name is NOT written to the yaml: the
   * field is blanked in the GUI only, and startRun() refuses an empty name, so nothing can start
   * under the old project by accident.
   */
  function resetForNewRun({ stopped = false, announce = true } = {}) {
    state.newRun = true;
    state.snapshot = null;
    state.run = null;
    state.busy = null;
    renderStatus();

    // Empty the run name so Start cannot silently resume the previous project.
    const nameField = refs.fields && refs.fields.runName;
    if (nameField) {
      setFieldValue(nameField, "");
      checkField(nameField);
      applyCollapsed(false);            // the strip folds away during a run; unfold it to type
      if (nameField.input) { nameField.input.focus(); }
    }
    // applySettings() only repaints the chip when settings are reloaded, so set it
    // here too or the tab bar keeps advertising the project we just walked away from.
    if (refs.runChip) refs.runChip.textContent = "unnamed";

    // Each tab owns its own history; they clear themselves on this event.
    dispatch("lm3:newrun", { stopped });
    if (announce) toast("Ready for a new run", "Name the project, then press Start LM3.", { kind: "ok" });
  }

  /**
   * Opening the app IS starting a new run, so it opens in the state the New run button produces:
   * no project name, nothing on the stage bar.
   *
   * Without this the name box comes up holding the LAST project's name (it lives in the settings
   * yaml, which a run writes), one press of Start away from silently resuming or overwriting it.
   *
   * The exception is a job still running headless -- the GUI is the only way to watch it, so a
   * live run is shown, not blanked. `state.newRun` would recover on its own if a running frame
   * arrived a moment later, but the emptied NAME would not, so this asks first rather than
   * blanking optimistically and hoping.
   */
  async function openFresh() {
    let rec = null;
    try {
      rec = await api.get("/v1/run/active", { timeout: 10000 });
    } catch (_) {
      /* no metrics_api, or offline: fall through and open clean, which is the safe default */
    }
    if (rec && rec.active) {
      offerExistingProject();          // (bails by itself while something is running)
      return;
    }
    resetForNewRun({ announce: false });   // no toast: nothing happened, this is just how it opens
  }

  /** Stop a running LM3 job (and WAIT for it) before quitting the desktop shell. */
  async function closeApp() {
    const running = state.snapshot && state.snapshot.state === "running";
    // confirmModal, not window.confirm(): a native modal blocks this page's SSE handlers.
    const ok = await confirmModal({
      title: "Close LeafMachine3",
      message: running ? "Stop the running LM3 job and close the app?" : "Close LeafMachine3?",
      detail: running
        ? "The run is checkpointed before it stops, so it stays resumable — reopen and press "
          + "Start LM3 to continue from where it left off."
        : "No LM3 job is running.",
      confirmLabel: running ? "Stop and close" : "Close",
      danger: true,
    });
    if (!ok) return;

    refs.closeBtn.disabled = true;
    refs.closeBtn.textContent = running ? "Stopping…" : "Closing…";
    if (running) {
      try {
        await api.stopRun();
        // give LM3 a moment to checkpoint the ledger so the run stays resumable
        for (let i = 0; i < 20; i += 1) {
          const s = await api.getStatus().catch(() => null);
          if (!s || s.state !== "running") break;
          await new Promise((r) => setTimeout(r, 500));
        }
      } catch (err) {
        toast("Could not stop the run", String((err && err.message) || err), { kind: "bad" });
      }
    }
    const desktop = window.lm3desktop;
    if (desktop && typeof desktop.quit === "function") desktop.quit({ running });
    else window.close();          // plain browser: best effort
  }

  /* The micro-row under a path field. .pfield has no published sub-label rule,
     so the layout is inline; every color still comes from a token. */
  function statusRow(...children) {
    return el("div", {
      style: { display: "flex", alignItems: "center", gap: "6px", minHeight: "18px", flexWrap: "wrap" },
    }, children);
  }

  /**
   * One editable folder field: input + Browse, live validation on the way in,
   * PUT /v1/settings on commit.
   */
  function pathField(cfg) {
    const input = el("input.mono", {
      type: "text", spellcheck: "false", placeholder: cfg.placeholder || "",
      title: cfg.title || "", style: { minWidth: "0" },
    });
    const browseBtn = el("button.btn.sm", { type: "button", title: "Browse for a folder" }, "Browse");
    const badge = el("span.badge", "…");
    const extra = el("span");
    const node = el("div.pfield",
      el("label", cfg.label),
      el("div.withbtn", input, browseBtn),
      statusRow(badge, extra));

    const f = {
      cfg, node, input, badge, extra, browseBtn,
      saved: "",                    // what is on disk right now
      lastChecked: null,
      valid: true,
    };

    const check = debounce(() => validateField(f), 380);
    input.addEventListener("input", () => { setBadge(f, "", "checking…"); check(); });
    input.addEventListener("blur", () => commitField(f));
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); input.blur(); }
      if (ev.key === "Escape") { input.value = f.saved; setInputError(f, false); validateField(f); }
    });
    browseBtn.addEventListener("click", async () => {
      const picked = await browseFolder({
        title: `Choose the ${cfg.label.toLowerCase()}`,
        start: input.value.trim() && input.value.trim() !== "auto" ? input.value.trim() : null,
        countImages: !!cfg.countImages,
      });
      if (!picked) return;
      input.value = picked;
      await validateField(f);
      await commitField(f);
    });
    return f;
  }

  /**
   * A plain (non-path) text setting in the primary strip -- currently the project name.
   *
   * It sits next to the input folder because those two together are what actually identify a run:
   * WHICH images, and WHAT the output folder is called. Committing writes project.run_name straight
   * back to LM3_settings.yaml, the same way the path fields do.
   */
  function textField(cfg) {
    const input = el("input.mono", {
      type: "text", spellcheck: "false", placeholder: cfg.placeholder || "",
      title: cfg.title || "", style: { minWidth: "0" },
    });
    const badge = el("span.badge", "…");
    const extra = el("span");
    const node = el("div.pfield",
      el("label", cfg.label),
      el("div.withbtn", input),
      statusRow(badge, extra));

    const f = { cfg, node, input, badge, extra, browseBtn: null, saved: "", lastChecked: null, valid: true };

    const mark = () => {
      const v = input.value.trim();
      // LM3 uses the run name as a folder name, so keep it to filesystem-safe characters.
      const bad = /[/\\:*?"<>|]/.test(v);
      f.valid = !!v && !bad;
      setInputError(f, !f.valid);
      if (!v) setBadge(f, "warn", "required");
      else if (bad) setBadge(f, "bad", "invalid characters");
      else setBadge(f, v === f.saved ? "ok" : "", v === f.saved ? "saved" : "unsaved");
    };
    /* A project name is NOT a path. Without its own validator it would fall through to
       validateField(), which POSTs the name to /v1/settings/browse and then badges a perfectly
       good run name "will be created" because no folder called "demo" exists beside the app. */
    f.validate = mark;
    input.addEventListener("input", mark);
    input.addEventListener("blur", () => { if (f.valid) commitField(f); else input.value = f.saved; mark(); });
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); input.blur(); }
      if (ev.key === "Escape") { input.value = f.saved; mark(); }
    });
    return f;
  }

  function setBadge(f, cls, text, title) {
    f.badge.className = `badge${cls ? ` ${cls}` : ""}`;
    f.badge.textContent = text;
    if (title) f.badge.title = title; else f.badge.removeAttribute("title");
  }

  function setInputError(f, bad) {
    f.input.style.borderColor = bad ? "var(--bad)" : "";
  }

  /** Re-check a field with the validator its KIND uses (paths hit the server; plain text does not). */
  function checkField(f) {
    return f.validate ? f.validate() : validateField(f);
  }

  /** Ask the server what that path actually is. Never throws. */
  async function validateField(f) {
    const raw = f.input.value.trim();
    if (!raw) {
      f.valid = false;
      setBadge(f, "bad", "required");
      return;
    }
    if (f.cfg.allowAuto && raw.toLowerCase() === "auto") {
      f.valid = true;
      setInputError(f, false);
      setBadge(f, "info", "auto", "Temp files go to <output>/<run name>/_tmp_original");
      return;
    }
    if (!state.settingsApi) { setBadge(f, "warn", "unchecked"); return; }
    try {
      const r = await api.post("/v1/settings/browse", {
        path: raw, count_images: !!f.cfg.countImages, max_entries: 1,
      }, { timeout: 20000 });
      f.lastChecked = r;
      if (r.error) {
        f.valid = false; setInputError(f, true); setBadge(f, "bad", "unreadable", r.error);
      } else if (r.fell_back || r.exists === false) {
        /* not an error for an output folder — LM3 creates it */
        f.valid = !f.cfg.countImages;
        setInputError(f, !f.valid);
        setBadge(f, f.valid ? "warn" : "bad",
          f.valid ? "will be created" : "not found",
          `Nearest existing folder: ${r.path}`);
      } else if (r.is_dir === false) {
        f.valid = false; setInputError(f, true); setBadge(f, "bad", "not a folder");
      } else if (r.readable === false) {
        f.valid = false; setInputError(f, true); setBadge(f, "bad", "no permission");
      } else if (f.cfg.countImages) {
        const n = Number(r.n_images || 0);
        f.valid = true; setInputError(f, false);
        setBadge(f, n > 0 ? "ok" : "warn",
          n > 0 ? `${fmtNum(n, { digits: 0 })} image${n === 1 ? "" : "s"}` : "no images here",
          n > 0 ? `${r.path}` : `${r.path} — LM3 also searches subfolders when project.input.recursive is on`);
      } else {
        f.valid = true; setInputError(f, false);
        setBadge(f, "ok", "found", r.path);
      }
    } catch (err) {
      setBadge(f, "warn", "unchecked", errText(err));
    }
  }

  /** Write the field back to LM3_settings.yaml if it actually changed. */
  async function commitField(f) {
    const raw = f.input.value.trim();
    if (raw === f.saved) return;
    if (!raw) {
      /* Config.validate treats project.input.dirs: [] as a HARD error and
         refuses the whole write, so an empty box is caught here and the last
         saved value restored — clearing a folder must not be able to leave
         LM3_settings.yaml in a state that cannot be saved again. */
      f.input.value = f.saved;
      await checkField(f);
      toast("That setting cannot be empty", `${f.cfg.label} was restored.`, { kind: "warn", ms: 4200 });
      return;
    }
    const value = f.cfg.isList ? mergeListHead(f, raw) : raw;
    const ok = await saveSetting(f.cfg.path, value, f);
    if (ok) {
      f.saved = raw;
      toast("Saved", `${f.cfg.label} → ${fmtPath(raw, 3)}`, { kind: "ok", ms: 2600 });
      // The output folder + project name together ARE the project. Whenever either moves, check
      // whether they now point at a project that already exists on disk.
      if (f.cfg.path === "project.run_name" || f.cfg.path === "project.output.dir") {
        offerExistingProject();
      }
    }
  }

  /* --------------------------------------------------------------------------
     EXISTING PROJECT
     <output folder>/<project name>/ may already hold a finished or partial LM3 project. Silently
     pointing a "new" run at it would resume it -- which is sometimes exactly what you want and
     sometimes a surprise -- so ask, and offer a free name if it is meant to be new.
     ----------------------------------------------------------------------- */
  let lastOfferedRun = null;

  async function offerExistingProject({ force = false } = {}) {
    const eff = effective();
    const name = String(deepGet(eff, "project.run_name", "") || "").trim();
    if (!name) return;
    if (!force && name === lastOfferedRun) return;         // do not nag about the same name twice
    lastOfferedRun = name;

    let rec = null;
    try {
      rec = await api.get(`/v1/runs/${encodeURIComponent(name)}`);
    } catch (_) {
      return;                                              // 404 -> nothing there, nothing to ask
    }
    if (!rec || !(rec.has_db || rec.has_reports)) return;
    if (state.snapshot && state.snapshot.state === "running") return;   // mid-run: not the moment

    showExistingProjectDialog(rec, name);
  }

  function showExistingProjectDialog(rec, name) {
    const done = rec.modules_done ?? "?";
    const total = rec.modules_total ?? 16;
    const imgs = rec.n_images ?? rec.n_images_done;
    const suggestion = `${name}_2`;

    const nameInput = el("input.mono", { type: "text", value: suggestion, spellcheck: "false" });

    // Uses the shared openModal(): it owns the fixed centered .backdrop > .modal structure, the
    // Escape / click-outside handling and the close button, so this dialog behaves exactly like
    // the Stop confirmation and the folder browser.
    openModal({
      title: "This project already exists",
      width: 580,
      build: (panel, close) => {
        panel.appendChild(el("div.panel-bd",
          el("p", { style: { margin: "0 0 6px" } }, `A project named "${name}" is already here:`),
          el("p.mono.dim", { style: { margin: "0 0 12px", wordBreak: "break-all" } }, rec.path || ""),
          el("ul.kv", { style: { margin: "0 0 12px" } },
            el("li", el("span.k", "state"), el("span.v", String(rec.state || "unknown"))),
            el("li", el("span.k", "modules finished"), el("span.v", `${done} / ${total}`)),
            el("li", el("span.k", "specimens"), el("span.v", imgs == null ? "–" : String(imgs)))),
          el("p", { style: { margin: "0 0 12px", fontSize: "12.8px", color: "var(--mute)", lineHeight: "1.55" } },
            "Open it to view or continue that project — LM3 resumes where it left off and will not "
            + "redo finished modules. If this was meant to be a NEW run, give it a different name "
            + "so the existing project is left untouched."),
          el("div.field", el("label", { for: "lm3-newrun" }, "New project name"), nameInput)));

        const useNew = async () => {
          const v = nameInput.value.trim();
          if (!v) return;
          close(undefined);
          const f = refs.fields.runName;
          f.input.value = v;
          await commitField(f);
        };
        const useExisting = () => {
          close(undefined);
          toast("Working on the existing project", rec.path || name, { kind: "ok" });
        };

        panel.appendChild(el("div.panel-ft",
          { style: { display: "flex", gap: "8px", justifyContent: "flex-end" } },
          el("button.btn.sm.ghost", { type: "button", onclick: () => close(undefined) }, "Cancel"),
          el("button.btn.sm", { type: "button", onclick: useNew }, "Use the new name"),
          el("button.btn.sm.accent", { type: "button", onclick: useExisting }, "Open this project")));

        setTimeout(() => { nameInput.focus(); nameInput.select(); }, 0);
      },
    });
  }

  /** Keep any extra input folders the Settings tab configured. */
  function mergeListHead(f, head) {
    const cur = deepGet(effective(), f.cfg.path, []);
    const rest = Array.isArray(cur) ? cur.slice(1) : [];
    return [head, ...rest];
  }

  /**
   * PUT the full tree with one leaf changed. `if_mtime` is optimistic
   * concurrency: if a second window (or the Settings tab) wrote the file since
   * we loaded it, the server 409s instead of clobbering, and we offer the
   * overwrite explicitly rather than deciding for the user.
   */
  async function saveSetting(path, value, field = null, { force = false } = {}) {
    if (!state.settings) return false;
    if (state.settings.readonly) {
      toast("Settings file is read-only", state.settings.yaml_path, { kind: "warn" });
      return false;
    }
    const values = cloneTree(state.settings.values || {});
    deepSet(values, path, value);
    const body = { values, backup: true };
    if (!force && state.settings.mtime !== null && state.settings.mtime !== undefined) {
      body.if_mtime = state.settings.mtime;
    }
    try {
      const res = await api.put("/v1/settings", body, { timeout: 20000 });
      state.settings = res;
      applySettings();
      if (res.warnings && res.warnings.length) {
        toast("Saved with warnings", res.warnings.slice(0, 2).map((w) => w.msg).join(" · "), { kind: "warn" });
      }
      dispatch("lm3:settings-saved", { path, value, settings: res });
      return true;
    } catch (err) {
      if (err.status === 409) {
        toast("LM3_settings.yaml changed on disk",
          "Another window saved it after this one loaded. Reload to see that version, or overwrite it with your change.",
          { kind: "warn", action: { label: "Overwrite", onClick: () => saveSetting(path, value, field, { force: true }) } });
        await loadSettings();
        return false;
      }
      const detail = err.body && err.body.detail;
      const fieldErrs = detail && detail.field_errors ? detail.field_errors[path] : null;
      if (field) {
        setInputError(field, true);
        setBadge(field, "bad", "rejected", (fieldErrs && fieldErrs.join(" · ")) || errText(err));
      }
      toast("Could not save", errText(err), { kind: "bad", ms: 9000 });
      return false;
    }
  }

  async function onDeviceChange() {
    const key = refs.deviceSel.value;
    const value = keyToDevices(key);
    const prev = deepGet(effective(), "compute.devices", "auto");
    if (JSON.stringify(value) === JSON.stringify(prev)) return;
    const label = refs.deviceSel.selectedOptions[0] ? refs.deviceSel.selectedOptions[0].textContent : key;
    refs.deviceSel.disabled = true;
    const ok = await saveSetting("compute.devices", value);
    refs.deviceSel.disabled = false;
    if (ok) toast("Saved", `Device → ${label}`, { kind: "ok", ms: 2600 });
    else refs.deviceSel.value = devicesToKey(prev);
  }

  /* compute.devices is a dual-type knob: "auto" | "cpu" | [0,1]. The select
     round-trips it through a string key so the DOM stays simple. */
  function devicesToKey(v) {
    if (Array.isArray(v)) return `gpu:${v.join(",")}`;
    if (typeof v === "number") return `gpu:${v}`;
    const s = String(v ?? "auto").toLowerCase();
    return s === "cpu" ? "cpu" : "auto";
  }
  function keyToDevices(key) {
    if (key === "auto" || key === "cpu") return key;
    return key.slice(4).split(",").filter(Boolean).map((n) => Number(n));
  }

  function renderDeviceOptions() {
    const gpus = (state.hardware && state.hardware.gpus) || [];
    const cur = devicesToKey(deepGet(effective(), "compute.devices", "auto"));
    const opts = [
      ["auto", gpus.length ? `Auto — all ${gpus.length} GPU${gpus.length === 1 ? "" : "s"}` : "Auto"],
      ["cpu", "CPU only"],
    ];
    for (const g of gpus) {
      opts.push([`gpu:${g.index}`, `GPU ${g.index} — ${shortGpuName(g.name)}`]);
    }
    if (gpus.length > 1) {
      opts.push([`gpu:${gpus.map((g) => g.index).join(",")}`, `GPUs ${gpus.map((g) => g.index).join(" + ")}`]);
    }
    if (!opts.some(([k]) => k === cur)) opts.push([cur, cur]);

    clear(refs.deviceSel);
    for (const [value, label] of opts) refs.deviceSel.appendChild(el("option", { value }, label));
    refs.deviceSel.value = cur;

    const prov = state.hardware && (state.hardware.provider || state.hardware.raw?.provider);
    refs.deviceBadge.className = `badge${gpus.length ? " ok" : " warn"}`;
    refs.deviceBadge.textContent = gpus.length
      ? `${prov ? String(prov).replace("ExecutionProvider", "") : "cuda"} · ${state.hardware.precision || "fp16"}`
      : "no GPU profile";
    refs.deviceBadge.title = gpus.length
      ? gpus.map((g) => `GPU ${g.index}: ${g.name} (${fmtNum(g.total_vram_mb, { digits: 0 })} MB)`).join("\n")
      : "Run the hardware profiler (LM3 Setup) so LM3 can size each module.";
  }

  function shortGpuName(name) {
    return String(name || "GPU").replace(/^NVIDIA\s+/i, "").replace(/\s+Generation$/i, "");
  }

  /** Push the loaded settings into the strip's controls. */
  function applySettings() {
    const eff = effective();

    const dirs = deepGet(eff, "project.input.dirs", []);
    const head = Array.isArray(dirs) ? (dirs[0] || "") : String(dirs || "");
    setFieldValue(refs.fields.input, head);
    clear(refs.fields.input.extra);
    if (Array.isArray(dirs) && dirs.length > 1) {
      const more = el("span.badge.info", { title: dirs.slice(1).join("\n"), style: { cursor: "pointer" } },
        `+${dirs.length - 1} more`);
      more.addEventListener("click", () => go("settings", { path: "project.input.dirs" }));
      refs.fields.input.extra.appendChild(more);
    }

    setFieldValue(refs.fields.runName, String(deepGet(eff, "project.run_name", "") || ""));
    setFieldValue(refs.fields.output, String(deepGet(eff, "project.output.dir", "") || ""));
    setFieldValue(refs.fields.tmp, String(deepGet(eff, "project.output.tmp_dir", "auto") || "auto"));

    refs.cfgInput.value = (state.settings && state.settings.yaml_path) || "";
    refs.cfgInput.title = refs.cfgInput.value;
    const ro = !!(state.settings && state.settings.readonly);
    refs.cfgBadge.className = `badge${ro ? " warn" : (state.settings && state.settings.exists) ? " ok" : " bad"}`;
    refs.cfgBadge.textContent = ro ? "read-only" : (state.settings && state.settings.exists) ? "loaded" : "missing";
    refs.cfgBadge.title = state.settings && state.settings.backups && state.settings.backups.length
      ? `${state.settings.backups.length} backup(s) kept beside it` : "";

    for (const f of Object.values(refs.fields)) {
      f.input.disabled = ro;
      if (f.browseBtn) f.browseBtn.disabled = ro;   // the project-name field has no Browse button
    }
    refs.deviceSel.disabled = ro;

    refs.runChip.textContent = state.newRun
      ? "unnamed"                       // the reset emptied the field; do not show the old name
      : String(deepGet(eff, "project.run_name", "") || "unnamed");
    renderDeviceOptions();
  }

  function setFieldValue(f, value) {
    f.saved = value;
    /* never stomp on something the user is mid-edit */
    if (document.activeElement !== f.input) f.input.value = value;
    setInputError(f, false);
    checkField(f);
  }

  /** Commit anything typed but not yet blurred — called before Start. */
  async function flushPending() {
    for (const f of Object.values(refs.fields || {})) {
      if (f.input.value.trim() !== f.saved) await commitField(f);
    }
    // ...and the Settings tab, which is where every module knob lives. It keeps edits in a draft
    // until you press Save, while Start is docked in the TAB BAR -- so changing a module setting
    // and pressing Start while looking straight at it is the normal way to use this. Without
    // this the run reads the old yaml: the form says one thing, the run does another, and
    // nothing on screen explains the difference. (startRun's own comment has always promised
    // that flushPending leaves the yaml current for the module settings.)
    if (typeof opts.flushSettings === "function") {
      const ok = await opts.flushSettings();
      if (ok === false) throw new Error("unsaved settings changes could not be written");
    }
  }


  /* ======================================================================
     RUN CONTROL
     ====================================================================== */

  function renderRunControls() {
    // pollRun() keeps refreshing /v1/run/active, which still describes the run we
    // just walked away from -- so after "New run" the chip reads idle until either
    // Start is pressed or a real run appears.
    const r = (state.newRun && state.busy !== "starting") ? null : state.run;
    const s = state.snapshot;
    const active = !!(r && r.active) || state.busy === "starting";
    let key = (r && r.state) || "idle";
    if (state.busy === "starting") key = "running";
    if (state.busy === "stopping") key = "stopping";

    let label = RUN_STATE_TEXT[key] || key;
    let cls = RUN_STATE_BADGE[key] || "";
    if (state.busy === "starting") label = "Starting";
    else if (state.busy === "stopping") label = "Stopping";
    else if (active && s && s.stale) { label = "Stalled"; cls = "warn"; }
    else if (r && r.adopted && active) label = "Running (adopted)";

    refs.stateBadge.className = `badge dotd${cls ? ` ${cls}` : ""}`;
    refs.stateBadge.textContent = label;
    refs.stateBadge.title = runTitle(r);

    refs.startBtn.disabled = active || !state.runApi || !!state.busy;
    refs.stopBtn.disabled = !active || !state.runApi || state.busy === "stopping";
    refs.startBtn.textContent = active ? "▶  Running" : "▶  Start LM3";
    if (!state.runApi) {
      refs.startBtn.title = "Run control is not mounted on this LM3 server (POST /v1/run/start is missing).";
      refs.stopBtn.title = refs.startBtn.title;
    } else {
      refs.startBtn.title = "Launch LM3 with the settings shown here";
      refs.stopBtn.title = "Stop the active run (it stays resumable)";
    }
  }

  function runTitle(r) {
    if (!r) return "";
    const bits = [];
    if (r.run_name) bits.push(`run ${r.run_name}`);
    if (r.pid) bits.push(`pid ${r.pid}`);
    if (r.returncode !== null && r.returncode !== undefined) bits.push(`exit ${r.returncode}`);
    if (r.error) bits.push(r.error);
    if (r.run_dir) bits.push(r.run_dir);
    return bits.join("\n");
  }

  async function startRun() {
    if (state.busy) return;
    // "New run" empties the name field but does NOT write the empty value to
    // LM3_settings.yaml, so flushPending() would see nothing to commit and the
    // run would quietly start under the PREVIOUS project name. Refuse instead.
    const nameField = refs.fields && refs.fields.runName;
    if (nameField && !String(nameField.input.value || "").trim()) {
      setInputError(nameField, true);
      applyCollapsed(false);
      nameField.input.focus();
      toast("Name the project first", "A run writes to <output folder>/<project name>/, so it "
        + "needs a name before it can start.", { kind: "warn" });
      return;
    }
    state.newRun = false;   // pressing Start re-attaches the top bar to the live feed
    state.busy = "starting";
    renderRunControls();
    try {
      await flushPending();
      /* POST /v1/run/start takes only input/output/restart overrides; the run
         name, temp dir and module toggles come from LM3_settings.yaml, which
         flushPending() has just made sure is current. */
      const rec = await api.post("/v1/run/start", {}, { timeout: 60000 });
      applyRun(rec);   /* re-pins the status stream to rec.db_path */
      toast("LM3 started", `Run ${rec.run_name || ""} — pid ${rec.pid || "?"}`, { kind: "ok" });
    } catch (err) {
      if (err.status === 409) {
        toast("A run is already active", "Stop it before starting another.", { kind: "warn" });
      } else if (err.isNotFound) {
        state.runApi = false;
        toast("Run control unavailable", "This LM3 server does not expose POST /v1/run/start.", { kind: "bad" });
      } else {
        toast("LM3 did not start", errText(err), { kind: "bad", ms: 12000 });
      }
    } finally {
      state.busy = null;
      renderRunControls();
      pollRun();
    }
  }

  async function stopRun() {
    if (state.busy) return;
    const yes = await confirmModal({
      title: "Stop LM3?",
      message: "Stop the active run now?",
      detail: "LM3 is resumable — every module that was mid-flight rolls back to pending, and starting "
            + "again picks up exactly where this run left off. Finished modules are not repeated.",
      confirmLabel: "Stop LM3",
      danger: true,
    });
    if (!yes) return;
    state.busy = "stopping";
    renderRunControls();
    try {
      const rec = await api.post("/v1/run/stop", { grace_s: 10 }, { timeout: 40000 });
      applyRun(rec);
      toast("LM3 stopped", "The run is resumable — press Start to continue it.", { kind: "warn" });
    } catch (err) {
      if (err.status === 409) toast("Nothing to stop", "No LM3 run is active.", { kind: "warn" });
      else toast("Could not stop LM3", errText(err), { kind: "bad" });
    } finally {
      state.busy = null;
      renderRunControls();
      pollRun();
    }
  }

  function applyRun(rec) {
    const before = state.run;
    state.run = rec;
    const moved = !before
      || before.state !== rec.state
      || before.pid !== rec.pid
      || before.run_name !== rec.run_name
      || before.returncode !== rec.returncode;
    if (moved) {
      dispatch("lm3:run", rec);
      if (before && before.active && !rec.active) {
        if (rec.state === "error") {
          toast("LM3 failed", rec.error || `exit code ${rec.returncode}`, { kind: "bad", ms: 12000 });
        } else if (!rec.stopped_by_user) {
          toast("LM3 complete", `Run ${rec.run_name || ""} finished`, { kind: "ok" });
        }
      }
      /* Follow the launcher's own resolution of the ledger path — it applied
         Config.load with the same overrides machine3 will, including the
         `tmp_dir: auto` rule, so it beats re-deriving <output>/<run_name>.
         Only while the run is ACTIVE: once it ends we stay pinned to the run
         the user just watched instead of snapping to whatever discovery finds. */
      if (rec.active && rec.db_path && rec.db_path !== state.statusDb) connectStatus(rec.db_path);
    }
    renderRunControls();
  }


  /* ======================================================================
     DATA LOADING
     ====================================================================== */

  async function loadHealth() {
    try {
      const h = await api.health();
      if (h && h.version) refs.ver.textContent = `v${h.version}`;
      if (h && h.provider) refs.ver.title = `ONNX Runtime provider: ${h.provider}`;
    } catch { /* the version badge is decoration; the app works without it */ }
  }

  async function loadSettings() {
    try {
      state.settings = await api.get("/v1/settings", { timeout: 20000 });
      state.settingsApi = true;
      applySettings();
    } catch (err) {
      if (err.isNotFound) {
        state.settingsApi = false;
        for (const f of Object.values(refs.fields || {})) {
          f.input.disabled = true;
          if (f.browseBtn) f.browseBtn.disabled = true;   // project-name field has none
          setBadge(f, "bad", "no settings API");
        }
        refs.cfgBadge.className = "badge bad";
        refs.cfgBadge.textContent = "unavailable";
      } else if (!err.isOffline) {
        toast("Could not read LM3_settings.yaml", errText(err), { kind: "bad" });
      }
    }
  }

  async function loadHardware() {
    /* metrics_api's profile is the richer view; app.py's /v1/hardware returns
       the raw hardware_settings.yaml, which carries the same `gpus` list. */
    try {
      state.hardware = await api.get("/v1/hardware/profile", { timeout: 15000 });
      if (state.hardware && state.hardware.available === false) throw new Error("no profile");
    } catch {
      try {
        const raw = await api.getHardware();
        state.hardware = { gpus: (raw && raw.gpus) || [], provider: raw && raw.provider, precision: raw && raw.precision, raw };
      } catch { state.hardware = { gpus: [] }; }
    }
    if (refs.deviceSel) renderDeviceOptions();
  }

  /**
   * Subscribe to the status stream. `db` pins it to one project ledger — worth
   * doing the moment the launcher tells us where the ledger IS, because
   * otherwise progress_api has to discover the run off disk and a brand-new
   * run is invisible until its first write.
   */
  function connectStatus(db) {
    if (state.destroyed) return;
    if (state.closeStatus) { state.closeStatus(); state.closeStatus = null; }
    state.statusDb = db || null;
    setConn("wait");
    state.closeStatus = api.sse("/v1/status/stream", {
      params: db ? { db } : undefined,
      onOpen: () => setConn("up"),
      onError: () => setConn("down"),
      onMessage: (frame) => {
        if (!frame || frame.type !== "status" || !frame.snapshot) return;
        setConn("up");
        // After "New run" -- and on launch -- the feed still describes the PREVIOUS run (the
        // server discovers it off disk). Ignore it until something is actually running.
        //
        // "running" alone is not enough: a run that was killed mid-flight leaves its ledger
        // saying `running` forever, so that frame re-attaches instantly and undoes the reset.
        // `stale` is the server's own "nothing has written to this ledger lately" flag -- a live
        // run keeps clearing it, an abandoned one never does. Pressing Start does not rely on
        // any of this; startRun() clears state.newRun itself.
        if (state.newRun) {
          if (frame.snapshot.state !== "running" || frame.snapshot.stale) return;
          state.newRun = false;
        }
        state.snapshot = frame.snapshot;
        state.snapshotAt = performance.now();
        onRunStateEdge(frame.snapshot);
        renderStatus();
        dispatch("lm3:status", frame.snapshot);
      },
    });
  }

  /**
   * Fold the main-settings strip away the moment a run begins, and unfold it again when the run
   * ends -- the fields are for setting a run up, not for watching one. Only fires on the EDGE, so a
   * user who deliberately expands mid-run is left alone until the next run.
   */
  function onRunStateEdge(snap) {
    const running = !!snap && snap.state === "running";
    if (running && !state.wasRunning) {
      state.userPinnedOpen = false;
      applyCollapsed(true);
    } else if (!running && state.wasRunning && !state.userPinnedOpen) {
      applyCollapsed(false);
    }
    state.wasRunning = running;
  }

  function setConn(kind) {
    if (state.conn === kind) return;
    state.conn = kind;
    refs.conn.className = `conn ${kind}`;
    refs.conn.textContent = kind === "up" ? "live" : kind === "wait" ? "connecting" : "offline";
  }

  /** Poll the run record; fast while something is in flight, slow when idle. */
  async function pollRun() {
    if (state.destroyed) return;
    if (state.runTimer) { clearTimeout(state.runTimer); state.runTimer = null; }
    if (!state.runApi) return;
    try {
      const rec = await api.get("/v1/run/active", { timeout: 10000 });
      applyRun(rec);
    } catch (err) {
      if (err.isNotFound) {
        /* metrics_api is not mounted — say so on the buttons rather than
           retrying a route that will never exist */
        state.runApi = false;
        renderRunControls();
        return;
      }
      /* offline: the connection dot already says so — keep polling */
    }
    if (state.destroyed) return;
    const active = !!(state.run && state.run.active) || !!state.busy;
    state.runTimer = setTimeout(() => pollRun(), active ? RUN_POLL_ACTIVE_MS : RUN_POLL_IDLE_MS);
  }


  /* ======================================================================
     NAVIGATION + TEARDOWN
     ====================================================================== */

  function go(tab, extra = {}) {
    const detail = { tab, ...extra };
    if (opts.onNavigate) { try { opts.onNavigate(detail); } catch (err) { console.error("[lm3] onNavigate threw", err); } }
    dispatch("lm3:navigate", detail);
  }

  function dispatch(name, detail) {
    document.dispatchEvent(new CustomEvent(name, { detail, bubbles: true }));
  }

  function destroy() {
    state.destroyed = true;
    if (state.closeStatus) state.closeStatus();
    if (state.runTimer) clearTimeout(state.runTimer);
    if (state.tickTimer) clearInterval(state.tickTimer);
    window.removeEventListener("resize", onResize);
    tip.destroy();
    clear(bar);
    clear(strip);
  }

  /* ------------------------------------------------------------ controller */
  return {
    el: { topbar: bar, primary: strip },
    get snapshot() { return state.snapshot; },
    get run() { return state.run; },
    get settings() { return state.settings; },
    get modules() { return moduleList(); },
    /**
     * Re-read settings, hardware and the run record, and re-point the status
     * stream. Also the escape hatch for a run started OUTSIDE the app (from
     * the machine3 CLI): dropping the `?db=` pin hands the choice back to
     * progress_api's on-disk discovery, which finds the newest run.
     */
    async refresh() {
      connectStatus(state.run && state.run.active ? state.run.db_path : null);
      await Promise.all([loadSettings(), loadHardware()]);
      await pollRun();
      renderStatus();
    },
    start: startRun,
    stop: stopRun,
    focusModule,
    toast,
    destroy,
  };
}


/* ==========================================================================
   MOUNT RESOLUTION
   The .app grid declares "topbar" and "primary" as separate areas, so the two
   pieces are siblings, not nested. We adopt whatever index.html already has
   and only create what is missing — in the order the grid expects.
   ========================================================================== */
function resolveMounts(root) {
  const host = typeof root === "string" ? document.querySelector(root) : root;
  const scope = host || document.querySelector(".app") || document.body;

  let bar = null;
  let strip = null;

  if (scope.classList && scope.classList.contains("topbar")) {
    bar = scope;
    const parent = bar.parentNode || document.body;
    strip = parent.querySelector(".primary");
  } else {
    bar = scope.querySelector(":scope > .topbar") || scope.querySelector(".topbar");
    strip = scope.querySelector(":scope > .primary") || scope.querySelector(".primary");
  }

  if (!bar) {
    bar = el("header.topbar");
    scope.insertBefore(bar, scope.firstChild);
  }
  if (!strip) {
    strip = el("section.primary");
    const parent = bar.parentNode || scope;
    parent.insertBefore(strip, bar.nextSibling);
  }
  return { bar, strip };
}

export default initTopBar;
