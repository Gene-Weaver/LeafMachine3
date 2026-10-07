/* ==========================================================================
   LM3 — Results tab
   --------------------------------------------------------------------------
   Two inspectors over a finished (or in-flight) LM3 run:

     MEDIA     every product folder LM3 wrote, as a category sidebar plus a
               lazy-loading thumbnail grid, with a viewer that knows the
               difference between a sheet overlay, the timing report, a CSV
               and an .h5 coordinate blob.
     DATABASE  every table AND view in the run's project SQLite, paginated
               and sorted SERVER-side so a 740k-row ledger opens as fast as a
               19-row one.

   Backend contract (results_api.py, prefix /v1/runs) — verified live:
     GET /v1/runs                      -> {t, n, roots, runs:[summary]}
     GET /{run}/media                  -> {run, groups, categories:[...]}
     GET /{run}/media/list             -> one page of {files:[...]}
     GET /{run}/file?path=             -> raw bytes, correct content-type
     GET /{run}/thumb?path=&w=         -> cached JPEG (w SNAPS UP a ladder)
     DELETE /{run}/thumbs              -> drop the thumbnail cache
     GET /{run}/db/tables              -> schema of every table and view
     GET /{run}/db/table/{name}        -> {columns, rows:[[...]], n_total, ...}

   TWO CONTRACT DETAILS THAT SHAPE THIS WHOLE FILE:
     1. db rows are ARRAYS aligned to `columns`, not objects. That keeps a
        3,379 x 44 page small on the wire; we zip them at render time.
     2. <img> cannot send an Authorization header, so every thumb/file URL
        carries ?token= instead. api.url() + api.token builds those.
   ========================================================================== */

import {
  api, el, clear, append, esc,
  fmtBytes, fmtNum, fmtTime, debounce,
} from "../api.js";


/* ------------------------------------------------------------------ state -- */

/** Page sizes offered for the media grid and the DB grid. */
const MEDIA_PAGE_SIZES = [60, 120, 240, 480];
const DB_PAGE_SIZES = [50, 100, 250, 500, 1000];

/** Thumbnail widths. These mirror the server's snap ladder so we never ask for
    a width it will silently round up — the tile CSS and the JPEG then agree. */
const TILE_SIZES = {
  sm: { grid: "mediagrid sm", w: 128 },
  md: { grid: "mediagrid", w: 256 },
  lg: { grid: "mediagrid lg", w: 512 },
};

const S = {
  runs: [],
  run: null,             // the selected run summary
  view: "media",         // "media" | "db"

  media: null,           // GET /media payload for S.run
  category: null,        // category id currently listed
  includeWorking: false, // show the "working" intermediate folders
  page: { offset: 0, limit: 120, sort: "name", desc: false, q: "" },
  listing: null,         // GET /media/list payload
  tile: "md",
  thumbNonce: 0,         // bumped by "Rebuild previews" to defeat the browser cache

  tables: null,          // GET /db/tables payload
  table: null,           // selected table/view name
  db: { offset: 0, limit: 100, order_by: null, desc: false, where: "" },
  rows: null,            // GET /db/table payload

  loaded: false,         // initResults() has run
  available: false,      // hasResults() answer
};

/** Live DOM references, filled by build(). */
const D = {};

let pollTimer = null;
let probeStarted = false;
let thumbObserver = null;


/* ------------------------------------------------------------- primitives -- */

/** A run has something worth showing once it wrote reports or a ledger. */
function runHasOutput(r) {
  return Boolean(r && (r.has_reports || r.has_db
    || r.state === "done" || r.state === "error" || r.state === "partial"));
}

/**
 * Normalize one run summary.
 *
 * TWO backend modules publish GET /v1/runs — progress_api's run list and
 * results_api's — and app.py mounts progress_api FIRST, so that is the payload
 * this tab actually receives. It names its fields differently (`run_name`, no
 * `id`, `db_path` instead of `has_db`), which left every downstream call
 * fetching /v1/runs/undefined/... Both shapes are accepted here instead of
 * betting on the mount order; results_api's `{run}` parameter resolves a run
 * NAME as happily as its own hashed id, so the name is a safe fallback key.
 */
function normalizeRun(r) {
  const name = r.name || r.run_name || "";
  return {
    ...r,
    name,
    id: r.id || name,
    has_db: r.has_db === undefined ? Boolean(r.db_path) : r.has_db,
    started_at: r.started_at || r.started || null,
    finished_at: r.finished_at || r.finished || null,
  };
}

/** Badge class for a run/table state word. */
function stateBadge(state) {
  const cls = { done: "ok", running: "info", error: "bad", partial: "warn" }[state] || "";
  return el(`span.badge.${cls || "dotd"}`, state || "unknown");
}

/**
 * Shorten a long filename from the MIDDLE.
 * LM3 product names are `<specimen>__<tag>__<x1>_<y1>_<x2>_<y2>.png`; the tail
 * carries the bounding box that distinguishes one leaf from the next, so a
 * plain end-ellipsis would render every tile of a specimen identically.
 */
function midEllipsis(name, max = 40) {
  const s = String(name ?? "");
  if (s.length <= max) return s;
  const tail = Math.max(12, Math.floor(max * 0.62));
  const head = max - tail - 1;
  return `${s.slice(0, head)}…${s.slice(-tail)}`;
}

/** Copy text to the clipboard; falls back to a hidden textarea off http. */
async function copyText(text, what = "path") {
  const s = String(text ?? "");
  try {
    await navigator.clipboard.writeText(s);
    toast(`Copied ${what}`, "ok");
    return true;
  } catch {
    // navigator.clipboard needs a secure context; 127.0.0.1 qualifies, but a
    // remote-origin browse of the same UI would not.
    try {
      const ta = el("textarea", { style: { position: "fixed", opacity: "0", left: "-9999px" } });
      ta.value = s;
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      toast(ok ? `Copied ${what}` : "Could not copy", ok ? "ok" : "warn");
      return ok;
    } catch {
      toast("Could not copy", "warn");
      return false;
    }
  }
}

/** Reuse the app's toast stack if the shell made one, else create it. */
function toast(msg, kind = "", title = "") {
  let host = document.querySelector(".toasts");
  if (!host) {
    host = el("div.toasts");
    document.body.appendChild(host);
  }
  const t = el(`div.toast${kind ? `.${kind}` : ""}`,
    title ? el("span.t", title) : null, msg);
  host.appendChild(t);
  setTimeout(() => {
    t.classList.add("leaving");
    setTimeout(() => t.remove(), 220);
  }, 3200);
}

/** A standard empty-state block. */
function empty(icon, title, sub) {
  return el("div.empty", el("div.ic", icon), el("div.t", title), sub ? el("div.s", sub) : null);
}

/** The absolute on-disk path of a run-relative media path. */
function absPath(rel) {
  if (!S.run) return String(rel || "");
  const base = String(S.run.path || "").replace(/\/+$/, "");
  return `${base}/${String(rel || "").replace(/^\/+/, "")}`;
}

/** Tokenized URL for one media file (an <img>/<iframe>/<a> target). */
function fileUrl(rel, download = false) {
  return api.url(`/v1/runs/${encodeURIComponent(S.run.id)}/file`,
    { path: rel, token: api.token || undefined, download: download || undefined });
}

/**
 * Tokenized URL for a cached thumbnail.
 * `cb` is a cache-buster: the server sends `Cache-Control: max-age=86400`, so
 * after "Rebuild previews" wipes the server-side cache the browser would keep
 * showing the stale JPEG from its own. Bumping the nonce changes the URL and
 * nothing else — FastAPI ignores the unknown query param (verified).
 */
function thumbUrl(rel, w) {
  return api.url(`/v1/runs/${encodeURIComponent(S.run.id)}/thumb`,
    { path: rel, w, token: api.token || undefined, cb: S.thumbNonce || undefined });
}

/** Show an ApiError the way the user needs to see it. */
function failure(err, where) {
  const msg = err && err.isAuth ? "the server rejected this session's token"
    : err && err.isOffline ? "the LM3 server is unreachable"
      : String((err && err.message) || err);
  return el("div.card.bad",
    el("h4", `Could not load ${where}`),
    el("p", msg));
}


/* ==========================================================================
   PUBLIC API
   ========================================================================== */

/**
 * Does any run have output worth showing? The tab bar calls this to decide
 * whether the Results tab is enabled.
 *
 * It answers from cache so it is safe to call on every render tick. The FIRST
 * call also kicks off a background probe, so the tab can reveal itself even if
 * initResults() has not run yet (an unopened tab never initializes).
 */
export function hasResults() {
  if (!probeStarted) {
    probeStarted = true;
    loadRuns().catch(() => { /* the tab simply stays disabled */ });
  }
  return S.available;
}

/** The run the Results tab is currently showing (for the shell's status line). */
export function currentRun() {
  return S.run;
}

/** Re-read the run list, and the current view with it. Safe to call any time. */
export async function refresh() {
  await loadRuns(true);
  if (!S.loaded) return;
  if (S.view === "media") await loadMedia(true);
  else await loadTables();
}

/**
 * Mount the Results tab into `root` (its .tabpane).
 * Idempotent: calling it again just refreshes.
 */
export function initResults(root) {
  if (S.loaded && D.root === root) { refresh(); return; }
  D.root = root;
  clear(root);
  build(root);
  S.loaded = true;

  loadRuns(true).then(() => {
    if (S.run) selectRun(S.run.id);
  }).catch((err) => {
    clear(D.body);
    D.body.appendChild(failure(err, "the run list"));
  });
}


/* ==========================================================================
   SHELL
   ========================================================================== */

function build(root) {
  root.classList.add("resultstab");

  /* -- run picker + run facts ------------------------------------------- */
  D.runSelect = el("select", {
    onchange: (e) => selectRun(e.target.value),
    style: { minWidth: "260px", flex: "0 1 340px" },
    title: "Every run directory the server can see, newest first",
  });

  D.runMeta = el("div.mute", {
    style: { display: "flex", gap: "14px", alignItems: "center", flexWrap: "wrap",
             fontSize: "12px", marginLeft: "auto" },
  });

  const bar = el("div.toolbar.boxed",
    el("span", { style: { fontSize: "11.5px", letterSpacing: ".05em",
                          textTransform: "uppercase", color: "var(--mute)", fontWeight: "650" } },
      "Run"),
    D.runSelect,
    el("button.btn.ghost.sm", {
      onclick: () => refresh(),
      title: "Re-scan the output folders and reload this run",
    }, "Refresh"),
    D.runMeta,
  );

  /* -- media / database switch ------------------------------------------ */
  D.viewTabs = el("div.subtabs",
    el("button.subtab.active", { onclick: () => setView("media"), dataset: { view: "media" } },
      "Media browser"),
    el("button.subtab", { onclick: () => setView("db"), dataset: { view: "db" } },
      "Project database"),
  );

  D.body = el("div");

  append(root, [bar, D.viewTabs, D.body]);
}

function setView(view) {
  if (S.view === view) return;
  S.view = view;
  for (const b of D.viewTabs.querySelectorAll(".subtab")) {
    b.classList.toggle("active", b.dataset.view === view);
  }
  renderView();
}

function renderView() {
  if (!S.run) {
    clear(D.body);
    D.body.appendChild(empty("—", "No run selected",
      "LM3 has not written any output the server can see yet. Start a run from the Status tab, "
      + "or point project.output.dir at an existing output folder in Settings."));
    return;
  }
  if (S.view === "media") renderMediaView();
  else renderDbView();
}


/* ==========================================================================
   RUNS
   ========================================================================== */

async function loadRuns(refreshServer = false) {
  const payload = await api.get("/v1/runs", { params: { refresh: refreshServer || undefined } });
  S.runs = (Array.isArray(payload && payload.runs) ? payload.runs : []).map(normalizeRun);

  const was = S.available;
  S.available = S.runs.some(runHasOutput);
  if (S.available !== was) {
    // The shell owns the tab bar; tell it rather than reaching into its DOM.
    document.dispatchEvent(new CustomEvent("lm3:results-available", {
      detail: { available: S.available, n: S.runs.length },
    }));
  }

  // Keep probing until the first run appears, then stop — an idle poll that
  // never terminates is exactly the kind of thing that makes an "idle" app
  // burn a core.
  if (!S.available && !pollTimer) {
    pollTimer = setInterval(() => {
      loadRuns().catch(() => { /* transient; the next tick retries */ });
    }, 20000);
  } else if (S.available && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }

  if (S.loaded) fillRunSelect();

  // Keep the selection if it survived the rescan, else take the newest.
  if (S.run) {
    const still = S.runs.find((r) => r.id === S.run.id);
    S.run = still || null;
  }
  // After "New run" the tab must NOT re-adopt the newest run every time it is
  // reopened -- the point of the reset is that no project is selected yet.
  if (!S.run && !S.noAutoSelect) S.run = S.runs.find(runHasOutput) || S.runs[0] || null;
  if (S.loaded) {
    D.runSelect.value = S.run ? S.run.id : "";
    renderRunMeta();
  }
  return S.runs;
}

function fillRunSelect() {
  const sel = D.runSelect;
  clear(sel);
  if (!S.runs.length) {
    sel.appendChild(el("option", { value: "" }, "no runs found"));
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  for (const r of S.runs) {
    const when = r.finished_at || r.started_at || "";
    const bits = [r.name];
    if (r.n_images) bits.push(`${fmtNum(r.n_images)} img`);
    bits.push(r.state);
    if (when) bits.push(String(when).replace("T", " ").replace("Z", ""));
    sel.appendChild(el("option", { value: r.id, title: r.path }, bits.join("  ·  ")));
  }
}

function renderRunMeta() {
  clear(D.runMeta);
  const r = S.run;
  if (!r) return;

  const done = Number(r.modules_done || 0);
  const skipped = Number(r.modules_skipped || 0);
  const total = Number(r.modules_total || 0);

  const facts = [
    ["modules", total ? `${done}/${total}${skipped ? ` (+${skipped} skipped)` : ""}` : "–"],
    ["images", r.n_images ? `${fmtNum(r.n_images_done)}/${fmtNum(r.n_images)}` : "–"],
  ];
  if (r.n_files !== null && r.n_files !== undefined) {
    facts.push(["files", `${fmtNum(r.n_files)} · ${fmtBytes(r.bytes)}`]);
  }

  append(D.runMeta, [
    stateBadge(r.state),
    ...facts.map(([k, v]) => el("span", { style: { whiteSpace: "nowrap" } },
      el("span.dim", `${k} `),
      el("span.mono", { style: { color: "var(--ink)" } }, v))),
    el("button.btn.ghost.sm", {
      onclick: () => copyText(r.path, "run folder"),
      title: r.path,
    }, "Copy path"),
  ]);
}

async function selectRun(runId) {
  S.noAutoSelect = false;    // an explicit choice ends the post-reset suppression
  const next = S.runs.find((r) => r.id === runId);
  if (!next) return;
  S.run = next;
  D.runSelect.value = next.id;
  renderRunMeta();

  // Everything downstream belongs to the old run.
  S.media = null; S.category = null; S.listing = null;
  S.tables = null; S.table = null; S.rows = null;
  S.page.offset = 0; S.page.q = "";
  S.db.offset = 0; S.db.where = ""; S.db.order_by = null; S.db.desc = false;

  renderView();
}


/* ==========================================================================
   MEDIA BROWSER
   ========================================================================== */

function renderMediaView() {
  clear(D.body);

  D.mediaSide = el("div", {
    style: {
      background: "var(--panel)", border: "1px solid var(--line)",
      borderRadius: "var(--r)", overflow: "hidden",
      position: "sticky", top: "0", maxHeight: "calc(100vh - 300px)",
      display: "flex", flexDirection: "column", minHeight: "0",
    },
  });
  D.mediaMain = el("div", { style: { minWidth: "0" } });

  D.body.appendChild(el("div", {
    style: {
      display: "grid", gridTemplateColumns: "minmax(210px,258px) minmax(0,1fr)",
      gap: "14px", alignItems: "start",
    },
  }, D.mediaSide, D.mediaMain));

  D.mediaSide.appendChild(el("div.panel-hd", el("span.t", "Categories"), el("span.spinner")));
  D.mediaMain.appendChild(el("div.empty", el("span.spinner.lg")));

  loadMedia().catch((err) => {
    clear(D.mediaSide); clear(D.mediaMain);
    D.mediaMain.appendChild(failure(err, "this run's media"));
  });
}

async function loadMedia(force = false) {
  if (!S.run) return;
  if (S.media && !force) { renderCategories(); renderMediaMain(); return; }

  const payload = await api.getResults(S.run.id, {
    refresh: force || undefined,
    include_working: true,        // fetch both groups; the sidebar filters locally
  });
  S.media = payload;

  // /media is the only call that indexes the tree, so it is also the only one
  // that can fill in n_files/bytes on the run summary.
  if (payload && payload.run) {
    // Keep OUR key: the <option> values and every later /v1/runs/{run}/... call
    // are built from it, and results_api's own `id` is not what the run list
    // handed us. Everything else in the richer summary is adopted.
    const id = S.run.id;
    Object.assign(S.run, payload.run, { id });
    const idx = S.runs.findIndex((r) => r.id === S.run.id);
    if (idx >= 0) S.runs[idx] = S.run;
    renderRunMeta();
  }

  const cats = visibleCategories();
  if (!S.category || !cats.some((c) => c.name === S.category)) {
    // Open on the most useful thing in a herbarium run: the summary overlay.
    const preferred = cats.find((c) => c.name === "Overlay/Overlay_Summary")
      || cats.find((c) => c.group === "reports" && c.kind === "image")
      || cats[0];
    S.category = preferred ? preferred.name : null;
    S.page.offset = 0;
  }
  renderCategories();
  await loadListing();
}

function visibleCategories() {
  const all = (S.media && S.media.categories) || [];
  return S.includeWorking ? all : all.filter((c) => c.group !== "working");
}

function renderCategories() {
  clear(D.mediaSide);

  const cats = visibleCategories();
  const nWorking = ((S.media && S.media.groups) || [])
    .filter((g) => g.id === "working")
    .reduce((a, g) => a + Number(g.n_categories || 0), 0);

  D.mediaSide.appendChild(el("div.panel-hd",
    el("span.t", "Categories"),
    el("span.tools",
      el("span.mono.dim", { style: { fontSize: "11px" } }, `${cats.length}`))));

  const list = el("div", { style: { overflow: "auto", flex: "1 1 auto", minHeight: "0" } });

  // Group headings so "working" intermediates never sit next to real reports.
  let lastGroup = null;
  const ul = el("ul.tablist");
  for (const c of cats) {
    if (c.group !== lastGroup) {
      lastGroup = c.group;
      const g = ((S.media && S.media.groups) || []).find((x) => x.id === c.group);
      ul.appendChild(el("li", {
        style: {
          cursor: "default", background: "var(--panel2)", padding: "6px 11px",
          fontSize: "11px", letterSpacing: ".05em", textTransform: "uppercase",
          color: "var(--dim)", fontWeight: "650", borderLeft: "2px solid transparent",
        },
      }, el("span.nm", { style: { fontFamily: "var(--sans)" } },
        (g && g.label) || c.group),
      el("span.n", g ? fmtNum(g.n_files) : "")));
    }
    ul.appendChild(el(`li${c.name === S.category ? ".active" : ""}`, {
      onclick: () => selectCategory(c.name),
      title: `${c.name}\n${c.blurb || ""}\n${fmtNum(c.count)} files · ${fmtBytes(c.bytes)}`,
    },
    el("span.nm", c.label || c.name),
    el("span.n", fmtNum(c.count))));
  }
  list.appendChild(ul);
  D.mediaSide.appendChild(list);

  // Working-files toggle lives at the bottom so it reads as a filter, not a category.
  D.mediaSide.appendChild(el("div.panel-ft",
    el("label.check",
      el("input", {
        type: "checkbox", checked: S.includeWorking || undefined,
        onchange: (e) => {
          S.includeWorking = e.target.checked;
          const cur = visibleCategories();
          if (!cur.some((x) => x.name === S.category)) {
            S.category = cur.length ? cur[0].name : null;
            S.page.offset = 0;
            renderCategories();
            loadListing();
            return;
          }
          renderCategories();
        },
      }),
      el("span.lbl", `Working files${nWorking ? ` (${nWorking})` : ""}`)),
  ));
}

function selectCategory(name) {
  if (S.category === name) return;
  S.category = name;
  S.page.offset = 0;
  renderCategories();
  loadListing();
}

async function loadListing() {
  if (!S.run || !S.category) {
    S.listing = null;
    renderMediaMain();
    return;
  }
  clear(D.mediaMain);
  D.mediaMain.appendChild(el("div.empty", el("span.spinner.lg")));
  try {
    S.listing = await api.get(`/v1/runs/${encodeURIComponent(S.run.id)}/media/list`, {
      params: {
        category: S.category,
        offset: S.page.offset,
        limit: S.page.limit,
        sort: S.page.sort,
        desc: S.page.desc || undefined,
        q: S.page.q || undefined,
      },
    });
  } catch (err) {
    clear(D.mediaMain);
    D.mediaMain.appendChild(failure(err, "this category"));
    return;
  }
  renderMediaMain();
}

function renderMediaMain() {
  clear(D.mediaMain);
  if (!S.media) return;

  const cats = visibleCategories();
  if (!cats.length) {
    D.mediaMain.appendChild(empty("—", "Nothing to browse",
      "This run wrote no report folders. Enable the Reporter module in Settings and re-run."));
    return;
  }

  const L = S.listing;
  const cat = (S.media.categories || []).find((c) => c.name === S.category);

  /* -- header: what this category is ------------------------------------ */
  const head = el("div", { style: { margin: "0 0 10px" } },
    el("div", { style: { display: "flex", alignItems: "baseline", gap: "10px", flexWrap: "wrap" } },
      el("h2", { style: { margin: "0" } }, (cat && cat.label) || S.category),
      el("span.mono.dim", { style: { fontSize: "11.5px" } }, (cat && cat.rel) || ""),
      cat && cat.group === "working"
        ? el("span.badge.warn", { title: "An intermediate LM3 writes for its own use" }, "working")
        : null),
    cat && cat.blurb ? el("p.hint", { style: { margin: "5px 0 0" } }, cat.blurb) : null);

  /* -- toolbar ---------------------------------------------------------- */
  const search = el("input", {
    type: "search", placeholder: "filter filenames…", value: S.page.q,
    oninput: debounce((e) => {
      S.page.q = e.target.value.trim();
      S.page.offset = 0;
      loadListing();
    }, 260),
  });

  const sortSel = el("select", {
    style: { width: "auto" },
    onchange: (e) => { S.page.sort = e.target.value; S.page.offset = 0; loadListing(); },
  }, ...[["name", "name"], ["mtime", "date written"], ["bytes", "size"]]
    .map(([v, t]) => el("option", { value: v, selected: S.page.sort === v || undefined }, `sort: ${t}`)));

  const sizeSel = el("select", {
    style: { width: "auto" },
    onchange: (e) => { S.tile = e.target.value; renderMediaMain(); },
  }, ...[["sm", "small"], ["md", "medium"], ["lg", "large"]]
    .map(([v, t]) => el("option", { value: v, selected: S.tile === v || undefined }, `tiles: ${t}`)));

  const perSel = el("select", {
    style: { width: "auto" },
    onchange: (e) => { S.page.limit = Number(e.target.value); S.page.offset = 0; loadListing(); },
  }, ...MEDIA_PAGE_SIZES.map((n) =>
    el("option", { value: n, selected: S.page.limit === n || undefined }, `${n} / page`)));

  const toolbar = el("div.toolbar",
    el("div.searchbox", search),
    sortSel,
    el("button.btn.ghost.sm", {
      onclick: () => { S.page.desc = !S.page.desc; S.page.offset = 0; loadListing(); },
      title: "Reverse the sort order",
    }, S.page.desc ? "▼ desc" : "▲ asc"),
    sizeSel,
    perSel,
    el("span.spacer"),
    el("button.btn.ghost.sm", {
      onclick: rebuildThumbs,
      title: "Delete the cached thumbnails for this run and render them again",
    }, "Rebuild previews"));

  append(D.mediaMain, [head, toolbar]);

  if (!L || !L.files || !L.files.length) {
    D.mediaMain.appendChild(empty("—", S.page.q ? "No file matches that filter" : "This category is empty",
      S.page.q ? `Nothing in ${S.category} contains “${S.page.q}”.` : ""));
    return;
  }

  /* -- the grid --------------------------------------------------------- */
  const tile = TILE_SIZES[S.tile] || TILE_SIZES.md;
  const grid = el(`div.${tile.grid.split(" ").join(".")}`);
  if (thumbObserver) thumbObserver.disconnect();
  thumbObserver = new IntersectionObserver((entries, obs) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const img = entry.target;
      obs.unobserve(img);
      // Only NOW does the browser request the JPEG — a 480-tile page would
      // otherwise fire 480 renders at once and stall the threadpool.
      img.src = img.dataset.src;
    }
  }, { rootMargin: "400px 0px", threshold: 0.01 });

  L.files.forEach((f, i) => grid.appendChild(mediaTile(f, i, tile.w)));
  D.mediaMain.appendChild(grid);

  /* -- pager ------------------------------------------------------------ */
  D.mediaMain.appendChild(pager({
    offset: L.offset, limit: L.limit, total: L.n_total, unit: "files",
    onGo: (off) => { S.page.offset = off; loadListing(); },
    extra: `${fmtBytes((cat && cat.bytes) || 0)} in this category`,
  }));
}

/** One grid tile. Images lazy-load; everything else gets a typed placeholder. */
function mediaTile(f, index, width) {
  const cap = el("div.cap", { title: f.name },
    midEllipsis(f.name, S.tile === "sm" ? 22 : 40),
    el("div.sz", `${fmtBytes(f.bytes)} · ${fmtTime(f.mtime)}`));

  let thumb;
  if (f.thumbable) {
    thumb = el("img.thumb", {
      alt: f.name, loading: "lazy", decoding: "async",
      dataset: { src: thumbUrl(f.path, width) },
      onerror: (e) => {
        // A thumb can 415/501 (no Pillow) or 404 (file removed mid-browse).
        // Swap in a placeholder rather than leaving a broken-image glyph.
        const ph = placeholderThumb(f.ext || "?", "var(--dim)");
        e.target.replaceWith(ph);
      },
    });
    thumbObserver.observe(thumb);
  } else {
    thumb = placeholderThumb(f.ext || "file", kindColor(f.kind));
  }

  const tileEl = el(`div.mediatile${f.thumbable ? "" : ".folder"}`, {
    onclick: () => openViewer(index),
    title: f.path,
  }, thumb, cap);
  return tileEl;
}

function placeholderThumb(label, color) {
  return el("div.thumb", {
    style: {
      display: "flex", alignItems: "center", justifyContent: "center",
      aspectRatio: "4/3", background: "var(--panel2)",
      font: "650 15px/1 var(--mono)", letterSpacing: ".08em",
      textTransform: "uppercase", color: color || "var(--acc)",
    },
  }, String(label).slice(0, 6));
}

function kindColor(kind) {
  return {
    html: "var(--acc2)", table: "var(--acc3)", data: "var(--vio)",
    text: "var(--mute)", image: "var(--acc)",
  }[kind] || "var(--dim)";
}

async function rebuildThumbs() {
  if (!S.run) return;
  try {
    const res = await api.del(`/v1/runs/${encodeURIComponent(S.run.id)}/thumbs`);
    S.thumbNonce = (S.thumbNonce || 0) + 1;
    toast(`Removed ${fmtNum(res && res.removed)} cached previews`, "ok");
    renderMediaMain();
  } catch (err) {
    toast(String(err.message || err), "bad", "Rebuild failed");
  }
}


/* ------------------------------------------------------------------ pager -- */

/**
 * A shared offset pager. `onGo(newOffset)` refetches — we never slice a big
 * result client-side, so this is the only way pages move.
 */
function pager({ offset, limit, total, unit, onGo, extra }) {
  const first = total ? offset + 1 : 0;
  const last = Math.min(offset + limit, total);
  const page = Math.floor(offset / limit) + 1;
  const pages = Math.max(1, Math.ceil(total / limit));
  const lastOffset = (pages - 1) * limit;

  const btn = (label, target, enabled, title) =>
    el(`button.btn.ghost.sm${enabled ? "" : ".disabled"}`, {
      onclick: enabled ? () => onGo(target) : null,
      disabled: enabled ? undefined : true,
      title: title || "",
    }, label);

  const jump = el("input", {
    type: "number", min: "1", max: String(pages), value: String(page),
    style: { width: "72px", textAlign: "center" },
    onchange: (e) => {
      const p = Math.max(1, Math.min(pages, Number(e.target.value) || 1));
      onGo((p - 1) * limit);
    },
  });

  return el("div.tblfoot",
    el("span", `${fmtNum(first)}–${fmtNum(last)} of ${fmtNum(total)} ${unit}`),
    extra ? el("span.dim", `· ${extra}`) : null,
    el("span.pager",
      btn("⏮", 0, offset > 0, "First page"),
      btn("‹ prev", Math.max(0, offset - limit), offset > 0),
      jump,
      el("span.dim", `/ ${fmtNum(pages)}`),
      btn("next ›", offset + limit, last < total),
      btn("⏭", lastOffset, last < total, "Last page")));
}


/* ==========================================================================
   VIEWER  (lightbox / iframe / CSV table / metadata)
   ========================================================================== */

const V = { open: false, index: 0, zoom: "fit", node: null, onKey: null, natural: null };

function openViewer(index) {
  const L = S.listing;
  if (!L || !L.files || !L.files[index]) return;
  V.index = index;
  V.zoom = "fit";
  V.natural = null;

  if (!V.open) {
    V.open = true;
    V.node = el("div.backdrop", {
      onclick: (e) => { if (e.target === V.node) closeViewer(); },
    });
    document.body.appendChild(V.node);
    V.onKey = (e) => {
      if (e.key === "Escape") { closeViewer(); return; }
      if (e.key === "ArrowRight" || e.key === "PageDown") { e.preventDefault(); step(1); }
      else if (e.key === "ArrowLeft" || e.key === "PageUp") { e.preventDefault(); step(-1); }
      else if (e.key === "f" || e.key === "F") { V.zoom = "fit"; renderViewer(); }
      else if (e.key === "1") { V.zoom = 1; renderViewer(); }
    };
    document.addEventListener("keydown", V.onKey);
  }
  renderViewer();
}

function closeViewer() {
  if (!V.open) return;
  V.open = false;
  document.removeEventListener("keydown", V.onKey);
  if (V.node) V.node.remove();
  V.node = null;
}

/**
 * Move by ±1 within the page, crossing page boundaries when there are more
 * files server-side. Loading the neighboring page keeps arrow-key browsing
 * continuous through a 661-file category instead of dead-ending every 120.
 */
async function step(delta) {
  const L = S.listing;
  if (!L) return;
  const next = V.index + delta;

  if (next >= 0 && next < L.files.length) {
    V.index = next; V.zoom = "fit"; V.natural = null;
    renderViewer();
    return;
  }
  const forward = delta > 0;
  const newOffset = forward ? L.offset + L.limit : L.offset - L.limit;
  if (forward ? newOffset >= L.n_total : newOffset < 0) return;   // at the ends

  S.page.offset = Math.max(0, newOffset);
  await loadListing();
  const files = (S.listing && S.listing.files) || [];
  if (!files.length) return;
  V.index = forward ? 0 : files.length - 1;
  V.zoom = "fit"; V.natural = null;
  renderViewer();
}

function renderViewer() {
  if (!V.open || !V.node) return;
  const L = S.listing;
  const f = L && L.files && L.files[V.index];
  if (!f) { closeViewer(); return; }

  clear(V.node);
  const globalIndex = (L.offset || 0) + V.index + 1;

  const tools = el("div.cap",
    el("span.mono", { style: { color: "var(--ink)" } }, f.name),
    el("span.dim", `${fmtNum(globalIndex)} / ${fmtNum(L.n_total)}`),
    el("span.dim", `${fmtBytes(f.bytes)} · ${fmtTime(f.mtime, { withDate: true })}`));

  const actions = el("div", { style: { display: "flex", gap: "6px", alignItems: "center", flexWrap: "wrap" } },
    el("button.btn.ghost.sm", { onclick: () => step(-1), title: "Left arrow" }, "‹ prev"),
    el("button.btn.ghost.sm", { onclick: () => step(1), title: "Right arrow" }, "next ›"));

  if (f.kind === "image") {
    const isFit = V.zoom === "fit";
    append(actions, [
      el(`button.btn.ghost.sm${isFit ? ".on" : ""}`, {
        onclick: () => { V.zoom = "fit"; renderViewer(); }, title: "F",
      }, "Fit"),
      el(`button.btn.ghost.sm${V.zoom === 1 ? ".on" : ""}`, {
        onclick: () => { V.zoom = 1; renderViewer(); }, title: "1",
      }, "1:1"),
      el("button.btn.ghost.sm", {
        onclick: () => { V.zoom = zoomStep(-1); renderViewer(); },
      }, "−"),
      el("span.mono.dim", { style: { minWidth: "48px", textAlign: "center", fontSize: "11.5px" } },
        isFit ? "fit" : `${Math.round(V.zoom * 100)}%`),
      el("button.btn.ghost.sm", {
        onclick: () => { V.zoom = zoomStep(1); renderViewer(); },
      }, "+"),
    ]);
  }

  append(actions, [
    el("button.btn.ghost.sm", {
      onclick: () => copyText(absPath(f.path), "file path"),
      title: absPath(f.path),
    }, "Copy path"),
    el("a.btn.ghost.sm", { href: fileUrl(f.path, true), download: f.name }, "Download"),
    el("a.btn.ghost.sm", { href: fileUrl(f.path), target: "_blank", rel: "noopener" }, "Open raw"),
    el("button.btn.sm", { onclick: closeViewer, title: "Escape" }, "Close"),
  ]);

  const box = el("div.lightbox", viewerContent(f), tools, actions);
  V.node.appendChild(box);
}

function zoomStep(dir) {
  const cur = V.zoom === "fit" ? 1 : V.zoom;
  const next = dir > 0 ? cur * 1.4 : cur / 1.4;
  return Math.max(0.1, Math.min(8, Number(next.toFixed(3))));
}

/** The body of the viewer, chosen by what the file actually is. */
function viewerContent(f) {
  switch (f.kind) {
    case "image": return imageView(f);
    case "html": return htmlView(f);
    case "table": return textView(f, "table");
    case "text": return textView(f, "text");
    default: return metaView(f);
  }
}

function imageView(f) {
  const fit = V.zoom === "fit";
  const img = el("img", {
    src: fileUrl(f.path),
    alt: f.name,
    style: fit
      ? { maxWidth: "100%", maxHeight: "calc(100vh - 190px)", width: "auto", height: "auto" }
      : { maxWidth: "none", maxHeight: "none", width: "auto", height: "auto" },
    onload: (e) => {
      V.natural = { w: e.target.naturalWidth, h: e.target.naturalHeight };
      if (V.zoom !== "fit") e.target.style.width = `${Math.round(V.natural.w * V.zoom)}px`;
    },
  });

  if (fit) return img;

  // At any explicit zoom the image can exceed the viewport, so it needs its own
  // scroller. Ctrl/Cmd+wheel zooms (matching every image viewer on the box);
  // a plain wheel scrolls, which is what you want when reading a big sheet.
  const wrap = el("div", {
    style: {
      overflow: "auto", maxWidth: "100%", maxHeight: "calc(100vh - 190px)",
      background: "var(--void)", border: "1px solid var(--line)",
      borderRadius: "var(--r)", padding: "0",
    },
    onwheel: (e) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      V.zoom = zoomStep(e.deltaY < 0 ? 1 : -1);
      renderViewer();
    },
  }, img);
  if (V.natural) img.style.width = `${Math.round(V.natural.w * V.zoom)}px`;
  return wrap;
}

function htmlView(f) {
  // The server already serves run HTML under `Content-Security-Policy: sandbox
  // allow-scripts`; the iframe sandbox is the second half of that belt-and-
  // braces pair. timing.html is fully self-contained, so allow-scripts is
  // enough for it to render its own charts while staying unable to read the
  // localStorage that holds the server token.
  return el("iframe", {
    src: fileUrl(f.path),
    sandbox: "allow-scripts",
    title: f.name,
    style: {
      width: "min(1180px, 92vw)", height: "calc(100vh - 190px)",
      border: "1px solid var(--line)", borderRadius: "var(--r)",
      background: "var(--bg)",
    },
  });
}

function textView(f, mode) {
  const host = el("div", {
    style: {
      width: "min(1180px, 92vw)", maxHeight: "calc(100vh - 190px)",
      overflow: "auto", background: "var(--panel)",
      border: "1px solid var(--line)", borderRadius: "var(--r)",
    },
  }, el("div.empty", el("span.spinner.lg")));

  api.get(`/v1/runs/${encodeURIComponent(S.run.id)}/file`,
    { params: { path: f.path }, raw: true })
    .then((res) => res.text())
    .then((text) => {
      clear(host);
      if (mode === "table" && /\.(csv|tsv)$/i.test(f.name)) {
        host.appendChild(csvTable(text, f.name.toLowerCase().endsWith(".tsv") ? "\t" : ","));
      } else {
        host.appendChild(el("pre", {
          style: {
            margin: "0", padding: "12px 14px", font: "12px/1.6 var(--mono)",
            color: "var(--mute)", whiteSpace: "pre-wrap", overflowWrap: "anywhere",
          },
        }, text));
      }
    })
    .catch((err) => { clear(host); host.appendChild(failure(err, f.name)); });

  return host;
}

/** Minimal RFC4180 parser: quoted fields, escaped quotes, embedded newlines. */
function parseCsv(text, sep = ",") {
  const rows = [];
  let row = [], field = "", quoted = false, i = 0;
  const s = String(text).replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  while (i < s.length) {
    const ch = s[i];
    if (quoted) {
      if (ch === '"') {
        if (s[i + 1] === '"') { field += '"'; i += 2; continue; }
        quoted = false; i += 1; continue;
      }
      field += ch; i += 1; continue;
    }
    if (ch === '"') { quoted = true; i += 1; continue; }
    if (ch === sep) { row.push(field); field = ""; i += 1; continue; }
    if (ch === "\n") { row.push(field); rows.push(row); row = []; field = ""; i += 1; continue; }
    field += ch; i += 1;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows.filter((r) => r.length > 1 || (r[0] || "").length);
}

function csvTable(text, sep) {
  const rows = parseCsv(text, sep);
  if (!rows.length) return empty("—", "Empty file", "");
  const [header, ...body] = rows;
  const isNum = (v) => v !== "" && Number.isFinite(Number(v));

  return el("div.tblwrap", { style: { margin: "0", border: "none", borderRadius: "0" } },
    el("table.datatable.dense",
      el("thead", el("tr", ...header.map((h, i) =>
        el(`th${i === 0 ? ".l" : ""}`, h)))),
      el("tbody", ...body.map((r) => el("tr", ...header.map((_, i) => {
        const v = r[i] === undefined ? "" : r[i];
        return el(`td${isNum(v) ? ".num" : ""}`, { title: v }, v);
      }))))));
}

/**
 * The fallback for a file we should NOT try to render: .h5 coordinate blobs
 * above all. Half a megabyte of HDF5 in an <img> is a broken tile; its
 * identity is what the user actually wants.
 */
function metaView(f) {
  const abs = absPath(f.path);
  // LM3 product names encode the specimen, the product tag and the source
  // bounding box: <specimen>__<tag>__<x1>_<y1>_<x2>_<y2>.<ext>
  const m = /^(.+?)__(.+?)__(\d+)_(\d+)_(\d+)_(\d+)\.[^.]+$/.exec(f.name);
  const parsed = m ? [
    ["Specimen", m[1]],
    ["Product tag", m[2]],
    ["Source box (x1 y1 x2 y2)", `${m[3]}  ${m[4]}  ${m[5]}  ${m[6]}`],
    ["Box size (px)", `${Number(m[5]) - Number(m[3])} × ${Number(m[6]) - Number(m[4])}`],
  ] : [];

  const note = f.ext === "h5"
    ? "An HDF5 array of leaf-contour coordinates and Euler Characteristic Transform values, "
      + "written by the ECT module. Open it with h5py or LM3's own readers — it is not an image."
    : "LM3 has no preview for this file type. Download it to open in the right tool.";

  return el("div", {
    style: {
      width: "min(720px, 92vw)", background: "var(--panel)",
      border: "1px solid var(--line)", borderRadius: "var(--r)", overflow: "hidden",
    },
  },
  el("div.panel-hd", el("span.t", `${String(f.ext || "file").toUpperCase()} file`)),
  el("div.panel-bd",
    el("p.hint", { style: { margin: "0 0 12px" } }, note),
    el("ul.kv",
      el("li", el("span.k", "Name"), el("span.v", f.name)),
      el("li", el("span.k", "Size"), el("span.v", fmtBytes(f.bytes))),
      el("li", el("span.k", "Written"), el("span.v", fmtTime(f.mtime, { withDate: true }))),
      ...parsed.map(([k, v]) => el("li", el("span.k", k), el("span.v", v))),
      el("li", el("span.k", "Path"), el("span.v", { style: { fontSize: "11.5px" } }, abs)))));
}


/* ==========================================================================
   DATABASE INSPECTOR
   ========================================================================== */

function renderDbView() {
  clear(D.body);

  if (!S.run.has_db) {
    D.body.appendChild(empty("—", "This run has no project database",
      "LM3 writes <run>.sqlite when it starts. A run directory without one never got past setup."));
    return;
  }

  D.dbSide = el("div", {
    style: {
      background: "var(--panel)", border: "1px solid var(--line)",
      borderRadius: "var(--r)", overflow: "hidden",
      position: "sticky", top: "0", maxHeight: "calc(100vh - 300px)",
      display: "flex", flexDirection: "column", minHeight: "0",
    },
  });
  D.dbMain = el("div", { style: { minWidth: "0" } });

  D.body.appendChild(el("div", {
    style: {
      display: "grid", gridTemplateColumns: "minmax(210px,258px) minmax(0,1fr)",
      gap: "14px", alignItems: "start",
    },
  }, D.dbSide, D.dbMain));

  D.dbSide.appendChild(el("div.panel-hd", el("span.t", "Tables"), el("span.spinner")));
  D.dbMain.appendChild(el("div.empty", el("span.spinner.lg")));

  loadTables().catch((err) => {
    clear(D.dbSide); clear(D.dbMain);
    D.dbMain.appendChild(failure(err, "the project database"));
  });
}

async function loadTables() {
  if (!S.run) return;
  S.tables = await api.getTables(S.run.id);
  const objs = (S.tables && S.tables.tables) || [];
  if (!S.table || !objs.some((t) => t.name === S.table)) {
    // `specimen` is the spine of the ledger — the most useful default by far.
    const preferred = objs.find((t) => t.name === "specimen") || objs[0];
    S.table = preferred ? preferred.name : null;
    S.db.offset = 0; S.db.order_by = null; S.db.desc = false; S.db.where = "";
  }
  renderTableList();
  await loadRows();
}

function renderTableList() {
  clear(D.dbSide);
  const t = S.tables || {};
  const objs = t.tables || [];

  D.dbSide.appendChild(el("div.panel-hd",
    el("span.t", "Tables & views"),
    el("span.tools", el("span.mono.dim", { style: { fontSize: "11px" } },
      `${t.n_tables || 0}+${t.n_views || 0}`))));

  const list = el("div", { style: { overflow: "auto", flex: "1 1 auto", minHeight: "0" } });
  const ul = el("ul.tablist");
  for (const o of objs) {
    ul.appendChild(el(`li${o.name === S.table ? ".active" : ""}`, {
      onclick: () => selectTable(o.name),
      title: `${o.type} · ${fmtNum(o.n_rows)} rows · ${o.n_columns} columns`
        + (o.pk && o.pk.length ? `\nprimary key: ${o.pk.join(", ")}` : ""),
    },
    el("span.nm", o.name),
    o.type === "view" ? el("span.badge.vio", { style: { marginLeft: "4px" } }, "view") : null,
    el("span.n", fmtNum(o.n_rows))));
  }
  list.appendChild(ul);
  D.dbSide.appendChild(list);

  D.dbSide.appendChild(el("div.panel-ft",
    el("span.mono", { style: { fontSize: "11px", overflowWrap: "anywhere" } },
      `${fmtBytes(t.bytes)} · ${fmtNum(t.page_count)} pages`)));
}

function selectTable(name) {
  if (S.table === name) return;
  S.table = name;
  S.db.offset = 0; S.db.order_by = null; S.db.desc = false; S.db.where = "";
  renderTableList();
  loadRows();
}

async function loadRows() {
  if (!S.run || !S.table) return;
  clear(D.dbMain);
  D.dbMain.appendChild(el("div.empty", el("span.spinner.lg")));
  try {
    S.rows = await api.getTable(S.run.id, S.table, {
      offset: S.db.offset,
      limit: S.db.limit,
      order_by: S.db.order_by || undefined,
      desc: S.db.desc || undefined,
      where: S.db.where || undefined,
    });
  } catch (err) {
    clear(D.dbMain);
    // A bad `where` or order_by is a 400 with a message written for the user.
    D.dbMain.appendChild(el("div.card.bad",
      el("h4", "Query rejected"),
      el("p", String((err.body && err.body.detail) || err.message || err))));
    D.dbMain.appendChild(dbToolbar());
    return;
  }
  renderRows();
}

function dbToolbar() {
  const schema = ((S.tables && S.tables.tables) || []).find((t) => t.name === S.table);

  const filter = el("input", {
    type: "search", value: S.db.where,
    placeholder: "filter: text · col:text · col=v · col>N · col=NULL",
    title: "Whitespace-separated terms are ANDed. \"quoted phrases\" work.\n"
      + "  foo         any column contains foo\n"
      + "  col:foo     that column contains foo\n"
      + "  col=foo     equals (numeric when the value is a number)\n"
      + "  col!=foo  col>N  col<N  col>=N  col<=N\n"
      + "  col=NULL    IS NULL",
    oninput: debounce((e) => {
      S.db.where = e.target.value.trim();
      S.db.offset = 0;
      loadRows();
    }, 320),
  });

  const perSel = el("select", {
    style: { width: "auto" },
    onchange: (e) => { S.db.limit = Number(e.target.value); S.db.offset = 0; loadRows(); },
  }, ...DB_PAGE_SIZES.map((n) =>
    el("option", { value: n, selected: S.db.limit === n || undefined }, `${n} rows / page`)));

  return el("div.toolbar",
    el("div.searchbox", { style: { flex: "1 1 320px" } }, filter),
    perSel,
    S.db.order_by
      ? el("button.btn.ghost.sm", {
        onclick: () => { S.db.order_by = null; S.db.desc = false; S.db.offset = 0; loadRows(); },
        title: `Currently sorted by ${S.db.order_by}`,
      }, "Clear sort")
      : null,
    el("span.spacer"),
    el("button.btn.ghost.sm", {
      onclick: () => copyPageCsv(),
      title: "Copy the rows currently on screen as CSV",
    }, "Copy page as CSV"),
    el("button.btn.ghost.sm", {
      onclick: () => toggleSchema(schema),
      title: "Show every column with its declared type",
    }, "Schema"),
    el("button.btn.ghost.sm", { onclick: () => loadRows() }, "Reload"));
}

function toggleSchema(schema) {
  if (!schema) return;
  const existing = D.dbMain.querySelector(".schemapanel");
  if (existing) { existing.remove(); return; }

  const panel = el("div.panel.schemapanel",
    el("div.panel-hd",
      el("span.t", `${schema.name} — ${schema.n_columns} columns`),
      el("span.tools",
        schema.pk && schema.pk.length
          ? el("span.badge.acc", `pk: ${schema.pk.join(", ")}`)
          : el("span.badge", "no primary key"))),
    el("div.panel-bd.flush",
      el("div.tblwrap", { style: { margin: "0", border: "none", borderRadius: "0" } },
        el("table.datatable.dense",
          el("thead", el("tr",
            el("th.l", "column"), el("th.l", "type"),
            el("th.l", "pk"), el("th.l", "not null"), el("th.l", "default"))),
          el("tbody", ...schema.columns.map((c) => el("tr",
            el("td", { style: { color: "var(--ink)" } }, c.name),
            el("td.mono", { style: { color: "var(--acc2)" } }, c.type || "—"),
            el("td", c.pk ? el("span.badge.acc", "pk") : el("span.dim", "")),
            el("td", c.notnull ? el("span.badge.warn", "not null") : el("span.dim", "")),
            el("td", c.default === null || c.default === undefined
              ? el("span.dim", "—") : String(c.default)))))))));

  // Slot the schema directly under the toolbar so it reads as an annotation
  // of the grid below it, not a separate page.
  const anchor = D.dbMain.querySelector(".toolbar");
  if (anchor && anchor.nextSibling) D.dbMain.insertBefore(panel, anchor.nextSibling);
  else D.dbMain.appendChild(panel);
}

function renderRows() {
  clear(D.dbMain);
  const R = S.rows;
  const schema = ((S.tables && S.tables.tables) || []).find((t) => t.name === S.table);

  const head = el("div", { style: { margin: "0 0 4px", display: "flex",
    alignItems: "baseline", gap: "10px", flexWrap: "wrap" } },
  el("h2", { style: { margin: "0" } }, S.table),
  R.type === "view" ? el("span.badge.vio", "view") : el("span.badge", "table"),
  el("span.dim", { style: { fontSize: "12px" } },
    `${fmtNum(R.n_rows_table)} rows · ${R.columns.length} columns`),
  S.db.where
    ? el("span.badge.info", `filtered to ${fmtNum(R.n_total)}`)
    : null);

  append(D.dbMain, [head, dbToolbar()]);

  if (!R.rows.length) {
    D.dbMain.appendChild(empty("—",
      S.db.where ? "No row matches that filter" : "This table is empty",
      S.db.where ? `Nothing in ${S.table} matches “${S.db.where}”.` : ""));
    return;
  }

  /* -- the grid: sticky head, pinned first column, server-side sort ------ */
  const types = R.column_types || [];
  const thead = el("thead", el("tr", ...R.columns.map((c, i) => {
    const sorted = S.db.order_by === c;
    const th = el(`th.l.sortable${sorted ? ".sorted" : ""}${sorted && S.db.desc ? ".desc" : ""}`, {
      title: `${c}\ntype: ${types[i] || "—"}\nclick to sort server-side`,
      onclick: () => {
        // Same column -> flip direction; new column -> ascending.
        if (S.db.order_by === c) S.db.desc = !S.db.desc;
        else { S.db.order_by = c; S.db.desc = false; }
        S.db.offset = 0;
        loadRows();
      },
    }, c);
    return th;
  })));

  const tbody = el("tbody", ...R.rows.map((row, ri) => {
    const tr = el("tr", {
      onclick: () => openRowModal(R, row, (R.offset || 0) + ri + 1),
      style: { cursor: "pointer" },
      title: "Click for the full row",
    }, ...row.map((v, ci) => cell(v, types[ci])));
    return tr;
  }));

  const wrap = el("div.tblwrap.tall", { style: { margin: "0" } },
    el("table.datatable.dense.pin1", thead, tbody));

  D.dbMain.appendChild(wrap);
  D.dbMain.appendChild(pager({
    offset: R.offset, limit: R.limit, total: R.n_total, unit: "rows",
    onGo: (off) => { S.db.offset = off; loadRows(); },
    extra: schema && schema.pk && schema.pk.length ? `key ${schema.pk.join(", ")}` : "",
  }));
}

/** One data cell, typed the way the report tables type theirs. */
function cell(v, declaredType) {
  if (v === null || v === undefined) return el("td.null", "NULL");
  if (typeof v === "number") return el("td.num", { title: String(v) }, fmtNum(v, { digits: null }));
  if (typeof v === "boolean") return el("td", v ? "true" : "false");
  const s = String(v);
  // BLOBs come back as the literal marker "<blob N B>" — never as bytes.
  if (/^<blob \d+ B>$/.test(s)) return el("td", { style: { color: "var(--vio)" } }, s);
  const numeric = /^(INT|REAL|NUM|FLOA|DOUB|DEC)/i.test(String(declaredType || ""))
    && s !== "" && Number.isFinite(Number(s));
  return el(`td${numeric ? ".num" : ""}`, { title: s }, s);
}

/** The 87-column tables are unreadable sideways; one row, vertically, is not. */
function openRowModal(R, row, rowNumber) {
  const types = R.column_types || [];
  const backdrop = el("div.backdrop", {
    onclick: (e) => { if (e.target === backdrop) backdrop.remove(); },
  });
  const onKey = (e) => {
    if (e.key === "Escape") { backdrop.remove(); document.removeEventListener("keydown", onKey); }
  };
  document.addEventListener("keydown", onKey);

  const asObject = () => {
    const o = {};
    R.columns.forEach((c, i) => { o[c] = row[i]; });
    return o;
  };

  backdrop.appendChild(el("div.modal",
    el("div.panel-hd",
      el("span.t", `${R.table} · row ${fmtNum(rowNumber)}`),
      el("span.tools",
        el("button.btn.ghost.sm", {
          onclick: () => copyText(JSON.stringify(asObject(), null, 2), "row as JSON"),
        }, "Copy JSON"),
        el("button.btn.sm", {
          onclick: () => { backdrop.remove(); document.removeEventListener("keydown", onKey); },
        }, "Close"))),
    el("div.panel-bd",
      el("ul.kv", ...R.columns.map((c, i) => el("li",
        el("span.k", { title: types[i] || "" }, c),
        el("span.v", row[i] === null || row[i] === undefined
          ? el("span.nul", "NULL")
          : String(row[i]))))))));

  document.body.appendChild(backdrop);
}

function copyPageCsv() {
  const R = S.rows;
  if (!R || !R.rows.length) { toast("Nothing on this page to copy", "warn"); return; }
  const q = (v) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [R.columns.map(q).join(",")];
  for (const row of R.rows) lines.push(row.map(q).join(","));
  copyText(lines.join("\n"), `${fmtNum(R.rows.length)} rows as CSV`);
}


/** Drop the selected run so the tab does not keep showing the previous project. */
document.addEventListener("lm3:newrun", () => {
  S.run = null;
  S.media = null;
  S.category = null;
  S.listing = null;
  S.noAutoSelect = true;
  if (!S.loaded) return;
  if (D.runSelect) D.runSelect.value = "";
  if (D.body) {
    clear(D.body);
    D.body.appendChild(empty("—", "No run selected",
      "Start a run, or pick an existing one above."));
  }
  if (D.runMeta) clear(D.runMeta);
});

export default initResults;
