/* ==========================================================================
   LM3 — Settings tab (LM3_settings.yaml editor)
   --------------------------------------------------------------------------
   The whole settings file, rendered from its own structure: every nesting
   level of the YAML becomes a collapsible section, every leaf becomes a typed
   control, and the settings themselves are organized into sub-tabs taken from
   `_sections` in settings_meta.json.

   Three rules shape the design:

     1. NEVER WRITE AN INVALID CONFIG. Every edit is re-validated against
        POST /v1/settings/validate (which always answers 200), so errors land
        inline next to the field that caused them long before Save is pressed.
        Save re-validates and refuses on a hard error.

     2. EDIT A DEEP CLONE, WRITE THE WHOLE TREE. PUT /v1/settings takes the
        full nested object, not a patch, so we keep `orig` (what the file said)
        and `draft` (what the user means) side by side and diff them. Nothing
        mutates the object the server handed us.

     3. THE IMPORTANT SETTINGS ARE THE POINT. 95 of the 289 settings are marked
        important in the metadata — those are the knobs a person actually
        turns. They get the green `.row-important` wash, a green dot, a count on
        every group header, and a one-click filter that hides everything else.

   Exports `initSettings(root)`; the returned controller is what the primary
   strip above the tabs uses to read/write input dir, output dir and temp dir
   without duplicating any of this.
   ========================================================================== */

import {
  api, el, append, clear, esc, debounce, fmtBytes, fmtTime,
} from "../api.js";
import { PHASES, MODULE_BY_KEY } from "../modules.js";

/* Remembered across reloads: which section was open, which groups were
   expanded, and the two view toggles. Deliberately per-browser, not per-file —
   a user's reading habits do not belong in LM3_settings.yaml. */
const LS_OPEN = "lm3.settings.open";
const LS_SECTION = "lm3.settings.section";
const LS_GROUP = "lm3.settings.group";
const LS_IMPORTANT = "lm3.settings.importantOnly";
const LS_DESC = "lm3.settings.showDesc";

/* Depth beyond this stops indenting. report.overlay.classes.plant.Leaf_WHOLE
   is five levels down; at 14px a level the label column would be all padding. */
const MAX_INDENT_DEPTH = 3;

/**
 * Which control types the "important" highlight actually PAINTS.
 *
 * `important` marks 96 of the 299 settings, but 53 of those are switches and
 * dropdowns -- and a green wash behind a toggle says nothing the toggle does not
 * already say, so the highlight read as decoration instead of guidance. Restrict
 * the paint to the types you TYPE a value into and it goes back to meaning
 * "you will want to put a number/path/name here": every detector conf and iou,
 * every imgsz, the model paths, the input and output folders.
 *
 * The metadata, the "Important only" filter and every count are deliberately
 * untouched -- this gates the wash and the dot, nothing else.
 */
const HIGHLIGHT_TYPES = new Set(["string", "path", "list", "int", "float"]);
const showsImportant = (leaf) => !!leaf.important && HIGHLIGHT_TYPES.has(leaf.type);


/* =========================================================================
   VALUE PLUMBING — deep get/set/delete against a cloned tree
   ========================================================================= */

function clone(v) {
  if (v === null || typeof v !== "object") return v;
  if (Array.isArray(v)) return v.map(clone);
  const out = {};
  for (const k of Object.keys(v)) out[k] = clone(v[k]);
  return out;
}

/** Structural equality for the JSON-shaped values a YAML file can hold. */
function same(a, b) {
  if (a === b) return true;
  if (a === null || b === null || a === undefined || b === undefined) return false;
  if (typeof a !== typeof b) return false;
  if (typeof a !== "object") return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(a)) {
    if (a.length !== b.length) return false;
    return a.every((x, i) => same(x, b[i]));
  }
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && same(a[k], b[k]));
}

/* `parts` is carried on every leaf descriptor rather than re-split from the
   dotted path, so a YAML key that itself contains a dot still reads and writes
   correctly (only its metadata lookup would miss, and that degrades to a
   derived label). */
function getAt(tree, parts) {
  let cur = tree;
  for (const k of parts) {
    if (cur === null || typeof cur !== "object" || !(k in cur)) return undefined;
    cur = cur[k];
  }
  return cur;
}

function hasAt(tree, parts) {
  let cur = tree;
  for (const k of parts) {
    if (cur === null || typeof cur !== "object" || !Object.prototype.hasOwnProperty.call(cur, k)) {
      return false;
    }
    cur = cur[k];
  }
  return true;
}

function setAt(tree, parts, value) {
  let cur = tree;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const k = parts[i];
    if (cur[k] === null || typeof cur[k] !== "object" || Array.isArray(cur[k])) cur[k] = {};
    cur = cur[k];
  }
  cur[parts[parts.length - 1]] = value;
}

function delAt(tree, parts) {
  const stack = [];
  let cur = tree;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const k = parts[i];
    if (cur === null || typeof cur !== "object" || !(k in cur)) return;
    stack.push([cur, k]);
    cur = cur[k];
  }
  if (cur === null || typeof cur !== "object") return;
  delete cur[parts[parts.length - 1]];
  // Drop parents that the deletion just emptied, so a reset never leaves a
  // bare `modules: {plant_detector: {}}` husk behind in the written YAML.
  for (let i = stack.length - 1; i >= 0; i -= 1) {
    const [parent, key] = stack[i];
    if (parent[key] && typeof parent[key] === "object" && !Array.isArray(parent[key])
        && Object.keys(parent[key]).length === 0) {
      delete parent[key];
    } else break;
  }
}

/** Every leaf of a nested object as [dottedPath, parts, value]. */
function walkLeaves(node, parts, out) {
  if (node !== null && typeof node === "object" && !Array.isArray(node)
      && Object.keys(node).length > 0) {
    for (const k of Object.keys(node)) walkLeaves(node[k], parts.concat(k), out);
    return out;
  }
  if (parts.length) out.push({ path: parts.join("."), parts, value: node });
  return out;
}


/* =========================================================================
   DISPLAY HELPERS
   ========================================================================= */

/** "min_frame_cm" -> "Min frame cm" — the fallback when metadata is missing. */
function humanize(key) {
  return String(key)
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^./, (c) => c.toUpperCase());
}

/** A value as the short mono string used in diffs, hints and tooltips. */
function show(v) {
  if (v === undefined) return "—";
  if (v === null) return "null";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") return String(v);
  if (Array.isArray(v)) return `[${v.map(show).join(", ")}]`;
  if (typeof v === "object") return JSON.stringify(v);
  return v === "" ? '""' : String(v);
}

function clampByte(n) { return Math.max(0, Math.min(255, Math.round(Number(n) || 0))); }

function rgbToHex(rgb) {
  const [r, g, b] = Array.isArray(rgb) ? rgb : [0, 0, 0];
  return `#${[r, g, b].map((c) => clampByte(c).toString(16).padStart(2, "0")).join("")}`;
}

function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{6}|[0-9a-f]{3})$/i.exec(String(hex || "").trim());
  if (!m) return null;
  let h = m[1];
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

/**
 * Coerce free text for a `string`-typed setting.
 *
 * WHY this is not just `String(text)`: six of these keys are the "auto"-or-a-
 * real-value knobs (compute.devices, compute.io_workers, …). Config accepts
 * `devices: [0, 1]` but REJECTS the string `"[0,1]"`, and accepts io_workers as
 * either "auto" or a number — so typing `[0, 1]` has to become a real list and
 * `12` has to become a real int, or the save is refused for a reason the user
 * cannot see in the field they typed into.
 */
function coerceScalar(text) {
  const s = String(text);
  const t = s.trim();
  if (t === "") return "";
  if (t === "null" || t === "~") return null;
  if (t === "true") return true;
  if (t === "false") return false;
  if ((t.startsWith("[") && t.endsWith("]")) || (t.startsWith("{") && t.endsWith("}"))) {
    try { return JSON.parse(t); } catch { /* not JSON — keep the literal text */ }
  }
  if (/^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(t)) {
    const n = Number(t);
    if (Number.isFinite(n)) return n;
  }
  return s;
}

/** Same coercion for one entry of a `list`. */
function coerceItem(text) {
  const v = coerceScalar(text);
  return v === "" ? String(text) : v;
}


/* =========================================================================
   TOASTS
   The stack is shared app chrome; reuse whatever is already in the document
   so two tabs never stack two containers on top of each other.
   ========================================================================= */
function toast(message, { kind = "", title = "", ms = 4200 } = {}) {
  let host = document.querySelector(".toasts");
  if (!host) {
    host = el("div.toasts");
    document.body.appendChild(host);
  }
  const node = el(`div.toast${kind ? `.${kind}` : ""}`,
    title ? el("span.t", title) : null,
    message);
  host.appendChild(node);
  setTimeout(() => {
    node.classList.add("leaving");
    setTimeout(() => node.remove(), 220);
  }, ms);
  return node;
}


/* =========================================================================
   YAML PRESENTATION — highlighting + a line diff for the preview pane
   ========================================================================= */

/** Colorize one YAML line into the .yamlview token classes (.k/.s/.n/.c). */
function highlightYamlLine(line) {
  const hash = line.indexOf("#");
  let body = line, comment = "";
  if (hash >= 0) {
    // Only treat a '#' as a comment when it starts the line or follows a space,
    // so `color: '#ff0000'` keeps its hex intact.
    if (hash === 0 || /\s/.test(line[hash - 1])) {
      body = line.slice(0, hash);
      comment = line.slice(hash);
    }
  }
  const m = /^(\s*-?\s*)([A-Za-z0-9_.\-"' ]+?)(:)(\s*)(.*)$/.exec(body);
  let html;
  if (m) {
    const [, lead, key, colon, gap, rest] = m;
    html = `${esc(lead)}<span class="k">${esc(key)}</span>${colon}${esc(gap)}${valueSpan(rest)}`;
  } else {
    html = valueSpan(body);
  }
  return html + (comment ? `<span class="c">${esc(comment)}</span>` : "");
}

function valueSpan(rest) {
  if (!rest) return "";
  const t = rest.trim();
  if (!t) return esc(rest);
  if (/^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(t) || t === "true" || t === "false" || t === "null") {
    return `<span class="n">${esc(rest)}</span>`;
  }
  return `<span class="s">${esc(rest)}</span>`;
}

/**
 * Minimal LCS line diff. The preview the server returns is the exact YAML it
 * WOULD write, so diffing it against a preview of the untouched values shows
 * the real byte-level consequence of the edits — including the reordering or
 * re-quoting a value change can cause, which a key-by-key diff would miss.
 */
function diffLines(a, b) {
  const n = a.length, m = b.length;
  // Trim the common head and tail first: on a 350-line file with three edits
  // that leaves an LCS table of a few rows instead of 120k cells.
  let head = 0;
  while (head < n && head < m && a[head] === b[head]) head += 1;
  let tail = 0;
  while (tail < n - head && tail < m - head && a[n - 1 - tail] === b[m - 1 - tail]) tail += 1;
  const A = a.slice(head, n - tail), B = b.slice(head, m - tail);

  const out = [];
  for (let i = 0; i < head; i += 1) out.push({ op: " ", text: a[i] });

  const rows = A.length, cols = B.length;
  if (rows && cols) {
    const L = new Uint32Array((rows + 1) * (cols + 1));
    for (let i = rows - 1; i >= 0; i -= 1) {
      for (let j = cols - 1; j >= 0; j -= 1) {
        L[i * (cols + 1) + j] = A[i] === B[j]
          ? L[(i + 1) * (cols + 1) + (j + 1)] + 1
          : Math.max(L[(i + 1) * (cols + 1) + j], L[i * (cols + 1) + (j + 1)]);
      }
    }
    let i = 0, j = 0;
    while (i < rows && j < cols) {
      if (A[i] === B[j]) { out.push({ op: " ", text: A[i] }); i += 1; j += 1; }
      else if (L[(i + 1) * (cols + 1) + j] >= L[i * (cols + 1) + (j + 1)]) {
        out.push({ op: "-", text: A[i] }); i += 1;
      } else { out.push({ op: "+", text: B[j] }); j += 1; }
    }
    while (i < rows) { out.push({ op: "-", text: A[i] }); i += 1; }
    while (j < cols) { out.push({ op: "+", text: B[j] }); j += 1; }
  } else {
    for (const line of A) out.push({ op: "-", text: line });
    for (const line of B) out.push({ op: "+", text: line });
  }

  for (let i = m - tail; i < m; i += 1) out.push({ op: " ", text: b[i] });
  return out;
}


/* =========================================================================
   THE TAB
   ========================================================================= */

/**
 * Mount the settings editor into `root`.
 * @param {HTMLElement} root  the tab pane
 * @returns {object} controller — see the bottom of this function
 */
export function initSettings(root) {
  if (!root) throw new Error("initSettings(root): no mount element");

  /* ------------------------------------------------------------- state -- */
  const S = {
    activeGroup: null,
    meta: {},               // settings_meta.json
    sections: [],           // _sections, ordered
    file: null,             // the last GET /v1/settings body
    orig: {},               // deep clone of file.values — the on-disk baseline
    draft: {},              // deep clone of file.values — what the user is editing
    defaults: {},           // config.builtin_defaults() — SPARSE, see defaultFor()
    shape: {},              // file.effective — the union that decides which rows exist
    mtime: null,
    yamlPath: "",
    dir: "",
    readonly: false,
    leaves: [],             // ordered leaf descriptors
    rows: new Map(),        // path -> row record
    trees: new Map(),       // sectionId -> render tree node
    panes: new Map(),       // sectionId -> pane element
    subtabs: new Map(),     // sectionId -> {btn, count}
    activeSection: null,
    pane: "sections",        // "sections" | "overview" — what the right side shows
    pendingSection: null,    // a deep link that arrived before load() finished
    pendingFocus: null,
    moduleToggle: null,      // the enabled switch currently in the pane header
    importantOnly: localStorage.getItem(LS_IMPORTANT) === "1",
    // OFF by default: in the rail layout each row is 28px and carries its help as a
    // hover title, so an always-on help line would undo most of the density. The
    // Descriptions button pins it back inline and the choice is remembered.
    showDesc: localStorage.getItem(LS_DESC) === "1",
    query: "",
    openMap: readJson(LS_OPEN, {}),
    fieldErrors: {},
    errors: [],
    warnings: [],
    preview: "",
    basePreview: "",        // preview of the UNTOUCHED values, for the diff view
    view: "form",           // "form" | "yaml"
    yamlMode: "diff",       // "file" | "preview" | "diff"
    busy: false,
    destroyed: false,
    listeners: new Set(),
  };

  function readJson(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch { return fallback; }
  }
  const saveOpenMap = debounce(() => {
    try { localStorage.setItem(LS_OPEN, JSON.stringify(S.openMap)); } catch { /* full / private */ }
  }, 400);

  /* ------------------------------------------------------------- shell -- */
  clear(root);

  const search = el("input", {
    type: "search", placeholder: "Search settings…", spellcheck: "false",
    "aria-label": "Search settings",
  });
  const searchBox = el("div.searchbox", search);

  const impBtn = el("button.btn.sm", { type: "button", title: "Show only the settings people actually adjust" },
    el("span.impdot"), " Important only ", el("span.mono.dim", { id: "lm3-imp-n" }, ""));
  const descBtn = el("button.btn.sm.ghost", { type: "button", title: "Show or hide the one-line explanations" },
    "Descriptions");
  const legend = el("span.implegend", {
    title: "Switches and dropdowns are self-explanatory, so they are not shaded. "
         + "\"Important only\" still finds every important setting, shaded or not.",
  }, el("span.impdot"), "green = settings you might want to adjust");

  const shownCount = el("span.mono.dim", { style: { fontSize: "11.5px" } }, "");
  const fileInfo = el("span.mono.dim", { style: { fontSize: "11.5px" }, title: "" }, "");

  const viewForm = el("button.btn.sm.on", { type: "button" }, "Form");
  const viewYaml = el("button.btn.sm", { type: "button" }, "YAML");
  const viewGroup = el("div.btngroup", viewForm, viewYaml);

  const presetsBtn = el("button.btn.sm.ghost", { type: "button" }, "Presets…");
  const reloadBtn = el("button.btn.sm.ghost", { type: "button", title: "Discard edits and re-read the file from disk" }, "Reload");
  const revertBtn = el("button.btn.sm.ghost", { type: "button", disabled: true }, "Revert");
  const saveCount = el("span.mono", { style: { marginLeft: "6px" } }, "");
  const saveBtn = el("button.btn.sm.primary", { type: "button", disabled: true }, "Save", saveCount);

  const toolbar = el("div.toolbar.boxed",
    searchBox, impBtn, descBtn, legend,
    el("span.spacer"),
    shownCount, fileInfo, viewGroup, presetsBtn, reloadBtn, revertBtn, saveBtn);

  const subtabs = el("div.subtabs");

  // The toolbar is a fixed band ABOVE the two columns, not a sticky overlay they scroll behind:
  // the pane is a fill tab (app.js), so the shell owns the height and only the rail and the
  // content column scroll, each on its own.
  const header = el("div", {
    style: { flex: "0 0 auto", paddingBottom: "2px" },
  }, toolbar);

  // The rail carries the navigation the accordion used to: sections, and the
  // groups inside the open one. Scrolling is for reading a group, not for
  // getting to it.
  const railFoot = el("div.rail-ft");
  const rail = el("div.rail", subtabs, railFoot);
  /* The rail keeps its scroll position. renderRail() empties and refills it, which collapses it
     to zero height and would snap it back to the top on every filter keystroke or save; and a
     hidden tab (display:none) forgets its scrollTop entirely. So the position is tracked here and
     restored after each rebuild and whenever the tab comes back into view. */
  let railScroll = 0;
  rail.addEventListener("scroll", () => { railScroll = rail.scrollTop; }, { passive: true });
  function restoreRailScroll() {
    if (railScroll && rail.scrollTop !== railScroll) rail.scrollTop = railScroll;
  }

  const alerts = el("div");
  const diffBox = el("div");
  const paneHd = el("div.rail-hd");
  const panesHost = el("div");
  const overviewHost = el("div.modgrid-host", { hidden: true });
  const paneWrap = el("div.rail-pane", paneHd, overviewHost, panesHost);
  const yamlHost = el("div", { hidden: true, style: { padding: "12px 14px" } });

  const body = el("div", {
    style: { display: "flex", flexDirection: "column", minWidth: "0", minHeight: "0", overflow: "auto" },
  }, alerts, diffBox, paneWrap, yamlHost);

  const railwrap = el("div.railwrap", rail, body);

  // A fill tab: the shell is the pane's full height (the gutter the shell used to inherit from
  // .tabbody is applied here instead), the header takes what it needs, and .railwrap gets the
  // rest, so the rail and the content column scroll independently under a fixed toolbar.
  const shell = el("div", {
    style: {
      display: "flex", flexDirection: "column", flex: "1 1 auto", minHeight: "0", minWidth: "0",
      padding: "var(--gutter)",
    },
  }, header, railwrap);

  root.classList.add("fill");
  root.appendChild(shell);
  // Bring the rail back to where it was when the tab is shown again (class toggled by app.js).
  new MutationObserver(() => { if (root.classList.contains("active")) restoreRailScroll(); })
    .observe(root, { attributes: true, attributeFilter: ["class"] });

  /* --------------------------------------------------------- interactions */
  search.addEventListener("input", debounce(() => {
    S.query = search.value.trim();
    applyFilter();
  }, 140));
  search.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && search.value) { search.value = ""; S.query = ""; applyFilter(); }
  });

  impBtn.addEventListener("click", () => {
    S.importantOnly = !S.importantOnly;
    try { localStorage.setItem(LS_IMPORTANT, S.importantOnly ? "1" : "0"); } catch { /* ignore */ }
    applyFilter();
  });
  descBtn.addEventListener("click", () => {
    S.showDesc = !S.showDesc;
    try { localStorage.setItem(LS_DESC, S.showDesc ? "1" : "0"); } catch { /* ignore */ }
    applyDescriptions();
  });
  viewForm.addEventListener("click", () => setView("form"));
  viewYaml.addEventListener("click", () => setView("yaml"));
  presetsBtn.addEventListener("click", () => openPresets());
  reloadBtn.addEventListener("click", async () => {
    if (isDirty() && !window.confirm(
      `Discard ${dirtyPaths().length} unsaved change(s) and re-read ${S.yamlPath}?`)) return;
    await load();
  });
  revertBtn.addEventListener("click", () => revert());
  saveBtn.addEventListener("click", () => save());

  const onKey = (ev) => {
    // Ctrl/Cmd+S saves, but only while this tab is the one on screen.
    if ((ev.ctrlKey || ev.metaKey) && (ev.key === "s" || ev.key === "S")) {
      if (!root.isConnected || root.offsetParent === null) return;
      ev.preventDefault();
      save();
    }
  };
  document.addEventListener("keydown", onKey);

  const onBeforeUnload = (ev) => {
    if (!isDirty()) return undefined;
    ev.preventDefault();
    ev.returnValue = "";
    return "";
  };
  window.addEventListener("beforeunload", onBeforeUnload);


  /* =======================================================================
     LOAD
     ======================================================================= */
  async function load() {
    setBusy(true);
    clear(alerts);
    clear(panesHost);
    // The spinner and the failure card both render into panesHost, which the
    // overview owns while it is up. Take the slot back first, or a reload
    // started from "All modules" shows neither -- and a FAILED one leaves the
    // stale grid on screen looking like a successful no-op. A successful load's
    // applyFilter() puts the overview straight back.
    overviewHost.hidden = true;
    panesHost.hidden = false;
    append(panesHost, el("div.empty", el("span.spinner.lg"), el("div.t", "Reading LM3_settings.yaml…")));
    try {
      const [file, meta] = await Promise.all([api.getSettings(), api.getSettingsMeta()]);
      if (S.destroyed) return;
      S.file = file;
      S.meta = (meta && typeof meta === "object") ? meta : {};
      S.sections = normalizeSections(S.meta, file);
      S.orig = clone(file.values || {});
      S.draft = clone(file.values || {});
      S.defaults = clone(file.defaults || {});
      S.shape = clone(file.effective || file.values || {});
      S.mtime = file.mtime ?? null;
      S.yamlPath = file.yaml_path || "";
      S.dir = file.dir || "";
      S.readonly = !!file.readonly;
      S.fieldErrors = {};
      S.errors = [];
      S.warnings = [];
      S.basePreview = "";

      buildLeaves();
      buildSubtabs();
      buildPanes();
      restoreGroup();
      applyDescriptions();
      applyFilter();
      renderAlerts();
      renderDiff();
      updateDirtyUi();
      renderFileInfo();

      // Replay a deep link that arrived while this was still fetching — the
      // stage bar can navigate here before the first row exists.
      if (S.pendingFocus) focusPath(S.pendingFocus);
      else if (S.pendingSection) setSection(S.pendingSection);

      // The untouched preview is the left-hand side of the YAML diff. It costs
      // one extra call at load and turns "did I really change that?" into a
      // question the app can answer exactly.
      api.post("/v1/settings/validate", { values: S.orig })
        .then((r) => { if (!S.destroyed) { S.basePreview = r.preview || ""; if (S.view === "yaml") renderYaml(); } })
        .catch(() => { /* the diff view degrades to Preview; nothing else cares */ });

      scheduleValidate.flush ? scheduleValidate() : validateNow();
    } catch (err) {
      // Leave overview mode for good on a failure: the grid is built from the
      // S.subtabs/S.leaves this load just failed to replace, so putting it back
      // would show pre-failure data next to an error saying nothing was read.
      S.pane = "sections";
      overviewHost.hidden = true;
      panesHost.hidden = false;
      clear(panesHost);
      append(panesHost, errorCard("Could not read the settings file", err));
    } finally {
      setBusy(false);
    }
  }

  /**
   * The sections, in rail order.
   *
   * A section is either a SETUP area (project, compute, ingest, naming, timing)
   * or exactly one pipeline MODULE, carrying the `stage` key that ties it to
   * modules.js and to pipeline.STAGE_ORDER. `phase` groups them under the rail's
   * headings and `order` is the module's true run position, printed as a badge --
   * ECT reads under Leaf but is still stage 17, and the badge is what keeps that
   * fact from being lost.
   */
  function normalizeSections(meta, file) {
    const secs = Array.isArray(meta._sections) ? meta._sections : null;
    if (secs && secs.length) {
      const out = secs.map((s) => ({
        id: String(s.id),
        label: s.label || humanize(s.id),
        blurb: s.blurb || "",
        keys: Array.isArray(s.keys) && s.keys.length ? s.keys.map(String) : [String(s.id)],
        stage: s.stage ? String(s.stage) : null,
        phase: s.phase ? String(s.phase) : "setup",
        order: Number.isFinite(s.order) ? s.order : null,
        subsections: Array.isArray(s.subsections) && s.subsections.length ? s.subsections : null,
      }));
      // Sort defensively rather than trusting the file's order: a hand-edited
      // settings_meta.json must not be able to shuffle the pipeline out of
      // sequence in the one place a user reads it as a sequence.
      const phaseAt = new Map(PHASES.map((p, i) => [p.id, i]));
      out.sort((a, b) => {
        const pa = phaseAt.has(a.phase) ? phaseAt.get(a.phase) : 99;
        const pb = phaseAt.has(b.phase) ? phaseAt.get(b.phase) : 99;
        if (pa !== pb) return pa - pb;
        const ra = (MODULE_BY_KEY.get(a.stage) || {}).railOrder ?? 0;
        const rb = (MODULE_BY_KEY.get(b.stage) || {}).railOrder ?? 0;
        if (ra !== rb) return ra - rb;
        return secs.findIndex((s) => s.id === a.id) - secs.findIndex((s) => s.id === b.id);
      });
      return out;
    }
    // No metadata (or a corrupt file): fall back to one section per top-level
    // YAML key so the editor still works instead of showing nothing.
    const tree = file.effective || file.values || {};
    return Object.keys(tree).map((k) => ({
      id: k, label: humanize(k), blurb: "", keys: [k],
      stage: null, phase: "setup", order: null, subsections: null,
    }));
  }

  /** The section record for an id (null for the synthetic overview entry). */
  function sectionById(id) { return S.sections.find((s) => s.id === id) || null; }

  /**
   * The one setting promoted out of a module's list and into its pane header.
   *
   * `modules.<key>.enabled` is the most-flipped value in the file and it decides
   * whether anything else in the pane matters, so it belongs beside the title --
   * reachable from every one of the Reporter's twelve sub-sections, not buried
   * as row one of the first.
   */
  function promotedPath(sectionId) {
    const sec = sectionById(sectionId);
    return sec && sec.stage ? `modules.${sec.stage}.enabled` : null;
  }

  /** Longest-prefix match, exactly as the settings_meta contract specifies. */
  function sectionFor(path) {
    let best = null, bestLen = -1;
    for (const s of S.sections) {
      for (const k of s.keys) {
        if ((path === k || path.startsWith(`${k}.`)) && k.length > bestLen) {
          bestLen = k.length; best = s;
        }
      }
    }
    return best || S.sections[0] || null;
  }

  function metaFor(path, parts) {
    const m = S.meta[path];
    if (m && typeof m === "object") return m;
    // Unknown key (the YAML grew a setting the metadata has not caught up to):
    // derive something usable rather than dropping the row on the floor.
    const key = parts[parts.length - 1];
    return {
      label: humanize(key), help: "", type: null, important: false,
      group: parts.length > 1 ? humanize(parts[parts.length - 2]) : "",
      _derived: true,
    };
  }

  /** Infer a control type from the live value when metadata says nothing. */
  function inferType(value) {
    if (typeof value === "boolean") return "bool";
    if (typeof value === "number") return Number.isInteger(value) ? "int" : "float";
    if (Array.isArray(value)) {
      if (value.length === 3 && value.every((x) => typeof x === "number" && x >= 0 && x <= 255)) {
        return "color";
      }
      return "list";
    }
    return "string";
  }

  function buildLeaves() {
    S.leafByPath = new Map();
    // The shape comes from `effective` (defaults deep-merged with the file), so
    // rows exist for keys the YAML has never mentioned — the whole `timing`
    // block is exactly that case.
    const found = walkLeaves(S.shape, [], []);
    const seen = new Set(found.map((l) => l.path));
    // Any metadata path with no leaf in the tree still deserves a row; that is
    // how an absent-but-required key (modules.*.model.path) becomes editable.
    for (const key of Object.keys(S.meta)) {
      if (key === "_sections" || seen.has(key)) continue;
      const parts = key.split(".");
      // Skip a metadata path that names a BRANCH of the live tree.
      if ([...seen].some((p) => p.startsWith(`${key}.`))) continue;
      found.push({ path: key, parts, value: undefined });
      seen.add(key);
    }

    S.leaves = found.map((leaf) => {
      const meta = metaFor(leaf.path, leaf.parts);
      const type = meta.type || inferType(leaf.value);
      const section = sectionFor(leaf.path);
      return {
        path: leaf.path,
        parts: leaf.parts,
        key: leaf.parts[leaf.parts.length - 1],
        meta,
        type,
        important: !!meta.important,
        sectionId: section ? section.id : (S.sections[0] && S.sections[0].id),
        haystack: [
          leaf.path, meta.label || "", meta.help || "", meta.group || "",
        ].join(" ").toLowerCase(),
      };
    });
    for (const leaf of S.leaves) S.leafByPath.set(leaf.path, leaf);
  }

  /* ------------------------------------------------------ value accessors */

  /** Is a built-in default known for this path? (`defaults` is deliberately sparse.) */
  function hasDefault(parts) { return hasAt(S.defaults, parts); }
  function defaultFor(parts) { return getAt(S.defaults, parts); }

  /** What the file said before any editing. */
  function baseFor(leaf) {
    if (hasAt(S.orig, leaf.parts)) return getAt(S.orig, leaf.parts);
    if (hasDefault(leaf.parts)) return defaultFor(leaf.parts);
    return undefined;
  }

  /** What the editor currently holds. */
  function valueFor(leaf) {
    if (hasAt(S.draft, leaf.parts)) return getAt(S.draft, leaf.parts);
    if (hasDefault(leaf.parts)) return defaultFor(leaf.parts);
    return undefined;
  }

  function isChanged(leaf) { return !same(valueFor(leaf), baseFor(leaf)); }

  /**
   * Commit an edit. Writing a value that matches the baseline for a key the
   * FILE never had removes it again, so merely looking at a settings page never
   * grows LM3_settings.yaml by 100 lines of restated defaults.
   */
  function setValue(leaf, value, { silent = false } = {}) {
    const base = baseFor(leaf);
    if (same(value, base) && !hasAt(S.orig, leaf.parts)) delAt(S.draft, leaf.parts);
    else setAt(S.draft, leaf.parts, value);
    if (silent) return;
    const rec = S.rows.get(leaf.path);
    if (rec) rec.refresh();
    afterEdit();
  }

  /** Reset: to the built-in default when one exists, otherwise to the file value. */
  function resetTarget(leaf) {
    return hasDefault(leaf.parts)
      ? { value: defaultFor(leaf.parts), what: "default" }
      : { value: baseFor(leaf), what: "file value" };
  }


  /* =======================================================================
     RENDER — sub-tabs
     ======================================================================= */
  function buildSubtabs() {
    S.subtabs.clear();
    const wanted = localStorage.getItem(LS_SECTION);
    let first = null;
    for (const sec of S.sections) {
      const n = S.leaves.filter((l) => l.sectionId === sec.id).length;
      if (!n) continue;
      const nImp = S.leaves.filter((l) => l.sectionId === sec.id && l.important).length;
      S.subtabs.set(sec.id, { section: sec, n, nImp });
      if (!first) first = sec.id;
    }
    S.activeSection = S.subtabs.has(wanted) ? wanted : first;
  }

  function sectionRootOf(node) { return node.root || node; }

  /** How many settings a section may hold before the rail grows a second level. */
  const RAIL_SPLIT = 26;

  /**
   * A group this small is not worth its own rail row; it folds into the entry
   * before it. Without this the Reporter's automatic split ends in four
   * one-setting entries ("Ruler Overlay", 1) that cost a click to read a switch.
   */
  const RAIL_MERGE_MIN = 3;

  /* Rail-group keys are composites (`<section>:<path>`, plus a suffix for the
     synthetic entries). U+0001 is the delimiter because it cannot occur in a
     section id or a YAML key path -- but written raw it is invisible in a diff,
     so it lives here under a name. */
  const SEP = "\u0001";

  /** Every openKey on the path from a section root down to `path`, inclusive. */
  function chainKeys(sectionId, path) {
    const parts = String(path).split(".");
    const out = [];
    for (let i = 1; i <= parts.length; i += 1) {
      out.push(`${sectionId}:${parts.slice(0, i).join(".")}`);
    }
    return out;
  }

  function impIn(paths) {
    return paths.filter((path) => (S.leafByPath.get(path) || {}).important).length;
  }

  /**
   * Pick what the rail's SECOND level lists for a section — if anything.
   *
   * Three answers, in order of preference:
   *
   *   1. NOTHING. A section that fits on one screen has no sub-navigation at
   *      all; its nested blocks render as inline headings instead. Fifteen of
   *      the seventeen modules take this branch, which is the whole point of
   *      mirroring the pipeline: "Petiole Width" is one click and three rows,
   *      not a folder to go digging in.
   *   2. WHAT settings_meta.json SAYS. The Reporter alone holds 132 of the 299
   *      settings across masks, crops, per-leaf products and the overlay
   *      palette; those divisions are editorial, not structural, so they are
   *      declared rather than derived.
   *   3. THE TREE, split recursively -- the original behavior, now reached only
   *      by File Naming.
   */
  function chooseRailGroups(tree, sec) {
    const all = tree.allGroups || [];
    const byKey = new Map(all.map((g) => [g.key, g]));
    const total = S.leaves.filter((l) => l.sectionId === tree.sectionId).length;

    if (sec && sec.subsections) return declaredRailGroups(tree, sec, byKey);
    if (total <= RAIL_SPLIT) return [];

    // Recursive, because one pass is not enough: splitting Overlays' 92-setting
    // group yields a "Classes" child still holding 46. Descend until every entry
    // fits, or until there is nothing left to descend into.
    const expand = (g, depth) => {
      const kids = g.childKeys.map((k) => byKey.get(k)).filter(Boolean);
      if (g.n <= RAIL_SPLIT || !kids.length || depth > 4) return [g];
      const out = [];
      if (g.ownPaths.length) {
        out.push({ ...g, key: `${g.key}${SEP}own`, label: `${g.label} · main`,
          paths: g.ownPaths, n: g.ownPaths.length, imp: impIn(g.ownPaths), childKeys: [] });
      }
      for (const k of kids) out.push(...expand(k, depth + 1));
      return out;
    };

    const flat = [];
    for (const g of all.filter((x) => x.depth === 0)) flat.push(...expand(g, 0));

    // Fold the runts forward, so a one-switch group rides along with the entry
    // above it instead of being a destination of its own.
    const out = [];
    for (const g of flat) {
      const prev = out[out.length - 1];
      if (prev && g.n < RAIL_MERGE_MIN && prev.n + g.n <= RAIL_SPLIT) {
        prev.paths = prev.paths.concat(g.paths);
        prev.n += g.n;
        prev.imp += g.imp;
        prev.mergedKeys.push(g.key);
        continue;
      }
      out.push({ ...g, mergedKeys: [] });
    }
    // The rail names the entry, so the entry's own in-pane heading (and the
    // pass-through chain of ancestors above it) is redundant. A group merged IN
    // keeps its heading: it is a different block that happens to ride along.
    for (const g of out) {
      const ownPath = (String(g.key).split(":")[1] || "").split(SEP)[0];
      const hide = new Set(ownPath ? chainKeys(tree.sectionId, ownPath) : []);
      for (const mk of g.mergedKeys) {
        const p = (String(mk).split(":")[1] || "").split(SEP)[0];
        if (p) for (const k of chainKeys(tree.sectionId, p).slice(0, -1)) hide.add(k);
      }
      g.hideKeys = hide;
    }
    return out;
  }

  /**
   * Resolve a section's declared `subsections` against the rendered tree.
   *
   * Each entry names one or more group PATHS. `own: true` takes only a group's
   * own loose leaves and leaves its children to their own entries -- that is how
   * "Overlay · Drawing" gets the thirteen drawing switches without swallowing
   * the ninety-odd color rows beneath them.
   */
  function declaredRailGroups(tree, sec, byKey) {
    const out = [];
    const used = new Set();

    sec.subsections.forEach((sub, i) => {
      const paths = [];
      const hide = new Set();
      (sub.paths || []).forEach((p, j) => {
        const g = byKey.get(`${sec.id}:${p}`);
        if (!g) return;
        for (const x of (sub.own ? g.ownPaths : g.paths)) paths.push(x);
        // The FIRST path is the entry's subject, so the rail already names it and
        // its in-pane heading is noise. A second path is a different block that
        // happens to belong here, so it keeps its heading -- only its ancestors
        // (the pass-through chain down from the section root) are hidden.
        const chain = chainKeys(sec.id, p);
        for (const k of (j === 0 ? chain : chain.slice(0, -1))) hide.add(k);
      });
      if (!paths.length) return;
      for (const p of paths) used.add(p);
      out.push({
        key: `${sec.id}${SEP}sub${i}`, label: sub.label || `Group ${i + 1}`,
        blurb: sub.blurb || "", el: null, depth: 0,
        n: paths.length, imp: impIn(paths), paths, ownPaths: paths,
        childKeys: [], hideKeys: hide,
      });
    });

    // Anything the declared list forgot still needs somewhere to live. A new
    // setting must never become unreachable just because settings_meta.json has
    // not caught up -- that is the failure mode this whole rework exists to fix.
    const rest = S.leaves
      .filter((l) => l.sectionId === sec.id && !used.has(l.path))
      .map((l) => l.path);
    if (rest.length) {
      out.push({
        key: `${sec.id}${SEP}rest`, label: "Other settings",
        blurb: "Not yet filed into a group in settings_meta.json.",
        el: null, depth: 0, n: rest.length, imp: impIn(rest),
        paths: rest, ownPaths: rest, childKeys: [], hideKeys: new Set(),
      });
    }
    return out;
  }

  /** The groups the rail offers for a section, in pane order. */
  function groupsOf(sectionId) {
    const tree = S.trees.get(sectionId);
    return (tree && tree.railGroups) || [];
  }

  /** Is this module switched off right now? (null when it is not a module.) */
  function moduleEnabled(sec) {
    if (!sec || !sec.stage) return null;
    const leaf = S.leafByPath.get(`modules.${sec.stage}.enabled`);
    return leaf ? valueFor(leaf) !== false : null;
  }

  /**
   * Rebuild the rail: the overview, every section under its phase heading, and
   * under the open one its groups.
   *
   * The rail is a VERTICAL ECHO OF THE STAGE BAR — same seventeen modules, same
   * names, same order — because the alternative is what this replaced: a reader
   * looking at "Petiole" on the stage bar and having to know it was filed under
   * "Leaf Analysis" to change it. Each module prints its true run position, so
   * regrouping ECT under Leaf costs no information.
   *
   * Rendered from S.subtabs/S.trees rather than kept in sync incrementally --
   * it is at most ~40 rows and a rebuild is far cheaper than the class of bug
   * where the rail and the pane disagree about what is on screen.
   */
  function renderRail(perSection) {
    clear(subtabs);
    S.railItems = [];

    const overviewBtn = el(`button.subtab.rail-all${S.pane === "overview" ? ".active" : ""}`, {
      type: "button",
      title: "Every module on one page, with its switch and its unsaved-change count",
      onclick: () => showOverview(),
    }, el("span.stg.all", "◫"), el("span.lb", "All modules"),
    el("span.rail-dot", { title: "holds an unsaved change" }),
    el("span.n", String(S.leaves.length)));
    subtabs.appendChild(overviewBtn);
    S.railItems.push({ el: overviewBtn, paths: S.leaves.map((l) => l.path) });

    let lastPhase = null;
    for (const [id, t] of S.subtabs) {
      const sec = t.section;
      if (sec.phase !== lastPhase) {
        lastPhase = sec.phase;
        const ph = PHASES.find((p) => p.id === sec.phase);
        subtabs.appendChild(el("div.rail-phase", { title: ph ? ph.blurb : "" },
          ph ? ph.label : humanize(sec.phase)));
      }

      const active = id === S.activeSection && S.pane !== "overview";
      const n = perSection ? (perSection.get(id) || 0) : t.n;
      const count = el("span.n", String(n));
      if (perSection && n === 0) count.style.color = "var(--bad)";
      const on = moduleEnabled(sec);
      const btn = el(`button.subtab${active ? ".active" : ""}${on === false ? ".off" : ""}`, {
        type: "button",
        dataset: { sec: id },
        title: `${sec.blurb || sec.label} — ${t.n} settings, ${t.nImp} important`
          + (sec.order ? `\nRuns ${ordinal(sec.order)} of 17` : "")
          + (on === false ? "\nThis module is switched OFF and will be skipped." : ""),
        onclick: () => setSection(id),
      },
      sec.order ? el("span.stg", String(sec.order)) : el("span.stg.dot", "·"),
      el("span.lb", sec.label),
      el("span.rail-dot", { title: "holds an unsaved change" }), count);
      if (perSection && n === 0) btn.style.opacity = ".42";
      subtabs.appendChild(btn);
      S.railItems.push({ el: btn, paths: S.leaves.filter((l) => l.sectionId === id).map((l) => l.path) });
      if (!active) continue;

      // A section that fits on one screen has no second level; its nested blocks
      // are inline headings in the pane. Saying "no groups" here would imply
      // something is missing.
      const groups = groupsOf(id);
      if (!groups.length) continue;
      for (const g of groups) {
        const hits = perSection ? groupHits(g) : g.n;
        const gcount = el("span.n", String(hits));
        if (perSection && hits === 0) gcount.style.color = "var(--bad)";
        const gbtn = el(`button.rail-grp${g.key === S.activeGroup ? ".on" : ""}`, {
          type: "button",
          title: `${g.label} — ${g.n} settings${g.imp ? `, ${g.imp} important` : ""}`
            + (perSection ? ` · ${hits} match the current filter` : ""),
          onclick: () => setGroup(g.key),
        }, g.label,
        el("span.rail-dot", { title: "holds an unsaved change" }),
        g.imp ? el("span.impn", String(g.imp)) : null,
        gcount);
        // Dim, do not hide: a group that has nothing right now should still be
        // reachable, otherwise the rail reshuffles under the cursor mid-filter.
        if (perSection && hits === 0) gbtn.style.opacity = ".42";
        subtabs.appendChild(gbtn);
        S.railItems.push({ el: gbtn, paths: g.paths });
      }
    }
    refreshRailDots();
  }

  /**
   * Light pass over the rail for the unsaved-change markers.
   *
   * Separate from renderRail because setValue() fires this on every edit --
   * including continuously while a color swatch is dragged -- and rebuilding
   * sixty buttons per frame to move one dot would be silly.
   */
  function refreshRailDots() {
    const dirty = new Set(dirtyPaths());
    for (const item of S.railItems || []) {
      item.el.classList.toggle("has-dirty", item.paths.some((path) => dirty.has(path)));
    }
    // A module switched off is dimmed rather than hidden, and the class is
    // toggled here (not in renderRail) because flipping `enabled` must recolor
    // the rail without rebuilding forty buttons mid-click.
    for (const [id, t] of S.subtabs) {
      const btn = subtabs.querySelector(`.subtab[data-sec="${CSS.escape(id)}"]`);
      if (btn) btn.classList.toggle("off", moduleEnabled(t.section) === false);
    }
    const nOff = Array.from(S.subtabs.values())
      .filter((t) => moduleEnabled(t.section) === false).length;
    clear(railFoot);
    append(railFoot, [
      dirty.size ? el("span.rail-dot", { style: { visibility: "visible" } }) : null,
      el("span", dirty.size ? `${dirty.size} unsaved` : `${S.leaves.length} settings`),
      nOff ? el("span.dim", { title: "modules switched off — they will be skipped" },
        `· ${nOff} off`) : null,
    ]);
  }

  /** "3" -> "3rd". Used only in rail tooltips, so English-only is fine. */
  function ordinal(n) {
    const s = ["th", "st", "nd", "rd"];
    const v = n % 100;
    return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`;
  }

  function setSection(id) {
    if (!S.subtabs.has(id)) {
      // Called before load() finished (a deep link from the stage bar), or for a
      // module whose settings have not been read yet. Remember and replay.
      S.pendingSection = id;
      return;
    }
    S.pendingSection = null;
    S.pane = "sections";
    S.activeSection = id;
    try { localStorage.setItem(LS_SECTION, id); } catch { /* ignore */ }
    const groups = groupsOf(id);
    S.activeGroup = groups.length ? defaultGroup(groups) : null;
    syncGroupPaths();
    applyFilter();
    body.scrollTop = 0;
  }

  /** The landing page: every module at once, with its switch and its counts. */
  function showOverview() {
    S.pane = "overview";
    if (S.query) { search.value = ""; S.query = ""; }
    applyFilter();
    body.scrollTop = 0;
  }

  function syncGroupPaths() {
    const g = groupsOf(S.activeSection).find((x) => x.key === S.activeGroup);
    S.groupPaths = g ? new Set(g.paths) : null;
  }

  function setGroup(key) {
    S.activeGroup = key;
    syncGroupPaths();
    try { localStorage.setItem(LS_GROUP, `${S.activeSection}\u0000${key}`); } catch { /* ignore */ }
    applyFilter();
    body.scrollTop = 0;
  }

  /** The group a section should open on: the remembered one, else the most useful. */
  function defaultGroup(groups) {
    // Prefer the group holding the most "important" settings -- landing on a
    // stray one-row "General" because it sorts first wastes the first screen.
    let best = groups[0];
    for (const g of groups) if (g.imp > (best.imp || 0)) best = g;
    if (!best.imp) for (const g of groups) if (g.n > best.n) best = g;
    return best.key;
  }

  /** Restore the remembered group, or fall back to the section's default. */
  function restoreGroup() {
    const groups = groupsOf(S.activeSection);
    if (!groups.length) { S.activeGroup = null; return; }
    let wanted = null;
    try {
      const raw = localStorage.getItem(LS_GROUP) || "";
      const [sec, key] = raw.split("\u0000");
      if (sec === S.activeSection) wanted = key;
    } catch { /* ignore */ }
    S.activeGroup = groups.some((g) => g.key === wanted) ? wanted : defaultGroup(groups);
    syncGroupPaths();
  }

  /**
   * Name what the pane is showing, and carry the module's own switch.
   *
   * The rows themselves are filtered by matches(), which tests rail-group
   * membership alongside search and important-only -- visitGroup then collapses
   * any group left with no rows, so a rail entry can sit at ANY depth without
   * this having to reason about which <details> to open or hide.
   */
  function applyGroupVisibility() {
    clear(paneHd);
    S.moduleToggle = null;

    if (S.query) {
      append(paneHd, [
        el("span.t", "Search results"),
        el("span.p", "every section with a hit"),
      ]);
      return;
    }
    if (S.pane === "overview") {
      append(paneHd, [
        el("span.t", "All modules"),
        el("span.p", "the pipeline, in the order it runs"),
        el("span.n", `${S.leaves.length} settings`),
      ]);
      return;
    }

    const rec = S.subtabs.get(S.activeSection);
    if (!rec) return;
    const sec = rec.section;
    const g = groupsOf(S.activeSection).find((x) => x.key === S.activeGroup);

    append(paneHd, [
      sec.order ? el("span.stg.hd", { title: `Runs ${ordinal(sec.order)} of 17` },
        String(sec.order)) : null,
      el("span.t", sec.label),
      moduleSwitch(sec),
      g ? el("span.p", g.label) : null,
      el("span.n", g
        ? `${g.n} of ${rec.n} setting${rec.n === 1 ? "" : "s"}`
        : `${rec.n} setting${rec.n === 1 ? "" : "s"}`),
    ]);
    if (g && g.blurb) append(paneHd, el("div.rail-hd-sub", g.blurb));
  }

  /**
   * The module's `enabled` switch, in the pane header.
   *
   * It is the most-flipped value in the file and it decides whether anything
   * else on the page matters, so it sits beside the title -- and for the
   * Reporter that means reachable from all twelve of its sub-sections instead of
   * only the one that happens to contain it. The row itself is suppressed while
   * the header carries it (see matches()) so the value never appears twice.
   */
  function moduleSwitch(sec) {
    const path = promotedPath(sec.id);
    const leaf = path ? S.leafByPath.get(path) : null;
    if (!leaf) return null;

    const input = el("input", { type: "checkbox" });
    const txt = el("span.modsw-t");
    const wrap = el("label.switch.modsw", { title: `${path}\nSwitch this module off to skip it entirely.` },
      input);
    const sync = () => {
      const on = valueFor(leaf) !== false;
      input.checked = on;
      input.disabled = S.readonly;
      txt.textContent = on ? "on" : "off";
      txt.classList.toggle("off", !on);
    };
    input.addEventListener("change", () => setValue(leaf, input.checked));
    sync();
    S.moduleToggle = { sync, leaf };
    return el("span.modsw-wrap", wrap, txt);
  }

  /** Keep the header switch honest when the value is changed from anywhere else. */
  function syncModuleToggle() {
    if (S.moduleToggle) S.moduleToggle.sync();
  }

  /**
   * Re-read every row from the draft — and the header switch with them.
   *
   * The module's `enabled` is rendered TWICE (its row, and the switch beside the
   * pane title), so any bulk refresh that walks S.rows must walk the switch too.
   * Revert used to skip it: the rail went back to "on" while the header still
   * said "off".
   */
  function refreshAllRows() {
    for (const rec of S.rows.values()) rec.refresh();
    syncModuleToggle();
    refreshOverview();
  }


  /* =======================================================================
     RENDER — the nested tree
     ======================================================================= */

  /**
   * Turn a section's flat leaf list into the nested structure the disclosure
   * rows mirror. Two shaping rules keep the result readable:
   *   - the prefix EVERY leaf shares is stripped (the Detection tab shows
   *     "Plant Detector", not "modules > plant_detector"), and
   *   - a branch with a single branch child is merged into it, so the Exports
   *     tab never shows a `modules` group that contains only `reporter`.
   */
  function buildSectionModel(sectionId) {
    const leaves = S.leaves.filter((l) => l.sectionId === sectionId);
    if (!leaves.length) return null;

    let strip = leaves[0].parts.slice(0, -1);
    for (const l of leaves) {
      const p = l.parts.slice(0, -1);
      let i = 0;
      while (i < strip.length && i < p.length && strip[i] === p[i]) i += 1;
      strip = strip.slice(0, i);
    }

    const root = { name: "", parts: [], children: new Map(), leaves: [] };
    for (const l of leaves) {
      const rel = l.parts.slice(strip.length);
      let node = root;
      for (let i = 0; i < rel.length - 1; i += 1) {
        const k = rel[i];
        if (!node.children.has(k)) {
          node.children.set(k, {
            name: k,
            parts: strip.concat(rel.slice(0, i + 1)),
            children: new Map(),
            leaves: [],
          });
        }
        node = node.children.get(k);
      }
      node.leaves.push(l);
    }

    // Rebuilt in place rather than delete-then-set: a Map re-insert moves the
    // fused branch to the END, so `modules.ruler_cf` used to render below the
    // `report` overlay switches that ride along in its section.
    (function fuse(node) {
      const kept = new Map();
      for (const [k, child] of node.children) {
        fuse(child);
        if (child.leaves.length === 0 && child.children.size === 1) {
          const [gk, grand] = Array.from(child.children)[0];
          grand.name = `${child.name}.${gk}`;
          kept.set(grand.name, grand);
        } else {
          kept.set(k, child);
        }
      }
      node.children = kept;
    })(root);

    return { root, strip };
  }

  function groupLabelFor(node) {
    // The metadata `group` is, in practice, the friendly name of the nearest
    // branch ("Plant Detector" for modules.plant_detector.*), so use it as the
    // heading whenever every descendant agrees on one.
    const groups = new Set();
    (function collect(n) {
      for (const l of n.leaves) if (l.meta.group) groups.add(l.meta.group);
      for (const c of n.children.values()) collect(c);
    })(node);
    if (groups.size === 1) return Array.from(groups)[0];
    return node.name.split(".").map(humanize).join(" · ");
  }

  /** Do two labels say the same thing once case and separators are ignored? */
  function sameWords(a, b) {
    const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "");
    return norm(a) === norm(b);
  }

  /** Name a section's loose top-level leaves from their shared metadata group. */
  function looseLabel(node) {
    const groups = new Set(node.leaves.map((l) => l.meta.group).filter(Boolean));
    return groups.size === 1 ? Array.from(groups)[0] : "General";
  }

  function countIn(node) {
    let n = 0, imp = 0;
    (function walk(x) {
      for (const l of x.leaves) { n += 1; if (l.important) imp += 1; }
      for (const c of x.children.values()) walk(c);
    })(node);
    return { n, imp };
  }

  function buildPanes() {
    clear(panesHost);
    S.rows.clear();
    S.trees.clear();
    S.panes.clear();

    for (const sec of S.sections) {
      if (!S.subtabs.has(sec.id)) continue;
      const model = buildSectionModel(sec.id);
      if (!model) continue;

      const pane = el("div", { hidden: true, dataset: { section: sec.id } });
      const hd = el("div.sechd", { hidden: true }, sec.label);   // shown only in search mode
      pane.appendChild(hd);
      if (sec.blurb) pane.appendChild(el("p.hint.secblurb", sec.blurb));

      // The section root carries its own label so renderNode's "does this
      // heading just repeat its parent?" test has something to compare against:
      // every leaf under modules.plant_detector reports the metadata group
      // "Plant Detector", so without this the sub-groups render as
      // "PLANT DETECTOR model" and "PLANT DETECTOR dedup" instead of
      // "MODEL" and "DEDUP".
      const treeNode = { kind: "group", el: null, children: [], sectionId: sec.id, label: sec.label };
      renderNode(model.root, pane, treeNode, 0, sec.id, true);

      const empty = el("div.empty", { hidden: true },
        el("div.ic", "⌕"),
        el("div.t", "Nothing matches"),
        el("div.s", "Clear the search box, or turn off Important only."));
      pane.appendChild(empty);
      treeNode.emptyEl = empty;

      panesHost.appendChild(pane);
      S.panes.set(sec.id, pane);
      S.trees.set(sec.id, treeNode);
      treeNode.railGroups = chooseRailGroups(treeNode, sec);
    }
  }

  function renderNode(node, host, treeNode, depth, sectionId, isRoot) {
    // Leaves of this level first, then the sub-groups: a group header always
    // marks a step DOWN, never a step back up.
    if (node.leaves.length) {
      const wrap = isRoot
        ? el("div.collapse", { style: { marginBottom: node.children.size ? "9px" : "0" } })
        : host;
      let target = host;
      if (isRoot) {
        // Loose leaves at a section's root still get a frame, so they do not
        // float unbordered above the first real group.
        wrap.dataset.grp = "__general";
        wrap.appendChild(el("div.body"));
        target = wrap.lastChild;
        host.appendChild(wrap);
        const paths = node.leaves.map((l) => l.path);
        treeNode.allGroups = treeNode.allGroups || [];
        treeNode.allGroups.push({
          // "General" is the fallback, not the answer: File Naming's loose leaves
          // are all the prefix settings and the rail should say so.
          key: "__general", label: looseLabel(node), el: wrap, depth: 0,
          n: node.leaves.length, imp: node.leaves.filter((l) => l.important).length,
          paths, ownPaths: paths, childKeys: [],
        });
      }
      for (const leaf of node.leaves) {
        const rec = buildRow(leaf, depth);
        target.appendChild(rec.el);
        treeNode.children.push({ kind: "row", el: rec.el, rec, leaf });
        S.rows.set(leaf.path, rec);
      }
    }

    for (const child of node.children.values()) {
      const { n, imp } = countIn(child);
      const path = child.parts.join(".");
      const openKey = `${sectionId}:${path}`;
      const details = el("details.collapse", {
        open: S.openMap[openKey] !== false && (S.openMap[openKey] === true || depth === 0),
      });
      const badges = el("span", {
        style: { marginLeft: "auto", display: "flex", gap: "8px", alignItems: "center" },
      },
        imp ? el("span.impn", `${imp} important`) : null,
        el("span.n", String(n)));
      // meta.group is the friendly name of the nearest MODULE, so every class under
      // "Plant colors" reports "Plant colors" -- printing it on each sub-header gives
      // eleven identical headings. When it repeats the parent, the node's own name
      // is the only part carrying information.
      const gl = groupLabelFor(child);
      const parentLabel = treeNode.label || "";
      const heading = gl === parentLabel ? humanize(child.name.split(".").pop()) : gl;
      // The raw key next to the heading earns its place only when it says
      // something the heading does not. "INPUT input" and "LOGGING logging" are
      // the same word twice, and in module-sized panes that is most of them.
      const echo = sameWords(heading, child.name) ? "" : child.name;
      const summary = el("summary",
        el("span.key", heading),
        el("span.mono.dim", { style: { fontSize: "10.5px", opacity: ".7" } }, echo),
        badges);
      details.appendChild(summary);
      const inner = el("div.body");
      details.appendChild(inner);
      details.addEventListener("toggle", () => {
        // Only record the user's intent, not the forced-open state a filter set.
        if (S.query || S.importantOnly) return;
        S.openMap[openKey] = details.open;
        saveOpenMap();
      });

      const sub = { kind: "group", el: details, children: [], openKey, path,
        label: gl, root: sectionRootOf(treeNode) };
      treeNode.children.push(sub);
      renderNode(child, inner, sub, depth + 1, sectionId, false);
      host.appendChild(details);

      // Depth 0 is what the rail lists. Its <summary> is hidden in rail mode
      // (the rail already names it) and it stays open -- the rail decides which
      // group is on screen, so a second collapse layer would just hide rows twice.
      // Every group is a rail CANDIDATE, at any depth: a section like Overlays has
      // one depth-0 group holding 92 settings, which is exactly the "one screen per
      // group" promise broken. chooseRailGroups() picks the level that fits.
      if (depth === 0) { details.dataset.grp = openKey; details.open = true; }
      const paths = [];
      (function collect(x) {
        for (const l of x.leaves) paths.push(l.path);
        for (const c of x.children.values()) collect(c);
      })(child);
      const root = sectionRootOf(treeNode);
      root.allGroups = root.allGroups || [];
      root.allGroups.push({
        key: openKey, label: groupLabelFor(child), el: details, n, imp, depth, paths,
        ownPaths: child.leaves.map((l) => l.path),
        childKeys: Array.from(child.children.values())
          .map((c) => `${sectionId}:${c.parts.join(".")}`),
      });
    }
  }


  /* =======================================================================
     RENDER — one leaf row
     ======================================================================= */
  function buildRow(leaf, depth) {
    const meta = leaf.meta;
    const row = el(`div.treerow.t-${leaf.type || "string"}`, {
      dataset: { path: leaf.path },
      style: { "--depth": String(Math.min(depth, MAX_INDENT_DEPTH)) },
    });

    const keyEl = el("span.key", { title: leaf.path }, meta.label || humanize(leaf.key));
    const lbl = el("div.lbl",
      showsImportant(leaf)
        ? el("span.impdot", { title: "Important — a value people actually type in" })
        : null,
      keyEl);

    const ctl = el("div.ctl");
    const unit = el("span.unit");
    const resetBtn = el("button.reset", { type: "button", title: "" }, "↺");
    resetBtn.addEventListener("click", () => {
      const t = resetTarget(leaf);
      setValue(leaf, clone(t.value));
    });

    const desc = el("div.desc");
    const helpEl = el("span", meta.help || "");
    const dfltEl = el("span.mono", { style: { color: "var(--dim)" } }, "");
    const wasEl = el("span.mono", { style: { color: "var(--acc)" } }, "");
    append(desc, [helpEl, dfltEl, wasEl]);

    const errEl = el("div.err", { hidden: true });
    const warnEl = el("div.err", { hidden: true, style: { color: "var(--warn)" } });

    const control = makeControl(leaf, ctl);
    append(ctl, [unit, resetBtn]);
    append(row, [lbl, ctl, desc, warnEl, errEl]);

    if (meta.help) row.title = `${leaf.path}\n${meta.help}`;
    else row.title = leaf.path;

    const rec = {
      el: row, leaf, control, desc, helpEl, dfltEl, wasEl, errEl, warnEl, unit, resetBtn,
      refresh() {
        const v = valueFor(leaf);
        control.write(v);
        const changed = isChanged(leaf);
        row.classList.toggle("changed", changed);
        row.classList.toggle("row-important", showsImportant(leaf));

        // Range hint doubles as the `.unit` slot — a bare "0–1" next to a
        // number box is more use than repeating the key's suffix.
        const bits = [];
        if (meta.min !== undefined && meta.max !== undefined) bits.push(`${meta.min}–${meta.max}`);
        else if (meta.min !== undefined) bits.push(`≥ ${meta.min}`);
        else if (meta.max !== undefined) bits.push(`≤ ${meta.max}`);
        unit.textContent = bits.join(" ");

        const t = resetTarget(leaf);
        const canReset = t.value !== undefined && !same(v, t.value);
        resetBtn.hidden = !canReset;
        resetBtn.title = canReset ? `Reset to ${t.what}: ${show(t.value)}` : "";

        if (hasDefault(leaf.parts) && !same(v, defaultFor(leaf.parts))) {
          dfltEl.textContent = ` default: ${show(defaultFor(leaf.parts))}`;
        } else dfltEl.textContent = "";

        wasEl.textContent = changed ? ` was: ${show(baseFor(leaf))}` : "";

        // Client-side range violations WARN, they never block: settings_meta's
        // bounds are UI guidance, and Config is the only real authority on what
        // is acceptable.
        let warn = "";
        if (typeof v === "number") {
          if (meta.min !== undefined && v < meta.min) warn = `Below the suggested minimum of ${meta.min}.`;
          else if (meta.max !== undefined && v > meta.max) warn = `Above the suggested maximum of ${meta.max}.`;
        }
        warnEl.textContent = warn;
        warnEl.hidden = !warn;

        const errs = S.fieldErrors[leaf.path] || [];
        errEl.textContent = errs.join(" · ");
        errEl.hidden = !errs.length;
        row.classList.toggle("invalid", !!errs.length);

        updateDescVisibility(rec);
      },
    };
    rec.refresh();
    return rec;
  }

  function updateDescVisibility(rec) {
    rec.helpEl.hidden = !S.showDesc;
    const any = (S.showDesc && rec.helpEl.textContent)
      || rec.dfltEl.textContent || rec.wasEl.textContent;
    rec.desc.hidden = !any;
  }

  function applyDescriptions() {
    descBtn.classList.toggle("on", S.showDesc);
    for (const rec of S.rows.values()) updateDescVisibility(rec);
  }


  /* =======================================================================
     CONTROLS — one builder per metadata type
     Each returns { write(value) } and commits through setValue().
     ======================================================================= */
  function makeControl(leaf, ctl) {
    const disabled = () => S.readonly;
    switch (leaf.type) {
      case "bool": return boolControl(leaf, ctl, disabled);
      case "enum": return enumControl(leaf, ctl, disabled);
      case "int":
      case "float": return numberControl(leaf, ctl, disabled);
      case "color": return colorControl(leaf, ctl, disabled);
      case "list": return listControl(leaf, ctl, disabled);
      case "path": return pathControl(leaf, ctl, disabled);
      default: return stringControl(leaf, ctl, disabled);
    }
  }

  function boolControl(leaf, ctl) {
    const input = el("input", { type: "checkbox" });
    const sw = el("label.switch", input);
    const txt = el("span.mono.dim", { style: { fontSize: "11.5px" } }, "");
    input.addEventListener("change", () => setValue(leaf, input.checked));
    append(ctl, [sw, txt]);
    return {
      write(v) {
        input.checked = v === true;
        input.disabled = S.readonly;
        txt.textContent = v === true ? "on" : (v === false ? "off" : show(v));
      },
    };
  }

  function enumControl(leaf) {
    const sel = el("select");
    const choices = Array.isArray(leaf.meta.enum) ? leaf.meta.enum.slice() : [];
    let extra = null;
    const fill = (current) => {
      clear(sel);
      const list = choices.slice();
      if (current !== undefined && current !== null && !list.includes(current)) {
        extra = String(current);
        list.push(extra);
      }
      for (const c of list) {
        sel.appendChild(el("option", { value: String(c) },
          c === extra ? `${c}  (not a listed choice)` : String(c)));
      }
    };
    fill(undefined);
    sel.addEventListener("change", () => setValue(leaf, sel.value));
    return {
      node: sel,
      write(v) {
        if (!Array.from(sel.options).some((o) => o.value === String(v))) fill(v);
        sel.value = String(v ?? "");
        sel.disabled = S.readonly;
      },
      mount: true,
    };
  }

  function numberControl(leaf, ctl) {
    const isInt = leaf.type === "int";
    const input = el("input", {
      type: "number",
      step: leaf.meta.step !== undefined ? String(leaf.meta.step) : (isInt ? "1" : "any"),
      placeholder: leaf.meta.placeholder || (isInt ? "integer" : "number"),
    });
    if (leaf.meta.min !== undefined) input.min = String(leaf.meta.min);
    if (leaf.meta.max !== undefined) input.max = String(leaf.meta.max);

    const commit = () => {
      const raw = input.value.trim();
      if (raw === "") { setValue(leaf, null); return; }   // null is meaningful (ruler_cf.min_frame_cm)
      let n = Number(raw);
      if (!Number.isFinite(n)) { setValue(leaf, null); return; }
      if (isInt) n = Math.round(n);
      setValue(leaf, n);
    };
    input.addEventListener("change", commit);
    input.addEventListener("blur", commit);
    ctl.appendChild(input);

    const nullTag = el("span.mono.dim", { style: { fontSize: "11px" } }, "");
    ctl.appendChild(nullTag);
    return {
      write(v) {
        const active = document.activeElement === input;
        if (!active) input.value = (v === null || v === undefined) ? "" : String(v);
        input.disabled = S.readonly;
        nullTag.textContent = v === null ? "null" : "";
      },
    };
  }

  function stringControl(leaf, ctl) {
    const input = el("input", {
      type: "text", spellcheck: "false",
      placeholder: leaf.meta.placeholder || "",
    });
    const commit = () => setValue(leaf, coerceScalar(input.value));
    input.addEventListener("change", commit);
    input.addEventListener("blur", commit);
    ctl.appendChild(input);
    return {
      write(v) {
        if (document.activeElement !== input) {
          input.value = (v === null || v === undefined) ? "" : (typeof v === "object" ? show(v) : String(v));
        }
        input.disabled = S.readonly;
      },
    };
  }

  function pathControl(leaf, ctl) {
    const input = el("input", { type: "text", spellcheck: "false", placeholder: leaf.meta.placeholder || "" });
    const commit = () => setValue(leaf, input.value.trim());
    input.addEventListener("change", commit);
    input.addEventListener("blur", commit);

    // `<...>.path` names a FILE; everything else here names a folder. The picker
    // only lists folders (that is all POST /browse returns), so for a file
    // setting we swap the directory and keep the basename the user already has.
    const wantsFile = leaf.key === "path";
    const btn = el("button.btn.sm.ghost", { type: "button", title: "Browse for a folder" }, "Browse…");
    btn.addEventListener("click", async () => {
      const current = String(valueFor(leaf) ?? "");
      const abs = toAbsolute(current);
      const start = wantsFile ? parentOf(abs) : abs;
      const picked = await pickFolder({
        start,
        title: leaf.meta.label || leaf.path,
        note: wantsFile
          ? `Choose the folder that holds “${baseNameOf(current) || "the file"}”.`
          : leaf.meta.help || "",
      });
      if (picked === null) return;
      let next = toRelative(picked);
      if (wantsFile) {
        const base = baseNameOf(current);
        next = base ? `${next.replace(/\/+$/, "")}/${base}` : next;
      }
      input.value = next;
      setValue(leaf, next);
    });

    append(ctl, [input, btn]);
    return {
      write(v) {
        if (document.activeElement !== input) input.value = v === undefined || v === null ? "" : String(v);
        input.disabled = S.readonly;
        btn.disabled = S.readonly;
        input.title = v ? toAbsolute(String(v)) : "";
      },
    };
  }

  function colorControl(leaf, ctl) {
    const swatch = el("input", { type: "color" });
    const hex = el("input", { type: "text", spellcheck: "false", style: { maxWidth: "110px", flex: "0 1 110px" } });
    const rgb = el("span.mono.dim", { style: { fontSize: "11px" } }, "");

    const push = (arr) => setValue(leaf, arr);
    swatch.addEventListener("input", () => { const a = hexToRgb(swatch.value); if (a) push(a); });
    const commitHex = () => {
      const a = hexToRgb(hex.value);
      if (a) push(a);
      else {
        // Also accept "255, 0, 70" — that is how the value reads in the YAML.
        const nums = String(hex.value).split(/[,\s]+/).filter(Boolean).map(Number);
        if (nums.length === 3 && nums.every(Number.isFinite)) push(nums.map(clampByte));
        else { const rec = S.rows.get(leaf.path); if (rec) rec.refresh(); }
      }
    };
    hex.addEventListener("change", commitHex);
    hex.addEventListener("blur", commitHex);

    append(ctl, [swatch, hex, rgb]);
    return {
      write(v) {
        const arr = Array.isArray(v) && v.length === 3 ? v.map(clampByte) : [0, 0, 0];
        swatch.value = rgbToHex(arr);
        if (document.activeElement !== hex) hex.value = rgbToHex(arr);
        rgb.textContent = `${arr[0]}, ${arr[1]}, ${arr[2]}`;
        swatch.disabled = S.readonly;
        hex.disabled = S.readonly;
      },
    };
  }

  function listControl(leaf, ctl) {
    const chips = el("div", {
      style: { display: "flex", flexWrap: "wrap", gap: "5px", alignItems: "center",
               flex: "1 1 auto", minWidth: "0" },
    });
    const add = el("input", {
      type: "text", spellcheck: "false", placeholder: "add…",
      style: { flex: "0 1 130px", minWidth: "90px" },
    });

    const current = () => {
      const v = valueFor(leaf);
      return Array.isArray(v) ? v.slice() : (v === null || v === undefined ? [] : [v]);
    };
    const commitAdd = () => {
      const raw = add.value.trim();
      if (!raw) return;
      // One paste of "a, b, c" should become three chips, not one.
      const items = raw.split(",").map((s) => s.trim()).filter(Boolean).map(coerceItem);
      if (!items.length) return;
      add.value = "";
      setValue(leaf, current().concat(items));
    };
    add.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); commitAdd(); }
      else if (ev.key === "Backspace" && add.value === "") {
        const list = current();
        if (list.length) { list.pop(); setValue(leaf, list); }
      }
    });
    add.addEventListener("blur", commitAdd);

    // The one list that is really a list of folders gets the folder picker too.
    const isDirList = /(^|\.)dirs$/.test(leaf.path);
    const browse = isDirList
      ? el("button.btn.sm.ghost", { type: "button" }, "Add folder…")
      : null;
    if (browse) {
      browse.addEventListener("click", async () => {
        const list = current();
        const picked = await pickFolder({
          start: list.length ? toAbsolute(String(list[list.length - 1])) : S.dir,
          title: leaf.meta.label || leaf.path,
          note: leaf.meta.help || "",
        });
        if (picked === null) return;
        const rel = toRelative(picked);
        if (!list.some((x) => String(x) === rel)) setValue(leaf, list.concat([rel]));
      });
    }

    append(ctl, [chips, add, browse]);

    return {
      write(v) {
        clear(chips);
        const list = Array.isArray(v) ? v : (v === null || v === undefined ? [] : [v]);
        if (!list.length) chips.appendChild(el("span.mono.dim", { style: { fontSize: "11.5px" } }, "(empty)"));
        list.forEach((item, i) => {
          const x = el("span.x", { title: "Remove" }, "×");
          const chip = el("span.chip", { title: show(item) }, String(item), S.readonly ? null : x);
          x.addEventListener("click", (ev) => {
            ev.stopPropagation();
            const next = list.slice();
            next.splice(i, 1);
            setValue(leaf, next);
          });
          chips.appendChild(chip);
        });
        add.disabled = S.readonly;
        add.hidden = S.readonly;
        if (browse) browse.disabled = S.readonly;
      },
    };
  }

  /* enumControl returns a node that still has to be inserted; do it here so the
     switch above stays a plain one-liner per type. */
  const _origMakeControl = makeControl;
  function makeControlWrapped(leaf, ctl) {
    const c = _origMakeControl(leaf, ctl);
    if (c && c.mount && c.node && !c.node.parentNode) ctl.insertBefore(c.node, ctl.firstChild);
    return c;
  }
  makeControl = makeControlWrapped;   // eslint-disable-line no-func-assign


  /* =======================================================================
     PATH HELPERS — relative paths stay relative
     ======================================================================= */
  function toAbsolute(p) {
    const s = String(p || "").trim();
    if (!s || s === "auto") return S.dir;
    if (s.startsWith("/")) return s;
    return `${S.dir.replace(/\/+$/, "")}/${s}`;
  }

  /**
   * The shipped file writes `models/plant_detector/model.onnx`, not an absolute
   * path, because those paths resolve against the settings directory and the
   * checkout has to stay portable. So a picked path under that directory is
   * written back relative, exactly as a human would have typed it.
   */
  function toRelative(p) {
    const s = String(p || "");
    const base = S.dir.replace(/\/+$/, "");
    if (base && s.startsWith(`${base}/`)) return s.slice(base.length + 1);
    if (base && s === base) return ".";
    return s;
  }

  function parentOf(p) {
    const s = String(p || "").replace(/\/+$/, "");
    const i = s.lastIndexOf("/");
    return i > 0 ? s.slice(0, i) : (i === 0 ? "/" : S.dir);
  }

  function baseNameOf(p) {
    const s = String(p || "").replace(/\/+$/, "");
    const i = s.lastIndexOf("/");
    return i >= 0 ? s.slice(i + 1) : s;
  }


  /* =======================================================================
     FILTERING — search + "important only"
     ======================================================================= */
  /** Search + important-only, WITHOUT the rail-group restriction. */
  function matchesBase(leaf) {
    if (S.importantOnly && !leaf.important) return false;
    if (!S.query) return true;
    const q = S.query.toLowerCase();
    // Space-separated terms are ANDed, so "plant conf" finds exactly one row.
    return q.split(/\s+/).filter(Boolean).every((term) => {
      if (leaf.haystack.includes(term)) return true;
      return show(valueFor(leaf)).toLowerCase().includes(term);
    });
  }

  function matches(leaf) {
    // Group membership is part of the filter rather than a separate pass of
    // element hiding: visitGroup already collapses a group with no surviving
    // rows, so a rail group at depth 2 hides its ancestors' other branches for
    // free and nothing has to reason about which <details> to leave open.
    // Scoped to the ACTIVE section: S.groupPaths holds only that section's
    // selected group, so applying it to every leaf made visitGroup report 0
    // surviving rows for every OTHER section, and "Important only" then painted
    // the whole rail as empty.
    if (!S.query && S.groupPaths && leaf.sectionId === S.activeSection
        && !S.groupPaths.has(leaf.path)) return false;
    // The module's own `enabled` is on screen as the header switch, so showing
    // it again as a row would be the same value twice. Only while browsing: a
    // SEARCH for "enabled" must still find it, wherever it lives.
    if (!S.query && leaf.sectionId === S.activeSection
        && leaf.path === promotedPath(S.activeSection)) return false;
    return matchesBase(leaf);
  }

  /** Is this section's promoted `enabled` on screen as the header switch right now? */
  function promotedVisible(sectionId) {
    if (S.query || sectionId !== S.activeSection) return false;
    const leaf = S.leafByPath.get(promotedPath(sectionId) || "");
    return !!leaf && matchesBase(leaf);
  }

  /** How many of a rail group's settings survive the CURRENT search / important filter. */
  function groupHits(g) {
    return g.paths.reduce((n, path) => {
      const leaf = S.leafByPath.get(path);
      return n + (leaf && matchesBase(leaf) ? 1 : 0);
    }, 0);
  }

  function applyFilter() {
    const filtering = !!(S.query || S.importantOnly);
    // Nothing in a rail pane collapses. The RAIL is the navigation now, and a
    // module pane holds three to thirteen settings -- a disclosure triangle on
    // top of that hides one line to save none. (It also fixes the case that made
    // this obvious: with no rail sub-navigation there was no group selection to
    // force the <details> open, so Project opened showing one row of fourteen.)
    const forceOpen = true;
    let shown = 0;
    const perSection = new Map();

    for (const [sectionId, tree] of S.trees) {
      // The promoted `enabled` has no row -- it is the header switch -- but it is
      // still ON SCREEN, so it counts. Without this, Metric Grounding (3 settings,
      // the only important one being `enabled`) answers "Important only" with
      // "Nothing matches" while the very setting you asked for sits in its header.
      const extra = promotedVisible(sectionId) ? 1 : 0;
      const n = visitGroup(tree, forceOpen) + extra;
      perSection.set(sectionId, n);
      shown += n;
      if (tree.emptyEl) tree.emptyEl.hidden = n > 0;
    }

    // In search mode every section with a hit is on screen at once, each under
    // its own heading — a search that only looked inside the open tab would
    // quietly hide most of the file.
    for (const [sectionId, pane] of S.panes) {
      const n = perSection.get(sectionId) || 0;
      const hd = pane.querySelector(".sechd");
      if (S.query) {
        pane.hidden = n === 0;
        if (hd) hd.hidden = false;
      } else {
        pane.hidden = sectionId !== S.activeSection;
        if (hd) hd.hidden = true;
      }
    }

    // A rail entry NAMES what the pane is showing, so the matching in-pane
    // heading -- and the pass-through chain of ancestors above it -- is
    // redundant. Everything strictly below the entry keeps its heading.
    const activeGroup = groupsOf(S.activeSection).find((x) => x.key === S.activeGroup);
    const hide = (!S.query && activeGroup && activeGroup.hideKeys) || null;
    for (const [, tree] of S.trees) {
      applyHeadings(tree, hide);
    }

    // The overview replaces the section panes rather than living beside them:
    // it is the same navigation, at a size you can read all at once.
    const overview = S.pane === "overview" && !S.query;
    panesHost.hidden = overview;
    overviewHost.hidden = !overview;
    if (overview) renderOverview();

    applyGroupVisibility();
    renderRail(filtering ? perSection : null);
    restoreRailScroll();

    const total = S.leaves.length;
    const nImp = S.leaves.filter((l) => l.important).length;
    document.getElementById("lm3-imp-n") && (document.getElementById("lm3-imp-n").textContent = String(nImp));
    impBtn.classList.toggle("on", S.importantOnly);
    shownCount.textContent = filtering ? `${shown} / ${total} shown` : `${total} settings`;

    // Which HALF of the body is on screen is setView's business, not the
    // filter's. This block used to hide yamlHost unconditionally on a query:
    // typing in the search box while the YAML view was open hid the YAML and
    // unhid panesHost -- which lives inside the already-hidden paneWrap -- so
    // both halves went dark and clearing the box could not bring either back.
    yamlHost.hidden = S.view !== "yaml";
    if (S.view === "yaml") panesHost.hidden = false;
  }

  /**
   * The landing page: one card per section, grouped by phase.
   *
   * "Where do I start" is a real question against 299 settings and a
   * twenty-two-entry rail, and the honest answer is a map. Each card carries the
   * three facts that decide whether you need to open it -- is this module ON,
   * how much is in there, and does it already hold an edit I have not saved.
   */
  function renderOverview() {
    clear(overviewHost);
    const dirty = new Set(dirtyPaths());

    for (const ph of PHASES) {
      const inPhase = S.sections.filter((s) => s.phase === ph.id && S.subtabs.has(s.id));
      if (!inPhase.length) continue;

      const grid = el("div.modgrid");
      for (const sec of inPhase) {
        const rec = S.subtabs.get(sec.id);
        const paths = S.leaves.filter((l) => l.sectionId === sec.id).map((l) => l.path);
        const nDirty = paths.filter((p) => dirty.has(p)).length;
        const on = moduleEnabled(sec);

        const card = el(`div.modcard${on === false ? ".off" : ""}`, {
          role: "button", tabindex: "0",
          onclick: () => setSection(sec.id),
          onkeydown: (ev) => {
            // The switch inside the card is focusable in its own right, and
            // Space is how you toggle a checkbox. Without this guard the card
            // swallowed it and navigated instead, so the switch could be
            // reached by keyboard but never operated by one.
            if (ev.target !== ev.currentTarget) return;
            if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); setSection(sec.id); }
          },
        },
        el("div.mc-hd",
          sec.order ? el("span.stg", { title: `Runs ${ordinal(sec.order)} of 17` }, String(sec.order))
            : el("span.stg.dot", "·"),
          el("span.mc-t", sec.label),
          on === null ? null : el("span.mc-sw", moduleSwitchFor(sec))),
        el("div.mc-b", sec.blurb || ""),
        el("div.mc-f",
          el("span.n", `${rec.n} setting${rec.n === 1 ? "" : "s"}`),
          rec.nImp ? el("span.impn", `${rec.nImp} important`) : null,
          el("span.spacer"),
          nDirty ? el("span.mc-dirty", `${nDirty} unsaved`) : null));
        grid.appendChild(card);
      }
      append(overviewHost, [
        el("div.modgrid-hd", el("span.t", ph.label), el("span.p", ph.blurb)),
        grid,
      ]);
    }
  }

  /**
   * A standalone `enabled` switch for an overview card.
   *
   * Deliberately NOT the same node as the pane header's -- that one registers
   * itself as S.moduleToggle so edits from elsewhere can sync it, and there are
   * seventeen of these on screen at once. They re-render with the grid instead.
   */
  function moduleSwitchFor(sec) {
    const leaf = S.leafByPath.get(promotedPath(sec.id) || "");
    if (!leaf) return null;
    const input = el("input", {
      type: "checkbox", checked: valueFor(leaf) !== false, disabled: S.readonly,
      dataset: { sw: sec.id },
    });
    input.addEventListener("click", (ev) => ev.stopPropagation());   // do not open the card
    input.addEventListener("keydown", (ev) => ev.stopPropagation()); // ...nor on Space
    // setValue -> afterEdit -> refreshOverview redraws the whole grid, which
    // destroys THIS element mid-event and drops focus to <body>. Put it back on
    // the replacement so keyboard users keep their place.
    input.addEventListener("change", () => {
      const hadFocus = document.activeElement === input;
      setValue(leaf, input.checked);
      if (!hadFocus) return;
      const next = overviewHost.querySelector(`input[data-sw="${CSS.escape(sec.id)}"]`);
      if (next) next.focus();
    });
    return el("label.switch.sm", { title: `${leaf.path}\nSwitch this module off to skip it.` }, input);
  }

  /** Hide the in-pane headings the rail already speaks for. */
  function applyHeadings(group, hideKeys) {
    for (const child of group.children) {
      if (child.kind === "row") continue;
      if (child.el && child.el.tagName === "DETAILS") {
        child.el.classList.toggle("hdr-off", !!(hideKeys && hideKeys.has(child.openKey)));
      }
      applyHeadings(child, hideKeys);
    }
  }

  /** Returns how many rows in this group survived the filter. */
  function visitGroup(group, filtering) {
    let n = 0;
    for (const child of group.children) {
      if (child.kind === "row") {
        const ok = matches(child.leaf);
        child.el.hidden = !ok;
        if (ok) n += 1;
      } else {
        const sub = visitGroup(child, filtering);
        child.el.hidden = sub === 0;
        if (child.el.tagName === "DETAILS") {
          // While filtering, open everything holding a hit; otherwise restore
          // whatever the user last chose for this group.
          if (filtering) child.el.open = sub > 0;
          else child.el.open = S.openMap[child.openKey] !== undefined
            ? !!S.openMap[child.openKey]
            : child.el.open;
        }
        n += sub;
      }
    }
    return n;
  }


  /* =======================================================================
     DIRTY STATE, DIFF SUMMARY, ALERTS
     ======================================================================= */
  function dirtyLeaves() { return S.leaves.filter((l) => isChanged(l)); }
  function dirtyPaths() { return dirtyLeaves().map((l) => l.path); }
  function isDirty() { return !same(S.draft, S.orig); }

  function updateDirtyUi() {
    const d = dirtyLeaves();
    const n = d.length;
    saveCount.textContent = n ? ` ${n}` : "";
    saveBtn.disabled = S.readonly || (n === 0 && same(S.draft, S.orig)) || S.busy;
    revertBtn.disabled = n === 0 && same(S.draft, S.orig);
    saveBtn.title = S.readonly
      ? "The settings file is read-only"
      : (n ? `Write ${n} change(s) to ${S.yamlPath}` : "No changes to write");

    // Mark the groups that contain an edit, so a change made in a group the rail
    // is not currently showing is never invisible. The rail's own section and
    // group markers are refreshed by refreshRailDots(), which owns those buttons.
    const dirtySet = new Set(d.map((l) => l.path));
    for (const [, tree] of S.trees) markDirty(tree, dirtySet);
    refreshRailDots();
  }

  function markDirty(group, dirtySet) {
    let any = false;
    for (const child of group.children) {
      if (child.kind === "row") { if (dirtySet.has(child.leaf.path)) any = true; }
      else if (markDirty(child, dirtySet)) any = true;
    }
    if (group.el && group.el.tagName === "DETAILS") group.el.classList.toggle("dirty", any);
    return any;
  }

  function renderDiff() {
    clear(diffBox);
    const d = dirtyLeaves();
    if (!d.length) return;

    const details = el("details.collapse", { open: true, style: { marginTop: "2px" } });
    details.appendChild(el("summary",
      el("span.key", "Unsaved changes"),
      el("span", { style: { marginLeft: "auto", display: "flex", gap: "8px", alignItems: "center" } },
        el("span.n", String(d.length)))));
    const bodyEl = el("div.body");
    for (const leaf of d) {
      const rowEl = el("div.treerow", { style: { "--depth": "0" } });
      const jump = el("span.key", { style: { cursor: "pointer", textDecoration: "underline dotted" },
                                    title: "Go to this setting" }, leaf.path);
      jump.addEventListener("click", () => focusPath(leaf.path));
      const undo = el("button.reset", { type: "button", title: "Undo this change", style: { opacity: "1" } }, "↩");
      undo.addEventListener("click", () => setValue(leaf, clone(baseFor(leaf))));
      append(rowEl, [
        el("div.lbl", showsImportant(leaf) ? el("span.impdot") : null, jump),
        el("div.ctl",
          el("span.mono.dim", { style: { textDecoration: "line-through" } }, show(baseFor(leaf))),
          el("span.dim", "→"),
          el("span.mono", { style: { color: "var(--acc)" } }, show(valueFor(leaf))),
          el("span.spacer"), undo),
      ]);
      bodyEl.appendChild(rowEl);
    }
    details.appendChild(bodyEl);
    diffBox.appendChild(details);
  }

  function renderAlerts() {
    clear(alerts);

    if (S.readonly) {
      alerts.appendChild(el("div.card.warn",
        el("h4", "Read-only"),
        el("p", "LM3 cannot write ", el("b", S.yamlPath),
          ". Editing is disabled until the file's permissions allow a write.")));
    }
    if (S.file && S.file.error) {
      alerts.appendChild(el("div.card.bad", el("h4", "Problem reading the file"), el("p", String(S.file.error))));
    }
    if (S.errors.length) {
      alerts.appendChild(el("div.card.bad",
        el("h4", `${S.errors.length} error${S.errors.length === 1 ? "" : "s"} — LM3 will not save this`),
        el("ul", S.errors.map((e) => el("li", String(e))))));
    }
    // Model-path warnings ("missing_model") are the Models tab's business; everything else stays here.
    const warns = S.warnings.filter((w) => !(w && w.code === "missing_model"));
    if (warns.length) {
      alerts.appendChild(el("div.card.warn",
        el("h4", `${warns.length} thing${warns.length === 1 ? "" : "s"} to check`),
        el("ul", warns.map((w) => {
          const item = el("li");
          append(item, w.msg || String(w));
          if (w.path) {
            const a = el("span.mono", { style: { cursor: "pointer", marginLeft: "6px", color: "var(--acc2)" } },
              w.path);
            a.addEventListener("click", () => focusPath(w.path));
            item.appendChild(a);
          }
          return item;
        }))));
    }
  }

  function errorCard(title, err) {
    const detail = err && err.body && err.body.detail;
    const lines = [];
    if (detail && typeof detail === "object") {
      if (detail.message) lines.push(detail.message);
      for (const e of detail.errors || []) lines.push(e);
    } else if (err) lines.push(err.message || String(err));
    return el("div.card.bad", el("h4", title),
      lines.length ? el("ul", lines.map((l) => el("li", l))) : el("p", "No further detail."));
  }

  function renderFileInfo() {
    if (!S.file) return;
    const size = S.file.size ? fmtBytes(S.file.size) : "";
    fileInfo.textContent = `${S.yamlPath.split("/").pop()} · ${size}`;
    fileInfo.title = `${S.yamlPath}\nlast written ${fmtTime(S.mtime, { withDate: true })}`;
  }


  /* =======================================================================
     VALIDATION
     ======================================================================= */
  const scheduleValidate = debounce(() => validateNow(), 450);

  async function validateNow() {
    if (S.destroyed) return null;
    try {
      const r = await api.post("/v1/settings/validate", { values: S.draft });
      if (S.destroyed) return null;
      S.errors = Array.isArray(r.errors) ? r.errors : [];
      S.fieldErrors = (r.field_errors && typeof r.field_errors === "object") ? r.field_errors : {};
      S.warnings = Array.isArray(r.warnings) ? r.warnings : [];
      S.preview = r.preview || "";
      refreshAllRows();
      renderAlerts();
      if (S.view === "yaml") renderYaml();
      return r;
    } catch (err) {
      // The validate route always answers 200; a failure here means the server
      // went away, which the topbar's connection dot already reports.
      if (!S.destroyed) console.warn("[lm3] settings validate failed", err);
      return null;
    }
  }

  function afterEdit() {
    renderDiff();
    updateDirtyUi();
    syncModuleToggle();
    refreshOverview();
    scheduleValidate();
    emit();
  }

  /**
   * Redraw the module cards if they are what is on screen.
   *
   * Every card shows a live switch and a live unsaved count, but the grid is
   * only built by applyFilter() -- so a Revert, a preset load, or an undo from
   * the diff list left seventeen cards showing values that were no longer true.
   */
  function refreshOverview() {
    if (S.pane === "overview" && !overviewHost.hidden) renderOverview();
  }


  /* =======================================================================
     SAVE / REVERT
     ======================================================================= */
  function revert() {
    S.draft = clone(S.orig);
    refreshAllRows();
    renderDiff();
    updateDirtyUi();
    validateNow();
    emit();
    toast("Edits discarded; the editor matches the file again.", { kind: "ok", title: "Reverted" });
  }

  async function save({ force = false } = {}) {
    if (S.readonly || S.busy) return false;
    const n = dirtyLeaves().length;
    if (!n && same(S.draft, S.orig)) return false;

    setBusy(true);
    try {
      const check = await validateNow();
      if (check && check.ok === false) {
        toast("Fix the highlighted settings first — nothing was written.",
              { kind: "bad", title: "Invalid settings" });
        const firstBad = Object.keys(S.fieldErrors)[0];
        if (firstBad) focusPath(firstBad);
        return false;
      }

      const bodyOut = { values: S.draft, backup: true };
      if (!force && S.mtime !== null && S.mtime !== undefined) bodyOut.if_mtime = S.mtime;

      const res = await api.put("/v1/settings", bodyOut);
      adoptSaved(res);
      toast(`${n} change${n === 1 ? "" : "s"} written to ${S.yamlPath.split("/").pop()}${
        res.backup ? ` (backup: ${String(res.backup).split("/").pop()})` : ""}`,
        { kind: "ok", title: "Saved" });
      root.dispatchEvent(new CustomEvent("lm3:settings-saved", {
        bubbles: true, detail: { yaml_path: S.yamlPath, values: clone(S.orig) },
      }));
      return true;
    } catch (err) {
      const detail = err && err.body && err.body.detail;
      if (err && err.status === 409 && detail && detail.conflict) {
        renderConflict(detail);
        return false;
      }
      if (detail && typeof detail === "object") {
        S.errors = Array.isArray(detail.errors) && detail.errors.length
          ? detail.errors : [detail.message || "The server refused the write."];
        S.fieldErrors = (detail.field_errors && typeof detail.field_errors === "object")
          ? detail.field_errors : {};
        refreshAllRows();
        renderAlerts();
        const firstBad = Object.keys(S.fieldErrors)[0];
        if (firstBad) focusPath(firstBad);
        toast("The settings file was NOT changed.", { kind: "bad", title: "Save refused" });
      } else {
        clear(alerts);
        alerts.appendChild(errorCard("Could not save the settings file", err));
        toast(err && err.message ? err.message : "Save failed", { kind: "bad", title: "Save failed" });
      }
      return false;
    } finally {
      setBusy(false);
    }
  }

  /** Adopt a write/restore/apply response (all three share read_settings()'s shape). */
  function adoptSaved(res) {
    if (!res || !res.values) return;
    const shapeChanged = !same(Object.keys(walkLeaves(res.effective || res.values, [], [])
      .reduce((a, l) => { a[l.path] = 1; return a; }, {})),
      Object.keys(S.leaves.reduce((a, l) => { a[l.path] = 1; return a; }, {})));

    S.file = res;
    S.orig = clone(res.values || {});
    S.draft = clone(res.values || {});
    S.shape = clone(res.effective || res.values || {});
    S.defaults = clone(res.defaults || S.defaults || {});
    S.mtime = res.mtime ?? null;
    S.readonly = !!res.readonly;
    S.errors = [];
    S.fieldErrors = {};
    S.warnings = Array.isArray(res.warnings) ? res.warnings : [];
    S.basePreview = "";

    if (shapeChanged) {
      buildLeaves();
      buildSubtabs();
      buildPanes();
      restoreGroup();
      applyDescriptions();
    } else {
      refreshAllRows();
    }
    applyFilter();
    renderAlerts();
    renderDiff();
    updateDirtyUi();
    renderFileInfo();
    emit();

    api.post("/v1/settings/validate", { values: S.orig })
      .then((r) => { if (!S.destroyed) { S.basePreview = r.preview || ""; S.preview = r.preview || ""; if (S.view === "yaml") renderYaml(); } })
      .catch(() => { /* diff view degrades gracefully */ });
  }

  function renderConflict(detail) {
    clear(alerts);
    const card = el("div.card.bad",
      el("h4", "The file changed underneath this editor"),
      el("p", detail.message || "Another window (or LM3 itself) wrote LM3_settings.yaml after this tab read it. Nothing was saved."),
      el("div.toolbar",
        el("button.btn.sm.danger", { type: "button", onclick: () => save({ force: true }) },
          "Overwrite with my edits"),
        el("button.btn.sm.ghost", { type: "button", onclick: () => load() },
          "Discard my edits and reload")));
    alerts.appendChild(card);
    alerts.scrollIntoView({ block: "nearest" });
    toast("Save blocked: the file changed on disk.", { kind: "warn", title: "Conflict" });
  }

  function setBusy(b) {
    S.busy = b;
    saveBtn.disabled = b || S.readonly || (!isDirty());
    reloadBtn.disabled = b;
    presetsBtn.disabled = b;
  }


  /* =======================================================================
     NAVIGATION
     ======================================================================= */
  function focusPath(path) {
    const rec = S.rows.get(path);
    // isConnected, not just existence: during a RELOAD the old S.rows survive
    // until buildPanes() runs after the await, so a deep link landing in that
    // window would find a stale record and play the whole reveal against a
    // detached node -- succeeding silently, and consuming the request.
    if (!rec || !rec.el.isConnected) {
      S.pendingFocus = (S.busy || !S.leaves.length) ? path : null;
      return false;
    }
    S.pendingFocus = null;
    setView("form");
    S.pane = "sections";
    if (S.query) { search.value = ""; S.query = ""; }
    if (S.importantOnly && !rec.leaf.important) {
      S.importantOnly = false;
      try { localStorage.setItem(LS_IMPORTANT, "0"); } catch { /* ignore */ }
    }
    setSection(rec.leaf.sectionId);
    // setSection lands on the section's DEFAULT rail group. If the path lives in
    // a different one, matches() filters its row out and everything below would
    // scroll to, and ring, a hidden element. Follow the path into its own group.
    // (Only the Reporter and File Naming have rail groups at all; everywhere
    // else S.groupPaths is null and this is a no-op.)
    const grp = groupsOf(rec.leaf.sectionId).find((g) => g.paths.includes(path));
    if (grp && grp.key !== S.activeGroup) {
      S.activeGroup = grp.key;
      syncGroupPaths();
    }
    for (let node = rec.el.parentElement; node && node !== panesHost; node = node.parentElement) {
      if (node.tagName === "DETAILS") node.open = true;
    }
    applyFilter();

    // A module's `enabled` has no row while its section is open — it IS the
    // header switch. Ring that instead of the suppressed row, or the most-linked
    // setting in the file is the one deep link that visibly does nothing.
    const target = (rec.el.hidden && path === promotedPath(rec.leaf.sectionId))
      ? (paneHd.querySelector(".modsw-wrap") || paneHd)
      : rec.el;
    target.scrollIntoView({ block: "center", behavior: "smooth" });
    // A one-shot ring, because scrolling a 289-row page to "somewhere in the
    // middle" without saying where is not actually an answer.
    target.style.outline = "2px solid var(--acc2)";
    target.style.outlineOffset = "-2px";
    setTimeout(() => { target.style.outline = ""; target.style.outlineOffset = ""; }, 1600);
    return true;
  }


  /* =======================================================================
     YAML VIEW
     ======================================================================= */
  /** The rail navigates the FORM; in YAML view there is nothing for it to point at. */
  function syncRailVisibility() {
    const yaml = S.view === "yaml";
    rail.hidden = yaml;
    paneWrap.hidden = yaml;
    // .railwrap is a two-column grid; with the rail display:none the body became its FIRST grid
    // item and sat in the 240px rail column. The class collapses the grid to one full-width column.
    railwrap.classList.toggle("yaml", yaml);
  }

  function setView(v) {
    S.view = v;
    viewForm.classList.toggle("on", v === "form");
    viewYaml.classList.toggle("on", v === "yaml");
    diffBox.hidden = v !== "form";
    yamlHost.hidden = v !== "yaml";
    syncRailVisibility();
    if (v === "yaml") renderYaml();
  }

  function renderYaml() {
    clear(yamlHost);
    const modes = [
      ["diff", "Diff", "Only what this edit would change"],
      ["preview", "Preview", "The exact YAML that would be written"],
      ["file", "File", "The file on disk, comments and all"],
    ];
    const group = el("div.btngroup");
    for (const [id, label, title] of modes) {
      const b = el(`button.btn.sm${S.yamlMode === id ? ".on" : ""}`, { type: "button", title },
        label);
      b.addEventListener("click", () => { S.yamlMode = id; renderYaml(); });
      group.appendChild(b);
    }
    const bar = el("div.toolbar",
      group,
      el("span.spacer"),
      el("span.hint", { style: { margin: "0" } },
        "Comments are not preserved on save — LM3 keeps a pristine copy and five rotating backups."),
      el("button.btn.sm.ghost", { type: "button", onclick: () => copyYaml() }, "Copy"));
    yamlHost.appendChild(bar);

    const pre = el("pre.yamlview", { style: { maxHeight: "none" } });
    if (S.yamlMode === "file") {
      const text = (S.file && S.file.text) || "";
      pre.innerHTML = text.split("\n").map(highlightYamlLine).join("\n");
      if (S.file && S.file.text_truncated) {
        yamlHost.appendChild(el("div.card.warn", el("p", "The file is large; only the first part is shown.")));
      }
    } else if (S.yamlMode === "preview") {
      pre.innerHTML = (S.preview || "").split("\n").map(highlightYamlLine).join("\n");
    } else {
      const base = (S.basePreview || "").split("\n");
      const next = (S.preview || "").split("\n");
      if (!S.basePreview || !S.preview) {
        pre.textContent = "Computing…";
      } else if (same(base, next)) {
        clear(pre);
        pre.appendChild(el("span.c", "No difference — the editor matches the file."));
      } else {
        const html = [];
        let run = 0;
        for (const d of diffLines(base, next)) {
          if (d.op === " ") {
            run += 1;
            // Collapse long unchanged stretches; three lines of context on each
            // side is enough to place a change in the file.
            if (run > 3) continue;
            html.push(`<span style="opacity:.45">  ${highlightYamlLine(d.text)}</span>`);
          } else {
            run = 0;
            const bg = d.op === "+" ? "rgba(74,222,128,.13)" : "rgba(248,113,113,.13)";
            const col = d.op === "+" ? "var(--acc3)" : "var(--bad)";
            html.push(`<span style="display:block;background:${bg}"><span style="color:${col}">${d.op} </span>${highlightYamlLine(d.text)}</span>`);
          }
        }
        pre.innerHTML = html.join("\n");
      }
    }
    yamlHost.appendChild(pre);
  }

  async function copyYaml() {
    const text = S.yamlMode === "file" ? ((S.file && S.file.text) || "") : (S.preview || "");
    try {
      await navigator.clipboard.writeText(text);
      toast("YAML copied to the clipboard.", { kind: "ok" });
    } catch {
      toast("The browser refused clipboard access.", { kind: "warn" });
    }
  }


  /* =======================================================================
     FOLDER PICKER
     ======================================================================= */
  function pickFolder({ start, title, note }) {
    return new Promise((resolve) => {
      let cur = start || S.dir;
      let data = null;

      const crumbs = el("div.crumbs");
      const rootsRow = el("div.toolbar", { style: { padding: "0 0 8px" } });
      const typed = el("input", { type: "text", spellcheck: "false", placeholder: "/absolute/path" });
      const goBtn = el("button.btn.sm.ghost", { type: "button" }, "Go");
      const listWrap = el("div.tblwrap.tall", { style: { margin: "0" } });
      const noteBox = el("div");
      const summary = el("div.hint", { style: { margin: "8px 0 0" } }, "");

      const useBtn = el("button.btn.sm.primary", { type: "button" }, "Use this folder");
      const cancelBtn = el("button.btn.sm.ghost", { type: "button" }, "Cancel");
      const showHidden = el("input", { type: "checkbox" });

      const panel = el("div.modal",
        el("div.panel-hd", el("span.t", "Choose a folder"),
          el("span.tools", el("span.mono.dim", { style: { fontSize: "11px" } }, title || ""))),
        el("div.panel-bd",
          note ? el("p.hint", note) : null,
          rootsRow,
          crumbs,
          el("div.toolbar", typed, goBtn,
            el("label.check", showHidden, el("span.lbl", "Show hidden"))),
          noteBox,
          listWrap,
          summary),
        el("div.panel-ft",
          el("div.toolbar", { style: { padding: "0" } },
            el("span.mono.dim", { style: { fontSize: "11.5px" }, id: "lm3-pick-cur" }, ""),
            el("span.spacer"), cancelBtn, useBtn)));

      const backdrop = el("div.backdrop", panel);
      backdrop.addEventListener("mousedown", (ev) => { if (ev.target === backdrop) done(null); });
      const esckey = (ev) => { if (ev.key === "Escape") done(null); };
      document.addEventListener("keydown", esckey);

      function done(value) {
        document.removeEventListener("keydown", esckey);
        backdrop.remove();
        resolve(value);
      }
      cancelBtn.addEventListener("click", () => done(null));
      useBtn.addEventListener("click", () => done(data && data.path ? data.path : cur));
      goBtn.addEventListener("click", () => { cur = typed.value.trim() || cur; go(); });
      typed.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") { ev.preventDefault(); cur = typed.value.trim() || cur; go(); }
      });
      showHidden.addEventListener("change", () => go());

      async function go() {
        clear(listWrap);
        listWrap.appendChild(el("div.empty", el("span.spinner.lg"), el("div.t", "Listing…")));
        try {
          data = await api.post("/v1/settings/browse", {
            path: cur, show_hidden: showHidden.checked, count_images: true, max_entries: 500,
          });
        } catch (err) {
          clear(listWrap);
          listWrap.appendChild(errorCard("Could not list that folder", err));
          return;
        }
        cur = data.path || cur;
        typed.value = data.requested || cur;
        document.getElementById("lm3-pick-cur").textContent = cur;

        clear(rootsRow);
        for (const r of data.roots || []) {
          const chip = el("span.chip", { title: r.path }, r.label);
          chip.addEventListener("click", () => { cur = r.path; go(); });
          rootsRow.appendChild(chip);
        }

        clear(crumbs);
        const segs = String(cur).split("/").filter(Boolean);
        const mk = (label, target) => {
          const a = el("a", { title: target }, label);
          a.addEventListener("click", () => { cur = target; go(); });
          return a;
        };
        crumbs.appendChild(mk("/", "/"));
        let acc = "";
        segs.forEach((s, i) => {
          acc += `/${s}`;
          if (i) crumbs.appendChild(el("span.sep", "/"));
          crumbs.appendChild(i === segs.length - 1 ? el("span.cur", s) : mk(s, acc));
        });

        clear(noteBox);
        if (data.fell_back) {
          // `path` is the nearest EXISTING ancestor — that is precisely the cue
          // to offer to create what the user typed.
          const target = data.requested;
          const mkBtn = el("button.btn.sm.accent", { type: "button" }, "Create it");
          mkBtn.addEventListener("click", async () => {
            try {
              await api.post("/v1/settings/mkdir", { path: target });
              toast(`Created ${target}`, { kind: "ok" });
              cur = target;
              go();
            } catch (err) {
              noteBox.appendChild(errorCard("Could not create that folder", err));
            }
          });
          noteBox.appendChild(el("div.card.warn",
            el("h4", "That folder does not exist yet"),
            el("p", el("span.mono", target), " is not on disk. Showing ", el("span.mono", cur), " instead."),
            el("div.toolbar", { style: { padding: "6px 0 0" } }, mkBtn)));
        }
        if (data.readable === false) {
          noteBox.appendChild(el("div.card.bad", el("h4", "Not readable"),
            el("p", "LM3 cannot list this folder with its current permissions.")));
        }
        if (data.error) {
          noteBox.appendChild(el("div.card.bad", el("h4", "Problem"), el("p", String(data.error))));
        }

        clear(listWrap);
        const list = el("ul.tablist");
        if (data.parent) {
          const up = el("li", el("span.nm", ".."), el("span.n", "up"));
          up.addEventListener("click", () => { cur = data.parent; go(); });
          list.appendChild(up);
        }
        for (const e of data.entries || []) {
          const bits = [];
          if (e.n_images) bits.push(`${e.n_images}${e.truncated ? "+" : ""} img`);
          if (e.n_dirs) bits.push(`${e.n_dirs} dir`);
          if (!e.readable) bits.push("locked");
          const li = el("li", { title: e.path },
            el("span.nm", e.name),
            el("span.n", bits.join(" · ")));
          li.addEventListener("click", () => { cur = e.path; go(); });
          list.appendChild(li);
        }
        if (!(data.entries || []).length) {
          list.appendChild(el("li", el("span.nm.dim", "(no subfolders)")));
        }
        listWrap.appendChild(list);

        const parts = [];
        parts.push(`${data.n_dirs || 0} subfolder${data.n_dirs === 1 ? "" : "s"}`);
        parts.push(`${data.n_images || 0}${data.truncated ? "+" : ""} image${data.n_images === 1 ? "" : "s"} here`);
        if (data.entries_truncated) parts.push("list truncated");
        summary.textContent = parts.join(" · ");
      }

      document.body.appendChild(backdrop);
      typed.value = cur;
      go();
    });
  }


  /* =======================================================================
     PRESETS + BACKUPS
     ======================================================================= */
  async function openPresets() {
    const presetList = el("div");
    const backupList = el("div");
    const nameInput = el("input", { type: "text", spellcheck: "false", placeholder: "my-fast-run" });
    const noteInput = el("input", { type: "text", spellcheck: "false", placeholder: "what this preset is for (optional)" });
    const saveAsBtn = el("button.btn.sm.accent", { type: "button" }, "Save current editor state");

    const closeBtn = el("button.btn.sm.ghost", { type: "button" }, "Close");
    const panel = el("div.modal", { style: { maxWidth: "min(900px,100%)" } },
      el("div.panel-hd", el("span.t", "Presets and backups")),
      el("div.panel-bd",
        el("p.hint",
          "A preset is a named copy of the whole settings tree. Loading one merges it into the editor ",
          "so you can review the diff before writing; applying one writes it to the file straight away. ",
          "Backups are made automatically on every save."),
        el("div.grid.g2",
          el("div.panel",
            el("div.panel-hd", el("span.t", "Presets")),
            el("div.panel-bd.flush", presetList),
            el("div.panel-ft",
              el("div.field.stack",
                el("label", "Save the current editor state as a preset"),
                el("div.toolbar", { style: { padding: "0" } }, nameInput, saveAsBtn),
                el("div.sub", "Letters, digits, space, dot, dash or underscore; 64 characters max."),
                noteInput))),
          el("div.panel",
            el("div.panel-hd", el("span.t", "Backups")),
            el("div.panel-bd.flush", backupList),
            el("div.panel-ft", el("span.hint", { style: { margin: "0" } },
              "Five rotating backups plus one pristine copy of the original commented file."))))),
      el("div.panel-ft", el("div.toolbar", { style: { padding: "0" } }, el("span.spacer"), closeBtn)));

    const backdrop = el("div.backdrop", panel);
    backdrop.addEventListener("mousedown", (ev) => { if (ev.target === backdrop) close(); });
    const esckey = (ev) => { if (ev.key === "Escape") close(); };
    document.addEventListener("keydown", esckey);
    function close() {
      document.removeEventListener("keydown", esckey);
      backdrop.remove();
    }
    closeBtn.addEventListener("click", close);
    document.body.appendChild(backdrop);

    saveAsBtn.addEventListener("click", async () => {
      const name = nameInput.value.trim();
      if (!name) { toast("Give the preset a name first.", { kind: "warn" }); return; }
      try {
        await api.post(`/v1/settings/presets/${encodeURIComponent(name)}`,
                       { values: S.draft, note: noteInput.value.trim() || undefined });
        nameInput.value = ""; noteInput.value = "";
        toast(`Preset “${name}” saved.`, { kind: "ok" });
        refreshPresets();
      } catch (err) {
        clear(presetList);
        presetList.appendChild(errorCard("Preset not saved", err));
        refreshPresets();
      }
    });

    async function refreshPresets() {
      clear(presetList);
      presetList.appendChild(el("div.empty", el("span.spinner")));
      let data;
      try { data = await api.get("/v1/settings/presets"); }
      catch (err) { clear(presetList); presetList.appendChild(errorCard("Could not list presets", err)); return; }
      clear(presetList);
      if (!data.presets || !data.presets.length) {
        presetList.appendChild(el("div.empty", el("div.ic", "◇"), el("div.t", "No presets yet"),
          el("div.s", `They live in ${data.dir}.`)));
        return;
      }
      const list = el("div");
      for (const p of data.presets) {
        const loadBtn = el("button.btn.sm", { type: "button", title: "Merge into the editor without writing the file" }, "Load");
        const applyBtn = el("button.btn.sm.warn", { type: "button", title: "Write this preset to the settings file now" }, "Apply");
        const delBtn = el("button.btn.sm.danger", { type: "button", title: "Delete this preset" }, "×");

        loadBtn.addEventListener("click", async () => {
          try {
            const full = await api.get(`/v1/settings/presets/${encodeURIComponent(p.name)}`);
            // MERGE, not replace: a preset written by an older LM3 could be
            // missing keys the current file has, and dropping them silently
            // would be a data loss the diff could not even show.
            S.draft = deepMerge(clone(S.orig), full.values || {});
            refreshAllRows();
            renderDiff(); updateDirtyUi(); validateNow(); emit();
            close();
            const n = dirtyLeaves().length;
            toast(n ? `“${p.name}” loaded — ${n} change${n === 1 ? "" : "s"} pending. Review, then Save.`
                    : `“${p.name}” matches the current file; nothing changed.`,
                  { kind: "ok", title: "Preset loaded" });
          } catch (err) { toast(err.message || "Could not load preset", { kind: "bad" }); }
        });

        applyBtn.addEventListener("click", async () => {
          if (!window.confirm(`Write preset “${p.name}” straight to ${S.yamlPath}?`
            + (isDirty() ? "\n\nYour unsaved edits will be discarded." : ""))) return;
          try {
            const res = await api.post(`/v1/settings/presets/${encodeURIComponent(p.name)}/apply`, {});
            adoptSaved(res);
            close();
            toast(`Preset “${p.name}” written to the settings file.`, { kind: "ok", title: "Applied" });
          } catch (err) {
            clear(presetList);
            presetList.appendChild(errorCard("Preset not applied", err));
          }
        });

        delBtn.addEventListener("click", async () => {
          if (!window.confirm(`Delete preset “${p.name}”?`)) return;
          try { await api.del(`/v1/settings/presets/${encodeURIComponent(p.name)}`); refreshPresets(); }
          catch (err) { toast(err.message || "Could not delete", { kind: "bad" }); }
        });

        list.appendChild(el("div.treerow", { style: { "--depth": "0" } },
          el("div.lbl", el("span.key", p.name)),
          el("div.ctl",
            el("span.mono.dim", { style: { fontSize: "11px" } },
              `${fmtTime(p.mtime, { withDate: true })} · ${fmtBytes(p.size)}`),
            el("span.spacer"), loadBtn, applyBtn, delBtn),
          p.note ? el("div.desc", p.note) : null));
      }
      presetList.appendChild(list);
    }

    async function refreshBackups() {
      clear(backupList);
      backupList.appendChild(el("div.empty", el("span.spinner")));
      let data;
      try { data = await api.get("/v1/settings/backups"); }
      catch (err) { clear(backupList); backupList.appendChild(errorCard("Could not list backups", err)); return; }
      clear(backupList);
      if (!data.backups || !data.backups.length) {
        backupList.appendChild(el("div.empty", el("div.ic", "⟲"), el("div.t", "No backups yet"),
          el("div.s", "One is written every time you save.")));
        return;
      }
      const list = el("div");
      for (const b of data.backups) {
        const btn = el("button.btn.sm.warn", { type: "button" }, "Restore");
        btn.addEventListener("click", async () => {
          if (!window.confirm(`Restore ${b.name} over ${S.yamlPath}?`
            + (isDirty() ? "\n\nYour unsaved edits will be discarded." : ""))) return;
          try {
            const res = await api.post("/v1/settings/restore", { name: b.name });
            adoptSaved(res);
            close();
            toast(`Restored ${b.name}.`, { kind: "ok", title: "Restored" });
          } catch (err) {
            clear(backupList);
            backupList.appendChild(errorCard("Could not restore", err));
          }
        });
        list.appendChild(el("div.treerow", { style: { "--depth": "0" } },
          el("div.lbl", el("span.key", b.name),
            b.pristine ? el("span.badge.ok", "original") : null),
          el("div.ctl",
            el("span.mono.dim", { style: { fontSize: "11px" } },
              `${fmtTime(b.mtime, { withDate: true })} · ${fmtBytes(b.size)}`),
            el("span.spacer"), btn)));
      }
      backupList.appendChild(list);
    }

    refreshPresets();
    refreshBackups();
  }

  /** Recursive merge used when loading a preset over the current tree. */
  function deepMerge(base, patch) {
    if (patch === null || typeof patch !== "object" || Array.isArray(patch)) return clone(patch);
    const out = (base !== null && typeof base === "object" && !Array.isArray(base)) ? base : {};
    for (const k of Object.keys(patch)) out[k] = deepMerge(out[k], patch[k]);
    return out;
  }


  /* =======================================================================
     CONTROLLER — what other modules are allowed to touch
     ======================================================================= */
  function emit() {
    const payload = {
      dirty: isDirty(),
      dirty_paths: dirtyPaths(),
      values: S.draft,
      yaml_path: S.yamlPath,
    };
    for (const fn of S.listeners) {
      try { fn(payload); } catch (err) { console.error("[lm3] settings listener threw", err); }
    }
    root.dispatchEvent(new CustomEvent("lm3:settings-change", { bubbles: true, detail: payload }));
  }

  const controller = {
    /** Re-read LM3_settings.yaml from the server, discarding edits. */
    reload: load,
    /** Are there unsaved edits? */
    isDirty,
    /** Dotted paths of every setting that differs from the file. */
    dirtyPaths,
    /** Current editor value at a dotted path (undefined when unknown). */
    get(path) {
      const leaf = S.leaves.find((l) => l.path === path);
      return leaf ? valueFor(leaf) : undefined;
    },
    /**
     * Set a value from outside (the primary strip above the tabs uses this for
     * input dir / output dir / temp-file location). Pass `{save:true}` to write
     * straight through.
     */
    async set(path, value, { save: doSave = false } = {}) {
      const leaf = S.leaves.find((l) => l.path === path);
      if (!leaf) return false;
      setValue(leaf, clone(value));
      if (doSave) return save();
      return true;
    },
    /** Validate + write the whole tree. Resolves true when the file changed. */
    save,
    /** Scroll to and highlight one setting. */
    focusPath,
    /** Switch the visible settings sub-tab. */
    showSection: setSection,
    /** Subscribe to editor changes; returns an unsubscribe function. */
    onChange(fn) { S.listeners.add(fn); return () => S.listeners.delete(fn); },
    /**
     * Write any unsaved edits, for a caller that must not act on a stale file (Start LM3).
     *
     * Returns true when the yaml is current afterwards. False means the write was REFUSED --
     * invalid values, a conflicting change on disk, or a read-only file -- and the caller must
     * not proceed as though its settings had been applied. `save()` cannot say this on its own:
     * it returns false for "nothing to write" too.
     */
    async flush() {
      if (!isDirty() && !dirtyLeaves().length) return true;
      if (S.readonly) return false;
      return save();
    },
    /** The full editor tree (a live reference — treat it as read-only). */
    get values() { return S.draft; },
    get yamlPath() { return S.yamlPath; },
    get readonly() { return S.readonly; },
    destroy() {
      S.destroyed = true;
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("beforeunload", onBeforeUnload);
      S.listeners.clear();
      clear(root);
    },
  };

  load();
  return controller;
}

export default initSettings;
