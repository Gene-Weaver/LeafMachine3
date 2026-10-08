/* ==========================================================================
   LM3 — Postprocessing tab
   --------------------------------------------------------------------------
   Hosts the standalone tools that run AFTER an LM3 run: today the STL
   3D-file builder, tomorrow whatever else lands in the registry.

   THE WHOLE TAB IS GENERATED FROM THE REGISTRY. Nothing here knows what
   "generate_stl_from_mask" is: the cards come from GET /v1/postprocess/tools,
   each parameter form is built from that tool's `inputs` descriptors, and the
   run panel is driven by the task SSE stream. Adding a second tool is a
   Tool(...) entry plus a runner in postprocess_api.py — this file does not
   change. That is the "obviously extensible" the brief asked for, enforced by
   construction rather than by comment.

   Backend contract (postprocess_api.py) — verified live:
     GET  /v1/postprocess/tools            -> a BARE ARRAY of tool descriptors
     GET  /v1/postprocess/context          -> {settings_path, roots, active, runs, ...}
     POST /v1/postprocess/run              -> {task_id, tool_id, state, started_at}
     GET  /v1/postprocess/tasks            -> {tasks:[snapshot], active:{tool_id:task_id}}
     GET  /v1/postprocess/tasks/{id}       -> one snapshot (?since= log cursor)
     GET  /v1/postprocess/tasks/{id}/events-> SSE hello|log|progress|ping|done
     POST /v1/postprocess/pick-masks       -> the mask chooser's candidates

   INPUT DESCRIPTOR KEYS (every key is ALWAYS present — branch on null, never
   on presence): key label type default help required accepts multi must_exist
   item_type enum min max step placeholder group important
   ========================================================================== */

import {
  api, el, clear, append,
  fmtBytes, fmtNum, fmtDuration, fmtTime, debounce,
} from "../api.js";


/* ------------------------------------------------------------------ state -- */

const S = {
  tools: [],
  context: null,
  toolId: null,          // the selected tool
  values: {},            // toolId -> {key: value}
  group: {},             // toolId -> the rail group on screen
  showHelp: false,       // inline help lines (off: the row tooltip carries it)
  task: null,            // the snapshot of the task being watched
  taskToolId: null,      // which tool that task belongs to
  logSeq: 0,             // highest log line seq rendered (SSE dedupe cursor)
  // The accumulated log lines. Module state rather than a local in watchTask,
  // because the panel is re-rendered from places that are not the SSE handler
  // (switching tools, "All tools" mid-run) and those used to redraw it with an
  // empty log -- the server strips `log` from progress frames, so nothing else
  // was holding them.
  logLines: [],
  active: {},            // {tool_id: task_id} from the server
  loaded: false,
  followLog: true,
};

const D = {};
let closeStream = null;

/** Restrained mono glyphs, not emoji — the registry's `icon` is a name. */
const ICONS = {
  cube: "◧", mesh: "◫", image: "▣", table: "▤", chart: "▥",
  file: "▢", folder: "▤", wand: "✦", ruler: "▭", leaf: "❧",
};

/* The same rule the Settings tab uses: the "important" wash is painted only on
   the types you TYPE a value into, because a green background behind a switch
   adds nothing the switch does not already say. Kept in step with
   HIGHLIGHT_TYPES in tabs/settings.js -- the two tabs share a type vocabulary,
   so they must share the rule or the app contradicts itself. */
const HIGHLIGHT_TYPES = new Set(["string", "path", "list", "int", "float"]);
const showsImportant = (inp) => {
  if (!inp.important) return false;
  // A mask "path" here is a picker modal, not a text box -- see control().
  if (inp.type === "path" && (inp.multi || inp.accepts === "mask_png")) return false;
  return HIGHLIGHT_TYPES.has(inp.type === "color_or_none" ? "color" : inp.type);
};


/* ------------------------------------------------------------- primitives -- */

function toast(msg, kind = "", title = "") {
  let host = document.querySelector(".toasts");
  if (!host) {
    host = el("div.toasts");
    document.body.appendChild(host);
  }
  const t = el(`div.toast${kind ? `.${kind}` : ""}`, title ? el("span.t", title) : null, msg);
  host.appendChild(t);
  setTimeout(() => {
    t.classList.add("leaving");
    setTimeout(() => t.remove(), 220);
  }, 3600);
}

async function copyText(text, what = "path") {
  const s = String(text ?? "");
  try {
    await navigator.clipboard.writeText(s);
    toast(`Copied ${what}`, "ok");
  } catch {
    try {
      const ta = el("textarea", { style: { position: "fixed", opacity: "0", left: "-9999px" } });
      ta.value = s;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      toast(`Copied ${what}`, "ok");
    } catch {
      toast("Could not copy", "warn");
    }
  }
}

function empty(icon, title, sub) {
  return el("div.empty", el("div.ic", icon), el("div.t", title), sub ? el("div.s", sub) : null);
}

function failure(err, where) {
  const msg = err && err.isAuth ? "the server rejected this session's token"
    : err && err.isOffline ? "the LM3 server is unreachable"
      : String((err && err.detailText) || (err && err.message) || err);
  return el("div.card.bad", el("h4", `Could not load ${where}`), el("p", msg));
}

/** FastAPI's `detail` is usually a string but is an OBJECT on 409. */
function detailOf(err) {
  const d = err && err.body && err.body.detail;
  if (d && typeof d === "object") return d;
  if (typeof d === "string") return { message: d };
  return { message: String((err && err.message) || err) };
}

const tool = (id) => S.tools.find((t) => t.id === id) || null;

/**
 * The run that currently holds the deployment, as `GET /v1/postprocess/context` reports it.
 *
 * Section 2.8: a postprocessor is REFUSED when its target is the active pipeline's run directory,
 * and allowed against any other completed run. The server is the guard (`check_target_allowed`);
 * this is only the part that says so before the user presses Run, instead of letting them find out
 * through a 409 on a tool they configured for two minutes.
 */
function activeTarget() {
  const a = S.context && S.context.active_run;
  return a && a.artifact_dir ? a : null;
}

/** Is `path` inside, equal to, or a parent of the active run's artifact directory? */
function collidesWithActiveRun(path) {
  const a = activeTarget();
  if (!a) return false;
  const target = String(path || "").replace(/\/+$/, "");
  const active = String(a.artifact_dir || "").replace(/\/+$/, "");
  if (!target || !active) return false;
  return target === active
    || target.startsWith(`${active}/`)
    || active.startsWith(`${target}/`);
}

/** The last basename of a path. */
const baseName = (p) => String(p || "").split("/").filter(Boolean).pop() || String(p || "");

/** The run (from /context) that contains an absolute path, or null. */
function runContaining(absolute) {
  const runs = (S.context && S.context.runs) || [];
  const p = String(absolute || "");
  let best = null;
  for (const r of runs) {
    const root = String(r.path || "").replace(/\/+$/, "");
    if (root && p.startsWith(`${root}/`) && (!best || root.length > best.path.length)) {
      best = { name: r.name, path: root, rel: p.slice(root.length + 1) };
    }
  }
  return best;
}

/**
 * A download URL for a produced file, IF it landed inside a run the results
 * module can serve. Outputs can be written anywhere in allowed_roots(), so
 * this legitimately returns null and the caller falls back to copy-path.
 */
function downloadUrl(absolute) {
  const hit = runContaining(absolute);
  if (!hit) return null;
  return api.url(`/v1/runs/${encodeURIComponent(hit.name)}/file`,
    { path: hit.rel, download: true, token: api.token || undefined });
}


/* ==========================================================================
   PUBLIC API
   ========================================================================== */

/** Mount the Postprocessing tab into `root` (its .tabpane). Idempotent. */
export function initPostprocess(root) {
  if (S.loaded && D.root === root) { void reloadTools(); return; }
  D.root = root;
  clear(root);
  build(root);
  S.loaded = true;
  void reloadTools();
}

/** Is a tool running right now? (For a live dot on the tab.) */
export function isRunning() {
  return Boolean(S.task && S.task.state === "running");
}

/**
 * Re-read the tool context whenever the window starts describing a different run.
 *
 * The context carries `active_run`, which is what decides whether a target is refused (section
 * 2.8), so it has to be re-read when the active run changes -- not when a client-side "New run"
 * reset fired, which is what this listener used to key off. Tool history is NOT thrown away: a
 * finished tool run belongs to the run it was run against, and a new pipeline run starting
 * elsewhere is no reason to forget it.
 */
document.addEventListener("lm3:runtime", () => {
  if (!S.loaded) return;
  void reloadTools();
});

/** The user is setting up the next run; the tool context's next-run target may move with it. */
document.addEventListener("lm3:prepare-next-run", () => {
  if (S.loaded) void reloadTools();
});

export default initPostprocess;


/* ==========================================================================
   SHELL
   ========================================================================== */

function build(root) {
  root.classList.add("postprocesstab");

  D.intro = el("div", { style: { margin: "0 0 12px" } },
    el("div.kicker", "Postprocessing"),
    el("p.hint", { style: { margin: "4px 0 0" } },
      "Standalone tools that run on the output of a finished LM3 run. Each one reads its defaults "
      + "from postprocessing_settings.yaml."));

  D.cards = el("div.toolgrid");
  D.detail = el("div", { style: { marginTop: "16px" } });

  append(root, [D.intro, D.cards, D.detail]);
}

async function reloadTools() {
  clear(D.cards);
  // renderCards() hides this strip once a tool is open. Unhide it before the
  // spinner: the early return in the catch below never reaches renderCards, so a
  // registry that fails while a tool is selected would render its error into a
  // hidden element and simply look like nothing happened.
  D.cards.hidden = false;
  D.intro.hidden = false;
  D.cards.appendChild(el("div.empty", el("span.spinner.lg")));
  try {
    const [tools, context, tasks] = await Promise.all([
      api.listTools(),
      api.get("/v1/postprocess/context"),
      api.get("/v1/postprocess/tasks", { params: { limit: 25 } }),
    ]);
    S.tools = Array.isArray(tools) ? tools : [];
    S.context = context || null;
    S.active = (tasks && tasks.active) || {};

    // Re-attach to whatever is already in flight, so reloading the page (or
    // switching tabs) never orphans a running tool.
    const runningId = Object.values(S.active)[0];
    if (runningId && (!S.task || S.task.task_id !== runningId)) {
      const owner = Object.keys(S.active).find((k) => S.active[k] === runningId);
      S.toolId = S.toolId || owner || null;
      watchTask(runningId, owner);
    } else if (!S.toolId && S.tools.length === 1) {
      // A single tool needs no menu step.
      S.toolId = S.tools[0].id;
    }
  } catch (err) {
    clear(D.cards);
    D.cards.appendChild(failure(err, "the tool registry"));
    return;
  }
  renderCards();
  renderDetail();
}

function renderCards() {
  clear(D.cards);
  // The cards ARE the tool picker while nothing is chosen -- the same role the
  // Settings tab's "All modules" grid plays. Once a tool is open the rail does
  // the switching, so leaving them on screen would be two pickers for one job.
  D.cards.hidden = !!S.toolId;
  D.intro.hidden = !!S.toolId;
  if (!S.tools.length) {
    D.cards.appendChild(empty("—", "No postprocessing tools registered",
      "Tools are declared in leafmachine3/server/postprocess_api.py."));
    return;
  }
  for (const t of S.tools) {
    const running = Boolean(S.active[t.id]);
    const selected = S.toolId === t.id;
    D.cards.appendChild(el(`div.tool${running ? ".running" : ""}`, {
      style: selected ? { borderColor: "var(--acc2)" } : null,
      onclick: () => selectTool(t.id),
    },
    el("div.tname",
      el("span.ic", { style: { color: "var(--acc)" } }, ICONS[t.icon] || ICONS.file),
      t.name,
      running ? el("span.badge.info.dotd", { style: { marginLeft: "auto" } }, "running") : null),
    el("div.tdesc", t.description),
    el("div.tfoot",
      el(`button.btn.sm${selected ? ".accent" : ".ghost"}`, {
        onclick: (e) => { e.stopPropagation(); selectTool(t.id); },
      }, selected ? "Selected" : "Configure & run"),
      el("span.spacer"),
      el("span.dim", { style: { fontSize: "11px" },
        title: `${t.inputs.length} parameters` },
      `${t.inputs.filter((i) => i.important).length} key settings`))));
  }
}

function selectTool(id) {
  if (S.toolId === id) return;
  S.toolId = id;
  renderCards();
  renderDetail();
  D.detail.scrollIntoView({ block: "nearest", behavior: "smooth" });
}


/* ==========================================================================
   PARAMETER FORM
   ========================================================================== */

/** Current form values for a tool, seeded from the registry defaults. */
function valuesFor(id) {
  if (!S.values[id]) {
    const t = tool(id);
    const v = {};
    for (const inp of (t && t.inputs) || []) {
      v[inp.key] = Array.isArray(inp.default) ? inp.default.slice() : inp.default;
    }
    S.values[id] = v;
  }
  return S.values[id];
}

function renderDetail() {
  clear(D.detail);
  const t = tool(S.toolId);
  if (!t) {
    // clear(D.detail) just detached the previous run panel. A task the SSE
    // stream is still feeding would keep rendering progress, output links and
    // its completion into that orphan -- invisible, and never reattached. So
    // rebuild the panel here and let a live run keep reporting from the tool
    // list, which is exactly where "All tools" now lands you mid-run.
    D.runPanel = el("div", { style: { marginTop: "14px" } });
    D.detail.appendChild(D.runPanel);
    if (S.task) renderRunPanel();
    return;
  }

  const v = valuesFor(t.id);

  /* -- group the inputs exactly as the registry grouped them ------------ */
  const groups = [];
  for (const inp of t.inputs) {
    const name = inp.group || "Parameters";
    let g = groups.find((x) => x.name === name);
    if (!g) { g = { name, inputs: [] }; groups.push(g); }
    g.inputs.push(inp);
  }

  /* -- master-detail, in the same shape as the Settings tab ---------------
     TOOL -> its groups, under a phase heading, exactly as Settings reads
     MODULE -> its sub-sections. A postprocessing tool IS a module as far as a
     reader is concerned, so the two tabs must not invent two different
     navigations for the same idea.

     (The pane shows one group at a time because the collage builder has 29
     inputs across 6 groups; stacked open that was ~2,750px, so the Run button
     sat four screens below the first setting.) */
  const nav = el("div.subtabs");
  const paneHd = el("div.rail-hd");
  const paneBody = el("div");
  const form = el(`div.railwrap.pp${S.showHelp ? ".showhelp" : ""}`,
    el("div.rail", nav),
    el("div.rail-pane", paneHd, paneBody));

  const bodies = new Map();
  for (const g of groups) {
    const body = el("div.body");
    for (const inp of g.inputs) body.appendChild(treeRow(t, inp, v));
    bodies.set(g.name, body);
    paneBody.appendChild(body);
  }

  const selectGroup = (name) => {
    S.group[t.id] = name;
    for (const [n, b] of bodies) b.hidden = n !== name;
    const g = groups.find((x) => x.name === name);
    clear(paneHd);
    append(paneHd, [
      el("span.t", t.name),
      el("span.p", name),
      el("span.n", `${g.inputs.length} of ${t.inputs.length} setting${t.inputs.length === 1 ? "" : "s"}`),
    ]);
    for (const b of nav.querySelectorAll(".rail-grp")) {
      b.classList.toggle("on", b.dataset.g === name);
    }
  };

  // Every tool sits in the rail, not only the selected one: switching from the
  // STL builder to the collage builder should cost the same one click that
  // switching from Plant Detector to Leaf Segmenter costs.
  nav.appendChild(el("button.subtab.rail-all", {
    type: "button", title: "Back to the tool list",
    onclick: () => { S.toolId = null; renderCards(); renderDetail(); },
  }, el("span.stg.all", "◫"), el("span.lb", "All tools"),
  el("span.n", String(S.tools.length))));
  nav.appendChild(el("div.rail-phase", { title: "Standalone tools that run on a finished LM3 run" },
    "Postprocessing"));
  for (const other of S.tools) {
    const isCur = other.id === t.id;
    const running = Boolean(S.active[other.id]);
    nav.appendChild(el(`button.subtab${isCur ? ".active" : ""}`, {
      type: "button",
      title: other.description || other.name,
      onclick: () => selectTool(other.id),
    },
    el("span.stg", ICONS[other.icon] || ICONS.file),
    el("span.lb", other.name),
    running ? el("span.impn", "run") : null,
    el("span.n", String(other.inputs.length))));

    if (!isCur) continue;
    for (const g of groups) {
      const nImportant = g.inputs.filter((i) => i.important).length;
      const nRequired = g.inputs.filter((i) => i.required).length;
      nav.appendChild(el("button.rail-grp", {
        type: "button",
        dataset: { g: g.name },
        title: `${g.name} — ${g.inputs.length} settings`
          + (nImportant ? `, ${nImportant} key` : "")
          + (nRequired ? `, ${nRequired} required` : ""),
        onclick: () => selectGroup(g.name),
      },
      g.name,
      nImportant ? el("span.impn", String(nImportant)) : null,
      el("span.n", String(g.inputs.length))));
    }
  }

  const wanted = S.group[t.id];
  selectGroup(groups.some((g) => g.name === wanted) ? wanted : (groups[0] || {}).name);

  D.runBtn = el("button.btn.primary.lg", { onclick: () => startRun(t) }, "Run tool");

  /* Section 2.8, said before the attempt rather than after it. The button is NOT disabled from
     this: the collision test here is a path comparison over one context read, while the server
     compares realpaths under the lease it can actually see -- so this warns and the server
     refuses. A client-side guard that silently disabled the button would be a second, weaker
     implementation of the rule, which is exactly what "standalone tools use the same guard as the
     HTTP API, not a parallel one" forbids. */
  const active = activeTarget();
  const collides = active && (t.target_keys || []).some((k) => collidesWithActiveRun(v[k]));
  const guard = active ? el(`div.card.${collides ? "warn" : "info"}`, { style: { margin: "0 0 12px" } },
    el("p", collides
      ? `“${active.run_name}” is running right now and this tool targets its output folder. `
        + "The server will refuse it — point the tool at a finished run instead."
      : `“${active.run_name}” is running. Tools may run against any OTHER finished run; its own `
        + "output folder is off limits until it ends.")) : null;

  const actions = el("div.toolbar.runbar",
    D.runBtn,
    el("button.btn.ghost", {
      onclick: () => {
        delete S.values[t.id];
        renderDetail();
        toast("Parameters reset to the tool's current defaults", "ok");
      },
      title: `Defaults come from ${t.settings_path}`,
    }, "Reset to defaults"),
    // This tab rendered every help line unconditionally, which doubled the height of
    // all 29 rows. The text stays on the row as a tooltip either way.
    el(`button.btn.ghost${S.showHelp ? ".on" : ""}`, {
      title: "Show or hide the one-line explanations",
      onclick: (e) => {
        S.showHelp = !S.showHelp;
        form.classList.toggle("showhelp", S.showHelp);
        e.currentTarget.classList.toggle("on", S.showHelp);
      },
    }, "Descriptions"),
    el("span.spacer"),
    el("span.implegend", {
      title: "Switches and dropdowns are self-explanatory, so they are not shaded.",
    }, el("span.impdot"), "green = you'll want to type a value here"));

  D.runPanel = el("div", { style: { marginTop: "14px" } });

  D.detail.appendChild(el("div.panel",
    el("div.panel-hd",
      // The way back, where the eye lands first. The rail's "All tools" row does the same thing
      // but sits below the fold of the form and reads as a filter, so on its own the detail view
      // felt one-way.
      el("button.btn.sm.ghost", {
        type: "button", title: "Back to the tool list",
        style: { marginRight: "10px", flex: "0 0 auto" },
        onclick: () => { S.toolId = null; renderCards(); renderDetail(); },
      }, "← All tools"),
      el("span.ic", { style: { color: "var(--acc)" } }, ICONS[t.icon] || ICONS.file),
      el("span.t", t.name),
      el("span.tools",
        el("span.badge", { title: t.module }, t.settings_key))),
    el("div.panel-bd",
      el("p.hint", { style: { margin: "0 0 12px" } }, t.description),
      t.outputs_description
        ? el("div.card.info", el("h4", "What this produces"), el("p", t.outputs_description))
        : null,
      guard,
      form,
      actions,
      el("details.collapse", { style: { marginTop: "10px" } },
        el("summary", el("span.key", "Command line equivalent")),
        el("div.body", el("div", {
          style: { padding: "10px 12px", font: "12px/1.6 var(--mono)", color: "var(--mute)",
                   overflowWrap: "anywhere" },
        }, t.cli))))));

  D.detail.appendChild(D.runPanel);
  renderRunPanel();
}

/**
 * One parameter row, in the same idiom as the Settings tab: a `.treerow` whose
 * `.row-important` variant is the green shading — painted only on typed inputs,
 * see showsImportant().
 */
function treeRow(t, inp, v) {
  const kind = inp.type === "color_or_none" ? "color" : (inp.type || "string");
  const lit = showsImportant(inp);
  const row = el(`div.treerow.t-${kind}${lit ? ".row-important" : ""}`);

  const label = el("div.lbl",
    lit ? el("span.impdot") : null,
    el("span.key", { title: inp.key }, inp.label || inp.key),
    inp.required ? el("span", { style: { color: "var(--acc)" }, title: "required" }, "*") : null);

  const ctl = el("div.ctl");
  ctl.appendChild(control(t, inp, v, row));

  append(row, [label, ctl]);
  if (inp.help) row.appendChild(el("div.desc", inp.help));
  row.title = inp.help ? `${inp.key}\n${inp.help}` : inp.key;
  row.dataset.key = inp.key;
  return row;
}

/** Build the right control for a descriptor. Everything branches HERE. */
function control(t, inp, v, row) {
  const set = (val) => { v[inp.key] = val; markDirty(t, inp, v, row); };

  // A mask path field is the mask picker, not a text box — whether it takes many masks (the STL
  // builder's batch) or exactly one (a collage's primary mask).
  if (inp.type === "path" && (inp.multi || inp.accepts === "mask_png")) return maskField(t, inp, v, row);
  if (inp.type === "path") return pathField(inp, v, set);
  if (inp.type === "bool") return boolField(inp, v, set);
  if (inp.type === "enum") return enumField(inp, v, set);
  if (inp.type === "color" || inp.type === "color_or_none") return colorField(inp, v, set);
  if (inp.type === "list") return listField(inp, v, set);
  if (inp.type === "int" || inp.type === "float") return numberField(inp, v, set);
  return stringField(inp, v, set);
}

function markDirty(t, inp, v, row) {
  const isDefault = JSON.stringify(v[inp.key] ?? null) === JSON.stringify(inp.default ?? null);
  row.classList.toggle("changed", !isDefault);
  row.classList.remove("invalid");
  const err = row.querySelector(".err");
  if (err) err.remove();
}

function boolField(inp, v, set) {
  const box = el("label.switch",
    el("input", {
      type: "checkbox",
      checked: v[inp.key] ? true : undefined,
      onchange: (e) => set(e.target.checked),
    }));
  return box;
}

function enumField(inp, v, set) {
  return el("select", { onchange: (e) => set(e.target.value) },
    ...(inp.enum || []).map((o) =>
      el("option", { value: o, selected: v[inp.key] === o || undefined }, String(o))));
}

function stringField(inp, v, set) {
  return el("input", {
    type: "text",
    value: v[inp.key] === null || v[inp.key] === undefined ? "" : String(v[inp.key]),
    placeholder: inp.placeholder || "",
    oninput: (e) => set(e.target.value),
  });
}

function numberField(inp, v, set) {
  const num = el("input", {
    type: "number",
    value: v[inp.key] === null || v[inp.key] === undefined ? "" : String(v[inp.key]),
    min: inp.min !== null && inp.min !== undefined ? String(inp.min) : undefined,
    max: inp.max !== null && inp.max !== undefined ? String(inp.max) : undefined,
    step: inp.step !== null && inp.step !== undefined ? String(inp.step) : undefined,
  });

  // A slider only where the span is small enough for one to mean something:
  // color_tolerance 0-255 and simplify 0-100 get one, length_mm 0.1-5000 and
  // min_area_px 0-1e9 would just be a twitchy pixel-per-20-units mess.
  // A coarse `step` also qualifies a wide range — max_dim_px 256-30000 by 100
  // is 297 stops, which drags meaningfully; the number box still takes any value.
  const span = Number.isFinite(inp.min) && Number.isFinite(inp.max) ? inp.max - inp.min : 0;
  const stops = Number.isFinite(inp.step) && inp.step > 0 ? span / inp.step : Infinity;
  const spanned = span > 0 && (span <= 512 || stops <= 512);
  const slider = spanned ? el("input", {
    type: "range",
    min: String(inp.min), max: String(inp.max),
    step: inp.step !== null && inp.step !== undefined ? String(inp.step) : "any",
    value: String(v[inp.key] ?? inp.min),
  }) : null;

  const push = (raw) => {
    if (raw === "") { set(null); return; }
    const n = inp.type === "int" ? parseInt(raw, 10) : parseFloat(raw);
    if (Number.isNaN(n)) return;
    set(n);
  };
  num.addEventListener("input", (e) => {
    push(e.target.value);
    if (slider) slider.value = e.target.value;
  });
  if (slider) {
    slider.addEventListener("input", (e) => {
      num.value = e.target.value;
      push(e.target.value);
    });
  }

  return el("span", { style: { display: "flex", alignItems: "center", gap: "10px",
                               flex: "1 1 auto", minWidth: "0" } },
  num,
  slider,
  Number.isFinite(inp.min) && Number.isFinite(inp.max)
    ? el("span.unit", { title: "allowed range" },
      `${fmtNum(inp.min, { digits: null })} – ${fmtNum(inp.max, { compact: true })}`)
    : null);
}

/**
 * A single folder/file path. The EMPTY value is meaningful for output_dir:
 * null means "write beside each mask", which is not the same as any real path,
 * so we send null and never coerce it to "".
 */
function pathField(inp, v, set) {
  const input = el("input", {
    type: "text",
    value: v[inp.key] === null || v[inp.key] === undefined ? "" : String(v[inp.key]),
    placeholder: inp.placeholder || "",
    oninput: (e) => set(e.target.value.trim() === "" ? null : e.target.value),
  });

  const browse = el("button.btn.ghost.sm", {
    onclick: () => openFolderPicker(v[inp.key] || (S.context && S.context.cwd) || "", (chosen) => {
      input.value = chosen;
      set(chosen);
    }),
    title: "Browse the server's folders",
  }, "Browse…");

  return el("span", { style: { display: "flex", gap: "0", flex: "1 1 auto", minWidth: "0" } },
    el("span", { style: { display: "flex", flex: "1 1 auto", minWidth: "0" } }, input, browse));
}

/** A chip editor. `item_type:"color"` gets swatches and a color input. */
function listField(inp, v, set) {
  const isColor = inp.item_type === "color";
  const wrap = el("span", { style: { display: "flex", flexWrap: "wrap", gap: "6px",
                                     alignItems: "center", flex: "1 1 auto", minWidth: "0" } });

  const redraw = () => {
    clear(wrap);
    const items = Array.isArray(v[inp.key]) ? v[inp.key] : [];
    items.forEach((item, i) => {
      const text = colorText(item);
      wrap.appendChild(el("span.chip.on", { title: text },
        isColor ? el("span", {
          style: {
            width: "11px", height: "11px", borderRadius: "3px",
            background: cssColor(item), border: "1px solid var(--line)", display: "inline-block",
          },
        }) : null,
        text,
        el("span.x", {
          title: "remove",
          onclick: () => {
            const next = items.slice();
            next.splice(i, 1);
            set(next);
            redraw();
          },
        }, "✕")));
    });

    const entry = el("input", {
      type: "text",
      placeholder: isColor ? "white, #ff0000 or 255,0,0" : (inp.placeholder || "add…"),
      style: { flex: "0 1 200px", minWidth: "120px" },
      onkeydown: (e) => {
        if (e.key !== "Enter") return;
        e.preventDefault();
        const raw = e.target.value.trim();
        if (!raw) return;
        set([...(Array.isArray(v[inp.key]) ? v[inp.key] : []), raw]);
        redraw();
      },
    });
    wrap.appendChild(entry);

    if (isColor) {
      // The native swatch is the fastest way to a hex the server accepts.
      wrap.appendChild(el("input", {
        type: "color", value: "#ffffff",
        title: "pick a color to add",
        onchange: (e) => {
          set([...(Array.isArray(v[inp.key]) ? v[inp.key] : []), e.target.value]);
          redraw();
        },
      }));
    }
  };
  redraw();
  return wrap;
}

/**
 * ONE color: a native swatch, the text the server actually receives, and — for
 * `color_or_none` — a transparent toggle. The text field stays authoritative so
 * "white" and "255,0,0" remain typeable; the swatch just writes a hex into it.
 */
function colorField(inp, v, set) {
  const allowNone = inp.type === "color_or_none";
  const isNone = (x) => allowNone && typeof x === "string" && /^\s*(transparent|none)\s*$/i.test(x);
  // Remembered so ticking "transparent" and unticking it again gives back the colour you chose,
  // rather than the #ffffff fallback hexOf() returns for the word "transparent".
  let lastColor = isNone(v[inp.key]) ? "#ffffff" : (v[inp.key] ?? "#ffffff");

  const wrap = el("span", { style: { display: "flex", gap: "8px", alignItems: "center",
                                     flex: "1 1 auto", minWidth: "0", flexWrap: "wrap" } });

  const redraw = () => {
    clear(wrap);
    const val = v[inp.key];
    const none = isNone(val);

    const text = el("input", {
      type: "text",
      // An emptied field is null (the server falls back to the default) — but it must render as
      // EMPTY, not as the four characters "null", which is what String(null) would put here.
      value: none ? "transparent" : (val === null || val === undefined ? "" : colorText(val)),
      placeholder: inp.placeholder || "white, #ff0000 or 255,0,0",
      disabled: none || undefined,
      style: { flex: "0 1 190px", minWidth: "120px" },
      oninput: (e) => {
        const s = e.target.value.trim();
        if (s) lastColor = s;
        set(s === "" ? null : s);
      },
      onchange: () => redraw(),
    });

    // `change`, NOT `input`: <input type=color> fires input continuously while the OS dialog is
    // open, and redrawing would remove the very node that dialog is anchored to. The text box is
    // updated in place so the two stay in sync without tearing the swatch down.
    const swatch = el("input", {
      type: "color", value: hexOf(none ? lastColor : val),
      title: none ? "turn off transparent to choose a color" : "pick a color",
      disabled: none || undefined,
      onchange: (e) => { lastColor = e.target.value; set(e.target.value); text.value = e.target.value; },
    });
    append(wrap, [swatch, text]);

    if (allowNone) {
      wrap.appendChild(el("label.check",
        el("input", {
          type: "checkbox", checked: none || undefined,
          onchange: (e) => {
            if (e.target.checked) {
              if (!isNone(val) && val !== null && val !== undefined) lastColor = val;
              set("transparent");
            } else {
              set(lastColor);
            }
            redraw();
          },
        }),
        el("span.lbl", "transparent")));
    }
  };
  redraw();
  return wrap;
}

/** Any accepted color spelling -> "#rrggbb", for the native swatch input. */
function hexOf(item) {
  const hx = (n) => Math.max(0, Math.min(255, Math.round(Number(n) || 0))).toString(16).padStart(2, "0");
  if (Array.isArray(item) && item.length >= 3) return `#${item.slice(0, 3).map(hx).join("")}`;
  const s = String(item ?? "").trim();
  if (/^#[0-9a-f]{6}$/i.test(s)) return s.toLowerCase();
  if (/^#[0-9a-f]{3}$/i.test(s)) return `#${s.slice(1).split("").map((c) => c + c).join("")}`;
  const m = s.match(/^(\d+)\s*[,\s]\s*(\d+)\s*[,\s]\s*(\d+)/);
  if (m) return `#${m.slice(1, 4).map(hx).join("")}`;
  if (/^black$/i.test(s)) return "#000000";
  return "#ffffff";                            // "white", empty, or anything unrecognized
}

/** "white" | "#rrggbb" | [r,g,b] | "r,g,b" -> a display string. */
function colorText(item) {
  if (Array.isArray(item)) return item.join(",");
  return String(item);
}

/** The same, as something CSS will actually paint. */
function cssColor(item) {
  if (Array.isArray(item) && item.length >= 3) return `rgb(${item[0]},${item[1]},${item[2]})`;
  const s = String(item).trim();
  if (/^#[0-9a-f]{3,8}$/i.test(s)) return s;
  if (/^\d+\s*,\s*\d+\s*,\s*\d+$/.test(s)) return `rgb(${s})`;
  return s;                                  // a CSS color keyword ("white")
}


/* ==========================================================================
   MASK FIELD + PICKER
   ========================================================================== */

/**
 * The mask chooser, in both arities. `multi` collects a batch (the STL builder);
 * a single `mask_png` path collects exactly one (a collage's primary mask) and
 * stores a bare string rather than an array, because that is what the server's
 * `type:"path"` coercion expects.
 */
function maskField(t, inp, v, row) {
  const single = !inp.multi;
  const wrap = el("span", { style: { display: "flex", flexDirection: "column", gap: "7px",
                                     flex: "1 1 auto", minWidth: "0" } });

  // Seed the picker's score filter from the tool's own threshold, when it has one:
  // "pick from the leaves this tool would actually use" is the whole point of the filter. Read at
  // CLICK time, not render time — the form does not re-render on edit, so a value captured here
  // would filter at the old threshold after the user changed it.
  const scoreInput = (t.inputs || []).find((i) => i.key === "min_archetype_score");
  const currentSeed = () => {
    if (!scoreInput) return null;
    const n = Number(valuesFor(t.id)[scoreInput.key] ?? scoreInput.default);
    return Number.isFinite(n) ? n : null;
  };

  const commit = (chosen) => {
    v[inp.key] = single ? (chosen[0] || null) : chosen;
    markDirty(t, inp, v, row);
    redraw();
  };

  const redraw = () => {
    clear(wrap);
    const paths = single
      ? (v[inp.key] ? [v[inp.key]] : [])
      : (Array.isArray(v[inp.key]) ? v[inp.key] : []);

    const bar = el("span", { style: { display: "flex", gap: "7px", alignItems: "center",
                                      flexWrap: "wrap" } },
    el("button.btn.accent.sm", {
      onclick: () => openMaskPicker(paths, commit, { single, scoreSeed: currentSeed() }),
    }, single ? "Choose a mask…" : "Pick from a run…"),
    el("button.btn.ghost.sm", {
      onclick: () => openPastePaths(paths, (chosen) => commit(single ? chosen.slice(0, 1) : chosen)),
    }, single ? "Paste a path…" : "Paste paths…"),
    paths.length
      ? el("button.btn.ghost.sm", { onclick: () => commit([]) }, "Clear")
      : null,
    single
      ? el("span.badge" + (paths.length ? ".ok" : ""), paths.length ? baseName(paths[0]) : "none chosen")
      : el("span.badge" + (paths.length ? ".ok" : ""), `${fmtNum(paths.length)} selected`));
    wrap.appendChild(bar);

    if (paths.length) {
      const list = el("div", {
        style: {
          maxHeight: "168px", overflow: "auto", background: "var(--void)",
          border: "1px solid var(--line)", borderRadius: "var(--r-sm)",
          padding: "6px 8px", font: "11.5px/1.6 var(--mono)", color: "var(--mute)",
        },
      });
      paths.forEach((p, i) => list.appendChild(el("div", {
        style: { display: "flex", gap: "8px", alignItems: "baseline" }, title: p,
      },
      el("span.dim", { style: { flex: "0 0 auto" } }, String(i + 1).padStart(3, " ")),
      el("span.ell", { style: { flex: "1 1 auto" } }, baseName(p)),
      el("span.x", {
        style: { cursor: "pointer", color: "var(--dim)", flex: "0 0 auto" },
        title: "remove",
        onclick: () => {
          const next = paths.slice();
          next.splice(i, 1);
          commit(next);
        },
      }, "✕"))));
      wrap.appendChild(list);
    }
  };
  redraw();
  return wrap;
}

function openPastePaths(current, onDone) {
  const ta = el("textarea", {
    style: { minHeight: "180px", fontFamily: "var(--mono)", fontSize: "12px" },
    placeholder: "One absolute path per line",
  });
  ta.value = (current || []).join("\n");

  modal("Paste mask paths", el("div",
    el("p.hint", { style: { margin: "0 0 10px" } },
      "One path per line. Paths must sit inside a folder LM3 is allowed to read — the run output "
      + "folders, the input folders, and the temp folder."),
    ta), [
    { label: "Use these paths", cls: "primary", onClick: (close) => {
      const paths = ta.value.split("\n").map((s) => s.trim()).filter(Boolean);
      onDone(paths);
      close();
    } },
  ]);
}

/* -- the run/group/mask chooser ---------------------------------------- */

// perGroup starts at 50: each tile renders a real thumbnail of the mask, so a large first page
// costs hundreds of image requests before anything is usable. The picker offers larger pages.
// Module-scope so run/query/page-size persist between opens (a convenience). Anything that belongs
// to ONE field -- single, minScore, scrollTop, rerender -- is reset in openMaskPicker so the STL
// batch picker and the collage's primary-mask picker cannot inherit each other's mode.
const P = { run: null, query: "", includeRgb: false, perGroup: 50, data: null, selected: null,
  single: false, minScore: null, scrollTop: 0, rerender: null };

function openMaskPicker(current, onDone, opts = {}) {
  P.selected = new Set(current || []);
  P.data = null;
  P.single = Boolean(opts.single);
  P.scrollTop = 0;
  P.rerender = null;
  // Filtering to non-vetoed, high-scoring leaves is ON by default wherever the tool has a score
  // threshold to seed it from — that is exactly the "pick one of the good ones" case.
  P.minScore = Number.isFinite(opts.scoreSeed) ? Number(opts.scoreSeed) : null;
  if (P.single && P.minScore === null) P.includeRgb = false;

  const body = el("div", el("div.empty", el("span.spinner.lg")));
  const footer = el("span.dim", "");

  const close = modal(P.single ? "Choose a mask" : "Choose masks", body, [
    { label: P.single ? "Use this mask" : "Add selected", cls: "primary", onClick: (dismiss) => {
      onDone(Array.from(P.selected));
      dismiss();
    } },
  ], footer, "min(1100px, 94vw)");

  const refresh = async () => {
    clear(body);
    body.appendChild(el("div.empty", el("span.spinner.lg")));
    try {
      P.data = await api.post("/v1/postprocess/pick-masks", {
        run: P.run || undefined,
        query: P.query || "",
        include_rgb: P.includeRgb,
        per_group: P.perGroup,
        limit: Math.max(200, P.perGroup * 8),   // bounded by the page size, not a flat 4000
        min_archetype_score: P.minScore,
      });
      if (!P.run && P.data.run) P.run = P.data.run;
    } catch (err) {
      clear(body);
      body.appendChild(failure(err, "the mask list"));
      return;
    }
    renderPicker(body, footer, refresh);
  };

  // Seed the run from whatever the tool context saw most recently.
  const runs = (S.context && S.context.runs) || [];
  P.run = P.run || (runs.find((r) => r.has_reports) || runs[0] || {}).name || null;
  void refresh();
  return close;
}

function renderPicker(body, footer, refresh) {
  clear(body);
  P.rerender = () => renderPicker(body, footer, refresh);
  const d = P.data || {};

  const runSel = el("select", {
    style: { width: "auto", minWidth: "240px" },
    onchange: (e) => { P.run = e.target.value; refresh(); },
  }, ...((d.runs || []).map((r) =>
    el("option", { value: r.name, selected: r.name === P.run || undefined, title: r.path },
      `${r.name}${r.has_reports ? "" : "  (no reports)"}`))));

  const search = el("input", {
    type: "search", value: P.query, placeholder: "filter by specimen or product…",
    oninput: debounce((e) => { P.query = e.target.value.trim(); refresh(); }, 300),
  });

  const perSel = el("select", {
    style: { width: "auto" },
    onchange: (e) => { P.perGroup = Number(e.target.value); refresh(); },
  }, ...[50, 200, 500, 2000].map((n) =>
    el("option", { value: n, selected: P.perGroup === n || undefined }, `${n} per group`)));

  const scoreOn = P.minScore !== null;
  const scoreNum = el("input", {
    type: "number", min: "0", max: "1", step: "0.01",
    value: scoreOn ? String(P.minScore) : "0.8",
    disabled: scoreOn ? undefined : true,
    style: { width: "84px" },
    onchange: (e) => {
      const n = parseFloat(e.target.value);
      P.minScore = Number.isNaN(n) ? 0 : Math.max(0, Math.min(1, n));
      refresh();
    },
  });

  body.appendChild(el("div.toolbar",
    runSel,
    el("div.searchbox", { style: { flex: "1 1 220px" } }, search),
    perSel,
    el("label.check",
      el("input", {
        type: "checkbox", checked: scoreOn || undefined,
        onchange: (e) => {
          P.minScore = e.target.checked ? (parseFloat(scoreNum.value) || 0) : null;
          refresh();
        },
      }),
      el("span.lbl", { title: "Show only leaves that cleared every structural veto and scored "
                              + "above this. Whole-sheet masks have no per-leaf score, so they "
                              + "drop out while this is on." },
      "Only leaves scoring above")),
    scoreNum,
    el("label.check",
      el("input", {
        type: "checkbox", checked: P.includeRgb || undefined,
        onchange: (e) => { P.includeRgb = e.target.checked; refresh(); },
      }),
      el("span.lbl", { title: "RGB cutouts share their background color with mask holes, so the "
                              + "STL builder cannot separate foreground from hole in one." },
      "Include RGB cutouts"))));

  if (d.scored) {
    body.appendChild(el("p.hint", { style: { margin: "2px 0 0" } },
      `${fmtNum(d.n_scored_leaves)} leaf/leaves in ${d.run} cleared the vetoes and scored above `
      + `${d.min_archetype_score}, best first.`));
  } else if (scoreOn && d.message) {
    // The filter was asked for but the run cannot supply scores. Saying so beats silently
    // listing every mask under a checked box, which reads as "these are the good ones".
    body.appendChild(el("div.card.warn", el("p", d.message)));
  }

  if (!d.ready) {
    body.appendChild(empty("—", "Pick a run", d.message || "Choose a run to list its masks."));
    return;
  }
  if (!d.groups || !d.groups.length) {
    body.appendChild(empty("—", "No masks found",
      d.message || `${d.run} has no binary-mask folders. Enable the mask exports in Settings and re-run.`));
    return;
  }
  if (d.truncated) {
    body.appendChild(el("div.card.warn",
      el("p", `Showing the first ${fmtNum(P.perGroup)} masks in each group `
        + `(${fmtNum(d.n_masks)} listed). Raise "per group" or narrow the filter to see the rest.`)));
  }

  const scroller = el("div", { style: { maxHeight: "56vh", overflow: "auto" } });
  // Selecting a tile re-renders the whole body; without this the list snaps back to the top on
  // every click, which is worst in single-select where choosing IS the interaction.
  scroller.addEventListener("scroll", () => { P.scrollTop = scroller.scrollTop; });
  requestAnimationFrame(() => { scroller.scrollTop = P.scrollTop || 0; });

  for (const g of d.groups) {
    const allIn = g.masks.length && g.masks.every((m) => P.selected.has(m.path));
    const head = el("div.sechd", { style: { margin: "12px 0 8px" } },
      P.single
        ? el("span.lbl", { style: { color: "var(--ink)", fontWeight: "650" } }, g.label)
        : el("label.check",
          el("input", {
            type: "checkbox", checked: allIn || undefined,
            onchange: (e) => {
              for (const m of g.masks) {
                if (e.target.checked) P.selected.add(m.path);
                else P.selected.delete(m.path);
              }
              renderPicker(body, footer, refresh);
            },
          }),
          el("span.lbl", { style: { color: "var(--ink)", fontWeight: "650" } }, g.label)),
      el("span.badge" + (g.kind === "leaf" ? ".ok" : g.kind === "sheet" ? ".info" : ".vio"), g.kind),
      el("span.dim", { style: { fontSize: "11.5px" } },
        `${fmtNum(g.n_listed)} of ${fmtNum(g.n)}`));
    scroller.appendChild(head);

    const grid = el("div.mediagrid.sm");
    for (const m of g.masks) grid.appendChild(maskTile(m, body, footer, refresh));
    scroller.appendChild(grid);
  }

  body.appendChild(scroller);
  updatePickerFooter(footer);
}

function maskTile(m, body, footer, refresh) {
  const on = P.selected.has(m.path);

  // The mask lives inside a run, so the results module can thumbnail it by
  // ABSOLUTE path (safe_path accepts absolutes literally inside the run root).
  // If that endpoint is unavailable the tile degrades to a text placeholder.
  const img = el("img.thumb", {
    alt: m.name, loading: "lazy", decoding: "async",
    src: api.url(`/v1/runs/${encodeURIComponent(P.run || "")}/thumb`,
      { path: m.path, w: 128, token: api.token || undefined }),
    onerror: (e) => e.target.replaceWith(el("div.thumb", {
      style: { display: "flex", alignItems: "center", justifyContent: "center",
               aspectRatio: "4/3", background: "var(--panel2)",
               font: "650 12px/1 var(--mono)", color: "var(--dim)" },
    }, "MASK")),
  });

  const hasScore = m.score !== null && m.score !== undefined;
  return el(`div.mediatile${on ? ".sel" : ""}`, {
    title: `${m.name}\n${m.path}${hasScore ? `\narchetype score ${m.score}` : ""}`,
    onclick: () => {
      if (P.single) {
        // Exactly one: re-clicking the chosen tile deselects, anything else replaces it.
        const was = P.selected.has(m.path);
        P.selected.clear();
        if (!was) P.selected.add(m.path);
      } else if (P.selected.has(m.path)) {
        P.selected.delete(m.path);
      } else {
        P.selected.add(m.path);
      }
      renderPicker(body, footer, refresh);
    },
  },
  img,
  el("div.cap", { style: { display: "flex", gap: "6px", alignItems: "center" } },
    el("span", { style: { color: on ? "var(--acc2)" : "var(--dim)", flex: "0 0 auto" } },
      on ? (P.single ? "◉" : "☑") : (P.single ? "◯" : "☐")),
    el("span.ell", { title: m.specimen }, m.specimen || m.name),
    hasScore
      ? el("span.badge.ok", { style: { marginLeft: "auto", flex: "0 0 auto" },
        title: "archetype score" }, Number(m.score).toFixed(3))
      : el("span.sz", { style: { marginLeft: "auto", flex: "0 0 auto" }, }, fmtBytes(m.size_bytes))));
}

function updatePickerFooter(footer) {
  clear(footer);
  const chosen = Array.from(P.selected);
  append(footer, [
    P.single
      ? el("span.badge" + (chosen.length ? ".ok" : ""), { title: chosen[0] || "" },
        chosen.length ? baseName(chosen[0]) : "nothing chosen")
      : el("span.badge" + (chosen.length ? ".ok" : ""), `${fmtNum(chosen.length)} selected`),
    chosen.length
      ? el("button.btn.ghost.sm", {
        // Re-render: clearing the set alone left every tile still showing as selected and the
        // footer still naming the old choice, so "Use this mask" then committed nothing.
        onclick: () => { P.selected.clear(); P.rerender && P.rerender(); toast("Selection cleared"); },
      }, P.single ? "Clear" : "Clear selection")
      : null,
  ]);
}


/* ==========================================================================
   FOLDER PICKER  (POST /v1/settings/browse — degrades to plain typing)
   ========================================================================== */

function openFolderPicker(startPath, onPick) {
  let cur = startPath || "";
  const body = el("div", el("div.empty", el("span.spinner.lg")));
  let chosen = cur;

  const close = modal("Choose a folder", body, [
    { label: "Use this folder", cls: "primary", onClick: (dismiss) => { onPick(chosen); dismiss(); } },
  ], null, "min(760px, 94vw)");

  const load = async (path) => {
    clear(body);
    body.appendChild(el("div.empty", el("span.spinner.lg")));
    let d;
    try {
      d = await api.post("/v1/settings/browse", { path, show_hidden: false, count_images: false });
    } catch (err) {
      // The settings module may not be mounted; typing a path still works.
      clear(body);
      body.appendChild(el("div.card.warn",
        el("h4", "Folder browsing is unavailable"),
        el("p", "Type the folder path into the field instead.")));
      body.appendChild(failure(err, "the folder listing"));
      return;
    }
    cur = d.path;
    chosen = d.path;
    clear(body);

    const crumbs = el("div.crumbs");
    for (const r of d.roots || []) {
      crumbs.appendChild(el("a", { onclick: () => load(r.path), title: r.path }, r.label));
      crumbs.appendChild(el("span.sep", "·"));
    }
    body.appendChild(crumbs);

    const typed = el("input", {
      type: "text", value: d.path, style: { fontFamily: "var(--mono)" },
      onchange: (e) => load(e.target.value),
    });
    body.appendChild(el("div.toolbar",
      d.parent ? el("button.btn.ghost.sm", { onclick: () => load(d.parent) }, "↑ up") : null,
      el("span", { style: { flex: "1 1 260px" } }, typed)));

    if (d.fell_back) {
      body.appendChild(el("div.card.warn",
        el("p", `That path does not exist yet — showing ${d.path} instead. `
          + "Use this folder to have the tool create it on write.")));
    }
    if (!d.readable) {
      body.appendChild(el("div.card.bad", el("p", "This folder cannot be read.")));
    }

    const ul = el("ul.tablist", { style: { maxHeight: "44vh", overflow: "auto" } });
    for (const e2 of d.entries || []) {
      ul.appendChild(el("li", {
        onclick: () => load(e2.path),
        title: e2.path,
      },
      el("span.nm", e2.name),
      el("span.n", e2.n_dirs ? `${fmtNum(e2.n_dirs)} dirs` : "")));
    }
    body.appendChild(el("div.panel", el("div.panel-bd.flush",
      (d.entries || []).length ? ul : empty("—", "No subfolders", ""))));
  };

  void load(cur);
  return close;
}


/* ==========================================================================
   RUNNING A TOOL
   ========================================================================== */

/** Turn the form values into the request body. */
function payloadFor(t) {
  const v = valuesFor(t.id);
  const params = {};
  for (const inp of t.inputs) {
    let val = v[inp.key];
    if (inp.type === "path" && !inp.multi) {
      // Empty is MEANINGFUL here (null = write beside each mask), so it is
      // sent as null rather than dropped or coerced to "".
      params[inp.key] = (val === "" || val === undefined) ? null : val;
      continue;
    }
    if (val === undefined) continue;         // never sent -> server uses its default
    params[inp.key] = val;
  }
  return params;
}

async function startRun(t) {
  const params = payloadFor(t);
  D.runBtn.disabled = true;
  D.runBtn.textContent = "Starting…";
  try {
    const res = await api.post("/v1/postprocess/run", { tool_id: t.id, params });
    S.logSeq = 0;
    S.task = null;
    watchTask(res.task_id, t.id);
    toast(`${t.name} started`, "ok");
  } catch (err) {
    const d = detailOf(err);
    if (err.status === 409 && d.reason === "target_active") {
      // Section 2.8's refusal: the target IS the run in progress. Name it, and name the rule.
      toast(d.message
        || `${d.run_name || "That run"} is active — a tool cannot write into a run in progress.`,
        "warn", "Target is the active run");
      void reloadTools();                  // re-read the context so the notice above appears
    } else if (err.status === 409 && d.reason === "target_locked") {
      toast(d.message || "Another read/write tool is already working on that run.",
        "warn", "Run is locked");
    } else if (err.status === 409) {
      // detail is an OBJECT on 409 — offer to watch the task already running.
      toast(d.message || "That tool is already running", "warn", "Already running");
      if (d.task_id) watchTask(d.task_id, d.tool_id || t.id);
    } else {
      // 422 messages are written to be shown to the user verbatim.
      showFormError(t, d.message || String(err.message || err));
    }
  } finally {
    D.runBtn.disabled = false;
    D.runBtn.textContent = "Run tool";
  }
}

/** Surface a 422 next to the field it names, when it names one. */
function showFormError(t, message) {
  const panel = el("div.card.bad", el("h4", "The tool refused these parameters"), el("p", message));
  clear(D.runPanel);
  D.runPanel.appendChild(panel);

  // The server phrases these as "<Label>: <problem>", so match on the label.
  for (const inp of t.inputs) {
    if (!message.startsWith(`${inp.label}:`)) continue;
    const row = D.detail.querySelector(`.treerow[data-key="${CSS.escape(inp.key)}"]`);
    if (!row) continue;
    row.classList.add("invalid");
    if (!row.querySelector(".err")) row.appendChild(el("div.err", message));
    row.scrollIntoView({ block: "center", behavior: "smooth" });
  }
  toast(message, "bad", "Cannot run");
}

/**
 * Attach to a task: one SSE connection fills the log AND keeps it live.
 * Frames are typed (hello|log|progress|ping|done) — switch, never sniff.
 */
function watchTask(taskId, toolId) {
  if (closeStream) { closeStream(); closeStream = null; }
  S.taskToolId = toolId || S.taskToolId;
  S.logSeq = 0;
  S.task = { task_id: taskId, tool_id: toolId, state: "running", pct: 0, log: [] };
  renderRunPanel();

  // The same array S holds, not a copy: the SSE handler appends to it and every
  // other caller of renderRunPanel() reads it back.
  const lines = S.logLines;
  lines.length = 0;

  closeStream = api.sse(`/v1/postprocess/tasks/${encodeURIComponent(taskId)}/events`, {
    onMessage: (frame) => {
      if (!frame || typeof frame !== "object") return;
      switch (frame.type) {
        case "hello":
          S.task = frame.task;
          lines.length = 0;
          appendLines(lines, frame.task.log || []);
          renderRunPanel(lines);
          break;
        case "log":
          appendLines(lines, frame.lines || []);
          paintLog(lines);
          break;
        case "progress":
          S.task = { ...S.task, ...frame.task };
          paintProgress();
          break;
        case "done":
          S.task = { ...S.task, ...frame.task };
          if (closeStream) { closeStream(); closeStream = null; }
          renderRunPanel(lines);
          void afterFinish();
          break;
        case "ping":
        default:
          break;
      }
    },
    onError: () => {
      // The browser reconnects on its own; a `hello` will re-seed the panel.
      if (D.connDot) D.connDot.className = "conn wait";
    },
    onOpen: () => { if (D.connDot) D.connDot.className = "conn up"; },
  });
}

/** Append only lines we have not rendered — SSE can replay after a reconnect. */
function appendLines(store, incoming) {
  for (const ln of incoming) {
    if (Number(ln.seq) <= S.logSeq) continue;
    S.logSeq = Number(ln.seq);
    store.push(ln);
  }
  // A tool can be chatty; keep the DOM bounded.
  if (store.length > 4000) store.splice(0, store.length - 4000);
}

async function afterFinish() {
  try {
    const tasks = await api.get("/v1/postprocess/tasks", { params: { limit: 25 } });
    S.active = (tasks && tasks.active) || {};
  } catch { /* the card badge is cosmetic; a stale one is harmless */ }
  renderCards();
  const t = S.task;
  if (!t) return;
  if (t.state === "error") toast(t.error || "The tool failed", "bad", "Failed");
  else if (t.result && Number(t.result.n_failed) > 0) {
    toast(`${fmtNum(t.result.n_written)} written, ${fmtNum(t.result.n_failed)} failed`,
      "warn", "Finished with failures");
  } else toast(`${fmtNum(t.n_outputs)} file(s) produced`, "ok", "Finished");
}


/* ==========================================================================
   RUN PANEL
   ========================================================================== */

function renderRunPanel(lines = S.logLines) {
  if (!D.runPanel) return;
  clear(D.runPanel);
  const t = S.task;
  if (!t) return;

  const running = t.state === "running";
  const bad = t.state === "error";

  D.bar = el(`div.bar.lg${running ? ".running" : ""}${bad ? ".bad" : t.state === "done" ? ".ok" : ""}`,
    el("span.fill", { style: { width: `${Math.max(0, Math.min(100, Number(t.pct) || 0))}%` } }),
    el("span.lbl", `${fmtNum(t.pct, { digits: 0 })}%`));

  D.progMeta = el("div", { style: { display: "flex", gap: "16px", flexWrap: "wrap",
                                    marginTop: "8px", fontSize: "12px", color: "var(--mute)" } });
  D.connDot = el("span.conn.up", running ? "live" : "");

  const header = el("div.panel-hd",
    el("span.t", `${t.tool_name || tool(t.tool_id)?.name || "Tool"} — run`),
    running ? el("span.spinner") : null,
    el("span.badge" + (bad ? ".bad" : t.state === "done" ? ".ok" : ".info"), t.state || "…"),
    el("span.tools",
      running ? D.connDot : null,
      el("button.btn.ghost.sm", {
        onclick: () => copyText((lines || []).map(fmtLogLine).join("\n"), "log"),
      }, "Copy log")));

  const bodyParts = [D.bar, D.progMeta];

  if (bad && t.error) {
    bodyParts.push(el("div.card.bad", { style: { marginTop: "12px" } },
      el("h4", "The tool failed"),
      el("p.mono", { style: { overflowWrap: "anywhere" } }, t.error)));
  }

  if (t.result && Array.isArray(t.result.failures) && t.result.failures.length) {
    bodyParts.push(el("div.card.warn", { style: { marginTop: "12px" } },
      el("h4", `${fmtNum(t.result.failures.length)} input(s) failed`),
      el("ul", ...t.result.failures.map((f) => el("li",
        el("span.mono", baseName(f.mask || f.path || f.input || "")),
        " — ", String(f.error || f.reason || "failed"))))));
  }

  if (!running && t.outputs && t.outputs.length) bodyParts.push(outputsPanel(t));
  if (t.result && Array.isArray(t.result.results) && t.result.results.length) {
    bodyParts.push(resultTable(t.result.results));
  }

  D.console = el("div.console.boxed", { style: { maxHeight: "300px", marginTop: "12px" } });
  bodyParts.push(el("div.console-hd",
    el("span", "Tool output"),
    el("span.tools",
      el("label.check",
        el("input", {
          type: "checkbox", checked: S.followLog || undefined,
          onchange: (e) => { S.followLog = e.target.checked; },
        }),
        el("span.lbl", "follow")))));
  bodyParts.push(D.console);

  D.runPanel.appendChild(el("div.panel", header, el("div.panel-bd", ...bodyParts)));

  paintProgress();
  paintLog(lines || (t.log || []));
}

function paintProgress() {
  const t = S.task;
  if (!t || !D.bar || !D.progMeta) return;
  const fill = D.bar.querySelector(".fill");
  const lbl = D.bar.querySelector(".lbl");
  const pct = Math.max(0, Math.min(100, Number(t.pct) || 0));
  if (fill) fill.style.width = `${pct}%`;
  if (lbl) lbl.textContent = `${fmtNum(pct, { digits: 0 })}%`;
  D.bar.classList.toggle("running", t.state === "running");

  clear(D.progMeta);
  const facts = [
    ["status", t.message || t.state || "…"],
    ["items", t.n_total ? `${fmtNum(t.n_done)} / ${fmtNum(t.n_total)}` : "–"],
    ["elapsed", fmtDuration(t.elapsed_s)],
    ["produced", fmtNum(t.n_outputs || 0)],
  ];
  if (t.started_at) facts.push(["started", fmtTime(t.started_at)]);
  append(D.progMeta, facts.map(([k, v]) =>
    el("span", { style: { whiteSpace: "nowrap" } },
      el("span.dim", `${k} `),
      el("span.mono", { style: { color: "var(--ink)" } }, String(v)))));
}

function fmtLogLine(ln) {
  return `${fmtTime(ln.t)} ${String(ln.level || "INFO").padEnd(8)} ${ln.src || ""}: ${ln.msg || ""}`;
}

function paintLog(lines) {
  if (!D.console) return;
  clear(D.console);
  const rows = lines || [];
  if (!rows.length) {
    D.console.appendChild(el("div.console-empty", "No output yet."));
    return;
  }
  for (const ln of rows) {
    // `level` is a Python logging name, so it drops straight into the console
    // CSS classes the Status tab already defines.
    D.console.appendChild(el(`div.line.${String(ln.level || "INFO").toUpperCase()}`,
      el("span.t", fmtTime(ln.t)),
      el("span.src", { title: ln.src || "" }, ln.src || ""),
      el("span.msg", ln.msg || "")));
  }
  if (S.followLog) D.console.scrollTop = D.console.scrollHeight;
}

function outputsPanel(t) {
  const list = el("div", { style: { display: "flex", flexDirection: "column", gap: "1px",
                                    background: "var(--line)", border: "1px solid var(--line)",
                                    borderRadius: "var(--r-sm)", overflow: "hidden" } });

  for (const out of t.outputs) {
    const dl = downloadUrl(out);
    list.appendChild(el("div", {
      style: { display: "flex", gap: "10px", alignItems: "center", padding: "7px 10px",
               background: "var(--panel)", minWidth: "0" },
      title: out,
    },
    el("span.mono", { style: { color: "var(--acc3)", flex: "0 0 auto", fontSize: "11px" } }, "→"),
    el("span.mono.ell", { style: { flex: "1 1 auto", fontSize: "12px", color: "var(--ink)" } },
      baseName(out)),
    dl
      ? el("a.btn.ghost.sm", { href: dl, download: baseName(out) }, "Download")
      : el("span.dim", { style: { fontSize: "11px" },
        title: "This file is outside every run folder the server serves" }, "on disk"),
    el("button.btn.ghost.sm", { onclick: () => copyText(out, "file path") }, "Copy path")));
  }

  return el("div", { style: { marginTop: "14px" } },
    el("div.sechd", `Produced files (${fmtNum(t.outputs.length)})`),
    list,
    el("div", { style: { marginTop: "8px" } },
      el("button.btn.ghost.sm", {
        onclick: () => copyText(t.outputs.join("\n"), "all paths"),
      }, "Copy all paths")));
}

/**
 * The per-item result table. Columns are DISCOVERED from the rows, so a future
 * tool reporting different fields renders without a change here; the few keys
 * we do know about get friendlier headers and units.
 */
function resultTable(rows) {
  const HEADERS = {
    mask: "input", stl: "output", n_parts: "parts", size_mm: "size (mm)",
    thickness_mm: "thickness (mm)", length_mm: "length (mm)", watertight: "watertight",
    scale_mm_per_px: "mm / px", fill_holes: "holes filled",
  };
  const cols = [];
  for (const r of rows) for (const k of Object.keys(r)) if (!cols.includes(k)) cols.push(k);

  const cell = (v) => {
    if (v === null || v === undefined) return el("td.null", "—");
    if (typeof v === "boolean") {
      return el("td", el("span.badge" + (v ? ".ok" : ".warn"), v ? "yes" : "no"));
    }
    if (Array.isArray(v)) return el("td.num", v.map((x) => fmtNum(x, { digits: 1 })).join(" × "));
    if (typeof v === "number") return el("td.num", fmtNum(v, { digits: null }));
    const s = String(v);
    // Full paths are unreadable in a grid; the basename plus a title is not.
    return s.includes("/")
      ? el("td.mono", { title: s }, baseName(s))
      : el("td", { title: s }, s);
  };

  return el("div", { style: { marginTop: "14px" } },
    el("div.sechd", "Per-item results"),
    el("div.tblwrap.tall",
      el("table.datatable.dense",
        el("thead", el("tr", ...cols.map((c, i) =>
          el(`th${i === 0 ? ".l" : ""}`, { title: c }, HEADERS[c] || c.replace(/_/g, " "))))),
        el("tbody", ...rows.map((r) => el("tr", ...cols.map((c) => cell(r[c]))))))));
}


/* ==========================================================================
   MODAL
   ========================================================================== */

/**
 * A backdrop + .modal with a header, a body and footer buttons.
 * Returns a `close()` so a caller can dismiss it programmatically.
 */
function modal(title, body, buttons, footerExtra, width) {
  const backdrop = el("div.backdrop", {
    onclick: (e) => { if (e.target === backdrop) close(); },
  });

  function close() {
    backdrop.remove();
    document.removeEventListener("keydown", onKey);
  }
  const onKey = (e) => { if (e.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);

  const foot = el("div.panel-ft",
    footerExtra || null,
    el("span.spacer"),
    ...(buttons || []).map((b) =>
      el(`button.btn${b.cls ? `.${b.cls}` : ".ghost"}`, { onclick: () => b.onClick(close) }, b.label)),
    el("button.btn.ghost", { onclick: close }, "Cancel"));

  backdrop.appendChild(el("div.modal", {
    style: { maxWidth: width || "min(820px, 94vw)", width: width || "min(820px, 94vw)" },
  },
  el("div.panel-hd", el("span.t", title),
    el("span.tools", el("button.btn.ghost.sm", { onclick: close }, "✕"))),
  el("div.panel-bd", body),
  foot));

  document.body.appendChild(backdrop);
  return close;
}
