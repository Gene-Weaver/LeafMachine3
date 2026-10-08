/**
 * models.js — the models catalog, shared by the Models tab (the full per-module view) and the Live
 * Status tab (a banner that only shows while a default is missing or outdated).
 *
 * Both call `mountModelsPanel(host, {compact})`. The full view is driven entirely by
 * GET /v1/models/catalog: one card per pipeline stage, one row per Hub unit, one chip per format
 * the lock pins for that unit. Nothing in here names a stage, a model or a format -- a new one shows
 * up by regenerating the lock. The "Active in LM3" selector reads `modules.<stage>.model` from the
 * settings file and writes it back through POST /v1/models/activate: the YAML stays the truth, so a
 * hand edit shows up here on the next refresh, and a change here is a normal settings write.
 *
 * One install at a time: the server returns the running task if a second press lands while one is in
 * flight, and every mounted panel follows the same stream.
 */
import { api, el, clear, fmtBytes } from "./api.js";

const STATE_LABEL = {
  current: "up to date",
  missing: "missing",
  outdated: "update available",
  pending: "local copy (not published yet)",
  unavailable: "not published yet",
};
const STATE_BADGE = { current: "ok", missing: "bad", outdated: "warn", pending: "info", unavailable: "info" };
const HF = "https://huggingface.co/";
const CATALOG_POLL_MS = 6000;    // while the full view is on screen: catches a hand-edited yaml

/* a tiny shared store so several panels stay in step */
const panels = new Set();
let lastStatus = null;       // GET /v1/models/status  (the banner + tab tone)
let lastCatalog = null;      // GET /v1/models/catalog (the full view)
let modelWarnings = [];      // settings-file warnings about model paths (code "missing_model")
let activeTask = null;       // {id, close, label} while an install streams
let pollTimer = 0;

/** Model-path warnings from the settings validator; shown on the Models tab, not under Settings. */
export function isModelWarning(w) { return !!w && w.code === "missing_model"; }
export function getModelWarnings() { return modelWarnings; }

/** The Models tab label goes warning-orange while anything needs attention, plain otherwise. */
function applyTabTone() {
  const btn = document.querySelector('.tab[data-tab="models"]');
  if (!btn) return;
  const s = lastStatus && lastStatus.summary;
  const unpublished = !!(s && ((s.unavailable && s.unavailable.length) || s.tab_attention));
  const attention = needsAttention(lastStatus) || unpublished || modelWarnings.length > 0 || !!(lastStatus && lastStatus.error);
  btn.style.color = attention ? "var(--warn)" : "";
  btn.title = attention ? "Models need attention" : "";
}

export async function refreshModelsStatus() {
  const wantCatalog = [...panels].some((p) => !p.compact);
  const [st, settings, cat] = await Promise.allSettled([
    api.getModelsStatus(), api.getSettings(), wantCatalog ? api.getModelsCatalog() : Promise.resolve(null)]);
  lastStatus = st.status === "fulfilled" ? st.value : { error: st.reason && st.reason.message ? st.reason.message : String(st.reason) };
  const warns = settings.status === "fulfilled" && Array.isArray(settings.value.warnings) ? settings.value.warnings : [];
  modelWarnings = warns.filter(isModelWarning);
  if (wantCatalog) {
    lastCatalog = cat.status === "fulfilled" ? cat.value
      : { error: cat.reason && cat.reason.message ? cat.reason.message : String(cat.reason) };
  }
  applyTabTone();
  for (const p of panels) p.render();
  return lastStatus;
}

/** Button text per the spec: missing → install; present but older → update available. */
export function buttonLabel(st) {
  if (!st || st.error) return "Install Models from Hugging Face";
  const s = st.summary;
  if (!s) return "Install Models from Hugging Face";
  if (s.missing && s.missing.length) return "Install missing defaults";
  if (s.outdated && s.outdated.length) return "Update default models";
  return "Models are up to date";
}

function needsAttention(st) {
  return !!(st && st.summary && st.summary.needs_attention);
}

/** "archival_detector" -> "Archival Detector"; the stage key is the only name the lock carries. */
function stageLabel(key) {
  return String(key).split("_").map((w) => w === "mp" ? "MP" : w === "ect" ? "ECT" : w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}

/* -------------------------------------------------------------- modal -- */
function openModal({ title, build, width = 520 }) {
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
    if (build.onClose) build.onClose(result);
  };
  const onKey = (ev) => { if (ev.key === "Escape") { ev.stopPropagation(); close(undefined); } };
  closeBtn.addEventListener("click", () => close(undefined));
  backdrop.addEventListener("mousedown", (ev) => { if (ev.target === backdrop) close(undefined); });
  document.addEventListener("keydown", onKey, true);
  build(panel, close);
  document.body.appendChild(backdrop);
  return close;
}

/** Short revision id with a link to the commit on the Hub, or an em dash. */
function rev(repo, id) {
  if (!id) return el("span.mono.dim", "—");
  return el("a.mono", { href: `${HF}${repo}/commit/${id}`, target: "_blank", rel: "noopener", title: id,
    style: { color: "var(--ink)", textDecoration: "none", borderBottom: "1px dotted var(--dim)" } }, id.slice(0, 10));
}

/**
 * Ask before anything on disk is overwritten; a plain download of something absent just goes.
 * `rows` are the replacements: [{label, repo, from, to}] -- what is installed → what the lock pins.
 */
function confirmReplace({ title, rows, bytes, verb = "Update" }) {
  return new Promise((resolve) => {
    openModal({
      title,
      width: 720,
      build: Object.assign((panel, close) => {
        const version = (lastCatalog && lastCatalog.app_version) || "";
        const tbl = el("table.tbl", { style: { width: "100%", fontSize: "12.5px", margin: "0 0 10px", whiteSpace: "nowrap" } },
          el("thead", el("tr", el("th", "Model"), el("th", "Installed"), el("th", ""),
            el("th", { title: "the Hub revision the models lock of this LM3 release pins" }, `Pinned by LM3 ${version}`))),
          el("tbody", rows.map((r) => el("tr",
            el("td", el("span.ink", r.label), r.sub ? el("span.dim", { style: { display: "block", fontSize: "11px" } }, r.sub) : null),
            el("td", rev(r.repo, r.from)), el("td.dim", "→"), el("td", rev(r.repo, r.to))))));
        panel.appendChild(el("div.panel-bd", tbl,
          el("p.hint", { style: { margin: 0 } },
            `${bytes ? fmtBytes(bytes) + " to download. " : ""}The replaced files are kept as .backup until the new ones verify, then dropped; a failure puts them back.`)));
        const ok = el("button.btn.sm.primary", { type: "button" }, verb);
        ok.addEventListener("click", () => close(true));
        panel.appendChild(el("div.panel-ft", { style: { display: "flex", gap: "8px", justifyContent: "flex-end" } },
          el("button.btn.sm.ghost", { type: "button", onclick: () => close(false) }, "Cancel"), ok));
        setTimeout(() => ok.focus(), 0);
      }, { onClose: (r) => resolve(r === true) }),
    });
  });
}

/* ------------------------------------------------------------ install -- */
/**
 * Run one install request and stream its progress into every panel. `body` is what
 * POST /v1/models/install takes: {} for "the missing/outdated defaults", {actions:[stage],
 * formats:[fmt]} for one format of a default, {models:[[stage,key]], formats:[fmt]} for an alternate.
 */
async function runInstall(body, label) {
  if (activeTask) return activeTask;
  let res;
  let poll = 0;
  try {
    res = await api.installModels(body);
  } catch (err) {
    for (const p of panels) p.log(`could not start: ${err.message || err}`, "bad");
    return null;
  }
  const close = api.streamModelsInstall(res.task_id, {
    onMessage: (ev) => {
      if (!ev || typeof ev !== "object") return;
      if (ev.type === "start") {
        for (const p of panels) p.log(`${label}: ${ev.actions.length} model(s), ${fmtBytes(ev.total_bytes || 0)} …`);
      } else if (ev.type === "file" && ev.phase === "download") {
        for (const p of panels) p.log(`${ev.action}: downloading ${ev.file} (${fmtBytes(ev.bytes || 0)})`);
      } else if (ev.type === "action_done") {
        for (const p of panels) p.log(`${ev.action}: installed`, "ok");
      } else if (ev.type === "skip") {
        for (const p of panels) p.log(`${ev.action}: skipped (${ev.reason})`, "dim");
      } else if (ev.type === "error" || ev.type === "failed") {
        for (const p of panels) p.log(`${ev.action ? ev.action + ": " : ""}${ev.message}`, "bad");
      } else if (ev.state) {                                   // "done" frame
        const ok = ev.state === "done";
        for (const p of panels) p.log(ok ? `${label}: done, files verified` : `install ${ev.state}: ${ev.error || ""}`, ok ? "ok" : "bad");
        finish();
      }
    },
    // If the stream cannot be read (auth, proxy, a dropped connection) fall back to polling the
    // task snapshot, so a chip can never sit on "installing…" after the install has ended.
    onError: () => startPolling(),
  });
  activeTask = { id: res.task_id, close, label, body };
  for (const p of panels) p.render();
  function startPolling() {
    if (poll) return;
    poll = setInterval(async () => {
      try {
        const snap = await api.getModelsInstall(res.task_id, 0);
        if (snap.state !== "running") {
          const ok = snap.state === "done";
          for (const p of panels) p.log(ok ? `${label}: done, files verified` : `install ${snap.state}: ${snap.error || ""}`, ok ? "ok" : "bad");
          finish();
        }
      } catch { /* server busy or restarting: keep trying */ }
    }, 1000);
  }
  async function finish() {
    if (poll) { clearInterval(poll); poll = 0; }
    if (activeTask) { try { activeTask.close(); } catch { /* noop */ } }
    activeTask = null;
    await refreshModelsStatus();
  }
  return activeTask;
}

/** The header button: every missing or outdated default, in the default formats. */
async function installDefaults() {
  const st = lastStatus && !lastStatus.error ? lastStatus : await refreshModelsStatus();
  const s = st && st.summary;
  const replacing = s && s.outdated && s.outdated.length;
  if (replacing) {
    const cat = lastCatalog && !lastCatalog.error ? lastCatalog : await api.getModelsCatalog();
    const rows = [];
    let bytes = 0;
    for (const stage of cat.stages) {
      if (!s.outdated.includes(stage.stage)) continue;
      const v = stage.variants.find((x) => x.default);
      for (const u of v.units) for (const [fmt, f] of Object.entries(u.formats)) {
        if (!cat.default_formats.includes(fmt) || f.state === "current") continue;
        rows.push({ label: `${stageLabel(stage.stage)} · ${fmt}`, sub: u.repo_id, repo: u.repo_id, from: u.installed_revision, to: u.revision });
        bytes += f.bytes || 0;
      }
    }
    if (!(await confirmReplace({ title: `Update ${rows.length} default model${rows.length === 1 ? "" : "s"}`, rows, bytes }))) return null;
  }
  return runInstall({}, "defaults");
}

/** One (stage, variant, unit, format) chip. Downloads what is absent; asks before replacing. */
async function installFormat(stage, variant, fmt, f) {
  const body = variant.default ? { actions: [stage], formats: [fmt] } : { models: [[stage, variant.model_key]], formats: [fmt] };
  if (f.state === "outdated" || f.state === "partial") {
    body.force = true;
    const unit = variant.units.find((u) => u.formats[fmt] === f) || variant.units[0];
    const ok = await confirmReplace({
      title: `${f.state === "outdated" ? "Update" : "Repair"} ${stageLabel(stage)} · ${variant.model_key} · ${fmt}`,
      rows: [{ label: `${variant.model_key} · ${fmt}`, sub: unit.repo_id, repo: unit.repo_id, from: unit.installed_revision, to: unit.revision }],
      bytes: f.bytes, verb: f.state === "outdated" ? "Update" : "Repair",
    });
    if (!ok) return null;
  }
  return runInstall(body, `${variant.model_key} · ${fmt}`);
}

/* ------------------------------------------------------------ activate -- */
async function activate(stage, modelKey, fmt, panel) {
  try {
    const cat = await api.activateModel(stage, modelKey, fmt);
    lastCatalog = cat;
    panel.log(`${stageLabel(stage)}: now runs ${modelKey} · ${fmt} → ${cat.activated && cat.activated.path}`, "ok");
    if (cat.settings && cat.settings.warnings && cat.settings.warnings.length) {
      panel.log(`saved with warnings: ${cat.settings.warnings.slice(0, 2).map((w) => w.msg || w).join(" · ")}`, "dim");
    }
    // The settings file changed under everyone else (the top strip, the Settings tab): tell them the
    // same way the top strip does when it saves a field.
    document.dispatchEvent(new CustomEvent("lm3:settings-saved", {
      detail: { path: `modules.${stage}.model`, value: cat.activated, settings: cat.settings },
    }));
    for (const p of panels) p.render();
    refreshModelsStatus();
  } catch (err) {
    const d = err && err.detail;
    const msg = (d && (d.message || d)) || err.message || String(err);
    panel.log(`could not activate ${modelKey} · ${fmt}: ${typeof msg === "string" ? msg : JSON.stringify(msg)}`, "bad");
    for (const p of panels) p.render();   // put the selector back on what the yaml says
  }
}

/* -------------------------------------------------------------- panel -- */
const COLS = ["15%", "11%", "19%", "31%", "17%", "7%"];
const HEADS = ["Module", "State", "Active in LM3", "Variant", "Formats", "Hub rev"];

function grid(rows) {
  return el("table.mh-grid",
    el("colgroup", COLS.map((w) => el("col", { style: { width: w } }))),
    el("thead", el("tr", HEADS.map((h) => el("th", h)))),
    el("tbody", rows));
}

/** Is this (stage, variant, format) what the running install is fetching? */
function installing(stage, variant, fmt, f) {
  const b = activeTask && activeTask.body;
  if (!b) return false;
  const fmts = b.formats || (lastCatalog && lastCatalog.default_formats) || [];
  if (!fmts.includes(fmt)) return false;
  if (b.models) return b.models.some(([st, key]) => st === stage && key === variant.model_key);
  if (b.actions) return variant.default && b.actions.includes(stage);
  // the header button: every default that is missing or outdated, in the default formats
  return variant.default && f.state !== "current";
}

function fmtChip(stage, variant, fmt, f, panel) {
  const busy = !!activeTask;
  const size = installing(stage, variant, fmt, f)
    ? el("span.spin", { title: "downloading…" })
    : el("span.sz", fmtBytes(f.bytes || 0));
  let chip;
  if (f.state === "current") {
    chip = el(`span.mh-fmt.on${f.active ? ".act" : ""}`, { title: `installed${f.active ? " · active in LM3" : ""}` }, "✓ ", fmt, " ", size);
  } else if (f.state === "outdated") {
    chip = el(`span.mh-fmt.new${f.active ? ".act" : ""}`, { title: "a newer revision is pinned in the lock — click to update" }, "↻ ", fmt, " ", size);
  } else if (f.state === "partial") {
    chip = el("span.mh-fmt.part", { title: "some of this format's files are missing — click to repair" }, "! ", fmt, " ", size);
  } else {
    chip = el("span.mh-fmt.get", { title: `download ${fmtBytes(f.bytes || 0)}` }, "⤓ ", fmt, " ", size);
  }
  if (f.state !== "current") {
    if (busy) chip.classList.add("busy");
    else chip.addEventListener("click", () => installFormat(stage, variant, fmt, f).then(() => panel.render()));
  }
  return chip;
}

function activeSelect(stage, s, panel) {
  const opts = [];
  // every installed, runnable (variant × format) pair; nothing that is not on disk can be chosen
  for (const v of s.variants) for (const u of v.units) for (const [fmt, f] of Object.entries(u.formats)) {
    if (!f.runnable || f.state === "missing" || f.state === "partial") continue;
    const label = `${v.units.length > 1 ? u.model_key : v.model_key} · ${fmt}`;
    opts.push(el("option", { value: `${v.model_key}|${fmt}`, selected: !!f.active }, label));
  }
  if (!s.activatable) {
    const name = s.variants.find((v) => v.default) || s.variants[0];
    const label = name && name.units.length > 1 ? `ensemble (${name.units.length} models)` : (name ? name.model_key : "default");
    const fmt = (s.active && s.active.matched_format) || (lastCatalog && lastCatalog.default_formats[0]) || "";
    // a control that cannot be opened would only pretend to be one: a label says the same thing
    return el("span.mh-lock", { title: "This module always uses its default model" },
      el("span.mono", `${label}${fmt ? " · " + fmt : ""}`), el("span", "🔒"));
  }
  const sel = el("select.mh-sel");
  if (!(s.active && s.active.matched)) {
    // the yaml points somewhere the lock does not know (a custom file); show it, do not touch it
    sel.appendChild(el("option", { value: "", selected: true }, s.active && s.active.path ? `custom: ${s.active.path}` : "(not set)"));
  }
  for (const o of opts) sel.appendChild(o);
  sel.title = s.active && s.active.path ? `modules.${stage}.model.path = ${s.active.path}` : "";
  sel.addEventListener("change", () => {
    const [modelKey, fmt] = String(sel.value).split("|");
    if (!modelKey) return;
    sel.disabled = true;
    activate(stage, modelKey, fmt, panel);
  });
  return sel;
}

function stageCard(s, panel) {
  const rows = [];
  const units = s.variants.flatMap((v) => v.units.map((u) => ({ v, u })));
  const n = units.length;
  units.forEach(({ v, u }, i) => {
    const chips = Object.entries(u.formats).map(([fmt, f]) => fmtChip(s.stage, v, fmt, f, panel));
    const subs = [];
    if (u.model_key && v.units.length > 1) subs.push(u.model_key);
    const extra = Object.entries(v.settings || {}).map(([k, val]) => `${k} ${val}`).join(" · ");
    if (extra) subs.push(extra);
    const state = s.active_state || s.state;   // the model the module runs, not necessarily its default
    const lead = i === 0 ? [
      el("td.mh-mod.span", { rowspan: n }, el("span.ink", stageLabel(s.stage)), el("span.id.mono", s.stage)),
      el("td.span", { rowspan: n }, el(`span.badge.${STATE_BADGE[state] || "info"}`, {
        title: s.active_state && s.active_state !== s.state ? `the active model is ${STATE_LABEL[s.active_state]}; the default is ${STATE_LABEL[s.state]}` : "",
      }, STATE_LABEL[state] || state || "?")),
      el("td.span", { rowspan: n }, activeSelect(s.stage, s, panel)),
    ] : [];
    rows.push(el("tr.v", ...lead,
      el("td.v.mh-k",
        el("a", { href: HF + u.repo_id, target: "_blank", rel: "noopener", title: "open the model card on Hugging Face" }, u.repo_id),
        v.default ? el("span.def", { title: "the default model for this module" }, "⚡") : null,
        subs.length ? el("span.sub", subs.join(" · ")) : null),
      el("td.v", el("span.mh-fmts", chips)),
      el("td.v.mh-rev.mono.dim", { title: u.revision || "" }, u.revision ? u.revision.slice(0, 10) : "—")));
  });
  const state = s.active_state || s.state;
  const tone = STATE_BADGE[state] === "ok" ? "good" : STATE_BADGE[state] === "warn" ? "warn" : STATE_BADGE[state] === "bad" ? "bad" : "good";
  return el(`div.card.mh-card.${tone}`, grid(rows));
}

/**
 * Mount a models panel. `compact` renders the banner form (Live Status): a one-line notice plus the
 * button, hidden entirely when everything is current. The full form (Models tab) renders the catalog.
 */
export function mountModelsPanel(host, { compact = false, fullWidth = false } = {}) {
  const wide = fullWidth ? { maxWidth: "none" } : {};
  // compact: the old one-card banner. full: a header block + a card per stage.
  const card = el(`div.card${compact ? ".warn" : ".mh-hdr"}`, { style: { margin: "0 0 8px", ...wide } });
  if (!compact) { card.classList.remove("card"); card.style.border = "0"; card.style.padding = "0 4px 6px"; }
  const title = el(`div${compact ? "" : ".mh-title"}`, { style: compact ? { display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" } : {} });
  const btn = el("button.btn.sm.primary", { type: "button" }, "Install Models from Hugging Face");
  const verifyBtn = el("button.btn.sm.ghost", { type: "button", title: "re-hash every installed file against the lock" }, "Verify hashes");
  const text = el("span", { style: { flex: "1 1 240px", fontSize: "13px" } });
  const lockChip = el("span.chip.mono", { style: { fontSize: "11px" } });
  const hint = el("p.hint", { style: { margin: "0 0 10px" } },
    "Only use onnx model formats unless you specifically require another format. Onnx is the default.");
  const headGrid = el("div");
  title.append(el("strong", compact ? "Models" : "Models from Hugging Face"), text,
    ...(compact ? [btn] : [el("span.spacer", { style: { flex: "1" } }), lockChip, btn, verifyBtn]));
  card.append(title, ...(compact ? [] : [hint, headGrid]));
  const cards = el("div");
  const legend = el("div.mh-legend.hint", { style: { display: compact ? "none" : "" } });
  const logBox = el("div.mh-log");
  host.appendChild(card);
  if (!compact) host.append(cards, legend);
  host.appendChild(logBox);
  // Settings-file problems about model paths live here too (the "things to check" card), so the
  // Settings tab is not the place a user hunts for a models problem.
  const warnCard = el("div.card.warn", { style: { margin: "0 0 8px", display: "none", ...wide } });
  if (!compact) host.appendChild(warnCard);

  const panel = {
    compact,
    render() {
      const st = lastStatus;
      const busy = !!activeTask;
      btn.textContent = busy ? (activeTask.label === "defaults" ? "Installing…" : `Installing ${activeTask.label}…`) : buttonLabel(st);
      btn.disabled = busy || (!needsAttention(st) && !!st && !st.error);
      verifyBtn.disabled = busy;
      if (!st) { text.textContent = "checking installed models…"; card.style.display = compact ? "none" : ""; return; }
      if (st.error) {
        text.textContent = `could not read model status: ${st.error}`;
        card.style.display = "";
        return;
      }
      const attention = needsAttention(st);
      if (compact) {
        card.classList.toggle("good", !attention);
        card.classList.toggle("warn", attention);
        card.style.display = !attention && !busy ? "none" : "";
      }
      const s = st.summary;
      text.textContent = attention
        ? (s.missing.length ? `${s.missing.length} required model${s.missing.length > 1 ? "s are" : " is"} missing` : `${s.outdated.length} model${s.outdated.length > 1 ? "s have" : " has"} an update available`)
          + ` — folder: ${st.root}`
        : (s.unavailable && s.unavailable.length
            ? `${s.unavailable.length} model${s.unavailable.length > 1 ? "s are" : " is"} not published yet (${s.unavailable.join(", ")}) — folder: ${st.root}`
            : `all default models installed — folder: ${st.root}`);
      if (compact) return;

      const cat = lastCatalog;
      lockChip.textContent = cat && cat.lm3_version ? `lock ${cat.lm3_version}` : "";
      clear(headGrid); clear(cards); clear(legend);
      if (!cat) { cards.appendChild(el("p.hint", "reading the model catalog…")); return; }
      if (cat.error) { cards.appendChild(el("div.card.bad", el("p", `could not read the model catalog: ${cat.error}`))); return; }
      headGrid.appendChild(grid([]));
      for (const stage of cat.stages) cards.appendChild(stageCard(stage, panel));
      legend.append(
        el("span.mh-fmt.get", "⤓ coreml"), el("span", "not installed — click to download"),
        el("span.mh-fmt.on", "✓ onnx"), el("span", "installed"),
        el("span.mh-fmt.on.act", "✓ onnx"), el("span", "installed and active in LM3"),
        el("span.mh-fmt.new", "↻ onnx"), el("span", "update available."),
        el("span", "⚡ the module's default model"), el("span", "🔒 locked — always runs its default."),
        el("span", "Variant names open the model card on Hugging Face."));

      clear(warnCard);
      const n = modelWarnings.length;
      warnCard.style.display = n ? "" : "none";
      if (n) {
        warnCard.append(
          el("h4", `${n} thing${n === 1 ? "" : "s"} to check`),
          el("p.hint", { style: { margin: "0 0 8px" } },
            "From the settings file: these model paths do not point at a file. Installing fixes a missing default; "
            + "a path that is wrong (or the wrong format) is edited under Settings."),
          el("ul", modelWarnings.map((w) => el("li", w.msg || String(w), w.path ? el("span.mono", { style: { marginLeft: "6px", color: "var(--acc2)" } }, w.path) : null))));
      }
    },
    log(msg, kind = "") {
      logBox.style.display = "";
      const line = el("div", { style: { color: kind === "bad" ? "var(--bad)" : kind === "ok" ? "var(--ok)" : kind === "dim" ? "var(--mute)" : "var(--ink)" } }, msg);
      logBox.appendChild(line);
      logBox.scrollTop = logBox.scrollHeight;
    },
    destroy() { panels.delete(panel); card.remove(); cards.remove(); legend.remove(); logBox.remove(); warnCard.remove(); stopPolling(); },
  };
  btn.addEventListener("click", () => installDefaults().then(() => panel.render()));
  verifyBtn.addEventListener("click", async () => {
    verifyBtn.disabled = true;
    try {
      lastStatus = await api.getModelsStatus(true);
      lastCatalog = await api.getModelsCatalog();
      panel.log(needsAttention(lastStatus) ? "verify: some files do not match the lock" : "verify: every installed file matches the lock", needsAttention(lastStatus) ? "bad" : "ok");
    } catch (err) { panel.log(`verify failed: ${err.message || err}`, "bad"); }
    verifyBtn.disabled = false;
    applyTabTone();
    for (const p of panels) p.render();
  });
  panels.add(panel);
  panel.render();
  refreshModelsStatus();

  // The yaml is the truth: while the full view is on screen, re-read it so a hand edit (or a save
  // from the Settings tab or the top strip) moves the selector without a reload.
  function onSaved() { if (!activeTask) refreshModelsStatus(); }
  function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = 0; } document.removeEventListener("lm3:settings-saved", onSaved); }
  if (!compact) {
    document.addEventListener("lm3:settings-saved", onSaved);
    pollTimer = setInterval(() => {
      const pane = host.closest(".tabpane");
      if (document.hidden || (pane && !pane.classList.contains("active")) || activeTask) return;
      refreshModelsStatus();
    }, CATALOG_POLL_MS);
  }
  return panel;
}
