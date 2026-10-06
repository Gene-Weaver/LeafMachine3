/**
 * models.js — the "Install Models from Hugging Face" affordance, shared by the Settings tab (a card
 * at the top of the pane) and the Live Status tab (a banner that only shows while something is
 * missing or outdated).
 *
 * Both call `mountModelsPanel(host, {compact})`; the panel owns its own refresh loop and talks to
 * `/v1/models/*` through api.js. One install at a time: the server returns the running task if a
 * second press lands while one is in flight, and every mounted panel follows the same stream.
 */
import { api, el, clear, fmtBytes } from "./api.js";

const STATE_LABEL = {
  current: "up to date",
  missing: "missing",
  outdated: "newer available",
  pending: "local copy (not published yet)",
  unavailable: "not published yet",
};
const STATE_BADGE = { current: "ok", missing: "bad", outdated: "warn", pending: "info", unavailable: "info" };

/* a tiny shared store so several panels stay in step */
const panels = new Set();
let lastStatus = null;
let activeTask = null;       // {id, close} while an install streams

export async function refreshModelsStatus() {
  try {
    lastStatus = await api.getModelsStatus();
  } catch (err) {
    lastStatus = { error: err && err.message ? err.message : String(err) };
  }
  for (const p of panels) p.render();
  return lastStatus;
}

/** Button text per the spec: missing → install; present but older → newer available. */
export function buttonLabel(st) {
  if (!st || st.error) return "Install Models from Hugging Face";
  return st.summary && st.summary.button_label ? st.summary.button_label : "Install Models from Hugging Face";
}

function needsAttention(st) {
  return !!(st && st.summary && st.summary.needs_attention);
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

function confirmInstall(st) {
  const root = st && st.root ? st.root : "(models folder)";
  const existing = st && st.actions
    ? Object.entries(st.actions).filter(([, a]) => a.files.some((f) => f.present) && !a.placeholder).map(([k]) => k)
    : [];
  return new Promise((resolve) => {
    openModal({
      title: "Install models from Hugging Face",
      build: Object.assign((panel, close) => {
        panel.appendChild(el("div.panel-bd",
          el("p", { style: { margin: "0 0 8px", color: "var(--ink)" } },
            "We're about to download the default LeafMachine3 models and overwrite any existing models in:"),
          el("p.mono", { style: { margin: "0 0 10px", wordBreak: "break-all", fontSize: "12.5px" } }, root),
          el("p", { style: { margin: 0, fontSize: "12.5px", color: "var(--mute)", lineHeight: "1.55" } },
            existing.length
              ? `Existing: ${existing.join(", ")}. Each file is renamed to .backup before its replacement is written and restored automatically if the download or the check fails, so a failed update never leaves the folder broken.`
              : "Nothing is installed there yet. Files are downloaded, hash-checked, then moved into place."),
        ));
        const ok = el("button.btn.sm.primary", { type: "button" }, "Download and install");
        ok.addEventListener("click", () => close(true));
        panel.appendChild(el("div.panel-ft", { style: { display: "flex", gap: "8px", justifyContent: "flex-end" } },
          el("button.btn.sm.ghost", { type: "button", onclick: () => close(false) }, "Cancel"), ok));
        setTimeout(() => ok.focus(), 0);
      }, { onClose: (r) => resolve(r === true) }),
    });
  });
}

/* ------------------------------------------------------------ install -- */
async function startInstall() {
  if (activeTask) return activeTask;
  const st = lastStatus && !lastStatus.error ? lastStatus : await refreshModelsStatus();
  if (!(await confirmInstall(st))) return null;
  let res;
  try {
    res = await api.installModels({});
  } catch (err) {
    for (const p of panels) p.log(`could not start: ${err.message || err}`, "bad");
    return null;
  }
  const close = api.streamModelsInstall(res.task_id, {
    onMessage: (ev) => {
      if (!ev || typeof ev !== "object") return;
      if (ev.type === "start") {
        for (const p of panels) p.log(`installing ${ev.actions.length} model(s), ${fmtBytes(ev.total_bytes || 0)} …`);
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
        for (const p of panels) p.log(ok ? "all models are installed and verified" : `install ${ev.state}: ${ev.error || ""}`, ok ? "ok" : "bad");
        finish();
      }
    },
    onError: () => { /* EventSource retries; the done frame or a snapshot poll ends it */ },
  });
  activeTask = { id: res.task_id, close };
  for (const p of panels) p.render();
  async function finish() {
    if (activeTask) { try { activeTask.close(); } catch { /* noop */ } }
    activeTask = null;
    await refreshModelsStatus();
  }
  return activeTask;
}

/* -------------------------------------------------------------- panel -- */
/**
 * Mount a models panel. `compact` renders the banner form (Live Status): a one-line notice plus the
 * button, hidden entirely when everything is current. The full form (Settings) always shows and
 * lists every action with its state.
 */
export function mountModelsPanel(host, { compact = false, fullWidth = false } = {}) {
  const card = el(`div.card${compact ? ".warn" : ""}`, { style: { margin: "0 0 8px", ...(fullWidth ? { maxWidth: "none" } : {}) } });
  const title = el("div", { style: { display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" } });
  const btn = el("button.btn.sm.primary", { type: "button" }, "Install Models from Hugging Face");
  const text = el("span", { style: { flex: "1 1 240px", fontSize: "13px" } });
  const list = el("div", { style: { marginTop: "8px", display: compact ? "none" : "block" } });
  const logBox = el("div.mono", { style: { marginTop: "6px", fontSize: "12px", maxHeight: "120px", overflow: "auto", display: "none" } });
  title.append(el("strong", compact ? "Models" : "Models from Hugging Face"), text, btn);
  card.append(title, list, logBox);
  host.appendChild(card);

  const panel = {
    render() {
      const st = lastStatus;
      const busy = !!activeTask;
      btn.textContent = busy ? "Installing…" : buttonLabel(st);
      btn.disabled = busy;
      if (!st) { text.textContent = "checking installed models…"; card.style.display = compact ? "none" : ""; return; }
      if (st.error) {
        text.textContent = `could not read model status: ${st.error}`;
        card.style.display = "";
        return;
      }
      const attention = needsAttention(st);
      card.style.display = compact && !attention && !busy ? "none" : "";
      const s = st.summary;
      text.textContent = attention
        ? (s.missing.length ? `${s.missing.length} required model${s.missing.length > 1 ? "s are" : " is"} missing` : `${s.outdated.length} model${s.outdated.length > 1 ? "s have" : " has"} a newer version`)
          + ` — folder: ${st.root}`
        : `all default models installed — folder: ${st.root}`;
      if (!compact) {
        clear(list);
        const tbl = el("table.tbl", { style: { width: "100%", fontSize: "12.5px" } },
          el("thead", el("tr", el("th", "Module"), el("th", "State"), el("th", "Detail"), el("th", "Hub revision"))));
        const body = el("tbody");
        for (const [k, a] of Object.entries(st.actions)) {
          const rev = Object.values(a.lock_revision || {})[0];
          body.appendChild(el("tr",
            el("td.mono", k),
            el("td", el(`span.badge.${STATE_BADGE[a.state] || "info"}`, STATE_LABEL[a.state] || a.state)),
            el("td", { style: { color: "var(--mute)" } }, a.detail || ""),
            el("td.mono", { style: { color: "var(--mute)" } }, rev ? rev.slice(0, 10) : "—")));
        }
        tbl.appendChild(body);
        list.appendChild(tbl);
        list.appendChild(el("p.hint", { style: { margin: "6px 0 0" } },
          "ONNX runtime models pinned by this LM3 release. Missing or outdated models are downloaded, hash-checked, "
          + "and swapped in with a .backup of anything replaced; a failure restores the previous files. "
          + "Command line: lm3 models install  (or: uv run install_models.py)."));
      }
    },
    log(msg, kind = "") {
      logBox.style.display = "";
      const line = el("div", { style: { color: kind === "bad" ? "var(--bad)" : kind === "ok" ? "var(--ok)" : kind === "dim" ? "var(--mute)" : "var(--ink)" } }, msg);
      logBox.appendChild(line);
      logBox.scrollTop = logBox.scrollHeight;
    },
    destroy() { panels.delete(panel); card.remove(); },
  };
  btn.addEventListener("click", () => startInstall());
  panels.add(panel);
  panel.render();
  if (!lastStatus) refreshModelsStatus(); else panel.render();
  return panel;
}
