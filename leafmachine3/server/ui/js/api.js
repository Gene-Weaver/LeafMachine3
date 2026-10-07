/* ==========================================================================
   LM3 — shared API client + DOM helpers
   --------------------------------------------------------------------------
   Every tab module imports from here. One place knows the Bearer token, the
   base URL, the error shape and the SSE reconnect rules, so no tab has to
   reinvent them (and no tab can get them subtly wrong).

   Plain ES module, no dependencies, no build step:
       import { api, el, fmtBytes } from "../api.js";

   Served by leafmachine3/server/app.py at "/" alongside index.html, so the
   default base URL is the same origin. The Electron shell just points a
   BrowserWindow at http://127.0.0.1:8765 and gets the identical code path.
   ========================================================================== */

/* ---------------------------------------------------------------- config -- */

const TOKEN_KEY = "lm3.token";

/**
 * Read a `<meta name="...">` value, or null.
 *
 * Guarded on `document` because this module is imported at the TOP of every renderer module, and
 * the renderer's pure decision logic (topbar.js `deriveView`) is unit-tested under plain node,
 * which has no DOM. Without the guard the import itself throws and the only way to test that logic
 * would be to copy it into the test -- i.e. to test a copy.
 */
function meta(name) {
  if (typeof document === "undefined" || !document.querySelector) return null;
  const m = document.querySelector(`meta[name="${name}"]`);
  const v = m && m.getAttribute("content");
  return v && v.trim() ? v.trim() : null;
}

/* The server injects <meta name="lm3-api-base"> only when the UI is served from
   somewhere other than the API origin. Normally it is empty -> same origin. */
const BASE = (meta("lm3-api-base") || "").replace(/\/+$/, "");

/**
 * Resolve the shared secret, in priority order:
 *   1. <meta name="lm3-token">  — what app.py injects into index.html
 *   2. ?token= in the page URL  — how the Electron shell hands it over
 *   3. localStorage             — remembered from a previous load
 * Whatever we find is persisted so a later reload without the meta still works.
 */
function resolveToken() {
  const fromMeta = meta("lm3-token");
  if (fromMeta) return remember(fromMeta);

  try {
    const q = new URLSearchParams(location.search).get("token");
    if (q) return remember(q);
  } catch { /* opaque origin (file://), or no window at all — fall through to storage */ }

  try {
    return localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

function remember(tok) {
  try { localStorage.setItem(TOKEN_KEY, tok); } catch { /* private mode, or no browser at all */ }
  return tok;
}

let TOKEN = resolveToken();

/** Current Bearer token (may be ""). */
export function getToken() { return TOKEN; }

/** Override the token at runtime (a settings field, a re-auth prompt). */
export function setToken(tok) { TOKEN = remember(String(tok || "")); return TOKEN; }

/** Absolute URL for an API path, with query params applied. */
export function url(path, params) {
  const p = path.startsWith("/") ? path : `/${path}`;
  const u = new URL(BASE + p, location.href);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v === undefined || v === null || v === "") continue;
      if (Array.isArray(v)) v.forEach((x) => u.searchParams.append(k, String(x)));
      else u.searchParams.set(k, String(v));
    }
  }
  return u.toString();
}


/* ----------------------------------------------------------------- errors -- */

/**
 * Every failure out of this module is an ApiError, so callers can branch on
 * `.status` / `.kind` instead of sniffing message strings.
 *   kind: "http"    — server answered with a non-2xx (status is meaningful)
 *         "network" — request never completed (server down, CORS, abort)
 *         "parse"   — 2xx body was not the JSON we expected
 */
export class ApiError extends Error {
  constructor(message, { status = 0, kind = "http", path = "", body = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.kind = kind;
    this.path = path;
    this.body = body;
  }
  /** 401/403 — the shared secret is wrong or missing. */
  get isAuth() { return this.status === 401 || this.status === 403; }
  /** The server is unreachable (not merely unhappy). */
  get isOffline() { return this.kind === "network"; }
  get isNotFound() { return this.status === 404; }
}


/* ---------------------------------------------------------------- request -- */

function headers(extra) {
  const h = { Accept: "application/json", ...(extra || {}) };
  if (TOKEN) h.Authorization = `Bearer ${TOKEN}`;
  return h;
}

/**
 * Core fetch wrapper. Returns parsed JSON (or null for 204/empty bodies).
 * `opts.timeout` (ms) aborts a stuck request; `opts.signal` is respected too.
 */
async function request(method, path, { body, params, timeout = 30000, signal, raw = false, headers: hx } = {}) {
  const target = url(path, params);
  const ctl = new AbortController();
  const timer = timeout > 0 ? setTimeout(() => ctl.abort(), timeout) : null;
  if (signal) signal.addEventListener("abort", () => ctl.abort(), { once: true });

  const init = { method, headers: headers(hx), signal: ctl.signal, cache: "no-store" };
  if (body !== undefined) {
    if (body instanceof FormData) {
      init.body = body;                       // let the browser set the boundary
    } else {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
  }

  let res;
  try {
    res = await fetch(target, init);
  } catch (err) {
    const aborted = err && err.name === "AbortError";
    throw new ApiError(
      aborted ? `request timed out after ${timeout} ms` : "cannot reach the LM3 server",
      { kind: "network", path, body: String(err && err.message || err) },
    );
  } finally {
    if (timer) clearTimeout(timer);
  }

  if (!res.ok) {
    // FastAPI puts the reason in {"detail": ...}; fall back to the raw text.
    let detail = res.statusText;
    let payload = null;
    try {
      const text = await res.text();
      if (text) {
        try {
          payload = JSON.parse(text);
          detail = payload.detail || payload.message || text;
        } catch { detail = text.slice(0, 400); }
      }
    } catch { /* body already consumed / unreadable */ }
    throw new ApiError(`${method} ${path} — ${res.status} ${detail}`,
                       { status: res.status, kind: "http", path, body: payload });
  }

  if (raw) return res;
  if (res.status === 204) return null;

  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    throw new ApiError(`${method} ${path} — response was not JSON`,
                       { status: res.status, kind: "parse", path, body: text.slice(0, 400) });
  }
}

export const get = (path, opts) => request("GET", path, opts);
export const post = (path, body, opts) => request("POST", path, { ...opts, body });
export const put = (path, body, opts) => request("PUT", path, { ...opts, body });
export const patch = (path, body, opts) => request("PATCH", path, { ...opts, body });
export const del = (path, opts) => request("DELETE", path, opts);


/* -------------------------------------------------------------------- SSE -- */

/**
 * Subscribe to a Server-Sent-Events endpoint. Returns a `close()` function.
 *
 * NOTE ON AUTH: EventSource cannot set request headers, so the Bearer token
 * goes on the query string as `?token=...` instead. That is safe here because
 * the server binds loopback only and the token never leaves this machine — but
 * it is WHY these endpoints must accept the token both ways (header for
 * fetch(), query param for EventSource).
 *
 * The browser reconnects EventSource on its own after a drop; we surface each
 * drop through `onError` and pass `retries` so a caller can show "reconnecting…"
 * and give up after N attempts. Anything non-JSON in a frame is handed to
 * `onMessage` verbatim as `{ raw }` rather than thrown away.
 *
 * @param {string} path
 * @param {{onMessage?:Function, onError?:Function, onOpen?:Function,
 *          events?:string[], params?:object, maxRetries?:number}} handlers
 * @returns {Function} close()
 */
export function sse(path, { onMessage, onError, onOpen, events, params, maxRetries = Infinity } = {}) {
  const src = new EventSource(url(path, { ...(params || {}), token: TOKEN || undefined }));
  let closed = false;
  let retries = 0;

  const deliver = (ev) => {
    if (closed || !onMessage) return;
    let data = ev.data;
    if (typeof data === "string" && data.length) {
      try { data = JSON.parse(data); } catch { data = { raw: ev.data }; }
    }
    try { onMessage(data, ev); } catch (err) { console.error("[lm3] SSE handler threw", err); }
  };

  src.onopen = () => {
    retries = 0;
    if (!closed && onOpen) onOpen();
  };
  src.onmessage = deliver;                       // unnamed "data:" frames
  for (const name of events || []) src.addEventListener(name, deliver);

  src.onerror = () => {
    if (closed) return;
    retries += 1;
    if (onError) {
      onError(new ApiError(`stream ${path} dropped`, { kind: "network", path }), retries);
    }
    if (retries > maxRetries) close();
  };

  function close() {
    if (closed) return;
    closed = true;
    try { src.close(); } catch { /* already gone */ }
  }
  close.source = src;                            // escape hatch for readyState checks
  return close;
}


/* =========================================================================
   ENDPOINTS
   One named method per route. Tabs call these, never fetch() directly, so a
   path change lands in exactly one file.
   ========================================================================= */

export const api = {
  /* -- identity / plumbing ---------------------------------------------- */
  get token() { return TOKEN; },
  setToken,
  url,
  get, post, put, patch, del, sse,

  /** Liveness + which ONNX provider would bind. Unauthenticated. */
  health: () => get("/healthz", { timeout: 5000 }),

  /** The tuned hardware_settings.yaml (cores, RAM, GPUs, per-module sizing). */
  getHardware: () => get("/v1/hardware"),

  /* -- settings ---------------------------------------------------------- */
  /** The whole LM3_settings.yaml as a nested plain object. */
  getSettings: () => get("/v1/settings"),

  /**
   * Persist LM3_settings.yaml. Send the FULL nested object — the server writes
   * it wholesale so partial posts would drop keys.
   */
  putSettings: (settings) => put("/v1/settings", settings),

  /**
   * Per-key metadata driving the settings tab: type, choices, range, unit,
   * human label, help text, and `important: true` for the green-shaded rows.
   */
  getSettingsMeta: () => get("/v1/settings/meta"),

  /* -- models (Hugging Face installer) ----------------------------------- */
  /** Per-action install state + the resolved models folder; `verify` re-hashes files. */
  getModelsStatus: (verify = false) => get("/v1/models/status", { params: verify ? { verify: 1 } : undefined, timeout: 60000 }),
  /** Start an install; {task_id}. Body: {actions?, formats?, force?}. */
  installModels: (body = {}) => post("/v1/models/install", body),
  /** Snapshot of an install task (state, events since `since`). */
  getModelsInstall: (taskId, since = 0) => get(`/v1/models/install/${encodeURIComponent(taskId)}`, { params: { since } }),
  /** Live progress events for an install task; ends with a "done" event. */
  streamModelsInstall: (taskId, handlers) => sse(`/v1/models/install/${encodeURIComponent(taskId)}/events`, { ...handlers, events: ["progress", "done", "ping"] }),

  /* -- machine performance ----------------------------------------------- */
  /** One-shot sample: CPU, RAM, GPU, VRAM, disk. */
  getMetrics: () => get("/v1/metrics", { timeout: 8000 }),

  /** Continuous samples for the pinned bottom perf panel. */
  streamMetrics: (handlers) => sse("/v1/metrics/stream", handlers),

  /* -- the runtime registry ------------------------------------------------ */
  /**
   * THE canonical runtime view (plan section 5): `{active, last, next_run_settings, server,
   * diagnostics}`. Identity comes from the deployment's lease record -- never from the settings
   * YAML and never from a filesystem recency guess (invariant 5).
   *
   * 404 means this server predates the route (or `LM3_RUNTIME_V2` is off). Callers fall back to
   * `GET /v1/run/active`, which describes only runs THIS server launched; topbar.js marks that
   * view `supported: false` so nothing pretends the two are the same answer.
   */
  getRuntime: () => get("/v1/runtime", { timeout: 10000 }),

  /** The legacy per-server run record. Compatibility projection; superseded by getRuntime(). */
  getActiveRun: () => get("/v1/run/active", { timeout: 10000 }),

  /* -- live run status ---------------------------------------------------- */
  /**
   * Current LM3 run: modules, per-module counts, per-worker slots.
   * `opts` may pin the answer to one run: `{db}` (a ledger path) or `{run}` (a run name/id).
   * Unpinned, the server answers for whatever `resolve_run()` follows -- the active record first.
   */
  getStatus: (opts) => get("/v1/status", { timeout: 8000, params: opts || undefined }),

  /** Same shape as getStatus(), pushed as it changes. Accepts the same `params: {db|run}` pin. */
  streamStatus: (handlers) => sse("/v1/status/stream", handlers),

  /** The log stream that feeds the console pane. Accepts the same `params: {db|run}` pin. */
  streamLogs: (handlers) => sse("/v1/logs/stream", handlers),

  /* -- runs / results ----------------------------------------------------- */
  /** Every run directory the server can see, newest first. */
  listRuns: (opts) => get("/v1/runs", { params: opts || undefined }),

  /**
   * The run list PLUS which row is live: `{runs, active, last, follow, selected_default,
   * runtime}`. `active` is non-null only when the deployment lease is held and the record is
   * compatible, so it is the one field a renderer may read as "a run is happening right now".
   *
   * Two literal path segments (`/-/selector`) so it can neither shadow nor be shadowed by
   * `GET /v1/runs/{run}`. 404 on a server that predates it -- callers fall back to listRuns().
   */
  runSelector: (opts) => get("/v1/runs/-/selector", { params: opts || undefined }),

  /**
   * Media produced by a run.
   * @param {string} run  run name
   * @param {{dir?:string, kind?:string, limit?:number, offset?:number}} [opts]
   *        dir = subfolder to list, kind = "image" | "mask" | "csv" | ...
   */
  getResults: (run, opts) => get(`/v1/runs/${encodeURIComponent(run)}/media`, { params: opts }),

  /**
   * Direct URL for one media file — for <img src>, <a download> and the
   * lightbox. Same EventSource problem: an <img> cannot send an Authorization
   * header, so the token rides the query string.
   */
  mediaUrl: (run, path, params) =>
    url(`/v1/runs/${encodeURIComponent(run)}/media/file`,
        { path, token: TOKEN || undefined, ...(params || {}) }),

  /* -- project DB inspector ------------------------------------------------ */
  /** Every table in the run's SQLite ledger: name, row count, columns. */
  getTables: (run) => get(`/v1/runs/${encodeURIComponent(run)}/db/tables`),

  /**
   * A page of rows from one table.
   * @param {{limit?:number, offset?:number, order?:string, dir?:"asc"|"desc",
   *          q?:string}} [opts]
   */
  getTable: (run, table, opts) =>
    get(`/v1/runs/${encodeURIComponent(run)}/db/table/${encodeURIComponent(table)}`, { params: opts }),

  /* -- postprocessing ------------------------------------------------------ */
  /** Available post-run tools (id, name, description, parameter schema). */
  listTools: () => get("/v1/postprocess/tools"),

  /**
   * Run one tool. Long jobs answer with {job_id} — poll getJob() / stream its
   * events; short ones answer with their result inline.
   */
  runTool: (tool, params) => post("/v1/postprocess/run", { tool, params: params || {} },
                                  { timeout: 0 }),

  /* -- jobs ---------------------------------------------------------------- */
  /**
   * Start an LM3 run from the saved LM3_settings.yaml. Optional overrides:
   * {config_path, input_dir, output_dir, restart}.
   *
   * NOTE: this is /v1/run/start (metrics_api), NOT /v1/jobs. /v1/jobs is the older
   * upload-a-batch job API and does not launch a run from the current settings.
   */
  startRun: (payload) => post("/v1/run/start", payload || {}, { timeout: 60000 }),

  /**
   * Ask the server to stop the active run. LM3 is resumable, so it drains to a checkpointed
   * state and can be continued later. `grace_s` is how long to wait before escalating to a kill.
   */
  stopRun: (opts) => post("/v1/run/stop", { grace_s: 10, ...(opts || {}) }, { timeout: 40000 }),

  /** Job record: state, per-module counters, images done/total. */
  getJob: (jobId) => get(`/v1/jobs/${encodeURIComponent(jobId)}`),

  /** Detections / CF / masks / grounded areas for a finished job. */
  getJobResults: (jobId) => get(`/v1/jobs/${encodeURIComponent(jobId)}/results`),

  /** Per-job progress frames straight off its project ledger. */
  streamJobEvents: (jobId, handlers) =>
    sse(`/v1/jobs/${encodeURIComponent(jobId)}/events`, handlers),

  /* -- hardware profiler ---------------------------------------------------- */
  /** Re-run the hardware profiler that sizes every module. */
  runSetup: (opts) => post("/v1/setup", undefined,
                           { params: { optimize: true, force: false, ...(opts || {}) } }),

  /** Profiler progress (polled — the profiler reports in coarse steps). */
  getSetupEvents: (jobId) => get("/v1/setup/events", { params: { jid: jobId } }),
};


/* =========================================================================
   DOM + FORMAT HELPERS
   Deliberately tiny. Enough to build the app without a framework, not enough
   to become one.
   ========================================================================= */

/**
 * Create an element.
 *   el("div")                              -> <div>
 *   el("div.card.good")                    -> <div class="card good">
 *   el("span#id.mono", "42")               -> <span id="id" class="mono">42</span>
 *   el("button.btn", {onclick: f}, "Run")  -> attrs object is optional
 *
 * Attribute keys: `class`/`className`, `style` (string or object), `dataset`
 * (object), `on*` (event listeners), anything else becomes an attribute.
 * Children may be nodes, strings, numbers, or nested arrays; null/false/
 * undefined are skipped so `cond && el(...)` works inline.
 */
export function el(spec, ...rest) {
  const m = /^([a-zA-Z][\w-]*)?(#[\w:-]+)?((?:\.[\w-]+)*)$/.exec(spec || "div");
  if (!m) throw new Error(`el(): bad spec "${spec}"`);
  const node = document.createElement(m[1] || "div");
  if (m[2]) node.id = m[2].slice(1);
  if (m[3]) node.className = m[3].slice(1).split(".").join(" ");

  let children = rest;
  const first = rest[0];
  const isAttrs = first && typeof first === "object" && !Array.isArray(first)
    && !(first instanceof Node);
  if (isAttrs) {
    children = rest.slice(1);
    for (const [k, v] of Object.entries(first)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class" || k === "className") {
        node.className = node.className ? `${node.className} ${v}` : String(v);
      } else if (k === "style" && typeof v === "object") {
        Object.assign(node.style, v);
      } else if (k === "dataset") {
        Object.assign(node.dataset, v);
      } else if (k.startsWith("on") && typeof v === "function") {
        node.addEventListener(k.slice(2).toLowerCase(), v);
      } else if (k === "html") {
        node.innerHTML = v;                      // caller owns the sanitizing
      } else if (v === true) {
        node.setAttribute(k, "");
      } else {
        node.setAttribute(k, String(v));
      }
    }
  }
  append(node, children);
  return node;
}

/** Append children (nodes / strings / arrays / nullish) to a parent. */
export function append(parent, children) {
  for (const c of Array.isArray(children) ? children : [children]) {
    if (c === null || c === undefined || c === false || c === true) continue;
    if (Array.isArray(c)) append(parent, c);
    else parent.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return parent;
}

/** Remove every child of a node (faster and safer than innerHTML = ""). */
export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export const qs = (sel, root = document) => root.querySelector(sel);
export const qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Escape text for the rare spot where innerHTML is genuinely easier. */
export function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}


/* -------------------------------------------------------------- formatting */

/** Bytes as a compact human string: 1536 -> "1.5 KB". */
export function fmtBytes(bytes, digits = 1) {
  if (bytes === null || bytes === undefined || Number.isNaN(Number(bytes))) return "–";
  let n = Number(bytes);
  const neg = n < 0;
  n = Math.abs(n);
  if (n < 1024) return `${neg ? "-" : ""}${n.toFixed(0)} B`;
  const units = ["KB", "MB", "GB", "TB", "PB"];
  let i = -1;
  do { n /= 1024; i += 1; } while (n >= 1024 && i < units.length - 1);
  // whole numbers past a GB read better without a trailing .0
  const d = n >= 100 ? 0 : digits;
  return `${neg ? "-" : ""}${n.toFixed(d)} ${units[i]}`;
}

/** Megabytes (what the LM3 samplers report) as GB/MB, matching timing.py. */
export function fmtMB(mb, digits = 1) {
  if (mb === null || mb === undefined || Number.isNaN(Number(mb))) return "–";
  const n = Number(mb);
  if (n < 1) return "0";
  return n >= 1024 ? `${(n / 1024).toFixed(digits)} GB` : `${n.toFixed(0)} MB`;
}

/**
 * Seconds as a duration.
 *   fmtDuration(9.4)    -> "9.4s"
 *   fmtDuration(94)     -> "1m 34s"
 *   fmtDuration(9400)   -> "2h 36m"
 *   fmtDuration(94, {clock:true}) -> "01:34"
 */
export function fmtDuration(seconds, { clock = false, compact = true } = {}) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "–";
  let s = Math.max(0, Number(seconds));
  if (clock) {
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
    const pad = (v) => String(v).padStart(2, "0");
    return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${pad(m)}:${pad(sec)}`;
  }
  if (s < 1) return `${(s * 1000).toFixed(0)}ms`;
  if (s < 60) return `${s < 10 ? s.toFixed(1) : s.toFixed(0)}s`;
  const parts = [];
  const d = Math.floor(s / 86400); s -= d * 86400;
  const h = Math.floor(s / 3600);  s -= h * 3600;
  const m = Math.floor(s / 60);    s -= m * 60;
  if (d) parts.push(`${d}d`);
  if (h) parts.push(`${h}h`);
  if (m && parts.length < 2) parts.push(`${m}m`);
  if (!d && !h && parts.length < 2) parts.push(`${Math.floor(s)}s`);
  return compact ? parts.slice(0, 2).join(" ") : parts.join(" ");
}

/**
 * A number for display: thousands separators, compact suffixes past 10k when
 * asked, and precision that scales DOWN as magnitude goes up — three decimals
 * below 1, none past 100 — because a dense readout wants "1,235" not
 * "1,234.5217". Pass `digits` to override.
 *   fmtNum(1234.5)                 -> "1,235"
 *   fmtNum(1234.5, {digits:1})     -> "1,234.5"
 *   fmtNum(1234567, {compact:true})-> "1.2M"
 *   fmtNum(0.0421)                 -> "0.042"
 */
export function fmtNum(value, { digits = null, compact = false, pad = false } = {}) {
  if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return "–";
  const n = Number(value);
  if (!Number.isFinite(n)) return n > 0 ? "∞" : "-∞";
  if (compact && Math.abs(n) >= 10000) {
    const units = [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "k"]];
    for (const [div, suf] of units) {
      if (Math.abs(n) >= div) return `${(n / div).toFixed(Math.abs(n / div) >= 100 ? 0 : 1)}${suf}`;
    }
  }
  let d = digits;
  if (d === null) {
    const a = Math.abs(n);
    if (Number.isInteger(n)) d = 0;
    else if (a >= 100) d = 0;
    else if (a >= 10) d = 1;
    else if (a >= 1) d = 2;
    else d = 3;
  }
  const out = n.toLocaleString("en-US", { minimumFractionDigits: pad ? d : 0,
                                          maximumFractionDigits: d });
  return out;
}

/** A 0–1 ratio (or an already-scaled percent) as "42%" / "42.3%". */
export function fmtPct(value, { digits = 0, of = 1 } = {}) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "–";
  const pct = of === 1 ? Number(value) * 100 : (Number(value) / of) * 100;
  return `${pct.toFixed(digits)}%`;
}

/** A unix timestamp (seconds or ms) as a local wall-clock time. */
export function fmtTime(ts, { withDate = false } = {}) {
  if (!ts) return "–";
  const ms = Number(ts) > 1e11 ? Number(ts) : Number(ts) * 1000;
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return "–";
  const t = d.toLocaleTimeString("en-US", { hour12: false });
  return withDate ? `${d.toLocaleDateString("en-US")} ${t}` : t;
}

/** Shorten a long path to "…/parent/leaf" for tight labels. */
export function fmtPath(path, keep = 2) {
  const s = String(path ?? "");
  const parts = s.split(/[/\\]/).filter(Boolean);
  if (parts.length <= keep) return s;
  return `…/${parts.slice(-keep).join("/")}`;
}


/* ------------------------------------------------------------------ timing */

/**
 * Trailing debounce. `fn` fires `ms` after the last call.
 * The returned function carries `.cancel()` and `.flush()`.
 */
export function debounce(fn, ms = 200) {
  let timer = null, lastArgs = null, lastThis = null;
  const wrapped = function (...args) {
    lastArgs = args; lastThis = this;
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => { timer = null; fn.apply(lastThis, lastArgs); }, ms);
  };
  wrapped.cancel = () => { if (timer) clearTimeout(timer); timer = null; };
  wrapped.flush = () => {
    if (!timer) return;
    clearTimeout(timer); timer = null;
    fn.apply(lastThis, lastArgs);
  };
  return wrapped;
}

/** Leading-edge throttle: at most one call per `ms`, trailing call preserved. */
export function throttle(fn, ms = 100) {
  let last = 0, timer = null, lastArgs = null, lastThis = null;
  const wrapped = function (...args) {
    const now = Date.now();
    lastArgs = args; lastThis = this;
    if (now - last >= ms) {
      last = now; fn.apply(lastThis, lastArgs);
    } else if (!timer) {
      timer = setTimeout(() => {
        timer = null; last = Date.now(); fn.apply(lastThis, lastArgs);
      }, ms - (now - last));
    }
  };
  wrapped.cancel = () => { if (timer) clearTimeout(timer); timer = null; };
  return wrapped;
}

/**
 * Coalesce repeated calls to one per animation frame. Use this for anything
 * that redraws a canvas from a live stream — metric frames arrive faster than
 * the display refreshes, and drawing per frame just burns CPU.
 */
export function rafThrottle(fn) {
  let queued = false, lastArgs = null;
  const wrapped = (...args) => {
    lastArgs = args;
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; fn(...lastArgs); });
  };
  wrapped.cancel = () => { queued = false; };
  return wrapped;
}

/** await sleep(250) — for retry backoff and staged reveals. */
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export default api;
