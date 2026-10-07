/* ==========================================================================
   LM3 — machine performance monitor
   --------------------------------------------------------------------------
   The panel pinned across the bottom of the app, outside every tab. It answers
   one question continuously: is this machine actually working, and on what.

   Three sizes, remembered in localStorage:
     collapsed  38px  header only — one sparkline + one number per resource
     compact   186px  the plot row: CPU, RAM, GPU, VRAM, disk I/O
     detailed  ~40vh  the plot row plus the per-core heatmap, the per-GPU
                      tiles and the system readouts

   Data comes from leafmachine3/server/metrics.py: GET /v1/metrics/history
   seeds the five-minute window on load, GET /v1/metrics/stream keeps it live
   at the sampler's own 2 Hz. Both are handled here; nothing else in the app
   samples hardware.

   Everything is drawn by hand into <canvas> — no plotting library, no build
   step, works offline in the Electron shell exactly as it does in a browser.
   ========================================================================== */

import { api, el, clear, fmtDuration, fmtNum, rafThrottle } from "./api.js";


/* ---------------------------------------------------------------- config -- */

/* The sampler's own ring is 300 s wide (LM3_METRICS_WINDOW_S), so drawing the
   same window means a reload shows exactly what the server remembers. */
const WINDOW_S = 300;
const TICK_S = 60;                 /* one vertical rule per minute            */

const PAD_T = 3;                   /* px of headroom so a 100% line is not clipped */
const PAD_B = 1;

const LS_COLLAPSED = "lm3.perf.collapsed";
const LS_DETAIL = "lm3.perf.detail";

/* GPU identity is colored consistently across the util plot, the VRAM plot and
   the tiles: GPU 0 is always orange, GPU 1 always sky, and so on. Mixing the
   per-cell accent in instead would make "which card is that line?" unanswerable
   on a two-card box. */
const GPU_COLORS = ["--acc", "--acc2", "--vio", "--acc3", "--warn", "--bad"];


/* ------------------------------------------------------- design tokens ---- */

/* Read the palette out of the live stylesheet rather than hard-coding hexes, so
   the canvases stay in step with app.css if a token is ever retuned. */
const TOKENS = {};
function tok(name, fallback) {
  if (TOKENS[name] === undefined) {
    let v = "";
    try { v = getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
    catch { /* detached document */ }
    TOKENS[name] = v || fallback;
  }
  return TOKENS[name];
}

/** #rgb / #rrggbb / rgb(...) -> [r,g,b]; anything unparseable -> mid gray. */
function rgbOf(color) {
  const s = String(color || "").trim();
  if (s.startsWith("#")) {
    const h = s.slice(1);
    if (h.length === 3) return [parseInt(h[0] + h[0], 16), parseInt(h[1] + h[1], 16), parseInt(h[2] + h[2], 16)];
    if (h.length >= 6) return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  const m = /rgba?\(([^)]+)\)/.exec(s);
  if (m) {
    const p = m[1].split(/[,\s/]+/).filter(Boolean).map(Number);
    if (p.length >= 3) return [p[0], p[1], p[2]];
  }
  return [111, 117, 127];
}

function rgba(color, alpha) {
  const [r, g, b] = rgbOf(color);
  return `rgba(${r},${g},${b},${alpha})`;
}

/** Linear blend of two colors, t in 0..1 — the core heatmap ramp. */
function mix(a, b, t) {
  const A = rgbOf(a), B = rgbOf(b);
  const k = Math.max(0, Math.min(1, t));
  return `rgb(${Math.round(A[0] + (B[0] - A[0]) * k)},${Math.round(A[1] + (B[1] - A[1]) * k)},${Math.round(A[2] + (B[2] - A[2]) * k)})`;
}


/* ------------------------------------------------------------ formatting -- */

const isNum = (v) => typeof v === "number" && Number.isFinite(v);

function pct(v, digits) {
  if (!isNum(v)) return "–";
  const d = digits === undefined ? (v >= 99.95 || v >= 10 ? 0 : 1) : digits;
  return `${v.toFixed(d)}%`;
}

/** Megabytes as GB with app-density precision ("503 GB", "44.4 GB", "612 MB"). */
function gb(mb, { whole = false } = {}) {
  if (!isNum(mb)) return "–";
  if (mb < 1024 && !whole) return `${mb.toFixed(0)} MB`;
  const g = mb / 1024;
  if (whole) return `${g >= 10 ? g.toFixed(0) : g.toFixed(1)} GB`;
  return `${g >= 100 ? g.toFixed(0) : g.toFixed(1)} GB`;
}

function mbps(v) {
  if (!isNum(v)) return "–";
  if (v >= 100) return `${v.toFixed(0)} MB/s`;
  if (v >= 10) return `${v.toFixed(1)} MB/s`;
  return `${v.toFixed(2)} MB/s`;
}

/**
 * The window summary under each plot header.
 *
 * The unit is printed ONCE, on the last figure: a cell is ~300 px wide and
 * "min 128 GB · avg 130 GB · max 132 GB" does not fit, so it would ellipsize
 * away the very number that matters. `kind` also picks one unit for all three
 * figures off the maximum, so min and max are never in different units.
 */
function statLine(st, kind) {
  if (!st) return { text: "", title: "" };
  let scale = 1, unit = "", digits = 0;
  if (kind === "pct") {
    unit = "%";
    digits = st.max < 10 ? 1 : 0;
  } else if (kind === "mb") {
    const asGb = st.max >= 1024;
    scale = asGb ? 1 / 1024 : 1;
    unit = asGb ? " GB" : " MB";
    digits = asGb ? (st.max / 1024 >= 100 ? 0 : 1) : 0;
  } else {
    unit = " MB/s";
    digits = st.max >= 100 ? 0 : st.max >= 10 ? 1 : 2;
  }
  const f = (v) => (v * scale).toFixed(digits);
  return {
    text: `min ${f(st.min)} · avg ${f(st.avg)} · max ${f(st.max)}${unit}`,
    title: `Over the last 5 minutes (${st.n} samples): minimum ${f(st.min)}${unit}, `
         + `average ${f(st.avg)}${unit}, maximum ${f(st.max)}${unit}`,
  };
}

/** 1 -> 1, 1.3 -> 2, 37 -> 50, 640 -> 1000. Keeps an auto axis on round steps. */
function niceCeil(v) {
  if (!(v > 0)) return 1;
  const e = Math.pow(10, Math.floor(Math.log10(v)));
  const m = v / e;
  const s = m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10;
  return s * e;
}


/* =========================================================================
   RING STORE
   One shared timeline plus a Float64Array per series. Fixed capacity, no
   allocation per sample, and a missing reading is a NaN rather than a hole,
   so the plotter can break the line exactly where the sampler lost a device.
   ========================================================================= */

class Store {
  constructor(cap = 900) {
    this.cap = cap;
    this.n = 0;
    this.head = 0;                 /* next write slot */
    this.t = new Float64Array(cap);
    this.s = new Map();            /* key -> Float64Array(cap) */
  }

  _blank() {
    const a = new Float64Array(this.cap);
    a.fill(NaN);
    return a;
  }

  key(name) {
    let a = this.s.get(name);
    if (!a) { a = this._blank(); this.s.set(name, a); }
    return a;
  }

  /** Physical index of logical position p (0 = oldest retained sample). */
  idx(p) { return (this.head - this.n + p + this.cap) % this.cap; }

  push(t, values) {
    const i = this.head;
    this.t[i] = t;
    for (const a of this.s.values()) a[i] = NaN;      // a series absent from this
    for (const k in values) {                         // sample must not inherit
      const v = values[k];                            // the previous reading
      this.key(k)[i] = isNum(v) ? v : NaN;
    }
    this.head = (i + 1) % this.cap;
    if (this.n < this.cap) this.n += 1;
  }

  /** Widen the ring once the server tells us its real sample interval. */
  grow(cap) {
    if (cap <= this.cap) return;
    const old = { cap: this.cap, n: this.n, head: this.head, t: this.t, s: this.s };
    this.cap = cap;
    this.n = 0;
    this.head = 0;
    this.t = new Float64Array(cap);
    this.s = new Map();
    for (const name of old.s.keys()) this.key(name);
    for (let p = 0; p < old.n; p += 1) {
      const i = (old.head - old.n + p + old.cap) % old.cap;
      const vals = {};
      for (const [name, arr] of old.s) vals[name] = arr[i];
      this.push(old.t[i], vals);
    }
  }

  reset() {
    this.n = 0;
    this.head = 0;
    for (const a of this.s.values()) a.fill(NaN);
  }

  lastT() { return this.n ? this.t[this.idx(this.n - 1)] : 0; }

  /** min / avg / max over every finite reading at or after `tMin`. */
  stats(key, tMin) {
    const a = this.s.get(key);
    if (!a) return null;
    let mn = Infinity, mx = -Infinity, sum = 0, cnt = 0;
    for (let p = 0; p < this.n; p += 1) {
      const i = this.idx(p);
      if (this.t[i] < tMin) continue;
      const v = a[i];
      if (!Number.isFinite(v)) continue;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      sum += v;
      cnt += 1;
    }
    return cnt ? { min: mn, max: mx, avg: sum / cnt, n: cnt } : null;
  }

  last(key) {
    const a = this.s.get(key);
    if (!a) return NaN;
    for (let p = this.n - 1; p >= 0; p -= 1) {
      const v = a[this.idx(p)];
      if (Number.isFinite(v)) return v;
    }
    return NaN;
  }
}


/* =========================================================================
   CANVAS
   ========================================================================= */

/**
 * Match a canvas' backing store to its CSS box at the current devicePixelRatio
 * and return a context whose units are CSS pixels. Returns null when the canvas
 * is not laid out (collapsed panel, hidden detail row) — the caller then skips
 * the draw entirely instead of painting into a 0x0 buffer.
 */
function fit(cv) {
  const r = cv.getBoundingClientRect();
  const w = r.width, h = r.height;
  if (w < 2 || h < 2) return null;
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const pw = Math.max(1, Math.round(w * dpr));
  const ph = Math.max(1, Math.round(h * dpr));
  if (cv.width !== pw || cv.height !== ph) { cv.width = pw; cv.height = ph; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  cv._w = w;
  cv._h = h;
  return ctx;
}

/**
 * Project one series onto the plot box as a flat [x,y, x,y, …] list, where a
 * NaN,NaN pair marks a break in the data. Time is mapped absolutely, so an
 * irregular sample cadence still lands at the right horizontal position and the
 * newest reading always touches the right edge.
 */
function project(store, key, tMin, tEnd, w, h, yMax) {
  const a = store.s.get(key);
  const out = [];
  if (!a) return out;
  const span = (tEnd - tMin) || 1;
  const top = PAD_T, bot = h - PAD_B, ih = Math.max(1, bot - top);
  for (let p = 0; p < store.n; p += 1) {
    const i = store.idx(p);
    const t = store.t[i];
    if (t < tMin) continue;
    const x = ((t - tMin) / span) * w;
    const v = a[i];
    if (!Number.isFinite(v)) { out.push(NaN, NaN); continue; }
    out.push(x, bot - Math.max(0, Math.min(1, v / yMax)) * ih);
  }
  return out;
}

/** Walk the flat point list, calling `fn(startIndex, endIndex)` per unbroken run. */
function runs(pts, fn) {
  let start = -1;
  for (let i = 0; i < pts.length; i += 2) {
    const ok = Number.isFinite(pts[i]) && Number.isFinite(pts[i + 1]);
    if (ok && start < 0) start = i;
    if (!ok && start >= 0) { fn(start, i - 2); start = -1; }
  }
  if (start >= 0) fn(start, pts.length - 2);
}

/**
 * Draw one scrolling plot.
 *
 * `series` is [{key, color, fill, dash, label}]; `axisMax` fixes the y range so
 * a fill height always means the same thing (RAM against installed RAM, util
 * against 100%). `axisLabel` is printed top-left because an auto-scaled plot
 * with no number on it is decoration, not instrumentation.
 */
function drawPlot(cv, store, series, axisMax, axisLabel, tEnd, opts = {}) {
  const ctx = fit(cv);
  if (!ctx) return;
  const w = cv._w, h = cv._h;
  ctx.clearRect(0, 0, w, h);

  const tMin = tEnd - WINDOW_S;
  const line = tok("--line", "#2a2b30");
  const dim = tok("--dim", "#6f757f");

  /* minute rules — the only way to read "how long ago" off a scrolling plot */
  ctx.save();
  ctx.strokeStyle = rgba(line, 0.85);
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 4]);
  for (let k = 1; k * TICK_S < WINDOW_S; k += 1) {
    const x = Math.round(w - (k * TICK_S / WINDOW_S) * w) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  ctx.restore();

  if (!store.n) {
    ctx.fillStyle = rgba(dim, 0.8);
    ctx.font = `10px ${tok("--mono", "monospace")}`;
    ctx.textBaseline = "middle";
    ctx.fillText("waiting for samples…", 6, h / 2);
    return;
  }

  /* Two overlapping fills at full strength turn to mud, so the alpha backs off
     as soon as a cell carries more than one line (two GPUs, read + write). */
  const nFill = series.filter((s) => s.fill !== false).length;
  const fillA = nFill <= 1 ? 0.30 : nFill === 2 ? 0.17 : 0.11;

  for (const s of series) {
    const pts = project(store, s.key, tMin, tEnd, w, h, axisMax);
    if (!pts.length) continue;

    if (s.fill !== false) {
      const grad = ctx.createLinearGradient(0, PAD_T, 0, h);
      grad.addColorStop(0, rgba(s.color, fillA));
      grad.addColorStop(1, rgba(s.color, 0.01));
      ctx.fillStyle = grad;
      runs(pts, (a, b) => {
        if (b <= a) return;                       // a lone point has no area
        ctx.beginPath();
        ctx.moveTo(pts[a], h);
        for (let i = a; i <= b; i += 2) ctx.lineTo(pts[i], pts[i + 1]);
        ctx.lineTo(pts[b], h);
        ctx.closePath();
        ctx.fill();
      });
    }

    ctx.save();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.dash ? 1.1 : 1.5;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    if (s.dash) ctx.setLineDash(s.dash);
    else { ctx.shadowColor = rgba(s.color, 0.5); ctx.shadowBlur = 5; }
    runs(pts, (a, b) => {
      ctx.beginPath();
      if (b === a) { ctx.moveTo(pts[a] - 0.6, pts[a + 1]); ctx.lineTo(pts[a] + 0.6, pts[a + 1]); }
      else for (let i = a; i <= b; i += 2) (i === a ? ctx.moveTo : ctx.lineTo).call(ctx, pts[i], pts[i + 1]);
      ctx.stroke();
    });
    ctx.restore();

    /* the live head: a dot at the newest reading, ringed so it stays legible
       wherever it lands on the fill */
    for (let i = pts.length - 2; i >= 0; i -= 2) {
      if (!Number.isFinite(pts[i]) || !Number.isFinite(pts[i + 1])) continue;
      ctx.beginPath();
      ctx.arc(pts[i], pts[i + 1], 2.6, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = rgba(tok("--void", "#0b0b0d"), 0.85);
      ctx.stroke();
      break;
    }
  }

  /* guide at the current value — only when one line owns the cell, otherwise it
     silently belongs to whichever series happened to be first */
  if (series.length === 1 && opts.guide !== false) {
    const v = store.last(series[0].key);
    if (Number.isFinite(v)) {
      const bot = h - PAD_B, ih = Math.max(1, bot - PAD_T);
      const y = Math.round(bot - Math.max(0, Math.min(1, v / axisMax)) * ih) + 0.5;
      ctx.save();
      ctx.strokeStyle = rgba(series[0].color, 0.28);
      ctx.setLineDash([1, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      ctx.restore();
    }
  }

  ctx.font = `9.5px ${tok("--mono", "monospace")}`;
  ctx.textBaseline = "top";
  if (axisLabel) {
    ctx.fillStyle = rgba(dim, 0.95);
    ctx.fillText(axisLabel, 3, 2);
  }

  /* in-canvas legend: cheaper than DOM and it can never wrap away from its plot */
  const legend = series.filter((s) => s.label);
  if (legend.length > 1) {
    let x = w - 3;
    for (let i = legend.length - 1; i >= 0; i -= 1) {
      const s = legend[i];
      const tw = ctx.measureText(s.label).width;
      ctx.fillStyle = rgba(dim, 0.95);
      ctx.fillText(s.label, x - tw, 2);
      x -= tw + 5;
      ctx.fillStyle = s.color;
      ctx.fillRect(x - 7, 5, 7, 2.5);
      x -= 12;
    }
  }
}

/** The collapsed header's 88x20 summary: fill + line, nothing else. */
function drawSpark(cv, store, key, color, axisMax, tEnd) {
  const ctx = fit(cv);
  if (!ctx) return;
  const w = cv._w, h = cv._h;
  ctx.clearRect(0, 0, w, h);
  if (!store.n) return;
  const pts = project(store, key, tEnd - WINDOW_S, tEnd, w, h, axisMax);
  if (!pts.length) return;
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, rgba(color, 0.34));
  grad.addColorStop(1, rgba(color, 0.02));
  ctx.fillStyle = grad;
  runs(pts, (a, b) => {
    if (b <= a) return;
    ctx.beginPath();
    ctx.moveTo(pts[a], h);
    for (let i = a; i <= b; i += 2) ctx.lineTo(pts[i], pts[i + 1]);
    ctx.lineTo(pts[b], h);
    ctx.closePath();
    ctx.fill();
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  ctx.lineJoin = "round";
  runs(pts, (a, b) => {
    ctx.beginPath();
    for (let i = a; i <= b; i += 2) (i === a ? ctx.moveTo : ctx.lineTo).call(ctx, pts[i], pts[i + 1]);
    ctx.stroke();
  });
}


/* =========================================================================
   SCOPED STYLES
   app.css owns every shared class this panel uses (.perfpanel, .perfcell,
   .plotwrap, .gputile, .bar, .badge…). These few rules cover the pieces that
   exist only here — the core heatmap, the detail row and the header's machine
   line — and they are namespaced .pm-* under .perfpanel so nothing leaks.
   ========================================================================= */

const STYLE_ID = "lm3-perfmon-css";
const STYLE = `
.perfpanel .pm-machine{
  flex:0 1 auto;min-width:0;text-transform:none;letter-spacing:0;font-weight:400;
  font-size:11px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.perfpanel.collapsed .pm-machine{display:none}
.perfpanel .perf-hd .tools .btn{white-space:nowrap}

/* the always-visible core spectrum, tucked under the CPU plot */
.perfpanel .pm-corestrip{
  display:grid;gap:1px;margin-top:5px;flex:0 0 auto;grid-auto-rows:5px;
}
.perfpanel .pm-corestrip i{display:block;border-radius:1px;background:var(--panel2)}

/* the taller row: cores, GPUs, system readouts */
.perfpanel .pm-detail{
  flex:0 0 auto;display:grid;gap:1px;background:var(--line);border-top:1px solid var(--line);
  grid-template-columns:minmax(240px,1fr) minmax(300px,1.5fr) minmax(210px,.9fr);
  min-height:0;overflow:hidden;
}
@media (max-width:980px){.perfpanel .pm-detail{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
  .perfpanel .pm-dcell.sys{display:none}}
.perfpanel.collapsed .pm-detail{display:none}
.perfpanel .pm-dcell{background:#111214;padding:9px 12px 10px;min-width:0;overflow:auto}
.perfpanel .pm-dhd{
  font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--dim);
  font-weight:650;margin:0 0 7px;display:flex;align-items:baseline;gap:8px;
}
.perfpanel .pm-dhd .n{font:11px/1 var(--mono);color:var(--mute);margin-left:auto;
  font-variant-numeric:tabular-nums}

/* per-core heatmap — the ramp runs var(--panel2) -> var(--acc) so an idle core
   is indistinguishable from the panel and a pinned core glows */
.perfpanel .pm-coregrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(13px,1fr));gap:2px}
.perfpanel .pm-core{
  aspect-ratio:1;border-radius:2px;background:var(--panel2);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.03);
}
.perfpanel .pm-gmeta{
  margin-top:7px;font:11px/1.5 var(--mono);color:var(--dim);
  font-variant-numeric:tabular-nums;display:flex;gap:10px;flex-wrap:wrap;
}
.perfpanel .pm-dcell .gpugrid{grid-template-columns:repeat(auto-fit,minmax(228px,1fr));gap:10px;margin:0}
.perfpanel .pm-dcell .gputile{padding:10px 12px}
.perfpanel .pm-dcell .kv li{padding:3px 0;font-size:12px}
@media (max-width:900px){.perfpanel .mini-readouts .sparkline{display:none}}
@media (max-width:700px){.perfpanel .mini-readouts .r.opt{display:none}}
`;

function injectStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const s = document.createElement("style");
  s.id = STYLE_ID;
  s.textContent = STYLE;
  document.head.appendChild(s);
}


/* ------------------------------------------------------------ persistence -- */

function lsGet(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : v === "1";
  } catch { return fallback; }
}
function lsSet(key, on) {
  try { localStorage.setItem(key, on ? "1" : "0"); } catch { /* private mode */ }
}


/* =========================================================================
   THE PANEL
   ========================================================================= */

/**
 * Build the performance monitor into `root` and start following the metrics
 * stream.
 *
 * `root` is the element the app shell reserved for the panel. If it is not
 * already `.perfpanel` the class is added to it — the grid places the panel by
 * `.app > .perfpanel`, so wrapping it in a second element would drop it out of
 * the layout.
 *
 * @param {HTMLElement} [root]
 * @returns {{panel:HTMLElement, destroy:Function, setCollapsed:Function,
 *            setDetail:Function, refresh:Function}}
 */
/**
 * Re-run the LM3 hardware profiler and report the result.
 *
 * The sweep can take a couple of minutes (it benchmarks the CPU process-pool knee, the spawn cost
 * and disk write speed), so this polls the setup job rather than blocking, and disables the button
 * for the duration so a second sweep cannot be started on top of the first.
 *
 * With ``opts.calibrate`` it additionally MEASURES per-worker VRAM by running LM3 for real over a
 * few bundled images with one worker per GPU. That is much slower but replaces the estimate that
 * otherwise decides how many workers fit on a card -- and the estimate ran ~5x high for detectors,
 * which is what kept every GPU module down to a handful of workers on a 48 GB card.
 */
async function reprofile(btn, opts) {
  const calibrate = !!(opts && opts.calibrate);
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = calibrate ? "Measuring…" : "Profiling…";
  try {
    // force:true is REQUIRED. run_setup() early-returns when the machine fingerprint is unchanged,
    // and the fingerprint does not include free VRAM -- so the common reason to re-profile (a GPU
    // freed up, or you switched which GPU to use) would otherwise silently do nothing.
    const job = await api.runSetup({ optimize: true, force: true, calibrate });
    const jid = job && (job.job_id || job.id);
    // poll to completion; the endpoint returns the job's recorded sweep events.
    // Calibration runs LM3 for real over example images, so it needs a far longer leash.
    const maxPolls = calibrate ? 1800 : 240;
    for (let i = 0; jid && i < maxPolls; i += 1) {
      await new Promise((r) => setTimeout(r, 2000));
      let rec = null;
      try { rec = await api.getSetupEvents(jid); } catch (_) { break; }
      const st = rec && rec.state;
      if (st === "done" || st === "error") {
        if (st === "error") throw new Error((rec && rec.error) || "the hardware sweep failed");
        break;
      }
    }
    // The tuned per-module sizing has changed on disk. The live plots keep streaming regardless;
    // anything showing the PROFILE (worker counts, VRAM sizing) listens for this.
    document.dispatchEvent(new CustomEvent("lm3:hardware-reprofiled"));
    btn.textContent = calibrate ? "Measured ✓" : "Profiled ✓";
    setTimeout(() => { btn.textContent = label; }, 4000);
  } catch (err) {
    console.error("LM3: re-profile failed", err);
    btn.textContent = calibrate ? "Measure failed" : "Profile failed";
    setTimeout(() => { btn.textContent = label; }, 5000);
  } finally {
    btn.disabled = false;
  }
}

export function initPerfMon(root) {
  const host = root || document.querySelector(".perfpanel");
  if (!host) throw new Error("initPerfMon(): no root element");
  if (host._lm3perf) host._lm3perf.destroy();

  injectStyle();
  host.classList.add("perfpanel");
  clear(host);

  /* ---------------------------------------------------------------- state */
  const store = new Store(900);
  let machine = null;              /* describe_machine() */
  let latest = null;               /* newest snapshot() */
  let lastSeq = -1;
  let seeded = false;
  let queued = [];                 /* stream samples that beat the history seed */
  let nGpu = 0;
  let nCores = 0;
  let diskAxis = 10;               /* MB/s, hysteretic (see axisForDisk)      */
  let disposed = false;
  let dirty = true;

  let collapsed = lsGet(LS_COLLAPSED, false);
  let detail = lsGet(LS_DETAIL, false);

  let stream = null;
  let retryTimer = null;
  let pollTimer = null;
  let attempt = 0;

  /* ------------------------------------------------------------------ DOM */
  const machineLine = el("span.pm-machine");
  const conn = el("span.badge.dotd.warn", "Connecting");
  const mini = el("div.mini-readouts");

  const btnDetail = el("button.btn.sm.ghost", {
    type: "button",
    title: "Show the per-core heatmap, the GPU tiles and the system readouts",
    onclick: () => setDetail(!detail),
  }, "Details");

  /* Re-run LM3_Setup. The tuned profile decides workers-per-GPU, the CPU process-pool knee and the
     disk-write cap, and it is a SNAPSHOT: profiled while another job held a GPU, it sizes every GPU
     module for the VRAM that was free at that moment. Being able to re-profile from here -- after
     that job ends, or after changing GPUs -- is the difference between a stale profile and a
     correct one. */
  const btnProfile = el("button.btn.sm.ghost", {
    type: "button",
    title: "Re-measure this machine (LM3_Setup): GPU VRAM sizing, CPU worker knee, disk write speed",
    onclick: () => reprofile(btnProfile),
  }, "Re-profile");

  /* Measure, don't guess. Re-profile alone only re-runs the ESTIMATE; this runs real images
     through each GPU module with one worker and records the VRAM actually charged to it. */
  const btnCalibrate = el("button.btn.sm.ghost", {
    type: "button",
    title: "Measure real per-worker VRAM by running example images through each GPU module "
         + "(a few minutes). Replaces the built-in estimate, so more workers fit per GPU.",
    onclick: () => reprofile(btnCalibrate, { calibrate: true }),
  }, "Calibrate VRAM");

  const btnCollapse = el("button.btn.sm.ghost.icon", {
    type: "button",
    title: "Collapse the performance monitor",
    "aria-expanded": "true",
    onclick: () => setCollapsed(!collapsed),
  }, "▾");

  const head = el("div.perf-hd",
    el("span", "Machine"),
    machineLine,
    mini,
    el("div.tools", conn, btnCalibrate, btnProfile, btnDetail, btnCollapse));

  const grid = el("div.perfgrid");
  const detailRow = el("div.pm-detail");
  host.appendChild(head);
  host.appendChild(grid);
  host.appendChild(detailRow);

  /* ------------------------------------------------------------- plot cells */
  /* Each cell owns its canvas, its header refs and the closures that say what
     to plot and what the big number is — so `render` stays a loop, not a
     five-way switch. */
  const cells = [];

  function makeCell({ id, cls, label, build }) {
    const kEl = el("span.k", label);
    const vEl = el("span.v", "–");
    const sEl = el("span.sub", "");
    const cv = el("canvas.plot");
    const wrap = el("div.plotwrap.ruled", cv);
    const node = el(`div.perfcell.${cls}`, el("div.ch", kEl, vEl, sEl), wrap);
    const cell = { id, node, cv, vEl, sEl, kEl, wrap };
    build(cell);
    cells.push(cell);
    grid.appendChild(node);
    return cell;
  }

  /* CPU ------------------------------------------------------------------- */
  const coreStrip = el("div.pm-corestrip");
  const stripCells = [];
  makeCell({
    id: "cpu", cls: "cpu", label: "CPU",
    build: (c) => {
      c.node.appendChild(coreStrip);
      c.plot = () => ({
        series: [{ key: "cpu", color: tok("--acc3", "#4ade80") }],
        axisMax: 100,
        axisLabel: "100%",
      });
      c.value = () => pct(latest && latest.cpu_pct);
      c.statKey = "cpu";
      c.statFmt = (v) => pct(v, 0);
    },
  });

  /* RAM ------------------------------------------------------------------- */
  makeCell({
    id: "ram", cls: "ram", label: "RAM",
    build: (c) => {
      c.plot = () => {
        const total = (latest && latest.ram_total_mb) || (machine && machine.memory && machine.memory.ram_total_mb) || 1024;
        return {
          series: [{ key: "ram", color: tok("--acc2", "#38bdf8") }],
          axisMax: total,
          axisLabel: gb(total, { whole: true }),
        };
      };
      c.value = () => gb(latest && latest.ram_used_mb);
      c.statKey = "ram";
      c.statFmt = (v) => gb(v);
    },
  });

  /* GPU utilization ------------------------------------------------------- */
  const gpuCell = makeCell({
    id: "gpu", cls: "gpu", label: "GPU",
    build: (c) => {
      c.plot = () => ({
        series: gpuSeries("gu"),
        axisMax: 100,
        axisLabel: "100%",
      });
      c.value = () => pct(latest && latest.gpu_util_pct);
      c.statKey = "gpu";
      c.statFmt = (v) => pct(v, 0);
    },
  });

  /* VRAM ------------------------------------------------------------------ */
  const vramCell = makeCell({
    id: "vram", cls: "vram", label: "VRAM",
    build: (c) => {
      c.plot = () => {
        const per = perGpuVramTotal();
        return {
          series: gpuSeries("gm"),
          axisMax: per,
          axisLabel: nGpu > 1 ? `${gb(per, { whole: true })} / GPU` : gb(per, { whole: true }),
        };
      };
      c.value = () => gb(latest && latest.gpu_mem_used_mb);
      c.statKey = "vram";
      c.statFmt = (v) => gb(v);
    },
  });

  /* Disk I/O -------------------------------------------------------------- */
  makeCell({
    id: "disk", cls: "disk", label: "Disk",
    build: (c) => {
      c.plot = () => {
        const st = store.stats("disk", store.lastT() - WINDOW_S);
        return {
          series: [
            { key: "diskW", color: tok("--warn", "#fbbf24"), label: "write" },
            { key: "diskR", color: tok("--acc2", "#38bdf8"), label: "read", fill: false, dash: [3, 3] },
          ],
          axisMax: axisForDisk(st ? st.max : 0),
          axisLabel: `${fmtNum(diskAxis, { digits: diskAxis < 10 ? 1 : 0 })} MB/s`,
          guide: false,
        };
      };
      c.value = () => mbps(latest ? (latest.disk_read_mbps || 0) + (latest.disk_write_mbps || 0) : NaN);
      c.statKey = "disk";
      c.statFmt = (v) => mbps(v);
    },
  });

  /* ---------------------------------------------------------- detail cells */
  const coreGrid = el("div.pm-coregrid");
  const coreCount = el("span.n");
  const coreCells = [];
  const gpuGrid = el("div.gpugrid");
  const gpuTiles = [];
  const sysList = el("ul.kv");
  const sysRows = new Map();

  const dCores = el("div.pm-dcell.cores",
    el("div.pm-dhd", el("span", "Logical cores"), coreCount), coreGrid);
  const dGpus = el("div.pm-dcell.gpus",
    el("div.pm-dhd", el("span", "Graphics"), el("span.n")), gpuGrid);
  const dSys = el("div.pm-dcell.sys",
    el("div.pm-dhd", el("span", "System")), sysList);
  detailRow.appendChild(dCores);
  detailRow.appendChild(dGpus);
  detailRow.appendChild(dSys);

  for (const [key, label] of [
    ["load", "Load average"], ["swap", "Swap"], ["net", "Network"],
    ["procs", "Processes"], ["busy", "Disk busy"], ["added", "Added by LM3"],
    ["sampler", "Sampler"],
  ]) {
    const v = el("span.v.mono", "–");
    sysList.appendChild(el("li", el("span.k", label), v));
    sysRows.set(key, v);
  }

  /* -------------------------------------------------- collapsed readouts */
  const miniRows = [];
  function addMini(id, label, color, optional) {
    const cv = el("canvas");
    const v = el("span.v", "–");
    const row = el(`div.r${optional ? ".opt" : ""}`,
      el("span.k", label), el("span.sparkline", cv), v);
    mini.appendChild(row);
    miniRows.push({ id, cv, v, color, row });
  }
  addMini("cpu", "CPU", () => tok("--acc3", "#4ade80"), false);
  addMini("ram", "RAM", () => tok("--acc2", "#38bdf8"), false);
  addMini("gpu", "GPU", () => tok("--acc", "#fb923c"), false);
  addMini("vram", "VRAM", () => tok("--vio", "#c084fc"), true);
  addMini("disk", "Disk", () => tok("--warn", "#fbbf24"), true);


  /* ===================================================== derived helpers == */

  /** One line per GPU, colored by card index (see GPU_COLORS). */
  function gpuSeries(prefix) {
    const out = [];
    for (let i = 0; i < Math.max(1, nGpu); i += 1) {
      out.push({
        key: prefix + i,
        color: tok(GPU_COLORS[i % GPU_COLORS.length], "#fb923c"),
        label: nGpu > 1 ? String(i) : null,
      });
    }
    return out;
  }

  /** Largest single-card VRAM, so both cards share one honest y range. */
  function perGpuVramTotal() {
    let m = 0;
    const list = (latest && latest.gpus) || (machine && machine.gpus) || [];
    for (const g of list) m = Math.max(m, g.mem_total_mb || g.vram_total_mb || 0);
    return m || 1024;
  }

  /**
   * Disk is the one plot with no natural ceiling. Snap up immediately so a
   * burst is never clipped, but only step down once the window max has fallen
   * below half the current axis — otherwise the scale flaps on every sample and
   * the line looks like it is breathing.
   */
  function axisForDisk(windowMax) {
    const target = Math.max(10, niceCeil((windowMax || 0) * 1.15));
    if (target > diskAxis || target <= diskAxis / 2) diskAxis = target;
    return diskAxis;
  }


  /* ========================================================== ingestion == */

  function applyMachine(m) {
    if (!m || typeof m !== "object") return;
    machine = m;
    const gpus = m.gpus || [];
    if (gpus.length !== nGpu) { nGpu = gpus.length; layoutGrid(); buildGpuTiles(); }
    if (m.cpu && m.cpu.cores_logical) setCores(m.cpu.cores_logical);

    /* Widen the ring if this server samples faster than the 0.5 s default. */
    const iv = m.sampler && m.sampler.interval_s;
    if (isNum(iv) && iv > 0) store.grow(Math.min(6000, Math.ceil(WINDOW_S / iv) + 120));

    const c = m.cpu || {};
    const bits = [];
    if (m.hostname) bits.push(m.hostname);
    if (c.cores_logical) bits.push(`${c.cores_physical || c.cores_logical} cores / ${c.cores_logical} threads`);
    if (m.memory && m.memory.ram_total_mb) bits.push(`${gb(m.memory.ram_total_mb, { whole: true })} RAM`);
    if (gpus.length) {
      const name = String(gpus[0].name || "GPU").replace(/^NVIDIA\s+/i, "");
      bits.push(gpus.length > 1 ? `${gpus.length} × ${name}` : name);
    }
    if (m.gpu_driver) bits.push(`driver ${m.gpu_driver}`);
    machineLine.textContent = bits.join("  ·  ");
    machineLine.title = `${m.platform || ""}\nPython ${m.python || "?"}`;
    dirty = true;
  }

  function setCores(n) {
    if (n === nCores || !(n > 0)) return;
    nCores = n;

    /* compact strip: two rows once there are enough cores to make one row a
       hairline; the squares stretch so the strip always spans the cell */
    clear(coreStrip);
    stripCells.length = 0;
    const rows = n >= 24 ? 2 : 1;
    const cols = Math.ceil(n / rows);
    coreStrip.style.gridTemplateColumns = `repeat(${cols},minmax(0,1fr))`;
    for (let i = 0; i < n; i += 1) {
      const b = el("i");
      coreStrip.appendChild(b);
      stripCells.push(b);
    }

    clear(coreGrid);
    coreCells.length = 0;
    for (let i = 0; i < n; i += 1) {
      const b = el("div.pm-core", { title: `core ${i}` });
      coreGrid.appendChild(b);
      coreCells.push(b);
    }
    coreCount.textContent = String(n);
  }

  function ingest(snap) {
    if (!snap || snap.ready === false) return false;
    const seq = Number(snap.seq) || 0;
    if (seq && seq <= lastSeq) return false;
    if (seq) lastSeq = seq;

    const gpus = snap.gpus || [];
    if (gpus.length !== nGpu) { nGpu = gpus.length; layoutGrid(); buildGpuTiles(); }
    if (Array.isArray(snap.cpu_per_core_pct) && snap.cpu_per_core_pct.length) {
      setCores(snap.cpu_per_core_pct.length);
    }

    const r = snap.disk_read_mbps, w = snap.disk_write_mbps;
    const vals = {
      cpu: snap.cpu_pct,
      ram: snap.ram_used_mb,
      gpu: snap.gpu_util_pct,
      vram: snap.gpu_mem_used_mb,
      diskR: r, diskW: w,
      disk: (isNum(r) || isNum(w)) ? (r || 0) + (w || 0) : NaN,
    };
    for (const g of gpus) {
      vals[`gu${g.index}`] = g.util_pct;
      vals[`gm${g.index}`] = g.mem_used_mb;
    }
    store.push(Number(snap.t) || Date.now() / 1000, vals);
    latest = snap;
    return true;
  }

  /**
   * Seed from GET /v1/metrics/history (or the stream's hello frame). Idempotent
   * on purpose: whichever of the two arrives second only appends rows newer
   * than `lastSeq`, so a slow history response can never wipe live samples.
   */
  function seed(h) {
    if (!h || !Array.isArray(h.t)) return;
    const S = h.series || {};
    const G = h.gpus || [];
    const seqs = h.seq || [];
    if (!seeded && G.length !== nGpu) { nGpu = G.length; layoutGrid(); buildGpuTiles(); }
    if (isNum(h.interval_s) && h.interval_s > 0) {
      store.grow(Math.min(6000, Math.ceil(WINDOW_S / h.interval_s) + 120));
    }

    const pick = (arr, i) => (Array.isArray(arr) ? arr[i] : undefined);
    for (let i = 0; i < h.t.length; i += 1) {
      const sq = Number(seqs[i]) || 0;
      if (sq && sq <= lastSeq) continue;
      const r = pick(S.disk_read_mbps, i), w = pick(S.disk_write_mbps, i);
      const vals = {
        cpu: pick(S.cpu_pct, i),
        ram: pick(S.ram_used_mb, i),
        gpu: pick(S.gpu_util_pct, i),
        vram: pick(S.gpu_mem_used_mb, i),
        diskR: r, diskW: w,
        disk: (isNum(r) || isNum(w)) ? (r || 0) + (w || 0) : NaN,
      };
      for (const g of G) {
        vals[`gu${g.index}`] = pick(g.util_pct, i);
        vals[`gm${g.index}`] = pick(g.mem_used_mb, i);
      }
      store.push(Number(h.t[i]) || 0, vals);
      if (sq) lastSeq = sq;
    }
    dirty = true;
  }

  function finishSeed() {
    if (seeded) return;
    seeded = true;
    for (const s of queued) ingest(s);
    queued = [];
    schedule();
  }

  /** GET /v1/metrics answers bare snapshot() (metrics.py) or {snapshot, machine,
      …} (metrics_api.py). Both are mounted in this codebase, so accept either. */
  function unwrap(payload) {
    if (!payload || typeof payload !== "object") return null;
    if (payload.snapshot && typeof payload.snapshot === "object") {
      if (payload.machine && !machine) applyMachine(payload.machine);
      return payload.snapshot;
    }
    return payload;
  }


  /* ============================================================ rendering = */

  function layoutGrid() {
    /* GPU cells are meaningless without a GPU; drop them and let the remaining
       plots use the full width rather than showing three empty boxes. */
    const show = nGpu > 0;
    gpuCell.node.hidden = !show;
    vramCell.node.hidden = !show;
    dGpus.hidden = !show;
    const n = cells.filter((c) => !c.node.hidden).length;
    grid.style.gridTemplateColumns = `repeat(${n},minmax(0,1fr))`;
    dirty = true;
  }

  function buildGpuTiles() {
    clear(gpuGrid);
    gpuTiles.length = 0;
    const list = (machine && machine.gpus) || (latest && latest.gpus) || [];
    const n = Math.max(nGpu, list.length);
    for (let i = 0; i < n; i += 1) {
      const info = list[i] || {};
      const badge = el("span.ubadge.idle", "idle");
      const name = el("div.gname",
        el("span", { style: { color: tok(GPU_COLORS[i % GPU_COLORS.length], "#fb923c") } }, `GPU ${i}`),
        ` ${String(info.name || "").replace(/^NVIDIA\s+/i, "") || "graphics device"}`,
        badge);
      const used = el("b", "–");
      const vram = el("div.gvram", used, " / ", el("span", "– GB"));
      const fill = el("span");
      const bar = el("div.gbar", fill);
      const meta = el("div.pm-gmeta");
      const tile = el("div.gputile", name, vram, bar, meta);
      gpuGrid.appendChild(tile);
      gpuTiles.push({ tile, badge, used, total: vram.lastChild, fill, meta });
    }
    dGpus.querySelector(".pm-dhd .n").textContent = n ? `${n} device${n > 1 ? "s" : ""}` : "";
    dirty = true;
  }

  function renderCells(tEnd, tMin) {
    for (const c of cells) {
      if (c.node.hidden) continue;
      const spec = c.plot();
      c.vEl.textContent = c.value();
      const st = store.stats(c.statKey, tMin);
      c.sEl.textContent = st
        ? `min ${c.statFmt(st.min)} · avg ${c.statFmt(st.avg)} · max ${c.statFmt(st.max)}`
        : "";
      drawPlot(c.cv, store, spec.series, spec.axisMax, spec.axisLabel, tEnd,
               { guide: spec.guide });
    }
  }

  function renderCores() {
    const per = (latest && latest.cpu_per_core_pct) || null;
    const base = tok("--panel2", "#1f2024");
    const hot = tok("--acc", "#fb923c");
    for (let i = 0; i < stripCells.length; i += 1) {
      const v = per && isNum(per[i]) ? per[i] : 0;
      /* ^0.72 lifts the midrange: a core at 20% should read as "doing something",
         and a straight linear ramp buries it in the panel color. */
      const col = mix(base, hot, Math.pow(Math.max(0, Math.min(100, v)) / 100, 0.72));
      stripCells[i].style.background = col;
      if (detail && coreCells[i]) {
        coreCells[i].style.background = col;
        coreCells[i].title = `core ${i} — ${v.toFixed(0)}%`;
      }
    }
  }

  function renderGpus() {
    const list = (latest && latest.gpus) || [];
    for (let i = 0; i < gpuTiles.length; i += 1) {
      const t = gpuTiles[i];
      const g = list[i];
      if (!g) { t.tile.classList.remove("used", "hot"); continue; }
      const util = isNum(g.util_pct) ? g.util_pct : null;
      const total = g.mem_total_mb || 0;
      t.badge.textContent = util === null ? "n/a" : `${util.toFixed(0)}% util`;
      t.badge.classList.toggle("idle", !(util > 3));
      t.used.textContent = isNum(g.mem_used_mb) ? gb(g.mem_used_mb, { whole: true }).replace(" GB", "") : "–";
      t.total.textContent = gb(total, { whole: true });
      t.fill.style.width = `${Math.max(0, Math.min(100, g.mem_pct || 0))}%`;
      t.tile.classList.toggle("used", (util || 0) > 3 || (g.mem_pct || 0) > 5);
      /* "hot" is the card you would actually worry about: near the thermal
         ceiling or pinned at its power limit */
      const hot = (isNum(g.temp_c) && g.temp_c >= 80)
        || (isNum(g.power_w) && isNum(g.power_limit_w) && g.power_limit_w > 0
            && g.power_w >= g.power_limit_w * 0.92);
      t.tile.classList.toggle("hot", hot);

      const meta = [];
      if (isNum(g.temp_c)) meta.push(`${g.temp_c.toFixed(0)} °C`);
      if (isNum(g.power_w)) {
        meta.push(isNum(g.power_limit_w) && g.power_limit_w > 0
          ? `${g.power_w.toFixed(0)} / ${g.power_limit_w.toFixed(0)} W`
          : `${g.power_w.toFixed(0)} W`);
      }
      if (isNum(g.sm_clock_mhz)) meta.push(`${g.sm_clock_mhz.toFixed(0)} MHz`);
      if (isNum(g.fan_pct)) meta.push(`fan ${g.fan_pct.toFixed(0)}%`);
      t.meta.textContent = meta.join("  ·  ");
    }
  }

  function renderSys() {
    const s = latest || {};
    const set = (k, v) => { const n = sysRows.get(k); if (n) n.textContent = v; };
    set("load", Array.isArray(s.load_avg)
      ? s.load_avg.map((v) => fmtNum(v, { digits: 2 })).join("  ")
      : "–");
    set("swap", isNum(s.swap_used_mb)
      ? `${gb(s.swap_used_mb)} / ${gb(s.swap_total_mb, { whole: true })}  (${pct(s.swap_pct, 0)})`
      : "–");
    set("net", `↓ ${mbps(s.net_recv_mbps)}   ↑ ${mbps(s.net_sent_mbps)}`);
    set("procs", isNum(s.n_procs) ? fmtNum(s.n_procs) : "–");
    set("busy", isNum(s.disk_busy_pct) ? pct(s.disk_busy_pct, 0) : "not reported");
    /* ram_delta/vram_delta are measured against the baseline metrics.py takes
       when a run starts — i.e. what THIS run added, not what the box is using. */
    set("added", `${gb(s.ram_delta_mb)} RAM   ·   ${gb(s.vram_delta_mb)} VRAM`);
    const sm = (machine && machine.sampler) || {};
    set("sampler", isNum(sm.interval_s)
      ? `${(1 / sm.interval_s).toFixed(sm.interval_s < 1 ? 0 : 1)} Hz · ${fmtDuration(sm.window_s)} window · ${store.n} samples`
      : `${store.n} samples`);
  }

  function renderMini(tEnd) {
    for (const m of miniRows) {
      const color = m.color();
      let axis = 100, key = m.id, text = "–";
      if (m.id === "cpu") { text = pct(latest && latest.cpu_pct, 0); }
      else if (m.id === "ram") {
        axis = (latest && latest.ram_total_mb) || 1024;
        text = pct(latest && latest.ram_pct, 0);
      } else if (m.id === "gpu") { text = pct(latest && latest.gpu_util_pct, 0); }
      else if (m.id === "vram") {
        axis = (latest && latest.gpu_mem_total_mb) || 1024;
        text = gb(latest && latest.gpu_mem_used_mb);
      } else if (m.id === "disk") {
        axis = diskAxis;
        text = mbps(latest ? (latest.disk_read_mbps || 0) + (latest.disk_write_mbps || 0) : NaN);
      }
      m.v.textContent = text;
      drawSpark(m.cv, store, key, color, axis, tEnd);
    }
  }

  function render() {
    if (disposed) return;
    const tEnd = store.lastT() || Date.now() / 1000;
    const tMin = tEnd - WINDOW_S;

    if (collapsed) {
      /* GPU/VRAM readouts are noise on a CPU-only box */
      for (const m of miniRows) {
        if (m.id === "gpu" || m.id === "vram") m.row.hidden = nGpu === 0;
      }
      renderMini(tEnd);
      return;
    }

    renderCells(tEnd, tMin);
    renderCores();
    if (detail) { renderGpus(); renderSys(); }
  }

  const draw = rafThrottle(render);

  function schedule() {
    if (disposed) return;
    dirty = true;
    if (document.hidden) return;   /* nothing to paint into a hidden window */
    dirty = false;
    draw();
  }


  /* ============================================================== controls = */

  function applyHeight() {
    /* The expanded height rides on --perf-h, the same variable app.css uses for
       the panel AND for keeping the toast stack above it, so both follow. */
    if (detail) {
      const h = Math.max(300, Math.min(400, Math.round(window.innerHeight * 0.42)));
      document.documentElement.style.setProperty("--perf-h", `${h}px`);
    } else {
      document.documentElement.style.removeProperty("--perf-h");
    }
  }

  function setCollapsed(on) {
    collapsed = !!on;
    lsSet(LS_COLLAPSED, collapsed);
    host.classList.toggle("collapsed", collapsed);
    btnCollapse.textContent = collapsed ? "▴" : "▾";
    btnCollapse.title = collapsed ? "Expand the performance monitor" : "Collapse the performance monitor";
    btnCollapse.setAttribute("aria-expanded", collapsed ? "false" : "true");
    btnDetail.disabled = collapsed;
    /* The CSS transition means the canvases are mid-resize for ~220 ms; redraw
       when it lands so nothing is left stretched. */
    schedule();
    setTimeout(schedule, 240);
  }

  function setDetail(on) {
    detail = !!on;
    lsSet(LS_DETAIL, detail);
    detailRow.hidden = !detail;
    btnDetail.classList.toggle("on", detail);
    applyHeight();
    schedule();
    setTimeout(schedule, 240);
  }

  function setConn(state, note) {
    conn.className = `badge dotd ${state === "up" ? "ok" : state === "poll" ? "info" : state === "wait" ? "warn" : "bad"}`;
    conn.textContent = state === "up" ? "Live"
      : state === "poll" ? "Polling"
      : state === "wait" ? "Connecting" : "Offline";
    conn.title = note || (state === "up" ? "Following /v1/metrics/stream"
      : state === "poll" ? "Stream unavailable — sampling /v1/metrics every 2 s"
      : "The LM3 server is not answering");
  }


  /* =============================================================== network = */

  function handleFrame(f) {
    if (!f || typeof f !== "object") return;
    switch (f.type) {
      case "hello":
        if (f.machine) applyMachine(f.machine);
        if (f.history) seed(f.history);
        finishSeed();
        schedule();
        break;
      case "metrics":
        if (!seeded) { queued.push(f.snapshot); if (queued.length > 400) queued.shift(); }
        else if (ingest(f.snapshot)) schedule();
        break;
      case "workers":
      case "run":
        break;                     /* the status tab owns those */
      default:
        break;
    }
  }

  function closeStream() {
    if (stream) { try { stream(); } catch { /* already closed */ } stream = null; }
  }

  function startPoll() {
    if (pollTimer || disposed) return;
    pollTimer = setInterval(async () => {
      try {
        const snap = unwrap(await api.getMetrics());
        if (snap) { if (!seeded) finishSeed(); if (ingest(snap)) schedule(); }
        setConn("poll");
      } catch { setConn("down"); }
    }, 2000);
  }

  function stopPoll() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  /**
   * Follow the metrics stream, owning the reconnect ourselves.
   *
   * EventSource retries on its own every ~3 s forever, which hammers a server
   * that is down and gives no way to tell the user what is happening. So the
   * first error closes the socket and we come back on an exponential backoff
   * with jitter; after two failures we also start polling /v1/metrics, which
   * keeps the panel alive behind a proxy that eats event streams.
   */
  function connect() {
    if (disposed) return;
    clearTimeout(retryTimer);
    closeStream();
    setConn(attempt ? "wait" : "wait");
    stream = api.streamMetrics({
      onOpen: () => { attempt = 0; stopPoll(); setConn("up"); },
      onMessage: handleFrame,
      onError: () => {
        closeStream();
        attempt += 1;
        if (attempt >= 2) { startPoll(); setConn("down"); } else setConn("wait");
        const base = Math.min(30000, 800 * Math.pow(1.7, attempt - 1));
        retryTimer = setTimeout(connect, base * (0.85 + Math.random() * 0.3));
      },
    });
  }

  async function prime() {
    /* The hello frame carries both of these, but asking directly paints the
       plots a beat sooner and covers a server that only mounts metrics.router
       without the stream reaching us first. Both paths dedupe on `seq`. */
    const [m, h] = await Promise.allSettled([
      api.get("/v1/metrics/machine", { timeout: 8000 }),
      api.get("/v1/metrics/history", { timeout: 12000 }),
    ]);
    if (disposed) return;
    if (m.status === "fulfilled") applyMachine(m.value);
    if (h.status === "fulfilled") seed(h.value);
    finishSeed();
    schedule();
  }


  /* ================================================================ events = */

  const onVisibility = () => { if (!document.hidden && dirty) schedule(); };
  document.addEventListener("visibilitychange", onVisibility);

  const onResize = () => { applyHeight(); schedule(); };
  window.addEventListener("resize", onResize);

  /* A ResizeObserver is the only thing that catches the tab body being dragged
     or the Electron window being snapped — both change the canvas box without
     firing a window resize in every case. */
  let ro = null;
  if (typeof ResizeObserver === "function") {
    ro = new ResizeObserver(() => schedule());
    ro.observe(grid);
    ro.observe(head);
  }


  /* ================================================================ start == */

  setCollapsed(collapsed);
  setDetail(detail);
  layoutGrid();
  buildGpuTiles();
  setConn("wait");
  prime();
  connect();
  schedule();

  const controller = {
    panel: host,
    setCollapsed,
    setDetail,
    refresh: schedule,
    /** Newest sample, for anything that wants a number without its own poll. */
    get snapshot() { return latest; },
    get machine() { return machine; },
    destroy() {
      if (disposed) return;
      disposed = true;
      closeStream();
      stopPoll();
      clearTimeout(retryTimer);
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("resize", onResize);
      if (ro) ro.disconnect();
      document.documentElement.style.removeProperty("--perf-h");
      if (host._lm3perf === controller) delete host._lm3perf;
      clear(host);
    },
  };
  host._lm3perf = controller;
  return controller;
}

export default initPerfMon;
