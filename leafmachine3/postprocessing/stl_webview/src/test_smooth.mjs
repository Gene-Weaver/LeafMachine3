/* Prototype curved-boundary smoothing methods and measure them against the
   pixel staircase: vertex count, area change, self-intersection, timing. */
import { readFileSync, writeFileSync } from "node:fs";
import { selectMask, fillHoles, maskToPolygons } from "./geom.js";

const which = process.argv[2] || "sample";
const b = readFileSync(`/tmp/claude-1001/bnd_${which}.bin`);
const w = b.readUInt32LE(0), h = b.readUInt32LE(4);
const img = { width: w, height: h, data: new Uint8Array(b.buffer, b.byteOffset + 8, w * h * 4) };
let m = selectMask(img, ["white"], 0);
m = fillHoles(m, w, h);

/* ----------------------------------------------------------------- helpers */
const openRing = (r) => {
  const n = r.length;
  return (n > 1 && r[0][0] === r[n - 1][0] && r[0][1] === r[n - 1][1]) ? r.slice(0, -1) : r.slice();
};
const closeRing = (r) => r.concat([r[0].slice()]);
const area = (r) => {
  let a = 0;
  for (let i = 0, j = r.length - 1; i < r.length; j = i++) a += r[j][0] * r[i][1] - r[i][0] * r[j][1];
  return a / 2;
};
const perim = (r) => {
  let p = 0;
  for (let i = 0; i < r.length; i++) {
    const q = r[(i + 1) % r.length];
    p += Math.hypot(q[0] - r[i][0], q[1] - r[i][1]);
  }
  return p;
};

/** Uniform arc-length resample of a CLOSED ring. */
function resample(ring, spacing) {
  const p = openRing(ring);
  const n = p.length;
  if (n < 3) return p;
  const L = perim(p);
  const count = Math.max(8, Math.round(L / spacing));
  const step = L / count;
  const out = [];
  let seg = 0, acc = 0;
  let cur = 0;
  for (let k = 0; k < count; k++) {
    const target = k * step;
    while (true) {
      const a = p[seg % n], bb = p[(seg + 1) % n];
      const d = Math.hypot(bb[0] - a[0], bb[1] - a[1]);
      if (acc + d >= target || seg >= n * 2) {
        const t = d > 0 ? (target - acc) / d : 0;
        out.push([a[0] + (bb[0] - a[0]) * t, a[1] + (bb[1] - a[1]) * t]);
        break;
      }
      acc += d; seg++;
    }
  }
  return out;
}

/** Periodic Gaussian convolution of the coordinate sequence. sigma in POINTS. */
function gaussClosed(pts, sigmaPts) {
  const n = pts.length;
  if (n < 5 || sigmaPts <= 0) return pts;
  const rad = Math.max(1, Math.ceil(sigmaPts * 3));
  const k = [];
  let sum = 0;
  for (let i = -rad; i <= rad; i++) { const v = Math.exp(-(i * i) / (2 * sigmaPts * sigmaPts)); k.push(v); sum += v; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let x = 0, y = 0;
    for (let j = -rad; j <= rad; j++) {
      const p = pts[((i + j) % n + n) % n];
      const wj = k[j + rad];
      x += p[0] * wj; y += p[1] * wj;
    }
    out[i] = [x, y];
  }
  return out;
}

/** Chaikin corner cutting on a closed ring. */
function chaikin(ring, iters) {
  let p = openRing(ring);
  for (let it = 0; it < iters; it++) {
    const out = [];
    for (let i = 0; i < p.length; i++) {
      const a = p[i], bb = p[(i + 1) % p.length];
      out.push([a[0] * 0.75 + bb[0] * 0.25, a[1] * 0.75 + bb[1] * 0.25]);
      out.push([a[0] * 0.25 + bb[0] * 0.75, a[1] * 0.25 + bb[1] * 0.75]);
    }
    p = out;
  }
  return p;
}

/** Taubin lambda/mu smoothing -- shrink-free low pass. */
function taubin(pts, iters, lam = 0.5, mu = -0.53) {
  let p = pts.map((q) => q.slice());
  const n = p.length;
  if (n < 5) return p;
  const pass = (f) => {
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      const a = p[(i - 1 + n) % n], c = p[(i + 1) % n];
      const dx = (a[0] + c[0]) / 2 - p[i][0];
      const dy = (a[1] + c[1]) / 2 - p[i][1];
      out[i] = [p[i][0] + f * dx, p[i][1] + f * dy];
    }
    p = out;
  };
  for (let i = 0; i < iters; i++) { pass(lam); pass(mu); }
  return p;
}

/** Douglas-Peucker on a closed ring (same as geom.js, standalone for the test). */
function dp(ring, tol) {
  const pts = openRing(ring);
  if (!tol || tol <= 0 || pts.length < 4) return pts;
  const keep = new Uint8Array(pts.length);
  keep[0] = 1; keep[pts.length - 1] = 1;
  const stack = [[0, pts.length - 1]];
  while (stack.length) {
    const [a, bb] = stack.pop();
    let worst = -1, wi = -1;
    const [ax, ay] = pts[a], [bx, by] = pts[bb];
    const dx = bx - ax, dy = by - ay;
    const len = Math.hypot(dx, dy) || 1e-12;
    for (let i = a + 1; i < bb; i++) {
      const d = Math.abs((pts[i][0] - ax) * dy - (pts[i][1] - ay) * dx) / len;
      if (d > worst) { worst = d; wi = i; }
    }
    if (worst > tol && wi > 0) { keep[wi] = 1; stack.push([a, wi], [wi, bb]); }
  }
  const out = [];
  for (let i = 0; i < pts.length; i++) if (keep[i]) out.push(pts[i]);
  return out.length >= 3 ? out : pts;
}

/** Does a closed ring self-intersect? Proper test, ignores shared endpoints. */
function selfIntersects(r) {
  const n = r.length;
  const orient = (p, q, s) => {
    const v = (q[0] - p[0]) * (s[1] - p[1]) - (q[1] - p[1]) * (s[0] - p[0]);
    return v > 1e-12 ? 1 : v < -1e-12 ? -1 : 0;
  };
  const cross = (a, b2, c, d) =>
    orient(a, b2, c) !== orient(a, b2, d) && orient(c, d, a) !== orient(c, d, b2);
  let hits = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 2; j < n; j++) {
      if (i === 0 && j === n - 1) continue;
      if (cross(r[i], r[(i + 1) % n], r[j], r[(j + 1) % n])) hits++;
    }
  }
  return hits;
}

/* -------------------------------------------------------------------- run */
const raw = maskToPolygons(m, w, h, { simplifyPx: 0, minAreaPx: 4 });
const outer = raw[0].outer;
const A0 = Math.abs(area(openRing(outer)));
console.log(`raw outline: ${openRing(outer).length} verts, area ${A0.toFixed(0)} px^2, perim ${perim(openRing(outer)).toFixed(0)} px\n`);

const METHODS = {
  "dp1.5 (today)": () => dp(outer, 1.5),
  "chaikin x3 + dp0.3": () => dp(closeRing(chaikin(outer, 3)), 0.3),
  "chaikin x5 + dp0.3": () => dp(closeRing(chaikin(outer, 5)), 0.3),
  "resample1 + gauss s=2 + dp0.3": () => dp(closeRing(gaussClosed(resample(outer, 1), 2)), 0.3),
  "resample1 + gauss s=4 + dp0.3": () => dp(closeRing(gaussClosed(resample(outer, 1), 4)), 0.3),
  "resample1 + gauss s=8 + dp0.3": () => dp(closeRing(gaussClosed(resample(outer, 1), 8)), 0.3),
  "resample1 + taubin x20 + dp0.3": () => dp(closeRing(taubin(resample(outer, 1), 20)), 0.3),
  "resample1 + taubin x60 + dp0.3": () => dp(closeRing(taubin(resample(outer, 1), 60)), 0.3),
  "resample2 + gauss s=4 + dp0.5": () => dp(closeRing(gaussClosed(resample(outer, 2), 4)), 0.5),
};

const dump = {};
for (const [name, fn] of Object.entries(METHODS)) {
  const t0 = performance.now();
  const r = fn();
  const ms = performance.now() - t0;
  const A = Math.abs(area(r));
  const si = r.length < 4000 ? selfIntersects(r) : "skipped";
  console.log(`${name.padEnd(32)} ${String(r.length).padStart(6)} verts  `
    + `area ${((A / A0 - 1) * 100).toFixed(2).padStart(6)}%  `
    + `self-int ${String(si).padStart(4)}  ${ms.toFixed(0)} ms`);
  dump[name] = r;
}
writeFileSync(`/tmp/claude-1001/smooth_${which}.json`, JSON.stringify(dump));
