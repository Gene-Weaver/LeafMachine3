/* ==========================================================================
   veins.js -- turn a Lamina_RGB leaf photo into a binary vein mask.

   Leaf veins in a herbarium scan are DARK, THIN and LOW CONTRAST: on the test
   sheet the in-leaf standard deviation is about 15 grey levels out of 255, and
   the veins sit only a few levels below the lamina around them. A global
   threshold cannot separate that, because the lamina's own brightness varies
   far more across the leaf than a vein differs from its neighbourhood.

   So the pipeline is local, in this order:

     channel      pick the plane with the best vein contrast (green, usually)
     background   subtract a LOCAL mean -- the top-hat idea. This is what turns
                  "a bit darker than its surroundings" into an absolute signal.
     ridge        optional directional opening: keep only what survives being
                  probed with a LINE. Leaf texture is blobby and dies; veins are
                  elongated and live. This is the single biggest noise win.
     threshold    manual, Otsu, or adaptive (local mean minus an offset)
     cleanup      close small gaps, drop specks, optionally thicken for printing
     domain       finally AND with the leaf silhouette, eroded off the rim

   Every window operation runs off integral images, so radius is free: a 41 px
   background window costs the same as a 3 px one. That matters because this
   re-runs on every slider drag.
   ========================================================================== */

/** Grayscale plane. `mode` is one of luma | r | g | b | max | min | sat. */
export function toGray(imageData, mode = "g") {
  const { data, width: w, height: h } = imageData;
  const out = new Uint8ClampedArray(w * h);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    const r = data[p], g = data[p + 1], b = data[p + 2];
    let v;
    switch (mode) {
      case "luma": v = 0.299 * r + 0.587 * g + 0.114 * b; break;
      case "r": v = r; break;
      case "g": v = g; break;
      case "b": v = b; break;
      case "max": v = Math.max(r, g, b); break;
      case "min": v = Math.min(r, g, b); break;
      case "sat": { const mx = Math.max(r, g, b), mn = Math.min(r, g, b); v = mx ? ((mx - mn) * 255) / mx : 0; break; }
      default: v = g;
    }
    out[i] = v;
  }
  return out;
}

/**
 * Local mean over a (2r+1) box, computed ONLY over pixels where `dom` is set.
 *
 * The domain matters more than it looks. With a plain blur, a window straddling
 * the leaf edge averages in the black background, the local mean collapses, and
 * the whole rim lights up as a false "vein" brighter than any real one. Dividing
 * by the count of in-leaf pixels instead of the window area removes that
 * entirely -- the edge is then just an edge.
 */
export function maskedMean(src, dom, w, h, r) {
  const W = w + 1;
  const sum = new Float64Array(W * (h + 1));
  const cnt = new Float64Array(W * (h + 1));
  for (let y = 0; y < h; y++) {
    let rs = 0, rc = 0;
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const on = dom ? dom[i] : 1;
      rs += on ? src[i] : 0;
      rc += on ? 1 : 0;
      sum[(y + 1) * W + x + 1] = sum[y * W + x + 1] + rs;
      cnt[(y + 1) * W + x + 1] = cnt[y * W + x + 1] + rc;
    }
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
      const A = y0 * W + x0, B = y0 * W + x1 + 1, C = (y1 + 1) * W + x0, D = (y1 + 1) * W + x1 + 1;
      const c = cnt[D] - cnt[B] - cnt[C] + cnt[A];
      out[y * w + x] = c > 0 ? (sum[D] - sum[B] - sum[C] + sum[A]) / c : 0;
    }
  }
  return out;
}

/**
 * Local background subtraction. `dark` selects the polarity: veins darker than
 * their surroundings (the usual case) give mean - value.
 */
export function topHat(src, dom, w, h, r, dark = true) {
  const mean = maskedMean(src, dom, w, h, r);
  const out = new Uint8ClampedArray(w * h);
  for (let i = 0; i < out.length; i++) {
    if (dom && !dom[i]) { out[i] = 0; continue; }
    out[i] = dark ? mean[i] - src[i] : src[i] - mean[i];
  }
  return out;
}

/* ------------------------------------------------------------ morphology */
/* Rank filters along one of four rasterisable directions. Rolling a window
   along rows / columns / diagonals keeps everything O(1) per pixel, which is
   what makes a live preview possible at all. */

const DIRS = {
  0:   [1, 0],
  45:  [1, -1],
  90:  [0, 1],
  135: [1, 1],
};

/** Min (erode) or max (dilate) along `dir` over a run of `len` pixels. */
function rank1d(src, w, h, dir, len, useMax) {
  const [dx, dy] = DIRS[dir];
  const out = new Uint8ClampedArray(src.length);
  const half = len >> 1;
  const starts = [];
  // every line through the image, walked once
  if (dx === 1 && dy === 0) { for (let y = 0; y < h; y++) starts.push([0, y]); }
  else if (dx === 0 && dy === 1) { for (let x = 0; x < w; x++) starts.push([x, 0]); }
  else if (dy === 1) { for (let x = 0; x < w; x++) starts.push([x, 0]); for (let y = 1; y < h; y++) starts.push([0, y]); }
  else { for (let x = 0; x < w; x++) starts.push([x, h - 1]); for (let y = 0; y < h - 1; y++) starts.push([0, y]); }

  const buf = [];
  for (const [sx, sy] of starts) {
    buf.length = 0;
    let x = sx, y = sy;
    while (x >= 0 && y >= 0 && x < w && y < h) { buf.push(src[y * w + x]); x += dx; y += dy; }
    const n = buf.length;
    const res = new Uint8ClampedArray(n);
    for (let i = 0; i < n; i++) {
      let best = useMax ? 0 : 255;
      const a = Math.max(0, i - half), b = Math.min(n - 1, i + half);
      for (let k = a; k <= b; k++) best = useMax ? Math.max(best, buf[k]) : Math.min(best, buf[k]);
      res[i] = best;
    }
    x = sx; y = sy;
    for (let i = 0; i < n; i++) { out[y * w + x] = res[i]; x += dx; y += dy; }
  }
  return out;
}

/**
 * Directional opening, maximum over several angles.
 *
 * An opening with a LINE keeps only what is at least `len` long in that
 * direction. Leaf areole texture is blobby and vanishes; veins are elongated
 * and survive. Taking the max over angles means a vein only has to be straight
 * enough along ONE of them.
 */
export function ridgeOpen(src, w, h, len, angles = [0, 45, 90, 135]) {
  const out = new Uint8ClampedArray(src.length);
  for (const a of angles) {
    const opened = rank1d(rank1d(src, w, h, a, len, false), w, h, a, len, true);
    for (let i = 0; i < out.length; i++) if (opened[i] > out[i]) out[i] = opened[i];
  }
  return out;
}

/** Binary dilate/erode with a square SE, via two 1D passes. */
export function binMorph(mask, w, h, r, grow) {
  if (!r) return mask;
  const len = 2 * r + 1;
  const a = rank1d(mask, w, h, 0, len, grow);
  return rank1d(a, w, h, 90, len, grow);
}

/* ------------------------------------------------------------- threshold */

/** Otsu's threshold over the pixels inside `dom`. */
export function otsu(src, dom) {
  const hist = new Float64Array(256);
  let n = 0;
  for (let i = 0; i < src.length; i++) {
    if (dom && !dom[i]) continue;
    hist[src[i]]++; n++;
  }
  if (!n) return 128;
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0, wB = 0, best = 0, bestVar = -1;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (!wB) continue;
    const wF = n - wB;
    if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const v = wB * wF * (mB - mF) * (mB - mF);
    if (v > bestVar) { bestVar = v; best = t; }
  }
  return best;
}

/** Drop connected blobs smaller than `minArea` (8-connected). */
export function despeckle(mask, w, h, minArea) {
  if (!minArea || minArea <= 1) return mask;
  const out = new Uint8ClampedArray(mask.length);
  const seen = new Uint8Array(mask.length);
  const stack = new Int32Array(mask.length);
  for (let s = 0; s < mask.length; s++) {
    if (!mask[s] || seen[s]) continue;
    let top = 0, count = 0;
    stack[top++] = s; seen[s] = 1;
    const blob = [];
    while (top) {
      const i = stack[--top];
      blob.push(i); count++;
      const x = i % w, y = (i / w) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const j = ny * w + nx;
          if (mask[j] && !seen[j]) { seen[j] = 1; stack[top++] = j; }
        }
      }
    }
    if (count >= minArea) for (const i of blob) out[i] = 1;
  }
  return out;
}

/* ---------------------------------------------------------------- driver */

/*
 * Defaults measured on a real oriented pair (Liquidambar, 767x780).
 *
 * `ridge` is OFF despite being the strongest noise filter, because the two
 * controls overlap: the directional opening earns its keep at a LOW threshold,
 * where texture would otherwise swamp the result. At a threshold high enough to
 * be usable on its own (18) the noise is already gone, and all the opening does
 * is delete the secondaries -- measured 1.3% vein coverage with it against 3.2%
 * without, on the same image. It stays as a toggle for flat, noisy scans.
 */
export const VEIN_DEFAULTS = {
  channel: "g",
  invert: false,          // true when veins are LIGHTER than the lamina
  bgRadius: 21,           // local-background window; ~2x the widest vein
  gain: 4,                // the top-hat signal is only a few grey levels
  ridge: false,
  ridgeLen: 11,
  ridgeAngles: 4,
  mode: "manual",         // manual | otsu | adaptive
  threshold: 18,
  adaptiveRadius: 15,
  adaptiveOffset: 6,
  closeRadius: 1,
  minArea: 40,
  thicken: 0,
  rimInset: 6,            // stay this far off the silhouette edge
};

/**
 * Full pipeline. `dom` is the leaf silhouette (same pixel grid) or null.
 * Returns { mask, preview } -- preview is the enhanced grayscale, for the UI.
 */
export function extractVeins(imageData, dom, opts = {}) {
  const o = { ...VEIN_DEFAULTS, ...opts };
  const w = imageData.width, h = imageData.height;

  let domain = dom;
  if (domain && o.rimInset > 0) domain = binMorph(domain, w, h, o.rimInset, false);

  const gray = toGray(imageData, o.channel);
  let enh = topHat(gray, domain, w, h, Math.max(1, o.bgRadius | 0), !o.invert);
  if (o.gain !== 1) {
    const g = o.gain;
    const amp = new Uint8ClampedArray(enh.length);
    for (let i = 0; i < enh.length; i++) amp[i] = enh[i] * g;
    enh = amp;
  }
  if (o.ridge) {
    const angles = o.ridgeAngles === 2 ? [0, 90] : [0, 45, 90, 135];
    enh = ridgeOpen(enh, w, h, Math.max(3, o.ridgeLen | 0), angles);
  }

  let bin = new Uint8ClampedArray(w * h);
  if (o.mode === "adaptive") {
    const mean = maskedMean(enh, domain, w, h, Math.max(1, o.adaptiveRadius | 0));
    for (let i = 0; i < bin.length; i++) bin[i] = enh[i] > mean[i] + o.adaptiveOffset ? 1 : 0;
  } else {
    const t = o.mode === "otsu" ? otsu(enh, domain) : o.threshold;
    for (let i = 0; i < bin.length; i++) bin[i] = enh[i] >= t ? 1 : 0;
  }
  if (domain) for (let i = 0; i < bin.length; i++) if (!domain[i]) bin[i] = 0;

  if (o.closeRadius > 0) {
    bin = binMorph(binMorph(bin, w, h, o.closeRadius, true), w, h, o.closeRadius, false);
  }
  bin = despeckle(bin, w, h, o.minArea | 0);
  if (o.thicken > 0) bin = binMorph(bin, w, h, o.thicken | 0, true);
  if (domain) for (let i = 0; i < bin.length; i++) if (!domain[i]) bin[i] = 0;

  let on = 0;
  for (let i = 0; i < bin.length; i++) on += bin[i];
  return { mask: bin, preview: enh, width: w, height: h, coverage: on / (w * h) };
}
