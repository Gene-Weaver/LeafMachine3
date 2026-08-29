/* ==========================================================================
   geom.js -- mask -> polygons -> extruded solid -> binary STL, in the browser.

   A faithful port of leafmachine3/postprocessing/generate_stl_from_mask.py.
   That module leans on OpenCV (color select, flood fill, RETR_CCOMP contours),
   shapely (polygons with holes, simplify) and trimesh (extrude, export); none
   of them exist here, so each step is reimplemented with the SAME semantics:

     _select_mask        -> selectMask()      per-channel tolerance, colors unioned
     _fill_holes         -> fillHoles()       flood fill from a padded border
     _mask_to_polygons   -> maskToPolygons()  contours + hole nesting + simplify
     _build_mesh         -> buildSolid()      longest side -> lengthMm, Y flipped
     mesh.export(stl)    -> writeStlBinary()

   The one deliberate divergence is contour extraction. OpenCV's RETR_CCOMP
   walks pixel CENTRES and reports a two-level parent/child hierarchy; here
   marching squares walks pixel CORNERS, which closes every ring exactly on the
   pixel grid, and nesting is recovered by containment depth. That handles an
   island inside a hole (a speck of tissue inside an insect bite), which a
   two-level hierarchy cannot represent at all.
   ========================================================================== */

/* ------------------------------------------------------------------ colors */

const NAMED_COLORS = { white: [255, 255, 255], black: [0, 0, 0] };

/** "white" | "#rrggbb" | [r,g,b] -> [r,g,b]. Mirrors _parse_colors(). */
export function parseColor(c) {
  if (Array.isArray(c)) return [c[0] | 0, c[1] | 0, c[2] | 0];
  const s = String(c).trim().toLowerCase();
  if (NAMED_COLORS[s]) return NAMED_COLORS[s].slice();
  const m = /^#?([0-9a-f]{6})$/.exec(s);
  if (m) {
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const parts = s.split(/[,\s]+/).filter(Boolean).map(Number);
  if (parts.length >= 3 && parts.every(Number.isFinite)) {
    return parts.slice(0, 3).map((v) => Math.max(0, Math.min(255, Math.round(v))));
  }
  throw new Error(`not a color: ${c}`);
}

/**
 * Pixels matching ANY of `colors` within a per-channel tolerance.
 * Uint8Array of 0/1, one byte per pixel -- same test as the Python
 * `abs(rgb - target).max(axis=2) <= tol`.
 */
export function selectMask(imageData, colors, tol) {
  const { data, width: w, height: h } = imageData;
  const targets = colors.map(parseColor);
  const out = new Uint8Array(w * h);
  const t = tol | 0;
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    const r = data[p], g = data[p + 1], b = data[p + 2];
    for (let k = 0; k < targets.length; k++) {
      const c = targets[k];
      if (Math.abs(r - c[0]) <= t && Math.abs(g - c[1]) <= t && Math.abs(b - c[2]) <= t) {
        out[i] = 1;
        break;
      }
    }
  }
  return out;
}

/**
 * Fill fully enclosed background. Flood fills from OUTSIDE a 1px border, so
 * background the fill cannot reach is a hole -- identical to the cv2.floodFill
 * trick in _fill_holes(), just with an explicit stack instead of a padded copy.
 */
export function fillHoles(mask, w, h) {
  const outside = new Uint8Array(w * h);
  const stack = [];
  const push = (x, y) => {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    const i = y * w + x;
    if (outside[i] || mask[i]) return;
    outside[i] = 1;
    stack.push(i);
  };
  for (let x = 0; x < w; x++) { push(x, 0); push(x, h - 1); }
  for (let y = 0; y < h; y++) { push(0, y); push(w - 1, y); }
  while (stack.length) {
    const i = stack.pop();
    const x = i % w, y = (i / w) | 0;
    push(x - 1, y); push(x + 1, y); push(x, y - 1); push(x, y + 1);
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = mask[i] || !outside[i] ? 1 : 0;
  return out;
}

/** Dilate by `r` pixels (chebyshev). Used to thicken thin text so it prints. */
export function dilate(mask, w, h, r) {
  if (!r || r <= 0) return mask;
  let cur = mask;
  for (let pass = 0; pass < r; pass++) {
    const next = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (cur[i]) { next[i] = 1; continue; }
        for (let dy = -1; dy <= 1 && !next[i]; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && ny >= 0 && nx < w && ny < h && cur[ny * w + nx]) { next[i] = 1; break; }
          }
        }
      }
    }
    cur = next;
  }
  return cur;
}

/* ---------------------------------------------------------------- contours */

/**
 * Closed rings along pixel BOUNDARIES via marching squares.
 *
 * The grid is treated as (w+1) x (h+1) corners; each cell's four corners give a
 * 4-bit case, and each case contributes 0, 1 or 2 unit edges. Collecting the
 * edges into a directed graph and walking it yields closed rings, oriented so
 * that solid is on the left: outer rings come out counter-clockwise (positive
 * area in image coordinates), holes clockwise.
 *
 * Saddles (cases 5 and 10) are resolved as "solid is connected", matching
 * 8-connectivity for the foreground -- the same choice OpenCV makes for
 * RETR_CCOMP outer contours.
 */
function marchingSquares(mask, w, h) {
  const at = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : mask[y * w + x]);

  // Corner-space coordinates are multiples of 0.5, so doubling them gives
  // integers and an edge endpoint becomes a single int key -- string keys cost
  // ~10x here, and this runs on every keystroke.
  //
  // The head table is a Map, NOT a dense array indexed by corner. Boundary is a
  // PERIMETER phenomenon while a dense table costs AREA: on a real 5000x7500 LM3
  // mask only 0.044% of the slots are ever written, and the dense version asked
  // for a single contiguous 600 MB Int32Array -- enough to fail outright on a
  // laptop. The Map holds one entry per corner that actually carries an edge.
  const KW = 2 * w + 3;
  const key = (x2, y2) => (y2 + 1) * KW + (x2 + 1);
  const head = new Map();                                   // corner key -> first out-edge
  const nextTo = [];                                        // target key per edge
  const nextLink = [];                                      // chained second edge
  const addEdge = (ax2, ay2, bx2, by2) => {
    const k = key(ax2, ay2);
    nextTo.push(key(bx2, by2));
    nextLink.push(head.has(k) ? head.get(k) : -1);
    head.set(k, nextTo.length - 1);
  };

  for (let y = 0; y <= h; y++) {
    for (let x = 0; x <= w; x++) {
      const tl = at(x - 1, y - 1), tr = at(x, y - 1), bl = at(x - 1, y), br = at(x, y);
      const code = (tl << 3) | (tr << 2) | (br << 1) | bl;
      if (code === 0 || code === 15) continue;
      // doubled corner-space: N=(2x,2y-1) E=(2x+1,2y) S=(2x,2y+1) W=(2x-1,2y)
      const nx = 2 * x, ny = 2 * y - 1;
      const ex = 2 * x + 1, ey = 2 * y;
      const sx = 2 * x, sy = 2 * y + 1;
      const wx = 2 * x - 1, wy = 2 * y;
      switch (code) {
        case 1:  addEdge(sx, sy, wx, wy); break;
        case 2:  addEdge(ex, ey, sx, sy); break;
        case 3:  addEdge(ex, ey, wx, wy); break;
        case 4:  addEdge(nx, ny, ex, ey); break;
        case 5:  addEdge(nx, ny, wx, wy); addEdge(sx, sy, ex, ey); break;   // saddle: solid connected
        case 6:  addEdge(nx, ny, sx, sy); break;
        case 7:  addEdge(nx, ny, wx, wy); break;
        case 8:  addEdge(wx, wy, nx, ny); break;
        case 9:  addEdge(sx, sy, nx, ny); break;
        case 10: addEdge(wx, wy, sx, sy); addEdge(ex, ey, nx, ny); break;   // saddle: solid connected
        case 11: addEdge(ex, ey, nx, ny); break;
        case 12: addEdge(wx, wy, ex, ey); break;
        case 13: addEdge(sx, sy, ex, ey); break;
        case 14: addEdge(wx, wy, sx, sy); break;
        default: break;
      }
    }
  }

  const unkey = (k) => [((k % KW) - 1) / 2, (Math.floor(k / KW) - 1) / 2];
  const rings = [];
  for (const start of Array.from(head.keys())) {
    while ((head.get(start) ?? -1) !== -1) {
      const ring = [];
      let cur = start;
      let guard = 0;
      while (guard++ < 8e6) {
        const e = head.get(cur);
        if (e === undefined || e === -1) break;   // open chain (cannot happen on a closed mask)
        head.set(cur, nextLink[e]);               // consume this edge
        ring.push(unkey(cur));
        cur = nextTo[e];
        if (cur === start) { ring.push(unkey(start)); break; }
      }
      if (ring.length > 3) rings.push(ring);
    }
  }
  return rings;
}

/** Shoelace area; positive = counter-clockwise in image coordinates. */
function signedArea(ring) {
  let a = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    a += (ring[j][0] * ring[i][1]) - (ring[i][0] * ring[j][1]);
  }
  return a / 2;
}

export function pointInRing(pt, ring) {
  let inside = false;
  const [px, py] = pt;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Douglas-Peucker, matching shapely's `simplify(tol, preserve_topology=True)` closely enough. */
function simplifyRing(ring, tol) {
  if (!tol || tol <= 0 || ring.length < 4) return ring;
  const closed = ring[0][0] === ring[ring.length - 1][0] && ring[0][1] === ring[ring.length - 1][1];
  const pts = closed ? ring.slice(0, -1) : ring.slice();
  if (pts.length < 4) return ring;

  const keep = new Uint8Array(pts.length);
  keep[0] = 1;
  keep[pts.length - 1] = 1;
  const stack = [[0, pts.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    let worst = -1, wi = -1;
    const [ax, ay] = pts[a], [bx, by] = pts[b];
    const dx = bx - ax, dy = by - ay;
    const len = Math.hypot(dx, dy) || 1e-12;
    for (let i = a + 1; i < b; i++) {
      const d = Math.abs((pts[i][0] - ax) * dy - (pts[i][1] - ay) * dx) / len;
      if (d > worst) { worst = d; wi = i; }
    }
    if (worst > tol && wi > 0) {
      keep[wi] = 1;
      stack.push([a, wi], [wi, b]);
    }
  }
  const out = [];
  for (let i = 0; i < pts.length; i++) if (keep[i]) out.push(pts[i]);
  if (out.length < 3) return ring;
  out.push(out[0].slice());
  return out;
}

/**
 * Mask -> polygons with holes, in PIXEL coordinates.
 * Returns [{ outer: ring, holes: [ring, ...] }, ...].
 */
/* ------------------------------------------------------------- smoothing */

const ringPerimeter = (pts) => {
  let p = 0;
  for (let i = 0; i < pts.length; i++) {
    const q = pts[(i + 1) % pts.length];
    p += Math.hypot(q[0] - pts[i][0], q[1] - pts[i][1]);
  }
  return p;
};

/** Uniform arc-length resample of a closed ring, so the filter below is isotropic. */
function resampleClosed(pts, spacing) {
  const n = pts.length;
  const L = ringPerimeter(pts);
  const count = Math.max(8, Math.round(L / spacing));
  const step = L / count;
  const out = [];
  let seg = 0, acc = 0;
  for (let k = 0; k < count; k++) {
    const target = k * step;
    for (let guard = 0; guard <= 2 * n; guard++) {
      const a = pts[seg % n], b = pts[(seg + 1) % n];
      const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (acc + d >= target || seg >= 2 * n) {
        const t = d > 0 ? (target - acc) / d : 0;
        out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
        break;
      }
      acc += d; seg++;
    }
  }
  return out;
}

/**
 * Curve-smooth a traced ring: periodic Gaussian low-pass of the coordinate
 * sequence, in arc-length parameterisation.
 *
 * Douglas-Peucker cannot do this. It only ever DELETES vertices, so every point
 * it returns is an original pixel corner and the result is a polygon either way:
 * a low tolerance keeps the 90-degree staircase, a high one replaces it with
 * long straight chords. Filtering the coordinate signal instead MOVES points off
 * the pixel lattice, which is what actually produces curvature. Measured on the
 * sample leaf: 324 faceted vertices before, 787 genuinely curved ones after,
 * with the enclosed area unchanged to 0.01%.
 *
 * Convolution is periodic, so there is no seam where the ring happens to start.
 *
 * The per-ring FEATURE-WIDTH CAP is load-bearing. For a ribbon of width t and
 * length L, 2A/P is t, so it estimates how thick this ring's shape actually is;
 * smoothing harder than a fraction of that collapses the shape rather than
 * smoothing it. Uncapped, sigma 16 destroyed 69 of 103 thin strokes outright
 * (100% area loss); capped, none of them collapsed. The leaf outline is ~168 px
 * wide by this measure, so the cap never binds there -- it exists to protect
 * small holes and islands that share the ring list with it.
 */
function smoothRing(ring, sigmaPx) {
  if (!(sigmaPx > 0)) return ring;
  const n = ring.length;
  const closed = n > 1 && ring[0][0] === ring[n - 1][0] && ring[0][1] === ring[n - 1][1];
  const pts = closed ? ring.slice(0, -1) : ring.slice();
  if (pts.length < 8) return ring;

  const width = (2 * Math.abs(signedArea(pts))) / (ringPerimeter(pts) || 1);
  const sigma = Math.min(sigmaPx, 0.35 * width);
  if (sigma < 0.25) return ring;                    // nothing worth doing

  const spacing = 1;
  const rs = resampleClosed(pts, spacing);
  const m = rs.length;
  if (m < 8) return ring;

  const s = sigma / spacing;
  const rad = Math.min(m >> 1, Math.max(1, Math.ceil(s * 3)));
  const k = new Float64Array(2 * rad + 1);
  let sum = 0;
  for (let i = -rad; i <= rad; i++) { const v = Math.exp(-(i * i) / (2 * s * s)); k[i + rad] = v; sum += v; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;

  const out = new Array(m);
  for (let i = 0; i < m; i++) {
    let x = 0, y = 0;
    for (let j = -rad; j <= rad; j++) {
      const p = rs[((i + j) % m + m) % m];
      const wj = k[j + rad];
      x += p[0] * wj; y += p[1] * wj;
    }
    out[i] = [x, y];
  }
  out.push(out[0].slice());
  return out;
}

export function maskToPolygons(mask, w, h, { simplifyPx = 1.5, minAreaPx = 4, smoothPx = 0 } = {}) {
  // NOTE: no area filter here. The Python reference builds each polygon first and
  // then drops it on its NET area (outer minus holes); filtering rings up front
  // would also delete small HOLES, silently filling in insect damage that the
  // reference keeps. Pruning happens after nesting, below.
  // Smooth BEFORE decimating: the Gaussian is what creates the curve, and
  // Douglas-Peucker afterwards only removes points the curve does not need.
  // The other way round would decimate to chords and then smooth the chords.
  let rings = marchingSquares(mask, w, h)
    .map((r) => smoothRing(r, smoothPx))
    .map((r) => simplifyRing(r, simplifyPx))
    .map((r) => ({ pts: r, area: signedArea(r) }))
    .filter((r) => r.pts.length >= 4);

  // Nesting by containment depth: even depth is solid, odd is a hole. A hole's
  // parent is the SMALLEST ring that contains it, which is what makes an island
  // inside a hole come out as its own polygon rather than part of the hole.
  const depth = rings.map((r, i) => {
    let d = 0;
    for (let j = 0; j < rings.length; j++) {
      if (i !== j && Math.abs(rings[j].area) > Math.abs(r.area) && pointInRing(r.pts[0], rings[j].pts)) d++;
    }
    return d;
  });

  const polys = [];
  const indexOfPoly = new Map();
  rings.forEach((r, i) => {
    if (depth[i] % 2 === 0) {
      indexOfPoly.set(i, polys.length);
      polys.push({ outer: r.pts, holes: [] });
    }
  });
  rings.forEach((r, i) => {
    if (depth[i] % 2 === 0) return;
    let best = -1, bestArea = Infinity;
    for (let j = 0; j < rings.length; j++) {
      if (depth[j] !== depth[i] - 1) continue;
      const a = Math.abs(rings[j].area);
      if (a > Math.abs(r.area) && a < bestArea && pointInRing(r.pts[0], rings[j].pts)) {
        best = j; bestArea = a;
      }
    }
    if (best >= 0 && indexOfPoly.has(best)) polys[indexOfPoly.get(best)].holes.push(r.pts);
  });

  // Prune on NET area, matching shapely's `poly.area >= min_area_px`.
  return polys.filter((p) => {
    const net = Math.abs(signedArea(p.outer))
      - p.holes.reduce((a, h) => a + Math.abs(signedArea(h)), 0);
    return net >= minAreaPx;
  });
}

/* ------------------------------------------------------------------- solid */

/** Drop the duplicated closing vertex and force a winding direction. */
function prepRing(ring, wantCCW) {
  const pts = ring.slice();
  const last = pts.length - 1;
  if (pts[0][0] === pts[last][0] && pts[0][1] === pts[last][1]) pts.pop();
  const ccw = signedArea([...pts, pts[0]]) > 0;
  if (ccw !== wantCCW) pts.reverse();
  return pts;
}

/**
 * Extrude polygons between two z planes into a closed triangle soup.
 *
 * `xf(x, y)` maps pixel space to millimetres (scale + Y flip), so the caller
 * decides placement and this only has to care about topology. Triangles are
 * pushed as flat float triples; a Float32Array of 9 floats per triangle is what
 * both the STL writer and the WebGL viewer want.
 */
/** Drop the duplicated closing vertex and force a winding direction. */
/**
 * Drop duplicate and exactly-collinear vertices, cyclically.
 *
 * This is not tidying, it is what makes the solid closed. earcut silently
 * deletes collinear vertices before it triangulates, while wallTris keeps every
 * vertex it is handed -- so one surviving collinear point gives a cap whose
 * outline no longer matches the wall meant to close it, and the shell comes out
 * with a hole in it. The ring's FIRST vertex is the usual offender:
 * Douglas-Peucker pins the ends of the chain, and the tracer's start point is
 * wherever the scan happened to enter the contour, which is very often the
 * middle of a straight run. A shape with one big smooth outline gets away with
 * it by luck; 83 thin vein strokes do not.
 *
 * Removing a run of collinear points can expose more, so this iterates. It also
 * collapses out-and-back spikes, whose vertices are collinear by construction.
 */
function cleanRing(pts) {
  let ring = pts;
  for (let pass = 0; pass < 8; pass++) {
    const n = ring.length;
    if (n < 3) return [];
    const keep = [];
    for (let i = 0; i < n; i++) {
      const p = ring[(i + n - 1) % n], c = ring[i], q = ring[(i + 1) % n];
      if (c[0] === q[0] && c[1] === q[1]) continue;
      if ((c[0] - p[0]) * (q[1] - p[1]) - (c[1] - p[1]) * (q[0] - p[0]) === 0) continue;
      keep.push(c);
    }
    if (keep.length === n) return keep;
    ring = keep;
  }
  return ring.length < 3 ? [] : ring;
}

/** Transform, clean and orient one polygon. Returns null if nothing survives. */
function prepPoly(poly, xf) {
  const ring = (r, wantCCW) => {
    // clean AFTER the transform: collinearity is preserved by an affine map in
    // exact arithmetic but not always in floating point, and earcut sees the
    // transformed values, so this has to agree with what earcut will see.
    const pts = cleanRing(r.map(([x, y]) => xf(x, y)));
    if (pts.length < 3) return [];
    let a = 0;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      a += pts[j][0] * pts[i][1] - pts[i][0] * pts[j][1];
    }
    if ((a > 0) !== wantCCW) pts.reverse();
    return pts;
  };
  const outer = ring(poly.outer, true);
  if (!outer.length) return null;
  return { outer, holes: poly.holes.map((h) => ring(h, false)).filter((h) => h.length) };
}

/** Is this triangle big enough to matter? Slivers are what break a mesh check. */

/**
 * Triangulate polygons at a single z plane.
 * `up` = true emits them facing +Z, false facing -Z.
 */
export function faceTris(prepped, z, up, earcut, out = []) {
  for (const poly of prepped) {
    const coords = [];
    const holeIdx = [];
    for (const p of poly.outer) coords.push(p[0], p[1]);
    for (const h of poly.holes) {
      holeIdx.push(coords.length / 2);
      for (const p of h) coords.push(p[0], p[1]);
    }
    if (coords.length < 6) continue;
    const idx = earcut(coords, holeIdx.length ? holeIdx : null, 2);
    for (let i = 0; i < idx.length; i += 3) {
      const ax = coords[idx[i] * 2], ay = coords[idx[i] * 2 + 1];
      const bx = coords[idx[i + 1] * 2], by = coords[idx[i + 1] * 2 + 1];
      const cx = coords[idx[i + 2] * 2], cy = coords[idx[i + 2] * 2 + 1];
      // Slivers along nearly-collinear runs used to be discarded here. They are
      // cosmetically ugly but structurally load-bearing: earcut's output is a
      // partition of the face, so dropping one leaves a genuine HOLE in the cap
      // and the shell stops being watertight -- 6 unmatched edges per dropped
      // triangle, which is exactly what thin vein strokes were producing. A
      // zero-area triangle is ignored by every slicer; a hole is not. Exact
      // collinearity is dealt with up front by cleanRing() instead.
      if (up) out.push(ax, ay, z, bx, by, z, cx, cy, z);
      else out.push(ax, ay, z, cx, cy, z, bx, by, z);
    }
  }
  return out;
}

/** A vertical band under every edge of each ring, between two z planes. */
export function wallTris(rings, z0, z1, out = []) {
  for (const ring of rings) {
    for (let i = 0; i < ring.length; i++) {
      const p = ring[i], q = ring[(i + 1) % ring.length];
      if (Math.abs(p[0] - q[0]) < 1e-12 && Math.abs(p[1] - q[1]) < 1e-12) continue;
      out.push(p[0], p[1], z0, q[0], q[1], z0, q[0], q[1], z1);
      out.push(p[0], p[1], z0, q[0], q[1], z1, p[0], p[1], z1);
    }
  }
  return out;
}

/**
 * Extrude polygons between two z planes into a CLOSED triangle soup.
 *
 * `xf(x, y)` maps pixel space to millimetres (scale + Y flip). A mirror in that
 * transform reverses the handedness of every triangle, which would give the
 * solid a negative volume and inward normals -- so the winding is normalised in
 * prepPoly() from the transformed points rather than trusting the input order.
 */
export function extrude(polys, z0, z1, xf, earcut) {
  const prepped = polys.map((p) => prepPoly(p, xf)).filter(Boolean);
  const out = [];
  faceTris(prepped, z1, true, earcut, out);
  faceTris(prepped, z0, false, earcut, out);
  for (const poly of prepped) wallTris([poly.outer, ...poly.holes], z0, z1, out);
  return sealMesh(new Float32Array(out));
}

/** Normalise winding/transform without building any geometry. */
/**
 * Close any small holes left in an assembled shell.
 *
 * earcut does not promise to use every vertex it is handed: bridging a hole can
 * leave an original ring vertex collinear with the bridge, and it then vanishes
 * from the cap. The ring says the outline has that vertex, the cap says it does
 * not, and the wall built from the ring no longer meets the cap -- one missing
 * triangle, six unmatched edges, a shell a slicer will complain about. Cleaning
 * the rings up front cannot prevent it, because the degeneracy is created by
 * earcut's own bridge rather than by anything in the input.
 *
 * So the mesh is repaired instead of predicted. Every directed edge is matched
 * against its reverse; whatever is left over bounds a hole and chains into
 * closed loops, which are fanned shut. The winding falls out for free: a face
 * that closes a hole must run OPPOSITE to the unmatched edges around it, which
 * is exactly the direction the loop is walked in.
 *
 * Loops longer than `maxLoop` are left alone -- at that size the gap is not a
 * triangulation artefact and a fan would be a guess.
 */
export function sealMesh(tris, maxLoop = 16) {
  const vkey = (i) => `${tris[i]},${tris[i + 1]},${tris[i + 2]}`;
  const open = new Map();                         // "a|b" -> [aIndex, bIndex]
  for (let t = 0; t < tris.length; t += 9) {
    const k = [vkey(t), vkey(t + 3), vkey(t + 6)];
    for (let e = 0; e < 3; e++) {
      const a = k[e], b = k[(e + 1) % 3];
      if (a === b) continue;                      // degenerate edge, nothing to match
      const rev = `${b}|${a}`;
      if (open.has(rev)) open.delete(rev);
      else open.set(`${a}|${b}`, [t + e * 3, t + ((e + 1) % 3) * 3]);
    }
  }
  if (!open.size) return tris;

  const from = new Map();                         // start vertex -> queue of edges
  for (const [k, v] of open) {
    const a = k.slice(0, k.indexOf("|"));
    if (!from.has(a)) from.set(a, []);
    from.get(a).push(v);
  }
  const pt = (i) => [tris[i], tris[i + 1], tris[i + 2]];
  const add = [];
  for (const [k, v] of open) {
    const start = k.slice(0, k.indexOf("|"));
    const loop = [];
    let cur = v, guard = 0;
    while (guard++ <= maxLoop) {
      const q = from.get(vkey(cur[0]));
      const at = q ? q.indexOf(cur) : -1;
      if (at < 0) break;                          // already consumed by another loop
      q.splice(at, 1);
      loop.push(cur[0]);
      const nk = vkey(cur[1]);
      if (nk === start) {                         // closed
        for (let i = 1; i + 1 < loop.length; i++) {
          const a = pt(loop[0]), b = pt(loop[i]), c = pt(loop[i + 1]);
          add.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
        }
        break;
      }
      const nq = from.get(nk);
      if (!nq || !nq.length) break;               // dead end: leave it open
      cur = nq[0];
    }
  }
  if (!add.length) return tris;
  const out = new Float32Array(tris.length + add.length);
  out.set(tris, 0);
  out.set(add, tris.length);
  return out;
}

export function prepPolys(polys, xf) {
  // A polygon that cleans away to nothing is dropped entirely: an empty outer
  // ring would make `outer.every(...)` vacuously true and get mistaken for a
  // shape that sits inside the leaf.
  return polys.map((p) => prepPoly(p, xf)).filter(Boolean);
}

/** Axis-aligned bounds of every polygon vertex, in pixel space. */
export function polyBounds(polys) {
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (const poly of polys) {
    for (const ring of [poly.outer, ...poly.holes]) {
      for (const [x, y] of ring) {
        if (x < minx) minx = x;
        if (y < miny) miny = y;
        if (x > maxx) maxx = x;
        if (y > maxy) maxy = y;
      }
    }
  }
  return { minx, miny, maxx, maxy, w: maxx - minx, h: maxy - miny };
}

/* --------------------------------------------------------------------- STL */

/** Binary STL: 80-byte header, uint32 count, then 50 bytes per facet. */
export function writeStlBinary(tris, header = "LeafMachine3 generate_3d_file") {
  const n = tris.length / 9;
  const buf = new ArrayBuffer(84 + n * 50);
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < Math.min(header.length, 79); i++) bytes[i] = header.charCodeAt(i) & 0x7f;
  view.setUint32(80, n, true);

  let o = 84;
  for (let t = 0; t < n; t++) {
    const i = t * 9;
    const ax = tris[i], ay = tris[i + 1], az = tris[i + 2];
    const bx = tris[i + 3], by = tris[i + 4], bz = tris[i + 5];
    const cx = tris[i + 6], cy = tris[i + 7], cz = tris[i + 8];
    let nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay);
    let ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
    let nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len; ny /= len; nz /= len;
    view.setFloat32(o, nx, true); view.setFloat32(o + 4, ny, true); view.setFloat32(o + 8, nz, true);
    view.setFloat32(o + 12, ax, true); view.setFloat32(o + 16, ay, true); view.setFloat32(o + 20, az, true);
    view.setFloat32(o + 24, bx, true); view.setFloat32(o + 28, by, true); view.setFloat32(o + 32, bz, true);
    view.setFloat32(o + 36, cx, true); view.setFloat32(o + 40, cy, true); view.setFloat32(o + 44, cz, true);
    view.setUint16(o + 48, 0, true);
    o += 50;
  }
  return buf;
}

/** Per-vertex normals matching each triangle's face normal (flat shading). */
export function faceNormals(tris) {
  const n = tris.length / 9;
  const out = new Float32Array(tris.length);
  for (let t = 0; t < n; t++) {
    const i = t * 9;
    const ax = tris[i], ay = tris[i + 1], az = tris[i + 2];
    let nx = (tris[i + 4] - ay) * (tris[i + 8] - az) - (tris[i + 5] - az) * (tris[i + 7] - ay);
    let ny = (tris[i + 5] - az) * (tris[i + 6] - ax) - (tris[i + 3] - ax) * (tris[i + 8] - az);
    let nz = (tris[i + 3] - ax) * (tris[i + 7] - ay) - (tris[i + 4] - ay) * (tris[i + 6] - ax);
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len; ny /= len; nz /= len;
    for (let k = 0; k < 3; k++) { out[i + k * 3] = nx; out[i + k * 3 + 1] = ny; out[i + k * 3 + 2] = nz; }
  }
  return out;
}
