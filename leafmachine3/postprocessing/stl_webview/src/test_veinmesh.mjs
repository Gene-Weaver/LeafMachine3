import { readFileSync } from "node:fs";
import { createRequire as cr } from "node:module";
import { maskToPolygons, extrude } from "./geom.js";
import { extractVeins } from "./veins.js";
const earcut = cr(import.meta.url)("../assets/earcut.js");

const rd = (p) => { const b = readFileSync(p); return { w: b.readUInt32LE(0), h: b.readUInt32LE(4), b }; };
const R = rd("/tmp/claude-1001/vein_rgb.bin"), M = rd("/tmp/claude-1001/vein_mask.bin");
const img = { width: R.w, height: R.h, data: new Uint8Array(R.b.buffer, R.b.byteOffset + 8, R.w * R.h * 4) };
const dom = new Uint8Array(M.b.buffer, M.b.byteOffset + 8, M.w * M.h);
const v = extractVeins(img, dom, { thicken: 1 });

const K = (x, y, z) => `${Math.round(x * 4096)},${Math.round(y * 4096)},${Math.round(z * 4096)}`;
function openEdges(tris) {
  const m = new Map();
  for (let i = 0; i < tris.length; i += 9) {
    const p = [0, 1, 2].map((k) => K(tris[i + k * 3], tris[i + k * 3 + 1], tris[i + k * 3 + 2]));
    for (let k = 0; k < 3; k++) {
      const a = p[k], b = p[(k + 1) % 3];
      const key = a < b ? `${a}|${b}` : `${b}|${a}`;
      m.set(key, (m.get(key) || 0) + 1);
    }
  }
  let open = 0, over = 0;
  for (const c of m.values()) { if (c === 1) open++; else if (c > 2) over++; }
  return { open, over };
}
// does a closed ring self-intersect? O(n^2), fine for the small ones
const seg = (a, b, c, d) => {
  const s = (p, q, r) => Math.sign((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]));
  return s(a,b,c) !== s(a,b,d) && s(c,d,a) !== s(c,d,b);
};
function selfIntersects(r) {
  const n = r.length;
  if (n > 400) return null;                       // skip the monsters
  for (let i = 0; i < n; i++) for (let j = i + 2; j < n; j++) {
    if (i === 0 && j === n - 1) continue;
    if (seg(r[i], r[(i+1)%n], r[j], r[(j+1)%n])) return true;
  }
  return false;
}

for (const simplifyPx of [0.6, 0.3, 0]) {
  const polys = maskToPolygons(v.mask, R.w, R.h, { simplifyPx, minAreaPx: 3 });
  let badMesh = 0, badRing = 0, totOpen = 0, tinyRings = 0;
  for (const p of polys) {
    const t = extrude([p], 0, 0.2, (x, y) => [x * 0.1923, -y * 0.1923], earcut);
    const e = openEdges(t);
    if (e.open || e.over) { badMesh++; totOpen += e.open; }
    for (const r of [p.outer, ...p.holes]) {
      if (r.length < 4) tinyRings++;
      if (selfIntersects(r) === true) { badRing++; break; }
    }
  }
  console.log(`simplifyPx ${simplifyPx}: ${polys.length} polys, ${badMesh} with open/over edges (${totOpen} open), ${badRing} self-intersecting rings, ${tinyRings} rings <4 pts`);
}
