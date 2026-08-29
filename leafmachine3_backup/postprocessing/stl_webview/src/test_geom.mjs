import { readFileSync, writeFileSync } from "node:fs";
import { createRequire as cr } from "node:module";
import { selectMask, fillHoles, maskToPolygons, extrude, polyBounds, writeStlBinary } from "./geom.js";
const require = cr(import.meta.url);
const earcut = require("../assets/earcut.js");

const buf = readFileSync("/tmp/claude-1001/mask.bin");
const w = buf.readUInt32LE(0), h = buf.readUInt32LE(4);
const imageData = { width: w, height: h, data: new Uint8Array(buf.buffer, buf.byteOffset + 8, w * h * 4) };

let m = selectMask(imageData, ["white"], 0);
const fg = m.reduce((a, b) => a + b, 0);
m = fillHoles(m, w, h);
const filled = m.reduce((a, b) => a + b, 0);

const polys = maskToPolygons(m, w, h, { simplifyPx: 1.5, minAreaPx: 4 });
const b = polyBounds(polys);
const lengthMm = 150, thicknessMm = 2;
const scale = lengthMm / Math.max(b.w, b.h);
const xf = (x, y) => [(x - b.minx) * scale, (b.maxy - y) * scale];   // scale + flip Y, as Python does
const tris = extrude(polys, 0, thicknessMm, xf, earcut);

// bounds + volume via the divergence theorem, to compare with trimesh
let mnx = Infinity, mny = Infinity, mxx = -Infinity, mxy = -Infinity, vol = 0;
for (let i = 0; i < tris.length; i += 9) {
  const [ax, ay, az, bx, by, bz, cx, cy, cz] = tris.slice(i, i + 9);
  vol += (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6;
  for (const [x, y] of [[ax, ay], [bx, by], [cx, cy]]) {
    mnx = Math.min(mnx, x); mny = Math.min(mny, y); mxx = Math.max(mxx, x); mxy = Math.max(mxy, y);
  }
}
// every directed edge must appear exactly twice, once per direction => closed surface
const edges = new Map();
const key = (a, b) => `${a}|${b}`;
const V = (i) => `${tris[i].toFixed(4)},${tris[i+1].toFixed(4)},${tris[i+2].toFixed(4)}`;
for (let i = 0; i < tris.length; i += 9) {
  const a = V(i), b2 = V(i + 3), c = V(i + 6);
  for (const [p, q] of [[a, b2], [b2, c], [c, a]]) edges.set(key(p, q), (edges.get(key(p, q)) || 0) + 1);
}
let unmatched = 0;
for (const [k, n] of edges) { const [p, q] = k.split("|"); if ((edges.get(key(q, p)) || 0) !== n) unmatched++; }

writeFileSync("/tmp/claude-1001/js.stl", Buffer.from(writeStlBinary(tris)));
console.log(JSON.stringify({
  image: `${w}x${h}`, fgPixels: fg, afterFillHoles: filled,
  polygons: polys.length, holes: polys.reduce((a, p) => a + p.holes.length, 0),
  triangles: tris.length / 9,
  size_mm: [ +(mxx - mnx).toFixed(3), +(mxy - mny).toFixed(3), thicknessMm ],
  scale_mm_per_px: +scale.toFixed(6),
  volume_mm3: +vol.toFixed(1),
  closedSurface: unmatched === 0, unmatchedEdges: unmatched,
}, null, 2));
