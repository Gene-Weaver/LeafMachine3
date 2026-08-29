import { readFileSync, writeFileSync } from "node:fs";
import { extractVeins, VEIN_DEFAULTS } from "./veins.js";
const rd = (p) => { const b = readFileSync(p); return { w: b.readUInt32LE(0), h: b.readUInt32LE(4), b }; };
const R = rd("/tmp/claude-1001/vein_rgb.bin");
const M = rd("/tmp/claude-1001/vein_mask.bin");
const img = { width: R.w, height: R.h, data: new Uint8Array(R.b.buffer, R.b.byteOffset + 8, R.w * R.h * 4) };
const dom = new Uint8Array(M.b.buffer, M.b.byteOffset + 8, M.w * M.h);

const variants = [
  ["defaults", {}],
  ["no ridge", { ridge: false }],
  ["thr 10", { threshold: 10 }],
  ["thr 30", { threshold: 30 }],
  ["otsu", { mode: "otsu" }],
  ["adaptive", { mode: "adaptive" }],
  ["ridgeLen 21", { ridgeLen: 21 }],
  ["bgRadius 41", { bgRadius: 41 }],
  ["blue ch", { channel: "b" }],
];
const out = {};
for (const [name, o] of variants) {
  const t0 = performance.now();
  const v = extractVeins(img, dom, o);
  const ms = performance.now() - t0;
  out[name] = { ms: +ms.toFixed(0), coveragePct: +(v.coverage * 100).toFixed(2) };
  writeFileSync(`/tmp/claude-1001/vein_${name.replace(/\W+/g, "_")}.raw`, Buffer.from(v.mask));
}
console.log(JSON.stringify(out, null, 1));
console.log("size", R.w, R.h);
