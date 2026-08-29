/* Adversarial cases for curved smoothing: thin strokes, convoluted outlines,
   extreme sigma, and the feature-width guard. */
import { readFileSync } from "node:fs";
import { selectMask, fillHoles, maskToPolygons } from "./geom.js";
import { extractVeins } from "./veins.js";

const openRing = (r) => { const n = r.length;
  return (n > 1 && r[0][0] === r[n-1][0] && r[0][1] === r[n-1][1]) ? r.slice(0,-1) : r.slice(); };
const area = (r) => { let a=0; for (let i=0,j=r.length-1;i<r.length;j=i++) a += r[j][0]*r[i][1]-r[i][0]*r[j][1]; return a/2; };
const perim = (r) => { let p=0; for (let i=0;i<r.length;i++){const q=r[(i+1)%r.length]; p+=Math.hypot(q[0]-r[i][0],q[1]-r[i][1]);} return p; };
function resample(ring, spacing) {
  const p = openRing(ring), n = p.length;
  if (n < 3) return p;
  const L = perim(p), count = Math.max(8, Math.round(L / spacing)), step = L / count;
  const out = []; let seg = 0, acc = 0;
  for (let k = 0; k < count; k++) { const target = k * step;
    while (true) { const a = p[seg%n], b = p[(seg+1)%n];
      const d = Math.hypot(b[0]-a[0], b[1]-a[1]);
      if (acc + d >= target || seg >= n*2) { const t = d>0 ? (target-acc)/d : 0;
        out.push([a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t]); break; }
      acc += d; seg++; } }
  return out;
}
function gaussClosed(pts, sigmaPts) {
  const n = pts.length;
  if (n < 5 || sigmaPts <= 0) return pts;
  const rad = Math.max(1, Math.ceil(sigmaPts*3)), k = []; let s = 0;
  for (let i=-rad;i<=rad;i++){const v=Math.exp(-(i*i)/(2*sigmaPts*sigmaPts)); k.push(v); s+=v;}
  for (let i=0;i<k.length;i++) k[i]/=s;
  const out = new Array(n);
  for (let i=0;i<n;i++){ let x=0,y=0;
    for (let j=-rad;j<=rad;j++){ const p=pts[((i+j)%n+n)%n], wj=k[j+rad]; x+=p[0]*wj; y+=p[1]*wj; }
    out[i]=[x,y]; }
  return out;
}
function selfIntersects(r) {
  const n = r.length; if (n > 3000) return "skip";
  const o = (p,q,s) => { const v=(q[0]-p[0])*(s[1]-p[1])-(q[1]-p[1])*(s[0]-p[0]); return v>1e-12?1:v<-1e-12?-1:0; };
  const X = (a,b,c,d) => o(a,b,c)!==o(a,b,d) && o(c,d,a)!==o(c,d,b);
  let hits = 0;
  for (let i=0;i<n;i++) for (let j=i+2;j<n;j++){ if(i===0&&j===n-1) continue;
    if (X(r[i], r[(i+1)%n], r[j], r[(j+1)%n])) hits++; }
  return hits;
}
/** mean width of a ring: for a ribbon of width t and length L, 2A/P ~= t */
const meanWidth = (r) => 2 * Math.abs(area(r)) / (perim(r) || 1);

const rd = (p) => { const b = readFileSync(p); return { w: b.readUInt32LE(0), h: b.readUInt32LE(4), b }; };

function ringsOf(which) {
  const B = rd(`/tmp/claude-1001/bnd_${which}.bin`);
  const img = { width: B.w, height: B.h, data: new Uint8Array(B.b.buffer, B.b.byteOffset+8, B.w*B.h*4) };
  let m = selectMask(img, ["white"], 0);
  m = fillHoles(m, B.w, B.h);
  return maskToPolygons(m, B.w, B.h, { simplifyPx: 0, minAreaPx: 4 })
    .flatMap((p) => [p.outer, ...p.holes]);
}
function veinRings() {
  const R = rd("/tmp/claude-1001/vein_rgb.bin"), M = rd("/tmp/claude-1001/vein_mask.bin");
  const img = { width: R.w, height: R.h, data: new Uint8Array(R.b.buffer, R.b.byteOffset+8, R.w*R.h*4) };
  const dom = new Uint8Array(M.b.buffer, M.b.byteOffset+8, M.w*M.h);
  const v = extractVeins(img, dom, {});
  return maskToPolygons(v.mask, R.w, R.h, { simplifyPx: 0, minAreaPx: 3 })
    .flatMap((p) => [p.outer, ...p.holes]);
}

const SETS = { "vein strokes": veinRings() };
const GUARD = Number(process.argv[2] ?? 0.35);   // sigma <= GUARD * meanWidth

console.log("GUARD =", GUARD);
for (const [label, rings] of Object.entries(SETS)) {
  const widths = rings.map(meanWidth).sort((a,b)=>a-b);
  console.log(`\n=== ${label}: ${rings.length} rings, mean width min ${widths[0].toFixed(2)} / median ${widths[widths.length>>1].toFixed(2)} px`);
  for (const sigma of [1, 2, 4, 8, 16]) {
    let si = 0, lost = 0, capped = 0, skipped = 0, worstArea = 0;
    for (const r of rings) {
      const mw = meanWidth(r);
      const eff = Math.min(sigma, GUARD * mw);
      if (eff < sigma - 1e-9) capped++;
      const pts = resample(r, 1);
      if (pts.length < 8) { skipped++; continue; }
      const sm = gaussClosed(pts, eff);
      const a0 = Math.abs(area(openRing(r))), a1 = Math.abs(area(sm));
      if (a1 < a0 * 0.25) lost++;
      worstArea = Math.max(worstArea, Math.abs(a1/a0 - 1));
      const x = selfIntersects(sm);
      if (x !== "skip") si += x;
    }
    console.log(`  sigma ${String(sigma).padStart(2)}: self-int ${String(si).padStart(4)}  `
      + `collapsed ${String(lost).padStart(3)}  guard-capped ${String(capped).padStart(3)}/${rings.length}  `
      + `worst area change ${(worstArea*100).toFixed(1)}%`);
  }
}
