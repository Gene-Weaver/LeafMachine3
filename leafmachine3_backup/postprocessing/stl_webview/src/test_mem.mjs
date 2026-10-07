import { createRequire } from "node:module";
import { maskToPolygons, fillHoles } from "./geom.js";
createRequire(import.meta.url);
// a realistic LM3 whole-sheet mask: 5000 x 7500 (the review measured these on real runs)
for (const [W,H] of [[1361,1256],[5000,7500]]) {
  const m = new Uint8Array(W*H);
  for (let y=0;y<H;y++) for (let x=0;x<W;x++) {
    const dx=(x-W/2)/(W*0.44), dy=(y-H/2)/(H*0.46);
    if (dx*dx+dy*dy<1) m[y*W+x]=1;
  }
  if (global.gc) global.gc();
  const before = process.memoryUsage().heapUsed;
  const t0 = performance.now();
  const p = maskToPolygons(m, W, H, { simplifyPx: 1.5, minAreaPx: 4 });
  const t1 = performance.now();
  const peak = process.memoryUsage().heapUsed - before;
  const px = W*H;
  console.log(`${W}x${H} (${(px/1e6).toFixed(1)} MP): ${(t1-t0).toFixed(0)}ms  polys=${p.length}  `
    + `heapDelta=${(peak/1048576).toFixed(0)} MB  (dense table would have been `
    + `${(((2*W+3)*(2*H+3)*4)/1048576).toFixed(0)} MB)`);
}
