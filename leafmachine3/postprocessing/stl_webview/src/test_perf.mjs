import { createRequire } from "node:module";
import { maskToPolygons, fillHoles, selectMask } from "./geom.js";
createRequire(import.meta.url);
for (const [W,H] of [[800,600],[1600,1200],[3000,2000]]) {
  const m = new Uint8Array(W*H);
  for (let y=0;y<H;y++) for (let x=0;x<W;x++) {
    const dx=(x-W/2)/(W*0.42), dy=(y-H/2)/(H*0.45);
    if (dx*dx+dy*dy<1 && !(Math.hypot(x-W*0.4,y-H*0.4)<H*0.06)) m[y*W+x]=1;
  }
  const t0=performance.now(); const f=fillHoles(m,W,H); const t1=performance.now();
  const p=maskToPolygons(m,W,H,{simplifyPx:1.5,minAreaPx:4}); const t2=performance.now();
  console.log(`${W}x${H}: fillHoles ${(t1-t0).toFixed(0)}ms  contour+simplify ${(t2-t1).toFixed(0)}ms  polys=${p.length} holes=${p.reduce((a,q)=>a+q.holes.length,0)}`);
}
