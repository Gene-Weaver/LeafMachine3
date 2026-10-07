import { createRequire } from "node:module";
import { maskToPolygons, extrude, fillHoles, polyBounds } from "./geom.js";
const earcut = createRequire(import.meta.url)("../assets/earcut.js");

const W = 200, H = 200;
const m = new Uint8Array(W * H);
const disc = (cx, cy, r, v) => {
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) m[y * W + x] = v;
};
disc(100, 100, 80, 1);   // solid disc
disc(100, 100, 45, 0);   // punch a hole -> annulus
disc(100, 100, 18, 1);   // island floating inside the hole
disc(30, 30, 6, 1);      // a separate small blob (tests min-area + multi-part)

const vol = (tris) => {
  let v = 0;
  for (let i = 0; i < tris.length; i += 9) {
    const [ax,ay,az,bx,by,bz,cx,cy,cz] = tris.slice(i, i+9);
    v += (ax*(by*cz-bz*cy) - ay*(bx*cz-bz*cx) + az*(bx*cy-by*cx))/6;
  }
  return v;
};
const closed = (tris) => {
  const e = new Map(), V = i => `${tris[i].toFixed(4)},${tris[i+1].toFixed(4)},${tris[i+2].toFixed(4)}`;
  for (let i = 0; i < tris.length; i += 9) {
    const [a,b,c] = [V(i), V(i+3), V(i+6)];
    for (const [p,q] of [[a,b],[b,c],[c,a]]) e.set(p+"|"+q, (e.get(p+"|"+q)||0)+1);
  }
  let bad = 0;
  for (const [k,n] of e) { const [p,q] = k.split("|"); if ((e.get(q+"|"+p)||0) !== n) bad++; }
  return bad === 0;
};

const xf = (x, y) => [x, H - y];              // mirrored, like the real transform
for (const minArea of [4, 500]) {
  const polys = maskToPolygons(m, W, H, { simplifyPx: 0, minAreaPx: minArea });
  const tris = extrude(polys, 0, 1, xf, earcut);
  console.log(`minAreaPx=${String(minArea).padStart(3)}  polys=${polys.length}` +
    `  holes=${polys.reduce((a,p)=>a+p.holes.length,0)}` +
    `  vol=${vol(tris).toFixed(0)}  closed=${closed(tris)}`);
}
// expected areas: annulus pi(80^2-45^2)=13744, island pi*18^2=1018, blob pi*6^2=113
console.log("expected solid area ~", Math.round(Math.PI*(80**2-45**2) + Math.PI*18**2 + Math.PI*6**2));

// fillHoles must swallow the ring hole but the island is not background, so area = full disc + blob
const filled = fillHoles(m, W, H);
const fp = maskToPolygons(filled, W, H, { simplifyPx: 0, minAreaPx: 4 });
console.log("after fillHoles: polys=", fp.length, " holes=", fp.reduce((a,p)=>a+p.holes.length,0),
            " vol=", vol(extrude(fp, 0, 1, xf, earcut)).toFixed(0),
            " expected ~", Math.round(Math.PI*80**2 + Math.PI*6**2));
