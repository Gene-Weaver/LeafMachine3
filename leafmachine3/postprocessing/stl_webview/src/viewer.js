/* ==========================================================================
   viewer.js -- a small WebGL viewer: orbit, zoom, pan, flat-shaded triangles.

   Deliberately hand-rolled rather than pulling in three.js. The page has to be
   ONE self-contained file, and three.js would be ~600 KB of the budget to draw
   two flat-shaded meshes and cast one ray at one plane. Everything here is
   about 250 lines, and the ray we need for dragging text hits a PLANE (the top
   of the slab), not the mesh -- so no BVH, no mesh raycaster.
   ========================================================================== */

/* ------------------------------------------------------------------- math */
function mat4() { return new Float32Array(16); }
function identity(m) {
  m.fill(0); m[0] = m[5] = m[10] = m[15] = 1; return m;
}
function multiply(out, a, b) {
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1]
        + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}
function perspective(out, fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  out.fill(0);
  out[0] = f / aspect; out[5] = f; out[11] = -1;
  out[10] = (far + near) / (near - far);
  out[14] = (2 * far * near) / (near - far);
  return out;
}
function lookAt(out, eye, center, up) {
  const z = norm3([eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]]);
  const x = norm3(cross3(up, z));
  const y = cross3(z, x);
  out[0] = x[0]; out[1] = y[0]; out[2] = z[0]; out[3] = 0;
  out[4] = x[1]; out[5] = y[1]; out[6] = z[1]; out[7] = 0;
  out[8] = x[2]; out[9] = y[2]; out[10] = z[2]; out[11] = 0;
  out[12] = -dot3(x, eye); out[13] = -dot3(y, eye); out[14] = -dot3(z, eye); out[15] = 1;
  return out;
}
function invert(out, m) {
  const inv = new Float64Array(16);
  inv[0] = m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
  inv[4] = -m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
  inv[8] = m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9];
  inv[12] = -m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9];
  inv[1] = -m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
  inv[5] = m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
  inv[9] = -m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9];
  inv[13] = m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9];
  inv[2] = m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6];
  inv[6] = -m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6];
  inv[10] = m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5];
  inv[14] = -m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5];
  inv[3] = -m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6];
  inv[7] = m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6];
  inv[11] = -m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5];
  inv[15] = m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5];
  let det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
  if (!det) return identity(out);
  det = 1 / det;
  for (let i = 0; i < 16; i++) out[i] = inv[i] * det;
  return out;
}
const cross3 = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const dot3 = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
function norm3(v) { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0]/l, v[1]/l, v[2]/l]; }

/* ---------------------------------------------------------------- shaders */
const VS = `
attribute vec3 aPos;
attribute vec3 aNrm;
uniform mat4 uMVP;
varying vec3 vN;
varying vec3 vP;
void main() {
  vN = aNrm;
  vP = aPos;
  gl_Position = uMVP * vec4(aPos, 1.0);
}`;

const FS = `
precision highp float;
varying vec3 vN;
varying vec3 vP;
uniform vec3 uColor;
uniform vec3 uEye;
void main() {
  vec3 n = normalize(vN);
  vec3 v = normalize(uEye - vP);
  if (dot(n, v) < 0.0) n = -n;                  // two-sided: never a black facet
  vec3 key = normalize(vec3(0.45, 0.35, 0.82));
  float d = max(dot(n, key), 0.0);
  float hemi = 0.5 + 0.5 * n.z;                 // sky above, ground below
  vec3 col = uColor * (0.34 + 0.52 * d + 0.28 * hemi);
  float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.22;
  gl_FragColor = vec4(col + rim, 1.0);
}`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error("shader: " + gl.getShaderInfoLog(sh));
  }
  return sh;
}

/* ---------------------------------------------------------------- viewer */
export function createViewer(canvas) {
  const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
  if (!gl) throw new Error("WebGL is not available in this browser.");

  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error("link: " + gl.getProgramInfoLog(prog));
  }
  gl.useProgram(prog);
  const loc = {
    aPos: gl.getAttribLocation(prog, "aPos"),
    aNrm: gl.getAttribLocation(prog, "aNrm"),
    uMVP: gl.getUniformLocation(prog, "uMVP"),
    uColor: gl.getUniformLocation(prog, "uColor"),
    uEye: gl.getUniformLocation(prog, "uEye"),
  };
  gl.enable(gl.DEPTH_TEST);

  const meshes = [];                    // { pos, nrm, count, color, visible }
  const cam = { yaw: -0.6, pitch: 0.95, dist: 300, target: [0, 0, 0] };
  const mvp = mat4(), proj = mat4(), view = mat4(), invMVP = mat4();
  let eye = [0, 0, 1];

  function setMesh(slot, tris, normals, color) {
    let m = meshes[slot];
    if (!m) {
      m = meshes[slot] = { pos: gl.createBuffer(), nrm: gl.createBuffer(), count: 0, color, visible: true };
    }
    m.color = color || m.color;
    m.count = tris ? tris.length / 3 : 0;
    if (!tris || !tris.length) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, m.pos);
    gl.bufferData(gl.ARRAY_BUFFER, tris, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.nrm);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
  }

  function frame(radius, center) {
    cam.target = center || [0, 0, 0];
    cam.dist = Math.max(radius * 2.4, 10);
  }

  function updateCamera() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    eye = [
      cam.target[0] + cam.dist * cp * Math.cos(cam.yaw),
      cam.target[1] + cam.dist * cp * Math.sin(cam.yaw),
      cam.target[2] + cam.dist * sp,
    ];
    perspective(proj, 0.7, w / h, 0.5, 20000);
    lookAt(view, eye, cam.target, [0, 0, 1]);
    multiply(mvp, proj, view);
    invert(invMVP, mvp);
  }

  function render() {
    updateCamera();
    gl.clearColor(0.055, 0.06, 0.07, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.uniformMatrix4fv(loc.uMVP, false, mvp);
    gl.uniform3fv(loc.uEye, new Float32Array(eye));
    for (const m of meshes) {
      if (!m || !m.count || !m.visible) continue;
      gl.uniform3fv(loc.uColor, new Float32Array(m.color));
      gl.bindBuffer(gl.ARRAY_BUFFER, m.pos);
      gl.enableVertexAttribArray(loc.aPos);
      gl.vertexAttribPointer(loc.aPos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, m.nrm);
      gl.enableVertexAttribArray(loc.aNrm);
      gl.vertexAttribPointer(loc.aNrm, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLES, 0, m.count);
    }
  }

  /** Screen point -> the (x, y) where its ray crosses the plane z = planeZ. */
  function pickOnPlane(clientX, clientY, planeZ) {
    const r = canvas.getBoundingClientRect();
    const ndcX = ((clientX - r.left) / r.width) * 2 - 1;
    const ndcY = 1 - ((clientY - r.top) / r.height) * 2;
    const un = (z) => {
      const x = invMVP[0]*ndcX + invMVP[4]*ndcY + invMVP[8]*z + invMVP[12];
      const y = invMVP[1]*ndcX + invMVP[5]*ndcY + invMVP[9]*z + invMVP[13];
      const zz = invMVP[2]*ndcX + invMVP[6]*ndcY + invMVP[10]*z + invMVP[14];
      const w = invMVP[3]*ndcX + invMVP[7]*ndcY + invMVP[11]*z + invMVP[15];
      return [x / w, y / w, zz / w];
    };
    const a = un(-1), b = un(1);
    const dz = b[2] - a[2];
    if (Math.abs(dz) < 1e-9) return null;              // ray parallel to the plane
    const t = (planeZ - a[2]) / dz;
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }

  return {
    gl, cam, setMesh, frame, render, pickOnPlane,
    setVisible(slot, on) { if (meshes[slot]) meshes[slot].visible = on; },
    resetView() { cam.yaw = -0.6; cam.pitch = 0.95; },
    topView() { cam.yaw = -Math.PI / 2; cam.pitch = 1.5533; },
  };
}
