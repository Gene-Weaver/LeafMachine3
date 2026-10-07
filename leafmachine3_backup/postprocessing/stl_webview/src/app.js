/* ==========================================================================
   app.js -- wiring: settings -> geometry -> viewer -> STL download.

   The mask and the text go through the SAME pipeline. Text is drawn to an
   offscreen canvas at high DPI and then treated as just another binary mask,
   which is why any font works and why the text controls read like the mask
   controls (there is even a dilate, because thin strokes do not print).
   ========================================================================== */

export function initApp({ geom, viewer, fonts, sampleMask, samplePair, veins }) {
  const {
    selectMask, fillHoles, dilate, maskToPolygons, extrude,
    polyBounds, writeStlBinary, faceNormals, parseColor, pointInRing,
    prepPolys, faceTris, wallTris, sealMesh,
  } = geom;
  const { extractVeins, VEIN_DEFAULTS } = veins;

  const IDENTITY = (x, y) => [x, y];
  /** Bake a pixel->mm transform into the rings, so every polygon lives in one space. */
  const toMm = (polys, xf) => polys.map((p) => ({
    outer: p.outer.map(([x, y]) => xf(x, y)),
    holes: p.holes.map((h) => h.map(([x, y]) => xf(x, y))),
  }));

  const $ = (sel, root = document) => root.querySelector(sel);
  const earcut = window.earcut;

  /* ------------------------------------------------------------- settings */
  // Names and defaults mirror generate_stl_from_mask.py / postprocessing_settings.yaml.
  const S = {
    colors: ["white"],
    colorTolerance: 0,
    fillHoles: true,
    boundarySmoothPx: 2,
    simplifyTolerancePx: 0.3,
    minAreaPx: 4,
    lengthMm: 150,
    thicknessMm: 2,

    text: "",
    font: fonts[0].id,
    textSizeMm: 12,
    textThicknessMm: 0.2,
    textRotationDeg: 0,
    textX: 0,
    textY: 0,
    textTrackingEm: 0,
    textBoldPx: 0,
    engrave: false,

    veinOn: true,
    veinThicknessMm: 0.2,
    veinEngrave: false,
    veinX: 0,
    veinY: 0,
    veinScalePct: 100,
    veinRotationDeg: 0,
  };

  let source = null;         // { data, width, height } of the loaded mask
  let base = null;           // { polys, bounds, scale, xf, tris }
  let text = null;           // { polys, wMm, hMm } in mm, centred on the text origin
  let vein = null;           // { polys, bounds } in mm -- the accepted vein layer
  let solid = null;          // the assembled triangle soup that gets exported
  let busy = false;
  let dirty = false;

  /* The vein photo lives on its own working grid: the extraction settings are
     all in PIXELS, so previewing at a different resolution than the final run
     would make the preview a polite fiction. One grid, capped, used for both. */
  const VEIN_MAX = 1400;
  let veinPhoto = null;      // ImageData of the photo, at the working grid
  let veinDom = null;        // leaf silhouette resampled to that grid
  let veinMask = null;       // the accepted binary mask, same grid
  const V = { ...VEIN_DEFAULTS };

  /* -------------------------------------------------------------- helpers */
  /**
   * Read a number field.
   *
   * `+"" === 0` and 0 is finite, so the obvious `Number.isFinite(+v) ? +v : d`
   * turns a CLEARED field into zero rather than the default -- which is reachable
   * just by select-all-and-retype, and quietly produced a 0 x 0 mm model that the
   * page still called "Ready" and happily exported as an 84-byte empty STL.
   */
  const num = (v, d) => {
    const t = String(v ?? "").trim();
    if (t === "") return d;
    const n = Number(t);
    return Number.isFinite(n) ? n : d;
  };
  const status = (msg, kind = "") => {
    const el = $("#status");
    el.textContent = msg;
    el.className = `status ${kind}`;
  };

  /* ------------------------------------------------------------ the model */
  function buildBase() {
    if (!source) { base = null; return; }
    let m = selectMask(source, S.colors, S.colorTolerance);
    let on = 0;
    for (let i = 0; i < m.length; i++) on += m[i];
    if (!on) throw new Error(`No pixels matched ${S.colors.join(", ")} at tolerance ${S.colorTolerance}.`);
    if (S.fillHoles) m = fillHoles(m, source.width, source.height);

    // Smoothing is deliberately scoped to the LEAF OUTLINE. buildText() and
    // buildVeins() call maskToPolygons() with their own fixed tolerances and no
    // smoothing at all: a glyph's corners should stay sharp, and a vein stroke
    // is only 2-4 px wide, which is thinner than any useful smoothing radius.
    const polys = maskToPolygons(m, source.width, source.height, {
      simplifyPx: S.simplifyTolerancePx,
      minAreaPx: S.minAreaPx,
      smoothPx: S.boundarySmoothPx,
    });
    if (!polys.length) throw new Error(`Nothing survived the minimum blob area of ${S.minAreaPx} px².`);

    const b = polyBounds(polys);
    const scale = S.lengthMm / Math.max(b.w, b.h);
    const cx = (b.minx + b.maxx) / 2, cy = (b.miny + b.maxy) / 2;
    // centred on the origin and Y-flipped: image rows run down, millimetres run up
    const xf = (x, y) => [(x - cx) * scale, (cy - y) * scale];
    base = { polys: toMm(polys, xf), nParts: polys.length, bounds: b, scale, xf,
             wMm: b.w * scale, hMm: b.h * scale,
             nHoles: polys.reduce((a, p) => a + p.holes.length, 0) };
  }

  /**
   * Place the accepted vein mask in the same millimetre space as the slab.
   *
   * The photo and the mask are the same oriented crop, so the honest default is
   * a straight overlay: map vein-grid pixels into the MASK's pixel grid, then
   * through the mask's own pixel->mm transform. With scale 100% and no offset
   * that is exact, whatever either image's resolution happens to be -- which is
   * what makes "it just lines up" true rather than lucky.
   */
  function buildVeins() {
    vein = null;
    if (!veinMask || !base || !S.veinOn) return;
    const vw = veinPhoto.width, vh = veinPhoto.height;
    const polys = maskToPolygons(veinMask, vw, vh, { simplifyPx: 0.6, minAreaPx: 3 });
    if (!polys.length) return;

    const kx = source.width / vw, ky = source.height / vh;   // vein grid -> mask grid
    const s = S.veinScalePct / 100;
    const rot = (S.veinRotationDeg * Math.PI) / 180;
    const cos = Math.cos(rot), sin = Math.sin(rot);
    const xf = (x, y) => {
      const [mx, my] = base.xf(x * kx, y * ky);
      const lx = mx * s, ly = my * s;
      return [S.veinX + lx * cos - ly * sin, S.veinY + lx * sin + ly * cos];
    };
    const placed = toMm(polys, xf);
    vein = { polys: placed, nParts: placed.length, bounds: polyBounds(placed) };
  }

  /**
   * Rasterise the text, then run it through the mask pipeline.
   *
   * The canvas is sized from the metrics rather than a fixed box so the glyph
   * bounds are exact; `actualBoundingBox*` gives the inked extent, which is what
   * should be centred on the drag point -- font ascent/descent would put a word
   * with no descenders visibly off-centre.
   */
  function buildText() {
    text = null;
    const str = S.text;
    if (!str.trim() || !base) return;

    const PX = 200;                                   // render height in px; scaled to mm after
    const face = fonts.find((f) => f.id === S.font) || fonts[0];
    const cvs = document.createElement("canvas");
    let ctx = cvs.getContext("2d");
    ctx.font = `${PX}px ${face.css}`;
    const tracking = S.textTrackingEm * PX;

    const chars = Array.from(str);
    const widths = chars.map((c) => ctx.measureText(c).width);
    const totalW = widths.reduce((a, b) => a + b, 0) + tracking * Math.max(0, chars.length - 1);
    const probe = ctx.measureText(str);
    const ascent = probe.actualBoundingBoxAscent || PX * 0.8;
    const descent = probe.actualBoundingBoxDescent || PX * 0.2;
    const pad = Math.ceil(PX * 0.35) + S.textBoldPx * 2 + 4;

    cvs.width = Math.ceil(totalW) + pad * 2;
    cvs.height = Math.ceil(ascent + descent) + pad * 2;
    ctx = cvs.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cvs.width, cvs.height);
    ctx.fillStyle = "#fff";
    ctx.textBaseline = "alphabetic";
    ctx.font = `${PX}px ${face.css}`;
    // Draw glyph by glyph so letter spacing works everywhere (the `letterSpacing`
    // canvas property is still not universal).
    let pen = pad;
    for (let i = 0; i < chars.length; i++) {
      ctx.fillText(chars[i], pen, pad + ascent);
      pen += widths[i] + tracking;
    }

    const img = ctx.getImageData(0, 0, cvs.width, cvs.height);
    let tm = selectMask(img, ["white"], 128);          // anti-aliased edges -> generous tolerance
    tm = dilate(tm, cvs.width, cvs.height, S.textBoldPx);
    const polys = maskToPolygons(tm, cvs.width, cvs.height, {
      simplifyPx: 0.75,
      minAreaPx: 2,
    });
    if (!polys.length) return;

    const tb = polyBounds(polys);
    const s = S.textSizeMm / (tb.h || 1);              // cap-to-descender height -> requested mm
    const tcx = (tb.minx + tb.maxx) / 2, tcy = (tb.miny + tb.maxy) / 2;
    const rot = (S.textRotationDeg * Math.PI) / 180;
    const cos = Math.cos(rot), sin = Math.sin(rot);
    const xf = (x, y) => {
      const lx = (x - tcx) * s;
      const ly = (tcy - y) * s;                        // flip Y here too
      return [S.textX + lx * cos - ly * sin, S.textY + lx * sin + ly * cos];
    };
    text = { polys: toMm(polys, xf), wMm: tb.w * s, hMm: tb.h * s };
  }

  /**
   * Assemble the solid.
   *
   * RAISED is easy: the slab, plus the letters extruded on top of it.
   *
   * ENGRAVED is not "the same letters, moved down". Placing a solid inside the
   * slab is a no-op -- a slicer unions coincident shells, so the export would
   * print perfectly flat. A real engraving has to REMOVE material, so the slab
   * is split at the pocket floor and the letters become HOLES in the upper
   * slice; a letter's counters (the middle of an 'o') come back as solid
   * islands inside that hole, which the polygon-with-holes extruder already
   * handles. No CSG library needed, because the pocket is prismatic.
   */
  /** The optional layers, in the order they are reported. */
  const stampList = () => {
    const out = [];
    if (text) out.push({ polys: text.polys, thick: S.textThicknessMm, engrave: S.engrave, name: "lettering", slot: 1 });
    if (vein) out.push({ polys: vein.polys, thick: S.veinThicknessMm, engrave: S.veinEngrave, name: "vein layer", slot: 2 });
    return out;
  };

  const ringBox = (r) => {
    let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity;
    for (const [x, y] of r) { if (x < a) a = x; if (y < b) b = y; if (x > c) c = x; if (y > d) d = y; }
    return [a, b, c, d];
  };
  const boxHit = (bx, x, y) => x >= bx[0] && x <= bx[2] && y >= bx[1] && y <= bx[3];

  /**
   * Do two placed layers actually share ground?
   *
   * Only asked when BOTH are engraved, where overlapping pockets would put two
   * sets of walls through each other. A bounding-box test is useless here -- the
   * vein layer's box is the whole leaf, so it would fire even for a caption in a
   * bare corner -- so this samples real vertices against real rings, with a box
   * prefilter per ring to keep it cheap.
   */
  function stampsOverlap(a, b) {
    const rings = b.polys.flatMap((p) => [p.outer, ...p.holes]).map((r) => ({ r, bx: ringBox(r) }));
    const pts = a.polys.flatMap((p) => p.outer);
    const stride = Math.max(1, Math.ceil(pts.length / 400));
    for (let i = 0; i < pts.length; i += stride) {
      const [x, y] = pts[i];
      for (const { r, bx } of rings) {
        if (boxHit(bx, x, y) && pointInRing([x, y], r)) return true;
      }
    }
    return false;
  }

  /**
   * Assemble the solid.
   *
   * RAISED is easy: the slab, plus each layer extruded on top of it as its own
   * shell.
   *
   * ENGRAVED is not "the same shapes, moved down". Placing a solid inside the
   * slab is a no-op -- a slicer unions coincident shells, so the export would
   * print perfectly flat. A real engraving has to REMOVE material, so every
   * engraved layer becomes HOLES in the slab's top face plus a pocket floor, all
   * in ONE shell. A letter's counters (the middle of an 'o') come back as solid
   * islands inside that hole, which the polygon-with-holes extruder already
   * handles. No CSG library needed, because the pockets are prismatic.
   */
  function assemble() {
    if (!base) return { tris: new Float32Array(0), warn: "" };
    const T = S.thicknessMm;
    const bp = prepPolys(base.polys, IDENTITY);
    const stamps = stampList();
    const raised = stamps.filter((s) => !s.engrave);
    const cut = stamps.filter((s) => s.engrave);
    const warns = [];
    const chunks = [];

    // ---- the slab, pocketed if anything is engraved ---------------------
    if (!cut.length) {
      chunks.push({ slot: 0, tris: extrude(base.polys, 0, T, IDENTITY, earcut) });
    } else {
      // Orientation is the whole game here. prepPolys() returns every OUTER ring
      // counter-clockwise and every HOLE clockwise, and wallTris() puts the normal
      // to the RIGHT of travel. A shape arrives as an OUTER (CCW) but is used as a
      // HOLE in the top face and as the wall of a VOID, both of which need CW --
      // get that backwards and the mesh inflates instead of engraving.
      const rev = (r) => r.slice().reverse();
      const out = [];
      const pockets = [];

      for (const st of cut) {
        const floor = T - Math.min(st.thick, T * 0.9);
        let off = 0;
        for (const t of prepPolys(st.polys, IDENTITY)) {
          const host = bp.find((b) => t.outer.every((pt) => pointInRing(pt, b.outer)));
          if (host) pockets.push({ t, floor, host });
          else off++;
        }
        if (off) warns.push(`Some of the ${st.name} hangs off the leaf and was not engraved.`);
      }
      if (cut.length > 1 && stampsOverlap(cut[0], cut[1])) {
        warns.push("The engraved lettering sits on top of the engraved veins — raise one of them, "
          + "or their pocket walls will cut through each other.");
      }

      // top face: the leaf, minus its own holes, minus every pocket
      faceTris(bp.map((b) => ({
        outer: b.outer,
        holes: b.holes.concat(pockets.filter((p) => p.host === b).map((p) => rev(p.t.outer))),
      })), T, true, earcut, out);
      // counters (the middle of an "o") stay solid to full height
      faceTris(pockets.flatMap((p) => p.t.holes.map((h) => ({ outer: rev(h), holes: [] }))),
        T, true, earcut, out);
      // bottom face and the outside walls span the whole thickness
      faceTris(bp, 0, false, earcut, out);
      for (const b of bp) wallTris([b.outer, ...b.holes], 0, T, out);
      // the pockets: walls down from the top face, then a floor that closes them
      for (const p of pockets) {
        wallTris([rev(p.t.outer)], p.floor, T, out);                      // void wall: CW
        for (const h of p.t.holes) wallTris([rev(h)], p.floor, T, out);   // counter wall: CCW
        // floor faces UP (material below), so outer stays CCW and counters stay CW
        faceTris([{ outer: p.t.outer, holes: p.t.holes }], p.floor, true, earcut, out);
      }
      chunks.push({ slot: 0, tris: sealMesh(new Float32Array(out)) });
    }

    // ---- raised layers: their own shells resting on the face -------------
    for (const st of raised) {
      chunks.push({ slot: st.slot, tris: extrude(st.polys, T, T + st.thick, IDENTITY, earcut) });
      // Anything hanging off the shape is a floating island: nothing under it,
      // so it cannot print. Same containment test as the engrave path.
      const loose = prepPolys(st.polys, IDENTITY)
        .filter((t) => !bp.some((b) => t.outer.some((pt) => pointInRing(pt, b.outer)))).length;
      if (loose) warns.push(`Some of the ${st.name} sits off the shape and would print as loose pieces.`);
    }

    let n = 0;
    for (const c of chunks) n += c.tris.length;
    const all = new Float32Array(n);
    let at = 0;
    for (const c of chunks) { all.set(c.tris, at); at += c.tris.length; }
    return { tris: all, warn: warns.join(" "), chunks };
  }

  function rebuild({ refit = false } = {}) {
    if (busy) { dirty = true; return; }
    busy = true;
    try {
      buildBase();
      buildText();
      buildVeins();
      if (!base) {
        for (const s of [0, 1, 2]) viewer.setMesh(s, null, null);
        status("Load a mask image to begin.", "");
        return;
      }
      const built = assemble();
      solid = built.tris;
      // The viewer tints each layer, but the SOLID is what gets exported: an
      // engraved layer is part of the slab's own interlocking mesh, so it comes
      // back in slot 0 and only the RAISED layers get their own colour.
      const COLOR = { 0: [0.42, 0.72, 0.46], 1: [0.98, 0.86, 0.42], 2: [0.36, 0.78, 0.97] };
      for (const slot of [0, 1, 2]) {
        const c = built.chunks.find((k) => k.slot === slot);
        if (c) viewer.setMesh(slot, c.tris, faceNormals(c.tris), COLOR[slot]);
        else viewer.setMesh(slot, null, null);
      }
      if (refit) viewer.frame(Math.max(base.wMm, base.hMm) / 2, [0, 0, S.thicknessMm / 2]);

      const nTri = solid.length / 9;
      if (!nTri) throw new Error("Nothing to export — check the size and thickness settings.");
      const parts = base.nParts;
      const holes = base.nHoles;
      $("#facts").innerHTML =
        `<b>${base.wMm.toFixed(1)} × ${base.hMm.toFixed(1)} × ${S.thicknessMm}</b> mm`
        + ` · ${parts} part${parts === 1 ? "" : "s"}`
        + (holes ? ` · ${holes} hole${holes === 1 ? "" : "s"}` : "")
        + (vein ? ` · ${vein.nParts.toLocaleString()} vein strokes` : "")
        + ` · ${nTri.toLocaleString()} triangles`
        + ` · ${(base.scale).toFixed(4)} mm/px`;
      if (built.warn) status(built.warn, "warn");
      else if (text && vein) status("Ready — drag the text to place it, alt-drag to move the veins.", "ok");
      else if (vein) status("Ready — alt-drag the model to move the vein layer.", "ok");
      else status(text ? "Ready — drag the text on the model to move it." : "Ready.", "ok");
      $("#dl").disabled = false;
    } catch (err) {
      status(String(err.message || err), "bad");
      $("#dl").disabled = true;
    } finally {
      busy = false;
      if (dirty) { dirty = false; rebuild(); }
      viewer.render();
    }
  }

  /* ---------------------------------------------------------------- input */
  function loadImageData(img) {
    const c = document.createElement("canvas");
    c.width = img.naturalWidth || img.width;
    c.height = img.naturalHeight || img.height;
    const cx = c.getContext("2d", { willReadFrequently: true });
    cx.drawImage(img, 0, 0);
    source = cx.getImageData(0, 0, c.width, c.height);
    $("#imginfo").textContent = `${c.width} × ${c.height} px`;
    rebuild({ refit: true });
  }

  function loadFromBlobOrUrl(src) {
    const img = new Image();
    img.onload = () => loadImageData(img);
    img.onerror = () => status("That file could not be read as an image.", "bad");
    img.src = src;
  }

  /**
   * PNG only, decided by the file's MAGIC BYTES rather than its name or its
   * reported MIME type -- both are attacker-controlled and neither says what
   * the decoder will actually do with the bytes.
   */
  const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  const JPEG_MAGIC = [0xff, 0xd8, 0xff];
  function sniff(file) {
    return new Promise((resolve) => {
      if (!file) return resolve(null);
      const head = new FileReader();
      head.onload = () => {
        const b = new Uint8Array(head.result);
        if (b.length >= 8 && PNG_MAGIC.every((v, i) => b[i] === v)) return resolve("png");
        if (b.length >= 3 && JPEG_MAGIC.every((v, i) => b[i] === v)) return resolve("jpeg");
        resolve(null);
      };
      head.onerror = () => resolve(null);
      head.readAsArrayBuffer(file.slice(0, 8));
    });
  }

  /** Read an accepted file as a data URI. Never touches the network either way. */
  const readAsDataUrl = (file) => new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = () => reject(new Error("That file could not be read."));
    fr.readAsDataURL(file);
  });

  async function takeFile(file) {
    if (!file) return;
    if ((await sniff(file)) !== "png") {
      status(`“${file.name}” is not a PNG. This page reads PNG masks only.`, "bad");
      return;
    }
    readAsDataUrl(file).then(loadFromBlobOrUrl, (e) => status(e.message, "bad"));
  }

  function wireFile() {
    const input = $("#file");
    input.addEventListener("change", () => {
      takeFile(input.files && input.files[0]);
      input.value = "";                       // re-selecting the same file must re-fire
    });
    const drop = $("#drop");
    ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => {
      ev.preventDefault(); drop.classList.add("over");
    }));
    ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => {
      ev.preventDefault(); drop.classList.remove("over");
    }));
    drop.addEventListener("drop", (ev) => {
      ev.preventDefault();
      takeFile(ev.dataTransfer.files && ev.dataTransfer.files[0]);
    });
    $("#sample").addEventListener("click", () => loadFromBlobOrUrl(sampleMask));
  }

  /* ------------------------------------------------------------ vein layer */

  /**
   * Downscale the photo onto the working grid.
   *
   * Every vein setting is measured in PIXELS, so previewing at one resolution
   * and running at another would make the preview a polite fiction -- a 21 px
   * background window is a different filter on a half-size image. There is
   * therefore exactly one grid, capped, and the preview IS the final run.
   */
  function toWorkGrid(img) {
    const iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height;
    const k = Math.min(1, VEIN_MAX / Math.max(iw, ih));
    const w = Math.max(1, Math.round(iw * k)), h = Math.max(1, Math.round(ih * k));
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const cx = c.getContext("2d", { willReadFrequently: true });
    cx.drawImage(img, 0, 0, w, h);
    return { data: cx.getImageData(0, 0, w, h), scaled: k < 1, srcW: iw, srcH: ih };
  }

  /**
   * The leaf silhouette, resampled onto the vein grid.
   *
   * Taken from the MASK rather than thresholded out of the photo: the mask is
   * ground truth for where the leaf is, so a vein can never escape the shape it
   * decorates. Deliberately the UNFILLED selection — veins should not bridge an
   * insect hole even when "fill internal holes" closes it in the slab.
   */
  function veinDomainFor(vw, vh) {
    if (!source) return null;
    const m = selectMask(source, S.colors, S.colorTolerance);
    const sw = source.width, sh = source.height;
    const out = new Uint8ClampedArray(vw * vh);
    for (let y = 0; y < vh; y++) {
      const sy = Math.min(sh - 1, Math.floor(((y + 0.5) * sh) / vh));
      for (let x = 0; x < vw; x++) {
        const sx = Math.min(sw - 1, Math.floor(((x + 0.5) * sw) / vw));
        out[y * vw + x] = m[sy * sw + sx];
      }
    }
    return out;
  }

  function loadVeinPhoto(src) {
    if (!source) {
      status("Load the mask first — the vein layer is aligned to it.", "bad");
      return;
    }
    const img = new Image();
    img.onload = () => {
      const g = toWorkGrid(img);
      veinPhoto = g.data;
      veinDom = veinDomainFor(veinPhoto.width, veinPhoto.height);
      const same = g.srcW === source.width && g.srcH === source.height;
      $("#veininfo").textContent = `${g.srcW} × ${g.srcH} px · `
        + (same ? "matches the mask, aligned 1:1" : "different size from the mask — it will be stretched to fit");
      $("#veinwork").textContent = `${veinPhoto.width} × ${veinPhoto.height} px`
        + (g.scaled ? " (downscaled for speed)" : "");
      openVeinModal();
    };
    img.onerror = () => status("That file could not be read as an image.", "bad");
    img.src = src;
  }

  async function takeVeinFile(file) {
    if (!file) return;
    const kind = await sniff(file);
    // PNG *or* JPEG here, unlike the mask: LeafMachine3 writes Lamina_RGB as
    // JPEG, so PNG-only would reject the exact file this feature asks for. Still
    // decided by magic bytes, so an SVG or an HTML file named .png is refused
    // just the same, and nothing is uploaded either way.
    if (kind !== "png" && kind !== "jpeg") {
      status(`“${file.name}” is not a PNG or a JPEG.`, "bad");
      return;
    }
    readAsDataUrl(file).then(loadVeinPhoto, (e) => status(e.message, "bad"));
  }

  /* ---- the extraction modal ---- */
  let vBusy = false, vDirty = false, vResult = null, vMode = "overlay";
  const vOff = document.createElement("canvas");

  function paintVeins() {
    if (!veinPhoto || !vResult) return;
    const w = veinPhoto.width, h = veinPhoto.height;
    vOff.width = w; vOff.height = h;
    const octx = vOff.getContext("2d");
    const out = octx.createImageData(w, h);
    const src = veinPhoto.data, m = vResult.mask, e = vResult.preview, o = out.data;
    for (let i = 0, p = 0; i < m.length; i++, p += 4) {
      if (vMode === "mask") {
        o[p] = o[p + 1] = o[p + 2] = m[i] ? 255 : 0;
      } else if (vMode === "enhanced") {
        o[p] = o[p + 1] = o[p + 2] = e[i];
      } else if (vMode === "photo") {
        o[p] = src[p]; o[p + 1] = src[p + 1]; o[p + 2] = src[p + 2];
      } else if (m[i]) {                       // overlay: veins in the accent orange
        o[p] = 251; o[p + 1] = 146; o[p + 2] = 60;
      } else {                                 // the photo, dimmed, so the mask reads
        o[p] = src[p] * 0.6; o[p + 1] = src[p + 1] * 0.6; o[p + 2] = src[p + 2] * 0.6;
      }
      o[p + 3] = 255;
    }
    octx.putImageData(out, 0, 0);

    const cv = $("#veincanvas");
    const box = cv.parentElement.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const k = Math.min(box.width / w, box.height / h) || 1;
    cv.width = Math.max(1, Math.round(w * k * dpr));
    cv.height = Math.max(1, Math.round(h * k * dpr));
    cv.style.width = `${Math.round(w * k)}px`;
    cv.style.height = `${Math.round(h * k)}px`;
    cv.getContext("2d").drawImage(vOff, 0, 0, cv.width, cv.height);
  }

  function runVeinPreview() {
    if (!veinPhoto || $("#veinmodal").hidden) return;
    if (vBusy) { vDirty = true; return; }
    vBusy = true;
    $("#veinstat").textContent = "working…";
    // Yield twice so the label actually paints before a few hundred ms of
    // synchronous filtering: rAF alone still runs before the next paint.
    requestAnimationFrame(() => setTimeout(() => {
      try {
        const t0 = performance.now();
        vResult = extractVeins(veinPhoto, veinDom, V);
        const ms = Math.round(performance.now() - t0);
        paintVeins();
        $("#veinstat").textContent =
          `${(vResult.coverage * 100).toFixed(2)}% vein · ${ms} ms`;
      } catch (err) {
        $("#veinstat").textContent = String(err.message || err);
      } finally {
        vBusy = false;
        if (vDirty) { vDirty = false; runVeinPreview(); }
      }
    }, 0));
  }

  /** Show only the controls that the current mode actually uses. */
  function veinModeVisibility() {
    const show = (sel, on) => { const el = $(sel); if (el) el.hidden = !on; };
    show("#vridgelen-row", V.ridge);
    show("#vridgeang-row", V.ridge);
    show("#vthr-row", V.mode === "manual");
    show("#vadaptr-row", V.mode === "adaptive");
    show("#vadapto-row", V.mode === "adaptive");
    show("#votsu-note", V.mode === "otsu");
  }

  function syncVeinControls() {
    const put = (sel, val) => {
      const el = $(sel);
      if (!el) return;
      if (el.type === "checkbox") el.checked = !!val;
      else el.value = val;
      const o = $(`${sel}-out`);
      if (o && el.type !== "checkbox") o.textContent = el.value;
    };
    put("#vchannel", V.channel);
    put("#vinvert", V.invert);
    put("#vbg", V.bgRadius);
    put("#vgain", V.gain);
    put("#vridge", V.ridge);
    put("#vridgelen", V.ridgeLen);
    put("#vridgeang", V.ridgeAngles);
    put("#vmode", V.mode);
    put("#vthr", V.threshold);
    put("#vadaptr", V.adaptiveRadius);
    put("#vadapto", V.adaptiveOffset);
    put("#vclose", V.closeRadius);
    put("#vminarea", V.minArea);
    put("#vthicken", V.thicken);
    put("#vrim", V.rimInset);
    veinModeVisibility();
  }

  function openVeinModal() {
    if (!veinPhoto) return;
    $("#veinmodal").hidden = false;
    syncVeinControls();
    runVeinPreview();
  }
  const closeVeinModal = () => { $("#veinmodal").hidden = true; };

  function bindV(id, key) {
    const el = $(id);
    if (!el) return;
    const set = () => {
      if (el.type === "checkbox") V[key] = el.checked;
      else if (el.tagName === "SELECT") {
        V[key] = Number.isFinite(+el.value) && typeof VEIN_DEFAULTS[key] === "number"
          ? +el.value : el.value;
      } else V[key] = num(el.value, VEIN_DEFAULTS[key]);
      const o = $(`${id}-out`);
      if (o && el.type !== "checkbox") o.textContent = el.value;
      veinModeVisibility();
      runVeinPreview();
    };
    el.addEventListener("input", set);
    el.addEventListener("change", set);
  }

  function clearVeins() {
    veinPhoto = veinDom = veinMask = vResult = null;
    $("#veinctl").hidden = true;
    $("#veindrop").hidden = false;
    $("#veinstate").textContent = "optional";
    $("#veininfo").textContent = "";
    rebuild();
  }

  function wireVeins() {
    const input = $("#veinfile");
    input.addEventListener("change", () => {
      takeVeinFile(input.files && input.files[0]);
      input.value = "";
    });
    const drop = $("#veindrop");
    ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => {
      ev.preventDefault(); drop.classList.add("over");
    }));
    ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => {
      ev.preventDefault(); drop.classList.remove("over");
    }));
    drop.addEventListener("drop", (ev) => {
      ev.preventDefault();
      takeVeinFile(ev.dataTransfer.files && ev.dataTransfer.files[0]);
    });

    // The sample photo only lines up with the sample MASK, so the pair loads
    // together — offering a photo that does not match the loaded mask would
    // demonstrate the feature by breaking its central promise.
    $("#veinsample").addEventListener("click", () => {
      const img = new Image();
      img.onload = () => { loadImageData(img); loadVeinPhoto(samplePair.rgb); };
      img.onerror = () => status("The sample pair could not be loaded.", "bad");
      img.src = samplePair.mask;
    });

    $("#veinedit").addEventListener("click", openVeinModal);
    $("#veinclear").addEventListener("click", clearVeins);
    $("#veinclose").addEventListener("click", closeVeinModal);
    $("#veincancel").addEventListener("click", closeVeinModal);
    $("#veinmodal").addEventListener("click", (ev) => {
      if (ev.target === ev.currentTarget) closeVeinModal();
    });
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && !$("#veinmodal").hidden) closeVeinModal();
    });
    $("#veindefaults").addEventListener("click", () => {
      Object.assign(V, VEIN_DEFAULTS);
      syncVeinControls();
      runVeinPreview();
    });
    $("#veinuse").addEventListener("click", () => {
      if (!vResult) return;
      veinMask = vResult.mask;
      $("#veinctl").hidden = false;
      $("#veindrop").hidden = true;
      $("#veinstate").textContent = "in use";
      $("#veinfold").open = true;
      closeVeinModal();
      rebuild();
    });
    $("#veinreset").addEventListener("click", () => {
      S.veinX = 0; S.veinY = 0; S.veinScalePct = 100; S.veinRotationDeg = 0;
      $("#veinX").value = 0; $("#veinY").value = 0;
      $("#veinscale").value = 100; $("#veinscale-out").textContent = "100";
      $("#veinrot").value = 0; $("#veinrot-out").textContent = "0";
      rebuild();
    });

    $(".vmodes").addEventListener("click", (ev) => {
      const b = ev.target.closest("button[data-mode]");
      if (!b) return;
      vMode = b.dataset.mode;
      $(".vmodes").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      paintVeins();
    });

    bindV("#vchannel", "channel");
    bindV("#vinvert", "invert");
    bindV("#vbg", "bgRadius");
    bindV("#vgain", "gain");
    bindV("#vridge", "ridge");
    bindV("#vridgelen", "ridgeLen");
    bindV("#vridgeang", "ridgeAngles");
    bindV("#vmode", "mode");
    bindV("#vthr", "threshold");
    bindV("#vadaptr", "adaptiveRadius");
    bindV("#vadapto", "adaptiveOffset");
    bindV("#vclose", "closeRadius");
    bindV("#vminarea", "minArea");
    bindV("#vthicken", "thicken");
    bindV("#vrim", "rimInset");
  }

  /* -------------------------------------------------------------- binding */
  function bind(id, key, { parse = num, fallback = 0, refit = false } = {}) {
    const el = $(id);
    if (!el) return;
    const write = () => {
      const out = $(`${id}-out`);
      if (out) out.textContent = el.type === "checkbox" ? "" : el.value;
    };
    const set = (commit) => {
      let v = el.type === "checkbox" ? el.checked : parse(el.value, fallback);
      // The min/max on the inputs are only advisory: this page never runs form
      // validation, so without this a negative thickness would export an
      // inside-out solid and the browser would not say a word.
      if (typeof v === "number") {
        if (el.min !== "" && el.min != null && Number.isFinite(+el.min)) v = Math.max(+el.min, v);
        if (el.max !== "" && el.max != null && Number.isFinite(+el.max)) v = Math.min(+el.max, v);
      }
      // On commit (blur / Enter), show what is actually being used -- a field
      // reading -5 while the model is built at 0.05 is just a lie.
      if (commit && typeof v === "number" && el.value.trim() !== "" && +el.value !== v) {
        el.value = String(v);
      }
      S[key] = v;
      write();
      rebuild({ refit });
    };
    el.addEventListener("input", () => set(false));
    el.addEventListener("change", () => set(true));
    write();
  }

  function wireColors() {
    const list = $("#colorlist");
    const draw = () => {
      list.innerHTML = "";
      S.colors.forEach((c, i) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        const sw = document.createElement("i");
        // Never hand a raw user string to a style property: `url(https://…)` in a
        // background would make the browser fetch it, and this page must make no
        // requests at all. Re-serialise from the parsed RGB instead.
        let swatch = "#000";
        try {
          const [r, g, b] = parseColor(c);
          swatch = `rgb(${r},${g},${b})`;
        } catch { swatch = "repeating-linear-gradient(45deg,#f87171 0 4px,#2a2b30 4px 8px)"; }
        sw.style.background = swatch;
        chip.appendChild(sw);
        chip.appendChild(document.createTextNode(c));
        if (S.colors.length > 1) {
          const x = document.createElement("b");
          x.textContent = "✕";
          x.title = "remove";
          x.onclick = () => { S.colors.splice(i, 1); draw(); rebuild(); };
          chip.appendChild(x);
        }
        list.appendChild(chip);
      });
    };
    $("#addcolor").addEventListener("click", () => {
      const v = $("#newcolor").value.trim();
      if (!v) return;
      S.colors.push(v);
      draw();
      rebuild();
    });
    $("#colorpick").addEventListener("change", (e) => {
      S.colors.push(e.target.value);
      draw();
      rebuild();
    });
    draw();
  }

  /* ----------------------------------------------------------- 3D pointer */
  function wirePointer() {
    const cvs = $("#view");
    let mode = null, last = null, grab = null;

    const overText = (ev) => {
      if (!text || !base) return false;
      const p = viewer.pickOnPlane(ev.clientX, ev.clientY, S.thicknessMm);
      if (!p) return false;
      // into the text's own frame, so a rotated label still hit-tests correctly
      const rot = (-S.textRotationDeg * Math.PI) / 180;
      const dx = p[0] - S.textX, dy = p[1] - S.textY;
      const lx = dx * Math.cos(rot) - dy * Math.sin(rot);
      const ly = dx * Math.sin(rot) + dy * Math.cos(rot);
      return Math.abs(lx) <= text.wMm / 2 + 1 && Math.abs(ly) <= text.hMm / 2 + 1;
    };

    cvs.addEventListener("pointermove", (ev) => {
      if (!mode) cvs.style.cursor = (overText(ev) || (ev.altKey && vein)) ? "move" : "grab";
    });

    cvs.addEventListener("pointerdown", (ev) => {
      cvs.setPointerCapture(ev.pointerId);
      last = [ev.clientX, ev.clientY];
      // The vein layer covers the whole leaf, so a plain drag over it has to stay
      // an orbit — otherwise the model becomes impossible to turn. Alt claims it.
      if (ev.button === 0 && ev.altKey && vein) {
        mode = "vein";
        const p = viewer.pickOnPlane(ev.clientX, ev.clientY, S.thicknessMm);
        grab = p ? [S.veinX - p[0], S.veinY - p[1]] : [0, 0];
        cvs.style.cursor = "move";
      } else if (ev.button === 0 && !ev.shiftKey && overText(ev)) {
        mode = "text";
        const p = viewer.pickOnPlane(ev.clientX, ev.clientY, S.thicknessMm);
        grab = [S.textX - p[0], S.textY - p[1]];
        cvs.style.cursor = "move";
      } else if (ev.button === 1 || ev.shiftKey) {
        mode = "pan";
      } else {
        mode = "orbit";
        cvs.style.cursor = "grabbing";
      }
    });

    cvs.addEventListener("pointermove", (ev) => {
      if (!mode) return;
      const dx = ev.clientX - last[0], dy = ev.clientY - last[1];
      last = [ev.clientX, ev.clientY];
      if (mode === "orbit") {
        viewer.cam.yaw -= dx * 0.008;
        viewer.cam.pitch = Math.max(-1.5, Math.min(1.5533, viewer.cam.pitch + dy * 0.008));
      } else if (mode === "pan") {
        const k = viewer.cam.dist * 0.0016;
        viewer.cam.target[0] -= dx * k * Math.sin(viewer.cam.yaw) * -1;
        viewer.cam.target[1] += dx * k * Math.cos(viewer.cam.yaw) * -1;
        viewer.cam.target[2] += dy * k;
      } else if (mode === "text" || mode === "vein") {
        const p = viewer.pickOnPlane(ev.clientX, ev.clientY, S.thicknessMm);
        if (p) {
          const kx = mode === "text" ? "textX" : "veinX";
          const ky = mode === "text" ? "textY" : "veinY";
          S[kx] = Math.round((p[0] + grab[0]) * 100) / 100;
          S[ky] = Math.round((p[1] + grab[1]) * 100) / 100;
          $(`#${kx}`).value = S[kx];
          $(`#${ky}`).value = S[ky];
          rebuild();
          return;
        }
      }
      viewer.render();
    });

    const end = (ev) => {
      mode = null; grab = null;
      cvs.style.cursor = "grab";
      if (ev.pointerId != null && cvs.hasPointerCapture(ev.pointerId)) cvs.releasePointerCapture(ev.pointerId);
    };
    cvs.addEventListener("pointerup", end);
    cvs.addEventListener("pointercancel", end);

    cvs.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      viewer.cam.dist = Math.max(5, Math.min(6000, viewer.cam.dist * (1 + Math.sign(ev.deltaY) * 0.12)));
      viewer.render();
    }, { passive: false });

    $("#v-reset").addEventListener("click", () => { viewer.resetView(); viewer.render(); });
    $("#v-top").addEventListener("click", () => { viewer.topView(); viewer.render(); });
    $("#v-fit").addEventListener("click", () => {
      if (base) viewer.frame(Math.max(base.wMm, base.hMm) / 2, [0, 0, S.thicknessMm / 2]);
      viewer.render();
    });
  }

  /* -------------------------------------------------------------- download */
  function wireDownload() {
    $("#dl").addEventListener("click", () => {
      if (!base || !solid) return;
      const all = solid;
      const blob = new Blob([writeStlBinary(all)], { type: "model/stl" });
      const name = ($("#fname").value.trim() || "leafmachine_model").replace(/[^\w.-]+/g, "_");
      const url = URL.createObjectURL(blob);
      const a2 = document.createElement("a");
      a2.href = url;
      a2.download = `${name}.stl`;
      a2.click();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
      status(`Saved ${name}.stl — ${(all.length / 9).toLocaleString()} triangles.`, "ok");
    });
  }

  /* ----------------------------------------------------------------- boot */
  function wireFonts() {
    const sel = $("#font");
    for (const f of fonts) {
      const o = document.createElement("option");
      o.value = f.id;
      o.textContent = f.label;
      o.style.fontFamily = f.css;
      sel.appendChild(o);
    }
    sel.value = S.font;
  }

  // Explanations are off by default and toggled as a block, exactly like the
  // desktop app's Settings tab -- they are worth ~40px each on a 28px row.
  $("#notes").addEventListener("click", (e) => {
    const on = $("#controls").classList.toggle("notes-off");
    $("#veinpanel").classList.toggle("notes-off", on);
    e.currentTarget.classList.toggle("on", !on);
  });

  wireFonts();
  wireFile();
  wireColors();
  wirePointer();
  wireVeins();
  wireDownload();

  bind("#tolerance", "colorTolerance", { fallback: 0 });
  bind("#fillholes", "fillHoles");
  bind("#smooth", "boundarySmoothPx", { fallback: 2 });
  bind("#simplify", "simplifyTolerancePx", { fallback: 0.3 });
  bind("#minarea", "minAreaPx", { fallback: 4 });
  bind("#length", "lengthMm", { fallback: 150, refit: true });
  bind("#thickness", "thicknessMm", { fallback: 2, refit: true });
  bind("#text", "text", { parse: (v) => v });
  bind("#font", "font", { parse: (v) => v });
  bind("#textsize", "textSizeMm", { fallback: 12 });
  bind("#textthick", "textThicknessMm", { fallback: 0.2 });
  bind("#textrot", "textRotationDeg", { fallback: 0 });
  bind("#textX", "textX", { fallback: 0 });
  bind("#textY", "textY", { fallback: 0 });
  bind("#tracking", "textTrackingEm", { fallback: 0 });
  bind("#bold", "textBoldPx", { fallback: 0 });
  bind("#engrave", "engrave");
  bind("#veinon", "veinOn");
  bind("#veinthick", "veinThicknessMm", { fallback: 0.2 });
  bind("#veinengrave", "veinEngrave");
  bind("#veinX", "veinX", { fallback: 0 });
  bind("#veinY", "veinY", { fallback: 0 });
  bind("#veinscale", "veinScalePct", { fallback: 100 });
  bind("#veinrot", "veinRotationDeg", { fallback: 0 });

  window.addEventListener("resize", () => viewer.render());
  status("Load a mask image, or press “Use a sample leaf”.", "");
  viewer.render();

  // Fonts arrive asynchronously; text measured before they land would be laid
  // out in the fallback face and silently change size a moment later.
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => { if (S.text) rebuild(); });
  }
}
