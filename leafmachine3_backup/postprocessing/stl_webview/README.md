# stl_webview — `generate_3d_file.html`

A single self-contained page for **leafmachine.org**: drop in a PNG mask, get a printable `.stl`.
It is a browser port of `leafmachine3/postprocessing/generate_stl_from_mask.py`, plus a 3D preview
and a lettering layer.

```
generate_3d_file.html      <- THE DELIVERABLE. Copy this one file to the website.
_build.py                  <- assembles it from src/ + assets/. Edit sources, never the HTML.
src/geom.js                <- mask -> polygons -> extruded solid -> binary STL
src/veins.js               <- leaf photo -> binary vein mask (top-hat + morphology)
src/viewer.js              <- ~250-line WebGL viewer (orbit / zoom / pan)
src/app.js                 <- settings, text rasterising, drag placement, export
src/template.html, style.css
src/test_*.mjs             <- node harnesses (`node src/test_geom.mjs`)
assets/                    <- earcut (ISC), 10 WOFF2 fonts, sample mask + sample pair
```

Rebuild after editing anything in `src/` or `assets/`:

```bash
python3 _build.py     # -> generate_3d_file.html  (~357 KB)
```

## Deploying

Copy `generate_3d_file.html` anywhere that serves static files. There is nothing else to upload, no
build step on the server, and no runtime dependency. It works from `file://` too.

**Set a Content-Security-Policy.** The page makes no network requests — but that is a property of
the code, and a header makes it a property the browser *enforces*, including against any future edit
that forgets. This one is sufficient and was checked against what the page actually needs:

```
Content-Security-Policy: default-src 'none'; img-src 'self' data: blob:; font-src data:;
  style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; form-action 'none';
  base-uri 'none'; frame-ancestors 'none'
```

`connect-src 'none'` is the load-bearing part: with it, no `fetch`/XHR/WebSocket can leave the page
even if one were introduced. The `'unsafe-inline'` entries are unavoidable for a single-file page
(all the CSS and JS are inline); if you prefer, `_build.py` can be changed to emit hashes instead.

## Privacy, verified

The site owner's requirement was that submitted files never reach the server. Measured, not assumed:
loading the page over HTTP and exercising upload, lettering, all ten fonts, colours, engraving and
download produced **zero network requests** beyond the initial document — everything else is a
`data:` URI already inside the file. There is no `fetch`, XHR, WebSocket, `sendBeacon`, form, or
external URL anywhere in the page, and nothing is written to storage or cookies.

**Decided by magic bytes.** The mask must start with `89 50 4E 47 0D 0A 1A 0A` — PNG only, as
specified. The optional vein photo also accepts JPEG (`FF D8 FF`), because LeafMachine3 writes
`Lamina_RGB` as `.jpg` and a PNG-only rule would reject the very file the feature exists to read;
that is the one deliberate relaxation, and it is a relaxation of the FORMAT, not of the rule that
matters. Filename and MIME type are trusted for neither input — both are attacker-controlled. An SVG
saved as `evil.png` is refused from either drop zone, and nothing is uploaded in any case.

Two hardening fixes came out of building it: the colour field used to be passed straight into a CSS
`background`, where `url(https://…)` would have made the browser fetch it; and the page now carries
an inline `<link rel="icon">`, because without one a headed browser requests `/favicon.ico` — the
single HTTP request the page would otherwise cause.

## How faithful is it to the Python builder?

Every setting matches `generate_stl_from_mask.py`: `colors`, `color_tolerance`, `fill_holes`,
`simplify_tolerance_px`, `min_area_px`, `length_mm`, `thickness_mm`. On a real `Lamina_Mask` the two
agree to **0.08%** (150.0 × 95.36 mm here vs 150.0 × 95.28 mm; identical volume, 19,833 mm³) and both
produce a closed, correctly-wound solid.

The one deliberate divergence: OpenCV's `RETR_CCOMP` traces pixel **centres** and reports a two-level
parent/child hierarchy, while this traces pixel **boundaries** with marching squares and recovers
nesting by containment depth. That accounts for the ~1 px size difference, and it also handles an
island inside a hole — a speck of tissue inside an insect bite — which a two-level hierarchy cannot
represent.

## Lettering

Text is drawn to an offscreen canvas and then run through the *same* mask pipeline, which is why any
font works and why there is a stroke-thickening control (thin strokes do not survive a 0.4 mm
nozzle). Ten families are embedded as subsetted WOFF2.

- **Raised** — the letters are extruded on top and exported as their own shell resting on the face.
- **Engraved** — a real pocket. The letters become holes in the top face and the pocket gets its own
  floor, so the exported solid is one shell with *less* material (measured: −67.8 mm³ against
  +70.5 mm³ raised). Counters — the middle of an "o" — stay solid.

Drag the lettering directly on the model to place it; the pointer ray is intersected with the slab's
top plane, so no mesh raycasting is involved.

## The vein layer

An optional third layer, alongside the slab and the lettering. Feed it the `Lamina_RGB` photo that
accompanies your mask — both come from the same oriented crop, so they are the same pixel grid and
the default placement is a straight 1:1 overlay, exact rather than lucky. Uploading it opens a
preview window; when the extraction looks right, "Use this vein mask" turns it into a layer that
raises or engraves and moves exactly like the text (alt-drag on the model, since the veins cover the
whole leaf and a plain drag has to stay an orbit).

Veins are **dark, thin and low contrast**: on the test sheet the in-leaf standard deviation is ~15
grey levels and a vein sits only a few levels below the tissue beside it. A global threshold cannot
separate that, because the lamina's own brightness varies more across the leaf than a vein differs
from its neighbourhood. So the signal is a **local background subtraction** (a top-hat) computed
over integral images — radius is free, which is what makes a live preview possible — and restricted
to the leaf silhouette taken from the mask. That restriction is load-bearing: a plain blur averages
in the black background at the edge, the local mean collapses, and the entire rim lights up brighter
than any real vein.

The directional-opening filter ("keep only elongated shapes") is **off by default** despite being
the strongest noise filter, because it and the threshold do the same job. It earns its keep at a low
threshold, where texture would otherwise swamp the result; at a threshold high enough to stand alone
it mostly deletes the secondary veins — measured 1.3% vein coverage with it against 3.2% without, on
the same image. It stays as a toggle for flat, noisy scans.

Extraction settings are all in **pixels**, so the photo is resampled once onto a working grid capped
at 1400 px and the preview runs on that same grid — previewing at one resolution and exporting at
another would make the preview a polite fiction. A full pass is 80–220 ms at 767 × 780.

## Verified output

Measured with `trimesh` on files exported from the live page, all on one leaf:

| | shells | watertight | volume vs plain slab |
|---|---|---|---|
| plain slab | 1 | 1/1 | — |
| raised veins | 102 | 102/102 | **+140.7 mm³** |
| engraved veins | 1 | 1/1 | **−140.7 mm³** |

The two deltas being equal and opposite is the check that matters: the pocket removes exactly the
material the raised layer adds, so the engraving is real rather than a solid parked inside the slab.

Exports carry a small number of exactly-zero-area facets (18 of 15,268 on the lettered-and-veined
model). That is deliberate. They used to be filtered out, but earcut's output is a *partition* of the
face — dropping one leaves a genuine hole and six unmatched edges, and the shell stops being
watertight. Every slicer ignores a zero-area facet; none of them ignores a hole. The only visible
cost is that `trimesh.is_winding_consistent` reads False, because a degenerate triangle has no
normal to be consistent with; drop them and it reads True again, along with the holes.

## Boundary smoothing

Two controls under Shape cleanup, doing two different jobs, applied in that order:

| control | default | what it does |
|---|---|---|
| **Boundary smoothing** | 2 px | periodic Gaussian low-pass of the outline, in arc-length parameterisation |
| **Simplify** | 0.3 px | Douglas–Peucker decimation — purely a vertex/file-size control |

Douglas–Peucker alone cannot produce a curve, and no amount of tuning changes that: it only ever
*deletes* vertices, so every point it returns is an original pixel corner. Low tolerance keeps the
90° staircase; high tolerance replaces it with long straight chords; by 6 px it starts cutting
corners off the mask outright. Filtering the coordinate signal instead MOVES points off the pixel
lattice, which is what actually creates curvature.

Measured on the sample leaf: **324 faceted vertices before, 787 genuinely curved ones after**, with
the enclosed area unchanged to 0.01%. In the live page the whole lettered-and-veined model goes from
14,732 to 15,700 triangles — 6.6% for a curved outline. Smoothing also *helps* decimation, because
removing the staircase lets Douglas–Peucker do its job: on a 37.5 MP sheet mask, simplify 0.3 alone
leaves 18,576 vertices, and smoothing 2 with the same simplify leaves 2,944.

**Scoped to the leaf outline only.** The lettering and the vein layer call the same polygon builder
with their own fixed tolerances and no smoothing at all — a glyph's corners should stay sharp, and a
vein stroke is 2–4 px wide, thinner than any useful smoothing radius. Sweeping the control from 0 to
8 px leaves the vein-stroke count at exactly 101 throughout.

**Corner cutting was tried and rejected**, not on taste but on measurement: Chaikin subdivision
(×3, ×5) still carries the pixel staircase as a visible ripple, and costs 3,409 vertices to the
Gaussian's 787. It cuts corners locally without removing the frequency that makes the edge look
choppy in the first place.

**The per-ring feature-width cap.** For a ribbon of width `t` and length `L`, `2A/P` is `t`, which
estimates how thick a ring's shape actually is; sigma is capped at 0.35 of it. Uncapped, sigma 16
destroys 69 of 103 thin strokes outright (100% area loss); capped, none of them collapse and the
worst area change is 13.2%. The leaf outline measures ~168 px wide, so the cap never binds there —
it exists to protect small holes and islands that share the ring list with it.

Every setting tested (smoothing 0/1/2/4/8, simplify 1.5 and 0.3) exports with **all shells
watertight**, and none of the outlines self-intersect even at sigma 16. Cost at 37.5 MP: 272 ms
against 260 ms before, and +9 MB.

**Smoothing 0 with simplify 1.5 reproduces the original `generate_stl_from_mask` behaviour exactly**,
which is how the Python-agreement check above is still run.

## Known limits

- **Very large masks.** A 37.5 MP whole-sheet mask (5000 × 7500) builds in ~200 ms using ~4 MB of
  scratch. An earlier dense implementation needed 572 MB for the same mask and could fail outright
  on a laptop; the contour table is now sparse, because boundary is a perimeter phenomenon while a
  dense table costs area. Chrome still refuses to decode PNGs beyond roughly 30000 × 30000.
- **Smoothing moves rings, and nesting is recovered by containment.** The feature-width cap stops a
  ring collapsing into itself, but it does not know how far a hole sits from its parent — a wide hole
  very close to the outline, smoothed hard, could in principle bulge across it and be re-classified.
  Not observed on any real mask (and "fill internal holes" is on by default, which removes the case
  entirely), but it is the one topological assumption the cap does not cover.
- **Simplify above ~1 px undoes the smoothing**, flattening the curve back into chords. That is the
  honest interaction of the two controls rather than a bug: one adds curvature, the other removes
  vertices.
- **"Text height"** is the inked height of the whole string, so a descender makes the capitals
  proportionally shorter. That is the honest measurement of what gets printed.
- Alpha channels are ignored when matching colours, so a PNG with transparency may select a
  different foreground than the Python reference would.
- **Two engraved layers that overlap.** Engraving the lettering *and* the veins where they cross puts
  two sets of pocket walls through each other. The page detects it — by sampling real vertices
  against real rings, not bounding boxes, which for a layer covering the whole leaf would always
  fire — and says so, but does not resolve it. Raise one of the two.

## Licences

The sample pair (`sample_pair_mask.png` / `sample_pair_rgb.jpg`) is one *Liquidambar orientalis*
leaf from a LeafMachine3 run, shipped as a matched pair so the vein demo cannot silently show a photo
that does not line up with the mask.

Fonts are SIL Open Font License or Apache 2.0 (Inter, Montserrat, Oswald, Bebas Neue, EB Garamond,
Playfair Display, Roboto Slab, Lora, Pacifico, Roboto Mono) and are redistributable when embedded.
Triangulation is [earcut](https://github.com/mapbox/earcut) (ISC). Everything else is LeafMachine3.
