# Bilateral symmetry — experiment → production handoff

**Audience:** the agent implementing the production `bilateral_symmetry` pipeline stage.
**Source:** the experiment in this directory (`leafmachine3/modules/experiments/bilateral_symmetry/`),
its report `../bilateral_symmetry.html`, and `tests/test_bilateral_symmetry.py` (11 tests, passing).

The experiment measured 187 quantities per leaf across two axes to work out *which* ones are worth
keeping. Production keeps **four numbers and one picture**. Most of this document is about the
geometry that has to be exactly right and the traps that were already found and fixed — reintroducing
any of them silently corrupts the output rather than raising.

---

## 1. Scope

### Produce

| output | what |
|---|---|
| `si_a` | Shi et al. (2018) standardized symmetry index, **on the midvein axis**. 0 = perfect. |
| `archetype_score` | composite in [0, 1], weighted geometric mean of 4 terms |
| `is_archetypal` / `gates_pass` | the usable/exemplar decision + structural vetoes |
| QC panel PNG | `reports/Leaf_Data/Bilateral_Symmetry/<stem>__BSYM-leaf__x1_y1_x2_y2.png` |

Carry `a_star`, `dice` and `sinuosity` too — the score already computes them and they cost nothing
extra, `dice` is an input to the symmetry term, and `sinuosity` is the only metric in the whole suite
that is *uncorrelated* with the others (it is independent information, see §7).

### Do NOT port

The chord axis (the experiment's control), Taylor's power law, the Spearman correlation matrix,
Hausdorff, margin curvature / lobe counting, half-centroids and second moments, the cumulative
`C(s)` curve, the width-profile family, the cohort HTML report, and `figures.py`'s eight cohort
figures. All of it is in the experiment if a question comes back; none of it earns its keep per-leaf.

**Dropping the chord axis halves the compute** (see §8). The QC panel still *draws* a dashed chord
line, but that only needs the two endpoints — it does **not** need `build_axis("chord")`, which is
the expensive part. Draw it as a straight `tip → base` segment.

---

## 2. Architecture — stage computes, Reporter renders

```
… → LeafOrientation → PetioleWidth → BilateralSymmetry → MetricGrounding → Reporter → ECT
                                     ^^^^^^^^^^^^^^^^^   new
```

`pipeline.py:62` has an explicit `# <---------- extensible slot ---------->` comment immediately
after `PetioleWidth`. Insert there. `STAGE_ORDER` is the single source of truth — adding the entry
is all that is needed for `project_status` seeding and `depends_on` gating.

**The split is a hard requirement:** the stage computes and persists; the **Reporter** writes the QC
PNGs. That is why the `bilateral_symmetry` table must carry enough to redraw (§5).

This is not a new pattern — it is exactly how `Overlay_Petiole` already works. Copy that shape:

- `PetioleWidth` (`modules/petiole_width.py`) computes into `leaf_petiole`.
- `Reporter._export_petiole_overlays` (`modules/reporter.py:424-457`) reads `b.petioles`, rebuilds
  the crop, and writes `Overlay/Overlay_Petiole/<stem>__PET-leaf__x_y_x_y.<ext>`.
- The shared geometry lives in `core/petiole.py`, imported by both.

So: put the shared geometry in **`core/bilateral.py`**, imported by both the stage and the Reporter.
Add a `bilateral: list` field to the reporter bundle in `core/records.py` (alongside `petioles`,
`landmarks`, `morphology` at :259-262) and populate it from the new table.

### Why this stage can run before the Reporter

`ECT` has `depends_on = ("reporter",)` because it *loads the Reporter's `Leaf_Oriented` PNGs*.
**Bilateral symmetry does not.** The experiment rebuilds the oriented frame from
`leaf_segmentation` polygons + `leaf_morphology.oriented_leaf_rotation_angle_degreesCW`, and the
rebuild was verified to match the saved PNG at **IoU = 1.000000**. That is what lets the stage sit
at slot 11.5 and still hand the Reporter everything it needs.

```
depends_on = ("leaf_segmenter", "landmark_detector", "morphology", "leaf_orientation")
```

---

## 3. The geometry — copy this exactly

Lift `geometry.py` and the midvein half of `axes.py` into `core/bilateral.py` essentially verbatim.
The pieces that matter:

### 3a. Replaying the oriented frame onto landmarks

Landmarks are stored in **working (whole-sheet)** coords; the oriented mask is the result of
crop → rotate → content-fit. `geometry.load_leaves` replays that chain on the keypoints:

1. **Crop**, exactly as `Reporter._export_leaf_products` does — note it clamps:
   `cx1, cy1 = max(0, x1), max(0, y1)`, `cx2, cy2 = min(W, x2), min(H, y2)`. The *filename* uses the
   **unclamped** `det_box`; the *pixel origin* is the **clamped** `(cx1, cy1)`. Getting these
   backwards shifts every keypoint off the mask on any leaf whose box overruns the sheet edge.
2. **Rotate** with the same affine `core.imaging.rotate_image` builds — `geometry.oriented_affine`
   mirrors it line for line. Do not re-derive it.
3. **Content-fit**: subtract the `mask_bbox` origin of the **rotated silhouette**. The
   `Lamina_Holes_Mask` product fits on `silhouette` (holes filled), *not* on `lamina`.

Verify after porting: rebuilt mask vs the saved
`reports/Leaf_Oriented/Lamina_Holes_Mask/*.png` must give **IoU 1.000000**, and `lamina_tip` must
land at `frac_y ≈ 0.00` with `lamina_base` at `≈ 1.00`.

### 3b. The mask being measured

**Holes-filled silhouette** (`OrientedLeaf.silhouette`; LM3's `mask_includes="lamina"` /
`Lamina_Holes_Mask` product). The question is whether the *outline* mirrors; an insect hole is not
part of that outline. Holes are excluded from the gates and zeroed before scoring (§6), though
`hole_frac` is still recorded as a diagnostic.

### 3c. The curvilinear `(s, u)` frame

`s` = normalized arclength, **0 at the tip, 1 at the base**. `u` = signed perpendicular offset in
px, **positive to the viewer's LEFT**. Because every mask is tip-up oriented, "left" is the same
physical side for every leaf.

Two properties that must survive the port, both asserted in the tests:

- **Pixels are assigned to their nearest point on the axis** (a Voronoi partition of the polyline),
  *not* to equal-arclength perpendicular strips. Perpendicular strips along a curved axis overlap on
  the inside of a bend and gap on the outside. The Voronoi partition tiles the lamina exactly once,
  so `area_l.sum() + area_r.sum() == mask.sum()` **exactly**.
- **On-axis pixels (`u == 0`) split half to each side** for areas, and are **excluded** from
  centroids/margins. Assigning them all to one side made a perfectly mirror-symmetric mask report
  `A* = -0.009` instead of `0.0`.

### 3d. Midvein smoothing

`lamina_tip, midvein_0..14, lamina_base` (17 stations, `MIDVEIN_NAMES`), conf-filtered, fitted with
a smoothing spline (`SPLINE_SMOOTH = 3.0` per point) then resampled to equal arclength. **Do not skip
the smoothing** — raw pose jitter of a few px becomes spurious local curvature, which rotates the
perpendicular and manufactures asymmetry. Falls back to a plain polyline resample if the spline fails.

---

## 4. Traps — verified failures, do not reintroduce

Each of these was demonstrated by running code during an adversarial review. All are fixed in the
experiment; a fresh reimplementation can easily walk back into them.

| trap | symptom | correct behavior |
|---|---|---|
| **Forward-scattering pixels into the `(s,u)` grid** for the straightened/overlap views | Dice collapsed for small leaves — 2 leaves scored a literal `0.0`, and Dice correlated **0.884** with lamina area | Sample **backward**: map each grid cell's `(s, |u|)` to `path[s] ± |u|·normal[s]` and test the mask there. Fixed the correlation to 0.166 (n.s.). See `axes.straighten`. |
| Missing/NaN symmetry metrics | `archetype_score({}, qual)` returned **1.0** and passed `is_archetypal` — a leaf with no measurement outranked every measured leaf | return `nan` / `False`. Drop-and-renormalize is fine for the other three terms, never for the term that *is* the hypothesis. |
| `np.mean`/`np.min` on keypoint confidences | one NaN confidence → `kpt_conf_mean/min = nan` → trace term silently dropped → score **0.9925** | `nanmean`/`nanmin`; all-NaN → `nan`, not a passing value |
| Non-monotonic trace term | **deleting** the weakest midvein keypoint *raised* the score 0.0000 → 0.9661 | removing information must never improve the score |
| `np.argmax` on an all-zero/NaN array | reported the tip as "the location of the largest mismatch" | return `nan` |
| `0.0` for degenerate input | a 1-px mask read as "perfectly symmetric" | `nan`, never `0.0` |
| Denominator guard not applied to width bins | `aw_max_abs` pinned at the noise ceiling `1.0000` | guard both area and width profiles (not needed if you drop the width family, as recommended) |
| `flush_edge_frac` as a truncation gate | r = **+0.848** against `1/√area_px` — it is a leaf-**size** proxy, fires on 1 of 105 leaves | do not gate on it; `truncated` answers the question directly |
| Raw `n_components` as a gate | vetoed a leaf scoring **0.96** whose second component held **0.001%** of the area | gate on `largest_frac` instead (§6) |

---

## 5. The `bilateral_symmetry` table

Follow `leaf_ect` / `leaf_petiole` in `core/schema.sql`. All SQL goes in `core/db.py` — no stage
embeds SQL.

```sql
-- bilateral_symmetry : one row per oriented Leaf_WHOLE leaf (BilateralSymmetry stage).
-- Carries the metrics AND the frame geometry the Reporter needs to redraw the QC panel, so the
-- Reporter never has to re-derive the oriented frame from scratch.
CREATE TABLE IF NOT EXISTS bilateral_symmetry (
    leaf_id        INTEGER PRIMARY KEY REFERENCES leaf_segmentation(leaf_id) ON DELETE CASCADE,
    specimen_id    INTEGER NOT NULL REFERENCES specimen(specimen_id) ON DELETE CASCADE,
    detection_id   INTEGER, instance_index INTEGER,
    -- metrics, midvein axis, holes-filled silhouette
    si_a           REAL,     -- Shi et al. standardized index; 0 = perfect
    a_star         REAL,     -- signed total imbalance in [-1,1]; + = viewer's LEFT half larger
    dice           REAL,     -- straightened mirrored-half overlap; 1 = perfect
    sinuosity      REAL,     -- midvein arclength / straight tip-base distance
    -- composite
    archetype_score REAL,
    term_symmetry REAL, term_integrity REAL, term_completeness REAL, term_trace REAL,
    gates_pass     INTEGER,  -- structural vetoes passed
    is_archetypal  INTEGER,  -- gates_pass AND archetype_score >= min_score
    reasons_json   TEXT,     -- human-readable penalty/veto reasons
    -- quality diagnostics (reported; see section 6 for which are SCORED)
    largest_frac REAL, solidity REAL, perimeter_ratio REAL, hole_frac REAL,
    kpt_conf_mean REAL, kpt_conf_min REAL, n_midvein_kpts INTEGER, truncated INTEGER,
    -- frame geometry: everything needed to rebuild the oriented frame for the QC panel
    angle_cw       REAL,     -- CW rotation applied (duplicated from leaf_morphology on purpose)
    crop_w INTEGER, crop_h INTEGER,     -- pre-rotation crop dims (CLAMPED to the sheet)
    fit_x1 INTEGER, fit_y1 INTEGER,     -- content-fit box origin in the rotated canvas
    mask_w INTEGER, mask_h INTEGER,     -- final oriented mask dims
    tip_x REAL, tip_y REAL,             -- ORIENTED-frame coords
    base_x REAL, base_y REAL,
    midvein_json   TEXT,     -- (N,2) tip->base polyline, ORIENTED coords, conf-filtered stations
    n_bins         INTEGER,  -- arclength bins used
    qc_png         TEXT,     -- reports/Leaf_Data/Bilateral_Symmetry/<leaf>.png
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_bsym_spec ON bilateral_symmetry (specimen_id);
```

**Why geometry and not the profiles.** Panels 2 and 3 of the QC figure are rendered from the
per-pixel `(s, u)` field, not from the binned profiles, so the Reporter has to rebuild the frame
regardless — storing 200-bin profiles would add ~8 kB/leaf (~176 MB on a 22 k-leaf run) and still not
remove the rebuild. Storing the midvein polyline + tip/base + box origins (~1 kB/leaf) lets the
Reporter call `core.bilateral.build_axis` directly with no re-derivation from landmarks, no
re-reading of `leaf_landmark`, and no dependence on the confidence filter still being set the same
way. **Measured Reporter-side cost when QC images are on: 308 ms/leaf** — read §8 before choosing a
default, it is the single most expensive decision in this module.

**Migration.** New table via `CREATE TABLE IF NOT EXISTS` in `schema.sql` is enough; existing project
DBs pick it up on `init_schema`. If you later add a column, follow the `_ECT_MIGRATIONS` /
`_PETIOLE_MIGRATIONS` `ALTER TABLE` pattern at `core/db.py:146-171` — `CREATE TABLE IF NOT EXISTS`
cannot add columns to an existing DB.

Also add `qc_png` to `_DEFAULT_ARTIFACT_COLS` (`core/db.py:1123`) so the file is cleaned up when the
stage is reset, and add `"bilateral_symmetry"` to the stage's `owns_tables`.

---

## 6. Scoring and gates

Port `quality.compute_quality` + `archetype_score`, and the **gate policy from `run.py`**, not
`quality.is_archetypal`'s stock gate set (the driver deliberately overrides it).

```python
GATE_MIN_LARGEST_FRAC = 0.995
GATE_MIN_MIDVEIN_KPTS = 10
# gates: not truncated · area_px > 0 · largest_frac >= 0.995 · n_midvein_kpts >= 10
```

Deliberately **not** gates — each with the measured reason:

- **holes** — the measured shape is the holes-filled silhouette. Also zero `hole_frac` before
  scoring (`dataclasses.replace(q, hole_frac=0.0)`) so it cannot reach the composite either.
- **raw `n_components`** — counts specks. 18 of 105 leaves have >1 component but 12 have <0.1% of
  area outside the main blob. Note this is *not* because holes are filled: filling a hole removes an
  interior void, it cannot merge disjoint blobs, and the counts were **identical on the holes-filled
  and holes-punched masks for all 105 leaves**. `largest_frac` keeps the part that matters.
- **`flush_edge_frac`** — size proxy (§4).

Weights (`ARCHETYPE_WEIGHTS`): symmetry 0.40, integrity 0.25, completeness 0.05, trace 0.15,
combined as a **weighted geometric mean** so any single near-zero term vetoes the leaf. Score cut
default **0.50** (`min_score`).

**State this in the docs you write.** On a cohort of already-clean masks the composite tracks `si_a`
at Spearman **ρ = −0.97**, because integrity/completeness/trace saturate at 1.0 for 94/99/76% of
leaves. Symmetry is the only term that varies, so it *is* the ranking. That is a fact about the data,
not a validation of the composite. Every constant is a first proposal to be tuned against the QC
images, not an established value.

---

## 7. What the experiment concluded (carry into the module docstring)

- **The midvein axis matters.** Against the straight chord the published methods must use, median
  `si_a` runs 0.151 vs **0.142** on the midvein and Dice 0.857 vs **0.895**. The chord overstates
  asymmetry for 62% of leaves, and midvein **sinuosity predicts the penalty** (Spearman **+0.47**,
  p = 4e-07). Production measures on the midvein for this reason.
- **Symmetry currently detects bad landmark traces at least as well as bad masks.** The
  lowest-Dice leaf in the cohort was not asymmetric — its midvein had been traced along the margin.
  Extreme `|a_star|` coincides with low `kpt_conf_min` (0.37 vs 0.94). While the pose model is alpha
  this may be the more useful application.
- **Sinuosity is the one uncorrelated metric.** Everything else clusters (Dice/IoU are a
  deterministic reparameterization, ρ = 1.00; `wsi_a` tracks `si_a` at 0.95). Keep sinuosity.
- **A leaf can be genuinely asymmetric and perfectly masked** (oblique bases are normal in many
  taxa). Low symmetry is evidence of a bad mask only together with the structural diagnostics — which
  is what the QC images are for.

---

## 8. Cost, parallelism, settings

Measured on the 105-leaf cohort (median lamina 72,882 px), per leaf:

| | ms/leaf |
|---|---|
| `load_leaves` (amortized) | 4.1 |
| `build_axis` ×2 (chord + midvein) | 106.9 |
| `profiles` ×2 | 4.5 |
| `compute_metrics` ×2 | 1.0 |
| `straighten` ×1 | 3.0 |
| **total, both axes** | **119.4** |

Production drops the chord axis → **~65 ms/leaf** for compute. `build_axis` dominates (a
`cKDTree` query of every lamina pixel against 512 axis vertices); if that proves too slow at scale,
reduce `N_AXIS` before touching anything else.

### The QC images are the expensive part — decide the default deliberately

Measured Reporter-side, per leaf, with `qc_images: true`:

| | ms/leaf |
|---|---|
| rebuild the frame (`build_axis` + `profiles`) | 61 |
| `fig_leaf_panel` (the 3-up) | **247** |
| **total added to the Reporter** | **308** |

On the 22 k-leaf `runs/demo` that is **~114 minutes** added to a stage that is otherwise
disk-write-bound. The figure, not the geometry, is 80% of it — so storing the profiles in the table
would buy back at most 20%, which is why §5 does not recommend it.

Three ways to make this affordable; pick one and document it:

1. **`qc_images: flagged`** (recommended default) — render only leaves that fail the gates or fall
   below `min_score`. Those are the ones anyone actually opens, and on a clean cohort that is ~15%
   of leaves (~17 min on 22 k) instead of 100%.
2. **Single-panel variant** — `fig_leaf_split` (panel 1 only: oriented mask, tinted halves, both
   axes) costs **46 ms/leaf**, 5.4× cheaper, and is the panel that actually distinguishes a bad
   midvein trace from an asymmetric leaf. Panels 2 and 3 are explanatory rather than diagnostic.
3. **`qc_images: all`** — full 3-up for every leaf; fine for example-scale runs, budget the 114 min
   on a large sweep.

Whichever is chosen, `qc_images: false` must cost **zero** — no frame rebuild, no figure.

Mirror the `ECT` stage's executor shape — matplotlib is GIL-bound and pyplot is not thread-safe:

```python
device_kind = "cpu"; cpu_parallel = "process"; fanout = True
est_item_seconds = 0.07          # compute only; the QC figure is the Reporter's cost
```

Use the object-oriented `Figure` + `FigureCanvasAgg` API, **never pyplot** (global state, unsafe in a
process pool). `figures.to_data_uri` / `_figure` show the pattern.

### Settings

```yaml
  bilateral_symmetry:
    enabled: true
    min_kpt_conf: 0.25      # keypoint confidence floor for the midvein trace
    n_bins: 200             # arclength strips per leaf
    min_score: 0.50         # archetype score needed for is_archetypal (gates always apply)
    qc_images: flagged      # none | flagged | all -- see the cost table above
```

`qc_images` is deliberately an enum rather than a bool: at `all` it adds ~114 min to a 22 k-leaf
run, so the user needs the middle option. If you prefer a bool for consistency with
`modules.ect.radial_viz`, pair it with a second key (`qc_images_scope`) rather than making `true`
mean the 114-minute path.

Add to `LM3_settings.yaml` **and** `LM3_settings.orig.yaml` (commented form), a
`modules.bilateral_symmetry.*` block in `server/ui/settings_meta.json` (copy the shape of
`modules.ect.radial_viz` at :1361), and a `_CATEGORY_BLURB` entry for
`Leaf_Data/Bilateral_Symmetry` in `server/results_api.py:126`.

Every key needs an inline default in the stage's `_settings()` — that is the single source of truth
in this codebase, so a missing YAML key is never an error (see `modules/ect.py:58-77`).

Because rendering lives in the Reporter, `qc_images` is read by the Reporter. Follow ECT's
`enforce_report_deps` precedent (`modules/ect.py:268-294`) if the module needs to force any Reporter
behavior on.

---

## 9. Naming — no `#`

The experiment used `f"{stem}#{leaf_id}"` as a display key. **Do not carry that into production.**
Leaf crops already have a canonical identifier:

```
<stem>__<LABEL>__<x1>_<y1>_<x2>_<y2>.<ext>
```

built by `core.imaging.crop_filename(stem, label, xyxy, ext)` and parsed back by
`parse_crop_filename`. The three logical parts are joined by `__` so the stem and the coords keep
their own single underscores.

- Label: `crop_label(cfg, "bilateral", "Leaf")` → **`BSYM-leaf`**. Add
  `bilateral_prefix: "BSYM"` to the `naming:` block (`LM3_settings.orig.yaml:50-55`) and to
  `_PREFIX_KEYS` in `core/naming.py`.
- Use the **unclamped** `det_box` in the filename (matches every other product; the clamped box is
  only the pixel origin).
- Multi-leaf crops: fold the instance into the stem exactly as ECT does —
  `stem if inst == 0 else f"{stem}__i{inst}"` (`modules/ect.py:149`) — so two instances sharing
  one detection box do not collide.

Final path:

```
reports/Leaf_Data/Bilateral_Symmetry/<stem>__BSYM-leaf__<x1>_<y1>_<x2>_<y2>.png
```

`core.imaging.save_image` already does `makedirs(..., exist_ok=True)`, so no directory plumbing.

---

## 10. The QC panel

Port `figures.fig_leaf_panel` — the three-up used in the report's Most/Least archetypal galleries:

1. **Oriented mask** with the left half tinted `#38bdf8` and the right `#fb923c`, the midvein axis
   solid `#4ade80`, the chord dashed `#6f757f`, and tip/base markers.
2. **Straightened** `(s, u)` view — the same two halves with the midvein now a straight vertical line.
3. **Mirrored overlap** — left half against the reflected right half, symmetric difference in
   `#f87171`, annotated with Dice and SD.

Palette is the `timing.html` house theme (`bg #101012`, `panel #191a1d`, `ink #e8e8ea`,
`line #2a2b30`, `acc #fb923c`, `acc2 #38bdf8`, `acc3 #4ade80`, `bad #f87171`) — keep it so the QC
images match the rest of the LM3 reports. Left = blue, right = orange, consistently.

Label the `s` axis **tip (0) → base (1)** in that direction. `fig_leaf_panel` currently labels
whatever it is handed as `frame_mid` "midvein axis" without checking `.kind`; in production only the
midvein frame is built, so label from `frame.kind` rather than from the argument position.

Write PNG (masks and flat fills, so PNG beats JPEG at this size and avoids ringing on the tinted
halves).

---

## 11. Verification checklist

Before calling it done:

1. Rebuilt oriented silhouette vs `reports/Leaf_Oriented/Lamina_Holes_Mask/*.png` → **IoU 1.000000**.
2. `area_l.sum() + area_r.sum() == mask.sum()` exactly, for every leaf.
3. `lamina_tip` → `s ≈ 0` at the top of the mask; `lamina_base` → `s ≈ 1`; `width_left` → `u > 0`.
4. A synthetic exactly mirror-symmetric mask → `si_a ≈ 0`, `a_star == 0.0` exactly, `dice ≈ 1`.
5. Mirroring a real mask **negates** `a_star` exactly and leaves `si_a` unchanged.
6. `archetype_score` with missing symmetry → `nan`, `is_archetypal` → `False`.
7. Dice must **not** correlate with lamina area (the straighten trap): Spearman well under 0.3.
8. QC filenames round-trip through `parse_crop_filename`.
9. `--restart bilateral_symmetry` on a pre-existing project DB creates the table, repopulates, and
   rewrites the QC images.
10. Port the relevant cases from `tests/test_bilateral_symmetry.py` (11 tests) to cover the new
    `core/bilateral.py`.

---

## 12. File map

| experiment file | production destination |
|---|---|
| `geometry.py` | `core/bilateral.py` — verbatim, it is the verified part |
| `axes.py` (midvein path only) | `core/bilateral.py` — `build_axis`, `profiles`, `straighten` |
| `metrics.py` | `core/bilateral.py` — only `SI_A`, `A_star`; drop the other 40 fields |
| `shape.py` | `core/bilateral.py` — only the Dice/IoU/SD overlap block |
| `quality.py` | `core/bilateral.py` — `compute_quality`, `archetype_score`, `archetype_subscores` |
| `run.py` `gates_pass()` + `GATE_*` | `core/bilateral.py` — this is the production gate policy |
| `figures.py` `fig_leaf_panel` | `reporting/bilateral_viz.py` — Reporter-side |
| `run.py` (driver) | `modules/bilateral_symmetry.py` — stage; drop the cohort/report layer |
| `report.py`, the other 8 figures, `../bilateral_symmetry.html` | **not ported** — reference only |
