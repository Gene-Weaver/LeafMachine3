# rotated_bbox_comparision

A **manual/visual** comparison harness (not a pytest test) for the leaf rotated
("length × width") bounding-box methods. It renders a labeled 2×3 panel per leaf comparing
six ways to fit the box and reports, for each, the box length/width (px), aspect ratio,
long-axis tilt, and `box_area / mask_area` (1.0 = perfectly tight).

## Methods

| # | method | in the repo? |
|---|--------|--------------|
| 1 | **LM2 `fit_min_bbox`** — rotate until the axis-aligned box's long side = min-enclosing-circle diameter | ✅ default (`morphology.method: lm2`) |
| 2 | **`cv2.minAreaRect`** — true minimum-area rotated rectangle | ✅ option (`morphology.method: minarearect`) |
| 3 | **Feret axis + hull extents** — orient by the longest-chord axis, measure hull extents | prototype (here only) |
| 4 | **PCA principal axis** — orient by the mask's second-moment axis | prototype (here only) |
| 5 | **`cv2.fitEllipse` axis** — orient by a least-squares ellipse major axis | prototype (here only) |
| 6 | **Hybrid** — Feret, unless it disagrees with PCA by > 20° → PCA | prototype (here only) |

Methods 1–2 call the production code (`leafmachine3.core.morphometrics.polygon_morphology`);
3–6 are prototypes implemented in `compare_methods.py` only (the repo ships 1–2).

## Run

```bash
cd LM3
PYTHONPATH=. python tests/rotated_bbox_comparision/compare_methods.py            # all three leaves
python tests/rotated_bbox_comparision/compare_methods.py mask.png out.png "Name" # one arbitrary mask
```

## Inputs / outputs (self-contained)

`inputs/` holds the three leaf masks (and their RGB-crop backdrops) copied from a real run, so
the comparison reproduces without needing `examples_out/`:

- `…Catalpa_speciosa__…_1372_308_1818_929` — **cordate / heart-shaped**, AR ≈ 2.0
- `…Platanus_macrophylla__…_35_739_2008_2992` — **palmate / lobed** (maple-like), AR ≈ 1.17
- `…Posoqueria_mutisii__…_1284_301_2334_1529` — **simple / entire** ("normal"), AR ≈ 2.24

Panels: `panel_catalpa_cordate.png`, `panel_platanus_lobed.png`, `panel_posoqueria_normal.png`.

## What the panels show

- **Normal leaf (Posoqueria):** all six methods are ~identical (tilt within 4°, box/mask 1.38×).
  A clean elongated leaf has an unambiguous long axis; method choice doesn't matter.
- **Cordate leaf (Catalpa):** length/width agree (<3%); the improved methods (Feret/PCA/ellipse)
  independently converge on the true apex→petiole midrib (79°), while cv2 snaps to 90°.
- **Lobed leaf (Platanus):** the opposite — **LM2 and cv2 stay axis-aligned and hug the leaf
  cleanly**, while the geometric "improvements" impose a spurious ~15–22° tilt (the principal
  axis is ill-defined for a near-round shape).

**Takeaway:** `lm2` is a sound default (aligns when elongated, stays axis-aligned when the shape
is ambiguous); the geometric alternatives trade elongated-leaf orientation for round-leaf
robustness rather than being strictly better. The one clear future win is **landmark-based
orientation** (petiole→apex midvein), which is botanically correct by construction.
