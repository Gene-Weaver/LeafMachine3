# exp__settingslayouts

Five denser alternatives to the **Settings** and **Postprocessing** forms, as static HTML you can
open in a browser. Nothing here is wired up — they exist to be looked at and argued with.

```
open exp__settingslayouts/index.html      # all six, with a switcher and a density chart
```

| file | layout |
|---|---|
| `index.html` | comparison shell — baseline + all five, Settings/Postprocessing toggle per pane |
| `01-columns.html` | **A** · Column Grid |
| `02-rail.html` | **B** · Master–Detail Rail |
| `03-table.html` | **C** · Property Sheet |
| `04-inspector.html` | **D** · Inspector Split |
| `05-workbench.html` | **E** · Search Workbench |
| `_build.py` | the generator — edit this, not the HTML, then re-run it |

The palette is copied verbatim from `leafmachine3/server/ui/css/app.css`, and the content is real:
settings from `ui/settings_meta.json` + `LM3_settings.yaml`, collage inputs from
`postprocess_api._COLLAGE_INPUTS`, overlay colors from `report.overlay.classes`.

## Why the current form scrolls so much

Measured, not estimated — from `app.css` box metrics over the real 290-setting corpus.

| | |
|---|---|
| visible form area, 1080p window, perf panel open | **515 px** |
| average row, descriptions on | **63.5 px** (41.8 px off) |
| settings on screen | **~8 of 290** |
| whole file expanded | **22,068 px ≈ 43 screens** |
| Overlays alone (103 settings) | **7,937 px ≈ 15.4 screens** |
| group chrome (85 headers) | **3,310 px** before a single setting |
| leaf-collage tool (29 inputs) | **2,756 px ≈ 4.4 screens** |

Three causes:

1. **`.desc` is `grid-column:2`.** `.treerow` is a two-column grid, and the help text — plus both
   `.err` slots — each claim a whole extra grid row under the control. Help costs **+21.7 px on
   every row**, which is 52% of the form's height; all 290 settings have help text.
2. **Controls inherit `width:100%`.** A number input spans ~1200 px to hold four characters, so the
   row can never be shared. 102 of 290 settings are a 19 px toggle sitting in a 63.5 px row.
3. **Nesting.** 25 top-level + 60 nested groups. Even collapsed, the headers alone are 3,310 px.

What already helps: *Important only* hides 194 of 290 rows, *Descriptions* recovers 6,293 px (29%),
and the 10 category chips keep one section on screen. **The Postprocessing tab has none of these** —
no search, no important-only, no descriptions toggle, and its help always renders.

## The five

**A · Column Grid — 55 visible.** Keep the groups and every existing control; flow rows into
responsive columns at 28 px and give each control only the width its type needs. Help becomes a
hover tooltip with a global ⓘ toggle to pin it inline. *Lowest risk — the row builders barely
change.* Tradeoff: sub-groups (Input / Output / Run mode) flatten into their parent.

**B · Master–Detail Rail — 34 visible.** A persistent category→group rail replaces the accordion.
One group on screen at a time and it always fits; the rail doubles as an overview and shows which
groups hold changes. Tradeoff: two clicks to reach a setting you can't name.

**C · Property Sheet — 19 visible.** One sortable, filterable table: 26 px rows, sticky header,
inline editors, and a **Default** column that makes drift obvious. Sort by *changed* to audit a run.
Tradeoff: least discoverable for someone who doesn't already know the file.

**D · Inspector Split — 21 visible.** A 24 px scan list beside a docked panel that explains whatever
has focus; arrow keys walk the list. The only layout that gets *more* readable as help text gets
longer. Tradeoff: costs ~320 px of width.

**E · Search Workbench — nothing is browsed.** Nothing expanded by default. A palette searches all
290 by label, path, help *and current value*; a pinned board holds the six you change per run; bulk
types get purpose-built editors — the 40 overlay colors become a swatch grid, so the worst section
in the file goes from 15.4 screens to one. Tradeoff: the biggest departure, and it hides structure
from someone exploring.

## Outcome

**B (Master–Detail Rail) shipped** into both real tabs on 2026-08-07 — see
`leafmachine3/server/ui/css/app.css` (the `.railwrap` / `.rail-pane` block) and the two tab modules.
The mockups below are kept as the record of what was compared.

## The original recommendation

**A for Settings, B for Postprocessing.** A is a contained change to `.treerow` plus a grid
container and buys the largest factor (~7×); B suits Postprocessing because a tool's inputs are
already grouped and one group always fits, which also kills the 4.4-screen scroll to reach *Run*.
E's swatch grid is worth stealing for Overlays regardless of which layout wins — that one section is
36% of the entire file and 80% of it is color-and-toggle.
