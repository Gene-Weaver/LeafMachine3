#!/usr/bin/env python3
"""Generate the settings-layout experiments.

Emits six panes (the current design + five alternatives) twice: as standalone `NN-<name>.html`
files, and merged into `index.html` with a switcher. Every pane renders CLIENT-SIDE from the real
corpus, so the navigation each design depends on can actually be tried.

    .venv_LM3/bin/python exp__settingslayouts/_extract.py > exp__settingslayouts/settings.json
    python3 exp__settingslayouts/_build.py

The corpus is real: 290 settings from ui/settings_meta.json + core.config.builtin_defaults() +
LM3_settings.yaml, the leaf collage tool's 29 inputs from postprocess_api._COLLAGE_INPUTS, and the
palette copied verbatim from ui/css/app.css.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# The leaf collage tool, shaped like the settings corpus so one renderer serves both
# --------------------------------------------------------------------------- #
_COLLAGE = [
    ("input", "Input", [
        ("run_dir", "Run folder", "path", "examples_out/collage_test", 1, None, "The finished LM3 run to read. Its project database supplies the archetype scores and the veto flags; its reports/ tree supplies the leaf masks."),
        ("primary_mask", "Primary mask", "path", "US_1321635795_Moraceae_Morus_alba__SEG-lamina__721_118_1317_928.png", 1, None, "The mask whose white area becomes the collage's overall shape. Pick from the run's own high-scoring leaves for a leaf built out of leaves."),
        ("output_dir", "Output folder", "path", None, 0, None, "Where the collage PNG lands. Empty writes it to the run's reports/Collage folder."),
        ("name", "File name", "string", None, 0, None, "Output file stem, without the .png. Empty names it after the layout and the primary mask."),
    ]),
    ("leaves", "Which leaves", [
        ("min_archetype_score", "Minimum archetype score", "float", 0.8, 1, 0.8, "Keep leaves scoring STRICTLY above this (0-1). Leaves that failed a structural veto are excluded no matter how they scored."),
        ("max_leaves", "Leaf cap", "int", 0, 1, 0, "At most this many leaves, highest score first. 0 uses every leaf that passes."),
        ("tree", "Leaf orientation", "enum", "Leaf_Oriented", 0, "Leaf_Oriented", "Leaf_Oriented is rotated tip-up and reads as a coherent collage; Leaf_Original keeps each leaf as it sat on the sheet."),
        ("mask_variant", "Mask version", "enum", "lamina_mask", 1, "lamina_mask", "Which per-leaf product to tile. lamina_mask is the lamina with its holes CUT OUT."),
    ]),
    ("appearance", "Appearance", [
        ("style", "Leaf rendering", "enum", "mask", 1, "mask", "mask recolors each binary mask to the color below. rgb uses the matching RGB cutout."),
        ("color", "Leaf color", "color", [255, 255, 255], 1, [255, 255, 255], "The color solid leaves are drawn in. Used by the mask style only."),
        ("background", "Background", "color", "transparent", 1, "transparent", "Transparent drops the collage onto any slide or poster. Pick a color to flatten it onto a solid sheet."),
        ("max_dim_px", "Longest side", "int", 10000, 1, 10000, "Longest side of the output PNG in pixels; every tile is scaled to suit."),
        ("tile_scale", "Tile fill", "float", 0.98, 0, 0.98, "How much of its cell each leaf fills. Below 1.0 opens a little air between leaves."),
    ]),
    ("arrangement", "Arrangement", [
        ("layout", "Arrangement", "enum", "grid", 1, "grid", "grid solves one square cell per leaf. mosaic is a quadtree. organic packs each leaf into the largest remaining pocket."),
        ("ranking", "Score placement", "enum", "center", 1, "center", "center puts the best leaves deepest inside the silhouette (and, in the mosaic, on the biggest tiles)."),
        ("random_seed", "Random seed", "int", 0, 0, 0, "Same seed, same collage. Change it to re-roll the organic packing and the shuffle."),
        ("shuffle_top", "Shuffle the top N", "int", 0, 0, 0, "Shuffle the N highest-scoring leaves among themselves before placing them."),
        ("layout_px", "Layout resolution", "int", 2048, 0, 2048, "Resolution the arrangement is solved at, NOT the output size."),
    ]),
    ("primary", "Primary mask", [
        ("primary_colors", "Primary foreground", "list", ["white"], 0, ["white"], "Which color(s) in the primary mask count as its shape. LM3 masks are white on black."),
        ("primary_color_tolerance", "Primary color tolerance", "int", 0, 0, 0, "Per-channel match slack, 0-255. 0 is exact, which is right for a clean binary mask."),
        ("primary_fill_holes", "Fill the primary mask's holes", "bool", True, 0, True, "Fill the outline mask's own holes so the collage silhouette is solid."),
    ]),
    ("tuning", "Layout tuning", [
        ("mosaic_min_cell_px", "Mosaic smallest tile", "float", 24, 0, 24, "The quadtree stops subdividing below this tile size, in output pixels."),
        ("organic_rotate", "Organic rotation", "bool", True, 0, True, "Give each packed leaf a random rotation. Off keeps every leaf upright."),
        ("organic_fill", "Organic pocket fill", "float", 0.9, 0, 0.9, "Leaf size as a fraction of its pocket's inscribed circle."),
        ("organic_max_scale", "Organic size spread", "float", 3.0, 0, 3.0, "How much bigger the largest leaf may be than the average one."),
        ("organic_gap_px", "Organic clearance", "float", 2, 0, 2, "Minimum space kept between packed leaves, in output pixels."),
        ("organic_min_tile_px", "Organic smallest tile", "float", 12, 0, 12, "Stop packing once the biggest free pocket falls below this."),
        ("tmp_dir", "Scratch folder", "path", None, 0, None, "Where the collage is staged while it encodes, in a _leaf_collage subfolder."),
        ("write_manifest", "Write the leaf manifest", "bool", True, 0, True, "Also write a .json next to the collage listing every placed leaf, its score, and where it landed."),
    ]),
]


def tool_dataset() -> dict:
    rows = []
    for sid, label, items in _COLLAGE:
        for k, l, t, v, i, d, h in items:
            row = {"k": k, "l": l, "t": t, "h": h, "i": i, "g": label, "s": sid, "v": v, "d": d}
            if d is not None and v != d:
                row["c"] = 1
            rows.append(row)
    return {"title": "leaf collage builder",
            "sections": [{"id": s, "label": lb} for s, lb, _ in _COLLAGE],
            "rows": rows}


# --------------------------------------------------------------------------- #
# Panes
# --------------------------------------------------------------------------- #
PANES = [
    ("baseline", "0", "Today", "the current form, to scale", "8", "settings visible",
     "Reference, rendered from the same data at the real 63.5px row height. Help text sits on its own "
     "grid line; 102 of 290 settings are a 19px toggle in a 63.5px row.", False),
    ("columns", "A", "Column Grid", "same model, packed into columns", "55", "settings visible",
     "Lowest-risk: reuses every existing control and the group model. Rows drop to 28px and flow into "
     "responsive columns, so a number input stops being 1200px wide. Help moves to hover, with a "
     "toggle to pin it back inline. <b>Try:</b> the section chips, Important only, ⓘ Descriptions.", True),
    ("rail", "B", "Master–Detail Rail", "navigate instead of scroll", "34", "settings visible",
     "A persistent section→group rail replaces the accordion. One group on screen at a time and it "
     "always fits; the rail doubles as an overview and marks which sections hold changes. "
     "<b>Try:</b> click a section to expand its groups, then pick a group; or filter the rail.", True),
    ("table", "C", "Property Sheet", "one sortable, filterable table", "19", "settings visible",
     "The about:config / IDE property-grid idiom. 26px rows, sticky header, and a Default column that "
     "makes drift visible. <b>Try:</b> click any column header to sort, use the filter chips, or type "
     "in the box — it searches name, path, help <i>and current value</i> across all 290.", True),
    ("inspector", "D", "Inspector Split", "list scans, panel explains", "21", "settings visible",
     "A 24px scan list beside a docked panel that follows focus. Help costs the list nothing, so it can "
     "be as long as it needs to be. <b>Try:</b> click a row, then use ↑ and ↓ — the panel follows.", True),
    ("workbench", "E", "Search Workbench", "you never browse 290", "290", "reachable in 2 keys",
     "Search-first: a palette over label, path, help <i>and current value</i>; a pinned board for the "
     "handful you touch per run; and purpose-built bulk editors — the 41 color settings become one "
     "swatch grid. <b>Try:</b> type <code>leaf</code> or <code>0.35</code> in the palette; click a toggle.", True),
]


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
CSS = r"""
/* LM3 tokens -- verbatim from leafmachine3/server/ui/css/app.css */
:root{
  --bg:#101012; --panel:#191a1d; --panel2:#1f2024; --ink:#e8e8ea; --mute:#9ca3af;
  --dim:#6f757f; --line:#2a2b30; --acc:#fb923c; --acc2:#38bdf8; --acc3:#4ade80;
  --warn:#fbbf24; --bad:#f87171; --vio:#c084fc;
  --panel3:#141519; --hover:#1c1d21; --void:#0b0b0d; --ok:var(--acc3);
  --mono:ui-monospace,SFMono-Regular,Menlo,"DejaVu Sans Mono",Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
  --r:9px; --r-sm:6px; --r-xs:4px;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 var(--sans);
     -webkit-font-smoothing:antialiased}
*{scrollbar-width:thin;scrollbar-color:#3a3c43 transparent}
*::-webkit-scrollbar{width:10px;height:10px}
*::-webkit-scrollbar-thumb{background:#33353c;border-radius:8px;border:3px solid transparent;background-clip:content-box}
.ell{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.sp{flex:1 1 auto}
code{font-family:var(--mono);font-size:.92em;background:var(--panel2);padding:1px 5px;border-radius:var(--r-xs)}
h4{margin:0;font-size:1rem;letter-spacing:-.01em}
input{font:inherit}

.kicker{font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--acc);font-weight:700}
.impdot{width:6px;height:6px;border-radius:50%;background:var(--acc3);display:inline-block;flex:0 0 auto}
.chgdot{width:6px;height:6px;border-radius:50%;background:var(--acc);display:inline-block;flex:0 0 auto}
.nodot{width:6px;height:6px;display:inline-block;flex:0 0 auto}

/* stand-in controls */
.f{display:inline-flex;align-items:center;gap:6px;background:#0f1013;border:1px solid var(--line);
   border-radius:var(--r-sm);padding:4px 8px;font:12.5px/1.3 var(--mono);color:var(--ink);
   min-width:0;max-width:100%}
.f.sm{padding:2px 7px;font-size:12px}
.f.num{min-width:70px;justify-content:flex-end;font-variant-numeric:tabular-nums}
.f.sel b{color:var(--dim);font-weight:400;margin-left:auto;padding-left:6px}
.f.pathf{flex:1 1 auto;min-width:110px} .f.pathf b{color:var(--dim);margin-left:auto;padding-left:6px}
.f.listf{color:var(--acc2)}
.f.col .swatch{width:12px;height:12px;border-radius:3px;border:1px solid var(--line);flex:0 0 auto}
.checker{background:
  linear-gradient(45deg,#2a2b30 25%,transparent 25%,transparent 75%,#2a2b30 75%),
  linear-gradient(45deg,#2a2b30 25%,#0f1013 25%,#0f1013 75%,#2a2b30 75%);
  background-size:8px 8px;background-position:0 0,4px 4px}
.sw{width:30px;height:17px;border-radius:9px;background:#2a2b30;position:relative;display:inline-block;
  flex:0 0 auto;cursor:pointer}
.sw i{position:absolute;top:2px;left:2px;width:13px;height:13px;border-radius:50%;background:var(--dim);transition:.15s}
.sw.on{background:rgba(74,222,128,.28)} .sw.on i{left:15px;background:var(--acc3)}
button.mini{background:var(--panel2);border:1px solid var(--line);color:var(--mute);border-radius:var(--r-xs);
  font:11.5px var(--sans);padding:3px 8px;cursor:pointer}

/* shared control bar (A, D) */
.ctlbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 10px}
.chips{display:flex;gap:5px;flex-wrap:wrap}
.chip{background:var(--panel2);border:1px solid var(--line);color:var(--mute);border-radius:99px;
  padding:3px 11px;font:11.5px var(--sans);cursor:pointer;display:inline-flex;gap:6px;align-items:center}
.chip:hover{border-color:#3a3d45;color:var(--ink)}
.chip b{font:400 10.5px var(--mono);color:var(--dim)}
.chip.on{background:rgba(56,189,248,.14);color:var(--acc2);border-color:rgba(56,189,248,.45)}
.chip.on b{color:var(--acc2)}
.tgl{background:var(--panel2);border:1px solid var(--line);color:var(--mute);border-radius:var(--r-sm);
  padding:4px 10px;font:11.5px var(--sans);cursor:pointer}
.tgl:hover{border-color:#3a3d45;color:var(--ink)}
.tgl.on{background:rgba(74,222,128,.12);color:var(--acc3);border-color:rgba(74,222,128,.42)}
.readout{font:11px var(--mono);color:var(--dim)}

/* ---------------------------------------------------------------- 0 baseline */
.bl{border:1px solid var(--line);border-radius:var(--r);overflow:hidden;background:var(--panel)}
.bl-group{display:flex;align-items:center;gap:9px;padding:9px 12px;background:var(--panel2);
  border-bottom:1px solid var(--line);font-size:12.5px;font-weight:650}
.bl-group .path{font:11.5px var(--mono);color:var(--dim)}
.bl-group .cnt{margin-left:auto;font:11px var(--mono);color:var(--dim)}
.bl-group.closed{color:var(--mute);font-weight:500}
.bl-row{display:grid;grid-template-columns:minmax(190px,min(34%,380px)) minmax(0,1fr);
  align-items:center;gap:6px 16px;padding:7px 12px;border-bottom:1px solid var(--line)}
.bl-row.imp{background:rgba(74,222,128,.045);box-shadow:inset 2px 0 0 var(--acc3)}
.bl-lbl{display:flex;align-items:center;gap:8px;font-size:13px}
.bl-ctl{display:flex} .bl-ctl .f{width:100%}
.bl-desc{grid-column:2;font-size:11.5px;color:var(--dim);line-height:1.45;margin-top:-1px}

/* ---------------------------------------------------------------- A columns */
.cg{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:0 20px;
  border:1px solid var(--line);border-radius:var(--r);background:var(--panel);padding:0 12px 10px;
  align-content:start}
.cg.cg-2{grid-template-columns:repeat(auto-fill,minmax(340px,1fr));border:0;padding:0;background:none}
.cg-hd{grid-column:1/-1;display:flex;align-items:center;gap:9px;margin:12px 0 5px;
  font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--mute);font-weight:700}
.cg-hd::after{content:"";flex:1 1 auto;height:1px;background:var(--line)}
.cg-hd .cnt{font:10.5px var(--mono);color:var(--dim);order:3}
.cg-row{display:flex;align-items:center;gap:8px;height:28px;border-bottom:1px solid rgba(42,43,48,.55)}
.cg-row:hover{background:var(--hover)}
.cg-row.imp{box-shadow:inset 2px 0 0 var(--acc3);padding-left:6px;margin-left:-6px}
.cg-row.imp .cg-lbl{color:var(--ink)}
.cg-lbl{font-size:12.5px;color:var(--mute);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  flex:1 1 auto;min-width:0}
.cg-ctl{flex:0 0 auto;display:flex;max-width:58%}
.cg-help{display:none;font-size:11px;color:var(--dim);line-height:1.4;padding:0 0 6px;
  border-bottom:1px solid rgba(42,43,48,.55);margin-top:-1px}
.cg.showhelp .cg-help{display:block}

/* ---------------------------------------------------------------- B rail */
.md{display:grid;grid-template-columns:212px minmax(0,1fr);gap:14px;
  border:1px solid var(--line);border-radius:var(--r);background:var(--panel);overflow:hidden;
  min-height:430px}
.md-rail{background:var(--panel3);border-right:1px solid var(--line);padding:10px 0;display:flex;
  flex-direction:column;gap:1px;max-height:640px;overflow:auto}
.md-railhd{font:10.5px var(--mono);color:var(--dim);padding:0 12px 8px}
.md-search{margin:0 10px 8px;padding:5px 8px;background:#0f1013;border:1px solid var(--line);
  border-radius:var(--r-sm);font-size:12px;color:var(--ink);width:calc(100% - 20px)}
.md-item{display:flex;align-items:center;gap:8px;padding:6px 12px;font-size:12.5px;color:var(--mute);
  cursor:pointer;border-left:2px solid transparent}
.md-item:hover{background:var(--hover);color:var(--ink)}
.md-item.open{color:var(--ink);font-weight:650;background:var(--panel2)}
.md-item .n,.md-sub .n{margin-left:auto;font:10.5px var(--mono);color:var(--dim)}
.md-sub{display:flex;align-items:center;gap:8px;padding:5px 12px 5px 24px;font-size:12px;
  color:var(--mute);cursor:pointer;border-left:2px solid transparent}
.md-sub:hover{background:var(--hover);color:var(--ink)}
.md-sub.on{background:rgba(56,189,248,.1);color:var(--acc2);border-left-color:var(--acc2);font-weight:650}
.md-sub.on .n{color:var(--acc2)}
.md-item .chg,.md-railft .chg{width:5px;height:5px;border-radius:50%;background:var(--acc);flex:0 0 auto}
.md-railft{margin-top:auto;padding:10px 12px 0;border-top:1px solid var(--line);
  font:11px var(--mono);color:var(--dim);display:flex;align-items:center;gap:6px}
.md-pane{padding:14px 16px 16px 2px;min-width:0;display:flex;flex-direction:column}
.md-hd{display:flex;align-items:baseline;gap:10px;margin-bottom:12px}
.md-hd .cnt{font:11px var(--mono);color:var(--dim)}
.md-ends{margin-top:auto;padding-top:18px;font:11px var(--mono);color:var(--dim);line-height:1.6}

/* ---------------------------------------------------------------- C table */
.pt-bar{display:flex;align-items:center;gap:7px;margin-bottom:9px;flex-wrap:wrap}
.pt-f{font:11.5px var(--sans);color:var(--mute);background:var(--panel2);border:1px solid var(--line);
  border-radius:99px;padding:3px 10px;cursor:pointer}
.pt-f b{font-family:var(--mono);color:var(--dim);margin-left:5px;font-weight:400}
.pt-f.on{background:rgba(56,189,248,.14);color:var(--acc2);border-color:rgba(56,189,248,.45)}
.pt-f.on b{color:var(--acc2)}
.pt-q{flex:1 1 260px;max-width:430px;padding:4px 10px;background:#0f1013;border:1px solid var(--line);
  border-radius:var(--r-sm);color:var(--ink);font-size:12.5px}
.pt-sort{font:11.5px var(--mono);color:var(--dim)}
.pt-wrap{border:1px solid var(--line);border-radius:var(--r);overflow:auto;background:var(--panel);
  max-height:640px}
.pt{width:100%;border-collapse:collapse;font-size:12.5px}
.pt thead th{position:sticky;top:0;background:var(--panel2);border-bottom:1px solid var(--line);
  text-align:left;padding:7px 10px;font:650 10.5px/1 var(--sans);letter-spacing:.08em;
  text-transform:uppercase;color:var(--mute);z-index:2}
.pt thead th.th-s{cursor:pointer;user-select:none}
.pt thead th.th-s:hover{color:var(--acc2)}
.pt tbody tr{height:26px;border-bottom:1px solid rgba(42,43,48,.5)}
.pt tbody tr:hover{background:var(--hover)}
.pt tbody tr.chg{box-shadow:inset 2px 0 0 var(--acc)}
.pt td{padding:0 10px;vertical-align:middle}
.c-dot{width:30px;padding-left:11px!important;white-space:nowrap}
.c-name{display:flex;align-items:baseline;gap:12px}
.c-name .k{color:var(--ink);flex:0 0 auto}
.c-name .p{font:10.5px var(--mono);color:var(--dim);margin-left:auto;text-align:right;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.c-val{width:240px} .c-val .f{width:100%}
.c-def{width:110px;font:11.5px var(--mono);color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.c-grp{width:150px;font-size:11.5px;color:var(--mute);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* ---------------------------------------------------------------- D inspector */
.ins{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:14px;align-items:start}
.ins-list{border:1px solid var(--line);border-radius:var(--r);background:var(--panel);overflow:auto;
  max-height:600px;outline:none}
.ins-list:focus{border-color:rgba(56,189,248,.45)}
.ins-grp{position:sticky;top:0;background:var(--panel2);border-bottom:1px solid var(--line);
  padding:4px 11px;font:700 10px/1.3 var(--sans);letter-spacing:.09em;text-transform:uppercase;
  color:var(--mute);z-index:2}
.ins-row{display:flex;align-items:center;gap:9px;height:24px;padding:0 11px;font-size:12.5px;
  border-bottom:1px solid rgba(42,43,48,.5);cursor:pointer}
.ins-row:hover{background:var(--hover)}
.ins-row.on{background:rgba(56,189,248,.13);box-shadow:inset 2px 0 0 var(--acc2)}
.ins-row.chg .ins-v{color:var(--acc)}
.ins-lbl{flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--mute)}
.ins-row.on .ins-lbl{color:var(--ink)}
.ins-v{font:11.5px var(--mono);color:var(--dim);max-width:44%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ins-doc{border:1px solid var(--line);border-radius:var(--r);background:var(--panel);padding:14px;
  position:sticky;top:8px}
.ins-kicker{font:10.5px var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--acc);font-weight:700}
.ins-doc h4{margin:5px 0 2px}
.ins-path{font:11px var(--mono);color:var(--dim);margin-bottom:12px;word-break:break-all}
.ins-bigctl{display:flex;align-items:center;gap:8px;margin-bottom:12px}
.ins-bigctl .f{flex:1 1 auto}
.ins-help{font-size:12.5px;color:var(--mute);line-height:1.6;margin:0 0 13px}
.ins-meta{border-top:1px solid var(--line);padding-top:9px;display:flex;flex-direction:column;gap:4px}
.ins-m{display:flex;font:11.5px var(--mono);gap:8px}
.ins-m span{color:var(--dim);width:58px;flex:0 0 auto}
.ins-m b{color:var(--mute);font-weight:400;word-break:break-all;min-width:0}

/* ---------------------------------------------------------------- E workbench */
.wb-omni{display:flex;align-items:center;gap:10px;background:var(--panel);border:1px solid var(--line);
  border-radius:var(--r);padding:9px 14px}
.wb-omni .mag{color:var(--dim);font-size:15px}
.wb-q{flex:1 1 auto;background:transparent;border:0;outline:none;color:var(--ink);
  font:14px var(--mono);padding:2px 0}
.wb-omni .kbd{font:11px var(--mono);color:var(--dim);border:1px solid var(--line);
  border-radius:var(--r-xs);padding:2px 6px}
.wb-hits{border:1px solid var(--line);border-top:0;border-radius:0 0 var(--r) var(--r);
  background:var(--panel3)}
.wb-hit{display:flex;align-items:center;gap:10px;padding:5px 14px;font-size:12.5px;
  border-bottom:1px solid rgba(42,43,48,.45)}
.wb-hit:first-child{background:rgba(56,189,248,.09)}
.wb-hit .k{color:var(--ink);flex:0 0 auto} .wb-hit .p{font:10.5px var(--mono);color:var(--dim)}
.wb-hit .f{margin-left:auto;flex:0 0 auto}
.wb-hit .g{font-size:11px;color:var(--mute);width:104px;text-align:right;flex:0 0 auto}
.wb-more{padding:6px 14px;font:11px var(--mono);color:var(--dim)}
.wb-sec{display:flex;align-items:center;gap:10px;margin:18px 0 8px;font:700 11px/1 var(--sans);
  letter-spacing:.09em;text-transform:uppercase;color:var(--mute)}
.wb-sec .cnt{font:10.5px var(--mono);color:var(--dim);letter-spacing:0;text-transform:none}
.wb-sec::after{content:"";flex:1 1 auto;height:1px;background:var(--line)}
.wb-board{display:grid;grid-template-columns:repeat(auto-fill,minmax(235px,1fr));gap:8px}
.wb-card{background:var(--panel);border:1px solid var(--line);border-radius:var(--r-sm);padding:8px 10px;
  display:flex;flex-direction:column;gap:6px}
.wb-cl{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--mute)}
.wb-card .f{width:100%}
.sw-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px}
.sw-tile{display:flex;align-items:center;gap:8px;background:var(--panel);border:1px solid var(--line);
  border-radius:var(--r-sm);padding:5px 8px;font-size:11.5px;color:var(--mute)}
.sw-tile:hover{border-color:#3a3d45;color:var(--ink)}
.sw-tile i{width:16px;height:16px;border-radius:var(--r-xs);border:1px solid var(--line);flex:0 0 auto}
.sw-tile span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bmx-wrap{display:flex;flex-wrap:wrap;gap:6px}
.bmx{font-size:11.5px;color:var(--dim);background:var(--panel);border:1px solid var(--line);
  border-radius:99px;padding:3px 11px;cursor:pointer}
.bmx.on{color:var(--acc3);border-color:rgba(74,222,128,.4);background:rgba(74,222,128,.09)}
.bmx.on::before{content:"\2713 "}
"""

SHELL_CSS = r"""
.wrap{max-width:1560px;margin:0 auto;padding:22px 24px 60px}
.hd{display:flex;align-items:flex-end;gap:16px;margin-bottom:6px;flex-wrap:wrap}
.hd h1{font-size:1.55rem;margin:0;letter-spacing:-.022em}
.lede{color:var(--mute);font-size:13px;max-width:98ch;margin:6px 0 20px;line-height:1.65}
.lede b{color:var(--ink)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:9px;margin:0 0 22px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:var(--r-sm);padding:10px 12px}
.stat .v{font:700 20px/1.15 var(--mono);color:var(--acc)}
.stat .l{font-size:11px;color:var(--dim);margin-top:3px;line-height:1.4}
.switch-bar{display:flex;gap:7px;flex-wrap:wrap;margin:0 0 4px;position:sticky;top:0;z-index:20;
  background:linear-gradient(var(--bg) 76%,transparent);padding:10px 0 12px}
.sw-btn{display:flex;align-items:center;gap:9px;background:var(--panel);border:1px solid var(--line);
  border-radius:var(--r-sm);padding:7px 12px;cursor:pointer;font:inherit;color:var(--mute);text-align:left}
.sw-btn:hover{border-color:#3a3d45;color:var(--ink)}
.sw-btn.on{background:rgba(56,189,248,.12);border-color:rgba(56,189,248,.5);color:var(--acc2)}
.sw-btn .id{font:700 12px/1 var(--mono);color:var(--dim);border:1px solid var(--line);border-radius:var(--r-xs);
  padding:3px 5px;flex:0 0 auto}
.sw-btn.on .id{color:var(--acc2);border-color:rgba(56,189,248,.5)}
.sw-btn .nm{font-size:12.5px;font-weight:650;display:block}
.sw-btn .sub{font-size:11px;color:var(--dim);display:block;margin-top:1px}
.sw-btn .dens{margin-left:8px;font:700 13px/1 var(--mono);color:var(--acc3)}
.sw-btn .dens small{display:block;font:400 9.5px/1.3 var(--sans);color:var(--dim);margin-top:2px}
.modebar{display:flex;margin:14px 0 12px;align-items:center}
.mode{background:var(--panel2);border:1px solid var(--line);color:var(--mute);padding:5px 13px;
  font:650 12px var(--sans);cursor:pointer}
.mode:first-child{border-radius:var(--r-sm) 0 0 var(--r-sm)}
.mode:last-child{border-radius:0 var(--r-sm) var(--r-sm) 0;margin-left:-1px}
.mode.on{background:rgba(56,189,248,.14);color:var(--acc2);border-color:rgba(56,189,248,.45);z-index:1}
.rationale{background:var(--panel3);border:1px solid var(--line);border-radius:var(--r-sm);
  padding:11px 14px;margin:0 0 16px;font-size:12.5px;color:var(--mute);line-height:1.65}
.rationale .kicker{margin-right:8px}
.rationale b{color:var(--ink)}
.bars{margin:26px 0 0;border-top:1px solid var(--line);padding-top:18px}
.bar{display:flex;align-items:center;gap:11px;margin-bottom:7px;font-size:12.5px}
.bar .bn{width:160px;flex:0 0 auto;color:var(--mute)}
.bar .bt{height:15px;border-radius:3px;background:var(--acc2);opacity:.75;flex:0 0 auto}
.bar.base .bt{background:var(--bad)}
.bar.alt .bt{background:repeating-linear-gradient(90deg,var(--vio) 0 7px,transparent 7px 11px)}
.bar .bv{font:12px var(--mono);color:var(--dim);white-space:nowrap}
.foot{margin-top:34px;padding-top:16px;border-top:1px solid var(--line);font-size:12px;color:var(--dim);line-height:1.7}
.pane{display:none} .pane.on{display:block}
.viewport-note{font:11px var(--mono);color:var(--dim);margin:0 0 9px}
.live{display:inline-flex;align-items:center;gap:6px;font:10.5px var(--mono);color:var(--acc3);
  border:1px solid rgba(74,222,128,.35);background:rgba(74,222,128,.08);border-radius:99px;padding:2px 9px}
"""


def page(title: str, body: str, data_js: str, extra_css: str = "") -> str:
    app = (HERE / "_app.js").read_text()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{CSS}{extra_css}</style>
</head><body>{body}
<script>{data_js}</script>
<script>{app}</script>
</body></html>
"""


def build() -> None:
    corpus = json.loads((HERE / "settings.json").read_text())
    corpus["title"] = "LM3_settings.yaml"
    data_js = ("const DATA=" + json.dumps(corpus, separators=(",", ":"))
               + ";const TOOL=" + json.dumps(tool_dataset(), separators=(",", ":")) + ";")
    n = len(corpus["rows"])

    def mount(layout: str, src: str) -> str:
        return f'<div class="mount" data-layout="{layout}" data-src="{src}"></div>'

    # ---- standalone files -------------------------------------------------
    for i, (slug, ident, name, tag, dens, dens_l, why, live) in enumerate(PANES):
        if slug == "baseline":
            continue
        body = f"""
<div class="wrap">
  <div class="hd"><span class="kicker">Layout {ident}</span><h1>{name}</h1>
    <span class="live">&#9679; interactive</span></div>
  <p class="lede">{why}</p>
  <div class="rationale"><span class="kicker">density</span>
    <b style="color:var(--acc3)">{dens}</b> {dens_l} in the 515px form viewport, against 8 today.</div>
  <div class="viewport-note">&#9660; SETTINGS &mdash; LM3_settings.yaml &middot; {n} real settings</div>
  {mount(slug, "settings")}
  <div class="viewport-note" style="margin-top:26px">&#9660; POSTPROCESSING &mdash; leaf collage builder &middot; 29 inputs</div>
  {mount(slug, "tool")}
</div>"""
        (HERE / f"{i:02d}-{slug}.html").write_text(
            page(f"LM3 layout {ident} — {name}", body, data_js, SHELL_CSS))

    # ---- comparison shell -------------------------------------------------
    btns, panes, bars = [], [], []
    for i, (slug, ident, name, tag, dens, dens_l, why, live) in enumerate(PANES):
        on = " on" if i == 1 else ""
        btns.append(f"""<button class="sw-btn{on}" data-p="{slug}">
      <span class="id">{ident}</span>
      <span><span class="nm">{name}</span><span class="sub">{tag}</span></span>
      <span class="dens">{dens}<small>{dens_l}</small></span></button>""")
        if slug == "baseline":
            inner = mount(slug, "settings")
        else:
            inner = (f'<div class="modebar"><button class="mode on" data-m="settings">Settings &middot; {n}</button>'
                     f'<button class="mode" data-m="tool">Postprocessing &middot; 29</button></div>'
                     f'<div class="sub-pane" data-m="settings">{mount(slug, "settings")}</div>'
                     f'<div class="sub-pane" data-m="tool" style="display:none">{mount(slug, "tool")}</div>')
        panes.append(f"""<section class="pane{on}" data-p="{slug}">
      <div class="rationale"><span class="kicker">{ident}</span>{why}</div>
      {inner}</section>""")
        alt = slug == "workbench"
        w = 0 if alt else min(100, round(float(dens) / 55 * 82))
        cls = "base" if slug == "baseline" else ("alt" if alt else "")
        bar = ('<span class="bt" style="width:70%"></span>' if alt
               else f'<span class="bt" style="width:{max(3, w)}%"></span>')
        note = " (not a scroll metric &mdash; nothing is browsed)" if alt else ""
        bars.append(f'<div class="bar {cls}"><span class="bn">{ident} &middot; {name}</span>{bar}'
                    f'<span class="bv">{dens} {dens_l}{note}</span></div>')

    body = f"""
<div class="wrap">
  <div class="hd"><span class="kicker">experiment</span><h1>Denser settings layouts</h1>
    <span class="live">&#9679; every layout below is interactive</span></div>
  <p class="lede">Five alternatives to the LM3 Settings and Postprocessing forms, measured against what
  ships today. Each renders the <b>real corpus</b> &mdash; all {n} settings from
  <code>settings_meta.json</code> + <code>builtin_defaults()</code> + your current
  <code>LM3_settings.yaml</code>, and the leaf collage tool's 29 inputs &mdash; so the navigation each
  design depends on can actually be tried. <b>Nothing is saved</b>: toggles flip so the page feels
  live, but no other control edits and nothing talks to the server.</p>

  <div class="stats">
    <div class="stat"><div class="v">515px</div><div class="l">visible form area on a 1080p window with the perf panel open</div></div>
    <div class="stat"><div class="v">63.5px</div><div class="l">average row today &mdash; help text takes its own grid line</div></div>
    <div class="stat"><div class="v">~8</div><div class="l">settings on screen, of {n}</div></div>
    <div class="stat"><div class="v">22,068px</div><div class="l">full form expanded &asymp; 43 screens</div></div>
    <div class="stat"><div class="v">7,937px</div><div class="l">Overlays alone (103 settings) &asymp; 15.4 screens</div></div>
    <div class="stat"><div class="v">2,756px</div><div class="l">the collage tool's 29 inputs &asymp; 4.4 screens</div></div>
  </div>

  <div class="switch-bar">{''.join(btns)}</div>
  {''.join(panes)}

  <div class="bars">
    <div class="wb-sec" style="margin-top:0">Settings visible in one 515px viewport</div>
    {''.join(bars)}
  </div>

  <div class="foot">
    <b>Where the height goes today.</b> <code>.treerow</code> is a two-column grid, and
    <code>.desc</code> / <code>.err</code> are <code>grid-column:2</code> &mdash; each claims a whole
    extra grid row under the control. 102 of {n} settings are a 19px toggle in a 63.5px row, and a
    number input inherits <code>width:100%</code> so it spans ~1200px to hold four characters.
    85 group headers add 3,310px before a single setting is drawn.<br><br>
    <b>What already helps.</b> "Important only" hides 194 of {n} rows; "Descriptions" recovers
    6,293px (29%); the 10 category chips keep one section on screen. The Postprocessing tab has
    <i>none</i> of these &mdash; no search, no important-only, no descriptions toggle, and its help
    text always renders.<br><br>
    Rebuild with <code>.venv_LM3/bin/python exp__settingslayouts/_extract.py &gt; exp__settingslayouts/settings.json</code>
    then <code>python3 exp__settingslayouts/_build.py</code>.
  </div>
</div>
<script>
document.querySelectorAll(".sw-btn").forEach(function (b) {{
  b.addEventListener("click", function () {{
    document.querySelectorAll(".sw-btn").forEach(function (x) {{ x.classList.remove("on"); }});
    b.classList.add("on");
    document.querySelectorAll(".pane").forEach(function (p) {{
      p.classList.toggle("on", p.dataset.p === b.dataset.p);
    }});
    window.scrollTo({{ top: 0, behavior: "smooth" }});
  }});
}});
document.querySelectorAll(".mode").forEach(function (m) {{
  m.addEventListener("click", function () {{
    var pane = m.closest(".pane");
    pane.querySelectorAll(".mode").forEach(function (x) {{ x.classList.remove("on"); }});
    m.classList.add("on");
    pane.querySelectorAll(".sub-pane").forEach(function (s) {{
      s.style.display = s.dataset.m === m.dataset.m ? "block" : "none";
    }});
  }});
}});
</script>"""
    (HERE / "index.html").write_text(page("LM3 — denser settings layouts", body, data_js, SHELL_CSS))
    print("wrote:", ", ".join(sorted(p.name for p in HERE.glob("*.html"))))


if __name__ == "__main__":
    build()
