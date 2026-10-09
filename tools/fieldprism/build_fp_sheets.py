"""Build leafmachine3/inference/ruler_lattice/fieldprism_sheets.json from the FieldPrism source.

Geometry is COMPUTED from the FieldSheetBuilder code (PageInfo + xy_drawMarker + xy_draw10cm +
drawMarker/draw10cm), never hand-typed, then every square center is cross-checked against the
vector content of the PDFs the Android/iOS apps ship. The legacy Legal layout is parsed with `ast`
from `git show 2aba56a:QR_code_builder/build_PDF_utils.py` and cross-checked against the
manuscript PDF print_for_manuscript/Legal_Field_Sheet.pdf.

Needs the FieldPrism Python repo (a git checkout, for the legacy layout) and the Android app's
res/raw PDFs. Re-run only when the field sheets change; bump `catalog_version` when the output does.

    python tools/fieldprism/build_fp_sheets.py [--fieldprism DIR] [--app-raw DIR] [--out FILE]
"""
import argparse
import ast
import importlib.util
import json
import re
import subprocess
import sys
import zlib
from pathlib import Path

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--fieldprism", default="/home/brlab/Dropbox/FieldPrism",
                 help="FieldPrism Python repo (git checkout)")
_ap.add_argument("--app-raw", default="/home/brlab/Dropbox/FieldPrism_Anrdoid/app/src/main/res/raw",
                 help="Android app res/raw folder holding *_field_sheet.pdf")
_ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "leafmachine3" / "inference"
                                       / "ruler_lattice" / "fieldprism_sheets.json"))
_args = _ap.parse_args()

FP = Path(_args.fieldprism)
QB = FP / "QR_code_builder"
APP_RAW = Path(_args.app_raw)
OUT = Path(_args.out)
PT_PER_MM = 72.0 / 25.4

spec = importlib.util.spec_from_file_location("fp_pagesizes", QB / "build_PDF_PageSizes.py")
ps = importlib.util.module_from_spec(spec)
sys.modules["fp_pagesizes"] = ps
spec.loader.exec_module(ps)
PI = ps.PageInfo()

# FieldSheetBuilder drawMarker (build_PDF_utils.py:200-229): squares at these cell offsets (mm).
CELL = 10.0
FILLED = {"TL": (0, 0), "TR": (20, 0), "BL": (0, 20), "C": (10, 10)}
EMPTY_BR = (20, 20)


def square_centers(corner):
    x, y = corner
    out = {r: [x + dx + CELL / 2, y + dy + CELL / 2] for r, (dx, dy) in FILLED.items()}
    out["BR"] = [x + EMPTY_BR[0] + CELL / 2, y + EMPTY_BR[1] + CELL / 2]
    return {k: out[k] for k in ("TL", "TR", "C", "BL", "BR")}


def pdf_content(path):
    b = Path(path).read_bytes()
    mb = [float(v) for v in re.search(rb"/MediaBox\s*\[([^\]]*)\]", b).group(1).split()]
    s = re.search(rb"stream\r?\n(.*?)endstream", b, re.S).group(1)
    try:
        c = zlib.decompress(s).decode("latin1")
    except zlib.error:
        c = s.decode("latin1")
    return mb, c


def pdf_squares_mm(path):
    """Centers (mm, page top-left origin, y down) of every filled `re B` rect in the PDF."""
    mb, c = pdf_content(path)
    H = mb[3]
    out = []
    for m in re.finditer(r"([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) re B", c):
        x, y, w, h = map(float, m.groups())
        cx = (x + w / 2) / PT_PER_MM
        cy = (H - (y + h / 2)) / PT_PER_MM
        out.append((cx, cy, abs(w) / PT_PER_MM))
    bars = []
    for m in re.finditer(r"([-\d.]+) ([-\d.]+) m ([-\d.]+) ([-\d.]+) l S", c):
        x0, y0, x1, y1 = map(float, m.groups())
        bars.append((x0 / PT_PER_MM, (H - y0) / PT_PER_MM, x1 / PT_PER_MM))
    texts = re.findall(r"Td \((.*?)\)", c)
    return (mb[2] / PT_PER_MM, mb[3] / PT_PER_MM), out, bars, texts


def legacy_tables():
    """Parse x_pagesize/y_pagesize per PAGESIZE out of the 2022-12-02 drawMarker."""
    src = subprocess.run(["git", "-C", str(FP), "show", "2aba56a:QR_code_builder/build_PDF_utils.py"],
                         capture_output=True, text=True, check=True).stdout
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "drawMarker")
    tab = {"x_pagesize": {}, "y_pagesize": {}}
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) \
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "PAGESIZE":
            key = node.test.comparators[0].value
            for st in node.body:
                if isinstance(st, ast.Assign) and st.targets[0].id in tab:
                    tab[st.targets[0].id][key] = st.value.value
    consts = {}
    for st in fn.body:
        if isinstance(st, ast.Assign) and isinstance(st.value, ast.Constant):
            consts[st.targets[0].id] = st.value.value
    lines = src.splitlines()
    first = fn.lineno
    return tab, consts, first, fn.end_lineno


SHEETS = [  # (key, get_dim alias, app PDF, fpdf page_mm source)
    ("A5", "A5", "a5_field_sheet.pdf"),
    ("A4", "A4", "a4_field_sheet.pdf"),
    ("A3", "A3", "a3_field_sheet.pdf"),
    ("Letter", "Letter", "letter_field_sheet.pdf"),
    ("Legal", "Legal", "legal_field_sheet.pdf"),
    ("Tabloid", "Tabloid", "tabloid_field_sheet.pdf"),
]


def check(sheet, pdf_path):
    page_mm, sq, bars, texts = pdf_squares_mm(pdf_path)
    want = []
    for corner in ("TL", "TR", "BL", "BR"):
        for role in ("TL", "TR", "C", "BL"):
            want.append(sheet["square_centers_mm"][corner][role])
    assert len(sq) == 16, (pdf_path, len(sq))
    errs = []
    for wx, wy in want:
        d = min(((wx - x) ** 2 + (wy - y) ** 2) ** 0.5 for x, y, _ in sq)
        errs.append(d)
    edge = max(abs(e - 10.0) for _, _, e in sq)
    bar_err = max(min(abs(by - b["y"]) for _, by, _ in bars) for b in sheet["scale_bar_mm"].values())
    return {
        "file": str(pdf_path),
        "media_box_mm": [round(page_mm[0], 2), round(page_mm[1], 2)],
        "n_squares": len(sq),
        "max_center_err_mm": round(max(errs), 4),
        "max_edge_err_mm": round(edge, 4),
        "max_bar_y_err_mm": round(bar_err, 4),
        "caption": texts[0] if texts else None,
    }, page_mm


def build_sheet(key, W, H, page_mm, label, top_y, bottom_y, legacy=False, x_right=None, y_bottom=None):
    xl, yt = 20, 23
    xr = W - 50 if x_right is None else x_right
    yb = H - 54 if y_bottom is None else y_bottom
    corners = {"TL": [xl, yt], "TR": [xr, yt], "BL": [xl, yb], "BR": [xr, yb]}
    www_x = 75 if key == "A5" else 80
    sheet = {
        "label": label,
        "legacy": legacy,
        "family": "Legal" if key.startswith("Legal") else key,
        "page_mm": [round(page_mm[0], 1), round(page_mm[1], 1)],
        "layout_mm": [W, H],
        "marker_corner_mm": corners,
        "square_centers_mm": {c: square_centers(xy) for c, xy in corners.items()},
        "delta_x_mm": xr - xl,
        "delta_y_mm": yb - yt,
        "scale_bar_mm": {
            "top": {"x0": 20.4, "x1": 120.0, "y": top_y, "line_width": 1.0, "cap": "projecting"},
            "bottom": {"x0": 20.4, "x1": 120.0, "y": bottom_y, "line_width": 1.0, "cap": "projecting"},
        },
        "text_mm": {
            pos: [
                {"text": "10cm - <PAGESIZE config string>", "x": 35.0, "baseline_y": y - 2.0,
                 "font": "Helvetica-Bold", "size_pt": 16},
                {"text": "www.FieldPrism.org", "x": float(www_x), "baseline_y": y - 2.0,
                 "font": "Helvetica-Bold", "size_pt": 6},
            ]
            for pos, y in (("top", top_y), ("bottom", bottom_y))
        },
    }
    return sheet


def main():
    sheets = {}
    for key, alias, pdf in SHEETS:
        W, H = PI.get_dim(alias)
        x, y = PI.xy_drawMarker("bottom", "right", alias)
        assert (x, y) == (W - 50, H - 54)
        assert PI.xy_drawMarker("top", "left", alias) == (20, 23)
        page_mm, _, _, _ = pdf_squares_mm(APP_RAW / pdf)
        s = build_sheet(key, W, H, page_mm, key, PI.xy_draw10cm("top", alias), PI.xy_draw10cm("bottom", alias))
        chk, _ = check(s, APP_RAW / pdf)
        s["pdf_check"] = chk
        assert chk["max_center_err_mm"] < 0.01 and chk["max_bar_y_err_mm"] < 0.01, chk
        sheets[key] = s

    # Legacy Legal (FieldPrism commit 2aba56a 2022-12-02 .. replaced by PageInfo on 2023-01-17).
    tab, consts, l0, l1 = legacy_tables()
    lx, ly = tab["x_pagesize"]["Legal"], tab["y_pagesize"]["Legal"]
    assert consts["y_init_top"] == 23 and consts["x_init_top"] == 20
    for k in ("A3", "A4", "A5", "Custom"):  # every other legacy size equals today's layout
        W, H = PI.get_dim(k)
        assert (tab["x_pagesize"][k], tab["y_pagesize"][k]) == (W - 70, H - 77), k
    man = QB / "bin_PDF/print_for_manuscript/Legal_Field_Sheet.pdf"
    page_mm, _, _, _ = pdf_squares_mm(man)
    s = build_sheet("Legal_legacy", 20 + lx + 50, 23 + ly + 54, page_mm, "Legal (legacy)",
                    55, 55 + ly, legacy=True, x_right=20 + lx, y_bottom=23 + ly)
    s["layout_mm"] = None  # legacy drawMarker hard-coded the spacing; there was no PageInfo
    chk, _ = check(s, man)
    assert chk["max_center_err_mm"] < 0.01, chk
    s["pdf_check"] = chk
    s["legacy_source"] = {
        "commit": "2aba56a (2022-12-02 'adding fieldprism'); unchanged in 2984029 (2023-01-11, which only added the 'legal'/'L' aliases); "
                  "replaced by PageInfo in 09127d0/6b22948 (2023-01-17)",
        "file": "QR_code_builder/build_PDF_utils.py",
        "lines": f"{l0}-{l1} (drawMarker: Legal x_pagesize={lx}, y_pagesize={ly})",
        "bar_rule": "draw10cm in the same commit: y_init_top=55, bottom = 55 + y_pagesize",
    }
    sheets["Legal_legacy"] = s

    catalog = {
        "catalog_version": "fp-sheets-2026.10.09",
        "description": "Literal FieldPrism field-sheet geometry (FieldSheetBuilder PDFs) for sheet-type "
                       "identification from the photogrammetric markers. Generated by computing the "
                       "builder code, not hand-typed; every square center was cross-checked against "
                       "the PDFs bundled in the Android/iOS apps (byte-identical between platforms).",
        "generator": "tools/fieldprism/build_fp_sheets.py",
        "units": "mm",
        "origin": "page top-left corner, x right, y down (fpdf unit='mm', orientation='P')",
        "provenance": {
            "page_info": "FieldPrism/QR_code_builder/build_PDF_PageSizes.py:4-38 (PageInfo integer mm), "
                         ":40-75 (get_dim; 'Custom' -> A4)",
            "marker_layout": "build_PDF_PageSizes.py:77-101 (xy_drawMarker)",
            "scale_bar_layout": "build_PDF_PageSizes.py:103-125 (xy_draw10cm)",
            "marker_drawing": "FieldPrism/QR_code_builder/build_PDF_utils.py:197-229 (draw_1cm, drawMarker)",
            "scale_bar_drawing": "build_PDF_utils.py:231-247 (draw10cm; line width 1 mm, fpdf '2 J' cap)",
            "page_assembly": "build_PDF_utils.py:499-529 (newPage: 4 drawMarker calls, draw10cm SPACE=6)",
            "cross_check_pdfs": "FieldPrism_Anrdoid/app/src/main/res/raw/*_field_sheet.pdf (== "
                                "FieldPrism_iOS/.../Resources/*_field_sheet.pdf, md5-identical)",
            "app_spacing_tables": "RulerHomography.kt:18-36, RulerProcessor.swift:24-29, Constants.swift:57-62",
        },
        "marker_design": {
            "grid": "3x3 cells of 10 mm (30 x 30 mm marker)",
            "cell_mm": CELL,
            "filled_cells": {"TL": [0, 0], "TR": [2, 0], "C": [1, 1], "BL": [0, 2]},
            "empty_cell_BR": [2, 2],
            "square_centers_rel_corner_mm": square_centers((0, 0)),
            "square_pitch_mm": 20.0,
            "orientation_note": "all four markers on a sheet are TRANSLATED copies with the same "
                                "orientation; the empty BR cell fixes the sheet's orientation",
        },
        "layout_rule": {
            "marker_corner_x_mm": "20 (left markers) | W - 50 (right markers)",
            "marker_corner_y_mm": "23 (top markers) | H - 54 (bottom markers)",
            "delta_x_mm": "W - 70 (same-role squares, left -> right)",
            "delta_y_mm": "H - 77 (same-role squares, top -> bottom)",
            "W_H": "INTEGER PageInfo mm (layout_mm), not the physical page (page_mm)",
            "scale_bar_y_mm": "55 (top) | H - 22 (bottom); x 20.4..120",
        },
        "excluded": {
            "Custom": "identical geometry to A4 (get_dim 'Custom' -> A4); user-defined spacing "
                      "cannot be told apart, so it is not a separate entry",
            "credit_card": "xy_drawMarker_credit_card is the Size Check page, not a field sheet",
        },
        "sheets": sheets,
    }
    OUT.write_text(json.dumps(catalog, indent=2) + "\n")
    for k, s in sheets.items():
        print(k, s["delta_x_mm"], s["delta_y_mm"], s["page_mm"], s["pdf_check"]["max_center_err_mm"],
              s["pdf_check"]["caption"])


if __name__ == "__main__":
    main()
