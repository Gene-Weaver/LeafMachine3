#!/usr/bin/env python3
"""Build the README demo composites in ``docs/readme_github/`` from a finished LM3 run.

Every image in the README comes from the same three specimens so it reads as one story. They are
sheets that ship in ``examples/images/``, so anyone can reproduce every figure by running LM3 on the
bundled examples. Which sheets and which leaves is the CONFIG block below -- swapping the README's
specimens is an edit there, not here. The composites are committed, so this script only needs to
run again when the Reporter's output changes or a different run should illustrate the README.

Usage::

    python tools/release/build_readme_images.py \
        [--run runs/for_readme] [--originals examples/images] \
        [--gui-shots <dir with gui_settings.png, gui_status_running.png, gui_console_running.png,
                      gui_results.png, gui_models.png, gui_postprocess.png>] \
        [--out docs/readme_github]

``--gui-shots`` is optional: the GUI screenshots are captured separately (headless Chrome over the
DevTools protocol against ``lm3 serve``) and are only re-encoded here. When the folder is absent
the committed GUI images are left untouched.

Only Pillow is required.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- CONFIG
# The three sheets (image stems in examples/images/), in figure order.
SPECIMENS = [
    "A_1998857721_Platanaceae_Platanus_kerrii",
    "PH_3865012217_Bignoniaceae_Catalpa_erubescens",
    "YPM_1038926111_Bignoniaceae_Catalpa_purpurea",
]
# (primary leaf, secondary leaf) per sheet, as the x1_y1_x2_y2 token in the Reporter's file names --
# chosen by eye from the Overlay_Landmarks contact sheets. The primary must have a petiole mask.
LEAVES = {
    "A_1998857721_Platanaceae_Platanus_kerrii": ["0_2_867_880", "969_1770_1564_2084"],
    "PH_3865012217_Bignoniaceae_Catalpa_erubescens": ["630_1666_1791_2598", "788_286_2069_1971"],
    "YPM_1038926111_Bignoniaceae_Catalpa_purpurea": ["947_53_1336_437", "774_83_971_389"],
}
# Leaves whose lamina has holes (leaf_morphology.n_holes > 0); any sheet, three rows.
HOLE_LEAVES = [
    ("A_1998857721_Platanaceae_Platanus_kerrii", "1533_156_2151_1831"),
    ("A_1998857721_Platanaceae_Platanus_kerrii", "850_990_1155_1840"),
    ("PH_3865012217_Bignoniaceae_Catalpa_erubescens", "630_1666_1791_2598"),
]
# Non-leaf organ crops for the Plant Detector figure: (sheet, Crops/RGB__<class>, caption).
ORGANS = [
    ("A_1998857721_Platanaceae_Platanus_kerrii", "fruit", "fruit"),
    ("PH_3865012217_Bignoniaceae_Catalpa_erubescens", "fruitMany", "fruit cluster"),
    ("YPM_1038926111_Bignoniaceae_Catalpa_purpurea", "flower", "flower"),
]
ARCHIVAL_SHEET = "A_1998857721_Platanaceae_Platanus_kerrii"     # the archival crop strip
REPORTER_LEAF = ("YPM_1038926111_Bignoniaceae_Catalpa_purpurea", "947_53_1336_437")
# Bilateral-symmetry panels: one leaf per sheet. The run only draws panels for the leaves its
# qc_images setting selects (by default the FLAGGED ones), so these are re-drawn here from the run
# database with the stage's own renderer -- the same picture the stage would write.
BSYM_LEAVES = [
    ("A_1998857721_Platanaceae_Platanus_kerrii", "0_2_867_880"),
    ("PH_3865012217_Bignoniaceae_Catalpa_erubescens", "788_286_2069_1971"),
    ("YPM_1038926111_Bignoniaceae_Catalpa_purpurea", "947_53_1336_437"),
]


def label(stem: str) -> str:
    """'PH_3865012217_Bignoniaceae_Catalpa_erubescens' -> 'Catalpa erubescens'."""
    parts = stem.split("_")
    return " ".join(parts[-2:]) if len(parts) >= 4 else stem


BG = (255, 255, 255)
GAP = 12
QUALITY = 85

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
except OSError:
    FONT = ImageFont.load_default()


# ---------------------------------------------------------------- helpers
def load(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def fit(im: Image.Image, *, h: int | None = None, w: int | None = None) -> Image.Image:
    if h is not None:
        return im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)
    return im.resize((w, max(1, round(im.height * w / im.width))), Image.LANCZOS)


def caption(im: Image.Image, text: str) -> Image.Image:
    """Add a one-line caption strip under an image, widening the cell so the text never clips."""
    pad = 34
    tw = ImageDraw.Draw(im).textlength(text, font=FONT)
    w = max(im.width, int(tw) + 16)
    out = Image.new("RGB", (w, im.height + pad), BG)
    out.paste(im, ((w - im.width) // 2, 0))
    ImageDraw.Draw(out).text(((w - tw) / 2, im.height + 6), text, fill=(40, 40, 40), font=FONT)
    return out


def row(ims: list[Image.Image], gap: int = GAP) -> Image.Image:
    h = max(i.height for i in ims)
    w = sum(i.width for i in ims) + gap * (len(ims) - 1)
    out = Image.new("RGB", (w, h), BG)
    x = 0
    for i in ims:
        out.paste(i, (x, (h - i.height) // 2))
        x += i.width + gap
    return out


def col(ims: list[Image.Image], gap: int = GAP) -> Image.Image:
    w = max(i.width for i in ims)
    h = sum(i.height for i in ims) + gap * (len(ims) - 1)
    out = Image.new("RGB", (w, h), BG)
    y = 0
    for i in ims:
        out.paste(i, ((w - i.width) // 2, y))
        y += i.height + gap
    return out


class Builder:
    def __init__(self, run: Path, originals: Path, out: Path) -> None:
        self.reports = run / "reports"
        self.originals = originals
        self.out = out
        out.mkdir(parents=True, exist_ok=True)

    def original(self, stem: str) -> Path:
        hits = sorted(p for p in self.originals.glob(f"{stem}.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        if not hits:
            raise FileNotFoundError(f"{stem}.* under {self.originals}")
        return hits[0]

    def one(self, pattern: str) -> str:
        hits = sorted(glob.glob(str(self.reports / pattern)))
        if not hits:
            raise FileNotFoundError(f"{pattern} under {self.reports}")
        return hits[0]

    def largest(self, pattern: str) -> str:
        hits = sorted(glob.glob(str(self.reports / pattern)), key=os.path.getsize)
        if not hits:
            raise FileNotFoundError(f"{pattern} under {self.reports}")
        return hits[-1]

    def save(self, im: Image.Image, name: str, max_w: int = 1600) -> None:
        if im.width > max_w:
            im = fit(im, w=max_w)
        p = self.out / name
        im.save(p, quality=QUALITY, optimize=True)
        print(f"{name:40s} {im.width}x{im.height}  {p.stat().st_size // 1024} KB")

    # ------------------------------------------------------------ composites
    def hero(self) -> None:
        originals = [caption(fit(load(self.original(s)), h=760), f"{label(s)}  original") for s in SPECIMENS]
        overlays = [caption(fit(load(self.one(f"Overlay/Overlay_Summary/{s}__Overlay.jpg")), h=760),
                            f"{label(s)}  LeafMachine3 overlay") for s in SPECIMENS]
        self.save(col([row(originals), row(overlays)]), "hero_before_after.jpg", max_w=1800)

    def archival_detector(self) -> None:
        cells = []
        for cls in ["ruler", "barcode", "label", "colorcard", "envelope"]:
            try:
                p = self.largest(f"Crops/RGB__{cls}/{ARCHIVAL_SHEET}__BBOX-{cls}__*.jpg")
            except FileNotFoundError:
                p = self.largest(f"Crops/RGB__{cls}/*.jpg")
            im = load(p)
            cells.append(caption(fit(im, h=200) if im.width >= im.height else fit(im, w=260), cls))
        self.save(row(cells), "archival_detector_crops.jpg")

    def plant_detector(self) -> None:
        cells = []
        for s in SPECIMENS:
            tok = LEAVES[s][0]
            cells.append(caption(fit(load(self.one(f"Leaf_Original/Leaf_BBox/{s}__og-BBOX-leaf__{tok}.jpg")), h=260), "leaf"))
        for s, cls, name in ORGANS:
            cells.append(caption(fit(load(self.largest(f"Crops/RGB__{cls}/{s}__*.jpg")), h=260), name))
        self.save(row(cells), "plant_detector_crops.jpg")

    def specimen_segmenter(self) -> None:
        seg = [caption(fit(load(self.one(f"Overlay/Overlay_Specimen_Segmentation/{s}__SpecimenSeg.jpg")), w=1500), label(s))
               for s in SPECIMENS]
        self.save(col(seg), "specimen_segmenter.jpg", max_w=1500)

    def ruler_classifier(self) -> None:
        def landscape(im: Image.Image) -> Image.Image:   # a ruler laid vertically on the sheet reads sideways
            return im.rotate(-90, expand=True) if im.height > im.width else im
        rulers = [caption(fit(landscape(load(self.one(f"Crops/RGB__ruler/{s}__BBOX-ruler__*.jpg"))), w=1300), label(s))
                  for s in SPECIMENS]
        self.save(col(rulers), "ruler_classifier_crops.jpg", max_w=1300)

    def ruler_cf(self) -> None:
        lat = [caption(fit(load(self.one(f"Overlay/Overlay_Ruler_Lattice/{s}__RulerLattice.png")), h=900), label(s))
               for s in SPECIMENS]
        self.save(row(lat), "ruler_cf_lattice.jpg", max_w=1800)

    def leaf_segmenter(self) -> None:
        rows = []
        for s in SPECIMENS:
            tok = LEAVES[s][0]
            ims = [load(self.one(f"Leaf_Original/Leaf_BBox/{s}__og-BBOX-leaf__{tok}.jpg")),
                   load(self.one(f"Leaf_Original/LaminaPetiole_Mask/{s}__og-SEG-laminaPetiole__{tok}.png")),
                   load(self.one(f"Leaf_Original/LaminaPetiole_RGB/{s}__og-RGB-laminaPetiole__{tok}.jpg"))]
            labels = ["Plant Detector crop", "lamina + petiole mask", "RGB cutout"]
            rows.append(row([caption(fit(i, h=360), l) for i, l in zip(ims, labels)]))
        self.save(col(rows), "leaf_segmenter.jpg")

    def morphology(self) -> None:
        rows = []
        for s, tok in HOLE_LEAVES:
            ims = [load(self.one(f"Leaf_Original/Lamina_RGB/{s}__og-RGB-lamina__{tok}.jpg")),
                   load(self.one(f"Leaf_Original/Lamina_Mask/{s}__og-SEG-lamina__{tok}.png")),
                   load(self.one(f"Leaf_Original/Lamina_Holes_Mask/{s}__og-SEG-laminaHoles__{tok}.png"))]
            labels = ["lamina RGB", "lamina mask (holes removed)", "lamina silhouette (holes filled)"]
            rows.append(row([caption(fit(i, h=320), l) for i, l in zip(ims, labels)]))
        self.save(col(rows), "morphology_holes.jpg")

    def landmarks(self) -> None:
        for idx, name in [(0, "landmark_detector.jpg"), (1, "landmark_measurements.jpg")]:
            cells = [caption(fit(load(self.one(f"Overlay/Overlay_Landmarks/{s}__LM-leaf__{LEAVES[s][idx]}.jpg")), h=560), label(s))
                     for s in SPECIMENS]
            self.save(row(cells), name, max_w=1800)

    def leaf_orientation(self) -> None:
        pairs = []
        for s in SPECIMENS:
            tok = LEAVES[s][0]
            a = load(self.one(f"Leaf_Original/Lamina_RGB/{s}__og-RGB-lamina__{tok}.jpg"))
            b = load(self.one(f"Leaf_Oriented/Lamina_RGB/{s}__or-RGB-lamina__{tok}.jpg"))
            pairs.append(row([caption(fit(a, h=360), "as mounted"), caption(fit(b, h=360), "oriented tip-up")], gap=6))
        self.save(row(pairs, gap=40), "leaf_orientation.jpg", max_w=1800)

    def petiole_width(self) -> None:
        cells = []
        for s in SPECIMENS:
            for tok in LEAVES[s]:  # not every leaf has a petiole mask; take the first that does
                hits = glob.glob(str(self.reports / f"Overlay/Overlay_Petiole/{s}__PET-leaf__{tok}.jpg"))
                if hits:
                    cells.append(caption(fit(load(hits[0]), h=420), label(s)))
                    break
        self.save(row(cells), "petiole_width.jpg", max_w=1800)

    def bilateral_symmetry(self) -> None:
        panels = [caption(fit(im, w=1400), label(st))
                  for st, im, score in self._bsym_panels()]
        self.save(col(panels), "bilateral_symmetry.jpg", max_w=1400)

    def _bsym_panels(self) -> list[tuple[str, Image.Image, float]]:
        """Draw each BSYM_LEAVES panel with the stage's renderer (core/bilateral + reporting/
        bilateral_viz), from its bilateral_symmetry row and the holes-filled silhouette the stage
        measured -- the Reporter's Leaf_Oriented/Lamina_Holes_Mask, same frame. Read-only on the run."""
        import sqlite3

        import numpy as np

        from leafmachine3.reporting.bilateral_viz import render_qc_panel

        dbs = sorted(self.reports.parent.glob("*.sqlite"))
        if not dbs:
            raise FileNotFoundError(f"no run database beside {self.reports}")
        db = sqlite3.connect(f"file:{dbs[0]}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        out = []
        for st, tok in BSYM_LEAVES:
            x1, y1, x2, y2 = (int(v) for v in tok.split("_"))
            rows = db.execute(
                "select b.* from bilateral_symmetry b join specimen s on s.specimen_id = b.specimen_id "
                "join plant_detection d on d.detection_id = b.detection_id where s.image_stem = ? "
                "and cast(round(d.x1) as int) = ? and cast(round(d.y1) as int) = ? "
                "and cast(round(d.x2) as int) = ? and cast(round(d.y2) as int) = ?", (st, x1, y1, x2, y2)).fetchall()
            if not rows:
                raise LookupError(f"no bilateral_symmetry row for {st} {tok}")
            row = rows[0]
            sil = np.asarray(load(self.one(f"Leaf_Oriented/Lamina_Holes_Mask/{st}__or-SEG-laminaHoles__{tok}.png")).convert("L")) > 127
            if sil.shape != (row["mask_h"], row["mask_w"]):
                raise ValueError(f"{st} {tok}: silhouette {sil.shape} is not the stored frame {(row['mask_h'], row['mask_w'])}")
            img = render_qc_panel(sil, row)
            if img is None:
                raise ValueError(f"{st} {tok}: the renderer returned nothing")
            out.append((st, Image.fromarray(img), float(row["archetype_score"])))
        return out

    def ect(self) -> None:
        rows = []
        for s in SPECIMENS:
            tok = LEAVES[s][0]
            ims = [load(self.one(f"Leaf_Oriented/Lamina_RGB/{s}__or-RGB-lamina__{tok}.jpg")),
                   load(self.one(f"Leaf_Data/Oriented_Leaf_ECT/{s}__ECT__{tok}.png")),
                   load(self.one(f"Leaf_Data/Oriented_Leaf_Radial_ECT/{s}__ECT-radial__{tok}.png")),
                   load(self.one(f"Leaf_Data/Oriented_Leaf_Radial_ECT_Overlay/{s}__ECT-radial-overlay__{tok}.png"))]
            labels = ["oriented leaf", "ECT (Cartesian)", "ECT (radial)", "radial ECT + outline"]
            rows.append(row([caption(fit(i, h=300), l) for i, l in zip(ims, labels)]))
        self.save(col(rows), "ect.jpg")

    def reporter(self) -> None:
        s, tok = REPORTER_LEAF
        products = [("Leaf_BBox", "BBOX-leaf", "jpg", "bbox crop"),
                    ("Lamina_Mask", "SEG-lamina", "png", "lamina mask"),
                    ("LaminaPetiole_Mask", "SEG-laminaPetiole", "png", "lamina+petiole mask"),
                    ("Lamina_Holes_Mask", "SEG-laminaHoles", "png", "lamina silhouette"),
                    ("Lamina_RGB", "RGB-lamina", "jpg", "lamina RGB"),
                    ("LaminaPetiole_RGB", "RGB-laminaPetiole", "jpg", "lamina+petiole RGB"),
                    ("Lamina_Holes_RGB", "RGB-laminaHoles", "jpg", "lamina-holes RGB")]
        trees = []
        for tree, tag in [("Leaf_Original", "og"), ("Leaf_Oriented", "or")]:
            cells = [caption(fit(load(self.one(f"{tree}/{folder}/{s}__{tag}-{pref}__{tok}.{ext}")), h=240), label)
                     for folder, pref, ext, label in products]
            trees.append(caption(row(cells), f"{tree}/"))
        self.save(col(trees, gap=30), "reporter_leaf_products.jpg", max_w=1800)

    def gui(self, shots: Path) -> None:
        self.save(load(shots / "gui_settings.png"), "gui_settings.jpg", max_w=1700)
        st = load(shots / "gui_status_running.png")
        # splice out the transient "model missing" banner (rows ~175..245) so the shot shows the run itself
        top, bottom = st.crop((0, 0, st.width, 172)), st.crop((0, 248, st.width, st.height))
        self.save(col([top, bottom], gap=0), "gui_live_status.jpg", max_w=1700)
        self.save(load(shots / "gui_console_running.png"), "gui_console.jpg", max_w=1700)
        for tab in ("results", "models", "postprocess"):  # captured on a finished run, models in sync with the lock
            self.save(load(shots / f"gui_{tab}.png"), f"gui_{tab}.jpg", max_w=1700)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, default=Path("runs/for_readme"),
                    help="a finished LM3 run directory (contains reports/)")
    ap.add_argument("--originals", type=Path, default=Path("examples/images"),
                    help="folder holding the three original specimen images")
    ap.add_argument("--out", type=Path, default=Path("docs/readme_github"))
    ap.add_argument("--gui-shots", type=Path, default=None,
                    help="folder of GUI PNG screenshots to re-encode (optional)")
    a = ap.parse_args()

    b = Builder(a.run, a.originals, a.out)
    for step in (b.hero, b.archival_detector, b.plant_detector, b.specimen_segmenter, b.ruler_classifier,
                 b.ruler_cf, b.leaf_segmenter, b.morphology, b.landmarks, b.leaf_orientation,
                 b.petiole_width, b.bilateral_symmetry, b.ect, b.reporter):
        step()
    if a.gui_shots and a.gui_shots.is_dir():
        b.gui(a.gui_shots)
    else:
        print("gui shots: skipped (no --gui-shots folder); committed GUI images kept")


if __name__ == "__main__":
    main()
