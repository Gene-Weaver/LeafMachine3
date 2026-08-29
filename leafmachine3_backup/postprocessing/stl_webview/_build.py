#!/usr/bin/env python3
"""Assemble the single-file 3D generator page.

Everything -- CSS, three JS modules, earcut, ten WOFF2 fonts and a sample mask --
is inlined into ONE `generate_3d_file.html` that runs from `file://` with no
network. Deploying it to leafmachine.org is copying that one file.

    python3 _build.py

Sources live in `src/`, third-party assets in `assets/`. Edit those, never the
generated HTML.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
ASSETS = HERE / "assets"
OUT = HERE / "generate_3d_file.html"

# id -> (menu label, weight/style the WOFF2 actually carries)
FONTS = [
    ("Inter", "Inter"),
    ("Montserrat", "Montserrat"),
    ("Oswald", "Oswald"),
    ("BebasNeue", "Bebas Neue"),
    ("EBGaramond", "EB Garamond"),
    ("PlayfairDisplay", "Playfair Display"),
    ("RobotoSlab", "Roboto Slab"),
    ("Lora", "Lora"),
    ("Pacifico", "Pacifico"),
    ("RobotoMono", "Roboto Mono"),
]


def strip_modules(js: str) -> str:
    """Drop ESM keywords so the pieces can be concatenated into one module scope.

    `src/*.js` keep their `export`s because the node tests import them directly;
    inside the built page they all share one <script type="module">, so the
    exports are meaningless and `import` lines would 404.
    """
    js = re.sub(r"^\s*import[^;]+;\s*$", "", js, flags=re.M)
    js = re.sub(r"^export\s+", "", js, flags=re.M)
    return js


def font_faces() -> str:
    css = []
    for ident, label in FONTS:
        path = ASSETS / "fonts" / f"{ident}.woff2"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        css.append(
            f"@font-face{{font-family:'LM3 {label}';font-display:block;"
            f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}"
        )
    return "\n".join(css)


def build() -> None:
    css = (SRC / "style.css").read_text() + "\n" + font_faces()
    fonts_js = json.dumps(
        [{"id": i, "label": l, "css": f"'LM3 {l}'"} for i, l in FONTS],
        separators=(",", ":"),
    )
    def data_uri(name: str, mime: str) -> str:
        raw = (ASSETS / name).read_bytes()
        return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")

    sample_uri = data_uri("sample_mask.png", "image/png")
    # The vein layer only means anything on a photo that matches the mask, so the
    # demo ships as a PAIR from one oriented crop rather than a lone photo.
    pair_mask_uri = data_uri("sample_pair_mask.png", "image/png")
    pair_rgb_uri = data_uri("sample_pair_rgb.jpg", "image/jpeg")

    html = (SRC / "template.html").read_text()
    for token, value in (
        ("__CSS__", css),
        ("__EARCUT__", (ASSETS / "earcut.js").read_text()),
        ("__GEOM__", strip_modules((SRC / "geom.js").read_text())),
        ("__VIEWER__", strip_modules((SRC / "viewer.js").read_text())),
        ("__VEINS__", strip_modules((SRC / "veins.js").read_text())),
        ("__APP__", strip_modules((SRC / "app.js").read_text())),
        ("__FONTS__", fonts_js),
        ("__SAMPLE__", sample_uri),
        ("__SAMPLE_PAIR_MASK__", pair_mask_uri),
        ("__SAMPLE_PAIR_RGB__", pair_rgb_uri),
    ):
        assert html.count(token) == 1, f"token {token} appears {html.count(token)}x"
        html = html.replace(token, value)

    OUT.write_text(html)
    kb = len(html.encode()) / 1024
    print(f"wrote {OUT.name}  {kb:,.0f} KB  ({len(FONTS)} fonts inlined)")


if __name__ == "__main__":
    build()
