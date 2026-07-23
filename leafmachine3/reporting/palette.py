"""LeafMachine2 overlay palette + config-driven overlay styling for LeafMachine3.

All overlay colors and per-part display flags are user-editable in
``LM3_settings.yaml`` under ``report.overlay`` (see that file). The dictionaries
below are the built-in **fallback defaults** (the verbatim LeafMachine2 palette),
used only when a class is missing from the config. Reporter code should build an
``OverlayStyle`` from the config and query it, so users can fully restyle the
Summary_Image without touching code.
"""
from __future__ import annotations

from dataclasses import dataclass

RGB = tuple[int, int, int]

# -- built-in fallback defaults (verbatim LeafMachine2 palette) --------------------
ARCHIVAL: dict[str, RGB] = {
    "Ruler": (255, 0, 70), "Barcode": (0, 137, 65), "Colorcard": (242, 255, 0),
    "Label": (0, 0, 255), "Map": (0, 251, 255), "Envelope": (163, 0, 89),
    "Photo": (255, 205, 220), "Attached_Item": (255, 172, 40), "Weights": (140, 140, 140),
}
PLANT: dict[str, RGB] = {
    "Leaf_WHOLE": (0, 255, 55), "Leaf_PARTIAL": (0, 255, 250), "Leaflet": (255, 203, 0),
    "Seed_Fruit_ONE": (252, 255, 0), "Seed_Fruit_MANY": (0, 0, 80),
    "Flower_ONE": (255, 52, 255), "Flower_MANY": (154, 0, 255), "Bud": (255, 0, 9),
    "Specimen": (0, 0, 0), "Roots": (255, 134, 0), "Wood": (144, 22, 22),
}
SEGMENTATION: dict[str, RGB] = {
    "Leaf": (46, 255, 0), "Petiole": (255, 0, 150), "Hole": (200, 0, 255),
}

_DEFAULTS: dict[str, RGB] = {
    **{k.lower(): v for k, v in ARCHIVAL.items()},
    **{k.lower(): v for k, v in PLANT.items()},
    **{k.lower(): v for k, v in SEGMENTATION.items()},
}
_DEFAULT_RGB: RGB = (128, 128, 128)


def default_color_for(cls_name: str) -> RGB:
    """The built-in fallback color for a class name (case-insensitive)."""
    return _DEFAULTS.get(str(cls_name).lower(), _DEFAULT_RGB)


@dataclass
class OverlayStyle:
    """Config-driven overlay style resolved from ``report.overlay`` in LM3_settings.yaml.

    Per-class ``color``/``show`` come from the config; anything absent falls back to the
    LeafMachine2 defaults above, so the overlay always renders even if a user prunes the
    config. Build one via :meth:`from_config` and query ``color_for`` / ``show`` / the flags.
    """
    enabled: bool = True
    draw_boxes_archival: bool = True
    draw_boxes_plant: bool = True
    draw_masks: bool = True
    draw_labels: bool = True
    draw_confidence: bool = True
    draw_cf_banner: bool = True
    alpha: float = 0.45
    line_width_archival: int = 3
    line_width_plant: int = 3
    line_width_mask: int = 2
    font_scale: float = 1.0
    label_text_color: RGB = (255, 255, 255)
    cf_banner_color: RGB = (255, 255, 255)
    box_style: str = "rotated"        # leaf boxes: "rotated" (min bbox) | "yolo" (axis-aligned)
    # {lowercased class name: (color, show)} merged from config over the defaults
    _classes: dict[str, tuple[RGB, bool]] | None = None

    @classmethod
    def from_config(cls, cfg) -> "OverlayStyle":
        """Build from a parsed config's ``report.overlay`` mapping (dict-like or dot-access)."""
        ov = _get(cfg, "report", "overlay") or {}
        classes: dict[str, tuple[RGB, bool]] = {
            name: (rgb, True) for name, rgb in _DEFAULTS.items()   # start from LM2 defaults
        }
        for group in ("archival", "plant", "segmentation"):
            for name, spec in (_as_dict(_get_key(ov, "classes", group)) or {}).items():
                spec = _as_dict(spec) or {}
                color = tuple(spec.get("color", default_color_for(name)))  # type: ignore[arg-type]
                show = bool(spec.get("show", True))
                classes[str(name).lower()] = (color, show)               # config overrides default
        return cls(
            enabled=bool(_getk(ov, "enabled", True)),
            draw_boxes_archival=bool(_getk(ov, "draw_boxes_archival", True)),
            draw_boxes_plant=bool(_getk(ov, "draw_boxes_plant", True)),
            draw_masks=bool(_getk(ov, "draw_masks", True)),
            draw_labels=bool(_getk(ov, "draw_labels", True)),
            draw_confidence=bool(_getk(ov, "draw_confidence", True)),
            draw_cf_banner=bool(_getk(ov, "draw_cf_banner", True)),
            alpha=float(_getk(ov, "alpha", 0.45)),
            line_width_archival=int(_getk(ov, "line_width_archival", 3)),
            line_width_plant=int(_getk(ov, "line_width_plant", 3)),
            line_width_mask=int(_getk(ov, "line_width_mask", 2)),
            font_scale=float(_getk(ov, "font_scale", 1.0)),
            label_text_color=tuple(_getk(ov, "label_text_color", (255, 255, 255))),
            cf_banner_color=tuple(_getk(ov, "cf_banner_color", (255, 255, 255))),
            box_style=str(_getk(ov, "box_style", "rotated")).lower(),
            _classes=classes,
        )

    def color_for(self, cls_name: str) -> RGB:
        c = (self._classes or {}).get(str(cls_name).lower())
        return c[0] if c else default_color_for(cls_name)

    def show(self, cls_name: str) -> bool:
        c = (self._classes or {}).get(str(cls_name).lower())
        return c[1] if c else True

    def fill_for(self, cls_name: str, alpha: float | None = None) -> tuple[int, int, int, int]:
        r, g, b = self.color_for(cls_name)
        a = self.alpha if alpha is None else alpha
        return (r, g, b, int(round(max(0.0, min(1.0, a)) * 255)))


# -- tiny access helpers tolerant of dict OR dot-access config objects -------------
def _as_dict(x):
    if x is None:
        return None
    return dict(x) if hasattr(x, "keys") else x


def _get(cfg, *path):
    cur = cfg
    for p in path:
        cur = _getk(cur, p, None)
        if cur is None:
            return None
    return cur


def _get_key(obj, *path):
    cur = obj
    for p in path:
        cur = _getk(cur, p, None)
        if cur is None:
            return None
    return cur


def _getk(obj, key, default):
    if obj is None:
        return default
    if hasattr(obj, "get"):
        try:
            v = obj.get(key, default)
            return default if v is None else v
        except Exception:
            pass
    v = getattr(obj, key, default)
    return default if v is None else v
