"""LeafMachine2 overlay palette + config-driven overlay styling for LeafMachine3.

All overlay colors and per-part display flags are user-editable in
``LM3_settings.yaml`` under ``report.overlay`` (see that file). The dictionaries
below are the built-in **fallback defaults** (the verbatim LeafMachine2 palette),
used only when a class is missing from the config. Reporter code should build an
``OverlayStyle`` from the config and query it, so users can fully restyle the
Summary_Image without touching code.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger("leafmachine3.palette")

RGB = tuple[int, int, int]

# Overlay keys removed in the move to per-group (leaf/plant/archival) border+fill styling. An old
# config that still sets these is warned about once at load, since the behavior now lives under
# report.overlay.groups and the old keys would otherwise be silently ignored.
_REMOVED_OVERLAY_KEYS = (
    "draw_boxes_archival", "draw_boxes_plant", "line_width_archival", "line_width_plant",
)

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

# per-keypoint-GROUP overlay colors (landmark pose). Groups come from core.landmarks.KPT_GROUP.
LANDMARK_GROUPS: dict[str, RGB] = {
    "lamina": (255, 255, 255),   # lamina_tip / lamina_base anchors
    "midvein": (0, 200, 255),    # midrib trace
    "apex": (255, 60, 60),       # apex triple
    "base": (255, 60, 255),      # base triple
    "petiole": (255, 170, 0),    # petiole trace + tip
    "width": (0, 255, 128),      # width_left / width_right
}
# skeleton edge KIND -> group color (see core.landmarks.SKELETON kinds)
_EDGE_KIND_GROUP: dict[str, str] = {
    "midvein": "midvein", "petiole": "petiole", "apex": "apex",
    "base": "base", "width": "width", "lamina_length": "lamina",
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


# Plant detections split by class into the "leaf" vs "plant" style groups; everything from the
# ArchivalDetector is the "archival" group. Leaves are drawn as rotated min-bboxes in rotated mode.
LEAF_DET_CLASSES: frozenset[str] = frozenset({"Leaf_WHOLE", "Leaf_PARTIAL"})


@dataclass(frozen=True)
class GroupStyle:
    """How one box-style GROUP (leaf / plant / archival) is drawn on the Summary_Image.

    ``border`` strokes the box outline and ``fill`` shades its interior at ``fill_alpha`` (0..1),
    both in the class color; the two are independent. ``line_width`` is the outline width in px at
    the reference resolution (2592 long side) and is scaled with the image size at draw time. A box
    is drawn at all only when ``border`` or ``fill`` is on (:pyattr:`visible`).
    """
    border: bool = True
    fill: bool = False
    fill_alpha: float = 0.30
    line_width: int = 3

    @property
    def visible(self) -> bool:
        return bool(self.border or self.fill)


# Default per-group styling (matches LM3_settings.yaml): plant + leaves get a border and no fill,
# archival elements get a translucent fill and no border. Used when the config omits a group.
_GROUP_DEFAULTS: dict[str, GroupStyle] = {
    "leaf":     GroupStyle(border=True,  fill=False, fill_alpha=0.30, line_width=3),
    "plant":    GroupStyle(border=True,  fill=False, fill_alpha=0.30, line_width=3),
    "archival": GroupStyle(border=False, fill=True,  fill_alpha=0.30, line_width=3),
}


@dataclass
class OverlayStyle:
    """Config-driven overlay style resolved from ``report.overlay`` in LM3_settings.yaml.

    Per-class ``color``/``show`` come from the config; anything absent falls back to the
    LeafMachine2 defaults above, so the overlay always renders even if a user prunes the
    config. Build one via :meth:`from_config` and query ``color_for`` / ``show`` / the flags.
    """
    enabled: bool = True
    draw_masks: bool = True
    draw_landmarks: bool = True
    draw_petiole: bool = True
    draw_labels: bool = True
    draw_confidence: bool = True
    draw_cf_banner: bool = True
    # the two CF scale overlays; styled by CFScalebarStyle (report.overlay.cf_scalebar). Both
    # default OFF so a config written before they existed keeps producing the same overlay.
    insert_cf_in_rulers: bool = False   # 1 cm + 1 inch bars on a white raft over every Ruler
    insert_cf_exterior: bool = False    # 1 cm checkerboard appended OUTSIDE the top + left edges
    alpha: float = 0.45
    line_width_mask: int = 2
    font_scale: float = 1.0
    label_text_color: RGB = (255, 255, 255)
    cf_banner_color: RGB = (255, 255, 255)
    box_style: str = "rotated"        # leaf boxes: "rotated" (min bbox) | "yolo" (axis-aligned)
    # per-group (leaf / plant / archival) border+fill styling; None -> _GROUP_DEFAULTS
    groups: dict[str, GroupStyle] | None = None
    # {lowercased class name: (color, show)} merged from config over the defaults
    _classes: dict[str, tuple[RGB, bool]] | None = None

    @classmethod
    def from_config(cls, cfg) -> "OverlayStyle":
        """Build from a parsed config's ``report.overlay`` mapping (dict-like or dot-access)."""
        ov = _get(cfg, "report", "overlay") or {}
        stale = [k for k in _REMOVED_OVERLAY_KEYS if _getk(ov, k, None) is not None]
        if stale:
            log.warning(
                "report.overlay keys %s are no longer used; box border/fill is now configured "
                "per-group under report.overlay.groups (leaf / plant / archival). Ignoring them.",
                stale,
            )
        classes: dict[str, tuple[RGB, bool]] = {
            name: (rgb, True) for name, rgb in _DEFAULTS.items()   # start from LM2 defaults
        }
        for group in ("archival", "plant", "segmentation"):
            for name, spec in (_as_dict(_get_key(ov, "classes", group)) or {}).items():
                spec = _as_dict(spec) or {}
                color = tuple(spec.get("color", default_color_for(name)))  # type: ignore[arg-type]
                show = bool(spec.get("show", True))
                classes[str(name).lower()] = (color, show)               # config overrides default
        gcfg = _as_dict(_getk(ov, "groups", None)) or {}
        groups: dict[str, GroupStyle] = {}
        for key, dflt in _GROUP_DEFAULTS.items():
            spec = _as_dict(_get_key(gcfg, key)) or {}
            groups[key] = GroupStyle(
                border=bool(spec.get("border", dflt.border)),
                fill=bool(spec.get("fill", dflt.fill)),
                fill_alpha=float(spec.get("fill_alpha", dflt.fill_alpha)),
                line_width=int(spec.get("line_width", dflt.line_width)),
            )
        return cls(
            enabled=bool(_getk(ov, "enabled", True)),
            draw_masks=bool(_getk(ov, "draw_masks", True)),
            draw_landmarks=bool(_getk(ov, "draw_landmarks", True)),
            draw_petiole=bool(_getk(ov, "draw_petiole", True)),
            draw_labels=bool(_getk(ov, "draw_labels", True)),
            draw_confidence=bool(_getk(ov, "draw_confidence", True)),
            draw_cf_banner=bool(_getk(ov, "draw_cf_banner", True)),
            insert_cf_in_rulers=bool(_getk(ov, "insert_cf_in_rulers", False)),
            insert_cf_exterior=bool(_getk(ov, "insert_cf_exterior", False)),
            alpha=float(_getk(ov, "alpha", 0.45)),
            line_width_mask=int(_getk(ov, "line_width_mask", 2)),
            font_scale=float(_getk(ov, "font_scale", 1.0)),
            label_text_color=tuple(_getk(ov, "label_text_color", (255, 255, 255))),
            cf_banner_color=tuple(_getk(ov, "cf_banner_color", (255, 255, 255))),
            box_style=str(_getk(ov, "box_style", "rotated")).lower(),
            groups=groups,
            _classes=classes,
        )

    def group_key(self, source: str, cls_name: str) -> str:
        """Which style group a detection belongs to: ``archival`` | ``leaf`` | ``plant``."""
        if str(source) == "archival":
            return "archival"
        return "leaf" if str(cls_name) in LEAF_DET_CLASSES else "plant"

    def group(self, key: str) -> GroupStyle:
        """The :class:`GroupStyle` for a group key, falling back to the built-in default."""
        g = (self.groups or {}).get(key)
        return g if g is not None else _GROUP_DEFAULTS[key]

    def group_for(self, source: str, cls_name: str) -> GroupStyle:
        """The :class:`GroupStyle` for a detection given its ``source`` and class name."""
        return self.group(self.group_key(source, cls_name))

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


@dataclass
class LandmarkStyle:
    """Config-driven style for drawing leaf landmarks (points + skeleton) on the overlays.

    Colors are per keypoint GROUP (``lamina``/``midvein``/``apex``/``base``/``petiole``/``width``)
    and live under ``report.overlay.landmark`` in LM3_settings.yaml; anything absent falls back to
    the :data:`LANDMARK_GROUPS` defaults. ``min_conf`` hides occluded/uncertain keypoints so the
    overlay never draws junk points.
    """
    draw_points: bool = True
    draw_skeleton: bool = True
    point_radius: int = 4
    line_width: int = 2
    min_conf: float = 0.25
    label_color: RGB = (255, 255, 255)
    curvature_color: RGB = (0, 0, 0)         # the bend lines (midvein ends -> curvature_point); black, drawn UNDER the cyan/white
    _groups: dict[str, RGB] | None = None

    @classmethod
    def from_config(cls, cfg) -> "LandmarkStyle":
        lm = _get(cfg, "report", "overlay", "landmark") or {}
        groups: dict[str, RGB] = dict(LANDMARK_GROUPS)
        for name, color in (_as_dict(_getk(lm, "colors", None)) or {}).items():
            groups[str(name).lower()] = tuple(color)  # type: ignore[assignment]
        return cls(
            draw_points=bool(_getk(lm, "draw_points", True)),
            draw_skeleton=bool(_getk(lm, "draw_skeleton", True)),
            point_radius=int(_getk(lm, "point_radius", 4)),
            line_width=int(_getk(lm, "line_width", 2)),
            min_conf=float(_getk(lm, "min_conf", 0.25)),
            label_color=tuple(_getk(lm, "label_color", (255, 255, 255))),
            curvature_color=tuple(_getk(lm, "curvature_color", (0, 0, 0))),
            _groups=groups,
        )

    def color_for_group(self, group: str) -> RGB:
        return (self._groups or LANDMARK_GROUPS).get(str(group).lower(), (255, 255, 255))

    def color_for_kind(self, kind: str) -> RGB:
        return self.color_for_group(_EDGE_KIND_GROUP.get(str(kind), "lamina"))


@dataclass
class CFScalebarStyle:
    """Config-driven style for the two conversion-factor scale overlays (``report.overlay.cf_scalebar``).

    Both are drawn at the sheet's FINAL px/cm, converted to the original frame the overlay renders
    on, so a viewer can measure straight off the image:

    ``insert_cf_in_rulers`` lays a white raft over every detected Ruler carrying two solid bars --
    one exactly 1 cm long (``cm_color``) and one exactly 1 inch long (``inch_color``) -- running
    along the ruler's long axis so they read against the ruler's own graduations.

    ``insert_cf_exterior`` appends a checkerboard of 1 cm cells to the top and left of the sheet
    (``exterior_cells`` cells thick), OUTSIDE the image so it obscures nothing.

    ``bar_thickness`` is in pixels at the reference resolution (2592 px long side) and scales with
    the image, like the box ``line_width`` values; ``brim`` is a literal pixel count, since it is a
    hairline gutter rather than a drawn feature.
    """
    cm_color: RGB = (0, 255, 255)             # the 1 cm bar -- cyan
    inch_color: RGB = (0, 255, 0)             # the 1 inch bar -- bright green
    raft_color: RGB = (255, 255, 255)         # the solid backing the two bars sit on -- white
    brim: int = 3                             # white margin around and between the bars, px
    bar_thickness: int = 10                   # bar width across the ruler, px at the reference resolution
    exterior_cells: int = 2                   # checkerboard band thickness, in 1 cm cells
    exterior_light: RGB = (255, 255, 255)     # the "O" cells
    exterior_dark: RGB = (0, 0, 0)            # the "X" cells

    @classmethod
    def from_config(cls, cfg) -> "CFScalebarStyle":
        s = _get(cfg, "report", "overlay", "cf_scalebar") or {}
        return cls(
            cm_color=tuple(_getk(s, "cm_color", (0, 255, 255))),
            inch_color=tuple(_getk(s, "inch_color", (0, 255, 0))),
            raft_color=tuple(_getk(s, "raft_color", (255, 255, 255))),
            brim=int(_getk(s, "brim", 3)),
            bar_thickness=int(_getk(s, "bar_thickness", 10)),
            exterior_cells=int(_getk(s, "exterior_cells", 2)),
            exterior_light=tuple(_getk(s, "exterior_light", (255, 255, 255))),
            exterior_dark=tuple(_getk(s, "exterior_dark", (0, 0, 0))),
        )


@dataclass
class PetioleStyle:
    """Config-driven style for the petiole-width overlays (``report.overlay.petiole``). The reported
    width band is drawn in ``width_color`` (purple), the per-sample width probes in ``sample_color``
    (light purple); the Leaf/Petiole masks use the segmentation palette (filled, no outline)."""
    width_color: RGB = (3, 32, 252)           # reported (median) width band + the 1px zoom line -- blue
    sample_color: RGB = (6, 124, 191)         # per-sample width segments -- light blue
    band_thickness: int = 4
    sample_thickness: int = 2
    mask_alpha: float = 0.45
    label_color: RGB = (255, 255, 255)
    # right-half zoom panel: the petiole stays full color; the background is tinted toward this color.
    zoom_bg_tint: float = 0.0                 # 0 = no tint (default); raise to tint the background
    zoom_bg_color: RGB = (255, 0, 0)          # red

    @classmethod
    def from_config(cls, cfg) -> "PetioleStyle":
        p = _get(cfg, "report", "overlay", "petiole") or {}
        return cls(
            width_color=tuple(_getk(p, "width_color", (3, 32, 252))),
            sample_color=tuple(_getk(p, "sample_color", (6, 124, 191))),
            band_thickness=int(_getk(p, "band_thickness", 4)),
            sample_thickness=int(_getk(p, "sample_thickness", 2)),
            mask_alpha=float(_getk(p, "mask_alpha", 0.45)),
            label_color=tuple(_getk(p, "label_color", (255, 255, 255))),
            zoom_bg_tint=float(_getk(p, "zoom_bg_tint", 0.0)),
            zoom_bg_color=tuple(_getk(p, "zoom_bg_color", (255, 0, 0))),
        )


@dataclass
class SpecimenStyle:
    """Config-driven style for the ``Overlay_Specimen_Segmentation`` view (``report.overlay_specimen``).

    A 2-panel QC image: LEFT = the sheet with the final specimen mask filled (``mask_color`` at
    ``mask_alpha``), the paperclean-removed region tinted (``refined_color`` at ``refined_alpha``),
    and the paper-sampling locations boxed (``sample_box_color``); RIGHT = the final masked RGB
    cutout on black. Colors are ``[R, G, B]`` like the rest of the overlay palette. ``display_max_dim``
    fits each panel's long side so the QC image stays a manageable size."""
    enabled: bool = True
    mask_color: RGB = (70, 200, 70)          # final specimen mask fill -- green
    refined_color: RGB = (235, 60, 60)       # paperclean-removed region -- red
    sample_box_color: RGB = (30, 120, 245)   # paper-sampling box outlines -- blue
    mask_alpha: float = 0.40
    refined_alpha: float = 0.55
    outline: bool = True                     # stroke the final mask contour
    display_max_dim: int = 2048              # fit each panel's long side to this (0 = full res)

    @classmethod
    def from_config(cls, cfg) -> "SpecimenStyle":
        s = _get(cfg, "report", "overlay_specimen") or {}
        return cls(
            enabled=bool(_getk(s, "enabled", True)),
            mask_color=tuple(_getk(s, "mask_color", (70, 200, 70))),
            refined_color=tuple(_getk(s, "refined_color", (235, 60, 60))),
            sample_box_color=tuple(_getk(s, "sample_box_color", (30, 120, 245))),
            mask_alpha=float(_getk(s, "mask_alpha", 0.40)),
            refined_alpha=float(_getk(s, "refined_alpha", 0.55)),
            outline=bool(_getk(s, "outline", True)),
            display_max_dim=int(_getk(s, "display_max_dim", 2048)),
        )


# -- user-supplied fill colors (mask/cutout backgrounds) --------------------------
#: The two names the settings UI offers by name. Any other color is given as a triple.
NAMED_FILLS: dict[str, RGB] = {"white": (255, 255, 255), "black": (0, 0, 0)}

_HEX_RE = re.compile(r"^#?[0-9a-f]{3}(?:[0-9a-f]{3})?$")
_SPLIT_RE = re.compile(r"[,\s()\[\]]+")


def parse_fill_color(value, default: RGB = (255, 255, 255)) -> RGB:
    """Resolve a user-supplied fill color to an ``(R, G, B)`` triple, 0-255.

    Accepts every form a settings file can plausibly carry, because this value reaches the config
    by three different routes -- the shipped YAML, the settings form's text row, and a hand edit::

        white | black          the two named colors (case-insensitive)
        [255, 0, 0]            a YAML list -- what the UI's color control writes
        "255, 0, 0"            the same triple typed into a free-text row
        "#ff0000" / "#f00"     hex, either length

    Components are rounded and clamped to 0-255. An unparseable value falls back to ``default``
    with a warning rather than raising: a mistyped color must cost one folder's appearance, never
    a finished pipeline its report.

    Returns RGB, matching the rest of the config (``report.overlay`` colors, ``hole_rgb_color``).
    Callers drawing with OpenCV must reverse it -- see ``overlay._bgr``.
    """
    rgb = _coerce_rgb(value)
    if rgb is None:
        log.warning("unrecognized fill color %r -- falling back to %s", value, default)
        return default
    return rgb


def _coerce_rgb(value) -> RGB | None:
    """``value`` -> an ``(R, G, B)`` triple, or ``None`` when it is not a color at all."""
    if isinstance(value, str):
        s = value.strip().lower()
        if s in NAMED_FILLS:
            return NAMED_FILLS[s]
        if _HEX_RE.match(s):
            h = s.lstrip("#")
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        return _triple([p for p in _SPLIT_RE.split(s) if p])
    if isinstance(value, (list, tuple)):
        return _triple(value)
    return None


def _triple(parts) -> RGB | None:
    if len(parts) != 3:
        return None
    try:
        vals = [int(round(float(p))) for p in parts]
    except (TypeError, ValueError):
        return None
    return (max(0, min(255, vals[0])), max(0, min(255, vals[1])), max(0, min(255, vals[2])))


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
