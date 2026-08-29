"""Tests for :mod:`leafmachine3.reporting.palette` -- config-driven overlay styling."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from leafmachine3.core.config import Config
from leafmachine3.reporting.palette import OverlayStyle, default_color_for, parse_fill_color


def _cfg(tmp_path: Path, overlay: dict) -> Config:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"report": {"overlay": overlay}}), encoding="utf-8")
    return Config.load(path)


def test_defaults_when_config_empty(tmp_path: Path) -> None:
    style = OverlayStyle.from_config(_cfg(tmp_path, {}))
    # falls back to the built-in LeafMachine2 palette
    assert style.color_for("Ruler") == default_color_for("Ruler") == (255, 0, 70)
    assert style.show("Ruler") is True
    assert style.draw_masks is True


def test_config_color_overrides_default(tmp_path: Path) -> None:
    style = OverlayStyle.from_config(
        _cfg(
            tmp_path,
            {"classes": {"archival": {"Ruler": {"color": [1, 2, 3], "show": False}}}},
        )
    )
    assert style.color_for("Ruler") == (1, 2, 3)            # config wins
    assert style.show("Ruler") is False                     # show flag honored
    # a class not mentioned in the config keeps its default
    assert style.color_for("Label") == default_color_for("Label")


def test_global_flags_from_config(tmp_path: Path) -> None:
    style = OverlayStyle.from_config(
        _cfg(
            tmp_path,
            {"draw_masks": False, "draw_confidence": False, "alpha": 0.2,
             "label_text_color": [10, 20, 30]},
        )
    )
    assert style.draw_masks is False
    assert style.draw_confidence is False
    assert style.alpha == 0.2
    assert style.label_text_color == (10, 20, 30)


def test_group_defaults(tmp_path: Path) -> None:
    # with no `groups` config the built-in defaults apply: leaf/plant border-no-fill, archival fill-no-border
    style = OverlayStyle.from_config(_cfg(tmp_path, {}))
    assert style.group("leaf").border is True and style.group("leaf").fill is False
    assert style.group("plant").border is True and style.group("plant").fill is False
    assert style.group("archival").border is False and style.group("archival").fill is True
    assert style.group("archival").fill_alpha == 0.30


def test_group_from_config(tmp_path: Path) -> None:
    style = OverlayStyle.from_config(
        _cfg(
            tmp_path,
            {"groups": {"plant": {"border": False, "fill": True, "fill_alpha": 0.4, "line_width": 7}}},
        )
    )
    g = style.group("plant")
    assert g.border is False and g.fill is True and g.fill_alpha == 0.4 and g.line_width == 7
    assert g.visible is True
    # a group absent from the config keeps its default
    assert style.group("leaf").border is True and style.group("leaf").fill is False


def test_group_for_resolves_source_and_class(tmp_path: Path) -> None:
    style = OverlayStyle.from_config(_cfg(tmp_path, {}))
    assert style.group_for("archival", "Ruler") is style.group("archival")
    assert style.group_for("plant", "Leaf_WHOLE") is style.group("leaf")
    assert style.group_for("plant", "Leaf_PARTIAL") is style.group("leaf")
    assert style.group_for("plant", "Flower_ONE") is style.group("plant")


def test_fill_for_appends_alpha(tmp_path: Path) -> None:
    style = OverlayStyle.from_config(_cfg(tmp_path, {"alpha": 0.5}))
    r, g, b, a = style.fill_for("Ruler")
    assert (r, g, b) == (255, 0, 70)
    assert a == int(round(0.5 * 255))


# --------------------------------------------------------------------------- #
# parse_fill_color -- report.masks.inverse_fill
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "want"),
    [
        ("white", (255, 255, 255)),
        ("black", (0, 0, 0)),
        ("WHITE", (255, 255, 255)),           # case-insensitive
        ("  white  ", (255, 255, 255)),       # a hand-edited YAML value keeps its spaces
        ([255, 0, 0], (255, 0, 0)),           # what the settings UI's color control writes
        ((12, 34, 56), (12, 34, 56)),
        ("255, 0, 0", (255, 0, 0)),           # ...and what a free-text settings row writes
        ("[255, 0, 0]", (255, 0, 0)),         # a list typed into a text row, unparsed
        ("(0, 128, 255)", (0, 128, 255)),
        ("#ff0000", (255, 0, 0)),
        ("#F00", (255, 0, 0)),                # short hex, uppercase
        ([200.6, 0.4, 128.0], (201, 0, 128)),  # floats round, they do not truncate
        ([300, -20, 40], (255, 0, 40)),       # out of range clamps instead of wrapping
    ],
)
def test_parse_fill_color_accepts_every_config_form(value, want) -> None:
    assert parse_fill_color(value) == want


@pytest.mark.parametrize(
    "value",
    [None, "", "chartreuse", "1, 2", "1, 2, 3, 4", [1, 2], {"r": 1}, "#ff00", 255, True],
)
def test_parse_fill_color_falls_back_rather_than_raising(value) -> None:
    """A mistyped color costs one folder its appearance, never the run its report."""
    assert parse_fill_color(value) == (255, 255, 255)
    assert parse_fill_color(value, default=(0, 0, 0)) == (0, 0, 0)
