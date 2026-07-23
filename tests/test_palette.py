"""Tests for :mod:`leafmachine3.reporting.palette` -- config-driven overlay styling."""
from __future__ import annotations

from pathlib import Path

import yaml

from leafmachine3.core.config import Config
from leafmachine3.reporting.palette import OverlayStyle, default_color_for


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
             "line_width_plant": 7, "label_text_color": [10, 20, 30]},
        )
    )
    assert style.draw_masks is False
    assert style.draw_confidence is False
    assert style.alpha == 0.2
    assert style.line_width_plant == 7
    assert style.label_text_color == (10, 20, 30)


def test_fill_for_appends_alpha(tmp_path: Path) -> None:
    style = OverlayStyle.from_config(_cfg(tmp_path, {"alpha": 0.5}))
    r, g, b, a = style.fill_for("Ruler")
    assert (r, g, b) == (255, 0, 70)
    assert a == int(round(0.5 * 255))
