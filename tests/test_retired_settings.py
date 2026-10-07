"""Retired settings are warned about ONCE per run, by machine3, not at every use.

Found 2026-10-07 in an install test: a settings file carrying report.crops.source produced the same
warning 22 times (once per specimen) from the Reporter. core.config.RETIRED_SETTINGS is now the one
list; Config.retired_settings() reports what a file still sets, and the use sites stay silent.
"""
from __future__ import annotations

import logging

import yaml

from leafmachine3.core.config import RETIRED_SETTINGS, Config
from leafmachine3.reporting.palette import OverlayStyle


def _load(tmp_path, extra: dict) -> Config:
    raw = {"version": 3, "project": {"run_name": "r", "input": {"dirs": ["in"]}, "output": {"dir": "runs"}}}
    for dotted, value in extra.items():
        node = raw
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    (tmp_path / "in").mkdir(exist_ok=True)
    path = tmp_path / "LM3_settings.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return Config.load(path)


def test_a_clean_file_has_no_retired_settings(tmp_path):
    assert _load(tmp_path, {}).retired_settings() == []


def test_every_retired_key_present_is_reported_with_its_reason(tmp_path):
    cfg = _load(tmp_path, {"report.crops.source": "working", "project.output.keep_tmp": False,
                           "report.overlay.line_width_plant": 3})
    found = dict(cfg.retired_settings())
    assert set(found) == {"report.crops.source", "project.output.keep_tmp", "report.overlay.line_width_plant"}
    assert all(found[k] == RETIRED_SETTINGS[k] for k in found)


def test_the_shipped_default_settings_carry_no_retired_key():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    for name in ("LM3_settings.yaml",):
        assert Config.load(repo / name).retired_settings() == [], name


def test_the_overlay_palette_no_longer_warns_per_call(tmp_path, caplog):
    cfg = _load(tmp_path, {"report.overlay.draw_boxes_archival": True})
    with caplog.at_level(logging.WARNING):
        OverlayStyle.from_config(cfg)
        OverlayStyle.from_config(cfg)
    assert not [r for r in caplog.records if "no longer used" in r.getMessage()]


def test_the_reporter_no_longer_warns_per_specimen():
    from pathlib import Path

    import leafmachine3.modules.reporter as reporter

    assert "report.crops.source is no longer used" not in Path(reporter.__file__).read_text(encoding="utf-8")
