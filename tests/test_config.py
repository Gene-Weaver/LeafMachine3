"""Tests for :mod:`leafmachine3.core.config` -- layering, accessors, and validation."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from leafmachine3.core.config import Config, Section


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_defaults_merge_and_override(tmp_path: Path) -> None:
    """File values win over defaults; ``overrides`` win over the file."""
    cfg_path = _write(
        tmp_path,
        {
            "project": {"run_name": "from_file", "input": {"dirs": ["a"]}},
            "compute": {"mock": True},
        },
    )
    cfg = Config.load(cfg_path, overrides={"project": {"run_name": "from_override"}})

    assert cfg.project.run_name == "from_override"          # override beats file
    assert cfg.project.input.dirs == ["a"]                  # file beats default
    assert cfg.compute.mock is True
    # untouched default survives the merge
    assert cfg.ingest.max_working_dim == 3200


def test_section_attribute_and_get_access(tmp_path: Path) -> None:
    cfg = Config.load(_write(tmp_path, {"project": {"input": {"dirs": ["x"]}}}))
    node = cfg.project.input
    assert isinstance(node, Section)
    assert node.recursive is True                            # dot access
    assert node.get("missing", "fallback") == "fallback"    # tolerant .get


def test_is_enabled_and_stage(tmp_path: Path) -> None:
    cfg = Config.load(
        _write(
            tmp_path,
            {"modules": {"leaf_segmenter": {"enabled": True, "conf": 0.3},
                         "reporter": {"enabled": False}}},
        )
    )
    assert cfg.is_enabled("leaf_segmenter") is True
    assert cfg.is_enabled("reporter") is False
    assert cfg.stage("leaf_segmenter").conf == 0.3
    assert cfg.stage("does_not_exist") == Section()         # empty, not an error


def test_restart_normalization(tmp_path: Path) -> None:
    def restart(value):
        return Config.load(_write(tmp_path, {"project": {"run_mode": {"restart": value}}})).restart

    assert restart([]) is None
    assert restart("all") == "all"
    assert restart(["leaf_segmenter"]) == ["leaf_segmenter"]
    assert restart(["all", "reporter"]) == "all"            # any 'all' collapses to full


def test_validate_rejects_empty_input(tmp_path: Path) -> None:
    cfg = Config.load(_write(tmp_path, {"compute": {"mock": True}}))
    with pytest.raises(ValueError, match="input.dirs is empty"):
        cfg.validate()


def test_validate_rejects_unknown_restart_key(tmp_path: Path) -> None:
    cfg = Config.load(
        _write(
            tmp_path,
            {"project": {"input": {"dirs": ["x"]}, "run_mode": {"restart": ["not_a_stage"]}},
             "compute": {"mock": True}},
        )
    )
    with pytest.raises(ValueError, match="unknown stage key"):
        cfg.validate()


def test_validate_passes_in_mock_without_models(tmp_path: Path) -> None:
    """Mock mode does not require model.path / models_dir to be set."""
    cfg = Config.load(
        _write(tmp_path, {"project": {"input": {"dirs": ["x"]}}, "compute": {"mock": True}})
    )
    cfg.validate()  # must not raise


def test_validate_requires_models_when_not_mock(tmp_path: Path) -> None:
    cfg = Config.load(
        _write(
            tmp_path,
            {"project": {"input": {"dirs": ["x"]}},
             "compute": {"mock": False, "devices": "cpu"},
             "modules": {"archival_detector": {"enabled": True}}},
        )
    )
    with pytest.raises(ValueError, match="model.path"):
        cfg.validate()


def test_settings_hash_changes_with_config(tmp_path: Path) -> None:
    cfg_a = Config.load(_write(tmp_path, {"modules": {"leaf_segmenter": {"enabled": True, "conf": 0.3}}}))
    hash_a = cfg_a.stage_settings_hash("leaf_segmenter")
    cfg_b = Config.load(_write(tmp_path, {"modules": {"leaf_segmenter": {"enabled": True, "conf": 0.9}}}))
    hash_b = cfg_b.stage_settings_hash("leaf_segmenter")
    assert hash_a != hash_b
    assert cfg_a.stage_settings_hash("leaf_segmenter") == hash_a  # stable / deterministic


def test_resolve_path_absolutizes(tmp_path: Path) -> None:
    cfg = Config.load(_write(tmp_path, {"compute": {"mock": True}}))
    assert Path(cfg.resolve_path("/etc/hosts")).is_absolute()
    assert Path(cfg.resolve_path("relative/thing")).is_absolute()
