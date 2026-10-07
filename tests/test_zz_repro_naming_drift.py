"""TEMP repro harness for the naming-drift claim (delete after use)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from leafmachine3.machine3 import machine3
from tests.conftest import build_mock_config, fresh_out_dir


@pytest.fixture
def env(synthetic_images: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output_dir = fresh_out_dir("repro_naming_drift")
    cfg = build_mock_config(synthetic_images, output_dir, run_name="e2e")
    cfg["modules"]["ect"]["enabled"] = False          # 'ect' package not installed in this env
    cfg_path = tmp_path / "LM3_settings.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return cfg_path


def _names(root: Path) -> list[str]:
    return sorted(p.name for p in root.rglob("*") if p.is_file())


def test_repro_naming_rename(env: Path) -> None:
    project = machine3(env)
    reports = Path(str(project.dirs.reports))
    inv_dir = reports / "Specimen_Masks" / "Binary_Masks_Specimen_Inverse"
    print("\n--- RUN 1 inverse dir ---")
    print(_names(inv_dir))
    crops_before = _names(reports / "Crops")
    print("crops:", crops_before[:4])

    # user renames the friendly name + the mask_full prefix in the settings UI
    cfg = yaml.safe_load(env.read_text())
    cfg["naming"]["friendly_names"]["Specimen_Inverse"] = "notPlant"
    cfg["naming"]["mask_full_prefix"] = "WholeMask"
    cfg["naming"]["friendly_names"]["Ruler"] = "RULERZ"
    env.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    project2 = machine3(env)
    reports2 = Path(str(project2.dirs.reports))
    inv_dir2 = reports2 / "Specimen_Masks" / "Binary_Masks_Specimen_Inverse"
    print("\n--- RUN 2 inverse dir (same reports root: %s) ---" % (reports2 == reports))
    print(_names(inv_dir2))
    print("crops dirs:", sorted(d.name for d in (reports2 / "Crops").iterdir()))
    print("reporter state:", project2.db.stage_state("reporter"))
    print("new-name files present:", [n for n in _names(inv_dir2) if "notPlant" in n or "WholeMask" in n])
