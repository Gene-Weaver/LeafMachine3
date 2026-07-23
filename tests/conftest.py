"""Shared pytest fixtures for the LeafMachine3 test suite.

The fixtures here synthesize small herbarium-ish specimen JPEGs with numpy + cv2 (no
network, no model weights) and assemble a ``compute.mock: true`` ``LM3_settings.yaml`` that
points at them, so the whole pipeline can be exercised end-to-end without a GPU or exports.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

# Make ``leafmachine3`` importable when pytest is launched from the repo root without
# ``PYTHONPATH=.`` (harmless when it is already on the path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# All test/run artifacts land here (gitignored), one subdir per test name, so a human can
# inspect the DB + overlays after a run instead of digging through pytest tmp dirs.
EXAMPLES_OUT = _REPO_ROOT / "examples_out"


def fresh_out_dir(name: str) -> Path:
    """Return (and clear) ``examples_out/<name>`` so each test run starts clean."""
    import shutil

    d = EXAMPLES_OUT / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_specimen_image(path: Path, seed: int) -> Path:
    """Write one deterministic synthetic specimen JPEG (a green 'leaf' on a pale sheet)."""
    rng = np.random.default_rng(seed)
    h, w = 900, 700
    img = np.full((h, w, 3), 232, dtype=np.uint8)  # pale herbarium-sheet background
    # a leaf-ish green blob
    cv2.ellipse(img, (int(0.35 * w), int(0.45 * h)), (140, 220), 25, 0, 360, (40, 130, 45), -1)
    # a 'ruler' strip along the bottom
    cv2.rectangle(img, (40, h - 70), (w - 40, h - 40), (60, 60, 60), -1)
    # a 'label' rectangle
    cv2.rectangle(img, (int(0.62 * w), int(0.12 * h)), (int(0.92 * w), int(0.34 * h)), (250, 250, 240), -1)
    # a touch of noise so the two specimens differ
    noise = rng.integers(0, 12, size=(h, w, 3), dtype=np.uint8)
    img = cv2.subtract(img, noise)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


@pytest.fixture
def synthetic_images(tmp_path: Path) -> Path:
    """Directory holding two synthetic specimen JPEGs."""
    images_dir = tmp_path / "input_images"
    for i in range(2):
        make_specimen_image(images_dir / f"specimen_{i:02d}.jpg", seed=i + 1)
    return images_dir


def build_mock_config(
    images_dir: Path,
    output_dir: Path,
    *,
    run_name: str = "test_run",
    ruler_classifier_enabled: bool = False,
) -> dict:
    """Return a full mock LM3 config dict pointing at ``images_dir`` / ``output_dir``.

    ``compute.mock`` is on (deterministic synthetic models) and ``output.tmp_dir`` is a
    concrete path so the run never depends on a tuned scratch directory.
    """
    return {
        "version": 3,
        "project": {
            "run_name": run_name,
            "input": {
                "dirs": [str(images_dir)],
                "recursive": True,
                "image_extensions": [".jpg", ".jpeg", ".png"],
            },
            "output": {"dir": str(output_dir), "tmp_dir": str(output_dir / "_scratch"), "keep_tmp": True},
            "run_mode": {"overwrite": False, "restart": [], "fail_fast": True},
            "logging": {"level": "WARNING", "to_file": False, "to_console": False},
        },
        "compute": {"devices": "cpu", "mock": True, "precision": "fp32"},
        "ingest": {"max_working_dim": 3200, "jpg_quality": 95},
        "modules": {
            "archival_detector": {
                "enabled": True,
                "classes": ["Ruler", "Barcode", "Colorcard", "Label"],
            },
            "plant_detector": {
                "enabled": True,
                "classes": ["Leaf_WHOLE", "Leaf_PARTIAL", "Seed_Fruit_ONE"],
            },
            "phenology_detector": {
                "enabled": True,
                "targets": {
                    "leaves": {"min_conf": 0.3, "min_count": 1},
                    "flowers": {"min_conf": 0.4, "min_count": 1},
                    "fruits": {"min_conf": 0.4, "min_count": 1},
                },
            },
            "ruler_classifier": {
                "enabled": ruler_classifier_enabled,
                "models_dir": "models/ruler_classifier",
                "ensemble_members": ["a", "b", "c"],
                "min_conf": 0.35,
            },
            "ruler_cf": {"enabled": False},
            "leaf_segmenter": {"enabled": True, "include_partial": False},
            "metric_grounding": {"enabled": True, "round_ndigits": 4},
            "reporter": {"enabled": True},
        },
        "report": {
            "overlay": {"enabled": True, "draw_masks": True, "draw_labels": True},
            "export_binary_masks": True,
            "export_rgb_on_black": True,
            "export_rgb_on_white": False,
            "formats": {"image_ext": "jpg", "jpg_quality": 95, "mask_ext": "png"},
        },
    }


@pytest.fixture
def mock_config_path(synthetic_images: Path, tmp_path: Path) -> Path:
    """Write a mock ``LM3_settings.yaml`` and return its path."""
    output_dir = fresh_out_dir("mock_config")
    cfg = build_mock_config(synthetic_images, output_dir)
    cfg_path = tmp_path / "LM3_settings.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path
