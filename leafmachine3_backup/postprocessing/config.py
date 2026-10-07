"""Load ``postprocessing_settings.yaml`` -> per-module settings.

Deliberately tiny and separate from the pipeline's ``leafmachine3.core.config`` so postprocessing
tools have zero coupling to the main run config. Every top-level key in the YAML is one module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_SETTINGS_PATH = "postprocessing_settings.yaml"


def load_settings(path: Optional[str] = None) -> dict[str, Any]:
    """Return the parsed ``postprocessing_settings.yaml`` (``{}`` if the file is absent)."""
    p = Path(path or DEFAULT_SETTINGS_PATH)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def module_settings(all_settings: dict[str, Any], module_name: str) -> dict[str, Any]:
    """Return the settings block for one module (a fresh dict; ``{}`` if the key is missing)."""
    block = all_settings.get(module_name)
    return dict(block) if isinstance(block, dict) else {}
