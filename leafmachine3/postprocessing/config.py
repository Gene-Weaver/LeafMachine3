"""Load the deployment's postprocessing settings -> per-module settings.

Deliberately tiny and separate from the pipeline's ``leafmachine3.core.config`` so postprocessing
tools have zero coupling to the main run config. Every top-level key in the YAML is one module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


def load_settings(path: Optional[str] = None) -> dict[str, Any]:
    """Return the parsed postprocessing settings (``{}`` if the file is absent).

    With no explicit path this is the deployment's ``postprocessing.yaml`` -- section 3.1 row 3,
    ``LM3_POSTPROCESS_SETTINGS`` else ``<user-config>/lm3/<deployment>/postprocessing.yaml``. It
    WAS a bare ``postprocessing_settings.yaml`` off the CWD, which meant the Postprocess tab and
    the standalone CLIs configured different files unless both were launched from the same
    directory; section 3.1 is explicit that no path falls back to the working directory.
    """
    if path is None:
        # Imported inside the function, mirroring ``postprocess_api._settings_path()``, so this
        # module keeps its "zero coupling at import time" property for the standalone tools.
        from leafmachine3.core.paths import PathsError, postprocessing_settings_path

        try:
            p = postprocessing_settings_path()
        except PathsError:
            # e.g. DeploymentIdentityError from an empty LM3_DEPLOYMENT_ID. Row 3's on-miss cell is
            # "packaged defaults", so degrade to the tools' own defaults instead of crashing --
            # the same choice postprocess_api makes when the resolver cannot answer.
            return {}
    else:
        p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def module_settings(all_settings: dict[str, Any], module_name: str) -> dict[str, Any]:
    """Return the settings block for one module (a fresh dict; ``{}`` if the key is missing)."""
    block = all_settings.get(module_name)
    return dict(block) if isinstance(block, dict) else {}
