"""LeafMachine3 POSTPROCESSING -- small, discrete tools that run AFTER a project finishes.

These are deliberately NOT part of the sequential pipeline (leafmachine3.pipeline). Each module
here is a standalone tool with a ``run(...)`` API and a ``python -m`` CLI, configured from
``postprocessing_settings.yaml`` (separate from the main ``LM3_settings.yaml``). Add new tools as
``leafmachine3/postprocessing/<name>.py`` and a matching ``<name>:`` block in the settings file.
"""
from leafmachine3.postprocessing.config import load_settings, module_settings

__all__ = ["load_settings", "module_settings"]
