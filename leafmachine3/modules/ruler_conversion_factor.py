"""Stage 5 — Ruler Conversion Factor (disabled stub).

Measuring ruler tick spacing and deriving the pixel->metric conversion factor (px per cm) is
not yet ported from the legacy pipeline. This stage exists so the canonical ``STAGE_ORDER``
imports cleanly and downstream ``depends_on=('ruler_cf', ...)`` links resolve; it is disabled
in the config (``modules.ruler_cf.enabled: false``) and therefore never actually runs. Its
model/inference hooks raise :class:`NotImplementedError` if ever invoked.
"""
from __future__ import annotations

import logging

from leafmachine3.core.stage import PipelineStage, WorkItem

log = logging.getLogger("leafmachine3.ruler_cf")


class RulerConversionFactor(PipelineStage):
    """Placeholder for the ruler tick-spacing conversion-factor stage (not implemented)."""

    key: str = "ruler_cf"
    name: str = "Ruler Conversion Factor"
    depends_on: tuple[str, ...] = ("ruler_classifier",)
    owns_tables: tuple[str, ...] = ("ruler_cf",)
    device_kind: str = "cpu"

    def build_model(self, device):
        """Not implemented — the tick-spacing CF has not been ported."""
        raise NotImplementedError("ruler_cf: tick-spacing CF not yet ported")

    def collect_items(self, project) -> list[WorkItem]:
        """No work: the stage is disabled and never scheduled."""
        return []

    def infer(self, item: WorkItem, model):
        """Not implemented — the tick-spacing CF has not been ported."""
        raise NotImplementedError("ruler_cf: tick-spacing CF not yet ported")

    def persist(self, project, item: WorkItem, payload) -> None:
        """Not implemented — the tick-spacing CF has not been ported."""
        raise NotImplementedError("ruler_cf: tick-spacing CF not yet ported")
