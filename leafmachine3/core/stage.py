"""The ONE PipelineStage contract + run context (single-writer, picklable workers).

Split across the process boundary:
  collect_items(project)          PARENT (has DB): read inputs, build picklable WorkItems.
  build_model(device)             WORKER: warm-load the export ONCE.
  infer(item, model)              WORKER: pure compute; returns the payload to persist. No DB.
  persist(project, item, payload) PARENT collector: write rows (the executor adds the
                                  image_status checkpoint in the SAME transaction).
The stage is pickled to workers, so it holds only ``cfg`` (never the DB or a live model).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkItem:
    """One unit of work. ``payload`` carries everything ``infer`` needs (no DB in workers)."""
    specimen_id: int
    payload: Any


@dataclass
class RunContext:
    """Per-run services handed to every ``stage.run()``."""
    cfg: Any
    dirs: Any
    timer: Any = None
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("leafmachine3"))

    def executor(self, stage: "PipelineStage"):
        from leafmachine3.core.executor import StageExecutor
        return StageExecutor(self.cfg, stage)


class PipelineStage(ABC):
    """One sequential stage. Concrete stages set the class constants and implement
    ``collect_items`` / ``infer`` / ``persist`` (+ ``build_model`` for GPU stages)."""

    key: str = ""
    name: str = ""
    depends_on: tuple[str, ...] = ()
    owns_tables: tuple[str, ...] = ()
    device_kind: str = "cuda"          # "cuda" | "cpu"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._model = None

    def __getstate__(self) -> dict:
        s = self.__dict__.copy()
        s["_model"] = None             # never ship a live model / DB across spawn
        return s

    # ---- hooks -------------------------------------------------------------
    @abstractmethod
    def collect_items(self, project) -> list[WorkItem]:
        ...

    @abstractmethod
    def infer(self, item: WorkItem, model: Any) -> Any:
        ...

    def build_model(self, device) -> Any:
        return None                    # cpu-only stages have no model

    @abstractmethod
    def persist(self, project, item: WorkItem, payload: Any) -> None:
        ...

    # ---- resumability (DB-backed defaults; canonical everywhere) -----------
    def is_complete(self, project) -> bool:
        return project.db.stage_state(self.key) == "done" and not self.pending(project)

    def pending(self, project) -> list[int]:
        return project.db.pending_specimens(self.key, depends_on=self.depends_on)

    # ---- the work (identical shape for all stages) -------------------------
    def run(self, project, ctx: RunContext) -> None:
        items = self.collect_items(project)
        ctx.executor(self).run(project, items)
