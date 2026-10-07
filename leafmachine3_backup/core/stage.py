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
    # CPU stages default to a shared-process thread pool (fine for I/O / light compute). A
    # compute-heavy CPU stage whose work is GIL-bound (pure cv2/numpy/matplotlib) sets
    # ``cpu_parallel = "process"`` to run on the spawn process pool instead -- threads give it
    # no speedup (the GIL serialises the Python parts) whereas processes scale near-linearly.
    cpu_parallel: str = "thread"       # "thread" | "process"  (only consulted for device_kind="cpu")
    # A thread-pooled CPU stage whose wall time is dominated by writing many files to disk (the
    # Reporter). `cv2.imwrite`/PIL release the GIL so threads DO scale it, but the disk saturates
    # well below cpu_cores-2 workers -- hardware_setup caps such stages at the measured disk-write knee.
    io_bound: bool = False
    # Rough per-WorkItem compute cost (seconds) for a cpu_parallel="process" stage. hardware_setup
    # divides the measured spawn-pool startup cost by this to get the batch size below which spawning
    # the pool is NOT worth it (``min_pool_items``); such small batches run serially in-process.
    est_item_seconds: float = 0.5
    # When True, ``collect_items`` may emit MORE THAN ONE WorkItem per specimen (e.g. one per leaf)
    # so the pool load-balances sub-items across workers. The executor then marks a specimen done
    # only after its LAST sub-item checkpoints, keeping per-image resume correct.
    fanout: bool = False
    # Optional STATIC worker policy, as ``((max_items, workers), ...)`` with a trailing
    # ``(None, workers)`` catch-all. The first entry whose ``max_items`` exceeds the batch size
    # wins. Empty (the default) means the stage is sized purely by free VRAM.
    #
    # This exists because VRAM is the wrong input for a cheap, short stage: it may fit 16 workers
    # while only ever justifying 2-4, and each extra worker costs real ramp (spawn + model load)
    # that a small batch never repays. The tier is an UPPER BOUND -- the VRAM planner still has
    # the final say, so a machine that cannot host the tier simply gets the most it can hold.
    worker_tiers: tuple[tuple[int | None, int], ...] = ()

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
