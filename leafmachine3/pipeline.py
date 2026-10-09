"""leafmachine3.pipeline -- the canonical stage order + the run loop.

``STAGE_ORDER`` is the single source of truth for stage identity and sequence.
Every layer imports :data:`STAGE_KEYS` from here (``PipelineStage.key``,
``project_status`` seeding, ``depends_on`` gating), so the key drift that used to
break the ledger foreign keys is impossible by construction. Adding a stage is a
single entry in :data:`STAGE_ORDER`.

The module is orchestration only -- it holds no model code, no SQL, and no
threading. Each stage saturates the hardware internally through its executor; the
run loop simply walks the enabled stages in order, honours the resume ledger, and
releases GPU memory between stages.
"""
from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Iterable

from leafmachine3.modules.mp_conversion_factor import MPConversionFactor
from leafmachine3.modules.archival_detector import ArchivalDetector
from leafmachine3.modules.plant_detector import PlantDetector
from leafmachine3.modules.specimen_segmenter import SpecimenSegmenter
from leafmachine3.modules.phenology_detector import PhenologyDetector
from leafmachine3.modules.ruler_classifier import RulerClassifier
from leafmachine3.modules.ruler_conversion_factor import RulerConversionFactor
from leafmachine3.modules.leaf_segmenter import LeafSegmenter
from leafmachine3.modules.morphology import Morphology
from leafmachine3.modules.landmark_detector import LandmarkDetector
from leafmachine3.modules.landmark_measurements import LandmarkMeasurements
from leafmachine3.modules.leaf_orientation import LeafOrientation
from leafmachine3.modules.bilateral_symmetry import BilateralSymmetry
from leafmachine3.modules.petiole_width import PetioleWidth
from leafmachine3.modules.metric_grounding import MetricGrounding
from leafmachine3.modules.reporter import Reporter
from leafmachine3.modules.ect import ECT, enforce_report_deps
from leafmachine3.modules.momocs import Momocs, enforce_report_deps as enforce_momocs_report_deps

if TYPE_CHECKING:  # pragma: no cover - typing only
    from leafmachine3.core.config import Config
    from leafmachine3.core.project import Project
    from leafmachine3.core.stage import PipelineStage, RunContext

log = logging.getLogger("leafmachine3")

# THE one place the pipeline is declared. Strictly sequential by DATA dependency;
# each stage saturates the hardware INTERNALLY. The extensible slot is the gap
# between ``LeafSegmenter`` and ``MetricGrounding`` -- append future stages there
# and nothing else changes.
STAGE_ORDER: tuple[type, ...] = (
    MPConversionFactor,      # 0  resolution -> CF linear prediction (specimen.cf_px_per_cm_predicted_by_mp)
    ArchivalDetector,        # 1  archival sheet elements (Ruler, Barcode, Label, ...)
    PlantDetector,           # 2  plant organs (Leaf_WHOLE, Leaf_PARTIAL, ...)
    SpecimenSegmenter,       # 2b whole-sheet plant-vs-background binary mask (+ paperclean)
    PhenologyDetector,       # 3  plant boxes -> leaves / flowers / fruits presence
    RulerClassifier,         # 4  Ruler crops -> unit-type            (CF step a)
    RulerConversionFactor,   # 5  unit-type + crop -> CF (px per cm)  (CF step b)
    LeafSegmenter,           # 6  Leaf_WHOLE crops -> instance masks
    Morphology,              # 7  leaf masks -> shape metrics + rotated (min) bounding box
    LandmarkDetector,        # 8  leaf crops -> 31-keypoint pose (mid15_pet5)
    LandmarkMeasurements,    # 9  keypoints -> traces / extent / width / apex+base angles / curvature
    LeafOrientation,         # 10 keypoints -> upright rotation (tip up) stored on leaf_morphology
    PetioleWidth,            # 11 Petiole mask + landmark centerline -> median perpendicular width
    BilateralSymmetry,       # 12 midvein-axis symmetry + archetype score (leaf-quality ranking)
    # <---------- extensible slot: new PipelineStage subclasses go here ---------->
    MetricGrounding,         # 12 apply CF: area_px -> cm^2, perimeter_px -> cm
    Reporter,                # 13 overlays / mask PNGs / RGB crops / oriented + original leaf products
    ECT,                     # 14 Euler Characteristic Transform per oriented leaf (consumes Reporter masks)
    Momocs,                  # 15 Momocs/Momocs2 leaf images + outlines (consumes Reporter masks)
)
STAGE_KEYS: tuple[str, ...] = tuple(cls.key for cls in STAGE_ORDER)  # the canonical id set


def build_pipeline(cfg: "Config") -> list["PipelineStage"]:
    """Instantiate the ENABLED stages in canonical order.

    Disabled stages are omitted from the returned list; they are satisfied for
    downstream ``depends_on`` gating by :func:`run_pipeline`, which marks them
    complete-with-no-work before the loop begins.
    """
    enforce_report_deps(cfg)                      # ECT (if on) forces the Reporter to export the mask it needs
    enforce_momocs_report_deps(cfg)               # so does Momocs (its holes-filled product)
    stages = [cls(cfg) for cls in STAGE_ORDER if cfg.is_enabled(cls.key)]
    _assert_dependencies_precede(stages)          # raises on a mis-ordered depends_on graph
    return stages


def run_pipeline(
    stages: list["PipelineStage"],
    project: "Project",
    ctx: "RunContext",
    *,
    restart: None | str | list[str] = None,
) -> None:
    """Run the enabled ``stages`` in order against ``project``.

    First every DISABLED canonical stage is marked complete-with-no-work so that a
    downstream ``depends_on`` on it is satisfied. Then the restart directive
    cascade-cleans stale output. Finally each enabled stage is skipped when already
    complete, otherwise run to completion with a resumable ledger and GPU cleanup
    between stages.
    """
    enabled_keys = {stage.key for stage in stages}
    for key in STAGE_KEYS:
        if key not in enabled_keys:
            project.db.mark_stage_complete_no_work(key)

    _apply_restart(project, stages, restart)      # cascade-clean stale output FIRST

    for stage in stages:
        _sync_settings_hash(project, stage, ctx.cfg, stages)   # config-drift => force a rerun (+ dependents)
        if stage.is_complete(project):
            log.info("skip  %-22s (up to date)", stage.key)
            continue

        log.info("start %-22s (%s)", stage.key, stage.name)
        project.db.mark_stage_running(stage.key, n_total=len(stage.pending(project)))
        with ctx.timer.stage(stage.key):
            try:
                stage.run(project, ctx)           # saturates all GPUs / CPUs internally
            except Exception:
                project.db.mark_stage_error(stage.key)   # leaves a resumable ledger
                log.exception("FAILED %s", stage.key)
                raise
        # Mark 'done' ONLY when the per-image ledger is truly drained (else it stays
        # resumable and the next run re-enters to finish the stragglers).
        if not stage.pending(project):
            project.db.mark_stage_done(stage.key)
        _gpu_cleanup()                            # release VRAM before the next model


def _apply_restart(
    project: "Project",
    stages: list["PipelineStage"],
    restart: None | str | list[str],
) -> None:
    """Honour ``--restart``.

    ``'all'`` resets every canonical stage; ``KEY[,KEY]`` resets those stages AND
    their transitive dependents (their inputs changed, so their outputs are stale).
    """
    if restart is None:
        return
    by_key = {stage.key: stage for stage in stages}
    keys = STAGE_KEYS if restart == "all" else _closure(stages, _as_list(restart))
    project.db.reset_stages(keys, by_key)         # purges method rows + crops + ledger, in txn


def _sync_settings_hash(
    project: "Project", stage: "PipelineStage", cfg: "Config", stages: list["PipelineStage"],
) -> None:
    """Force a ``done`` stage back to pending when its settings or model changed.

    The hash covers the resolved ``modules.<key>`` block plus the model artifact's
    size/mtime, so any edit to the YAML or swap of the export invalidates the stage
    AND its transitive dependents (their inputs changed) and the DB always corresponds
    to the config -- e.g. swapping the specimen/plant/leaf model also rebuilds the
    overlays and mask exports that consume it. ``depends_on`` ordering guarantees every
    reset dependent still lies ahead in the loop, so it re-runs this same session.
    """
    new_hash = cfg.stage_settings_hash(stage.key)
    if (
        project.db.stage_state(stage.key) == "done"
        and project.db.stage_settings_hash(stage.key) != new_hash
    ):
        keys = _closure(stages, [stage.key])          # the stage + its transitive dependents
        log.warning("CONFIG DRIFT: %s settings changed -> rerun %s", stage.key, list(keys))
        project.db.reset_stages(keys, {s.key: s for s in stages})
    project.db.set_stage_settings_hash(stage.key, new_hash)


def _closure(stages: list["PipelineStage"], keys: Iterable[str]) -> tuple[str, ...]:
    """Return ``keys`` plus every transitive dependent, in canonical order.

    A dependent is any stage that lists a member of the growing set in its
    ``depends_on``. Only the supplied ``stages`` participate, so a restart of a
    disabled or absent key simply resets that key alone.
    """
    seeds = {str(k) for k in keys}
    reachable = set(seeds)
    changed = True
    while changed:
        changed = False
        for stage in stages:
            if stage.key in reachable:
                continue
            if any(dep in reachable for dep in stage.depends_on):
                reachable.add(stage.key)
                changed = True
    # keep canonical ordering; append any non-canonical seed keys deterministically.
    ordered = [k for k in STAGE_KEYS if k in reachable]
    ordered += sorted(k for k in reachable if k not in STAGE_KEYS)
    return tuple(ordered)


def _gpu_cleanup() -> None:
    """Release Python and CUDA caches between stages (best effort)."""
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _as_list(value: str | list[str]) -> list[str]:
    return value if isinstance(value, list) else [value]


def _assert_dependencies_precede(stages: list["PipelineStage"]) -> None:
    """Raise ``ValueError`` if any enabled stage precedes one of its enabled deps.

    Dependencies that are disabled (absent from ``stages``) are satisfied out of
    band by ``mark_stage_complete_no_work``, so only ordering among the present
    stages is checked.
    """
    seen: set[str] = set()
    for stage in stages:
        present_deps = {dep for dep in stage.depends_on if dep in {s.key for s in stages}}
        missing = present_deps - seen
        if missing:
            raise ValueError(
                f"stage {stage.key!r} depends on {sorted(missing)} which do not precede it "
                f"in STAGE_ORDER"
            )
        seen.add(stage.key)


__all__ = ["STAGE_ORDER", "STAGE_KEYS", "build_pipeline", "run_pipeline"]
