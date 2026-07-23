"""leafmachine3.machine3 -- the LeafMachine3 pipeline manager.

Mirrors the STRUCTURE of LeafMachine2's ``machine.py`` (set the multiprocessing
start-method -> load config -> build output dirs -> start logging -> validate ML
artifacts -> build the SQL-backed :class:`Project` -> ingest images -> run the
ordered stages with per-stage GPU cleanup -> emit a time report) but stays SMALL:
every processing step is a self-contained ``PipelineStage`` and all persistence
lives behind :class:`ProjectDB`, so this module is orchestration ONLY -- no model
code, no SQL, no threading.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Sequence

from leafmachine3.core.config import Config
from leafmachine3.core.db import ProjectDB
from leafmachine3.core.dirs import build_dirs
from leafmachine3.core.ingest import ImageIngestor
from leafmachine3.core.logging_setup import start_logging
from leafmachine3.core.project import Project
from leafmachine3.core.stage import RunContext
from leafmachine3.core.timing import TimeReport
from leafmachine3.core.validate import validate_ml_artifacts
from leafmachine3.pipeline import build_pipeline, run_pipeline
from leafmachine3.setup.hardware_setup import ensure_hardware_profile

log = logging.getLogger("leafmachine3")

RestartArg = None | str | list[str]


def machine3(
    cfg_path: str | Path,
    *,
    restart: RestartArg = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Project:
    """Run the full LM3 pipeline for ``cfg_path`` and return the resolved project.

    Parameters
    ----------
    cfg_path:
        Path to ``LM3_settings.yaml``.
    restart:
        ``None`` resumes; a stage key (or list of keys) reruns those stages and
        everything downstream; ``"all"`` rebuilds the whole project.
    input_dir / output_dir:
        Optional CLI overrides for ``project.input.dirs`` / ``project.output.dir``.
    """
    _set_spawn()                                     # MUST precede any CUDA import

    cfg = Config.load(cfg_path, overrides=_cli_overrides(input_dir, output_dir, restart))
    cfg.validate()                                   # raises on missing / incoherent settings

    ensure_hardware_profile(cfg)                     # bind hardware_settings.yaml; auto-run setup if ABSENT

    dirs = build_dirs(cfg)                           # working set, _tmp, crops, reports, logs, db
    start_logging(dirs, cfg)
    log.info("LM3 start | out=%s | restart=%s", dirs.root, cfg.restart)

    validate_ml_artifacts(cfg)                       # fail fast if an export is missing

    db = ProjectDB.open_or_create(dirs.db_path)      # an existing DB here => RESUME
    db.reclaim_running()                             # crash recovery: 'running' -> 'pending'
    project = Project(cfg, dirs, db)

    ImageIngestor(cfg, db).run(restart=cfg.wipe_working_set)   # immutable originals -> symlinks

    timer = TimeReport()
    stages = build_pipeline(cfg)                     # ordered, ENABLED PipelineStage instances
    ctx = RunContext(cfg=cfg, dirs=dirs, timer=timer)
    run_pipeline(stages, project, ctx, restart=cfg.restart)

    timer.log_report(log)
    log.info("LM3 complete | %s", dirs.db_path)
    return project


def _set_spawn() -> None:
    """Force the ``spawn`` start-method so workers begin with no inherited CUDA state."""
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass                                         # already set by an outer caller


def _cli_overrides(
    input_dir: str | Path | None,
    output_dir: str | Path | None,
    restart: RestartArg,
) -> dict:
    """Build the override tree so CLI flags win over the YAML (deep-merged in ``Config.load``)."""
    overrides: dict = {"project": {}}
    if input_dir:
        overrides["project"]["input"] = {"dirs": [str(input_dir)]}
    if output_dir:
        overrides["project"]["output"] = {"dir": str(output_dir)}
    if restart is not None:
        overrides["project"]["run_mode"] = {"restart": restart}
    return overrides


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console entry point (``machine3 --config LM3_settings.yaml``)."""
    parser = argparse.ArgumentParser(prog="machine3", description="Run the LeafMachine3 pipeline.")
    parser.add_argument("--config", required=True, help="path to LM3_settings.yaml")
    parser.add_argument(
        "--restart",
        action="append",
        default=None,
        metavar="STAGE|all",
        help="rerun a stage (repeatable) and its dependents, or 'all'",
    )
    parser.add_argument("--input", default=None, help="override the YAML input dir")
    parser.add_argument("--output", default=None, help="override the YAML output dir")
    args = parser.parse_args(argv)

    restart: RestartArg = "all" if args.restart == ["all"] else args.restart   # "all" | list | None
    machine3(args.config, restart=restart, input_dir=args.input, output_dir=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
