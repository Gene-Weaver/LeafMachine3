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
from leafmachine3.core.executor import reap_orphaned_workers
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

    # Crash recovery, part two: a previously SIGKILLed run leaves its GPU workers orphaned and
    # still holding their CUDA contexts, which silently shrinks the VRAM this run gets sized
    # against. Reap them BEFORE any device planning reads free VRAM.
    reap_orphaned_workers()

    db = ProjectDB.open_or_create(dirs.db_path)      # an existing DB here => RESUME
    db.reclaim_running()                             # crash recovery: 'running' -> 'pending'
    project = Project(cfg, dirs, db)

    ImageIngestor(cfg, db).run(restart=cfg.wipe_working_set)   # immutable originals -> symlinks

    timer = TimeReport()
    stages = build_pipeline(cfg)                     # ordered, ENABLED PipelineStage instances
    ctx = RunContext(cfg=cfg, dirs=dirs, timer=timer)

    sampler = None
    if bool(cfg.timing.get("enabled", False)):       # run-timing profiler -> reports/Timing/
        from leafmachine3.setup.timing import UtilizationSampler
        sampler = UtilizationSampler(interval=float(cfg.timing.get("sample_interval_s", 0.25)),
                                     gpu_indices=_timed_gpu_indices(cfg))
        sampler.start()
    try:
        run_pipeline(stages, project, ctx, restart=cfg.restart)
    finally:
        if sampler is not None:
            sampler.stop()
            try:
                from leafmachine3.setup.timing import write_timing_reports
                write_timing_reports(stages, timer, sampler, project, dirs.reports / "Timing",
                                     cfg, run_name=str(cfg.project.run_name))
            except Exception:                        # a timing-report failure must not fail the run
                log.exception("timing report generation failed")

    timer.log_report(log)
    log.info("LM3 complete | %s", dirs.db_path)
    return project


def _timed_gpu_indices(cfg) -> "list[int] | None":
    """The GPU ordinals the run uses, so the sampler ignores other jobs' GPUs. None = all GPUs."""
    dev = cfg.compute.get("devices", "auto")
    if isinstance(dev, (list, tuple)):
        out = [int(d) for d in dev if isinstance(d, (int, float)) or str(d).isdigit()]
        return out or None
    return None


_LIBPATH_FLAG = "LM3_CUDA_LIBPATH_SET"


def _exec_with_cuda_libpath() -> None:
    """Put the venv's bundled NVIDIA libs on ``LD_LIBRARY_PATH`` and RE-EXEC once if needed.

    onnxruntime-gpu dlopen's ``libonnxruntime_providers_cuda.so``, which links ``libcublasLt.so.12`` /
    cuDNN 9 from the ``nvidia-*`` wheels. The dynamic loader reads ``LD_LIBRARY_PATH`` only at PROCESS
    START, so exporting it from inside Python is too late -- ORT then silently falls back to
    CPUExecutionProvider and every ONNX stage runs on the CPU. Re-exec'ing this interpreter once with
    the variable set is what actually makes the CUDA provider loadable (spawn workers inherit it).
    """
    import os
    import sys

    if os.environ.get(_LIBPATH_FLAG):                # already re-exec'd (or explicitly disabled)
        return
    from leafmachine3.core.cuda_libs import nvidia_lib_dirs

    libdirs = nvidia_lib_dirs()                      # via nvidia.__path__: survives a deleted __init__.py
    os.environ[_LIBPATH_FLAG] = "1"
    if not libdirs:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = current.split(os.pathsep) if current else []
    missing = [d for d in libdirs if d not in parts]
    if not missing:
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(missing + parts)
    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)   # restart with a loader-visible path
    except Exception:                                # noqa: BLE001 - never block a run on the re-exec
        log.warning("could not re-exec for CUDA libpath; ONNX stages may fall back to CPU")


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
    """Build the override tree so CLI flags win over the YAML (deep-merged in ``Config.load``).

    Relative override paths are absolutized HERE, against the caller's CWD (plan section 3.5 rules
    2 and 3). It has to happen at this boundary: once these values are deep-merged into the config
    mapping, nothing downstream can tell a CLI-supplied ``output.dir`` from a YAML-supplied one, and
    the two take different bases -- the caller's CWD for a flag, the settings file's directory for
    YAML. Absolutize while the provenance still exists, or lose the ability to be correct.
    """
    def _from_cwd(value: str | Path) -> str:
        return str(Path(str(value)).expanduser().resolve())

    overrides: dict = {"project": {}}
    if input_dir:
        overrides["project"]["input"] = {"dirs": [_from_cwd(input_dir)]}
    if output_dir:
        overrides["project"]["output"] = {"dir": _from_cwd(output_dir)}
    if restart is not None:
        overrides["project"]["run_mode"] = {"restart": restart}
    return overrides


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console entry point (``machine3 --config LM3_settings.yaml``)."""
    # Only the CLI may do this -- it RE-EXECS this process, which would restart an embedding host
    # (pytest, a GUI, a notebook) instead of LM3. Library callers of machine3() must set
    # LD_LIBRARY_PATH themselves if they need the ONNX CUDA provider.
    _exec_with_cuda_libpath()

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
