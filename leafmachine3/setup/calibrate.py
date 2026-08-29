"""leafmachine3.setup.calibrate -- measure what a GPU module ACTUALLY costs in VRAM.

The worker-sizing question ("how many copies of this module fit on this card?") is only as
good as the per-worker VRAM figure it divides by. That figure used to come from a hardcoded
lookup table in :mod:`hardware_setup` -- 3500 MB resident + 350 MB/image for any detector --
which over-estimated the real cost by more than 5x and so throttled every GPU module to a
fraction of the workers the card could hold.

This module replaces the guess with a measurement. It runs LM3 for real over a small image
set with exactly ONE worker per GPU, samples NVML per-process VRAM throughout, and reports
the high-water mark each module reached. One worker is the whole point: with a single worker
the measured peak IS the per-worker cost, so the run-time allocator can multiply it out.

GPU modules cannot be measured in isolation -- ``landmark_detector`` needs leaf crops from
``plant_detector``, ``ruler_classifier`` needs ruler crops from ``archival_detector`` -- so
calibration is a real end-to-end run rather than a set of synthetic probes. Running the real
thing also captures costs a synthetic probe would miss, notably the ~300-500 MB CUDA context
each worker process pays.

Entry point: :func:`calibrate_gpu_stages`. Invoked by ``python -m leafmachine3.setup
--calibrate`` and by the desktop app's Re-profile button.
"""
from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from leafmachine3.core import paths

log = logging.getLogger("leafmachine3.setup.calibrate")

# Bundled sheets that exercise every GPU module (rulers, labels, leaves, petioles). They ship
# INSIDE the package and are resolved through importlib.resources, never as a path relative to the
# process CWD: calibration has to work identically from a wheel, a container image and this
# checkout, and "examples/images" only ever existed in the last of the three (section 3.1, row 7).
CALIBRATION_IMAGE_PACKAGE = "leafmachine3.setup.calibration_images"
DEFAULT_N_IMAGES = 6                    # enough to reach steady state; keeps calibration ~minutes
CALIBRATION_RUN_NAME = "_lm3_calibration"


def default_image_dir() -> Path:
    """The packaged calibration sheets.

    A function rather than a module constant on purpose: a constant is bound at import time, which
    is exactly how the old CWD-relative default survived so long -- it looked resolved when it was
    only deferred to whoever happened to read it.

    Raises :class:`leafmachine3.core.paths.PackagedResourceError` when the data did not ship and no
    development checkout is detectable; callers treat that as "no calibration" rather than an abort.
    """
    return paths.calibration_images_dir(package=CALIBRATION_IMAGE_PACKAGE)


def calibrate_gpu_stages(
    cfg_path: str | Path,
    *,
    gpu_index: Optional[int] = None,
    gpu_keys: Optional[set] = None,
    n_images: int = DEFAULT_N_IMAGES,
    image_dir: str | Path | None = None,
    timeout_s: float = 3600.0,
    on_progress: Callable[..., None] | None = None,
) -> dict[str, dict]:
    """Run a one-worker LM3 pass and return measured per-worker VRAM per GPU module.

    ``gpu_keys`` restricts the result to the modules that actually run on the GPU; without it
    every module is reported, and a CPU module picks up whatever VRAM was still resident from
    the GPU module before it.

    ``image_dir`` overrides the bundled sheets; unset, they come from the packaged resource, so
    the answer does not change with the directory the caller was launched from.

    Returns ``{stage_key: {"vram_per_worker_mb": float, "seconds": float, ...}}``.
    Returns ``{}`` -- never raises -- if calibration cannot run or produces nothing usable;
    callers fall back to the heuristic estimate.
    """
    cfg_path = Path(cfg_path).resolve()
    if image_dir is not None:
        src_dir = Path(image_dir).expanduser()
    else:
        try:
            src_dir = default_image_dir()
        except paths.PathsError as exc:
            # A packaging fault, not a user error: say so plainly instead of reporting a missing
            # directory the user never chose and could not have created.
            log.warning("calibration: %s -- keeping heuristic VRAM estimates", exc)
            return {}
    if not src_dir.is_dir():
        log.warning("calibration: no image dir at %s -- keeping heuristic VRAM estimates", src_dir)
        return {}

    scratch = Path(tempfile.mkdtemp(prefix="lm3_calib_"))
    try:
        images = _stage_images(src_dir, scratch / "images", n_images)
        if not images:
            log.warning("calibration: no usable images in %s -- keeping heuristic estimates", src_dir)
            return {}

        out_dir = scratch / "out"
        cal_cfg = _write_calibration_config(cfg_path, scratch / "calibration_settings.yaml", gpu_index)
        _emit(on_progress, "calibrate", f"running {len(images)} images with 1 worker/GPU")

        started = time.time()
        ok = _run_pipeline(cal_cfg, scratch / "images", out_dir, timeout_s, on_progress)
        elapsed = time.time() - started
        if not ok:
            return {}

        measured = _read_timing_csv(
            out_dir / CALIBRATION_RUN_NAME / "reports" / "Timing" / "timing.csv", gpu_keys)
        if not measured:
            log.warning("calibration: run finished but no timing rows carried VRAM -- keeping estimates")
            return {}

        log.info("calibration: measured %d GPU module(s) in %.0fs", len(measured), elapsed)
        for key, rec in sorted(measured.items()):
            log.info("  %-22s %6.0f MB/worker  (%s, %d items)", key, rec["vram_per_worker_mb"],
                     rec["measured_by"], rec["n_items"])
        return measured
    except Exception:  # noqa: BLE001 - calibration is best effort; a failure must not break setup
        log.exception("calibration failed -- keeping heuristic VRAM estimates")
        return {}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _stage_images(src: Path, dst: Path, n: int) -> list[Path]:
    """Copy up to ``n`` images into an isolated dir so the run cannot touch user data."""
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    picks = sorted(p for p in src.iterdir() if p.suffix.lower() in exts)[: max(1, n)]
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for p in picks:
        target = dst / p.name
        shutil.copy2(p, target)
        out.append(target)
    return out


def _write_calibration_config(src: Path, dst: Path, gpu_index: Optional[int]) -> Path:
    """Derive a calibration config: same modules, but ONE worker per GPU and timing on.

    ``max_workers_per_gpu: 1`` is what pins it to a single worker -- the run-time allocator
    clamps to it -- and ``per_worker_mb: auto`` is forced so a stale explicit override cannot
    change the plan out from under the measurement.
    """
    cfg = yaml.safe_load(src.read_text()) or {}

    compute = cfg.setdefault("compute", {})
    vram = compute.setdefault("vram", {})
    vram["max_workers_per_gpu"] = 1              # the measurement's whole premise
    vram["per_worker_mb"] = "auto"
    if gpu_index is not None:
        compute["devices"] = [int(gpu_index)]
    compute["mock"] = False                      # a mocked run would measure nothing

    cfg["timing"] = {"enabled": True, "sample_interval_s": 0.2}

    project = cfg.setdefault("project", {})
    project["run_name"] = CALIBRATION_RUN_NAME
    project.setdefault("run_mode", {})["overwrite"] = True   # never resume a stale calibration

    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dst


def _run_pipeline(cfg: Path, in_dir: Path, out_dir: Path, timeout_s: float,
                  on_progress: Callable[..., None] | None) -> bool:
    """Run LM3 in a SUBPROCESS so all measured VRAM is released before we size anything."""
    cmd = [sys.executable, "-m", "leafmachine3.machine3",
           "--config", str(cfg), "--input", str(in_dir), "--output", str(out_dir)]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    log.info("calibration: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        log.warning("calibration run exceeded %.0fs -- abandoning it", timeout_s)
        return False
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-25:])
        log.warning("calibration run exited %d -- keeping heuristic estimates\n%s", proc.returncode, tail)
        return False
    _emit(on_progress, "calibrate", "run complete, reading measurements")
    return True


def _read_timing_csv(path: Path, gpu_keys: Optional[set] = None) -> dict[str, dict]:
    """Pull per-module measured VRAM out of the timing report the calibration run wrote.

    Prefers ``vram_own_max_mb`` (NVML per-process accounting -- ours no matter what else is on
    the card) and falls back to ``vram_delta_max_mb`` (device total over a pre-run baseline).
    Modules that never touched the GPU are dropped rather than recorded as 0.
    """
    if not path.exists():
        log.warning("calibration: no timing CSV at %s", path)
        return {}

    out: dict[str, dict] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("stage") or "").strip()
            if not key or (gpu_keys is not None and key not in gpu_keys):
                continue
            own = _num(row.get("vram_own_max_mb"))
            delta = _num(row.get("vram_delta_max_mb"))
            vram = own if own > 1.0 else delta
            if vram <= 1.0:                       # CPU-only module, or nothing measured
                continue
            out[key] = {
                "vram_per_worker_mb": round(vram, 1),
                "measured_by": "nvml_process" if own > 1.0 else "device_delta",
                "seconds": _num(row.get("seconds")),
                "n_items": int(_num(row.get("n_items"))),
            }
    return out


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _emit(cb: Callable[..., None] | None, phase: str, message: str) -> None:
    if cb is None:
        return
    try:
        cb(phase, message)
    except Exception:  # noqa: BLE001 - progress reporting is best effort
        pass
