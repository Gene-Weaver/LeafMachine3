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

Two plan section 2.2 contracts land here, both behind ``LM3_RUNTIME_V2``:

* **Calibration is an inherited subactivity, never a second root.** The parent is mid-``run_setup``
  and holds the deployment lease, so the child cannot acquire one of its own -- it inherits the
  parent's lease reference through :meth:`~leafmachine3.core.runtime.execution.ActivityHandle.launch_subactivity`,
  which composes the handoff and the one-use capability into ONE ``Popen``. Building those keywords
  by hand is what silently drops one of them.
* **Requested calibration failure is loud.** A nonzero child exit used to be downgraded to
  ``log.warning(... "keeping heuristic estimates")``, so a setup that measured nothing was
  indistinguishable from one that measured nothing *usable*. With ``strict`` the failure raises
  :class:`CalibrationError`, which the CLI turns into a nonzero exit and the GUI into a job error.
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
from leafmachine3.core.runtime._types import CALIBRATION_RUN_NAME as _CALIBRATION_RUN_NAME
from leafmachine3.core.runtime._types import EXIT_CODE_BUSY
from leafmachine3.core.runtime.execution import runtime_v2_enabled
from leafmachine3.core.runtime.launch import LaunchContribution

log = logging.getLogger("leafmachine3.setup.calibrate")

# Bundled sheets that exercise every GPU module (rulers, labels, leaves, petioles). They ship
# INSIDE the package and are resolved through importlib.resources, never as a path relative to the
# process CWD: calibration has to work identically from a wheel, a container image and this
# checkout, and "examples/images" only ever existed in the last of the three (section 3.1, row 7).
CALIBRATION_IMAGE_PACKAGE = "leafmachine3.setup.calibration_images"
DEFAULT_N_IMAGES = 6                    # enough to reach steady state; keeps calibration ~minutes
#: Re-exported from the runtime type module rather than re-spelled. The record schema (section 3.2)
#: pins the child's ``run_name`` to this string, and two definitions of it would eventually let the
#: registry and the run directory disagree about which project a calibration wrote.
CALIBRATION_RUN_NAME = _CALIBRATION_RUN_NAME

#: Name of the derived config written beside the user's settings file (section 3.5, rule 1). See
#: :func:`_write_calibration_config` for why "beside" and not "in the scratch dir".
DERIVED_CONFIG_NAME = ".lm3_calibration_settings.yaml"


class CalibrationError(RuntimeError):
    """A calibration run that was explicitly REQUESTED did not produce measurements.

    Only ever raised when the caller passed ``strict=True``. Without it every failure still
    degrades to a warning and an empty result, because an unrequested calibration falling back to
    heuristic estimates is the documented behavior and not an error (plan section 2.2).
    """


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
    activity: Any = None,
    hardware_profile: str | Path | None = None,
    strict: bool = False,
) -> dict[str, dict]:
    """Run a one-worker LM3 pass and return measured per-worker VRAM per GPU module.

    ``gpu_keys`` restricts the result to the modules that actually run on the GPU; without it
    every module is reported, and a CPU module picks up whatever VRAM was still resident from
    the GPU module before it.

    ``image_dir`` overrides the bundled sheets; unset, they come from the packaged resource, so
    the answer does not change with the directory the caller was launched from.

    ``activity`` is the parent's ``hardware_setup`` root handle. When it is a live handle the child
    is launched as an inherited ``calibration_pipeline`` subactivity (section 2.2): it runs under
    the parent's lease rather than contending for one, which is the only reason a calibration run
    can start at all while ``run_setup`` holds the deployment.

    ``hardware_profile`` is the PROVISIONAL profile the parent wrote before calling us. Pointing the
    child at it with ``LM3_HARDWARE`` is what stops the child's ``ensure_hardware_profile`` from
    running a second full setup sweep inside the measurement (section 2.2, "Calibration must not
    bootstrap itself recursively"), and it also makes the measurement happen under the configuration
    the real run will use.

    Returns ``{stage_key: {"vram_per_worker_mb": float, "seconds": float, ...}}``.

    Returns ``{}`` -- never raises -- when ``strict`` is false and calibration cannot run or
    produces nothing usable; callers fall back to the heuristic estimate. With ``strict`` every one
    of those paths raises :class:`CalibrationError` instead, because a calibration the user asked
    for and did not get is a failure and not a preference (section 2.2).
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
            return _fail(strict, f"calibration: {exc}")
    if not src_dir.is_dir():
        return _fail(strict, f"calibration: no image dir at {src_dir}")

    scratch = Path(tempfile.mkdtemp(prefix="lm3_calib_"))
    cal_cfg: Path | None = None
    try:
        images = _stage_images(src_dir, scratch / "images", n_images)
        if not images:
            return _fail(strict, f"calibration: no usable images in {src_dir}")

        out_dir = scratch / "out"
        cal_cfg = _write_calibration_config(cfg_path, scratch, gpu_index)
        _emit(on_progress, "calibrate", f"running {len(images)} images with 1 worker/GPU")

        started = time.time()
        _run_pipeline(cal_cfg, scratch / "images", out_dir, timeout_s, on_progress,
                      activity=activity, hardware_profile=hardware_profile)
        elapsed = time.time() - started

        measured = _read_timing_csv(
            out_dir / CALIBRATION_RUN_NAME / "reports" / "Timing" / "timing.csv", gpu_keys)
        if not measured:
            return _fail(strict, "calibration: run finished but no timing rows carried VRAM")

        log.info("calibration: measured %d GPU module(s) in %.0fs", len(measured), elapsed)
        for key, rec in sorted(measured.items()):
            log.info("  %-22s %6.0f MB/worker  (%s, %d items)", key, rec["vram_per_worker_mb"],
                     rec["measured_by"], rec["n_items"])
        return measured
    except CalibrationError as exc:
        # Already a precise, human-readable diagnosis from a step that knew what it was doing;
        # re-raise it unchanged under strict rather than re-wrapping it in a vaguer message.
        if strict:
            raise
        log.warning("%s -- keeping heuristic VRAM estimates", exc)
        return {}
    except Exception as exc:  # noqa: BLE001 - calibration is best effort unless it was requested
        log.exception("calibration failed")
        if strict:
            raise CalibrationError(f"calibration failed: {exc}") from exc
        return {}
    finally:
        _discard(cal_cfg)
        shutil.rmtree(scratch, ignore_errors=True)


def _fail(strict: bool, message: str) -> dict[str, dict]:
    """One place where "requested" decides between an exception and a warning (section 2.2)."""
    if strict:
        raise CalibrationError(message)
    log.warning("%s -- keeping heuristic VRAM estimates", message)
    return {}


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


def _write_calibration_config(src: Path, scratch: Path, gpu_index: Optional[int]) -> Path:
    """Derive a calibration config: same modules, but ONE worker per GPU and timing on.

    ``max_workers_per_gpu: 1`` is what pins it to a single worker -- the run-time allocator
    clamps to it -- and ``per_worker_mb: auto`` is forced so a stale explicit override cannot
    change the plan out from under the measurement.

    **Where the derived file lands is load-bearing** (section 3.5, rule 1). ``Config.resolve_path``
    resolves a relative path in the settings against the directory of the settings FILE, so writing
    the derivative into the scratch directory re-bases every relative path in the user's config to
    ``/tmp/lm3_calib_xxxx/`` -- and this repo's own examples reach their weights as
    ``../models/...``. The child then fails ``validate_ml_artifacts`` for a reason that has nothing
    to do with the machine being profiled. Under ``LM3_RUNTIME_V2`` the derivative is therefore
    written BESIDE the source config, where relative paths mean exactly what they meant before, and
    :func:`_discard` removes it afterwards. A read-only settings directory falls back to the scratch
    copy with a warning: a possibly-wrong calibration beats no calibration, and the warning names
    the reason.
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

    payload = yaml.safe_dump(cfg, sort_keys=False)
    if runtime_v2_enabled():
        beside = src.parent / DERIVED_CONFIG_NAME
        try:
            beside.write_text(payload)
            return beside
        except OSError as exc:
            log.warning("calibration: cannot write %s (%s) -- deriving into the scratch dir, so any "
                        "RELATIVE path in %s will not resolve", beside, exc, src.name)
    dst = scratch / "calibration_settings.yaml"
    dst.write_text(payload)
    return dst


def _discard(path: Path | None) -> None:
    """Remove the derived config. Never the user's own settings file, whatever else happened."""
    if path is None or path.name != DERIVED_CONFIG_NAME:
        return
    try:
        path.unlink()
    except OSError:
        log.debug("could not remove %s", path, exc_info=True)


def _run_pipeline(cfg: Path, in_dir: Path, out_dir: Path, timeout_s: float,
                  on_progress: Callable[..., None] | None, *, activity: Any = None,
                  hardware_profile: str | Path | None = None) -> None:
    """Run LM3 in a SUBPROCESS so all measured VRAM is released before we size anything.

    Returns ``None`` on success and raises :class:`CalibrationError` on every failure, including
    the timeout -- which the previous shape could not express, since ``False`` said "it did not
    work" without saying whether that was a crash, a timeout, or a busy deployment.
    """
    cmd = [sys.executable, "-m", "leafmachine3.machine3",
           "--config", str(cfg), "--input", str(in_dir), "--output", str(out_dir)]
    log.info("calibration: %s", " ".join(cmd))
    if getattr(activity, "enabled", False):
        returncode, stderr = _run_as_subactivity(activity, cmd, out_dir, timeout_s, hardware_profile)
    else:
        returncode, stderr = _run_unmanaged(cmd, timeout_s)

    if returncode != 0:
        tail = "\n".join((stderr or "").strip().splitlines()[-25:])
        if returncode == EXIT_CODE_BUSY:
            # The child asked for a lease of its own instead of inheriting ours, which means the
            # handoff did not reach it. Saying so beats reporting an opaque exit code.
            raise CalibrationError(
                f"calibration run exited {EXIT_CODE_BUSY} (deployment busy): the child did not "
                f"inherit this setup's lease\n{tail}")
        raise CalibrationError(f"calibration run exited {returncode}\n{tail}")
    _emit(on_progress, "calibrate", "run complete, reading measurements")


def _run_unmanaged(cmd: list[str], timeout_s: float) -> tuple[int, str]:
    """The pre-runtime-v2 launch, unchanged: a whole-environment copy and a blocking wait."""
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired as exc:
        raise CalibrationError(f"calibration run exceeded {timeout_s:.0f}s -- abandoning it") from exc
    return proc.returncode, proc.stderr or ""


def _run_as_subactivity(activity: Any, cmd: list[str], out_dir: Path, timeout_s: float,
                        hardware_profile: str | Path | None) -> tuple[int, str]:
    """Launch the child as an inherited ``calibration_pipeline`` subactivity (section 2.2).

    Everything the child inherits -- the lease descriptor, the one-use capability, the identity
    variables, the provisional profile -- is composed by ``launch_subactivity`` into ONE ``env`` and
    ONE set of ``Popen`` keywords. ``pass_fds`` is exhaustive, so a caller that added its own would
    silently drop the lease handoff; that is the whole reason the composer exists.

    The base environment is built here rather than left to default, because ``LM3_HARDWARE`` may
    already be set in this process and the composer REFUSES to pick a winner between two values for
    one variable. Stripping it (and its one-release legacy alias) leaves exactly one source of truth
    for the child: the provisional profile contribution below.
    """
    base_env = {k: v for k, v in os.environ.items() if k not in _HARDWARE_ENV_NAMES}
    base_env.setdefault("PYTHONUNBUFFERED", "1")

    contributions = []
    if hardware_profile is not None:
        contributions.append(LaunchContribution(
            name="provisional-profile",
            env={paths.ENV_HARDWARE: str(Path(hardware_profile).resolve())},
        ))

    with activity.launch_subactivity(
        run_name=CALIBRATION_RUN_NAME,
        run_dir=out_dir / CALIBRATION_RUN_NAME,
        env=base_env,
        contributions=contributions,
    ) as launch:
        proc = subprocess.Popen(cmd, env=launch.env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, **launch.popen_kwargs)
        launch.attach(proc)          # the root can now join or terminate this child (section 3.3)
        try:
            _, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            # Kill it here rather than leaving it to the root's join: while it lives it holds the
            # inherited lease reference, and the deployment stays occupied after we give up on it.
            proc.kill()
            _, stderr = proc.communicate()
            launch.completed(returncode=proc.returncode)
            raise CalibrationError(
                f"calibration run exceeded {timeout_s:.0f}s -- abandoning it\n"
                f"{(stderr or '').strip()[-2000:]}") from exc
        launch.completed(returncode=proc.returncode)
        return proc.returncode, stderr or ""


#: ``LM3_HARDWARE`` and the one-release legacy spelling ``core.paths`` still honors. Both have to be
#: stripped from a calibration child's base environment, or the legacy name would quietly outrank
#: the provisional profile we just wrote.
_HARDWARE_ENV_NAMES = frozenset(
    {paths.ENV_HARDWARE, paths.LEGACY_ENV_ALIASES.get(paths.ENV_HARDWARE, "")} - {""}
)


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
