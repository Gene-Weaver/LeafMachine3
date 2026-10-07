"""leafmachine3.setup.hardware_setup -- LM3_Setup: the one-time (rerunnable) profiler.

Runs ONCE per install (or on demand) to measure THIS specific machine and write the
hardware profile -- the tuned, machine-derived layer every module reads for worker counts,
batch sizes, queue sizes, tmp location, GPU availability + VRAM, precision, and the bound
execution provider.

The profile is deployment-scoped and machine-keyed
(``<user-config>/lm3/<deployment>/hardware_settings.<machine-key>.yaml``, see
:func:`hardware_profile_path`), NOT a ``hardware_settings.yaml`` beside whatever directory the
process was started in. That older location is still adopted once, by copy, for one release.

The profiler is deliberately conservative and dependency-light: every probe degrades
gracefully so that a CPU-only host (or a ``compute.mock`` run) always produces a valid
profile without a GPU, ``pynvml``, ``onnxruntime`` or ``psutil`` installed. The per-stage
"dry-run sweep" is a VRAM-fit search over candidate ``(batch, workers_per_gpu)`` combos
that keeps the throughput-optimal plan fitting the GPU's VRAM budget; it needs no model
weights, so it never touches user data -- it only measures and writes the YAML.

CLI:  ``python -m leafmachine3.setup [--optimize] [--quick] [--force] [--calibrate] [--tmp DIR]``
GUI:  the Electron "Hardware Setup" panel button -> ``POST /v1/setup``.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import yaml

from leafmachine3 import __version__ as _pkg_version
from leafmachine3.core import paths

log = logging.getLogger("leafmachine3.setup")

#: Stamped into the hardware profile so a stale profile is detected across an LM3 upgrade. Derived
#: from the package version rather than restated, so the profile, /healthz and the runtime launch
#: manifest can never disagree about which LM3 wrote what.
LM3_VERSION = _pkg_version

ProgressCB = Callable[..., None]


def hardware_profile_path(cfg: Any = None) -> Path:
    """Where THIS deployment's profile for THIS machine lives (section 3.1, table row 2).

    ``LM3_HARDWARE``, else
    ``<user-config>/lm3/<canonical-deployment>/hardware_settings.<machine-key>.yaml``, else a
    one-release legacy adopt that COPIES a profile sitting beside the resolved settings file into
    that path and uses the copy.

    Two properties are load-bearing and neither is obvious:

    * **Deployment-scoped, not machine-scoped.** ``run_setup`` sizes GPU stages against the
      *selected* ``compute.devices`` subset, instantiates only the stages the supplied config
      enables, and fingerprints config-derived model hashes. Two named deployments hold separate
      leases, so nothing stops them tuning concurrently -- one pinned to GPU 0, one to GPU 1 --
      and a machine-scoped file would let each carry the other's measurements forward. The
      deployment lock cannot protect a path outside the deployment.
    * **Resolved per call, never a module constant.** The old ``HW_PATH`` was bound at import, so
      it silently named a different file from every launch directory, and a spawned child could
      not be pointed anywhere by its environment. Resolution has to happen after the process has
      its environment, not while it is being imported.

    ``cfg`` is accepted for call-site symmetry but no longer moves the path: ``--config`` selects
    settings and nothing else.

    **Pure.** Resolution never writes. The one-release legacy adopt is
    :func:`migrate_legacy_profile`, called from the two controlled entry points below.
    """
    return paths.hardware_profile_path()


def migrate_legacy_profile(cfg: Any = None) -> Path | None:
    """Adopt a beside-the-settings ``hardware_settings.yaml`` once, at a controlled entry point.

    Deliberately separate from :func:`hardware_profile_path`: a function that RESOLVES a path is
    called from status snapshots and health probes many times a second, and one that WRITES must
    not be. Returns the adopted target, or ``None`` when there was nothing to adopt.
    """
    settings_file = getattr(cfg, "source_path", None) if cfg is not None else None
    return paths.migrate_legacy_hardware_profile(settings_file=settings_file)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class GpuInfo:
    """One GPU as reported by NVML / ``nvidia-smi``."""

    index: int
    name: str
    total_vram_mb: int
    free_vram_mb: int
    driver: str


@dataclass
class Fingerprint:
    """Identifies the machine + software so a STALE profile is detected on a run."""

    os: str
    cpu: str
    cpu_cores: int
    ram_gb: int
    gpus: list                      # [[name, total_vram_mb], ...]
    driver: str
    ort_version: str
    ort_providers: list
    model_hashes: dict              # {stage_key: "sha256:..."} -> rerun if a model changed
    lm3_version: str


@dataclass
class HardwareSettings:
    """The complete, serialisable machine profile written to the resolved profile path."""

    fingerprint: Fingerprint
    provider: str
    precision: str
    gpus: list                      # [asdict(GpuInfo), ...]
    cpu_cores: int
    ram_gb: int
    tmp_dir: str
    io_workers: int
    stages: dict
    generated_at: str = ""
    lm3_version: str = LM3_VERSION


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def ensure_hardware_profile(cfg: Any) -> Path:
    """Bind a hardware profile onto ``cfg``, auto-running LM3_Setup on first use.

    If the profile is MISSING, the full LM3_Setup runs once (the only time it self-triggers)
    so the first run is tuned. If it EXISTS it is never re-run automatically: a drifted
    fingerprint only LOGS a suggestion to rerun. The profile is then bound so every ``auto``
    in the config resolves from it.

    The path comes from :func:`hardware_profile_path`, so it is the same file whatever directory
    the run was launched from -- and resolving it here is also what performs the one-release
    legacy adopt, before the "is it missing?" question is asked.
    """
    migrate_legacy_profile(cfg)          # controlled entry point: adopt once, here
    hw_path = hardware_profile_path(cfg)
    if not hw_path.exists():
        log.info(
            "no hardware profile at %s -- running LM3_Setup once to tune this machine "
            "(cached for every future run)", hw_path
        )
        run_setup(cfg, optimize=True)
    else:
        try:
            if _load(hw_path).fingerprint != _fingerprint(cfg):
                log.warning(
                    "hardware / driver / models changed since the last LM3_Setup -- this run uses the "
                    "EXISTING profile at %s; rerun `python -m leafmachine3.setup` to re-tune when "
                    "convenient", hw_path
                )
        except Exception as exc:  # noqa: BLE001 - a malformed profile must not abort the run
            log.warning("could not read %s (%s) -- rebuilding profile", hw_path, exc)
            run_setup(cfg, optimize=True, force=True)

    try:
        cfg.bind_hardware(_load(hw_path))
    except Exception as exc:  # noqa: BLE001 - never let profile binding crash a run
        log.warning("could not bind %s (%s) -- proceeding without a tuned profile", hw_path, exc)
    return hw_path


def run_setup(
    cfg: Any,
    *,
    optimize: bool = True,
    quick: bool = False,
    force: bool = False,
    calibrate: bool = False,
    tmp_override: str | None = None,
    on_progress: ProgressCB | None = None,
) -> Path:
    """Profile the machine and (re)write this deployment's hardware profile.

    Idempotent: a valid profile whose fingerprint already matches this machine is kept
    unless ``force``. Returns the path written (or reused).

    With ``calibrate``, GPU modules are additionally MEASURED -- a real one-worker LM3 run
    over a handful of bundled images, sampling NVML per-process VRAM -- and the measured
    per-worker cost replaces the heuristic estimate. That costs a few minutes, so it is
    opt-in; without it, any measurement from a previous calibration is carried forward
    rather than discarded.
    """
    migrate_legacy_profile(cfg)          # controlled entry point: adopt once, here
    hw_path = hardware_profile_path(cfg)
    fingerprint = _fingerprint(cfg)
    if hw_path.exists() and not force:
        try:
            if _load(hw_path).fingerprint == fingerprint:
                log.info("%s is current -- nothing to do (use --force to redo)", hw_path)
                return hw_path
        except Exception:  # noqa: BLE001 - fall through and rewrite a broken file
            pass

    gpus = _discover_gpus()
    cpu_cores, ram_gb = _discover_cpu_ram()
    provider = _probe_bound_provider(cfg)
    tmp_dir = Path(tmp_override) if tmp_override else _choose_tmp_dir(cfg, min_free_gb=50)
    io_workers = max(1, cpu_cores - 2)

    # Size GPU stages against the GPU(s) the run will actually use (compute.devices), not gpus[0]:
    # if the user pins GPU 1, workers_per_gpu is computed from GPU 1's free VRAM. 'auto' -> all GPUs.
    chosen = _chosen_gpu_indices(cfg)
    sweep_gpus = [g for g in gpus if g.index in chosen] if chosen else list(gpus)
    sweep_gpus = sweep_gpus or list(gpus)
    stages: dict[str, dict] = {}
    for stage in _gpu_stages(cfg):
        stages[stage.key] = (
            _optimize_stage(stage, sweep_gpus, provider, quick=quick,
                            on_progress=on_progress, cfg=cfg)
            if optimize and sweep_gpus
            else _heuristic_stage(stage, sweep_gpus)
        )
    # Real per-worker VRAM beats the heuristic by a wide margin (the estimates over-counted
    # detectors ~5x), so measure when asked and otherwise carry any earlier measurement forward
    # -- a routine re-profile must not silently downgrade measured numbers back to guesses.
    _apply_vram_measurements(stages, cfg, sweep_gpus, calibrate=calibrate,
                             on_progress=on_progress, profile_path=hw_path)
    # CPU stages: light thread-pooled stages get flat io_workers; process-pooled stages
    # (cpu_parallel="process") get the measured process-scaling knee (so the spawn pool isn't
    # over-subscribed); disk-write-bound thread stages (io_bound, e.g. the Reporter) get the
    # measured parallel-write knee (a disk saturates well below cpu_cores-2 concurrent writers).
    cpu_stages = _cpu_stages(cfg)
    proc_stages = [s for s in cpu_stages if getattr(s, "cpu_parallel", "thread") == "process"]
    io_stages = [s for s in cpu_stages
                 if getattr(s, "io_bound", False) and getattr(s, "cpu_parallel", "thread") != "process"]
    proc_workers = (_benchmark_cpu_workers(cpu_cores, on_progress)
                    if (optimize and not quick and proc_stages) else max(1, cpu_cores - 2))
    # spawn cost of a full process pool; per stage, the batch below which the pool isn't worth it is
    # spawn_overhead / that stage's est_item_seconds (a fast-item stage like ect needs a bigger batch).
    spawn_s = (_benchmark_spawn_overhead(proc_workers)
               if (optimize and not quick and proc_stages) else 0.0)
    disk_knee, disk_mbps = (_benchmark_disk_writers(_reports_fs_dir(cfg, tmp_dir), on_progress)
                            if (optimize and not quick and io_stages) else (8, 0.0))
    # imwrite is buffered (not fsync'd), so the raw probe is pessimistic; a disk-write-bound stage
    # gains from concurrent ENCODE up to a point but over-subscribing the disk thrashes. Clamp the
    # measured knee to the sane 8..16 band (a fast disk -> 16, a slow one -> 8) and never exceed io.
    disk_workers = min(io_workers, max(8, min(16, disk_knee)))
    for stage in cpu_stages:
        if getattr(stage, "cpu_parallel", "thread") == "process":
            est = float(getattr(stage, "est_item_seconds", 0.5)) or 0.5
            min_items = max(2, round(spawn_s / est)) if spawn_s else 8
            stages[stage.key] = {"workers": proc_workers, "cpu_parallel": "process",
                                 "min_pool_items": min_items, "spawn_overhead_s": round(spawn_s, 1)}
        elif getattr(stage, "io_bound", False):
            stages[stage.key] = {"workers": disk_workers, "io_bound": True,
                                 "disk_write_mbps": round(float(disk_mbps), 1)}
        else:
            stages[stage.key] = {"workers": io_workers}

    profile = HardwareSettings(
        fingerprint=fingerprint,
        provider=provider,
        precision=_best_precision(gpus, provider, cfg),
        gpus=[asdict(g) for g in gpus],
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        tmp_dir=str(tmp_dir),
        io_workers=io_workers,
        stages=stages,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    _write(hw_path, profile)
    log.info("wrote %s | %d GPU(s) | provider=%s | tmp=%s", hw_path, len(gpus), provider, tmp_dir)
    return hw_path


# --------------------------------------------------------------------------- #
# Per-stage tuning (VRAM-fit dry-run sweep -- no weights required)
# --------------------------------------------------------------------------- #
def _optimize_stage(
    stage: Any,
    gpus: list[GpuInfo],
    provider: str,
    *,
    quick: bool,
    on_progress: ProgressCB | None,
    cfg: Any = None,
) -> dict:
    """Sweep ``(batch, workers_per_gpu)`` and keep the throughput-optimal combo that fits VRAM.

    Throughput is estimated as ``batch * workers_per_gpu`` (more concurrent work is faster
    until VRAM is exhausted); the peak-VRAM model is a linear per-image footprint plus a
    fixed model-resident cost. The best combo that stays inside the safety-fraction budget
    wins. This needs no model file, so setup stays fast and never reads user data.
    """
    # Size against the GPU with the MOST free VRAM, not gpus[0]: a run typically lands on a free
    # GPU, so sizing workers_per_gpu against a busy card (e.g. one holding another job) would wrongly
    # cap it at 1. free falls back to total when a probe returns 0.
    gpu = max(gpus, key=lambda g: (g.free_vram_mb or g.total_vram_mb or 0))
    budget_mb = _vram_budget(gpu, stage_cfg=None)
    resident_mb, per_img_mb = _footprint_model(stage)
    max_wpg = _max_workers_per_gpu(stage, cfg)

    batches = [1, 2, 4] if quick else [1, 2, 4, 8, 16, 32]
    best: dict | None = None
    for batch in batches:
        for wpg in range(1, max_wpg + 1):
            peak = resident_mb * wpg + per_img_mb * batch * wpg
            if peak > budget_mb:
                continue
            rate = float(batch * wpg)                       # relative throughput proxy
            if on_progress is not None:
                try:
                    on_progress(stage.key, batch, wpg, rate)
                except Exception:  # noqa: BLE001 - progress reporting is best effort
                    pass
            if best is None or rate > best["_rate"]:
                best = {"batch": batch, "workers_per_gpu": wpg, "_rate": rate, "peak_vram_mb": int(peak)}

    if best is None:                                        # not even batch=1/wpg=1 fits -> minimum plan
        best = {
            "batch": 1,
            "workers_per_gpu": 1,
            "_rate": 1.0,
            "peak_vram_mb": int(resident_mb + per_img_mb),
        }

    n_gpus = max(1, len(gpus))
    return {
        "batch": best["batch"],
        "workers_per_gpu": best["workers_per_gpu"],
        "queue_size": 2 * best["workers_per_gpu"] * n_gpus,
        "peak_vram_mb": best["peak_vram_mb"],           # total across workers_per_gpu workers
        # Per-worker cost is what the run-time allocator divides free VRAM by. Keep it separate
        # from peak_vram_mb (a total) so the two can never be confused for one another again.
        "est_vram_per_worker_mb": int(resident_mb + per_img_mb * best["batch"]),
        "vram_measured": False,                         # True only after a calibration run
    }


def _apply_vram_measurements(
    stages: dict[str, dict],
    cfg: Any,
    sweep_gpus: list[GpuInfo],
    *,
    calibrate: bool,
    on_progress: ProgressCB | None,
    profile_path: Path,
) -> None:
    """Fold measured per-worker VRAM into ``stages`` (mutates in place).

    Runs a calibration pass when ``calibrate``; otherwise reuses whatever the previous
    profile measured. Measured values also drive ``workers_per_gpu`` here so the profile
    itself reads truthfully, even though the run-time allocator recomputes the count from
    live free VRAM anyway.
    """
    measured: dict[str, dict] = {}
    if calibrate:
        cfg_path = getattr(cfg, "source_path", None)
        if not cfg_path:
            log.warning("cannot calibrate: config has no source path -- keeping estimates")
        else:
            from leafmachine3.setup.calibrate import calibrate_gpu_stages

            gpu = sweep_gpus[0].index if sweep_gpus else None
            measured = calibrate_gpu_stages(cfg_path, gpu_index=gpu, gpu_keys=set(stages),
                                            on_progress=on_progress)
    else:
        measured = _previous_measurements(profile_path)

    if not measured:
        return

    for key, plan in stages.items():
        rec = measured.get(key)
        if not rec or "workers_per_gpu" not in plan:            # GPU stages only
            continue
        per_worker = float(rec["vram_per_worker_mb"])
        plan["vram_per_worker_mb"] = round(per_worker, 1)
        plan["vram_measured"] = True
        plan["vram_measured_at"] = rec.get("measured_at") or time.strftime("%Y-%m-%dT%H:%M:%S")
        plan["vram_measured_by"] = rec.get("measured_by", "nvml_process")
        if sweep_gpus and per_worker > 0:
            # Mirror the run-time allocator exactly -- same headroom multiplier, same reserve --
            # so the profile does not advertise more workers than a run will actually start.
            budgeted = per_worker * _headroom_factor(cfg)
            gpu = max(sweep_gpus, key=lambda g: (g.free_vram_mb or g.total_vram_mb or 0))
            fit = int((_vram_budget(gpu, stage_cfg=None) - _reserve_mb(cfg)) // budgeted)
            plan["workers_per_gpu"] = max(1, min(fit, _max_workers_per_gpu(None, cfg)))
            plan["queue_size"] = 2 * plan["workers_per_gpu"] * max(1, len(sweep_gpus))
            plan["peak_vram_mb"] = int(budgeted * plan["workers_per_gpu"])


def _previous_measurements(profile_path: Path) -> dict[str, dict]:
    """Measured VRAM from the existing profile, so a re-profile does not lose it.

    Takes the path rather than reading a module constant: the profile is deployment-scoped, so
    "the existing profile" is only well defined relative to the one this run resolved.
    """
    if not profile_path.exists():
        return {}
    try:
        raw = yaml.safe_load(profile_path.read_text()) or {}
    except Exception:  # noqa: BLE001 - a broken profile just means no carry-forward
        return {}
    out: dict[str, dict] = {}
    for key, plan in (raw.get("stages") or {}).items():
        if isinstance(plan, dict) and plan.get("vram_measured") and plan.get("vram_per_worker_mb"):
            out[key] = {
                "vram_per_worker_mb": float(plan["vram_per_worker_mb"]),
                "measured_at": plan.get("vram_measured_at"),
                "measured_by": plan.get("vram_measured_by", "nvml_process"),
            }
    if out:
        log.info("carrying forward %d measured VRAM figure(s) from the previous profile", len(out))
    return out


def _heuristic_stage(stage: Any, gpus: list[GpuInfo]) -> dict:
    """Minimal safe plan for a GPU stage with no usable GPU (single-worker, batch 1)."""
    return {"batch": 1, "workers_per_gpu": 1, "queue_size": 2, "peak_vram_mb": 0,
            "est_vram_per_worker_mb": 0, "vram_measured": False}


def _footprint_model(stage: Any) -> tuple[float, float]:
    """Return ``(resident_mb, per_image_mb)`` estimates for a GPU stage.

    Segmenters are the memory-hungry stage (retina masks); classifiers are cheap;
    detectors sit in between. Values are conservative so the sweep never over-commits.
    """
    key = getattr(stage, "key", "")
    if "segment" in key or key == "leaf_segmenter":
        return 6000.0, 900.0
    if "classifier" in key:
        return 1500.0, 90.0
    return 3500.0, 350.0                                     # detectors (archival / plant)


def _vram_budget(gpu: GpuInfo, stage_cfg: Any) -> float:
    """Usable VRAM (MB) = free VRAM * safety_fraction (default 0.90), never below 512 MB."""
    free = float(gpu.free_vram_mb or gpu.total_vram_mb or 0)
    return max(512.0, free * 0.90)


def _vram_cfg(cfg: Any) -> dict:
    """``compute.vram`` as a plain dict, tolerating a Section, a mapping, or nothing at all."""
    try:
        compute = getattr(cfg, "compute", None) or {}
        vram = compute.get("vram", {}) if isinstance(compute, dict) else (getattr(compute, "vram", {}) or {})
        return dict(vram) if vram else {}
    except Exception:  # noqa: BLE001 - a malformed config must not break setup
        return {}


def _headroom_factor(cfg: Any) -> float:
    """OOM breathing room applied to a MEASURED per-worker figure (one sample, one image set)."""
    try:
        return float(_vram_cfg(cfg).get("headroom_factor", 1.15)) or 1.15
    except (TypeError, ValueError):
        return 1.15


def _reserve_mb(cfg: Any) -> float:
    """VRAM held back on every card, over and above the safety fraction."""
    try:
        return float(_vram_cfg(cfg).get("reserve_mb", 1024) or 0.0)
    except (TypeError, ValueError):
        return 1024.0


def _max_workers_per_gpu(stage: Any, cfg: Any = None) -> int:
    """Cap on concurrent workers per GPU, read from ``compute.vram.max_workers_per_gpu``.

    This is a safety rail, not a target: the real count is recomputed per GPU at run time
    from that card's live free VRAM (see ``DeviceManager._vram_plan``).
    """
    try:
        value = int(_vram_cfg(cfg).get("max_workers_per_gpu", 16))
        return value if value > 0 else 16
    except (TypeError, ValueError):
        return 16


# --------------------------------------------------------------------------- #
# Probes (all read-only, all degrade gracefully)
# --------------------------------------------------------------------------- #
def _discover_gpus() -> list[GpuInfo]:
    """Inventory NVIDIA GPUs via NVML, falling back to ``nvidia-smi``; ``[]`` if none."""
    gpus = _discover_gpus_nvml()
    if gpus:
        return gpus
    return _discover_gpus_smi()


def _discover_gpus_nvml() -> list[GpuInfo]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            driver = _nvml_str(pynvml.nvmlSystemGetDriverVersion())
            out: list[GpuInfo] = []
            for i in range(int(pynvml.nvmlDeviceGetCount())):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                out.append(
                    GpuInfo(
                        index=i,
                        name=_nvml_str(pynvml.nvmlDeviceGetName(handle)),
                        total_vram_mb=int(mem.total // (1024 * 1024)),
                        free_vram_mb=int(mem.free // (1024 * 1024)),
                        driver=driver,
                    )
                )
            return out
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - pynvml absent / no driver
        return []


def _discover_gpus_smi() -> list[GpuInfo]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        result = subprocess.run(
            [smi, "--query-gpu=index,name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except Exception:  # noqa: BLE001 - nvidia-smi failed
        return []
    gpus: list[GpuInfo] = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_vram_mb=int(float(parts[2])),
                    free_vram_mb=int(float(parts[3])),
                    driver=parts[4],
                )
            )
        except ValueError:
            continue
    return gpus


def _nvml_str(value: Any) -> str:
    """NVML returns ``bytes`` on some driver versions and ``str`` on others."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _discover_cpu_ram() -> tuple[int, int]:
    """Return ``(logical_cpu_cores, ram_gb)`` with pure-stdlib fallbacks."""
    cores = os.cpu_count() or 1
    ram_gb = 0
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        ram_gb = int(page_size * phys_pages / (1024 ** 3))
    except (ValueError, OSError, AttributeError):
        try:
            import psutil  # type: ignore

            ram_gb = int(psutil.virtual_memory().total / (1024 ** 3))
        except Exception:  # noqa: BLE001 - psutil optional
            ram_gb = 0
    return cores, ram_gb


def _probe_bound_provider(cfg: Any) -> str:
    """Return the execution provider that would actually bind (GPU-first ladder)."""
    try:
        from leafmachine3.inference.providers import build_providers

        providers = build_providers(cfg)
        if providers:
            first = providers[0]
            return first[0] if isinstance(first, (list, tuple)) else str(first)
    except Exception:  # noqa: BLE001 - ORT absent or fail-loud config; fall back below
        pass
    try:
        import onnxruntime as ort  # type: ignore

        available = set(ort.get_available_providers())
        for ep in ("CUDAExecutionProvider", "TensorrtExecutionProvider", "DmlExecutionProvider",
                   "ROCMExecutionProvider", "CoreMLExecutionProvider", "OpenVINOExecutionProvider"):
            if ep in available:
                return ep
    except Exception:  # noqa: BLE001 - onnxruntime not installed
        pass
    return "CPUExecutionProvider"


def _choose_tmp_dir(cfg: Any, *, min_free_gb: int = 50) -> Path:
    """Pick the fastest LOCAL scratch dir with enough headroom.

    Candidates: an explicit ``project.output.tmp_dir`` (non-``auto``), the platform
    temp dir, ``/scratch``, and the output dir. The first with ``>= min_free_gb`` free
    wins; otherwise the candidate with the most free space is used.
    """
    candidates: list[Path] = []
    tmp_cfg = str(_cfg_get(_cfg_get(cfg.project, "output"), "tmp_dir", "auto"))
    if tmp_cfg and tmp_cfg.lower() != "auto":
        candidates.append(Path(cfg.resolve_path(tmp_cfg)))
    for env in ("TMPDIR", "TEMP", "TMP"):
        val = os.environ.get(env)
        if val:
            candidates.append(Path(val))
    candidates += [Path("/scratch"), Path("/tmp"), Path(cfg.resolve_path(str(
        _cfg_get(_cfg_get(cfg.project, "output"), "dir", "runs"))))]

    best: tuple[float, Path] | None = None
    for cand in candidates:
        free_gb = _free_gb(cand)
        if free_gb is None:
            continue
        if free_gb >= min_free_gb:
            return cand
        if best is None or free_gb > best[0]:
            best = (free_gb, cand)
    return best[1] if best is not None else Path("/tmp")


def _free_gb(path: Path) -> Optional[float]:
    """Free space in GB for the filesystem holding ``path`` (nearest existing parent)."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        return None
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        return None
    return usage.free / (1024 ** 3)


def _best_precision(gpus: list[GpuInfo], provider: str, cfg: Any) -> str:
    """``fp16`` on a real CUDA/TensorRT GPU, else ``fp32`` (CPU has no fp16 win)."""
    requested = str(_cfg_get(cfg.compute, "precision", "fp16")).lower()
    gpu_provider = provider in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
    if requested == "fp32" or not gpus or not gpu_provider:
        return "fp32"
    return "fp16"


# --------------------------------------------------------------------------- #
# Fingerprint + stage selection
# --------------------------------------------------------------------------- #
def _fingerprint(cfg: Any) -> Fingerprint:
    """Build the machine/software fingerprint used to detect a stale profile."""
    gpus = _discover_gpus()
    cpu_cores, ram_gb = _discover_cpu_ram()
    driver = gpus[0].driver if gpus else ""
    ort_version, ort_providers = _ort_signature()
    return Fingerprint(
        os=f"{platform.system()}-{platform.release().split('-')[0]}",
        cpu=platform.processor() or platform.machine() or "unknown",
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpus=[[g.name, g.total_vram_mb] for g in gpus],
        driver=driver,
        ort_version=ort_version,
        ort_providers=ort_providers,
        model_hashes=_model_hashes(cfg),
        lm3_version=LM3_VERSION,
    )


def _ort_signature() -> tuple[str, list]:
    try:
        import onnxruntime as ort  # type: ignore

        return str(ort.__version__), list(ort.get_available_providers())
    except Exception:  # noqa: BLE001 - onnxruntime optional
        return "", []


def _model_hashes(cfg: Any) -> dict:
    """Map each model-backed stage to a cheap signature of its artifact(s).

    Uses the config's own stage-artifact resolution and a size/mtime signature (a full
    sha256 would slow setup for large exports and buys nothing over size+mtime here).
    """
    hashes: dict[str, str] = {}
    try:
        # ``Config._stage_artifacts`` is the same resolver used for settings hashing.
        resolver = getattr(cfg, "_stage_artifacts", None)
    except Exception:  # noqa: BLE001
        resolver = None
    from leafmachine3.core.config import CANONICAL_STAGE_KEYS

    for key in CANONICAL_STAGE_KEYS:
        if resolver is None:
            continue
        try:
            artifacts = resolver(key)
        except Exception:  # noqa: BLE001
            artifacts = []
        for artifact in artifacts:
            try:
                st = Path(artifact).stat()
            except OSError:
                continue
            hashes[key] = f"sig:{st.st_size}:{int(st.st_mtime)}"
    return hashes


def _chosen_gpu_indices(cfg: Any) -> list[int]:
    """The GPU ordinals the user pinned via ``compute.devices`` ([1] -> [1]); empty for 'auto'/'cpu'."""
    dev = _cfg_get(cfg.compute, "devices", "auto")
    if isinstance(dev, (list, tuple)):
        out = []
        for d in dev:
            try:
                out.append(int(d))
            except (TypeError, ValueError):
                pass
        return out
    return []


def _gpu_stages(cfg: Any) -> list[Any]:
    """Enabled GPU stages (``device_kind == 'cuda'``) as instantiated stage objects."""
    return [s for s in _enabled_stages(cfg) if getattr(s, "device_kind", "cpu") == "cuda"]


def _cpu_stages(cfg: Any) -> list[Any]:
    """Enabled CPU stages (``device_kind == 'cpu'``) as instantiated stage objects."""
    return [s for s in _enabled_stages(cfg) if getattr(s, "device_kind", "cpu") != "cuda"]


def _enabled_stages(cfg: Any) -> list[Any]:
    """Instantiate the enabled pipeline stages (lazy import to avoid a cycle)."""
    from leafmachine3.pipeline import STAGE_ORDER

    return [cls(cfg) for cls in STAGE_ORDER if cfg.is_enabled(cls.key)]


# --------------------------------------------------------------------------- #
# CPU process-pool scaling probe
#
# ``cpu_parallel="process"`` stages (ruler_cf, ect) run their GIL-bound cv2/numpy/matplotlib
# work on a spawn process pool. More workers is not always better: past the machine's real
# parallel knee (physical cores, memory bandwidth) throughput plateaus and then declines from
# spawn + contention. This probe measures that knee ONCE with a synthetic single-threaded FFT
# kernel (no user data, dependency-light) so the profile can size ``workers`` sensibly instead of
# blindly using ``cpu_cores - 2`` (which over-spawns ~2x on a hyperthreaded box).
# --------------------------------------------------------------------------- #
def _cpu_probe_task(seed: int) -> float:
    """~single-threaded CPU work (pocketfft is single-threaded) approximating a per-item stage cost."""
    import numpy as np

    rng = np.random.RandomState(seed % 101)
    a = rng.rand(320, 320)
    for _ in range(8):
        a = np.abs(np.fft.fft2(a)).real
        a /= (a.max() or 1.0)
    return float(a.sum())


def _benchmark_cpu_workers(cores: int, on_progress: Optional[Any] = None) -> int:
    """Measure the spawn process-pool throughput knee; return the smallest worker count within 95%
    of peak throughput (the efficient knee), or a conservative fallback if the probe cannot run."""
    import time as _time
    from concurrent.futures import ProcessPoolExecutor

    fallback = max(1, cores - 2)
    try:
        ctx = mp.get_context("spawn")
        cands = sorted({w for w in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64) if 1 <= w <= cores} | {cores})
        rates: dict[int, float] = {}
        for w in cands:
            try:
                with ProcessPoolExecutor(max_workers=w, mp_context=ctx) as ex:
                    list(ex.map(_cpu_probe_task, range(w)))            # warm: spawn every worker
                    k = max(4 * w, 48)
                    t0 = _time.perf_counter()
                    list(ex.map(_cpu_probe_task, range(k)))            # timed: steady state
                    dt = _time.perf_counter() - t0
                rates[w] = (k / dt) if dt > 0 else 0.0
            except Exception:
                continue
            if on_progress:
                try:
                    on_progress("cpu_scaling", w, 1, rates[w])
                except Exception:
                    pass
        if not rates:
            return fallback
        peak = max(rates.values())
        knee = min(w for w in sorted(rates) if rates[w] >= 0.95 * peak)
        log.info("CPU process-pool knee: %d workers (peak %.0f tasks/s at ~%d cores)",
                 knee, peak, cores)
        return max(1, knee)
    except Exception as exc:  # noqa: BLE001 - never let the probe abort setup
        log.warning("CPU worker benchmark failed (%s) -- using cpu_cores-2=%d", exc, fallback)
        return fallback


def _reports_fs_dir(cfg: Any, tmp_dir: Path) -> Path:
    """The filesystem the run's reports/ will live on (``project.output.dir``), for the disk probe.
    Falls back to the parent dir, then the tuned tmp_dir, so the probe never fails on a missing dir."""
    try:
        out = Path(cfg.resolve_path(str(_cfg_get(_cfg_get(cfg.project, "output"), "dir", "runs"))))
        for cand in (out, out.parent):
            if cand.exists():
                return cand
    except Exception:  # noqa: BLE001
        pass
    return tmp_dir


def _spawn_probe_task(_seed: int) -> int:
    """Warm the heavy imports a real process worker pays for (cv2/numpy for ruler_cf; +matplotlib/h5py
    for ect), so the measured spawn cost reflects a realistic worker startup, not a bare interpreter."""
    import numpy  # noqa: F401
    import cv2  # noqa: F401
    try:
        from matplotlib.figure import Figure  # noqa: F401
        import h5py  # noqa: F401
    except Exception:  # noqa: BLE001 - a stage lacking these just has a cheaper startup
        pass
    return 1


def _benchmark_spawn_overhead(workers: int) -> float:
    """Wall seconds to stand up a `workers`-wide spawn pool and warm-import cv2/numpy in each -- the
    fixed cost a cpu_parallel='process' stage pays before ANY real work. 0.0 if it cannot run."""
    import time as _time
    from concurrent.futures import ProcessPoolExecutor
    try:
        ctx = mp.get_context("spawn")
        t0 = _time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            list(ex.map(_spawn_probe_task, range(workers)))
        dt = _time.perf_counter() - t0
        log.info("process-pool spawn overhead: %.1fs for %d workers", dt, workers)
        return dt
    except Exception as exc:  # noqa: BLE001
        log.warning("spawn-overhead benchmark failed (%s)", exc)
        return 0.0


def _disk_write_task(args) -> int:
    """Write one blob with an fsync (cv2.imwrite/PIL release the GIL similarly at write time)."""
    path, blob = args
    with open(path, "wb") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())
    return len(blob)


def _benchmark_disk_writers(target_dir: Path, on_progress: Optional[Any] = None,
                            blob_mb: float = 2.0) -> tuple[int, float]:
    """Measure the reports-filesystem parallel-write knee (threads, since imwrite releases the GIL).
    Returns (knee_writers, peak_MB_per_s); a conservative (8, 0.0) if it cannot run."""
    import shutil
    import tempfile
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    fallback = (8, 0.0)
    probe_dir = None
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        probe_dir = Path(tempfile.mkdtemp(prefix="lm3_diskprobe_", dir=str(target_dir)))
        blob = os.urandom(int(blob_mb * 1_000_000))
        ctx_spawn = None  # threads, not processes
        rates: dict[int, float] = {}
        for w in (1, 2, 4, 8, 16):
            k = w * 6
            args = [(probe_dir / f"w{w}_{j}.bin", blob) for j in range(k)]
            t0 = _time.perf_counter()
            with ThreadPoolExecutor(max_workers=w) as ex:
                list(ex.map(_disk_write_task, args))
            dt = _time.perf_counter() - t0
            rates[w] = (k * blob_mb / dt) if dt > 0 else 0.0
            for p, _ in args:
                try:
                    os.unlink(p)
                except OSError:
                    pass
            if on_progress:
                try:
                    on_progress("disk_write", w, 1, rates[w])
                except Exception:
                    pass
        if not rates:
            return fallback
        peak = max(rates.values())
        knee = min(w for w in sorted(rates) if rates[w] >= 0.95 * peak)
        log.info("disk-write knee: %d writers (peak %.0f MB/s at %s)", knee, peak, target_dir)
        return max(1, knee), peak
    except Exception as exc:  # noqa: BLE001
        log.warning("disk-write benchmark failed (%s) -- using %d writers", exc, fallback[0])
        return fallback
    finally:
        if probe_dir is not None:
            import shutil as _sh
            _sh.rmtree(probe_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# YAML load / write
# --------------------------------------------------------------------------- #
def _load(path: Path) -> HardwareSettings:
    """Read a profile YAML back into a :class:`HardwareSettings`."""
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    fp_data = dict(data.get("fingerprint", {}) or {})
    fingerprint = Fingerprint(
        os=str(fp_data.get("os", "")),
        cpu=str(fp_data.get("cpu", "")),
        cpu_cores=int(fp_data.get("cpu_cores", 0) or 0),
        ram_gb=int(fp_data.get("ram_gb", 0) or 0),
        gpus=[list(g) for g in (fp_data.get("gpus", []) or [])],
        driver=str(fp_data.get("driver", "")),
        ort_version=str(fp_data.get("ort_version", "")),
        ort_providers=list(fp_data.get("ort_providers", []) or []),
        model_hashes=dict(fp_data.get("model_hashes", {}) or {}),
        lm3_version=str(fp_data.get("lm3_version", LM3_VERSION)),
    )
    return HardwareSettings(
        fingerprint=fingerprint,
        provider=str(data.get("provider", "CPUExecutionProvider")),
        precision=str(data.get("precision", "fp32")),
        gpus=list(data.get("gpus", []) or []),
        cpu_cores=int(data.get("cpu_cores", 0) or 0),
        ram_gb=int(data.get("ram_gb", 0) or 0),
        tmp_dir=str(data.get("tmp_dir", "")),
        io_workers=int(data.get("io_workers", 1) or 1),
        stages=dict(data.get("stages", {}) or {}),
        generated_at=str(data.get("generated_at", "")),
        lm3_version=str(data.get("lm3_version", LM3_VERSION)),
    )


def _write(path: Path, profile: HardwareSettings) -> None:
    """Serialise ``profile`` to ``path`` as plain YAML (dataclasses -> dicts).

    Creates the deployment config directory first. It used to be safe not to: the target was a
    file in an existing CWD. The canonical target is under ``<user-config>/lm3/<deployment>/``,
    which on a first run has never existed, and ``Path.replace`` onto a missing parent is an
    ``OSError``, not a mkdir.
    """
    payload = asdict(profile)
    _mkdir_private(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(
            "# LM3 hardware profile -- MACHINE-DERIVED runtime settings, written by LM3_Setup\n"
            "# (python -m leafmachine3.setup). Reused by every module so runs never re-probe.\n"
            "# Do NOT hand-edit unless you know why; rerun LM3_Setup after a hardware/driver/model change.\n"
        )
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)
    tmp.replace(path)


def _mkdir_private(directory: Path) -> None:
    """``mkdir -p`` with user-only permissions on POSIX, matching how the resolver creates it.

    Best effort on the mode: an existing directory keeps whatever permissions it has, and a
    filesystem that cannot express them (a Windows share, a FAT scratch disk) is not a reason to
    refuse to write a profile.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        try:
            os.chmod(directory, 0o700)
        except OSError:  # noqa: PERF203 - permissions are a nicety here, not a precondition
            log.debug("could not tighten permissions on %s", directory, exc_info=True)


def _cfg_get(section: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a Section/dict/attr object tolerantly."""
    if section is None:
        return default
    getter = getattr(section, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:  # noqa: BLE001
            pass
    return getattr(section, key, default)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console entry point: ``python -m leafmachine3.setup`` / ``lm3-setup``.

    Both of its paths come from the canonical resolver (section 3.1 puts the standalone
    hardware-setup CLI explicitly in scope), so this command and a GUI- or CLI-started run agree
    on which settings file and which profile they are talking about no matter where each was
    launched from.
    """
    parser = argparse.ArgumentParser(
        prog="lm3-setup", description="Profile this machine and write its LM3 hardware profile."
    )
    # No default: an OMITTED --config runs the full precedence chain (LM3_SETTINGS, the deployment
    # workspace pointer, the deployment settings file, this checkout), while a --config the user
    # actually typed and that does not exist is a hard error rather than a silent fall-through to
    # some other configuration. A relative default string could not tell those two cases apart.
    parser.add_argument("--config", default=None,
                        help="path to LM3_settings.yaml (default: the canonical resolved settings)")
    parser.add_argument("--optimize", action="store_true", help="run the per-stage VRAM-fit sweep")
    parser.add_argument("--quick", action="store_true", help="shorter sweep (batches 1/2/4)")
    parser.add_argument("--force", action="store_true", help="rewrite even if the profile is current")
    parser.add_argument("--tmp", default=None, help="override the tmp scratch dir")
    parser.add_argument("--calibrate", action="store_true",
                        help="MEASURE per-worker VRAM by running example images with 1 worker "
                             "(a few minutes) instead of estimating it")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from leafmachine3.core.config import Config

    try:
        cfg_path = paths.settings_path(args.config)
    except paths.PathsError as exc:
        # A stated intent that cannot be satisfied. Report it as a usage error instead of a
        # traceback, and never guess at a different settings file.
        log.error("%s", exc)
        return 2
    log.info("settings: %s", cfg_path)

    cfg = Config.load(str(cfg_path))
    log.info("hardware profile: %s", hardware_profile_path(cfg))
    run_setup(
        cfg,
        optimize=args.optimize or not args.quick,
        quick=args.quick,
        force=args.force,
        calibrate=args.calibrate,
        tmp_override=args.tmp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
