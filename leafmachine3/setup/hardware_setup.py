"""leafmachine3.setup.hardware_setup -- LM3_Setup: the one-time (rerunnable) profiler.

Runs ONCE per install (or on demand) to measure THIS specific machine and write
``hardware_settings.yaml`` -- the tuned, machine-derived layer every module reads for
worker counts, batch sizes, queue sizes, tmp location, GPU availability + VRAM,
precision, and the bound execution provider.

The profiler is deliberately conservative and dependency-light: every probe degrades
gracefully so that a CPU-only host (or a ``compute.mock`` run) always produces a valid
profile without a GPU, ``pynvml``, ``onnxruntime`` or ``psutil`` installed. The per-stage
"dry-run sweep" is a VRAM-fit search over candidate ``(batch, workers_per_gpu)`` combos
that keeps the throughput-optimal plan fitting the GPU's VRAM budget; it needs no model
weights, so it never touches user data -- it only measures and writes the YAML.

CLI:  ``python -m leafmachine3.setup [--optimize] [--quick] [--force] [--tmp DIR]``
GUI:  the Electron "Hardware Setup" panel button -> ``POST /v1/setup``.
"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import yaml

log = logging.getLogger("leafmachine3.setup")

HW_PATH = Path("hardware_settings.yaml")     # top-level, beside LM3_settings.yaml
LM3_VERSION = "3.0.0"

ProgressCB = Callable[..., None]


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
    """The complete, serialisable machine profile written to ``hardware_settings.yaml``."""

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

    If ``hardware_settings.yaml`` is MISSING, the full LM3_Setup runs once (the only
    time it self-triggers) so the first run is tuned. If it EXISTS it is never re-run
    automatically: a drifted fingerprint only LOGS a suggestion to rerun. The profile
    is then bound so every ``auto`` in the config resolves from it.
    """
    if not HW_PATH.exists():
        log.info(
            "no hardware_settings.yaml found -- running LM3_Setup once to tune this machine "
            "(cached for every future run)"
        )
        run_setup(cfg, optimize=True)
    else:
        try:
            if _load(HW_PATH).fingerprint != _fingerprint(cfg):
                log.warning(
                    "hardware / driver / models changed since the last LM3_Setup -- this run uses the "
                    "EXISTING profile; rerun `python -m leafmachine3.setup` to re-tune when convenient"
                )
        except Exception as exc:  # noqa: BLE001 - a malformed profile must not abort the run
            log.warning("could not read %s (%s) -- rebuilding profile", HW_PATH, exc)
            run_setup(cfg, optimize=True, force=True)

    try:
        cfg.bind_hardware(_load(HW_PATH))
    except Exception as exc:  # noqa: BLE001 - never let profile binding crash a run
        log.warning("could not bind %s (%s) -- proceeding without a tuned profile", HW_PATH, exc)
    return HW_PATH


def run_setup(
    cfg: Any,
    *,
    optimize: bool = True,
    quick: bool = False,
    force: bool = False,
    tmp_override: str | None = None,
    on_progress: ProgressCB | None = None,
) -> Path:
    """Profile the machine and (re)write ``hardware_settings.yaml``.

    Idempotent: a valid profile whose fingerprint already matches this machine is kept
    unless ``force``. Returns the path written (or reused).
    """
    fingerprint = _fingerprint(cfg)
    if HW_PATH.exists() and not force:
        try:
            if _load(HW_PATH).fingerprint == fingerprint:
                log.info("hardware_settings.yaml is current -- nothing to do (use --force to redo)")
                return HW_PATH
        except Exception:  # noqa: BLE001 - fall through and rewrite a broken file
            pass

    gpus = _discover_gpus()
    cpu_cores, ram_gb = _discover_cpu_ram()
    provider = _probe_bound_provider(cfg)
    tmp_dir = Path(tmp_override) if tmp_override else _choose_tmp_dir(cfg, min_free_gb=50)
    io_workers = max(1, cpu_cores - 2)

    stages: dict[str, dict] = {}
    for stage in _gpu_stages(cfg):
        stages[stage.key] = (
            _optimize_stage(stage, gpus, provider, quick=quick, on_progress=on_progress)
            if optimize and gpus
            else _heuristic_stage(stage, gpus)
        )
    for stage in _cpu_stages(cfg):
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
    _write(HW_PATH, profile)
    log.info("wrote %s | %d GPU(s) | provider=%s | tmp=%s", HW_PATH, len(gpus), provider, tmp_dir)
    return HW_PATH


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
) -> dict:
    """Sweep ``(batch, workers_per_gpu)`` and keep the throughput-optimal combo that fits VRAM.

    Throughput is estimated as ``batch * workers_per_gpu`` (more concurrent work is faster
    until VRAM is exhausted); the peak-VRAM model is a linear per-image footprint plus a
    fixed model-resident cost. The best combo that stays inside the safety-fraction budget
    wins. This needs no model file, so setup stays fast and never reads user data.
    """
    gpu = gpus[0]
    budget_mb = _vram_budget(gpu, stage_cfg=None)
    resident_mb, per_img_mb = _footprint_model(stage)
    max_wpg = _max_workers_per_gpu(stage)

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
        "peak_vram_mb": best["peak_vram_mb"],
    }


def _heuristic_stage(stage: Any, gpus: list[GpuInfo]) -> dict:
    """Minimal safe plan for a GPU stage with no usable GPU (single-worker, batch 1)."""
    return {"batch": 1, "workers_per_gpu": 1, "queue_size": 2, "peak_vram_mb": 0}


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


def _max_workers_per_gpu(stage: Any) -> int:
    """Cap concurrent workers per GPU (mirrors ``compute.vram.max_workers_per_gpu`` default)."""
    return 6


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
# YAML load / write
# --------------------------------------------------------------------------- #
def _load(path: Path) -> HardwareSettings:
    """Read ``hardware_settings.yaml`` back into a :class:`HardwareSettings`."""
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
    """Serialise ``profile`` to ``path`` as plain YAML (dataclasses -> dicts)."""
    payload = asdict(profile)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(
            "# hardware_settings.yaml -- MACHINE-DERIVED runtime settings, written by LM3_Setup\n"
            "# (python -m leafmachine3.setup). Reused by every module so runs never re-probe.\n"
            "# Do NOT hand-edit unless you know why; rerun LM3_Setup after a hardware/driver/model change.\n"
        )
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)
    tmp.replace(path)


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
    """Console entry point: ``python -m leafmachine3.setup`` / ``lm3-setup``."""
    parser = argparse.ArgumentParser(
        prog="lm3-setup", description="Profile this machine and write hardware_settings.yaml."
    )
    parser.add_argument("--config", default="LM3_settings.yaml", help="path to LM3_settings.yaml")
    parser.add_argument("--optimize", action="store_true", help="run the per-stage VRAM-fit sweep")
    parser.add_argument("--quick", action="store_true", help="shorter sweep (batches 1/2/4)")
    parser.add_argument("--force", action="store_true", help="rewrite even if the profile is current")
    parser.add_argument("--tmp", default=None, help="override the tmp scratch dir")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from leafmachine3.core.config import Config

    cfg = Config.load(args.config)
    run_setup(
        cfg,
        optimize=args.optimize or not args.quick,
        quick=args.quick,
        force=args.force,
        tmp_override=args.tmp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
