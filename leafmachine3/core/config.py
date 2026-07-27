"""leafmachine3.core.config -- the single, merged configuration surface.

``Config`` is the in-memory result of layering three sources in ascending
precedence::

    built-in defaults  <  LM3_settings.yaml  <  CLI / programmatic overrides

Every nested node is a :class:`Section` -- a ``dict`` subclass that additionally
supports attribute access (``cfg.project.output.dir``) so callers can read config
the same way regardless of how deeply nested a key lives, while still using the
tolerant ``.get(key, default)`` form for optional keys.

Machine-derived tuning (worker counts, batch sizes, tmp location, bound execution
provider) lives in a separate ``hardware_settings.yaml`` that ``LM3_Setup`` writes;
it is layered on at runtime via :meth:`Config.bind_hardware` and read back through
:attr:`Config.hardware`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import yaml

log = logging.getLogger("leafmachine3.config")

# The canonical pipeline stages, in execution order. These are the only keys that may
# appear in ``project.run_mode.restart`` and the keys the ProjectDB seeds
# ``project_status`` from. Kept in lockstep with ``leafmachine3.pipeline.STAGE_ORDER``.
CANONICAL_STAGE_KEYS: tuple[str, ...] = (
    "archival_detector",
    "plant_detector",
    "phenology_detector",
    "ruler_classifier",
    "ruler_cf",
    "leaf_segmenter",
    "morphology",
    "landmark_detector",
    "landmark_measurements",
    "leaf_orientation",
    "petiole_width",
    "metric_grounding",
    "reporter",
)

# Stages that require an exported single-file model artifact when enabled.
_MODEL_PATH_STAGES: tuple[str, ...] = ("archival_detector", "plant_detector", "leaf_segmenter",
                                       "landmark_detector")

_MISSING = object()


# --------------------------------------------------------------------------- #
# Section: a dict that also answers to attribute access.
# --------------------------------------------------------------------------- #
class Section(dict):
    """A ``dict`` whose string keys are also reachable as attributes.

    Nested mappings (and mappings nested inside lists) are converted to
    ``Section`` on construction so an attribute chain like
    ``cfg.compute.vram.safety_fraction`` works all the way down. Missing keys
    raise :class:`AttributeError` (not ``KeyError``) so ``getattr(node, key,
    default)`` behaves as callers expect, while ``.get(key, default)`` remains
    available for optional lookups.
    """

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if data:
            for key, value in data.items():
                self[str(key)] = _wrap(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(name) from exc


def _wrap(value: Any) -> Any:
    """Recursively convert mappings to :class:`Section` (and inside lists/tuples)."""
    if isinstance(value, Section):
        return value
    if isinstance(value, Mapping):
        return Section(value)
    if isinstance(value, (list, tuple)):
        return [_wrap(v) for v in value]
    return value


def _plain(value: Any) -> Any:
    """Inverse of :func:`_wrap`: a plain ``dict``/``list`` tree for hashing/JSON."""
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` deep-merged with ``override`` (override wins).

    Nested mappings merge recursively; every other type (including lists) is
    replaced wholesale. ``base`` is not mutated.
    """
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = out.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            out[key] = _deep_merge(dict(existing), value)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# Hardware profile adapter (bound from hardware_settings.yaml by LM3_Setup).
# --------------------------------------------------------------------------- #
class HardwareProfile:
    """Uniform read-only view over a bound hardware profile.

    ``LM3_Setup`` may hand us either the parsed ``hardware_settings.yaml`` mapping
    or a dataclass instance; this adapter exposes both as attribute access
    (``cfg.hardware.tmp_dir``) plus a :meth:`stage` accessor returning the tuned
    per-stage plan (or ``None`` when the profile has no entry for that stage).
    """

    def __init__(self, source: Any) -> None:
        object.__setattr__(self, "_src", source)

    def _get(self, name: str, default: Any = None) -> Any:
        src = object.__getattribute__(self, "_src")
        if isinstance(src, Mapping):
            return src.get(name, default)
        return getattr(src, name, default)

    def stage(self, key: str) -> Section | None:
        """Return the tuned plan Section for ``key`` (batch/workers/...), or ``None``."""
        stages = self._get("stages", {}) or {}
        if isinstance(stages, Mapping):
            plan = stages.get(key)
        else:
            plan = getattr(stages, key, None)
        if plan is None:
            return None
        return plan if isinstance(plan, Section) else _wrap(plan)

    def __getattr__(self, name: str) -> Any:
        value = self._get(name, _MISSING)
        if value is _MISSING:
            raise AttributeError(name)
        return value


# --------------------------------------------------------------------------- #
# Built-in defaults -- the bottom layer of the merge.
# --------------------------------------------------------------------------- #
def builtin_defaults() -> dict[str, Any]:
    """Return a fresh copy of the built-in default configuration tree."""
    return {
        "version": 3,
        "project": {
            "run_name": "run",
            "input": {
                "dirs": [],
                "recursive": True,
                "image_extensions": [".jpg", ".jpeg", ".png", ".tif", ".tiff"],
            },
            "output": {"dir": "runs", "tmp_dir": "auto", "keep_tmp": False},
            "run_mode": {"overwrite": False, "restart": [], "fail_fast": False},
            "logging": {"level": "INFO", "to_file": True, "to_console": True},
        },
        "compute": {
            "devices": "auto",
            "tensorrt": False,
            "mock": False,
            "io_workers": "auto",
            "force_reprobe": False,
            "vram": {
                "safety_fraction": 0.90,
                "reserve_mb": 1024,
                "per_worker_mb": "auto",
                "max_workers_per_gpu": 6,
            },
            "workers": {"default": "auto"},
            "onnxruntime": {
                "providers": "auto",
                "ld_library_path": "auto",
                "allow_cpu_fallback": False,
            },
            "precision": "fp16",
        },
        "ingest": {"max_working_dim": 3200, "jpg_quality": 100},
        "modules": {
            "archival_detector": {"enabled": True},
            "plant_detector": {"enabled": True},
            "phenology_detector": {"enabled": True},
            "ruler_classifier": {"enabled": True},
            "ruler_cf": {"enabled": False},
            "leaf_segmenter": {"enabled": True},
            "morphology": {"enabled": True},
            "landmark_detector": {"enabled": True},
            "landmark_measurements": {"enabled": True},
            "leaf_orientation": {"enabled": True},
            "petiole_width": {"enabled": True},
            "metric_grounding": {"enabled": True},
            "reporter": {"enabled": True},
        },
        "report": {},
    }


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class Config:
    """The merged, validated LeafMachine3 configuration."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._raw: Section = Section(data)
        self._hardware: HardwareProfile | None = None

    # -- construction ------------------------------------------------------- #
    @classmethod
    def load(cls, path: str | os.PathLike[str], overrides: Mapping[str, Any] | None = None) -> "Config":
        """Load ``path`` and layer it over the built-in defaults, then ``overrides``.

        Precedence (ascending): built-in defaults < YAML file < ``overrides``.
        """
        merged = builtin_defaults()

        file_path = Path(path).expanduser()
        if not file_path.is_file():
            raise FileNotFoundError(f"config file not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"config file did not parse to a mapping: {file_path}")
        merged = _deep_merge(merged, loaded)

        if overrides:
            merged = _deep_merge(merged, overrides)

        return cls(merged)

    # -- top-level nodes ---------------------------------------------------- #
    @property
    def project(self) -> Section:
        return self._raw.get("project", Section())

    @property
    def compute(self) -> Section:
        return self._raw.get("compute", Section())

    @property
    def ingest(self) -> Section:
        return self._raw.get("ingest", Section())

    @property
    def modules(self) -> Section:
        return self._raw.get("modules", Section())

    @property
    def report(self) -> Section:
        return self._raw.get("report", Section())

    @property
    def naming(self) -> Section:
        """Crop/mask filename settings (``bbox_prefix``/``seg_prefix``/``friendly_names``)."""
        return self._raw.get("naming", Section())

    # -- stage access ------------------------------------------------------- #
    def stage(self, key: str) -> Section:
        """Return the ``modules.<key>`` Section (empty Section if absent)."""
        node = self.modules.get(key)
        return node if isinstance(node, Section) else Section(node or {})

    def is_enabled(self, key: str) -> bool:
        """True iff ``modules.<key>.enabled`` is truthy."""
        node = self.modules.get(key)
        return bool(node and node.get("enabled", False))

    # -- run-mode ----------------------------------------------------------- #
    @property
    def restart(self) -> None | str | list[str]:
        """Normalized restart directive: ``None`` (resume), ``'all'``, or a list of keys."""
        raw = self.project.get("run_mode", Section()).get("restart", [])
        if raw is None:
            return None
        if isinstance(raw, str):
            return "all" if raw.strip().lower() == "all" else [raw]
        if isinstance(raw, (list, tuple)):
            keys = [str(k) for k in raw]
            if not keys:
                return None
            if any(k.strip().lower() == "all" for k in keys):
                return "all"
            return keys
        # any other scalar -> treat as a single key
        return [str(raw)]

    @property
    def overwrite(self) -> bool:
        return bool(self.project.get("run_mode", Section()).get("overwrite", False))

    @property
    def wipe_working_set(self) -> bool:
        """True when the working set must be rebuilt (overwrite, or a full restart)."""
        return self.overwrite or self.restart == "all"

    @property
    def fail_fast(self) -> bool:
        return bool(self.project.get("run_mode", Section()).get("fail_fast", False))

    @property
    def force_reprobe(self) -> bool:
        """When True the executor live-probes VRAM even if a tuned profile exists."""
        return bool(self.compute.get("force_reprobe", False))

    # -- path / worker helpers --------------------------------------------- #
    def resolve_path(self, p: str | os.PathLike[str]) -> str:
        """Absolutize ``p``: expand ``~``; absolute paths pass through; else join CWD."""
        path = Path(str(p)).expanduser()
        if path.is_absolute():
            return str(path)
        return str(Path.cwd() / path)

    def io_workers(self) -> int:
        """Resolve the ingest / CPU-stage worker count (``auto`` -> tuned or cpu_count-2)."""
        raw = self.compute.get("io_workers", "auto")
        if isinstance(raw, bool):  # guard: bool is an int subclass
            raw = "auto"
        if isinstance(raw, int):
            return max(1, raw)
        hw = self._hardware
        if hw is not None:
            tuned = getattr(hw, "io_workers", None)
            if isinstance(tuned, int) and not isinstance(tuned, bool):
                return max(1, tuned)
        return max(1, (os.cpu_count() or 2) - 2)

    def resolve_gpus(self) -> list[int]:
        """Resolve ``compute.devices`` to a list of physical NVIDIA GPU ordinals.

        ``'cpu'`` -> ``[]``; an explicit list -> those ordinals; ``'auto'`` ->
        every visible NVIDIA GPU (empty list if none / probing fails).
        """
        devices = self.compute.get("devices", "auto")
        if isinstance(devices, str):
            if devices.strip().lower() == "cpu":
                return []
            if devices.strip().lower() == "auto":
                return _discover_nvidia_ordinals()
            return []
        if isinstance(devices, (list, tuple)):
            out: list[int] = []
            for d in devices:
                try:
                    out.append(int(d))
                except (TypeError, ValueError):
                    continue
            return out
        return []

    def probe_vram_mb(self, stage: Any) -> int:
        """Best-effort per-worker VRAM budget (MB) for ``stage``.

        Prefers an explicit ``compute.vram.per_worker_mb``; otherwise falls back
        to the tuned profile's measured ``peak_vram_mb`` for the stage, then a
        conservative default. Used by the executor only when no tuned plan is
        available and a live probe is not performed.
        """
        per_worker = self.compute.get("vram", Section()).get("per_worker_mb", "auto")
        if isinstance(per_worker, int) and not isinstance(per_worker, bool):
            return max(1, per_worker)
        hw = self._hardware
        key = getattr(stage, "key", None)
        if hw is not None and key is not None:
            plan = hw.stage(key)
            if plan is not None:
                peak = plan.get("peak_vram_mb")
                if isinstance(peak, int) and not isinstance(peak, bool):
                    return max(1, peak)
        return 4096

    # -- hardware profile --------------------------------------------------- #
    @property
    def hardware(self) -> HardwareProfile | None:
        return self._hardware

    def bind_hardware(self, profile: Any) -> None:
        """Attach the machine-derived tuning profile (from ``hardware_settings.yaml``)."""
        if profile is None:
            self._hardware = None
        elif isinstance(profile, HardwareProfile):
            self._hardware = profile
        elif isinstance(profile, Mapping):
            self._hardware = HardwareProfile(Section(profile))
        else:
            self._hardware = HardwareProfile(profile)

    # -- settings hashing (resume invalidation) ----------------------------- #
    def stage_settings_hash(self, key: str) -> str:
        """Stable hash of a stage's resolved settings plus its model file signature.

        Changing any knob under ``modules.<key>`` -- or replacing the exported
        model file (its size/mtime) -- yields a different hash, letting the DB
        invalidate that stage's prior results on resume.
        """
        hasher = hashlib.sha256()
        block = _plain(self.stage(key))
        hasher.update(json.dumps(block, sort_keys=True, default=str).encode("utf-8"))
        for artifact in self._stage_artifacts(key):
            try:
                st = Path(artifact).stat()
            except OSError:
                continue
            hasher.update(f"|{artifact}:{st.st_size}:{int(st.st_mtime)}".encode("utf-8"))
        return hasher.hexdigest()

    def _stage_artifacts(self, key: str) -> list[str]:
        """Resolved model file(s) / dir referenced by a stage (for hashing)."""
        blk = self.stage(key)
        out: list[str] = []
        model = blk.get("model")
        if isinstance(model, Mapping):
            path = model.get("path")
            if path:
                out.append(self.resolve_path(path))
        models_dir = blk.get("models_dir")
        if models_dir:
            out.append(self.resolve_path(models_dir))
        return out

    # -- validation --------------------------------------------------------- #
    def validate(self) -> None:
        """Raise :class:`ValueError` on an incoherent configuration.

        Checks structural coherence only -- NOT artifact existence on disk, which
        is the job of :func:`leafmachine3.core.validate.validate_ml_artifacts`.
        """
        errors: list[str] = []
        mock = bool(self.compute.get("mock", False))

        precision = str(self.compute.get("precision", "fp16")).lower()
        if precision not in {"fp16", "fp32"}:
            errors.append(f"compute.precision must be 'fp16' or 'fp32', got {precision!r}")

        devices = self.compute.get("devices", "auto")
        if isinstance(devices, str):
            if devices.strip().lower() not in {"auto", "cpu"}:
                errors.append(f"compute.devices string must be 'auto' or 'cpu', got {devices!r}")
        elif isinstance(devices, (list, tuple)):
            for d in devices:
                if isinstance(d, bool) or not isinstance(d, int):
                    errors.append(f"compute.devices list entries must be int ordinals, got {d!r}")
                    break
        else:
            errors.append(f"compute.devices must be 'auto', 'cpu', or a list, got {type(devices).__name__}")

        if not mock:
            for key in _MODEL_PATH_STAGES:
                if not self.is_enabled(key):
                    continue
                model = self.stage(key).get("model")
                path = model.get("path") if isinstance(model, Mapping) else None
                if not path:
                    errors.append(f"modules.{key} is enabled but has no model.path (set one or compute.mock: true)")
            if self.is_enabled("ruler_classifier"):
                if not self.stage("ruler_classifier").get("models_dir"):
                    errors.append("modules.ruler_classifier is enabled but has no models_dir")

        restart = self.restart
        if isinstance(restart, list):
            bad = [k for k in restart if k not in CANONICAL_STAGE_KEYS]
            if bad:
                errors.append(
                    f"project.run_mode.restart has unknown stage key(s): {bad}; "
                    f"valid keys are {list(CANONICAL_STAGE_KEYS)} or 'all'"
                )

        if not self.project.get("input", Section()).get("dirs"):
            errors.append("project.input.dirs is empty -- no input images to process")

        if errors:
            raise ValueError("invalid LM3 configuration:\n  - " + "\n  - ".join(errors))


# --------------------------------------------------------------------------- #
# NVIDIA discovery (pynvml -> nvidia-smi -L -> none)
# --------------------------------------------------------------------------- #
def _discover_nvidia_ordinals() -> list[int]:
    """Return visible NVIDIA GPU ordinals, or ``[]`` if none / probing fails."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            count = int(pynvml.nvmlDeviceGetCount())
        finally:
            pynvml.nvmlShutdown()
        if count > 0:
            return list(range(count))
    except Exception:  # noqa: BLE001 - pynvml absent / no driver / init failure
        pass

    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "-L"], capture_output=True, text=True, timeout=10, check=False
            ).stdout
            ordinals = [i for i, line in enumerate(out.splitlines()) if line.startswith("GPU ")]
            if ordinals:
                return ordinals
        except Exception:  # noqa: BLE001 - nvidia-smi failed
            pass
    return []


__all__ = ["Config", "Section", "HardwareProfile", "CANONICAL_STAGE_KEYS", "builtin_defaults"]
