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
    "mp_conversion_factor",
    "archival_detector",
    "plant_detector",
    "specimen_segmenter",
    "phenology_detector",
    "ruler_classifier",
    "ruler_cf",
    "leaf_segmenter",
    "morphology",
    "landmark_detector",
    "landmark_measurements",
    "leaf_orientation",
    "petiole_width",
    "bilateral_symmetry",
    "metric_grounding",
    "reporter",
    "ect",
    "momocs",
)

# Stages that require an exported single-file model artifact when enabled.
_MODEL_PATH_STAGES: tuple[str, ...] = ("mp_conversion_factor", "archival_detector", "plant_detector",
                                       "specimen_segmenter", "leaf_segmenter", "landmark_detector")

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


def jsonable(value: Any) -> Any:
    """Deterministic, JSON-safe projection of a config subtree.

    Stronger than :func:`_plain` in the two ways the plan's launch manifest needs (section 3.4):
    every mapping comes back with its keys in sorted order, and anything JSON cannot carry --
    ``Path``, the ``datetime.date`` PyYAML produces for an unquoted ``2026-08-28``, an enum -- is
    normalized to a string rather than exploding at dump time. Paths go through ``Path`` first so
    ``runs/`` and ``runs`` cannot fingerprint differently.

    Sorting here as well as at ``json.dumps(sort_keys=True)`` is deliberate: the DICT itself is then
    already canonical, so a caller that embeds it in a larger structure, hashes ``repr``, or diffs
    two dumps gets the same answer as the serializer does.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, os.PathLike):
        return str(Path(value))
    # Anything else (dates, enums, arbitrary objects an override smuggled in) is described, never
    # dropped: a manifest that silently loses a key is worse than one carrying "2026-08-28".
    return str(value)


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
            # No keep_tmp knob. It was declared for a cleanup that was never implemented, and under the
            # working-frame contract it must never be: _tmp_original holds the DOWNSAMPLED copies, which
            # are now the only pixels the Reporter reads. A switch promising to delete them is a footgun.
            # ``auto``, not ``runs``: plan section 3.5. A relative default would follow rule 1 into
            # <user-config>/lm3/<deployment>/, i.e. write run outputs into a hidden CONFIG dir.
            # ``auto`` is <checkout>/runs in a dev checkout and <user-data>/lm3/<deployment>/runs
            # when installed -- see paths.default_output_dir().
            "output": {"dir": "auto", "tmp_dir": "auto"},
            "run_mode": {"overwrite": False, "restart": [], "fail_fast": False},
            "logging": {"level": "INFO", "to_file": True, "to_console": True},
        },
        "compute": {
            "devices": "auto",
            "tensorrt": False,
            "mock": False,
            "io_workers": "auto",
            "force_reprobe": False,
            # Dispatch the biggest WorkItems first so a stage does not finish waiting on one
            # huge sheet while every other worker idles. Set False to restore raw DB order.
            "longest_first": True,
            "vram": {
                "safety_fraction": 0.90,
                "reserve_mb": 1024,
                "per_worker_mb": "auto",
                # A safety rail, not a target: the real count comes from each card's live free
                # VRAM divided by one worker's measured cost. Kept high enough that VRAM -- not
                # this number -- is what limits a big card.
                "max_workers_per_gpu": 16,
                # OOM breathing room on a MEASURED per-worker figure. Calibration samples one
                # image set on one card, so a bigger sheet can cost more than was seen; 1.15
                # buys 15% slack before the allocator commits to a worker count.
                "headroom_factor": 1.15,
                # Workers that may warm-load a model at once. Loading strictly one at a time
                # costs n x load_time of ramp, which makes a short module SLOWER the more
                # workers it gets. 0 -> unbounded (all n load together).
                "concurrent_warm_loads": 4,
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
            "mp_conversion_factor": {"enabled": True},
            "archival_detector": {"enabled": True},
            "plant_detector": {"enabled": True},
            # The model is chosen by NAME; the default names the UNet++ (see inference/specimen_models.py).
            "specimen_segmenter": {"enabled": True, "model": {"key": "unetpp_effb7_1024"},
                                   "yolo": {"conf": 0.25, "iou": 0.5, "max_det": 300}},
            "phenology_detector": {"enabled": True},
            "ruler_classifier": {"enabled": True},
            # lattice conversion-factor method. use_CF_predicted_by_MP is here (not only in the
            # YAML) so the settings form, which renders leaves of the MERGED tree, always shows it.
            # The `fieldprism` block drives FieldPrism (FP) marker sheets; its tolerances mirror
            # inference.ruler_lattice.fieldprism.FP_PEER_TOL / FP_ANCHOR_TOL.
            "ruler_cf": {"enabled": True, "use_CF_predicted_by_MP": True,
                         "fieldprism": {"enabled": True, "peer_tol": 0.03, "anchor_tol": 0.03,
                                        "allow_single_marker": True}},
            "leaf_segmenter": {"enabled": True},
            "morphology": {"enabled": True},
            "landmark_detector": {"enabled": True},
            "landmark_measurements": {"enabled": True},
            "leaf_orientation": {"enabled": True},
            "petiole_width": {"enabled": True},
            # Was absent, which made this the ONE stage that defaulted off:
            # is_enabled() reads `bool(node and node.get("enabled", False))`, so a
            # missing default is a disabled module. Every settings UI reads the
            # merged tree and showed it as ON, so a config without this key ran
            # 16 of 17 stages while reporting 17.
            "bilateral_symmetry": {"enabled": True},
            "metric_grounding": {"enabled": True},
            "reporter": {"enabled": True},
            "ect": {"enabled": True},
            # Every knob, not just `enabled`: the settings form renders only leaves of the MERGED
            # tree, and existing LM3_settings.yaml files predate this block.
            "momocs": {"enabled": True, "include_petiole": False, "oriented": True, "pad_px": 10,
                       "largest_component_only": True, "jpg_quality": 100, "outline_points": 0,
                       "write_fac_csv": True, "write_momit_json": True},
        },
        # NB: no EMPTY "report" mapping. An empty mapping is not a no-op here -- the
        # settings form walks the merged tree and treats a childless dict as a LEAF,
        # so `"report": {}` rendered as a free-text row holding "{}" on a fresh
        # install. Config.report already returns an empty Section when the key is
        # absent, so nothing needs a placeholder. Populated subtrees are fine, and
        # report.data is here because it has to be: the settings form renders a row
        # only for a leaf of the MERGED tree, so a knob that lives solely in Python
        # defaults is invisible and unsettable. report.data was exactly that -- the
        # ten CSV toggles were documented in settings_meta.json and rendered nowhere,
        # so the export bundle could not be configured from the file or the GUI at
        # all. Keep this in step with reporting.data_export.FILES; a test compares them.
        "report": {
            "data": {
                "enabled": True,
                "folder": "Data",
                "format": "csv",
                "na_rep": "",
                "float_precision": 6,
                "files": {
                    "leaf_measurements": True,
                    "specimen_summary": True,
                    "phenology": True,
                    "detections": True,
                    "landmarks": True,
                    "ruler_conversion_factor": True,
                    "ruler_crops": True,
                    "fieldprism_markers": True,
                    "fieldprism_sheets": True,
                    "run_stages": True,
                    "stage_errors": True,
                    "data_dictionary": True,
                },
            },
        },
        "timing": {"enabled": False, "sample_interval_s": 0.25},
    }


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

#: Settings LM3 no longer reads, and why. A file that still carries one is warned ONCE per run, by
#: machine3, naming the key and the file -- not at every use (the Reporter used to warn once per
#: specimen) and not inside Config.load(), which the server calls on every settings request.
RETIRED_SETTINGS: dict[str, str] = {
    "project.output.keep_tmp": "_tmp_original holds the downsampled copies the Reporter reads, so it is never deleted",
    "report.crops.source": "crops are always cut from the working image, the frame every measurement is made in",
    "report.overlay.draw_boxes_archival": "box border/fill is configured per group under report.overlay.groups",
    "report.overlay.draw_boxes_plant": "box border/fill is configured per group under report.overlay.groups",
    "report.overlay.line_width_archival": "line widths are configured per group under report.overlay.groups",
    "report.overlay.line_width_plant": "line widths are configured per group under report.overlay.groups",
}

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

        cfg = cls(merged)
        # Remember where this came from: the VRAM calibrator has to re-launch LM3 as a
        # subprocess against a derived copy of this same file.
        object.__setattr__(cfg, "source_path", str(file_path.resolve()))
        return cfg

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

    @property
    def timing(self) -> Section:
        """Run-timing profiler settings (``enabled`` -> reports/Timing/timing.{csv,html})."""
        return self._raw.get("timing", Section())

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
        """Absolutize ``p`` against the SETTINGS FILE that produced this config.

        Plan section 3.5 rule 1: a relative path written in a YAML settings file resolves against the
        directory containing that file -- never the process CWD. This one method is the seam: input
        dirs (``core/ingest.py``), artifact validation (``core/validate.py``,
        ``server/settings_api.py``), model and ``models_dir`` loading (``inference/factory.py``) and
        the setup scratch probe (``setup/hardware_setup.py``) all route through it, so they cannot
        drift apart again.

        It used to join ``Path.cwd()``. That made the same YAML mean different files depending on
        where the launcher happened to be standing, and once Step 1 moved settings resolution to a
        canonical path it also made ``build_dirs()`` and ``runtime.config_io.resolve_run_paths()``
        disagree about the run directory inside a single process.

        A relative path with no known settings file raises: rule 6 forbids inventing a base.
        """
        path = Path(str(p)).expanduser()
        if path.is_absolute():
            return str(path)
        source = getattr(self, "source_path", None)
        if not source:
            raise ValueError(
                f"cannot resolve the relative path {str(p)!r}: this Config has no source_path, and "
                f"LM3 never joins a configured path onto the current working directory "
                f"(plan section 3.5 rule 6). Load the config with Config.load() or pass an absolute path."
            )
        base = Path(source).resolve().parent
        # normpath, not resolve(): collapse ".." lexically so a migrated "../models/x.onnx" comes
        # back as a clean absolute path, without following symlinks or requiring the file to exist.
        return os.path.normpath(str(base / path))

    def resolve_model_path(self, p: str | os.PathLike[str]) -> str:
        """Like :meth:`resolve_path`, but a relative ``models/...`` path re-roots onto ``$LM3_MODELS_DIR``.

        The settings YAML keeps saying ``models/archival_detector/model.onnx`` everywhere; a packaged
        app, a Docker image or a cluster job points ``LM3_MODELS_DIR`` at wherever ``lm3 models
        install`` put the files, and a source checkout leaves it unset so the path resolves beside
        the settings file as before. Every model/``models_dir`` load goes through here.
        """
        path = Path(str(p)).expanduser()
        if path.is_absolute():
            return str(path)
        root = os.environ.get("LM3_MODELS_DIR")
        parts = path.parts
        if root and parts and parts[0] == "models":
            return os.path.normpath(str(Path(root).expanduser() / Path(*parts[1:])))
        return self.resolve_path(p)

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

    def probe_vram_mb(self, stage: Any) -> int | None:
        """An EXPLICIT per-worker VRAM override (MB) for ``stage``, or ``None``.

        This is only the user's ``compute.vram.per_worker_mb``, taken verbatim -- an escape
        hatch for when the measurement is wrong. ``None`` means "no override", which hands the
        decision back to the executor's own chain (measured -> estimated -> default, each with
        the OOM headroom factor applied).

        It deliberately no longer falls back to the profile's ``peak_vram_mb``: that field is
        the total across ALL workers, so returning it here as a per-worker cost made the
        executor fit exactly one worker per GPU no matter how much VRAM was free.
        """
        per_worker = self.compute.get("vram", Section()).get("per_worker_mb", "auto")
        if isinstance(per_worker, (int, float)) and not isinstance(per_worker, bool):
            return max(1, int(per_worker))
        return None

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

    # -- serialization ------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """The complete EFFECTIVE config -- defaults < YAML < overrides -- as plain JSON data.

        This is what the section 3.4 launch manifest embeds and what config fingerprinting reads,
        so it must be deterministic: the same ``Config`` produces byte-identical JSON every time.
        :func:`jsonable` sorts every mapping and normalizes paths to get there.

        It is a deep COPY. Mutating the result cannot reach back into the live config, which is why
        the manifest writer may hand it straight to ``json.dump`` without a defensive copy.

        ``source_path`` is deliberately not folded in: it describes where the config came from, not
        what it says, and the manifest carries it separately alongside the file's SHA-256.
        """
        return jsonable(self._raw)

    # -- settings hashing (resume invalidation) ----------------------------- #
    #: Stages whose behavior is driven by config OUTSIDE their own ``modules.<key>`` block.
    #: The Reporter is configured almost entirely by ``report.*`` -- ``modules.reporter`` holds
    #: nothing but ``enabled`` -- so hashing only that block made every export toggle invisible to
    #: drift detection: switching one on for a finished run left the stage ``done`` and produced
    #: nothing, with no error anywhere. Anything listed here is hashed alongside the module block.
    _EXTRA_HASH_BLOCKS: dict[str, tuple[str, ...]] = {"reporter": ("report",)}

    def stage_settings_hash(self, key: str) -> str:
        """Stable hash of a stage's resolved settings plus its model file signature.

        Changing any knob under ``modules.<key>`` (plus, for the Reporter, anything under
        ``report``) -- or replacing the exported model file (its size/mtime) -- yields a different
        hash, letting the DB invalidate that stage's prior results on resume.
        """
        hasher = hashlib.sha256()
        block = _plain(self.stage(key))
        hasher.update(json.dumps(block, sort_keys=True, default=str).encode("utf-8"))
        for extra in self._EXTRA_HASH_BLOCKS.get(key, ()):
            hasher.update(f"|{extra}:".encode("utf-8"))
            hasher.update(
                json.dumps(_plain(self._raw.get(extra, Section())), sort_keys=True,
                           default=str).encode("utf-8")
            )
        for artifact in self._stage_artifacts(key):
            try:
                st = Path(artifact).stat()
            except OSError:
                continue
            hasher.update(f"|{artifact}:{st.st_size}:{int(st.st_mtime)}".encode("utf-8"))
        return hasher.hexdigest()

    def retired_settings(self) -> list[tuple[str, str]]:
        """``[(dotted key, why)]`` for every :data:`RETIRED_SETTINGS` key this file still sets. Pure."""
        found = []
        for dotted, why in RETIRED_SETTINGS.items():
            node: Any = self._raw
            for part in dotted.split("."):
                node = node.get(part) if hasattr(node, "get") else None
                if node is None:
                    break
            if node is not None:
                found.append((dotted, why))
        return found

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

        # The specimen segmenter is chosen by NAME (model.key); an unknown name cannot run in any
        # mode. A missing key is not an error -- it resolves to the UNet++ default with a warning.
        if self.is_enabled("specimen_segmenter"):
            from leafmachine3.inference.specimen_models import (  # noqa: PLC0415 - lazy, avoids a cycle
                UnknownSpecimenModel, resolve_specimen_model,
            )
            try:
                resolve_specimen_model(self.stage("specimen_segmenter"))
            except UnknownSpecimenModel as exc:
                errors.append(str(exc))

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


__all__ = ["Config", "Section", "HardwareProfile", "CANONICAL_STAGE_KEYS", "builtin_defaults",
           "jsonable"]
