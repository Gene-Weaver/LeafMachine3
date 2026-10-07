"""Shared pytest fixtures for the LeafMachine3 test suite.

The fixtures here synthesize small herbarium-ish specimen JPEGs with numpy + cv2 (no
network, no model weights) and assemble a ``compute.mock: true`` ``LM3_settings.yaml`` that
points at them, so the whole pipeline can be exercised end-to-end without a GPU or exports.

This module also owns the suite's RUNTIME ISOLATION (plan section 4 Step 1, gate 46): every
LM3 environment variable that steers path resolution is rewritten, at import time, to point
inside a throwaway per-session sandbox. See the block below for why it cannot be a fixture.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import pytest
import yaml

# Make ``leafmachine3`` importable when pytest is launched from the repo root without
# ``PYTHONPATH=.`` (harmless when it is already on the path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------------------------- #
# Runtime isolation -- plan section 4 Step 1 (lines 1381-1394) and gate 46
# --------------------------------------------------------------------------------------------- #
# WHY this is module-body code and not a fixture. ``tmp_path`` is function-scoped, and even a
# session-scoped fixture built on ``tmp_path_factory`` first runs when the first test runs -- long
# after pytest has imported every test module, and therefore long after
# ``from leafmachine3.machine3 import machine3`` has bound whatever the ambient environment said.
# Step 1 states the requirement in those words and the plan calls it not optional: once Step 3 wires
# the deployment lease into ``machine3()``, an un-isolated suite would take the DEVELOPER's real
# lease (``tests/test_pipeline_mock.py`` calls ``machine3()`` at :46, :335, :346, :359 and :360,
# ``tests/test_zz_repro_naming_drift.py`` at :29 and :44, and ``tests/test_working_frame.py`` at
# :158) and could block a production run on this machine.
#
# The module body of the *initial* conftest is the earliest hook pytest offers: it is imported
# during ``_prepareconfig``, before plugins are configured and before a single test module is
# imported. The block is deliberately stdlib-only and MUST stay that way -- importing anything from
# ``leafmachine3`` here would be the exact ordering bug it exists to prevent. (The imports above it
# are third-party leaves: cv2, numpy, pytest and yaml know nothing about LM3, and keeping them at
# the top of the file is what satisfies ruff E402.)

_SANDBOX_ENV = "LM3_PYTEST_SANDBOX"            # private: how an xdist worker finds the controller's sandbox
_ORIGINAL_ENV_VAR = "LM3_PYTEST_ORIGINAL_ENV"  # private: the developer's real values, JSON, controller -> workers

# Every variable the isolation takes over: the canonical LM3 names from section 3.1, the two legacy
# aliases in ``paths.LEGACY_ENV_ALIASES``, the three root-list variables the tree reads that section
# 3.1 does not list, and the XDG roots ``core/paths.py`` consults for <user-config>, <user-state>,
# <user-cache> and the platform user-runtime location.
_ISOLATED_ENV: tuple[str, ...] = (
    "LM3_RUNTIME_DIR",
    "LM3_DEPLOYMENT_ID",
    "LM3_PORT",
    "LM3_ALLOW_NETWORK_RUNTIME",
    "LM3_RUNTIME_V2",
    "LM3_SETTINGS",
    "LM3_SETTINGS_PATH",            # legacy alias of LM3_SETTINGS
    "LM3_HARDWARE",
    "LM3_HARDWARE_SETTINGS",        # legacy alias of LM3_HARDWARE
    "LM3_POSTPROCESS_SETTINGS",
    "LM3_SERVER_JOBS",
    "LM3_RUNS_ROOTS",
    "LM3_STATUS_ROOTS",
    "LM3_POSTPROCESS_ROOTS",
    "XDG_CONFIG_HOME",
    "XDG_STATE_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_RUNTIME_DIR",
    "MPLCONFIGDIR",
    _SANDBOX_ENV,
    _ORIGINAL_ENV_VAR,
)

# This process's pre-isolation values, for the session-end restore. In an xdist WORKER these are
# already the controller's isolated values, which is why the developer's real values travel
# separately in ``_ORIGINAL_ENV_VAR``.
_ENV_BEFORE: dict[str, str | None] = {name: os.environ.get(name) for name in _ISOLATED_ENV}


def _slugify(raw: str) -> str:
    """Lowercase ASCII ``[a-z0-9-]``, so a deployment id survives section 2.1's slug unchanged."""
    out = [ch if (ch.isascii() and ch.isalnum()) else "-" for ch in raw.lower()]
    return "".join(out).strip("-") or "x"


def lm3_worker_id(env: Mapping[str, str] | None = None) -> str:
    """``gw0``/``gw1``/... in a pytest-xdist worker, ``main`` in the controller or a plain run.

    xdist sets ``PYTEST_XDIST_WORKER`` in the worker process immediately before it builds that
    worker's config (``xdist/remote.py``), i.e. before this conftest is imported there -- which is
    precisely what makes an import-time derivation legal. It is never set in the controller.
    """
    values = os.environ if env is None else env
    return values.get("PYTEST_XDIST_WORKER") or "main"


def lm3_deployment_id(session_token: str, worker: str) -> str:
    """The per-session, per-worker ``LM3_DEPLOYMENT_ID``.

    ``session_token`` is the sandbox directory's unique suffix, so every worker of one session
    shares it (they inherit the sandbox through the environment) while two *sessions* never
    collide; ``worker`` separates the workers, so two of them can never contend for one lease.
    """
    return f"pytest-{_slugify(session_token)}-{_slugify(worker)}"


def _pytest_port(worker: str) -> str:
    """A distinct, never-bound port per worker.

    Gate 41 makes a named deployment without ``LM3_PORT`` a startup error, and every pytest
    deployment is named by construction, so the suite must state one. No test binds a socket; the
    value exists so a resolver that asks cannot fail the whole session.
    """
    digits = "".join(ch for ch in worker if ch.isdigit())
    return str(18765 + (int(digits) if digits else 0) % 1000)


def _isolate_runtime_environment() -> tuple[Path, bool, dict[str, str | None]]:
    """Rewrite the LM3 environment into a private sandbox. Returns (sandbox, owns_it, real_env)."""
    inherited = os.environ.get(_SANDBOX_ENV)
    if inherited and Path(inherited).is_dir():
        sandbox, owns = Path(inherited), False       # an xdist worker: reuse the controller's sandbox
    else:
        # NOT tmp_path/tmp_path_factory: neither exists yet. ``mkdtemp`` still honors TMPDIR, so a
        # box with a full /tmp can redirect the whole sandbox without touching this file.
        sandbox, owns = Path(tempfile.mkdtemp(prefix="lm3-pytest-")), True

    # The developer's real values. A worker inherits an ALREADY isolated environment, so it cannot
    # recompute them -- it reads the controller's JSON instead. Without this, every "would the real
    # runtime directory have been used?" assertion in a worker would compare the sandbox with
    # itself and pass vacuously.
    raw_original = os.environ.get(_ORIGINAL_ENV_VAR)
    if raw_original:
        real_env: dict[str, str | None] = dict(json.loads(raw_original))
    else:
        real_env = dict(_ENV_BEFORE)

    runtime = sandbox / "runtime"
    for child in (runtime, sandbox / "config", sandbox / "state", sandbox / "cache",
                  sandbox / "xdg-runtime", sandbox / "jobs"):
        child.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            os.chmod(child, 0o700)   # section 3.1 wants the runtime directory user-only

    worker = lm3_worker_id()
    # Unconditional assignment, never ``setdefault``: a developer who exports LM3_RUNTIME_DIR or
    # LM3_DEPLOYMENT_ID in their shell must not be able to aim the suite at their real deployment,
    # and under xdist a ``setdefault`` would hand every worker the controller's inherited id --
    # one shared deployment, which is exactly what gate 46's "including under xdist" forbids.
    os.environ[_SANDBOX_ENV] = str(sandbox)
    os.environ[_ORIGINAL_ENV_VAR] = json.dumps(real_env)
    os.environ["LM3_RUNTIME_DIR"] = str(runtime)
    os.environ["LM3_DEPLOYMENT_ID"] = lm3_deployment_id(sandbox.name, worker)
    os.environ["LM3_PORT"] = _pytest_port(worker)
    os.environ["XDG_CONFIG_HOME"] = str(sandbox / "config")
    os.environ["XDG_STATE_HOME"] = str(sandbox / "state")
    os.environ["XDG_CACHE_HOME"] = str(sandbox / "cache")
    # XDG_DATA_HOME matters as much as the other three: plan section 3.5 resolves
    # ``project.output.dir: auto`` to <user-data>/lm3/<deployment>/runs on an installed deployment,
    # so a test exercising the non-checkout branch would otherwise write RUN OUTPUT into the
    # developer's real data directory.
    os.environ["XDG_DATA_HOME"] = str(sandbox / "data")
    os.environ["XDG_RUNTIME_DIR"] = str(sandbox / "xdg-runtime")
    os.environ["LM3_SERVER_JOBS"] = str(sandbox / "jobs")
    # Pointed at sandbox paths that do NOT exist rather than left unset: these two files are the
    # ones the tree WRITES (the hardware profile) or reads from the CWD, so naming a sandbox path
    # positively prevents a test from reading or clobbering the developer's real profile. Every
    # reader tolerates a missing file ({} / None), which is the same answer an isolated box gives.
    os.environ["LM3_HARDWARE"] = str(sandbox / "hardware_settings.yaml")
    os.environ["LM3_POSTPROCESS_SETTINGS"] = str(sandbox / "postprocessing_settings.yaml")
    # Left UNSET rather than pointed anywhere: the settings resolver's precedence chain (section
    # 3.1 rows 1-5) is what several tests characterize, and naming a file would silently move them
    # from "unset" onto row 2 -- or, once the resolver lands, turn a missing file into a hard
    # SettingsMissingError. Unset is the honest isolated state; the XDG rewrite above keeps every
    # remaining fallback inside the sandbox.
    # The two LEGACY aliases are cleared, never set: naming one puts the whole session on the
    # deprecation path, and ``resolve_legacy_env`` warns once per process -- which would burn the
    # latch before the test that asserts on that warning ever runs.
    # LM3_RUNTIME_V2 is cleared for the same reason and then some: it is the single most
    # behavior-steering variable in the tree. A developer who exports it in their shell would
    # otherwise silently run the WHOLE suite down the other path -- including the tests that
    # characterize flag-off behavior, and including gate 60's pre-refactor comparison. Tests that
    # want the new runtime turn it on explicitly (monkeypatch.setenv), which is also what makes
    # "which path did this test exercise?" answerable by reading the test.
    for name in ("LM3_SETTINGS", "LM3_SETTINGS_PATH", "LM3_HARDWARE_SETTINGS", "LM3_RUNS_ROOTS",
                 "LM3_STATUS_ROOTS", "LM3_POSTPROCESS_ROOTS", "LM3_ALLOW_NETWORK_RUNTIME",
                 ):
        os.environ.pop(name, None)

    # LM3_RUNTIME_V2 is PINNED, not cleared. A test must never depend on what the flag DEFAULTS to:
    # the default is a product decision that has already changed once (off through Steps 3-7, on
    # from the cutover), and a suite that inherits it silently reinterprets every flag-off test the
    # day it moves. Pinning "0" here makes the ambient state explicit and stable; the ~30 tests that
    # exercise the new runtime set "1" themselves, and gate 60 now runs BOTH paths deliberately.
    # It also still closes the leak this variable was added to _ISOLATED_ENV for: a developer with
    # LM3_RUNTIME_V2 exported cannot steer the suite either way.
    os.environ["LM3_RUNTIME_V2"] = "0"

    # Purely a performance carve-out, unrelated to LM3: matplotlib caches its font list under
    # XDG_CACHE_HOME, and a sandboxed cache makes every session rebuild it (~4.5 s). Point it at
    # the cache directory matplotlib was already using, but only if that directory already exists
    # -- never create anything in the developer's home from here.
    if real_env.get("MPLCONFIGDIR") is None:
        real_cache = real_env.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        mpl_cache = Path(real_cache) / "matplotlib"
        if mpl_cache.is_dir():
            os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    return sandbox, owns, real_env


LM3_PYTEST_SANDBOX, _OWNS_SANDBOX, _REAL_ENV = _isolate_runtime_environment()


def real_environment() -> dict[str, str]:
    """``os.environ`` as it looked BEFORE the isolation -- the developer's real environment.

    The guard test resolves the *default* runtime/config/state locations against this, which is what
    makes "the default runtime directory was never touched" a real assertion instead of a
    tautology over the sandbox the suite just set up.
    """
    env = dict(os.environ)
    for name, value in _REAL_ENV.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return env


def _stat_signature(path: Path) -> tuple[bool, int | None]:
    """(exists, mtime_ns). A directory's mtime moves when an entry is created inside it, so this
    catches "a test created ``<default>/lm3/<deployment>``" as well as outright creation."""
    try:
        return True, path.stat().st_mtime_ns
    except OSError:
        return False, None


# Filled in by ``pytest_configure``: {label: (path, (exists, mtime_ns))} for the LM3-specific
# directories the DEFAULT (un-isolated) resolution would have used.
DEFAULT_LOCATIONS: dict[str, tuple[Path, tuple[bool, int | None]]] = {}
DEFAULT_LOCATION_ERRORS: dict[str, str] = {}

# Same shape, for the configuration files that live in the CHECKOUT. Reading one does not move its
# mtime, so an unchanged signature means no test wrote the developer's own settings or profile --
# the failure mode the CWD-relative ``hardware_setup.HW_PATH`` makes easy.
CHECKOUT_FILES: dict[str, tuple[Path, tuple[bool, int | None]]] = {}
_CHECKOUT_FILENAMES = ("LM3_settings.yaml", "hardware_settings.yaml", "postprocessing_settings.yaml")


def _default_locations(env: dict[str, str]) -> dict[str, Path]:
    """The LM3-owned directories a *non-isolated* LM3 would use, resolved against ``env``.

    Uses ``core/paths.py`` when it is importable -- ``runtime_base_dir`` is a pure query with
    ``create=False``, exactly so this can ask without creating anything -- and always adds the
    literal platform defaults as a floor, so the guard keeps working if the resolver moves.
    """
    home = Path(env.get("HOME") or Path.home())
    literal = {
        "literal:config": Path(env.get("XDG_CONFIG_HOME") or home / ".config") / "lm3",
        "literal:state": Path(env.get("XDG_STATE_HOME") or home / ".local" / "state") / "lm3",
        "literal:data": Path(env.get("XDG_DATA_HOME") or home / ".local" / "share") / "lm3",
        "literal:cache": Path(env.get("XDG_CACHE_HOME") or home / ".cache") / "lm3",
    }
    xdg_runtime = env.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        literal["literal:xdg-runtime"] = Path(xdg_runtime) / "lm3"

    try:
        from leafmachine3.core import paths as P
    except Exception as exc:                      # noqa: BLE001 - the floor above still applies
        DEFAULT_LOCATION_ERRORS["import"] = f"{type(exc).__name__}: {exc}"
        return literal

    resolved = dict(literal)
    probes: dict[str, object] = {
        "resolver:runtime-base": lambda: P.runtime_base_dir(env=env, create=False, check_filesystem=False),
        "resolver:config": lambda: P.user_config_dir(env) / P.APP_DIRNAME,
        "resolver:state": lambda: P.user_state_dir(env) / P.APP_DIRNAME,
        "resolver:cache": lambda: P.user_cache_dir(env) / P.APP_DIRNAME,
    }
    for label, probe in probes.items():
        try:
            resolved[label] = Path(probe())       # type: ignore[operator]
        except Exception as exc:                  # noqa: BLE001 - a probe that cannot answer is recorded, not fatal
            DEFAULT_LOCATION_ERRORS[label] = f"{type(exc).__name__}: {exc}"
    return resolved


def pytest_configure(config: pytest.Config) -> None:
    """Snapshot the default LM3 locations before any test runs.

    Deliberately here and not in the module body: this is the first hook that may import
    ``leafmachine3`` -- the environment is already isolated by then, and no test module has been
    imported yet, so the snapshot is a true "before" picture.
    """
    del config
    for label, path in _default_locations(real_environment()).items():
        DEFAULT_LOCATIONS[label] = (path, _stat_signature(path))
    for filename in _CHECKOUT_FILENAMES:
        path = _REPO_ROOT / filename
        CHECKOUT_FILES[f"checkout:{filename}"] = (path, _stat_signature(path))


def _restore_environment() -> None:
    for name, value in _ENV_BEFORE.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _teardown_runtime_isolation() -> None:
    """Remove the sandbox and put the environment back. Idempotent; safe from any rank.

    Only the process that CREATED the sandbox removes it. An xdist worker inherits the
    controller's directory, and ``pytest_sessionfinish`` runs in every worker as well as in the
    controller -- a worker deleting the tree would delete its siblings' live deployment
    directories. The controller's session finishes only after every worker has exited, so the
    single owning delete is also the last one.
    """
    if getattr(_teardown_runtime_isolation, "_done", False):
        return
    _teardown_runtime_isolation._done = True       # type: ignore[attr-defined]
    if _OWNS_SANDBOX:
        shutil.rmtree(LM3_PYTEST_SANDBOX, ignore_errors=True)
    _restore_environment()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del session, exitstatus
    _teardown_runtime_isolation()


# A collection error can end the run without ``pytest_sessionfinish`` firing for a directory the
# module body already created, so the cleanup is also armed at interpreter exit.
atexit.register(_teardown_runtime_isolation)


# All test/run artifacts land here (gitignored), one subdir per test name, so a human can
# inspect the DB + overlays after a run instead of digging through pytest tmp dirs.
EXAMPLES_OUT = _REPO_ROOT / "examples_out"


def fresh_out_dir(name: str) -> Path:
    """Return (and clear) ``examples_out/<name>`` so each test run starts clean."""
    import shutil

    d = EXAMPLES_OUT / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_specimen_image(path: Path, seed: int) -> Path:
    """Write one deterministic synthetic specimen JPEG (a green 'leaf' on a pale sheet)."""
    rng = np.random.default_rng(seed)
    h, w = 900, 700
    img = np.full((h, w, 3), 232, dtype=np.uint8)  # pale herbarium-sheet background
    # a leaf-ish green blob
    cv2.ellipse(img, (int(0.35 * w), int(0.45 * h)), (140, 220), 25, 0, 360, (40, 130, 45), -1)
    # a 'ruler' strip along the bottom
    cv2.rectangle(img, (40, h - 70), (w - 40, h - 40), (60, 60, 60), -1)
    # a 'label' rectangle
    cv2.rectangle(img, (int(0.62 * w), int(0.12 * h)), (int(0.92 * w), int(0.34 * h)), (250, 250, 240), -1)
    # a touch of noise so the two specimens differ
    noise = rng.integers(0, 12, size=(h, w, 3), dtype=np.uint8)
    img = cv2.subtract(img, noise)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


@pytest.fixture
def synthetic_images(tmp_path: Path) -> Path:
    """Directory holding two synthetic specimen JPEGs."""
    images_dir = tmp_path / "input_images"
    for i in range(2):
        make_specimen_image(images_dir / f"specimen_{i:02d}.jpg", seed=i + 1)
    return images_dir


def build_mock_config(
    images_dir: Path,
    output_dir: Path,
    *,
    run_name: str = "test_run",
    ruler_classifier_enabled: bool = False,
) -> dict:
    """Return a full mock LM3 config dict pointing at ``images_dir`` / ``output_dir``.

    ``compute.mock`` is on (deterministic synthetic models) and ``output.tmp_dir`` is a
    concrete path so the run never depends on a tuned scratch directory.
    """
    return {
        "version": 3,
        "project": {
            "run_name": run_name,
            "input": {
                "dirs": [str(images_dir)],
                "recursive": True,
                "image_extensions": [".jpg", ".jpeg", ".png"],
            },
            "output": {"dir": str(output_dir), "tmp_dir": str(output_dir / "_scratch")},
            "run_mode": {"overwrite": False, "restart": [], "fail_fast": True},
            "logging": {"level": "WARNING", "to_file": False, "to_console": False},
        },
        "compute": {"devices": "cpu", "mock": True, "precision": "fp32"},
        "ingest": {"max_working_dim": 3200, "jpg_quality": 95},
        "modules": {
            "mp_conversion_factor": {
                "enabled": True,
                "model": {"path": "models/mp_conversion_factor/model.json", "format": "json"},
            },
            "archival_detector": {
                "enabled": True,
                "classes": ["Ruler", "Barcode", "Colorcard", "Label"],
            },
            "plant_detector": {
                "enabled": True,
                "classes": ["Leaf_WHOLE", "Leaf_PARTIAL", "Seed_Fruit_ONE"],
            },
            "specimen_segmenter": {"enabled": True, "paperclean": True},
            "phenology_detector": {
                "enabled": True,
                "targets": {
                    "leaves": {"min_conf": 0.3, "min_count": 1},
                    "flowers": {"min_conf": 0.4, "min_count": 1},
                    "fruits": {"min_conf": 0.4, "min_count": 1},
                },
            },
            "ruler_classifier": {
                "enabled": ruler_classifier_enabled,
                "models_dir": "models/ruler_classifier",
                "ensemble_members": ["a", "b", "c"],
                "min_conf": 0.35,
            },
            "ruler_cf": {"enabled": ruler_classifier_enabled},   # lattice CF; needs the classifier's tiles + verdicts
            "leaf_segmenter": {"enabled": True, "include_partial": False},
            "morphology": {"enabled": True, "classes": ["Leaf"], "find_minimum_bounding_box": True},
            "landmark_detector": {"enabled": True, "source_classes": ["Leaf_WHOLE"], "include_partial": False},
            "landmark_measurements": {"enabled": True, "min_kpt_conf": 0.25},
            "leaf_orientation": {"enabled": True, "min_kpt_conf": 0.25, "min_midvein": 5},
            "petiole_width": {"enabled": True, "min_kpt_conf": 0.25, "touch_dist_px": 20},
            "metric_grounding": {"enabled": True, "round_ndigits": 4},
            "reporter": {"enabled": True},
            "ect": {"enabled": True, "num_dirs": 64, "radial_viz": True, "cartesian_viz": True,
                    "radial_overlay_viz": True},
        },
        "naming": {
            "bbox_prefix": "BBOX",
            "seg_prefix": "SEG",
            "landmark_prefix": "LM",
            "friendly_names": {"Leaf_WHOLE": "leaf", "Leaf_PARTIAL": "leafReject",
                               "Ruler": "ruler", "Label": "label", "Leaf": "leaf",
                               "Specimen": "specimen", "Specimen_Inverse": "specimenInverse"},
        },
        "report": {
            "overlay": {"enabled": True, "draw_masks": True, "draw_landmarks": True,
                        "draw_labels": True, "box_style": "rotated"},
            "overlay_landmarks": {"enabled": True},
            "masks": {
                "classes": ["Leaf"],
                "background": "black",
                "subtract_holes": True,
                "Binary_Masks_Full_Image": True,
                "Binary_Masks": True,
                "RGB_Masks_Full_Image": True,
                "RGB_Masks": True,
                # the not-specimen complement, with a fill that is neither black nor white so the
                # end-to-end test can tell "the fill was applied" from "the background leaked in"
                "Binary_Masks__Specimen_Inverse": True,
                "RGB_Masks__Specimen_Inverse": True,
                "inverse_fill": [255, 0, 0],
            },
            "crops": {"enabled": True, "classes": "all"},
            "overlay_petiole": {"enabled": True},
            "overlay_specimen": {"enabled": True},
            "leaf_products": {"enabled": True, "original": True, "oriented": True, "background": "black"},
            "formats": {"image_ext": "jpg", "jpg_quality": 95, "mask_ext": "png"},
        },
    }


@pytest.fixture
def mock_config_path(synthetic_images: Path, tmp_path: Path) -> Path:
    """Write a mock ``LM3_settings.yaml`` and return its path."""
    output_dir = fresh_out_dir("mock_config")
    cfg = build_mock_config(synthetic_images, output_dir)
    cfg_path = tmp_path / "LM3_settings.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path
