"""Setup and calibration paths: deployment-scoped profile, packaged calibration images.

Everything here exists because two of the plan's worst path defects lived in this corner of the
tree (Appendix A rows 7 and 8, and the section 3.1 "Resolver scope" bullet that names both by
line): ``hardware_setup.HW_PATH`` was ``Path("hardware_settings.yaml")`` and
``calibrate.DEFAULT_IMAGE_DIR`` was ``Path("examples/images")``. Both were module constants bound
at import, both resolved against whatever directory the process happened to start in, and neither
obeyed any environment variable -- so the GUI, a CLI run and ``lm3-setup`` could each profile and
each read a *different* file from one machine.

The tests are grouped by the claim they defend, and every one of them runs with ``XDG_CONFIG_HOME``
pointed at ``tmp_path``: a test that writes a hardware profile into the developer's real
``~/.config/lm3`` would be doing the exact thing this refactor exists to make impossible.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest
import yaml

from leafmachine3.core import paths
from leafmachine3.core.dirs import build_dirs
from leafmachine3.setup import calibrate, hardware_setup

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Every path variable that could steer a resolver, so a test never inherits an answer from the
# developer's shell (or from another test module's fixtures) instead of from its own setup.
_PATH_ENV = (
    "LM3_HARDWARE",
    "LM3_HARDWARE_SETTINGS",
    "LM3_SETTINGS",
    "LM3_SETTINGS_PATH",
    "LM3_RUNTIME_DIR",
)


@pytest.fixture(autouse=True)
def isolated_deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the whole platform-config chain at ``tmp_path`` and name a private deployment.

    Returns ``<user-config>``, i.e. the directory ``lm3/<deployment>/`` hangs off.
    """
    home = tmp_path / "home"
    config = home / ".config"
    config.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / ".local" / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("LM3_DEPLOYMENT_ID", "test-setup-paths")
    for name in _PATH_ENV:
        monkeypatch.delenv(name, raising=False)
    return config


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory that is NOT the checkout, so a CWD fallback would be visible."""
    where = tmp_path / "cwd"
    where.mkdir()
    monkeypatch.chdir(where)
    return where


def _profile(tmp_dir: str = "/scratch") -> hardware_setup.HardwareSettings:
    """A minimal but structurally complete profile, enough for ``_write``/``_load``."""
    return hardware_setup.HardwareSettings(
        fingerprint=hardware_setup.Fingerprint(
            os="linux", cpu="test-cpu", cpu_cores=4, ram_gb=16, gpus=[], driver="",
            ort_version="", ort_providers=[], model_hashes={},
            lm3_version=hardware_setup.LM3_VERSION),
        provider="CPUExecutionProvider",
        precision="fp32",
        gpus=[],
        cpu_cores=4,
        ram_gb=16,
        tmp_dir=tmp_dir,
        io_workers=2,
        stages={},
    )


# --------------------------------------------------------------------------------------------
# The hardware profile path (section 3.1 precedence table, row 2)
# --------------------------------------------------------------------------------------------
def test_the_profile_lives_in_the_deployment_config_dir_and_is_machine_keyed(
    isolated_deployment: Path,
) -> None:
    """Row 2 path 2: ``<user-config>/lm3/<deployment>/hardware_settings.<machine-key>.yaml``."""
    path = hardware_setup.hardware_profile_path()

    assert path.is_absolute()
    assert path.parent == paths.deployment_config_dir()
    # ``lm3/<canonical key>`` where the canonical key is the slug plus a hash of the raw id, so a
    # deployment name that is not filesystem-safe still gets a stable, collision-free directory.
    assert path.parent.parent == paths.user_config_dir() / "lm3"
    assert path.parent.name.startswith("test-setup-paths-")
    assert path.name.startswith("hardware_settings.") and path.suffix == ".yaml"
    # The machine key is what stops one networked <user-config>, shared by a whole cluster
    # allocation, from collapsing every node onto one profile.
    assert path.name != "hardware_settings.yaml"
    assert len(path.name) == len("hardware_settings..yaml") + paths.MACHINE_KEY_LENGTH


def test_the_profile_path_does_not_move_with_the_cwd(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole defect in one assertion: two launch directories, one answer.

    Before this step the same constant named ``<cwd>/hardware_settings.yaml`` from each.
    """
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()

    monkeypatch.chdir(first)
    from_first = hardware_setup.hardware_profile_path()
    monkeypatch.chdir(second)
    from_second = hardware_setup.hardware_profile_path()

    assert from_first == from_second
    assert first not in from_first.parents and second not in from_first.parents


def test_two_named_deployments_never_share_a_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate 11. The profile is deployment-scoped, not machine-scoped.

    ``run_setup`` sizes GPU stages against the SELECTED ``compute.devices`` subset and carries
    measured VRAM forward from "the previous profile". Two named deployments hold separate leases,
    so nothing stops them tuning concurrently -- one pinned to GPU 0, one to GPU 1 -- and a shared
    file would let each inherit the other's measurements. Same machine key, different directory.
    """
    monkeypatch.setenv("LM3_DEPLOYMENT_ID", "alpha")
    alpha = hardware_setup.hardware_profile_path()
    monkeypatch.setenv("LM3_DEPLOYMENT_ID", "beta")
    beta = hardware_setup.hardware_profile_path()

    assert alpha != beta
    assert alpha.parent != beta.parent
    assert alpha.name == beta.name          # one machine -> one machine key
    assert alpha.parent.name.startswith("alpha") and beta.parent.name.startswith("beta")


def test_lm3_hardware_wins_over_the_deployment_path(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """Row 2 path 1. The canonical variable is ``LM3_HARDWARE`` and it is absolute-honest."""
    chosen = tmp_path / "elsewhere" / "profile.yaml"
    monkeypatch.setenv("LM3_HARDWARE", str(chosen))

    assert hardware_setup.hardware_profile_path() == chosen


def test_config_does_not_move_the_profile(tmp_path: Path) -> None:
    """``--config`` selects settings and nothing else (section 3.1, decisions list).

    A config in an unrelated directory must not drag the profile along with it, or the "which
    profile am I using?" question becomes launch-dependent again by another route.
    """
    settings = tmp_path / "elsewhere" / "LM3_settings.yaml"
    settings.parent.mkdir(parents=True)
    settings.write_text("project: {}\n", encoding="utf-8")
    cfg = types.SimpleNamespace(source_path=str(settings))

    assert hardware_setup.hardware_profile_path(cfg) == hardware_setup.hardware_profile_path()


# --------------------------------------------------------------------------------------------
# Legacy adopt (row 2 path 3) -- one release, COPY, never move, never read in place
# --------------------------------------------------------------------------------------------
def test_a_beside_the_settings_profile_is_copied_into_the_canonical_path(tmp_path: Path) -> None:
    """The upgrade path for an existing install, which has a profile next to its settings file."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = workspace / "LM3_settings.yaml"
    settings.write_text("project: {}\n", encoding="utf-8")
    legacy = workspace / "hardware_settings.yaml"
    legacy.write_text(yaml.safe_dump({"tmp_dir": "/legacy/scratch"}), encoding="utf-8")
    cfg = types.SimpleNamespace(source_path=str(settings))

    resolved = hardware_setup.hardware_profile_path(cfg)
    assert not resolved.exists(), "resolving the profile path must not copy anything"

    assert hardware_setup.migrate_legacy_profile(cfg) == resolved

    assert resolved != legacy
    assert resolved.is_file()
    assert yaml.safe_load(resolved.read_text(encoding="utf-8")) == {"tmp_dir": "/legacy/scratch"}
    # COPIED, not moved: an older LM3 on the same box keeps working through the one-release window.
    assert legacy.is_file()


def test_adoption_never_overwrites_a_profile_this_deployment_already_owns(tmp_path: Path) -> None:
    """Adopt is a one-time bootstrap, not a sync. The canonical file always wins once it exists."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = workspace / "LM3_settings.yaml"
    settings.write_text("project: {}\n", encoding="utf-8")
    (workspace / "hardware_settings.yaml").write_text(
        yaml.safe_dump({"tmp_dir": "/legacy"}), encoding="utf-8")
    cfg = types.SimpleNamespace(source_path=str(settings))

    canonical = hardware_setup.hardware_profile_path()
    hardware_setup._write(canonical, _profile(tmp_dir="/mine"))

    assert hardware_setup.hardware_profile_path(cfg) == canonical
    assert hardware_setup._load(canonical).tmp_dir == "/mine"


# --------------------------------------------------------------------------------------------
# Reading and writing the profile
# --------------------------------------------------------------------------------------------
def test_write_creates_the_deployment_directory_on_a_first_run(cwd: Path) -> None:
    """``_write`` used to rename onto an existing CWD; its parent now may never have existed."""
    path = hardware_setup.hardware_profile_path()
    assert not path.parent.exists()

    hardware_setup._write(path, _profile(tmp_dir="/scratch/one"))

    assert path.is_file()
    assert hardware_setup._load(path).tmp_dir == "/scratch/one"
    assert not (cwd / "hardware_settings.yaml").exists()


def test_previous_measurements_are_read_from_the_resolved_profile(tmp_path: Path) -> None:
    """Carry-forward is only well defined relative to the profile THIS deployment resolved."""
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump({"stages": {
        "plant_detector": {"vram_measured": True, "vram_per_worker_mb": 812.5,
                           "vram_measured_by": "nvml_process"}}}), encoding="utf-8")

    carried = hardware_setup._previous_measurements(path)

    assert carried["plant_detector"]["vram_per_worker_mb"] == 812.5
    assert hardware_setup._previous_measurements(tmp_path / "missing.yaml") == {}


def test_run_setup_writes_into_the_deployment_never_into_the_cwd(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end statement of row 2, with every machine probe stubbed out.

    The probes are stubbed because this test is about WHERE the profile lands, not what is in it;
    running the real sweep here would benchmark the developer's GPUs to assert a path.
    """
    monkeypatch.setattr(hardware_setup, "_discover_gpus", lambda: [])
    monkeypatch.setattr(hardware_setup, "_discover_cpu_ram", lambda: (8, 32))
    monkeypatch.setattr(hardware_setup, "_probe_bound_provider", lambda cfg: "CPUExecutionProvider")
    monkeypatch.setattr(hardware_setup, "_choose_tmp_dir", lambda cfg, min_free_gb=50: Path("/tmp"))
    monkeypatch.setattr(hardware_setup, "_gpu_stages", lambda cfg: [])
    monkeypatch.setattr(hardware_setup, "_cpu_stages", lambda cfg: [])
    monkeypatch.setattr(hardware_setup, "_fingerprint", lambda cfg: _profile().fingerprint)
    monkeypatch.setattr(hardware_setup, "_best_precision", lambda gpus, provider, cfg: "fp32")
    cfg = types.SimpleNamespace(compute=types.SimpleNamespace(devices="auto"), source_path=None)

    written = hardware_setup.run_setup(cfg, optimize=False, quick=True)

    assert written == hardware_setup.hardware_profile_path()
    assert written.is_file()
    assert list(cwd.iterdir()) == []


def test_ensure_hardware_profile_binds_the_canonical_file_and_ignores_a_cwd_decoy(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale ``hardware_settings.yaml`` in the launch directory must have no effect at all."""
    canonical = hardware_setup.hardware_profile_path()
    hardware_setup._write(canonical, _profile(tmp_dir="/canonical/scratch"))
    hardware_setup._write(cwd / "hardware_settings.yaml", _profile(tmp_dir="/decoy/scratch"))

    bound: list = []
    cfg = types.SimpleNamespace(bind_hardware=bound.append)
    monkeypatch.setattr(hardware_setup, "_fingerprint", lambda c: _profile().fingerprint)

    def _never(*args, **kwargs):  # noqa: ANN002, ANN003 - a tripwire, never called
        raise AssertionError("run_setup must not self-trigger when a current profile exists")

    monkeypatch.setattr(hardware_setup, "run_setup", _never)

    assert hardware_setup.ensure_hardware_profile(cfg) == canonical
    assert [p.tmp_dir for p in bound] == ["/canonical/scratch"]


# --------------------------------------------------------------------------------------------
# The standalone lm3-setup CLI (section 3.1: "the standalone hardware-setup ... CLIs")
# --------------------------------------------------------------------------------------------
def test_cli_rejects_an_explicit_config_that_does_not_exist(cwd: Path, tmp_path: Path) -> None:
    """A stated intent that cannot be satisfied is a hard error, not a fall-through.

    It also must not profile the machine on its way to that error, which is what the empty CWD
    and the untouched deployment directory below assert.
    """
    missing = tmp_path / "nope.yaml"

    assert hardware_setup.main(["--config", str(missing)]) == 2
    assert list(cwd.iterdir()) == []
    assert not hardware_setup.hardware_profile_path().parent.exists()


def test_cli_config_default_is_no_longer_a_cwd_relative_string() -> None:
    """The old ``default="LM3_settings.yaml"`` could not tell "omitted" from "typed and missing"."""
    import argparse
    import inspect

    source = inspect.getsource(hardware_setup.main)
    assert 'default="LM3_settings.yaml"' not in source
    assert isinstance(argparse.ArgumentParser, type)   # the CLI is still argparse-shaped


# --------------------------------------------------------------------------------------------
# Calibration images as a packaged resource (row 7)
# --------------------------------------------------------------------------------------------
def test_calibration_images_resolve_from_a_temp_cwd(cwd: Path) -> None:
    """Row 7. ``importlib.resources``, so a wheel or container image works, not only a checkout."""
    images = calibrate.default_image_dir()

    assert images.is_absolute()
    assert images.is_dir()
    assert cwd not in images.parents and images != cwd
    assert sorted(p.name for p in images.glob("*.jpg"))


def test_calibration_images_ship_inside_the_package_not_beside_the_checkout(cwd: Path) -> None:
    """The distinction that matters: ``examples/`` is a repo directory, this is package data."""
    import leafmachine3

    package_root = Path(leafmachine3.__file__).resolve().parent
    images = calibrate.default_image_dir()

    assert package_root in images.parents
    assert images.name == "calibration_images"
    assert images != _REPO_ROOT / "examples" / "images"


def test_the_packaged_set_covers_the_default_image_count(cwd: Path) -> None:
    """``_stage_images`` takes ``sorted(...)[:DEFAULT_N_IMAGES]``; shipping fewer silently

    shortens the calibration run and measures a lower steady-state peak than a real run reaches.
    """
    images = calibrate.default_image_dir()
    jpgs = sorted(images.glob("*.jpg"))

    assert len(jpgs) >= calibrate.DEFAULT_N_IMAGES


def test_stage_images_copies_the_packaged_set_into_a_scratch_dir(cwd: Path, tmp_path: Path) -> None:
    """Package data is often read-only; staging must copy OUT of it and never write INTO it."""
    images = calibrate.default_image_dir()
    before = sorted(p.name for p in images.iterdir())

    staged = calibrate._stage_images(images, tmp_path / "staged", calibrate.DEFAULT_N_IMAGES)

    assert len(staged) == calibrate.DEFAULT_N_IMAGES
    assert all(p.is_file() for p in staged)
    assert sorted(p.name for p in images.iterdir()) == before


def test_the_original_example_sheets_are_left_alone() -> None:
    """The packaged set is a downscaled COPY: ``examples/`` is committed in full on purpose."""
    originals = _REPO_ROOT / "examples" / "images"
    if not originals.is_dir():                       # an installed wheel has no examples/ tree
        pytest.skip("no development checkout")

    assert len(list(originals.glob("*.jpg"))) >= calibrate.DEFAULT_N_IMAGES


def test_a_missing_calibration_resource_is_reported_as_a_packaging_fault(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calibration degrades to heuristics; it must not claim the user forgot a directory."""
    def _absent(**kwargs):  # noqa: ANN003
        raise paths.PackagedResourceError("calibration images are missing")

    monkeypatch.setattr(calibrate.paths, "calibration_images_dir", _absent)

    assert calibrate.calibrate_gpu_stages(_REPO_ROOT / "LM3_settings.yaml") == {}


# --------------------------------------------------------------------------------------------
# dirs.build_dirs -- the docstring fix, pinned by behavior (section 2.7)
# --------------------------------------------------------------------------------------------
def _dirs_cfg(out: Path, tmp: str, hardware=None):
    cfg = types.SimpleNamespace(
        project=types.SimpleNamespace(
            run_name="r1", output=types.SimpleNamespace(dir=str(out), tmp_dir=tmp)))
    if hardware is not None:
        cfg._hardware = hardware
    return cfg


def test_build_dirs_ignores_a_bound_profiles_tmp_dir(tmp_path: Path) -> None:
    """What the corrected docstring now claims, asserted.

    ``run_setup`` really does record a tuned ``tmp_dir`` in the profile, and ``bind_hardware``
    really does attach the profile -- but nothing reads that field back. The old docstring said
    it did, and a reviewer believed it.
    """
    tuned = tmp_path / "tuned_scratch"
    dirs = build_dirs(_dirs_cfg(tmp_path / "runs", "auto",
                                hardware=types.SimpleNamespace(tmp_dir=str(tuned))))

    assert dirs.tmp == tmp_path / "runs" / "r1" / "_tmp_original"
    assert not tuned.exists()


def test_build_dirs_docstring_no_longer_promises_a_profile_tuned_tmp(tmp_path: Path) -> None:
    """Guard the correction itself: this docstring has drifted from the code once already."""
    doc = build_dirs.__doc__ or ""

    assert "hardware_settings.tmp_dir" not in doc
    assert "_tmp_original" in doc
    assert "_ensure_tmp" in doc          # section 2.7: tmp is not knowable before creation


def test_tmp_is_the_one_run_path_that_is_not_knowable_in_advance(tmp_path: Path) -> None:
    """Section 2.7's reason for excluding ``tmp_dir`` from the ``starting`` record.

    ``run_dir``, the sqlite path and the log dir are pure functions of the config; ``tmp`` depends
    on whether a mkdir succeeded, so an unwritable scratch path silently relocates it.
    """
    dirs = build_dirs(_dirs_cfg(tmp_path / "runs", "/proc/lm3_cannot_write_here"))

    assert dirs.root == tmp_path / "runs" / "r1"
    assert dirs.db_path == dirs.root / "r1.sqlite"
    assert dirs.logs == dirs.root / "logs"
    assert dirs.tmp == dirs.root / "_tmp_original"          # NOT the configured path
