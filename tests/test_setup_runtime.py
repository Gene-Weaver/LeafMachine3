"""Hardware setup and calibration as runtime activities -- plan Step 3.

Sections under test:

* **2.2** -- ``hardware_setup`` is a ROOT activity; ``calibration_pipeline`` is the only inherited
  subactivity; the child gets a PROVISIONAL profile so it does not bootstrap a second setup sweep
  inside the measurement; a calibration that was REQUESTED and failed is loud.
* **2.13** -- setup requires a resolved config (a missing one is a named error, never an
  ``AttributeError``), and it runs as an ``lm3-setup`` subprocess in its own process group so Stop
  can kill the tuning tree without killing the server.
* **3.2** -- the ``hardware_setup`` record requires ``config`` and ``hardware{destination_path}``
  and deliberately carries NO project block (invariant 6).

Everything here is behind ``LM3_RUNTIME_V2``, so each test says which side of the flag it is on.
With the flag off the behavior must be what it was before Step 3: a whole-environment copy, a
warning, and heuristic estimates.

Nothing in this file starts a real pipeline, touches a GPU, or runs a benchmark: the expensive
probes are stubbed and the child process is a fake ``Popen``. What is real is the lease, the grant,
and the composed launch keywords -- the parts that break silently.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from leafmachine3.core import paths
from leafmachine3.core.config import Config
from leafmachine3.core.runtime import execution as ex
from leafmachine3.core.runtime._types import (
    ACTIVE_RECORD_FILENAME,
    CALIBRATION_RUN_NAME,
    ENV_LEASE_CAPABILITY,
    ENV_STATUS_FD,
    EXIT_CODE_BUSY,
    Activity,
    RuntimeBusyError,
)
from leafmachine3.setup import calibrate as cal
from leafmachine3.setup import hardware_setup as hw

KEY = "pytest-setup-runtime"


# --------------------------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LM3_RUNTIME_V2`` is NOT in ``tests/conftest.py``'s isolation list.

    So a developer with it exported would otherwise run this whole file in the opposite mode from
    CI, and the flag-off assertions would pass for the wrong reason. Force the default here.
    """
    monkeypatch.setenv(ex.ENV_RUNTIME_V2, "0")


@pytest.fixture()
def v2(monkeypatch: pytest.MonkeyPatch, flag_off: None) -> None:
    """Step 3 wiring on. Depends on ``flag_off`` so it wins the ordering, not luck."""
    monkeypatch.setenv(ex.ENV_RUNTIME_V2, "1")


def write_settings(tmp_path: Path, name: str = "LM3_settings.yaml") -> Path:
    """A minimal but realistic settings file, with absolute paths so nothing depends on the CWD."""
    data = {
        "project": {
            "run_name": "acer_rubrum",
            "input": {"dirs": [str(tmp_path / "input")]},
            "output": {"dir": str(tmp_path / "output"), "tmp_dir": "auto"},
        },
        "compute": {"mock": True},
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    return Config.load(write_settings(tmp_path))


@pytest.fixture()
def deployment(tmp_path: Path) -> Path:
    """A private deployment runtime directory. Never the developer's -- see tests/conftest.py."""
    return tmp_path / "runtime" / KEY


@pytest.fixture()
def hw_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``LM3_HARDWARE`` at a tmp profile, so no test can reach the real one."""
    target = tmp_path / "profile" / "hardware_settings.pytest.yaml"
    monkeypatch.setenv(paths.ENV_HARDWARE, str(target))
    return target


def setup_root(deployment_dir: Path, config: Config, **kwargs):
    """A live ``hardware_setup`` root, with the flag forced on and no handshake."""
    kwargs.setdefault("enabled", True)
    kwargs.setdefault("announce", False)
    kwargs.setdefault("deployment_dir", deployment_dir)
    kwargs.setdefault("deployment_key", KEY)
    return hw.hardware_setup_activity(config, **kwargs)


def active_json(deployment_dir: Path) -> dict:
    return json.loads((deployment_dir / ACTIVE_RECORD_FILENAME).read_text(encoding="utf-8"))


class FakeProc:
    """Enough of ``Popen`` for the launch plumbing: a handle that can be polled and joined."""

    def __init__(self, returncode: int = 0, stderr: str = "", *, hang: bool = False) -> None:
        self.returncode = returncode
        self._stderr = stderr
        self._hang = hang
        self.killed = False
        self.terminated = False

    def communicate(self, timeout: float | None = None):
        if self._hang:
            self._hang = False                     # a kill(), then a second drain, must succeed
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0.0)
        return "", self._stderr

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def stub_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace everything in ``run_setup`` that touches real hardware or takes real time."""
    class _Stage:
        def __init__(self, key: str) -> None:
            self.key = key

    monkeypatch.setattr(hw, "_discover_gpus", lambda: [])
    monkeypatch.setattr(hw, "_discover_cpu_ram", lambda: (8, 32))
    monkeypatch.setattr(hw, "_probe_bound_provider", lambda cfg: "cpu")
    monkeypatch.setattr(hw, "_choose_tmp_dir", lambda cfg, min_free_gb=50: Path("/tmp"))
    monkeypatch.setattr(hw, "_best_precision", lambda gpus, provider, cfg: "fp32")
    monkeypatch.setattr(hw, "_gpu_stages", lambda cfg: [_Stage("plant_detector")])
    monkeypatch.setattr(hw, "_cpu_stages", lambda cfg: [])
    monkeypatch.setattr(hw, "_fingerprint", lambda cfg: hw.Fingerprint(
        os="linux", cpu="x", cpu_cores=8, ram_gb=32, gpus=[], driver="", ort_version="",
        ort_providers=[], model_hashes={}, lm3_version=hw.LM3_VERSION))


# --------------------------------------------------------------------------------------------- #
# Section 2.13 -- setup requires a resolved config
# --------------------------------------------------------------------------------------------- #

def test_run_setup_with_no_config_is_a_named_error_not_an_attributeerror() -> None:
    """The live crash section 2.13 describes: ``app.py`` passed ``None`` and ``_fingerprint(cfg)``
    dereferenced it three frames later."""
    with pytest.raises(hw.SetupConfigError) as excinfo:
        hw.run_setup(None)
    assert "resolved config" in str(excinfo.value)


def test_hardware_setup_activity_refuses_a_missing_config_too(deployment: Path) -> None:
    with pytest.raises(hw.SetupConfigError):
        with hw.hardware_setup_activity(None, deployment_dir=deployment, deployment_key=KEY):
            pass


def test_resolve_setup_config_names_the_path_it_could_not_find(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere" / "LM3_settings.yaml"
    with pytest.raises(hw.SetupConfigError) as excinfo:
        hw.resolve_setup_config(missing)
    assert str(missing) in str(excinfo.value)


def test_resolve_setup_config_loads_the_settings_it_resolved(tmp_path: Path) -> None:
    path = write_settings(tmp_path)
    resolved, cfg = hw.resolve_setup_config(path)
    assert resolved == path.resolve()
    assert Path(cfg.source_path).resolve() == path.resolve()


def test_resolve_setup_config_rejects_an_unreadable_settings_file(tmp_path: Path) -> None:
    broken = tmp_path / "LM3_settings.yaml"
    broken.write_text("project: [this is not a mapping\n", encoding="utf-8")
    with pytest.raises(hw.SetupConfigError):
        hw.resolve_setup_config(broken)


# --------------------------------------------------------------------------------------------- #
# Section 2.13 -- the lm3-setup subprocess
# --------------------------------------------------------------------------------------------- #

def test_setup_argv_carries_every_flag_the_server_can_set(tmp_path: Path) -> None:
    argv = hw.setup_argv(config=tmp_path / "s.yaml", optimize=True, quick=True, force=True,
                         calibrate=True, tmp=tmp_path, allow_calibration_fallback=True,
                         events=tmp_path / "events.jsonl")
    assert argv[1:3] == ["-m", "leafmachine3.setup"]
    for flag in ("--optimize", "--quick", "--force", "--calibrate",
                 "--allow-calibration-fallback", "--config", "--tmp", "--events"):
        assert flag in argv
    # It must be runnable from a development checkout, where the console script is not on PATH.
    assert argv[0].endswith(("python", "python3", "python.exe")) or "python" in argv[0]


def test_setup_argv_omits_what_was_not_asked_for() -> None:
    assert hw.setup_argv(optimize=False) == [hw.sys.executable, "-m", "leafmachine3.setup"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_setup_gets_its_own_process_group_so_stop_spares_the_server() -> None:
    """Section 2.13: Stop must kill the tuning tree and leave the server running."""
    assert hw.setup_popen_kwargs() == {"start_new_session": True}


def test_jsonl_progress_appends_one_object_per_event(tmp_path: Path) -> None:
    log_path = tmp_path / "events" / "setup.jsonl"
    emit = hw.jsonl_progress(log_path)
    emit("sweep", "plant_detector")
    emit("done", "written")
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["phase"] for line in lines] == ["sweep", "done"]
    assert json.loads(lines[0])["message"] == "plant_detector"


def test_jsonl_progress_never_aborts_a_setup_run(tmp_path: Path) -> None:
    """Progress reporting is a nicety; a full disk must not lose the profile."""
    emit = hw.jsonl_progress(tmp_path / "events.jsonl")
    (tmp_path / "events.jsonl").mkdir()            # a directory where the log should be
    emit("sweep", "still fine")


# --------------------------------------------------------------------------------------------- #
# Section 3.2 -- the hardware_setup record
# --------------------------------------------------------------------------------------------- #

def test_flag_off_setup_acquires_nothing_and_publishes_nothing(cfg: Config, deployment: Path,
                                                               hw_path: Path) -> None:
    with hw.hardware_setup_activity(cfg, deployment_dir=deployment, deployment_key=KEY) as activity:
        assert activity.enabled is False
    assert not deployment.exists()


def test_hardware_setup_record_carries_config_and_destination_but_no_project(
    cfg: Config, deployment: Path, hw_path: Path, v2: None
) -> None:
    """Section 3.2's activity table, and invariant 6: tuning the machine is not work on a project."""
    with setup_root(deployment, cfg) as activity:
        assert activity.enabled is True
        record = active_json(deployment)
    assert record["activity"] == "hardware_setup"
    assert record["activity_role"] == "root"
    assert record["config"]["path"] == str(Path(cfg.source_path).resolve())
    assert record["config"]["sha256"]
    assert record["hardware"]["destination_path"] == str(hw_path.resolve())
    assert record.get("project") is None


def test_a_second_root_is_refused_while_setup_holds_the_deployment(
    cfg: Config, deployment: Path, hw_path: Path, v2: None
) -> None:
    """Step 3's exit gate, from the setup side."""
    with setup_root(deployment, cfg):
        with pytest.raises(RuntimeBusyError) as excinfo:
            with setup_root(deployment, cfg):
                pass
    assert excinfo.value.exit_code == EXIT_CODE_BUSY


# --------------------------------------------------------------------------------------------- #
# Section 2.2 -- the provisional profile
# --------------------------------------------------------------------------------------------- #

def test_provisional_profile_is_written_handed_to_the_child_and_then_removed(
    cfg: Config, hw_path: Path, monkeypatch: pytest.MonkeyPatch, v2: None
) -> None:
    """Section 2.2: the child must find a profile, or it runs a second sweep inside the
    measurement it is supposed to be taking."""
    stub_probes(monkeypatch)
    seen: dict[str, object] = {}

    def fake_calibrate(cfg_path, **kwargs):
        provisional = Path(kwargs["hardware_profile"])
        seen["path"] = provisional
        seen["exists_during_the_run"] = provisional.exists()
        seen["profile"] = yaml.safe_load(provisional.read_text())
        return {}

    monkeypatch.setattr(cal, "calibrate_gpu_stages", fake_calibrate)
    hw.run_setup(cfg, optimize=False, calibrate=True, allow_calibration_fallback=True)

    assert seen["exists_during_the_run"] is True
    assert seen["profile"]["provider"] == "cpu"                     # a real, bindable profile
    assert "plant_detector" in seen["profile"]["stages"]
    # Provisional, so it must not survive the run and be mistaken for the tuned profile.
    assert not Path(seen["path"]).exists()
    assert Path(seen["path"]).name.endswith(hw.PROVISIONAL_SUFFIX)
    assert hw_path.exists()


def test_no_provisional_profile_is_written_with_the_flag_off(
    cfg: Config, hw_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_probes(monkeypatch)
    seen: dict[str, object] = {}
    monkeypatch.setattr(cal, "calibrate_gpu_stages",
                        lambda cfg_path, **kw: seen.setdefault("hardware_profile",
                                                               kw["hardware_profile"]) or {})
    hw.run_setup(cfg, optimize=False, calibrate=True)
    assert seen["hardware_profile"] is None


def test_a_pointed_at_profile_makes_the_childs_auto_setup_a_no_op(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the fix: ``LM3_HARDWARE`` naming an existing profile is exactly what
    ``ensure_hardware_profile`` checks, so the child's auto-setup never triggers."""
    provisional = tmp_path / "hardware_settings.pytest.yaml.provisional"
    provisional.parent.mkdir(parents=True, exist_ok=True)
    provisional.write_text(yaml.safe_dump({"stages": {}}), encoding="utf-8")
    monkeypatch.setenv(paths.ENV_HARDWARE, str(provisional))

    def _explode(*args, **kwargs):
        raise AssertionError("the calibration child must not run a second setup sweep")

    monkeypatch.setattr(hw, "run_setup", _explode)
    monkeypatch.setattr(hw, "_fingerprint", lambda c: None)
    assert hw.ensure_hardware_profile(cfg) == provisional


def test_discard_provisional_never_deletes_the_real_profile(tmp_path: Path) -> None:
    real = tmp_path / "hardware_settings.yaml"
    real.write_text("stages: {}\n", encoding="utf-8")
    hw._discard_provisional(real)
    assert real.exists()


# --------------------------------------------------------------------------------------------- #
# Section 2.2 -- calibration is an inherited subactivity
# --------------------------------------------------------------------------------------------- #

def launch_child(monkeypatch: pytest.MonkeyPatch, activity, tmp_path: Path, *,
                 hardware_profile: Path | None = None, proc: FakeProc | None = None) -> dict:
    """Drive ``calibrate._run_pipeline`` with a fake child and capture how it was launched."""
    captured: dict = {}

    def fake_popen(cmd, env=None, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["kwargs"] = kwargs
        return proc or FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    cal._run_pipeline(tmp_path / "cal.yaml", tmp_path / "in", tmp_path / "out", 30.0, None,
                      activity=activity, hardware_profile=hardware_profile)
    return captured


def test_the_calibration_child_inherits_the_lease_and_the_capability(
    cfg: Config, deployment: Path, hw_path: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, v2: None
) -> None:
    """Section 2.2: the child runs under the parent's lease. It never acquires one, because the
    parent is mid-``run_setup`` and holds the deployment."""
    provisional = tmp_path / "prov.yaml"
    provisional.write_text("stages: {}\n", encoding="utf-8")
    with setup_root(deployment, cfg) as activity:
        captured = launch_child(monkeypatch, activity, tmp_path, hardware_profile=provisional)
        env = captured["env"]
        assert env[ENV_LEASE_CAPABILITY]                                  # the one-use grant
        assert env[ex.ENV_CHILD_ACTIVITY] == Activity.CALIBRATION_PIPELINE.value
        assert env[ex.ENV_PARENT_RUN_ID] == activity.run_id
        grant = deployment / "children" / f"{env[ex.ENV_CHILD_RUN_ID]}.grant.json"
        assert grant.exists()
        # The fake child has already "exited", so the root has promoted it out of current_child.
        assert active_json(deployment)["last_child"]["run_name"] == CALIBRATION_RUN_NAME

    # pass_fds is EXHAUSTIVE: whatever is not in it is closed in the child, so the composer had to
    # build it. A hand-rolled Popen keyword here would have dropped the lease descriptor silently.
    if os.name != "nt":
        assert captured["kwargs"]["pass_fds"]
        assert captured["kwargs"]["close_fds"] is True


def test_the_child_env_carries_the_provisional_profile(
    cfg: Config, deployment: Path, hw_path: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, v2: None
) -> None:
    provisional = tmp_path / "prov.yaml"
    provisional.write_text("stages: {}\n", encoding="utf-8")
    with setup_root(deployment, cfg) as activity:
        captured = launch_child(monkeypatch, activity, tmp_path, hardware_profile=provisional)
    assert captured["env"][paths.ENV_HARDWARE] == str(provisional.resolve())


def test_a_preexisting_LM3_HARDWARE_does_not_collide_with_the_provisional(
    cfg: Config, deployment: Path, hw_path: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, v2: None
) -> None:
    """``compose_launch`` REFUSES to pick a winner between two values for one variable, so a
    developer with ``LM3_HARDWARE`` already exported would otherwise fail every calibration."""
    # ``hw_path`` already exported LM3_HARDWARE; the legacy alias must name the SAME file, or
    # ``paths`` refuses the split brain before we ever reach the composer.
    monkeypatch.setenv("LM3_HARDWARE_SETTINGS", str(hw_path))
    provisional = tmp_path / "prov.yaml"
    provisional.write_text("stages: {}\n", encoding="utf-8")
    with setup_root(deployment, cfg) as activity:
        captured = launch_child(monkeypatch, activity, tmp_path, hardware_profile=provisional)
    assert captured["env"][paths.ENV_HARDWARE] == str(provisional.resolve())
    # The legacy alias outranks nothing: paths honors it, so leaving it set would silently win.
    assert "LM3_HARDWARE_SETTINGS" not in captured["env"]


def test_the_child_never_inherits_a_stale_status_descriptor(
    cfg: Config, deployment: Path, hw_path: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, v2: None
) -> None:
    """``dict(os.environ)`` used to hand the child a descriptor NUMBER that means something else
    in the child, where the low numbers belong to somebody else."""
    monkeypatch.setenv(ENV_STATUS_FD, "9")
    with setup_root(deployment, cfg) as activity:
        captured = launch_child(monkeypatch, activity, tmp_path)
    assert ENV_STATUS_FD not in captured["env"]


def test_the_root_records_the_finished_child_as_last_child(
    cfg: Config, deployment: Path, hw_path: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, v2: None
) -> None:
    with setup_root(deployment, cfg) as activity:
        launch_child(monkeypatch, activity, tmp_path)
        record = active_json(deployment)
    assert record["last_child"]["run_name"] == CALIBRATION_RUN_NAME
    assert record["last_child"]["state"] == "done"
    assert record.get("current_child") is None


def test_flag_off_launch_is_the_old_whole_environment_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the flag off nothing about the launch may change: same ``subprocess.run``, same env."""
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("must not use Popen"))
    cal._run_pipeline(tmp_path / "cal.yaml", tmp_path / "in", tmp_path / "out", 30.0, None,
                      activity=None, hardware_profile=None)
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    assert ENV_LEASE_CAPABILITY not in captured["env"]
    assert "pass_fds" not in captured


# --------------------------------------------------------------------------------------------- #
# Section 2.2 -- requested calibration failure is loud
# --------------------------------------------------------------------------------------------- #

def test_a_nonzero_child_exit_raises_with_the_stderr_tail(
    cfg: Config, deployment: Path, hw_path: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, v2: None
) -> None:
    with setup_root(deployment, cfg) as activity:
        with pytest.raises(cal.CalibrationError) as excinfo:
            launch_child(monkeypatch, activity, tmp_path,
                         proc=FakeProc(returncode=2, stderr="ImportError: no ect"))
    assert "exited 2" in str(excinfo.value)
    assert "ImportError" in str(excinfo.value)


def test_an_exit_75_child_is_reported_as_a_lost_lease_handoff(
    cfg: Config, deployment: Path, hw_path: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, v2: None
) -> None:
    """An opaque ``exited 75`` would send the reader hunting for a config bug."""
    with setup_root(deployment, cfg) as activity:
        with pytest.raises(cal.CalibrationError) as excinfo:
            launch_child(monkeypatch, activity, tmp_path, proc=FakeProc(returncode=EXIT_CODE_BUSY))
    assert "did not inherit" in str(excinfo.value)


def test_a_timed_out_child_is_killed_rather_than_left_holding_the_lease(
    cfg: Config, deployment: Path, hw_path: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, v2: None
) -> None:
    """The plan does not name the timeout path, but it fails the same way: while the child lives it
    holds the inherited lease reference and the deployment stays occupied."""
    proc = FakeProc(hang=True)
    with setup_root(deployment, cfg) as activity:
        with pytest.raises(cal.CalibrationError) as excinfo:
            launch_child(monkeypatch, activity, tmp_path, proc=proc)
    assert "exceeded" in str(excinfo.value)
    assert proc.killed is True


def test_calibrate_gpu_stages_is_quiet_without_strict_and_loud_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = write_settings(tmp_path)

    def boom(*args, **kwargs):
        raise cal.CalibrationError("calibration run exited 1")

    monkeypatch.setattr(cal, "_run_pipeline", boom)
    monkeypatch.setattr(cal, "_stage_images", lambda src, dst, n: [dst / "a.jpg"])
    monkeypatch.setattr(cal, "default_image_dir", lambda: tmp_path)

    assert cal.calibrate_gpu_stages(cfg_path) == {}
    with pytest.raises(cal.CalibrationError):
        cal.calibrate_gpu_stages(cfg_path, strict=True)


def test_a_missing_image_resource_is_loud_under_strict_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every path that returns nothing has to answer to ``strict``, not just the child exit: a
    packaging fault the user asked to calibrate through is still a calibration they did not get."""
    def absent() -> Path:
        raise paths.PackagedResourceError("calibration images did not ship")

    monkeypatch.setattr(cal, "default_image_dir", absent)
    cfg_path = write_settings(tmp_path)
    assert cal.calibrate_gpu_stages(cfg_path) == {}
    with pytest.raises(cal.CalibrationError):
        cal.calibrate_gpu_stages(cfg_path, strict=True)


def test_run_setup_still_writes_the_profile_when_calibration_fails(
    cfg: Config, hw_path: Path, monkeypatch: pytest.MonkeyPatch, v2: None
) -> None:
    """The sweep is good work: a failed measurement must not also cost the tuning that succeeded."""
    stub_probes(monkeypatch)

    def boom(*args, **kwargs):
        raise cal.CalibrationError("calibration run exited 1")

    monkeypatch.setattr(cal, "calibrate_gpu_stages", boom)
    with pytest.raises(cal.CalibrationError):
        hw.run_setup(cfg, optimize=False, calibrate=True)
    assert hw_path.exists()
    assert yaml.safe_load(hw_path.read_text())["stages"]["plant_detector"]["vram_measured"] is False


def test_calibration_failure_is_not_loud_with_the_flag_off(
    cfg: Config, hw_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loudness rides ``LM3_RUNTIME_V2`` with everything else in Step 3: off, the pre-Step-3
    behavior stands, warning and heuristic estimates included."""
    stub_probes(monkeypatch)
    seen: dict[str, object] = {}
    monkeypatch.setattr(cal, "calibrate_gpu_stages",
                        lambda cfg_path, **kw: seen.setdefault("strict", kw["strict"]) or {})
    hw.run_setup(cfg, optimize=False, calibrate=True)
    assert seen["strict"] is False
    assert hw_path.exists()


def test_opting_into_fallback_keeps_calibration_failure_quiet(
    cfg: Config, hw_path: Path, monkeypatch: pytest.MonkeyPatch, v2: None
) -> None:
    stub_probes(monkeypatch)
    seen: dict[str, object] = {}
    monkeypatch.setattr(cal, "calibrate_gpu_stages",
                        lambda cfg_path, **kw: seen.setdefault("strict", kw["strict"]) or {})
    hw.run_setup(cfg, optimize=False, calibrate=True, allow_calibration_fallback=True)
    assert seen["strict"] is False
    assert hw_path.exists()


def test_unrequested_calibration_is_never_strict(
    cfg: Config, hw_path: Path, monkeypatch: pytest.MonkeyPatch, v2: None
) -> None:
    """Silent fallback stays acceptable when calibration was not requested (section 2.2)."""
    stub_probes(monkeypatch)
    monkeypatch.setattr(cal, "calibrate_gpu_stages",
                        lambda *a, **k: pytest.fail("must not calibrate"))
    hw.run_setup(cfg, optimize=False, calibrate=False)
    assert hw_path.exists()


def test_a_config_with_no_source_path_is_loud_when_calibration_was_requested(
    hw_path: Path, monkeypatch: pytest.MonkeyPatch, v2: None
) -> None:
    stages: dict[str, dict] = {"plant_detector": {"workers_per_gpu": 1}}
    with pytest.raises(cal.CalibrationError):
        hw._apply_vram_measurements(stages, object(), [], calibrate=True, on_progress=None,
                                    profile_path=hw_path, strict=True)


# --------------------------------------------------------------------------------------------- #
# Section 3.5 -- where the derived calibration config lands
# --------------------------------------------------------------------------------------------- #

def test_the_derived_config_is_written_beside_the_source_under_v2(
    tmp_path: Path, v2: None
) -> None:
    """Section 3.5, rule 1: ``Config.resolve_path`` resolves a relative path against the SETTINGS
    FILE's directory, so deriving into a tempdir re-bases every ``../models/...`` in the user's
    config and the child fails for a reason that has nothing to do with the machine."""
    src = write_settings(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    derived = cal._write_calibration_config(src, scratch, gpu_index=0)
    assert derived.parent == src.parent
    assert derived.name == cal.DERIVED_CONFIG_NAME
    data = yaml.safe_load(derived.read_text())
    assert data["project"]["run_name"] == CALIBRATION_RUN_NAME
    assert data["compute"]["vram"]["max_workers_per_gpu"] == 1
    assert data["compute"]["devices"] == [0]
    cal._discard(derived)
    assert not derived.exists()


def test_the_derived_config_stays_in_the_scratch_dir_with_the_flag_off(tmp_path: Path) -> None:
    src = write_settings(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    derived = cal._write_calibration_config(src, scratch, gpu_index=None)
    assert derived.parent == scratch


def test_discard_never_removes_the_users_settings_file(tmp_path: Path) -> None:
    src = write_settings(tmp_path)
    cal._discard(src)
    assert src.exists()


# --------------------------------------------------------------------------------------------- #
# The lm3-setup CLI's exit codes
# --------------------------------------------------------------------------------------------- #

def test_main_returns_zero_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_settings(tmp_path)
    monkeypatch.setattr(hw, "run_setup", lambda cfg, **kwargs: tmp_path / "profile.yaml")
    events = tmp_path / "events.jsonl"
    assert hw.main(["--config", str(path), "--quick", "--events", str(events)]) == 0
    assert json.loads(events.read_text().splitlines()[-1])["phase"] == "done"


def test_main_returns_a_usage_code_for_a_missing_config(tmp_path: Path) -> None:
    assert hw.main(["--config", str(tmp_path / "nope.yaml")]) == hw.EXIT_CODE_CONFIG


def test_main_returns_nonzero_when_requested_calibration_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 2.2: a calibration failure the user asked for must make the CLI exit nonzero."""
    path = write_settings(tmp_path)

    def boom(cfg, **kwargs):
        raise cal.CalibrationError("calibration run exited 1")

    monkeypatch.setattr(hw, "run_setup", boom)
    events = tmp_path / "events.jsonl"
    code = hw.main(["--config", str(path), "--calibrate", "--events", str(events)])
    assert code == hw.EXIT_CODE_SETUP_FAILED
    assert json.loads(events.read_text().splitlines()[-1])["phase"] == "error"


def test_main_exits_75_when_another_root_holds_the_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 2.3: 75 means "retry", not "this is broken"."""
    path = write_settings(tmp_path)

    def busy(cfg, **kwargs):
        raise RuntimeBusyError(KEY)

    monkeypatch.setattr(hw, "run_setup", busy)
    assert hw.main(["--config", str(path), "--quick"]) == EXIT_CODE_BUSY
