"""``lm3 doctor`` (leafmachine3/doctor.py): every check's pass and failure path, driven through a
hand-built :class:`Env` so no test needs a broken real install, a GPU, or a particular driver.

The live failure modes were also exercised against real environments built from uv.lock (2026-10-06):
GPUs hidden -> check 5; loader-path step disabled -> check 6 naming libcublasLt.so.12; CPU onnxruntime
pip-installed over onnxruntime-gpu -> check 2, and the printed fix restored READY.
"""
from __future__ import annotations

import json
import sys

import pytest

from leafmachine3 import doctor
from leafmachine3.doctor import FAIL, OK, WARN, Env

CONTRACT = {
    "lm3_version": "3.0.0",
    "python": "3.11.17",
    "uv": "0.12.23",
    "cuda_min_driver": {"linux": "525.60.13", "win32": "528.33"},
    "dev_only": ["torch", "torchvision", "ultralytics", "triton"],
    "extras": {
        "gpu": [
            {"name": "numpy", "version": "2.4.4", "marker": ""},
            {"name": "onnxruntime-gpu", "version": "1.20.2",
             "marker": "(platform_machine == 'x86_64' and sys_platform == 'linux') or "
                       "(platform_machine == 'AMD64' and sys_platform == 'win32')"},
            {"name": "colorama", "version": "0.4.6", "marker": "sys_platform == 'win32'"},
        ],
        "cpu": [
            {"name": "numpy", "version": "2.4.4", "marker": ""},
            {"name": "onnxruntime", "version": "1.20.1", "marker": ""},
        ],
        "macos": [
            {"name": "numpy", "version": "2.4.4", "marker": ""},
            {"name": "onnxruntime", "version": "1.20.1", "marker": "sys_platform == 'darwin'"},
        ],
    },
}
UV_BASE = "/home/u/.local/share/uv/python/cpython-3.11.17-linux-x86_64-gnu"


def make_env(**over) -> Env:
    base = dict(python_version="3.11.17", executable="/x/.venv/bin/python", prefix="/x/.venv",
                base_prefix=UV_BASE, sys_platform="linux", machine="x86_64",
                installed={"numpy": "2.4.4", "onnxruntime-gpu": "1.20.2"}, environ={}, contract=CONTRACT)
    base.update(over)
    return Env(**base)


GOOD_NVML = lambda: ("560.35.05", ["NVIDIA RTX 6000 Ada Generation"])  # noqa: E731
GOOD_PROBE = lambda p, e: {"requested": p, "bound": p, "max_abs_diff": 1e-7, "error": None}  # noqa: E731


def run(env, **kw):
    kw.setdefault("nvml", GOOD_NVML)
    kw.setdefault("probe", GOOD_PROBE)
    return doctor.diagnose(env, **kw)


def statuses(rep):
    return {c.num: c.status for c in rep.checks}


# -------------------------------------------------------------------------------------------------- happy paths

def test_a_correct_gpu_install_is_ready():
    rep = run(make_env())
    assert rep.ready and rep.variant == "gpu"
    assert statuses(rep) == {1: OK, 2: OK, 3: OK, 4: OK, 5: OK, 6: OK, 8: OK}


def test_cpu_install_skips_the_driver_check_and_probes_the_cpu_provider():
    seen = []
    rep = run(make_env(installed={"numpy": "2.4.4", "onnxruntime": "1.20.1"}),
              probe=lambda p, e: seen.append(p) or GOOD_PROBE(p, e))
    assert rep.ready and rep.variant == "cpu" and 5 not in statuses(rep)
    assert seen == ["CPUExecutionProvider"]


def test_macos_install_probes_coreml():
    seen = []
    env = make_env(sys_platform="darwin", machine="arm64", installed={"numpy": "2.4.4", "onnxruntime": "1.20.1"})
    rep = run(env, probe=lambda p, e: seen.append(p) or GOOD_PROBE(p, e))
    assert rep.ready and rep.variant == "macos" and seen == ["CoreMLExecutionProvider"]


def test_quick_runs_only_checks_1_to_4_and_never_touches_gpu_or_subprocess():
    boom = lambda *a: pytest.fail("quick must not probe")  # noqa: E731
    rep = doctor.diagnose(make_env(), quick=True, nvml=boom, probe=boom)
    assert sorted(statuses(rep)) == [1, 2, 3, 4] and rep.ready


# -------------------------------------------------------------------------------------------------- check 1

def test_wrong_minor_python_fails():
    rep = run(make_env(python_version="3.12.4"))
    assert statuses(rep)[1] == FAIL and "requires Python 3.11" in rep.checks[0].detail
    assert len(rep.checks) == 1                              # nothing after a failed interpreter check runs


def test_system_python_outside_a_venv_fails():
    rep = run(make_env(prefix=UV_BASE, base_prefix=UV_BASE))
    assert statuses(rep)[1] == FAIL and "not inside a virtual environment" in rep.checks[0].detail


def test_patch_mismatch_or_non_uv_interpreter_warns_but_stays_ready():
    rep = run(make_env(python_version="3.11.5", base_prefix="/home/u/miniconda3"))
    c = rep.checks[0]
    assert c.status == WARN and "3.11.17" in c.detail and "not installed by uv" in c.detail
    assert rep.ready


# -------------------------------------------------------------------------------------------------- check 2

def test_both_onnxruntime_builds_installed_is_the_2026_10_06_incident():
    env = make_env(installed={"numpy": "2.4.4", "onnxruntime-gpu": "1.20.2", "onnxruntime": "1.30.0"})
    rep = run(env)
    c = rep.checks[1]
    assert c.status == FAIL and "both onnxruntime 1.30.0 and onnxruntime-gpu 1.20.2" in c.detail
    assert c.fix.startswith("uv sync --frozen --extra gpu --reinstall-package onnxruntime-gpu")
    assert not rep.ready


def test_no_onnxruntime_at_all_fails():
    rep = run(make_env(installed={"numpy": "2.4.4"}))
    assert rep.checks[1].status == FAIL and "no onnxruntime" in rep.checks[1].detail


def test_lm3_extra_mismatch_fails():
    rep = run(make_env(environ={"LM3_EXTRA": "cpu"}))
    assert rep.checks[1].status == FAIL and "LM3_EXTRA=cpu" in rep.checks[1].detail


# -------------------------------------------------------------------------------------------------- check 3

def test_a_drifted_package_fails_the_contract_with_the_sync_command():
    rep = run(make_env(installed={"numpy": "2.5.0", "onnxruntime-gpu": "1.20.2"}))
    c = rep.checks[2]
    assert c.status == FAIL and "numpy is 2.5.0, expected 2.4.4" in c.detail
    assert c.fix.endswith("uv sync --frozen --extra gpu")


def test_markers_exclude_packages_for_other_platforms():
    # colorama is win32-only in the contract; a Linux install without it is correct
    rep = run(make_env())
    assert rep.checks[2].status == OK and rep.checks[2].detail.startswith("2 packages")
    win = make_env(sys_platform="win32", machine="AMD64")
    assert run(win).checks[2].status == FAIL                       # ...and required on Windows
    win.installed["colorama"] = "0.4.6"
    assert run(win).checks[2].status == OK


# -------------------------------------------------------------------------------------------------- check 4

@pytest.mark.parametrize("plat,machine", [("linux", "aarch64"), ("darwin", "x86_64")])
def test_unsupported_platforms_fail(plat, machine):
    installed = {"numpy": "2.4.4", "onnxruntime": "1.20.1"}
    rep = run(make_env(sys_platform=plat, machine=machine, installed=installed))
    c = next(c for c in rep.checks if c.num == 4)
    assert c.status == FAIL and f"{plat} {machine} is not supported" in c.detail


# -------------------------------------------------------------------------------------------------- check 5

@pytest.mark.parametrize("cvd", ["", "-1"])
def test_hidden_gpus_fail_before_nvml_is_asked(cvd):
    rep = run(make_env(environ={"CUDA_VISIBLE_DEVICES": cvd}), nvml=lambda: pytest.fail("must not reach NVML"))
    c = next(c for c in rep.checks if c.num == 5)
    assert c.status == FAIL and "hides every GPU" in c.detail


def test_no_driver_fails_and_offers_the_cpu_variant():
    def nvml():
        raise OSError("libnvidia-ml.so.1: cannot open shared object file")
    c = next(c for c in run(make_env(), nvml=nvml).checks if c.num == 5)
    assert c.status == FAIL and "no NVIDIA driver found" in c.detail and "--extra cpu" in c.fix


def test_old_driver_fails_with_the_minimum():
    c = next(c for c in run(make_env(), nvml=lambda: ("470.256.02", ["Tesla V100"])).checks if c.num == 5)
    assert c.status == FAIL and "470.256.02 is below 525.60.13" in c.detail


def test_windows_uses_the_windows_minimum():
    env = make_env(sys_platform="win32", machine="AMD64",
                   installed={"numpy": "2.4.4", "onnxruntime-gpu": "1.20.2", "colorama": "0.4.6"})
    assert next(c for c in run(env, nvml=lambda: ("528.33", ["RTX"])).checks if c.num == 5).status == OK
    assert next(c for c in run(env, nvml=lambda: ("527.99", ["RTX"])).checks if c.num == 5).status == FAIL


def test_driver_versions_compare_numerically_not_as_strings():
    assert doctor._version_tuple("1000.1") > doctor._version_tuple("525.60.13")
    assert doctor._version_tuple("525.105.17") > doctor._version_tuple("525.60.13")


# -------------------------------------------------------------------------------------------------- check 6

def test_silent_cpu_fallback_fails_and_names_the_library():
    probe = lambda p, e: {"requested": p, "bound": "CPUExecutionProvider", "max_abs_diff": 0.0, "error": None,  # noqa: E731
                          "load_error": "libonnxruntime_providers_cuda.so: libcublasLt.so.12: cannot open shared object file"}
    c = next(c for c in run(make_env(), probe=probe).checks if c.num == 6)
    assert c.status == FAIL and "every model would run on the CPU" in c.detail and "libcublasLt.so.12" in c.detail


def test_libpath_flag_in_the_environment_is_called_out():
    probe = lambda p, e: {"requested": p, "bound": "CPUExecutionProvider", "max_abs_diff": 0.0, "error": None}  # noqa: E731
    c = next(c for c in run(make_env(environ={"LM3_CUDA_LIBPATH_SET": "1"}), probe=probe).checks if c.num == 6)
    assert c.status == FAIL and "LM3_CUDA_LIBPATH_SET is set" in c.detail


def test_wrong_numbers_from_the_accelerator_fail():
    probe = lambda p, e: {"requested": p, "bound": p, "max_abs_diff": 0.5, "error": None}  # noqa: E731
    assert next(c for c in run(make_env(), probe=probe).checks if c.num == 6).status == FAIL


def test_probe_error_fails():
    probe = lambda p, e: {"requested": p, "bound": None, "error": "RuntimeError: CUDA failure 100"}  # noqa: E731
    c = next(c for c in run(make_env(), probe=probe).checks if c.num == 6)
    assert c.status == FAIL and "CUDA failure 100" in c.detail


def test_the_real_probe_runs_on_the_cpu_provider_in_a_child_process():
    """No mocks: the shipped probe model, in a child, through the real subprocess plumbing."""
    pytest.importorskip("onnxruntime")
    res = doctor.run_probe_subprocess("CPUExecutionProvider", {**__import__("os").environ})
    assert res["error"] is None and res["bound"] == "CPUExecutionProvider" and res["max_abs_diff"] == 0.0


def test_probe_model_decodes_and_has_the_documented_shape():
    ort = pytest.importorskip("onnxruntime")
    from leafmachine3 import doctor_probe

    data = doctor_probe.probe_model_bytes()
    assert len(data) == 1078
    sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    y = sess.run(None, {"x": doctor_probe.probe_input()})[0]
    assert y.shape == (1, 8, 32, 32) and (y >= 0).all() and y.max() > 0


# -------------------------------------------------------------------------------------------------- ordering, dev env, gate, CLI

def test_checks_stop_at_the_first_failure():
    rep = run(make_env(installed={"numpy": "2.5.0", "onnxruntime-gpu": "1.20.2"}),
              nvml=lambda: pytest.fail("must stop before check 5"))
    assert [c.status for c in rep.checks] == [OK, OK, FAIL]
    assert rep.first_failure.num == 3


def test_development_environment_is_a_note_unless_production():
    env = make_env(installed={"numpy": "2.4.4", "onnxruntime-gpu": "1.20.2", "torch": "2.6.0+cu124",
                              "ultralytics": "8.4.107"})
    rep = run(env)
    assert rep.ready and "torch 2.6.0+cu124" in rep.notes[0] and "ultralytics 8.4.107" in rep.notes[0]
    prod = run(env, production=True)
    assert not prod.ready and prod.first_failure.name == "production"
    assert "fix the production check first" in doctor.render(prod, env)


def test_startup_gate_raises_with_the_failed_check_and_its_fix():
    env = make_env(installed={"numpy": "2.4.4", "onnxruntime-gpu": "1.20.2", "onnxruntime": "1.30.0"})
    with pytest.raises(doctor.DoctorError) as ei:
        doctor.startup_gate(env)
    msg = str(ei.value)
    assert "check 2 (hardware variant)" in msg and "--reinstall-package onnxruntime-gpu" in msg
    assert doctor.startup_gate(make_env()).ready


def test_json_report_round_trips(capsys, monkeypatch):
    real = doctor.diagnose
    monkeypatch.setattr(doctor.Env, "current", classmethod(lambda cls: make_env()))
    monkeypatch.setattr(doctor, "diagnose", lambda env, **kw: real(env, nvml=GOOD_NVML, probe=GOOD_PROBE, **kw))
    assert doctor.main(["--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ready"] is True and out["variant"] == "gpu" and out["checks"][0]["name"] == "interpreter"


def test_lm3_cli_dispatches_doctor():
    from leafmachine3 import cli

    assert "doctor" in cli._COMMANDS


def test_shipped_contract_loads_and_covers_every_variant():
    c = doctor.load_contract()
    assert set(c["extras"]) == {"gpu", "cpu", "macos"}
    assert c["python"].startswith("3.11.")
    names = {r["name"] for r in c["extras"]["gpu"]}
    assert "onnxruntime-gpu" in names and "nvidia-cudnn-cu12" in names
    assert not names & set(c["dev_only"]), "the production contract must not contain dev-only packages"


@pytest.mark.skipif(sys.platform != "linux", reason="the CUDA loader-path step is Linux-only")
def test_probe_main_routes_cuda_through_machine3s_loader_step(monkeypatch):
    calls = []
    import leafmachine3.machine3 as m3
    from leafmachine3 import doctor_probe

    monkeypatch.setattr(m3, "_exec_with_cuda_libpath", lambda: calls.append(list(sys.argv)))
    monkeypatch.setattr(doctor_probe, "run_probe", lambda p: {"requested": p})
    monkeypatch.setattr(sys, "argv", ["/abs/leafmachine3/doctor_probe.py", "CUDAExecutionProvider"])
    doctor_probe.main()
    # the re-exec must relaunch the MODULE, not the file as a bare script
    assert calls == [["-m", "leafmachine3.doctor_probe", "CUDAExecutionProvider"]]
