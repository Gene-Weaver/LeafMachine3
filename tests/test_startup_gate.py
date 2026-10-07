"""The startup gate: `lm3 doctor` checks 1-4 before machine3 / lm3 serve start, and /healthz/doctor.

The suite pins LM3_STARTUP_GATE=0 (tests/conftest.py) so pipeline tests do not depend on which
environment runs them; every test here turns it back on.
"""
from __future__ import annotations

import pytest

from leafmachine3 import doctor
from leafmachine3.doctor import FAIL, OK, WARN, Check, DoctorError, Report


def _failing_report() -> Report:
    rep = Report(lm3_version="3.0.0", variant=None)
    rep.checks = [Check(1, "interpreter", OK, "Python 3.11.17"),
                  Check(2, "hardware variant", FAIL, "both onnxruntime 1.30.0 and onnxruntime-gpu 1.20.2 are installed",
                        "uv sync --frozen --extra gpu --reinstall-package onnxruntime-gpu")]
    return rep


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setenv(doctor.GATE_ENV, "1")


def _refuse(monkeypatch):
    def refuse(env=None):
        raise DoctorError(_failing_report())
    monkeypatch.setattr(doctor, "startup_gate", refuse)


def test_machine3_refuses_a_broken_environment_with_the_fix(gate_on, monkeypatch, capsys):
    import leafmachine3.machine3 as m3

    _refuse(monkeypatch)
    monkeypatch.setattr(m3, "_exec_with_cuda_libpath", lambda: None)
    monkeypatch.setattr(m3, "machine3", lambda *a, **k: pytest.fail("the pipeline must not start"))
    assert m3.main(["--config", "unused.yaml"]) == doctor.EXIT_CODE_ENVIRONMENT == 78
    err = capsys.readouterr().err
    assert "check 2 (hardware variant)" in err and "--reinstall-package onnxruntime-gpu" in err
    assert "lm3 doctor" in err


def test_lm3_serve_refuses_before_binding_a_port(gate_on, monkeypatch, capsys):
    from leafmachine3.server import app as app_mod

    _refuse(monkeypatch)
    monkeypatch.setattr(app_mod, "serve", lambda *a, **k: pytest.fail("the server must not start"))
    assert app_mod.main(["--port", "8799"]) == 78
    assert "check 2 (hardware variant)" in capsys.readouterr().err


def test_the_gate_can_be_bypassed_explicitly(monkeypatch):
    import leafmachine3.machine3 as m3

    monkeypatch.setenv(doctor.GATE_ENV, "0")
    _refuse(monkeypatch)
    ran = []
    monkeypatch.setattr(m3, "_exec_with_cuda_libpath", lambda: None)
    monkeypatch.setattr(m3, "machine3", lambda *a, **k: ran.append(True))
    assert m3.main(["--config", "unused.yaml"]) == 0 and ran == [True]


def test_warnings_are_printed_and_do_not_stop_anything(gate_on, monkeypatch, capsys):
    rep = Report(lm3_version="3.0.0", variant="gpu",
                 checks=[Check(1, "interpreter", WARN, "Python 3.11.5; the interpreter was not installed by uv")])
    monkeypatch.setattr(doctor, "startup_gate", lambda env=None: rep)
    assert doctor.run_startup_gate("machine3") is None
    assert "warning: interpreter: Python 3.11.5" in capsys.readouterr().err


def test_a_broken_gate_never_blocks_lm3(gate_on, monkeypatch, capsys):
    def boom(env=None):
        raise RuntimeError("contract file unreadable")
    monkeypatch.setattr(doctor, "startup_gate", boom)
    assert doctor.run_startup_gate("machine3") is None
    assert "could not run (RuntimeError: contract file unreadable)" in capsys.readouterr().err


def test_healthz_doctor_requires_the_token_and_returns_the_report(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from leafmachine3.server.app import JobManager, create_app

    monkeypatch.setenv("LM3_SERVER_TOKEN", "t0ken-for-this-test")
    monkeypatch.setattr(doctor, "diagnose", lambda env=None, quick=True, **kw: _failing_report())
    client = TestClient(create_app(JobManager(tmp_path / "jobs")))
    assert client.get("/healthz/doctor").status_code == 401
    r = client.get("/healthz/doctor", headers={"Authorization": "Bearer t0ken-for-this-test"})
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is False and body["checks"][1]["name"] == "hardware variant"
