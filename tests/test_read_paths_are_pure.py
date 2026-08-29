"""Reading must not write, and polling must not re-probe.

Two real defects motivate this file. Resolving the hardware-profile path used to perform the
one-release legacy adopt as a side effect, so a READ endpoint copied a file -- that is how two
legacy profiles were written into a real developer's ``~/.config/lm3`` during a test run. And the
machine key behind that path initializes, enumerates and shuts down NVML, while the status stream
resolves it at 2 Hz.

The rule these tests pin: path RESOLUTION is pure and memoized; MIGRATION is explicit, once, at a
controlled startup path; and a deliberate settings change is still noticed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml
from fastapi.testclient import TestClient

from leafmachine3.core import paths
from tests._contract_helpers import (
    Sandbox, bearer, isolate_server_paths, reset_server_module_state,
)


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Sandbox]:
    reset_server_module_state()
    box = isolate_server_paths(tmp_path, monkeypatch)
    try:
        yield box
    finally:
        reset_server_module_state()


@pytest.fixture
def client(sandbox: Sandbox) -> Iterator[TestClient]:
    from leafmachine3.server.app import JobManager, create_app
    yield TestClient(create_app(JobManager(sandbox.jobs_root / "managed")))


def tree_state(root: Path) -> dict[str, tuple[int, float]]:
    """Every file under ``root`` with its size and mtime -- a change-detector, not a listing."""
    if not root.exists():
        return {}
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


READ_ENDPOINTS = ["/healthz", "/v1/status", "/v1/runs", "/v1/run/active", "/v1/settings"]


def test_repeated_read_requests_create_and_modify_nothing(client: TestClient, sandbox: Sandbox) -> None:
    """The whole isolated world must be byte-identical after a burst of read traffic."""
    sandbox.write_settings()
    roots = [sandbox.root]
    for _ in range(2):                                   # warm every lazy path first
        for ep in READ_ENDPOINTS:
            client.get(ep, headers=bearer())

    before = [tree_state(r) for r in roots]
    for _ in range(20):                                  # ~10 seconds of 2 Hz polling
        for ep in READ_ENDPOINTS:
            client.get(ep, headers=bearer())
    after = [tree_state(r) for r in roots]

    for root, b, a in zip(roots, before, after):
        assert b == a, f"read traffic mutated {root}: {set(a.items()) ^ set(b.items())}"


def test_reads_never_adopt_a_legacy_hardware_profile(
    client: TestClient, sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact mechanism that leaked into ~/.config/lm3."""
    import os

    sandbox.write_settings()
    legacy = sandbox.settings_path.parent / paths.LEGACY_HARDWARE_FILENAME
    legacy.write_text(yaml.safe_dump({"marker": "legacy"}), encoding="utf-8")
    monkeypatch.delenv(paths.ENV_HARDWARE, raising=False)
    monkeypatch.delenv("LM3_HARDWARE_SETTINGS", raising=False)

    from leafmachine3.server import app
    app.reset_path_caches()
    target = paths.hardware_profile_path(env=dict(os.environ))
    assert not target.exists()

    for _ in range(10):
        for ep in READ_ENDPOINTS:
            client.get(ep, headers=bearer())

    assert not target.exists(), "a read endpoint performed the legacy adopt"
    assert legacy.is_file()


def test_the_machine_probe_runs_once_not_once_per_poll(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NVML init/enumerate/shutdown at 2 Hz forever is the cost this memo removes."""
    calls: list[int] = []
    real = paths.detect_machine_identity

    def counting(*a: Any, **k: Any):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(paths, "detect_machine_identity", counting)
    paths.reset_machine_key_cache()

    for _ in range(30):
        client.get("/healthz")
        client.get("/v1/status", headers=bearer())

    assert len(calls) <= 1, f"the machine identity was probed {len(calls)} times across 60 reads"


def test_the_hardware_path_memo_still_notices_a_deliberate_change(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cache that hides an intentional change is a bug, not an optimization."""
    from leafmachine3.server import app

    monkeypatch.delenv(paths.ENV_HARDWARE, raising=False)
    monkeypatch.delenv("LM3_HARDWARE_SETTINGS", raising=False)
    app.reset_path_caches()
    first = app.canonical_hardware_path()
    assert app.canonical_hardware_path() == first          # memoized

    override = tmp_path / "explicit_profile.yaml"
    monkeypatch.setenv(paths.ENV_HARDWARE, str(override))
    assert app.canonical_hardware_path() == override, "the memo masked an LM3_HARDWARE change"

    monkeypatch.delenv(paths.ENV_HARDWARE)
    assert app.canonical_hardware_path() == first


def test_a_deliberate_settings_change_is_still_visible_through_the_api(
    client: TestClient, sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rule: memoize what cannot change in-process; never memoize the selected settings file."""
    sandbox.write_settings()
    monkeypatch.delenv("LM3_SETTINGS_PATH", raising=False)
    first = client.get("/v1/settings", headers=bearer()).json().get("yaml_path")

    chosen = tmp_path / "chosen" / "LM3_settings.yaml"
    chosen.parent.mkdir(parents=True, exist_ok=True)
    chosen.write_text(yaml.safe_dump({"version": 3, "project": {"run_name": "chosen"}}), encoding="utf-8")
    monkeypatch.setenv(paths.ENV_SETTINGS, str(chosen))
    reset_server_module_state()

    second = client.get("/v1/settings", headers=bearer()).json().get("yaml_path")
    assert second != first
    assert Path(second) == chosen
