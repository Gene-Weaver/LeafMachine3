"""Server path resolution -- these were the PRE-refactor characterization tests, now updated.

They were written to pin the eight-row table in Appendix A ("Settings resolution split -- Step 1
must unify the fallbacks, not just the names") and were marked EXPECTED TO CHANGE. Step 1 has
landed, so every test below has been rewritten against the normative §3.1 precedence table whose
governing rule is "**No path falls back to the current working directory.**" Each one names the
clause that rewrote it.

What changed, in one sentence per module:

* ``settings_api`` obeyed ``LM3_SETTINGS_PATH`` and nothing else, falling back to ``$CWD`` --
  §3.1 row 1 gives it the same chain as everyone else, with ``LM3_SETTINGS_PATH`` honored for one
  release as a deprecated alias (§4 Step 1's migration bullet).
* ``metrics_api`` scanned ``[CWD, checkout root]`` -- §4 Step 1 "Unify fallbacks, not just names",
  so the scan is gone rather than renamed.
* ``postprocess_api`` and ``progress_api`` each had their own ``LM3_settings.yaml``-off-the-CWD
  fallback (and ``progress_api``'s was not even absolute) -- both now call the one resolver.
* The hardware profile is deployment-scoped and machine-keyed (§3.1 row 2), so the three
  disagreeing hardware readers became one.

The ``hardware_setup.HW_PATH`` and ``calibrate.DEFAULT_IMAGE_DIR`` cases that used to live here
have moved to ``tests/test_setup_paths.py``: both constants no longer exist (§3.1 "Resolver
scope"), and that module owns the setup-side half of the same change.

Every assertion is still DRIVEN -- the module is imported and its own resolver is called under a
monkeypatched CWD and environment -- because §4 Step 1's warning is precisely that "renaming
variables while leaving three fallbacks reachable from three CWDs preserves the bug". A
source-text assertion would survive that rename; a driven one will not.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest
import yaml

# The ``characterization`` marker is not registered in a pytest ini section (this repo has none),
# and Step 1 owns pyproject.toml, so the marker is silenced locally instead of registered
# globally. Narrowed to this one mark name so a genuine typo in another module still warns.
warnings.filterwarnings(
    "ignore",
    message=r".*Unknown pytest\.mark\.characterization.*",
    category=pytest.PytestUnknownMarkWarning,
)

pytestmark = pytest.mark.characterization

# Every environment variable that steers one of the Appendix A paths. Cleared before each test so
# a developer's shell (or another test's leftovers) cannot decide the answer. ``LM3_RUNTIME_DIR``
# and ``LM3_DEPLOYMENT_ID`` are deliberately NOT in this list: ``tests/conftest.py`` sets them at
# import time to keep the suite out of the real deployment, and clearing them here would undo it.
_PATH_ENV_VARS = (
    "LM3_SETTINGS_PATH",
    "LM3_SETTINGS",
    "LM3_SETTINGS_META",
    "LM3_HARDWARE_SETTINGS",
    "LM3_HARDWARE",
    "LM3_POSTPROCESS_SETTINGS",
    "LM3_SERVER_JOBS",
    "LM3_RUNS_ROOTS",
    "LM3_STATUS_ROOTS",
)


@pytest.fixture(autouse=True)
def _clean_path_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from leafmachine3.server import app as server_app

    for name in _PATH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # <user-config> / <user-state> are per-test here as well as per-session, so a legacy adopt or
    # a seeded settings file lands in tmp_path and cannot leak between tests.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    # The deprecation warnings are memoized per process; drop the memo so each test sees the
    # migration behave as it would on a fresh server.
    server_app._LEGACY_CACHE.clear()
    server_app._LEGACY_CONFLICTS.clear()


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Chdir into a scratch directory and hand back the CWD as the process reports it.

    ``os.getcwd()`` is fully resolved, while ``tmp_path`` may still contain a symlink (``/tmp``
    is a symlink on some hosts), so comparisons must use the former or they compare two
    spellings of the same directory.
    """
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    return Path(os.getcwd())


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# The settings split, one module at a time (was Appendix A rows 1-4)
# --------------------------------------------------------------------------- #
def test_settings_api_honors_the_canonical_variable(cwd: Path, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """§3.1 row 1 step 2: the Settings API reads ``LM3_SETTINGS``, like every other module.

    WAS: ``settings_api`` read ``LM3_SETTINGS_PATH`` only, so the canonical name of §3.1 was
    invisible to the GUI's settings editor -- the split itself, in one assertion.
    """
    from leafmachine3.server import settings_api

    chosen = _write_yaml(tmp_path / "chosen" / "LM3_settings.yaml", {"version": 3})
    monkeypatch.setenv("LM3_SETTINGS", str(chosen))

    assert settings_api.settings_path() == chosen


def test_settings_api_no_longer_falls_back_to_the_cwd(cwd: Path) -> None:
    """§3.1: "No path falls back to the current working directory."

    WAS: with no environment at all, ``settings_api`` answered ``<CWD>/LM3_settings.yaml`` -- so a
    file that happened to sit beside the launcher silently became the project.
    """
    from leafmachine3.server import settings_api

    decoy = _write_yaml(cwd / "LM3_settings.yaml", {"project": {"run_name": "from_the_cwd"}})

    assert settings_api.settings_path() != decoy


def test_metrics_api_no_longer_scans_the_cwd_or_the_checkout(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4 Step 1: "Unify fallbacks, not just names."

    WAS: ``_find_file`` scanned ``[CWD, checkout root]``, so an ``LM3_settings.yaml`` in the launch
    directory won, and under an installed wheel the second root was ``site-packages`` itself. The
    helpers are deleted rather than renamed, which is what makes the fallback unreachable.
    """
    from leafmachine3.server import metrics_api

    _write_yaml(cwd / metrics_api.CFG_FILENAME, {"version": 3})

    assert not hasattr(metrics_api, "_search_roots")
    assert not hasattr(metrics_api, "_find_file")
    assert not hasattr(metrics_api, "_repo_root")
    assert metrics_api.default_config_path() != cwd / metrics_api.CFG_FILENAME


def test_metrics_api_returns_none_when_the_canonical_variable_names_a_missing_file(
    cwd: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.1 row 1: an explicitly named settings file that does not exist is a hard error.

    ``start_run`` turns the ``None`` into a 400. WAS: the same ``None``, but for a different
    reason -- ``_find_file`` short-circuited on a set-but-missing override instead of falling
    through, a failure shape unique to this module. Now the refusal is the resolver's, and it is
    the same refusal every other module gets.
    """
    from leafmachine3.server import metrics_api

    _write_yaml(cwd / metrics_api.CFG_FILENAME, {"version": 3})
    monkeypatch.setenv("LM3_SETTINGS", str(tmp_path / "does-not-exist.yaml"))

    assert metrics_api.default_config_path() is None


def test_postprocess_api_reads_the_canonical_settings_file(cwd: Path, tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """§3.1 row 1. Observed through the parsed document -- a sentinel key proves WHICH file was read.

    WAS: ``LM3_SETTINGS`` else ``<CWD>/LM3_settings.yaml``; the decoy below would have won.
    """
    from leafmachine3.server import postprocess_api

    other = _write_yaml(tmp_path / "env" / "LM3_settings.yaml",
                        {"project": {"run_name": "from_the_env"}})
    _write_yaml(cwd / "LM3_settings.yaml", {"project": {"run_name": "from_the_cwd"}})
    monkeypatch.setenv("LM3_SETTINGS", str(other))

    assert postprocess_api._lm3_settings()["project"]["run_name"] == "from_the_env"


def test_postprocess_api_ignores_a_settings_file_in_the_cwd(cwd: Path) -> None:
    """§3.1: no CWD fallback. WAS: an empty environment made the CWD copy the whole config."""
    from leafmachine3.server import postprocess_api

    _write_yaml(cwd / "LM3_settings.yaml", {"project": {"run_name": "from_the_cwd"}})

    assert postprocess_api._lm3_settings().get("project", {}).get("run_name") != "from_the_cwd"


def test_progress_api_returns_an_absolute_canonical_settings_path(cwd: Path) -> None:
    """§3.1 row 1. WAS: the BARE relative name ``LM3_settings.yaml``.

    Unlike the other three, ``progress_api``'s fallback was never made absolute, so the path it
    handed to ``Path.open`` was re-interpreted against whatever the CWD happened to be at the
    moment of the read -- late binding, on every status frame.
    """
    from leafmachine3.server import progress_api

    resolved = progress_api._settings_path()

    assert resolved.is_absolute()
    assert resolved != Path("LM3_settings.yaml")
    assert resolved.resolve() != cwd / "LM3_settings.yaml"


def test_progress_api_env_override_is_used_verbatim(cwd: Path, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from leafmachine3.server import progress_api

    chosen = _write_yaml(tmp_path / "chosen" / "LM3_settings.yaml", {"version": 3})
    monkeypatch.setenv("LM3_SETTINGS", str(chosen))

    assert progress_api._settings_path() == chosen


def test_the_four_modules_agree_from_one_cwd_with_one_environment(
    cwd: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The split, closed. One process, one CWD, one env -> ONE answer.

    §4 Step 1's exit gate: "from any supported CWD every subsystem reports the same canonical
    settings path". WAS: four different answers from these exact inputs -- the Settings API edited
    the CWD copy while the run launcher, the postprocessor and the status stream read the
    env-named one.
    """
    from leafmachine3.server import metrics_api, postprocess_api, progress_api, settings_api

    env_file = _write_yaml(tmp_path / "env" / "LM3_settings.yaml",
                           {"project": {"run_name": "from_the_env"}})
    _write_yaml(cwd / "LM3_settings.yaml", {"project": {"run_name": "from_the_cwd"}})
    monkeypatch.setenv("LM3_SETTINGS", str(env_file))

    assert settings_api.settings_path() == env_file
    assert metrics_api.default_config_path() == env_file
    assert progress_api._settings_path() == env_file
    assert postprocess_api._lm3_settings()["project"]["run_name"] == "from_the_env"


def test_the_four_modules_agree_again_with_no_environment_at_all(cwd: Path) -> None:
    """Same demonstration with an EMPTY environment, where the three fallbacks used to diverge.

    §4 Step 1: "Renaming variables while leaving three fallbacks reachable from three CWDs
    preserves the bug." With nothing set, all four now land on the same deployment/checkout file,
    and none of them looks in the CWD.
    """
    from leafmachine3.server import app, metrics_api, postprocess_api, progress_api, settings_api

    _write_yaml(cwd / "LM3_settings.yaml", {"project": {"run_name": "from_the_cwd"}})
    canonical = app.canonical_settings_path()

    assert settings_api.settings_path() == canonical
    assert progress_api._settings_path() == canonical
    assert postprocess_api._lm3_settings_path() == canonical
    assert metrics_api.default_config_path() in (canonical, None)   # None only if it is absent
    assert canonical != cwd / "LM3_settings.yaml"


# --------------------------------------------------------------------------- #
# The hardware split (was Appendix A rows 5-7)
# --------------------------------------------------------------------------- #
def test_metrics_api_hardware_honors_the_canonical_variable(cwd: Path, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """§3.1 row 2 step 1: ``LM3_HARDWARE`` is the canonical name.

    WAS: ``metrics_api`` obeyed ``LM3_HARDWARE_SETTINGS`` and ``progress_api`` obeyed
    ``LM3_HARDWARE`` -- two names for one file, in one process.
    """
    from leafmachine3.server import metrics_api

    chosen = _write_yaml(tmp_path / "hw" / "hardware_settings.yaml", {"marker": "canonical"})
    monkeypatch.setenv("LM3_HARDWARE", str(chosen))

    assert metrics_api.hardware_path() == chosen


def test_progress_api_hardware_honors_the_canonical_variable(cwd: Path, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    from leafmachine3.server import progress_api

    chosen = _write_yaml(tmp_path / "hw" / "hardware_settings.yaml", {"marker": "canonical"})
    monkeypatch.setenv("LM3_HARDWARE", str(chosen))

    assert progress_api._hardware() == {"marker": "canonical"}


def test_the_three_hardware_readers_agree_from_one_cwd_with_one_environment(
    cwd: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One process, one CWD -> ONE profile file, for all three server readers.

    §3.1 row 2 replaces the three disagreeing lookups with a single deployment-scoped,
    machine-keyed path. WAS: ``metrics_api`` read ``$LM3_HARDWARE_SETTINGS``, ``progress_api`` read
    ``$LM3_HARDWARE``, and ``app.read_hardware_settings`` read ``<CWD>/hardware_settings.yaml``
    with no override at all -- three files, three answers.
    """
    from leafmachine3.server import app, metrics_api, progress_api

    chosen = _write_yaml(tmp_path / "hw" / "hardware_settings.yaml", {"marker": "canonical"})
    _write_yaml(cwd / "hardware_settings.yaml", {"marker": "cwd"})
    monkeypatch.setenv("LM3_HARDWARE", str(chosen))

    assert metrics_api.hardware_path() == chosen
    assert progress_api._hardware() == {"marker": "canonical"}
    assert app.read_hardware_settings() == {"marker": "canonical"}


def test_app_read_hardware_settings_ignores_a_profile_in_the_cwd(cwd: Path) -> None:
    """§3.1: no CWD fallback. ``GET /v1/hardware`` used to serve whatever sat beside the launcher."""
    from leafmachine3.server import app

    _write_yaml(cwd / "hardware_settings.yaml", {"marker": "cwd"})

    assert app.read_hardware_settings() != {"marker": "cwd"}
