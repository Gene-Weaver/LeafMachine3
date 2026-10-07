"""Section 3.1 row 3 for the standalone postprocessing tools.

``leafmachine3.postprocessing.config.load_settings`` used to fall back to a bare
``postprocessing_settings.yaml`` off the current working directory, so the Postprocess tab (which
already resolves canonically, ``postprocess_api._settings_path``) and the CLIs configured different
files depending on where each was launched. Section 3.1's governing rule for the precedence table
is "No path falls back to the current working directory", and its resolver-scope bullet names "the
standalone hardware-setup and postprocessing CLIs" explicitly; section 4 step 1 says to cover the
full scope in section 3.1 and to "unify fallbacks, not just names", with the exit gate "from any
supported CWD every subsystem reports the same canonical settings path".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from leafmachine3.core import paths
from leafmachine3.postprocessing import config as pp_config


def test_no_explicit_path_reads_the_deployment_file_not_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 3.1 row 3. A decoy in the CWD must be invisible to ``load_settings(None)``."""
    canonical = tmp_path / "canonical.yaml"
    canonical.write_text("generate_stl_from_mask:\n  thickness_mm: 7\n", encoding="utf-8")
    monkeypatch.setenv(paths.ENV_POSTPROCESS_SETTINGS, str(canonical))

    work = tmp_path / "work"
    work.mkdir()
    # The exact legacy name and the exact key the CLI reads: if the CWD fallback survived anywhere,
    # this decoy -- not the canonical file -- is what the tool would be configured by.
    (work / "postprocessing_settings.yaml").write_text(
        "generate_stl_from_mask:\n  thickness_mm: 999\n", encoding="utf-8"
    )
    monkeypatch.chdir(work)

    loaded = pp_config.load_settings(None)

    assert loaded == {"generate_stl_from_mask": {"thickness_mm": 7}}
    assert pp_config.module_settings(loaded, "generate_stl_from_mask")["thickness_mm"] == 7


def test_the_cli_loader_and_the_postprocess_tab_agree_from_any_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 1's exit gate: both halves of section 3.1's "hardware-setup and postprocessing CLIs"
    sentence must name one file, from every working directory."""
    from leafmachine3.server import postprocess_api

    # Its own file rather than the session sandbox's: conftest deliberately names a postprocessing
    # path that does NOT exist, and creating it would change what every other test resolves to.
    canonical = tmp_path / "deployment" / "postprocessing.yaml"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("generate_leaf_collage:\n  columns: 4\n", encoding="utf-8")
    monkeypatch.setenv(paths.ENV_POSTPROCESS_SETTINGS, str(canonical))

    for name in ("a", "b"):
        work = tmp_path / name
        work.mkdir()
        (work / "postprocessing_settings.yaml").write_text(
            "generate_leaf_collage:\n  columns: 99\n", encoding="utf-8"
        )
        monkeypatch.chdir(work)

        assert postprocess_api._settings_path() == canonical
        assert pp_config.load_settings(None) == {"generate_leaf_collage": {"columns": 4}}


def test_an_explicit_path_still_wins_and_a_missing_one_is_still_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row 3's on-miss cell is "packaged defaults", not a hard error, so an absent explicit file
    keeps returning ``{}`` -- the hard-error rule belongs to the next-run-settings row."""
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("generate_stl_from_mask:\n  thickness_mm: 3\n", encoding="utf-8")
    monkeypatch.setenv(paths.ENV_POSTPROCESS_SETTINGS, str(tmp_path / "canonical.yaml"))

    assert pp_config.load_settings(str(explicit)) == {"generate_stl_from_mask": {"thickness_mm": 3}}
    assert pp_config.load_settings(str(tmp_path / "absent.yaml")) == {}
    assert pp_config.load_settings(None) == {}   # canonical file does not exist -> packaged defaults


def test_an_unresolvable_deployment_degrades_to_packaged_defaults(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DeploymentIdentityError`` is a ``PathsError``; the tools degrade where the server does
    (``postprocess_api._settings_path``) instead of crashing a standalone CLI."""
    monkeypatch.delenv(paths.ENV_POSTPROCESS_SETTINGS, raising=False)
    monkeypatch.setenv(paths.ENV_DEPLOYMENT_ID, "   ")   # empty/whitespace -> DeploymentIdentityError

    with pytest.raises(paths.PathsError):
        paths.postprocessing_settings_path()
    assert pp_config.load_settings(None) == {}


def test_the_module_no_longer_publishes_a_cwd_relative_default() -> None:
    """The removed ``DEFAULT_SETTINGS_PATH`` was the CWD fallback itself; re-adding it would put
    section 8 gate 44 ("no resolved path falls back to the current working directory") back at
    risk for every importer."""
    assert not hasattr(pp_config, "DEFAULT_SETTINGS_PATH")
