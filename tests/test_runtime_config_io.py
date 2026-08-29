"""``leafmachine3.core.runtime.config_io`` -- config serialization, the section 2.7 bounded pure
resolver, the section 2.10 storage roles, and the section 3.4 launch-manifest builder.

The three properties these tests exist to pin, all of which are cheap to break later:

* ``Config.to_dict()`` is DETERMINISTIC. It is what the launch manifest embeds and what a config
  fingerprint reads, so "the same config" has to be a byte-level claim.
* the early resolver publishes no ``tmp_dir`` of any spelling (section 2.7) and creates nothing --
  it runs before a lease is even attempted.
* a desktop run resolves ``in-place`` with a NULL pointer and no copying (gate 18). Modes exist so
  today's desktop behavior stays byte-for-byte identical; a pointer appearing in a desktop run is
  the regression this file is here to catch.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from leafmachine3.core import paths
from leafmachine3.core.config import Config
from leafmachine3.core.runtime import _types as t
from leafmachine3.core.runtime import config_io as cio


# --------------------------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------------------------- #

def _settings(tmp_path: Path, **project_output: object) -> Path:
    """Write a minimal but realistic LM3_settings.yaml and return its path."""
    data = {
        "project": {
            "run_name": "acer_rubrum",
            "input": {"dirs": [str(tmp_path / "input")]},
            "output": {"dir": str(tmp_path / "output"), "tmp_dir": "auto", **project_output},
        },
        "compute": {"mock": True},
    }
    path = tmp_path / "LM3_settings.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture()
def desktop_cfg(tmp_path: Path) -> Config:
    return Config.load(_settings(tmp_path))


@pytest.fixture()
def cluster_cfg(tmp_path: Path) -> Config:
    """A cluster-shaped config: artifacts on shared storage, active state on node-local scratch."""
    return Config.load(_settings(tmp_path, active_state_dir=str(tmp_path / "scratch")))


# --------------------------------------------------------------------------------------------- #
# Config.to_dict() -- determinism and completeness (plan section 3.4)
# --------------------------------------------------------------------------------------------- #

def test_to_dict_is_byte_identical_across_calls_and_loads(tmp_path: Path) -> None:
    cfg_path = _settings(tmp_path)
    first, second = Config.load(cfg_path), Config.load(cfg_path)

    assert cio.canonical_json(first.to_dict()) == cio.canonical_json(first.to_dict())
    # Two independent loads of the same bytes must fingerprint the same, or the manifest could not
    # be compared against a re-read of the run's own config.
    assert cio.config_json(first) == cio.config_json(second)


def test_to_dict_carries_the_effective_config_after_defaults_and_overrides(tmp_path: Path) -> None:
    cfg = Config.load(_settings(tmp_path), overrides={"ingest": {"jpg_quality": 42}})
    dumped = cfg.to_dict()

    assert dumped["project"]["run_name"] == "acer_rubrum"       # from the file
    assert dumped["ingest"]["jpg_quality"] == 42                # from the overrides
    assert dumped["ingest"]["max_working_dim"] == 3200          # untouched built-in default
    assert dumped["modules"]["reporter"]["enabled"] is True     # defaults are present, not implied


def test_to_dict_sorts_keys_and_normalizes_unjsonable_values(tmp_path: Path) -> None:
    cfg = Config.load(
        _settings(tmp_path),
        overrides={"zzz": 1, "aaa": {"z": Path("/abs/path/"), "a": dt.date(2026, 8, 28)}},
    )
    dumped = cfg.to_dict()

    assert list(dumped) == sorted(dumped)                       # the dict itself is canonical
    assert list(dumped["aaa"]) == ["a", "z"]
    assert dumped["aaa"]["z"] == "/abs/path"                    # Path normalized, trailing slash gone
    assert dumped["aaa"]["a"] == "2026-08-28"                   # a YAML date survives as a string
    json.dumps(dumped)                                          # and the whole tree is JSON-safe


def test_to_dict_is_a_deep_copy(desktop_cfg: Config) -> None:
    dumped = desktop_cfg.to_dict()
    dumped["project"]["run_name"] = "mutated"
    assert desktop_cfg.project.run_name == "acer_rubrum"


def test_config_to_dict_accepts_a_bare_section_tree(desktop_cfg: Config) -> None:
    """Callers holding a raw Section (settings API, fixtures) get the same normalization."""
    assert cio.config_to_dict(desktop_cfg._raw) == desktop_cfg.to_dict()


def test_config_sha256_hashes_the_bytes_actually_loaded(tmp_path: Path) -> None:
    cfg_path = _settings(tmp_path)
    expected = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    assert cio.config_sha256(cfg_path) == expected

    ref = cio.config_ref(Config.load(cfg_path))
    assert ref.path == str(cfg_path.resolve()) and ref.sha256 == expected


def test_config_ref_refuses_to_guess_a_source_path() -> None:
    with pytest.raises(t.RecordSchemaError):
        cio.config_ref(Config({"project": {}}))


# --------------------------------------------------------------------------------------------- #
# The bounded pure resolver (plan section 2.7)
# --------------------------------------------------------------------------------------------- #

def test_early_resolver_emits_exactly_the_deterministic_paths(desktop_cfg: Config, tmp_path: Path) -> None:
    early = cio.resolve_run_paths(desktop_cfg)
    emitted = cio.resolved_paths(early)

    assert set(emitted) == {"run_dir", "log_path", "artifact_dir", "active_state_dir",
                            "active_db_path", "archive_pointer_path"}
    assert emitted["run_dir"] == str(tmp_path / "output" / "acer_rubrum")            # dirs.py 42-44
    assert emitted["active_db_path"] == str(tmp_path / "output" / "acer_rubrum" / "acer_rubrum.sqlite")
    assert emitted["log_path"] == str(tmp_path / "output" / "acer_rubrum" / "logs" / "lm3.log")


def test_early_resolver_never_publishes_a_tmp_dir(desktop_cfg: Config) -> None:
    """Section 2.7: ``_ensure_tmp`` falls back on OSError, so tmp is unknowable in advance."""
    emitted = cio.resolved_paths(cio.resolve_run_paths(desktop_cfg))
    assert not [key for key in emitted if "tmp" in key.lower()]
    assert not hasattr(cio.resolve_run_paths(desktop_cfg), "tmp_dir")


def test_early_resolver_is_pure(desktop_cfg: Config, tmp_path: Path) -> None:
    """It runs before a lease is attempted, so it must create and probe nothing."""
    early = cio.resolve_run_paths(desktop_cfg)
    assert not (tmp_path / "output").exists()
    assert not early.run_dir.exists() and not early.log_path.parent.exists()


def test_relative_output_dir_resolves_against_the_settings_file_not_the_cwd(tmp_path: Path) -> None:
    """Gate 44: no resolved path may fall back to the current working directory."""
    home = tmp_path / "home"
    home.mkdir()
    cfg_path = home / "LM3_settings.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"project": {"run_name": "r1", "output": {"dir": "runs"}}}), encoding="utf-8"
    )
    early = cio.resolve_run_paths(Config.load(cfg_path))
    assert early.run_dir == home / "runs" / "r1"


def test_resolver_rejects_a_config_with_no_run_name(tmp_path: Path) -> None:
    with pytest.raises(t.RecordSchemaError):
        cio.resolve_run_paths(Config({"project": {"run_name": "", "output": {"dir": str(tmp_path)}}}))


# --------------------------------------------------------------------------------------------- #
# The four storage roles and the two archive modes (plan section 2.10, gate 18)
# --------------------------------------------------------------------------------------------- #

def test_desktop_run_resolves_in_place_with_a_null_pointer(desktop_cfg: Config) -> None:
    """Gate 18, the whole point of modes: nothing to stage, so no pointer and no copying."""
    early = cio.resolve_run_paths(desktop_cfg)
    roles = cio.storage_roles(early)

    assert cio.archive_mode(early) is t.ArchiveMode.IN_PLACE
    assert roles[t.StorageRole.ARTIFACT_DIR] == roles[t.StorageRole.ACTIVE_STATE_DIR]
    assert roles[t.StorageRole.ARCHIVE_POINTER_PATH] is None
    assert set(roles) == set(t.StorageRole)                    # exactly the four roles, no more


def test_desktop_project_block_matches_the_section_3_2_example(desktop_cfg: Config) -> None:
    block = cio.project_block(desktop_cfg)
    assert block.archive_mode is t.ArchiveMode.IN_PLACE
    assert block.archive_status is t.ArchiveStatus.NOT_APPLICABLE
    assert block.archive_status.value == "n/a"                 # with a slash; the GUI compares it
    assert block.archive_pointer_path is None
    assert block.archived_db_path == block.active_db_path      # the active file IS the archive
    assert cio.resolve_archived_db_path(block) == block.active_db_path
    assert block.input_dirs and all(Path(d).is_absolute() for d in block.input_dirs)


def test_cluster_config_resolves_staged_with_a_mandatory_pointer(cluster_cfg: Config,
                                                                 tmp_path: Path) -> None:
    early = cio.resolve_run_paths(cluster_cfg)
    roles = cio.storage_roles(early)

    assert cio.archive_mode(early) is t.ArchiveMode.STAGED
    assert roles[t.StorageRole.ARTIFACT_DIR] == str(tmp_path / "output" / "acer_rubrum")
    assert roles[t.StorageRole.ACTIVE_STATE_DIR] == str(tmp_path / "scratch" / "acer_rubrum")
    assert roles[t.StorageRole.ACTIVE_DB_PATH] == str(
        tmp_path / "scratch" / "acer_rubrum" / "acer_rubrum.sqlite"
    )
    assert roles[t.StorageRole.ARCHIVE_POINTER_PATH] == str(
        tmp_path / "output" / "acer_rubrum" / "archive.current.json"
    )


def test_explicit_active_state_dir_argument_beats_the_config(desktop_cfg: Config, tmp_path: Path) -> None:
    early = cio.resolve_run_paths(desktop_cfg, active_state_dir=tmp_path / "node_local")
    assert cio.archive_mode(early) is t.ArchiveMode.STAGED
    assert early.active_state_dir == tmp_path / "node_local" / "acer_rubrum"


def test_staged_run_starts_pending_with_no_archive(cluster_cfg: Config) -> None:
    """Gate 21: the window before the first snapshot is expected, not a failure state."""
    block = cio.project_block(cluster_cfg)
    assert block.archive_status is t.ArchiveStatus.PENDING
    assert block.archived_db_path is None
    assert cio.resolve_archived_db_path(block) is None


def test_deprecated_db_path_projects_active_db_path(desktop_cfg: Config) -> None:
    early = cio.resolve_run_paths(desktop_cfg)
    block = cio.project_block(desktop_cfg)

    assert cio.db_path(early, warn=False) == str(early.active_db_path)
    assert cio.db_path(block, warn=False) == block.active_db_path
    with pytest.warns(DeprecationWarning):
        cio.db_path(early)


# --------------------------------------------------------------------------------------------- #
# Archive pointer resolution (plan section 2.10, gate 14)
# --------------------------------------------------------------------------------------------- #

def _staged_block(cluster_cfg: Config, **kwargs: object) -> t.ProjectBlock:
    return cio.project_block(cluster_cfg, **kwargs)  # type: ignore[arg-type]


def _write_pointer(path: Path, named: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "run_id": "r", "generation": "r.1",
                                "archived_db_path": str(named), "committed_at": 1.0}),
                    encoding="utf-8")


def test_ready_staged_run_resolves_through_the_pointer(cluster_cfg: Config, tmp_path: Path) -> None:
    block = _staged_block(cluster_cfg, archive_status=t.ArchiveStatus.READY)
    generation = tmp_path / "output" / "acer_rubrum" / "archive.abc.1.sqlite"
    generation.parent.mkdir(parents=True, exist_ok=True)
    generation.write_bytes(b"")
    _write_pointer(Path(block.archive_pointer_path or ""), generation)

    assert cio.resolve_archived_db_path(block) == str(generation)


@pytest.mark.parametrize("damage", ["missing", "malformed", "stale"])
def test_pointer_failures_are_precise_never_a_guessed_filename(cluster_cfg: Config, tmp_path: Path,
                                                               damage: str) -> None:
    """Gate 14: while ``ready``, all three failure modes name the fault instead of inventing a path."""
    block = _staged_block(cluster_cfg, archive_status=t.ArchiveStatus.READY)
    pointer = Path(block.archive_pointer_path or "")
    if damage == "malformed":
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text("{not json", encoding="utf-8")
    elif damage == "stale":
        _write_pointer(pointer, tmp_path / "output" / "acer_rubrum" / "archive.gone.1.sqlite")

    with pytest.raises(t.ArchivePointerError) as excinfo:
        cio.resolve_archived_db_path(block)
    assert damage in str(excinfo.value) or "unreadable" in str(excinfo.value)


def test_terminal_statuses_report_the_last_good_generation_or_nothing(cluster_cfg: Config) -> None:
    """Section 2.10's terminal table, and gate 25's two halves."""
    stale = _staged_block(cluster_cfg, archive_status=t.ArchiveStatus.STALE,
                          archived_db_path="/shared/output/archive.abc.7.sqlite",
                          archive_error="final checkpoint failed")
    failed = _staged_block(cluster_cfg, archive_status=t.ArchiveStatus.FAILED,
                           archive_error="no snapshot ever committed")

    assert cio.resolve_archived_db_path(stale) == "/shared/output/archive.abc.7.sqlite"
    assert cio.resolve_archived_db_path(failed) is None


# --------------------------------------------------------------------------------------------- #
# The launch manifest builder (plan section 3.4)
# --------------------------------------------------------------------------------------------- #

def _manifest(cfg: Config, tmp_path: Path, **kwargs: object) -> dict:
    early = cio.resolve_run_paths(cfg)
    return cio.build_launch_manifest(
        cfg, run_id="11111111-2222-3333-4444-555555555555", launcher=t.Launcher.CLI,
        early=early, tmp_dir=tmp_path / "output" / "acer_rubrum" / "_tmp_original",
        # 1787918400.0 is 2026-08-28T12:00:00Z -- the plan section 3.2 example timestamp.
        clock=lambda: 1787918400.0, **kwargs,  # type: ignore[arg-type]
    )


def test_manifest_carries_every_required_field(desktop_cfg: Config, tmp_path: Path) -> None:
    manifest = _manifest(desktop_cfg, tmp_path, overrides={"ingest": {"jpg_quality": 42}})

    assert set(manifest) == {"schema_version", "run_id", "parent_run_id", "config",
                             "effective_config", "overrides", "project", "launcher", "versions",
                             "started_at"}
    assert manifest["schema_version"] == t.SCHEMA_VERSION
    assert manifest["parent_run_id"] is None
    assert Path(manifest["config"]["path"]).is_absolute()
    assert manifest["config"]["sha256"] == cio.config_sha256(desktop_cfg.source_path)
    assert manifest["effective_config"] == desktop_cfg.to_dict()
    assert manifest["overrides"] == {"ingest": {"jpg_quality": 42}}
    assert manifest["launcher"] == "cli"
    assert set(manifest["versions"]) == {"lm3", "python", "platform"}
    assert manifest["started_at"] == "2026-08-28T12:00:00Z"


def test_manifest_project_block_has_the_storage_roles_and_the_settled_tmp_dir(desktop_cfg: Config,
                                                                             tmp_path: Path) -> None:
    project = _manifest(desktop_cfg, tmp_path)["project"]

    assert set(project) == {"run_name", "input_dirs", "archive_mode", "run_dir", "log_path",
                            "artifact_dir", "active_state_dir", "active_db_path",
                            "archive_pointer_path", "tmp_dir"}
    assert project["archive_mode"] == "in-place"
    assert project["archive_pointer_path"] is None
    # tmp_dir is present ONLY here, and only because build_dirs() has already settled it -- the
    # manifest is written after directory creation (section 3.4), the early record before it.
    assert project["tmp_dir"] == str(tmp_path / "output" / "acer_rubrum" / "_tmp_original")


def test_manifest_records_a_child_parent_and_serializes_deterministically(desktop_cfg: Config,
                                                                         tmp_path: Path) -> None:
    kwargs = {"parent_run_id": "aaaa-bbbb", "versions": {"lm3": "0.1.0", "python": "3.10.0",
                                                         "platform": "linux-x86_64"}}
    first = _manifest(desktop_cfg, tmp_path, **kwargs)
    second = _manifest(desktop_cfg, tmp_path, **kwargs)

    assert first["parent_run_id"] == "aaaa-bbbb"
    assert cio.canonical_json(first) == cio.canonical_json(second)


def test_manifest_path_lives_beside_the_log_under_artifact_dir(cluster_cfg: Config,
                                                               tmp_path: Path) -> None:
    """Section 3.4: on a cluster the manifest must survive the allocation, so never node-local."""
    early = cio.resolve_run_paths(cluster_cfg)
    manifest_path = cio.launch_manifest_path(early)

    assert manifest_path == tmp_path / "output" / "acer_rubrum" / "logs" / "run_manifest.json"
    assert early.artifact_dir in manifest_path.parents
    assert early.active_state_dir not in manifest_path.parents


def test_early_run_paths_contract_is_the_one_in_core_paths(desktop_cfg: Config) -> None:
    """The resolver delegates; it does not re-derive the layout (section 2.7 / core.paths)."""
    early = cio.resolve_run_paths(desktop_cfg)
    assert isinstance(early, paths.EarlyRunPaths)
    assert early.run_dir == early.artifact_dir
