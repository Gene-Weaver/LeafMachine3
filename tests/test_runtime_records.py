"""``leafmachine3.core.runtime.records`` -- the record half of the unified runtime.

Covers plan section 3.2 (record schemas, writer ownership, the sequence, retention, the recovery
writer), section 3.3 (finalization ordering), section 2.9 (schema skew), and section 2.10's archive
contract as it appears *in the record*. The named regression gates exercised here are 4, 5, 14, 18,
21, 25, 26 and 50.

Everything runs inside ``tmp_path``: no test touches a real deployment runtime directory, and none
of them needs a lease -- ``read_runtime`` takes an injected ``lease_probe`` and the recovery writer
takes an injected lease object, which is what lets the record rules be tested independently of the
platform lock adapters.
"""
from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from leafmachine3.core.runtime import _types as T
from leafmachine3.core.runtime import records as R

# --------------------------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------------------------- #

NOW = "2026-08-28T12:00:00Z"
LATER = "2026-08-28T12:30:00Z"


def deployment_dir(tmp_path: Path) -> Path:
    path = tmp_path / "runtime" / "default"
    (path / T.CHILDREN_DIRNAME).mkdir(parents=True, exist_ok=True)
    return path


def config_ref(tmp_path: Path) -> T.ConfigRef:
    return T.ConfigRef(path=str(tmp_path / "LM3_settings.yaml"), sha256="a" * 64)


def in_place_project(tmp_path: Path, run_name: str = "acer_rubrum") -> T.ProjectBlock:
    run_dir = tmp_path / "out" / run_name
    db = run_dir / f"{run_name}.sqlite"
    return T.ProjectBlock(
        run_name=run_name,
        input_dirs=(str(tmp_path / "input"),),
        artifact_dir=str(run_dir),
        active_state_dir=str(run_dir),
        active_db_path=str(db),
        archive_mode=T.ArchiveMode.IN_PLACE,
        archive_status=T.ArchiveStatus.NOT_APPLICABLE,
        archive_pointer_path=None,
        archived_db_path=str(db),
        run_dir=str(run_dir),
        log_path=str(run_dir / "logs" / "lm3.log"),
    )


def staged_project(
    tmp_path: Path,
    *,
    status: T.ArchiveStatus = T.ArchiveStatus.PENDING,
    archived: str | None = None,
    archive_error: str | None = None,
    run_name: str = "acer_rubrum",
) -> T.ProjectBlock:
    artifact_dir = tmp_path / "persistent" / run_name
    scratch = tmp_path / "scratch" / run_name
    return T.ProjectBlock(
        run_name=run_name,
        input_dirs=(str(tmp_path / "input"),),
        artifact_dir=str(artifact_dir),
        active_state_dir=str(scratch),
        active_db_path=str(scratch / f"{run_name}.sqlite"),
        archive_mode=T.ArchiveMode.STAGED,
        archive_status=status,
        archive_pointer_path=str(artifact_dir / T.ARCHIVE_POINTER_FILENAME),
        archived_db_path=archived,
        run_dir=str(artifact_dir),
        log_path=str(artifact_dir / "logs" / "lm3.log"),
        archive_error=archive_error,
    )


def root_record(tmp_path: Path, **overrides) -> T.RuntimeRecord:
    base = dict(
        run_id="11111111-1111-4111-8111-111111111111",
        activity=T.Activity.PIPELINE,
        activity_role=T.ActivityRole.ROOT,
        state=T.RunState.RUNNING,
        launcher=T.Launcher.CLI,
        pid=4321,
        process_started_at=1787932800.25,
        started_at=NOW,
        updated_at=NOW,
        deployment=T.DeploymentInfo(id="default", node="gpu042"),
        config=config_ref(tmp_path),
        project=in_place_project(tmp_path),
    )
    base.update(overrides)
    return T.RuntimeRecord(**base)


def child_record(tmp_path: Path, **overrides) -> T.RuntimeRecord:
    base = dict(
        run_id="22222222-2222-4222-8222-222222222222",
        activity=T.Activity.CALIBRATION_PIPELINE,
        activity_role=T.ActivityRole.CHILD,
        state=T.RunState.RUNNING,
        launcher=T.Launcher.PYTHON,
        pid=4322,
        process_started_at=1787932801.25,
        started_at=NOW,
        updated_at=NOW,
        deployment=T.DeploymentInfo(id="default"),
        parent_run_id="11111111-1111-4111-8111-111111111111",
        # calibration_pipeline is the only subactivity left (plan revision 14), and its run_name is
        # fixed by the schema -- so the default child project must be the calibration one.
        project=in_place_project(tmp_path, T.CALIBRATION_RUN_NAME),
    )
    base.update(overrides)
    return T.RuntimeRecord(**base)


def summary_of(record: T.RuntimeRecord) -> T.ChildSummary:
    return T.ChildSummary(
        run_id=record.run_id,
        activity=record.activity,
        state=record.state,
        started_at=record.started_at,
        run_name=record.project.run_name if record.project else None,
        run_dir=record.project.run_dir if record.project else None,
    )


class FakeLease:
    """The minimum ``recover_abandoned`` inspects: is it held, and for which deployment."""

    def __init__(self, dir_: Path, *, held: bool = True) -> None:
        self.deployment_dir = Path(dir_)
        self.held = held


def free() -> bool:
    return False


def occupied() -> bool:
    return True


# --------------------------------------------------------------------------------------------- #
# 1. atomic IO
# --------------------------------------------------------------------------------------------- #

def test_atomic_write_publishes_the_payload_user_only_and_leaves_no_temp_file(tmp_path: Path):
    target = tmp_path / "nested" / "active.json"
    R.atomic_write_json(target, {"schema_version": 1, "run_id": "x"})

    assert json.loads(target.read_text()) == {"schema_version": 1, "run_id": "x"}
    assert (target.stat().st_mode & 0o777) == 0o600
    assert [p.name for p in target.parent.iterdir()] == ["active.json"]


def test_a_failed_write_leaves_the_previous_record_intact_and_removes_the_temp(tmp_path: Path):
    target = tmp_path / "active.json"
    R.atomic_write_json(target, {"schema_version": 1, "run_id": "first"})

    with pytest.raises(TypeError):
        R.atomic_write_json(target, {"schema_version": 1, "bad": object()})

    assert json.loads(target.read_text())["run_id"] == "first"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["active.json"]


def test_atomic_write_replaces_rather_than_truncating(tmp_path: Path):
    target = tmp_path / "active.json"
    R.atomic_write_json(target, {"a": "x" * 500})
    first_inode = target.stat().st_ino
    R.atomic_write_json(target, {"a": "y"})
    # A rename, not an in-place rewrite: the inode changes, which is why a concurrent reader either
    # sees the whole old record or the whole new one.
    assert target.stat().st_ino != first_inode
    assert json.loads(target.read_text()) == {"a": "y"}


@pytest.mark.parametrize(
    "make, expects_error",
    [
        (lambda p: None, False),                                     # absent
        (lambda p: p.write_text("{not json"), True),                 # malformed
        (lambda p: p.write_text('"a string"'), True),                # not an object
        (lambda p: p.write_bytes(b"\xff\xfe not utf8"), True),       # not UTF-8
        (lambda p: p.write_text("{}" + " " * (R.MAX_READ_BYTES + 1)), True),  # oversized
    ],
)
def test_read_json_file_never_raises(tmp_path: Path, make, expects_error: bool):
    path = tmp_path / "active.json"
    make(path)
    payload, error = R.read_json_file(path)
    assert payload is None
    assert (error is not None) is expects_error


def test_read_json_file_reports_a_directory_instead_of_raising(tmp_path: Path):
    payload, error = R.read_json_file(tmp_path)
    assert payload is None and error


def test_read_json_file_refuses_an_oversized_record_without_allocating_it(tmp_path: Path):
    """The reader bound must bound the *read*, not merely reject what was already loaded.

    Section 3.1 requires readers to "bound field sizes on read", and :data:`R.MAX_READ_BYTES`
    promises an over-large file is "reported as unreadable instead of loaded into memory". Reading
    the whole file first and comparing lengths afterwards satisfies neither: a corrupt or grown
    ``active.json`` would then cost its full size on every status poll. Allocation is measured with
    ``tracemalloc`` because it is deterministic -- ``ru_maxrss`` is a lifetime high-water mark that
    an earlier peak can mask -- and the rusage growth is asserted too, generously, as a cross-check.
    """
    import tracemalloc

    path = tmp_path / "active.json"
    chunk = b"x" * (64 * 1024)
    with open(path, "wb") as handle:                      # written in chunks: the writer itself
        for _ in range(128):                              # must not allocate anything large either
            handle.write(chunk)
    file_size = path.stat().st_size
    assert file_size > R.MAX_READ_BYTES

    rusage = pytest.importorskip("resource")              # absent on Windows
    before = rusage.getrusage(rusage.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()
    try:
        payload, error = R.read_json_file(path)
        _, traced_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    grew_bytes = (rusage.getrusage(rusage.RUSAGE_SELF).ru_maxrss - before) * 1024

    assert payload is None
    assert error is not None and "oversized" in error
    assert traced_peak < 1024 * 1024                      # nowhere near the 8 MiB on disk
    assert grew_bytes < 1024 * 1024


# --------------------------------------------------------------------------------------------- #
# 2. serialization round trip
# --------------------------------------------------------------------------------------------- #

def test_record_round_trips_through_json(tmp_path: Path):
    record = root_record(tmp_path, current_child=summary_of(child_record(tmp_path)))
    assert R.record_from_dict(R.record_to_dict(record)) == record


def test_a_missing_optional_key_and_an_explicit_null_are_the_same(tmp_path: Path):
    payload = R.record_to_dict(root_record(tmp_path))
    with_nulls = dict(payload)
    with_nulls.update(error=None, finished_at=None, returncode=None, parent_run_id=None)
    without = {k: v for k, v in payload.items()
               if k not in {"error", "finished_at", "returncode", "parent_run_id"}}
    assert R.record_from_dict(with_nulls) == R.record_from_dict(without)


def test_a_higher_schema_version_is_never_parsed(tmp_path: Path):
    payload = R.record_to_dict(root_record(tmp_path))
    payload["schema_version"] = T.SCHEMA_VERSION + 1
    with pytest.raises(T.IncompatibleSchemaError) as excinfo:
        R.record_from_dict(payload)
    assert excinfo.value.schema_version == T.SCHEMA_VERSION + 1


@pytest.mark.parametrize("missing", ["run_id", "activity", "state", "pid", "deployment"])
def test_a_missing_required_common_field_is_a_schema_error(tmp_path: Path, missing: str):
    payload = R.record_to_dict(root_record(tmp_path))
    payload.pop(missing)
    with pytest.raises(T.RecordSchemaError):
        R.record_from_dict(payload)


def test_a_bad_enum_value_is_a_schema_error(tmp_path: Path):
    payload = R.record_to_dict(root_record(tmp_path))
    payload["activity"] = "mining_bitcoin"
    with pytest.raises(T.RecordSchemaError):
        R.record_from_dict(payload)


# --------------------------------------------------------------------------------------------- #
# 3. per-activity validation (plan section 3.2 table)
# --------------------------------------------------------------------------------------------- #

def test_a_well_formed_record_of_every_activity_validates(tmp_path: Path):
    calibration_project = in_place_project(tmp_path, T.CALIBRATION_RUN_NAME)
    cases = [
        root_record(tmp_path),
        root_record(tmp_path, activity=T.Activity.HARDWARE_SETUP, project=None,
                    hardware=T.HardwareBlock(destination_path=str(tmp_path / "hardware.yaml"))),
        child_record(tmp_path, project=calibration_project),
    ]
    for record in cases:
        R.validate_record(record)


@pytest.mark.parametrize("dropped", ["config", "project"])
def test_pipeline_requires_config_and_project(tmp_path: Path, dropped: str):
    record = root_record(tmp_path, **{dropped: None})
    with pytest.raises(T.RecordSchemaError, match=dropped):
        R.validate_record(record)


def test_hardware_setup_requires_config_and_hardware_and_forbids_project(tmp_path: Path):
    hardware = T.HardwareBlock(destination_path=str(tmp_path / "hardware.yaml"))
    with pytest.raises(T.RecordSchemaError, match="hardware"):
        R.validate_record(root_record(tmp_path, activity=T.Activity.HARDWARE_SETUP, project=None))
    with pytest.raises(T.RecordSchemaError, match="config"):
        R.validate_record(root_record(tmp_path, activity=T.Activity.HARDWARE_SETUP, project=None,
                                      config=None, hardware=hardware))
    with pytest.raises(T.RecordSchemaError, match="project"):
        R.validate_record(root_record(tmp_path, activity=T.Activity.HARDWARE_SETUP,
                                      hardware=hardware))


@pytest.mark.parametrize("activity",
                         [T.Activity.CALIBRATION_PIPELINE])
def test_a_child_requires_parent_run_id_and_a_project(tmp_path: Path, activity: T.Activity):
    project = in_place_project(
        tmp_path, T.CALIBRATION_RUN_NAME if activity is T.Activity.CALIBRATION_PIPELINE else "q"
    )
    with pytest.raises(T.RecordSchemaError, match="parent_run_id"):
        R.validate_record(child_record(tmp_path, activity=activity, project=project,
                                       parent_run_id=None))
    with pytest.raises(T.RecordSchemaError, match="project"):
        R.validate_record(child_record(tmp_path, activity=activity, project=None))


def test_calibration_pins_the_reserved_run_name(tmp_path: Path):
    record = child_record(tmp_path, activity=T.Activity.CALIBRATION_PIPELINE,
                          project=in_place_project(tmp_path, "acer_rubrum"))
    with pytest.raises(T.RecordSchemaError, match=T.CALIBRATION_RUN_NAME):
        R.validate_record(record)
    R.validate_record(dataclasses.replace(
        record, project=in_place_project(tmp_path, T.CALIBRATION_RUN_NAME)
    ))


def test_activity_role_must_match_the_activity(tmp_path: Path):
    # Only a root record corresponds to a lock acquisition; a child claiming root would assert one
    # that never happened.
    record = child_record(tmp_path, activity_role=T.ActivityRole.ROOT)
    with pytest.raises(T.RecordSchemaError, match="activity_role"):
        R.validate_record(record)


def test_a_root_may_not_carry_a_parent_run_id(tmp_path: Path):
    with pytest.raises(T.RecordSchemaError, match="parent_run_id"):
        R.validate_record(root_record(tmp_path, parent_run_id="somebody-else"))


def test_paths_must_be_absolute(tmp_path: Path):
    relative = dataclasses.replace(in_place_project(tmp_path), run_dir="out/acer_rubrum")
    with pytest.raises(T.RecordSchemaError, match="absolute"):
        R.validate_record(root_record(tmp_path, project=relative))
    with pytest.raises(T.RecordSchemaError, match="absolute"):
        R.validate_record(root_record(tmp_path, config=T.ConfigRef(path="LM3_settings.yaml",
                                                                   sha256="a" * 64)))


def test_a_terminal_record_must_say_when_it_finished(tmp_path: Path):
    with pytest.raises(T.RecordSchemaError, match="finished_at"):
        R.validate_record(root_record(tmp_path, state=T.RunState.DONE))
    with pytest.raises(T.RecordSchemaError, match="finished_at"):
        R.validate_record(root_record(tmp_path, state=T.RunState.RUNNING, finished_at=LATER))
    R.validate_record(root_record(tmp_path, state=T.RunState.DONE, finished_at=LATER))


# --- secrets -----------------------------------------------------------------------------------#

def test_an_environment_dump_is_refused_rather_than_ignored(tmp_path: Path):
    payload = R.record_to_dict(root_record(tmp_path))
    payload["environ"] = {"PATH": "/usr/bin"}
    with pytest.raises(T.RecordSchemaError, match="unknown field"):
        R.record_from_dict(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.__setitem__("bearer_token", "abc"),
        lambda p: p["deployment"].__setitem__("authorization", "Basic abc"),
        lambda p: p["project"].__setitem__("api_key", "abc"),
    ],
)
def test_a_secret_shaped_key_is_refused_at_any_depth(tmp_path: Path, mutate):
    payload = R.record_to_dict(root_record(tmp_path))
    mutate(payload)
    with pytest.raises(T.RecordSchemaError, match="secret-shaped"):
        R.record_from_dict(payload)


def test_validate_rejects_a_credential_in_a_value_and_sanitize_redacts_it(tmp_path: Path):
    leaky = root_record(
        tmp_path,
        error="POST /v1/run/start failed: Authorization: Bearer eyJhbGciOi.notreal.signature",
    )
    with pytest.raises(T.RecordSchemaError, match="credential"):
        R.validate_record(leaky)

    cleaned = R.sanitize_record(leaky)
    assert "eyJhbGciOi" not in (cleaned.error or "")
    assert T.REDACTION_PLACEHOLDER in (cleaned.error or "")
    R.validate_record(cleaned)  # sanitize-then-validate is the order RecordStore uses


def test_ordinary_prose_containing_the_word_token_survives_sanitization(tmp_path: Path):
    message = "stage failed: the ruler token detector found no lattice"
    cleaned = R.sanitize_record(root_record(tmp_path, error=message))
    assert cleaned.error == message


# --- bounds ------------------------------------------------------------------------------------#

def test_validate_rejects_an_unbounded_field_and_sanitize_truncates_it(tmp_path: Path):
    record = root_record(tmp_path, error="x" * (T.MAX_ERROR_CHARS + 500))
    with pytest.raises(T.RecordSchemaError, match="exceeds"):
        R.validate_record(record)
    cleaned = R.sanitize_record(record)
    assert len(cleaned.error or "") == T.MAX_ERROR_CHARS
    assert (cleaned.error or "").endswith(R.TRUNCATION_MARKER)
    R.validate_record(cleaned)


def test_sanitize_shrinks_a_record_that_the_per_field_bounds_alone_cannot_bound(tmp_path: Path):
    # 64 input dirs at 4096 characters each is 256 KiB -- four times MAX_RECORD_BYTES -- so the
    # per-field caps are satisfied while the record is not.
    fat = tuple(f"/{'d' * 4000}/{index}" for index in range(T.MAX_INPUT_DIRS))
    project = dataclasses.replace(in_place_project(tmp_path), input_dirs=fat)
    record = root_record(tmp_path, project=project)
    with pytest.raises(T.RecordSchemaError, match="bounded description"):
        R.validate_record(record)

    cleaned = R.sanitize_record(record)
    assert len(json.dumps(R.record_to_dict(cleaned))) <= T.MAX_RECORD_BYTES
    assert len(cleaned.project.input_dirs) < T.MAX_INPUT_DIRS
    R.validate_record(cleaned)


def test_the_active_record_stays_bounded_however_many_children_a_root_launches(tmp_path: Path):
    """active.json carries child SUMMARIES, never child records: six keys each, two of them.

    The bound is what lets a root launch children indefinitely without its record growing. It was
    written for the batch; plan revision 14 removed that activity, but a long-lived
    ``hardware_setup`` re-running calibration has exactly the same shape, so the property still
    matters and is still worth pinning.
    """
    store = R.RecordStore(deployment_dir(tmp_path), run_id="11111111-1111-4111-8111-111111111111")
    record = root_record(tmp_path, activity=T.Activity.HARDWARE_SETUP, project=None,
                         hardware=T.HardwareBlock(destination_path=str(tmp_path / "hw.yaml")))
    store.write_active(record)
    for index in range(200):
        child = child_record(tmp_path, run_id=f"child-{index}",
                             project=in_place_project(tmp_path, f"child_run_{index}"))
        store.set_current_child(summary_of(child))
        store.promote_current_child()
    assert store.active_path.stat().st_size < T.MAX_RECORD_BYTES


# --------------------------------------------------------------------------------------------- #
# 4. the archive contract in the record (section 2.10)
# --------------------------------------------------------------------------------------------- #

def test_a_desktop_run_is_in_place_with_a_null_pointer(tmp_path: Path):
    """Gate 18, record side: in-place means no pointer, no generations, no copying."""
    project = in_place_project(tmp_path)
    assert project.archive_pointer_path is None
    assert project.archived_db_path == project.active_db_path
    R.validate_record(root_record(tmp_path, project=project))


def test_a_staged_run_before_its_first_snapshot_is_pending_not_a_failure(tmp_path: Path):
    """Gate 21: the pending window is expected, and a null archive path is correct there."""
    project = staged_project(tmp_path, status=T.ArchiveStatus.PENDING)
    assert project.archived_db_path is None
    R.validate_record(root_record(tmp_path, project=project))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(status=T.ArchiveStatus.PENDING, archived="/persistent/a.sqlite"), "pending"),
        (dict(status=T.ArchiveStatus.READY, archived=None), "ready"),
        (dict(status=T.ArchiveStatus.STALE, archived=None), "stale"),
        (dict(status=T.ArchiveStatus.STALE, archived="/persistent/a.sqlite"), "archive_error"),
        (dict(status=T.ArchiveStatus.FAILED, archived="/persistent/a.sqlite",
              archive_error="boom"), "failed"),
        (dict(status=T.ArchiveStatus.FAILED, archived=None), "archive_error"),
        (dict(status=T.ArchiveStatus.NOT_APPLICABLE), "n/a"),
    ],
)
def test_every_illegal_staged_pairing_is_rejected(tmp_path: Path, kwargs, match: str):
    with pytest.raises(T.RecordSchemaError, match=match):
        R.validate_record(root_record(tmp_path, project=staged_project(tmp_path, **kwargs)))


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("archive_status", T.ArchiveStatus.PENDING, "in-place"),
        ("archive_pointer_path", "/persistent/archive.current.json", "pointer"),
        ("archived_db_path", "/somewhere/else.sqlite", "must equal"),
        ("archive_error", "boom", "archive_error"),
    ],
)
def test_every_illegal_in_place_pairing_is_rejected(tmp_path: Path, field, value, match):
    project = dataclasses.replace(in_place_project(tmp_path), **{field: value})
    with pytest.raises(T.RecordSchemaError, match=match):
        R.validate_record(root_record(tmp_path, project=project))


@pytest.mark.parametrize(
    "status, archived_factory, archive_error",
    [
        (T.ArchiveStatus.READY, lambda p: str(p / "persistent" / "acer_rubrum" / "gen.1.sqlite"),
         None),
        (T.ArchiveStatus.STALE, lambda p: str(p / "persistent" / "acer_rubrum" / "gen.1.sqlite"),
         "final checkpoint failed"),
        (T.ArchiveStatus.FAILED, lambda p: None, "no snapshot ever committed"),
    ],
)
def test_last_json_accepts_exactly_the_terminal_archive_shapes(
    tmp_path: Path, status, archived_factory, archive_error
):
    """Gate 14: a resolved path under ready/stale, null only under failed."""
    store = R.RecordStore(deployment_dir(tmp_path), run_id="11111111-1111-4111-8111-111111111111")
    project = staged_project(tmp_path, status=status, archived=archived_factory(tmp_path),
                             archive_error=archive_error)
    store.finalize(root_record(tmp_path, project=project, state=T.RunState.DONE,
                               finished_at=LATER))
    written = store.read_last()
    assert written is not None
    assert written.project.archive_status is status


def test_last_json_refuses_a_pending_archive(tmp_path: Path):
    store = R.RecordStore(deployment_dir(tmp_path), run_id="11111111-1111-4111-8111-111111111111")
    record = root_record(tmp_path, project=staged_project(tmp_path), state=T.RunState.DONE,
                         finished_at=LATER)
    with pytest.raises(T.RecordSchemaError, match="pending"):
        store.finalize(record)
    assert not store.last_path.exists()


def test_last_json_never_names_a_node_local_scratch_path(tmp_path: Path):
    """Gate 26: that storage is gone with the allocation."""
    scratch_generation = str(tmp_path / "scratch" / "acer_rubrum" / "gen.1.sqlite")
    project = staged_project(tmp_path, status=T.ArchiveStatus.READY, archived=scratch_generation)
    store = R.RecordStore(deployment_dir(tmp_path), run_id="11111111-1111-4111-8111-111111111111")
    with pytest.raises(T.RecordSchemaError, match="node-local"):
        store.finalize(root_record(tmp_path, project=project, state=T.RunState.DONE,
                                   finished_at=LATER))


# --------------------------------------------------------------------------------------------- #
# 5. writer ownership (section 3.2)
# --------------------------------------------------------------------------------------------- #

def test_a_child_store_cannot_write_the_root_files(tmp_path: Path):
    child = child_record(tmp_path)
    store = R.RecordStore(deployment_dir(tmp_path), run_id=child.run_id, role=T.ActivityRole.CHILD)
    with pytest.raises(T.WriterOwnershipError):
        store.write_active(root_record(tmp_path))
    with pytest.raises(T.WriterOwnershipError):
        store.finalize(root_record(tmp_path, state=T.RunState.DONE, finished_at=LATER))
    with pytest.raises(T.WriterOwnershipError):
        store.set_current_child(summary_of(child))
    with pytest.raises(T.WriterOwnershipError):
        store.promote_current_child()


def test_a_root_store_cannot_write_a_child_record(tmp_path: Path):
    store = R.RecordStore(deployment_dir(tmp_path), run_id="11111111-1111-4111-8111-111111111111")
    with pytest.raises(T.WriterOwnershipError):
        store.write_child(child_record(tmp_path))


def test_a_child_writes_only_its_own_record(tmp_path: Path):
    mine = child_record(tmp_path)
    theirs = child_record(tmp_path, run_id="33333333-3333-4333-8333-333333333333")
    store = R.RecordStore(deployment_dir(tmp_path), run_id=mine.run_id, role=T.ActivityRole.CHILD)
    store.write_child(mine)
    with pytest.raises(T.WriterOwnershipError):
        store.write_child(theirs)
    assert store.read_child(mine.run_id) == mine
    assert store.read_child(theirs.run_id) is None


def test_a_root_store_refuses_a_record_for_another_run(tmp_path: Path):
    store = R.RecordStore(deployment_dir(tmp_path), run_id="mine")
    with pytest.raises(T.WriterOwnershipError):
        store.write_active(root_record(tmp_path))


def test_a_root_will_not_edit_another_runs_active_record(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111").write_active(
        root_record(tmp_path)
    )
    intruder = R.RecordStore(dep, run_id="99999999-9999-4999-8999-999999999999")
    with pytest.raises(T.WriterOwnershipError):
        intruder.set_current_child(summary_of(child_record(tmp_path)))


def test_concurrent_root_and_child_writes_do_not_clobber_each_other(tmp_path: Path):
    """The lease does not serialize these writes -- static ownership plus atomic replace does.

    The root and the child share one open file description, so nothing in the lock stops their
    JSON writes from interleaving. Each therefore owns a different file and every publication is a
    rename, which is why a reader interleaved with both never sees a partial record.
    """
    dep = deployment_dir(tmp_path)
    child = child_record(tmp_path)
    root_store = R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111")
    child_store = R.RecordStore(dep, run_id=child.run_id, role=T.ActivityRole.CHILD)
    root_store.write_active(root_record(tmp_path))
    child_store.write_child(child)

    iterations = 60
    start = threading.Barrier(3)
    failures: list[str] = []

    def write_root() -> None:
        start.wait()
        for index in range(iterations):
            try:
                root_store.write_active(root_record(tmp_path, updated_at=NOW,
                                                    returncode=None, pid=1000 + index))
            except Exception as exc:  # noqa: BLE001 - recorded, asserted on below
                failures.append(f"root: {exc!r}")

    def write_child() -> None:
        start.wait()
        for index in range(iterations):
            try:
                child_store.write_child(child_record(tmp_path, pid=2000 + index))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"child: {exc!r}")

    def read_both() -> None:
        start.wait()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            for path in (root_store.active_path,
                         dep / T.CHILDREN_DIRNAME / f"{child.run_id}.json"):
                payload, error = R.read_json_file(path)
                if error is not None:
                    failures.append(f"reader saw {path.name}: {error}")
                elif payload is not None and "run_id" not in payload:
                    failures.append(f"reader saw a partial {path.name}")

    threads = [threading.Thread(target=fn) for fn in (write_root, write_child, read_both)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert failures == []
    # Both files survived, each still describing its own writer's run.
    assert root_store.read_last() is None
    final_root = R.record_from_dict(json.loads(root_store.active_path.read_text()))
    assert final_root.run_id == root_store.run_id
    assert final_root.pid == 1000 + iterations - 1
    final_child = root_store.read_child(child.run_id)
    assert final_child is not None and final_child.pid == 2000 + iterations - 1


# --------------------------------------------------------------------------------------------- #
# 6. the sequence (section 3.2) and finalization ordering (section 3.3)
# --------------------------------------------------------------------------------------------- #

def test_the_root_publishes_current_child_before_the_child_exists(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    store = R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111")
    store.write_active(root_record(tmp_path))
    child = child_record(tmp_path)

    store.set_current_child(summary_of(child))
    snapshot = R.read_runtime(dep, lease_probe=occupied)
    assert snapshot.record.current_child.run_id == child.run_id
    assert snapshot.children == ()          # nothing written by the child yet

    R.RecordStore(dep, run_id=child.run_id, role=T.ActivityRole.CHILD).write_child(child)
    snapshot = R.read_runtime(dep, lease_probe=occupied)
    assert [c.run_id for c in snapshot.children] == [child.run_id]


def test_normal_child_completion_moves_current_child_to_last_child(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    store = R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111")
    store.write_active(root_record(tmp_path))
    child = child_record(tmp_path)
    store.set_current_child(summary_of(child))

    store.promote_current_child()
    record = R.read_runtime(dep, lease_probe=occupied).record
    assert record.current_child is None
    assert record.last_child.run_id == child.run_id
    store.promote_current_child()  # idempotent with nothing current
    assert R.read_runtime(dep, lease_probe=occupied).record.last_child.run_id == child.run_id


def test_a_root_cannot_finalize_gracefully_while_an_approved_child_is_alive(tmp_path: Path):
    """Gate 5. active.json is never removed under a live child."""
    dep = deployment_dir(tmp_path)
    store = R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111")
    store.write_active(root_record(tmp_path))
    child = child_record(tmp_path, state=T.RunState.RUNNING)
    store.set_current_child(summary_of(child))
    R.RecordStore(dep, run_id=child.run_id, role=T.ActivityRole.CHILD).write_child(child)

    terminal = root_record(tmp_path, state=T.RunState.DONE, finished_at=LATER,
                           current_child=summary_of(child))
    with pytest.raises(T.RecordError, match="gate 5"):
        store.finalize(terminal)
    assert store.active_path.exists()
    assert not store.last_path.exists()

    # The caller's own list of live children blocks it too, before the child has written anything.
    with pytest.raises(T.RecordError, match="gate 5"):
        store.finalize(terminal, live_children=["a-child-that-has-not-reported-yet"])


def test_finalization_succeeds_once_the_child_is_terminal(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    store = R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111")
    store.write_active(root_record(tmp_path))
    child = child_record(tmp_path)
    store.set_current_child(summary_of(child))
    child_store = R.RecordStore(dep, run_id=child.run_id, role=T.ActivityRole.CHILD)
    child_store.write_child(child)
    child_store.write_child(child_record(tmp_path, state=T.RunState.DONE, finished_at=LATER))
    store.promote_current_child()

    terminal = root_record(tmp_path, state=T.RunState.DONE, finished_at=LATER,
                           last_child=summary_of(child))
    store.finalize(terminal)

    assert not store.active_path.exists()          # removed AFTER last.json is durable
    assert store.read_last().state is T.RunState.DONE
    assert R.read_runtime(dep, lease_probe=free).classification is T.RecordClassification.ABSENT


def test_last_json_holds_roots_only(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    child = child_record(tmp_path, state=T.RunState.DONE, finished_at=LATER)
    store = R.RecordStore(dep, run_id=child.run_id)   # a child's id, but a ROOT store
    with pytest.raises(T.WriterOwnershipError, match="roots only"):
        store.finalize(child)


def test_a_hard_killed_root_leaves_its_record_in_place(tmp_path: Path):
    """Gate 4, record side: the stale record is the only description of what is running."""
    dep = deployment_dir(tmp_path)
    store = R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111")
    store.write_active(root_record(tmp_path))
    child = child_record(tmp_path)
    store.set_current_child(summary_of(child))
    R.RecordStore(dep, run_id=child.run_id, role=T.ActivityRole.CHILD).write_child(child)

    # The root is gone; the child still holds the inherited lease, so the deployment is occupied.
    snapshot = R.read_runtime(dep, lease_probe=occupied)
    assert snapshot.active is True
    assert snapshot.classification is T.RecordClassification.LIVE
    assert snapshot.record.run_id == store.run_id

    # And nobody may recover it while that child keeps the lease held elsewhere.
    with pytest.raises(T.WriterOwnershipError):
        R.recover_abandoned(dep, lease=FakeLease(dep, held=False))
    assert store.active_path.exists()


# --------------------------------------------------------------------------------------------- #
# 7. the recovery writer
# --------------------------------------------------------------------------------------------- #

def _really_dead_pid() -> int:
    """A PID that certainly is not running: a real child, killed with SIGKILL and reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)
    return proc.pid


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL is POSIX; the record logic is platform-free")
def test_recovery_after_a_hard_kill_finalizes_the_stale_record(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    dead = _really_dead_pid()
    store = R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111")
    store.write_active(root_record(tmp_path, pid=dead, state=T.RunState.RUNNING))

    # With the lease now free, the record classifies as abandoned rather than live.
    assert R.read_runtime(dep, lease_probe=free).classification is T.RecordClassification.ABANDONED

    recovered = R.recover_abandoned(dep, lease=FakeLease(dep))
    assert recovered is not None
    assert recovered.state is T.RunState.INTERRUPTED
    assert recovered.finished_at is not None
    assert "abandoned" in (recovered.error or "")
    assert not store.active_path.exists()
    assert store.read_last().run_id == store.run_id
    # The dead PID is preserved for diagnostics and never signaled.
    assert store.read_last().pid == dead


def test_recovery_requires_the_activity_lock(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111").write_active(
        root_record(tmp_path)
    )
    with pytest.raises(T.WriterOwnershipError, match="activity lease"):
        R.recover_abandoned(dep, lease=FakeLease(dep, held=False))


def test_a_lease_authorizes_recovery_of_its_own_deployment_only(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    other = tmp_path / "runtime" / "other"
    other.mkdir(parents=True)
    with pytest.raises(T.WriterOwnershipError, match="own deployment"):
        R.recover_abandoned(dep, lease=FakeLease(other))


def test_recovery_of_a_staged_run_with_no_snapshot_finalizes_as_failed(tmp_path: Path):
    """Gate 25, first half: null archive path, and the GUI says recovery data is unavailable."""
    dep = deployment_dir(tmp_path)
    R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111").write_active(
        root_record(tmp_path, project=staged_project(tmp_path, status=T.ArchiveStatus.PENDING))
    )
    recovered = R.recover_abandoned(dep, lease=FakeLease(dep))
    assert recovered.project.archive_status is T.ArchiveStatus.FAILED
    assert recovered.project.archived_db_path is None
    assert recovered.project.archive_error


def test_recovery_of_a_staged_run_with_a_good_snapshot_finalizes_as_stale(tmp_path: Path):
    """Gate 25, second half: the last good generation is named, not discarded."""
    dep = deployment_dir(tmp_path)
    generation = str(tmp_path / "persistent" / "acer_rubrum" / "archive.gen.1.sqlite")
    R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111").write_active(
        root_record(tmp_path, project=staged_project(tmp_path, status=T.ArchiveStatus.READY,
                                                     archived=generation))
    )
    recovered = R.recover_abandoned(dep, lease=FakeLease(dep))
    assert recovered.project.archive_status is T.ArchiveStatus.STALE
    assert recovered.project.archived_db_path == generation
    assert recovered.project.archive_error


def test_recovery_with_nothing_to_recover_is_a_no_op(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    assert R.recover_abandoned(dep, lease=FakeLease(dep)) is None
    assert not (dep / T.LAST_RECORD_FILENAME).exists()


def test_recovery_republishes_a_record_caught_between_finalize_and_remove(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111").write_active(
        root_record(tmp_path, state=T.RunState.DONE, finished_at=LATER)
    )
    recovered = R.recover_abandoned(dep, lease=FakeLease(dep))
    assert recovered.state is T.RunState.DONE
    assert not (dep / T.ACTIVE_RECORD_FILENAME).exists()
    assert (dep / T.LAST_RECORD_FILENAME).exists()


@pytest.mark.parametrize("payload, kind", [("{not json", "unreadable"), ('{"schema_version": 99}',
                                                                        "incompatible")])
def test_an_uninterpretable_record_is_quarantined_not_deleted(tmp_path: Path, payload, kind):
    """Gate 50: preserve the stale record, then permit a new root under the cleanup lock."""
    dep = deployment_dir(tmp_path)
    active = dep / T.ACTIVE_RECORD_FILENAME
    active.write_text(payload)

    assert R.recover_abandoned(dep, lease=FakeLease(dep)) is None
    assert not active.exists()
    quarantined = [p for p in dep.iterdir() if kind in p.name]
    assert len(quarantined) == 1
    assert quarantined[0].read_text() == payload
    assert not (dep / T.LAST_RECORD_FILENAME).exists()

    # A new root may now start.
    R.RecordStore(dep, run_id="new-root").write_active(
        root_record(tmp_path, run_id="new-root")
    )
    assert R.read_runtime(dep, lease_probe=occupied).record.run_id == "new-root"


# --------------------------------------------------------------------------------------------- #
# 8. the compatibility reader (section 2.9)
# --------------------------------------------------------------------------------------------- #

def test_a_newer_record_blocks_start_without_granting_control(tmp_path: Path):
    """Gate 50: occupancy still comes from the lock; nothing in the record is interpreted."""
    dep = deployment_dir(tmp_path)
    payload = R.record_to_dict(root_record(tmp_path))
    payload["schema_version"] = T.SCHEMA_VERSION + 3
    payload["some_future_field"] = {"stop_me": True}
    (dep / T.ACTIVE_RECORD_FILENAME).write_text(json.dumps(payload))

    snapshot = R.read_runtime(dep, lease_probe=occupied)
    assert snapshot.active is True
    assert snapshot.compatible is False
    assert snapshot.classification is T.RecordClassification.INCOMPATIBLE
    assert snapshot.record is None                      # no field was interpreted
    assert snapshot.raw["some_future_field"] == {"stop_me": True}
    assert snapshot.schema_version == T.SCHEMA_VERSION + 3
    assert "newer LM3" in snapshot.message


def test_a_newer_record_over_a_free_lease_is_incompatible_but_not_active(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    payload = R.record_to_dict(root_record(tmp_path))
    payload["schema_version"] = T.SCHEMA_VERSION + 1
    (dep / T.ACTIVE_RECORD_FILENAME).write_text(json.dumps(payload))
    snapshot = R.read_runtime(dep, lease_probe=free)
    assert snapshot.active is False and snapshot.compatible is False


def test_a_held_lease_with_no_record_is_occupied_but_unidentifiable(tmp_path: Path):
    snapshot = R.read_runtime(deployment_dir(tmp_path), lease_probe=occupied)
    assert snapshot.active is True
    assert snapshot.classification is T.RecordClassification.ABSENT
    assert "no active record" in snapshot.message


def test_a_malformed_record_is_unreadable_and_still_occupied(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    (dep / T.ACTIVE_RECORD_FILENAME).write_text("{ truncated")
    snapshot = R.read_runtime(dep, lease_probe=occupied)
    assert snapshot.active is True
    assert snapshot.compatible is True
    assert snapshot.classification is T.RecordClassification.UNREADABLE
    assert snapshot.record is None and snapshot.message


def test_an_invalid_record_is_reported_rather_than_trusted(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    payload = R.record_to_dict(root_record(tmp_path))
    payload.pop("project")                       # a pipeline without its project block
    (dep / T.ACTIVE_RECORD_FILENAME).write_text(json.dumps(payload))
    snapshot = R.read_runtime(dep, lease_probe=occupied)
    assert snapshot.classification is T.RecordClassification.UNREADABLE
    assert snapshot.record is None
    assert "project" in snapshot.message


@pytest.mark.parametrize(
    "state, lease, expected",
    [
        (T.RunState.RUNNING, True, T.RecordClassification.LIVE),
        (T.RunState.RUNNING, False, T.RecordClassification.ABANDONED),
        (T.RunState.DONE, True, T.RecordClassification.FINALIZED),
        (T.RunState.DONE, False, T.RecordClassification.FINALIZED),
    ],
)
def test_classification_takes_occupancy_from_the_lock(tmp_path: Path, state, lease, expected):
    record = root_record(tmp_path, state=state,
                         finished_at=LATER if state in T.TERMINAL_STATES else None)
    assert R.classify_record(record, lease_occupied=lease) is expected


def test_classification_of_an_absent_or_unreadable_record(tmp_path: Path):
    assert R.classify_record(None, lease_occupied=True) is T.RecordClassification.ABSENT
    assert (R.classify_record(None, lease_occupied=True, parse_error="boom")
            is T.RecordClassification.UNREADABLE)
    assert (R.classify_record(None, lease_occupied=True, parse_error="boom",
                              schema_version=T.SCHEMA_VERSION + 1)
            is T.RecordClassification.INCOMPATIBLE)


def test_read_runtime_bounds_a_record_it_reads(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    payload = R.record_to_dict(root_record(tmp_path))
    payload["error"] = "x" * (T.MAX_ERROR_CHARS * 4)
    (dep / T.ACTIVE_RECORD_FILENAME).write_text(json.dumps(payload))
    snapshot = R.read_runtime(dep, lease_probe=occupied)
    assert snapshot.record is not None
    assert len(snapshot.record.error) == T.MAX_ERROR_CHARS


def test_read_last_bounds_and_redacts_a_record_it_reads(tmp_path: Path):
    """``read_last`` is caller-facing, so it owes the same section 3.2 guarantee as the other two.

    Every writer of ``last.json`` in this build sanitizes, but the file may have been hand-edited or
    written by an older build, and "no bearer tokens ... a record is a bounded description" is a
    property of what a reader *hands out*, not only of what a writer stores.
    """
    dep = deployment_dir(tmp_path)
    payload = R.record_to_dict(root_record(tmp_path, state=T.RunState.DONE, finished_at=LATER))
    payload["error"] = "Authorization: Bearer sk-abcdefghijklmnop " + "x" * (T.MAX_ERROR_CHARS * 4)
    (dep / T.LAST_RECORD_FILENAME).write_text(json.dumps(payload))
    store = R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111")
    record = store.read_last()
    assert record is not None
    assert len(record.error) == T.MAX_ERROR_CHARS
    assert "sk-abcdefghijklmnop" not in record.error


def test_read_last_is_still_none_when_the_file_is_absent(tmp_path: Path):
    # sanitize_record() would raise AttributeError on the absent-file None, which the method's
    # except clause does not catch -- the guard is what keeps "absent" a quiet None.
    store = R.RecordStore(deployment_dir(tmp_path), run_id="11111111-1111-4111-8111-111111111111")
    assert store.read_last() is None


def test_read_active_uses_the_real_lease_probe(tmp_path: Path):
    pytest.importorskip("leafmachine3.core.runtime.lease")
    dep = deployment_dir(tmp_path)
    store = R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111")
    store.write_active(root_record(tmp_path))
    snapshot = store.read_active()
    assert snapshot.record.run_id == store.run_id
    assert snapshot.active is False          # nothing holds this throwaway deployment's lease


# --------------------------------------------------------------------------------------------- #
# 9. retention (section 3.2)
# --------------------------------------------------------------------------------------------- #

def write_child_file(dep: Path, run_id: str, *, age_s: float = 0.0) -> Path:
    path = dep / T.CHILDREN_DIRNAME / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))
    return path


def write_grant_file(dep: Path, run_id: str, *, expires_at: float, consumed: bool = False) -> Path:
    suffix = T.GRANT_CONSUMED_SUFFIX if consumed else T.GRANT_SUFFIX
    path = dep / T.CHILDREN_DIRNAME / f"{run_id}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"child_run_id": run_id, "expires_at": expires_at}))
    return path


def test_pruning_preserves_everything_the_root_records_reference(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    store = R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111")
    current = child_record(tmp_path, run_id="current-child")
    store.write_active(root_record(tmp_path, current_child=summary_of(current),
                                   last_child=summary_of(child_record(tmp_path,
                                                                      run_id="last-child"))))
    kept = [write_child_file(dep, name, age_s=10_000)
            for name in ("current-child", "last-child")]
    swept = [write_child_file(dep, f"old-{index}", age_s=10_000 + index) for index in range(5)]

    removed = R.prune_children(dep, retention=0)
    assert removed == len(swept)
    assert all(path.exists() for path in kept)
    assert not any(path.exists() for path in swept)


def test_pruning_keeps_the_newest_records_up_to_the_retention_limit(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    files = [write_child_file(dep, f"child-{index}", age_s=index * 60) for index in range(10)]
    assert R.prune_children(dep, retention=4) == 6
    assert [path.exists() for path in files] == [True] * 4 + [False] * 6


def test_the_default_retention_is_the_last_fifty_children(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    for index in range(60):
        write_child_file(dep, f"child-{index:03d}", age_s=index * 60)
    R.prune_children(dep)
    assert len(list((dep / T.CHILDREN_DIRNAME).iterdir())) == T.DEFAULT_CHILD_RETENTION


def test_pruning_removes_expired_grants_and_keeps_live_ones(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    now = time.time()
    expired = write_grant_file(dep, "expired-child", expires_at=now - 1)
    live = write_grant_file(dep, "approved-child", expires_at=now + 600)
    unreadable = dep / T.CHILDREN_DIRNAME / f"garbage{T.GRANT_SUFFIX}"
    unreadable.write_text("{not json")

    R.prune_children(dep, retention=0, clock=lambda: now)
    assert not expired.exists()
    # An approved child that has not started yet is referenced by nothing -- and must survive.
    assert live.exists()
    assert not unreadable.exists()


def test_pruning_sweeps_consumed_grants_beyond_the_limit(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    now = time.time()
    old = [write_grant_file(dep, f"c{index}", expires_at=now + 600, consumed=True)
           for index in range(4)]
    for index, path in enumerate(old):
        stamp = now - 1000 * (index + 1)
        os.utime(path, (stamp, stamp))
    R.prune_children(dep, retention=2, clock=lambda: now)
    assert [path.exists() for path in old] == [True, True, False, False]


def test_finalization_prunes_and_keeps_its_own_run(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    store = R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111")
    store.write_active(root_record(tmp_path))
    oldest = [write_child_file(dep, f"old-{index:03d}", age_s=99_999 + index)
              for index in range(T.DEFAULT_CHILD_RETENTION + 5)]
    mine = write_child_file(dep, store.run_id)

    store.finalize(root_record(tmp_path, state=T.RunState.DONE, finished_at=LATER))
    assert mine.exists()                                 # referenced by last.json's run_id
    assert not any(path.exists() for path in oldest[-5:])
    assert len(list((dep / T.CHILDREN_DIRNAME).iterdir())) == T.DEFAULT_CHILD_RETENTION


def test_recovery_also_prunes(tmp_path: Path):
    dep = deployment_dir(tmp_path)
    R.RecordStore(dep, run_id="11111111-1111-4111-8111-111111111111").write_active(
        root_record(tmp_path)
    )
    oldest = [write_child_file(dep, f"old-{index:03d}", age_s=99_999 + index)
              for index in range(T.DEFAULT_CHILD_RETENTION + 5)]
    R.recover_abandoned(dep, lease=FakeLease(dep))
    assert not any(path.exists() for path in oldest[-5:])
    assert len(list((dep / T.CHILDREN_DIRNAME).iterdir())) == T.DEFAULT_CHILD_RETENTION


def test_pruning_an_absent_children_directory_is_a_no_op(tmp_path: Path):
    assert R.prune_children(tmp_path / "nothing-here") == 0


# --------------------------------------------------------------------------------------------- #
# 10. small helpers
# --------------------------------------------------------------------------------------------- #

def test_utc_now_matches_the_records_wire_format():
    # The plan's example record pairs "2026-08-28T12:00:00Z" with process_started_at
    # 1787932800.25; those are two different clocks (the ISO field is the activity's start, the
    # float is the OS process creation time), so this pins the FORMAT against a computed instant.
    assert R.utc_now(lambda: 1787918400.0) == "2026-08-28T12:00:00Z"
    assert R.utc_now(lambda: 1787932800.25) == "2026-08-28T16:00:00Z"
    assert R._TIMESTAMP_RE.match(R.utc_now())


def test_new_run_id_is_unique():
    assert len({R.new_run_id() for _ in range(100)}) == 100


def test_process_start_time_is_positive_for_this_process_and_zero_for_a_dead_one():
    assert R.process_start_time() > 0
    assert R.process_start_time(os.getpid()) == R.process_start_time()
    assert R.process_start_time(999_999_999) == 0.0
