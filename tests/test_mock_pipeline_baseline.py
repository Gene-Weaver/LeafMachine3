"""Frozen pre-refactor baseline for the deterministic CLI mock pipeline.

This module exists to make plan §8 gate 60 -- "the deterministic mock pipeline's DB rows,
resume behavior, and output inventory match the pre-refactor baseline" -- *checkable*. A gate
that names a baseline is unsatisfiable until that baseline is recorded, so this records it:
one canonical, order-stable fingerprint of a full ``machine3()`` mock run, stored in
``tests/golden/mock_pipeline_baseline.json`` and re-compared on every later run.

Why the config is frozen HERE instead of reused from ``tests/conftest.py``
-------------------------------------------------------------------------
A baseline whose *inputs* can drift is not a baseline. ``conftest.build_mock_config`` is shared,
edited freely, and is expected to keep growing as modules land; if the golden were taken against
it, a later edit there would silently redefine what the golden means and gate 60 would compare
two different experiments. So the settings tree, the synthetic images, and the run name are all
pinned in this file. Nothing outside this module can move the baseline without a visible diff.

Why ``modules.ect.enabled`` is FALSE in the baseline config
-----------------------------------------------------------
``tests/test_pipeline_mock.py`` fails in this interpreter with ``ModuleNotFoundError: No module
named 'ect'`` -- the third-party ECT package is not installed. It *is* installable from PyPI
(``ect`` 1.3.0 resolves and downloads cleanly here), so option (a) was available. It was
rejected: installing it mutates the shared interpreter that the rest of the suite -- and any
concurrent work -- runs in, and it would pin the golden to an unpinned third party's numeric
output, so a future ``ect`` release would fail gate 60 for a reason that has nothing to do with
the runtime refactor. Option (b) is taken instead: the baseline disables the ``ect`` module, which
needs no environment change at all and is therefore re-runnable here today and after any
reinstall. The ``ect`` stage still appears in ``project_status`` (marked complete-with-no-work by
``run_pipeline``), so the *ledger shape* is still covered; only the ECT numerics are out of scope.

What the fingerprint deliberately excludes
------------------------------------------
Timestamps (``ingested_at``, ``started_at``, ``finished_at``, ``updated_at``, ``orig_mtime``),
absolute paths (``original_path``, ``working_path``, ``crop_path``, ...), and the raw
``specimen.specimen_id`` VALUES. That last one is measured, not assumed: two back-to-back runs of
this exact config assign ``specimen_id`` 1 and 2 to opposite stems, because ingest fans images out
across workers. So specimen rows are keyed by ``image_stem`` and the id column is captured only as
the sorted *set* of ids that were handed out. Everything else that a run produces -- per-stage
project status, the per-(specimen x stage) ledger, every table's row count, the report manifest,
and the full relative-path inventory of the run directory -- is byte-stable and is captured.

SQLite's ``-wal`` / ``-shm`` sidecars are dropped from the inventory: whether they exist at the
end of a run depends on when the last checkpoint happened, not on pipeline behavior.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from leafmachine3.machine3 import machine3

# Bump when the fingerprint FORMAT changes (new section, different normalization). A format bump
# invalidates the golden on purpose; a behavior change must NOT be papered over with one.
FINGERPRINT_VERSION = 1

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "mock_pipeline_baseline.json"

#: Files the Step 3 runtime adds to a run directory that the pre-refactor pipeline never produced.
#: Plan section 3.4 makes the launch manifest MANDATORY, so this is an intended delta -- but it is
#: named here rather than folded into the golden, so the gate keeps its teeth: with the runtime
#: enabled the inventory must be the golden PLUS EXACTLY THIS, and any other new file still fails.
RUNTIME_V2_ADDED_FILES: frozenset[str] = frozenset({"logs/run_manifest.json"})

RUN_NAME = "baseline"
N_SPECIMENS = 2

# specimen columns that are legitimately nondeterministic or environment-bound (see module
# docstring). Everything else in the table is part of the baseline.
_VOLATILE_SPECIMEN_COLS = frozenset(
    {"specimen_id", "original_path", "working_path", "orig_mtime", "ingested_at"}
)

# Working scratch that a resumed run legitimately re-populates under FRESH detection ids, leaving
# the interrupted run's tiles orphaned beside the new ones. Measured, not guessed: after a ledger
# wipe + re-run the crops come back as det7/det10 rather than det1/det4. These are internal
# working dirs, not run outputs, so a resume may ADD paths here and nowhere else.
_RESUME_SCRATCH_PREFIXES = ("_ruler_squarify/", "_ruler_cf_lattice/")


# --------------------------------------------------------------------------------------------
# the frozen experiment: images + settings
# --------------------------------------------------------------------------------------------
def _make_specimen_image(path: Path, seed: int) -> Path:
    """Write one deterministic synthetic specimen JPEG (a green 'leaf' on a pale sheet).

    Pinned here rather than imported so the baseline's *pixels* cannot drift out from under the
    golden. ``default_rng(seed)`` makes the noise reproducible, and cv2's JPEG encoder is
    deterministic for a fixed quality, so the file bytes (and therefore ``orig_size_bytes``)
    repeat exactly.
    """
    rng = np.random.default_rng(seed)
    h, w = 900, 700
    img = np.full((h, w, 3), 232, dtype=np.uint8)                       # pale herbarium sheet
    cv2.ellipse(img, (int(0.35 * w), int(0.45 * h)), (140, 220), 25, 0, 360, (40, 130, 45), -1)
    cv2.rectangle(img, (40, h - 70), (w - 40, h - 40), (60, 60, 60), -1)        # a 'ruler' strip
    cv2.rectangle(img, (int(0.62 * w), int(0.12 * h)), (int(0.92 * w), int(0.34 * h)),
                  (250, 250, 240), -1)                                          # a 'label'
    img = cv2.subtract(img, rng.integers(0, 12, size=(h, w, 3), dtype=np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def _baseline_config(images_dir: Path, output_dir: Path) -> dict:
    """The pinned mock settings tree. ``modules.ect.enabled`` is False -- see module docstring."""
    return {
        "version": 3,
        "project": {
            "run_name": RUN_NAME,
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
            "archival_detector": {"enabled": True,
                                  "classes": ["Ruler", "Barcode", "Colorcard", "Label"]},
            "plant_detector": {"enabled": True,
                               "classes": ["Leaf_WHOLE", "Leaf_PARTIAL", "Seed_Fruit_ONE"]},
            "specimen_segmenter": {"enabled": True, "paperclean": True},
            "phenology_detector": {
                "enabled": True,
                "targets": {
                    "leaves": {"min_conf": 0.3, "min_count": 1},
                    "flowers": {"min_conf": 0.4, "min_count": 1},
                    "fruits": {"min_conf": 0.4, "min_count": 1},
                },
            },
            "ruler_classifier": {"enabled": True, "models_dir": "models/ruler_classifier",
                                 "ensemble_members": ["a", "b", "c"], "min_conf": 0.35},
            "ruler_cf": {"enabled": True},
            "leaf_segmenter": {"enabled": True, "include_partial": False},
            "morphology": {"enabled": True, "classes": ["Leaf"], "find_minimum_bounding_box": True},
            "landmark_detector": {"enabled": True, "source_classes": ["Leaf_WHOLE"],
                                  "include_partial": False},
            "landmark_measurements": {"enabled": True, "min_kpt_conf": 0.25},
            "leaf_orientation": {"enabled": True, "min_kpt_conf": 0.25, "min_midvein": 5},
            "petiole_width": {"enabled": True, "min_kpt_conf": 0.25, "touch_dist_px": 20},
            "metric_grounding": {"enabled": True, "round_ndigits": 4},
            "reporter": {"enabled": True},
            # OFF by decision: the third-party `ect` package is absent from this interpreter and
            # installing it is not a prerequisite this baseline is allowed to impose.
            "ect": {"enabled": False, "num_dirs": 64, "radial_viz": True, "cartesian_viz": True,
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
                "Binary_Masks__Specimen_Inverse": True,
                "RGB_Masks__Specimen_Inverse": True,
                "inverse_fill": [255, 0, 0],
            },
            "crops": {"enabled": True, "classes": "all"},
            "overlay_petiole": {"enabled": True},
            "overlay_specimen": {"enabled": True},
            "leaf_products": {"enabled": True, "original": True, "oriented": True,
                              "background": "black"},
            "formats": {"image_ext": "jpg", "jpg_quality": 95, "mask_ext": "png"},
        },
    }


def _stage_run(workspace: Path) -> Path:
    """Materialize one isolated run workspace and return the settings path to hand ``machine3``.

    Each workspace gets its OWN cwd because ``hardware_setup.HW_PATH`` is still a bare relative
    path (the very split §3.1 exists to fix): the profile lands in whatever directory the process
    happens to be in. Giving run A and run B separate cwds means each tunes its own profile, so
    the determinism assertion below is comparing two genuinely independent runs rather than two
    runs that shared one cached profile.
    """
    images = workspace / "input_images"
    for i in range(N_SPECIMENS):
        _make_specimen_image(images / f"specimen_{i:02d}.jpg", seed=i + 1)
    cwd = workspace / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    cfg_path = cwd / "LM3_settings.yaml"
    cfg_path.write_text(
        yaml.safe_dump(_baseline_config(images, workspace / "out"), sort_keys=False),
        encoding="utf-8",
    )
    return cfg_path


# --------------------------------------------------------------------------------------------
# the fingerprint
# --------------------------------------------------------------------------------------------
def _round(value: Any) -> Any:
    """Round REAL columns so a last-bit float difference cannot masquerade as a behavior change."""
    return round(value, 6) if isinstance(value, float) else value


def _inventory(root: Path) -> list[str]:
    """Sorted relative-path inventory of ``root``; directories carry a trailing '/'.

    Directories are included because an EMPTY directory the pipeline creates (``logs/``, for one)
    is part of the output contract and would otherwise be invisible. The SQLite WAL sidecars are
    dropped -- their existence is a checkpoint-timing artifact, not behavior.
    """
    out: list[str] = []
    for path in root.rglob("*"):
        name = path.name
        if name.endswith("-wal") or name.endswith("-shm"):
            continue
        rel = path.relative_to(root).as_posix()
        # is_dir() on a symlink-to-file is False, so `_working/*.jpg` stays a file entry.
        out.append(rel + "/" if path.is_dir() else rel)
    return sorted(out)


def _fingerprint(db_path: Path, root: Path) -> dict:
    """Canonical, order-stable description of one finished run."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name")]
        row_counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}

        # stem is the stable specimen identity; specimen_id is assignment order (see docstring)
        stem_of = {r["specimen_id"]: r["image_stem"]
                   for r in conn.execute("SELECT specimen_id, image_stem FROM specimen")}

        specimen_cols = [c[1] for c in conn.execute("PRAGMA table_info(specimen)")
                         if c[1] not in _VOLATILE_SPECIMEN_COLS]
        specimens = {}
        for row in conn.execute("SELECT * FROM specimen"):
            specimens[row["image_stem"]] = {c: _round(row[c]) for c in specimen_cols}

        project_status = [
            [r["stage_key"], r["stage_order"], r["state"], r["n_total"], r["n_done"]]
            for r in conn.execute(
                "SELECT stage_key, stage_order, state, n_total, n_done FROM project_status "
                "ORDER BY stage_order, stage_key")
        ]
        image_status = sorted(
            [stem_of[r["specimen_id"]], r["stage_key"], r["state"], r["no_work"]]
            for r in conn.execute("SELECT specimen_id, stage_key, state, no_work FROM image_status")
        )
        manifest_kinds: dict[str, int] = {}
        for r in conn.execute("SELECT kind FROM report_manifest"):
            manifest_kinds[r["kind"]] = manifest_kinds.get(r["kind"], 0) + 1
    finally:
        conn.close()

    inventory = _inventory(root)
    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "specimen_ids": sorted(stem_of),
        "specimens": specimens,
        "project_status": project_status,
        "image_status": image_status,
        "table_row_counts": row_counts,
        "report_manifest_kinds": manifest_kinds,
        "inventory": inventory,
        "reports_inventory": [p for p in inventory if p.startswith("reports/")],
    }


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _explain(a: dict, b: dict, label_a: str, label_b: str) -> str:
    """Section-by-section diff, so a failure names WHAT moved instead of dumping two blobs."""
    lines = []
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key), b.get(key)
        if va == vb:
            continue
        lines.append(f"section {key!r} differs:")
        if isinstance(va, list) and isinstance(vb, list):
            only_a = [x for x in va if x not in vb][:10]
            only_b = [x for x in vb if x not in va][:10]
            lines.append(f"  only in {label_a}: {only_a}")
            lines.append(f"  only in {label_b}: {only_b}")
        elif isinstance(va, dict) and isinstance(vb, dict):
            for k in sorted(set(va) | set(vb)):
                if va.get(k) != vb.get(k):
                    lines.append(f"  {k}: {label_a}={va.get(k)!r} {label_b}={vb.get(k)!r}")
        else:
            lines.append(f"  {label_a}={va!r} {label_b}={vb!r}")
    return "\n".join(lines) or "(no section-level difference found)"


def _run_and_fingerprint(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run the pinned mock pipeline end to end in ``workspace`` and fingerprint the result."""
    cfg_path = _stage_run(workspace)
    monkeypatch.chdir(cfg_path.parent)          # hardware_settings.yaml lands here, not in the repo
    project = machine3(cfg_path)
    assert project.dirs.db_path.exists(), "the mock run produced no SQLite database"
    return _fingerprint(project.dirs.db_path, project.dirs.root)


# --------------------------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------------------------
def test_baseline_is_reproducible_and_matches_the_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§8 gate 60: record (once) and then defend the pre-refactor mock-pipeline baseline.

    The two independent runs come FIRST and on purpose. Writing a golden from a single run would
    hand the project a gate that fails at random the first time an unordered query or a worker
    race shifts something; proving byte-equality between two fresh runs before the golden is ever
    written is what makes this gate trustworthy.
    """
    first = _run_and_fingerprint(tmp_path / "run_a", monkeypatch)
    second = _run_and_fingerprint(tmp_path / "run_b", monkeypatch)
    assert first == second, (
        "the mock pipeline is NOT deterministic across two identical runs -- refusing to record a "
        "flaky baseline:\n" + _explain(first, second, "run_a", "run_b"))

    if not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(_canonical({"_meta": _META, "fingerprint": first}), encoding="utf-8")
        pytest.fail(
            f"recorded a NEW mock-pipeline baseline at {GOLDEN_PATH} -- review and commit it, then "
            "re-run; this only happens when the golden is missing")

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert golden["fingerprint"]["fingerprint_version"] == FINGERPRINT_VERSION, (
        f"golden was recorded at fingerprint format v{golden['fingerprint']['fingerprint_version']} "
        f"but this module produces v{FINGERPRINT_VERSION}; re-record deliberately")
    assert first == golden["fingerprint"], (
        "the mock pipeline no longer matches the recorded pre-refactor baseline "
        f"({GOLDEN_PATH.name}):\n" + _explain(golden["fingerprint"], first, "golden", "now"))


def _expected_with_runtime_v2(golden_fingerprint: dict) -> dict:
    """The golden, adjusted for the files the Step 3 runtime legitimately adds."""
    expected = json.loads(json.dumps(golden_fingerprint))          # deep copy
    expected["inventory"] = sorted(set(expected["inventory"]) | RUNTIME_V2_ADDED_FILES)
    return expected


def test_the_baseline_still_holds_with_the_runtime_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate 60 on the path it actually exists to protect.

    Running this only with ``LM3_RUNTIME_V2`` off proves nothing: off is the configuration in which
    nothing changed. The comparison that matters is that turning the runtime ON leaves the DB rows,
    the resume behavior and the output inventory identical EXCEPT for the one file plan section 3.4
    requires -- so a stray artifact, a renamed report or a changed ledger row is still caught.
    """
    if not GOLDEN_PATH.exists():
        pytest.skip("no golden recorded yet; the flag-off test records it first")
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["fingerprint"]

    monkeypatch.setenv("LM3_RUNTIME_V2", "1")
    got = _run_and_fingerprint(tmp_path / "run_v2", monkeypatch)

    manifest_only = set(got["inventory"]) - set(golden["inventory"])
    assert manifest_only == set(RUNTIME_V2_ADDED_FILES), (
        "enabling the runtime changed the output inventory by more than the mandatory launch "
        f"manifest; unexpected additions: {sorted(manifest_only - RUNTIME_V2_ADDED_FILES)}, "
        f"missing: {sorted(RUNTIME_V2_ADDED_FILES - manifest_only)}")
    assert got == _expected_with_runtime_v2(golden), (
        "with the runtime enabled the mock pipeline diverges from the baseline beyond the launch "
        "manifest:\n" + _explain(_expected_with_runtime_v2(golden), got, "golden+manifest", "now"))


def test_resume_after_an_interrupt_adds_no_rows_and_reaches_the_same_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§8 gate 60, resume half: an interrupted run re-converges on the identical ledger.

    The interrupt is simulated the way a SIGKILL actually presents itself to the next invocation:
    every stage left ``running`` (which ``ProjectDB.reclaim_running`` demotes to ``pending``) and
    the per-image ledger gone, so every stage re-collects every specimen and every ``persist``
    runs a second time over rows that already exist. If any stage inserted instead of
    replacing-per-specimen, the row counts would grow here.
    """
    workspace = tmp_path / "run"
    cfg_path = _stage_run(workspace)
    monkeypatch.chdir(cfg_path.parent)

    project = machine3(cfg_path)
    db_path, root = project.dirs.db_path, project.dirs.root
    before = _fingerprint(db_path, root)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("UPDATE project_status SET state = 'running', n_done = 0")
        conn.execute("DELETE FROM image_status")
        conn.commit()
    finally:
        conn.close()

    resumed = machine3(cfg_path)
    assert resumed.dirs.db_path == db_path, "the resumed run opened a different database"
    after = _fingerprint(db_path, root)

    # 1. no duplicate rows anywhere -- the whole point of the resume contract
    assert after["table_row_counts"] == before["table_row_counts"], (
        "resume changed row counts:\n"
        + _explain({"table_row_counts": before["table_row_counts"]},
                   {"table_row_counts": after["table_row_counts"]}, "before", "after"))

    # 2. the ledger and the specimen table land exactly where they were
    for section in ("specimen_ids", "specimens", "project_status", "image_status",
                    "report_manifest_kinds", "reports_inventory"):
        assert after[section] == before[section], (
            f"resume changed {section}:\n"
            + _explain({section: before[section]}, {section: after[section]}, "before", "after"))

    # 3. ...and the terminal fingerprint equals the recorded baseline, not merely itself
    if GOLDEN_PATH.exists():
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["fingerprint"]
        for section in ("specimens", "project_status", "image_status", "table_row_counts",
                        "report_manifest_kinds", "reports_inventory"):
            assert after[section] == golden[section], (
                f"resumed run does not match the golden baseline in {section}:\n"
                + _explain({section: golden[section]}, {section: after[section]}, "golden", "now"))

    # 4. The run-root inventory may only GROW, and only under working scratch. A resumed run
    #    re-crops the rulers under fresh detection ids, so the interrupted run's squarify tiles and
    #    lattice rasters are left orphaned beside the new ones. That is recorded pre-refactor
    #    behavior; anything appearing OUTSIDE those scratch dirs, or any baseline path going
    #    missing, is not.
    missing = sorted(set(before["inventory"]) - set(after["inventory"]))
    assert not missing, f"resume deleted outputs the first run produced: {missing[:20]}"
    extra = sorted(set(after["inventory"]) - set(before["inventory"]))
    stray = [p for p in extra if not p.startswith(_RESUME_SCRATCH_PREFIXES)]
    assert not stray, f"resume produced new outputs outside working scratch: {stray[:20]}"


_META = {
    "gate": "plan §8 gate 60 -- the deterministic mock pipeline's DB rows, resume behavior, and "
            "output inventory match the pre-refactor baseline",
    "plan_step": "§4 Step 1 -- Characterization, isolation, and path unification",
    "recorded_by": "tests/test_mock_pipeline_baseline.py",
    "ect_module": "disabled -- the third-party `ect` package is not installed in this interpreter; "
                  "the baseline must not require installing it (see the module docstring)",
    "excluded_from_fingerprint": [
        "timestamps (ingested_at, started_at, finished_at, updated_at, orig_mtime)",
        "absolute paths (original_path, working_path, crop_path, ...)",
        "specimen_id VALUES -- ingest assigns them in worker-completion order, measured to swap "
        "between runs; captured only as the sorted id set, with rows keyed by image_stem",
        "SQLite -wal / -shm sidecars, whose presence is checkpoint timing",
    ],
    "n_specimens": N_SPECIMENS,
    "run_name": RUN_NAME,
}
