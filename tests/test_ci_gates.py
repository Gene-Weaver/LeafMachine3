"""Negative controls for the two CI gates.

A gate is only worth having if it FAILS on the thing it exists to catch. Both of these were written
after review found them passing builds they should have rejected:

- the test gate greped only ``FAILED``, so a brand-new ``ERROR`` outcome was invisible, and it
  swallowed pytest's real exit status through a pipe;
- the lint ratchet compared whole ``"<count> <rule> <file>"`` lines, so FIXING one finding of two
  looked like new debt and failed the build.

Every case below is driven through the real checker entry points.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.ci import check_lint_ratchet as lint  # noqa: E402
from tools.ci import check_test_baseline as gate  # noqa: E402

BASELINE_NODES = {
    "tests/test_pipeline_mock.py::test_end_to_end_mock_pipeline",
    "tests/test_settings_ui.py::test_no_metadata_for_settings_that_do_not_exist",
}
MIN_EXECUTED = 980
BASELINE_ONLY_LOG = (
    "FAILED tests/test_pipeline_mock.py::test_end_to_end_mock_pipeline\n"
    "FAILED tests/test_settings_ui.py::test_no_metadata_for_settings_that_do_not_exist\n"
    "2 failed, 1004 passed, 6 skipped in 60.00s\n"
)


def ruff_findings_or_skip() -> tuple[str, int]:
    """Run ruff over the tree, or skip if it is not installed in THIS interpreter.

    ``python -m ruff`` with no ruff exits **1** -- the same code ruff itself uses for "found
    something" -- so an exit-code check alone cannot tell "ruff reported findings" from "there is no
    ruff here", and the caller ends up parsing an empty stdout as though it were a clean tree. The
    interpreter-specific case is real: CI's `test` extra provides ruff, a bare 3.10 venv does not.
    """
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--output-format=json", "leafmachine3", "tests"],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode not in lint.OK_EXIT_CODES or not proc.stdout.lstrip().startswith("["):
        pytest.skip(f"ruff is not available to {sys.executable} (exit {proc.returncode})")
    return proc.stdout, proc.returncode


# --------------------------------------------------------------------------------------------- #
# The test baseline gate
# --------------------------------------------------------------------------------------------- #

def test_the_baseline_only_run_passes() -> None:
    assert gate.check(BASELINE_ONLY_LOG, 1, BASELINE_NODES) == []


def test_a_new_failing_node_id_fails() -> None:
    log = BASELINE_ONLY_LOG.replace(
        "2 failed", "3 failed").replace(
        "FAILED tests/test_settings_ui.py",
        "FAILED tests/test_brand_new.py::test_regression\nFAILED tests/test_settings_ui.py")
    problems = gate.check(log, 1, BASELINE_NODES)
    assert any("new failing test node IDs" in p and "test_brand_new" in p for p in problems)


def test_a_new_error_outcome_fails() -> None:
    """The exact negative control review used: a baseline FAILED plus a brand-new ERROR.

    A fixture, setup or teardown that raises reports ``ERROR <node>``, never ``FAILED <node>``, so a
    gate that greps only FAILED waves it through.
    """
    log = (
        "FAILED tests/test_pipeline_mock.py::test_end_to_end_mock_pipeline\n"
        "ERROR tests/test_brand_new.py::test_fixture_exploded\n"
        "1 failed, 970 passed, 1 error in 60.00s\n"
    )
    problems = gate.check(log, 1, BASELINE_NODES)
    assert any("ERROR outcomes are never acceptable" in p for p in problems), problems
    assert any("test_fixture_exploded" in p for p in problems)


def test_application_log_noise_is_not_mistaken_for_a_node_id() -> None:
    """Found by running the gate against a real suite, not by construction.

    LM3's stage executor prints ``FAILED ect ...`` on a stage error, and pytest interleaves captured
    output with its own short summary. A ``^FAILED\\s+(\\S+)`` pattern reported ``ect`` as a brand-new
    failing node id and failed a perfectly good build. Only ``.py`` paths count.
    """
    log = (
        "ect: sub-item failed on specimen 1: ModuleNotFoundError(\"No module named 'ect'\")\n"
        "FAILED ect\n"
        "FAILED tests/test_pipeline_mock.py::test_end_to_end_mock_pipeline\n"
        "1 failed, 970 passed in 60.00s\n"
    )
    assert gate.check(log, 1, BASELINE_NODES) == [], "log noise was parsed as a node id"


def test_a_real_collection_error_without_a_node_is_still_caught() -> None:
    """Collection errors have no ``::`` part -- ``ERROR tests/test_x.py`` -- and must still fail."""
    log = ("ERROR tests/test_broken.py\n"
           "1 failed, 900 passed, 1 error in 60.00s\n"
           "FAILED tests/test_pipeline_mock.py::test_end_to_end_mock_pipeline\n")
    problems = gate.check(log, 1, BASELINE_NODES)
    assert any("test_broken.py" in p for p in problems), problems


def test_a_collection_failure_fails_rather_than_passing_on_an_empty_diff() -> None:
    """No FAILED lines at all is not the same as nothing broken."""
    log = "ImportError while loading conftest: No module named 'fastapi'\n"
    problems = gate.check(log, 4, BASELINE_NODES)
    assert any("no result summary" in p for p in problems), problems
    assert any("exited 4" in p for p in problems)


@pytest.mark.parametrize("status", [2, 3, 4, 5])
def test_every_non_ok_exit_code_fails(status: int) -> None:
    """2 interrupted, 3 internal error, 4 usage, 5 nothing collected. Only 0 and 1 are permitted."""
    problems = gate.check(BASELINE_ONLY_LOG, status, BASELINE_NODES)
    assert any(f"exited {status}" in p for p in problems), problems


def test_an_internal_error_with_a_partial_summary_still_fails() -> None:
    """Exit 3 can still print something summary-shaped; the status is what settles it."""
    log = "INTERNALERROR> RuntimeError: boom\n1 failed, 12 passed\n"
    assert gate.check(log, 3, BASELINE_NODES), "an internal error was accepted"


def test_a_newly_passing_baseline_test_does_not_fail_the_gate() -> None:
    log = ("FAILED tests/test_pipeline_mock.py::test_end_to_end_mock_pipeline\n"
           "1 failed, 971 passed in 60.00s\n")
    assert gate.check(log, 1, BASELINE_NODES) == []


def test_a_collection_failure_does_not_report_the_whole_baseline_as_newly_passing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that collected nothing has no FAILED lines, so every baseline node looks "fixed"."""
    log = tmp_path / "pytest.log"
    log.write_text("INTERNALERROR> RuntimeError: plugin conflict\n", encoding="utf-8")
    rc = gate.main(["--log", str(log), "--status", "3",
                    "--baseline", "docs/verification/STEP2_BASELINE_NODE_IDS.txt"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "newly PASSING" not in out, "a collection failure was reported as fixing the whole baseline"


def test_the_real_baseline_file_parses_to_the_expected_node_ids() -> None:
    """The parser must return exactly the node IDs the file lists -- no more, no fewer.

    This pinned a literal count (11), which meant every legitimate fix that removed an ID failed
    this test. What it guards is the PARSER: the expected set is re-derived here independently
    (every non-comment line that looks like a node ID), so a parser that silently drops or invents
    entries still fails, and an empty baseline -- the strictest gate -- is allowed.
    """
    text = Path("docs/verification/STEP2_BASELINE_NODE_IDS.txt").read_text(encoding="utf-8")
    nodes = gate.parse_baseline(text)
    expected = {ln.strip() for ln in text.splitlines()
                if ln.strip().startswith("tests/") and "::" in ln}
    assert set(nodes) == expected
    assert all(n.startswith("tests/") and "::" in n for n in nodes)


# --------------------------------------------------------------------------------------------- #
# The lint ratchet
# --------------------------------------------------------------------------------------------- #

def ruff_json(rows: list[tuple[str, str]]) -> str:
    return json.dumps([{"code": code, "filename": path} for code, path in rows])


def test_equal_debt_passes() -> None:
    current = lint.parse_ruff_json(ruff_json([("F401", "a.py"), ("F401", "a.py")]))
    baseline = lint.parse_baseline("2 F401 a.py\n")
    assert lint.check(current, baseline) == ([], [])


def test_a_DECREASE_passes_and_is_reported(  # noqa: N802 - the bug this exists for
) -> None:
    """The bug review found: 2 -> 1 read as new debt because the count was part of the key."""
    current = lint.parse_ruff_json(ruff_json([("F401", "a.py")]))
    baseline = lint.parse_baseline("2 F401 a.py\n")
    regressions, improvements = lint.check(current, baseline)
    assert regressions == [], f"fixing one of two findings was reported as new debt: {regressions}"
    assert improvements == ["a.py: F401 2 -> 1"]


def test_an_increase_fails() -> None:
    current = lint.parse_ruff_json(ruff_json([("F401", "a.py")] * 3))
    baseline = lint.parse_baseline("2 F401 a.py\n")
    regressions, _ = lint.check(current, baseline)
    assert regressions == ["a.py: F401 x3 (was 2)"]


def test_a_new_rule_for_an_existing_file_fails() -> None:
    current = lint.parse_ruff_json(ruff_json([("F401", "a.py"), ("E702", "a.py")]))
    baseline = lint.parse_baseline("1 F401 a.py\n")
    regressions, _ = lint.check(current, baseline)
    assert any("E702" in r and "NEW rule" in r for r in regressions), regressions


def test_a_finding_in_a_brand_new_file_fails() -> None:
    current = lint.parse_ruff_json(ruff_json([("F401", "brand_new.py")]))
    regressions, _ = lint.check(current, lint.parse_baseline(""))
    assert any("brand_new.py" in r for r in regressions)


@pytest.mark.parametrize("status", [2, 127])
def test_a_ruff_that_could_not_run_fails(status: int, tmp_path: Path) -> None:
    """`|| true` means a broken tool yields an empty finding set that looks like a clean tree."""
    blob = tmp_path / "ruff.json"
    blob.write_text("[]", encoding="utf-8")
    rc = lint.main(["--json", str(blob), "--status", str(status),
                    "--baseline", "docs/verification/RUFF_BASELINE.txt"])
    assert rc == 1, "a ruff invocation that failed to execute was treated as a clean run"


def test_unparseable_ruff_output_fails(tmp_path: Path) -> None:
    blob = tmp_path / "ruff.json"
    blob.write_text("not json at all", encoding="utf-8")
    rc = lint.main(["--json", str(blob), "--status", "1",
                    "--baseline", "docs/verification/RUFF_BASELINE.txt"])
    assert rc == 1


def test_a_malformed_baseline_fails_rather_than_accepting_everything(tmp_path: Path) -> None:
    blob = tmp_path / "ruff.json"
    blob.write_text(ruff_json([("F401", "a.py")]), encoding="utf-8")
    bad = tmp_path / "baseline.txt"
    bad.write_text("this is not a baseline line\n", encoding="utf-8")
    assert lint.main(["--json", str(blob), "--status", "1", "--baseline", str(bad)]) == 1


def test_the_real_repository_is_at_its_baseline() -> None:
    """The committed baseline must actually describe this tree, or the ratchet is theatre."""
    stdout, _ = ruff_findings_or_skip()
    current = lint.parse_ruff_json(stdout)
    baseline = lint.parse_baseline(
        Path("docs/verification/RUFF_BASELINE.txt").read_text(encoding="utf-8"))
    regressions, _ = lint.check(current, baseline)
    assert regressions == [], f"the tree has lint debt the baseline does not record: {regressions}"


# --------------------------------------------------------------------------------------------- #
# The coverage floor -- volume, not identity
# --------------------------------------------------------------------------------------------- #
# Everything above protects failure IDENTITIES. None of it protects how many tests RAN. Review
# demonstrated the hole directly: check("6 skipped in 0.01s", 0, set()) returned [].

def test_an_entirely_skipped_suite_fails_the_floor() -> None:
    problems = gate.check("6 skipped in 0.01s", 0, BASELINE_NODES, MIN_EXECUTED)
    assert any("only 0 tests executed" in p for p in problems), problems


def test_a_suite_reduced_to_one_passing_test_fails_the_floor() -> None:
    """A broken conftest or a stray marker expression can do this, and it looks perfectly green."""
    problems = gate.check("1 passed in 0.10s", 0, BASELINE_NODES, MIN_EXECUTED)
    assert any("only 1 tests executed" in p for p in problems), problems


def test_a_major_drop_in_executed_tests_fails_the_floor() -> None:
    """~1014 executing today; half of them vanishing must not pass as 'no new failures'."""
    log = ("FAILED tests/test_pipeline_mock.py::test_end_to_end_mock_pipeline\n"
           "1 failed, 499 passed, 6 skipped in 30.00s\n")
    problems = gate.check(log, 1, BASELINE_NODES, MIN_EXECUTED)
    assert any("only 500 tests executed" in p for p in problems), problems
    assert not any("new failing" in p for p in problems), "the identity check should be satisfied"


def test_only_the_final_terminal_summary_counts() -> None:
    """A nested pytest run's summary must not inflate the parent's executed count.

    Review's negative control: a child run reporting 500 passed, then the real final summary of
    ``11 failed, 503 passed``. Summing every ``N passed``/``N failed`` phrase anywhere in the log
    reported 1014 executed for a run that executed 514 -- enough to falsely clear the 980 floor
    while half the suite had vanished. A nested invocation, captured subprocess output, or a test
    whose own assertion data is summary-shaped all produce this.
    """
    log = "500 passed in child run\n11 failed, 503 passed in 5s\n"
    assert gate.executed_count(log) == 514, "a child run's counts leaked into the parent's total"
    problems = gate.check(log, 1, set(), MIN_EXECUTED)
    assert any("only 514 tests executed" in p for p in problems), problems


def test_summary_shaped_text_earlier_in_the_log_is_ignored() -> None:
    """This very file contains strings like "1 failed, 970 passed" as test data."""
    log = ('assert "1 failed, 970 passed, 6 skipped in 60.00s" in captured\n'
           "11 failed, 1003 passed, 6 skipped in 71.54s (0:01:11)\n")
    assert gate.executed_count(log) == 1014


def test_a_decorated_summary_line_is_recognized() -> None:
    """Without -q pytest wraps the line in `=` padding."""
    assert gate.executed_count("========== 5 passed in 0.10s ==========") == 5


def test_a_log_with_no_terminal_summary_counts_nothing_and_is_rejected() -> None:
    log = "ImportError while loading conftest\n"
    assert gate.executed_count(log) == 0
    assert any("no result summary" in p for p in gate.check(log, 1, set(), MIN_EXECUTED))


def test_skips_do_not_count_toward_the_floor() -> None:
    """A test that skipped ran nothing; counting it would let an all-skipped suite pass."""
    assert gate.executed_count("11 failed, 1003 passed, 6 skipped in 70.00s") == 1014
    assert gate.executed_count("980 passed, 500 skipped in 10.00s") == 980


def test_the_normal_run_clears_the_floor() -> None:
    assert gate.check(BASELINE_ONLY_LOG, 1, BASELINE_NODES, MIN_EXECUTED) == []


def test_the_committed_floor_is_declared_and_below_the_current_suite_size() -> None:
    text = Path("docs/verification/STEP2_BASELINE_NODE_IDS.txt").read_text(encoding="utf-8")
    floor = gate.parse_min_executed(text)
    assert floor >= 900, f"the committed floor {floor} is too low to catch catastrophic loss"


# --------------------------------------------------------------------------------------------- #
# Baseline regeneration must be idempotent
# --------------------------------------------------------------------------------------------- #

def test_rewriting_the_lint_baseline_is_idempotent(tmp_path: Path) -> None:
    """The first implementation split on a delimiter the file does not contain, so it preserved the
    WHOLE file -- findings included -- and appended the new ones, doubling every count and quietly
    loosening the ratchet on the next run."""
    findings = [("F401", "a.py"), ("F401", "a.py"), ("E702", "b.py")]
    blob = tmp_path / "ruff.json"
    blob.write_text(ruff_json(findings), encoding="utf-8")
    baseline = tmp_path / "RUFF_BASELINE.txt"
    baseline.write_text("# a header comment\n# spanning two lines\n2 F401 a.py\n1 E702 b.py\n",
                        encoding="utf-8")

    args = ["--json", str(blob), "--status", "1", "--baseline", str(baseline), "--write-baseline"]
    assert lint.main(args) == 0
    first = baseline.read_text(encoding="utf-8")
    assert lint.main(args) == 0
    second = baseline.read_text(encoding="utf-8")

    assert first == second, "rewriting a rewritten baseline changed it"
    assert first.startswith("# a header comment\n# spanning two lines\n"), "the header was lost"
    assert sum(lint.parse_baseline(first).values()) == len(findings), (
        f"counts were duplicated: {first!r}")
    assert lint.check(lint.parse_ruff_json(ruff_json(findings)), lint.parse_baseline(first)) == ([], [])


def test_rewriting_the_real_baseline_round_trips_to_the_same_total(tmp_path: Path) -> None:
    stdout, returncode = ruff_findings_or_skip()
    real = Path("docs/verification/RUFF_BASELINE.txt")
    total_before = sum(lint.parse_baseline(real.read_text(encoding="utf-8")).values())

    copy = tmp_path / "RUFF_BASELINE.txt"
    copy.write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    blob = tmp_path / "ruff.json"
    blob.write_text(stdout, encoding="utf-8")
    args = ["--json", str(blob), "--status", str(returncode),
            "--baseline", str(copy), "--write-baseline"]
    assert lint.main(args) == 0
    once = copy.read_text(encoding="utf-8")
    assert lint.main(args) == 0
    assert copy.read_text(encoding="utf-8") == once, "the real baseline does not round-trip"
    assert sum(lint.parse_baseline(once).values()) == total_before == 79
