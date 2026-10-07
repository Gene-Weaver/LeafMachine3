#!/usr/bin/env python3
"""Gate a pytest run against the persisted baseline of already-failing node IDs.

The suite has known pre-existing failures, so a bare ``pytest -q`` can only ever be red. What CI must
answer is not "did anything fail" but "did anything fail that was not ALREADY failing".

Four ways a naive version of this gate lets a broken build through, all of them guarded here:

1. **ERROR outcomes are invisible.** A fixture that raises reports ``ERROR <node>``, not
   ``FAILED <node>``. A gate that greps only ``FAILED`` passes a run with a brand-new error in it.
   There are no baseline errors, so ANY error fails.
2. **The exit status is swallowed.** With ``| tee`` the pipeline reports tee's status. pytest exits
   2 (interrupted), 3 (internal error), 4 (usage) and 5 (no tests collected) -- and 3 in particular
   can still print a partial summary that looks parseable. Only 0 and 1 are permitted.
3. **A collection failure produces no FAILED lines at all**, so the node-id diff is empty and the
   gate "passes" on a run that executed nothing.
3b. **Nothing above protects coverage VOLUME.** ``6 skipped in 0.01s`` has no failures, no errors, a
   valid summary and exit 0 -- so an entirely skipped suite, or one reduced to a handful of tests by
   a broken conftest or a stray marker expression, sails through an identity-only gate. Hence the
   ``min-executed:`` floor declared in the baseline file.
4. **Counting instead of identifying.** A count comparison hides a swap in which one test starts
   failing as another starts passing.

Usage:
    pytest tests/ -q -rfE | tee pytest.log; status=${PIPESTATUS[0]}
    python tools/ci/check_test_baseline.py --log pytest.log --status "$status" \
        --baseline docs/verification/STEP2_BASELINE_NODE_IDS.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: pytest exit codes that a baseline-gated run may legitimately produce.
#: 0 = all passed, 1 = some tests failed (which the baseline diff then judges).
#: 2 interrupted, 3 internal error, 4 usage error, 5 no tests collected are all hard failures.
OK_EXIT_CODES = frozenset({0, 1})

#: pytest's short-summary lines are "FAILED <path>.py[::node] - message" / "ERROR <path>.py[::node]".
#: The token MUST look like a test path, because application logging also emits lines beginning with
#: FAILED: LM3's stage executor prints "FAILED ect ..." on a stage error, and a looser `^FAILED\s+(\S+)`
#: happily reported `ect` as a brand-new failing node id. Requiring a `.py` path is what separates
#: pytest's summary from anything else that happens to start with the same word.
_NODE = r"((?:[^\s:]+/)*[^\s:]+\.py(?:::\S+)?)"
_FAILED = re.compile(r"^FAILED\s+" + _NODE, re.M)
_ERROR = re.compile(r"^ERROR\s+" + _NODE, re.M)
#: pytest's TERMINAL SUMMARY line -- the last thing it prints -- e.g.
#:     "11 failed, 1003 passed, 6 skipped in 71.54s (0:01:11)"
#:     "===== 5 passed in 0.10s ====="
#: The trailing "in <duration>s" is the discriminator. Matching outcome counts ANYWHERE in the log
#: instead was a real defect: a nested pytest invocation, captured subprocess output, or a test whose
#: own fixture data is summary-shaped all contribute counts that were never executed by this run.
#: Demonstrated: "500 passed in child run\n11 failed, 503 passed in 5s" reported 1014 executed when
#: the run had executed 514, which would falsely clear the coverage floor.
_TERMINAL_SUMMARY = re.compile(
    r"^=*\s*(?P<counts>\d+\s+\w+(?:\s*,\s*\d+\s+\w+)*)\s+in\s+[\d.]+\s*s\b.*$", re.M)
#: Individual outcome counts WITHIN that line. "Executed" is passed + failed: a skipped test ran
#: nothing, so counting skips would let an all-skipped suite satisfy a coverage floor.
_COUNT = re.compile(r"\b(\d+)\s+(passed|failed)\b")


def terminal_summary(log: str) -> str | None:
    """The LAST pytest terminal-summary line in ``log``, or None if there is none.

    Last, not first: a nested run's summary appears before the parent's, and the parent's is the
    final thing pytest writes.
    """
    matches = list(_TERMINAL_SUMMARY.finditer(log))
    return matches[-1].group("counts") if matches else None
#: Directive line in the baseline file naming the minimum number of tests that must EXECUTE.
_MIN_EXECUTED = re.compile(r"^min-executed:\s*(\d+)\s*$", re.M)


def executed_count(log: str) -> int:
    """passed + failed from pytest's FINAL terminal-summary line. Skips/deselects do not count."""
    counts = terminal_summary(log)
    return sum(int(n) for n, _ in _COUNT.findall(counts)) if counts else 0


def parse_min_executed(text: str) -> int:
    """The committed coverage floor, or 0 when the baseline does not declare one."""
    match = _MIN_EXECUTED.search(text)
    return int(match.group(1)) if match else 0


def parse_baseline(text: str) -> set[str]:
    """Node IDs from the baseline file: lines starting with ``tests/`` and containing ``::``."""
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("tests/") and "::" in line and not line.lstrip().startswith("#")
    }


def check(log: str, status: int, baseline: set[str], min_executed: int = 0) -> list[str]:
    """Return a list of human-readable problems. Empty means the gate passes."""
    problems: list[str] = []

    if status not in OK_EXIT_CODES:
        problems.append(
            f"pytest exited {status}; only {sorted(OK_EXIT_CODES)} are permitted "
            f"(2=interrupted, 3=internal error, 4=usage error, 5=nothing collected)"
        )

    if terminal_summary(log) is None:
        problems.append(
            "pytest printed no result summary -- collection almost certainly failed, so an empty "
            "failure diff below would be meaningless"
        )

    errors = sorted(set(_ERROR.findall(log)))
    if errors:
        problems.append(
            "ERROR outcomes are never acceptable (the baseline contains none); a fixture, setup or "
            "teardown raised in: " + ", ".join(errors)
        )

    failed = set(_FAILED.findall(log))
    new = sorted(failed - baseline)
    if new:
        problems.append("new failing test node IDs: " + ", ".join(new))

    # A coverage FLOOR, because everything above protects failure IDENTITIES and none of it protects
    # VOLUME. `check("6 skipped in 0.01s", 0, baseline)` passed every check above: an entirely
    # skipped suite has no failures, no errors, a valid summary and a clean exit. So can a suite
    # accidentally reduced to one passing test by a broken conftest, a bad marker expression or a
    # collection filter. Those are catastrophic and completely invisible to an identity diff.
    if min_executed:
        executed = executed_count(log)
        if executed < min_executed:
            problems.append(
                f"only {executed} tests executed (passed+failed); the committed floor is "
                f"{min_executed}. Either tests disappeared, or the suite shrank deliberately and "
                f"`min-executed:` in the baseline file must be updated to match"
            )

    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, type=Path, help="captured pytest output")
    ap.add_argument("--status", required=True, type=int, help="pytest's real exit code (PIPESTATUS[0])")
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--min-executed", type=int, default=None,
                    help="override the `min-executed:` floor declared in the baseline file")
    args = ap.parse_args(argv)

    log = args.log.read_text(encoding="utf-8", errors="replace")
    baseline_text = args.baseline.read_text(encoding="utf-8")
    baseline = parse_baseline(baseline_text)
    floor = args.min_executed if args.min_executed is not None else parse_min_executed(baseline_text)
    problems = check(log, args.status, baseline, floor)

    # Only meaningful on a run that actually executed. A collection failure produces no FAILED lines
    # at all, which would otherwise render the ENTIRE baseline as "newly passing" -- exactly the
    # false reassurance this gate exists to prevent.
    if not problems:
        fixed = sorted(baseline - set(_FAILED.findall(log)))
        if fixed:
            print("newly PASSING (thank you; update the baseline deliberately):")
            for node in fixed:
                print(f"  {node}")

    if problems:
        for p in problems:
            print(f"::error::{p}")
        return 1
    print(f"OK: {executed_count(log)} tests executed (floor {floor}), no new failing node IDs, "
          f"no errors, pytest exited {args.status}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
