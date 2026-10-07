#!/usr/bin/env python3
"""Ratchet ruff findings against a committed baseline, keyed by (rule, file).

The tree carries pre-existing lint debt this refactor did not create and will not fix wholesale.
``ruff check ... || true`` is decoration rather than a gate, and a blocking full-tree check would be
red on arrival and switched off within a week. So debt may stay level or fall, never rise.

Two failure modes in the obvious line-based `comm` implementation, both guarded here:

1. **A decrease looks like new debt.** Comparing whole ``"<count> <rule> <file>"`` lines treats the
   COUNT as part of the identity, so ``2 F401 x.py`` -> ``1 F401 x.py`` reads as "a line vanished and
   a new one appeared" and CI fails you for FIXING something. The key is (rule, file); the count is
   the value being compared.
2. **A broken tool passes.** If ruff cannot execute, or its output cannot be parsed, a pipeline
   ending in ``|| true`` yields an empty finding set that looks like a perfectly clean tree.

Usage:
    ruff check --output-format=json leafmachine3 tests > ruff.json; status=$?
    python tools/ci/check_lint_ratchet.py --json ruff.json --status "$status" \
        --baseline docs/verification/RUFF_BASELINE.txt [--write-baseline]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

#: ruff exits 0 when clean and 1 when it found something. Anything else is the tool failing.
OK_EXIT_CODES = frozenset({0, 1})


def parse_ruff_json(text: str) -> Counter:
    """``Counter[(rule, file)]`` from ruff's JSON output. Raises ValueError on anything unparseable."""
    try:
        rows = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ruff output is not valid JSON: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError(f"ruff JSON should be a list of findings, got {type(rows).__name__}")
    counts: Counter = Counter()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"ruff finding is not an object: {row!r}")
        code = row.get("code") or "UNKNOWN"
        filename = row.get("filename")
        if not filename:
            raise ValueError(f"ruff finding has no filename: {row!r}")
        counts[(code, os.path.relpath(filename))] += 1
    return counts


def parse_baseline(text: str) -> Counter:
    """``Counter[(rule, file)]`` from the committed ``<count> <rule> <file>`` baseline."""
    counts: Counter = Counter()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"malformed baseline line (expected '<count> <rule> <file>'): {line!r}")
        count, rule, filename = parts
        counts[(rule, filename)] += int(count)
    return counts


def render(counts: Counter) -> str:
    return "".join(f"{n} {rule} {path}\n" for (rule, path), n in sorted(counts.items(), key=lambda kv: (kv[0][1], kv[0][0])))


def header_of(path: Path) -> str:
    """The leading comment block of an existing baseline, or the default header.

    Rewriting must be IDEMPOTENT. The first attempt split the file on a blank-line delimiter that
    does not occur in it, so it preserved the WHOLE file -- findings included -- and appended the new
    ones, doubling every count and quietly loosening the ratchet on the next run. Taking exactly the
    leading ``#`` lines has no such failure mode: rewriting a rewritten file is a fixed point.
    """
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        header = []
        for line in lines:
            if line.startswith("#") or not line.strip():
                header.append(line)
            else:
                break
        if header:
            return "\n".join(header).rstrip("\n") + "\n"
    return DEFAULT_HEADER


DEFAULT_HEADER = (
    "# Ruff debt baseline: \"<count> <rule> <file>\", sorted by file.\n"
    "# Regenerate with: python tools/ci/check_lint_ratchet.py --json ruff.json --status <rc> \\\n"
    "#                    --baseline docs/verification/RUFF_BASELINE.txt --write-baseline\n"
)


def check(current: Counter, baseline: Counter) -> tuple[list[str], list[str]]:
    """Return (regressions, improvements) as human-readable lines."""
    regressions, improvements = [], []
    for key in sorted(set(current) | set(baseline), key=lambda k: (k[1], k[0])):
        rule, path = key
        now, was = current.get(key, 0), baseline.get(key, 0)
        if now > was:
            label = "NEW rule for this file" if was == 0 else f"was {was}"
            regressions.append(f"{path}: {rule} x{now} ({label})")
        elif now < was:
            improvements.append(f"{path}: {rule} {was} -> {now}")
    return regressions, improvements


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, type=Path, help="ruff --output-format=json output")
    ap.add_argument("--status", required=True, type=int, help="ruff's exit code")
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--write-baseline", action="store_true", help="rewrite the baseline from the current findings")
    args = ap.parse_args(argv)

    if args.status not in OK_EXIT_CODES:
        print(f"::error::ruff exited {args.status}; it could not run, so its findings mean nothing")
        return 1
    try:
        current = parse_ruff_json(args.json.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        print(f"::error::could not read ruff findings: {exc}")
        return 1

    if args.write_baseline:
        args.baseline.write_text(header_of(args.baseline) + render(current), encoding="utf-8")
        print(f"baseline rewritten: {sum(current.values())} findings")
        return 0

    try:
        baseline = parse_baseline(args.baseline.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        print(f"::error::could not read the lint baseline: {exc}")
        return 1

    regressions, improvements = check(current, baseline)
    if improvements:
        print("lint debt REMOVED (thank you; regenerate the baseline deliberately):")
        for line in improvements:
            print(f"  {line}")
    if regressions:
        print("lint debt ADDED:")
        for line in regressions:
            print(f"  {line}")
        print("::error::new lint findings; fix them or regenerate the baseline deliberately")
        return 1
    print(f"OK: {sum(current.values())} findings, none new (baseline {sum(baseline.values())}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
