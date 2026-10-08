#!/usr/bin/env python3
"""Bump the one LM3 version and re-stamp everything that derives from it.

``VERSION`` (repo root, beside ``uv.lock``) is the single place the version is written. This script
is how it changes. The scheme is ``MAJOR.MINOR.PATCH`` where PATCH carries a 4-digit commit counter:
every commit adds one (3.0.1 -> 3.0.1001 -> 3.0.1002 ...); ``--patch`` moves to the next thousand
(3.0.1002 -> 3.0.2000) and ``--minor`` moves the second number (-> 3.1.0), both only when Will calls
a revision big enough. MAJOR is reserved. After writing VERSION it brings the derived artifacts into step:

* ``app/package.json`` and ``app/package-lock.json`` -- the Electron shell's version;
* ``leafmachine3/_env_contract.json`` and ``app/desktop-contract.json`` -- via
  tools/release/write_env_contract.py, which also records the sha256 of ``uv.lock`` and of
  ``leafmachine3/modelhub/models.lock.yaml`` plus every Hub model revision, so one LM3 version pins
  exactly one dependency lock and one set of models.

``pyproject.toml`` needs no edit: it reads VERSION at build time, and ``leafmachine3.__version__``
reads the same file in a checkout. ``--check`` verifies all of the above agree (the test suite runs
it too), so a commit that forgot the bump, or re-locked dependencies without one, fails CI.

Usage::

    python tools/release/bump_version.py            # 3.0.1001 -> 3.0.1002  (every commit; 3.0.1 -> 3.0.1001)
    python tools/release/bump_version.py --patch    # 3.0.1002 -> 3.0.2000  (a bigger revision)
    python tools/release/bump_version.py --minor    # 3.0.2000 -> 3.1.0     (a major-enough revision)
    python tools/release/bump_version.py --set 3.2.0
    python tools/release/bump_version.py --check    # exit 1 when anything disagrees
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import write_env_contract  # noqa: E402

VERSION_FILE = ROOT / "VERSION"
PACKAGE_JSON = ROOT / "app" / "package.json"
PACKAGE_LOCK = ROOT / "app" / "package-lock.json"
CONTRACT = ROOT / "leafmachine3" / "_env_contract.json"
DESKTOP_CONTRACT = ROOT / "app" / "desktop-contract.json"
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def read_version() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip()


COMMIT_FIELD = 1000   # PATCH = <patch digit> * 1000 + commit counter (3 digits per thousand)


def bumped(current: str, *, minor: bool = False, patch: bool = False) -> str:
    m = SEMVER.match(current)
    if not m:
        raise SystemExit(f"VERSION is not MAJOR.MINOR.PATCH: {current!r}")
    major, mnr, pt = (int(x) for x in m.groups())
    if minor:
        return f"{major}.{mnr + 1}.0"
    if patch:   # 3.0.1 -> 3.0.2000, 3.0.1002 -> 3.0.2000
        digit = pt if pt < COMMIT_FIELD else pt // COMMIT_FIELD
        return f"{major}.{mnr}.{(digit + 1) * COMMIT_FIELD}"
    # the per-commit step: a bare patch digit (3.0.1) first opens its 4-digit field (3.0.1001)
    nxt = pt * COMMIT_FIELD + 1 if pt < COMMIT_FIELD else pt + 1
    if nxt % COMMIT_FIELD == 0:
        raise SystemExit(f"{current}: the commit counter is full; use --patch to move to the next thousand")
    return f"{major}.{mnr}.{nxt}"


def _rewrite_json_version(path: Path, version: str) -> None:
    """Replace the top-level ``"version"`` (and package-lock's ``packages[""].version``) in place,
    touching nothing else so the file's formatting -- and npm's own ordering -- survives."""
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(r'^(\s*"version":\s*")[^"]*(")', rf"\g<1>{version}\g<2>", text, count=2 if path.name == "package-lock.json" else 1, flags=re.M)
    if n == 0:
        raise SystemExit(f"{path}: no \"version\" field to rewrite")
    path.write_text(new, encoding="utf-8")


def sync(version: str) -> None:
    VERSION_FILE.write_text(version + "\n", encoding="utf-8")
    _rewrite_json_version(PACKAGE_JSON, version)
    _rewrite_json_version(PACKAGE_LOCK, version)
    write_env_contract.main([])            # re-stamps lm3_version + both lock hashes


def check() -> list[str]:
    """Every derived artifact agrees with VERSION and with the locks as committed."""
    problems: list[str] = []
    version = read_version()
    if not SEMVER.match(version):
        problems.append(f"VERSION {version!r} is not MAJOR.MINOR.PATCH")
    for path in (PACKAGE_JSON, PACKAGE_LOCK):
        got = json.loads(path.read_text(encoding="utf-8")).get("version")
        if got != version:
            problems.append(f"{path.relative_to(ROOT)} says {got}, VERSION says {version}")
    lock_pkg = json.loads(PACKAGE_LOCK.read_text(encoding="utf-8")).get("packages", {}).get("", {}).get("version")
    if lock_pkg != version:
        problems.append(f"app/package-lock.json packages[\"\"].version says {lock_pkg}, VERSION says {version}")
    for path in (CONTRACT, DESKTOP_CONTRACT):
        got = json.loads(path.read_text(encoding="utf-8")).get("lm3_version")
        if got != version:
            problems.append(f"{path.relative_to(ROOT)} says {got}, VERSION says {version}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    locks = contract.get("locks") or {}
    for key, path in (("uv_lock_sha256", write_env_contract.UV_LOCK), ("models_lock_sha256", write_env_contract.MODELS_LOCK)):
        now = hashlib.sha256(path.read_bytes()).hexdigest()
        if locks.get(key) != now:
            problems.append(f"{path.relative_to(ROOT)} changed since the contract was written "
                            f"(contract {str(locks.get(key))[:12]}, file {now[:12]}): bump the version")
    if write_env_contract.main(["--check"]) != 0:
        problems.append("the env contract is stale: run tools/release/bump_version.py")
    return problems


def summary() -> str:
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    locks = c.get("locks") or {}
    return (f"LM3 {c.get('lm3_version')} · uv.lock {str(locks.get('uv_lock_sha256'))[:12]} · "
            f"models.lock {str(locks.get('models_lock_sha256'))[:12]} ({len(locks.get('hub_revisions') or {})} Hub revisions)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--minor", action="store_true", help="bump the second number (a major-enough revision)")
    g.add_argument("--patch", action="store_true", help="move to the next thousand in the third field (a bigger revision)")
    g.add_argument("--set", metavar="X.Y.Z", help="write this exact version")
    g.add_argument("--check", action="store_true", help="verify VERSION and every derived artifact agree")
    a = ap.parse_args(argv)
    if a.check:
        problems = check()
        for p in problems:
            print(p, file=sys.stderr)
        print(summary() if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    current = read_version()
    new = a.set if a.set else bumped(current, minor=a.minor, patch=a.patch)
    if a.set and not SEMVER.match(new):
        raise SystemExit(f"--set wants MAJOR.MINOR.PATCH, got {new!r}")
    sync(new)
    print(f"{current} -> {new}")
    print(summary())
    try:                                   # what the commit will carry, for the eye
        out = subprocess.run(["git", "status", "--short", "--", "VERSION", "app/package.json", "app/package-lock.json",
                              "leafmachine3/_env_contract.json", "app/desktop-contract.json"], cwd=ROOT,
                             capture_output=True, text=True, check=False).stdout
        if out.strip():
            print(out.rstrip())
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
