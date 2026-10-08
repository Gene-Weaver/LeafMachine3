"""``lm3 version``: the one LM3 version and what it pins.

Prints the version, the dependency lock and model lock this build was stamped with (from
``leafmachine3/_env_contract.json``, written by tools/release/bump_version.py), and the Hub
revision of every model the models lock pins. ``--check`` compares the stamp against the files on
disk in a checkout, so a drift (a re-locked ``uv.lock`` or a regenerated models lock without a
version bump) is reported here as well as by the test suite.

Usage::

    lm3 version            # one line: LM3 3.0.1 · uv.lock 3f1c… · models.lock 9ab0… (13 Hub revisions)
    lm3 version --json
    lm3 version --check    # exit 1 when the locks on disk are not the ones the version pins
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from leafmachine3 import __version__

PKG = Path(__file__).resolve().parent
CONTRACT = PKG / "_env_contract.json"
MODELS_LOCK = PKG / "modelhub" / "models.lock.yaml"
UV_LOCK = PKG.parent / "uv.lock"          # present in a checkout, absent in an installed wheel


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def identity() -> dict:
    """The version stamp and, where the files exist, the on-disk hashes to compare it with."""
    try:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        contract = {}
    locks = contract.get("locks") or {}
    return {
        "version": __version__,
        "contract_version": contract.get("lm3_version"),
        "pinned": {"uv_lock_sha256": locks.get("uv_lock_sha256"), "models_lock_sha256": locks.get("models_lock_sha256"),
                   "models_lock_version": locks.get("models_lock_version"), "hub_revisions": locks.get("hub_revisions") or {}},
        "on_disk": {"uv_lock_sha256": _sha256(UV_LOCK), "models_lock_sha256": _sha256(MODELS_LOCK)},
    }


def problems(info: dict) -> list[str]:
    out = []
    if info["contract_version"] and info["contract_version"] != info["version"]:
        out.append(f"running {info['version']} but the contract was written for {info['contract_version']}")
    for key, label in (("models_lock_sha256", "models.lock.yaml"), ("uv_lock_sha256", "uv.lock")):
        pinned, disk = info["pinned"].get(key), info["on_disk"].get(key)
        if pinned and disk and pinned != disk:
            out.append(f"{label} on disk ({disk[:12]}) is not the one LM3 {info['version']} pins ({pinned[:12]}); "
                       "bump the version (tools/release/bump_version.py)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lm3 version", description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--check", action="store_true", help="exit 1 when the locks on disk are not the pinned ones")
    a = ap.parse_args(argv)
    info = identity()
    bad = problems(info)
    if a.json:
        print(json.dumps({**info, "problems": bad}, indent=2))
    else:
        p = info["pinned"]
        print(f"LM3 {info['version']} · uv.lock {str(p['uv_lock_sha256'])[:12]} · models.lock {str(p['models_lock_sha256'])[:12]}"
              f" ({len(p['hub_revisions'])} Hub revisions, lock {p['models_lock_version'] or '?'})")
        for repo, rev in sorted(p["hub_revisions"].items()):
            print(f"  {repo:60s} {str(rev)[:12]}")
        for line in bad:
            print(f"! {line}", file=sys.stderr)
    return 1 if (a.check and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
