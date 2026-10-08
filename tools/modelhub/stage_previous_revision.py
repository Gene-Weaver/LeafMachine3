#!/usr/bin/env python3
"""Put a PREVIOUS revision of a default model back in place so the updater has something to update.

Development helper for exercising the Models tab's "update available" path on a machine that is
already current. It copies ``models/_previous_revisions/<stage>/<file>`` over the installed file and
rewrites that file's entry in ``models/installed.json`` to the old sha, so ``lm3 models status``
(and the GUI) report the stage as outdated. Clicking the orange ``↻ onnx`` chip, or
``lm3 models install``, then downloads the pinned revision and replaces it again.

Usage::

    python tools/modelhub/stage_previous_revision.py plant_detector            # stage the old file
    python tools/modelhub/stage_previous_revision.py plant_detector --status   # just report

The previous revision is fetched once from the Hub and kept under ``models/_previous_revisions/``
(gitignored with the rest of ``models/``); a ``REVISION.txt`` beside it says what it is.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from leafmachine3.modelhub import installer, registry  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", help="a default action in the lock, e.g. plant_detector")
    ap.add_argument("--file", default="model.onnx", help="file name under models/<stage>/ (default: model.onnx)")
    ap.add_argument("--status", action="store_true", help="report the stage's state and exit")
    a = ap.parse_args(argv)

    lock = registry.load_lock()
    root = installer.models_root()
    action = lock.action(a.stage)
    st = installer.status(root, lock=lock)["actions"][a.stage]
    print(f"{a.stage}: {st['state']} ({st['detail'] or 'matches the lock'})")
    if a.status:
        return 0

    old = root / "_previous_revisions" / a.stage / a.file
    if not old.is_file():
        print(f"no previous revision kept at {old}", file=sys.stderr)
        return 1
    lf = next((f for u in action.units for f in u.files if Path(f.dest).name == a.file), None)
    if lf is None:
        print(f"{a.stage} has no file named {a.file} in the lock", file=sys.stderr)
        return 1
    dest = root / lf.dest
    digest = sha256(old)
    if digest == lf.sha256:
        print(f"{old} IS the pinned revision; nothing to stage", file=sys.stderr)
        return 1
    shutil.copyfile(old, dest)
    record = installer.read_record(root)
    rec = record.setdefault("actions", {}).setdefault(a.stage, {})
    rec.setdefault("files", {})[lf.dest] = {"sha256": digest, "bytes": dest.stat().st_size, "src": lf.src,
                                            "repo_id": action.units[0].repo_id,
                                            "stat": installer._stat_sig(dest)}
    rev_file = old.with_name("REVISION.txt")
    old_rev = rev_file.read_text().split("@", 1)[1].split()[0] if rev_file.is_file() and "@" in rev_file.read_text() else "previous"
    rec.setdefault("revisions", {})[action.units[0].repo_id] = old_rev
    installer.write_record(root, record)
    st = installer.status(root, lock=lock)["actions"][a.stage]
    print(f"staged {old.name} ({digest[:12]}) -> {dest}; {a.stage} is now: {st['state']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
