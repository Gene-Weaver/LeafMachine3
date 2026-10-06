#!/usr/bin/env python3
"""Write ``leafmachine3/_env_contract.json``: what a correct production install contains.

``lm3 doctor`` compares the running environment against this file, so it can say "your environment
is not the one this release was tested with" even when a user bypassed uv. It is generated, never
hand-edited, from three sources that are themselves the release's single sources of truth:

* ``uv.lock`` -- through ``uv export`` once per hardware extra, so the contract inherits uv's own
  resolution and marker evaluation instead of re-deriving the dependency graph here;
* ``tools/release/versions.env`` -- the Python patch, the uv version, the driver minimums;
* ``pyproject.toml`` -- the LM3 version.

The development group (``full``: torch, ultralytics, ...) is excluded on purpose. The contract
describes what ships; the doctor separately reports a development environment when it sees those.

Usage::

    python tools/release/write_env_contract.py           # (re)write the contract
    python tools/release/write_env_contract.py --check   # exit 1 if the committed file is stale
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "leafmachine3" / "_env_contract.json"
VERSIONS_ENV = ROOT / "tools" / "release" / "versions.env"
HARDWARE_EXTRAS = ("gpu", "cpu", "macos")
#: Packages whose presence marks a DEVELOPMENT environment (the `full` group). Never in production.
DEV_ONLY = ("torch", "torchvision", "ultralytics", "triton")


def read_versions_env(path: Path = VERSIONS_ENV) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def export_extra(extra: str) -> list[dict[str, str]]:
    """Pinned production requirements for one hardware extra, with their environment markers."""
    cmd = ["uv", "export", "--frozen", "--extra", extra, "--no-default-groups", "--no-emit-project",
           "--no-hashes", "--no-header", "--no-annotate", "--format", "requirements-txt"]
    text = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        req, _, marker = line.partition(";")
        name, _, version = req.strip().partition("==")
        if not version:
            raise SystemExit(f"uv export produced an unpinned requirement: {line!r}")
        out.append({"name": _normalize(name), "version": version.strip(), "marker": marker.strip()})
    return sorted(out, key=lambda r: r["name"])


def build() -> dict:
    env = read_versions_env()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return {
        "_generated_by": "tools/release/write_env_contract.py -- do not edit by hand",
        "lm3_version": pyproject["project"]["version"],
        "python": env["PYTHON_VERSION"],
        "uv": env["UV_VERSION"],
        "cuda_min_driver": {"linux": env["CUDA_MIN_DRIVER_LINUX"], "win32": env["CUDA_MIN_DRIVER_WINDOWS"]},
        "dev_only": list(DEV_ONLY),
        "extras": {extra: export_extra(extra) for extra in HARDWARE_EXTRAS},
    }


def render(contract: dict) -> str:
    return json.dumps(contract, indent=1, sort_keys=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if the committed contract is stale")
    args = ap.parse_args(argv)
    text = render(build())
    if args.check:
        current = CONTRACT.read_text() if CONTRACT.exists() else ""
        if current != text:
            print(f"{CONTRACT.relative_to(ROOT)} is stale; run tools/release/write_env_contract.py",
                  file=sys.stderr)
            return 1
        print("env contract is current")
        return 0
    CONTRACT.write_text(text)
    c = json.loads(text)
    print(f"wrote {CONTRACT.relative_to(ROOT)}: "
          + ", ".join(f"{k}={len(v)} pkgs" for k, v in c["extras"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
