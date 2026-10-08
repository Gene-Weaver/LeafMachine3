#!/usr/bin/env python3
"""Write ``leafmachine3/_env_contract.json``: what a correct production install contains.

``lm3 doctor`` compares the running environment against this file, so it can say "your environment
is not the one this release was tested with" even when a user bypassed uv. It is generated, never
hand-edited, from three sources that are themselves the release's single sources of truth:

* ``uv.lock`` -- through ``uv export`` once per hardware extra, so the contract inherits uv's own
  resolution and marker evaluation instead of re-deriving the dependency graph here;
* ``tools/release/versions.env`` -- the Python patch, the uv version, the driver minimums;
* ``VERSION`` -- the LM3 version (pyproject.toml reads the same file at build time);
* ``uv.lock`` and ``leafmachine3/modelhub/models.lock.yaml`` -- hashed, so a given LM3 version pins
  exactly one dependency lock and one set of Hub model revisions (listed in the contract too).

The development group (``full``: torch, ultralytics, ...) is excluded on purpose. The contract
describes what ships; the doctor separately reports a development environment when it sees those.

Usage::

    python tools/release/write_env_contract.py           # (re)write the contract
    python tools/release/write_env_contract.py --check   # exit 1 if the committed file is stale
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from leafmachine3.desktop import dependency_digest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "leafmachine3" / "_env_contract.json"
DESKTOP_CONTRACT = ROOT / "app" / "desktop-contract.json"
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


MODELS_LOCK = ROOT / "leafmachine3" / "modelhub" / "models.lock.yaml"
UV_LOCK = ROOT / "uv.lock"
VERSION_FILE = ROOT / "VERSION"


def read_version(path: Path = VERSION_FILE) -> str:
    """The LM3 version: the one line in VERSION."""
    return path.read_text(encoding="utf-8").strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_pins() -> dict:
    """What this LM3 version pins: the dependency lock and the model lock, by content hash, plus
    the Hub revision of every model unit so the contract reads as a release manifest."""
    import yaml  # noqa: PLC0415 - tooling only

    models = yaml.safe_load(MODELS_LOCK.read_text(encoding="utf-8")) or {}
    revisions = {}
    for key, action in (models.get("actions") or {}).items():
        for u in action.get("units") or []:
            revisions[u["repo_id"]] = u.get("revision")
    for stage, alts in (models.get("alternates") or {}).items():
        for key, action in (alts or {}).items():
            for u in action.get("units") or []:
                revisions[u["repo_id"]] = u.get("revision")
    return {
        "uv_lock_sha256": _sha256(UV_LOCK),
        "models_lock_sha256": _sha256(MODELS_LOCK),
        "models_lock_version": str(models.get("lm3_version", "")),
        "models_lock_generated_at": str(models.get("generated_at", "")),
        "hub_revisions": dict(sorted(revisions.items())),
    }


def build() -> dict:
    env = read_versions_env()
    package = json.loads((ROOT / "app" / "package.json").read_text())
    return {
        "_generated_by": "tools/release/write_env_contract.py -- do not edit by hand",
        "lm3_version": read_version(),
        "locks": lock_pins(),
        "python": env["PYTHON_VERSION"],
        "uv": env["UV_VERSION"],
        "desktop": {
            "node": env["NODE_VERSION"],
            "electron": package["devDependencies"]["electron"],
            "electron_builder": package["devDependencies"]["electron-builder"],
            # dependency content only: the version stamped into both files changes every commit
            "package_lock_sha256": dependency_digest(ROOT / "app" / "package-lock.json"),
            "package_json_sha256": dependency_digest(ROOT / "app" / "package.json"),
        },
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
    c = json.loads(text)
    desktop = render({
        "_generated_by": "tools/release/write_env_contract.py -- do not edit by hand",
        "lm3_version": c["lm3_version"], "python": c["python"], "uv": c["uv"],
        **c["desktop"],
        "backend_contract_sha256": hashlib.sha256(text.encode()).hexdigest(),
    })
    if args.check:
        for path, expected in ((CONTRACT, text), (DESKTOP_CONTRACT, desktop)):
            current = path.read_bytes() if path.exists() else b""
            if current != expected.encode("utf-8"):
                print(f"{path.relative_to(ROOT)} is stale; run tools/release/write_env_contract.py",
                      file=sys.stderr)
                return 1
        print("env contract is current")
        return 0
    CONTRACT.write_text(text, encoding="utf-8", newline="\n")
    DESKTOP_CONTRACT.write_text(desktop, encoding="utf-8", newline="\n")
    print(f"wrote {CONTRACT.relative_to(ROOT)}: "
          + ", ".join(f"{k}={len(v)} pkgs" for k, v in c["extras"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
