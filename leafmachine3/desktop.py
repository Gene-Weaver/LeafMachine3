"""Run the Electron development tools using the project's uv-locked Python, Node and npm.

Select a hardware extra and the desktop group with uv before using this command. npm has its own
lock format: installation always uses `npm ci`, and subsequent commands require its lock receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from leafmachine3.doctor import Env, OK, check_interpreter, diagnose


class DesktopEnvironmentError(RuntimeError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_project(root: Path, env: Env) -> dict:
    """Reject an unrelated interpreter, stale JavaScript lock, or missing desktop tool pins."""
    if Path(env.prefix).resolve() != (root / ".venv").resolve():
        raise DesktopEnvironmentError("Use this checkout's .venv through uv; other Python environments are unsupported.")
    interpreter = check_interpreter(env)
    if interpreter.status != OK:
        raise DesktopEnvironmentError(interpreter.detail)
    report = diagnose(env, quick=True, production=True)
    if not report.ready:
        raise DesktopEnvironmentError(report.first_failure.detail)
    contract = env.contract
    for name, want in (("nodejs-wheel", contract["desktop"]["node"]),
                       ("nodejs-wheel-binaries", contract["desktop"]["node"]),
                       ("uv", contract["uv"])):
        if env.installed.get(name) != want:
            raise DesktopEnvironmentError(f"{name}=={want} is required; run uv sync --frozen --extra <gpu|cpu|macos> --group desktop.")
        if env.missing_files.get(name):
            raise DesktopEnvironmentError(f"{name} has missing files; rebuild the desktop group with uv sync --frozen --extra <gpu|cpu|macos> --group desktop --reinstall.")
    for filename, key in (("package-lock.json", "package_lock_sha256"), ("package.json", "package_json_sha256")):
        if digest(root / "app" / filename) != contract["desktop"][key]:
            raise DesktopEnvironmentError(f"app/{filename} differs from the release contract; regenerate tools/release/write_env_contract.py.")
    if digest(root / "leafmachine3" / "_env_contract.json") != json.loads(
            (root / "app" / "desktop-contract.json").read_text())["backend_contract_sha256"]:
        raise DesktopEnvironmentError("The Python and Electron release contracts disagree; regenerate them.")
    return contract


def receipt_for(contract: dict) -> dict:
    return {"node": contract["desktop"]["node"],
            "package_lock_sha256": contract["desktop"]["package_lock_sha256"],
            "package_json_sha256": contract["desktop"]["package_json_sha256"],
            "platform": sys.platform}


def verify_install(app: Path, contract: dict) -> None:
    receipt = app / "node_modules" / ".lm3-lock.json"
    if not receipt.is_file() or json.loads(receipt.read_text()) != receipt_for(contract):
        raise DesktopEnvironmentError("Desktop dependencies need installation: run the same uv command with `lm3-desktop install`.")
    for name, key in (("electron", "electron"), ("electron-builder", "electron_builder")):
        package = app / "node_modules" / name / "package.json"
        if not package.is_file() or json.loads(package.read_text())["version"] != contract["desktop"][key]:
            raise DesktopEnvironmentError(f"{name} differs from the lock; run lm3-desktop install through uv.")
    runtime = app / "node_modules" / "electron" / "dist" / "version"
    if not runtime.is_file() or runtime.read_text().strip() != contract["desktop"]["electron"]:
        raise DesktopEnvironmentError("The pinned Electron runtime is missing; run lm3-desktop install through uv.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("install", "test", "start", "pack", "dist:linux", "dist:win", "dist:mac", "dist:mac:unsigned"))
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to the npm script after --")
    args = parser.parse_args(argv)
    root = Path(os.environ.get("LM3_ROOT") or Path(__file__).resolve().parents[1]).resolve()
    try:
        target = {"dist:linux": "linux", "dist:win": "win32", "dist:mac": "darwin", "dist:mac:unsigned": "darwin"}.get(args.action)
        if target and target != sys.platform:
            raise DesktopEnvironmentError("Build installers on their target OS using its uv-locked native runtime.")
        env = Env.current()
        contract = verify_project(root, env)
        app = root / "app"
        if args.action != "install":
            verify_install(app, contract)
        # Use the wheel's real Node binary both for npm and for its scripts/install hooks.
        from nodejs_wheel import npm  # noqa: PLC0415
        from nodejs_wheel.executable import ROOT_DIR  # noqa: PLC0415

        child_env = dict(os.environ)
        node_bin = ROOT_DIR if os.name == "nt" else str(Path(ROOT_DIR) / "bin")
        scripts = str(Path(env.prefix) / ("Scripts" if os.name == "nt" else "bin"))
        # The wheel's Python console wrappers also pin npm/npx for tools that resolve them on
        # PATH (electron-builder does). Its raw bin/npm shim has an invalid relative require.
        child_env["PATH"] = os.pathsep.join((scripts, node_bin, child_env.get("PATH", "")))
        for key in ("ELECTRON_RUN_AS_NODE", "ELECTRON_NO_ATTACH_CONSOLE", "LM3_PYTHON", "PYTHONPATH", "PYTHONHOME",
                    "NODE_PATH", "NODE_OPTIONS", "ELECTRON_OVERRIDE_DIST_PATH", "ELECTRON_INSTALL_PLATFORM",
                    "ELECTRON_INSTALL_ARCH", "electron_use_remote_checksums", "npm_config_electron_use_remote_checksums",
                    "npm_config_arch", "npm_config_platform", "NPM_CONFIG_ARCH", "NPM_CONFIG_PLATFORM"):
            child_env.pop(key, None)
        child_env.update(LM3_ROOT=str(root), LM3_EXTRA=diagnose(env, quick=True).variant,
                         LM3_STARTUP_GATE="1", LM3_DESKTOP="1")
        command = ["ci"] if args.action == "install" else ["run", args.action]
        forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
        if forwarded:
            command += ["--", *forwarded]
        result = npm(command, cwd=app, env=child_env)
        if result == 0 and args.action == "install":
            # Electron 43 downloads its runtime explicitly/on first launch, rather than in a
            # postinstall hook. Do it now using the locked package's pinned checksum manifest.
            result = npm(["run", "install:electron"], cwd=app, env=child_env)
            if result == 0:
                (app / "node_modules" / ".lm3-lock.json").write_text(json.dumps(receipt_for(contract)) + "\n")
                verify_install(app, contract)
        return result
    except (DesktopEnvironmentError, OSError, ValueError) as exc:
        print(f"lm3-desktop: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
