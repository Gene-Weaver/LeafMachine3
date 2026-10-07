"""``lm3 models`` -- status / install / verify the default models from the Hugging Face Hub."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from leafmachine3.modelhub import installer, registry

_STATE_MARK = {"current": "ok", "missing": "MISSING", "outdated": "OUTDATED", "pending": "pending (local copy)",
               "unavailable": "unavailable (not published yet)"}


def _root(args: argparse.Namespace) -> Path:
    return Path(args.dest).expanduser() if args.dest else installer.models_root()


def _print_status(st: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(st, indent=2))
        return
    print(f"models root : {st['root']}")
    print(f"lock        : {st['lock_path']} (LM3 {st['lm3_version']}, formats: {', '.join(st['formats'])})")
    for key, a in st["actions"].items():
        mark = _STATE_MARK.get(a["state"], a["state"])
        print(f"  {key:22s} {mark:32s} {a['detail']}")
    s = st["summary"]
    print(f"summary     : {s['button_label']}"
          + (f"  (missing: {', '.join(s['missing'])})" if s["missing"] else "")
          + (f"  (outdated: {', '.join(s['outdated'])})" if s["outdated"] else ""))
    for r in st.get("repaired") or []:
        print(f"  repaired  : {r['action']} {r['file']}")


def _progress_printer(ev: dict) -> None:
    t = ev.get("type")
    if t == "start":
        mb = (ev.get("total_bytes") or 0) / 1e6
        print(f"installing {len(ev['actions'])} action(s) into {ev['root']} ({mb:.0f} MB) ...")
    elif t == "skip":
        print(f"  skip {ev['action']}: {ev['reason']}")
    elif t == "file" and ev.get("phase") == "download":
        mb = (ev.get("bytes") or 0) / 1e6
        print(f"  {ev['action']}: downloading {ev['file']} ({mb:.1f} MB) from {ev['repo_id']}", flush=True)
    elif t == "action_done":
        print(f"  {ev['action']}: installed")
    elif t == "error":
        print(f"  {ev['action']}: ERROR {ev['message']}", file=sys.stderr)


def _parse_models(raw) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in raw or []:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"lm3 models: --model expects STAGE=KEY, got {part!r}")
            stage, key = (x.strip() for x in part.split("=", 1))
            out.append((stage, key))
    return out


#: stages whose settings name their model with ``model.key`` (the name picks the input workflow)
KEYED_STAGES = {"specimen_segmenter"}


def _print_settings_snippet(alt, root: Path, formats) -> None:
    """The settings lines that point ``alt.stage`` at this alternate, using the installed paths."""
    onnx = next((f.dest for _u, f in alt.files(formats) if f.format == "onnx"), None)
    if onnx is None:
        return
    # relative "models/..." only when the files went to the default folder beside the settings file;
    # anywhere else ($LM3_MODELS_DIR or --dest) the snippet names the absolute path it really used
    try:
        default_root = installer.models_root(env={}).resolve()
    except Exception:  # noqa: BLE001 - no settings file to anchor on
        default_root = None
    rel = f"models/{onnx}" if (default_root is not None and Path(root).resolve() == default_root
                               and not os.environ.get(installer.ENV_ROOT)) else str(Path(root).resolve() / onnx)
    model = {"path": rel, "format": "onnx"}
    if alt.stage in KEYED_STAGES:
        model = {"key": alt.model_key, **model}
    inner = ", ".join(f'{k}: "{v}"' for k, v in model.items())
    print(f"\nTo use {alt.model_key} for {alt.stage}, set in LM3_settings.yaml:\n")
    print("modules:")
    print(f"  {alt.stage}:")
    print(f"    model: {{ {inner} }}")
    for k, v in (alt.settings or {}).items():
        print(f"    {k}: {v}")


def build_parser() -> argparse.ArgumentParser:
    def common(parser: argparse.ArgumentParser, *, after: bool) -> None:
        # The same options are accepted before AND after the subcommand (`lm3 models install --dest X`
        # reads naturally); SUPPRESS keeps a subcommand's unset option from overriding the global one.
        d = {"default": argparse.SUPPRESS} if after else {}
        parser.add_argument("--dest", help=f"models folder (default: ${installer.ENV_ROOT} or <settings dir>/models)", **d)
        parser.add_argument("--lock", help="alternate models.lock.yaml (testing)", **d)
        parser.add_argument("--formats", help="comma list of formats to manage (default: onnx)", **d)
        parser.add_argument("--json", action="store_true", help="machine-readable output", **d)

    p = argparse.ArgumentParser(prog="lm3 models", description="Manage the default LM3 models (Hugging Face Hub).")
    common(p, after=False)
    sub = p.add_subparsers(dest="cmd", metavar="<status|install|verify>")
    common(sub.add_parser("status", help="what is installed vs. what the lock pins (no network)"), after=True)
    common(sub.add_parser("verify", help="status, re-hashing every present file against the lock"), after=True)
    ins = sub.add_parser("install", help="download missing/outdated models (backs up, verifies, rolls back on failure)")
    common(ins, after=True)
    ins.add_argument("--actions", default=None, help="comma list of actions (default: all)")
    ins.add_argument("--model", action="append", default=None, metavar="STAGE=KEY",
                     help="install an alternate model instead, e.g. specimen_segmenter=yolo26x_seg_1280 "
                          "(repeatable or comma-separated); prints the settings lines that select it"),
    ins.add_argument("--list-alternates", action="store_true", help="list the alternate models in the lock and exit")
    ins.add_argument("--force", action="store_true", help="re-download actions that are already current")
    ins.add_argument("--yes", "-y", action="store_true", help="do not ask before overwriting existing models")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    lock = registry.load_lock(args.lock)
    formats = [s.strip() for s in args.formats.split(",")] if args.formats else None
    root = _root(args)
    cmd = args.cmd or "status"
    try:
        if cmd == "status":
            _print_status(installer.status(root, lock=lock, formats=formats), args.json)
            return 0
        if cmd == "verify":
            st = installer.verify(root, lock=lock, formats=formats)
            _print_status(st, args.json)
            return 0 if not st["summary"]["needs_attention"] else 1
        if getattr(args, "list_alternates", False):
            for stage, ks in lock.alternates.items():
                for k, a in ks.items():
                    print(f"  {stage}={k:28s} {a.units[0].repo_id}")
            return 0
        actions = [s.strip() for s in args.actions.split(",")] if args.actions else None
        models = _parse_models(getattr(args, "model", None))
        if models:
            for stage, k in models:
                lock.alternate(stage, k)          # unknown -> KeyError with the known list
            st = installer.install(root, lock=lock, actions=actions, formats=formats, force=args.force, models=models,
                                   progress=None if args.json else _progress_printer)
            if args.json:
                print(json.dumps(st, indent=2))
            else:
                for stage, k in models:
                    _print_settings_snippet(lock.alternate(stage, k), root, formats or lock.default_formats)
            return 0
        before = installer.status(root, lock=lock, formats=formats)
        existing = [k for k, a in before["actions"].items() if any(f["present"] for f in a["files"])
                    and (actions is None or k in actions) and a["state"] in ("outdated", "current") and (args.force or a["state"] == "outdated")]
        if existing and not args.yes and not args.json:
            print(f"About to download and OVERWRITE existing models in {root}: {', '.join(existing)}")
            print("Each file is backed up first and restored if anything fails. Continue? [y/N] ", end="", flush=True)
            if sys.stdin.readline().strip().lower() not in ("y", "yes"):
                print("aborted")
                return 1
        st = installer.install(root, lock=lock, actions=actions, formats=formats, force=args.force,
                               progress=None if args.json else _progress_printer)
        _print_status(st, args.json)
        return 0 if not st["summary"]["missing"] else 1
    except installer.InstallError as exc:
        print(f"lm3 models: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
