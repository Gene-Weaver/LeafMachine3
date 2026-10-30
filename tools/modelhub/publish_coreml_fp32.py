"""Add ``coreml_fp32/`` packages to the default models' Hugging Face repos.

The packages come from ``tools/modelhub/export_coreml_fp32.py``, run on a Mac so each one carries a
passing load test. This is the publishing half of that exporter, for every future export:

    python tools/modelhub/publish_coreml_fp32.py stage --packages DIR --out STAGE
    python tools/modelhub/publish_coreml_fp32.py upload --stage STAGE

``DIR`` mirrors the models folder (``<unit dir>/coreml_fp32/model.mlpackage`` + ``model.json``), as
the exporter writes it. ``stage`` checks every package before anything is written:

* its ``model.mlmodel`` matches the sha256 in its ``model.json``;
* it was made from the ONNX file the lock pins for that unit (same sha256);
* its Mac load test passed (``--allow-untested`` skips this, for a dry run only).

Then, per repo, it writes ``coreml_fp32/`` and an updated ``manifest.json`` (the package files' sha256
and bytes, ``coreml_fp32`` in ``lm3.formats``, one ``patches`` entry) and ``README.md`` (the format in
the front matter, a Files row, one sentence under "Use in LeafMachine3"), and prints the diffs to review.

``upload`` pushes each staged folder as one commit (nothing is deleted) and checks the Hub's sha256 of
every new file against the staged manifest. Regenerate the lock afterwards (``build_lock.py``): it
discovers ``coreml_fp32/`` from the manifests.
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
LOCK = HERE.parents[1] / "leafmachine3" / "modelhub" / "models.lock.yaml"
FORMAT = "coreml_fp32"
FILES_ROW = f"| `{FORMAT}/` | `model.mlpackage/`, `model.json` | LM3 on macOS: Core ML on the Apple GPU, float32 |"
USE_LINE = (f"On macOS, LM3 runs `{FORMAT}/` with Core ML on the Apple GPU instead; its output matches this "
            f"ONNX model to float32 rounding.")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def units() -> list[dict]:
    """The lock's default units that run an ONNX model (the ones a package can stand in for)."""
    lock = yaml.safe_load(LOCK.read_text())
    return [u | {"action": a} for a, spec in lock["actions"].items() for u in spec["units"]
            if any(f["format"] == "onnx" for f in u["files"])]


def package_dir(packages: Path, u: dict) -> Path | None:
    """The unit's package under ``packages``: beside its ONNX file or one folder up, where the runtime
    looks (``coreml_native.package_for``; the ruler members keep their ONNX in ``exported/``)."""
    onnx = Path(next(f["dest"] for f in u["files"] if f["format"] == "onnx"))
    for d in (onnx.parent, onnx.parent.parent):
        if (packages / d / FORMAT / "model.json").is_file():
            return packages / d / FORMAT
    return None


def check_package(pkg_dir: Path, u: dict, allow_untested: bool) -> dict:
    meta = json.loads((pkg_dir / "model.json").read_text())
    if meta.get("format") != FORMAT:
        raise SystemExit(f"{pkg_dir}: format {meta.get('format')!r}, expected {FORMAT!r}")
    mlmodel = pkg_dir / meta["package"] / "Data" / "com.apple.CoreML" / "model.mlmodel"
    if sha256(mlmodel) != meta["mlmodel_sha256"]:
        raise SystemExit(f"{pkg_dir}: model.mlmodel does not match model.json")
    onnx = next(f for f in u["files"] if f["format"] == "onnx")
    if meta["onnx"]["sha256"] != onnx["sha256"]:
        raise SystemExit(f"{pkg_dir}: made from ONNX {meta['onnx']['sha256'][:12]}, the lock pins "
                         f"{onnx['sha256'][:12]} for {u['repo_id']}")
    lt = meta.get("load_test") or {}
    if not lt.get("ok") and not allow_untested:
        raise SystemExit(f"{pkg_dir}: no passing Mac load test in model.json (export it on a Mac)")
    return meta


def package_files(pkg_dir: Path) -> dict[str, dict]:
    """Repo path -> {sha256, bytes} for every file of the package folder."""
    out = {}
    for p in sorted(pkg_dir.rglob("*")):
        if p.is_file() and p.name != ".DS_Store":
            out[f"{FORMAT}/{p.relative_to(pkg_dir).as_posix()}"] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    return out


def add_format(formats: list[str]) -> list[str]:
    if FORMAT in formats:
        return formats
    i = formats.index("coreml") + 1 if "coreml" in formats else len(formats)
    return formats[:i] + [FORMAT] + formats[i:]


def edit_manifest(manifest: dict, files: dict, meta: dict, verified_on: str, commit: str) -> dict:
    m = json.loads(json.dumps(manifest))
    m["files"] = {k: v for k, v in m["files"].items() if not k.startswith(f"{FORMAT}/")} | files
    m["lm3"]["formats"] = add_format(m["lm3"].get("formats", []))
    lt = meta.get("load_test") or {}
    check = max((max(c["max_rel_diff"]) for c in meta.get("onnx_check") or []), default=None)
    m.setdefault("patches", []).append({
        "date": dt.date.today().isoformat(),
        "file": f"{FORMAT}/",
        "change": f"added {FORMAT}: float32 Core ML package with every input size LM3 uses "
                  f"({len(meta['input']['shapes'])}), made from onnx/model.onnx's checkpoint",
        "reason": "LeafMachine3's macOS runtime format (Core ML on the Apple GPU)",
        "tool": f"LeafMachine3 tools/modelhub/export_coreml_fp32.py ({commit})",
        "verification": {
            "onnx_sha256": meta["onnx"]["sha256"],
            "torch_vs_onnx_max_rel_diff": check,
            "coreml_vs_onnx_max_rel_diff": lt.get("max_rel_diff"),
            "load_test_on": verified_on,
            "details": f"{FORMAT}/model.json (onnx_check, load_test)",
        },
    })
    return m


def edit_card(card: str) -> str:
    lines = card.split("\n")
    end = lines.index("---", 1) if lines and lines[0] == "---" else 0     # the front matter's closing line
    for i in range(1, end):
        m = re.match(r"^(\s*formats:\s*\[)(.*)(\]\s*)$", lines[i])
        if m:
            fmts = add_format([f.strip() for f in m.group(2).split(",") if f.strip()])
            lines[i] = f"{m.group(1)}{', '.join(fmts)}{m.group(3)}"
    text = "\n".join(lines)
    if f"`{FORMAT}/`" not in text:
        rows = [ln for ln in text.split("\n") if ln.startswith("| `coreml/`")] or \
               [ln for ln in text.split("\n") if ln.startswith("| `onnx/`")]
        if not rows:
            raise SystemExit("card has no Files row to put coreml_fp32 after")
        text = text.replace(rows[0] + "\n", rows[0] + "\n" + FILES_ROW + "\n", 1)
    if USE_LINE not in text:
        m = re.search(r"## Use in LeafMachine3\n\n(.+?)\n\n", text, flags=re.S)
        if not m:
            raise SystemExit("card has no 'Use in LeafMachine3' paragraph")
        text = text[:m.end(1)] + "\n\n" + USE_LINE + text[m.end(1):]
    return text


def stage(args) -> int:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    packages, out = Path(args.packages), Path(args.out)
    planned = []
    for u in units():
        pkg = package_dir(packages, u)
        if pkg is None:
            continue
        meta = check_package(pkg, u, args.allow_untested)
        head = api.model_info(u["repo_id"]).sha
        if head != u["revision"]:
            raise SystemExit(f"{u['repo_id']}: the Hub is at {head[:7]}, the lock pins {u['revision'][:7]}; "
                             f"regenerate the lock first")
        planned.append((u, pkg, meta, head))
    if not planned:
        raise SystemExit(f"no {FORMAT} packages under {packages}")
    for u, pkg, meta, head in planned:
        repo = u["repo_id"]
        d = out / repo.split("/")[1]
        if d.exists():
            shutil.rmtree(d)
        (d / FORMAT).parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(pkg, d / FORMAT, copy_function=_link_or_copy,
                        ignore=shutil.ignore_patterns(".DS_Store"))
        manifest = json.loads(Path(hf_hub_download(repo, "manifest.json", revision=head)).read_text())
        card = Path(hf_hub_download(repo, "README.md", revision=head)).read_text(encoding="utf-8")
        new_manifest = edit_manifest(manifest, package_files(d / FORMAT), meta, args.verified_on, args.commit)
        new_card = edit_card(card)
        (d / "manifest.json").write_text(json.dumps(new_manifest, indent=1) + "\n")
        (d / "README.md").write_text(new_card, encoding="utf-8")
        (d / ".base_revision").write_text(head + "\n")
        print(f"== {repo} (from {head[:7]}): {len(meta['input']['shapes'])} shape(s), "
              f"{sum(f['bytes'] for k, f in new_manifest['files'].items() if k.startswith(FORMAT)) / 1e6:.0f} MB")
        for name, a, b in (("README.md", card, new_card),
                           ("manifest.json", json.dumps(manifest, indent=1), json.dumps(new_manifest, indent=1))):
            sys.stdout.writelines(difflib.unified_diff(a.splitlines(True), b.splitlines(True), f"hub/{name}",
                                                       f"staged/{name}", n=1))
            print()
    print(f"staged {len(planned)} repo(s) in {out}")
    return 0


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def upload(args) -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    staged = sorted(p for p in Path(args.stage).iterdir() if (p / "manifest.json").is_file())
    for d in staged:
        manifest = json.loads((d / "manifest.json").read_text())
        repo, base = manifest["repo_id"], (d / ".base_revision").read_text().strip()
        head = api.model_info(repo).sha
        if head != base:
            raise SystemExit(f"{repo}: the Hub moved ({base[:7]} -> {head[:7]}) since staging; stage again")
        info = api.upload_folder(repo_id=repo, folder_path=str(d), commit_message=f"Add {FORMAT}/ (Core ML, float32)",
                                 ignore_patterns=[".base_revision", "**/.DS_Store"])
        sib = {s.rfilename: s for s in api.model_info(repo, files_metadata=True).siblings}
        bad = [k for k, v in manifest["files"].items() if k.startswith(FORMAT)
               and (k not in sib or (sib[k].lfs.sha256 if sib[k].lfs else None) not in (v["sha256"], None))]
        if bad:
            raise SystemExit(f"{repo}: uploaded but these files do not match: {bad}")
        print(f"{repo}: {info.oid[:7]} ({sum(k.startswith(FORMAT) for k in manifest['files'])} {FORMAT} files verified)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="check the packages and stage each repo's changes")
    s.add_argument("--packages", required=True, help="folder in the models layout with <unit>/coreml_fp32/")
    s.add_argument("--out", required=True, help="staging folder (one subfolder per repo)")
    s.add_argument("--verified-on", required=True, help='the Mac that ran the load tests, e.g. "Apple M2 Pro, macOS 26.7.1"')
    s.add_argument("--commit", required=True, help="the LM3 commit whose exporter made the packages")
    s.add_argument("--allow-untested", action="store_true", help="accept packages without a Mac load test (dry run)")
    u = sub.add_parser("upload", help="upload staged repos and verify them")
    u.add_argument("--stage", required=True)
    args = ap.parse_args(argv)
    return stage(args) if args.cmd == "stage" else upload(args)


if __name__ == "__main__":
    sys.exit(main())
