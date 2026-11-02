#!/usr/bin/env python3
"""Upload a staged model-repo folder as ONE commit (files not in the folder are removed), then verify it.

The folder must hold a manifest.json with per-file sha256 + bytes (the staging step writes it). After the
commit, every remote file is checked against it (size, and LFS sha256 for large files; the Hub's own
.gitattributes is ignored) and the card's `lm3:` front-matter block must carry the manifest's model_key.

A repo's `coreml_fp32/` packages are made from its ONNX model, so before the commit this refuses to:

* delete them while the ONNX model stays the same (a card edit staged from a folder that predates them;
  --drop-coreml-fp32 overrides this, and publish_coreml_fp32.py adds them back);
* upload packages made from a different ONNX model than the one staged.

New weights do remove the old packages: tools/macos/README.md explains how to make new ones.

    python tools/release/hf_upload_verify.py <staged dir> <namespace/repo> "<commit message>" [--private]
Needs huggingface_hub and a logged-in token with write access to the namespace.
"""
import argparse
import json
from pathlib import Path

COREML_FP32 = "coreml_fp32/"


def onnx_sha256(shas: dict) -> "str | None":
    """The sha256 of the repo's ONNX model, from a {repo path: sha256} mapping."""
    return next((s for p, s in sorted(shas.items()) if p.startswith("onnx/") and p.endswith(".onnx")), None)


def check_coreml_fp32(repo: str, staged: dict, remote: dict, contract: "dict | None", drop: bool) -> "str | None":
    """Stop an upload that would leave ``coreml_fp32/`` wrong; return a note to print after the commit.

    ``staged`` and ``remote`` map repo paths to sha256 (None when the Hub reports no LFS hash), and
    ``contract`` is the staged package's model.json, if there is one.
    """
    staged_onnx = onnx_sha256(staged)
    if contract is not None:
        made_from = (contract.get("onnx") or {}).get("sha256")
        if made_from != staged_onnx:
            raise SystemExit(f"{repo}: the staged coreml_fp32/ was made from ONNX {str(made_from)[:12]}, but the "
                             f"staged ONNX is {str(staged_onnx)[:12]}; export the package again")
        return None
    if not any(p.startswith(COREML_FP32) for p in remote):
        return None
    remote_onnx = onnx_sha256(remote)
    if not drop and (remote_onnx is None or remote_onnx == staged_onnx):
        raise SystemExit(f"{repo}: the Hub has coreml_fp32/ and the staged folder does not, but the ONNX model is "
                         f"not changing, so this upload would delete valid packages. Stage from the Hub's current "
                         f"files, or pass --drop-coreml-fp32 and add them back with publish_coreml_fp32.py")
    return (f"{repo}: this commit removes coreml_fp32/. A Mac runs this model on the CPU until new packages are "
            f"published (tools/macos/README.md)")


def main() -> None:
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("staged")
    ap.add_argument("repo")
    ap.add_argument("message")
    ap.add_argument("--private", action="store_true", help="create the repo private if it does not exist yet")
    ap.add_argument("--drop-coreml-fp32", action="store_true",
                    help="delete the repo's coreml_fp32/ packages even though the ONNX model is not changing")
    a = ap.parse_args()

    api = HfApi()
    api.create_repo(a.repo, private=a.private, exist_ok=True)
    man = json.load(open(f"{a.staged}/manifest.json"))

    def tree(**kw) -> dict:
        return {f.path: f for f in api.list_repo_tree(a.repo, recursive=True, expand=True, **kw)
                if f.__class__.__name__ == "RepoFile" and f.path != ".gitattributes"}

    contract_path = Path(a.staged) / COREML_FP32 / "model.json"
    note = check_coreml_fp32(a.repo, {p: m["sha256"] for p, m in man["files"].items()},
                             {p: (f.lfs.sha256 if f.lfs else None) for p, f in tree().items()},
                             json.loads(contract_path.read_text()) if contract_path.is_file() else None,
                             a.drop_coreml_fp32)

    sha = api.upload_folder(folder_path=a.staged, repo_id=a.repo, commit_message=a.message, delete_patterns=["*"]).oid
    remote = tree(revision=sha)
    want = set(man["files"]) | {"manifest.json"}
    assert set(remote) == want, sorted(set(remote) ^ want)[:10]
    bad = [p for p, m in man["files"].items()
           if remote[p].size != m["bytes"] or (remote[p].lfs and remote[p].lfs.sha256 != m["sha256"])]
    assert not bad, bad[:5]
    card = api.model_info(a.repo, revision=sha).card_data
    assert card and card.get("lm3", {}).get("model_key") == man["lm3"]["model_key"], "card lm3 block"
    print(f"{a.repo}: commit {sha[:7]}, {len(remote)} files verified (sizes + LFS sha256), card lm3 block ok")
    if note:
        print(f"NOTE: {note}")


if __name__ == "__main__":
    main()
