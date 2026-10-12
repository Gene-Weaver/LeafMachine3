#!/usr/bin/env python3
"""Upload a staged model-repo folder as ONE commit (files not in the folder are removed), then verify it.

The folder must hold a manifest.json with per-file sha256 + bytes (the staging step writes it). After the
commit, every remote file is checked against it (size, and LFS sha256 for large files; the Hub's own
.gitattributes is ignored) and the card's `lm3:` front-matter block must carry the manifest's model_key.

    python tools/release/hf_upload_verify.py <staged dir> <namespace/repo> "<commit message>" [--private]
Needs huggingface_hub and a logged-in token with write access to the namespace.
"""
import argparse
import json

from huggingface_hub import HfApi

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("staged")
ap.add_argument("repo")
ap.add_argument("message")
ap.add_argument("--private", action="store_true", help="create the repo private if it does not exist yet")
a = ap.parse_args()

api = HfApi()
api.create_repo(a.repo, private=a.private, exist_ok=True)
sha = api.upload_folder(folder_path=a.staged, repo_id=a.repo, commit_message=a.message, delete_patterns=["*"]).oid
man = json.load(open(f"{a.staged}/manifest.json"))
remote = {f.path: f for f in api.list_repo_tree(a.repo, revision=sha, recursive=True, expand=True)
          if f.__class__.__name__ == "RepoFile" and f.path != ".gitattributes"}
want = set(man["files"]) | {"manifest.json"}
assert set(remote) == want, sorted(set(remote) ^ want)[:10]
bad = [p for p, m in man["files"].items()
       if remote[p].size != m["bytes"] or (remote[p].lfs and remote[p].lfs.sha256 != m["sha256"])]
assert not bad, bad[:5]
card = api.model_info(a.repo, revision=sha).card_data
assert card and card.get("lm3", {}).get("model_key") == man["lm3"]["model_key"], "card lm3 block"
print(f"{a.repo}: commit {sha[:7]}, {len(remote)} files verified (sizes + LFS sha256), card lm3 block ok")
