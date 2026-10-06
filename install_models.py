#!/usr/bin/env python
"""Install the default LeafMachine3 models from the Hugging Face Hub.

    uv run install_models.py            # or: .venv_LM3/bin/python install_models.py
    uv run install_models.py --yes      # non-interactive (Docker / HPC provisioning)
    uv run install_models.py --dest /shared/lm3/models   # a shared or mounted models folder

Thin wrapper over ``lm3 models install``; see ``lm3 models --help`` for status/verify.
"""
import sys

from leafmachine3.modelhub.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(args if args and args[0] in ("status", "install", "verify") else ["install", *args]))
