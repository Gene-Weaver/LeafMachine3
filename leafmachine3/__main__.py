"""``python -m leafmachine3`` -- run the pipeline CLI (delegates to ``machine3.main``)."""
from __future__ import annotations

from leafmachine3.machine3 import main

if __name__ == "__main__":
    raise SystemExit(main())
