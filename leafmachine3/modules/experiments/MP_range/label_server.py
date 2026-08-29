"""MP_range step 1 -- a local labeling tool for ground-truth conversion factors.

Serves a zoomable viewer for N sheets picked to span the corpus's megapixel range. You click two
points 1 cm apart on each sheet's ruler; the server records them in ORIGINAL pixel coordinates, so
the resulting ``px_per_cm`` is the sheet's true CF at its native resolution.

Only the ORIGINALS are ever labeled. Every other point in the experiment is derived geometrically
by :mod:`expand_and_plot`, because a uniform resize scales the CF by exactly the same factor it
scales the pixels -- there is nothing to re-measure, and re-measuring would only add click noise.

Run::

    python -m leafmachine3.modules.experiments.MP_range.label_server            # then open the URL
    python -m leafmachine3.modules.experiments.MP_range.label_server --n 20 --port 8765

Writes ``manifest.json`` (the chosen sheets, reproducible) and ``labels.json`` (your clicks,
autosaved after every change, so closing the tab loses nothing).

Precision note: a 1 cm span is ~130 px on a 20 MP sheet. Fit-to-window shrinks that to ~35 screen
px, where a single-pixel slip is a 3% CF error -- which is the size of the effect this experiment
is trying to resolve. Hence the wheel zoom and the 8x magnifier: place points at 100% or closer.
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import sys
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

Image.MAX_IMAGE_PIXELS = None                    # herbarium sheets trip the decompression-bomb guard

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGES = Path("/datac/Labelbox_Dump/data/plant-broadsample-randomizedspp-og__hr994cc9/images")
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ---------------------------------------------------------------- selection
def scan(images_dir: Path) -> list[dict]:
    """Every readable image with its dimensions, sorted by megapixels."""
    out = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:                        # a corrupt file must not stop the survey
            continue
        out.append({"name": p.name, "path": str(p), "width": w, "height": h, "mp": round(w * h / 1e6, 4)})
    out.sort(key=lambda r: r["mp"])
    return out


def pick_spanning(rows: list[dict], n: int) -> list[dict]:
    """``n`` sheets spread evenly across the MP RANGE (not across the rank order).

    Quantile sampling would cluster: this corpus is 14.7-38.6 MP but half of it sits in 18-21 MP,
    so evenly-spaced ranks would spend 10 of 20 picks inside one narrow band and leave the high end
    -- the part the shipped fit extrapolates through -- represented by one or two sheets. Binning on
    the VALUE and taking the nearest sheet to each bin center buys coverage where it matters.
    """
    if not rows:
        return []
    if len(rows) <= n:
        return list(rows)
    lo, hi = rows[0]["mp"], rows[-1]["mp"]
    picked: list[dict] = []
    used: set[str] = set()
    for i in range(n):
        target = lo + (hi - lo) * (i / (n - 1))
        best = min((r for r in rows if r["name"] not in used),
                   key=lambda r: abs(r["mp"] - target))
        used.add(best["name"])
        picked.append(best)
    picked.sort(key=lambda r: r["mp"])
    return picked


# ---------------------------------------------------------------- state
class Store:
    """manifest.json + labels.json, written through on every mutation."""

    def __init__(self, out_dir: Path) -> None:
        self.dir = out_dir
        self.manifest_path = out_dir / "manifest.json"
        self.labels_path = out_dir / "labels.json"
        self.lock = threading.Lock()
        self.manifest: list[dict] = []
        self.labels: dict = {}

    def load_or_build(self, images_dir: Path, n: int, reselect: bool) -> None:
        if self.manifest_path.exists() and not reselect:
            self.manifest = json.loads(self.manifest_path.read_text())
        else:
            rows = scan(images_dir)
            if not rows:
                raise SystemExit(f"no readable images under {images_dir}")
            self.manifest = pick_spanning(rows, n)
            self.manifest_path.write_text(json.dumps(self.manifest, indent=2))
        if self.labels_path.exists():
            self.labels = json.loads(self.labels_path.read_text())
        self.labels.setdefault("labels", {})
        self.labels.setdefault("finished", False)

    def put(self, name: str, entry: dict | None) -> None:
        with self.lock:
            if entry is None:
                self.labels["labels"].pop(name, None)
            else:
                self.labels["labels"][name] = entry
            self._flush()

    def finish(self, value: bool) -> None:
        with self.lock:
            self.labels["finished"] = bool(value)
            self._flush()

    def _flush(self) -> None:
        self.labels_path.write_text(json.dumps(self.labels, indent=2))


# ---------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    store: Store
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):           # keep the console readable
        pass

    # -- helpers
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _by_name(self, name: str) -> dict | None:
        return next((r for r in self.store.manifest if r["name"] == name), None)

    # -- routes
    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            self._send(200, (HERE / "app.html").read_bytes(), "text/html; charset=utf-8")
        elif route == "/api/state":
            self._json({"images": self.store.manifest, **self.store.labels})
        elif route.startswith("/img/"):
            name = route[len("/img/"):]
            row = self._by_name(name)            # serve ONLY manifest entries, never arbitrary paths
            if row is None:
                self._json({"error": "unknown image"}, 404)
                return
            data = Path(row["path"]).read_bytes()
            ctype = mimetypes.guess_type(row["path"])[0] or "application/octet-stream"
            self._send(200, data, ctype)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if route == "/api/label":
            name = str(payload.get("name", ""))
            row = self._by_name(name)
            if row is None:
                self._json({"error": "unknown image"}, 404)
                return
            pts = payload.get("points") or []
            if len(pts) != 2:
                self.store.put(name, None)       # fewer than two points = not labeled
                self._json({"ok": True, "cleared": True})
                return
            (x1, y1), (x2, y2) = pts
            px = math.dist((x1, y1), (x2, y2))
            self.store.put(name, {
                "name": name, "width": row["width"], "height": row["height"], "mp": row["mp"],
                "points": [[x1, y1], [x2, y2]],
                "px_per_cm": round(px, 4),
                # Sanity signal, shown live in the UI: a herbarium sheet is ~28-32 cm across, so an
                # implied width outside that is a mis-click (or a ruler in inches) and not a CF.
                "implied_sheet_w_cm": round(row["width"] / px, 2) if px else None,
                "implied_sheet_h_cm": round(row["height"] / px, 2) if px else None,
            })
            self._json({"ok": True})
        elif route == "/api/finish":
            self.store.finish(bool(payload.get("finished", True)))
            n = len(self.store.labels["labels"])
            print(f"\n  FINISHED -- {n} sheet(s) labeled -> {self.store.labels_path}")
            print("  next:  python -m leafmachine3.modules.experiments.MP_range.expand_and_plot\n")
            self._json({"ok": True, "n": n})
        else:
            self._json({"error": "not found"}, 404)


def _serve(first_port: int, store: Store, tries: int = 12) -> ThreadingHTTPServer:
    """Bind the first free port at or after ``first_port``.

    This box already runs other local services, and a hard failure on a busy port is a pointless
    way to lose a labeling session -- so walk forward and report where we actually landed.
    """
    handler = partial(type("H", (Handler,), {"store": store}))
    last: OSError | None = None
    for port in range(first_port, first_port + tries):
        try:
            return ThreadingHTTPServer(("127.0.0.1", port), handler)
        except OSError as exc:                   # EADDRINUSE -> try the next one
            last = exc
            print(f"  port {port} busy, trying {port + 1}", flush=True)
    raise SystemExit(f"no free port in {first_port}-{first_port + tries - 1}: {last}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Label 1 cm on sheets spanning the corpus MP range.")
    ap.add_argument("--images", type=Path, default=DEFAULT_IMAGES, help="source image directory")
    ap.add_argument("--out", type=Path, default=HERE, help="where manifest.json / labels.json go")
    ap.add_argument("--n", type=int, default=20, help="how many sheets to label")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--reselect", action="store_true", help="rebuild manifest.json from scratch")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    store = Store(args.out)
    store.load_or_build(args.images, args.n, args.reselect)

    mps = [r["mp"] for r in store.manifest]
    print(f"  {len(store.manifest)} sheets selected, {mps[0]:.1f} - {mps[-1]:.1f} MP", flush=True)
    print(f"  {len(store.labels['labels'])} already labeled")
    server = _serve(args.port, store)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"  serving {url}   (ctrl-c to stop)\n", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
