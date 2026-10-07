"""generate_stl_from_mask -- turn a binary-mask PNG into a 3D-printable ``.stl``.

The 2D mask is extruded to a fixed ``thickness_mm`` z-height: a circle mask becomes a cylinder, a
leaf mask becomes a flat leaf slab. The model is scaled so the mask's LONGEST dimension equals
``length_mm``. Foreground is selected by color(s) (white by default); internal holes are optionally
filled. Watertight STL is written via trimesh (shapely polygons-with-holes + earcut triangulation).

Standalone postprocessing tool -- NOT part of the pipeline. Configure in ``postprocessing_settings.yaml``:

    .venv_LM3/bin/python -m leafmachine3.postprocessing.generate_stl_from_mask \
        --config postprocessing_settings.yaml --paths mask.png --length-mm 150 --thickness-mm 2

Heavy deps (trimesh, shapely, mapbox_earcut) are imported lazily so the module imports without them.

The CLI is subject to plan section 2.8 exactly as the Postprocess tab is -- it refuses a target
that belongs to the running pipeline and serializes two read/write tools on one completed run --
through the same guard the HTTP API calls (see :func:`guarded_targets`). A refusal exits
``EXIT_REFUSED``.
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("leafmachine3.postprocessing.generate_stl_from_mask")

# Defaults mirror postprocessing_settings.yaml (used when a key is absent).
_DEFAULTS: dict[str, Any] = {
    "paths": None, "output_dir": None, "fill_holes": True, "colors": ["white"],
    "color_tolerance": 0, "length_mm": 150.0, "thickness_mm": 2.0,
    "simplify_tolerance_px": 1.5, "min_area_px": 4.0,
}
_NAMED_COLORS = {"white": (255, 255, 255), "black": (0, 0, 0)}

#: Exit code for a plan section 2.8 refusal (the target is the live run, or another read/write
#: tool holds that run's lock). Deliberately DISTINCT from 1 (the tool failed) and 2 (argparse),
#: so a batch script can tell "refused, retry later" apart from "this input is broken".
EXIT_REFUSED = 3


# -- color selection ---------------------------------------------------------------
def _parse_colors(colors) -> list[tuple[int, int, int]]:
    """Normalize a color spec to a list of RGB triples. Accepts a single name/``[R,G,B(,A)]`` or a
    list of them; alpha (4th value) is ignored for matching."""
    if colors is None:
        colors = ["white"]
    if isinstance(colors, str):
        colors = [colors]
    elif isinstance(colors, (list, tuple)) and colors and all(isinstance(c, (int, float)) for c in colors):
        colors = [colors]                          # a single [R,G,B(,A)] passed directly
    out: list[tuple[int, int, int]] = []
    for c in (colors or ["white"]):
        if isinstance(c, str):
            key = c.strip().lower()
            if key not in _NAMED_COLORS:
                raise ValueError(f"unknown color name {c!r}; use 'white'/'black' or an [R,G,B] list")
            out.append(_NAMED_COLORS[key])
        else:
            seq = [int(round(float(v))) for v in c]
            if len(seq) < 3:
                raise ValueError(f"color {c!r} needs at least R, G, B")
            out.append((seq[0], seq[1], seq[2]))   # RGB; alpha ignored
    return out


def _select_mask(mask_path, colors, tol: int):
    """Read a PNG and return a bool foreground mask of pixels matching any of ``colors``."""
    import cv2
    import numpy as np

    im = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"cannot read mask image: {mask_path}")
    if im.ndim == 2:
        rgb = np.repeat(im[:, :, None], 3, axis=2)
    elif im.shape[2] == 4:
        rgb = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
    elif im.shape[2] == 3:
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"unsupported channel count {im.shape} for {mask_path}")
    rgb = rgb.astype(np.int16)
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    for tgt in _parse_colors(colors):
        diff = np.abs(rgb - np.array(tgt, np.int16)).max(axis=2)
        mask |= diff <= int(tol)
    return mask


def _fill_holes(mask):
    """Fill fully-enclosed background holes (flood-fill from a padded border)."""
    import cv2
    import numpy as np

    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    pad = np.zeros((H + 2, W + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    ff = pad.copy()
    ff_mask = np.zeros((H + 4, W + 4), np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 1)          # 1 = background reachable from the border
    holes = (ff[1:-1, 1:-1] == 0) & (m == 0)       # enclosed background -> fill
    return (m.astype(bool) | holes)


# -- mask -> polygons -> extruded mesh ---------------------------------------------
def _iter_polygons(geom):
    from shapely.geometry import MultiPolygon, Polygon

    if isinstance(geom, Polygon):
        if not geom.is_empty:
            yield geom
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            if not g.is_empty:
                yield g
    else:  # GeometryCollection etc. -- keep only polygon parts
        for g in getattr(geom, "geoms", []):
            if isinstance(g, Polygon) and not g.is_empty:
                yield g


def _mask_to_polygons(mask, simplify_px: float, min_area_px: float) -> list:
    """Contour the mask into shapely polygons (outer boundary + hole rings) in pixel coords."""
    import cv2
    import numpy as np
    from shapely.geometry import Polygon

    m = (mask > 0).astype(np.uint8)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or not len(cnts):
        return []
    hier = hier[0]                                  # [next, prev, first_child, parent] per contour
    polys: list = []
    for i, h in enumerate(hier):
        if h[3] != -1:                              # a hole (has a parent) -> handled by its parent
            continue
        outer = cnts[i].reshape(-1, 2)
        if len(outer) < 3:
            continue
        holes = []
        j = h[2]                                    # first child = first hole
        while j != -1:
            hp = cnts[j].reshape(-1, 2)
            if len(hp) >= 3:
                holes.append(hp)
            j = hier[j][0]                          # next sibling hole
        poly = Polygon(outer, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if simplify_px and float(simplify_px) > 0:
            poly = poly.simplify(float(simplify_px), preserve_topology=True)
        for pp in _iter_polygons(poly):
            if pp.area >= float(min_area_px):
                polys.append(pp)
    return polys


def _build_mesh(polys, length_mm: float, thickness_mm: float):
    """Scale (longest mask dim -> length_mm), flip Y, and extrude each polygon to thickness_mm."""
    import trimesh
    from shapely import affinity

    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    longest = max(maxx - minx, maxy - miny)
    if longest <= 0:
        raise ValueError("degenerate mask (zero extent)")
    scale = float(length_mm) / float(longest)
    meshes = []
    for p in polys:
        # x' = (x-minx)*scale ; y' = (maxy-y)*scale -- scale to mm, flip Y (image y is top-down),
        # and translate so the model's min corner sits at the origin.
        p2 = affinity.affine_transform(p, [scale, 0.0, 0.0, -scale, -minx * scale, maxy * scale])
        try:
            meshes.append(trimesh.creation.extrude_polygon(p2, height=float(thickness_mm)))
        except Exception as exc:  # noqa: BLE001 - skip an un-triangulable part, keep the rest
            log.warning("skipping a mask part that could not be extruded: %s", exc)
    if not meshes:
        raise ValueError("no polygon could be extruded into a solid")
    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    return mesh, scale


def generate_stl(
    mask_path,
    out_path,
    *,
    colors=("white",),
    fill_holes: bool = True,
    color_tolerance: int = 0,
    length_mm: float = 150.0,
    thickness_mm: float = 2.0,
    simplify_tolerance_px: float = 1.5,
    min_area_px: float = 4.0,
) -> dict:
    """Generate one ``.stl`` from a binary-mask PNG. Returns a summary dict."""
    mask = _select_mask(mask_path, colors, color_tolerance)
    if not mask.any():
        raise ValueError(f"no pixels matched colors={colors} (tolerance={color_tolerance}) in {mask_path}")
    if fill_holes:
        mask = _fill_holes(mask)
    polys = _mask_to_polygons(mask, simplify_tolerance_px, min_area_px)
    if not polys:
        raise ValueError(f"no usable foreground polygon in {mask_path} (min_area_px={min_area_px})")
    mesh, scale = _build_mesh(polys, length_mm, thickness_mm)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    ext = [round(float(v), 3) for v in mesh.extents]     # bounding-box size [x, y, z] in mm
    return {
        "mask": str(mask_path), "stl": str(out_path), "n_parts": len(polys),
        "size_mm": ext, "thickness_mm": float(thickness_mm), "length_mm": float(length_mm),
        "watertight": bool(mesh.is_watertight), "scale_mm_per_px": round(scale, 6),
        "fill_holes": bool(fill_holes),
    }


# -- driver + CLI ------------------------------------------------------------------
def _out_path(mask_path, output_dir) -> Path:
    mp = Path(mask_path)
    return (Path(output_dir) / f"{mp.stem}.stl") if output_dir else mp.with_suffix(".stl")


def run(settings: Optional[dict] = None, paths=None, output_dir=None) -> list[dict]:
    """Run generate_stl_from_mask over every input path. ``paths``/``output_dir`` override settings."""
    s = {**_DEFAULTS, **(settings or {})}
    paths = paths if paths is not None else s.get("paths")
    if not paths:
        raise ValueError("no input paths: set generate_stl_from_mask.paths in the yaml or pass --paths")
    if isinstance(paths, str):
        paths = [paths]
    outdir = output_dir if output_dir is not None else s.get("output_dir")
    seen: set[str] = set()
    results: list[dict] = []
    for p in paths:
        key = str(Path(p))
        if key in seen:                              # de-dupe repeated paths
            continue
        seen.add(key)
        results.append(generate_stl(
            p, _out_path(p, outdir),
            colors=s["colors"], fill_holes=bool(s["fill_holes"]), color_tolerance=s["color_tolerance"],
            length_mm=float(s["length_mm"]), thickness_mm=float(s["thickness_mm"]),
            simplify_tolerance_px=float(s["simplify_tolerance_px"]), min_area_px=float(s["min_area_px"]),
        ))
    return results


# -- section 2.8 concurrency guard ------------------------------------------------
#: This CLI's id in the postprocessing registry. The registry owns the tool's ``access``
#: (``read_write`` here) and its ``target_keys``, so the guard below is driven by the SAME
#: metadata the HTTP layer uses instead of a second description of what this tool touches.
TOOL_ID = "generate_stl_from_mask"


def _normalize_target(value):
    """Realpath one CLI-supplied target so it can be compared with the active run's directory.

    Not policy -- spelling. ``postprocess_api.active_run_target`` realpaths the record's artifact
    dir because every HTTP target arrives realpathed through ``resolve_path``; a CLI path reaches
    the guard raw, and two spellings of one directory (a symlinked output root is the common case
    on a cluster) never meet. Only the resolution is borrowed: the sandbox half of ``resolve_path``
    is an HTTP-input concern and applying it here would refuse CLI targets that are legal today.
    """
    if value is None or value == "":
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_target(v) for v in value]
    try:
        return os.path.realpath(os.path.expanduser(str(value)))
    except (OSError, ValueError):                    # an unresolvable path is left as written
        return str(value)


@contextlib.contextmanager
def guarded_targets(paths, output_dir):
    """Hold plan section 2.8's guarantees around this tool's write, or raise.

    Both of section 2.8's rules, through the ONE implementation in
    :mod:`leafmachine3.server.postprocess_api` -- the last bullet of that section requires a
    standalone CLI to use the HTTP API's guard and not a parallel one:

    1. ``check_target_allowed`` refuses a target that is (or contains, or sits inside) the run a
       pipeline is writing right now, and raises :class:`~...postprocess_api.TargetActive`;
    2. ``_acquire_artifact_locks`` takes the per-run advisory lock in the LOCAL deployment runtime
       registry, so two ``read_write`` tools aimed at one completed run serialize instead of
       interleaving, and raises :class:`~...postprocess_api.TargetLocked` when another holds it.

    The lock helper is private today; calling it is still correct -- re-implementing the lock here
    is exactly the "parallel guard" the plan forbids. See ``follow_ups``: postprocess_api should
    export this context manager and both CLIs should then call the exported name.

    Importing ``postprocess_api`` is lazy and cheap (stdlib plus ``core.paths``; no fastapi, no
    torch) and it must stay lazy, because that module's runner imports this one.
    """
    from leafmachine3.server import postprocess_api

    tool = postprocess_api.get_tool(TOOL_ID)
    supplied = {"paths": paths, "output_dir": output_dir}
    unknown = [k for k in tool.target_keys if k not in supplied]
    if unknown:
        # A tripwire, not defensive noise: if the registry grows a target this CLI does not pass,
        # the guard would silently cover only part of what the tool writes.
        raise RuntimeError(
            f"{TOOL_ID}: the registry declares target key(s) {unknown} that this CLI does not "
            f"supply; the section 2.8 guard would only see part of the target"
        )
    params = {key: _normalize_target(supplied[key]) for key in tool.target_keys}
    targets = postprocess_api.check_target_allowed(tool, params)
    locks = postprocess_api._acquire_artifact_locks(tool, targets)
    try:
        yield targets
    finally:
        for lock in locks:
            lock.release()


def _colors_from_cli(tokens) -> list:
    """`--colors white` -> ['white']; `--colors 255 255 255` -> [[255,255,255]] (one RGB color)."""
    try:
        return [[int(t) for t in tokens]]
    except ValueError:
        return list(tokens)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate a 3D-printable .stl by extruding a binary-mask PNG.")
    ap.add_argument(
        "--config", default=None,
        help="postprocessing settings YAML (default: the deployment's canonical "
             "postprocessing.yaml -- plan section 3.1 row 3, never the working directory)")
    ap.add_argument("--paths", nargs="+", default=None, help="mask PNG path(s); overrides the yaml `paths`")
    ap.add_argument("--output-dir", default=None, help="output dir (default: next to each mask)")
    ap.add_argument("--length-mm", type=float, default=None)
    ap.add_argument("--thickness-mm", type=float, default=None)
    ap.add_argument("--fill-holes", dest="fill_holes", action="store_true", default=None)
    ap.add_argument("--no-fill-holes", dest="fill_holes", action="store_false")
    ap.add_argument("--colors", nargs="+", default=None, help="e.g. 'white', or '255 255 255' (one color)")
    ap.add_argument("--color-tolerance", type=int, default=None)
    ap.add_argument("--simplify-tolerance-px", type=float, default=None)
    ap.add_argument("--min-area-px", type=float, default=None)
    args = ap.parse_args(argv)

    from leafmachine3.postprocessing.config import load_settings, module_settings

    # A CLI main() is a controlled entry point -- once per process, user-initiated -- so the
    # one-release adopt of a checkout-level postprocessing_settings.yaml belongs here and not
    # in load_settings(), which must stay a pure read.
    if args.config is None:
        from leafmachine3.core.paths import PathsError, migrate_legacy_postprocessing_settings
        try:
            migrate_legacy_postprocessing_settings()
        except PathsError:
            pass                      # row 3 on-miss is "packaged defaults", never a crash

    s = module_settings(load_settings(args.config), "generate_stl_from_mask")
    overrides = {
        "length_mm": args.length_mm, "thickness_mm": args.thickness_mm, "fill_holes": args.fill_holes,
        "color_tolerance": args.color_tolerance, "simplify_tolerance_px": args.simplify_tolerance_px,
        "min_area_px": args.min_area_px,
    }
    for k, v in overrides.items():
        if v is not None:
            s[k] = v
    if args.colors is not None:
        s["colors"] = _colors_from_cli(args.colors)

    # The EFFECTIVE targets, resolved the same way ``run`` resolves them: a yaml-configured
    # ``paths`` writes into a run just as surely as ``--paths`` does, so the guard must see the
    # values that will actually be written, not only the flags that happened to be typed.
    paths = args.paths if args.paths is not None else s.get("paths")
    output_dir = args.output_dir if args.output_dir is not None else s.get("output_dir")

    from leafmachine3.server import postprocess_api

    try:
        with guarded_targets(paths, output_dir):
            results = run(s, paths=paths, output_dir=output_dir)
    except (postprocess_api.TargetActive, postprocess_api.TargetLocked) as exc:
        # Section 2.8 is a "come back later", not a broken input -- say so on stderr and exit with
        # a code a caller can branch on.
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED
    for r in results:
        print(f"  {Path(r['stl']).name}  size={r['size_mm']}mm  parts={r['n_parts']}  "
              f"watertight={r['watertight']}")
    print(f"wrote {len(results)} stl file(s)"
          + (f" -> {args.output_dir}" if args.output_dir else ""))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
