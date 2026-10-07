#!/usr/bin/env python3
"""Dump the REAL settings corpus as JSON for the layout experiments.

Reads the same three sources the Settings tab does -- ui/settings_meta.json (labels, help, types,
groups, sections), core.config.builtin_defaults() (defaults), and LM3_settings.yaml (current values)
-- and flattens them into one array. Section assignment mirrors settings.js:558 sectionFor():
longest-prefix match against _sections[].keys, falling back to the first section.

    .venv_LM3/bin/python exp__settingslayouts/_extract.py > exp__settingslayouts/settings.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

META = ROOT / "leafmachine3" / "server" / "ui" / "settings_meta.json"
YAML = ROOT / "LM3_settings.yaml"


def flatten(node, prefix=""):
    """Every LEAF of a nested dict as ``dotted.path -> value``."""
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) and v:
                out.update(flatten(v, p))
            else:
                out[p] = v
    return out


def main() -> int:
    meta = json.loads(META.read_text())
    sections = meta.pop("_sections")

    from leafmachine3.core.config import builtin_defaults
    defaults = flatten(builtin_defaults())

    live = {}
    if YAML.is_file():
        import yaml as _y
        live = flatten(_y.safe_load(YAML.read_text()) or {})

    def section_for(path: str) -> str:
        """Longest-prefix match, exactly like settings.js sectionFor()."""
        best, best_len = sections[0]["id"], -1
        for sec in sections:
            for key in sec.get("keys", []):
                if (path == key or path.startswith(key + ".")) and len(key) > best_len:
                    best, best_len = sec["id"], len(key)
        return best

    rows = []
    for path in sorted(set(meta) | set(defaults)):
        m = meta.get(path)
        if not m:
            continue                        # a live leaf with no metadata is not shown by the tab
        dflt = defaults.get(path)
        val = live.get(path, dflt)
        row = {
            "k": path,
            "l": m.get("label") or path.rsplit(".", 1)[-1],
            "t": m.get("type", "string"),
            "h": m.get("help", ""),
            "i": 1 if m.get("important") else 0,
            "g": m.get("group", ""),
            "s": section_for(path),
            "v": val,
            "d": dflt,
        }
        # "Changed" only counts where a default is actually KNOWN. builtin_defaults() is a skeleton
        # -- 242 of 290 leaves have no entry -- so comparing against None would paint the whole
        # file as drifted when in fact only three keys differ.
        if dflt is not None and json.dumps(val, default=str) != json.dumps(dflt, default=str):
            row["c"] = 1
        for src, dst in (("enum", "e"), ("min", "mn"), ("max", "mx")):
            if src in m:
                row[dst] = m[src]
        rows.append(row)

    json.dump({"sections": sections, "rows": rows}, sys.stdout, separators=(",", ":"), default=str)
    print(f"\n-- {len(rows)} settings, {len(sections)} sections", file=sys.stderr)
    changed = [r["k"] for r in rows if json.dumps(r["v"], default=str) != json.dumps(r["d"], default=str)]
    print(f"-- {len(changed)} differ from default", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
