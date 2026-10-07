#!/usr/bin/env python3
"""Parse TODO.md -> todo.json (structured) and inject it into todo_template.html
-> todo.html (self-contained, editable).  After this, **todo.json is the source of
truth**: Claude edits it to add/clear items; the HTML edits save back to it (File
System Access API, or localStorage + download fallback).  Re-run only to re-seed
todo.html's embedded default from todo.json (it reads todo.json, not TODO.md, if
todo.json already exists)."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "TODO.md")
JSON_OUT = os.path.join(HERE, "todo.json")
TEMPLATE = os.path.join(HERE, "todo_template.html")
HTML_OUT = os.path.join(HERE, "todo.html")


def parse_md(md):
    lines = md.split("\n")
    h1 = next((i for i, l in enumerate(lines) if l.startswith("# ")), -1)
    title = lines[h1][2:].strip() if h1 >= 0 else "TODO"
    first = next((i for i, l in enumerate(lines) if l.startswith("## ")), len(lines))
    intro = "\n".join(lines[h1 + 1:first]).strip()
    idxs = [i for i, l in enumerate(lines) if l.startswith("## ")]
    items = []
    for j, start in enumerate(idxs):
        end = idxs[j + 1] if j + 1 < len(idxs) else len(lines)
        header = lines[start][3:].strip()
        m = re.match(r"(\d+)\.\s*(.*)", header)
        num, heading = (m.group(1), m.group(2)) if m else (str(j + 1), header)
        body = "\n".join(lines[start + 1:end]).strip()
        body = re.sub(r"\n?-{3,}\s*$", "", body).strip()          # drop trailing hr separator
        done = bool(re.search(r"✅|\bDONE\b", heading))
        items.append({"id": f"i{num}", "num": num, "heading": heading, "body": body,
                      "done": done, "deleted": False, "notes": []})
    return {"title": title, "intro": intro, "items": items, "settings": {"hideDone": False}}


# Prefer an existing todo.json (post-migration source of truth); else parse TODO.md.
if os.path.exists(JSON_OUT):
    data = json.load(open(JSON_OUT, encoding="utf-8"))
    print("re-using existing todo.json as source (", len(data.get("items", [])), "items )")
else:
    data = parse_md(open(MD, encoding="utf-8").read())
    json.dump(data, open(JSON_OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("parsed TODO.md -> todo.json (", len(data["items"]), "items )")

tpl = open(TEMPLATE, encoding="utf-8").read()
seed = json.dumps(data, ensure_ascii=False)
assert "</script" not in seed.lower(), "seed contains </script> -- would break the embed"
open(HTML_OUT, "w", encoding="utf-8").write(tpl.replace("__SEED__", seed))
print("wrote", HTML_OUT)
