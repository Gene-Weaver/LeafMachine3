"""settings_meta.json -- the contract between the pipeline and the settings UI.

The settings tab renders one section per pipeline module, keyed to
``pipeline.STAGE_KEYS`` through each section's ``stage`` field, and it routes
every leaf of the effective config to a section by longest-prefix match. Both of
those can fail SILENTLY:

  * a module with no section falls through to ``S.sections[0]`` and its settings
    appear under "Project" -- which is exactly what happened to Bilateral
    Symmetry, whose five settings sat in the wrong place, unlabeled and with no
    help text, for as long as the module existed;
  * a setting with no metadata still renders, but with a humanized key for a
    label and no explanation of what it does.

Neither shows up as an error anywhere. These tests are the only thing that
notices, so adding a module means adding its section, and adding a setting means
adding its metadata.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

from leafmachine3.pipeline import STAGE_KEYS
from leafmachine3.server.settings_api import read_settings

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
UI = REPO_ROOT / "leafmachine3" / "server" / "ui"
META_PATH = UI / "settings_meta.json"
MODULES_JS = UI / "js" / "modules.js"
# Named explicitly, never resolved from the CWD: settings_api.settings_path()
# falls back to $LM3_SETTINGS_PATH and then ./LM3_settings.yaml, so a bare
# read_settings() would validate whichever config the invoking shell points at.
SETTINGS_YAML = REPO_ROOT / "LM3_settings.yaml"

VALID_PHASES = {"setup", "sheet", "scale", "leaf", "output"}


@pytest.fixture(scope="module")
def meta() -> dict:
    return json.loads(META_PATH.read_text())


@pytest.fixture(scope="module")
def sections(meta) -> list[dict]:
    return meta["_sections"]


@pytest.fixture(scope="module")
def leaf_paths() -> list[str]:
    """Every dotted path the settings form will render a row for."""
    effective = read_settings(SETTINGS_YAML, with_text=False)["effective"]

    def walk(node, parts, out):
        if isinstance(node, dict) and node:
            for key, value in node.items():
                walk(value, parts + [key], out)
            return out
        if parts:
            out.append(".".join(parts))
        return out

    return walk(effective, [], [])


def _section_for(path: str, sections: list[dict]) -> dict | None:
    """The UI's longest-prefix match, reimplemented (settings.js sectionFor)."""
    best, best_len = None, -1
    for section in sections:
        for key in section["keys"]:
            if (path == key or path.startswith(key + ".")) and len(key) > best_len:
                best, best_len = section, len(key)
    return best


# --------------------------------------------------------------------------- #
# sections mirror the pipeline
# --------------------------------------------------------------------------- #

def test_every_pipeline_stage_has_exactly_one_section(sections):
    by_stage: dict[str, list[str]] = {}
    for section in sections:
        stage = section.get("stage")
        if stage:
            by_stage.setdefault(stage, []).append(section["id"])

    missing = sorted(set(STAGE_KEYS) - set(by_stage))
    assert not missing, (
        f"pipeline stages with no settings section: {missing}. Their settings will "
        f"fall through to the first section instead of appearing under their own "
        f"module. Add a _sections entry with \"stage\": \"<key>\"."
    )
    duplicated = {s: ids for s, ids in by_stage.items() if len(ids) > 1}
    assert not duplicated, f"more than one section claims the same stage: {duplicated}"


def test_no_section_names_a_stage_the_pipeline_does_not_have(sections):
    unknown = sorted(
        section["stage"] for section in sections
        if section.get("stage") and section["stage"] not in STAGE_KEYS
    )
    assert not unknown, (
        f"sections point at stages that are not in pipeline.STAGE_ORDER: {unknown}"
    )


def test_module_sections_declare_their_run_order(sections):
    from leafmachine3.pipeline import STAGE_ORDER

    expected = {cls.key: i + 1 for i, cls in enumerate(STAGE_ORDER)}
    wrong = {
        section["id"]: (section.get("order"), expected[section["stage"]])
        for section in sections
        if section.get("stage") and section.get("order") != expected[section["stage"]]
    }
    assert not wrong, (
        f"section `order` disagrees with pipeline.STAGE_ORDER (got, want): {wrong}. "
        f"The rail prints this as the run-order badge, so a wrong value is a lie "
        f"about when the module runs."
    )


def test_every_stage_has_a_builtin_default():
    """A stage with no entry in builtin_defaults() is OFF, and the UI cannot tell.

    Config.is_enabled() is `bool(node and node.get("enabled", False))`, so an
    absent module block means the pipeline skips the stage -- while the settings
    form, which renders the merged tree, shows a switch that reads ON because the
    value is `undefined`, not `false`. Bilateral Symmetry shipped exactly that
    way: 16 of 17 stages ran and the app reported 17.
    """
    from leafmachine3.core.config import builtin_defaults

    mods = builtin_defaults().get("modules", {})
    missing = sorted(set(STAGE_KEYS) - set(mods))
    assert not missing, (
        f"stages with no builtin default, so they silently default to DISABLED: {missing}"
    )
    off = sorted(k for k, v in mods.items() if v.get("enabled") is not True)
    assert not off, f"stages whose builtin default is not enabled:True: {off}"


def test_builtin_defaults_have_no_empty_mappings():
    """An empty dict is a LEAF to the settings form, not an empty branch.

    walkLeaves() only recurses into objects with at least one key, so a
    `"report": {}` placeholder rendered as a free-text row containing "{}" on a
    fresh install -- and typing in it wrote a string over the whole report block.
    """
    from leafmachine3.core.config import builtin_defaults

    empties = []

    def walk(node, parts):
        if isinstance(node, dict):
            if not node and parts:
                empties.append(".".join(parts))
            for key, value in node.items():
                walk(value, parts + [key])

    walk(builtin_defaults(), [])
    assert not empties, (
        f"empty mappings in builtin_defaults() render as bogus text rows: {empties}"
    )


def test_sections_have_a_known_phase(sections):
    bad = {s["id"]: s.get("phase") for s in sections if s.get("phase") not in VALID_PHASES}
    assert not bad, f"sections with an unknown rail phase: {bad} (valid: {sorted(VALID_PHASES)})"


_MODULE_ENTRY = re.compile(
    r'\{\s*key:\s*"(?P<key>[a-z0-9_]+)".*?'
    r'\border:\s*(?P<order>\d+).*?'
    r'\bphase:\s*"(?P<phase>[a-z]+)".*?'
    r'\brailOrder:\s*(?P<rail>\d+)\s*,',
    re.S,
)


@pytest.fixture(scope="module")
def modules_js() -> dict[str, dict]:
    """The MODULES table in js/modules.js, parsed.

    Regex over JS is only safe because the assertions below pin the COUNT and
    the key set: a malformed or reformatted table drops entries, and a dropped
    entry fails those instead of silently shrinking the corpus this file checks.
    """
    entries = {
        m.group("key"): {
            "order": int(m.group("order")),
            "phase": m.group("phase"),
            "railOrder": int(m.group("rail")),
        }
        for m in _MODULE_ENTRY.finditer(MODULES_JS.read_text())
    }
    assert len(entries) == len(STAGE_KEYS), (
        f"parsed {len(entries)} entries out of js/modules.js but the pipeline has "
        f"{len(STAGE_KEYS)} stages. Either a module is missing or the table was "
        f"reformatted past what this parser handles -- fix whichever it is, do not "
        f"loosen the parser."
    )
    return entries


def test_modules_js_lists_the_same_seventeen_modules(modules_js):
    """js/modules.js feeds both the stage bar and the settings rail."""
    listed = set(modules_js)
    assert listed == set(STAGE_KEYS), (
        f"js/modules.js and pipeline.STAGE_ORDER disagree.\n"
        f"  only in modules.js: {sorted(listed - set(STAGE_KEYS))}\n"
        f"  only in pipeline:   {sorted(set(STAGE_KEYS) - listed)}"
    )


def test_modules_js_run_order_matches_the_pipeline(modules_js):
    from leafmachine3.pipeline import STAGE_ORDER

    expected = {cls.key: i + 1 for i, cls in enumerate(STAGE_ORDER)}
    wrong = {k: (v["order"], expected[k]) for k, v in modules_js.items()
             if v["order"] != expected[k]}
    assert not wrong, f"js/modules.js `order` disagrees with STAGE_ORDER (got, want): {wrong}"


def test_modules_js_and_settings_meta_agree_on_phase(modules_js, sections):
    """The two files that decide where a module appears must not drift apart.

    modules.js supplies the rail's phase HEADINGS and its sort; settings_meta.json
    supplies each section's phase. Nothing at runtime reconciles them -- a module
    filed under "leaf" in one and "output" in the other simply sorts into a
    heading it does not belong under.
    """
    meta_phase = {s["stage"]: s.get("phase") for s in sections if s.get("stage")}
    mismatched = {
        key: (entry["phase"], meta_phase.get(key))
        for key, entry in modules_js.items()
        if meta_phase.get(key) != entry["phase"]
    }
    assert not mismatched, (
        f"phase disagreement (modules.js, settings_meta.json): {mismatched}"
    )


def test_rail_order_is_a_dense_permutation(modules_js):
    """railOrder drives the rail's sort; a duplicate or a gap makes it arbitrary."""
    rails = sorted(v["railOrder"] for v in modules_js.values())
    assert rails == list(range(1, len(modules_js) + 1)), (
        f"railOrder must be 1..{len(modules_js)} with no gaps or duplicates, got {rails}"
    )


def test_rail_order_groups_phases_contiguously(modules_js):
    """A phase heading is emitted whenever the phase changes while walking the rail.

    So if a phase's modules are not contiguous in railOrder, that heading appears
    twice and the rail reads as though there were two "Leaf" sections.
    """
    walk = sorted(modules_js.values(), key=lambda v: v["railOrder"])
    seen: list[str] = []
    for entry in walk:
        if not seen or seen[-1] != entry["phase"]:
            seen.append(entry["phase"])
    assert len(seen) == len(set(seen)), (
        f"phases are interleaved in railOrder, so a heading would repeat: {seen}"
    )


# --------------------------------------------------------------------------- #
# every rendered row is documented and reachable
# --------------------------------------------------------------------------- #

def test_every_setting_has_metadata(meta, leaf_paths):
    missing = sorted(p for p in leaf_paths if p not in meta)
    assert not missing, (
        f"{len(missing)} settings render with no metadata, so they get a guessed "
        f"label, a guessed control type and no help text: {missing}"
    )


def test_no_metadata_for_settings_that_do_not_exist(meta, leaf_paths):
    """Every documented setting must exist -- except ones explicitly marked ``"retired": true``.

    A retired key keeps its entry on purpose: an older settings file that still carries it renders
    a row that explains the retirement instead of a guessed label with no help text.
    """
    known = set(leaf_paths)
    stale = sorted(k for k, v in meta.items()
                   if k != "_sections" and k not in known and not (isinstance(v, dict) and v.get("retired")))
    assert not stale, f"settings_meta.json documents settings that no longer exist: {stale}"


def test_retired_settings_say_so_and_are_not_in_the_default_file(meta, leaf_paths):
    retired = [k for k, v in meta.items() if isinstance(v, dict) and v.get("retired")]
    for key in retired:
        assert "retired" in meta[key]["label"].lower(), f"{key} is retired but its label does not say so"
        assert key not in set(leaf_paths), f"{key} is retired but the default settings still set it"


def test_every_setting_routes_to_a_section(sections, leaf_paths):
    unrouted = sorted(p for p in leaf_paths if _section_for(p, sections) is None)
    assert not unrouted, (
        f"{len(unrouted)} settings match no section's keys and would silently land "
        f"in whichever section happens to be first: {unrouted}"
    )


def test_reporter_subsections_cover_the_reporter(sections, leaf_paths):
    """The Reporter's 12 declared sub-sections must not lose any of its settings.

    A path the declared list forgets still gets an "Other settings" catch-all in
    the UI, so nothing becomes unreachable -- but the catch-all is a symptom, not
    a design, and it should never fire in a shipped file.
    """
    reporter = next(s for s in sections if s.get("stage") == "reporter")
    declared = {p for sub in reporter.get("subsections", []) for p in sub["paths"]}
    assert declared, "the Reporter section lost its `subsections` list"

    owned = [p for p in leaf_paths if _section_for(p, sections) is reporter]
    orphans = sorted(
        p for p in owned
        if not any(p == d or p.startswith(d + ".") for d in declared)
    )
    assert not orphans, (
        f"Reporter settings in no declared sub-section (they will show up under "
        f"\"Other settings\"): {orphans}"
    )


def _rendered_group_paths(section: dict, sections: list[dict], leaf_paths: list[str]) -> set[str]:
    """The group paths renderNode actually emits keys for, for one section.

    Reimplements buildSectionModel (settings.js): strip the prefix EVERY leaf
    shares, then fuse any branch that has no leaves of its own and exactly one
    child into that child. A declared sub-section path has to survive both, or
    declaredRailGroups' `byKey.get("<id>:<path>")` misses and the whole
    sub-section silently empties into the "Other settings" catch-all.

    Prefix-matching a leaf path is NOT the same test: "report.overlay.classes"
    prefix-matches plenty of leaves but would not survive fusion if it had a
    single child, and a leaf's own path prefix-matches itself while never being
    a group at all.
    """
    owned = [p for p in leaf_paths if _section_for(p, sections) is section]
    if not owned:
        return set()

    parts = [p.split(".") for p in owned]
    strip = parts[0][:-1]
    for pieces in parts:
        parent = pieces[:-1]
        i = 0
        while i < len(strip) and i < len(parent) and strip[i] == parent[i]:
            i += 1
        strip = strip[:i]

    # tree of branch nodes: absolute path -> {"kids": {...}, "leaves": int}
    root: dict = {"kids": {}, "leaves": 0, "path": None}
    for pieces in parts:
        rel = pieces[len(strip):]
        node = root
        for i in range(len(rel) - 1):
            key = rel[i]
            node = node["kids"].setdefault(
                key, {"kids": {}, "leaves": 0, "path": ".".join(strip + rel[: i + 1])}
            )
        node["leaves"] += 1

    def fuse(node):
        for key, child in list(node["kids"].items()):
            fuse(child)
            if child["leaves"] == 0 and len(child["kids"]) == 1:
                grand = next(iter(child["kids"].values()))
                del node["kids"][key]
                node["kids"][grand["path"]] = grand

    fuse(root)

    out: set[str] = set()

    def collect(node):
        for child in node["kids"].values():
            out.add(child["path"])
            collect(child)

    collect(root)
    return out


def test_declared_subsection_paths_resolve_to_real_rail_groups(sections, leaf_paths):
    for section in sections:
        subs = section.get("subsections")
        if not subs:
            continue
        rendered = _rendered_group_paths(section, sections, leaf_paths)
        for sub in subs:
            for path in sub["paths"]:
                assert path in rendered, (
                    f"section {section['id']!r} sub-section {sub['label']!r} names "
                    f"{path!r}, which is not a group the settings form renders "
                    f"(it is either a leaf, or a branch that the common-prefix strip "
                    f"or the single-child fusion removed). declaredRailGroups would "
                    f"skip it and its settings would fall into \"Other settings\".\n"
                    f"groups this section actually renders: {sorted(rendered)}"
                )


def test_own_subsections_actually_have_own_leaves(sections, leaf_paths):
    """`own: true` takes a group's LOOSE leaves; a group with none yields nothing."""
    for section in sections:
        for sub in section.get("subsections", []):
            if not sub.get("own"):
                continue
            owned_directly = [
                p for p in leaf_paths
                for base in sub["paths"]
                if p.startswith(base + ".") and "." not in p[len(base) + 1:]
            ]
            assert owned_directly, (
                f"sub-section {sub['label']!r} is declared own:true but none of "
                f"{sub['paths']} has any direct leaves, so it renders empty"
            )


# --------------------------------------------------------------------------- #
# metadata quality
# --------------------------------------------------------------------------- #

def test_metadata_entries_are_well_formed(meta):
    allowed_types = {"bool", "int", "float", "string", "enum", "list", "path", "color"}
    problems = []
    for key, entry in meta.items():
        if key == "_sections":
            continue
        if not entry.get("label"):
            problems.append(f"{key}: no label")
        if entry.get("type") not in allowed_types:
            problems.append(f"{key}: type {entry.get('type')!r} is not one of {sorted(allowed_types)}")
        if entry.get("type") == "enum" and not entry.get("enum"):
            problems.append(f"{key}: declared enum but lists no choices")
    assert not problems, "\n".join(problems)


def test_help_text_is_american_english(meta):
    """LM3 ships US spelling; a mixed corpus reads like two products."""
    british = ("colour", "normalise", "normalised", "behaviour", "centre",
               "optimise", "optimised", "analyse", "grey")
    hits = []
    for key, entry in meta.items():
        if key == "_sections":
            continue
        blob = f"{entry.get('label', '')} {entry.get('help', '')}".lower()
        hits += [f"{key}: {w!r}" for w in british if w in blob]
    for section in meta["_sections"]:
        blob = f"{section.get('label', '')} {section.get('blurb', '')}".lower()
        hits += [f"_sections/{section['id']}: {w!r}" for w in british if w in blob]
    assert not hits, "British spellings in settings metadata:\n" + "\n".join(hits)
