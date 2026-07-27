"""Canonical leaf-landmark keypoint structure for the LM3 pose model (mid15_pet5, 31 kpts).

Baked from `LM3_Landmark_Detector` (yolo26x-pose). Single source of truth for the keypoint
names/order, semantic groups, skeleton (relationships between points), horizontal-flip index,
and the white-padding fraction the model was trained with. `ProjectDB.init_schema` seeds the
`landmark_schema` / `landmark_skeleton` reference tables from here, so stored landmarks are
self-describing and can be referenced by name/relationship elsewhere.

Keypoint index order (0..30):
    0        lamina_tip
    1,2,3    apex_left, apex_center, apex_right          (apex angle)
    4..18    midvein_0..14                               (midvein trace, oriented tip->base)
    19,20,21 base_left, base_center, base_right          (base angle)
    22       lamina_base
    23..27   petiole_0..4                                (petiole trace, oriented base->tip)
    28       petiole_tip
    29,30    width_left, width_right                     (lamina width)
"""
from __future__ import annotations

MIDVEIN_N = 15
PETIOLE_N = 5

KPT_NAMES: list[str] = (
    ["lamina_tip", "apex_left", "apex_center", "apex_right"]
    + [f"midvein_{i}" for i in range(MIDVEIN_N)]
    + ["base_left", "base_center", "base_right", "lamina_base"]
    + [f"petiole_{i}" for i in range(PETIOLE_N)]
    + ["petiole_tip", "width_left", "width_right"]
)
N_KPTS = len(KPT_NAMES)                                   # 31
KPT_INDEX: dict[str, int] = {n: i for i, n in enumerate(KPT_NAMES)}

# horizontal-flip index (swap L/R of apex / base / width; traces + centers map to self)
FLIP_IDX: list[int] = [0, 3, 2, 1, *range(4, 19), 21, 20, 19, 22, *range(23, 29), 30, 29]

# the pose model is trained on leaf crops with a WHITE border of this fraction per side; the
# inference wrapper re-adds it and then subtracts the offset (coords "as if never padded").
WHITE_PAD_FRAC = 0.10


def _group(name: str) -> str:
    for prefix in ("midvein", "petiole", "apex", "base", "width"):
        if name.startswith(prefix):
            return prefix
    return "lamina"                                       # lamina_tip / lamina_base


KPT_GROUP: dict[str, str] = {n: _group(n) for n in KPT_NAMES}

# ordered polylines (traces) for reference / downstream reassembly
TRACES: dict[str, list[str]] = {
    "midvein": [f"midvein_{i}" for i in range(MIDVEIN_N)],
    "petiole": [f"petiole_{i}" for i in range(PETIOLE_N)],
}


def _chain(prefix: str, n: int, kind: str) -> list[tuple[str, str, str]]:
    return [(f"{prefix}_{i}", f"{prefix}_{i + 1}", kind) for i in range(n - 1)]


# skeleton = relationships between keypoints as (a_name, b_name, kind). Seeded into
# landmark_skeleton (a/b -> kpt_index). Used for drawing + trace/angle reconstruction.
SKELETON: list[tuple[str, str, str]] = (
    # midvein (midrib): lamina_tip -> midvein_0..14 -> lamina_base
    [("lamina_tip", "midvein_0", "midvein")]
    + _chain("midvein", MIDVEIN_N, "midvein")
    + [(f"midvein_{MIDVEIN_N - 1}", "lamina_base", "midvein")]
    # petiole: lamina_base -> petiole_0..4 -> petiole_tip
    + [("lamina_base", "petiole_0", "petiole")]
    + _chain("petiole", PETIOLE_N, "petiole")
    + [(f"petiole_{PETIOLE_N - 1}", "petiole_tip", "petiole")]
    # apex angle (vertex apex_center) and base angle (vertex base_center)
    + [("apex_left", "apex_center", "apex"), ("apex_center", "apex_right", "apex")]
    + [("base_left", "base_center", "base"), ("base_center", "base_right", "base")]
    # lamina width and lamina length (tip -> base)
    + [("width_left", "width_right", "width")]
    + [("lamina_tip", "lamina_base", "lamina_length")]
)

assert len(KPT_INDEX) == N_KPTS == 31
