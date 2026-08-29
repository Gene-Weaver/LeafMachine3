"""Unit table + cross-validation, ported from the ground-truth app.

Mirrors LM3_Ruler_Distance_Groundtruth/{config,crossval}.py so the segmentation
pipeline reconciles detected tick periods against the SAME known physical
relationships the GT labelling app uses (1 cm = 10 mm, 1 in = 2.54 cm,
1/16 in = 1/16 in, ...) instead of inventing its own.

The key idea taken from crossval.py: when a ruler shows ticks at several
spacings, each spacing is an independent estimate of px/cm once you know which
named unit it is. Their agreement is the confidence signal.
"""
from __future__ import annotations

import math

MM_PER_INCH = 25.4

# physical size of ONE unit, in mm -- verbatim from the GT app's MINIMUM_UNITS
MINIMUM_UNITS: dict[str, dict] = {
    "metric__2_MM": {"system": "metric", "mm": 0.5},
    "metric__MM":   {"system": "metric", "mm": 1.0},
    "metric__4_CM": {"system": "metric", "mm": 2.5},   # a QUARTER of a cm
    "metric__2_CM": {"system": "metric", "mm": 5.0},
    "metric__CM":   {"system": "metric", "mm": 10.0},
    "metric__MM_CM": {"system": "metric", "mm": 1.0, "dual": True},
    "std__32_IN":   {"system": "std", "inch": 1.0 / 32},
    "std__16_IN":   {"system": "std", "inch": 1.0 / 16},
    "std__8_IN":    {"system": "std", "inch": 1.0 / 8},
    "std__4_IN":    {"system": "std", "inch": 1.0 / 4},
    "std__2_IN":    {"system": "std", "inch": 1.0 / 2},
    "std__IN":      {"system": "std", "inch": 1.0},
}
# tolerance constants, same values as the GT app
MULT_TOL = 0.10      # a detected ratio must be within 10% of a named unit ratio
AGREE_TOL = 0.03     # independent per-unit px/cm estimates must agree within 3%


#: Unit keys that were renamed. ``metric__4_MM`` was the 2.5 mm unit -- a QUARTER of a cm -- but
#: was spelled with an MM suffix while every other key is 1/N of the unit it names (2_MM = 0.5 mm,
#: 2_CM = 5 mm, 32_IN = 1/32 in). It reads as "4 mm" and is not. The classifier already calls this
#: class METRIC_CM4, which is the right convention, so only the unit key moved.
#:
#: The old name is still resolved because it is PERSISTED: ``admissible_units``, ``unit_estimates``
#: and ``cf_contributions`` on ruler_CF_lattice_crop hold it verbatim, and the Reporter rebuilds QC
#: panels from those rows. Dropping the alias would KeyError on every pre-rename run.
LEGACY_UNIT_NAMES: dict[str, str] = {"metric__4_MM": "metric__4_CM"}


def canon_unit(u: str) -> str:
    """Current name for a unit key, mapping any renamed-away spelling onto it."""
    return LEGACY_UNIT_NAMES.get(u, u)


def unit_mm(u: str) -> float:
    spec = MINIMUM_UNITS[canon_unit(u)]
    return float(spec["mm"]) if spec["system"] == "metric" \
        else float(spec["inch"]) * MM_PER_INCH


def system_of(u: str) -> str:
    return MINIMUM_UNITS[canon_unit(u)]["system"]


def is_dual(u: str) -> bool:
    return bool(MINIMUM_UNITS.get(canon_unit(u), {}).get("dual"))


def real_units(system: str | None = None) -> list[str]:
    """Named units, excluding the dual-scale flag pseudo-units."""
    return [u for u, s in MINIMUM_UNITS.items()
            if not s.get("dual") and (system is None or s["system"] == system)]


# --------------------------------------------------------------------------- #
# Which units can a ruler of this class actually show?
# --------------------------------------------------------------------------- #
def class_units(unit_class: str) -> list[str]:
    """The units a class declares PRESENT, finest first -- straight from CLASS_SPEC.

    This used to be hardcoded as
        ``[unit_class] if unit_class in MINIMUM_UNITS else ["metric__MM"]``
    which silently returned a 1 mm METRIC default for every CLASSIFIER-namespace
    class (STD_IN16, STD_IN8, METRIC_CM, ...), because those names live in a
    different namespace from the GT app's MINIMUM_UNITS. Two consumers read the
    result -- the minimum-unit prior and the sub-lattice support divisors -- so an
    imperial ruler was scored against a 1 mm base and got no minimum-unit prior at
    all. Never default here; an unknown class has no declared units.
    """
    sp = spec_of(unit_class)
    if sp.get("units"):
        return list(sp["units"])
    return [canon_unit(unit_class)] if canon_unit(unit_class) in MINIMUM_UNITS else []


def base_candidates(unit_class: str) -> list[str]:
    """Plausible identities for the FINEST detected lattice.

    The class names the finest unit the ruler prints -- but that unit is not
    always resolvable in a given crop (mm ticks at 3 px/mm are gone), so the
    finest lattice we actually detect may be a coarser named unit. Candidates
    are therefore the declared unit plus all coarser units of the same system.
    """
    declared = class_units(unit_class)
    sysset = set(class_systems(unit_class)) or {system_of(u) for u in declared}
    finest = min(unit_mm(u) for u in declared)
    # Allow one step FINER than declared as well: the class names the unit a
    # human read off the ruler, but a crop can resolve a finer printed tick
    # (e.g. a "1/8 in" ruler whose row actually carries 1/16 in ticks).
    out = []
    for u in real_units():
        if system_of(u) in sysset and unit_mm(u) >= finest * 0.49:
            out.append(u)
    return sorted(set(out), key=unit_mm)


def match_named(size_mm: float, system: str, tol: float = MULT_TOL):
    """Nearest named unit of `system` to a physical size, or None."""
    best = None
    for u in real_units(system):
        e = abs(size_mm / unit_mm(u) - 1.0)
        if e <= tol and (best is None or e < best[0]):
            best = (e, u)
    return None if best is None else best[1]


# --------------------------------------------------------------------------- #
# Cross-validation across the detected levels
# --------------------------------------------------------------------------- #
# units a ruler is actually NUMBERED in -- the top tick level almost always
# lands on one of these, which is a strong prior on the base identity
NUMBERED = {"metric__CM": 1.0, "std__IN": 1.0, "std__2_IN": 0.4, "metric__2_CM": 0.4}


def reconcile_levels(P_base_px, levels, unit_class, width_px,
                     len_mu=2.5382, len_sig=0.38, acf_mult=None):
    """Identify the finest lattice, then read every coarser level as a named unit.

    P_base_px    px spacing of the finest detected tick lattice
    level_mults  integer multipliers of the coarser tick levels, e.g. [2,4,8]
                 (ticks that are systematically taller / longer)
    Returns the best hypothesis dict, or None.

    Scoring uses only GT-free evidence:
      * how many levels land on a NAMED unit of the ruler's system,
      * whether the declared class unit is used,
      * plausibility of the implied physical ruler length,
      * agreement between the independent px/cm estimates each level gives.
    """
    declared = class_units(unit_class)
    declared_set = set(declared)
    best = None

    for u0 in base_candidates(unit_class):
        sysu = system_of(u0)
        base_mm = unit_mm(u0)
        pxcm = P_base_px / base_mm * 10.0
        if not math.isfinite(pxcm) or pxcm <= 0:
            continue
        L = width_px / pxcm                      # implied physical crop length, cm

        named, ests, labels = 0, [pxcm], {1: u0}
        top_lab, unnamed = u0, 0
        for (m, spacing) in levels:
            size = base_mm * m
            lab = match_named(size, sysu)
            if lab is None:
                unnamed += 1
                continue
            named += 1
            labels[m] = lab
            top_lab = lab
            if spacing and math.isfinite(spacing) and spacing > 0:
                ests.append(spacing / unit_mm(lab) * 10.0)

        # crossval.group_agreement: do the independent per-unit estimates agree?
        spread = ((max(ests) - min(ests)) / (sum(ests) / len(ests))
                  if len(ests) >= 2 else None)

        assigned = set(labels.values())
        nlev = max(1, len(levels))

        score = 0.0
        # The implied physical length is the single most informative term (the
        # 69-crop fit is tight: sigma(log) = 0.38), so it leads.
        score += -0.5 * ((math.log(max(L, 1e-6)) - len_mu) / len_sig) ** 2
        # The declared class unit must appear SOMEWHERE in the assignment, not
        # necessarily as the base -- a mm+cm ruler whose mm ticks are unresolved
        # still legitimately shows cm as a coarser level.
        score += 1.5 if (assigned & declared_set) else 0.0
        score += 0.75 if u0 in declared_set else 0.0
        score += 0.5 if sysu in {system_of(d) for d in declared} else 0.0
        score += 2.0 * NUMBERED.get(top_lab, 0.0)   # rulers are numbered in cm / in
        score += 1.0 * (named / nlev) - 1.0 * (unnamed / nlev)
        if spread is not None:
            score += 1.0 * (1.0 - min(spread / AGREE_TOL, 2.0))
        # the ACF lattice itself should also be a named unit
        if acf_mult and acf_mult > 1:
            if match_named(base_mm * acf_mult, sysu) is not None:
                score += 1.0

        cand = dict(base_unit=u0, pxcm=pxcm, implied_len_cm=L, n_named=named,
                    n_unnamed=unnamed, labels=labels, estimates=ests,
                    spread=spread, agree=(spread is not None and spread <= AGREE_TOL),
                    score=score)
        if best is None or score > best["score"]:
            best = cand
    return best


# --------------------------------------------------------------------------- #
# Scale decision on the ACF lattice (the most reliably detected quantity)
# --------------------------------------------------------------------------- #
METRIC_P = {"1 mm": 0.1, "5 mm": 0.5, "1 cm": 1.0}
STD_P = {'1/16"': 2.54 / 16, '1/8"': 2.54 / 8, '1/4"': 2.54 / 4,
         '1/2"': 2.54 / 2, '1"': 2.54}
LEN_MU, LEN_SIG = 2.5382, 0.3824


def identify_period(P_px, unit_class, width_px, sublattice_support=None,
                    own_bonus=1.0, sup_weight=4.0, extra_candidates=None,
                    min_unit_bonus=3.0, cf_anchor=None, anchor_sigma=0.35):
    """Which physical period is the detected lattice? GT-free.

    A metric-class ruler is barred from imperial candidates (it has no inch
    scale); an imperial-class ruler keeps metric candidates because those rulers
    commonly carry a cm row as well.
    """
    declared = class_units(unit_class)
    sp = spec_of(unit_class)
    # THE class decides what is admissible: its systems, from its finest printed
    # unit up to that system's coarsest major. Nothing else may widen this.
    cands = dict(admissible_periods(unit_class))
    # layouts where a single-row scan does not see the unit period (stagger: two
    # offset rows of cm blocks; block: filled cm block + blank cm gap) contribute
    # their row period as an extra admissible reading.
    if sp.get("row_period_cm"):
        cands[f"{sp['row_period_cm']:g} cm ({sp['layout']} row)"] = float(sp["row_period_cm"])
    if extra_candidates:
        cands = {**cands, **extra_candidates}
    if not cands:
        return None
    # The class declares a printed minimum PER SYSTEM, not one global minimum.
    # Taking `min()` over all declared units is only correct for a single-system
    # class. On a cross-system ruler the two scales are physically INDEPENDENT --
    # STD_IN8_CM is an inch face graduated to 1/8 in and a cm face graduated to
    # 1 cm -- and the class asserts that BOTH are printed. Comparing 3.175 mm with
    # 10 mm across that boundary is meaningless: it handed the "class says this
    # unit is printed" prior to the inch face purely because an eighth of an inch
    # is the smaller number, and demoted the 1 cm graduation -- which the class
    # states outright is present -- to an ordinary coarse-major candidate with no
    # class support at all. It then lost to 1/4 in and 1/2 in, which pick up the
    # sub-lattice bonus because the cm face's own 5 mm ticks sit at exactly half
    # the period. Grouping by system restores the class's actual claim.
    # Single-system classes are unaffected by construction: all their declared
    # units share one system, so the per-system minimum IS the global minimum.
    _min_by_sys: dict[str, float] = {}
    for u in declared:
        s_u = system_of(u)
        if s_u not in _min_by_sys or unit_mm(u) < _min_by_sys[s_u]:
            _min_by_sys[s_u] = unit_mm(u)
    declared_minima = set(_min_by_sys.values())

    best = None
    for name, pcm in cands.items():
        pxcm = P_px / pcm
        if pxcm <= 0:
            continue
        L = width_px / pxcm
        # The crop-length prior always applies. An independent px/cm anchor
        # (LM3 `specimen.cf_px_per_cm_predicted_by_mp`, regressed from the image's
        # megapixel count) is ADDED to it rather than replacing it: measured on the
        # 69 GT crops, replacing costs 3 crops when the anchor is only good to
        # +-35%, because the length prior was carrying real information. Added, a
        # weak anchor cannot do worse than no anchor.
        #
        # The anchor's job is only to pin the harmonic -- log(2) = 0.69 is 2 sigma
        # at sigma = 0.35, so even a rough anchor rules out a factor-of-2 mis-read.
        # It never becomes the estimate; precision still comes from the comb fit.
        s = -0.5 * ((math.log(max(L, 1e-6)) - LEN_MU) / LEN_SIG) ** 2
        if cf_anchor and cf_anchor > 0:
            s += -0.5 * ((math.log(pxcm) - math.log(cf_anchor)) / anchor_sigma) ** 2
        # every candidate is now own-system by construction, so `own_bonus` is
        # vestigial; the class's declared minimum still gets an explicit prior.
        # Weight chosen on the 69 GT crops (0 -> 49/69, 1.6 -> 51/69, 3.0 -> 52/69);
        # only the magnitude is fitted, the direction is a stated fact about the
        # class labels, and the result is flat between 1.6 and 3.0.
        if any(abs(pcm * 10.0 - m) < 1e-6 for m in declared_minima):
            s += min_unit_bonus
        if sublattice_support:
            s += sup_weight * sublattice_support.get(name, 0.0)
        if best is None or s > best[0]:
            best = (s, name, pxcm, L)
    if best is None:
        return None
    sc, name, pxcm, L = best
    # the winning candidate's own system (row-period pseudo-candidates are metric)
    sysu = system_of(name) if name in MINIMUM_UNITS else (
        class_systems(unit_class)[0] if class_systems(unit_class) else "metric")
    # `score` is exported so a caller that offers SEVERAL candidate periods for the
    # same band (the ACF argmax and its sub-harmonics) can compare their
    # identities. Every other term -- implied ruler length, MP-anchor agreement,
    # sub-lattice presence -- is a statement about the resulting px/cm and stays
    # comparable when P changes. `min_unit_bonus` is the exception and such a
    # caller must zero it for the sub-harmonics; see ruler_analysis._identify_best.
    return dict(period_name=name, pxcm=pxcm, implied_len_cm=L, system=sysu,
                score=float(sc), admissible=sorted(cands))


def label_levels(pxcm, P_base_px, levels, system):
    """Name the base lattice and every coarser tick level, given a scale.

    Each level's MEASURED spacing yields its own independent px/cm estimate, so
    their spread is a genuine cross-validation (crossval.group_agreement), not an
    algebraic identity.
    """
    def name_of(px):
        mm = px / pxcm * 10.0
        lab = match_named(mm, system)
        return lab, mm

    base_lab, base_mm = name_of(P_base_px)
    out = [dict(mult=1, spacing=P_base_px, unit=base_lab, mm=base_mm,
                est=(P_base_px / unit_mm(base_lab) * 10.0) if base_lab else None)]
    for (m, spacing) in levels:
        sp = spacing if (spacing and math.isfinite(spacing)) else m * P_base_px
        lab, mm = name_of(sp)
        out.append(dict(mult=m, spacing=sp, unit=lab, mm=mm,
                        est=(sp / unit_mm(lab) * 10.0) if lab else None))
    ests = [d["est"] for d in out if d["est"]]
    spread = ((max(ests) - min(ests)) / (sum(ests) / len(ests))
              if len(ests) >= 2 else None)
    return out, ests, spread, (spread is not None and spread <= AGREE_TOL)


# --------------------------------------------------------------------------- #
# Full reconciliation across the detected tick levels
# --------------------------------------------------------------------------- #
MERGE_TOL = 0.03      # two levels within 3% are measuring the same thing


def rank_units(mm_size, allowed=None):
    """Candidate units ranked by how well they explain a physical size.

    `allowed` MUST be supplied in production -- it is the class's admissible set.
    Ranking over every named unit is what let an imperial ruler resolve to mm.
    """
    pool = list(allowed) if allowed else real_units()
    return sorted(((u, abs(mm_size / unit_mm(u) - 1.0)) for u in pool),
                  key=lambda t: t[1])


def reconcile_units(pxcm_seed, levels, declared_units=(), n_iter=4,
                    allowed_units=None):
    """Reconcile every detected tick level into a coherent physical reading.

    `levels` : [{'mult': int, 'spacing': px, 'n': int}] -- one entry per tick
               level, `spacing` MEASURED from that level's own tick positions.

    Two distinct things get averaged, and both are tracked:

      * SAME-UNIT duplicates -- two levels that resolve to the same named unit
        with slightly different measured spacings (< 3% apart). These are one
        physical unit measured twice, so they collapse into a single averaged
        level.
      * CORROBORATING units -- levels that resolve to *different* units but whose
        independent px/cm estimates still agree within 3%. These are genuine
        cross-validation (a cm row and an inch row agreeing is the strongest
        confirmation available without ground truth), so they are combined into
        the final estimate.

    The unit assignment depends on the scale and the scale depends on the
    assignment, so the whole thing is iterated to a fixed point.
    """
    declared = set(declared_units)
    levels = [dict(lv) for lv in levels if lv.get("spacing") and
              math.isfinite(lv["spacing"]) and lv["spacing"] > 0]
    if not levels:
        return None

    pxcm = float(pxcm_seed)
    groups, assigned = {}, []
    for _ in range(max(1, n_iter)):
        assigned = []
        for lv in levels:
            mm = lv["spacing"] / pxcm * 10.0
            ranked = rank_units(mm, allowed_units)
            u, err = ranked[0]
            assigned.append(dict(mult=lv["mult"], spacing=lv["spacing"],
                                 n=max(1, int(lv.get("n", 1))), mm=mm,
                                 unit=(u if err <= MULT_TOL else None),
                                 unit_err=err,
                                 alt=[(a, round(e, 4)) for a, e in ranked[1:3]]))

        groups = {}
        for a in assigned:
            if a["unit"]:
                groups.setdefault(a["unit"], []).append(a)
        if not groups:
            break

        gsum = {}
        for u, mem in groups.items():
            ests = [m["spacing"] / unit_mm(u) * 10.0 for m in mem]
            ws = [m["n"] for m in mem]
            tot = float(sum(ws))
            est = sum(e * w for e, w in zip(ests, ws)) / tot
            rel = ((max(ests) - min(ests)) / (sum(ests) / len(ests))
                   if len(ests) > 1 else 0.0)
            gsum[u] = dict(unit=u, system=system_of(u), n_levels=len(mem),
                           mults=[m["mult"] for m in mem],
                           n_ticks=int(tot), est_pxcm=est,
                           spacing=sum(m["spacing"] * w for m, w in zip(mem, ws)) / tot,
                           averaged=len(mem) > 1, spread_within=rel,
                           merge_ok=(rel <= MERGE_TOL))
        groups = gsum

        new = (sum(g["est_pxcm"] * g["n_ticks"] for g in groups.values())
               / sum(g["n_ticks"] for g in groups.values()))
        if abs(new / pxcm - 1.0) < 1e-9:
            pxcm = new
            break
        pxcm = new

    if not groups:
        return dict(ok=False, relation="unresolved", pxcm=pxcm_seed,
                    groups=[], assigned=assigned, systems=[],
                    n_same_unit_averaged=0, n_corroborating=0,
                    spread=None, agree=False, declared_seen=False)

    ests = [g["est_pxcm"] for g in groups.values()]
    spread = ((max(ests) - min(ests)) / (sum(ests) / len(ests))
              if len(ests) > 1 else None)
    systems = sorted({g["system"] for g in groups.values()})

    # which distinct units corroborate each other within 3%?
    corro = 0
    gl = list(groups.values())
    for i in range(len(gl)):
        for j in range(i + 1, len(gl)):
            a, b = gl[i]["est_pxcm"], gl[j]["est_pxcm"]
            if abs(a / b - 1.0) <= MERGE_TOL:
                corro += 1

    n_avg = sum(1 for g in groups.values() if g["averaged"])
    if len(groups) == 1:
        u = next(iter(groups))
        relation = ("single unit, averaged" if groups[u]["averaged"]
                    else "single unit")
    elif len(systems) == 2:
        relation = "cross-system (metric + imperial)"
    elif systems == ["metric"]:
        relation = f"metric multi-scale ({len(groups)} units)"
    else:
        relation = f"imperial multi-scale ({len(groups)} units)"
    if n_avg:
        relation += f" +{n_avg} averaged"

    return dict(ok=True, relation=relation, pxcm=pxcm,
                groups=sorted(groups.values(), key=lambda g: unit_mm(g["unit"])),
                assigned=assigned, systems=systems,
                n_same_unit_averaged=n_avg, n_corroborating=corro,
                spread=spread, agree=(spread is not None and spread <= AGREE_TOL),
                declared_seen=bool(declared & set(groups)))


# --------------------------------------------------------------------------- #
# Explicit registry of EVERY ruler class
# --------------------------------------------------------------------------- #
# layout:
#   nested      one scale; coarser units marked by LONGER ticks on the same row
#   transition  ONE scale changes along x (mm run then cm run) -- the ratio
#               between the two runs is a free cross-validation
#   stacked     two scales on separate rows (handled by band selection)
#   block       alternating filled blocks rather than picket-fence ticks
#   grid        2-D ruled grid
#   none        not a ruler
#
# `units` lists every unit the class guarantees is PRESENT, finest first.
# The ruler class is the SINGLE AUTHORITY on what a crop may contain. Every field
# is load-bearing:
#
#   systems         closed set of measurement systems present. ("std",) means
#                   IMPERIAL ONLY -- metric units are then inadmissible, full stop.
#                   ("metric","std") is the ONLY way both are allowed.
#   units           the units actually PRINTED, finest first. The finest is the
#                   floor; coarser majors up to the system maximum (1 cm / 1 inch)
#                   are admissible because the finest may be unresolvable in a crop.
#   layout          which detector applies (see below).
#   row_period_cm   for layouts where scanning ONE row does not see the unit period:
#                   stagger (two offset rows of 1 cm blocks) and block (alternating
#                   filled/blank cm blocks) both give a 2 cm single-row period.
#   ratio           for `transition`, the known coarse/fine period ratio.
#   skip            set -> no CF is attempted, with the reason recorded.
#
# layouts: nested | transition | stagger | block | grid | stacked | none
CLASS_SPEC: dict[str, dict] = {
    # ---- GT-app minimum_unit names (what groundtruth.db stores) -------------
    "metric__2_MM": dict(systems=("metric",), units=["metric__2_MM"], layout="nested"),
    "metric__MM":   dict(systems=("metric",), units=["metric__MM"],   layout="nested"),
    "metric__4_CM": dict(systems=("metric",), units=["metric__4_CM"], layout="nested"),
    "metric__2_CM": dict(systems=("metric",), units=["metric__2_CM"], layout="nested"),
    "metric__CM":   dict(systems=("metric",), units=["metric__CM"],   layout="nested"),
    "metric__MM_CM": dict(systems=("metric",), units=["metric__MM", "metric__CM"],
                          layout="transition", ratio=10.0),
    "std__32_IN": dict(systems=("std",), units=["std__32_IN"], layout="nested"),
    "std__16_IN": dict(systems=("std",), units=["std__16_IN"], layout="nested"),
    "std__8_IN":  dict(systems=("std",), units=["std__8_IN"],  layout="nested"),
    "std__4_IN":  dict(systems=("std",), units=["std__4_IN"],  layout="nested"),
    "std__2_IN":  dict(systems=("std",), units=["std__2_IN"],  layout="nested"),
    "std__IN":    dict(systems=("std",), units=["std__IN"],    layout="nested"),

    # ---- LM3_Ruler_Classifier labels ---------------------------------------
    "METRIC_MM":   dict(systems=("metric",), units=["metric__MM"],   layout="nested"),
    "METRIC_MM2":  dict(systems=("metric",), units=["metric__2_MM"], layout="nested"),
    "METRIC_CM":   dict(systems=("metric",), units=["metric__CM"],   layout="nested"),
    "METRIC_CM2":  dict(systems=("metric",), units=["metric__2_CM"], layout="nested"),
    "METRIC_CM4":  dict(systems=("metric",), units=["metric__4_CM"], layout="nested"),
    "METRIC_MM_CM": dict(systems=("metric",), units=["metric__MM", "metric__CM"],
                         layout="transition", ratio=10.0),
    "METRIC_MM_CM2": dict(systems=("metric",), units=["metric__MM", "metric__2_CM"],
                          layout="nested"),
    # BLOCK rulers alternate a filled cm block with a blank cm gap, so a single-row
    # scan sees a 2 cm period -- same arithmetic as stagger, one row instead of two.
    "METRIC_CM_BLOCK": dict(systems=("metric",), units=["metric__CM"],
                            layout="block", row_period_cm=2.0),
    "METRIC_MM_CM_BLOCK": dict(systems=("metric",), units=["metric__MM", "metric__CM"],
                               layout="block", row_period_cm=2.0, ratio=10.0),
    "METRIC_CM_STAGGER": dict(systems=("metric",), units=["metric__CM"],
                              layout="stagger", row_period_cm=2.0),
    "GRID_CM":     dict(systems=("metric",), units=["metric__CM"], layout="grid"),
    "STD_IN8":     dict(systems=("std",), units=["std__8_IN"],  layout="nested"),
    "STD_IN16":    dict(systems=("std",), units=["std__16_IN"], layout="nested"),
    "STD_IN32_IN16": dict(systems=("std",), units=["std__32_IN", "std__16_IN"],
                          layout="nested"),
    "STD_IN16_IN8_IN2": dict(systems=("std",),
                             units=["std__16_IN", "std__8_IN", "std__2_IN"],
                             layout="nested"),
    # the only genuinely cross-system classes
    "STD_IN8_CM":  dict(systems=("metric", "std"), units=["std__8_IN", "metric__CM"],
                        layout="stacked"),
    "METRIC_MM_IN16": dict(systems=("metric", "std"),
                           units=["metric__MM", "std__16_IN"], layout="stacked"),

    # ---- non-CF classes ----------------------------------------------------
    # FP is a SPECIAL ruler kind, not a false positive -- deferred, not discarded.
    "FP": dict(systems=(), units=[], layout="none",
               skip="special ruler kind (FP) -- deferred, not yet handled"),
    # messy = unreadable, or genuinely not a ruler. No CF is possible.
    "messy": dict(systems=(), units=[], layout="none",
                  skip="unreadable or not a ruler"),
}

# The coarsest major a ruler is graduated to, per system. Coarser majors than the
# printed minimum are admissible up to this, and no further.
SYSTEM_MAX = {"metric": "metric__CM", "std": "std__IN"}

UNCERTAIN_CLASSES = sorted(k for k, v in CLASS_SPEC.items() if v.get("uncertain"))
SKIP_CLASSES = sorted(k for k, v in CLASS_SPEC.items() if v.get("skip"))


def spec_of(unit_class: str) -> dict:
    """Registry entry for a class. An UNKNOWN class is flagged, never guessed into
    a metric ruler -- silently defaulting is how imperial rulers got read as mm."""
    sp = CLASS_SPEC.get(unit_class)
    if sp:
        return sp
    if unit_class in MINIMUM_UNITS:
        return dict(systems=(system_of(unit_class),), units=[unit_class],
                    layout="nested", unknown=True)
    return dict(systems=(), units=[], layout="none", unknown=True,
                skip=f"unknown ruler class {unit_class!r}")


def class_systems(unit_class: str) -> tuple[str, ...]:
    return tuple(spec_of(unit_class).get("systems") or ())


def is_skipped(unit_class: str):
    """Reason this class yields no CF, or None."""
    return spec_of(unit_class).get("skip")


def admissible_units(unit_class: str) -> list[str]:
    """Every unit the class permits: per system, from the finest PRINTED unit up to
    that system's coarsest major (1 cm / 1 inch).

    This is the whole point of the class. A ("std",) class can never yield a metric
    unit; coarser majors ARE allowed because the printed minimum is often
    unresolvable in a crop and the lattice locks a major instead.
    """
    sp = spec_of(unit_class)
    out: list[str] = []
    for sysu in sp.get("systems") or ():
        declared = [u for u in sp["units"] if system_of(u) == sysu]
        if not declared:
            continue
        lo = min(unit_mm(u) for u in declared)
        hi = unit_mm(SYSTEM_MAX[sysu])
        out += [u for u in real_units(sysu)
                if lo - 1e-9 <= unit_mm(u) <= hi + 1e-9]
    return sorted(set(out), key=unit_mm)


def admissible_periods(unit_class: str) -> dict[str, float]:
    """{unit -> period in cm} for the class. Replaces the global METRIC_P/STD_P
    candidate dicts, which leaked metric units into imperial rulers."""
    return {u: unit_mm(u) / 10.0 for u in admissible_units(unit_class)}


# --------------------------------------------------------------------------- #
# The megapixel-regressed CF anchor (LM3 project DB)
# --------------------------------------------------------------------------- #
CF_ANCHOR_COLUMN = "cf_px_per_cm_predicted_by_mp"


def read_cf_anchors(db_path, table="specimen", key_column=None):
    """Load {key: px_per_cm} from an LM3 project DB, or {} if unavailable.

    Tolerant by design: the column does not exist in the current schema.sql and
    only appears in production, so a missing DB / table / column is a normal
    no-anchor run rather than an error.
    """
    import os
    import sqlite3
    if not db_path or not os.path.exists(db_path):
        return {}
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        if CF_ANCHOR_COLUMN not in cols:
            return {}
        key = key_column or next((c for c in ("filename", "image_name", "basename",
                                              "specimen_id", "id") if c in cols), None)
        if key is None:
            return {}
        return {r[0]: float(r[1]) for r in
                con.execute(f"select {key}, {CF_ANCHOR_COLUMN} from {table} "
                            f"where {CF_ANCHOR_COLUMN} is not null")}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Final CF: every detected unit contributes, bounded by the MP anchor + crop size
# --------------------------------------------------------------------------- #
# The physical length a ruler crop may imply. The ceiling was 40 cm and was rejecting REAL rulers:
# five sheets in the 549-sheet ground-truth run carry a 41.8-43.3 cm ruler, and the reading thrown
# out for exceeding 40 cm was in each case the best-evidenced one on the sheet (296-418 ticks) and
# within 0.2-3.4% of the human measurement. 60 cm clears every ruler observed while still excluding
# the 2x-and-up harmonic errors this bound exists to catch. The floor drops to 0.5 cm so a small
# scale bar or a partial crop is measurable rather than silently discarded.
PLAUSIBLE_LEN_CM = (0.5, 60.0)
ANCHOR_TOL = 0.25                  # a reading this far from the MP CF is a wrong harmonic


def min_rung_ratio() -> float:
    """Smallest ratio between any two DISTINCT named unit sizes.

    Every way this engine can get a CF wrong by misreading the unit is a jump
    between two rungs of MINIMUM_UNITS, so the smallest gap between adjacent rungs
    is the smallest error a naming mistake can possibly produce. Over the current
    table that is metric__MM (1.000 mm) against std__32_IN (0.79375 mm) = 1.2598,
    i.e. +26.0% / -20.6%. Derived from the table rather than written down, so it
    stays correct if a unit is ever added or removed.
    """
    sizes = sorted({round(unit_mm(u), 9) for u in real_units()})
    return min(b / a for a, b in zip(sizes, sizes[1:]))


# Half the minimum rung, in the LOG metric the unit ladder actually lives in: the
# decision boundary between "this reading is the anchor's value" and "this reading
# is one rung off". A tolerance WIDER than this cannot do the job ANCHOR_TOL
# documents -- at 0.25 the window [0.75, 1.25] contains the one-rung-down
# alternative at 0.794 outright, so a reading nearer a wrong rung than the anchor
# was still being called anchor-supported.
RUNG_HALF_LOG = math.log(min_rung_ratio()) / 2.0     # 0.1155 -> -10.9% / +12.2%


def combine_cf(estimates, anchor=None, width_px=None,
               anchor_tol=ANCHOR_TOL, len_bounds=PLAUSIBLE_LEN_CM):
    """Fuse every independently measured unit into one conversion factor.

    `estimates` : [{unit, est_pxcm, n_ticks, source}] -- the finest unit AND every
    coarser major that was detected, each measured from its OWN tick positions so
    each is an independent estimate of the same physical quantity.

    Two bounds decide which are admitted, exactly as specified:
      * the MP-predicted CF (working frame) -- a reading more than `anchor_tol`
        away is a wrong harmonic, not a noisy measurement;
      * the crop's pixel length -- the implied physical ruler length must be
        physically possible.

    Returns the fused CF plus full provenance, so the DB records WHY.
    """
    rows = []
    for e in estimates:
        v = e.get("est_pxcm")
        if not v or not math.isfinite(v) or v <= 0:
            continue
        L = (width_px / v) if width_px else None
        ok_len = True if L is None else (len_bounds[0] <= L <= len_bounds[1])
        d_anchor = (v / anchor - 1.0) if anchor else None
        ok_anchor = True if d_anchor is None else (abs(d_anchor) <= anchor_tol)
        # A human-readable reason, built here where both bounds and both measured values are in
        # scope. The QC used to print "REJECTED (out of bounds)" for every failure, which conflated
        # a wrong-harmonic anchor miss with a physically impossible ruler length -- two different
        # problems needing two different fixes.
        why = []
        if not ok_len and L is not None:
            why.append(f"implies a {L:.1f} cm ruler, outside {len_bounds[0]:g}-{len_bounds[1]:g} cm")
        if not ok_anchor and d_anchor is not None:
            why.append(f"{100 * d_anchor:+.0f}% vs the MP anchor, outside +/-{100 * anchor_tol:.0f}%")
        rows.append(dict(unit=e.get("unit"), source=e.get("source", "level"),
                         est_pxcm=float(v), n_ticks=int(e.get("n_ticks") or 1),
                         implied_len_cm=L, pct_vs_anchor=(None if d_anchor is None
                                                          else 100.0 * d_anchor),
                         ok_len=bool(ok_len), ok_anchor=bool(ok_anchor),
                         reject_reason=("; ".join(why) or None),
                         used=bool(ok_len and ok_anchor)))
    if not rows:
        return None

    used = [r for r in rows if r["used"]]
    if not used:
        # Nothing cleared the bounds. Previously this fell back to fusing the
        # rejected readings anyway, which threw away the check's own verdict and
        # reported a confident CF built entirely from values it had just rejected.
        # Emit NO CF instead, with the provenance kept so the reason is auditable.
        return dict(cf_px_per_cm=None, ok=False, n_used=0, n_rejected=len(rows),
                    fallback=True, spread=None, agree=False, implied_len_cm=None,
                    pct_vs_anchor=None, contributions=rows,
                    reason="every reading fell outside the MP-anchor / crop-length bounds")

    w = [max(1, r["n_ticks"]) for r in used]
    cf = sum(r["est_pxcm"] * wi for r, wi in zip(used, w)) / sum(w)
    vals = [r["est_pxcm"] for r in used]
    spread = ((max(vals) - min(vals)) / (sum(vals) / len(vals))) if len(vals) > 1 else None

    return dict(cf_px_per_cm=cf, ok=True,
                n_used=len(used), n_rejected=len(rows) - len(used),
                fallback=False, reason=None,
                spread=spread,
                agree=(spread is not None and spread <= AGREE_TOL),
                implied_len_cm=(width_px / cf) if width_px else None,
                pct_vs_anchor=(None if not anchor else 100.0 * (cf / anchor - 1.0)),
                contributions=rows)
