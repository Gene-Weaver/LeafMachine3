"""v2 pipeline driver.

Order of operations, and why:

  0. UNSHARP    PIL UnsharpMask(radius=2, percent=250, threshold=0) -- the
                variant already evaluated in the GT app's edge_enhance test.
                Applied before anything else so every downstream stage sees
                crisper tick edges.
  1. ORIENT     rotate (not shear) so ticks are vertical, black padding + a
                validity mask so padding never enters a projection.
  2. BANDS      candidate bands = fixed splits UNION adaptive runs; the band is
                chosen by projection periodicity, which measurably picks the
                scale-bearing row better than either source alone.
  3. SCALE      decide what the ACF lattice physically IS from the band's ACF
                period. That is the most reliably detected quantity in the
                pipeline, so the scale hangs off it rather than off the descended
                lattice (which is easy to over-descend).
  4. LATTICE    descend to the finest lattice that is genuinely occupied, gated
                by tick yield and an alternation test. Drives ticks and masks.
  5. LEVELS     measure every tick's length; find which residue classes are
                systematically longer. Those are the coarser units printed in the
                SAME row -- 1/16, 1/8, 1/4, 1/2 and 1 inch on one strip.
  6. CROSSVAL   name each level from its own MEASURED spacing and check the
                independent px/cm estimates agree (crossval.group_agreement).
"""
from __future__ import annotations

import math
import numpy as np
import cv2
from PIL import Image, ImageFilter

from .lattice import (orient, candidate_bands, adaptive_bands, band_profiles,
                        periodicity, refine_period, extract_ticks, find_levels,
                        level_of, text_penalty, bandpass, alternation,
                        detect_transition, subharmonic_lags, comb_lag, MIN_PERIOD,
                        MAX_LEVELS)
from .units import (identify_period, label_levels, reconcile_units, combine_cf,
                      class_units, unit_mm, system_of, spec_of, class_systems,
                      admissible_units, admissible_periods, is_skipped, ANCHOR_TOL,
                      METRIC_P, STD_P, MINIMUM_UNITS)

DESCEND_DIVS = (2, 3, 4, 5, 8, 10, 16)

# ---- band reconciliation ---------------------------------------------------
# Bands are no longer SELECTED, they are RECONCILED. A crop offers ~10 candidate
# bands and each yields an independent estimate of one physical quantity, exactly
# like the unit levels within a band and the sibling crops within a sheet.
# Measured on the 69 GT crops: picking the most-periodic band gives 53/69 within
# 3% (median 0.65%); clustering the estimates gives 56/69 (median 0.47%).
# Ranking by extraction quality instead was much WORSE (keep_ratio alone: 42/69),
# which is why periodicity survives as a weight rather than being replaced.
BAND_TOPK = 5              # full evaluation is ~10x a periodicity probe, so cap it
BAND_MIN_PERIODICITY = 0.45
BAND_PEER_TOL = 0.03       # bands within 3% are measuring the same thing
# A cluster pair at one of these ratios is a genuine two-system stacked ruler.
CROSS_SYSTEM_RATIOS = (2.54, 1.27, 5.08)
CROSS_SYSTEM_TOL = 0.05
# A minority cluster at a clean rung of the ladder is a MISREAD, not a second
# scale. Folding these in was measured and gave no gain, so they are recorded as
# a warning only -- a strong 2x shadow is the DBG failure signature.
HARMONIC_SHADOW = (2.0, 2.5, 4.0, 5.0, 8.0, 10.0)
HARMONIC_TOL = 0.05
UNSHARP = dict(radius=2, percent=250, threshold=0)


def unsharp(gray):
    """UnsharpMask, same parameters as LM3_Ruler_Distance_Groundtruth's
    edge_unsharp test panel."""
    im = Image.fromarray(np.asarray(gray, np.uint8))
    return np.asarray(im.filter(ImageFilter.UnsharpMask(**UNSHARP)), np.uint8)


def _sublattice_support(i_bp, P, unit_class):
    """For each candidate identity of P, would the class's finest unit still be
    resolvable -- and if so, is that sub-lattice actually present?"""
    declared = class_units(unit_class)
    # The divisor must be the finest unit printed on the CANDIDATE'S OWN SCALE.
    # A single global `min(declared)` is only correct for a single-system class;
    # on a cross-system ruler it asks whether the OTHER face's graduation divides
    # this face's period, which is a physically meaningless question and is never
    # a whole number, so one whole system was structurally pinned at zero support.
    # On METRIC_MM_IN16 (declared 1 mm + 1/16 in) every imperial candidate was
    # tested against 1 mm -- 1/16 in / 1 mm = 1.5875, 1 in / 1 mm = 25.4 -- so no
    # imperial candidate could ever earn support. Mirror case on STD_IN8_CM.
    finest_by_sys: dict[str, str] = {}
    for u in declared:
        s_u = system_of(u)
        if s_u not in finest_by_sys or unit_mm(u) < unit_mm(finest_by_sys[s_u]):
            finest_by_sys[s_u] = u
    cands = admissible_periods(unit_class)
    _, _, sup0 = refine_period(i_bp, P)
    out = {}
    for name, pcm in cands.items():
        # `name` is a unit key for every admissible candidate, so its system is
        # known; fall back to the global finest for any pseudo-candidate.
        u_fine = finest_by_sys.get(system_of(name)) if name in MINIMUM_UNITS else None
        if u_fine is None:
            u_fine = min(declared, key=unit_mm)
        umm = unit_mm(u_fine)
        ratio = pcm * 10.0 / umm
        d = int(round(ratio))
        v = 0.0
        # Only meaningful when the class minimum divides the candidate a WHOLE
        # number of times. round(2.5)==2 previously tested a 1.25 mm lattice and
        # credited it as evidence for 1 mm ticks.
        if d >= 2 and abs(ratio - d) < 0.02 and P / d >= MIN_PERIOD:
            _, _, sq = refine_period(i_bp, P / d, frac=0.02, steps=41)
            v = min(max(sq / sup0 if sup0 > 0 else 0.0, 0.0), 1.0)
        out[name] = v
    return out


def _choose_lattice(rot, valid, band, pr, P_acf, txt):
    """Finest lattice that is genuinely occupied."""
    divs = [1] + [d for d in DESCEND_DIVS if P_acf / d >= MIN_PERIOD]
    chosen = None
    for d in divs:                                  # ascending d = finer
        q = P_acf / d
        qq, qph, sup = refine_period(pr["i_bp"], q,
                                     frac=0.06 if d == 1 else 0.03,
                                     steps=121 if d == 1 else 61)
        if sup <= 0:
            continue
        tk, n_cells, pk = extract_ticks(rot, valid, band, qq, qph,
                                        pr["inten_off"], pr["pol"], txt)
        if len(tk) < 6 or n_cells == 0:
            continue
        if alternation(pk) is not None:             # this lattice is too fine
            continue
        occ = len(tk) / n_cells
        n_keep = sum(1 for t in tk if not t["reject"])
        keep = n_keep / len(tk)
        if occ >= 0.80 and n_keep >= 6 and keep >= 0.45:
            chosen = (qq, qph, d, tk, keep, occ, False)
    if chosen is not None:
        return chosen
    qq, qph, _ = refine_period(pr["i_bp"], P_acf)
    tk, n_cells, pk = extract_ticks(rot, valid, band, qq, qph,
                                    pr["inten_off"], pr["pol"], txt)
    if len(tk) < 3 or n_cells == 0:
        return None
    n_keep = sum(1 for t in tk if not t["reject"])
    return (qq, qph, 1, tk, n_keep / len(tk), len(tk) / n_cells, True)


def _identify_best(pr, P_cands, unit_class, W, cf_anchor, extra, comb_P=None):
    """Pick (period, unit) jointly over the ACF argmax AND its sub-harmonics.

    A comb autocorrelates at every multiple of its period, so the ACF argmax is
    biased toward a MULTIPLE of the true tick spacing (ruler_lattice.subharmonic_lags
    explains the physics). Naming that multiple afterwards cannot recover: 3 mm is
    not a printed unit, so identify_period rounds it to 2.5 mm or 5 mm and the
    error is permanent.

    The repair is not to force the smallest lag -- the smallest supported lag is
    sometimes a sub-division the CLASS cannot name (a STD_IN8 row that also
    carries 1/16 ticks). It is to hand every ACF-supported period to the naming
    step and let the class decide, on the evidence identify_period already uses:
    implied ruler length, the MP CF anchor, and sub-lattice presence.

    One term of that score must NOT be handed to the sub-harmonics: the class's
    minimum-unit prior. It says "the lattice the profile locked is most likely the
    class's finest printed unit", and it is a statement about the LOCK -- the ACF
    argmax, the one period the image evidence actually singled out. A sub-harmonic
    is not an independent lock: a comb autocorrelates at every multiple of its
    period, so a sub-harmonic exists mechanically whenever the argmax does, and it
    is always the finer of the pair. Granting each of them the same +3 rewards
    nothing but smallness and degenerates into "always take the smallest lag" --
    exactly the over-descent this search must not introduce. Measured: left in, it
    outweighs a 1.3-sigma anchor violation and returns a px/cm 60% off on a crop
    whose anchor was dead-on. So sub-harmonics are scored with
    `min_unit_bonus=0`; the argmax keeps the prior and therefore keeps its
    pre-existing behaviour exactly when it is the only candidate.

    Ties go to the COARSER period (P_cands is ordered coarsest first), which is
    also the pre-existing behaviour.
    """
    scored = []
    for i, P in enumerate(P_cands):
        sup = _sublattice_support(pr["i_bp"], P, unit_class)
        kw = {} if i == 0 else dict(min_unit_bonus=0.0)
        ident = identify_period(P, unit_class, W, sublattice_support=sup,
                                extra_candidates=extra, cf_anchor=cf_anchor, **kw)
        if ident is not None:
            scored.append((P, ident))
    if not scored:
        return None
    # The MP CF anchor is already the pipeline's stated test for a wrong harmonic
    # (ruler_units.ANCHOR_TOL: "a reading this far from the MP CF is a wrong
    # harmonic, not a noisy measurement"), and combine_cf applies it to the final
    # readings. Apply the same rule HERE, where the harmonic is actually chosen:
    # a candidate period whose reading is off the anchor by more than ANCHOR_TOL
    # is not in the running. Never empties the list -- if nothing clears the
    # anchor the anchor is useless for this crop and every candidate stays.
    if cf_anchor and cf_anchor > 0:
        near = [t for t in scored
                if abs(t[1]["pxcm"] / cf_anchor - 1.0) <= ANCHOR_TOL]
        if near:
            scored = near

    # The comb candidate is offered to the SCORE, not forced ahead of it. Forcing it was measured
    # first -- "comb first, current as fallback" -- and is worse in the real pipeline than here in
    # the isolated harness that motivated it: sub-5% crop readings 858 -> 817, catastrophic >25%
    # readings 15 -> 17, and 40 crops lost their reading entirely. The harness compared the comb
    # against a weaker baseline and named periods by anchor-proximity alone, while identify_period
    # also weighs sub-lattice support and the class's minimum-unit prior. Overriding that evidence
    # with a single spacing estimate throws away more than the comb recovers. Scored, the same
    # candidate is a clear gain: sheets publishing 523 -> 540 and one fewer catastrophic reading.
    best = None
    for P, ident in scored:
        if best is None or ident["score"] > best[1]["score"] + 1e-9:
            best = (P, ident)
    return best


def _eval_band(rot, valid, txt, band, kind, pr, P_cands, s_per, unit_class, W,
               cf_anchor, extra, min_ticks=3, comb_P=None):
    """Fully evaluate ONE candidate band -> an independent px/cm estimate."""
    got = _identify_best(pr, P_cands, unit_class, W, cf_anchor, extra, comb_P=comb_P)
    if got is None:
        return None
    P_acf, ident = got
    ch = _choose_lattice(rot, valid, band, pr, P_acf, txt)
    if ch is None:
        return None
    P, phase, div, ticks, keep_ratio, occupancy, fallback = ch
    kept = [t for t in ticks if not t["reject"]]
    if len(kept) < min_ticks:
        return None
    return dict(band=band, kind=kind, pr=pr, periodicity=s_per, P_acf=P_acf,
                ident=ident, pxcm=ident["pxcm"], P=P, phase=phase, div=div,
                ticks=ticks, n_kept=len(kept), keep_ratio=keep_ratio,
                occupancy=occupancy, fallback=fallback,
                weight=max(1e-6, s_per * keep_ratio))


def _cluster_bands(recs, tol=BAND_PEER_TOL):
    """Group band estimates that agree, heaviest cluster first."""
    cl = []
    for r in sorted(recs, key=lambda r: -r["weight"]):
        for g in cl:
            if abs(r["pxcm"] / g[0]["pxcm"] - 1.0) <= tol:
                g.append(r); break
        else:
            cl.append([r])
    cl.sort(key=lambda g: (sum(x["weight"] for x in g), len(g)), reverse=True)
    return cl


def _cluster_cf(g):
    w = [x["weight"] for x in g]
    return sum(x["pxcm"] * wi for x, wi in zip(g, w)) / sum(w)


def _match_ratio(a, b, pool, tol):
    r = max(a, b) / min(a, b)
    h = min(pool, key=lambda x: abs(math.log(r / x)))
    return h if abs(r / h - 1.0) <= tol else None


def analyse(gray, unit_class, cf_anchor=None):
    # The class can rule a crop out before any pixels are touched: FP is a special
    # ruler kind we defer, `messy` is unreadable / not a ruler, and an unknown class
    # is not guessed at.
    skip = is_skipped(unit_class)
    if skip:
        return dict(skipped=True, skip_reason=skip, unit_class=unit_class)
    sharp = unsharp(gray)
    o = orient(sharp)
    rot, valid = o["rot"], o["valid"]
    txt = text_penalty(rot)
    W = rot.shape[1]

    # ---- 2/3. evaluate candidate bands, then RECONCILE them ----------------
    spec0 = spec_of(unit_class)
    extra = ({f"{spec0['row_period_cm']:g} cm ({spec0['layout']} row)": spec0["row_period_cm"]}
             if spec0.get("row_period_cm") else None)

    prelim = []
    for (y0, y1, kind) in candidate_bands(rot, valid):
        pr = band_profiles(rot, valid, (y0, y1))
        if pr is None:
            continue
        sp, p = periodicity(pr["resp"])
        if not np.isfinite(p):
            continue
        rbp = bandpass(pr["resp"])
        # the ACF argmax AND every sub-harmonic the ACF supports, each comb-refined
        P_cands = []
        for lg in subharmonic_lags(pr["resp"]):
            Pc, _, _ = refine_period(rbp, lg)
            if np.isfinite(Pc) and Pc >= MIN_PERIOD:
                P_cands.append(Pc)
        # The comb candidate, APPENDED so the argmax stays at index 0 and keeps its minimum-unit
        # prior (see _identify_best) -- this only ever adds an option, it never displaces one, so a
        # crop the engine already read correctly is scored exactly as before. It is deliberately
        # NOT filtered by MIN_PERIOD: a period recovered from peak SPACING is legitimate below the
        # search floor, which is the entire class of failure it exists to reach.
        cl = comb_lag(pr["resp"])
        comb_P = None
        if cl is not None and not any(abs(cl / Pc - 1.0) <= 0.02 for Pc in P_cands):
            comb_P = float(cl)
            P_cands.append(comb_P)
        if not P_cands:
            continue
        prelim.append((sp, (y0, y1), kind, pr, P_cands, comb_P))
    if not prelim:
        return None
    prelim.sort(key=lambda t: -t[0])
    short = [t for t in prelim if t[0] >= BAND_MIN_PERIODICITY][:BAND_TOPK] or prelim[:1]

    recs = []
    for (sp, band_i, kind_i, pr_i, Pc_i, comb_i) in short:
        r = _eval_band(rot, valid, txt, band_i, kind_i, pr_i, Pc_i, sp,
                       unit_class, W, cf_anchor, extra, comb_P=comb_i)
        if r:
            recs.append(r)
    if not recs:
        # `_eval_band` requires >= 3 accepted ticks, which the old single-band
        # path never did (it leaned on _choose_lattice's unconditional fallback).
        # Crops where the shape gate rejects EVERY tick used to survive via the
        # mask-salvage step further down in analyse(); _eval_band was bailing
        # before salvage could run. min_ticks=0 hands the band back so salvage
        # still happens, exactly as before the band-reconciliation change.
        for (sp, band_i, kind_i, pr_i, Pc_i, comb_i) in short:
            r = _eval_band(rot, valid, txt, band_i, kind_i, pr_i, Pc_i, sp,
                           unit_class, W, cf_anchor, extra, min_ticks=0, comb_P=comb_i)
            if r:
                r["thin"] = True
                recs.append(r)
                break
    if not recs:
        return None

    clusters = _cluster_bands(recs)
    win = clusters[0]
    band_cf = _cluster_cf(win)

    # STAGE 2 -- a stacked class genuinely carries two systems, so a cluster pair
    # at a cross-system ratio identifies them. Triggered by the CLASS, never by
    # cluster count: 84% of crops produce multiple clusters and only 6/69 show a
    # real 2.54 signature, so cluster count carries no information.
    stacked_pair = None
    if spec0.get("layout") == "stacked" and len(clusters) > 1:
        for g in clusters[1:]:
            other = _cluster_cf(g)
            h = _match_ratio(band_cf, other, CROSS_SYSTEM_RATIOS, CROSS_SYSTEM_TOL)
            if h:
                folded = other / h if other > band_cf else other * h
                w0 = sum(x["weight"] for x in win)
                w1 = sum(x["weight"] for x in g)
                stacked_pair = dict(ratio=h, other_cf=other, folded=folded,
                                    n_bands=len(g))
                band_cf = (band_cf * w0 + folded * w1) / (w0 + w1)
                break

    # STAGE 3 -- a minority cluster at a clean rung is a MISREAD, not a second
    # scale. Folding them in was measured and gave no gain, so record only.
    shadows = []
    for g in clusters[1:]:
        other = _cluster_cf(g)
        h = _match_ratio(band_cf, other, HARMONIC_SHADOW, HARMONIC_TOL)
        if h:
            shadows.append(dict(ratio=round(h, 3), cf=round(other, 3),
                                n_bands=len(g),
                                weight=round(sum(x["weight"] for x in g), 4)))

    # the heaviest band IN the winning cluster drives ticks / masks / levels
    rep = max(win, key=lambda r: r["weight"])
    band, kind, pr, P_acf, ident = rep["band"], rep["kind"], rep["pr"], rep["P_acf"], rep["ident"]
    bscore = rep["periodicity"]
    pxcm = band_cf
    vals = [r["pxcm"] for r in win]
    band_spread = ((max(vals) - min(vals)) / (sum(vals) / len(vals))) if len(vals) > 1 else None

    # ---- 4. finest occupied lattice, for ticks + masks ---------------------
    P, phase, div = rep["P"], rep["phase"], rep["div"]
    ticks, keep_ratio = rep["ticks"], rep["keep_ratio"]
    occupancy, fallback = rep["occupancy"], rep["fallback"]
    kept = [t for t in ticks if not t["reject"]]

    # Never emit an empty mask. If the shape gate rejected everything (block
    # rulers and very low-contrast crops can trip every rule at once), reinstate
    # the most rectangle-like half and flag the crop, because a flagged
    # imperfect mask is more useful downstream than no mask at all.
    if not kept and ticks:
        order = sorted(ticks, key=lambda t: -t["rect"])
        for t in order[:max(3, len(order) // 2)]:
            t["reject"] = False
        kept = [t for t in ticks if not t["reject"]]
        salvaged = True
    else:
        salvaged = False
    keep_ratio = len(kept) / max(1, len(ticks))

    # ---- 5/6. levels + full reconciliation ---------------------------------
    ladder = find_levels(ticks)
    acc = sorted([t for t in ticks if not t["reject"]], key=lambda t: t["xc"])

    # one entry per tick level, with its MEASURED spacing, its tick count, and
    # the x of its first accepted tick (the anchor the predicted comb hangs off)
    # anchor on the first accepted tick that is actually ON-FRAME: the leftmost
    # lattice cell can straddle x=0 and yield a negative centre, which is both a
    # poor anchor and (previously) an empty slice when drawing
    def _anchor(cands):
        pos = [t["xc"] for t in cands if t["xc"] >= 0]
        if pos:
            return pos[0]
        return max(0.0, cands[0]["xc"]) if cands else None

    lv_in = [dict(mult=1, spacing=P, n=len(acc), anchor=_anchor(acc))]
    for (m, r, _g, sp) in ladder:
        mem = [t for t in acc if t["k"] % m == r]
        lv_in.append(dict(mult=m, spacing=sp, n=len(mem),
                          anchor=(_anchor(mem) if mem else _anchor(acc))))

    # ---- transition classes: the class GUARANTEES a fine run and a coarse run
    # with a known ratio, so measure both and use the ratio as a free check.
    spec = spec_of(unit_class)
    trans = None
    if spec["layout"] in ("transition", "block"):
        trans = detect_transition(pr["i_bp"], expect_ratio=spec.get("ratio", 10.0))
        if trans:
            # the fine run IS the class's finest declared unit -- an independent
            # scale estimate that does not depend on identify_period at all
            u_fine = spec["units"][0]
            trans["pxcm_fine"] = trans["P_fine"] / unit_mm(u_fine) * 10.0
            trans["unit_fine"] = u_fine
            if len(spec["units"]) > 1:
                u_coarse = spec["units"][-1]
                trans["pxcm_coarse"] = trans["P_coarse"] / unit_mm(u_coarse) * 10.0
                trans["unit_coarse"] = u_coarse
                trans["pxcm_agree"] = abs(trans["pxcm_coarse"] /
                                          trans["pxcm_fine"] - 1.0) <= 0.03

    rec = reconcile_units(pxcm, lv_in, set(class_units(unit_class)),
                          allowed_units=admissible_units(unit_class))
    # The SCALE otherwise stays on the comb-refined P_acf, which fits every tick coherently
    # and is measurably the most precise estimator available (within-1% 64% vs 52%
    # for any average over level spacings). Reconciliation instead supplies the
    # unit identities and an INDEPENDENT cross-check: each level's spacing is
    # measured from its own tick positions, so its agreement with the comb fit is
    # real evidence rather than an algebraic restatement.
    # A validated transition beats the global comb fit: its two periods are
    # measured on disjoint x ranges and their ratio is known a priori, so passing
    # the ratio check confirms both. Measured over the 69 GT crops this moves the
    # median error 0.72% -> 0.50%.
    pxcm_comb = pxcm
    trans_used = False
    if trans and trans.get("pxcm_fine") and trans["ratio_err"] <= 0.03 \
            and trans.get("pxcm_agree", True) and not spec.get("weak_ratio"):
        cands = [trans["pxcm_fine"]] + ([trans["pxcm_coarse"]]
                                        if trans.get("pxcm_coarse") else [])
        pxcm = float(np.mean(cands))
        trans_used = True

    pxcm_rec = rec["pxcm"] if (rec and rec.get("ok")) else None
    rec_conflict = bool(pxcm_rec and abs(pxcm_rec / pxcm - 1.0) > 0.03)
    if rec and rec.get("ok"):
        # attach an anchor to each reconciled unit group, from its finest member
        by_mult = {lv["mult"]: lv for lv in lv_in}
        for g in rec["groups"]:
            a = [by_mult[m]["anchor"] for m in g["mults"]
                 if by_mult.get(m) and by_mult[m]["anchor"] is not None]
            g["anchor"] = a[0] if a else (_anchor(acc) or 0.0)
    named, ests, spread, agree = label_levels(pxcm, P, [(l["mult"], l["spacing"])
                                                        for l in lv_in[1:]],
                                              ident["system"])

    # ---- final CF: the finest unit AND every coarser major, each measured
    # independently, fused under the MP-anchor and crop-length bounds ----------
    contribs = []
    if rec and rec.get("ok"):
        for g in rec["groups"]:
            contribs.append(dict(unit=g["unit"], est_pxcm=g["est_pxcm"],
                                 n_ticks=g["n_ticks"], source="level"))
    if trans and trans.get("pxcm_fine"):
        contribs.append(dict(unit=trans.get("unit_fine"), est_pxcm=trans["pxcm_fine"],
                             n_ticks=len(acc), source="transition-fine"))
        if trans.get("pxcm_coarse"):
            contribs.append(dict(unit=trans.get("unit_coarse"),
                                 est_pxcm=trans["pxcm_coarse"],
                                 n_ticks=max(1, len(acc) // 10),
                                 source="transition-coarse"))
    if not contribs:
        contribs.append(dict(unit=ident["period_name"], est_pxcm=pxcm,
                             n_ticks=len(acc), source="acf-lattice"))
    fused = combine_cf(contribs, anchor=cf_anchor, width_px=W)
    if fused and fused.get("ok"):
        pxcm = fused["cf_px_per_cm"]

    return dict(orient=o, rot=rot, valid=valid, txt=txt, sharp=sharp,
                band=band, band_kind=kind, band_score=bscore,
                P_acf=P_acf, P=P, phase=phase, div=div, fallback=fallback,
                ticks=ticks, n_ticks=len(ticks), kept=len(kept),
                keep_ratio=keep_ratio, occupancy=occupancy, salvaged=salvaged,
                ladder=ladder, levels=named, estimates=ests,
                spread=spread, agree=agree, rec=rec, lv_in=lv_in,
                pxcm_rec=pxcm_rec, rec_conflict=rec_conflict,
                spec=spec, trans=trans, trans_used=trans_used,
                bands_thin=any(r.get("thin") for r in recs),
                bands_evaluated=len(recs), n_band_clusters=len(clusters),
                n_bands_agree=len(win), band_spread=band_spread,
                band_cf=band_cf, stacked_pair=stacked_pair, shadows=shadows,
                fused=fused, admissible=admissible_units(unit_class),
                skipped=False,
                pxcm_comb=pxcm_comb,
                first_tick_x=(_anchor(acc) or 0.0),
                pxcm=pxcm, period_name=ident["period_name"],
                implied_len_cm=ident["implied_len_cm"], system=ident["system"],
                pol=pr["pol"], x_off=pr["inten_off"])


def build_masks(res):
    """Per-level tick label image over the whole rotated crop: 0 = background,
    1 = finest unit, 2.. = successively coarser units."""
    lab = np.zeros(res["rot"].shape, np.uint8)
    ladder = res["ladder"]
    for t in res["ticks"]:
        if t["reject"]:
            continue
        lv = level_of(t, ladder) + 1
        sub = lab[t["y0"]:t["y1"], t["a"]:t["b"]]
        m = t["mask"] > 0
        sub[m] = np.maximum(sub[m], lv)
    return lab


def summarise(res):
    return dict(
        angle=res["orient"]["angle"], tick_coh=res["orient"]["tick_coh"],
        body_tilt=res["orient"]["body_tilt"],
        band=res["band"], band_kind=res["band_kind"], band_score=res["band_score"],
        P_acf=res["P_acf"], P=res["P"], div=res["div"], fallback=res["fallback"],
        n_ticks=res["n_ticks"], kept=res["kept"],
        keep_ratio=res["keep_ratio"], occupancy=res["occupancy"],
        salvaged=res["salvaged"],
        ladder=[m for m, _r, _g, _sp in res["ladder"]],
        level_units=[(d["mult"], d["unit"], round(d["spacing"], 2))
                     for d in res["levels"]],
        estimates=[round(e, 2) for e in res["estimates"]],
        spread=res["spread"], agree=res["agree"],
        relation=(res["rec"]["relation"] if res["rec"] else "unresolved"),
        systems=(res["rec"]["systems"] if res["rec"] else []),
        rec_groups=([(g["unit"], g["mults"], round(g["est_pxcm"], 3),
                      g["averaged"], round(g["spread_within"], 4), g["n_ticks"])
                     for g in res["rec"]["groups"]] if res["rec"] and res["rec"]["ok"] else []),
        n_averaged=(res["rec"]["n_same_unit_averaged"] if res["rec"] else 0),
        n_corroborating=(res["rec"]["n_corroborating"] if res["rec"] else 0),
        rec_spread=(res["rec"]["spread"] if res["rec"] else None),
        rec_agree=(res["rec"]["agree"] if res["rec"] else False),
        pxcm_rec=res["pxcm_rec"], rec_conflict=res["rec_conflict"],
        layout=res["spec"]["layout"],
        class_units_declared=res["spec"]["units"],
        class_uncertain=res["spec"].get("uncertain"),
        trans=res["trans"], trans_used=res["trans_used"],
        admissible=res["admissible"],
        bands_evaluated=res["bands_evaluated"],
        bands_thin=res["bands_thin"],
        n_band_clusters=res["n_band_clusters"],
        n_bands_agree=res["n_bands_agree"],
        band_spread=res["band_spread"],
        band_cf=res["band_cf"],
        stacked_pair=res["stacked_pair"],
        shadows=res["shadows"],
        fused=res["fused"],
        # 7-tuple: (unit, est_pxcm, n_ticks, source, used, pct_vs_anchor, reject_reason).
        # The 7th element was appended, not inserted, so records written before it existed still
        # unpack correctly everywhere they are read.
        cf_contributions=([(c["unit"], round(c["est_pxcm"], 2), c["n_ticks"],
                            c["source"], c["used"],
                            (None if c["pct_vs_anchor"] is None
                             else round(c["pct_vs_anchor"], 2)),
                            c.get("reject_reason"))
                           for c in res["fused"]["contributions"]]
                          if res["fused"] else []),
        cf_n_used=(res["fused"]["n_used"] if res["fused"] else 0),
        cf_n_rejected=(res["fused"]["n_rejected"] if res["fused"] else 0),
        cf_spread=(res["fused"]["spread"] if res["fused"] else None),
        cf_fallback=(res["fused"]["fallback"] if res["fused"] else False),
        pxcm_comb=res["pxcm_comb"], first_tick_x=res["first_tick_x"],
        pxcm=res["pxcm"], period_name=res["period_name"],
        cf_anchor=res.get("cf_anchor"),
        implied_len_cm=res["implied_len_cm"],
    )
