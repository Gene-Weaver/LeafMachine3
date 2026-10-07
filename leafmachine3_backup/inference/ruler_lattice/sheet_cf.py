"""One conversion factor per PARENT IMAGE.

A sheet usually yields a single ruler crop, but some yield several -- the DBG
example has two, the cm face and the inch face of the same physical ruler. By
definition every ruler crop from one parent image measures the SAME sheet, so
they must all reconcile to ONE final CF. That makes the set of crops an extra,
free reconciliation stage on top of the per-crop unit ladder:

  * crops that agree corroborate each other;
  * a crop that sits at a clean 2x / 5x / 10x from the others has locked a wrong
    harmonic and can be rejected by its peers;
  * a crop whose class is skipped (FP / messy) contributes nothing but does not
    invalidate the rest.

IMPORTANT and measured: agreement between crops is NECESSARY BUT NOT SUFFICIENT.
In the DBG case both crops were wrong by the same factor of 2 and agreed with each
other to 0.7%, because they are two faces of one ruler read through the same
mis-scored lattice. Only the MP anchor caught it. So peer agreement is reported and
used to reject outliers, but the anchor remains the absolute reference.

All crops of a parent share its working frame, so the CFs are directly comparable
with no work_scale juggling.
"""
from __future__ import annotations

import math

from .units import RUNG_HALF_LOG

HARMONICS = (0.1, 0.125, 0.2, 0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 8.0, 10.0)
PEER_TOL = 0.03          # crops within 3% are measuring the same thing
HARMONIC_TOL = 0.05      # how close to a clean ratio counts as "a wrong harmonic"
# An anchor implying a sheet narrower than this is physically impossible for a
# herbarium plate, so it is dropped rather than trusted. COLO_2573432348 is the
# case: anchor 75.38 over a 1400 px working frame implies an 18.6 cm sheet, on a
# 30x45 cm plate. 12 of 140 parents in the reference run fail this test; the
# threshold is stable anywhere in 19-29 cm.
MIN_FRAME_CM = 20.0
# How much RULER a reading needs behind it before it may outvote a disagreeing MP anchor.
# Measured on 853 crops of 549 human-measured sheets: a reading implying a ruler shorter than this
# is wrong by 24.6% on average and exceeds 25% error a quarter of the time, while one implying 9 cm
# or more averages 1.28% and exceeds 25% once in a hundred. The mechanism is physical, not
# statistical -- a short ruler carries few graduations and no redundancy to average an error out.
# The threshold sits on a plateau: every cut from 7 to 10 cm isolates a 14-31% mean-error group,
# and only at 11 does the test start discarding good readings instead.
MIN_TRUSTED_RULER_CM = 9.0
# Anchor DISTANCE may only rank a cluster that carries real evidence. Without this
# floor a 1-2 tick cluster that happens to sit near the anchor outranks a
# well-supported one (it regressed EKY 74.381 -> 69.716). Stable over 4-10.
DISTANCE_WEIGHT_FLOOR = 6
# The same floor answers the confidence question "does this crop carry real
# evidence", so a crop REJECTED by its peers only counts as a genuine
# disagreement once it clears it -- a 2-tick outlier is noise, a 30-tick one is a
# second opinion. Measured on the 133 GT sheets the accept set is bit-identical
# for every floor from 6 to 30, so nothing here is fitted.
DISAGREEMENT_FLOOR = DISTANCE_WEIGHT_FLOOR


def _harmonic_of(a: float, b: float):
    """If a/b is close to a clean harmonic ratio, return it, else None."""
    if not (a and b) or a <= 0 or b <= 0:
        return None
    r = a / b
    best = min(HARMONICS, key=lambda h: abs(math.log(r / h)))
    return best if abs(r / best - 1.0) <= HARMONIC_TOL and abs(best - 1.0) > 1e-9 else None


def reconcile_parent(crops, anchor=None, anchor_tol=0.25, frame_width_px=None):
    """Fuse every ruler crop of one parent image into a single CF.

    crops : [{key, cf, weight, ruler_class, skipped, skip_reason}]
            `cf` is that crop's own fused CF in the parent's WORKING frame;
            `weight` is its evidence count (ticks used).
    anchor: the parent's MP-predicted CF, working frame.

    Returns a dict with the parent CF, per-crop verdicts and the reasoning.
    """
    # Plausibility-check the anchor BEFORE trusting it. The MP regression floors
    # at ~69-83 px/cm in the working frame regardless of true scale, so on a
    # low-MP sheet it can be ~2x wrong while still looking healthy.
    anchor_dropped = False
    if anchor and frame_width_px:
        implied_cm = float(frame_width_px) / float(anchor)
        if implied_cm < MIN_FRAME_CM:
            anchor_dropped = True
            anchor = None

    live = [c for c in crops if not c.get("skipped") and c.get("cf")
            and math.isfinite(c["cf"]) and c["cf"] > 0]
    out = dict(n_crops=len(crops), n_live=len(live), anchor=anchor,
               anchor_dropped=anchor_dropped,
               cf_px_per_cm=None, cf_px_per_cm_measured=None,
               ok=False, confidence="low",
               confidence_reasons=["no CF was produced"],
               anchor_supported=False, anchor_log_dist=None,
               corroborated_by_peers=False, n_dissenting=0,
               method=None, spread=None,
               agree=False, per_crop=[], reason=None)
    if not live:
        out["reason"] = "no crop produced a CF"
        for c in crops:
            out["per_crop"].append(dict(key=c.get("key"), cf=c.get("cf"),
                                        ruler_class=c.get("ruler_class"),
                                        verdict="skipped" if c.get("skipped") else "no CF",
                                        note=c.get("skip_reason")))
        return out

    # 1. anchor admissibility -- the absolute reference
    for c in live:
        c["_ok_anchor"] = (True if not anchor
                           else abs(c["cf"] / anchor - 1.0) <= anchor_tol)

    # 2. peer clusters: group crops that agree within PEER_TOL
    clusters: list[list[dict]] = []
    for c in sorted(live, key=lambda x: -x.get("weight", 1)):
        for cl in clusters:
            if abs(c["cf"] / cl[0]["cf"] - 1.0) <= PEER_TOL:
                cl.append(c); break
        else:
            clusters.append([c])

    # 3. rank clusters:
    #      1st  crops inside the anchor gate
    #      2nd  closeness to the anchor -- but ONLY for a PLAUSIBLE anchor and ONLY
    #           for a cluster carrying >= DISTANCE_WEIGHT_FLOOR kept ticks
    #      3rd  total tick evidence      4th  crop count
    # Distance was tried as an ungated 2nd key once and reverted, because on
    # MnhnL_2417928731 the anchor is itself ~21% wrong. The plausibility check and
    # the evidence floor are what make it safe to bring back.
    def score(cl):
        w = sum(c.get("weight", 1) for c in cl)
        if anchor and w >= DISTANCE_WEIGHT_FLOOR:
            closeness = -min(abs(c["cf"] / anchor - 1.0) for c in cl)
        else:
            # NOT 0.0 -- real distances are negative, so 0.0 would outrank every
            # genuinely-scored cluster. An under-evidenced cluster must sit BELOW
            # them and fall through to the tick-weight key.
            closeness = -math.inf
        return (sum(1 for c in cl if c["_ok_anchor"]), closeness, w, len(cl))

    clusters.sort(key=score, reverse=True)
    win = clusters[0]
    anchored = any(c["_ok_anchor"] for c in win)

    w = [max(1, c.get("weight", 1)) for c in win]
    cf = sum(c["cf"] * wi for c, wi in zip(win, w)) / sum(w)
    vals = [c["cf"] for c in win]
    spread = ((max(vals) - min(vals)) / (sum(vals) / len(vals))) if len(vals) > 1 else None

    if anchor_dropped:
        method = "peer-cluster (anchor DROPPED as implausible)"
    elif anchor is None:
        method = "peer-cluster (no anchor available)"
    elif anchored:
        method = "peer-cluster+anchor"
    else:
        method = "peer-cluster (NO anchor support)"

    # ---- how much can this CF be trusted? ----------------------------------
    # `ok` used to mean only "some crop landed within +-25% of the MP anchor".
    # That window is wider than the smallest mistake the engine can make: every
    # way of misreading the unit moves the CF by at least one rung of the unit
    # ladder, and the smallest rung is 1.2598 (see ruler_units.min_rung_ratio), so
    # +-25% contains the one-rung-down alternative at 0.794 outright. A reading
    # sitting nearer a WRONG rung than the anchor was still being called
    # anchor-supported, which is why the old accept set carried a 17.6% p90.
    #
    # Three independent things can corroborate a sheet's CF, and the flag now
    # says which of them actually did:
    #   ANCHOR  the CF sits within half a rung (log-space) of a PLAUSIBLE MP
    #           anchor -- i.e. it is closer to the anchor than to any misnaming
    #           of itself.
    #   PEERS   two or more crops of the same sheet independently agree. They
    #           are separate photographs of the same physical scale, so this is
    #           real corroboration -- but NOT sufficient on its own (see the
    #           module docstring: the DBG pair agreed to 0.7% and were both 2x
    #           wrong), so it only carries a sheet that has no usable anchor.
    #   No DISSENT  no crop that carries real evidence was rejected. A rejected
    #           sibling with a genuine tick count means two measurements of one
    #           sheet disagree and nothing here can say which is right.
    anchor_log = (abs(math.log(cf / anchor)) if (anchor and cf > 0) else None)
    by_anchor = bool(anchor_log is not None and anchor_log <= RUNG_HALF_LOG)
    # LENGTH-BACKED ACCEPTANCE. When the reading and the anchor disagree, one of them is wrong and
    # the anchor distance cannot say which -- measured on the sheets this gate withheld, the true
    # failures (0.143, 0.145, 0.192) sit INTERLEAVED with the correct readings (0.117 ... 0.228), so
    # no threshold on that distance separates them. Ruler length does, because it is independent
    # evidence: all six correct readings came off 10.3-11.4 cm of ruler and all three genuine
    # failures off 5.3-5.8 cm.
    #
    # Still bounded by ONE FULL rung -- the smallest error any unit misnaming can produce -- so a
    # reading two rungs from the anchor is never admitted however long its ruler. That bound is a
    # guard rail rather than a measured one: no crop in the reference corpus reaches it.
    #
    # The cost is a genuinely short ruler that also disagrees with the anchor, which is deferred to
    # the anchor. That is the right call: a short ruler read CORRECTLY agrees with the anchor and
    # never reaches this branch at all.
    win_len = max((c.get("implied_len_cm") or 0.0) for c in win) if win else 0.0
    by_long_ruler = bool(anchor_log is not None and not by_anchor
                         and anchor_log <= 2.0 * RUNG_HALF_LOG
                         and win_len >= MIN_TRUSTED_RULER_CM)
    by_anchor_weak = bool(anchor_log is not None and not by_anchor
                          and anchor_log <= math.log1p(anchor_tol))
    by_peers = bool(len(win) >= 2 and spread is not None and spread <= PEER_TOL)
    dissent = [c for c in crops
               if not c.get("skipped") and c.get("cf") and id(c) not in
               {id(x) for x in win} and (c.get("weight") or 0) >= DISAGREEMENT_FLOOR]

    reasons = []
    if by_anchor:
        reasons.append("CF within half a unit-ladder rung of a plausible MP anchor")
    elif by_anchor_weak:
        reasons.append("CF agrees with the MP anchor only loosely (>half a rung)")
    elif anchor_dropped:
        reasons.append("MP anchor implied an impossible sheet width and was dropped")
    elif anchor is None:
        reasons.append("no MP anchor available")
    else:
        reasons.append("no crop agreed with the MP anchor")
    if by_long_ruler:
        reasons.append(f"CF is off the MP anchor but rests on {win_len:.1f} cm of ruler "
                       f"(>= {MIN_TRUSTED_RULER_CM:g} cm), within one unit-ladder rung")
    if by_peers:
        reasons.append(f"{len(win)} crops of this sheet agree within {PEER_TOL:.0%}")
    if dissent:
        overruled = " (OVERRULED by the anchor)" if by_anchor else ""
        reasons.append(f"{len(dissent)} well-supported crop(s) DISAGREE "
                       f"({', '.join(str(c.get('key')) for c in dissent)}){overruled}")

    # Dissent can no longer veto a winner the ANCHOR has independently confirmed. It used to veto
    # unconditionally, which cost real CFs: on the 549 human-measured sheets, 22 of the 26 withheld
    # sheets had a measurement within 10% of the human value, and 19 of those were vetoed by a
    # single dissenting crop while the anchor agreed with the winner. Requiring `not by_anchor`
    # recovered all 19 (mean error 1.29% -- the same as the humans' own repeat measurements) and
    # leaked ZERO bad CFs: of the 20 sheets that had dissent at all, the veto was blocking a good CF
    # on 19 and a bad one on 1, and that one is still blocked because the anchor does not confirm it.
    #
    # This does NOT weaken the anchor test, which is the thing that catches the failure the module
    # docstring warns about (the DBG pair agreed with each other to 0.7% and were both 2x wrong --
    # the anchor caught it, and would still catch it). It only stops a dissenting sibling from
    # overruling the absolute reference this module already declares is the absolute reference.
    # See leafmachine3/modules/experiments/MP_range (dissent_eval.py) for the replay + evidence.
    if dissent and not by_anchor:
        confidence = "low"
    elif by_anchor:
        confidence = "high"
    elif by_long_ruler:
        confidence = "high"
    elif anchor is None and by_peers:
        # The weakest member of `high`, admitted deliberately: the docstring's
        # warning that peer agreement is insufficient is about peer agreement
        # OVERRULING an anchor, and this branch is reached only when there is no
        # usable anchor at all, so the alternative is not a dissenting reference
        # but no reference. Both sheets carried this way in the GT set (2 and 3
        # agreeing crops) land within 1.3%.
        confidence = "high"
    elif by_anchor_weak:
        confidence = "medium"
    else:
        confidence = "low"

    # A sheet that does not clear the gate publishes NO conversion factor at all.
    # Will's call, and the right one: a wrong CF silently corrupts every downstream
    # measurement on that sheet, whereas a null one is a visible absence that falls
    # back to the MP-regressed anchor (specimen.cf_px_per_cm_predicted_by_mp), which
    # carries a known 4.4 px/cm rmse instead of an unknown unit-misnaming error.
    # `cf_px_per_cm_measured` keeps the rejected reading for QC and debugging -- it
    # must never be published as the sheet's CF.
    out.update(cf_px_per_cm=(cf if confidence == "high" else None),
               cf_px_per_cm_measured=cf,
               ok=(confidence == "high"),
               confidence=confidence, confidence_reasons=reasons,
               anchor_supported=bool(anchored and anchor is not None),
               length_backed=by_long_ruler, win_ruler_len_cm=(win_len or None),
               anchor_log_dist=anchor_log, corroborated_by_peers=by_peers,
               n_dissenting=len(dissent),
               method=method,
               spread=spread, agree=(spread is not None and spread <= PEER_TOL),
               n_clusters=len(clusters),
               pct_vs_anchor=(None if not anchor else 100.0 * (cf / anchor - 1.0)))
    if anchor_dropped:
        out["reason"] = ("the MP anchor implied an impossible sheet width and was dropped; "
                         "this CF rests on peer agreement alone")
    elif not anchored:
        out["reason"] = ("no crop agreed with the MP anchor; the crops agree with each "
                         "OTHER but may share a common harmonic error")

    winners = {id(c) for c in win}
    for c in crops:
        if c.get("skipped"):
            out["per_crop"].append(dict(key=c.get("key"), cf=None,
                                        ruler_class=c.get("ruler_class"),
                                        verdict="skipped", note=c.get("skip_reason")))
            continue
        if id(c) in winners:
            v, note = "used", None
        else:
            h = _harmonic_of(c.get("cf") or 0.0, cf)
            v = "rejected"
            note = (f"{h:g}x the parent CF -- wrong harmonic" if h
                    else "disagrees with the parent CF")
        out["per_crop"].append(dict(
            key=c.get("key"), cf=c.get("cf"), ruler_class=c.get("ruler_class"),
            weight=c.get("weight"), verdict=v, note=note,
            pct_vs_parent=(None if not c.get("cf") else 100.0 * (c["cf"] / cf - 1.0)),
            pct_vs_anchor=(None if not (anchor and c.get("cf"))
                           else 100.0 * (c["cf"] / anchor - 1.0))))
    return out
