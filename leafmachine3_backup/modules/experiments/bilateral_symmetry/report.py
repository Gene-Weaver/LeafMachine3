"""The bilateral-symmetry HTML report.

Style is matched to ``leafmachine3/setup/timing.py`` -- the ``:root`` palette and the core element
rules below are copied VERBATIM from that file's ``_HTML_SHELL`` so the two reports read as one
system; everything after the ``EXTENSIONS`` marker is new and specific to this report.

The report is deliberately a teaching document as much as a results dump: every metric is stated as
a formula, with what it is sensitive to and what it is blind to, because the point of the experiment
is to decide WHICH symmetry metrics are worth keeping.
"""
from __future__ import annotations

import html
import json
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# metric catalog -- (key, display, formula, what it tells you)
# --------------------------------------------------------------------------- #
CATALOG: list[tuple[str, str, str, str, str]] = [
    ("area", "AR", "AR = &Sigma;A<sub>Li</sub> / &Sigma;A<sub>Ri</sub>",
     "Shi et al. 2018",
     "Total left:right area ratio. Simple and comparable to the literature, but <b>blind to "
     "localized asymmetry</b> &mdash; a bulge on the left near the tip cancels a bulge on the right "
     "near the base and AR still reads 1.0."),
    ("area", "A*", "A* = (A<sub>L</sub> &minus; A<sub>R</sub>) / (A<sub>L</sub> + A<sub>R</sub>)",
     "signed, bounded",
     "The same information as AR but bounded to [&minus;1, 1] and signed: positive means the "
     "viewer's left half is larger. Bounded means it can be averaged across a cohort without one "
     "extreme leaf dominating."),
    ("area", "SI<sub>A</sub>",
     "SI<sub>A</sub> = (1/n) &Sigma; |A<sub>Li</sub> &minus; A<sub>Ri</sub>| / (A<sub>Li</sub> + A<sub>Ri</sub>)",
     "Shi et al. 2018",
     "The paper's headline index: mean <i>proportional</i> disagreement between paired strips. "
     "Catches local mismatch that AR cancels away. Every strip counts equally, so a sliver of "
     "lamina at the very tip carries the same weight as the widest part of the blade."),
    ("area", "WSI<sub>A</sub>",
     "WSI<sub>A</sub> = &Sigma;|A<sub>Li</sub> &minus; A<sub>Ri</sub>| / &Sigma;(A<sub>Li</sub> + A<sub>Ri</sub>)",
     "area-weighted",
     "The area-weighted counterpart of SI<sub>A</sub>, and the closest thing here to a directly "
     "interpretable &ldquo;what fraction of the leaf fails to mirror&rdquo;. Read it as a <b>lower "
     "bound</b> on the symmetric-difference fraction, not as equal to it: summing "
     "|A<sub>Li</sub>&minus;A<sub>Ri</sub>| collapses the distribution of lamina <i>within</i> each "
     "strip, so it equals the true symmetric difference only when the two halves of every strip "
     "nest perfectly. The Dice/SD family below measures the real overlap."),
    ("area", "DSI<sub>A</sub>",
     "DSI<sub>A</sub> = (1/n) &Sigma; (A<sub>Li</sub> &minus; A<sub>Ri</sub>) / (A<sub>Li</sub> + A<sub>Ri</sub>)",
     "signed strip mean",
     "Mean <i>directional</i> bias. Read it against SI<sub>A</sub>: SI high with DSI &asymp; 0 means "
     "large differences that alternate sides; SI &asymp; |DSI| means one side is consistently larger."),
    ("area", "RMSE<sub>A</sub>",
     "RMSE<sub>A</sub> = &radic;( (1/n) &Sigma;(A<sub>Li</sub> &minus; A<sub>Ri</sub>)&sup2; )",
     "Shi et al. 2018",
     "Unstandardized, so it grows with leaf size. Included for comparability with the literature; "
     "prefer SI/WSI when comparing leaves of different sizes."),
    ("width", "MAE<sub>w</sub> / RMSE<sub>w</sub> / NRMSE<sub>w</sub>",
     "NRMSE<sub>w</sub> = &radic;( mean( ((w<sub>L</sub>&minus;w<sub>R</sub>)/(w<sub>L</sub>+w<sub>R</sub>))&sup2; ) )",
     "margin envelope",
     "Distance between the two half-width profiles. The normalized form is scale-free. Widths here "
     "are the margin <i>envelope</i> (max |u| on that side), which is what a caliper measures &mdash; "
     "not area/&Delta;s, which differs for a lobed leaf with a deep sinus."),
    ("width", "r<sub>w</sub>", "r<sub>w</sub> = corr( w<sub>L</sub>(s), w<sub>R</sub>(s) )",
     "profile shape",
     "Whether the two sides expand and contract at the <i>same longitudinal positions</i>, "
     "independent of how large those expansions are. RMSE and r<sub>w</sub> are complementary: a "
     "leaf can have well-aligned lobes of unequal size (high r, high RMSE) or equal-sized lobes in "
     "the wrong places (low r, moderate RMSE)."),
    ("shape", "Dice / IoU / SD",
     "Dice = 2|L&prime; &cap; R&prime;| / (|L&prime;| + |R&prime;|) &nbsp;&middot;&nbsp; SD = 1 &minus; Dice",
     "whole-shape overlap",
     "The two halves straightened into (s, u) coordinates, the right one reflected, then overlapped. "
     "This is the only family that accounts for the <b>entire outline at once</b> rather than a "
     "reduction to strips. Reflection happens after straightening, so a curved midvein does not "
     "count as asymmetry."),
    ("shape", "Hausdorff",
     "H = max{ sup<sub>x&isin;L&prime;</sub> inf<sub>y&isin;R&prime;</sub> d(x,y), &nbsp;sup<sub>y&isin;R&prime;</sub> inf<sub>x&isin;L&prime;</sub> d(x,y) }",
     "worst-case margin",
     "Sensitive to a <i>single</i> unmatched lobe or tear, where the mean measures stay low. That "
     "makes it the natural detector for one-off segmentation damage as opposed to gentle overall "
     "lopsidedness."),
    ("shape", "D<sub>c</sub>",
     "D<sub>c</sub> = &radic;( (s&#772;<sub>L</sub>&minus;s&#772;<sub>R</sub>)&sup2; + (u&#772;<sub>L</sub>&minus;u&#772;<sub>R</sub>)&sup2; )",
     "half centroids",
     "Separates <i>longitudinal</i> displacement of area (one half's mass sits nearer the base) from "
     "<i>lateral</i> extent (one half reaches further from the midvein). Two leaves with identical "
     "SI<sub>A</sub> can differ entirely in which of those is happening."),
    ("spatial", "C(s), C<sub>max</sub>, I<sub>C</sub>",
     "C(s<sub>k</sub>) = &Sigma;<sub>i&le;k</sub>(A<sub>Li</sub> &minus; A<sub>Ri</sub>) / (A<sub>L</sub> + A<sub>R</sub>)",
     "cumulative",
     "The running signed imbalance from tip to base. Its endpoint is A*, but the <i>path</i> is the "
     "information: two leaves can both end at zero, one by staying balanced throughout and the other "
     "by cancelling a left-heavy apex against a right-heavy base."),
    ("spatial", "apex / mid / base thirds",
     "SI<sub>apex</sub> = mean|a(s)| for s &isin; [0, &#8531;]",
     "where it happens",
     "Localizes the disagreement. s = 0 is the <b>tip</b> and s = 1 the <b>base</b>. Apex asymmetry "
     "often means a damaged or folded tip; basal asymmetry is frequently genuine biology "
     "(oblique-based leaves) or a petiole-junction segmentation artifact."),
    ("axis", "sinuosity, D<sub>m</sub>, K<sub>m</sub>",
     "S<sub>m</sub> = L<sub>midvein</sub> / D<sub>base,tip</sub>",
     "midvein geometry",
     "How far the traced midvein departs from a straight chord: arclength ratio, maximum "
     "perpendicular deviation, and total absolute turning. These are the <i>cause</i> of the "
     "chord-vs-midvein gap below, so they predict which leaves the literature's straight-axis method "
     "will misjudge."),
    ("axis", "&Delta;Q<sub>axis</sub>", "&Delta;Q<sub>axis</sub> = Q<sub>chord</sub> &minus; Q<sub>midvein</sub>",
     "the LM3-only measurement",
     "Every metric computed twice &mdash; once against the straight base&rarr;tip chord the published "
     "methods must use, once against the actual traced midvein. The difference is <b>apparent "
     "asymmetry that is really just a curved midvein</b>. No silhouette-only method can measure this."),
]


def _esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def _stat(label: str, value: Any) -> str:
    return (f'<div class="stat"><div class="v">{_esc(value)}</div>'
            f'<div class="l">{_esc(label)}</div></div>')


def _fig(uri: Optional[str], caption: str = "") -> str:
    if not uri:
        return '<div class="note">figure unavailable</div>'
    cap = f'<figcaption>{caption}</figcaption>' if caption else ""
    return f'<figure class="fig"><img src="{uri}" alt="">{cap}</figure>'


def _catalog_html() -> str:
    fams = {"area": "Area, from paired strips", "width": "Half-width profiles",
            "shape": "Whole-shape overlap", "spatial": "Where the asymmetry sits",
            "axis": "The axis itself"}
    out = []
    for fam, title in fams.items():
        items = [c for c in CATALOG if c[0] == fam]
        rows = "".join(
            f'<tr><td class="mono mkey">{name}</td>'
            f'<td class="formula">{formula}<div class="src">{_esc(src)}</div></td>'
            f'<td class="mdesc">{desc}</td></tr>'
            for _f, name, formula, src, desc in items)
        out.append(f'<h3 class="fam">{_esc(title)}</h3>'
                   f'<div class="tblwrap"><table class="cat"><tbody>{rows}</tbody></table></div>')
    return "".join(out)


def _leaf_card(c: dict) -> str:
    badges = ""
    if c.get("is_archetypal"):
        badges += '<span class="badge good">archetypal</span>'
    if c.get("truncated"):
        badges += '<span class="badge bad">truncated</span>'
    for r in (c.get("reasons") or [])[:3]:
        badges += f'<span class="badge warn">{_esc(r)}</span>'
    chips = "".join(
        f'<span class="chip"><b>{_esc(k)}</b> {_esc(v)}</span>' for k, v in (c.get("chips") or []))
    return (f'<div class="leafcard">'
            f'<div class="lhead"><span class="lname mono">{_esc(c["name"])}</span>{badges}</div>'
            f'<div class="chips">{chips}</div>'
            f'<img src="{c["panel"]}" alt="">'
            f'</div>')


def _interactive_scatter(pts: list[dict]) -> str:
    """A clickable SVG scatter of archetype score vs SI_A, wired to a per-leaf thumbnail.

    Hand-built rather than a matplotlib PNG because the point of it is to be CLICKED: every leaf
    carries its own oriented-mask thumbnail, so a suspicious point can be inspected without
    scrolling to a gallery. Plain inline SVG + a few lines of JS keeps the page self-contained.
    """
    if not pts:
        return ""
    W, H = 660, 470
    L, R, T, B = 62, 16, 16, 46
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    x1 = max(0.05, max(xs) * 1.06)
    y1 = 1.0

    def sx(v: float) -> float:
        return L + (v / x1) * (W - L - R)

    def sy(v: float) -> float:
        return T + (1.0 - v / y1) * (H - T - B)

    grid = []
    for i in range(6):
        gx = x1 * i / 5
        grid.append(f'<line x1="{sx(gx):.1f}" y1="{T}" x2="{sx(gx):.1f}" y2="{H - B}" class="gl"/>'
                    f'<text x="{sx(gx):.1f}" y="{H - B + 16}" class="tk" text-anchor="middle">{gx:.2f}</text>')
        gy = i / 5
        grid.append(f'<line x1="{L}" y1="{sy(gy):.1f}" x2="{W - R}" y2="{sy(gy):.1f}" class="gl"/>'
                    f'<text x="{L - 8}" y="{sy(gy) + 3.5:.1f}" class="tk" text-anchor="end">{gy:.1f}</text>')

    # Three tiers, because they answer two different questions. VETOED (a score term hit 0) is the
    # only hard reject; USABLE is everything else; EXEMPLAR is the strict top slice, drawn as a
    # ringed usable point rather than a separate color so the usable/vetoed split stays the
    # dominant read.
    circles = []
    for i, p in enumerate(pts):
        # Vetoed = disqualified for a STRUCTURAL reason: either a score term was driven to zero, or
        # a hard gate failed. A gate failure with a nonzero score would otherwise sit in the green
        # tier looking usable, which is exactly what it is not.
        if p["y"] <= 0.001 or not p.get("gates", True):
            cls = "pt zero"
        elif p["arch"]:
            cls = "pt use exemplar"
        else:
            cls = "pt use"
        circles.append(
            f'<circle cx="{sx(p["x"]):.1f}" cy="{sy(p["y"]):.1f}" r="5" class="{cls}" '
            f'data-i="{i}" tabindex="0"><title>{_esc(p["name"])}</title></circle>')

    n_zero = sum(1 for p in pts if p["y"] <= 0.001 or not p.get("gates", True))
    n_ex = sum(1 for p in pts if p["arch"])
    n_use = len(pts) - n_zero

    payload = ",".join(
        "{n:%s,s:%s,d:%s,a:%s,t:%s,img:%s}" % (
            json.dumps(p["name"]), json.dumps(p["chips"]), json.dumps(p["dice"]),
            json.dumps(p["astar"]), json.dumps(bool(p["arch"])), json.dumps(p["img"]))
        for p in pts)

    return f"""
<div class="scatterwrap">
  <div class="scatterbox">
    <svg viewBox="0 0 {W} {H}" class="scatter" role="img">
      <g class="grid">{''.join(grid)}</g>
      <text x="{L + (W - L - R) / 2:.0f}" y="{H - 6}" class="axlab" text-anchor="middle">SI_A on the midvein axis &#8212; lower is more symmetric</text>
      <text x="14" y="{T + (H - T - B) / 2:.0f}" class="axlab" text-anchor="middle" transform="rotate(-90 14 {T + (H - T - B) / 2:.0f})">archetype score</text>
      <g class="pts">{''.join(circles)}</g>
    </svg>
    <div class="skey">
      <span><i class="d use"></i>usable &mdash; {n_use}</span>
      <span><i class="d use exemplar"></i>exemplar &mdash; {n_ex}</span>
      <span><i class="d zero"></i>vetoed &mdash; {n_zero}</span>
      <span class="hint2">click any point</span></div>
  </div>
  <div class="detail" id="lfDetail">
    <div class="dname mono" id="lfName">click a point to inspect a leaf</div>
    <img id="lfImg" alt="" style="display:none">
    <div class="dchips" id="lfChips"></div>
    <div class="dlegend"><span><i style="background:#38bdf8"></i>left half</span>
      <span><i style="background:#fb923c"></i>right half</span>
      <span><i style="background:#4ade80"></i>midvein axis</span>
      <span><i style="background:#6f757f"></i>chord axis</span></div>
  </div>
</div>
<script>
(function(){{
  var D=[{payload}];
  var img=document.getElementById('lfImg'), nm=document.getElementById('lfName'),
      ch=document.getElementById('lfChips'), sel=null;
  function show(i){{
    var d=D[i]; if(!d) return;
    if(sel) sel.classList.remove('sel');
    sel=document.querySelector('circle[data-i="'+i+'"]'); if(sel) sel.classList.add('sel');
    nm.textContent=d.n+(d.t?'  \\u2713 archetypal':'');
    img.src=d.img; img.style.display='block';
    ch.innerHTML='<span class="chip"><b>SI_A</b> '+d.s+'</span>'+
                 '<span class="chip"><b>Dice</b> '+d.d+'</span>'+
                 '<span class="chip"><b>A*</b> '+d.a+'</span>';
  }}
  document.querySelectorAll('circle.pt').forEach(function(c){{
    c.addEventListener('click',function(){{ show(+c.dataset.i); }});
    c.addEventListener('keydown',function(e){{ if(e.key==='Enter'||e.key===' '){{ e.preventDefault(); show(+c.dataset.i); }} }});
  }});
}})();
</script>"""


def render_report(ctx: dict) -> str:
    """Render the full report. ``ctx`` is assembled by :mod:`run`."""
    stats = "".join(_stat(l, v) for l, v in ctx["stats"])
    figs = ctx.get("figures", {})

    gallery_sections = []
    for title, hint, cards in ctx.get("galleries", []):
        body = "".join(_leaf_card(c) for c in cards)
        gallery_sections.append(
            f'<h3 class="fam">{_esc(title)}</h3><p class="hint">{hint}</p>'
            f'<div class="gallery">{body}</div>')

    findings = "".join(f"<li>{f}</li>" for f in ctx.get("findings", []))
    limits = "".join(f"<li>{f}</li>" for f in ctx.get("limitations", []))

    return _SHELL.format(
        run=_esc(ctx["run_name"]),
        subtitle=ctx.get("subtitle", ""),
        stats=stats,
        findings=(f'<div class="note keynote"><b>What the numbers say</b><ul>{findings}</ul></div>'
                  if findings else ""),
        method=ctx.get("method_html", ""),
        catalog=_catalog_html(),
        fig_axis=_fig(figs.get("chord_vs_midvein"),
                      "Each point is one leaf: the same metric measured against the straight chord "
                      "(y) and against the traced midvein (x). Points above the diagonal are leaves "
                      "the published straight-axis method reports as more asymmetric than they are."),
        axis_note=ctx.get("axis_note", ""),
        fig_dist=_fig(figs.get("distributions"),
                      "Cohort distribution of each metric, median marked."),
        fig_corr=_fig(figs.get("correlation"),
                      "Spearman correlation between metrics. Blocks of near-1 correlation are "
                      "redundant measurements &mdash; keep one per block."),
        corr_note=ctx.get("corr_note", ""),
        fig_taylor=(_fig(figs.get("taylor"),
                         "<b>Raw px&sup2;.</b> Taylor's power law for leaf asymmetry (Wang et al. "
                         "2018): per-leaf mean vs variance of the absolute strip differences, "
                         "log-log, with the OLS fit.")
                    + _fig(figs.get("taylor_norm"),
                           "<b>Size-normalized.</b> The same fit on the dimensionless per-strip "
                           "differences, which removes the leaf-size scaling that pins the raw "
                           "exponent near 2.")),
        taylor_note=ctx.get("taylor_note", ""),
        fig_score=(_interactive_scatter(ctx.get("scatter_points", []))
                   or _fig(figs.get("score_vs_symmetry"),
                           "Archetype score against SI<sub>A</sub>, colored by the archetypal flag.")),
        score_note=ctx.get("score_note", ""),
        galleries="".join(gallery_sections),
        table=ctx.get("table_html", ""),
        limitations=(f'<div class="note"><b>Limitations &amp; caveats</b><ul>{limits}</ul></div>'
                     if limits else ""),
        generated=_esc(ctx.get("generated", "")),
        cohort=_esc(ctx.get("cohort_note", "")),
    )


_SHELL = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LM3 — leaf bilateral symmetry · {run}</title>
<style>
:root{{--bg:#101012;--panel:#191a1d;--panel2:#1f2024;--ink:#e8e8ea;--mute:#9ca3af;--dim:#6f757f;
 --line:#2a2b30;--acc:#fb923c;--acc2:#38bdf8;--acc3:#4ade80;--warn:#fbbf24;--bad:#f87171}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
 font:16px/1.66 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
 -webkit-font-smoothing:antialiased}}
.wrap{{max-width:1180px;margin:0 auto;padding:48px 26px 90px}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,"DejaVu Sans Mono",monospace}}
header{{border-bottom:2px solid var(--line);padding-bottom:22px;margin-bottom:8px}}
.kicker{{font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--acc);font-weight:700}}
h1{{font-size:2.2rem;line-height:1.1;margin:.3em 0 .3em;letter-spacing:-.022em}}
.sub{{color:var(--mute);font-size:1.02rem;max-width:80ch;margin:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:26px 0 8px}}
.grid.g3{{grid-template-columns:repeat(3,minmax(0,1fr))}}
.stat{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:13px 15px}}
.stat .v{{font-size:1.34rem;font-weight:650;letter-spacing:-.02em;color:var(--acc2);
 font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,"DejaVu Sans Mono",monospace}}
.stat .l{{font-size:12px;color:var(--mute);margin-top:6px}}
h2{{font-size:1.16rem;margin:44px 0 4px;letter-spacing:-.01em}}
.hint{{color:var(--dim);font-size:13px;margin:0 0 16px;max-width:96ch}}
.tblwrap{{overflow-x:auto;margin:20px 0;border:1px solid var(--line);border-radius:9px;background:var(--panel)}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
thead th{{text-align:right;padding:11px 13px;color:var(--mute);font-weight:600;font-size:11.5px;
 letter-spacing:.05em;text-transform:uppercase;border-bottom:1px solid var(--line);white-space:nowrap;
 cursor:help;border-bottom-style:dotted;border-bottom-width:1px}}
thead th:hover{{color:var(--ink)}}
thead th:first-child{{text-align:left}}
tbody td{{padding:9px 13px;border-bottom:1px solid var(--line);vertical-align:middle}}
tbody tr:last-child td{{border-bottom:none}}
td.num{{text-align:right;font-variant-numeric:tabular-nums;
 font-family:ui-monospace,SFMono-Regular,Menlo,"DejaVu Sans Mono",monospace;white-space:nowrap}}
td.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,"DejaVu Sans Mono",monospace}}
.dim{{color:var(--dim)}}
tbody tr:hover td{{background:#1c1d21}}
.note{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--warn);
 border-radius:8px;padding:12px 16px;margin:18px 0 4px;font-size:13.4px;color:var(--mute);
 line-height:1.6;max-width:96ch}}
.note b{{color:var(--ink);font-weight:640}}
.note ul{{margin:.5em 0 .7em;padding-left:20px}}
.note li{{margin:.34em 0}}
footer{{margin-top:40px;color:var(--dim);font-size:12px;border-top:1px solid var(--line);padding-top:16px}}
/* ---------------- EXTENSIONS (specific to this report) ---------------- */
.keynote{{border-left-color:var(--acc3)}}
.fam{{font-size:.86rem;color:var(--mute);letter-spacing:.09em;text-transform:uppercase;
 margin:26px 0 8px;font-weight:650}}
figure.fig{{margin:16px 0 8px;background:var(--panel);border:1px solid var(--line);
 border-radius:9px;padding:14px}}
figure.fig img{{width:100%;height:auto;display:block;border-radius:5px}}
figure.fig figcaption{{color:var(--dim);font-size:12.5px;margin-top:11px;line-height:1.55}}
table.cat td{{vertical-align:top}}
td.mkey{{white-space:nowrap;color:var(--acc2);font-weight:650;width:1%;padding-right:18px}}
td.formula{{font-family:ui-monospace,SFMono-Regular,Menlo,"DejaVu Sans Mono",monospace;
 font-size:12.2px;color:var(--ink);white-space:nowrap;width:1%;padding-right:22px}}
td.formula .src{{font-family:inherit;font-size:11px;color:var(--dim);margin-top:5px;
 letter-spacing:.03em;text-transform:uppercase;white-space:nowrap}}
td.mdesc{{color:var(--mute);font-size:13.2px;line-height:1.6}}
td.mdesc b{{color:var(--ink);font-weight:640}}
.gallery{{display:grid;grid-template-columns:1fr;gap:14px;margin:14px 0 8px}}
.leafcard{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:12px 14px}}
.leafcard img{{width:100%;height:auto;display:block;border-radius:5px;margin-top:10px}}
.lhead{{display:flex;align-items:center;gap:9px;flex-wrap:wrap}}
.lname{{font-size:12.5px;color:var(--ink)}}
.badge{{padding:1px 7px;border-radius:4px;font-size:9.5px;letter-spacing:.05em;
 text-transform:uppercase;white-space:nowrap;border:1px solid}}
.badge.good{{color:var(--acc3);border-color:var(--acc3)}}
.badge.warn{{color:var(--warn);border-color:var(--warn);text-transform:none;letter-spacing:.02em}}
.badge.bad{{color:var(--bad);border-color:var(--bad)}}
.chips{{display:flex;gap:14px;flex-wrap:wrap;margin-top:7px}}
.chip{{font-size:11.5px;color:var(--dim);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.chip b{{color:var(--mute);font-weight:500}}
.eq{{background:var(--panel2);border:1px solid var(--line);border-radius:7px;padding:10px 14px;
 margin:12px 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
 color:var(--ink);overflow-x:auto}}
p.body{{color:var(--mute);font-size:14px;max-width:88ch;line-height:1.7}}
p.body b{{color:var(--ink);font-weight:640}}
/* clickable scatter + leaf inspector */
.scatterwrap{{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(230px,1fr);gap:14px;
 margin:16px 0 8px;align-items:start}}
.scatterbox,.detail{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:12px 14px}}
svg.scatter{{width:100%;height:auto;display:block}}
svg.scatter .gl{{stroke:var(--line);stroke-width:1}}
svg.scatter .tk{{fill:var(--dim);font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
svg.scatter .axlab{{fill:var(--mute);font-size:12px}}
svg.scatter circle.pt{{fill:var(--dim);fill-opacity:.85;stroke:var(--bg);stroke-width:1;cursor:pointer;
 transition:r .08s ease,fill-opacity .08s ease}}
svg.scatter circle.pt:hover,svg.scatter circle.pt:focus{{r:8;fill-opacity:1;outline:none}}
svg.scatter circle.use{{fill:var(--acc3)}}
svg.scatter circle.exemplar{{stroke:var(--ink);stroke-width:2.2}}
svg.scatter circle.zero{{fill:var(--bad)}}
svg.scatter circle.sel{{stroke:var(--acc2);stroke-width:2.6;r:8}}
.skey{{display:flex;gap:16px;flex-wrap:wrap;margin-top:9px;font-size:11.5px;color:var(--mute)}}
.skey i.d{{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--acc3);
 margin-right:6px;vertical-align:0}}
.skey i.d.exemplar{{box-shadow:0 0 0 2px var(--ink)}}
.skey i.d.zero{{background:var(--bad)}}
.skey .hint2{{color:var(--dim);font-style:italic}}
.detail img{{width:100%;height:auto;display:block;border-radius:6px;margin:10px 0 2px;
 background:var(--panel2)}}
.dname{{font-size:11.5px;color:var(--ink);word-break:break-all;line-height:1.45}}
.dchips{{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px}}
.dlegend{{display:flex;gap:12px;flex-wrap:wrap;margin-top:11px;font-size:11px;color:var(--dim);
 border-top:1px solid var(--line);padding-top:9px}}
.dlegend i{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:0}}
@media (max-width:820px){{.scatterwrap{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<header><div class="kicker">LeafMachine3 · experiment</div>
<h1>Leaf bilateral symmetry</h1>
<p class="sub">{subtitle}</p></header>
<div class="grid g3">{stats}</div>
{findings}
<h2>Method</h2>
{method}
<h2>The metrics</h2>
<p class="hint">Every metric below is computed twice per leaf — once against the straight chord, once
against the traced midvein. n = number of arclength strips.</p>
{catalog}
<h2>Straight chord vs. traced midvein</h2>
<p class="hint">The measurement LM3 can make that a silhouette-only method cannot.</p>
{fig_axis}
{axis_note}
<h2>Cohort distributions</h2>
{fig_dist}
<h2>Which metrics are redundant?</h2>
{fig_corr}
{corr_note}
<h2>Taylor's power law</h2>
{fig_taylor}
{taylor_note}
<h2>Do symmetry metrics find archetypal leaves?</h2>
{fig_score}
{score_note}
{galleries}
<h2>Per-leaf results</h2>
<p class="hint">All leaves, ranked by archetype score. Hover a heading for its definition.</p>
{table}
{limitations}
<footer>Generated by <span class="mono">leafmachine3.modules.experiments.bilateral_symmetry</span>
 · {cohort} · {generated}. Left = the viewer's left of the tip-up oriented mask, which is consistent
 across leaves precisely because every mask is oriented; it is <b>not</b> a botanical
 adaxial/abaxial or anodic/cathodic call, so a cohort-mean signed imbalance near zero is expected
 even where real directional asymmetry exists.</footer>
</div></body></html>"""
