/* ==========================================================================
   Layout experiments -- shared renderer.

   Every mockup renders from the SAME real corpus (290 settings from
   settings_meta.json + builtin_defaults() + LM3_settings.yaml, and the leaf
   collage tool's 29 inputs). Each <div class="mount" data-layout data-src> is
   rendered by the matching function below and re-rendered on interaction, so
   the navigation each design depends on can actually be tried.

   These are MOCKUPS: bools toggle so the page feels alive, but nothing is
   persisted and no other control edits.
   ========================================================================== */
(function () {
  "use strict";

  var SRC = { settings: DATA, tool: TOOL };
  var esc = function (s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  };
  var el = function (h) { var d = document.createElement("div"); d.innerHTML = h; return d.firstElementChild; };

  /* ---------------------------------------------------------------- values */
  function hexOf(v) {
    if (Array.isArray(v) && v.length >= 3) {
      return "#" + v.slice(0, 3).map(function (c) {
        return Math.max(0, Math.min(255, Math.round(+c || 0))).toString(16).padStart(2, "0");
      }).join("");
    }
    var s = String(v == null ? "" : v).trim();
    if (/^#[0-9a-f]{6}$/i.test(s)) return s;
    if (/^white$/i.test(s)) return "#ffffff";
    if (/^black$/i.test(s)) return "#000000";
    return "#888888";
  }
  function short(v) {
    if (v == null || v === "") return "";
    if (typeof v === "boolean") return v ? "on" : "off";
    if (Array.isArray(v)) {
      if (v.length >= 3 && v.every(function (x) { return typeof x === "number"; })) return v.join(",");
      if (!v.length) return "none";
      return String(v[0]) + (v.length > 1 ? " +" + (v.length - 1) : "");
    }
    var s = String(v);
    return s.length > 34 ? "…" + s.slice(-33) : s;
  }

  /** A stand-in control, sized to its type instead of stretched to the row. */
  function ctl(r, size) {
    var c = "f" + (size ? " " + size : "");
    var v = r.v;
    if (r.t === "bool") {
      return '<span class="sw ' + (v ? "on" : "") + '" data-tog="' + esc(r.k) + '"><i></i></span>';
    }
    if (r.t === "enum") return '<span class="' + c + ' sel">' + esc(short(v)) + "<b>▾</b></span>";
    if (r.t === "color") {
      if (String(v).toLowerCase() === "transparent") {
        return '<span class="' + c + ' col"><i class="swatch checker"></i>transparent</span>';
      }
      return '<span class="' + c + ' col"><i class="swatch" style="background:' + hexOf(v)
        + '"></i>' + esc(short(v)) + "</span>";
    }
    if (r.t === "path") {
      return '<span class="' + c + ' pathf"><span class="ell">' + (esc(short(v)) || "&nbsp;") + "</span><b>…</b></span>";
    }
    if (r.t === "list") return '<span class="' + c + ' listf">' + (esc(short(v)) || "none") + "</span>";
    if (r.t === "int" || r.t === "float") return '<span class="' + c + ' num">' + esc(short(v)) + "</span>";
    return '<span class="' + c + '"><span class="ell">' + (esc(short(v)) || "&nbsp;") + "</span></span>";
  }
  function dots(r) {
    return (r.i ? '<i class="impdot" title="worth adjusting"></i>' : "")
      + (r.c ? '<i class="chgdot" title="differs from default"></i>' : "")
      + (!r.i && !r.c ? '<i class="nodot"></i>' : "");
  }

  /* ---------------------------------------------------------------- helpers */
  function bySection(d, id) { return d.rows.filter(function (r) { return r.s === id; }); }
  function groupsOf(rows) {
    var seen = [], by = {};
    rows.forEach(function (r) {
      var g = r.g || "Other";
      if (!by[g]) { by[g] = []; seen.push(g); }
      by[g].push(r);
    });
    return seen.map(function (g) { return { name: g, rows: by[g] }; });
  }
  function secLabel(d, id) {
    var s = d.sections.filter(function (x) { return x.id === id; })[0];
    return s ? s.label : id;
  }
  function countChanged(rows) { return rows.filter(function (r) { return r.c; }).length; }

  /* one compact label+control line, shared by A and B */
  function cgRow(r) {
    return '<div class="cg-row ' + (r.i ? "imp" : "") + '" title="' + esc(r.k) + "\n" + esc(r.h) + '">'
      + dots(r) + '<span class="cg-lbl">' + esc(r.l) + "</span>"
      + '<span class="cg-ctl">' + ctl(r) + "</span></div>"
      + '<div class="cg-help" hidden>' + esc(r.h) + "</div>";
  }

  /* ===================================================== A -- column grid == */
  function renderColumns(mount, d, st) {
    var rows = bySection(d, st.sec);
    if (st.imp) rows = rows.filter(function (r) { return r.i; });
    var chips = d.sections.map(function (s) {
      var n = bySection(d, s.id).length;
      return '<button class="chip ' + (s.id === st.sec ? "on" : "") + '" data-sec="' + s.id + '">'
        + esc(s.label) + "<b>" + n + "</b></button>";
    }).join("");
    var body = groupsOf(rows).map(function (g) {
      return '<div class="cg-hd">' + esc(g.name) + '<span class="cnt">' + g.rows.length + "</span></div>"
        + g.rows.map(cgRow).join("");
    }).join("");
    mount.innerHTML =
      '<div class="ctlbar"><div class="chips">' + chips + "</div>"
      + '<button class="tgl ' + (st.imp ? "on" : "") + '" data-imp>● Important only</button>'
      + '<button class="tgl ' + (st.desc ? "on" : "") + '" data-desc>ⓘ Descriptions</button>'
      + '<span class="sp"></span><span class="readout">' + rows.length + " of "
      + d.rows.length + " shown</span></div>"
      + '<div class="cg' + (st.desc ? " showhelp" : "") + '">' + (body || '<div class="cg-hd">nothing matches</div>') + "</div>";

    mount.querySelectorAll("[data-sec]").forEach(function (b) {
      b.onclick = function () { st.sec = b.dataset.sec; renderColumns(mount, d, st); };
    });
    mount.querySelector("[data-imp]").onclick = function () { st.imp = !st.imp; renderColumns(mount, d, st); };
    mount.querySelector("[data-desc]").onclick = function () { st.desc = !st.desc; renderColumns(mount, d, st); };
    wireToggles(mount, d, function () { renderColumns(mount, d, st); });
  }

  /* ======================================================== B -- rail ====== */
  function renderRail(mount, d, st) {
    var secRows = bySection(d, st.sec);
    var groups = groupsOf(secRows);
    if (!groups.some(function (g) { return g.name === st.grp; })) st.grp = groups.length ? groups[0].name : null;
    var cur = groups.filter(function (g) { return g.name === st.grp; })[0] || { name: "", rows: [] };

    var nav = d.sections.map(function (s) {
      var rs = bySection(d, s.id), ch = countChanged(rs);
      var open = s.id === st.sec;
      var sub = !open ? "" : groupsOf(rs).map(function (g) {
        return '<a class="md-sub ' + (g.name === st.grp ? "on" : "") + '" data-grp="' + esc(g.name)
          + '">' + esc(g.name) + '<span class="n">' + g.rows.length + "</span></a>";
      }).join("");
      return '<a class="md-item ' + (open ? "open" : "") + '" data-sec="' + s.id + '">' + esc(s.label)
        + '<span class="n">' + rs.length + "</span>"
        + (ch ? '<i class="chg" title="' + ch + ' changed"></i>' : "") + "</a>" + sub;
    }).join("");

    mount.innerHTML =
      '<div class="md"><nav class="md-rail"><div class="md-railhd">' + esc(d.title) + "</div>"
      + '<input class="md-search" placeholder="filter…">' + nav
      + '<div class="md-railft"><i class="chg"></i> ' + countChanged(d.rows) + " changed</div></nav>"
      + '<section class="md-pane"><header class="md-hd"><h4>' + esc(cur.name) + "</h4>"
      + '<span class="cnt">' + cur.rows.length + " settings · " + secLabel(d, st.sec) + "</span></header>"
      + '<div class="cg cg-2">' + cur.rows.map(cgRow).join("") + "</div>"
      + '<div class="md-ends">▲ this group ends here — no scrolling, and the rail still shows the '
      + "other " + (d.sections.length - 1) + " sections and where the changes are</div></section></div>";

    mount.querySelectorAll("[data-sec]").forEach(function (a) {
      a.onclick = function () { st.sec = a.dataset.sec; st.grp = null; renderRail(mount, d, st); };
    });
    mount.querySelectorAll("[data-grp]").forEach(function (a) {
      a.onclick = function (e) { e.stopPropagation(); st.grp = a.dataset.grp; renderRail(mount, d, st); };
    });
    var q = mount.querySelector(".md-search");
    q.oninput = function () {
      var t = q.value.trim().toLowerCase();
      mount.querySelectorAll(".md-item,.md-sub").forEach(function (a) {
        a.style.display = !t || a.textContent.toLowerCase().indexOf(t) >= 0 ? "" : "none";
      });
    };
    wireToggles(mount, d, function () { renderRail(mount, d, st); });
  }

  /* ======================================================= C -- table ====== */
  function renderTable(mount, d, st) {
    var rows = d.rows.slice();
    if (st.filter === "imp") rows = rows.filter(function (r) { return r.i; });
    if (st.filter === "chg") rows = rows.filter(function (r) { return r.c; });
    if (st.q) {
      var t = st.q.toLowerCase();
      rows = rows.filter(function (r) {
        return (r.k + " " + r.l + " " + r.h + " " + short(r.v)).toLowerCase().indexOf(t) >= 0;
      });
    }
    var key = st.sort;
    rows.sort(function (a, b) {
      var A = key === "grp" ? a.s + a.g : key === "val" ? short(a.v) : key === "chg" ? (a.c ? 0 : 1) : a.l;
      var B = key === "grp" ? b.s + b.g : key === "val" ? short(b.v) : key === "chg" ? (b.c ? 0 : 1) : b.l;
      return (A > B ? 1 : A < B ? -1 : 0) * (st.dir || 1);
    });
    var f = function (id, lbl, n) {
      return '<button class="pt-f ' + (st.filter === id ? "on" : "") + '" data-f="' + id + '">'
        + lbl + "<b>" + n + "</b></button>";
    };
    var arrow = function (k) { return st.sort === k ? (st.dir > 0 ? " ▲" : " ▼") : ""; };
    mount.innerHTML =
      '<div class="pt-bar">' + f("all", "All ", d.rows.length)
      + f("imp", "Important ", d.rows.filter(function (r) { return r.i; }).length)
      + f("chg", "Changed ", countChanged(d.rows))
      + '<input class="pt-q" placeholder="filter 290 by name, path, help or value…" value="' + esc(st.q || "") + '">'
      + '<span class="sp"></span><span class="pt-sort">' + rows.length + " rows</span></div>"
      + '<div class="pt-wrap"><table class="pt"><thead><tr><th data-s="chg" class="th-s"></th>'
      + '<th data-s="lbl" class="th-s">Setting' + arrow("lbl") + "</th>"
      + '<th data-s="val" class="th-s">Value' + arrow("val") + "</th>"
      + "<th>Default</th>"
      + '<th data-s="grp" class="th-s">Group' + arrow("grp") + "</th></tr></thead><tbody>"
      + rows.map(function (r) {
        return '<tr class="' + (r.c ? "chg" : "") + '" title="' + esc(r.h) + '">'
          + '<td class="c-dot">' + dots(r) + "</td>"
          + '<td class="c-name"><span class="k">' + esc(r.l) + '</span><span class="p">' + esc(r.k) + "</span></td>"
          + '<td class="c-val">' + ctl(r, "sm") + "</td>"
          + '<td class="c-def">' + (r.d == null ? "—" : esc(short(r.d))) + "</td>"
          + '<td class="c-grp">' + esc(r.g) + "</td></tr>";
      }).join("") + "</tbody></table></div>";

    mount.querySelectorAll("[data-f]").forEach(function (b) {
      b.onclick = function () { st.filter = b.dataset.f; renderTable(mount, d, st); };
    });
    mount.querySelectorAll("[data-s]").forEach(function (h) {
      h.onclick = function () {
        st.dir = st.sort === h.dataset.s ? -(st.dir || 1) : 1;
        st.sort = h.dataset.s; renderTable(mount, d, st);
      };
    });
    var q = mount.querySelector(".pt-q");
    q.oninput = function () { st.q = q.value; var p = q.selectionStart; renderTable(mount, d, st);
      var n = mount.querySelector(".pt-q"); n.focus(); n.setSelectionRange(p, p); };
    wireToggles(mount, d, function () { renderTable(mount, d, st); });
  }

  /* =================================================== D -- inspector ====== */
  function renderInspector(mount, d, st) {
    var rows = bySection(d, st.sec);
    if (!rows.some(function (r) { return r.k === st.focus; })) st.focus = rows.length ? rows[0].k : null;
    var foc = rows.filter(function (r) { return r.k === st.focus; })[0];
    var chips = d.sections.map(function (s) {
      return '<button class="chip ' + (s.id === st.sec ? "on" : "") + '" data-sec="' + s.id + '">'
        + esc(s.label) + "<b>" + bySection(d, s.id).length + "</b></button>";
    }).join("");
    var lastG = null;
    var list = rows.map(function (r) {
      var head = r.g !== lastG ? '<div class="ins-grp">' + esc(lastG = r.g) + "</div>" : "";
      return head + '<div class="ins-row ' + (r.k === st.focus ? "on" : "") + " " + (r.c ? "chg" : "")
        + '" data-k="' + esc(r.k) + '">' + dots(r) + '<span class="ins-lbl">' + esc(r.l) + "</span>"
        + '<span class="ins-v">' + esc(short(r.v)) + "</span></div>";
    }).join("");
    var meta = foc ? [["type", foc.t], ["default", foc.d == null ? "—" : short(foc.d)],
      (foc.mn != null ? ["range", foc.mn + " – " + foc.mx] : null),
      (foc.e ? ["choices", foc.e.join(" · ")] : null), ["group", foc.g], ["yaml", foc.k]]
      .filter(Boolean) : [];
    mount.innerHTML =
      '<div class="ctlbar"><div class="chips">' + chips + '</div><span class="sp"></span>'
      + '<span class="readout">↑ ↓ to walk the list</span></div>'
      + '<div class="ins"><div class="ins-list" tabindex="0">' + list + "</div>"
      + '<aside class="ins-doc">' + (!foc ? "" :
        '<div class="ins-kicker">setting</div><h4>' + esc(foc.l) + "</h4>"
        + '<div class="ins-path">' + esc(foc.k) + "</div>"
        + '<div class="ins-bigctl">' + ctl(foc) + '<button class="mini">↺ reset</button></div>'
        + '<p class="ins-help">' + esc(foc.h) + "</p>"
        + '<div class="ins-meta">' + meta.map(function (m) {
          return '<div class="ins-m"><span>' + esc(m[0]) + "</span><b>" + esc(m[1]) + "</b></div>";
        }).join("") + "</div>") + "</aside></div>";

    mount.querySelectorAll("[data-sec]").forEach(function (b) {
      b.onclick = function () { st.sec = b.dataset.sec; st.focus = null; renderInspector(mount, d, st); };
    });
    mount.querySelectorAll("[data-k]").forEach(function (a) {
      a.onclick = function () { st.focus = a.dataset.k; renderInspector(mount, d, st); };
    });
    var lst = mount.querySelector(".ins-list");
    lst.onkeydown = function (e) {
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
      e.preventDefault();
      var i = rows.findIndex(function (r) { return r.k === st.focus; });
      i = Math.max(0, Math.min(rows.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)));
      st.focus = rows[i].k;
      renderInspector(mount, d, st);
      var n = mount.querySelector(".ins-list");
      n.focus();
      var sel = n.querySelector(".ins-row.on");
      if (sel) sel.scrollIntoView({ block: "nearest" });
    };
    wireToggles(mount, d, function () { renderInspector(mount, d, st); });
  }

  /* =================================================== E -- workbench ====== */
  function renderWorkbench(mount, d, st) {
    var q = (st.q || "").trim().toLowerCase();
    var hits = !q ? [] : d.rows.filter(function (r) {
      return (r.k + " " + r.l + " " + r.h + " " + short(r.v)).toLowerCase().indexOf(q) >= 0;
    });
    var pinned = d.rows.filter(function (r) { return r.i; }).slice(0, 8);
    var colors = d.rows.filter(function (r) { return r.t === "color"; });
    var bools = d.rows.filter(function (r) { return r.t === "bool"; }).slice(0, 26);

    mount.innerHTML =
      '<div class="wb"><div class="wb-omni"><span class="mag">⌕</span>'
      + '<input class="wb-q" placeholder="search ' + d.rows.length
      + ' settings by name, path, help or current value…" value="' + esc(st.q || "") + '">'
      + '<span class="kbd">⌘K</span></div>'
      + (!q ? "" : '<div class="wb-hits">' + (hits.length ? hits.slice(0, 9).map(function (r) {
        return '<div class="wb-hit">' + dots(r) + '<span class="k">' + esc(r.l) + "</span>"
          + '<span class="p">' + esc(r.k) + "</span>" + ctl(r, "sm")
          + '<span class="g">' + esc(secLabel(d, r.s)) + "</span></div>";
      }).join("") + (hits.length > 9 ? '<div class="wb-more">+ ' + (hits.length - 9)
        + " more · ↵ to jump</div>" : "")
        : '<div class="wb-more">no match for “' + esc(st.q) + "”</div>") + "</div>")
      + '<div class="wb-sec">Pinned <span class="cnt">the ones you actually touch per run</span></div>'
      + '<div class="wb-board">' + pinned.map(function (r) {
        return '<div class="wb-card"><div class="wb-cl">' + dots(r) + esc(r.l) + "</div>" + ctl(r) + "</div>";
      }).join("") + "</div>"
      + (!colors.length ? "" :
        '<div class="wb-sec">Colors <span class="cnt">' + colors.length
        + " color settings — one swatch grid instead of " + colors.length + " form rows</span></div>"
        + '<div class="sw-grid">' + colors.map(function (r) {
          return '<span class="sw-tile" title="' + esc(r.k) + '"><i style="background:' + hexOf(r.v)
            + '"></i><span>' + esc(r.l) + "</span></span>";
        }).join("") + "</div>")
      + (!bools.length ? "" :
        '<div class="wb-sec">Toggles <span class="cnt">'
        + d.rows.filter(function (r) { return r.t === "bool"; }).length
        + " boolean settings — click to flip</span></div>"
        + '<div class="bmx-wrap">' + bools.map(function (r) {
          return '<span class="bmx ' + (r.v ? "on" : "") + '" data-tog="' + esc(r.k) + '" title="'
            + esc(r.k) + '">' + esc(r.l) + "</span>";
        }).join("") + "</div>")
      + "</div>";

    var i = mount.querySelector(".wb-q");
    i.oninput = function () {
      var p = i.selectionStart; st.q = i.value; renderWorkbench(mount, d, st);
      var n = mount.querySelector(".wb-q"); n.focus(); n.setSelectionRange(p, p);
    };
    wireToggles(mount, d, function () { renderWorkbench(mount, d, st); });
  }

  /* ================================================== 0 -- baseline ======== */
  function renderBaseline(mount, d, st) {
    var rows = bySection(d, "project").slice(0, 7);
    mount.innerHTML = '<div class="bl">'
      + '<div class="bl-group">▾ Project <span class="path">project</span>'
      + '<span class="cnt">7 important · 14</span></div>'
      + rows.map(function (r) {
        return '<div class="bl-row ' + (r.i ? "imp" : "") + '"><div class="bl-lbl">' + dots(r)
          + '<span class="k">' + esc(r.l) + "</span></div>"
          + '<div class="bl-ctl">' + ctl(r) + "</div>"
          + '<div class="bl-desc">' + esc(r.h) + "</div></div>";
      }).join("")
      + '<div class="bl-group closed">▸ Compute <span class="path">compute</span><span class="cnt">8 important · 17</span></div>'
      + '<div class="bl-group closed">▸ Detection <span class="path">modules</span><span class="cnt">19 important · 37</span></div>'
      + "</div>";
    wireToggles(mount, d, function () { renderBaseline(mount, d, st); });
  }

  /* --------------------------------------------------------------- toggles */
  function wireToggles(mount, d, redraw) {
    mount.querySelectorAll("[data-tog]").forEach(function (n) {
      n.onclick = function (e) {
        e.stopPropagation();
        var r = d.rows.filter(function (x) { return x.k === n.dataset.tog; })[0];
        if (!r) return;
        r.v = !r.v;
        redraw();
      };
    });
  }

  /* ------------------------------------------------------------------ boot */
  var R = {
    baseline: renderBaseline, columns: renderColumns, rail: renderRail,
    table: renderTable, inspector: renderInspector, workbench: renderWorkbench,
  };
  window.LM3MOCK = { render: function (mount) {
    var d = SRC[mount.dataset.src] || DATA;
    if (!mount._st) {
      mount._st = { sec: d.sections[0].id, grp: null, imp: false, desc: false,
        filter: "all", sort: "grp", dir: 1, q: "", focus: null };
      if (mount.dataset.layout === "rail" && mount.dataset.src === "settings") mount._st.sec = "compute";
    }
    R[mount.dataset.layout](mount, d, mount._st);
  } };
  document.querySelectorAll(".mount").forEach(function (m) { window.LM3MOCK.render(m); });
})();
