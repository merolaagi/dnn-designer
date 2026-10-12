"""Explainers: an answer as a page you can look at and play with.

The Explainer agent writes a small description of the page (a "spec"), never HTML. This module checks
it and draws it, so a model cannot put scripts or links of its own into a page your phone opens.
A page is made of sections, in order:

  text   a paragraph or two (Markdown; $…$ and $$…$$ for maths inline)
  math   an equation, typeset (LaTeX), with a caption
  steps  a worked solution, one step at a time, each with an optional equation
  plot   curves y = f(x), with sliders for the letters in them, so you can see what a parameter does
  graph  nodes and arrows, like Neo4j: drag them, and press Play to watch the steps light them up in turn
  table  a few rows of numbers

KaTeX (equations), math.js (evaluating the curves) and vis-network (the node graph) come from
jsDelivr when the page opens. Pages are kept in your workspace and listed under Files.
"""

from __future__ import annotations

import html
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import agentlab as AL

MAX_SECTIONS = 24
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,11}$")
EXPR = re.compile(r"^[0-9A-Za-z_+\-*/^().,\s|!<>=]{1,200}$")
FUNCS = {"sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "log", "log10", "log2", "sqrt",
         "abs", "floor", "ceil", "round", "min", "max", "pow", "sign", "pi", "e", "x", "mod", "cbrt"}


def _dir() -> Path:
    path = AL._dir() / "explainers"
    path.mkdir(exist_ok=True)
    return path


def _s(v: Any, n: int) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()[:n]


def _num(v: Any, name: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.")
    if f != f or abs(f) > 1e9:
        raise ValueError(f"{name} is out of range.")
    return f


def _expr(text: str, letters: List[str]) -> str:
    text = str(text or "").strip()
    if not EXPR.match(text):
        raise ValueError(f"The curve '{text[:40]}' uses characters a plot cannot take: write it like a*x^2 + b*sin(x).")
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        if word not in FUNCS and word not in letters:
            raise ValueError(f"The curve '{text[:40]}' uses '{word}': give it a slider (params), or use x and functions "
                             "such as sin, cos, exp, log, sqrt.")
    return text


def clean(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The spec, checked and trimmed. Raises ValueError with what to fix."""
    if not isinstance(spec, dict):
        raise ValueError("The spec is an object: {title, summary, sections: [...]}.")
    title = _s(spec.get("title"), 120)
    if len(title) < 2:
        raise ValueError("Give the page a title.")
    out = {"title": title, "summary": str(spec.get("summary") or "")[:1500], "sections": []}
    sections = spec.get("sections") or []
    if not isinstance(sections, list) or not sections:
        raise ValueError("Add at least one section (text, math, steps, plot, graph or table).")
    for i, s in enumerate(sections[:MAX_SECTIONS]):
        if not isinstance(s, dict):
            raise ValueError(f"Section {i + 1} is not an object.")
        kind = s.get("kind")
        head = _s(s.get("title"), 100)
        if kind == "text":
            out["sections"].append({"kind": kind, "title": head, "md": str(s.get("md") or "")[:5000]})
        elif kind == "math":
            out["sections"].append({"kind": kind, "title": head, "latex": str(s.get("latex") or "")[:2000],
                                    "caption": str(s.get("caption") or "")[:600]})
        elif kind == "steps":
            items = []
            for it in (s.get("items") or [])[:30]:
                items.append({"title": _s(it.get("title"), 100), "md": str(it.get("md") or "")[:2000],
                              "latex": str(it.get("latex") or "")[:1000]})
            if not items:
                raise ValueError(f"Section {i + 1} (steps) has no items.")
            out["sections"].append({"kind": kind, "title": head, "items": items})
        elif kind == "plot":
            params = []
            for p in (s.get("params") or [])[:6]:
                name = str(p.get("name") or "")
                if not NAME.match(name) or name in FUNCS:
                    raise ValueError(f"A slider is named like a, b, k or omega (not '{name[:20]}').")
                lo, hi = _num(p.get("min", -5), "min"), _num(p.get("max", 5), "max")
                if hi <= lo:
                    raise ValueError(f"Slider {name}: max must be more than min.")
                val = min(hi, max(lo, _num(p.get("value", (lo + hi) / 2), "value")))
                params.append({"name": name, "min": lo, "max": hi, "value": val,
                               "step": abs(_num(p.get("step", (hi - lo) / 100), "step")) or (hi - lo) / 100,
                               "label": _s(p.get("label"), 60)})
            letters = [p["name"] for p in params]
            funcs = []
            for f in (s.get("functions") or [])[:5]:
                funcs.append({"expr": _expr(f.get("expr"), letters), "label": _s(f.get("label"), 60)})
            if not funcs:
                raise ValueError(f"Section {i + 1} (plot) has no functions.")
            x = s.get("x") or [-10, 10]
            x0, x1 = _num(x[0], "x from"), _num(x[1], "x to")
            if x1 <= x0:
                raise ValueError("The plot's x range goes from smaller to larger.")
            y = s.get("y")
            yr = [_num(y[0], "y from"), _num(y[1], "y to")] if y else None
            points = [{"x": _num(pt.get("x"), "point x"), "y": _num(pt.get("y"), "point y"), "label": _s(pt.get("label"), 40)}
                      for pt in (s.get("points") or [])[:20]]
            out["sections"].append({"kind": kind, "title": head, "functions": funcs, "params": params, "x": [x0, x1],
                                    "y": yr, "points": points, "caption": str(s.get("caption") or "")[:600]})
        elif kind == "graph":
            nodes, ids = [], set()
            for n in (s.get("nodes") or [])[:80]:
                nid = _s(n.get("id"), 40)
                if not nid or nid in ids:
                    raise ValueError("Every node needs its own id.")
                ids.add(nid)
                nodes.append({"id": nid, "label": _s(n.get("label") or nid, 60), "group": _s(n.get("group"), 30),
                              "note": _s(n.get("note"), 200)})
            edges = []
            for e in (s.get("edges") or [])[:200]:
                a, b = _s(e.get("from"), 40), _s(e.get("to"), 40)
                if a not in ids or b not in ids:
                    raise ValueError(f"An arrow goes from '{a}' to '{b}', but both must be node ids.")
                edges.append({"from": a, "to": b, "label": _s(e.get("label"), 40)})
            steps = []
            for st in (s.get("steps") or [])[:40]:
                lit = [str(x) for x in (st.get("highlight") or []) if str(x) in ids]
                steps.append({"highlight": lit, "note": str(st.get("note") or "")[:400]})
            if not nodes:
                raise ValueError(f"Section {i + 1} (graph) has no nodes.")
            out["sections"].append({"kind": kind, "title": head, "nodes": nodes, "edges": edges, "steps": steps,
                                    "directed": bool(s.get("directed", True)), "caption": str(s.get("caption") or "")[:600]})
        elif kind == "table":
            cols = [_s(c, 40) for c in (s.get("columns") or [])[:8]]
            rows = [[_s(v, 60) for v in (r or [])[:8]] for r in (s.get("rows") or [])[:40]]
            out["sections"].append({"kind": kind, "title": head, "columns": cols, "rows": rows})
        else:
            raise ValueError(f"Section {i + 1}: kind is text, math, steps, plot, graph or table (not '{kind}').")
    return out


def save(spec: Dict[str, Any], question: str = "") -> Dict[str, Any]:
    page = clean(spec)
    eid = "ex" + uuid.uuid4().hex[:10]
    (_dir() / f"{eid}.json").write_text(json.dumps({**page, "question": str(question or "")[:500], "at": time.time()}))
    url = f"/api/explain/{eid}"
    try:
        import files
        files.record("explainer", page["title"], url=url, detail="An explainer you can play with", key=f"explain-{eid}")
    except Exception:  # noqa: BLE001
        pass
    return {"id": eid, "url": url, "title": page["title"]}


def load(eid: str) -> Dict[str, Any]:
    if not re.fullmatch(r"ex[0-9a-f]{10}", str(eid or "")):
        raise KeyError(eid)
    path = _dir() / f"{eid}.json"
    if not path.exists():
        raise KeyError(eid)
    return json.loads(path.read_text())


CDN = {
    "katex_css": "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css",
    "katex": "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js",
    "autorender": "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js",
    "mathjs": "https://cdn.jsdelivr.net/npm/mathjs@13.2.0/lib/browser/math.js",
    "vis": "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js",
}


def render(page: Dict[str, Any]) -> str:
    """The page. The spec goes in as JSON data; the drawing code is fixed here."""
    # every < > & in the data written as a JSON escape, so nothing in it can close or open a tag
    data = json.dumps(page).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(page['title'])}</title>
<link rel="stylesheet" href="{CDN['katex_css']}">
<script defer src="{CDN['katex']}"></script><script defer src="{CDN['autorender']}"></script>
<script defer src="{CDN['mathjs']}"></script><script defer src="{CDN['vis']}"></script>
<style>
:root{{--bg:#FAF7F2;--card:#fff;--ink:#1E2228;--muted:#6F7682;--line:#ECE6DC;--accent:#2563EB;--soft:#F2EEE7;--hi:#F2A93B}}
@media (prefers-color-scheme:dark){{:root{{--bg:#15171b;--card:#1e2126;--ink:#ECEDEF;--muted:#9aa1ab;--line:#2c3036;--soft:#262a30}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,system-ui,sans-serif}}
main{{max-width:760px;margin:0 auto;padding:18px 16px 60px}} h1{{font-size:24px;margin:4px 0 6px}} h2{{font-size:17px;margin:0 0 8px}}
.sum{{color:var(--muted);margin:0 0 16px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:12px 0;overflow-x:auto}}
.cap{{color:var(--muted);font-size:14px;margin-top:6px}} .math{{text-align:center;font-size:19px;padding:6px 0}}
.step{{display:none}} .step.on{{display:block}} .nav{{display:flex;gap:8px;align-items:center;margin-top:10px}}
button{{font:600 14px system-ui;border:1px solid var(--line);background:var(--soft);color:var(--ink);border-radius:10px;padding:7px 12px}}
button.go{{background:var(--accent);border-color:var(--accent);color:#fff}} canvas{{width:100%;height:280px;display:block}}
.sl{{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:14px}} .sl input{{flex:1}} .sl b{{min-width:56px;text-align:right}}
.net{{height:340px;border-radius:10px;background:var(--soft)}} .note{{min-height:22px;font-size:14.5px;margin-top:8px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}}
.legend{{font-size:13px;color:var(--muted);margin-top:4px}} code{{background:var(--soft);padding:1px 4px;border-radius:4px}}
</style></head><body><main id="m"></main>
<script id="spec" type="application/json">{data}</script>
<script>
const P = JSON.parse(document.getElementById("spec").textContent);
const esc = s => String(s || "").replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}})[c]);
const md = s => esc(s).replace(/\\*\\*(.+?)\\*\\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>")
  .split(/\\n{{2,}}/).map(p => "<p>" + p.replace(/\\n/g, "<br>") + "</p>").join("");
const COLORS = ["#2563EB", "#D9480F", "#2F8A57", "#9333EA", "#B7791F"];
const m = document.getElementById("m");
m.innerHTML = `<h1>${{esc(P.title)}}</h1>${{P.summary ? `<div class="sum">${{md(P.summary)}}</div>` : ""}}`;
const later = [];
P.sections.forEach((s, i) => {{
  const c = document.createElement("section"); c.className = "card"; m.appendChild(c);
  const h = s.title ? `<h2>${{esc(s.title)}}</h2>` : "";
  if (s.kind === "text") c.innerHTML = h + md(s.md);
  if (s.kind === "math") c.innerHTML = h + `<div class="math" data-tex="${{esc(s.latex)}}"></div>` + (s.caption ? `<div class="cap">${{md(s.caption)}}</div>` : "");
  if (s.kind === "table") c.innerHTML = h + `<table><tr>${{s.columns.map(x => `<th>${{esc(x)}}</th>`).join("")}}</tr>${{s.rows.map(r => `<tr>${{r.map(x => `<td>${{esc(x)}}</td>`).join("")}}</tr>`).join("")}}</table>`;
  if (s.kind === "steps") {{
    c.innerHTML = h + s.items.map((it, k) => `<div class="step${{k ? "" : " on"}}"><b>Step ${{k + 1}} of ${{s.items.length}}${{it.title ? ": " + esc(it.title) : ""}}</b>${{md(it.md)}}${{it.latex ? `<div class="math" data-tex="${{esc(it.latex)}}"></div>` : ""}}</div>`).join("")
      + `<div class="nav"><button data-a="prev">Back</button><button class="go" data-a="next">Next step</button><button data-a="all">Show all</button></div>`;
    let k = 0; const all = c.querySelectorAll(".step");
    const show = () => all.forEach((e, j) => e.classList.toggle("on", j === k));
    c.querySelector('[data-a="next"]').onclick = () => {{ k = Math.min(all.length - 1, k + 1); show(); }};
    c.querySelector('[data-a="prev"]').onclick = () => {{ k = Math.max(0, k - 1); show(); }};
    c.querySelector('[data-a="all"]').onclick = () => all.forEach(e => e.classList.add("on"));
  }}
  if (s.kind === "plot") {{
    c.innerHTML = h + `<canvas></canvas><div class="legend">${{s.functions.map((f, k) => `<span style="color:${{COLORS[k]}}">■</span> ${{esc(f.label || "y = " + f.expr)}}`).join(" &nbsp; ")}}</div>`
      + s.params.map(p => `<div class="sl"><span>${{esc(p.label || p.name)}}</span><input type="range" min="${{p.min}}" max="${{p.max}}" step="${{p.step}}" value="${{p.value}}" data-p="${{esc(p.name)}}"><b>${{p.value}}</b></div>`).join("")
      + (s.caption ? `<div class="cap">${{md(s.caption)}}</div>` : "");
    later.push(() => plot(c, s));
  }}
  if (s.kind === "graph") {{
    c.innerHTML = h + `<div class="net"></div><div class="note"></div>`
      + (s.steps.length ? `<div class="nav"><button class="go" data-a="play">Play</button><button data-a="prev">Back</button><button data-a="next">Next</button><span class="cap" data-a="n"></span></div>` : "")
      + (s.caption ? `<div class="cap">${{md(s.caption)}}</div>` : "");
    later.push(() => graph(c, s));
  }}
}});
function plot(c, s) {{
  const cv = c.querySelector("canvas"), ctx = cv.getContext("2d"), vals = {{}};
  s.params.forEach(p => vals[p.name] = p.value);
  const fns = s.functions.map(f => {{ try {{ return math.compile(f.expr); }} catch (e) {{ return null; }} }});
  const draw = () => {{
    const W = cv.width = cv.clientWidth * devicePixelRatio, H = cv.height = cv.clientHeight * devicePixelRatio;
    const [x0, x1] = s.x, N = 400, ys = [];
    const pts = fns.map(fn => {{ const a = []; for (let i = 0; i <= N; i++) {{ const x = x0 + (x1 - x0) * i / N; let y = NaN;
      try {{ y = fn ? Number(fn.evaluate({{...vals, x}})) : NaN; }} catch (e) {{}} a.push([x, y]); if (isFinite(y)) ys.push(y); }} return a; }});
    let [y0, y1] = s.y || [Math.min(...ys, 0), Math.max(...ys, 0)]; if (y1 - y0 < 1e-9) {{ y0 -= 1; y1 += 1; }}
    const pad = (y1 - y0) * 0.06; if (!s.y) {{ y0 -= pad; y1 += pad; }}
    const X = x => (x - x0) / (x1 - x0) * W, Y = y => H - (y - y0) / (y1 - y0) * H;
    const css = getComputedStyle(document.body); ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = css.getPropertyValue("--line"); ctx.lineWidth = devicePixelRatio;
    ctx.beginPath(); if (x0 < 0 && x1 > 0) {{ ctx.moveTo(X(0), 0); ctx.lineTo(X(0), H); }} if (y0 < 0 && y1 > 0) {{ ctx.moveTo(0, Y(0)); ctx.lineTo(W, Y(0)); }} ctx.stroke();
    ctx.fillStyle = css.getPropertyValue("--muted"); ctx.font = `${{11 * devicePixelRatio}}px system-ui`;
    ctx.fillText(x0.toPrecision(3), 4, H - 4); ctx.fillText(x1.toPrecision(3), W - 40 * devicePixelRatio, H - 4); ctx.fillText(y1.toPrecision(3), 4, 12 * devicePixelRatio);
    pts.forEach((a, k) => {{ ctx.strokeStyle = COLORS[k]; ctx.lineWidth = 2.2 * devicePixelRatio; ctx.beginPath(); let on = false;
      a.forEach(([x, y]) => {{ if (!isFinite(y) || y < y0 - (y1 - y0) || y > y1 + (y1 - y0)) {{ on = false; return; }} on ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); on = true; }}); ctx.stroke(); }});
    s.points.forEach(p => {{ ctx.fillStyle = "#D9480F"; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 4 * devicePixelRatio, 0, 7); ctx.fill();
      if (p.label) {{ ctx.fillStyle = css.getPropertyValue("--ink"); ctx.fillText(p.label, X(p.x) + 6, Y(p.y) - 6); }} }});
  }};
  c.querySelectorAll("input[type=range]").forEach(r => r.oninput = () => {{ vals[r.dataset.p] = Number(r.value); r.nextElementSibling.textContent = r.value; draw(); }});
  addEventListener("resize", draw); draw();
}}
function graph(c, s) {{
  const box = c.querySelector(".net"), note = c.querySelector(".note");
  const groups = [...new Set(s.nodes.map(n => n.group))];
  const base = n => COLORS[Math.max(0, groups.indexOf(n.group)) % COLORS.length];
  const nodes = new vis.DataSet(s.nodes.map(n => ({{id: n.id, label: n.label, title: n.note || n.label, color: {{background: base(n), border: base(n)}}, font: {{color: getComputedStyle(document.body).getPropertyValue("--ink").trim() || "#1E2228"}}, shape: "dot", size: 16}})));
  const edges = new vis.DataSet(s.edges.map((e, k) => ({{id: k, from: e.from, to: e.to, label: e.label, arrows: s.directed ? "to" : "", font: {{size: 12, align: "middle"}}, color: {{color: "#9aa1ab"}}}})));
  const net = new vis.Network(box, {{nodes, edges}}, {{physics: {{stabilization: true, barnesHut: {{springLength: 120}}}}, interaction: {{hover: true}},
    nodes: {{font: {{size: 14}}}}, edges: {{smooth: {{type: "dynamic"}}}}}});
  net.on("click", e => {{ const n = s.nodes.find(x => x.id === e.nodes[0]); if (n) note.innerHTML = "<b>" + esc(n.label) + "</b> " + esc(n.note || ""); }});
  if (!s.steps.length) return;
  let k = -1, timer = null; const counter = c.querySelector('[data-a="n"]');
  const show = () => {{ const st = s.steps[k] || {{highlight: [], note: ""}};
    nodes.update(s.nodes.map(n => ({{id: n.id, size: st.highlight.includes(n.id) ? 26 : 16,
      color: st.highlight.includes(n.id) ? {{background: "#F2A93B", border: "#B7791F"}} : {{background: base(n), border: base(n)}}}})));
    note.innerHTML = md(st.note); counter.textContent = k >= 0 ? `step ${{k + 1}} of ${{s.steps.length}}` : ""; }};
  c.querySelector('[data-a="next"]').onclick = () => {{ k = Math.min(s.steps.length - 1, k + 1); show(); }};
  c.querySelector('[data-a="prev"]').onclick = () => {{ k = Math.max(0, k - 1); show(); }};
  const play = c.querySelector('[data-a="play"]');
  play.onclick = () => {{ if (timer) {{ clearInterval(timer); timer = null; play.textContent = "Play"; return; }}
    k = -1; play.textContent = "Pause"; timer = setInterval(() => {{ k += 1; if (k >= s.steps.length) {{ clearInterval(timer); timer = null; play.textContent = "Play"; k = s.steps.length - 1; }} show(); }}, 1600); }};
}}
addEventListener("load", () => {{
  document.querySelectorAll("[data-tex]").forEach(e => {{ try {{ katex.render(e.dataset.tex, e, {{displayMode: true, throwOnError: false}}); }} catch (x) {{ e.textContent = e.dataset.tex; }} }});
  if (window.renderMathInElement) renderMathInElement(m, {{delimiters: [{{left: "$$", right: "$$", display: true}}, {{left: "$", right: "$", display: false}}], throwOnError: false}});
  later.forEach(f => {{ try {{ f(); }} catch (e) {{ console.error(e); }} }});
}});
</script></body></html>"""


GUIDE = (
    "You explain things to the person visually: maths, equations, problems, how a system or an algorithm works, "
    "relationships between things. Tools: run_python (check every number and every step of algebra before you "
    "show it) and make_explainer (builds the page from a spec, and returns its link).\n"
    "1. Work the problem out first, checking it with run_python (sympy for algebra and calculus, numbers for "
    "results). Never show a step or a value you did not check.\n"
    "2. Build the spec: {title, summary (two plain sentences), sections: [...]}. Sections, in a sensible order:\n"
    "   - {kind: 'text', title, md}: plain words; $…$ for maths inline.\n"
    "   - {kind: 'math', title, latex, caption}: one equation, LaTeX without $ signs.\n"
    "   - {kind: 'steps', title, items: [{title, md, latex}]}: a worked solution, one step per item.\n"
    "   - {kind: 'plot', title, functions: [{expr, label}], params: [{name, min, max, value, step, label}], x: [from, to], "
    "points: [{x, y, label}], caption}: curves in x, written like a*x^2 + b*sin(x) (math.js syntax); every other "
    "letter becomes a slider in params, so the person can see what it does.\n"
    "   - {kind: 'graph', title, nodes: [{id, label, group, note}], edges: [{from, to, label}], directed, steps: "
    "[{highlight: [node ids], note}], caption}: nodes and arrows they can drag, like Neo4j; steps light nodes up in "
    "turn when they press Play (an algorithm's order, a proof's chain, how data flows).\n"
    "   - {kind: 'table', title, columns, rows}.\n"
    "   Use a plot whenever a function or a parameter is involved, a graph whenever things connect, and steps for "
    "any calculation.\n"
    "3. Call make_explainer with the spec and the question. If it says what to fix, fix the spec and call it again.\n"
    "4. Answer briefly: what it shows, the result, and the link it returned, exactly as given, on its own line."
)
