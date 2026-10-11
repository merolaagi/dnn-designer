"""The store report: the latest business day at a glance, as a page and as a PDF.

The figures come straight from the store's database with SQL written here, read-only, the same
way the station benchmark works out its right answers; no model is involved, so the numbers are
the database's. The morning brief's text, which a model writes, goes underneath, labelled as such.
"""

from __future__ import annotations

import datetime as dt
import html as _html
import re
import time
from typing import Any, Dict, List, Optional

import minipdf

ACCENT = "#2a78d6"          # one series, one hue (the reference palette's first slot)
ACCENT_DARK = "#3987e5"
GOOD, CRITICAL, WARNING = "#0ca30c", "#d03b3b", "#fab219"
LOW_DAYS = 2.0              # a tank with less than this many days of supply is flagged


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def snapshot(cfg=None) -> Dict[str, Any]:
    """Everything the report shows, from the database, with what could not be read noted."""
    import station as ST

    def rows(sql):
        return ST.query(sql.replace("{mine}", mine), cfg=cfg)["rows"]

    pick = ST.query("SELECT d.site_id, s.name, max(d.business_date) FROM day_reports d JOIN sites s ON s.id = d.site_id "
                    f"WHERE s.tenant_id = current_setting('app.current_tenant_id', true) AND d.{ST.MINE} "
                    "GROUP BY d.site_id, s.name ORDER BY 3 DESC, 2 LIMIT 1", cfg=cfg)["rows"]
    if not pick or not pick[0][2]:
        raise ValueError("There are no day reports yet, so there is nothing to report.")
    site, name, last = pick[0]
    mine = f"site_id = '{site.replace(chr(39), chr(39) * 2)}' AND {ST.MINE}"
    fin = rows("SELECT max(business_date) FROM day_reports WHERE {mine} AND business_date < current_date")
    day = dt.date.fromisoformat(((fin[0][0] if fin and fin[0][0] else "") or last)[:10])
    D, W = day.isoformat(), (day - dt.timedelta(days=7)).isoformat()
    snap: Dict[str, Any] = {"site": name, "day": D, "week_before": W, "made": time.time(), "notes": [], "sections": []}

    def section(key, fn):
        try:
            snap[key] = fn()
            snap["sections"].append(key)
        except (RuntimeError, ValueError) as exc:
            snap["notes"].append(f"{key.replace('_', ' ')}: could not be read ({exc})")

    def days():
        out = []
        for r in rows("SELECT business_date, fuel_volume_gallons, fuel_net_sales, inside_net_sales, total_net_sales, "
                      "cash_over_short FROM day_reports WHERE {mine} AND business_date > '" + D + "'::date - 14 "
                      "AND business_date <= '" + D + "' ORDER BY business_date"):
            out.append({"date": r[0][:10], "gallons": _f(r[1]), "fuel": _f(r[2]), "inside": _f(r[3]), "total": _f(r[4]),
                        "cash": _f(r[5])})
        return out
    section("days", days)
    by = {d["date"]: d for d in snap.get("days") or []}
    snap["today"], snap["last_week"] = by.get(D), by.get(W)

    def tanks():
        t = rows("WITH last AS (SELECT tank_number, max(product_name) AS product, sum(close_volume) AS close "
                 "FROM fuel_inventory_daily WHERE {mine} AND business_date = '" + D + "' GROUP BY tank_number), "
                 "rate AS (SELECT tank_number, sum(sales_volume) / nullif(count(DISTINCT business_date), 0) AS per_day "
                 "FROM fuel_inventory_daily WHERE {mine} AND business_date > '" + D + "'::date - 7 AND business_date <= '"
                 + D + "' GROUP BY tank_number), cap AS (SELECT tank_number, max(capacity_gal) AS capacity FROM fuel_tanks "
                 "WHERE {mine} GROUP BY tank_number) SELECT l.tank_number, l.product, l.close, c.capacity, r.per_day "
                 "FROM last l LEFT JOIN rate r USING (tank_number) LEFT JOIN cap c USING (tank_number) ORDER BY 1")
        out = []
        for n, product, close, cap, per in t:
            close, cap, per = _f(close), _f(cap), _f(per)
            out.append({"tank": n, "product": product, "close": close, "capacity": cap,
                        "full": close / cap if close is not None and cap else None,
                        "days": close / per if close is not None and per else None})
        return out
    section("tanks", tanks)

    def merch():
        r = rows("SELECT coalesce(nullif(description, ''), merchandise_code), sum(sales_amount), sum(sales_quantity) "
                 "FROM naxml_mcm_daily WHERE {mine} AND business_date = '" + D + "' AND coalesce(description, '') !~* "
                 "'(\\mfuel\\M|diesel|unleaded|gasoline)' GROUP BY 1 ORDER BY 2 DESC LIMIT 6")
        return [{"name": a.strip(), "sales": _f(b), "qty": _f(c)} for a, b, c in r]
    section("merch", merch)

    def deliveries():
        r = rows("SELECT business_date, coalesce(grade_name, grade_code), gallons, supplier, bol_number FROM fuel_deliveries "
                 "WHERE {mine} AND business_date <= '" + D + "' ORDER BY business_date DESC LIMIT 5")
        return [{"date": a[:10], "grade": b, "gallons": _f(c), "supplier": d or "", "bol": e or ""} for a, b, c, d, e in r]
    section("deliveries", deliveries)

    def freshness():
        r = rows("SELECT max(received_at) FROM xml_files WHERE {mine}")
        last_file = r[0][0] if r and r[0] and r[0][0] else None
        age = (dt.date.today() - day).days
        return {"last_file": last_file, "day_age": age}
    section("freshness", freshness)
    fr = snap.get("freshness") or {}
    if fr.get("day_age", 0) > 2:
        snap["notes"].insert(0, f"The newest day report is {D}, {fr['day_age']} days ago: these are not yesterday's figures.")
    if snap.get("today") and any(d["days"] is not None and d["days"] < LOW_DAYS for d in snap.get("tanks") or []):
        snap["low_tanks"] = [d for d in snap["tanks"] if d["days"] is not None and d["days"] < LOW_DAYS]
    return snap


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def money(v: Optional[float], cents: bool = False) -> str:
    if v is None:
        return "–"
    return ("-" if v < 0 else "") + "$" + (f"{abs(v):,.2f}" if cents else f"{abs(v):,.0f}")


def gal(v: Optional[float]) -> str:
    return "–" if v is None else f"{v:,.0f}"


def change(now: Optional[float], before: Optional[float]) -> Optional[float]:
    if now is None or not before:
        return None
    return (now - before) / abs(before) * 100


def _pretty_day(iso: str) -> str:
    d = dt.date.fromisoformat(iso)
    return f"{d.strftime('%A')}, {d.strftime('%B')} {d.day}, {d.year}"


def tiles(snap: Dict[str, Any]) -> List[Dict[str, Any]]:
    t, w = snap.get("today") or {}, snap.get("last_week") or {}
    out = []
    for key, label, fmt in (("total", "Total net sales", money), ("inside", "Inside sales", money),
                            ("fuel", "Fuel sales", money), ("gallons", "Fuel gallons", gal)):
        out.append({"label": label, "value": fmt(t.get(key)), "delta": change(t.get(key), w.get(key)),
                    "before": fmt(w.get(key))})
    cash = t.get("cash")
    out.append({"label": "Cash over/short", "value": money(abs(cash), True) if cash is not None else "–",
                "delta": None, "before": None,
                "status": None if cash is None else ("over" if cash > 0 else "short" if cash < 0 else "even")})
    return out


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

def _e(s) -> str:
    return _html.escape(str(s if s is not None else ""))


def _bars(days: List[Dict[str, Any]], key: str, fmt, label: str, highlight: str) -> str:
    """One series of daily bars with a hover title on each, rounded data ends on the baseline."""
    vals = [(d["date"], d.get(key)) for d in days]
    top = max([v for _, v in vals if v is not None] or [1]) or 1
    w, h, pad_l, pad_b = 640, 180, 70, 26
    n = max(len(vals), 1)
    step = (w - pad_l - 6) / n
    bw = max(4.0, min(28.0, step - 6))
    grid = "".join(f'<line x1="{pad_l}" x2="{w}" y1="{h - pad_b - (h - pad_b - 10) * k / 3:.1f}" '
                   f'y2="{h - pad_b - (h - pad_b - 10) * k / 3:.1f}" class="g"/>'
                   f'<text x="{pad_l - 6}" y="{h - pad_b - (h - pad_b - 10) * k / 3 + 4:.1f}" class="ax" text-anchor="end">'
                   f'{_e(fmt(top * k / 3))}</text>' for k in range(1, 4))
    marks = []
    for i, (date, v) in enumerate(vals):
        x = pad_l + i * step + (step - bw) / 2
        if v is not None and v > 0:
            bh = (h - pad_b - 10) * v / top
            y = h - pad_b - bh
            r = min(4, bw / 2, bh)
            path = (f"M{x:.1f},{h - pad_b} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + bw - r:.1f} "
                    f"Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{h - pad_b} Z")
            marks.append(f'<g class="bar{" on" if date == highlight else ""}"><title>{_e(_pretty_day(date))}: {_e(fmt(v))}</title>'
                         f'<rect x="{x - 3:.1f}" y="10" width="{bw + 6:.1f}" height="{h - pad_b - 10}" class="hit"/>'
                         f'<path d="{path}"/></g>')
        if i == 0 or i == len(vals) - 1 or date == highlight or n <= 8:
            marks.append(f'<text x="{x + bw / 2:.1f}" y="{h - 6}" class="ax" text-anchor="middle">'
                         f'{_e(dt.date.fromisoformat(date).strftime("%b %-d"))}</text>')
    return (f'<figure><figcaption>{_e(label)}, last {len(vals)} business days</figcaption>'
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{_e(label)} by day">{grid}'
            f'<line x1="{pad_l}" x2="{w}" y1="{h - pad_b}" y2="{h - pad_b}" class="base"/>{"".join(marks)}</svg></figure>')


def _md_html(text: str) -> str:
    out, lines, i = [], str(text or "").replace("\r", "").split("\n"), 0
    inline = lambda t: re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", re.sub(r"`([^`]+)`", r"<code>\1</code>", _e(t)))
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            head = [c.strip() for c in ln.strip().strip("|").split("|")]
            i += 2
            body = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                body.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append("<table><tr>" + "".join(f"<th>{inline(h)}</th>" for h in head) + "</tr>" + "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body) + "</table>")
            continue
        if re.match(r"^#{1,4}\s", ln):
            out.append(f"<h4>{inline(ln.lstrip('# '))}</h4>")
        elif re.match(r"^\s*([-*•]|\d+[.)])\s+", ln):
            items = []
            while i < len(lines) and re.match(r"^\s*([-*•]|\d+[.)])\s+", lines[i]):
                items.append(re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", lines[i]))
                i += 1
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
            continue
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return "".join(out)


def html(snap: Dict[str, Any], brief: Optional[Dict[str, Any]] = None) -> str:
    def delta(t):
        if t["delta"] is None:
            return f'<span class="muted">{"vs " + _e(t["before"]) + " a week before" if t.get("before") not in (None, "–") else "no figure a week before" if t.get("status") is None else ""}</span>'
        up = t["delta"] >= 0
        return (f'<span class="{"up" if up else "down"}">{"▲" if up else "▼"} {abs(t["delta"]):.1f}%</span>'
                f' <span class="muted">vs {_e(t["before"])} a week before</span>')
    tile_html = "".join(
        f'<div class="tile"><div class="lab">{_e(t["label"])}</div><div class="val">{_e(t["value"])}'
        f'{" <small>" + _e(t["status"]) + "</small>" if t.get("status") else ""}</div><div class="d">{delta(t) if not t.get("status") else ""}</div></div>'
        for t in tiles(snap)) if snap.get("today") else '<p class="note">There is no day report for this day.</p>'
    days = snap.get("days") or []
    charts = (_bars(days, "gallons", gal, "Fuel gallons", snap["day"]) + _bars(days, "inside", money, "Inside sales", snap["day"])) if days else ""
    def tank_row(t):
        full = "" if t["full"] is None else f"{t['full']:.0%}"
        width = 0 if t["full"] is None else min(100, t["full"] * 100)
        supply = "–" if t["days"] is None else f"{t['days']:.1f}"
        low = " <span class=low>● low</span>" if t["days"] is not None and t["days"] < LOW_DAYS else ""
        return (f'<tr><td>{_e(t["tank"])}</td><td>{_e(t["product"])}</td><td class="n">{gal(t["close"])}</td>'
                f'<td><div class="meter" title="{full} full"><i style="width:{width:.0f}%"></i></div>'
                f'<span class="muted">{full}</span></td><td class="n">{supply}{low}</td></tr>')
    tank_rows = "".join(tank_row(t) for t in snap.get("tanks") or [])

    def merch_row(m):
        qty = "" if m["qty"] is None else f"{m['qty']:,.0f}"
        return f'<tr><td>{_e(m["name"])}</td><td class="n">{money(m["sales"], True)}</td><td class="n">{qty}</td></tr>'
    merch_rows = "".join(merch_row(m) for m in snap.get("merch") or [])
    dl_rows = "".join(f'<tr><td>{_e(d["date"])}</td><td>{_e(d["grade"])}</td><td class="n">{gal(d["gallons"])}</td>'
                      f'<td>{_e(d["supplier"])}</td><td>{_e(d["bol"])}</td></tr>' for d in snap.get("deliveries") or [])
    notes = "".join(f'<p class="note">{_e(n)}</p>' for n in snap.get("notes") or [])
    brief_html = ""
    if brief and brief.get("answer"):
        brief_html = (f'<section><h2>Morning brief</h2><p class="muted">Written by the Station analyst (a model) '
                      f'{_e(time.strftime("%b %-d, %-I:%M %p", time.localtime(brief.get("at") or time.time())))}; '
                      f'its numbers come from its own queries. The figures above come from the database directly.</p>'
                      f'<div class="brief">{_md_html(brief["answer"])}</div></section>')
    table = lambda head, body: (f'<table><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr>{body}</table>' if body
                                else '<p class="muted">Nothing recorded.</p>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(snap["site"])} · {_e(snap["day"])}</title>
<style>
:root{{color-scheme:light;--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--base:#c3c2b7;
  --series:{ACCENT};--good:{GOOD};--crit:{CRITICAL};--card:#ffffff}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#1a1a19;--ink:#f0efec;
  --ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--base:#383835;--series:{ACCENT_DARK};--card:#222220}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
  --base:#383835;--series:{ACCENT_DARK};--card:#222220}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
main{{max-width:880px;margin:0 auto;padding:24px 16px 48px}}
h1{{font-size:22px;margin:0}} h2{{font-size:16px;margin:28px 0 8px}}
.sub{{color:var(--ink2);margin:2px 0 16px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}
.tile{{background:var(--card);border:1px solid var(--grid);border-radius:12px;padding:12px}}
.lab{{color:var(--ink2);font-size:13px}} .val{{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums;margin:2px 0}}
.val small{{font-size:13px;font-weight:500;color:var(--ink2)}}
.d{{font-size:12.5px}} .up{{color:var(--ink)}} .down{{color:var(--ink)}} .muted{{color:var(--muted)}}
figure{{margin:16px 0 0}} figcaption{{color:var(--ink2);font-size:13px;margin-bottom:4px}}
svg{{width:100%;height:auto;display:block}}
svg .g{{stroke:var(--grid);stroke-width:1}} svg .base{{stroke:var(--base);stroke-width:1}}
svg .ax{{fill:var(--muted);font-size:11px}}
svg .bar path{{fill:var(--series);opacity:.55}} svg .bar.on path{{opacity:1}} svg .bar:hover path{{opacity:.9}}
svg .hit{{fill:transparent}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th{{text-align:left;color:var(--ink2);font-weight:600;border-bottom:1px solid var(--base);padding:6px 8px}}
td{{border-bottom:1px solid var(--grid);padding:6px 8px}} td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.meter{{display:inline-block;width:90px;height:8px;border-radius:4px;background:var(--grid);vertical-align:middle;margin-right:6px;overflow:hidden}}
.meter i{{display:block;height:100%;background:var(--series);border-radius:4px}}
.low{{color:var(--crit);font-weight:600;white-space:nowrap}}
.note{{background:color-mix(in srgb,{WARNING} 16%,transparent);border-radius:10px;padding:8px 12px;margin:8px 0}}
.brief{{background:var(--card);border:1px solid var(--grid);border-radius:12px;padding:4px 16px}}
.brief table{{font-size:13px;margin:8px 0}}
.tw{{overflow-x:auto}}
footer{{color:var(--muted);font-size:12px;margin-top:28px}}
@media (max-width:560px){{svg .ax{{font-size:20px}} .val{{font-size:21px}}}}
@media print{{body{{background:#fff}} main{{padding:0}} .tile,.brief{{break-inside:avoid}}}}
</style></head><body><main>
<h1>{_e(snap["site"])}</h1>
<p class="sub">{_e(_pretty_day(snap["day"]))} · compared with {_e(_pretty_day(snap["week_before"]))}</p>
{notes}
<div class="tiles">{tile_html}</div>
{charts}
<h2>Tanks</h2><div class="tw">{table(["Tank", "Product", "Gallons", "Full", "Days left"], tank_rows)}</div>
<p class="muted" style="font-size:12.5px">Gallons at the day's close. Days left: the close volume over the average daily sales of the last 7 business days.</p>
<h2>Top merchandise (fuel left out)</h2><div class="tw">{table(["Category", "Sales", "Items"], merch_rows)}</div>
<h2>Recent deliveries</h2><div class="tw">{table(["Date", "Grade", "Gallons", "Supplier", "BOL"], dl_rows)}</div>
{brief_html}
<footer>Read-only from the store's database at {_e(time.strftime("%b %-d, %Y %-I:%M %p", time.localtime(snap["made"])))}.
Every figure above is a query result; none is a model's.</footer>
</main></body></html>"""


# --------------------------------------------------------------------------
# the PDF
# --------------------------------------------------------------------------

def pdf(snap: Dict[str, Any], brief: Optional[Dict[str, Any]] = None) -> bytes:
    P = minipdf
    doc = P.Doc(f"{snap['site']} - {snap['day']}")
    doc.y -= 4
    doc.text(P.MARGIN, doc.y - 16, snap["site"], 18, True)
    doc.y -= 34
    doc.text(P.MARGIN, doc.y - 10, f"{_pretty_day(snap['day'])} - compared with {_pretty_day(snap['week_before'])}", 10,
             color="#52514e")
    doc.y -= 22
    for n in snap.get("notes") or []:
        doc.need(30)
        lines = P.wrap(n, 9.5, P.W - 2 * P.MARGIN - 20)
        doc.rect(P.MARGIN, doc.y - 8 - 13 * len(lines), P.W - 2 * P.MARGIN, 8 + 13 * len(lines), "#fdf0d2")
        for ln in lines:
            doc.y -= 13
            doc.text(P.MARGIN + 10, doc.y - 1, ln, 9.5)
        doc.y -= 14
    if snap.get("today"):
        ts = tiles(snap)
        tw = (P.W - 2 * P.MARGIN - 4 * 8) / 5
        doc.need(70)
        top = doc.y
        for k, t in enumerate(ts):
            x = P.MARGIN + k * (tw + 8)
            doc.rect(x, top - 62, tw, 62, "#f3f2ee")
            doc.text(x + 8, top - 16, t["label"], 8, color="#52514e")
            doc.text(x + 8, top - 36, t["value"] + (f" {t['status']}" if t.get("status") else ""), 13, True)
            if t["delta"] is not None:
                doc.text(x + 8, top - 52, f"{'+' if t['delta'] >= 0 else '-'}{abs(t['delta']):.1f}% vs {t['before']}", 7.5,
                         color="#52514e")
        doc.y = top - 78
    days = snap.get("days") or []
    for key, label, fmt in (("gallons", "Fuel gallons", gal), ("inside", "Inside sales", money)):
        vals = [(d["date"], d.get(key)) for d in days]
        if not any(v for _, v in vals):
            continue
        doc.need(130)
        doc.text(P.MARGIN, doc.y - 10, f"{label}, last {len(vals)} business days", 9, color="#52514e")
        base, height, left = doc.y - 110, 86, P.MARGIN + 44
        top_v = max(v for _, v in vals if v) or 1
        for k in range(1, 4):
            yy = base + height * k / 3
            doc.line(left, yy, P.W - P.MARGIN, yy, "#e1e0d9", 0.4)
            doc.text(left - 4, yy - 3, fmt(top_v * k / 3), 7, color="#898781", align="right")
        doc.line(left, base, P.W - P.MARGIN, base, "#c3c2b7", 0.6)
        step = (P.W - P.MARGIN - left) / max(len(vals), 1)
        bw = max(3, min(20, step - 4))
        for i, (date, v) in enumerate(vals):
            x = left + i * step + (step - bw) / 2
            if v and v > 0:
                doc.rect(x, base, bw, height * v / top_v, ACCENT if date == snap["day"] else "#9cc0ec")
            if i in (0, len(vals) - 1) or date == snap["day"]:
                doc.text(x + bw / 2, base - 10, dt.date.fromisoformat(date).strftime("%b %d"), 7, color="#898781",
                         align="center")
        doc.y = base - 24
    sections = [
        ("Tanks", ["Tank", "Product", "Gallons at close", "Full", "Days of supply"],
         [[str(t["tank"]), t["product"] or "", gal(t["close"]), "" if t["full"] is None else f"{t['full']:.0%}",
           ("–" if t["days"] is None else f"{t['days']:.1f}") + (" LOW" if t["days"] is not None and t["days"] < LOW_DAYS else "")]
          for t in snap.get("tanks") or []]),
        ("Top merchandise (fuel left out)", ["Category", "Sales", "Items"],
         [[m["name"], money(m["sales"], True), "" if m["qty"] is None else f"{m['qty']:,.0f}"] for m in snap.get("merch") or []]),
        ("Recent deliveries", ["Date", "Grade", "Gallons", "Supplier", "BOL"],
         [[d["date"], d["grade"] or "", gal(d["gallons"]), d["supplier"], d["bol"]] for d in snap.get("deliveries") or []]),
    ]
    for title, head, body in sections:
        doc.need(50)
        doc.y -= 8
        doc.para(title, 12, True, gap=2)
        if body:
            P.table(doc, [head] + body)
        else:
            doc.para("Nothing recorded.", 9.5, color="#898781")
    if brief and brief.get("answer"):
        doc.need(60)
        doc.y -= 10
        doc.para("Morning brief", 12, True, gap=1)
        doc.para("Written by the Station analyst (a model); its numbers come from its own queries. The figures above "
                 "come from the database directly.", 8.5, color="#898781", gap=6)
        P.markdown(doc, brief["answer"], 9.5)
    doc.y -= 10
    doc.para(f"Read-only from the store's database at {time.strftime('%b %d, %Y %I:%M %p', time.localtime(snap['made']))}. "
             "Every figure in the tables and charts is a query result.", 7.5, color="#898781")
    return doc.bytes()


def answer_pdf(title: str, question: str, answer: str, meta: str = "") -> bytes:
    """Any answer, saved as a PDF: the question, then the answer's Markdown."""
    P = minipdf
    doc = P.Doc(title)
    doc.para(title, 15, True, gap=2)
    if meta:
        doc.para(meta, 8.5, color="#898781", gap=8)
    doc.para(question, 10, color="#52514e", gap=10)
    P.markdown(doc, answer, 10)
    return doc.bytes()


def latest_brief() -> Optional[Dict[str, Any]]:
    """The newest finished morning brief: its answer and when."""
    import agentlab as AL
    import schedules
    for s in schedules.listing():
        if not s.get("conv") or not str(s.get("name", "")).lower().startswith("morning brief"):
            continue
        try:
            turns = AL.conversation(s["conv"])["turns"]
        except KeyError:
            continue
        done = [t for t in turns if t.get("status") == "done" and t.get("answer")]
        if done:
            return {"answer": done[-1]["answer"], "at": done[-1].get("at"), "conv": s["conv"]}
    return None
