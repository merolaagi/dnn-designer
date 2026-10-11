"""Station data: your c-store platform's database, read-only, for the lab's agents.

The agents read the database the platform already keeps (PostgreSQL), as a login that can only
read the operational tables (see station_reader.sql). Queries go through `psql`, so nothing extra
needs installing, and every query is checked here as well: one SELECT (or WITH … SELECT), no
second statement, at most MAX_ROWS rows back, given up after a few seconds.

The same functions serve three callers: the MCP server the Station analyst's tools come from
(station_mcp.py), the Settings page's Test button, and the ingest watchdog the scheduler runs.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
MAX_ROWS = 200
TIMEOUT = 15
DEFAULTS = {"host": "localhost", "port": 5432, "database": "cstore_platform", "user": "agent_reader",
            "password": "", "stale_hours": 36}


def _config_path() -> Path:
    path = HERE / "data"
    path.mkdir(exist_ok=True)
    return path / "station_data.json"


def config() -> Dict[str, Any]:
    try:
        saved = json.loads(_config_path().read_text())
    except (OSError, ValueError):
        saved = {}
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}


def view() -> Dict[str, Any]:
    """The settings for the page: the password only as whether one is saved."""
    c = config()
    return {**{k: v for k, v in c.items() if k != "password"}, "has_password": bool(c["password"]),
            "configured": _config_path().exists(), "psql": bool(psql_path())}


def save(change: Dict[str, Any]) -> Dict[str, Any]:
    c = config()
    for k in ("host", "database", "user"):
        if k in change and change[k] is not None:
            v = str(change[k]).strip()
            if not re.fullmatch(r"[\w.\-]+", v):
                raise ValueError(f"{k} may only hold letters, digits, dots, dashes and underscores.")
            c[k] = v
    if change.get("port") is not None:
        c["port"] = max(1, min(65535, int(change["port"])))
    if change.get("stale_hours") is not None:
        c["stale_hours"] = max(1, min(24 * 14, int(change["stale_hours"])))
    if change.get("password"):
        c["password"] = str(change["password"])
    path = _config_path()
    path.write_text(json.dumps(c, indent=1))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return view()


def psql_path() -> Optional[str]:
    for candidate in (shutil.which("psql"), "/opt/homebrew/bin/psql", "/usr/local/bin/psql",
                      "/opt/homebrew/opt/postgresql@16/bin/psql", "/usr/bin/psql"):
        if candidate and Path(candidate).exists():
            return candidate
    return None


# --------------------------------------------------------------------------
# running a query
# --------------------------------------------------------------------------

_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)
_STRINGS = re.compile(r"'(?:[^']|'')*'")
# the login is read-only whatever is sent; this only says why sooner, and catches writes hidden in a WITH
_WRITES = re.compile(r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|copy)\b", re.I)


MINE = "site_id IN (SELECT id FROM sites WHERE tenant_id = current_setting('app.current_tenant_id', true))"


def check_sql(sql: str, cfg: Optional[Dict[str, Any]] = None) -> str:
    """The query as it will run, or ValueError saying why it may not."""
    text = _COMMENTS.sub(" ", str(sql or "")).strip().rstrip(";").strip()
    bare = _STRINGS.sub("''", text)
    if not text:
        raise ValueError("Write a SELECT query.")
    if ";" in bare:
        raise ValueError("One statement at a time: remove the semicolon in the middle.")
    if not re.match(r"(?is)^(select|with)\b", bare):
        raise ValueError("Only SELECT queries (or WITH … SELECT) can run here.")
    found = _WRITES.search(bare)
    if found:
        raise ValueError(f"“{found.group(0)}” is not allowed: this connection only reads.")
    if not re.search(r"current_setting\s*\(\s*'app\.current_tenant_id'", text, re.I):
        named = [t for t in unprotected(cfg) if re.search(rf"\b{t}\b", bare, re.I)]
        if named:
            # these tables hold every tenant's rows and the database does not filter them: without this
            # filter a sum would include the other stores on the platform
            where = ("WHERE tenant_id = current_setting('app.current_tenant_id', true)" if named[0] == "sites"
                     else f"WHERE {MINE}")
            raise ValueError(f"{named[0]} holds every store's rows on the platform, not only yours: add "
                             f"{where} to the query (with AND if it already has a WHERE).")
    return text


_UNPROTECTED: Dict[str, Any] = {"key": None, "at": 0.0, "names": []}


def unprotected(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """Readable tables that row-level security does not limit to this login's tenant (cached ten minutes)."""
    cfg = cfg or config()
    key = (cfg.get("host"), cfg.get("port"), cfg.get("database"), cfg.get("user"))
    if _UNPROTECTED["key"] == key and time.time() - _UNPROTECTED["at"] < 600:
        return _UNPROTECTED["names"]
    try:
        out = _run("SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                   "WHERE n.nspname = 'public' AND c.relkind = 'r' AND NOT c.relrowsecurity AND c.relname IN "
                   "(SELECT table_name FROM information_schema.columns WHERE table_schema = 'public' "
                   "AND column_name IN ('site_id', 'tenant_id')) ORDER BY 1", cfg)
    except RuntimeError:
        return []                                  # not reachable: the query that follows will say so
    names = [r[0] for r in list(csv.reader(io.StringIO(out)))[1:] if r]
    _UNPROTECTED.update(key=key, at=time.time(), names=names)
    return names


def _run(sql: str, cfg: Optional[Dict[str, Any]] = None) -> str:
    """CSV text from psql, or RuntimeError with psql's own message."""
    cfg = cfg or config()
    exe = psql_path()
    if not exe:
        raise RuntimeError("psql was not found on this computer. Install PostgreSQL's client (brew install libpq).")
    env = {**os.environ, "PGPASSWORD": cfg.get("password") or "", "PGCONNECT_TIMEOUT": "5",
           "PGOPTIONS": "-c default_transaction_read_only=on -c statement_timeout=10000"}
    args = [exe, "-X", "-q", "--csv", "-v", "ON_ERROR_STOP=1", "-h", str(cfg["host"]), "-p", str(cfg["port"]),
            "-U", str(cfg["user"]), "-d", str(cfg["database"]), "-c", sql]
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"The query took more than {TIMEOUT} seconds.") from None
    if done.returncode != 0:
        # psql's message, not the caret line it prints under the query
        lines = [ln for ln in (done.stderr or done.stdout or "psql failed").strip().splitlines() if ln.strip()]
        said = next((ln for ln in lines if re.match(r"\s*(psql:.*)?(ERROR|FATAL)", ln)), lines[-1] if lines else "psql failed")
        hint = next((ln.strip() for ln in lines if ln.strip().startswith("HINT")), "")
        raise RuntimeError((said.strip() + (" " + hint if hint else ""))[:500])
    return done.stdout


def query(sql: str, limit: int = MAX_ROWS, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rows of a checked SELECT, at most `limit` of them."""
    text = check_sql(sql, cfg)
    limit = max(1, min(int(limit or MAX_ROWS), MAX_ROWS))
    out = _run(f"SELECT * FROM ({text}) AS q LIMIT {limit + 1}", cfg)
    rows = list(csv.reader(io.StringIO(out)))
    header, body = (rows[0], rows[1:]) if rows else ([], [])
    # psql writes a row of one NULL as an empty line, which csv reads as no fields: give every row its columns
    body = [r + [""] * (len(header) - len(r)) for r in body]
    more = len(body) > limit
    return {"columns": header, "rows": body[:limit], "more": more}


def as_text(result: Dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(result["columns"])
    w.writerows(result["rows"])
    note = f"\n[{len(result['rows'])} rows shown; there are more: narrow the query or aggregate.]" if result["more"] else \
        f"\n[{len(result['rows'])} row{'' if len(result['rows']) == 1 else 's'}]"
    return buf.getvalue().rstrip("\n") + note


def tables(cfg=None) -> List[Dict[str, Any]]:
    # information_schema lists only the columns this login may read
    out = query("SELECT table_name, count(*) AS columns FROM information_schema.columns "
                "WHERE table_schema = 'public' GROUP BY table_name ORDER BY table_name", cfg=cfg)
    return [{"table": r[0], "columns": int(r[1])} for r in out["rows"]]


def describe(table: str, cfg=None) -> Dict[str, Any]:
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", str(table or "")):
        raise ValueError("Name one table, such as day_cash_summary.")
    out = query("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' "
                f"AND table_name = '{table}' ORDER BY ordinal_position", cfg=cfg)
    if not out["rows"]:
        raise ValueError(f"There is no table {table} that this login can read. list_tables shows the ones it can.")
    return {"table": table, "columns": [{"name": r[0], "type": r[1]} for r in out["rows"]]}


def test(cfg=None) -> Dict[str, Any]:
    started = time.time()
    try:
        found = tables(cfg)
    except (RuntimeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    sites = []
    try:
        sites = query("SELECT name FROM sites WHERE tenant_id = current_setting('app.current_tenant_id', true) "
                      "ORDER BY name", cfg=cfg)["rows"]
    except (RuntimeError, ValueError):
        pass
    return {"ok": True, "tables": len(found), "sites": [r[0] for r in sites], "unfiltered": unprotected(cfg),
            "ms": round((time.time() - started) * 1000)}


# --------------------------------------------------------------------------
# a day's figures, the way the platform's Day Close page reads them
# --------------------------------------------------------------------------
# The platform's day_reports table stopped being written: for a Gilbarco site it builds each day from
# NAXML. The Day Close page reads day_cash_summary; gallons are the sum of fuel_grade_sales; and the
# platform's sales detail splits naxml_mcm_daily into fuel (department 1024) and the rest.

FUEL_DEPT = "1024"


def anchor(cfg=None) -> Tuple[str, str, str]:
    """(site id, site name, business day): the latest finished day of the tenant's busiest-recent site."""
    rows = query("SELECT c.site_id, s.name, max(c.business_date) FROM day_cash_summary c JOIN sites s ON s.id = c.site_id "
                 f"WHERE s.tenant_id = current_setting('app.current_tenant_id', true) AND c.{MINE} "
                 "AND c.business_date < current_date GROUP BY c.site_id, s.name ORDER BY 3 DESC, 2 LIMIT 1", cfg=cfg)["rows"]
    if not rows or not rows[0][2]:
        rows = query("SELECT c.site_id, s.name, max(c.business_date) FROM day_cash_summary c JOIN sites s ON s.id = c.site_id "
                     f"WHERE s.tenant_id = current_setting('app.current_tenant_id', true) AND c.{MINE} "
                     "GROUP BY c.site_id, s.name ORDER BY 3 DESC, 2 LIMIT 1", cfg=cfg)["rows"]
    if not rows or not rows[0][2]:
        raise ValueError("There is no day in day_cash_summary yet: no business day has been built.")
    return rows[0][0], rows[0][1], rows[0][2][:10]


def site_filter(site: str) -> str:
    return f"site_id = '{site.replace(chr(39), chr(39) * 2)}' AND {MINE}"


def days(site: str, start: str, end: str, cfg=None) -> List[Dict[str, Any]]:
    """Each business day from start to end: Day Close's figures, gallons, and the MCM split to cross-check."""
    mine = site_filter(site)
    rows = query("WITH c AS (SELECT business_date, sum(fuel_only_sales) AS fuel, sum(mdse_only_sales) AS nonfuel, "
                 "sum(tax_collected) AS tax, sum(sales_total) AS sales_total, sum(payouts_total) AS payouts, "
                 "sum(safe_drops_total) AS drops, sum(sales_transaction_count) AS txns FROM day_cash_summary "
                 f"WHERE {mine} AND business_date BETWEEN '{start}' AND '{end}' GROUP BY business_date), "
                 "g AS (SELECT business_date, sum(sales_volume) AS gallons FROM fuel_grade_sales "
                 f"WHERE {mine} AND business_date BETWEEN '{start}' AND '{end}' GROUP BY business_date), "
                 f"m AS (SELECT business_date, sum(CASE WHEN merchandise_code = '{FUEL_DEPT}' THEN sales_amount END) "
                 f"AS mcm_fuel, sum(CASE WHEN merchandise_code <> '{FUEL_DEPT}' THEN sales_amount END) AS mcm_nonfuel "
                 f"FROM naxml_mcm_daily WHERE {mine} AND business_date BETWEEN '{start}' AND '{end}' GROUP BY business_date) "
                 "SELECT c.business_date, c.fuel, c.nonfuel, c.tax, c.sales_total, c.payouts, c.drops, c.txns, g.gallons, "
                 "m.mcm_fuel, m.mcm_nonfuel FROM c LEFT JOIN g USING (business_date) LEFT JOIN m USING (business_date) "
                 "ORDER BY 1", cfg=cfg)["rows"]

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    out = []
    for r in rows:
        d = dict(zip(("date", "fuel", "nonfuel", "tax", "sales_total", "payouts", "drops", "txns", "gallons",
                      "mcm_fuel", "mcm_nonfuel"), [r[0][:10]] + [num(x) for x in r[1:]]))
        d["sales"] = round((d["fuel"] or 0) + (d["nonfuel"] or 0), 2) if d["fuel"] is not None or d["nonfuel"] is not None else None
        out.append(d)
    return out


# --------------------------------------------------------------------------
# the ingest watchdog
# --------------------------------------------------------------------------

WATCH_SQL = ("SELECT s.id, s.name, max(x.received_at) AS last_file, "
             "round((extract(epoch FROM now() - max(x.received_at)) / 3600)::numeric, 1) AS hours, "
             "(SELECT max(d.business_date) FROM day_cash_summary d WHERE d.site_id = s.id) AS last_day "
             "FROM sites s LEFT JOIN xml_files x ON x.site_id = s.id "
             "WHERE coalesce(s.is_active, true) "
             # the sites table has no row-level security of its own: keep to this login's tenant
             "AND s.tenant_id = current_setting('app.current_tenant_id', true) "
             "GROUP BY s.id, s.name ORDER BY s.name")


def watchdog(cfg=None) -> Dict[str, Any]:
    """Each active site: when its newest file arrived, and whether that is too long ago."""
    cfg = cfg or config()
    limit = float(cfg.get("stale_hours") or 36)
    rows = query(WATCH_SQL, cfg=cfg)["rows"]
    sites = []
    for sid, name, last_file, hours, last_day in rows:
        h = float(hours) if hours not in ("", None) else None
        sites.append({"id": sid, "name": name, "last_file": last_file or None, "hours": h,
                      "last_day": last_day or None, "stale": h is None or h >= limit})
    return {"limit": limit, "sites": sites, "stale": [s for s in sites if s["stale"]]}


def watchdog_news(result: Dict[str, Any], before: Dict[str, Any]) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:
    """What to tell the inbox, comparing with the last look: one alert per incident, one note on recovery."""
    news, state = [], {}
    for s in result["sites"]:
        was = (before or {}).get(s["id"], {}).get("stale", False)
        state[s["id"]] = {"stale": s["stale"], "since": (before or {}).get(s["id"], {}).get("since") if was else None}
        if s["stale"] and not was:
            state[s["id"]]["since"] = time.time()
            age = "no files at all" if s["hours"] is None else f"{s['hours']:.0f} hours since the last file ({s['last_file']})"
            news.append(("alert", f"{s['name']}: no new POS data for {result['limit']:.0f}+ hours",
                         f"{age}. Last business day built: {s['last_day'] or 'none'}. Check the store agent "
                         "(the Windows CStoreAgent task) and the XMLGateway share."))
        elif was and not s["stale"]:
            news.append(("recovered", f"{s['name']}: POS data is arriving again",
                         f"The newest file came in {s['hours']:.1f} hours ago."))
    return news, state


# --------------------------------------------------------------------------
# what the Station analyst is told about the data
# --------------------------------------------------------------------------

GUIDE = (
    "You answer questions about the user's gas station and convenience store from its own database, with the "
    "station tools: list_tables, describe_table and query (one read-only SELECT; at most 200 rows back, so "
    "aggregate in SQL). Describe a table before you query it if you are unsure of its columns. Report only "
    "numbers a query returned, and say which table and dates they come from.\n"
    "Where things are:\n"
    "- day_cash_summary: one row per site and business_date, the figures on the platform's Day Close page: "
    "fuel_only_sales (Fuel sales), mdse_only_sales (Non-fuel sales), tax_collected, sales_total, inside_sales, "
    "outside_fuel_sales, payouts_total, payins_total, safe_drops_total, safe_loans_total, refunds_total, "
    "sales_transaction_count. Use it for a day's sales.\n"
    "- Fuel gallons for a day: sum(sales_volume) from fuel_grade_sales (per business_date, grade_code, "
    "fuel_position_id and price tier: sum them). It also has sales_amount, discount_amount and dispenser_discount_amt.\n"
    "- day_reports and shift_reports are no longer written (their newest rows are old): do not use them.\n"
    "- day_mop_totals: sales by method of payment per day; day_cash_summary has the cash drawer totals.\n"
    "- fuel_inventory_daily: per business_date and tank_number: product_name, open_volume, close_volume, "
    "sales_volume, delivery_volume, ullage_gal, water_depth_inches, over_short_volume. fuel_tanks has capacity_gal.\n"
    "- Deliveries: fuel_inventory_daily.delivery_volume per tank and day is where they show up. fuel_deliveries "
    "(business_date, grade_code, gallons, price_per_gallon, supplier, bol_number) holds only those entered by hand.\n"
    "- fuel_price_changes: occurred_at, product_code, new_price. fuel_costs and dept_margins hold costs and margins.\n"
    "- naxml_mcm_daily: sales per business_date and merchandise_code (description, sales_quantity, sales_amount, "
    "refund_amount, discount_amount). Code 1024 (\"Fuel 1\") is fuel: leave it out when asked about merchandise.\n"
    "- department_totals (per shift: dept_name, net_sales, cost_of_goods), cashier_stats, "
    "tender_totals; lottery_* for lottery; purchase_orders, items, inventory_levels for stock.\n"
    "- xml_files (received_at) and agent_pings show whether the store's POS data is arriving.\n"
    "A few tables hold every store on the platform (the query tool says which, and gives the filter to add): "
    "sum only this store's rows.\n"
    "Days of fuel supply for a tank = its latest close_volume divided by its average daily sales_volume over the "
    "last 7 business days. Compare a day with the same weekday a week earlier, not the day before.\n"
    "Always answer in the language the user writes in.")


# the morning brief's task; a brief scheduled before 4.15 read day_reports and is moved onto this one
BRIEF_TASK = (
    "Write my morning brief for yesterday's business day. For each site: fuel sales, non-fuel sales and tax from "
    "day_cash_summary (the Day Close figures) and fuel gallons from fuel_grade_sales, each compared with the same "
    "weekday one week earlier (amount and percent); safe drops and pay-outs; each tank's latest close volume, how "
    "full it is against capacity, and its days of supply at the last 7 days' average sales (fuel_inventory_daily); "
    "fuel delivered into the tanks and any price changes; the five merchandise codes in naxml_mcm_daily (not code "
    "1024, which is fuel) whose sales rose or fell the most against the week before. Then list anything that needs "
    "my attention today. Use only numbers the queries return, keep it short, and say if yesterday's Day Close is "
    "missing.")
OLD_BRIEF = "Write my morning brief for yesterday's business day. For each site: total, inside and fuel net sales"
