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


def check_sql(sql: str) -> str:
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
    return text


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
        raise RuntimeError((done.stderr or done.stdout or "psql failed").strip().splitlines()[-1][:500])
    return done.stdout


def query(sql: str, limit: int = MAX_ROWS, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rows of a checked SELECT, at most `limit` of them."""
    text = check_sql(sql)
    limit = max(1, min(int(limit or MAX_ROWS), MAX_ROWS))
    out = _run(f"SELECT * FROM ({text}) AS q LIMIT {limit + 1}", cfg)
    rows = list(csv.reader(io.StringIO(out)))
    header, body = (rows[0], rows[1:]) if rows else ([], [])
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
        raise ValueError("Name one table, such as day_reports.")
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
    return {"ok": True, "tables": len(found), "sites": [r[0] for r in sites],
            "ms": round((time.time() - started) * 1000)}


# --------------------------------------------------------------------------
# the ingest watchdog
# --------------------------------------------------------------------------

WATCH_SQL = ("SELECT s.id, s.name, max(x.received_at) AS last_file, "
             "round((extract(epoch FROM now() - max(x.received_at)) / 3600)::numeric, 1) AS hours, "
             "(SELECT max(d.business_date) FROM day_reports d WHERE d.site_id = s.id) AS last_day "
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
    "- day_reports: one row per site and business_date: inside_net_sales, fuel_net_sales, fuel_volume_gallons, "
    "total_net_sales, cash_expected, cash_actual, cash_over_short, is_finalized.\n"
    "- fuel_grade_sales: per business_date and grade_code: sales_volume (gallons), sales_amount, discount_amount.\n"
    "- fuel_inventory_daily: per business_date and tank_number: product_name, open_volume, close_volume, "
    "sales_volume, delivery_volume, ullage_gal, water_depth_inches, over_short_volume. fuel_tanks has capacity_gal.\n"
    "- fuel_deliveries: business_date, grade_code, gallons, price_per_gallon, total_cost, supplier, bol_number.\n"
    "- fuel_price_changes: occurred_at, product_code, new_price. fuel_costs and dept_margins hold costs and margins.\n"
    "- naxml_mcm_daily: merchandise sales per business_date and merchandise_code (description, sales_quantity, "
    "sales_amount, refund_amount, discount_amount).\n"
    "- shift_reports, department_totals (per shift: dept_name, net_sales, cost_of_goods), cashier_stats, "
    "tender_totals; lottery_* for lottery; purchase_orders, items, inventory_levels for stock.\n"
    "- xml_files (received_at) and agent_pings show whether the store's POS data is arriving.\n"
    "Days of fuel supply for a tank = its latest close_volume divided by its average daily sales_volume over the "
    "last 7 business days. Compare a day with the same weekday a week earlier, not the day before.\n"
    "Always answer in the language the user writes in.")
