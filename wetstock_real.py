"""Leak detection on real tank data, where nothing is known in advance.

Simulation can only say what my assumptions imply. Real gauge history has the
tank's actual chart error, its actual temperature behaviour and its actual
meters — and no ground truth: nobody records whether a tank was leaking. So
this module does three things, and is careful about which of them is evidence:

    where the station sits    overnight sales as a share of the average hour,
                              measured from the station's own sales. The
                              simulator said this decides which method works.

    an estimate per month     the overnight and hourly methods on each tank
                              and each 30-day window. An estimate, not a
                              verdict: without ground truth a high number may
                              be a leak, a chart error or a bookkeeping gap.

    the spike-in test         a synthetic leak of known size subtracted from
                              the real readings, then looked for. The method
                              is linear, so this is a test of one thing only:
                              whether the month's real noise is low enough
                              for a leak that size to cross the threshold. It
                              assumes the original month was tight; if it was
                              not, detection is overstated for that month.

Formats differ by gauge and by export, so columns are recognised by name and
the mapping is shown before anything is analysed. A guessed column is the
failure that would make every number afterwards meaningless.
"""

from __future__ import annotations

import csv
import io
import math
from typing import Any, Dict, List, Optional

import numpy as np

import wetstock as W

#: What each column is usually called. Matched case-insensitively against
#: whole words in the header, most specific first.
NAMES = {
    "time": ["timestamp", "date time", "datetime", "date/time", "time", "date"],
    "tank": ["tank number", "tank id", "tank no", "tank", "product"],
    "volume": ["tc volume", "tc vol", "net volume", "volume tc", "temp comp",
               "gross volume", "volume", "vol"],
    "temp": ["temperature", "temp", "product temp"],
    "sold": ["gallons sold", "volume sold", "sales gallons", "sold",
             "gallons", "quantity", "qty"],
    "delivered": ["delivered", "delivery gallons", "received", "receipt",
                  "gallons delivered", "net delivered", "gallons", "volume"],
}


class RealDataError(Exception):
    pass


def _rows(text: str) -> List[Dict[str, str]]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    rows = [r for r in reader if any((v or "").strip() for v in r.values())]
    if not rows:
        raise RealDataError("The file has a header and no rows.")
    return rows


def recognise(headers: List[str], wanted: List[str],
              override: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Match each wanted field to a header, or say it could not."""
    found: Dict[str, str] = {}
    lowered = {h: " ".join(h.lower().replace("_", " ").split()) for h in headers}
    for field in wanted:
        if override and override.get(field):
            if override[field] not in headers:
                raise RealDataError(f"No column called “{override[field]}”.")
            found[field] = override[field]
            continue
        for name in NAMES[field]:
            hit = [h for h, low in lowered.items()
                   if (low == name or f" {name} " in f" {low} ")
                   and h not in found.values()]
            if hit:
                found[field] = hit[0]
                break
    return found


def _when(text: str) -> float:
    """Hours since the epoch, from the formats gauges and tills commonly use."""
    from datetime import datetime

    text = text.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
                "%m/%d/%y %H:%M", "%m/%d/%Y %I:%M %p", "%m/%d/%Y %I:%M:%S %p",
                "%d/%m/%Y %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).timestamp() / 3600
        except ValueError:
            continue
    raise RealDataError(f"Could not read the time “{text}”. Times like "
                        f"2026-09-01 14:00 or 09/01/2026 2:00 PM are understood.")


def _number(text: str) -> float:
    cleaned = (text or "").replace(",", "").replace("gal", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def load(gauge: str, sales: str, deliveries: str = "",
         override: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, Any]:
    """Three exports in, one hourly record per tank out — and the mapping used."""
    override = override or {}
    gauge_rows = _rows(gauge)
    sales_rows = _rows(sales)
    delivery_rows = _rows(deliveries) if deliveries.strip() else []

    g_map = recognise(list(gauge_rows[0]), ["time", "tank", "volume", "temp"],
                      override.get("gauge"))
    s_map = recognise(list(sales_rows[0]), ["time", "tank", "sold"],
                      override.get("sales"))
    d_map = (recognise(list(delivery_rows[0]), ["time", "tank", "delivered"],
                       override.get("deliveries")) if delivery_rows else {})
    mapping_headers = {"gauge": list(gauge_rows[0]),
                       "sales": list(sales_rows[0])}
    for name, mapping, need in (("gauge", g_map, ["time", "volume"]),
                                ("sales", s_map, ["time", "sold"])):
        missing = [f for f in need if f not in mapping]
        if missing:
            # The page has no box for naming a column, so the fix has to be
            # something the person can do to the file: say which header
            # names are understood.
            def examples(field):
                names = NAMES[field]
                picked = names[:3] + ([field] if field in names else [])
                return ", ".join(f"“{n}”" for n in dict.fromkeys(picked))

            hints = "; ".join(f"{f}: {examples(f)}" for f in missing)
            raise RealDataError(
                f"In the {name} export I could not find a column for "
                f"{', '.join(missing)}. Its headers are "
                f"{', '.join(mapping_headers[name])}. Renaming the column to "
                f"one of these would work — {hints}.")

    def tank_of(row, mapping):
        return (row.get(mapping.get("tank", ""), "") or "1").strip() or "1"

    tanks: Dict[str, Dict[str, Any]] = {}
    for row in gauge_rows:
        t = _when(row[g_map["time"]])
        tank = tanks.setdefault(tank_of(row, g_map),
                                {"readings": [], "sales": [], "deliveries": []})
        tank["readings"].append((t, _number(row[g_map["volume"]]),
                                 _number(row[g_map["temp"]]) if "temp" in g_map
                                 else float("nan")))
    if not tanks:
        raise RealDataError("No gauge readings were read.")
    only = next(iter(tanks)) if len(tanks) == 1 else None
    for row in sales_rows:
        key = only or tank_of(row, s_map)
        if key in tanks:
            tanks[key]["sales"].append((_when(row[s_map["time"]]),
                                        _number(row[s_map["sold"]])))
    for row in delivery_rows:
        key = only or tank_of(row, d_map)
        if key in tanks:
            tanks[key]["deliveries"].append((_when(row[d_map["time"]]),
                                             _number(row[d_map["delivered"]])))

    hourly = {name: _hourly(name, data) for name, data in tanks.items()}
    return {"mapping": {"gauge": g_map, "sales": s_map, "deliveries": d_map},
            "tanks": hourly,
            "temperature_compensated": any(
                w in g_map["volume"].lower() for w in ("tc", "net", "comp"))}


def _hourly(name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Align everything on the hour: the last gauge reading in each hour,
    sales and deliveries summed within it. An hour with no reading is filled
    from its neighbours and counted, because a gap is not a leak."""
    readings = sorted(r for r in data["readings"] if math.isfinite(r[1]))
    if len(readings) < 48:
        raise RealDataError(f"Tank {name} has {len(readings)} readings; at "
                            f"least two days of hourly readings are needed.")
    start = math.floor(readings[0][0])
    end = math.floor(readings[-1][0])
    n = end - start + 1
    volume = np.full(n, np.nan)
    temp = np.full(n, np.nan)
    for t, v, tc in readings:
        i = int(math.floor(t)) - start
        volume[i], temp[i] = v, tc
    gaps = int(np.isnan(volume).sum())
    idx = np.arange(n)
    good = ~np.isnan(volume)
    volume = np.interp(idx, idx[good], volume[good])
    if np.isnan(temp).all():
        temp = np.full(n, 60.0)
    else:
        ok = ~np.isnan(temp)
        temp = np.interp(idx, idx[ok], temp[ok])
    sales = np.zeros(n)
    for t, g in data["sales"]:
        i = int(math.floor(t)) - start
        if 0 <= i < n and math.isfinite(g):
            sales[i] += g
    delivered = np.zeros(n)
    for t, g in data["deliveries"]:
        i = int(math.floor(t)) - start
        if 0 <= i < n and math.isfinite(g):
            delivered[i] += g
    from datetime import datetime

    hour = np.array([datetime.fromtimestamp((start + i) * 3600).hour
                     for i in range(n)], dtype=float)
    return {"gauge": volume, "temp": temp, "sales": sales,
            "delivery": delivered, "hour": hour, "start": start,
            "gaps": gaps, "hours": n}


# --------------------------------------------------------------------------
# what the data says
# --------------------------------------------------------------------------

def overnight_share(tank: Dict[str, Any]) -> float:
    """Sales per hour from 11 pm to 5 am, as a share of the average hour."""
    night = (tank["hour"] >= 23) | (tank["hour"] <= 5)
    average = tank["sales"].mean()
    return float(tank["sales"][night].mean() / average) if average > 0 else 0.0


def _window(tank, start: int, length: int, compensated: bool) -> Dict[str, Any]:
    piece = {k: tank[k][start:start + length]
             for k in ("gauge", "temp", "sales", "delivery", "hour")}
    if compensated:
        # already at 60 °F: tell the methods so, rather than compensating twice
        piece["temp"] = np.full(length, 60.0)
    return piece


def analyse(loaded: Dict[str, Any], window: int = W.HOURS,
            spikes=(0.1, 0.2)) -> Dict[str, Any]:
    compensated = loaded["temperature_compensated"]
    report = {"tanks": {}, "compensated": compensated}
    for name, tank in loaded["tanks"].items():
        share = overnight_share(tank)
        months = []
        for start in range(0, max(1, tank["hours"] - window + 1), window):
            if start + window > tank["hours"]:
                break
            piece = _window(tank, start, window, compensated)
            row = {"start_hour": start,
                   "overnight": _safe(W.quiet_hours, piece, window),
                   "hourly": _safe(W.hourly_balance, piece, window)}
            # The spike-in test: subtract a known leak from the real readings
            # and see whether it is flagged. The overnight estimator is linear
            # in the readings, so the spiked estimate is exactly the month's
            # own estimate plus the leak — reporting how much was "recovered"
            # would always say 100% and measure nothing. What this does
            # measure is whether the month's real noise is low enough for a
            # leak of that size to cross the threshold.
            caught = {}
            for rate in spikes:
                spiked = dict(piece)
                spiked["gauge"] = piece["gauge"] - rate * np.arange(window)
                estimate = _safe(W.quiet_hours, spiked, window)
                caught[str(rate)] = {
                    "estimate": estimate,
                    "flagged": estimate is not None and estimate >= 0.1,
                }
            row["spike"] = caught
            row["needs_a_look"] = (row["overnight"] is not None
                                   and abs(row["overnight"]) >= 0.1)
            months.append(row)
        partial = tank["hours"] < window
        report["tanks"][name] = {
            "hours": tank["hours"], "gaps": tank["gaps"],
            "overnight_share": share,
            "regime": ("quiet enough: the overnight method met the standard "
                       "in simulation at this level" if share <= 0.25 else
                       "busy overnight: in simulation the overnight method was "
                       "marginal here, and the tank chart was the limit"),
            "windows": months,
            "partial": partial,
        }
        if partial:
            report["tanks"][name]["note"] = (
                f"{tank['hours']} hours is less than a 30-day window, so no "
                f"monthly estimate is made. The overnight share is still "
                f"measured.")
    return report


def _safe(fn, piece, window):
    old = W.HOURS
    try:
        W.HOURS = window
        value = fn(piece)
        return None if value is None or not math.isfinite(value) else float(value)
    except Exception:  # noqa: BLE001
        return None
    finally:
        W.HOURS = old


# --------------------------------------------------------------------------
# a round trip through the format, for testing the importer honestly
# --------------------------------------------------------------------------

def to_csv(month: Dict[str, Any], start: str = "2026-09-01 00:00"
           ) -> Dict[str, str]:
    """Write a simulated month out as three exports, as a gauge and a till
    would, so the importer can be tested on a month whose leak is known."""
    from datetime import datetime, timedelta

    t0 = datetime.strptime(start, "%Y-%m-%d %H:%M")
    gauge = ["Date/Time,Tank,TC Volume,Temperature"]
    net = month["gauge"] / (1 + W.BETA * (month["temp"] - 60))
    for i, v in enumerate(net):
        when = (t0 + timedelta(hours=i, minutes=55)).strftime("%m/%d/%Y %H:%M")
        gauge.append(f"{when},1,{v:.1f},{month['temp'][i]:.1f}")
    sales = ["Timestamp,Tank,Gallons Sold"]
    for i, g in enumerate(month["sales"]):
        when = (t0 + timedelta(hours=i, minutes=30)).strftime("%Y-%m-%d %H:%M")
        sales.append(f"{when},1,{g:.3f}")
    deliveries = ["Date,Tank,Delivered"]
    for i, g in enumerate(month["delivery"]):
        if g:
            when = (t0 + timedelta(hours=i, minutes=20)).strftime("%Y-%m-%d %H:%M")
            deliveries.append(f"{when},1,{g:.1f}")
    return {"gauge": "\n".join(gauge), "sales": "\n".join(sales),
            "deliveries": "\n".join(deliveries)}
