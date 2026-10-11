"""Travel: live flight prices from Google Flights, through SerpApi.

Google has no public flights API, and a web search only finds articles about fares, which is why a
plain search answers with last year's averages. SerpApi runs the real Google Flights search and
returns what it shows right now: each itinerary's price, airlines, stops, times, and Google's
own "prices are low/typical/high" judgement. Its free plan covers 250 searches a month.

The key is kept in data/travel.json on this Mac, with your home airport, which is where a trip
starts when you don't say. Nothing is ever booked: each answer links to the same search on
Google Flights, and you book there.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
DEFAULTS = {"serpapi_key": "", "home": "", "currency": "USD"}
ENDPOINT = "https://serpapi.com/search.json"
CABINS = {"economy": 1, "premium": 2, "premium economy": 2, "business": 3, "first": 4}
STOPS = {"any": 0, "nonstop": 1, "1": 2, "one": 2, "2": 3, "two": 3}
TIMEOUT = 60


class TravelError(RuntimeError):
    pass


def _path() -> Path:
    path = HERE / "data"
    path.mkdir(exist_ok=True)
    return path / "travel.json"


def config() -> Dict[str, Any]:
    try:
        saved = json.loads(_path().read_text())
    except (OSError, ValueError):
        saved = {}
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}


def view() -> Dict[str, Any]:
    c = config()
    return {"home": c["home"], "currency": c["currency"], "has_key": bool(c["serpapi_key"]),
            "configured": bool(c["serpapi_key"])}


def _codes(text: str) -> str:
    """'sfo, oak' -> 'SFO,OAK'. Airports are IATA codes; several are searched together."""
    codes = [x.strip().upper() for x in re.split(r"[,\s/]+", str(text or "")) if x.strip()]
    if not codes or not all(re.fullmatch(r"[A-Z]{3}", x) for x in codes):
        raise ValueError("Airports are three-letter codes, such as SFO or KTM (several with commas: SFO,OAK,SJC).")
    return ",".join(codes[:7])


def save(change: Dict[str, Any]) -> Dict[str, Any]:
    c = config()
    if change.get("serpapi_key"):
        key = str(change["serpapi_key"]).strip()
        if not re.fullmatch(r"[A-Za-z0-9]{32,80}", key):
            raise ValueError("A SerpApi key is a long string of letters and digits (from serpapi.com → Your account).")
        c["serpapi_key"] = key
    if change.get("home") is not None:
        c["home"] = _codes(change["home"]) if str(change["home"]).strip() else ""
    if change.get("currency"):
        cur = str(change["currency"]).strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", cur):
            raise ValueError("A currency is a three-letter code, such as USD.")
        c["currency"] = cur
    if change.get("forget"):
        c["serpapi_key"] = ""
    _path().write_text(json.dumps(c, indent=1))
    try:
        os.chmod(_path(), 0o600)
    except OSError:
        pass
    return view()


def _date(text: str, name: str) -> str:
    text = str(text or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError(f"The {name} date is YYYY-MM-DD, such as 2026-11-12.")
    if text < time.strftime("%Y-%m-%d"):
        raise ValueError(f"The {name} date {text} is in the past (today is {time.strftime('%Y-%m-%d')}).")
    return text


def _get(params: Dict[str, Any]) -> Dict[str, Any]:
    url = ENDPOINT + "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"accept": "application/json"}),
                                    timeout=TIMEOUT) as reply:
            data = json.loads(reply.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            why = json.loads(exc.read().decode()).get("error")
        except Exception:  # noqa: BLE001
            why = None
        raise TravelError(f"SerpApi said {exc.code}: {why or exc.reason}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TravelError(f"Could not reach SerpApi: {exc}") from None
    if data.get("error"):
        raise TravelError(f"SerpApi: {data['error']}")
    return data


def _leg(f: Dict[str, Any]) -> Dict[str, Any]:
    d, a = f.get("departure_airport") or {}, f.get("arrival_airport") or {}
    return {"from": d.get("id"), "to": a.get("id"), "departs": d.get("time"), "arrives": a.get("time"),
            "airline": f.get("airline"), "flight": f.get("flight_number"), "minutes": f.get("duration"),
            "overnight": bool(f.get("overnight"))}


def _option(o: Dict[str, Any], best: bool) -> Dict[str, Any]:
    legs = [_leg(f) for f in o.get("flights") or []]
    return {"price": o.get("price"), "best": best, "airlines": sorted({x["airline"] for x in legs if x["airline"]}),
            "stops": max(0, len(legs) - 1), "minutes": o.get("total_duration"),
            "layovers": [{"at": x.get("id"), "minutes": x.get("duration"), "overnight": bool(x.get("overnight"))}
                         for x in o.get("layovers") or []],
            "legs": legs}


def search_flights(origin: str = "", destination: str = "", depart: str = "", return_date: str = "", adults: int = 1,
                   cabin: str = "economy", stops: str = "any", c: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Google Flights' results right now: the cheapest options first, with Google's price judgement."""
    c = c or config()
    if not c["serpapi_key"]:
        raise TravelError("Flight search is not connected: add a SerpApi key first.")
    origin = _codes(origin or c["home"]) if (origin or c["home"]) else ""
    if not origin:
        raise ValueError("Where from? Give the airport (for example SFO), or set your home airport.")
    destination = _codes(destination)
    depart = _date(depart, "departure")
    if return_date:
        return_date = _date(return_date, "return")
        if return_date < depart:
            raise ValueError("The return date is before the departure date.")
    cab = CABINS.get(str(cabin or "economy").lower().strip(), 1)
    stop = STOPS.get(str(stops or "any").lower().strip(), 0)
    data = _get({"engine": "google_flights", "departure_id": origin, "arrival_id": destination, "outbound_date": depart,
                 "return_date": return_date or None, "type": 1 if return_date else 2,
                 "adults": max(1, min(9, int(adults or 1))), "travel_class": cab, "stops": stop or None,
                 "currency": c["currency"], "hl": "en", "gl": "us", "api_key": c["serpapi_key"]})
    options = [_option(o, True) for o in data.get("best_flights") or []] + \
              [_option(o, False) for o in data.get("other_flights") or []]
    options = [o for o in options if o["price"] is not None]
    options.sort(key=lambda o: o["price"])
    pi = data.get("price_insights") or {}
    return {"origin": origin, "destination": destination, "depart": depart, "return": return_date or None,
            "adults": max(1, min(9, int(adults or 1))), "cabin": cabin, "currency": c["currency"],
            "searched_at": time.strftime("%Y-%m-%d %H:%M"), "options": options[:12], "found": len(options),
            "insight": {"lowest": pi.get("lowest_price"), "level": pi.get("price_level"),
                        "typical": pi.get("typical_price_range")},
            "link": (data.get("search_metadata") or {}).get("google_flights_url")}


def _hm(minutes) -> str:
    if not minutes:
        return "?"
    return f"{int(minutes) // 60}h {int(minutes) % 60:02d}m"


def as_text(r: Dict[str, Any]) -> str:
    """What the travel agent's model reads: one line per option, the price judgement, and the link."""
    head = (f"Google Flights, searched {r['searched_at']}: {r['origin']} → {r['destination']}, {r['depart']}"
            + (f" returning {r['return']} (round trip, price is for the whole trip)" if r["return"] else " (one way)")
            + f", {r['adults']} adult(s), {r['cabin']}. {r['found']} options; cheapest first. Prices in {r['currency']}.")
    if not r["options"]:
        return head + "\nNo flights were found for these airports and dates."
    lines = [head]
    for i, o in enumerate(r["options"], 1):
        first, last = o["legs"][0], o["legs"][-1]
        via = ", ".join(f"{x['at']} {_hm(x['minutes'])}{' overnight' if x['overnight'] else ''}" for x in o["layovers"])
        lines.append(f"{i}. {r['currency']} {o['price']:,} · {' + '.join(o['airlines'])} · "
                     f"{'nonstop' if not o['stops'] else str(o['stops']) + ' stop' + ('s' if o['stops'] > 1 else '')}"
                     f"{' via ' + via if via else ''} · departs {first['departs']} from {first['from']}, arrives "
                     f"{last['arrives']} at {last['to']} · {_hm(o['minutes'])} outbound"
                     f"{' · Google marks it a best flight' if o['best'] else ''}")
    ins = r["insight"]
    if ins.get("level"):
        typ = ins.get("typical") or []
        lines.append(f"Google says prices are {ins['level']} now"
                     + (f" (typical {r['currency']} {typ[0]:,}–{typ[1]:,})" if len(typ) == 2 else "")
                     + (f"; lowest {r['currency']} {ins['lowest']:,}" if ins.get("lowest") else "") + ".")
    if r.get("link"):
        lines.append(f"The same search on Google Flights (to book): {r['link']}")
    return "\n".join(lines)


def test(c: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check the key with SerpApi's account endpoint (it does not use a search)."""
    c = c or config()
    if not c["serpapi_key"]:
        return {"ok": False, "error": "Add a SerpApi key first."}
    try:
        url = "https://serpapi.com/account.json?" + urllib.parse.urlencode({"api_key": c["serpapi_key"]})
        with urllib.request.urlopen(url, timeout=20) as reply:
            acct = json.loads(reply.read().decode())
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": "SerpApi did not accept that key." if exc.code in (401, 403) else f"SerpApi said {exc.code}."}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Could not reach SerpApi: {exc}"}
    if acct.get("error"):
        return {"ok": False, "error": acct["error"]}
    return {"ok": True, "plan": acct.get("plan_name"), "left": acct.get("total_searches_left")}


GUIDE = (
    "You find flights for the user with one tool, search_flights, which runs the real Google Flights search "
    "right now. Today's date is given below the request: use it. A date without a year is its next one.\n"
    "- Airports are three-letter codes; you may give several at once (SFO,OAK,SJC for the Bay Area). When the "
    "user does not say where from, leave origin empty: the tool uses their home airport, and tells you if none is "
    "set (then ask them).\n"
    "- To find a cheaper flight, also try nearby airports or a day either side when the user allows it, and say "
    "which search each price came from.\n"
    "- Answer with a short table of the best 3 to 5 options: price, airlines, stops (and where), departure and "
    "arrival times, total time. Then Google's price judgement (low, typical or high) and the Google Flights link "
    "to book. Say that prices change and are for the whole trip when it is a round trip.\n"
    "- Every price, time and airline you write comes from a search_flights result in this conversation. Never "
    "give fares from memory, from articles or as general ranges. If the tool fails, say what it said.\n"
    "- You cannot book or hold seats. The user books on Google Flights."
)
