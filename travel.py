"""Travel: live flight and hotel prices from Google, through SerpApi; booking through Duffel.

Google has no public flights API, and a web search only finds articles about fares, which is why a
plain search answers with last year's averages. SerpApi runs the real Google Flights search and
returns what it shows right now: each itinerary's price, airlines, stops, times, and Google's
own "prices are low/typical/high" judgement. Its free plan covers 250 searches a month.

The same key searches Google Hotels.

Booking is separate and optional: Duffel (duffel.com) sells real airline tickets through an API and
is paid from your Duffel balance, so no card number passes through this Mac. A booking is a
CHANGES tool: the agent stops and shows you the exact flight, price and passenger, and nothing is
bought until you approve. A test token (duffel_test_…) books Duffel's pretend airline, for trying
it; a live token (duffel_live_…) buys real tickets.

Keys, your home airport and the traveller's details (name as on the passport, date of birth, email,
phone) are kept in data/travel.json on this Mac, readable only by you.
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
DEFAULTS = {"serpapi_key": "", "home": "", "currency": "USD", "duffel_token": "", "traveler": {}}
DUFFEL = "https://api.duffel.com"
TRAVELER_FIELDS = ("title", "given_name", "family_name", "born_on", "gender", "email", "phone_number")
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
    tok = c["duffel_token"]
    return {"home": c["home"], "currency": c["currency"], "has_key": bool(c["serpapi_key"]),
            "configured": bool(c["serpapi_key"]), "has_duffel": bool(tok),
            "booking": "live" if tok.startswith("duffel_live_") else "test" if tok else None,
            "traveler": dict(c["traveler"] or {}), "traveler_ready": not traveler_missing(c)}


def traveler_missing(c: Optional[Dict[str, Any]] = None) -> List[str]:
    t = (c or config())["traveler"] or {}
    return [f for f in TRAVELER_FIELDS if not t.get(f)]


def _traveler(raw: Dict[str, Any]) -> Dict[str, Any]:
    t = {k: str(raw.get(k) or "").strip() for k in TRAVELER_FIELDS}
    if t["title"]:
        t["title"] = t["title"].lower().rstrip(".")
        if t["title"] not in ("mr", "ms", "mrs", "miss", "dr"):
            raise ValueError("Title is one of Mr, Ms, Mrs, Miss or Dr.")
    for k in ("given_name", "family_name"):
        if t[k] and not re.fullmatch(r"[A-Za-z][A-Za-z '\-]{0,40}", t[k]):
            raise ValueError("Names are written as on the passport, in Latin letters.")
    if t["born_on"] and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", t["born_on"]):
        raise ValueError("Date of birth is YYYY-MM-DD.")
    if t["gender"]:
        t["gender"] = t["gender"][0].lower()
        if t["gender"] not in ("m", "f"):
            raise ValueError("Airlines take gender as m or f, as on the passport.")
    if t["email"] and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", t["email"]):
        raise ValueError("That does not look like an email address.")
    if t["phone_number"]:
        t["phone_number"] = re.sub(r"[\s().\-]", "", t["phone_number"])
        if not t["phone_number"].startswith("+"):
            t["phone_number"] = "+1" + t["phone_number"] if len(t["phone_number"]) == 10 else "+" + t["phone_number"]
        if not re.fullmatch(r"\+\d{8,15}", t["phone_number"]):
            raise ValueError("Phone is in international form, such as +15105551234.")
    return t


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
    if change.get("duffel_token"):
        tok = str(change["duffel_token"]).strip()
        if not re.fullmatch(r"duffel_(test|live)_[\w\-]{10,200}", tok):
            raise ValueError("A Duffel access token starts with duffel_test_ or duffel_live_ (Duffel → Developers → Access tokens).")
        c["duffel_token"] = tok
    if change.get("traveler") is not None:
        c["traveler"] = _traveler(change["traveler"] or {})
    if change.get("forget_duffel"):
        c["duffel_token"] = ""
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


# --------------------------------------------------------------------------
# hotels (Google Hotels, same SerpApi key)
# --------------------------------------------------------------------------

def search_hotels(place: str, check_in: str, check_out: str, adults: int = 2, max_price: Optional[int] = None,
                  c: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    c = c or config()
    if not c["serpapi_key"]:
        raise TravelError("Hotel search is not connected: add a SerpApi key first.")
    place = re.sub(r"\s+", " ", str(place or "")).strip()
    if len(place) < 2:
        raise ValueError("Say where: a city, an area or a hotel's name.")
    check_in, check_out = _date(check_in, "check-in"), _date(check_out, "check-out")
    if check_out <= check_in:
        raise ValueError("Check-out is after check-in.")
    data = _get({"engine": "google_hotels", "q": place, "check_in_date": check_in, "check_out_date": check_out,
                 "adults": max(1, min(9, int(adults or 2))), "currency": c["currency"], "hl": "en", "gl": "us",
                 "max_price": max_price or None, "sort_by": 3, "api_key": c["serpapi_key"]})
    out = []
    for h in data.get("properties") or []:
        night = (h.get("rate_per_night") or {}).get("extracted_lowest")
        if night is None:
            continue
        out.append({"name": h.get("name"), "night": night, "total": (h.get("total_rate") or {}).get("extracted_lowest"),
                    "rating": h.get("overall_rating"), "reviews": h.get("reviews"), "stars": h.get("extracted_hotel_class"),
                    "link": h.get("link"), "amenities": (h.get("amenities") or [])[:6], "type": h.get("type")})
    return {"place": place, "check_in": check_in, "check_out": check_out, "adults": max(1, min(9, int(adults or 2))),
            "currency": c["currency"], "searched_at": time.strftime("%Y-%m-%d %H:%M"), "hotels": out[:10],
            "link": (data.get("search_metadata") or {}).get("google_hotels_url")}


def hotels_text(r: Dict[str, Any]) -> str:
    head = (f"Google Hotels, searched {r['searched_at']}: {r['place']}, {r['check_in']} to {r['check_out']}, "
            f"{r['adults']} adult(s). Prices in {r['currency']}, cheapest first.")
    if not r["hotels"]:
        return head + "\nNo hotels with prices were found."
    lines = [head]
    for i, h in enumerate(r["hotels"], 1):
        lines.append(f"{i}. {h['name']} · {r['currency']} {h['night']:,}/night"
                     + (f" ({r['currency']} {h['total']:,} total)" if h.get("total") else "")
                     + (f" · {h['stars']}-star" if h.get("stars") else "")
                     + (f" · rated {h['rating']} from {h['reviews']:,} reviews" if h.get("rating") and h.get("reviews") else "")
                     + (f" · {', '.join(h['amenities'])}" if h["amenities"] else "")
                     + (f" · {h['link']}" if h.get("link") else ""))
    if r.get("link"):
        lines.append(f"The same search on Google Hotels (to book): {r['link']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# booking (Duffel): bookable offers, then an order paid from your Duffel balance
# --------------------------------------------------------------------------

def _duffel(method: str, path: str, body: Optional[Dict[str, Any]] = None, c: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    c = c or config()
    if not c["duffel_token"]:
        raise TravelError("Booking is not connected: add a Duffel access token first (Settings → Flights).")
    req = urllib.request.Request(DUFFEL + path, method=method,
                                 data=json.dumps({"data": body}).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {c['duffel_token']}", "Duffel-Version": "v2",
                                          "Accept": "application/json", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as reply:
            return json.loads(reply.read().decode()).get("data") or {}
    except urllib.error.HTTPError as exc:
        try:
            errs = json.loads(exc.read().decode()).get("errors") or []
            why = "; ".join(e.get("message") or e.get("title") or "" for e in errs) or exc.reason
        except Exception:  # noqa: BLE001
            why = exc.reason
        raise TravelError(f"Duffel said {exc.code}: {why}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TravelError(f"Could not reach Duffel: {exc}") from None


def _offer(o: Dict[str, Any]) -> Dict[str, Any]:
    slices = []
    for sl in o.get("slices") or []:
        segs = sl.get("segments") or []
        slices.append({"from": (sl.get("origin") or {}).get("iata_code"), "to": (sl.get("destination") or {}).get("iata_code"),
                       "duration": sl.get("duration"), "stops": max(0, len(segs) - 1),
                       "departs": segs[0].get("departing_at") if segs else None,
                       "arrives": segs[-1].get("arriving_at") if segs else None,
                       "flights": [f"{(g.get('marketing_carrier') or {}).get('iata_code', '')}{g.get('marketing_carrier_flight_number', '')}"
                                   for g in segs],
                       "via": [(g.get("destination") or {}).get("iata_code") for g in segs[:-1]]})
    return {"offer_id": o.get("id"), "price": o.get("total_amount"), "currency": o.get("total_currency"),
            "airline": (o.get("owner") or {}).get("name"), "expires": o.get("expires_at"),
            "refundable": ((o.get("conditions") or {}).get("refund_before_departure") or {}).get("allowed"),
            "changeable": ((o.get("conditions") or {}).get("change_before_departure") or {}).get("allowed"),
            "slices": slices}


def bookable_flights(origin: str, destination: str, depart: str, return_date: str = "", adults: int = 1,
                     cabin: str = "economy", c: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Offers that can actually be bought, through Duffel, cheapest first."""
    c = c or config()
    origin = _codes(origin or c["home"]).split(",")[0] if (origin or c["home"]) else ""
    if not origin:
        raise ValueError("Where from? Give the airport (for example SFO), or set your home airport.")
    destination = _codes(destination).split(",")[0]
    depart = _date(depart, "departure")
    slices = [{"origin": origin, "destination": destination, "departure_date": depart}]
    if return_date:
        return_date = _date(return_date, "return")
        slices.append({"origin": destination, "destination": origin, "departure_date": return_date})
    cab = {"premium": "premium_economy", "premium economy": "premium_economy"}.get(str(cabin).lower(), str(cabin or "economy").lower())
    data = _duffel("POST", "/air/offer_requests?return_offers=true&supplier_timeout=20000",
                   {"slices": slices, "passengers": [{"type": "adult"}] * max(1, min(9, int(adults or 1))),
                    "cabin_class": cab, "max_connections": 2}, c)
    offers = sorted((_offer(o) for o in data.get("offers") or []), key=lambda o: float(o["price"] or 1e12))
    return {"mode": "live" if c["duffel_token"].startswith("duffel_live_") else "test", "offers": offers[:8],
            "found": len(offers), "searched_at": time.strftime("%Y-%m-%d %H:%M")}


def bookable_text(r: Dict[str, Any]) -> str:
    head = (f"Duffel ({'LIVE: real tickets' if r['mode'] == 'live' else 'TEST mode: pretend airline, nothing real'}), "
            f"searched {r['searched_at']}: {r['found']} bookable offers, cheapest first. Book one with book_flight and its "
            "offer_id; offers expire within about 30 minutes.")
    if not r["offers"]:
        return head + "\nNo bookable offers."
    lines = [head]
    for i, o in enumerate(r["offers"], 1):
        legs = " | ".join(f"{s['from']}→{s['to']} departs {s['departs']} arrives {s['arrives']}, "
                          f"{'nonstop' if not s['stops'] else str(s['stops']) + ' stop(s) via ' + ','.join(s['via'])} "
                          f"({' '.join(s['flights'])})" for s in o["slices"])
        rules = [("refundable" if o["refundable"] else "not refundable") if o["refundable"] is not None else "",
                 ("changeable" if o["changeable"] else "no changes") if o["changeable"] is not None else ""]
        lines.append(f"{i}. offer_id {o['offer_id']} · {o['currency']} {o['price']} · {o['airline']} · {legs}"
                     + (f" · {', '.join(x for x in rules if x)}" if any(rules) else ""))
    return "\n".join(lines)


def book_flight(offer_id: str, c: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Buy one Duffel offer for the saved traveller, paid from the Duffel balance, at today's price."""
    c = c or config()
    missing = traveler_missing(c)
    if missing:
        raise ValueError("The traveller's details are missing (" + ", ".join(missing) + "). Add them under Settings → "
                         "Flights, as on the passport.")
    offer_id = str(offer_id or "").strip()
    if not re.fullmatch(r"off_[\w]{6,80}", offer_id):
        raise ValueError("Give the offer_id from bookable_flights (it starts with off_).")
    offer = _duffel("GET", f"/air/offers/{offer_id}?return_available_services=false", None, c)   # today's price
    passengers = offer.get("passengers") or []
    if len(passengers) != 1:
        raise ValueError("Booking is set up for one traveller (the one saved in Settings). Search again for 1 adult.")
    t = c["traveler"]
    order = _duffel("POST", "/air/orders", {
        "type": "instant", "selected_offers": [offer_id],
        "passengers": [{"id": passengers[0]["id"], "title": t["title"], "given_name": t["given_name"],
                        "family_name": t["family_name"], "born_on": t["born_on"], "gender": t["gender"],
                        "email": t["email"], "phone_number": t["phone_number"]}],
        "payments": [{"type": "balance", "currency": offer.get("total_currency"), "amount": offer.get("total_amount")}]}, c)
    done = {"order_id": order.get("id"), "booking_reference": order.get("booking_reference"),
            "price": order.get("total_amount"), "currency": order.get("total_currency"),
            "airline": (order.get("owner") or {}).get("name"),
            "mode": "live" if c["duffel_token"].startswith("duffel_live_") else "test"}
    try:
        import trips
        done["trip"] = trips.from_booking(order, offer)
    except Exception:  # noqa: BLE001 - the ticket is bought either way
        pass
    try:
        import files
        files.record("booking", f"Flight booked: {done['airline']} {done['booking_reference']}",
                     detail=f"{done['currency']} {done['price']} · order {done['order_id']} · {done['mode']}")
    except Exception:  # noqa: BLE001
        pass
    return done


def test_duffel(c: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    c = c or config()
    if not c["duffel_token"]:
        return {"ok": False, "error": "Add a Duffel access token first."}
    try:
        _duffel("GET", "/air/airlines?limit=1", None, c)
    except TravelError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "mode": "live" if c["duffel_token"].startswith("duffel_live_") else "test"}


GUIDE = (
    "You find flights and hotels for the user. search_flights runs the real Google Flights search right now; "
    "search_hotels runs Google Hotels. Today's date is given below the request: use it. A date without a year is its next one.\n"
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
    "- Hotels: search_hotels, with the place and dates. Same rule: every hotel price comes from the tool.\n"
    "- Booking (only when the user asks to book, and only if bookable_flights is among your tools): run "
    "bookable_flights for the same trip, pick the offer that matches what the user chose (or the cheapest that "
    "fits), and call book_flight with its offer_id and a one-line summary. Do not ask for confirmation in words "
    "first: book_flight shows the user the exact offer and they approve or decline it there, before anything is "
    "bought. Ask only when it is unclear which flight they want. After it, give the booking reference and price. If the tools say booking is not "
    "connected, say so and give the Google Flights link instead.\n"
    "- In Duffel TEST mode the airline is pretend: say plainly that nothing real was booked."
)
