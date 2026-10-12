"""Trips: your itinerary, with a map, directions, and reminders before each thing.

A trip is a list of items in time order: flights, hotel stays, trains and buses, things to do,
notes. Saathi fills it from what it books for you (a flight booked through Duffel goes straight in)
and from the confirmations in your email; you can add a stop yourself. Times are the local time of
the place, as on the booking.

Each item with a place is put on the map (OpenStreetMap's geocoder; results are kept, so a place is
looked up once). "How do I get there" uses Google Maps' transit directions through SerpApi (the same
key as flight prices): which train or bus, from which stop, when it leaves, where to change.

Reminders go to your Inbox and phone: online check-in a day before a flight, leaving for the
airport three hours before, the hotel's address and check-in time on the day, an hour before
anything else. On each day of a trip, a morning note lists the day's plan.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentlab as AL

KINDS = ("flight", "hotel", "train", "bus", "transfer", "activity", "food", "note")
_LOCK = threading.Lock()
_GEO_LOCK = threading.Lock()
_LAST_GEO = [0.0]
AGENT = "Saathi/1 (personal itinerary on one Mac)"
DIGEST_HOUR = 7


def _path() -> Path:
    return AL._dir() / "trips.json"


def _load() -> List[Dict[str, Any]]:
    try:
        return json.loads(_path().read_text())
    except (OSError, ValueError):
        return []


def _store(items: List[Dict[str, Any]]) -> None:
    _path().write_text(json.dumps(items, indent=1))


def _when(text: Any, name: str = "time") -> Optional[str]:
    """'2026-11-12T23:55' (local time of the place); a date alone is allowed."""
    if text in (None, ""):
        return None
    text = str(text).strip().replace(" ", "T")
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})(?:T(\d{1,2}):(\d{2}))?(?::\d{2})?", text)
    if not m:
        raise ValueError(f"The {name} is YYYY-MM-DD or YYYY-MM-DDTHH:MM, such as 2026-11-12T23:55.")
    return m.group(1) + (f"T{int(m.group(2)):02d}:{m.group(3)}" if m.group(2) else "")


def _parse(stamp: Optional[str]) -> Optional[dt.datetime]:
    if not stamp:
        return None
    return dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M") if "T" in stamp else dt.datetime.strptime(stamp, "%Y-%m-%d")


# --------------------------------------------------------------------------
# trips and their items
# --------------------------------------------------------------------------

def listing() -> List[Dict[str, Any]]:
    out = []
    today = time.strftime("%Y-%m-%d")
    for t in _load():
        state = "past" if (t.get("end") or "9") < today else "now" if (t.get("start") or "0") <= today else "upcoming"
        out.append({k: t.get(k) for k in ("id", "title", "destination", "start", "end")} | {"items": len(t["items"]),
                                                                                         "state": state})
    order = {"now": 0, "upcoming": 1, "past": 2}
    return sorted(out, key=lambda t: (order[t["state"]], t["start"] or ""))


def _find(items, tid):
    for t in items:
        if t["id"] == tid:
            return t
    raise KeyError(tid)


def get(tid: str) -> Dict[str, Any]:
    t = _find(_load(), tid)
    items = sorted(t["items"], key=lambda i: (i.get("start") or "9999", i.get("title") or ""))
    days: Dict[str, List[Dict[str, Any]]] = {}
    for i in items:
        days.setdefault((i.get("start") or "")[:10] or "undated", []).append({**i, "links": links(i)})
    return {**t, "items": items, "days": [{"date": d, "items": v} for d, v in days.items()]}


def create(title: str, destination: str = "", start: str = "", end: str = "") -> Dict[str, Any]:
    title = re.sub(r"\s+", " ", str(title or "")).strip()
    if len(title) < 2:
        raise ValueError("Give the trip a name, such as Kathmandu, November.")
    t = {"id": "tr" + uuid.uuid4().hex[:8], "title": title[:80], "destination": str(destination or "").strip()[:80],
         "start": _when(start, "start date") and _when(start)[:10], "end": _when(end, "end date") and _when(end)[:10],
         "items": [], "created": time.time()}
    with _LOCK:
        items = _load()
        items.append(t)
        _store(items)
    ensure_reminders()
    return t


def update(tid: str, change: Dict[str, Any]) -> Dict[str, Any]:
    with _LOCK:
        items = _load()
        t = _find(items, tid)
        for k in ("title", "destination"):
            if change.get(k):
                t[k] = str(change[k]).strip()[:80]
        for k in ("start", "end"):
            if change.get(k):
                t[k] = _when(change[k], k)[:10]
        _store(items)
    return t


def remove(tid: str) -> Dict[str, Any]:
    with _LOCK:
        items = _load()
        _find(items, tid)
        _store([t for t in items if t["id"] != tid])
    return {"ok": True}


def _clean_item(raw: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(raw.get("kind") or "activity").lower()
    if kind not in KINDS:
        raise ValueError(f"An item is one of: {', '.join(KINDS)}.")
    title = re.sub(r"\s+", " ", str(raw.get("title") or "")).strip()
    if len(title) < 2:
        raise ValueError("Say what it is, such as Qatar Airways QR 738 SFO → DOH, or Hotel Yak & Yeti.")
    item = {"kind": kind, "title": title[:120], "start": _when(raw.get("start"), "start"), "end": _when(raw.get("end"), "end"),
            "place": str(raw.get("place") or "").strip()[:120], "address": str(raw.get("address") or "").strip()[:200],
            "confirmation": str(raw.get("confirmation") or "").strip()[:60], "details": str(raw.get("details") or "").strip()[:600],
            "to_place": str(raw.get("to_place") or "").strip()[:120], "source": str(raw.get("source") or "you")[:40]}
    where = item["address"] or item["place"]
    if where:
        geo = geocode(where)
        if geo:
            item.update(lat=geo["lat"], lon=geo["lon"])
    return item


def add_item(tid: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    item = {"id": "it" + uuid.uuid4().hex[:8], **_clean_item(raw), "sent": []}
    with _LOCK:
        items = _load()
        t = _find(items, tid)
        same = [i for i in t["items"] if i["title"].lower() == item["title"].lower() and i.get("start") == item["start"]]
        if same:
            return same[0]                                   # already there (an email read twice)
        t["items"].append(item)
        day = (item["start"] or "")[:10]
        if day:
            t["start"] = min(t.get("start") or day, day)
            t["end"] = max(t.get("end") or day, (item.get("end") or day)[:10])
        _store(items)
    return item


def update_item(tid: str, iid: str, change: Dict[str, Any]) -> Dict[str, Any]:
    with _LOCK:
        items = _load()
        t = _find(items, tid)
        i = _find(t["items"], iid)
        merged = _clean_item({**i, **{k: v for k, v in change.items() if v is not None}})
        moved = merged["start"] != i.get("start")
        i.update(merged)
        if moved:
            i["sent"] = []                                   # a new time: remind again
        _store(items)
    return i


def remove_item(tid: str, iid: str) -> Dict[str, Any]:
    with _LOCK:
        items = _load()
        t = _find(items, tid)
        _find(t["items"], iid)
        t["items"] = [i for i in t["items"] if i["id"] != iid]
        _store(items)
    return {"ok": True}


# --------------------------------------------------------------------------
# where things are, and how to get there
# --------------------------------------------------------------------------

def _geo_path() -> Path:
    return AL._dir() / "trips-geo.json"


def geocode(text: str) -> Optional[Dict[str, Any]]:
    """Latitude and longitude of a place or address, from OpenStreetMap; looked up once, then kept."""
    key = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not key:
        return None
    with _GEO_LOCK:
        try:
            cache = json.loads(_geo_path().read_text())
        except (OSError, ValueError):
            cache = {}
        if key in cache:
            return cache[key]
        wait = 1.1 - (time.time() - _LAST_GEO[0])           # OpenStreetMap asks for at most one lookup a second
        if wait > 0:
            time.sleep(wait)
        _LAST_GEO[0] = time.time()
        try:
            url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({"q": text, "format": "json", "limit": 1})
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}), timeout=15) as r:
                found = json.loads(r.read().decode())
        except Exception:  # noqa: BLE001 - no map pin is better than no item
            return None
        hit = {"lat": float(found[0]["lat"]), "lon": float(found[0]["lon"]), "name": found[0].get("display_name", "")[:160]} \
            if found else None
        cache[key] = hit
        _geo_path().write_text(json.dumps(cache))
        return hit


def links(item: Dict[str, Any]) -> Dict[str, str]:
    """Open the place, or directions to it by public transport, in Apple Maps or Google Maps."""
    if item.get("lat") is not None:
        at = f"{item['lat']},{item['lon']}"
    elif item.get("address") or item.get("place"):
        at = item.get("address") or item.get("place")
    else:
        return {}
    q = urllib.parse.quote(at)
    name = urllib.parse.quote(item.get("place") or item.get("title") or "")
    return {"apple": f"https://maps.apple.com/?daddr={q}&dirflg=r&q={name}",
            "google": f"https://www.google.com/maps/dir/?api=1&destination={q}&travelmode=transit",
            "map": f"https://maps.apple.com/?ll={q}&q={name}" if item.get("lat") is not None else f"https://maps.apple.com/?q={q}"}


MODES = {"best": 6, "driving": 0, "transit": 3, "walking": 2, "cycling": 1}


def directions(start: str, end: str, mode: str = "transit", depart: str = "", prefer: str = "") -> Dict[str, Any]:
    """Google Maps directions now (or leaving at a time): for transit, each line, its stops and times."""
    import travel
    c = travel.config()
    if not c["serpapi_key"]:
        raise ValueError("Directions use the SerpApi key, which is not set yet (Hub → Me → Travel).")
    start, end = str(start or "").strip(), str(end or "").strip()
    if not start or not end:
        raise ValueError("Say where from and where to.")
    params = {"engine": "google_maps_directions", "start_addr": start, "end_addr": end,
              "travel_mode": MODES.get(str(mode or "transit").lower(), 3), "hl": "en", "api_key": c["serpapi_key"]}
    if depart:
        when = _parse(_when(depart, "departure"))
        params["time"] = f"depart_at:{int(when.timestamp())}"
    if prefer in ("bus", "subway", "train", "tram_light_rail"):
        params["prefer"] = prefer
    data = travel._get(params)
    routes = []
    for d in (data.get("directions") or [])[:3]:
        legs = []
        for trip in d.get("trips") or []:
            leg = {"mode": trip.get("travel_mode"), "title": trip.get("title"), "minutes": round((trip.get("duration") or 0) / 60),
                   "from": (trip.get("start_stop") or {}).get("name"), "leaves": (trip.get("start_stop") or {}).get("time"),
                   "to": (trip.get("end_stop") or {}).get("name"), "arrives": (trip.get("end_stop") or {}).get("time"),
                   "stops": len(trip.get("stops") or []), "by": (trip.get("service_run_by") or {}).get("name")}
            if not leg["from"] and trip.get("details"):
                leg["steps"] = [x.get("title") for x in trip["details"][:6] if x.get("title")]
            legs.append(leg)
        routes.append({"mode": d.get("travel_mode"), "via": d.get("via"), "leaves": d.get("start_time"),
                       "arrives": d.get("end_time"), "minutes": round((d.get("duration") or 0) / 60),
                       "cost": d.get("cost"), "currency": d.get("currency"), "legs": legs})
    q = urllib.parse.quote
    return {"from": start, "to": end, "routes": routes,
            "apple": f"https://maps.apple.com/?saddr={q(start)}&daddr={q(end)}&dirflg=r",
            "google": f"https://www.google.com/maps/dir/?api=1&origin={q(start)}&destination={q(end)}&travelmode=transit"}


def directions_text(r: Dict[str, Any]) -> str:
    if not r["routes"]:
        return f"Google Maps found no route from {r['from']} to {r['to']}. Open it in maps: {r['apple']}"
    out = [f"Google Maps, {r['from']} → {r['to']}:"]
    for n, d in enumerate(r["routes"], 1):
        head = f"{n}. {d['mode'] or ''} {d['minutes']} min" + (f", leaves {d['leaves']}" if d.get("leaves") else "") + \
            (f", arrives {d['arrives']}" if d.get("arrives") else "") + (f", via {d['via']}" if d.get("via") else "") + \
            (f", costs {d['cost']} {d['currency'] or ''}" if d.get("cost") else "")
        out.append(head)
        for leg in d["legs"]:
            if leg.get("from"):
                out.append(f"   - {leg['mode']}: {leg['title'] or ''} from {leg['from']} at {leg['leaves'] or '?'} to "
                           f"{leg['to']} ({leg['arrives'] or '?'}), {leg['stops']} stops" + (f", run by {leg['by']}" if leg.get("by") else ""))
            else:
                out.append(f"   - {leg['mode']} {leg['minutes']} min" + (": " + "; ".join(leg["steps"]) if leg.get("steps") else ""))
    out.append(f"Open it in Apple Maps: {r['apple']}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# reminders
# --------------------------------------------------------------------------

def _reminders_for(i: Dict[str, Any]) -> List[tuple]:
    """(key, when, title, text) for one item."""
    at = _parse(i.get("start"))
    if not at or "T" not in (i.get("start") or ""):
        return []
    where = i.get("address") or i.get("place") or ""
    conf = f" Confirmation {i['confirmation']}." if i.get("confirmation") else ""
    link = links(i).get("apple", "")
    if i["kind"] == "flight":
        return [("checkin", at - dt.timedelta(hours=24), f"Check in online: {i['title']}",
                 f"Your flight leaves {at:%a %b %-d at %-I:%M %p}.{conf} Online check-in usually opens 24 hours before."),
                ("leave", at - dt.timedelta(hours=3), f"Leave for the airport: {i['title']}",
                 f"It leaves at {at:%-I:%M %p}{' from ' + where if where else ''}. Aim to be there 2 to 3 hours before an "
                 f"international flight.{conf}" + (f" Directions: {link}" if link else ""))]
    if i["kind"] == "hotel":
        return [("day", at.replace(hour=8, minute=0) if at.hour >= 9 else at - dt.timedelta(hours=1),
                 f"Today: check in at {i['title']}",
                 f"Check-in from {at:%-I:%M %p}.{(' ' + where + '.') if where else ''}{conf}"
                 + (f" Directions: {link}" if link else ""))]
    return [("soon", at - dt.timedelta(hours=1), f"In an hour: {i['title']}",
             f"At {at:%-I:%M %p}{(', ' + where) if where else ''}.{conf}" + (f" Directions: {link}" if link else ""))]


def due(now: Optional[dt.datetime] = None) -> List[Dict[str, Any]]:
    """Reminders whose time has come (within the last 2 hours: a late one is still worth sending, a stale one is not)."""
    now = now or dt.datetime.now()
    out = []
    for t in _load():
        for i in t["items"]:
            for key, when, title, text in _reminders_for(i):
                if key not in i.get("sent", []) and when <= now < when + dt.timedelta(hours=2):
                    out.append({"trip": t["id"], "item": i["id"], "key": key, "title": title, "text": text})
    return out


def digest(t: Dict[str, Any], day: str) -> Optional[str]:
    todays = sorted((i for i in t["items"] if (i.get("start") or "")[:10] == day), key=lambda i: i.get("start") or "")
    if not todays:
        return None
    lines = []
    for i in todays:
        at = _parse(i["start"])
        lines.append(f"{at:%-I:%M %p} {i['title']}" if "T" in i["start"] else i["title"])
    return "\n".join(lines)


def check(s: Dict[str, Any], now: Optional[dt.datetime] = None) -> None:
    """The scheduled look (every hour): send the reminders that are due, and each trip day's morning plan."""
    import schedules
    now = now or dt.datetime.now()
    sent = 0
    for r in due(now):
        schedules.notify("trip", r["title"], r["text"], s.get("id"))
        with _LOCK:
            items = _load()
            try:
                i = _find(_find(items, r["trip"])["items"], r["item"])
                i.setdefault("sent", []).append(r["key"])
                _store(items)
            except KeyError:
                pass
        sent += 1
    day = now.strftime("%Y-%m-%d")
    if now.hour >= DIGEST_HOUR:
        with _LOCK:
            items = _load()
            for t in items:
                if (t.get("start") or "9") <= day <= (t.get("end") or "0") and day not in t.setdefault("digests", []):
                    plan = digest(t, day)
                    if plan:
                        schedules.notify("trip", f"{t['title']}: today", plan, s.get("id"))
                        sent += 1
                    t["digests"].append(day)
            _store(items)
    s["last_status"] = "ok"
    s["last_detail"] = f"{sent} sent" if sent else "nothing due"


def ensure_reminders() -> None:
    import schedules
    if any(x["agent"] == {"kind": "check", "key": "trips"} for x in schedules.listing()):
        return
    try:
        schedules.create("Trip reminders", {"kind": "check", "key": "trips"},
                         "Send trip reminders and each trip day's plan.", "hour", "00:00")
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------
# a flight Saathi booked goes into a trip
# --------------------------------------------------------------------------

def from_booking(order: Dict[str, Any], offer: Dict[str, Any]) -> Optional[str]:
    """Put a booked flight's legs into the trip around those dates (making one if there is none)."""
    slices = offer.get("slices") or []
    if not slices:
        return None
    first = (slices[0].get("segments") or [{}])[0]
    dest = ((slices[0].get("destination") or {}).get("city_name") or (slices[0].get("destination") or {}).get("iata_code") or "Trip")
    day = (first.get("departing_at") or "")[:10]
    tid = None
    for t in _load():
        if t.get("start") and t.get("end") and t["start"] <= day <= t["end"]:
            tid = t["id"]
            break
    if tid is None:
        tid = create(f"{dest}, {day[:7]}" if day else str(dest), str(dest), day, day)["id"]
    for sl in slices:
        for g in sl.get("segments") or []:
            carrier = (g.get("marketing_carrier") or {})
            frm, to = (g.get("origin") or {}), (g.get("destination") or {})
            add_item(tid, {"kind": "flight", "title": f"{carrier.get('name') or ''} {carrier.get('iata_code', '')}"
                                                      f"{g.get('marketing_carrier_flight_number', '')} {frm.get('iata_code')} → {to.get('iata_code')}".strip(),
                           "start": (g.get("departing_at") or "")[:16], "end": (g.get("arriving_at") or "")[:16],
                           "place": f"{frm.get('name') or frm.get('iata_code')} airport", "to_place": to.get("name") or to.get("iata_code") or "",
                           "confirmation": order.get("booking_reference") or "", "source": "booked by Saathi"})
    return tid


GUIDE = (
    "You keep the person's trips: the itinerary, where things are, and how to get between them. Tools: "
    "list_trips, show_trip, add_trip, add_item, update_item, remove_item, directions (Google Maps, with public "
    "transport lines and times), plus their email (search_mail, read_mail) and the web.\n"
    "- To build or fill a trip, search their email for bookings around the dates (flight confirmations, hotel "
    "reservations, trains, buses, tours: words like confirmation, itinerary, booking, reservation, e-ticket, and the "
    "destination), read them, and add one item each with add_item: kind, title, start and end in the local time "
    "of the place (YYYY-MM-DDTHH:MM, as the booking says), the place's name and street address, the confirmation "
    "number. Make the trip first with add_trip if there is none for those dates.\n"
    "- For how to get somewhere, use directions from where they will be (the airport, the hotel, the last stop) "
    "to the place, with mode transit, leaving at the right time. Explain it step by step: walk to which stop, "
    "which train or bus (its line name), when it leaves, how many stops, where to change, how long, what it "
    "costs if shown. Then the Apple Maps link. If transit is poor, say what a taxi or ride would be instead.\n"
    "- When asked what is next or about a day, show_trip and answer with that day's plan, times and addresses.\n"
    "Every booking detail comes from an email you read or a tool result; never invent times, numbers or "
    "addresses. Say where you found each item."
)
