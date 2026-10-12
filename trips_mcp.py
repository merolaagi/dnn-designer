"""An MCP server (stdio) for the Trips agent: the itinerary and directions. Started as `python3 trips_mcp.py`."""

import json
import sys

import trips

ITEM = {"kind": {"type": "string", "description": "flight, hotel, train, bus, transfer, activity, food or note"},
        "title": {"type": "string", "description": "e.g. Qatar Airways QR 738 SFO → DOH, or Hotel Yak & Yeti"},
        "start": {"type": "string", "description": "Local time of the place, YYYY-MM-DDTHH:MM (check-in time for a hotel)"},
        "end": {"type": "string", "description": "YYYY-MM-DDTHH:MM (arrival, or check-out for a hotel)"},
        "place": {"type": "string", "description": "The place's name (the departure airport for a flight)"},
        "address": {"type": "string", "description": "Street address, for the map and directions"},
        "to_place": {"type": "string", "description": "Where it arrives, for a flight, train or bus"},
        "confirmation": {"type": "string"}, "details": {"type": "string", "description": "Seat, room, notes"},
        "source": {"type": "string", "description": "Where you found it, e.g. 'email from Qatar Airways, Oct 3'"}}

TOOLS = [
    {"name": "list_trips", "description": "The person's trips: id, name, destination, dates, how many items, and whether "
     "it is now, upcoming or past.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "show_trip", "description": "One trip day by day: each item with its time, place, address, confirmation and "
     "map links.", "inputSchema": {"type": "object", "properties": {"trip": {"type": "string"}}, "required": ["trip"]}},
    {"name": "add_trip", "description": "Start a trip.", "inputSchema": {"type": "object", "properties": {
        "title": {"type": "string"}, "destination": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"}},
        "required": ["title"]}},
    {"name": "add_item", "description": "Add a flight, stay, train, bus, transfer, activity, meal or note to a trip.",
     "inputSchema": {"type": "object", "properties": {"trip": {"type": "string"}, **ITEM}, "required": ["trip", "kind", "title"]}},
    {"name": "update_item", "description": "Change an item (only the fields given).",
     "inputSchema": {"type": "object", "properties": {"trip": {"type": "string"}, "item": {"type": "string"}, **ITEM},
                     "required": ["trip", "item"]}},
    {"name": "remove_item", "description": "Take an item out of a trip.",
     "inputSchema": {"type": "object", "properties": {"trip": {"type": "string"}, "item": {"type": "string"}},
                     "required": ["trip", "item"]}},
    {"name": "directions", "description": "Google Maps directions now or leaving at a time: for transit, each train or bus "
     "line, the stop to board at, departure and arrival times, stops, changes and cost; with an Apple Maps link.",
     "inputSchema": {"type": "object", "properties": {
         "start": {"type": "string", "description": "Where from: an address, a place or 'lat,lon'"},
         "end": {"type": "string", "description": "Where to"},
         "mode": {"type": "string", "description": "transit (default), walking, driving, best"},
         "depart": {"type": "string", "description": "When leaving, YYYY-MM-DDTHH:MM; empty for now"},
         "prefer": {"type": "string", "description": "For transit: bus, subway, train or tram_light_rail"}},
         "required": ["start", "end"]}},
]


def _day_text(t):
    lines = [f"{t['title']} ({t['id']}): {t.get('destination') or ''} {t.get('start') or '?'} to {t.get('end') or '?'}"]
    for d in t["days"]:
        lines.append(f"\n{d['date']}:")
        for i in d["items"]:
            when = (i.get("start") or "")[11:] or "any time"
            lines.append(f"  [{i['id']}] {when} {i['kind']}: {i['title']}" + (f" at {i['place']}" if i.get("place") else "")
                         + (f", {i['address']}" if i.get("address") else "") + (f" → {i['to_place']}" if i.get("to_place") else "")
                         + (f" · until {i['end']}" if i.get("end") else "") + (f" · conf {i['confirmation']}" if i.get("confirmation") else "")
                         + (f" · {i['details']}" if i.get("details") else "") + ("" if i.get("lat") is not None else " · (not on the map)"))
    if not t["items"]:
        lines.append("No items yet.")
    return "\n".join(lines)


def call(name, a):
    fields = {k: a.get(k) for k in ITEM if a.get(k) not in (None, "")}
    if name == "list_trips":
        found = trips.listing()
        return "\n".join(f"[{t['id']}] {t['title']} · {t.get('destination') or ''} · {t.get('start')} to {t.get('end')} · "
                         f"{t['items']} items · {t['state']}" for t in found) or "No trips yet."
    if name == "show_trip":
        return _day_text(trips.get(a.get("trip") or ""))
    if name == "add_trip":
        t = trips.create(a.get("title") or "", a.get("destination") or "", a.get("start") or "", a.get("end") or "")
        return f"Trip {t['id']} started: {t['title']}."
    if name == "add_item":
        i = trips.add_item(a.get("trip") or "", fields)
        return f"Added [{i['id']}] {i['title']}" + ("" if i.get("lat") is not None else " (could not place it on the map; give a street address)")
    if name == "update_item":
        i = trips.update_item(a.get("trip") or "", a.get("item") or "", fields)
        return f"Changed [{i['id']}] {i['title']}."
    if name == "remove_item":
        trips.remove_item(a.get("trip") or "", a.get("item") or "")
        return "Removed."
    if name == "directions":
        return trips.directions_text(trips.directions(a.get("start") or "", a.get("end") or "", a.get("mode") or "transit",
                                                      a.get("depart") or "", a.get("prefer") or ""))
    raise ValueError(f"There is no tool {name}.")


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "trips", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params") or {}
        try:
            text, error = call(params.get("name"), params.get("arguments") or {}), False
        except KeyError:
            text, error = "There is no trip or item with that id: list_trips or show_trip first.", True
        except (ValueError, RuntimeError) as exc:
            text, error = str(exc), True
        result = {"content": [{"type": "text", "text": text}], "isError": error}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Unknown method {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = handle(json.loads(line))
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
