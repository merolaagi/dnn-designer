"""An MCP server (stdio) for flights: search_flights, the live Google Flights search through SerpApi.

The Agent lab's Travel agent starts it as `python3 travel_mcp.py`; it reads its key from
data/travel.json. It only searches: nothing is booked.
"""

import json
import sys

import travel

TOOLS = [
    {"name": "search_flights", "description": "Search Google Flights now. Returns the options cheapest first (price "
     "for all passengers, for the whole trip on a round trip), airlines, stops and layovers, times, Google's "
     "low/typical/high price judgement, and the Google Flights link to book.",
     "inputSchema": {"type": "object", "properties": {
         "origin": {"type": "string", "description": "Airport code(s), comma-separated, e.g. SFO or SFO,OAK,SJC. "
                                                     "Empty: the user's home airport."},
         "destination": {"type": "string", "description": "Airport code(s), e.g. KTM"},
         "depart": {"type": "string", "description": "Departure date, YYYY-MM-DD"},
         "return_date": {"type": "string", "description": "Return date, YYYY-MM-DD; empty for one way"},
         "adults": {"type": "integer", "description": "Passengers, 1 to 9"},
         "cabin": {"type": "string", "description": "economy, premium, business or first"},
         "stops": {"type": "string", "description": "any, nonstop, 1 (at most one stop) or 2"}},
         "required": ["destination", "depart"]}},
]


def call(name, args):
    if name == "search_flights":
        return travel.as_text(travel.search_flights(args.get("origin") or "", args.get("destination") or "",
                                                    args.get("depart") or "", args.get("return_date") or "",
                                                    args.get("adults") or 1, args.get("cabin") or "economy",
                                                    args.get("stops") or "any"))
    raise ValueError(f"There is no tool {name}.")


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "travel", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params") or {}
        try:
            text, error = call(params.get("name"), params.get("arguments") or {}), False
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
            msg = json.loads(line)
        except ValueError:
            continue
        reply = handle(msg)
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
