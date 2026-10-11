"""An MCP server (stdio) for travel: search_flights and search_hotels (Google, through SerpApi),
bookable_flights and book_flight (Duffel).

The Agent lab's Travel agent starts it as `python3 travel_mcp.py`; it reads its keys from
data/travel.json. book_flight is the only tool that changes anything, and the agent's graph puts
it behind your approval.
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
    {"name": "search_hotels", "description": "Search Google Hotels now: hotels with their price per night and in "
     "total, star class, rating and reviews, amenities and a link, cheapest first, and the Google Hotels link.",
     "inputSchema": {"type": "object", "properties": {
         "place": {"type": "string", "description": "City, area or hotel, e.g. Thamel, Kathmandu"},
         "check_in": {"type": "string", "description": "YYYY-MM-DD"},
         "check_out": {"type": "string", "description": "YYYY-MM-DD"},
         "adults": {"type": "integer", "description": "Guests, 1 to 9"},
         "max_price": {"type": "integer", "description": "Highest price per night, optional"}},
         "required": ["place", "check_in", "check_out"]}},
    {"name": "bookable_flights", "description": "Offers that can actually be bought (through Duffel), cheapest first, "
     "each with an offer_id for book_flight, its fare rules and when it expires. Use it only when the user asks to book.",
     "inputSchema": {"type": "object", "properties": {
         "origin": {"type": "string", "description": "One airport code; empty for the user's home airport"},
         "destination": {"type": "string", "description": "One airport code"},
         "depart": {"type": "string", "description": "YYYY-MM-DD"},
         "return_date": {"type": "string", "description": "YYYY-MM-DD; empty for one way"},
         "cabin": {"type": "string", "description": "economy, premium, business or first"}},
         "required": ["destination", "depart"]}},
    {"name": "book_flight", "description": "Buy one offer from bookable_flights for the traveller saved on the Mac, paid "
     "from the Duffel balance. The user approves it first. Returns the booking reference.",
     "inputSchema": {"type": "object", "properties": {
         "offer_id": {"type": "string", "description": "The offer_id from bookable_flights (off_…)"},
         "summary": {"type": "string", "description": "For the user's approval: airline, flights, dates and the price"}},
         "required": ["offer_id", "summary"]}},
]
READ_TOOLS = ["search_flights", "search_hotels", "bookable_flights"]
WRITE_TOOLS = ["book_flight"]


def call(name, args):
    if name == "search_flights":
        return travel.as_text(travel.search_flights(args.get("origin") or "", args.get("destination") or "",
                                                    args.get("depart") or "", args.get("return_date") or "",
                                                    args.get("adults") or 1, args.get("cabin") or "economy",
                                                    args.get("stops") or "any"))
    if name == "search_hotels":
        return travel.hotels_text(travel.search_hotels(args.get("place") or "", args.get("check_in") or "",
                                                       args.get("check_out") or "", args.get("adults") or 2,
                                                       args.get("max_price")))
    if name == "bookable_flights":
        return travel.bookable_text(travel.bookable_flights(args.get("origin") or "", args.get("destination") or "",
                                                            args.get("depart") or "", args.get("return_date") or "", 1,
                                                            args.get("cabin") or "economy"))
    if name == "book_flight":
        d = travel.book_flight(args.get("offer_id") or "")
        return (f"Booked{' (TEST mode: pretend airline, nothing real)' if d['mode'] == 'test' else ''}: {d['airline']}, "
                f"booking reference {d['booking_reference']}, {d['currency']} {d['price']} paid from the Duffel balance. "
                f"Order {d['order_id']}. The airline emails the ticket to the traveller.")
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
