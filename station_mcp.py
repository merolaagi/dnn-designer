"""An MCP server (stdio) offering Station data's read-only tools: list_tables, describe_table, query.

The Agent lab's Station analyst starts it as `python3 station_mcp.py`; any MCP client can. It speaks
newline-delimited JSON-RPC on stdin and stdout, and reads its connection from data/station_data.json.
"""

import json
import sys

import station

TOOLS = [
    {"name": "list_tables", "description": "The tables of the station's database this login can read, with how "
     "many columns each has.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "describe_table", "description": "The columns of one table and their types.",
     "inputSchema": {"type": "object", "properties": {"table": {"type": "string", "description": "Table name"}},
                     "required": ["table"]}},
    {"name": "query", "description": "Run one read-only SELECT (or WITH … SELECT) on the station's database. "
     "Returns CSV, at most 200 rows: aggregate with GROUP BY and filter by business_date rather than reading "
     "raw rows. Queries stop after 10 seconds.",
     "inputSchema": {"type": "object", "properties": {"sql": {"type": "string", "description": "The SELECT"},
                                                      "limit": {"type": "integer", "description": "Rows, up to 200"}},
                     "required": ["sql"]}},
]


def call(name, args):
    if name == "list_tables":
        found = station.tables()
        return "\n".join(f"{t['table']} ({t['columns']} columns)" for t in found) or "No readable tables."
    if name == "describe_table":
        d = station.describe(args.get("table"))
        return d["table"] + "\n" + "\n".join(f"  {c['name']}: {c['type']}" for c in d["columns"])
    if name == "query":
        return station.as_text(station.query(args.get("sql"), args.get("limit") or station.MAX_ROWS))
    raise ValueError(f"There is no tool {name}.")


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None                                  # a notification: nothing to answer
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "station-data", "version": "1"}}
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
            reply = handle(json.loads(line))
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
