"""A tiny MCP server over stdio, for the tests: an echo tool and an adder."""
import json
import sys

TOOLS = [
    {"name": "echo", "description": "Say the text back.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "add", "description": "Add two numbers.",
     "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                     "required": ["a", "b"]}},
]

for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method, params = msg["method"], msg.get("params") or {}
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "echo", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        args = params.get("arguments") or {}
        if params["name"] == "echo":
            result = {"content": [{"type": "text", "text": "echo: " + str(args.get("text"))}]}
        elif params["name"] == "add":
            try:
                result = {"content": [{"type": "text", "text": str(float(args["a"]) + float(args["b"]))}]}
            except (KeyError, TypeError, ValueError):
                result = {"content": [{"type": "text", "text": "a and b must be numbers"}], "isError": True}
        else:
            result = {"content": [{"type": "text", "text": "no such tool"}], "isError": True}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "unknown"}}),
              flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
