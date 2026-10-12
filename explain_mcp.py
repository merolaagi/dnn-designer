"""An MCP server (stdio) for the Explainer agent: make_explainer. Started as `python3 explain_mcp.py`."""

import json
import sys

import explain

TOOLS = [
    {"name": "make_explainer", "description": "Build an explainer page from a spec (title, summary, sections of kind "
     "text, math, steps, plot, graph or table) and return its link. Says exactly what to fix if the spec is wrong.",
     "inputSchema": {"type": "object", "properties": {
         "spec": {"type": "object", "description": "{title, summary, sections: [...]}, as described in your instructions"},
         "question": {"type": "string", "description": "The person's question, in their words"}}, "required": ["spec"]}},
]


def call(name, a):
    if name == "make_explainer":
        spec = a.get("spec")
        if isinstance(spec, str):
            try:
                spec = json.loads(spec)
            except ValueError:
                raise ValueError("The spec is JSON: an object with title, summary and sections.")
        made = explain.save(spec, a.get("question") or "")
        return f"Made: {made['title']}. The link to give the person (exactly as written): {made['url']}"
    raise ValueError(f"There is no tool {name}.")


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "explain", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params") or {}
        try:
            text, error = call(params.get("name"), params.get("arguments") or {}), False
        except (ValueError, RuntimeError, KeyError, TypeError, AttributeError) as exc:
            text, error = f"Fix the spec: {exc}", True
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
