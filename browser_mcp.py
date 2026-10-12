"""An MCP server (stdio) for the Support agent: the browser actions, and commit (behind your approval).

The Agent lab starts it as `python3 browser_mcp.py`; it drives Saathi's browser through browser.py.
"""

import json
import sys

import browser
import support

TOOLS = [
    {"name": "open", "description": "Open a web address in the browser, then show the page as look does.",
     "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    {"name": "look", "description": "The page now: its address, its text (the newest at the end, chat windows "
     "included), and its buttons, links and boxes, each with an id to click or type into.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "click", "description": "Click a button or link by its id from look.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
    {"name": "type", "description": "Type into a box by its id from look (or the box in focus if no id); submit=true "
     "presses Enter to send. Not for passwords, and not for agreeing to anything (use commit).",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"},
                                                      "submit": {"type": "boolean"}}, "required": ["text"]}},
    {"name": "press", "description": "Press a key: Enter, Tab, Escape, Backspace, ArrowDown, ArrowUp, PageDown, PageUp.",
     "inputSchema": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}},
    {"name": "scroll", "description": "Scroll the page down or up.",
     "inputSchema": {"type": "object", "properties": {"direction": {"type": "string"}}}},
    {"name": "back", "description": "Go back one page.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "wait", "description": "Wait for a reply, 1 to 60 seconds, then look.",
     "inputSchema": {"type": "object", "properties": {"seconds": {"type": "integer"}}}},
    {"name": "commit", "description": "Send a message that agrees to something (accept an offer, change or cancel a "
     "service). The person sees the summary and approves it first.",
     "inputSchema": {"type": "object", "properties": {
         "summary": {"type": "string", "description": "Exactly what is being agreed: the offer, price, length, contract"},
         "id": {"type": "string", "description": "The chat box's id from look"},
         "text": {"type": "string", "description": "The message to send"}}, "required": ["summary", "text"]}},
]
READ_TOOLS = ["open", "look", "click", "type", "press", "scroll", "back", "wait"]
WRITE_TOOLS = ["commit"]


def page_text(v):
    lines = [f"Page: {v['title']} ({v['url']})"]
    if v.get("asks_password"):
        lines.append("THIS PAGE ASKS FOR A PASSWORD. Do not type it: stop and ask the person to sign in from "
                     "Saathi → Hub → Browser, then to tell you to continue.")
    lines.append("Text:\n" + (v["text"] or "(no text)"))
    lines.append("Things to click or type into:")
    for e in v["elements"]:
        lines.append(f"  [{e['id']}] {e['kind']}{' (password)' if e.get('password') else ''}"
                     f"{' (in focus)' if e.get('focused') else ''}: {e['label'] or '(no label)'}")
    return "\n".join(lines)


def call(name, args):
    if name == "open":
        return page_text(browser.call("open", url=args.get("url") or ""))
    if name == "look":
        return page_text(browser.call("look"))
    if name == "click":
        browser.call("click", id=args.get("id"))
        return "Clicked. " + page_text(browser.call("look"))
    if name == "type":
        text = str(args.get("text") or "")
        if support.COMMIT.search(text):
            raise ValueError("That message agrees to something. Send it with commit, with a summary, so the person "
                             "can approve it first.")
        browser.call("type", text=text, id=args.get("id"), submit=bool(args.get("submit")))
        return ("Sent." if args.get("submit") else "Typed.") + " Wait for the reply, then look."
    if name == "press":
        browser.call("press", key=args.get("key") or "Enter")
        return "Pressed. " + page_text(browser.call("look"))
    if name == "scroll":
        browser.call("scroll", direction=args.get("direction") or "down")
        return page_text(browser.call("look"))
    if name == "back":
        browser.call("back")
        return page_text(browser.call("look"))
    if name == "wait":
        browser.call("wait", seconds=args.get("seconds") or 10)
        return page_text(browser.call("look"))
    if name == "commit":
        browser.call("type", text=str(args.get("text") or ""), id=args.get("id"), submit=True)
        return "Sent, with the person's approval: " + str(args.get("summary") or "") + ". Wait, then look for the reply."
    raise ValueError(f"There is no tool {name}.")


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "browser", "version": "1"}}
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
