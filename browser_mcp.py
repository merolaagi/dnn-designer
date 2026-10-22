"""An MCP server (stdio) for the Support agent: the browser actions, and commit (behind your approval).

The Agent lab starts it as `python3 browser_mcp.py`; it drives Saathi's browser through browser.py.
"""

import json
import re
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
    {"name": "purchase", "description": "Click the button that books or pays (Book now, Complete booking, Pay, Reserve, "
     "Place order, Confirm and pay). The person sees the summary and approves it first.",
     "inputSchema": {"type": "object", "properties": {
         "summary": {"type": "string", "description": "Exactly what is bought: what, dates, guests, total price with taxes, "
                                                       "cancellation terms"},
         "id": {"type": "string", "description": "The button's id from look"}}, "required": ["summary", "id"]}},
    {"name": "traveller", "description": "The person's name, email and phone as saved on the Mac, and the shipping "
     "address if one is saved, for a booking or checkout form (never card details).", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "commit", "description": "Send a message that agrees to something (accept an offer, change or cancel a "
     "service). The person sees the summary and approves it first.",
     "inputSchema": {"type": "object", "properties": {
         "summary": {"type": "string", "description": "Exactly what is being agreed: the offer, price, length, contract"},
         "id": {"type": "string", "description": "The chat box's id from look"},
         "text": {"type": "string", "description": "The message to send"}}, "required": ["summary", "text"]}},
]
READ_TOOLS = ["open", "look", "click", "type", "press", "scroll", "back", "wait"]
WRITE_TOOLS = ["commit"]
BOOK_READ_TOOLS = READ_TOOLS + ["traveller"]
BOOK_WRITE_TOOLS = ["purchase"]
# a button that spends money: only purchase (your approval) clicks it
PAY = re.compile(r"\b(book( now| it)?|complete (booking|purchase|order)|confirm (and pay|booking|purchase|reservation)|"
                 r"pay( now)?|place (order|booking)|purchase|reserve( now)?|buy( now)?|checkout|submit (order|payment)|"
                 r"finish booking)\b", re.I)


def page_text(v):
    lines = [f"Page: {v['title']} ({v['url']})"]
    if v.get("asks_password"):
        lines.append("THIS PAGE ASKS FOR A PASSWORD, CARD DETAILS OR A CODE. Do not type them: stop and ask the person to "
                     "enter them from Saathi → Hub → Browser, then to tell you to continue.")
    lines.append("Text:\n" + (v["text"] or "(no text)"))
    lines.append("Things to click or type into:")
    for e in v["elements"]:
        lines.append(f"  [{e['id']}] {e['kind']}{' (only the person types here)' if e.get('password') else ''}"
                     f"{' (in focus)' if e.get('focused') else ''}: {e['label'] or '(no label)'}")
    return "\n".join(lines)


def call(name, args):
    if name == "open":
        return page_text(browser.call("open", url=args.get("url") or ""))
    if name == "look":
        return page_text(browser.call("look"))
    if name == "click":
        here = browser.call("look")
        target = next((e for e in here["elements"] if e["id"] == args.get("id")), None)
        if target and target["kind"] == "button" and PAY.search(target.get("label") or "") and \
                (here.get("asks_password") or re.search(r"\b(payment|pay now|card number|billing|purchase)\b",
                                                        here.get("text") or "", re.I)):
            raise ValueError(f"'{target['label']}' books or pays. Use purchase, with a summary of what is bought and the "
                             "total, so the person can approve it.")
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
    if name == "traveller":
        import travel
        t = travel.config().get("traveler") or {}
        keep = {k: t.get(k) for k in ("title", "given_name", "family_name", "email", "phone_number") if t.get(k)}
        try:
            import shop
            if shop.prefs().get("ship_to"):
                keep["ship_to"] = shop.prefs()["ship_to"]
        except Exception:  # noqa: BLE001
            pass
        return json.dumps(keep) if keep else ("No traveller details are saved. Ask the person for the name, email and "
                                              "phone for the booking, or to fill them in under Hub → Me → Travel.")
    if name == "purchase":
        browser.call("click", id=args.get("id"))
        return ("Clicked, with the person's approval: " + str(args.get("summary") or "") + ". " +
                page_text(browser.call("look")))
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
