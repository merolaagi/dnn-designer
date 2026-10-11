"""An MCP server (stdio) for your email: search_mail, read_mail and draft_reply. It never sends.

The Agent lab's Mail assistant starts it as `python3 mail_mcp.py`; it reads its connection from
data/mail.json. Drafts go to Gmail's Drafts folder for you to send yourself.
"""

import json
import sys

import mail

TOOLS = [
    {"name": "search_mail", "description": "Find messages with Gmail's search words (for example \"is:unread "
     "newer_than:1d\", \"from:chevron.com\", \"subject:invoice\"). Returns id, from, subject, date and whether it "
     "is unread, newest first; read_mail gives the text.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string", "description": "Gmail search words"},
                                                      "limit": {"type": "integer", "description": "How many, up to 25"}}}},
    {"name": "read_mail", "description": "One message's sender, recipients, subject, date, attachments' names and "
     "text, by the id search_mail gave. Reading does not mark it as read.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string", "description": "The message id"}},
                     "required": ["id"]}},
    {"name": "draft_reply", "description": "Save a draft in Gmail's Drafts, threaded as a reply when reply_to_id is "
     "given. It is never sent: the user sends it from Gmail.",
     "inputSchema": {"type": "object", "properties": {
         "to": {"type": "string", "description": "Recipient address(es), comma-separated"},
         "subject": {"type": "string", "description": "Subject; left empty on a reply, it becomes Re: the original"},
         "body": {"type": "string", "description": "The draft's text"},
         "reply_to_id": {"type": "string", "description": "The id of the message this answers, if any"}},
         "required": ["to", "body"]}},
]
READ_TOOLS = ["search_mail", "read_mail"]
WRITE_TOOLS = ["draft_reply"]


def _lines(found):
    if not found:
        return "No messages match."
    return "\n".join(f"[{m['id']}] {'UNREAD ' if m['unread'] else ''}{(m['date'] or '')[:16]} · {m['from']} · {m['subject']}"
                     for m in found)


def call(name, args):
    if name == "search_mail":
        return _lines(mail.search(args.get("query") or "in:inbox newer_than:2d", args.get("limit") or 15))
    if name == "read_mail":
        m = mail.read(args.get("id"))
        head = [f"From: {m['from']}", f"To: {m['to']}"] + ([f"Cc: {m['cc']}"] if m["cc"] else []) + \
               [f"Date: {m['date']}", f"Subject: {m['subject']}"] + \
               ([f"Attachments: {', '.join(m['attachments'])}"] if m["attachments"] else [])
        return "\n".join(head) + "\n\n" + (m["body"] or "(no text)")
    if name == "draft_reply":
        d = mail.draft(args.get("to"), args.get("subject") or "", args.get("body"), args.get("reply_to_id"))
        return f"{d['note']} To: {d['to']}. Subject: {d['subject']}."
    raise ValueError(f"There is no tool {name}.")


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "mail", "version": "1"}}
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
