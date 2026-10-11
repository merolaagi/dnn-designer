"""Alerts on your phone, through ntfy.

Whatever reaches the inbox (a store alert, a scheduled run finished or waiting for approval, a check
that failed) can also go to your phone as a push notification. ntfy (ntfy.sh, or a server you run)
is a free relay: install its app, subscribe to your topic, and messages published to that topic
arrive as notifications. The topic name is the only secret, so it is a long random one.

By default a notification carries only its title ("University Chevron: no new POS data for 36+
hours"). With details on, it carries the text too, and the morning brief comes with its PDF. Those
go through the ntfy server you choose.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
KINDS = {
    "alert": "Store alerts: POS data stopped arriving",
    "recovered": "POS data arriving again",
    "approval": "A run is waiting for your approval",
    "done": "A scheduled run finished (the morning brief)",
    "error": "A schedule or check failed",
    "stopped": "A scheduled run was stopped",
    "skipped": "A schedule was skipped (its last run still going)",
}
DEFAULTS = {"server": "https://ntfy.sh", "topic": "", "kinds": ["alert", "recovered", "approval", "done", "error"],
            "details": False, "public_url": "", "token": ""}
PRIORITY = {"alert": 5, "approval": 4, "error": 4, "recovered": 3, "done": 3, "stopped": 3, "skipped": 2}
TAGS = {"alert": ["rotating_light"], "recovered": ["white_check_mark"], "approval": ["raised_hand"],
        "done": ["sunrise"], "error": ["warning"], "stopped": ["stop_sign"], "skipped": ["fast_forward"]}
_STATUS: Dict[str, Any] = {"last_sent": None, "last_error": None}


def _path() -> Path:
    path = HERE / "data"
    path.mkdir(exist_ok=True)
    return path / "push.json"


def config() -> Dict[str, Any]:
    try:
        saved = json.loads(_path().read_text())
    except (OSError, ValueError):
        saved = {}
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}


def view() -> Dict[str, Any]:
    c = config()
    return {**{k: v for k, v in c.items() if k != "token"}, "has_token": bool(c["token"]), "kinds_all": KINDS,
            "on": bool(c["topic"]), **_STATUS}


def new_topic() -> str:
    return "muse-" + secrets.token_urlsafe(18).replace("_", "x").replace("-", "y").lower()


def save(change: Dict[str, Any]) -> Dict[str, Any]:
    c = config()
    if change.get("server") is not None:
        server = str(change["server"]).strip().rstrip("/")
        if server and not re.fullmatch(r"https?://[\w.\-:]+(/[\w.\-/]*)?", server):
            raise ValueError("The ntfy server is an address such as https://ntfy.sh.")
        c["server"] = server or DEFAULTS["server"]
    if change.get("topic") is not None:
        topic = str(change["topic"]).strip()
        if topic and not re.fullmatch(r"[\w\-]{1,64}", topic):
            raise ValueError("A topic may only hold letters, digits, dashes and underscores (up to 64).")
        if topic and len(topic) < 12:
            raise ValueError("Use a long topic (12 characters or more): anyone who knows it can read your alerts.")
        c["topic"] = topic
    if change.get("kinds") is not None:
        c["kinds"] = [k for k in change["kinds"] if k in KINDS]
    if change.get("details") is not None:
        c["details"] = bool(change["details"])
    if change.get("public_url") is not None:
        url = str(change["public_url"]).strip().rstrip("/")
        if url and not re.fullmatch(r"https?://[\w.\-:]+(/[\w.\-/]*)?", url):
            raise ValueError("The address of this app is something like https://muse.example.com.")
        c["public_url"] = url
    if change.get("token"):
        c["token"] = str(change["token"]).strip()
    if change.get("clear_token"):
        c["token"] = ""
    _path().write_text(json.dumps(c, indent=1))
    try:
        os.chmod(_path(), 0o600)
    except OSError:
        pass
    return view()


def _click(c: Dict[str, Any], row: Dict[str, Any]) -> Optional[str]:
    if not c["public_url"]:
        return None
    return c["public_url"] + "/m" + (f"#conv={row['conv']}" if row.get("conv") else "#inbox")


def message_for(row: Dict[str, Any], c: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The ntfy JSON message for one inbox row."""
    c = c or config()
    body = (row.get("text") or "").strip() if c["details"] else ""
    msg = {"topic": c["topic"], "title": (row.get("title") or "Muse")[:200],
           "message": (body or "Open Muse to read it.")[:3500],
           "priority": PRIORITY.get(row.get("kind"), 3), "tags": TAGS.get(row.get("kind"), [])}
    click = _click(c, row)
    if click:
        msg["click"] = click
    return msg


def _ascii(text: str) -> str:
    return (text or "").encode("ascii", "replace").decode().replace("?", "-")[:250]


def _post(c: Dict[str, Any], url: str, data: bytes, headers: Dict[str, str], method: str = "POST") -> None:
    headers = {**headers, **({"Authorization": f"Bearer {c['token']}"} if c["token"] else {})}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=20) as reply:
        reply.read()


def deliver(row: Dict[str, Any], attachment: Optional[Dict[str, Any]] = None, c: Optional[Dict[str, Any]] = None) -> None:
    """Send now (raises on failure). An attachment {name, data} goes only when details are on."""
    c = c or config()
    msg = message_for(row, c)
    if attachment and c["details"]:
        # a file is the request body, so the rest goes in headers, which must be plain ASCII
        headers = {"Filename": attachment["name"], "Title": _ascii(msg["title"]),
                   "Message": _ascii(msg["message"]).replace("\n", " "), "Priority": str(msg["priority"]),
                   "Tags": ",".join(msg["tags"])}
        if msg.get("click"):
            headers["Click"] = msg["click"]
        _post(c, f"{c['server']}/{c['topic']}", attachment["data"], headers, "PUT")
    else:
        _post(c, c["server"], json.dumps(msg).encode(), {"Content-Type": "application/json"})
    _STATUS.update(last_sent=time.time(), last_error=None)


def send(row: Dict[str, Any], attachment: Optional[Dict[str, Any]] = None) -> bool:
    """Push one inbox row if it is a kind you chose, in the background. True if it was sent on its way."""
    c = config()
    if not c["topic"] or row.get("kind") not in c["kinds"] or os.environ.get("AGENTLAB_NO_PUSH"):
        return False

    def go():
        try:
            deliver(row, attachment, c)
        except Exception as exc:  # noqa: BLE001 - a push failing must never stop the scheduler
            _STATUS["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
    threading.Thread(target=go, daemon=True).start()
    return True


def test() -> Dict[str, Any]:
    c = config()
    if not c["topic"]:
        raise ValueError("Choose a topic first (Generate makes one).")
    try:
        deliver({"kind": "done", "title": "Muse: test notification",
                 "text": "If you can read this on your phone, alerts will reach you."}, None, c)
    except Exception as exc:  # noqa: BLE001
        _STATUS["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
        raise ValueError(f"ntfy did not take the message: {exc}") from None
    return view()


def subscribe_url(c: Optional[Dict[str, Any]] = None) -> str:
    c = c or config()
    return f"{c['server']}/{c['topic']}" if c["topic"] else ""


def kinds() -> List[str]:
    return list(KINDS)
