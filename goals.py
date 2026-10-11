"""Goals: what you are working towards, and where each one stands.

A goal is a line in your words ("get the Core-Mark credit for the short delivery") with a status:
tracking, waiting (on someone else), stuck, or done. "Check on it" asks Muse to look, in the goal's
own conversation, and to end its answer with a STATUS line, which updates the goal. "Every day"
makes that check a schedule, so the goals list stays current without you asking.

In a chat, "add a goal: …" or "track: …" adds one.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentlab as AL

STATUSES = ("tracking", "waiting", "stuck", "done")
_LOCK = threading.Lock()
_ADD = re.compile(r"^\s*(?:please\s+)?(?:add (?:a |this as a )?goal|new goal|track|keep track of|my goal is)\s*[:\-–]?\s+(.+)$",
                  re.I | re.S)
STATUS_LINE = re.compile(r"STATUS:\s*(tracking|waiting|stuck|done)\b\s*[-–:]?\s*(.*)", re.I)


def _path() -> Path:
    return AL._dir() / "goals.json"


def _load() -> List[Dict[str, Any]]:
    try:
        return json.loads(_path().read_text())
    except (OSError, ValueError):
        return []


def _store(items: List[Dict[str, Any]]) -> None:
    _path().write_text(json.dumps(items, indent=1))


def _refresh(g: Dict[str, Any]) -> bool:
    """Take the status from the goal's latest finished check, if it is newer than what the goal says."""
    if not g.get("conv"):
        return False
    try:
        turns = AL.conversation(g["conv"])["turns"]
    except KeyError:
        return False
    if turns and turns[-1].get("status") in AL.LIVE_STATUSES:
        changed = g.get("checking") is not True
        g["checking"] = True
        return changed
    was = g.pop("checking", None)
    done = [t for t in turns if t.get("status") == "done" and t.get("answer") and (t.get("at") or 0) > (g.get("checked") or 0)]
    if not done:
        return was is not None
    t = done[-1]
    m = STATUS_LINE.search(t["answer"])
    g["checked"] = t.get("at") or time.time()
    if m:
        g["status"] = m.group(1).lower()
        g["note"] = (m.group(2).strip() or g.get("note") or "")[:300]
    else:
        g["note"] = AL._one_line(t["answer"], 300)
    g["updated"] = time.time()
    return True


def listing() -> List[Dict[str, Any]]:
    with _LOCK:
        items = _load()
        if any([_refresh(g) for g in items]):
            _store(items)
    order = {"stuck": 0, "waiting": 1, "tracking": 2, "done": 3}
    return sorted(items, key=lambda g: (order.get(g["status"], 9), -(g.get("updated") or 0)))


def add(title: str, detail: str = "") -> Dict[str, Any]:
    title = re.sub(r"\s+", " ", str(title or "")).strip().rstrip(".")
    if len(title) < 3:
        raise ValueError("Say what the goal is.")
    if len(title) > 200:
        raise ValueError("Keep a goal's title under 200 characters; put the rest in its details.")
    g = {"id": "gl" + uuid.uuid4().hex[:8], "title": title[0].upper() + title[1:], "detail": str(detail or "").strip()[:1000],
         "status": "tracking", "note": "", "created": time.time(), "updated": time.time(), "conv": None, "schedule": None}
    with _LOCK:
        items = _load()
        items.append(g)
        _store(items)
    return g


def _find(items, gid):
    for g in items:
        if g["id"] == gid:
            return g
    raise KeyError(gid)


def update(gid: str, change: Dict[str, Any]) -> Dict[str, Any]:
    with _LOCK:
        items = _load()
        g = _find(items, gid)
        if change.get("title"):
            g["title"] = str(change["title"]).strip()[:200]
        if change.get("detail") is not None:
            g["detail"] = str(change["detail"]).strip()[:1000]
        if change.get("status"):
            if change["status"] not in STATUSES:
                raise ValueError(f"A goal's status is one of: {', '.join(STATUSES)}.")
            g["status"] = change["status"]
        if change.get("note") is not None:
            g["note"] = str(change["note"]).strip()[:300]
        g["updated"] = time.time()
        _store(items)
    return g


def remove(gid: str) -> Dict[str, Any]:
    with _LOCK:
        items = _load()
        g = _find(items, gid)
        _store([x for x in items if x["id"] != gid])
    if g.get("schedule"):
        import schedules
        try:
            schedules.delete(g["schedule"])
        except Exception:  # noqa: BLE001
            pass
    return {"ok": True}


def check_task(g: Dict[str, Any]) -> str:
    return (f"Check on this goal of mine and tell me where it stands: {g['title']}."
            + (f"\nDetails: {g['detail']}" if g.get("detail") else "")
            + (f"\nLast time: {g['note']}" if g.get("note") else "")
            + "\nLook at whatever bears on it (my email, my store's data, the web), say briefly what changed and what "
              "I should do next, and end with one line: STATUS: tracking, waiting, stuck or done, then a few words why.")


def check(gid: str, mode: str = "live") -> Dict[str, Any]:
    """Ask Muse to look at the goal now, as the next turn of the goal's own conversation."""
    with _LOCK:
        items = _load()
        g = _find(items, gid)
        if not g.get("conv"):
            conv = AL.new_conversation({"kind": "muse", "key": "muse"}, mode)
            conv["title"] = f"Goal: {g['title'][:50]}"
            AL._conv_save(conv)
            g["conv"] = conv["id"]
            _store(items)
    AL.say(g["conv"], check_task(g), mode, AL._dir())
    with _LOCK:
        items = _load()
        g = _find(items, gid)
        g["checking"] = True
        _store(items)
    return g


def follow(gid: str, at: str = "08:00", mode: str = "live") -> Dict[str, Any]:
    """Check on the goal every day, in its own conversation."""
    import schedules
    with _LOCK:
        items = _load()
        g = _find(items, gid)
    if not g.get("conv"):
        conv = AL.new_conversation({"kind": "muse", "key": "muse"}, mode)
        conv["title"] = f"Goal: {g['title'][:50]}"
        AL._conv_save(conv)
        g["conv"] = conv["id"]
    s = schedules.create(f"Goal: {g['title'][:50]}", {"kind": "muse", "key": "muse"}, check_task(g), "day", at, 0, mode,
                         g["conv"])
    with _LOCK:
        items = _load()
        x = _find(items, gid)
        x.update(conv=g["conv"], schedule=s["id"])
        _store(items)
    return {**x, "when": s["when"]}


def command(text: str) -> Optional[str]:
    """ "add a goal: …" or "track: …" adds a goal; returns the reply, or None for anything else."""
    m = _ADD.match(text or "")
    if not m or "?" in m.group(1)[-2:]:
        return None
    g = add(m.group(1))
    return (f"Added to your goals: **{g['title']}**. I'll keep it under **Goals**. Tap Check on it there to have me look "
            "now, or Every day to have me check each morning.")
