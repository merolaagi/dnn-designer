"""Ideas: things Saathi could do for you right now, worked out from what it can see.

Each idea comes from a rule over your own data, not from a model's guess: a tank under two days of
supply, non-fuel sales well down on the same weekday last week, POS files not arriving, an invoice
whose lines do not add up, unread mail since yesterday, a goal that has gone quiet, nothing
scheduled yet. Each carries an action: a question to ask now, a schedule to set up, or a document to
open. Dismissed ideas stay away for a week. The list is worked out again at most every ten minutes.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentlab as AL

CACHE_SECONDS = 600
DISMISS_DAYS = 7
_CACHE: Dict[str, Any] = {}


def _state_path() -> Path:
    return AL._dir() / "ideas.json"


def _state() -> Dict[str, Any]:
    try:
        return json.loads(_state_path().read_text())
    except (OSError, ValueError):
        return {"dismissed": {}}


def _idea(key: str, icon: str, title: str, text: str, action: Dict[str, Any], topic: str) -> Dict[str, Any]:
    return {"id": "id" + hashlib.sha1(key.encode()).hexdigest()[:10], "icon": icon, "title": title, "text": text,
            "action": action, "topic": topic}


def _store_ideas() -> List[Dict[str, Any]]:
    import station
    v = station.view()
    if not (v["configured"] and v["has_password"]):
        return [_idea("connect-store", "storefront-outline", "Connect your store's data",
                      "With your store's database connected I can answer sales, fuel and tank questions from the "
                      "real numbers, and warn you before a tank runs low.", {"type": "open", "target": "settings"}, "Store")]
    out = []
    try:
        import report
        snap = report.snapshot()
    except Exception:  # noqa: BLE001 - no figures, no store ideas
        return out
    for t in snap.get("tanks") or []:
        if t.get("days") is not None and t["days"] < 2.0:
            out.append(_idea(f"tank-{t['tank']}-{snap['day']}", "water-outline",
                             f"Tank {t['tank']} ({t['product']}) has {t['days']:.1f} days left",
                             f"At the last week's pace it runs dry around {t['days']:.1f} days after {snap['day']}. I can "
                             "work out how much to order for each tank.",
                             {"type": "ask", "task": "Which tanks need fuel ordered now, and how many gallons each, to "
                                                     "last a week at the recent pace?"}, "Fuel"))
    today, before = snap.get("today") or {}, snap.get("last_week") or {}
    for key, label in (("nonfuel", "Non-fuel sales"), ("gallons", "Fuel gallons")):
        a, b = today.get(key), before.get(key)
        if a is not None and b and (a - b) / b <= -0.15:
            out.append(_idea(f"drop-{key}-{snap['day']}", "trending-down-outline",
                             f"{label} were {abs((a - b) / b) * 100:.0f}% below the same day last week",
                             f"{snap['day']} against {snap['week_before']}. I can look at which categories or grades "
                             "made the difference.",
                             {"type": "ask", "task": f"{label} on {snap['day']} were well below {snap['week_before']}. "
                                                     "Which merchandise categories or fuel grades made the difference?"},
                             "Sales"))
    for note in snap.get("notes") or []:
        if "not yesterday's" in note:
            out.append(_idea(f"stale-{snap['day']}", "alert-circle-outline", "The newest business day is not yesterday's",
                             note + " I can check whether the store's POS files are arriving.",
                             {"type": "ask", "task": "Are the store's POS files arriving? When did the last one come in, "
                                                     "and which business days are missing?"}, "Store"))
    return out


def _document_ideas() -> List[Dict[str, Any]]:
    import documents
    out = []
    for d in documents.listing()[:20]:
        if d["status"] == "done" and d.get("problems"):
            out.append(_idea(f"doc-{d['id']}", "receipt-outline",
                             f"{d.get('vendor') or 'A document'} {d.get('number') or ''} has {d['problems']} thing"
                             f"{'' if d['problems'] == 1 else 's'} to check".replace("  ", " "),
                             "Its lines or totals do not add up, or it is not in your records. I can check it against "
                             "your deliveries and invoices.", {"type": "open", "target": f"doc:{d['id']}"}, "Invoices"))
    return out


def _mail_ideas() -> List[Dict[str, Any]]:
    import mail
    if not mail.view()["configured"]:
        return [_idea("connect-mail", "mail-outline", "Connect your email",
                      "I can sort your inbox each morning, find bills and invoices, and draft replies for you to send.",
                      {"type": "ask", "task": "Can you check my emails?"}, "Email")]
    try:
        found = mail.search("in:inbox is:unread newer_than:1d", 25)
    except Exception:  # noqa: BLE001
        return []
    if not found:
        return []
    return [_idea(f"unread-{time.strftime('%Y-%m-%d')}-{len(found) // 5}", "mail-unread-outline",
                  f"{len(found)}{'+' if len(found) >= 25 else ''} unread emails since yesterday",
                  "I can sort them into needs a reply, needs action, for your information and ignore.",
                  {"type": "ask", "task": "Sort my unread email since yesterday into needs a reply, needs action, for "
                                          "your information, and can be ignored."}, "Email")]


def _routine_ideas() -> List[Dict[str, Any]]:
    import schedules
    mine = schedules.listing()
    text = " ".join((s.get("name", "") + " " + s.get("task", "")).lower() for s in mine)
    out = []
    if "brief" not in text:
        out.append(_idea("brief-daily", "sunny-outline", "A morning brief every day",
                         "Yesterday against the same day last week, tanks, deliveries and anything to look at, ready "
                         "at 6 each morning.", {"type": "preset", "which": "brief"}, "Routine"))
    if "mail" not in text and "inbox" not in text:
        out.append(_idea("inbox-daily", "time-outline", "Sort your inbox every morning",
                         "At 7 each morning: what needs a reply, what needs action, and what can wait.",
                         {"type": "schedule", "text": "Every day at 7am check my email and tell me what needs a reply "
                                                      "or action"}, "Routine"))
    return out


def _goal_ideas() -> List[Dict[str, Any]]:
    import goals
    out = []
    for g in goals.listing():
        if g["status"] in ("tracking", "waiting") and time.time() - (g.get("updated") or 0) > 3 * 86400:
            out.append(_idea(f"goal-{g['id']}-{int(g.get('updated') or 0)}", "flag-outline",
                             f"Check on: {g['title']}", f"No news for {int((time.time() - g['updated']) / 86400)} days. "
                             "I can look at where it stands.", {"type": "goal", "goal": g["id"]}, "Goals"))
    return out


def _profile_ideas() -> List[Dict[str, Any]]:
    import muse_profile
    if muse_profile.listing():
        return []
    return [_idea("profile", "person-outline", "Tell me about yourself",
                  "A few lines about your store and how you like answers, and every request starts with them.",
                  {"type": "open", "target": "me"}, "You")]


SOURCES = (_store_ideas, _document_ideas, _mail_ideas, _goal_ideas, _routine_ideas, _profile_ideas)


def listing(fresh: bool = False) -> List[Dict[str, Any]]:
    key = str(AL._dir())
    hit = _CACHE.get(key)
    if hit and not fresh and time.time() - hit["at"] < CACHE_SECONDS:
        found = hit["ideas"]
    else:
        found = []
        for source in SOURCES:
            try:
                found += source()
            except Exception:  # noqa: BLE001 - one source failing must not hide the others
                continue
        _CACHE[key] = {"at": time.time(), "ideas": found}
    gone = _state().get("dismissed", {})
    now = time.time()
    return [i for i in found if now - gone.get(i["id"], 0) > DISMISS_DAYS * 86400]


def dismiss(idea_id: str) -> Dict[str, Any]:
    st = _state()
    st.setdefault("dismissed", {})[idea_id] = time.time()
    _state_path().write_text(json.dumps(st))
    return {"ok": True}


def forget_cache() -> None:
    _CACHE.clear()
