"""Scheduled agents, and an inbox they report to.

A schedule runs one agent (Muse, a starting design, a specialist or a saved agent) on one task
at a time you choose: every hour, every day, on weekdays, or once a week, by this computer's
clock. Each schedule keeps its own conversation in Talk to agents, so every run is a turn you
can read, with its steps. The scheduler watches each run and writes to the inbox when it
finishes, fails, or stops to wait for your approval.

When the server starts again after being stopped, runs a schedule had going are resumed from
their last finished step, and a schedule that came due while the server was down runs once.
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

EVERY = ("hour", "day", "weekdays", "week")
TICK = 20.0                     # seconds between the scheduler's looks
_LOCK = threading.RLock()
_THREAD: Optional[threading.Thread] = None


def _path() -> Path:
    return AL._dir() / "schedules.json"


def _inbox_path() -> Path:
    return AL._dir() / "inbox.jsonl"


def _load() -> List[Dict[str, Any]]:
    try:
        return json.loads(_path().read_text())
    except (OSError, ValueError):
        return []


def _store(items: List[Dict[str, Any]]) -> None:
    _path().write_text(json.dumps(items, indent=1))


# --------------------------------------------------------------------------
# when a schedule is due
# --------------------------------------------------------------------------

def _hm(at: str):
    try:
        h, m = [int(x) for x in str(at or "06:00").split(":")[:2]]
    except ValueError:
        raise ValueError("Give the time as HH:MM, such as 06:00.") from None
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError("Give the time as HH:MM, such as 06:00.")
    return h, m


def next_run(s: Dict[str, Any], after: float) -> float:
    """The first moment after `after` that the schedule should run, by this computer's local time."""
    h, m = _hm(s.get("at"))
    every = s.get("every")
    if every == "hour":
        t = time.localtime(after)
        base = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, m, 0, 0, 0, -1))
        return base if base > after else base + 3600
    t = time.localtime(after)
    day = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, h, m, 0, 0, 0, -1))
    for k in range(0, 9):
        cand = time.mktime((t.tm_year, t.tm_mon, t.tm_mday + k, h, m, 0, 0, 0, -1))
        if cand <= after:
            continue
        wd = time.localtime(cand).tm_wday             # Monday is 0
        if every == "weekdays" and wd >= 5:
            continue
        if every == "week" and wd != int(s.get("weekday") or 0):
            continue
        return cand
    return day + 86400


def describe(s: Dict[str, Any]) -> str:
    h, m = _hm(s.get("at"))
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return {"hour": f"every hour at :{m:02d}", "day": f"every day at {h:02d}:{m:02d}",
            "weekdays": f"weekdays at {h:02d}:{m:02d}",
            "week": f"every {days[int(s.get('weekday') or 0)]} at {h:02d}:{m:02d}"}[s["every"]]


# --------------------------------------------------------------------------
# the schedules themselves
# --------------------------------------------------------------------------

def listing() -> List[Dict[str, Any]]:
    out = []
    for s in _load():
        out.append({**s, "when": describe(s)})
    return out


def create(name: str, agent: Dict[str, Any], task: str, every: str, at: str = "06:00", weekday: int = 0,
           mode: str = "live", conv: Optional[str] = None) -> Dict[str, Any]:
    name, task = str(name or "").strip(), str(task or "").strip()
    if not task:
        raise ValueError("Say what the agent should do each time.")
    if every not in EVERY:
        raise ValueError(f"Choose how often: {', '.join(EVERY)}.")
    if mode not in ("live", "rehearsal"):
        raise ValueError("Choose live or rehearsal.")
    if not (agent or {}).get("kind"):
        raise ValueError("Choose which agent runs it.")
    if agent["kind"] == "check" and agent.get("key") not in CHECKS:
        raise ValueError(f"There is no built-in check called {agent.get('key')}.")
    if agent["kind"] not in ("muse", "check"):
        AL._agent_graph(agent)                        # raises when the agent does not exist
    s = {"id": "sc" + uuid.uuid4().hex[:8], "name": name or task[:50], "agent": {"kind": agent["kind"], "key": agent.get("key") or "muse"},
         "task": task, "every": every, "at": at, "weekday": max(0, min(6, int(weekday or 0))), "mode": mode,
         "enabled": True, "created": time.time(), "conv": conv, "last_run": None, "last_status": None,
         "watching": None}
    _hm(at)
    s["next_run"] = next_run(s, time.time())
    with _LOCK:
        items = _load()
        items.append(s)
        _store(items)
    return {**s, "when": describe(s)}


def update(sid: str, change: Dict[str, Any]) -> Dict[str, Any]:
    with _LOCK:
        items = _load()
        s = next((x for x in items if x["id"] == sid), None)
        if s is None:
            raise KeyError(sid)
        for k in ("name", "task", "every", "at", "weekday", "mode", "enabled"):
            if k in change and change[k] is not None:
                s[k] = change[k]
        if s["every"] not in EVERY:
            raise ValueError(f"Choose how often: {', '.join(EVERY)}.")
        _hm(s["at"])
        s["next_run"] = next_run(s, time.time())
        _store(items)
    return {**s, "when": describe(s)}


def delete(sid: str) -> Dict[str, Any]:
    with _LOCK:
        items = [x for x in _load() if x["id"] != sid]
        _store(items)
    return {"ok": True}


def run_now(sid: str) -> Dict[str, Any]:
    with _LOCK:
        items = _load()
        s = next((x for x in items if x["id"] == sid), None)
        if s is None:
            raise KeyError(sid)
        _fire(s)
        _store(items)
    return {**s, "when": describe(s)}


# --------------------------------------------------------------------------
# the inbox
# --------------------------------------------------------------------------

def notify(kind: str, title: str, text: str = "", schedule: Optional[str] = None, conv: Optional[str] = None,
           attachment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write to the inbox, and push it to your phone if you chose that kind (see push.py)."""
    row = {"id": "in" + uuid.uuid4().hex[:8], "t": time.time(), "kind": kind, "title": title[:200],
           "text": AL._one_line(text, 600), "schedule": schedule, "conv": conv, "read": False}
    with _LOCK:
        with open(_inbox_path(), "a") as f:
            f.write(json.dumps(row) + "\n")
    try:
        import push
        push.send({**row, "text": text if kind == "done" else row["text"]}, attachment)
    except Exception:  # noqa: BLE001 - the inbox is what matters; a push is extra
        pass
    return row


def inbox(limit: int = 50) -> Dict[str, Any]:
    try:
        rows = [json.loads(line) for line in _inbox_path().read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        rows = []
    rows = rows[-500:]
    return {"items": list(reversed(rows))[:limit], "unread": sum(1 for r in rows if not r.get("read"))}


def mark_read(ids: Optional[List[str]] = None) -> Dict[str, Any]:
    with _LOCK:
        try:
            rows = [json.loads(line) for line in _inbox_path().read_text().splitlines() if line.strip()]
        except (OSError, ValueError):
            rows = []
        for r in rows:
            if ids is None or r["id"] in ids:
                r["read"] = True
        _inbox_path().write_text("".join(json.dumps(r) + "\n" for r in rows[-500:]))
    return inbox()


# --------------------------------------------------------------------------
# the scheduler
# --------------------------------------------------------------------------

def _ingest_watchdog(s: Dict[str, Any]) -> None:
    """The built-in check: is each site's POS data still arriving? One alert per incident, one note on recovery."""
    import station
    result = station.watchdog()
    news, s["state"] = station.watchdog_news(result, s.get("state") or {})
    for kind, title, text in news:
        notify(kind, title, text, s["id"])
    s["last_status"] = "alert" if any(x["stale"] for x in result["sites"]) else "ok"
    s["last_detail"] = "; ".join(f"{x['name']}: {x['hours']:.0f} h" if x["hours"] is not None else f"{x['name']}: no files"
                                 for x in result["sites"])


def _feed(s: Dict[str, Any]) -> None:
    import feed
    feed.check(s)


def _trips(s: Dict[str, Any]) -> None:
    import trips
    trips.check(s)


CHECKS = {"ingest_watchdog": _ingest_watchdog, "feed": _feed, "trips": _trips}


def _fire(s: Dict[str, Any], now: Optional[float] = None) -> None:
    """Start the schedule's agent on its task, as the next turn of its own conversation."""
    now = max(now or 0, time.time())
    if s["agent"]["kind"] == "check":
        try:
            CHECKS[s["agent"]["key"]](s)
        except Exception as exc:  # noqa: BLE001
            if s.get("last_status") != "error":
                notify("error", f"{s['name']}: the check could not run", f"{type(exc).__name__}: {exc}", s["id"])
            s["last_status"] = "error"
        s["last_run"], s["next_run"] = now, next_run(s, now)
        return
    try:
        import station
        if s.get("task", "").startswith(station.OLD_BRIEF):
            s["task"] = station.BRIEF_TASK              # the old brief read day_reports, which is no longer written
        conv = None
        if s.get("conv"):
            try:
                conv = AL.conversation(s["conv"])
            except KeyError:
                conv = None
        if conv and conv["turns"] and conv["turns"][-1].get("status") in AL.LIVE_STATUSES:
            notify("skipped", f"{s['name']}: skipped", "The last run was still going.", s["id"], conv["id"])
            return
        if conv is None:
            conv = AL.new_conversation(s["agent"], s["mode"])
            conv["title"] = f"Scheduled: {s['name']}"
            AL._conv_save(conv)
            s["conv"] = conv["id"]
        AL.say(conv["id"], s["task"], s["mode"], AL._dir())
        s.update(last_run=now, last_status="running", watching=len(AL.conversation(conv["id"])["turns"]) - 1,
                 asked=False)
    except Exception as exc:  # noqa: BLE001 - one broken schedule must not stop the others
        s.update(last_run=now, last_status="error")
        notify("error", f"{s['name']}: did not start", f"{type(exc).__name__}: {exc}", s["id"], s.get("conv"))
    finally:
        s["next_run"] = next_run(s, now)


def _watch(s: Dict[str, Any]) -> None:
    """Report on the run a schedule started: finished, failed, or waiting for you."""
    if s.get("watching") is None or not s.get("conv"):
        return
    try:
        turn = AL.conversation(s["conv"])["turns"][s["watching"]]
    except (KeyError, IndexError):
        s["watching"] = None
        return
    status = turn.get("status")
    if status == "waiting" and not s.get("asked"):
        s["asked"] = True
        notify("approval", f"{s['name']}: waiting for your approval",
               "Open the conversation to approve or decline what it wants to do.", s["id"], s["conv"])
    if status in AL.LIVE_STATUSES:
        return
    s["last_status"], s["watching"] = status, None
    if status == "done":
        notify("done", f"{s['name']}: done", turn.get("answer") or "", s["id"], s["conv"], _brief_pdf(s, turn))
    else:
        notify("error" if status == "error" else "stopped", f"{s['name']}: {status}",
               turn.get("error") or "", s["id"], s["conv"])


def _brief_pdf(s: Dict[str, Any], turn: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """For the morning brief: the store report with the brief's text, as a PDF to go with the notification."""
    if not str(s.get("name", "")).lower().startswith("morning brief"):
        return None
    try:
        import report
        snap = report.snapshot()
        made = {"name": f"brief-{snap['day']}.pdf",
                "data": report.pdf(snap, {"answer": turn.get("answer"), "at": turn.get("at")})}
    except Exception:  # noqa: BLE001 - no PDF is better than no notification
        return None
    try:
        import files
        files.record("brief", f"Morning brief for {snap['day']}", data=made["data"], conv=s.get("conv"),
                     key=f"brief-{snap['day']}")
    except Exception:  # noqa: BLE001
        pass
    return made


def tick(now: Optional[float] = None) -> int:
    """One look at every schedule in this workspace. Returns how many started."""
    now = now or time.time()
    started = 0
    with _LOCK:
        items = _load()
        for s in items:
            _watch(s)
            if s.get("enabled") and s.get("watching") is None and s.get("next_run") and now >= s["next_run"]:
                _fire(s, now)
                started += 1
        if items:
            _store(items)
    return started


def resume_interrupted() -> List[str]:
    """After a restart: carry on any schedule's run the server stopped in the middle of."""
    resumed = []
    convs = {s.get("conv") for s in _load() if s.get("conv")}
    for conv_id in convs:
        try:
            conv = AL.conversation(conv_id)
        except KeyError:
            continue
        for turn in conv["turns"]:
            if turn.get("status") == "interrupted" and turn.get("run") and not turn.get("muse"):
                try:
                    AL.resume_run(turn["run"], AL._dir())
                    resumed.append(turn["run"])
                except (KeyError, ValueError):
                    continue
    return resumed


def _workspaces() -> List[Optional[str]]:
    """Every account (None: no accounts), each once."""
    import auth
    names: List[Optional[str]] = [None] + sorted(auth.users())
    seen, out = set(), []
    for n in names:
        place = str(auth.workspace_for(n))
        if place not in seen:
            seen.add(place)
            out.append(n)
    return out


def _loop() -> None:
    import auth
    first = True
    while True:
        for name in _workspaces():
            auth.set_current(name)
            try:
                if first:
                    for run in resume_interrupted():
                        notify("resumed", "Resumed after a restart", f"Run {run} carried on from its last finished step.")
                tick()
            except Exception:  # noqa: BLE001 - the scheduler keeps going whatever one workspace does
                pass
        first = False
        time.sleep(TICK)


def start() -> None:
    """Start the scheduler once per server process."""
    global _THREAD
    if _THREAD is None or not _THREAD.is_alive():
        _THREAD = threading.Thread(target=_loop, name="schedules", daemon=True)
        _THREAD.start()


# --------------------------------------------------------------------------
# scheduling in your own words, from a chat
# --------------------------------------------------------------------------
# "Every day at 7am check my email and tell me what needs a reply" becomes a schedule that asks Muse
# that task each morning; "stop the daily email check" removes it. Anything else is left to the agents.

_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_TIME = r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?"
_WHEN = re.compile(
    r"\b(?:(every|each)\s+(day|morning|evening|night|weekday|hour|" + "|".join(_DAYS) + r")s?|daily|hourly|"
    r"(on\s+)?weekdays)\b(?:\s*,?\s*" + _TIME + r")?", re.I)
_AT_FIRST = re.compile(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\s+(every|each)\s+(day|morning|evening|weekday|"
                       + "|".join(_DAYS) + r")s?\b", re.I)
_STOP = re.compile(r"^\s*(?:please\s+)?(?:stop|cancel|delete|remove|turn off|pause)\b(.*)$", re.I | re.S)


def _clock(hour: Optional[str], minute: Optional[str], half: Optional[str], word: str) -> Optional[str]:
    if hour is None:
        default = {"morning": "07:00", "evening": "18:00", "night": "21:00"}.get(word)
        return default
    h, m = int(hour), int(minute or 0)
    half = (half or "").lower().replace(".", "")
    if half == "pm" and h < 12:
        h += 12
    if half == "am" and h == 12:
        h = 0
    if not half and word in ("evening", "night") and h < 12:
        h += 12
    if not (0 <= h < 24 and 0 <= m < 60):
        return None
    return f"{h:02d}:{m:02d}"


_QUESTION = re.compile(r"^\s*(how|what|which|when|why|who|did|do|does|is|are|was|were|has|have)\b", re.I)


def from_words(text: str) -> Optional[Dict[str, Any]]:
    """{every, at, weekday, task} when the message asks for something to happen on a schedule; else None.

    A question about the data ("how much do we sell every day at noon?") is a question, not a schedule."""
    text = str(text or "").strip()
    if _QUESTION.match(text):
        return None
    m = _AT_FIRST.search(text)
    if m:
        word = m.group(5).lower()
        at = _clock(m.group(1), m.group(2), m.group(3), word)
        span = m.span()
    else:
        m = _WHEN.search(text)
        if not m:
            return None
        word = (m.group(2) or ("hour" if m.group(0).lower().startswith("hourly") else
                               "weekday" if "weekday" in m.group(0).lower() else "day")).lower()
        at = _clock(m.group(4), m.group(5), m.group(6), word)
        span = m.span()
    if word == "hour":
        every, at = "hour", "00:00"
    elif word == "weekday":
        every = "weekdays"
    elif word in _DAYS:
        every = "week"
    else:
        every = "day"
    if every != "hour" and not at:
        return None                                      # "every day" with no time: ask rather than guess
    task = re.sub(r"\s{2,}", " ", text[:span[0]] + " " + text[span[1]:]).strip(" ,.;:-")
    task = re.sub(r"^(?:can you|could you|please|i want you to|i'd like you to)\s+", "", task, flags=re.I).strip(" ,.")
    task = re.sub(r"^(?:and|then)\s+", "", task, flags=re.I)
    if len(task) < 6:
        return None
    return {"every": every, "at": at or "00:00", "weekday": _DAYS.index(word) if word in _DAYS else 0,
            "task": task[0].upper() + task[1:]}


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in
            {"the", "and", "every", "each", "day", "schedule", "scheduled", "please", "that", "this", "check", "stop"}}


def command(text: str, mode: str = "live") -> Optional[Dict[str, Any]]:
    """If the message sets up or stops a schedule, do it and return {answer, schedule}; otherwise None."""
    stop = _STOP.match(text or "")
    if stop and re.search(r"\b(schedul|every|daily|hourly|weekday|morning|brief|digest|check)", stop.group(1), re.I):
        want = _words(stop.group(1))
        mine = [s for s in listing() if s["agent"]["kind"] != "check"]
        scored = sorted(((len(want & _words(s["name"] + " " + s["task"])), s) for s in mine), key=lambda x: -x[0])
        if not scored or scored[0][0] == 0:
            return {"answer": "I found no scheduled task like that. Your scheduled tasks are listed under **Me**."}
        s = scored[0][1]
        delete(s["id"])
        return {"answer": f"Stopped: **{s['name']}** ({s['when']}). It will not run again."}
    plan = from_words(text)
    if not plan:
        m = _WHEN.search(text or "")
        if m and re.search(r"\b(schedule|remind|set up|automatically|each time)\b", text, re.I) and not m.group(4) and (m.group(2) or "").lower() not in ("hour", "morning", "evening", "night") \
                and not m.group(0).lower().startswith("hourly"):
            rest = re.sub(r"\s{2,}", " ", text[:m.start()] + text[m.end():]).strip(" ,.")
            rest = re.sub(r"^(?:please\s+)?(?:schedule|set up|remind me to)[:\s]+", "", rest, flags=re.I).lower()
            return {"answer": f"What time should I do it? Say it again with a time, for example: “every day at 7am {rest}”."}
        return None
    name = plan["task"][:60]
    s = create(name, {"kind": "muse", "key": "muse"}, plan["task"], plan["every"], plan["at"], plan["weekday"], mode)
    first = time.strftime("%a %b %-d at %-I:%M %p", time.localtime(s["next_run"]))
    return {"schedule": s, "answer": f"Scheduled: **{s['task']}**, {s['when']}. The first run is {first}. Each result "
            "goes to your Inbox, with a notification if you have them on. Say \"stop the "
            f"{name[:30].lower()} schedule\" to end it, or switch it off under **Me**."}
