"""Feed: a short morning read on the things you follow.

Topics are yours to choose (fuel prices in California, convenience-store news, Chevron, anything).
Each build searches the web for every topic and has the model pick the few items worth your time,
each with a two-line summary and its link. Without a model (a rehearsal, or none chosen), the feed is
the search results as they came. It builds once a day as a check on the schedule ("feed"), or
whenever you pull to refresh.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentlab as AL

DEFAULT_TOPICS = ["California gas prices", "convenience store industry news", "Chevron station news",
                  "fuel wholesale prices"]
PER_TOPIC = 6
KEEP = 3
MAX_TOPICS = 12
_LOCK = threading.Lock()
_BUILDING: Dict[str, bool] = {}
SYSTEM = ("You pick news for a busy gas station owner. From the search results for one topic, choose up to {keep} "
          "items that are recent and would matter to him, skipping adverts, listings and duplicates. Reply with JSON "
          'only: {{"items": [{{"n": <result number>, "title": "<short headline>", "summary": "<two plain sentences: '
          'what happened and why it matters to a station owner>"}}]}}. Use only what the results say; do not add facts.')


def _path() -> Path:
    return AL._dir() / "feed.json"


def _load() -> Dict[str, Any]:
    try:
        return json.loads(_path().read_text())
    except (OSError, ValueError):
        return {"topics": list(DEFAULT_TOPICS), "items": [], "built": None, "error": None}


def _store(d: Dict[str, Any]) -> None:
    _path().write_text(json.dumps(d, indent=1))


def view() -> Dict[str, Any]:
    d = _load()
    return {**d, "building": bool(_BUILDING.get(str(AL._dir())))}


def set_topics(topics: List[str]) -> Dict[str, Any]:
    clean = []
    for t in topics or []:
        t = re.sub(r"\s+", " ", str(t)).strip()[:80]
        if t and t.lower() not in [c.lower() for c in clean]:
            clean.append(t)
    if not clean:
        raise ValueError("Follow at least one topic.")
    if len(clean) > MAX_TOPICS:
        raise ValueError(f"Follow up to {MAX_TOPICS} topics.")
    with _LOCK:
        d = _load()
        d["topics"] = clean
        _store(d)
    return view()


def _search(query: str, count: int) -> List[tuple]:
    ns: Dict[str, Any] = {"json": json, "re": re, "urllib": urllib, "os": __import__("os")}
    exec(AL.WEB_CODE, ns)
    return ns["web_results"](query, count)


def _pick(topic: str, results: List[tuple], live: bool) -> List[Dict[str, Any]]:
    raw = [{"topic": topic, "title": t, "url": u, "summary": s, "source": re.sub(r"^www\.", "", (re.match(r"https?://([^/]+)", u) or [None, ""])[1])}
           for t, u, s in results if u.startswith("http")]
    if not live or not raw:
        return raw[:KEEP]
    import evolve
    listing = "\n".join(f"{i + 1}. {r['title']} ({r['source']})\n   {r['summary']}" for i, r in enumerate(raw))
    try:
        out = evolve.ask(SYSTEM.format(keep=KEEP), f"Topic: {topic}\n\nResults:\n{listing}", 800)
        chosen = (evolve._json_in(out["text"]) or {}).get("items") or []
    except Exception:  # noqa: BLE001 - the plain results are still worth showing
        return raw[:KEEP]
    picked = []
    for c in chosen[:KEEP]:
        try:
            r = raw[int(c["n"]) - 1]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        picked.append({**r, "title": str(c.get("title") or r["title"])[:160], "summary": str(c.get("summary") or r["summary"])[:400]})
    return picked or raw[:KEEP]


def build(live: Optional[bool] = None) -> Dict[str, Any]:
    """Search every topic and keep the best few items of each. Returns the feed."""
    live = AL.live_available() if live is None else live
    key = str(AL._dir())
    _BUILDING[key] = True
    try:
        d = _load()
        items, errors = [], []
        for topic in d["topics"]:
            try:
                items += _pick(topic, _search(topic + " news", PER_TOPIC), live)
            except Exception as exc:  # noqa: BLE001 - one topic failing must not empty the feed
                errors.append(f"{topic}: {type(exc).__name__}")
        seen, unique = set(), []
        for it in items:
            if it["url"] not in seen:
                seen.add(it["url"])
                unique.append({**it, "id": "fd" + str(abs(hash(it["url"])) % 10 ** 10)})
        with _LOCK:
            d = _load()
            if unique or not d["items"]:
                d["items"] = unique
                d["built"] = time.time()
            d["error"] = ("Could not search: " + "; ".join(errors)) if errors and not unique else None
            _store(d)
        return view()
    finally:
        _BUILDING[key] = False


def build_soon() -> Dict[str, Any]:
    if not _BUILDING.get(str(AL._dir())):
        threading.Thread(target=build, daemon=True).start()
        _BUILDING[str(AL._dir())] = True
    return view()


def check(s: Dict[str, Any]) -> None:
    """The scheduled build ("feed" check)."""
    f = build()
    s["last_status"] = "ok" if f["items"] else "error"
    s["last_detail"] = f"{len(f['items'])} items" if f["items"] else (f.get("error") or "nothing found")
