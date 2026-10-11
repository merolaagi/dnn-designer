"""What Muse knows about you: a short list of facts you wrote or told it to remember.

Every Muse request and every conversation with an agent starts with this list, so you do not
repeat your store's name, which tenant is yours, who your fuel supplier is or how you like
answers. Nothing is added unless you say so: write a line yourself, or tell Muse
"remember: …" (and "forget: …" to take one out). Each account has its own list.

Benchmarks do not read it, so their scores stay comparable.
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

MAX_FACTS = 60
MAX_CHARS = 300
_LOCK = threading.Lock()
_REMEMBER = re.compile(r"^\s*(?:please\s+)?(?:remember|note|keep in mind)(?:\s+that)?\s*[:,\-]?\s+(.+)$", re.I | re.S)
_FORGET = re.compile(r"^\s*(?:please\s+)?forget(?:\s+that)?\s*[:,\-]?\s+(.+)$", re.I | re.S)
_STOP = set("the a an my our is are was of to and that this it in on for at with i we me".split())


def _path() -> Path:
    return AL._dir() / "profile.json"


def listing() -> List[Dict[str, Any]]:
    try:
        return json.loads(_path().read_text())
    except (OSError, ValueError):
        return []


def _store(facts: List[Dict[str, Any]]) -> None:
    _path().write_text(json.dumps(facts, indent=1))


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip().rstrip(".") + "."
    if len(text) < 3:
        raise ValueError("Write the fact first.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Keep each fact under {MAX_CHARS} characters: split a long one in two.")
    return text[0].upper() + text[1:]


def add(text: str, by: str = "you") -> Dict[str, Any]:
    text = _clean(text)
    with _LOCK:
        facts = listing()
        if any(f["text"].lower() == text.lower() for f in facts):
            return next(f for f in facts if f["text"].lower() == text.lower())
        if len(facts) >= MAX_FACTS:
            raise ValueError(f"The profile holds {MAX_FACTS} facts: remove one first.")
        fact = {"id": "pf" + uuid.uuid4().hex[:8], "text": text, "by": by, "t": time.time()}
        facts.append(fact)
        _store(facts)
    return fact


def update(fact_id: str, text: str) -> Dict[str, Any]:
    text = _clean(text)
    with _LOCK:
        facts = listing()
        for f in facts:
            if f["id"] == fact_id:
                f.update(text=text, t=time.time())
                _store(facts)
                return f
    raise KeyError(fact_id)


def remove(fact_id: str) -> Dict[str, Any]:
    with _LOCK:
        facts = listing()
        kept = [f for f in facts if f["id"] != fact_id]
        if len(kept) == len(facts):
            raise KeyError(fact_id)
        _store(kept)
    return {"ok": True}


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9#]+", text.lower()) if w not in _STOP and len(w) > 1}


def forget(about: str) -> List[Dict[str, Any]]:
    """Remove the facts that best match the words given: those sharing most of them."""
    want = _words(about)
    if not want:
        return []
    with _LOCK:
        facts = listing()
        scored = [(len(want & _words(f["text"])) / len(want), f) for f in facts]
        best = max((s for s, _ in scored), default=0)
        if best < 0.5:
            return []
        gone = [f for s, f in scored if s == best]
        _store([f for f in facts if f not in gone])
    return gone


def command(text: str) -> Optional[str]:
    """If the message is "remember: …" or "forget: …", do it and return the reply; otherwise None."""
    m = _REMEMBER.match(text or "")
    if m and "?" not in m.group(1)[-2:]:
        fact = add(m.group(1), "muse")
        return f"Noted. From now on every request starts with this:\n\n- {fact['text']}\n\nChange or remove it under **Me** (or Settings → What Muse knows about you)."
    m = _FORGET.match(text or "")
    if m:
        gone = forget(m.group(1))
        if not gone:
            return "I found nothing like that in what I know about you. The list is under **Me**."
        return "Forgotten:\n\n" + "\n".join(f"- {f['text']}" for f in gone)
    return None


def as_text() -> str:
    facts = listing()
    if not facts:
        return ""
    return ("About the person you are working for (their own profile; follow it unless this request says "
            "otherwise):\n" + "\n".join(f"- {f['text']}" for f in facts))


def with_profile(task: str) -> str:
    about = as_text()
    return f"{task}\n\n{about}" if about else task
