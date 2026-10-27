"""Photos sent in a chat: Saathi looks at the picture, then answers about it.

A photo sent with a message (a screenshot, a page of notes, a chart, a sign, a plant) is kept in the
account's workspace. Before Muse plans, a vision model reads it completely: every word on it, each
diagram and what it shows. That reading goes to the agents with the message, so any agent can answer
about the picture (the Explainer can teach the statistics on a cheat sheet, the general agent can say
what a plant is). Invoices and BOLs still go to their own reader (Hub → Invoices and BOLs).

The model that looks: Claude when Claude is the default model in Settings (it sees pictures well and
fast), otherwise the vision model in Ollama (Settings → Reading invoices).
"""

from __future__ import annotations

import base64
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import agentlab as AL

_ID = re.compile(r"ph[0-9a-f]{10}")
MAX_BYTES = 25 * 1024 * 1024
PROMPT = ("The person sent this picture with the message: \"{question}\".\n"
          "Report exactly what is in the picture, completely, so someone who cannot see it can answer them: copy "
          "every piece of text word for word (titles, headings, labels, numbers, captions, the small print), in "
          "reading order; describe each diagram, chart, table or illustration and what it shows (axes, groups, "
          "shapes, values); say what kind of picture it is (a screenshot of a social media post, a page, a photo "
          "of a thing or a place). Leave out nothing on it. Do not answer the message yet: only report what is there.")


class PhotoError(RuntimeError):
    pass


def _dir() -> Path:
    path = AL._dir() / "photos"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(data: bytes, content_type: str = "") -> Dict[str, Any]:
    if not data:
        raise ValueError("The photo was empty.")
    if len(data) > MAX_BYTES:
        raise ValueError("That photo is over 25 MB.")
    if content_type and not content_type.startswith("image/"):
        raise ValueError("Send a photo (JPEG, PNG, WebP or HEIC).")
    import documents
    pid = "ph" + uuid.uuid4().hex[:10]
    (_dir() / f"{pid}.jpg").write_bytes(documents._prepare(data))
    return {"id": pid, "t": time.time()}


def path_of(pid: str) -> Path:
    if not _ID.fullmatch(str(pid or "")):
        raise KeyError(pid)
    p = _dir() / f"{pid}.jpg"
    if not p.exists():
        raise KeyError(pid)
    return p


def _claude(image: bytes, prompt: str) -> Optional[Dict[str, Any]]:
    """Claude looks, when it is the default model and has a key; None otherwise."""
    import providers
    data = providers.load()
    pid = data["defaults"].get("provider") or ""
    if providers.CATALOG.get(pid, {}).get("kind") != providers.ANTHROPIC or not providers.key_for(pid, data):
        return None
    model = data["defaults"].get("model") or providers.CATALOG[pid]["suggested"][0]
    base = providers.base_url_for(pid, data).rstrip("/")
    reply = providers._request(base + "/v1/messages", {
        "model": model, "max_tokens": 2500, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(image).decode()}},
            {"type": "text", "text": prompt}]}]}, providers._headers(pid, providers.key_for(pid, data)), 180)
    text = "".join(b.get("text", "") for b in reply.get("content", []) if b.get("type") == "text").strip()
    return {"text": text, "model": model}


def _ollama(image: bytes, prompt: str) -> Dict[str, Any]:
    import documents
    import providers
    model = documents.config()["model"]
    reply = providers._request(providers._ollama_root() + "/api/chat", {
        "model": model, "stream": False, "think": False,
        "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(image).decode()]}],
        "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 2500}}, {}, 600)
    return {"text": ((reply.get("message") or {}).get("content") or "").strip(), "model": model}


def read(pid: str, question: str = "") -> Dict[str, Any]:
    """What is in the photo, every word and diagram, read by a vision model."""
    image = path_of(pid).read_bytes()
    prompt = PROMPT.format(question=(question or "What is this?")[:400].replace('"', "'"))
    try:
        out = _claude(image, prompt) or _ollama(image, prompt)
    except Exception as exc:  # noqa: BLE001
        why = str(exc)
        if "not found" in why.lower():
            why = "the vision model isn't in Ollama yet (download it in Settings → Reading invoices)"
        elif "refused" in why or "urlopen" in why:
            why = "Ollama isn't running on the Mac"
        raise PhotoError(f"I couldn't look at the photo: {why}.") from None
    if not out["text"]:
        raise PhotoError("I couldn't make out anything in the photo. Try a sharper one.")
    return out


def note(readings) -> str:
    """The part of an agent's task that says what the photos show."""
    parts = []
    for i, r in enumerate(readings, 1):
        head = "The photo" if len(readings) == 1 else f"Photo {i}"
        parts.append(f"{head} the person sent (read by {r['model']}; you cannot see it, so rely on this):\n{r['text']}")
    return "\n\n".join(parts)
