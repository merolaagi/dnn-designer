"""The browser Saathi uses on your behalf: one Chrome window on this Mac, driven through Playwright.

It runs as its own small process (started on first use by browser.py) because a browser has to be
driven from a single thread, and both the Support agent and Saathi's Browser screen use the same
window: the agent chats, and you step in from your phone to sign in.

The Chrome profile is kept in data/browser-profile, so a site you signed in to stays signed in, the
way it does in your own browser. Your passwords are never stored by Saathi: you type them into the
site yourself, and only a request that comes from you (Saathi's Browser screen) may type into a
password field. The agent's requests may not.

It listens on 127.0.0.1 only, and every request must carry the token in data/browser-token.

    python3 browser_service.py            (browser.py starts it; you do not need to)
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PORT = int(os.environ.get("SAATHI_BROWSER_PORT", "8772"))
VIEW = {"width": 1100, "height": 900}
TEXT_LIMIT = 6000

# what counts as something to click or type into, in every frame (chat windows are often in an iframe)
TAG_JS = r"""
() => {
  const seen = [];
  const vis = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 2 && r.height > 2 && s.visibility !== 'hidden' && s.display !== 'none' && r.bottom > 0 && r.top < innerHeight * 3; };
  const label = el => (el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('title') ||
    (el.labels && el.labels[0] && el.labels[0].innerText) || el.innerText || el.value || el.getAttribute('name') || '')
    .replace(/\s+/g, ' ').trim().slice(0, 80);
  const q = 'a[href], button, input:not([type=hidden]), textarea, select, [role=button], [role=link], [role=textbox], [contenteditable=""], [contenteditable=true]';
  let n = 0;
  document.querySelectorAll('[data-saathi]').forEach(e => e.removeAttribute('data-saathi'));
  for (const el of document.querySelectorAll(q)) {
    if (!vis(el) || el.disabled) continue;
    n += 1; el.setAttribute('data-saathi', String(n));
    const tag = el.tagName.toLowerCase(), type = (el.getAttribute('type') || '').toLowerCase();
    const kind = tag === 'a' || el.getAttribute('role') === 'link' ? 'link'
      : (tag === 'input' && !['button', 'submit', 'checkbox', 'radio'].includes(type)) || tag === 'textarea' || el.isContentEditable || el.getAttribute('role') === 'textbox' ? 'field'
      : tag === 'select' ? 'choice' : 'button';
    seen.push({n, kind, label: label(el), password: type === 'password', focused: document.activeElement === el});
    if (seen.length >= 120) break;
  }
  const text = (document.body ? document.body.innerText : '').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
  return {elements: seen, text};
}
"""


class Browser:
    def __init__(self) -> None:
        self.pw = None
        self.ctx = None
        self.page = None
        self.log: List[Dict[str, Any]] = []

    # ---- the window ------------------------------------------------------
    def start(self) -> None:
        if self.page is not None and not self.page.is_closed():
            return
        from playwright.sync_api import sync_playwright
        if self.pw is None:
            self.pw = sync_playwright().start()
        profile = DATA / "browser-profile"
        profile.mkdir(parents=True, exist_ok=True)
        headless = os.environ.get("SAATHI_BROWSER_HEADLESS") == "1"
        last = None
        for channel in ("chrome", None):            # your Google Chrome if it is installed, else Playwright's Chromium
            try:
                self.ctx = self.pw.chromium.launch_persistent_context(
                    str(profile), channel=channel, headless=headless, viewport=VIEW,
                    args=["--disable-blink-features=AutomationControlled"])
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
        else:
            raise RuntimeError("Could not start a browser. Install Google Chrome, or run: python3 -m playwright install "
                               f"chromium ({last})")
        self.page = self.ctx.pages[0] if self.ctx.pages else self.ctx.new_page()

    def _note(self, what: str) -> None:
        self.log.append({"t": time.time(), "what": what[:300]})
        self.log = self.log[-200:]

    def _settle(self) -> None:
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=8000)
        except Exception:  # noqa: BLE001
            pass
        self.page.wait_for_timeout(600)

    def _frames(self):
        return [f for f in self.page.frames if not f.is_detached()]

    def _locate(self, ident: str):
        try:
            fi, n = str(ident).split(".", 1)
            frame = self._frames()[int(fi)]
        except (ValueError, IndexError):
            raise ValueError(f"There is no element {ident} on the page now: call look again for fresh ids.")
        loc = frame.locator(f'[data-saathi="{int(n)}"]')
        if loc.count() == 0:
            raise ValueError(f"Element {ident} is gone (the page changed): call look again.")
        return loc.first

    # ---- what the agent and you can do -------------------------------------
    def status(self) -> Dict[str, Any]:
        alive = self.page is not None and not self.page.is_closed()
        return {"running": alive, "url": self.page.url if alive else None,
                "title": self.page.title() if alive else None, "log": self.log[-20:]}

    def open(self, url: str) -> Dict[str, Any]:
        self.start()
        url = str(url or "").strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        self.page.goto(url, wait_until="domcontentloaded", timeout=45000)
        self._settle()
        self._note(f"opened {url}")
        return self.look()

    def look(self) -> Dict[str, Any]:
        self.start()
        elements, texts = [], []
        for fi, frame in enumerate(self._frames()):
            if fi:
                try:
                    if not frame.frame_element().is_visible():
                        continue                        # a chat window that is not open yet
                except Exception:  # noqa: BLE001
                    continue
            try:
                got = frame.evaluate(TAG_JS)
            except Exception:  # noqa: BLE001 - a frame that is loading or cross-origin-blocked
                continue
            for e in got["elements"]:
                elements.append({"id": f"{fi}.{e['n']}", "kind": e["kind"], "label": e["label"],
                                 **({"password": True} if e["password"] else {}), **({"focused": True} if e["focused"] else {})})
            if got["text"]:
                texts.append(got["text"] if fi == 0 else f"[inside a frame]\n{got['text']}")
        text = "\n\n".join(texts)
        if len(text) > TEXT_LIMIT:
            text = "[... earlier text cut]\n" + text[-TEXT_LIMIT:]   # the newest chat lines are at the end
        return {"url": self.page.url, "title": self.page.title(), "text": text, "elements": elements[:150],
                "asks_password": any(e.get("password") for e in elements)}

    def click(self, ident: Optional[str] = None, x: Optional[float] = None, y: Optional[float] = None) -> Dict[str, Any]:
        self.start()
        if ident:
            el = self._locate(ident)
            el.scroll_into_view_if_needed(timeout=5000)
            el.click(timeout=8000)
            self._note(f"clicked {ident}")
        elif x is not None and y is not None:
            self.page.mouse.click(float(x), float(y))
            self._note(f"tapped at {int(x)},{int(y)}")
        else:
            raise ValueError("Say what to click: an element id from look, or a point.")
        self._settle()
        return {"ok": True}

    def type(self, text: str, ident: Optional[str] = None, submit: bool = False, from_person: bool = False) -> Dict[str, Any]:
        self.start()
        text = str(text or "")
        if ident:
            el = self._locate(ident)
            is_pw = el.evaluate("e => (e.getAttribute('type') || '').toLowerCase() === 'password'")
            if is_pw and not from_person:
                raise ValueError("That is a password field. Only the person types passwords: ask them to sign in "
                                 "from Saathi (Hub → Browser).")
            el.click(timeout=8000)
            try:
                el.fill(text, timeout=8000)
            except Exception:  # noqa: BLE001 - contenteditable chat boxes
                self.page.keyboard.type(text, delay=15)
        else:
            focused_pw = self.page.evaluate(
                "() => { let d = document; let e = d.activeElement; while (e && e.tagName === 'IFRAME') "
                "{ try { e = e.contentDocument.activeElement } catch (x) { break } } "
                "return !!e && (e.getAttribute('type') || '').toLowerCase() === 'password'; }")
            if focused_pw and not from_person:
                raise ValueError("The box in focus is a password field. Only the person types passwords.")
            self.page.keyboard.type(text, delay=15)
        if submit:
            self.page.keyboard.press("Enter")
        self._note(("you typed into " if from_person else "typed into ") + (ident or "the focused box")
                   + ("" if from_person else f": {text[:120]}") + (" and pressed Enter" if submit else ""))
        self._settle()
        return {"ok": True}

    def press(self, key: str) -> Dict[str, Any]:
        self.start()
        if key not in ("Enter", "Tab", "Escape", "Backspace", "ArrowDown", "ArrowUp", "PageDown", "PageUp", "Space"):
            raise ValueError("Keys: Enter, Tab, Escape, Backspace, ArrowDown, ArrowUp, PageDown, PageUp, Space.")
        self.page.keyboard.press(" " if key == "Space" else key)
        self._settle()
        return {"ok": True}

    def scroll(self, direction: str = "down") -> Dict[str, Any]:
        self.start()
        self.page.mouse.wheel(0, 700 if direction != "up" else -700)
        self.page.wait_for_timeout(400)
        return {"ok": True}

    def back(self) -> Dict[str, Any]:
        self.start()
        self.page.go_back(timeout=20000)
        self._settle()
        return {"ok": True}

    def wait(self, seconds: float = 10) -> Dict[str, Any]:
        self.start()
        self.page.wait_for_timeout(max(1, min(60, float(seconds or 10))) * 1000)
        return {"ok": True}

    def screen(self) -> bytes:
        self.start()
        return self.page.screenshot(type="jpeg", quality=60)

    def close(self) -> Dict[str, Any]:
        if self.ctx is not None:
            try:
                self.ctx.close()
            except Exception:  # noqa: BLE001
                pass
        self.ctx = self.page = None
        return {"ok": True}


BROWSER = Browser()
OPS = {"status", "open", "look", "click", "type", "press", "scroll", "back", "wait", "close"}


def token() -> str:
    path = DATA / "browser-token"
    DATA.mkdir(exist_ok=True)
    if not path.exists():
        path.write_text(secrets.token_urlsafe(24))
        os.chmod(path, 0o600)
    return path.read_text().strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _reply(self, code: int, body: bytes, kind: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("content-type", kind)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self) -> bool:
        if self.headers.get("x-saathi-token") != TOKEN:
            self._reply(403, b'{"error": "wrong token"}')
            return False
        return True

    def do_GET(self):
        if self.path == "/health":
            return self._reply(200, b'{"ok": true}')
        if not self._authorised():
            return
        if self.path.startswith("/screen"):
            try:
                return self._reply(200, BROWSER.screen(), "image/jpeg")
            except Exception as exc:  # noqa: BLE001
                return self._reply(500, json.dumps({"error": str(exc)}).encode())
        self._reply(404, b'{"error": "no such path"}')

    def do_POST(self):
        if not self._authorised():
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            op = req.pop("op", "")
            if op not in OPS:
                raise ValueError(f"Unknown browser action {op!r}.")
            if op == "type":
                out = BROWSER.type(req.get("text", ""), req.get("id"), bool(req.get("submit")), bool(req.get("from_person")))
            elif op == "click":
                out = BROWSER.click(req.get("id"), req.get("x"), req.get("y"))
            elif op in ("open",):
                out = BROWSER.open(req.get("url", ""))
            elif op == "press":
                out = BROWSER.press(req.get("key", "Enter"))
            elif op == "scroll":
                out = BROWSER.scroll(req.get("direction", "down"))
            elif op == "wait":
                out = BROWSER.wait(req.get("seconds", 10))
            else:
                out = getattr(BROWSER, op)()
            self._reply(200, json.dumps(out).encode())
        except ValueError as exc:
            self._reply(400, json.dumps({"error": str(exc)}).encode())
        except Exception as exc:  # noqa: BLE001
            self._reply(500, json.dumps({"error": f"{type(exc).__name__}: {exc}",
                                         "trace": traceback.format_exc()[-800:]}).encode())


TOKEN = token()

if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)        # one request at a time: the browser has one thread
    print(f"Saathi's browser is listening on 127.0.0.1:{PORT}", flush=True)
    try:
        server.serve_forever()
    finally:
        BROWSER.close()
        sys.exit(0)
