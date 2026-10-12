"""Talking to Saathi's browser (browser_service.py): starts it when it is not running, then sends it actions.

Used by the Support agent's tools (browser_mcp.py) and by the Browser screen in Saathi (main.py), which
share one Chrome window on this Mac.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("SAATHI_BROWSER_PORT", "8772"))
BASE = f"http://127.0.0.1:{PORT}"
TIMEOUT = 120


class BrowserError(RuntimeError):
    pass


def installed() -> bool:
    return importlib.util.find_spec("playwright") is not None


def _token() -> str:
    path = HERE / "data" / "browser-token"
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _alive() -> bool:
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def ensure() -> None:
    if _alive():
        return
    if not installed():
        raise BrowserError("The browser part is not installed on the Mac yet. In ~/Sites/new_dnn run: "
                           "./serve.sh setup-browser, then ./serve.sh")
    log = open(HERE / "data" / "browser.log", "a")
    subprocess.Popen([sys.executable, str(HERE / "browser_service.py")], cwd=str(HERE), stdout=log, stderr=log,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(60):
        time.sleep(0.25)
        if _alive():
            return
    raise BrowserError("The browser did not start; see data/browser.log on the Mac.")


def call(op: str, **args: Any) -> Dict[str, Any]:
    ensure()
    req = urllib.request.Request(BASE + "/", method="POST", data=json.dumps({"op": op, **args}).encode(),
                                 headers={"content-type": "application/json", "x-saathi-token": _token()})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            why = json.loads(exc.read().decode()).get("error")
        except Exception:  # noqa: BLE001
            why = None
        raise (ValueError if exc.code == 400 else BrowserError)(why or f"The browser said {exc.code}.") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise BrowserError(f"Could not reach the browser: {exc}") from None


def screen() -> bytes:
    ensure()
    req = urllib.request.Request(BASE + "/screen", headers={"x-saathi-token": _token()})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def status() -> Dict[str, Any]:
    if not installed():
        return {"installed": False, "running": False}
    if not _alive():
        return {"installed": True, "running": False}
    return {"installed": True, **call("status")}
