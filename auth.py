"""Accounts and per-user workspaces.

What this does: keeps one person's designs, runs, studies and layout separate
from another's, and requires a password to reach them.

What this does not do, and it matters: make the app safe to expose to the
internet. Importing pasted code, importing a folder, and the blocks and recipes
folders all execute Python by design — that is the point of them. Any account
that can reach those endpoints can run code as this process. Authentication
separates users from each other, not from the machine. Keep this on a network
you trust.

Accounts are optional. With none registered the app behaves exactly as it did
before, on a shared workspace, so nothing breaks for someone who never wanted
accounts. Registering the first account turns authentication on.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXAMPLES = HERE / "examples"
USERS = DATA / "users.json"
SESSIONS = DATA / "sessions.json"

SESSION_COOKIE = "dnn_session"
SESSION_DAYS = 30

_LOCK = threading.Lock()
_current: ContextVar[Optional[str]] = ContextVar("current_user", default=None)

NAME_OK = re.compile(r"^[a-z0-9][a-z0-9._-]{1,30}$")


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------

def _read(path: Path, fallback):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - a corrupt file must not lock anyone out
        return fallback


def _write(path: Path, blob) -> None:
    DATA.mkdir(exist_ok=True)
    path.write_text(json.dumps(blob, indent=1))


def users() -> Dict[str, Any]:
    return _read(USERS, {})


def enabled() -> bool:
    """Authentication is on once somebody has registered."""
    return bool(users())


# --------------------------------------------------------------------------
# passwords
# --------------------------------------------------------------------------

def _hash(password: str, salt: bytes) -> str:
    # scrypt rather than a plain digest: deliberately slow, so a stolen file is
    # not a list of passwords
    return hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1,
                          dklen=32).hex()


def register(name: str, password: str) -> Dict[str, Any]:
    name = (name or "").strip().lower()
    if not NAME_OK.match(name):
        raise ValueError("Pick a name of 2–31 characters: letters, digits, dot, "
                         "dash or underscore, starting with a letter or digit.")
    if len(password or "") < 8:
        raise ValueError("Use a password of at least 8 characters.")
    with _LOCK:
        everyone = users()
        if name in everyone:
            raise ValueError("That name is taken.")
        salt = secrets.token_bytes(16)
        everyone[name] = {
            "salt": salt.hex(),
            "hash": _hash(password, salt),
            "created": time.time(),
            # the first account inherits whatever was in the shared workspace,
            # so turning accounts on does not hide the work already done
            "adopts_legacy": not everyone,
        }
        _write(USERS, everyone)
    home = workspace_for(name)
    home.mkdir(parents=True, exist_ok=True)
    seed(home)
    return {"name": name}


def check(name: str, password: str) -> bool:
    record = users().get((name or "").strip().lower())
    if not record:
        # spend the time anyway, so a wrong name and a wrong password take
        # equally long to answer
        _hash(password or "", b"0" * 16)
        return False
    return secrets.compare_digest(
        _hash(password or "", bytes.fromhex(record["salt"])), record["hash"])


def change_password(name: str, old: str, new: str) -> None:
    if not check(name, old):
        raise ValueError("The current password is not right.")
    if len(new or "") < 8:
        raise ValueError("Use a password of at least 8 characters.")
    with _LOCK:
        everyone = users()
        salt = secrets.token_bytes(16)
        everyone[name]["salt"] = salt.hex()
        everyone[name]["hash"] = _hash(new, salt)
        _write(USERS, everyone)
    # every other session for this account stops working
    with _LOCK:
        live = {token: s for token, s in _read(SESSIONS, {}).items()
                if s.get("user") != name}
        _write(SESSIONS, live)


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------

def open_session(name: str) -> str:
    token = secrets.token_urlsafe(32)
    with _LOCK:
        live = _read(SESSIONS, {})
        cutoff = time.time()
        live = {t: s for t, s in live.items() if s.get("expires", 0) > cutoff}
        live[token] = {"user": name,
                       "expires": time.time() + SESSION_DAYS * 86400}
        _write(SESSIONS, live)
    return token


def close_session(token: str) -> None:
    with _LOCK:
        live = _read(SESSIONS, {})
        live.pop(token, None)
        _write(SESSIONS, live)


def user_for(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    session = _read(SESSIONS, {}).get(token)
    if not session or session.get("expires", 0) <= time.time():
        return None
    if session["user"] not in users():
        return None
    return session["user"]


# --------------------------------------------------------------------------
# workspaces
# --------------------------------------------------------------------------

def workspace_for(name: Optional[str]) -> Path:
    """Where one account's designs, runs and studies live.

    With no accounts, that is the folder layout the app always had, so an
    existing install keeps working untouched.
    """
    if not name:
        return HERE
    record = users().get(name) or {}
    if record.get("adopts_legacy"):
        return HERE
    return DATA / "users" / name


def seed(target: Path) -> int:
    """Give a new workspace the designs the app ships with.

    Only into an empty one. Someone who deleted the examples on purpose should
    not find them back the next time they sign in — `restore_examples` is there
    for wanting them again.
    """
    marker = target / ".seeded"
    if marker.exists():
        return 0
    saved = target / "saved"
    saved.mkdir(parents=True, exist_ok=True)
    copied = 0 if any(saved.iterdir()) else restore_examples(target)
    marker.write_text("the shipped designs were copied in once\n")
    return copied


def restore_examples(target: Path) -> int:
    """Copy the shipped designs in, without overwriting anything of the same name."""
    import shutil

    saved = target / "saved"
    saved.mkdir(parents=True, exist_ok=True)
    copied = 0
    if not EXAMPLES.is_dir():
        return 0
    for source in sorted(EXAMPLES.glob("*.json")):
        destination = saved / source.name
        if destination.exists() or (saved / source.stem).exists():
            continue                      # never tread on the user's own work
        shutil.copy2(source, destination)
        copied += 1
    return copied


def set_current(name: Optional[str]) -> None:
    _current.set(name)


def current() -> Optional[str]:
    return _current.get()


def workspace() -> Path:
    return workspace_for(_current.get())


def sub(kind: str) -> Path:
    """A directory inside the current workspace, created on demand."""
    path = workspace() / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def whoami() -> Dict[str, Any]:
    name = current()
    return {
        "enabled": enabled(),
        "user": name,
        "accounts": sorted(users()),
        "workspace": str(workspace()),
        # said plainly in the UI, because it is easy to assume otherwise
        "warning": ("Accounts separate people's work from each other. They do "
                    "not make this safe to expose publicly: importing code and "
                    "the blocks folder run Python by design."),
    }
