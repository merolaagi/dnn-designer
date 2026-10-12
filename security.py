"""Tougher sign-in: two-factor codes, lockouts, a sign-in log with alerts, and the public address switch.

Two-factor: an authenticator app (iPhone Passwords, Google Authenticator, 1Password…) makes a
6-digit code every 30 seconds from a secret it shares with this Mac (RFC 6238, TOTP). Once it is on,
signing in in a browser needs the password and the code. A paired phone (Saathi) is not affected:
it signs in with its own token, made when you paired it with a code shown here.

Lockouts: 5 wrong sign-ins in 15 minutes lock that name and that address out for 15 minutes; 10
wrong pairing codes in 10 minutes lock the address out.

Log and alerts: every sign-in, good or bad, is kept with the address it came from and the way it
arrived (Cloudflare, Tailscale or this Mac). A sign-in from an address not seen before, and every
lockout, goes to the Inbox and your phone.

Public address: requests that arrive through the Cloudflare tunnel (dnn.fueldeskpro.com) carry
Cloudflare's headers. With the public address turned off they are all refused, before any sign-in,
and only this Mac and Tailscale (Saathi) get in.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import struct
import threading
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import auth

_LOCK = threading.Lock()
_FAILS: Dict[str, List[float]] = {}
LOG = auth.DATA / "auth_log.jsonl"
WINDOW, LIMIT, LOCK_FOR = 15 * 60, 5, 15 * 60
CLAIM_WINDOW, CLAIM_LIMIT = 10 * 60, 10
ISSUER = "Deep Network Designer"


# --------------------------------------------------------------------------
# where a request came from
# --------------------------------------------------------------------------

def origin(request) -> Tuple[str, str]:
    """(address, way): way is cloudflare, tailscale or local."""
    h = request.headers
    if h.get("cf-connecting-ip") or h.get("cf-ray"):
        return (h.get("cf-connecting-ip") or "?").strip(), "cloudflare"
    fwd = (h.get("x-forwarded-for") or "").split(",")[0].strip()
    host = request.client.host if request.client else ""
    ip = fwd or host
    if h.get("tailscale-user-login") or ip.startswith("100.") or ip.startswith("fd7a:115c:a1e0"):
        return ip, "tailscale"
    return ip or "?", "local" if ip in ("127.0.0.1", "::1", "localhost", "") else "network"


def public_on() -> bool:
    return bool(auth._read(auth.SETTINGS, {}).get("public", True))


def set_public(on: bool) -> Dict[str, Any]:
    blob = auth._read(auth.SETTINGS, {})
    blob["public"] = bool(on)
    auth._write(auth.SETTINGS, blob)
    return {"public": public_on()}


# --------------------------------------------------------------------------
# lockouts
# --------------------------------------------------------------------------

def _recent(key: str, window: float) -> List[float]:
    now = time.time()
    got = [t for t in _FAILS.get(key, []) if now - t < window]
    _FAILS[key] = got
    return got


def locked(keys: List[str], window: float = WINDOW, limit: int = LIMIT) -> Optional[int]:
    """Minutes left if any of these is locked out, else None."""
    with _LOCK:
        for k in keys:
            got = _recent(k, window)
            if len(got) >= limit:
                left = LOCK_FOR - (time.time() - got[-1])
                if left > 0:
                    return max(1, int(left // 60) + 1)
    return None


def fail(keys: List[str]) -> None:
    with _LOCK:
        for k in keys:
            _FAILS.setdefault(k, []).append(time.time())


def clear(keys: List[str]) -> None:
    with _LOCK:
        for k in keys:
            _FAILS.pop(k, None)


# --------------------------------------------------------------------------
# the sign-in log
# --------------------------------------------------------------------------

def log(user: str, ok: bool, ip: str, way: str, what: str = "") -> None:
    row = {"t": time.time(), "user": user[:40], "ok": ok, "ip": ip[:64], "via": way, "what": what[:80]}
    with _LOCK:
        try:
            lines = LOG.read_text().splitlines()[-499:] if LOG.exists() else []
        except OSError:
            lines = []
        seen = any(json.loads(x).get("ip") == ip and json.loads(x).get("ok") for x in lines if x.strip())
        auth.DATA.mkdir(exist_ok=True)
        LOG.write_text("\n".join(lines + [json.dumps(row)]) + "\n")
    if (ok and not seen and way != "local") or what.startswith("locked"):
        try:
            import schedules
            title = (f"Sign-in to the Deep Network Designer from a new place: {ip} ({way})" if ok
                     else f"Sign-ins locked out: {ip} ({way})")
            schedules.notify("security", title, f"Account {user or '?'}. {what or ''} If this was not you, turn the "
                             "public address off and change your password (Settings → Security).")
        except Exception:  # noqa: BLE001
            pass


def recent(limit: int = 30) -> List[Dict[str, Any]]:
    try:
        rows = [json.loads(x) for x in LOG.read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        rows = []
    return rows[-limit:][::-1]


# --------------------------------------------------------------------------
# two-factor codes (TOTP, RFC 6238)
# --------------------------------------------------------------------------

def _code(secret: str, counter: int) -> str:
    key = base64.b32decode(secret + "=" * (-len(secret) % 8))
    mac = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    off = mac[-1] & 0x0F
    return f"{(struct.unpack('>I', mac[off:off + 4])[0] & 0x7FFFFFFF) % 1_000_000:06d}"


def code_now(secret: str, at: Optional[float] = None) -> str:
    return _code(secret, int((at or time.time()) // 30))


def _matches(secret: str, code: str, last: int) -> Optional[int]:
    code = "".join(c for c in str(code or "") if c.isdigit())
    if len(code) != 6:
        return None
    now = int(time.time() // 30)
    for c in (now - 1, now, now + 1):                 # a clock a little off is fine
        if c > last and hmac.compare_digest(_code(secret, c), code):
            return c
    return None


def status(name: str) -> Dict[str, Any]:
    u = auth.users().get(name) or {}
    return {"on": bool(u.get("totp_on")), "pending": bool(u.get("totp_pending"))}


def setup(name: str) -> Dict[str, Any]:
    """A new secret, waiting for a code from the app before it counts."""
    secret = base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")
    with auth._LOCK:
        everyone = auth.users()
        everyone[name]["totp_pending"] = secret
        auth._write(auth.USERS, everyone)
    label = urllib.parse.quote(f"{ISSUER}:{name}")
    uri = f"otpauth://totp/{label}?secret={secret}&issuer={urllib.parse.quote(ISSUER)}&digits=6&period=30"
    return {"secret": " ".join(secret[i:i + 4] for i in range(0, len(secret), 4)), "uri": uri}


def confirm(name: str, code: str) -> Dict[str, Any]:
    with auth._LOCK:
        everyone = auth.users()
        u = everyone[name]
        secret = u.get("totp_pending")
        if not secret:
            raise ValueError("Start two-factor first.")
        c = _matches(secret, code, 0)
        if c is None:
            raise ValueError("That code does not match. Check the app shows the Deep Network Designer, then try the "
                             "newest code.")
        u.update(totp=secret, totp_on=True, totp_last=c)
        u.pop("totp_pending", None)
        auth._write(auth.USERS, everyone)
    return status(name)


def verify(name: str, code: str) -> bool:
    """The code for a signed-in step; each code works once."""
    with auth._LOCK:
        everyone = auth.users()
        u = everyone.get(name) or {}
        if not u.get("totp_on"):
            return True
        c = _matches(u["totp"], code, int(u.get("totp_last") or 0))
        if c is None:
            return False
        u["totp_last"] = c
        auth._write(auth.USERS, everyone)
        return True


def disable(name: str, password: str, code: str) -> Dict[str, Any]:
    if not auth.check(name, password):
        raise ValueError("The password is not right.")
    if not verify(name, code):
        raise ValueError("That code does not match.")
    with auth._LOCK:
        everyone = auth.users()
        for k in ("totp", "totp_on", "totp_last", "totp_pending"):
            everyone[name].pop(k, None)
        auth._write(auth.USERS, everyone)
    return status(name)
