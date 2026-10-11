"""Invoices and bills of lading, read from a photo by a vision model on this computer.

You photograph a vendor invoice or a fuel BOL. A vision model running in Ollama reads it into
fields (vendor, number, date, lines, totals, gallons by grade). Then code, not the model, checks it:
do the lines multiply out, do they add up to the subtotal, does subtotal plus tax make the total.
If the store's data is connected, a BOL is matched against the deliveries recorded there and an
invoice against the invoices table, read-only. Anything still unclear can go to the Station analyst.

Nothing leaves the Mac: the image goes to Ollama on this computer, and is kept in the account's
workspace with what was read from it.
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentlab as AL

HERE = Path(__file__).resolve().parent
SUGGESTED = ["qwen2.5vl:7b", "qwen2.5vl:32b", "gemma3:27b", "llama3.2-vision:11b", "minicpm-v:8b"]
DEFAULTS = {"model": "qwen2.5vl:7b"}
MAX_SIDE = 2000
TIMEOUT = 600
KINDS = ("image/jpeg", "image/png", "image/webp", "image/heic", "image/heif")

SCHEMA = {
    "type": "object",
    "properties": {
        "doc_type": {"type": "string", "enum": ["invoice", "bol", "receipt", "other"]},
        "vendor": {"type": "string"},
        "number": {"type": "string"},
        "date": {"type": "string"},
        "bol_number": {"type": "string"},
        "lines": {"type": "array", "items": {"type": "object", "properties": {
            "description": {"type": "string"}, "item_code": {"type": "string"}, "quantity": {"type": "number"},
            "unit_price": {"type": "number"}, "amount": {"type": "number"}}, "required": ["description"]}},
        "fuel": {"type": "array", "items": {"type": "object", "properties": {
            "grade": {"type": "string"}, "gallons": {"type": "number"}}, "required": ["grade", "gallons"]}},
        "subtotal": {"type": "number"}, "tax": {"type": "number"}, "total": {"type": "number"},
    },
    "required": ["doc_type", "vendor", "lines"],
}
PROMPT = ("Read this document from a gas station's back office. It is a vendor invoice, a fuel bill of lading "
          "(BOL), a receipt, or something else. Copy what is printed; do not guess or calculate. Use null for "
          "anything not on the page. Dates as YYYY-MM-DD. Numbers without currency signs or thousands separators. "
          "For a BOL, put the gallons delivered of each fuel grade in fuel (net gallons if both gross and net are "
          "shown), and the BOL number in bol_number. For an invoice, one entry in lines per printed line item.")


# --------------------------------------------------------------------------
# settings and storage
# --------------------------------------------------------------------------

def _config_path() -> Path:
    path = HERE / "data"
    path.mkdir(exist_ok=True)
    return path / "documents.json"


def config() -> Dict[str, Any]:
    try:
        saved = json.loads(_config_path().read_text())
    except (OSError, ValueError):
        saved = {}
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}


def save_config(change: Dict[str, Any]) -> Dict[str, Any]:
    c = config()
    model = str(change.get("model") or "").strip()
    if model:
        if not re.fullmatch(r"[\w.\-:/]+", model):
            raise ValueError("A model name looks like qwen2.5vl:7b.")
        c["model"] = model
    _config_path().write_text(json.dumps(c, indent=1))
    return settings()


def installed() -> List[str]:
    import providers
    try:
        tags = providers._request(providers._ollama_root() + "/api/tags", None, {}, 5)
        return [m.get("name") for m in tags.get("models") or []]
    except Exception:  # noqa: BLE001 - Ollama not running: say so where it is shown
        return []


def settings() -> Dict[str, Any]:
    c, have = config(), installed()
    return {**c, "suggested": SUGGESTED, "installed": [m for m in have if m],
            "ready": c["model"] in have or (c["model"] + ":latest") in have, "ollama": bool(have)}


def _dir() -> Path:
    path = AL._dir() / "documents"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _doc_dir(doc_id: str) -> Path:
    if not re.fullmatch(r"dc[0-9a-f]{10}", doc_id or ""):
        raise KeyError(doc_id)
    return _dir() / doc_id


def load(doc_id: str) -> Dict[str, Any]:
    path = _doc_dir(doc_id) / "doc.json"
    if not path.exists():
        raise KeyError(doc_id)
    d = json.loads(path.read_text())
    if d.get("status") == "reading" and time.time() - d.get("started", 0) > TIMEOUT + 60:
        d.update(status="error", error="The server stopped while this was being read. Read it again.")
    return d


def _save(d: Dict[str, Any]) -> None:
    (_doc_dir(d["id"]) / "doc.json").write_text(json.dumps(d, indent=1, default=str))


def listing() -> List[Dict[str, Any]]:
    out = []
    for path in sorted(_dir().glob("dc*/doc.json"), key=lambda p: -p.stat().st_mtime)[:100]:
        try:
            d = load(path.parent.name)
        except (KeyError, ValueError):
            continue
        x = d.get("extracted") or {}
        out.append({"id": d["id"], "status": d["status"], "t": d["t"], "name": d.get("name"),
                    "doc_type": x.get("doc_type"), "vendor": x.get("vendor"), "number": x.get("number") or x.get("bol_number"),
                    "date": x.get("date"), "total": x.get("total"),
                    "problems": sum(1 for c in d.get("checks") or [] if c.get("ok") is False)})
    return out


def image_path(doc_id: str) -> Path:
    found = sorted(_doc_dir(doc_id).glob("image.*"))
    if not found:
        raise KeyError(doc_id)
    return found[0]


def delete(doc_id: str) -> Dict[str, Any]:
    folder = _doc_dir(doc_id)
    if not folder.exists():
        raise KeyError(doc_id)
    for f in folder.iterdir():
        f.unlink()
    folder.rmdir()
    return {"ok": True}


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def _prepare(data: bytes) -> bytes:
    """A JPEG no larger than MAX_SIDE on its long side, upright: what the model reads best and fastest."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return data
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        out = io.BytesIO()
        img.save(out, "JPEG", quality=88)
        return out.getvalue()
    except Exception:  # noqa: BLE001 - a format Pillow cannot open: let the model try it as sent
        return data


def submit(data: bytes, filename: str = "", content_type: str = "", background: bool = True) -> Dict[str, Any]:
    if not data:
        raise ValueError("The photo was empty.")
    if len(data) > 25 * 1024 * 1024:
        raise ValueError("That file is over 25 MB: take the photo again at a smaller size.")
    if content_type and content_type not in KINDS and not content_type.startswith("image/"):
        raise ValueError("Send a photo (JPEG, PNG, WebP or HEIC).")
    doc_id = "dc" + uuid.uuid4().hex[:10]
    folder = _dir() / doc_id
    folder.mkdir()
    (folder / "image.jpg").write_bytes(_prepare(data))
    d = {"id": doc_id, "t": time.time(), "name": Path(filename or "photo").name[:120], "status": "reading",
         "model": config()["model"], "started": time.time(), "extracted": None, "checks": [], "matches": [],
         "error": None, "seconds": None}
    _save(d)
    if background:
        import contextvars
        threading.Thread(target=contextvars.copy_context().run, args=(_read, d), daemon=True).start()
    else:
        _read(d)
    return load(doc_id)


def reread(doc_id: str, background: bool = True) -> Dict[str, Any]:
    d = load(doc_id)
    if d["status"] == "reading":
        raise ValueError("It is still being read.")
    d.update(status="reading", started=time.time(), model=config()["model"], error=None, extracted=None, checks=[],
             matches=[])
    _save(d)
    if background:
        import contextvars
        threading.Thread(target=contextvars.copy_context().run, args=(_read, d), daemon=True).start()
    else:
        _read(d)
    return load(doc_id)


def ask_model(image: bytes, model: str) -> Dict[str, Any]:
    """One call to the vision model through Ollama's own API, asking for JSON in SCHEMA's shape."""
    import providers
    reply = providers._request(providers._ollama_root() + "/api/chat", {
        "model": model, "stream": False, "think": False, "format": SCHEMA,
        "messages": [{"role": "user", "content": PROMPT, "images": [base64.b64encode(image).decode()]}],
        "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 4096}}, {}, TIMEOUT)
    text = ((reply.get("message") or {}).get("content") or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"The model did not return the fields asked for: {text[:200]}") from None


def _read(d: Dict[str, Any]) -> None:
    started = time.time()
    try:
        raw = ask_model(image_path(d["id"]).read_bytes(), d["model"])
        d["extracted"] = normalise(raw)
        d["checks"] = arithmetic(d["extracted"])
        d["matches"] = match(d["extracted"])
        d["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        why = str(exc)
        if "not found" in why.lower() and "model" in why.lower():
            why = f"The vision model {d['model']} is not in Ollama yet: download it in Settings."
        elif "Connection refused" in why or "urlopen error" in why:
            why = "Ollama is not running on this computer: start it in Settings."
        d.update(status="error", error=f"{why}"[:500])
    finally:
        d["seconds"] = round(time.time() - started, 1)
        _save(d)


# --------------------------------------------------------------------------
# checking what was read: by rule, not by the model
# --------------------------------------------------------------------------

def _number(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    text = str(v).strip().replace(",", "").replace("$", "")
    neg = text.startswith("(") and text.endswith(")")
    try:
        x = float(text.strip("()"))
        return -x if neg else x
    except ValueError:
        return None


def _date(v) -> Optional[str]:
    if not v:
        return None
    text = str(v).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%b %d, %Y", "%B %d, %Y", "%d %b %Y"):
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def normalise(raw: Dict[str, Any]) -> Dict[str, Any]:
    text = lambda v: (str(v).strip() if v not in (None, "") and str(v).strip().lower() not in ("null", "none", "n/a") else None)
    lines = []
    for line in raw.get("lines") or []:
        if not isinstance(line, dict):
            continue
        lines.append({"description": text(line.get("description")) or "", "item_code": text(line.get("item_code")),
                      "quantity": _number(line.get("quantity")), "unit_price": _number(line.get("unit_price")),
                      "amount": _number(line.get("amount"))})
    fuel = [{"grade": text(f.get("grade")) or "?", "gallons": _number(f.get("gallons"))}
            for f in raw.get("fuel") or [] if isinstance(f, dict) and _number(f.get("gallons"))]
    kind = raw.get("doc_type") if raw.get("doc_type") in ("invoice", "bol", "receipt", "other") else "other"
    return {"doc_type": kind, "vendor": text(raw.get("vendor")), "number": text(raw.get("number")),
            "date": _date(raw.get("date")), "date_printed": text(raw.get("date")), "bol_number": text(raw.get("bol_number")),
            "lines": lines, "fuel": fuel, "subtotal": _number(raw.get("subtotal")), "tax": _number(raw.get("tax")),
            "total": _number(raw.get("total"))}


def _close(a: float, b: float, rel: float = 0.005, floor: float = 0.02) -> bool:
    return abs(a - b) <= max(floor, rel * max(abs(a), abs(b)))


def arithmetic(x: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What can be checked on the page alone. ok None means it could not be checked."""
    out = []
    for i, line in enumerate(x["lines"], 1):
        q, p, a = line["quantity"], line["unit_price"], line["amount"]
        if None not in (q, p, a):
            ok = _close(q * p, a)
            out.append({"check": f"Line {i}: {q:g} × {p:g} = {a:,.2f}", "ok": ok,
                        "detail": None if ok else f"{q:g} × {p:g} is {q * p:,.2f}, not {a:,.2f}"})
    amounts = [line["amount"] for line in x["lines"] if line["amount"] is not None]
    if amounts and x["subtotal"] is not None:
        s = sum(amounts)
        ok = _close(s, x["subtotal"])
        out.append({"check": f"Lines add up to the subtotal ({x['subtotal']:,.2f})", "ok": ok,
                    "detail": None if ok else f"the lines add up to {s:,.2f}" + (
                        "; a line may be missing from what was read" if s < x["subtotal"] else "")})
    if x["subtotal"] is not None and x["total"] is not None:
        expect = x["subtotal"] + (x["tax"] or 0)
        ok = _close(expect, x["total"])
        out.append({"check": "Subtotal" + (" plus tax" if x["tax"] else "") + f" makes the total ({x['total']:,.2f})",
                    "ok": ok, "detail": None if ok else f"they make {expect:,.2f}"})
    elif amounts and x["total"] is not None and x["subtotal"] is None:
        s = sum(amounts) + (x["tax"] or 0)
        ok = _close(s, x["total"])
        out.append({"check": f"Lines{' plus tax' if x['tax'] else ''} add up to the total ({x['total']:,.2f})", "ok": ok,
                    "detail": None if ok else f"they make {s:,.2f}"})
    if x["date"] is None:
        out.append({"check": "A date was read", "ok": False if x.get("date_printed") else None,
                    "detail": f"printed as “{x['date_printed']}”" if x.get("date_printed") else "no date found"})
    return out


# --------------------------------------------------------------------------
# matching against the store's own records (read-only)
# --------------------------------------------------------------------------

def _q(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def match(x: Dict[str, Any]) -> List[Dict[str, Any]]:
    import station
    if not station.view().get("configured"):
        return [{"what": "Store records", "ok": None, "detail": "Connect your store's data in Settings to match this "
                 "against the deliveries and invoices recorded there."}]
    try:
        if x["doc_type"] == "bol" or x["fuel"]:
            return _match_bol(x)
        if x["doc_type"] in ("invoice", "receipt"):
            return _match_invoice(x)
    except (RuntimeError, ValueError) as exc:
        return [{"what": "Store records", "ok": None, "detail": f"Could not read the store's data: {exc}"}]
    return []


def _match_bol(x: Dict[str, Any]) -> List[Dict[str, Any]]:
    import station
    total = sum(f["gallons"] for f in x["fuel"]) if x["fuel"] else None
    where = [station.MINE]
    if x["date"]:
        where.append(f"business_date BETWEEN {_q(x['date'])}::date - 3 AND {_q(x['date'])}::date + 3")
    rows = station.query("SELECT business_date, coalesce(grade_name, grade_code) AS grade, gallons, bol_number, supplier "
                         f"FROM fuel_deliveries WHERE {' AND '.join(where)} ORDER BY business_date DESC LIMIT 40")["rows"]
    bol = (x.get("bol_number") or x.get("number") or "").strip().lower()
    by_bol = [r for r in rows if bol and (r[3] or "").strip().lower() == bol]
    if by_bol:
        got = sum(float(r[2]) for r in by_bol)
        ok = total is None or _close(got, total, 0.01, 1)
        return [{"what": f"BOL {bol.upper()} in fuel_deliveries", "ok": ok,
                 "detail": f"recorded on {by_bol[0][0]}: {got:,.1f} gal" + ("" if ok else f", against {total:,.1f} gal on the BOL")}]
    out = []
    for f in x["fuel"]:
        near = [r for r in rows if _close(float(r[2]), f["gallons"], 0.01, 1)]
        out.append({"what": f"{f['grade']}: {f['gallons']:,.1f} gal", "ok": bool(near),
                    "detail": f"a delivery of {float(near[0][2]):,.1f} gal ({near[0][1]}) is recorded on {near[0][0]}"
                    + (f", BOL {near[0][3]}" if near[0][3] else "") if near else
                    ("no delivery of that size recorded within 3 days of " + (x["date"] or "the BOL's date")
                     + ": it may not have been entered yet")})
    if not x["fuel"]:
        out.append({"what": "Fuel delivered", "ok": None, "detail": "no gallons were read from the BOL"})
    return out


_COLS = {"number": r"^(invoice_?(number|no|num)|number|doc_?number|reference)$",
         "total": r"^(total|total_amount|amount|grand_total|invoice_total|amount_due)$",
         "date": r"^(invoice_date|date|business_date|issued_on|doc_date)$",
         "vendor": r"^(vendor_name|vendor|supplier|supplier_name)$"}


def _match_invoice(x: Dict[str, Any]) -> List[Dict[str, Any]]:
    import station
    try:
        cols = [c["name"] for c in station.describe("invoices")["columns"]]
    except ValueError:
        return [{"what": "Invoices", "ok": None, "detail": "this login cannot read an invoices table"}]
    pick = {k: next((c for c in cols if re.match(rx, c)), None) for k, rx in _COLS.items()}
    if not pick["number"] and not (pick["total"] and pick["date"]):
        return [{"what": "Invoices", "ok": None, "detail": "the invoices table has no column this can match on "
                 f"({', '.join(cols[:12])}); ask the Station analyst instead"}]
    shown = [c for c in (pick["number"], pick["date"], pick["vendor"], pick["total"]) if c]
    base = f"SELECT {', '.join(shown)} FROM invoices WHERE {station.MINE}"
    rows = []
    if pick["number"] and x["number"]:
        rows = station.query(f"{base} AND lower({pick['number']}::text) = {_q(x['number'].lower())} LIMIT 5")["rows"]
    if not rows and pick["total"] and pick["date"] and x["total"] is not None and x["date"]:
        rows = station.query(f"{base} AND {pick['date']}::date BETWEEN {_q(x['date'])}::date - 5 AND "
                             f"{_q(x['date'])}::date + 5 AND abs({pick['total']}::numeric - {x['total']}) <= "
                             f"greatest(0.02, {x['total']} * 0.005) LIMIT 5")["rows"]
    if not rows:
        return [{"what": "Invoices", "ok": False, "detail": "no invoice with this number"
                 + (" or this total and date" if pick["total"] and pick["date"] else "")
                 + " is recorded: it may not have been entered yet"}]
    found = dict(zip(shown, rows[0]))
    detail = ", ".join(f"{k} {v}" for k, v in found.items() if v)
    ok = True
    if pick["total"] and x["total"] is not None and found.get(pick["total"]) not in (None, ""):
        ok = _close(float(found[pick["total"]]), x["total"])
        if not ok:
            detail += f"; the recorded total is not the {x['total']:,.2f} on the paper"
    return [{"what": "Invoice in the store's records", "ok": ok, "detail": detail}]


def analyst_task(d: Dict[str, Any]) -> str:
    x = d.get("extracted") or {}
    return ("I photographed this " + {"bol": "fuel bill of lading", "invoice": "vendor invoice"}.get(x.get("doc_type"), "document")
            + ". Check it against my store's records (deliveries, purchase orders, receiving and invoices, as "
            "fits) and tell me what matches, what does not, and anything I should follow up with the vendor.\n\n"
            "What was read from it:\n" + json.dumps({k: v for k, v in x.items() if v not in (None, [], "")}, indent=1)
            + ("\n\nChecks already made on the page:\n" + "\n".join(
                f"- {c['check']}: {'ok' if c['ok'] else 'NOT ok' if c['ok'] is False else 'not checked'}"
                + (f" ({c['detail']})" if c.get("detail") else "") for c in d.get("checks") or []) if d.get("checks") else ""))
