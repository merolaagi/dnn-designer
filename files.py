"""Files: what Saathi has made or read for you, in one list.

Briefs (the PDF that came with each morning brief), answers you saved as PDFs, email drafts it put
in your Drafts folder, scans of invoices and bills of lading, and the store report (always the
latest). A brief's PDF is kept on disk; an answer's PDF is made again from its conversation, and a
draft lives in your mailbox, so those rows only point to them.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentlab as AL

KEEP = 300
_LOCK = threading.Lock()


def _dir() -> Path:
    path = AL._dir() / "files"
    path.mkdir(exist_ok=True)
    return path


def _log() -> Path:
    return _dir() / "log.jsonl"


def record(kind: str, title: str, *, data: Optional[bytes] = None, ext: str = "pdf", url: Optional[str] = None,
           conv: Optional[str] = None, detail: str = "", key: Optional[str] = None) -> Dict[str, Any]:
    """Add one file. A row with the same key replaces the older one (saving the same answer twice)."""
    row = {"id": "fl" + uuid.uuid4().hex[:8], "kind": kind, "title": title[:200], "t": time.time(), "url": url,
           "conv": conv, "detail": detail[:300], "key": key, "ext": ext}
    if data is not None:
        (_dir() / f"{row['id']}.{ext}").write_bytes(data)
        row["url"] = f"/api/files/{row['id']}"
    with _LOCK:
        rows = _rows()
        if key:
            rows = [r for r in rows if r.get("key") != key]
        rows.append(row)
        for old in rows[:-KEEP]:
            (_dir() / f"{old['id']}.{old.get('ext', 'pdf')}").unlink(missing_ok=True)
        _log().write_text("".join(json.dumps(r) + "\n" for r in rows[-KEEP:]))
    return row


def _rows() -> List[Dict[str, Any]]:
    try:
        return [json.loads(x) for x in _log().read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return []


def path_of(fid: str) -> Path:
    for r in _rows():
        if r["id"] == fid:
            p = _dir() / f"{fid}.{r.get('ext', 'pdf')}"
            if p.exists():
                return p
    raise KeyError(fid)


def listing() -> List[Dict[str, Any]]:
    out = [{**r, "has_file": (_dir() / f"{r['id']}.{r.get('ext', 'pdf')}").exists()} for r in _rows()]
    try:
        import documents
        for d in documents.listing():
            label = " ".join(x for x in [d.get("vendor"), (d.get("doc_type") or "").replace("_", " "), d.get("number")] if x)
            out.append({"id": d["id"], "kind": "scan", "title": label or d.get("name") or "Scan", "t": d["t"],
                        "url": f"/api/documents/{d['id']}/image", "doc": d["id"], "has_file": True,
                        "detail": (f"{d['problems']} to check" if d.get("problems") else d["status"])})
    except Exception:  # noqa: BLE001
        pass
    out.sort(key=lambda r: -(r.get("t") or 0))
    out.insert(0, {"id": "store-report", "kind": "report", "title": "Store report (latest day)", "t": time.time(),
                   "url": "/api/report/store.pdf", "has_file": True, "detail": "Made fresh each time you open it"})
    return out


def remove(fid: str) -> Dict[str, Any]:
    with _LOCK:
        rows = _rows()
        gone = [r for r in rows if r["id"] == fid]
        if not gone:
            raise KeyError(fid)
        (_dir() / f"{fid}.{gone[0].get('ext', 'pdf')}").unlink(missing_ok=True)
        _log().write_text("".join(json.dumps(r) + "\n" for r in rows if r["id"] != fid))
    return {"ok": True}
