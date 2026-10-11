"""A very small PDF writer: Helvetica text, filled rectangles and lines on Letter pages.

Enough for the store report and for saving an answer as a PDF, with nothing to install. Text is
WinAnsi (Latin-1): characters outside it are replaced, so a report never fails over a symbol.
"""

from __future__ import annotations

import zlib
from typing import List, Optional, Tuple

W, H = 612, 792            # US Letter, in points
MARGIN = 48
# Helvetica's advance widths for ASCII 32..126, per 1000 units of font size
_WIDTHS = [278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556, 556,
           556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556, 1015, 667, 667, 722, 722, 667, 611, 778, 722, 278,
           500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469,
           556, 333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500,
           278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584]
_SWAP = {"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "…": "...",
         "•": "-", "−": "-", " ": " ", "✓": "ok", "✗": "x", "×": "x", "→": "->"}


def latin(text: str) -> str:
    text = "".join(_SWAP.get(c, c) for c in str(text))
    return text.encode("latin-1", "replace").decode("latin-1")


def width(text: str, size: float, bold: bool = False) -> float:
    total = sum(_WIDTHS[ord(c) - 32] if 32 <= ord(c) <= 126 else 556 for c in latin(text))
    return total * size / 1000 * (1.06 if bold else 1.0)


def wrap(text: str, size: float, room: float, bold: bool = False) -> List[str]:
    out = []
    for para in str(text).split("\n"):
        line = ""
        for word in para.split(" "):
            trial = (line + " " + word) if line else word
            if width(trial, size, bold) <= room or not line:
                line = trial
            else:
                out.append(line)
                line = word
        out.append(line)
    return out


def _esc(text: str) -> str:
    return latin(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _rgb(color: str) -> Tuple[float, float, float]:
    c = color.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


class Doc:
    def __init__(self, title: str = ""):
        self.title = title
        self.pages: List[List[str]] = []
        self.page()

    def page(self) -> None:
        self.pages.append([])
        self.y = H - MARGIN

    @property
    def ops(self) -> List[str]:
        return self.pages[-1]

    def need(self, height: float) -> None:
        """Start a new page if there is not this much room left."""
        if self.y - height < MARGIN:
            self.page()

    def text(self, x: float, y: float, s: str, size: float = 10, bold: bool = False, color: str = "#1d2329",
             align: str = "left") -> None:
        if align == "right":
            x -= width(s, size, bold)
        elif align == "center":
            x -= width(s, size, bold) / 2
        r, g, b = _rgb(color)
        self.ops.append(f"BT /{'F2' if bold else 'F1'} {size:g} Tf {r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} Td "
                        f"({_esc(s)}) Tj ET")

    def rect(self, x: float, y: float, w: float, h: float, color: str) -> None:
        r, g, b = _rgb(color)
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} {w:.2f} {h:.2f} re f")

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = "#c3c2b7", w: float = 0.6) -> None:
        r, g, b = _rgb(color)
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} RG {w:g} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def para(self, s: str, size: float = 10, bold: bool = False, color: str = "#1d2329", indent: float = 0,
             gap: float = 4, x: Optional[float] = None) -> None:
        x = MARGIN + indent if x is None else x
        lead = size * 1.35
        for ln in wrap(s, size, W - MARGIN - x, bold):
            self.need(lead)
            self.y -= lead
            self.text(x, self.y + size * 0.3, ln, size, bold, color)
        self.y -= gap

    def bytes(self) -> bytes:
        objs: List[bytes] = []

        def add(body: bytes) -> int:
            objs.append(body)
            return len(objs)

        f1 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        f2 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        pages_id = len(objs) + 1 + 2 * len(self.pages)
        kids = []
        for n, ops in enumerate(self.pages, 1):
            if self.title:
                ops = ops + [f"BT /F1 7.5 Tf 0.54 0.53 0.51 rg {MARGIN} 26 Td ({_esc(self.title)}) Tj ET",
                             f"BT /F1 7.5 Tf 0.54 0.53 0.51 rg {W - MARGIN - 40} 26 Td (Page {n} of {len(self.pages)}) Tj ET"]
            stream = zlib.compress("\n".join(ops).encode("latin-1"))
            content = add(b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(stream) + stream + b"\nendstream")
            kids.append(add(f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {W} {H}] /Contents {content} 0 R "
                            f"/Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R >> >> >>".encode()))
        add(f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode())
        info = add(f"<< /Title ({_esc(self.title)}) /Producer (Deep Network Designer) >>".encode())
        catalog = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, body in enumerate(objs, 1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
        out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
        out += f"trailer\n<< /Size {len(objs) + 1} /Root {catalog} 0 R /Info {info} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        return bytes(out)


def markdown(doc: Doc, text: str, size: float = 10) -> None:
    """Answers are Markdown: headings, bullets and tables come out as such; the rest as paragraphs."""
    lines = str(text or "").replace("\r", "").split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.strip().startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            rows = [ln]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            table(doc, [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows], size=size - 1)
            continue
        plain = ln.replace("**", "").replace("`", "")
        if not plain.strip():
            doc.y -= size * 0.4
        elif plain.lstrip().startswith("#"):
            doc.y -= 4
            doc.para(plain.lstrip("# ").strip(), size + 1.5, True, gap=2)
        elif plain.lstrip()[:2] in ("- ", "* ", "• ") or (plain.lstrip()[:1].isdigit() and plain.lstrip()[1:3] in (". ", ") ")):
            body = plain.lstrip()[2:] if not plain.lstrip()[:1].isdigit() else plain.lstrip()
            doc.need(size * 1.35)
            doc.text(MARGIN + 6, doc.y - size * 1.35 + size * 0.3, "-" if not plain.lstrip()[:1].isdigit() else "", size)
            doc.para(body, size, indent=16, gap=1)
        else:
            doc.para(plain.strip(), size, gap=3)
        i += 1


def table(doc: Doc, rows: List[List[str]], size: float = 9, widths: Optional[List[float]] = None,
          numeric: Optional[List[bool]] = None) -> None:
    if not rows:
        return
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    room = W - 2 * MARGIN
    if widths is None:
        want = [max(width(r[c], size, r is rows[0]) for r in rows) + 12 for c in range(n)]
        scale = min(1.0, room / sum(want))
        widths = [w * scale for w in want]
    if numeric is None:
        numeric = [all(_isnum(r[c]) for r in rows[1:] if r[c]) and any(r[c] for r in rows[1:]) for c in range(n)]
    lead = size * 1.6
    for k, r in enumerate(rows):
        doc.need(lead)
        doc.y -= lead
        x = MARGIN
        for c in range(n):
            cell = r[c]
            while cell and width(cell, size, k == 0) > widths[c] - 6:
                cell = cell[:-2] + "…" if len(cell) > 2 else ""
            color = "#52514e" if k == 0 else "#1d2329"
            if numeric[c]:
                doc.text(x + widths[c] - 4, doc.y + size * 0.45, cell, size, k == 0, color, "right")
            else:
                doc.text(x + 2, doc.y + size * 0.45, cell, size, k == 0, color)
            x += widths[c]
        doc.line(MARGIN, doc.y, MARGIN + sum(widths), doc.y, "#e1e0d9" if k else "#c3c2b7", 0.5)
    doc.y -= 6


def _isnum(s: str) -> bool:
    t = s.replace(",", "").replace("$", "").replace("%", "").replace("+", "").strip()
    try:
        float(t.split()[0] if t else "x")
        return True
    except ValueError:
        return False
