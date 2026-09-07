"""
dnn_bench/ingest.py — paper in, candidate structures out.

This does the mechanical half of the extraction: pull text out of a PDF, split
it into sections, and surface the passages that plausibly contain a structural
result — theorem statements, the conditions attached to them, and the displayed
dynamics.

It deliberately does NOT decide which of them matters. That judgment is the
whole difficulty, and a keyword search is not going to make it. What this buys
is that you (or the proposer in propose.py) look at fifteen ranked passages
instead of forty pages.

The ranking is a heuristic, not a verdict. A structural result that never uses
the word "unique" will rank low and still be the right one.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# words that mark a well-posedness claim worth turning into a layer
CLAIM_WORDS = [
    'unique', 'uniqueness', 'exactly one', 'exists a unique', 'well-posed',
    'asymptotically stable', 'globally stable', 'locally stable', 'stability',
    'converges', 'convergence', 'equilibrium', 'equilibria', 'fixed point',
    'steady state', 'attractor', 'invariant', 'conserved', 'conservation',
    'lyapunov', 'contraction', 'monotone', 'convex', 'coercive',
]
# words that mark the CONDITION the claim depends on -- the thing an ablation
# has to be able to break
CONDITION_WORDS = [
    'if and only if', 'suppose that', 'assume that', 'provided that',
    'whenever', 'satisfies', 'is said to be', 'we assume', 'under the '
    'assumption', 'necessary and sufficient', 'holds for all', 'for every',
]
HEADING = re.compile(
    r'^\s*(?:(?:\d+(?:\.\d+)*)\s+)?'
    r'(abstract|introduction|preliminaries|background|notation|'
    r'main results?|results?|theory|methods?|model|setup|definitions?|'
    r'theorem|lemma|proposition|corollary|proof|discussion|conclusions?|'
    r'references|appendix)\b[:.]?\s*$', re.I)
STATEMENT = re.compile(
    r'\b(theorem|lemma|proposition|corollary|definition|assumption|condition)'
    r'\s*([0-9]+(?:\.[0-9]+)*)?\b', re.I)


@dataclass
class Passage:
    kind: str                 #: theorem | definition | dynamics | prose
    label: str                #: e.g. "Theorem 2.8"
    text: str
    section: str = ''
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class Paper:
    title: str
    text: str
    source: str
    sections: Dict[str, str] = field(default_factory=dict)
    passages: List[Passage] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)

    def brief(self, n_passages: int = 12, chars: int = 14000) -> str:
        """A compact view for a human or a proposer: the abstract, the
        top-ranked passages, and the displayed equations."""
        parts = []
        abstract = self.sections.get('abstract', '')[:1800]
        if abstract:
            parts.append("ABSTRACT\n" + abstract)
        parts.append("\nCANDIDATE STRUCTURES (ranked, heuristic)\n")
        for p in self.passages[:n_passages]:
            parts.append(f"[{p.kind}] {p.label or '—'} "
                         f"(score {p.score:.1f}; {', '.join(p.reasons) or 'n/a'})\n"
                         f"{p.text[:900]}\n")
        if self.equations:
            parts.append("\nDISPLAYED EXPRESSIONS\n"
                         + "\n".join(self.equations[:25]))
        return "\n".join(parts)[:chars]


# ----------------------------------------------------------------------------

def read_text(path_or_text: str) -> tuple:
    """Accepts a .pdf path, a .txt/.md path, or raw text. Returns (text, source)."""
    p = path_or_text
    if os.path.isfile(p):
        if p.lower().endswith('.pdf'):
            try:
                from pypdf import PdfReader
            except ImportError:
                raise RuntimeError("reading PDFs needs pypdf: pip install pypdf")
            reader = PdfReader(p)
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or '')
                except Exception:
                    pages.append('')
            return "\n".join(pages), os.path.basename(p)
        with open(p, errors='ignore') as fh:
            return fh.read(), os.path.basename(p)
    return p, 'pasted text'


def _clean(text: str) -> str:
    text = text.replace('\r', '\n')
    text = re.sub(r'-\n(?=[a-z])', '', text)          # de-hyphenate line breaks
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def split_sections(text: str) -> Dict[str, str]:
    lines, sections, cur, buf = text.split('\n'), {}, 'preamble', []
    for line in lines:
        m = HEADING.match(line.strip())
        if m and len(line.strip()) < 60:
            sections[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1).lower(), []
        else:
            buf.append(line)
    sections[cur] = "\n".join(buf).strip()
    return {k: v for k, v in sections.items() if v}


def _score(text: str) -> tuple:
    low, score, why = text.lower(), 0.0, []
    hits = [w for w in CLAIM_WORDS if w in low]
    if hits:
        score += 1.6 * min(len(hits), 5)
        why.append('claims: ' + ', '.join(hits[:4]))
    conds = [w for w in CONDITION_WORDS if w in low]
    if conds:
        score += 1.2 * min(len(conds), 3)
        why.append('conditional')
    if re.search(r'\b(d\s*[a-z]\s*/\s*d\s*t|\\dot|=\s*0\s*$)', text):
        score += 2.0
        why.append('dynamics or an equilibrium condition')
    if re.search(r'[=<>≤≥∈∑∏∇∂]', text):
        score += 1.0
        why.append('formal content')
    if re.search(r'\bfor (all|every|any)\b.*\b(parameter|constant|rate|weight)', low):
        score += 2.5
        why.append('claim quantified over ALL parameters — the strong kind')
    if len(text) < 120:
        score -= 1.5
    return score, why


def find_passages(text: str, sections: Dict[str, str]) -> List[Passage]:
    out, sec_of = [], {}
    for name, body in sections.items():
        for chunk in body.split('\n\n'):
            sec_of[chunk[:60]] = name

    blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
    for i, b in enumerate(blocks):
        m = STATEMENT.search(b[:120])
        kind = (m.group(1).lower() if m else
                ('dynamics' if re.search(r'(d\s*[a-z]\s*/\s*d\s*t|\\dot)', b)
                 else 'prose'))
        label = (f"{m.group(1).title()} {m.group(2) or ''}".strip() if m else '')
        # a theorem's meaning usually needs the sentence after it
        body = b if len(b) > 400 or i + 1 >= len(blocks) else b + "\n" + blocks[i + 1]
        s, why = _score(body)
        if kind in ('theorem', 'proposition', 'lemma', 'corollary'):
            s += 3.0
            why.append('formal statement')
        elif kind in ('definition', 'assumption', 'condition'):
            s += 1.5
            why.append('names a condition')
        if s <= 1.5:
            continue
        out.append(Passage(kind=kind, label=label, text=body.strip(),
                           section=sec_of.get(b[:60], ''), score=s, reasons=why))
    out.sort(key=lambda p: -p.score)
    return out[:40]


def find_equations(text: str) -> List[str]:
    eqs = []
    for line in text.split('\n'):
        t = line.strip()
        if len(t) < 4 or len(t) > 220:
            continue
        symbols = sum(c in '=+-*/^∑∏∇∂≤≥<>' for c in t)
        letters = sum(c.isalpha() for c in t)
        if symbols >= 2 and letters and symbols / max(letters, 1) > 0.12:
            eqs.append(t)
    seen, out = set(), []
    for e in eqs:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out[:60]


def ingest(path_or_text: str, title: str = '') -> Paper:
    raw, source = read_text(path_or_text)
    text = _clean(raw)
    sections = split_sections(text)
    if not title:
        head = [l.strip() for l in text.split('\n')[:12] if len(l.strip()) > 12]
        title = head[0][:140] if head else source
    return Paper(title=title, text=text, source=source, sections=sections,
                 passages=find_passages(text, sections),
                 equations=find_equations(text))
