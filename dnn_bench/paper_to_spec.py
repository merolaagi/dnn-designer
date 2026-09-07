"""
paper_to_spec.py — research question -> starting paper -> model spec skeleton.

SELF-CONTAINED ON PURPOSE. Standard library only, no imports from any package,
no torch. Copy this one file into another project and it works:

    from paper_to_spec import discover, ingest, blank_spec
    res = discover("why are some people insulin resistant")
    paper = ingest(res['foundational'][0].source_url)
    spec = blank_spec(paper)

Or from the shell:

    python paper_to_spec.py "why are some people insulin resistant"
    python paper_to_spec.py --read some.pdf

Optional: pypdf, only for reading PDFs. Everything else is stdlib.

WHAT IT DOES
  1. Turns a question into field-scoped literature queries.
  2. Finds hub reviews, then follows their reference lists to what they all
     point back at -- usually a modelling paper decades older than anything a
     keyword search returns.
  3. Ranks by fit to a fixed-point/dynamical-systems harness, not by citations.
  4. Extracts text from the chosen paper and ranks the passages that plausibly
     carry a structural result.
  5. Emits a spec skeleton to fill in.

WHAT IT DOES NOT DO
  Answer the question. It finds the paper worth starting from. Deciding whether
  that paper offers a CONSTRAINT or merely a STORY is the step no software does
  for you, and it is the step that decides whether any of this was worth it.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

__all__ = ['Hit', 'Paper', 'Passage', 'discover', 'ingest', 'blank_spec',
           'score_fit', 'build_queries', 'search_epmc', 'search_arxiv',
           'references_of', 'enrich', 'full_text', 'SPEC_TEMPLATE', 'CAVEAT']

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
ARXIV = "http://export.arxiv.org/api/query"
UA = {'User-Agent': 'paper-to-spec/1.0 (research tool)'}


# ============================================================================
# scoring
# ============================================================================
#
# The first version of this scored words like "homeostasis", "flux", "stable"
# and "kinetics" as evidence of mathematical structure. They are not -- they are
# ordinary molecular-biology vocabulary, and a review with no equations in it
# scored 7 out of them. So:
#
#   HARD markers are the only primary evidence. No hard marker, score zero,
#   regardless of how much soft vocabulary a paper contains.
#   SOFT markers only modulate a score that a hard marker has already earned.

HARD = {
    'differential equation': 5.0, 'ordinary differential': 5.0,
    'system of equations': 4.5, 'dynamical system': 5.0,
    'compartment model': 4.5, 'compartmental model': 4.5,
    'mathematical model': 4.0, 'mathematical modelling': 4.0,
    'mathematical modeling': 4.0, 'minimal model': 4.5,
    'kinetic model': 4.0, 'stability analysis': 4.5, 'lyapunov': 5.0,
    'bifurcation': 4.5, 'phase plane': 4.5, 'state space': 3.5,
    'parameter estimation': 4.0, 'parameter identification': 4.0,
    'identifiability': 4.5, 'in silico': 3.5, 'numerical simulation': 3.5,
    'computational model': 3.5, 'mass action': 4.5, 'stoichiometr': 4.5,
    'control theor': 4.0, 'steady-state analysis': 4.0,
    'model predicts': 2.5, 'simulation model': 3.5,
    'ode model': 5.0, 'pde model': 5.0, 'fokker': 5.0,
    'markov model': 3.5, 'stochastic differential': 5.0,
    'flux balance': 4.5, 'metabolic control analysis': 5.0,
}
SOFT = {
    'steady state': 1.5, 'equilibri': 1.5, 'feedback loop': 1.5,
    'nonlinear': 1.0, 'oscillat': 1.0, 'conservation': 1.0,
    'homeostasis': 0.5, 'flux': 0.5, 'kinetics': 0.5, 'stable': 0.5,
    'feedback': 0.5, 'network topology': 1.0, 'invariant': 1.5,
    'rate constant': 1.5, 'convex': 1.0,
}
# Real science, wrong shape. An association is not a system.
OBSERVATIONAL = {
    'cohort': -4.0, 'prevalence': -4.0, 'cross-sectional': -4.0,
    'questionnaire': -4.0, 'odds ratio': -4.0, 'epidemiolog': -3.5,
    'genome-wide': -4.0, 'association study': -4.0, 'case-control': -4.0,
    'meta-analysis': -3.0, 'randomized controlled': -3.0,
    'randomised controlled': -3.0, 'retrospective': -2.5,
    'risk factor': -2.0, 'hazard ratio': -3.5, 'survey of': -2.5,
}
REVIEW_TYPES = {'review', 'systematic review', 'narrative review'}

STOPWORDS = {
    'a', 'about', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'been', 'but',
    'by', 'can', 'cause', 'causes', 'do', 'does', 'for', 'from', 'get', 'have',
    'how', 'i', 'in', 'is', 'it', 'its', 'me', 'my', 'not', 'of', 'on', 'or',
    'people', 'person', 'reason', 'reasons', 'some', 'someone', 'that', 'the',
    'their', 'them', 'there', 'these', 'they', 'this', 'to', 'us', 'was',
    'we', 'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with',
    'would', 'you', 'your',
}
# question words -> the noun phrase people actually mean
PHRASE_HINTS = {
    'insulin resistant': 'insulin resistance',
    'insulin resistance': 'insulin resistance',
    'blood sugar': 'glucose',
    'sugar': 'glucose',
}


def key_terms(question: str) -> List[str]:
    """Content words from a natural-language question.

    "why are some people insulin resistant" -> ['insulin', 'resistant'].
    Without this the stopwords dominate the query and Europe PMC's relevance
    ranking drifts to whatever recent papers share the leftover vocabulary,
    which is how an aging review ends up answering a question about insulin.
    """
    q = question.lower()
    words = [w for w in re.findall(r"[a-z][a-z0-9\-]+", q)
             if w not in STOPWORDS and len(w) > 2]
    return words


def key_phrases(question: str) -> List[str]:
    """Multi-word phrases worth quoting in the query."""
    q = question.lower()
    out = []
    for k, v in PHRASE_HINTS.items():
        if k in q:
            out.append(v)
    words = key_terms(question)
    for i in range(len(words) - 1):
        p = f"{words[i]} {words[i + 1]}"
        if p not in out:
            out.append(p)
    return out or words[:2]


def topical(text: str, terms: List[str]) -> int:
    """How many of the question's content words appear. Crude stemming: a
    trailing 'e'/'s'/'ance'/'ant' is stripped so resistant ~ resistance."""
    t = text.lower()
    n = 0
    for w in terms:
        stem = re.sub(r'(ance|ence|ant|ent|ing|ion|s|e)$', '', w)
        if len(stem) >= 4 and stem in t:
            n += 1
        elif w in t:
            n += 1
    return n


def score_fit(title: str, abstract: str, keywords: str = '',
              is_review: bool = False) -> Tuple[float, List[str]]:
    """Fit to a fixed-point/dynamical-systems harness. Not quality, not impact.

    Returns 0 unless the paper names an actual mathematical object. Soft
    biological vocabulary alone earns nothing.
    """
    text = f"{title} {abstract} {keywords}".lower()
    hard = [(t, w) for t, w in HARD.items() if t in text]
    if not hard:
        return 0.0, ['no mathematical object named — soft vocabulary alone '
                     'does not count']
    score = sum(w for _, w in hard)
    soft = [(t, w) for t, w in SOFT.items() if t in text]
    score += sum(w for _, w in soft)
    obs = [(t, w) for t, w in OBSERVATIONAL.items() if t in text]
    score += sum(w for _, w in obs)
    reasons = ['maths: ' + ', '.join(t for t, _ in
                                     sorted(hard, key=lambda x: -x[1])[:3])]
    if soft:
        reasons.append('supporting: ' + ', '.join(t for t, _ in soft[:3]))
    if obs:
        reasons.append('observational: ' + ', '.join(t for t, _ in obs[:3]))
    if is_review:
        reasons.append('review — a signpost, not a destination')
    return score, reasons


# ============================================================================
# search
# ============================================================================

@dataclass
class Hit:
    source: str = 'MED'
    id: str = ''
    title: str = ''
    authors: str = ''
    year: str = ''
    journal: str = ''
    abstract: str = ''
    doi: str = ''
    pmcid: str = ''
    cited_by: int = 0
    open_access: bool = False
    in_epmc: bool = False          #: full text actually deposited in Europe PMC
    pdf_url: str = ''
    is_review: bool = False
    fit: float = 0.0
    topic: int = 0
    reasons: List[str] = field(default_factory=list)
    recurrence: int = 0

    @property
    def ref(self) -> str:
        return f"{self.source}:{self.id}"

    @property
    def source_url(self) -> Optional[str]:
        return full_text(self)

    def line(self) -> str:
        bits = [f"fit {self.fit:.1f}"]
        if self.recurrence:
            bits.append(f"cited by {self.recurrence} hubs")
        if self.cited_by:
            bits.append(f"{self.cited_by} citations")
        if self.open_access:
            bits.append("open access")
        return (f"{' · '.join(bits)}\n      {self.title[:110]}\n"
                f"      {self.authors[:64]}  {self.journal[:38]} {self.year}")


def build_queries(question: str) -> Dict[str, str]:
    """Field-scoped Europe PMC queries.

    Sending the raw question as free text is what produced myokine and aging
    reviews for a question about insulin: the stopwords carry no weight and
    relevance drifts. These pin the topic to title/abstract and require a
    methodological term for the modelling query.
    """
    phrases = key_phrases(question)[:3]
    terms = key_terms(question)[:4]
    topic = " OR ".join(
        [f'TITLE_ABS:"{p}"' for p in phrases]
        + [f'TITLE_ABS:"{t}"' for t in terms])
    method = " OR ".join(f'TITLE_ABS:"{m}"' for m in [
        'mathematical model', 'dynamical system', 'differential equation',
        'compartment model', 'kinetic model', 'minimal model',
        'computational model', 'in silico', 'parameter estimation',
        'stability analysis'])
    return {
        'modelling': f"({topic}) AND ({method})",
        'hubs': f"({topic}) AND (PUB_TYPE:\"Review\")",
        'topic_only': f"({topic})",
    }


def _get(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _parse_epmc(payload: dict) -> List[Hit]:
    out = []
    for r in payload.get('resultList', {}).get('result', []):
        types = [t.lower() for t in
                 (r.get('pubTypeList') or {}).get('pubType', [])]
        kws = ' '.join((r.get('keywordList') or {}).get('keyword', []))
        pdf = ''
        for u in (r.get('fullTextUrlList') or {}).get('fullTextUrl', []):
            if (u.get('documentStyle') == 'pdf'
                    and 'Open' in (u.get('availability') or '')):
                pdf = u.get('url', '')
        is_review = any(t in REVIEW_TYPES for t in types)
        h = Hit(source=r.get('source', 'MED'), id=str(r.get('id', '')),
                title=(r.get('title') or '').strip(),
                authors=r.get('authorString', ''),
                year=str(r.get('pubYear', '')),
                journal=((r.get('journalInfo') or {}).get('journal') or {})
                .get('title', ''),
                abstract=re.sub(r'<[^>]+>', ' ', r.get('abstractText') or ''),
                doi=r.get('doi', ''), pmcid=r.get('pmcid', ''),
                cited_by=int(r.get('citedByCount') or 0),
                open_access=(r.get('isOpenAccess') == 'Y'),
                in_epmc=(r.get('inEPMC') == 'Y'),
                pdf_url=pdf, is_review=is_review)
        h.fit, h.reasons = score_fit(h.title, h.abstract, kws, is_review)
        out.append(h)
    return out


def _parse_refs(payload: dict) -> List[Hit]:
    out = []
    for r in (payload.get('referenceList') or {}).get('reference', []):
        if not r.get('id'):
            continue
        h = Hit(source=r.get('source', 'MED'), id=str(r['id']),
                title=(r.get('title') or '').strip(),
                authors=r.get('authorString', ''),
                year=str(r.get('pubYear', '')),
                journal=r.get('journalAbbreviation', ''))
        out.append(h)
    return out


def _parse_arxiv(xml_text: str) -> List[Hit]:
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    out = []
    for e in ET.fromstring(xml_text).findall('a:entry', ns):
        t = (e.findtext('a:title', '', ns) or '').strip().replace('\n', ' ')
        s = (e.findtext('a:summary', '', ns) or '').strip().replace('\n', ' ')
        aid = (e.findtext('a:id', '', ns) or '').rsplit('/', 1)[-1]
        auth = ', '.join(x.findtext('a:name', '', ns)
                         for x in e.findall('a:author', ns))[:120]
        h = Hit(source='arxiv', id=aid, title=t, authors=auth,
                year=(e.findtext('a:published', '', ns) or '')[:4],
                journal='arXiv', abstract=s, open_access=True,
                pdf_url=f"https://arxiv.org/pdf/{aid}")
        h.fit, h.reasons = score_fit(t, s)
        out.append(h)
    return out


def search_epmc(query: str, n: int = 25, sort: str = '') -> List[Hit]:
    url = (f"{EPMC}/search?query={urllib.parse.quote(query)}"
           f"&resultType=core&pageSize={n}&format=json")
    if sort:
        url += f"&sort={urllib.parse.quote(sort)}"
    return _parse_epmc(_get(url))


def search_arxiv(query: str, n: int = 15) -> List[Hit]:
    url = (f"{ARXIV}?search_query=all:{urllib.parse.quote(query)}"
           f"&start=0&max_results={n}&sortBy=relevance")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return _parse_arxiv(r.read().decode('utf-8', 'replace'))


def references_of(hit: Hit, n: int = 100, log=lambda *a: None) -> List[Hit]:
    """Reference list of one paper. Failures are REPORTED, not swallowed --
    a silent empty list here looks identical to a paper with no references,
    and that ambiguity hid a broken stage-2 for a whole release."""
    url = f"{EPMC}/{hit.source}/{hit.id}/references?format=json&pageSize={n}"
    try:
        refs = _parse_refs(_get(url))
        log(f"    {hit.ref}: {len(refs)} references")
        return refs
    except Exception as exc:
        log(f"    {hit.ref}: reference fetch FAILED — {exc}")
        return []


def enrich(hits: List[Hit], log=lambda *a: None) -> List[Hit]:
    """Fetch abstracts for reference-list entries before judging them.

    A reference list gives a title and nothing else, and a title alone is close
    to useless: "Quantitative estimation of insulin sensitivity" names no
    mathematical object and is the Bergman minimal model.
    """
    out = []
    for h in hits:
        if h.abstract:
            out.append(h)
            continue
        try:
            q = f'EXT_ID:{h.id} AND SRC:{h.source}'
            full = _parse_epmc(_get(
                f"{EPMC}/search?query={urllib.parse.quote(q)}"
                f"&resultType=core&pageSize=1&format=json"))
            if full:
                full[0].recurrence = h.recurrence
                out.append(full[0])
                continue
        except Exception as exc:
            log(f"    could not enrich {h.ref}: {exc}")
        h.fit, h.reasons = score_fit(h.title, '')
        out.append(h)
    return out


def full_text(hit: Hit) -> Optional[str]:
    """Where the full text can actually be fetched, or None.

    A PMCID is NOT enough: only the Open Access subset has full text XML, so a
    paywalled article with a PMCID returns 404. Offering a button that can only
    fail is worse than offering none.
    """
    if hit.source == 'arxiv':
        # Source, not PDF: the equations are exact in the author's .tex and
        # shredded in the PDF.
        return f"https://arxiv.org/e-print/{hit.id}"
    if hit.pmcid and (hit.in_epmc or hit.open_access):
        return f"{EPMC}/{hit.pmcid}/fullTextXML"
    return hit.pdf_url or None


def full_text_alternates(hit: Hit) -> List[str]:
    """Every URL worth trying, best first. Availability flags are metadata and
    metadata is sometimes wrong, so the caller tries in order rather than
    trusting the first."""
    urls = []
    if hit.source == 'arxiv':
        return [u for u in (f"https://arxiv.org/e-print/{hit.id}",
                            hit.pdf_url) if u]
    if hit.pmcid:
        urls.append(f"{EPMC}/{hit.pmcid}/fullTextXML")
        urls.append(f"https://europepmc.org/articles/{hit.pmcid}?pdf=render")
    if hit.pdf_url and hit.pdf_url not in urls:
        urls.append(hit.pdf_url)
    return urls


def discover(question: str, n_hubs: int = 6, use_arxiv: bool = True,
             min_topic: int = 1, log=lambda *a: None) -> dict:
    """Question -> hubs -> what the hubs point back at."""
    terms = key_terms(question)
    q = build_queries(question)
    log(f"  terms: {', '.join(terms)}")
    log(f"  query: {q['modelling'][:120]}…")

    pool: Dict[str, Hit] = {}

    def add(hits):
        for h in hits:
            if h.ref not in pool:
                h.topic = topical(f"{h.title} {h.abstract}", terms)
                pool[h.ref] = h

    for name, sort in (('modelling', ''), ('hubs', 'CITED desc'),
                       ('modelling', 'CITED desc')):
        try:
            add(search_epmc(q[name], n=25, sort=sort))
        except Exception as exc:
            log(f"  Europe PMC ({name}) unavailable: {exc}")

    # Topical gate: however well a paper scores, if it is not about the
    # question it is not an answer to it.
    on_topic = [h for h in pool.values() if h.topic >= min_topic]
    log(f"  {len(pool)} results, {len(on_topic)} on topic")

    hubs = sorted([h for h in on_topic if h.is_review or h.cited_by > 20],
                  key=lambda h: -(h.topic * 3 + min(h.cited_by, 400) / 100
                                  + (2 if h.is_review else 0)))[:n_hubs]
    log(f"  stage 1 — {len(hubs)} hubs")

    log("  stage 2 — following their reference lists")
    refs: Dict[str, Hit] = {}
    for h in hubs:
        for r in references_of(h, log=log):
            if r.ref in refs:
                refs[r.ref].recurrence += 1
            else:
                r.recurrence = 1
                refs[r.ref] = r
    log(f"  {len(refs)} distinct references")

    shortlist = sorted(refs.values(), key=lambda h: -h.recurrence)[:24]
    if shortlist:
        log(f"  enriching {len(shortlist)} recurring references")
        shortlist = enrich(shortlist, log)
    for h in shortlist:
        h.topic = topical(f"{h.title} {h.abstract}", terms)
    foundational = sorted(
        [h for h in shortlist if h.fit > 0 and h.topic >= min_topic],
        key=lambda h: -(h.fit + h.recurrence * 1.5))[:12]

    direct = sorted([h for h in on_topic if not h.is_review and h.fit > 0],
                    key=lambda h: -h.fit)[:12]

    preprints = []
    if use_arxiv:
        try:
            aq = ' '.join(key_phrases(question)[:2] + ['model'])
            preprints = sorted([h for h in search_arxiv(aq)
                                if h.fit > 0 and topical(h.title + h.abstract,
                                                         terms) >= min_topic],
                               key=lambda h: -h.fit)[:8]
        except Exception as exc:
            log(f"  arXiv unavailable: {exc}")

    return {'question': question, 'terms': terms, 'queries': q,
            'hubs': hubs, 'foundational': foundational,
            'direct': direct, 'preprints': preprints}


# ============================================================================
# ingestion
# ============================================================================

CLAIM_WORDS = ['unique', 'exactly one', 'well-posed', 'asymptotically stable',
               'globally stable', 'stability', 'converges', 'equilibri',
               'fixed point', 'steady state', 'attractor', 'invariant',
               'conserved', 'lyapunov', 'contraction', 'monotone', 'convex']
CONDITION_WORDS = ['if and only if', 'suppose that', 'assume that',
                   'provided that', 'whenever', 'satisfies', 'we assume',
                   'necessary and sufficient', 'holds for all', 'for every']
HEADING = re.compile(
    r'^\s*(?:\u00a7\s+(?P<declared>.{1,70})$)|'
    r'^\s*(?:(?:\d+(?:\.\d+)*)\s+)?(abstract|introduction|preliminaries|'
    r'background|notation|main results?|results?|theory|methods?|model|setup|'
    r'definitions?|theorem|lemma|proposition|corollary|proof|discussion|'
    r'conclusions?|references|appendix)\b[:.]?\s*$', re.I)
STATEMENT = re.compile(
    r'\b(theorem|lemma|proposition|corollary|definition|assumption|condition)'
    r'\s*([0-9]+(?:\.[0-9]+)*)?\b', re.I)


@dataclass
class Passage:
    kind: str
    label: str
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

    def brief(self, n_passages: int = 12, chars: int = 14000,
              only: Optional[List[int]] = None) -> str:
        """`only` restricts the brief to chosen passage indices. Worth using:
        the top-ranked passage is not always the load-bearing one, and a
        proposer fed the wrong theorem will confidently model the wrong thing."""
        parts = []
        if self.sections.get('abstract'):
            parts.append("ABSTRACT\n" + self.sections['abstract'][:1800])
        chosen = ([self.passages[i] for i in only if 0 <= i < len(self.passages)]
                  if only else self.passages[:n_passages])
        parts.append("\nCANDIDATE STRUCTURES"
                     + (" (chosen by the operator)" if only
                        else " (ranked, heuristic)") + "\n")
        for p in chosen:
            parts.append(f"[{p.kind}] {p.label or '—'} (score {p.score:.1f}; "
                         f"{', '.join(p.reasons) or 'n/a'})\n{p.text[:900]}\n")
        if self.equations:
            parts.append("\nDISPLAYED EXPRESSIONS\n"
                         + "\n".join(self.equations[:25]))
        return "\n".join(parts)[:chars]


# ---- JATS / MathML --------------------------------------------------------
#
# PMC full text is structured XML. Stripping it with a regex throws away exactly
# the part that matters: dG/dt = -p1*G - X*G becomes "d G d t = - p 1 G - X G",
# from which the fraction cannot be recovered, and with no paragraph breaks left
# the whole article collapses into a single passage. Parse it properly instead.

_MO_SPACED = set('=+-<>≤≥≠±×÷∈')


def _mathml(el) -> str:
    """MathML -> readable infix. Not a full translator; enough that a reader
    (human or model) can see the equation the paper actually wrote."""
    tag = el.tag.split('}')[-1]
    kids = list(el)
    txt = (el.text or '').strip()

    def k(i):
        return _mathml(kids[i]) if i < len(kids) else ''

    if tag in ('mi', 'mn', 'mtext'):
        return txt
    if tag == 'mo':
        return f" {txt} " if txt in _MO_SPACED else txt
    if tag == 'mspace':
        return ' '
    if tag == 'mfrac':
        return f"({k(0)})/({k(1)})"
    if tag == 'msup':
        return f"{k(0)}^{k(1)}"
    if tag == 'msub':
        return f"{k(0)}_{k(1)}"
    if tag == 'msubsup':
        return f"{k(0)}_{k(1)}^{k(2)}"
    if tag == 'msqrt':
        return "sqrt(" + ''.join(_mathml(c) for c in kids) + ")"
    if tag == 'mroot':
        return f"({k(0)})^(1/{k(1)})"
    if tag == 'mfenced':
        return "(" + ", ".join(_mathml(c) for c in kids) + ")"
    if tag in ('munder', 'mover', 'munderover'):
        out = k(0)
        if len(kids) > 1:
            out += f"_{{{k(1)}}}"
        if len(kids) > 2:
            out += f"^{{{k(2)}}}"
        return out
    if tag in ('mtable', 'mtr'):
        return ' ; '.join(_mathml(c) for c in kids)
    return ''.join(_mathml(c) for c in kids) or txt


def _formula(el) -> str:
    """A JATS formula element. LaTeX is preferred when the publisher supplied
    it, because it is what the author wrote."""
    for tex in el.iter():
        if tex.tag.split('}')[-1] == 'tex-math' and (tex.text or '').strip():
            return re.sub(r'\\s+', ' ',
                          re.sub(r'\\(begin|end)\{[^}]*\}', '',
                                 tex.text).strip())
    for m in el.iter():
        if m.tag.split('}')[-1] == 'math':
            out = re.sub(r'\s{2,}', ' ', _mathml(m)).strip()
            if out:
                return out
    return ' '.join((el.itertext())).strip()


#: Prefix on a heading the source document declared, rather than one guessed
#: from a regex. Stripped before the text is shown anywhere.
SECTION_MARK = '\u00a7 '

_SKIP_TAGS = {'xref', 'table-wrap', 'fig', 'graphic', 'media', 'label',
              'ref-list', 'back', 'funding-group', 'contrib-group', 'aff'}


def _flatten(el, eqs: List[str]) -> str:
    tag = el.tag.split('}')[-1]
    if tag in _SKIP_TAGS:
        return ' '
    if tag in ('disp-formula', 'inline-formula'):
        f = _formula(el)
        if not f:
            return ' '
        if tag == 'disp-formula':
            eqs.append(f)
            return f"\n\n{f}\n\n"
        return f" {f} "
    parts = [el.text or '']
    for c in el:
        parts.append(_flatten(c, eqs))
        parts.append(c.tail or '')
    return ''.join(parts)


_TEX_ENVS = ('equation', 'align', 'eqnarray', 'gather', 'multline',
             'displaymath', 'dmath')
_TEX_STRIP = re.compile(
    r'\\(?:label|ref|eqref|cite[a-z]*|footnote|vspace|hspace|noindent|'
    r'textbf|textit|emph|mathrm|mathbf|left|right|bigl|bigr|quad|qquad)'
    r'\s*(?:\{[^{}]*\})?')


def parse_latex(src: str) -> Tuple[str, str, List[str]]:
    """LaTeX source -> (text, title, equations).

    arXiv serves the author's original .tex at /e-print/. Its equations are
    exact, where the PDF's are shredded by the text extractor: a PDF of
    D d^2G/dx^2 + dG/dt + c dG/dx = G_in - aGI comes back as three fragments
    with the fractions and coefficients gone.
    """
    src = re.sub(r'(?<!\\)%.*', '', src)              # comments
    eqs: List[str] = []

    def grab(m):
        body = re.sub(r'\s+', ' ', m.group(1)).strip()
        body = _TEX_STRIP.sub('', body).strip()
        if 3 < len(body) < 400:
            eqs.append(body)
        return f"\n\n{body}\n\n"

    for env in _TEX_ENVS:
        src = re.sub(r'\\begin\{' + env + r'\*?\}(.*?)\\end\{' + env + r'\*?\}',
                     grab, src, flags=re.S)
    src = re.sub(r'\$\$(.+?)\$\$', grab, src, flags=re.S)
    src = re.sub(r'\\\[(.+?)\\\]', grab, src, flags=re.S)

    title = ''
    m = re.search(r'\\title\{(.+?)\}', src, re.S)
    if m:
        title = re.sub(r'\s+', ' ', _TEX_STRIP.sub('', m.group(1))).strip()

    body = src
    m = re.search(r'\\begin\{document\}(.*)', body, re.S)
    if m:
        body = m.group(1)
    body = re.sub(r'\\(?:section|subsection|subsubsection)\*?\{(.+?)\}',
                  lambda x: f"\n\n{SECTION_MARK}{x.group(1)}\n\n", body)
    body = re.sub(r'\\begin\{(figure|table|tabular|thebibliography)\*?\}'
                  r'.*?\\end\{\1\*?\}', ' ', body, flags=re.S)
    body = _TEX_STRIP.sub('', body)
    body = re.sub(r'\\[a-zA-Z]+\*?', ' ', body)
    body = re.sub(r'[{}]', ' ', body)
    body = re.sub(r'[ \t]{2,}', ' ', body)
    return body, title, eqs


def parse_jats(raw: str) -> Tuple[str, str, List[str]]:
    """PMC/JATS XML -> (text, title, equations). Sections and paragraphs are
    preserved so passages separate; equations survive as readable infix."""
    root = ET.fromstring(raw)
    eqs: List[str] = []
    out: List[str] = []
    title = ''
    for e in root.iter():
        if e.tag.split('}')[-1] == 'article-title':
            title = ' '.join(e.itertext()).strip()
            break
    if title:
        out.append(title)
    for e in root.iter():
        if e.tag.split('}')[-1] == 'abstract':
            out.append("Abstract")
            out.append(_flatten(e, eqs).strip())
            break
    body = next((e for e in root.iter()
                 if e.tag.split('}')[-1] == 'body'), None)
    if body is not None:
        for sec in body.iter('sec'):
            head = next((c for c in sec if c.tag.split('}')[-1] == 'title'), None)
            if head is not None:
                h = ' '.join(head.itertext()).strip()
                if h:
                    out.append(SECTION_MARK + h)
            for p in sec:
                if p.tag.split('}')[-1] in ('p', 'disp-formula', 'statement'):
                    t = _flatten(p, eqs).strip()
                    if t:
                        out.append(t)
        if not any(body.iter('sec')):
            out.append(_flatten(body, eqs).strip())
    text = "\n\n".join(x for x in out if x)
    return text, title, eqs


def _read(path_or_text: str) -> Tuple[str, str, List[str]]:
    import os
    p = path_or_text
    if p.startswith(('http://', 'https://')):
        req = urllib.request.Request(p, headers=UA)
        with urllib.request.urlopen(req, timeout=90) as r:
            raw = r.read()
        if raw[:2] == b'\x1f\x8b' or 'e-print' in p:
            tex = _untar_tex(raw)
            if tex:
                text, _t, eqs = parse_latex(tex)
                return text, p, eqs
        if raw[:5] == b'%PDF-':
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as fh:
                fh.write(raw)
                tmp = fh.name
            try:
                t, _, e = _read(tmp)
                return t, p, e
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        body = raw.decode('utf-8', 'replace')
        return _read_xml_or_text(body, p)
    if os.path.isfile(p):
        if p.lower().endswith('.tex'):
            with open(p, errors='ignore') as fh:
                t, _title, e = parse_latex(fh.read())
            return t, os.path.basename(p), e
        if p.lower().endswith(('.xml', '.nxml', '.jats')):
            with open(p, errors='ignore') as fh:
                return _read_xml_or_text(fh.read(), os.path.basename(p))
        if p.lower().endswith('.pdf'):
            try:
                from pypdf import PdfReader
            except ImportError:
                raise RuntimeError("reading PDFs needs pypdf: pip install pypdf")
            pages = []
            for page in PdfReader(p).pages:
                try:
                    pages.append(page.extract_text() or '')
                except Exception:
                    pages.append('')
            return "\n".join(pages), os.path.basename(p), []
        with open(p, errors='ignore') as fh:
            return fh.read(), os.path.basename(p), []
    return _read_xml_or_text(p, 'pasted text')


def _untar_tex(raw: bytes) -> str:
    """arXiv e-print bundles: gzip, usually containing a tar of .tex sources."""
    import gzip
    import io
    import tarfile
    try:
        data = gzip.decompress(raw)
    except Exception:
        data = raw
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data))
        parts = []
        for m in tf.getmembers():
            if m.isfile() and m.name.lower().endswith('.tex'):
                f = tf.extractfile(m)
                if f:
                    parts.append(f.read().decode('utf-8', 'replace'))
        return "\n".join(parts)
    except Exception:
        pass
    try:
        text = data.decode('utf-8', 'replace')
        return text if '\\begin{document}' in text or '\\documentclass' in text else ''
    except Exception:
        return ''


def _read_xml_or_text(body: str, source: str) -> Tuple[str, str, List[str]]:
    if body.lstrip()[:400].lstrip().startswith(('<?xml', '<article', '<!DOCTYPE')):
        try:
            text, _title, eqs = parse_jats(body)
            if text.strip():
                return text, source, eqs
        except Exception:
            pass                       # malformed XML: fall back to stripping
        return re.sub(r'<[^>]+>', ' ', body), source, []
    return body, source, []


def _clean(text: str) -> str:
    text = text.replace('\r', '\n')
    text = re.sub(r'-\n(?=[a-z])', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text)


def _split_sections(text: str) -> Dict[str, str]:
    sections, cur, buf = {}, 'preamble', []
    for line in text.split('\n'):
        m = HEADING.match(line.strip())
        if m and (m.group('declared') or len(line.strip()) < 60):
            sections[cur] = "\n".join(buf).strip()
            cur = (m.group('declared') or m.group(2) or '').strip().lower()
            buf = []
        else:
            buf.append(line)
    sections[cur] = "\n".join(buf).strip()
    return {k: v for k, v in sections.items() if v}


def _score_passage(text: str) -> Tuple[float, List[str]]:
    low, score, why = text.lower(), 0.0, []
    hits = [w for w in CLAIM_WORDS if w in low]
    if hits:
        score += 1.6 * min(len(hits), 5)
        why.append('claims: ' + ', '.join(hits[:4]))
    if [w for w in CONDITION_WORDS if w in low]:
        score += 2.0
        why.append('conditional')
    if re.search(r'\b(d\s*[a-z]\s*/\s*d\s*t|\\dot|=\s*0\s*$)', text):
        score += 2.0
        why.append('dynamics or an equilibrium condition')
    if re.search(r'[=<>≤≥∈∑∏∇∂]', text):
        score += 1.0
        why.append('formal content')
    if re.search(r'\bfor (all|every|any)\b.*\b(parameter|constant|rate|weight)',
                 low):
        score += 2.5
        why.append('quantified over ALL parameters — the strong kind')
    if len(text) < 120:
        score -= 1.5
    return score, why


def ingest(path_or_text: str, title: str = '') -> Paper:
    raw, source, marked_eqs = _read(path_or_text)
    text = _clean(raw)
    sections = _split_sections(text)
    blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
    passages = []
    for i, b in enumerate(blocks):
        m = STATEMENT.search(b[:120])
        kind = (m.group(1).lower() if m else
                ('dynamics' if re.search(r'(d\s*[a-z]\s*/\s*d\s*t|\\dot)', b)
                 else 'prose'))
        body = b if len(b) > 400 or i + 1 >= len(blocks) else b + "\n" + blocks[i + 1]
        s, why = _score_passage(body)
        if kind in ('theorem', 'proposition', 'lemma', 'corollary'):
            s += 3.0
            why.append('formal statement')
        elif kind in ('definition', 'assumption', 'condition'):
            s += 1.5
            why.append('names a condition')
        if s <= 1.5:
            continue
        passages.append(Passage(
            kind=kind, text=body.replace(SECTION_MARK, '').strip(),
            score=s, reasons=why,
            label=(f"{m.group(1).title()} {m.group(2) or ''}".strip() if m else '')))
    passages.sort(key=lambda p: -p.score)

    # Equations the publisher marked as such beat anything guessed by symbol
    # density, so they go first and the heuristic only fills in behind them.
    eqs, seen = [], set()
    for e in marked_eqs:
        e = re.sub(r'\s{2,}', ' ', e).strip()
        if e and e not in seen:
            seen.add(e)
            eqs.append(e)
    for line in text.split('\n'):
        t = line.strip()
        if not (4 <= len(t) <= 220):
            continue
        sym = sum(c in '=+-*/^∑∏∇∂≤≥<>' for c in t)
        let = sum(c.isalpha() for c in t)
        if sym >= 2 and let and sym / max(let, 1) > 0.12 and t not in seen:
            seen.add(t)
            eqs.append(t)

    if not title:
        head = [l.strip() for l in text.split('\n')[:12] if len(l.strip()) > 12]
        title = head[0][:140] if head else source
    return Paper(title=title, text=text, source=source, sections=sections,
                 passages=passages[:40], equations=eqs[:60])


# ============================================================================
# spec skeleton
# ============================================================================

SPEC_TEMPLATE = {
    "key": "my_structure",
    "title": "",
    "paper": "",
    "claim": "<the theorem, in one sentence>",
    "n": 12,
    "consts": {"K": 16, "alpha": 0.5},
    "params": {
        "W": {"shape": ["K", "n"], "init": "randn/sqrt(n)"},
        "s": {"shape": ["K"], "init": "randn*0.3", "transform": "softplus"},
    },
    "residual": "<expression in x, x0 and your parameters, returning (batch, n)>",
    "inject": "residual",
    "positive": False,
    "variants": [
        {"key": "holds", "label": "<the condition holds>", "structured": True},
        {"key": "broken", "label": "<the condition broken>",
         "params": {"s": {"transform": "free"}}},
    ],
    "notes": "",
}


def blank_spec(paper: Optional[Paper] = None, key: str = 'my_structure') -> dict:
    """A spec skeleton to fill in. The layer's output is the x where `residual`
    is zero; `x0` is the input. Variants may override constants and parameter
    transforms but never SHAPES — arms must have identical parameter counts or
    the comparison between them means nothing."""
    d = json.loads(json.dumps(SPEC_TEMPLATE))
    d['key'] = key
    if paper:
        d['title'] = paper.title[:80]
        d['paper'] = paper.source
        d['notes'] = 'Filled in by hand from the ranked passages. Validate it.'
    return d


CAVEAT = """
  These papers are a starting point for a MODEL, not an answer to your question.
  This finds the paper worth starting from; it cannot tell you why anyone is
  insulin resistant, or anything else about the world.

  The answerable version is narrower: given a published model of a system, which
  of its structural assumptions actually matter when it is fitted to data?
""".rstrip()


# ============================================================================

def _cli():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('question', nargs='*')
    p.add_argument('--read', help='a pdf/txt path or URL to ingest instead')
    p.add_argument('--show', type=int, default=5)
    p.add_argument('--json', action='store_true')
    p.add_argument('--quiet', action='store_true')
    a = p.parse_args()
    log = (lambda *x: None) if a.quiet else (lambda *x: print(*x))

    if a.read:
        paper = ingest(a.read)
        print(f"\n{paper.title}\n{'=' * 70}")
        print(paper.brief(n_passages=a.show))
        print("\nSPEC SKELETON\n" + json.dumps(blank_spec(paper), indent=2))
        return
    if not a.question:
        p.error('give a question, or --read a file')

    q = ' '.join(a.question)
    print(f"\nQUESTION: {q}\n" + "=" * 70)
    res = discover(q, log=log)
    if a.json:
        print(json.dumps({k: [h.__dict__ for h in v]
                          for k, v in res.items() if isinstance(v, list)},
                         indent=1, default=str))
        return
    for name, note in (('foundational', 'What the hubs keep pointing back at.'),
                       ('direct', 'Modelling papers from the search itself.'),
                       ('hubs', 'Signposts, not destinations.'),
                       ('preprints', 'arXiv.')):
        hits = res.get(name) or []
        if not hits:
            continue
        print(f"\n{name.upper()} — {note}\n" + "-" * 70)
        for i, h in enumerate(hits[:a.show], 1):
            print(f"  {i:2d}. {h.line()}")
            print(f"      → {'; '.join(h.reasons[:2])}")
            if h.source_url:
                print(f"      python paper_to_spec.py --read \"{h.source_url}\"")
            print()
    print(CAVEAT)


if __name__ == '__main__':
    _cli()
