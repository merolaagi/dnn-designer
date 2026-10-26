"""Medical: answers to health questions from the medical literature, with the sources named.

Where the evidence comes from, all free and public:

    PubMed         NCBI's index of the medical literature: guidelines, systematic reviews, trials,
                   with abstracts (E-utilities, no key needed)
    Europe PMC     the same literature plus preprints, and which articles have free full text
    openFDA        the FDA drug label: uses, dosing as the label states it, warnings, interactions,
                   use in children, pregnancy
    your library   guidelines, protocols and papers you add (Settings → Medical), searched here on
                   the Mac: by meaning with an Ollama embedding model when one is installed, by
                   keywords otherwise. Every account starts with three guideline summaries carried
                   over from MediConsult AI, which this replaces: nothing else needs to run.

The question is also read for the patient it is about: age, conditions, medicines, allergies ("my
mom is 72, on warfarin, allergic to penicillin"). Each medicine's label is checked against the
others and against the allergies, and anything found is put at the top of the answer as a safety
check, the way MediConsult's safety alerts were.

Before any of that, a request is screened for an emergency happening now ("I have crushing chest
pain", "my dad can't breathe"). An emergency is answered at once, with what to do and who to call,
and nothing is looked up: a minute of searching is a minute lost.

Each answer is written for whoever is asking: plain words for most people, or, for an account set
to clinician, MediConsult's structure: clinical assessment, evidence summary, recommendations
with their strength, considerations, follow-up. Either way it cites only what
was found, says how strong the evidence is, and says when to see someone. It never gives a child's
medicine dose from memory: only what the product's label says, or a pointer to the label, a
pharmacist or the child's doctor.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentlab as AL

USER_AGENT = "deep-network-designer-medical (personal use)"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
OPENFDA = "https://api.fda.gov/drug/label.json"
AUDIENCES = ("public", "clinician")
PAUSE = 0.35                # NCBI asks for no more than three calls a second
KINDS = {"any": "", "guideline": "Guideline[pt] OR Practice Guideline[pt]",
         "review": "Systematic Review[pt] OR Meta-Analysis[pt] OR Review[pt]",
         "trial": "Randomized Controlled Trial[pt] OR Clinical Trial[pt]"}

GUIDE = """You answer health and medical questions from the evidence, for the person asking.

How you work:
1. The literature has usually been searched for you already (it is below the question). If it is thin or
   off the point, call search_literature again with better terms (the condition, the treatment, the
   population), or with kind "guideline" or "review" for the strongest evidence. Use read_article for the
   full abstract of one that matters. For a medicine, call drug_label for what the FDA label says.
   search_library searches the person's own library (guidelines and protocols they added); cite what
   comes from it like any other source, and say when a library item is only a summary.
   If there are safety checks (an interaction or an allergy found on a label), put them first.
2. Answer the question asked, first, in plain words (or clinical terms if the person is a clinician).
   Then what the evidence says and how strong it is: a guideline or a systematic review counts for more
   than one small study, and say which it is. Give numbers when the sources give them.
3. Cite every claim that comes from a source with its number, like [2], and end with the sources as a short
   list: number, title, journal and year, and the link. Cite only sources you were given. Never invent a
   study, a statistic, a dose or a link. If the evidence does not settle it, say so plainly.
4. Say when to see a doctor, and how soon, when it matters: new or worsening symptoms, red flags, anything
   that needs examining. You are not their doctor and cannot examine them; say so once, briefly, not as a
   wall of disclaimers.
5. Medicines for a child: never give a dose from memory. Report exactly what the product's label says for
   the child's age and weight (drug_label), or, if you do not have the label or the child's age and weight,
   say so and point them to the label's dosing chart, a pharmacist or the child's doctor. A prescribed
   medicine: the dose on the prescription label, and the pharmacist for questions.
6. If at any point it sounds like an emergency now, open with what to do (call 911, or 988 for a crisis)
   before anything else.
Match the length to the question: a symptom question gets a short answer; a question about the evidence gets
the trials, their effect sizes, the guideline grades, the risks and the gaps, each cited."""

# an emergency happening now, said in the first person or about someone with them: answered at once
_NOW = r"(i|i'm|im|i am|i've|ive|my\s+\w+(\s+\w+)?|he|she|they|we|someone|somebody|a friend|my (son|daughter|baby|child|kid|wife|husband|mom|mum|dad|father|mother|friend|partner))"
EMERGENCY = [
    (re.compile(_NOW + r"\b.{0,40}\b(chest (pain|pressure|tightness)|crushing (pain|pressure))", re.I), "chest"),
    (re.compile(_NOW + r"\b.{0,40}\b(can'?t|cannot|can not|struggling to|unable to|hard to) breathe?", re.I), "breathing"),
    (re.compile(_NOW + r"\b.{0,40}\b(face (is )?droop|slurred speech|can'?t (move|feel) (my|his|her|their) (arm|leg|face)|"
                r"(having|had) a stroke|stroke symptoms)", re.I), "stroke"),
    (re.compile(_NOW + r"\b.{0,40}\b(unconscious|passed out and|won'?t wake|not breathing|no pulse|turning blue|seizure (now|right now|won'?t stop))", re.I), "collapse"),
    (re.compile(_NOW + r"\b.{0,40}\b(overdos|took (too many|a whole bottle)|swallowed (poison|bleach|pills))", re.I), "poison"),
    (re.compile(_NOW + r"\b.{0,40}\b(bleeding (heavily|a lot|won'?t stop)|won'?t stop bleeding)", re.I), "bleeding"),
    (re.compile(_NOW + r"\b.{0,40}\b(throat (is )?(closing|swelling)|anaphyla|tongue (is )?swelling)", re.I), "allergy"),
    (re.compile(r"\b(i (want|wanna|am going|'m going) to (die|kill myself|end (it|my life))|kill myself|end my life|"
                r"i('m| am)( feeling)? suicidal|i('ve| have) been (cutting|hurting) myself|i want to hurt myself)\b", re.I), "crisis"),
]
# asking about a condition is not having one: these get the full, researched answer
_ASKING = re.compile(r"\b(how (do|can|would|should) (i|you|we) (tell|know|recogni[sz]e|spot)|what are the (signs|symptoms)|"
                     r"signs of|symptoms of|warning signs|what (causes|is|are)|why do(es)?|explain|in general|"
                     r"for (a|my) (class|exam|essay|paper))\b", re.I)
ADVICE = {
    "chest": "Chest pain or pressure can be a heart attack. Call 911 now; don't drive yourself. If you are not allergic "
             "and have been told you may, chew one regular aspirin (325 mg) while you wait.",
    "breathing": "Trouble breathing is an emergency. Call 911 now. Sit upright; if there is a prescribed inhaler or "
                 "epinephrine pen, use it.",
    "stroke": "These can be signs of a stroke, and every minute counts. Call 911 now, and note the time the symptoms "
              "started. Don't give food, drink or medicine.",
    "collapse": "Call 911 now. If they aren't breathing normally, start CPR: hard, fast pushes in the centre of the chest, "
                "100 to 120 a minute; the dispatcher will talk you through it.",
    "poison": "Call 911 now if they are drowsy, confused, having trouble breathing or seizing. Otherwise call Poison "
              "Control at 1-800-222-1222, day or night. Keep the bottle or packet to tell them what was taken.",
    "bleeding": "Call 911 now. Press hard on the wound with a clean cloth and keep pressing; don't lift it to look.",
    "allergy": "This can be anaphylaxis. Use an epinephrine pen if there is one (outer thigh), then call 911 now, even "
               "if it gets better.",
    "crisis": "I'm really sorry you're going through this. You can call or text 988, the Suicide & Crisis Lifeline, any "
              "time, day or night, to talk with someone now. If you're in danger right now, call 911. If you'd like, "
              "tell me what's going on; I'm here to listen.",
}


class MedicalError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# settings (per account): who answers are written for, and MediConsult
# --------------------------------------------------------------------------

def _path() -> Path:
    return AL._dir() / "medical.json"


def _load() -> Dict[str, Any]:
    try:
        d = json.loads(_path().read_text())
    except (OSError, ValueError):
        d = {}
    d.setdefault("audience", "public")
    return {"audience": d["audience"] if d["audience"] in AUDIENCES else "public"}


def view() -> Dict[str, Any]:
    return {"audience": _load()["audience"], "library": library_listing(), "embeddings": embed_model() or None,
            "pdf": _pdf_reader() is not None}


def save(change: Dict[str, Any]) -> Dict[str, Any]:
    d = _load()
    if change.get("audience") is not None:
        if change["audience"] not in AUDIENCES:
            raise ValueError("Answers are written for the public or for a clinician.")
        d["audience"] = change["audience"]
    _path().write_text(json.dumps(d, indent=1))
    return view()


CLINICIAN_FORMAT = ("The person asking is a clinician. Use clinical terms, doses as the sources give them, and grades of "
                    "evidence, in this order (short headings): Clinical assessment; Evidence summary; Recommendations "
                    "(each with the strength of its evidence); Considerations (patient factors, contraindications, "
                    "alternatives); Follow-up (monitoring, referral, work-up); Sources. Clinical judgment comes first; say "
                    "so once, briefly.")


def with_audience(task: str) -> str:
    who = _load()["audience"]
    return task + ("\n\n" + CLINICIAN_FORMAT if who == "clinician" else
                   "\n\nThe person asking is not a clinician: plain words, explain any medical term you must use.")


# --------------------------------------------------------------------------
# the emergency screen
# --------------------------------------------------------------------------

def emergency(text: str) -> Optional[str]:
    """What to do now, if the request describes an emergency happening now; None otherwise."""
    text = text or ""
    for pattern, kind in EMERGENCY:
        if pattern.search(text) and (kind == "crisis" or not _ASKING.search(text)):
            return ADVICE[kind]
    return None


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

def _get(url: str, timeout: float = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as reply:
        return reply.read()


def _clean(text: Any, limit: int = 1500) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text or ""))).strip()[:limit]


def pubmed(query: str, kind: str = "any", years: int = 0, count: int = 6) -> List[Dict[str, Any]]:
    """The best-matching PubMed articles, with abstracts."""
    term = query.strip()
    if KINDS.get(kind):
        term = f"({term}) AND ({KINDS[kind]})"
    if years:
        term += f" AND (\"last {int(years)} years\"[dp])"
    base = f"tool=dnn-designer&db=pubmed"
    found = json.loads(_get(f"{EUTILS}/esearch.fcgi?{base}&retmode=json&sort=relevance&retmax={count}&term="
                            + urllib.parse.quote(term)))
    ids = (found.get("esearchresult") or {}).get("idlist") or []
    if not ids:
        return []
    time.sleep(PAUSE)
    root = ET.fromstring(_get(f"{EUTILS}/efetch.fcgi?{base}&retmode=xml&id=" + ",".join(ids)))
    out = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID") or ""
        a = art.find(".//Article")
        if a is None:
            continue
        abstract = " ".join(((t.get("Label") + ": ") if t.get("Label") else "") + "".join(t.itertext())
                            for t in a.findall(".//Abstract/AbstractText"))
        year = a.findtext(".//Journal/JournalIssue/PubDate/Year") or \
            (a.findtext(".//Journal/JournalIssue/PubDate/MedlineDate") or "")[:4]
        types = [t.text for t in a.findall(".//PublicationTypeList/PublicationType") if t.text]
        out.append({"source": "PubMed", "pmid": pmid, "title": _clean("".join(a.find("ArticleTitle").itertext())
                                                                       if a.find("ArticleTitle") is not None else "", 300),
                    "journal": _clean(a.findtext(".//Journal/ISOAbbreviation") or a.findtext(".//Journal/Title"), 120),
                    "year": year, "types": [t for t in types if t != "Journal Article"][:4],
                    "abstract": _clean(abstract, 1800), "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
    order = {p: i for i, p in enumerate(ids)}
    return sorted(out, key=lambda r: order.get(r["pmid"], 99))


def europepmc(query: str, count: int = 4) -> List[Dict[str, Any]]:
    """Europe PMC: the same literature plus preprints, with whether the full text is free."""
    url = (f"{EPMC}/search?format=json&resultType=core&sort=CITED%20desc&pageSize={count}&query="
           + urllib.parse.quote(f"({query}) AND HAS_ABSTRACT:y"))
    hits = (json.loads(_get(url)).get("resultList") or {}).get("result") or []
    out = []
    for r in hits:
        link = (f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/" if r.get("pmid") else
                f"https://doi.org/{r['doi']}" if r.get("doi") else f"https://europepmc.org/article/{r.get('source')}/{r.get('id')}")
        out.append({"source": "Europe PMC", "pmid": r.get("pmid") or "", "title": _clean(r.get("title"), 300),
                    "journal": _clean((r.get("journalInfo") or {}).get("journal", {}).get("isoabbreviation")
                                      or r.get("bookOrReportDetails", {}).get("publisher") or "", 120),
                    "year": str(r.get("pubYear") or ""), "types": (r.get("pubTypeList") or {}).get("pubType", [])[:3],
                    "abstract": _clean(r.get("abstractText"), 1800), "url": link,
                    "free_full_text": r.get("isOpenAccess") == "Y"})
    return out


STOP = set("""a an the is are was were be been being am do does did done doing have has had having i me my mine we us our
you your he him his she her it its they them their what which who whom whose when where why how that this these those
there here of to in on at by for with about against between into through during before after above below from up down
out off over under again further then once can could should would will shall may might must not no nor only own same so
than too very just also any all both each few more most other some such please tell know want need get got take taking
took safe okay ok good bad best better worse really much many lot thing things way someone something anything ever
doctor see seeing still since week weeks day days month months year years old ago now today evidence say says
said i've i'm ive im one two three four five six seven eight nine ten latest current new recent research study studies""".split())


def terms(question: str, keep: int = 7) -> str:
    """Search terms from a question: the medical words, without the question around them."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+'-]*", question)
    picked = [w for w in words if w.lower() not in STOP and len(w) > 2]
    return " ".join(list(dict.fromkeys(picked))[:keep])


def literature(query: str, kind: str = "any", years: int = 0, count: int = 6) -> List[Dict[str, Any]]:
    """PubMed first, from the question's terms (fewer terms if nothing matches); Europe PMC when PubMed has little
    or cannot be reached. Duplicates dropped."""
    out, problems = [], []
    q = terms(query) or query
    try:
        out = pubmed(q, kind, years, count)
        if not out and len(q.split()) > 3:
            time.sleep(PAUSE)
            out = pubmed(" ".join(q.split()[:3]), kind, years, count)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"PubMed: {type(exc).__name__}")
    if len(out) < 3:
        try:
            seen = {r["pmid"] for r in out if r["pmid"]}
            out += [r for r in europepmc(q, count) if not r["pmid"] or r["pmid"] not in seen]
        except Exception as exc:  # noqa: BLE001
            problems.append(f"Europe PMC: {type(exc).__name__}")
    if not out and problems:
        raise MedicalError("Could not reach the medical literature (" + "; ".join(problems) + ").")
    return out[:count + 2]


def article(pmid: str) -> Dict[str, Any]:
    rows = pubmed(f"{re.sub(r'[^0-9]', '', pmid)}[uid]", count=1)
    if not rows:
        raise MedicalError(f"PubMed has no article {pmid}.")
    return rows[0]


LABEL_PARTS = [("indications_and_usage", "Uses"), ("dosage_and_administration", "Dosing (as the label states it)"),
               ("pediatric_use", "Children"), ("boxed_warning", "Boxed warning"), ("contraindications", "Do not use"),
               ("warnings_and_cautions", "Warnings"), ("warnings", "Warnings"), ("drug_interactions", "Interactions"),
               ("pregnancy", "Pregnancy"), ("do_not_use", "Do not use"), ("ask_doctor", "Ask a doctor before use if"),
               ("stop_use", "Stop use and ask a doctor if"), ("adverse_reactions", "Side effects")]


def drug_label(name: str) -> Dict[str, Any]:
    """What the FDA label says for a medicine, by its generic or brand name."""
    n = re.sub(r"[^\w\s-]", "", name).strip()
    if not n:
        raise MedicalError("Name the medicine.")
    q = f'openfda.generic_name:"{n}" OR openfda.brand_name:"{n}"'
    try:
        rows = json.loads(_get(OPENFDA + "?limit=1&search=" + urllib.parse.quote(q))).get("results") or []
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise MedicalError(f"The FDA has no label for {name}.") from None
        raise
    if not rows:
        raise MedicalError(f"The FDA has no label for {name}.")
    r = rows[0]
    fda = r.get("openfda") or {}
    parts = {label: _clean(" ".join(r[k]), 1400) for k, label in LABEL_PARTS if r.get(k)}
    return {"source": "FDA label (openFDA)", "name": ", ".join((fda.get("brand_name") or [])[:2] or [name]),
            "generic": ", ".join((fda.get("generic_name") or [])[:2]), "parts": parts,
            "url": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={urllib.parse.quote(n)}"}


# --------------------------------------------------------------------------
# your library: guidelines, protocols and papers you add, searched on this Mac
# --------------------------------------------------------------------------

STARTER = Path(__file__).resolve().parent / "medical_library.json"
CHUNK, OVERLAP = 1200, 200
EMBED_MODELS = ("nomic-embed-text", "mxbai-embed-large", "bge-m3", "snowflake-arctic-embed", "all-minilm")
MAX_DOC = 400_000                    # characters kept from one document
_LIB_LOCK = threading.Lock()


def _lib_path() -> Path:
    return AL._dir() / "medical_library.json"


def _lib() -> Dict[str, Any]:
    try:
        return json.loads(_lib_path().read_text())
    except (OSError, ValueError):
        pass
    # a new account starts with the starter library
    lib = {"docs": []}
    try:
        for d in json.loads(STARTER.read_text())["documents"]:
            lib["docs"].append(_make_doc(d["title"], d["text"], d.get("source", ""), d.get("year", ""), d.get("kind", ""),
                                         d.get("url", ""), d.get("note", ""), starter=True))
    except (OSError, ValueError, KeyError):
        pass
    _lib_store(lib)
    return lib


def _lib_store(lib: Dict[str, Any]) -> None:
    _lib_path().write_text(json.dumps(lib))


def _chunks(text: str) -> List[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= CHUNK:
        return [text] if text else []
    out, i = [], 0
    while i < len(text):
        end = min(len(text), i + CHUNK)
        cut = text.rfind(". ", i + CHUNK // 2, end) if end < len(text) else end
        end = cut + 1 if cut > i else end
        out.append(text[i:end].strip())
        if end >= len(text):
            break
        i = max(end - OVERLAP, i + 1)
    return [c for c in out if c]


def _make_doc(title: str, text: str, source: str = "", year: str = "", kind: str = "", url: str = "", note: str = "",
              starter: bool = False) -> Dict[str, Any]:
    text = str(text or "")[:MAX_DOC]
    return {"id": "ml" + hashlib.sha1((title + text[:2000]).encode()).hexdigest()[:10], "title": _clean(title, 300) or "Untitled",
            "source": _clean(source, 160), "year": _clean(year, 12), "kind": _clean(kind, 40) or "document",
            "url": url if re.match(r"^https?://", url or "") else "", "note": _clean(note, 300), "starter": starter,
            "added": time.time(), "chars": len(text), "chunks": [{"text": c} for c in _chunks(text)]}


def library_listing() -> List[Dict[str, Any]]:
    return [{k: d[k] for k in ("id", "title", "source", "year", "kind", "url", "note", "starter", "added", "chars")}
            | {"pieces": len(d["chunks"]), "by_meaning": all("vec" in c for c in d["chunks"])} for d in _lib()["docs"]]


def _pdf_reader():
    try:
        import pypdf
        return pypdf
    except ImportError:
        return None


def text_of(data: bytes, filename: str) -> str:
    """The text of an uploaded document: PDF, plain text, Markdown or HTML."""
    name = (filename or "").lower()
    if name.endswith(".pdf") or data[:5] == b"%PDF-":
        pdf = _pdf_reader()
        if pdf is None:
            raise MedicalError("Reading PDFs needs pypdf on the Mac: in ~/Sites/new_dnn run "
                               ".venv/bin/python -m pip install pypdf, then ./serve.sh.")
        import io
        try:
            reader = pdf.PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # noqa: BLE001
            raise MedicalError(f"That PDF would not open ({type(exc).__name__}).") from None
        if len(text.strip()) < 200:
            raise MedicalError("That PDF has almost no text in it (a scan?), so there is nothing to search.")
        return text
    text = data.decode("utf-8", "replace")
    if name.endswith((".html", ".htm")) or "<html" in text[:500].lower():
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
    return text


def add_document(title: str, text: str, source: str = "", year: str = "", kind: str = "", url: str = "") -> Dict[str, Any]:
    if len((text or "").strip()) < 80:
        raise MedicalError("Paste the document's text, or upload it: there is too little to search.")
    doc = _make_doc(title or (text.strip().split("\n")[0][:120]), text, source, year, kind, url)
    model = embed_model()
    if model:
        try:
            for c, v in zip(doc["chunks"], embed([c["text"] for c in doc["chunks"]], model)):
                c["vec"] = v
            doc["model"] = model
        except Exception:  # noqa: BLE001 - keywords still find it
            for c in doc["chunks"]:
                c.pop("vec", None)
    with _LIB_LOCK:
        lib = _lib()
        lib["docs"] = [d for d in lib["docs"] if d["id"] != doc["id"]] + [doc]
        _lib_store(lib)
    return view()


def remove_document(doc_id: str) -> Dict[str, Any]:
    with _LIB_LOCK:
        lib = _lib()
        keep = [d for d in lib["docs"] if d["id"] != doc_id]
        if len(keep) == len(lib["docs"]):
            raise KeyError(doc_id)
        lib["docs"] = keep
        _lib_store(lib)
    return view()


def embed_model() -> str:
    """An embedding model installed in Ollama, or '' (the library is then searched by keywords)."""
    try:
        import providers as P
        names = [m["name"] for m in P.ollama_status().get("models") or [] if m.get("name")]
    except Exception:  # noqa: BLE001
        return ""
    for want in EMBED_MODELS:
        for n in names:
            if n.split(":")[0] == want:
                return n
    return ""


def embed(texts: List[str], model: str) -> List[List[float]]:
    import providers as P
    root = P._ollama_root()
    out: List[List[float]] = []
    for i in range(0, len(texts), 32):
        reply = P._request(root + "/api/embed", {"model": model, "input": texts[i:i + 32]}, {}, 120)
        out += reply.get("embeddings") or []
    if len(out) != len(texts):
        raise MedicalError("The embedding model did not answer for every piece.")
    return out


def _words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1]


def search_library(query: str, count: int = 4) -> List[Dict[str, Any]]:
    """The pieces of your library that best answer the query: by meaning and keywords together when an
    embedding model is installed, by keywords (BM25) otherwise."""
    lib = _lib()
    pieces = [(d, c) for d in lib["docs"] for c in d["chunks"]]
    if not pieces:
        return []
    q = _words(query)
    toks = [_words(c["text"] + " " + d["title"]) for d, c in pieces]
    n, avg = len(toks), sum(len(t) for t in toks) / max(1, len(toks))
    df: Dict[str, int] = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    bm, hits = [], []
    for t in toks:
        tf: Dict[str, int] = {}
        for w in t:
            tf[w] = tf.get(w, 0) + 1
        score = 0.0
        hits.append(sum(1 for w in set(q) if w in tf))
        for w in set(q):
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                score += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(t) / avg))
        bm.append(score)
    top = max(bm) or 1.0
    scores = [b / top for b in bm]
    model = embed_model() if any("vec" in c for _, c in pieces) else ""
    if model:
        try:
            qv = embed([query], model)[0]
            qn = math.sqrt(sum(x * x for x in qv)) or 1.0
            for i, (_, c) in enumerate(pieces):
                v = c.get("vec")
                if v and len(v) == len(qv):
                    cos = sum(a * b for a, b in zip(qv, v)) / (qn * (math.sqrt(sum(x * x for x in v)) or 1.0))
                    scores[i] = 0.4 * scores[i] + 0.6 * max(0.0, cos)
        except Exception:  # noqa: BLE001 - keywords alone, then
            pass
    need = max(1, math.ceil(len(set(q)) / 2))          # one stray shared word ("treatment") is not a match
    ranked = sorted(range(n), key=lambda i: -scores[i])
    out, used = [], set()
    for i in ranked:
        d, c = pieces[i]
        strong = (model and scores[i] > 0.5) or (hits[i] >= need and scores[i] >= 0.3)
        if not strong or d["id"] in used:
            continue
        used.add(d["id"])
        out.append({"source": "Your library", "pmid": "", "title": d["title"], "journal": d.get("source") or "your library",
                    "year": d.get("year") or "", "types": [d.get("kind")] if d.get("kind") else [], "url": d.get("url") or "",
                    "abstract": c["text"][:1800], "note": d.get("note") or "", "score": round(scores[i], 3)})
        if len(out) >= count:
            break
    return out


# --------------------------------------------------------------------------
# the patient the question is about, and what their medicines' labels say about each other
# --------------------------------------------------------------------------

ALLERGY_CLASS = {"penicillin": ["penicillin", "amoxicillin", "ampicillin", "beta-lactam", "cephalosporin"],
                 "sulfa": ["sulfonamide", "sulfa", "sulfamethoxazole"], "nsaid": ["nsaid", "ibuprofen", "naproxen", "aspirin"],
                 "aspirin": ["aspirin", "salicylate", "nsaid"], "codeine": ["codeine", "opioid"], "morphine": ["morphine", "opioid"]}


def patient_of(text: str) -> Dict[str, Any]:
    """Age, sex, conditions, medicines and allergies, as the question states them. Nothing is guessed."""
    t = " " + (text or "").lower() + " "
    age = None
    m = re.search(r"\b(\d{1,3})[- ]?(?:years?[- ]old|yrs?[- ]old|y/?o|yo)\b|\b(?:aged?|is|she's|he's|i'm|i am)\s+(\d{1,3})\b(?!\s*(?:mg|kg|lb|%|minutes|hours|days|weeks))", t)
    if m:
        v = int(m.group(1) or m.group(2))
        age = v if 0 < v < 120 else None
    months = re.search(r"\b(\d{1,2})[- ]?months?[- ]old\b", t)
    if months and age is None:
        age = round(int(months.group(1)) / 12, 1)
    sex = "female" if re.search(r"\b(she|her|woman|female|mother|mom|mum|wife|daughter|sister|grandmother|pregnant)\b", t) \
        else "male" if re.search(r"\b(he|him|his|man|male|father|dad|husband|son|brother|grandfather)\b", t) else None
    allergies = re.findall(r"allerg(?:ic|y|ies) to ([a-z][a-z -]{2,30}?)(?=[,.;]| and | but |$)", t)
    allergies += re.findall(r"\b([a-z]{4,20}) allerg(?:y|ic)\b", t)
    conditions = []
    for c in re.findall(r"\b(?:has|have|had|with|diagnosed with|history of|h/o)\s+([a-z0-9][a-z0-9' -]{3,60}?)(?=[,.;?]| but | on | taking | is | and (?:is|takes|takes|on)\b|$)", t):
        conditions += [x.strip() for x in re.split(r"\s+and\s+", c) if len(x.strip()) > 3]
    meds = _drug_names(text or "")
    meds += [w for w in re.findall(r"\b(?:on|taking|takes|prescribed)\s+([a-z][a-z-]{3,})", t) if w not in meds]
    junk = {"medication", "medications", "medicine", "meds", "pills", "something", "them", "this", "that", "it", "insurance",
            "time", "vacation", "call", "duty"}
    allergens = {w for a in allergies for w in a.split()}
    return {"age": age, "sex": sex, "conditions": [c.strip() for c in conditions if c.strip() not in junk][:6],
            "medicines": [x for x in dict.fromkeys(meds) if x not in junk and x not in allergens][:6],
            "allergies": [a.strip() for a in dict.fromkeys(allergies) if a.strip() not in junk][:4]}


def safety_checks(patient: Dict[str, Any], labels: List[Dict[str, Any]]) -> List[str]:
    """What the labels say about these medicines together, these allergies and this age. Only label text is quoted."""
    alerts = []
    meds = patient.get("medicines") or []
    for lab in labels:
        parts = lab["parts"]
        name = lab.get("generic") or lab["name"]
        inter = (parts.get("Interactions") or "").lower()
        for other in meds:
            other_lab = next((l for l in labels if other in (l.get("generic") or "").lower() or other in l["name"].lower()), None)
            other_names = {other} | ({(other_lab.get("generic") or "").lower()} if other_lab else set())
            if lab is other_lab or other.lower() in name.lower():
                continue
            hit = next((w for w in other_names if w and w in inter), None)
            if hit:
                i = inter.index(hit)
                alerts.append(f"Interaction: the {name} label mentions {other}: \"…{_clean(inter[max(0, i - 160):i + 220], 400)}…\" "
                              f"({lab['source']})")
        avoid = " ".join(parts.get(k, "") for k in ("Do not use", "Boxed warning", "Warnings")).lower()
        for allergen in patient.get("allergies") or []:
            words = ALLERGY_CLASS.get(allergen.split()[0], [allergen.split()[0]])
            hit = next((w for w in words if w in avoid or w in name.lower()), None)
            if hit:
                alerts.append(f"Allergy: they are allergic to {allergen}, and the {name} label says: "
                              f"\"…{_clean(avoid[max(0, avoid.find(hit) - 120):avoid.find(hit) + 200], 340)}…\" ({lab['source']})"
                              if hit in avoid else f"Allergy: they are allergic to {allergen}, and {name} is in that family.")
        if parts.get("Boxed warning"):
            alerts.append(f"Boxed warning on {name}: {_clean(parts['Boxed warning'], 300)} ({lab['source']})")
        age = patient.get("age")
        if age is not None and age < 12 and parts.get("Children"):
            alerts.append(f"Child aged {age}: the {name} label on children says: {_clean(parts['Children'], 300)} "
                          "Dose only by the label for the child's age and weight.")
    return list(dict.fromkeys(alerts))[:6]


# --------------------------------------------------------------------------
# what the agent is handed before it starts
# --------------------------------------------------------------------------

def _drug_names(text: str) -> List[str]:
    """Medicines named in the question, for the label: words before 'dose', after 'take', or common endings."""
    t = text.lower()
    names = re.findall(r"\b([a-z][a-z-]{3,})\b(?=\s+(?:dose|dosage|dosing|for (?:a|my)|side effects?|interact))", t)
    names += re.findall(r"\b(?:take|taking|took|give|giving|on)\s+([a-z][a-z-]{3,})\b", t)
    names += re.findall(r"\b([a-z]+(?:pril|sartan|olol|statin|mab|nib|azole|cillin|mycin|floxacin|prazole|"
                        r"tidine|dipine|gliptin|gliflozin|glutide|profen|tadine|zepam|xetine|triptan))\b", t)
    common = {"tylenol", "acetaminophen", "paracetamol", "ibuprofen", "advil", "motrin", "aspirin", "naproxen", "aleve",
              "benadryl", "diphenhydramine", "metformin", "insulin", "zyrtec", "cetirizine", "claritin", "loratadine",
              "amoxicillin", "prednisone", "melatonin", "warfarin", "eliquis", "apixaban", "ozempic", "semaglutide"}
    names += [w for w in re.findall(r"[a-z]+", t) if w in common]
    stop = {"this", "that", "with", "much", "many", "what", "medicine", "medication", "pills", "daily", "care", "them",
            "more", "less", "some", "child", "baby", "kids", "while"}
    return list(dict.fromkeys(n for n in names if n not in stop))[:5]


def gather(question: str) -> Dict[str, Any]:
    """Your library, the literature, the labels of any medicine named, and the safety checks, fetched before the
    agent runs."""
    found: Dict[str, Any] = {"library": [], "articles": [], "labels": [], "patient": patient_of(question),
                             "alerts": [], "problems": []}
    try:
        found["library"] = search_library(question)
    except Exception as exc:  # noqa: BLE001
        found["problems"].append(f"your library: {type(exc).__name__}")
    try:
        found["articles"] = literature(question)
    except MedicalError as exc:
        found["problems"].append(str(exc))
    for name in found["patient"]["medicines"][:4]:
        try:
            found["labels"].append(drug_label(name))
        except Exception:  # noqa: BLE001 - not every word that looks like a medicine is one
            continue
    found["alerts"] = safety_checks(found["patient"], found["labels"])
    return found


def articles_text(rows: List[Dict[str, Any]], start: int = 1) -> str:
    out = []
    for i, r in enumerate(rows, start):
        kind = ", ".join(r.get("types") or [])
        out.append(f"[{i}] {r['title']} ({r.get('journal') or r['source']}, {r.get('year') or 'n.d.'})"
                   + (f" [{kind}]" if kind else "") + (" [your library]" if r["source"] == "Your library" else "")
                   + (f" {r['url']}" if r.get("url") else "")
                   + ("  · free full text" if r.get("free_full_text") else "")
                   + (f"\n    Note: {r['note']}" if r.get("note") else "")
                   + f"\n    {r.get('abstract') or '(no abstract)'}")
    return "\n".join(out)


def label_text(lab: Dict[str, Any]) -> str:
    head = f"{lab['source']}: {lab['name']}" + (f" ({lab['generic']})" if lab.get("generic") else "") + f" {lab['url']}"
    return head + "".join(f"\n  {k}: {v}" for k, v in lab["parts"].items())


def gathered_text(found: Dict[str, Any]) -> str:
    parts = []
    if found.get("alerts"):
        parts.append("Safety checks from the labels (put these first in the answer):\n" + "\n".join("- " + a for a in found["alerts"]))
    pt = found.get("patient") or {}
    said = [f"age {pt['age']}" if pt.get("age") is not None else "", pt.get("sex") or "",
            ("conditions: " + ", ".join(pt["conditions"])) if pt.get("conditions") else "",
            ("medicines: " + ", ".join(pt["medicines"])) if pt.get("medicines") else "",
            ("allergies: " + ", ".join(pt["allergies"])) if pt.get("allergies") else ""]
    if any(said):
        parts.append("The patient, as the question describes them: " + "; ".join(x for x in said if x))
    rows = (found.get("library") or []) + (found.get("articles") or [])
    if rows:
        parts.append("Sources (cite these by number):\n" + articles_text(rows))
    for lab in found.get("labels") or []:
        parts.append(label_text(lab))
    if found.get("problems"):
        parts.append("Could not reach: " + "; ".join(found["problems"]))
    if not (rows or found.get("labels")):
        parts.append("Nothing was found yet: search again with different terms before answering, and if there is still "
                     "nothing, say the evidence could not be reached rather than answering from memory.")
    return "\n\n".join(parts)
