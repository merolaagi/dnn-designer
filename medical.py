"""Medical: answers to health questions from the medical literature, with the sources named.

Where the evidence comes from, all free and public:

    PubMed         NCBI's index of the medical literature: guidelines, systematic reviews, trials,
                   with abstracts (E-utilities, no key needed)
    Europe PMC     the same literature plus preprints, and which articles have free full text
    openFDA        the FDA drug label: uses, dosing as the label states it, warnings, interactions,
                   use in children, pregnancy
    MediConsult    your own MediConsult AI, when it is running and connected (Settings → Medical):
                   its answer from its own library is one more source, named as such

Before any of that, a request is screened for an emergency happening now ("I have crushing chest
pain", "my dad can't breathe"). An emergency is answered at once, with what to do and who to call,
and nothing is looked up: a minute of searching is a minute lost.

Each answer is written for whoever is asking: plain words for most people, or, for an account set
to clinician, the terms, doses and evidence grades a clinician wants. Either way it cites only what
was found, says how strong the evidence is, and says when to see someone. It never gives a child's
medicine dose from memory: only what the product's label says, or a pointer to the label, a
pharmacist or the child's doctor.
"""

from __future__ import annotations

import json
import re
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
   full abstract of one that matters. For a medicine, call drug_label for what the FDA label says. If
   MediConsult answered, weigh it as one more source and name it as "MediConsult (your library)".
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
Keep it short: the answer, the evidence, what to do, the sources."""

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
    d.setdefault("mediconsult_url", "")
    d.setdefault("mediconsult_email", "")
    d.setdefault("mediconsult_password", "")
    return d


def view() -> Dict[str, Any]:
    d = _load()
    return {"audience": d["audience"], "mediconsult_url": d["mediconsult_url"], "mediconsult_email": d["mediconsult_email"],
            "has_password": bool(d["mediconsult_password"]), "mediconsult": bool(d["mediconsult_url"])}


def save(change: Dict[str, Any]) -> Dict[str, Any]:
    d = _load()
    if change.get("audience") is not None:
        if change["audience"] not in AUDIENCES:
            raise ValueError("Answers are written for the public or for a clinician.")
        d["audience"] = change["audience"]
    if change.get("mediconsult_url") is not None:
        url = str(change["mediconsult_url"]).strip().rstrip("/")
        if url and not re.match(r"^https?://[^\s/]+(:\d+)?(/.*)?$", url):
            raise ValueError("MediConsult's address looks like http://127.0.0.1:8010 (its backend, not the web page).")
        d["mediconsult_url"] = url
    if change.get("mediconsult_email") is not None:
        d["mediconsult_email"] = str(change["mediconsult_email"]).strip()
    if change.get("mediconsult_password"):
        d["mediconsult_password"] = str(change["mediconsult_password"])
    _path().write_text(json.dumps(d, indent=1))
    _TOKEN.clear()
    return view()


def with_audience(task: str) -> str:
    who = _load()["audience"]
    return task + ("\n\nThe person asking is a clinician: use clinical terms, doses as the sources give them, and grades "
                   "of evidence." if who == "clinician" else
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
# MediConsult, when it is connected
# --------------------------------------------------------------------------

_TOKEN: Dict[str, Any] = {}


def _mc_call(path: str, body: Dict[str, Any], token: Optional[str] = None, timeout: float = 30) -> Dict[str, Any]:
    d = _load()
    headers = {"content-type": "application/json", "User-Agent": USER_AGENT}
    if token:
        headers["authorization"] = "Bearer " + token
    req = urllib.request.Request(d["mediconsult_url"] + path, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as reply:
            return json.loads(reply.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            why = json.loads(exc.read() or b"{}").get("detail")
        except ValueError:
            why = None
        raise MedicalError(f"MediConsult said {exc.code}: {why or exc.reason}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise MedicalError(f"MediConsult is not answering at {d['mediconsult_url']} ({getattr(exc, 'reason', exc)}). "
                           "Is its Docker stack running?") from None


def _mc_token() -> str:
    d = _load()
    key = d["mediconsult_url"] + d["mediconsult_email"]
    if _TOKEN.get("key") == key and time.time() < _TOKEN.get("until", 0):
        return _TOKEN["token"]
    if not (d["mediconsult_email"] and d["mediconsult_password"]):
        raise MedicalError("MediConsult needs the email and password of one of its accounts (Settings → Medical).")
    out = _mc_call("/api/v1/auth/login", {"email": d["mediconsult_email"], "password": d["mediconsult_password"]})
    if not out.get("access_token"):
        raise MedicalError("MediConsult did not sign in.")
    _TOKEN.update(key=key, token=out["access_token"], until=time.time() + max(60, int(out.get("expires_in") or 1800) - 60))
    return out["access_token"]


def mediconsult(question: str, specialty: str = "") -> Dict[str, Any]:
    """Ask your MediConsult AI. Its answer comes from its own library, and says so."""
    if not _load()["mediconsult_url"]:
        raise MedicalError("MediConsult is not connected (Settings → Medical).")
    q = question.strip()
    if len(q) < 10:
        q = q + " (please explain)"
    body = {"query": q[:5000], "include_clinical_trials": True, "include_drug_interactions": True}
    if specialty:
        body["specialty"] = specialty
    out = _mc_call("/api/v1/consult/", body, token=_mc_token(), timeout=180)
    return {"source": "MediConsult (your library)", "answer": _clean(out.get("response"), 6000),
            "citations": [{"title": c.get("title"), "year": c.get("year"), "url": c.get("url"), "pmid": c.get("pmid")}
                          for c in (out.get("citations") or [])][:6], "model": out.get("model")}


def test_mediconsult() -> Dict[str, Any]:
    _TOKEN.clear()
    _mc_token()
    return {"ok": True, "message": "MediConsult signed in."}


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
    return list(dict.fromkeys(n for n in names if n not in stop))[:2]


def gather(question: str) -> Dict[str, Any]:
    """The literature, any medicine's label, and MediConsult's answer, fetched before the agent runs."""
    found: Dict[str, Any] = {"articles": [], "labels": [], "mediconsult": None, "problems": []}
    try:
        found["articles"] = literature(question)
    except MedicalError as exc:
        found["problems"].append(str(exc))
    for name in _drug_names(question):
        try:
            found["labels"].append(drug_label(name))
        except Exception:  # noqa: BLE001 - not every word that looks like a medicine is one
            continue
    if _load()["mediconsult_url"]:
        try:
            found["mediconsult"] = mediconsult(question)
        except MedicalError as exc:
            found["problems"].append(str(exc))
    return found


def articles_text(rows: List[Dict[str, Any]], start: int = 1) -> str:
    out = []
    for i, r in enumerate(rows, start):
        kind = ", ".join(r.get("types") or [])
        out.append(f"[{i}] {r['title']} ({r.get('journal') or r['source']}, {r.get('year') or 'n.d.'})"
                   + (f" [{kind}]" if kind else "") + f" {r['url']}"
                   + ("  · free full text" if r.get("free_full_text") else "")
                   + f"\n    {r.get('abstract') or '(no abstract)'}")
    return "\n".join(out)


def label_text(lab: Dict[str, Any]) -> str:
    head = f"{lab['source']}: {lab['name']}" + (f" ({lab['generic']})" if lab.get("generic") else "") + f" {lab['url']}"
    return head + "".join(f"\n  {k}: {v}" for k, v in lab["parts"].items())


def gathered_text(found: Dict[str, Any]) -> str:
    parts = []
    if found["articles"]:
        parts.append("The literature (cite these by number):\n" + articles_text(found["articles"]))
    for lab in found["labels"]:
        parts.append(label_text(lab))
    mc = found.get("mediconsult")
    if mc:
        parts.append("MediConsult (your library) answered:\n" + mc["answer"]
                     + ("\n  Its sources: " + "; ".join(f"{c['title']} ({c.get('year') or ''}) {c.get('url') or ''}"
                                                        for c in mc["citations"]) if mc["citations"] else ""))
    if found["problems"]:
        parts.append("Could not reach: " + "; ".join(found["problems"]))
    if not (found["articles"] or found["labels"] or mc):
        parts.append("Nothing was found yet: search again with different terms before answering, and if there is still "
                     "nothing, say the evidence could not be reached rather than answering from memory.")
    return "\n\n".join(parts)
