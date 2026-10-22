"""Radar: what is new in AI, and what this lab could build from it.

Once a day (or when you pull to refresh) Radar reads the places where new work appears first:

    papers     arXiv, newest first, in five fields: AI itself, maths, the physical sciences,
               biology and neuroscience (in each of the last four, only work that uses or
               studies learning machines)
    preprints  Europe PMC's preprint servers (bioRxiv, medRxiv and others) for AI in biology
               and neuroscience
    code       GitHub repositories made this week that are already drawing stars
    news       Hacker News stories about AI that people voted up, and a web search

It drops what it has shown you before, then has the model pick the few findings worth your time.
Each pick says what is new, why it matters here, and what the lab could do with it: build and
train a model that tests the idea, check a mathematical claim with code, or explain it visually.

The doing is not Radar's. "Formulate" hands the finding to Muse, which plans the work across the
agents you already have (a researcher to read the paper, the model builder to design and train a
small test against a plain baseline, the maths agent to check a claim) and reports back in its own
conversation. The daily check formulates the top pick or two by itself (you choose how many,
zero included) and writes to your inbox.

Without a model the findings are still listed, newest first, with no picks and nothing formulated.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentlab as AL

USER_AGENT = "deep-network-designer-radar (personal research digest)"
PER_SOURCE = 12             # results asked of each source, per field
KEEP = 10                   # findings kept from one build
SEEN_DAYS = 45              # a finding shown once stays away this long
ARXIV_PAUSE = 3.0           # arXiv asks for a few seconds between calls
ACTS = ("model", "maths", "explain", "none")
_LOCK = threading.Lock()
_BUILDING: Dict[str, bool] = {}

FIELDS: Dict[str, Dict[str, Any]] = {
    "ai": {"label": "AI", "arxiv": "cat:cs.LG OR cat:cs.AI OR cat:cs.CL OR cat:cs.NE"},
    "maths": {"label": "Maths", "arxiv": '(cat:math.NA OR cat:math.OC OR cat:math.DS OR cat:math.PR OR cat:math.LO OR '
                                         'cat:math.CO OR cat:math.ST) AND (abs:"neural network" OR abs:"machine learning" '
                                         'OR abs:"deep learning" OR abs:"language model")'},
    "science": {"label": "Science", "arxiv": '(cat:physics.comp-ph OR cat:physics.flu-dyn OR cat:cond-mat.dis-nn OR '
                                             'cat:physics.chem-ph OR cat:astro-ph.IM) AND (abs:neural OR abs:"machine learning")'},
    "biology": {"label": "Biology", "arxiv": '(cat:q-bio.BM OR cat:q-bio.QM OR cat:q-bio.GN OR cat:q-bio.MN) AND '
                                             '(abs:"deep learning" OR abs:neural OR abs:"language model" OR abs:"machine learning")',
                "epmc": '("deep learning" OR "machine learning" OR "language model") AND (protein OR genome OR cell OR '
                        'molecule OR drug)'},
    "neuroscience": {"label": "Neuroscience", "arxiv": "cat:q-bio.NC",
                     "epmc": '("deep learning" OR "neural network" OR "machine learning") AND (neuron* OR brain OR '
                             'cortex OR neural coding OR connectome)'},
}
NEWS_QUERIES = ["AI research breakthrough this week", "new AI model released"]

PICK = """You keep a research lab's radar. The lab designs and trains neural networks on a canvas, and has
these agents: a model builder (designs, trains and packages networks on built-in mnist, fashion_mnist, cifar10,
synthetic data or a CSV), a maths agent (finds and proves results with code), an explainer (interactive pages)
and researchers who read papers. Its layers include physics-informed networks, Hamiltonian and equivariant
layers, implicit (equilibrium) layers, Hopfield memories, predictive coding, tensor-train layers and the usual
convolution, attention and recurrent ones. It runs on one Mac, so a test must train in minutes.

From the findings below pick up to {keep} that are genuinely new and that matter most, spread across fields
when the quality allows. Skip adverts, surveys with nothing new, duplicates and hype without substance.
For each pick, say what the lab could do with it:
  "model"    an idea a small network here could test: a layer, an architecture, a training method or a
             loss, tried against a plain baseline on data the lab has
  "maths"    a mathematical claim or construction worth checking with code
  "explain"  worth understanding well, as an interactive page, but nothing to build
  "none"     worth knowing, nothing to do
and, unless it is "none", the task: one self-contained instruction for the lab's agents, naming the dataset,
the baseline and what to measure.

Reply with strict JSON only:
{{"picks": [{{"n": <finding number>, "field": "<ai|maths|science|biology|neuroscience>",
  "headline": "<short, plain headline>", "what": "<two plain sentences: what is new>",
  "why": "<one sentence: why it matters to this lab>", "act": "<model|maths|explain|none>",
  "task": "<the instruction, or empty>"}}]}}
Use only what the findings say; do not invent results."""


# --------------------------------------------------------------------------
# where findings come from (each returns [{title, url, summary, source, field, date}])
# --------------------------------------------------------------------------

def _get(url: str, timeout: float = 40) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as reply:
        return reply.read()


def _clean(text: str, limit: int = 420) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def arxiv(field: str, count: int = PER_SOURCE) -> List[Dict[str, Any]]:
    query = FIELDS[field]["arxiv"]
    url = ("https://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(query)
           + f"&sortBy=submittedDate&sortOrder=descending&max_results={count}")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(_get(url))
    out = []
    for e in root.findall("a:entry", ns):
        link = (e.findtext("a:id", "", ns) or "").strip()
        out.append({"title": _clean(e.findtext("a:title", "", ns), 200), "url": link.replace("http://", "https://"),
                    "summary": _clean(e.findtext("a:summary", "", ns)), "source": "arXiv", "field": field,
                    "date": (e.findtext("a:published", "", ns) or "")[:10]})
    return out


def europepmc(field: str, count: int = PER_SOURCE) -> List[Dict[str, Any]]:
    query = FIELDS[field].get("epmc")
    if not query:
        return []
    today = _dt.date.today()
    since = (today - _dt.timedelta(days=10)).isoformat()
    q = f"({query}) AND SRC:PPR AND FIRST_PDATE:[{since} TO {today.isoformat()}]"
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=core&pageSize="
           + str(count) + "&query=" + urllib.parse.quote(q))
    found = json.loads(_get(url)).get("resultList", {}).get("result", [])
    out = []
    for r in found:
        doi = r.get("doi")
        link = f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/{r.get('source', 'PPR')}/{r.get('id')}"
        out.append({"title": _clean(r.get("title"), 200), "url": link, "summary": _clean(r.get("abstractText")),
                    "source": (r.get("bookOrReportDetails") or {}).get("publisher") or "preprint", "field": field,
                    "date": r.get("firstPublicationDate") or ""})
    return out


def github(count: int = PER_SOURCE) -> List[Dict[str, Any]]:
    since = (_dt.date.today() - _dt.timedelta(days=7)).isoformat()
    out, seen = [], set()
    for topic in ("machine-learning", "deep-learning", "llm"):
        q = f"topic:{topic} created:>{since} stars:>25"
        url = ("https://api.github.com/search/repositories?q=" + urllib.parse.quote(q)
               + f"&sort=stars&order=desc&per_page={max(1, min(20, count))}")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=40) as reply:
            items = json.loads(reply.read()).get("items", [])
        for it in items:
            if it["full_name"] in seen:
                continue
            seen.add(it["full_name"])
            out.append({"title": it["full_name"], "url": it["html_url"],
                        "summary": _clean(it.get("description") or "") + f" ({it.get('stargazers_count', 0)} stars)",
                        "source": "GitHub", "field": "ai", "date": (it.get("created_at") or "")[:10]})
    return out[:count * 2]


def hackernews(count: int = PER_SOURCE) -> List[Dict[str, Any]]:
    since = int(time.time() - 3 * 86400)
    url = ("https://hn.algolia.com/api/v1/search?tags=story&query=" + urllib.parse.quote("AI")
           + f"&numericFilters=created_at_i>{since},points>80&hitsPerPage={count}")
    hits = json.loads(_get(url)).get("hits", [])
    return [{"title": _clean(h.get("title"), 200),
             "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
             "summary": f"{h.get('points', 0)} points on Hacker News", "source": "Hacker News", "field": "ai",
             "date": (h.get("created_at") or "")[:10]} for h in hits if h.get("title")]


def web_news(count: int = 6) -> List[Dict[str, Any]]:
    ns: Dict[str, Any] = {"json": json, "re": re, "urllib": urllib, "os": __import__("os")}
    exec(AL.WEB_CODE, ns)
    out = []
    for q in NEWS_QUERIES:
        for title, url, snippet in ns["web_results"](q, count):
            if url.startswith("http"):
                out.append({"title": _clean(title, 200), "url": url, "summary": _clean(snippet), "field": "ai",
                            "source": re.sub(r"^www\.", "", (re.match(r"https?://([^/]+)", url) or [None, ""])[1]),
                            "date": ""})
    return out


def _sources(fields: List[str]) -> List[tuple]:
    """(name, call) for everything one build reads; tests swap these for canned results."""
    calls: List[tuple] = []
    for f in fields:
        calls.append((f"arXiv {FIELDS[f]['label']}", (lambda f=f: arxiv(f)), True))
        if FIELDS[f].get("epmc"):
            calls.append((f"preprints {FIELDS[f]['label']}", (lambda f=f: europepmc(f)), False))
    calls += [("GitHub", github, False), ("Hacker News", hackernews, False), ("news", web_news, False)]
    return calls


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

def _path() -> Path:
    return AL._dir() / "radar.json"


def _load() -> Dict[str, Any]:
    try:
        d = json.loads(_path().read_text())
    except (OSError, ValueError):
        d = {}
    d.setdefault("fields", list(FIELDS))
    d.setdefault("auto", 1)
    d.setdefault("items", [])
    d.setdefault("seen", {})
    d.setdefault("formulated", [])
    d.setdefault("built", None)
    d.setdefault("error", None)
    return d


def _store(d: Dict[str, Any]) -> None:
    _path().write_text(json.dumps(d, indent=1))


def _id(url: str) -> str:
    return "rd" + hashlib.sha1(url.encode()).hexdigest()[:10]


def view() -> Dict[str, Any]:
    d = _load()
    done = {(f.get("item") or {}).get("id") for f in d["formulated"]}
    items = [{**i, "brief": task_for(i) if i.get("act") in WHAT_TO_DO else "", "done": i["id"] in done} for i in d["items"]]
    return {"fields": d["fields"], "auto": d["auto"], "items": items, "built": d["built"], "error": d["error"],
            "formulated": d["formulated"][-30:], "all_fields": {k: v["label"] for k, v in FIELDS.items()},
            "building": bool(_BUILDING.get(str(AL._dir())))}


def settings(fields: Optional[List[str]] = None, auto: Optional[int] = None) -> Dict[str, Any]:
    with _LOCK:
        d = _load()
        if fields is not None:
            clean = [f for f in dict.fromkeys(fields) if f in FIELDS]
            if not clean:
                raise ValueError("Follow at least one field: " + ", ".join(FIELDS) + ".")
            d["fields"] = clean
        if auto is not None:
            if not 0 <= int(auto) <= 3:
                raise ValueError("Formulate between 0 and 3 findings a day.")
            d["auto"] = int(auto)
        _store(d)
    return view()


# --------------------------------------------------------------------------
# building the radar
# --------------------------------------------------------------------------

def _pick(found: List[Dict[str, Any]], live: bool) -> List[Dict[str, Any]]:
    plain = [{**f, "headline": f["title"], "what": f["summary"], "why": "", "act": "none", "task": ""} for f in found]
    if not live or not found:
        return plain[:KEEP]
    import evolve
    listing = "\n".join(f"{i + 1}. [{f['field']}, {f['source']}{', ' + f['date'] if f['date'] else ''}] {f['title']}\n"
                        f"   {f['summary'][:320]}" for i, f in enumerate(found))
    try:
        out = evolve.ask(PICK.format(keep=KEEP), "Findings:\n" + listing, 3000)
        picks = (evolve._json_in(out["text"]) or {}).get("picks") or []
    except Exception:  # noqa: BLE001 - the plain list is still worth showing
        return plain[:KEEP]
    chosen, used = [], set()
    for p in picks:
        try:
            n = int(p["n"]) - 1
            f = found[n]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        if n in used:
            continue
        used.add(n)
        act = str(p.get("act") or "none").lower()
        act = act if act in ACTS else "none"
        task = str(p.get("task") or "").strip()[:1500]
        if act != "none" and len(task) < 20:
            act, task = "none", ""                      # an action with no instruction is no action
        chosen.append({**f, "field": p.get("field") if p.get("field") in FIELDS else f["field"],
                       "headline": _clean(p.get("headline") or f["title"], 160), "what": _clean(p.get("what") or f["summary"], 500),
                       "why": _clean(p.get("why") or "", 300), "act": act, "task": task})
        if len(chosen) >= KEEP:
            break
    return chosen or plain[:KEEP]


def build(live: Optional[bool] = None) -> Dict[str, Any]:
    """Read every source, drop what was shown before, keep the best few. Returns the radar."""
    live = AL.live_available() if live is None else live
    key = str(AL._dir())
    _BUILDING[key] = True
    try:
        d = _load()
        now = time.time()
        seen = {u: t for u, t in d["seen"].items() if now - t < SEEN_DAYS * 86400}
        found, errors, first_arxiv = [], [], True
        for name, call, is_arxiv in _sources(d["fields"]):
            if is_arxiv and not first_arxiv:
                time.sleep(ARXIV_PAUSE)
            first_arxiv = first_arxiv and not is_arxiv
            try:
                found += call()
            except Exception as exc:  # noqa: BLE001 - one source failing must not empty the radar
                errors.append(f"{name}: {type(exc).__name__}")
        fresh, urls = [], set()
        for f in found:
            u = f.get("url") or ""
            if u.startswith("http") and u not in seen and u not in urls and f.get("title"):
                urls.add(u)
                fresh.append(f)
        items = [{**it, "id": _id(it["url"])} for it in _pick(fresh, live)]
        with _LOCK:
            d = _load()
            for it in items:
                seen[it["url"]] = now
            d["seen"] = seen
            if items or not d["items"]:
                d["items"] = items
                d["built"] = now
            d["error"] = ("Could not read: " + "; ".join(errors)) if errors and not items else \
                (("Some sources did not answer: " + "; ".join(errors)) if errors else None)
            _store(d)
        return view()
    finally:
        _BUILDING[key] = False


def build_soon() -> Dict[str, Any]:
    key = str(AL._dir())
    if not _BUILDING.get(key):
        _BUILDING[key] = True
        import contextvars
        threading.Thread(target=contextvars.copy_context().run, args=(build,), daemon=True).start()
    return view()


# --------------------------------------------------------------------------
# formulating: the finding becomes work for the agents you have
# --------------------------------------------------------------------------

WHAT_TO_DO = {
    "model": ("Design a small model on the network canvas that tests this idea, and a plain baseline beside it. Train "
              "both briefly on the same data, so it finishes in minutes on this Mac, and compare them on the same "
              "measure. Say plainly whether the idea helped, by how much, and what you would try next. If the idea "
              "needs a layer the canvas does not have, build the closest honest version and say what is missing."),
    "maths": ("Check the claim with code: state it exactly, test it on cases you can compute, and say whether it held, "
              "with the evidence. If it is a construction, build it and verify it."),
    "explain": "Explain it as an interactive page: the key equation, a worked example, and a plot or graph to play with.",
}


def task_for(item: Dict[str, Any]) -> str:
    act = item.get("act") if item.get("act") in WHAT_TO_DO else "explain"
    return (f"A new finding from the radar ({FIELDS.get(item.get('field'), {}).get('label', 'AI')}, {item.get('source')}): "
            f"{item.get('headline')}. {item.get('what')}\nSource: {item.get('url')}\n\n"
            + (f"What to do: {item['task']}\n\n" if item.get("task") else "") + WHAT_TO_DO[act])


def find(item_id: str) -> Dict[str, Any]:
    d = _load()
    for it in d["items"] + [f.get("item") or {} for f in d["formulated"]]:
        if it.get("id") == item_id:
            return it
    raise KeyError(item_id)


def formulate(item_id: str, mode: Optional[str] = None) -> Dict[str, Any]:
    """Hand a finding to Muse in a conversation of its own. Returns {conv, item}."""
    item = find(item_id)
    with _LOCK:
        d = _load()
        for f in d["formulated"]:
            if (f.get("item") or {}).get("id") == item_id:
                return {"conv": f["conv"], "item": item, "again": True}
    mode = mode or ("live" if AL.live_available() else "rehearsal")
    conv = AL.new_conversation({"kind": "muse", "key": "muse"}, mode)
    conv_id = conv["id"]
    c = AL.conversation(conv_id)
    c["title"] = ("Radar: " + item.get("headline", ""))[:90]
    AL._conv_save(c)
    AL.say(conv_id, task_for(item), mode)
    with _LOCK:
        d = _load()
        d["formulated"].append({"item": item, "conv": conv_id, "at": time.time()})
        d["formulated"] = d["formulated"][-100:]
        _store(d)
    import ideas
    ideas.forget_cache()
    return {"conv": conv_id, "item": item, "again": False}


def mark(item_id: str, conv: Optional[str] = None) -> Dict[str, Any]:
    """Record that a finding was handed on elsewhere (Saathi sends it into the chat you are in)."""
    item = find(item_id)
    with _LOCK:
        d = _load()
        if not any((f.get("item") or {}).get("id") == item_id for f in d["formulated"]):
            d["formulated"].append({"item": item, "conv": conv, "at": time.time()})
            d["formulated"] = d["formulated"][-100:]
            _store(d)
    import ideas
    ideas.forget_cache()
    return view()


def _actionable(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    done = {(f.get("item") or {}).get("id") for f in d["formulated"]}
    order = {"model": 0, "maths": 1, "explain": 2}
    return sorted([i for i in d["items"] if i.get("act") in order and i["id"] not in done],
                  key=lambda i: order[i["act"]])


def check(s: Dict[str, Any]) -> None:
    """The scheduled build ("radar" check): read, pick, formulate the top few, and say so in the inbox."""
    r = build()
    d = _load()
    started = []
    if AL.live_available():
        for item in _actionable(d)[:d["auto"]]:
            try:
                started.append((item, formulate(item["id"])["conv"]))
            except Exception:  # noqa: BLE001 - a finding that cannot start is still listed
                continue
    s["last_status"] = "ok" if r["items"] else "error"
    s["last_detail"] = (f"{len(r['items'])} findings, {len(started)} formulated" if r["items"]
                        else (r.get("error") or "nothing new"))
    if r["items"]:
        import schedules
        heads = "; ".join(i["headline"] for i in r["items"][:4])
        more = (" Formulating: " + "; ".join(i["headline"] for i, _ in started) + ".") if started else ""
        schedules.notify("done", f"Radar: {len(r['items'])} new findings", heads + "." + more,
                         conv=started[0][1] if started else None)
