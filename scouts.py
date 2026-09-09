"""Agents that go and look, and then check what they found.

The useful thing about a scout here is not that it can search — anyone can
search — but that this application can *verify*. A repository either imports
into a graph or it does not. A class either has the parameter count it claims or
it does not. A theorem either survives its own Jacobian check or it does not.

So each scout ends every finding with evidence produced by running it:

    code       fetch the repository, scan it, try to import its classes,
               report which imported, at what size, and why the rest refused
    papers     search the literature, rank the passages that carry a
               structural result, report which are fetchable
    maths      for each layer of the current design, the equation with this
               network's own numbers, and where its parameters actually are

A scout that returned opinions would be worth nothing, because an opinion about
a repository is cheaper to produce than to check. A scout that returns "this
class imports as 11 layers with 7,087,872 parameters, matching torch exactly"
has done something a person would otherwise spend an afternoon on.

Nothing here decides. The shortlist is ranked by what was verified, and the
reasons are attached so the ranking can be disagreed with.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import graph as G

USER_AGENT = "deep-network-designer"


@dataclass
class Scout:
    """One autonomous errand, with its findings so far."""

    id: str
    kind: str
    goal: str
    status: str = "running"
    steps: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    stop: threading.Event = field(default_factory=threading.Event)

    def say(self, text: str, detail: str = "") -> None:
        self.steps.append({"at": round(time.time() - self.started, 1),
                           "text": text, "detail": detail})

    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "goal": self.goal,
            "status": self.status, "steps": self.steps[-40:],
            "findings": self.findings, "error": self.error,
            "seconds": round((self.finished or time.time()) - self.started, 1),
        }


SCOUTS: Dict[str, Scout] = {}

CATALOG = [
    {"id": "code", "name": "Find code worth importing",
     "doc": "Searches GitHub for repositories on your topic, downloads each "
            "one, and tries to import the classes it finds. Reports which "
            "actually import, at what size, and the specific reason the rest "
            "refuse. Nothing runs until a class is tried, and every number "
            "comes from running it.",
     "params": [{"name": "repos", "kind": "int", "default": 4},
                {"name": "classes_each", "kind": "int", "default": 4}]},
    {"id": "papers", "name": "Find a structural result",
     "doc": "Searches Europe PMC and arXiv, follows the reference lists of the "
            "reviews it finds, and ranks the passages that state a theorem. "
            "Reports which papers have fetchable text, because one that does "
            "not cannot be read here.",
     "params": [{"name": "hubs", "kind": "int", "default": 6}]},
    {"id": "maths", "name": "Explain this design",
     "doc": "Walks the design on the canvas layer by layer: the equation with "
            "this network's own numbers substituted, where the parameters are, "
            "and what each layer's settings would do. Needs no internet.",
     "params": []},
]


def start(kind: str, goal: str, config: Dict[str, Any],
          graph: Optional[Dict[str, Any]] = None) -> Scout:
    scout = Scout(id=uuid.uuid4().hex[:12], kind=kind, goal=goal)
    SCOUTS[scout.id] = scout
    runner = {"code": _scout_code, "papers": _scout_papers,
              "maths": _scout_maths}.get(kind)
    if runner is None:
        scout.status = "error"
        scout.error = f"No scout called {kind!r}."
        return scout

    def work():
        try:
            runner(scout, config or {}, graph or {})
            scout.status = "stopped" if scout.stop.is_set() else "done"
        except Exception as exc:  # noqa: BLE001 - a scout must not take the app with it
            scout.status = "error"
            scout.error = f"{type(exc).__name__}: {exc}"
        finally:
            scout.finished = time.time()

    threading.Thread(target=work, daemon=True).start()
    return scout


# --------------------------------------------------------------------------
# code
# --------------------------------------------------------------------------

def _search_github(query: str, count: int) -> List[Dict[str, Any]]:
    address = ("https://api.github.com/search/repositories?q="
               + urllib.parse.quote(query + " language:python")
               + f"&sort=stars&per_page={max(1, min(20, count))}")
    request = urllib.request.Request(
        address, headers={"User-Agent": USER_AGENT,
                          "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=40) as reply:
        found = json.loads(reply.read())
    return [{"full_name": item["full_name"],
             "stars": item.get("stargazers_count", 0),
             "description": (item.get("description") or "")[:200]}
            for item in found.get("items", [])]


def _scout_code(scout: Scout, config: Dict[str, Any],
                graph: Dict[str, Any]) -> None:
    import importer

    want_repos = max(1, min(8, int(config.get("repos", 4))))
    per_repo = max(1, min(8, int(config.get("classes_each", 4))))

    scout.say(f"Searching GitHub for “{scout.goal}”")
    try:
        repos = _search_github(scout.goal, want_repos)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"GitHub search is unreachable: {exc}. This scout is the only part "
            f"of the app that needs the internet.")
    if not repos:
        scout.say("Nothing matched.", "Try naming the architecture rather than "
                                      "the task.")
        return
    scout.say(f"{len(repos)} repositories to look at",
              ", ".join(r["full_name"] for r in repos))

    shape = _input_shape(graph) or [3, 224, 224]
    for repo in repos:
        if scout.stop.is_set():
            return
        name = repo["full_name"]
        scout.say(f"Fetching {name}")
        try:
            info = importer.fetch_repo(name)
            scan = importer.scan_folder(info["root"])
        except Exception as exc:  # noqa: BLE001
            scout.say(f"{name} could not be read", str(exc)[:160])
            continue

        models = scan.get("models") or []
        scout.say(f"{name}: {scan['files']} files, {len(models)} nn.Module classes")
        if not models:
            continue

        # the ones that take no arguments first: they can be tried without
        # guessing at a config, and a guess would make the evidence worthless
        models.sort(key=lambda m: (m.get("arguments", 0), m["cls"]))
        tried = []
        for model in models[:per_repo]:
            if scout.stop.is_set():
                return
            verdict = _try_class(info["root"], model, shape)
            tried.append(verdict)
            scout.say(f"   {model['cls']}: {verdict['summary']}")

        # An import whose parameter count disagrees with torch is not a
        # success: the graph is not the model, and every number downstream of
        # it — size, precision, ablation — would be about something else.
        worked = [t for t in tried if t["imported"] and t.get("exact", True)]
        inexact = [t for t in tried if t["imported"] and not t.get("exact", True)]
        scout.findings.append({
            "kind": "repo",
            "name": name,
            "stars": repo["stars"],
            "description": repo["description"],
            "root": info["root"],
            "files": scan["files"],
            "classes": len(models),
            "tried": tried,
            "imported": len(worked),
            "inexact": len(inexact),
            # ranked on what was verified, not on stars: a popular repository
            # whose classes cannot be traced is of no use here, and one that
            # traces to the wrong size is worse than one that refuses
            "score": len(worked) * 10 - len(inexact) * 2
                     + min(repo["stars"], 20000) / 20000,
        })

    scout.findings.sort(key=lambda f: -f["score"])
    best = [f for f in scout.findings if f["imported"]]
    if best:
        scout.say(f"{len(best)} of {len(scout.findings)} repositories offered a "
                  f"class that imports", best[0]["name"])
    else:
        scout.say("Nothing imported cleanly.",
                  "Every candidate refused, and each reason is on its card. "
                  "This is common: most published code branches on tensor "
                  "values somewhere.")


def _try_class(root: str, model: Dict[str, Any],
               shape: List[int]) -> Dict[str, Any]:
    """Import one class for real and report what happened."""
    import importer
    import layers  # noqa: F401 - registry must be loaded

    entry = {"cls": model["cls"], "file": model["file"],
             "arguments": model.get("arguments", 0), "imported": False}
    if model.get("arguments", 0) > 0:
        entry["summary"] = (f"needs {model['arguments']} constructor argument"
                            f"{'s' if model['arguments'] > 1 else ''}, so it "
                            f"was not guessed at")
        entry["why"] = ("A class taking a config can only be built once "
                        "somebody supplies one. Guessing would make any result "
                        "meaningless.")
        return entry
    try:
        built = importer.from_folder(root, model["file"], model["cls"], shape, "")
    except Exception as exc:  # noqa: BLE001
        entry["summary"] = "refused"
        entry["why"] = str(exc)[:300]
        return entry

    notes = built.pop("_notes", [])
    expected = built.pop("_expected_parameters", 0)
    built.pop("_entry", None)
    built.pop("_routing", None)
    nodes = built.get("nodes", [])
    opaque = sum(1 for n in nodes if n["type"] == "Custom")
    report = G.analyze(G.parse(built))
    counted = report.get("total_learnables", 0)

    entry.update({
        "imported": True,
        "layers": len(nodes),
        "opaque": opaque,
        "parameters": counted,
        "expected": expected,
        "exact": bool(expected) and counted == expected,
        "shapes_ok": bool(report.get("ok")),
        "graph": built,
        "notes": notes[:4],
    })
    entry["summary"] = (
        f"{len(nodes)} layers, {counted:,} parameters"
        + (", matching torch exactly" if entry["exact"] else
           f" but torch says {expected:,} — the graph is not the model"
           if expected else "")
        + (f", {opaque} opaque" if opaque else ""))
    if expected and not entry["exact"]:
        entry["why"] = (
            f"The traced graph has {counted:,} parameters and the class has "
            f"{expected:,}. Something was dropped or duplicated in tracing, so "
            f"this graph is not a faithful copy — size, precision and ablation "
            f"numbers taken from it would be about a different network.")
    return entry


def _input_shape(graph: Dict[str, Any]) -> Optional[List[int]]:
    for node in (graph or {}).get("nodes", []):
        if node.get("type") == "Input":
            shape = (node.get("params") or {}).get("shape")
            if shape:
                return [int(d) for d in shape]
    return None


# --------------------------------------------------------------------------
# papers
# --------------------------------------------------------------------------

def _scout_papers(scout: Scout, config: Dict[str, Any],
                  graph: Dict[str, Any]) -> None:
    from dnn_bench import paper_to_spec as p2s

    scout.say(f"Searching the literature for “{scout.goal}”")
    found = p2s.discover(scout.goal, n_hubs=max(1, min(12, int(config.get("hubs", 6)))),
                         use_arxiv=True,
                         log=lambda *a: scout.say(" ".join(str(x) for x in a)[:200]))

    groups = [("foundational", "cited by the reviews"),
              ("direct", "matched the question"),
              ("preprints", "on arXiv"),
              ("hubs", "a review")]
    for key, why in groups:
        for hit in (found.get(key) or [])[:6]:
            if scout.stop.is_set():
                return
            text_url = p2s.full_text(hit)
            scout.findings.append({
                "kind": "paper",
                "title": hit.title,
                "authors": hit.authors,
                "year": hit.year,
                "journal": hit.journal,
                "cited_by": hit.cited_by,
                "recurrence": hit.recurrence,
                "reasons": list(hit.reasons or []),
                "why_group": why,
                "doi": hit.doi,
                "text_url": text_url,
                # a paper that cannot be read here cannot become a layer here,
                # however good it is
                "score": (3 if text_url else 0) + len(hit.reasons or [])
                         + min(hit.recurrence or 0, 5),
            })

    if not scout.findings:
        scout.say("Nothing on topic.",
                  "The gate is deliberate: a paper that scores well but is not "
                  "about the question is not an answer to it.")
        return

    scout.findings.sort(key=lambda f: -f["score"])
    readable = sum(1 for f in scout.findings if f["text_url"])
    scout.say(f"{len(scout.findings)} papers, {readable} with fetchable text",
              scout.findings[0]["title"][:120])


# --------------------------------------------------------------------------
# maths
# --------------------------------------------------------------------------

def _scout_maths(scout: Scout, config: Dict[str, Any],
                 graph: Dict[str, Any]) -> None:
    import mathbook

    if not (graph or {}).get("nodes"):
        scout.say("Nothing on the canvas to explain.")
        return

    g = G.parse(graph)
    report = G.analyze(g)
    nodes = g.by_id()
    total = report.get("total_learnables", 0)
    scout.say(f"Reading {len(report.get('order', []))} layers",
              f"{total:,} parameters in total")

    incoming: Dict[str, List[str]] = {nid: [] for nid in nodes}
    for edge in g.edges:
        incoming.setdefault(edge.target, []).append(edge.source)

    for nid in report.get("order", []):
        if scout.stop.is_set():
            return
        node = nodes[nid]
        info = report["nodes"].get(nid, {})
        in_shapes = [report["nodes"].get(s, {}).get("out_shape")
                     for s in incoming.get(nid, [])]
        entry = mathbook.explain(node.type, G.resolved_params(node),
                                 in_shapes, info.get("out_shape"))
        count = info.get("learnables", 0)
        scout.findings.append({
            "kind": "layer",
            "id": nid,
            "name": node.label or node.type,
            "type": node.type,
            "title": entry.get("title") or node.type,
            "equation": entry.get("equation", ""),
            "arithmetic": entry.get("arithmetic", []),
            "freedom": entry.get("freedom", [])[:3],
            "missing": entry.get("missing"),
            "parameters": count,
            "share": round(count / total * 100, 1) if total else 0,
            "out_shape": info.get("out_shape"),
        })
        scout.say(f"   {node.label or node.type}: {entry.get('title') or node.type}")

    explained = sum(1 for f in scout.findings if not f["missing"])
    scout.say(f"{explained} of {len(scout.findings)} layers have a write-up",
              "The rest say so rather than inventing one.")
