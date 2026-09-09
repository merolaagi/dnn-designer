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
    #: set once the errand is finished *and* written down
    settled: threading.Event = field(default_factory=threading.Event)

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

#: Finished scouts are written here. An errand that took forty seconds and
#: reached the internet should survive leaving the page — and two searches for
#: the same thing return slightly different repositories, which is worth being
#: able to compare rather than only regret.
HISTORY = Path(__file__).resolve().parent / "scouts"


def _keep(scout: "Scout") -> None:
    try:
        HISTORY.mkdir(parents=True, exist_ok=True)
        blob = scout.snapshot()
        blob["kept_at"] = time.time()
        (HISTORY / f"{scout.id}.json").write_text(json.dumps(blob, indent=1))
    except Exception:  # noqa: BLE001 - never let bookkeeping end a scout badly
        pass


def history(limit: int = 40) -> List[Dict[str, Any]]:
    """Past errands, newest first, without their findings."""
    out = []
    if not HISTORY.exists():
        return out
    for path in sorted(HISTORY.glob("*.json"),
                       key=lambda p: -p.stat().st_mtime)[:limit]:
        try:
            blob = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        out.append({"id": blob.get("id", path.stem), "kind": blob.get("kind", ""),
                    "goal": blob.get("goal", ""), "status": blob.get("status", ""),
                    "seconds": blob.get("seconds", 0),
                    "findings": len(blob.get("findings") or []),
                    "kept_at": blob.get("kept_at", path.stat().st_mtime)})
    return out


def recall(scout_id: str) -> Optional[Dict[str, Any]]:
    """A past errand in full, findings and all."""
    live = SCOUTS.get(scout_id)
    if live:
        return live.snapshot()
    path = HISTORY / f"{scout_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def forget(scout_id: str) -> bool:
    path = HISTORY / f"{scout_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False

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
    {"id": "draft", "name": "Draft a design for me",
     "doc": "Finds the guided build closest to what you describe, assembles "
            "the whole thing, fits it to your input shape and number of "
            "classes, and checks it before putting it on the canvas: shapes "
            "resolve, the parameter count matches PyTorch, and a batch goes "
            "through. It assembles from patterns that exist rather than "
            "inventing an architecture, which is why the result can be "
            "checked at all.",
     "params": [{"name": "shape", "kind": "text", "default": ""},
                {"name": "classes", "kind": "int", "default": 0}]},
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
              "draft": _scout_draft, "maths": _scout_maths}.get(kind)
    if runner is None:
        scout.status = "error"
        scout.error = f"No scout called {kind!r}."
        return scout

    def work():
        # The status is set last, after the errand is on disk. A client that
        # sees "done" and immediately asks for the history has to find it
        # there — flipping the status first leaves a window where a finished
        # scout does not exist yet.
        outcome, message = "done", ""
        try:
            runner(scout, config or {}, graph or {})
            outcome = "stopped" if scout.stop.is_set() else "done"
        except Exception as exc:  # noqa: BLE001 - a scout must not take the app with it
            outcome, message = "error", f"{type(exc).__name__}: {exc}"
        scout.finished = time.time()
        scout.error = message
        scout.status = outcome
        _keep(scout)
        scout.settled.set()

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

        # Test files, examples and benchmarks first get set aside: a class
        # defined in tests/test_linear4bit.py is a stub for exercising
        # something else, and importing it teaches nothing about the
        # repository. They are still counted, so the totals stay honest.
        models = [m for m in models if not _is_scaffolding(m["file"])] or models

        # then the ones taking no arguments, which can be tried without
        # guessing at a config — and a guess would make the evidence worthless
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
        return

    # A list of identical refusals is a conclusion nobody has drawn yet. Say
    # which kind of refusal dominated, because the three kinds have different
    # answers and only one of them is a dead end.
    every = [t for f in scout.findings for t in f["tried"]]
    wanting = [t for t in every if t.get("arguments", 0) > 0]
    missing = [t for t in every if "did not import" in (t.get("why") or "")]
    untraceable = [t for t in every if "cannot be traced" in (t.get("why") or "")]

    if wanting and len(wanting) >= len(every) / 2:
        names = ", ".join(sorted({t["cls"] for t in wanting})[:4])
        scout.say(f"{len(wanting)} of {len(every)} classes want a constructor "
                  f"argument.",
                  f"That is the shape of most language model code: one config "
                  f"object built once and passed everywhere ({names}). None of "
                  f"it is out of reach — Import can take a Setup block that "
                  f"builds the config, and then these classes import like any "
                  f"other. A scout cannot write that block, because guessing "
                  f"the values would make every number it reported meaningless.")
    elif missing and len(missing) >= len(every) / 2:
        scout.say(f"{len(missing)} of {len(every)} classes need packages that "
                  f"are not installed here.",
                  "Their repositories import something at module level that "
                  "this machine does not have. Installing it into the same "
                  "environment as the app would let them through.")
    elif untraceable:
        scout.say(f"{len(untraceable)} of {len(every)} could not be traced.",
                  "Their forward() branches on tensor values, which no single "
                  "static graph can represent. That one is a real dead end for "
                  "importing, though the source is still readable.")
    else:
        scout.say("Nothing imported cleanly.",
                  "Each reason is on its card.")


#: What a parameter's name suggests it wants. A guess, but a guess that is
#: then built, counted against torch and run — and reported, so it can be
#: corrected. My first version refused to guess at all, on the grounds that a
#: guess makes the evidence meaningless. That is true of an *unverified* guess
#: and wrong about a verified one: "nin, nout" is plainly two channel counts,
#: and if the class builds at exactly torch's parameter count and a batch goes
#: through, the hypothesis has passed. Refusing left whole repositories
#: unreachable for want of the number 3.
GUESSES: List[tuple] = [
    (("in_channel", "in_channels", "in_ch", "nin", "input_channels", "in_dim",
      "input_dim", "in_features", "channels", "c_in"), "channels"),
    (("out_channel", "out_channels", "out_ch", "nout", "output_channels",
      "out_dim", "output_dim", "out_features", "c_out", "cnn_channels",
      "hidden", "hidden_dim", "dim", "width", "planes"), 32),
    (("num_classes", "n_classes", "nb_classes", "classes", "num_class"), 10),
    (("kernel_size", "kernel", "k", "ksize"), 3),
    (("stride", "s"), 1),
    (("padding", "pad", "p"), 1),
    (("dilation", "d"), 1),
    (("dropout", "drop", "p_dropout", "cnn_dropout"), 0.25),
    (("num_layers", "n_layers", "depth", "blocks", "num_blocks"), 2),
    (("heads", "num_heads", "n_heads", "nhead"), 4),
    (("bias",), True),
]


def _guess_arguments(wants: List[str], shape: List[int]) -> Optional[List[Any]]:
    """Propose a value for every required argument, or give up.

    Giving up matters: a name nothing recognises means the class is asking for
    something structural — a config, a list of layer widths — and inventing
    that would be the meaningless kind of guess.
    """
    channels = int(shape[0]) if shape else 3
    values: List[Any] = []
    for name in wants:
        lowered = name.lower().strip("_")
        picked = None
        for names, value in GUESSES:
            if lowered in names:
                picked = channels if value == "channels" else value
                break
        if picked is None:
            return None
        values.append(picked)
    return values


SCAFFOLDING = ("test", "tests", "testing", "benchmark", "benchmarks",
               "example", "examples", "conftest", "setup")


def _is_scaffolding(path: str) -> bool:
    """Files that hold classes nobody wants to import."""
    parts = [p.lower() for p in Path(path).parts]
    stem = Path(path).stem.lower()
    return (any(p in SCAFFOLDING for p in parts)
            or stem.startswith("test_") or stem.endswith("_test"))


def _try_class(root: str, model: Dict[str, Any],
               shape: List[int]) -> Dict[str, Any]:
    """Import one class for real and report what happened."""
    import importer
    import layers  # noqa: F401 - registry must be loaded

    entry = {"cls": model["cls"], "file": model["file"],
             "arguments": model.get("arguments", 0), "imported": False,
             "wants": list(model.get("wants") or [])}

    guessed: Optional[List[Any]] = None
    if model.get("arguments", 0) > 0:
        guessed = _guess_arguments(entry["wants"], shape)
        if guessed is None:
            named = ", ".join(entry["wants"]) or f"{model['arguments']} arguments"
            entry["summary"] = f"wants {named}, which cannot be worked out"
            entry["why"] = (
                f"Those are structural — a config object, or a list of widths. "
                f"A number can be inferred from the input shape; a design "
                f"decision cannot, and inventing one would make every figure "
                f"reported about it meaningless. Import can take a Setup block "
                f"that builds it.")
            return entry
        entry["guessed"] = dict(zip(entry["wants"], guessed))
    args = ", ".join(repr(v) for v in guessed) if guessed else ""

    try:
        built = importer.from_folder(root, model["file"], model["cls"], shape, args)
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
    tried_with = (", built with "
                  + ", ".join(f"{k}={v!r}" for k, v in entry["guessed"].items())
                  if entry.get("guessed") else "")
    entry["summary"] = (
        f"{len(nodes)} layers, {counted:,} parameters"
        + (", matching torch exactly" if entry["exact"] else
           f" but torch says {expected:,} — the graph is not the model"
           if expected else "")
        + (f", {opaque} opaque" if opaque else "") + tried_with)
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
                "fit": round(float(getattr(hit, "fit", 0) or 0), 2),
                "topic": int(getattr(hit, "topic", 0) or 0),
                # The library works out how well a paper matches the question
                # (topic) and how much mathematics it names (fit), and my
                # first version threw both away — ranking on fetchability and
                # citation recurrence alone. On "differential diagnosis from
                # chest Xrays" that put a paper about ribosome abundance
                # control at the top, which is not a near miss but a different
                # subject. Topical match has to dominate; being readable here
                # is a convenience and belongs last.
                "score": (int(getattr(hit, "topic", 0) or 0) * 6
                          + float(getattr(hit, "fit", 0) or 0) * 2
                          + min(hit.recurrence or 0, 4)
                          + (1 if text_url else 0)),
            })

    if not scout.findings:
        scout.say("Nothing on topic.",
                  "The gate is deliberate: a paper that scores well but is not "
                  "about the question is not an answer to it.")
        return

    scout.findings.sort(key=lambda f: -f["score"])
    readable = sum(1 for f in scout.findings if f["text_url"])
    on_topic = sum(1 for f in scout.findings if f["topic"] > 0)
    scout.say(f"{len(scout.findings)} papers, {on_topic} matching the question, "
              f"{readable} with fetchable text",
              scout.findings[0]["title"][:120])
    if not on_topic:
        scout.say("None of them actually match the question.",
                  "They were kept for naming a mathematical object, which is "
                  "the other half of the test. Ranking them would be ranking "
                  "the wrong papers, so the order below means little — try "
                  "naming the system you want modelled rather than the task.")


# --------------------------------------------------------------------------
# draft
# --------------------------------------------------------------------------

def assemble(plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn a guided build's plan into one graph.

    The plan is a chain: each step's nodes follow the last node of the step
    before. The frontend has always walked this a step at a time; nothing did
    it in one go, which is what a scout needs.
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    previous: Optional[str] = None
    for step in plan:
        for spec in step.get("nodes") or []:
            nid = f"n{len(nodes) + 1}"
            nodes.append({
                "id": nid,
                "type": spec["type"],
                "params": dict(spec.get("params") or {}),
                "label": spec.get("label") or "",
                "x": 0,
                "y": len(nodes) * 130,
            })
            if previous:
                edges.append({"id": f"e{len(edges) + 1}", "source": previous,
                              "target": nid, "port": 0})
            previous = nid
    return {"name": "draft", "nodes": nodes, "edges": edges}


def _fit_to(graph: Dict[str, Any], shape: Optional[List[int]],
            classes: int) -> List[str]:
    """Point the input and the head at what was asked for."""
    changed = []
    if shape:
        for node in graph["nodes"]:
            if node["type"] == "Input":
                if list(node["params"].get("shape") or []) != shape:
                    node["params"]["shape"] = shape
                    changed.append("input shape set to "
                                   + "\u00d7".join(str(d) for d in shape))
                break
    if classes:
        dense = [n for n in graph["nodes"] if n["type"] == "Linear"]
        if dense and int(dense[-1]["params"].get("units", 0)) != classes:
            dense[-1]["params"]["units"] = classes
            changed.append(f"head set to {classes} classes")
    return changed


def _scout_draft(scout: Scout, config: Dict[str, Any],
                 graph: Dict[str, Any]) -> None:
    import projectloader

    # The catalog is loaded on demand: a scout runs in its own thread and
    # cannot assume the pages that usually load it have been opened.
    if not projectloader.catalog():
        projectloader.load_all()

    shape = None
    raw = str(config.get("shape") or "").strip()
    if raw:
        try:
            shape = [int(x) for x in raw.replace("x", ",").split(",") if x.strip()]
        except ValueError:
            scout.say(f"Could not read “{raw}” as a shape", "Ignoring it.")
    classes = int(config.get("classes") or 0)

    scout.say(f"Looking for a build like “{scout.goal}”")
    # suggest returns the query it understood alongside the matches, which is
    # worth saying: a search that read the words differently explains a
    # surprising shortlist better than the shortlist does.
    found = projectloader.suggest(scout.goal, limit=4)
    matches = found.get("matches") or []
    if found.get("terms"):
        scout.say("Read that as: " + ", ".join(found["terms"]))
    if not matches:
        scout.say("Nothing matched.",
                  "This assembles from the guided builds rather than inventing "
                  "an architecture, so a topic none of them covers has no "
                  "honest answer here.")
        return
    scout.say(f"{len(matches)} candidates",
              ", ".join(m["name"] for m in matches))

    for match in matches:
        if scout.stop.is_set():
            return
        project = projectloader.get(match["id"])
        if not project:
            continue
        built = assemble(project.get("plan") or [])
        if not built["nodes"]:
            continue
        built["name"] = project["name"]
        fitted = _fit_to(built, shape, classes)
        scout.say(f"Assembling {project['name']}",
                  f"{len(built['nodes'])} layers"
                  + (f"; {', '.join(fitted)}" if fitted else ""))

        verdict = _verify(built)
        scout.say(f"   {verdict['summary']}")
        scout.findings.append({
            "kind": "draft",
            "name": project["name"],
            "project": project["id"],
            "summary": project.get("summary", ""),
            "category": project.get("category", ""),
            "layers": len(built["nodes"]),
            "fitted": fitted,
            "graph": built,
            **verdict,
        })

    scout.findings.sort(key=lambda f: (not f["ok"], -f.get("parameters", 0)))
    good = [f for f in scout.findings if f["ok"]]
    if good:
        scout.say(f"{len(good)} of {len(scout.findings)} assembled and ran",
                  good[0]["name"])
    else:
        scout.say("None of them survived the check.",
                  "Each card says what went wrong. An unverified design is not "
                  "offered, because putting one on the canvas would look like "
                  "an answer.")


def _verify(built: Dict[str, Any]) -> Dict[str, Any]:
    """Shapes resolve, the count matches torch, and a batch goes through."""
    import codegen

    report = G.analyze(G.parse(built))
    if not report.get("ok"):
        return {"ok": False, "why": (report.get("errors") or ["it does not resolve"])[0],
                "summary": "the shapes do not resolve"}
    counted = report.get("total_learnables", 0)
    try:
        import torch

        import train as T

        g = G.parse(built)
        model = T.build_model(codegen.to_pytorch(g, report),
                              codegen.model_class_name(g))
        real = sum(p.numel() for p in model.parameters() if p.requires_grad)
        shapes = [report["nodes"][i]["out_shape"]
                  for i in codegen.input_order(g, report)]
        with torch.no_grad():
            model(*[torch.randn(2, *map(int, s)) for s in shapes])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "parameters": counted,
                "why": f"{type(exc).__name__}: {exc}"[:300],
                "summary": "it assembled but would not run"}
    if real != counted:
        return {"ok": False, "parameters": counted, "torch": real,
                "why": f"the canvas counts {counted:,} and torch {real:,}",
                "summary": "the parameter counts disagree"}
    return {"ok": True, "parameters": counted, "torch": real,
            "summary": f"{counted:,} parameters, matching torch, and a batch "
                       f"went through"}


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
