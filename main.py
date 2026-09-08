"""Deep Network Designer — server.

Run it:  uvicorn main:app --reload --port 8770
Then open http://127.0.0.1:8770
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     Response, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

import agents
import advisor
import assistant
import auth
import blockloader
import codegen
import importer
import mathbook
import needs
import projectloader
import recipeloader
import recipes_sdk
import graph as G
import workbook
import quantize as quant
import tracer
import train as T
import walkthrough as walk
from layers import REGISTRY, catalog
from version import __version__

blockloader.load_all()
recipeloader.load_all()
projectloader.load_all()

HERE = Path(__file__).resolve().parent


def _find_frontend() -> Path:
    """Locate index.html without caring how the project was laid out.

    Works whether main.py sits in backend/ next to a sibling frontend/, or
    directly beside the frontend folder, or in the same directory as the page.
    """
    candidates = [
        HERE.parent / "frontend" / "index.html",
        HERE / "frontend" / "index.html",
        HERE / "index.html",
        HERE.parent / "index.html",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


FRONTEND_FILE = _find_frontend()
def saved_dir() -> Path:
    """Designs live in the signed-in account's workspace.

    A function rather than a constant because which directory that is depends
    on who is asking, and that is only known per request.
    """
    return auth.sub("saved")

async def bind_user(request: Request) -> None:
    """Put the signed-in account where the storage helpers can find it.

    An async dependency, and both of those words are load-bearing. Middleware
    will not do: Starlette runs its call_next in a separate task, so a context
    variable set there is invisible to the endpoint. A *synchronous* dependency
    will not do either: FastAPI runs those in a worker thread, which is a
    different context again. An async dependency runs in the request's own
    context, and the endpoint inherits it.

    Both wrong versions failed the same silent way — every account resolved to
    the same workspace — which is exactly the bug worth not having.
    """
    auth.set_current(auth.user_for(request.cookies.get(auth.SESSION_COOKIE)))



app = FastAPI(title="Deep Network Designer", version=__version__, dependencies=[Depends(bind_user)])
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class GraphPayload(BaseModel):
    graph: Dict[str, Any]


class BlockPayload(BaseModel):
    source: str


class TrainPayload(BaseModel):
    graph: Dict[str, Any]
    config: Dict[str, Any] = {}
    book: Optional[Dict[str, Any]] = None    # the whole workbook, if there is one


def _analyze(payload: Dict[str, Any]):
    g = G.parse(payload)
    report = G.analyze(g)
    return g, report


def _codegen(g: G.Graph, report: Dict[str, Any], node_code=None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    stub = ("# Code cannot be generated yet.\n"
            "# {kind}: {msg}\n"
            "# Fix the layers flagged on the canvas and this will fill in.")
    try:
        out["pytorch"] = codegen.to_pytorch(g, report, node_code)
    except Exception as exc:  # noqa: BLE001
        out["pytorch"] = stub.format(kind=type(exc).__name__, msg=exc)
    try:
        out["keras"] = codegen.to_keras(g, report)
    except Exception as exc:  # noqa: BLE001
        out["keras"] = stub.format(kind=type(exc).__name__, msg=exc)
    return out


@app.get("/api/catalog")
def get_catalog():
    return {
        "layers": catalog(),
        "datasets": [
            {"id": k, **v} for k, v in T.BUILTIN_DATASETS.items()
        ] + [{"id": "csv", "label": "Uploaded table (CSV)", "shape": None, "classes": None}],
        "optimizers": ["adamw", "adam", "sgd", "rmsprop"],
        "augmentations": T.AUGMENTATIONS,
        "recipes": recipes_sdk.catalog(),
        "version": __version__,
    }


class BookPayload(BaseModel):
    book: Dict[str, Any]
    sheet: str = ""


@app.post("/api/analyze-book")
def analyze_book(body: BookPayload):
    """The whole workbook at once: every sheet's report, cross-sheet shapes
    resolved, code for the combined model, and per-node code for one sheet."""
    book = workbook.wrap(body.book)
    analysis = workbook.analyze(book)
    source = workbook.to_pytorch(book, analysis)

    active = body.sheet or analysis.get("main") or "main"
    node_code: Dict[str, Any] = {}
    active_report = analysis["sheets"].get(active)
    if active_report and active_report.get("order"):
        try:
            # generated only to collect the per-node constructor lines the
            # canvas shows; the combined source above is what the user sees
            g = G.parse(workbook.sheet_graph(book, active))
            codegen.to_pytorch(g, active_report, node_code)
        except Exception:  # noqa: BLE001
            node_code = {}

    return {
        "sheets": {name: report for name, report in analysis["sheets"].items()},
        "order": workbook.order_sheets(book)[0],
        "cycle": analysis.get("cycle"),
        "main": analysis.get("main"),
        "ok": analysis["ok"],
        "total_learnables": analysis["total_learnables"],
        "approximate": analysis.get("approximate", False),
        "code": {"pytorch": source, "keras": ""},
        "node_code": node_code,
        "signatures": analysis.get("signatures", {}),
    }


@app.post("/api/analyze")
def post_analyze(body: GraphPayload):
    g, report = _analyze(body.graph)
    node_code: Dict[str, Any] = {}
    code = (_codegen(g, report, node_code) if report["order"]
            else {"pytorch": "", "keras": ""})
    try:
        requires = needs.requirements(g, report)
    except Exception as exc:  # noqa: BLE001 - never let this break analysis
        requires = {"error": f"{type(exc).__name__}: {exc}"}
    return {"report": report, "code": code, "node_code": node_code,
            "requires": requires}


class AgentPayload(BaseModel):
    graph: Dict[str, Any]
    agent: str
    config: Dict[str, Any] = {}


@app.get("/api/agents")
def list_agents():
    return {"catalog": agents.CATALOG, "studies": agents.listing()}


@app.post("/api/agents")
def start_agent(body: AgentPayload):
    g, report = _analyze(body.graph)
    if not report["ok"]:
        raise HTTPException(400, detail={
            "message": "Fix the highlighted layers before running a study."})
    try:
        agent = agents.start(body.agent, body.graph, body.config)
    except ValueError as exc:
        raise HTTPException(400, detail={"message": str(exc)})
    return agent.snapshot()


@app.get("/api/agents/{agent_id}")
def get_agent(agent_id: str):
    try:
        return agents.read(agent_id)
    except KeyError:
        raise HTTPException(404, detail={"message": "No such study."})


@app.post("/api/agents/{agent_id}/stop")
def stop_agent(agent_id: str):
    agent = agents.AGENTS.get(agent_id)
    if not agent:
        raise HTTPException(404, detail={"message": "No such study."})
    agent.stop.set()
    return {"ok": True}


@app.get("/api/agents/{agent_id}/graph/{index}")
def agent_trial_graph(agent_id: str, index: int):
    """The exact variant a trial used, so a winner can be opened on the canvas."""
    try:
        study = agents.read(agent_id)
    except KeyError:
        raise HTTPException(404, detail={"message": "No such study."})
    trials = study.get("trials") or []
    if not 0 <= index < len(trials):
        raise HTTPException(404, detail={"message": "No such trial."})
    return trials[index]["graph"]


class AssistantPayload(BaseModel):
    graph: Dict[str, Any]
    message: str = ""


@app.post("/api/assistant")
def ask_assistant(body: AssistantPayload):
    try:
        return assistant.handle(json.loads(json.dumps(body.graph)), body.message)
    except Exception as exc:  # noqa: BLE001 - never let a phrasing crash the panel
        return {"reply": f"That went wrong on my side: {type(exc).__name__}: {exc}"}


class AdvicePayload(BaseModel):
    graph: Dict[str, Any]
    selected: str = ""


@app.post("/api/assistant/advise")
def assistant_advise(body: AdvicePayload):
    """What is worth trying here, and what the selected layer is doing.

    Grounded in the graph: nothing is suggested that this design does not
    actually invite. Whether a change helps is what a study measures; this only
    says what is worth measuring.
    """
    try:
        g = G.parse(json.loads(json.dumps(body.graph)))
        report = G.analyze(g)
        return advisor.advise(g, report, body.selected or None)
    except Exception as exc:  # noqa: BLE001
        return {"ideas": [], "layer": None,
                "error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/assistant/review")
def assistant_review(name: str = ""):
    return {"suggestions": assistant.SUGGESTIONS, "help": assistant.HELP,
            "model": bool(os.environ.get("ASSISTANT_API_KEY"))}


class TestLayerPayload(BaseModel):
    graph: Dict[str, Any]
    node: str


class MathPayload(BaseModel):
    graph: Dict[str, Any]
    node: str


@app.post("/api/math")
def layer_math(body: MathPayload):
    """The mathematics of one layer, with its own numbers in it."""
    g, report = _analyze(body.graph)
    nodes = g.by_id()
    node = nodes.get(body.node)
    if node is None:
        raise HTTPException(404, detail={"message": "No such layer."})

    inc = G.incoming_map(g)
    in_shapes = [report["nodes"].get(e.source, {}).get("out_shape")
                 for e in inc[node.id]]
    out_shape = report["nodes"].get(node.id, {}).get("out_shape")
    entry = mathbook.explain(node.type, G.resolved_params(node),
                             in_shapes, out_shape)
    entry["type"] = node.type
    entry["label"] = node.label or node.type
    entry["in_shapes"] = in_shapes
    entry["out_shape"] = out_shape
    entry["learnables"] = report["nodes"].get(node.id, {}).get("learnables", 0)
    return entry


class TracePayload(BaseModel):
    graph: Dict[str, Any]
    batch: int = 2


class WalkPayload(BaseModel):
    graph: Dict[str, Any]
    batch: int = 1


SPECS_DIR = HERE / "specs"


def _paper_json(paper) -> Dict[str, Any]:
    """The ranked passages, as data. Nothing here decides anything."""
    return {
        "title": getattr(paper, "title", ""),
        "source": getattr(paper, "source", ""),
        "brief": getattr(paper, "brief", ""),
        "sections": sorted(getattr(paper, "sections", {}) or {}),
        "equations": (getattr(paper, "equations", []) or [])[:12],
        "passages": [{"kind": getattr(x, "kind", ""),
                      "label": getattr(x, "label", ""),
                      "text": getattr(x, "text", "")[:1200]}
                     for x in (getattr(paper, "passages", []) or [])[:20]],
    }


def _spec_json(spec) -> Dict[str, Any]:
    import dataclasses

    if dataclasses.is_dataclass(spec):
        return json.loads(json.dumps(dataclasses.asdict(spec), default=str))
    if hasattr(spec, "__dict__"):
        return json.loads(json.dumps(vars(spec), default=str))
    return spec


class Question(BaseModel):
    question: str
    hubs: int = 6
    arxiv: bool = True


def _hit_json(hit) -> Dict[str, Any]:
    from dnn_bench import paper_to_spec as p2s

    return {
        "ref": hit.ref,
        "source": hit.source,
        "title": hit.title,
        "authors": hit.authors,
        "year": hit.year,
        "journal": hit.journal,
        "abstract": (hit.abstract or "")[:900],
        "doi": hit.doi,
        "cited_by": hit.cited_by,
        "is_review": hit.is_review,
        "fit": round(float(hit.fit), 2),
        "topic": hit.topic,
        "reasons": list(hit.reasons or []),
        "recurrence": hit.recurrence,
        # None when the text cannot actually be fetched. A button that can only
        # fail is worse than no button, so the page hides it in that case.
        "text_url": p2s.full_text(hit),
    }


@app.post("/api/paper/discover")
def paper_discover(body: Question):
    """A research question, and what the literature offers in reply.

    Three passes: what matches the question, what the well-cited reviews on it
    point back at, and what is on arXiv. The middle one is the useful one — a
    result worth modelling is usually cited by the reviews rather than being the
    top hit itself.

    This reaches the internet. It is the only part of the app that does.
    """
    if not (body.question or "").strip():
        raise HTTPException(400, detail={"message": "Ask it something."})
    from dnn_bench import paper_to_spec as p2s

    log: List[str] = []
    try:
        found = p2s.discover(body.question.strip(), n_hubs=max(1, min(12, body.hubs)),
                             use_arxiv=bool(body.arxiv), log=lambda *a: log.append(" ".join(map(str, a))))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, detail={
            "message": f"The search did not complete: {type(exc).__name__}: {exc}. "
                       f"This is the one part of the app that needs the internet."})
    groups = []
    for key, title, why in (
        ("direct", "Directly on the question",
         "Matched the question itself."),
        ("foundational", "What the reviews point back at",
         "Turned up repeatedly in the reference lists of the reviews. This is "
         "usually where a modellable result is."),
        ("hubs", "Reviews and surveys",
         "Well cited and broad. Good for orientation, rarely the result itself."),
        ("preprints", "Preprints",
         "From arXiv. The source is available, so the equations come through exactly."),
    ):
        items = found.get(key) or []
        if items:
            groups.append({"key": key, "title": title, "why": why,
                           "papers": [_hit_json(h) for h in items]})
    # discover swallows a network failure and returns nothing, which reads as
    # "no papers matched" when the truth is "nothing was asked". Those are
    # different answers and the second one is not the searcher's fault.
    unreachable = [line.strip() for line in log if "unavailable" in line]
    return {"question": found.get("question", ""),
            "terms": found.get("terms", []),
            "groups": groups,
            "reachable": not (unreachable and not groups),
            "trouble": unreachable[:3],
            "log": log[-40:]}


class FetchPaper(BaseModel):
    url: str
    title: str = ""


@app.post("/api/paper/fetch")
def paper_fetch(body: FetchPaper):
    """Pull one paper's text and rank its passages."""
    if not body.url:
        raise HTTPException(400, detail={
            "message": "That paper's full text is not fetchable — only the open "
                       "access subset has one. Try another, or paste the theorem."})
    from dnn_bench import paper_to_spec as p2s

    try:
        paper = p2s.ingest(body.url, title=body.title)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, detail={
            "message": f"Could not read it: {type(exc).__name__}: {exc}"})
    return _paper_json(paper)


class PaperText(BaseModel):
    text: str = ""
    title: str = ""
    url: str = ""
    only: List[int] = []          # passage indices the reader chose


@app.post("/api/paper/ingest")
async def paper_ingest(file: Optional[UploadFile] = File(None),
                       text: str = Form(""), title: str = Form("")):
    """Rank the passages that plausibly carry a structural result.

    Mechanical: it decides nothing. It means reading fifteen ranked passages
    instead of forty pages, and a result that never says "unique" will rank low
    and still be the right one.
    """
    from dnn_bench import ingest

    source = text
    if file is not None:
        blob = await file.read()
        target = Path(tempfile.gettempdir()) / f"paper-{uuid.uuid4().hex[:8]}.pdf"
        target.write_bytes(blob)
        source = str(target)
        title = title or file.filename or ""
    if not source.strip():
        raise HTTPException(400, detail={"message": "Give it a PDF or some text."})
    try:
        paper = ingest.ingest(source, title=title)
    except RuntimeError as exc:
        raise HTTPException(400, detail={
            "message": _missing_package(exc) or str(exc)})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, detail={"message": f"{type(exc).__name__}: {exc}"})
    return _paper_json(paper)


@app.post("/api/paper/propose")
def paper_propose(body: PaperText):
    """Ask for a candidate spec. The step to trust least, and it can refuse.

    `only` is the passages the reader ticked. It matters more than it looks:
    the top-ranked passage is not always the load-bearing one, and a proposer
    fed the wrong theorem will confidently model the wrong thing. Choosing the
    passages is the reader's judgment, and it is the judgment that carries.
    """
    from dnn_bench import ingest, propose

    source = body.url or body.text or ""
    if not source.strip():
        raise HTTPException(400, detail={"message": "Read a paper first."})
    try:
        paper = ingest.ingest(source, title=body.title)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, detail={
            "message": f"Could not re-read it: {type(exc).__name__}: {exc}"})
    chosen = [i for i in body.only if 0 <= i < len(paper.passages)] or None
    if not propose.available():
        # Without a proposer, hand back a skeleton carrying what is actually
        # known: the title, the source, and the chosen passages as the claim to
        # be turned into a residual. Everything invented is left as a blank to
        # fill rather than a plausible guess to correct.
        blank = _spec_json(propose.blank_spec(paper))
        if chosen:
            picked = [paper.passages[i] for i in chosen]
            blank["claim"] = (getattr(picked[0], "text", "") or "")[:300]
            blank["_passages"] = [getattr(p, "text", "")[:600] for p in picked]
        if paper.equations:
            blank["_equations"] = paper.equations[:12]
        return {"proposed": False,
                "reason": "No ANTHROPIC_API_KEY in the server's environment, so "
                          "there is no proposer. Below is the skeleton with what "
                          "is known filled in — the title, the source, your "
                          "chosen passages and any displayed expressions. The "
                          "residual is the part that needs a person.",
                "spec": blank}
    try:
        result = propose.propose(paper, only=chosen)
    except Exception as exc:  # noqa: BLE001
        return {"proposed": False, "reason": f"{type(exc).__name__}: {exc}",
                "spec": _spec_json(propose.blank_spec(paper))}
    if isinstance(result, dict) and result.get("usable") is False:
        return {"proposed": False,
                "reason": result.get("why")
                          or "The proposer judged this paper to offer a narrative "
                             "mechanism rather than a constraint. Refusing is the "
                             "right answer more often than not.",
                "spec": _spec_json(propose.blank_spec(paper))}
    return {"proposed": True, "spec": _spec_json(result)}


class SpecBody(BaseModel):
    spec: Dict[str, Any]


@app.get("/api/paper/examples")
def spec_examples():
    """The specs that already work, to start from.

    Writing a residual from a blank template is the hardest step in the whole
    pipeline, and nothing in the app was helping with it. A working spec of the
    same shape is far more use than a description of the fields: a paper about a
    compartmental system wants the Bergman one, and editing three lines of it is
    a different task from inventing the file.
    """
    out = []
    for path in sorted(SPECS_DIR.glob("*.json")):
        try:
            blob = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        out.append({
            "key": blob.get("key", path.stem),
            "title": blob.get("title", path.stem),
            "claim": blob.get("claim", ""),
            "shape": ("a compartmental system of ordinary differential equations"
                      if "cat(" in str(blob.get("residual", ""))
                      else "a potential whose gradient vanishes at the answer"),
            "spec": blob,
        })
    return {"examples": out}


@app.post("/api/paper/check")
def paper_check(body: SpecBody):
    """Load the spec as a domain and run the bench's checks on it.

    A spec supplies only the residual; the Jacobian is differentiated from it,
    so the two cannot disagree. What is left to get wrong is the physics, which
    is what this is for.
    """
    from dnn_bench import spec as specmod, validate

    key = str(body.spec.get("key") or "").strip()
    if not key:
        raise HTTPException(400, detail={"message": "The spec needs a key."})
    from dnn_bench import core

    # Loading a spec registers it, and a spec that fails its checks must not be
    # left in the registry offering itself as a layer. Anything that was not
    # already there when we started is withdrawn unless it passes.
    known = set(core.REGISTRY)
    scratch = Path(tempfile.gettempdir()) / f"spec-{uuid.uuid4().hex[:8]}.json"
    scratch.write_text(json.dumps(body.spec))

    def withdraw():
        for added in set(core.REGISTRY) - known:
            core.REGISTRY.pop(added, None)

    try:
        specmod.load_spec_file(str(scratch))
    except Exception as exc:  # noqa: BLE001
        withdraw()
        return {"ok": False, "stage": "load",
                "message": f"{type(exc).__name__}: {exc}"}
    finally:
        scratch.unlink(missing_ok=True)
    try:
        result = validate.validate_domain(key, quick=True)
    except Exception as exc:  # noqa: BLE001
        withdraw()
        return {"ok": False, "stage": "validate",
                "message": f"{type(exc).__name__}: {exc}"}
    if not result.get("ok"):
        withdraw()
    return {"ok": bool(result.get("ok")), "stage": "validate", **result}


@app.post("/api/paper/save")
def paper_save(body: SpecBody):
    """Write the spec to specs/ and register it, so the layer offers it at once."""
    key = re.sub(r"[^a-z0-9_]+", "_", str(body.spec.get("key") or "").lower()).strip("_")
    if not key:
        raise HTTPException(400, detail={"message": "The spec needs a key."})
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPECS_DIR / f"{key}.json"
    saved = dict(body.spec, key=key)
    path.write_text(json.dumps(saved, indent=1))
    try:
        from dnn_bench import spec as specmod

        specmod.load_spec_file(str(path))
        blockloader.load_all()          # the domain list is read when a block installs
    except Exception as exc:  # noqa: BLE001
        path.unlink(missing_ok=True)
        raise HTTPException(400, detail={
            "message": f"Saved nothing: {type(exc).__name__}: {exc}"})
    from dnn_bench import core

    return {"saved": str(path.name), "key": key,
            "domains": [e["key"] for e in core.all_domains()]}


@app.get("/api/domains")
def list_domains(seed: int = 0):
    """Every implicit domain, its arms, and what each arm actually is.

    The facts come from the layer describing itself after it is built, not from
    the spec that asked for it — so what is shown is what exists.
    """
    try:
        from dnn_bench import core, domains  # noqa: F401
    except ImportError:
        return {"domains": [], "reason": "dnn_bench is not installed beside the app."}

    out = []
    for entry in core.all_domains():
        spec = core.get(entry["key"])
        arms = []
        try:
            built = spec.variants(spec.defaults(), seed=int(seed))
        except Exception as exc:  # noqa: BLE001
            out.append({**entry, "arms": [], "error": f"{type(exc).__name__}: {exc}"})
            continue
        for variant in built:
            try:
                layer = variant.build()
                described = layer.describe() or {}
                cert = layer.certificate() or {}
            except Exception as exc:  # noqa: BLE001
                arms.append({"key": variant.key, "label": variant.label,
                             "structured": variant.structured,
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            arms.append({
                "key": variant.key,
                "label": variant.label,
                "structured": bool(variant.structured),
                "summary": described.get("summary", ""),
                "facts": described.get("facts", []),
                "expression": described.get("expression", ""),
                "holds": bool(cert.get("holds")),
                "claim": cert.get("claim", ""),
                "parameters": sum(q.numel() for q in layer.parameters()),
            })
        out.append({**entry, "arms": arms})
    return {"domains": out, "seed": int(seed)}


class DomainCheck(BaseModel):
    domain: str
    seed: int = 0
    quick: bool = True


@app.post("/api/check-domain")
def check_domain(body: DomainCheck):
    """Run the bench's own checks on an implicit domain.

    This is the layer checking its mathematics rather than the canvas checking
    its arithmetic: the Jacobian against the residual it claims to differentiate,
    the implicit gradients against finite differences, the equilibrium against
    several starting points. A Jacobian with a sign error still converges, so
    without this the only symptom is silently wrong gradients.
    """
    try:
        from dnn_bench import domains, validate  # noqa: F401
    except ImportError:
        raise HTTPException(400, detail={
            "message": "dnn_bench is not installed beside the app."})
    try:
        return validate.validate_domain(body.domain, seed=int(body.seed),
                                        quick=bool(body.quick))
    except KeyError:
        raise HTTPException(400, detail={
            "message": f"No domain called {body.domain!r}."})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, detail={"message": f"{type(exc).__name__}: {exc}"})


@app.post("/api/walkthrough")
def explain_pass(body: WalkPayload):
    """Step through the network with real values at every layer."""
    try:
        return walk.walkthrough(body.graph, batch=max(1, min(8, body.batch)))
    except ValueError as exc:
        raise HTTPException(400, detail={"message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, detail={
            "message": _missing_package(exc) or f"{type(exc).__name__}: {exc}"})


class QuantPayload(BaseModel):
    graph: Dict[str, Any]
    batch: int = 32
    weights: str = ""        # a saved checkpoint, so the drift means something


@app.post("/api/quantize")
def quantize_design(body: QuantPayload):
    """What the design costs at lower precision, and which layer objects."""
    # Drift measured on random weights is a property of the shapes, not of the
    # model — a trained network and an untrained one of the same shape are not
    # equally sensitive to losing precision. Measuring a checkpoint is the
    # question anyone actually has.
    path = ""
    if body.weights:
        target = T.CHECKPOINTS / Path(body.weights).name
        if not target.exists():
            raise HTTPException(400, detail={
                "message": f"No saved weights called {body.weights}."})
        path = str(target)
    try:
        return quant.quantize_report(body.graph, batch=max(1, min(128, body.batch)),
                                     weights=path or None)
    except ValueError as exc:
        raise HTTPException(400, detail={"message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, detail={
            "message": _missing_package(exc) or f"{type(exc).__name__}: {exc}"})


@app.post("/api/trace")
def trace_forward(body: TracePayload):
    """Run one batch through the design and report what each layer did."""
    try:
        return tracer.run_trace(body.graph, batch=max(1, min(64, body.batch)))
    except ValueError as exc:
        raise HTTPException(400, detail={"message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, detail={
            "message": _missing_package(exc) or f"{type(exc).__name__}: {exc}"})


@app.post("/api/test-layer")
def test_layer(body: TestLayerPayload):
    """Build one layer on its own and push a tensor through it.

    The canvas predicts shapes arithmetically. This checks that prediction
    against what PyTorch actually does for this layer, with these settings, on
    this input — which is a different claim, and the one that matters.
    """
    import time as _time

    import torch

    g, report = _analyze(body.graph)
    nodes = g.by_id()
    node = nodes.get(body.node)
    if node is None:
        raise HTTPException(404, detail={"message": "No such layer."})

    spec = REGISTRY.get(node.type)
    if spec is None or spec.kind == "runtime" or node.type in ("Input", "Output"):
        raise HTTPException(400, detail={
            "message": f"{node.type} is not something that can be run on its own."})

    inc = G.incoming_map(g)
    in_shapes = []
    for e in inc[node.id]:
        shape = report["nodes"].get(e.source, {}).get("out_shape")
        if not shape:
            raise HTTPException(400, detail={
                "message": "The layer feeding this one has not resolved yet."})
        in_shapes.append(shape)
    if not in_shapes:
        raise HTTPException(400, detail={"message": "Nothing is connected to this layer."})

    params = G.resolved_params(node)
    predicted = report["nodes"].get(node.id, {}).get("out_shape")

    try:
        ctor = spec.torch_init(params, in_shapes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, detail={"message": f"Cannot build it: {exc}"})

    namespace: Dict[str, Any] = {"torch": torch, "nn": torch.nn,
                                 "F": torch.nn.functional, "math": __import__("math")}
    if spec.torch_prelude:
        exec(compile(spec.torch_prelude, "<layer>", "exec"), namespace)  # noqa: S102

    batch = 2
    tensors = []
    for shape in in_shapes:
        if node.type == "Embedding":
            vocab = int(params.get("vocab", 1000))
            tensors.append(torch.randint(0, vocab, (batch, *[int(d) for d in shape])))
        else:
            tensors.append(torch.randn(batch, *[int(d) for d in shape]))

    try:
        module = eval(ctor, namespace) if ctor else None  # noqa: S307
        names = [f"x{i}" for i in range(len(tensors))]
        call = spec.torch_call(params, names, "m", in_shapes)
        scope = dict(namespace)
        scope["m"] = module
        for name, tensor in zip(names, tensors):
            scope[name] = tensor
        started = _time.perf_counter()
        with torch.no_grad():
            out = eval(call, scope)  # noqa: S307
        elapsed = (_time.perf_counter() - started) * 1000
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "constructor": ctor, "inputs": in_shapes}

    actual = list(out.shape)[1:]
    learnables = (sum(p.numel() for p in module.parameters())
                  if hasattr(module, "parameters") else 0)
    return {
        "ok": True,
        "constructor": ctor,
        "inputs": in_shapes,
        "predicted": predicted,
        "actual": actual,
        "matches": list(predicted or []) == actual,
        "learnables": learnables,
        "ms": round(elapsed, 2),
        "dtype": str(out.dtype).replace("torch.", ""),
    }


@app.post("/api/train")
def post_train(body: TrainPayload):
    g, report = _analyze(body.graph)
    if not report["ok"]:
        raise HTTPException(
            400,
            detail={
                "message": "Fix the highlighted layers before training.",
                "errors": report["errors"] or [
                    v["error"] for v in report["nodes"].values() if v["error"]
                ],
            },
        )

    # forward() takes its arguments in topological order, so the loader has to
    # hand tensors over in that same order or the towers get swapped.
    nodes = g.by_id()
    inputs = [nodes[i] for i in report["order"] if nodes[i].type == "Input"]
    outputs = [nodes[i] for i in report["order"] if nodes[i].type == "Output"]
    in_shapes = [report["nodes"][n.id]["out_shape"] for n in inputs]
    in_ids = [n.id for n in inputs]
    out_shape = report["nodes"][outputs[0].id]["out_shape"]
    tasks = [G.resolved_params(n).get("task", "classification") for n in outputs] \
        or ["classification"]

    cfg = dict(body.config)
    cfg["graph"] = body.graph          # stored inside the checkpoint

    # Fail on the obvious mismatches here, where the message can reach the form,
    # rather than a few seconds later inside the training thread.
    try:
        # before the loaders, because an image loader asked for a token
        # sequence does not fail politely — it takes the process with it
        T.check_dataset_fits(cfg, in_shapes)
        T._make_loaders  # noqa: B018 - presence check only
        if cfg.get("dataset") == "csv":
            if not cfg.get("csv_file"):
                raise T.DataError("Choose an uploaded table first.")
            if not cfg.get("target_column"):
                raise T.DataError("Choose which column is the target.")
        if cfg.get("dataset") == "folder":
            T.inspect_folder(cfg.get("folder") or "")
        if cfg.get("dataset") == "text":
            if not cfg.get("text_file"):
                raise T.DataError("Choose an uploaded .txt corpus first.")
            T.inspect_text(cfg["text_file"])
    except T.DataError as exc:
        raise HTTPException(400, detail={"message": str(exc)})

    try:
        # A sheet of a workbook cannot be generated on its own: the model sheet
        # refers to classes defined by the others, and the file that comes out
        # names a class that is not in it. GPT2 fails with
        # "NameError: name 'Block' is not defined", which is true and unhelpful.
        if body.book and len(body.book.get("sheets") or []) > 1:
            analysis = workbook.analyze(body.book)
            if not analysis["ok"]:
                bad = [f"{name}: {rep['errors'][0]}"
                       for name, rep in analysis["sheets"].items()
                       if rep.get("errors")]
                raise HTTPException(400, detail={
                    "message": "Fix the sheets before training.", "errors": bad})
            source = workbook.to_pytorch(body.book, analysis)
            class_name = codegen.model_class_name(
                G.parse(workbook.sheet_graph(body.book, body.book.get("main"))))
        else:
            source = codegen.to_pytorch(g, report)
            class_name = codegen.model_class_name(g)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, detail={"message": f"Code generation failed: {exc}"})

    job = T.start(source, cfg, in_shapes, in_ids, out_shape, tasks, class_name)
    return {"job": job.snapshot(), "source": source,
            "inputs": [{"id": i, "shape": s} for i, s in zip(in_ids, in_shapes)]}


# --------------------------------------------------------------------------
# uploaded tables
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# importing an existing model
# --------------------------------------------------------------------------

TORCHVISION_MODELS = [
    "resnet18", "resnet34", "resnet50", "resnet101", "vgg16", "vgg19",
    "alexnet", "squeezenet1_1", "densenet121", "mobilenet_v2",
    "mobilenet_v3_large", "efficientnet_b0", "regnet_y_400mf",
    "convnext_tiny", "shufflenet_v2_x1_0", "googlenet", "inception_v3",
]


class ImportPayload(BaseModel):
    arch: str = ""
    weights: str = "none"
    input_shape: List[int] = [3, 224, 224]


class CodePayload(BaseModel):
    source: str
    input_shape: List[int] = [3, 224, 224]
    entry: str = ""


class ScanPayload(BaseModel):
    root: str


class FolderImportPayload(BaseModel):
    root: str
    picks: List[Dict[str, Any]]        # [{file, cls, arguments, input_shape}]
    as_sheets: bool = True
    setup: str = ""


def _finish_import(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze what came in so the response can say what needs attention."""
    expected = graph.pop("_expected_parameters", None)
    routing = graph.pop("_routing", None)
    notes = graph.pop("_notes", [])
    g = G.parse(graph)
    report = G.analyze(g)
    problems = [v["error"] for v in report["nodes"].values() if v["error"]]

    warnings = []
    if routing:
        warnings.append(
            f"This model decides at run time which path data takes "
            f"({', '.join(routing)}). A diagram is a fixed structure, so every "
            f"branch is drawn as though it always runs — for a mixture of "
            f"experts that means all the experts appear, when the design is "
            f"that only a few of them run per token.")
    if expected is not None and report["ok"] and expected != report["total_learnables"]:
        gap = expected - report["total_learnables"]
        warnings.append(
            f"The model holds {expected:,} parameters but the diagram accounts "
            f"for {report['total_learnables']:,}, a difference of {abs(gap):,}. "
            f"That usually means a learned tensor is used in plain arithmetic "
            f"rather than through a layer — a hand-written norm's scale, for "
            f"instance — so it belongs to no node here.")

    return {
        "graph": graph,
        "notes": notes,
        "ok": report["ok"],
        "nodes": len(graph["nodes"]),
        "problems": (report["errors"] + problems)[:6],
        "problem_count": len(report["errors"]) + len(problems),
        "learnables": report["total_learnables"],
        "expected_parameters": expected,
        "warnings": warnings,
    }


def _missing_package(exc: Exception) -> Optional[str]:
    text = str(exc)
    if "torchvision" in text:
        return ("Importing an architecture by name needs torchvision, which is "
                "not installed for the interpreter running this server. "
                f"Install it with: {sys.executable} -m pip install torchvision")
    if "No module named" in text:
        return f"{text}. Install it for {sys.executable}."
    return None


@app.get("/api/import/models")
def import_models():
    return {"torchvision": TORCHVISION_MODELS}


@app.post("/api/import/torchvision")
def import_torchvision(body: ImportPayload):
    if body.arch not in TORCHVISION_MODELS:
        raise HTTPException(400, detail={"message": f"Unknown model {body.arch}."})
    try:
        graph = importer.from_torchvision(body.arch, body.weights, body.input_shape)
    except importer.ImportError_ as exc:
        raise HTTPException(400, detail={"message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, detail={
            "message": _missing_package(exc) or f"{type(exc).__name__}: {exc}"})
    return _finish_import(graph)


@app.post("/api/scan-folder")
def scan_folder(body: ScanPayload):
    """List every nn.Module in a folder. Reads syntax trees only — nothing runs
    until a class is actually picked for import."""
    try:
        return importer.scan_folder(body.root)
    except importer.ImportError_ as exc:
        raise HTTPException(400, detail={"message": str(exc)})


class RepoPayload(BaseModel):
    url: str
    refresh: bool = False


@app.post("/api/github")
def fetch_github(body: RepoPayload):
    """Download a public repository and scan it.

    Downloading and reading are safe. Nothing in it runs until a class is picked
    for import, which is the same rule as for a folder already on disk.
    """
    try:
        info = importer.fetch_repo(body.url, refresh=body.refresh)
        found = importer.scan_folder(info["root"])
    except importer.ImportError_ as exc:
        raise HTTPException(400, detail={"message": str(exc)})
    return {**found, **info}


@app.post("/api/scan-folder/tree")
def scan_tree(body: ScanPayload):
    """The scanned folder as a tree, so it can be browsed rather than re-scanned.

    Python files carry the classes found in them, so the sidebar can show which
    files actually hold models.
    """
    base = Path(body.root).expanduser()
    if not base.is_dir():
        raise HTTPException(400, detail={"message": f"{base} is not a folder."})
    try:
        found = importer.scan_folder(body.root)
    except importer.ImportError_ as exc:
        raise HTTPException(400, detail={"message": str(exc)})

    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for model in found["models"]:
        by_file.setdefault(model["file"], []).append(model)
    broken = {item["file"]: item["why"] for item in found["skipped"]}

    entries = []
    for path in sorted(base.rglob("*")):
        if set(path.relative_to(base).parts) & importer.SKIP_BELOW:
            continue
        rel = str(path.relative_to(base))
        if path.is_dir():
            entries.append({"path": rel, "dir": True})
        elif path.suffix in (".py", ".json", ".yaml", ".yml", ".toml", ".txt",
                             ".md", ".cfg", ".ini"):
            entries.append({"path": rel, "dir": False,
                            "size": path.stat().st_size,
                            "models": by_file.get(rel, []),
                            "broken": broken.get(rel)})
        if len(entries) > 1200:
            break
    return {"root": str(base), "entries": entries,
            "models": len(found["models"])}


@app.post("/api/scan-folder/file")
def scan_file(body: Dict[str, Any]):
    """One file's text, for reading in the panel. Never executes anything."""
    base = Path(str(body.get("root", ""))).expanduser().resolve()
    target = (base / str(body.get("path", ""))).resolve()
    # a path that climbs out of the scanned folder is not a file of this project
    if not str(target).startswith(str(base)):
        raise HTTPException(400, detail={"message": "That is outside the folder."})
    if not target.is_file():
        raise HTTPException(404, detail={"message": "No such file."})
    if target.stat().st_size > 400_000:
        raise HTTPException(400, detail={"message": "That file is too large to show."})
    return {"path": str(target.relative_to(base)),
            "text": target.read_text(encoding="utf-8", errors="replace")}


def _failure_kind(reason: str) -> str:
    """Which sort of failure this was.

    Saying "will not trace" for every failure sends people to look at the wrong
    thing. Not being given a config is a different problem from a forward() that
    cannot be traced, and only one of them is the user's to fix.
    """
    text = reason.lower()
    if "could not be built with" in text or "missing" in text and "argument" in text:
        return "arguments"
    if "did not import" in text or "no module named" in text:
        return "module"
    if "cannot be traced" in text or "variadic" in text:
        return "trace"
    if "has no class called" in text:
        return "missing"
    return "error"


class ClassProbe(BaseModel):
    root: str
    file: str
    cls: str
    arguments: str = ""
    setup: str = ""
    input_shape: List[int] = [3, 224, 224]


def _class_source(root: str, file: str, cls: str) -> Dict[str, Any]:
    """The source of one class, found by its syntax tree rather than by guessing
    where it ends."""
    import ast

    base = Path(root).expanduser().resolve()
    target = (base / file).resolve()
    if not str(target).startswith(str(base)) or not target.is_file():
        raise HTTPException(404, detail={"message": "No such file."})
    text = target.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            lines = text.splitlines()
            start = (node.decorator_list[0].lineno - 1
                     if node.decorator_list else node.lineno - 1)
            end = node.end_lineno or (start + 40)
            body = "\n".join(lines[start:end])
            doc = ast.get_docstring(node) or ""
            methods = [f.name for f in node.body
                       if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))]
            init = next((f for f in node.body if isinstance(f, ast.FunctionDef)
                         and f.name == "__init__"), None)
            signature = ""
            if init:
                args = [a.arg for a in init.args.args][1:]
                defaults = [ast.unparse(d) for d in init.args.defaults]
                pad = [""] * (len(args) - len(defaults)) + defaults
                signature = ", ".join(
                    a + (f"={d}" if d else "") for a, d in zip(args, pad))
            return {"source": body, "doc": doc, "methods": methods,
                    "signature": signature, "line": node.lineno,
                    "lines": end - start}
    raise HTTPException(404, detail={"message": f"{cls} is not in {file}."})


@app.post("/api/scan-folder/source")
def scan_source(body: ClassProbe):
    return _class_source(body.root, body.file, body.cls)


@app.post("/api/scan-folder/try")
def scan_try(body: ClassProbe):
    """Attempt one class and report how it went, without touching the canvas.

    With hundreds of classes in a library, most of which cannot be traced,
    trying one should not mean committing to it.
    """
    try:
        graph = importer.from_folder(body.root, body.file, body.cls,
                                     body.input_shape, body.arguments,
                                     body.setup)
    except importer.ImportError_ as exc:
        return {"ok": False, "reason": str(exc), "kind": _failure_kind(str(exc))}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}",
                "kind": _failure_kind(str(exc))}

    graph.pop("_entry", None)
    notes = graph.pop("_notes", [])
    expected = graph.pop("_expected_parameters", None)
    routing = graph.pop("_routing", None)
    g, report = _analyze(graph)
    opaque = sum(1 for n in graph["nodes"] if n["type"] == "Custom")
    return {
        "ok": report["ok"],
        "kind": "" if report["ok"] else "shapes",
        "nodes": len(graph["nodes"]),
        "opaque": opaque,
        "learnables": report["total_learnables"],
        "expected": expected,
        "routing": routing,
        "reason": (report["errors"][:1] or [""])[0] if not report["ok"] else "",
        "notes": notes[:3],
    }


@app.post("/api/import/folder")
def import_folder(body: FolderImportPayload):
    """Import chosen classes from a scanned folder.

    With as_sheets, each class becomes its own sheet in a workbook — a project
    that spans several files arrives as several sheets, not one tangle.
    """
    if not body.picks:
        raise HTTPException(400, detail={"message": "Nothing was picked."})
    sheets, notes, failures = [], [], []
    for pick in body.picks:
        try:
            graph = importer.from_folder(
                body.root, pick.get("file", ""), pick.get("cls", ""),
                pick.get("input_shape") or [3, 224, 224],
                str(pick.get("arguments") or ""), body.setup)
        except importer.ImportError_ as exc:
            failures.append({"cls": pick.get("cls"), "why": str(exc)})
            continue
        graph.pop("_entry", None)
        notes.extend(f"{pick.get('cls')}: {n}" for n in graph.pop("_notes", []))
        sheets.append({"name": pick.get("cls") or f"sheet{len(sheets)+1}",
                       "nodes": graph["nodes"], "edges": graph["edges"]})
    if not sheets:
        raise HTTPException(400, detail={
            "message": "None of the picks imported. "
                       + "; ".join(f["why"] for f in failures[:2])})

    if not body.as_sheets and len(sheets) == 1:
        payload = {"name": sheets[0]["name"], "nodes": sheets[0]["nodes"],
                   "edges": sheets[0]["edges"]}
        result = _finish_import(payload)
        result["failures"] = failures
        return result

    book = {"name": Path(body.root).name or "Imported",
            "main": sheets[-1]["name"],
            "sheets": sheets}
    analysis = workbook.analyze(book)
    return {"book": book, "analysis": _book_summary(analysis),
            "notes": notes[:12], "failures": failures,
            "ok": analysis["ok"], "learnables": analysis["total_learnables"]}


def _book_summary(analysis: Dict[str, Any]) -> Dict[str, Any]:
    return {name: {"ok": r.get("ok"), "errors": r.get("errors", [])[:2],
                   "learnables": r.get("total_learnables", 0)}
            for name, r in analysis.get("sheets", {}).items()}


@app.post("/api/import/code")
def import_code(body: CodePayload):
    """Trace a module from pasted source.

    This executes the code, which is the only way to obtain a module to trace:
    no static reader can tell you what forward() does. Same trust assumption as
    the blocks and recipes folders — fine locally, not something to expose.
    """
    try:
        graph = importer.from_source(body.source, body.input_shape, body.entry)
    except importer.ImportError_ as exc:
        raise HTTPException(400, detail={"message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, detail={"message": f"{type(exc).__name__}: {exc}"})
    entry = graph.pop("_entry", "")
    result = _finish_import(graph)
    result["entry"] = entry
    return result


@app.post("/api/import/upload")
async def import_upload(file: UploadFile = File(...), shape: str = "3,224,224"):
    name = Path(file.filename or "model").name
    target = T.UPLOADS / name
    target.write_bytes(await file.read())
    dims = [int(x) for x in re.split(r"[^0-9]+", shape) if x] or [3, 224, 224]
    try:
        if name.lower().endswith(".onnx"):
            graph = importer.from_onnx(str(target))
        elif name.lower().endswith((".pt", ".pth")):
            graph = importer.from_torch_file(str(target), dims)
        else:
            raise importer.ImportError_(
                "Upload a .onnx file, or a .pt holding the module itself.")
    except importer.ImportError_ as exc:
        raise HTTPException(400, detail={"message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, detail={"message": f"{type(exc).__name__}: {exc}"})
    return _finish_import(graph)


# --------------------------------------------------------------------------
# talking to a trained model
# --------------------------------------------------------------------------

_LOADED: Dict[str, Any] = {}


class ChatPayload(BaseModel):
    checkpoint: str
    prompt: str = ""
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40
    stop: str = ""


def _load_for_chat(file: str):
    """Rebuild the network a checkpoint came from and load its weights.

    Cached, because rebuilding and reloading on every message would make the
    panel feel broken even though the sampling itself is fast.
    """
    import torch

    if file in _LOADED:
        return _LOADED[file]

    path = T.CHECKPOINTS / Path(file).name
    if not path.exists():
        raise HTTPException(404, detail={"message": f"No checkpoint named {path.name}."})
    blob = torch.load(path, map_location="cpu", weights_only=False)

    vocab = blob.get("vocab")
    graph_blob = blob.get("graph") or {}
    if not graph_blob.get("nodes"):
        raise HTTPException(400, detail={
            "message": f"{path.name} was saved without its design, so the network "
                       f"cannot be rebuilt to talk to."})

    g = G.parse(graph_blob)
    report = G.analyze(g)
    if not report["ok"]:
        raise HTTPException(400, detail={
            "message": "The design stored in that checkpoint no longer analyzes cleanly."})

    model = T.build_model(codegen.to_pytorch(g, report), codegen.model_class_name(g))
    T.load_into(model, path.name)
    model.eval()

    if not vocab:
        # older checkpoints predate vocab storage; fall back to the corpus the
        # TextGenerator node points at
        nodes = g.by_id()
        gen = next((n for n in nodes.values() if n.type == "TextGenerator"), None)
        vpath = Path(G.resolved_params(gen).get("vocab_path", "")) if gen else None
        if vpath and vpath.name:
            candidate = vpath if vpath.exists() else T.UPLOADS / vpath.name
            if candidate.exists():
                vocab = json.loads(candidate.read_text())
    if not vocab:
        raise HTTPException(400, detail={
            "message": f"{path.name} carries no vocabulary, so its output cannot be "
                       f"turned back into text. Retrain on a text corpus, or point a "
                       f"TextGenerator node at the .vocab.json file."})

    inputs = [nid for nid in report["order"] if g.by_id()[nid].type == "Input"]
    block = blob.get("block") or report["nodes"][inputs[0]]["out_shape"][0]
    _LOADED.clear()                      # one model resident at a time
    _LOADED[file] = (model, vocab, int(block))
    return _LOADED[file]


@app.get("/api/chat/models")
def chat_models():
    """Checkpoints that carry a vocabulary and can therefore be talked to."""
    return {"checkpoints": [c for c in T.list_checkpoints() if c.get("chat")]}


@app.post("/api/chat")
def chat(body: ChatPayload):
    model, vocab, block = _load_for_chat(body.checkpoint)
    result = T.generate_text(
        model, vocab, block, body.prompt,
        max_new_tokens=max(1, min(int(body.max_new_tokens), 2000)),
        temperature=float(body.temperature), top_k=int(body.top_k),
        stop=body.stop)
    result["block_size"] = block
    result["vocab_size"] = len(vocab)
    return result


@app.get("/api/folder")
def describe_folder(path: str):
    try:
        return T.inspect_folder(path)
    except T.DataError as exc:
        raise HTTPException(400, detail={"message": str(exc)})


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------

@app.get("/api/checkpoints")
def get_checkpoints():
    return {"checkpoints": T.list_checkpoints()}


@app.get("/api/checkpoints/{file}/download")
def download_checkpoint(file: str):
    path = T.CHECKPOINTS / Path(file).name
    if not path.exists():
        raise HTTPException(404, detail={"message": "No checkpoint by that name."})
    return FileResponse(path, media_type="application/octet-stream",
                        filename=path.name)


@app.get("/api/checkpoints/{file}/graph")
def checkpoint_graph(file: str):
    """The design a checkpoint was trained from, so it can be reopened on the canvas."""
    import torch

    path = T.CHECKPOINTS / Path(file).name
    if not path.exists():
        raise HTTPException(404, detail={"message": "No checkpoint by that name."})
    blob = torch.load(path, map_location="cpu", weights_only=False)
    graph = blob.get("graph") or {}
    if not graph.get("nodes"):
        raise HTTPException(404, detail={
            "message": "That checkpoint was saved without its design."})
    return graph


@app.delete("/api/checkpoints/{file}")
def delete_checkpoint(file: str):
    (T.CHECKPOINTS / Path(file).name).unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/datasets")
def list_datasets():
    files = sorted(p.name for p in T.UPLOADS.glob("*.csv"))
    return {"files": files}


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    name = Path(file.filename or "table.csv").name
    lower = name.lower()
    if not lower.endswith((".csv", ".txt")):
        raise HTTPException(400, detail={"message": "Upload a .csv table or a .txt corpus."})
    target = T.UPLOADS / name
    target.write_bytes(await file.read())
    try:
        info = T.inspect_text(name) if lower.endswith(".txt") else T.inspect_csv(name)
    except Exception as exc:  # noqa: BLE001
        target.unlink(missing_ok=True)
        raise HTTPException(400, detail={"message": f"That file would not parse: {exc}"})
    info["kind"] = "text" if lower.endswith(".txt") else "csv"
    return info


@app.get("/api/corpora")
def list_corpora():
    return {"files": sorted(p.name for p in T.UPLOADS.glob("*.txt"))}


@app.get("/api/corpora/{name}")
def describe_corpus(name: str):
    try:
        return T.inspect_text(name)
    except T.DataError as exc:
        raise HTTPException(404, detail={"message": str(exc)})


@app.get("/api/datasets/{name}")
def describe_dataset(name: str):
    try:
        return T.inspect_csv(name)
    except T.DataError as exc:
        raise HTTPException(404, detail={"message": str(exc)})


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    (T.UPLOADS / Path(name).name).unlink(missing_ok=True)
    return {"ok": True}


# --------------------------------------------------------------------------
# executions: every run recorded, browsable, tied to the design that made it
# --------------------------------------------------------------------------

@app.get("/api/runs")
def get_runs(design: Optional[str] = None):
    return {"runs": T.list_runs(design)}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        blob = T.read_run(run_id)
    except T.DataError as exc:
        raise HTTPException(404, detail={"message": str(exc)})
    # a live run's in-memory state is fresher than the last file write
    live = T.JOBS.get(run_id)
    if live:
        blob.update(live.snapshot())
    return blob


@app.get("/api/runs/{run_id}/graph")
def run_graph(run_id: str):
    """The exact design a run used, so a result can be reproduced."""
    try:
        blob = T.read_run(run_id)
    except T.DataError as exc:
        raise HTTPException(404, detail={"message": str(exc)})
    graph = blob.get("graph") or {}
    if not graph.get("nodes"):
        raise HTTPException(404, detail={
            "message": "That run was recorded without its design."})
    return graph


@app.delete("/api/runs/{run_id}")
def remove_run(run_id: str):
    T.delete_run(run_id)
    return {"ok": True}


@app.get("/api/train/{job_id}")
def get_job(job_id: str):
    job = T.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No such training run.")
    return job.snapshot()


@app.post("/api/train/{job_id}/stop")
def stop_job(job_id: str):
    job = T.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No such training run.")
    job.stop.set()
    return {"ok": True}


@app.get("/api/train/{job_id}/stream")
def stream_job(job_id: str):
    job = T.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No such training run.")

    def events():
        yield f"data: {json.dumps({'kind': 'hello', **job.snapshot()})}\n\n"
        idle = 0.0
        while True:
            try:
                event = job.events.get(timeout=1.0)
            except Exception:  # noqa: BLE001 - queue.Empty
                idle += 1.0
                yield ": keep-alive\n\n"
                if job.status in ("done", "error", "stopped") and idle > 2:
                    break
                continue
            idle = 0.0
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("kind") in ("finished", "error"):
                break

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------
# saved graphs
# --------------------------------------------------------------------------

def _safe(name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", name).strip("-")
    if not slug:
        raise HTTPException(400, "Give the design a name.")
    return slug[:64]


# Designs are versioned: every save writes a new file rather than overwriting,
# so a design you liked three edits ago is still there. A flat <name>.json from
# before versioning is treated as version 1 and left where it is.

def _folder(name: str) -> Path:
    return saved_dir() / _safe(name)


def _versions(name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    legacy = saved_dir() / f"{_safe(name)}.json"
    if legacy.exists():
        out.append({"version": 1, "path": legacy,
                    "saved_at": legacy.stat().st_mtime, "legacy": True})
    folder = _folder(name)
    if folder.is_dir():
        for path in folder.glob("v*.json"):
            try:
                number = int(path.stem[1:])
            except ValueError:
                continue
            out.append({"version": number, "path": path,
                        "saved_at": path.stat().st_mtime, "legacy": False})
    out.sort(key=lambda v: v["version"])
    return out


def _describe(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": entry["version"],
        "saved_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(entry["saved_at"])),
    }


@app.post("/api/examples/restore")
def restore_examples():
    """Copy the shipped designs into this workspace.

    Skips any name already there, so it can never overwrite your own work.
    """
    copied = auth.restore_examples(auth.workspace())
    return {"copied": copied,
            "message": (f"Added {copied} example design{'' if copied == 1 else 's'}."
                        if copied else
                        "They are all here already, under their original names.")}


@app.get("/api/graphs")
def list_graphs():
    # deliver anything newly shipped before listing, so an example added in a
    # later version reaches workspaces that already existed
    auth.seed(auth.workspace())
    names = {p.stem for p in saved_dir().glob("*.json")}
    names |= {p.name for p in saved_dir().iterdir() if p.is_dir()}
    items = []
    for name in sorted(names):
        versions = _versions(name)
        if not versions:
            continue
        newest = versions[-1]
        items.append({
            "name": name,
            "latest": newest["version"],
            "versions": len(versions),
            "saved_at": time.strftime("%Y-%m-%d %H:%M",
                                      time.localtime(newest["saved_at"])),
        })
    return {"graphs": items}


@app.get("/api/graphs/{name}/versions")
def graph_versions(name: str):
    versions = _versions(name)
    if not versions:
        raise HTTPException(404, detail={"message": "No design saved under that name."})
    return {"name": _safe(name), "versions": [_describe(v) for v in versions],
            "latest": versions[-1]["version"]}


@app.get("/api/graphs/{name}")
def load_graph(name: str, version: Optional[int] = None):
    versions = _versions(name)
    if not versions:
        raise HTTPException(404, detail={"message": "No design saved under that name."})
    chosen = versions[-1]
    if version is not None:
        match = [v for v in versions if v["version"] == version]
        if not match:
            raise HTTPException(404, detail={
                "message": f"{_safe(name)} has no version {version}."})
        chosen = match[0]
    graph = json.loads(chosen["path"].read_text())
    graph["_version"] = chosen["version"]
    graph["_latest"] = versions[-1]["version"]
    return graph


@app.put("/api/graphs/{name}")
def save_graph(name: str, body: GraphPayload):
    """Saving always creates the next version. Nothing is overwritten."""
    folder = _folder(name)
    folder.mkdir(parents=True, exist_ok=True)
    versions = _versions(name)
    number = (versions[-1]["version"] + 1) if versions else 1
    (folder / f"v{number}.json").write_text(json.dumps(body.graph, indent=2))
    return {"ok": True, "name": _safe(name), "version": number,
            "versions": len(versions) + 1}


@app.delete("/api/graphs/{name}")
def delete_graph(name: str, version: Optional[int] = None):
    """Without a version this removes the whole design and its history."""
    if version is None:
        (saved_dir() / f"{_safe(name)}.json").unlink(missing_ok=True)
        folder = _folder(name)
        if folder.is_dir():
            for path in folder.glob("*.json"):
                path.unlink()
            folder.rmdir()
        return {"ok": True, "removed": "all"}
    for entry in _versions(name):
        if entry["version"] == version:
            entry["path"].unlink(missing_ok=True)
            return {"ok": True, "removed": version}
    raise HTTPException(404, detail={"message": f"No version {version}."})


# --------------------------------------------------------------------------
# frontend
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# plug-in blocks
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# guided projects
# --------------------------------------------------------------------------

@app.get("/api/projects")
def get_projects():
    return {"projects": projectloader.catalog(),
            "categories": projectloader.categories(),
            "errors": projectloader.LAST_ERRORS}


@app.get("/api/projects/suggest")
def suggest_project(q: str = ""):
    return projectloader.suggest(q)


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    try:
        return projectloader.get(project_id)
    except KeyError:
        raise HTTPException(404, detail={"message": f"No project named {project_id}."})


@app.get("/api/recipes")
def get_recipes():
    return {"files": recipeloader.listing(), "errors": recipeloader.LAST_ERRORS,
            "recipes": recipes_sdk.catalog()}


@app.get("/api/recipes/new/{name}")
def new_recipe(name: str):
    return {"file": f"{name.lower()}.py", "source": recipeloader.scaffold(name)}


@app.get("/api/recipes/{file}")
def get_recipe(file: str):
    try:
        return {"file": file, "source": recipeloader.read(file)}
    except FileNotFoundError as exc:
        raise HTTPException(404, detail={"message": str(exc)})


@app.put("/api/recipes/{file}")
def put_recipe(file: str, body: BlockPayload):
    errors = recipeloader.write(file, body.source)
    mine = [e for e in errors if e["file"] == Path(file).name]
    return {"ok": not mine, "errors": errors, "mine": mine,
            "files": recipeloader.listing(), "recipes": recipes_sdk.catalog()}


@app.delete("/api/recipes/{file}")
def remove_recipe(file: str):
    errors = recipeloader.delete(file)
    return {"ok": True, "errors": errors, "files": recipeloader.listing(),
            "recipes": recipes_sdk.catalog()}


@app.post("/api/recipes/reload")
def reload_recipes():
    errors = recipeloader.load_all()
    return {"ok": not errors, "errors": errors, "files": recipeloader.listing(),
            "recipes": recipes_sdk.catalog()}


@app.get("/api/blocks")
def get_blocks():
    return {"files": blockloader.listing(), "errors": blockloader.LAST_ERRORS}


@app.get("/api/blocks/{file}")
def get_block(file: str):
    try:
        return {"file": file, "source": blockloader.read(file)}
    except FileNotFoundError as exc:
        raise HTTPException(404, detail={"message": str(exc)})


@app.put("/api/blocks/{file}")
def put_block(file: str, body: BlockPayload):
    errors = blockloader.write(file, body.source)
    mine = [e for e in errors if e["file"] == Path(file).name]
    return {"ok": not mine, "errors": errors, "mine": mine,
            "files": blockloader.listing(), "layers": catalog()}


@app.delete("/api/blocks/{file}")
def remove_block(file: str):
    errors = blockloader.delete(file)
    return {"ok": True, "errors": errors, "files": blockloader.listing(),
            "layers": catalog()}


@app.post("/api/blocks/reload")
def reload_blocks():
    errors = blockloader.load_all()
    return {"ok": not errors, "errors": errors, "files": blockloader.listing(),
            "layers": catalog()}


@app.get("/api/blocks/new/{name}")
def new_block(name: str):
    return {"file": f"{name.lower()}.py", "source": blockloader.scaffold(name)}


# --------------------------------------------------------------------------
# workspace preferences: where the panels sit and how big they are
# --------------------------------------------------------------------------

def prefs_file() -> Path:
    return auth.workspace() / "prefs.json"


@app.get("/api/prefs")
def get_prefs():
    if not prefs_file().exists():
        return {}
    try:
        return json.loads(prefs_file().read_text())
    except Exception:  # noqa: BLE001 - a corrupt file should not block the app
        return {}


@app.put("/api/prefs")
def put_prefs(body: Dict[str, Any]):
    """Stored on the server rather than in the browser, so the arrangement
    follows the project rather than the machine that opened it."""
    prefs_file().write_text(json.dumps(body, indent=1))
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    if FRONTEND_FILE.exists():
        return FileResponse(FRONTEND_FILE)
    looked = "\n".join(
        f"  {p}" for p in [
            HERE.parent / "frontend" / "index.html",
            HERE / "frontend" / "index.html",
            HERE / "index.html",
            HERE.parent / "index.html",
        ]
    )
    return HTMLResponse(
        "<pre style='font:13px ui-monospace;padding:28px;line-height:1.6'>"
        "index.html was not found.\n\n"
        f"main.py is running from:\n  {HERE}\n\n"
        f"Looked for the page at:\n{looked}\n\n"
        "Put index.html in one of those places and reload. The API is already "
        "up — /health and /api/catalog work.</pre>",
        status_code=503,
    )


# --------------------------------------------------------------------------
# accounts
# --------------------------------------------------------------------------

OPEN_PATHS = ("/api/auth/", "/health", "/favicon")


@app.middleware("http")
async def gate(request: Request, call_next):
    """Refuse API calls without a session, once accounts exist.

    With no accounts registered this does nothing, so an install that never
    wanted accounts is unaffected.
    """
    path = request.url.path
    if (auth.enabled() and path.startswith("/api/")
            and not path.startswith(OPEN_PATHS)
            and not auth.user_for(request.cookies.get(auth.SESSION_COOKIE))):
        return JSONResponse({"detail": {"message": "Sign in first.",
                                        "auth": True}}, status_code=401)
    return await call_next(request)


class Credentials(BaseModel):
    name: str
    password: str


class PasswordChange(BaseModel):
    old: str
    new: str


@app.get("/api/auth/me")
def auth_me():
    return auth.whoami()


@app.post("/api/auth/register")
def auth_register(body: Credentials, response: Response):
    first = not auth.enabled()
    try:
        auth.register(body.name, body.password)
    except ValueError as exc:
        raise HTTPException(400, detail={"message": str(exc)})
    token = auth.open_session(body.name.strip().lower())
    response.set_cookie(auth.SESSION_COOKIE, token, httponly=True,
                        samesite="lax", max_age=auth.SESSION_DAYS * 86400)
    auth.set_current(body.name.strip().lower())
    return {"ok": True, "user": body.name.strip().lower(), "first": first,
            "workspace": str(auth.workspace())}


@app.post("/api/auth/login")
def auth_login(body: Credentials, response: Response):
    name = (body.name or "").strip().lower()
    if not auth.check(name, body.password):
        raise HTTPException(401, detail={"message": "That name and password do "
                                                    "not match an account."})
    token = auth.open_session(name)
    response.set_cookie(auth.SESSION_COOKIE, token, httponly=True,
                        samesite="lax", max_age=auth.SESSION_DAYS * 86400)
    return {"ok": True, "user": name}


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response):
    auth.close_session(request.cookies.get(auth.SESSION_COOKIE) or "")
    response.delete_cookie(auth.SESSION_COOKIE)
    return {"ok": True}


@app.post("/api/auth/password")
def auth_password(body: PasswordChange):
    name = auth.current()
    if not name:
        raise HTTPException(401, detail={"message": "Sign in first."})
    try:
        auth.change_password(name, body.old, body.new)
    except ValueError as exc:
        raise HTTPException(400, detail={"message": str(exc)})
    return {"ok": True, "message": "Password changed. Other sessions were signed out."}


@app.get("/health")
def health():
    return {"ok": True, "version": __version__}
