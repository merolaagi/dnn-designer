"""Deep Network Designer — server.

Run it:  uvicorn main:app --reload --port 8770
Then open http://127.0.0.1:8770
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (Depends, FastAPI, File, HTTPException, Request,
                     Response, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

import agents
import assistant
import auth
import blockloader
import codegen
import importer
import needs
import projectloader
import recipeloader
import recipes_sdk
import graph as G
import workbook
import train as T
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


@app.get("/api/assistant/review")
def assistant_review(name: str = ""):
    return {"suggestions": assistant.SUGGESTIONS, "help": assistant.HELP,
            "model": bool(os.environ.get("ASSISTANT_API_KEY"))}


class TestLayerPayload(BaseModel):
    graph: Dict[str, Any]
    node: str


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
        source = codegen.to_pytorch(g, report)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, detail={"message": f"Code generation failed: {exc}"})

    job = T.start(source, cfg, in_shapes, in_ids, out_shape, tasks,
                  codegen.model_class_name(g))
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


def _finish_import(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze what came in so the response can say what needs attention."""
    notes = graph.pop("_notes", [])
    g = G.parse(graph)
    report = G.analyze(g)
    problems = [v["error"] for v in report["nodes"].values() if v["error"]]
    return {
        "graph": graph,
        "notes": notes,
        "ok": report["ok"],
        "nodes": len(graph["nodes"]),
        "problems": (report["errors"] + problems)[:6],
        "problem_count": len(report["errors"]) + len(problems),
        "learnables": report["total_learnables"],
    }


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
        raise HTTPException(400, detail={"message": f"{type(exc).__name__}: {exc}"})
    return _finish_import(graph)


@app.post("/api/scan-folder")
def scan_folder(body: ScanPayload):
    """List every nn.Module in a folder. Reads syntax trees only — nothing runs
    until a class is actually picked for import."""
    try:
        return importer.scan_folder(body.root)
    except importer.ImportError_ as exc:
        raise HTTPException(400, detail={"message": str(exc)})


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
        parts = set(path.parts)
        if parts & {".venv", "venv", "site-packages", "__pycache__", "build",
                    "node_modules", ".git", ".mypy_cache"}:
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
                str(pick.get("arguments") or ""))
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
