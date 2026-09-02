"""A workbook of sheets.

One graph on one canvas stops being readable somewhere around a hundred layers,
and a project that spans several files has no reason to become a single sheet.
So a design is a workbook: several named sheets, each an ordinary graph, and a
`Subgraph` node on one sheet that stands for another.

The reference is not a drawing convenience. A referenced sheet is generated as
its own `nn.Module` class and the parent instantiates it, which is how anyone
would write it by hand — a class per file. Shapes propagate through the
reference, so the parent sheet knows what comes back out.

Sheets that reference each other in a circle are refused, because the generated
code would not terminate.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import codegen
import graph as G

MAIN = "main"


def sheets_of(book: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {sheet["name"]: sheet for sheet in book.get("sheets", [])}


def is_workbook(payload: Dict[str, Any]) -> bool:
    return isinstance(payload.get("sheets"), list)


def wrap(graph: Dict[str, Any]) -> Dict[str, Any]:
    """A plain graph is a workbook of one sheet, so both can travel one path."""
    if is_workbook(graph):
        return graph
    return {"name": graph.get("name") or "Design",
            "main": MAIN,
            "sheets": [{"name": MAIN, "nodes": graph.get("nodes", []),
                        "edges": graph.get("edges", [])}]}


def sheet_graph(book: Dict[str, Any], name: str) -> Dict[str, Any]:
    sheet = sheets_of(book).get(name)
    if sheet is None:
        raise KeyError(name)
    return {"name": name, "nodes": sheet["nodes"], "edges": sheet["edges"]}


# --------------------------------------------------------------------------
# references between sheets
# --------------------------------------------------------------------------

def references(book: Dict[str, Any], name: str) -> List[str]:
    sheet = sheets_of(book).get(name)
    if not sheet:
        return []
    return [n["params"].get("sheet") for n in sheet["nodes"]
            if n["type"] == "Subgraph" and n["params"].get("sheet")]


def order_sheets(book: Dict[str, Any]) -> Tuple[List[str], Optional[str]]:
    """Sheets in dependency order, or the cycle that stops that being possible."""
    names = [s["name"] for s in book.get("sheets", [])]
    state: Dict[str, int] = {}
    ordered: List[str] = []
    trouble: Optional[str] = None

    def walk(name: str, trail: List[str]):
        nonlocal trouble
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            trouble = " → ".join(trail + [name])
            return
        state[name] = 1
        for child in references(book, name):
            if child in names:
                walk(child, trail + [name])
        state[name] = 2
        ordered.append(name)

    for name in names:
        walk(name, [])
    return ordered, trouble


def analyze(book: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze every sheet, resolving references in dependency order.

    A Subgraph node cannot know its own output shape — the sheet it points at
    does. So each sheet is analyzed after the ones it depends on, and the
    resolved shape is written onto the node for its own inference to read.
    """
    ordered, cycle = order_sheets(book)
    reports: Dict[str, Any] = {}

    if cycle:
        for sheet in book.get("sheets", []):
            reports[sheet["name"]] = {
                "ok": False, "nodes": {}, "edges": {}, "order": [],
                "total_learnables": 0, "approximate": False,
                "errors": [f"These sheets reference each other in a circle: {cycle}. "
                           f"The generated code would not terminate."],
            }
        return {"sheets": reports, "cycle": cycle,
                "total_learnables": 0, "ok": False}

    signatures: Dict[str, Dict[str, Any]] = {}
    approximate = False

    for name in ordered:
        payload = sheet_graph(book, name)
        # tell each Subgraph node what the sheet it points at produces
        for node in payload["nodes"]:
            if node["type"] != "Subgraph":
                continue
            target = node["params"].get("sheet")
            signature = signatures.get(target)
            node["params"]["_out"] = signature["out"] if signature else None
            node["params"]["_learnables"] = signature["learnables"] if signature else 0

        g = G.parse(payload)
        report = G.analyze(g)
        reports[name] = report
        approximate = approximate or report.get("approximate", False)

        nodes = g.by_id()
        outputs = [nodes[i] for i in report.get("order", [])
                   if nodes[i].type == "Output"]
        out_shape = (report["nodes"].get(outputs[0].id, {}).get("out_shape")
                     if outputs else None)
        signatures[name] = {"out": out_shape,
                            "learnables": report.get("total_learnables", 0)}

    main = book.get("main") or (ordered[-1] if ordered else MAIN)
    # the main sheet's total already counts every sheet it references, through
    # each Subgraph node — summing the sheets as well would count them twice,
    # and a sheet nothing references is not part of the model at all
    total = reports.get(main, {}).get("total_learnables", 0)
    return {
        "sheets": reports,
        "cycle": None,
        "main": main,
        "ok": all(r.get("ok") for r in reports.values()),
        "total_learnables": total,
        "approximate": approximate,
        "signatures": signatures,
    }


# --------------------------------------------------------------------------
# code for a whole workbook
# --------------------------------------------------------------------------

def to_pytorch(book: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """Every referenced sheet becomes its own class, the main sheet the model."""
    ordered, cycle = order_sheets(book)
    if cycle:
        return f"# These sheets reference each other in a circle: {cycle}\n"

    main = analysis.get("main") or book.get("main") or MAIN

    # only what the main sheet reaches becomes code — an unreferenced sheet in
    # the file would be a class nothing constructs
    reachable = set()
    frontier = [main]
    while frontier:
        current = frontier.pop()
        if current in reachable:
            continue
        reachable.add(current)
        frontier.extend(references(book, current))

    pieces: List[str] = []
    seen_header = False

    for name in ordered:
        if name == main or name not in reachable:
            continue
        report = analysis["sheets"].get(name)
        if not report or not report.get("ok"):
            pieces.append(f"# sheet {name!r} does not resolve, so it is not generated\n")
            continue
        payload = sheet_graph(book, name)
        source = codegen.to_pytorch(G.parse(payload), report)
        body = _class_only(source)
        if not seen_header:
            pieces.append(_header_of(source))
            seen_header = True
        pieces.append(body)

    report = analysis["sheets"].get(main)
    if not report or not report.get("ok"):
        return "\n\n".join(pieces + [f"# the main sheet {main!r} does not resolve yet\n"])
    main_source = codegen.to_pytorch(G.parse(sheet_graph(book, main)), report)
    if not seen_header:
        return main_source
    return "\n\n".join(pieces + [_strip_header(main_source)])


def _header_of(source: str) -> str:
    lines = source.splitlines()
    end = 0
    for i, line in enumerate(lines):
        if line.startswith("class ") or line.startswith("def "):
            end = i
            break
    return "\n".join(lines[:end]).rstrip()


def _strip_header(source: str) -> str:
    lines = source.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("class ") or line.startswith("def "):
            return "\n".join(lines[i:])
    return source


def _class_only(source: str) -> str:
    """The class definitions from a generated file, without its imports or main."""
    body = _strip_header(source)
    marker = 'if __name__ == "__main__":'
    if marker in body:
        body = body[:body.index(marker)]
    return body.rstrip() + "\n"
