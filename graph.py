"""Graph analysis: order the nodes, push shapes through them, report what breaks.

Analysis never raises on a bad user graph. It returns a report with per-node
errors so the canvas can highlight exactly which layer is unhappy and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from layers import REGISTRY, ShapeError

Shape = List[int]


@dataclass
class Node:
    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    label: str = ""
    x: float = 0.0
    y: float = 0.0


@dataclass
class Edge:
    id: str
    source: str
    target: str
    port: int = 0


@dataclass
class Graph:
    name: str
    nodes: List[Node]
    edges: List[Edge]

    def by_id(self) -> Dict[str, Node]:
        return {n.id: n for n in self.nodes}


def parse(payload: Dict[str, Any]) -> Graph:
    nodes = [
        Node(
            id=str(n["id"]),
            type=str(n["type"]),
            params=dict(n.get("params") or {}),
            label=str(n.get("label") or ""),
            x=float(n.get("x", 0)),
            y=float(n.get("y", 0)),
        )
        for n in payload.get("nodes", [])
    ]
    edges = [
        Edge(
            id=str(e.get("id") or f"{e['source']}->{e['target']}:{e.get('port', 0)}"),
            source=str(e["source"]),
            target=str(e["target"]),
            port=int(e.get("port", 0)),
        )
        for e in payload.get("edges", [])
    ]
    return Graph(name=str(payload.get("name") or "Net"), nodes=nodes, edges=edges)


# --------------------------------------------------------------------------
# learnable parameter estimates
# --------------------------------------------------------------------------

def _learnables(node: Node, in_shapes: List[Shape], out_shape: Shape) -> int:
    spec = REGISTRY.get(node.type)
    if spec is not None and spec.learnables is not None:
        try:
            params = {**spec.defaults(), **node.params}
            return int(spec.learnables(params, in_shapes, out_shape))
        except Exception:  # noqa: BLE001 - a bad estimate must not break analysis
            return 0
    p, t = node.params, node.type
    try:
        if t == "Linear":
            n = in_shapes[0][-1] * int(p["units"])
            return n + (int(p["units"]) if p.get("bias", True) else 0)
        if t == "Conv2d":
            k = int(p["kernel"])
            n = (in_shapes[0][0] // int(p.get("groups", 1))) * k * k * int(p["filters"])
            return n + (int(p["filters"]) if p.get("bias", True) else 0)
        if t == "Conv1d":
            return in_shapes[0][1] * int(p["kernel"]) * int(p["filters"]) + int(p["filters"])
        if t == "ConvTranspose2d":
            k = int(p["kernel"])
            return in_shapes[0][0] * k * k * int(p["filters"]) + int(p["filters"])
        if t == "SeparableConv2d":
            c, k, f = in_shapes[0][0], int(p["kernel"]), int(p["filters"])
            return c * k * k + c * f + f
        if t in ("BatchNorm2d", "BatchNorm1d", "GroupNorm"):
            return 2 * in_shapes[0][0]
        if t == "LayerNorm":
            return 2 * in_shapes[0][-1]
        if t == "Embedding":
            return int(p["vocab"]) * int(p["dim"])
        if t in ("LSTM", "GRU", "SimpleRNN"):
            gates = {"LSTM": 4, "GRU": 3, "SimpleRNN": 1}[t]
            units, layers_n = int(p["units"]), int(p.get("num_layers", 1))
            dirs = 2 if p.get("bidirectional") else 1
            total, in_dim = 0, in_shapes[0][1]
            for layer_i in range(layers_n):
                d = in_dim if layer_i == 0 else units * dirs
                total += dirs * gates * ((d + units) * units + 2 * units)
            return total
        if t == "SelfAttention":
            d = in_shapes[0][1]
            return 4 * d * d + 4 * d
        if t == "TransformerEncoder":
            d, ff, depth = in_shapes[0][1], int(p["ff_dim"]), int(p["depth"])
            return depth * (4 * d * d + 4 * d + 2 * d * ff + ff + d + 4 * d)
    except Exception:  # noqa: BLE001 - estimates are best effort only
        return 0
    return 0


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def analyze(g: Graph) -> Dict[str, Any]:
    nodes = g.by_id()
    report: Dict[str, Any] = {
        "ok": False,
        "nodes": {},
        "edges": {},
        "errors": [],
        "order": [],
        "total_learnables": 0,
        "approximate": False,
    }
    errors: List[str] = report["errors"]

    if not g.nodes:
        errors.append("The canvas is empty. Drag an Input layer in to start.")
        return report

    # incoming edges per node, ordered by port
    incoming: Dict[str, List[Edge]] = {n.id: [] for n in g.nodes}
    outdeg: Dict[str, int] = {n.id: 0 for n in g.nodes}
    for e in g.edges:
        if e.source not in nodes or e.target not in nodes:
            errors.append(f"Edge {e.id} points at a layer that is no longer on the canvas.")
            continue
        incoming[e.target].append(e)
        outdeg[e.source] += 1
    for eid in incoming:
        incoming[eid].sort(key=lambda e: e.port)

    # structural checks
    inputs = [n for n in g.nodes if n.type == "Input"]
    outputs = [n for n in g.nodes if n.type == "Output"]
    if not inputs:
        errors.append("No Input layer. Every network needs at least one.")
    if not outputs:
        errors.append("No Output layer. Add one so training knows which tensor carries the loss.")

    for n in g.nodes:
        spec = REGISTRY.get(n.type)
        if spec is None:
            errors.append(f"Unknown layer type '{n.type}'.")
            continue
        for e in incoming[n.id]:
            src = nodes.get(e.source)
            src_spec = REGISTRY.get(src.type) if src else None
            if src and src.type == "Output" and spec.kind != "runtime":
                errors.append(
                    f"{n.label or n.type} draws from an Output. Output marks where the "
                    f"loss is taken; only runtime blocks can attach past it."
                )
            if src_spec and src_spec.kind == "runtime":
                errors.append(
                    f"{src.label or src.type} is a runtime block and produces no "
                    f"activation, so nothing can read from it."
                )
        got = len(incoming[n.id])
        if spec.n_inputs == 0 and got:
            errors.append(f"{n.label or n.type} is an Input and cannot take an incoming connection.")
        elif spec.n_inputs == 1 and got != 1:
            errors.append(
                f"{n.label or n.type} needs exactly one input, has {got}."
                if got else f"{n.label or n.type} has no input connected."
            )
        elif spec.n_inputs == -1 and got < 2:
            errors.append(f"{n.label or n.type} needs at least two inputs, has {got}.")
        elif spec.n_inputs == -2 and got < 1:
            errors.append(f"{n.label or n.type} has no input connected.")

    # topological order (Kahn)
    indeg = {n.id: len(incoming[n.id]) for n in g.nodes}
    queue = [n.id for n in g.nodes if indeg[n.id] == 0]
    order: List[str] = []
    while queue:
        queue.sort()
        nid = queue.pop(0)
        order.append(nid)
        for e in g.edges:
            if e.source == nid and e.target in indeg:
                indeg[e.target] -= 1
                if indeg[e.target] == 0:
                    queue.append(e.target)
    if len(order) != len(g.nodes):
        stuck = sorted(set(nodes) - set(order))
        names = ", ".join(nodes[i].label or nodes[i].type for i in stuck[:4])
        errors.append(f"These layers form a loop: {names}. Data has to flow one way.")
        report["order"] = order
        return report

    report["order"] = order

    # shape propagation
    shapes: Dict[str, Shape] = {}
    total = 0
    for nid in order:
        n = nodes[nid]
        spec = REGISTRY.get(n.type)
        entry: Dict[str, Any] = {"out_shape": None, "error": None, "learnables": 0}
        report["nodes"][nid] = entry
        if spec is None:
            entry["error"] = f"Unknown layer type '{n.type}'."
            continue

        if spec.kind == "runtime":
            # Search, solvers and self-play wrap the model; they have no activation.
            entry["out_shape"] = None
            entry["runtime"] = True
            continue

        params = {**spec.defaults(), **n.params}
        in_shapes = []
        missing = False
        for e in incoming[nid]:
            if e.source not in shapes:
                missing = True
                break
            in_shapes.append(shapes[e.source])
        if missing:
            entry["error"] = "Waiting on an upstream layer that could not be resolved."
            continue
        if spec.n_inputs != 0 and not in_shapes:
            entry["error"] = "Nothing connected to this layer's input."
            continue

        try:
            out = spec.infer(params, in_shapes)
        except ShapeError as exc:
            entry["error"] = str(exc)
            continue
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"could not resolve shape: {exc}"
            continue

        shapes[nid] = out
        entry["out_shape"] = out
        entry["learnables"] = _learnables(n, in_shapes, out)
        entry["frozen"] = bool(n.params.get("_frozen"))
        if entry["frozen"]:
            entry["frozen_learnables"] = entry["learnables"]
            entry["learnables"] = 0        # frozen weights are not being trained
        entry["approx"] = bool(spec.learnables_approx and entry["learnables"])
        if entry["approx"]:
            report["approximate"] = True
        total += entry["learnables"]

    for e in g.edges:
        if e.source in shapes:
            report["edges"][e.id] = {"shape": shapes[e.source]}

    report["total_learnables"] = total
    report["ok"] = not errors and all(v["error"] is None for v in report["nodes"].values())
    return report


def resolved_params(node: Node) -> Dict[str, Any]:
    spec = REGISTRY.get(node.type)
    if spec is None:
        return dict(node.params)
    return {**spec.defaults(), **node.params}


def incoming_map(g: Graph) -> Dict[str, List[Edge]]:
    inc: Dict[str, List[Edge]] = {n.id: [] for n in g.nodes}
    for e in g.edges:
        if e.target in inc:
            inc[e.target].append(e)
    for k in inc:
        inc[k].sort(key=lambda e: e.port)
    return inc
