"""Run one batch through the model and record what each layer did.

A workflow engine can show a task going green because tasks execute one at a
time and report back. A forward pass is the same shape: modules run in order,
each taking a measurable amount of time and producing a definite tensor. So this
is not an animation of what probably happens — it is what happened, measured
with hooks on the model the canvas generated.

What comes back per layer: when it started, how long it took, the shape it
produced, how much memory that tensor occupies, and whether it ran at all. A
layer that never fires is as interesting as a slow one — it usually means the
graph is not wired the way it looks.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import codegen
import graph as G


def run_trace(payload: Dict[str, Any], batch: int = 2,
              device: str = "cpu") -> Dict[str, Any]:
    """Execute the design once and return a record of every layer."""
    import torch

    import train as T

    g = G.parse(payload)
    report = G.analyze(g)
    if not report["ok"]:
        raise ValueError(report["errors"][0] if report["errors"]
                         else "The design does not resolve.")

    node_code: Dict[str, Any] = {}
    source = codegen.to_pytorch(g, report, node_code)
    model = T.build_model(source, codegen.model_class_name(g))
    model.eval()

    # the generated model names each module after the node it came from, which
    # is what lets a hook report back to the right box on the canvas
    var_to_node = {entry["var"]: nid for nid, entry in node_code.items()
                   if entry.get("var")}
    module_to_node = {}
    for name, module in model.named_modules():
        if name in var_to_node:
            module_to_node[module] = var_to_node[name]

    records: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    started = [0.0]

    def before(module, args):
        records.setdefault(module_to_node[module], {})["began"] = time.perf_counter()

    def after(module, args, output):
        nid = module_to_node[module]
        entry = records.setdefault(nid, {})
        finished = time.perf_counter()
        entry["ms"] = (finished - entry.get("began", finished)) * 1000
        entry["at"] = (entry.get("began", finished) - started[0]) * 1000
        tensor = output[0] if isinstance(output, (tuple, list)) and output else output
        if hasattr(tensor, "shape"):
            entry["shape"] = list(tensor.shape)
            entry["bytes"] = int(tensor.numel() * tensor.element_size())
            entry["dtype"] = str(tensor.dtype).replace("torch.", "")
        if nid not in order:
            order.append(nid)

    # measured after a warm-up, because the first call through a layer includes
    # one-off kernel selection and allocation. Unwarmed, a small convolution
    # reports several hundred milliseconds and looks like the bottleneck it is
    # not.
    handles = []

    ids = codegen.input_order(g, report)
    nodes = g.by_id()
    inputs = []
    for nid in ids:
        shape = report["nodes"][nid]["out_shape"]
        params = G.resolved_params(nodes[nid])
        if str(params.get("dtype", "float")).startswith("long"):
            vocab = 1000
            for other in report["order"]:
                if nodes[other].type == "Embedding":
                    vocab = int(G.resolved_params(nodes[other]).get("vocab", 1000))
                    break
            inputs.append(torch.randint(0, vocab, (batch, *map(int, shape))))
        else:
            inputs.append(torch.randn(batch, *map(int, shape)))

    try:
        with torch.no_grad():
            model(*inputs)                      # warm up, unmeasured
        for module, nid in module_to_node.items():
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after))
        with torch.no_grad():
            started[0] = time.perf_counter()
            output = model(*inputs)
            total = (time.perf_counter() - started[0]) * 1000
    except Exception as exc:  # noqa: BLE001 - a failed run is a result too
        for handle in handles:
            handle.remove()
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "layers": _layers(g, report, records, order, node_code),
                "order": order}
    finally:
        for handle in handles:
            handle.remove()

    outputs = output if isinstance(output, (tuple, list)) else [output]
    return {
        "ok": True,
        "total_ms": round(total, 3),
        "batch": batch,
        "device": device,
        "layers": _layers(g, report, records, order, node_code),
        "order": order,
        "warmed": True,
        "outputs": [{"shape": list(t.shape),
                     "dtype": str(t.dtype).replace("torch.", "")}
                    for t in outputs if hasattr(t, "shape")],
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable": int(sum(p.numel() for p in model.parameters()
                             if p.requires_grad)),
    }


def _layers(g, report, records, order, node_code) -> List[Dict[str, Any]]:
    """One row per layer, in the order the graph runs them.

    Layers with no module of their own — an Add, a Flatten — never fire a hook.
    They are reported as having run without a time rather than being left out,
    because a missing row reads as a layer that failed.
    """
    nodes = g.by_id()
    rows = []
    for index, nid in enumerate(report["order"]):
        node = nodes[nid]
        entry = records.get(nid, {})
        code = node_code.get(nid) or {}
        measured = "ms" in entry
        rows.append({
            "id": nid,
            "seq": index + 1,
            "name": node.label or node.type,
            "type": node.type,
            "ms": round(entry.get("ms", 0.0), 3) if measured else None,
            "at": round(entry.get("at", 0.0), 3) if measured else None,
            "shape": entry.get("shape") or (
                [None] + list(report["nodes"].get(nid, {}).get("out_shape") or [])),
            "bytes": entry.get("bytes"),
            "dtype": entry.get("dtype"),
            "parameters": report["nodes"].get(nid, {}).get("learnables", 0),
            "measured": measured,
            "var": code.get("var"),
        })
    return rows
