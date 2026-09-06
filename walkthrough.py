"""What happens to the data, step by step, with the actual numbers.

The Run tab answers "what did that cost". This answers "what is going on" — for
each layer in turn: what arrives, what the layer does to it mathematically, what
leaves, and what the values look like once it has.

The statistics are measured, not described. A ReLU is not said to "zero the
negatives"; it is run, and the panel reports that 47% of the tensor is now
exactly zero. A softmax is not said to "produce a distribution"; the row is
summed and shown to be 1.

The closing step reads the output the way the task means it. A classification
head's logits become the top few classes with their probabilities; a language
model's become the tokens it would choose next; a regression head's stay
numbers. Which one is used comes from the Output layer's task, so this works for
whatever is on the canvas rather than for one architecture.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import codegen
import graph as G
import mathbook


def walkthrough(payload: Dict[str, Any], batch: int = 2,
                top_k: int = 5) -> Dict[str, Any]:
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

    var_to_node = {entry["var"]: nid for nid, entry in node_code.items()
                   if entry.get("var")}
    module_to_node = {module: var_to_node[name]
                      for name, module in model.named_modules()
                      if name in var_to_node}

    seen: Dict[str, Dict[str, Any]] = {}

    def capture(module, args, output):
        tensor = output[0] if isinstance(output, (tuple, list)) and output else output
        if hasattr(tensor, "float"):
            seen[module_to_node[module]] = _describe(tensor)

    handles = [m.register_forward_hook(capture) for m in module_to_node]

    ids = codegen.input_order(g, report)
    nodes = g.by_id()
    inputs, input_notes = [], []
    for nid in ids:
        shape = report["nodes"][nid]["out_shape"]
        params = G.resolved_params(nodes[nid])
        if str(params.get("dtype", "float")).startswith("long"):
            vocab = 1000
            for other in report["order"]:
                if nodes[other].type == "Embedding":
                    vocab = int(G.resolved_params(nodes[other]).get("vocab", 1000))
                    break
            tensor = torch.randint(0, vocab, (batch, *map(int, shape)))
            input_notes.append(f"{_count(shape)} indices, each below {vocab:,}")
        else:
            tensor = torch.randn(batch, *map(int, shape))
            input_notes.append(f"{_count(shape)} numbers drawn from a normal")
        inputs.append(tensor)
        seen[nid] = _describe(tensor)

    try:
        with torch.no_grad():
            output = model(*inputs)
    except Exception as exc:  # noqa: BLE001
        for handle in handles:
            handle.remove()
        raise ValueError(f"The pass failed: {type(exc).__name__}: {exc}")
    finally:
        for handle in handles:
            handle.remove()

    steps = []
    inc = G.incoming_map(g)
    for index, nid in enumerate(report["order"]):
        node = nodes[nid]
        in_shapes = [report["nodes"].get(e.source, {}).get("out_shape")
                     for e in inc[nid]]
        out_shape = report["nodes"].get(nid, {}).get("out_shape")
        entry = mathbook.explain(node.type, G.resolved_params(node),
                                 in_shapes, out_shape)
        steps.append({
            "seq": index + 1,
            "id": nid,
            "name": node.label or node.type,
            "type": node.type,
            "title": entry.get("title") or node.type,
            "equation": entry.get("equation", ""),
            "family": entry.get("family", "unknown"),
            "missing": entry.get("missing"),
            "in_shapes": in_shapes,
            "out_shape": out_shape,
            "parameters": report["nodes"].get(nid, {}).get("learnables", 0),
            "values": seen.get(nid),
            # a layer with no module of its own fires no hook; say so rather
            # than leaving a blank that reads as "nothing happened"
            "unwatched": nid not in seen,
            "note": _note(node, entry, in_shapes, out_shape, seen.get(nid)),
        })

    outputs = output if isinstance(output, (tuple, list)) else [output]
    return {
        "ok": True,
        "batch": batch,
        "name": payload.get("name") or "this network",
        "opening": _opening(g, report, input_notes),
        "steps": steps,
        "closing": _closing(g, report, outputs, top_k),
    }


def _count(shape) -> str:
    total = 1
    for d in shape or []:
        total *= int(d)
    return f"{total:,}"


def _describe(tensor) -> Dict[str, Any]:
    """A few honest facts about a tensor: enough to see what changed."""
    import torch

    raw = tensor.detach()
    flat = raw.float().flatten()
    if not flat.numel():
        return {}

    # the average of a set of token ids means nothing; their range does
    if not raw.dtype.is_floating_point:
        return {
            "shape": list(tensor.shape),
            "count": int(flat.numel()),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "indices": True,
            "low": int(flat.min()),
            "high": int(flat.max()),
            "distinct": int(raw.unique().numel()),
        }

    zeros = float((flat == 0).float().mean()) * 100
    return {
        "shape": list(tensor.shape),
        "mean": round(float(flat.mean()), 4),
        "std": round(float(flat.std()) if flat.numel() > 1 else 0.0, 4),
        "min": round(float(flat.min()), 4),
        "max": round(float(flat.max()), 4),
        "zeros": round(zeros, 1),
        "count": int(flat.numel()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
    }


def _note(node, entry, in_shapes, out_shape, values) -> str:
    """One sentence about what this step did to the data that arrived."""
    kind = node.type
    if kind == "Input":
        if values and values.get("indices"):
            return (f"{values['count']} indices arrive, {values['distinct']} of them "
                    f"distinct, ranging from {values['low']} to {values['high']}. "
                    f"They are positions in a vocabulary, not quantities.")
        return f"The data enters as {_shape(out_shape)} per example."
    if kind == "Output":
        return "Nothing happens here; it marks what the network is asked for."

    before = in_shapes[0] if in_shapes and in_shapes[0] else None
    if values and values.get("zeros", 0) > 5 and kind == "Activation":
        return (f"{values['zeros']:.0f}% of the values are now exactly zero, "
                f"which is the negatives being cut away.")
    if kind in ("BatchNorm2d", "BatchNorm1d", "LayerNorm", "RMSNorm") and values:
        return (f"The values now sit around {values['mean']:+.3f} with a spread "
                f"of {values['std']:.3f}, which is what the normalization is for.")
    if kind == "Flatten" and before:
        return (f"The same {_count(before)} numbers, read as one row instead of "
                f"{_shape(before)}.")
    if kind in ("MaxPool2d", "AvgPool2d") and before and out_shape:
        kept = _count(out_shape)
        return f"{_count(before)} values become {kept}: the map is coarser now."
    if kind == "GlobalAvgPool" and before:
        return (f"Each of the {before[0]} channels collapses to a single number, "
                f"so position is gone and only presence remains.")
    if kind in ("Linear",) and before and out_shape:
        return (f"Every one of the {_count(before)} inputs reaches every one of "
                f"the {_count(out_shape)} outputs.")
    if kind in ("Conv2d", "Conv1d") and out_shape:
        return (f"The same filters are applied at every position, producing "
                f"{out_shape[0]} feature maps.")
    if kind == "Attention":
        return "Every position reads every other, weighted by how well they match."
    if kind == "Embedding" and out_shape:
        return f"Each index is replaced by its row: {out_shape[-1]} numbers."
    if kind == "Dropout":
        return "In training some values would be zeroed here; in evaluation this does nothing."
    if node.type == "Subgraph":
        return f"The whole of sheet \u201c{node.params.get('sheet','')}\u201d runs here."
    if entry.get("missing"):
        return ""
    return f"{_shape(before)} becomes {_shape(out_shape)}."


def _shape(shape) -> str:
    if not shape:
        return "\u2014"
    return "\u00d7".join(str(int(d)) for d in shape)


def _opening(g, report, input_notes) -> Dict[str, Any]:
    nodes = g.by_id()
    ins = [nid for nid in report["order"] if nodes[nid].type == "Input"]
    return {
        "title": "What goes in",
        "text": "; ".join(input_notes) or "nothing",
        "shapes": [report["nodes"][nid]["out_shape"] for nid in ins],
    }


def _closing(g, report, outputs, top_k) -> Dict[str, Any]:
    """Read the final tensor the way the Output layer says it should be read."""
    import torch

    nodes = g.by_id()
    outs = [nid for nid in report["order"] if nodes[nid].type == "Output"]
    task = (G.resolved_params(nodes[outs[0]]).get("task", "classification")
            if outs else "classification")
    tensor = outputs[0]
    if not hasattr(tensor, "shape"):
        return {"task": task, "text": "The network produced no tensor."}

    row = tensor[0]
    # a sequence model scores every position; the last one is the prediction
    sequence = row.dim() > 1
    if sequence:
        row = row[-1]

    if task in ("classification", "binary"):
        probs = torch.softmax(row.float(), dim=-1)
        k = min(top_k, probs.numel())
        best = torch.topk(probs, k)
        picks = [{"index": int(i), "p": round(float(p), 4)}
                 for p, i in zip(best.values, best.indices)]
        total = float(probs.sum())
        return {
            "task": task,
            "title": "What comes out",
            "text": (f"{row.numel():,} scores"
                     + (" for the last position in the sequence" if sequence else "")
                     + f", turned into probabilities by softmax. They sum to "
                       f"{total:.3f}."),
            "picks": picks,
            "sequence": sequence,
            "note": ("For a language model each index is a token, so this is the "
                     "next token it would choose."
                     if sequence else
                     "Each index is a class, so the first row is its answer."),
        }
    values = row.flatten()[:top_k]
    return {
        "task": task,
        "title": "What comes out",
        "text": f"{row.numel():,} number{'' if row.numel() == 1 else 's'}, "
                f"used as they are.",
        "picks": [{"index": i, "p": round(float(v), 4)}
                  for i, v in enumerate(values)],
        "sequence": sequence,
        "note": "A regression head's output is the prediction itself, not a score.",
    }
