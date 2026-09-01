"""The assistant.

Deliberately not a language model. It parses a small command grammar over the
graph and runs a set of architecture checks, both of which are deterministic,
inspectable and correct — where a small model would be fluent and wrong.

It does two things:

  commands   "add dropout after conv2d", "set filters to 64", "freeze backbone"
             return a modified graph, which the canvas applies so you watch the
             change happen.

  review     reads the graph and reports what is actually wrong with it. Two
             Linear layers with nothing between them collapse into one; a
             Flatten of a large feature map hands a Linear tens of millions of
             parameters; a convolution whose activation sees unnormalized scale.
             These are findable by inspection, so they are found by inspection.

If ASSISTANT_API_KEY is set in the environment, free-text that matches no
command is forwarded to an Anthropic-compatible endpoint along with a summary of
the graph. Without it, the assistant says plainly that it did not understand
rather than inventing an answer.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import layers
from graph import Graph, analyze, incoming_map, parse, resolved_params

MAX_SUGGESTIONS = 5


# --------------------------------------------------------------------------
# finding the layer someone means
# --------------------------------------------------------------------------

def _candidates(g: Graph) -> List[Tuple[str, Any]]:
    return [(n.id, n) for n in g.nodes]


def find_node(g: Graph, phrase: str, report: Dict[str, Any]) -> Optional[Any]:
    """Resolve 'the conv', 'conv2d_2', 'my head', 'the last layer' to one node."""
    phrase = (phrase or "").strip().lower().strip(".,")
    if not phrase:
        return None
    order = report.get("order") or [n.id for n in g.nodes]
    nodes = g.by_id()

    if phrase in ("the last layer", "last layer", "the end", "the last one", "last"):
        for nid in reversed(order):
            if nodes[nid].type != "Output":
                return nodes[nid]
    if phrase in ("the first layer", "first layer", "the input", "input"):
        for nid in order:
            if nodes[nid].type == "Input":
                return nodes[nid]

    phrase = re.sub(r"^(the|my|a|an)\s+", "", phrase)

    exact = [n for _, n in _candidates(g) if (n.label or "").lower() == phrase]
    if exact:
        return exact[0]
    typed = [n for _, n in _candidates(g) if n.type.lower() == phrase]
    if typed:
        return typed[-1]
    partial = [n for _, n in _candidates(g)
               if phrase in (n.label or "").lower() or phrase in n.type.lower()]
    if partial:
        return partial[-1]

    # the names in the generated file — conv2d_1, linear_2 — are what people read
    # off the canvas, so they should resolve too
    numbered = re.match(r"^([a-z0-9_]+?)_(\d+)$", phrase)
    if numbered:
        base, index = numbered.group(1), int(numbered.group(2))
        matches = [n for nid, n in _candidates(g)
                   if n.type.lower() == base or base in n.type.lower()
                   or (n.label or "").lower() == base]
        ordered = [n for nid in order for n in matches if n.id == nid] or matches
        if 1 <= index <= len(ordered):
            return ordered[index - 1]
    return None


def find_type(phrase: str) -> Optional[str]:
    phrase = (phrase or "").strip().lower().strip(".,")
    phrase = re.sub(r"^(a|an|the)\s+", "", phrase)
    aliases = {
        "dropout": "Dropout", "batchnorm": "BatchNorm2d", "batch norm": "BatchNorm2d",
        "normalization": "BatchNorm2d", "normalisation": "BatchNorm2d",
        "relu": "Activation", "activation": "Activation", "gelu": "Activation",
        "conv": "Conv2d", "convolution": "Conv2d", "dense": "Linear",
        "linear": "Linear", "fully connected": "Linear", "pool": "MaxPool2d",
        "pooling": "MaxPool2d", "maxpool": "MaxPool2d", "flatten": "Flatten",
        "layernorm": "LayerNorm", "layer norm": "LayerNorm",
        "residual": "ResidualBlock", "backbone": "Backbone",
        "attention": "SelfAttention", "embedding": "Embedding",
        "global pool": "GlobalAvgPool", "global average pool": "GlobalAvgPool",
    }
    if phrase in aliases:
        return aliases[phrase]
    for name in layers.REGISTRY:
        if name.lower() == phrase:
            return name
    for name in layers.REGISTRY:
        if phrase and phrase in name.lower():
            return name
    return None


# --------------------------------------------------------------------------
# architecture review
# --------------------------------------------------------------------------

def review(g: Graph, report: Dict[str, Any]) -> List[Dict[str, str]]:
    """Read the graph and report what is genuinely wrong or wasteful."""
    found: List[Dict[str, str]] = []
    nodes = g.by_id()
    order = report.get("order") or []
    inc = incoming_map(g)
    shapes = {nid: report["nodes"].get(nid, {}).get("out_shape") for nid in order}

    def name(n):
        return n.label or n.type

    def successors(nid):
        return [e.target for e in g.edges if e.source == nid]

    for nid in order:
        node = nodes.get(nid)
        if node is None:
            continue
        nexts = [nodes[t] for t in successors(nid) if t in nodes]

        # two dense layers with nothing between them are one dense layer
        if node.type == "Linear":
            for nxt in nexts:
                if nxt.type == "Linear":
                    found.append({
                        "level": "warn",
                        "layer": name(node),
                        "text": (f"{name(node)} feeds straight into {name(nxt)} with no "
                                 f"activation between them. Two linear layers in a row "
                                 f"compose to a single linear layer, so the extra one "
                                 f"buys nothing but parameters. Put an Activation "
                                 f"between them."),
                    })

        # a convolution whose activation sees unnormalized scale
        if node.type in ("Conv2d", "Conv1d") and len(order) > 6:
            direct = [n for n in nexts if n.type == "Activation"]
            if direct and not any(n.type.startswith("BatchNorm") or n.type == "GroupNorm"
                                  for n in nexts):
                found.append({
                    "level": "note",
                    "layer": name(node),
                    "text": (f"{name(node)} goes straight to its activation. A "
                             f"BatchNorm2d between the two steadies the scale the "
                             f"activation sees, and usually lets you train faster."),
                })

        # a flatten that hands a linear layer an enormous matrix
        if node.type == "Flatten":
            width = (shapes.get(nid) or [0])[0]
            feeding = [e.source for e in inc[nid]]
            before = shapes.get(feeding[0]) if feeding else None
            channels = before[0] if before and len(before) == 3 else None
            for nxt in nexts:
                if nxt.type == "Linear":
                    units = int(resolved_params(nxt).get("units", 0))
                    cost = width * units
                    if cost > 2_000_000:
                        alternative = (
                            f" GlobalAvgPool instead would give {channels} features "
                            f"and a head of {channels * units:,} weights."
                            if channels else "")
                        found.append({
                            "level": "warn",
                            "layer": name(node),
                            "text": (f"{name(node)} produces {width:,} numbers, so "
                                     f"{name(nxt)} needs {cost:,} weights — most of the "
                                     f"network in one layer.{alternative}"),
                        })

    # a trunk with nothing to regularize it
    trainable = report.get("total_learnables", 0)
    kinds = {nodes[nid].type for nid in order if nid in nodes}
    if trainable > 200_000 and not ({"Dropout", "Dropout2d"} & kinds):
        found.append({
            "level": "note", "layer": "",
            "text": (f"{trainable:,} trainable parameters and no Dropout anywhere. On a "
                     f"small dataset that head will memorize before the trunk "
                     f"generalizes."),
        })

    # spatial size collapsed before the network got deep
    for nid in order:
        shape = shapes.get(nid)
        node = nodes.get(nid)
        if node and shape and len(shape) == 3 and shape[1] == 1 and shape[2] == 1:
            after = [n for n in successors(nid) if nodes.get(n)
                     and nodes[n].type in ("Conv2d", "MaxPool2d", "AvgPool2d")]
            if after:
                found.append({
                    "level": "warn", "layer": name(node),
                    "text": (f"After {name(node)} the feature map is 1x1, and there are "
                             f"still convolutions below it. They have no spatial extent "
                             f"left to look at."),
                })
            break

    # everything frozen
    frozen = [nid for nid in order
              if nodes.get(nid) and nodes[nid].params.get("_frozen")]
    if frozen and trainable == 0:
        found.append({
            "level": "warn", "layer": "",
            "text": ("Every layer with weights is frozen, so training would change "
                     "nothing. Unfreeze the head at least."),
        })

    # the head's width against the task
    outputs = [nodes[nid] for nid in order if nodes.get(nid) and nodes[nid].type == "Output"]
    for out in outputs:
        task = resolved_params(out).get("task", "classification")
        incoming = inc[out.id]
        if not incoming:
            continue
        shape = shapes.get(incoming[0].source)
        if not shape:
            continue
        if task == "regression" and shape[-1] > 32:
            found.append({
                "level": "note", "layer": name(out),
                "text": (f"The Output is set to regression but {shape[-1]} values arrive "
                         f"at it. If those are class scores, the task should be "
                         f"classification."),
            })
        if task == "classification" and shape[-1] == 1:
            found.append({
                "level": "warn", "layer": name(out),
                "text": ("One value arrives at a classification Output. Cross-entropy "
                         "needs one score per class; for a two-class problem use the "
                         "binary task instead."),
            })

    if not found:
        found.append({"level": "good", "layer": "",
                      "text": "Nothing to flag. The shapes resolve and the structure looks sound."})
    return found


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def _new_id(graph: Dict[str, Any], prefix: str = "a") -> str:
    used = {n["id"] for n in graph["nodes"]}
    i = 1
    while f"{prefix}{i}" in used:
        i += 1
    return f"{prefix}{i}"


def _place_after(graph: Dict[str, Any], source_id: str, type_: str,
                 params: Dict[str, Any]) -> str:
    """Insert a node after another, rewiring whatever followed it."""
    nid = _new_id(graph)
    source = next(n for n in graph["nodes"] if n["id"] == source_id)
    graph["nodes"].append({
        "id": nid, "type": type_, "params": params, "label": "",
        "x": source.get("x", 0), "y": source.get("y", 0) + 120,
    })
    following = [e for e in graph["edges"] if e["source"] == source_id]
    for e in following:
        e["source"] = nid
    graph["edges"].append({"id": _new_id(graph, "ae"), "source": source_id,
                           "target": nid, "port": 0})
    return nid


def handle(payload: Dict[str, Any], message: str) -> Dict[str, Any]:
    graph = payload
    g = parse(graph)
    report = analyze(g)
    text = (message or "").strip()
    low = text.lower()

    if not low:
        return {"reply": "Ask me to change something, or say review."}

    # ---- questions ----
    if low in ("help", "what can you do", "?"):
        return {"reply": HELP, "suggestions": SUGGESTIONS}

    if re.search(r"\b(review|check|what.s wrong|any problems|critique|look over)\b", low):
        notes = review(g, report)
        lines = [("• " + n["text"]) for n in notes]
        return {"reply": "\n".join(lines), "observations": notes}

    if re.search(r"\bhow many (parameters|weights|learnables)\b", low):
        total = report.get("total_learnables", 0)
        biggest = sorted(
            ((report["nodes"][nid].get("learnables", 0), nid) for nid in report.get("order", [])
             if nid in report["nodes"]), reverse=True)[:3]
        nodes = g.by_id()
        parts = ", ".join(f"{nodes[nid].label or nodes[nid].type} {count:,}"
                          for count, nid in biggest if count)
        return {"reply": f"{total:,} trainable parameters."
                         + (f" The largest are {parts}." if parts else "")}

    match = re.search(r"\b(what does|what is|explain)\s+(?:the\s+|a\s+|an\s+)?"
                      r"(.+?)(?:\s+(?:do|mean|doing))?\s*\??$", low)
    if match:
        target = find_node(g, match.group(2), report)
        type_name = target.type if target else find_type(match.group(2))
        spec = layers.REGISTRY.get(type_name or "")
        if spec:
            shape = report["nodes"].get(target.id, {}).get("out_shape") if target else None
            extra = f" Here it outputs {shape}." if shape else ""
            return {"reply": f"{spec.name}. {spec.doc}{extra}"}
        return {"reply": f"I do not have a layer called {match.group(2)}."}

    # ---- edits ----
    match = re.search(r"\b(?:add|insert|put)\s+(?:a\s+|an\s+)?(.+?)\s+"
                      r"(?:(after|before)\s+(.+?))\s*$", low)
    if match:
        type_ = find_type(match.group(1))
        if not type_:
            return {"reply": f"I do not know a layer called {match.group(1)}."}
        target = find_node(g, match.group(3), report)
        if not target:
            return {"reply": f"I cannot find a layer called {match.group(3)}."}
        anchor = target.id
        if match.group(2) == "before":
            preceding = [e.source for e in incoming_map(g)[target.id]]
            if not preceding:
                return {"reply": f"{target.label or target.type} has nothing before it."}
            anchor = preceding[0]
        spec = layers.REGISTRY[type_]
        _place_after(graph, anchor, type_, dict(spec.defaults()))
        where = "before" if match.group(2) == "before" else "after"
        return {"reply": f"Added a {type_} {where} {target.label or target.type}.",
                "graph": graph, "changed": True}

    match = re.search(r"\b(?:remove|delete|drop)\s+(?:the\s+)?(.+?)\s*$", low)
    if match:
        target = find_node(g, match.group(1), report)
        if not target:
            return {"reply": f"I cannot find a layer called {match.group(1)}."}
        # stitch the gap so the chain survives the removal
        before = [e.source for e in incoming_map(g)[target.id]]
        after = [e["target"] for e in graph["edges"] if e["source"] == target.id]
        graph["nodes"] = [n for n in graph["nodes"] if n["id"] != target.id]
        graph["edges"] = [e for e in graph["edges"]
                          if e["source"] != target.id and e["target"] != target.id]
        if before and after:
            graph["edges"].append({"id": _new_id(graph, "ae"), "source": before[0],
                                   "target": after[0], "port": 0})
        return {"reply": f"Removed {target.label or target.type}"
                         + (" and joined the gap." if before and after else "."),
                "graph": graph, "changed": True}

    match = re.search(r"\b(freeze|unfreeze)\s+(?:the\s+)?(.+?)\s*$", low)
    if match:
        target = find_node(g, match.group(2), report)
        if not target:
            return {"reply": f"I cannot find a layer called {match.group(2)}."}
        node = next(n for n in graph["nodes"] if n["id"] == target.id)
        if match.group(1) == "freeze":
            node["params"]["_frozen"] = True
        else:
            node["params"].pop("_frozen", None)
        verb = "Froze" if match.group(1) == "freeze" else "Unfroze"
        return {"reply": f"{verb} {target.label or target.type}.",
                "graph": graph, "changed": True}

    match = re.search(r"\bset\s+(?:the\s+)?([a-z_ ]+?)\s+(?:to|=)\s+([\w.\-]+)"
                      r"(?:\s+(?:on|for|of)\s+(?:the\s+)?(.+?))?\s*$", low)
    if match:
        key = match.group(1).strip().replace(" ", "_")
        raw = match.group(2)
        target = (find_node(g, match.group(3), report) if match.group(3)
                  else (g.by_id()[report["order"][-2]] if len(report.get("order", [])) > 1
                        else None))
        if not target:
            return {"reply": "Say which layer, like 'set filters to 64 on conv2d'."}
        spec = layers.REGISTRY.get(target.type)
        known = {q["name"] for q in spec.params} if spec else set()
        if key not in known:
            return {"reply": f"{target.type} has no setting called {key}. "
                             f"It has: {', '.join(sorted(known)) or 'none'}."}
        value: Any = raw
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    pass
        node = next(n for n in graph["nodes"] if n["id"] == target.id)
        node["params"][key] = value
        return {"reply": f"Set {key} to {value} on {target.label or target.type}.",
                "graph": graph, "changed": True}

    match = re.search(r"\b(?:rename|call)\s+(?:the\s+)?(.+?)\s+(?:to|as)\s+(.+?)\s*$", low)
    if match:
        target = find_node(g, match.group(1), report)
        if not target:
            return {"reply": f"I cannot find a layer called {match.group(1)}."}
        node = next(n for n in graph["nodes"] if n["id"] == target.id)
        node["label"] = match.group(2).strip()
        return {"reply": f"Renamed it to {node['label']}.", "graph": graph, "changed": True}

    match = re.search(r"\b(?:note|comment)\s+(?:on\s+)?(?:the\s+)?(.+?)[:,]\s*(.+)$", text,
                      re.I)
    if match:
        target = find_node(g, match.group(1), report)
        if not target:
            return {"reply": f"I cannot find a layer called {match.group(1)}."}
        node = next(n for n in graph["nodes"] if n["id"] == target.id)
        node["params"]["_note"] = match.group(2).strip()
        return {"reply": f"Noted on {target.label or target.type}.",
                "graph": graph, "changed": True}

    return escalate(text, g, report)


HELP = """I edit the graph and check it over. I am not a language model — I match
a small set of phrasings exactly, so what I do is predictable.

Editing
  add dropout after conv2d
  add batchnorm before the activation
  remove the flatten
  set filters to 64 on conv2d
  set units to 10 on the head
  freeze the backbone
  rename linear_1 to head
  note on conv2d: widened this for the tiles

Asking
  review
  how many parameters
  what does GlobalAvgPool do"""

SUGGESTIONS = [
    "review",
    "how many parameters",
    "add dropout after the last layer",
    "freeze the backbone",
    "what does GlobalAvgPool do",
]


def escalate(text: str, g: Graph, report: Dict[str, Any]) -> Dict[str, Any]:
    """Hand an unrecognised question to a real model, if one is configured."""
    key = os.environ.get("ASSISTANT_API_KEY")
    if not key:
        return {
            "reply": ("I did not recognise that. I match a small set of phrasings "
                      "exactly rather than guessing — say `help` to see them, or "
                      "`review` to have me look the network over.\n\n"
                      "For open conversation, set ASSISTANT_API_KEY in the "
                      "environment and I will pass questions to a real model along "
                      "with a summary of this graph."),
            "suggestions": SUGGESTIONS,
        }
    try:
        import json
        import urllib.request

        nodes = g.by_id()
        summary = "\n".join(
            f"{nodes[nid].label or nodes[nid].type} ({nodes[nid].type}) -> "
            f"{report['nodes'].get(nid, {}).get('out_shape')}"
            for nid in report.get("order", []) if nid in nodes)
        body = json.dumps({
            "model": os.environ.get("ASSISTANT_MODEL", "claude-sonnet-4-6"),
            "max_tokens": 700,
            "messages": [{"role": "user", "content":
                          f"This is a neural network being designed:\n{summary}\n\n"
                          f"Question: {text}"}],
        }).encode()
        request = urllib.request.Request(
            os.environ.get("ASSISTANT_API_URL", "https://api.anthropic.com/v1/messages"),
            data=body,
            headers={"content-type": "application/json", "x-api-key": key,
                     "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(request, timeout=40) as reply:
            data = json.loads(reply.read())
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        return {"reply": "\n".join(parts).strip() or "(no answer came back)",
                "source": "model"}
    except Exception as exc:  # noqa: BLE001
        return {"reply": f"The configured model could not be reached: {exc}"}
