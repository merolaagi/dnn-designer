"""What to do next with this network, and what it would cost.

`review` in assistant.py finds what is wrong. This answers a different question:
given where you are, what is worth trying — and it answers it from the graph
rather than from a list of general advice, because "add batch norm" is useless
until you know there is a convolution without one.

Every suggestion carries three things: what to do, why here specifically, and
what it would cost in parameters. The cost matters. Most architectural advice is
delivered without it, and a suggestion that quietly triples the model is a
different suggestion from one that does not.

Nothing here is a prediction that a change will help. Whether it helps is what
the Studies tab measures; this only says what is worth measuring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import graph as G
import layers
from graph import Graph


def _share(part: int, whole: int) -> str:
    if not whole:
        return ""
    return f"{part / whole * 100:.0f}% of the network"


def advise(g: Graph, report: Dict[str, Any],
           selected: Optional[str] = None) -> Dict[str, Any]:
    """Suggestions for the whole design, and for the layer in hand."""
    nodes = g.by_id()
    order = report.get("order", [])
    total = report.get("total_learnables", 0)
    outgoing: Dict[str, List[str]] = {nid: [] for nid in nodes}
    incoming: Dict[str, List[str]] = {nid: [] for nid in nodes}
    for edge in g.edges:
        outgoing.setdefault(edge.source, []).append(edge.target)
        incoming.setdefault(edge.target, []).append(edge.source)

    def kind(nid) -> str:
        return nodes[nid].type if nid in nodes else ""

    def after(nid) -> List[str]:
        return [kind(x) for x in outgoing.get(nid, [])]

    ideas: List[Dict[str, Any]] = []

    def add(title, why, cost="", action=None, weight=0):
        ideas.append({"title": title, "why": why, "cost": cost,
                      "action": action, "weight": weight})

    # --- where the parameters actually are ------------------------------
    sized = [(nid, report["nodes"].get(nid, {}).get("learnables", 0))
             for nid in order]
    sized = [(nid, n) for nid, n in sized if n]
    if sized and total:
        heaviest, count = max(sized, key=lambda x: x[1])
        if count / total > 0.5:
            node = nodes[heaviest]
            add(f"{node.label or node.type} holds {_share(count, total)}",
                "One layer carrying most of the weights is where both the "
                "capacity and the overfitting live. If you are going to change "
                "one number, change this one first — and if you are going to "
                "quantize, this is what the Precision view will be arguing "
                "about.",
                f"{count:,} parameters today", weight=3)

    # --- structural gaps, each checked against this graph ---------------
    for nid in order:
        node = nodes[nid]
        shape = report["nodes"].get(nid, {}).get("out_shape") or []
        nxt = after(nid)

        if node.type in ("Conv2d", "Conv1d") and not any(
                k in ("BatchNorm2d", "BatchNorm1d", "GroupNorm", "LayerNorm")
                for k in nxt):
            add(f"Normalize after {node.label or node.type}",
                "A convolution with no normalization after it trains slower and "
                "is fussier about the learning rate. This is the cheapest "
                "structural change available.",
                f"{2 * (shape[0] if shape else 0):,} parameters",
                action=f"add batchnorm2d after {node.label or node.type}",
                weight=2)

        if node.type == "Linear" and any(k == "Linear" for k in nxt):
            add("Put an activation between the two dense layers",
                "Linear into Linear composes to a single linear map, so the "
                "second one is buying parameters and no expressiveness. This is "
                "not a matter of taste.",
                "no parameters",
                action=f"add activation after {node.label or node.type}",
                weight=5)

        if node.type == "Flatten":
            width = 1
            for d in shape:
                width *= int(d)
            downstream = [x for x in outgoing.get(nid, [])
                          if kind(x) == "Linear"]
            if downstream and width > 4096:
                units = int(G.resolved_params(nodes[downstream[0]]).get("units", 0))
                add("Pool before flattening",
                    f"Flatten hands the next layer {width:,} numbers, and that "
                    f"one dense layer is {width * units:,} weights on its own. "
                    f"A GlobalAvgPool or one more stride would cut it by a "
                    f"factor of four and usually costs little accuracy.",
                    f"would save about {width * units * 3 // 4:,} parameters",
                    action="add globalavgpool before flatten", weight=4)

    # --- regularization, but only when the shape of the net asks for it --
    has_dropout = any(nodes[n].type == "Dropout" for n in order)
    dense = sum(report["nodes"].get(n, {}).get("learnables", 0)
                for n in order if nodes[n].type == "Linear")
    if not has_dropout and total > 100_000 and dense > total * 0.4:
        add("Add dropout before the head",
            f"{dense:,} of {total:,} parameters are in dense layers and nothing "
            f"is regularizing them. On a small dataset that head memorizes "
            f"before the trunk generalizes.",
            "no parameters",
            action="add dropout before the last layer", weight=3)

    # --- depth and width, phrased as an experiment ----------------------
    convs = [n for n in order if nodes[n].type in ("Conv2d", "Conv1d")]
    if convs and len(convs) < 3:
        add("Go deeper before going wider",
            f"There {'is' if len(convs) == 1 else 'are'} {len(convs)} "
            f"convolution{'' if len(convs) == 1 else 's'}. Depth buys receptive "
            f"field and reuse; width buys capacity at a square cost. On images "
            f"the first usually wins, but this is exactly what the architecture "
            f"search settles rather than an argument to have.",
            "a second conv block roughly doubles the trunk", weight=2)

    residual = any(nodes[n].type in ("Add", "ResidualBlock") for n in order)
    if len(convs) >= 4 and not residual:
        add("Add skip connections",
            f"{len(convs)} convolutions deep with nothing to carry the gradient "
            f"back. This is the depth at which plain stacks start to train worse "
            f"than shallower ones — the observation ResNet was built on.",
            "no parameters for the skip itself", weight=4)

    # --- sequence models -------------------------------------------------
    embeds = [n for n in order if nodes[n].type == "Embedding"]
    attends = [n for n in order if nodes[n].type in ("Attention", "SelfAttention",
                                                     "TransformerEncoder")]
    positions = any(nodes[n].type == "LearnedPositions" for n in order)
    if attends and embeds and not positions:
        add("Nothing tells this model the order of the tokens",
            "Attention is permutation invariant: without positions, "
            "\u201cdog bites man\u201d and \u201cman bites dog\u201d are the "
            "same input. Almost always a bug rather than a choice.",
            "context \u00d7 width parameters",
            action="add learnedpositions after embedding", weight=5)

    ideas.sort(key=lambda x: -x["weight"])

    return {
        "ideas": ideas[:6],
        "layer": _about(nodes, report, selected, incoming, outgoing, total),
        "total": total,
    }


def _about(nodes, report, selected, incoming, outgoing, total) -> Optional[Dict[str, Any]]:
    """What the selected layer is doing, and what can be changed about it."""
    if not selected or selected not in nodes:
        return None
    node = nodes[selected]
    info = report["nodes"].get(selected, {})
    spec = layers.REGISTRY.get(node.type)
    params = G.resolved_params(node)
    count = info.get("learnables", 0)

    knobs = []
    for entry in (spec.params if spec else []):
        name = entry["name"]
        if name not in params:
            continue
        knobs.append({
            "name": name,
            "value": params[name],
            "help": entry.get("help", ""),
            "effect": _effect(node.type, name, params, count),
        })

    return {
        "id": selected,
        "type": node.type,
        "name": node.label or node.type,
        "in_shapes": [report["nodes"].get(s, {}).get("out_shape")
                      for s in incoming.get(selected, [])],
        "out_shape": info.get("out_shape"),
        "parameters": count,
        "share": _share(count, total) if count else "",
        "doc": (spec.doc if spec else ""),
        "knobs": knobs[:6],
        "feeds": [nodes[x].label or nodes[x].type
                  for x in outgoing.get(selected, []) if x in nodes],
    }


def _effect(kind: str, name: str, params: Dict[str, Any], count: int) -> str:
    """What moving this number does, in the terms that matter."""
    if name in ("units", "filters"):
        return ("Parameters scale with this linearly here, and with its square "
                "if you change the layer feeding it too.")
    if name == "kernel":
        k = int(params.get("kernel", 3) or 3)
        return (f"Parameters go as the square: {k}\u00d7{k} is "
                f"{k * k / 9:.1f}\u00d7 a 3\u00d73. Two stacked 3\u00d73 see "
                f"as far as one 5\u00d75 for fewer weights.")
    if name == "stride":
        return "Halving the map quarters everything downstream of it."
    if name == "heads":
        return ("Free in parameters — the width is split, not multiplied. More "
                "heads means narrower ones.")
    if name == "rate":
        return "No parameters. Costs training speed, buys generalization."
    if name in ("bias",):
        return "One per output. Rarely matters after a normalization layer."
    if name == "groups":
        return "Divides the parameter count. Equal to the width makes it depthwise."
    return ""
