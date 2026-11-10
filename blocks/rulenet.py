"""A network assembled from rules, and its sequence counterpart: what the super agent builds.

Source is shared with rulesource.py, which the super agent runs and measures, so the block on the canvas
is the identical class.
"""
from blocks_sdk import Block, Param, ShapeError, install

import rulesource


def _rules(p):
    return [r.strip() for r in str(p.get("rules") or "").replace("+", ",").split(",") if r.strip()]


def _signs(p):
    text = str(p.get("signs") or "").strip()
    return [int(float(v)) for v in text.replace(";", ",").split(",") if v.strip()] if text else None


def _args(p, ins):
    shape = [int(v) for v in ins[0]]
    return dict(shape=shape, out=int(p["out"]), rules=_rules(p), hidden=int(p["hidden"]), k=int(p["k"]),
                momenta=bool(p["momenta"]), signs=_signs(p))


def _infer(p, ins):
    shape = list(ins[0])
    if len(shape) not in (1, 2):
        raise ShapeError(f"RuleNet reads one row of numbers [d] or N rows [N, d]; got {shape}.")
    rules = _rules(p)
    unknown = [r for r in rules if r not in rulesource.RULES]
    if unknown:
        raise ShapeError(f"RuleNet has no rule called {unknown[0]!r}. There are: {', '.join(rulesource.RULES)}.")
    return shape if "energy" in rules else [int(p["out"])]


def _count(p, ins, out):
    try:
        return rulesource.learnables(**_args(p, ins))
    except Exception:  # noqa: BLE001 - an unbuildable combination is reported by the shape check, not here
        return 0


install(Block(
    name="RuleNet",
    category="Laws · composed",
    doc="One network assembled from rules, each held by its structure for any weights: energy (the output is "
        "the motion of one learned energy), permutation, rotation, translation and reflection for rows of "
        "particles or points; monotone, convex and additive for a row of numbers. They compose: a "
        "rotation-invariant energy conserves angular momentum. Built by the super agent, or by hand.",
    params=[
        Param("rules", "text", "energy, permutation", help="Comma-separated: " + ", ".join(rulesource.RULES)),
        Param("out", "int", 1, min=1, help="Outputs per example. Ignored with energy: the output is the motion, shaped like the input"),
        Param("hidden", "int", 32, min=1),
        Param("k", "int", 1, min=1, help="Coordinates per position vector in a row"),
        Param("momenta", "bool", False, help="Each row is positions then momenta (always so with energy)"),
        Param("signs", "text", "", help="For monotone: one of 1, -1, 0 per input, comma-separated"),
    ],
    infer=_infer,
    learnables=_count,
    imports="import math",
    prelude=rulesource.RULENET,
    torch_init=lambda p, ins: ("RuleNet({shape}, out={out}, rules={rules!r}, hidden={hidden}, k={k}, momenta={momenta}, "
                               "signs={signs!r})").format(**_args(p, ins)),
))


def _seq_infer(p, ins):
    shape = list(ins[0])
    if len(shape) != 2:
        raise ShapeError(f"SeqNet reads a sequence as [L, features]; got {shape}.")
    if p["rule"] not in ("slots", "fastweights", "permutation"):
        raise ShapeError(f"SeqNet has no rule called {p['rule']!r}.")
    return [int(p["out"])]


def _seq_count(p, ins, out):
    try:
        net = rulesource.build("SeqNet")(int(ins[0][-1]), int(p["out"]), p["rule"], hidden=int(p["hidden"]), slots=int(p["slots"]))
        return sum(q.numel() for q in net.parameters())
    except Exception:  # noqa: BLE001
        return 0


install(Block(
    name="SeqNet",
    category="Laws · composed",
    doc="A sequence model with one rule about memory or order built in: slots (only K vectors leave, however "
        "long the list), fastweights (a memory written by a local Hebbian rule) or permutation (a bag: order "
        "cannot matter). Built by the super agent, or by hand.",
    params=[
        Param("rule", "select", "slots", options=["slots", "fastweights", "permutation"]),
        Param("out", "int", 4, min=1),
        Param("hidden", "int", 32, min=1),
        Param("slots", "int", 4, min=1, help="For slots: how many vectors may leave"),
    ],
    infer=_seq_infer,
    learnables=_seq_count,
    prelude=rulesource.source("SeqNet"),
    torch_init=lambda p, ins: (f"SeqNet({int(ins[0][-1])}, {int(p['out'])}, {p['rule']!r}, hidden={int(p['hidden'])}, "
                               f"slots={int(p['slots'])})"),
))
