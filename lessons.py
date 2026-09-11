"""The ML Math Lab curriculum, tied to the layers it explains.

Eighteen lessons, each with the problem it answers, why the answer is what it
is, a derivation in steps, a worked example, code, a deeper note, the mistake
people make, and a question to check yourself on. They were written for a
separate application; what makes them worth having here is that most of them
are about something this application already computes.

So they are not a separate section. A lesson appears beside the layer it
explains — `Linear transformation` next to `Linear`, `Scaled dot-product
attention` next to `Attention` — and the layer's own equation from `mathbook`
sits next to the lesson's. When those two disagree, that is worth seeing.

The mapping from lesson to layer is written down rather than guessed, and a
test checks that every lesson is accounted for and that every layer named is a
layer that exists. Four lessons name no layer on purpose: the chain rule,
gradient descent and Bellman optimality are about training rather than
structure, and nothing in the palette is one of them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent / "lessons"

#: Where the curriculum came from, kept beside it because a reader deserves to
#: know whose explanation they are reading.
SOURCE = {
    "name": "ML Math Lab",
    "note": "A math-first learning app with an inspectable computation graph, "
            "live intermediate values, derivations and numerical experiments.",
}

_cache: Dict[str, Any] = {}


def _load() -> Dict[str, Any]:
    if _cache:
        return _cache
    try:
        lessons = json.loads((HERE / "ml-math-lab.json").read_text())
        mapping = json.loads((HERE / "layers.json").read_text())
    except Exception:  # noqa: BLE001 - absent curriculum is not a failure
        _cache.update({"lessons": [], "by_layer": {}, "by_id": {}})
        return _cache

    by_layer: Dict[str, List[str]] = {}
    for lesson_id, layers in mapping.items():
        for name in layers:
            by_layer.setdefault(name, []).append(lesson_id)

    _cache.update({
        "lessons": lessons,
        "by_id": {entry["id"]: entry for entry in lessons},
        "map": mapping,
        "by_layer": by_layer,
    })
    return _cache


def catalog() -> Dict[str, Any]:
    """Every lesson, with what it explains and what it needs first."""
    data = _load()
    out = []
    for entry in data["lessons"]:
        out.append({
            "id": entry["id"], "name": entry["name"], "group": entry["group"],
            "level": entry["level"], "eq": entry["eq"],
            "problem": entry["problem"],
            "prereq": entry.get("prereq", []),
            "uses": entry.get("uses", []),
            "layers": data["map"].get(entry["id"], []),
        })
    return {"lessons": out, "source": SOURCE,
            "groups": sorted({e["group"] for e in out})}


def lesson(lesson_id: str) -> Optional[Dict[str, Any]]:
    data = _load()
    entry = data["by_id"].get(lesson_id)
    if not entry:
        return None
    return {**entry, "layers": data["map"].get(lesson_id, []),
            "source": SOURCE}


def for_layer(layer: str) -> List[Dict[str, Any]]:
    """The lessons that explain one layer, in the order they should be read."""
    data = _load()
    ids = data["by_layer"].get(layer, [])
    order = {entry["id"]: i for i, entry in enumerate(data["lessons"])}
    return [data["by_id"][i] for i in sorted(ids, key=lambda x: order.get(x, 99))]


def path_to(lesson_id: str) -> List[str]:
    """Everything to read first, deepest prerequisite first.

    The curriculum declares prerequisites per lesson; walking them gives a
    reading order rather than a list. A cycle would loop forever, so the walk
    remembers where it has been — the lessons have none today, and a later
    edit should not be able to hang the page.
    """
    data = _load()
    seen: List[str] = []
    walking: set = set()

    def walk(here: str) -> None:
        if here in walking or here in seen:
            return
        walking.add(here)
        entry = data["by_id"].get(here)
        for need in (entry or {}).get("prereq", []):
            walk(need)
        walking.discard(here)
        if here != lesson_id:
            seen.append(here)

    walk(lesson_id)
    return seen


def covering(graph: Dict[str, Any]) -> Dict[str, Any]:
    """What a design's layers are explained by, and what is left unexplained."""
    data = _load()
    types = [n.get("type", "") for n in (graph or {}).get("nodes", [])]
    taught: Dict[str, List[str]] = {}
    silent = []
    for name in sorted(set(types)):
        ids = data["by_layer"].get(name, [])
        if ids:
            taught[name] = ids
        elif name not in ("Input", "Output"):
            silent.append(name)
    return {"taught": taught, "silent": silent,
            "lessons": {i: data["by_id"][i]["name"]
                        for ids in taught.values() for i in ids}}
