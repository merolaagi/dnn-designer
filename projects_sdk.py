"""Guided projects — a build, one layer at a time, with the reasoning attached.

A project is an ordered list of steps. Each step places one or more layers,
says why that layer and not another, and names what you would reach for
instead. The point is not to produce a finished network by clicking Next; it is
to make the reasoning visible while the shapes update in front of you.

Projects are declared by builder functions rather than written out by hand, so
the rationale can quote the actual numbers. "Flatten turns [64, 8, 8] into
[4096], which is why the next Linear is wide" is worth reading; "Flatten
flattens" is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["Project", "Step", "install", "REGISTRY", "chain", "node"]


def node(type_: str, params: Optional[Dict[str, Any]] = None,
         id: Optional[str] = None, label: str = "") -> Dict[str, Any]:
    return {"type": type_, "params": params or {}, "id": id, "label": label}


@dataclass
class Step:
    """One move in the build."""

    title: str
    why: str
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    alternatives: str = ""
    watch: str = ""
    # by default a step chains onto the last node of the previous step; give an
    # explicit id to branch, as a GAN or a two-headed network needs
    connect_from: Optional[str] = None
    connect: List[Tuple[str, str, int]] = field(default_factory=list)

    def to_json(self, index: int) -> Dict[str, Any]:
        return {
            "index": index,
            "title": self.title,
            "why": self.why,
            "alternatives": self.alternatives,
            "watch": self.watch,
            "nodes": self.nodes,
            "connect_from": self.connect_from,
            "connect": [list(c) for c in self.connect],
        }


@dataclass
class Project:
    id: str
    name: str
    category: str
    summary: str
    steps: List[Step]
    tags: List[str] = field(default_factory=list)
    difficulty: str = "starter"          # starter | intermediate | advanced
    data: str = ""                       # what to feed it
    recipe: str = ""                     # training loop, blank means the standard one
    training: str = ""                   # settings worth starting from
    expect: str = ""                     # what a working run looks like
    caution: str = ""                    # where this gets you into trouble

    def to_json(self, full: bool = False) -> Dict[str, Any]:
        out = {
            "id": self.id, "name": self.name, "category": self.category,
            "summary": self.summary, "tags": self.tags,
            "difficulty": self.difficulty, "recipe": self.recipe,
            "steps": len(self.steps),
        }
        if full:
            out.update({
                "data": self.data, "training": self.training,
                "expect": self.expect, "caution": self.caution,
                "plan": [s.to_json(i) for i, s in enumerate(self.steps)],
            })
        return out


REGISTRY: Dict[str, Project] = {}


def install(project: Project) -> Project:
    if project.id in REGISTRY:
        raise ValueError(f"duplicate project id {project.id}")
    if not project.steps:
        raise ValueError(f"{project.id}: a project needs at least one step")
    REGISTRY[project.id] = project
    return project


def chain(*steps: Step) -> List[Step]:
    return list(steps)


# --------------------------------------------------------------------------
# shape arithmetic, so the rationale can quote real numbers
# --------------------------------------------------------------------------

def after_pools(size: int, times: int) -> int:
    for _ in range(times):
        size = max(1, size // 2)
    return size


def commas(n: int) -> str:
    return f"{n:,}"
