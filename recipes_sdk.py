"""Training recipes — plug-in training loops.

`blocks/` made the layer set extensible. This does the same for the thing that
was actually blocking everything else: the training loop.

The built-in loop assumes one model, one optimizer, and a loss computed from
predictions and labels. That assumption is the whole reason GANs, diffusion,
contrastive pretraining, reinforcement learning and detection were out of reach
— not the architectures, which build on the canvas already, but the loop around
them. A recipe owns that loop.

A recipe file lives in `recipes/`, ends by calling `install(...)`, and hot
reloads like a block. It gets a context holding the models, the device and its
own settings, and it does whatever it likes per batch: two optimizers, a noise
schedule, an environment rollout, several forward passes. It reports back a
dictionary of numbers and the trainer charts them.

Deliberately, a recipe owns its own backward pass and optimizer steps. Anything
less general could not express a GAN, where the discriminator and generator
updates interleave and each needs its own graph retained or freed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Recipe", "Param", "install", "REGISTRY", "Context"]


def Param(name: str, kind: str, default: Any, **extra) -> Dict[str, Any]:
    out = {"name": name, "kind": kind, "default": default}
    out.update(extra)
    return out


@dataclass
class Context:
    """Everything a recipe is handed. `state` is yours to scribble in."""

    models: Dict[str, Any]              # "main" is the canvas graph; others by name
                                        # extras are saved designs, built and ready
    device: str
    cfg: Dict[str, Any]                 # the recipe's own settings, defaults filled
    in_shapes: List[List[int]]
    out_shape: Optional[List[int]]
    epoch: int = 0
    step_index: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    optimizers: Dict[str, Any] = field(default_factory=dict)

    @property
    def model(self):
        return self.models["main"]

    def parameters(self, name: str = "main"):
        return [p for p in self.models[name].parameters() if p.requires_grad]


@dataclass
class Recipe:
    name: str
    doc: str = ""
    params: List[Dict[str, Any]] = field(default_factory=list)

    # extra networks this recipe needs, given as saved design names the user
    # picks in the form. A GAN asks for one; most recipes ask for none.
    extra_models: List[str] = field(default_factory=list)

    # data this recipe can work with: any of image, tabular, text, none
    accepts: List[str] = field(default_factory=lambda: ["image"])
    uses_labels: bool = False

    # True when the recipe makes its own data — reinforcement learning rolls out
    # an environment, and nothing in a DataLoader describes that. The trainer
    # then calls step() a fixed number of times per epoch with no batch.
    self_supplied: bool = False
    steps_per_epoch: int = 200

    # What shape the incoming data should be, when it differs from the model's
    # own Input. A GAN takes noise but reads images, so the loader has to be
    # told about the images rather than the noise.
    data_shape: Optional[Callable] = None   # (ctx) -> list of shapes

    setup: Optional[Callable] = None      # (ctx) -> None
    step: Optional[Callable] = None       # (ctx, xs, y) -> dict of floats
    evaluate: Optional[Callable] = None   # (ctx, xs, y) -> dict of floats
    preview: Optional[Callable] = None    # (ctx) -> str shown in the log
    check: Optional[Callable] = None      # (ctx) -> str complaint, or None

    # which metric decides the best checkpoint, and whether lower wins
    objective: str = "loss"
    lower_is_better: bool = True

    def defaults(self) -> Dict[str, Any]:
        return {q["name"]: q["default"] for q in self.params}


REGISTRY: Dict[str, Recipe] = {}


def install(recipe: Recipe) -> Recipe:
    if recipe.step is None:
        raise ValueError(f"{recipe.name}: a recipe needs a step function")
    REGISTRY[recipe.name] = recipe
    return recipe


def catalog() -> List[Dict[str, Any]]:
    return [
        {
            "name": r.name,
            "doc": r.doc,
            "params": r.params,
            "extra_models": r.extra_models,
            "accepts": r.accepts,
            "uses_labels": r.uses_labels,
            "objective": r.objective,
        }
        for r in REGISTRY.values()
    ]
