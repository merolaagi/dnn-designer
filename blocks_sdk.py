"""The plug-in SDK.

A block is one Python file in ``blocks/`` that ends by calling ``install(...)``.
Drop a file in, hit Reload, and it appears in the palette — no server restart and
nothing to edit in the core.

There are two kinds of block, and the difference is not cosmetic:

``kind="layer"``
    Differentiable. It sits in the graph, declares how it reshapes an
    activation, and its code lands inside ``forward()``.

``kind="runtime"``
    Everything that is not a tensor transform: tree search, solvers that need a
    training loop around them, self-play. It attaches to the graph so the
    designer knows it exists, but it generates a separate section of the file
    that wraps the finished model. It has no output shape because it does not
    produce an activation.

Every block carries its own ``prelude`` — real class definitions that are copied
into the generated file. That is the point: a block is source you can read and
edit, not an opaque node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from layers import LayerSpec, ShapeError, conv_out, need_rank, prod, register

__all__ = [
    "Block", "Param", "install", "ShapeError",
    "conv_out", "need_rank", "prod",
]


def Param(name: str, kind: str, default: Any, **extra) -> Dict[str, Any]:
    """One editable setting.

    kind is one of: int, float, bool, text, code, select, shape, padding, dict.
    Pass ``options=[...]`` with select, and ``help="..."`` for the hint under the
    field in the inspector.
    """
    out = {"name": name, "kind": kind, "default": default}
    out.update(extra)
    return out


@dataclass
class Block:
    name: str
    category: str = "Blocks"
    kind: str = "layer"                      # layer | runtime
    doc: str = ""
    params: List[Dict[str, Any]] = field(default_factory=list)
    n_inputs: int = 1                        # -1 two or more, -2 one or more

    # layer blocks
    infer: Optional[Callable] = None         # (p, in_shapes) -> shape
    torch_init: Optional[Callable] = None    # (p, in_shapes) -> constructor source
    torch_call: Optional[Callable] = None    # (p, in_vars, mod, in_shapes) -> expression
    prelude: str = ""                        # class definitions for the PyTorch file

    # keras is optional; leave keras_call None if there is no clean equivalent
    keras_call: Optional[Callable] = None
    keras_prelude: str = ""

    # runtime blocks
    runtime_init: Optional[Callable] = None  # (p, in_vars) -> constructor source
    runtime_name: Optional[Callable] = None  # (p) -> attribute name in build_runtime

    imports: str = ""

    # optional: report parameter count so the canvas total stays honest
    learnables: Optional[Callable] = None   # (p, in_shapes, out_shape) -> int
    learnables_approx: bool = False         # set when the figure is published, not derived

    def defaults(self) -> Dict[str, Any]:
        return {q["name"]: q["default"] for q in self.params}


def _default_call(p, ins, mod, shapes):
    return f"{mod}({', '.join(ins)})"


def _passthrough_infer(p, ins):
    return list(ins[0]) if ins else []


def install(block: Block) -> LayerSpec:
    """Register a block so it shows up in the palette and the code generators."""
    if block.kind not in ("layer", "runtime"):
        raise ValueError(f"{block.name}: kind must be 'layer' or 'runtime'")

    prelude = block.prelude
    if block.imports:
        prelude = block.imports.rstrip() + "\n\n\n" + prelude if prelude else block.imports

    spec = LayerSpec(
        name=block.name,
        category=block.category,
        kind=block.kind,
        params=list(block.params),
        n_inputs=block.n_inputs,
        doc=block.doc,
        infer=block.infer or _passthrough_infer,
        torch_init=block.torch_init or (lambda p, ins: None),
        torch_call=block.torch_call or _default_call,
        keras_call=block.keras_call,
        torch_prelude=prelude,
        keras_prelude=block.keras_prelude,
        runtime_init=block.runtime_init,
        runtime_name=block.runtime_name,
        learnables=block.learnables,
        learnables_approx=block.learnables_approx,
        source="block",
    )
    return register(spec)
