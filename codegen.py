"""Turn an analyzed graph into source you can run.

Both generators walk the same topological order and reuse the same resolved
shapes, so the PyTorch and Keras files always describe the same architecture.

One difference is unavoidable: the IR is channels-first and Keras is
channels-last. Only the Input layer needs converting, since every Keras layer
infers the rest. Reshape and Permute are the two nodes where that difference is
visible, and the generated Keras file carries a comment saying so.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from graph import Graph, incoming_map, resolved_params
from layers import REGISTRY


def _record(sink, nid, spec, var, init, call) -> None:
    """Remember one node's contribution to the generated file."""
    if sink is None:
        return
    sink[nid] = {
        "var": var,
        "init": init,
        "call": call,
        "prelude": spec.torch_prelude.strip() or None,
        "source": spec.source,
        "origin": spec.origin,
        "kind": spec.kind,
        "keras": spec.keras_call is not None or spec.kind == "runtime",
    }


def _slug(text: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    return s or "layer"


def _class_name(name: str) -> str:
    parts = re.split(r"[^0-9a-zA-Z]+", name)
    out = "".join(p[:1].upper() + p[1:] for p in parts if p)
    if not out or out[0].isdigit():
        out = "Net" + out
    return out


def input_order(g: Graph, report: Dict[str, Any]) -> List[str]:
    """Node ids of the Input layers, in the order forward() accepts them."""
    nodes = g.by_id()
    return [i for i in report["order"] if nodes[i].type == "Input"]


def model_class_name(g: Graph) -> str:
    """The class the generated PyTorch file defines."""
    return _class_name(g.name)


def _assign_names(g: Graph, order: List[str]) -> Dict[str, str]:
    """Readable, collision-free variable names: conv2d_1, relu_2, and so on."""
    nodes = g.by_id()
    counters: Dict[str, int] = {}
    names: Dict[str, str] = {}
    taken = set()
    for nid in order:
        n = nodes[nid]
        base = _slug(n.label) if n.label else _slug(n.type)
        if n.type == "Activation":
            base = _slug(n.params.get("kind", "activation"))
        if n.type == "Custom" and n.params.get("label"):
            base = _slug(n.params["label"])
        # Sequential submodules are named "0", "1", ... which cannot start an
        # identifier, so fall back to the layer type in that case.
        if not base or not (base[0].isalpha() or base[0] == "_"):
            base = f"{_slug(n.type)}_{base}" if base else _slug(n.type)
        counters[base] = counters.get(base, 0) + 1
        name = f"{base}_{counters[base]}"
        while name in taken:
            counters[base] += 1
            name = f"{base}_{counters[base]}"
        taken.add(name)
        names[nid] = name
    return names


def _preludes(g: Graph, order: List[str], attr: str) -> List[str]:
    seen, out = set(), []
    nodes = g.by_id()
    for nid in order:
        spec = REGISTRY.get(nodes[nid].type)
        if spec is None:
            continue
        block = getattr(spec, attr)
        if block and block not in seen:
            seen.add(block)
            out.append(block.rstrip())
    return out


def _custom_imports(g: Graph, order: List[str]) -> List[str]:
    nodes, out, seen = g.by_id(), [], set()
    for nid in order:
        n = nodes[nid]
        if n.type != "Custom":
            continue
        for line in (n.params.get("imports") or "").splitlines():
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                out.append(line)
    return out


# --------------------------------------------------------------------------
# PyTorch
# --------------------------------------------------------------------------

def to_pytorch(g: Graph, report: Dict[str, Any],
               node_code: Optional[Dict[str, Any]] = None) -> str:
    """Generate the PyTorch file.

    Pass node_code to also collect, per node, the lines it contributed and the
    class source behind it — that is what the inspector shows when you select a
    layer, so the panel can never drift from the generated file.
    """
    order = report["order"]
    nodes = g.by_id()
    inc = incoming_map(g)
    names = _assign_names(g, order)
    shapes = {nid: report["nodes"].get(nid, {}).get("out_shape") for nid in order}

    init_lines: List[str] = []
    body_lines: List[str] = []
    input_args: List[str] = []
    outputs: List[str] = []
    runtimes: List[tuple] = []

    for nid in order:
        n = nodes[nid]
        spec = REGISTRY[n.type]
        params = resolved_params(n)
        var = names[nid]
        in_shapes = [shapes[e.source] for e in inc[nid] if shapes.get(e.source)]
        in_vars = [names[e.source] for e in inc[nid]]

        if spec.kind == "runtime":
            attr = spec.runtime_name(params) if spec.runtime_name else var
            ctor = spec.runtime_init(params, in_vars) if spec.runtime_init else "None"
            runtimes.append((attr, ctor, n.label or n.type, in_vars))
            _record(node_code, nid, spec, attr, f"{attr} = {ctor}",
                    "# runtime: built in build_runtime(model), not in forward()")
            continue

        if n.type == "Input":
            input_args.append(var)
            dims = ", ".join(str(d) for d in (shapes[nid] or []))
            line = f"# {var}: [B, {dims}]"
            body_lines.append(line)
            _record(node_code, nid, spec, var, None, line)
            continue

        ctor = spec.torch_init(params, in_shapes)
        mod = None
        if ctor:
            mod = f"self.{var}"
            init_lines.append(f"        {mod} = {ctor}")
            if n.params.get("_frozen"):
                # frozen here means the same thing it means anywhere: this layer
                # keeps its values and the optimizer leaves it alone
                init_lines.append(
                    f"        for p in {mod}.parameters():")
                init_lines.append(
                    f"            p.requires_grad_(False)   # {var} is frozen")
        expr = spec.torch_call(params, in_vars, mod, in_shapes)

        if n.type == "Output":
            outputs.append(in_vars[0])
            _record(node_code, nid, spec, var, None, f"return {in_vars[0]}")
            continue

        dims = ", ".join(str(d) for d in (shapes[nid] or []))
        line = f"{var} = {expr}  # -> [B, {dims}]"
        body_lines.append(line)
        _record(node_code, nid, spec, var,
                f"self.{var} = {ctor}" if ctor else None, line)

    if not outputs:
        outputs = [names[order[-1]]] if order else ["x"]
    if not input_args:
        input_args = ["x"]

    cls = _class_name(g.name)
    imports = ["import math", "", "import torch", "import torch.nn as nn",
               "import torch.nn.functional as F"]
    imports += _custom_imports(g, order)
    preludes = _preludes(g, order, "torch_prelude")

    ret = outputs[0] if len(outputs) == 1 else "(" + ", ".join(outputs) + ")"
    total = report.get("total_learnables", 0)

    parts = ["\n".join(imports), ""]
    if preludes:
        parts += ["", "\n\n\n".join(preludes), ""]
    parts += [
        "",
        f"class {cls}(nn.Module):",
        f'    """Generated from the designer canvas. About {total:,} learnable parameters."""',
        "",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    parts += init_lines or ["        pass"]
    parts += ["", f"    def forward(self, {', '.join(input_args)}):"]
    parts += ["        " + line for line in body_lines] or ["        pass"]
    parts += [f"        return {ret}", ""]

    if runtimes:
        parts += [
            "",
            "def build_runtime(model):",
            '    """Components that wrap the trained network but are not part of forward().',
            "",
            "    These are not differentiable and do not appear in the graph's data path.",
            '    """',
        ]
        for attr, ctor, label, ins in runtimes:
            reading = ", ".join(ins) or "the model"
            parts.append(f"    # {label}, reading {reading}")
            parts.append(f"    {attr} = {ctor}")
        joined = ", ".join(f'"{a}": {a}' for a, _, _, _ in runtimes)
        parts += ["    return {" + joined + "}", ""]

    example = ", ".join(
        "torch.randn(2, " + ", ".join(str(d) for d in (shapes[nid] or [1])) + ")"
        if nodes[nid].params.get("dtype", "float") != "long"
        else "torch.randint(0, 100, (2, " + ", ".join(str(d) for d in (shapes[nid] or [1])) + "))"
        for nid in order if nodes[nid].type == "Input"
    ) or "torch.randn(2, 8)"

    parts += [
        "",
        'if __name__ == "__main__":',
        f"    model = {cls}()",
        f"    y = model({example})",
        '    n = sum(p.numel() for p in model.parameters() if p.requires_grad)',
        '    print(model)',
        '    print("output:", tuple(y.shape) if hasattr(y, "shape") else [tuple(t.shape) for t in y])',
        '    print(f"learnable parameters: {n:,}")',
    ]
    if runtimes:
        parts += [
            "    runtime = build_runtime(model)",
            '    print("runtime components:", list(runtime))',
        ]
    parts.append("")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Keras
# --------------------------------------------------------------------------

def _keras_input_shape(shape: List[int]) -> str:
    if len(shape) == 3:                      # [C, H, W] -> (H, W, C)
        dims = [shape[1], shape[2], shape[0]]
    else:
        dims = list(shape)
    inner = ", ".join(str(d) for d in dims)
    return f"({inner},)" if len(dims) == 1 else f"({inner})"


def to_keras(g: Graph, report: Dict[str, Any]) -> str:
    order = report["order"]
    nodes = g.by_id()
    inc = incoming_map(g)
    names = _assign_names(g, order)
    shapes = {nid: report["nodes"].get(nid, {}).get("out_shape") for nid in order}

    body: List[str] = []
    inputs: List[str] = []
    outputs: List[str] = []
    unsupported: List[tuple] = []
    uses_numpy = False

    for nid in order:
        n = nodes[nid]
        spec = REGISTRY[n.type]
        params = resolved_params(n)
        var = names[nid]
        in_vars = [names[e.source] for e in inc[nid]]
        in_shapes = [shapes[e.source] for e in inc[nid] if shapes.get(e.source)]

        if n.type == "Input":
            dtype = "int32" if params.get("dtype") == "long" else "float32"
            body.append(
                f'{var} = keras.Input(shape={_keras_input_shape(shapes[nid] or [1])}, '
                f'dtype="{dtype}", name="{var}")'
            )
            inputs.append(var)
            continue

        if spec.kind == "runtime":
            unsupported.append((var, n.type, "runtime blocks have no Keras equivalent"))
            continue

        if n.type == "Output":
            outputs.append(in_vars[0])
            continue

        if n.type == "PositionalEncoding":
            uses_numpy = True

        if spec.keras_call is None:
            unsupported.append((var, n.type, "this block ships PyTorch code only"))
            body.append(
                f"{var} = layers.Lambda(lambda t: t)({in_vars[0]})"
                f"  # TODO: {n.type} has no Keras equivalent — port it or use the PyTorch file"
            )
            continue

        expr = spec.keras_call(params, in_vars, in_shapes)
        note = ""
        if n.type in ("Reshape", "Permute"):
            note = "  # channels-last: check this axis order against the PyTorch version"
        body.append(f"{var} = {expr}{note}")

    if not outputs:
        outputs = [names[order[-1]]] if order else []

    imports = ["import keras", "from keras import layers"]
    if uses_numpy:
        imports.insert(0, "import numpy as np")
    imports += _custom_imports(g, order)
    preludes = _preludes(g, order, "keras_prelude")

    parts = []
    if unsupported:
        parts += ["# " + "-" * 68,
                  "# This file is incomplete. The graph uses nodes with no Keras form:"]
        for var, kind, why in unsupported:
            parts.append(f"#   {kind} ({var}) — {why}")
        parts += ["# The PyTorch file is the complete one.",
                  "# " + "-" * 68, ""]
    parts += ["\n".join(imports), ""]
    if preludes:
        parts += ["", "\n\n\n".join(preludes), ""]

    ins = "[" + ", ".join(inputs) + "]"
    outs = "[" + ", ".join(outputs) + "]"
    parts += [
        "",
        "def build_model():",
        f'    """Generated from the designer canvas. Shapes here are channels-last."""',
    ]
    parts += ["    " + line for line in body]
    parts += [
        f'    return keras.Model(inputs={ins}, outputs={outs}, name="{_slug(g.name)}")',
        "",
        "",
        'if __name__ == "__main__":',
        "    model = build_model()",
        "    model.summary()",
        "",
    ]
    return "\n".join(parts)
