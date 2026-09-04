"""Bring an existing model onto the canvas.

Two routes in, with different fidelity.

**torch.fx** is the good one. Tracing an ``nn.Module`` gives back nodes that are
still whole layers — a ``call_module`` pointing at an ``nn.Conv2d`` carries that
layer's real arguments — so what lands on the canvas is what you would have built
by hand. It fails on models whose ``forward`` branches on tensor values, because
there is no single graph to trace in that case, and the error says so.

**ONNX** is the cross-framework route. It costs fidelity: ONNX has already
decomposed some layers into primitive ops, so a few nodes arrive flatter than
they left.

Neither route pretends to total coverage. An operation with no equivalent in the
registry becomes an Opaque node that keeps the original call and is flagged for
you to look at, rather than being silently dropped or silently guessed at.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NW_STEP_X = 240
NW_STEP_Y = 108


class ImportError_(ValueError):
    """A model could not be brought onto the canvas."""


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _new_graph(name: str) -> Dict[str, Any]:
    return {"name": name, "nodes": [], "edges": []}


class Builder:
    """Accumulates nodes and edges, then lays them out by depth."""

    def __init__(self, name: str):
        self.graph = _new_graph(name)
        self.n = 0
        self.notes: List[str] = []

    def add(self, type_: str, params: Dict[str, Any], label: str = "") -> str:
        self.n += 1
        nid = f"i{self.n}"
        self.graph["nodes"].append({
            "id": nid, "type": type_, "params": params, "label": label,
            "x": 0, "y": 0,
        })
        return nid

    def link(self, source: str, target: str, port: int = 0) -> None:
        if source is None or target is None:
            return
        self.graph["edges"].append({
            "id": f"e{len(self.graph['edges']) + 1}",
            "source": source, "target": target, "port": port,
        })

    def opaque(self, label: str, code: str, why: str) -> str:
        """A node the registry has no form for.

        The original is preserved verbatim in the node's values so you can see
        what was there, but the generated constructor is a stub. Emitting a
        multi-line module repr inline would produce a file that does not parse,
        and quietly substituting something plausible would produce a file that
        parses and is wrong. A stub plus a loud note is the honest option.
        """
        self.notes.append(f"{label}: {why}")
        one_line = " ".join(code.split())[:160]
        return self.add("Custom", {
            "label": label,
            "shape_rule": "list(shape)",
            "torch_code": f"nn.Identity()  # TODO: {label} was not reconstructed",
            "keras_code": 'layers.Activation("linear")',
            "imports": "",
            "values": {"original": one_line, "reason": why},
        }, label=label)

    def layout(self, rows_per_column: int = 34) -> Dict[str, Any]:
        """Place nodes by topological depth, wrapping into columns.

        A deep network laid out as one strip is unusable: DenseNet121 is 429
        layers, which at one row each is a canvas 46,000 pixels tall and a
        minimap you cannot read. Wrapping every few dozen rows into a fresh
        column keeps the whole graph in a shape you can actually see.
        """
        nodes = {n["id"]: n for n in self.graph["nodes"]}
        incoming: Dict[str, List[str]] = {k: [] for k in nodes}
        for e in self.graph["edges"]:
            if e["target"] in incoming:
                incoming[e["target"]].append(e["source"])

        depth: Dict[str, int] = {}
        for nid in nodes:                       # nodes are appended in trace order
            ins = [s for s in incoming[nid] if s in depth]
            depth[nid] = max((depth[s] + 1 for s in ins), default=0)

        deepest = max(depth.values(), default=0)
        wrap = deepest >= rows_per_column * 2

        lanes: Dict[int, int] = {}
        placement: Dict[str, Tuple[int, int, int]] = {}
        for nid in nodes:
            d = depth[nid]
            lane = lanes.get(d, 0)
            lanes[d] = lane + 1
            col = d // rows_per_column if wrap else 0
            row = d % rows_per_column if wrap else d
            placement[nid] = (col, row, lane)

        # a column is as wide as its busiest row, so columns never overlap
        widest: Dict[int, int] = {}
        for col, _, lane in placement.values():
            widest[col] = max(widest.get(col, 1), lane + 1)
        offset, running = {}, 0
        for col in sorted(widest):
            offset[col] = running
            running += widest[col] * NW_STEP_X + 90

        for nid, (col, row, lane) in placement.items():
            nodes[nid]["x"] = offset[col] + lane * NW_STEP_X
            nodes[nid]["y"] = row * NW_STEP_Y
        return self.graph


# --------------------------------------------------------------------------
# PyTorch, via torch.fx
# --------------------------------------------------------------------------

_ACTIVATIONS = {
    "ReLU": "relu", "LeakyReLU": "leaky_relu", "GELU": "gelu", "SiLU": "silu",
    "Tanh": "tanh", "Sigmoid": "sigmoid", "ELU": "elu", "Softmax": "softmax",
    "Identity": "identity", "ReLU6": "relu", "Hardswish": "silu",
    "Hardsigmoid": "sigmoid", "Mish": "silu",
}

_FUNCTIONAL = {
    "relu": "relu", "gelu": "gelu", "silu": "silu", "tanh": "tanh",
    "sigmoid": "sigmoid", "softmax": "softmax", "elu": "elu",
    "leaky_relu": "leaky_relu", "hardswish": "silu", "relu6": "relu",
}


def _pad_of(module) -> Any:
    p = getattr(module, "padding", 0)
    if isinstance(p, str):
        return p
    if isinstance(p, (tuple, list)):
        p = p[0]
    k = getattr(module, "kernel_size", 1)
    if isinstance(k, (tuple, list)):
        k = k[0]
    s = getattr(module, "stride", 1)
    if isinstance(s, (tuple, list)):
        s = s[0]
    if s == 1 and int(p) * 2 + 1 == int(k):
        return "same"
    return int(p)


def _first(value, default=1):
    if isinstance(value, (tuple, list)):
        return int(value[0])
    return int(value) if value is not None else default


def _module_to_node(module, b: Builder, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Map one nn.Module onto a registry layer, or return None if unknown."""
    import torch.nn as nn

    cls = type(module).__name__

    if isinstance(module, nn.Conv2d):
        return "Conv2d", {
            "filters": module.out_channels, "kernel": _first(module.kernel_size),
            "stride": _first(module.stride), "padding": _pad_of(module),
            "dilation": _first(module.dilation), "groups": module.groups,
            "bias": module.bias is not None,
        }
    if isinstance(module, nn.Conv1d):
        return "Conv1d", {
            "filters": module.out_channels, "kernel": _first(module.kernel_size),
            "stride": _first(module.stride), "padding": _pad_of(module),
            "dilation": _first(module.dilation),
        }
    if isinstance(module, nn.ConvTranspose2d):
        return "ConvTranspose2d", {
            "filters": module.out_channels, "kernel": _first(module.kernel_size),
            "stride": _first(module.stride),
            "padding": _first(module.padding, 0),
            "output_padding": _first(module.output_padding, 0),
        }
    if isinstance(module, nn.Linear):
        return "Linear", {"units": module.out_features, "bias": module.bias is not None}
    if isinstance(module, nn.BatchNorm2d):
        return "BatchNorm2d", {"momentum": module.momentum or 0.1, "eps": module.eps}
    if isinstance(module, (nn.BatchNorm1d,)):
        return "BatchNorm1d", {"momentum": module.momentum or 0.1, "eps": module.eps}
    if isinstance(module, nn.LayerNorm):
        return "LayerNorm", {"eps": module.eps}
    if isinstance(module, nn.GroupNorm):
        return "GroupNorm", {"groups": module.num_groups}
    if isinstance(module, nn.Dropout2d):
        return "Dropout2d", {"rate": module.p}
    if isinstance(module, nn.Dropout):
        return "Dropout", {"rate": module.p}
    if isinstance(module, nn.MaxPool2d):
        return "MaxPool2d", {"kernel": _first(module.kernel_size),
                             "stride": _first(module.stride),
                             "padding": _first(module.padding, 0)}
    if isinstance(module, nn.AvgPool2d):
        return "AvgPool2d", {"kernel": _first(module.kernel_size),
                             "stride": _first(module.stride),
                             "padding": _first(module.padding, 0)}
    if isinstance(module, nn.MaxPool1d):
        return "MaxPool1d", {"kernel": _first(module.kernel_size),
                             "stride": _first(module.stride)}
    if isinstance(module, nn.AdaptiveAvgPool2d):
        size = module.output_size
        size = size[0] if isinstance(size, (tuple, list)) else size
        return "AdaptiveAvgPool2d", {"size": int(size or 1)}
    if isinstance(module, nn.Embedding):
        return "Embedding", {"vocab": module.num_embeddings, "dim": module.embedding_dim}
    if isinstance(module, nn.LSTM):
        return "LSTM", {"units": module.hidden_size, "num_layers": module.num_layers,
                        "bidirectional": bool(module.bidirectional),
                        "return_sequences": True, "dropout": module.dropout}
    if isinstance(module, nn.GRU):
        return "GRU", {"units": module.hidden_size, "num_layers": module.num_layers,
                       "bidirectional": bool(module.bidirectional),
                       "return_sequences": True, "dropout": module.dropout}
    if isinstance(module, nn.RNN):
        return "SimpleRNN", {"units": module.hidden_size, "num_layers": module.num_layers,
                             "bidirectional": bool(module.bidirectional),
                             "return_sequences": True, "dropout": module.dropout}
    if isinstance(module, nn.Flatten):
        return "Flatten", {}
    if isinstance(module, nn.Upsample):
        scale = module.scale_factor or 2
        return "Upsample2d", {"scale": int(scale),
                              "mode": module.mode if module.mode in ("nearest", "bilinear")
                              else "nearest"}
    if cls in _ACTIVATIONS:
        return "Activation", {"kind": _ACTIVATIONS[cls]}
    return None


# Operations that decide at run time which path data takes. A traced graph is a
# fixed structure, so a mixture of experts comes out with every expert drawn as
# though all of them run — when the point of the design is that two of them do.
# Better to say so than to draw a confident lie.
ROUTING_OPS = {"topk", "nonzero", "scatter", "scatter_add", "index_put",
               "index_select", "masked_select", "where", "argmax", "argmin",
               "gather", "bincount", "unique"}


def _why_untraceable(model, exc) -> str:
    """Name the actual obstacle rather than offering one generic guess.

    The three causes look nothing alike to someone reading their own code, and
    two of them have a straightforward way round.
    """
    import inspect

    detail = str(exc)
    head = "This model cannot be traced into a single graph. "

    try:
        signature = inspect.signature(model.forward)
        variadic = [n for n, p in signature.parameters.items()
                    if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]
        tensor_args = [n for n, p in signature.parameters.items()
                       if p.default is inspect._empty
                       and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]
    except (TypeError, ValueError):
        variadic, tensor_args = [], []

    if "Proxy object cannot be iterated" in detail:
        extra = (f"Its forward() takes {'/'.join('*' + v for v in variadic)}, and "
                 if variadic else "It ")
        return (head + extra + "tracing cannot see through a variadic argument or "
                "a loop over a traced value. Recent transformers attention and "
                "mixture-of-experts modules are written this way, and wrapping "
                "them does not help — the same pattern appears inside. The same "
                "architectures written directly, as nanoGPT writes them, import "
                "without trouble.")
    if len(tensor_args) > 1:
        return (head + f"Its forward() takes {len(tensor_args)} required "
                f"arguments ({', '.join(tensor_args[:4])}), and the import "
                f"supplies one input. Wrap it in a module that takes the one "
                f"tensor and supplies the others.")
    if "slice indices" in detail or "__index__" in detail:
        return (head + "Its forward() slices a tensor using a length taken from "
                "another tensor — `x[:, :, :T]` where T came from `x.size()`. "
                "Tracing has no value for T. Written with "
                "F.scaled_dot_product_attention, or with the length fixed, it "
                "imports without trouble.")
    return (head + "Usually this means forward() branches on tensor values. "
            f"torch.fx reported: {type(exc).__name__}: {detail[:160]}")


def _wants_indices(model) -> bool:
    """True when the first thing the model does is an embedding lookup."""
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, nn.Embedding):
            return True
    return False


def from_pytorch(model, name: str = "Imported",
                 input_shape: Optional[List[int]] = None) -> Dict[str, Any]:
    """Trace an nn.Module and rebuild it as a designer graph."""
    import operator

    import torch
    import torch.fx as fx

    try:
        traced = fx.symbolic_trace(model)
    except Exception as exc:  # noqa: BLE001
        raise ImportError_(_why_untraceable(model, exc)) from exc

    # Ask PyTorch what shape every intermediate actually has, by running one
    # example through. Reconstructing `view(B, T, h, C // h)` from the traced
    # arithmetic is guesswork; this is the real answer.
    real_shapes: Dict[Any, List[int]] = {}
    if input_shape:
        try:
            from torch.fx.passes.shape_prop import ShapeProp

            sample = (torch.zeros(1, *input_shape, dtype=torch.long)
                      if _wants_indices(model)
                      else torch.zeros(1, *input_shape))
            with torch.no_grad():
                ShapeProp(traced).propagate(sample)
            for n in traced.graph.nodes:
                meta = n.meta.get("tensor_meta")
                if meta is not None and hasattr(meta, "shape"):
                    real_shapes[n] = [int(d) for d in meta.shape]
        except Exception as exc:  # noqa: BLE001 - shapes are a bonus, not a requirement
            import os
            if os.environ.get("DNN_DEBUG"):
                print("ShapeProp:", type(exc).__name__, exc)
            real_shapes = {}

    routing_found: set = set()
    b = Builder(name)
    produced: Dict[Any, str] = {}
    output_sources: List[str] = []

    # Nodes that carry shape numbers rather than tensors. `B, T, C = x.size()`
    # and `C // self.n_head` are bookkeeping the author does to reshape with —
    # drawing them as layers buries the architecture in arithmetic. They are
    # tracked so they can be left out, not guessed at.
    shape_valued = set()

    # A split produces a tuple; the getitem that follows picks which piece.
    # Emitting the Split when the split is seen would give every branch piece 0,
    # which silently makes q, k and v the same tensor. So the split is recorded
    # and a node is emitted per getitem, with the index it actually selects.
    pending_splits: Dict[Any, Dict[str, Any]] = {}

    SHAPE_SOURCES = {"size", "dim", "numel", "shape"}
    SHAPE_OPS = {operator.floordiv, operator.truediv, operator.mul, operator.add,
                 operator.sub, operator.mod, operator.getitem}

    def is_shape_node(node) -> bool:
        target = node.target
        fname = getattr(target, "__name__", str(target)).split(".")[-1]
        if node.op == "call_method" and fname in SHAPE_SOURCES:
            return True
        if node.op == "call_function" and target in SHAPE_OPS:
            # only when every tensor-ish input is itself shape-valued
            refs = [a for a in node.args if isinstance(a, fx.Node)]
            return bool(refs) and all(a in shape_valued for a in refs)
        if node.op == "call_function" and fname in SHAPE_SOURCES:
            return True
        return False

    # Operations that return their input unchanged as far as the diagram is
    # concerned. Keeping them adds nodes and says nothing.
    PASSTHROUGH = {"contiguous", "detach", "clone", "to", "cpu", "cuda", "float",
                   "double", "half", "requires_grad_", "type_as"}

    def resolve(arg) -> Optional[str]:
        return produced.get(arg) if isinstance(arg, fx.Node) else None

    def tensor_args(node) -> List[str]:
        out = []
        for a in node.args:
            if isinstance(a, fx.Node) and a in produced:
                out.append(produced[a])
            elif isinstance(a, (list, tuple)):
                out.extend(produced[x] for x in a if isinstance(x, fx.Node) and x in produced)
        return out

    for node in traced.graph.nodes:
        if is_shape_node(node):
            shape_valued.add(node)
            continue

        fname_ = getattr(node.target, "__name__", str(node.target)).split(".")[-1]
        if node.op in ("call_method", "call_function") and fname_ in PASSTHROUGH:
            upstream = [a for a in node.args if isinstance(a, fx.Node) and a in produced]
            if upstream:
                produced[node] = produced[upstream[0]]
                continue

        if node.op == "placeholder":
            shape = [int(d) for d in (input_shape or [3, 224, 224])]
            produced[node] = b.add("Input", {"shape": shape, "dtype": "float"},
                                   label=str(node.target))
            continue

        if node.op == "output":
            for src in tensor_args(node):
                output_sources.append(src)
            continue

        if node.op == "call_module":
            submodule = traced.get_submodule(node.target)
            mapped = _module_to_node(submodule, b, node.target)
            ins = tensor_args(node)
            if mapped:
                nid = b.add(mapped[0], mapped[1], label=str(node.target).split(".")[-1])
            else:
                nid = b.opaque(type(submodule).__name__, repr(submodule),
                               f"no equivalent layer; kept as written at {node.target}")
            for i, src in enumerate(ins):
                b.link(src, nid, i)
            produced[node] = nid
            continue

        if node.op in ("call_function", "call_method"):
            target = node.target
            fname = target if isinstance(target, str) else getattr(target, "__name__", str(target))
            ins = tensor_args(node)

            if fname in _FUNCTIONAL:
                nid = b.add("Activation", {"kind": _FUNCTIONAL[fname]})
            elif target is operator.add or fname in ("add", "add_"):
                nid = b.add("Add", {}) if len(ins) > 1 else None
            elif target is operator.mul or fname in ("mul", "mul_"):
                nid = b.add("Multiply", {}) if len(ins) > 1 else None
            elif fname in ("cat", "concat"):
                if len(ins) < 2:
                    produced[node] = ins[0] if ins else None
                    continue
                axis = 0
                if len(node.args) > 1 and isinstance(node.args[1], int):
                    axis = max(0, int(node.args[1]) - 1)
                nid = b.add("Concat", {"axis": axis})
            elif fname in ("flatten",):
                nid = b.add("Flatten", {})
            elif fname in ("view", "reshape") and node in real_shapes:
                # the measured shape, not one reconstructed from traced arithmetic
                nid = b.add("Reshape", {"shape": real_shapes[node][1:]})
            elif fname in ("view", "reshape"):
                dims = [int(a) for a in node.args[1:] if isinstance(a, int)]
                nid = b.add("Reshape", {"shape": dims[1:] or [-1]})
            elif fname == "scaled_dot_product_attention":
                nid = b.add("Attention",
                            {"causal": bool(node.kwargs.get("is_causal", False))})
            elif fname in ("max_pool2d", "avg_pool2d", "max_pool1d"):
                # the functional forms are as common as the modules in real code
                size = node.args[1] if len(node.args) > 1 else 2
                size = size[0] if isinstance(size, (tuple, list)) else size
                stride = node.args[2] if len(node.args) > 2 else None
                if isinstance(stride, (tuple, list)):
                    stride = stride[0]
                kind = {"max_pool2d": "MaxPool2d", "avg_pool2d": "AvgPool2d",
                        "max_pool1d": "MaxPool1d"}[fname]
                params = {"kernel": int(size or 2),
                          "stride": int(stride) if stride else 0}
                if kind != "MaxPool1d":
                    params["padding"] = 0
                nid = b.add(kind, params)
            elif fname == "adaptive_avg_pool2d":
                size = node.args[1] if len(node.args) > 1 else 1
                size = size[0] if isinstance(size, (tuple, list)) else size
                nid = b.add("AdaptiveAvgPool2d", {"size": int(size or 1)})
            elif fname == "mean" and not node.kwargs.get("dim") \
                    and not any(isinstance(a, int) for a in node.args[1:]):
                nid = b.add("GlobalAvgPool", {})
            elif fname in ("dropout",):
                nid = b.add("Dropout", {"rate": 0.5})
            elif fname in ("pow", "rsqrt", "sqrt", "exp", "log", "abs", "neg",
                           "reciprocal", "sign"):
                params = {"op": fname}
                if fname == "pow" and len(node.args) > 1 \
                        and isinstance(node.args[1], (int, float)):
                    params["n"] = float(node.args[1])
                nid = b.add("Elementwise", params)
            elif fname in ("mean", "sum", "amax", "amin"):
                axis = node.kwargs.get("dim")
                if axis is None and len(node.args) > 1 \
                        and isinstance(node.args[1], int):
                    axis = node.args[1]
                nid = b.add("Reduce", {
                    "op": {"amax": "max", "amin": "min"}.get(fname, fname),
                    "axis": int(axis if axis is not None else -1),
                    "keep": bool(node.kwargs.get("keepdim", False))})
            elif fname in ("transpose", "swapaxes"):
                dims = [a for a in node.args[1:] if isinstance(a, int)]
                nid = b.add("Transpose", {"dim_a": dims[0] if dims else 1,
                                          "dim_b": dims[1] if len(dims) > 1 else 2})
            elif fname in ("split", "chunk"):
                axis = node.kwargs.get("dim")
                if axis is None and len(node.args) > 2:
                    axis = node.args[2]
                width = None
                if real_shapes.get(node.args[0]) and len(node.args) > 1 \
                        and isinstance(node.args[1], int):
                    full = real_shapes[node.args[0]][int(axis or 0)]
                    # torch.split takes a piece size, chunk takes a count
                    width = (full // node.args[1] if fname == "split"
                             else node.args[1])
                sources = tensor_args(node)
                pending_splits[node] = {
                    "source": sources[0] if sources else None,
                    "axis": max(0, int(axis or 0) - 1),
                    "pieces": int(width or 3),
                }
                continue
            elif fname == "getitem" and node.args and node.args[0] in pending_splits:
                info = pending_splits[node.args[0]]
                index = node.args[1] if isinstance(node.args[1], int) else 0
                nid = b.add("Split", {"pieces": info["pieces"], "axis": info["axis"],
                                      "take": index})
                if info["source"]:
                    b.link(info["source"], nid, 0)
                produced[node] = nid
                continue
            elif fname in ("getattr", "getitem"):
                produced[node] = ins[0] if ins else None
                continue
            elif fname in ROUTING_OPS:
                nid = b.opaque(fname, f"# {node.format_node()}",
                               "chooses at run time which path the data takes, "
                               "which a fixed diagram cannot show")
                routing_found.add(fname)
            else:
                nid = b.opaque(fname, f"# {node.format_node()}",
                               "operation has no layer equivalent")

            if nid is None:                       # a scalar add or mul, not a merge
                produced[node] = ins[0] if ins else None
                continue
            for i, src in enumerate(ins):
                b.link(src, nid, i)
            produced[node] = nid
            continue

        if node.op == "get_attr":
            continue

    for src in output_sources or ([produced[n] for n in list(produced)[-1:]] if produced else []):
        out = b.add("Output", {"task": "classification"})
        b.link(src, out, 0)

    graph = b.layout()
    graph["_notes"] = b.notes
    if routing_found:
        graph["_routing"] = sorted(routing_found)
    # What the model actually holds, so the import can be checked against it.
    # A parameter reached through get_attr and used in plain arithmetic — an
    # RMSNorm's scale, say — belongs to no layer here and would otherwise go
    # missing from the count without anything saying so.
    try:
        graph["_expected_parameters"] = sum(p.numel() for p in model.parameters())
    except Exception:  # noqa: BLE001
        pass
    return graph


def from_torchvision(arch: str, weights: str = "none",
                     input_shape: Optional[List[int]] = None) -> Dict[str, Any]:
    import torchvision

    factory = getattr(torchvision.models, arch, None)
    if factory is None:
        raise ImportError_(f"torchvision has no model called {arch}.")
    model = factory(weights="DEFAULT" if weights not in ("none", "", None) else None)
    return from_pytorch(model, name=arch, input_shape=input_shape or [3, 224, 224])


def scan_folder(root: str, limit: int = 400) -> Dict[str, Any]:
    """Find every nn.Module in a folder without running any of it.

    Discovery reads the syntax tree rather than importing, so pointing this at
    an unfamiliar repository is safe. Only the class you then choose to import
    gets executed.
    """
    import ast

    base = Path(root).expanduser()
    if not base.is_dir():
        raise ImportError_(f"{base} is not a folder.")

    found, scanned, skipped = [], 0, []
    for path in sorted(base.rglob("*.py")):
        parts = set(path.parts)
        if parts & {".venv", "venv", "site-packages", "__pycache__", "build",
                    "node_modules", ".git"}:
            continue
        if scanned >= limit:
            break
        scanned += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            skipped.append({"file": str(path.relative_to(base)), "why": f"syntax: {exc.msg}"})
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Attribute):
                    bases.append(b.attr)
                elif isinstance(b, ast.Name):
                    bases.append(b.id)
            if not any(b in ("Module", "Sequential") for b in bases):
                continue
            init = next((f for f in node.body
                         if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
            required = 0
            if init:
                args = init.args
                positional = [a.arg for a in args.args][1:]   # drop self
                required = max(0, len(positional) - len(args.defaults))
            found.append({
                "file": str(path.relative_to(base)),
                "cls": node.name,
                "line": node.lineno,
                "bases": bases,
                "arguments": required,
                "forward": any(isinstance(f, ast.FunctionDef) and f.name == "forward"
                               for f in node.body),
            })
    return {"root": str(base), "files": scanned, "models": found,
            "skipped": skipped[:10]}


def from_folder(root: str, file: str, cls: str,
                input_shape: Optional[List[int]] = None,
                arguments: str = "") -> Dict[str, Any]:
    """Import one class out of a folder, with its own package on the path.

    Importing the module properly rather than exec'ing the file in isolation is
    what makes multi-file projects work: the relative imports inside it resolve
    the way they do when the project runs.
    """
    import importlib.util
    import sys

    base = Path(root).expanduser().resolve()
    target = (base / file).resolve()
    if not target.exists():
        raise ImportError_(f"{file} is not in {base}.")

    added = [str(base)]
    package_root = target.parent
    while (package_root / "__init__.py").exists() and package_root != base:
        package_root = package_root.parent
    added.append(str(package_root))
    for entry in added:
        if entry not in sys.path:
            sys.path.insert(0, entry)

    spec = importlib.util.spec_from_file_location(
        f"scanned_{target.stem}", target)
    if spec is None or spec.loader is None:
        raise ImportError_(f"Could not load {file}.")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise ImportError_(
            f"{file} did not import: {type(exc).__name__}: {exc}. "
            f"If it needs the project installed, install it first.") from exc

    candidate = getattr(module, cls, None)
    if candidate is None:
        raise ImportError_(f"{file} has no class called {cls}.")
    try:
        model = eval(f"candidate({arguments})", {"candidate": candidate,  # noqa: S307
                                                 "module": module})
    except Exception as exc:  # noqa: BLE001
        raise ImportError_(
            f"{cls} could not be built with ({arguments}): {type(exc).__name__}: {exc}")
    graph = from_pytorch(model, name=cls, input_shape=input_shape)
    graph["_entry"] = cls
    return graph


def from_source(source: str, input_shape: Optional[List[int]] = None,
                entry: str = "") -> Dict[str, Any]:
    """Trace a module defined in pasted PyTorch source.

    The code is executed, which is the only way to get a module object to trace —
    a static reader could not tell you what `forward` does. That is the same
    trust assumption as the blocks and recipes folders, and fine for a tool you
    run on your own machine, but it is worth saying out loud.

    Accepts either a class definition or a variable holding a module, and picks
    the last thing defined when there is more than one.
    """
    import torch
    import torch.nn as nn

    namespace: Dict[str, Any] = {
        "__name__": "pasted_model",
        "torch": torch, "nn": nn, "F": torch.nn.functional,
    }
    try:
        exec(compile(source, "<pasted>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        raise ImportError_(f"The code did not run: {type(exc).__name__}: {exc}") from exc

    if entry:
        if entry not in namespace:
            raise ImportError_(f"There is nothing called {entry} in that code.")
        candidate = namespace[entry]
        model = candidate() if isinstance(candidate, type) else candidate
        if not isinstance(model, nn.Module):
            raise ImportError_(f"{entry} is not an nn.Module.")
        return _traced(model, entry, input_shape)

    # a module already built and assigned to a name
    instances = [(key, value) for key, value in namespace.items()
                 if isinstance(value, nn.Module) and not key.startswith("_")]
    if instances:
        key, model = instances[-1]
        return _traced(model, key, input_shape)

    # otherwise a class we can construct with no arguments
    classes = [(key, value) for key, value in namespace.items()
               if isinstance(value, type) and issubclass(value, nn.Module)
               and value is not nn.Module
               and getattr(value, "__module__", "") == "pasted_model"]
    if not classes:
        raise ImportError_(
            "That code defines no nn.Module. Paste a class that subclasses "
            "nn.Module, or assign a model to a variable such as "
            "`model = nn.Sequential(...)`.")

    problems = []
    for key, cls in reversed(classes):
        try:
            return _traced(cls(), key, input_shape)
        except TypeError as exc:
            problems.append(f"{key} needs constructor arguments ({exc})")
        except ImportError_ as exc:
            problems.append(f"{key}: {exc}")
    raise ImportError_(
        "None of the classes could be built with no arguments. "
        + "; ".join(problems[:3])
        + ". Add a line that builds one, like `model = MyNet(3, 10)`.")


def _traced(model, name: str, input_shape: Optional[List[int]]) -> Dict[str, Any]:
    graph = from_pytorch(model, name=name, input_shape=input_shape)
    graph["_entry"] = name
    return graph


def from_torch_file(path: str, input_shape: Optional[List[int]] = None) -> Dict[str, Any]:
    """Load a whole saved module and trace it.

    Note this executes the pickled class definition, which is the same trust
    assumption as running the file yourself.
    """
    import torch

    p = Path(path)
    obj = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        raise ImportError_(
            f"{p.name} holds a state_dict, not a model. A state_dict is only "
            f"weights — it has no architecture to put on the canvas. Save the "
            f"module itself with torch.save(model, path), or import the ONNX export."
        )
    return from_pytorch(obj, name=p.stem, input_shape=input_shape)


# --------------------------------------------------------------------------
# ONNX
# --------------------------------------------------------------------------

_ONNX_ACT = {
    "Relu": "relu", "LeakyRelu": "leaky_relu", "Gelu": "gelu", "Tanh": "tanh",
    "Sigmoid": "sigmoid", "Softmax": "softmax", "Elu": "elu", "HardSigmoid": "sigmoid",
    "HardSwish": "silu", "Mish": "silu", "Identity": "identity",
}


def _attrs(node) -> Dict[str, Any]:
    from onnx import numpy_helper

    out: Dict[str, Any] = {}
    for a in node.attribute:
        if a.type == 1:
            out[a.name] = a.f
        elif a.type == 2:
            out[a.name] = a.i
        elif a.type == 3:
            out[a.name] = a.s.decode(errors="replace")
        elif a.type == 4:
            out[a.name] = numpy_helper.to_array(a.t)
        elif a.type == 6:
            out[a.name] = list(a.floats)
        elif a.type == 7:
            out[a.name] = list(a.ints)
    return out


def from_onnx(path: str) -> Dict[str, Any]:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(path)
    graph = model.graph
    initializers = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

    b = Builder(Path(path).stem or "Imported")
    produced: Dict[str, str] = {}

    for inp in graph.input:
        if inp.name in initializers:
            continue
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        shape = [int(d) for d in dims[1:]] or [1]     # drop the batch axis
        produced[inp.name] = b.add("Input", {"shape": shape, "dtype": "float"},
                                   label=inp.name)

    for node in graph.node:
        op, at = node.op_type, _attrs(node)
        ins = [produced[i] for i in node.input if i in produced]
        weight = next((initializers[i] for i in node.input if i in initializers), None)
        nid = None

        if op == "Conv":
            k = at.get("kernel_shape", [3, 3])[0]
            pads = at.get("pads", [0, 0, 0, 0])
            stride = at.get("strides", [1, 1])[0]
            filters = int(weight.shape[0]) if weight is not None else 1
            padding = "same" if (stride == 1 and pads and pads[0] * 2 + 1 == k) else int(pads[0] if pads else 0)
            nid = b.add("Conv2d", {"filters": filters, "kernel": int(k),
                                   "stride": int(stride), "padding": padding,
                                   "dilation": int(at.get("dilations", [1])[0]),
                                   "groups": int(at.get("group", 1)),
                                   "bias": len(node.input) > 2})
        elif op in ("Gemm", "MatMul"):
            if weight is not None:
                units = int(weight.shape[0] if op == "Gemm" and at.get("transB", 1)
                            else weight.shape[-1])
            else:
                units = 1
            nid = b.add("Linear", {"units": units, "bias": len(node.input) > 2})
        elif op == "BatchNormalization":
            nid = b.add("BatchNorm2d", {"eps": float(at.get("epsilon", 1e-5)),
                                        "momentum": float(at.get("momentum", 0.9))})
        elif op == "MaxPool":
            nid = b.add("MaxPool2d", {"kernel": int(at.get("kernel_shape", [2])[0]),
                                      "stride": int(at.get("strides", [2])[0]),
                                      "padding": int(at.get("pads", [0])[0])})
        elif op == "AveragePool":
            nid = b.add("AvgPool2d", {"kernel": int(at.get("kernel_shape", [2])[0]),
                                      "stride": int(at.get("strides", [2])[0]),
                                      "padding": int(at.get("pads", [0])[0])})
        elif op in ("GlobalAveragePool", "ReduceMean"):
            nid = b.add("GlobalAvgPool", {})
        elif op in _ONNX_ACT:
            nid = b.add("Activation", {"kind": _ONNX_ACT[op]})
        elif op == "Flatten":
            nid = b.add("Flatten", {})
        elif op == "Reshape":
            nid = b.add("Reshape", {"shape": [-1]})
        elif op == "Concat":
            nid = b.add("Concat", {"axis": max(0, int(at.get("axis", 1)) - 1)})
        elif op == "Add" and len(ins) > 1:
            nid = b.add("Add", {})
        elif op == "Mul" and len(ins) > 1:
            nid = b.add("Multiply", {})
        elif op == "Dropout":
            nid = b.add("Dropout", {"rate": float(at.get("ratio", 0.5))})
        elif op in ("Resize", "Upsample"):
            nid = b.add("Upsample2d", {"scale": 2, "mode": "nearest"})
        elif op in ("Constant", "Shape", "Unsqueeze", "Squeeze", "Cast", "Gather",
                    "Slice", "Transpose", "Identity", "Pad", "Clip"):
            # plumbing ONNX emits around real layers; pass the tensor through
            if ins:
                for out_name in node.output:
                    produced[out_name] = ins[0]
            continue
        else:
            nid = b.opaque(op, f"# ONNX {op}", "no layer equivalent in the registry")

        for i, src in enumerate(ins):
            b.link(src, nid, i)
        for out_name in node.output:
            produced[out_name] = nid

    for out in graph.output:
        src = produced.get(out.name)
        if src:
            node_id = b.add("Output", {"task": "classification"})
            b.link(src, node_id, 0)

    result = b.layout()
    result["_notes"] = b.notes
    return result
