"""Layer registry.

One place defines everything about a node type: its editable parameters, how it
transforms an activation shape, and how it turns into PyTorch and Keras code.
Adding a layer means adding one LayerSpec here and nothing else.

Shape convention (the IR): channels-first, batch dimension omitted.
  images    -> [C, H, W]
  sequences -> [L, C]
  vectors   -> [F]
  token ids -> [L]
Keras codegen converts to channels-last at the boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

Shape = List[int]


class ShapeError(ValueError):
    """Raised when a layer cannot accept the shape it was handed."""


# --------------------------------------------------------------------------
# parameter descriptors (drive the inspector panel in the UI)
# --------------------------------------------------------------------------

def P(name, kind, default, **kw) -> Dict[str, Any]:
    d = {"name": name, "kind": kind, "default": default}
    d.update(kw)
    return d


@dataclass
class LayerSpec:
    name: str
    category: str
    params: List[Dict[str, Any]] = field(default_factory=list)
    n_inputs: int = 1   # -1 = two or more (merges), -2 = one or more (custom)
    infer: Optional[Callable] = None       # (p, in_shapes) -> Shape
    torch_init: Optional[Callable] = None  # (p, in_shapes) -> str | None
    torch_call: Optional[Callable] = None  # (p, ins, mod, in_shapes) -> str
    keras_call: Optional[Callable] = None  # (p, ins, in_shapes) -> str; None = no equivalent
    torch_prelude: str = ""
    keras_prelude: str = ""
    doc: str = ""
    kind: str = "layer"                    # layer | runtime
    runtime_init: Optional[Callable] = None
    runtime_name: Optional[Callable] = None
    learnables: Optional[Callable] = None  # (p, in_shapes, out_shape) -> int
    learnables_approx: bool = False        # True when the count is a published figure
    torch_class: Optional[str] = None      # for a link to the PyTorch reference
    source: str = "core"                   # core | block — where the spec came from
    origin: Optional[str] = None           # block filename, for the Edit button

    def defaults(self) -> Dict[str, Any]:
        return {q["name"]: q["default"] for q in self.params}


REGISTRY: Dict[str, LayerSpec] = {}


def register(spec: LayerSpec) -> LayerSpec:
    REGISTRY[spec.name] = spec
    return spec


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _pair(v) -> tuple:
    if isinstance(v, (list, tuple)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def conv_out(size: int, k: int, s: int, pad, d: int = 1) -> int:
    if pad == "same":
        if s != 1:
            # Keras allows this, PyTorch does not. Since the generated PyTorch is
            # what trains here, accepting it would mean the canvas approving a
            # model that cannot be built.
            raise ShapeError(
                f"padding 'same' does not work with stride {s} in PyTorch. Use an "
                f"explicit padding of {(k - 1) // 2} for a {k}x{k} kernel, which "
                f"gives the same result."
            )
        return math.ceil(size / s)
    p = int(pad)
    out = (size + 2 * p - d * (k - 1) - 1) // s + 1
    if out < 1:
        raise ShapeError(
            f"output size collapses to {out}: input {size}, kernel {k}, "
            f"stride {s}, padding {p}"
        )
    return out


def need_rank(shape: Shape, rank: int, layer: str, hint: str) -> None:
    if len(shape) != rank:
        raise ShapeError(
            f"{layer} needs a rank-{rank} activation {hint}, got {list(shape)}"
        )


def _pad_arg(pad) -> str:
    return "'same'" if pad == "same" else str(int(pad))


def _q(value) -> str:
    """Double-quoted literal. Kept out of f-strings so this file parses on Python < 3.12."""
    return '"' + str(value) + '"'


def _kpad(p) -> str:
    return _q("same") if p.get("padding") == "same" else _q("valid")


def prod(shape: Shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


# --------------------------------------------------------------------------
# input / output
# --------------------------------------------------------------------------

register(LayerSpec(
    name="Input",
    category="Data",
    n_inputs=0,
    doc="Entry point. Shape is channels-first and excludes the batch dimension.",
    params=[
        P("shape", "shape", [3, 32, 32], help="e.g. 3,32,32 for RGB images or 784 for a flat vector"),
        P("dtype", "select", "float", options=["float", "long"],
          help="Use long for token ids feeding an Embedding"),
    ],
    infer=lambda p, ins: [int(x) for x in p["shape"]],
    torch_init=lambda p, ins: None,
    torch_call=lambda p, ins, mod, s: ins[0],
    keras_call=lambda p, ins, s: None,
))

register(LayerSpec(
    name="Output",
    category="Data",
    doc="Marks the tensor the loss is computed on.",
    params=[
        P("task", "select", "classification",
          options=["classification", "regression", "binary", "language_modeling"],
          help="Selects the loss used during training"),
    ],
    infer=lambda p, ins: list(ins[0]),
    torch_init=lambda p, ins: None,
    torch_call=lambda p, ins, mod, s: ins[0],
    keras_call=lambda p, ins, s: ins[0],
))


# --------------------------------------------------------------------------
# dense
# --------------------------------------------------------------------------

def _linear_infer(p, ins):
    s = ins[0]
    if len(s) == 0:
        raise ShapeError("Linear needs at least one dimension")
    return list(s[:-1]) + [int(p["units"])]


register(LayerSpec(
    name="Linear",
    category="Dense",
    doc="Fully connected layer applied over the last dimension.",
    params=[
        P("units", "int", 128, min=1),
        P("bias", "bool", True),
    ],
    infer=_linear_infer,
    torch_init=lambda p, ins: f"nn.Linear({ins[0][-1]}, {int(p['units'])}, bias={bool(p['bias'])})",
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: f"layers.Dense({int(p['units'])}, use_bias={bool(p['bias'])})({ins[0]})",
))


# --------------------------------------------------------------------------
# convolution
# --------------------------------------------------------------------------

def _conv2d_infer(p, ins):
    s = ins[0]
    need_rank(s, 3, "Conv2d", "[C, H, W]")
    kh, kw = _pair(p["kernel"])
    sh, sw = _pair(p["stride"])
    d = int(p["dilation"])
    return [int(p["filters"]),
            conv_out(s[1], kh, sh, p["padding"], d),
            conv_out(s[2], kw, sw, p["padding"], d)]


register(LayerSpec(
    name="Conv2d",
    category="Convolution",
    doc="2-D convolution over [C, H, W].",
    params=[
        P("filters", "int", 32, min=1),
        P("kernel", "int", 3, min=1),
        P("stride", "int", 1, min=1),
        P("padding", "padding", "same"),
        P("dilation", "int", 1, min=1),
        P("groups", "int", 1, min=1),
        P("bias", "bool", True),
    ],
    infer=_conv2d_infer,
    torch_init=lambda p, ins: (
        f"nn.Conv2d({ins[0][0]}, {int(p['filters'])}, kernel_size={int(p['kernel'])}, "
        f"stride={int(p['stride'])}, padding={_pad_arg(p['padding'])}, "
        f"dilation={int(p['dilation'])}, groups={int(p['groups'])}, bias={bool(p['bias'])})"
    ),
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: (
        f"layers.Conv2D({int(p['filters'])}, {int(p['kernel'])}, "
        f"strides={int(p['stride'])}, padding={_kpad(p)}, "
        f"dilation_rate={int(p['dilation'])}, groups={int(p['groups'])}, "
        f"use_bias={bool(p['bias'])})({ins[0]})"
    ),
))


def _conv1d_infer(p, ins):
    s = ins[0]
    need_rank(s, 2, "Conv1d", "[L, C]")
    return [conv_out(s[0], int(p["kernel"]), int(p["stride"]), p["padding"], int(p["dilation"])),
            int(p["filters"])]


register(LayerSpec(
    name="Conv1d",
    category="Convolution",
    doc="1-D convolution over [L, C]. PyTorch code transposes to [C, L] internally.",
    params=[
        P("filters", "int", 64, min=1),
        P("kernel", "int", 3, min=1),
        P("stride", "int", 1, min=1),
        P("padding", "padding", "same"),
        P("dilation", "int", 1, min=1),
    ],
    infer=_conv1d_infer,
    torch_init=lambda p, ins: (
        f"nn.Conv1d({ins[0][1]}, {int(p['filters'])}, kernel_size={int(p['kernel'])}, "
        f"stride={int(p['stride'])}, padding={_pad_arg(p['padding'])}, "
        f"dilation={int(p['dilation'])})"
    ),
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]}.transpose(1, 2)).transpose(1, 2)",
    keras_call=lambda p, ins, s: (
        f"layers.Conv1D({int(p['filters'])}, {int(p['kernel'])}, strides={int(p['stride'])}, "
        f"padding={_kpad(p)}, "
        f"dilation_rate={int(p['dilation'])})({ins[0]})"
    ),
))


def _deconv_infer(p, ins):
    s = ins[0]
    need_rank(s, 3, "ConvTranspose2d", "[C, H, W]")
    k, st = int(p["kernel"]), int(p["stride"])
    pad = 0 if p["padding"] == "same" else int(p["padding"])
    op = int(p["output_padding"])
    out_h = (s[1] - 1) * st - 2 * pad + k + op
    out_w = (s[2] - 1) * st - 2 * pad + k + op
    return [int(p["filters"]), out_h, out_w]


register(LayerSpec(
    name="ConvTranspose2d",
    category="Convolution",
    doc="Learned upsampling. Common in decoders and GAN generators.",
    params=[
        P("filters", "int", 32, min=1),
        P("kernel", "int", 4, min=1),
        P("stride", "int", 2, min=1),
        P("padding", "int", 1, min=0),
        P("output_padding", "int", 0, min=0),
    ],
    infer=_deconv_infer,
    torch_init=lambda p, ins: (
        f"nn.ConvTranspose2d({ins[0][0]}, {int(p['filters'])}, kernel_size={int(p['kernel'])}, "
        f"stride={int(p['stride'])}, padding={int(p['padding'])}, "
        f"output_padding={int(p['output_padding'])})"
    ),
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: (
        f"layers.Conv2DTranspose({int(p['filters'])}, {int(p['kernel'])}, "
        f"strides={int(p['stride'])}, padding={_q('same')})({ins[0]})"
    ),
))


register(LayerSpec(
    name="SeparableConv2d",
    category="Convolution",
    doc="Depthwise convolution followed by a pointwise mix. Cheap replacement for Conv2d.",
    params=[
        P("filters", "int", 32, min=1),
        P("kernel", "int", 3, min=1),
        P("stride", "int", 1, min=1),
        P("padding", "padding", "same"),
    ],
    infer=lambda p, ins: _conv2d_infer({**p, "dilation": 1}, ins),
    torch_init=lambda p, ins: (
        "nn.Sequential(nn.Conv2d({c}, {c}, kernel_size={k}, stride={s}, padding={pad}, groups={c}), "
        "nn.Conv2d({c}, {f}, kernel_size=1))".format(
            c=ins[0][0], f=int(p["filters"]), k=int(p["kernel"]),
            s=int(p["stride"]), pad=_pad_arg(p["padding"]))
    ),
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: (
        f"layers.SeparableConv2D({int(p['filters'])}, {int(p['kernel'])}, "
        f"strides={int(p['stride'])}, "
        f"padding={_kpad(p)})({ins[0]})"
    ),
))


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------

def _pool2d_infer(p, ins):
    s = ins[0]
    need_rank(s, 3, "Pool2d", "[C, H, W]")
    k = int(p["kernel"])
    st = int(p["stride"]) if int(p["stride"]) > 0 else k
    return [s[0], conv_out(s[1], k, st, p["padding"]), conv_out(s[2], k, st, p["padding"])]


for _pool, _torch, _keras in (("MaxPool2d", "nn.MaxPool2d", "layers.MaxPooling2D"),
                              ("AvgPool2d", "nn.AvgPool2d", "layers.AveragePooling2D")):
    register(LayerSpec(
        name=_pool,
        category="Pooling",
        doc="Downsamples spatially. Stride 0 means 'same as kernel'.",
        params=[
            P("kernel", "int", 2, min=1),
            P("stride", "int", 0, min=0),
            P("padding", "int", 0, min=0),
        ],
        infer=_pool2d_infer,
        torch_init=(lambda t: lambda p, ins: (
            f"{t}(kernel_size={int(p['kernel'])}, "
            f"stride={int(p['stride']) or int(p['kernel'])}, padding={int(p['padding'])})"
        ))(_torch),
        torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
        keras_call=(lambda k: lambda p, ins, s: (
            f"{k}(pool_size={int(p['kernel'])}, "
            f"strides={int(p['stride']) or int(p['kernel'])}, padding={_q('valid')})({ins[0]})"
        ))(_keras),
    ))


register(LayerSpec(
    name="MaxPool1d",
    category="Pooling",
    doc="Downsamples a sequence along its length.",
    params=[P("kernel", "int", 2, min=1), P("stride", "int", 0, min=0)],
    infer=lambda p, ins: (
        need_rank(ins[0], 2, "MaxPool1d", "[L, C]") or
        [conv_out(ins[0][0], int(p["kernel"]), int(p["stride"]) or int(p["kernel"]), 0), ins[0][1]]
    ),
    torch_init=lambda p, ins: (
        f"nn.MaxPool1d(kernel_size={int(p['kernel'])}, "
        f"stride={int(p['stride']) or int(p['kernel'])})"
    ),
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]}.transpose(1, 2)).transpose(1, 2)",
    keras_call=lambda p, ins, s: (
        f"layers.MaxPooling1D(pool_size={int(p['kernel'])}, "
        f"strides={int(p['stride']) or int(p['kernel'])})({ins[0]})"
    ),
))


register(LayerSpec(
    name="AdaptiveAvgPool2d",
    category="Pooling",
    doc="Pools to a fixed output grid regardless of input size.",
    params=[P("size", "int", 1, min=1)],
    infer=lambda p, ins: (
        need_rank(ins[0], 3, "AdaptiveAvgPool2d", "[C, H, W]") or
        [ins[0][0], int(p["size"]), int(p["size"])]
    ),
    torch_init=lambda p, ins: f"nn.AdaptiveAvgPool2d({int(p['size'])})",
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: (
        f"layers.Resizing({int(p['size'])}, {int(p['size'])}, interpolation={_q('bilinear')})({ins[0]})"
    ),
))


def _gap_infer(p, ins):
    s = ins[0]
    if len(s) == 3:
        return [s[0]]
    if len(s) == 2:
        return [s[1]]
    raise ShapeError(f"GlobalAvgPool needs [C, H, W] or [L, C], got {list(s)}")


register(LayerSpec(
    name="GlobalAvgPool",
    category="Pooling",
    doc="Averages away every spatial or temporal position. A cheap alternative to Flatten.",
    params=[],
    infer=_gap_infer,
    torch_init=lambda p, ins: None,
    torch_call=lambda p, ins, mod, s: (
        f"{ins[0]}.mean(dim=(2, 3))" if len(s[0]) == 3 else f"{ins[0]}.mean(dim=1)"
    ),
    keras_call=lambda p, ins, s: (
        f"layers.GlobalAveragePooling2D()({ins[0]})" if len(s[0]) == 3
        else f"layers.GlobalAveragePooling1D()({ins[0]})"
    ),
))


register(LayerSpec(
    name="Upsample2d",
    category="Pooling",
    doc="Resizes by a whole-number factor. No parameters to learn.",
    params=[
        P("scale", "int", 2, min=2),
        P("mode", "select", "nearest", options=["nearest", "bilinear"]),
    ],
    infer=lambda p, ins: (
        need_rank(ins[0], 3, "Upsample2d", "[C, H, W]") or
        [ins[0][0], ins[0][1] * int(p["scale"]), ins[0][2] * int(p["scale"])]
    ),
    torch_init=lambda p, ins: (
        f"nn.Upsample(scale_factor={int(p['scale'])}, mode='{p['mode']}')"
    ),
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: (
        f"layers.UpSampling2D(size={int(p['scale'])}, "
        f"interpolation={_q(p['mode'])})({ins[0]})"
    ),
))


# --------------------------------------------------------------------------
# normalization and regularization
# --------------------------------------------------------------------------

register(LayerSpec(
    name="BatchNorm2d",
    category="Normalization",
    doc="Normalizes each channel across the batch. Put it between a convolution and its activation.",
    params=[P("momentum", "float", 0.1), P("eps", "float", 1e-5)],
    infer=lambda p, ins: (
        need_rank(ins[0], 3, "BatchNorm2d", "[C, H, W]") or list(ins[0])
    ),
    torch_init=lambda p, ins: (
        f"nn.BatchNorm2d({ins[0][0]}, eps={float(p['eps'])}, momentum={float(p['momentum'])})"
    ),
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: (
        f"layers.BatchNormalization(momentum={1 - float(p['momentum'])}, "
        f"epsilon={float(p['eps'])})({ins[0]})"
    ),
))

register(LayerSpec(
    name="BatchNorm1d",
    category="Normalization",
    doc="Batch normalization for vectors or sequences.",
    params=[P("momentum", "float", 0.1), P("eps", "float", 1e-5)],
    infer=lambda p, ins: list(ins[0]),
    torch_init=lambda p, ins: f"nn.BatchNorm1d({ins[0][-1]}, eps={float(p['eps'])})",
    torch_call=lambda p, ins, mod, s: (
        f"{mod}({ins[0]})" if len(s[0]) == 1
        else f"{mod}({ins[0]}.transpose(1, 2)).transpose(1, 2)"
    ),
    keras_call=lambda p, ins, s: f"layers.BatchNormalization()({ins[0]})",
))

register(LayerSpec(
    name="LayerNorm",
    category="Normalization",
    doc="Normalizes each sample over its last dimension. The default in transformers.",
    params=[P("eps", "float", 1e-5)],
    infer=lambda p, ins: list(ins[0]),
    torch_init=lambda p, ins: f"nn.LayerNorm({ins[0][-1]}, eps={float(p['eps'])})",
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: f"layers.LayerNormalization(epsilon={float(p['eps'])})({ins[0]})",
))


def _groupnorm_infer(p, ins):
    s = ins[0]
    need_rank(s, 3, "GroupNorm", "[C, H, W]")
    g = int(p["groups"])
    if s[0] % g:
        raise ShapeError(f"GroupNorm: {s[0]} channels is not divisible by {g} groups")
    return list(s)


register(LayerSpec(
    name="GroupNorm",
    category="Normalization",
    doc="Normalizes within channel groups. Stable at small batch sizes.",
    params=[P("groups", "int", 8, min=1)],
    infer=_groupnorm_infer,
    torch_init=lambda p, ins: f"nn.GroupNorm({int(p['groups'])}, {ins[0][0]})",
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: f"layers.GroupNormalization(groups={int(p['groups'])})({ins[0]})",
))

register(LayerSpec(
    name="Dropout",
    category="Normalization",
    doc="Zeros random activations during training only.",
    params=[P("rate", "float", 0.5, min=0.0, max=0.95)],
    infer=lambda p, ins: list(ins[0]),
    torch_init=lambda p, ins: f"nn.Dropout({float(p['rate'])})",
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: f"layers.Dropout({float(p['rate'])})({ins[0]})",
))

register(LayerSpec(
    name="Dropout2d",
    category="Normalization",
    doc="Drops whole feature maps. Better than plain dropout after convolutions.",
    params=[P("rate", "float", 0.25, min=0.0, max=0.95)],
    infer=lambda p, ins: (
        need_rank(ins[0], 3, "Dropout2d", "[C, H, W]") or list(ins[0])
    ),
    torch_init=lambda p, ins: f"nn.Dropout2d({float(p['rate'])})",
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: f"layers.SpatialDropout2D({float(p['rate'])})({ins[0]})",
))


# --------------------------------------------------------------------------
# activations
# --------------------------------------------------------------------------

_ACT_TORCH = {
    "relu": "nn.ReLU()", "leaky_relu": "nn.LeakyReLU(0.01)", "gelu": "nn.GELU()",
    "silu": "nn.SiLU()", "tanh": "nn.Tanh()", "sigmoid": "nn.Sigmoid()",
    "elu": "nn.ELU()", "softmax": "nn.Softmax(dim=-1)", "identity": "nn.Identity()",
}
_ACT_KERAS = {
    "relu": "relu", "leaky_relu": "leaky_relu", "gelu": "gelu", "silu": "silu",
    "tanh": "tanh", "sigmoid": "sigmoid", "elu": "elu", "softmax": "softmax",
    "identity": "linear",
}

register(LayerSpec(
    name="Activation",
    category="Activation",
    doc="Elementwise nonlinearity. Leave the final layer linear when training with cross-entropy.",
    params=[P("kind", "select", "relu", options=list(_ACT_TORCH.keys()))],
    infer=lambda p, ins: list(ins[0]),
    torch_init=lambda p, ins: _ACT_TORCH[p["kind"]],
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: f"layers.Activation({_q(_ACT_KERAS[p['kind']])})({ins[0]})",
))


# --------------------------------------------------------------------------
# shape surgery
# --------------------------------------------------------------------------

register(LayerSpec(
    name="Flatten",
    category="Shape",
    doc="Collapses everything after the batch dimension into one vector.",
    params=[],
    infer=lambda p, ins: [prod(ins[0])],
    torch_init=lambda p, ins: "nn.Flatten()",
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: f"layers.Flatten()({ins[0]})",
))


def _reshape_infer(p, ins):
    target = [int(x) for x in p["shape"]]
    total = prod(ins[0])
    if -1 in target:
        if target.count(-1) > 1:
            raise ShapeError("Reshape allows at most one -1")
        known = prod([d for d in target if d != -1])
        if known == 0 or total % known:
            raise ShapeError(f"cannot reshape {prod(ins[0])} elements into {target}")
        return [total // known if d == -1 else d for d in target]
    if prod(target) != total:
        raise ShapeError(
            f"Reshape target {target} holds {prod(target)} elements but the input has {total}"
        )
    return target


register(LayerSpec(
    name="Reshape",
    category="Shape",
    doc="Reinterprets the same values under a new shape. Use -1 for one inferred dimension.",
    params=[P("shape", "shape", [-1], help="e.g. 64,7,7")],
    infer=_reshape_infer,
    torch_init=lambda p, ins: None,
    torch_call=lambda p, ins, mod, s: (
        f"{ins[0]}.reshape({ins[0]}.size(0), " + ", ".join(str(int(x)) for x in p["shape"]) + ")"
    ),
    keras_call=lambda p, ins, s: (
        "layers.Reshape((" + ", ".join(str(int(x)) for x in _reshape_infer(p, s)) + f"))({ins[0]})"
    ),
))


def _permute_infer(p, ins):
    order = [int(x) for x in p["order"]]
    s = ins[0]
    if sorted(order) != list(range(len(s))):
        raise ShapeError(
            f"Permute order {order} must be a rearrangement of 0..{len(s) - 1}"
        )
    return [s[i] for i in order]


register(LayerSpec(
    name="Permute",
    category="Shape",
    doc="Reorders dimensions. Indices exclude the batch dimension.",
    params=[P("order", "shape", [1, 0], help="e.g. 1,0 to swap [C,H,W] axes")],
    infer=_permute_infer,
    torch_init=lambda p, ins: None,
    torch_call=lambda p, ins, mod, s: (
        f"{ins[0]}.permute(0, " + ", ".join(str(int(x) + 1) for x in p["order"]) + ")"
    ),
    keras_call=lambda p, ins, s: (
        "layers.Permute((" + ", ".join(str(int(x) + 1) for x in p["order"]) + f"))({ins[0]})"
    ),
))


# --------------------------------------------------------------------------
# merges
# --------------------------------------------------------------------------

def _elementwise_infer(name):
    """Broadcasting rules, as PyTorch applies them.

    Axes must match or be 1. This is what lets a [C, 1, 1] channel weight
    multiply a [C, H, W] feature map, which is how every squeeze-excite and
    gating block is built.
    """

    def infer(p, ins):
        rank = len(ins[0])
        for other in ins[1:]:
            if len(other) != rank:
                raise ShapeError(
                    f"{name} needs inputs of the same rank, got "
                    f"{list(ins[0])} and {list(other)}"
                )
        out = []
        for axis in range(rank):
            sizes = {int(s[axis]) for s in ins}
            sizes.discard(1)
            if len(sizes) > 1:
                raise ShapeError(
                    f"{name}: axis {axis} has sizes {sorted(sizes)}. They must "
                    f"match, or be 1 to broadcast."
                )
            out.append(sizes.pop() if sizes else 1)
        return out

    return infer


register(LayerSpec(
    name="Add",
    category="Merge",
    n_inputs=-1,
    doc="Elementwise sum, with broadcasting. This is how you build a residual connection.",
    params=[],
    infer=_elementwise_infer("Add"),
    torch_init=lambda p, ins: None,
    torch_call=lambda p, ins, mod, s: " + ".join(ins),
    keras_call=lambda p, ins, s: f"layers.Add()([{', '.join(ins)}])",
))

register(LayerSpec(
    name="Multiply",
    category="Merge",
    n_inputs=-1,
    doc="Elementwise product, with broadcasting. A [C, 1, 1] weight against a "
        "[C, H, W] map is the squeeze-excite pattern.",
    params=[],
    infer=_elementwise_infer("Multiply"),
    torch_init=lambda p, ins: None,
    torch_call=lambda p, ins, mod, s: " * ".join(ins),
    keras_call=lambda p, ins, s: f"layers.Multiply()([{', '.join(ins)}])",
))


def _concat_infer(p, ins):
    axis = int(p["axis"])
    first = list(ins[0])
    if not (0 <= axis < len(first)):
        raise ShapeError(f"Concat axis {axis} is out of range for shape {first}")
    total = 0
    for other in ins:
        o = list(other)
        if len(o) != len(first):
            raise ShapeError(f"Concat inputs differ in rank: {first} vs {o}")
        for i, (a, b) in enumerate(zip(first, o)):
            if i != axis and a != b:
                raise ShapeError(
                    f"Concat inputs must match on every axis except {axis}: {first} vs {o}"
                )
        total += o[axis]
    out = list(first)
    out[axis] = total
    return out


register(LayerSpec(
    name="Concat",
    category="Merge",
    n_inputs=-1,
    doc="Joins inputs along one axis. Axis 0 is channels for images.",
    params=[P("axis", "int", 0, min=0)],
    infer=_concat_infer,
    torch_init=lambda p, ins: None,
    torch_call=lambda p, ins, mod, s: (
        f"torch.cat([{', '.join(ins)}], dim={int(p['axis']) + 1})"
    ),
    keras_call=lambda p, ins, s: (
        f"layers.Concatenate(axis=-1)([{', '.join(ins)}])"
    ),
))


# --------------------------------------------------------------------------
# recurrent
# --------------------------------------------------------------------------

def _rnn_infer(name):
    def infer(p, ins):
        s = ins[0]
        need_rank(s, 2, name, "[L, C]")
        mult = 2 if p.get("bidirectional") else 1
        units = int(p["units"]) * mult
        return [s[0], units] if p["return_sequences"] else [units]
    return infer


for _rnn, _t, _k in (("LSTM", "nn.LSTM", "layers.LSTM"),
                     ("GRU", "nn.GRU", "layers.GRU"),
                     ("SimpleRNN", "nn.RNN", "layers.SimpleRNN")):
    register(LayerSpec(
        name=_rnn,
        category="Recurrent",
        doc="Processes a sequence step by step. Turn off return_sequences to keep only the last step.",
        params=[
            P("units", "int", 128, min=1),
            P("num_layers", "int", 1, min=1),
            P("bidirectional", "bool", False),
            P("return_sequences", "bool", True),
            P("dropout", "float", 0.0, min=0.0, max=0.9),
        ],
        infer=_rnn_infer(_rnn),
        torch_init=(lambda t: lambda p, ins: (
            f"{t}({ins[0][1]}, {int(p['units'])}, num_layers={int(p['num_layers'])}, "
            f"batch_first=True, bidirectional={bool(p['bidirectional'])}, "
            f"dropout={float(p['dropout']) if int(p['num_layers']) > 1 else 0.0})"
        ))(_t),
        torch_call=lambda p, ins, mod, s: (
            f"{mod}({ins[0]})[0]" if p["return_sequences"] else f"{mod}({ins[0]})[0][:, -1]"
        ),
        keras_call=(lambda k: lambda p, ins, s: (
            (f"layers.Bidirectional({k}({int(p['units'])}, "
             f"return_sequences={bool(p['return_sequences'])}))({ins[0]})")
            if p["bidirectional"] else
            (f"{k}({int(p['units'])}, "
             f"return_sequences={bool(p['return_sequences'])})({ins[0]})")
        ))(_k),
    ))


# --------------------------------------------------------------------------
# embeddings and attention
# --------------------------------------------------------------------------

def _embed_infer(p, ins):
    s = ins[0]
    if len(s) != 1:
        raise ShapeError(f"Embedding expects [L] token ids, got {list(s)}")
    return [s[0], int(p["dim"])]


register(LayerSpec(
    name="Embedding",
    category="Sequence",
    doc="Looks up a learned vector per token id. Feed it an Input with dtype long.",
    params=[P("vocab", "int", 10000, min=1), P("dim", "int", 256, min=1)],
    infer=_embed_infer,
    torch_init=lambda p, ins: f"nn.Embedding({int(p['vocab'])}, {int(p['dim'])})",
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: (
        f"layers.Embedding({int(p['vocab'])}, {int(p['dim'])})({ins[0]})"
    ),
))


_POS_TORCH = '''class SinusoidalPositionalEncoding(nn.Module):
    """Adds fixed sinusoidal position information to a [B, L, C] tensor."""

    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)
'''

_POS_KERAS = '''class SinusoidalPositionalEncoding(keras.layers.Layer):
    """Adds fixed sinusoidal position information to a [B, L, C] tensor."""

    def build(self, input_shape):
        length, dim = int(input_shape[1]), int(input_shape[2])
        pos = np.arange(length)[:, None]
        div = np.exp(np.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe = np.zeros((length, dim), dtype="float32")
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.pe = self.add_weight(
            shape=(length, dim), initializer=keras.initializers.Constant(pe),
            trainable=False, name="pe")

    def call(self, x):
        return x + self.pe
'''

register(LayerSpec(
    name="PositionalEncoding",
    category="Sequence",
    doc="Gives the model a sense of token order. Insert right after Embedding.",
    params=[],
    infer=lambda p, ins: (
        need_rank(ins[0], 2, "PositionalEncoding", "[L, C]") or list(ins[0])
    ),
    torch_init=lambda p, ins: f"SinusoidalPositionalEncoding({ins[0][1]})",
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: f"SinusoidalPositionalEncoding()({ins[0]})",
    torch_prelude=_POS_TORCH,
    keras_prelude=_POS_KERAS,
))


def _attn_infer(p, ins):
    s = ins[0]
    need_rank(s, 2, "SelfAttention", "[L, C]")
    h = int(p["heads"])
    if s[1] % h:
        raise ShapeError(f"SelfAttention: {s[1]} channels is not divisible by {h} heads")
    return list(s)


register(LayerSpec(
    name="SelfAttention",
    category="Sequence",
    doc="Each position attends to every other position. Channels must divide evenly by heads.",
    params=[
        P("heads", "int", 8, min=1),
        P("dropout", "float", 0.0, min=0.0, max=0.9),
        P("causal", "bool", False, help="Mask out future positions, as in a decoder"),
    ],
    infer=_attn_infer,
    torch_init=lambda p, ins: (
        f"nn.MultiheadAttention({ins[0][1]}, {int(p['heads'])}, "
        f"dropout={float(p['dropout'])}, batch_first=True)"
    ),
    torch_call=lambda p, ins, mod, s: (
        f"{mod}({ins[0]}, {ins[0]}, {ins[0]}, need_weights=False, is_causal={bool(p['causal'])})[0]"
    ),
    keras_call=lambda p, ins, s: (
        f"layers.MultiHeadAttention(num_heads={int(p['heads'])}, "
        f"key_dim={s[0][1] // int(p['heads'])}, dropout={float(p['dropout'])})"
        f"({ins[0]}, {ins[0]}, use_causal_mask={bool(p['causal'])})"
    ),
))


_TX_KERAS = '''def transformer_encoder_block(x, heads, ff_dim, rate=0.0):
    """Pre-norm transformer encoder block: attention, then feed-forward."""
    dim = x.shape[-1]
    h = keras.layers.LayerNormalization(epsilon=1e-5)(x)
    h = keras.layers.MultiHeadAttention(
        num_heads=heads, key_dim=dim // heads, dropout=rate)(h, h)
    x = keras.layers.Add()([x, h])
    h = keras.layers.LayerNormalization(epsilon=1e-5)(x)
    h = keras.layers.Dense(ff_dim, activation="gelu")(h)
    h = keras.layers.Dropout(rate)(h)
    h = keras.layers.Dense(dim)(h)
    return keras.layers.Add()([x, h])
'''

register(LayerSpec(
    name="TransformerEncoder",
    category="Sequence",
    doc="A stack of attention plus feed-forward blocks with residual connections.",
    params=[
        P("heads", "int", 8, min=1),
        P("ff_dim", "int", 512, min=1),
        P("depth", "int", 2, min=1),
        P("dropout", "float", 0.1, min=0.0, max=0.9),
    ],
    infer=_attn_infer,
    torch_init=lambda p, ins: (
        "nn.TransformerEncoder(nn.TransformerEncoderLayer("
        f"d_model={ins[0][1]}, nhead={int(p['heads'])}, "
        f"dim_feedforward={int(p['ff_dim'])}, dropout={float(p['dropout'])}, "
        f"activation='gelu', batch_first=True, norm_first=True), "
        f"num_layers={int(p['depth'])})"
    ),
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: (
        f"_stack_blocks({ins[0]}, {int(p['depth'])}, {int(p['heads'])}, "
        f"{int(p['ff_dim'])}, {float(p['dropout'])})"
    ),
    keras_prelude=_TX_KERAS + '''

def _stack_blocks(x, depth, heads, ff_dim, rate):
    for _ in range(depth):
        x = transformer_encoder_block(x, heads, ff_dim, rate)
    return x
''',
))


# --------------------------------------------------------------------------
# custom node
# --------------------------------------------------------------------------

_SAFE_EVAL = {
    "__builtins__": {},
    "int": int, "len": len, "sum": sum, "min": min, "max": max, "abs": abs,
    "list": list, "range": range, "round": round, "math": math,
}


def _custom_infer(p, ins):
    rule = (p.get("shape_rule") or "").strip()
    if not rule:
        return list(ins[0])
    env = dict(_SAFE_EVAL)
    env["shape"] = list(ins[0])
    env["shapes"] = [list(s) for s in ins]
    env["p"] = p.get("values") or {}
    try:
        out = eval(rule, env)  # noqa: S307 - restricted namespace, local single-user tool
    except ShapeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ShapeError(f"shape rule failed: {exc}") from exc
    if not isinstance(out, (list, tuple)) or not all(isinstance(d, int) for d in out):
        raise ShapeError("shape rule must return a list of integers")
    return [int(d) for d in out]


def _custom_render(template: str, p, in_shapes) -> str:
    env = dict(_SAFE_EVAL)
    env["shape"] = list(in_shapes[0]) if in_shapes else []
    env["shapes"] = [list(s) for s in in_shapes]
    env["p"] = p.get("values") or {}
    out, i = [], 0
    while i < len(template):
        if template[i] == "{" and template[i + 1:i + 2] == "{":
            out.append("{")      # {{ is a literal brace
            i += 2
            continue
        if template[i] == "}" and template[i + 1:i + 2] == "}":
            out.append("}")
            i += 2
            continue
        if template[i] == "{":
            j = template.find("}", i)
            if j == -1:
                raise ShapeError("unclosed { in code template")
            expr = template[i + 1:j]
            try:
                out.append(str(eval(expr, env)))  # noqa: S307
            except Exception as exc:  # noqa: BLE001
                raise ShapeError(f"template expression {{{expr}}} failed: {exc}") from exc
            i = j + 1
        else:
            out.append(template[i])
            i += 1
    return "".join(out)


register(LayerSpec(
    name="Custom",
    category="Custom",
    n_inputs=-2,
    doc=("Your own layer. Write a shape rule and a code snippet per framework. "
         "Inside braces you can use shape, shapes and p."),
    params=[
        P("label", "text", "MyLayer", help="Name shown on the node"),
        P("shape_rule", "code", "list(shape)",
          help="Python expression returning the output shape, e.g. [p['dim']] + shape[1:]"),
        P("torch_code", "code", "nn.Identity()",
          help="Module constructor, e.g. MyLayer(dim={p['dim']})"),
        P("keras_code", "code", "layers.Activation(\"linear\")",
          help="Layer constructor; it is called on the input for you"),
        P("imports", "code", "",
          help="Extra import lines placed at the top of the generated file"),
        P("values", "dict", {}, help="Your own named numbers, available as p['name']"),
    ],
    infer=_custom_infer,
    torch_init=lambda p, ins: _custom_render(p.get("torch_code") or "nn.Identity()", p, ins),
    torch_call=lambda p, ins, mod, s: f"{mod}({', '.join(ins)})",
    keras_call=lambda p, ins, s: (
        _custom_render(p.get("keras_code") or 'layers.Activation("linear")', p, s)
        + f"({ins[0] if len(ins) == 1 else '[' + ', '.join(ins) + ']'})"
    ),
))


# The torch class each core layer wraps, so the inspector can link straight to
# the reference page rather than making you search for it.
TORCH_CLASS = {
    "Linear": "Linear", "Conv2d": "Conv2d", "Conv1d": "Conv1d",
    "ConvTranspose2d": "ConvTranspose2d", "MaxPool2d": "MaxPool2d",
    "AvgPool2d": "AvgPool2d", "MaxPool1d": "MaxPool1d",
    "AdaptiveAvgPool2d": "AdaptiveAvgPool2d", "BatchNorm2d": "BatchNorm2d",
    "BatchNorm1d": "BatchNorm1d", "LayerNorm": "LayerNorm",
    "GroupNorm": "GroupNorm", "Dropout": "Dropout", "Dropout2d": "Dropout2d",
    "Flatten": "Flatten", "Embedding": "Embedding", "LSTM": "LSTM",
    "GRU": "GRU", "SimpleRNN": "RNN", "SelfAttention": "MultiheadAttention",
    "TransformerEncoder": "TransformerEncoder", "Upsample2d": "Upsample",
    "SeparableConv2d": "Conv2d",
}

ACTIVATION_CLASS = {
    "relu": "ReLU", "leaky_relu": "LeakyReLU", "gelu": "GELU", "silu": "SiLU",
    "tanh": "Tanh", "sigmoid": "Sigmoid", "elu": "ELU", "softmax": "Softmax",
    "identity": "Identity",
}


def docs_url(name: str) -> Optional[str]:
    cls = TORCH_CLASS.get(name)
    if not cls:
        return None
    return f"https://docs.pytorch.org/docs/stable/generated/torch.nn.{cls}.html"


def catalog() -> List[Dict[str, Any]]:
    """Everything the palette and inspector need to render, in one payload."""
    return [
        {
            "name": s.name,
            "category": s.category,
            "params": s.params,
            "n_inputs": s.n_inputs,
            "doc": s.doc,
            "kind": s.kind,
            "source": s.source,
            "origin": s.origin,
            "keras": s.keras_call is not None or s.kind == "runtime",
            "docs": docs_url(s.name),
        }
        for s in REGISTRY.values()
    ]


CORE_NAMES = None


def snapshot_core() -> None:
    """Remember the built-in layers so a block reload can drop only block specs."""
    global CORE_NAMES
    if CORE_NAMES is None:
        CORE_NAMES = set(REGISTRY)


def drop_blocks() -> None:
    """Remove every block-provided spec, ready for a fresh scan of blocks/."""
    if CORE_NAMES is None:
        return
    for name in [n for n in REGISTRY if n not in CORE_NAMES]:
        REGISTRY.pop(name, None)
