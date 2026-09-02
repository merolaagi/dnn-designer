"""What each layer is, mathematically, with this node's numbers in it.

A textbook gives you the general form. That is the wrong thing when you are
sitting in front of a particular convolution deciding whether to widen it: you
want *this* layer's equation, *this* layer's shape arithmetic worked through,
and *this* layer's parameter count derived rather than asserted.

So every entry produces three things from the node's real parameters and the
shapes flowing through it:

  equation     the operation, in the notation the literature uses
  arithmetic   the shape and parameter formulas with the numbers substituted,
               so you can see where each one came from
  freedom      what can be varied and what varying it does to the mathematics —
               the part that matters if you are trying to find something new

Where a count is approximate it says so. Nothing here is a guess dressed as a
fact: layers whose mathematics is not written down here say so plainly rather
than producing something plausible.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import layers


def _fmt(n) -> str:
    if isinstance(n, float) and not n.is_integer():
        return f"{n:g}"
    return f"{int(n):,}"


def _prod(values) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


# --------------------------------------------------------------------------
# entries
# --------------------------------------------------------------------------

def linear(p, ins, out):
    d_in = ins[0][-1] if ins and ins[0] else 0
    d_out = int(p.get("units", 0))
    bias = p.get("bias", True)
    return {
        "family": "linear",
        "title": "Affine map",
        "equation": "y = W x + b",
        "shape": "W \u2208 \u211d^(out\u00d7in),  x \u2208 \u211d^in,  b \u2208 \u211d^out",
        "symbols": [
            ("x", f"the {_fmt(d_in)} numbers arriving"),
            ("W", f"a learned {_fmt(d_out)}\u00d7{_fmt(d_in)} matrix"),
            ("b", f"a learned shift, one per output"),
            ("y", f"the {_fmt(d_out)} numbers leaving"),
        ],
        "arithmetic": [
            ("output size", f"{_fmt(d_out)}, because you set it — a dense layer's "
                            f"width is a choice, not a consequence"),
            ("parameters", f"in\u00d7out{' + out' if bias else ''} = "
                           f"{_fmt(d_in)}\u00d7{_fmt(d_out)}"
                           f"{f' + {_fmt(d_out)}' if bias else ''} = "
                           f"{_fmt(d_in * d_out + (d_out if bias else 0))}"),
            ("multiplies per sample", f"{_fmt(d_in * d_out)}"),
        ],
        "freedom": [
            "Every output sees every input, so cost grows as the product of the "
            "two widths. That product is usually where a network's parameters go.",
            "Two of these in a row with nothing between them multiply to a single "
            "matrix W\u2082W\u2081 — the depth buys nothing without a nonlinearity.",
            "Factorising W \u2248 UV with U \u2208 \u211d^(out\u00d7r), V \u2208 \u211d^(r\u00d7in) "
            f"costs r\u00d7({_fmt(d_in)}+{_fmt(d_out)}) instead of "
            f"{_fmt(d_in * d_out)}. Below r \u2248 "
            f"{max(1, (d_in * d_out) // max(1, d_in + d_out))} it is cheaper, and "
            "it constrains the map to rank r — sometimes a useful prior rather "
            "than only a saving.",
        ],
    }


def conv2d(p, ins, out):
    shape = ins[0] if ins and ins[0] else [0, 0, 0]
    c_in, h, w = (shape + [0, 0, 0])[:3]
    k = int(p.get("kernel", 3))
    s = int(p.get("stride", 1)) or 1
    d = int(p.get("dilation", 1)) or 1
    g = int(p.get("groups", 1)) or 1
    c_out = int(p.get("filters", 0))
    pad = p.get("padding", 0)
    effective = d * (k - 1) + 1
    pad_n = (effective - 1) // 2 if pad == "same" else int(pad or 0)
    bias = p.get("bias", True)
    weights = (c_in // g) * k * k * c_out
    out_h = (h + 2 * pad_n - effective) // s + 1 if h else 0

    return {
        "family": "conv",
        "title": "Discrete cross-correlation",
        "equation": ("y[c,i,j] = b[c] + \u03a3_k \u03a3_u \u03a3_v "
                     "W[c,k,u,v] \u00b7 x[k, s\u00b7i + d\u00b7u \u2212 p, "
                     "s\u00b7j + d\u00b7v \u2212 p]"),
        "shape": f"k over {_fmt(c_in // g)} input channels, u,v over {k}\u00d7{k}",
        "symbols": [
            ("W", f"{_fmt(c_out)} learned filters, each "
                  f"{_fmt(c_in // g)}\u00d7{k}\u00d7{k}"),
            ("s", f"stride {s} — how far the window moves per output position"),
            ("d", f"dilation {d} — the gap between the window's taps"),
            ("p", f"padding {pad_n} on each side"),
            ("g", f"groups {g}" + (" (each filter sees every input channel)"
                                   if g == 1 else
                                   f" (each filter sees only {_fmt(c_in // g)} channels)")),
        ],
        "arithmetic": [
            ("effective window", f"d(k\u22121)+1 = {d}\u00d7{k - 1}+1 = {effective}"),
            ("output size", f"\u230a(H + 2p \u2212 d(k\u22121) \u2212 1)/s\u230b + 1 = "
                            f"\u230a({h} + {2 * pad_n} \u2212 {effective - 1} \u2212 1)/{s}\u230b + 1 "
                            f"= {out_h}"),
            ("parameters", f"(C_in/g)\u00b7k\u00b2\u00b7C_out"
                           f"{' + C_out' if bias else ''} = "
                           f"{_fmt(c_in // g)}\u00d7{k * k}\u00d7{_fmt(c_out)}"
                           f"{f' + {_fmt(c_out)}' if bias else ''} = "
                           f"{_fmt(weights + (c_out if bias else 0))}"),
            ("multiplies", f"parameters \u00d7 output positions \u2248 "
                           f"{_fmt(weights)} \u00d7 {_fmt(out_h * out_h)} = "
                           f"{_fmt(weights * out_h * out_h)}"),
        ],
        "freedom": [
            f"The same {_fmt(weights)} weights are reused at every one of the "
            f"{_fmt(out_h * out_h)} positions. That reuse is the whole idea: it "
            "assumes what matters is the same everywhere in the image, and it is "
            "why a convolution has so few parameters for what it does.",
            f"Dilation widens the window without adding a single parameter: "
            f"d={d} already reaches {effective} pixels using {k}\u00d7{k} weights. "
            "Stacking dilations 1,2,4,8 grows the reach exponentially with depth.",
            f"Groups divide the parameters by g. At g=C_in each filter sees one "
            f"channel — that is a depthwise convolution, "
            f"{_fmt(k * k * c_in)} parameters instead of {_fmt(weights)}, and a "
            "1\u00d71 convolution after it mixes the channels back. That "
            "factorisation is most of what makes mobile architectures small.",
            "Receptive field after L such layers is 1 + L\u00b7d(k\u22121) — linear in "
            "depth. Stride multiplies it instead, which is why downsampling buys "
            "reach so cheaply.",
        ],
    }


def pooling(kind):
    def build(p, ins, out):
        shape = ins[0] if ins and ins[0] else [0, 0, 0]
        c, h, w = (list(shape) + [0, 0, 0])[:3]
        k = int(p.get("kernel", 2))
        s = int(p.get("stride", 0)) or k
        pad_n = int(p.get("padding", 0) or 0)
        out_h = (h + 2 * pad_n - k) // s + 1 if h else 0
        op = "max" if kind == "max" else "mean"
        formula = ("y[c,i,j] = max over the window  x[c, s\u00b7i+u, s\u00b7j+v]"
                   if kind == "max" else
                   "y[c,i,j] = (1/k\u00b2) \u03a3_u \u03a3_v x[c, s\u00b7i+u, s\u00b7j+v]")
        return {
            "family": "pool",
            "title": f"{op.capitalize()} pooling",
            "equation": formula,
            "shape": f"u,v over {k}\u00d7{k}, stride {s}",
            "symbols": [("k", f"window {k}\u00d7{k}"), ("s", f"stride {s}"),
                        ("C", f"{_fmt(c)} channels, each pooled independently")],
            "arithmetic": [
                ("output size", f"\u230a(H + 2p \u2212 k)/s\u230b + 1 = "
                                f"\u230a({h} + {2 * pad_n} \u2212 {k})/{s}\u230b + 1 = {out_h}"),
                ("parameters", "none — the operation has nothing to learn"),
                ("kept", f"{out_h * out_h} of {h * w} positions "
                         f"({100 * out_h * out_h / max(1, h * w):.0f}%)"),
            ],
            "freedom": [
                "Max is a hard selection: the gradient reaches only the winning "
                "element, so the rest of the window learns nothing from this step. "
                "Mean spreads it evenly.",
                "Max keeps the strongest response and discards where in the window "
                "it was. That invariance is useful for recognition and harmful for "
                "anything needing precise location.",
                "A strided convolution downsamples too, and learns how rather than "
                "being told. It costs parameters; pooling costs none.",
            ],
        }
    return build


def global_pool(p, ins, out):
    shape = ins[0] if ins and ins[0] else [0, 0, 0]
    c, h, w = (list(shape) + [0, 0, 0])[:3]
    return {
        "family": "pool",
        "title": "Global average pooling",
        "equation": "y[c] = (1/HW) \u03a3_i \u03a3_j x[c,i,j]",
        "shape": f"{_fmt(c)}\u00d7{h}\u00d7{w} \u2192 {_fmt(c)}",
        "symbols": [("H,W", f"{h}\u00d7{w}, averaged away"),
                    ("C", f"{_fmt(c)} channels, one number each")],
        "arithmetic": [
            ("output size", f"C = {_fmt(c)}, one number per channel"),
            ("parameters", "none"),
            ("compared with flattening",
             f"a Linear on the flattened map would need "
             f"{_fmt(c * h * w)} inputs; after this it needs {_fmt(c)} — a "
             f"factor of {h * w} fewer weights in the head"),
        ],
        "freedom": [
            "Each channel becomes one number: how strongly its feature was present "
            "anywhere. All spatial information is gone by construction.",
            "It accepts any H and W, so the network stops caring about input size.",
            "The reason a classifier head after this is tiny — and the reason it "
            "cannot express anything about where things were.",
        ],
    }


def batchnorm(p, ins, out):
    shape = ins[0] if ins and ins[0] else [0]
    c = shape[0] if shape else 0
    eps = p.get("eps", 1e-5)
    return {
        "family": "norm",
        "title": "Batch normalization",
        "equation": "y = \u03b3 \u00b7 (x \u2212 \u03bc_B) / \u221a(\u03c3\u00b2_B + \u03b5) + \u03b2",
        "shape": "\u03bc, \u03c3\u00b2 taken per channel over the batch and all positions",
        "symbols": [
            ("\u03bc_B, \u03c3\u00b2_B", "mean and variance of this batch, per channel"),
            ("\u03b3, \u03b2", f"{_fmt(c)} learned scales and {_fmt(c)} learned shifts"),
            ("\u03b5", f"{eps}, so the division cannot blow up"),
        ],
        "arithmetic": [
            ("output size", "unchanged — this rescales, it does not reshape"),
            ("parameters", f"2C = 2\u00d7{_fmt(c)} = {_fmt(2 * c)} trained"),
            ("also stored", f"{_fmt(2 * c)} running statistics, updated but not "
                            f"trained by the optimizer"),
        ],
        "freedom": [
            "The statistics come from the batch, so what this layer computes for "
            "one sample depends on the others beside it. That is why small batches "
            "hurt it and why it behaves differently in eval mode, where the running "
            "statistics take over.",
            "\u03b3 and \u03b2 let the network undo the normalization if that is "
            "better — initialising \u03b3=0 in a residual branch starts the block as "
            "the identity, which is a common trick for training very deep stacks.",
            "LayerNorm takes the statistics per sample instead, so batch size stops "
            "mattering. That is why transformers use it.",
        ],
    }


def layernorm(p, ins, out):
    shape = ins[0] if ins and ins[0] else [0]
    n = shape[-1] if shape else 0
    return {
        "family": "norm",
        "title": "Layer normalization",
        "equation": "y = \u03b3 \u00b7 (x \u2212 \u03bc) / \u221a(\u03c3\u00b2 + \u03b5) + \u03b2",
        "shape": "\u03bc, \u03c3\u00b2 taken over the features of each sample separately",
        "symbols": [
            ("\u03bc, \u03c3\u00b2", f"mean and variance across this sample's "
                                    f"{_fmt(n)} features"),
            ("\u03b3, \u03b2", f"{_fmt(n)} learned scales and shifts"),
        ],
        "arithmetic": [
            ("output size", "unchanged"),
            ("parameters", f"2\u00d7{_fmt(n)} = {_fmt(2 * n)}"),
        ],
        "freedom": [
            "Nothing here depends on the other samples in the batch, so batch size "
            "is irrelevant and training and inference compute exactly the same "
            "function.",
            "Removing the mean subtraction gives RMSNorm — cheaper, and in practice "
            "usually just as good, which is a hint that the centring was doing less "
            "than the scaling.",
        ],
    }


def dropout(p, ins, out):
    rate = float(p.get("rate", 0.5))
    keep = 1 - rate
    return {
        "family": "dropout",
        "title": "Dropout",
        "equation": "y = x \u00b7 m / (1 \u2212 p),   m ~ Bernoulli(1 \u2212 p)",
        "shape": "one independent coin per element, per forward pass",
        "symbols": [
            ("p", f"{rate:g} — the chance any given element is zeroed"),
            ("m", "a fresh random mask every step, in training only"),
            ("1/(1\u2212p)", f"\u00d7{1 / keep:.3f} on the survivors, so the "
                             f"expected sum is unchanged"),
        ],
        "arithmetic": [
            ("output size", "unchanged"),
            ("parameters", "none"),
            ("at evaluation", "the identity — nothing is dropped and nothing is "
                              "rescaled, because the scaling already happened "
                              "during training"),
        ],
        "freedom": [
            "The rescaling is the part people forget. Without it the network would "
            "see a differently-sized signal at evaluation than at training.",
            "It stops any one path being relied on, which is often described as "
            "training an ensemble of subnetworks sharing weights.",
            "On convolutional feature maps neighbouring pixels are correlated "
            "enough that dropping single elements does little; Dropout2d drops "
            "whole channels for that reason.",
        ],
    }


ACTIVATION_MATH = {
    "relu": ("max(0, x)", "Zero below the origin, identity above. The gradient is "
                          "0 or 1, which is why it trains well and why a unit stuck "
                          "below zero never recovers."),
    "leaky_relu": ("max(\u03b1x, x)", "Like ReLU but with a small slope below zero, "
                                       "so a unit that goes negative can still learn."),
    "gelu": ("x \u00b7 \u03a6(x)", "x weighted by the probability a standard normal "
                                    "falls below it. Smooth everywhere, unlike ReLU's "
                                    "corner at the origin."),
    "silu": ("x \u00b7 \u03c3(x)", "Also called Swish. Smooth, and slightly negative "
                                    "just below zero rather than flat."),
    "tanh": ("(e^x \u2212 e^\u2212x)/(e^x + e^\u2212x)",
             "Squashes into (\u22121,1). Saturates at both ends, so gradients vanish "
             "for large |x|."),
    "sigmoid": ("1/(1 + e^\u2212x)", "Squashes into (0,1). Its maximum gradient is "
                                      "0.25, so stacking these shrinks gradients fast."),
    "softmax": ("e^(x_i) / \u03a3_j e^(x_j)",
                "Turns scores into a distribution over the axis. Shift-invariant: "
                "adding a constant to every score changes nothing, which is what "
                "makes the max-subtraction trick safe."),
    "elu": ("x if x>0 else \u03b1(e^x \u2212 1)",
            "Smooth, and saturates to \u2212\u03b1 rather than to zero."),
    "identity": ("x", "Nothing at all — useful as a placeholder."),
}


def activation(p, ins, out):
    kind = str(p.get("kind", "relu"))
    formula, note = ACTIVATION_MATH.get(kind, ("\u2014", ""))
    return {
        "family": "activation",
        "kind": kind,
        "title": f"{kind} — pointwise nonlinearity",
        "equation": f"y = {formula}",
        "shape": "applied to every element independently",
        "symbols": [("x", "one number, and every number separately")],
        "arithmetic": [
            ("output size", "unchanged — nothing moves between positions"),
            ("parameters", "none"),
        ],
        "freedom": [
            note,
            "Without something like this the whole network collapses: any stack of "
            "affine maps is one affine map. This is the only reason depth adds "
            "expressive power.",
            "It acts on each number alone, so it can bend the function but never "
            "mix information between positions.",
        ],
    }


def attention(p, ins, out):
    shape = ins[0] if ins and ins[0] else [0, 0]
    tokens = shape[0] if len(shape) > 1 else 0
    d = shape[-1] if shape else 0
    heads = int(p.get("heads", 1)) or 1
    d_k = d // heads if heads else d
    params = 4 * d * d + 4 * d
    return {
        "family": "attention",
        "title": "Scaled dot-product attention",
        "equation": ("Attention(Q,K,V) = softmax(QK\u1d40 / \u221ad_k) V"),
        "shape": f"Q,K,V \u2208 \u211d^({_fmt(tokens)}\u00d7{_fmt(d_k)}) per head",
        "symbols": [
            ("Q,K,V", f"projections of the input, {_fmt(d)} \u2192 {_fmt(d)}, "
                      f"split across {heads} head{'' if heads == 1 else 's'}"),
            ("QK\u1d40", f"a {_fmt(tokens)}\u00d7{_fmt(tokens)} score for every "
                          f"pair of positions"),
            ("\u221ad_k", f"\u221a{_fmt(d_k)} \u2248 {math.sqrt(max(1, d_k)):.2f}, "
                           f"which keeps the scores' variance near 1 before the "
                           f"softmax"),
        ],
        "arithmetic": [
            ("per head width", f"d/h = {_fmt(d)}/{heads} = {_fmt(d_k)}"),
            ("score matrix", f"{_fmt(tokens)}\u00d7{_fmt(tokens)} = "
                             f"{_fmt(tokens * tokens)} entries per head, "
                             f"{_fmt(tokens * tokens * heads)} in total"),
            ("parameters", f"4d\u00b2 + 4d = 4\u00d7{_fmt(d)}\u00b2 + 4\u00d7{_fmt(d)} "
                           f"= {_fmt(params)} (three input projections and one "
                           f"output projection)"),
            ("cost", f"grows as tokens\u00b2 = {_fmt(tokens)}\u00b2 = "
                     f"{_fmt(tokens * tokens)} — doubling the sequence quadruples "
                     f"this"),
        ],
        "freedom": [
            "Every position can read every other in one step, with no notion of "
            "distance. That is the strength, and the reason positions have to be "
            "encoded separately — the operation itself is permutation-equivariant.",
            f"The \u221ad_k divisor exists because a dot product of two "
            f"{_fmt(d_k)}-dimensional vectors with unit-variance entries has "
            f"variance {_fmt(d_k)}. Without it the softmax saturates and gradients "
            f"vanish.",
            "The quadratic term is the whole subject of efficient-attention work: "
            "sparsity, low-rank approximations of the score matrix, and kernel "
            "tricks that avoid forming it at all.",
            "Heads are the same operation on disjoint slices, so they cost no extra "
            "parameters — h is a choice about how to divide d, not how much to "
            "spend.",
        ],
    }


def embedding(p, ins, out):
    vocab = int(p.get("vocab", 0))
    dim = int(p.get("dim", 0))
    return {
        "family": "embedding",
        "title": "Lookup table",
        "equation": "y = W[i]   (equivalently W\u1d40 one_hot(i))",
        "shape": f"W \u2208 \u211d^({_fmt(vocab)}\u00d7{_fmt(dim)})",
        "symbols": [
            ("i", "an integer index, not a vector"),
            ("W", f"{_fmt(vocab)} rows of {_fmt(dim)} numbers, all learned"),
        ],
        "arithmetic": [
            ("output size", f"{_fmt(dim)} per index"),
            ("parameters", f"V\u00d7d = {_fmt(vocab)}\u00d7{_fmt(dim)} = "
                           f"{_fmt(vocab * dim)}"),
        ],
        "freedom": [
            "Formally a matrix multiply against a one-hot vector, which is why the "
            "gradient reaches exactly one row per occurrence. Implemented as a "
            "lookup because multiplying by a one-hot would be wasteful.",
            "Rows for rare indices are updated rarely and stay near their "
            "initialisation — a real effect on long-tailed vocabularies.",
            "Tying this matrix to the output projection halves the parameters and "
            "usually helps, since both are maps between the same two spaces.",
        ],
    }


def lstm(p, ins, out):
    shape = ins[0] if ins and ins[0] else [0, 0]
    d_in = shape[-1] if shape else 0
    hidden = int(p.get("hidden", 0))
    layers_n = int(p.get("layers", 1)) or 1
    bidir = bool(p.get("bidirectional", False))
    directions = 2 if bidir else 1
    per = 4 * (hidden * d_in + hidden * hidden + 2 * hidden)
    return {
        "family": "recurrent",
        "title": "Long short-term memory",
        "equation": ("i,f,o = \u03c3(W\u00b7[x_t, h_{t\u22121}] + b)   "
                     "g = tanh(\u00b7)   "
                     "c_t = f\u2299c_{t\u22121} + i\u2299g   "
                     "h_t = o\u2299tanh(c_t)"),
        "shape": f"h, c \u2208 \u211d^{_fmt(hidden)}",
        "symbols": [
            ("c_t", "the cell state — carried forward almost unchanged when f\u22481"),
            ("f", "forget gate: how much of the old cell state survives"),
            ("i", "input gate: how much of the new candidate enters"),
            ("o", "output gate: how much of the cell state is revealed"),
        ],
        "arithmetic": [
            ("gates", "four, each with its own weights — hence the 4"),
            ("parameters per layer",
             f"4\u00b7(h\u00b7in + h\u00b2 + 2h) = 4\u00d7({_fmt(hidden)}\u00d7{_fmt(d_in)} "
             f"+ {_fmt(hidden)}\u00b2 + 2\u00d7{_fmt(hidden)}) = {_fmt(per)}"),
            ("total", f"\u00d7{layers_n} layer{'' if layers_n == 1 else 's'}"
                      f"{' \u00d72 directions' if bidir else ''} \u2248 "
                      f"{_fmt(per * layers_n * directions)}"),
        ],
        "freedom": [
            "c_t = f\u2299c_{t\u22121} + i\u2299g is the whole point: when the forget "
            "gate is near 1 the cell state passes through almost untouched, so the "
            "gradient does too. That additive path is what a plain RNN lacks and "
            "why it cannot remember.",
            "Sequential by construction — step t needs step t\u22121 — so it cannot "
            "be parallelised across time the way attention can. That, more than "
            "quality, is why transformers displaced it.",
            "A GRU merges the input and forget gates into one and drops the "
            "separate cell state: three gates instead of four, about 3/4 of the "
            "parameters, usually very close in quality.",
        ],
    }


def flatten(p, ins, out):
    shape = ins[0] if ins and ins[0] else []
    total = _prod(shape) if shape else 0
    return {
        "family": "reshape",
        "title": "Reinterpretation",
        "equation": "y[n] = x[i,j,k]   with n = (i\u00b7H + j)\u00b7W + k",
        "shape": f"{'\u00d7'.join(str(int(v)) for v in shape)} \u2192 {_fmt(total)}",
        "symbols": [("n", "a single index walking the same memory in order")],
        "arithmetic": [
            ("output size", f"the product: {' \u00d7 '.join(str(int(v)) for v in shape)}"
                            f" = {_fmt(total)}"),
            ("parameters", "none — not one number changes"),
        ],
        "freedom": [
            "Nothing is computed here. The same values are read as one long vector.",
            f"What it costs comes next: a dense layer on {_fmt(total)} inputs needs "
            f"{_fmt(total)} weights per output. Pooling first is how that bill is "
            f"avoided.",
            "Because the ordering is positional, the layer after it treats position "
            "37 as a fixed meaning — which is why flattening destroys translation "
            "invariance that the convolutions built.",
        ],
    }


def merge(kind):
    def build(p, ins, out):
        shapes = [s for s in ins if s]
        if kind == "Add":
            equation = "y = x\u2081 + x\u2082"
            note = ("Requires identical shapes. The gradient reaches both inputs "
                    "unchanged, which is exactly why residual connections train: "
                    "\u2202y/\u2202x\u2081 = 1 gives the gradient a path that skips the "
                    "block entirely.")
        elif kind == "Multiply":
            equation = "y = x\u2081 \u2299 x\u2082"
            note = ("Each input scales the other, so the gradient to one is the "
                    "value of the other — a gate. If one side is near zero, nothing "
                    "flows to the other.")
        else:
            equation = "y = [x\u2081 ; x\u2082]  along the channel axis"
            note = ("Nothing is combined here — both are kept and the next layer "
                    "decides what to do with them. Costs channels rather than "
                    "arithmetic.")
        return {
            "family": "merge",
            "title": f"{kind}",
            "equation": equation,
            "shape": " , ".join("\u00d7".join(str(int(v)) for v in s) for s in shapes)
                     or "\u2014",
            "symbols": [("x\u2081, x\u2082", "the incoming branches, in port order")],
            "arithmetic": [
                ("output size", "\u00d7".join(str(int(v)) for v in out) if out else "\u2014"),
                ("parameters", "none"),
            ],
            "freedom": [note],
        }
    return build


def subgraph(p, ins, out):
    return {
        "family": "reshape",
        "title": "Another sheet",
        "equation": "y = f(x)   where f is the sheet named below",
        "shape": f"defined by that sheet: \u2192 "
                 f"{'\u00d7'.join(str(int(v)) for v in out) if out else '\u2014'}",
        "symbols": [("f", f"the sheet \u201c{p.get('sheet', '')}\u201d, generated as "
                          f"its own module class")],
        "arithmetic": [("parameters", "whatever that sheet holds — open it to see "
                                      "the mathematics inside")],
        "freedom": ["Composition, nothing more: the mathematics is on the other sheet."],
    }


ENTRIES: Dict[str, Callable] = {
    "Linear": linear,
    "Conv2d": conv2d,
    "Conv1d": conv2d,
    "MaxPool2d": pooling("max"),
    "AvgPool2d": pooling("avg"),
    "MaxPool1d": pooling("max"),
    "GlobalAvgPool": global_pool,
    "AdaptiveAvgPool2d": global_pool,
    "BatchNorm2d": batchnorm,
    "BatchNorm1d": batchnorm,
    "LayerNorm": layernorm,
    "Dropout": dropout,
    "Dropout2d": dropout,
    "Activation": activation,
    "SelfAttention": attention,
    "Embedding": embedding,
    "LSTM": lstm,
    "GRU": lstm,
    "SimpleRNN": lstm,
    "Flatten": flatten,
    "Add": merge("Add"),
    "Concat": merge("Concat"),
    "Multiply": merge("Multiply"),
    "Subgraph": subgraph,
}


def explain(node_type: str, params: Dict[str, Any],
            in_shapes: List[Any], out_shape: Optional[List[int]]) -> Dict[str, Any]:
    """The mathematics of one layer, with this node's numbers substituted."""
    builder = ENTRIES.get(node_type)
    if builder is None:
        spec = layers.REGISTRY.get(node_type)
        return {
            "family": "unknown",
            "title": node_type,
            "equation": "",
            "shape": "",
            "symbols": [],
            "arithmetic": [],
            "freedom": [],
            "missing": (
                f"The mathematics of {node_type} is not written up here yet. "
                + ((spec.doc + " ") if spec and spec.doc else "")
                + "Rather than generate something plausible, this says nothing."),
        }
    try:
        return builder(params, in_shapes, out_shape)
    except Exception as exc:  # noqa: BLE001 - a panel must never break the canvas
        return {"family": "unknown", "title": node_type, "equation": "",
                "symbols": [], "arithmetic": [], "freedom": [],
                "missing": f"Could not work that through: {type(exc).__name__}: {exc}"}


def covered() -> List[str]:
    return sorted(ENTRIES)
