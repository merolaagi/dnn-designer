"""What a design costs at lower precision, and which layer objects.

A release like NVIDIA-Nemotron-…-NVFP4 is not a new model. The architecture is
unchanged, the training is unchanged, and the weights are the same weights read
at fewer bits. That is a fourth thing a "new model" can be, alongside a new
architecture, a new training run, and new post-training — and it is the cheapest
of the four to try, because it needs no gradients and no data.

Two questions are worth answering before shipping a network at reduced
precision, and only the second is interesting:

    what does it save        arithmetic, and knowable in advance
    what does it cost        measurable only by running it

The second is answered here by pushing the same batch through both models and
comparing the outputs. Per-layer, it is answered by quantizing one layer at a
time and leaving the rest alone — which finds the layer that must stay in higher
precision. That is what a mixed-precision release is: not every layer at four
bits, but every layer that can be.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import codegen
import graph as G


def _bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _fake_quantize(tensor, bits: int, group: int = 64):
    """Round to a grid, per group of weights, and come back to float.

    Real 4-bit formats store the codes packed with one scale per group. Here the
    values are put on that grid and then read back as floats, so the arithmetic
    error is exactly right while the storage is only calculated. Reporting a
    measured error and a computed size is honest; reporting a measured size for
    a format torch cannot store would not be.
    """
    import torch

    flat = tensor.detach().reshape(-1)
    pad = (-flat.numel()) % group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, group)

    span = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    levels = 2 ** (bits - 1) - 1
    step = span / levels
    coded = torch.clamp(torch.round(blocks / step), -levels - 1, levels)
    out = (coded * step).reshape(-1)[: tensor.numel()].reshape(tensor.shape)
    return out.to(tensor.dtype)


SCHEMES = [
    {"key": "float16", "label": "float16", "bits": 16,
     "note": "Every weight at half width. Supported natively on your GPU."},
    {"key": "bfloat16", "label": "bfloat16", "bits": 16,
     "note": "Same width as float16 with float32's exponent range: less "
             "precision per value, far less prone to overflow."},
    {"key": "int8", "label": "8-bit, groups of 64", "bits": 8,
     "note": "Weights on an 8-bit grid with a scale per small group. The usual "
             "first stop, and often free."},
    {"key": "int4", "label": "4-bit, groups of 64", "bits": 4,
     "note": "The shape of an NVFP4 or AWQ release: a grid per small group of "
             "weights. Where the savings are, and where the damage starts."},
]


def quantize_report(payload: Dict[str, Any], batch: int = 32,
                    weights: Optional[str] = None) -> Dict[str, Any]:
    """Measure every scheme against the design, then find the tender layer."""
    import torch

    import train as T

    g = G.parse(payload)
    report = G.analyze(g)
    if not report["ok"]:
        raise ValueError(report["errors"][0] if report["errors"]
                         else "The design does not resolve.")

    node_code: Dict[str, Any] = {}
    source = codegen.to_pytorch(g, report, node_code)
    name = codegen.model_class_name(g)
    model = T.build_model(source, name).eval()
    if weights:
        model.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)

    ids = codegen.input_order(g, report)
    nodes = g.by_id()
    inputs = []
    for nid in ids:
        shape = report["nodes"][nid]["out_shape"]
        params = G.resolved_params(nodes[nid])
        if str(params.get("dtype", "float")).startswith("long"):
            vocab = 1000
            for other in report["order"]:
                if nodes[other].type == "Embedding":
                    vocab = int(G.resolved_params(nodes[other]).get("vocab", 1000))
                    break
            inputs.append(torch.randint(0, vocab, (batch, *map(int, shape))))
        else:
            inputs.append(torch.randn(batch, *map(int, shape)))

    with torch.no_grad():
        reference = model(*inputs)
    reference = (reference[0] if isinstance(reference, (list, tuple))
                 else reference).float()
    scale = reference.abs().mean().clamp(min=1e-12)
    full_bytes = _bytes(model)

    rows = []
    for scheme in SCHEMES:
        copy_model = T.build_model(source, name).eval()
        copy_model.load_state_dict(model.state_dict())
        try:
            drift = _apply(copy_model, scheme["key"], inputs, reference, scale)
        except Exception as exc:  # noqa: BLE001
            rows.append({**scheme, "ok": False,
                         "why": f"{type(exc).__name__}: {exc}"})
            continue
        bits = scheme["bits"]
        # both integer schemes carry a 16-bit scale for every group of 64
        overhead = (16 / 64) if scheme["key"] in ("int4", "int8") else 0.0
        predicted = int(full_bytes * (bits + overhead) / 32)
        rows.append({**scheme, "ok": True, "drift": drift,
                     "bytes": predicted,
                     "saving": round(1 - predicted / max(1, full_bytes), 3)})

    return {
        "ok": True,
        "parameters": sum(p.numel() for p in model.parameters()),
        "bytes": full_bytes,
        "batch": batch,
        "schemes": rows,
        "layers": _sensitivity(model, source, name, node_code, inputs,
                               reference, scale),
    }


def _apply(model, scheme: str, inputs, reference, scale) -> float:
    """Convert, run, and report how far the output moved."""
    import torch

    with torch.no_grad():
        if scheme in ("float16", "bfloat16"):
            dtype = torch.float16 if scheme == "float16" else torch.bfloat16
            model = model.to(dtype)
            fed = [x.to(dtype) if x.is_floating_point() else x for x in inputs]
            out = model(*fed)
        else:
            # Both integer schemes go on the grid the same way. torch's dynamic
            # int8 path needs a quantized backend, and which of those exists
            # differs by machine — it is absent on Apple silicon, where this
            # reported "unavailable" for a scheme that is perfectly measurable.
            # Measuring both the same way also makes the two rows comparable,
            # which matters more than using the vendor kernel for one of them.
            bits = 8 if scheme == "int8" else 4
            for param in model.parameters():
                if param.dim() >= 2:              # weights, not biases or norms
                    param.copy_(_fake_quantize(param, bits))
            out = model(*inputs)

    out = (out[0] if isinstance(out, (list, tuple)) else out).float()
    return round(float((out - reference).abs().mean() / scale), 6)


def _sensitivity(model, source, name, node_code, inputs, reference, scale,
                 bits: int = 4) -> List[Dict[str, Any]]:
    """Quantize one layer at a time to find the one that cannot take it.

    A mixed-precision release is exactly this list: the layers that tolerate
    four bits go to four bits, and the one or two that do not are left alone.
    Without measuring, the choice is folklore — 'keep the first and last layer'
    is often right and sometimes wrong, and which it is depends on the network.
    """
    import torch

    import train as T

    by_var = {entry["var"]: nid for nid, entry in node_code.items()
              if entry.get("var")}
    rows = []
    for var, nid in by_var.items():
        probe = T.build_model(source, name).eval()
        probe.load_state_dict(model.state_dict())
        target = dict(probe.named_modules()).get(var)
        if target is None:
            continue
        weights = [p for p in target.parameters(recurse=False) if p.dim() >= 2]
        if not weights:
            continue
        with torch.no_grad():
            for param in weights:
                param.copy_(_fake_quantize(param, bits))
            out = probe(*inputs)
        out = (out[0] if isinstance(out, (list, tuple)) else out).float()
        rows.append({
            "id": nid, "var": var,
            "parameters": int(sum(p.numel() for p in weights)),
            "drift": round(float((out - reference).abs().mean() / scale), 6),
        })
    rows.sort(key=lambda r: -r["drift"])
    return rows
