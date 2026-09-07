"""What a design needs before it will run.

The canvas tells you whether the shapes work. It does not tell you that this
graph pulls in two plug-in blocks, wants torchvision installed, will download
ImageNet weights on its first epoch, and cannot be trained by the standard loop
at all. That is a different question, and it is the one that usually costs
someone an afternoon.

Everything here is derived from the graph rather than declared, so it cannot go
stale when the graph changes.
"""

from __future__ import annotations

from typing import Any, Dict, List

import layers
import recipes_sdk
from graph import Graph, resolved_params


def _dataset_options(in_shapes: List[List[int]], dtypes: List[str]) -> List[str]:
    """Which dataset kinds could feed these Inputs."""
    if not in_shapes:
        return []
    options = ["synthetic"]
    if any(d == "long" for d in dtypes) and all(len(s) == 1 for s in in_shapes):
        options.append("text")
    if all(len(s) == 3 for s in in_shapes):
        options.extend(["mnist", "fashion_mnist", "cifar10", "folder"])
    if all(len(s) in (1, 2) for s in in_shapes) and not any(d == "long" for d in dtypes):
        options.append("csv")
    return options


def _suggest_recipe(g: Graph, used: set, tasks: List[str]) -> Dict[str, Any]:
    """The training loop this design most likely wants."""
    if "GPTStack" in used or "language_modeling" in tasks:
        return {"name": "", "note": "The standard loop, with the Output set to "
                                    "language_modeling."}
    if any(layers.REGISTRY.get(t) and layers.REGISTRY[t].kind == "runtime"
           for t in used) and {"PolicyHead", "ValueHead"} & used:
        return {"name": "", "note": "Train supervised against search output; the "
                                    "self-play loop is not in the trainer yet."}
    inputs = [n for n in g.nodes if n.type == "Input"]
    outputs = [n for n in g.nodes if n.type == "Output"]
    if inputs and outputs:
        in_shape = [int(x) for x in resolved_params(inputs[0]).get("shape", [])]
        if len(in_shape) == 1 and in_shape and in_shape[0] < 512 and len(inputs) == 1:
            # a noise vector into an image is the generator shape
            pass
    if "Detection" in used:
        return {"name": "Detection", "note": ""}
    return {"name": "", "note": ""}


def requirements(g: Graph, report: Dict[str, Any]) -> Dict[str, Any]:
    nodes = g.by_id()
    used = {n.type for n in g.nodes}

    blocks, runtime, pretrained, no_keras = [], [], [], []
    for name in sorted(used):
        spec = layers.REGISTRY.get(name)
        if spec is None:
            continue
        if spec.source == "block":
            blocks.append({"name": name, "file": spec.origin or "?",
                           "category": spec.category})
        if spec.kind == "runtime":
            runtime.append({"name": name, "note": spec.doc.split(".")[0] + "."})
        if spec.keras_call is None and spec.kind != "runtime" and name not in (
                "Input", "Output"):
            no_keras.append(name)

    for n in g.nodes:
        if n.type == "Backbone":
            params = resolved_params(n)
            pretrained.append({
                "arch": params.get("arch", "?"),
                "weights": params.get("weights", "DEFAULT"),
                "frozen": int(params.get("trainable_stages", 0)) == 0,
            })

    packages = [{"name": "torch", "why": "everything"}]
    if pretrained or any(len(v.get("out_shape") or []) == 3
                         for v in report.get("nodes", {}).values()):
        packages.append({"name": "torchvision",
                         "why": "pretrained weights and image datasets"})
    if "pandas" not in [p["name"] for p in packages]:
        packages.append({"name": "pandas", "why": "CSV tables"})

    inputs = [nodes[i] for i in report.get("order", []) if nodes[i].type == "Input"]
    outputs = [nodes[i] for i in report.get("order", []) if nodes[i].type == "Output"]
    in_shapes = [report["nodes"][n.id]["out_shape"] for n in inputs
                 if report["nodes"].get(n.id, {}).get("out_shape")]
    dtypes = [resolved_params(n).get("dtype", "float") for n in inputs]
    tasks = [resolved_params(n).get("task", "classification") for n in outputs]

    notes = []
    if len(inputs) > 1:
        notes.append(f"{len(inputs)} Inputs. forward() takes them in the order "
                     f"{', '.join(n.label or n.type for n in inputs)}, and the "
                     f"loader feeds them in that same order.")
    if len(outputs) > 1:
        notes.append(f"{len(outputs)} Outputs. All heads train against the same "
                     f"target, with later ones scaled by the extra-head weight.")
    if pretrained:
        downloads = [p["arch"] for p in pretrained if p["weights"] != "none"]
        if downloads:
            notes.append(f"First run downloads weights for "
                         f"{', '.join(downloads)}, so it needs a network connection.")
    if no_keras:
        notes.append(f"The Keras export will be incomplete: "
                     f"{', '.join(no_keras)} has no Keras form.")
    if "ImplicitEquilibrium" in used:
        notes.append("ImplicitEquilibrium solves to a tolerance single precision "
                     "cannot reach, so it is built in float64. Set the training "
                     "precision to double.")
        # a control arm is a deliberate choice, but it should never be a silent
        # one: nothing guarantees the solver converges there
        controls = []
        for n in g.nodes:
            if n.type != "ImplicitEquilibrium":
                continue
            params = resolved_params(n)
            try:
                from dnn_bench import core, domains  # noqa: F401

                spec = core.get(str(params.get("domain", "")))
                arms = spec.variants(spec.defaults(), seed=int(params.get("seed", 0)))
                arm = next((v for v in arms
                            if v.key == str(params.get("variant"))), None)
                if arm is not None and not arm.build().certificate().get("holds"):
                    controls.append(f"{n.label or n.type} ({params.get('domain')} "
                                    f"/ {arm.key})")
            except Exception:  # noqa: BLE001
                continue
        if controls:
            notes.append(f"On a control arm: {', '.join(controls)}. One condition "
                         f"of the theorem is broken on purpose, so nothing "
                         f"guarantees a unique equilibrium and the solver may "
                         f"not converge. That is deliberate for a comparison, "
                         f"and wrong for a model you mean to keep.")
    if "EquilibriumCRN" in used:
        # the equilibrium solver runs to 1e-10, which single precision cannot
        # reach, so this is a property of the layer rather than a preference
        notes.append("EquilibriumCRN solves to a tolerance single precision "
                     "cannot reach, so it is built in float64. Set the training "
                     "precision to double, or the first batch will fail on a "
                     "dtype mismatch.")

    return {
        "blocks": blocks,
        "runtime": runtime,
        "pretrained": pretrained,
        "packages": packages,
        "datasets": _dataset_options(in_shapes, dtypes),
        "tasks": tasks,
        "recipes": sorted(recipes_sdk.REGISTRY),
        "suggested_recipe": _suggest_recipe(g, used, tasks),
        "inputs": [{"name": n.label or n.type,
                    "shape": report["nodes"].get(n.id, {}).get("out_shape")}
                   for n in inputs],
        "outputs": [{"name": n.label or n.type,
                     "task": resolved_params(n).get("task", "classification"),
                     "shape": report["nodes"].get(n.id, {}).get("out_shape")}
                    for n in outputs],
        "notes": notes,
        "keras_gaps": no_keras,
    }
