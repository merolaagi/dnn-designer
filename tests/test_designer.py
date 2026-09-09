"""Smoke tests. Run with `python tests/test_designer.py` or `pytest tests/`.

The tests that need torch skip themselves when it is absent, so the suite still
means something in a bare environment. The ones that matter most check that
generated code actually runs and that shapes predicted on the canvas match what
PyTorch produces — a designer whose predictions disagree with the framework is
worse than no designer.
"""

import json
import inspect
import time
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import blockloader  # noqa: E402
import codegen  # noqa: E402
import graph as G  # noqa: E402
import layers  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

PASSED, FAILED = [], []


def check(name):
    def wrap(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  pass  {name}")
        except AssertionError as exc:
            FAILED.append((name, str(exc)))
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            FAILED.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
        return fn
    return wrap


def build(nodes, edges, name="T"):
    return {
        "name": name,
        "nodes": [{"id": i, "type": t, "params": p} for i, t, p in nodes],
        "edges": [{"id": f"e{k}", "source": a, "target": b, "port": port}
                  for k, (a, b, port) in enumerate(edges)],
    }


def analyzed(payload):
    g = G.parse(payload)
    return g, G.analyze(g)


blockloader.load_all()
print(f"\nregistry: {len(layers.REGISTRY)} layers "
      f"({sum(1 for s in layers.REGISTRY.values() if s.source == 'block')} from blocks)")
print(f"torch: {'available' if HAVE_TORCH else 'not installed, those tests skip'}\n")


@check("blocks all load without error")
def _():
    errors = blockloader.load_all()
    assert not errors, f"{len(errors)} block(s) failed: {errors}"


@check("core layer set is intact")
def _():
    for name in ("Input", "Output", "Conv2d", "Linear", "LSTM", "Concat", "Flatten"):
        assert name in layers.REGISTRY, f"{name} missing from the registry"


@check("conv shape arithmetic")
def _():
    spec = layers.REGISTRY["Conv2d"]
    out = spec.infer({"filters": 32, "kernel": 3, "stride": 2, "padding": 1,
                      "dilation": 1, "groups": 1, "bias": True}, [[3, 32, 32]])
    assert out == [32, 16, 16], out
    out = spec.infer({"filters": 32, "kernel": 3, "stride": 1, "padding": "same",
                      "dilation": 1, "groups": 1, "bias": True}, [[3, 32, 32]])
    assert out == [32, 32, 32], out
    out = spec.infer({"filters": 8, "kernel": 5, "stride": 1, "padding": 0,
                      "dilation": 1, "groups": 1, "bias": True}, [[3, 32, 32]])
    assert out == [8, 28, 28], out


@check("a bad shape is rejected with a readable message")
def _():
    spec = layers.REGISTRY["Conv2d"]
    try:
        spec.infer({"filters": 8, "kernel": 3, "stride": 1, "padding": "same",
                    "dilation": 1, "groups": 1, "bias": True}, [[10]])
        raise AssertionError("a rank-1 input should not be accepted")
    except layers.ShapeError as exc:
        assert "rank-3" in str(exc), str(exc)


@check("padding 'same' with a stride is refused, as PyTorch would")
def _():
    spec = layers.REGISTRY["Conv2d"]
    try:
        spec.infer({"filters": 8, "kernel": 3, "stride": 2, "padding": "same",
                    "dilation": 1, "groups": 1, "bias": True}, [[3, 32, 32]])
        raise AssertionError("stride 2 with 'same' should be rejected")
    except layers.ShapeError as exc:
        assert "stride" in str(exc), str(exc)


@check("Add and Multiply broadcast like PyTorch")
def _():
    spec = layers.REGISTRY["Multiply"]
    assert spec.infer({}, [[16, 8, 8], [16, 1, 1]]) == [16, 8, 8]
    try:
        spec.infer({}, [[16, 8, 8], [8, 8, 8]])
        raise AssertionError("mismatched channels should be rejected")
    except layers.ShapeError:
        pass


@check("a cycle is reported rather than hanging")
def _():
    _, rep = analyzed(build(
        [("a", "Linear", {"units": 4}), ("b", "Linear", {"units": 4})],
        [("a", "b", 0), ("b", "a", 0)]))
    assert not rep["ok"]
    assert any("loop" in e for e in rep["errors"]), rep["errors"]


@check("only runtime blocks may read past an Output")
def _():
    _, rep = analyzed(build(
        [("i", "Input", {"shape": [4]}), ("l", "Linear", {"units": 4}),
         ("o", "Output", {}), ("x", "Linear", {"units": 2})],
        [("i", "l", 0), ("l", "o", 0), ("o", "x", 0)]))
    assert any("Output" in e for e in rep["errors"]), rep["errors"]


@check("a runtime block carries no activation shape")
def _():
    _, rep = analyzed(build(
        [("i", "Input", {"shape": [8, 8, 8]}),
         ("p", "PolicyHead", {"actions": 12}), ("o1", "Output", {}),
         ("v", "ValueHead", {"hidden": 16}), ("o2", "Output", {"task": "regression"}),
         ("m", "MCTSSearch", {})],
        [("i", "p", 0), ("p", "o1", 0), ("i", "v", 0), ("v", "o2", 0),
         ("o1", "m", 0), ("o2", "m", 1)]))
    assert rep["ok"], rep["errors"]
    assert rep["nodes"]["m"]["out_shape"] is None


@check("generated PyTorch mentions every layer")
def _():
    g, rep = analyzed(build(
        [("i", "Input", {"shape": [3, 32, 32]}),
         ("c", "Conv2d", {"filters": 16, "kernel": 3, "padding": "same"}),
         ("a", "Activation", {"kind": "relu"}), ("f", "Flatten", {}),
         ("l", "Linear", {"units": 10}), ("o", "Output", {})],
        [("i", "c", 0), ("c", "a", 0), ("a", "f", 0), ("f", "l", 0), ("l", "o", 0)]))
    assert rep["ok"], rep["errors"]
    src = codegen.to_pytorch(g, rep)
    for token in ("nn.Conv2d", "nn.ReLU", "nn.Flatten", "nn.Linear", "def forward"):
        assert token in src, f"{token} missing from the generated file"


@check("the inspector's code matches the generated file exactly")
def _():
    g, rep = analyzed(build(
        [("i", "Input", {"shape": [3, 32, 32]}),
         ("r", "ResidualBlock", {"filters": 32}),
         ("p", "AdaptiveAvgPool2d", {"size": 1}), ("f", "Flatten", {}),
         ("l", "Linear", {"units": 5}), ("o", "Output", {})],
        [("i", "r", 0), ("r", "p", 0), ("p", "f", 0), ("f", "l", 0), ("l", "o", 0)]))
    assert rep["ok"], rep["errors"]
    per_node = {}
    src = codegen.to_pytorch(g, rep, per_node)
    for nid, entry in per_node.items():
        for line in (entry["init"], entry["call"]):
            if line and not line.startswith(("#", "return")):
                assert line in src, f"{nid}: {line!r} is not in the file"


@check("Keras output flags nodes it cannot translate")
def _():
    g, rep = analyzed(build(
        [("i", "Input", {"shape": [3, 16, 16]}),
         ("c", "Conv2d", {"filters": 8, "kernel": 3, "padding": "same"}),
         ("ode", "ODEBlock", {"field": "conv", "hidden": 16, "steps": 2}),
         ("p", "GlobalAvgPool", {}), ("l", "Linear", {"units": 3}), ("o", "Output", {})],
        [("i", "c", 0), ("c", "ode", 0), ("ode", "p", 0), ("p", "l", 0), ("l", "o", 0)]))
    assert rep["ok"], rep["errors"]
    src = codegen.to_keras(g, rep)
    assert "incomplete" in src.lower(), "the Keras file should say what it could not do"
    assert "ODEBlock" in src


def _run_model(payload, *tensors):
    import train as T

    g, rep = analyzed(payload)
    assert rep["ok"], rep["errors"]
    model = T.build_model(codegen.to_pytorch(g, rep), codegen.model_class_name(g))
    order = codegen.input_order(g, rep)
    ids = [n["id"] for n in payload["nodes"] if n["type"] == "Input"]
    args = [tensors[ids.index(i)] for i in order]
    return model, model(*args), rep


if HAVE_TORCH:
    @check("generated code runs and matches the predicted shape")
    def _():
        import torch
        payload = build(
            [("i", "Input", {"shape": [3, 32, 32]}),
             ("c", "Conv2d", {"filters": 16, "kernel": 3, "stride": 2, "padding": 1}),
             ("b", "BatchNorm2d", {}), ("a", "Activation", {"kind": "gelu"}),
             ("g", "GlobalAvgPool", {}), ("l", "Linear", {"units": 7}), ("o", "Output", {})],
            [("i", "c", 0), ("c", "b", 0), ("b", "a", 0), ("a", "g", 0),
             ("g", "l", 0), ("l", "o", 0)])
        _, y, rep = _run_model(payload, torch.randn(2, 3, 32, 32))
        assert tuple(y.shape)[1:] == tuple(rep["nodes"]["l"]["out_shape"]), y.shape

    @check("the parameter estimate matches PyTorch exactly")
    def _():
        import torch
        payload = build(
            [("i", "Input", {"shape": [3, 32, 32]}),
             ("r", "ResidualBlock", {"filters": 64, "stride": 2}),
             ("s", "SqueezeExcite", {"reduction": 8}),
             ("g", "GlobalAvgPool", {}), ("l", "Linear", {"units": 10}), ("o", "Output", {})],
            [("i", "r", 0), ("r", "s", 0), ("s", "g", 0), ("g", "l", 0), ("l", "o", 0)])
        model, _, rep = _run_model(payload, torch.randn(1, 3, 32, 32))
        actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert actual == rep["total_learnables"], \
            f"canvas says {rep['total_learnables']:,}, torch says {actual:,}"

    @check("a multi-input graph wires towers to the right arguments")
    def _():
        import torch
        payload = build(
            [("a", "Input", {"shape": [6]}), ("b", "Input", {"shape": [4]}),
             ("h1", "Linear", {"units": 8}), ("h2", "Linear", {"units": 8}),
             ("cat", "Concat", {"axis": 0}), ("l", "Linear", {"units": 3}),
             ("o", "Output", {})],
            [("a", "h1", 0), ("b", "h2", 0), ("h1", "cat", 0), ("h2", "cat", 1),
             ("cat", "l", 0), ("l", "o", 0)])
        _, y, _r = _run_model(payload, torch.randn(2, 6), torch.randn(2, 4))
        assert tuple(y.shape) == (2, 3), y.shape

    @check("the causal mask in GPTStack actually masks")
    def _():
        import torch
        import train as T
        payload = build(
            [("t", "Input", {"shape": [16], "dtype": "long"}),
             ("e", "Embedding", {"vocab": 40, "dim": 32}),
             ("g", "GPTStack", {"depth": 2, "heads": 4, "dropout": 0.0}),
             ("h", "Linear", {"units": 40}),
             ("o", "Output", {"task": "language_modeling"})],
            [("t", "e", 0), ("e", "g", 0), ("g", "h", 0), ("h", "o", 0)])
        g, rep = analyzed(payload)
        assert rep["ok"], rep["errors"]
        model = T.build_model(codegen.to_pytorch(g, rep),
                              codegen.model_class_name(g)).eval()
        x = torch.randint(0, 40, (1, 16))
        with torch.no_grad():
            before = model(x)
            x2 = x.clone()
            x2[0, 8] = (x2[0, 8] + 3) % 40
            after = model(x2)
        assert torch.allclose(before[0, :8], after[0, :8], atol=1e-5), \
            "changing a token altered an earlier prediction: the mask is not working"
        assert not torch.allclose(before[0, 8:], after[0, 8:], atol=1e-5), \
            "changing a token had no effect at all"

    @check("importing resnet18 reproduces it exactly")
    def _():
        try:
            import torchvision  # noqa: F401
        except ImportError:
            print("        (torchvision absent, skipped)")
            return
        import importer
        import torchvision
        import train as T

        payload = importer.from_torchvision("resnet18", "none", [3, 224, 224])
        payload.pop("_notes", None)
        g, rep = analyzed(payload)
        assert rep["ok"], rep["errors"][:2]
        model = T.build_model(codegen.to_pytorch(g, rep), codegen.model_class_name(g))
        mine = sum(p.numel() for p in model.parameters())
        ref = sum(p.numel() for p in torchvision.models.resnet18(weights=None).parameters())
        assert mine == ref, f"rebuilt {mine:,} against the original {ref:,}"


# --------------------------------------------------------------------------
# training recipes
# --------------------------------------------------------------------------

import recipeloader  # noqa: E402
import recipes_sdk  # noqa: E402

recipeloader.load_all()
print(f"\nrecipes: {len(recipes_sdk.REGISTRY)} loaded")


@check("recipes all load without error")
def _():
    errors = recipeloader.load_all()
    assert not errors, f"{len(errors)} recipe(s) failed: {errors}"


@check("every recipe declares a step function")
def _():
    for name, recipe in recipes_sdk.REGISTRY.items():
        assert recipe.step is not None, f"{name} has no step"
        assert recipe.doc, f"{name} has no description"


@check("a recipe rejects a graph it cannot train")
def _():
    ctx = recipes_sdk.Context(models={}, device="cpu",
                              cfg=recipes_sdk.REGISTRY["Autoencoder"].defaults(),
                              in_shapes=[[16]], out_shape=[4])
    complaint = recipes_sdk.REGISTRY["Autoencoder"].check(ctx)
    assert complaint and "16" in complaint, complaint


@check("diffusion insists on the timestep channel")
def _():
    ctx = recipes_sdk.Context(models={}, device="cpu",
                              cfg=recipes_sdk.REGISTRY["Diffusion"].defaults(),
                              in_shapes=[[3, 32, 32]], out_shape=[3, 32, 32])
    complaint = recipes_sdk.REGISTRY["Diffusion"].check(ctx)
    assert complaint and "[4, 32, 32]" in complaint, complaint


if HAVE_TORCH:
    @check("an autoencoder trains through the recipe path")
    def _():
        import torch
        import train as T

        payload = build(
            [("i", "Input", {"shape": [12]}), ("e", "Linear", {"units": 4}),
             ("a", "Activation", {"kind": "tanh"}), ("d", "Linear", {"units": 12}),
             ("o", "Output", {})],
            [("i", "e", 0), ("e", "a", 0), ("a", "d", 0), ("d", "o", 0)], "AE")
        g, rep = analyzed(payload)
        assert rep["ok"], rep["errors"]
        job = T.start(codegen.to_pytorch(g, rep),
                      {"recipe": "Autoencoder", "dataset": "synthetic", "epochs": 3,
                       "device": "cpu", "train_samples": 256, "graph": payload,
                       "save_checkpoints": False,
                       "recipe_config": {"lr": 0.02, "noise": 0.0}},
                      [[12]], ["i"], [12], ["classification"], "Ae")
        losses = []
        while True:
            ev = job.events.get(timeout=180)
            if ev["kind"] == "epoch":
                losses.append(ev["val_loss"])
            if ev["kind"] == "error":
                raise AssertionError(ev["message"])
            if ev["kind"] == "finished":
                break
        assert len(losses) == 3, losses
        assert losses[-1] < losses[0], f"reconstruction did not improve: {losses}"

    @check("the DDIM sampler stays inside the data range")
    def _():
        import torch
        recipe = recipes_sdk.REGISTRY["Diffusion"]
        ctx = recipes_sdk.Context(
            models={"main": torch.nn.Conv2d(4, 3, 1)}, device="cpu",
            cfg={**recipe.defaults(), "steps": 40, "preview_steps": 10},
            in_shapes=[[4, 16, 16]], out_shape=[3, 16, 16])
        recipe.setup(ctx)
        ctx.state["lo"], ctx.state["hi"] = 0.0, 1.0
        text = recipe.preview(ctx)
        assert "diverging" not in text, text


@check("every recipe that needs a second network says so")
def _():
    gan = recipes_sdk.REGISTRY["GAN"]
    assert gan.extra_models == ["discriminator"], gan.extra_models
    ctx = recipes_sdk.Context(models={}, device="cpu", cfg=gan.defaults(),
                              in_shapes=[[3, 32, 32]], out_shape=[3, 32, 32])
    complaint = gan.check(ctx)
    assert complaint and "noise" in complaint, complaint


@check("self-supplied recipes declare it")
def _():
    for name in ("Reinforce", "Detection"):
        r = recipes_sdk.REGISTRY[name]
        assert r.self_supplied, f"{name} makes its own data but does not say so"
        assert "none" in r.accepts, f"{name} should not ask for a dataset"


@check("the GAN loader is told about images, not noise")
def _():
    gan = recipes_sdk.REGISTRY["GAN"]
    ctx = recipes_sdk.Context(models={}, device="cpu", cfg=gan.defaults(),
                              in_shapes=[[64]], out_shape=[3, 32, 32])
    assert gan.data_shape(ctx) == [[3, 32, 32]], gan.data_shape(ctx)


@check("detection checks the head width against the class count")
def _():
    d = recipes_sdk.REGISTRY["Detection"]
    ctx = recipes_sdk.Context(models={}, device="cpu", cfg=d.defaults(),
                              in_shapes=[[3, 64, 64]], out_shape=[4, 8, 8])
    complaint = d.check(ctx)
    assert complaint and "7 channels" in complaint, complaint


@check("reinforce checks the policy against the environment")
def _():
    r = recipes_sdk.REGISTRY["Reinforce"]
    ctx = recipes_sdk.Context(models={}, device="cpu", cfg=r.defaults(),
                              in_shapes=[[8]], out_shape=[2])
    complaint = r.check(ctx)
    assert complaint and "4 observations" in complaint, complaint


if HAVE_TORCH:
    @check("a self-supplied recipe trains with no dataset at all")
    def _():
        import train as T

        payload = build(
            [("i", "Input", {"shape": [4]}), ("h", "Linear", {"units": 32}),
             ("a", "Activation", {"kind": "tanh"}), ("o2", "Linear", {"units": 2}),
             ("o", "Output", {})],
            [("i", "h", 0), ("h", "a", 0), ("a", "o2", 0), ("o2", "o", 0)], "Policy")
        g, rep = analyzed(payload)
        assert rep["ok"], rep["errors"]
        job = T.start(codegen.to_pytorch(g, rep),
                      {"recipe": "Reinforce", "epochs": 2, "device": "cpu",
                       "graph": payload, "save_checkpoints": False,
                       "recipe_config": {"lr": 0.01, "steps_per_epoch": 8,
                                         "max_steps": 60}},
                      [[4]], ["i"], [2], ["classification"], "Policy")
        rows = []
        while True:
            ev = job.events.get(timeout=300)
            if ev["kind"] == "epoch":
                rows.append(ev)
            if ev["kind"] == "error":
                raise AssertionError(ev["message"])
            if ev["kind"] == "finished":
                break
        assert len(rows) == 2, rows
        # the objective is the return, and a return can never be negative here
        assert rows[-1]["objective"] == "return"
        assert rows[-1]["train_loss"] > 0, \
            f"return reported as {rows[-1]['train_loss']}: the metric collided again"


# --------------------------------------------------------------------------
# guided projects
# --------------------------------------------------------------------------

import projectloader  # noqa: E402
import projects_sdk  # noqa: E402
from layers import REGISTRY as LAYER_REGISTRY  # noqa: E402

projectloader.load_all()
print(f"\nprojects: {len(projects_sdk.REGISTRY)} in {len(projectloader.categories())} categories")


def build_from_plan(plan, name):
    """Apply a project's steps exactly as the Build tab does."""
    nodes, edges, ids = [], [], {}
    seq = [0]

    def nid():
        seq[0] += 1
        return f"n{seq[0]}"

    for index, step in enumerate(plan):
        detached = step["connect_from"] == "__none__"
        previous = None if detached else (
            ids.get(step["connect_from"]) if step["connect_from"] else ids.get("__last"))
        first_id = step["nodes"][0].get("id") if step["nodes"] else None
        explicit_first = bool(first_id and any(c[1] == first_id for c in step["connect"]))
        placed = []
        for i, spec in enumerate(step["nodes"]):
            node_id = nid()
            nodes.append({"id": node_id, "type": spec["type"],
                          "params": spec["params"] or {}, "label": spec.get("label") or ""})
            placed.append(node_id)
            if spec.get("id"):
                ids[spec["id"]] = node_id
            spec_obj = LAYER_REGISTRY.get(spec["type"])
            if previous and not (i == 0 and explicit_first) and spec_obj \
                    and spec_obj.n_inputs != 0:
                edges.append({"id": f"e{len(edges)}", "source": previous,
                              "target": node_id, "port": 0})
            previous = node_id
        for src, dst, port in step["connect"]:
            a, b = ids.get(src), ids.get(dst)
            if a and b and not any(e["source"] == a and e["target"] == b for e in edges):
                edges.append({"id": f"x{len(edges)}", "source": a, "target": b, "port": port})
        if not detached:
            ids["__last"] = previous
        ids[f"__step{index + 1}"] = placed[-1] if placed else previous
    return {"name": name, "nodes": nodes, "edges": edges}


@check("projects all load without error")
def _():
    errors = projectloader.load_all()
    assert not errors, f"{len(errors)} project file(s) failed: {errors}"


@check("every project has a summary, steps and reasoning")
def _():
    for pid, p in projects_sdk.REGISTRY.items():
        assert p.summary, f"{pid} has no summary"
        assert p.steps, f"{pid} has no steps"
        for i, step in enumerate(p.steps):
            assert step.why, f"{pid} step {i} has no explanation"
            assert step.title, f"{pid} step {i} has no title"


@check("every project's steps use layers that exist")
def _():
    for pid, p in projects_sdk.REGISTRY.items():
        for step in p.steps:
            for spec in step.nodes:
                assert spec["type"] in LAYER_REGISTRY, \
                    f"{pid} places a {spec['type']}, which is not in the registry"


@check("every project builds into a graph that resolves and generates code")
def _():
    broken = []
    for pid in projects_sdk.REGISTRY:
        p = projectloader.get(pid)
        slug = "".join(c for c in p["name"] if c.isalnum()) or "P"
        payload = build_from_plan(p["plan"], slug)
        g = G.parse(payload)
        rep = G.analyze(g)
        if not rep["ok"]:
            err = (rep["errors"]
                   or [v["error"] for v in rep["nodes"].values() if v["error"]])[:1]
            broken.append(f"{pid}: {err[0] if err else '?'}")
            continue
        try:
            codegen.to_pytorch(g, rep)
        except Exception as exc:  # noqa: BLE001
            broken.append(f"{pid}: codegen {exc}")
    assert not broken, f"{len(broken)} project(s) do not build: " + "; ".join(broken[:4])


@check("the request matcher finds the right project")
def _():
    expected = {
        "classify histopathology slides": "med-mil-slide",
        "forecast weekly sales": "seq-sales",
        "something with molecules": "graph-molecule",
        "balance a pole": "rl-cartpole",
        "spoken keyword spotting": "audio-keyword",
    }
    for query, wanted in expected.items():
        hits = [m["id"] for m in projectloader.suggest(query, 3)["matches"]]
        assert hits and hits[0] == wanted, f"{query!r} gave {hits[:3]}, wanted {wanted}"


@check("the matcher admits when it has no idea")
def _():
    result = projectloader.suggest("underwater basket weaving")
    assert not result["confident"]
    assert result["advice"], "a miss should still say something useful"


# --------------------------------------------------------------------------
# saved designs are versioned
# --------------------------------------------------------------------------

@check("saving a design creates a new version rather than overwriting")
def _():
    # The route functions are called directly rather than through Starlette's
    # TestClient, which needs an HTTP client library. The suite is meant to run
    # in a bare checkout, so it should not pull one in for a filesystem test.
    import main
    from fastapi import HTTPException

    name = "__version_test__"
    payload = type("Body", (), {})()

    def cleanup():
        try:
            main.delete_graph(name)
        except HTTPException:
            pass

    cleanup()
    graph = {"name": name, "nodes": [
        {"id": "i", "type": "Input", "params": {"shape": [4]}},
        {"id": "l", "type": "Linear", "params": {"units": 2}},
        {"id": "o", "type": "Output", "params": {}}],
        "edges": [{"id": "e1", "source": "i", "target": "l"},
                  {"id": "e2", "source": "l", "target": "o"}]}
    try:
        widths = [2, 8, 32]
        reply = None
        for width in widths:
            graph["nodes"][1]["params"]["units"] = width
            payload.graph = json.loads(json.dumps(graph))
            reply = main.save_graph(name, payload)
        assert reply["version"] == 3, reply

        listed = main.graph_versions(name)
        assert len(listed["versions"]) == 3, listed
        assert listed["latest"] == 3, listed

        for version, width in zip((1, 2, 3), widths):
            got = main.load_graph(name, version=version)
            assert got["nodes"][1]["params"]["units"] == width, \
                f"version {version} should still hold width {width}, got {got['nodes'][1]['params']}"

        assert main.load_graph(name)["nodes"][1]["params"]["units"] == 32, \
            "loading without a version should give the newest"

        try:
            main.load_graph(name, version=99)
            raise AssertionError("a version that does not exist should raise")
        except HTTPException as exc:
            assert exc.status_code == 404
    finally:
        cleanup()


# --------------------------------------------------------------------------
# what a design requires
# --------------------------------------------------------------------------

import needs  # noqa: E402


@check("requirements name the blocks a design pulls in")
def _():
    payload = build(
        [("i", "Input", {"shape": [3, 32, 32]}),
         ("r", "ResidualBlock", {"filters": 32}),
         ("g", "GlobalAvgPool", {}), ("l", "Linear", {"units": 4}), ("o", "Output", {})],
        [("i", "r", 0), ("r", "g", 0), ("g", "l", 0), ("l", "o", 0)])
    g, rep = analyzed(payload)
    assert rep["ok"], rep["errors"]
    req = needs.requirements(g, rep)
    names = [b["name"] for b in req["blocks"]]
    assert names == ["ResidualBlock"], names
    assert req["blocks"][0]["file"] == "residual.py", req["blocks"][0]


@check("requirements flag a download and a Keras gap")
def _():
    payload = build(
        [("i", "Input", {"shape": [3, 224, 224]}),
         ("b", "Backbone", {"arch": "resnet18", "weights": "DEFAULT"}),
         ("g", "GlobalAvgPool", {}), ("l", "Linear", {"units": 5}), ("o", "Output", {})],
        [("i", "b", 0), ("b", "g", 0), ("g", "l", 0), ("l", "o", 0)])
    g, rep = analyzed(payload)
    assert rep["ok"], rep["errors"]
    req = needs.requirements(g, rep)
    assert req["pretrained"] and req["pretrained"][0]["arch"] == "resnet18"
    assert any("download" in n for n in req["notes"]), req["notes"]
    assert "Backbone" in req["keras_gaps"], req["keras_gaps"]
    assert any(p["name"] == "torchvision" for p in req["packages"])


@check("requirements pick datasets that actually fit the inputs")
def _():
    tabular = build(
        [("i", "Input", {"shape": [12]}), ("l", "Linear", {"units": 3}),
         ("o", "Output", {})],
        [("i", "l", 0), ("l", "o", 0)])
    g, rep = analyzed(tabular)
    req = needs.requirements(g, rep)
    assert "csv" in req["datasets"] and "cifar10" not in req["datasets"], req["datasets"]

    tokens = build(
        [("i", "Input", {"shape": [64], "dtype": "long"}),
         ("e", "Embedding", {"vocab": 50, "dim": 32}),
         ("g", "GlobalAvgPool", {}), ("l", "Linear", {"units": 50}),
         ("o", "Output", {"task": "language_modeling"})],
        [("i", "e", 0), ("e", "g", 0), ("g", "l", 0), ("l", "o", 0)])
    g, rep = analyzed(tokens)
    req = needs.requirements(g, rep)
    assert "text" in req["datasets"], req["datasets"]


@check("requirements warn about multi-input ordering")
def _():
    payload = build(
        [("a", "Input", {"shape": [6]}), ("b", "Input", {"shape": [4]}),
         ("h1", "Linear", {"units": 8}), ("h2", "Linear", {"units": 8}),
         ("cat", "Concat", {"axis": 0}), ("l", "Linear", {"units": 2}),
         ("o", "Output", {})],
        [("a", "h1", 0), ("b", "h2", 0), ("h1", "cat", 0), ("h2", "cat", 1),
         ("cat", "l", 0), ("l", "o", 0)])
    g, rep = analyzed(payload)
    assert rep["ok"], rep["errors"]
    req = needs.requirements(g, rep)
    assert any("Inputs" in n and "order" in n for n in req["notes"]), req["notes"]


# --------------------------------------------------------------------------
# executions are recorded, not just streamed
# --------------------------------------------------------------------------

if HAVE_TORCH:
    @check("a training run is recorded to disk with its design and version")
    def _():
        import train as T

        payload = build(
            [("i", "Input", {"shape": [10]}), ("h", "Linear", {"units": 8}),
             ("a", "Activation", {"kind": "relu"}), ("l", "Linear", {"units": 3}),
             ("o", "Output", {"task": "classification"})],
            [("i", "h", 0), ("h", "a", 0), ("a", "l", 0), ("l", "o", 0)],
            "__run_test__")
        g, rep = analyzed(payload)
        assert rep["ok"], rep["errors"]
        job = T.start(codegen.to_pytorch(g, rep),
                      {"dataset": "synthetic", "epochs": 2, "device": "cpu",
                       "train_samples": 96, "graph": payload, "design_version": 7,
                       "save_checkpoints": False, "batch_size": 16},
                      [[10]], ["i"], [3], ["classification"],
                      codegen.model_class_name(g))
        try:
            while True:
                event = job.events.get(timeout=180)
                if event["kind"] == "error":
                    raise AssertionError(event["message"])
                if event["kind"] == "finished":
                    break

            stored = T.read_run(job.id)
            assert stored["design"] == "__run_test__", stored["design"]
            assert stored["version"] == 7, stored["version"]
            assert stored["status"] == "done", stored["status"]
            assert len(stored["history"]) == 2, stored["history"]
            assert stored["graph"]["nodes"], "the run should keep the design it used"

            listed = [r for r in T.list_runs() if r["id"] == job.id]
            assert listed, "the run should appear in the listing"
            assert listed[0]["best"] is not None

            filtered = T.list_runs(design="__run_test__")
            assert all(r["design"] == "__run_test__" for r in filtered), filtered
        finally:
            T.delete_run(job.id)


@check("a run that cannot start is still recorded, with its reason")
def _():
    import train as T

    payload = build(
        [("i", "Input", {"shape": [10]}), ("l", "Linear", {"units": 3}),
         ("o", "Output", {})],
        [("i", "l", 0), ("l", "o", 0)], "__fail_test__")
    g, rep = analyzed(payload)
    job = T.start("this is not python", {"dataset": "synthetic", "epochs": 1,
                                         "device": "cpu", "graph": payload},
                  [[10]], ["i"], [3], ["classification"], "Nope")
    try:
        while True:
            event = job.events.get(timeout=60)
            if event["kind"] in ("error", "finished"):
                break
        stored = T.read_run(job.id)
        assert stored["status"] == "error", stored["status"]
        assert stored["error"], "a failed run should record why"
    finally:
        T.delete_run(job.id)


# --------------------------------------------------------------------------
# the page itself
# --------------------------------------------------------------------------

PAGE = (ROOT / "frontend" / "index.html").read_text()


@check("the stylesheet has balanced braces")
def _():
    # An unclosed rule makes the browser discard every rule after it, which
    # looks like "the design broke" rather than "the CSS is malformed". This
    # exact fault shipped once.
    css = PAGE[PAGE.index("<style>") + 7: PAGE.index("</style>")]
    opened, closed = css.count("{"), css.count("}")
    assert opened == closed, f"{opened} open braces against {closed} close"


@check("no rule is left with an empty body")
def _():
    import re

    css = PAGE[PAGE.index("<style>") + 7: PAGE.index("</style>")]
    empty = re.findall(r"([^{}\n]+)\{\s*\}", css)
    assert not empty, f"empty rules: {[e.strip() for e in empty][:4]}"


@check("every element the script reaches for exists or is built at runtime")
def _():
    import re

    markup = PAGE[: PAGE.index("<script>")]
    script = PAGE[PAGE.index("<script>"):]
    present = set(re.findall(r'id="([^"]+)"', markup))
    created = set(re.findall(r'id="([^"]+)"', script))
    created |= set(re.findall(r'id=\\?"([^"\\]+)', script))
    # elements built in code assign their id rather than carrying it in markup
    created |= set(re.findall(r'\.id\s*=\s*"([^"]+)"', script))
    wanted = set(re.findall(r'\$\("([a-zA-Z0-9_]+)"\)', script))
    missing = sorted(w for w in wanted if w not in present and w not in created)
    assert not missing, f"the script looks for elements nothing creates: {missing}"


@check("the canvas draws a shape per layer role")
def _():
    for token in ("node-terminal", "node-diamond", "node-hex", "node-card",
                  "gridMinor", "port-plus"):
        assert token in PAGE, f"{token} is not in the page"
    # a circle for entry and exit, a diamond for merges, a hexagon for runtime
    assert 'shape: "circle"' in PAGE and 'shape: "diamond"' in PAGE \
        and 'shape: "hex"' in PAGE, "nodeBox does not assign all four shapes"


@check("every implicit domain and arm builds, counts and runs")
def _():
    """Each domain is a theorem; each arm is either covered by it or a control
    that breaks one condition at the same parameter count. All of them have to
    work as layers, and the matched counts are what make the comparison fair."""
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    try:
        from dnn_bench import core, domains  # noqa: F401
    except ImportError:
        print("        (dnn_bench absent, skipped)")
        return
    import torch

    import train as T

    assert "ImplicitEquilibrium" in layers.REGISTRY, "the block did not install"

    counts: dict = {}
    for entry in core.all_domains():
        spec = core.get(entry["key"])
        arms = spec.variants(spec.defaults(), seed=0)
        assert any(v.structured for v in arms), \
            f"{entry['key']} ships no arm the theorem covers"
        assert any(not v.structured for v in arms), \
            f"{entry['key']} ships no control, so nothing can be compared"

        for arm in arms:
            payload = build(
                [("i", "Input", {"shape": [10]}),
                 ("c", "ImplicitEquilibrium", {"units": 6, "domain": entry["key"],
                                               "variant": arm.key, "seed": 0}),
                 ("o", "Output", {"task": "classification"})],
                [("i", "c", 0), ("c", "o", 0)], "Imp")
            g, rep = analyzed(payload)
            assert rep["ok"], f"{entry['key']}/{arm.key}: {rep['errors'][:1]}"

            model = T.build_model(codegen.to_pytorch(g, rep),
                                  codegen.model_class_name(g)).double()
            real = sum(q.numel() for q in model.parameters())
            assert real == rep["total_learnables"], \
                f"{entry['key']}/{arm.key}: canvas {rep['total_learnables']}, torch {real}"
            with torch.no_grad():
                out = model(torch.randn(2, 10, dtype=torch.float64))
            assert tuple(out.shape) == (2, 6)
            counts.setdefault(entry["key"], set()).add(real)

    # within a domain the arms must cost the same, or the ablation is confounded
    for domain, sizes in counts.items():
        assert len(sizes) == 1, \
            f"{domain} arms differ in size ({sorted(sizes)}), so they are not comparable"


@check("a design cannot shadow the layer it is named after")
def _():
    """Naming a design ResidualBlock put a class of that name in the same file
    as the prelude defining the layer, and the model quietly shadowed it. The
    symptom was a TypeError about arguments the model does not take."""
    reserved = ["ResidualBlock", "CapsuleLayer", "Model", "GraphConv"]
    for name in reserved:
        g = G.parse({"name": name,
                     "nodes": [{"id": "i", "type": "Input", "params": {"shape": [4]}},
                               {"id": "o", "type": "Output", "params": {}}],
                     "edges": [{"id": "e", "source": "i", "target": "o"}]})
        cls = codegen.model_class_name(g)
        assert cls not in layers.REGISTRY, f"{name} still generates class {cls}"
        assert cls != "Model", "the model class collides with the default name"

    # an ordinary name is left alone
    g = G.parse({"name": "MyNet",
                 "nodes": [{"id": "i", "type": "Input", "params": {"shape": [4]}},
                           {"id": "o", "type": "Output", "params": {}}],
                 "edges": [{"id": "e", "source": "i", "target": "o"}]})
    assert codegen.model_class_name(g) == "MyNet"

    # and the file that is written declares the class the loader looks for
    if HAVE_TORCH:
        import train as T

        g, rep = analyzed({"name": "CapsuleLayer",
                           "nodes": [{"id": "i", "type": "Input",
                                      "params": {"shape": [4]}},
                                     {"id": "l", "type": "Linear",
                                      "params": {"units": 2}},
                                     {"id": "o", "type": "Output", "params": {}}],
                           "edges": [{"id": "e1", "source": "i", "target": "l"},
                                     {"id": "e2", "source": "l", "target": "o"}]})
        source = codegen.to_pytorch(g, rep)
        name = codegen.model_class_name(g)
        assert f"class {name}(" in source, \
            "the emitter and the lookup disagree about the class name"
        T.build_model(source, name)


@check("the architectures that do not fit the mould still work")
def _():
    """Four layers whose oddness is contained in forward(): a routing loop, a
    simulated time loop, an energy model, and a draw from a distribution.

    Each is checked for the behaviour that makes it that architecture, not just
    for producing a tensor of the right shape — a capsule layer whose routing
    does nothing still returns the right shape.
    """
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import train as T

    for name in ("CapsuleLayer", "SpikingDense", "RBM", "Sampling",
                 "GraphAttention", "MessagePassing"):
        assert name in layers.REGISTRY, f"{name} did not install"

    scope = {"torch": torch, "nn": nn, "F": F}
    for name in ("CapsuleLayer", "SpikingDense", "RBM", "Sampling"):
        exec(layers.REGISTRY[name].torch_prelude, scope)

    torch.manual_seed(0)

    # routing has to change the answer, or the loop is decorative
    three = scope["CapsuleLayer"](16, 8, 6, 12, routes=3).eval()
    one = scope["CapsuleLayer"](16, 8, 6, 12, routes=1).eval()
    one.load_state_dict(three.state_dict())
    x = torch.randn(2, 16, 8)
    with torch.no_grad():
        moved = float((three(x) - one(x)).abs().max())
        norms = three(x).norm(dim=-1)
    assert moved > 1e-3, f"routing changed the output by {moved:.2e}: it is inert"
    assert float(norms.max()) < 1.0, "squash should keep capsule norms below 1"
    assert float(norms.mean()) > 0.1, \
        "capsule norms are near zero, so the initialisation is too small"

    # spikes: bounded, quantised by the step count, and differentiable anyway
    snn = scope["SpikingDense"](12, 6, steps=10)
    x = torch.randn(4, 12, requires_grad=True)
    rates = snn(x)
    assert float(rates.min()) >= 0 and float(rates.max()) <= 1, "rates left [0,1]"
    values = {round(v, 6) for v in rates.detach().flatten().tolist()}
    assert len(values) <= 11, f"{len(values)} distinct rates from 10 steps"
    rates.sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0, \
        "no gradient through the surrogate, so it cannot train"

    # an RBM has an energy and a sampler, which is what makes it one
    rbm = scope["RBM"](10, 5)
    v = (torch.rand(3, 10) > 0.5).float()
    assert rbm.free_energy(v).shape == (3,), "free energy is not per example"
    assert rbm.gibbs(v, steps=2).shape == v.shape, "gibbs did not return a visible"

    # the trick: random while training, the mean when measured, gradient to both
    sampler = scope["Sampling"]()
    mu = torch.zeros(4, 5, requires_grad=True)
    logvar = torch.zeros(4, 5, requires_grad=True)
    sampler.train()
    assert not torch.equal(sampler(mu, logvar), sampler(mu, logvar)), \
        "two training draws were identical, so nothing is being sampled"
    sampler.eval()
    assert torch.equal(sampler(mu, logvar), mu), \
        "evaluation is not deterministic, so a measurement cannot be repeated"
    sampler.train()
    sampler(mu, logvar).sum().backward()
    assert float(mu.grad.abs().sum()) > 0 and float(logvar.grad.abs().sum()) > 0, \
        "the gradient does not reach both parameters"
    assert abs(float(scope["Sampling"].kl(mu, logvar))) < 1e-6, \
        "the KL of a standard normal should be zero"


@check("every new layer builds, counts and runs on the canvas")
def _():
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import torch

    import train as T

    cases = [
        ("CapsuleLayer", [32, 8], {"capsules": 10, "dim": 16, "routes": 3}, 1),
        ("SpikingDense", [64], {"units": 32, "steps": 8}, 1),
        ("RBM", [64], {"units": 32}, 1),
        ("Sampling", [16], {}, 2),
        ("GraphAttention", [10, 16], {"units": 8, "heads": 2}, 2),
        ("MessagePassing", [10, 16], {"units": 12, "hidden": 12}, 2),
    ]
    for name, shape, params, inputs in cases:
        second = [10, 10] if name in ("GraphAttention", "MessagePassing") else shape
        nodes = [{"id": "i", "type": "Input", "params": {"shape": shape}, "y": 0}]
        edges = [{"id": "e1", "source": "i", "target": "c", "port": 0}]
        if inputs == 2:
            nodes.append({"id": "j", "type": "Input",
                          "params": {"shape": second}, "y": 100})
            edges.append({"id": "e2", "source": "j", "target": "c", "port": 1})
        nodes += [{"id": "c", "type": name, "params": params, "y": 200},
                  {"id": "o", "type": "Output", "params": {}, "y": 300}]
        edges.append({"id": "e9", "source": "c", "target": "o"})

        g, rep = analyzed({"name": name, "nodes": nodes, "edges": edges})
        assert rep["ok"], f"{name}: {rep['errors'][:1]}"

        model = T.build_model(codegen.to_pytorch(g, rep),
                              codegen.model_class_name(g))
        real = sum(q.numel() for q in model.parameters())
        assert real == rep["total_learnables"], \
            f"{name}: canvas {rep['total_learnables']}, torch {real}"

        order = codegen.input_order(g, rep)
        feed = {"i": torch.randn(3, *shape),
                "j": (torch.rand(3, *second) > 0.5).float()
                     if name in ("GraphAttention", "MessagePassing")
                     else torch.randn(3, *second)}
        with torch.no_grad():
            model(*[feed[k] for k in order])


@check("the domains page shows what each arm actually is")
def _():
    """The facts come from the built layer describing itself, not from the spec
    that asked for it — so the page cannot claim a width or a parameter count
    the layer does not have."""
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    try:
        from dnn_bench import core, domains  # noqa: F401
    except ImportError:
        print("        (dnn_bench absent, skipped)")
        return
    import main

    listed = main.list_domains(seed=0)
    assert listed["domains"], "no domains listed"

    for entry in listed["domains"]:
        arms = entry.get("arms") or []
        assert arms, f"{entry['key']} lists no arms"
        assert sum(1 for a in arms if a.get("structured")) == 1, \
            f"{entry['key']} does not have exactly one covered arm"
        sizes = {a["parameters"] for a in arms if "parameters" in a}
        assert len(sizes) == 1, \
            f"{entry['key']} arms differ in size {sorted(sizes)}, so the page " \
            f"would be showing an unfair comparison"
        for arm in arms:
            assert arm.get("facts"), f"{entry['key']}/{arm['key']} describes nothing"
            if arm.get("structured"):
                assert arm["holds"], "the covered arm does not claim the guarantee"
                assert arm["claim"], "the covered arm states no claim"
            else:
                assert not arm["holds"], "a control claims the guarantee"
                assert arm["label"], "a control does not say what it breaks"

    for token in ("function renderDomainsPage", "function armCard",
                  "function placeDomain", 'id="pageDomains"'):
        assert token in PAGE, f"{token} is missing from the domains page"


@check("a design opened from elsewhere arrives on the canvas, in view")
def _():
    """Two faults that only appeared once the launch pad opened first.

    Importing left you on the launch pad with no sign anything had happened.
    And the fit ran while the canvas was hidden, where the SVG reports zero
    size — so the graph was centred inside a zero box and ended up off screen.
    Clicking Canvas then showed an empty-looking canvas with the design
    somewhere outside it.
    """
    script = PAGE[PAGE.index("<script>"):]

    fit = script[script.index("function fitView"):]
    fit = fit[: fit.index("\n}\n")]
    assert "r.width < 2" in fit, \
        "fitView computes a view from a hidden canvas, which has no size"
    assert "state.needsFit = true" in fit, "a deferred fit is not remembered"

    show = script[script.index("function showPage"):]
    show = show[: show.index("\n}\n")]
    assert "state.needsFit" in show, \
        "a fit deferred while hidden is never honoured"

    # anything that loads a whole design has to land you where you can see it
    for name in ("openDesign", "openRun", "importFolderPicks", "buildFromDomain"):
        body = script[script.index(f"function {name}("):]
        body = body[: body.index("\n}\n")]
        if "fitView()" not in body:
            continue
        assert 'showPage("pageDesign")' in body, \
            f"{name} loads a design without bringing the canvas up"


@check("the launch pad is what opens")
def _():
    import re

    on = re.findall(r'<section class="page on" id="(page\w+)"', PAGE)
    assert on == ["pagePad"], f"the page that opens is {on}"
    rail = PAGE[PAGE.index('<nav id="rail">'):PAGE.index("</nav>")]
    lit = re.findall(r'<button data-page="(\w+)"[^>]*class="[^"]*\bon\b', rail)
    assert lit == ["pagePad"], f"the rail says {lit} is open instead"

    boot = PAGE[PAGE.index("async function boot"):]
    boot = boot[: boot.index("\n}\n")]
    assert "renderLaunchPad()" in boot, "boot never fills the launch pad"


@check("a scout verifies what it finds rather than describing it")
def _():
    """The useful thing about a scout here is not that it can search — anyone
    can search — but that this application can check. So every finding has to
    carry evidence produced by running it, and an import whose parameter count
    disagrees with torch is not a success: the graph is not the model, and
    every number taken from it afterwards would be about something else.
    """
    import scouts

    ids = {entry["id"] for entry in scouts.CATALOG}
    assert ids == {"code", "papers", "draft", "maths"}, ids

    # A drafted design must be checked before it is offered: assembled,
    # counted against torch, and run. Putting an unverified one on the canvas
    # would look exactly like an answer.
    drafted = inspect.getsource(scouts._verify)
    assert "real != counted" in drafted, "the count is not checked against torch"
    assert "model(*[" in drafted, "no batch is put through it"
    for failure in ("the shapes do not resolve", "it assembled but would not run",
                    "the parameter counts disagree"):
        assert failure in drafted, f"the {failure!r} case is not distinguished"

    source = inspect.getsource(scouts)
    # the ranking must be built from what was verified, not from popularity
    assert "len(worked) * 10" in source, "the shortlist is not ranked on evidence"
    assert 'if t["imported"] and t.get("exact", True)' in source, \
        "an inexact import counts as a success"
    assert "the graph is not the model" in source, \
        "an inexact import does not say what is wrong with it"

    # a class needing arguments must not be guessed at: a guess would make the
    # evidence worthless, which is worse than no evidence
    assert "was not guessed at" in source, "constructor arguments are invented"

    # the maths scout runs on the canvas alone, so it can be checked here
    g = build([("i", "Input", {"shape": [3, 32, 32]}),
               ("c", "Conv2d", {"filters": 32, "kernel": 3}),
               ("g", "GlobalAvgPool", {}),
               ("h", "Linear", {"units": 10}),
               ("o", "Output", {"task": "classification"})],
              [("i", "c", 0), ("c", "g", 0), ("g", "h", 0), ("h", "o", 0)],
              "Explain")
    scout = scouts.start("maths", "", {}, g)
    for _ in range(80):
        if scout.status != "running":
            break
        time.sleep(0.1)
    assert scout.status == "done", f"{scout.status}: {scout.error}"
    assert len(scout.findings) == 5, len(scout.findings)

    conv = next(f for f in scout.findings if f["type"] == "Conv2d")
    assert conv["equation"], "the convolution has no equation"
    assert conv["parameters"] == 3 * 3 * 3 * 32 + 32, conv["parameters"]
    assert conv["share"] > 50, "the share of parameters is not reported"

    # an unknown kind is refused rather than started
    bad = scouts.start("nonsense", "x", {}, {})
    assert bad.status == "error" and "nonsense" in bad.error

    # the two fitting boxes exist and are read when the button is pressed
    assert 'id="draftShape"' in PAGE and 'id="draftClasses"' in PAGE
    sender = PAGE[PAGE.index('const config = {};'):]
    sender = sender[: sender.index("sendScout(")]
    assert "config.shape" in sender and "config.classes" in sender, \
        "the shape and class boxes are shown but never sent"

    # placing one must refuse anything unverified, not merely hide the button
    place = PAGE[PAGE.index("function placeDraft"):]
    place = place[: place.index("\n}\n")]
    assert "!f.ok" in place, \
        "placeDraft trusts the button rather than checking the finding"

    # a drafted design is only placeable once verified
    card = PAGE[PAGE.index("function scoutCard"):]
    draft_part = card[card.index('f.kind === "draft"'):]
    draft_part = draft_part[: draft_part.index('f.kind === "paper"')]
    assert "f.ok\n        ? `<button" in draft_part or "${f.ok" in draft_part, \
        "an unverified draft can be placed"
    assert "data-draft=" in draft_part

    # and the page distinguishes the three verdicts. The class names are built
    # by interpolation, so the literal strings never appear — check the
    # expression that produces them and the styles that render them.
    assert 't.exact === false ? "warn" : "ok"' in PAGE, \
        "the page does not tell an inexact import from an exact one"
    assert "function openScoutFind" in PAGE
    for style in (".tried.ok{", ".tried.warn{", ".tried.no{"):
        assert style in PAGE, f"{style} is not styled, so the verdicts look alike"

    # Only an exact import may be opened: opening an unfaithful graph would
    # quietly hand back a different network. Anchored inside scoutCard —
    # data-open is also the design picker's attribute, and searching the whole
    # page found that one instead.
    card = PAGE[PAGE.index("function scoutCard"):]
    card = card[: card.index("\nfunction openScoutFind")]
    at = card.index("data-open=")
    assert "t.exact !== false" in card[max(0, at - 200): at], \
        "an inexact import can be opened on the canvas"


@check("the advice is about this design, not about designs in general")
def _():
    """A list of general advice is the same list for every network, which is
    the same as no advice. Every suggestion here has to be one this particular
    graph invites, and has to carry what it would cost."""
    import advisor

    # a design with real faults: no norm after the conv, two dense layers with
    # nothing between them, and a flatten handing over a fortune
    g, rep = analyzed(build(
        [("i", "Input", {"shape": [3, 32, 32]}),
         ("c", "Conv2d", {"filters": 64, "kernel": 5}),
         ("f", "Flatten", {}),
         ("d", "Linear", {"units": 512}),
         ("h", "Linear", {"units": 10}),
         ("o", "Output", {"task": "classification"})],
        [("i", "c", 0), ("c", "f", 0), ("f", "d", 0), ("d", "h", 0), ("h", "o", 0)],
        "Faulty"))
    out = advisor.advise(g, rep, selected="c")
    titles = " ".join(i["title"].lower() for i in out["ideas"])
    assert "activation" in titles, "two dense layers in a row went unremarked"
    assert "normaliz" in titles, "an unnormalized convolution went unremarked"
    assert "pool" in titles, "a flatten over 65k values went unremarked"
    for idea in out["ideas"]:
        assert idea["why"], f"{idea['title']} gives no reason"
        assert idea["cost"], f"{idea['title']} does not say what it costs"

    # and a clean design must not be given the same advice anyway
    clean, rep2 = analyzed(build(
        [("i", "Input", {"shape": [3, 32, 32]}),
         ("c", "Conv2d", {"filters": 16, "kernel": 3}),
         ("n", "BatchNorm2d", {}),
         ("a", "Activation", {"kind": "relu"}),
         ("g", "GlobalAvgPool", {}),
         ("h", "Linear", {"units": 10}),
         ("o", "Output", {"task": "classification"})],
        [("i", "c", 0), ("c", "n", 0), ("n", "a", 0), ("a", "g", 0),
         ("g", "h", 0), ("h", "o", 0)], "Clean"))
    tidy = " ".join(i["title"].lower() for i in advisor.advise(clean, rep2)["ideas"])
    assert "normaliz" not in tidy, "advised normalizing a layer that is normalized"
    assert "activation" not in tidy, "advised an activation where there is one"

    # the selected layer is described from the graph, with its real numbers
    layer = out["layer"]
    assert layer["out_shape"] == [64, 32, 32], layer["out_shape"]
    assert layer["parameters"] == 3 * 5 * 5 * 64 + 64, layer["parameters"]
    kernel = next(k for k in layer["knobs"] if k["name"] == "kernel")
    assert "\u00d7" in kernel["effect"], "the effect text has an unrendered escape"
    assert "5\u00d75" in kernel["effect"], kernel["effect"]

    # an unparseable graph must not take the panel down
    broken = advisor.advise.__module__  # keep the import used
    import main

    answer = main.assistant_advise(
        main.AdvicePayload(graph={"nodes": [{"bad": True}], "edges": []}))
    assert "ideas" in answer, answer

    for token in ("function dockAdvice", "advidea", "data-do="):
        assert token in PAGE, f"{token} is missing from the dock"


@check("the assistant dock talks to the assistant")
def _():
    """The dock posts to the same endpoint the panel does, and the endpoint
    reads `message`. Sending `text` would have looked like it worked — a reply
    comes back either way, just the wrong one."""
    import inspect

    import main

    fields = set(main.AssistantPayload.model_fields)
    script = PAGE[PAGE.index("<script>"):]
    body = script[script.index("async function dockSend"):]
    body = body[: body.index("\n}\n")]
    assert "message:" in body, \
        "the dock does not send `message`, which is the field the endpoint reads"
    assert "message" in fields, fields

    for token in ('id="dockTab"', "function toggleDock", "function dockWelcome",
                  "function assistantDock"):
        assert token in PAGE, f"{token} is missing from the dock"

    # it must be reachable from every page, not parented to the canvas
    markup = PAGE[: PAGE.index("<script>")]
    tab = markup[markup.index('id="dockTab"') - 200: markup.index('id="dockTab"')]
    assert "<section" not in tab, "the launcher sits inside a page"


@check("every launch pad route goes somewhere real")
def _():
    """A door, not a dashboard — and a door with a dead handle is worse than
    no door."""
    script = PAGE[PAGE.index("<script>"):]
    assert "const ROUTES" in script, "there are no routes"
    block = script[script.index("const ROUTES"):]
    block = block[: block.index("\n];")]

    import re

    pages = set(re.findall(r'<section class="page[^"]*" id="(page\w+)"', PAGE))
    targets = set(re.findall(r'showPage\("(\w+)"\)', block))
    assert targets, "no route opens a page"
    missing = targets - pages
    assert not missing, f"routes point at pages that do not exist: {sorted(missing)}"

    # every function a route calls has to exist
    called = set(re.findall(r"(\w+)\(", block)) - {"showPage", "if", "for"}
    for name in called:
        if name.startswith("$") or name in ("map", "join", "filter"):
            continue
        assert f"function {name}" in script or f"{name} =" in script, \
            f"a route calls {name}, which is not defined"

    # and a route must not fake a click on a control it does not own
    assert '.click()' not in block, \
        "a route synthesises a click rather than calling the function"

    cards = len(re.findall(r'data-route="', PAGE))
    keys = len(re.findall(r'key: "', block))
    assert cards == 0 or keys > 0, "the cards and the routes disagree"


@check("every page paints its own background")
def _():
    """The page background is dark, so a page that does not paint one shows it.

    This has now happened twice: once as a black canvas, once as a page of
    dark-on-dark text. Both times the page itself was fine and simply had no
    surface under it.
    """
    import re

    css = PAGE[PAGE.index("<style>"): PAGE.index("</style>")]
    painted = set()
    for match in re.finditer(r"([^{}]*)\{([^}]*)\}", css):
        if "background" not in match.group(2):
            continue
        for part in match.group(1).split(","):
            part = part.strip()
            if part.startswith("#page"):
                painted.add(part.split(":")[0].split()[0])

    pages = set(re.findall(r'<section class="page[^"]*" id="(page\w+)"', PAGE))
    assert pages, "no pages found at all"
    unpainted = {p for p in pages if f"#{p}" not in painted}
    # the design page is the canvas, which paints itself through #stage
    unpainted.discard("pageDesign")
    assert not unpainted, \
        f"these pages let the dark page background through: {sorted(unpainted)}"


@check("a question can be turned into a reading list")
def _():
    """Search is the only part of the app that touches the internet, so it is
    also the only part that can fail for reasons nothing here controls. What is
    checked is the shaping and the honesty, not the network."""
    try:
        from dnn_bench import paper_to_spec as p2s
    except ImportError:
        print("        (dnn_bench absent, skipped)")
        return
    import main

    # a paywalled result offers no button, because a button that can only fail
    # is worse than none
    walled = p2s.Hit(source="MED", id="1", title="A minimal model",
                     year="1979", cited_by=812, recurrence=4,
                     reasons=["unique steady state"])
    shaped = main._hit_json(walled)
    assert shaped["text_url"] is None, "a paywalled paper claims fetchable text"
    assert shaped["cited_by"] == 812 and shaped["recurrence"] == 4
    assert shaped["reasons"] == ["unique steady state"]

    # an arXiv paper offers its source, not the PDF: the equations are exact there
    preprint = p2s.Hit(source="arxiv", id="2401.00001", title="Networks")
    link = main._hit_json(preprint)["text_url"]
    assert link and "e-print" in link, f"arXiv text_url is {link}"

    # an empty question is refused rather than searched for
    try:
        main.paper_discover(main.Question(question="   "))
        raise AssertionError("an empty question was accepted")
    except Exception as exc:  # noqa: BLE001
        assert "Ask it something" in str(exc), exc

    # the page must tell a network failure apart from an empty result
    script = PAGE[PAGE.index("<script>"):]
    body = script[script.index("async function findPapers"):]
    body = body[: body.index("\nfunction paperRow")]
    assert "reachable === false" in body, \
        "the page cannot tell 'nothing matched' from 'nothing was asked'"
    assert "could not be reached" in body, "a network failure reads as no results"
    for token in ('id="findQ"', 'id="btnFind"', "function paperRow",
                  "function fetchPaper"):
        assert token in PAGE, f"{token} is missing from the search stage"

    # the result lands in the reading stage, which on a long results list is
    # off screen — so the button has to report on itself too, or pressing it
    # looks like nothing happening
    fetch = script[script.index("async function fetchPaper"):]
    fetch = fetch[: fetch.index("\nfunction showPaperPassages")]
    assert "button.disabled = true" in fetch, "the button gives no sign it was pressed"
    assert "fetchnote" in fetch, "no feedback appears beside the button"
    assert "scrollIntoView" in fetch, "the reader is not brought into view"
    for outcome in ("did not answer", "Could not read it"):
        assert outcome in fetch, f"the {outcome!r} case is not reported"
    assert fetch.count('button.textContent = "Read this one"') >= 2, \
        "the button can be left saying 'reading…' after a failure"


@check("every outcome of saving a spec reports itself")
def _():
    """Pressing "Check and save" showed the checks passing and then nothing at
    all, which reads as a save that worked.

    Two faults, both mine. The failure path was a bare `return`, and the
    success path refreshed the palette *before* writing the confirmation — so
    an exception there swallowed the only sign that anything had been saved.
    """
    script = PAGE[PAGE.index("<script>"):]
    body = script[script.index("async function checkSpec"):]
    body = body[: body.index("\n/* The point of the whole pipeline")]

    save = body[body.index("if (thenSave){"):]
    assert "catch { return; }" not in save, \
        "a failed save still returns without saying anything"
    for outcome in ("got no answer", "saving failed", "Saved as"):
        assert outcome in save, f"the {outcome!r} case is not reported"

    # the confirmation must not depend on the palette refresh succeeding
    refresh = save[save.index("loadCatalog"):]
    refresh = refresh[: refresh.index("Saved as")]
    assert "catch" in refresh, \
        "an exception refreshing the palette will swallow the confirmation"

    # and a saved domain has to lead somewhere
    assert "function buildFromDomain" in PAGE, \
        "nothing turns the saved domain into a network"
    build = script[script.index("async function buildFromDomain"):]
    build = build[: build.index("\nfunction checkTable")]
    assert "ImplicitEquilibrium" in build, "the built network does not use the layer"
    assert "showPage(\"pageDesign\")" in build, "it never reaches the canvas"


@check("the editor offers a spec that already works")
def _():
    """Writing a residual from a blank template is the hardest step in the
    pipeline, and nothing was helping with it. Editing three lines of a working
    spec is a different task from inventing the file."""
    import main

    served = main.spec_examples()["examples"]
    assert served, "no starting points are offered"
    for example in served:
        spec = example["spec"]
        assert spec.get("residual"), f"{example['key']} has no residual"
        assert not str(spec["residual"]).startswith("<"), \
            f"{example['key']} offers a placeholder as a starting point"
        assert spec.get("variants"), f"{example['key']} has no arms"
        assert example["claim"], f"{example['key']} states no claim"

    # the two shipped ones cover the two shapes a paper tends to have
    shapes = {e["shape"] for e in served}
    assert len(shapes) >= 2, f"every example is the same shape: {shapes}"

    for token in ("function loadStarters", 'id="specStarters"',
                  'id="specProblems"'):
        assert token in PAGE, f"{token} is missing"

    # a rejection has to appear where the fix happens, not only in stage three
    script = PAGE[PAGE.index("<script>"):]
    body = script[script.index("async function checkSpec"):]
    body = body[: body.index("\nfunction checkTable")]
    assert 'specProblems' in body, \
        "the reasons appear only in the stage that reports them, not where they are fixed"
    assert "whylist" in body, "the reasons are shown as one wall of prose"
    assert "scrollIntoView" in body, "the editor is not brought into view"


@check("the reader's chosen passages are what the draft is built from")
def _():
    """Choosing the passages is the judgment that carries.

    The top-ranked passage is not always the load-bearing one, and a draft made
    from the wrong theorem models the wrong thing confidently. So the choice is
    the reader's, it is carried through, and what cannot be known — the residual
    — is left blank rather than guessed.
    """
    try:
        from dnn_bench import ingest, propose  # noqa: F401
    except ImportError:
        print("        (dnn_bench absent, skipped)")
        return
    import main

    text = ("Introduction. Many models exist.\n\n"
            "Theorem 3.1. For positive rate constants the two-compartment "
            "system has a unique asymptotically stable steady state.\n\n"
            "Discussion. It fits the data.")
    paper = ingest.ingest(text, title="A minimal model")
    ranked = main._paper_json(paper)
    assert len(ranked["passages"]) >= 2

    theorem = next(i for i, p in enumerate(ranked["passages"])
                   if p["kind"] == "theorem")
    drafted = main.paper_propose(main.PaperText(text=text, title="A minimal model",
                                                only=[theorem]))
    spec = drafted["spec"]
    assert spec["title"] == "A minimal model", spec["title"]

    if not drafted["proposed"]:
        # no proposer: the skeleton must carry what is known and nothing else
        assert "Theorem 3.1" in spec["claim"], \
            "the chosen passage did not become the claim"
        assert spec.get("_passages"), "the chosen passages were not carried through"
        assert spec["residual"].startswith("<"), \
            "the residual was guessed at instead of left to a person"
        assert drafted["reason"], "it gave no reason for not proposing"

    # an empty request is refused rather than answered with an empty skeleton
    try:
        main.paper_propose(main.PaperText())
        raise AssertionError("drafting from nothing was allowed")
    except Exception as exc:  # noqa: BLE001
        assert "Read a paper first" in str(exc), exc

    # the page ties the two together
    for token in ("function draftSpec", 'id="btnDraft"', "data-pick="):
        assert token in PAGE, f"{token} is missing from the bridge"
    script = PAGE[PAGE.index("<script>"):]
    body = script[script.index("async function draftSpec"):]
    body = body[: body.index("\n}\n")]
    assert "only" in body, "the chosen passages are not sent"
    assert "_passages" in body, "the chosen text is not shown beside the editor"


@check("a paper becomes a spec becomes a layer")
def _():
    """Three stages, and only one of them involves judgment.

    Ingestion ranks passages and decides nothing. Proposal is the step to trust
    least and is allowed to refuse. The check is what makes a proposal worth
    having at all: a wrong extraction fails loudly instead of becoming a
    plausible architecture that quietly wastes a week.

    These go through the endpoint functions rather than HTTP. The routes require
    a signed-in account once a workspace has any, and creating one here would
    write into the user's own data to test something that is not about auth.
    """
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    try:
        from dnn_bench import core, domains, ingest  # noqa: F401
    except ImportError:
        print("        (dnn_bench absent, skipped)")
        return
    import json as _json

    import main

    text = ("Theorem 2.2. Let f be a strongly convex potential with modulus "
            "alpha > 0. Then f has exactly one stationary point. The dynamics "
            "are dx/dt = -grad f(x) + x0.")

    paper = ingest.ingest(text, title="note")
    ranked = main._paper_json(paper)
    assert ranked["passages"], "nothing was ranked"
    assert any(p["kind"] == "theorem" for p in ranked["passages"]), \
        "the theorem was not recognised as one"

    # with no key the proposer refuses and says why, rather than inventing one
    proposed = main.paper_propose(
        main.PaperText(text=text, title="note"))
    assert "spec" in proposed, proposed
    if not proposed["proposed"]:
        assert proposed["reason"], "it refused without saying why"

    source = _json.loads((ROOT / "specs" / "convex_ridge.json").read_text())
    known = set(core.REGISTRY)

    # a spec that does not work must fail loudly and leave nothing registered
    broken = dict(source, key="test_broken_spec", residual="x0 - nonsense(x)")
    result = main.paper_check(main.SpecBody(spec=broken))
    assert not result["ok"], "a broken residual passed"
    assert "test_broken_spec" not in core.REGISTRY, \
        "a spec that failed its checks was left offering itself as a layer"
    assert set(core.REGISTRY) == known, "the registry was left dirty"

    # one that works passes and can be placed
    good = dict(source, key="test_good_spec")
    result = main.paper_check(main.SpecBody(spec=good))
    assert result["ok"], result
    assert result["counts"]["fail"] == 0
    core.REGISTRY.pop("test_good_spec", None)

    # and the routes are guarded, which is why this test avoids them
    guarded = {r.path for r in main.app.routes
               if getattr(r, "path", "").startswith("/api/paper")}
    for route in ("/api/paper/discover", "/api/paper/fetch", "/api/paper/ingest",
                  "/api/paper/propose", "/api/paper/check", "/api/paper/save"):
        assert route in guarded, f"{route} is missing"


@check("an ablation runs the arms against this network")
def _():
    """The bench compares arms on its own fixed model. Run here, the same
    comparison answers whether the structure helps in the architecture you
    actually have — with the parameter counts matched by construction, since
    every arm is the same design with one setting swapped."""
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    try:
        from dnn_bench import core, domains  # noqa: F401
    except ImportError:
        print("        (dnn_bench absent, skipped)")
        return
    import agents

    assert any(a["id"] == "ablation" for a in agents.CATALOG), \
        "the study is not offered"

    graph = build(
        [("i", "Input", {"shape": [4]}),
         ("c", "ImplicitEquilibrium", {"units": 8, "domain": "convex",
                                       "variant": "convex", "seed": 0}),
         ("o", "Output", {"task": "classification"})],
        [("i", "c", 0), ("c", "o", 0)], "Ablate")

    trials = agents.ablation_trials(graph, {"seed": 0})
    assert len(trials) == 3, f"{len(trials)} arms"
    assert sum(1 for t in trials if t["guarantee"]) == 1, \
        "there should be exactly one arm the theorem covers"
    assert len({t["learnables"] for t in trials}) == 1, \
        "the arms differ in size, so they are not a controlled comparison"
    for trial in trials:
        assert trial["note"], "an arm does not say which condition it satisfies"

    # a design with no implicit layer has nothing to ablate, and says so by
    # producing no trials rather than by failing
    plain = build([("i", "Input", {"shape": [4]}),
                   ("l", "Linear", {"units": 2}),
                   ("o", "Output", {"task": "classification"})],
                  [("i", "l", 0), ("l", "o", 0)], "Plain")
    assert agents.ablation_trials(plain, {}) == []


@check("the ablation verdict does not overclaim")
def _():
    """Beating one control is not beating the condition."""
    import agents

    def verdict(scores):
        agent = agents.Agent(home=None, id="t", kind="ablation", design="d",
                             trials=[
            {"label": "d/covered", "score": scores[0], "guarantee": True},
            {"label": "d/one", "score": scores[1], "guarantee": False},
            {"label": "d/two", "score": scores[2], "guarantee": False}])
        return agent.snapshot()["verdict"]

    assert "beat every control" in verdict([0.1, 0.2, 0.3])
    assert "beat no control" in verdict([0.9, 0.2, 0.3])

    partial = verdict([0.25, 0.2, 0.3])
    assert "1 of 2" in partial, partial
    assert "not beating the condition" in partial, \
        "a partial result is being reported as a win"
    assert "d/one" in partial, "the control it failed against is not named"

    # other study kinds get no verdict at all
    other = agents.Agent(home=None, id="t", kind="sweep", design="d", trials=[])
    assert other.snapshot()["verdict"] is None


@check("a model built in double precision trains")
def _():
    """A layer whose solver needs float64 makes the whole model double, and
    every batch has to follow — including the validation batches, which is
    where this first failed."""
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import torch

    import train as T

    class Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(4, 4, dtype=torch.float64)

        def forward(self, x):
            return self.a(x)

    model = Mixed()
    assert T.needs_double(model, {}), "a double parameter was not noticed"
    assert T.needs_double(torch.nn.Linear(2, 2), {"precision": "float64"}), \
        "an explicit request was ignored"
    assert not T.needs_double(torch.nn.Linear(2, 2), {}), \
        "an ordinary model was forced to double"

    # every batch path converts, not just the training one
    source = (ROOT / "train.py").read_text()
    unconverted = [line.strip() for line in source.split("\n")
                   if ".to(device)" in line and "xs" in line
                   and "_match" not in line]
    assert not unconverted, f"these batches never convert: {unconverted}"


@check("an implicit layer can check its own mathematics")
def _():
    """Test layer asks whether the canvas agrees with torch. This asks whether
    the layer's mathematics agrees with itself — a Jacobian with a sign error
    still converges, so otherwise the only symptom is silently wrong gradients.
    """
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    try:
        from dnn_bench import domains, validate  # noqa: F401
    except ImportError:
        print("        (dnn_bench absent, skipped)")
        return

    result = validate.validate_domain("crn", seed=0, quick=True)
    assert result["ok"], result["counts"]
    assert result["counts"]["fail"] == 0, result["counts"]
    names = {c["name"] for c in result["checks"]}
    for wanted in ("jacobian", "matched parameters", "certificates"):
        assert wanted in names, f"{wanted} is not among the checks: {sorted(names)}"

    # the button exists and reads its settings the way the form does
    assert 'id="btnCheckDomain"' in PAGE, "there is no way to run the checks"
    assert "function checkDomain" in PAGE
    script = PAGE[PAGE.index("<script>"):]
    body = script[script.index("function checkDomain"):]
    body = body[: body.index("\nasync function testLayer")]
    assert "state.specs" in body and "q.default" in body, \
        "the check does not fall back to defaults, so an untouched node sends undefined"
    assert "resolvedParams" not in PAGE, \
        "a helper that does not exist is being called"


@check("a control arm is never a silent choice")
def _():
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    try:
        from dnn_bench import core, domains  # noqa: F401
    except ImportError:
        print("        (dnn_bench absent, skipped)")
        return
    import mathbook
    import needs

    def design(arm):
        payload = build(
            [("i", "Input", {"shape": [10]}),
             ("c", "ImplicitEquilibrium", {"units": 6, "domain": "contraction",
                                           "variant": arm, "seed": 0}),
             ("o", "Output", {"task": "classification"})],
            [("i", "c", 0), ("c", "o", 0)], "Imp")
        return analyzed(payload)

    g, rep = design("unconstrained")
    notes = " ".join(needs.requirements(g, rep)["notes"])
    assert "control arm" in notes, "a broken-condition arm passes without a word"

    g, rep = design("monotone")
    notes = " ".join(needs.requirements(g, rep)["notes"])
    assert "control arm" not in notes, "the covered arm was called a control"

    # and the maths panel says which case you are in either way
    for arm, expected in (("monotone", "yes"), ("unconstrained", "no")):
        entry = mathbook.explain("ImplicitEquilibrium",
                                 {"units": 6, "domain": "contraction",
                                  "variant": arm, "seed": 0}, [[10]], [6])
        says = dict(entry["arithmetic"])["theorem applies"]
        assert says.startswith(expected), f"{arm} reported {says!r}"


@check("the equilibrium layer is a layer like any other")
def _():
    """A deficiency-zero implicit layer, vendored from separate work.

    What it needs from the designer is what every layer needs: a shape it can
    infer, a parameter count that matches what torch builds, and code that
    runs. The guarantee it carries — a fixed point for any positive rates,
    with no projection — is a property of the reaction graph, so the test
    checks that the graph really has it.
    """
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    try:
        import crn_deq
    except ImportError:
        print("        (crn_deq absent, skipped)")
        return
    import train as T

    assert "EquilibriumCRN" in layers.REGISTRY, "the block did not install"

    payload = build(
        [("i", "Input", {"shape": [16]}),
         ("c", "EquilibriumCRN", {"units": 8, "species": 6, "classes": 2,
                                  "extra_edges": 0, "deficiency": "0",
                                  "seed": 0, "tol": 1e-10, "max_iter": 200}),
         ("l", "Linear", {"units": 3}),
         ("o", "Output", {"task": "classification"})],
        [("i", "c", 0), ("c", "l", 0), ("l", "o", 0)], "CRN")
    g, rep = analyzed(payload)
    assert rep["ok"], rep["errors"]

    source = codegen.to_pytorch(g, rep)
    assert "from crn_deq import" in source, "the generated file cannot find the layer"
    model = T.build_model(source, codegen.model_class_name(g)).double()
    real = sum(p.numel() for p in model.parameters())
    assert real == rep["total_learnables"], \
        f"canvas {rep['total_learnables']}, torch {real}"

    import torch

    with torch.no_grad():
        out = model(torch.randn(2, 16, dtype=torch.float64))
    assert tuple(out.shape) == (2, 3), out.shape

    # the guarantee rests on these two facts about the graph
    net = crn_deq.generate_network(6, [3, 3], extra_edges=0,
                                   target_deficiency=0, seed=0)
    assert net.deficiency() == 0, "the network is not deficiency zero"
    assert net.is_weakly_reversible(), "the network is not weakly reversible"

    # the maths panel's deficiency line must be arithmetic that holds
    import re

    import mathbook

    for classes, seed in ((2, 0), (2, 3), (3, 1)):
        entry = mathbook.explain(
            "EquilibriumCRN",
            {"units": 8, "species": 6, "classes": classes, "extra_edges": 0,
             "deficiency": "0", "seed": seed}, [[16]], [8])
        line = dict(entry["arithmetic"]).get("deficiency", "")
        assert line, "the deficiency is not shown"
        numbers = [int(x) for x in re.findall(r"-?\d+", line)]
        m, l, rank, delta = numbers[-4:]
        assert m - l - rank == delta, f"printed arithmetic does not hold: {line}"
        assert delta == 0, f"a deficiency-zero graph reported {delta}"

    # and the design says out loud what precision it needs
    import needs

    notes = " ".join(needs.requirements(g, rep)["notes"])
    assert "float64" in notes, "nothing warns that this layer needs double precision"


@check("the walkthrough narrates a real pass, not an idea of one")
def _():
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import walkthrough as walk

    graph = build(
        [("i", "Input", {"shape": [3, 16, 16]}),
         ("c", "Conv2d", {"filters": 8, "kernel": 3, "padding": "same"}),
         ("a", "Activation", {"kind": "relu"}),
         ("g", "GlobalAvgPool", {}),
         ("l", "Linear", {"units": 4}),
         ("o", "Output", {"task": "classification"})],
        [("i", "c", 0), ("c", "a", 0), ("a", "g", 0), ("g", "l", 0), ("l", "o", 0)],
        "Story")

    story = walk.walkthrough(graph, batch=1)
    assert len(story["steps"]) == 6

    # the ReLU's claim is measured from the tensor, not asserted about relus
    relu = next(s for s in story["steps"] if s["type"] == "Activation")
    assert relu["values"] and relu["values"]["zeros"] > 0, relu["values"]
    assert "zero" in relu["note"], relu["note"]
    assert relu["values"]["min"] == 0.0, "a relu should leave nothing negative"

    # each step carries the mathematics for that layer
    conv = next(s for s in story["steps"] if s["type"] == "Conv2d")
    assert conv["equation"], "no equation for the convolution"
    assert conv["parameters"] > 0

    # a layer with no module says so rather than showing a blank
    quiet = [s for s in story["steps"] if s["unwatched"]]
    assert quiet, "nothing was reported as unwatched"
    assert all(s["values"] is None for s in quiet)

    closing = story["closing"]
    assert closing["task"] == "classification"
    assert len(closing["picks"]) == 4, closing
    assert abs(sum(p["p"] for p in closing["picks"]) - 1.0) < 1e-3, \
        "the probabilities should account for the whole distribution"


@check("the walkthrough reads a sequence model as next-token")
def _():
    """The closing step follows the task, not the architecture, so it suits
    whatever is on the canvas."""
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import walkthrough as walk

    sequence = build(
        [("i", "Input", {"shape": [8], "dtype": "long"}),
         ("e", "Embedding", {"vocab": 50, "dim": 16}),
         ("n", "RMSNorm", {}),
         ("h", "Linear", {"units": 50}),
         ("o", "Output", {"task": "classification"})],
        [("i", "e", 0), ("e", "n", 0), ("n", "h", 0), ("h", "o", 0)], "Seq")
    story = walk.walkthrough(sequence, batch=1)
    closing = story["closing"]
    assert closing["sequence"], "a per-position output was not recognised"
    assert "next token" in closing["note"], closing["note"]

    # token indices are described as indices, never averaged
    first = story["steps"][0]
    assert first["values"]["indices"], "token ids were treated as quantities"
    assert "vocabulary" in first["note"], first["note"]

    regression = build(
        [("i", "Input", {"shape": [4]}),
         ("l", "Linear", {"units": 1}),
         ("o", "Output", {"task": "regression"})],
        [("i", "l", 0), ("l", "o", 0)], "Reg")
    closing = walk.walkthrough(regression, batch=1)["closing"]
    assert closing["task"] == "regression"
    assert "not a score" in closing["note"], closing["note"]


@check("precision can be measured on trained weights")
def _():
    """Drift on random weights describes the shapes, not the model — a trained
    network and a random one of the same size are not equally sensitive to
    losing precision, which is measurable and was measured: 0.112 against 0.131
    at four bits on the same design."""
    import main

    assert "weights" in main.QuantPayload.model_fields, \
        "the endpoint cannot be pointed at a checkpoint"

    # weights that belong to another design must be refused, not half-loaded
    import inspect

    import quantize

    body = inspect.getsource(quantize.quantize_report)
    assert "strict=False" in body, "a partial match would raise instead of loading"
    assert "every tensor was" in body, \
        "weights from a different network would load nothing and say nothing"

    # a name that is not there is refused before anything is built
    try:
        main.quantize_design(main.QuantPayload(
            graph={"nodes": [], "edges": []}, weights="not-a-real-file.pt"))
        raise AssertionError("a missing checkpoint was accepted")
    except Exception as exc:  # noqa: BLE001
        assert "No saved weights" in str(exc), exc

    # and the panel says which weights a number came from
    assert 'id="quantWeights"' in PAGE, "there is no way to choose the weights"
    assert "measured on untrained weights" in PAGE, \
        "an untrained measurement does not say that it is one"


@check("quantization is measured, not asserted")
def _():
    """A 4-bit release is the same architecture and the same weights read at
    fewer bits. What it saves is arithmetic; what it costs has to be run.

    The per-layer pass is the useful half: it finds the layer that cannot take
    the reduction, which is rarely the biggest one.
    """
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import quantize

    payload = build(
        [("i", "Input", {"shape": [64]}),
         ("a", "Linear", {"units": 128}),
         ("r", "Activation", {"kind": "relu"}),
         ("h", "Linear", {"units": 8}),
         ("o", "Output", {"task": "classification"})],
        [("i", "a", 0), ("a", "r", 0), ("r", "h", 0), ("h", "o", 0)], "Quant")

    result = quantize.quantize_report(payload, batch=16)
    assert result["ok"]
    working = [s for s in result["schemes"] if s.get("ok")]
    assert len(working) >= 3, [s.get("why") for s in result["schemes"]]

    by_key = {s["key"]: s for s in working}

    # Every scheme should run everywhere: measuring them all the same way is
    # what makes that true, and what makes the rows comparable. torch's dynamic
    # int8 needs a quantized backend, and which exists differs by machine — it
    # is absent on Apple silicon, where this once reported a scheme unavailable
    # that is perfectly measurable.
    missing = [s["key"] for s in result["schemes"] if not s.get("ok")]
    assert not missing, (
        f"these schemes did not run: {missing}. A measurement that depends on "
        f"the machine's kernels is not the same measurement everywhere.")

    # fewer bits must cost more, or the measurement is not measuring
    assert by_key["float16"]["drift"] < by_key["int4"]["drift"], \
        f"4-bit drifted less than 16-bit: {by_key}"
    assert by_key["float16"]["saving"] == 0.5, by_key["float16"]["saving"]
    assert by_key["int4"]["saving"] > by_key["int8"]["saving"] > 0.5

    # a scheme reporting no drift has not been applied at all
    for key in ("int8", "int4"):
        assert by_key[key]["drift"] > 0, f"{key} changed nothing at all"

    # the per-layer pass names layers that exist and is ordered worst first
    ids = {n["id"] for n in payload["nodes"]}
    assert result["layers"], "no per-layer sensitivity was measured"
    for row in result["layers"]:
        assert row["id"] in ids, f"{row['id']} is not a node in the design"
        assert row["parameters"] > 0
    drifts = [row["drift"] for row in result["layers"]]
    assert drifts == sorted(drifts, reverse=True), "not ordered worst first"

    # quantizing one layer must hurt less than quantizing all of them
    assert max(drifts) <= by_key["int4"]["drift"] * 1.5, \
        "a single layer drifted more than the whole model, which cannot be"


@check("a forward pass is traced layer by layer")
def _():
    """A workflow shows tasks going green because they run one at a time. A
    forward pass has the same shape, so this measures it rather than animating
    a guess."""
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import tracer

    graph = build(
        [("i", "Input", {"shape": [1, 28, 28]}),
         ("c", "Conv2d", {"filters": 16, "kernel": 3, "padding": "same"}),
         ("b", "BatchNorm2d", {}),
         ("a", "Activation", {"kind": "relu"}),
         ("p", "MaxPool2d", {"kernel": 2}),
         ("f", "Flatten", {}),
         ("l", "Linear", {"units": 10}),
         ("o", "Output", {"task": "classification"})],
        [("i", "c", 0), ("c", "b", 0), ("b", "a", 0), ("a", "p", 0),
         ("p", "f", 0), ("f", "l", 0), ("l", "o", 0)], "Trace")

    result = tracer.run_trace(graph, batch=4)
    assert result["ok"], result.get("error")
    assert result["warmed"], "timings must come from a warmed pass"

    rows = result["layers"]
    assert len(rows) == 8, f"{len(rows)} rows for 8 layers"
    assert [r["seq"] for r in rows] == list(range(1, 9)), "rows are out of order"

    measured = [r for r in rows if r["measured"]]
    assert len(measured) >= 5, f"only {len(measured)} layers were timed"
    for row in measured:
        assert row["ms"] >= 0, row
        assert row["shape"] and row["shape"][0] == 4, \
            f"{row['name']} reported {row['shape']} for a batch of 4"
        assert row["bytes"] and row["bytes"] > 0, row

    # a layer with no module of its own is reported, not dropped
    terminals = [r for r in rows if r["type"] in ("Input", "Output")]
    assert terminals and all(not r["measured"] for r in terminals)
    assert all(r["ms"] is None for r in terminals), \
        "an untimed layer should say so rather than claim zero"

    # the times are steady-state, not first-call kernel setup
    slowest = max(measured, key=lambda r: r["ms"])
    assert slowest["ms"] < 200, \
        f"{slowest['name']} took {slowest['ms']}ms, which looks unwarmed"

    assert result["outputs"][0]["shape"] == [4, 10], result["outputs"]


@check("the canvas shows a run as it progresses")
def _():
    for token in ("function renderRunPanel", "function animateTrace",
                  "function traceStateOf", "rantick", "lanebar",
                  'data-side="run"', 'id="runBody"'):
        assert token in PAGE, f"{token} is missing from the run view"

    script = PAGE[PAGE.index("<script>"):]
    body = script[script.index("function traceStateOf"):]
    body = body[: body.index("\n}\n")]
    assert "state.traceAt === null" in body, \
        "there is no settled state, so the run never finishes visibly"


@check("a layer can be run on its own and checked against the canvas")
def _():
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import main

    graph = {"name": "T", "nodes": [
        {"id": "i", "type": "Input", "params": {"shape": [3, 32, 32]}},
        {"id": "c", "type": "Conv2d",
         "params": {"filters": 64, "kernel": 3, "stride": 2, "padding": 1}},
        {"id": "o", "type": "Output", "params": {}}],
        "edges": [{"id": "e1", "source": "i", "target": "c"},
                  {"id": "e2", "source": "c", "target": "o"}]}
    body = type("B", (), {"graph": graph, "node": "c"})()
    result = main.test_layer(body)
    assert result["ok"], result
    assert result["matches"], f"canvas said {result['predicted']}, torch gave {result['actual']}"
    assert result["actual"] == [64, 16, 16], result["actual"]
    assert result["learnables"] == 1792, result["learnables"]


@check("freezing a layer is honoured by the generated code and the count")
def _():
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import train as T

    payload = build(
        [("i", "Input", {"shape": [3, 32, 32]}),
         ("c", "Conv2d", {"filters": 32, "kernel": 3, "padding": "same",
                          "_frozen": True}),
         ("g", "GlobalAvgPool", {}), ("l", "Linear", {"units": 10}),
         ("o", "Output", {})],
        [("i", "c", 0), ("c", "g", 0), ("g", "l", 0), ("l", "o", 0)], "Frz")
    g, rep = analyzed(payload)
    assert rep["ok"], rep["errors"]
    assert rep["nodes"]["c"]["frozen"], "the flag did not reach the report"
    assert rep["nodes"]["c"]["learnables"] == 0, "frozen weights should not count as trainable"

    src = codegen.to_pytorch(g, rep)
    assert "requires_grad_(False)" in src, "codegen did not freeze it"
    model = T.build_model(src, codegen.model_class_name(g))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == rep["total_learnables"], \
        f"canvas says {rep['total_learnables']} trainable, torch says {trainable}"


@check("core layers link to their PyTorch reference")
def _():
    entries = {e["name"]: e for e in layers.catalog()}
    assert entries["Conv2d"]["docs"].endswith("torch.nn.Conv2d.html"), entries["Conv2d"]
    assert entries["LSTM"]["docs"], "LSTM has no reference link"
    assert entries["ResidualBlock"]["docs"] is None, \
        "a block has no PyTorch page to link to"


@check("the plus buttons open an anchored picker, not a centred list")
def _():
    for token in ('id="quickAdd"', 'id="qaSearch"', 'id="qaBody"', "class=\"qgrid\"",
                  "function openQuickAdd", "function renderQuickAdd",
                  "function chooseQuickAdd", "const GLYPHS", "QUICK_ADD"):
        assert token in PAGE, f"{token} is missing from the picker"
    # it is positioned from the click, not fixed in the middle of the window
    assert "clientX" in PAGE[PAGE.index("function openQuickAdd"):
                             PAGE.index("function openQuickAdd") + 1200]
    # every category the palette can show needs a glyph, or tiles come out blank
    import re

    glyphs = set(re.findall(r'\n  "?([A-Za-z ]+)"?:\s*"M',
                            PAGE[PAGE.index("const GLYPHS"):
                                 PAGE.index("const QUICK_ADD")]))
    categories = {spec.category for spec in layers.REGISTRY.values()}
    missing = sorted(c for c in categories if c not in glyphs)
    assert not missing, f"no glyph for: {missing}"


@check("a latent-attention block imports and is counted honestly")
def _():
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import importer
    import main

    source = """
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Latent(nn.Module):
    def __init__(self, dim=256, heads=4, latent=64):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.hd = dim // heads
        self.n = RMSNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv_down = nn.Linear(dim, latent, bias=False)
        self.kv_up = nn.Linear(latent, 2 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        h = self.n(x)
        q = self.q(h).view(B, T, self.heads, self.hd).transpose(1, 2)
        k, v = self.kv_up(self.kv_down(h)).split(self.dim, dim=2)
        k = k.view(B, T, self.heads, self.hd).transpose(1, 2)
        v = v.view(B, T, self.heads, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, C))
"""
    payload = importer.from_source(source + "\nmodel = Latent()\n", [32, 256])
    payload.pop("_entry", None)
    notes = payload.pop("_notes", [])
    expected = payload.get("_expected_parameters")
    result = main._finish_import(payload)

    assert result["ok"], result["problems"]
    assert not notes, f"low-rank attention should import cleanly: {notes}"
    kinds = [n["type"] for n in result["graph"]["nodes"]]
    for wanted in ("Split", "Attention", "Elementwise", "Reduce"):
        assert wanted in kinds, f"{wanted} missing from {set(kinds)}"

    # the norm's learned scale belongs to no layer, and that is said out loud
    assert expected and expected != result["learnables"]
    assert result["warnings"], "an unaccounted parameter passed silently"
    assert "difference of" in result["warnings"][0]


@check("runtime routing is refused a confident diagram")
def _():
    """A mixture of experts cannot be drawn as a fixed graph.

    fx unrolls the loop over experts, so all of them appear as though all of
    them run — and the parameter count comes out far above the real one. The
    diagram is not wrong about the code; it is wrong about the model, which is
    worse. So it says so.
    """
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import importer
    import main

    source = """
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(x)


class MoE(nn.Module):
    def __init__(self, dim=64, experts=4, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim, experts, bias=False)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(experts)])

    def forward(self, x):
        scores = F.softmax(self.gate(x), dim=-1)
        weights, picks = torch.topk(scores, self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            for i, expert in enumerate(self.experts):
                mask = (picks[..., slot] == i).unsqueeze(-1)
                out = out + mask * expert(x) * weights[..., slot:slot + 1]
        return out
"""
    payload = importer.from_source(source + "\nmodel = MoE()\n", [16, 64])
    payload.pop("_entry", None)
    assert payload.get("_routing"), "topk routing was not noticed"
    result = main._finish_import(payload)
    assert result["warnings"], "the diagram claimed to represent the model"
    assert "run time" in result["warnings"][0]
    assert "experts" in result["warnings"][0]


@check("a transformer block imports with nothing left opaque")
def _():
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import importer

    source = """
import torch.nn as nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, n_embd=768, n_head=12):
        super().__init__()
        self.n_head, self.n_embd = n_head, n_embd
        self.ln_1 = nn.LayerNorm(n_embd)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        h = self.ln_1(x)
        q, k, v = self.c_attn(h).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.c_proj(y)
        return x + self.c_proj2(self.gelu(self.c_fc(self.ln_2(x))))
"""
    payload = importer.from_source(source, [64, 768])
    notes = payload.pop("_notes", [])
    payload.pop("_entry", None)
    kinds = [n["type"] for n in payload["nodes"]]

    assert "Custom" not in kinds, f"opaque nodes remain: {notes}"
    for expected in ("Split", "Transpose", "Attention", "LayerNorm", "Add"):
        assert expected in kinds, f"{expected} did not come through: {kinds}"
    # the shape bookkeeping the author wrote must not appear as layers
    assert "size" not in json.dumps(notes)

    g, rep = analyzed(payload)
    assert rep["ok"], rep["errors"]

    import train as T
    model = T.build_model(codegen.to_pytorch(g, rep), codegen.model_class_name(g))
    real = sum(p.numel() for p in model.parameters())
    assert real == rep["total_learnables"], \
        f"canvas {rep['total_learnables']:,}, torch {real:,}"


@check("a split's branches keep their own piece")
def _():
    """q, k and v came out as the same tensor.

    fx models `q, k, v = t.split(n, dim=2)` as one split followed by three
    getitems. Emitting the Split when the split appeared gave every branch
    piece 0 — the shapes were right, the parameter count was right, and the
    network computed something else entirely. Silent wrong answers are the
    worst kind, so this checks the indices.
    """
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import importer

    payload = importer.from_source("""
import torch.nn as nn


class Three(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 48)

    def forward(self, x):
        a, b, c = self.proj(x).split(16, dim=2)
        return a + b * c
""", [8, 16])
    payload.pop("_notes", None)
    payload.pop("_entry", None)
    takes = sorted(n["params"]["take"] for n in payload["nodes"]
                   if n["type"] == "Split")
    assert takes == [0, 1, 2], f"the branches take pieces {takes}, not one each"


@check("an imported model computes what the original did")
def _():
    """Shapes matching is not the same as behaviour matching."""
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import torch

    import importer
    import train as T

    source = """
import torch.nn as nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(32)
        self.qkv = nn.Linear(32, 96)
        self.out = nn.Linear(32, 32)

    def forward(self, x):
        B, T, C = x.size()
        h = self.qkv(self.ln(x))
        q, k, v = h.split(32, dim=2)
        q = q.view(B, T, 4, 8).transpose(1, 2)
        k = k.view(B, T, 4, 8).transpose(1, 2)
        v = v.view(B, T, 4, 8).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(B, T, C))
"""
    namespace = {}
    exec(compile(source, "<t>", "exec"), namespace)  # noqa: S102
    original = namespace["Net"]().eval()

    payload = importer.from_source(source, [8, 32])
    payload.pop("_notes", None)
    payload.pop("_entry", None)
    g, rep = analyzed(payload)
    assert rep["ok"], rep["errors"]

    rebuilt = T.build_model(codegen.to_pytorch(g, rep),
                            codegen.model_class_name(g)).eval()
    # Pair the n-th tensor of a given shape with the n-th of that shape, rather
    # than the first that fits: a LayerNorm's weight and bias are the same
    # shape, so shape alone would swap them.
    def by_shape(state):
        buckets = {}
        for key, tensor in state.items():
            buckets.setdefault(tuple(tensor.shape), []).append((key, tensor))
        return buckets

    source_buckets = by_shape(original.state_dict())
    mapped = {}
    for shape, entries in by_shape(rebuilt.state_dict()).items():
        origin = source_buckets.get(shape, [])
        assert len(origin) == len(entries), \
            f"{len(entries)} tensors of shape {shape} but {len(origin)} to fill them"
        for (key, _), (_, tensor) in zip(entries, origin):
            mapped[key] = tensor
    rebuilt.load_state_dict(mapped, strict=False)

    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        gap = (original(x) - rebuilt(x)).abs().max().item()
    assert gap < 1e-5, f"the reconstruction computes something else: {gap}"


@check("the rearrangement layers agree with torch")
def _():
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import torch

    import layers as L

    assert L.REGISTRY["Transpose"].infer({"dim_a": 1, "dim_b": 2},
                                         [[64, 12, 64]]) == [12, 64, 64]
    x = torch.zeros(2, 64, 12, 64)
    assert list(x.transpose(1, 2).shape)[1:] == [12, 64, 64], "the canvas disagrees with torch"

    assert L.REGISTRY["Split"].infer({"pieces": 3, "axis": 1, "take": 0},
                                     [[64, 2304]]) == [64, 768]
    assert list(torch.zeros(2, 64, 2304).chunk(3, dim=2)[0].shape)[1:] == [64, 768]

    assert L.REGISTRY["Attention"].infer({}, [[12, 64, 64]] * 3) == [12, 64, 64]
    q = torch.zeros(2, 12, 64, 64)
    got = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    assert list(got.shape)[1:] == [12, 64, 64]

    for params, shapes, why in [
        ({"pieces": 5, "axis": 0}, [[7, 4]], "an uneven split"),
        ({}, [[12, 64, 64], [12, 64, 32], [12, 64, 64]], "mismatched query and key widths"),
    ]:
        name = "Split" if "pieces" in params else "Attention"
        try:
            L.REGISTRY[name].infer(params, shapes)
            raise AssertionError(f"{why} should be refused")
        except L.ShapeError:
            pass


@check("pasted code becomes a diagram")
def _():
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import importer

    source = """
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.h = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.avg_pool2d(F.relu(self.c2(x)), 2)
        return self.h(x.flatten(1))
"""
    payload = importer.from_source(source, [3, 32, 32])
    notes = payload.pop("_notes", [])
    assert payload.pop("_entry") == "Net"
    assert not notes, f"nothing should have come in as a stub: {notes}"
    g, rep = analyzed(payload)
    assert rep["ok"], rep["errors"]
    kinds = [n["type"] for n in payload["nodes"]]
    assert kinds == ["Input", "Conv2d", "Activation", "MaxPool2d", "Conv2d",
                     "Activation", "AvgPool2d", "Flatten", "Linear", "Output"], kinds

    # a module assigned to a name works as well as a class
    seq = importer.from_source(
        "model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))", [8])
    seq.pop("_notes", None)
    assert seq.pop("_entry") == "model"
    assert analyzed(seq)[1]["ok"]


@check("a refusal names the obstacle it actually hit")
def _():
    """Three different problems, and the message should say which.

    A variadic signature is the one that stops the Hugging Face attention and
    mixture-of-experts modules, and it has no way round — wrapping them does not
    help, because the same pattern appears inside.
    """
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import importer

    try:
        importer.from_source("""
import torch.nn as nn


class Variadic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x, **kwargs):
        for key in kwargs:
            x = x + kwargs[key]
        return self.fc(x)
""", [4, 8])
        raise AssertionError("a variadic forward should not have traced")
    except importer.ImportError_ as exc:
        assert "variadic" in str(exc), str(exc)
        assert "wrapping them does not help" in str(exc), \
            "the message should not suggest a workaround that fails"


@check("pasted code that cannot be traced says why")
def _():
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import importer

    for source, expected in [
        ("x = 1", "defines no nn.Module"),
        ("class Oops(nn.Module)\n    pass", "did not run"),
        ("class Cfg(nn.Module):\n"
         "    def __init__(self, w):\n"
         "        super().__init__()\n"
         "        self.fc = nn.Linear(8, w)\n"
         "    def forward(self, x): return self.fc(x)", "no arguments"),
    ]:
        try:
            importer.from_source(source, [8])
            raise AssertionError(f"{source[:24]!r} should have been refused")
        except importer.ImportError_ as exc:
            assert expected in str(exc), f"got {exc}, wanted {expected!r}"


@check("the environment check reports on what is actually installed")
def _():
    import subprocess

    result = subprocess.run([sys.executable, str(ROOT / "doctor.py")],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-400:]
    report = result.stdout
    for expected in ("python", "fastapi", "torch", "storage", "free space"):
        assert expected in report, f"{expected} is not reported"
    # it must state a verdict either way
    assert "Ready." in report or "Install first" in report, report[-200:]
    # and say which interpreter it checked, since that is the usual confusion
    assert sys.executable in report, \
        "the report does not say which python it looked at"

    # and it must not claim something is present when it is not
    import importlib

    for module in ("transformers", "onnx"):
        try:
            importlib.import_module(module)
            installed = True
        except Exception:  # noqa: BLE001
            installed = False
        line = [l for l in report.splitlines() if l.strip().startswith(module)]
        assert line, f"{module} is not mentioned"
        says_present = "yes" in line[0]
        assert says_present == installed, \
            f"the report says {module} is {'present' if says_present else 'absent'}"


@check("every module parses under Python 3.11")
def _():
    """Guards a difference between 3.11 and 3.12 that a 3.12 machine cannot see.

    Backslashes inside an f-string expression only became legal in 3.12
    (PEP 701). Written on 3.12 they look fine and import fine; on 3.11 the
    module will not even parse, which takes the whole server down at startup.
    Since this is checked with the syntax tree rather than by running another
    interpreter, it catches the mistake whichever version is running the tests.
    """
    import ast

    offenders = []
    for path in sorted(ROOT.rglob("*.py")):
        # Only our own code. data/ holds repositories the scouts downloaded,
        # and whether somebody else's file parses under 3.11 is not this
        # project's business — nor is it a reason to fail the suite.
        parts = set(path.parts)
        if "__pycache__" in parts or "data" in parts or ".venv" in parts:
            continue
        source = path.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            offenders.append(f"{path.name}:{exc.lineno} does not parse: {exc.msg}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.JoinedStr):
                continue
            for part in node.values:
                if not isinstance(part, ast.FormattedValue):
                    continue
                segment = ast.get_source_segment(source, part.value) or ""
                if "\\" in segment:
                    offenders.append(
                        f"{path.name}:{part.lineno} f-string expression contains a "
                        f"backslash, which Python 3.11 rejects: {segment[:60]}")
    assert not offenders, "\n        " + "\n        ".join(offenders)


@check("the mathematics is instantiated with the node's own numbers")
def _():
    import mathbook

    conv = mathbook.explain(
        "Conv2d",
        {"filters": 64, "kernel": 3, "stride": 2, "padding": 1,
         "dilation": 1, "groups": 1, "bias": True},
        [[3, 32, 32]], [64, 16, 16])
    text = json.dumps(conv)
    # the derivation must show the substitution, not just the answer
    assert "3\u00d79\u00d764 + 64 = 1,792" in text.replace("\\u00d7", "\u00d7") \
        or "3×9×64 + 64 = 1,792" in text, conv["arithmetic"]
    assert any("16" in v for _, v in conv["arithmetic"]), "no output size worked through"
    assert conv["family"] == "conv"
    assert conv["freedom"], "no discussion of what can be varied"

    # the same layer with different settings must give different arithmetic
    wider = mathbook.explain(
        "Conv2d",
        {"filters": 128, "kernel": 5, "stride": 1, "padding": 2,
         "dilation": 1, "groups": 1, "bias": True},
        [[3, 32, 32]], [128, 32, 32])
    assert wider["arithmetic"] != conv["arithmetic"], \
        "the explanation is generic, not this node's"

    # a grouped convolution divides the parameters
    grouped = mathbook.explain(
        "Conv2d",
        {"filters": 64, "kernel": 3, "stride": 1, "padding": 1,
         "dilation": 1, "groups": 4, "bias": False},
        [[64, 32, 32]], [64, 32, 32])
    assert "16" in json.dumps(grouped["arithmetic"]), \
        "groups should reduce the channels each filter sees"

    linear = mathbook.explain("Linear", {"units": 10, "bias": True},
                              [[64]], [10])
    assert "64\u00d710 + 10 = 650" in json.dumps(linear).replace("\\u00d7", "\u00d7") \
        or "64×10 + 10 = 650" in json.dumps(linear), linear["arithmetic"]


@check("a layer with no write-up says so rather than inventing one")
def _():
    import mathbook

    entry = mathbook.explain("SomethingUnwritten", {}, [[4]], [4])
    assert entry.get("missing"), "it should admit the gap"
    assert not entry["equation"], "it invented an equation"
    assert not entry["arithmetic"], "it invented a derivation"


@check("the maths panel draws every family it can return")
def _():
    import mathbook

    families = set()
    for name in mathbook.covered():
        try:
            entry = mathbook.explain(name, {"units": 4, "filters": 4, "kernel": 3,
                                            "hidden": 8, "vocab": 10, "dim": 4,
                                            "heads": 2, "rate": 0.5, "sheet": "s"},
                                     [[8, 8, 8]], [8])
        except Exception:  # noqa: BLE001
            continue
        if not entry.get("missing"):
            families.add(entry["family"])
    for family in sorted(families):
        assert f'kind === "{family}"' in PAGE, \
            f"the panel has no diagram for the {family} family"
    assert 'data-side="math"' in PAGE and 'id="mathBody"' in PAGE
    assert "function renderMathPanel" in PAGE
    # reachable from the sidebar, and grouped with the design rather than status
    rail = PAGE[PAGE.index('<nav id="rail">'):PAGE.index("</nav>")]
    assert 'data-side="math"' in rail, "Maths has no sidebar entry"
    definitions = rail[rail.index("Definitions"):rail.index("Executions")]
    assert 'data-side="math"' in definitions, \
        "Maths belongs with the design, not with status"


@check("workspace state cannot ride along in a release")
def _():
    """The seeding marker shipped inside a release and suppressed delivery.

    It lived at the top of the project, beside the source, so packaging swept it
    in; unpacking then told the receiving machine that examples it had never
    seen were already delivered.

    What this checks is that such files are excluded and that the marker sits
    with the designs — not that they are absent from disk. On a machine that is
    actually running the app they are supposed to be there, and an earlier
    version of this check failed on exactly that.
    """
    import subprocess

    ignored = (ROOT / ".gitignore").read_text()
    for name in (".seeded", "prefs.json", "data/", "runs/", "studies/"):
        assert name in ignored, f"{name} is not ignored, so it can ship"

    # the marker belongs beside the designs it describes, not at the root
    assert 'target / "saved" / ".seeded"' in (ROOT / "auth.py").read_text(), \
        "the marker is back at the project root, where releases pick it up"

    # and nothing of the sort is actually tracked
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                                 capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001 - git is not required to run the tests
        return
    if tracked.returncode != 0:
        return
    listed = set(tracked.stdout.split())
    stray = [name for name in (".seeded", "prefs.json", "saved/.seeded",
                               "microgo.pt", "saved/MicroGo.json",
                               "saved/MicroLean.json")
             if name in listed]
    assert not stray, (
        f"{', '.join(stray)} tracked, so a release will carry it. Ignoring a "
        f"file does not untrack one already committed:\n"
        f"        git rm --cached {' '.join(stray)}")


@check("every account starts with the example designs")
def _():
    import shutil

    import auth

    assert auth.EXAMPLES.is_dir(), "the examples are not shipped"
    shipped = {p.stem for p in auth.EXAMPLES.glob("*.json")}
    assert shipped, "the examples folder is empty"

    backup = None
    if auth.DATA.exists():
        backup = auth.DATA.with_suffix(".backup")
        shutil.move(auth.DATA, backup)
    try:
        auth.register("firstone", "longenough1")     # adopts the shared workspace
        auth.register("secondone", "longenough2")
        home = auth.workspace_for("secondone")
        got = {p.stem for p in (home / "saved").glob("*.json")}
        assert shipped <= got, f"a new account is missing {shipped - got}"

        # deleting them is a decision, not an accident to be undone
        for path in (home / "saved").glob("*.json"):
            path.unlink()
        assert auth.seed(home) == 0, "a deleted example must not come back"
        assert not list((home / "saved").glob("*.json"))

        # but an example shipped in a later version does reach an old workspace
        probe = auth.EXAMPLES / "_probe_example.json"
        probe.write_text('{"name":"_probe_example","nodes":[],"edges":[]}')
        try:
            assert auth.seed(home) == 1, \
                "a newly shipped example never reaches existing workspaces"
            assert (home / "saved" / "_probe_example.json").exists()
            # and it is not delivered twice
            (home / "saved" / "_probe_example.json").unlink()
            assert auth.seed(home) == 0
        finally:
            probe.unlink(missing_ok=True)

        # an old workspace with the previous marker format, holding only some of
        # the examples, should still receive the ones it has never seen
        for path in (home / "saved").glob("*.json"):
            path.unlink()
        (home / "saved" / ".seeded").unlink(missing_ok=True)
        (home / ".seeded").write_text("the shipped designs were copied in once\n")
        kept = sorted(shipped)[:2]
        for name in kept:
            (home / "saved" / f"{name}.json").write_text("{}")

        delivered = auth.seed(home)
        assert delivered == len(shipped) - len(kept), \
            f"delivered {delivered}, expected {len(shipped) - len(kept)}"
        assert not (home / ".seeded").exists(), \
            "the old marker was left at the root, where a release would ship it"
        assert (home / "saved" / ".seeded").exists(), "the marker was not migrated"

        # but asking for them back works — from a clean slate, so the count is
        # the whole set rather than whatever the checks above left behind
        for path in (home / "saved").glob("*.json"):
            path.unlink()
        assert auth.restore_examples(home) == len(shipped), \
            "restoring should bring back every shipped example"

        # and restoring never treads on work of the same name
        mine = home / "saved" / (sorted(shipped)[0] + ".json")
        mine.write_text('{"name":"mine","nodes":[],"edges":[]}')
        assert auth.restore_examples(home) == 0
        assert json.loads(mine.read_text())["name"] == "mine", \
            "restoring overwrote the user's own design"
    finally:
        shutil.rmtree(auth.DATA, ignore_errors=True)
        if backup:
            shutil.move(backup, auth.DATA)


@check("every node on a workbook sheet actually draws")
def _():
    """Rendering stops at the first exception, so one bad node hides the rest.

    A reference to `inset` placed above its own declaration threw on the first
    Subgraph node and left sixteen of eighteen layers invisible — no error
    anywhere the user could see, just a mostly empty canvas with arrows
    pointing at nothing.
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        print("        (node absent, skipped)")
        return

    example = ROOT / "examples" / "GPT2.json"
    if not example.exists():
        print("        (GPT2 example absent, skipped)")
        return

    import main
    import workbook

    book = json.loads(example.read_text())
    analysis = workbook.analyze(book)
    sheet = next(s for s in book["sheets"] if s["name"] == book["main"])

    script = ROOT / "tests" / "_render_probe.js"
    page = PAGE[PAGE.index("<script>") + 8: PAGE.rindex("</script>")]
    page = page.replace("\nboot();", "\n")
    harness = """
const store = {};
function mk(id){return{id,value:"",textContent:"",innerHTML:"",style:{},dataset:{},
 classList:{_s:new Set(),add(...c){c.forEach(x=>this._s.add(x))},
  remove(...c){c.forEach(x=>this._s.delete(x))},toggle(){},contains(){return false}},
 addEventListener(){},appendChild(){},insertBefore(){},setAttribute(){},
 querySelector:()=>null,querySelectorAll:()=>[],focus(){},remove(){},
 scrollIntoView(){},closest:()=>null,firstChild:null,children:[],
 getBoundingClientRect:()=>({left:0,top:0,right:900,bottom:600,width:900,height:600})};}
global.window={addEventListener(){},open(){},innerWidth:1600,innerHeight:900,
  fetch:async()=>({ok:true,status:200,json:async()=>({})})};
global.document={activeElement:{tagName:"BODY"},body:mk("b"),
 getElementById:id=>store[id]||(store[id]=mk(id)),
 querySelectorAll:()=>[],querySelector:()=>mk("q"),
 createElement:()=>mk("n"),createElementNS:()=>mk("s")};
global.fetch=global.window.fetch;
global.EventSource=function(){};global.navigator={clipboard:{writeText:async()=>{}}};
global.setTimeout=(f)=>{try{f&&f()}catch(e){}};global.clearTimeout=()=>{};
"""
    tail = """
let titles = 0;
const realEl = el;
el = (tag, attrs, parent) => {
  if (tag === "text" && attrs && attrs.class === "node-title") titles++;
  return realEl(tag, attrs, parent);
};
state.graph = DATA.graph;
state.report = DATA.report;
state.specs = {};
DATA.specs.forEach(s => state.specs[s.name] = s);
let threw = "";
try { render(); } catch (e) { threw = e.constructor.name + ": " + e.message; }
console.log(JSON.stringify({ threw, titles }));
"""
    payload = {
        "graph": {"name": "GPT2", "nodes": sheet["nodes"], "edges": sheet["edges"]},
        "report": analysis["sheets"][book["main"]],
        "specs": layers.catalog(),
    }
    script.write_text(harness + "const DATA = " + json.dumps(payload) + ";\n"
                      + page + tail)
    try:
        result = subprocess.run([node, str(script)], capture_output=True,
                                text=True, timeout=90)
        line = [l for l in result.stdout.splitlines() if l.startswith("{")]
        assert line, f"the probe produced nothing: {result.stderr[-300:]}"
        outcome = json.loads(line[-1])
        assert not outcome["threw"], f"rendering threw: {outcome['threw']}"
        # Input and Output are circles and draw their text another way
        drawable = sum(1 for n in sheet["nodes"]
                       if n["type"] not in ("Input", "Output"))
        assert outcome["titles"] == drawable, \
            f"only {outcome['titles']} of {drawable} layers drew"
    finally:
        script.unlink(missing_ok=True)


@check("designs are opened by clicking, not by retyping a name")
def _():
    """The browser prompt listed the designs and then made you type one in."""
    for token in ('id="picker"', 'id="pickBody"', 'id="pickSearch"',
                  "function openPicker", "function openDesign", "data-open="):
        assert token in PAGE, f"{token} is missing from the picker"

    script = PAGE[PAGE.index("<script>"):]
    opener = script[script.index("function openPicker"):
                    script.index("function openDesign")]
    assert "prompt(" not in opener, "the picker still asks you to type a name"

    # the remaining prompts are for naming something new, which has nothing to
    # pick from — those are fine; picking from a list is not
    import re

    for match in re.finditer(r"[^\w.]prompt\(([^,)]*)", script):
        argument = match.group(1).lower()
        assert "which" not in argument and "open" not in argument, \
            f"something is still asking you to type a choice: {match.group(1)[:60]}"


@check("signing out locks the page instead of emptying it")
def _():
    """Signing out used to look exactly like losing everything.

    The page loaded, every API call answered 401, and the result was an app
    with no layers, no designs and no projects. Nothing was lost — but nothing
    said so.
    """
    assert "function lockOut" in PAGE, "there is no locked state"
    assert "window.fetch = async" in PAGE, \
        "401s are not intercepted, so a lost session empties the screen silently"
    boot = PAGE[PAGE.index("async function boot("):]
    boot = boot[: boot.index("\n}\n")]
    assert "lockOut()" in boot and "return" in boot, \
        "boot carries on loading with no session"
    assert "still on disk" in PAGE, \
        "the locked screen does not say the work is safe"


@check("accounts can be managed and turned off again")
def _():
    import shutil

    import accounts
    import auth

    backup = None
    if auth.DATA.exists():
        backup = auth.DATA.with_suffix(".backup")
        shutil.move(auth.DATA, backup)
    try:
        auth.register("one", "longenough1")
        auth.register("two", "longenough2")

        accounts.cmd_remove("two", purge=False)
        assert "two" not in auth.users()
        assert auth.workspace_for("two").exists(), \
            "removing an account should not delete its work"
        assert auth.workspace_for("one") == auth.HERE

        accounts.cmd_off()
        assert not auth.enabled(), "turning accounts off should leave none"
        assert auth.workspace_for(None) == auth.HERE, "the shared workspace comes back"
    finally:
        shutil.rmtree(auth.DATA, ignore_errors=True)
        if backup:
            shutil.move(backup, auth.DATA)


@check("accounts keep one person's work out of another's")
def _():
    import shutil

    import auth

    backup = None
    if auth.DATA.exists():
        backup = auth.DATA.with_suffix(".backup")
        shutil.move(auth.DATA, backup)
    try:
        assert not auth.enabled(), "a fresh install has no accounts"
        assert auth.workspace_for(None) == auth.HERE, \
            "with no accounts the app keeps its original layout"

        auth.register("first", "longenough1")
        assert auth.enabled()
        # the first account adopts the existing work rather than hiding it
        assert auth.workspace_for("first") == auth.HERE

        auth.register("second", "longenough2")
        assert auth.workspace_for("second") != auth.HERE
        assert auth.workspace_for("second") != auth.workspace_for("first")

        assert auth.check("first", "longenough1")
        assert not auth.check("first", "longenough2"), "passwords are not interchangeable"
        assert not auth.check("nobody", "longenough1")

        stored = auth.users()["first"]
        assert "longenough1" not in json.dumps(stored), "the password was stored"
        assert stored["salt"] != auth.users()["second"]["salt"], "salts must differ"

        token = auth.open_session("first")
        assert auth.user_for(token) == "first"
        auth.change_password("first", "longenough1", "brandnewpass")
        assert auth.user_for(token) is None, \
            "changing a password must end the other sessions"
        assert auth.check("first", "brandnewpass")

        for name in ("", "a", "has space", "UPPER!"):
            try:
                auth.register(name, "longenough1")
                raise AssertionError(f"{name!r} should be refused")
            except ValueError:
                pass
        try:
            auth.register("fine", "short")
            raise AssertionError("a short password should be refused")
        except ValueError:
            pass
    finally:
        shutil.rmtree(auth.DATA, ignore_errors=True)
        if backup:
            shutil.move(backup, auth.DATA)


@check("the account is bound where the endpoint can see it")
def _():
    """The binding must be an async dependency.

    Middleware runs call_next in another task and a sync dependency runs in a
    worker thread; a context variable set in either is invisible to the
    endpoint. Both wrong versions failed silently, giving every account the
    same workspace.
    """
    import inspect

    import main

    assert inspect.iscoroutinefunction(main.bind_user), \
        "bind_user must be async, or every account shares one workspace"
    assert any(getattr(d, "dependency", None) is main.bind_user
               for d in main.app.router.dependencies), \
        "bind_user is not applied to the app"


@check("a job keeps hold of the workspace it started in")
def _():
    import agents
    import train as T

    assert "home" in T.Job.__dataclass_fields__, \
        "a training thread cannot ask who is signed in, so it must be told"
    assert "home" in agents.Agent.__dataclass_fields__


@check("browsing a scanned folder cannot leave it")
def _():
    import main

    for attempt in ("../../etc/passwd", "/etc/passwd", "models/../../../etc/passwd"):
        try:
            main.scan_file({"root": "/tmp", "path": attempt})
            raise AssertionError(f"{attempt} should have been refused")
        except Exception as exc:  # noqa: BLE001
            assert "outside" in str(exc) or "No such file" in str(exc), exc


@check("the Go rules are the rules")
def _():
    import microgo as go

    def position(rows, to_move=go.BLACK, komi=0.5):
        size = len(rows)
        stones = []
        for row in rows:
            for ch in row.split():
                stones.append({"X": go.BLACK, "O": go.WHITE, ".": go.EMPTY}[ch])
        return go.Board(size, tuple(stones), to_move, None, 0, komi)

    # a stone with one liberty left is taken when it is filled
    board = position([". X . . .",
                      "X O . . .",
                      ". X . . .",
                      ". . . . .",
                      ". . . . ."])
    assert sorted(board.group(6)[1]) == [7], "liberties counted wrongly"
    assert board.play(7).stones[6] == go.EMPTY, "the stone was not captured"

    # a whole group goes together
    board = position([". X X . .",
                      "X O O X .",
                      ". X . . .",
                      ". . . . .",
                      ". . . . ."])
    after = board.play(12)
    assert after.stones[6] == after.stones[7] == go.EMPTY, "the group survived"

    # suicide is refused, unless it captures
    board = position([". O . . .",
                      "O . O . .",
                      ". O . . .",
                      ". . . . .",
                      ". . . . ."])
    assert not board.legal(6), "suicide was allowed"
    board = position(["X O . . .",
                      "O . . . .",
                      ". . . . .",
                      ". . . . .",
                      ". . . . ."], to_move=go.WHITE)
    assert board.legal(6), "a capturing move was refused as suicide"

    # ko: the immediate retake is forbidden
    board = position([". X O . .",
                      "X . X O .",
                      ". X O . .",
                      ". . . . .",
                      ". . . . ."], to_move=go.WHITE)
    taken = board.play(6)
    assert taken.ko is not None, "no ko point was recorded"
    assert not taken.legal(taken.ko), "the ko was retaken immediately"

    # area scoring, with komi
    even = position(["X X X X X", "X X X X X", ". . . . .",
                     "O O O O O", "O O O O O"])
    assert even.score() == -0.5, f"even position scored {even.score()}"
    alone = position(["X . . . .", ". . . . .", ". . . . .",
                      ". . . . .", ". . . . ."])
    assert alone.score() == 24.5, f"whole board scored {alone.score()}"

    # games finish and are decided
    import random

    rng = random.Random(3)
    board = go.Board(5)
    for _ in range(300):
        if board.over:
            break
        moves = go.sensible_moves(board)
        board = board.play(rng.choice(moves))
    assert board.over, "a random game did not finish"
    assert board.winner() in (go.BLACK, go.WHITE)


@check("search prefers the centre on an empty small board")
def _():
    import random

    import microgo as go

    rng = random.Random(0)
    board = go.Board(5)
    counts, _ = go.mcts(board, go.random_evaluator(rng), simulations=120, rng=rng)
    best = max(range(len(counts)), key=lambda i: counts[i])
    row, col = divmod(best, 5)
    assert 1 <= row <= 3 and 1 <= col <= 3, \
        f"search preferred {row},{col}, which is not near the centre"
    # passing is not offered while there is anything to play
    assert board.pass_move not in go.sensible_moves(board), \
        "a search that considers passing everywhere teaches a policy to pass"


@check("the proof kernel accepts only real proofs")
def _():
    import microlean as ml

    theorem = ml.Theorem.parse("((a * 1) * b)", "(a * b)")
    assert ml.check(theorem, [(0, (1,))]), "a correct proof was rejected"
    assert not ml.check(theorem, [(6, ())]), "a wrong proof was accepted"
    assert not ml.check(theorem, [(2, (1,))]), "an illegal move was accepted"
    assert not ml.check(theorem, []), "an empty proof closed a real goal"
    assert not ml.check(theorem, [(99, ())]), "a nonexistent rule was accepted"

    # everything the generator emits is checked before it is handed out
    made = ml.corpus(120, steps=3, seed=5)
    assert len(made) > 60, f"only {len(made)} theorems generated"
    for statement, proof in made:
        assert ml.check(statement, proof), f"generated an unprovable goal: {statement}"
        assert statement.lhs != statement.rhs, "a goal that was already closed"


@check("training and evaluation theorems can be kept apart")
def _():
    """A differently-seeded sample is not a held-out set.

    The reachable space is small, so an independent draw shared about a third
    of its theorems with training — which would have flattered the result badly.
    """
    import microlean as ml

    train = ml.corpus(1500, steps=3, seed=0)
    used = ml.statements(train)

    naive = ml.corpus(120, steps=3, seed=4242)
    shared = len(ml.statements(naive) & used)
    assert shared > 0, ("this test exists because independent seeds overlap; "
                        "if they no longer do, the risk it guards has changed")

    clean = ml.corpus(120, steps=3, seed=4242, exclude=used)
    assert not (ml.statements(clean) & used), "exclude did not keep them apart"


@check("the tactic space covers what the generator produces")
def _():
    import microlean as ml

    rows, labels = ml.training_pairs(400, steps=3, seed=1)
    assert rows and labels
    assert len(rows) == len(labels)
    assert all(len(r) == ml.CONTEXT for r in rows), "a state was the wrong length"
    assert all(0 <= v < ml.VOCAB for r in rows for v in r), "a token id is out of range"
    assert all(0 <= v < ml.N_TACTICS for v in labels), "a tactic id is out of range"


@check("a class can be read and tried before importing it")
def _():
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import shutil
    import tempfile

    import main

    root = Path(tempfile.mkdtemp())
    try:
        (root / "net.py").write_text(
            "import torch.nn as nn\n"
            "\n"
            "\n"
            "class Small(nn.Module):\n"
            '    """A tiny thing."""\n'
            "    def __init__(self, width=8):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(width, width)\n"
            "\n"
            "    def forward(self, x):\n"
            "        return self.fc(x)\n")

        probe = type("P", (), {"root": str(root), "file": "net.py",
                               "cls": "Small", "arguments": "", "setup": "",
                               "input_shape": [8]})()
        source = main.scan_source(probe)
        assert "class Small" in source["source"]
        assert source["signature"] == "width=8", source["signature"]
        assert source["doc"] == "A tiny thing."
        assert source["methods"] == ["__init__", "forward"]

        verdict = main.scan_try(probe)
        assert verdict["ok"], verdict
        assert verdict["learnables"] == 8 * 8 + 8, verdict

        # and a class that cannot be traced reports why rather than throwing
        (root / "bad.py").write_text(
            "import torch.nn as nn\n"
            "\n"
            "\n"
            "class Variadic(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(8, 8)\n"
            "\n"
            "    def forward(self, x, **kw):\n"
            "        for k in kw:\n"
            "            x = x + kw[k]\n"
            "        return self.fc(x)\n")
        bad = type("P", (), {"root": str(root), "file": "bad.py",
                             "cls": "Variadic", "arguments": "", "setup": "",
                             "input_shape": [8]})()
        verdict = main.scan_try(bad)
        assert not verdict["ok"]
        assert "variadic" in verdict["reason"], verdict["reason"]
    finally:
        shutil.rmtree(root)


@check("a failed try says which kind of failure it was")
def _():
    """One label for every failure sends people to fix the wrong thing.

    A class that was never given its config had the same badge as one whose
    forward() genuinely cannot be traced — but only the first is the user's to
    fix, and it is fixed by typing in the box beside it.
    """
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import shutil
    import tempfile

    import main

    root = Path(tempfile.mkdtemp())
    try:
        (root / "net.py").write_text(
            "import torch.nn as nn\n"
            "\n"
            "\n"
            "class NeedsConfig(nn.Module):\n"
            "    def __init__(self, width):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(width, width)\n"
            "\n"
            "    def forward(self, x):\n"
            "        return self.fc(x)\n"
            "\n"
            "\n"
            "class Variadic(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(8, 8)\n"
            "\n"
            "    def forward(self, x, **kw):\n"
            "        for k in kw:\n"
            "            x = x + kw[k]\n"
            "        return self.fc(x)\n")

        def probe(cls, arguments=""):
            body = type("P", (), {"root": str(root), "file": "net.py", "cls": cls,
                                  "arguments": arguments, "setup": "",
                                  "input_shape": [8]})()
            return main.scan_try(body)

        # not given its argument: the user's to fix, and the badge should say so
        missing = probe("NeedsConfig")
        assert not missing["ok"]
        assert missing["kind"] == "arguments", missing

        # given it, the same class imports
        given = probe("NeedsConfig", "8")
        assert given["ok"], given

        # genuinely untraceable is a different answer
        hopeless = probe("Variadic")
        assert hopeless["kind"] == "trace", hopeless

        # and a name that is not there again
        gone = probe("NotHere")
        assert gone["kind"] == "missing", gone

    finally:
        shutil.rmtree(root)

    assert "needs arguments" in PAGE and "cannot be traced" in PAGE, \
        "the badge does not distinguish the kinds"


@check("the import dialog can be moved, resized and split")
def _():
    for token in ('id="sheetGrip"', 'id="scanDrag"', 'id="importGrab"',
                  "function applyImportSize", "function wireScanDrag",
                  "nwse-resize", "col-resize"):
        assert token in PAGE, f"{token} is missing from the dialog sizing"

    # a size remembered on a large screen must be clamped, not applied blindly
    script = PAGE[PAGE.index("function applyImportSize"):]
    body = script[: script.index("\n}\n")]
    assert "window.innerWidth" in body and "Math.min" in body, \
        "applyImportSize does not clamp to the window"

    # the argument boxes must beat the dialog's generic field rule
    css = PAGE[PAGE.index("<style>"): PAGE.index("</style>")]
    assert "input[type=text].scanargs" in css, \
        "the argument boxes will stretch to the full row width"


@check("the scan results are browsable, not just listed")
def _():
    for token in ("function renderScanResults", "function peekAt",
                  "function tryClass", 'id="scanFilter"', 'id="scanPeek"',
                  "data-try=", "sheet wide"):
        assert token in PAGE, f"{token} is missing from the scan browser"
    # a library scan returns hundreds of classes; a filter is not optional
    assert "Filter " in PAGE, "there is no way to narrow the list"


@check("no placeholder can be mistaken for a value")
def _():
    """The repository box suggested exactly the address people wanted.

    Grey placeholder text reading `https://github.com/karpathy/minGPT` looks
    filled in, so Fetch was pressed on an empty field and answered "paste a
    repository address" — about a box that appeared to contain one.
    """
    import re

    markup = PAGE[: PAGE.index("<script>")]
    for match in re.finditer(r'placeholder="([^"]+)"', markup):
        text = match.group(1)
        assert not text.startswith("http"), \
            f"the placeholder {text!r} is a usable value and will look entered"
        assert not text.startswith("/"), \
            f"the placeholder {text!r} is a usable path and will look entered"


@check("import does not guess what you meant")
def _():
    """Pressing Import with nothing chosen fetched resnet18.

    The architecture dropdown always held a value, so "nothing chosen" was not a
    state the dialog could be in — and the failure that surfaced was about
    torchvision, which had nothing to do with what the person was doing.
    """
    assert "\u2014 none \u2014" in PAGE, \
        "the architecture list has no empty option, so it always means something"

    script = PAGE[PAGE.index("<script>"):]
    handler = script[script.index('$("btnImportGo").addEventListener'):]
    handler = handler[: handler.index("\n});")]
    assert "Nothing chosen yet" in handler, \
        "the handler still falls through instead of saying nothing was chosen"
    for control in ("im_code", "im_file", "im_arch", "im_repo"):
        assert control in handler, f"{control} is not considered before importing"


@check("a missing package says how to install it")
def _():
    import main

    message = main._missing_package(ModuleNotFoundError("No module named 'torchvision'"))
    assert message and "pip install torchvision" in message, message
    assert sys.executable in message, "it does not say which interpreter"

    # anything else is passed through rather than dressed up
    assert main._missing_package(ValueError("unrelated")) is None


@check("a GitHub address is understood before anything is downloaded")
def _():
    import importer

    for address, owner, repo, ref, path in [
        ("https://github.com/karpathy/minGPT", "karpathy", "minGPT", "", ""),
        ("github.com/karpathy/minGPT", "karpathy", "minGPT", "", ""),
        ("https://github.com/pytorch/vision.git", "pytorch", "vision", "", ""),
        ("https://github.com/karpathy/minGPT/tree/master/mingpt",
         "karpathy", "minGPT", "master", "mingpt"),
        # what people type when they are not copying an address
        ("karpathy/minGPT", "karpathy", "minGPT", "", ""),
    ]:
        found = importer.parse_repo(address)
        assert found["owner"] == owner and found["repo"] == repo, found
        assert found["ref"] == ref and found["path"] == path, found

    for bad in ("", "onlyowner", "https://github.com/onlyowner"):
        try:
            importer.parse_repo(bad)
            raise AssertionError(f"{bad!r} should have been refused")
        except importer.ImportError_:
            pass


@check("a class can be built from a few lines of setup")
def _():
    """One expression is not enough for a real configuration.

    minGPT wants a default config with half a dozen fields set on it before
    anything can be constructed, which no single argument expression expresses.
    """
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import shutil
    import tempfile

    import importer

    root = Path(tempfile.mkdtemp())
    try:
        (root / "net.py").write_text(
            "import torch.nn as nn\n"
            "\n"
            "\n"
            "class Config:\n"
            "    width = 0\n"
            "\n"
            "\n"
            "class Net(nn.Module):\n"
            "    def __init__(self, cfg):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(cfg.width, cfg.width)\n"
            "\n"
            "    def forward(self, x):\n"
            "        return self.fc(x)\n")

        setup = "cfg = Config()\ncfg.width = 8\n"
        graph = importer.from_folder(str(root), "net.py", "Net", [8],
                                     "cfg", setup)
        graph.pop("_entry", None)
        graph.pop("_notes", None)
        expected = graph.pop("_expected_parameters", 0)
        g, rep = analyzed(graph)
        assert rep["ok"], rep["errors"]
        assert rep["total_learnables"] == expected == 8 * 8 + 8

        # and a setup that does not run says so, rather than failing later
        try:
            importer.from_folder(str(root), "net.py", "Net", [8], "cfg",
                                 "cfg = Config(\ncfg.width = 8")
            raise AssertionError("a broken setup should have been refused")
        except importer.ImportError_ as exc:
            assert "setup did not run" in str(exc), exc
    finally:
        shutil.rmtree(root)


@check("a folder inside a virtual environment can still be scanned")
def _():
    """Pointing deliberately at a library is not the same as crawling into one.

    The skip list judged the absolute path, so a folder that merely sat inside
    .venv had every one of its files ignored — the scan reported reading zero
    files and finding nothing, with no indication why.
    """
    import shutil
    import tempfile

    import importer

    root = Path(tempfile.mkdtemp())
    try:
        inside = root / ".venv" / "lib" / "pkg"
        inside.mkdir(parents=True)
        (inside / "net.py").write_text(
            "import torch.nn as nn\n"
            "class Deep(nn.Module):\n"
            "    def forward(self, x): return x\n")

        # pointed at the library itself, it is read
        result = importer.scan_folder(str(inside))
        assert result["files"] == 1, f"read {result['files']} files"
        assert [m["cls"] for m in result["models"]] == ["Deep"]

        # pointed at the project above it, the environment is left alone
        (root / "mine.py").write_text(
            "import torch.nn as nn\n"
            "class Mine(nn.Module):\n"
            "    def forward(self, x): return x\n")
        outer = importer.scan_folder(str(root))
        assert [m["cls"] for m in outer["models"]] == ["Mine"], \
            "a project scan should not crawl into its own environment"
    finally:
        shutil.rmtree(root)


@check("a folder is scanned without running any of it")
def _():
    import shutil
    import tempfile

    import importer

    root = Path(tempfile.mkdtemp())
    try:
        (root / "models").mkdir()
        (root / "models" / "__init__.py").write_text("")
        (root / "models" / "net.py").write_text(
            "import torch.nn as nn\n"
            "BOOBY_TRAP = exec  # a scan must never execute anything\n"
            "class Good(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(4, 2)\n"
            "    def forward(self, x): return self.fc(x)\n"
            "class Picky(nn.Module):\n"
            "    def __init__(self, width, depth):\n"
            "        super().__init__()\n"
            "    def forward(self, x): return x\n")
        (root / "broken.py").write_text("class Oops(nn.Module)\n    pass\n")
        (root / "evil.py").write_text(
            "raise RuntimeError('scanning must not import this file')\n")

        result = importer.scan_folder(str(root))
        names = {m["cls"]: m for m in result["models"]}
        assert set(names) == {"Good", "Picky"}, names
        assert names["Good"]["arguments"] == 0
        assert names["Picky"]["arguments"] == 2
        assert any("broken.py" in item["file"] for item in result["skipped"])
        # evil.py raising at import time proves nothing ran: scanning survived it
    finally:
        shutil.rmtree(root)


@check("a class imports from a folder with its relative imports intact")
def _():
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import shutil
    import tempfile

    import importer

    root = Path(tempfile.mkdtemp())
    try:
        (root / "models").mkdir()
        (root / "models" / "__init__.py").write_text("")
        (root / "models" / "blocks.py").write_text(
            "import torch.nn as nn\n"
            "class Piece(nn.Module):\n"
            "    def __init__(self, cin=3, cout=8):\n"
            "        super().__init__()\n"
            "        self.conv = nn.Conv2d(cin, cout, 3, padding=1)\n"
            "    def forward(self, x): return self.conv(x)\n")
        (root / "models" / "whole.py").write_text(
            "import torch.nn as nn\n"
            "from models.blocks import Piece\n"
            "class Whole(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.stem = Piece()\n"
            "        self.head = nn.Linear(8 * 32 * 32, 2)\n"
            "    def forward(self, x):\n"
            "        return self.head(self.stem(x).flatten(1))\n")
        graph = importer.from_folder(str(root), "models/whole.py", "Whole",
                                     [3, 32, 32])
        graph.pop("_entry", None)
        graph.pop("_notes", None)
        g, rep = analyzed(graph)
        assert rep["ok"], rep["errors"]
        kinds = [n["type"] for n in graph["nodes"]]
        assert "Conv2d" in kinds and "Linear" in kinds, kinds
    finally:
        shutil.rmtree(root)


@check("a workbook resolves shapes across sheets and counts once")
def _():
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import train as T
    import workbook

    book = {"name": "Split", "main": "main", "sheets": [
        {"name": "stem", "nodes": [
            {"id": "i", "type": "Input", "params": {"shape": [3, 32, 32]}},
            {"id": "c", "type": "Conv2d",
             "params": {"filters": 32, "kernel": 3, "padding": "same"}},
            {"id": "o", "type": "Output", "params": {}}],
         "edges": [{"id": "e1", "source": "i", "target": "c"},
                   {"id": "e2", "source": "c", "target": "o"}]},
        {"name": "orphan", "nodes": [
            {"id": "i", "type": "Input", "params": {"shape": [4]}},
            {"id": "l", "type": "Linear", "params": {"units": 999}},
            {"id": "o", "type": "Output", "params": {}}],
         "edges": [{"id": "e1", "source": "i", "target": "l"},
                   {"id": "e2", "source": "l", "target": "o"}]},
        {"name": "main", "nodes": [
            {"id": "i", "type": "Input", "params": {"shape": [3, 32, 32]}},
            {"id": "s", "type": "Subgraph", "params": {"sheet": "stem"}},
            {"id": "g", "type": "GlobalAvgPool", "params": {}},
            {"id": "l", "type": "Linear", "params": {"units": 10}},
            {"id": "o", "type": "Output", "params": {"task": "classification"}}],
         "edges": [{"id": "e1", "source": "i", "target": "s"},
                   {"id": "e2", "source": "s", "target": "g"},
                   {"id": "e3", "source": "g", "target": "l"},
                   {"id": "e4", "source": "l", "target": "o"}]}]}

    analysis = workbook.analyze(book)
    assert analysis["ok"], analysis
    assert analysis["sheets"]["main"]["nodes"]["s"]["out_shape"] == [32, 32, 32]

    source = workbook.to_pytorch(book, analysis)
    assert "class Stem(nn.Module):" in source
    assert "Stem()" in source
    assert "999" not in source, "the orphan sheet is not part of the model"

    import torch
    model = T.build_model(source, "Main")
    real = sum(p.numel() for p in model.parameters())
    assert real == analysis["total_learnables"], \
        f"canvas {analysis['total_learnables']}, torch {real}"
    assert tuple(model(torch.randn(2, 3, 32, 32)).shape) == (2, 10)


@check("sheets referencing each other in a circle are refused")
def _():
    import workbook

    book = {"name": "Loop", "main": "a", "sheets": [
        {"name": "a", "nodes": [
            {"id": "i", "type": "Input", "params": {"shape": [4]}},
            {"id": "s", "type": "Subgraph", "params": {"sheet": "b"}},
            {"id": "o", "type": "Output", "params": {}}],
         "edges": [{"id": "e1", "source": "i", "target": "s"},
                   {"id": "e2", "source": "s", "target": "o"}]},
        {"name": "b", "nodes": [
            {"id": "i", "type": "Input", "params": {"shape": [4]}},
            {"id": "s", "type": "Subgraph", "params": {"sheet": "a"}},
            {"id": "o", "type": "Output", "params": {}}],
         "edges": [{"id": "e1", "source": "i", "target": "s"},
                   {"id": "e2", "source": "s", "target": "o"}]}]}
    analysis = workbook.analyze(book)
    assert not analysis["ok"]
    assert analysis["cycle"], "the cycle was not named"
    assert "circle" in analysis["sheets"]["a"]["errors"][0]
    code = workbook.to_pytorch(book, analysis)
    assert code.startswith("#"), "no code should be generated for a cycle"


@check("the canvas carries sheet tabs and cross-sheet continuation")
def _():
    for token in ('id="sheetTabs"', "function switchSheet", "function addSheet",
                  "function renameSheet", "function deleteSheet",
                  "function openReferencedSheet", "continues on",
                  "analyze-book", "function scanFolder", 'id="im_folder"'):
        assert token in PAGE, f"{token} is missing"


@check("agents propose variants that actually build")
def _():
    import agents

    graph = {"name": "Study", "nodes": [
        {"id": "i", "type": "Input", "params": {"shape": [3, 16, 16]}},
        {"id": "c", "type": "Conv2d",
         "params": {"filters": 16, "kernel": 3, "padding": "same"}},
        {"id": "a", "type": "Activation", "params": {"kind": "relu"}},
        {"id": "f", "type": "Flatten", "params": {}},
        {"id": "l1", "type": "Linear", "params": {"units": 64}},
        {"id": "l2", "type": "Linear", "params": {"units": 4}, "label": "head"},
        {"id": "o", "type": "Output", "params": {"task": "classification"}}],
        "edges": [{"id": "e1", "source": "i", "target": "c"},
                  {"id": "e2", "source": "c", "target": "a"},
                  {"id": "e3", "source": "a", "target": "f"},
                  {"id": "e4", "source": "f", "target": "l1"},
                  {"id": "e5", "source": "l1", "target": "l2"},
                  {"id": "e6", "source": "l2", "target": "o"}]}

    for kind in ("sweep", "search", "repair"):
        trials = agents.BUILDERS[kind](json.loads(json.dumps(graph)), {"trials": 6})
        assert trials, f"{kind} proposed nothing"
        for trial in trials:
            rep = G.analyze(G.parse(trial["graph"]))
            assert rep["ok"], \
                f"{kind} proposed '{trial['label']}' which does not build: {rep['errors'][:1]}"
            assert trial.get("learnables") is not None, \
                f"{kind} trial '{trial['label']}' reports no size"


@check("architecture search leaves the head alone")
def _():
    import agents

    graph = {"name": "S", "nodes": [
        {"id": "i", "type": "Input", "params": {"shape": [8]}},
        {"id": "h", "type": "Linear", "params": {"units": 32}},
        {"id": "a", "type": "Activation", "params": {"kind": "relu"}},
        {"id": "o2", "type": "Linear", "params": {"units": 5}, "label": "head"},
        {"id": "o", "type": "Output", "params": {"task": "classification"}}],
        "edges": [{"id": "e1", "source": "i", "target": "h"},
                  {"id": "e2", "source": "h", "target": "a"},
                  {"id": "e3", "source": "a", "target": "o2"},
                  {"id": "e4", "source": "o2", "target": "o"}]}
    trials = agents.BUILDERS["search"](json.loads(json.dumps(graph)), {"trials": 8})
    widened = [t for t in trials if "width" in t["label"]]
    assert widened, "search proposed no width variants"
    for trial in widened:
        head = [n for n in trial["graph"]["nodes"] if n.get("label") == "head"][0]
        assert head["params"]["units"] == 5, (
            f"{trial['label']} resized the head to {head['params']['units']}; "
            f"the number of classes is not a hyperparameter")


@check("the repair agent turns review findings into trials")
def _():
    import agents

    graph = {"name": "R", "nodes": [
        {"id": "i", "type": "Input", "params": {"shape": [8]}},
        {"id": "l1", "type": "Linear", "params": {"units": 64}},
        {"id": "l2", "type": "Linear", "params": {"units": 4}, "label": "head"},
        {"id": "o", "type": "Output", "params": {"task": "classification"}}],
        "edges": [{"id": "e1", "source": "i", "target": "l1"},
                  {"id": "e2", "source": "l1", "target": "l2"},
                  {"id": "e3", "source": "l2", "target": "o"}]}
    trials = agents.BUILDERS["repair"](json.loads(json.dumps(graph)), {"trials": 6})
    labels = [t["label"] for t in trials]
    assert "as drawn" in labels, "there is no baseline to compare against"
    assert any("activation between" in l for l in labels), labels
    # and the fix must actually separate the two dense layers
    fixed = [t for t in trials if "activation between" in t["label"]][0]
    kinds = [n["type"] for n in fixed["graph"]["nodes"]]
    assert kinds.count("Activation") == 1, kinds


@check("the assistant edits the graph and finds real problems")
def _():
    import assistant

    graph = {"name": "T", "nodes": [
        {"id": "i", "type": "Input", "params": {"shape": [3, 32, 32]}},
        {"id": "c", "type": "Conv2d",
         "params": {"filters": 32, "kernel": 3, "padding": "same"}},
        {"id": "a", "type": "Activation", "params": {"kind": "relu"}},
        {"id": "f", "type": "Flatten", "params": {}},
        {"id": "l1", "type": "Linear", "params": {"units": 256}},
        {"id": "l2", "type": "Linear", "params": {"units": 10}},
        {"id": "o", "type": "Output", "params": {"task": "classification"}}],
        "edges": [{"id": "e1", "source": "i", "target": "c"},
                  {"id": "e2", "source": "c", "target": "a"},
                  {"id": "e3", "source": "a", "target": "f"},
                  {"id": "e4", "source": "f", "target": "l1"},
                  {"id": "e5", "source": "l1", "target": "l2"},
                  {"id": "e6", "source": "l2", "target": "o"}]}

    def fresh():
        return json.loads(json.dumps(graph))

    # two Linear layers with nothing between them are one Linear layer
    found = assistant.handle(fresh(), "review")["reply"]
    assert "compose to a single linear layer" in found, found
    assert "8,388,608 weights" in found, found

    # an edit comes back as a graph, with the gap stitched where needed
    added = assistant.handle(fresh(), "add dropout after the activation")
    assert added["changed"], added
    assert [n["type"] for n in added["graph"]["nodes"]].count("Dropout") == 1

    removed = assistant.handle(fresh(), "remove the flatten")
    kinds = [n["type"] for n in removed["graph"]["nodes"]]
    assert "Flatten" not in kinds, kinds
    sources = {e["source"] for e in removed["graph"]["edges"]}
    targets = {e["target"] for e in removed["graph"]["edges"]}
    assert "a" in sources and "l1" in targets, "the chain was not rejoined"

    # generated names, which is what people read off the canvas
    frozen = assistant.handle(fresh(), "freeze conv2d_1")
    assert frozen["changed"] and frozen["reply"].startswith("Froze"), frozen
    conv = [n for n in frozen["graph"]["nodes"] if n["id"] == "c"][0]
    assert conv["params"].get("_frozen") is True

    settings = assistant.handle(fresh(), "set units to 64 on linear_1")
    target = [n for n in settings["graph"]["nodes"] if n["id"] == "l1"][0]
    assert target["params"]["units"] == 64, target

    # and it says so when it does not understand, rather than inventing
    puzzled = assistant.handle(fresh(), "make it better somehow")
    assert not puzzled.get("changed")
    assert "did not recognise" in puzzled["reply"]


@check("the assistant refuses settings a layer does not have")
def _():
    import assistant

    graph = {"name": "T", "nodes": [
        {"id": "i", "type": "Input", "params": {"shape": [8]}},
        {"id": "l", "type": "Linear", "params": {"units": 4}},
        {"id": "o", "type": "Output", "params": {}}],
        "edges": [{"id": "e1", "source": "i", "target": "l"},
                  {"id": "e2", "source": "l", "target": "o"}]}
    reply = assistant.handle(json.loads(json.dumps(graph)),
                             "set kernel to 3 on linear")
    assert not reply.get("changed"), "it should not invent a setting"
    assert "no setting called kernel" in reply["reply"], reply


@check("selecting a layer leaves the open tab alone")
def _():
    """The panel used to jump to Layer unless the open tab was on a list of
    exceptions — a list that grew with every new tab and had never gained Code,
    so clicking a layer to find its line threw you out of the file.

    Only two tabs are useless without a selection. Naming those is a list that
    stays short; naming the others is one that goes stale.
    """
    script = PAGE[PAGE.index("<script>"):]
    body = script[script.index("function renderInspector"):]
    body = body[: body.index("\n}\n")]

    assert "NEEDS_NOTHING_SELECTED" in body, \
        "the panel still decides by listing the tabs it must not disturb"
    assert 'state.sidePanel !== "code"' not in body, \
        "the old exception list is still there"

    listed = body[body.index("NEEDS_NOTHING_SELECTED"):]
    listed = listed[: listed.index("]")]
    for tab in ("code", "math", "run", "story", "train", "ask", "guide"):
        assert f'"{tab}"' not in listed, \
            f"selecting a layer would still throw you out of the {tab} tab"


@check("a workbook trains as a whole, not a sheet at a time")
def _():
    """GPT2's model sheet places twelve Subgraph nodes that stand for a class
    the block sheet defines. Generating the model sheet alone produces a file
    naming a class that is not in it, and training it fails with
    "NameError: name 'Block' is not defined" — true, and no help at all.
    """
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import json as _json

    import main
    import workbook as wb

    example = ROOT / "examples" / "GPT2.json"
    if not example.exists():
        print("        (GPT2 absent, skipped)")
        return
    book = _json.loads(example.read_text())
    assert len(book["sheets"]) > 1, "this test needs a multi-sheet design"

    sheet = wb.sheet_graph(book, book.get("main"))
    g, rep = analyzed(sheet)
    alone = codegen.to_pytorch(g, rep)
    assert "Block()" in alone, "the sheet does not reference another sheet"
    assert "class Block" not in alone, \
        "the sheet already carries the class, so this test proves nothing"

    whole = wb.to_pytorch(book, wb.analyze(book))
    assert "class Block" in whole, "the workbook does not define the class"

    # the endpoint has to be told about the book, and the page has to send it
    assert "book" in main.TrainPayload.model_fields, \
        "the training endpoint cannot receive a workbook"
    assert "function workbookForRun" in PAGE, \
        "the page never sends the workbook"
    script = PAGE[PAGE.index("<script>"):]
    body = script[script.index("function workbookForRun"):]
    body = body[: body.index("\n}\n")]
    assert "commitSheet()" in body, \
        "the canvas is not written back before the book is sent"


@check("a dataset that cannot fit the design is refused before the run")
def _():
    """A language model asked to read MNIST does not fail politely: the image
    loader is asked to resize a picture into a token sequence, and on one
    machine that aborted the process outright."""
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import train as T

    tokens = {"nodes": [{"id": "i", "type": "Input",
                         "params": {"shape": [64], "dtype": "long"}}]}
    pixels = {"nodes": [{"id": "i", "type": "Input",
                         "params": {"shape": [1, 28, 28]}}]}

    for dataset in ("mnist", "cifar10"):
        try:
            T.check_dataset_fits({"dataset": dataset, "graph": tokens}, [[64]])
            raise AssertionError(f"{dataset} was allowed for a token model")
        except T.DataError as exc:
            assert "token indices" in str(exc), exc

    try:
        T.check_dataset_fits({"dataset": "text", "graph": pixels}, [[1, 28, 28]])
        raise AssertionError("text was allowed for an image model")
    except T.DataError as exc:
        assert "whole numbers" in str(exc), exc

    # and the pairings that do work are not disturbed
    T.check_dataset_fits({"dataset": "mnist", "graph": pixels}, [[1, 28, 28]])
    T.check_dataset_fits({"dataset": "text", "graph": tokens}, [[64]])
    T.check_dataset_fits({"dataset": "synthetic", "graph": tokens}, [[64]])

    # the check runs in the endpoint, where the message reaches the form
    import inspect

    import main

    assert "check_dataset_fits" in inspect.getsource(main.post_train), \
        "the check runs in the training thread, too late to be shown"


@check("a failed run says why, where the button was pressed")
def _():
    """The reason was written into the log strip at the bottom of the panel,
    which is usually collapsed. What a failed run actually said, on screen, was
    the word "error" in a corner."""
    assert 'id="trainError"' in PAGE, "there is nowhere for the reason to appear"

    script = PAGE[PAGE.index("<script>"):]
    assert "function showTrainError" in script
    assert "showTrainError(m.message)" in script, \
        "the error message is still only logged"

    # the banner sits above the button that starts a run, not below the fold.
    # Positions are taken from the whole page: the first <script> tag comes
    # before this markup, so slicing at it leaves nothing to search.
    assert PAGE.index('id="trainError"') < PAGE.index('id="btnStart"'), \
        "the reason appears after the button, out of sight"

    # and a new run clears the last failure rather than leaving it to confuse
    at = script.index('API + "/api/train"')
    starter = script[at - 600: at + 600]
    assert "clearTrainError()" in starter, \
        "a previous failure is still showing when the next run starts"

    # Every way a run can fail has to say so. Fixing one branch and leaving its
    # sibling is how "error" became "blocked" and stayed just as silent.
    import re

    for match in re.finditer(r'\$\("trainStatus"\)\.textContent\s*=\s*([^;]+);',
                             script):
        setting = match.group(1).strip().strip('"')
        if setting in ("blocked", "error", "disconnected"):
            window = script[max(0, match.start() - 900): match.end() + 400]
            assert "showTrainError(" in window, \
                f"the {setting!r} state gives no reason in the panel"

    # a dropped stream is not the same as a finished run
    dropped = script[script.index("es.onerror"):]
    dropped = dropped[: dropped.index("\n  };") + 5]
    assert "showTrainError(" in dropped, \
        "losing the connection ends the run in silence"


@check("the corpus hint changes the design rather than describing the change")
def _():
    """It said "set Embedding vocab and final Linear units to 48" and left you
    to find the two layers. On GPT2 that mismatch is 77M parameters of
    embedding table for a 48-symbol alphabet — worth acting on, and worth
    acting on with one press."""
    assert "function matchVocabulary" in PAGE, "the hint is still only a sentence"

    script = PAGE[PAGE.index("<script>"):]
    body = script[script.index("function matchVocabulary"):]
    body = body[: body.index("\n}\n")]
    assert "pushHistory()" in body, "the change cannot be undone"
    assert "commitSheet()" in body, \
        "a workbook would keep the old numbers, since the canvas is not the truth"
    assert "analyze()" in body, "the parameter count would not be recomputed"
    for called in ("describeCorpus", "toast"):
        assert called in body, f"{called} is not called"
        assert f"function {called}" in script or f"{called} =" in script, \
            f"{called} does not exist"


@check("a run says how much data there is per parameter")
def _():
    """Two numbers that explain most disappointing runs and were on screen
    nowhere: parameters per training value, and how many passes over the data
    the run will make. A GPT-2 on a 470k-character corpus is 346 parameters per
    character seeing it 1.09 times, and neither figure was visible."""
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import train as T

    body = inspect.getsource(T._report_capacity)
    assert "per value" in body, "the density is not reported"
    assert "passes" in body, "how much of the data will be seen is not reported"

    # it must be called where the loader exists, not before it
    run = inspect.getsource(T._run)
    at = run.index("_report_capacity(")
    before = run[:at]
    assert "train_loader, val_loader, meta = _make_loaders(" in before, \
        "the report runs before the loaders exist, which is an UnboundLocalError"

    # and it must not refuse anything: plenty of useful work happens at
    # ratios like these deliberately
    assert "raise" not in body, "the report blocks a run instead of describing it"

    class Stub:
        learnables = 2_000_000
        notes: list = []

        def emit(self, kind, **payload):
            self.notes.append((kind, payload.get("message", "")))

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(torch.zeros(64, 16, dtype=torch.long),
                                      torch.zeros(64, 16, dtype=torch.long)),
                        batch_size=8)
    job = Stub()
    T._report_capacity(job, {"epochs": 1}, loader, [[16]])
    kinds = [k for k, _ in job.notes]
    assert "warning" in kinds, "nothing was reported at all"
    said = " ".join(m for _, m in job.notes)
    assert "per value" in said and "memorise" in said, said
    assert "1.0 time" in said, said


@check("sampling works whatever the prompt's length")
def _():
    """A model written with explicit reshapes has its context length baked
    into its arithmetic.

    The imported GPT-2 block reshapes to [B, 64, 12, 64], so a twenty-character
    prompt produced "shape '[1, 64, 12, 64]' is invalid for input of size
    15360" — which is arithmetic, not an explanation. The window is padded to
    the full context now.
    """
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import json as _json

    import torch

    import train as T
    import workbook as wb

    body = inspect.getsource(T.generate_text)
    assert "used < block" in body, "the window is not padded to the context length"
    assert "new_zeros" in body, "nothing pads a short prompt"

    example = ROOT / "examples" / "GPT2.json"
    if not example.exists():
        print("        (GPT2 absent, skipped)")
        return
    book = _json.loads(example.read_text())
    # shrink it so this stays a test rather than a training run
    for sheet in book["sheets"]:
        for node in sheet["nodes"]:
            if node["type"] == "Embedding":
                node["params"]["vocab"] = 48
            if node["type"] == "Linear" and node.get("label") == "lm_head":
                node["params"]["units"] = 48
    analysis = wb.analyze(book)
    assert analysis["ok"], "the shrunk book does not resolve"
    model = T.build_model(wb.to_pytorch(book, analysis), "Model").eval()

    vocab = [chr(97 + (i % 26)) for i in range(48)]
    for prompt in ("user: how do I add a", "", "z" * 200):
        out = T.generate_text(model, vocab, 64, prompt, max_new_tokens=3)
        assert out["continuation"], f"nothing was generated from {len(prompt)} chars"
        assert out["prompt"] == prompt or not prompt, \
            "the prompt was not returned as given"


@check("a model that predicts at every position trains")
def _():
    """A design scoring every position IS a language model, whatever its
    Output layer is labelled.

    Trusting the label gave "Expected target size [64, 50257], got [64, 64]" —
    which describes the tensors and not the mistake. And the same label check
    appeared twice: once for the loss and once for accuracy, where it produced
    a different error from the same cause.
    """
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import inspect

    import train as T

    body = inspect.getsource(T._run)

    # The question is asked once and the answer used everywhere. Five separate
    # label checks is how the loss came to be fixed while the accuracy, the
    # perplexity and the sampling still disagreed with it.
    assert "o.dim() == 3 and yb.dim() == 2" in body, \
        "the loss still trusts the Output layer's label alone"
    assert "lm_like = [" in body, "there is no single determination"
    assert "is_lm" not in body, \
        "the old label-only flag is still being consulted somewhere"

    # everything downstream reads that one flag. The anchors are code, not
    # prose: searching for "perplexity" found the comment above explaining the
    # change and passed on that.
    for feature in ('row["perplexity"]', "_sample_text(", "argmax(-1)"):
        at = body.index(feature)
        window = body[max(0, at - 400): at]
        assert "lm_like[0]" in window, \
            f"{feature} decides for itself rather than reading the flag"

    # the note about it must be said once, not once per batch
    assert "said_lm = [False]" in body, \
        "nothing tracks whether the note has been said"
    assert "not job.notes" not in body, \
        "the guard reads a list that emit() never appends to, so it always fires"

    # and the shipped example should not need the inference in the first place
    import json as _json

    example = ROOT / "examples" / "GPT2.json"
    if example.exists():
        book = _json.loads(example.read_text())
        main_sheet = next(sh for sh in book["sheets"]
                          if sh["name"] == book.get("main"))
        out = next(n for n in main_sheet["nodes"] if n["type"] == "Output")
        assert out["params"]["task"] == "language_modeling", \
            f"GPT2's output is labelled {out['params']['task']}, which is wrong"


@check("a design with nothing to learn says so")
def _():
    """An activation on its own is a fine layer and a useless model. torch's
    own message is "optimizer got an empty parameter list", which says what
    broke and not why — and the import of minGPT's NewGELU produces exactly
    this design."""
    if not HAVE_TORCH:
        print("        (torch absent, skipped)")
        return
    import train as T

    guard = inspect.getsource(T._run)
    assert "no trainable parameters" in guard, \
        "nothing checks whether there is anything to train"
    assert "requires_grad" in guard, "the check does not look at the parameters"

    # the message has to say what to do about it
    start = guard.index("no trainable parameters")
    message = guard[start: start + 600]
    for hint in ("Linear", "Conv2d", "Embedding"):
        assert hint in message, f"the message does not suggest {hint}"

    # and the exception raised has to be one that exists and is caught
    assert "raise DataError(" in guard[start - 200: start], \
        "raises an exception the module does not define"


@check("selecting a layer finds it in the generated file")
def _():
    """The Code tab shows the whole file, so a layer has to point at its line.

    With a workbook the file holds several classes, so the line number alone
    does not say where you have landed — hence the class name beside it.
    """
    assert "function classAround" in PAGE, "there is no way to name the class"
    assert 'id="codeWhere"' in PAGE, "nothing reports where the highlight went"

    inspector = PAGE[PAGE.index("function renderInspector"):]
    inspector = inspector[: inspector.index("\n}\n")]
    assert "highlightInCode" in inspector, \
        "selecting a layer does not move the open code panel"

    # and every layer with a constructor should be findable
    import json as _json

    import blockloader
    import codegen as cg
    import workbook as wb

    blockloader.load_all()
    example = ROOT / "examples" / "MicroLean.json"
    if not example.exists():
        print("        (MicroLean absent, skipped)")
        return
    book = _json.loads(example.read_text())
    analysis = wb.analyze(book)
    source = wb.to_pytorch(book, analysis)
    for sheet in book["sheets"]:
        node_code: dict = {}
        cg.to_pytorch(G.parse(wb.sheet_graph(book, sheet["name"])),
                      analysis["sheets"][sheet["name"]], node_code)
        for entry in node_code.values():
            if not entry.get("init"):
                continue                    # terminals have no constructor
            var = entry["var"]
            assert f"self.{var} =" in source, \
                f"{var} has a constructor but no line in the file"


@check("the code panel renders a real viewer, not a dump")
def _():
    for token in ("function highlightPython", "function drawCodeMap",
                  "tok-kw", "tok-str", "tok-com", "tok-def",
                  'id="codeGutter"', 'id="codeMap"', 'id="codeCount"'):
        assert token in PAGE, f"{token} is missing from the code panel"
    # tokenising before splitting lines is what keeps a docstring in one piece
    assert "Tokenise first" in PAGE or "PY_TOKEN.exec" in PAGE
    assert "escapeHtml" in PAGE, "code is rendered without escaping"


@check("shape follows flowchart convention, not decoration")
def _():
    for shape in ("circle", "diamond", "hex", "data", "stadium",
                  "predefined", "card"):
        assert f'shape: "{shape}"' in PAGE, f"nodeBox never returns {shape}"
    for cls in ("node-data", "node-stadium", "node-bars"):
        assert cls in PAGE, f"{cls} has no styling"
    # the reshaping layers are the ones that get the data symbol
    assert "RESHAPERS" in PAGE and "Flatten" in PAGE and "Permute" in PAGE


@check("the whole graph can be dragged, three ways")
def _():
    assert "function beginPan" in PAGE, "no pan implementation"
    assert "function panActive" in PAGE, "no pan predicate"
    assert 'id="zpan"' in PAGE, "no hand tool"
    assert "spaceHeld" in PAGE, "space does not pan"
    assert "button === 1" in PAGE, "the middle button does not pan"
    # the capturing listener is what stops a node moving instead of the canvas
    assert "}, true);" in PAGE, "the pan handler does not capture"


@check("the canvas can flow either way")
def _():
    assert 'flow === "horizontal"' in PAGE, "no horizontal layout branch"
    assert 'id="zflow"' in PAGE, "no control to switch orientation"
    assert 'id="zgrid"' in PAGE, "no control for the grid"


@check("nothing absolutely positioned lies across the side panels")
def _():
    """The status strip used to span the whole page.

    It is 30px tall and pinned to the bottom, so it sat on top of the last 30px
    of every docked panel — which buried the assistant's input box and the end
    of the palette. Anything pinned to the page edge has to live inside the
    canvas instead.
    """
    import re

    markup = PAGE[: PAGE.index("<script>")]
    page = markup[markup.index('id="pageDesign"'):markup.index('id="pageBuild"')]
    stage = page[page.index('id="stage"'):]
    stage = stage[: stage.index("</main>")]
    for pinned in ("statusbar", "problemPanel"):
        assert f'id="{pinned}"' in stage, \
            f"{pinned} is pinned to the page rather than the canvas, so it covers the panels"


@check("the assistant's input stays reachable")
def _():
    import re

    css = PAGE[PAGE.index("<style>") + 7: PAGE.index("</style>")]
    rules = {m.group(1).strip(): m.group(2)
             for m in re.finditer(r"(?:^|\n)([^\n{}]+)\{([^}]*)\}", css)}
    log = rules.get("#askLog", "")
    body = rules.get("#askBody", "")
    bar = rules.get(".askbar", "")
    assert "min-height:0" in log.replace(" ", ""), "#askLog cannot shrink, so it pushes the bar down"
    assert "flex:1" in body.replace(" ", ""), "#askBody does not fill the panel"
    assert "flex:none" in bar.replace(" ", ""), "the input bar is allowed to be squeezed away"


@check("a rail entry that is not a page cannot blank the screen")
def _():
    """showPage(undefined) leaves no page displayed at all.

    The rail's handler was bound to every button in it. The Layers and Files
    entries carry no data-page, so pressing one called showPage(undefined),
    which removed `on` from every page and added it to none — the dark body
    showing through where the canvas should be.
    """
    script = PAGE[PAGE.index("<script>"):]
    assert 'querySelectorAll("#rail button")' not in script, (
        "a rail handler is bound to every button, including the ones that open "
        "the palette rather than a page")

    # the entries that open the palette must not claim to be pages
    rail = PAGE[PAGE.index('<nav id="rail">'):PAGE.index("</nav>")]
    import re

    for match in re.finditer(r"<button([^>]*)>", rail):
        attrs = match.group(1)
        if "railpal" in attrs:
            assert "data-page" not in attrs, \
                "a palette entry also carries data-page, so it will switch pages"

    # and showPage must be given something real
    body = script[script.index("function showPage"):]
    body = body[: body.index("\n}\n")]
    assert "p.id === pageId" in body, "showPage no longer matches on the id"


@check("the layer palette opens from the rail")
def _():
    """One left column, not two.

    Layers and Files are entries in the rail like everything else, and the
    palette is the part of that column which expands. Pressing the entry already
    showing folds it away, so the chevron's promise holds both ways.
    """
    rail = PAGE[PAGE.index('<nav id="rail">'):PAGE.index("</nav>")]
    for which in ("layers", "files"):
        assert f'data-pal="{which}"' in rail, f"{which} is not in the rail"
    assert "railpal" in rail and "chev" in rail, "the entries carry no chevron"

    script = PAGE[PAGE.index("<script>"):]
    body = script[script.index("function showPalette"):]
    body = body[: body.index("\n}\n")]
    assert "toggleCollapse" in body, "pressing the open section does not fold it"
    assert "state.paletteTab" in body, "the section showing is not remembered"

    # and the rail keeps up when the layout changes some other way
    collapse = script[script.index("function applyCollapse"):]
    collapse = collapse[: collapse.index("\n}\n")]
    assert "markRailPalette()" in collapse, \
        "folding by another route leaves the rail entry looking open"


@check("one line divides a panel from the canvas")
def _():
    """There were three: the panel's border, the drag handle's hover bar, and
    the dark page background showing through the transparent splitter between
    them."""
    import re

    css = PAGE[PAGE.index("<style>"): PAGE.index("</style>")]
    rules = {m.group(1).strip(): m.group(2)
             for m in re.finditer(r"(?:^|\n)([^\n{}]+)\{([^}]*)\}", css)}

    for panel in ("#palette", "#inspector"):
        rule = rules.get(panel, "")
        assert "border-left" not in rule and "border-right" not in rule, \
            f"{panel} still draws its own edge beside the splitter"

    splitter = rules.get(".splitter", "")
    assert "background:transparent" not in splitter.replace(" ", ""), \
        "the splitter is transparent, so the page background shows through it"
    assert "var(--board)" in splitter, "the splitter does not match the canvas"

    # and it still has to be a handle, not just a line
    assert "col-resize" in splitter, "the splitter is no longer draggable"
    assert "row-resize" in rules.get(".splitter.horiz", ""), \
        "the horizontal splitter is no longer draggable"


@check("a folded panel leaves nothing behind")
def _():
    """The rail opens the panels, so a strip beside the canvas is a second
    control for a job already done — and it took up room in the state the
    fold exists to clear."""
    assert "panelstrip" not in PAGE, \
        "a folded panel still leaves a strip beside the canvas"

    for token in ('class="panelfold"', "function applyCollapse",
                  "function toggleCollapse", 'class="tabfold"'):
        assert token in PAGE, f"{token} is missing from the folding"

    # a tab has to be able to bring its panel back, or folding it strands you
    script = PAGE[PAGE.index("<script>"):]
    body = script[script.index("function showSide"):]
    body = body[: body.index("\n}\n")]
    assert "hidden.inspector = false" in body, \
        "asking for a tab does not reopen the panel it lives in"

    # the chevron has to point the way the panel will move, which depends on
    # which edge it is docked to
    script = PAGE[PAGE.index("function collapseGlyph"):]
    body = script[: script.index("\n}\n")]
    assert "dock[name]" in body and "hidden[name]" in body, \
        "the glyph ignores where the panel is docked or whether it is hidden"

    # a hidden panel must not hold the bottom row open
    layout = PAGE[PAGE.index("function applyLayout"):]
    layout = layout[: layout.index("\n}\n")]
    assert "hidden" in layout, "applyLayout ignores the hidden state"

    # and the fold control must not be treated as one of the tabs
    assert '.ptabs button")' not in PAGE, \
        "the tab handlers will pick up the fold chevron as a tab"


@check("docking, resizing and folding do not undo each other")
def _():
    """Folding is part of the layout, not a separate mode.

    applyLayout ended by redrawing without reapplying the fold state, so
    re-docking or resizing a folded panel silently brought it back.
    """
    script = PAGE[PAGE.index("<script>"):]
    layout = script[script.index("function applyLayout"):]
    layout = layout[: layout.index("\n}\n")]
    assert "applyCollapse()" in layout, \
        "applyLayout does not reapply the fold state, so docking will unfold a panel"

    collapse = script[script.index("function applyCollapse"):]
    collapse = collapse[: collapse.index("\n}\n")]
    for effect in ("style.display", "splitter", "bottomRow"):
        assert effect in collapse, f"folding does not deal with {effect}"


@check("nothing renders against a system-coloured default")
def _():
    """The scrollbar track was never styled, only the thumb.

    On a machine set to dark mode the browser's default track is nearly black,
    and it read as a heavy rule down the edge of the canvas — a control the app
    had never drawn.
    """
    css = PAGE[PAGE.index("<style>"): PAGE.index("</style>")]
    assert "color-scheme:light" in css.replace(" ", ""), \
        "the page can inherit dark form controls from the system"
    assert "::-webkit-scrollbar-track" in css, "the scrollbar track is unstyled"
    assert "scrollbar-color" in css, "Firefox scrollbars are unstyled"


@check("panels can be docked and resized")
def _():
    for token in ('id="mainRow"', 'id="bottomRow"', 'class="splitter"',
                  'data-dock="bottom"', "applyLayout", "dragSplitter"):
        assert token in PAGE, f"{token} is missing from the page"


@check("the workspace layout is stored on the server")
def _():
    import main

    prefs = main.prefs_file()
    saved = prefs.exists()
    backup = prefs.read_text() if saved else None
    try:
        prefs.unlink(missing_ok=True)
        assert main.get_prefs() == {}, "a missing file should read as empty"
        main.put_prefs({"dock": {"palette": "bottom"},
                        "sizes": {"inspector": 480}})
        back = main.get_prefs()
        assert back["dock"]["palette"] == "bottom", back
        assert back["sizes"]["inspector"] == 480, back

        prefs.write_text("{ this is not json")
        assert main.get_prefs() == {}, "a corrupt file should not raise"
    finally:
        prefs.unlink(missing_ok=True)
        if backup is not None:
            prefs.write_text(backup)


@check("a project can be brought in whole, in part, or step by step")
def _():
    for token in ("function importSteps", "function placeStep", "function startGuided",
                  "function renderGuide", 'id="btnAddAll"', 'id="btnAddPicked"',
                  'id="keepCanvas"', 'data-side="guide"'):
        assert token in PAGE, f"{token} is missing"
    # stepping and bulk import must build through the same routine, or the two
    # would drift and produce different graphs from the same plan
    assert PAGE.count("function placeStep") == 1
    guided = PAGE[PAGE.index("function applyStep"):]
    assert "placeStep(" in guided[:600], "the stepper no longer shares placeStep"


@check("no class name is styled as two different things")
def _():
    """A second CSS rule does not replace the first, it merges with it, which is
    worse than replacing.

    Two rules adding to each other is normal and fine. What is not fine is one
    rule constraining a name to a fixed height with the overflow hidden while
    another treats the same name as a container to lay things out in — that is
    what `.pickbar` was, a 13px progress bar and a panel, and the panel came out
    clipped to a single cut-off line while everything else about it looked
    correct.
    """
    import re
    from collections import defaultdict

    css = PAGE[PAGE.index("<style>"): PAGE.index("</style>")]
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    rules = defaultdict(list)
    for match in re.finditer(r"([^{}]+)\{([^}]*)\}", css):
        body = match.group(2).replace(" ", "").replace("\n", "")
        for selector in match.group(1).split(","):
            selector = selector.strip()
            if re.fullmatch(r"\.[\w-]+", selector):
                rules[selector].append(body)

    clashes = []
    for name, bodies in rules.items():
        if len(bodies) < 2:
            continue
        boxed = any(re.search(r"height:\d", b) and "overflow:hidden" in b
                    for b in bodies)
        container = any("display:flex" in b or "display:grid" in b
                        or "padding:" in b for b in bodies)
        if boxed and container:
            clashes.append(name)

    assert not clashes, (
        "these names are both a fixed clipped box and a container, so whichever "
        "is meant will come out clipped: " + ", ".join(sorted(clashes)))


@check("no two functions on the page share a name")
def _():
    """A later definition silently replaces an earlier one.

    Two functions called scanFolder — one scanning Python files for models, one
    scanning a folder of images for training — meant the Import dialog's Scan
    button ran the dataset scanner, read an empty field and complained that no
    path had been typed. Nothing was undefined, so the existing checks passed.
    """
    import re
    from collections import Counter

    script = PAGE[PAGE.index("<script>"):]
    names = re.findall(r"^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(",
                       script, re.M)
    clashes = {name: n for name, n in Counter(names).items() if n > 1}
    assert not clashes, (
        "these names are defined more than once, so only the last one runs: "
        + ", ".join(f"{name} ({n}\u00d7)" for name, n in sorted(clashes.items())))


@check("every function the page calls is actually defined")
def _():
    """Catches a whole section being deleted.

    Four releases shipped with the Build, Runs, Chat and Import sections gone:
    the markup was still there so the pages rendered, and nothing referenced a
    missing element, so every other check passed. Clicking those pages threw.
    """
    import re

    script = PAGE[PAGE.index("<script>"):]

    declared = set(re.findall(r"function\s+([A-Za-z_$][\w$]*)\s*\(", script))
    declared |= set(re.findall(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
                               r"(?:async\s*)?(?:function|\()", script))
    declared |= set(re.findall(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", script))

    # the entry point of every page and panel, by name
    entry_points = [
        "loadProjects", "renderProjectList", "openProject", "applyStep",
        "refreshRuns", "openRun", "drawRunChart",
        "refreshChatModels", "sendChat", "chatSay",
        "refreshBlocks", "openBlock", "saveBlock",
        "showPage", "showSide", "applyLayout", "loadLayout",
        "openInserter", "insertIntoEdge", "openAppender", "appendAfterNode",
        "importSteps", "placeStep", "startGuided", "renderGuide", "renderPlanOverview",
        "highlightPython", "drawCodeMap", "syncCodeMap", "escapeHtml", "testLayer",
        "classAround", "highlightInCode",
        "sendAsk", "askSay", "loadAssistant", "askObservations",
        "openQuickAdd", "renderQuickAdd", "chooseQuickAdd", "closeQuickAdd", "glyphFor",
        "checkDomain",
        "renderLaunchPad", "padFill", "padSearch", "assistantDock", "toggleDock",
        "renderDomainsPage", "domFill", "armCard", "placeDomain", "validateAllDomains",
        "findPapers", "paperRow", "fetchPaper", "showPaperPassages", "draftSpec",
        "loadStarters", "buildFromDomain",
        "dockWelcome", "dockSend", "dockAdvice", "openImportDialog",
        "renderStoryPanel", "loadStory", "paintStory", "stepStory", "playStory",
        "storyOpening", "storyStep", "storyClosing",
        "runPrecision", "measurePrecision",
        "renderRunPanel", "runTrace", "animateTrace", "traceStateOf", "runTable",
        "runTimeline", "runSummary", "formatBytes",
        "refreshAgents", "renderAgentForm", "startStudy", "openStudy", "openTrial",
        "switchSheet", "addSheet", "renameSheet", "deleteSheet", "renderSheetTabs",
        "ensureBook", "commitSheet", "openReferencedSheet", "scanFolder",
        "importFolderPicks", "scanCodeFolder", "fetchRepo", "renderScanResults", "peekAt",
        "tryClass", "scanRow", "modelFor", "paintPeek",
        "applyImportSize", "wireScanDrag", "resetImportSize",
        "applyCollapse", "toggleCollapse", "collapseGlyph", "showPalette",
        "markRailPalette",
        "loadProjectTree", "renderProjectTree",
        "openProjectFile", "importFromTree", "loadAccount", "showSignIn",
        "submitSignIn", "signOut", "renderMathPanel", "mathDiagram", "paintMath",
        "openPicker", "renderPickList", "openDesign",
        "diagramConv", "diagramActivation", "diagramAttention",
        "renderNetworkPanel", "renderNeedsPanel", "refreshVersions",
        "buildTrainForm", "startTraining", "refreshCheckpoints",
        "nodeBox", "wirePath", "portIn", "portOut",
    ]
    missing = [name for name in entry_points if name not in declared]
    assert not missing, f"the page calls these but nothing defines them: {missing}"


@check("each page's loader is wired to a function that exists")
def _():
    import re

    script = PAGE[PAGE.index("<script>"):]
    block = re.search(r"PAGE_SETUP\s*=\s*\{(.*?)\}", script, re.S)
    assert block, "PAGE_SETUP is gone; the rail would open empty pages"
    called = re.findall(r"=>\s*([A-Za-z_$][\w$]*)\s*\(", block.group(1))
    assert called, "PAGE_SETUP wires up nothing"
    declared = set(re.findall(r"(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", script))
    missing = [c for c in called if c not in declared]
    assert not missing, f"PAGE_SETUP calls undefined functions: {missing}"


print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    for name, why in FAILED:
        print(f"  {name}: {why}")
    sys.exit(1)
