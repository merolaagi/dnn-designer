"""Smoke tests. Run with `python tests/test_designer.py` or `pytest tests/`.

The tests that need torch skip themselves when it is absent, so the suite still
means something in a bare environment. The ones that matter most check that
generated code actually runs and that shapes predicted on the canvas match what
PyTorch produces — a designer whose predictions disagree with the framework is
worse than no designer.
"""

import json
import sys
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
        if "__pycache__" in str(path):
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
        assert auth.seed(home) == 0, "seeding must happen once, not on every sign-in"
        assert not list((home / "saved").glob("*.json"))

        # but asking for them back works
        assert auth.restore_examples(home) == len(shipped)

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
        "sendAsk", "askSay", "loadAssistant", "askObservations",
        "openQuickAdd", "renderQuickAdd", "chooseQuickAdd", "closeQuickAdd", "glyphFor",
        "refreshAgents", "renderAgentForm", "startStudy", "openStudy", "openTrial",
        "switchSheet", "addSheet", "renameSheet", "deleteSheet", "renderSheetTabs",
        "ensureBook", "commitSheet", "openReferencedSheet", "scanFolder",
        "importFolderPicks", "loadProjectTree", "renderProjectTree",
        "openProjectFile", "importFromTree", "loadAccount", "showSignIn",
        "submitSignIn", "signOut", "renderMathPanel", "mathDiagram", "paintMath",
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
