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


@check("the canvas can flow either way")
def _():
    assert 'flow === "horizontal"' in PAGE, "no horizontal layout branch"
    assert 'id="zflow"' in PAGE, "no control to switch orientation"
    assert 'id="zgrid"' in PAGE, "no control for the grid"


@check("panels can be docked and resized")
def _():
    for token in ('id="mainRow"', 'id="bottomRow"', 'class="splitter"',
                  'data-dock="bottom"', "applyLayout", "dragSplitter"):
        assert token in PAGE, f"{token} is missing from the page"


@check("the workspace layout is stored on the server")
def _():
    import main

    saved = main.PREFS.exists()
    backup = main.PREFS.read_text() if saved else None
    try:
        main.PREFS.unlink(missing_ok=True)
        assert main.get_prefs() == {}, "a missing file should read as empty"
        main.put_prefs({"dock": {"palette": "bottom"},
                        "sizes": {"inspector": 480}})
        back = main.get_prefs()
        assert back["dock"]["palette"] == "bottom", back
        assert back["sizes"]["inspector"] == 480, back

        main.PREFS.write_text("{ this is not json")
        assert main.get_prefs() == {}, "a corrupt file should not raise"
    finally:
        main.PREFS.unlink(missing_ok=True)
        if backup is not None:
            main.PREFS.write_text(backup)


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
        "loadProjects", "renderProjectList", "openProject", "renderPlan", "applyStep",
        "refreshRuns", "openRun", "drawRunChart",
        "refreshChatModels", "sendChat", "chatSay",
        "refreshBlocks", "openBlock", "saveBlock",
        "showPage", "showSide", "applyLayout", "loadLayout",
        "openInserter", "insertIntoEdge", "openAppender", "appendAfterNode",
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
