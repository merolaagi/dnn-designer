"""Smoke tests. Run with `python tests/test_designer.py` or `pytest tests/`.

The tests that need torch skip themselves when it is absent, so the suite still
means something in a bare environment. The ones that matter most check that
generated code actually runs and that shapes predicted on the canvas match what
PyTorch produces — a designer whose predictions disagree with the framework is
worse than no designer.
"""

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


print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    for name, why in FAILED:
        print(f"  {name}: {why}")
    sys.exit(1)
