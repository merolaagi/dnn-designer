"""Law-built layers: each guarantee checked numerically, on untrained layers.

Run with `python tests/test_laws.py`. The checks that need torch skip themselves
when it is absent. A guarantee that does not hold is a finding: the test says
which law failed and by how much, rather than loosening the tolerance.
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["AGENTLAB_NO_PUSH"] = "1"

import blockloader  # noqa: E402
import codegen  # noqa: E402
import graph as G  # noqa: E402
import lawsource  # noqa: E402

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

PASSED, FAILED = [], []


def check(name):
    def wrap(fn):
        if os.environ.get("ONLY") and os.environ["ONLY"] not in name:
            return fn
        try:
            fn()
            PASSED.append(name)
            print(f"  pass  {name}")
        except AssertionError as exc:
            FAILED.append((name, str(exc)))
            print(f"  FAIL  {name}: {exc}")
            if os.environ.get("TRACE"):
                import traceback
                traceback.print_exc()
        except Exception as exc:  # noqa: BLE001
            FAILED.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
            if os.environ.get("TRACE"):
                import traceback
                traceback.print_exc()
        return fn
    return wrap


def skip():
    print("        (torch absent, skipped)")


blockloader.load_all()
print(f"\nlaws: {len(lawsource.NAMES)} layers, torch {'available' if HAVE_TORCH else 'absent'}\n")

# each block, the canvas parameters it is built with, and two input shapes (batch left out)
CASES = {
    "LagrangianField": ({"hidden": 16, "eps": 1e-3, "form": "mechanical"}, [2], [4]),
    "LagrangianField:general": ({"hidden": 16, "eps": 1e-3, "form": "general"}, [2], [4]),
    "ConvexPotential": ({"hidden": 16, "layers": 3, "out": 1}, [5], [8]),
    "LipschitzLinear": ({"units": 7, "bound": 1.0}, [5], [12]),
    "InvariantPool": ({"hidden": 16, "out": 3, "pool": "sum"}, [6, 2], [11, 4]),
    "DaleLinear": ({"units": 9, "frac": 0.8}, [5], [10]),
    "HebbianFastWeights": ({"decay": 0.9, "rate": 0.5}, [5, 4], [9, 6]),
    "WorkingMemory": ({"slots": 4}, [3, 8], [300, 8]),
    "SetHamiltonian": ({"hidden": 16, "pairwise": True}, [3, 2], [5, 4]),
}


def ctor_kwargs(name, p, shape):
    """The block's own constructor call, evaluated: the canvas and the test build the same layer."""
    from layers import REGISTRY
    src = REGISTRY[name].torch_init(p, [shape])
    cls = lawsource.build(name)
    return eval(src, {name: cls})  # noqa: S307 - our own generated call


def graph_of(name, p, shape):
    return G.parse({"name": name, "nodes": [
        {"id": "i", "type": "Input", "params": {"shape": shape}, "y": 0},
        {"id": "c", "type": name, "params": p, "y": 100},
        {"id": "o", "type": "Output", "params": {"task": "regression"}, "y": 200}],
        "edges": [{"id": "e1", "source": "i", "target": "c", "port": 0}, {"id": "e2", "source": "c", "target": "o"}]})


@check("all seven law blocks load under their Laws categories, and SelfPaced is a recipe")
def _():
    from layers import REGISTRY
    for name in lawsource.NAMES:
        assert name in REGISTRY, name
        assert REGISTRY[name].category.startswith("Laws · "), (name, REGISTRY[name].category)
        assert REGISTRY[name].keras_call is None, f"{name} claims a Keras translation it does not have"
    import recipeloader
    import recipes_sdk
    assert not recipeloader.load_all() and "SelfPaced" in recipes_sdk.REGISTRY


@check("each block's real output shape is what the canvas predicts, for two input shapes")
def _():
    if not HAVE_TORCH:
        return skip()
    for case, (p, s1, s2) in CASES.items():
        name = case.split(":")[0]
        for shape in (s1, s2):
            g = graph_of(name, p, shape)
            rep = G.analyze(g)
            assert rep["ok"], (name, shape, rep["errors"][:1])
            want = rep["nodes"]["c"]["out_shape"]
            torch.manual_seed(0)
            layer = ctor_kwargs(name, p, shape)
            out = layer(torch.randn(3, *shape))
            assert list(out.shape[1:]) == want, f"{name} on {shape}: torch {list(out.shape[1:])}, canvas {want}"


@check("each block's parameter count is what lawsource.learnables says, and the canvas total agrees")
def _():
    if not HAVE_TORCH:
        return skip()
    import train as T
    for case, (p, s1, s2) in CASES.items():
        name = case.split(":")[0]
        for shape in (s1, s2):
            layer = ctor_kwargs(name, p, shape)
            real = sum(q.numel() for q in layer.parameters())
            g = graph_of(name, p, shape)
            rep = G.analyze(g)
            assert rep["nodes"]["c"]["learnables"] == real, \
                f"{name} on {shape}: canvas {rep['nodes']['c']['learnables']}, torch {real}"
            model = T.build_model(codegen.to_pytorch(g, rep), codegen.model_class_name(g))
            assert sum(q.numel() for q in model.parameters()) == rep["total_learnables"], name


@check("the generated PyTorch file for a one-block graph imports and runs, for every law")
def _():
    if not HAVE_TORCH:
        return skip()
    import train as T
    for case, (p, s1, _s2) in CASES.items():
        name = case.split(":")[0]
        g = graph_of(name, p, s1)
        rep = G.analyze(g)
        src = codegen.to_pytorch(g, rep)
        assert lawsource.SOURCES[name].strip().splitlines()[1] in src, f"{name}: the tested class is not the emitted one"
        model = T.build_model(src, codegen.model_class_name(g))
        model(torch.randn(2, *s1))


@check("InvariantPool: reordering the set's rows does not change the output (1e-5)")
def _():
    if not HAVE_TORCH:
        return skip()
    torch.manual_seed(1)
    for pool in ("sum", "mean", "max"):
        layer = lawsource.build("InvariantPool")(2, hidden=32, out=4, pool=pool)
        x = torch.randn(16, 7, 2)
        for _ in range(5):
            perm = torch.randperm(7)
            gap = (layer(x) - layer(x[:, perm])).abs().max().item()
            assert gap < 1e-5, f"{pool} pooling changed by {gap:.2e} under a reordering"


@check("LipschitzLinear: the weight actually applied never exceeds the bound, from the first call")
def _():
    if not HAVE_TORCH:
        return skip()
    Lip = lawsource.build("LipschitzLinear")
    for seed in range(20):
        torch.manual_seed(seed)
        for bound in (0.5, 1.0, 3.0):
            layer = Lip(13, 11, bound=bound)
            with torch.no_grad():
                layer.linear.weight.mul_(5.0)           # far over the bound, so the scaling has work to do
            for mode in ("train", "eval"):
                getattr(layer, mode)()
                W = layer.effective_weight().detach()
                top = torch.linalg.matrix_norm(W, ord=2).item()
                assert top <= bound * (1 + 1e-5), f"seed {seed}, {mode}: ‖W_eff‖₂ = {top:.4f} > bound {bound}"
            x, x2 = torch.randn(64, 13), torch.randn(64, 13)
            ratio = ((layer(x) - layer(x2)).norm(dim=1) / (x - x2).norm(dim=1)).max().item()
            assert ratio <= bound * (1 + 1e-5), f"seed {seed}: ‖f(x)−f(x′)‖/‖x−x′‖ = {ratio:.4f} > {bound}"


@check("LipschitzLinear: a weight already under the bound is left as it is")
def _():
    if not HAVE_TORCH:
        return skip()
    torch.manual_seed(0)
    layer = lawsource.build("LipschitzLinear")(6, 4, bound=10.0)
    assert torch.allclose(layer.effective_weight(), layer.linear.weight)


@check("DaleLinear: every column of the weight actually applied has one sign, and the E share is frac")
def _():
    if not HAVE_TORCH:
        return skip()
    torch.manual_seed(2)
    layer = lawsource.build("DaleLinear")(10, 7, frac=0.8)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(7, 10) * 4)      # any weights at all
    W = layer.effective_weight()
    for j in range(10):
        col = W[:, j]
        assert (col > 0).all() or (col < 0).all(), f"column {j} mixes signs: {col.tolist()}"
    assert int((W[0] > 0).sum()) == 8, "the first ⌈0.8·10⌉ = 8 inputs should be excitatory"
    torch.manual_seed(3)
    x = torch.randn(4, 10)
    assert torch.allclose(layer(x), x @ W.t() + layer.bias)


@check("ConvexPotential: f(½a+½b) ≤ ½f(a)+½f(b) on 1000 random pairs, for any weights")
def _():
    if not HAVE_TORCH:
        return skip()
    for seed in range(3):
        torch.manual_seed(seed)
        f = lawsource.build("ConvexPotential")(4, hidden=32, layers=3, out=1)
        with torch.no_grad():
            for prm in f.parameters():
                prm.add_(torch.randn_like(prm) * 2)      # scramble: the law must not depend on the init
        a, b = torch.randn(1000, 4) * 3, torch.randn(1000, 4) * 3
        with torch.no_grad():
            gap = (f((a + b) / 2) - (f(a) + f(b)) / 2).max().item()
        assert gap <= 1e-4, f"seed {seed}: midpoint above the chord by {gap:.2e}, so f is not convex"


@check("ConvexPotential starts at the scale of its input, so it can be trained at all")
def _():
    if not HAVE_TORCH:
        return skip()
    torch.manual_seed(9)
    f = lawsource.build("ConvexPotential")(4, hidden=64, layers=3, out=1)
    with torch.no_grad():
        scale = f(torch.rand(256, 4) * 6 - 3).abs().mean().item()
    assert scale < 20, f"untrained output averages {scale:.0f}: the non-negative weights start too large to train"


@check("WorkingMemory: exactly K vectors leave, whether the list is 3 items long or 300")
def _():
    if not HAVE_TORCH:
        return skip()
    torch.manual_seed(4)
    wm = lawsource.build("WorkingMemory")(8, slots=4)
    for L in (3, 300):
        assert tuple(wm(torch.randn(2, L, 8)).shape) == (2, 4, 8), L


@check("HebbianFastWeights: the fast matrix is written only by the Hebbian rule, never by the optimizer")
def _():
    if not HAVE_TORCH:
        return skip()
    torch.manual_seed(5)
    layer = lawsource.build("HebbianFastWeights")(4, decay=0.9, rate=0.5)
    assert [n for n, _ in layer.named_parameters()] == ["slow.weight", "slow.bias"], "A must not be a parameter"
    x = torch.randn(2, 6, 4)
    out, trace = layer(x, return_fast=True)
    assert all(not A.requires_grad for A in trace), "a fast matrix carries a gradient path back to the weights"
    # each A is the rule applied to the step's own output: A_{t+1} = λA_t + η h_t h_tᵀ
    A = torch.zeros(2, 4, 4)
    for t in range(6):
        assert torch.allclose(trace[t], A, atol=1e-6), f"step {t}: A is not what the Hebbian rule wrote"
        h = torch.nn.functional.layer_norm(out[:, t].detach(), (4,))
        A = 0.9 * A + 0.5 * h.unsqueeze(-1) @ h.unsqueeze(-2)
    # and the gradient the optimizer sees is the one through the slow weights only:
    # rebuild the forward with every A frozen to the trace, and the gradients must match exactly
    out.sum().backward()
    g1 = layer.slow.weight.grad.clone()
    layer.zero_grad()
    hs = []
    for t in range(6):
        xt = x[:, t]
        hs.append(torch.tanh(layer.slow(xt) + torch.bmm(trace[t], xt.unsqueeze(-1)).squeeze(-1)))
    torch.stack(hs, 1).sum().backward()
    assert torch.allclose(g1, layer.slow.weight.grad, atol=1e-6), "gradient flowed through the fast weights"


@check("LagrangianField: with L = ½q̇² − (1 − cos q) set by hand, Euler–Lagrange gives q̈ = −sin q (1e-3)")
def _():
    if not HAVE_TORCH:
        return skip()
    import torch.nn as nn

    class Pendulum(nn.Module):
        def forward(self, z):                         # z = (q, q̇)
            q, qd = z[:, :1], z[:, 1:]
            return 0.5 * qd ** 2 - (1 - torch.cos(q))

    layer = lawsource.build("LagrangianField")(2, hidden=8, eps=0.0, form="general")
    layer.lagrangian = Pendulum()
    torch.manual_seed(6)
    z = torch.cat([torch.rand(50, 1) * 6 - 3, torch.randn(50, 1)], dim=1)
    out = layer(z).detach()
    assert torch.allclose(out[:, :1], z[:, 1:]), "the first half must be q̇ itself"
    err = (out[:, 1:] + torch.sin(z[:, :1])).abs().max().item()
    assert err < 1e-3, f"q̈ differs from −sin q by {err:.2e}"
    # two coupled coordinates: L = ½ q̇ᵀ M q̇ − ½ qᵀ K q gives q̈ = −M⁻¹ K q
    M = torch.tensor([[2.0, 0.3], [0.3, 1.0]])
    K = torch.tensor([[1.0, -0.2], [-0.2, 0.5]])

    class Coupled(nn.Module):
        def forward(self, z):
            q, qd = z[:, :2], z[:, 2:]
            return 0.5 * ((qd @ M) * qd).sum(1, keepdim=True) - 0.5 * ((q @ K) * q).sum(1, keepdim=True)

    layer = lawsource.build("LagrangianField")(4, hidden=8, eps=0.0, form="general")
    layer.lagrangian = Coupled()
    z = torch.randn(30, 4)
    want = -(torch.linalg.solve(M, K) @ z[:, :2].t()).t()
    err = (layer(z)[:, 2:].detach() - want).abs().max().item()
    assert err < 1e-4, f"coupled: q̈ differs from −M⁻¹Kq by {err:.2e}"


@check("LagrangianField: the mechanical form's mass is positive definite and dE/dt = 0, for any weights")
def _():
    if not HAVE_TORCH:
        return skip()
    for seed in range(3):
        torch.manual_seed(seed)
        layer = lawsource.build("LagrangianField")(4, hidden=16, eps=1e-3).double()   # the property, not float32's rounding
        with torch.no_grad():
            for prm in layer.parameters():
                prm.add_(torch.randn_like(prm))            # any weights
        z = torch.randn(64, 4, dtype=torch.float64) * 2
        zz = z.clone().requires_grad_(True)
        g = torch.autograd.grad(layer.lagrangian(zz).sum(), zz, create_graph=True)[0]
        Hm = torch.stack([torch.autograd.grad(g[:, 2 + i].sum(), zz, retain_graph=True)[0][:, 2:] for i in range(2)], 1)
        assert torch.linalg.eigvalsh(Hm).min().item() > 0, f"seed {seed}: a mass matrix is not positive definite"

        # the law in continuous time: dE/dt = ∇E · ż = 0 at every state, with E = q̇·∂L/∂q̇ − L
        zz = z.clone().requires_grad_(True)
        Lz = layer.lagrangian(zz)
        dL = torch.autograd.grad(Lz.sum(), zz, create_graph=True)[0]
        E = (zz[:, 2:] * dL[:, 2:]).sum(1) - Lz[:, 0]
        gradE = torch.autograd.grad(E.sum(), zz)[0]
        zdot = layer(z).detach()
        rate = (gradE * zdot).sum(1).abs() / ((gradE.norm(dim=1) * zdot.norm(dim=1)) + 1e-12)
        assert rate.max().item() < 1e-6, f"seed {seed}: dE/dt is {rate.max().item():.2e} of |∇E||ż|, not zero"


@check("SetHamiltonian, untrained: energy is unchanged by relabelling, and the motion relabels with the input (1e-5)")
def _():
    if not HAVE_TORCH:
        return skip()
    for seed, pairwise in ((0, True), (1, False)):
        torch.manual_seed(seed)
        layer = lawsource.build("SetHamiltonian")(4, hidden=16, pairwise=pairwise)
        Z = torch.randn(8, 5, 4)
        for _ in range(5):
            perm = torch.randperm(5)
            e_gap = (layer.energy(Z) - layer.energy(Z[:, perm])).abs().max().item()
            assert e_gap < 1e-5, f"pairwise={pairwise}: energy changed by {e_gap:.2e} under a relabelling"
            m_gap = (layer(Z[:, perm]) - layer(Z)[:, perm]).abs().max().item()
            assert m_gap < 1e-5, f"pairwise={pairwise}: the motion of a relabelled set is off by {m_gap:.2e}"


@check("SetHamiltonian, untrained: its own energy drifts by < 1e-3 over a 600-step RK4 rollout")
def _():
    if not HAVE_TORCH:
        return skip()
    for seed in range(2):
        torch.manual_seed(seed)
        layer = lawsource.build("SetHamiltonian")(2, hidden=32, pairwise=True)
        z = torch.randn(1, 3, 2) * 0.5
        with torch.no_grad():
            e0 = layer.energy(z).item()
        dt = 0.01
        for _ in range(600):
            k1 = layer(z).detach(); k2 = layer(z + dt / 2 * k1).detach()
            k3 = layer(z + dt / 2 * k2).detach(); k4 = layer(z + dt * k3).detach()
            z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        with torch.no_grad():
            drift = abs(layer.energy(z).item() - e0) / max(abs(e0), 1e-6)
        assert drift < 1e-3, f"seed {seed}: energy drifted by {drift:.2e} of its value over 600 steps"


@check("SetHamiltonian: the real parameter count is lawsource.learnables, and the shape is what the canvas says")
def _():
    if not HAVE_TORCH:
        return skip()
    for pairwise in (True, False):
        for shape in ([3, 2], [5, 4]):
            p = {"hidden": 24, "pairwise": pairwise}
            layer = ctor_kwargs("SetHamiltonian", p, shape)
            real = sum(q.numel() for q in layer.parameters())
            want = lawsource.learnables("SetHamiltonian", dim=shape[-1], hidden=24, pairwise=pairwise)
            assert real == want, f"{shape}, pairwise={pairwise}: torch {real}, learnables {want}"
            g = graph_of("SetHamiltonian", p, shape)
            rep = G.analyze(g)
            assert rep["ok"] and rep["nodes"]["c"]["learnables"] == real, rep["errors"][:1]
            out = layer(torch.randn(2, *shape))
            assert list(out.shape[1:]) == rep["nodes"]["c"]["out_shape"] == shape, (shape, list(out.shape))
    try:
        ctor_kwargs("SetHamiltonian", {"hidden": 8, "pairwise": True}, [3, 3])
        raise AssertionError("an odd particle width is refused")
    except ValueError:
        pass


@check("SelfPaced admits only examples under the threshold, and the threshold rises each epoch")
def _():
    if not HAVE_TORCH:
        return skip()
    import types
    import recipes_sdk
    rec = recipes_sdk.REGISTRY["SelfPaced"]
    torch.manual_seed(8)
    model = torch.nn.Linear(3, 1)
    ctx = types.SimpleNamespace(model=model, optimizers={}, state={}, epoch=0,
                                cfg={"lr": 1e-2, "start": 0.5, "growth": 2.0},
                                parameters=lambda: model.parameters())
    rec.setup(ctx)
    x = torch.randn(64, 3)
    y = torch.cat([torch.zeros(32, 1), torch.full((32, 1), 50.0)])     # half easy, half hopeless at first
    r0 = rec.step(ctx, [x], y)
    assert r0["threshold"] == 0.5 and 0 < r0["admitted"] < 1, r0
    ctx.epoch = 1
    r1 = rec.step(ctx, [x], y)
    assert r1["threshold"] == 1.0, r1
    assert "loss" in rec.evaluate(ctx, [x], y)


@check("laws.py: every arm runs end to end at a tiny size and gives a verdict sentence")
def _():
    if not HAVE_TORCH:
        return skip()
    import laws
    assert set(laws.ARMS) >= {"physics", "math", "lagrangian", "biology", "psychology", "dale"}, sorted(laws.ARMS)
    assert set(laws.GUARANTEES) == {"convex", "lipschitz"} and not set(laws.ARMS) & set(laws.GUARANTEES)
    for name in laws.ARMS:
        res = laws.run_arm(name, [8, 16], seeds=1, steps=5, lr=1e-3, quiet=True)
        assert len(res["rows"]) == 2 and res["verdict"].endswith("."), (name, res)
        json.dumps(res)
    for name in laws.GUARANTEES:                                 # guarantees: one size, every measure, no verdict
        res = laws.run_guarantee(name, 16, seeds=1, steps=5, lr=1e-3, quiet=True)
        assert res["kind"] == "guarantee" and "verdict" not in res and res["rows"], (name, res)
        assert all(0 <= r["plain"] and 0 <= r["law"] for r in res["rows"])
        json.dumps(res)
    _, _, (xt, yt) = laws.ARMS["dale"]()["data"](8, 0)                 # a task any model can guess is no test
    share = torch.bincount(yt, minlength=3).float() / len(yt)
    assert share.max().item() < 0.6, f"the Dale task's labels are lopsided: {share.tolist()}"
    labels = [r["label"] for r in laws.run_guarantee("lipschitz", 16, 1, 5, 1e-3, quiet=True)["rows"]]
    assert labels == ["no nudge", "worst nudge of norm 0.3", "worst nudge of norm 1.0", "worst nudge of norm 3.0"]
    assert laws.overview()["guarantees"] and all(a["status"] != "shelved" for a in laws.overview()["arms"])


@check("a saved run the card cannot label is treated as not run, and every listed law has a measurement")
def _():
    import tempfile
    import agentlab as AL
    import laws
    real = AL._dir
    tmp = Path(tempfile.mkdtemp())
    AL._dir = lambda: tmp
    try:
        old = {"arm": "convex", "rows": [{"n": 8, "plain": 3.8, "law": 4.9}], "verdict": "an old data run"}
        laws.result_path("convex").write_text(json.dumps(old))              # a 4.35 data-efficiency run of convex
        assert laws.saved("convex") is None
        assert next(g for g in laws.overview()["guarantees"] if g["arm"] == "convex")["result"] is None
        partial = {"kind": "guarantee", "n": 512, "rows": [{"plain": 1.0, "law": 0.0}]}   # no labels
        laws.result_path("lipschitz").write_text(json.dumps(partial))
        assert laws.saved("lipschitz") is None
        good = {"kind": "guarantee", "n": 512, "rows": [{"label": "no nudge", "plain": 0.08, "law": 0.09}]}
        laws.result_path("lipschitz").write_text(json.dumps(good))
        assert laws.saved("lipschitz")["rows"][0]["label"] == "no nudge"
        laws.result_path("math").write_text(json.dumps(good))             # a guarantee file where a data run belongs
        assert laws.saved("math") is None
    finally:
        AL._dir = real
    page = (ROOT / "frontend" / "index.html").read_text()
    listed = re.findall(r'^  \["[a-z]+", "[^"]+", "[^"]+", "[^"]*", "([a-z]*)"\],$', page[page.index("const LAW_LIST"):], re.M)
    import laws as L
    measured = set(L.ARMS) | set(L.GUARANTEES)
    assert listed and "" in listed and all(a in measured or a == "" for a in listed)   # the page filters by arm
    assert "LAW_LIST.filter(x => measured.has(x[4]))" in page and 'fixed n = ' in page and "Not yet run" in page
    assert page.index('section("Composed"') < page.index('section("Data savers, confirmed"')
    assert "lawsDataCard(a, d, ghosts)" in page and "they are not the same task" in page
    import laws as L2
    assert L2.STATUS["compose"] == "composed" and next(a for a in L2.overview()["arms"] if a["arm"] == "compose")["status"] == "composed"


@check("a verdict says a loss is a loss, before the law's first win as well as after it")
def _():
    import laws
    rows = [{"n": 32, "plain": 1.08, "law": 1.61, "plain_spread": 0.004, "law_spread": 0.02},
            {"n": 512, "plain": 1.44, "law": 0.98, "plain_spread": 0.003, "law_spread": 0.002},
            {"n": 2048, "plain": 1.07, "law": 0.93, "plain_spread": 0.01, "law_spread": 0.01}]
    v = laws._verdict(rows)
    assert v.startswith("Below n=512 the plain model was ahead (at n=32"), v
    tie = [dict(rows[0], law=1.09)] + rows[1:]
    assert "within the noise" in laws._verdict(tie)
    assert laws.STATUS["biology"] == "shelved" and "biology" not in [a["arm"] for a in laws.overview()["arms"]]
    assert "biology" in laws.ARMS                              # shelved, not deleted: it can still be run by hand


@check("the Laws page's Run starts an arm in the background, streams its rows, and saves only a finished run")
def _():
    if not HAVE_TORCH:
        return skip()
    import tempfile
    import time
    import agentlab as AL
    import laws
    real = AL._dir
    tmp = Path(tempfile.mkdtemp())
    AL._dir = lambda: tmp
    try:
        ov = laws.overview()
        assert all(a["result"] is None for a in ov["arms"] + ov["guarantees"])    # nothing drawn before a run here
        try:
            laws.start("math", sizes=[1, 8])
            raise AssertionError("a size of 1 is refused")
        except ValueError:
            pass
        job = laws.start("math", sizes=[4, 8], seeds=1, steps=3)
        assert job["status"] == "running"
        for _ in range(200):
            if (laws.saved("math") or {}).get("status") == "done":
                break
            time.sleep(0.1)
        r = laws.saved("math")
        assert r and r["status"] == "done" and [row["n"] for row in r["rows"]] == [4, 8] and r["verdict"].endswith(".")
        mine = next(a for a in laws.overview()["arms"] if a["arm"] == "math")
        assert mine["result"]["rows"] == r["rows"]
        import main
        from fastapi.testclient import TestClient
        c = TestClient(main.app)
        assert c.get("/api/laws").status_code == 200
        assert c.post("/api/laws/astrology/run", json={}).status_code == 404
        assert c.post("/api/laws/math/run", json={"sizes": [99999]}).status_code == 400
    finally:
        AL._dir = real


print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    for name, why in FAILED:
        print(f"  {name}: {why}")
    sys.exit(1)
