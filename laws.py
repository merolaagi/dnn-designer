"""laws.py — how much data does a law save?

One question, asked the same way of every law we build a model on:

    Take a task the law describes. Train a plain network and a law-built
    network on N examples, for several N. For each, how small must N be
    before the model fails?

The answer is a *data-efficiency curve*: error against N, two lines. The gap
between them is what the law bought. A law whose line is not below the plain
one did not help on this task, and the script says so rather than hiding it.

Every arm registers itself in ARMS with four things: the task's data, a plain
model, a law model, and an error. The harness does the rest, so adding a law
is adding one function.

Run:  python laws.py                   every arm
      python laws.py physics           one arm (physics, math, ...)
      python laws.py physics --sizes 8 32 128 512 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Callable, Dict, List

ARMS: Dict[str, Callable] = {}
# options an arm may read, set from the command line with --set key=value (e.g. eps=1e-4 activation=tanh)
OPTIONS: Dict[str, Any] = {}


def arm(name):
    def wrap(fn):
        ARMS[name] = fn
        return fn
    return wrap


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------

def _fit(model, xs, ys, loss_fn, steps, lr):
    import torch
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model, xs, ys)
        loss.backward()
        opt.step()
    return float(loss.item())


def run_arm(name: str, sizes: List[int], seeds: int, steps: int, lr: float,
            quiet: bool = False, on_row: Callable = None, stop: Callable = None) -> Dict[str, Any]:
    import torch
    spec = ARMS[name]()
    rows = []
    for n in sizes:
        if stop and stop():
            break
        errs = {"plain": [], "law": []}
        for seed in range(seeds):
            torch.manual_seed(seed)
            xs, ys, test = spec["data"](n, seed)
            for kind in ("plain", "law"):
                torch.manual_seed(seed)
                model = spec[kind]()
                _fit(model, xs, ys, spec["loss"], steps, lr)
                errs[kind].append(spec["error"](model, test))
        row = {"n": n}
        for kind in errs:
            vals = errs[kind]
            row[kind] = sum(vals) / len(vals)
            row[kind + "_spread"] = (max(vals) - min(vals)) / 2
        row["ratio"] = row["plain"] / max(row["law"], 1e-12)
        rows.append(row)
        if on_row:
            on_row(row)
        if not quiet:
            print(f"  n={n:5d}   plain {row['plain']:.4f}   law {row['law']:.4f}"
                  f"   plain/law = {row['ratio']:.1f}x")
    return {"arm": name, "kind": "data", "law": spec["law_name"], "task": spec["task"],
            "params": {"plain": _count(spec["plain"]()), "law": _count(spec["law"]())},
            "measure": spec.get("measure", "error"), "sizes": sizes, "seeds": seeds, "steps": steps,
            "rows": rows, "verdict": _verdict(rows)}


def _count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _state(r) -> str:
    """ahead, behind, or tie: a difference inside the seeds' ranges is a tie, not a result."""
    lo_law, hi_law = r["law"] - r.get("law_spread", 0), r["law"] + r.get("law_spread", 0)
    lo_plain, hi_plain = r["plain"] - r.get("plain_spread", 0), r["plain"] + r.get("plain_spread", 0)
    if hi_law < lo_plain:
        return "ahead"
    if hi_plain < lo_law:
        return "behind"
    return "tie"


def _verdict(rows) -> str:
    """Say what the curve shows, in one sentence or two, without overclaiming.

    The question is where the law starts to win, whether it keeps winning, and how much more
    data the plain model needs to reach the law model's error there. A size where the seeds'
    ranges overlap is a tie; a size where both fail is not a loss for the law.
    """
    if not rows:
        return "No sizes were run."
    states = [_state(r) for r in rows]
    if "ahead" not in states:
        if "behind" in states:
            return "The plain model was better beyond the seed-to-seed spread; the law did not help on this task."
        return "The law did not reduce error beyond the seed-to-seed spread at any data size on this task."
    first = states.index("ahead")
    if "behind" in states[first:]:
        n = rows[first + states[first:].index("behind")]["n"]
        return f"The law was ahead at n={rows[first]['n']} but behind at n={n}; it is not a data saver here."
    at = rows[first]
    ties = [r["n"] for r, st in zip(rows[first:], states[first:]) if st == "tie"]
    if first == len(rows) - 1 and len(rows) > 1:
        return (f"Only at the largest size, n={at['n']}, is the law model ahead ({at['law']:.3g} against "
                f"{at['plain']:.3g}); that is not enough to call it a data saving. Try larger sizes.")
    before = states[:first]
    if not before:
        lead = ""
    elif "behind" in before:
        lost = rows[before.index("behind")]
        lead = (f"Below n={at['n']} the plain model was ahead (at n={lost['n']}: plain {lost['plain']:.3g}, "
                f"law {lost['law']:.3g}). ")
    else:
        lead = (f"Below n={at['n']} the two are within the noise of each other (at n={rows[0]['n']}: "
                f"plain {rows[0]['plain']:.3g}, law {rows[0]['law']:.3g}). ")
    tail = (f" At n={', '.join(map(str, ties))} the seeds' ranges overlap, so the lead there is within the noise."
            if ties else "")
    matched = next((r["n"] for r in rows if r["plain"] <= at["law"]), None)
    if matched:
        return (lead + f"From n={at['n']} the law model is ahead; the plain model needs n={matched} to reach its "
                f"error there: about {matched / at['n']:.0f}x less data." + tail)
    return (lead + f"From n={at['n']} the law model is ahead, and the plain model does not reach its error there "
            f"({at['law']:.3g}) even at n={rows[-1]['n']}: more than {rows[-1]['n'] / at['n']:.0f}x less data." + tail)


# ---------------------------------------------------------------------------
# arm 1 — physics: energy conservation (Hamiltonian field vs plain MLP)
# ---------------------------------------------------------------------------

@arm("physics")
def physics_arm():
    """A pendulum. The data are (state, velocity) pairs; the error is how far
    a long rollout drifts from the truth. The plain MLP learns dz/dt directly.
    The Hamiltonian layer learns one scalar H and reads dz/dt off its gradient,
    so energy is conserved for every value of the weights."""
    import torch
    import torch.nn as nn
    from conservation import _rk4, _truth, layer_class

    Hamiltonian = layer_class()
    field, energy = _truth(damping=0.0)
    dim, hidden = 2, 64

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        q = (torch.rand(n, 1, generator=g) * 2 - 1) * 2.5     # angles up to ~143°
        p = (torch.rand(n, 1, generator=g) * 2 - 1) * 1.5
        z = torch.cat([q, p], dim=-1)
        y = field(z)
        # test: one long rollout from a fixed start, 60 time units
        z0 = torch.tensor([[1.5, 0.0]])
        truth = _rk4(field, z0, 0.05, 1200)
        return z, y, truth

    def plain():
        return nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                             nn.Linear(hidden, hidden), nn.Tanh(),
                             nn.Linear(hidden, dim))

    def law():
        return Hamiltonian(dim, hidden=hidden, mode="field")

    def loss(model, xs, ys):
        return ((model(xs) - ys) ** 2).mean()

    def error(model, truth):
        with torch.enable_grad():   # Hamiltonian field differentiates inside forward
            path = _rk4(lambda z: model(z), truth[0], 0.05, len(truth) - 1).detach()
        return float(((path - truth) ** 2).mean().sqrt())

    return {"law": law, "plain": plain, "data": data, "loss": loss, "error": error,
            "task": "pendulum rollout RMSE over 60 time units",
            "law_name": "energy conservation (Hamiltonian field)"}


# ---------------------------------------------------------------------------
# arm 2 — mathematics: permutation invariance (Deep Sets vs flat MLP)
# ---------------------------------------------------------------------------

@arm("math")
def math_arm():
    """A set of 6 points in the plane; the target is a symmetric function of
    the set, Σᵢ sin(xᵢ)·cos(yᵢ). A flat MLP sees the 12 numbers in one order
    and must learn from data that order is irrelevant. InvariantPool cannot
    see order at all, so that lesson is free."""
    import torch
    import torch.nn as nn
    from lawsource import build

    InvariantPool = build("InvariantPool")
    N, D = 6, 2

    def target(x):                       # x: [B, N, D]
        return (torch.sin(x[..., 0]) * torch.cos(x[..., 1])).sum(-1, keepdim=True)

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.rand(n, N, D, generator=g) * 4 - 2
        xt = torch.rand(2000, N, D, generator=g) * 4 - 2
        return x, target(x), (xt, target(xt))

    class Flat(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(N * D, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(self, x):
            return self.net(x.flatten(1))

    def plain():
        return Flat()

    def law():
        return InvariantPool(D, hidden=32, out=1)

    def loss(model, xs, ys):
        return ((model(xs) - ys) ** 2).mean()

    def error(model, test):
        xt, yt = test
        with torch.no_grad():
            return float(((model(xt) - yt) ** 2).mean().sqrt())

    return {"law": law, "plain": plain, "data": data, "loss": loss, "error": error,
            "task": "symmetric function of a 6-point set, RMSE on 2000 fresh sets",
            "law_name": "permutation invariance (Deep Sets)"}


# ---------------------------------------------------------------------------
# arm 1b — physics again: the same pendulum, through LagrangianField
# ---------------------------------------------------------------------------

@arm("lagrangian")
def lagrangian_arm():
    """The physics arm's pendulum, with the law written as Euler–Lagrange instead of Hamilton.
    With unit mass the momentum is the velocity, so the same (q, q̇) data serve; the law model
    learns one scalar L(q, q̇) and the motion is derived from it."""
    spec = physics_arm()
    LagrangianField = __import__("lawsource").build("LagrangianField")
    eps, act = float(OPTIONS.get("eps", 1e-3)), str(OPTIONS.get("activation", "tanh"))
    spec["law"] = lambda: LagrangianField(2, hidden=64, eps=eps, activation=act)
    spec["law_name"] = "Euler–Lagrange (Lagrangian field)"
    return spec


# ---------------------------------------------------------------------------
# arm 3 — biology: Hebbian fast weights vs a plain GRU, on recalling a pattern from a cue
# ---------------------------------------------------------------------------

@arm("biology")
def biology_arm():
    """Three random ±1 patterns are shown, then a cue: one of them with a quarter of its signs
    flipped. The answer is the clean pattern. A Hebbian memory stores what it sees in its fast
    synapses by the local rule and completes the cue from them; the GRU has to learn to store
    and compare from examples. Error is the RMSE against the clean pattern on fresh sequences."""
    import torch
    import torch.nn as nn
    from lawsource import build

    Hebbian = build("HebbianFastWeights")
    D, K = 32, 3

    def make(n, g):
        pats = torch.randint(0, 2, (n, K, D), generator=g).float() * 2 - 1
        pick = torch.randint(0, K, (n,), generator=g)
        target = pats[torch.arange(n), pick]
        flip = torch.rand(n, D, generator=g) < 0.25
        cue = torch.where(flip, -target, target)
        return torch.cat([pats, cue.unsqueeze(1)], dim=1), target

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        x, y = make(n, g)
        return x, y, make(2000, g)

    class Law(nn.Module):
        def __init__(self):
            super().__init__()
            self.mem = Hebbian(D, decay=0.95, rate=0.5)
            self.read = nn.Linear(D, D)
        def forward(self, x):
            return self.read(self.mem(x)[:, -1])

    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(D, 48, batch_first=True)
            self.read = nn.Linear(48, D)
        def forward(self, x):
            return self.read(self.rnn(x)[0][:, -1])

    def loss(model, xs, ys):
        return ((model(xs) - ys) ** 2).mean()

    def error(model, test):
        xt, yt = test
        with torch.no_grad():
            return float(((model(xt) - yt) ** 2).mean().sqrt())

    return {"law": Law, "plain": Plain, "data": data, "loss": loss, "error": error,
            "task": "recall a stored ±1 pattern from a 25%-corrupted cue, RMSE on 2000 fresh sequences",
            "law_name": "Hebbian locality (fast weights)"}


# ---------------------------------------------------------------------------
# arm 3b — biology: Dale's law, on tabular data made by a network that obeys it
# ---------------------------------------------------------------------------

@arm("dale")
def dale_arm():
    """Tabular classification, 20 features into 3 classes, labelled by a teacher network that obeys Dale's
    law: its first 16 inputs only excite and the last 4 only inhibit, and likewise its hidden units, with 5%
    of labels flipped. The law-built MLP (DaleLinear → ReLU → DaleLinear) is told which inputs are excitatory,
    as a neuroscientist would be; the plain MLP of the same size must learn every weight's sign from data.
    Error is the share of 2000 fresh rows classified wrongly."""
    import math as _m
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from lawsource import build

    Dale = build("DaleLinear")
    D, H, C = 20, 32, 3

    def signs(n, frac=0.8):
        s = torch.ones(n)
        s[int(_m.ceil(frac * n)):] = -1.0
        return s

    g0 = torch.Generator().manual_seed(1234)               # one fixed teacher for every seed and size
    W1 = torch.rand(16, D, generator=g0) * signs(D)
    b1 = -torch.rand(16, generator=g0) * 2
    W2 = torch.rand(C, 16, generator=g0) * signs(16)

    def raw(x):
        return F.relu(x @ W1.t() + b1) @ W2.t()

    # standardise each class's score over the input range, so the three classes come out roughly balanced
    # (left raw, one class won almost every row and any model scored near zero by guessing it)
    ref = raw(torch.rand(20000, D, generator=g0) * 2)
    mu, sd = ref.mean(0), ref.std(0) + 1e-6

    def teacher(x):
        return (raw(x) - mu) / sd

    def make(n, g):
        x = torch.rand(n, D, generator=g) * 2
        y = teacher(x).argmax(1)
        flip = torch.rand(n, generator=g) < 0.05
        y = torch.where(flip, torch.randint(0, C, (n,), generator=g), y)
        return x, y

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        x, y = make(n, g)
        xt = torch.rand(2000, D, generator=g) * 2
        return x, y, (xt, teacher(xt).argmax(1))

    def loss(model, xs, ys):
        return F.cross_entropy(model(xs), ys)

    def error(model, test):
        xt, yt = test
        with torch.no_grad():
            return float((model(xt).argmax(1) != yt).float().mean())

    return {"law": lambda: nn.Sequential(Dale(D, H, frac=0.8), nn.ReLU(), Dale(H, C, frac=0.8)),
            "plain": lambda: nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, C)),
            "data": data, "loss": loss, "error": error,
            "task": "3-class tabular data labelled by a Dale's-law teacher (5% label noise), error rate on 2000 fresh rows",
            "law_name": "Dale's law (excitatory and inhibitory neurons)", "measure": "error rate"}


# ---------------------------------------------------------------------------
# arm 4 — psychology: a working-memory bottleneck vs averaging, on finding one item in a long list
# ---------------------------------------------------------------------------

@arm("psychology")
def psychology_arm():
    """A list of 40 noise items with one of four prototype items hidden at a random place; the class
    is which prototype. Averaging the list dilutes that one item forty-fold; four attention slots
    can hold it. The baseline is what the brief names: mean-pool, then a classifier. Error is the
    share of fresh lists classified wrongly."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from lawsource import build

    WorkingMemory = build("WorkingMemory")
    L, D, C = 40, 8, 4

    def make(n, g, protos):
        x = torch.randn(n, L, D, generator=g) * 0.7
        cls = torch.randint(0, C, (n,), generator=g)
        where = torch.randint(0, L, (n,), generator=g)
        x[torch.arange(n), where] = protos[cls]
        return x, cls

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        protos = F.normalize(torch.randn(C, D, generator=g), dim=1) * 3
        x, y = make(n, g, protos)
        return x, y, make(2000, g, protos)

    class Law(nn.Module):
        def __init__(self):
            super().__init__()
            self.wm = WorkingMemory(D, slots=4)
            self.head = nn.Linear(4 * D, C)
        def forward(self, x):
            return self.head(self.wm(x).flatten(1))

    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, C))
        def forward(self, x):
            return self.head(x.mean(dim=1))

    def loss(model, xs, ys):
        return F.cross_entropy(model(xs), ys)

    def error(model, test):
        xt, yt = test
        with torch.no_grad():
            return float((model(xt).argmax(1) != yt).float().mean())

    return {"law": Law, "plain": Plain, "data": data, "loss": loss, "error": error,
            "task": "which of 4 prototypes is hidden in a list of 40, error rate on 2000 fresh lists",
            "law_name": "capacity limit (working memory, K = 4)", "measure": "error rate"}


# ---------------------------------------------------------------------------
# composed — two confirmed laws in one layer (SetHamiltonian: energy + permutation)
# ---------------------------------------------------------------------------

@arm("compose")
def compose_arm():
    """Three identical particles on a line, coupled by springs and each in an
    anharmonic well. The plain MLP sees the six numbers flattened, in one
    order. SetHamiltonian sees a set and derives motion from one energy, so
    it gets conservation *and* indifference to particle order for free. This
    is the first model built on two laws at once; the gap should exceed
    either law alone."""
    import torch
    import torch.nn as nn
    from conservation import _rk4
    from lawsource import build

    SetHamiltonian = build("SetHamiltonian")
    N, k = 3, 1.0

    def field(z):                          # z: [B, N, 2]
        q, p = z[..., 0], z[..., 1]
        dq = p
        coupling = k * (N * q - q.sum(-1, keepdim=True))     # Σⱼ k (qᵢ − qⱼ)
        dp = -coupling - q ** 3
        return torch.stack([dq, dp], dim=-1)

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        z = (torch.rand(n, N, 2, generator=g) * 2 - 1) * torch.tensor([1.5, 1.0])
        z0 = torch.tensor([[[1.0, 0.0], [-0.5, 0.3], [0.2, -0.4]]])
        truth = _rk4(field, z0, 0.05, 1200)
        return z, field(z), truth

    class Flat(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2 * N, 64), nn.Tanh(),
                                     nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2 * N))
        def forward(self, z):
            return self.net(z.flatten(-2)).reshape(*z.shape[:-2], N, 2)

    def plain():
        return Flat()

    def law():
        return SetHamiltonian(2, hidden=48, pairwise=True)

    def loss(model, xs, ys):
        return ((model(xs) - ys) ** 2).mean()

    def error(model, truth):
        with torch.enable_grad():
            path = _rk4(lambda z: model(z), truth[0], 0.05, len(truth) - 1).detach()
        return float(((path - truth) ** 2).mean().sqrt())

    return {"law": law, "plain": plain, "data": data, "loss": loss, "error": error,
            "task": "3 coupled particles, rollout RMSE over 60 time units",
            "law_name": "energy conservation + permutation invariance (SetHamiltonian)"}


# ---------------------------------------------------------------------------
# guarantees, measured: not "does it save data" but "does the guarantee buy what it promises"
# ---------------------------------------------------------------------------
#
# Convexity and a Lipschitz bound are not data savers on their tasks (the plain network did as
# well or better with data). What they promise is different: one minimum wherever you start, and a
# bounded change under a bounded nudge. These measure exactly that, plain against law, at a fixed
# amount of data, with no data-efficiency verdict.

GUARANTEES: Dict[str, Callable] = {}


def guarantee(name):
    def wrap(fn):
        GUARANTEES[name] = fn
        return fn
    return wrap


def run_guarantee(name: str, n: int = 512, seeds: int = 2, steps: int = 1500, lr: float = 3e-3,
                  quiet: bool = False, on_row: Callable = None, stop: Callable = None) -> Dict[str, Any]:
    """Fit plain and law on n examples, per seed, then take every measure of the guarantee."""
    import torch
    spec = GUARANTEES[name]()
    per = {label: {"plain": [], "law": []} for label, _ in spec["measures"]}
    for seed in range(seeds):
        if stop and stop():
            break
        torch.manual_seed(seed)
        xs, ys, extra = spec["data"](n, seed)
        for kind in ("plain", "law"):
            torch.manual_seed(seed)
            model = spec[kind]()
            _fit(model, xs, ys, spec["loss"], steps, lr)
            for label, measure in spec["measures"]:
                per[label][kind].append(measure(model, extra))
    rows = []
    betters = spec["better"] if isinstance(spec["better"], list) else [spec["better"]] * len(spec["measures"])
    for (label, _), better in zip(spec["measures"], betters):
        row = {"label": label, "better": better}
        for kind in ("plain", "law"):
            vals = per[label][kind]
            row[kind] = sum(vals) / max(1, len(vals))
            row[kind + "_spread"] = (max(vals) - min(vals)) / 2 if vals else 0.0
        rows.append(row)
        if on_row:
            on_row(row)
        if not quiet:
            print(f"  {label:44s}  plain {row['plain']:.4f} ±{row['plain_spread']:.4f}   "
                  f"law {row['law']:.4f} ±{row['law_spread']:.4f}")
    return {"arm": name, "kind": "guarantee", "law": spec["law_name"], "task": spec["task"], "n": n,
            "seeds": seeds, "steps": steps, "better": spec["better"], "unit": spec["unit"], "rows": rows}


@guarantee("convex")
def convex_guarantee():
    """A convex energy in 4-D, sampled at n points inside the box [-3, 3]⁴. Each model is fitted, then
    gradient descent runs on it from 100 starts *outside* that box (every start has some coordinate
    beyond ±3, out to ±6), where there was no data. A convex fit has one minimum, wherever you start;
    a plain network is free to have wells, ridges or slopes out there. Measured: the share of the 100
    descents that end within 0.25 of the true minimum."""
    import torch
    import torch.nn as nn
    from lawsource import build

    Convex = build("ConvexPotential")
    D = 4
    A = torch.tensor([[2.0, 0.3, 0.0, 0.1], [0.3, 1.0, 0.2, 0.0], [0.0, 0.2, 1.5, 0.3], [0.1, 0.0, 0.3, 0.8]])
    c = torch.tensor([0.6, -0.4, 0.3, -0.7])
    B = torch.tensor([[1.0, -1.0, 0.5, 0.0], [-0.5, 0.0, 1.0, 1.0]])

    def energy(x):
        d = x - c
        return 0.5 * ((d @ A) * d).sum(-1, keepdim=True) + torch.logsumexp(x @ B.t(), dim=-1, keepdim=True)

    x = torch.zeros(1, D, requires_grad=True)
    opt = torch.optim.LBFGS([x], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        e = energy(x).sum()
        e.backward()
        return e
    opt.step(closure)
    x_star = x.detach()

    def far_starts(g, count=100):
        out = []
        while len(out) < count:
            p = torch.rand(D, generator=g) * 12 - 6
            if p.abs().max() > 3:
                out.append(p)
        return torch.stack(out)

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.rand(n, D, generator=g) * 6 - 3
        return x, energy(x), far_starts(g)

    ends: Dict[str, Any] = {}

    def descend(model, starts):
        # one descent per model, read by every measure; the model itself is the key (an id can be reused
        # once the previous model is freed, which would hand the law model the plain model's descents)
        if ends.get("model") is not model:
            z = starts.clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=0.05)
            for _ in range(600):
                opt.zero_grad()
                model(z).sum().backward()
                opt.step()
                with torch.no_grad():
                    z.clamp_(-6, 6)
            ends.update(model=model, dist=(z.detach() - x_star).norm(dim=1))
        return ends["dist"]

    def reached(model, starts):
        return float((descend(model, starts) < 0.25).float().mean())

    def distance(model, starts):
        return float(descend(model, starts).mean())

    return {"law": lambda: Convex(D, hidden=64, layers=2, out=1),
            "plain": lambda: nn.Sequential(nn.Linear(D, 64), nn.Softplus(), nn.Linear(64, 64), nn.Softplus(), nn.Linear(64, 1)),
            "data": data, "loss": lambda m, xs, ys: ((m(xs) - ys) ** 2).mean(),
            "measures": [("descents from outside the data that reach the minimum", reached),
                         ("mean distance of where they end from the minimum", distance)],
            "better": ["higher", "lower"], "unit": "share of 100 starts, then distance",
            "task": "fit a convex 4-D energy on n points in [-3, 3]⁴, then descend it from 100 starts outside that box",
            "law_name": "convexity (input-convex network)"}


@guarantee("lipschitz")
def lipschitz_guarantee():
    """A 1-Lipschitz target in 8-D, y = sin(u·x) with |u| = 1, fitted on n points. Each of 1000 test points
    is then nudged by the worst nudge of a given size that 10 steps of projected gradient ascent can find,
    against the error to the truth at the nudged point. A plain network can be steep between its data;
    a stack of 1-Lipschitz layers cannot change by more than the nudge. Measured: RMSE with no nudge, and
    under the worst nudge of norm 0.3, 1.0 and 3.0."""
    import torch
    import torch.nn as nn
    from lawsource import build

    Lip = build("LipschitzLinear")
    D = 8
    u = torch.nn.functional.normalize(torch.tensor([1.0, -0.5, 0.3, 0.8, -0.2, 0.6, 0.1, -0.7]), dim=0)

    def truth(x):
        return torch.sin(x @ u).unsqueeze(-1)

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, D, generator=g)
        return x, truth(x), torch.randn(1000, D, generator=g)

    def worst(eps):
        def measure(model, xt):
            delta = torch.zeros_like(xt)
            if eps > 0:
                for _ in range(10):
                    delta.requires_grad_(True)
                    x = xt + delta
                    ((model(x) - truth(x)) ** 2).sum().backward()
                    with torch.no_grad():
                        step = delta.grad / (delta.grad.norm(dim=1, keepdim=True) + 1e-12)
                        delta = delta + (eps / 4) * step
                        norm = delta.norm(dim=1, keepdim=True)
                        delta = delta * torch.clamp(eps / (norm + 1e-12), max=1.0)
            x = (xt + delta).detach()
            with torch.no_grad():
                return float(((model(x) - truth(x)) ** 2).mean().sqrt())
        return measure

    return {"law": lambda: nn.Sequential(Lip(D, 64, bound=1.0), nn.ReLU(), Lip(64, 64, bound=1.0), nn.ReLU(), Lip(64, 1, bound=1.0)),
            "plain": lambda: nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)),
            "data": data, "loss": lambda m, xs, ys: ((m(xs) - ys) ** 2).mean(),
            "measures": [("no nudge", worst(0.0)), ("worst nudge of norm 0.3", worst(0.3)),
                         ("worst nudge of norm 1.0", worst(1.0)), ("worst nudge of norm 3.0", worst(3.0))],
            "better": "lower", "unit": "RMSE against the truth",
            "task": "fit y = sin(u·x) in 8-D on n points; RMSE on 1000 test points under the worst nudge of each size",
            "law_name": "Lipschitz bound (1 per layer)"}


# ---------------------------------------------------------------------------
# running an arm in the background, for the Laws page: rows are written as they arrive
# ---------------------------------------------------------------------------

_JOBS: Dict[str, Dict[str, Any]] = {}
_STOPS: Dict[str, bool] = {}
DEFAULTS = {"sizes": [8, 32, 128, 512], "seeds": 2, "steps": 1500, "lr": 3e-3}


def results_dir():
    import agentlab as AL
    from pathlib import Path
    path = Path(AL._dir()) / "studies"
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_path(name: str):
    return results_dir() / f"laws_{name}.json"


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def usable(result, kind: str) -> bool:
    """Whether a saved run has what its card draws. A file from another kind of run (an old data-efficiency
    run of what is now a guarantee test, say) or one missing fields is treated as no run at all, so the page
    never shows numbers the card cannot honestly label."""
    if not isinstance(result, dict) or not isinstance(result.get("rows"), list) or not result["rows"]:
        return False
    rows = result["rows"]
    if kind == "guarantee":
        return result.get("kind") == "guarantee" and _num(result.get("n")) and all(
            isinstance(r, dict) and isinstance(r.get("label"), str) and r["label"]
            and _num(r.get("plain")) and _num(r.get("law")) for r in rows)
    return result.get("kind", "data") == "data" and all(
        isinstance(r, dict) and _num(r.get("n")) and _num(r.get("plain")) and _num(r.get("law")) for r in rows)


def saved(name: str, kind: str = ""):
    try:
        r = json.loads(result_path(name).read_text())
    except (OSError, ValueError):
        return None
    kind = kind or ("guarantee" if name in GUARANTEES else "data")
    return r if usable(r, kind) else None


# Where each law stands, from the measurements on the development machine (CHANGELOG has the numbers).
# confirmed: a data saver on its own task. measuring: not settled yet. shelved: no gap after a fair try;
# the block stays in the palette, but the Laws page does not show it.
STATUS = {"physics": "confirmed", "math": "confirmed", "psychology": "confirmed",
          # level up to n=128; from n=512 clearly ahead (0.094 against 0.185, and 0.043 against 0.103 at 2048)
          "dale": "confirmed",
          # with Tanh it no longer blows up and is ahead at n=8 and n=512, but level at 32 and 128: not yet clean
          "lagrangian": "measuring",
          # with a normalised write, still no recall (error ≥ 0.93 where answering zero scores 1.0); its only
          # lead was over a GRU doing worse than zero, which is not a data saving
          "biology": "shelved",
          # two laws in one layer; shown in its own group, judged by its own run
          "compose": "composed"}
GUARANTEE_N = 512


def overview() -> Dict[str, Any]:
    """Every arm the page shows, by standing, and every guarantee test, each with its last run on this machine
    (if any) or the run going now. Shelved arms are left out."""
    arms, tests = [], []
    for name, fn in ARMS.items():
        status = STATUS.get(name, "measuring")
        if status == "shelved":
            continue
        spec = fn()
        arms.append({"arm": name, "kind": "data", "status": status, "law": spec["law_name"], "task": spec["task"],
                     "measure": spec.get("measure", "error"), "doc": " ".join((fn.__doc__ or "").split()),
                     "result": _JOBS.get(name) or saved(name)})
    for name, fn in GUARANTEES.items():
        spec = fn()
        tests.append({"arm": name, "kind": "guarantee", "status": "guarantee", "law": spec["law_name"],
                      "task": spec["task"], "better": spec["better"], "unit": spec["unit"],
                      "doc": " ".join((fn.__doc__ or "").split()), "result": _JOBS.get(name) or saved(name)})
    shelved = [n for n in ARMS if STATUS.get(n) == "shelved"]
    return {"arms": arms, "guarantees": tests, "shelved": shelved, "defaults": DEFAULTS, "guarantee_n": GUARANTEE_N}


def start(name: str, sizes=None, seeds=None, steps=None) -> Dict[str, Any]:
    if name not in ARMS and name not in GUARANTEES:
        raise KeyError(name)
    if (_JOBS.get(name) or {}).get("status") == "running":
        return _JOBS[name]
    import threading
    kind = "guarantee" if name in GUARANTEES else "data"
    cfg = {"sizes": [int(n) for n in (sizes or DEFAULTS["sizes"])][:8], "seeds": int(seeds or DEFAULTS["seeds"]),
           "steps": int(steps or DEFAULTS["steps"]), "lr": DEFAULTS["lr"]}
    if not cfg["sizes"] or min(cfg["sizes"]) < 2 or max(cfg["sizes"]) > 4096 or not 1 <= cfg["seeds"] <= 5 \
            or not 1 <= cfg["steps"] <= 10000:
        raise ValueError("Sizes between 2 and 4096 (up to 8 of them), 1 to 5 seeds, and up to 10,000 steps.")
    spec = (GUARANTEES if kind == "guarantee" else ARMS)[name]()
    job = {"arm": name, "kind": kind, "law": spec["law_name"], "task": spec["task"], "status": "running", "rows": [],
           "started": time.time(), "verdict": None, **cfg}
    if kind == "guarantee":
        job.update(n=GUARANTEE_N, better=spec["better"], unit=spec["unit"])
    _JOBS[name] = job
    _STOPS[name] = False
    path = result_path(name)

    def go():
        try:
            if kind == "guarantee":
                res = run_guarantee(name, GUARANTEE_N, cfg["seeds"], cfg["steps"], cfg["lr"], quiet=True,
                                    on_row=lambda r: job["rows"].append(r), stop=lambda: _STOPS.get(name))
            else:
                res = run_arm(name, cfg["sizes"], cfg["seeds"], cfg["steps"], cfg["lr"], quiet=True,
                              on_row=lambda r: job["rows"].append(r), stop=lambda: _STOPS.get(name))
            res.update(status="stopped" if _STOPS.get(name) else "done", started=job["started"],
                       seconds=round(time.time() - job["started"], 1), finished=time.time())
            if res["status"] == "done":
                path.write_text(json.dumps(res, indent=2))
            job.update(res)
        except Exception as exc:  # noqa: BLE001
            job.update(status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            if job.get("status") == "done":
                _JOBS.pop(name, None)

    import contextvars
    threading.Thread(target=contextvars.copy_context().run, args=(go,), daemon=True).start()
    return job


def stop(name: str) -> Dict[str, Any]:
    _STOPS[name] = True
    return _JOBS.get(name) or {"arm": name, "status": "idle"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="*", default=list(ARMS) + list(GUARANTEES))
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 32, 128, 512])
    ap.add_argument("--n", type=int, default=GUARANTEE_N, help="examples for a guarantee test (convex, lipschitz)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--json", help="write the results here as well (one list, every arm)")
    ap.add_argument("--save", action="store_true", help="also save each arm where the Laws page reads it")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="options an arm reads, e.g. --set eps=1e-4 activation=tanh (the lagrangian arm)")
    a = ap.parse_args()

    for item in a.set:
        key, _, value = item.partition("=")
        OPTIONS[key.strip()] = value.strip()
    results = []
    for name in a.arms:
        if name not in ARMS and name not in GUARANTEES:
            raise SystemExit(f"no arm called {name!r}; have {sorted(ARMS) + sorted(GUARANTEES)}")
        t0 = time.time()
        if name in GUARANTEES:
            spec = GUARANTEES[name]()
            print(f"\n[{name}]  guarantee: {spec['law_name']}\n  task: {spec['task']} (n={a.n})")
            res = run_guarantee(name, a.n, a.seeds, a.steps, a.lr)
            res["seconds"] = round(time.time() - t0, 1)
            print(f"  ({res['seconds']}s; no data-efficiency verdict for a guarantee)")
        else:
            spec = ARMS[name]()
            print(f"\n[{name}]  law: {spec['law_name']}\n  task: {spec['task']}")
            res = run_arm(name, a.sizes, a.seeds, a.steps, a.lr)
            res["seconds"] = round(time.time() - t0, 1)
            print(f"  → {res['verdict']}  ({res['seconds']}s)")
        if OPTIONS:
            res["options"] = dict(OPTIONS)
        results.append(res)
        if a.save:
            res.update(status="done", started=t0, finished=time.time())
            result_path(name).write_text(json.dumps(res, indent=2))

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
