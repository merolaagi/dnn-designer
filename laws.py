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
    return {"arm": name, "law": spec["law_name"], "task": spec["task"],
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
    lead = "" if first == 0 else (f"Below n={at['n']} the two are within the noise of each other (at n={rows[0]['n']}: "
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
    spec["law"] = lambda: LagrangianField(2, hidden=64)
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
# arm 5 — mathematics: convexity, judged by where gradient descent on the fitted energy ends up
# ---------------------------------------------------------------------------

@arm("convex")
def convex_arm():
    """A convex energy in 4-D (a tilted quadratic plus a log-sum-exp), sampled at N points. Each
    model is fitted to the samples, then gradient descent runs on the fitted model from 100 random
    starts. A convex fit has one minimum, so every start lands in the same place; a plain MLP can
    have spurious wells. Error is the mean distance of the 100 end points from the true minimiser,
    so a fit with extra wells, or one minimum in the wrong place, both count."""
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

    def minimiser():
        x = torch.zeros(1, D, requires_grad=True)
        opt = torch.optim.LBFGS([x], max_iter=200, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            e = energy(x).sum()
            e.backward()
            return e
        opt.step(closure)
        return x.detach()

    x_star = minimiser()

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.rand(n, D, generator=g) * 6 - 3
        starts = torch.rand(100, D, generator=g) * 6 - 3
        return x, energy(x), starts

    def plain():
        return nn.Sequential(nn.Linear(D, 64), nn.Softplus(), nn.Linear(64, 64), nn.Softplus(), nn.Linear(64, 1))

    def law():
        return Convex(D, hidden=64, layers=2, out=1)

    def loss(model, xs, ys):
        return ((model(xs) - ys) ** 2).mean()

    def error(model, starts):
        x = starts.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            model(x).sum().backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(-3, 3)                       # stay where there was data
        return float((x.detach() - x_star).norm(dim=1).mean())

    return {"law": law, "plain": plain, "data": data, "loss": loss, "error": error,
            "task": "fit a convex 4-D energy, then descend it from 100 starts: mean distance from the true minimum",
            "law_name": "convexity (input-convex network)", "measure": "distance from the minimum"}


# ---------------------------------------------------------------------------
# arm 6 — mathematics: a Lipschitz bound, judged by error under a worst-case nudge of the input
# ---------------------------------------------------------------------------

@arm("lipschitz")
def lipschitz_arm():
    """A 1-Lipschitz target in 8-D, y = sin(u·x) with |u| = 1. Each input of the test set is nudged
    by 0.3 in the direction that most increases that model's error (one gradient-sign step, scaled
    to norm 0.3), and the error is measured against the truth at the nudged point. A plain MLP fitted
    to few points can be steep between them; a stack of 1-Lipschitz layers cannot."""
    import torch
    import torch.nn as nn
    from lawsource import build

    Lip = build("LipschitzLinear")
    D, EPS = 8, 0.3
    u = torch.nn.functional.normalize(torch.tensor([1.0, -0.5, 0.3, 0.8, -0.2, 0.6, 0.1, -0.7]), dim=0)

    def truth(x):
        return torch.sin(x @ u).unsqueeze(-1)

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, D, generator=g)
        xt = torch.randn(2000, D, generator=g)
        return x, truth(x), xt

    def plain():
        return nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def law():
        return nn.Sequential(Lip(D, 64, bound=1.0), nn.ReLU(), Lip(64, 64, bound=1.0), nn.ReLU(), Lip(64, 1, bound=1.0))

    def loss(model, xs, ys):
        return ((model(xs) - ys) ** 2).mean()

    def error(model, xt):
        x = xt.clone().requires_grad_(True)
        ((model(x) - truth(x)) ** 2).sum().backward()
        g = x.grad
        x2 = (xt + EPS * g / (g.norm(dim=1, keepdim=True) + 1e-12)).detach()
        with torch.no_grad():
            return float(((model(x2) - truth(x2)) ** 2).mean().sqrt())

    return {"law": law, "plain": plain, "data": data, "loss": loss, "error": error,
            "task": "fit y = sin(u·x) in 8-D; RMSE after a worst-case nudge of norm 0.3 on 2000 test points",
            "law_name": "Lipschitz bound (1 per layer)", "measure": "RMSE under a nudge"}


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


def saved(name: str):
    try:
        return json.loads(result_path(name).read_text())
    except (OSError, ValueError):
        return None


def overview() -> Dict[str, Any]:
    """Every arm: its law and task, and its last run on this machine (if any), or the run going now."""
    out = []
    for name, fn in ARMS.items():
        spec = fn()
        live = _JOBS.get(name)
        out.append({"arm": name, "law": spec["law_name"], "task": spec["task"],
                    "measure": spec.get("measure", "error"), "doc": " ".join((fn.__doc__ or "").split()),
                    "result": live or saved(name)})
    return {"arms": out, "defaults": DEFAULTS}


def start(name: str, sizes=None, seeds=None, steps=None) -> Dict[str, Any]:
    if name not in ARMS:
        raise KeyError(name)
    if (_JOBS.get(name) or {}).get("status") == "running":
        return _JOBS[name]
    import threading
    cfg = {"sizes": [int(n) for n in (sizes or DEFAULTS["sizes"])][:8], "seeds": int(seeds or DEFAULTS["seeds"]),
           "steps": int(steps or DEFAULTS["steps"]), "lr": DEFAULTS["lr"]}
    if not cfg["sizes"] or min(cfg["sizes"]) < 2 or max(cfg["sizes"]) > 4096 or not 1 <= cfg["seeds"] <= 5 \
            or not 1 <= cfg["steps"] <= 10000:
        raise ValueError("Sizes between 2 and 4096 (up to 8 of them), 1 to 5 seeds, and up to 10,000 steps.")
    spec = ARMS[name]()
    job = {"arm": name, "law": spec["law_name"], "task": spec["task"], "status": "running", "rows": [],
           "started": time.time(), "verdict": None, **cfg}
    _JOBS[name] = job
    _STOPS[name] = False
    path = result_path(name)

    def go():
        try:
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
            if job.get("status") != "running":
                _JOBS.pop(name, None) if job["status"] == "done" else None

    import contextvars
    threading.Thread(target=contextvars.copy_context().run, args=(go,), daemon=True).start()
    return job


def stop(name: str) -> Dict[str, Any]:
    _STOPS[name] = True
    return _JOBS.get(name) or {"arm": name, "status": "idle"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="*", default=list(ARMS))
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 32, 128, 512])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--json", help="write the results here as well (one list, every arm)")
    ap.add_argument("--save", action="store_true", help="also save each arm where the Laws page reads it")
    a = ap.parse_args()

    results = []
    for name in a.arms:
        if name not in ARMS:
            raise SystemExit(f"no arm called {name!r}; have {sorted(ARMS)}")
        spec = ARMS[name]()
        print(f"\n[{name}]  law: {spec['law_name']}\n  task: {spec['task']}")
        t0 = time.time()
        res = run_arm(name, a.sizes, a.seeds, a.steps, a.lr)
        res["seconds"] = round(time.time() - t0, 1)
        print(f"  → {res['verdict']}  ({res['seconds']}s)")
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
