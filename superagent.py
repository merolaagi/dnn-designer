"""superagent.py — one agent that takes a goal and carries it from reading to a measured model.

The other agents here each do one thing: a scout reads, a study trains variants, the Laws page measures a
law. This one runs the whole line, start to finish, with nobody pressing anything in between:

    brief      the goal (typed, or picked from Radar) is matched to a kind of problem and a mock task
    swarm      assistants go out in parallel, one per angle, read papers and bring back rule cards:
               a rule, the passage that states it, and where it was read
    rules      each card is matched to a construction that holds the rule in the structure of the network
    verify     every construction is run with random weights; a rule that does not hold exactly is dropped
    design     candidate networks are assembled from the rules that survived (rulesource.RuleNet)
    mock data  training data is generated from the task's own equations, so no dataset is needed
    measure    each candidate and a conventional network no smaller than it are trained on the same data
               at several sizes; error, the rules each one breaks, size and speed are recorded
    slim       the winner is shrunk while it still beats the conventional network at its full size
    explain    each finding gets what it means and how far it reaches (superexplain.py)
    report     what was read, what was built, what was measured, and the model, on the canvas

What it will not do is call something a win that was not measured as one. A candidate wins a size only
when its error, plus the spread across seeds, is below the conventional network's minus its spread. If
nothing wins, the report says so, and says which rule each candidate was built on.

Without a model in Settings the assistants still run: they match rules by vocabulary instead of reading
for meaning, and the report says that is what happened.
"""
from __future__ import annotations

import contextvars
import itertools
import json
import math
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# the rule library: what can be built in, and how to tell that it holds
# ---------------------------------------------------------------------------

RULES: Dict[str, Dict[str, Any]] = {
    "energy": {
        "name": "Energy conservation",
        "statement": "The motion comes from one scalar energy H by Hamilton's equations, so H cannot change along it.",
        "built": "The network learns H and returns J∇H; it never outputs a motion directly.",
        "check": "volume change of the motion, |trace J| / |J| (0 for every Hamiltonian motion)",
        "words": [r"hamiltonian", r"energy[- ]conserv", r"conserv\w+ (of )?energy", r"symplectic", r"conserved quantit"],
        "source": "Greydanus, Dzamba, Yosinski: Hamiltonian Neural Networks (2019)",
    },
    "permutation": {
        "name": "Permutation invariance",
        "statement": "Relabelling identical parts (particles, points, items) cannot change the answer.",
        "built": "Every row goes through the same network and the results are summed; an order is never seen.",
        "check": "change of the output when the rows are reordered, over its spread",
        "words": [r"permutation[- ](in|equi)varian", r"deep sets?", r"order[- ]invarian", r"exchangeab", r"set[- ]function",
                  r"identical particles"],
        "source": "Zaheer et al.: Deep Sets (2017)",
    },
    "rotation": {
        "name": "Rotation invariance",
        "statement": "Turning the whole system cannot change its energy or its answer.",
        "built": "The network is only shown dot products between vectors, which do not change under rotation.",
        "check": "change of the output when every vector is rotated, over its spread",
        "words": [r"rotation(al)?[- ]?(in|equi)varian", r"rotational symmetr", r"\bso\(\d\)", r"\bse\(\d\)", r"\be\(\d\)",
                  r"angular momentum", r"isotrop", r"central (force|potential)"],
        "source": "Noether (1918); Villar et al.: Scalars are universal (2021)",
    },
    "translation": {
        "name": "Translation invariance",
        "statement": "Moving the whole system cannot change its energy or its answer.",
        "built": "Positions are read relative to their centroid, so a common shift is removed before anything is computed.",
        "check": "change of the output when every position is shifted by the same amount, over its spread",
        "words": [r"translation(al)?[- ]?(in|equi)varian", r"translational symmetr", r"conserv\w+ (of )?(linear )?momentum",
                  r"momentum conserv", r"galilean"],
        "source": "Noether (1918)",
    },
    "reflection": {
        "name": "Reflection symmetry",
        "statement": "The mirror image of a state has the same energy or the same answer.",
        "built": "The network's output is averaged with its output on the mirrored input.",
        "check": "change of the output when the input is mirrored, over its spread",
        "words": [r"reflection symmetr", r"parity", r"mirror symmetr", r"\beven function", r"time[- ]reversal", r"z_?2 symmetr"],
        "source": "group averaging (Reynolds operator)",
    },
    "monotone": {
        "name": "Monotonicity",
        "statement": "The answer can only rise (or only fall) as a given input rises.",
        "built": "Weights on those inputs are held to one sign and every later weight is non-negative.",
        "check": "share of probes where raising a constrained input moved the output the wrong way",
        "words": [r"monoton", r"isotonic", r"non-?decreasing", r"non-?increasing", r"law of demand", r"shape constrain"],
        "source": "Sill: Monotonic Networks (1998)",
    },
    "convex": {
        "name": "Convexity",
        "statement": "The answer is convex in the inputs: one minimum, costs that grow at least linearly.",
        "built": "Input-convex network: non-negative hidden weights and convex, non-decreasing activations.",
        "check": "share of probes where the value at a midpoint was above the average of the two ends",
        "words": [r"convex", r"\bicnn\b", r"diminishing returns", r"concav"],
        "source": "Amos, Xu, Kolter: Input Convex Neural Networks (2017)",
    },
    "additive": {
        "name": "Additivity",
        "statement": "Each input contributes separately; no two inputs interact.",
        "built": "A sum of one-input networks.",
        "check": "interaction between two inputs (mixed difference), over the output's spread",
        "words": [r"additive model", r"generali[sz]ed additive", r"separab", r"\bgam\b", r"no interaction"],
        "source": "Hastie, Tibshirani: Generalized Additive Models (1986)",
    },
    "slots": {
        "name": "Capacity-limited memory",
        "statement": "Only a few items can be held at once, however long the list.",
        "built": "K learned queries attend over the list; only the K slot vectors go on.",
        "check": "whether the number of vectors that leave changes with the length of the list",
        "words": [r"working memory", r"slot attention", r"capacity limit", r"chunk", r"object[- ]centric"],
        "source": "Cowan (2001); Locatello et al.: Slot Attention (2020)",
    },
    "fastweights": {
        "name": "Hebbian fast memory",
        "statement": "A memory can be written in one exposure by a local rule: what fires together wires together.",
        "built": "A fast weight matrix is updated by the outer product of the activity, with no gradient.",
        "check": "whether the optimizer can reach the fast memory",
        "words": [r"hebbian", r"fast weights?", r"associative memor", r"hopfield", r"one[- ]shot (memor|recall)", r"pattern completion"],
        "source": "Ba et al.: Using Fast Weights to Attend to the Recent Past (2016)",
    },
}

KINDS: Dict[str, Dict[str, Any]] = {
    "dynamics": {
        "name": "Dynamics and physics",
        "words": [r"dynamic", r"physic", r"motion", r"\bmov", r"\bbod(y|ies)\b", r"pendulum", r"orbit", r"planet", r"gravit", r"oscillat", r"spring",
                  r"particle", r"trajector", r"mechanic", r"energy", r"simulat", r"molecul", r"\bode\b", r"differential equation"],
        "angles": [("conservation", "energy conservation Hamiltonian neural network learning dynamics"),
                   ("symmetry", "symmetry equivariant neural network physical system Noether conserved"),
                   ("structure", "structure-preserving learning dynamical systems symplectic")],
        "rules": ["energy", "permutation", "rotation", "translation", "reflection"],
    },
    "sets": {
        "name": "Sets and symmetry",
        "words": [r"\bsets?\b", r"point cloud", r"symmetr", r"invarian", r"equivarian", r"permut", r"unordered", r"rotat",
                  r"geometr", r"graph"],
        "angles": [("order", "permutation invariant neural network sets deep sets"),
                   ("geometry", "rotation translation invariant neural network point cloud scalars")],
        "rules": ["permutation", "rotation", "translation", "reflection"],
    },
    "sequences": {
        "name": "Sequences and memory",
        "words": [r"sequenc", r"memor", r"recall", r"\blist\b", r"token", r"language", r"text", r"\bcopy", r"count", r"attention",
                  r"time series", r"remember"],
        "angles": [("capacity", "working memory capacity limit slot attention neural network"),
                   ("association", "Hebbian fast weights associative memory one-shot recall"),
                   ("order", "order invariant aggregation sequence bag of items neural network")],
        "rules": ["slots", "fastweights", "permutation"],
    },
    "tables": {
        "name": "Tables and forecasts",
        "words": [r"table", r"tabular", r"forecast", r"demand", r"price", r"pricing", r"cost", r"sales", r"revenue", r"fuel",
                  r"curve", r"regress", r"predict\w* (how much|the amount|volume)", r"store", r"inventory", r"risk", r"dose"],
        "angles": [("monotone", "monotonic neural network shape constraints tabular prediction"),
                   ("convex", "input convex neural network convexity constraint"),
                   ("additive", "neural additive models interpretable separable")],
        "rules": ["monotone", "convex", "additive", "reflection"],
    },
}

BUDGETS = {
    "quick":    {"tasks": 1, "candidates": 3, "sizes": [16, 128],          "seeds": 2, "steps": 500,  "papers": 4, "read": 0},
    "standard": {"tasks": 2, "candidates": 5, "sizes": [16, 64, 256],      "seeds": 2, "steps": 900,  "papers": 6, "read": 1},
    "thorough": {"tasks": 3, "candidates": 8, "sizes": [16, 64, 256, 512], "seeds": 3, "steps": 1500, "papers": 8, "read": 2},
}
STAGES = ["brief", "swarm", "rules", "verify", "design", "measure", "slim", "explain", "report"]
STAGE_NAMES = {"brief": "Brief", "swarm": "Research swarm", "rules": "Rules", "verify": "Verify", "design": "Design",
               "measure": "Mock data and training", "slim": "Slim", "explain": "Explain", "report": "Report"}
EXACT = 1e-4            # a rule "holds by construction" when its violation with random weights is below this
ARXIV_PAUSE = 3.1       # arXiv asks for no more than one request every three seconds
_ARXIV = threading.Lock()


# ---------------------------------------------------------------------------
# mock tasks: data generated from the rules of a system, with the truth known
# ---------------------------------------------------------------------------

def _hamiltonian_task(tid, name, summary, N, k, H, scale, z0, holds, words, extra=None):
    """A system given by its energy H(z), z: [B, N, 2k] (positions then momenta per row). The field is
    read off H, so the mock data obey every symmetry H has, exactly."""
    import torch
    import laws

    def field(z):
        with torch.enable_grad():
            x = z.detach().clone().requires_grad_(True)
            g = torch.autograd.grad(H(x).sum(), x)[0]
        return torch.cat([g[..., k:], -g[..., :k]], dim=-1)

    def sample(n, g):
        return (torch.rand(n, N, 2 * k, generator=g) * 2 - 1) * torch.tensor(scale).repeat_interleave(k)

    start = torch.tensor(z0, dtype=torch.float32).reshape(1, N, 2 * k)
    truth, errors = laws._dynamics_errors(field, lambda z: H(z), sample, start)
    for e in (extra or []):
        errors.append(e(start))

    def data(n, seed):
        z = sample(n, torch.Generator().manual_seed(seed))
        return z, field(z), truth

    return {"id": tid, "kind": "dynamics", "name": name, "summary": summary, "form": "rows", "shape": [N, 2 * k], "out": 0,
            "k": k, "momenta": True, "holds": holds, "words": words, "data": data, "sample": sample,
            "loss": "mse", "errors": errors, "primary": "field", "field": True,
            "measure": "one-step error of the predicted motion on 2,000 states it never saw"}


def _drift(label, key, quantity, short):
    """Largest change of a conserved quantity of the TRUE system along the model's own rollout."""
    def make(start):
        import torch
        from conservation import _rk4

        def error(model, test, seed):
            with torch.enable_grad():
                path = _rk4(lambda z: model(z), start, 0.05, 200).detach()
            v = quantity(path)
            return float((v - v[0]).abs().max() / (1.0 + v[0].abs().max()))
        return {"key": key, "label": label, "error": error, "short": short}
    return make


def task_pendulum():
    import torch
    return _hamiltonian_task(
        "pendulum", "A swinging pendulum", "Learn how a pendulum moves from pairs of (state, rate of change).", 1, 1,
        lambda z: 0.5 * z[..., 1].sum(-1) ** 2 + (1 - torch.cos(z[..., 0])).sum(-1), [2.5, 1.5], [[1.5, 0.0]],
        ["energy", "reflection"], [r"pendulum", r"swing", r"oscillat"])


def task_double_well():
    return _hamiltonian_task(
        "double_well", "A particle in a double well", "Learn the motion of a particle that can sit in either of two wells.", 1, 1,
        lambda z: (0.5 * z[..., 1] ** 2 + 0.25 * z[..., 0] ** 4 - 0.5 * z[..., 0] ** 2).sum(-1), [1.8, 1.2], [[1.2, 0.0]],
        ["energy", "reflection"], [r"well", r"bistab", r"duffing", r"switch"])


def task_chain():
    def H(z):
        q, p = z[..., 0], z[..., 1]
        pairs = (q.unsqueeze(-1) - q.unsqueeze(-2)) ** 2
        return (0.5 * p ** 2 + 0.25 * q ** 4).sum(-1) + 0.25 * pairs.sum((-1, -2))
    return _hamiltonian_task(
        "chain", "Three coupled particles", "Learn the motion of three identical particles joined by springs, each in its own well.",
        3, 1, H, [1.5, 1.0], [[1.0, 0.0], [-0.5, 0.3], [0.2, -0.4]],
        ["energy", "permutation", "reflection"], [r"chain", r"coupled", r"spring", r"lattice", r"particles", r"molecul"])


def task_orbit():
    def H(z):
        q, p = z[..., 0, :2], z[..., 0, 2:]
        return 0.5 * (p ** 2).sum(-1) - 1.0 / torch_sqrt((q ** 2).sum(-1) + 0.2)

    def angular(path):
        q, p = path[..., 0, :2], path[..., 0, 2:]
        return q[..., 0] * p[..., 1] - q[..., 1] * p[..., 0]
    return _hamiltonian_task(
        "orbit", "A body in orbit", "Learn the motion of a body around a centre that pulls it (a softened gravity).", 1, 2,
        H, [1.5, 1.0], [[1.0, 0.0, 0.0, 0.8]], ["energy", "rotation", "reflection"],
        [r"orbit", r"planet", r"gravit", r"central", r"kepler", r"satellit"],
        extra=[_drift("drift of the angular momentum over the model's rollout", "angular", angular, "Angular momentum drift")])


def task_two_body():
    def H(z):
        q, p = z[..., :2], z[..., 2:]
        d2 = ((q[..., 0, :] - q[..., 1, :]) ** 2).sum(-1)
        return 0.5 * (p ** 2).sum((-1, -2)) + 0.5 * d2 + 0.25 * d2 ** 2

    def momentum(path):
        return path[..., 2:].sum(-2).norm(dim=-1)
    return _hamiltonian_task(
        "two_body", "Two bodies bound together", "Learn the motion of two identical bodies in the plane joined by a stiffening spring.",
        2, 2, H, [1.0, 0.8], [[0.6, 0.0, 0.0, 0.5], [-0.6, 0.0, 0.0, -0.3]],
        ["energy", "permutation", "rotation", "translation", "reflection"],
        [r"two[- ]body", r"\bbodies", r"binary", r"molecul", r"bond", r"dimer", r"interact"],
        extra=[_drift("drift of the total momentum over the model's rollout", "momentum", momentum, "Momentum drift")])


def torch_sqrt(x):
    import torch
    return torch.sqrt(x)


def _regression_errors(holdout=None):
    import torch

    def rmse(model, test, seed):
        xt, yt = test[0], test[1]
        with torch.no_grad():
            return float(((model(xt) - yt) ** 2).mean().sqrt())

    out = [{"key": "rmse", "label": "error on 2,000 fresh examples", "error": rmse}]
    if holdout:
        def outside(model, test, seed):
            xo, yo = test[2], test[3]
            with torch.no_grad():
                return float(((model(xo) - yo) ** 2).mean().sqrt())
        out.append({"key": "outside", "label": holdout, "error": outside, "short": "Outside range"})
    return out


def _set_task(tid, name, summary, N, k, target, holds, words):
    import torch

    def sample(n, g):
        return torch.rand(n, N, k, generator=g) * 4 - 2

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        x, xt = sample(n, g), sample(2000, g)
        return x, target(x), (xt, target(xt))

    return {"id": tid, "kind": "sets", "name": name, "summary": summary, "form": "rows", "shape": [N, k], "out": 1, "k": k,
            "momenta": False, "holds": holds, "words": words, "data": data, "sample": sample, "loss": "mse",
            "errors": _regression_errors(), "primary": "rmse", "field": False,
            "measure": "error on 2,000 fresh sets"}


def task_set_sum():
    import torch
    return _set_task("set_sum", "A property of a set of points", "Predict a number that depends on six points but not on their order.",
                     6, 2, lambda x: (torch.sin(x[..., 0]) * torch.cos(x[..., 1])).sum(-1, keepdim=True),
                     ["permutation"], [r"\bset\b", r"order", r"unordered", r"bag", r"collection"])


def task_pair_energy():
    import torch

    def target(x):
        d2 = ((x.unsqueeze(2) - x.unsqueeze(1)) ** 2).sum(-1)
        return 0.5 * (torch.exp(-d2).sum((-1, -2)) - x.shape[1]).unsqueeze(-1)
    return _set_task("pair_energy", "The energy of a cluster of points",
                     "Predict the interaction energy of five points, which depends only on the distances between them.",
                     5, 2, target, ["permutation", "rotation", "translation", "reflection"],
                     [r"cluster", r"distance", r"geometr", r"point cloud", r"molecul", r"rotat", r"shape", r"energy"])


def _sequence_task(tid, name, summary, L, D, out, make, loss, holds, words, plain, measure):
    import torch

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        ctx = make(None, g)
        x, y = make(n, g, ctx)
        return x, y, make(2000, g, ctx)

    def error(model, test, seed):
        xt, yt = test
        with torch.no_grad():
            p = model(xt)
            return float((p.argmax(1) != yt).float().mean()) if loss == "ce" else float(((p - yt) ** 2).mean().sqrt())

    def sample(n, g):
        return make(n, g, make(None, torch.Generator().manual_seed(0)))[0]

    return {"id": tid, "kind": "sequences", "name": name, "summary": summary, "form": "sequence", "shape": [L, D], "out": out,
            "k": D, "momenta": False, "holds": holds, "words": words, "data": data, "sample": sample, "loss": loss,
            "errors": [{"key": "error", "label": measure, "error": error}], "primary": "error", "field": False,
            "measure": measure, "plain": plain}


def task_hidden_item():
    import torch
    import torch.nn.functional as F
    L, D, C = 40, 8, 4

    def make(n, g, protos=None):
        if n is None:
            return F.normalize(torch.randn(C, D, generator=g), dim=1) * 3
        x = torch.randn(n, L, D, generator=g) * 0.7
        cls = torch.randint(0, C, (n,), generator=g)
        x[torch.arange(n), torch.randint(0, L, (n,), generator=g)] = protos[cls]
        return x, cls
    return _sequence_task("hidden_item", "One item hidden in a long list",
                          "Say which of four known items is hidden somewhere in a list of forty.", L, D, C, make, "ce",
                          ["slots", "permutation"], [r"hidden", r"needle", r"find", r"detect", r"attention", r"working memory", r"search"],
                          "gru", "share of 2,000 fresh lists answered wrongly")


def task_majority():
    import torch
    L, D, C = 24, 6, 4

    def make(n, g, ctx=None):
        if n is None:
            return torch.randn(C, D, generator=g) * 1.2
        weights = torch.rand(n, C, generator=g) + 0.2
        tokens = torch.multinomial(weights, L, replacement=True, generator=g)          # [n, L]
        counts = torch.zeros(n, C).scatter_add_(1, tokens, torch.ones(n, L))
        counts = counts + torch.rand(n, C, generator=g) * 0.01                         # break ties, the same way every time
        x = ctx[tokens] + torch.randn(n, L, D, generator=g) * 0.6
        return x, counts.argmax(1)
    return _sequence_task("majority", "Which kind of item is most common",
                          "Say which of four kinds of item appears most often in a list of 24, whatever the order.", L, D, C,
                          make, "ce", ["permutation"], [r"count", r"majorit", r"most (common|frequent)", r"vote", r"tally", r"histogram"],
                          "gru", "share of 2,000 fresh lists answered wrongly")


def _table_task(tid, name, summary, lo, hi, target, holds, words, signs=None, noise=0.03):
    import torch
    d = len(lo)
    lo_t, hi_t = torch.tensor(lo), torch.tensor(hi)

    def sample(n, g, stretch=0.0):
        span = hi_t - lo_t
        return lo_t - stretch * span + torch.rand(n, d, generator=g) * span * (1 + 2 * stretch)

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        x, xt = sample(n, g), sample(2000, g)
        y = target(x)
        y = y + torch.randn(y.shape, generator=g) * noise * float(target(xt).std())
        xo = sample(4000, g, stretch=0.25)
        inside = ((xo >= lo_t) & (xo <= hi_t)).all(dim=1)
        xo = xo[~inside][:2000]
        return x, y, (xt, target(xt), xo, target(xo))

    return {"id": tid, "kind": "tables", "name": name, "summary": summary, "form": "vector", "shape": [d], "out": 1, "k": 1,
            "momenta": False, "holds": holds, "words": words, "data": data, "sample": lambda n, g: sample(n, g), "loss": "mse",
            "errors": _regression_errors("error outside the range it was trained on (inputs up to 25% beyond)"),
            "primary": "rmse", "field": False, "signs": signs, "span": [h - l for l, h in zip(lo, hi)], "steps_x": 3,
            "mirror": all(abs(l + h) < 1e-9 for l, h in zip(lo, hi)),
            "measure": "error on 2,000 fresh rows (targets it trained on carry 3% noise)"}


def task_demand():
    import torch
    return _table_task("demand", "Demand against price", "Predict sales from price, traffic and temperature: sales fall with price "
                       "and rise with traffic and warmth, but the mix is not a simple sum.", [0.5, 0.5, -1.0], [3.0, 2.5, 1.0],
                       lambda x: torch.log1p(2 * x[:, 1:2]) * torch.exp(-0.8 * x[:, 0:1]) * (1 + 0.3 * torch.tanh(2 * x[:, 2:3])),
                       ["monotone"], [r"demand", r"price", r"pricing", r"sales", r"fuel", r"volume", r"elastic", r"store"],
                       signs=[-1, 1, 1])


def task_cost():
    import torch
    return _table_task("cost", "A cost that grows faster and faster", "Predict a cost from two quantities; it is convex, as costs "
                       "with overtime and penalties are.", [-1.5, -1.5], [1.5, 1.5],
                       lambda x: torch.log(torch.exp(1.5 * x[:, 0:1] - x[:, 1:2]) + torch.exp(-x[:, 0:1] + 0.5 * x[:, 1:2]) + 1)
                       + 0.4 * x[:, 0:1] ** 2,
                       ["convex"], [r"cost", r"convex", r"penalt", r"risk", r"loss", r"optimi", r"inventory"])


def task_bill():
    import torch
    return _table_task("bill", "A total made of separate parts", "Predict a total from four inputs that each contribute on their own, "
                       "like line items on a bill.", [-2.0] * 4, [2.0] * 4,
                       lambda x: torch.sin(1.5 * x[:, 0:1]) + 0.5 * x[:, 1:2] ** 2 + torch.tanh(2 * x[:, 2:3]) + 0.3 * x[:, 3:4],
                       ["additive"], [r"additive", r"separate", r"bill", r"total", r"sum of", r"score", r"interpret", r"forecast"])


TASKS: Dict[str, Callable] = {
    "pendulum": task_pendulum, "double_well": task_double_well, "chain": task_chain, "orbit": task_orbit,
    "two_body": task_two_body, "set_sum": task_set_sum, "pair_energy": task_pair_energy, "hidden_item": task_hidden_item,
    "majority": task_majority, "demand": task_demand, "cost": task_cost, "bill": task_bill,
}
KIND_TASKS = {"dynamics": ["two_body", "orbit", "chain", "pendulum", "double_well"], "sets": ["pair_energy", "set_sum"],
              "sequences": ["hidden_item", "majority"], "tables": ["demand", "cost", "bill"]}
TASK_BLURB = {"pendulum": "a pendulum", "double_well": "a particle in a double well", "chain": "three coupled particles",
              "orbit": "a body in orbit", "two_body": "two bound bodies", "set_sum": "a property of a set of points",
              "pair_energy": "the energy of a cluster of points", "hidden_item": "one item hidden in a long list",
              "majority": "the most common kind of item", "demand": "demand against price",
              "cost": "a convex cost", "bill": "a total made of separate parts"}


def allowed_rules(task) -> List[str]:
    """The rules this task's shape lets a network be built on, whether or not they are true of the task."""
    if task["form"] == "sequence":
        return ["slots", "fastweights", "permutation"]
    if task["form"] == "vector":            # monotone needs to know which way each input pushes; a mirror
        return ((["monotone"] if task.get("signs") else []) + ["convex", "additive"]      # image must exist to be compared with
                + (["reflection"] if task.get("mirror") else []))
    N = task["shape"][0]
    out = (["energy"] if task["momenta"] else []) + (["permutation"] if N > 1 else [])
    out += (["rotation"] if task["k"] >= 2 else []) + (["translation"] if N > 1 else []) + ["reflection"]
    return out


# ---------------------------------------------------------------------------
# building and checking
# ---------------------------------------------------------------------------

def make_model(task, rules, hidden: int):
    import rulesource
    if task["form"] == "sequence":
        if len(rules) != 1:
            raise ValueError("A sequence model here is built on one rule at a time.")
        return rulesource.build("SeqNet")(task["shape"][1], task["out"], rules[0], hidden=hidden)
    return rulesource.build("RuleNet")(task["shape"], out=task["out"] or 1, rules=rules, hidden=hidden, k=task["k"],
                                       momenta=task["momenta"], signs=task.get("signs"))


def make_plain(task, at_least: int = 0, act: str = "relu"):
    """The conventional network: an MLP on the flattened input (a GRU for sequences), never smaller than
    the candidate it is compared with."""
    import torch
    import torch.nn as nn

    act_name = act
    n_in = math.prod(task["shape"])
    n_out = n_in if task["field"] else task["out"]
    shape = tuple(task["shape"])

    class Flat(nn.Module):
        def __init__(self, h):
            super().__init__()
            act = nn.Tanh if act_name == "tanh" else nn.ReLU
            self.net = nn.Sequential(nn.Linear(n_in, h), act(), nn.Linear(h, h), act(), nn.Linear(h, n_out))

        def forward(self, x):
            lead = x.shape[:-len(shape)]
            y = self.net(x.reshape(-1, n_in))
            return y.reshape(*lead, *shape) if task["field"] else y.reshape(*lead, n_out)

    class Recurrent(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.rnn = nn.GRU(shape[1], h, batch_first=True)
            self.read = nn.Linear(h, n_out)

        def forward(self, x):
            return self.read(self.rnn(x)[0][:, -1])

    cls = Recurrent if task.get("plain") == "gru" else Flat
    h = 64 if cls is Flat else 48
    model = cls(h)
    while count(model) < at_least and h < 1024:
        h = int(h * 1.25) + 1
        model = cls(h)
    return model


def count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _run(model, x):
    import torch
    with torch.enable_grad():
        return model(x).detach()


def violation(rule: str, model, task) -> Optional[float]:
    """How far a model is from obeying a rule, measured by running it. 0 means it holds; None means the rule
    is not a number that can be measured on this task."""
    import torch
    import laws

    g = torch.Generator().manual_seed(31_000)
    x = task["sample"](256, g)
    is_field, k = task["field"], task["k"]
    y = _run(model, x)
    spread = float(y.std()) + 1e-9

    def rel(a, b):
        return float((a - b).abs().mean()) / spread

    if rule == "energy":
        return laws._liouville(task["sample"], 128)["fn"](model) if is_field else None
    if rule == "permutation":
        if task["form"] == "vector" or task["shape"][0] < 2:
            return None
        perm = torch.randperm(task["shape"][0], generator=g)
        yp = _run(model, x[:, perm])
        return rel(yp, y[:, perm] if is_field else y)
    if rule == "rotation":
        if task["form"] != "rows" or k < 2:
            return None
        a = torch.rand(1, generator=g) * 6.283
        Rm = torch.eye(k)
        Rm[0, 0], Rm[0, 1], Rm[1, 0], Rm[1, 1] = torch.cos(a), -torch.sin(a), torch.sin(a), torch.cos(a)
        turn = lambda z: torch.cat([z[..., i:i + k] @ Rm.T for i in range(0, z.shape[-1], k)], dim=-1)   # noqa: E731
        yr = _run(model, turn(x))
        return rel(yr, turn(y) if is_field else y)
    if rule == "translation":
        if task["form"] != "rows" or task["shape"][0] < 2:
            return None
        shift = torch.zeros(1, 1, task["shape"][1])
        shift[..., :k] = torch.randn(k, generator=g)
        return rel(_run(model, x + shift), y)
    if rule == "reflection":
        if task["form"] == "sequence":
            return None
        ym = _run(model, -x)
        return rel(ym, -y if is_field else y)
    if rule == "monotone":
        signs = task.get("signs")
        if task["form"] != "vector" or not signs:
            return None
        bad, total = 0, 0
        for i, s in enumerate(signs):
            if not s:
                continue
            step = torch.zeros(len(signs))
            step[i] = 0.05 * task["span"][i]
            bad += int((s * (_run(model, x + step) - y) < -1e-5).sum())
            total += len(x)
        return bad / max(1, total)
    if rule == "convex":
        if task["form"] != "vector":
            return None
        b = task["sample"](256, g)
        mid = _run(model, 0.5 * (x + b))
        return float((mid > 0.5 * (y + _run(model, b)) + 1e-4 * (1 + spread)).float().mean())
    if rule == "additive":
        if task["form"] != "vector" or task["shape"][0] < 2:
            return None
        d = task["shape"][0]
        di, dj = torch.zeros(d), torch.zeros(d)
        di[0], dj[1] = 0.3 * task["span"][0], 0.3 * task["span"][1]
        mixed = _run(model, x + di + dj) - _run(model, x + di) - _run(model, x + dj) + y
        return float(mixed.abs().mean()) / spread
    if rule == "slots":
        body = getattr(model, "body", None)
        if task["form"] != "sequence" or body is None or getattr(model, "rule", "") != "slots":
            return None
        with torch.no_grad():
            return 0.0 if body(x).shape[1:] == body(torch.cat([x, x], dim=1)).shape[1:] else 1.0
    if rule == "fastweights":
        return None
    return None


def _loss(task):
    import torch.nn.functional as F
    if task["loss"] == "ce":
        return lambda model, xs, ys: F.cross_entropy(model(xs), ys)
    return lambda model, xs, ys: ((model(xs) - ys) ** 2).mean()


def latency_ms(model, task, rows: int = 1000) -> float:
    """Median time for one forward pass over 1,000 examples, in milliseconds."""
    import torch
    x = task["sample"](rows, torch.Generator().manual_seed(7))
    times = []
    for _ in range(5):
        t = time.perf_counter()
        _run(model, x)
        times.append(time.perf_counter() - t)
    return round(sorted(times)[len(times) // 2] * 1000, 2)


def measure(task, factory: Callable, sizes: List[int], seeds: int, steps: int, lr: float = 3e-3,
            stop: Callable = None, keep: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Train factory() on the task at each size and seed. Returns the rows (mean and spread of every error),
    and, in keep, the model trained on the most data with seed 0."""
    import torch
    import laws

    loss = _loss(task)
    rows = []
    for n in sizes:
        errs = {e["key"]: [] for e in task["errors"]}
        for seed in range(seeds):
            if stop and stop():
                return {"rows": rows, "stopped": True}
            torch.manual_seed(seed)
            xs, ys, test = task["data"](n, seed)
            torch.manual_seed(seed)
            model = factory()
            laws._fit(model, xs, ys, loss, steps, lr)
            model.eval()
            for e in task["errors"]:
                errs[e["key"]].append(laws._finite(float(e["error"](model, test, seed))))
            if keep is not None and n == sizes[-1] and seed == 0:
                keep["model"] = model
        row = {"n": n}
        for key, vals in errs.items():
            row[key], row[key + "_spread"] = sum(vals) / len(vals), (max(vals) - min(vals)) / 2
        row["error"], row["spread"] = row[task["primary"]], row[task["primary"] + "_spread"]
        rows.append(row)
    return {"rows": rows}


def compare(cand: List[Dict[str, Any]], base: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A candidate against the conventional network, size by size. A size is won only beyond both spreads."""
    wins, losses, ratios = [], [], []
    for c, b in zip(cand, base):
        ratios.append(max(b["error"], 1e-12) / max(c["error"], 1e-12))
        if c["error"] + c["spread"] < b["error"] - b["spread"]:
            wins.append(c["n"])
        elif b["error"] + b["spread"] < c["error"] - c["spread"]:
            losses.append(c["n"])
    gain = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else 1.0
    # the least data at which the candidate is already as good as the conventional network with the most
    target, reach = base[-1]["error"], None
    for c in cand:
        if c["error"] <= target:
            reach = c["n"]
            break
    return {"wins": wins, "losses": losses, "gain": round(gain, 2), "ratios": [round(r, 2) for r in ratios],
            "reaches_at": reach, "data_saving": round(base[-1]["n"] / reach, 1) if reach else None}


# ---------------------------------------------------------------------------
# reading: the swarm
# ---------------------------------------------------------------------------

def _ask(system: str, prompt: str, max_tokens: int = 1500) -> Optional[Dict[str, Any]]:
    """One model call that must answer in JSON. None when there is no model or the answer is not JSON."""
    import evolve
    try:
        return evolve._json_in(evolve.ask(system, prompt, max_tokens)["text"])
    except Exception:  # noqa: BLE001 - an assistant that cannot reach the model falls back to vocabulary
        return None


def _live() -> bool:
    import agentlab
    return agentlab.live_available()


def _search(query: str, n: int) -> List[Any]:
    from dnn_bench import paper_to_spec as p2s
    with _ARXIV:
        hits = p2s.search_arxiv(query, n=n)
        time.sleep(ARXIV_PAUSE)                      # only after an answer: an unreachable search owes no courtesy pause
        return hits


def match_rules(text: str) -> Dict[str, str]:
    """Rules named in a text, by vocabulary: {rule: the sentence that names it}."""
    found = {}
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    for rid, rule in RULES.items():
        for s in sentences:
            if any(re.search(w, s, re.I) for w in rule["words"]):
                found[rid] = s.strip()[:320]
                break
    return found


READ = """You are one assistant in a research swarm. Your angle: {angle}. The goal of the whole swarm: {goal}

You are given papers (title, abstract, sometimes a passage). Extract RULES: statements that something is conserved,
invariant, symmetric, monotone, convex, bounded, separable, or that a memory works by a specific mechanism. A rule is
only worth reporting if a network could be built so that it holds whatever the weights are.

Rules this lab can already build in, by id:
{library}

Answer with JSON only:
{{"cards": [{{"paper": <number>, "rule": "<an id above, or other>", "name": "<short name, only when other>",
  "statement": "<the rule in one plain sentence>", "quote": "<the words in the paper that state it, verbatim, at most 40 words>",
  "build": "<only when other: how a network could hold it by construction, one sentence>"}}]}}
Report at most {limit} cards. Do not report a rule the paper does not state. If nothing qualifies, return {{"cards": []}}."""


def read_angle(goal: str, angle: str, query: str, budget: Dict[str, Any], live: bool,
               say: Callable = lambda *a: None, down: Optional[threading.Event] = None) -> Dict[str, Any]:
    """One assistant: search, read, bring back cards. Never raises; an unreachable search is a result.
    `down` is shared by the swarm: once one assistant finds the search unreachable, the rest do not wait on it."""
    from dnn_bench import paper_to_spec as p2s

    out = {"angle": angle, "query": query, "papers": [], "cards": [], "note": "", "reachable": True}
    try:
        if down is not None and down.is_set():
            raise OSError("unreachable")
        hits = _search(query, budget["papers"])
    except Exception as exc:  # noqa: BLE001
        if down is not None:
            down.set()
        out["reachable"] = False
        out["note"] = "The search could not be reached."
        return out
    hits = [h for h in hits if h.title][: budget["papers"]]
    papers = []
    for i, h in enumerate(hits):
        text = h.abstract or ""
        if i < budget["read"]:                       # open the paper itself for the first few
            try:
                paper = p2s.ingest(p2s.full_text(h), title=h.title)
                best = sorted(paper.passages, key=lambda p: -p.score)[:3]
                text += " " + " ".join((p.text or "")[:600] for p in best)
                say(f"read “{h.title[:60]}” in full")
            except Exception:  # noqa: BLE001 - the abstract is still worth reading
                pass
        papers.append({"n": i + 1, "title": h.title, "year": h.year, "authors": h.authors,
                       "url": f"https://arxiv.org/abs/{h.id}" if h.source == "arxiv" else (h.pdf_url or ""), "text": text})
    out["papers"] = [{k: p[k] for k in ("title", "year", "authors", "url")} for p in papers]
    if not papers:
        out["note"] = "The search answered, with no papers for this angle."
        return out
    cards = []
    if live:
        library = "\n".join(f"- {rid}: {r['statement']}" for rid, r in RULES.items())
        listing = "\n\n".join(f"{p['n']}. {p['title']} ({p['year']})\n{p['text'][:1800]}" for p in papers)
        got = _ask(READ.format(angle=angle, goal=goal, library=library, limit=6), listing, 1800)
        for c in ((got or {}).get("cards") or [])[:8]:
            try:
                p = papers[int(c["paper"]) - 1]
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            rid = str(c.get("rule") or "other").strip().lower()
            quote = re.sub(r"\s+", " ", str(c.get("quote") or ""))[:320]
            if not quote:
                continue
            card = {"rule": rid if rid in RULES else "other", "statement": str(c.get("statement") or "")[:300], "quote": quote,
                    "title": p["title"], "url": p["url"], "year": p["year"], "by": "model"}
            if card["rule"] == "other":
                card["name"], card["build"] = str(c.get("name") or "A rule")[:80], str(c.get("build") or "")[:300]
            cards.append(card)
        if got is None:
            out["note"] = "The model did not answer in a form that could be read; rules were matched by vocabulary instead."
    if not cards:
        for p in papers:
            for rid, sentence in match_rules(p["title"] + ". " + p["text"]).items():
                cards.append({"rule": rid, "statement": RULES[rid]["statement"], "quote": sentence, "title": p["title"],
                              "url": p["url"], "year": p["year"], "by": "vocabulary"})
    out["cards"] = cards
    return out


# ---------------------------------------------------------------------------
# the brief
# ---------------------------------------------------------------------------

BRIEF = """A lab builds small neural networks whose structure enforces known rules, and tests them on mock data generated
from equations. Given a goal, choose the mock tasks that are closest to it.

Tasks, by id:
{tasks}

Answer with JSON only: {{"tasks": ["<id>", ...], "why": "<one sentence: how these stand in for the goal>",
"queries": ["<a literature search query, 4 to 8 words>", ...]}}
Choose at most {limit} tasks, the closest first, and at most 3 queries about rules or structure relevant to the goal."""


def brief(goal: str, limit: int, live: bool) -> Dict[str, Any]:
    """Match a goal to mock tasks. The model chooses when there is one; otherwise words do."""
    text = goal.lower()
    score = {kid: sum(1 for w in k["words"] if re.search(w, text)) for kid, k in KINDS.items()}
    kind = max(score, key=score.get) if max(score.values()) > 0 else "dynamics"
    specs = {tid: fn for tid, fn in TASKS.items()}
    picked, why, queries = [], "", []
    if live:
        listing = "\n".join(f"- {tid} ({KIND_OF[tid]}): {TASK_BLURB[tid]}" for tid in specs)
        got = _ask(BRIEF.format(tasks=listing, limit=limit), "Goal: " + goal, 600) or {}
        picked = [t for t in (got.get("tasks") or []) if t in specs][:limit]
        why = str(got.get("why") or "")[:300]
        queries = [str(q)[:120] for q in (got.get("queries") or []) if str(q).strip()][:3]
    if not picked:
        words = {tid: sum(1 for w in WORDS_OF[tid] if re.search(w, text)) for tid in KIND_TASKS[kind]}
        ranked = sorted(KIND_TASKS[kind], key=lambda t: (-words[t], KIND_TASKS[kind].index(t)))
        picked = ranked[:limit]
        why = (f"The goal reads as {KINDS[kind]['name'].lower()}; these are the closest mock tasks of that kind."
               if max(score.values()) > 0 else
               "The goal did not name a kind of problem I recognise, so I took dynamics, where rules are strongest.")
    kinds = list(dict.fromkeys(KIND_OF[t] for t in picked))
    return {"kind": kinds[0], "kinds": kinds, "tasks": picked, "why": why, "queries": queries,
            "by": "model" if live and why and queries else "words"}


KIND_OF = {t: k for k, ts in KIND_TASKS.items() for t in ts}
WORDS_OF: Dict[str, List[str]] = {}


def _load_words():
    if not WORDS_OF:
        for tid, fn in TASKS.items():
            WORDS_OF[tid] = fn()["words"]


def candidates_for(task, evidence: Dict[str, int], limit: int, suggested: Optional[List[List[str]]] = None) -> List[List[str]]:
    """Which combinations of rules to build, best-supported first: everything that was read about together,
    then each rule alone, then pairs. Combinations that cannot be built are dropped later, by building them."""
    allowed = allowed_rules(task)
    read = [r for r in allowed if evidence.get(r)]
    order = sorted(allowed, key=lambda r: (-evidence.get(r, 0), allowed.index(r)))
    out: List[List[str]] = []

    def add(rules):
        if task["field"] and rules:                      # a motion is always read off an energy here
            rules = ["energy"] + list(rules)
        rules = [r for r in allowed if r in set(rules)]
        if rules and rules not in out:
            out.append(rules)

    for s in (suggested or []):
        add(s)
    if task["form"] == "sequence":
        for r in order:
            add([r])
        return out[:limit]
    if len(read) > 1:
        add(read)
    add(allowed if task["form"] == "rows" else [])
    for r in order:
        add([r])
    lead = "energy" if "energy" in allowed else order[0]
    for r in order:
        if r != lead:
            add([lead, r])
    for a, b in itertools.combinations(order, 2):
        add([a, b])
    return out[:limit]


# ---------------------------------------------------------------------------
# a run
# ---------------------------------------------------------------------------

RUNS: Dict[str, "Run"] = {}


class Run:
    def __init__(self, goal: str, budget: str, home: Path, source: Optional[Dict[str, Any]] = None):
        self.id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.goal, self.budget, self.home, self.source = goal, budget, home, source
        self.status, self.stage, self.error = "running", "brief", ""
        self.started, self.finished = time.time(), None
        self.log: List[Dict[str, Any]] = []
        self.data: Dict[str, Any] = {"swarm": [], "cards": [], "rules": [], "tasks": [], "unbuilt": []}
        self.stop = threading.Event()
        self.lock = threading.Lock()

    def say(self, text: str, detail: str = "") -> None:
        with self.lock:
            self.log.append({"at": round(time.time() - self.started, 1), "stage": self.stage, "text": text, "detail": detail})

    def snapshot(self, full: bool = True) -> Dict[str, Any]:
        with self.lock:
            out = {"id": self.id, "goal": self.goal, "budget": self.budget, "status": self.status, "stage": self.stage,
                   "error": self.error, "source": self.source, "started": self.started,
                   "seconds": round((self.finished or time.time()) - self.started, 1),
                   "stages": [{"id": s, "name": STAGE_NAMES[s]} for s in STAGES]}
            if full:
                out.update(json.loads(json.dumps(self.data, default=str)))
                out["log"] = self.log[-80:]
            else:
                out["verdict"] = self.data.get("verdict", "")
                out["won"] = self.data.get("won", None)
            return out

    def save(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / f"{self.id}.json").write_text(json.dumps(self.snapshot(), indent=1))


def home() -> Path:
    import auth
    return auth.sub("superruns")


def listing() -> List[Dict[str, Any]]:
    seen, out = set(), []
    for run in sorted(RUNS.values(), key=lambda r: -r.started):
        if run.home == home():
            seen.add(run.id)
            out.append(run.snapshot(full=False))
    for path in sorted(home().glob("*.json"), reverse=True):
        if path.stem in seen:
            continue
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        out.append({k: d.get(k) for k in ("id", "goal", "budget", "status", "stage", "error", "source", "started", "seconds",
                                          "verdict", "won")})
    return out[:40]


def read(run_id: str) -> Dict[str, Any]:
    run = RUNS.get(run_id)
    if run and run.home == home():
        return run.snapshot()
    path = home() / f"{Path(run_id).name}.json"
    if not path.exists():
        raise KeyError(run_id)
    d = json.loads(path.read_text())
    if not d.get("explain") and d.get("status") in ("done", "stopped"):      # a run from before it could explain itself
        import superexplain
        d["explain"] = superexplain.explain(d)
    return d


def reexplain(run_id: str) -> Dict[str, Any]:
    """Explain a finished run again, with the model if one is connected now, and keep the result."""
    import superexplain
    run = RUNS.get(run_id)
    if run and run.home == home() and run.status == "running":
        raise ValueError("This run is still going. It explains itself when it finishes.")
    d = read(run_id)
    d["explain"] = superexplain.explain(d, ask=_ask if _live() else None)
    if not d.get("stages") or len(d["stages"]) != len(STAGES):
        d["stages"] = [{"id": s, "name": STAGE_NAMES[s]} for s in STAGES]
    (home() / f"{Path(run_id).name}.json").write_text(json.dumps(d, indent=1))
    if run and run.home == home():
        run.data["explain"] = d["explain"]
    return d


def stop(run_id: str) -> Dict[str, Any]:
    run = RUNS.get(run_id)
    if run and run.home == home():
        run.stop.set()
        run.say("Stopping after the step in hand.")
    return read(run_id)


def overview() -> Dict[str, Any]:
    return {"live": _live(), "budgets": {k: {"tasks": v["tasks"], "candidates": v["candidates"], "sizes": v["sizes"]}
                                         for k, v in BUDGETS.items()},
            "kinds": [{"id": k, "name": v["name"], "tasks": [{"id": t, "name": TASK_BLURB[t]} for t in KIND_TASKS[k]]}
                      for k, v in KINDS.items()],
            "rules": [{"id": rid, "name": r["name"], "statement": r["statement"], "built": r["built"], "source": r["source"]}
                      for rid, r in RULES.items()],
            "runs": listing()}


def from_radar() -> Dict[str, Any]:
    """Let it pick its own goal: the newest Radar finding that reads as one of the kinds it can build for."""
    import radar
    items = radar.view().get("items") or []
    best, best_score = None, 0
    for it in items:
        text = " ".join(str(it.get(k) or "") for k in ("headline", "what", "why", "task", "title")).lower()
        s = sum(1 for k in KINDS.values() for w in k["words"] if re.search(w, text))
        s += 2 * len(match_rules(text))
        if s > best_score:
            best, best_score = it, s
    if not best:
        raise ValueError("Radar has nothing I could build from yet. Open Radar and refresh it, or type a goal.")
    goal = f"{best.get('headline') or best.get('title')}. {best.get('what') or ''}".strip()[:600]
    return {"goal": goal, "source": {"kind": "radar", "title": best.get("headline") or best.get("title"), "url": best.get("url") or ""}}


def start(goal: str = "", budget: str = "standard", pick: bool = False, background: bool = True) -> Dict[str, Any]:
    source = None
    if pick or not (goal or "").strip():
        if not pick:
            raise ValueError("Type a goal, or let the super agent pick one from Radar.")
        got = from_radar()
        goal, source = got["goal"], got["source"]
    if budget not in BUDGETS:
        raise ValueError(f"No budget called {budget!r}. There are: {', '.join(BUDGETS)}.")
    running = [r for r in RUNS.values() if r.status == "running" and r.home == home()]
    if running:
        raise ValueError("A super agent is already running here. Stop it, or wait for its report.")
    run = Run(goal.strip()[:800], budget, home(), source)
    RUNS[run.id] = run
    live = _live()                    # resolved here: the account is not visible from a worker thread

    def work():
        try:
            _carry_out(run, live)
            run.status = "stopped" if run.stop.is_set() else "done"
        except Exception as exc:  # noqa: BLE001 - a run must not take the app with it
            run.status, run.error = "error", f"{type(exc).__name__}: {exc}"
        run.finished = time.time()
        try:
            run.save()
        except OSError:
            pass

    if background:
        # the signed-in account lives in a context variable: the worker has to be handed it, or the model
        # settings it reads would be the machine's rather than the account's
        threading.Thread(target=contextvars.copy_context().run, args=(work,), daemon=True).start()
    else:
        work()
    return run.snapshot()


def _carry_out(run: Run, live: bool) -> None:
    import torch

    B = BUDGETS[run.budget]
    D = run.data
    D["mode"] = "live" if live else "rehearsal"
    _load_words()

    # ---- brief -------------------------------------------------------------------------------
    run.stage = "brief"
    run.say("Reading the goal", run.goal[:200])
    br = brief(run.goal, B["tasks"], live)
    D["brief"] = {**br, "kind_name": KINDS[br["kind"]]["name"], "task_names": [TASK_BLURB[t] for t in br["tasks"]]}
    run.say(f"Kind of problem: {KINDS[br['kind']]['name']}", br["why"])
    if run.stop.is_set():
        return

    # ---- swarm -------------------------------------------------------------------------------
    run.stage = "swarm"
    angles = [("the goal as written", run.goal[:160])]
    for kind in br["kinds"]:
        angles += [(a, q) for a, q in KINDS[kind]["angles"]]
    angles += [(f"a question of its own ({i + 1})", q) for i, q in enumerate(br["queries"])]
    angles = angles[: 3 + 2 * B["tasks"]]
    for i, (angle, query) in enumerate(angles):
        D["swarm"].append({"id": i, "angle": angle, "query": query, "status": "waiting", "papers": [], "cards": 0, "note": ""})
    run.say(f"Sending out {len(angles)} assistants", ", ".join(a for a, _ in angles))

    down = threading.Event()

    def go(i):
        if run.stop.is_set():
            return None
        D["swarm"][i]["status"] = "reading"
        got = read_angle(run.goal, angles[i][0], angles[i][1], B, live, say=lambda t: run.say(f"Assistant {i + 1}: {t}"), down=down)
        D["swarm"][i].update(status="done" if got["reachable"] else "unreachable", papers=got["papers"],
                             cards=len(got["cards"]), note=got["note"])
        run.say(f"Assistant {i + 1} ({angles[i][0]}): {len(got['papers'])} papers, {len(got['cards'])} rule cards", got["note"])
        return got

    contexts = [contextvars.copy_context() for _ in angles]          # one each: a context cannot be entered twice at once
    with ThreadPoolExecutor(max_workers=min(4, len(angles))) as pool:
        results = [r for r in pool.map(lambda i: contexts[i].run(go, i), range(len(angles))) if r]
    cards = [c for r in results for c in r["cards"]]
    reachable = any(r["reachable"] for r in results)
    D["reachable"] = reachable
    D["papers_read"] = len({p["url"] or p["title"] for r in results for p in r["papers"]})
    if not reachable:
        run.say("The literature search could not be reached from this machine",
                "The assistants came back empty-handed. I am carrying on with the rules this lab already holds, "
                "and the report says no paper was read.")
    if run.stop.is_set():
        return

    # ---- rules -------------------------------------------------------------------------------
    run.stage = "rules"
    seen = set()
    for c in cards:
        key = (c["rule"], c["url"], c["quote"][:60])
        if key in seen:
            continue
        seen.add(key)
        (D["unbuilt"] if c["rule"] == "other" else D["cards"]).append(c)
    evidence = {rid: sum(1 for c in D["cards"] if c["rule"] == rid) for rid in RULES}
    wanted = list(dict.fromkeys(r for kind in br["kinds"] for r in KINDS[kind]["rules"]))
    for rid in wanted:
        D["rules"].append({"id": rid, "name": RULES[rid]["name"], "statement": RULES[rid]["statement"], "built": RULES[rid]["built"],
                           "check": RULES[rid]["check"], "source": RULES[rid]["source"], "evidence": evidence[rid],
                           "from": "reading" if evidence[rid] else "library"})
    read_n = sum(1 for r in D["rules"] if r["evidence"])
    run.say(f"{len(D['cards'])} rule cards from reading; {read_n} of {len(D['rules'])} rules for this kind were found in papers",
            (f"{len(D['unbuilt'])} more rules were read that this lab has no construction for yet." if D["unbuilt"] else ""))

    # ---- per task: verify, design, measure, slim ---------------------------------------------
    for tid in br["tasks"]:
        if run.stop.is_set():
            return
        task = TASKS[tid]()
        T = {"id": tid, "name": task["name"], "summary": task["summary"], "measure": task["measure"], "holds": task["holds"],
             "shape": task["shape"], "errors": [{"key": e["key"], "label": e["label"],
                         "short": e.get("short") or {"rollout": "Rollout", "energy": "Energy drift"}.get(e["key"], e["key"])}
                        for e in task["errors"]],
             "primary": task["primary"], "checks": [], "candidates": [], "sizes": B["sizes"], "seeds": B["seeds"],
             "steps": B["steps"] * task.get("steps_x", 1)}
        steps = T["steps"]
        D["tasks"].append(T)

        run.stage = "verify"
        run.say(f"Task: {task['name']}", task["summary"])
        combos = candidates_for(task, evidence, B["candidates"])
        built = []
        for rules in combos:
            try:
                model = make_model(task, rules, 32)
            except ValueError as exc:
                run.say(f"Could not build {' + '.join(rules)}", str(exc))
                continue
            ok, checks = True, []
            for r in rules:
                v = violation(r, model, task)
                checks.append({"rule": r, "value": v})
                if v is not None and v > EXACT:
                    ok = False
            T["checks"].append({"rules": rules, "checks": checks, "ok": ok})
            if ok:
                built.append(rules)
            else:
                run.say(f"Dropped {' + '.join(rules)}: a rule did not hold with random weights",
                        ", ".join(f"{c['rule']} {c['value']:.2g}" for c in checks if c["value"] is not None))
        run.say(f"{len(built)} of {len(combos)} designs hold their rules exactly with random weights")

        run.stage = "design"
        hidden = 32
        plans = []
        for rules in built:
            params = count(make_model(task, rules, hidden))
            plans.append({"rules": rules, "hidden": hidden, "params": params,
                          "true": [r for r in rules if r in task["holds"]], "untrue": [r for r in rules if r not in task["holds"]]})
        biggest = max([p["params"] for p in plans] or [0])
        plain = make_plain(task, at_least=biggest)
        T["baseline"] = {"params": count(plain), "kind": "GRU" if task.get("plain") == "gru" else "MLP on the flattened input"}
        run.say(f"{len(plans)} candidates designed; the conventional network has {T['baseline']['params']:,} weights",
                "It is never smaller than the candidate it is compared with.")

        run.stage = "measure"
        run.say("Generating mock data from the task's equations and training", f"sizes {B['sizes']}, {B['seeds']} seeds, {steps} steps")
        # the conventional network is the better of two: an MLP with ReLU and one with tanh (one GRU for a
        # sequence), so a candidate cannot win merely by having the activation that suits the task
        base, keep_plain = None, {}
        for act in (["gru"] if task.get("plain") == "gru" else ["relu", "tanh"]):
            keep_one: Dict[str, Any] = {}
            got = measure(task, lambda: make_plain(task, at_least=biggest, act=act), B["sizes"], B["seeds"], steps,
                          stop=run.stop.is_set, keep=keep_one)
            if got.get("stopped"):
                return
            if base is None or got["rows"][-1]["error"] < base["rows"][-1]["error"]:
                base, keep_plain = got, keep_one
                if act != "gru":
                    T["baseline"]["kind"] = f"MLP on the flattened input ({'ReLU' if act == 'relu' else 'tanh'}; the better of the two tried)"
        T["baseline"]["rows"] = base["rows"]
        T["baseline"]["ms"] = latency_ms(keep_plain["model"], task)
        T["baseline"]["breaks"] = {r: violation(r, keep_plain["model"], task) for r in allowed_rules(task)}
        run.say(f"Conventional network: error {base['rows'][-1]['error']:.4g} with {B['sizes'][-1]} examples")
        kept: Dict[int, Any] = {}
        for i, plan in enumerate(plans):
            if run.stop.is_set():
                return
            keep: Dict[str, Any] = {}
            got = measure(task, lambda: make_model(task, plan["rules"], plan["hidden"]), B["sizes"], B["seeds"], steps,
                          stop=run.stop.is_set, keep=keep)
            if got.get("stopped"):
                return
            cmp_ = compare(got["rows"], base["rows"])
            entry = {**plan, "rows": got["rows"], **cmp_, "ms": latency_ms(keep["model"], task),
                     "holds": {r: violation(r, keep["model"], task) for r in plan["rules"]}}
            T["candidates"].append(entry)
            kept[i] = keep["model"]
            run.say(f"{' + '.join(plan['rules'])}: error {got['rows'][-1]['error']:.4g}, "
                    f"{cmp_['gain']}x the conventional network's accuracy on average",
                    f"won at sizes {cmp_['wins'] or 'none'}; lost at {cmp_['losses'] or 'none'}")

        # the winner: must win beyond the spread at half the sizes or more, then the largest average gain
        need = math.ceil(len(B["sizes"]) / 2)
        eligible = [i for i, c in enumerate(T["candidates"]) if len(c["wins"]) >= need and not c["losses"]]
        T["winner"] = max(eligible, key=lambda i: T["candidates"][i]["gain"]) if eligible else None

        run.stage = "slim"
        if T["winner"] is not None:
            win = T["candidates"][T["winner"]]
            target = base["rows"][-1]["error"] - base["rows"][-1]["spread"]
            slim = {"hidden": win["hidden"], "params": win["params"], "error": win["rows"][-1]["error"], "ms": win["ms"]}
            best_model = kept[T["winner"]]
            for h in (16, 8, 4):
                if run.stop.is_set():
                    return
                keep = {}
                got = measure(task, lambda: make_model(task, win["rules"], h), [B["sizes"][-1]], B["seeds"], steps,
                              stop=run.stop.is_set, keep=keep)
                if got.get("stopped"):
                    return
                row = got["rows"][0]
                if row["error"] + row["spread"] < target:
                    slim = {"hidden": h, "params": count(keep["model"]), "error": row["error"], "ms": latency_ms(keep["model"], task)}
                    best_model = keep["model"]
                else:
                    break
            slim["smaller"] = round(T["baseline"]["params"] / slim["params"], 1)
            slim["faster"] = round(T["baseline"]["ms"] / max(slim["ms"], 1e-6), 2)
            T["slim"] = slim
            torch.save({"state": best_model.state_dict(), "task": tid, "rules": win["rules"], "hidden": slim["hidden"]},
                       run.home / f"{run.id}-{tid}.pt")
            run.say(f"Slimmed to {slim['params']:,} weights ({slim['smaller']}x smaller than the conventional network) "
                    "and still ahead of it", f"hidden width {slim['hidden']}; one pass over 1,000 examples takes {slim['ms']} ms")
        else:
            run.say("No candidate beat the conventional network beyond the spread across seeds", "Nothing to slim.")
        try:
            run.save()
        except OSError:
            pass

    # ---- explain -----------------------------------------------------------------------------
    run.stage = "explain"
    import superexplain
    run.say("Explaining each finding: what it means and how far it reaches")
    D["explain"] = superexplain.explain({**D, "goal": run.goal, "source": run.source}, ask=_ask if live else None)

    # ---- report ------------------------------------------------------------------------------
    run.stage = "report"
    D.update(verdict_of(D))
    run.say(D["verdict"])


def _few(ns: List[int]) -> str:
    ns = [str(n) for n in ns]
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]


def verdict_of(D: Dict[str, Any]) -> Dict[str, Any]:
    """The plain sentences at the top of the report, worked out from the numbers and nothing else."""
    lines, won = [], 0
    for T in D["tasks"]:
        if T.get("winner") is None:
            best = max(T["candidates"], key=lambda c: c["gain"]) if T["candidates"] else None
            where = ""
            if best and best["wins"]:
                where = (f"; it was ahead with {_few(best['wins'])} examples"
                         + (f" and behind with {_few(best['losses'])}" if best["losses"] else " but not at enough sizes"))
            lines.append(f"{T['name']}: no rule-built network stayed ahead of the conventional one"
                         + (f" (the closest, {' + '.join(best['rules'])}, averaged {best['gain']}x{where})." if best else "."))
            continue
        won += 1
        c, s = T["candidates"][T["winner"]], T["slim"]
        saving = (f" It reaches with {c['reaches_at']} examples what the conventional network needs {T['sizes'][-1]} for."
                  if c.get("reaches_at") and c["reaches_at"] < T["sizes"][-1] else "")
        speed = (f"{s['faster']}x faster to run" if s["faster"] >= 1.15 else
                 f"{round(1 / max(s['faster'], 1e-6), 1)}x slower to run" if s["faster"] <= 0.87 else "about as fast to run")
        lines.append(f"{T['name']}: built on {' + '.join(c['rules'])}, {c['gain']}x more accurate on average, "
                     f"{s['smaller']}x smaller and {speed}.{saving}")
    reading = ("No paper was read: the search could not be reached, so the rules came from this lab's own library."
               if not D.get("reachable") else
               f"{D.get('papers_read', 0)} papers were looked at and {len(D['cards'])} rule cards came back"
               + (" (matched by vocabulary: no model is connected)." if D.get("mode") != "live" else "."))
    head = (f"A rule-built network won on {won} of {len(D['tasks'])} mock tasks." if won else
            "No rule-built network won on these mock tasks.")
    return {"verdict": head, "lines": lines, "reading": reading, "won": won}


# ---------------------------------------------------------------------------
# what a finished run hands over
# ---------------------------------------------------------------------------

def _winner(run_id: str, task_id: str = ""):
    d = read(run_id)
    for T in d.get("tasks") or []:
        if T.get("winner") is not None and (not task_id or T["id"] == task_id):
            return d, T, T["candidates"][T["winner"]]
    raise ValueError("This run has no winning model to hand over.")


def to_canvas(run_id: str, task_id: str = "") -> Dict[str, Any]:
    """The winning network as a design: Input → RuleNet (or SeqNet) → Output."""
    d, T, c = _winner(run_id, task_id)
    task = TASKS[T["id"]]()
    hidden = T["slim"]["hidden"]
    if task["form"] == "sequence":
        block = {"type": "SeqNet", "params": {"rule": c["rules"][0], "out": task["out"], "hidden": hidden, "slots": 4}}
    else:
        block = {"type": "RuleNet", "params": {"rules": ", ".join(c["rules"]), "out": task["out"] or 1, "hidden": hidden,
                                               "k": task["k"], "momenta": bool(task["momenta"]),
                                               "signs": ", ".join(str(s) for s in (task.get("signs") or []))}}
    out_task = "classification" if task["loss"] == "ce" else "regression"
    nodes = [{"id": "in", "type": "Input", "params": {"shape": task["shape"]}, "x": 80, "y": 200},
             {"id": "net", **block, "x": 340, "y": 200},
             {"id": "out", "type": "Output", "params": {"task": out_task}, "x": 620, "y": 200}]
    name = "Rule" + "".join(w.capitalize() for w in T["id"].split("_"))
    return {"name": name, "nodes": nodes,
            "edges": [{"id": "e0", "source": "in", "target": "net", "port": 0}, {"id": "e1", "source": "net", "target": "out", "port": 0}],
            "note": f"Built by the super agent for: {d['goal'][:160]}"}


def model_file(run_id: str, task_id: str = "") -> str:
    """A standalone Python file: the class, how this model is constructed, and how to load its weights."""
    import rulesource
    d, T, c = _winner(run_id, task_id)
    task = TASKS[T["id"]]()
    s = T["slim"]
    if task["form"] == "sequence":
        cls, ctor = "SeqNet", f"SeqNet({task['shape'][1]}, {task['out']}, {c['rules'][0]!r}, hidden={s['hidden']})"
    else:
        cls = "RuleNet"
        ctor = (f"RuleNet({task['shape']}, out={task['out'] or 1}, rules={c['rules']!r}, hidden={s['hidden']}, k={task['k']}, "
                f"momenta={bool(task['momenta'])}, signs={task.get('signs')!r})")
    base = T["baseline"]["rows"][-1]["error"]
    return f'''"""{T["name"]}: a network built on {" + ".join(c["rules"])}.

Made by the super agent in Deep Network Designer.
Goal: {d["goal"][:300]}

Measured on mock data generated from the task's own equations ({T["measure"]}):
  this model        {s["error"]:.4g}   with {s["params"]:,} weights
  conventional      {base:.4g}   with {T["baseline"]["params"]:,} weights   ({T["baseline"]["kind"]})
Each rule is structure: it holds for any value of the weights.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

{rulesource.source(cls)}

def build():
    return {ctor}


if __name__ == "__main__":
    model = build()
    # weights saved by the run: {run_id}-{T["id"]}.pt, in the workspace's superruns folder
    # model.load_state_dict(torch.load("{run_id}-{T["id"]}.pt")["state"])
    x = torch.randn(4, *{task["shape"]})
    print(model(x).shape, sum(p.numel() for p in model.parameters()), "weights")
'''
