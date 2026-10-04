"""Predictive coding: inference as energy minimisation, learning by local rules.

Rao & Ballard (1999); Whittington & Bogacz, *An Approximation of the Error
Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian
Synaptic Plasticity* (Neural Computation, 2017).

Each layer predicts the next: μ_{l+1} = W_l g(x_l) + b_l. The mismatch at each
layer is an error, ε_l = x_l − μ_l, and the network's energy is

    F = Σ_l ½ ‖ε_l‖².

With the input and the target clamped at the two ends, inference relaxes the
hidden activity downhill on F:

    x_l ← x_l − η (ε_l − g'(x_l) ⊙ W_lᵀ ε_{l+1})

and learning changes each weight using only what is at its two ends — the
activity below and the error above:

    ΔW_l ∝ ε_{l+1} g(x_l)ᵀ.

No error is carried backwards through the network; each update is local.
That is the interest, for neuroscience and for hardware that cannot run a
global backward pass. It is not a speed-up in software: every training step
pays for the relaxation, and that cost is measured below rather than hidden.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List


class PredictiveCoding:
    """A layered network trained by predictive coding, written out by hand so
    that nothing global is used where the method says only local is."""

    def __init__(self, sizes: List[int], seed: int = 0):
        import torch

        g = torch.Generator().manual_seed(seed)
        self.sizes = sizes
        self.W = [torch.randn(sizes[l + 1], sizes[l], generator=g)
                  / math.sqrt(sizes[l]) for l in range(len(sizes) - 1)]
        self.b = [torch.zeros(sizes[l + 1]) for l in range(len(sizes) - 1)]

    @staticmethod
    def g(x, layer):
        import torch

        return x if layer == 0 else torch.tanh(x)

    @staticmethod
    def dg(x, layer):
        import torch

        return torch.ones_like(x) if layer == 0 else 1 - torch.tanh(x) ** 2

    def forward(self, x):
        values = [x]
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            values.append(self.g(values[-1], l) @ W.T + b)
        return values

    def energy(self, values):
        total = 0.0
        for l in range(1, len(values)):
            mu = self.g(values[l - 1], l - 1) @ self.W[l - 1].T + self.b[l - 1]
            total = total + 0.5 * ((values[l] - mu) ** 2).sum(1).mean()
        return float(total)

    def errors(self, values):
        return [None] + [
            values[l] - (self.g(values[l - 1], l - 1) @ self.W[l - 1].T
                         + self.b[l - 1])
            for l in range(1, len(values))]

    def relax(self, x, y, steps: int, rate: float = 0.1, trace=None):
        values = self.forward(x)
        values[-1] = y.clone()
        for _ in range(steps):
            eps = self.errors(values)
            for l in range(1, len(values) - 1):
                push = (eps[l + 1] @ self.W[l]) * self.dg(values[l], l)
                values[l] = values[l] - rate * (eps[l] - push)
            if trace is not None:
                trace.append(self.energy(values))
        return values

    def updates(self, values):
        """The local rule: activity below times error above, nothing else."""
        eps = self.errors(values)
        n = values[0].shape[0]
        return ([eps[l + 1].T @ self.g(values[l], l) / n
                 for l in range(len(self.W))],
                [eps[l + 1].mean(0) for l in range(len(self.W))])

    def train_step(self, x, y, steps: int, lr: float):
        values = self.relax(x, y, steps)
        dW, db = self.updates(values)
        for l in range(len(self.W)):
            self.W[l] += lr * dW[l]
            self.b[l] += lr * db[l]

    def predict(self, x):
        return self.forward(x)[-1]


def backprop_gradients(net: PredictiveCoding, x, y):
    """What backpropagation would do with the same weights and batch: the
    negative gradient of ½‖y − ŷ‖², so it points the way the update moves."""
    import torch

    W = [w.clone().requires_grad_(True) for w in net.W]
    b = [v.clone().requires_grad_(True) for v in net.b]
    h = x
    for l in range(len(W)):
        h = PredictiveCoding.g(h, l) @ W[l].T + b[l]
    loss = 0.5 * ((h - y) ** 2).sum(1).mean()
    grads = torch.autograd.grad(loss, W)
    return [-g for g in grads]


def _data(n, generator):
    import torch

    x = torch.randn(n, 8, generator=generator)
    teacher = torch.randn(4, 8, generator=torch.Generator().manual_seed(99))
    return x, torch.tanh(x @ teacher.T) + 0.3 * torch.sin(2 * x[:, :4])


def experiment(sizes=(8, 32, 32, 4), seed: int = 0, epochs: int = 300,
               relax_steps: int = 5, pc_lr: float = 0.5) -> Dict[str, Any]:
    """Five relaxation steps because that is where the updates aligned best
    with backpropagation — more relaxation aligned worse. A learning rate ten
    times backpropagation's because the local updates are much shorter: the
    prediction errors shrink as they relax towards the input, measured at 7%,
    32% and 72% of backpropagation's length layer by layer. At the same rate
    predictive coding reached 0.146 against 0.034; at ten times it reached
    0.038."""
    import torch

    g = torch.Generator().manual_seed(seed)
    x, y = _data(512, g)
    x_test, y_test = _data(1000, g)

    # 1. inference lowers the energy, step by step
    net = PredictiveCoding(list(sizes), seed)
    trace: List[float] = []
    net.relax(x[:64], y[:64], 60, trace=trace)
    worst_rise = max(b - a for a, b in zip(trace, trace[1:]))

    # 2. the local updates against backpropagation's gradients, by how long
    #    the network was allowed to relax
    alignment = []
    bp = backprop_gradients(net, x[:64], y[:64])
    for steps in (1, 5, 20, 100):
        dW, _ = net.updates(net.relax(x[:64], y[:64], steps))
        cos = [float(torch.nn.functional.cosine_similarity(
            a.reshape(1, -1), b.reshape(1, -1))) for a, b in zip(dW, bp)]
        length = [float(a.norm() / b.norm()) for a, b in zip(dW, bp)]
        alignment.append({"relax_steps": steps, "cosine_by_layer": cos,
                          "length_by_layer": length})

    # 3. training, both ways, same network and data
    pc = PredictiveCoding(list(sizes), seed)
    started = time.time()
    for _ in range(epochs):
        for i in range(0, 512, 64):
            pc.train_step(x[i:i + 64], y[i:i + 64], relax_steps, lr=pc_lr)
    pc_time = time.time() - started

    model = PredictiveCoding(list(sizes), seed)
    params = [w.clone().requires_grad_(True) for w in model.W] + \
             [v.clone().requires_grad_(True) for v in model.b]
    L = len(model.W)
    opt = torch.optim.SGD(params, lr=0.05)
    started = time.time()
    for _ in range(epochs):
        for i in range(0, 512, 64):
            h = x[i:i + 64]
            for l in range(L):
                h = PredictiveCoding.g(h, l) @ params[l].T + params[L + l]
            loss = 0.5 * ((h - y[i:i + 64]) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    bp_time = time.time() - started
    with torch.no_grad():
        h = x_test
        for l in range(L):
            h = PredictiveCoding.g(h, l) @ params[l].T + params[L + l]
        bp_err = float(((h - y_test) ** 2).mean())
    pc_err = float(((pc.predict(x_test) - y_test) ** 2).mean())
    baseline = float(((y_test - y_test.mean(0)) ** 2).mean())

    return {"energy_trace": trace[:30], "worst_energy_rise": worst_rise,
            "alignment": alignment,
            "training": {"predictive_coding": {"test_error": pc_err / baseline,
                                               "seconds": round(pc_time, 1)},
                         "backprop": {"test_error": bp_err / baseline,
                                      "seconds": round(bp_time, 1)},
                         "relax_steps": relax_steps, "epochs": epochs,
                         "pc_lr": pc_lr, "bp_lr": 0.05}}
