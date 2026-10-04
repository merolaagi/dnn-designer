"""Hamiltonian and port-Hamiltonian layers, and the measurement of what they buy.

A Hamiltonian network (Greydanus, Dzamba & Yosinski, 2019) does not learn the
motion directly. It learns one scalar, the energy H(q, p), and the motion is
read off its gradient:

    dq/dt =  ∂H/∂p        dp/dt = −∂H/∂q

Along any such motion dH/dt = ∂H/∂q·∂H/∂p − ∂H/∂p·∂H/∂q = 0, for every value
of the weights. Conservation is not learned from data; it is a property of the
architecture.

The port-Hamiltonian version adds friction that can only remove energy:

    dp/dt = −∂H/∂q − γ ∂H/∂p,    γ = softplus(r) ≥ 0

so dH/dt = −γ |∂H/∂p|² ≤ 0, again for every value of the weights. A plain
network trained on a damped system can, and does, put energy back in.

Two things this module is careful to say, because they are easy to overclaim:

    What is conserved is the network's *own* H. The true energy is conserved
    only as well as the learned H matches it.

    Continuous-time conservation survives integration only approximately. A
    numerical integrator adds its own error; RK4 with a small step adds very
    little, and the report separates that from model error.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

#: The layer, as source, because the palette block emits it into generated
#: code. The experiment below executes this same text, so what is measured is
#: exactly what goes on the canvas.
SOURCE = '''
class HamiltonianField(nn.Module):
    """Learns an energy H(z) and returns the motion it implies.

    z = (q, p), split in half. mode "field" returns dz/dt; mode "flow"
    integrates that field with RK4 for t_end and returns the new state.
    With dissipative=True a learned friction γ ≥ 0 acts on the momenta, so
    the energy can fall and can never rise.
    """
    def __init__(self, dim, hidden=64, dissipative=False, mode="field",
                 steps=4, t_end=1.0):
        super().__init__()
        if dim % 2:
            raise ValueError(f"HamiltonianField needs an even width, got {dim}")
        self.n = dim // 2
        self.energy = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1))
        self.dissipative = bool(dissipative)
        if self.dissipative:
            self.raw_friction = nn.Parameter(torch.full((self.n,), -2.0))
        self.mode, self.steps, self.t_end = mode, int(steps), float(t_end)

    def friction(self):
        return F.softplus(self.raw_friction) if self.dissipative else None

    def hamiltonian(self, z):
        return self.energy(z).squeeze(-1)

    def field(self, z):
        keep = torch.is_grad_enabled()
        with torch.enable_grad():
            if not z.requires_grad:
                z = z.clone().requires_grad_(True)
            grad = torch.autograd.grad(self.hamiltonian(z).sum(), z,
                                       create_graph=keep)[0]
        dq, dp = grad[..., :self.n], grad[..., self.n:]
        q_dot, p_dot = dp, -dq
        if self.dissipative:
            p_dot = p_dot - self.friction() * dp
        out = torch.cat([q_dot, p_dot], dim=-1)
        return out if keep else out.detach()

    def forward(self, z):
        if self.mode == "field":
            return self.field(z)
        h = self.t_end / self.steps
        for _ in range(self.steps):
            k1 = self.field(z)
            k2 = self.field(z + 0.5 * h * k1)
            k3 = self.field(z + 0.5 * h * k2)
            k4 = self.field(z + h * k3)
            z = z + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return z
'''


COUPLED_SOURCE = '''
class CoupledHamiltonian(nn.Module):
    """Subsystems joined so that, together, they can only lose energy.

    The state is the subsystems' states side by side; `parts` gives each
    one's width (each even: positions then momenta). The energy is each
    subsystem's own plus an optional interaction term. The motion is

        dz/dt = (J − R) ∇H,

    where J is the canonical structure of every subsystem plus a learned
    coupling A − Aᵀ — skew-symmetric by construction, so it moves energy
    between subsystems and never creates any — and R is a learned friction,
    one non-negative coefficient per subsystem, acting on its momenta.
    Since ∇HᵀJ∇H = 0 for any skew J, dH/dt = −∇HᵀR∇H ≤ 0 for every weight.
    """
    def __init__(self, parts, hidden=32, interaction=True, mode="field",
                 steps=4, t_end=1.0):
        super().__init__()
        self.parts = [int(p) for p in parts]
        if any(p % 2 for p in self.parts):
            raise ValueError(f"every subsystem needs an even width: {self.parts}")
        self.dim = sum(self.parts)
        self.energies = nn.ModuleList(
            nn.Sequential(nn.Linear(p, hidden), nn.Tanh(),
                          nn.Linear(hidden, hidden), nn.Tanh(),
                          nn.Linear(hidden, 1)) for p in self.parts)
        self.interaction = nn.Sequential(
            nn.Linear(self.dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)) if interaction else None
        self.coupling = nn.Parameter(torch.zeros(self.dim, self.dim))
        self.raw_friction = nn.Parameter(torch.full((len(self.parts),), -3.0))
        self.mode, self.steps, self.t_end = mode, int(steps), float(t_end)
        canonical = torch.zeros(self.dim, self.dim)
        momenta = torch.zeros(self.dim)
        at = 0
        for p in self.parts:
            n = p // 2
            canonical[at:at + n, at + n:at + p] = torch.eye(n)
            canonical[at + n:at + p, at:at + n] = -torch.eye(n)
            momenta[at + n:at + p] = 1.0
            at += p
        self.register_buffer("canonical", canonical)
        self.register_buffer("momenta", momenta)

    def friction(self):
        return F.softplus(self.raw_friction)

    def hamiltonian(self, z):
        total, at = 0, 0
        for p, energy in zip(self.parts, self.energies):
            total = total + energy(z[..., at:at + p]).squeeze(-1)
            at += p
        if self.interaction is not None:
            total = total + self.interaction(z).squeeze(-1)
        return total

    def structure(self):
        J = self.canonical + self.coupling - self.coupling.T
        gamma = torch.cat([self.friction()[i].expand(p)
                           for i, p in enumerate(self.parts)])
        R = torch.diag(gamma * self.momenta)
        return J, R

    def field(self, z):
        keep = torch.is_grad_enabled()
        with torch.enable_grad():
            if not z.requires_grad:
                z = z.clone().requires_grad_(True)
            grad = torch.autograd.grad(self.hamiltonian(z).sum(), z,
                                       create_graph=keep)[0]
        J, R = self.structure()
        out = grad @ (J - R).T
        return out if keep else out.detach()

    def forward(self, z):
        if self.mode == "field":
            return self.field(z)
        h = self.t_end / self.steps
        for _ in range(self.steps):
            k1 = self.field(z)
            k2 = self.field(z + 0.5 * h * k1)
            k3 = self.field(z + 0.5 * h * k2)
            k4 = self.field(z + h * k3)
            z = z + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return z
'''


def coupled_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    space = {"torch": torch, "nn": nn, "F": F, "math": math}
    exec(COUPLED_SOURCE, space)  # noqa: S102 - our own source, as the block emits it
    return space["CoupledHamiltonian"]


def coupled_learnables(parts, hidden: int, interaction: bool) -> int:
    dim = sum(parts)
    count = sum(p * hidden + hidden + hidden * hidden + hidden + hidden + 1
                for p in parts)
    if interaction:
        count += dim * hidden + hidden + hidden + 1
    return count + dim * dim + len(parts)


def coupled_experiment(spring: float = 0.5, damping: float = 0.1,
                       train_steps: int = 1500, hidden: int = 32,
                       horizon: float = 60.0, seed: int = 0) -> Dict[str, Any]:
    """Two pendulums joined by a spring, friction on the second only.

    The test of the structure is not only that energy never rises, but that
    the learned friction lands in the right subsystem — the composition keeps
    each part's identity, so its parameters mean something.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    def truth(z):
        q1, p1, q2, p2 = z[..., 0:1], z[..., 1:2], z[..., 2:3], z[..., 3:4]
        f = spring * (q1 - q2)
        return torch.cat([p1, -torch.sin(q1) - f,
                          p2, -torch.sin(q2) + f - damping * p2], dim=-1)

    def energy(z):
        q1, p1, q2, p2 = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
        return (0.5 * p1 ** 2 + 1 - torch.cos(q1) + 0.5 * p2 ** 2
                + 1 - torch.cos(q2) + 0.5 * spring * (q1 - q2) ** 2)

    starts = (torch.rand(60, 4) * 2 - 1) * torch.tensor([1.2, 0.8, 1.2, 0.8])
    with torch.no_grad():
        states = _rk4(truth, starts, 0.1, 60).reshape(-1, 4)
        targets = truth(states)

    Coupled = coupled_class()
    model = Coupled([2, 2], hidden)
    plain = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64),
                          nn.Tanh(), nn.Linear(64, 4))

    def fit(net, call):
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(train_steps):
            loss = torch.mean((call(states) - targets) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()

    started = time.time()
    fit(plain, plain)
    fit(model, model.field)

    test = torch.tensor([[1.0, 0.0, -0.5, 0.0], [0.3, 0.5, 0.8, -0.2],
                         [-0.9, 0.2, 0.4, 0.6]])
    steps = int(horizon / 0.1)
    with torch.no_grad():
        exact = _rk4(truth, test, 0.1, steps)
        roll_plain = _rk4(plain, test, 0.1, steps)
    roll_model = _rk4(lambda z: model.field(z), test, 0.1, steps).detach()

    def rises(path):
        e = energy(path)
        return float((e[1:] > e[:-1] + 1e-7).float().mean())

    probe = (torch.rand(4000, 4) * 2 - 1) * 2
    with torch.enable_grad():
        z = probe.clone().requires_grad_(True)
        gradH = torch.autograd.grad(model.hamiltonian(z).sum(), z)[0]
    rate = (gradH * model.field(probe).detach()).sum(-1)
    friction = model.friction().detach().tolist()
    return {
        "seconds": round(time.time() - started, 1),
        "true_friction": [0.0, damping],
        "learned_friction": friction,
        "own_energy_rate_max": float(rate.max()),
        "true_energy_rose": {"plain": rises(roll_plain),
                             "coupled": rises(roll_model)},
        "state_error": {"plain": float((roll_plain - exact).norm(dim=-1).mean()),
                        "coupled": float((roll_model - exact).norm(dim=-1).mean())},
        "parameters": {"plain": sum(p.numel() for p in plain.parameters()),
                       "coupled": sum(p.numel() for p in model.parameters())},
    }


def layer_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    space = {"torch": torch, "nn": nn, "F": F, "math": math}
    exec(SOURCE, space)  # noqa: S102 - our own source, the same text the block emits
    return space["HamiltonianField"]


def learnables(dim: int, hidden: int, dissipative: bool) -> int:
    count = dim * hidden + hidden + hidden * hidden + hidden + hidden + 1
    return count + (dim // 2 if dissipative else 0)


# --------------------------------------------------------------------------
# the experiment: a pendulum, ideal and damped
# --------------------------------------------------------------------------

def _truth(damping: float):
    """q̇ = p, ṗ = −sin q − γp, with energy H = p²/2 + (1 − cos q)."""
    import torch

    def field(z):
        q, p = z[..., :1], z[..., 1:]
        return torch.cat([p, -torch.sin(q) - damping * p], dim=-1)

    def energy(z):
        q, p = z[..., 0], z[..., 1]
        return 0.5 * p ** 2 + (1 - torch.cos(q))

    return field, energy


def _rk4(field, z, dt, steps):
    import torch

    path = [z]
    for _ in range(steps):
        k1 = field(z)
        k2 = field(z + 0.5 * dt * k1)
        k3 = field(z + 0.5 * dt * k2)
        k4 = field(z + dt * k3)
        z = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        path.append(z)
    return torch.stack(path)


def experiment(damping: float = 0.0, train_steps: int = 2000,
               hidden: int = 64, horizon: float = 60.0, seed: int = 0,
               noise: float = 0.0) -> Dict[str, Any]:
    """Train a plain network and a (port-)Hamiltonian one on the same data,
    then roll both far past the training window and measure the true energy.

    The training trajectories last 6 time units; the rollout lasts `horizon`.
    Errors that are invisible over the training window compound over a long
    one, which is exactly where a guarantee earns its keep.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    field, energy = _truth(damping)
    Hamiltonian = layer_class()

    # oscillations below the separatrix (H < 2), so every orbit is bounded
    q0 = (torch.rand(40, 1) * 2 - 1) * 1.6
    p0 = (torch.rand(40, 1) * 2 - 1) * 1.0
    starts = torch.cat([q0, p0], dim=1)
    with torch.no_grad():
        paths = _rk4(field, starts, 0.1, 60)
    states = paths.reshape(-1, 2)
    with torch.no_grad():
        targets = field(states)
    if noise:
        targets = targets + noise * torch.randn_like(targets)

    dissipative = damping > 0
    hnn = Hamiltonian(2, hidden, dissipative=dissipative)
    plain = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(),
                          nn.Linear(hidden, hidden), nn.Tanh(),
                          nn.Linear(hidden, 2))

    def fit(model, call):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(train_steps):
            loss = torch.mean((call(states) - targets) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss.detach())

    started = time.time()
    loss_plain = fit(plain, plain)
    loss_hnn = fit(hnn, hnn.field)

    test = torch.tensor([[1.2, 0.0], [0.5, 0.6], [-1.0, 0.4], [0.2, -0.9],
                         [1.5, 0.2], [-0.7, -0.7]])
    dt = 0.1
    steps = int(horizon / dt)
    with torch.no_grad():
        exact = _rk4(field, test, dt, steps)
    with torch.no_grad():
        roll_plain = _rk4(plain, test, dt, steps)
    roll_hnn = _rk4(lambda z: hnn.field(z), test, dt, steps).detach()

    def report(path):
        e = energy(path)
        e0 = energy(test)
        drift = (e[-1] - e0).abs() / e0.clamp_min(1e-9)
        rises = (e[1:] > e[:-1] + 1e-7).float().mean()
        err = (path - exact).norm(dim=-1).mean()
        return {"final_energy_drift": float(drift.mean()),
                "energy_rising_fraction": float(rises),
                "state_error": float(err),
                "energy": [float(v) for v in e[:, 0][::max(1, steps // 200)]]}

    out = {
        "damping": damping, "horizon": horizon, "train_window": 6.0,
        "seconds": round(time.time() - started, 1),
        "true": {"energy": [float(v) for v in
                            energy(exact)[:, 0][::max(1, steps // 200)]]},
        "plain": {**report(roll_plain), "fit": loss_plain,
                  "parameters": sum(p.numel() for p in plain.parameters())},
        "hamiltonian": {**report(roll_hnn), "fit": loss_hnn,
                        "parameters": sum(p.numel() for p in hnn.parameters())},
    }

    # The guarantee itself, checked on states the network never saw: the
    # network's own energy must not rise anywhere along its own motion.
    probe = (torch.rand(5000, 2) * 2 - 1) * torch.tensor([2.5, 2.0])
    with torch.enable_grad():
        z = probe.clone().requires_grad_(True)
        H = hnn.hamiltonian(z)
        gradH = torch.autograd.grad(H.sum(), z)[0]
    dHdt = (gradH * hnn.field(probe).detach()).sum(-1)
    out["hamiltonian"]["own_energy_rate_max"] = float(dHdt.max())
    if dissipative:
        out["hamiltonian"]["learned_friction"] = float(hnn.friction()[0])
    return out
