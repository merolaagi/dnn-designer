"""Finite basis physics-informed neural networks, and hard constraints.

Both from Moseley, *Physics-informed machine learning: from concepts to
real-world applications* (DPhil thesis, Oxford, 2022), chapter 6.

Hard constraints (§6.2.4, after Lagaris et al., 1998). Instead of a network
that approximates u and a loss that pleads with it to respect the boundary,
the boundary is written into the solution:

    û(x, t) = A(x, t) + t·(x − a)(b − x)·NN(x, t)

where A meets the initial and boundary conditions by itself and the second
term vanishes on all of them. The conditions are then satisfied exactly, for
every value of the network's weights, so they leave the loss — and so does the
competition between loss terms that makes PINN training stiff (Wang et al.).

FBPINNs (§6.4). PINNs scale badly with frequency: the network needs more
parameters, more points and more steps, and spectral bias means high
frequencies arrive last. FBPINNs split the domain into overlapping subdomains,
put a small network in each, normalise each network's input to [−1, 1] over
its own subdomain, and blend them with smooth windows:

    NN(x) = Σᵢ wᵢ(x) · unnorm(NNᵢ(normᵢ(x)))
    wᵢ(x) = Πⱼ φ((xⱼ − aᵢⱼ)/σ) · φ((bᵢⱼ − xⱼ)/σ),   φ the sigmoid

Each small network sees a slowly varying function in its own coordinates,
which is what spectral bias is good at. Nothing about the loss changes: the
physics is evaluated on the global coordinates, and gradients flow back
through each subdomain's normalisation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _mlp(inputs: int, width: int, depth: int, outputs: int = 1):
    import torch.nn as nn

    layers: List[Any] = [nn.Linear(inputs, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers.append(nn.Linear(width, outputs))
    return nn.Sequential(*layers)


def subdomains(lo: float, hi: float, n: int, overlap: float
               ) -> List[Dict[str, float]]:
    """n equal subdomains of [lo, hi], each overlapping its neighbours.

    a and b are the midpoints of the left and right overlap regions — where a
    window is at half height. The outer edges have no neighbour, so their
    window does not fall away there: an infinite a or b means "no edge".
    """
    cuts = [lo + (hi - lo) * k / n for k in range(n + 1)]
    out = []
    for i in range(n):
        out.append({
            "a": -math.inf if i == 0 else cuts[i],
            "b": math.inf if i == n - 1 else cuts[i + 1],
            # each network's input is normalised over the region it can see
            "lo": max(lo, cuts[i] - overlap / 2),
            "hi": min(hi, cuts[i + 1] + overlap / 2),
        })
    return out


class FBPINN:
    """A sum of windowed subdomain networks over a hyperrectangular grid."""

    def __new__(cls, *args, **kwargs):
        import torch.nn as nn

        class _FBPINN(nn.Module):
            def __init__(self, bounds: Sequence[Tuple[float, float]],
                         divisions: Sequence[int], overlap: Sequence[float],
                         width: int = 16, depth: int = 2,
                         scale: float = 1.0):
                super().__init__()
                import itertools

                per_dim = [subdomains(lo, hi, n, o) for (lo, hi), n, o
                           in zip(bounds, divisions, overlap)]
                self.cells = [list(c) for c in itertools.product(*per_dim)]
                # σ puts each window at φ(-4) ≈ 0.018 at the far edge of its
                # overlap — negligibly small outside, as §6.4.2 requires
                self.sigma = [o / 8 for o in overlap]
                self.reach = [o / 2 for o in overlap]
                self.scale = scale
                self.nets = nn.ModuleList(
                    _mlp(len(bounds), width, depth) for _ in self.cells)

            def window(self, x, cell):
                import torch

                w = torch.ones_like(x[:, :1])
                for j, part in enumerate(cell):
                    xj = x[:, j:j + 1]
                    if part["a"] != -math.inf:
                        w = w * torch.sigmoid((xj - part["a"]) / self.sigma[j])
                    if part["b"] != math.inf:
                        w = w * torch.sigmoid((part["b"] - xj) / self.sigma[j])
                return w

            def forward(self, x):
                """Each network is evaluated only where its window is not
                negligible (§6.4.4). Evaluating every network everywhere is
                the same answer at n times the cost: measured, thirty
                subdomains took 179 s for 2,000 steps that way."""
                import torch

                total = torch.zeros_like(x[:, :1])
                for net, cell in zip(self.nets, self.cells):
                    lo = torch.tensor([c["lo"] for c in cell], dtype=x.dtype)
                    hi = torch.tensor([c["hi"] for c in cell], dtype=x.dtype)
                    # half an overlap beyond the region the network is
                    # normalised over, where its window has fallen to
                    # sigmoid(-8) ≈ 3e-4 — small enough to stop there
                    reach = torch.tensor(self.reach, dtype=x.dtype)
                    inside = torch.all((x >= lo - reach) & (x <= hi + reach),
                                       dim=1)
                    idx = inside.nonzero().reshape(-1)
                    if idx.numel() == 0:
                        continue
                    xs = x.index_select(0, idx)
                    local = 2 * (xs - lo) / (hi - lo) - 1
                    part = self.window(xs, cell) * net(local) * self.scale
                    total = total.index_add(0, idx, part)
                return total

            def parameter_count(self) -> int:
                return sum(p.numel() for p in self.parameters())

        return _FBPINN(*args, **kwargs)


class Normalised:
    """A plain PINN with the same input normalisation, so a comparison
    between it and an FBPINN differs only in the decomposition."""

    def __new__(cls, bounds, width, depth, scale=1.0):
        import torch
        import torch.nn as nn

        class _Plain(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = _mlp(len(bounds), width, depth)
                self.register_buffer(
                    "lo", torch.tensor([b[0] for b in bounds]))
                self.register_buffer(
                    "hi", torch.tensor([b[1] for b in bounds]))
                self.scale = scale

            def forward(self, x):
                lo, hi = self.lo.to(x.dtype), self.hi.to(x.dtype)
                return self.net(2 * (x - lo) / (hi - lo) - 1) * self.scale

        return _Plain()


# --------------------------------------------------------------------------
# §6.3: the motivating experiment, which has an exact answer
# --------------------------------------------------------------------------

def motivating(omega: float, model_kind: str = "fbpinn", steps: int = 4000,
               width: int = 16, depth: int = 2, subdomain_count: int = 0,
               overlap: float = 0.0, seed: int = 0,
               progress=None) -> Dict[str, Any]:
    """du/dx = cos(ωx), u(0) = 0, x in [−2π, 2π]. Exactly u = sin(ωx)/ω.

    The ansatz û = tanh(ωx)·NN(x) satisfies u(0) = 0 for any network, so the
    loss is the residual alone (equation 6.7). Training points scale with the
    frequency, 200·ω, as in the thesis.
    """
    import torch

    torch.manual_seed(seed)
    lo, hi = -2 * math.pi, 2 * math.pi
    n_points = int(200 * max(1.0, omega))
    if model_kind == "fbpinn":
        n = subdomain_count or max(5, int(round(2 * omega)))
        o = overlap or (1.3 if omega <= 1 else 0.3)
        model = FBPINN([(lo, hi)], [n], [o], width, depth, scale=1 / omega)
        label = f"FBPINN, {n} subdomains of {depth}×{width}"
    else:
        model = Normalised([(lo, hi)], width, depth, scale=1 / omega)
        label = f"PINN, {depth}×{width}"

    x = torch.linspace(lo, hi, n_points).reshape(-1, 1)
    test = torch.linspace(lo, hi, 5000).reshape(-1, 1)
    exact = torch.sin(omega * test) / omega

    def solution(z):
        return torch.tanh(omega * z) * model(z)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    curve = []
    for step in range(1, steps + 1):
        z = x.clone().requires_grad_(True)
        u = solution(z)
        du = torch.autograd.grad(u, z, torch.ones_like(u), create_graph=True)[0]
        loss = torch.mean((du - torch.cos(omega * z)) ** 2)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % max(1, steps // 60) == 0 or step == 1:
            with torch.no_grad():
                l1 = float(torch.mean(torch.abs(solution(test) - exact)))
            curve.append({"step": step, "l1": l1, "loss": float(loss.detach())})
            if progress:
                progress(curve[-1])

    with torch.no_grad():
        final = solution(test)
        l1 = float(torch.mean(torch.abs(final - exact)))
        relative = l1 / float(torch.mean(torch.abs(exact)))
    thin = torch.linspace(0, 4999, 400).long()
    return {
        "label": label, "omega": omega,
        "parameters": sum(p.numel() for p in model.parameters()),
        "l1": l1, "relative_l1": relative, "curve": curve,
        "x": test[thin].reshape(-1).tolist(),
        "u": final[thin].reshape(-1).tolist(),
        "exact": exact[thin].reshape(-1).tolist(),
    }


# --------------------------------------------------------------------------
# §6.2.4: hard constraints for u(x, t) with fixed boundary values
# --------------------------------------------------------------------------

def hard_constrained(network, ic, left, right, x0: float, x1: float,
                     t0: float = 0.0, t1: float = 1.0,
                     sharpness: float = 0.1):
    """Wrap a network so the initial and boundary conditions hold exactly.

    û(x, t) = A(x, t) + tanh((t−t0)/τ)·tanh((x−x0)/s)·tanh((x1−x)/s)·NN(x, t),
    where

    A(x, t) = g(x) + [L(t) − L(t0)](x1 − x)/(x1 − x0)
                   + [R(t) − R(t0)](x − x0)/(x1 − x0).

    At t = t0 the correction terms vanish and A = g. At x = x0, A = g(x0) +
    L(t) − L(t0), which is L(t) provided the conditions agree at the corner —
    g(x0) = L(t0). That compatibility is checked rather than assumed: without
    it no smooth solution exists, and the ansatz would hide the conflict.
    """
    import torch
    import torch.nn as nn

    def value(fn, arg):
        v = fn(arg)
        return v if torch.is_tensor(v) else torch.full_like(arg, float(v))

    corner = torch.tensor([[x0], [x1]], dtype=torch.float64)
    t_zero = torch.full((1, 1), float(t0), dtype=torch.float64)
    gap_left = float(value(ic, corner[:1]) - value(left, t_zero))
    gap_right = float(value(ic, corner[1:]) - value(right, t_zero))
    if max(abs(gap_left), abs(gap_right)) > 1e-6:
        raise ValueError(
            f"The initial and boundary conditions disagree at a corner "
            f"(by {gap_left:.3g} on the left, {gap_right:.3g} on the right). "
            f"No smooth solution meets both, so they cannot be written into "
            f"the solution — use soft constraints, which will show the "
            f"conflict as a loss term that will not fall.")

    tau = sharpness * (t1 - t0)
    edge = sharpness * (x1 - x0)

    class _Constrained(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = network

        def forward(self, xt):
            x, t = xt[:, :1], xt[:, 1:2]
            span = x1 - x0
            base = (value(ic, x)
                    + (value(left, t) - value(left, torch.full_like(t, t0)))
                    * (x1 - x) / span
                    + (value(right, t) - value(right, torch.full_like(t, t0)))
                    * (x - x0) / span)
            # tanh rather than linear factors (§6.3): away from the boundary
            # they sit at ≈1, so the network need not compensate for them.
            # The linear version t·(x−a)(b−x) was measured at 20% error on
            # Burgers against 2.4% for soft constraints — the network spent
            # its capacity undoing the ansatz near t = 0 and at the walls.
            bubble = (torch.tanh((t - t0) / tau)
                      * torch.tanh((x - x0) / edge)
                      * torch.tanh((x1 - x) / edge))
            return base + bubble * self.inner(xt).reshape(-1, 1)

    return _Constrained()
