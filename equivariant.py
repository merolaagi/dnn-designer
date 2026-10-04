"""A layer that respects rotation, reflection, translation and relabelling.

Satorras, Hoogeboom & Welling, *E(n) Equivariant Graph Neural Networks*
(ICML 2021). The input is a set of N points, each with D coordinates and F
features. For every pair,

    m_ij = φ_e(h_i, h_j, ‖x_i − x_j‖²)                messages
    x_i' = x_i + (1/(N−1)) Σ_j (x_i − x_j) φ_x(m_ij)   coordinates
    h_i' = φ_h(h_i, Σ_j m_ij)                          features

A message sees only features and a squared distance, which no rotation,
reflection or translation changes. Coordinates move only along differences
between points, which rotate exactly as the points do. Sums over j do not
care about order. So for any orthogonal Q, translation t and permutation P:

    layer(P(xQᵀ + t), P h) = P(x'Qᵀ + t), P h'

exactly, for every weight — not learned, and not approximately. That is
checked numerically below, on an untrained layer.

What it buys is measured rather than assumed: a model that cannot tell two
rotated copies of the same configuration apart does not need to see both.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict

SOURCE = '''
class EquivariantLayer(nn.Module):
    """E(n)-equivariant message passing over all pairs of points.

    Input [N, coords + features]: each row a point. Output
    [N, coords + out_features]: coordinates moved equivariantly, features
    replaced invariantly.
    """
    def __init__(self, coords, features, hidden=32, out_features=16,
                 update_coords=True):
        super().__init__()
        self.d, self.f = int(coords), int(features)
        self.edge = nn.Sequential(
            nn.Linear(2 * self.f + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.coord = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)) if update_coords else None
        self.node = nn.Sequential(
            nn.Linear(self.f + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_features))

    def forward(self, z):
        x, h = z[..., :self.d], z[..., self.d:]
        B, N = x.shape[0], x.shape[1]
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        dist2 = (diff ** 2).sum(-1, keepdim=True)
        hi = h.unsqueeze(2).expand(B, N, N, self.f)
        hj = h.unsqueeze(1).expand(B, N, N, self.f)
        m = self.edge(torch.cat([hi, hj, dist2], dim=-1))
        mask = (1 - torch.eye(N, device=z.device, dtype=z.dtype)).unsqueeze(-1)
        m = m * mask
        if self.coord is not None:
            x = x + (diff * self.coord(m) * mask).sum(2) / max(N - 1, 1)
        h = self.node(torch.cat([h, m.sum(2)], dim=-1))
        return torch.cat([x, h], dim=-1)
'''


def layer_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    space = {"torch": torch, "nn": nn, "F": F, "math": math}
    exec(SOURCE, space)  # noqa: S102 - our own source, as the block emits it
    return space["EquivariantLayer"]


def learnables(features: int, hidden: int, out_features: int,
               update_coords: bool) -> int:
    edge = (2 * features + 1) * hidden + hidden + hidden * hidden + hidden
    coord = (hidden * hidden + hidden + hidden + 1) if update_coords else 0
    node = (features + hidden) * hidden + hidden + hidden * out_features + out_features
    return edge + coord + node


def random_orthogonal(d: int, generator=None):
    """A random rotation or reflection: the Q of a QR decomposition, with the
    sign fixed so it is uniformly distributed. Reflections are included on
    purpose — E(n) covers them, and a layer that only handled rotations would
    pass a weaker test."""
    import torch

    a = torch.randn(d, d, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r))


def symmetry_error(layer, points: int = 6, coords: int = 3, features: int = 4,
                   trials: int = 5, seed: int = 0) -> Dict[str, float]:
    """The largest disagreement over random rotations, reflections,
    translations and relabellings. Zero up to rounding, or the layer is not
    what it says it is."""
    import torch

    g = torch.Generator().manual_seed(seed)
    layer = layer.double()
    worst = {"coordinates": 0.0, "features": 0.0, "relabelling": 0.0}
    for _ in range(trials):
        x = torch.randn(2, points, coords, generator=g, dtype=torch.float64)
        h = torch.randn(2, points, features, generator=g, dtype=torch.float64)
        Q = random_orthogonal(coords, g)
        t = torch.randn(coords, generator=g, dtype=torch.float64) * 3
        with torch.no_grad():
            out = layer(torch.cat([x, h], -1))
            moved = layer(torch.cat([x @ Q.T + t, h], -1))
        expected_x = out[..., :coords] @ Q.T + t
        worst["coordinates"] = max(worst["coordinates"], float(
            (moved[..., :coords] - expected_x).abs().max()))
        worst["features"] = max(worst["features"], float(
            (moved[..., coords:] - out[..., coords:]).abs().max()))
        perm = torch.randperm(points, generator=g)
        with torch.no_grad():
            shuffled = layer(torch.cat([x, h], -1)[:, perm])
        worst["relabelling"] = max(worst["relabelling"], float(
            (shuffled - out[:, perm]).abs().max()))
    return worst


# --------------------------------------------------------------------------
# what symmetry buys: the energy of a spring system, from few examples
# --------------------------------------------------------------------------

def _energy(x):
    """Five particles, every pair joined by a spring of rest length 1."""
    import torch

    diff = x.unsqueeze(2) - x.unsqueeze(1)
    r = diff.norm(dim=-1)
    n = x.shape[1]
    iu = torch.triu_indices(n, n, 1)
    return ((r[:, iu[0], iu[1]] - 1) ** 2).sum(-1)


def experiment(train_sizes=(64, 512), points: int = 5, steps: int = 1500,
               seed: int = 0) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    Layer = layer_class()
    test_x = torch.randn(1000, points, 3)
    test_y = _energy(test_x)
    scale = float(test_y.var())

    class Invariant(nn.Module):
        def __init__(self):
            super().__init__()
            self.one = Layer(3, 1, 32, 32)
            self.two = Layer(3, 32, 32, 32)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            h = torch.ones(x.shape[0], x.shape[1], 1)
            z = self.two(self.one(torch.cat([x, h], -1)))
            return self.head(z[..., 3:].sum(1)).squeeze(-1)

    def plain():
        return nn.Sequential(nn.Flatten(), nn.Linear(points * 3, 128),
                             nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                             nn.Linear(128, 1))

    results = []
    started = time.time()
    for n in train_sizes:
        x = torch.randn(n, points, 3)
        y = _energy(x)
        row = {"train": n}
        for name, model in (("plain", plain()), ("equivariant", Invariant())):
            opt = torch.optim.Adam(model.parameters(), lr=2e-3)
            for _ in range(steps):
                loss = torch.mean((model(x).reshape(-1) - y) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                err = float(torch.mean((model(test_x).reshape(-1) - test_y) ** 2))
                # the same configurations, rotated: an invariant model must
                # give the same answer every time
                sample = test_x[:50]
                spins = torch.stack([
                    model(sample @ random_orthogonal(3).float().T).reshape(-1)
                    for _ in range(12)])
                spread = float(spins.std(0).mean())
            row[name] = {"test_error": err / scale,
                         "rotation_spread": spread,
                         "parameters": sum(p.numel() for p in model.parameters())}
        results.append(row)
    return {"rows": results, "seconds": round(time.time() - started, 1),
            "points": points}
