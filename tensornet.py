"""A weight matrix stored as a chain of small tensors.

Novikov, Podoprikhin, Osokin & Vetrov, *Tensorizing Neural Networks* (NeurIPS
2015). Write the input width as m₁m₂…m_d and the output width as n₁n₂…n_d.
The weight W[(i₁…i_d), (j₁…j_d)] is a product of d small cores,

    W = G₁[i₁, j₁] G₂[i₂, j₂] … G_d[i_d, j_d],   G_k of shape r_{k−1}×n_k×m_k×r_k

so it costs Σ r_{k−1} n_k m_k r_k parameters instead of Π n_k m_k. At full
rank any matrix is exact; below it, what survives depends entirely on whether
the matrix has structure across the modes — a random matrix does not compress
at all, and that is measured below, not assumed.

The honest rival is not the uncompressed layer. It is ordinary low-rank
compression, a truncated SVD of the same matrix at the same parameter count.
The experiment compresses one trained layer both ways and asks which leaves
the model more accurate.

This saves parameters, not arithmetic: the forward pass rebuilds the full
matrix, which is simple and exact. Contracting the cores against the input
directly is faster at large sizes and is not done here.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List

SOURCE = '''
class TensorTrainLinear(nn.Module):
    """A linear layer whose weight is a tensor train of rank `rank`."""
    def __init__(self, in_modes, out_modes, rank=4, bias=True):
        super().__init__()
        if len(in_modes) != len(out_modes):
            raise ValueError("in_modes and out_modes need the same length")
        self.in_modes = [int(m) for m in in_modes]
        self.out_modes = [int(n) for n in out_modes]
        d = len(self.in_modes)
        ranks = [1] + [int(rank)] * (d - 1) + [1]
        self.cores = nn.ParameterList()
        for k in range(d):
            fan = self.in_modes[k] * ranks[k]
            self.cores.append(nn.Parameter(
                torch.randn(ranks[k], self.out_modes[k], self.in_modes[k],
                            ranks[k + 1]) / math.sqrt(fan)))
        self.bias = nn.Parameter(torch.zeros(math.prod(self.out_modes))) if bias else None

    def weight(self):
        full = self.cores[0]
        for core in self.cores[1:]:
            full = torch.einsum("a...r,rnms->a...nms", full, core)
        d = len(self.cores)
        full = full.squeeze(0).squeeze(-1)
        order = [2 * k for k in range(d)] + [2 * k + 1 for k in range(d)]
        return full.permute(*order).reshape(math.prod(self.out_modes),
                                            math.prod(self.in_modes))

    def forward(self, x):
        out = x @ self.weight().T
        return out + self.bias if self.bias is not None else out
'''


def layer_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    space = {"torch": torch, "nn": nn, "F": F, "math": math}
    exec(SOURCE, space)  # noqa: S102 - our own source, as the block emits it
    return space["TensorTrainLinear"]


def learnables(in_modes: List[int], out_modes: List[int], rank: int,
               bias: bool) -> int:
    d = len(in_modes)
    ranks = [1] + [rank] * (d - 1) + [1]
    count = sum(ranks[k] * out_modes[k] * in_modes[k] * ranks[k + 1]
                for k in range(d))
    return count + (math.prod(out_modes) if bias else 0)


def tt_svd(W, out_modes: List[int], in_modes: List[int], rank: int):
    """Cores for W by successive truncated SVDs (Oseledets, 2011)."""
    import torch

    d = len(in_modes)
    T = W.reshape(*out_modes, *in_modes)
    order = [i for k in range(d) for i in (k, d + k)]
    T = T.permute(*order).contiguous()
    cores, r_prev = [], 1
    rest = T.reshape(r_prev * out_modes[0] * in_modes[0], -1)
    for k in range(d - 1):
        U, S, Vh = torch.linalg.svd(rest, full_matrices=False)
        r = min(rank, S.numel())
        cores.append(U[:, :r].reshape(r_prev, out_modes[k], in_modes[k], r))
        rest = (torch.diag(S[:r]) @ Vh[:r]).reshape(
            r * out_modes[k + 1] * in_modes[k + 1], -1)
        r_prev = r
    cores.append(rest.reshape(r_prev, out_modes[-1], in_modes[-1], 1))
    return cores


def from_cores(cores, out_modes, in_modes):
    import torch

    full = cores[0]
    for core in cores[1:]:
        full = torch.einsum("a...r,rnms->a...nms", full, core)
    d = len(cores)
    full = full.squeeze(0).squeeze(-1)
    order = [2 * k for k in range(d)] + [2 * k + 1 for k in range(d)]
    return full.permute(*order).reshape(math.prod(out_modes), math.prod(in_modes))


def low_rank(W, params: int):
    """A truncated SVD with as close to `params` numbers as a rank allows."""
    import torch

    out, inp = W.shape
    k = max(1, min(min(out, inp), params // (out + inp)))
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vh[:k], k * (out + inp)


# --------------------------------------------------------------------------
# the fair comparison
# --------------------------------------------------------------------------

def _fields(n, generator):
    """16×16 fields made of a few smooth bumps: inputs with real structure
    across their layout, which is the case a tensor train is built for."""
    import torch

    yy, xx = torch.meshgrid(torch.linspace(0, 1, 16), torch.linspace(0, 1, 16),
                            indexing="ij")
    out = torch.zeros(n, 16, 16)
    for _ in range(3):
        cx = torch.rand(n, 1, 1, generator=generator)
        cy = torch.rand(n, 1, 1, generator=generator)
        w = 0.05 + 0.15 * torch.rand(n, 1, 1, generator=generator)
        a = torch.randn(n, 1, 1, generator=generator)
        out = out + a * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * w ** 2))
    return out.reshape(n, 256)


def experiment(ranks=(1, 2, 4, 8), steps: int = 1500, seed: int = 0
               ) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    x, x_test = _fields(2000, g), _fields(1000, g)
    # a smooth teacher: blurred readings at a few places, then a nonlinearity
    teacher = torch.nn.functional.conv2d(
        torch.randn(10, 1, 16, 16, generator=g),
        torch.ones(1, 1, 5, 5) / 25, padding=2).reshape(10, 256)
    def target(z):
        return torch.tanh(z @ teacher.T / 4)
    y, y_test = target(x), target(x_test)

    model = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 10))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    started = time.time()
    for _ in range(steps):
        loss = torch.mean((model(x) - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    def error_with(W):
        with torch.no_grad():
            h = torch.tanh(x_test @ W.T + model[0].bias)
            return float(torch.mean((model[2](h) - y_test) ** 2))

    W = model[0].weight.detach()
    base = error_with(W)
    out_modes, in_modes = [4, 4, 2, 2], [4, 4, 4, 4]
    rows = []
    for r in ranks:
        cores = tt_svd(W, out_modes, in_modes, r)
        tt_params = sum(c.numel() for c in cores)
        W_tt = from_cores(cores, out_modes, in_modes)
        W_lr, lr_params = low_rank(W, tt_params)
        rows.append({"rank": r, "tt_parameters": tt_params,
                     "tt_error": error_with(W_tt),
                     "svd_parameters": lr_params,
                     "svd_error": error_with(W_lr)})

    # the control: a random matrix has no structure for a train to find
    random = torch.randn(64, 256)
    cores = tt_svd(random, out_modes, in_modes, 4)
    rand_tt = float((from_cores(cores, out_modes, in_modes) - random).norm()
                    / random.norm())
    trained_tt = float((from_cores(tt_svd(W, out_modes, in_modes, 4),
                                   out_modes, in_modes) - W).norm() / W.norm())
    return {"dense_parameters": W.numel(), "dense_error": base, "rows": rows,
            "relative_weight_error_rank4": {"trained": trained_tt,
                                            "random": rand_tt},
            "seconds": round(time.time() - started, 1)}
