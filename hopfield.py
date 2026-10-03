"""Attention is one step of a memory: the modern Hopfield network.

Ramsauer et al., *Hopfield Networks is All You Need* (ICLR 2021). A
continuous Hopfield network stores N patterns as the columns of X and has
energy

    E(ξ) = −lse(β, Xᵀξ) + ½ ξᵀξ + β⁻¹ log N + ½ M²

where lse(β, a) = β⁻¹ log Σ exp(β aᵢ) and M is the largest pattern norm.
Minimising E by the concave–convex procedure gives the update

    ξ_new = X softmax(β Xᵀ ξ)

which is attention: a query ξ, keys and values X, and β = 1/√d. Three
consequences follow, and each is measured here rather than quoted:

    the two computations are the same numbers, to floating-point precision;
    the energy never rises under the update — the CCCP guarantees it;
    β sets what is retrieved. Large β pulls the query to the single nearest
    stored pattern in one step. Small β settles on an average of several —
    a metastable state, which is what an attention head that "averages" is.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def energy(xi, X, beta: float):
    import torch

    N = X.shape[1]
    M = float(X.norm(dim=0).max())
    scores = beta * (X.T @ xi)
    lse = torch.logsumexp(scores, dim=0) / beta
    return (-lse + 0.5 * (xi * xi).sum(0) + math.log(N) / beta
            + 0.5 * M ** 2)


def update(xi, X, beta: float):
    import torch

    return X @ torch.softmax(beta * (X.T @ xi), dim=0)


def attention(Q, K, V, scale: float):
    """The transformer's own formula, written as it is in the papers."""
    import torch

    return torch.softmax(Q @ K.T * scale, dim=-1) @ V


def experiment(d: int = 64, patterns: int = 32, seed: int = 0) -> Dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    X = torch.randn(d, patterns, dtype=torch.float64)
    X = X / X.norm(dim=0, keepdim=True) * math.sqrt(d)
    beta = 1 / math.sqrt(d)

    # 1. the same numbers: a batch of queries through both formulas
    queries = torch.randn(d, 50, dtype=torch.float64)
    hop = update(queries, X, beta)
    att = attention(queries.T, X.T, X.T, beta).T
    same = float((hop - att).abs().max())

    # 2. the energy never rises, followed over repeated updates
    xi = torch.randn(d, 20, dtype=torch.float64) * 2
    trace: List[List[float]] = []
    worst_rise = -math.inf
    for _ in range(12):
        e = energy(xi, X, beta)
        trace.append([float(v) for v in e[:5]])
        nxt = update(xi, X, beta)
        worst_rise = max(worst_rise, float((energy(nxt, X, beta) - e).max()))
        xi = nxt

    # 3. what β retrieves, from a corrupted copy of a stored pattern
    target = X[:, :1]
    # noise of 0.6 per component, against pattern components of size ~1
    noisy = target + 0.6 * torch.randn(d, 200, dtype=torch.float64)
    sweep = []
    for b in (0.02, 0.05, 0.1, beta, 0.25, 0.5, 1.0):
        out = update(noisy, X, b)
        weights = torch.softmax(b * (X.T @ noisy), dim=0)
        nearest = (X.T @ out).argmax(0)
        sweep.append({
            "beta": round(b, 4),
            "is_transformer_beta": abs(b - beta) < 1e-12,
            # how much of the attention lands on the single largest pattern
            "top_weight": float(weights.max(0).values.mean()),
            # how many patterns share the weight, as an effective count
            "patterns_mixed": float((1 / (weights ** 2).sum(0)).mean()),
            "retrieved_the_right_one": float((nearest == 0).double().mean()),
            "cosine_to_target": float(torch.nn.functional.cosine_similarity(
                out, target.expand_as(out), dim=0).mean()),
        })

    return {"d": d, "patterns": patterns, "beta": beta,
            "max_difference": same, "energy_trace": trace,
            "worst_energy_rise": worst_rise, "sweep": sweep}
