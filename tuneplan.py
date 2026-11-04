"""Planned fine-tuning: start from a checkpoint and train small low-rank adapters only where the gradient on
your data points, instead of every weight.

Isomorph's fine-tune planner (Weight space → Plan a fine-tune) measured this against tuning every matrix,
the last blocks first and random matrices, at the same number of trainable values: on its test the
gradient's choice beat all three beyond the seeds' spread, while choosing from the weights' spectrum alone
did not. This is the gradient choice, built into training:

  1. one backward pass over a few training batches, with the checkpoint's weights;
  2. each loaded weight matrix is scored by the size of its gradient relative to its own size, and given the
     rank that holds 90% of that gradient's energy (at most 16);
  3. matrices are taken in that order until the budget (a share of the loaded weights) is spent; each gets a
     delta W + U V^T with U starting at zero, so training starts exactly where the checkpoint is;
  4. every loaded weight is frozen; layers the checkpoint did not fill (a resized head) train in full.

Saved checkpoints hold the merged weights (W + U V^T), so they load like any other.
"""
import math
from typing import Any, Callable, Dict, List

MAX_RANK = 16


def _lowrank(torch, shape, rank, seed):
    nn = torch.nn
    m, n = shape[0], int(math.prod(shape[1:]))

    class LowRank(nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.U = nn.Parameter(torch.zeros(m, rank))
            self.V = nn.Parameter(torch.randn(n, rank, generator=g) / math.sqrt(n))

        def forward(self, W):
            return W + (self.U @ self.V.T).reshape(W.shape).to(W.dtype)

    return LowRank()


def candidates(model, loaded_keys) -> List[Dict[str, Any]]:
    """Every weight matrix (2-D or more, not an embedding table) that the checkpoint filled and that the
    design does not freeze."""
    import torch

    loaded = set(loaded_keys)
    out = []
    for mname, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            continue
        for pname, p in mod.named_parameters(recurse=False):
            full = f"{mname}.{pname}" if mname else pname
            if p.ndim >= 2 and full in loaded and p.requires_grad:
                out.append({"name": full, "module": mod, "attr": pname, "shape": tuple(p.shape)})
    return out


def read_gradient(model, cands, batches, loss_fn) -> List[Dict[str, Any]]:
    import torch

    params = [getattr(c["module"], c["attr"]) for c in cands]
    for p in params:
        p.grad = None
    model.train()
    n = 0
    for xs, yb in batches:
        _, loss = loss_fn(xs, yb)
        loss.backward()
        n += 1
    out = []
    for c, p in zip(cands, params):
        if p.grad is None:
            out.append({"grad_rel": 0.0, "rank90": 1})
            continue
        G = p.grad.detach().float().reshape(p.shape[0], -1).cpu()
        s = torch.linalg.svdvals(G)
        e = (s ** 2).cumsum(0) / ((s ** 2).sum() + 1e-30)
        out.append({"grad_rel": float(G.norm() / (p.detach().float().norm() + 1e-12)), "rank90": int((e < 0.9).sum()) + 1})
    model.zero_grad(set_to_none=True)
    return out


def plan(cands, reads, budget):
    order = sorted(range(len(cands)), key=lambda i: -reads[i]["grad_rel"])
    pick, used = [], 0
    for i in order:
        if reads[i]["grad_rel"] <= 0:
            continue
        m, n = cands[i]["shape"][0], int(math.prod(cands[i]["shape"][1:]))
        r = min(reads[i]["rank90"], MAX_RANK, m, n, (budget - used) // (m + n))
        if r < 1:
            continue
        pick.append((i, int(r)))
        used += int(r) * (m + n)
        if budget - used < min(c["shape"][0] + int(math.prod(c["shape"][1:])) for c in cands):
            break
    return pick, used


def apply(model, loaded_keys, batches, loss_fn, budget_frac=0.01, seed=0) -> Dict[str, Any]:
    """Plan and attach the adapters; freeze every loaded weight. Returns the plan for the run's log."""
    import torch
    import torch.nn.utils.parametrize as P

    cands = candidates(model, loaded_keys)
    if not cands:
        return {"ok": False, "reason": "No weight matrix was loaded from the checkpoint, so there is nothing to adapt."}
    reads = read_gradient(model, cands, batches, loss_fn)
    loaded_total = sum(int(math.prod(c["shape"])) for c in cands)
    smallest = min(c["shape"][0] + int(math.prod(c["shape"][1:])) for c in cands)
    budget = max(int(budget_frac * loaded_total), smallest)
    pick, used = plan(cands, reads, budget)
    if not pick:
        return {"ok": False, "reason": "The gradient on these batches is zero everywhere the checkpoint loaded."}
    loaded = set(loaded_keys)
    for full, p in model.named_parameters():
        if full in loaded:
            p.requires_grad_(False)
    adapters = []
    for i, r in pick:
        c = cands[i]
        p = getattr(c["module"], c["attr"])
        lr = _lowrank(torch, c["shape"], r, seed * 7919 + i).to(p.device, p.dtype)
        P.register_parametrization(c["module"], c["attr"], lr)
        adapters.append({"name": c["name"], "shape": list(c["shape"]), "rank": r, "grad_rel": round(reads[i]["grad_rel"], 5)})
    fresh = sum(p.numel() for n_, p in model.named_parameters() if p.requires_grad and "parametrizations" not in n_)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"ok": True, "adapters": adapters, "adapter_values": used, "loaded_weights": loaded_total, "budget": budget,
            "budget_frac": budget_frac, "fresh_values": fresh, "trainable": trainable,
            "skipped": len(cands) - len(pick), "candidates": len(cands)}


def is_planned(model) -> bool:
    import torch.nn.utils.parametrize as P

    return any(P.is_parametrized(m) for m in model.modules())


def merged_state_dict(model) -> Dict[str, Any]:
    """The state dict as the plain design would have it: each adapted weight is W + U V^T, under its own name."""
    import torch
    import torch.nn.utils.parametrize as P

    sd = model.state_dict()
    if not is_planned(model):
        return sd
    merged = {}
    with torch.no_grad():
        for prefix, m in model.named_modules():
            if P.is_parametrized(m):
                for attr in m.parametrizations.keys():
                    merged[f"{prefix}.{attr}" if prefix else attr] = getattr(m, attr).detach().clone()
    out = {}
    for k, v in sd.items():
        if ".parametrizations." in k or k.startswith("parametrizations."):
            head, _, tail = k.partition("parametrizations.")
            attr = tail.split(".")[0]
            if tail == f"{attr}.original":
                out[f"{head}{attr}"] = merged[f"{head}{attr}"]
            continue
        out[k] = v
    return out
