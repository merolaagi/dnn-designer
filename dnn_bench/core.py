"""
dnn_bench/core.py — the domain-independent half.

The claim this package exists to test is always the same shape:

    a paper gives you a system whose equilibrium is guaranteed to exist, be
    unique, and be stable -- under some structural condition. Does enforcing
    that condition buy anything when the system is used as a neural layer?

Everything about *that* question is domain-free: solving F(x; theta) = 0,
differentiating through the solution, logging what the guarantee is silent
about, and comparing a structured arm against controls at matched parameter
count. Only three things are domain-specific:

    residual()          F(x; theta), whose root is the layer's output
    jacobian()          dF/dx  (override it; the autograd fallback is slow)
    variants()          the arm set: one structured, N controls, matched params

Subclass `ImplicitLayer`, register a `Domain`, and the solver, the implicit
gradients, the diagnostics, the sweep engine, the store and the UI all come for
free. See domains/crn.py for a layer with a constraint subspace and a
feasibility region, domains/contraction.py for the plainest possible case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------

@dataclass
class SolveStats:
    iters: int = 0
    residual: float = 0.0
    converged: bool = False
    diverged: bool = False
    cond_J: float = float('nan')
    """Conditioning of the Jacobian, restricted to the constraint subspace.

    Logged for every domain because it is the thing well-posedness theorems
    tend not to cover: existence, uniqueness and stability say nothing about
    how badly conditioned dF/dx is at the solution, and that matrix is what
    both the Newton solve and the implicit gradient invert."""


# ----------------------------------------------------------------------------

class ImplicitLayer(nn.Module):
    """A layer whose output is a root of F(x; theta) = 0.

    Optionally the root is sought only within an affine subspace x0 + span(S)
    -- that is how conservation laws are expressed -- and only within a
    feasible region, e.g. the positive orthant.

    Subclasses must implement `residual`. Everything else has a default.
    """

    n: int = 0                     # state width
    tol: float = 1e-10
    max_iter: int = 200
    max_linesearch: int = 60

    # --- domain hooks --------------------------------------------------------

    def residual(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """F(x; x0), shape (batch, n). The layer's output is the x where this
        is 0. `x0` is the layer's input, passed explicitly so the backward pass
        can differentiate through it -- an input injected via a mutable
        attribute is invisible to autograd, which silently zeroes its gradient."""
        raise NotImplementedError

    def jacobian(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """dF/dx, shape (batch, n, n).

        The default differentiates `residual` itself, so it cannot disagree with
        it. Override only when you have an analytic form worth the speed -- and
        when you do, `./validate` checks it against this one, because a Jacobian
        that quietly contradicts its residual gives a converging solver and
        silently wrong gradients."""
        xd, x0d = x.detach(), x0.detach()
        try:
            from torch.func import jacrev, vmap

            def single(xi, x0i):
                return self.residual(xi.unsqueeze(0), x0i.unsqueeze(0)).squeeze(0)

            return vmap(jacrev(single))(xd, x0d)
        except Exception:
            n = xd.shape[1]
            rows = []
            xg = xd.clone().requires_grad_(True)
            F = self.residual(xg, x0d)
            for i in range(n):
                g, = torch.autograd.grad(F[:, i].sum(), xg,
                                         retain_graph=(i < n - 1))
                rows.append(g)
            return torch.stack(rows, dim=1)

    def constraint_basis(self) -> Optional[torch.Tensor]:
        """Orthonormal (n, s) basis of the subspace the solution may move in,
        or None for the whole space. Returning a basis means every conserved
        quantity in its orthogonal complement is preserved exactly."""
        return None

    def feasible(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample bool: is this state admissible? Default: everywhere."""
        return torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    def step_ok(self, x: torch.Tensor, xn: torch.Tensor) -> bool:
        """May the line search accept this step?

        Feasibility alone is not enough when the feasible region is open. A
        Newton step can land a hair off the boundary, still satisfy `feasible`,
        and drive terms of the residual to underflow -- which reads as a
        converged solve at a point that is not an interior equilibrium. Domains
        with an open region should impose a floor here, absolute rather than
        relative to the current iterate; `self._solve_ref` holds the scale of
        the initial state for exactly that."""
        return bool(self.feasible(xn).all())

    def init_state(self, x0: torch.Tensor) -> torch.Tensor:
        """Map the encoder's output into the layer's state domain."""
        return x0

    def readout(self, x: torch.Tensor) -> torch.Tensor:
        """Map the solved state into features for the decoder."""
        return x

    def certificate(self) -> dict:
        """What this variant claims, and whether the claim actually holds.
        Shown in the bench so a control arm is never mistaken for the real one."""
        return {'holds': False, 'claim': 'no guarantee'}

    def describe(self) -> dict:
        """Facts for the UI. Include a 'graph' key -- {positions, edges} -- and
        the bench draws it; omit it and you get the facts table alone."""
        return {}

    # --- solver --------------------------------------------------------------

    def _merit(self, x: torch.Tensor, x0: torch.Tensor):
        F = self.residual(x, x0)
        return F, (F.abs().amax(dim=1, keepdim=True) / self._scale(x, F)).amax().item()

    def _scale(self, x: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """Denominator for the residual. Override when the residual's magnitude
        tracks the parameters -- mass-action fluxes, say -- or the solver stalls
        as soon as theta grows."""
        return 1.0 + x.abs().amax(dim=1, keepdim=True)

    def solve(self, x0: torch.Tensor, start: Optional[torch.Tensor] = None):
        """Damped Newton in constraint coordinates, with the residual-descent
        direction as fallback. Every iterate stays in x0 + span(S) and inside
        the feasible region by construction.

        `start` overrides the initial iterate while leaving the input untouched.
        Only the validator uses it: solving the same problem from several starts
        is how the uniqueness claim gets measured rather than asserted."""
        stats = SolveStats()
        x = self.init_state(x0)
        if start is not None:
            x = start
        # scale of the initial state, so step_ok can impose an ABSOLUTE floor.
        # A floor relative to the current iterate is not enough: it permits
        # walking to the boundary geometrically over successive steps.
        self._solve_ref = x.detach().abs().amax().clamp_min(1e-300)
        S = self.constraint_basis()
        if S is None:
            S = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
        eye = torch.eye(S.shape[1], dtype=x.dtype, device=x.device)

        F, resid = self._merit(x, x0)
        window: List[float] = [resid]
        for _ in range(self.max_iter):
            stats.residual = resid
            if not np.isfinite(resid) or resid > 1e14:
                stats.diverged = True
                break
            if resid < self.tol:
                stats.converged = True
                break

            J = torch.einsum('ip,bij,jq->bpq', S, self.jacobian(x, x0), S)
            Fs = F @ S
            dirs = []                       # (direction, is_newton)
            try:
                d = torch.linalg.solve(J + 1e-12 * eye, -Fs.unsqueeze(-1)).squeeze(-1)
                if torch.isfinite(d).all():
                    dirs.append((d, True))
            except Exception:
                pass
            # Both signs of the projected residual. For a domain where F is a
            # stabilising flow (dx/dt = F with a Lyapunov function), +F is the
            # globally convergent direction and descending ||F|| is not -- the
            # latter has spurious minima on the boundary of the feasible region.
            # Trying both costs one line search and needs no domain knowledge.
            dirs.append((Fs, False))
            dirs.append((-Fs, False))

            moved = False
            for d, is_newton in dirs:
                step = 1.0
                if not is_newton:           # scale the descent step sanely
                    dn = (d @ S.T).abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
                    step = min(float((0.1 * self._scale(x, F) / dn).amin()), 1.0)
                for _ in range(self.max_linesearch):
                    xn = x + step * (d @ S.T)
                    if self.step_ok(x, xn):
                        Fn, rn = self._merit(xn, x0)
                        # Non-monotone acceptance (Grippo-Lampariello-Lucidi):
                        # compare against the worst of a recent window, not the
                        # last value. A step along a stabilising flow can raise
                        # ||F|| transiently while still making real progress,
                        # and a strictly monotone rule rejects it and stalls.
                        if np.isfinite(rn) and rn < max(window):
                            x, F, resid = xn, Fn, rn
                            window.append(rn)
                            del window[:-8]
                            moved = True
                            break
                    step *= 0.5
                if moved:
                    stats.iters += 1
                    break
            if not moved:
                break

        stats.residual = resid
        # Strictly the requested tolerance. An earlier leniency floor here let a
        # solve that stalled against the feasibility boundary report success,
        # which is the worst possible failure mode: a wrong answer with a clean
        # flag on it.
        stats.converged = bool(np.isfinite(resid) and resid < self.tol)
        try:
            with torch.no_grad():
                J = torch.einsum('ip,bij,jq->bpq', S, self.jacobian(x, x0), S)
                c = torch.linalg.cond(J)
                c = c[torch.isfinite(c)]
                stats.cond_J = float(c.max()) if c.numel() else float('nan')
        except Exception:
            pass
        return x.detach(), stats

    # --- forward -------------------------------------------------------------

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        params = [p for p in self.parameters() if p.requires_grad]
        out = _ImplicitFn.apply(self, x0, *params)
        return out


class _ImplicitFn(torch.autograd.Function):
    """Solve without a tape, then differentiate by the implicit function
    theorem: one s x s solve, no memory in the depth of the solve."""

    @staticmethod
    def forward(ctx, layer, x0, *params):
        with torch.no_grad():
            xs, stats = layer.solve(x0)
        layer.last_stats = stats
        ctx.layer = layer
        ctx.save_for_backward(xs, x0)
        ctx.params = params
        return xs

    @staticmethod
    def backward(ctx, grad_out):
        """The solution satisfies  F(x*; x0, theta) = 0  with
        x* = s(x0) + S alpha,  s = init_state.  So x0 reaches x* by two routes:
        it shifts the affine subspace through s, and it enters F directly.
        Handling only the first silently zeroes the input gradient for any
        domain that injects its input into the residual."""
        layer, (xs, x0) = ctx.layer, ctx.saved_tensors
        S = layer.constraint_basis()
        if S is None:
            S = torch.eye(xs.shape[1], dtype=xs.dtype, device=xs.device)

        with torch.no_grad():
            G = layer.jacobian(xs, x0)                              # dF/dx
            J = torch.einsum('ip,bij,jq->bpq', S, G, S)
            eye = torch.eye(J.shape[-1], dtype=J.dtype, device=J.device)
            rhs = (grad_out @ S).unsqueeze(-1)
            try:
                w = torch.linalg.solve(J.transpose(-1, -2) + 1e-12 * eye, rhs)
            except Exception:
                w = torch.zeros_like(rhs)
            Sw = torch.nan_to_num(w.squeeze(-1)) @ S.T              # (b, n)
            #  the vector that the ds/dx0 route is contracted with
            through_s = grad_out - torch.einsum('bn,bnm->bm', Sw, G)

        #  direct route: vjp of F through x0 and the parameters, with -Sw
        with torch.enable_grad():
            x0d = x0.detach().requires_grad_(True)
            F = layer.residual(xs.detach(), x0d)
            gs = torch.autograd.grad(F, (x0d,) + tuple(ctx.params),
                                     grad_outputs=-Sw, allow_unused=True)
        grad_x0_direct, param_grads = gs[0], gs[1:]

        #  subspace route: vjp of s = init_state through x0, with through_s
        with torch.enable_grad():
            x0e = x0.detach().requires_grad_(True)
            s_out = layer.init_state(x0e)
            if s_out.requires_grad:
                gsub, = torch.autograd.grad(s_out, x0e, grad_outputs=through_s,
                                            allow_unused=True)
            else:
                gsub = None

        grad_x0 = torch.zeros_like(x0)
        if grad_x0_direct is not None:
            grad_x0 = grad_x0 + grad_x0_direct
        if gsub is not None:
            grad_x0 = grad_x0 + gsub

        params = tuple(torch.zeros_like(p) if g is None else torch.nan_to_num(g)
                       for g, p in zip(param_grads, ctx.params))
        return (None, torch.nan_to_num(grad_x0)) + params


# ----------------------------------------------------------------------------
# domain plugin protocol
# ----------------------------------------------------------------------------

@dataclass
class Variant:
    """One arm of an ablation."""
    key: str
    label: str                       # what condition this arm satisfies or breaks
    build: Callable[[], ImplicitLayer]
    structured: bool = False         # is this the arm with the guarantee?


class Domain(Protocol):
    key: str
    title: str
    paper: str                       # where the structure came from
    claim: str                       # the theorem, in one line

    def defaults(self) -> dict: ...
    def variants(self, cfg: dict, seed: int) -> List[Variant]: ...


REGISTRY: Dict[str, Domain] = {}


def register(domain: Domain) -> Domain:
    REGISTRY[domain.key] = domain
    return domain


def get(key: str) -> Domain:
    if key not in REGISTRY:
        raise KeyError(f"unknown domain {key!r}; have {sorted(REGISTRY)}")
    return REGISTRY[key]


def all_domains() -> List[dict]:
    return [{'key': d.key, 'title': d.title, 'paper': d.paper, 'claim': d.claim,
             'defaults': d.defaults(),
             'variants': [{'key': v.key, 'label': v.label,
                           'structured': v.structured}
                          for v in d.variants(d.defaults(), 0)]}
            for d in REGISTRY.values()]
