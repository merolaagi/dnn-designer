"""
Chemical reaction network theory — Feinberg / Horn–Jackson.

Guarantee mechanism: GRAPH TOPOLOGY. The condition is an integer computed once
at design time from the reaction graph, and it covers every positive parameter
value, so theta stays unconstrained during training.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..core import ImplicitLayer, Variant, register
from . import crn_network as cn


class CRNLayer(ImplicitLayer):
    """dx/dt = Y B v(x),  v_e = k_e * prod_i x_i^{y_i,src(e)},  k = exp(theta).

    The solution is sought inside x0 + im(YB), so every conserved quantity in
    the left null space of YB is preserved to machine precision.
    """

    def __init__(self, net: cn.ReactionNetwork, dtype=torch.float64,
                 tol=1e-10, max_iter=200):
        super().__init__()
        self.net, self.n, self.tol, self.max_iter = net, net.n, tol, max_iter
        self.register_buffer('Ymat', torch.tensor(net.Y, dtype=dtype))
        self.register_buffer('YB', torch.tensor(net.YB, dtype=dtype))
        self.register_buffer('S', torch.tensor(net.stoichiometric_basis(), dtype=dtype))
        self.register_buffer('src', torch.tensor([i for i, _ in net.edges],
                                                 dtype=torch.long))
        self.theta = nn.Parameter(torch.randn(net.n_edges, dtype=dtype) * 0.5)

    def _fluxes(self, x):
        u = torch.log(x.clamp_min(1e-300))
        return torch.exp(self.theta).unsqueeze(0) * torch.exp(u @ self.Ymat)[:, self.src]

    def residual(self, x, x0=None):
        # x0 does not appear: for a reaction network the input sets the
        # compatibility class through init_state, not a term in the dynamics.
        return self._fluxes(x) @ self.YB.T

    def jacobian(self, x, x0=None):
        v = self._fluxes(x)
        Ysrc = self.Ymat[:, self.src]
        return ((self.YB * v.unsqueeze(1)) @ Ysrc.T) / x.unsqueeze(1).clamp_min(1e-300)

    def _scale(self, x, F):
        # Flux-relative: how well do the fluxes cancel, measured against their
        # own magnitude. Scale-free in k, which matters because k = exp(theta)
        # spans many orders of magnitude during training.
        v = self._fluxes(x)
        return ((1.0 + x.abs().amax(dim=1, keepdim=True))
                * (1.0 + v.abs().amax(dim=1, keepdim=True)))

    def constraint_basis(self):
        return self.S

    def feasible(self, x):
        return (x > 0).all(dim=1)

    #: coordinates below this fraction of the initial scale are treated as the
    #: boundary. The theorem promises an equilibrium in the OPEN positive
    #: orthant; a boundary point where the monomials x^y have underflowed is a
    #: failed solve, not an answer, and must not be reported as converged.
    boundary_floor = 1e-9

    def step_ok(self, x, xn):
        return bool((xn > self.boundary_floor * self._solve_ref).all())

    def readout(self, x):
        return torch.log(x.clamp_min(1e-300))

    def certificate(self):
        wr, d = self.net.is_weakly_reversible(), self.net.deficiency()
        holds = wr and d == 0
        return {'holds': holds,
                'claim': ('unique positive equilibrium per compatibility class, '
                          'asymptotically stable, for every k > 0' if holds else
                          ('not weakly reversible: an equilibrium need not exist'
                           if not wr else
                           f'deficiency {d}: complex balance only on a '
                           f'codimension-{d} subvariety of k-space'))}

    def describe(self):
        d = cn_describe(self.net)
        d['n_params'] = self.net.n_edges
        return d


def cn_describe(net: cn.ReactionNetwork) -> dict:
    classes = net.linkage_classes()
    pos, cx = {}, 0.0
    for cls in classes:
        r = 1.0 + 0.28 * max(0, len(cls) - 3)
        for i, node in enumerate(cls):
            th = 2 * np.pi * i / len(cls) - np.pi / 2
            pos[node] = (cx + r * float(np.cos(th)), r * float(np.sin(th)))
        cx += 2 * r + 1.1
    s = net.m - len(classes) - net.deficiency()
    return {
        'summary': net.summary(),
        'facts': [(str(net.m), f'complexes in {len(classes)} linkage classes'),
                  (str(net.n_edges), 'reactions = trainable rates'),
                  (str(s), 'dim of stoichiometric subspace'),
                  (str(net.n - s), 'conserved quantities')],
        'graph': {'positions': [list(pos[i]) for i in range(net.m)],
                  'edges': [list(e) for e in net.edges]},
    }


class CRNDomain:
    key = 'crn'
    title = 'Reaction network topology'
    paper = 'Feinberg / Horn–Jackson, Deficiency Zero Theorem (arXiv:1805.10371 §2.2)'
    claim = ('A weakly reversible network of deficiency zero has exactly one '
             'positive equilibrium per compatibility class, for every k > 0.')

    def defaults(self):
        return {'n_species': 12, 'class_sizes': [4, 4], 'extra_edges': 4}

    def variants(self, cfg, seed):
        n = cfg.get('n_species', 12)
        cs = cfg.get('class_sizes', [4, 4])
        ee = cfg.get('extra_edges', 4)
        a = cn.generate_network(n, cs, ee, 0, seed=seed)
        b = cn.break_weak_reversibility(a, seed=seed)
        c = cn.generate_network(n, cs, ee, 1, seed=seed)
        mk = lambda net: (lambda: CRNLayer(net))
        return [
            Variant('wr0', 'weakly reversible, δ=0 — theorem applies',
                    mk(a), structured=True),
            Variant('broken_wr', 'one edge reversed — not weakly reversible', mk(b)),
            Variant('delta_one', 'deficiency 1 — complex balance not universal', mk(c)),
        ]


register(CRNDomain())
