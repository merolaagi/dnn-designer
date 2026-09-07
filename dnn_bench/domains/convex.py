"""
Convex ridge potentials — the equilibrium is the minimiser of a learned energy.

Guarantee mechanism: CONVEXITY OF A POTENTIAL. Well-posedness comes neither from
graph structure nor from a spectral margin but from the sign of a coefficient
vector, which is a third and quite different way to buy the same certificate.

    f(x) = sum_k s_k softplus(w_k . x + b_k) - <x0, x> + (alpha/2)||x||^2

s_k >= 0 makes f convex; alpha > 0 makes it strongly convex, hence a unique
minimiser. The layer returns the x where grad f = 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from ..core import ImplicitLayer, Variant, register


class ConvexPotentialLayer(ImplicitLayer):
    """F(x) = -grad f(x) = x0 - alpha x - sum_k s_k sigmoid(w_k.x + b_k) w_k"""

    def __init__(self, n, n_ridges=16, mode='convex', alpha=0.5,
                 dtype=torch.float64, tol=1e-10, max_iter=200):
        super().__init__()
        self.n, self.mode, self.K = n, mode, n_ridges
        self.alpha = 0.0 if mode == 'flat' else alpha
        self.tol, self.max_iter = tol, max_iter
        self.Wr = nn.Parameter(torch.randn(n_ridges, n, dtype=dtype) / n ** .5)
        self.b = nn.Parameter(torch.zeros(n_ridges, dtype=dtype))
        self.s_raw = nn.Parameter(torch.randn(n_ridges, dtype=dtype) * 0.3)

    def s(self):
        # 'signed' lets the coefficients go negative, which is exactly what
        # breaks convexity while keeping the parameter count identical.
        return self.s_raw if self.mode == 'signed' else Fn.softplus(self.s_raw)

    def residual(self, x, x0):
        z = x @ self.Wr.T + self.b                       # (b, K)
        grad = (torch.sigmoid(z) * self.s()) @ self.Wr   # (b, n)
        return x0 - self.alpha * x - grad

    def jacobian(self, x, x0=None):
        z = x @ self.Wr.T + self.b
        sig = torch.sigmoid(z)
        d = self.s() * sig * (1 - sig)                   # (b, K)
        H = torch.einsum('bk,ki,kj->bij', d, self.Wr, self.Wr)
        eye = torch.eye(self.n, dtype=x.dtype, device=x.device)
        return -H - self.alpha * eye.unsqueeze(0)

    def init_state(self, x0):
        return torch.zeros_like(x0)          # the input is injected, not the state

    def certificate(self):
        if self.mode == 'convex':
            return {'holds': True,
                    'claim': f'coefficients ≥ 0 and alpha={self.alpha}: f is '
                             'strongly convex, so the minimiser is unique'}
        if self.mode == 'flat':
            return {'holds': False,
                    'claim': 'convex but alpha=0: not strongly convex, so a '
                             'minimiser may not exist or may not be unique'}
        return {'holds': False,
                'claim': 'coefficients may be negative: f need not be convex, '
                         'so stationary points can be multiple or saddles'}

    def describe(self):
        with torch.no_grad():
            s = self.s()
            neg = int((s < 0).sum())
        return {'summary': f'n={self.n}, {self.K} ridges, mode={self.mode}, '
                           f'alpha={self.alpha}, {neg} negative coefficients',
                'facts': [(str(self.n), 'state width'),
                          (str(self.K), 'ridge functions'),
                          (f'{self.alpha:g}', 'strong convexity constant'),
                          (str(neg), 'negative coefficients (0 ⇒ convex)')],
                'n_params': self.K * (self.n + 2)}


class ConvexDomain:
    key = 'convex'
    title = 'Convex potential'
    paper = 'Convex ridge regularisers / input-convex energies'
    claim = ('A strongly convex potential has exactly one stationary point, '
             'and it is the global minimum.')

    def defaults(self):
        return {'n_species': 12, 'n_ridges': 16, 'alpha': 0.5}

    def variants(self, cfg, seed):
        n = cfg.get('n_species', 12)
        K = cfg.get('n_ridges', 16)
        a = cfg.get('alpha', 0.5)

        def mk(mode):
            def build():
                torch.manual_seed(seed)
                return ConvexPotentialLayer(n, K, mode=mode, alpha=a)
            return build

        return [
            Variant('convex', f'coefficients ≥ 0, α={a} — strongly convex',
                    mk('convex'), structured=True),
            Variant('flat', 'coefficients ≥ 0, α=0 — convex, not strongly',
                    mk('flat')),
            Variant('signed', 'coefficients free — convexity broken', mk('signed')),
        ]


register(ConvexDomain())
