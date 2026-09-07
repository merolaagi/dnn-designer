"""
Monotone operator theory — Winston & Kolter, monDEQ (arXiv:2006.08591).

Guarantee mechanism: A CONSTRAINT ON THE WEIGHTS. W is reparameterised so that
I - W stays strongly monotone no matter what the optimiser does. This is the
incumbent approach and the thing a topology-based guarantee has to beat, so it
belongs in the bench as a comparison arm rather than only as prior work.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..core import ImplicitLayer, Variant, register


class ContractionLayer(ImplicitLayer):
    """F(x) = tanh(Wx + x0) - x.  The input is injected, monDEQ-style.

    Three parameterisations of W, all with 2n^2 trainable entries:
      monotone      W = (1-m)I - A^T A + B - B^T   =>  sym(I - W) >= m I
      margin_zero   the same with m = 0            =>  monotone, not strongly
      unconstrained W = A + B                      =>  no structure at all
    """

    def __init__(self, n, mode='monotone', m=0.05, dtype=torch.float64,
                 tol=1e-10, max_iter=200):
        super().__init__()
        self.n, self.mode, self.m = n, mode, m
        self.tol, self.max_iter = tol, max_iter
        self.A = nn.Parameter(torch.randn(n, n, dtype=dtype) / n ** .5)
        self.B = nn.Parameter(torch.randn(n, n, dtype=dtype) / n ** .5)

    def W(self):
        if self.mode == 'unconstrained':
            return self.A + self.B
        m = 0.0 if self.mode == 'margin_zero' else self.m
        eye = torch.eye(self.n, dtype=self.A.dtype, device=self.A.device)
        return (1 - m) * eye - self.A.T @ self.A + self.B - self.B.T

    def residual(self, x, x0):
        return torch.tanh(x @ self.W().T + x0) - x

    def jacobian(self, x, x0):
        W = self.W()
        d = 1 - torch.tanh(x @ W.T + x0) ** 2                  # (b, n)
        eye = torch.eye(self.n, dtype=x.dtype, device=x.device)
        return d.unsqueeze(-1) * W.unsqueeze(0) - eye.unsqueeze(0)

    def init_state(self, x0):
        return torch.zeros_like(x0)          # the input is injected, not the state

    def certificate(self):
        if self.mode == 'monotone':
            return {'holds': True,
                    'claim': f'I - W strongly monotone with margin {self.m}: '
                             'unique fixed point, linear convergence'}
        if self.mode == 'margin_zero':
            return {'holds': False,
                    'claim': 'margin 0: monotone but not strongly, so '
                             'uniqueness and convergence are not guaranteed'}
        return {'holds': False,
                'claim': 'W unconstrained: a fixed point may fail to exist or '
                         'be non-unique'}

    def describe(self):
        with torch.no_grad():
            W = self.W()
            eye = torch.eye(self.n, dtype=W.dtype)
            sym = ((eye - W) + (eye - W).T) / 2
            margin = float(torch.linalg.eigvalsh(sym).min())
        return {'summary': f'n={self.n}, mode={self.mode}, '
                           f'measured monotonicity margin={margin:.3f}',
                'facts': [(str(self.n), 'state width'),
                          (str(2 * self.n ** 2), 'trainable weights'),
                          (f'{margin:.3f}', 'min eigenvalue of sym(I − W)'),
                          ('yes' if margin > 0 else 'no', 'strongly monotone')],
                'n_params': 2 * self.n ** 2}


class ContractionDomain:
    key = 'contraction'
    title = 'Monotone operator constraint'
    paper = 'Winston & Kolter, Monotone Operator Equilibrium Networks (arXiv:2006.08591)'
    claim = ('If I - W is strongly monotone the fixed point exists, is unique, '
             'and operator splitting converges linearly.')

    def defaults(self):
        return {'n_species': 12, 'margin': 0.05}

    def variants(self, cfg, seed):
        n, m = cfg.get('n_species', 12), cfg.get('margin', 0.05)

        def mk(mode):
            def build():
                torch.manual_seed(seed)
                return ContractionLayer(n, mode=mode, m=m)
            return build

        return [
            Variant('monotone', f'sym(I − W) ⪰ {m}I — theorem applies',
                    mk('monotone'), structured=True),
            Variant('margin_zero', 'margin 0 — monotone but not strongly',
                    mk('margin_zero')),
            Variant('unconstrained', 'W free — no guarantee', mk('unconstrained')),
        ]


register(ContractionDomain())
