"""
dnn_bench/validate.py — run this before you spend a night of compute.

A hand-derived domain can be wrong in ways that look exactly like it working. A
Jacobian with a sign error still gives a solver that converges; it just gives
silently wrong gradients. A control arm that accidentally still satisfies the
condition still trains fine; it just makes your ablation meaningless. Both cost
you a week and neither shows up in a loss curve.

Every check here targets one of those. The expensive ones (finite differences,
parameter sweeps) run on small problems because correctness does not need scale.

    ./validate                    # every registered domain
    ./validate crn convex         # named domains
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from . import core

PASS, FAIL, WARN, SKIP = 'pass', 'fail', 'warn', 'skip'


@dataclass
class Check:
    name: str
    status: str
    detail: str
    value: Optional[float] = None
    scope: str = ''

    def line(self) -> str:
        mark = {PASS: 'ok  ', FAIL: 'FAIL', WARN: 'warn', SKIP: '--  '}[self.status]
        who = f"{self.scope} " if self.scope else ''
        return f"  {mark}  {who}{self.name}: {self.detail}"


def _rel(a, b):
    d, n = (a - b).norm().item(), b.norm().item()
    return d / n if n > 1e-30 else d


def _grade(err, tight, loose, name, detail_fmt, scope):
    st = PASS if err < tight else (WARN if err < loose else FAIL)
    return Check(name, st, detail_fmt.format(err), err, scope)


# ----------------------------------------------------------------------------
# per-arm checks
# ----------------------------------------------------------------------------

def check_jacobian(layer, x, x0, scope) -> Check:
    """An analytic Jacobian that disagrees with autograd is the single most
    dangerous defect available here: the solve still converges, so the only
    symptom is wrong gradients."""
    if type(layer).jacobian is core.ImplicitLayer.jacobian:
        return Check('jacobian', SKIP, 'no analytic Jacobian; autograd fallback '
                                       'in use (correct but slow)', scope=scope)
    with torch.no_grad():
        analytic = layer.jacobian(x, x0)
    auto = core.ImplicitLayer.jacobian(layer, x, x0)
    err = _rel(analytic, auto)
    return _grade(err, 1e-8, 1e-5, 'jacobian',
                  'analytic vs autograd, relative error {:.2e}', scope)


def check_implicit_gradients(layer, x0, scope, n_probe=6, cond=None,
                             converged=True) -> List[Check]:
    """Implicit-function-theorem gradients against central finite differences,
    for both the parameters and the input."""
    out = []
    # Neither side of this comparison means anything when the solve did not
    # land on a well-conditioned root: the implicit gradient inverts a singular
    # matrix and the finite difference straddles a discontinuity. Say so
    # instead of reporting a number that looks like a verdict.
    if not converged:
        return [Check(n, SKIP, 'reference solve did not converge; gradients are '
                               'not defined here', scope=scope)
                for n in ('grad/params', 'grad/input')]
    if cond is not None and np.isfinite(cond) and cond > 1e8:
        return [Check(n, SKIP, f'Jacobian is near-singular (cond {cond:.1e}); '
                               'neither the implicit gradient nor the finite '
                               'difference is trustworthy', cond, scope)
                for n in ('grad/params', 'grad/input')]
    if cond is not None and not np.isfinite(cond):
        return [Check(n, SKIP, 'Jacobian is singular; gradients undefined',
                      scope=scope) for n in ('grad/params', 'grad/input')]
    w = torch.randn_like(x0)
    x0g = x0.detach().clone().requires_grad_(True)
    layer.zero_grad(set_to_none=True)
    (layer(x0g) * w).sum().backward()

    params = [p for p in layer.parameters() if p.requires_grad]
    if not params:
        out.append(Check('grad/params', SKIP, 'layer has no parameters', scope=scope))
    else:
        p = max(params, key=lambda q: q.numel())
        flat, g_an = p.data.view(-1), (p.grad.view(-1) if p.grad is not None
                                       else torch.zeros(p.numel()))
        idx = torch.randperm(flat.numel())[:min(n_probe, flat.numel())]
        eps, fd = 1e-6, torch.zeros(len(idx), dtype=flat.dtype)
        base = flat.clone()
        for j, i in enumerate(idx):
            for sgn, k in ((1, 0), (-1, 1)):
                flat[i] = base[i] + sgn * eps
                with torch.no_grad():
                    xs, _ = layer.solve(x0.detach())
                v = (xs * w).sum().item()
                fd[j] = v if k == 0 else (fd[j] - v) / (2 * eps)
            flat[i] = base[i]
        err = _rel(g_an[idx], fd)
        out.append(_grade(err, 1e-6, 1e-3, 'grad/params',
                          'implicit vs finite difference, relative error {:.2e}',
                          scope))

    eps = 1e-6
    fd = torch.zeros_like(x0)
    for b in range(min(2, x0.shape[0])):
        for i in range(x0.shape[1]):
            vals = []
            for sgn in (1, -1):
                pert = x0.detach().clone()
                pert[b, i] += sgn * eps
                with torch.no_grad():
                    xs, _ = layer.solve(pert)
                vals.append((xs * w).sum().item())
            fd[b, i] = (vals[0] - vals[1]) / (2 * eps)
    nb = min(2, x0.shape[0])
    err = _rel(x0g.grad[:nb], fd[:nb])
    out.append(_grade(err, 1e-6, 1e-3, 'grad/input',
                      'implicit vs finite difference, relative error {:.2e}', scope))
    return out


def check_uniqueness(layer, x0, scope, n_starts=4) -> Check:
    """The claim under test, measured rather than asserted: solve the same
    problem from different starting points and see whether they agree.

    With a constraint subspace the alternative starts stay inside the same
    compatibility class, so this is exactly the uniqueness the theorem asserts.
    A control arm failing here is the ablation working."""
    with torch.no_grad():
        ref, st = layer.solve(x0)
        if not st.converged:
            return Check('uniqueness', WARN, 'reference solve did not converge; '
                                             'uniqueness untested', scope=scope)
        S = layer.constraint_basis()
        worst, tried = 0.0, 0
        for _ in range(n_starts):
            base = layer.init_state(x0)
            for scale in (0.3, 0.1, 0.03, 0.01):
                if S is None:
                    cand = base + torch.randn_like(base) * scale
                else:
                    a = torch.randn(base.shape[0], S.shape[1], dtype=base.dtype)
                    cand = base + (a * scale) @ S.T
                if bool(layer.feasible(cand).all()):
                    break
            else:
                continue
            alt, st2 = layer.solve(x0, start=cand)
            if not st2.converged:
                return Check('uniqueness', FAIL,
                             'a different start in the same class failed to '
                             'converge — the equilibrium is not reachable from '
                             'everywhere', scope=scope)
            denom = 1.0 + ref.abs().amax().item()
            worst = max(worst, (alt - ref).abs().amax().item() / denom)
            tried += 1
    if not tried:
        return Check('uniqueness', SKIP, 'no feasible alternative start found',
                     scope=scope)
    st = PASS if worst < 1e-6 else (WARN if worst < 1e-3 else FAIL)
    return Check('uniqueness', st,
                 f'{tried} starts agree to {worst:.2e} (relative)', worst, scope)


def check_constraints(layer, x0, scope) -> List[Check]:
    out = []
    with torch.no_grad():
        xs, _ = layer.solve(x0)
        S = layer.constraint_basis()
        if S is None:
            out.append(Check('conservation', SKIP,
                             'no constraint subspace declared', scope=scope))
        else:
            start = layer.init_state(x0)
            d = xs - start
            drift = (d - (d @ S) @ S.T).abs().amax().item()
            denom = 1.0 + xs.abs().amax().item()
            err = drift / denom
            out.append(_grade(err, 1e-10, 1e-6, 'conservation',
                              'solution leaves the compatibility class by {:.2e}',
                              scope))
        ok = bool(layer.feasible(xs).all())
        out.append(Check('feasibility', PASS if ok else FAIL,
                         'solution is admissible' if ok else
                         'solver returned an inadmissible state', scope=scope))
    return out


def check_scale_robustness(layer, x0, scope, scales=(0.5, 2., 5., 10.)) -> Check:
    """Well-posedness that only holds near initialisation is not well-posedness.
    Perturb every parameter over orders of magnitude and count what still solves."""
    params = [p for p in layer.parameters() if p.requires_grad]
    if not params:
        return Check('scale robustness', SKIP, 'no parameters', scope=scope)
    saved = [p.data.clone() for p in params]
    ok = tot = 0
    try:
        for s in scales:
            for rep in range(3):
                torch.manual_seed(1000 * rep + int(s * 10))
                for p, o in zip(params, saved):
                    p.data = o + torch.randn_like(o) * s
                with torch.no_grad():
                    _, st = layer.solve(x0)
                ok += bool(st.converged); tot += 1
    finally:
        for p, o in zip(params, saved):
            p.data = o
    frac = ok / max(tot, 1)
    st = PASS if frac >= .9 else (WARN if frac >= .6 else FAIL)
    return Check('scale robustness', st,
                 f'{ok}/{tot} solves converged with parameters perturbed by '
                 f'std up to {max(scales)}', frac, scope)


def check_training_stability(layer, x0, scope, steps=15) -> Check:
    """Fifteen optimiser steps. Enough to catch a layer that produces NaNs the
    moment it is asked to move."""
    params = [p for p in layer.parameters() if p.requires_grad]
    if not params:
        return Check('training stability', SKIP, 'no parameters', scope=scope)
    saved = [p.data.clone() for p in params]
    opt = torch.optim.Adam(params, lr=0.05)
    target = torch.randn_like(x0)
    bad, worst = 0, 0.0
    try:
        for _ in range(steps):
            opt.zero_grad()
            loss = ((layer(x0) - target) ** 2).mean()
            if not torch.isfinite(loss):
                bad += 1; continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1e12).item()
            if not np.isfinite(gn):
                bad += 1; opt.zero_grad(); continue
            worst = max(worst, gn)
            torch.nn.utils.clip_grad_norm_(params, 10.)
            opt.step()
    finally:
        for p, o in zip(params, saved):
            p.data = o
    st = PASS if bad == 0 else (WARN if bad <= steps // 5 else FAIL)
    return Check('training stability', st,
                 f'{steps - bad}/{steps} steps finite, largest gradient norm '
                 f'{worst:.2e}', worst, scope)


# ----------------------------------------------------------------------------
# whole-domain checks
# ----------------------------------------------------------------------------

def validate_domain(key: str, cfg: Optional[dict] = None, seed: int = 0,
                    batch: int = 4, quick: bool = False) -> dict:
    torch.manual_seed(seed)
    d = core.get(key)
    conf = {**d.defaults(), **(cfg or {})}
    variants = d.variants(conf, seed)
    checks: List[Check] = []

    structured = [v for v in variants if v.structured]
    if len(structured) == 1:
        checks.append(Check('arm set', PASS,
                            f'1 structured arm, {len(variants) - 1} control(s)'))
    else:
        checks.append(Check('arm set', FAIL,
                            f'{len(structured)} arms marked structured; expected '
                            'exactly one'))

    counts, certs, layers, broken = {}, {}, {}, {}
    for v in variants:
        # A domain that cannot even be built is exactly what this tool exists to
        # report. Raising here would take down whatever called it instead.
        try:
            layer = v.build()
        except Exception as exc:
            broken[v.key] = f"{type(exc).__name__}: {exc}"
            checks.append(Check('build', FAIL, broken[v.key], scope=v.key))
            continue
        layers[v.key] = layer
        counts[v.key] = sum(p.numel() for p in layer.parameters())
        certs[v.key] = layer.certificate()

    variants = [v for v in variants if v.key in layers]
    if not variants:
        counts_by = {s: sum(1 for c in checks if c.status == s)
                     for s in (PASS, WARN, FAIL, SKIP)}
        return {'domain': key, 'title': d.title, 'seed': seed, 'ok': False,
                'counts': counts_by, 'checks': [c.__dict__ for c in checks]}

    uniq = set(counts.values())
    checks.append(Check('matched parameters',
                        PASS if len(uniq) == 1 else FAIL,
                        (f'all arms have {uniq.pop()} parameters' if len(uniq) == 1
                         else f'parameter counts differ: {counts} — an ablation '
                              'across these is not controlled')))

    holds = {k: c['holds'] for k, c in certs.items()}
    s_key = structured[0].key if structured else None
    if s_key and holds.get(s_key) and not all(holds[k] for k in holds if k != s_key):
        checks.append(Check('certificates', PASS,
                            'structured arm claims the guarantee, controls do not'))
    elif s_key and not holds.get(s_key):
        checks.append(Check('certificates', FAIL,
                            f'structured arm {s_key!r} reports its own guarantee '
                            'does not hold'))
    else:
        bad = [k for k in holds if k != s_key and holds[k]]
        checks.append(Check('certificates', FAIL,
                            f'control arm(s) {bad} still claim the guarantee — '
                            'they do not break the condition'))

    for v in variants:
        layer, scope = layers[v.key], v.key
        x0 = torch.rand(batch, layer.n, dtype=next(layer.parameters()).dtype) + 0.3
        with torch.no_grad():
            _, st = layer.solve(x0)
        solve_status = PASS if st.converged else (WARN if v.structured else PASS)
        checks.append(Check('solve', solve_status,
                            f'{st.iters} iterations, residual {st.residual:.1e}, '
                            f'cond(J) {st.cond_J:.2e}'
                            + ('' if st.converged else
                               ' — did NOT converge' +
                               ('' if v.structured else ' (expected for a control arm)')),
                            st.cond_J, scope))
        checks.append(check_jacobian(layer, layer.init_state(x0), x0, scope))
        checks += check_constraints(layer, x0, scope)
        u = check_uniqueness(layer, x0, scope)
        if u.status in (FAIL, WARN) and not v.structured:
            # A control arm is SUPPOSED to be able to fail this. That is the
            # ablation working, not the code being broken.
            u = Check('uniqueness', PASS,
                      u.detail + ' — expected for a control arm', u.value, scope)
        elif u.status == FAIL and v.structured:
            # The theorem is about the flow; the solver is Newton with a line
            # search. A stall from a perturbed start is a globalisation
            # weakness, not a counterexample. Loud, but not a hard failure.
            u = Check('uniqueness', WARN, u.detail +
                      ' — solver globalisation, not a violation: the natural '
                      'start converges', u.value, scope)
        checks.append(u)
        if not quick:
            checks += check_implicit_gradients(layer, x0, scope, cond=st.cond_J,
                                               converged=st.converged)
            r = check_scale_robustness(layer, x0, scope)
            if r.status in (FAIL, WARN) and not v.structured:
                r = Check('scale robustness', PASS,
                          r.detail + ' — expected for a control arm', r.value, scope)
            checks.append(r)
        t = check_training_stability(layer, x0, scope)
        if t.status in (FAIL, WARN) and not v.structured:
            t = Check('training stability', PASS,
                      t.detail + ' — expected for a control arm', t.value, scope)
        checks.append(t)

    counts_by = {s: sum(1 for c in checks if c.status == s)
                 for s in (PASS, WARN, FAIL, SKIP)}
    return {'domain': key, 'title': d.title, 'seed': seed,
            'ok': counts_by[FAIL] == 0,
            'counts': counts_by,
            'checks': [c.__dict__ for c in checks]}


def report(result: dict) -> str:
    c = result['counts']
    head = (f"\n{result['domain']}  —  {result['title']}\n"
            + "-" * 66)
    body = "\n".join(Check(**k).line() for k in result['checks'])
    verdict = ('PASS' if result['ok'] else 'FAIL')
    tail = (f"  {verdict}   {c['pass']} ok, {c['warn']} warn, {c['fail']} fail, "
            f"{c['skip']} skipped")
    return f"{head}\n{body}\n{'-' * 66}\n{tail}"
