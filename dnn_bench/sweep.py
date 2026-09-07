"""Domain-agnostic experiment engine.

An arm is identified by "<domain>.<variant>", so a single sweep can put a
topology-based guarantee, a weight-constraint guarantee and a convexity
guarantee side by side on the same task at comparable width.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from . import core
from .tasks import TASKS
from .store import new_id, save
from .verdict import assess


def _py(o):
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


class Model(nn.Module):
    """encoder -> implicit layer -> readout -> decoder. Domain-free."""

    def __init__(self, layer, in_dim, out_dim, dtype=torch.float64):
        super().__init__()
        self.enc = nn.Linear(in_dim, layer.n, dtype=dtype)
        self.layer = layer
        self.dec = nn.Linear(layer.n, out_dim, dtype=dtype)
        self.positive = layer.feasible(-torch.ones(1, layer.n, dtype=dtype)).item() == 0

    def forward(self, z):
        h = self.enc(z)
        if self.positive:                     # domain lives in the positive orthant
            h = nn.functional.softplus(h) + 1e-2
        return self.dec(self.layer.readout(self.layer(h)))


def arm_keys(domains: List[str], cfg: dict, seed: int = 0) -> List[str]:
    out = []
    for dk in domains:
        d = core.get(dk)
        out += [f"{dk}.{v.key}" for v in d.variants({**d.defaults(), **cfg}, seed)]
    return out


def _variant(arm: str, cfg: dict, seed: int):
    dk, vk = arm.split('.', 1)
    d = core.get(dk)
    for v in d.variants({**d.defaults(), **cfg}, seed):
        if v.key == vk:
            return d, v
    raise KeyError(f"no variant {vk!r} in domain {dk!r}")


def run_arm(arm, cfg, seed, on_epoch=None, should_stop=None) -> dict:
    domain, variant = _variant(arm, cfg, seed)
    layer = variant.build()
    layer.max_iter = cfg.get('max_solver_iter', 300)

    task = TASKS[cfg.get('task', 'spirals')]
    Xtr, ytr, ind, outd = task(cfg.get('n_train', 512), seed=seed)
    Xte, yte, _, _ = task(cfg.get('n_test', 256), seed=seed + 1000)

    torch.manual_seed(seed)
    model = Model(layer, ind, outd)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 0.05))
    lossf = nn.CrossEntropyLoss()
    batch, epochs = cfg.get('batch', 64), cfg.get('epochs', 20)

    hist = {k: [] for k in ('loss', 'acc', 'iters', 'fail', 'gnorm',
                            'residual', 'cond')}
    t0 = time.time()

    for ep in range(epochs):
        if should_stop and should_stop():
            break
        perm = torch.randperm(len(Xtr))
        s = dict(loss=0., it=0, fail=0, gn=0., res=0., cond=[], nb=0)
        nbatch = (len(Xtr) + batch - 1) // batch
        for i in range(0, len(Xtr), batch):
            opt.zero_grad()
            out = model(Xtr[perm[i:i + batch]])
            loss = lossf(out, ytr[perm[i:i + batch]])
            if not torch.isfinite(loss):
                s['fail'] += 1; continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e12).item()
            if not np.isfinite(gn):
                s['fail'] += 1; opt.zero_grad(); continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            st = getattr(layer, 'last_stats', None)
            s['loss'] += loss.item(); s['nb'] += 1; s['gn'] += gn
            s['it'] += st.iters if st else 0
            s['res'] += st.residual if st else 0.
            if st and np.isfinite(st.cond_J):
                s['cond'].append(st.cond_J)
            s['fail'] += 0 if (st and st.converged) else 1

        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).double().mean().item()
        nb = max(s['nb'], 1)
        hist['loss'].append(s['loss'] / nb); hist['acc'].append(acc)
        hist['iters'].append(s['it'] / nb); hist['fail'].append(s['fail'] / nbatch)
        hist['gnorm'].append(s['gn'] / nb); hist['residual'].append(s['res'] / nb)
        hist['cond'].append(float(np.median(s['cond'])) if s['cond'] else float('nan'))
        if on_epoch:
            on_epoch(arm, seed, ep, epochs, hist)

    conds = [c for c in hist['cond'] if np.isfinite(c)]
    cert = layer.certificate()
    return _py({
        'arm': arm, 'domain': domain.key, 'variant': variant.key, 'seed': seed,
        'label': variant.label, 'structured': variant.structured,
        'guarantee_holds': cert['holds'], 'claim': cert['claim'],
        'describe': layer.describe(),
        'n_params': sum(p.numel() for p in model.parameters()),
        'layer_params': sum(p.numel() for p in layer.parameters()),
        'wall_s': round(time.time() - t0, 1),
        'best_acc': max(hist['acc']) if hist['acc'] else float('nan'),
        'final_acc': hist['acc'][-1] if hist['acc'] else float('nan'),
        'mean_solve_iters': float(np.mean(hist['iters'])) if hist['iters'] else float('nan'),
        'solve_fail_rate': float(np.mean(hist['fail'])) if hist['fail'] else float('nan'),
        'mean_grad_norm': float(np.mean(hist['gnorm'])) if hist['gnorm'] else float('nan'),
        'max_grad_norm': float(np.max(hist['gnorm'])) if hist['gnorm'] else float('nan'),
        'median_cond_J': float(np.median(conds)) if conds else float('nan'),
        'max_cond_J': float(np.max(conds)) if conds else float('nan'),
        'hist': hist,
    })


def run_sweep(cfg, on_event=None, should_stop=None) -> List[dict]:
    seeds = cfg.get('seeds', [0, 1, 2])
    arms = cfg.get('arms') or arm_keys(cfg.get('domains', ['crn']), cfg, seeds[0])
    total, done, results = len(seeds) * len(arms), 0, []
    for seed in seeds:
        for arm in arms:
            if should_stop and should_stop():
                return results
            _, variant = _variant(arm, cfg, seed)
            if on_event:
                layer = variant.build()
                on_event(_py({'type': 'arm_start', 'arm': arm, 'seed': seed,
                              'done': done, 'total': total,
                              'label': variant.label,
                              'describe': layer.describe(),
                              'certificate': layer.certificate()}))

            def on_epoch(a, s, ep, eps, h):
                if on_event:
                    on_event(_py({'type': 'epoch', 'arm': a, 'seed': s,
                                  'epoch': ep, 'epochs': eps,
                                  'loss': h['loss'][-1], 'acc': h['acc'][-1],
                                  'iters': h['iters'][-1], 'fail': h['fail'][-1],
                                  'gnorm': h['gnorm'][-1], 'cond': h['cond'][-1],
                                  'done': done, 'total': total}))

            r = run_arm(arm, cfg, seed, on_epoch, should_stop)
            results.append(r); done += 1
            if on_event:
                on_event({'type': 'arm_done', 'result': r,
                          'done': done, 'total': total})
    return results


def aggregate(results: List[dict]) -> List[dict]:
    out, seen = [], []
    for r in results:
        if r['arm'] not in seen:
            seen.append(r['arm'])
    for arm in seen:
        rs = [r for r in results if r['arm'] == arm]
        f = lambda k: [r[k] for r in rs if r.get(k) is not None]
        mean = lambda k: float(np.mean(f(k))) if f(k) else None
        out.append({
            'arm': arm, 'domain': rs[0]['domain'], 'variant': rs[0]['variant'],
            'label': rs[0]['label'], 'structured': rs[0]['structured'],
            'guarantee_holds': rs[0]['guarantee_holds'], 'claim': rs[0]['claim'],
            'n_runs': len(rs), 'layer_params': rs[0]['layer_params'],
            'best_acc': mean('best_acc'),
            'acc_std': float(np.std(f('best_acc'))) if f('best_acc') else None,
            'mean_solve_iters': mean('mean_solve_iters'),
            'solve_fail_rate': mean('solve_fail_rate'),
            'mean_grad_norm': mean('mean_grad_norm'),
            'max_grad_norm': float(np.max(f('max_grad_norm'))) if f('max_grad_norm') else None,
            'median_cond_J': float(np.median(f('median_cond_J'))) if f('median_cond_J') else None,
            'max_cond_J': float(np.max(f('max_cond_J'))) if f('max_cond_J') else None,
            'wall_s': mean('wall_s'),
        })
    return _py(out)
