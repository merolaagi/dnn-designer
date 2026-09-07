"""
dnn_bench/spec.py — declare a domain instead of coding it.

A hand-written domain has to supply `residual` AND a matching `jacobian`, and a
Jacobian with a sign error is the most dangerous defect in this codebase: the
solver still converges, so the only symptom is silently wrong gradients. That
already happened once here.

A spec removes the risk by construction. You write only the residual, as an
expression; the Jacobian comes from `torch.func.jacrev`, so it cannot disagree
with it. What is left to get wrong is the physics, which is what `./validate`
and the ablation are for.

A spec is JSON, which matters for the ingestion pipeline: extracting a structure
from a paper becomes "produce this object" rather than "write correct PyTorch".

    {
      "key": "convex_ridge",
      "title": "Convex ridge potential",
      "paper": "...",
      "claim": "A strongly convex potential has exactly one stationary point.",
      "n": 12,
      "params": {
        "W": {"shape": ["K", "n"], "init": "randn/sqrt(n)"},
        "b": {"shape": ["K"], "init": "zeros"},
        "s": {"shape": ["K"], "init": "randn*0.3", "transform": "softplus"}
      },
      "consts": {"K": 16, "alpha": 0.5},
      "residual": "x0 - alpha*x - (sigmoid(x @ W.T + b) * s) @ W",
      "variants": [
        {"key": "convex", "label": "coefficients >= 0", "structured": true},
        {"key": "signed", "label": "convexity broken",
         "params": {"s": {"transform": "free"}}},
        {"key": "flat",   "label": "convex, not strongly", "consts": {"alpha": 0}}
      ]
    }

Variants override consts and parameter transforms, never shapes -- so parameter
counts stay matched across arms and the ablation stays controlled. The loader
enforces that.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .core import ImplicitLayer, Variant, register

# ----------------------------------------------------------------------------
# the expression sandbox
# ----------------------------------------------------------------------------

def _softplus(z):
    return Fn.softplus(z)


#: Everything a residual expression may refer to. Deliberately small: an
#: expression that needs something outside this list is a sign the structure
#: wants a hand-written domain, not a spec.
OPS: Dict[str, Any] = {
    'exp': torch.exp, 'log': torch.log, 'sqrt': torch.sqrt, 'abs': torch.abs,
    'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'softplus': _softplus,
    'relu': torch.relu, 'sin': torch.sin, 'cos': torch.cos,
    'clamp': torch.clamp, 'where': torch.where, 'sum': torch.sum,
    'stack': torch.stack, 'cat': torch.cat, 'eye': torch.eye,
    'minimum': torch.minimum, 'maximum': torch.maximum,
    'pi': math.pi, 'e': math.e,
}

TRANSFORMS = {
    'free': lambda t: t,
    'softplus': _softplus,
    'exp': torch.exp,
    'abs': torch.abs,
    'square': lambda t: t * t,
}

_INIT = re.compile(r'^\s*(zeros|ones|randn|rand|eye)'
                   r'(?:\s*\*\s*([0-9.eE+-]+))?'
                   r'(?:\s*/\s*sqrt\(\s*([A-Za-z_]\w*)\s*\))?\s*$')


def _make_init(expr: str, shape, dims: Dict[str, int], dtype):
    m = _INIT.match(expr or 'randn')
    if not m:
        raise ValueError(
            f"init {expr!r} not understood. Use zeros | ones | randn | rand | "
            f"eye, optionally '*<number>' and '/sqrt(<dim>)'.")
    kind, mul, div = m.group(1), m.group(2), m.group(3)
    t = {'zeros': torch.zeros, 'ones': torch.ones,
         'randn': torch.randn, 'rand': torch.rand}.get(kind)
    out = (torch.eye(shape[0], dtype=dtype) if kind == 'eye'
           else t(*shape, dtype=dtype))
    if mul:
        out = out * float(mul)
    if div:
        if div not in dims:
            raise ValueError(f"init refers to unknown dimension {div!r}")
        out = out / math.sqrt(dims[div])
    return out


# ----------------------------------------------------------------------------

@dataclass
class StructureSpec:
    key: str
    title: str
    paper: str = ''
    claim: str = ''
    n: int = 12
    residual: str = ''
    params: Dict[str, dict] = field(default_factory=dict)
    consts: Dict[str, float] = field(default_factory=dict)
    variants: List[dict] = field(default_factory=list)
    positive: bool = False          #: state confined to the open positive orthant
    fixed_n: bool = False           #: refuse external overrides of `n`
    inject: str = 'residual'        #: 'residual' | 'state' -- how x0 enters
    readout: str = 'identity'       #: 'identity' | 'log'
    notes: str = ''
    source: str = ''                #: where this spec came from

    @staticmethod
    def load(path: str) -> 'StructureSpec':
        with open(path) as fh:
            return StructureSpec.from_dict(json.load(fh))

    @staticmethod
    def from_dict(d: dict) -> 'StructureSpec':
        known = StructureSpec.__dataclass_fields__
        unknown = set(d) - set(known)
        if unknown:
            raise ValueError(f"unknown spec field(s): {sorted(unknown)}")
        return StructureSpec(**d)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def check(self) -> List[str]:
        """Structural problems worth catching before anything is built.
        Returns a list of human-readable complaints; empty means usable."""
        bad = []
        if not self.key or not re.match(r'^[a-z][a-z0-9_]*$', self.key):
            bad.append("key must be a lowercase identifier")
        res = self.residual.strip()
        if not res:
            bad.append("residual expression is empty")
        elif '<' in res and '>' in res:
            bad.append("residual is still the placeholder from the blank "
                       "template — write the expression before saving. The "
                       "layer's output is the x where it equals zero; x0 is "
                       "the input.")
        else:
            # Compile it here rather than at build time. A spec that cannot
            # compile must never register: it loads fine, then takes down the
            # next thing that touches it, far from the cause.
            try:
                compile(res, f'<residual:{self.key}>', 'eval')
            except SyntaxError as exc:
                bad.append(f"residual is not a valid expression: {exc.msg} "
                           f"(at offset {exc.offset})")
        if not self.variants:
            bad.append("no variants: an ablation needs a structured arm and "
                       "at least one control")
        s = [v for v in self.variants if v.get('structured')]
        if len(s) != 1:
            bad.append(f"{len(s)} variants marked structured; expected exactly 1")
        if len(self.variants) < 2:
            bad.append("only one variant: nothing to compare the structured "
                       "arm against")
        for v in self.variants:
            for pname, over in (v.get('params') or {}).items():
                if pname not in self.params:
                    bad.append(f"variant {v.get('key')!r} overrides unknown "
                               f"parameter {pname!r}")
                elif 'shape' in over:
                    bad.append(f"variant {v.get('key')!r} changes the shape of "
                               f"{pname!r}; that breaks matched parameter "
                               f"counts and invalidates the ablation")
                elif over.get('transform') not in (None, *TRANSFORMS):
                    bad.append(f"variant {v.get('key')!r}: unknown transform "
                               f"{over.get('transform')!r}")
        for v in self.variants:
            if '<' in str(v.get('label', '')) and '>' in str(v.get('label', '')):
                bad.append(f"variant {v.get('key')!r} still has a placeholder "
                           f"label — say what condition it breaks")
        if not self.fixed_n and re.search(r'x0?\s*\[', self.residual):
            bad.append("the residual slices the state, so it is written for a "
                       "specific width — set \"fixed_n\": true or it will be "
                       "resized by a caller and return the wrong shape")
        for pname, p in self.params.items():
            if 'shape' not in p:
                bad.append(f"parameter {pname!r} has no shape")
            if p.get('transform') not in (None, *TRANSFORMS):
                bad.append(f"parameter {pname!r}: unknown transform "
                           f"{p.get('transform')!r}")
        return bad


# ----------------------------------------------------------------------------

class SpecLayer(ImplicitLayer):
    """An implicit layer built from a StructureSpec.

    `jacobian` is inherited from ImplicitLayer, which differentiates `residual`
    directly -- so it agrees with the residual by construction. That removes the
    entire class of defect where a hand-derived Jacobian quietly disagrees.
    """

    def __init__(self, spec: StructureSpec, variant: dict,
                 dtype=torch.float64, tol=1e-10, max_iter=200):
        super().__init__()
        self.spec, self.variant = spec, variant
        self.n, self.tol, self.max_iter = spec.n, tol, max_iter
        self.consts = {**spec.consts, **(variant.get('consts') or {})}
        dims = {'n': spec.n, **{k: v for k, v in self.consts.items()
                                if isinstance(v, int)}}
        self.transforms: Dict[str, str] = {}
        for name, p in spec.params.items():
            over = (variant.get('params') or {}).get(name, {})
            shape = [dims.get(s, s) if isinstance(s, str) else s
                     for s in p['shape']]
            if any(not isinstance(s, int) for s in shape):
                raise ValueError(f"parameter {name!r}: shape {p['shape']} refers "
                                 f"to a dimension that is not in consts")
            self.register_parameter(
                name, nn.Parameter(_make_init(p.get('init', 'randn'),
                                              shape, dims, dtype)))
            self.transforms[name] = over.get('transform',
                                             p.get('transform', 'free'))
        self._code = compile(spec.residual, f'<residual:{spec.key}>', 'eval')

    def _env(self, x, x0):
        env = dict(OPS)
        env.update({k: float(v) for k, v in self.consts.items()})
        for name in self.spec.params:
            env[name] = TRANSFORMS[self.transforms[name]](getattr(self, name))
        env['x'], env['x0'] = x, x0
        return env

    def residual(self, x, x0):
        out = eval(self._code, {'__builtins__': {}}, self._env(x, x0))
        if out.shape != x.shape:
            raise ValueError(
                f"residual for {self.spec.key!r} returned shape "
                f"{tuple(out.shape)}, expected {tuple(x.shape)}")
        return out

    def init_state(self, x0):
        if self.spec.inject == 'state':
            return x0
        return torch.zeros_like(x0)

    def readout(self, x):
        return torch.log(x.clamp_min(1e-300)) if self.spec.readout == 'log' else x

    def feasible(self, x):
        if self.spec.positive:
            return (x > 0).all(dim=1)
        return super().feasible(x)

    def step_ok(self, x, xn):
        if self.spec.positive:
            return bool((xn > 1e-9 * self._solve_ref).all())
        return super().step_ok(x, xn)

    def certificate(self):
        holds = bool(self.variant.get('structured'))
        return {'holds': holds,
                'claim': (self.spec.claim if holds else
                          f"condition broken: {self.variant.get('label', '')}")}

    def describe(self):
        return {'summary': f"{self.spec.key}.{self.variant['key']} — n={self.n}",
                'facts': [(str(self.n), 'state width'),
                          (str(sum(p.numel() for p in self.parameters())),
                           'trainable parameters'),
                          (self.transforms.get(
                              next(iter(self.spec.params), ''), '—'),
                           'transform on the first parameter'),
                          ('yes' if self.spec.positive else 'no',
                           'state confined to positive orthant')],
                'expression': self.spec.residual}


class SpecDomain:
    def __init__(self, spec: StructureSpec):
        problems = spec.check()
        if problems:
            raise ValueError(f"spec {spec.key!r} is not usable:\n  - "
                             + "\n  - ".join(problems))
        self.spec = spec
        self.key, self.title = spec.key, spec.title
        self.paper, self.claim = spec.paper, spec.claim

    def defaults(self):
        return {'n_species': self.spec.n}

    def variants(self, cfg, seed):
        # A component-wise residual -- one that slices x[:, 0:1] -- is written
        # for a specific state width, so an external n_species must not silently
        # resize it. Such a spec declares fixed_n.
        n = (self.spec.n if self.spec.fixed_n
             else cfg.get('n_species', self.spec.n))
        out = []
        for v in self.spec.variants:
            def build(v=v, n=n):
                torch.manual_seed(seed)
                s = StructureSpec(**{**self.spec.to_dict(), 'n': n})
                return SpecLayer(s, v)
            out.append(Variant(v['key'], v.get('label', v['key']), build,
                               structured=bool(v.get('structured'))))
        return out


# ----------------------------------------------------------------------------

def load_spec_file(path: str):
    d = SpecDomain(StructureSpec.load(path))
    register(d)
    return d


def load_spec_dir(directory: str) -> List[str]:
    """Register every *.json in a directory as a domain. Bad specs are
    reported and skipped rather than taking the whole bench down with them."""
    loaded, problems = [], []
    if not os.path.isdir(directory):
        return loaded
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith('.json'):
            continue
        path = os.path.join(directory, fn)
        try:
            load_spec_file(path)
            loaded.append(fn)
        except Exception as exc:
            problems.append(f"{fn}: {exc}")
    if problems:
        import warnings
        warnings.warn("spec(s) skipped:\n  " + "\n  ".join(problems))
    return loaded
