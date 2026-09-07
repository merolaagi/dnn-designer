"""
dnn_bench/propose.py — paper text in, a candidate StructureSpec out.

This is the only part of the pipeline that exercises judgment, and it is the
part you should trust least. It produces a PROPOSAL. The proposal is worth
having only because `./validate` and the ablation can reject it: a wrong
extraction yields a spec that fails validation loudly, rather than a plausible
architecture that quietly wastes a week.

Read the proposal before you run it. The failure mode to watch for is a spec
that captures the paper's narrative rather than its constraint -- something that
gets built, converges, trains, and encodes nothing.

Needs ANTHROPIC_API_KEY. Without it the pipeline still works: `./paper` prints
the ranked passages and a blank spec for you to fill in by hand, which is the
same artifact by a slower route.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Optional

from .spec import StructureSpec, OPS, TRANSFORMS

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get('DNN_PAPER_MODEL', 'claude-sonnet-4-6')

SYSTEM = """You extract implicit-layer architectures from research papers.

You are given a paper's abstract, its highest-scoring passages, and its
displayed equations. You return ONE JSON object describing a neural layer whose
output is the root of F(x; x0) = 0, plus the ablation arms that test whether the
paper's structural condition is load-bearing.

THE TEST THAT MATTERS. Only propose a structure if the paper gives a condition
that CONSTRAINS the system -- an invariance, a conservation law, a sign
condition, a topological property, a convexity requirement. If the paper only
offers a metaphor or a narrative mechanism, say so and return
{"usable": false, "why": "..."} instead. A structure with a good story and no
constraint is the failure mode here; returning nothing is the correct output far
more often than not.

Ask yourself: does this give me a constraint that a generic MLP violates? If the
honest answer is "it gives a nice way to think about it", return usable: false.

THE ARMS. Exactly one variant is "structured": the one the theorem covers. Every
other variant breaks EXACTLY ONE of the theorem's conditions, and does so by
changing a constant or a parameter transform -- never a shape, because arms must
have identical parameter counts or the ablation is uncontrolled.

THE RESIDUAL. A single Python expression over:
  x    the state, shape (batch, n)
  x0   the layer input, shape (batch, n)
  your declared parameters and constants
Available functions: exp log sqrt abs tanh sigmoid softplus relu sin cos clamp
where sum stack cat eye minimum maximum, plus @ for matmul and .T for transpose.
It must return shape (batch, n). No loops, no imports, no other names.

Return ONLY the JSON object. No markdown fence, no commentary."""

TEMPLATE = """{
  "usable": true,
  "key": "lowercase_identifier",
  "title": "short human name",
  "paper": "citation or arXiv id",
  "claim": "the theorem in one sentence",
  "n": 12,
  "consts": {"K": 16, "alpha": 0.5},
  "params": {"W": {"shape": ["K", "n"], "init": "randn/sqrt(n)"},
             "s": {"shape": ["K"], "init": "randn*0.3", "transform": "softplus"}},
  "residual": "x0 - alpha*x - (sigmoid(x @ W.T) * s) @ W",
  "inject": "residual",
  "positive": false,
  "variants": [
    {"key": "structured", "label": "condition holds", "structured": true},
    {"key": "broken", "label": "condition broken",
     "params": {"s": {"transform": "free"}}}
  ],
  "notes": "what the condition is, and what each control arm breaks",
  "confidence": "high | medium | low, and why"
}

init accepts: zeros | ones | randn | rand | eye, optionally '*<number>' and
'/sqrt(<dim>)'. transform accepts: %s.
inject is "residual" when the input enters F directly, or "state" when it sets
the initial condition (use "state" for conservation-law systems).
Set positive: true when the state must stay in the open positive orthant.""" % (
    ' | '.join(TRANSFORMS))


def available() -> bool:
    return bool(os.environ.get('ANTHROPIC_API_KEY'))


def _call(prompt: str, max_tokens: int = 3000) -> str:
    key = os.environ.get('ANTHROPIC_API_KEY')
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it, or skip the proposer and "
            "write the spec yourself from the ranked passages.")
    body = json.dumps({
        'model': MODEL, 'max_tokens': max_tokens, 'system': SYSTEM,
        'messages': [{'role': 'user', 'content': prompt}],
    }).encode()
    req = urllib.request.Request(
        API_URL, data=body,
        headers={'content-type': 'application/json', 'x-api-key': key,
                 'anthropic-version': '2023-06-01'})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    return "".join(b.get('text', '') for b in data.get('content', [])
                   if b.get('type') == 'text')


def _parse(raw: str) -> dict:
    txt = re.sub(r'^\s*```(?:json)?|```\s*$', '', raw.strip(), flags=re.M).strip()
    start, end = txt.find('{'), txt.rfind('}')
    if start < 0 or end < 0:
        raise ValueError(f"proposer did not return JSON:\n{raw[:400]}")
    return json.loads(txt[start:end + 1])


def propose(paper, extra: str = '') -> dict:
    """Returns {'usable': bool, 'spec': StructureSpec|None, 'raw': dict,
    'problems': [str]}. `problems` is what spec.check() says about it -- a
    proposal that fails these is not silently repaired."""
    prompt = (f"PAPER: {paper.title}\nSOURCE: {paper.source}\n\n"
              f"{paper.brief()}\n\n"
              + (f"OPERATOR NOTES:\n{extra}\n\n" if extra else '')
              + f"Return one JSON object in exactly this shape:\n{TEMPLATE}")
    raw = _parse(_call(prompt))

    if not raw.get('usable', True):
        return {'usable': False, 'spec': None, 'raw': raw,
                'problems': [raw.get('why', 'proposer judged the paper '
                                             'unsuitable for this harness')]}
    payload = {k: v for k, v in raw.items()
               if k in StructureSpec.__dataclass_fields__}
    payload.setdefault('key', 'proposed')
    payload['source'] = f"{paper.source} (proposed, unverified)"
    notes = payload.get('notes', '')
    if raw.get('confidence'):
        notes = (notes + f"\n\nProposer confidence: {raw['confidence']}").strip()
    payload['notes'] = notes
    try:
        spec = StructureSpec.from_dict(payload)
    except Exception as exc:
        return {'usable': False, 'spec': None, 'raw': raw,
                'problems': [f"proposal is not a well-formed spec: {exc}"]}
    return {'usable': True, 'spec': spec, 'raw': raw,
            'problems': spec.check()}


def blank_spec(paper) -> StructureSpec:
    """The same artifact, empty, for filling in by hand when there is no key
    set or when you do not want a machine guessing at the structure."""
    return StructureSpec(
        key='my_structure', title=paper.title[:80] or 'untitled',
        paper=paper.source, claim='<the theorem, in one sentence>',
        n=12, consts={'K': 16},
        params={'W': {'shape': ['K', 'n'], 'init': 'randn/sqrt(n)'}},
        residual='<expression in x, x0, W ... returning shape (batch, n)>',
        variants=[{'key': 'structured', 'label': '<condition holds>',
                   'structured': True},
                  {'key': 'broken', 'label': '<condition broken>',
                   'params': {'W': {'transform': 'free'}}}],
        notes='Filled in by hand. Run ./validate before trusting it.',
        source=f'{paper.source} (hand-written)')
