"""
dnn_bench/verdict.py — did the structure do anything?

A sweep produces a table. The table is the evidence, but reading it correctly
takes attention that is easy to skip at 1 a.m., and the two mistakes are
symmetrical: calling a difference real when it sits inside the seed spread, and
missing a real reversal because the structured arm is the one you expected to
win.

So this reads the table for you, and it is deliberately hard to please:

  - A difference only counts if the per-seed RANGES do not overlap. With three
    seeds no significance test has power, so range separation is the honest
    substitute, and it is stated as a screen rather than a result.
  - It reports reversals as loudly as confirmations. The structured arm being
    WORSE than a control is a finding, not a glitch.
  - Accuracy is checked for saturation first. If every arm lands within a few
    points, accuracy is not evidence of anything and is reported as mute rather
    than as agreement.
  - Beating one control is not beating the condition. Separating against some
    controls but not others is reported as partial, with the names.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

#: metric -> (direction, threshold kind, threshold, label)
METRICS = [
    ('mean_solve_iters', 'lower', 'ratio', 1.5, 'solver iterations'),
    ('solve_fail_rate',  'lower', 'diff',  0.15, 'solve failure rate'),
    ('median_cond_J',    'lower', 'ratio', 10.0, 'conditioning of J'),
    ('max_grad_norm',    'lower', 'ratio', 10.0, 'largest gradient norm'),
    ('best_acc',         'higher', 'diff', 0.03, 'test accuracy'),
]
SOLVER_METRICS = {'mean_solve_iters', 'solve_fail_rate', 'median_cond_J',
                  'max_grad_norm'}


def _vals(results, arm, key) -> List[float]:
    out = []
    for r in results:
        if r['arm'] == arm:
            v = r.get(key)
            if v is not None and np.isfinite(v):
                out.append(float(v))
    return out


def _overlap(a: List[float], b: List[float]) -> bool:
    if not a or not b:
        return True
    return not (max(a) < min(b) or max(b) < min(a))


def _cmp(s: List[float], c: List[float], direction: str, kind: str,
         thresh: float) -> dict:
    """Compare the structured arm against one control on one metric."""
    if not s or not c:
        return {'usable': False}
    ms, mc = float(np.median(s)), float(np.median(c))
    overlap = _overlap(s, c)
    if kind == 'ratio':
        lo = min(abs(ms), abs(mc))
        size = (max(abs(ms), abs(mc)) / lo) if lo > 1e-30 else float('inf')
        better = (ms < mc) if direction == 'lower' else (ms > mc)
    else:
        size = abs(ms - mc)
        better = (ms < mc) if direction == 'lower' else (ms > mc)
    big = size >= thresh
    return {'usable': True, 'structured': ms, 'control': mc,
            'size': None if not np.isfinite(size) else float(size),
            'kind': kind, 'overlap': overlap,
            'separated': bool(big and not overlap),
            'structured_better': bool(better),
            'spread_structured': [min(s), max(s)],
            'spread_control': [min(c), max(c)]}


def assess(results: List[dict], summary: Optional[List[dict]] = None) -> dict:
    """Read a finished sweep. Returns per-domain findings plus prose."""
    if not results:
        return {'domains': [], 'lines': ['No results to read.'], 'headline': ''}

    seeds = sorted({r['seed'] for r in results})
    domains, out_lines = [], []

    by_domain: Dict[str, List[dict]] = {}
    for r in results:
        by_domain.setdefault(r['domain'], []).append(r)

    for dom, rs in by_domain.items():
        arms = []
        for r in rs:
            if r['arm'] not in arms:
                arms.append(r['arm'])
        structured = next((r['arm'] for r in rs if r.get('structured')), None)
        controls = [a for a in arms if a != structured]
        if not structured or not controls:
            out_lines.append(
                f"{dom}: needs a structured arm and at least one control to "
                f"say anything. Ran {len(arms)} arm(s).")
            continue

        accs = [v for a in arms for v in _vals(rs, a, 'best_acc')]
        acc_saturated = bool(accs) and (max(accs) - min(accs) < 0.05)

        findings, wins, reversals, partial = [], [], [], []
        for key, direction, kind, thresh, label in METRICS:
            s = _vals(rs, structured, key)
            per_control = {}
            for c in controls:
                per_control[c] = _cmp(s, _vals(rs, c, key), direction, kind, thresh)
            usable = {k: v for k, v in per_control.items() if v.get('usable')}
            if not usable:
                continue
            sep_for = [k for k, v in usable.items()
                       if v['separated'] and v['structured_better']]
            rev_for = [k for k, v in usable.items()
                       if v['separated'] and not v['structured_better']]
            findings.append({'metric': key, 'label': label,
                             'direction': direction,
                             'controls': usable,
                             'separated_against': sep_for,
                             'reversed_against': rev_for,
                             'muted': acc_saturated and key == 'best_acc'})
            if key == 'best_acc' and acc_saturated:
                continue
            if rev_for:
                reversals.append((label, rev_for))
            elif sep_for and len(sep_for) == len(usable):
                wins.append((label, key))
            elif sep_for:
                partial.append((label, sep_for,
                                [k for k in usable if k not in sep_for]))

        if reversals:
            status = 'reversed'
        elif any(k in SOLVER_METRICS for _, k in wins):
            status = 'load_bearing'
        elif partial:
            status = 'partial'
        else:
            status = 'not_separated'

        domains.append({
            'domain': dom, 'structured': structured, 'controls': controls,
            'status': status, 'n_seeds': len(seeds),
            'acc_saturated': acc_saturated, 'findings': findings,
        })
        out_lines += _prose(dom, structured, status, wins, partial, reversals,
                            acc_saturated, len(seeds))

    n = len(seeds)
    if n < 5:
        out_lines.append(
            f"This is a screen, not a result: {n} seed"
            f"{'s' if n != 1 else ''} is too few for any of these differences "
            f"to be more than suggestive. Re-run with ten or more before "
            f"believing a direction.")

    headline = ('; '.join(f"{d['domain']}: {d['status'].replace('_', ' ')}"
                          for d in domains) or 'nothing to read')
    return {'domains': domains, 'lines': out_lines, 'headline': headline,
            'seeds': seeds}


def _prose(dom, structured, status, wins, partial, reversals,
           acc_saturated, n_seeds) -> List[str]:
    out = []
    if status == 'reversed':
        for label, against in reversals:
            out.append(
                f"{dom}: the structured arm was WORSE on {label} than "
                f"{', '.join(against)}, by more than the seed spread. That is a "
                f"finding, not noise to tune away — the guarantee holds and the "
                f"arm still lost. Look at conditioning before anything else.")
    elif status == 'load_bearing':
        names = ', '.join(label for label, _ in wins)
        out.append(
            f"{dom}: the structured arm separated from every control on "
            f"{names}, with no overlap in the per-seed ranges. That is the "
            f"condition doing visible work.")
    elif status == 'partial':
        for label, against, not_against in partial:
            out.append(
                f"{dom}: the structured arm separated on {label} against "
                f"{', '.join(against)} but not against {', '.join(not_against)}. "
                f"Beating one control is not beating the condition — whatever "
                f"{', '.join(not_against)} preserves may be what actually "
                f"matters.")
    else:
        out.append(
            f"{dom}: nothing separated. Every metric's per-seed range overlaps "
            f"between the structured arm and its controls. On this evidence the "
            f"condition is not load-bearing here — which is worth knowing early "
            f"and cheaply, and is the most common honest outcome.")
    if acc_saturated:
        out.append(
            f"{dom}: accuracy is mute — every arm landed within five points, so "
            f"the task cannot discriminate them. Do not read agreement into "
            f"that. Change the task before drawing any accuracy conclusion.")
    return out


def report(v: dict) -> str:
    if not v.get('lines'):
        return "  (no verdict)"
    return "\n".join("  " + l for l in v['lines'])
