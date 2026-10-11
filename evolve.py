# Portions adapted from rrsi (github.com/google-research/rrsi): the edit-budget schedule,
# the history summaries and the admission rules. Copyright 2026 Google LLC, licensed under
# the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
"""Evolve an agent design: regularized recursive self-improvement of its harness.

After RRSI (Xia et al., "Regularized Recursive Self-Improvement of Agent
Harnesses", 2026; github.com/google-research/rrsi, Apache 2.0), applied to the
Agent lab's designs. The design on the canvas is the harness: its prompts,
settings, blocks and wires around a frozen model. A model proposes edits to it,
each candidate is run on a set of cases, and the regularizers decide what is
kept:

Proposal side (Algorithm 1)
- an annealed edit budget b_t caps how many independent edits one candidate may
  bundle: several early, one at a time late, so late gains are attributable;
- the proposer sees the whole edit history, so a falsified idea is not redrawn;
- when the score stalls, a candidate slot is reserved for a component the run
  has never tried;
- a critic screens every candidate before it is run, for edits that encode the
  cases themselves (their wording, their expected answers) instead of a general
  improvement.

Selection side (Algorithm 2)
- a noise-adjusted floor: a candidate must score at least S* - delta, where
  delta is measured from the noise of the starting design's own evaluation;
- a gain bigger than delta may cost more tokens only in proportion to the gain
  (beta0 + beta1 * dS); inside the band a candidate is kept only for a token
  saving or a new kind of machinery;
- the best admissible candidate becomes the incumbent, else the incumbent stays;
- components whose recent edits have not helped are offered for pruning.

Held-out cases, never seen by the search, measure the start and the end, so an
evolved design that only memorised its evolve set shows it.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics as st
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import agentlab as AL
import providers as P

# --------------------------------------------------------------------------
# configuration (paper Appendix "Hyperparameters"; rrsi/config.py)
# --------------------------------------------------------------------------

DEFAULTS = {"T": 6, "k": 2, "m": 2, "b_min": 1, "b_max": 3, "w": 2, "m_draft": 1, "delta": None,
            "delta_z": 2.0, "beta0": 0.10, "beta1": 40.0, "w_s": 100.0, "w_c": 15.0, "w_n": 0.5,
            "n_prune": 4, "repair_rounds": 2, "critic_model": True}
LIMITS = {"T": (1, 30), "k": (1, 10), "m": (1, 4), "b_min": (1, 6), "b_max": (1, 6), "w": (1, 10),
          "m_draft": (0, 4), "n_prune": (1, 20), "repair_rounds": (0, 5)}

# the editable components K of this harness, and the structural subset K_str (rrsi/components.py)
K = ["prompt", "config", "context_mgmt", "control_flow", "client_tool", "memory", "subagent"]
K_STR = ["client_tool", "memory", "subagent"]
COMPONENT_OF = {"system_prompt": "prompt", "summarizer": "context_mgmt", "short_mem": "context_mgmt",
                "long_mem": "memory", "retriever": "memory", "web_search": "client_tool", "code_exec": "client_tool",
                "tool": "client_tool", "mcp": "client_tool", "train_net": "client_tool", "designer": "client_tool",
                "experiment": "client_tool", "sub_agent": "subagent", "subgraph": "subagent",
                "reflector": "control_flow", "planner": "control_flow", "router": "control_flow",
                "guard_in": "control_flow", "guard_out": "control_flow", "human": "control_flow",
                "parallel": "control_flow", "llm": "config", "loop": "config"}
TEXT_PARAMS = {"text", "standards", "description", "role"}
# blocks an edit may add, and how each one is wired in
ATTACH = {"summarizer": "core", "web_search": "router", "code_exec": "router"}
BETWEEN = {"long_mem", "planner", "reflector", "guard_out"}


def _cfg(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    for k, v in (raw or {}).items():
        if k in DEFAULTS and v is not None and v != "":
            cfg[k] = v
    for k, (lo, hi) in LIMITS.items():
        cfg[k] = max(lo, min(hi, int(cfg[k])))
    cfg["b_max"] = max(cfg["b_min"], cfg["b_max"])
    cfg["m_draft"] = min(cfg["m_draft"], cfg["m"])
    for k in ("delta_z", "beta0", "beta1", "w_s", "w_c", "w_n"):
        cfg[k] = float(cfg[k])
    if cfg["delta"] is not None:
        cfg["delta"] = max(0.0, float(cfg["delta"]))
    cfg["critic_model"] = bool(cfg["critic_model"])
    return cfg


# --------------------------------------------------------------------------
# schedule, history summaries, selection (rrsi/schedule.py, history.py, selection.py)
# --------------------------------------------------------------------------

def edit_budget(t: int, T: int, b_min: int, b_max: int) -> int:
    """b_t = ceil(b_min + (b_max - b_min) * (1 + cos(pi t / T)) / 2)."""
    if T <= 0:
        return int(b_max)
    t = max(0, min(int(t), int(T)))
    v = b_min + (b_max - b_min) * 0.5 * (1.0 + math.cos(math.pi * t / T))
    return int(math.ceil(round(v, 9)))


def stall_flag(trajectory: List[float], t: int, w: int, delta: float) -> int:
    """sigma_t = 1[S_t - S_{t-w} <= delta]; 0 while fewer than w rounds exist."""
    if t < w or t >= len(trajectory):
        return 0
    return int(trajectory[t] - trajectory[t - w] <= delta)


def measured(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in history if r.get("delta_S") is not None and r.get("component") in K]


def tried(history) -> set:
    return {r["component"] for r in measured(history)}


def incumbent_counts(history) -> Dict[str, int]:
    counts = {c: 0 for c in K}
    for r in history:
        if r.get("accepted") and r.get("component") in counts:
            counts[r["component"]] += 1
    return counts


def prune_set(history, t: int, n_prune: int) -> List[Dict[str, Any]]:
    """B_t: tried components whose best recent gain is <= 0, with the accepted edits that put them in."""
    g = {c: -math.inf for c in tried(history)}
    for r in measured(history):
        if t - int(r["t"]) <= n_prune:
            g[r["component"]] = max(g.get(r["component"], -math.inf), float(r["delta_S"]))
    out = []
    for c in sorted(g):
        if g[c] <= 0:
            out.append({"component": c, "recent_best_gain": None if g[c] == -math.inf else round(g[c], 4),
                        "accepted_in_incumbent": [r["hypothesis"] for r in history
                                                  if r.get("accepted") and r.get("component") == c]})
    return out


def exploration(stall: int, done: set, m_draft: int) -> Dict[str, Any]:
    untried = [c for c in K if c not in done]
    if stall and untried:
        text = (f"STALL: the score has not moved by more than the noise band over the last rounds. {m_draft} "
                f"candidate slot(s) this round are reserved for components the run has never tried: {untried}.")
    elif untried:
        text = f"Components not yet tried in this run: {untried}. Not required this round."
    else:
        text = "Every component has been tried at least once."
    return {"sigma": stall, "untried": untried, "text": text}


def novelty(components: List[str], counts: Dict[str, int]) -> int:
    """nu: structural components the candidate touches that no accepted edit has touched yet."""
    return sum(1 for c in set(components) if c in K_STR and counts.get(c, 0) == 0)


def relative_cost_change(c_new, c_old) -> float:
    return 0.0 if not c_new or not c_old else (c_new - c_old) / c_old


def cost_rule(dS: float, dC: float, nov: int, delta: float, cfg) -> Tuple[bool, str]:
    if dS > delta:
        budget = cfg["beta0"] + cfg["beta1"] * dS
        ok = dC <= budget
        return ok, (f"gain {dS:+.3f} beats the noise band {delta:.3f}; tokens {dC:+.0%} "
                    f"{'within' if ok else 'over'} the allowance {budget:+.0%}")
    shaped = cfg["w_s"] * dS - cfg["w_c"] * dC + cfg["w_n"] * nov
    ok = shaped > 0
    return ok, (f"gain {dS:+.3f} is inside the noise band {delta:.3f}; kept only for a saving or new machinery: "
                f"{cfg['w_s']:g}·dS − {cfg['w_c']:g}·dC + {cfg['w_n']:g}·ν = {shaped:+.3f}")


def judge(ev: Optional[Dict[str, Any]], inc: Dict[str, Any], s_star: float, delta: float, cfg,
          components: List[str], counts: Dict[str, int]) -> Dict[str, Any]:
    if ev is None:
        return {"admissible": False, "reason": "not run"}
    dS = ev["S"] - inc["S"]
    dC = relative_cost_change(ev["C"], inc["C"])
    nov = novelty(components, counts)
    out = {"admissible": False, "S": ev["S"], "C": ev["C"], "delta_S": dS, "delta_C": dC, "novelty": nov}
    if ev["S"] < s_star - delta:
        out["reason"] = f"below the floor: {ev['S']:.3f} < best {s_star:.3f} − noise {delta:.3f}"
        return out
    ok, why = cost_rule(dS, dC, nov, delta, cfg)
    out["reason"] = ("admissible: " if ok else "cost rule failed: ") + why
    out["admissible"] = ok
    return out


# --------------------------------------------------------------------------
# evaluation (rrsi/evaluate.py) and the noise band (rrsi/calibrate.py)
# --------------------------------------------------------------------------

def _reward(checks: List[Dict[str, Any]], result: Dict[str, Any]) -> float:
    graded = [c for c in checks if c.get("passed") is not None]
    if not graded:
        return 1.0 if result.get("ok") and result.get("answer") else 0.0
    return sum(1 for c in graded if c["passed"]) / len(graded)


def evaluate(graph, cases, k: int, mode: str, approvals: str, stop=None, on_trial=None) -> Dict[str, Any]:
    """S = mean reward over every trial (a crashed trial scores 0), C = mean tokens per trial."""
    per_case = []
    for ci, case in enumerate(cases):
        row = {"case": ci, "rewards": [], "tokens": [], "errors": [], "answers": [], "failed": []}
        for _ in range(k):
            if stop is not None and stop.is_set():
                raise InterruptedError("stopped")
            result = AL.run(graph, case["task"], mode, approvals, source="evolve")
            checks = AL.score(result, case["checks"], case, mode)
            c = result.get("costs") or {}
            row["rewards"].append(round(_reward(checks, result), 4))
            row["tokens"].append((c.get("input_total") or 0) + (c.get("output_total") or 0))
            row["errors"].append(result.get("error"))
            row["answers"].append((result.get("answer") or "")[:800])
            row["failed"].append([{"check": f"{x['kind']}: {x['value']}"[:160], "why": (x.get("reason") or "")[:200]}
                                  for x in checks if x.get("passed") is False])
            if on_trial:
                on_trial()
        per_case.append(row)
    rewards = [r for row in per_case for r in row["rewards"]]
    tokens = [t for row in per_case for t in row["tokens"] if t]
    return {"S": round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
            "C": round(sum(tokens) / len(tokens), 1) if tokens else None,
            "k": k, "per_case": per_case}


def noise_band(ev: Dict[str, Any], z: float, reps: int = 2000, seed: int = 7) -> Dict[str, Any]:
    """delta = z · sqrt(2) · se(S), se bootstrapped over trials within each case
    (over cases when each case ran once): the spread of the difference between
    two independent evaluations of the same design."""
    rng = random.Random(seed)
    rows = [row["rewards"] for row in ev["per_case"] if row["rewards"]]
    if not rows:
        return {"delta": 0.0, "se": 0.0, "method": "no trials"}
    within = ev["k"] >= 2
    vals = []
    for _ in range(reps):
        if within:
            picks = [r[rng.randrange(len(r))] for r in rows for _ in r]
        else:
            picks = [rows[rng.randrange(len(rows))][0] for _ in rows]
        vals.append(sum(picks) / len(picks))
    se = st.pstdev(vals) if len(vals) > 1 else 0.0
    return {"delta": round(z * math.sqrt(2) * se, 6), "se": round(se, 6), "z": z,
            "method": "bootstrap over trials within each case" if within else "bootstrap over cases"}


# --------------------------------------------------------------------------
# edits: operations on the design, and their components
# --------------------------------------------------------------------------

def _nodes(g):
    return {n["id"]: n for n in g["nodes"]}


def _new_id(g) -> str:
    used = {n["id"] for n in g["nodes"]}
    i = len(used) + 1
    while f"x{i}" in used:
        i += 1
    return f"x{i}"


def _core_for(g, router_id=None) -> Optional[str]:
    nodes = _nodes(g)
    if router_id:
        for e in g["edges"]:
            if e["target"] == router_id and nodes.get(e["source"], {}).get("type") == "llm":
                return e["source"]
    return next((n["id"] for n in g["nodes"] if n["type"] == "llm"), None)


def _coerce(kind: str, value, options=None):
    if kind == "number":
        v = float(value)
        return int(v) if v.is_integer() else v
    if kind == "choice" and options and value not in options:
        raise ValueError(f"must be one of {options}")
    return value if isinstance(value, (str, int, float, bool)) else json.dumps(value)


def apply_ops(graph, ops: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """The design after the operations, the problems met, and the components touched."""
    g = json.loads(json.dumps(graph))
    problems, comps = [], []
    for op in ops or []:
        kind = str(op.get("op") or "")
        nodes = _nodes(g)
        try:
            if kind == "set":
                n = nodes.get(op.get("block"))
                if n is None:
                    raise ValueError(f"there is no block {op.get('block')}")
                spec = AL.BLOCKS[n["type"]].get("params", {}).get(op.get("param"))
                if spec is None:
                    raise ValueError(f"{n['type']} has no setting {op.get('param')}; it has "
                                     f"{sorted(AL.BLOCKS[n['type']].get('params', {}))}")
                n["params"] = {**n.get("params", {}), op["param"]: _coerce(spec["kind"], op.get("value"), spec.get("options"))}
                comps.append("prompt" if op["param"] in TEXT_PARAMS else COMPONENT_OF.get(n["type"], "config"))
            elif kind == "remove":
                n = nodes.get(op.get("block"))
                if n is None:
                    raise ValueError(f"there is no block {op.get('block')}")
                if n["type"] not in AL.REMOVABLE:
                    raise ValueError(f"a {n['type']} block cannot be removed")
                g = AL._without(g, n["id"])
                comps.append(COMPONENT_OF.get(n["type"], "control_flow"))
            elif kind == "add":
                t = str(op.get("type") or "")
                if t not in ATTACH and t not in BETWEEN:
                    raise ValueError(f"a {t or '?'} block cannot be added; addable: {sorted(set(ATTACH) | BETWEEN)}")
                nid = _new_id(g)
                params = AL.defaults(t)
                for p, v in (op.get("params") or {}).items():
                    spec = AL.BLOCKS[t].get("params", {}).get(p)
                    if spec is None:
                        raise ValueError(f"{t} has no setting {p}")
                    params[p] = _coerce(spec["kind"], v, spec.get("options"))
                if t in ATTACH:
                    target = nodes.get(op.get("attach_to"))
                    want = "llm" if ATTACH[t] == "core" else "router"
                    if target is None or target["type"] != want:
                        raise ValueError(f"attach a {t} to an {want} block (attach_to)")
                    x, y = target.get("x", 0) + 40, target.get("y", 0) + 150
                    g["nodes"].append({"id": nid, "type": t, "x": x, "y": y, "params": params})
                    if want == "llm":
                        if any(nodes[e["source"]]["type"] == t for e in g["edges"] if e["target"] == target["id"]):
                            raise ValueError(f"that core already has a {t}")
                        g["edges"].append({"id": f"ev_{nid}_a", "source": nid, "target": target["id"]})
                    else:
                        core = _core_for(g, target["id"])
                        g["edges"] += [{"id": f"ev_{nid}_a", "source": target["id"], "target": nid},
                                       {"id": f"ev_{nid}_b", "source": nid, "target": core}]
                else:
                    a, b = (op.get("between") or [None, None])[:2]
                    edge = next((e for e in g["edges"] if e["source"] == a and e["target"] == b), None)
                    if edge is None:
                        raise ValueError(f"put a {t} between two wired blocks: between [from, to], and there is "
                                         f"no wire {a} → {b}")
                    na, nb = nodes[a], nodes[b]
                    g["nodes"].append({"id": nid, "type": t, "x": (na.get("x", 0) + nb.get("x", 0)) // 2,
                                       "y": (na.get("y", 0) + nb.get("y", 0)) // 2 + 60, "params": params})
                    g["edges"] = [e for e in g["edges"] if e is not edge]
                    g["edges"] += [{"id": f"ev_{nid}_a", "source": a, "target": nid},
                                   {"id": f"ev_{nid}_b", "source": nid, "target": b}]
                    if t == "reflector":
                        core = _core_for(g, a if na["type"] == "router" else None)
                        g["edges"].append({"id": f"ev_{nid}_c", "source": nid, "target": core})
                comps.append(COMPONENT_OF.get(t, "control_flow"))
            else:
                raise ValueError(f"unknown operation {kind!r}; use set, add or remove")
        except (ValueError, TypeError, KeyError) as exc:
            problems.append(f"{json.dumps(op)[:160]}: {exc}")
    if not problems:
        problems += [p["message"] for p in AL.validate(g) if p["level"] == "error"]
    return g, problems, comps


def describe_design(graph) -> Dict[str, Any]:
    """The design as the proposer reads it: blocks with their settings, and the wires."""
    blocks = []
    for n in graph["nodes"]:
        spec = AL.BLOCKS[n["type"]]
        params = AL._params(n)
        blocks.append({"id": n["id"], "type": n["type"], "name": spec.get("name"),
                       "settings": {k: params.get(k) for k in spec.get("params", {})}})
    return {"blocks": blocks, "wires": [[e["source"], e["target"]] for e in graph["edges"]]}


# --------------------------------------------------------------------------
# the critic's leakage screen (rrsi/critic.py)
# --------------------------------------------------------------------------

def _shingles(text: str, n: int = 6) -> set:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def precheck(edits: List[Dict[str, Any]], cases) -> List[str]:
    """Deterministic: an edit may not quote a case's wording or carry one of its expected answers."""
    added = " ".join(str(op.get("value", "")) + " " + json.dumps(op.get("params") or {})
                     for e in edits for op in e.get("ops") or [])
    hits = []
    mine = _shingles(added)
    for i, case in enumerate(cases, 1):
        if mine & _shingles(case["task"]):
            hits.append(f"an edit quotes the wording of case {i}")
        for c in case["checks"]:
            v = str(c.get("value") or "").strip()
            if c["kind"] in ("contains", "regex") and len(v) >= 2 and re.search(
                    r"(?<![\w.])" + re.escape(v) + r"(?!\w)(?!\.\d)", added, re.IGNORECASE):
                hits.append(f"an edit contains case {i}'s expected answer {v!r}")
        for ref in (case.get("reference"), case.get("bad")):
            if ref and _shingles(ref, 5) & _shingles(added, 5):
                hits.append(f"an edit quotes case {i}'s reference answer")
    return sorted(set(hits))


CRITIC_SYSTEM = """You review a proposed change to an AI agent's design (its harness: prompts, settings,
blocks), in a loop that evolves the design against the same cases it is scored on. Catch changes that
would be cheating or self-destructive. REJECT if any of:
1. LEAKAGE: the change encodes knowledge of particular cases: their names, their wording, expected answers,
   magic constants that only fit one case, or branching on case-identifying features. General practice and
   general procedure are fine. Test: would it still make sense and still help on an unfamiliar task of the
   same kind?
2. DEGENERATE: a no-op dressed up as a mechanism, or it removes a safety mechanism (a step budget, a
   summarizer, a guard, a critic) with nothing in its place, without saying why that is safe.
3. GRADER GAMING: it tries to satisfy the checks' form rather than do the work better.
4. UNBOUNDED WORK: a loop or retry with no way to finish.
Otherwise ACCEPT. Reply with strict JSON only: {"verdict": "accept" or "reject", "reasons": ["..."]}"""


def review(edits, cases, mode: str, use_model: bool) -> Dict[str, Any]:
    hard = precheck(edits, cases)
    if hard:
        return {"verdict": "reject", "reasons": hard, "by": "precheck"}
    if mode != "live" or not use_model:
        return {"verdict": "accept", "reasons": [], "by": "precheck"}
    tasks = "\n".join(f"- {AL._one_line(c['task'], 160)}" for c in cases[:12])
    out = ask(CRITIC_SYSTEM, f"The cases the design is scored on (for spotting leakage):\n{tasks}\n\n"
                             f"The proposed edits:\n{json.dumps(edits, indent=1)[:6000]}", 1200)
    found = _json_in(out["text"]) or {}
    verdict = str(found.get("verdict", "accept")).lower()
    return {"verdict": "reject" if verdict.startswith("rej") else "accept",
            "reasons": [str(r)[:300] for r in found.get("reasons") or []][:5], "by": "model",
            "tokens": out["tokens"]}


# --------------------------------------------------------------------------
# the proposer (rrsi/propose.py)
# --------------------------------------------------------------------------

PROPOSER_SYSTEM = """You improve an AI agent's design (its harness) by proposing edits. The model inside is
frozen: only the harness can change. You see the design, how it did on its evaluation cases (the worst ones
in detail), and every edit tried so far with its measured effect. Propose ONE candidate: a small set of
independent edits, each with the component it changes and the hypothesis it tests.

Rules:
- At most {budget} edit(s) in this candidate. Fewer is fine; one is often best.
- Every edit must be GENERAL: it must still make sense, and still help, on unfamiliar tasks of the same kind.
  Never quote a case's wording, an expected answer, a name or a number taken from the cases. Edits that do
  are rejected before they are run.
- Do not repeat a hypothesis the history shows was already measured and rejected, unless you say what is
  different this time.
- Prefer edits whose effect can be measured on the cases you were shown.
- Components: prompt (instruction text), config (numbers: steps, temperature, tokens, time), context_mgmt
  (summarizer), control_flow (critic, planner, guards), client_tool (web search, code runner), memory
  (long-term notes), subagent.

Operations, applied in order:
  {{"op": "set", "block": "<id>", "param": "<setting>", "value": <new value>}}
  {{"op": "remove", "block": "<id>"}}
  {{"op": "add", "type": "summarizer", "attach_to": "<llm id>", "params": {{...}}}}
  {{"op": "add", "type": "web_search" or "code_exec", "attach_to": "<router id>", "params": {{...}}}}
  {{"op": "add", "type": "reflector" or "planner" or "long_mem" or "guard_out", "between": ["<from id>", "<to id>"], "params": {{...}}}}
A set on a system prompt's "text" replaces the whole text: write the complete new prompt.

Reply with strict JSON only:
{{"edits": [{{"id": "E1", "component": "<component>", "hypothesis": "<what this fixes and why>",
  "predicted_affected": [<case numbers it should help>], "ops": [<operations>]}}]}}"""


def _json_in(text: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r"```(?:json)?", "", text or "")
    start = text.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    found = json.loads(text[start:i + 1])
                    if isinstance(found, dict):
                        return found
                except ValueError:
                    pass
                break
        start = text.find("{", start + 1)
    return None


def ask(system: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
    """A plain call to the model that proposes and reviews: the judge from Settings, else the default."""
    d = P.load()["defaults"]
    pid = d.get("judge_provider") or d["provider"]
    model = (d.get("judge_model") if d.get("judge_provider") else "") or d["model"]
    local = (P.CATALOG.get(pid) or {}).get("local")
    timeout = max(float(d.get("timeout") or 120), AL.LOCAL_TIMEOUT if local else 300)
    tokens = max(max_tokens, int(d.get("max_tokens") or 0))
    return P.complete(pid, model, system, prompt, max_tokens=tokens, temperature=0.4, timeout=timeout)


def evidence(ev, cases, n_fail: int = 6, n_win: int = 2) -> List[Dict[str, Any]]:
    """The worst trial of the lowest-scoring cases, and the best of the highest (habits not to break)."""
    rows = sorted(ev["per_case"], key=lambda r: sum(r["rewards"]) / max(1, len(r["rewards"])))
    fails = rows[:n_fail]
    wins = [r for r in rows[-n_win:] if r not in fails]
    out = []
    for r, kind in [(r, "weak") for r in fails] + [(r, "strong") for r in wins]:
        i = r["rewards"].index(min(r["rewards"]) if kind == "weak" else max(r["rewards"]))
        out.append({"case": r["case"] + 1, "kind": kind, "mean_score": round(sum(r["rewards"]) / len(r["rewards"]), 3),
                    "task": AL._one_line(cases[r["case"]]["task"], 500), "answer": AL._one_line(r["answers"][i], 600),
                    "failed_checks": r["failed"][i][:6], "error": r["errors"][i]})
    return out


def propose(evo, t: int, variant: str, budget: int, explore, prunes, reserved: bool, siblings: List[str],
            objections: Optional[List[str]] = None) -> Dict[str, Any]:
    """One candidate: {"edits": [...]} from the model, or from the scripted proposer in a rehearsal."""
    inc = evo["incumbent"]
    if evo["mode"] != "live":
        return {"edits": scripted_edits(inc["graph"], t, variant, budget, reserved, explore["untried"]), "tokens": 0}
    history = [{k: r.get(k) for k in ("t", "variant", "component", "hypothesis", "delta_S", "delta_C", "outcome",
                                      "reason") if r.get(k) is not None} for r in evo["history"][-40:]]
    parts = [f"Round {t + 1} of {evo['cfg']['T']}. The design now scores S = {inc['eval']['S']:.3f} (fraction of checks "
             f"passed, mean over {len(evo['cases'])} cases × {inc['eval']['k']} trials), costing about "
             f"{inc['eval']['C'] or 0:.0f} tokens per trial. Best so far S* = {evo['S_star']:.3f}; noise band "
             f"delta = {evo['delta']:.3f}: a gain smaller than that is indistinguishable from luck.",
             "The design:\n" + json.dumps(describe_design(inc["graph"]), indent=1),
             "Its weakest and strongest cases on the last evaluation:\n" + json.dumps(evidence(inc["eval"], evo["cases"]), indent=1),
             "Edit history (most recent last):\n" + (json.dumps(history, indent=1) if history else "none yet"),
             explore["text"]]
    if prunes:
        parts.append("Components whose recent edits have not helped; removing what they added is a fair "
                     "candidate:\n" + json.dumps(prunes, indent=1))
    if reserved:
        parts.append(f"THIS candidate holds a reserved exploration slot: at least one edit must change one of "
                     f"{explore['untried']}.")
    if siblings:
        parts.append("Another candidate this round already tests: " + "; ".join(siblings) + ". Test something else.")
    if objections:
        parts.append("Your last proposal was rejected before it ran, for these reasons. Fix them:\n- "
                     + "\n- ".join(objections))
    out = ask(PROPOSER_SYSTEM.format(budget=budget), "\n\n".join(parts), 4000)
    found = _json_in(out["text"]) or {}
    edits = [e for e in (found.get("edits") or []) if isinstance(e, dict)]
    return {"edits": edits, "tokens": out["tokens"], "raw": out["text"][:1500] if not edits else ""}


GENERAL_CHECK = ("Before you give your final answer, check it against every requirement in the task, and fix "
                 "anything that does not meet one.")


def scripted_edits(graph, t: int, variant: str, budget: int, reserved: bool, untried: List[str]) -> List[Dict[str, Any]]:
    """The rehearsal's proposer: general edits chosen in turn, so the whole loop can run with no model."""
    nodes = graph["nodes"]
    core = next((n for n in nodes if n["type"] == "llm"), None)
    router = next((n for n in nodes if n["type"] == "router"), None)
    prompt = next((n for n in nodes if n["type"] == "system_prompt"), None)
    loop = next((n for n in nodes if n["type"] == "loop"), None)
    has = {n["type"] for n in nodes}
    menu = []
    if prompt and GENERAL_CHECK not in AL._params(prompt)["text"]:
        menu.append({"component": "prompt", "hypothesis": "answers miss stated requirements; ask for a final self-check",
                     "ops": [{"op": "set", "block": prompt["id"], "param": "text",
                              "value": AL._params(prompt)["text"].rstrip() + "\n" + GENERAL_CHECK}]})
    if loop:
        steps = int(AL._params(loop)["max_steps"])
        menu.append({"component": "config", "hypothesis": "runs end before finishing; allow more steps",
                     "ops": [{"op": "set", "block": loop["id"], "param": "max_steps", "value": steps + max(2, steps // 2)}]})
    if core and float(AL._params(core)["temperature"]) > 0:
        menu.append({"component": "config", "hypothesis": "answers vary run to run; sample greedily",
                     "ops": [{"op": "set", "block": core["id"], "param": "temperature", "value": 0}]})
    if core and "summarizer" not in has:
        menu.append({"component": "context_mgmt", "hypothesis": "long runs lose the thread; fold old history",
                     "ops": [{"op": "add", "type": "summarizer", "attach_to": core["id"]}]})
    if router and "code_exec" not in has:
        menu.append({"component": "client_tool", "hypothesis": "arithmetic slips; give the agent a code runner",
                     "ops": [{"op": "add", "type": "code_exec", "attach_to": router["id"]}]})
    if not menu:
        return []
    if reserved and untried:
        menu = [m for m in menu if m["component"] in untried] or menu
    start = (t * 2 + (ord(variant) - 65)) % len(menu)
    picked = [menu[(start + i) % len(menu)] for i in range(min(budget, len(menu)))]
    return [{"id": f"E{i + 1}", "predicted_affected": [], **m} for i, m in enumerate(picked)]


# --------------------------------------------------------------------------
# a run of the whole loop, in the background, saved as it goes
# --------------------------------------------------------------------------

_LIVE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def _evo_dir() -> Path:
    path = AL._dir() / "evolutions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(evo) -> None:
    keep = {k: v for k, v in evo.items() if not k.startswith("_")}
    (_evo_dir() / f"{evo['id']}.json").write_text(json.dumps(keep, default=str))


def load(evo_id: str) -> Dict[str, Any]:
    with _LOCK:
        if evo_id in _LIVE:
            return json.loads(json.dumps({k: v for k, v in _LIVE[evo_id].items() if not k.startswith("_")}, default=str))
    path = _evo_dir() / f"{AL._slug(evo_id)}.json"
    if not path.exists():
        raise KeyError(evo_id)
    evo = json.loads(path.read_text())
    if evo.get("status") in ("running", "stopping"):
        evo["status"], evo["error"] = "interrupted", "The server stopped while this was running."
    return evo


def listing() -> List[Dict[str, Any]]:
    out = []
    for path in sorted(_evo_dir().glob("*.json"), key=lambda p: -p.stat().st_mtime)[:30]:
        try:
            e = load(path.stem)
        except (KeyError, ValueError):
            continue
        out.append({"id": e["id"], "design": e["design"], "status": e["status"], "mode": e["mode"],
                    "started": e["started"], "S0": (e.get("base") or {}).get("S"),
                    "S": ((e.get("incumbent") or {}).get("eval") or {}).get("S")})
    return out


def plan(graph, cases_text: str, heldout_text: str = "", cfg=None) -> Dict[str, Any]:
    """What a run will cost, in agent runs, before it starts."""
    cfg = _cfg(cfg)
    cases = AL.parse_cases(cases_text)
    held = AL.parse_cases(heldout_text) if heldout_text.strip() else {"cases": [], "problems": []}
    n, h = len(cases["cases"]), len(held["cases"])
    runs = n * cfg["k"] * (1 + cfg["T"] * cfg["m"]) + 2 * h * cfg["k"]
    return {"cases": n, "heldout": h, "problems": cases["problems"] + held["problems"], "runs_at_most": runs,
            "budgets": [edit_budget(t, cfg["T"], cfg["b_min"], cfg["b_max"]) for t in range(cfg["T"])], "cfg": cfg}


def start(graph, cases_text: str, heldout_text: str = "", mode: str = "rehearsal", approvals: str = "approve",
          cfg=None, background: bool = True) -> Dict[str, Any]:
    cases = AL.parse_cases(cases_text)
    if cases["problems"]:
        raise ValueError(cases["problems"][0])
    held = AL.parse_cases(heldout_text)["cases"] if heldout_text.strip() else []
    if mode not in ("live", "rehearsal"):
        raise ValueError("Choose rehearsal or live.")
    if mode == "live" and AL.missing_keys(graph):
        raise ValueError(AL.missing_keys(graph)[0])
    if [p for p in AL.validate(graph) if p["level"] == "error"]:
        raise ValueError("Fix the design's errors first: the starting design has to run.")
    evo = {"id": "ev" + uuid.uuid4().hex[:8], "design": graph.get("name") or "agent", "mode": mode,
           "approvals": approvals, "cfg": _cfg(cfg), "cases": cases["cases"], "heldout": held,
           "status": "running", "error": None, "started": time.time(), "finished": None, "phase": "starting",
           "t": 0, "base_graph": json.loads(json.dumps(graph)), "base": None, "incumbent": None, "S_star": None,
           "delta": None, "calibration": None, "trajectory": [], "history": [], "rounds": [], "heldout_eval": None,
           "saved_as": None, "trials_done": 0, "trials_planned": None, "tokens_search": 0}
    evo["trials_planned"] = plan(graph, cases_text, heldout_text, evo["cfg"])["runs_at_most"]
    evo["_stop"] = threading.Event()
    with _LOCK:
        _LIVE[evo["id"]] = evo
    _save(evo)
    if background:
        import contextvars                      # the workspace (account) follows the thread
        threading.Thread(target=contextvars.copy_context().run, args=(_main, evo), daemon=True).start()
    else:
        _main(evo)
    return load(evo["id"])


def stop(evo_id: str) -> Dict[str, Any]:
    with _LOCK:
        evo = _LIVE.get(evo_id)
    if not evo:
        raise ValueError("That evolution is not running.")
    evo["_stop"].set()
    evo["status"] = "stopping"
    return load(evo_id)


def _tick(evo):
    evo["trials_done"] += 1
    _save(evo)


def _main(evo) -> None:
    stop_ev = evo["_stop"]
    cfg, mode, appr = evo["cfg"], evo["mode"], evo["approvals"]
    try:
        # H_0 on the evolve set, and the noise band from its own spread
        evo["phase"] = "measuring the starting design"
        base = evaluate(evo["base_graph"], evo["cases"], cfg["k"], mode, appr, stop_ev, lambda: _tick(evo))
        evo["base"] = {k: base[k] for k in ("S", "C", "k")}
        cal = noise_band(base, cfg["delta_z"])
        evo["calibration"] = cal
        evo["delta"] = cfg["delta"] if cfg["delta"] is not None else cal["delta"]
        evo["incumbent"] = {"t": 0, "graph": evo["base_graph"], "eval": base}
        evo["S_star"] = base["S"]
        evo["trajectory"] = [{"t": 0, "S": base["S"], "C": base["C"]}]
        evo["history"].append({"t": 0, "variant": "-", "component": None, "hypothesis": "the starting design",
                               "outcome": "BASELINE", "S": base["S"], "C": base["C"], "accepted": True,
                               "delta_S": None, "delta_C": None})
        _save(evo)
        for t in range(cfg["T"]):
            if stop_ev.is_set():
                break
            evo["t"] = t
            _round(evo, t)
            _save(evo)
        if evo["heldout"] and not stop_ev.is_set():
            evo["phase"] = "checking on the held-out cases"
            b = evaluate(evo["base_graph"], evo["heldout"], cfg["k"], mode, appr, stop_ev, lambda: _tick(evo))
            f = (b if evo["incumbent"]["t"] == 0 else
                 evaluate(evo["incumbent"]["graph"], evo["heldout"], cfg["k"], mode, appr, stop_ev, lambda: _tick(evo)))
            evo["heldout_eval"] = {"base": {"S": b["S"], "C": b["C"]}, "final": {"S": f["S"], "C": f["C"]}}
        if evo["incumbent"]["t"] > 0:
            g = json.loads(json.dumps(evo["incumbent"]["graph"]))
            g["name"] = f"{evo['design']} evolved"
            g["evolved_from"] = {"design": evo["design"], "evolution": evo["id"], "S0": evo["base"]["S"],
                                 "S": evo["incumbent"]["eval"]["S"]}
            evo["saved_as"] = AL.save(g)["name"]
        evo["status"] = "stopped" if stop_ev.is_set() else "done"
        evo["phase"] = "finished"
    except InterruptedError:
        evo["status"], evo["phase"] = "stopped", "stopped"
    except Exception as exc:  # noqa: BLE001
        evo["status"], evo["error"] = "error", f"{type(exc).__name__}: {exc}"
    finally:
        evo["finished"] = time.time()
        _save(evo)
        with _LOCK:
            _LIVE.pop(evo["id"], None)


def _round(evo, t: int) -> None:
    cfg, inc = evo["cfg"], evo["incumbent"]
    budget = edit_budget(t, cfg["T"], cfg["b_min"], cfg["b_max"])
    traj = [x["S"] for x in evo["trajectory"]]
    sigma = stall_flag(traj, t, cfg["w"], evo["delta"])
    explore = exploration(sigma, tried(evo["history"]), cfg["m_draft"])
    prunes = prune_set(evo["history"], t, cfg["n_prune"])
    rnd = {"t": t, "b_t": budget, "sigma": sigma, "untried": explore["untried"],
           "prune": [p["component"] for p in prunes], "candidates": [], "winner": None}
    evo["rounds"].append(rnd)
    # draw and screen m candidates
    siblings: List[str] = []
    for v in range(cfg["m"]):
        if evo["_stop"].is_set():
            return
        variant = "ABCD"[v]
        reserved = bool(sigma and explore["untried"] and v >= cfg["m"] - cfg["m_draft"])
        evo["phase"] = f"round {t + 1}: proposing candidate {variant}"
        _save(evo)
        cand = {"variant": variant, "edits": [], "gate": None, "detail": "", "reserved": reserved}
        objections, edits = None, []
        for attempt in range(cfg["repair_rounds"] + 1):
            got = propose(evo, t, variant, budget, explore, prunes, reserved, siblings, objections)
            evo["tokens_search"] += got.get("tokens") or 0
            edits = got["edits"][:budget]
            if not edits:
                cand.update(gate="no_proposal", detail=got.get("raw") or "The proposer returned no edits.")
                break
            ops = [op for e in edits for op in (e.get("ops") or [])]
            graph, problems, comps = apply_ops(inc["graph"], ops)
            if problems:
                objections = [f"the edits do not apply: {p}" for p in problems[:4]]
                cand.update(gate="smoke_fail", detail="; ".join(problems[:3]))
                continue
            # each edit's component, recovered from what it actually changes
            for e in edits:
                _, _, own = apply_ops(inc["graph"], e.get("ops") or [])
                e["component"] = own[0] if own else (e.get("component") if e.get("component") in K else "prompt")
            verdict = review(edits, evo["cases"], evo["mode"], cfg["critic_model"])
            evo["tokens_search"] += verdict.get("tokens") or 0
            if verdict["verdict"] == "reject":
                objections = verdict["reasons"]
                cand.update(gate="critic_reject", detail="; ".join(verdict["reasons"])[:600])
                continue
            if reserved and not any(e["component"] in explore["untried"] for e in edits):
                objections = [f"this slot is reserved: change one of {explore['untried']}"]
                cand.update(gate="reservation", detail=objections[0])
                continue
            cand.update(edits=edits, gate=None, detail="", graph=graph, components=comps)
            siblings += [str(e.get("hypothesis"))[:120] for e in edits]
            break
        cand["edits"] = cand["edits"] or edits       # a candidate that never ran keeps what it tried
        rnd["candidates"].append(cand)
    # evaluate the screened set
    for cand in rnd["candidates"]:
        if cand["gate"] is None:
            if evo["_stop"].is_set():
                return
            evo["phase"] = f"round {t + 1}: running candidate {cand['variant']} on the cases"
            _save(evo)
            cand["eval"] = evaluate(cand["graph"], evo["cases"], cfg["k"], evo["mode"], evo["approvals"],
                                    evo["_stop"], lambda: _tick(evo))
    # Algorithm 2
    counts = incumbent_counts(evo["history"])
    best = None
    for cand in rnd["candidates"]:
        d = judge(cand.get("eval"), inc["eval"], evo["S_star"], evo["delta"], cfg, cand.get("components") or [], counts)
        cand["decision"] = d
        if d["admissible"] and (best is None or d["S"] > best["decision"]["S"]):
            best = cand
    for cand in rnd["candidates"]:
        d = cand["decision"]
        outcome = (cand["gate"] or "not_run") if cand.get("eval") is None else (
            "ACCEPTED" if cand is best else "LOST" if d["admissible"] else "REJECTED")
        cand["outcome"] = outcome
        for e in cand["edits"] or [{"component": None, "hypothesis": cand.get("detail")}]:
            evo["history"].append({"t": t, "variant": cand["variant"], "component": e.get("component"),
                                   "hypothesis": str(e.get("hypothesis") or "")[:300], "ops": e.get("ops"),
                                   "delta_S": None if d.get("delta_S") is None else round(d["delta_S"], 4),
                                   "delta_C": None if d.get("delta_C") is None else round(d["delta_C"], 4),
                                   "S": d.get("S"), "C": d.get("C"), "accepted": cand is best, "outcome": outcome,
                                   "reason": d.get("reason") if cand.get("eval") is not None else cand.get("detail"),
                                   "bundle": len(cand["edits"] or [])})
    if best is not None:
        evo["incumbent"] = {"t": t + 1, "graph": best["graph"], "eval": best["eval"], "variant": best["variant"]}
        evo["S_star"] = max(evo["S_star"], best["eval"]["S"])
        rnd["winner"] = best["variant"]
    evo["trajectory"].append({"t": t + 1, "S": evo["incumbent"]["eval"]["S"], "C": evo["incumbent"]["eval"]["C"]})
    # keep the record small: a candidate's graph is only needed while it might win
    for cand in rnd["candidates"]:
        cand.pop("graph", None)
        if cand.get("eval"):
            cand["eval"] = {k: cand["eval"][k] for k in ("S", "C", "k")}
