"""A workbench for attacking a problem that will not fall in one sitting.

Let me be straight about what this is, because the alternative is theatre.

This will not solve the Riemann hypothesis. Nothing here is close to that, and
an application that implied otherwise would be lying to its user. What defeats
people on hard problems is rarely a missing idea — it is losing track of which
of their forty steps were checked, which were assumed, which quietly smuggled in
an extra hypothesis, and which were tried last month and failed for a reason
nobody wrote down.

That is a bookkeeping problem, and bookkeeping is something a program can do
perfectly. So this holds:

    the problem        stated once, with its assumptions fixed
    claims             atomic, each with a status nobody can set by opinion
    evidence           symbolic checks that ran, or an honest "assumed"
    dependencies       a graph, checked for cycles and for resting on
                       anything rejected
    failures           what was tried and why it did not work, kept

The critics are deterministic. They do not have opinions about whether a claim
is interesting; they check whether it has evidence, whether its dependencies
hold, whether it introduces an assumption the problem never granted, and
whether the algebra it asserts is actually true. A critic that cannot be
flattered is worth more than one that can.

The design of this follows MARE, which is a fuller engine for the same job —
branches, adversarial review, literature provenance, a Lean formalization gate.
Its central rule is the one that matters here too: **agreement is not
verification**.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

STATUSES = ("proposed", "verified", "rejected", "scope_blocked", "assumed")


def home() -> Path:
    import auth

    return auth.sub("problems")


def _path(problem_id: str) -> Path:
    return home() / f"{problem_id}.json"


def create(title: str, statement: str, scope: List[str]) -> Dict[str, Any]:
    """A problem, with the assumptions it is allowed to use written down first.

    Fixing the scope before any work starts is the whole point of it. A proof
    that quietly assumes smoothness is not a proof of the theorem that did not
    assume smoothness, and the only reliable way to catch that is to have
    written down what was granted before anyone was invested in the answer.
    """
    problem = {
        "id": uuid.uuid4().hex[:12],
        "title": title.strip() or "Untitled problem",
        "statement": statement.strip(),
        "scope": [s.strip() for s in scope if s.strip()],
        "created": time.time(),
        "claims": [],
        "failures": [],
    }
    save(problem)
    return problem


def save(problem: Dict[str, Any]) -> None:
    folder = home()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{problem['id']}.json").write_text(json.dumps(problem, indent=1))


def load(problem_id: str) -> Optional[Dict[str, Any]]:
    path = _path(problem_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def listing() -> List[Dict[str, Any]]:
    out = []
    folder = home()
    if not folder.exists():
        return out
    for path in sorted(folder.glob("*.json"), key=lambda p: -p.stat().st_mtime):
        try:
            blob = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        claims = blob.get("claims", [])
        out.append({
            "id": blob["id"], "title": blob["title"],
            "claims": len(claims),
            "verified": sum(1 for c in claims if c["status"] == "verified"),
            "created": blob.get("created", 0),
        })
    return out


def add_claim(problem: Dict[str, Any], text: str, depends: List[str],
              assumes: List[str], identity: str = "") -> Dict[str, Any]:
    """A claim starts proposed. Only evidence moves it."""
    claim = {
        "id": "C" + uuid.uuid4().hex[:8],
        "text": text.strip(),
        "depends": [d for d in depends if d],
        "assumes": [a.strip() for a in assumes if a.strip()],
        "identity": identity.strip(),
        "status": "proposed",
        "evidence": [],
        "objections": [],
    }
    problem["claims"].append(claim)
    return claim


# --------------------------------------------------------------------------
# the one kind of evidence that is not an opinion
# --------------------------------------------------------------------------

def check_identity(identity: str, timeout_note: str = "") -> Dict[str, Any]:
    """Decide whether a stated identity is actually true, with sympy.

    `lhs = rhs` is checked by simplifying the difference. This is the only
    thing in the whole workbench that can move a claim to verified on its own,
    because it is the only thing here that cannot be argued with.
    """
    if "=" not in identity:
        return {"ok": False, "kind": "symbolic",
                "detail": "Not an identity. Write it as lhs = rhs."}
    left, _, right = identity.partition("=")
    try:
        import sympy
        from sympy.parsing.sympy_parser import (parse_expr,
                                                standard_transformations,
                                                implicit_multiplication_application)

        rules = standard_transformations + (implicit_multiplication_application,)
        lhs = parse_expr(left.strip(), transformations=rules)
        rhs = parse_expr(right.strip(), transformations=rules)
        difference = sympy.simplify(lhs - rhs)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "kind": "symbolic",
                "detail": f"Could not read it: {type(exc).__name__}: {exc}"}

    if difference == 0:
        return {"ok": True, "kind": "symbolic",
                "detail": f"{left.strip()} − {right.strip()} simplifies to 0."}
    return {"ok": False, "kind": "symbolic",
            "detail": f"The difference simplifies to {difference}, not 0. "
                      f"Either the identity is wrong or it needs conditions "
                      f"that are not written down."}


# --------------------------------------------------------------------------
# critics: deterministic, and unable to be flattered
# --------------------------------------------------------------------------

def critique(problem: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Everything wrong with the argument as it stands.

    None of these is a judgement about whether the mathematics is interesting.
    Each is a fact about the structure that can be checked by looking.
    """
    claims = {c["id"]: c for c in problem.get("claims", [])}
    scope = {s.lower() for s in problem.get("scope", [])}
    found: List[Dict[str, Any]] = []

    def say(claim_id, kind, text, severity="warn"):
        found.append({"claim": claim_id, "kind": kind, "text": text,
                      "severity": severity})

    for claim in problem.get("claims", []):
        # a claim standing on nothing
        if claim["status"] == "verified" and not claim["evidence"]:
            say(claim["id"], "unevidenced",
                "Marked verified with no evidence recorded. Agreement is not "
                "verification.", "fatal")

        # a claim standing on something broken
        for need in claim["depends"]:
            if need not in claims:
                say(claim["id"], "missing",
                    f"Depends on {need}, which does not exist.", "fatal")
                continue
            parent = claims[need]
            if parent["status"] == "rejected":
                say(claim["id"], "rests_on_rejected",
                    f"Depends on {need}, which was rejected.", "fatal")
            elif claim["status"] == "verified" and parent["status"] != "verified":
                say(claim["id"], "ahead_of_itself",
                    f"Verified, but {need} beneath it is {parent['status']}. "
                    f"A conclusion cannot be more certain than what it rests "
                    f"on.", "fatal")

        # an assumption the problem never granted
        for assumption in claim["assumes"]:
            if assumption.lower() not in scope:
                say(claim["id"], "out_of_scope",
                    f"Assumes “{assumption}”, which the problem statement does "
                    f"not grant. The result would be about a different "
                    f"problem.", "fatal")

        # an identity that was written but never run
        if claim["identity"] and not any(
                e.get("kind") == "symbolic" for e in claim["evidence"]):
            say(claim["id"], "unchecked",
                "States an identity that has not been checked. It can be, in "
                "one press.", "warn")

    # a cycle is not an argument
    for loop in _cycles(claims):
        say(loop[0], "circular",
            "Circular dependency: " + " → ".join(loop)
            + ". Each step is waiting for the next.", "fatal")

    return found


def _cycles(claims: Dict[str, Any]) -> List[List[str]]:
    found: List[List[str]] = []
    seen: set = set()
    stack: List[str] = []
    on_stack: set = set()

    def walk(node: str) -> None:
        seen.add(node)
        stack.append(node)
        on_stack.add(node)
        for nxt in claims.get(node, {}).get("depends", []):
            if nxt not in claims:
                continue
            if nxt in on_stack:
                loop = stack[stack.index(nxt):] + [nxt]
                if loop not in found:
                    found.append(loop)
            elif nxt not in seen:
                walk(nxt)
        stack.pop()
        on_stack.discard(node)

    for name in sorted(claims):
        if name not in seen:
            walk(name)
    return found[:8]


def standing(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Where the argument actually is, counted rather than felt."""
    claims = problem.get("claims", [])
    by_status = {s: sum(1 for c in claims if c["status"] == s) for s in STATUSES}
    objections = critique(problem)
    fatal = [o for o in objections if o["severity"] == "fatal"]

    roots = [c for c in claims if not any(
        c["id"] in other["depends"] for other in claims)]
    closed = [c for c in roots
              if c["status"] == "verified" and _closed(c, claims)]


    # An argument with nothing in it has no fatal objections, which is not the
    # same as holding together — it is the difference between a clean bill of
    # health and an empty file. Saying "holds together" over an empty problem
    # is the exact flattery this whole thing exists to refuse.
    if not claims:
        verdict = "empty"
    elif fatal:
        verdict = "broken"
    elif not closed:
        verdict = "open"
    else:
        verdict = "closed"

    return {
        "counts": by_status,
        "objections": objections,
        "verdict": verdict,
        "sound": not fatal and bool(claims),
        "roots": [c["id"] for c in roots],
        "closed_roots": [c["id"] for c in closed],
        "assumed": sorted({a for c in claims for a in c["assumes"]}),
        "unchecked_identities": [c["id"] for c in claims
                                 if c["identity"] and not any(
                                     e.get("kind") == "symbolic"
                                     for e in c["evidence"])],
    }


def _closed(claim: Dict[str, Any], claims: List[Dict[str, Any]]) -> bool:
    """Verified, and everything under it verified, all the way down."""
    by_id = {c["id"]: c for c in claims}
    seen: set = set()

    def walk(here: str) -> bool:
        if here in seen:
            return True
        seen.add(here)
        entry = by_id.get(here)
        if entry is None or entry["status"] != "verified":
            return False
        return all(walk(need) for need in entry["depends"])

    return walk(claim["id"])
