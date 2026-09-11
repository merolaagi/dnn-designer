"""How people actually attack a problem that does not fall, as checkable moves.

The method is not a secret. Pólya wrote most of it down, Hadamard described the
psychology, and working mathematicians — Tao and Gowers among them — have
written at length about what they really do when stuck. It is the same short
list every time, and almost nobody follows it, because under pressure everyone
skips to the part they enjoy: attacking the general case head-on.

So this does not recite the list. It looks at the problem as it actually stands
and says which moves have been made and which have been skipped. "You have five
claims and none has been tested against a single example" is advice; "always
test examples" is a poster.

Each move says what it is for, what doing it looks like here, and how this
program can tell whether it has been done. That last part is the one that
matters: a step nobody can check is a step everybody claims.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

#: Every move carries a test over the problem's own state. `done` is a fact,
#: not a judgement, and the wording of each `next` is what to do *here*.
MOVES: List[Dict[str, Any]] = [
    {
        "id": "state",
        "name": "State it so precisely that you could be wrong",
        "why": "Most failed attempts are attempts on a different problem. Write "
               "the statement once, with every quantifier and hypothesis, "
               "before you are invested in an answer — afterwards you will "
               "read the hypotheses you need into it without noticing.",
        "here": "The statement and the granted assumptions at the top of this "
                "page. If a hypothesis is not on that list, a claim using it "
                "gets flagged, which is the point rather than a nuisance.",
    },
    {
        "id": "examples",
        "name": "Get your hands dirty before you theorise",
        "why": "Nobody solves a problem they have no feel for. Compute the "
               "smallest cases by hand. Look for the pattern. This is the "
               "step everyone skips and the one most likely to hand you the "
               "answer, or to hand you a counterexample and save you a month.",
        "here": "Add claims about small cases — n = 1, n = 2, the degenerate "
                "case — and put numbers into them. Each one is a claim like "
                "any other, and a false one is progress.",
    },
    {
        "id": "disprove",
        "name": "Try to break it first",
        "why": "An honest attempt to disprove a statement is the fastest way to "
               "understand it, and the cheapest way to discover it is false. "
               "If it survives your best attack you will know where its "
               "strength is, which is exactly where a proof will have to go.",
        "here": "Hunt a counterexample on every claim with an identity. A "
                "claim that survives is not proved — but the point where it "
                "nearly fails is usually where the difficulty lives.",
    },
    {
        "id": "simplest_hard",
        "name": "Find the simplest case you cannot do",
        "why": "The general problem is rarely the right problem. Strip it until "
               "it becomes easy, then add back one piece at a time. The first "
               "case that defeats you is the real problem, and it is usually "
               "much smaller than the one you started with.",
        "here": "State that case as its own claim. If you can do n = 2 and not "
                "n = 3, the whole difficulty is in that step and everything "
                "else is decoration.",
    },
    {
        "id": "weaken",
        "name": "Prove something weaker, on purpose",
        "why": "A special case, a weaker bound, an extra hypothesis you intend "
               "to remove later. This is not giving up; it is how nearly every "
               "hard theorem was actually reached, and the weaker result often "
               "shows which hypothesis was load-bearing.",
        "here": "Add the hypothesis to the problem's granted list and say so. "
                "It stops being smuggling the moment it is written down, and "
                "removing it later becomes a stated goal rather than a "
                "forgotten debt.",
    },
    {
        "id": "obstruction",
        "name": "Find out why other attempts failed",
        "why": "Hard problems have known obstructions — reasons a whole family "
               "of approaches cannot work. Not knowing them means rediscovering "
               "them slowly. This is the single highest-value hour available to "
               "somebody starting.",
        "here": "Search the literature from this page, including the run that "
                "looks for counterexamples and no-go results. Record what you "
                "find under what did not work.",
    },
    {
        "id": "failures",
        "name": "Write down what did not work, and why",
        "why": "An approach abandoned without a written reason will be tried "
               "again in six weeks, by you. The reason matters more than the "
               "approach: “energy methods fail because the L2 norm is "
               "conserved” is reusable knowledge; “tried energy methods” is "
               "not.",
        "here": "The dead ends section. One line each, with the reason.",
    },
    {
        "id": "audit",
        "name": "Check what you assumed while you were not looking",
        "why": "The commonest way a long argument fails is that step 14 quietly "
               "needs something step 3 never granted. You cannot catch this by "
               "rereading, because you will read your intention rather than "
               "your words.",
        "here": "The critics do this continuously. Every objection about scope "
                "is a hypothesis you used without being given it.",
    },
]


def where(problem: Dict[str, Any], standing: Dict[str, Any]) -> Dict[str, Any]:
    """Which moves have been made on this problem, and what to do next.

    Every test below reads the problem's own contents. None of them is an
    opinion about the mathematics, and none can be satisfied by agreeing with
    the advice.
    """
    claims = problem.get("claims", [])
    scope = problem.get("scope", [])
    failures = problem.get("failures", [])
    evidence = [e for c in claims for e in c.get("evidence", [])]

    numeric = [e for e in evidence if e.get("kind") == "numeric"]
    symbolic = [e for e in evidence if e.get("kind") == "symbolic"]
    with_identity = [c for c in claims if c.get("identity")]
    probed = {c["id"] for c in claims
              if any(e.get("kind") == "numeric" for e in c.get("evidence", []))}

    tests: Dict[str, Callable[[], Dict[str, Any]]] = {
        "state": lambda: {
            "done": bool((problem.get("statement") or "").strip()) and bool(scope),
            "note": (f"{len(scope)} assumption{'s' if len(scope) != 1 else ''} "
                     f"granted." if scope else
                     "Nothing is granted yet, so every assumption any claim "
                     "makes will be flagged. That is usually not what you "
                     "want: the hypotheses of the theorem belong on the list."),
        },
        "examples": lambda: {
            "done": len(claims) >= 2,
            "note": (f"{len(claims)} claims." if claims
                     else "No claims yet. Start with the smallest case you can "
                          "compute by hand."),
        },
        "disprove": lambda: {
            "done": bool(with_identity) and len(probed) >= len(with_identity),
            "note": (f"{len(probed)} of {len(with_identity)} identities have "
                     f"been attacked numerically."
                     if with_identity else
                     "No claim states an identity yet, so there is nothing to "
                     "attack. An identity is the part a machine can argue "
                     "with."),
        },
        "simplest_hard": lambda: {
            "done": any(c["status"] == "rejected" for c in claims)
                    or any(c["status"] == "proposed" for c in claims),
            "note": ("Something is still open or rejected, which is where the "
                     "difficulty is."
                     if any(c["status"] in ("proposed", "rejected") for c in claims)
                     else "Everything is settled. If the problem is not solved, "
                          "the hard case has not been stated yet."),
        },
        "weaken": lambda: {
            "done": bool(scope),
            "note": ("Extra hypotheses are written down rather than smuggled."
                     if scope else
                     "Nothing granted. If you are proving a special case, say "
                     "which — it stops being smuggling the moment it is on the "
                     "list."),
        },
        "obstruction": lambda: {
            "done": bool(problem.get("searched")),
            "note": ("The literature has been searched from here."
                     if problem.get("searched") else
                     "The literature has not been searched from here. For a "
                     "known problem this is the highest-value hour available."),
        },
        "failures": lambda: {
            "done": bool(failures),
            "note": (f"{len(failures)} dead end{'s' if len(failures) != 1 else ''} "
                     f"recorded." if failures else
                     "Nothing recorded. The first approach that fails is worth "
                     "a line."),
        },
        "audit": lambda: {
            "done": not [o for o in standing.get("objections", [])
                         if o["kind"] == "out_of_scope"],
            "note": (f"{len([o for o in standing.get('objections', []) if o['kind'] == 'out_of_scope'])} "
                     f"claims use something the problem never granted."
                     if [o for o in standing.get("objections", [])
                         if o["kind"] == "out_of_scope"]
                     else "No claim is using a hypothesis it was not given."),
        },
    }

    out = []
    for move in MOVES:
        verdict = tests[move["id"]]()
        out.append({**move, **verdict})

    skipped = [m for m in out if not m["done"]]
    return {
        "moves": out,
        "done": sum(1 for m in out if m["done"]),
        "total": len(out),
        # the next thing to do is the earliest skipped move, because the order
        # is not arbitrary — examples before theory, disproof before proof
        "next": skipped[0] if skipped else None,
        "counts": {"claims": len(claims), "symbolic": len(symbolic),
                   "numeric": len(numeric), "failures": len(failures)},
    }
