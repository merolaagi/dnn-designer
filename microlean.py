"""A very small proof assistant, for a network to learn to drive.

Not Lean. Lean is a dependently-typed language with a trusted kernel, an
elaborator, universes, inductive types and a tactic framework; none of that is a
neural network and none of it can be made of layers. What *is* learnable is the
part a prover cannot do by itself: choosing which step to take next. That is
what AlphaProof and its predecessors learn, and it is what this file provides a
world for.

So: equational logic over a small algebra. A goal is a pair of terms to be made
equal. A tactic is (rewrite rule, position). The kernel applies a rule and
checks the result — the network proposes, the kernel disposes, and a proof is
only a proof if the kernel says so. Nothing here trusts the model.

    >>> theorem = Theorem.parse("(a * 1) * b", "a * b")
    >>> proof = [(0, 0)]                     # right-identity at the root's left
    >>> check(theorem, proof)
    True
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# terms
# --------------------------------------------------------------------------

# A term is either a variable ("a"), a constant ("1", "0"), or an application
# (op, left, right). Kept as nested tuples so they hash and compare cheaply.
Term = object  # str | Tuple[str, Term, Term]

OPS = ("*", "+")


def is_leaf(t: Term) -> bool:
    return isinstance(t, str)


def show(t: Term) -> str:
    if is_leaf(t):
        return t
    op, left, right = t
    return f"({show(left)} {op} {show(right)})"


def parse(text: str) -> Term:
    """Read a term. Parentheses required around every application."""
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    position = [0]

    def walk() -> Term:
        token = tokens[position[0]]
        position[0] += 1
        if token != "(":
            return token
        left = walk()
        op = tokens[position[0]]
        position[0] += 1
        right = walk()
        assert tokens[position[0]] == ")", f"unbalanced in {text!r}"
        position[0] += 1
        return (op, left, right)

    term = walk()
    if position[0] != len(tokens):
        raise ValueError(f"trailing tokens in {text!r}")
    return term


def size(t: Term) -> int:
    return 1 if is_leaf(t) else 1 + size(t[1]) + size(t[2])


def positions(t: Term) -> List[Tuple[int, ...]]:
    """Every subterm address, root first, as a path of 1s and 2s."""
    found = [()]
    if not is_leaf(t):
        for index, child in ((1, t[1]), (2, t[2])):
            found.extend((index,) + rest for rest in positions(child))
    return found


def at(t: Term, path: Sequence[int]) -> Term:
    for step in path:
        t = t[step]
    return t


def replace(t: Term, path: Sequence[int], new: Term) -> Term:
    if not path:
        return new
    op, left, right = t
    if path[0] == 1:
        return (op, replace(left, path[1:], new), right)
    return (op, left, replace(right, path[1:], new))


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    name: str
    lhs: Term
    rhs: Term

    def __str__(self) -> str:
        return f"{self.name}: {show(self.lhs)} = {show(self.rhs)}"


def _rule(name: str, lhs: str, rhs: str) -> Rule:
    return Rule(name, parse(lhs), parse(rhs))


# A commutative monoid with an absorbing zero and distribution — enough
# structure that proofs need several steps and the wrong first move wastes them.
AXIOMS: List[Rule] = [
    _rule("mul_one",      "(x * 1)", "x"),
    _rule("one_mul",      "(1 * x)", "x"),
    _rule("mul_zero",     "(x * 0)", "0"),
    _rule("zero_mul",     "(0 * x)", "0"),
    _rule("add_zero",     "(x + 0)", "x"),
    _rule("zero_add",     "(0 + x)", "x"),
    _rule("mul_comm",     "(x * y)", "(y * x)"),
    _rule("add_comm",     "(x + y)", "(y + x)"),
    _rule("mul_assoc",    "((x * y) * z)", "(x * (y * z))"),
    _rule("add_assoc",    "((x + y) + z)", "(x + (y + z))"),
    _rule("distrib",      "(x * (y + z))", "((x * y) + (x * z))"),
    _rule("factor",       "((x * y) + (x * z))", "(x * (y + z))"),
]

VARIABLES = {"x", "y", "z"}
ATOMS = ["a", "b", "c", "0", "1"]


def match(pattern: Term, term: Term,
          bound: Optional[Dict[str, Term]] = None) -> Optional[Dict[str, Term]]:
    """One-way matching: variables in the pattern bind to subterms."""
    bound = {} if bound is None else bound
    if is_leaf(pattern):
        if pattern in VARIABLES:
            if pattern in bound:
                return bound if bound[pattern] == term else None
            bound = dict(bound)
            bound[pattern] = term
            return bound
        return bound if pattern == term else None
    if is_leaf(term) or term[0] != pattern[0]:
        return None
    left = match(pattern[1], term[1], bound)
    if left is None:
        return None
    return match(pattern[2], term[2], left)


def instantiate(pattern: Term, bound: Dict[str, Term]) -> Term:
    if is_leaf(pattern):
        return bound.get(pattern, pattern)
    return (pattern[0], instantiate(pattern[1], bound),
            instantiate(pattern[2], bound))


def apply_rule(term: Term, rule: Rule, path: Sequence[int]) -> Optional[Term]:
    """Rewrite at one position, or None if the rule does not fit there."""
    try:
        target = at(term, path)
    except (IndexError, TypeError):
        return None
    bound = match(rule.lhs, target)
    if bound is None:
        return None
    return replace(term, path, instantiate(rule.rhs, bound))


def legal_moves(term: Term, rules: Sequence[Rule] = tuple(AXIOMS),
                limit: int = 400) -> List[Tuple[int, Tuple[int, ...], Term]]:
    """Every (rule, position, result) available from here."""
    moves = []
    for path in positions(term):
        for index, rule in enumerate(rules):
            result = apply_rule(term, rule, path)
            if result is not None and result != term:
                moves.append((index, path, result))
                if len(moves) >= limit:
                    return moves
    return moves


# --------------------------------------------------------------------------
# theorems and proofs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Theorem:
    lhs: Term
    rhs: Term

    @staticmethod
    def parse(left: str, right: str) -> "Theorem":
        return Theorem(parse(left), parse(right))

    def __str__(self) -> str:
        return f"{show(self.lhs)} = {show(self.rhs)}"


def check(theorem: Theorem, proof: Sequence[Tuple[int, Tuple[int, ...]]],
          rules: Sequence[Rule] = tuple(AXIOMS)) -> bool:
    """Replay a proof. The only thing in this project that decides truth.

    A proof is a list of (rule index, position). It holds if applying them in
    order turns the left side into the right side. Anything the model claims is
    worthless until this returns True.
    """
    term = theorem.lhs
    for rule_index, path in proof:
        if not 0 <= rule_index < len(rules):
            return False
        result = apply_rule(term, rules[rule_index], tuple(path))
        if result is None:
            return False
        term = result
    return term == theorem.rhs


# --------------------------------------------------------------------------
# generating things to prove
# --------------------------------------------------------------------------

def random_term(rng: random.Random, depth: int = 2) -> Term:
    if depth <= 0 or rng.random() < 0.3:
        return rng.choice(ATOMS)
    return (rng.choice(OPS), random_term(rng, depth - 1), random_term(rng, depth - 1))


def make_theorem(rng: random.Random, steps: int = 3,
                 rules: Sequence[Rule] = tuple(AXIOMS)):
    """Walk backwards from a term to get a theorem with a proof that exists.

    Generating goals and hoping they are provable wastes most of the effort;
    walking a random path guarantees a proof and hands you the path as a label.
    """
    start = random_term(rng, depth=2)
    term = start
    proof = []
    for _ in range(steps):
        moves = legal_moves(term, rules)
        if not moves:
            break
        rule_index, path, result = rng.choice(moves)
        if size(result) > 14:
            continue
        proof.append((rule_index, tuple(path)))
        term = result
    if not proof:
        return None
    theorem = Theorem(start, term)
    if theorem.lhs == theorem.rhs:
        return None
    return theorem, proof


def corpus(count: int, steps: int = 3, seed: int = 0, exclude=None):
    """Theorems with a known proof, each verified before being handed out.

    `exclude` is a set of (lhs, rhs) already used elsewhere. The reachable space
    here is small enough that a differently-seeded sample overlaps a training
    set by about a third, which would flatter any evaluation that ignored it.
    """
    rng = random.Random(seed)
    seen = set(exclude or ())
    out = []
    guard = 0
    while len(out) < count and guard < count * 60:
        guard += 1
        made = make_theorem(rng, steps=rng.randint(1, steps))
        if not made:
            continue
        theorem, proof = made
        key = (show(theorem.lhs), show(theorem.rhs))
        if key in seen:
            continue
        if not check(theorem, proof):        # never emit an unverified example
            continue
        seen.add(key)
        out.append((theorem, proof))
    return out


def statements(pairs) -> set:
    """The (lhs, rhs) of each theorem, for keeping train and test apart."""
    return {(show(t.lhs), show(t.rhs)) for t, _ in pairs}


# --------------------------------------------------------------------------
# what the network sees
# --------------------------------------------------------------------------

# A proof state is the goal as it stands: the current term and the target. The
# network reads both and picks one of a fixed set of tactics. Keeping the tactic
# space finite — rule x position — means the head is a classifier and every
# prediction it can make is a legal thing to attempt, which the kernel then
# decides on.

MAX_POSITIONS = 7        # paths of depth 0..2, which covers terms of size 15
PATHS: List[Tuple[int, ...]] = [(), (1,), (2,), (1, 1), (1, 2), (2, 1), (2, 2)]

TOKENS = ["<pad>", "<goal>", "<sep>", "(", ")"] + list(OPS) + ATOMS + sorted(VARIABLES)
TOKEN_ID = {token: index for index, token in enumerate(TOKENS)}
VOCAB = len(TOKENS)
CONTEXT = 48

TACTICS = [(rule, path) for rule in range(len(AXIOMS)) for path in PATHS]
TACTIC_ID = {tactic: index for index, tactic in enumerate(TACTICS)}
N_TACTICS = len(TACTICS)


def tokenize(term: Term) -> List[int]:
    if is_leaf(term):
        return [TOKEN_ID[term]]
    op, left, right = term
    return ([TOKEN_ID["("]] + tokenize(left) + [TOKEN_ID[op]]
            + tokenize(right) + [TOKEN_ID[")"]])


def encode_state(current: Term, target: Term) -> List[int]:
    """The goal, as a fixed-length row of token ids."""
    row = ([TOKEN_ID["<goal>"]] + tokenize(current)
           + [TOKEN_ID["<sep>"]] + tokenize(target))
    row = row[:CONTEXT]
    return row + [TOKEN_ID["<pad>"]] * (CONTEXT - len(row))


def training_pairs(count: int, steps: int = 3, seed: int = 0):
    """(state, tactic) for every step of every generated proof.

    The label is the move that actually advanced this proof, so the network is
    learning a policy, not memorising theorems.
    """
    rows, labels = [], []
    for theorem, proof in corpus(count, steps=steps, seed=seed):
        term = theorem.lhs
        for rule_index, path in proof:
            key = (rule_index, tuple(path))
            if key not in TACTIC_ID:
                break                     # deeper than the tactic space reaches
            rows.append(encode_state(term, theorem.rhs))
            labels.append(TACTIC_ID[key])
            term = apply_rule(term, AXIOMS[rule_index], path)
    return rows, labels


def search(theorem: Theorem, score, width: int = 4, depth: int = 6):
    """Best-first proof search guided by a scoring function.

    `score(current, target)` returns a value per tactic. The kernel decides
    whether each proposed move is legal and whether the goal is closed, so a
    confident wrong suggestion costs a node and nothing else.
    """
    import heapq

    start = theorem.lhs
    seen = {show(start)}
    frontier = [(0.0, 0, show(start), start, [])]
    counter = 1
    while frontier:
        cost, _, _, term, proof = heapq.heappop(frontier)
        if term == theorem.rhs:
            return proof
        if len(proof) >= depth:
            continue
        values = score(term, theorem.rhs)
        ranked = sorted(range(len(TACTICS)), key=lambda i: -values[i])[:width]
        for tactic in ranked:
            rule_index, path = TACTICS[tactic]
            result = apply_rule(term, AXIOMS[rule_index], path)
            if result is None or result == term or size(result) > 16:
                continue
            key = show(result)
            if key in seen:
                continue
            seen.add(key)
            heapq.heappush(frontier, (cost - float(values[tactic]), counter, key,
                                      result, proof + [(rule_index, tuple(path))]))
            counter += 1
    return None
