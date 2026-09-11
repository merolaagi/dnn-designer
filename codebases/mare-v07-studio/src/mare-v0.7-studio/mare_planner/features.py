"""Versioned, pre-decision features; no targets, IDs or future events enter tensors."""

import hashlib
import math
import re

from mare.learned_policy import ResearchPolicy

VERSION = 1
NODE_DIM = 48
CONTEXT_DIM = 35
RELATIONS = ("parent", "contains", "asks", "requires", "supports", "contradicts", "failure")
EDGE_TYPES = len(RELATIONS) * 2


def text_features(text):
    # Deliberately local lexical hashing, NOT a pretrained semantic encoder.
    out = [0.0] * 32
    for token in re.findall(r"\w+", text.lower())[:4096]:
        h = hashlib.sha256(token.encode()).digest()
        out[int.from_bytes(h[:2], "big") % 32] += 1 if h[2] & 1 else -1
    norm = math.sqrt(sum(x * x for x in out)) or 1
    return [v / norm for v in out]


def problem_key(problem):
    # Exclude randomly generated IDs; retain the actual problem specification.
    import json

    body = problem.model_dump(exclude={"id"})
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def graph_record(state, max_nodes=2048):
    objects = [*state.branches, *state.questions, *state.claims, *state.evidence, *state.failures]
    if len(objects) > max_nodes:
        raise ValueError("Graph exceeds 2048 nodes; planner falls back without silently truncating evidence")
    ids = [o.id for o in objects]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate graph IDs")
    index = {v: i for i, v in enumerate(ids)}
    nodes, candidates, baseline, heuristic = [], [], [], []
    policy = ResearchPolicy()
    for obj in objects:
        kind = (
            0
            if obj in state.branches
            else 1
            if obj in state.questions
            else 2
            if obj in state.claims
            else 3
            if obj in state.evidence
            else 4
        )
        numeric = [0.0] * 11
        if kind == 0:
            numeric = policy.branch_features(state, obj)
            text = obj.title + " " + obj.research_question + " " + obj.strategy
            eligible = obj.status.value != "terminated"
            score = obj.heuristic_score
        elif kind == 1:
            numeric = policy.question_features(state, obj) + [0.0]
            text = obj.question + " " + obj.rationale
            eligible = obj.status.value in {"open", "assigned"}
            score = obj.priority
        elif kind == 2:
            numeric[:4] = [
                float(obj.status.value in {"verified", "formalized"}),
                float(obj.status.value in {"rejected", "scope_blocked"}),
                obj.confidence,
                min(len(obj.dependencies) / 10, 1),
            ]
            text = obj.statement
        elif kind == 3:
            numeric[:3] = [
                float(obj.result.value == "supports"),
                float(obj.result.value == "contradicts"),
                float(obj.reproducible),
            ]
            text = obj.details
        else:
            numeric[:2] = [float(bool(obj.reusable_constraint)), float(obj.severity == "fatal")]
            text = obj.failure_reason + " " + (obj.reusable_constraint or "")
        nodes.append([float(i == kind) for i in range(5)] + numeric + text_features(text))
        if kind < 2 and eligible:
            candidates.append(index[obj.id])
            baseline.append(numeric)
            heuristic.append(score)
    edges = set()

    def edge(a, b, relation):
        if a in index and b in index:
            r = RELATIONS.index(relation) * 2
            edges.add((index[a], index[b], r))
            edges.add((index[b], index[a], r + 1))

    for b in state.branches:
        edge(b.parent_id, b.id, "parent")
    for q in state.questions:
        edge(q.branch_id, q.id, "asks")
    for c in state.claims:
        edge(c.branch_id, c.id, "contains")
        for dep in c.dependencies:
            edge(dep, c.id, "requires")
    for d in state.dependency_edges:
        edge(d.parent_claim_id, d.child_claim_id, d.relation)
    for e in state.evidence:
        edge(
            e.id,
            e.claim_id,
            "contradicts"
            if e.result.value == "contradicts"
            else "supports"
            if e.result.value == "supports"
            else "contains",
        )
    for f in state.failures:
        edge(f.id, f.branch_id, "failure")
        edge(f.id, f.failed_claim_id, "failure")
    b = state.budget
    remaining = [
        max(0, 1 - b.model_calls_used / max(1, b.max_model_calls)),
        max(0, 1 - b.tool_calls_used / max(1, b.max_tool_calls)),
        max(0, 1 - state.current_round / max(1, b.max_rounds)),
    ]
    return dict(
        version=VERSION,
        problem_id=problem_key(state.problem),
        run_id=state.run_id,
        round=state.current_round,
        nodes=nodes,
        edges=sorted(edges),
        context=text_features(state.problem.canonical_statement) + remaining,
        node_ids=ids,
        candidates=candidates,
        candidate_ids=[ids[i] for i in candidates],
        baseline=baseline,
        heuristic=heuristic,
    )
