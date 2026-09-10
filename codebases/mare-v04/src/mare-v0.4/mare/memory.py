from __future__ import annotations

from dataclasses import dataclass

from .memory_backends import LexicalSimilarityBackend, SimilarityBackend
from .models import Claim, Failure, ResearchState, ResearchTask


_default_backend = LexicalSimilarityBackend()


def lexical_similarity(a: str, b: str) -> float:
    """Backwards-compatible helper used by tests and callers."""
    return _default_backend.similarity(a, b)


@dataclass(slots=True)
class RetrievedMemory:
    claims: list[Claim]
    failures: list[Failure]


class ResearchMemory:
    """Dependency-aware pluggable retrieval for compact agent context."""

    def __init__(self, similarity_backend: SimilarityBackend | None = None):
        self.similarity_backend = similarity_backend or LexicalSimilarityBackend()

    def retrieve(self, state: ResearchState, task: ResearchTask, top_k: int = 8) -> RetrievedMemory:
        branch = state.branch(task.branch_id)
        query = " ".join([task.title, task.objective, branch.research_question, branch.strategy])

        claim_scores: list[tuple[float, Claim]] = []
        for claim in state.claims:
            if claim.status.value in {"rejected", "scope_blocked"}:
                continue
            score = self.similarity_backend.similarity(query, " ".join([claim.statement, claim.derivation, *claim.tags]))
            if claim.branch_id == task.branch_id:
                score += 0.35
            if any(e.parent_claim_id == claim.id for e in state.dependency_edges):
                score += 0.05
            claim_scores.append((score, claim))

        failure_scores: list[tuple[float, Failure]] = []
        for failure in state.failures:
            text = " ".join(filter(None, [failure.failure_reason, failure.reusable_constraint or "", *failure.tags]))
            score = self.similarity_backend.similarity(query, text)
            if failure.branch_id == task.branch_id:
                score += 0.4
            failure_scores.append((score, failure))

        claims = [c for s, c in sorted(claim_scores, key=lambda x: x[0], reverse=True)[:top_k] if s > 0]
        failures = [f for s, f in sorted(failure_scores, key=lambda x: x[0], reverse=True)[: max(3, top_k // 2)] if s > 0]
        return RetrievedMemory(claims=claims, failures=failures)
