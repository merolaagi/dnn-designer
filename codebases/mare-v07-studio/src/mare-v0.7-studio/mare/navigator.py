from __future__ import annotations

import math

from .learned_policy import ResearchPolicy
from .models import BranchStatus, EvidenceResult, QuestionStatus, ResearchState


class Navigator:
    """UCB-inspired allocator with an optional learned neural research policy.

    The deterministic heuristic always remains present.  When the learned policy
    has accumulated enough outcome-labeled examples, its prediction is blended
    into the branch score with a capped weight.  This gives MARE a safe cold-start
    path and prevents a young policy model from taking over the research loop.
    """

    def __init__(self, policy: ResearchPolicy | None = None) -> None:
        self.policy = policy

    def score(self, state: ResearchState) -> None:
        total_spend = sum(max(1.0, b.compute_spent) for b in state.branches)

        for branch in state.branches:
            claims = [c for c in state.claims if c.branch_id == branch.id]
            open_questions = [
                q for q in state.questions
                if q.branch_id == branch.id and q.status in {QuestionStatus.OPEN, QuestionStatus.ASSIGNED}
            ]
            question_value = max((q.effective_priority for q in open_questions), default=0.0)

            if not claims:
                raw = max(0.0, min(1.0, 0.30 + 0.20 * branch.novelty + 0.20 * question_value))
            else:
                claim_ids = {c.id for c in claims}
                verified = sum(1 for c in claims if c.status.value in {"verified", "formalized"})
                rejected = sum(1 for c in claims if c.status.value in {"rejected", "scope_blocked"})
                supports = sum(
                    1 for e in state.evidence if e.claim_id in claim_ids and e.result == EvidenceResult.SUPPORTS
                )
                contradictions = sum(
                    1 for e in state.evidence if e.claim_id in claim_ids and e.result == EvidenceResult.CONTRADICTS
                )

                progress = (verified + 0.5 * supports) / max(1, len(claims))
                penalty = (rejected + contradictions) / max(1, len(claims))
                exploration = math.sqrt(math.log(total_spend + 1.0) / (branch.compute_spent + 1.0))
                uncertainty_value = max(0.0, min(1.0, branch.uncertainty))

                raw = (
                    0.38 * progress
                    + 0.15 * branch.novelty
                    + 0.14 * exploration
                    + 0.16 * question_value
                    + 0.07 * uncertainty_value
                    - 0.36 * penalty
                )
                raw = max(0.0, min(1.0, raw))
                branch.verified_progress = min(1.0, progress)
                branch.uncertainty = max(0.0, 1.0 - progress)

            branch.heuristic_score = raw
            branch.policy_score = None
            branch.policy_blend = 0.0
            branch.score = raw

            if self.policy and state.policy.enabled:
                predicted, blend = self.policy.predict_branch(state, branch)
                branch.policy_score = predicted
                branch.policy_blend = blend
                if predicted is not None and blend > 0.0:
                    branch.score = max(0.0, min(1.0, (1.0 - blend) * raw + blend * predicted))

    def update_lifecycle(self, state: ResearchState) -> None:
        for branch in state.branches:
            if branch.title == "Synthesis":
                branch.status = BranchStatus.PROMISING if branch.score >= 0.55 else BranchStatus.ACTIVE
                continue

            rejected = sum(
                1 for c in state.claims
                if c.branch_id == branch.id and c.status.value in {"rejected", "scope_blocked"}
            )
            verified = sum(
                1 for c in state.claims
                if c.branch_id == branch.id and c.status.value in {"verified", "formalized"}
            )
            open_high_value = any(
                q.branch_id == branch.id and q.status == QuestionStatus.OPEN and q.effective_priority >= 0.65
                for q in state.questions
            )

            if branch.score < 0.14 and rejected and not open_high_value:
                branch.consecutive_low_score_rounds += 1
            else:
                branch.consecutive_low_score_rounds = 0

            if branch.consecutive_low_score_rounds >= 1 and rejected > verified:
                branch.status = BranchStatus.TERMINATED
            elif branch.score >= 0.60 and verified:
                branch.status = BranchStatus.PROMISING
            elif branch.score < 0.28:
                branch.status = BranchStatus.DEPRIORITIZED
            else:
                branch.status = BranchStatus.ACTIVE

    def ranked(self, state: ResearchState):
        return sorted(state.branches, key=lambda b: b.score, reverse=True)
