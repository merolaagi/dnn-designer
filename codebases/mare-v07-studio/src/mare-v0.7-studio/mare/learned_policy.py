from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .models import (
    Branch,
    BranchStatus,
    ClaimStatus,
    EvidenceResult,
    PolicyExample,
    PolicyExampleKind,
    PolicyNetworkState,
    QuestionStatus,
    ResearchPolicyState,
    ResearchQuestion,
    ResearchState,
)


BRANCH_FEATURE_NAMES = [
    "novelty",
    "uncertainty",
    "log_compute",
    "claim_count",
    "verified_ratio",
    "rejected_ratio",
    "support_ratio",
    "contradiction_ratio",
    "open_question_count",
    "max_question_priority",
    "branch_depth",
]

QUESTION_FEATURE_NAMES = [
    "information_gain",
    "impact",
    "inverse_difficulty",
    "novelty",
    "spawn_branch",
    "parent_branch_score",
    "parent_verified_progress",
    "parent_uncertainty",
    "question_age",
    "branch_failure_pressure",
]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _safe_ratio(num: float, den: float) -> float:
    return 0.0 if den <= 0 else float(num) / float(den)


class TinyMLPRegressor:
    """A deliberately small, dependency-light neural regressor.

    Architecture: input -> ReLU(hidden) -> sigmoid(output).  It is intentionally
    tiny because MARE v0.4 is learning a scheduling policy, not mathematical truth.
    The implementation uses NumPy so a fresh local install does not require a GPU
    framework just to exercise the learned-policy path.
    """

    def __init__(self, input_size: int, hidden_size: int = 12, seed: int = 7) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)
        scale1 = math.sqrt(2.0 / max(1, input_size))
        scale2 = math.sqrt(2.0 / max(1, hidden_size))
        self.w1 = rng.normal(0.0, scale1, size=(input_size, hidden_size)).astype(float)
        self.b1 = np.zeros(hidden_size, dtype=float)
        self.w2 = rng.normal(0.0, scale2, size=(hidden_size, 1)).astype(float)
        self.b2 = np.zeros(1, dtype=float)
        self.loss: float | None = None
        self.epochs_trained = 0

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-x))

    def predict_array(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.w2 + self.b2
        return self._sigmoid(z2).reshape(-1)

    def predict(self, features: Iterable[float]) -> float:
        x = np.asarray(list(features), dtype=float)
        if x.shape != (self.input_size,):
            raise ValueError(f"expected {self.input_size} features, got {x.shape}")
        return _clamp(float(self.predict_array(x)[0]))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 250,
        learning_rate: float = 0.03,
        l2: float = 1e-4,
    ) -> float:
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError("invalid training matrix shape")
        if len(x) != len(y) or len(x) == 0:
            raise ValueError("training examples are empty or misaligned")
        y = y.reshape(-1, 1).astype(float)
        x = x.astype(float)
        n = float(len(x))

        for _ in range(epochs):
            z1 = x @ self.w1 + self.b1
            h1 = np.maximum(z1, 0.0)
            z2 = h1 @ self.w2 + self.b2
            pred = self._sigmoid(z2)

            # Mean-squared error on a bounded utility target.
            d_pred = (2.0 / n) * (pred - y)
            d_z2 = d_pred * pred * (1.0 - pred)
            d_w2 = h1.T @ d_z2 + l2 * self.w2
            d_b2 = d_z2.sum(axis=0)
            d_h1 = d_z2 @ self.w2.T
            d_z1 = d_h1 * (z1 > 0.0)
            d_w1 = x.T @ d_z1 + l2 * self.w1
            d_b1 = d_z1.sum(axis=0)

            self.w2 -= learning_rate * d_w2
            self.b2 -= learning_rate * d_b2
            self.w1 -= learning_rate * d_w1
            self.b1 -= learning_rate * d_b1

        pred = self.predict_array(x).reshape(-1, 1)
        self.loss = float(np.mean((pred - y) ** 2))
        self.epochs_trained += epochs
        return self.loss

    def to_state(self, *, kind: PolicyExampleKind, trained_examples: int) -> PolicyNetworkState:
        return PolicyNetworkState(
            kind=kind,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            trained_examples=trained_examples,
            epochs=self.epochs_trained,
            loss=self.loss,
            weights1=self.w1.tolist(),
            bias1=self.b1.tolist(),
            weights2=self.w2.reshape(-1).tolist(),
            bias2=float(self.b2[0]),
            active=True,
        )

    @classmethod
    def from_state(cls, state: PolicyNetworkState, seed: int = 7) -> "TinyMLPRegressor":
        net = cls(state.input_size, state.hidden_size, seed=seed)
        net.w1 = np.asarray(state.weights1, dtype=float)
        net.b1 = np.asarray(state.bias1, dtype=float)
        net.w2 = np.asarray(state.weights2, dtype=float).reshape(state.hidden_size, 1)
        net.b2 = np.asarray([state.bias2], dtype=float)
        net.loss = state.loss
        net.epochs_trained = state.epochs
        return net


class ResearchPolicy:
    """Hybrid learned/rule-based scheduler policy.

    It records feature snapshots at the beginning of each round and labels them
    from what happened by the end of that round.  Once enough examples exist, a
    tiny neural network predicts branch utility and question success.  MARE blends
    that prediction with the deterministic heuristic rather than surrendering
    control to a cold-start model.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @staticmethod
    def _branch_depth(state: ResearchState, branch: Branch) -> int:
        depth = 0
        parent = branch.parent_id
        seen: set[str] = set()
        while parent and parent not in seen:
            seen.add(parent)
            depth += 1
            match = next((b for b in state.branches if b.id == parent), None)
            parent = match.parent_id if match else None
        return depth

    def branch_features(self, state: ResearchState, branch: Branch) -> list[float]:
        claims = [c for c in state.claims if c.branch_id == branch.id]
        claim_ids = {c.id for c in claims}
        verified = sum(c.status in {ClaimStatus.VERIFIED, ClaimStatus.FORMALIZED} for c in claims)
        rejected = sum(c.status in {ClaimStatus.REJECTED, ClaimStatus.SCOPE_BLOCKED} for c in claims)
        supports = sum(
            e.result == EvidenceResult.SUPPORTS for e in state.evidence if e.claim_id in claim_ids
        )
        contradictions = sum(
            e.result == EvidenceResult.CONTRADICTS for e in state.evidence if e.claim_id in claim_ids
        )
        open_questions = [
            q for q in state.questions
            if q.branch_id == branch.id and q.status in {QuestionStatus.OPEN, QuestionStatus.ASSIGNED}
        ]
        max_q = max((q.priority for q in open_questions), default=0.0)
        depth = self._branch_depth(state, branch)
        return [
            _clamp(branch.novelty),
            _clamp(branch.uncertainty),
            _clamp(math.log1p(branch.compute_spent) / math.log(21.0)),
            _clamp(len(claims) / 8.0),
            _clamp(_safe_ratio(verified, max(1, len(claims)))),
            _clamp(_safe_ratio(rejected, max(1, len(claims)))),
            _clamp(_safe_ratio(supports, max(1, len(claims) * 2))),
            _clamp(_safe_ratio(contradictions, max(1, len(claims) * 2))),
            _clamp(len(open_questions) / 5.0),
            _clamp(max_q),
            _clamp(depth / 5.0),
        ]

    def question_features(self, state: ResearchState, question: ResearchQuestion) -> list[float]:
        branch = next((b for b in state.branches if b.id == question.branch_id), None)
        failures = [f for f in state.failures if branch and f.branch_id == branch.id]
        claims = [c for c in state.claims if branch and c.branch_id == branch.id]
        failure_pressure = _safe_ratio(len(failures), max(1, len(claims)))
        age = max(0, state.current_round - question.created_round)
        return [
            _clamp(question.expected_information_gain),
            _clamp(question.expected_impact),
            _clamp(1.0 - question.difficulty),
            _clamp(question.novelty),
            1.0 if question.spawn_new_branch else 0.0,
            _clamp(branch.score if branch else 0.5),
            _clamp(branch.verified_progress if branch else 0.0),
            _clamp(branch.uncertainty if branch else 1.0),
            _clamp(age / 5.0),
            _clamp(failure_pressure),
        ]

    @staticmethod
    def _branch_metrics(state: ResearchState, branch_id: str) -> dict[str, float]:
        claims = [c for c in state.claims if c.branch_id == branch_id]
        claim_ids = {c.id for c in claims}
        verified = sum(c.status in {ClaimStatus.VERIFIED, ClaimStatus.FORMALIZED} for c in claims)
        rejected = sum(c.status in {ClaimStatus.REJECTED, ClaimStatus.SCOPE_BLOCKED} for c in claims)
        supports = sum(
            e.result == EvidenceResult.SUPPORTS for e in state.evidence if e.claim_id in claim_ids
        )
        answered = sum(
            q.branch_id == branch_id and q.status == QuestionStatus.ANSWERED for q in state.questions
        )
        return {
            "claims": float(len(claims)),
            "verified": float(verified),
            "rejected": float(rejected),
            "supports": float(supports),
            "answered": float(answered),
        }

    def begin_round(self, state: ResearchState) -> None:
        if not self.enabled:
            return
        existing = {(e.kind, e.object_id, e.round_no) for e in state.policy.examples}
        for branch in state.branches:
            key = (PolicyExampleKind.BRANCH, branch.id, state.current_round)
            if key in existing or branch.status == BranchStatus.TERMINATED:
                continue
            state.policy.examples.append(PolicyExample(
                kind=PolicyExampleKind.BRANCH,
                round_no=state.current_round,
                object_id=branch.id,
                features=self.branch_features(state, branch),
                metadata=self._branch_metrics(state, branch.id),
            ))

        for question in state.questions:
            if question.status != QuestionStatus.ASSIGNED:
                continue
            key = (PolicyExampleKind.QUESTION, question.id, state.current_round)
            if key in existing:
                continue
            state.policy.examples.append(PolicyExample(
                kind=PolicyExampleKind.QUESTION,
                round_no=state.current_round,
                object_id=question.id,
                features=self.question_features(state, question),
                metadata={"created_round": question.created_round},
            ))

    @staticmethod
    def _branch_target(state: ResearchState, example: PolicyExample) -> float | None:
        branch = next((b for b in state.branches if b.id == example.object_id), None)
        if not branch:
            return None
        before = example.metadata
        after = ResearchPolicy._branch_metrics(state, branch.id)
        verified_gain = max(0.0, after["verified"] - float(before.get("verified", 0.0)))
        rejected_gain = max(0.0, after["rejected"] - float(before.get("rejected", 0.0)))
        support_gain = max(0.0, after["supports"] - float(before.get("supports", 0.0)))
        answered_gain = max(0.0, after["answered"] - float(before.get("answered", 0.0)))
        work_scale = max(1.0, after["claims"] - float(before.get("claims", 0.0)))
        utility = (
            0.12
            + 0.55 * _clamp(verified_gain / work_scale)
            + 0.15 * _clamp(support_gain / max(1.0, 2.0 * work_scale))
            + 0.13 * _clamp(answered_gain / 2.0)
            - 0.45 * _clamp(rejected_gain / work_scale)
        )
        if branch.status == BranchStatus.PROMISING:
            utility += 0.08
        elif branch.status == BranchStatus.TERMINATED:
            utility -= 0.12
        return _clamp(utility)

    @staticmethod
    def _question_target(state: ResearchState, example: PolicyExample) -> float | None:
        question = next((q for q in state.questions if q.id == example.object_id), None)
        if not question or question.status != QuestionStatus.ANSWERED:
            return None
        claims = [c for c in state.claims if c.id in set(question.answered_by_claim_ids)]
        if any(c.status in {ClaimStatus.VERIFIED, ClaimStatus.FORMALIZED} for c in claims):
            return 1.0
        if claims and all(c.status in {ClaimStatus.REJECTED, ClaimStatus.SCOPE_BLOCKED} for c in claims):
            return 0.0
        return 0.35

    def finalize_round_examples(self, state: ResearchState) -> int:
        finalized = 0
        for example in state.policy.examples:
            if example.target is not None or example.round_no != state.current_round:
                continue
            target = (
                self._branch_target(state, example)
                if example.kind == PolicyExampleKind.BRANCH
                else self._question_target(state, example)
            )
            if target is None:
                continue
            example.target = target
            example.finalized_round = state.current_round
            finalized += 1
        return finalized

    @staticmethod
    def _examples(state: ResearchState, kind: PolicyExampleKind) -> list[PolicyExample]:
        return [e for e in state.policy.examples if e.kind == kind and e.target is not None]

    def _train_one(
        self,
        state: ResearchState,
        *,
        kind: PolicyExampleKind,
        feature_count: int,
        min_samples: int,
        current: PolicyNetworkState | None,
    ) -> PolicyNetworkState | None:
        examples = self._examples(state, kind)
        if len(examples) < min_samples:
            return current
        x = np.asarray([e.features for e in examples], dtype=float)
        y = np.asarray([float(e.target) for e in examples], dtype=float)
        if current and current.input_size == feature_count:
            net = TinyMLPRegressor.from_state(current, seed=state.policy.seed)
        else:
            net = TinyMLPRegressor(feature_count, state.policy.hidden_size, seed=state.policy.seed)
        net.fit(
            x,
            y,
            epochs=state.policy.training_epochs,
            learning_rate=state.policy.learning_rate,
            l2=state.policy.l2,
        )
        return net.to_state(kind=kind, trained_examples=len(examples))

    def train(self, state: ResearchState) -> None:
        if not self.enabled:
            return
        state.policy.branch_network = self._train_one(
            state,
            kind=PolicyExampleKind.BRANCH,
            feature_count=len(BRANCH_FEATURE_NAMES),
            min_samples=state.policy.min_branch_examples,
            current=state.policy.branch_network,
        )
        state.policy.question_network = self._train_one(
            state,
            kind=PolicyExampleKind.QUESTION,
            feature_count=len(QUESTION_FEATURE_NAMES),
            min_samples=state.policy.min_question_examples,
            current=state.policy.question_network,
        )

    @staticmethod
    def _blend_for(network: PolicyNetworkState | None, minimum: int, policy: ResearchPolicyState) -> float:
        if not network or not network.active or network.trained_examples < minimum:
            return 0.0
        surplus = network.trained_examples - minimum + 1
        ramp = _clamp(surplus / max(1.0, float(policy.blend_warmup_examples)))
        return policy.max_neural_blend * ramp

    def predict_branch(self, state: ResearchState, branch: Branch) -> tuple[float | None, float]:
        if not self.enabled or not state.policy.branch_network:
            return None, 0.0
        network = state.policy.branch_network
        net = TinyMLPRegressor.from_state(network, seed=state.policy.seed)
        score = net.predict(self.branch_features(state, branch))
        blend = self._blend_for(network, state.policy.min_branch_examples, state.policy)
        return score, blend

    def score_questions(self, state: ResearchState) -> None:
        network = state.policy.question_network
        if not self.enabled or not network:
            for question in state.questions:
                question.policy_score = None
                question.policy_blend = 0.0
            return
        net = TinyMLPRegressor.from_state(network, seed=state.policy.seed)
        blend = self._blend_for(network, state.policy.min_question_examples, state.policy)
        for question in state.questions:
            if question.status not in {QuestionStatus.OPEN, QuestionStatus.ASSIGNED}:
                continue
            question.policy_score = net.predict(self.question_features(state, question))
            question.policy_blend = blend

    def learn_round(self, state: ResearchState) -> dict[str, int | float | None]:
        finalized = self.finalize_round_examples(state)
        self.train(state)
        self.score_questions(state)
        return {
            "finalized": finalized,
            "branch_examples": len(self._examples(state, PolicyExampleKind.BRANCH)),
            "question_examples": len(self._examples(state, PolicyExampleKind.QUESTION)),
            "branch_loss": state.policy.branch_network.loss if state.policy.branch_network else None,
            "question_loss": state.policy.question_network.loss if state.policy.question_network else None,
        }
