from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:10]}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ClaimStatus(str, Enum):
    PROPOSED = "proposed"
    CONTESTED = "contested"
    SURVIVED_REVIEW = "survived_review"
    VERIFIED = "verified"
    REJECTED = "rejected"
    FORMALIZED = "formalized"
    SCOPE_BLOCKED = "scope_blocked"


class EvidenceResult(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INCONCLUSIVE = "inconclusive"


class BranchStatus(str, Enum):
    ACTIVE = "active"
    PROMISING = "promising"
    DEPRIORITIZED = "deprioritized"
    TERMINATED = "terminated"
    SOLVED = "solved"


class QuestionStatus(str, Enum):
    OPEN = "open"
    ASSIGNED = "assigned"
    ANSWERED = "answered"
    INVALIDATED = "invalidated"


class ClaimScope(str, Enum):
    LOCAL_LEMMA = "local_lemma"
    RESTRICTED_RESULT = "restricted_result"
    ROOT_CANDIDATE = "root_candidate"


class AuditStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


class FormalizationStatus(str, Enum):
    QUEUED = "queued"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PolicyExampleKind(str, Enum):
    BRANCH = "branch"
    QUESTION = "question"


class ProblemSpec(BaseModel):
    id: str = Field(default_factory=lambda: new_id("P"))
    title: str
    canonical_statement: str
    definitions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    success_conditions: list[str] = Field(default_factory=list)
    disallowed_shortcuts: list[str] = Field(default_factory=list)
    scope_constraints: list[str] = Field(default_factory=list)
    verification_requirements: list[str] = Field(default_factory=list)
    formal_target: str | None = None


class ResearchBudget(BaseModel):
    """Hard research-run limits.

    V0.3 accounts for *reserved* model output tokens because provider-independent
    actual token usage is not always available. Providers may additionally report
    actual input/output usage on ModelInvocation.
    """

    max_model_calls: int = 250
    max_reserved_output_tokens: int = 1_000_000
    max_tool_calls: int = 2_000
    max_rounds: int = 100
    model_calls_used: int = 0
    reserved_output_tokens_used: int = 0
    tool_calls_used: int = 0

    def can_reserve_model_call(self, max_tokens: int) -> bool:
        return (
            self.model_calls_used + 1 <= self.max_model_calls
            and self.reserved_output_tokens_used + max_tokens <= self.max_reserved_output_tokens
        )

    def reserve_model_call(self, max_tokens: int) -> None:
        if not self.can_reserve_model_call(max_tokens):
            raise RuntimeError("research model-call budget exhausted")
        self.model_calls_used += 1
        self.reserved_output_tokens_used += max_tokens

    def can_run_tool(self, count: int = 1) -> bool:
        return self.tool_calls_used + count <= self.max_tool_calls

    def reserve_tool(self, count: int = 1) -> None:
        if not self.can_run_tool(count):
            raise RuntimeError("research tool-call budget exhausted")
        self.tool_calls_used += count


class Branch(BaseModel):
    id: str = Field(default_factory=lambda: new_id("B"))
    title: str
    research_question: str
    strategy: str
    parent_id: str | None = None
    status: BranchStatus = BranchStatus.ACTIVE
    score: float = 0.5
    compute_spent: float = 0.0
    verified_progress: float = 0.0
    novelty: float = 0.5
    uncertainty: float = 1.0
    consecutive_low_score_rounds: int = 0
    created_round: int = 0
    heuristic_score: float = 0.5
    policy_score: float | None = None
    policy_blend: float = 0.0


class ResearchQuestion(BaseModel):
    id: str = Field(default_factory=lambda: new_id("Q"))
    branch_id: str | None = None
    question: str
    rationale: str = ""
    expected_information_gain: float = 0.5
    expected_impact: float = 0.5
    difficulty: float = 0.5
    novelty: float = 0.5
    spawn_new_branch: bool = False
    suggested_branch_title: str | None = None
    suggested_strategy: str | None = None
    status: QuestionStatus = QuestionStatus.OPEN
    created_round: int = 0
    answered_by_claim_ids: list[str] = Field(default_factory=list)
    policy_score: float | None = None
    policy_blend: float = 0.0

    @property
    def priority(self) -> float:
        return max(
            0.0,
            min(
                1.0,
                0.45 * self.expected_information_gain
                + 0.35 * self.expected_impact
                + 0.20 * self.novelty
                - 0.15 * self.difficulty,
            ),
        )

    @property
    def effective_priority(self) -> float:
        if self.policy_score is None or self.policy_blend <= 0.0:
            return self.priority
        blend = max(0.0, min(1.0, self.policy_blend))
        return max(0.0, min(1.0, (1.0 - blend) * self.priority + blend * self.policy_score))


class ResearchTask(BaseModel):
    id: str = Field(default_factory=lambda: new_id("T"))
    branch_id: str
    title: str
    objective: str
    round_no: int = 1
    token_budget: int = 4000
    question_id: str | None = None
    source: Literal["director", "navigator", "question_engine", "human", "proof_graph"] = "director"


class ClaimDraft(BaseModel):
    statement: str
    claim_type: Literal[
        "definition", "observation", "conjecture", "lemma", "theorem",
        "counterexample", "bound", "identity"
    ] = "conjecture"
    derivation: str
    assumptions: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    suggested_checks: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    scope: ClaimScope = ClaimScope.LOCAL_LEMMA
    extra_assumptions: list[str] = Field(default_factory=list)
    formal_statement: str | None = None


class Claim(BaseModel):
    id: str = Field(default_factory=lambda: new_id("C"))
    branch_id: str
    statement: str
    claim_type: str
    derivation: str
    assumptions: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    suggested_checks: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    scope: ClaimScope = ClaimScope.LOCAL_LEMMA
    extra_assumptions: list[str] = Field(default_factory=list)
    formal_statement: str | None = None
    status: ClaimStatus = ClaimStatus.PROPOSED
    confidence: float = 0.0
    created_by: str
    round_no: int
    question_id: str | None = None


class DependencyEdge(BaseModel):
    id: str = Field(default_factory=lambda: new_id("D"))
    parent_claim_id: str
    child_claim_id: str
    relation: Literal["requires", "supports", "contradicts"] = "requires"
    created_round: int = 0


class Objection(BaseModel):
    id: str = Field(default_factory=lambda: new_id("O"))
    claim_id: str
    severity: Literal["minor", "major", "fatal"]
    description: str
    resolved: bool = False
    created_by: str = "assassin"


class Failure(BaseModel):
    id: str = Field(default_factory=lambda: new_id("F"))
    branch_id: str
    failed_claim_id: str | None = None
    mechanism: str
    failure_reason: str
    reusable_constraint: str | None = None
    severity: Literal["minor", "major", "fatal"] = "major"
    tags: list[str] = Field(default_factory=list)
    created_round: int = 0


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: new_id("E"))
    claim_id: str
    evidence_type: Literal[
        "symbolic", "numerical", "citation", "independent_derivation",
        "counterexample_search", "formal_proof", "adversarial_review", "scope_audit"
    ]
    result: EvidenceResult
    verifier: str
    reproducible: bool = True
    details: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("SRC"))
    provider: str
    title: str
    url: str | None = None
    external_id: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    retrieved_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimSourceLink(BaseModel):
    id: str = Field(default_factory=lambda: new_id("CS"))
    claim_id: str
    source_id: str
    relation: Literal["relevant", "supports", "contradicts", "background"] = "relevant"
    note: str = ""


class AssumptionAudit(BaseModel):
    id: str = Field(default_factory=lambda: new_id("AA"))
    claim_id: str
    status: AuditStatus
    extra_assumptions: list[str] = Field(default_factory=list)
    matched_problem_constraints: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    notes: str = ""
    auditor: str = "assumption-firewall"
    timestamp: datetime = Field(default_factory=utc_now)


class FormalizationRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("FM"))
    claim_id: str
    status: FormalizationStatus
    prover: str
    source_code: str = ""
    stdout: str = ""
    stderr: str = ""
    elapsed_ms: float = 0.0
    artifact_path: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class ToolInvocation(BaseModel):
    id: str = Field(default_factory=lambda: new_id("TI"))
    tool: str
    operation: str
    claim_id: str | None = None
    success: bool
    elapsed_ms: float = 0.0
    details: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class ModelInvocation(BaseModel):
    id: str = Field(default_factory=lambda: new_id("MI"))
    provider: str
    model: str | None = None
    schema_name: str
    system_hash: str
    prompt_hash: str
    max_tokens: int
    temperature: float
    elapsed_ms: float
    success: bool
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class PolicyExample(BaseModel):
    id: str = Field(default_factory=lambda: new_id("PE"))
    kind: PolicyExampleKind
    round_no: int
    object_id: str
    features: list[float]
    target: float | None = None
    finalized_round: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyNetworkState(BaseModel):
    kind: PolicyExampleKind
    input_size: int
    hidden_size: int
    trained_examples: int = 0
    epochs: int = 0
    loss: float | None = None
    weights1: list[list[float]] = Field(default_factory=list)
    bias1: list[float] = Field(default_factory=list)
    weights2: list[float] = Field(default_factory=list)
    bias2: float = 0.0
    active: bool = False
    updated_at: datetime = Field(default_factory=utc_now)


class ResearchPolicyState(BaseModel):
    enabled: bool = True
    hidden_size: int = 12
    min_branch_examples: int = 12
    min_question_examples: int = 16
    training_epochs: int = 220
    learning_rate: float = 0.03
    l2: float = 1e-4
    seed: int = 7
    max_neural_blend: float = 0.65
    blend_warmup_examples: int = 20
    examples: list[PolicyExample] = Field(default_factory=list)
    branch_network: PolicyNetworkState | None = None
    question_network: PolicyNetworkState | None = None


class ResearchEvent(BaseModel):
    id: str = Field(default_factory=lambda: new_id("EV"))
    event_type: str
    actor: str
    round_no: int
    object_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class ResearchState(BaseModel):
    run_id: str = Field(default_factory=lambda: new_id("RUN"))
    problem: ProblemSpec
    budget: ResearchBudget = Field(default_factory=ResearchBudget)
    policy: ResearchPolicyState = Field(default_factory=ResearchPolicyState)
    current_round: int = 0
    branches: list[Branch] = Field(default_factory=list)
    questions: list[ResearchQuestion] = Field(default_factory=list)
    tasks: list[ResearchTask] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    dependency_edges: list[DependencyEdge] = Field(default_factory=list)
    objections: list[Objection] = Field(default_factory=list)
    failures: list[Failure] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    sources: list[SourceRecord] = Field(default_factory=list)
    source_links: list[ClaimSourceLink] = Field(default_factory=list)
    assumption_audits: list[AssumptionAudit] = Field(default_factory=list)
    formalizations: list[FormalizationRecord] = Field(default_factory=list)
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    invocations: list[ModelInvocation] = Field(default_factory=list)
    events: list[ResearchEvent] = Field(default_factory=list)
    synthesis_notes: list[str] = Field(default_factory=list)

    def branch(self, branch_id: str) -> Branch:
        return next(b for b in self.branches if b.id == branch_id)

    def claim(self, claim_id: str) -> Claim:
        return next(c for c in self.claims if c.id == claim_id)

    def question(self, question_id: str) -> ResearchQuestion:
        return next(q for q in self.questions if q.id == question_id)

    def evidence_for(self, claim_id: str) -> list[Evidence]:
        return [e for e in self.evidence if e.claim_id == claim_id]

    def objections_for(self, claim_id: str) -> list[Objection]:
        return [o for o in self.objections if o.claim_id == claim_id]

    def audits_for(self, claim_id: str) -> list[AssumptionAudit]:
        return [a for a in self.assumption_audits if a.claim_id == claim_id]

    def latest_audit(self, claim_id: str) -> AssumptionAudit | None:
        audits = self.audits_for(claim_id)
        return audits[-1] if audits else None

    def dependencies_of(self, claim_id: str) -> list[str]:
        direct = [
            e.parent_claim_id
            for e in self.dependency_edges
            if e.child_claim_id == claim_id and e.relation == "requires"
        ]
        if direct:
            return direct
        try:
            return list(self.claim(claim_id).dependencies)
        except StopIteration:
            return []

    def dependency_closure(self, claim_id: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self.dependencies_of(claim_id))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.dependencies_of(current))
        return seen

    def proof_completeness(self, claim_id: str) -> float:
        try:
            self.claim(claim_id)
        except StopIteration:
            return 0.0
        nodes = self.dependency_closure(claim_id) | {claim_id}
        if not nodes:
            return 0.0
        good = 0
        for cid in nodes:
            try:
                status = self.claim(cid).status
            except StopIteration:
                continue
            if status in {ClaimStatus.VERIFIED, ClaimStatus.FORMALIZED}:
                good += 1
        return good / len(nodes)

    def unresolved_dependencies(self, claim_id: str) -> list[str]:
        out: list[str] = []
        for cid in self.dependency_closure(claim_id):
            try:
                if self.claim(cid).status not in {ClaimStatus.VERIFIED, ClaimStatus.FORMALIZED}:
                    out.append(cid)
            except StopIteration:
                out.append(cid)
        return out
