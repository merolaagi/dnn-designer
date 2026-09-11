from __future__ import annotations

import asyncio

from .agents import Assassin, Director, Explorer, QuestionGenerator, Synthesizer, VerifierPlanner
from .assumption_firewall import AssumptionFirewall
from .budget import BudgetController, BudgetExhausted
from .contracts import ResearchQuestionDraft
from .formalization import LeanService
from .literature import LiteratureProvider
from .learned_policy import ResearchPolicy
from .memory import ResearchMemory
from .models import (
    AuditStatus,
    Branch,
    BranchStatus,
    Claim,
    ClaimScope,
    ClaimSourceLink,
    ClaimStatus,
    DependencyEdge,
    Evidence,
    EvidenceResult,
    Failure,
    FormalizationStatus,
    Objection,
    QuestionStatus,
    ResearchEvent,
    ResearchQuestion,
    ResearchState,
    ResearchTask,
    ToolInvocation,
)
from .navigator import Navigator
from .providers.base import ModelProvider
from .providers.tracing import TracingProvider
from .storage import SQLiteRepository
from .verification import run_check


class ResearchEngine:
    def __init__(
        self,
        provider: ModelProvider,
        repository: SQLiteRepository | None = None,
        explorer_count: int = 3,
        literature_provider: LiteratureProvider | None = None,
        lean_service: LeanService | None = None,
        policy_enabled: bool = True,
        research_policy: ResearchPolicy | None = None,
    ) -> None:
        self.provider = provider if isinstance(provider, TracingProvider) else TracingProvider(provider)
        self.repository = repository
        self.literature_provider = literature_provider
        self.lean_service = lean_service
        self.memory = ResearchMemory()
        self.director = Director(self.provider)
        self.explorers = [Explorer(self.provider, f"explorer-{i+1}", self.memory) for i in range(explorer_count)]
        self.assassins = [Assassin(self.provider, "assassin-A"), Assassin(self.provider, "assassin-B")]
        self.verifier_planner = VerifierPlanner(self.provider)
        self.synthesizer = Synthesizer(self.provider)
        self.question_generator = QuestionGenerator(self.provider)
        self.research_policy = research_policy or ResearchPolicy(enabled=policy_enabled)
        self.navigator = Navigator(self.research_policy)
        self.firewall = AssumptionFirewall()
        self.budget = BudgetController()

    def _event(self, state: ResearchState, kind: str, actor: str, object_id: str | None = None, **payload) -> None:
        event = ResearchEvent(
            event_type=kind,
            actor=actor,
            round_no=state.current_round,
            object_id=object_id,
            payload=payload,
        )
        state.events.append(event)
        if self.repository:
            self.repository.append_event(state.run_id, event)

    def _flush_invocations(self, state: ResearchState) -> None:
        records = self.provider.drain_records()
        if not records:
            return
        state.invocations.extend(records)
        state.budget.model_calls_used += len(records)
        state.budget.reserved_output_tokens_used += sum(r.max_tokens for r in records)
        for record in records:
            self._event(
                state,
                "model_invocation",
                record.provider,
                record.id,
                schema=record.schema_name,
                prompt_hash=record.prompt_hash,
                success=record.success,
                elapsed_ms=round(record.elapsed_ms, 3),
                max_tokens=record.max_tokens,
            )

    def _can_afford_model(self, state: ResearchState, *token_reservations: int) -> bool:
        calls = len(token_reservations)
        return (
            state.budget.model_calls_used + calls <= state.budget.max_model_calls
            and state.budget.reserved_output_tokens_used + sum(token_reservations)
            <= state.budget.max_reserved_output_tokens
        )

    def _persist(self, state: ResearchState) -> None:
        if self.repository:
            self.repository.save(state)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    def _existing_claim(self, state: ResearchState, statement: str) -> Claim | None:
        key = self._normalize(statement)
        return next((c for c in state.claims if self._normalize(c.statement) == key), None)

    def _is_duplicate_question(self, state: ResearchState, text: str) -> bool:
        key = self._normalize(text)
        return any(self._normalize(q.question) == key for q in state.questions)

    def _register_dependencies(self, state: ResearchState, claim: Claim) -> None:
        existing = {c.id for c in state.claims}
        existing_edges = {(e.parent_claim_id, e.child_claim_id, e.relation) for e in state.dependency_edges}
        valid_dependencies: list[str] = []
        for dep in claim.dependencies:
            if dep not in existing or dep == claim.id:
                continue
            valid_dependencies.append(dep)
            key = (dep, claim.id, "requires")
            if key in existing_edges:
                continue
            edge = DependencyEdge(
                parent_claim_id=dep,
                child_claim_id=claim.id,
                relation="requires",
                created_round=state.current_round,
            )
            state.dependency_edges.append(edge)
            existing_edges.add(key)
            self._event(state, "dependency_created", "normalizer", edge.id, parent=dep, child=claim.id)
        claim.dependencies = valid_dependencies

    async def initialize(self, state: ResearchState) -> None:
        if not self._can_afford_model(state, 4000):
            raise BudgetExhausted("insufficient model budget to initialize research director")
        plan = await self.director.plan(state.problem)
        self._flush_invocations(state)
        branch_by_title: dict[str, Branch] = {}
        for bp in plan.branches:
            branch = Branch(
                title=bp.title,
                research_question=bp.research_question,
                strategy=bp.strategy,
                novelty=bp.novelty,
                created_round=state.current_round,
            )
            state.branches.append(branch)
            branch_by_title[branch.title] = branch
            self._event(state, "branch_created", "director", branch.id, title=branch.title)

        for tp in plan.tasks:
            branch = branch_by_title[tp.branch_title]
            task = ResearchTask(
                branch_id=branch.id,
                title=tp.title,
                objective=tp.objective,
                round_no=1,
                source="director",
            )
            state.tasks.append(task)
            self._event(state, "task_created", "director", task.id, branch_id=branch.id)
        self._persist(state)

    async def _explore_task(
        self, state: ResearchState, task: ResearchTask, explorer: Explorer
    ) -> tuple[list[Claim], list[str]]:
        out = await explorer.investigate(state.problem, task, state)
        claims: list[Claim] = []
        for draft in out.claims:
            claim = Claim(
                branch_id=task.branch_id,
                statement=draft.statement,
                claim_type=draft.claim_type,
                derivation=draft.derivation,
                assumptions=draft.assumptions,
                dependencies=draft.dependencies,
                suggested_checks=draft.suggested_checks,
                tags=draft.tags,
                scope=draft.scope,
                extra_assumptions=draft.extra_assumptions,
                formal_statement=draft.formal_statement,
                created_by=explorer.name,
                round_no=state.current_round,
                question_id=task.question_id,
            )
            claims.append(claim)
        return claims, out.new_questions

    def _add_freeform_question(self, state: ResearchState, branch_id: str, text: str) -> None:
        if self._is_duplicate_question(state, text):
            return
        q = ResearchQuestion(
            branch_id=branch_id,
            question=text,
            rationale="Raised by an explorer during active investigation.",
            expected_information_gain=0.55,
            expected_impact=0.55,
            difficulty=0.5,
            novelty=0.5,
            created_round=state.current_round,
        )
        state.questions.append(q)
        self._event(state, "question_created", "explorer", q.id, branch_id=branch_id, priority=q.effective_priority)

    async def exploration_round(self, state: ResearchState) -> None:
        pending = [t for t in state.tasks if t.round_no == state.current_round]
        if not pending:
            return

        affordable: list[ResearchTask] = []
        projected_calls = 0
        projected_tokens = 0
        for task in pending:
            if (
                state.budget.model_calls_used + projected_calls + 1 <= state.budget.max_model_calls
                and state.budget.reserved_output_tokens_used + projected_tokens + task.token_budget
                <= state.budget.max_reserved_output_tokens
            ):
                affordable.append(task)
                projected_calls += 1
                projected_tokens += task.token_budget
            else:
                self._event(state, "task_budget_skipped", "budget-controller", task.id, token_budget=task.token_budget)

        coros = []
        for i, task in enumerate(affordable):
            explorer = self.explorers[i % len(self.explorers)]
            coros.append(self._explore_task(state, task, explorer))

        if not coros:
            return
        results = await asyncio.gather(*coros)
        self._flush_invocations(state)
        for task, (claim_list, new_questions) in zip(affordable, results):
            for question in new_questions:
                self._add_freeform_question(state, task.branch_id, question)
            for claim in claim_list:
                existing = self._existing_claim(state, claim.statement)
                if existing:
                    self._event(
                        state,
                        "claim_deduplicated",
                        "normalizer",
                        existing.id,
                        duplicate_of=claim.id,
                        statement=claim.statement,
                    )
                    if task.question_id:
                        try:
                            q = state.question(task.question_id)
                            if existing.id not in q.answered_by_claim_ids:
                                q.answered_by_claim_ids.append(existing.id)
                        except StopIteration:
                            pass
                    continue
                state.claims.append(claim)
                self._register_dependencies(state, claim)
                state.branch(claim.branch_id).compute_spent += 1.0
                self._event(state, "claim_proposed", claim.created_by, claim.id, branch_id=claim.branch_id)
        self._persist(state)

    async def adversarial_review(self, state: ResearchState, claims: list[Claim]) -> None:
        for claim in claims:
            needed = [4000] * len(self.assassins)
            if not self._can_afford_model(state, *needed):
                claim.status = ClaimStatus.CONTESTED
                self._event(state, "review_budget_deferred", "budget-controller", claim.id)
                continue
            reviews = await asyncio.gather(*[a.attack(state.problem, claim) for a in self.assassins])
            self._flush_invocations(state)
            fatal = any(r.fatal for r in reviews)
            seen_objections: set[str] = set()
            reusable: list[str] = []

            for assassin, review in zip(self.assassins, reviews):
                for text in review.objections:
                    normalized = self._normalize(text)
                    if normalized in seen_objections:
                        continue
                    seen_objections.add(normalized)
                    objection = Objection(
                        claim_id=claim.id,
                        severity="fatal" if review.fatal else (
                            review.severity if review.severity in {"minor", "major", "fatal"} else "major"
                        ),
                        description=text,
                        created_by=assassin.name,
                    )
                    state.objections.append(objection)
                    self._event(state, "objection_created", assassin.name, objection.id, claim_id=claim.id)
                if review.reusable_failure and review.reusable_failure not in reusable:
                    reusable.append(review.reusable_failure)

            if fatal:
                existing_failure = next((f for f in state.failures if f.failed_claim_id == claim.id), None)
                if not existing_failure:
                    failure = Failure(
                        branch_id=claim.branch_id,
                        failed_claim_id=claim.id,
                        mechanism="adversarial_falsification",
                        failure_reason="; ".join(o.description for o in state.objections_for(claim.id))
                        or "Fatal adversarial review",
                        reusable_constraint="; ".join(reusable) or None,
                        severity="fatal",
                        tags=claim.tags,
                        created_round=state.current_round,
                    )
                    state.failures.append(failure)
                    self._event(state, "failure_recorded", "review-pool", failure.id, claim_id=claim.id)
                claim.status = ClaimStatus.REJECTED
            else:
                claim.status = ClaimStatus.SURVIVED_REVIEW
                self._event(state, "claim_survived_review", "review-pool", claim.id)
        self._persist(state)

    def _refresh_question_for_claim(self, state: ResearchState, claim: Claim) -> None:
        if not claim.question_id:
            return
        try:
            question = state.question(claim.question_id)
        except StopIteration:
            return
        if claim.id not in question.answered_by_claim_ids:
            question.answered_by_claim_ids.append(claim.id)
        if claim.status in {
            ClaimStatus.VERIFIED, ClaimStatus.REJECTED, ClaimStatus.FORMALIZED, ClaimStatus.SCOPE_BLOCKED
        }:
            question.status = QuestionStatus.ANSWERED
            self._event(state, "question_answered", "verifier", question.id, claim_id=claim.id)

    def audit_claim(self, state: ResearchState, claim: Claim) -> None:
        audit = self.firewall.audit(state.problem, claim)
        previous = state.latest_audit(claim.id)
        if previous and previous.status == audit.status and previous.violations == audit.violations:
            return
        state.assumption_audits.append(audit)
        result = EvidenceResult.CONTRADICTS if audit.status == AuditStatus.BLOCK else EvidenceResult.SUPPORTS
        state.evidence.append(Evidence(
            claim_id=claim.id,
            evidence_type="scope_audit",
            result=result,
            verifier=audit.auditor,
            reproducible=True,
            details="; ".join(audit.violations) if audit.violations else audit.notes,
            metadata={"audit_status": audit.status.value},
        ))
        self._event(state, "assumption_audit", audit.auditor, audit.id, claim_id=claim.id, status=audit.status.value)
        if audit.status == AuditStatus.BLOCK:
            claim.status = ClaimStatus.SCOPE_BLOCKED
            claim.confidence = 0.0
            self._event(state, "claim_scope_blocked", audit.auditor, claim.id, violations=audit.violations)

    async def verify(self, state: ResearchState, claims: list[Claim]) -> None:
        for claim in claims:
            if claim.status == ClaimStatus.REJECTED:
                checks = list(dict.fromkeys(claim.suggested_checks))
            elif self._can_afford_model(state, 4000):
                plan = await self.verifier_planner.plan(claim)
                self._flush_invocations(state)
                checks = list(dict.fromkeys(claim.suggested_checks + plan.checks))
            else:
                checks = list(dict.fromkeys(claim.suggested_checks))
                self._event(state, "verification_plan_budget_skipped", "budget-controller", claim.id)

            for check in checks:
                if not state.budget.can_run_tool():
                    self._event(state, "tool_budget_exhausted", "budget-controller", claim.id, check=check)
                    break
                state.budget.reserve_tool()
                evidence = run_check(claim.id, check)
                state.evidence.append(evidence)
                state.tool_invocations.append(ToolInvocation(
                    tool="deterministic-verifier",
                    operation=check,
                    claim_id=claim.id,
                    success=evidence.result != EvidenceResult.INCONCLUSIVE,
                    details=evidence.details,
                ))
                self._event(
                    state,
                    "evidence_recorded",
                    evidence.verifier,
                    evidence.id,
                    claim_id=claim.id,
                    result=evidence.result.value,
                )

            ev = [e for e in state.evidence_for(claim.id) if e.evidence_type != "scope_audit"]
            if claim.status != ClaimStatus.REJECTED:
                if any(e.result == EvidenceResult.CONTRADICTS for e in ev):
                    claim.status = ClaimStatus.REJECTED
                elif ev and any(e.result == EvidenceResult.SUPPORTS for e in ev):
                    claim.status = ClaimStatus.VERIFIED
                    claim.confidence = 0.85
                    self._event(state, "claim_verified", "tool-verifier", claim.id)
            self.audit_claim(state, claim)
            self._refresh_question_for_claim(state, claim)
        self._persist(state)

    async def enrich_literature(self, state: ResearchState, claims: list[Claim], limit_per_claim: int = 3) -> None:
        if not self.literature_provider:
            return
        linked_claim_ids = {l.claim_id for l in state.source_links}
        for claim in claims:
            if claim.status not in {ClaimStatus.VERIFIED, ClaimStatus.FORMALIZED} or claim.id in linked_claim_ids:
                continue
            if not state.budget.can_run_tool():
                self._event(state, "literature_budget_exhausted", "budget-controller", claim.id)
                break
            state.budget.reserve_tool()
            query = " ".join([claim.statement, *claim.tags])
            try:
                records = await asyncio.to_thread(self.literature_provider.search, query, limit_per_claim)
                success = True
                details = f"found {len(records)} source(s)"
            except Exception as exc:  # retrieval failure should not invalidate mathematics
                records = []
                success = False
                details = f"{type(exc).__name__}: {exc}"
            state.tool_invocations.append(ToolInvocation(
                tool=getattr(self.literature_provider, "name", "literature"),
                operation="search",
                claim_id=claim.id,
                success=success,
                details=details,
            ))
            self._event(state, "literature_search", getattr(self.literature_provider, "name", "literature"), claim.id, success=success, count=len(records))
            for record in records:
                existing = next((s for s in state.sources if (
                    record.external_id and s.external_id == record.external_id
                ) or (record.url and s.url == record.url) or self._normalize(s.title) == self._normalize(record.title)), None)
                source = existing or record
                if not existing:
                    state.sources.append(source)
                if not any(l.claim_id == claim.id and l.source_id == source.id for l in state.source_links):
                    state.source_links.append(ClaimSourceLink(
                        claim_id=claim.id,
                        source_id=source.id,
                        relation="relevant",
                        note="Retrieved as potentially relevant prior art; not treated as proof verification.",
                    ))
        self._persist(state)

    async def synthesize(self, state: ResearchState) -> list[Claim]:
        if not self._can_afford_model(state, 4000):
            self._event(state, "synthesis_budget_skipped", "budget-controller")
            return []
        out = await self.synthesizer.synthesize(state)
        self._flush_invocations(state)
        for note in [*out.cross_branch_links, out.notes]:
            if note and note not in state.synthesis_notes:
                state.synthesis_notes.append(note)
        if not out.claims:
            return []

        synthesis_branch = next((b for b in state.branches if b.title == "Synthesis"), None)
        if synthesis_branch is None:
            synthesis_branch = Branch(
                title="Synthesis",
                research_question="What follows by combining independently supported branches?",
                strategy="cross-pollination and dependency closure",
                novelty=0.75,
                created_round=state.current_round,
            )
            state.branches.append(synthesis_branch)
            self._event(state, "branch_created", "synthesizer", synthesis_branch.id, title="Synthesis")

        claims: list[Claim] = []
        verified_ids = [c.id for c in state.claims if c.status == ClaimStatus.VERIFIED]
        known_ids = {c.id for c in state.claims}
        for draft in out.claims:
            existing = self._existing_claim(state, draft.statement)
            if existing:
                self._event(state, "claim_deduplicated", "normalizer", existing.id, statement=draft.statement)
                continue
            explicit = [d for d in draft.dependencies if d in known_ids]
            dependencies = explicit or verified_ids
            scope = draft.scope
            if draft.claim_type == "theorem" and scope == ClaimScope.LOCAL_LEMMA:
                scope = ClaimScope.ROOT_CANDIDATE
            claim = Claim(
                branch_id=synthesis_branch.id,
                statement=draft.statement,
                claim_type=draft.claim_type,
                derivation=draft.derivation,
                assumptions=draft.assumptions,
                dependencies=dependencies,
                suggested_checks=draft.suggested_checks,
                tags=draft.tags,
                scope=scope,
                extra_assumptions=draft.extra_assumptions,
                formal_statement=draft.formal_statement,
                created_by="synthesizer",
                round_no=state.current_round,
            )
            state.claims.append(claim)
            self._register_dependencies(state, claim)
            claims.append(claim)
            self._event(state, "claim_proposed", "synthesizer", claim.id, branch_id=claim.branch_id)
        self._persist(state)
        return claims

    def _question_from_draft(self, state: ResearchState, draft: ResearchQuestionDraft) -> ResearchQuestion | None:
        if self._is_duplicate_question(state, draft.question):
            return None
        branch = next((b for b in state.branches if b.title == draft.branch_title), None)
        question = ResearchQuestion(
            branch_id=branch.id if branch else None,
            question=draft.question,
            rationale=draft.rationale,
            expected_information_gain=draft.expected_information_gain,
            expected_impact=draft.expected_impact,
            difficulty=draft.difficulty,
            novelty=draft.novelty,
            spawn_new_branch=draft.spawn_new_branch,
            suggested_branch_title=draft.branch_title if draft.spawn_new_branch else None,
            suggested_strategy=draft.suggested_strategy,
            created_round=state.current_round,
        )
        state.questions.append(question)
        self._event(
            state,
            "question_created",
            "question-generator",
            question.id,
            branch_id=question.branch_id,
            priority=question.priority,
        )
        return question

    async def generate_questions(self, state: ResearchState) -> list[ResearchQuestion]:
        if not self._can_afford_model(state, 4000):
            self._event(state, "question_generation_budget_skipped", "budget-controller")
            return []
        out = await self.question_generator.generate(state)
        self._flush_invocations(state)
        created: list[ResearchQuestion] = []
        for draft in out.questions:
            q = self._question_from_draft(state, draft)
            if q:
                created.append(q)
        self.research_policy.score_questions(state)
        self._persist(state)
        return created

    def _spawn_branch_for_question(self, state: ResearchState, q: ResearchQuestion) -> Branch | None:
        if not q.spawn_new_branch:
            return None
        title = q.suggested_branch_title or f"Question {q.id}"
        existing = next((b for b in state.branches if b.title == title), None)
        if existing:
            q.branch_id = existing.id
            return existing
        branch = Branch(
            title=title,
            research_question=q.question,
            strategy=q.suggested_strategy or "Investigate this distinct mechanism independently.",
            parent_id=q.branch_id,
            novelty=q.novelty,
            created_round=state.current_round + 1,
        )
        state.branches.append(branch)
        q.branch_id = branch.id
        self._event(state, "branch_spawned", "question-engine", branch.id, question_id=q.id, title=title)
        return branch

    def prepare_next_round(self, state: ResearchState, max_tasks: int = 3) -> None:
        self.research_policy.score_questions(state)
        self.navigator.score(state)
        self.navigator.update_lifecycle(state)
        next_round = state.current_round + 1
        if next_round > state.budget.max_rounds:
            self._event(state, "round_budget_exhausted", "budget-controller", next_round=next_round)
            return
        existing = {(t.question_id, t.round_no) for t in state.tasks if t.question_id}
        scheduled = 0
        projected_tokens = state.budget.reserved_output_tokens_used
        projected_calls = state.budget.model_calls_used

        def can_schedule(task: ResearchTask) -> bool:
            nonlocal projected_calls, projected_tokens
            if projected_calls + 1 > state.budget.max_model_calls:
                return False
            if projected_tokens + task.token_budget > state.budget.max_reserved_output_tokens:
                return False
            projected_calls += 1
            projected_tokens += task.token_budget
            return True

        open_questions = sorted(
            [q for q in state.questions if q.status == QuestionStatus.OPEN],
            key=lambda q: q.effective_priority,
            reverse=True,
        )
        for q in open_questions:
            if scheduled >= max_tasks:
                break
            if (q.id, next_round) in existing:
                continue
            if q.spawn_new_branch:
                self._spawn_branch_for_question(state, q)
            if not q.branch_id:
                candidates = [
                    b for b in self.navigator.ranked(state)
                    if b.title != "Synthesis" and b.status != BranchStatus.TERMINATED
                ]
                if not candidates:
                    continue
                q.branch_id = candidates[0].id
            branch = state.branch(q.branch_id)
            if branch.status == BranchStatus.TERMINATED:
                continue
            task = ResearchTask(
                branch_id=branch.id,
                title=f"Investigate {q.id}",
                objective=q.question,
                round_no=next_round,
                question_id=q.id,
                source="question_engine",
            )
            if not can_schedule(task):
                self._event(state, "task_budget_not_scheduled", "budget-controller", task.id, question_id=q.id)
                break
            state.tasks.append(task)
            q.status = QuestionStatus.ASSIGNED
            scheduled += 1
            self._event(state, "task_created", "question-engine", task.id, branch_id=branch.id, question_id=q.id)

        if scheduled < max_tasks:
            candidates = [
                b for b in self.navigator.ranked(state)
                if b.title != "Synthesis" and b.status != BranchStatus.TERMINATED
            ]
            existing_branch_tasks = {(t.branch_id, t.round_no) for t in state.tasks}
            for branch in candidates:
                if scheduled >= max_tasks:
                    break
                if (branch.id, next_round) in existing_branch_tasks:
                    continue
                task = ResearchTask(
                    branch_id=branch.id,
                    title=f"Deepen {branch.title}",
                    objective=f"Resolve the highest-value remaining uncertainty in: {branch.research_question}",
                    round_no=next_round,
                    source="navigator",
                )
                if not can_schedule(task):
                    self._event(state, "task_budget_not_scheduled", "budget-controller", task.id, branch_id=branch.id)
                    break
                state.tasks.append(task)
                scheduled += 1
                self._event(state, "task_created", "navigator", task.id, branch_id=branch.id)
        self._persist(state)

    def formalize_claim(self, state: ResearchState, claim_id: str, source_code: str):
        if not self.lean_service:
            raise RuntimeError("no Lean service configured")
        claim = state.claim(claim_id)
        self.audit_claim(state, claim)
        audit = state.latest_audit(claim.id)
        if audit and audit.status == AuditStatus.BLOCK:
            raise RuntimeError("claim is blocked by the assumption firewall")
        if state.proof_completeness(claim.id) < 1.0 or state.unresolved_dependencies(claim.id):
            raise RuntimeError("claim dependency graph is not verified/closed")
        if not state.budget.can_run_tool():
            raise BudgetExhausted("insufficient tool budget for Lean formalization")
        state.budget.reserve_tool()
        record = self.lean_service.check(claim, source_code)
        state.formalizations.append(record)
        state.tool_invocations.append(ToolInvocation(
            tool=record.prover,
            operation="formalize",
            claim_id=claim.id,
            success=record.status == FormalizationStatus.PASSED,
            elapsed_ms=record.elapsed_ms,
            details=record.stderr or record.stdout,
        ))
        if record.status == FormalizationStatus.PASSED:
            claim.status = ClaimStatus.FORMALIZED
            claim.confidence = 1.0
            state.evidence.append(Evidence(
                claim_id=claim.id,
                evidence_type="formal_proof",
                result=EvidenceResult.SUPPORTS,
                verifier=record.prover,
                reproducible=True,
                details="Lean service accepted the supplied formalization.",
            ))
            self._event(state, "claim_formalized", record.prover, record.id, claim_id=claim.id)
        else:
            self._event(state, "formalization_failed", record.prover, record.id, claim_id=claim.id, status=record.status.value)
        self._persist(state)
        return record

    async def run(self, state: ResearchState, rounds: int = 2) -> ResearchState:
        if rounds < 1:
            return state
        if not state.branches:
            await self.initialize(state)
            start_round = 1
        else:
            start_round = max(1, state.current_round + 1)
            if state.current_round > 0 and not any(t.round_no == start_round for t in state.tasks):
                self.prepare_next_round(state)

        final_round = min(start_round + rounds - 1, state.budget.max_rounds)
        for round_no in range(start_round, final_round + 1):
            self.budget.check_round(state, round_no)
            state.current_round = round_no
            self.research_policy.begin_round(state)
            self._event(
                state,
                "policy_round_started",
                "neural-policy",
                round_no=round_no,
                examples=len(state.policy.examples),
            )
            before = len(state.claims)
            await self.exploration_round(state)
            new_claims = state.claims[before:]
            if new_claims:
                await self.adversarial_review(state, new_claims)
                await self.verify(state, new_claims)
                await self.enrich_literature(state, new_claims)

            synth_claims = await self.synthesize(state)
            if synth_claims:
                await self.adversarial_review(state, synth_claims)
                await self.verify(state, synth_claims)
                await self.enrich_literature(state, synth_claims)

            policy_metrics = self.research_policy.learn_round(state)
            self._event(
                state,
                "policy_trained",
                "neural-policy",
                finalized=policy_metrics["finalized"],
                branch_examples=policy_metrics["branch_examples"],
                question_examples=policy_metrics["question_examples"],
                branch_loss=policy_metrics["branch_loss"],
                question_loss=policy_metrics["question_loss"],
                branch_active=bool(state.policy.branch_network and state.policy.branch_network.active),
                question_active=bool(state.policy.question_network and state.policy.question_network.active),
            )
            self.navigator.score(state)
            self.navigator.update_lifecycle(state)
            await self.generate_questions(state)
            if round_no < final_round:
                self.prepare_next_round(state)
            self._persist(state)

            # Stop cleanly when no further model call can be afforded.
            if not self._can_afford_model(state, 1):
                self._event(state, "research_budget_exhausted", "budget-controller")
                break

        return state
