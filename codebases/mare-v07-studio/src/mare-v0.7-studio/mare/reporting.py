from __future__ import annotations

from .models import ClaimStatus, ResearchState
from .proof_graph import ProofGraph


def text_report(state: ResearchState) -> str:
    b = state.budget
    lines = [
        f"MARE run: {state.run_id}",
        f"Problem: {state.problem.title}",
        f"Rounds: {state.current_round}",
        f"Model invocations: {len(state.invocations)}",
        (
            "Budget: "
            f"model_calls={b.model_calls_used}/{b.max_model_calls} "
            f"reserved_output_tokens={b.reserved_output_tokens_used}/{b.max_reserved_output_tokens} "
            f"tool_calls={b.tool_calls_used}/{b.max_tool_calls}"
        ),
        "",
        "Branches:",
    ]
    for branch in sorted(state.branches, key=lambda x: x.score, reverse=True):
        policy_text = (
            f" neural={branch.policy_score:.3f}@{branch.policy_blend:.2f}"
            if branch.policy_score is not None and branch.policy_blend > 0.0
            else ""
        )
        lines.append(
            f"  {branch.title:24} score={branch.score:.3f} heuristic={branch.heuristic_score:.3f}"
            f"{policy_text} status={branch.status.value:13} progress={branch.verified_progress:.2f}"
        )

    if state.questions:
        lines.append("\nResearch questions:")
        for q in sorted(state.questions, key=lambda x: x.effective_priority, reverse=True):
            policy_text = (
                f" neural={q.policy_score:.3f}@{q.policy_blend:.2f}"
                if q.policy_score is not None and q.policy_blend > 0.0
                else ""
            )
            lines.append(
                f"  [{q.status.value:10}] p={q.effective_priority:.3f} "
                f"heuristic={q.priority:.3f}{policy_text} {q.id}: {q.question}"
            )

    lines.append("\nClaims:")
    for c in state.claims:
        lines.append(f"  [{c.status.value:16}] {c.id}: {c.statement}")
        if c.dependencies:
            lines.append(f"      - requires: {', '.join(c.dependencies)}")
        audit = state.latest_audit(c.id)
        if audit:
            lines.append(f"      - scope-audit/{audit.status.value}: {audit.notes or '; '.join(audit.violations)}")
            for violation in audit.violations:
                lines.append(f"        ! {violation}")
        for e in state.evidence_for(c.id):
            if e.evidence_type == "scope_audit":
                continue
            lines.append(f"      - {e.result.value}: {e.verifier} — {e.details}")
        for o in state.objections_for(c.id):
            lines.append(f"      - objection/{o.severity}: {o.description}")
        links = [l for l in state.source_links if l.claim_id == c.id]
        for link in links:
            src = next((s for s in state.sources if s.id == link.source_id), None)
            if src:
                lines.append(f"      - source/{link.relation}: {src.title} ({src.provider})")

    if state.failures:
        lines.append("\nReusable failures:")
        for f in state.failures:
            lines.append(f"  {f.id}: {f.reusable_constraint or f.failure_reason}")

    graph = ProofGraph(state)
    theorem_roots = [cid for cid in graph.candidate_roots() if state.claim(cid).claim_type == "theorem"]
    if theorem_roots:
        lines.append("\nCandidate proof roots:")
        for cid in theorem_roots[:5]:
            claim = state.claim(cid)
            completeness = state.proof_completeness(cid)
            unresolved = state.unresolved_dependencies(cid)
            lines.append(
                f"  {cid}: completeness={completeness:.1%} status={claim.status.value} "
                f"unresolved={len(unresolved)} formalization_ready={graph.ready_for_formalization(cid)}"
            )

    if state.formalizations:
        lines.append("\nFormalization attempts:")
        for f in state.formalizations:
            lines.append(f"  {f.id}: claim={f.claim_id} prover={f.prover} status={f.status.value}")


    policy = state.policy
    branch_examples = sum(e.kind.value == "branch" and e.target is not None for e in policy.examples)
    question_examples = sum(e.kind.value == "question" and e.target is not None for e in policy.examples)
    lines.append("\nNeural research policy:")
    lines.append(
        f"  enabled={policy.enabled} branch_examples={branch_examples} question_examples={question_examples} "
        f"max_blend={policy.max_neural_blend:.2f}"
    )
    if policy.branch_network:
        lines.append(
            f"  branch_net: active={policy.branch_network.active} trained={policy.branch_network.trained_examples} "
            f"loss={policy.branch_network.loss if policy.branch_network.loss is not None else 'n/a'}"
        )
    else:
        lines.append(f"  branch_net: cold-start (needs {policy.min_branch_examples} labeled examples)")
    if policy.question_network:
        lines.append(
            f"  question_net: active={policy.question_network.active} trained={policy.question_network.trained_examples} "
            f"loss={policy.question_network.loss if policy.question_network.loss is not None else 'n/a'}"
        )
    else:
        lines.append(f"  question_net: cold-start (needs {policy.min_question_examples} labeled examples)")

    if state.synthesis_notes:
        lines.append("\nSynthesis:")
        for n in state.synthesis_notes:
            if n:
                lines.append(f"  - {n}")

    verified = sum(1 for c in state.claims if c.status in {ClaimStatus.VERIFIED, ClaimStatus.FORMALIZED})
    rejected = sum(1 for c in state.claims if c.status == ClaimStatus.REJECTED)
    blocked = sum(1 for c in state.claims if c.status == ClaimStatus.SCOPE_BLOCKED)
    lines.extend([
        "",
        f"Summary: verified={verified} rejected={rejected} scope_blocked={blocked} failures={len(state.failures)} "
        f"questions={len(state.questions)} dependency_edges={len(state.dependency_edges)} sources={len(state.sources)}",
    ])
    return "\n".join(lines)
