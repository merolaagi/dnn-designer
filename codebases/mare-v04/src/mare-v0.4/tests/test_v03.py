import pytest

from mare.assumption_firewall import AssumptionFirewall
from mare.benchmarks import burgers_blowup_problem
from mare.engine import ResearchEngine
from mare.formalization import MockLeanService
from mare.literature import InMemoryLiteratureProvider
from mare.models import (
    AuditStatus,
    Branch,
    Claim,
    ClaimScope,
    ClaimStatus,
    ResearchBudget,
    ResearchState,
    SourceRecord,
)
from mare.proof_graph import ProofGraph
from mare.providers.mock_provider import MockResearchProvider
from mare.storage import SQLiteRepository
from mare.tools.sandbox import DockerPythonSandbox


def test_assumption_firewall_blocks_extra_root_scope():
    problem = burgers_blowup_problem()
    claim = Claim(
        branch_id="B",
        statement="Restricted theorem",
        claim_type="theorem",
        derivation="d",
        extra_assumptions=["Assume axisymmetry"],
        scope=ClaimScope.ROOT_CANDIDATE,
        created_by="x",
        round_no=1,
    )
    audit = AssumptionFirewall().audit(problem, claim)
    assert audit.status == AuditStatus.BLOCK
    assert "axisymmetry" in audit.violations[0]


def test_proof_graph_detects_cycle():
    state = ResearchState(problem=burgers_blowup_problem())
    a = Claim(branch_id="B", statement="A", claim_type="lemma", derivation="", created_by="x", round_no=1, status=ClaimStatus.VERIFIED)
    b = Claim(branch_id="B", statement="B", claim_type="lemma", derivation="", created_by="x", round_no=1, status=ClaimStatus.VERIFIED)
    state.claims.extend([a, b])
    from mare.models import DependencyEdge
    state.dependency_edges.extend([
        DependencyEdge(parent_claim_id=a.id, child_claim_id=b.id),
        DependencyEdge(parent_claim_id=b.id, child_claim_id=a.id),
    ])
    assert ProofGraph(state).has_cycle()
    with pytest.raises(ValueError):
        ProofGraph(state).proof_order(a.id)


@pytest.mark.asyncio
async def test_literature_is_provenance_not_automatic_proof(tmp_path):
    source = SourceRecord(
        provider="fixture",
        title="Characteristics and gradient blow-up for Burgers equation",
        url="https://example.invalid/burgers",
        authors=["Example Author"],
        year=2026,
    )
    literature = InMemoryLiteratureProvider([source])
    state = ResearchState(problem=burgers_blowup_problem())
    engine = ResearchEngine(
        MockResearchProvider(),
        repository=SQLiteRepository(str(tmp_path / "mare.db")),
        literature_provider=literature,
    )
    result = await engine.run(state, rounds=1)
    assert result.sources
    assert result.source_links
    # Literature discovery is deliberately not counted as symbolic/formal proof evidence.
    linked = result.source_links[0]
    assert linked.relation == "relevant"


@pytest.mark.asyncio
async def test_budget_stops_unbounded_model_work(tmp_path):
    state = ResearchState(
        problem=burgers_blowup_problem(),
        budget=ResearchBudget(max_model_calls=2, max_reserved_output_tokens=8000, max_tool_calls=20, max_rounds=5),
    )
    engine = ResearchEngine(MockResearchProvider(), repository=SQLiteRepository(str(tmp_path / "mare.db")))
    result = await engine.run(state, rounds=3)
    assert result.budget.model_calls_used <= 2
    assert result.budget.reserved_output_tokens_used <= 8000
    assert any(e.event_type in {"task_budget_skipped", "review_budget_deferred", "synthesis_budget_skipped"} for e in result.events)


def test_docker_sandbox_command_has_security_controls(tmp_path):
    sandbox = DockerPythonSandbox()
    cmd = sandbox.build_command(tmp_path)
    joined = " ".join(cmd)
    assert "--network none" in joined
    assert "--read-only" in joined
    assert "--cap-drop ALL" in joined
    assert "no-new-privileges" in joined


def test_mock_lean_boundary_promotes_verified_closed_claim(tmp_path):
    state = ResearchState(problem=burgers_blowup_problem())
    branch = Branch(title="Formal", research_question="q", strategy="s")
    state.branches.append(branch)
    claim = Claim(
        branch_id=branch.id,
        statement="A verified root candidate",
        claim_type="theorem",
        derivation="d",
        scope=ClaimScope.ROOT_CANDIDATE,
        status=ClaimStatus.VERIFIED,
        created_by="x",
        round_no=1,
    )
    state.claims.append(claim)
    engine = ResearchEngine(
        MockResearchProvider(),
        repository=SQLiteRepository(str(tmp_path / "mare.db")),
        lean_service=MockLeanService(should_pass=True),
    )
    record = engine.formalize_claim(state, claim.id, "example : 1 = 1 := rfl")
    assert record.status.value == "passed"
    assert claim.status == ClaimStatus.FORMALIZED
    assert ProofGraph(state).ready_for_formalization(claim.id)


def test_pluggable_hash_embedding_memory_backend():
    from mare.memory import ResearchMemory
    from mare.memory_backends import HashEmbeddingBackend
    from mare.models import ResearchTask

    state = ResearchState(problem=burgers_blowup_problem())
    branch = Branch(title="Gradient", research_question="gradient blow-up", strategy="Riccati gradient dynamics")
    state.branches.append(branch)
    claim = Claim(
        branch_id=branch.id,
        statement="The gradient follows Riccati dynamics",
        claim_type="lemma",
        derivation="w prime equals minus w squared",
        tags=["gradient", "riccati"],
        status=ClaimStatus.VERIFIED,
        created_by="x",
        round_no=1,
    )
    state.claims.append(claim)
    task = ResearchTask(branch_id=branch.id, title="gradient", objective="study Riccati gradient dynamics")
    result = ResearchMemory(HashEmbeddingBackend()).retrieve(state, task)
    assert claim in result.claims
