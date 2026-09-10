import pytest

from mare.benchmarks import burgers_blowup_problem
from mare.engine import ResearchEngine
from mare.models import ClaimStatus, QuestionStatus, ResearchState
from mare.proof_graph import ProofGraph
from mare.providers.mock_provider import MockResearchProvider
from mare.storage import SQLiteRepository


@pytest.mark.asyncio
async def test_v02_questions_spawn_branch_and_reuse_failure(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "mare.db"))
    state = ResearchState(problem=burgers_blowup_problem())
    engine = ResearchEngine(MockResearchProvider(), repository=repo)
    result = await engine.run(state, rounds=2)

    assert any(b.title == "Failure reconciliation" for b in result.branches)
    assert any("conserved L2" in q.question for q in result.questions)
    assert any(q.status == QuestionStatus.ANSWERED for q in result.questions)
    assert any("compatible with unbounded spatial gradients" in c.statement and c.status == ClaimStatus.VERIFIED for c in result.claims)
    assert len(result.invocations) > 0
    assert all(i.prompt_hash and i.system_hash for i in result.invocations)


@pytest.mark.asyncio
async def test_v02_synthesis_has_dependency_dag(tmp_path):
    state = ResearchState(problem=burgers_blowup_problem())
    engine = ResearchEngine(MockResearchProvider(), repository=SQLiteRepository(str(tmp_path / "mare.db")))
    result = await engine.run(state, rounds=1)
    theorem = next(c for c in result.claims if c.claim_type == "theorem")
    assert theorem.dependencies
    assert len(result.dependency_edges) >= 2
    assert result.proof_completeness(theorem.id) == 1.0
    assert ProofGraph(result).is_dependency_closed(theorem.id)


@pytest.mark.asyncio
async def test_persisted_run_can_resume_for_additional_round(tmp_path):
    db = str(tmp_path / "mare.db")
    repo = SQLiteRepository(db)
    engine = ResearchEngine(MockResearchProvider(), repository=repo)
    first = await engine.run(ResearchState(problem=burgers_blowup_problem()), rounds=1)
    assert first.current_round == 1

    loaded = repo.load(first.run_id)
    resumed = await ResearchEngine(MockResearchProvider(), repository=repo).run(loaded, rounds=1)
    assert resumed.current_round == 2
    assert resumed.run_id == first.run_id
    assert len(resumed.invocations) > len(first.invocations)
