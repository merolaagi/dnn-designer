import pytest

from mare.benchmarks import burgers_blowup_problem
from mare.engine import ResearchEngine
from mare.models import ClaimStatus, ResearchState
from mare.providers.mock_provider import MockResearchProvider
from mare.storage import SQLiteRepository


@pytest.mark.asyncio
async def test_mock_research_loop_builds_and_falsifies(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "mare.db"))
    state = ResearchState(problem=burgers_blowup_problem())
    engine = ResearchEngine(MockResearchProvider(), repository=repo)
    result = await engine.run(state, rounds=1)

    assert len(result.branches) >= 4  # 3 director branches + synthesis
    assert any(c.status == ClaimStatus.REJECTED for c in result.claims)
    assert any(c.status == ClaimStatus.VERIFIED for c in result.claims)
    assert result.failures
    assert any("L2" in (f.reusable_constraint or "") for f in result.failures)
    assert any("min_x u0'" in c.statement for c in result.claims)

    reloaded = repo.load(result.run_id)
    assert reloaded.run_id == result.run_id
    assert len(reloaded.claims) == len(result.claims)


@pytest.mark.asyncio
async def test_two_rounds_preserve_research_state(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "mare.db"))
    state = ResearchState(problem=burgers_blowup_problem())
    engine = ResearchEngine(MockResearchProvider(), repository=repo)
    result = await engine.run(state, rounds=2)

    assert result.current_round == 2
    assert len(result.events) > 0
    assert any(c.created_by == "synthesizer" and c.status == ClaimStatus.VERIFIED for c in result.claims)
    assert all(0.0 <= b.score <= 1.0 for b in result.branches)
    statements = [" ".join(c.statement.lower().split()) for c in result.claims]
    assert len(statements) == len(set(statements))
    failed_claim_ids = [f.failed_claim_id for f in result.failures]
    assert len(failed_claim_ids) == len(set(failed_claim_ids))
