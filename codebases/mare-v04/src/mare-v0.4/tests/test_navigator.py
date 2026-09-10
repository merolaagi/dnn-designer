from mare.benchmarks import burgers_blowup_problem
from mare.models import Branch, Claim, ClaimStatus, ResearchState
from mare.navigator import Navigator


def test_navigator_prefers_verified_progress():
    state = ResearchState(problem=burgers_blowup_problem())
    good = Branch(title="good", research_question="q", strategy="s", compute_spent=1, novelty=0.5)
    empty = Branch(title="empty", research_question="q2", strategy="s2", compute_spent=1, novelty=0.5)
    state.branches.extend([good, empty])
    state.claims.append(Claim(
        branch_id=good.id,
        statement="useful",
        claim_type="lemma",
        derivation="d",
        created_by="x",
        round_no=1,
        status=ClaimStatus.VERIFIED,
    ))
    nav = Navigator()
    nav.score(state)
    assert good.score > empty.score
