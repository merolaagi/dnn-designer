from mare.benchmarks import burgers_blowup_problem
from mare.memory import ResearchMemory, lexical_similarity
from mare.models import Branch, Claim, ClaimStatus, Failure, ResearchState, ResearchTask


def test_lexical_similarity_prefers_overlap():
    assert lexical_similarity("gradient blowup", "gradient singularity blowup") > lexical_similarity(
        "gradient blowup", "unrelated topology"
    )


def test_memory_retrieves_same_branch_failure_and_claim():
    state = ResearchState(problem=burgers_blowup_problem())
    branch = Branch(title="Energy", research_question="Can L2 energy explain blow-up?", strategy="energy")
    state.branches.append(branch)
    claim = Claim(
        branch_id=branch.id,
        statement="L2 energy controls amplitude",
        claim_type="lemma",
        derivation="d",
        created_by="x",
        round_no=1,
        status=ClaimStatus.VERIFIED,
        tags=["energy"],
    )
    state.claims.append(claim)
    failure = Failure(
        branch_id=branch.id,
        mechanism="bad inference",
        failure_reason="L2 does not control spatial gradient",
        reusable_constraint="Do not infer derivative control from L2 control",
    )
    state.failures.append(failure)
    task = ResearchTask(branch_id=branch.id, title="gradient vs energy", objective="reconcile L2 and gradient")
    result = ResearchMemory().retrieve(state, task)
    assert claim in result.claims
    assert failure in result.failures
