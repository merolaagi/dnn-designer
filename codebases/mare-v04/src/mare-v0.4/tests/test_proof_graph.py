from mare.benchmarks import burgers_blowup_problem
from mare.models import Claim, ClaimStatus, DependencyEdge, ResearchState
from mare.proof_graph import ProofGraph


def test_proof_completeness_and_critical_dependencies():
    state = ResearchState(problem=burgers_blowup_problem())
    a = Claim(branch_id="B1", statement="A", claim_type="lemma", derivation="", created_by="x", round_no=1, status=ClaimStatus.VERIFIED)
    b = Claim(branch_id="B1", statement="B", claim_type="lemma", derivation="", created_by="x", round_no=1, status=ClaimStatus.PROPOSED)
    root = Claim(branch_id="B2", statement="T", claim_type="theorem", derivation="", created_by="x", round_no=1, status=ClaimStatus.VERIFIED, dependencies=[a.id, b.id])
    state.claims.extend([a, b, root])
    state.dependency_edges.extend([
        DependencyEdge(parent_claim_id=a.id, child_claim_id=root.id),
        DependencyEdge(parent_claim_id=b.id, child_claim_id=root.id),
    ])
    assert state.proof_completeness(root.id) == 2 / 3
    graph = ProofGraph(state)
    assert b.id in graph.ancestors(root.id)
    assert graph.is_dependency_closed(root.id) is False
    assert graph.critical_unresolved(root.id)[0][0] == b.id
