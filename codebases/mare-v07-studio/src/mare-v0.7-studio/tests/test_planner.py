# ruff: noqa: E402
import copy
import json
import pytest

torch = pytest.importorskip("torch")
from mare.benchmarks import burgers_blowup_problem
from mare.engine import ResearchEngine
from mare.models import Branch, ResearchState, ResearchTask, Claim, Evidence, EvidenceResult
from mare.providers.mock_provider import MockResearchProvider
from mare_planner.benchmark import synthetic_records, allocation
from mare_planner.features import graph_record
from mare_planner.network import GraphPlanner, collate, load_checkpoint, save_checkpoint
from mare_planner.policy import GraphResearchPolicy
from mare_planner.training import train, split_problems, validate


def test_variable_graph_batches_and_permutation():
    torch.manual_seed(1)
    r = synthetic_records(1)[0]
    s = copy.deepcopy(r)
    s["nodes"].append([0.0] * 48)
    model = GraphPlanner(16, 2).eval()
    combined = model(collate([r, s]))
    assert torch.allclose(combined[:6], model(collate([r])), atol=1e-6)
    n = len(r["nodes"])
    p = copy.deepcopy(r)
    p["nodes"] = list(reversed(r["nodes"]))
    p["edges"] = [[n - 1 - a, n - 1 - b, t] for a, b, t in r["edges"]]
    p["candidates"] = [n - 1 - i for i in r["candidates"]]
    assert torch.allclose(model(collate([r])), model(collate([p])), atol=1e-6)


def test_edges_and_gradients():
    r = synthetic_records(1)[0]
    model = GraphPlanner(16, 3)
    a = model(collate([r]))
    b = model(collate([dict(r, edges=[])]))
    assert not torch.allclose(a, b)
    a.sum().backward()
    assert model.project[0].weight.grad.abs().sum() > 0
    assert model.blocks[0].messages[2].weight.grad.abs().sum() > 0


def test_grouping_no_leakage():
    records = synthetic_records(12)
    records.append(copy.deepcopy(records[0]))
    groups = [{r["problem_id"] for r in p} for p in split_problems(records, 7)]
    assert not groups[0] & groups[1] and not groups[1] & groups[2] and not groups[0] & groups[2]
    with pytest.raises(ValueError):
        split_problems(records[:5], 7)


def test_features_no_ids_or_labels():
    state = ResearchState(
        problem=burgers_blowup_problem(), branches=[Branch(title="x", research_question="q", strategy="s")]
    )
    r = graph_record(state)
    state.run_id = "another"
    state.problem.id = "another"
    s = graph_record(state)
    assert r["nodes"] == s["nodes"] and r["context"] == s["context"] and r["problem_id"] == s["problem_id"]
    assert "targets" not in r


def test_training_checkpoint_and_fallback(tmp_path):
    model, metadata, _ = train(synthetic_records(12), epochs=4, width=16, depth=2)
    assert metadata["trained_examples"] > 0
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, metadata)
    loaded, _ = load_checkpoint(path)
    batch = collate(synthetic_records(1))
    assert torch.equal(model(batch), loaded(batch))
    policy = GraphResearchPolicy(checkpoint=path, mode="blend")
    assert policy.model is None and "shadow-only" in policy.status["reason"]
    assert GraphResearchPolicy(checkpoint=tmp_path / "missing").model is None


def test_partial_labels_and_counterexamples():
    state = ResearchState(problem=burgers_blowup_problem(), current_round=1)
    b = Branch(title="x", research_question="q", strategy="s")
    state.branches = [b, b.model_copy(update={"id": "other"})]
    state.tasks = [ResearchTask(branch_id=b.id, title="t", objective="o", round_no=1)]
    policy = GraphResearchPolicy()
    policy.begin_round(state)
    state = ResearchState.model_validate_json(state.model_dump_json())
    policy.begin_round(state)
    assert len(state.planner_data["traces"]) == 1
    c = Claim(
        branch_id=b.id, statement="false", claim_type="lemma", derivation="", created_by="test", round_no=1
    )
    state.claims.append(c)
    state.evidence.append(
        Evidence(
            claim_id=c.id,
            evidence_type="counterexample_search",
            result=EvidenceResult.CONTRADICTS,
            verifier="test",
            reproducible=True,
        )
    )
    policy.learn_round(state)
    trace = state.planner_data["traces"][0]
    assert trace["targets"][0][0] > 0 and trace["targets"][1] == [None, None, None]
    assert trace["complete"]


@pytest.mark.asyncio
async def test_engine_resume_shadow(tmp_path):
    state = ResearchState(problem=burgers_blowup_problem())
    await ResearchEngine(MockResearchProvider(), research_policy=GraphResearchPolicy()).run(state, rounds=1)
    assert state.planner_data["traces"][0]["complete"]
    restored = ResearchState.model_validate_json(state.model_dump_json())
    await ResearchEngine(MockResearchProvider(), research_policy=GraphResearchPolicy()).run(
        restored, rounds=1
    )
    assert [r["round"] for r in restored.planner_data["traces"]] == [1, 2]
    validate(restored.planner_data["traces"])
    path = tmp_path / "model.pt"
    save_checkpoint(path, GraphPlanner(16, 2), {"trained_examples": 1, "source": "synthetic"})
    before = json.dumps([c.model_dump(mode="json") for c in restored.claims])
    GraphResearchPolicy(checkpoint=path).score_questions(restored)
    assert before == json.dumps([c.model_dump(mode="json") for c in restored.claims])
    assert restored.planner_data["predictions"]


def test_equal_budget():
    r = synthetic_records(1)[0]
    for scores in (r["heuristic"], [0] * 6, list(range(6))):
        assert allocation(scores, r)[1] == 3


def test_invalid_data():
    r = synthetic_records(1)[0]
    r["nodes"][0][0] = float("nan")
    with pytest.raises(ValueError):
        validate([r])
    r = synthetic_records(1)[0]
    r["edges"].append([99999, 0, 0])
    with pytest.raises(ValueError):
        validate([r])


def test_checkpoint_pin_change_and_corrupt_fallback(tmp_path):
    state = ResearchState(
        problem=burgers_blowup_problem(), branches=[Branch(title="b", research_question="q", strategy="s")]
    )
    path = tmp_path / "model.pt"
    save_checkpoint(path, GraphPlanner(16, 2), {"trained_examples": 1, "source": "synthetic"})
    first = GraphResearchPolicy(checkpoint=path)
    assert first._predict(state)
    save_checkpoint(path, GraphPlanner(16, 2), {"trained_examples": 2, "source": "synthetic"})
    second = GraphResearchPolicy(checkpoint=path)
    assert second._predict(state) == {}
    assert "changed on resume" in state.planner_data["status"]["reason"]
    path.write_bytes(b"not a checkpoint")
    assert GraphResearchPolicy(checkpoint=path).model is None


def test_budget_skipped_tasks_are_not_labeled():
    state = ResearchState(problem=burgers_blowup_problem(), current_round=1)
    state.budget.max_model_calls = 1
    for i in range(2):
        b = Branch(id=f"b{i}", title="b", research_question="q", strategy="s")
        state.branches.append(b)
        state.tasks.append(ResearchTask(branch_id=b.id, title="t", objective="o", round_no=1))
    policy = GraphResearchPolicy()
    policy.begin_round(state)
    policy.learn_round(state)
    record = state.planner_data["traces"][0]
    assert record["selected"] == [True, False]
    assert record["targets"][1] == [None, None, None]
    assert record["targets"][0][1] == 0.5
