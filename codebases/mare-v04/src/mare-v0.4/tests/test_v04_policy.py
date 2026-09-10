import numpy as np
import pytest

from mare.benchmarks import burgers_blowup_problem
from mare.engine import ResearchEngine
from mare.learned_policy import ResearchPolicy, TinyMLPRegressor
from mare.models import PolicyExample, PolicyExampleKind, ResearchState
from mare.navigator import Navigator
from mare.providers.mock_provider import MockResearchProvider
from mare.storage import SQLiteRepository


def test_tiny_neural_policy_learns_nonlinear_signal():
    # Repeated binary corners give the tiny MLP a clean, deterministic signal.
    x = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 20,
        dtype=float,
    )
    y = np.asarray([0.05, 0.10, 0.90, 0.95] * 20, dtype=float)
    net = TinyMLPRegressor(input_size=2, hidden_size=8, seed=11)
    loss = net.fit(x, y, epochs=700, learning_rate=0.05)
    assert loss < 0.03
    assert net.predict([1.0, 0.5]) > net.predict([0.0, 0.5]) + 0.45


def test_branch_network_trains_from_labeled_policy_examples():
    state = ResearchState(problem=burgers_blowup_problem())
    state.policy.min_branch_examples = 4
    policy = ResearchPolicy()
    # Synthetic historical outcomes are useful for validating serialization/training
    # without pretending they prove anything about mathematics.
    for i in range(8):
        good = i >= 4
        features = [
            0.8 if good else 0.2,
            0.4,
            0.2,
            0.4,
            0.8 if good else 0.1,
            0.0 if good else 0.7,
            0.7 if good else 0.1,
            0.0 if good else 0.6,
            0.3,
            0.7,
            0.1,
        ]
        state.policy.examples.append(
            PolicyExample(
                kind=PolicyExampleKind.BRANCH,
                round_no=i + 1,
                object_id=f"B-{i}",
                features=features,
                target=0.9 if good else 0.1,
                finalized_round=i + 1,
            )
        )
    policy.train(state)
    assert state.policy.branch_network is not None
    assert state.policy.branch_network.active
    assert state.policy.branch_network.trained_examples == 8
    assert state.policy.branch_network.loss is not None


@pytest.mark.asyncio
async def test_live_engine_activates_neural_branch_policy(tmp_path):
    state = ResearchState(problem=burgers_blowup_problem())
    state.policy.min_branch_examples = 2
    state.policy.blend_warmup_examples = 1
    state.policy.training_epochs = 80
    engine = ResearchEngine(
        MockResearchProvider(),
        repository=SQLiteRepository(str(tmp_path / "mare.db")),
        policy_enabled=True,
    )
    result = await engine.run(state, rounds=1)
    assert result.policy.branch_network is not None
    assert result.policy.branch_network.active
    assert any(b.policy_score is not None and b.policy_blend > 0 for b in result.branches)
    for branch in result.branches:
        if branch.policy_score is not None and branch.policy_blend > 0:
            expected = (1.0 - branch.policy_blend) * branch.heuristic_score + branch.policy_blend * branch.policy_score
            assert branch.score == pytest.approx(expected)


@pytest.mark.asyncio
async def test_question_policy_learns_from_answered_questions(tmp_path):
    state = ResearchState(problem=burgers_blowup_problem())
    state.policy.min_branch_examples = 2
    state.policy.min_question_examples = 1
    state.policy.blend_warmup_examples = 1
    state.policy.training_epochs = 60
    engine = ResearchEngine(
        MockResearchProvider(),
        repository=SQLiteRepository(str(tmp_path / "mare.db")),
        policy_enabled=True,
    )
    result = await engine.run(state, rounds=2)
    assert result.policy.question_network is not None
    assert result.policy.question_network.active
    assert result.policy.question_network.trained_examples >= 1


@pytest.mark.asyncio
async def test_policy_weights_persist_across_resume(tmp_path):
    db = str(tmp_path / "mare.db")
    repo = SQLiteRepository(db)
    state = ResearchState(problem=burgers_blowup_problem())
    state.policy.min_branch_examples = 2
    state.policy.blend_warmup_examples = 1
    state.policy.training_epochs = 40
    first = await ResearchEngine(MockResearchProvider(), repository=repo).run(state, rounds=1)
    assert first.policy.branch_network is not None
    first_weights = first.policy.branch_network.weights1

    loaded = repo.load(first.run_id)
    assert loaded.policy.branch_network is not None
    assert loaded.policy.branch_network.weights1 == first_weights
    resumed = await ResearchEngine(MockResearchProvider(), repository=repo).run(loaded, rounds=1)
    assert resumed.run_id == first.run_id
    assert resumed.policy.branch_network is not None
    assert resumed.policy.branch_network.trained_examples >= first.policy.branch_network.trained_examples


def test_heuristic_mode_keeps_neural_policy_out_of_scoring():
    state = ResearchState(problem=burgers_blowup_problem())
    state.policy.enabled = False
    # Navigator without a policy is the strict deterministic fallback.
    navigator = Navigator(policy=None)
    navigator.score(state)
    assert all(b.policy_score is None and b.policy_blend == 0 for b in state.branches)
