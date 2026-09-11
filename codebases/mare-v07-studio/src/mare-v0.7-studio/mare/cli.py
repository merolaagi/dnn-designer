from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from .benchmarks import burgers_blowup_problem
from .engine import ResearchEngine
from .formalization import SubprocessLeanService
from .literature import OpenAlexLiteratureProvider
from .models import ResearchBudget, ResearchState
from .providers.mock_provider import MockResearchProvider
from .reporting import text_report
from .storage import SQLiteRepository


def provider_from_name(name: str):
    if name == "mock":
        return MockResearchProvider()
    if name == "openai":
        from .providers.openai_provider import OpenAIProvider
        return OpenAIProvider()
    raise ValueError(f"Unknown provider: {name}")


def literature_from_name(name: str):
    if name == "none":
        return None
    if name == "openalex":
        return OpenAlexLiteratureProvider(mailto=os.getenv("MARE_OPENALEX_MAILTO"))
    raise ValueError(f"Unknown literature provider: {name}")


async def _run(args) -> None:
    provider = provider_from_name(args.provider)
    repo = SQLiteRepository(args.db)
    state = ResearchState(
        problem=burgers_blowup_problem(),
        budget=ResearchBudget(
            max_model_calls=args.max_model_calls,
            max_reserved_output_tokens=args.max_reserved_tokens,
            max_tool_calls=args.max_tool_calls,
            max_rounds=args.max_rounds,
        ),
    )
    state.policy.enabled = args.policy == "hybrid"
    state.policy.min_branch_examples = args.policy_min_branch_examples
    state.policy.min_question_examples = args.policy_min_question_examples
    state.policy.max_neural_blend = args.policy_max_blend
    state.policy.hidden_size = args.policy_hidden_size
    engine = ResearchEngine(
        provider,
        repository=repo,
        explorer_count=args.explorers,
        literature_provider=literature_from_name(args.literature),
        policy_enabled=state.policy.enabled,
    )
    result = await engine.run(state, rounds=args.rounds)
    print(text_report(result))


async def _resume(args) -> None:
    provider = provider_from_name(args.provider)
    repo = SQLiteRepository(args.db)
    state = repo.load(args.run_id)
    state.policy.enabled = args.policy == "hybrid"
    state.policy.min_branch_examples = args.policy_min_branch_examples
    state.policy.min_question_examples = args.policy_min_question_examples
    state.policy.max_neural_blend = args.policy_max_blend
    state.policy.hidden_size = args.policy_hidden_size
    engine = ResearchEngine(
        provider,
        repository=repo,
        explorer_count=args.explorers,
        literature_provider=literature_from_name(args.literature),
        policy_enabled=state.policy.enabled,
    )
    result = await engine.run(state, rounds=args.rounds)
    print(text_report(result))


def _show(args) -> None:
    repo = SQLiteRepository(args.db)
    state = repo.load(args.run_id)
    print(text_report(state))


def _formalize(args) -> None:
    repo = SQLiteRepository(args.db)
    state = repo.load(args.run_id)
    source = Path(args.lean_file).read_text(encoding="utf-8")
    service = SubprocessLeanService(args.lean_project)
    engine = ResearchEngine(MockResearchProvider(), repository=repo, lean_service=service)
    record = engine.formalize_claim(state, args.claim_id, source)
    print(f"Formalization {record.status.value}: {record.claim_id} via {record.prover}")
    if record.stdout:
        print(record.stdout)
    if record.stderr:
        print(record.stderr)



def _policy(args) -> None:
    repo = SQLiteRepository(args.db)
    state = repo.load(args.run_id)
    policy = state.policy
    branch_examples = sum(e.kind.value == "branch" and e.target is not None for e in policy.examples)
    question_examples = sum(e.kind.value == "question" and e.target is not None for e in policy.examples)
    print(f"MARE neural policy for {state.run_id}")
    print(f"enabled: {policy.enabled}")
    print(f"labeled branch examples: {branch_examples}")
    print(f"labeled question examples: {question_examples}")
    if policy.branch_network:
        print(
            f"branch network: active={policy.branch_network.active} "
            f"hidden={policy.branch_network.hidden_size} "
            f"trained={policy.branch_network.trained_examples} loss={policy.branch_network.loss}"
        )
    else:
        print(f"branch network: cold-start; threshold={policy.min_branch_examples}")
    if policy.question_network:
        print(
            f"question network: active={policy.question_network.active} "
            f"hidden={policy.question_network.hidden_size} "
            f"trained={policy.question_network.trained_examples} loss={policy.question_network.loss}"
        )
    else:
        print(f"question network: cold-start; threshold={policy.min_question_examples}")
    print("\nCurrent branch decisions:")
    for branch in sorted(state.branches, key=lambda b: b.score, reverse=True):
        neural = "cold" if branch.policy_score is None else f"{branch.policy_score:.3f}@{branch.policy_blend:.2f}"
        print(
            f"  {branch.title:24} final={branch.score:.3f} "
            f"heuristic={branch.heuristic_score:.3f} neural={neural}"
        )

def _provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["mock", "openai"], default=os.getenv("MARE_PROVIDER", "mock"))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--explorers", type=int, default=3)
    parser.add_argument("--db", default=os.getenv("MARE_DB_PATH", "mare.db"))
    parser.add_argument("--literature", choices=["none", "openalex"], default="none")
    parser.add_argument("--policy", choices=["hybrid", "heuristic"], default="hybrid")
    parser.add_argument("--policy-min-branch-examples", type=int, default=12)
    parser.add_argument("--policy-min-question-examples", type=int, default=16)
    parser.add_argument("--policy-max-blend", type=float, default=0.65)
    parser.add_argument("--policy-hidden-size", type=int, default=12)


def main() -> None:
    parser = argparse.ArgumentParser(prog="mare")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start a new benchmark research experiment")
    _provider_args(run)
    run.add_argument("--max-model-calls", type=int, default=250)
    run.add_argument("--max-reserved-tokens", type=int, default=1_000_000)
    run.add_argument("--max-tool-calls", type=int, default=2_000)
    run.add_argument("--max-rounds", type=int, default=100)

    resume = sub.add_parser("resume", help="continue a persisted research run for additional rounds")
    resume.add_argument("run_id")
    _provider_args(resume)

    show = sub.add_parser("show", help="render a persisted research run")
    show.add_argument("run_id")
    show.add_argument("--db", default=os.getenv("MARE_DB_PATH", "mare.db"))

    formalize = sub.add_parser("formalize", help="check a supplied Lean file for a persisted claim")
    formalize.add_argument("run_id")
    formalize.add_argument("claim_id")
    formalize.add_argument("--lean-file", required=True)
    formalize.add_argument("--lean-project", required=True)
    formalize.add_argument("--db", default=os.getenv("MARE_DB_PATH", "mare.db"))

    policy = sub.add_parser("policy", help="inspect the learned branch/question scheduling policy")
    policy.add_argument("run_id")
    policy.add_argument("--db", default=os.getenv("MARE_DB_PATH", "mare.db"))

    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(_run(args))
    elif args.command == "resume":
        asyncio.run(_resume(args))
    elif args.command == "show":
        _show(args)
    elif args.command == "formalize":
        _formalize(args)
    elif args.command == "policy":
        _policy(args)


if __name__ == "__main__":
    main()
