import argparse
import asyncio
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="MARE experimental graph planner")
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--out", default="planner-results")
    bench.add_argument("--epochs", type=int, default=40)
    bench.add_argument("--task", choices=["context", "dependencies"], default="dependencies")
    training = sub.add_parser("train")
    training.add_argument("dataset")
    training.add_argument("--out", default="planner.pt")
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--seed", type=int, default=7)
    export = sub.add_parser("export")
    export.add_argument("snapshot")
    export.add_argument("--out", default="traces.jsonl")
    demo = sub.add_parser("collect-demo")
    demo.add_argument("--out", default="demo-snapshot.json")
    demo.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.command == "benchmark":
        from .benchmark import run_benchmark

        report = run_benchmark(args.out, epochs=args.epochs, task=args.task)
        print(json.dumps(report["mean_utility"], indent=2))
    elif args.command == "train":
        from .training import write_training

        records = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
        metadata = write_training(records, args.out, epochs=args.epochs, seed=args.seed)
        Path(args.out + ".json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata["test_metrics"], indent=2))
    elif args.command == "export":
        packet = json.loads(Path(args.snapshot).read_text())
        state = packet.get("snapshot", packet)
        traces = [r for r in state.get("planner_data", {}).get("traces", []) if r.get("complete")]
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in traces))
        print(f"Exported {len(traces)} completed decision records")
    else:
        from mare.benchmarks import burgers_blowup_problem
        from mare.engine import ResearchEngine
        from mare.models import ResearchState
        from mare.providers.mock_provider import MockResearchProvider
        from .policy import GraphResearchPolicy

        state = ResearchState(problem=burgers_blowup_problem())
        engine = ResearchEngine(MockResearchProvider(), research_policy=GraphResearchPolicy())
        asyncio.run(engine.run(state, rounds=args.rounds))
        Path(args.out).write_text(state.model_dump_json(indent=2))
        print(f"Recorded {len(state.planner_data.get('traces', []))} mock rounds; no real training evidence")


if __name__ == "__main__":
    main()
