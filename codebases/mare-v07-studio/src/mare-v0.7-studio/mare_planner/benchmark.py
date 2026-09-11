"""Synthetic contextual scheduling benchmark, not evidence of research ability."""

import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch

from mare.models import ResearchState, ProblemSpec, Branch, Claim, Evidence, EvidenceResult
from mare.navigator import Navigator
from .features import graph_record
from .network import collate, save_checkpoint
from .training import train, evaluate, fit_tiny


def synthetic_records(count=60, seed=101, dependencies=False):
    rng = random.Random(seed)
    records = []
    for problem in range(count):
        # IDs group distinct problem instances; goal words are actual context.
        counter = bool(problem % 2)
        state = ResearchState(
            problem=ProblemSpec(
                title=f"Synthetic instance {problem}",
                canonical_statement=("seek counterexamples" if counter else "seek supporting mechanisms")
                + f" system {problem}",
            )
        )
        targets = []
        costs = []
        for i in range(6):
            b = Branch(
                id=f"b{i}",
                title="Candidate mechanism",
                research_question="Test mechanism",
                strategy="experiment",
                novelty=rng.random(),
                compute_spent=rng.random() * 3,
            )
            state.branches.append(b)
            c = Claim(
                id=f"c{i}",
                branch_id=b.id,
                statement="Mechanism observation",
                claim_type="lemma",
                derivation="",
                created_by="synthetic",
                round_no=0,
            )
            state.claims.append(c)
            supports = rng.randrange(5)
            for j in range(4):
                state.evidence.append(
                    Evidence(
                        id=f"e{i}_{j}",
                        claim_id=c.id,
                        evidence_type="numerical",
                        result=EvidenceResult.SUPPORTS if j < supports else EvidenceResult.CONTRADICTS,
                        verifier="synthetic oracle",
                        details="Synthetic measurement",
                    )
                )
            utility = (4 - supports if counter else supports) / 4
            # Fixed per-action cost makes every method use precisely the same budget.
            targets.append([utility, 0.5, float(utility >= 0.75)])
            costs.append(1)
        if dependencies:
            permutation = list(range(6))
            rng.shuffle(permutation)
            for i, j in enumerate(permutation):
                state.claims[i].dependencies = [state.claims[j].id]
            targets = [targets[j] for j in permutation]
            state.problem.canonical_statement += " evaluate prerequisite evidence"
        Navigator().score(state)
        r = graph_record(state)
        r.update(
            source="synthetic",
            targets=targets,
            actual_costs=costs,
            selected=[True] * 6,
            inclusion_probability=[1.0] * 6,
            selection="synthetic full-information oracle",
            complete=True,
        )
        records.append(r)
    return records


def allocation(scores, record, budget=3):
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    utility = 0
    spent = 0
    for i in order:
        cost = record["actual_costs"][i]
        if spent + cost <= budget:
            spent += cost
            utility += record["targets"][i][0]
    return utility, spent


def run_benchmark(output, seeds=(7, 19, 41), epochs=40, task="dependencies"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = synthetic_records(dependencies=task == "dependencies")
    (output / "synthetic.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    results = []
    for seed in seeds:
        model, meta, (training, validation, test) = train(records, seed=seed, epochs=epochs)
        tiny = fit_tiny(training, seed)
        no_edges = [dict(r, edges=[]) for r in records]
        ablation, _, _ = train(no_edges, seed=seed, epochs=epochs)
        rows = []
        for r in test:
            with torch.inference_mode():
                pred = model(collate([r])).numpy()
                flat = ablation(collate([dict(r, edges=[])])).numpy()
            scores = dict(
                heuristic=r["heuristic"],
                tiny_mlp=tiny.predict_array(np.asarray(r["baseline"])),
                graph=pred[:, 0] / (1 + pred[:, 1]),
                no_edges=flat[:, 0] / (1 + flat[:, 1]),
                oracle=[t[0] for t in r["targets"]],
            )
            rows.append(
                dict(
                    problem_id=r["problem_id"],
                    **{name: dict(zip(("utility", "spent"), allocation(s, r))) for name, s in scores.items()},
                )
            )
        summary = {
            name: statistics.mean(row[name]["utility"] for row in rows)
            for name in ("heuristic", "tiny_mlp", "graph", "no_edges", "oracle")
        }
        results.append(
            dict(
                seed=seed,
                mean_utility=summary,
                test_metrics=evaluate(model, test),
                problems=rows,
                split=meta["split"],
                best_epoch=meta["best_epoch"],
            )
        )
        save_checkpoint(output / f"planner-seed-{seed}.pt", model, meta)
        print(json.dumps(dict(seed=seed, mean_utility=summary)), flush=True)
    report = dict(
        task=task,
        disclaimer="Synthetic planted contextual task. Does not establish real research improvement or mathematical correctness.",
        budget_per_problem=3,
        candidates_per_problem=6,
        seeds=results,
        mean_utility={
            name: statistics.mean(r["mean_utility"][name] for r in results)
            for name in results[0]["mean_utility"]
        },
    )
    (output / "results.json").write_text(json.dumps(report, indent=2))
    return report
