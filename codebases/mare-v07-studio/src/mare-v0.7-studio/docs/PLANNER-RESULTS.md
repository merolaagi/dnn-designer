# v0.6 test and benchmark results

Tested locally on macOS arm64, Python 3.12, PyTorch 2.14.0 CPU, Node 22.

## Verification

- Python: **59 passed**, including all 47 prior tests and 12 new planner/integration checks.
- Frontend: 3 unit tests passed; ESLint, TypeScript and production build passed.
- SQLite API + actual child-process worker: two rounds completed with capture, shadow predictions and persisted checkpoint hash.
- Model checks cover gradient flow, edge dependence, variable graph batching, permutation equivariance, problem splits, missing labels, reproducible counterexamples, budget skips, checkpoint round trips, changed-model fallback and malformed-model fallback.
- In-app browser: loaded the completed run and verified graph planner status, predictions, trace count, and unchanged proof/claim navigation.
- Compose base + planner overlay passed `docker compose config --quiet` (configuration validation only).
- New Linux Docker/CPU image, PostgreSQL planner execution, CUDA/MPS, live OpenAI and Lean, real research performance, and remote CI have not been tested. Prior v0.5 container results are historical, not evidence for these new integrations.

## Benchmark protocol

Two explicitly synthetic tasks, each with 60 problems and six candidates. Split by whole problem: 36 train, 12 validation, 12 test per seed. Seeds: 7, 19, 41. Each allocation spends exactly three unit-cost actions. Utility sum ranges from 0 to 3. No test-based model or seed selection. Graph and edge-removed models share widths/depth and training procedure. The original tiny MLP uses its existing 11-feature/12-hidden-unit architecture trained on these utility targets; this is an offline refit, not a complete replay of its online warmup/blending behavior. The oracle is an upper bound using synthetic labels.

## Context task

- heuristic: 1.4583 average held-out utility.

- tiny_mlp: 1.4722 average held-out utility.

- graph: 2.2222 average held-out utility.

- no_edges: 2.2153 average held-out utility.

- oracle: 2.2222 average held-out utility.


Per-seed graph utility: 7: 2.3333, 19: 2.1667, 41: 2.1667.

## Dependency task

- heuristic: 1.4375 average held-out utility.

- tiny_mlp: 1.3681 average held-out utility.

- graph: 1.7361 average held-out utility.

- no_edges: 1.4792 average held-out utility.

- oracle: 2.3056 average held-out utility.


Per-seed graph utility: 7: 1.3333, 19: 1.7500, 41: 2.1250.

## Interpretation

Context task: graph 2.2222 and no-edges 2.2153 are essentially tied. The improvement over the tiny MLP does not isolate a graph benefit; problem conditioning is sufficient here.

Dependency task: graph 1.7361 exceeds tiny MLP 1.3681, heuristic 1.4375, and no-edges 1.4792 on the three-seed mean. Seed 7 scores 1.3333, below its heuristic (1.3750); seed 41 drives much of the gain. There is material optimization/split variability. The small synthetic benchmark is not a confidence interval or evidence of real-world research superiority.

The dependency task permutes which claim's evidence controls a candidate's utility. Some synthetic dependency graphs contain cycles/self-dependencies; it tests relational scheduling, not proof-DAG validity. No claims about mathematical proof are made.

Raw records, splits, candidate-level allocation totals and all three checkpoints are included under each `artifacts/benchmark-*` directory. The deployment example intentionally uses seed 7 rather than selecting the best test seed. Demo question predictions come from a branch-only synthetic model and are unvalidated.

The appropriate next experiment is real-problem data collection and prospective budget-matched evaluation. No checkpoint is approved or automatically promoted for real research allocation.
