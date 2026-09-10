# Learned research policy

MARE v0.4 adds neural learning to the **control plane**, not the proof/truth plane.

## Branch network

Architecture:

```text
11 normalized branch features
        |
        v
Linear(input, hidden)
        |
       ReLU
        |
Linear(hidden, 1)
        |
      sigmoid
        |
 predicted research utility [0, 1]
```

Features include novelty, uncertainty, compute spent, claim volume, verified/rejected ratios, supporting/contradicting evidence, open-question value, and branch depth.

At the start of a round MARE snapshots the features. At the end it labels the snapshot based on what that unit of research produced. The label rewards new verified claims, supporting evidence, and answered questions, while penalizing newly rejected/scope-blocked claims.

## Question network

The question network uses information gain, impact, inverse difficulty, novelty, branch-spawn intent, parent branch score/progress/uncertainty, question age, and failure pressure.

An answered question is labeled `1.0` when it produces a verified/formalized claim, `0.0` when all its answers are rejected/scope-blocked, and an intermediate value for inconclusive resolution.

## Hybrid score

The rule-based score never disappears:

```text
final = (1 - blend) * heuristic + blend * neural
```

The neural blend is zero during cold start and ramps gradually, capped by `max_neural_blend`.

## Why a tiny NumPy network?

The point of v0.4 is to establish the architecture and learning loop without making CUDA/PyTorch a prerequisite. The network state is tiny enough to serialize directly inside `ResearchState`, so SQLite stop/resume includes the learned weights.

A future version can replace `TinyMLPRegressor` with PyTorch/JAX or a learned tree/graph policy without changing the surrounding policy contracts.

## Trust boundary

The neural policy cannot verify a claim. Verification remains downstream through adversarial review, deterministic/symbolic evidence, assumption auditing, proof-DAG closure, and Lean where configured.
