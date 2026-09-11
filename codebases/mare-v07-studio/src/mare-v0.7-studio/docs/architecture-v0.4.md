# MARE v0.4 architecture

## Architectural invariant

Agents communicate through persistent scientific state, not a shared free-form transcript. The newly learned component controls **research allocation only**; it is outside the mathematical trust chain.

## Hybrid neural-symbolic control loop

```text
                     immutable ProblemSpec
                             |
                             v
                    multi-agent discovery
                             |
                             v
                 claims / evidence / failures
                             |
             +---------------+---------------+
             |                               |
             v                               v
     deterministic/UCB features       outcome snapshots
             |                               |
             |                         later round labels
             |                               |
             |                         Tiny MLP training
             |                               |
             +---------------+---------------+
                             |
                             v
                  Hybrid Research Policy
                             |
                  branch/question ranking
                             |
                             v
                         next round
```

## Neural model

V0.4 intentionally uses a tiny NumPy MLP instead of a heavyweight GPU framework:

`features -> Linear -> ReLU -> Linear -> Sigmoid`

The branch network consumes normalized features such as novelty, uncertainty, compute spent, verified/rejected ratios, evidence ratios, open-question value, and branch depth. The question network uses information gain, impact, difficulty, novelty, parent-branch state, age, and failure pressure.

The target is not “is this theorem true?”. The target is a bounded estimate of **research utility** observed after spending compute.

## Cold start and safety

A network is inactive until a configurable minimum number of labeled outcomes exists. When active, its influence ramps gradually to `max_neural_blend` (default 0.65). The deterministic UCB-like heuristic is always retained.

No neural prediction can:

- mark a claim verified;
- override an Assassin objection;
- bypass deterministic/tool evidence;
- bypass the Assumption Firewall;
- close a missing proof dependency;
- make Lean accept a proof.

## Trust pipeline

```text
claim proposed
    |
    v
independent adversarial review
    |
    v
deterministic/tool evidence
    |
    v
assumption firewall
    |
    +---- scope violation ----> scope_blocked
    |
    v
verified claim
    |
    v
proof dependency DAG
    |
    v
formalization gate -> Lean
```

The neural policy sits beside this pipeline and decides only **what to attempt next**.

## Online learning lifecycle

At round start, MARE creates `PolicyExample` snapshots for active branches and assigned questions. At round end, it labels branch examples from gained verified evidence / rejections / answered questions, and question examples from the status of claims that answered them. The network is retrained on accumulated labeled examples and serialized in `ResearchState`.

This is a first step toward inference-time scientific learning: model weights of the underlying LLM remain frozen, but the surrounding research policy improves from experience.

## Next target

V0.5 should add a learned policy across **multiple research runs**, hold-out evaluation to detect overfitting, exploration floors, and formalization-repair feedback:

```text
Lean failure -> error classifier -> missing lemma/question -> research -> proof repair -> Lean again
```
