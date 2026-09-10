# MARE v0.4 — Mini-Astra Research Engine

MARE is an evidence-driven mathematical research engine. LLMs are replaceable workers inside the loop; the system of record is a persistent scientific state containing research branches, questions, atomic claims, failures, evidence, proof dependencies, scope audits, sources, budgets, formalization attempts, and provenance.

V0.4 upgrades MARE into a **hybrid neural-symbolic research system**. The neural component learns *research allocation* from prior branch/question outcomes; it never decides whether mathematics is true. The existing trust pipeline remains authoritative:

`ProblemSpec -> Director -> Explorers -> Assassins -> Tool Verification -> Assumption Firewall -> Evidence Ledger -> Proof DAG -> Neural + UCB Research Policy -> Next Round -> Formalization Gate`

The repository ships with a deterministic inviscid-Burgers research fixture, so the full architecture runs without an API key.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
mare run --provider mock --rounds 2 --db mare.db
```

Resume the same research state later:

```bash
mare resume RUN-xxxxxxxxxx --provider mock --rounds 2 --db mare.db
mare show RUN-xxxxxxxxxx --db mare.db
```

Use an OpenAI model:

```bash
export OPENAI_API_KEY="..."
export MARE_OPENAI_MODEL="gpt-5.6"
mare run --provider openai --rounds 2
```

The provider layer remains model-agnostic. Local or third-party models only need to implement `ModelProvider`.

## What changed in v0.4

### 1. A real neural research-policy model

`mare/learned_policy.py` implements a small NumPy neural network:

```text
research-state features -> ReLU hidden layer -> sigmoid research utility
```

There are two learned policies:

- a **branch policy** that predicts the value of spending the next unit of compute on a research branch;
- a **question policy** that predicts whether an assigned research question is likely to produce a verified claim.

The targets come from MARE's own later outcomes. At the beginning of a round MARE snapshots features. At the end of that round it labels the snapshot from verified/rejected claims, evidence, and answered questions. This is online inference-time policy learning, not retraining the underlying LLM weights.

### 2. Hybrid, guarded scheduling

The neural policy has a cold-start threshold. Before enough labeled outcomes exist, MARE remains entirely heuristic/UCB driven. Once the threshold is crossed:

```text
final_score = (1 - blend) * heuristic_score + blend * neural_score
```

The blend ramps up gradually and is capped (default `0.65`). A new network therefore cannot immediately take control of the research program.

### 3. Learned state is persistent

Network weights, losses, labeled policy examples, thresholds, and blend configuration are stored inside `ResearchState`, so stop/resume preserves the learned policy. No separate training server is required.

Inspect a persisted policy with:

```bash
mare policy RUN-xxxxxxxxxx --db mare.db
```

### 4. Policy controls research, never truth

A neural score can move a branch up or down the queue, but it cannot promote a claim. Claim promotion still requires adversarial review, evidence, scope audit, proof closure, and formal verification where configured.

### 5. Run the hybrid policy

```bash
mare run \
  --provider mock \
  --rounds 4 \
  --policy hybrid \
  --policy-min-branch-examples 6 \
  --policy-min-question-examples 4 \
  --db mare.db
```

For strict v0.3 behavior:

```bash
mare run --provider mock --rounds 2 --policy heuristic
```

## v0.3 trust-system capabilities retained

### 1. Assumption Firewall

Every claim can declare a scope:

- `local_lemma`
- `restricted_result`
- `root_candidate`

A root candidate cannot silently add assumptions absent from the immutable `ProblemSpec`. For example, a theorem claiming to solve a general problem while adding `Assume axisymmetry` is marked `scope_blocked` unless axisymmetry is part of the original problem scope.

A closed proof DAG is therefore **necessary but no longer sufficient** for promotion.

### 2. Hard research budgets

`ResearchBudget` limits:

- model calls;
- reserved output tokens;
- tool calls;
- research rounds.

The scheduler does not keep creating work after the run can no longer afford it. Reviews, verification planning, literature searches, and formalization also respect the same persisted budget ledger.

Example:

```bash
mare run \
  --provider mock \
  --rounds 10 \
  --max-model-calls 50 \
  --max-reserved-tokens 180000 \
  --max-tool-calls 100
```

### 3. Literature provenance boundary

MARE now has a `LiteratureProvider` protocol plus:

- `InMemoryLiteratureProvider` for tests/offline fixtures;
- `OpenAlexLiteratureProvider` for optional public scholarly discovery.

Retrieved papers are attached to claims as `relevant` source links. **Finding a paper does not automatically verify a mathematical claim.** This avoids the common failure of treating citation retrieval as proof validation.

Optional CLI use:

```bash
mare run --provider mock --rounds 2 --literature openalex
```

### 4. Stronger proof graph

`ProofGraph` now supports:

- ancestor/descendant traversal;
- dependency closure;
- proof completeness;
- critical unresolved claims;
- cycle detection;
- dependencies-first proof order;
- formalization readiness.

A cyclic proof graph is rejected as a valid proof plan.

### 5. Formalization service boundary

`formalization.py` defines a `LeanService` interface with:

- `MockLeanService` for architecture tests;
- `SubprocessLeanService` for a real Lean project.

A claim can be submitted to Lean only when:

1. it passes the Assumption Firewall;
2. its proof dependency graph is verified and closed;
3. the run still has tool budget.

Example:

```bash
mare formalize RUN-xxxxxxxxxx C-xxxxxxxxxx \
  --lean-file candidate.lean \
  --lean-project /path/to/lean/project \
  --db mare.db
```

The real service executes Lean using the configured Lake environment and records stdout, stderr, elapsed time, status, and provenance.

### 6. Safer generated-code boundary

`DockerPythonSandbox` never `eval`/`exec`s generated source in the host MARE process. It builds a disposable Docker execution command with:

- `--network none`
- read-only filesystem
- all Linux capabilities dropped
- `no-new-privileges`
- memory, CPU, PID and timeout limits
- small no-exec tmpfs

Docker is the security boundary. If Docker is unavailable, the secure runner refuses to execute rather than silently falling back to host execution.

### 7. Pluggable research-memory similarity

`ResearchMemory` now takes a `SimilarityBackend`.

Included backends:

- lexical cosine overlap — default and deterministic;
- hash-vector backend — offline interface test, **not** a learned semantic model.

`PgVectorStore` provides optional PostgreSQL/pgvector persistence. Install the extra with:

```bash
pip install -e ".[postgres]"
docker compose up -d postgres
```

The pgvector layer is intentionally optional; SQLite remains the zero-setup state store.

### 8. Current OpenAI provider default

The OpenAI provider defaults to `gpt-5.6` but is configurable through `MARE_OPENAI_MODEL`. MARE persists only structured research packets and invocation provenance; it does not depend on hidden chain-of-thought.

## Core architecture

```text
                         Immutable ProblemSpec
                                 |
                                 v
                         Research Director
                                 |
                   +-------------+-------------+
                   |             |             |
                   v             v             v
               Explorer 1    Explorer 2    Explorer 3
                   |             |             |
                   +-------------+-------------+
                                 |
                            atomic claims
                                 |
                        +--------+--------+
                        |                 |
                        v                 v
                   Assassin A       Assassin B
                        |                 |
                        +--------+--------+
                                 |
                                 v
                         deterministic checks
                                 |
                                 v
                         Assumption Firewall
                                 |
                       +---------+---------+
                       |                   |
                       v                   v
                  Evidence Ledger      Scope block
                       |
          +------------+-------------+
          |            |             |
          v            v             v
      Literature    Proof DAG     Failure Memory
      provenance        |             |
          |             |             |
          +-------------+-------------+
                        |
                        v
                    Synthesis
                        |
                        v
                 Question Generator
                        |
               information gain
                        |
                        v
              Hybrid Research Policy
              /                  \
        UCB heuristic        Tiny neural nets
              \                  /
               +--------+---------+
                        |
                        v
                    Navigator
                        |
             score / spawn / kill
                        |
                        +-------------> next round
                        |
                closed root candidate
                        |
                        v
                 Formalization Gate
                        |
                        v
                     Lean 4
```

## Scientific state

`ResearchState` stores:

- immutable problem specification;
- persisted research budget;
- outcome-labeled neural-policy examples and network weights;
- branches and lifecycle status;
- scored research questions;
- tasks;
- atomic claims and declared scope;
- explicit dependency edges;
- objections and reusable failures;
- deterministic evidence;
- source records and claim/source links;
- assumption audits;
- formalization attempts;
- tool invocation provenance;
- model invocation provenance;
- append-only research events;
- synthesis notes.

The central rule remains: **agreement is not verification**.

## Burgers benchmark

The deterministic benchmark explores three competing mechanisms:

1. characteristic-map loss of invertibility;
2. Riccati evolution of `u_x`;
3. an intentionally false L2-energy explanation.

The false energy branch is attacked, contradicted by a deterministic conservation check, terminated, and preserved as reusable negative knowledge. A new question then asks how bounded L2 energy can coexist with large gradients, producing a separate reconciliation branch.

The synthesized root theorem depends on the independently verified characteristic and gradient claims. In v0.3 it additionally passes the scope firewall before being considered ready for formalization.

## Repository map

```text
mare/
├── agents.py                 # Director, Explorer, Assassin, Synthesizer, QuestionGenerator
├── assumption_firewall.py    # immutable-scope auditing
├── benchmarks.py             # Burgers benchmark
├── budget.py                 # hard model/tool/round budgets
├── cli.py                    # run / resume / show / formalize
├── contracts.py              # structured LLM output contracts
├── engine.py                 # reference async research loop
├── formalization.py          # Lean service boundary
├── langgraph_engine.py       # optional StateGraph mapping
├── literature.py             # scholarly discovery providers
├── memory.py                 # dependency-aware context retrieval
├── memory_backends.py        # pluggable similarity backends
├── models.py                 # persistent scientific state models
├── navigator.py              # branch scoring/lifecycle
├── pgvector_store.py         # optional vector persistence
├── proof_graph.py            # DAG validation/readiness
├── reporting.py              # human-readable reports
├── storage.py                # SQLite snapshots + event log
├── verification.py           # deterministic symbolic checks
├── tools/
│   └── sandbox.py            # Docker Python execution boundary
└── providers/
    ├── base.py
    ├── mock_provider.py
    ├── openai_provider.py
    └── tracing.py
```

## Tests

```bash
pytest -q
```

Current release result:

```text
21 passed
```

The tests cover the original research loop plus v0.3 scope blocking, cyclic-proof rejection, source provenance, budget stopping, Docker security flags, Lean promotion, and pluggable memory similarity.

## Important limitations

MARE v0.3 is still an experimental research-engine skeleton, not a proof of autonomous mathematical discovery. In particular:

- the Burgers mock provider contains deterministic fixtures;
- literature retrieval discovers candidate prior art but does not semantically prove support;
- the hash-vector backend is not a learned embedding model;
- pgvector is optional and not required by the main benchmark;
- Docker hardening materially reduces risk but should still be reviewed before running hostile code on shared infrastructure;
- a real Lean formalization must be supplied/generated separately and checked in a properly maintained Lean project;
- the system does not yet run a hidden-proof benchmark or autonomous PDE research over hundreds of branches.

## Next target: v0.4

The next capability jump should focus on **autonomous proof repair and research evaluation**:

- formalization agent that translates proof-DAG nodes into Lean candidates;
- Lean error parser -> repair questions -> retry loop;
- actual learned embeddings with provider abstraction;
- literature claim extraction and independent citation review;
- branch-level budgets and information-gain-per-cost accounting;
- hidden-proof benchmark harness with leakage controls;
- model-role diversity (strong director, cheap explorers, independent critics);
- experiment registry for numerical PDE work;
- first viscous-Burgers benchmark, then 2D Euler / 2D Navier-Stokes stepping stones.
