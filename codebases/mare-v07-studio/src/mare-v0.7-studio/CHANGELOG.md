# v0.7-studio — 2026-09-10

- Added authenticated paper/text import and source-linked structured analysis, with clearly labeled offline detection and optional OpenAI interpretation.
- Added a restricted arithmetic model specification, one to four coupled ODEs, RK4/refinement experiments and numerical failure reporting.
- Added checkpointed source/analysis/experiment stages through the existing worker queue.
- Added a learning dashboard, editable hypotheses and assumptions, source highlighting, forked alternatives, research-record export, popup animation and WebM export.
- Added a source-provenance bridge into MARE research; imported material is not automatically verified.
- Added PDF/parser/model/provenance/workflow tests; updated API types, payload limits, and documentation.
- Preserved the v0.6 graph planner. General paper-to-DNN design/training is not implemented by this increment.

# v0.6-planner — 2026-09-10

- Added optional sparse relational PyTorch graph scheduler with three prediction heads.
- Added problem-grouped offline training, missing-label masks, validation selection, tensor checkpoints and provenance.
- Added contextual and dependency benchmarks across three seeds, existing heuristic/TinyMLP comparisons, and no-edge ablations; bundled raw outcomes and checkpoints.
- Added affordable-action trace recording, counterexample rewards, normalized reserved-token labels, resumability, and frozen checkpoint pinning.
- Added worker-configured capture/shadow/blend modes. Synthetic checkpoints remain shadow-only; blend is capped at 25%.
- Added neural policy UI diagnostics, regenerated OpenAPI/TypeScript, optional CPU Compose overlay, guides, and tests.
- Research engine changes are limited to optional trace state and policy dependency injection; verification modules are unchanged.

# v0.5-web — 2026-09-10

- Preserved every v0.4 engine source file and original engine test; added a SHA-256 preservation manifest.
- Added FastAPI service, typed OpenAPI contracts, structured errors/logs, bearer authentication hook, CORS/host/body limits and rate-limiting hooks.
- Added SQLAlchemy/Alembic persistence with PostgreSQL and local SQLite paths, transactional snapshots/projections and durable event cursors.
- Added a separate local worker, fenced leases, completed-round checkpoints, cancellation, timeout termination, retry/backoff and resume.
- Added React/TypeScript workbench pages, real engine state, evidence inspection, charts, interactive proof DAG, memory-only workspace token entry and SSE/poll updates.
- Added deterministic seed mode, optional OpenAI adapter configuration, container images, reverse proxy, Compose, CI, tests and deployment documentation.
- Explicitly retained v0.4 verification and cold-start policy boundaries; no cross-run learning, multi-tenancy, billing ledger or live Lean web execution is claimed.

---

# Changelog

## 0.4.0

MARE v0.4 adds an actual neural component to the research-control plane.

### Added

- `TinyMLPRegressor`: NumPy `input -> ReLU -> sigmoid` neural utility model.
- `ResearchPolicy` with separate branch and question networks.
- Round-start policy snapshots and round-end outcome labels.
- Persistent `PolicyExample`, `PolicyNetworkState`, and `ResearchPolicyState` models.
- Branch features for novelty, uncertainty, compute, claim/evidence ratios, question value and graph depth.
- Question features for information gain, impact, difficulty, novelty, branch state, age and failure pressure.
- Cold-start thresholds and gradual/capped neural blending.
- Hybrid branch score retaining the UCB-inspired deterministic score.
- Learned question priority blended with the existing information-gain priority.
- `mare policy` CLI command for policy inspection.
- `--policy hybrid|heuristic` plus neural-policy tuning flags.
- Neural-policy telemetry in human-readable reports.
- Neural policy weights and examples survive SQLite save/resume.
- Six v0.4 tests; total suite: 27 passing.

### Trust invariant

The learned policy can prioritize research but cannot verify claims, bypass scope checks, close proof dependencies, or override Lean.

## 0.3.0

MARE v0.3 adds an explicit trust boundary around research claims.

### Added

- `ClaimScope` and `AssumptionFirewall` to block root candidates that add unauthorized assumptions.
- `AssumptionAudit` persistence and scope-audit evidence.
- Hard `ResearchBudget` limits for model calls, reserved output tokens, tool calls and rounds.
- Budget-aware exploration, review, verification planning, question generation and next-round scheduling.
- `SourceRecord` / `ClaimSourceLink` provenance models.
- `LiteratureProvider`, `InMemoryLiteratureProvider` and optional `OpenAlexLiteratureProvider`.
- Literature retrieval that records relevance without falsely promoting citations to proof evidence.
- `ToolInvocation` provenance.
- Proof-DAG cycle detection and dependencies-first proof order.
- Formalization-readiness check combining proof closure and scope audit.
- `LeanService`, `MockLeanService` and `SubprocessLeanService`.
- `mare formalize` CLI command for supplied Lean source.
- Docker-based Python execution boundary with network, capability, filesystem, CPU, memory, PID and timeout restrictions.
- Pluggable research-memory `SimilarityBackend`.
- Deterministic `HashEmbeddingBackend` for offline interface testing.
- Optional `PgVectorStore` plus `postgres` package extra.
- Budget/source/formalization details in human-readable reports.
- Updated LangGraph mapping to include question generation and literature-enrichment phases.
- 7 new v0.3 tests; total test suite: 21 passing.

### Changed

- OpenAI default model changed from `gpt-5.5` to configurable `gpt-5.6`.
- Synthesized theorem claims are explicitly root candidates.
- Rejected and scope-blocked claims are excluded from normal research-memory retrieval.
- Candidate theorem reporting now includes `formalization_ready`.

### Safety / trust behavior

- A closed dependency graph is no longer sufficient for theorem promotion; root scope must also pass audit.
- Literature retrieval never counts as proof verification by default.
- Generated Python has no implicit host-execution fallback when Docker is unavailable.
