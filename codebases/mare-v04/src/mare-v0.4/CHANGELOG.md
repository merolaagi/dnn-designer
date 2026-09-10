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
