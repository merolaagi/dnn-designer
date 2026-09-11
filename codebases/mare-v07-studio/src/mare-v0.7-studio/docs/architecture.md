# MARE v0.7 studio architecture

## Studio and planner extensions

v0.7 adds `POST /api/studio` and three checkpointed stages: source import, source-linked analysis/model proposal, and bounded numerical experiment. Studio state lives in `ResearchState.studio_data` inside the existing run snapshot; it uses the same queue, leases, child-process isolation, cancellation and event persistence. The runner dispatches studio stages separately from ordinary research rounds. No schema migration is needed for this JSON extension.

The browser reads persisted public trace events and calculated samples to render the research theatre. It does not expose model internals or reconstruct hidden reasoning. The source-to-research bridge creates an ordinary MARE run with an explicitly unverified source record. See [studio details](PAPER-STUDIO.md) and [graph planner](GRAPH-PLANNER.md) for the v0.6 optional learned scheduling component.

## Boundaries

```text
React + typed API client
    │ same-origin HTTP / bearer-authenticated SSE
Nginx static web / reverse proxy
    │
FastAPI ────────── PostgreSQL (production) / SQLite (development)
                       │ runs + leases + checkpoints + events
                  LocalJobQueue
                       │ atomic claim, renewable lease, fencing token
                  LocalExecutor
                       │ child process, one round, timeout/cancellation
                  MARE ResearchEngine / studio stage dispatcher
                       │
                  mock / optional OpenAI provider
```

The API is not a research worker. It validates requests, persists queued jobs, exposes committed research state, and performs conditional lifecycle mutations. The engine runs in a child process of a separate worker. No executable Python or Lean code is accepted by the HTTP service.

The research engine is retained, with the v0.6 planner integration and v0.7 studio snapshot field added; the current `mare/` directory is not byte-for-byte v0.4. `mare_web/runner.py` invokes `ResearchEngine.run(state, rounds=1)` with no v0.4 SQLite repository. When the child succeeds, `LocalExecutor` validates its `ResearchState` and commits it through the web store. This avoids publishing partial round snapshots that v0.4 emits during execution.

## Persistence

`runs` holds immutable problem configuration inside a typed JSON snapshot, requested round count, lifecycle, attempt/failure counters, timestamps, error code, lease expiration and token. Its `current_round` is the last committed round.

`branches`, `claims`, `evidence`, `failures`, `questions`, and `dependencies` are transactional projections with composite `(run_id, id)` primary keys, run foreign keys, and selected indexed columns. Domain payloads retain all v0.4 fields. Snapshot JSON remains the canonical recovery format; it also retains policy examples and weights, audits, source links, invocations and formalizations. Projections are rebuilt only for the checkpointed run, in the same transaction as the snapshot and event append. They allow future query-heavy endpoints without changing the engine model.

`run_events` has a global integer sequence, run foreign key, event kind, payload and timestamp. Read endpoints filter by run and sequence. Engine event IDs are retained inside payloads. Lifecycle transitions and checkpoints serialize writes through the run row, preserving event order within a run even on PostgreSQL. Event IDs need not be consecutive across different runs.

Alembic revision `0001_web` contains explicit schema operations. There is no `create_all` on API startup and no migration of old CLI databases. Readiness verifies the expected migration revision and the runs table. SQLite enables foreign keys, WAL and a busy timeout. PostgreSQL uses asyncpg; JSON columns are portable and may become JSONB in a future explicit migration.

## Lifecycle

```text
queued → running → completed
   │        ├── budget_exhausted
   │        ├── transient failure → queued (backoff) → running
   │        ├── retry limit → failed
   │        └── cancel → cancelling → cancelled
   └── cancel → cancelled

cancelled / failed → explicit resume → queued
```

Creation returns 201 after the queue insert is committed. A worker selects an available queued row, then conditionally updates it from `queued` to `running` with a unique lease token. Only one contender can win. A losing worker returns to polling. A worker handles one run at a time and multiple sequential rounds within that run.

The lease defaults to 30 seconds. The parent checks cancellation and renews the lease approximately every 0.5 seconds while a child runs. Checkpoints require the matching lease token, a live lease and a `running` status. A cancelled or expired worker cannot publish stale output. On lease expiration, recovery respects a concurrent cancellation and either requeues or fails after the configured consecutive-failure bound. A database outage can delay cancellation/recovery until persistence is reachable.

Default retry policy: 3 consecutive failed attempts between successful checkpoints; exponential delays of 2^failures seconds capped at 60 seconds. Successful checkpoints reset the consecutive-failure count, while total attempts remain recorded. Manual resume resets the failure count and retains the previous checkpoint. A terminal budget-exhausted run cannot be resumed under the same exhausted limits; create a new run with a larger budget.

Cancellation kills the POSIX child process group and discards its partial output. On clean worker shutdown, the child is terminated and the job is requeued (or cancelled if cancellation was requested). On unclean shutdown the lease expires. Default per-round timeout is 600 seconds. Workers run as non-root in Compose with a read-only filesystem, memory/PID limits and bounded temporary storage. A subprocess is an execution boundary, not a security sandbox for arbitrary code; no arbitrary-code tool is enabled in this web execution path.

## Resumability and accounting

Recovery rehydrates `ResearchState`, including learned policy weights and examples, and continues from `current_round + 1`. The v0.4 engine prepares pending work for the next round itself. Checkpointing is at-least-once at the round boundary: external calls from an interrupted round may be repeated and billed twice. Failed-round usage is not in committed counters. The UI explicitly labels output reservations and unavailable dollar usage. Provider-side quotas are required for financial limits.

The two neural networks are the original v0.4 NumPy implementations. Their cold-start thresholds, deterministic seed, learning rules and maximum blend are preserved. There is no cross-run model registry or training pipeline.

## API and browser

OpenAPI describes request and serialized output models. The generated `web/src/schema.d.ts` is checked in. `scripts/export_openapi.py` and `npm run schemas` regenerate the contract; CI checks for drift. Output schemas mark serialized defaults as present, reflecting the Pydantic response serialization contract.

Primary routes: `/api/runs`, `/api/runs/{id}`, `/cancel`, `/resume`, `/branches`, `/claims`, `/evidence`, `/failures`, `/questions`, `/graph`, `/events`, and `/stream`; `/api/settings`; `/health/live` and `/health/ready`. Collections have bounded limits and offsets/cursors. The full run detail intentionally returns a complete checkpoint; it is suitable for bounded research runs, not unbounded histories.

SSE accepts the bearer header plus `Last-Event-ID` or an `after` cursor. Frames contain persisted event IDs. Each connection lasts at most 45 seconds, uses short-lived database sessions, and emits heartbeats. It drains terminal events before a `settled` notification. The fetch-based client reconnects with its cursor; a four-second snapshot poll provides fallback. The live log buffer is bounded to 1,000 events and older history can be loaded explicitly.

The SPA uses React Router with a shared workbench shell, Recharts for checkpoint/budget-policy comparisons, and a topological SVG proof graph. Graph edges display prerequisite→dependent relationships; cycle detection prevents treating a cyclic graph as closed. Node selection is accessible by keyboard. Claims remain inspectable as text even when the graph is large. React text rendering escapes model-generated content; there is no HTML injection or arbitrary markdown renderer.

An optional, feature-detected WebMCP read tool lists current research runs through the same authenticated API client. It accepts only an empty object and has no mutation capability. It is not required for ordinary browser use.

## Extending the queue

`JobExecutor.execute(run_id, token)` is the execution contract. A Celery/RQ/Arq adapter should dispatch identifiers only, obtain or maintain a fenced lease, reuse the same checkpoint transaction, renew leases while processing, preserve cancellation semantics, and assume redelivery. Add transactional outbox delivery before atomically coupling an external broker to run creation. Do not move the engine into FastAPI `BackgroundTasks`; those tasks are process-local and not a durable queue.

## Scale and scope

This release is one trusted workspace with one bearer credential. An identity provider can replace the authentication hook, but multi-tenancy also requires owner/workspace columns, query scoping, authorization on events and exports, and audited migrations. Do not claim multi-tenant isolation by merely adding login.

For production use PostgreSQL, bounded concurrent workers, an identity/TLS reverse proxy, database backups, provider spending controls and operational monitoring. SQLite is for a single local worker. Large research histories need projection-based pagination, retention and snapshot compaction; the current complete-state API is intentionally optimized for bounded investigations.
