# MARE v0.6-planner

A persistent, single-workspace research application around the MARE research engine, with an optional experimental graph scheduler. FastAPI + SQLAlchemy/Alembic + PostgreSQL (or SQLite) + a lease-based local worker + React 19/TypeScript/Vite. The interface includes investigations, branches, claims and evidence, failures, research questions, an interactive proof DAG, neural policy diagnostics, budgets, events, and settings.

This is a production-oriented deployment baseline, with tested local and container paths. It is not a multi-tenant SaaS or a claim that the research engine can prove arbitrary theorems. Read the operating boundaries below before exposing it publicly.


## New in v0.6 — graph research planner

This is a separate experimental release. v0.4 and v0.5 archives remain intact. The new `mare_planner` package provides a three-block relational neural network, offline training, two controlled synthetic benchmarks, shadow predictions, and checkpointed decision traces. Only `mare/models.py` (optional trace state) and `mare/engine.py` (policy injection) change in the original engine package; verification logic is retained.

Start with [the runnable planner guide](docs/GRAPH-PLANNER.md), [test and benchmark results](docs/PLANNER-RESULTS.md), and the included `artifacts/` directory. This model has synthetic training weights, not demonstrated mathematical research expertise. It stays in shadow mode by default.

Quick experiment, after creating/activating a Python 3.12 virtual environment:

```bash
pip install -c requirements.lock -e '.[dev,planner]'
python -m mare_planner.cli benchmark --task dependencies --out results/dependencies
python -m mare_planner.cli collect-demo --out demo.json
```

Default Docker runs the research application without PyTorch; see the planner guide for the optional CPU image. Base v0.5 Docker/PostgreSQL checks are historical; v0.6's planner image has not been built locally. Native v0.6 SQLite, worker, API, model training and frontend build checks passed.

## Run with Docker — recommended

Requires Docker Engine/Desktop with Compose v2. No external model key is needed.

```bash
unzip mare-v0.6-planner.zip
cd mare-v0.6-planner
cp .env.example .env
docker compose up --build -d
# Optional: queue the built-in four-round demo once
docker compose exec api python -m mare_web.seed
```

Open **http://localhost:8080**. Create a run, or open the seeded demo. The worker runs separately and commits results after each completed round. Startup runs migrations before the API and worker. The first image build downloads dependencies.

```bash
docker compose ps
docker compose logs -f api worker
curl http://localhost:8080/health/ready
docker compose stop                  # retain the PostgreSQL volume
```

`docker compose down` removes containers/network but retains data. **`docker compose down -v` deletes the database volume**; use only for disposable environments. The database and API have no published host ports; the web server binds only to loopback. `MARE_WEB_PORT` changes the web port. The worker is required for execution; starting only `db migrate api web` leaves jobs queued.

## Local development — SQLite

Tested with Python 3.12 and Node.js 22 on macOS; Linux is also supported. Use WSL2 or Docker on Windows because worker process-group cancellation is POSIX-specific.

From the extracted project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -c requirements.lock -e '.[dev]'
cp .env.example .env
alembic upgrade head
npm ci --prefix web
```

Start these in three terminals, each from the project root:

```bash
# Terminal 1
source .venv/bin/activate
uvicorn mare_web.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

```bash
# Terminal 2
source .venv/bin/activate
python -m mare_web.worker
```

```bash
# Terminal 3
npm run dev --prefix web
```

Open **http://localhost:5173**. Vite proxies `/api` and `/health` to port 8000. To queue the demo: `python -m mare_web.seed`. The default database is `mare-web.db` in the project root. Run the worker/API from the same root or use an absolute database URL.

The preserved CLI still works:

```bash
mare run --provider mock --rounds 2 --db legacy-demo.db
```

The CLI’s v0.4 SQLite format and the web database are different. Never point web migrations at a v0.4 database. No automatic import or destructive upgrade of v0.4 data is performed.

## What is implemented

- Async FastAPI endpoints; validated requests and output schemas; JSON errors with request IDs; structured service logs; liveness/readiness; explicit CORS and host allowlists; bounded request bodies; a replaceable rate-limit middleware; OpenAPI and a generated TypeScript contract.
- SQLAlchemy async persistence, an explicit Alembic migration, queryable branch/claim/evidence/failure/question/dependency projections, full typed research snapshots, and append-only progress events.
- Durable queue, atomic claims, renewable leases, stale-worker fencing, cancellation, round deadlines, bounded exponential retries, and recovery from the last completed round. Each research round executes in a child process, outside the API event loop.
- A responsive React interface with loading/error/empty states, filtered claims, review details, charts, keyboard-accessible proof nodes, a checkpoint JSON export, and memory-only bearer-token entry.
- Authenticated SSE with event IDs and replay; four-second polling fallback; Nginx buffering disabled for streaming.
- A deterministic Burgers demo that runs without OpenAI. The seed command queues actual engine execution rather than inserting fabricated outputs.
- Docker images/Compose, pinned tested dependency constraints and npm lockfile, CI, automated tests, screenshots, architecture and deployment notes.

The frontend uses React/Vite rather than Next.js: a same-origin static web client is sufficient for this workbench, and FastAPI owns all application APIs. No hosted Sites service is required to run this source distribution.

## Authentication and secrets

Local development allows unauthenticated access on loopback. Set a long `MARE_AUTH_TOKEN` and enter it on the web **Settings** page to enable workspace authentication. The token is not placed in URLs, browser storage, or logs. It clears when the page reloads.

For production, set `MARE_ENVIRONMENT=production`; startup fails unless the token has at least 32 characters and wildcard origins/hosts are absent. Use TLS and an identity-aware reverse proxy. The bearer token grants access to the entire workspace; it is not per-user authorization. See [deployment and security](docs/deployment.md) for the pre-deployment checklist, Docker secrets, proxy behavior, backups, and scaling limits.

`.env`, databases, downloaded dependencies, secrets, and generated test traces are excluded from the source archive. Do not put provider keys in the browser or a `VITE_*` variable.

## Optional OpenAI provider

For local use, first install `pip install -c requirements.lock -e '.[openai]'`. The Docker API image already includes the SDK, but the provider is disabled by default.

Configure the **API and worker** consistently:

```dotenv
MARE_OPENAI_ENABLED=true
MARE_OPENAI_MODEL=your-supported-structured-output-model
OPENAI_API_KEY=your-server-side-key
```

Restart the API/worker (or `docker compose up -d --force-recreate api worker`). Choose OpenAI when creating a run, and supply a canonical statement, assumptions, and success conditions. Configure provider-side spend limits first.

The web adapter selects the model explicitly, sets a provider-call timeout, and disables SDK-level retries; job retries are controlled by the worker. **No live OpenAI call was made during validation.** Available model names, account permissions, structured-output behavior, and live billing must be checked with your own account.

## Research and execution boundaries

- The mock provider supports only the built-in Burgers problem. Changing its display title does not change the mathematical target. Custom statements are rejected for mock runs.
- The v0.4 symbolic checker contains a small fixed library of Burgers checks. Unknown checks remain inconclusive. General OpenAI-proposed research is not automatically verified.
- “Completed” means the configured execution finished, not that a theorem is proved. Neural utility guides scheduling, never proof validity. Symbolic verification, assumption audits, and formalization remain distinct.
- The policy keeps the original cold-start thresholds (12 branch and 16 question examples). A short run may correctly show no trained network. We do not lower thresholds merely to make the UI look trained. Learning remains within a run; cross-run training is not implemented.
- Only **completed rounds** are durable. An interrupted round is replayed in full on resume. Events inside a round become visible when the checkpoint commits. Queued/running/cancelling transitions are persisted immediately.
- Exactly-once provider execution is not promised. A crash can lose uncommitted work after external API calls have been billed. Budget counters describe committed engine reservations, not a complete billing ledger or a hard dollar cap.
- The local worker abstraction is implemented. Celery/RQ/Arq adapters are extension points, not shipped integrations. PostgreSQL supports multiple local workers; SQLite is intended for one local worker.
- Generated code and Lean uploads are not exposed through the web API. The preserved CLI’s formalization and sandbox interfaces remain available for separately configured, trusted local use.

## Validate

```bash
source .venv/bin/activate
pytest -q
ruff check .
ruff format --check mare_web migrations tests/test_web.py
alembic check
python scripts/verify_engine.py
python scripts/export_openapi.py
npm run schemas --prefix web
npm run lint --prefix web
npm test --prefix web
npm run build --prefix web
npm audit --prefix web
```

For PostgreSQL integration tests, provide a **disposable database named `mare_test`**. The test fixture recreates its web tables:

```bash
MARE_TEST_DATABASE_URL=postgresql+asyncpg://mare:password@localhost:5432/mare_test pytest tests/test_web.py -q
```

Browser automation, with the API/worker/web servers already running:

```bash
cd web
npx playwright install chromium
npm run test:e2e
```

The Playwright CLI was blocked by the native Mac process sandbox in this authoring environment. Equivalent browser interactions were tested using the connected in-app browser; the report distinguishes those checks from the unexecuted CLI suite. GitHub Actions is configured to run the Playwright suite on Ubuntu, but a hosted Actions run was not executed here.

See [TEST_RESULTS.md](TEST_RESULTS.md) for exact verification and limitations, [architecture](docs/architecture.md) for lifecycle/concurrency details, and [CHANGELOG](CHANGELOG.md) for the version history. The original engine README and architecture are preserved under `docs/*v0.4.md`. `docs/engine-v0.4-manifest.json` records SHA-256 hashes of every preserved engine source file.
