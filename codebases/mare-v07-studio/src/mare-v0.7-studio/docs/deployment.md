# Deployment and security

## Supported deployment shape

Deploy the static web image, FastAPI image, PostgreSQL and one or more separate worker replicas. Run the migration job once before rolling out API/worker processes. Use PostgreSQL in production; SQLite is a local development convenience. Keep API and worker settings aligned, especially provider enablement and model selection.

Compose binds `127.0.0.1:8080` by default. Place a TLS-terminating proxy on that host in front of it, or attach your existing private proxy network. Configure explicit origins and hostnames. The bundled Nginx forwards the Host header as `api`; the API host allowlist therefore must retain `api`. Never expose the unauthenticated development configuration publicly.

Nginx disables buffering and caching for `/api/`, uses a 65-second read timeout for the bounded SSE connections, limits request body size, and adds CSP, frame protection, referrer and MIME-sniffing headers. Its rate limiter covers the same-origin API. If a front proxy is added, configure trusted client-IP handling for that proxy's exact addresses; do not trust arbitrary forwarded headers. The application rate limiter is per-process and bounded in memory; use a Redis-backed limiter or gateway quotas for multiple API replicas and stronger admission control.

A production reverse proxy should also limit concurrent connections, unauthenticated failures, open SSE streams and queue admission. The local rate limiter is a hook, not distributed abuse protection. Add database/provider quotas and monitor queued age before increasing concurrency.

## Secrets

For local development, `.env` is read by `pydantic-settings`. It is excluded from Git, Docker build context and the release package. The browser never receives provider keys or the database URL. The browser bearer token stays only in JavaScript memory and is cleared on reload.

For deployment, prefer your platform secret manager or Docker secrets over plaintext environment files. The image entrypoint supports `MARE_AUTH_TOKEN_FILE`, `OPENAI_API_KEY_FILE`, and `MARE_DATABASE_URL_FILE`; their contents populate the corresponding environment variables before the service starts. For example, an additional Compose override can mount a token:

```yaml
services:
  api:
    environment:
      MARE_AUTH_TOKEN_FILE: /run/secrets/mare_auth_token
    secrets: [mare_auth_token]
  worker:
    environment:
      MARE_AUTH_TOKEN_FILE: /run/secrets/mare_auth_token
    secrets: [mare_auth_token]
  migrate:
    environment:
      MARE_AUTH_TOKEN_FILE: /run/secrets/mare_auth_token
    secrets: [mare_auth_token]
secrets:
  mare_auth_token:
    file: ./secrets/mare_auth_token
```

Provision the file through your secret manager. Keep `secrets/` out of Git, restrict permissions, and rotate credentials. The same file convention is available for the provider key and database URL. Do not use frontend build-time variables for secrets. Environment values can be visible to Docker administrators; use platform-level access controls.

The default database password is a local-demo value. Replace it before production initialization. Changing `POSTGRES_PASSWORD` does not automatically rotate an already initialized PostgreSQL volume's password; rotate the database role deliberately and update API/worker credentials together.

The API has no cookie authentication, so bearer requests do not rely on ambient browser cookies. If replacing it with cookies, add CSRF protection and appropriate SameSite/Secure/HttpOnly attributes. Production fails closed for missing/short workspace tokens and wildcard origins/hosts, but these checks do not implement a full identity provider.

## Pre-deployment checklist

- [ ] Set `MARE_ENVIRONMENT=production` and a securely generated token of at least 32 characters.
- [ ] Put TLS and an identity-aware access boundary in front of the service; keep DB/API private.
- [ ] Replace demo database credentials and set explicit host/origin allowlists.
- [ ] Confirm everyone receiving the bearer token is authorized for the entire workspace. Multi-tenant authorization is not implemented.
- [ ] Enable OpenAI only when needed, select a supported model explicitly, and set provider-side spend limits.
- [ ] Review the inherited engine's verification limitations for the intended research domain.
- [ ] Keep arbitrary generated-code/Lean execution disabled in the web path. Do not mount a Docker socket into the worker.
- [ ] Apply database migrations, test backups/restores, and monitor disk/temporary-storage usage.
- [ ] Set worker CPU/memory/PID limits and per-round timeout; use PostgreSQL for multiple workers.
- [ ] Monitor readiness failures, worker exits, queued age, expired leases, repeated retries and sustained 429/5xx responses.
- [ ] Re-scan locked dependencies and pin image digests in your release process. This archive's scans are point-in-time results, not a permanent guarantee.
- [ ] Run the browser suite and validate your actual proxy, identity provider and provider-account configuration.

## Operations

Service logs are JSON records with a server-generated request ID, status and timing. Request bodies, query strings, authorization headers and provider exception messages are excluded. Worker errors record a stable error code and stack frame locations without secret-bearing exception text. Persisted engine event data can contain research content and should be treated as workspace-confidential. Configure log rotation and retention outside the process.

`/health/live` is process liveness. `/health/ready` checks DB connectivity and the expected Alembic revision; it does not certify that a worker is running. Supervise workers and monitor queue age separately. A deployment can have a healthy API with all jobs queued because no worker is available.

Default retry delays are 2^failure-count seconds capped at 60. The default maximum is three consecutive failures, reset by a successful checkpoint. Crashed leases expire after 30 seconds. Graceful shutdown kills the active child and requeues from the previous checkpoint; hard shutdown recovery relies on the lease. Active provider calls may already have incurred charges.

The API can run with multiple processes behind a proxy because state is in the database. The in-memory rate limiter is not shared. Workers can be scaled with `docker compose up -d --scale worker=2` on PostgreSQL after verifying provider budgets and workload. There is no external broker adapter in this release.

## Backups and migration rollback

Back up PostgreSQL using your managed service or `pg_dump`; protect snapshots as research data. Test restoration to a separate database. For local SQLite, stop API/worker before copying the database and its WAL files, or use SQLite's online backup API. Never copy only a live main DB file while ignoring a nonempty WAL.

Alembic supports `upgrade head`, `current`, `check`, and `downgrade base`. **The initial migration's downgrade drops all web tables and their data.** Use it only on disposable databases. Production rollback should restore a tested backup or use a reviewed forward migration, not casually drop the schema.

v0.4 files and databases remain separate. A future importer should validate `ResearchState`, assign web lifecycle metadata, and transactionally create projections/events; an importer is not part of this release.

## Untested integrations

No live OpenAI account call, real Lean formalization, arbitrary generated-code sandbox, external queue adapter, multi-host failover, hosted GitHub Actions run, production TLS/SSO gateway, or high-concurrency load test was performed. Native Playwright launch was blocked by the authoring Mac sandbox; in-app browser testing was performed instead. See the release test report for exact successful checks.


## v0.7 paper studio additions

The authenticated studio endpoints accept bounded source text or base64-encoded PDFs. API request and reverse-proxy limits are now 2 MB; decoded PDFs are limited to 1 MB, 40 pages, and 60,000 extracted characters. Keep the source within the same single-workspace access boundary. Uploaded equations pass a restricted AST interpreter; arbitrary paper code is not executed. Interpretation and simulation use the existing leased, timed child-process worker. Live OpenAI interpretation and new container execution require deployment validation; see STUDIO-RESULTS.md.
