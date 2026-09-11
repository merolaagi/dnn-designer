# MARE v0.5-web validation report

Validation completed 2026-09-10. Host: macOS Apple Silicon, Python 3.12, Node 22.22.2, Docker Engine 27.5.1. Docker runtime: Linux containers with Python 3.12, PostgreSQL 16, Node 22 build stage, and unprivileged Nginx.

## Automated checks that passed

- **47 Python tests passed** (`pytest -q`, 6.90 seconds in the recorded full run): all 27 original v0.4 tests plus 20 web/integration tests.
- **19 PostgreSQL web integration tests passed** (`MARE_TEST_DATABASE_URL=…/mare_test pytest tests/test_web.py -q`, 7.48 seconds in the recorded run). The database was a dedicated disposable PostgreSQL 16 container.
- **3 frontend unit tests passed** (`npm test`): authenticated API header handling, structured error propagation, and SSE frame/cursor parsing.
- **Ruff lint and formatting checks passed** for the new Python service, migrations and web tests. The preserved engine is intentionally not autoformatted.
- **TypeScript compilation, ESLint and Vite production build passed**. Run-detail/chart code is loaded separately from the dashboard. Build output: approximately 283 kB initial JavaScript and 411 kB run-view JavaScript before gzip.
- **npm audit: 0 vulnerabilities** in the final dependency tree on 2026-09-10. An earlier development dependency advisory was addressed by updating Vitest to 4.1.11. This is a point-in-time npm dependency audit, not a container/Python penetration assessment.
- **Alembic upgrade → schema check → downgrade → upgrade passed** on disposable SQLite. Current application schema reports no pending model differences. PostgreSQL migration ran successfully in Docker startup.
- **OpenAPI export is reproducible**, and the checked-in TypeScript contract is generated from it.
- **36 original engine/source-test files match the v0.4 archive byte for byte**, verified using SHA-256. The input archive was not modified.

## Covered lifecycle and API behavior

The new tests cover end-to-end two-round mock execution; persisted claim projections; graph cycle status; event cursor replay and terminal SSE drain; queued cancellation and resume; active cancellation fencing; concurrent claim contention; stale lease recovery; stale worker rejection; retry backoff and terminal failure; tiny-budget termination; preservation of learned network state on resume; authenticated access; rate limiting; CORS; request IDs; structured validation/404 errors; oversized bodies; mock-provider scope restrictions; readiness before migrations; production configuration failure without a secure token; run-scoped event reads; disabled OpenAI rejection; loading a masked OpenAI key from `.env`; child-process timeout termination; and cancellation of an actually running child process.

## Full-stack checks performed

- Built both production Docker images successfully.
- Started Compose with PostgreSQL, migration job, API, separate worker, and web/reverse proxy.
- Verified API readiness through the public local Nginx port.
- Re-ran the final Compose images in production mode with an ephemeral test token: unauthenticated API access returned 401; an authenticated two-round job completed; UTC timestamp serialization, SSE through Nginx, and CSP headers passed.
- Submitted a **four-round deterministic job through the Nginx API**; it reached `completed` with 7 claims and no worker error.
- Used the connected in-app browser to create a four-round run, observe completed persisted output, inspect claim evidence, filter to an empty result, select proof nodes using Enter, inspect failures/questions/budgets/events/settings, and verify neural cold-start versus trained behavior.
- The four-round browser run produced 5 branches, 7 claims (6 verified), 14 evidence records, a terminated energy branch, and a trained branch policy with 15 examples. Its question network remained below the original activation threshold; the UI correctly showed no trained question network.
- Verified the optional WebMCP read tool returns run summaries and rejects an unexpected input field.
- Inspected desktop and 390 × 844 mobile layouts. The mobile document/body/main widths equal the viewport width, with no document-level horizontal overflow. The wide investigation table scrolls within its own container.
- Saved screenshots in `docs/screenshots/`: dashboard, run overview, proof DAG, neural policy, and mobile viewport. Full-page captures showed browser-capture compositing artifacts, so the deliverable uses normal viewport screenshots.

## Honest limits

The native **Playwright CLI browser suite did not execute successfully**: Chromium startup was rejected by the Mac sandbox's Mach-port permission boundary. This was a browser launch/environment failure, not a passing end-to-end test result. Equivalent core journeys were exercised through the connected browser as described above. The repository includes the Playwright suite and an Ubuntu GitHub Actions workflow, but no hosted Actions run was executed in this session.

No live OpenAI call, provider-account billing validation, real Lean compilation, generated-code sandbox execution, external Celery/RQ/Arq broker, SSO provider, production TLS proxy, multi-host recovery, or load/soak/security-penetration test was performed. The Python dependency set and container OS were not subjected to a dedicated vulnerability scanner. WebMCP support is optional and browser-specific.

The application remains a single-workspace deployment. Exactly-once external calls, complete billing accounting, arbitrary theorem verification, and multi-tenant authorization are not claimed. Review `docs/deployment.md` before deployment beyond a trusted local/private environment.
