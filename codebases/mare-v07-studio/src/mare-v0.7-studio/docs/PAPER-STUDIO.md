# Paper research studio: v0.7 experimental release

The aim is a research workshop that makes its work inspectable and lets learners try alternative methods. This release implements a first complete path: source import → source-linked candidate rules → editable mathematical model → computed experiment → animated replay → alternative model → optional MARE research investigation.

It does not claim to understand every paper or solve arbitrary open problems. A numerical demonstration is not a theorem, and a plausible extracted rule is not verified knowledge.

## Start locally

Use Python 3.12 and Node 22. From the extracted project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -c requirements.lock -e '.[dev]'
cp .env.example .env
alembic upgrade head
npm ci --prefix web
```

Start these in separate terminals, with the same virtual environment active:

```bash
uvicorn mare_web.api:app --host 127.0.0.1 --port 8000
python -m mare_web.worker
npm run dev --prefix web
```

Open http://127.0.0.1:5173/studio. The Mathematics, Biology, and Chemistry buttons populate explicit authored examples. Choose Mathematics, run the proposal, and open the research theatre after the worker finishes. Scrub the timeline to inspect a computed point, play it, expand the first RK4 calculation, or export the animation as WebM. The source record can also be exported as JSON.

The existing Docker application remains available with `docker compose up --build -d`. This release uses a distinct v0.7 Compose project/image name. Set `MARE_WEB_PORT=8087` if an earlier version owns port 8080. Container rebuilds for v0.7 were not executed locally. Configuration validation was performed.

## What importing a paper does

Paste up to 60,000 characters, or upload a PDF/text file under 1 MB. PDFs must be unencrypted, contain extractable text, and have at most 40 pages. Scanned papers require OCR beforehand. Formula layout may be lost during PDF text extraction; inspect the preserved text. The original text hash and, for PDFs, original byte hash are retained. The original PDF file is not stored after successful extraction.

Offline mode detects candidate passages using lexical rules: equations, assumptions, algorithm descriptions, rules, and open questions. It does not semantically understand arbitrary papers and will not silently substitute a demonstration when no executable model is available.

Optional OpenAI mode uses the existing configured provider for structured paper interpretation. It requests source quotes, proposed rules/algorithms, assumptions, limitations, potential methods, and optionally a small dynamical model. Each quote is checked for exact occurrence in the imported text. Missing anchors are labeled **unanchored model output**. An exact quote match verifies provenance only, not the interpretation or truth of the statement.

Enable the provider with the existing `MARE_OPENAI_ENABLED`, `MARE_OPENAI_MODEL`, and `OPENAI_API_KEY` settings. Choosing OpenAI sends the imported text to that provider. No live OpenAI paper-analysis call was made during validation; mocked structured-provider responses tested the local integration. No new model key is bundled.

## Making an experimental model

You can keep the source and create a blank editable model, change a suggested model, or fork a prior attempt. The executable model family currently supports **one to four coupled first-order ODEs**, numeric parameters, initial values, a bounded horizon, and explicit assumptions. It is not yet a general neural-architecture generator or a theorem-proving model compiler.

The parser accepts numeric constants, named state variables/parameters, time `t`, arithmetic, bounded constant powers, and sin/cos/exp/log/sqrt/abs. It interprets an allowlisted expression tree. It does not call Python `eval`, execute uploaded algorithms, install paper dependencies, or run model-generated programs.

The numerical worker uses explicit classical RK4 at 200 steps, repeats at 400 steps, and records the maximum pointwise difference over the shared sample grid. This is a numerical consistency check, not a certified error bound. Stiff systems may require another solver. Overflow, singular arithmetic, and out-of-range trajectories are recorded as failed experiments, not as proofs of blow-up. Units and scientific applicability require review.

The cost of paper interpretation is represented by the existing reserved-output-token budget. Numerical experiments use fixed step counts and the existing child-process timeout/cancellation architecture. There is no GPU training in this workflow.

## Learning from an attempt

The dashboard displays the source, hypotheses, assumptions, model, calculation method, first-step RK4 slopes, observed result, limitations, and suggested next methods. These are public scientific artifacts and recorded computations, not hidden language-model chain-of-thought or invented internal reasoning.

The popup animates the actual recorded numerical samples. WebM export records the same canvas, including the model equation, time, values, and the numerical-not-proof label. Browser encoding support varies. Closing the popup while recording cancels the export. The source link and underlying packet should accompany shared animations when scientific provenance matters.

**Fork and try your own method** creates an independent run with a parent link. Editing the model updates its provenance to a learner-modified proposal. Earlier attempts remain available for comparison. There is no automatic claim that a fork is a scientific improvement.

## Connecting the paper to research on a harder problem

After analysis, enter a research objective and select **Start MARE research from this source**. This requires the configured OpenAI provider because the offline research provider only implements its existing Burgers benchmark.

The bridge creates an ordinary MARE research run, attaches the studio source with its hash and an **unverified** label, and includes up to 7,000 source characters alongside the objective. It caps the run at 20 model calls, 80,000 reserved output tokens and three rounds. The source excerpt can be incomplete for long papers; inspect the new run's problem statement before relying on it. Existing branches, critics, evidence checks, assumption audits, and proof-DAG rules remain in force. No extracted equation or numerical plot is automatically promoted to a verified claim.

This is the bridge toward studying open questions. It does not establish that an unsolved problem will be solved or that an undecidable problem becomes decidable.

## Persistence, API, and trust boundaries

Studio jobs use the existing SQL queue, lease fencing, retries, cancellation, and completed-round checkpoint mechanism. Three stages run in separate child processes. The `studio_data` field lives in the existing research snapshot JSON; no new database migration is needed. `RunSummary.workflow` distinguishes studio and research runs on the dashboard.

New authenticated endpoints:
- `POST /api/studio`: queue import, interpretation, and an optional numerical experiment.
- `POST /api/studio/{run_id}/research`: create a separate research investigation from a completed source analysis.

Existing run detail/export, events, cancel, and resume endpoints also apply. API and reverse-proxy body limits are raised to accommodate the bounded PDF payload. Source files, prompts, and uploaded code are treated as data. The parser and worker are not a general sandbox for arbitrary generated software.

## Remaining work toward the larger vision

- Broader paper-to-model adapters: PDEs, symbolic rewriting, formal proof search, statistical models, and neural surrogate training.
- Reliable formula/layout/OCR extraction and equation-to-source-region alignment.
- Research-driven model architecture search and GPU training with scientific baselines.
- Automated animation recipes for proof transformations and non-ODE experiments.
- Prospective evaluation across real mathematical, biological, and chemical papers.

The v0.6 graph planner remains available as an optional component; it has not been retrained on papers in this release. The uploaded DNN Designer has not been merged into the web application. Its general plugin/code-execution features require a separate integration design rather than exposing them through the public API.
