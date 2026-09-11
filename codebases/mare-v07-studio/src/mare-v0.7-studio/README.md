# MARE v0.7 — paper research studio

An experimental research-and-learning workbench: import mathematical content or a research paper, inspect source-linked candidate rules, build an explicit model, test it, replay the calculations, and fork an alternative method. A source claim, a model hypothesis, a numerical result, and a verified theorem remain different things.

## Quick start — no model key required

Python 3.12 and Node 22:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -c requirements.lock -e '.[dev]'
cp .env.example .env
alembic upgrade head
npm ci --prefix web
```

Run the following in three terminals from this directory (activate the virtual environment in Python terminals):

```bash
uvicorn mare_web.api:app --host 127.0.0.1 --port 8000
python -m mare_web.worker
npm run dev --prefix web
```

Open **http://127.0.0.1:5173/studio**. Select the Mathematics example, choose **Analyze and test proposal**, then open the animated research theatre. Or import your own text/PDF and create an editable model while preserving the source.

- [Paper studio guide and boundaries](docs/PAPER-STUDIO.md)
- [v0.7 test results](docs/STUDIO-RESULTS.md)
- [Application architecture](docs/architecture.md)
- [Deployment and security](docs/deployment.md)
- [Graph neural planner](docs/GRAPH-PLANNER.md)
- [Changelog](CHANGELOG.md)

## Capabilities in this release

- Text/markdown/LaTeX-text and bounded text-based PDF import, with source hashes and quote anchors.
- Offline lexical passage detection; optional OpenAI structured interpretation of rules, equations, assumptions, algorithms and open questions.
- Editable models with up to four coupled first-order differential equations, parameters, initial conditions and assumptions.
- Bounded RK4 experiments, refinement comparison, explicit numerical failure reporting, and actual first-step calculations.
- Animated popup with play/pause/scrubbing and WebM export of the calculated trajectory.
- Preserved alternative attempts, source-record export, and a bridge into the existing MARE branch/critic/evidence/proof-DAG workflow.
- Existing v0.6 graph planner, source code and synthetic checkpoints retained as an optional component.

This is the first studio increment, not a universal paper-understanding or paper-to-neural-network system. Arbitrary algorithm execution, automatic architecture training from papers, general PDE/proof animations, OCR and real research performance validation remain future work. See the detailed guide. The numerical demonstration cannot prove an open theorem.

## Optional language-model interpretation

Set `MARE_OPENAI_ENABLED=true`, an explicit `MARE_OPENAI_MODEL`, and `OPENAI_API_KEY` in `.env`; install `pip install -c requirements.lock -e '.[openai]'`. Restart API/worker. Choosing OpenAI interpretation sends the imported text to that configured provider. No API key is bundled. This live integration was not exercised locally.

## Docker

```bash
cp .env.example .env
docker compose up --build -d
```

Open http://localhost:8080/studio. Use `MARE_WEB_PORT=8087` when v0.5/v0.6 already owns port 8080. v0.7 has a separate Compose project name. Default containers do not install PyTorch. The optional CPU planner overlay remains available; it uses bundled synthetic checkpoints in shadow mode. v0.7 Compose configuration was validated, but its new image build and Linux/PostgreSQL runtime were not executed in this environment.

## Tests

```bash
pytest -q
ruff check .
npm run lint --prefix web
npm test --prefix web
npm run build --prefix web
```

Install `.[planner]` to run the optional graph neural-network tests and benchmark. Checked-in OpenAPI and generated TypeScript contracts are regenerated with `python scripts/export_openapi.py` and `npm run schemas --prefix web`.

Previous archives are preserved. v0.5/v0.6 operational notes and earlier test results are retained as historical documents; they are not evidence that new v0.7 integrations have been validated.
