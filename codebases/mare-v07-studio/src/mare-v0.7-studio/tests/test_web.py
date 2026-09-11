import asyncio
import os
from datetime import timedelta

import httpx
import pytest
from sqlalchemy import select, update

from mare.models import ResearchState
from mare_web.api import create_app
from mare_web.config import Settings
from mare_web.db import Base, ClaimRecord, Run, RunEvent, now
from mare_web.schemas import CreateRun
from mare_web.store import LocalJobQueue, make_run
from mare_web.worker import LocalExecutor


@pytest.fixture
async def service(tmp_path):
    test_url = os.environ.get("MARE_TEST_DATABASE_URL")
    if test_url:
        assert test_url.endswith("/mare_test"), "Integration tests require a disposable mare_test database"
    settings = Settings(
        environment="test", database_url=test_url or f"sqlite+aiosqlite:///{tmp_path}/test.db", _env_file=None
    )
    app = create_app(settings)
    db = app.state.db
    async with db.engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
    yield client, db, settings
    await client.aclose()
    async with db.engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await db.close()


async def queued(client, **kwargs):
    response = await client.post("/api/runs", json=kwargs)
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def execute(db, settings):
    queue = LocalJobQueue(db, settings)
    job = await queue.claim()
    assert job
    await LocalExecutor(db, settings, queue).execute(*job)
    return job


async def test_full_round_projection_graph_and_sse(service):
    client, db, settings = service
    run_id = await queued(client, rounds=2)
    await execute(db, settings)
    result = (await client.get(f"/api/runs/{run_id}")).json()
    assert result["status"] == "completed"
    assert result["current_round"] == 2
    state = result["snapshot"]
    assert any(c["status"] == "verified" for c in state["claims"])
    assert any(c["status"] == "rejected" for c in state["claims"])
    assert state["policy"]["examples"]
    assert state["policy"]["enabled"]
    async with db.session() as session:
        projections = (await session.scalars(select(ClaimRecord).where(ClaimRecord.run_id == run_id))).all()
    assert len(projections) == len(state["claims"])
    graph = (await client.get(f"/api/runs/{run_id}/graph")).json()
    assert graph["has_cycle"] is False
    events = (await client.get(f"/api/runs/{run_id}/events")).json()
    assert events[0]["kind"] == "queued"
    cursor = events[-1]["id"]
    response = await client.get(f"/api/runs/{run_id}/stream", headers={"Last-Event-ID": str(cursor)})
    assert response.status_code == 200
    assert "event: settled" in response.text
    assert f"id: {cursor}\n" not in response.text
    assert (await client.post(f"/api/runs/{run_id}/resume")).status_code == 409


async def test_cancel_queued_and_resume(service):
    client, db, settings = service
    run_id = await queued(client)
    assert (await client.post(f"/api/runs/{run_id}/cancel")).json()["status"] == "cancelled"
    assert await LocalJobQueue(db, settings).claim() is None
    assert (await client.post(f"/api/runs/{run_id}/resume")).json()["status"] == "queued"
    await execute(db, settings)
    assert (await client.get(f"/api/runs/{run_id}")).json()["status"] == "completed"


async def test_cancellation_fences_checkpoint(service):
    client, db, settings = service
    run_id = await queued(client)
    queue = LocalJobQueue(db, settings)
    _, token = await queue.claim()
    assert (await client.post(f"/api/runs/{run_id}/cancel")).json()["status"] == "cancelling"
    state = ResearchState.model_validate((await client.get(f"/api/runs/{run_id}")).json()["snapshot"])
    state.current_round = 1
    assert not await queue.checkpoint(run_id, token, state, "completed")
    assert not await queue.heartbeat(run_id, token)
    await queue.finish_interrupted(run_id, token)
    result = (await client.get(f"/api/runs/{run_id}")).json()
    assert result["status"] == "cancelled" and result["current_round"] == 0


async def test_concurrent_claim_has_one_winner(service):
    client, db, settings = service
    await queued(client)
    queue = LocalJobQueue(db, settings)
    jobs = await asyncio.gather(queue.claim(), queue.claim())
    assert sum(job is not None for job in jobs) == 1


async def test_expired_lease_recovery_rejects_old_worker(service):
    client, db, settings = service
    run_id = await queued(client)
    queue = LocalJobQueue(db, settings)
    _, old = await queue.claim()
    async with db.session.begin() as session:
        await session.execute(
            update(Run).where(Run.id == run_id).values(lease_until=now() - timedelta(seconds=1))
        )
    _, new = await queue.claim()
    assert old != new
    state = ResearchState.model_validate((await client.get(f"/api/runs/{run_id}")).json()["snapshot"])
    assert not await queue.checkpoint(run_id, old, state, "completed")
    assert await queue.heartbeat(run_id, new)


async def test_retry_backoff_then_terminal_failure(service):
    client, db, settings = service
    run_id = await queued(client)
    queue = LocalJobQueue(db, settings)
    for n in range(settings.max_attempts):
        job = await queue.claim()
        assert job
        await queue.finish_interrupted(*job, error="round_timeout")
        result = (await client.get(f"/api/runs/{run_id}")).json()
        assert result["status"] == ("failed" if n == settings.max_attempts - 1 else "queued")
        assert await queue.claim() is None
        async with db.session.begin() as session:
            await session.execute(update(Run).where(Run.id == run_id).values(available_at=now()))


async def test_tiny_budget_stops_without_provider_retry(service):
    client, db, settings = service
    run_id = await queued(client, max_reserved_output_tokens=1)
    await execute(db, settings)
    result = (await client.get(f"/api/runs/{run_id}")).json()
    assert result["status"] == "budget_exhausted"
    assert result["current_round"] == 0
    assert result["attempts"] == 1


async def test_resume_preserves_neural_checkpoint(service):
    client, db, settings = service
    run_id = await queued(client, rounds=1)
    async with db.session.begin() as session:
        row = await session.get(Run, run_id)
        snapshot = dict(row.snapshot)
        snapshot["policy"] = dict(snapshot["policy"], min_branch_examples=2)
        row.snapshot = snapshot
    await execute(db, settings)
    original = (await client.get(f"/api/runs/{run_id}")).json()["snapshot"]
    async with db.session.begin() as session:
        snapshot = dict(original)
        snapshot["budget"] = dict(snapshot["budget"], max_rounds=2)
        await session.execute(
            update(Run).where(Run.id == run_id).values(status="cancelled", target_rounds=2, snapshot=snapshot)
        )
    await client.post(f"/api/runs/{run_id}/resume")
    await execute(db, settings)
    final = (await client.get(f"/api/runs/{run_id}")).json()["snapshot"]
    assert original["policy"]["branch_network"]["weights1"]
    assert (
        final["policy"]["branch_network"]["trained_examples"]
        >= original["policy"]["branch_network"]["trained_examples"]
    )
    assert final["current_round"] == 2
    assert set(c["id"] for c in original["claims"]).issubset(c["id"] for c in final["claims"])
    assert len(final["policy"]["examples"]) >= len(original["policy"]["examples"])


async def test_auth_limits_errors_and_cors(service):
    _, db, settings = service
    secure = settings.model_copy(
        update={"auth_token": __import__("pydantic").SecretStr("test-token"), "rate_limit_per_minute": 5}
    )
    app = create_app(secure)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        assert (await client.get("/health/live")).status_code == 200
        response = await client.get("/api/runs")
        assert response.status_code == 401 and response.headers["x-request-id"]
        assert response.json()["error"]["request_id"] == response.headers["x-request-id"]
        headers = {"Authorization": "Bearer test-token", "Origin": "http://localhost:5173"}
        response = await client.get("/api/runs", headers=headers)
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
        assert (await client.get("/api/runs/nope", headers=headers)).status_code == 404
        assert (await client.get("/api/runs", headers=headers)).status_code == 200
        assert (await client.get("/api/runs", headers=headers)).status_code == 429
    await app.state.db.close()


async def test_body_validation_and_mock_scope(service):
    client, _, _ = service
    for body in [
        {"rounds": 0},
        {"provider": "mock", "statement": "a custom problem"},
        {"provider": "openai"},
        {"extra": "forbidden"},
    ]:
        response = await client.post("/api/runs", json=body)
        assert response.status_code == 422 and "error" in response.json()
    assert (await client.post("/api/runs", content=b"x" * 2000001)).status_code == 413
    assert (await client.get("/no-such-route")).json()["error"]["code"] == "http_404"
    assert (await client.get("/api/runs?limit=1000")).status_code == 422


async def test_readiness_requires_migrations(service):
    client, _, _ = service
    assert (await client.get("/health/ready")).status_code == 503


def test_production_config_fails_closed():
    with pytest.raises(ValueError):
        Settings(environment="production", auth_token=None, _env_file=None)


async def test_event_cursor_is_run_scoped(service):
    client, db, _ = service
    one = await queued(client)
    two = await queued(client)
    events = (await client.get(f"/api/runs/{one}/events")).json()
    assert all(e["run_id"] == one for e in events)
    assert two != one
    assert (await client.get(f"/api/runs/{one}/stream", headers={"Last-Event-ID": "bad"})).status_code == 422
    async with db.session() as session:
        assert len((await session.scalars(select(RunEvent))).all()) == 2


async def test_openai_requires_server_enablement(service):
    client, _, _ = service
    response = await client.post(
        "/api/runs", json={"provider": "openai", "statement": "Investigate a mathematical question"}
    )
    assert response.status_code == 409


def test_openapi_has_typed_state():
    schema = create_app(Settings(_env_file=None)).openapi()
    assert "ResearchState" in schema["components"]["schemas"]
    assert "snapshot" in schema["components"]["schemas"]["RunDetail"]["properties"]
    assert schema["paths"]["/api/runs"]["get"]["security"]


def test_run_factory_preserves_benchmark():
    run = make_run(CreateRun(title="My display title"))
    assert run.snapshot["problem"]["title"] != run.title
    assert "Burgers" in run.snapshot["problem"]["canonical_statement"]


async def test_timeout_kills_round_process(service, monkeypatch):
    client, db, settings = service
    settings = settings.model_copy(update={"round_timeout_seconds": 1, "max_attempts": 1})
    original = asyncio.create_subprocess_exec
    children = []

    async def slow_child(*args, **kwargs):
        import sys

        process = await original(sys.executable, "-c", "import time; time.sleep(30)", **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", slow_child)
    run_id = await queued(client)
    await execute(db, settings)
    result = (await client.get(f"/api/runs/{run_id}")).json()
    assert result["status"] == "failed" and result["error"] == "round_timeout"
    assert children and children[0].returncode is not None


async def test_running_cancellation_kills_child(service, monkeypatch):
    client, db, settings = service
    original = asyncio.create_subprocess_exec
    children = []
    started = asyncio.Event()

    async def slow_child(*args, **kwargs):
        import sys

        process = await original(sys.executable, "-c", "import time; time.sleep(30)", **kwargs)
        children.append(process)
        started.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", slow_child)
    run_id = await queued(client)
    job = asyncio.create_task(execute(db, settings))
    await asyncio.wait_for(started.wait(), timeout=5)
    await client.post(f"/api/runs/{run_id}/cancel")
    await asyncio.wait_for(job, timeout=5)
    result = (await client.get(f"/api/runs/{run_id}")).json()
    assert result["status"] == "cancelled" and result["current_round"] == 0
    assert children[0].returncode is not None


def test_alembic_upgrade_downgrade_upgrade(tmp_path):
    import subprocess
    import sys

    env = dict(os.environ, MARE_DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path}/migration.db")
    for command in [("upgrade", "head"), ("check",), ("downgrade", "base"), ("upgrade", "head")]:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", *command], env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


def test_openai_key_loads_from_dotenv_without_exposure(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    envfile = tmp_path / ".env"
    envfile.write_text("OPENAI_API_KEY=fixture-only-secret\n")
    settings = Settings(_env_file=envfile)
    assert settings.openai_api_key.get_secret_value() == "fixture-only-secret"
    assert "fixture-only-secret" not in repr(settings)


async def test_graph_planner_worker_roundtrip(service, tmp_path):
    pytest.importorskip("torch")
    from mare_planner.network import GraphPlanner, save_checkpoint

    client, db, settings = service
    path = tmp_path / "planner.pt"
    save_checkpoint(path, GraphPlanner(16, 2), {"source": "synthetic", "trained_examples": 10})
    settings.planner_capture = True
    settings.planner_checkpoint = str(path)
    settings.planner_mode = "shadow"
    run_id = await queued(client, rounds=2)
    await execute(db, settings)
    detail = (await client.get(f"/api/runs/{run_id}")).json()
    assert detail["status"] == "completed"
    data = detail["snapshot"]["planner_data"]
    assert data["status"]["active"] and data["status"]["mode"] == "shadow"
    assert len(data["traces"]) == 2 and all(r["complete"] for r in data["traces"])
    assert data["checkpoint"] and data["predictions"]


async def test_studio_queue_worker_and_fork(service):
    client, db, settings = service
    body = {
        "title": "Paper decay",
        "text": "Assume a first order law. dx/dt = -k*x.",
        "model": {
            "name": "Decay",
            "variables": ["x"],
            "derivatives": ["-k*x"],
            "initial": [1],
            "parameters": [{"name": "k", "value": 0.5}],
            "horizon": 4,
        },
    }
    created = await client.post("/api/studio", json=body)
    assert created.status_code == 201, created.text
    assert created.json()["workflow"] == "studio"
    key = created.json()["id"]
    await execute(db, settings)
    result = (await client.get(f"/api/runs/{key}")).json()
    assert result["status"] == "completed" and result["current_round"] == 3
    data = result["snapshot"]["studio_data"]
    assert len(data["trace"]) == 3 and data["experiment"]["comparison_complete"]
    assert data["analysis"]["findings"][0]["source_start"] >= 0
    fork = await client.post("/api/studio", json={**body, "parent_run_id": key})
    assert fork.status_code == 201 and fork.json()["id"] != key
    bad = await client.post(
        "/api/studio", json={**body, "model": {**body["model"], "derivatives": ['__import__("os")']}}
    )
    assert bad.status_code == 422
    assert (await client.post("/api/studio", json={**body, "provider": "openai"})).status_code == 409


async def test_studio_research_bridge_keeps_source_unverified(service):
    client, db, settings = service
    created = await client.post(
        "/api/studio",
        json={"title": "Imported paper", "text": "Assume a smooth field. An open question remains."},
    )
    key = created.json()["id"]
    await execute(db, settings)
    settings.openai_enabled = True
    settings.openai_model = "test-model-not-called"
    response = await client.post(
        f"/api/studio/{key}/research",
        json={"objective": "Investigate whether the proposed mechanism extends to higher dimensions."},
    )
    assert response.status_code == 201, response.text
    child = (await client.get("/api/runs/" + response.json()["id"])).json()
    assert child["workflow"] == "research" and child["provider"] == "openai"
    assert child["snapshot"]["budget"]["max_model_calls"] == 20
    assert child["snapshot"]["sources"][0]["metadata"]["status"] == "unverified imported source"
    assert child["snapshot"]["sources"][0]["external_id"] == key
    assert not child["snapshot"]["claims"]
