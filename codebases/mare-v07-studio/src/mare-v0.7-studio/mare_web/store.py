from datetime import timedelta
from uuid import uuid4

from sqlalchemy import case, delete, or_, select, update

from mare.benchmarks import burgers_blowup_problem
from mare.models import ProblemSpec, ResearchBudget, ResearchState

from .db import (
    BranchRecord,
    ClaimRecord,
    DependencyRecord,
    EvidenceRecord,
    FailureRecord,
    QuestionRecord,
    Run,
    RunEvent,
    now,
)


def emit(session, run_id, kind, **payload):
    session.add(RunEvent(run_id=run_id, kind=kind, payload=payload))


def make_run(body):
    problem = (
        burgers_blowup_problem()
        if body.provider == "mock"
        else ProblemSpec(
            title=body.title,
            canonical_statement=body.statement,
            assumptions=body.assumptions,
            success_conditions=body.success_conditions,
        )
    )
    state = ResearchState(
        problem=problem,
        budget=ResearchBudget(
            max_rounds=body.rounds,
            max_model_calls=body.max_model_calls,
            max_reserved_output_tokens=body.max_reserved_output_tokens,
            max_tool_calls=body.max_tool_calls,
        ),
    )
    state.policy.enabled = body.policy == "hybrid"
    return Run(
        id=state.run_id,
        title=body.title,
        provider=body.provider,
        target_rounds=body.rounds,
        snapshot=state.model_dump(mode="json"),
    )


async def project(session, run_id, state):
    mappings = [
        (BranchRecord, "branches"),
        (ClaimRecord, "claims"),
        (EvidenceRecord, "evidence"),
        (FailureRecord, "failures"),
        (QuestionRecord, "questions"),
        (DependencyRecord, "dependency_edges"),
    ]
    for model, field in mappings:
        await session.execute(delete(model).where(model.run_id == run_id))
        for item in getattr(state, field):
            data = item.model_dump(mode="json")
            extra = {}
            if hasattr(model, "status"):
                extra["status"] = data["status"]
            if model is EvidenceRecord:
                extra["claim_id"] = item.claim_id
            if model is DependencyRecord:
                extra.update(parent_id=item.parent_claim_id, child_id=item.child_claim_id)
            session.add(model(run_id=run_id, id=item.id, payload=data, **extra))


class LocalJobQueue:
    """Database queue with CAS claims and lease fencing; replace dispatch in a broker adapter."""

    def __init__(self, db, settings):
        self.db, self.settings = db, settings

    async def recover(self):
        async with self.db.session.begin() as session:
            rows = (
                await session.scalars(
                    select(Run).where(
                        Run.status.in_(["running", "cancelling"]),
                        Run.lease_until < now(),
                    )
                )
            ).all()
            for row in rows:
                status = (
                    "cancelled"
                    if row.status == "cancelling"
                    else (
                        "failed" if row.consecutive_failures + 1 >= self.settings.max_attempts else "queued"
                    )
                )
                result = await session.execute(
                    update(Run)
                    .execution_options(synchronize_session=False)
                    .where(
                        Run.id == row.id,
                        Run.lease_token == row.lease_token,
                        Run.lease_until < now(),
                    )
                    .values(
                        status=case((Run.status == "cancelling", "cancelled"), else_=status),
                        lease_token=None,
                        lease_until=None,
                        consecutive_failures=Run.consecutive_failures + 1,
                        updated_at=now(),
                    )
                    .returning(Run.status)
                )
                actual_status = result.scalar_one_or_none()
                if actual_status:
                    emit(
                        session,
                        row.id,
                        "lease_recovered",
                        status=actual_status,
                        message="Recovered from the last completed round; interrupted work may be replayed.",
                    )

    async def claim(self):
        await self.recover()
        async with self.db.session.begin() as session:
            candidate = await session.scalar(
                select(Run.id)
                .where(
                    Run.status == "queued",
                    Run.available_at <= now(),
                )
                .order_by(Run.created_at)
                .limit(1)
            )
            if not candidate:
                return None
            token = uuid4().hex
            result = await session.execute(
                update(Run)
                .execution_options(synchronize_session=False)
                .where(
                    Run.id == candidate,
                    Run.status == "queued",
                    Run.available_at <= now(),
                )
                .values(
                    status="running",
                    lease_token=token,
                    lease_until=now() + timedelta(seconds=self.settings.lease_seconds),
                    attempts=Run.attempts + 1,
                    updated_at=now(),
                )
            )
            if not result.rowcount:
                return None
            emit(session, candidate, "started")
            return candidate, token

    async def heartbeat(self, run_id, token):
        async with self.db.session.begin() as session:
            result = await session.execute(
                update(Run)
                .execution_options(synchronize_session=False)
                .where(
                    Run.id == run_id,
                    Run.lease_token == token,
                    Run.status == "running",
                    Run.lease_until > now(),
                )
                .values(lease_until=now() + timedelta(seconds=self.settings.lease_seconds))
            )
            return bool(result.rowcount)

    async def checkpoint(self, run_id, token, state, status):
        async with self.db.session.begin() as session:
            old = await session.get(Run, run_id)
            event_count = len(old.snapshot.get("events", []))
            values = dict(
                snapshot=state.model_dump(mode="json"),
                current_round=state.current_round,
                updated_at=now(),
                consecutive_failures=0,
                error=None,
                status=status,
            )
            if status != "running":
                values.update(lease_token=None, lease_until=None)
            result = await session.execute(
                update(Run)
                .execution_options(synchronize_session=False)
                .where(
                    Run.id == run_id,
                    Run.lease_token == token,
                    Run.status == "running",
                    Run.lease_until > now(),
                )
                .values(**values)
            )
            if not result.rowcount:
                return False
            await project(session, run_id, state)
            for e in state.events[event_count:]:
                emit(session, run_id, e.event_type, **e.model_dump(mode="json"))
            emit(
                session,
                run_id,
                "checkpoint",
                round=state.current_round,
                status=status,
                claims=len(state.claims),
                verified=sum(c.status in {"verified", "formalized"} for c in state.claims),
            )
            return True

    async def finish_interrupted(self, run_id, token, error=None):
        async with self.db.session.begin() as session:
            row = await session.get(Run, run_id)
            if not row or row.lease_token != token:
                return
            failures = row.consecutive_failures + 1
            status = (
                "cancelled"
                if row.status == "cancelling"
                else ("failed" if failures >= self.settings.max_attempts else "queued")
            )
            result = await session.execute(
                update(Run)
                .execution_options(synchronize_session=False)
                .where(
                    Run.id == run_id,
                    Run.lease_token == token,
                    or_(Run.status == "cancelling", Run.status == "running"),
                )
                .values(
                    status=case((Run.status == "cancelling", "cancelled"), else_=status),
                    lease_token=None,
                    lease_until=None,
                    error=error,
                    consecutive_failures=failures,
                    updated_at=now(),
                    available_at=now() + timedelta(seconds=min(60, 2**failures)),
                )
                .returning(Run.status)
            )
            actual_status = result.scalar_one_or_none()
            if actual_status:
                emit(
                    session,
                    run_id,
                    "interrupted",
                    status=actual_status,
                    message=error or "Cancellation acknowledged",
                )
