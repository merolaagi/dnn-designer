"""Queue a demo once, without bypassing the worker or fabricating research output."""

import asyncio
from sqlalchemy import select
from .config import Settings
from .db import Database, Run
from .schemas import CreateRun
from .store import emit, make_run


async def seed():
    db = Database(Settings().database_url)
    try:
        async with db.session.begin() as session:
            existing = await session.scalar(
                select(Run.id).where(Run.title == "Demo · Burgers gradient blow-up").limit(1)
            )
            if existing:
                print(existing)
                return
            run = make_run(CreateRun(title="Demo · Burgers gradient blow-up", rounds=4))
            session.add(run)
            await session.flush()
            emit(session, run.id, "queued", provider="mock", source="demo seed")
            print(run.id)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(seed())
