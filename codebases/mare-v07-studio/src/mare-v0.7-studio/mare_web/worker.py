import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Protocol

from mare.models import ResearchState

from .config import Settings
from .db import Database, Run
from .store import LocalJobQueue

log = logging.getLogger("mare.worker")


class JobExecutor(Protocol):
    async def execute(self, run_id: str, token: str) -> None: ...


class LocalExecutor:
    def __init__(self, db, settings, queue):
        self.db, self.settings, self.queue = db, settings, queue

    async def stop_child(self, process):
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await process.wait()

    async def execute(self, run_id, token):
        process = None
        try:
            while await self.queue.heartbeat(run_id, token):
                async with self.db.session() as session:
                    run = await session.get(Run, run_id)
                    packet = dict(
                        snapshot=run.snapshot,
                        provider=run.provider,
                        model=self.settings.openai_model,
                        planner_capture=self.settings.planner_capture,
                        planner_checkpoint=self.settings.planner_checkpoint,
                        planner_mode=self.settings.planner_mode,
                    )
                    target = run.target_rounds
                if run.provider == "openai" and not self.settings.openai_enabled:
                    raise RuntimeError("provider_disabled")
                with tempfile.TemporaryDirectory(prefix="mare-round-") as directory:
                    incoming, outgoing = Path(directory) / "in.json", Path(directory) / "out.json"
                    incoming.write_text(json.dumps(packet))
                    # Keep service secrets and DB access out of the engine process environment.
                    env = {
                        k: v
                        for k, v in os.environ.items()
                        if k
                        in {
                            "PATH",
                            "SYSTEMROOT",
                            "LANG",
                            "LC_ALL",
                            "SSL_CERT_FILE",
                            "SSL_CERT_DIR",
                        }
                    }
                    if run.provider == "openai":
                        env["OPENAI_API_KEY"] = (
                            self.settings.openai_api_key.get_secret_value()
                            if self.settings.openai_api_key
                            else ""
                        )
                    process = await asyncio.create_subprocess_exec(
                        sys.executable,
                        "-m",
                        "mare_web.runner",
                        str(incoming),
                        str(outgoing),
                        env=env,
                        start_new_session=True,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    started = asyncio.get_running_loop().time()
                    while process.returncode is None:
                        try:
                            await asyncio.wait_for(process.wait(), timeout=0.5)
                        except asyncio.TimeoutError:
                            pass
                        if not await self.queue.heartbeat(run_id, token):
                            await self.stop_child(process)
                            await self.queue.finish_interrupted(run_id, token)
                            return
                        if asyncio.get_running_loop().time() - started > self.settings.round_timeout_seconds:
                            raise TimeoutError("round_timeout")
                    if process.returncode:
                        raise RuntimeError("engine_round_failed")
                    result = json.loads(outgoing.read_text())
                    state = ResearchState.model_validate(result["snapshot"])
                exhausted = (
                    result["budget_exhausted"]
                    or state.budget.model_calls_used >= state.budget.max_model_calls
                    or state.budget.reserved_output_tokens_used + 4000
                    > state.budget.max_reserved_output_tokens
                    or state.budget.tool_calls_used >= state.budget.max_tool_calls
                )
                status = (
                    "completed"
                    if state.current_round >= target
                    else ("budget_exhausted" if exhausted else "running")
                )
                if not await self.queue.checkpoint(run_id, token, state, status):
                    await self.queue.finish_interrupted(run_id, token)
                    return
                if status != "running":
                    return
            await self.queue.finish_interrupted(run_id, token)
        except asyncio.CancelledError:
            if process:
                await self.stop_child(process)
            await self.queue.finish_interrupted(
                run_id, token, "Worker shutdown; last completed round retained"
            )
            raise
        except Exception as exc:
            if process:
                await self.stop_child(process)
            # Exceptions may contain provider secrets/prompts. Persist only an allowlisted code.
            code = (
                str(exc)
                if str(exc) in {"round_timeout", "provider_disabled", "engine_round_failed"}
                else "worker_error"
            )
            log.error(
                json.dumps(
                    dict(
                        event=code,
                        run_id=run_id,
                        exception_type=type(exc).__name__,
                        frames=[
                            f"{Path(f.filename).name}:{f.lineno}:{f.name}"
                            for f in traceback.extract_tb(exc.__traceback__)
                        ],
                    )
                )
            )
            await self.queue.finish_interrupted(run_id, token, code)


async def serve():
    settings = Settings()
    db = Database(settings.database_url)
    queue = LocalJobQueue(db, settings)
    executor = LocalExecutor(db, settings, queue)
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        while True:
            job = await queue.claim()
            if job:
                await executor.execute(*job)
            else:
                await asyncio.sleep(settings.poll_seconds)
    finally:
        await db.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        asyncio.run(serve())
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    main()
