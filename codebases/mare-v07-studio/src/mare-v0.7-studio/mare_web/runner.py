"""One isolated round. No API/database credentials are passed to this subprocess."""

import asyncio
import json
import sys
from pathlib import Path

from mare.engine import ResearchEngine
from mare.budget import BudgetExhausted
from mare.models import ResearchState
from mare.providers.mock_provider import MockResearchProvider


async def run(input_path, output_path):
    packet = json.loads(Path(input_path).read_text())
    state = ResearchState.model_validate(packet["snapshot"])
    if packet["provider"] == "openai":
        from mare.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(model=packet["model"])
        provider.client = provider.client.with_options(timeout=90, max_retries=0)
    else:
        provider = MockResearchProvider()
    if state.studio_data:
        from mare_web.studio import studio_step

        try:
            await studio_step(state, provider)
        except ValueError as exc:
            state.studio_data["error"] = str(exc)
            state.studio_data["stage"] = "needs correction"
            state.studio_data.setdefault("trace", []).append(
                {"title": "Stage requires correction", "detail": str(exc), "status": "not validated"}
            )
            state.current_round = 3
        Path(output_path).write_text(
            json.dumps({"snapshot": state.model_dump(mode="json"), "budget_exhausted": False})
        )
        return
    policy = None
    if packet.get("planner_capture") or packet.get("planner_checkpoint"):
        from mare_planner.policy import GraphResearchPolicy

        policy = GraphResearchPolicy(
            checkpoint=packet.get("planner_checkpoint"),
            mode=packet.get("planner_mode", "shadow"),
            source="real" if packet["provider"] == "openai" else "mock",
            enabled=state.policy.enabled,
        )
    engine = ResearchEngine(provider, policy_enabled=state.policy.enabled, research_policy=policy)
    exhausted = False
    try:
        await engine.run(state, rounds=1)
    except BudgetExhausted:
        state = ResearchState.model_validate(packet["snapshot"])
        exhausted = True
    Path(output_path).write_text(
        json.dumps({"snapshot": state.model_dump(mode="json"), "budget_exhausted": exhausted})
    )


if __name__ == "__main__":
    asyncio.run(run(*sys.argv[1:]))
