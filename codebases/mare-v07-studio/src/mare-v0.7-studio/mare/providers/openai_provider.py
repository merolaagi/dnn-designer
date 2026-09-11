from __future__ import annotations

import os
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class OpenAIProvider:
    """Structured-output provider using the OpenAI Responses API."""

    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("MARE_OPENAI_MODEL", "gpt-5.6")
        self.client = AsyncOpenAI()

    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        temperature: float = 0.2,
        max_tokens: int = 4000,
    ) -> T:
        # The Responses parse helper validates the returned text against the
        # supplied Pydantic schema. We intentionally do not depend on hidden
        # model reasoning; only the structured research packet is persisted.
        response = await self.client.responses.parse(
            model=self.model,
            instructions=system,
            input=prompt,
            text_format=schema,
            max_output_tokens=max_tokens,
        )

        for output in response.output:
            if output.type != "message":
                continue
            for item in output.content:
                if item.type == "output_text" and getattr(item, "parsed", None) is not None:
                    return item.parsed
        raise RuntimeError("OpenAI response did not contain a parsed structured output")
