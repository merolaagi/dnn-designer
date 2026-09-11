from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ModelProvider(Protocol):
    name: str

    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        temperature: float = 0.2,
        max_tokens: int = 4000,
    ) -> T:
        ...
