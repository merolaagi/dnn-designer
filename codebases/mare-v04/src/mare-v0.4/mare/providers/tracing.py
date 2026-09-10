from __future__ import annotations

import hashlib
import time
from typing import TypeVar

from pydantic import BaseModel

from ..models import ModelInvocation
from .base import ModelProvider

T = TypeVar("T", bound=BaseModel)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class TracingProvider:
    """Provider decorator that records reproducibility/provenance metadata."""

    def __init__(self, inner: ModelProvider):
        self.inner = inner
        self.name = getattr(inner, "name", inner.__class__.__name__)
        self.model = getattr(inner, "model", None)
        self._records: list[ModelInvocation] = []

    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        temperature: float = 0.2,
        max_tokens: int = 4000,
    ) -> T:
        start = time.perf_counter()
        success = False
        error: str | None = None
        try:
            result = await self.inner.generate(
                system=system,
                prompt=prompt,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            success = True
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._records.append(
                ModelInvocation(
                    provider=self.name,
                    model=self.model,
                    schema_name=schema.__name__,
                    system_hash=_hash(system),
                    prompt_hash=_hash(prompt),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    elapsed_ms=elapsed_ms,
                    success=success,
                    error=error,
                )
            )

    def drain_records(self) -> list[ModelInvocation]:
        records = self._records[:]
        self._records.clear()
        return records
