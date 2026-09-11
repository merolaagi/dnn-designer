from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_']+")


class SimilarityBackend(Protocol):
    name: str

    def similarity(self, query: str, text: str) -> float:
        ...


class LexicalSimilarityBackend:
    name = "lexical-cosine"

    def similarity(self, query: str, text: str) -> float:
        a = {t.lower() for t in _TOKEN_RE.findall(query) if len(t) > 1}
        b = {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1}
        if not a or not b:
            return 0.0
        return len(a & b) / math.sqrt(len(a) * len(b))


class HashEmbeddingBackend:
    """Deterministic local semantic-ish backend for tests/offline development.

    It is not a learned embedding model. Tokens are signed-hashed into a fixed
    vector so cosine similarity can exercise the embedding interface without a
    model/service dependency. Swap this for a real embedding provider in prod.
    """

    name = "hash-embedding"

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for token in _TOKEN_RE.findall(text.lower()):
            if len(token) <= 1:
                continue
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            n = int.from_bytes(digest, "big")
            idx = n % self.dimensions
            sign = 1.0 if ((n >> 8) & 1) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def similarity(self, query: str, text: str) -> float:
        a, b = self._embed(query), self._embed(text)
        return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
