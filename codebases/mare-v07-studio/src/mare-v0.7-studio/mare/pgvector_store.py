from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class VectorHit:
    key: str
    text: str
    score: float
    metadata: dict


class PgVectorStore:
    """Optional pgvector persistence for research-memory embeddings.

    Install with `pip install -e '.[postgres]'`. Imports are intentionally lazy
    so the zero-setup SQLite path remains dependency-light.
    """

    def __init__(self, dsn: str, dimensions: int):
        self.dsn = dsn
        self.dimensions = dimensions

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install MARE with the 'postgres' extra") from exc
        return psycopg.connect(self.dsn)

    def initialize(self) -> None:
        with self._connect() as con:
            with con.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS mare_memory (
                        key TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self.dimensions}) NOT NULL
                    )
                """)
            con.commit()

    def upsert(self, key: str, kind: str, text: str, embedding: Iterable[float], metadata: dict | None = None) -> None:
        values = list(embedding)
        if len(values) != self.dimensions:
            raise ValueError(f"expected embedding dimension {self.dimensions}, got {len(values)}")
        vector_literal = "[" + ",".join(str(float(x)) for x in values) + "]"
        with self._connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mare_memory(key, kind, text, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT(key) DO UPDATE SET
                      kind=excluded.kind, text=excluded.text,
                      metadata=excluded.metadata, embedding=excluded.embedding
                    """,
                    (key, kind, text, metadata or {}, vector_literal),
                )
            con.commit()

    def search(self, embedding: Iterable[float], limit: int = 8, kind: str | None = None) -> list[VectorHit]:
        values = list(embedding)
        if len(values) != self.dimensions:
            raise ValueError(f"expected embedding dimension {self.dimensions}, got {len(values)}")
        vector_literal = "[" + ",".join(str(float(x)) for x in values) + "]"
        where = "WHERE kind = %s" if kind else ""
        sql = f"""
            SELECT key, text, 1 - (embedding <=> %s::vector) AS score, metadata
            FROM mare_memory
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        args = (vector_literal, kind, vector_literal, limit) if kind else (vector_literal, vector_literal, limit)
        with self._connect() as con:
            with con.cursor() as cur:
                cur.execute(sql, args)
                rows = cur.fetchall()
        return [VectorHit(key=r[0], text=r[1], score=float(r[2]), metadata=r[3] or {}) for r in rows]
