from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import ResearchEvent, ResearchState


class SQLiteRepository:
    """Zero-setup V0.1 persistence: snapshots + append-only event log."""

    def __init__(self, path: str = "mare.db") -> None:
        self.path = Path(path)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            con.execute(
                """CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def save(self, state: ResearchState) -> None:
        with self._connect() as con:
            con.execute(
                """INSERT INTO runs(run_id, state_json) VALUES(?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET state_json=excluded.state_json, updated_at=CURRENT_TIMESTAMP""",
                (state.run_id, state.model_dump_json()),
            )

    def append_event(self, run_id: str, event: ResearchEvent) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO events(event_id, run_id, event_json) VALUES(?, ?, ?)",
                (event.id, run_id, event.model_dump_json()),
            )

    def load(self, run_id: str) -> ResearchState:
        with self._connect() as con:
            row = con.execute("SELECT state_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            raise KeyError(run_id)
        return ResearchState.model_validate_json(row[0])
