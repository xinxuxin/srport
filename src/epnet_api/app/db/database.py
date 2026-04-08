from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS inference_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT 'anonymous',
    input_width INTEGER NOT NULL,
    input_height INTEGER NOT NULL,
    output_width INTEGER NOT NULL,
    output_height INTEGER NOT NULL,
    input_bytes INTEGER NOT NULL,
    output_bytes INTEGER NOT NULL,
    upscale INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    parameter_count INTEGER NOT NULL,
    estimated_macs INTEGER NOT NULL,
    estimated_flops INTEGER NOT NULL,
    method TEXT NOT NULL DEFAULT 'epnet',
    checkpoint_name TEXT NOT NULL DEFAULT 'default',
    output_format TEXT NOT NULL DEFAULT 'PNG',
    tile_size INTEGER NOT NULL DEFAULT 0,
    input_artifact_url TEXT NOT NULL DEFAULT '',
    output_artifact_url TEXT NOT NULL DEFAULT ''
);
"""

MIGRATIONS: tuple[tuple[str, str], ...] = (
    (
        "session_id",
        "ALTER TABLE inference_events ADD COLUMN session_id TEXT NOT NULL DEFAULT 'anonymous'",
    ),
    ("method", "ALTER TABLE inference_events ADD COLUMN method TEXT NOT NULL DEFAULT 'epnet'"),
    (
        "checkpoint_name",
        "ALTER TABLE inference_events ADD COLUMN checkpoint_name TEXT NOT NULL DEFAULT 'default'",
    ),
    (
        "output_format",
        "ALTER TABLE inference_events ADD COLUMN output_format TEXT NOT NULL DEFAULT 'PNG'",
    ),
    ("tile_size", "ALTER TABLE inference_events ADD COLUMN tile_size INTEGER NOT NULL DEFAULT 0"),
    (
        "input_artifact_url",
        "ALTER TABLE inference_events ADD COLUMN input_artifact_url TEXT NOT NULL DEFAULT ''",
    ),
    (
        "output_artifact_url",
        "ALTER TABLE inference_events ADD COLUMN output_artifact_url TEXT NOT NULL DEFAULT ''",
    ),
)


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(SCHEMA)
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(inference_events)").fetchall()
            }
            for column_name, statement in MIGRATIONS:
                if column_name not in columns:
                    try:
                        connection.execute(statement)
                    except sqlite3.OperationalError as error:
                        if "duplicate column name" not in str(error).lower():
                            raise
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_inference_events_created_at
                ON inference_events(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_inference_events_session_id
                ON inference_events(session_id);
                """
            )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()
