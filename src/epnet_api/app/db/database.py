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
    estimated_flops INTEGER NOT NULL
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()
