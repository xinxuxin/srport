from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import datetime
from statistics import mean

from ..db.database import Database
from ..schemas.api import (
    AnalyticsPoint,
    AnalyticsSummaryResponse,
    EventPreview,
    HistoryEventResponse,
    RecentEventsResponse,
    Resolution,
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class AnalyticsService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def log_event(
        self,
        request_id: str,
        created_at: datetime,
        session_id: str,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
        input_bytes: int,
        output_bytes: int,
        upscale: int,
        latency_ms: float,
        parameter_count: int,
        estimated_macs: int,
        estimated_flops: int,
        method: str,
        checkpoint_name: str,
        output_format: str,
        tile_size: int,
        input_artifact_url: str,
        output_artifact_url: str,
    ) -> None:
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO inference_events (
                    request_id, created_at, session_id, input_width, input_height, output_width,
                    output_height, input_bytes, output_bytes, upscale, latency_ms,
                    parameter_count, estimated_macs, estimated_flops, method, checkpoint_name,
                    output_format, tile_size, input_artifact_url, output_artifact_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    created_at.isoformat(),
                    session_id,
                    input_width,
                    input_height,
                    output_width,
                    output_height,
                    input_bytes,
                    output_bytes,
                    upscale,
                    latency_ms,
                    parameter_count,
                    estimated_macs,
                    estimated_flops,
                    method,
                    checkpoint_name,
                    output_format,
                    tile_size,
                    input_artifact_url,
                    output_artifact_url,
                ),
            )

    def summary(self) -> AnalyticsSummaryResponse:
        rows = self._fetch_rows()
        if not rows:
            return AnalyticsSummaryResponse(
                total_requests=0,
                session_count=0,
                average_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                latest_request_at=None,
                average_output_megapixels=0.0,
                total_processed_pixels=0,
                latency_series=[],
                upscale_distribution=[],
                recent_events=[],
            )

        latencies = [float(row["latency_ms"]) for row in rows]
        output_pixels = [int(row["output_width"]) * int(row["output_height"]) for row in rows]
        by_upscale = Counter(int(row["upscale"]) for row in rows)
        session_ids = {str(row["session_id"]) for row in rows if str(row["session_id"]).strip()}

        return AnalyticsSummaryResponse(
            total_requests=len(rows),
            session_count=len(session_ids),
            average_latency_ms=mean(latencies),
            p50_latency_ms=_percentile(latencies, 0.5),
            p95_latency_ms=_percentile(latencies, 0.95),
            latest_request_at=datetime.fromisoformat(str(rows[0]["created_at"])),
            average_output_megapixels=mean(output_pixels) / 1_000_000.0,
            total_processed_pixels=sum(output_pixels),
            latency_series=self._latency_series(rows),
            upscale_distribution=[
                AnalyticsPoint(label=f"x{scale}", value=float(count))
                for scale, count in sorted(by_upscale.items())
            ],
            recent_events=self._recent_events(rows, limit=10),
        )

    def recent(self, limit: int = 10) -> RecentEventsResponse:
        return RecentEventsResponse(
            recent_events=self._recent_events(self._fetch_rows(), limit=limit)
        )

    def history_event(self, request_id: str) -> HistoryEventResponse | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT request_id, created_at, session_id, input_width, input_height,
                       output_width, output_height, latency_ms, upscale, method,
                       checkpoint_name, output_format, tile_size, parameter_count,
                       estimated_macs, estimated_flops, input_artifact_url, output_artifact_url
                FROM inference_events
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()

        if row is None:
            return None

        return HistoryEventResponse(
            request_id=str(row["request_id"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            session_id=str(row["session_id"]),
            method=str(row["method"]),
            checkpoint_name=str(row["checkpoint_name"]),
            output_format=str(row["output_format"]),
            tile_size=int(row["tile_size"]),
            input_resolution=Resolution(
                width=int(row["input_width"]),
                height=int(row["input_height"]),
            ),
            output_resolution=Resolution(
                width=int(row["output_width"]),
                height=int(row["output_height"]),
            ),
            latency_ms=float(row["latency_ms"]),
            upscale=int(row["upscale"]),
            parameter_count=int(row["parameter_count"]),
            estimated_macs=int(row["estimated_macs"]),
            estimated_flops=int(row["estimated_flops"]),
            input_image_url=str(row["input_artifact_url"]),
            output_image_url=str(row["output_artifact_url"]),
        )

    def _fetch_rows(self) -> list[sqlite3.Row]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT request_id, created_at, session_id, input_width, input_height,
                       output_width, output_height, latency_ms, upscale, method,
                       checkpoint_name, output_format, input_artifact_url,
                       output_artifact_url
                FROM inference_events
                ORDER BY created_at DESC
                """
            ).fetchall()
        return list(rows)

    def _latency_series(self, rows: list[sqlite3.Row]) -> list[AnalyticsPoint]:
        return [
            AnalyticsPoint(
                label=datetime.fromisoformat(str(row["created_at"])).strftime("%H:%M:%S"),
                value=float(row["latency_ms"]),
            )
            for row in reversed(rows[:20])
        ]

    def _recent_events(self, rows: list[sqlite3.Row], limit: int) -> list[EventPreview]:
        return [
            EventPreview(
                request_id=str(row["request_id"]),
                created_at=datetime.fromisoformat(str(row["created_at"])),
                session_id=str(row["session_id"]),
                input_resolution=Resolution(
                    width=int(row["input_width"]),
                    height=int(row["input_height"]),
                ),
                output_resolution=Resolution(
                    width=int(row["output_width"]),
                    height=int(row["output_height"]),
                ),
                latency_ms=float(row["latency_ms"]),
                upscale=int(row["upscale"]),
                method=str(row["method"]),
                checkpoint_name=str(row["checkpoint_name"]),
                output_format=str(row["output_format"]),
                input_image_url=str(row["input_artifact_url"]),
                output_image_url=str(row["output_artifact_url"]),
            )
            for row in rows[:limit]
        ]
