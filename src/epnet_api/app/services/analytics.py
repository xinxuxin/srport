from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean

from ..db.database import Database
from ..schemas.api import AnalyticsPoint, AnalyticsSummaryResponse, EventPreview, Resolution


class AnalyticsService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def log_event(
        self,
        request_id: str,
        created_at: datetime,
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
    ) -> None:
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO inference_events (
                    request_id, created_at, input_width, input_height, output_width,
                    output_height, input_bytes, output_bytes, upscale, latency_ms,
                    parameter_count, estimated_macs, estimated_flops
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    created_at.isoformat(),
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
                ),
            )

    def summary(self) -> AnalyticsSummaryResponse:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT request_id, created_at, input_width, input_height,
                       output_width, output_height, latency_ms, upscale
                FROM inference_events
                ORDER BY created_at DESC
                """
            ).fetchall()

        if not rows:
            return AnalyticsSummaryResponse(
                total_requests=0,
                average_latency_ms=0.0,
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

        latency_series = [
            AnalyticsPoint(
                label=datetime.fromisoformat(str(row["created_at"])).strftime("%H:%M:%S"),
                value=float(row["latency_ms"]),
            )
            for row in reversed(rows[:20])
        ]
        upscale_distribution = [
            AnalyticsPoint(label=f"x{scale}", value=float(count))
            for scale, count in sorted(by_upscale.items())
        ]
        recent_events = [
            EventPreview(
                request_id=str(row["request_id"]),
                created_at=datetime.fromisoformat(str(row["created_at"])),
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
            )
            for row in rows[:10]
        ]
        return AnalyticsSummaryResponse(
            total_requests=len(rows),
            average_latency_ms=mean(latencies),
            latest_request_at=datetime.fromisoformat(str(rows[0]["created_at"])),
            average_output_megapixels=mean(output_pixels) / 1_000_000.0,
            total_processed_pixels=sum(output_pixels),
            latency_series=latency_series,
            upscale_distribution=upscale_distribution,
            recent_events=recent_events,
        )
