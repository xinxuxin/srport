from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Resolution(BaseModel):
    width: int
    height: int


class ModelInfoResponse(BaseModel):
    name: str
    paper_title: str
    checkpoint_loaded: bool
    checkpoint_path: Optional[str]
    weights_source: str
    upscale: int
    parameter_count: int
    estimated_macs: int
    estimated_flops: int
    reference_latency_ms: float
    architecture: dict[str, object]


class RuntimeInfo(BaseModel):
    latency_ms: float
    parameter_count: int
    estimated_macs: int
    estimated_flops: int


class ImageInfo(BaseModel):
    resolution: Resolution
    format: str
    bytes: int


class InferenceResponse(BaseModel):
    request_id: str
    created_at: datetime
    output_image_base64: str
    model: ModelInfoResponse
    runtime: RuntimeInfo
    input: ImageInfo
    output: ImageInfo


class AnalyticsPoint(BaseModel):
    label: str
    value: float


class EventPreview(BaseModel):
    request_id: str
    created_at: datetime
    input_resolution: Resolution
    output_resolution: Resolution
    latency_ms: float
    upscale: int


class AnalyticsSummaryResponse(BaseModel):
    total_requests: int
    average_latency_ms: float
    latest_request_at: Optional[datetime]
    average_output_megapixels: float
    total_processed_pixels: int
    latency_series: list[AnalyticsPoint]
    upscale_distribution: list[AnalyticsPoint]
    recent_events: list[EventPreview]


class RecentEventsResponse(BaseModel):
    recent_events: list[EventPreview]
