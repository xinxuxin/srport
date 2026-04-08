from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Resolution(BaseModel):
    width: int
    height: int


class CheckpointOption(BaseModel):
    name: str
    scale: int
    checkpoint_path: Optional[str]
    weights_source: str
    model_version: str


class DeploymentInfo(BaseModel):
    model_version: str
    checkpoint_source: str
    build_time: str
    git_commit: str
    device_target: str
    api_version: str


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
    deployment: DeploymentInfo
    available_checkpoints: list[CheckpointOption]
    supported_methods: list[str]
    supported_scales: list[int]


class RuntimeInfo(BaseModel):
    latency_ms: float
    parameter_count: int
    estimated_macs: int
    estimated_flops: int


class ImageInfo(BaseModel):
    resolution: Resolution
    format: str
    bytes: int


class ComparisonOutput(BaseModel):
    method: str
    label: str
    image_url: str
    format: str
    bytes: int


class InferenceOptionsResponse(BaseModel):
    session_id: str
    method: str
    checkpoint_name: str
    scale: int
    output_format: str
    tile_size: int


class PipelineStageInfo(BaseModel):
    key: str
    label: str
    description: str
    duration_ms: float


class PipelineInfo(BaseModel):
    total_duration_ms: float
    stages: list[PipelineStageInfo]


class InferenceResponse(BaseModel):
    request_id: str
    created_at: datetime
    output_image_url: str
    model: ModelInfoResponse
    runtime: RuntimeInfo
    input: ImageInfo
    output: ImageInfo
    pipeline: PipelineInfo
    options: InferenceOptionsResponse
    input_image_url: str
    comparisons: list[ComparisonOutput]


class AnalyticsPoint(BaseModel):
    label: str
    value: float


class EventPreview(BaseModel):
    request_id: str
    created_at: datetime
    session_id: str
    input_resolution: Resolution
    output_resolution: Resolution
    latency_ms: float
    upscale: int
    method: str
    checkpoint_name: str
    output_format: str
    input_image_url: str
    output_image_url: str


class HistoryEventResponse(BaseModel):
    request_id: str
    created_at: datetime
    session_id: str
    method: str
    checkpoint_name: str
    output_format: str
    tile_size: int
    input_resolution: Resolution
    output_resolution: Resolution
    latency_ms: float
    upscale: int
    parameter_count: int
    estimated_macs: int
    estimated_flops: int
    input_image_url: str
    output_image_url: str


class AnalyticsSummaryResponse(BaseModel):
    total_requests: int
    session_count: int
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    latest_request_at: Optional[datetime]
    average_output_megapixels: float
    total_processed_pixels: int
    latency_series: list[AnalyticsPoint]
    upscale_distribution: list[AnalyticsPoint]
    recent_events: list[EventPreview]


class RecentEventsResponse(BaseModel):
    recent_events: list[EventPreview]


class BatchAggregateResponse(BaseModel):
    total_files: int
    completed_files: int
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_output_megapixels: float


class BatchInferenceResponse(BaseModel):
    session_id: str
    results: list[InferenceResponse]
    aggregate: BatchAggregateResponse
