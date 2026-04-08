from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from ..schemas.api import (
    AnalyticsSummaryResponse,
    BatchInferenceResponse,
    HistoryEventResponse,
    InferenceResponse,
    ModelInfoResponse,
    RecentEventsResponse,
)
from ..services.analytics import AnalyticsService
from ..services.inference import InferenceError, InferenceService

IMAGE_UPLOAD = File(...)
IMAGE_UPLOADS = File(...)
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}


def build_router(
    inference_service: InferenceService,
    analytics_service: AnalyticsService,
) -> APIRouter:
    router = APIRouter()
    settings = inference_service.settings

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/model/info", response_model=ModelInfoResponse)
    def model_info(checkpoint_name: Optional[str] = Query(default=None)) -> ModelInfoResponse:
        return inference_service.model_info(checkpoint_name=checkpoint_name)

    async def _validate_upload(file: UploadFile) -> tuple[bytes, str]:
        if not file.content_type or file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail="Only image uploads are supported.")
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(image_bytes) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds the {settings.max_upload_bytes} byte limit.",
            )
        format_name = file.content_type.split("/")[-1].upper()
        return image_bytes, format_name

    async def _run_inference(
        file: UploadFile = IMAGE_UPLOAD,
        *,
        session_id: str = "anonymous",
        method: str = "epnet",
        scale: Optional[int] = None,
        output_format: str = "PNG",
        tile_size: int = 0,
        checkpoint_name: Optional[str] = None,
    ) -> InferenceResponse:
        image_bytes, format_name = await _validate_upload(file)
        try:
            return inference_service.super_resolve(
                image_bytes=image_bytes,
                image_format=format_name,
                session_id=session_id,
                method=method,
                scale=scale,
                output_format=output_format,
                tile_size=tile_size,
                checkpoint_name=checkpoint_name,
            )
        except InferenceError as error:
            raise HTTPException(status_code=error.status_code, detail=error.detail) from error

    @router.post("/super-resolve", response_model=InferenceResponse)
    async def super_resolve(
        file: UploadFile = IMAGE_UPLOAD,
        session_id: Annotated[str, Form()] = "anonymous",
        method: Annotated[str, Form()] = "epnet",
        scale: Annotated[Optional[int], Form()] = None,
        output_format: Annotated[str, Form()] = "PNG",
        tile_size: Annotated[int, Form()] = 0,
        checkpoint_name: Annotated[Optional[str], Form()] = None,
    ) -> InferenceResponse:
        return await _run_inference(
            file,
            session_id=session_id,
            method=method,
            scale=scale,
            output_format=output_format,
            tile_size=tile_size,
            checkpoint_name=checkpoint_name,
        )

    @router.post("/infer", response_model=InferenceResponse)
    async def infer(
        file: UploadFile = IMAGE_UPLOAD,
        session_id: Annotated[str, Form()] = "anonymous",
        method: Annotated[str, Form()] = "epnet",
        scale: Annotated[Optional[int], Form()] = None,
        output_format: Annotated[str, Form()] = "PNG",
        tile_size: Annotated[int, Form()] = 0,
        checkpoint_name: Annotated[Optional[str], Form()] = None,
    ) -> InferenceResponse:
        return await _run_inference(
            file,
            session_id=session_id,
            method=method,
            scale=scale,
            output_format=output_format,
            tile_size=tile_size,
            checkpoint_name=checkpoint_name,
        )

    @router.post("/infer/batch", response_model=BatchInferenceResponse)
    async def infer_batch(
        files: list[UploadFile] = IMAGE_UPLOADS,
        session_id: Annotated[str, Form()] = "anonymous",
        method: Annotated[str, Form()] = "epnet",
        scale: Annotated[Optional[int], Form()] = None,
        output_format: Annotated[str, Form()] = "PNG",
        tile_size: Annotated[int, Form()] = 0,
        checkpoint_name: Annotated[Optional[str], Form()] = None,
    ) -> BatchInferenceResponse:
        if not files:
            raise HTTPException(status_code=400, detail="At least one file is required.")

        validated_files: list[tuple[bytes, str]] = []
        for file in files:
            validated_files.append(await _validate_upload(file))

        try:
            return inference_service.batch_super_resolve(
                validated_files,
                session_id=session_id,
                method=method,
                scale=scale,
                output_format=output_format,
                tile_size=tile_size,
                checkpoint_name=checkpoint_name,
            )
        except InferenceError as error:
            raise HTTPException(status_code=error.status_code, detail=error.detail) from error

    @router.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
    def analytics_summary() -> AnalyticsSummaryResponse:
        return analytics_service.summary()

    @router.get("/usage/summary", response_model=AnalyticsSummaryResponse)
    def usage_summary() -> AnalyticsSummaryResponse:
        return analytics_service.summary()

    @router.get("/usage/recent", response_model=RecentEventsResponse)
    def usage_recent(limit: int = Query(default=10, ge=1, le=50)) -> RecentEventsResponse:
        return analytics_service.recent(limit=limit)

    @router.get("/history/{request_id}", response_model=HistoryEventResponse)
    def history_event(request_id: str) -> HistoryEventResponse:
        event = analytics_service.history_event(request_id)
        if event is None:
            raise HTTPException(status_code=404, detail="History record was not found.")
        return event

    return router
