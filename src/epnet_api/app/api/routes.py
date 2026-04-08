from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas.api import (
    AnalyticsSummaryResponse,
    InferenceResponse,
    ModelInfoResponse,
    RecentEventsResponse,
)
from ..services.analytics import AnalyticsService
from ..services.inference import InferenceError, InferenceService

IMAGE_UPLOAD = File(...)
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
    def model_info() -> ModelInfoResponse:
        return inference_service.model_info()

    async def _run_inference(file: UploadFile = IMAGE_UPLOAD) -> InferenceResponse:
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
        try:
            return inference_service.super_resolve(
                image_bytes=image_bytes,
                image_format=format_name,
            )
        except InferenceError as error:
            raise HTTPException(status_code=error.status_code, detail=error.detail) from error

    @router.post("/super-resolve", response_model=InferenceResponse)
    async def super_resolve(file: UploadFile = IMAGE_UPLOAD) -> InferenceResponse:
        return await _run_inference(file)

    @router.post("/infer", response_model=InferenceResponse)
    async def infer(file: UploadFile = IMAGE_UPLOAD) -> InferenceResponse:
        return await _run_inference(file)

    @router.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
    def analytics_summary() -> AnalyticsSummaryResponse:
        return analytics_service.summary()

    @router.get("/usage/summary", response_model=AnalyticsSummaryResponse)
    def usage_summary() -> AnalyticsSummaryResponse:
        return analytics_service.summary()

    @router.get("/usage/recent", response_model=RecentEventsResponse)
    def usage_recent() -> RecentEventsResponse:
        return analytics_service.recent()

    return router
