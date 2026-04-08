from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas.api import AnalyticsSummaryResponse, InferenceResponse, ModelInfoResponse
from ..services.analytics import AnalyticsService
from ..services.inference import InferenceService

IMAGE_UPLOAD = File(...)


def build_router(
    inference_service: InferenceService,
    analytics_service: AnalyticsService,
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/model/info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        return inference_service.model_info()

    @router.post("/super-resolve", response_model=InferenceResponse)
    async def super_resolve(file: UploadFile = IMAGE_UPLOAD) -> InferenceResponse:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image uploads are supported.")
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        format_name = file.content_type.split("/")[-1].upper()
        return inference_service.super_resolve(image_bytes=image_bytes, image_format=format_name)

    @router.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
    def analytics_summary() -> AnalyticsSummaryResponse:
        return analytics_service.summary()

    return router
