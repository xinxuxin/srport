from __future__ import annotations

import base64
import io
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from epnet import EPNet, ModelConfig
from epnet.model import load_checkpoint, load_model_from_checkpoint
from epnet.profiling import ModelProfile, profile_model
from epnet.utils import pil_to_tensor, tensor_to_pil

from ..core.settings import Settings
from ..schemas.api import (
    ImageInfo,
    InferenceResponse,
    ModelInfoResponse,
    Resolution,
    RuntimeInfo,
)
from .analytics import AnalyticsService


@dataclass(frozen=True)
class RuntimeModel:
    model: EPNet
    profile: ModelProfile
    checkpoint_loaded: bool
    checkpoint_path: Path | None
    weights_source: str


@dataclass(frozen=True)
class InferenceError(Exception):
    status_code: int
    detail: str


class InferenceService:
    def __init__(self, settings: Settings, analytics: AnalyticsService) -> None:
        self.settings = settings
        self.analytics = analytics
        self.runtime = self._load_runtime_model()

    def _load_runtime_model(self) -> RuntimeModel:
        checkpoint_candidate = self.settings.checkpoint_path
        checkpoint_path: Path | None
        if checkpoint_candidate.exists():
            checkpoint = load_checkpoint(checkpoint_candidate, map_location="cpu")
            model = load_model_from_checkpoint(checkpoint)
            ema_state = checkpoint.get("ema_state")
            if isinstance(ema_state, dict):
                model.load_state_dict(ema_state)
            checkpoint_loaded = True
            checkpoint_path = checkpoint_candidate
            weights_source = "checkpoint"
        else:
            model = EPNet(ModelConfig())
            checkpoint_loaded = False
            checkpoint_path = None
            weights_source = "random-initialization"

        model.eval()
        sample = torch.rand(
            1,
            3,
            self.settings.profile_input_size,
            self.settings.profile_input_size,
        )
        profile = profile_model(model, sample)
        return RuntimeModel(
            model=model,
            profile=profile,
            checkpoint_loaded=checkpoint_loaded,
            checkpoint_path=checkpoint_path,
            weights_source=weights_source,
        )

    def model_info(self) -> ModelInfoResponse:
        runtime = self.runtime
        return ModelInfoResponse(
            name="EPNet",
            paper_title=(
                "EPNet: An Efficient Pyramid Network for Enhanced Single-Image "
                "Super-Resolution with Reduced Computational Requirements"
            ),
            checkpoint_loaded=runtime.checkpoint_loaded,
            checkpoint_path=str(runtime.checkpoint_path) if runtime.checkpoint_path else None,
            weights_source=runtime.weights_source,
            upscale=runtime.model.config.upscale,
            parameter_count=runtime.profile.parameters,
            estimated_macs=runtime.profile.macs,
            estimated_flops=runtime.profile.flops,
            reference_latency_ms=runtime.profile.latency_ms,
            architecture=runtime.model.config.to_dict(),
        )

    def super_resolve(self, image_bytes: bytes, image_format: str = "PNG") -> InferenceResponse:
        request_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        input_image = self._decode_image(image_bytes)
        input_tensor = pil_to_tensor(input_image).unsqueeze(0)
        start = time.perf_counter()
        with torch.inference_mode():
            output_tensor = self.runtime.model(input_tensor)[0]
        latency_ms = (time.perf_counter() - start) * 1000.0
        output_image = tensor_to_pil(output_tensor)

        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format=self._normalize_output_format(image_format))
        output_bytes = output_buffer.getvalue()
        base64_image = base64.b64encode(output_bytes).decode("utf-8")

        model_info = self.model_info()
        response = InferenceResponse(
            request_id=request_id,
            created_at=created_at,
            output_image_base64=base64_image,
            model=model_info,
            runtime=RuntimeInfo(
                latency_ms=latency_ms,
                parameter_count=model_info.parameter_count,
                estimated_macs=model_info.estimated_macs,
                estimated_flops=model_info.estimated_flops,
            ),
            input=ImageInfo(
                resolution=Resolution(width=input_image.width, height=input_image.height),
                format=image_format,
                bytes=len(image_bytes),
            ),
            output=ImageInfo(
                resolution=Resolution(width=output_image.width, height=output_image.height),
                format=image_format,
                bytes=len(output_bytes),
            ),
        )
        self.analytics.log_event(
            request_id=request_id,
            created_at=created_at,
            input_width=input_image.width,
            input_height=input_image.height,
            output_width=output_image.width,
            output_height=output_image.height,
            input_bytes=len(image_bytes),
            output_bytes=len(output_bytes),
            upscale=model_info.upscale,
            latency_ms=response.runtime.latency_ms,
            parameter_count=model_info.parameter_count,
            estimated_macs=model_info.estimated_macs,
            estimated_flops=model_info.estimated_flops,
        )
        return response

    def _decode_image(self, image_bytes: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except (UnidentifiedImageError, OSError) as error:
            raise InferenceError(
                status_code=400,
                detail="Uploaded file is not a valid image.",
            ) from error

        if image.width * image.height > self.settings.max_image_pixels:
            raise InferenceError(
                status_code=413,
                detail=(
                    f"Image is too large. Limit is {self.settings.max_image_pixels} total pixels."
                ),
            )

        return image.convert("RGB")

    def _normalize_output_format(self, image_format: str) -> str:
        normalized = image_format.upper()
        if normalized == "JPG":
            return "JPEG"
        if normalized in {"PNG", "JPEG", "WEBP", "BMP"}:
            return normalized
        return "PNG"
