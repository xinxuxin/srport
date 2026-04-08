from __future__ import annotations

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
from epnet.utils import ensure_dir, pil_to_tensor, tensor_to_pil

from ..core.settings import Settings
from ..schemas.api import (
    DeploymentInfo,
    ImageInfo,
    InferenceResponse,
    ModelInfoResponse,
    PipelineInfo,
    PipelineStageInfo,
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
    device_target: str


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
            device_target=str(next(model.parameters()).device),
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
            deployment=DeploymentInfo(
                model_version=f"epnet-x{runtime.model.config.upscale}-v1",
                checkpoint_source=runtime.weights_source,
                build_time=self.settings.build_time,
                git_commit=self.settings.git_commit,
                device_target=runtime.device_target,
                api_version=self.settings.app_version,
            ),
        )

    def super_resolve(self, image_bytes: bytes, image_format: str = "PNG") -> InferenceResponse:
        request_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        total_start = time.perf_counter()

        decode_start = time.perf_counter()
        input_image = self._decode_image(image_bytes)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0

        preprocess_start = time.perf_counter()
        input_tensor = pil_to_tensor(input_image).unsqueeze(0)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        infer_start = time.perf_counter()
        with torch.inference_mode():
            output_tensor = self.runtime.model(input_tensor)[0]
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        postprocess_start = time.perf_counter()
        output_image = tensor_to_pil(output_tensor)
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

        encode_start = time.perf_counter()
        output_buffer = io.BytesIO()
        normalized_format = self._normalize_output_format(image_format)
        output_image.save(output_buffer, format=normalized_format)
        output_bytes = output_buffer.getvalue()
        encode_ms = (time.perf_counter() - encode_start) * 1000.0

        write_start = time.perf_counter()
        output_image_url = self._store_output_artifact(
            request_id=request_id,
            image_bytes=output_bytes,
            image_format=normalized_format,
        )
        write_ms = (time.perf_counter() - write_start) * 1000.0

        model_info = self.model_info()
        response = InferenceResponse(
            request_id=request_id,
            created_at=created_at,
            output_image_url=output_image_url,
            model=model_info,
            runtime=RuntimeInfo(
                latency_ms=0.0,
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
                format=normalized_format,
                bytes=len(output_bytes),
            ),
            pipeline=PipelineInfo(
                total_duration_ms=0.0,
                stages=[
                    PipelineStageInfo(
                        key="upload",
                        label="Upload",
                        description="Client transfer to API boundary.",
                        duration_ms=0.0,
                    ),
                    PipelineStageInfo(
                        key="validate",
                        label="Validate",
                        description="MIME, byte-size, and pixel guardrails.",
                        duration_ms=0.0,
                    ),
                    PipelineStageInfo(
                        key="decode",
                        label="Decode",
                        description="Load the source image into RGB pixels.",
                        duration_ms=decode_ms,
                    ),
                    PipelineStageInfo(
                        key="preprocess",
                        label="Preprocess",
                        description="Convert pixels into EPNet tensor input.",
                        duration_ms=preprocess_ms,
                    ),
                    PipelineStageInfo(
                        key="infer",
                        label="Infer",
                        description="Run PFEM, ESPM, and reconstruction.",
                        duration_ms=infer_ms,
                    ),
                    PipelineStageInfo(
                        key="encode",
                        label="Encode",
                        description="Serialize the super-resolved output.",
                        duration_ms=encode_ms + postprocess_ms + write_ms,
                    ),
                    PipelineStageInfo(
                        key="log",
                        label="Log",
                        description="Persist telemetry and usage analytics.",
                        duration_ms=0.0,
                    ),
                ],
            ),
        )
        log_start = time.perf_counter()
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
        log_duration_ms = (time.perf_counter() - log_start) * 1000.0
        response.pipeline.stages[-1].duration_ms = log_duration_ms
        total_latency_ms = (time.perf_counter() - total_start) * 1000.0
        response.runtime.latency_ms = total_latency_ms
        response.pipeline.total_duration_ms = total_latency_ms
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

    def _store_output_artifact(self, request_id: str, image_bytes: bytes, image_format: str) -> str:
        extension_map = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp"}
        extension = extension_map.get(image_format, "png")
        ensure_dir(self.settings.artifacts_dir)
        artifact_path = self.settings.artifacts_dir / f"{request_id}.{extension}"
        artifact_path.write_bytes(image_bytes)
        return f"{self.settings.artifacts_url_prefix}/{artifact_path.name}"
