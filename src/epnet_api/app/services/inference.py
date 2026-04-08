from __future__ import annotations

import io
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import torch
from PIL import Image, UnidentifiedImageError

from epnet import EPNet, ModelConfig
from epnet.model import load_checkpoint, load_model_from_checkpoint
from epnet.profiling import ModelProfile, profile_model
from epnet.utils import ensure_dir, pil_to_tensor, resize_bicubic, tensor_to_pil

from ..core.settings import Settings
from ..schemas.api import (
    BatchAggregateResponse,
    BatchInferenceResponse,
    CheckpointOption,
    ComparisonOutput,
    DeploymentInfo,
    ImageInfo,
    InferenceOptionsResponse,
    InferenceResponse,
    ModelInfoResponse,
    PipelineInfo,
    PipelineStageInfo,
    Resolution,
    RuntimeInfo,
)
from .analytics import AnalyticsService

SUPPORTED_METHODS = ("epnet", "bicubic", "baseline")
SUPPORTED_SCALES = (2, 3, 4)


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


@dataclass(frozen=True)
class RuntimeModel:
    name: str
    model: EPNet
    profile: ModelProfile
    checkpoint_loaded: bool
    checkpoint_path: Path | None
    weights_source: str
    device_target: str
    scale: int
    model_version: str


@dataclass(frozen=True)
class InferenceError(Exception):
    status_code: int
    detail: str


class InferenceService:
    def __init__(self, settings: Settings, analytics: AnalyticsService) -> None:
        self.settings = settings
        self.analytics = analytics
        self.runtimes = self._load_runtime_models()
        self.default_runtime_name = self._select_default_runtime_name()

    def _load_runtime_models(self) -> dict[str, RuntimeModel]:
        runtimes: dict[str, RuntimeModel] = {}
        checkpoint_dir = self.settings.checkpoint_path.parent

        if checkpoint_dir.exists():
            for checkpoint_path in sorted(checkpoint_dir.glob("*.pt")):
                runtime = self._runtime_from_checkpoint(checkpoint_path)
                runtimes[runtime.name] = runtime

        if runtimes:
            return runtimes

        model = EPNet(ModelConfig())
        model.eval()
        sample = torch.rand(
            1,
            3,
            self.settings.profile_input_size,
            self.settings.profile_input_size,
        )
        profile = profile_model(model, sample)
        random_runtime = RuntimeModel(
            name="random-x4",
            model=model,
            profile=profile,
            checkpoint_loaded=False,
            checkpoint_path=None,
            weights_source="random-initialization",
            device_target=str(next(model.parameters()).device),
            scale=model.config.upscale,
            model_version="epnet-x4-random",
        )
        return {random_runtime.name: random_runtime}

    def _runtime_from_checkpoint(self, checkpoint_path: Path) -> RuntimeModel:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        model = load_model_from_checkpoint(checkpoint)
        ema_state = checkpoint.get("ema_state")
        if isinstance(ema_state, dict):
            model.load_state_dict(ema_state)
        model.eval()
        sample = torch.rand(
            1,
            3,
            self.settings.profile_input_size,
            self.settings.profile_input_size,
        )
        profile = profile_model(model, sample)
        scale = int(model.config.upscale)
        return RuntimeModel(
            name=checkpoint_path.stem,
            model=model,
            profile=profile,
            checkpoint_loaded=True,
            checkpoint_path=checkpoint_path,
            weights_source="checkpoint",
            device_target=str(next(model.parameters()).device),
            scale=scale,
            model_version=f"epnet-x{scale}-{checkpoint_path.stem}",
        )

    def _select_default_runtime_name(self) -> str:
        preferred_name = self.settings.checkpoint_path.stem
        if preferred_name in self.runtimes:
            return preferred_name
        return sorted(self.runtimes.keys())[0]

    def _available_checkpoints(self) -> list[CheckpointOption]:
        return [
            CheckpointOption(
                name=runtime.name,
                scale=runtime.scale,
                checkpoint_path=str(runtime.checkpoint_path) if runtime.checkpoint_path else None,
                weights_source=runtime.weights_source,
                model_version=runtime.model_version,
            )
            for runtime in sorted(self.runtimes.values(), key=lambda item: (item.scale, item.name))
        ]

    def _normalize_method(self, method: str) -> str:
        normalized = method.lower().strip()
        if normalized == "bilinear":
            return "baseline"
        return normalized

    def _resolve_runtime(self, checkpoint_name: str | None, scale: int | None) -> RuntimeModel:
        if checkpoint_name:
            runtime = self.runtimes.get(checkpoint_name)
            if runtime is None:
                raise InferenceError(404, f"Checkpoint '{checkpoint_name}' was not found.")
            if scale is not None and runtime.scale != scale:
                raise InferenceError(
                    400,
                    f"Checkpoint '{checkpoint_name}' supports x{runtime.scale}, not x{scale}.",
                )
            return runtime

        if scale is not None:
            for runtime in self.runtimes.values():
                if runtime.scale == scale:
                    return runtime
            raise InferenceError(404, f"No EPNet checkpoint is available for x{scale}.")

        return self.runtimes[self.default_runtime_name]

    def model_info(self, checkpoint_name: str | None = None) -> ModelInfoResponse:
        runtime = self._resolve_runtime(checkpoint_name=checkpoint_name, scale=None)
        return ModelInfoResponse(
            name="EPNet",
            paper_title=(
                "EPNet: An Efficient Pyramid Network for Enhanced Single-Image "
                "Super-Resolution with Reduced Computational Requirements"
            ),
            checkpoint_loaded=runtime.checkpoint_loaded,
            checkpoint_path=str(runtime.checkpoint_path) if runtime.checkpoint_path else None,
            weights_source=runtime.weights_source,
            upscale=runtime.scale,
            parameter_count=runtime.profile.parameters,
            estimated_macs=runtime.profile.macs,
            estimated_flops=runtime.profile.flops,
            reference_latency_ms=runtime.profile.latency_ms,
            architecture=runtime.model.config.to_dict(),
            deployment=DeploymentInfo(
                model_version=runtime.model_version,
                checkpoint_source=runtime.weights_source,
                build_time=self.settings.build_time,
                git_commit=self.settings.git_commit,
                device_target=runtime.device_target,
                api_version=self.settings.app_version,
            ),
            available_checkpoints=self._available_checkpoints(),
            supported_methods=list(SUPPORTED_METHODS),
            supported_scales=list(SUPPORTED_SCALES),
        )

    def super_resolve(
        self,
        image_bytes: bytes,
        image_format: str = "PNG",
        *,
        session_id: str = "anonymous",
        method: str = "epnet",
        scale: int | None = None,
        output_format: str = "PNG",
        tile_size: int = 0,
        checkpoint_name: str | None = None,
    ) -> InferenceResponse:
        method_name = self._normalize_method(method)
        if method_name not in SUPPORTED_METHODS:
            raise InferenceError(400, f"Unsupported inference method '{method}'.")

        normalized_input_format = self._normalize_output_format(image_format)
        normalized_output_format = self._normalize_output_format(output_format)
        requested_scale = scale or 4
        if requested_scale not in SUPPORTED_SCALES:
            raise InferenceError(400, "Scale must be one of x2, x3, or x4.")

        request_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        total_start = time.perf_counter()

        decode_start = time.perf_counter()
        input_image = self._decode_image(image_bytes)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0

        input_image_url = self._store_artifact(
            request_id=request_id,
            image_bytes=image_bytes,
            image_format=normalized_input_format,
            suffix="input",
        )

        preprocess_start = time.perf_counter()
        runtime = None
        input_tensor: torch.Tensor | None = None
        if method_name == "epnet":
            runtime = self._resolve_runtime(checkpoint_name=checkpoint_name, scale=requested_scale)
            requested_scale = runtime.scale
            input_tensor = pil_to_tensor(input_image).unsqueeze(0)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        infer_start = time.perf_counter()
        if method_name == "epnet":
            if runtime is None or input_tensor is None:
                raise InferenceError(500, "EPNet runtime was not initialized correctly.")
            with torch.inference_mode():
                if tile_size > 0:
                    output_tensor = self._tile_forward(
                        runtime.model,
                        input_tensor,
                        tile_size,
                        runtime.scale,
                    )[0]
                else:
                    output_tensor = runtime.model(input_tensor)[0]
            output_image = tensor_to_pil(output_tensor)
        else:
            output_image = self._baseline_resize(input_image, requested_scale, method_name)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        encode_start = time.perf_counter()
        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format=normalized_output_format)
        output_bytes = output_buffer.getvalue()
        output_image_url = self._store_artifact(
            request_id=request_id,
            image_bytes=output_bytes,
            image_format=normalized_output_format,
            suffix=method_name,
        )
        comparison_outputs = self._build_comparisons(
            request_id=request_id,
            input_image=input_image,
            output_format=normalized_output_format,
            requested_scale=requested_scale,
            primary_method=method_name,
            tile_size=tile_size,
            checkpoint_name=checkpoint_name,
        )
        encode_ms = (time.perf_counter() - encode_start) * 1000.0

        reference_model_info = self.model_info(checkpoint_name=runtime.name if runtime else None)
        total_latency_ms = (time.perf_counter() - total_start) * 1000.0

        response = InferenceResponse(
            request_id=request_id,
            created_at=created_at,
            output_image_url=output_image_url,
            model=reference_model_info,
            runtime=RuntimeInfo(
                latency_ms=total_latency_ms,
                parameter_count=reference_model_info.parameter_count,
                estimated_macs=reference_model_info.estimated_macs,
                estimated_flops=reference_model_info.estimated_flops,
            ),
            input=ImageInfo(
                resolution=Resolution(width=input_image.width, height=input_image.height),
                format=normalized_input_format,
                bytes=len(image_bytes),
            ),
            output=ImageInfo(
                resolution=Resolution(width=output_image.width, height=output_image.height),
                format=normalized_output_format,
                bytes=len(output_bytes),
            ),
            pipeline=PipelineInfo(
                total_duration_ms=total_latency_ms,
                stages=[
                    PipelineStageInfo(
                        key="upload",
                        label="Upload",
                        description="Ingress into the API boundary.",
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
                        description="Convert pixels into the selected runtime representation.",
                        duration_ms=preprocess_ms,
                    ),
                    PipelineStageInfo(
                        key="infer",
                        label="Infer",
                        description="Run EPNet or the selected baseline upsampler.",
                        duration_ms=infer_ms,
                    ),
                    PipelineStageInfo(
                        key="encode",
                        label="Encode",
                        description="Write output artifacts and optional comparison images.",
                        duration_ms=encode_ms,
                    ),
                    PipelineStageInfo(
                        key="log",
                        label="Log",
                        description="Persist usage analytics for replay and dashboards.",
                        duration_ms=0.0,
                    ),
                ],
            ),
            options=InferenceOptionsResponse(
                session_id=session_id,
                method=method_name,
                checkpoint_name=runtime.name if runtime else "baseline",
                scale=requested_scale,
                output_format=normalized_output_format,
                tile_size=tile_size,
            ),
            input_image_url=input_image_url,
            comparisons=comparison_outputs,
        )

        log_start = time.perf_counter()
        self.analytics.log_event(
            request_id=request_id,
            created_at=created_at,
            session_id=session_id,
            input_width=input_image.width,
            input_height=input_image.height,
            output_width=output_image.width,
            output_height=output_image.height,
            input_bytes=len(image_bytes),
            output_bytes=len(output_bytes),
            upscale=requested_scale,
            latency_ms=response.runtime.latency_ms,
            parameter_count=reference_model_info.parameter_count,
            estimated_macs=reference_model_info.estimated_macs,
            estimated_flops=reference_model_info.estimated_flops,
            method=method_name,
            checkpoint_name=runtime.name if runtime else "baseline",
            output_format=normalized_output_format,
            tile_size=tile_size,
            input_artifact_url=input_image_url,
            output_artifact_url=output_image_url,
        )
        response.pipeline.stages[-1].duration_ms = (time.perf_counter() - log_start) * 1000.0
        response.pipeline.total_duration_ms = (
            sum(stage.duration_ms for stage in response.pipeline.stages)
        )
        response.runtime.latency_ms = response.pipeline.total_duration_ms
        return response

    def batch_super_resolve(
        self,
        files: list[tuple[bytes, str]],
        *,
        session_id: str,
        method: str,
        scale: int | None,
        output_format: str,
        tile_size: int,
        checkpoint_name: str | None,
    ) -> BatchInferenceResponse:
        results = [
            self.super_resolve(
                image_bytes=image_bytes,
                image_format=image_format,
                session_id=session_id,
                method=method,
                scale=scale,
                output_format=output_format,
                tile_size=tile_size,
                checkpoint_name=checkpoint_name,
            )
            for image_bytes, image_format in files
        ]

        latencies = [item.runtime.latency_ms for item in results]
        megapixels = [
            (item.output.resolution.width * item.output.resolution.height) / 1_000_000.0
            for item in results
        ]
        return BatchInferenceResponse(
            session_id=session_id,
            results=results,
            aggregate=BatchAggregateResponse(
                total_files=len(files),
                completed_files=len(results),
                average_latency_ms=mean(latencies) if latencies else 0.0,
                p50_latency_ms=_percentile(latencies, 0.5),
                p95_latency_ms=_percentile(latencies, 0.95),
                total_output_megapixels=sum(megapixels),
            ),
        )

    def _build_comparisons(
        self,
        *,
        request_id: str,
        input_image: Image.Image,
        output_format: str,
        requested_scale: int,
        primary_method: str,
        tile_size: int,
        checkpoint_name: str | None,
    ) -> list[ComparisonOutput]:
        comparisons: list[ComparisonOutput] = []
        candidates = [method for method in SUPPORTED_METHODS if method != primary_method]
        for candidate in candidates:
            if candidate == "epnet":
                try:
                    runtime = self._resolve_runtime(
                        checkpoint_name=checkpoint_name,
                        scale=requested_scale,
                    )
                except InferenceError:
                    continue
                tensor = pil_to_tensor(input_image).unsqueeze(0)
                with torch.inference_mode():
                    if tile_size > 0:
                        candidate_tensor = self._tile_forward(
                            runtime.model,
                            tensor,
                            tile_size,
                            runtime.scale,
                        )[0]
                    else:
                        candidate_tensor = runtime.model(tensor)[0]
                candidate_image = tensor_to_pil(candidate_tensor)
            else:
                candidate_image = self._baseline_resize(input_image, requested_scale, candidate)

            candidate_buffer = io.BytesIO()
            candidate_image.save(candidate_buffer, format=output_format)
            candidate_bytes = candidate_buffer.getvalue()
            candidate_url = self._store_artifact(
                request_id=request_id,
                image_bytes=candidate_bytes,
                image_format=output_format,
                suffix=candidate,
            )
            comparisons.append(
                ComparisonOutput(
                    method=candidate,
                    label=(
                        "EPNet"
                        if candidate == "epnet"
                        else "Baseline"
                        if candidate == "baseline"
                        else candidate.title()
                    ),
                    image_url=candidate_url,
                    format=output_format,
                    bytes=len(candidate_bytes),
                )
            )
        return comparisons

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

    def _baseline_resize(self, image: Image.Image, scale: int, method: str) -> Image.Image:
        size = (image.width * scale, image.height * scale)
        if method == "bicubic":
            return resize_bicubic(image, size)
        if method == "baseline":
            return image.resize(size, resample=Image.Resampling.BILINEAR)
        raise InferenceError(400, f"Unsupported baseline method '{method}'.")

    def _tile_forward(
        self,
        model: EPNet,
        tensor: torch.Tensor,
        tile_size: int,
        scale: int,
        overlap: int = 8,
    ) -> torch.Tensor:
        batch, channels, height, width = tensor.shape
        if batch != 1:
            raise InferenceError(400, "Tile inference currently supports a batch size of 1.")

        output = torch.zeros(batch, channels, height * scale, width * scale)
        weight = torch.zeros_like(output)

        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = min(top + tile_size, height)
                right = min(left + tile_size, width)
                top_pad = max(top - overlap, 0)
                left_pad = max(left - overlap, 0)
                bottom_pad = min(bottom + overlap, height)
                right_pad = min(right + overlap, width)

                tile = tensor[:, :, top_pad:bottom_pad, left_pad:right_pad]
                tile_output = model(tile)

                crop_top = (top - top_pad) * scale
                crop_left = (left - left_pad) * scale
                crop_bottom = crop_top + (bottom - top) * scale
                crop_right = crop_left + (right - left) * scale

                out_top = top * scale
                out_left = left * scale
                out_bottom = bottom * scale
                out_right = right * scale

                output[:, :, out_top:out_bottom, out_left:out_right] += tile_output[
                    :, :, crop_top:crop_bottom, crop_left:crop_right
                ]
                weight[:, :, out_top:out_bottom, out_left:out_right] += 1

        return output / weight.clamp_min(1)

    def _normalize_output_format(self, image_format: str) -> str:
        normalized = image_format.upper()
        if normalized == "JPG":
            return "JPEG"
        if normalized in {"PNG", "JPEG", "WEBP", "BMP"}:
            return normalized
        return "PNG"

    def _store_artifact(
        self,
        *,
        request_id: str,
        image_bytes: bytes,
        image_format: str,
        suffix: str,
    ) -> str:
        extension_map = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp"}
        extension = extension_map.get(image_format, "png")
        ensure_dir(self.settings.artifacts_dir)
        artifact_path = self.settings.artifacts_dir / f"{request_id}_{suffix}.{extension}"
        artifact_path.write_bytes(image_bytes)
        return f"{self.settings.artifacts_url_prefix}/{artifact_path.name}"
