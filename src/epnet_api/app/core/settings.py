from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _trained_run_dir() -> Path:
    return _repo_root() / "outputs" / "run_x4_edge_default"


def _default_build_time() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_git_commit() -> str:
    env_commit = os.getenv("EPNET_GIT_COMMIT")
    if env_commit:
        return env_commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _default_checkpoint_path() -> Path:
    trained_inference = _trained_run_dir() / "inference.pt"
    if trained_inference.exists():
        return trained_inference
    return _repo_root() / "checkpoints" / "demo_x4.pt"


def _default_checkpoint_dir() -> Path | None:
    trained_dir = _trained_run_dir()
    if trained_dir.exists() and any(trained_dir.glob("*.pt")):
        return trained_dir
    return None


def _default_onnx_model_path() -> Path:
    return _trained_run_dir() / "model.onnx"


@dataclass(frozen=True)
class Settings:
    app_name: str = "EPNet Demo API"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    checkpoint_path: Path = _default_checkpoint_path()
    checkpoint_dir: Path | None = _default_checkpoint_dir()
    inference_backend: str = "pytorch"
    onnx_model_path: Path | None = _default_onnx_model_path()
    analytics_db_path: Path = _repo_root() / "outputs" / "usage_analytics.sqlite3"
    artifacts_dir: Path = _repo_root() / "outputs" / "generated"
    artifacts_url_prefix: str = "/artifacts"
    max_upload_bytes: int = 12 * 1024 * 1024
    max_image_pixels: int = 16_000_000
    build_time: str = field(default_factory=_default_build_time)
    git_commit: str = field(default_factory=_detect_git_commit)
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    profile_input_size: int = 64

    @staticmethod
    def from_env() -> Settings:
        raw_origins = os.getenv("EPNET_CORS_ORIGINS")
        default_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
        origins = (
            [origin.strip() for origin in raw_origins.split(",")]
            if raw_origins
            else default_origins
        )
        checkpoint = Path(
            os.getenv(
                "EPNET_CHECKPOINT_PATH",
                str(_default_checkpoint_path()),
            )
        )
        checkpoint_dir_env = os.getenv("EPNET_CHECKPOINT_DIR")
        checkpoint_dir = (
            Path(checkpoint_dir_env)
            if checkpoint_dir_env
            else _default_checkpoint_dir()
        )
        inference_backend = os.getenv("EPNET_INFERENCE_BACKEND", "pytorch").strip().lower()
        onnx_model_path_env = os.getenv("EPNET_ONNX_MODEL_PATH")
        onnx_model_path = (
            Path(onnx_model_path_env)
            if onnx_model_path_env
            else _default_onnx_model_path()
        )
        database = Path(
            os.getenv(
                "EPNET_ANALYTICS_DB_PATH",
                str(_repo_root() / "outputs" / "usage_analytics.sqlite3"),
            )
        )
        artifacts_dir = Path(
            os.getenv(
                "EPNET_ARTIFACTS_DIR",
                str(_repo_root() / "outputs" / "generated"),
            )
        )
        max_upload_bytes = int(os.getenv("EPNET_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
        max_image_pixels = int(os.getenv("EPNET_MAX_IMAGE_PIXELS", "16000000"))
        profile_size = int(os.getenv("EPNET_PROFILE_INPUT_SIZE", "64"))
        if inference_backend not in {"pytorch", "onnx"}:
            raise ValueError("EPNET_INFERENCE_BACKEND must be either 'pytorch' or 'onnx'.")
        return Settings(
            checkpoint_path=checkpoint,
            checkpoint_dir=checkpoint_dir,
            inference_backend=inference_backend,
            onnx_model_path=onnx_model_path,
            analytics_db_path=database,
            artifacts_dir=artifacts_dir,
            max_upload_bytes=max_upload_bytes,
            max_image_pixels=max_image_pixels,
            build_time=os.getenv("EPNET_BUILD_TIME", _default_build_time()),
            git_commit=_detect_git_commit(),
            cors_origins=origins,
            profile_input_size=profile_size,
        )
