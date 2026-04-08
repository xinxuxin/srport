from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class Settings:
    app_name: str = "EPNet Demo API"
    api_prefix: str = "/api/v1"
    checkpoint_path: Path = _repo_root() / "checkpoints" / "demo_x4.pt"
    analytics_db_path: Path = _repo_root() / "outputs" / "usage_analytics.sqlite3"
    max_upload_bytes: int = 12 * 1024 * 1024
    max_image_pixels: int = 16_000_000
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
                str(_repo_root() / "checkpoints" / "demo_x4.pt"),
            )
        )
        database = Path(
            os.getenv(
                "EPNET_ANALYTICS_DB_PATH",
                str(_repo_root() / "outputs" / "usage_analytics.sqlite3"),
            )
        )
        max_upload_bytes = int(os.getenv("EPNET_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
        max_image_pixels = int(os.getenv("EPNET_MAX_IMAGE_PIXELS", "16000000"))
        profile_size = int(os.getenv("EPNET_PROFILE_INPUT_SIZE", "64"))
        return Settings(
            checkpoint_path=checkpoint,
            analytics_db_path=database,
            max_upload_bytes=max_upload_bytes,
            max_image_pixels=max_image_pixels,
            cors_origins=origins,
            profile_input_size=profile_size,
        )
