from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.routes import build_router
from .core.settings import Settings
from .db.database import Database
from .services.analytics import AnalyticsService
from .services.inference import InferenceService


def create_app() -> FastAPI:
    settings = Settings.from_env()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    database = Database(settings.analytics_db_path)
    analytics_service = AnalyticsService(database)
    inference_service = InferenceService(settings, analytics_service)

    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount(
        settings.artifacts_url_prefix,
        StaticFiles(directory=settings.artifacts_dir),
        name="artifacts",
    )
    app.include_router(
        build_router(inference_service, analytics_service),
        prefix=settings.api_prefix,
    )
    return app


app = create_app()
