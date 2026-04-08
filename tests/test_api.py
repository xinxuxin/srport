from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from epnet_api.app.main import create_app  # noqa: E402


def test_health_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_super_resolve_endpoint_returns_metadata() -> None:
    client = TestClient(create_app())
    image = Image.new("RGB", (8, 8), color=(120, 30, 220))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    response = client.post(
        "/api/v1/super-resolve",
        files={"file": ("sample.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["input"]["resolution"] == {"width": 8, "height": 8}
    assert payload["output"]["resolution"] == {"width": 32, "height": 32}
    assert payload["model"]["name"] == "EPNet"


def test_infer_alias_and_usage_routes_work() -> None:
    client = TestClient(create_app())
    image = Image.new("RGB", (8, 8), color=(20, 60, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    infer_response = client.post(
        "/api/v1/infer",
        files={"file": ("sample.png", buffer.getvalue(), "image/png")},
    )
    assert infer_response.status_code == 200

    summary_response = client.get("/api/v1/usage/summary")
    recent_response = client.get("/api/v1/usage/recent")
    assert summary_response.status_code == 200
    assert recent_response.status_code == 200
    assert summary_response.json()["total_requests"] >= 1
    assert len(recent_response.json()["recent_events"]) >= 1


def test_invalid_image_bytes_return_friendly_error() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/infer",
        files={"file": ("fake.png", b"not really an image", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image."


def test_oversized_upload_is_rejected() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/infer",
        files={"file": ("huge.png", b"x" * (12 * 1024 * 1024 + 1), "image/png")},
    )
    assert response.status_code == 413
