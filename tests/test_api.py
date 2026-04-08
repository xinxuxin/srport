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
