import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
from fastapi.testclient import TestClient


def test_server_starts():
    """Server can be imported and creates TestClient."""
    from fastapi_server import app

    client = TestClient(app)
    assert client is not None


def test_api_logs_returns_json():
    """GET /api/logs returns JSON with files key."""
    from fastapi_server import app

    client = TestClient(app)
    response = client.get("/api/logs")
    assert response.status_code == 200
    data = response.json()
    assert "files" in data
    assert isinstance(data["files"], list)


def test_logs_endpoint_404_for_missing_file():
    """GET /logs/nonexistent returns 404."""
    from fastapi_server import app

    client = TestClient(app)
    response = client.get("/logs/nonexistent_file_xyz.jsonl")
    assert response.status_code == 404
