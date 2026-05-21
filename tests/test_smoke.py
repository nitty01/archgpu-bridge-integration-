from fastapi.testclient import TestClient

from archgpu_ollama_bridge.main import create_app


def test_healthcheck_returns_expected_payload() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "environment": "development",
    }
