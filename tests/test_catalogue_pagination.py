from pathlib import Path

from fastapi.testclient import TestClient

from archgpu_ollama_bridge.hf_index import model_matches_query, paginate_models, sort_models
from archgpu_ollama_bridge.main import create_app
from archgpu_ollama_bridge.registry import ModelRegistry
from archgpu_ollama_bridge.services import AppServices
from tests.test_ollama_adapter import FakeProxyClient, FakeRouter


def build_test_client() -> TestClient:
    registry = ModelRegistry.load(Path("config/models.yaml"))
    services = AppServices(
        registry=registry,
        router=FakeRouter(registry),
        proxy_client=FakeProxyClient(),
    )
    return TestClient(create_app(services))


def _sample_models() -> list[dict]:
    return [
        {
            "alias": "alpha",
            "display_name": "Alpha",
            "repo": "org/alpha",
            "capabilities": ["chat"],
            "metadata": {"publisher": "org", "downloads": 100, "likes": 5, "source": "static_catalogue"},
            "runtime_fit": {"recommendation": "recommended"},
            "installed": False,
            "downloadable": True,
        },
        {
            "alias": "beta",
            "display_name": "Beta",
            "repo": "org/beta",
            "capabilities": ["coding"],
            "metadata": {"publisher": "org", "downloads": 200, "likes": 10, "source": "live_discovery"},
            "runtime_fit": {"recommendation": "possible"},
            "installed": True,
            "downloadable": False,
        },
        {
            "alias": "gamma",
            "display_name": "Gamma",
            "repo": "other/gamma",
            "capabilities": ["reasoning"],
            "metadata": {"publisher": "other", "downloads": 50, "likes": 1, "source": "local_registry"},
            "runtime_fit": {"recommendation": "not_recommended"},
            "installed": False,
            "downloadable": True,
        },
    ]


def test_model_matches_query_field_prefixes() -> None:
    model = _sample_models()[1]
    assert model_matches_query(model, "repo:org/beta")
    assert model_matches_query(model, "publisher:org")
    assert model_matches_query(model, "capability:coding")
    assert model_matches_query(model, "source:live")
    assert model_matches_query(model, "fit:possible")
    assert not model_matches_query(model, "repo:missing")


def test_sort_models_by_downloads_desc() -> None:
    models = _sample_models()
    sorted_models = sort_models(models, "downloads", "desc")
    assert [m["alias"] for m in sorted_models] == ["beta", "alpha", "gamma"]


def test_paginate_models_envelope() -> None:
    envelope = paginate_models(_sample_models(), page=2, page_size=1)
    assert envelope["total"] == 3
    assert envelope["page"] == 2
    assert envelope["page_size"] == 1
    assert envelope["total_pages"] == 3
    assert envelope["has_prev"] is True
    assert envelope["has_next"] is True
    assert len(envelope["models"]) == 1


def test_catalogue_backward_compatible_without_pagination() -> None:
    client = build_test_client()
    response = client.get("/api/catalogue?live=false")
    assert response.status_code == 200
    body = response.json()
    assert "models" in body
    assert "system_profile" in body
    assert body["total"] == len(body["models"])
    assert body["page"] == 1
    assert body["has_prev"] is False
    assert body["has_next"] is False
    assert "live_index" in body


def test_catalogue_pagination_params() -> None:
    client = build_test_client()
    response = client.get("/api/catalogue?live=false&page=1&page_size=1")
    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 1
    assert body["page_size"] == 1
    assert body["total"] >= 1
    assert body["total_pages"] >= 1
    assert len(body["models"]) == 1


def test_catalogue_search_and_installed_filter() -> None:
    client = build_test_client()
    response = client.get("/api/catalogue?live=false&q=code&installed=false")
    assert response.status_code == 200
    body = response.json()
    assert all(not m.get("installed") for m in body["models"])
