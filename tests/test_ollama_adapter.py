import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi.testclient import TestClient

from archgpu_ollama_bridge.lifecycle import BackendHandle
from archgpu_ollama_bridge.main import create_app
from archgpu_ollama_bridge.registry import ModelRegistry
from archgpu_ollama_bridge.router import RouteTarget
from archgpu_ollama_bridge.services import AppServices, ProxyResponse


class FakeRouter:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.requests: list[str] = []

    async def route(self, identifier: str) -> RouteTarget:
        self.requests.append(identifier)
        model = self.registry.get(identifier)
        return RouteTarget(
            model=model,
            handle=BackendHandle(
                model_id=model.id,
                backend_id=f"llama-{model.id}",
                port=model.port,
            ),
            base_url=f"http://127.0.0.1:{model.port}",
        )


class FakeProxyClient:
    def __init__(self, sse_chunks: list[bytes] | None = None) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.stream_calls: list[tuple[str, str, dict]] = []
        self._sse_chunks = sse_chunks or []

    async def post_json(self, base_url: str, path: str, payload: dict) -> ProxyResponse:
        self.calls.append((base_url, path, payload))
        return ProxyResponse(
            status_code=200,
            json_body={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "translated response",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async def stream_post(self, base_url: str, path: str, payload: dict) -> AsyncIterator[bytes]:
        self.stream_calls.append((base_url, path, payload))
        for chunk in self._sse_chunks:
            yield chunk


def build_test_client(sse_chunks: list[bytes] | None = None) -> tuple[TestClient, FakeRouter, FakeProxyClient]:
    registry = ModelRegistry.load(Path("config/models.yaml"))
    router = FakeRouter(registry)
    proxy_client = FakeProxyClient(sse_chunks=sse_chunks)
    services = AppServices(
        registry=registry,
        router=router,
        proxy_client=proxy_client,
    )
    return TestClient(create_app(services)), router, proxy_client


def test_ollama_tags_lists_registered_models() -> None:
    client, _, _ = build_test_client()

    response = client.get("/api/tags")

    assert response.status_code == 200
    assert response.json()["models"] == [
        {
            "name": "mistral",
            "model": "mistral",
            "details": {"format": "gguf", "families": ["general"]},
        },
        {
            "name": "code",
            "model": "code",
            "details": {"format": "gguf", "families": ["code"]},
        },
        {
            "name": "finance",
            "model": "finance",
            "details": {"format": "gguf", "families": ["finance"]},
        },
    ]


def test_ollama_chat_translates_to_openai_chat_completions() -> None:
    client, router, proxy_client = build_test_client()

    response = client.post(
        "/api/chat",
        json={
            "model": "code",
            "messages": [{"role": "user", "content": "Explain the repo"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["model"] == "code"
    assert response.json()["message"]["content"] == "translated response"
    assert router.requests == ["code"]
    assert proxy_client.calls == [
        (
            "http://127.0.0.1:8081",
            "/v1/chat/completions",
            {
                "model": "code.gguf",
                "messages": [{"role": "user", "content": "Explain the repo"}],
                "stream": False,
            },
        )
    ]


def test_ollama_generate_translates_prompt_to_chat_completion() -> None:
    client, _, proxy_client = build_test_client()

    response = client.post(
        "/api/generate",
        json={
            "model": "finance",
            "prompt": "Summarize quarterly risk.",
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["model"] == "finance"
    assert response.json()["response"] == "translated response"
    assert proxy_client.calls == [
        (
            "http://127.0.0.1:8082",
            "/v1/chat/completions",
            {
                "model": "finance.gguf",
                "messages": [{"role": "user", "content": "Summarize quarterly risk."}],
                "stream": False,
            },
        )
    ]


def _sse(chunks: list[dict]) -> list[bytes]:
    return [(f"data: {json.dumps(c)}\n\n").encode("utf-8") for c in chunks] + [b"data: [DONE]\n\n"]


def test_ollama_chat_stream_translates_sse_to_ndjson() -> None:
    sse_chunks = _sse(
        [
            {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
        ]
    )
    client, _, proxy_client = build_test_client(sse_chunks=sse_chunks)

    response = client.post(
        "/api/chat",
        json={
            "model": "code",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")

    lines = [json.loads(line) for line in response.text.splitlines() if line]
    contents = [evt["message"]["content"] for evt in lines if not evt["done"]]
    assert contents == ["Hello", " world"]
    assert lines[-1]["done"] is True
    assert lines[-1]["done_reason"] == "stop"

    assert proxy_client.stream_calls[0][2]["stream"] is True


def test_ollama_show_returns_model_metadata() -> None:
    client, _, _ = build_test_client()

    response = client.post("/api/show", json={"name": "code"})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "code"
    assert body["details"]["families"] == ["code"]
    assert body["model_info"]["context_length"] == 8192


def test_ollama_show_returns_404_for_unknown_model() -> None:
    client, _, _ = build_test_client()

    response = client.post("/api/show", json={"name": "missing"})
    assert response.status_code == 404


def test_ollama_show_requires_name() -> None:
    client, _, _ = build_test_client()
    response = client.post("/api/show", json={})
    assert response.status_code == 422


def test_ollama_ps_returns_empty_when_no_lifecycle() -> None:
    client, _, _ = build_test_client()
    response = client.get("/api/ps")
    assert response.status_code == 200
    assert response.json() == {"models": []}


def test_ollama_ps_lists_loaded_models(tmp_path: Path) -> None:
    import asyncio

    from archgpu_ollama_bridge.lifecycle import (
        BackendHandle as _BH,
        LifecycleManager,
    )
    from archgpu_ollama_bridge.state import RuntimeStateStore

    class _Backend:
        def start(self, model):
            return _BH(model_id=model.id, backend_id=f"llama-{model.id}", port=model.port)

        def stop(self, handle):
            return None

        def is_healthy(self, handle):
            return True

    registry = ModelRegistry.load(Path("config/models.yaml"))
    store = RuntimeStateStore(tmp_path / "state.json")
    lifecycle = LifecycleManager(registry, store, _Backend(), max_loaded_models=8)
    asyncio.run(lifecycle.ensure_loaded("code"))

    services = AppServices(
        registry=registry,
        router=FakeRouter(registry),
        proxy_client=FakeProxyClient(),
        lifecycle=lifecycle,
    )
    client = TestClient(create_app(services))

    response = client.get("/api/ps")
    assert response.status_code == 200
    body = response.json()
    assert [m["model"] for m in body["models"]] == ["code"]
    assert body["models"][0]["last_used_at"] is not None


def test_ollama_pull_without_pull_manager_returns_503() -> None:
    client, _, _ = build_test_client()
    response = client.post("/api/pull", json={"name": "code"})
    assert response.status_code == 503


def test_ollama_delete_refuses_static_model() -> None:
    client, _, _ = build_test_client()
    response = client.post("/api/delete", json={"name": "code"})
    assert response.status_code == 400


def test_ollama_generate_stream_translates_sse_to_ndjson() -> None:
    sse_chunks = _sse(
        [
            {"choices": [{"delta": {"content": "chunk1"}}]},
            {"choices": [{"delta": {"content": "chunk2"}, "finish_reason": "stop"}]},
        ]
    )
    client, _, proxy_client = build_test_client(sse_chunks=sse_chunks)

    response = client.post(
        "/api/generate",
        json={
            "model": "finance",
            "prompt": "Summarize quarterly risk.",
        },
    )

    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.splitlines() if line]
    responses = [evt["response"] for evt in lines if not evt["done"]]
    assert responses == ["chunk1", "chunk2"]
    assert lines[-1]["done"] is True
    assert proxy_client.stream_calls[0][2]["stream"] is True
