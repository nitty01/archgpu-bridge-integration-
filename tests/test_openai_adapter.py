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
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async def stream_post(self, base_url: str, path: str, payload: dict) -> AsyncIterator[bytes]:
        self.stream_calls.append((base_url, path, payload))
        for chunk in self._sse_chunks:
            yield chunk


def build_test_client(
    sse_chunks: list[bytes] | None = None,
) -> tuple[TestClient, FakeRouter, FakeProxyClient]:
    registry = ModelRegistry.load(Path("config/models.yaml"))
    router = FakeRouter(registry)
    proxy_client = FakeProxyClient(sse_chunks=sse_chunks)
    services = AppServices(
        registry=registry,
        router=router,
        proxy_client=proxy_client,
    )
    return TestClient(create_app(services)), router, proxy_client


def test_openai_models_lists_registry_entries() -> None:
    client, _, _ = build_test_client()

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"] == [
        {"id": "mistral.gguf", "object": "model", "owned_by": "archgpu-bridge"},
        {"id": "code.gguf", "object": "model", "owned_by": "archgpu-bridge"},
        {"id": "finance.gguf", "object": "model", "owned_by": "archgpu-bridge"},
    ]


def test_openai_chat_completions_proxies_payload() -> None:
    client, router, proxy_client = build_test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "code.gguf",
            "messages": [{"role": "user", "content": "Say hi"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["model"] == "code.gguf"
    assert router.requests == ["code.gguf"]
    assert proxy_client.calls == [
        (
            "http://127.0.0.1:8081",
            "/v1/chat/completions",
            {
                "model": "code.gguf",
                "messages": [{"role": "user", "content": "Say hi"}],
            },
        )
    ]


def test_openai_chat_completions_streams_sse_passthrough() -> None:
    sse = [
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    client, _, proxy_client = build_test_client(sse_chunks=sse)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "code.gguf",
            "messages": [{"role": "user", "content": "stream please"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == b"".join(sse).decode()
    assert proxy_client.stream_calls and proxy_client.stream_calls[0][2]["stream"] is True
