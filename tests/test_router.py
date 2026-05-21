import asyncio
from pathlib import Path

import pytest

from archgpu_ollama_bridge.lifecycle import BackendHandle
from archgpu_ollama_bridge.registry import ModelRegistry
from archgpu_ollama_bridge.router import RequestRouter


class FakeLifecycleManager:
    def __init__(self) -> None:
        self.requests: list[str] = []
        self.touched: list[str] = []

    async def ensure_loaded(self, identifier: str) -> BackendHandle:
        self.requests.append(identifier)
        ports = {
            "mistral": 8080,
            "mistral.gguf": 8080,
            "code": 8081,
            "code.gguf": 8081,
            "finance": 8082,
            "finance.gguf": 8082,
        }
        model_id = identifier.replace(".gguf", "")
        return BackendHandle(
            model_id=model_id,
            backend_id=f"llama-{model_id}",
            port=ports[identifier],
        )

    def touch(self, identifier: str) -> None:
        self.touched.append(identifier)


def test_router_resolves_model_and_returns_backend_target() -> None:
    registry = ModelRegistry.load(Path("config/models.yaml"))
    lifecycle = FakeLifecycleManager()
    router = RequestRouter(registry, lifecycle)  # type: ignore[arg-type]

    target = asyncio.run(router.route("code.gguf"))

    assert target.model.id == "code"
    assert target.handle.backend_id == "llama-code"
    assert target.base_url == "http://127.0.0.1:8081"
    assert lifecycle.requests == ["code.gguf"]
    assert lifecycle.touched == ["code.gguf"]


def test_router_raises_for_unknown_model() -> None:
    registry = ModelRegistry.load(Path("config/models.yaml"))
    lifecycle = FakeLifecycleManager()
    router = RequestRouter(registry, lifecycle)  # type: ignore[arg-type]

    with pytest.raises(KeyError, match="Unknown model: missing"):
        asyncio.run(router.route("missing"))
