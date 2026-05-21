from dataclasses import dataclass

from .lifecycle import BackendHandle, LifecycleManager
from .registry import ModelRecord, ModelRegistry


@dataclass(slots=True)
class RouteTarget:
    model: ModelRecord
    handle: BackendHandle
    base_url: str


class RequestRouter:
    def __init__(
        self,
        registry: ModelRegistry,
        lifecycle: LifecycleManager,
        backend_host: str = "http://127.0.0.1",
    ) -> None:
        self.registry = registry
        self.lifecycle = lifecycle
        self.backend_host = backend_host.rstrip("/")

    async def route(self, identifier: str) -> RouteTarget:
        model = self.registry.get(identifier)
        handle = await self.lifecycle.ensure_loaded(identifier)
        self.lifecycle.touch(identifier)
        return RouteTarget(
            model=model,
            handle=handle,
            base_url=f"{self.backend_host}:{handle.port}",
        )
