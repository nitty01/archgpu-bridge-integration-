from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .backends import DockerBackendManager, build_docker_settings_from_app
from .catalogue import Catalogue
from .config import Settings
from .downloads import HuggingFaceDownloader
from .lifecycle import BackendManager, LifecycleManager, UnsupportedBackendManager
from .pulls import PullManager
from .registry import ModelRegistry, PortPool
from .router import RequestRouter
from .state import RuntimeStateStore


@dataclass(slots=True)
class ProxyResponse:
    status_code: int
    json_body: Any


class ProxyClient(Protocol):
    async def post_json(
        self, base_url: str, path: str, payload: dict[str, Any]
    ) -> ProxyResponse: ...

    def stream_post(
        self, base_url: str, path: str, payload: dict[str, Any]
    ) -> AsyncIterator[bytes]: ...


class HttpxProxyClient:
    def __init__(self, request_timeout_seconds: float = 300.0) -> None:
        self._timeout = request_timeout_seconds

    async def post_json(
        self, base_url: str, path: str, payload: dict[str, Any]
    ) -> ProxyResponse:
        async with httpx.AsyncClient(base_url=base_url, timeout=self._timeout) as client:
            response = await client.post(path, json=payload)
            response.raise_for_status()
            return ProxyResponse(
                status_code=response.status_code,
                json_body=response.json(),
            )

    async def stream_post(
        self, base_url: str, path: str, payload: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(base_url=base_url, timeout=self._timeout) as client:
            async with client.stream("POST", path, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk


@dataclass(slots=True)
class AppServices:
    registry: ModelRegistry
    router: RequestRouter
    proxy_client: ProxyClient
    settings: Settings | None = None
    lifecycle: LifecycleManager | None = None
    catalogue: Catalogue | None = None
    downloader: HuggingFaceDownloader | None = None
    pull_manager: PullManager | None = None


def build_backend_manager(settings: Settings) -> BackendManager:
    driver = (settings.backend_driver or "").lower()
    if driver == "docker":
        return DockerBackendManager(settings=build_docker_settings_from_app(settings))
    if driver in {"none", "stub", "disabled"}:
        return UnsupportedBackendManager()
    raise ValueError(f"Unknown backend_driver: {settings.backend_driver!r}")


def build_services(
    settings: Settings,
    *,
    backend_manager: BackendManager | None = None,
    proxy_client: ProxyClient | None = None,
    catalogue: Catalogue | None = None,
    downloader: HuggingFaceDownloader | None = None,
) -> AppServices:
    port_start, port_end = settings.dynamic_port_range
    registry = ModelRegistry.load(
        settings.registry_path,
        dynamic_path=settings.dynamic_models_path,
        port_pool=PortPool(port_start, port_end),
    )
    state_store = RuntimeStateStore(settings.state_path)
    backend = backend_manager or build_backend_manager(settings)
    lifecycle = LifecycleManager(
        registry=registry,
        state_store=state_store,
        backend_manager=backend,
        max_loaded_models=settings.max_loaded_models,
    )
    router = RequestRouter(registry=registry, lifecycle=lifecycle)
    catalogue = catalogue or Catalogue.load(
        settings.catalogue_path,
        allow_orgs=tuple(settings.hf_allow_orgs),
    )
    downloader = downloader or HuggingFaceDownloader(
        models_dir=settings.backend_models_host_dir,
        base_url=settings.hf_base_url,
        max_bytes=settings.pull_max_bytes,
    )
    pull_manager = PullManager(
        registry=registry,
        catalogue=catalogue,
        downloader=downloader,
        settings=settings,
    )
    return AppServices(
        registry=registry,
        router=router,
        proxy_client=proxy_client or HttpxProxyClient(),
        settings=settings,
        lifecycle=lifecycle,
        catalogue=catalogue,
        downloader=downloader,
        pull_manager=pull_manager,
    )
