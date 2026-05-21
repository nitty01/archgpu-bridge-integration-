from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

from .adapters.ollama import build_ollama_router
from .adapters.openai import build_openai_router
from .config import Settings, get_settings
from .lifecycle import LifecycleManager
from .middleware import access_log_middleware
from .services import AppServices, build_services

logger = logging.getLogger(__name__)


async def _idle_evictor(
    lifecycle: LifecycleManager,
    idle_ttl_seconds: float,
    interval_seconds: float,
) -> None:
    if idle_ttl_seconds <= 0 or interval_seconds <= 0:
        return
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await lifecycle.evict_idle(idle_ttl_seconds)
            except Exception:
                logger.exception("idle evictor iteration failed; continuing")
    except asyncio.CancelledError:
        return


def create_app(
    services: AppServices | None = None,
    *,
    settings: Settings | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()
    app_services = services or build_services(app_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        evictor_task: asyncio.Task[None] | None = None
        if app_services.lifecycle is not None and app_settings.idle_ttl_seconds > 0:
            interval = max(5.0, app_settings.idle_ttl_seconds / 4)
            evictor_task = asyncio.create_task(
                _idle_evictor(
                    app_services.lifecycle,
                    app_settings.idle_ttl_seconds,
                    interval,
                ),
                name="archgpu-bridge-idle-evictor",
            )
        try:
            yield
        finally:
            if evictor_task is not None:
                evictor_task.cancel()
                try:
                    await evictor_task
                except asyncio.CancelledError:
                    pass

    app = FastAPI(title=app_settings.app_name, lifespan=lifespan)
    app.add_middleware(BaseHTTPMiddleware, dispatch=access_log_middleware)

    @app.get("/health")
    async def healthcheck() -> dict[str, str]:
        return {
            "status": "ok",
            "environment": app_settings.app_env,
        }

    app.include_router(build_openai_router(app_services))
    app.include_router(build_ollama_router(app_services))
    return app


app = create_app()
