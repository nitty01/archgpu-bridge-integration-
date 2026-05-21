"""Console entrypoint for ``archgpu-bridge``.

Run via the installed script::

    archgpu-bridge

or directly::

    python -m archgpu_ollama_bridge
"""

from __future__ import annotations

import logging
import os

from .config import get_settings
from .logging_config import configure_logging


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    import uvicorn

    log_level = os.environ.get("ARCHGPU_BRIDGE_LOG_LEVEL", "INFO").upper()
    configure_logging(level=getattr(logging, log_level, logging.INFO))

    settings = get_settings()
    uvicorn.run(
        "archgpu_ollama_bridge.main:app",
        host=settings.host,
        port=settings.port,
        reload=_bool_env("ARCHGPU_BRIDGE_RELOAD", False),
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
