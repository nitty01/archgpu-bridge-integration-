"""End-to-end integration test for the bridge.

Requires:
- A running Docker daemon
- The ``local/llama.cpp:server-intel`` image (or ``ARCHGPU_BRIDGE_BACKEND_IMAGE``)
- GGUF files for ``code`` listed in ``config/models.yaml``
- ``/dev/dri`` available on the host

Run with::

    pytest -m integration
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from archgpu_ollama_bridge.config import Settings
from archgpu_ollama_bridge.services import build_services

pytestmark = pytest.mark.integration


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def _gguf_present(yaml_path: Path) -> bool:
    import yaml

    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    for entry in payload.get("models", []):
        if not Path(entry["gguf_path"]).exists():
            return False
    return True


@pytest.fixture(scope="module")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    if not _docker_available():
        pytest.skip("docker daemon not available")
    if not _gguf_present(Path("config/models.yaml")):
        pytest.skip("registered GGUF files are not present on disk")
    state_path = tmp_path_factory.mktemp("integration") / "state.json"
    return Settings(
        registry_path=Path("config/models.yaml"),
        state_path=state_path,
        idle_ttl_seconds=15.0,
        max_loaded_models=1,
        backend_startup_timeout_seconds=180.0,
    )


def test_ensure_loaded_starts_real_container_and_serves_request(settings: Settings) -> None:
    services = build_services(settings)
    lifecycle = services.lifecycle
    assert lifecycle is not None

    try:
        handle = asyncio.run(lifecycle.ensure_loaded("code"))

        with httpx.Client(timeout=60.0) as client:
            health = client.get(f"http://127.0.0.1:{handle.port}/health")
            assert health.status_code == 200

            chat = client.post(
                f"http://127.0.0.1:{handle.port}/v1/chat/completions",
                json={
                    "model": "code.gguf",
                    "messages": [{"role": "user", "content": "Reply with the word OK."}],
                    "max_tokens": 8,
                    "stream": False,
                },
            )
            assert chat.status_code == 200
            content = chat.json()["choices"][0]["message"]["content"]
            assert isinstance(content, str) and len(content) > 0
    finally:
        asyncio.run(lifecycle.stop("code"))


def test_idle_evictor_stops_container_after_ttl(settings: Settings) -> None:
    services = build_services(settings)
    lifecycle = services.lifecycle
    assert lifecycle is not None

    try:
        handle = asyncio.run(lifecycle.ensure_loaded("code"))

        with httpx.Client(timeout=10.0) as client:
            assert client.get(f"http://127.0.0.1:{handle.port}/health").status_code == 200

        time.sleep(settings.idle_ttl_seconds + 5.0)
        evicted = asyncio.run(lifecycle.evict_idle(settings.idle_ttl_seconds))
        assert "code" in evicted

        with httpx.Client(timeout=2.0) as client:
            with pytest.raises(httpx.HTTPError):
                client.get(f"http://127.0.0.1:{handle.port}/health")
    finally:
        asyncio.run(lifecycle.stop("code"))


def test_long_prompt_above_4096_tokens_succeeds_with_increased_context(settings: Settings) -> None:
    """Regression check for the issue noted in CHAT_CONTEXT_SUMMARY.md.

    The bridge configures ``-c 8192`` for the ``code`` model, so a prompt
    that previously failed at the 4096-token boundary should now succeed.
    """

    services = build_services(settings)
    lifecycle = services.lifecycle
    assert lifecycle is not None

    big_prompt = ("def f(x):\n    return x + 1\n\n" * 600).strip()

    try:
        handle = asyncio.run(lifecycle.ensure_loaded("code"))
        with httpx.Client(timeout=180.0) as client:
            response = client.post(
                f"http://127.0.0.1:{handle.port}/v1/chat/completions",
                json={
                    "model": "code.gguf",
                    "messages": [
                        {
                            "role": "user",
                            "content": big_prompt + "\n\nReply with the word OK.",
                        }
                    ],
                    "max_tokens": 4,
                    "stream": False,
                },
            )
        assert response.status_code == 200, response.text
    finally:
        asyncio.run(lifecycle.stop("code"))
