import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from archgpu_ollama_bridge.catalogue import HFRef
from archgpu_ollama_bridge.downloads import HuggingFaceDownloader


def _ref() -> HFRef:
    return HFRef(repo="Qwen/X-GGUF", filename="model.gguf")


def _mock_transport(payload: bytes, *, status: int = 200, headers: dict | None = None):
    headers = headers or {"content-length": str(len(payload))}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=status, content=payload, headers=headers)

    return httpx.MockTransport(handler)


def _client_factory(transport: httpx.MockTransport):
    def factory():
        return httpx.AsyncClient(
            transport=transport, follow_redirects=True, timeout=None
        )

    return factory


async def _drain(it: AsyncIterator[bytes]) -> list[dict]:
    events: list[dict] = []
    async for raw in it:
        events.append(json.loads(raw))
    return events


def test_pull_emits_ollama_shaped_events_and_atomically_writes_file(tmp_path: Path) -> None:
    payload = b"x" * 65 * 1024 * 1024  # 65 MiB so several progress events fire
    transport = _mock_transport(payload)
    downloader = HuggingFaceDownloader(
        models_dir=tmp_path,
        client_factory=_client_factory(transport),
        chunk_size=4 * 1024 * 1024,
        free_space_buffer_bytes=0,
    )

    events = asyncio.run(_drain(downloader.pull(_ref())))

    statuses = [e["status"] for e in events]
    assert statuses[0] == "pulling manifest"
    assert "downloading" in statuses
    assert statuses[-1] == "success"

    target = downloader.target_path(_ref())
    assert target.exists()
    assert target.stat().st_size == len(payload)
    assert not target.with_suffix(target.suffix + ".partial").exists()


def test_pull_skips_when_file_already_present(tmp_path: Path) -> None:
    transport = _mock_transport(b"never-fetched")
    downloader = HuggingFaceDownloader(
        models_dir=tmp_path,
        client_factory=_client_factory(transport),
    )
    target = downloader.target_path(_ref())
    target.write_bytes(b"already there")

    events = asyncio.run(_drain(downloader.pull(_ref())))
    statuses = [e["status"] for e in events]
    assert statuses == ["pulling manifest", "success"]


def test_pull_emits_error_event_on_http_error(tmp_path: Path) -> None:
    transport = _mock_transport(b"nope", status=404, headers={})
    downloader = HuggingFaceDownloader(
        models_dir=tmp_path,
        client_factory=_client_factory(transport),
        free_space_buffer_bytes=0,
    )

    events = asyncio.run(_drain(downloader.pull(_ref())))
    assert events[-1]["status"] == "error"
    assert "404" in events[-1]["error"]
    assert not downloader.target_path(_ref()).exists()
    assert not downloader.target_path(_ref()).with_suffix(".gguf.partial").exists()


def test_pull_refuses_when_content_length_exceeds_max_bytes(tmp_path: Path) -> None:
    payload = b"a" * 1024
    transport = _mock_transport(payload)
    downloader = HuggingFaceDownloader(
        models_dir=tmp_path,
        client_factory=_client_factory(transport),
        max_bytes=128,
        free_space_buffer_bytes=0,
    )

    events = asyncio.run(_drain(downloader.pull(_ref())))
    assert events[-1]["status"] == "error"
    assert "max_bytes" in events[-1]["error"]


def test_pull_cleans_up_partial_when_short_read(tmp_path: Path) -> None:
    payload = b"ABC"
    transport = _mock_transport(payload, headers={"content-length": "999"})
    downloader = HuggingFaceDownloader(
        models_dir=tmp_path,
        client_factory=_client_factory(transport),
        free_space_buffer_bytes=0,
    )

    events = asyncio.run(_drain(downloader.pull(_ref())))
    assert events[-1]["status"] == "error"
    target = downloader.target_path(_ref())
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".partial").exists()


def test_pull_writes_to_partial_first(tmp_path: Path) -> None:
    payload = b"y" * (1 * 1024 * 1024)
    transport = _mock_transport(payload)
    downloader = HuggingFaceDownloader(
        models_dir=tmp_path,
        client_factory=_client_factory(transport),
        chunk_size=64 * 1024,
        free_space_buffer_bytes=0,
    )

    events = asyncio.run(_drain(downloader.pull(_ref())))
    assert events[-1]["status"] == "success"
    target = downloader.target_path(_ref())
    assert target.exists()
    assert not target.with_suffix(target.suffix + ".partial").exists()
