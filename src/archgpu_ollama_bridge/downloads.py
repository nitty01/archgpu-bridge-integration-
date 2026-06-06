"""Async Hugging Face GGUF downloader with Ollama-style NDJSON progress.

Yields events shaped like the upstream Ollama ``/api/pull`` stream::

    {"status": "pulling manifest"}
    {"status": "downloading", "digest": "sha256:...", "total": 1234, "completed": 56}
    {"status": "verifying digest"}
    {"status": "writing manifest"}
    {"status": "success"}

The downloader writes to a ``<file>.partial`` next to the destination and
atomically renames on success. Cancellation/exception cleans up the partial.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx

from .catalogue import HFRef

logger = logging.getLogger(__name__)


_PROGRESS_INTERVAL_SECONDS = 0.25
_PROGRESS_BYTES = 4 * 1024 * 1024  # 4 MiB
_DEFAULT_CHUNK = 1 * 1024 * 1024
_FREE_SPACE_BUFFER_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB


class DownloadError(RuntimeError):
    """Raised for any unrecoverable download problem."""


@dataclass(slots=True)
class DownloadResult:
    path: Path
    total_bytes: int
    digest: str
    etag: str | None


def _ndjson(payload: dict) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


class HuggingFaceDownloader:
    """Streams a single GGUF file from Hugging Face to disk."""

    def __init__(
        self,
        *,
        models_dir: Path,
        base_url: str = "https://huggingface.co",
        client_factory=None,
        chunk_size: int = _DEFAULT_CHUNK,
        max_bytes: int | None = None,
        free_space_buffer_bytes: int = _FREE_SPACE_BUFFER_BYTES,
    ) -> None:
        self._models_dir = Path(models_dir)
        self._base_url = base_url.rstrip("/")
        self._client_factory = client_factory or (
            lambda: httpx.AsyncClient(follow_redirects=True, timeout=None)
        )
        self._chunk_size = chunk_size
        self._max_bytes = max_bytes
        self._free_space_buffer_bytes = free_space_buffer_bytes

    def target_path(self, ref: HFRef) -> Path:
        return self._models_dir / ref.safe_local_filename

    def partial_path(self, ref: HFRef) -> Path:
        target = self.target_path(ref)
        return target.with_suffix(target.suffix + ".partial")

    def already_present(self, ref: HFRef) -> Path | None:
        target = self.target_path(ref)
        if target.exists() and target.stat().st_size > 0:
            return target
        return None

    async def pull(
        self,
        ref: HFRef,
        *,
        keep_partial_on_cancel: bool = False,
        resume: bool = True,
    ) -> AsyncIterator[bytes]:
        """Yield Ollama-shaped NDJSON progress events for a single pull."""

        url = f"{self._base_url}/{ref.repo}/resolve/{ref.revision}/{ref.filename}"
        target = self.target_path(ref)
        partial = self.partial_path(ref)

        self._models_dir.mkdir(parents=True, exist_ok=True)

        yield _ndjson({"status": "pulling manifest"})

        if target.exists() and target.stat().st_size > 0:
            yield _ndjson(
                {
                    "status": "success",
                    "digest": "sha256:cached",
                    "total": target.stat().st_size,
                    "completed": target.stat().st_size,
                }
            )
            return

        try:
            async with self._client_factory() as client:
                async for event in self._stream_to_disk(
                    client,
                    url,
                    ref,
                    target,
                    partial,
                    resume=resume,
                ):
                    yield event
        except asyncio.CancelledError:
            if not keep_partial_on_cancel:
                self._cleanup(partial)
            raise
        except Exception as exc:
            self._cleanup(partial)
            logger.exception("pull failed for %s", ref.safe_local_filename)
            yield _ndjson({"status": "error", "error": str(exc)})
            return

    async def _stream_to_disk(
        self,
        client: httpx.AsyncClient,
        url: str,
        ref: HFRef,
        target: Path,
        partial: Path,
        *,
        resume: bool = True,
    ) -> AsyncIterator[bytes]:
        partial_size = partial.stat().st_size if (resume and partial.exists()) else 0
        headers: dict[str, str] = {}
        if partial_size > 0:
            headers["Range"] = f"bytes={partial_size}-"

        async with client.stream("GET", url, params={"download": "true"}, headers=headers) as response:
            if response.status_code >= 400:
                detail = (await response.aread()).decode("utf-8", errors="replace")[:500]
                raise DownloadError(
                    f"Hugging Face returned {response.status_code} for {url}: {detail}"
                )

            content_length = response.headers.get("content-length")
            etag = response.headers.get("x-linked-etag") or response.headers.get("etag")
            remaining = int(content_length) if content_length and content_length.isdigit() else 0
            total = remaining
            mode = "wb"
            completed = 0
            if partial_size > 0 and response.status_code == 206:
                total = partial_size + remaining
                completed = partial_size
                mode = "ab"
            elif partial_size > 0 and response.status_code == 200:
                # Upstream ignored Range; restart from zero.
                try:
                    partial.unlink()
                except OSError:
                    pass
                partial_size = 0
                completed = 0
                mode = "wb"

            if self._max_bytes is not None and total and total > self._max_bytes:
                raise DownloadError(
                    f"Refusing to download {total} bytes (max_bytes={self._max_bytes})"
                )

            self._check_free_space(total)

            raw_digest = etag or hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            digest_str = "sha256:" + raw_digest.strip('"')

            yield _ndjson(
                {
                    "status": "downloading",
                    "digest": digest_str,
                    "total": total or completed,
                    "completed": completed,
                }
            )

            sha = hashlib.sha256()
            last_emit = 0.0
            last_bytes = completed

            with partial.open(mode) as fh:
                async for chunk in response.aiter_bytes(chunk_size=self._chunk_size):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    sha.update(chunk)
                    completed += len(chunk)

                    if (
                        self._max_bytes is not None
                        and completed > self._max_bytes
                    ):
                        raise DownloadError(
                            f"Download exceeded max_bytes={self._max_bytes}"
                        )

                    now = time.monotonic()
                    if (
                        now - last_emit >= _PROGRESS_INTERVAL_SECONDS
                        and completed - last_bytes >= _PROGRESS_BYTES
                    ):
                        last_emit = now
                        last_bytes = completed
                        yield _ndjson(
                            {
                                "status": "downloading",
                                "digest": digest_str,
                                "total": total or completed,
                                "completed": completed,
                            }
                        )

            if total and completed != total:
                raise DownloadError(
                    f"Short read: expected {total} bytes, got {completed}"
                )

            yield _ndjson(
                {
                    "status": "downloading",
                    "digest": digest_str,
                    "total": total or completed,
                    "completed": completed,
                }
            )
            yield _ndjson({"status": "verifying digest"})

            partial.replace(target)

            yield _ndjson({"status": "writing manifest"})
            yield _ndjson(
                {
                    "status": "success",
                    "digest": digest_str,
                    "total": completed,
                    "completed": completed,
                }
            )

    def _cleanup(self, partial: Path) -> None:
        try:
            if partial.exists():
                partial.unlink()
        except OSError:
            logger.warning("failed to clean up partial file %s", partial)

    def _check_free_space(self, expected_bytes: int) -> None:
        try:
            usage = shutil.disk_usage(self._models_dir)
        except FileNotFoundError:
            return
        needed = max(expected_bytes, 0) + self._free_space_buffer_bytes
        if usage.free < needed:
            raise DownloadError(
                f"Insufficient disk space at {self._models_dir}: free={usage.free} "
                f"bytes, need ~{needed} bytes (file + 5 GiB buffer)"
            )
