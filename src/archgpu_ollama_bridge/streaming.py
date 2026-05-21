"""Helpers for translating upstream OpenAI-style SSE chunks into other framings.

`llama-server` speaks the OpenAI streaming format (``data: {...}\\n\\n`` lines
terminated by ``data: [DONE]``). Ollama clients expect newline-delimited JSON
where each event is ``{"...": ..., "done": false|true}``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def iter_sse_events(stream: AsyncIterator[bytes]) -> AsyncIterator[str | None]:
    """Yield SSE event payload strings.

    Yields the raw ``data:`` payload as a string for each event, and ``None``
    when the upstream emits a ``[DONE]`` sentinel. Empty / comment lines are
    ignored.
    """

    buffer = b""
    async for chunk in stream:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if not payload:
                continue
            if payload == b"[DONE]":
                yield None
                return
            yield payload.decode("utf-8", errors="replace")


async def sse_to_ollama_chat(
    stream: AsyncIterator[bytes],
    model_name: str,
) -> AsyncIterator[bytes]:
    """Translate upstream SSE chat-completion chunks into Ollama ``/api/chat`` NDJSON."""

    finish_reason: str | None = None
    saw_any_chunk = False

    async for event in iter_sse_events(stream):
        if event is None:
            break
        try:
            data = json.loads(event)
        except json.JSONDecodeError:
            continue

        choice = (data.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        content = delta.get("content") or ""
        role = delta.get("role") or "assistant"
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        if content:
            saw_any_chunk = True
            yield (
                json.dumps(
                    {
                        "model": model_name,
                        "created_at": _now_iso(),
                        "message": {"role": role, "content": content},
                        "done": False,
                    }
                )
                + "\n"
            ).encode("utf-8")

    yield (
        json.dumps(
            {
                "model": model_name,
                "created_at": _now_iso(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": finish_reason or ("stop" if saw_any_chunk else "stop"),
            }
        )
        + "\n"
    ).encode("utf-8")


async def sse_to_ollama_generate(
    stream: AsyncIterator[bytes],
    model_name: str,
) -> AsyncIterator[bytes]:
    """Translate upstream SSE chunks into Ollama ``/api/generate`` NDJSON."""

    finish_reason: str | None = None

    async for event in iter_sse_events(stream):
        if event is None:
            break
        try:
            data = json.loads(event)
        except json.JSONDecodeError:
            continue

        choice = (data.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        content = delta.get("content") or ""
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        if content:
            yield (
                json.dumps(
                    {
                        "model": model_name,
                        "created_at": _now_iso(),
                        "response": content,
                        "done": False,
                    }
                )
                + "\n"
            ).encode("utf-8")

    yield (
        json.dumps(
            {
                "model": model_name,
                "created_at": _now_iso(),
                "response": "",
                "done": True,
                "done_reason": finish_reason or "stop",
            }
        )
        + "\n"
    ).encode("utf-8")
