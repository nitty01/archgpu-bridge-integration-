"""Per-name serialisation for /api/pull and helpers to register pulled models."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

from .catalogue import Catalogue, CatalogueError, HFRef
from .config import Settings
from .downloads import HuggingFaceDownloader
from .registry import ModelRecord, ModelRegistry

logger = logging.getLogger(__name__)


_MODEL_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _ndjson(payload: dict) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def derive_model_id(ref: HFRef) -> str:
    """Build a stable, registry-safe model id from an HF ref."""

    base = ref.filename
    if base.lower().endswith(".gguf"):
        base = base[: -len(".gguf")]
    org, repo = ref.repo.split("/", 1)
    raw = f"{org}-{repo}-{base}"
    cleaned = _MODEL_ID_RE.sub("-", raw).strip("-")
    return cleaned.lower() or "model"


@dataclass(slots=True)
class PullPlan:
    ref: HFRef
    model_id: str


@dataclass(slots=True)
class PullStatus:
    model: str
    status: str
    total: int | None = None
    completed: int | None = None
    digest: str | None = None
    error: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "status": self.status,
            "total": self.total,
            "completed": self.completed,
            "digest": self.digest,
            "error": self.error,
            "updated_at": self.updated_at,
        }


class PullManager:
    """Coordinates concurrent pulls; serialised per model id."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        catalogue: Catalogue,
        downloader: HuggingFaceDownloader,
        settings: Settings,
    ) -> None:
        self._registry = registry
        self._catalogue = catalogue
        self._downloader = downloader
        self._settings = settings
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._status: dict[str, PullStatus] = {}

    def plan(self, name: str) -> PullPlan:
        ref = self._catalogue.resolve(name)
        cleaned = (name or "").strip()
        if "/" in cleaned:
            model_id = derive_model_id(ref)
        else:
            # For catalogue aliases, keep the alias as model id so UI-selected
            # names are directly routable after pull.
            model_id = cleaned.split(":", 1)[0]
        return PullPlan(ref=ref, model_id=model_id)

    async def lock_for(self, model_id: str) -> asyncio.Lock:
        async with self._global_lock:
            lock = self._locks.get(model_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[model_id] = lock
            return lock

    async def stream_pull(self, name: str) -> AsyncIterator[bytes]:
        try:
            plan = self.plan(name)
        except CatalogueError as exc:
            yield _ndjson({"status": "error", "error": str(exc)})
            return

        lock = await self.lock_for(plan.model_id)
        if lock.locked():
            yield _ndjson(
                {
                    "status": "error",
                    "error": f"pull for {plan.model_id!r} already in progress",
                }
            )
            return

        async with lock:
            self._set_status(plan.model_id, status="pulling")
            success = False
            try:
                async for event in self._downloader.pull(plan.ref):
                    self._ingest_event(plan.model_id, event)
                    yield event
                    if _is_success_event(event):
                        success = True
            except Exception as exc:  # pragma: no cover - downloader handles its own
                logger.exception("pull failed for %s", plan.model_id)
                self._set_status(plan.model_id, status="error", error=str(exc))
                yield _ndjson({"status": "error", "error": str(exc)})
                return

            if not success:
                current = self._status.get(plan.model_id)
                if current is None or current.status != "error":
                    self._set_status(
                        plan.model_id,
                        status="error",
                        error="pull stream ended before success event",
                    )
                return

            try:
                self._register(plan)
            except Exception as exc:
                logger.exception("registry update failed for %s", plan.model_id)
                self._set_status(plan.model_id, status="error", error=str(exc))
                yield _ndjson(
                    {
                        "status": "error",
                        "error": f"downloaded but failed to register: {exc}",
                    }
                )
                return

            self._set_status(
                plan.model_id,
                status="registered",
                completed=1,
                total=1,
            )
            yield _ndjson(
                {
                    "status": "registered",
                    "model": plan.model_id,
                }
            )

    def get_status(self, model: str) -> PullStatus | None:
        return self._status.get(model)

    def list_status(self) -> list[PullStatus]:
        return sorted(
            self._status.values(),
            key=lambda s: s.updated_at or "",
            reverse=True,
        )

    def _set_status(
        self,
        model: str,
        *,
        status: str,
        total: int | None = None,
        completed: int | None = None,
        digest: str | None = None,
        error: str | None = None,
    ) -> None:
        self._status[model] = PullStatus(
            model=model,
            status=status,
            total=total,
            completed=completed,
            digest=digest,
            error=error,
            updated_at=datetime.now(UTC).isoformat(),
        )

    def _ingest_event(self, model: str, line: bytes) -> None:
        try:
            parsed = json.loads(line)
        except (ValueError, TypeError):
            return
        if not isinstance(parsed, dict):
            return
        status = str(parsed.get("status", "unknown"))
        self._set_status(
            model,
            status=status,
            total=_as_int(parsed.get("total")),
            completed=_as_int(parsed.get("completed")),
            digest=_as_str(parsed.get("digest")),
            error=_as_str(parsed.get("error")),
        )

    def _register(self, plan: PullPlan) -> ModelRecord:
        if self._registry.has(plan.model_id):
            return self._registry.get(plan.model_id)

        port = self._registry.allocate_port()
        record = ModelRecord(
            id=plan.model_id,
            gguf_path=self._downloader.target_path(plan.ref),
            openai_name=f"{plan.model_id}.gguf",
            ollama_name=plan.model_id,
            port=port,
            context_length=plan.ref.suggested_context_length
            or self._settings.dynamic_default_context_length,
            tags=list(plan.ref.tags) or ["dynamic"],
            source="dynamic",
            hf_repo=plan.ref.repo,
            hf_filename=plan.ref.filename,
            hf_revision=plan.ref.revision,
        )
        try:
            return self._registry.register(record)
        except Exception:
            self._registry.release_port(port)
            raise


def _is_success_event(line: bytes) -> bool:
    try:
        parsed = json.loads(line)
    except (ValueError, TypeError):
        return False
    return isinstance(parsed, dict) and parsed.get("status") == "success"


def _as_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _as_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None
