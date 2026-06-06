"""Per-name serialisation for /api/pull and helpers to register pulled models."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
import re

from .catalogue import Catalogue, CatalogueError, HFRef
from .config import Settings
from .downloads import HuggingFaceDownloader
from .registry import ModelRecord, ModelRegistry

logger = logging.getLogger(__name__)


_MODEL_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")
_QUANT_RE = re.compile(r"(q\d(?:_[a-z0-9]+)+)", re.IGNORECASE)


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


def derive_display_name(ref: HFRef) -> str:
    """Build a compact human-readable name for UI display."""

    repo_name = ref.repo.split("/", 1)[-1] if "/" in ref.repo else ref.repo
    base = (repo_name or ref.display_name or "model").strip()
    if base.lower().endswith(".gguf"):
        base = base[: -len(".gguf")]

    filename = ref.filename or ""
    quant = None
    match = _QUANT_RE.search(filename)
    if match:
        quant = match.group(1).upper()

    if quant and quant.lower() not in base.lower():
        return f"{base} ({quant})"
    return base


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
        normalized = (self.status or "").lower()
        if normalized in {"pulling", "downloading"}:
            actions = ["stop", "restart", "purge"]
        elif normalized in {"paused", "stopped", "suspended"}:
            actions = ["resume", "restart", "purge", "clear"]
        elif normalized in {"error", "failed"}:
            actions = ["resume", "restart", "purge", "clear"]
        elif normalized in {"registered", "success", "done"}:
            actions = ["clear", "purge"]
        elif normalized == "purged":
            actions = ["clear"]
        else:
            actions = ["resume", "restart", "purge", "clear"]
        return {
            "model": self.model,
            "status": self.status,
            "total": self.total,
            "completed": self.completed,
            "digest": self.digest,
            "error": self.error,
            "updated_at": self.updated_at,
            "actions": actions,
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
        self._plans: dict[str, PullPlan] = {}
        self._tasks: dict[str, asyncio.Task] = {}

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

        # Guard: if this model is already registered, skip network activity.
        if self._registry.has(plan.model_id):
            self._plans[plan.model_id] = plan
            self._set_status(plan.model_id, status="registered", total=1, completed=1)
            yield _ndjson(
                {
                    "status": "registered",
                    "model": plan.model_id,
                    "detail": "already installed",
                }
            )
            return

        # Guard: if the GGUF is already on disk (e.g., previous pull/download),
        # register it and skip re-downloading.
        if self._downloader.already_present(plan.ref) is not None:
            self._plans[plan.model_id] = plan
            try:
                self._register(plan)
            except Exception as exc:
                self._set_status(plan.model_id, status="error", error=str(exc))
                yield _ndjson(
                    {
                        "status": "error",
                        "error": f"found local file but failed to register: {exc}",
                    }
                )
                return
            self._set_status(plan.model_id, status="registered", total=1, completed=1)
            yield _ndjson(
                {
                    "status": "registered",
                    "model": plan.model_id,
                    "detail": "already downloaded",
                }
            )
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
            self._plans[plan.model_id] = plan
            self._tasks[plan.model_id] = asyncio.current_task()
            self._set_status(plan.model_id, status="pulling")
            success = False
            try:
                async for event in self._downloader.pull(
                    plan.ref,
                    keep_partial_on_cancel=True,
                    resume=True,
                ):
                    self._ingest_event(plan.model_id, event)
                    yield event
                    if _is_success_event(event):
                        success = True
            except asyncio.CancelledError:
                current = self._status.get(plan.model_id)
                self._set_status(
                    plan.model_id,
                    status="stopped",
                    total=current.total if current else None,
                    completed=current.completed if current else None,
                    digest=current.digest if current else None,
                    error=None,
                )
                return
            except Exception as exc:  # pragma: no cover - downloader handles its own
                logger.exception("pull failed for %s", plan.model_id)
                self._set_status(plan.model_id, status="error", error=str(exc))
                yield _ndjson({"status": "error", "error": str(exc)})
                return
            finally:
                task = self._tasks.get(plan.model_id)
                if task is asyncio.current_task():
                    self._tasks.pop(plan.model_id, None)

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

    async def pause(self, model: str) -> PullStatus:
        # Backward-compatible alias for older UI/clients.
        return await self.stop(model)

    async def stop(self, model: str) -> PullStatus:
        status = self._status.get(model)
        task = self._tasks.get(model)
        if task is not None and not task.done():
            task.cancel()
            self._set_status(
                model,
                status="stopped",
                total=status.total if status else None,
                completed=status.completed if status else None,
                digest=status.digest if status else None,
                error=None,
            )
            return self._status[model]
        self._set_status(
            model,
            status="stopped",
            total=status.total if status else None,
            completed=status.completed if status else None,
            digest=status.digest if status else None,
            error=None,
        )
        return self._status[model]

    async def resume(self, model: str) -> PullStatus:
        if model in self._tasks and not self._tasks[model].done():
            return self._status.get(model) or PullStatus(model=model, status="pulling")
        plan = self._plans.get(model)
        if plan is None:
            plan = self.plan(model)
            self._plans[plan.model_id] = plan
            model = plan.model_id
        task = asyncio.create_task(self._run_background_pull(plan))
        self._tasks[plan.model_id] = task
        self._set_status(plan.model_id, status="pulling")
        return self._status[plan.model_id]

    async def restart(self, model: str) -> PullStatus:
        plan = self._plans.get(model)
        if plan is None:
            plan = self.plan(model)
            self._plans[plan.model_id] = plan
            model = plan.model_id
        await self.stop(model)
        self._delete_partial(plan)
        self._delete_target(plan)
        try:
            if self._registry.has(model) and self._registry.is_dynamic(model):
                self._registry.unregister(model)
        except Exception:
            pass
        task = asyncio.create_task(self._run_background_pull(plan))
        self._tasks[model] = task
        self._set_status(model, status="pulling")
        return self._status[model]

    def clear(self, model: str) -> None:
        self._status.pop(model, None)

    async def purge(self, model: str) -> PullStatus:
        await self.stop(model)
        plan = self._plans.get(model)
        status = self._status.get(model)
        if plan is not None:
            self._delete_partial(plan)
            self._delete_target(plan)
        try:
            if self._registry.has(model) and self._registry.is_dynamic(model):
                self._registry.unregister(model)
        except Exception:
            pass
        # Purge is intended to fully remove transfer artifacts; keep the response
        # informative but do not retain a persistent transfer row in status lists.
        result = PullStatus(
            model=model,
            status="purged",
            total=status.total if status else None,
            completed=status.completed if status else None,
            digest=status.digest if status else None,
            error=None,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._status.pop(model, None)
        return result

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
            display_name=derive_display_name(plan.ref),
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

    async def _run_background_pull(self, plan: PullPlan) -> None:
        lock = await self.lock_for(plan.model_id)
        if lock.locked():
            return
        async with lock:
            self._set_status(plan.model_id, status="pulling")
            success = False
            try:
                async for event in self._downloader.pull(
                    plan.ref,
                    keep_partial_on_cancel=True,
                    resume=True,
                ):
                    self._ingest_event(plan.model_id, event)
                    if _is_success_event(event):
                        success = True
            except asyncio.CancelledError:
                current = self._status.get(plan.model_id)
                self._set_status(
                    plan.model_id,
                    status="stopped",
                    total=current.total if current else None,
                    completed=current.completed if current else None,
                    digest=current.digest if current else None,
                    error=None,
                )
                return
            except Exception as exc:
                self._set_status(plan.model_id, status="error", error=str(exc))
                return
            finally:
                task = self._tasks.get(plan.model_id)
                if task is asyncio.current_task():
                    self._tasks.pop(plan.model_id, None)

            if not success:
                current = self._status.get(plan.model_id)
                if current is None or current.status not in {"error", "stopped"}:
                    self._set_status(
                        plan.model_id,
                        status="suspended",
                        total=current.total if current else None,
                        completed=current.completed if current else None,
                        digest=current.digest if current else None,
                        error="pull ended before success event",
                    )
                return

            try:
                self._register(plan)
            except Exception as exc:
                self._set_status(plan.model_id, status="error", error=str(exc))
                return
            self._set_status(plan.model_id, status="registered", total=1, completed=1)

    def _delete_partial(self, plan: PullPlan) -> None:
        partial = self._downloader.partial_path(plan.ref)
        try:
            if partial.is_file():
                partial.unlink()
        except OSError:
            pass

    def _delete_target(self, plan: PullPlan) -> None:
        target = self._downloader.target_path(plan.ref)
        try:
            if target.is_file():
                target.unlink()
        except OSError:
            pass


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
