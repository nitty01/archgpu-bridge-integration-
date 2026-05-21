from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from .registry import ModelRecord, ModelRegistry
from .state import ModelLifecycleState, RuntimeStateRecord, RuntimeStateStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BackendHandle:
    model_id: str
    backend_id: str
    port: int


class BackendManager(Protocol):
    def start(self, model: ModelRecord) -> BackendHandle: ...

    def stop(self, handle: BackendHandle) -> None: ...

    def is_healthy(self, handle: BackendHandle) -> bool: ...


class UnsupportedBackendManager:
    def start(self, model: ModelRecord) -> BackendHandle:
        raise RuntimeError("Backend manager is not configured")

    def stop(self, handle: BackendHandle) -> None:
        return None

    def is_healthy(self, handle: BackendHandle) -> bool:
        return False


_LOADED_STATES = {ModelLifecycleState.READY, ModelLifecycleState.BUSY}


class LifecycleManager:
    """Coordinates start/stop of model backends with concurrency safety.

    - ``ensure_loaded`` is serialised per model via an ``asyncio.Lock`` so two
      concurrent callers don't race to spawn the same backend.
    - Before starting a new backend, if the number of currently-loaded models
      is at or above ``max_loaded_models``, the least-recently-used model is
      evicted.
    - ``evict_idle`` can be called periodically to stop models that have been
      idle longer than ``idle_ttl_seconds``.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        state_store: RuntimeStateStore,
        backend_manager: BackendManager,
        *,
        max_loaded_models: int = 2,
    ) -> None:
        self.registry = registry
        self.state_store = state_store
        self.backend_manager = backend_manager
        self.max_loaded_models = max_loaded_models
        self._handles: dict[str, BackendHandle] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def ensure_loaded(self, identifier: str) -> BackendHandle:
        model = self.registry.get(identifier)
        async with self._lock_for(model.id):
            current = self.state_store.get(model.id)

            if current and current.state == ModelLifecycleState.READY:
                handle = self._require_handle(model.id, current)
                if await asyncio.to_thread(self.backend_manager.is_healthy, handle):
                    self._touch(model.id)
                    return handle
                logger.warning("model %s no longer healthy; will restart", model.id)

            await self._enforce_capacity(exclude=model.id)

            self.state_store.save(
                RuntimeStateRecord(
                    model_id=model.id,
                    state=ModelLifecycleState.STARTING,
                    port=model.port,
                )
            )

            try:
                handle = await asyncio.to_thread(self.backend_manager.start, model)
                if not await asyncio.to_thread(self.backend_manager.is_healthy, handle):
                    raise RuntimeError(f"Backend for model {model.id} failed health check")
            except Exception as exc:
                self.state_store.save(
                    RuntimeStateRecord(
                        model_id=model.id,
                        state=ModelLifecycleState.FAILED,
                        port=model.port,
                        error=str(exc),
                    )
                )
                raise

            self._handles[model.id] = handle
            self.state_store.save(
                RuntimeStateRecord(
                    model_id=model.id,
                    state=ModelLifecycleState.READY,
                    port=handle.port,
                    backend_id=handle.backend_id,
                )
            )
            return handle

    async def stop(self, identifier: str) -> None:
        model = self.registry.get(identifier)
        async with self._lock_for(model.id):
            await self._stop_locked(model.id)

    async def evict_idle(
        self,
        idle_ttl_seconds: float,
        *,
        now: datetime | None = None,
    ) -> list[str]:
        """Stop models that have been idle longer than ``idle_ttl_seconds``.

        Returns the list of model ids that were evicted.
        """

        moment = now or datetime.now(UTC)
        evicted: list[str] = []
        for record in self.state_store.list():
            if record.state not in _LOADED_STATES:
                continue
            age = (moment - record.last_used_at).total_seconds()
            if age >= idle_ttl_seconds:
                logger.info(
                    "evicting idle model %s (idle for %.1fs)", record.model_id, age
                )
                async with self._lock_for(record.model_id):
                    await self._stop_locked(record.model_id)
                evicted.append(record.model_id)
        return evicted

    def touch(self, identifier: str) -> None:
        """Mark a model as used now. Safe to call from sync code."""

        model = self.registry.get(identifier)
        self._touch(model.id)

    def loaded_model_ids(self) -> list[str]:
        """Return loaded model ids ordered by least-recently-used first."""

        records = [
            record
            for record in self.state_store.list()
            if record.state in _LOADED_STATES
        ]
        records.sort(key=lambda r: r.last_used_at)
        return [r.model_id for r in records]

    async def _enforce_capacity(self, *, exclude: str) -> None:
        if self.max_loaded_models <= 0:
            return
        loaded = [mid for mid in self.loaded_model_ids() if mid != exclude]
        while len(loaded) >= self.max_loaded_models:
            victim = loaded.pop(0)
            logger.info(
                "max_loaded_models=%d reached; evicting LRU model %s",
                self.max_loaded_models,
                victim,
            )
            await self._stop_locked(victim)

    async def _stop_locked(self, model_id: str) -> None:
        current = self.state_store.get(model_id)
        handle = self._handles.get(model_id)

        if current is None and handle is None:
            return

        if handle is None and current and current.backend_id and current.port:
            handle = BackendHandle(
                model_id=model_id,
                backend_id=current.backend_id,
                port=current.port,
            )

        if handle is not None:
            await asyncio.to_thread(self.backend_manager.stop, handle)
            self._handles.pop(model_id, None)

        port = current.port if current else (handle.port if handle else None)
        backend_id = current.backend_id if current else (handle.backend_id if handle else None)
        self.state_store.save(
            RuntimeStateRecord(
                model_id=model_id,
                state=ModelLifecycleState.STOPPED,
                port=port,
                backend_id=backend_id,
            )
        )

    def _lock_for(self, model_id: str) -> asyncio.Lock:
        lock = self._locks.get(model_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[model_id] = lock
        return lock

    def _touch(self, model_id: str) -> None:
        record = self.state_store.get(model_id)
        if record is None:
            return
        updated = record.model_copy(update={"last_used_at": datetime.now(UTC)})
        self.state_store.save(updated)

    def _require_handle(
        self,
        model_id: str,
        current: RuntimeStateRecord,
    ) -> BackendHandle:
        handle = self._handles.get(model_id)
        if handle is not None:
            return handle

        if current.backend_id is None or current.port is None:
            raise RuntimeError(f"Missing runtime handle for model {model_id}")

        handle = BackendHandle(
            model_id=model_id,
            backend_id=current.backend_id,
            port=current.port,
        )
        self._handles[model_id] = handle
        return handle
