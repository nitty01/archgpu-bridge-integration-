import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from archgpu_ollama_bridge.lifecycle import BackendHandle, LifecycleManager
from archgpu_ollama_bridge.registry import ModelRegistry
from archgpu_ollama_bridge.state import (
    ModelLifecycleState,
    RuntimeStateRecord,
    RuntimeStateStore,
)


class FakeBackendManager:
    def __init__(self, *, healthy: bool = True, fail_on_start: bool = False) -> None:
        self.healthy = healthy
        self.fail_on_start = fail_on_start
        self.started: list[str] = []
        self.stopped: list[str] = []

    def start(self, model) -> BackendHandle:
        self.started.append(model.id)
        if self.fail_on_start:
            raise RuntimeError("backend failed to start")
        return BackendHandle(
            model_id=model.id,
            backend_id=f"llama-{model.id}",
            port=model.port,
        )

    def stop(self, handle: BackendHandle) -> None:
        self.stopped.append(handle.model_id)

    def is_healthy(self, handle: BackendHandle) -> bool:
        return self.healthy


def _build(tmp_path: Path, **kwargs) -> tuple[LifecycleManager, FakeBackendManager, RuntimeStateStore]:
    max_loaded = kwargs.pop("max_loaded_models", 8)
    registry = ModelRegistry.load(Path("config/models.yaml"))
    store = RuntimeStateStore(tmp_path / "runtime-state.json")
    backend = FakeBackendManager(**kwargs)
    lifecycle = LifecycleManager(
        registry,
        store,
        backend,
        max_loaded_models=max_loaded,
    )
    return lifecycle, backend, store


def test_lifecycle_starts_and_persists_ready_state(tmp_path: Path) -> None:
    lifecycle, backend, store = _build(tmp_path)

    handle = asyncio.run(lifecycle.ensure_loaded("code"))
    record = store.get("code")

    assert handle.backend_id == "llama-code"
    assert backend.started == ["code"]
    assert record is not None
    assert record.state == ModelLifecycleState.READY
    assert record.port == 8081


def test_lifecycle_records_failed_state_when_start_fails(tmp_path: Path) -> None:
    lifecycle, _, store = _build(tmp_path, fail_on_start=True)

    with pytest.raises(RuntimeError, match="backend failed to start"):
        asyncio.run(lifecycle.ensure_loaded("finance"))

    record = store.get("finance")
    assert record is not None
    assert record.state == ModelLifecycleState.FAILED
    assert record.error == "backend failed to start"


def test_lifecycle_stops_running_backend(tmp_path: Path) -> None:
    lifecycle, backend, store = _build(tmp_path)

    asyncio.run(lifecycle.ensure_loaded("mistral"))
    asyncio.run(lifecycle.stop("mistral"))

    record = store.get("mistral")
    assert backend.stopped == ["mistral"]
    assert record is not None
    assert record.state == ModelLifecycleState.STOPPED


def test_concurrent_ensure_loaded_starts_backend_only_once(tmp_path: Path) -> None:
    lifecycle, backend, _ = _build(tmp_path)

    async def race() -> tuple[BackendHandle, BackendHandle, BackendHandle]:
        return await asyncio.gather(
            lifecycle.ensure_loaded("code"),
            lifecycle.ensure_loaded("code"),
            lifecycle.ensure_loaded("code.gguf"),
        )

    handles = asyncio.run(race())

    assert backend.started == ["code"]
    assert {h.backend_id for h in handles} == {"llama-code"}


def test_max_loaded_models_evicts_lru_before_starting_new(tmp_path: Path) -> None:
    lifecycle, backend, store = _build(tmp_path, max_loaded_models=2)

    async def scenario() -> None:
        await lifecycle.ensure_loaded("mistral")
        await lifecycle.ensure_loaded("code")
        await lifecycle.ensure_loaded("finance")

    asyncio.run(scenario())

    assert backend.started == ["mistral", "code", "finance"]
    assert backend.stopped == ["mistral"]
    mistral = store.get("mistral")
    assert mistral is not None and mistral.state == ModelLifecycleState.STOPPED


def test_evict_idle_stops_models_past_ttl(tmp_path: Path) -> None:
    lifecycle, backend, store = _build(tmp_path)

    asyncio.run(lifecycle.ensure_loaded("code"))

    record = store.get("code")
    assert record is not None
    stale = record.model_copy(
        update={"last_used_at": datetime.now(UTC) - timedelta(seconds=120)}
    )
    store.save(stale)

    evicted = asyncio.run(lifecycle.evict_idle(idle_ttl_seconds=60.0))

    assert evicted == ["code"]
    assert backend.stopped == ["code"]


def test_evict_idle_keeps_recent_models(tmp_path: Path) -> None:
    lifecycle, backend, _ = _build(tmp_path)

    asyncio.run(lifecycle.ensure_loaded("code"))

    evicted = asyncio.run(lifecycle.evict_idle(idle_ttl_seconds=600.0))

    assert evicted == []
    assert backend.stopped == []


def test_touch_updates_last_used_at_for_lru_ordering(tmp_path: Path) -> None:
    lifecycle, _, store = _build(tmp_path)

    asyncio.run(lifecycle.ensure_loaded("mistral"))
    asyncio.run(lifecycle.ensure_loaded("code"))

    lifecycle.touch("mistral")

    ordered = lifecycle.loaded_model_ids()
    assert ordered[-1] == "mistral"


def test_existing_ready_model_is_reused_when_healthy(tmp_path: Path) -> None:
    lifecycle, backend, _ = _build(tmp_path)

    asyncio.run(lifecycle.ensure_loaded("code"))
    asyncio.run(lifecycle.ensure_loaded("code"))

    assert backend.started == ["code"]


def test_route_aliases_resolve_to_same_lock(tmp_path: Path) -> None:
    """``code`` and ``code.gguf`` must serialise on the same model lock."""

    lifecycle, backend, _ = _build(tmp_path)

    async def race() -> None:
        await asyncio.gather(
            lifecycle.ensure_loaded("code"),
            lifecycle.ensure_loaded("code.gguf"),
        )

    asyncio.run(race())
    assert backend.started == ["code"]


def test_unused_lifecycle_record_creates_no_state(tmp_path: Path) -> None:
    """Importing the lifecycle without any calls leaves the store empty."""

    _, _, store = _build(tmp_path)
    assert store.list() == []


def test_record_validation_uses_pydantic(tmp_path: Path) -> None:
    """Sanity: store accepts records and round-trips them."""

    _, _, store = _build(tmp_path)
    store.save(
        RuntimeStateRecord(model_id="probe", state=ModelLifecycleState.READY, port=9999)
    )
    assert store.get("probe") is not None
