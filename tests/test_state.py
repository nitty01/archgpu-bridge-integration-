from pathlib import Path

from archgpu_ollama_bridge.state import (
    ModelLifecycleState,
    RuntimeStateRecord,
    RuntimeStateStore,
)


def test_state_store_persists_and_reloads_records(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime-state.json"
    store = RuntimeStateStore(state_path)

    store.save(
        RuntimeStateRecord(
            model_id="code",
            state=ModelLifecycleState.READY,
            port=8081,
            backend_id="llama-code",
        )
    )

    reloaded = RuntimeStateStore(state_path)
    record = reloaded.get("code")

    assert record is not None
    assert record.state == ModelLifecycleState.READY
    assert record.port == 8081
    assert record.backend_id == "llama-code"
