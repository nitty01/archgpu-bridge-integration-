from pathlib import Path

import pytest

from archgpu_ollama_bridge.registry import ModelRecord, ModelRegistry, PortPool


def _static_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(
        """
models:
  - id: mistral
    gguf_path: /models/mistral.gguf
    openai_name: mistral.gguf
    ollama_name: mistral
    port: 8080
""".strip(),
        encoding="utf-8",
    )
    return p


def _record(**overrides) -> ModelRecord:
    base = dict(
        id="qwen-coder",
        gguf_path=Path("/models/qwen.gguf"),
        openai_name="qwen-coder.gguf",
        ollama_name="qwen-coder",
        port=18000,
        context_length=8192,
        tags=["dynamic"],
        source="dynamic",
        hf_repo="Qwen/Coder",
        hf_filename="qwen.gguf",
        hf_revision="main",
    )
    base.update(overrides)
    return ModelRecord(**base)


def test_register_and_unregister_persists_to_dynamic_yaml(tmp_path: Path) -> None:
    static = _static_yaml(tmp_path)
    dynamic = tmp_path / "downloaded_models.yaml"

    registry = ModelRegistry.load(static, dynamic_path=dynamic)
    registry.register(_record())

    assert dynamic.exists()
    assert "qwen-coder" in dynamic.read_text(encoding="utf-8")

    fresh = ModelRegistry.load(static, dynamic_path=dynamic)
    assert fresh.has("qwen-coder")
    assert fresh.get("qwen-coder").hf_repo == "Qwen/Coder"

    fresh.unregister("qwen-coder")
    assert not fresh.has("qwen-coder")
    assert "qwen-coder" not in dynamic.read_text(encoding="utf-8")


def test_register_rejects_alias_collision_with_static(tmp_path: Path) -> None:
    registry = ModelRegistry.load(
        _static_yaml(tmp_path),
        dynamic_path=tmp_path / "dyn.yaml",
    )
    with pytest.raises(ValueError, match="already in use"):
        registry.register(_record(id="foo", openai_name="mistral.gguf", ollama_name="foo"))


def test_register_rejects_overriding_static_id(tmp_path: Path) -> None:
    registry = ModelRegistry.load(
        _static_yaml(tmp_path),
        dynamic_path=tmp_path / "dyn.yaml",
    )
    with pytest.raises(ValueError, match="static model id"):
        registry.register(_record(id="mistral", openai_name="x.gguf", ollama_name="x"))


def test_is_dynamic_distinguishes_static_and_dynamic(tmp_path: Path) -> None:
    registry = ModelRegistry.load(
        _static_yaml(tmp_path),
        dynamic_path=tmp_path / "dyn.yaml",
    )
    assert not registry.is_dynamic("mistral")
    registry.register(_record())
    assert registry.is_dynamic("qwen-coder")


def test_port_pool_allocates_in_range_and_skips_reserved() -> None:
    pool = PortPool(18000, 18002)
    pool.reserve(18000)
    assert pool.allocate() == 18001
    assert pool.allocate() == 18002
    with pytest.raises(RuntimeError):
        pool.allocate()


def test_port_pool_release_makes_port_available() -> None:
    pool = PortPool(20000, 20001)
    a = pool.allocate()
    b = pool.allocate()
    pool.release(a)
    c = pool.allocate()
    assert c == a
    assert b != c


def test_registry_allocate_port_uses_pool(tmp_path: Path) -> None:
    registry = ModelRegistry.load(
        _static_yaml(tmp_path),
        dynamic_path=tmp_path / "dyn.yaml",
        port_pool=PortPool(18000, 18001),
    )
    p = registry.allocate_port()
    assert p == 18000
    p2 = registry.allocate_port()
    assert p2 == 18001


def test_registry_unregister_releases_port(tmp_path: Path) -> None:
    registry = ModelRegistry.load(
        _static_yaml(tmp_path),
        dynamic_path=tmp_path / "dyn.yaml",
        port_pool=PortPool(18000, 18000),
    )
    record = _record(port=18000)
    registry.register(record)
    registry.unregister("qwen-coder")
    # After release we should be able to register a fresh one on the same port.
    record2 = _record(id="other", openai_name="other.gguf", ollama_name="other", port=18000)
    registry.register(record2)


def test_registry_get_returns_dynamic_record(tmp_path: Path) -> None:
    registry = ModelRegistry.load(
        _static_yaml(tmp_path),
        dynamic_path=tmp_path / "dyn.yaml",
    )
    registry.register(_record())
    record = registry.get("qwen-coder")
    assert record.id == "qwen-coder"
    assert record.source == "dynamic"


def test_registry_load_handles_missing_dynamic_file(tmp_path: Path) -> None:
    registry = ModelRegistry.load(
        _static_yaml(tmp_path),
        dynamic_path=tmp_path / "does-not-exist.yaml",
    )
    assert [m.id for m in registry.list_models()] == ["mistral"]
