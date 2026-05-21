from pathlib import Path

import pytest

from archgpu_ollama_bridge.registry import ModelRegistry


def test_registry_loads_models_and_aliases() -> None:
    registry = ModelRegistry.load(Path("config/models.yaml"))

    model = registry.get("mistral.gguf")

    assert model.id == "mistral"
    assert model.port == 8080
    assert registry.get("mistral").openai_name == "mistral.gguf"
    assert registry.get("code").ollama_name == "code"
    assert len(registry.list_models()) == 3


def test_registry_rejects_duplicate_aliases(tmp_path: Path) -> None:
    registry_file = tmp_path / "models.yaml"
    registry_file.write_text(
        """
models:
  - id: alpha
    gguf_path: /models/alpha.gguf
    openai_name: shared
    ollama_name: alpha
    port: 8001
  - id: beta
    gguf_path: /models/beta.gguf
    openai_name: beta
    ollama_name: shared
    port: 8002
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate model alias: shared"):
        ModelRegistry.load(registry_file)
