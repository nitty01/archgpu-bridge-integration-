from pathlib import Path

import pytest

from archgpu_ollama_bridge.backends import DockerBackendManager
from archgpu_ollama_bridge.config import Settings
from archgpu_ollama_bridge.lifecycle import UnsupportedBackendManager
from archgpu_ollama_bridge.services import build_backend_manager, build_services


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        registry_path=Path("config/models.yaml"),
        state_path=tmp_path / "runtime-state.json",
    )


def test_build_backend_manager_returns_docker_driver_by_default(tmp_path: Path) -> None:
    manager = build_backend_manager(_settings(tmp_path))
    assert isinstance(manager, DockerBackendManager)


def test_build_backend_manager_supports_disabled_driver(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.backend_driver = "none"
    manager = build_backend_manager(settings)
    assert isinstance(manager, UnsupportedBackendManager)


def test_build_backend_manager_rejects_unknown_driver(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.backend_driver = "kubernetes"
    with pytest.raises(ValueError, match="Unknown backend_driver"):
        build_backend_manager(settings)


def test_build_services_allows_dependency_injection(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    services = build_services(settings, backend_manager=UnsupportedBackendManager())
    assert services.lifecycle is not None
    assert isinstance(services.lifecycle.backend_manager, UnsupportedBackendManager)


def test_build_services_wires_catalogue_downloader_and_pull_manager(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.dynamic_models_path = tmp_path / "dyn.yaml"
    settings.catalogue_path = tmp_path / "missing.yaml"
    settings.backend_models_host_dir = tmp_path / "models"
    services = build_services(settings, backend_manager=UnsupportedBackendManager())
    assert services.catalogue is not None
    assert services.downloader is not None
    assert services.pull_manager is not None
    assert services.settings is settings
