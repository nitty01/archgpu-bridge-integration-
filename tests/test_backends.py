import subprocess
from pathlib import Path

import pytest

from archgpu_ollama_bridge.backends import (
    DockerBackendManager,
    DockerBackendSettings,
    DockerCommandError,
)
from archgpu_ollama_bridge.lifecycle import BackendHandle
from archgpu_ollama_bridge.registry import ModelRecord


def _proc(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class FakeRunner:
    def __init__(self, responses: dict[str, subprocess.CompletedProcess[str]] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.responses = responses or {}

    def __call__(self, args):
        args_list = list(args)
        self.calls.append(args_list)
        action = args_list[1] if len(args_list) > 1 else ""
        return self.responses.get(action, _proc(stdout="container-id\n"))


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def _settings(tmp_models: Path) -> DockerBackendSettings:
    return DockerBackendSettings(
        models_host_dir=tmp_models,
        models_container_dir="/models",
        startup_timeout_seconds=5.0,
        poll_interval_seconds=0.5,
    )


def _model(tmp_models: Path) -> ModelRecord:
    return ModelRecord(
        id="code",
        gguf_path=tmp_models / "code.gguf",
        openai_name="code.gguf",
        ollama_name="code",
        port=8081,
        context_length=8192,
        tags=["code"],
    )


def test_start_runs_expected_docker_command(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "code.gguf").touch()

    runner = FakeRunner()
    clock = FakeClock()

    healthy_responses = iter([False, True])
    manager = DockerBackendManager(
        settings=_settings(models_dir),
        runner=runner,
        health_probe=lambda url: next(healthy_responses),
        sleeper=lambda dt: clock.advance(dt),
        clock=clock,
    )

    handle = manager.start(_model(models_dir))

    assert handle.backend_id == "archgpu-bridge-code"
    assert handle.port == 8081

    assert runner.calls[0][:3] == ["docker", "rm", "-f"]
    assert runner.calls[0][3] == "archgpu-bridge-code"

    run_cmd = runner.calls[1]
    assert run_cmd[:4] == ["docker", "run", "-d", "--name"]
    assert "archgpu-bridge-code" in run_cmd
    assert "-p" in run_cmd
    p_index = run_cmd.index("-p")
    assert run_cmd[p_index + 1] == "127.0.0.1:8081:8080"
    assert "--device" in run_cmd
    assert "-e" in run_cmd
    e_index = run_cmd.index("-e")
    assert run_cmd[e_index + 1] == "SYCL_DEVICE_FILTER=level_zero:gpu"
    assert "-v" in run_cmd
    v_index = run_cmd.index("-v")
    assert run_cmd[v_index + 1] == f"{models_dir.as_posix()}:/models"
    assert "local/llama.cpp:server-intel" in run_cmd
    m_index = run_cmd.index("-m")
    assert run_cmd[m_index + 1] == "/models/code.gguf"
    c_index = run_cmd.index("-c")
    assert run_cmd[c_index + 1] == "8192"
    assert "-ngl" in run_cmd and "999" in run_cmd
    assert "--cache-ram" in run_cmd
    assert "-np" in run_cmd


def test_start_raises_on_docker_run_failure(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "code.gguf").touch()

    runner = FakeRunner(
        responses={
            "rm": _proc(returncode=0),
            "run": _proc(returncode=1, stderr="bind: address already in use"),
        }
    )

    manager = DockerBackendManager(
        settings=_settings(models_dir),
        runner=runner,
        health_probe=lambda url: True,
        sleeper=lambda dt: None,
        clock=FakeClock(),
    )

    with pytest.raises(DockerCommandError, match="address already in use"):
        manager.start(_model(models_dir))


def test_start_rolls_back_when_health_never_returns_true(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "code.gguf").touch()

    runner = FakeRunner()
    clock = FakeClock()

    manager = DockerBackendManager(
        settings=_settings(models_dir),
        runner=runner,
        health_probe=lambda url: False,
        sleeper=lambda dt: clock.advance(dt),
        clock=clock,
    )

    with pytest.raises(RuntimeError, match="did not become healthy"):
        manager.start(_model(models_dir))

    rm_calls = [c for c in runner.calls if c[:3] == ["docker", "rm", "-f"]]
    assert len(rm_calls) >= 2


def test_start_rejects_model_outside_host_dir(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "code.gguf").touch()

    model = ModelRecord(
        id="code",
        gguf_path=elsewhere / "code.gguf",
        openai_name="code.gguf",
        ollama_name="code",
        port=8081,
    )

    manager = DockerBackendManager(
        settings=_settings(models_dir),
        runner=FakeRunner(),
        health_probe=lambda url: True,
        sleeper=lambda dt: None,
        clock=FakeClock(),
    )

    with pytest.raises(RuntimeError, match="not under models_host_dir"):
        manager.start(model)


def test_stop_calls_docker_rm_force() -> None:
    runner = FakeRunner()
    manager = DockerBackendManager(
        settings=DockerBackendSettings(),
        runner=runner,
        health_probe=lambda url: True,
        sleeper=lambda dt: None,
        clock=FakeClock(),
    )

    manager.stop(BackendHandle(model_id="code", backend_id="archgpu-bridge-code", port=8081))

    assert runner.calls[-1] == ["docker", "rm", "-f", "archgpu-bridge-code"]


def test_is_healthy_uses_configured_health_path() -> None:
    seen_urls: list[str] = []

    def probe(url: str) -> bool:
        seen_urls.append(url)
        return True

    manager = DockerBackendManager(
        settings=DockerBackendSettings(health_path="/health"),
        runner=FakeRunner(),
        health_probe=probe,
        sleeper=lambda dt: None,
        clock=FakeClock(),
    )

    assert manager.is_healthy(BackendHandle(model_id="code", backend_id="x", port=8081))
    assert seen_urls == ["http://127.0.0.1:8081/health"]
