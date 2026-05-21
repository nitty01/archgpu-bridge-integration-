"""Concrete `BackendManager` drivers.

Currently provides a Docker-based driver that launches one ``llama-server``
container per model on the configured host port.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .lifecycle import BackendHandle
from .registry import ModelRecord

logger = logging.getLogger(__name__)


SubprocessRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]
HealthProbe = Callable[[str], bool]


def _default_runner(args: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(list(args), capture_output=True, text=True, check=False)


def _default_health_probe(url: str) -> bool:
    try:
        response = httpx.get(url, timeout=2.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


@dataclass(slots=True)
class DockerBackendSettings:
    """Tunables for the Docker llama.cpp driver.

    Defaults match the working Intel Arc + SYCL stack documented in
    ``CHAT_CONTEXT_SUMMARY.md``.
    """

    image: str = "local/llama.cpp:server-intel"
    container_prefix: str = "archgpu-bridge-"
    models_host_dir: Path = Path("/home/nitender-kumar/llm/models")
    models_container_dir: str = "/models"
    devices: tuple[str, ...] = ("/dev/dri/renderD128", "/dev/dri/card1")
    env: tuple[str, ...] = ("SYCL_DEVICE_FILTER=level_zero:gpu",)
    container_port: int = 8080
    host_bind: str = "127.0.0.1"
    server_host: str = "0.0.0.0"
    extra_server_args: tuple[str, ...] = ("-ngl", "999", "--cache-ram", "0", "-np", "1")
    startup_timeout_seconds: float = 90.0
    health_path: str = "/health"
    docker_binary: str = "docker"
    poll_interval_seconds: float = 1.0


class DockerCommandError(RuntimeError):
    """Raised when a ``docker`` invocation exits non-zero."""

    def __init__(self, args: Sequence[str], proc: "subprocess.CompletedProcess[str]") -> None:
        self.args = list(args)
        self.returncode = proc.returncode
        self.stderr = (proc.stderr or "").strip()
        super().__init__(
            f"docker command failed (rc={proc.returncode}): "
            f"{shlex.join(self.args)} :: {self.stderr}"
        )


@dataclass(slots=True)
class _RunRecord:
    args: list[str]
    proc: "subprocess.CompletedProcess[str]"


class DockerBackendManager:
    """Implements ``BackendManager`` by driving the ``docker`` CLI.

    One container per model, named ``<container_prefix><model_id>``, with the
    host ``model.port`` mapped to the in-container llama-server port.
    """

    def __init__(
        self,
        settings: DockerBackendSettings | None = None,
        *,
        runner: SubprocessRunner | None = None,
        health_probe: HealthProbe | None = None,
        sleeper: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings or DockerBackendSettings()
        self._runner: SubprocessRunner = runner or _default_runner
        self._health_probe: HealthProbe = health_probe or _default_health_probe
        self._sleep: Callable[[float], None] = sleeper or time.sleep
        self._now: Callable[[], float] = clock or time.monotonic
        self._history: list[_RunRecord] = []

    def start(self, model: ModelRecord) -> BackendHandle:
        name = self._container_name(model.id)
        container_model_path = self._container_model_path(model)

        self._run(
            [self.settings.docker_binary, "rm", "-f", name],
            check=False,
        )

        cmd = self._build_run_command(model, name, container_model_path)
        logger.info("starting llama-server container %s for model %s", name, model.id)
        self._run(cmd, check=True)

        handle = BackendHandle(
            model_id=model.id,
            backend_id=name,
            port=model.port,
        )

        if not self._wait_for_health(handle):
            logger.warning(
                "container %s did not become healthy within %.1fs; rolling back",
                name,
                self.settings.startup_timeout_seconds,
            )
            self.stop(handle)
            raise RuntimeError(
                f"llama-server for model {model.id} did not become healthy in time"
            )

        logger.info("container %s ready on port %d", name, model.port)
        return handle

    def stop(self, handle: BackendHandle) -> None:
        logger.info("stopping container %s", handle.backend_id)
        self._run(
            [self.settings.docker_binary, "rm", "-f", handle.backend_id],
            check=False,
        )

    def is_healthy(self, handle: BackendHandle) -> bool:
        url = (
            f"http://127.0.0.1:{handle.port}"
            f"{self.settings.health_path if self.settings.health_path.startswith('/') else '/' + self.settings.health_path}"
        )
        return self._health_probe(url)

    def _wait_for_health(self, handle: BackendHandle) -> bool:
        deadline = self._now() + self.settings.startup_timeout_seconds
        while self._now() < deadline:
            if self.is_healthy(handle):
                return True
            self._sleep(self.settings.poll_interval_seconds)
        return self.is_healthy(handle)

    def _build_run_command(
        self,
        model: ModelRecord,
        container_name: str,
        container_model_path: str,
    ) -> list[str]:
        s = self.settings
        cmd: list[str] = [
            s.docker_binary,
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{s.host_bind}:{model.port}:{s.container_port}",
        ]
        for device in s.devices:
            cmd += ["--device", f"{device}:{device}:rwm"]
        for env_pair in s.env:
            cmd += ["-e", env_pair]
        cmd += [
            "-v",
            f"{s.models_host_dir.as_posix()}:{s.models_container_dir}",
            s.image,
            "-m",
            container_model_path,
            "--host",
            s.server_host,
            "--port",
            str(s.container_port),
            "-c",
            str(model.context_length),
        ]
        cmd += list(s.extra_server_args)
        cmd += list(model.backend_args)
        return cmd

    def _run(self, args: Sequence[str], *, check: bool) -> "subprocess.CompletedProcess[str]":
        logger.debug("docker exec: %s", shlex.join(list(args)))
        proc = self._runner(args)
        self._history.append(_RunRecord(args=list(args), proc=proc))
        if check and proc.returncode != 0:
            raise DockerCommandError(args, proc)
        return proc

    def _container_name(self, model_id: str) -> str:
        return f"{self.settings.container_prefix}{model_id}"

    def _container_model_path(self, model: ModelRecord) -> str:
        host_path = model.gguf_path.expanduser().resolve()
        base = self.settings.models_host_dir.expanduser().resolve()
        try:
            relative = host_path.relative_to(base)
        except ValueError as exc:
            raise RuntimeError(
                f"model gguf_path {model.gguf_path} is not under models_host_dir "
                f"{self.settings.models_host_dir}"
            ) from exc
        return f"{self.settings.models_container_dir.rstrip('/')}/{relative.as_posix()}"

    @property
    def command_history(self) -> list[list[str]]:
        """Read-only view of the docker invocations issued so far (for tests)."""

        return [record.args for record in self._history]


def build_docker_settings_from_app(app_settings: Any) -> DockerBackendSettings:
    """Translate a flat ``Settings`` object into ``DockerBackendSettings``."""

    return DockerBackendSettings(
        image=app_settings.backend_image,
        container_prefix=app_settings.backend_container_prefix,
        models_host_dir=Path(app_settings.backend_models_host_dir),
        models_container_dir=app_settings.backend_models_container_dir,
        devices=tuple(app_settings.backend_devices),
        env=tuple(app_settings.backend_env),
        container_port=app_settings.backend_container_port,
        host_bind=app_settings.backend_host_bind,
        server_host=app_settings.backend_server_host,
        extra_server_args=tuple(app_settings.backend_extra_args),
        startup_timeout_seconds=app_settings.backend_startup_timeout_seconds,
        health_path=app_settings.backend_health_path,
        docker_binary=app_settings.backend_docker_binary,
    )
