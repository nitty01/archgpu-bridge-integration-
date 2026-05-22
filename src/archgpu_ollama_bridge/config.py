from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="ARCHGPU OLLAMA Bridge")
    app_env: str = Field(default="development")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    registry_path: Path = Field(default=Path("config/models.yaml"))
    state_path: Path = Field(default=Path("data/runtime-state.json"))

    backend_driver: str = Field(default="docker")
    backend_image: str = Field(default="local/llama.cpp:server-intel")
    backend_container_prefix: str = Field(default="archgpu-bridge-")
    backend_models_host_dir: Path = Field(
        default_factory=lambda: (Path.home() / "llm" / "models")
    )
    backend_models_container_dir: str = Field(default="/models")
    backend_devices: list[str] = Field(
        default_factory=lambda: ["/dev/dri/renderD128", "/dev/dri/card1"]
    )
    backend_env: list[str] = Field(
        default_factory=lambda: ["SYCL_DEVICE_FILTER=level_zero:gpu"]
    )
    backend_container_port: int = Field(default=8080)
    backend_host_bind: str = Field(default="127.0.0.1")
    backend_server_host: str = Field(default="0.0.0.0")
    backend_extra_args: list[str] = Field(
        default_factory=lambda: ["-ngl", "999", "--cache-ram", "0", "-np", "1"]
    )
    backend_startup_timeout_seconds: float = Field(default=300.0)
    backend_health_path: str = Field(default="/health")
    backend_docker_binary: str = Field(default="docker")

    idle_ttl_seconds: float = Field(default=600.0)
    max_loaded_models: int = Field(default=2)

    catalogue_path: Path = Field(default=Path("config/catalogue.yaml"))
    dynamic_models_path: Path = Field(default=Path("data/downloaded_models.yaml"))
    dynamic_port_range: tuple[int, int] = Field(default=(18000, 18099))
    dynamic_default_context_length: int = Field(default=8192, gt=0)
    hf_base_url: str = Field(default="https://huggingface.co")
    hf_allow_orgs: list[str] = Field(default_factory=list)
    hf_discovery_enabled: bool = Field(default=True)
    hf_discovery_ttl_seconds: int = Field(default=1800, ge=0)
    hf_discovery_per_query_limit: int = Field(default=12, ge=1, le=100)
    hf_discovery_max_models: int = Field(default=80, ge=1, le=500)
    hf_discovery_queries: list[str] = Field(
        default_factory=lambda: [
            "Qwen3 GGUF",
            "Qwen2.5 Instruct GGUF",
            "DeepSeek-R1 Distill GGUF",
            "Mistral Small Instruct GGUF",
            "Phi-4 GGUF",
            "Llama Instruct GGUF",
        ]
    )
    hf_discovery_owners: list[str] = Field(default_factory=list)
    pull_max_bytes: int | None = Field(default=None)
    pull_request_timeout_seconds: float = Field(default=3600.0)

    model_config = SettingsConfigDict(
        env_prefix="ARCHGPU_BRIDGE_",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
