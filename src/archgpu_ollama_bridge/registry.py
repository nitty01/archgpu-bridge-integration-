from __future__ import annotations

import threading
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelRecord(BaseModel):
    id: str
    gguf_path: Path
    openai_name: str
    ollama_name: str
    display_name: str | None = None
    port: int = Field(gt=0)
    context_length: int = Field(default=8192, gt=0)
    tags: list[str] = Field(default_factory=list)
    backend_args: list[str] = Field(default_factory=list)
    source: str = Field(default="static")
    hf_repo: str | None = None
    hf_filename: str | None = None
    hf_revision: str | None = None


class RegistryDocument(BaseModel):
    models: list[ModelRecord] = Field(default_factory=list)


class PortPool:
    """Allocate ports from a contiguous range, reserving any already in use."""

    def __init__(self, start: int, end: int, *, reserved: set[int] | None = None) -> None:
        if end < start:
            raise ValueError(f"Invalid port range: {start}-{end}")
        self._start = start
        self._end = end
        self._reserved: set[int] = set(reserved or set())

    def reserve(self, port: int) -> None:
        self._reserved.add(port)

    def release(self, port: int) -> None:
        self._reserved.discard(port)

    def allocate(self) -> int:
        for candidate in range(self._start, self._end + 1):
            if candidate not in self._reserved:
                self._reserved.add(candidate)
                return candidate
        raise RuntimeError(
            f"No free ports left in {self._start}-{self._end} (all {len(self._reserved)} reserved)"
        )


class ModelRegistry:
    """Two-layer registry merging static models.yaml with a writable dynamic file."""

    def __init__(
        self,
        *,
        static_models: dict[str, ModelRecord],
        dynamic_models: dict[str, ModelRecord],
        dynamic_path: Path | None,
        port_pool: PortPool | None = None,
    ) -> None:
        self._static = static_models
        self._dynamic = dynamic_models
        self._dynamic_path = dynamic_path
        self._port_pool = port_pool or PortPool(18000, 18099)
        self._lock = threading.RLock()
        for record in dynamic_models.values():
            self._port_pool.reserve(record.port)
        for record in static_models.values():
            self._port_pool.reserve(record.port)
        self._rebuild_aliases()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        dynamic_path: str | Path | None = None,
        port_pool: PortPool | None = None,
    ) -> "ModelRegistry":
        static_models = _load_yaml(Path(path))
        dynamic_models: dict[str, ModelRecord] = {}
        dyn_path: Path | None = None

        if dynamic_path is not None:
            dyn_path = Path(dynamic_path)
            if dyn_path.exists():
                dynamic_models = _load_yaml(dyn_path, source="dynamic")

        for alias, model_id in _build_aliases(static_models).items():
            for dyn in dynamic_models.values():
                if alias in {dyn.id, dyn.openai_name, dyn.ollama_name} and dyn.id != model_id:
                    raise ValueError(
                        f"Dynamic model {dyn.id!r} clashes with static alias {alias!r}"
                    )

        return cls(
            static_models=static_models,
            dynamic_models=dynamic_models,
            dynamic_path=dyn_path,
            port_pool=port_pool,
        )

    def list_models(self) -> list[ModelRecord]:
        with self._lock:
            return list(self._static.values()) + list(self._dynamic.values())

    def get(self, identifier: str) -> ModelRecord:
        with self._lock:
            model_id = self._aliases.get(identifier)
            if model_id is None:
                raise KeyError(f"Unknown model: {identifier}")
            if model_id in self._static:
                return self._static[model_id]
            return self._dynamic[model_id]

    def has(self, identifier: str) -> bool:
        with self._lock:
            return identifier in self._aliases

    def is_dynamic(self, identifier: str) -> bool:
        with self._lock:
            model_id = self._aliases.get(identifier)
            return model_id is not None and model_id in self._dynamic

    def register(self, record: ModelRecord) -> ModelRecord:
        """Add a dynamic model; persist immediately."""

        with self._lock:
            if record.id in self._static:
                raise ValueError(f"Cannot override static model id: {record.id}")
            for alias in {record.id, record.openai_name, record.ollama_name}:
                existing = self._aliases.get(alias)
                if existing is not None and existing != record.id:
                    raise ValueError(f"Alias {alias!r} already in use by {existing!r}")
            self._dynamic[record.id] = record
            self._port_pool.reserve(record.port)
            self._rebuild_aliases()
            self._save_dynamic_locked()
            return record

    def unregister(self, identifier: str) -> ModelRecord:
        with self._lock:
            model_id = self._aliases.get(identifier)
            if model_id is None or model_id not in self._dynamic:
                raise KeyError(f"Not a dynamic model: {identifier}")
            record = self._dynamic.pop(model_id)
            self._port_pool.release(record.port)
            self._rebuild_aliases()
            self._save_dynamic_locked()
            return record

    def allocate_port(self) -> int:
        with self._lock:
            return self._port_pool.allocate()

    def release_port(self, port: int) -> None:
        with self._lock:
            self._port_pool.release(port)

    def save_dynamic(self) -> None:
        with self._lock:
            self._save_dynamic_locked()

    def _save_dynamic_locked(self) -> None:
        if self._dynamic_path is None:
            return
        self._dynamic_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "models": [
                _dump_model(record) for record in self._dynamic.values()
            ]
        }
        self._dynamic_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )

    def _rebuild_aliases(self) -> None:
        aliases: dict[str, str] = {}
        for record in (*self._static.values(), *self._dynamic.values()):
            for alias in {record.id, record.openai_name, record.ollama_name}:
                if alias in aliases and aliases[alias] != record.id:
                    raise ValueError(f"Duplicate model alias: {alias}")
                aliases[alias] = record.id
        self._aliases = aliases


def _build_aliases(models: dict[str, ModelRecord]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for record in models.values():
        for alias in {record.id, record.openai_name, record.ollama_name}:
            aliases[alias] = record.id
    return aliases


def _load_yaml(path: Path, *, source: str = "static") -> dict[str, ModelRecord]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    document = RegistryDocument.model_validate(payload)
    models: dict[str, ModelRecord] = {}
    for record in document.models:
        record = record.model_copy(update={"source": source}) if source != record.source else record
        if record.id in models:
            raise ValueError(f"Duplicate model id: {record.id}")
        models[record.id] = record
    return models


def _dump_model(record: ModelRecord) -> dict:
    payload = record.model_dump(mode="json", exclude_none=True)
    if "gguf_path" in payload:
        payload["gguf_path"] = str(payload["gguf_path"])
    return payload
