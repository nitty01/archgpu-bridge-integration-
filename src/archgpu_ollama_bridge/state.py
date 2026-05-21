from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class ModelLifecycleState(StrEnum):
    REGISTERED = "registered"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


class RuntimeStateRecord(BaseModel):
    model_id: str
    state: ModelLifecycleState = ModelLifecycleState.REGISTERED
    port: int | None = None
    backend_id: str | None = None
    last_used_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None


class RuntimeStateDocument(BaseModel):
    models: dict[str, RuntimeStateRecord] = Field(default_factory=dict)


class RuntimeStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._document = self._load()

    def _load(self) -> RuntimeStateDocument:
        if not self.path.exists():
            return RuntimeStateDocument()
        return RuntimeStateDocument.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, record: RuntimeStateRecord) -> None:
        self._document.models[record.model_id] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            self._document.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def get(self, model_id: str) -> RuntimeStateRecord | None:
        return self._document.models.get(model_id)

    def list(self) -> list[RuntimeStateRecord]:
        return list(self._document.models.values())
