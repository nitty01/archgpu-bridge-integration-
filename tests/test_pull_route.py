import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi.testclient import TestClient

from archgpu_ollama_bridge.catalogue import Catalogue, HFRef
from archgpu_ollama_bridge.config import Settings
from archgpu_ollama_bridge.main import create_app
from archgpu_ollama_bridge.pulls import PullManager
from archgpu_ollama_bridge.registry import ModelRegistry, PortPool
from archgpu_ollama_bridge.router import RouteTarget
from archgpu_ollama_bridge.services import AppServices, ProxyResponse


class FakeRouter:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    async def route(self, identifier: str) -> RouteTarget:
        from archgpu_ollama_bridge.lifecycle import BackendHandle

        model = self.registry.get(identifier)
        return RouteTarget(
            model=model,
            handle=BackendHandle(model_id=model.id, backend_id="x", port=model.port),
            base_url=f"http://127.0.0.1:{model.port}",
        )


class FakeProxyClient:
    async def post_json(self, *_, **__) -> ProxyResponse:
        return ProxyResponse(status_code=200, json_body={})

    async def stream_post(self, *_, **__) -> AsyncIterator[bytes]:
        if False:
            yield b""


class FakeDownloader:
    def __init__(self, target_dir: Path, events: list[dict]) -> None:
        self._target_dir = target_dir
        self._events = events

    def target_path(self, ref: HFRef) -> Path:
        return self._target_dir / ref.safe_local_filename

    def already_present(self, ref: HFRef) -> Path | None:
        return None

    async def pull(self, ref: HFRef) -> AsyncIterator[bytes]:
        # Simulate creating the file then emitting events
        path = self.target_path(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUFFAKE")
        for event in self._events:
            yield (json.dumps(event) + "\n").encode("utf-8")


def _build(
    tmp_path: Path,
    *,
    events: list[dict],
    catalogue_body: str = "",
    allow_orgs: tuple[str, ...] = (),
):
    static = tmp_path / "models.yaml"
    static.write_text(
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
    cat_path = tmp_path / "catalogue.yaml"
    if catalogue_body:
        cat_path.write_text(catalogue_body, encoding="utf-8")

    settings = Settings(
        registry_path=static,
        state_path=tmp_path / "state.json",
        catalogue_path=cat_path,
        dynamic_models_path=tmp_path / "downloaded.yaml",
        backend_models_host_dir=tmp_path / "models",
        dynamic_port_range=(18000, 18099),
    )
    registry = ModelRegistry.load(
        static,
        dynamic_path=settings.dynamic_models_path,
        port_pool=PortPool(*settings.dynamic_port_range),
    )
    catalogue = Catalogue.load(cat_path, allow_orgs=allow_orgs)
    downloader = FakeDownloader(settings.backend_models_host_dir, events)
    pull_manager = PullManager(
        registry=registry,
        catalogue=catalogue,
        downloader=downloader,
        settings=settings,
    )
    services = AppServices(
        registry=registry,
        router=FakeRouter(registry),
        proxy_client=FakeProxyClient(),
        settings=settings,
        catalogue=catalogue,
        downloader=downloader,
        pull_manager=pull_manager,
    )
    return TestClient(create_app(services)), registry, downloader


def test_pull_streams_ndjson_and_registers_model(tmp_path: Path) -> None:
    events = [
        {"status": "pulling manifest"},
        {"status": "downloading", "digest": "sha256:abc", "total": 100, "completed": 100},
        {"status": "success", "digest": "sha256:abc", "total": 100, "completed": 100},
    ]
    client, registry, _ = _build(tmp_path, events=events)

    response = client.post(
        "/api/pull",
        json={"name": "Qwen/Test-GGUF:test.gguf"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")

    lines = [json.loads(line) for line in response.text.splitlines() if line]
    statuses = [event["status"] for event in lines]
    assert statuses[0] == "pulling manifest"
    assert "success" in statuses
    assert lines[-1]["status"] == "registered"

    listed_ids = [m.id for m in registry.list_models()]
    assert any("test" in mid for mid in listed_ids)


def test_pull_with_invalid_name_yields_error_event(tmp_path: Path) -> None:
    client, _, _ = _build(tmp_path, events=[])

    response = client.post("/api/pull", json={"name": "garbage"})
    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.splitlines() if line]
    assert lines[-1]["status"] == "error"


def test_pull_requires_name_or_model(tmp_path: Path) -> None:
    client, _, _ = _build(tmp_path, events=[])
    response = client.post("/api/pull", json={})
    assert response.status_code == 422


def test_pull_alias_resolves_via_catalogue(tmp_path: Path) -> None:
    body = """
models:
  - alias: my-model
    repo: Qwen/Foo
    filename: foo.gguf
    context_length: 4096
""".strip()
    events = [
        {"status": "pulling manifest"},
        {"status": "success", "total": 1, "completed": 1},
    ]
    client, registry, _ = _build(tmp_path, events=events, catalogue_body=body)

    response = client.post("/api/pull", json={"name": "my-model"})
    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.splitlines() if line]
    assert lines[-1]["status"] == "registered"

    record = next(
        m for m in registry.list_models() if m.id != "mistral"
    )
    assert record.context_length == 4096
    assert record.hf_repo == "Qwen/Foo"
    assert record.hf_filename == "foo.gguf"


def test_pull_idempotent_when_model_already_registered(tmp_path: Path) -> None:
    body = """
models:
  - alias: my-model
    repo: Qwen/Foo
    filename: foo.gguf
""".strip()
    events = [
        {"status": "pulling manifest"},
        {"status": "success", "total": 1, "completed": 1},
    ]
    client, registry, _ = _build(tmp_path, events=events, catalogue_body=body)

    first = client.post("/api/pull", json={"name": "my-model"})
    assert first.status_code == 200
    n_models = len(registry.list_models())

    second = client.post("/api/pull", json={"name": "my-model"})
    assert second.status_code == 200
    assert len(registry.list_models()) == n_models


def test_pull_status_endpoint_reports_progress_and_registration(tmp_path: Path) -> None:
    body = """
models:
  - alias: my-model
    repo: Qwen/Foo
    filename: foo.gguf
""".strip()
    events = [
        {"status": "pulling manifest"},
        {"status": "downloading", "digest": "sha256:abc", "total": 10, "completed": 10},
        {"status": "success", "digest": "sha256:abc", "total": 10, "completed": 10},
    ]
    client, _, _ = _build(tmp_path, events=events, catalogue_body=body)

    pull_response = client.post("/api/pull", json={"name": "my-model"})
    assert pull_response.status_code == 200

    status = client.get("/api/pull/status?model=my-model")
    assert status.status_code == 200
    models = status.json()["models"]
    assert len(models) == 1
    assert models[0]["model"] == "my-model"
    assert models[0]["status"] == "registered"

    tags = client.get("/api/tags")
    assert tags.status_code == 200
    tagged = [m for m in tags.json()["models"] if m["name"] == "my-model"]
    assert tagged
    assert tagged[0]["pull_status"]["status"] == "registered"


def test_catalogue_endpoint_separates_downloadable_and_installed(tmp_path: Path) -> None:
    body = """
models:
  - alias: my-model
    repo: Qwen/Foo
    filename: foo.gguf
  - alias: another-model
    repo: Qwen/Bar
    filename: bar.gguf
""".strip()
    events = [
        {"status": "pulling manifest"},
        {"status": "success", "total": 1, "completed": 1},
    ]
    client, _, _ = _build(tmp_path, events=events, catalogue_body=body)

    before = client.get("/api/catalogue")
    assert before.status_code == 200
    before_items = {m["alias"]: m for m in before.json()["models"]}
    assert before_items["my-model"]["downloadable"] is True
    assert before_items["my-model"]["installed"] is False

    pull = client.post("/api/pull", json={"name": "my-model"})
    assert pull.status_code == 200

    after = client.get("/api/catalogue")
    assert after.status_code == 200
    after_items = {m["alias"]: m for m in after.json()["models"]}
    assert after_items["my-model"]["installed"] is True
    assert after_items["my-model"]["downloadable"] is False
    assert after_items["another-model"]["installed"] is False


def test_delete_dynamic_model_unregisters_and_removes_file(tmp_path: Path) -> None:
    body = """
models:
  - alias: my-model
    repo: Qwen/Foo
    filename: foo.gguf
""".strip()
    events = [
        {"status": "pulling manifest"},
        {"status": "success", "total": 1, "completed": 1},
    ]
    client, registry, downloader = _build(tmp_path, events=events, catalogue_body=body)

    client.post("/api/pull", json={"name": "my-model"})

    record = next(m for m in registry.list_models() if m.id != "mistral")
    assert record.gguf_path.exists()

    response = client.post("/api/delete", json={"name": record.ollama_name})
    assert response.status_code == 200
    body_json = response.json()
    assert body_json["status"] == "success"
    assert body_json["removed_file"] is True
    assert not registry.has(record.ollama_name)
    assert not record.gguf_path.exists()


def test_delete_static_model_returns_400(tmp_path: Path) -> None:
    client, _, _ = _build(tmp_path, events=[])
    response = client.post("/api/delete", json={"name": "mistral"})
    assert response.status_code == 400


def test_delete_unknown_model_returns_404(tmp_path: Path) -> None:
    client, _, _ = _build(tmp_path, events=[])
    response = client.post("/api/delete", json={"name": "nope"})
    assert response.status_code == 404
