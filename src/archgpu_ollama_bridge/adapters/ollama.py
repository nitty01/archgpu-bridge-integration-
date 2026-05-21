from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..catalogue import CatalogueEntry
from ..pulls import derive_model_id
from ..services import AppServices
from ..streaming import sse_to_ollama_chat, sse_to_ollama_generate


def build_ollama_router(services: AppServices) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["ollama"])

    def _model_payload(
        name: str,
        tags: list[str],
        *,
        pull_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": name,
            "model": name,
            "details": {
                "format": "gguf",
                "families": tags,
            },
        }
        if pull_state is not None:
            payload["pull_status"] = pull_state
        return payload

    def _catalogue_lookup() -> dict[str, CatalogueEntry]:
        if services.catalogue is None:
            return {}
        return {entry.alias: entry for entry in services.catalogue.list_entries()}

    def _resolve_runtime_identifier(identifier: str) -> str:
        # Directly registered model names resolve as-is.
        if services.registry.has(identifier):
            return identifier
        # If a catalogue alias was pulled under a derived id, map it.
        entry = _catalogue_lookup().get(identifier)
        if entry is None:
            return identifier
        try:
            ref = services.catalogue.resolve(identifier) if services.catalogue else None
        except Exception:
            return identifier
        if ref is None:
            return identifier
        derived = derive_model_id(ref)
        if services.registry.has(derived):
            return derived
        return identifier

    @router.get("/tags")
    async def list_tags() -> dict[str, Any]:
        status_map: dict[str, dict[str, Any]] = {}
        if services.pull_manager is not None:
            status_map = {
                item.model: item.to_dict() for item in services.pull_manager.list_status()
            }

        models: list[dict[str, Any]] = [
            _model_payload(
                model.ollama_name,
                model.tags,
                pull_state=status_map.get(model.ollama_name),
            )
            for model in services.registry.list_models()
        ]

        return {"models": models}

    @router.get("/version")
    async def version() -> dict[str, str]:
        # Open WebUI expects Ollama's /api/version to include a "version" key.
        return {"version": "0.1.0-archgpu-bridge"}

    @router.post("/show")
    async def show(request: Request) -> JSONResponse:
        payload = await request.json()
        identifier = payload.get("name") or payload.get("model")
        if not identifier:
            raise HTTPException(status_code=422, detail="Request body must include name or model")
        try:
            model = services.registry.get(identifier)
        except KeyError:
            entry = _catalogue_lookup().get(identifier)
            if entry is None:
                raise HTTPException(status_code=404, detail=f"Unknown model: {identifier}") from None
            return JSONResponse(
                content={
                    "model": entry.alias,
                    "details": {
                        "format": "gguf",
                        "families": entry.tags or ["catalogue"],
                        "parameter_size": "unknown",
                        "quantization_level": "unknown",
                    },
                    "model_info": {
                        "general.architecture": "llama",
                        "context_length": entry.context_length or 8192,
                        "download_required": True,
                        "hf_repo": entry.repo,
                        "hf_filename": entry.filename,
                    },
                    "parameters": "",
                    "template": "",
                }
            )
        return JSONResponse(
            content={
                "model": model.ollama_name,
                "details": {
                    "format": "gguf",
                    "families": model.tags,
                    "parameter_size": "unknown",
                    "quantization_level": "unknown",
                },
                "model_info": {
                    "general.architecture": "llama",
                    "context_length": model.context_length,
                },
                "parameters": "",
                "template": "",
            }
        )

    @router.get("/ps")
    async def list_running() -> dict[str, Any]:
        if services.lifecycle is None:
            return {"models": []}
        loaded_ids = services.lifecycle.loaded_model_ids()
        models: list[dict[str, Any]] = []
        for model_id in loaded_ids:
            try:
                model = services.registry.get(model_id)
            except KeyError:
                continue
            record = services.lifecycle.state_store.get(model_id)
            models.append(
                {
                    "name": model.ollama_name,
                    "model": model.ollama_name,
                    "details": {
                        "format": "gguf",
                        "families": model.tags,
                    },
                    "expires_at": None,
                    "size_vram": 0,
                    "last_used_at": record.last_used_at.isoformat() if record else None,
                }
            )
        return {"models": models}

    @router.post("/pull")
    async def pull(request: Request):
        if services.pull_manager is None:
            raise HTTPException(
                status_code=503,
                detail="pull manager is not configured",
            )
        payload = await request.json()
        name = payload.get("name") or payload.get("model")
        if not name:
            raise HTTPException(
                status_code=422,
                detail="Request body must include 'name' or 'model'",
            )
        return StreamingResponse(
            services.pull_manager.stream_pull(name),
            media_type="application/x-ndjson",
        )

    @router.get("/pull/status")
    async def pull_status(request: Request) -> dict[str, Any]:
        if services.pull_manager is None:
            return {"models": []}
        model = request.query_params.get("model")
        if model:
            status = services.pull_manager.get_status(model)
            return {"models": [status.to_dict()] if status else []}
        return {"models": [item.to_dict() for item in services.pull_manager.list_status()]}

    @router.get("/catalogue")
    async def catalogue() -> dict[str, Any]:
        entries = list(_catalogue_lookup().values())
        status_map: dict[str, dict[str, Any]] = {}
        if services.pull_manager is not None:
            status_map = {
                item.model: item.to_dict() for item in services.pull_manager.list_status()
            }

        models: list[dict[str, Any]] = []
        for entry in entries:
            # Support both new alias-based ids and older derived ids.
            installed_as: str | None = None
            if services.registry.has(entry.alias):
                installed_as = entry.alias
            elif services.catalogue is not None:
                try:
                    derived = derive_model_id(services.catalogue.resolve(entry.alias))
                    if services.registry.has(derived):
                        installed_as = derived
                except Exception:
                    installed_as = None

            pull_state = status_map.get(entry.alias)
            if pull_state is None and installed_as is not None:
                pull_state = status_map.get(installed_as)

            models.append(
                {
                    "alias": entry.alias,
                    "display_name": entry.display_name or entry.alias,
                    "repo": entry.repo,
                    "filename": entry.filename,
                    "context_length": entry.context_length,
                    "tags": entry.tags,
                    "installed": installed_as is not None,
                    "installed_as": installed_as,
                    "downloadable": installed_as is None,
                    "pull_status": pull_state,
                }
            )

        return {"models": models}

    @router.get("/catalogue/ui")
    async def catalogue_ui() -> HTMLResponse:
        return HTMLResponse(
            content="""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ARCHGPU Model Manager</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #0b1020; color: #e8ecf1; }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 16px; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .btn { border: 1px solid #30415f; background: #13203a; color: #e8ecf1; padding: 8px 12px; border-radius: 8px; cursor: pointer; }
    .btn.active { background: #1d3b70; border-color: #3e6fc0; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .muted { color: #9fb0c5; font-size: 12px; }
    .card { border: 1px solid #253552; border-radius: 10px; background: #111a2c; padding: 12px; margin-top: 10px; }
    .name { font-weight: 600; }
    .pill { font-size: 11px; border: 1px solid #3d5075; color: #9fb0c5; padding: 2px 6px; border-radius: 999px; }
    .progress { height: 8px; background: #1a2740; border-radius: 999px; overflow: hidden; }
    .bar { height: 100%; background: linear-gradient(90deg, #3c7dff, #57b6ff); width: 0%; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
    .search { width: 280px; max-width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid #30415f; background: #0d1629; color: #e8ecf1; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="row">
      <h2 style="margin:0; margin-right: 8px;">Model Manager</h2>
      <span class="muted">Installed vs Downloadable (with pull progress)</span>
    </div>
    <div class="row" style="margin-top:10px;">
      <button id="tabInstalled" class="btn active">Installed</button>
      <button id="tabDownloadable" class="btn">Downloadable</button>
      <input id="search" class="search" placeholder="Search models..." />
      <button id="refreshBtn" class="btn">Refresh</button>
      <span id="summary" class="muted"></span>
    </div>
    <div id="list"></div>
  </div>
<script>
let allModels = [];
let active = "installed";
const listEl = document.getElementById("list");
const summaryEl = document.getElementById("summary");
const searchEl = document.getElementById("search");

function fmtPct(m) {
  if (!m.pull_status || !m.pull_status.total || !m.pull_status.completed) return null;
  return Math.max(0, Math.min(100, Math.round((m.pull_status.completed / m.pull_status.total) * 100)));
}

function modelMatches(m, q) {
  if (!q) return true;
  const hay = [m.alias, m.display_name, ...(m.tags || [])].join(" ").toLowerCase();
  return hay.includes(q.toLowerCase());
}

function render() {
  const q = searchEl.value.trim();
  const items = allModels.filter(m => (active === "installed" ? m.installed : m.downloadable)).filter(m => modelMatches(m, q));
  summaryEl.textContent = `${items.length} shown / ${allModels.length} total`;
  listEl.innerHTML = items.map(m => {
    const st = m.pull_status?.status || (m.installed ? "installed" : "not_downloaded");
    const pct = fmtPct(m);
    const tags = (m.tags || []).map(t => `<span class="pill">${t}</span>`).join(" ");
    return `
      <div class="card">
        <div class="row" style="justify-content:space-between;">
          <div>
            <div class="name">${m.display_name}</div>
            <div class="muted">${m.alias} • ${m.repo}:${m.filename}</div>
          </div>
          <div class="row">
            <span class="pill">${st}</span>
            ${m.downloadable ? `<button class="btn" onclick="startPull('${m.alias}')">Download</button>` : ""}
          </div>
        </div>
        <div class="row" style="margin-top:8px;">${tags}</div>
        ${pct !== null ? `<div style="margin-top:8px;" class="progress"><div class="bar" style="width:${pct}%"></div></div><div class="muted">${pct}%</div>` : ""}
      </div>`;
  }).join("");
}

async function loadModels() {
  const r = await fetch("/api/catalogue");
  const d = await r.json();
  allModels = d.models || [];
  render();
}

async function startPull(alias) {
  const res = await fetch("/api/pull", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({name: alias})
  });
  if (!res.body) return;
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    const lines = buf.split("\\n");
    buf = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const evt = JSON.parse(line);
        const m = allModels.find(x => x.alias === alias);
        if (m) m.pull_status = evt;
      } catch {}
    }
    render();
  }
  await loadModels();
}

document.getElementById("tabInstalled").onclick = () => {
  active = "installed";
  document.getElementById("tabInstalled").classList.add("active");
  document.getElementById("tabDownloadable").classList.remove("active");
  render();
};
document.getElementById("tabDownloadable").onclick = () => {
  active = "downloadable";
  document.getElementById("tabDownloadable").classList.add("active");
  document.getElementById("tabInstalled").classList.remove("active");
  render();
};
document.getElementById("refreshBtn").onclick = loadModels;
searchEl.oninput = render;
setInterval(loadModels, 5000);
loadModels();
</script>
</body>
</html>"""
        )

    @router.post("/delete")
    async def delete(request: Request) -> JSONResponse:
        payload = await request.json()
        name = payload.get("name") or payload.get("model")
        if not name:
            raise HTTPException(
                status_code=422,
                detail="Request body must include 'name' or 'model'",
            )
        keep_file = bool(payload.get("keep_file", False))
        try:
            record = services.registry.get(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if not services.registry.is_dynamic(name):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Refusing to delete static model {record.id!r}; "
                    "edit config/models.yaml instead."
                ),
            )

        if services.lifecycle is not None:
            try:
                await services.lifecycle.stop(record.id)
            except Exception:
                # Best-effort: continue with delete even if stop failed
                pass

        services.registry.unregister(record.id)

        if not keep_file:
            try:
                gguf = record.gguf_path
                if gguf.is_file():
                    gguf.unlink()
            except OSError:
                # File removal is best-effort; the registry change has succeeded
                pass

        return JSONResponse(
            content={
                "status": "success",
                "model": record.ollama_name,
                "removed_file": (not keep_file),
            }
        )

    @router.post("/chat")
    async def chat(request: Request):
        payload = await request.json()
        model_name = payload.get("model")
        messages = payload.get("messages")
        if not model_name or not messages:
            raise HTTPException(status_code=422, detail="Request body must include model and messages")
        route_identifier = _resolve_runtime_identifier(model_name)

        stream_requested = bool(payload.get("stream", True))

        try:
            target = await services.router.route(route_identifier)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"{exc}. Pull the model first via POST /api/pull.",
            ) from exc

        if stream_requested:
            upstream = services.proxy_client.stream_post(
                base_url=target.base_url,
                path="/v1/chat/completions",
                payload={
                    "model": target.model.openai_name,
                    "messages": messages,
                    "stream": True,
                },
            )
            return StreamingResponse(
                sse_to_ollama_chat(upstream, target.model.ollama_name),
                media_type="application/x-ndjson",
            )

        try:
            upstream = await services.proxy_client.post_json(
                base_url=target.base_url,
                path="/v1/chat/completions",
                payload={
                    "model": target.model.openai_name,
                    "messages": messages,
                    "stream": False,
                },
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        message = upstream.json_body["choices"][0]["message"]
        finish_reason = upstream.json_body["choices"][0].get("finish_reason", "stop")
        return JSONResponse(
            status_code=upstream.status_code,
            content={
                "model": target.model.ollama_name,
                "created_at": datetime.now(UTC).isoformat(),
                "message": message,
                "done": True,
                "done_reason": finish_reason,
            },
        )

    @router.post("/generate")
    async def generate(request: Request):
        payload = await request.json()
        model_name = payload.get("model")
        prompt = payload.get("prompt")
        if not model_name or prompt is None:
            raise HTTPException(status_code=422, detail="Request body must include model and prompt")
        route_identifier = _resolve_runtime_identifier(model_name)

        stream_requested = bool(payload.get("stream", True))

        try:
            target = await services.router.route(route_identifier)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"{exc}. Pull the model first via POST /api/pull.",
            ) from exc

        if stream_requested:
            upstream = services.proxy_client.stream_post(
                base_url=target.base_url,
                path="/v1/chat/completions",
                payload={
                    "model": target.model.openai_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                },
            )
            return StreamingResponse(
                sse_to_ollama_generate(upstream, target.model.ollama_name),
                media_type="application/x-ndjson",
            )

        try:
            upstream = await services.proxy_client.post_json(
                base_url=target.base_url,
                path="/v1/chat/completions",
                payload={
                    "model": target.model.openai_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        content = upstream.json_body["choices"][0]["message"]["content"]
        finish_reason = upstream.json_body["choices"][0].get("finish_reason", "stop")
        return JSONResponse(
            status_code=upstream.status_code,
            content={
                "model": target.model.ollama_name,
                "created_at": datetime.now(UTC).isoformat(),
                "response": content,
                "done": True,
                "done_reason": finish_reason,
            },
        )

    return router
