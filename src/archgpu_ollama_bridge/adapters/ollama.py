from datetime import UTC, datetime
import os
import re
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..catalogue import CatalogueEntry
from ..pulls import derive_model_id
from ..services import AppServices
from ..streaming import sse_to_ollama_chat, sse_to_ollama_generate


def build_ollama_router(services: AppServices) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["ollama"])
    live_catalogue_cache: dict[str, Any] = {"fetched_at": 0.0, "models": []}

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

    def _pick_preferred_gguf_filename(files: list[str]) -> str | None:
        if not files:
            return None
        preferences = (
            "Q4_K_M",
            "q4_k_m",
            "Q5_K_M",
            "q5_k_m",
            "Q6_K",
            "q6_k",
            "Q8_0",
            "q8_0",
        )
        for pref in preferences:
            for file_name in files:
                if pref in file_name:
                    return file_name
        return files[0]

    def _derive_live_capabilities(model_id: str, tags: list[str], context_length: int | None) -> list[str]:
        haystack = f"{model_id} {' '.join(tags)}".lower()
        caps: list[str] = []
        if "coder" in haystack or "code" in haystack:
            caps.append("coding")
        if "reason" in haystack or "deepseek-r1" in haystack:
            caps.append("reasoning")
        if context_length is not None and context_length >= 16384:
            caps.append("long-context")
        if "vision" in haystack or "vl" in haystack:
            caps.append("vision")
        if not caps:
            caps.append("chat")
        return caps

    def _trusted_publishers() -> set[str]:
        # Used by quality filters in Model Explorer.
        return {
            "lmstudio-community",
            "bartowski",
            "unsloth",
            "qwen",
            "microsoft",
            "mistralai",
            "meta-llama",
            "google",
        }

    def _extract_model_size_b(model_name: str) -> float | None:
        # Best-effort parse: "...14B..." -> 14.0
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_name)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _extract_quant_bits(filename: str) -> float:
        name = filename.lower()
        if "q2_" in name:
            return 2.0
        if "q3_" in name:
            return 3.0
        if "q4_" in name or "iq4" in name:
            return 4.0
        if "q5_" in name:
            return 5.0
        if "q6_" in name:
            return 6.0
        if "q8_" in name:
            return 8.0
        return 4.0

    def _detect_system_profile() -> dict[str, Any]:
        # Linux-centric best-effort system profile.
        total_ram_gb = None
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            total_ram_gb = round((page_size * phys_pages) / (1024 ** 3), 1)
        except Exception:
            total_ram_gb = None

        vram_candidates_gb: list[float] = []
        drm_root = "/sys/class/drm"
        try:
            if os.path.isdir(drm_root):
                for entry in os.listdir(drm_root):
                    if not entry.startswith("card"):
                        continue
                    vram_file = os.path.join(drm_root, entry, "device", "mem_info_vram_total")
                    if os.path.isfile(vram_file):
                        try:
                            raw = open(vram_file, "r", encoding="utf-8").read().strip()
                            vram_bytes = int(raw)
                            vram_candidates_gb.append(round(vram_bytes / (1024 ** 3), 1))
                        except Exception:
                            continue
        except Exception:
            pass

        detected_vram_gb = max(vram_candidates_gb) if vram_candidates_gb else None
        hint = None
        if services.settings is not None:
            hint = os.getenv("ARCHGPU_BRIDGE_GPU_VRAM_GB_HINT")
        if hint:
            try:
                detected_vram_gb = float(hint)
            except ValueError:
                pass

        return {
            "ram_gb": total_ram_gb,
            "gpu_vram_gb": detected_vram_gb,
        }

    def _runtime_fit_for_model(
        *,
        model_name: str,
        filename: str,
        context_length: int | None,
        system_profile: dict[str, Any],
    ) -> dict[str, Any]:
        params_b = _extract_model_size_b(model_name) or _extract_model_size_b(filename)
        quant_bits = _extract_quant_bits(filename)
        estimated_model_gb = None
        if params_b is not None:
            # Approx: params(B) * bits/8 + 25% overhead.
            estimated_model_gb = round(params_b * (quant_bits / 8.0) * 1.25, 1)

        gpu_vram_gb = system_profile.get("gpu_vram_gb")
        ram_gb = system_profile.get("ram_gb")
        recommendation = "unknown"
        fits_gpu = None
        fits_ram = None
        reason = "insufficient metadata"

        if estimated_model_gb is not None:
            if gpu_vram_gb is not None:
                fits_gpu = estimated_model_gb <= gpu_vram_gb * 0.9
            if ram_gb is not None:
                fits_ram = estimated_model_gb <= ram_gb * 0.6

            if fits_gpu is True:
                recommendation = "recommended"
                reason = "estimated to fit GPU VRAM"
            elif fits_gpu is False and fits_ram is True:
                recommendation = "possible"
                reason = "likely RAM/offload fallback, slower performance"
            elif fits_gpu is False and fits_ram is False:
                recommendation = "not_recommended"
                reason = "estimated memory footprint too large"
            else:
                recommendation = "possible"
                reason = "partial system information available"

        if context_length is not None and context_length > 65536 and recommendation == "possible":
            reason = f"{reason}; high context may reduce throughput"

        return {
            "estimated_model_gb": estimated_model_gb,
            "estimated_param_b": params_b,
            "quant_bits": quant_bits,
            "fits_gpu": fits_gpu,
            "fits_ram": fits_ram,
            "recommendation": recommendation,
            "reason": reason,
        }

    async def _discover_live_catalogue(
        *,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        settings = services.settings
        if settings is None or not settings.hf_discovery_enabled:
            return []

        now = time.time()
        ttl = max(0, settings.hf_discovery_ttl_seconds)
        cached_models = live_catalogue_cache.get("models", [])
        cached_at = float(live_catalogue_cache.get("fetched_at", 0.0))
        if (
            not force_refresh
            and cached_models
            and ttl > 0
            and (now - cached_at) < ttl
        ):
            return list(cached_models)

        base_url = settings.hf_base_url.rstrip("/")
        owners = {owner.lower() for owner in settings.hf_discovery_owners}
        per_query_limit = max(1, settings.hf_discovery_per_query_limit)
        max_models = max(1, settings.hf_discovery_max_models)

        discovered_ids: list[str] = []
        seen_ids: set[str] = set()
        timeout = httpx.Timeout(20.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            for query in settings.hf_discovery_queries:
                response = await client.get(
                    f"{base_url}/api/models",
                    params={
                        "search": query,
                        "limit": per_query_limit,
                        "sort": "lastModified",
                        "direction": -1,
                    },
                )
                response.raise_for_status()
                items = response.json()
                for item in items:
                    model_id = item.get("id")
                    if not model_id or model_id in seen_ids:
                        continue
                    owner = model_id.split("/", 1)[0].lower() if "/" in model_id else ""
                    if owners and owner not in owners:
                        continue
                    seen_ids.add(model_id)
                    discovered_ids.append(model_id)
                    if len(discovered_ids) >= max_models:
                        break
                if len(discovered_ids) >= max_models:
                    break

            results: list[dict[str, Any]] = []
            for model_id in discovered_ids:
                details_response = await client.get(f"{base_url}/api/models/{model_id}")
                if details_response.status_code != 200:
                    continue
                details = details_response.json()
                siblings = details.get("siblings", [])
                gguf_files = [
                    f.get("rfilename", "")
                    for f in siblings
                    if f.get("rfilename", "").lower().endswith(".gguf")
                ]
                filename = _pick_preferred_gguf_filename([f for f in gguf_files if f])
                if filename is None:
                    continue

                hf_tags = details.get("tags", [])
                context_length = None
                for tag in hf_tags:
                    match = re.match(r"ctx:(\d+)", str(tag), re.IGNORECASE)
                    if match:
                        context_length = int(match.group(1))
                        break

                pull_name = f"{model_id}:{filename}"
                installed_as: str | None = None
                if services.registry.has(pull_name):
                    installed_as = pull_name
                elif services.catalogue is not None:
                    try:
                        derived = derive_model_id(services.catalogue.resolve(pull_name))
                        if services.registry.has(derived):
                            installed_as = derived
                    except Exception:
                        installed_as = None

                caps = _derive_live_capabilities(model_id, [str(t) for t in hf_tags], context_length)
                results.append(
                    {
                        "alias": pull_name,
                        "pull_name": pull_name,
                        "display_name": model_id.split("/", 1)[-1],
                        "repo": model_id,
                        "filename": filename,
                        "context_length": context_length,
                        "tags": ["live", "auto"] + [str(t) for t in hf_tags[:8]],
                        "capabilities": caps,
                        "installed": installed_as is not None,
                        "installed_as": installed_as,
                        "deletable": bool(installed_as and services.registry.is_dynamic(installed_as)),
                        "downloadable": installed_as is None,
                        "pull_status": None,
                        "metadata": {
                            "source": "live_discovery",
                            "publisher": model_id.split("/", 1)[0] if "/" in model_id else None,
                            "last_modified": details.get("lastModified"),
                            "knowledge_last_update": details.get("lastModified"),
                            "downloads": details.get("downloads"),
                            "likes": details.get("likes"),
                            "pipeline_tag": details.get("pipeline_tag"),
                            "gated": bool(details.get("gated")),
                            "private": bool(details.get("private")),
                            "library_name": details.get("library_name"),
                        },
                    }
                )

        live_catalogue_cache["fetched_at"] = now
        live_catalogue_cache["models"] = results
        return list(results)

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
    async def catalogue(request: Request) -> dict[str, Any]:
        def _to_int(value: str | None, default: int = 0) -> int:
            try:
                return int(value) if value is not None else default
            except (TypeError, ValueError):
                return default

        query = (request.query_params.get("q") or "").strip().lower()
        include_live = request.query_params.get("live", "true").lower() != "false"
        force_refresh = request.query_params.get("refresh", "false").lower() == "true"
        trusted_only = request.query_params.get("trusted_only", "false").lower() == "true"
        min_downloads = _to_int(request.query_params.get("min_downloads"), 0)
        min_likes = _to_int(request.query_params.get("min_likes"), 0)
        quality_mode = request.query_params.get("quality", "off").strip().lower()
        system_profile = _detect_system_profile()

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

            deletable = False
            if installed_as is not None:
                try:
                    deletable = services.registry.is_dynamic(installed_as)
                except Exception:
                    deletable = False

            models.append(
                {
                    "alias": entry.alias,
                    "pull_name": entry.alias,
                    "display_name": entry.display_name or entry.alias,
                    "repo": entry.repo,
                    "filename": entry.filename,
                    "context_length": entry.context_length,
                    "tags": entry.tags,
                    "capabilities": _derive_live_capabilities(entry.alias, list(entry.tags), entry.context_length),
                    "installed": installed_as is not None,
                    "installed_as": installed_as,
                    "deletable": deletable,
                    "downloadable": installed_as is None,
                    "pull_status": pull_state,
                    "metadata": {
                        "source": "static_catalogue",
                        "publisher": entry.repo.split("/", 1)[0] if "/" in entry.repo else None,
                        "knowledge_last_update": None,
                        "last_modified": None,
                        "downloads": None,
                        "likes": None,
                        "pipeline_tag": None,
                    },
                }
            )

        if include_live:
            try:
                live_models = await _discover_live_catalogue(force_refresh=force_refresh)
            except Exception:
                live_models = []
            for lm in live_models:
                alias = lm.get("alias")
                if any(m.get("alias") == alias for m in models):
                    continue
                pull_state = status_map.get(alias)
                if pull_state is None and lm.get("installed_as"):
                    pull_state = status_map.get(lm["installed_as"])
                lm["pull_status"] = pull_state or lm.get("pull_status")
                models.append(lm)

        # Always include installed registry models even if they are not part of
        # curated catalogue aliases or live-discovered candidates.
        known_installed = {
            str(item.get("installed_as"))
            for item in models
            if item.get("installed_as") is not None
        }
        known_aliases = {str(item.get("alias")) for item in models}
        for reg_model in services.registry.list_models():
            if reg_model.id in known_installed or reg_model.ollama_name in known_aliases:
                continue
            pull_state = status_map.get(reg_model.ollama_name) or status_map.get(reg_model.id)
            models.append(
                {
                    "alias": reg_model.ollama_name,
                    "pull_name": reg_model.ollama_name,
                    "display_name": reg_model.openai_name or reg_model.ollama_name,
                    "repo": None,
                    "filename": reg_model.gguf_path.name,
                    "context_length": reg_model.context_length,
                    "tags": reg_model.tags,
                    "capabilities": _derive_live_capabilities(
                        reg_model.ollama_name, list(reg_model.tags), reg_model.context_length
                    ),
                    "installed": True,
                    "installed_as": reg_model.id,
                    "deletable": services.registry.is_dynamic(reg_model.id),
                    "downloadable": False,
                    "pull_status": pull_state,
                    "metadata": {
                        "source": "local_registry",
                        "publisher": None,
                        "knowledge_last_update": None,
                        "last_modified": None,
                        "downloads": None,
                        "likes": None,
                        "pipeline_tag": None,
                    },
                }
            )

        for model in models:
            model["runtime_fit"] = _runtime_fit_for_model(
                model_name=str(model.get("display_name") or model.get("alias") or ""),
                filename=str(model.get("filename") or ""),
                context_length=model.get("context_length"),
                system_profile=system_profile,
            )

        trusted_publishers = _trusted_publishers()
        if trusted_only:
            models = [
                model
                for model in models
                if str((model.get("metadata") or {}).get("publisher", "")).lower()
                in trusted_publishers
            ]

        if min_downloads > 0:
            models = [
                model
                for model in models
                if int((model.get("metadata") or {}).get("downloads") or 0) >= min_downloads
            ]

        if min_likes > 0:
            models = [
                model
                for model in models
                if int((model.get("metadata") or {}).get("likes") or 0) >= min_likes
            ]

        if quality_mode == "strict":
            models = [
                model
                for model in models
                if (
                    str((model.get("metadata") or {}).get("publisher", "")).lower() in trusted_publishers
                    and int((model.get("metadata") or {}).get("downloads") or 0) >= 1000
                    and int((model.get("metadata") or {}).get("likes") or 0) >= 10
                )
            ]

        if query:
            models = [
                model
                for model in models
                if query
                in " ".join(
                    [
                        str(model.get("alias", "")),
                        str(model.get("display_name", "")),
                        str(model.get("repo", "")),
                        str(model.get("filename", "")),
                        " ".join(str(t) for t in model.get("tags", [])),
                        " ".join(str(c) for c in model.get("capabilities", [])),
                        str((model.get("metadata") or {}).get("publisher", "")),
                        str((model.get("runtime_fit") or {}).get("recommendation", "")),
                        str((model.get("runtime_fit") or {}).get("reason", "")),
                    ]
                ).lower()
            ]

        return {"models": models, "system_profile": system_profile}

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
    @router.delete("/delete")
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
