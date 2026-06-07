"""Hugging Face GGUF model index with live API fetch and catalogue helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from .pulls import derive_model_id

logger = logging.getLogger(__name__)

_GGUF_PREFERENCES = (
    "Q4_K_M",
    "q4_k_m",
    "Q5_K_M",
    "q5_k_m",
    "Q6_K",
    "q6_k",
    "Q8_0",
    "q8_0",
)


class HfIndexError(Exception):
    """Raised when the Hugging Face index cannot be refreshed."""


def pick_preferred_gguf_filename(files: list[str]) -> str | None:
    if not files:
        return None
    for pref in _GGUF_PREFERENCES:
        for file_name in files:
            if pref in file_name:
                return file_name
    return files[0]


def hf_request_headers(settings: Any) -> dict[str, str]:
    token = getattr(settings, "hf_token", None)
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _parse_next_cursor(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.strip()
        if 'rel="next"' not in section:
            continue
        match = re.search(r"<([^>]+)>", section)
        if not match:
            continue
        parsed = urlparse(match.group(1))
        cursor_vals = parse_qs(parsed.query).get("cursor")
        if cursor_vals:
            return cursor_vals[0]
    return None


def _context_from_tags(tags: list) -> int | None:
    for tag in tags:
        match = re.match(r"ctx:(\d+)", str(tag), re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def build_live_model_entry(
    *,
    model_id: str,
    details: dict[str, Any],
    filename: str,
    derive_capabilities,
    registry,
    catalogue,
) -> dict[str, Any]:
    hf_tags = details.get("tags", [])
    context_length = _context_from_tags(hf_tags)
    pull_name = f"{model_id}:{filename}"
    installed_as: str | None = None
    if registry.has(pull_name):
        installed_as = pull_name
    elif catalogue is not None:
        try:
            derived = derive_model_id(catalogue.resolve(pull_name))
            if registry.has(derived):
                installed_as = derived
        except Exception:
            installed_as = None

    caps = derive_capabilities(model_id, [str(t) for t in hf_tags], context_length)
    return {
        "alias": pull_name,
        "pull_name": pull_name,
        "display_name": model_id.split("/", 1)[-1],
        "repo": model_id,
        "filename": filename,
        "context_length": context_length,
        "tags": ["live", "auto", "gguf"] + [str(t) for t in hf_tags[:8]],
        "capabilities": caps,
        "installed": installed_as is not None,
        "installed_as": installed_as,
        "deletable": bool(installed_as and registry.is_dynamic(installed_as)),
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


def _model_entry_from_list_item(
    item: dict[str, Any],
    *,
    derive_capabilities,
    registry,
    catalogue,
) -> dict[str, Any] | None:
    model_id = item.get("id")
    if not model_id:
        return None
    tags = [str(t) for t in item.get("tags", [])]
    haystack = f"{model_id} {' '.join(tags)}".lower()
    if "gguf" not in haystack and not model_id.lower().endswith("-gguf"):
        return None

    filename = pick_preferred_gguf_filename(
        [tag for tag in tags if tag.lower().endswith(".gguf")]
    )
    if filename is None:
        filename = f"{model_id.split('/')[-1]}.gguf"

    context_length = _context_from_tags(tags)
    pull_name = f"{model_id}:{filename}"
    installed_as: str | None = None
    if registry.has(pull_name):
        installed_as = pull_name
    elif catalogue is not None:
        try:
            derived = derive_model_id(catalogue.resolve(pull_name))
            if registry.has(derived):
                installed_as = derived
        except Exception:
            installed_as = None

    caps = derive_capabilities(model_id, tags, context_length)
    return {
        "alias": pull_name,
        "pull_name": pull_name,
        "display_name": model_id.split("/", 1)[-1],
        "repo": model_id,
        "filename": filename,
        "context_length": context_length,
        "tags": ["live", "auto", "gguf"] + tags[:8],
        "capabilities": caps,
        "installed": installed_as is not None,
        "installed_as": installed_as,
        "deletable": bool(installed_as and registry.is_dynamic(installed_as)),
        "downloadable": installed_as is None,
        "pull_status": None,
        "metadata": {
            "source": "live_discovery",
            "publisher": model_id.split("/", 1)[0] if "/" in model_id else None,
            "last_modified": item.get("lastModified"),
            "knowledge_last_update": item.get("lastModified"),
            "downloads": item.get("downloads"),
            "likes": item.get("likes"),
            "pipeline_tag": item.get("pipeline_tag"),
            "gated": bool(item.get("gated")),
            "private": bool(item.get("private")),
            "library_name": item.get("library_name"),
        },
    }


async def _fetch_model_details(
    client: httpx.AsyncClient,
    base_url: str,
    model_id: str,
    headers: dict[str, str],
) -> dict[str, Any] | None:
    response = await client.get(f"{base_url}/api/models/{model_id}", headers=headers)
    if response.status_code != 200:
        return None
    return response.json()


async def _build_entries_from_ids(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    headers: dict[str, str],
    model_ids: list[str],
    derive_capabilities,
    registry,
    catalogue,
    detail_concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, detail_concurrency))
    results: list[dict[str, Any]] = []

    async def _one(model_id: str) -> dict[str, Any] | None:
        async with semaphore:
            details = await _fetch_model_details(client, base_url, model_id, headers)
        if details is None:
            return None
        siblings = details.get("siblings", [])
        gguf_files = [
            f.get("rfilename", "")
            for f in siblings
            if f.get("rfilename", "").lower().endswith(".gguf")
        ]
        filename = pick_preferred_gguf_filename([f for f in gguf_files if f])
        if filename is None:
            return None
        return build_live_model_entry(
            model_id=model_id,
            details=details,
            filename=filename,
            derive_capabilities=derive_capabilities,
            registry=registry,
            catalogue=catalogue,
        )

    built = await asyncio.gather(*[_one(model_id) for model_id in model_ids])
    for entry in built:
        if entry is not None:
            results.append(entry)
    return results


async def _discover_model_ids(
    client: httpx.AsyncClient,
    *,
    settings: Any,
    headers: dict[str, str],
    force_refresh: bool,
) -> list[str]:
    base_url = settings.hf_base_url.rstrip("/")
    owners = {owner.lower() for owner in settings.hf_discovery_owners}
    page_size = max(1, min(settings.hf_index_page_size, 1000))
    max_models = max(1, settings.hf_index_max_models)
    max_pages = max(1, settings.hf_index_max_pages)

    discovered_ids: list[str] = []
    seen_ids: set[str] = set()

    async def _consume_items(items: list[dict[str, Any]]) -> bool:
        for item in items:
            model_id = item.get("id")
            if not model_id or model_id in seen_ids:
                continue
            if item.get("private"):
                continue
            owner = model_id.split("/", 1)[0].lower() if "/" in model_id else ""
            if owners and owner not in owners:
                continue
            seen_ids.add(model_id)
            discovered_ids.append(model_id)
            if len(discovered_ids) >= max_models:
                return True
        return False

    cursor: str | None = None
    pages_fetched = 0
    while pages_fetched < max_pages and len(discovered_ids) < max_models:
        params: dict[str, Any] = {
            "search": "GGUF",
            "limit": page_size,
            "sort": "downloads",
            "direction": -1,
        }
        if cursor:
            params["cursor"] = cursor

        response = await client.get(
            f"{base_url}/api/models",
            params=params,
            headers=headers,
        )
        if response.status_code == 429:
            raise HfIndexError(
                "Hugging Face rate limit reached. Set ARCHGPU_BRIDGE_HF_TOKEN to a valid HF access token."
            )
        response.raise_for_status()
        items = response.json()
        if not isinstance(items, list) or not items:
            break
        if await _consume_items(items):
            break

        pages_fetched += 1
        cursor = _parse_next_cursor(response.headers.get("link"))
        if not cursor:
            break

    per_query_limit = max(1, min(settings.hf_discovery_per_query_limit, 100))
    for query in settings.hf_discovery_queries:
        if len(discovered_ids) >= max_models:
            break
        response = await client.get(
            f"{base_url}/api/models",
            params={
                "search": query,
                "limit": per_query_limit,
                "sort": "lastModified",
                "direction": -1,
            },
            headers=headers,
        )
        if response.status_code == 429:
            raise HfIndexError(
                "Hugging Face rate limit reached. Set ARCHGPU_BRIDGE_HF_TOKEN to a valid HF access token."
            )
        if response.status_code != 200:
            continue
        items = response.json()
        if not isinstance(items, list):
            continue
        if await _consume_items(items):
            break

    return discovered_ids


async def fetch_hf_gguf_index(
    *,
    settings: Any,
    derive_capabilities,
    registry,
    catalogue,
    force_refresh: bool = False,
    cache: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch public GGUF models from Hugging Face into an in-memory session index."""
    meta: dict[str, Any] = {
        "source": "memory",
        "stale": False,
        "error": None,
        "count": 0,
        "fetched_at": None,
    }
    if not settings.hf_discovery_enabled:
        meta["error"] = "hf_discovery_disabled"
        return [], meta

    now = time.time()
    ttl = max(0, settings.hf_index_ttl_seconds)
    cache = cache if cache is not None else {}
    cached_models = cache.get("models", [])
    cached_at = float(cache.get("fetched_at", 0.0))

    if (
        not force_refresh
        and cached_models
        and ttl > 0
        and (now - cached_at) < ttl
    ):
        meta["count"] = len(cached_models)
        meta["fetched_at"] = cached_at
        return list(cached_models), meta

    disk_cache = _load_disk_cache(settings)
    if (
        not force_refresh
        and disk_cache.get("models")
        and ttl > 0
        and (now - float(disk_cache.get("fetched_at", 0.0))) < ttl
    ):
        cache["fetched_at"] = disk_cache["fetched_at"]
        cache["models"] = disk_cache["models"]
        meta["source"] = "disk"
        meta["count"] = len(disk_cache["models"])
        meta["fetched_at"] = disk_cache["fetched_at"]
        return list(disk_cache["models"]), meta

    headers = hf_request_headers(settings)
    timeout = httpx.Timeout(settings.hf_index_timeout_seconds)
    detail_concurrency = max(1, getattr(settings, "hf_index_detail_concurrency", 16))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            model_ids = await _discover_model_ids(
                client,
                settings=settings,
                headers=headers,
                force_refresh=force_refresh,
            )
            results = await _build_entries_from_ids(
                client,
                base_url=settings.hf_base_url.rstrip("/"),
                headers=headers,
                model_ids=model_ids,
                derive_capabilities=derive_capabilities,
                registry=registry,
                catalogue=catalogue,
                detail_concurrency=detail_concurrency,
            )
    except HfIndexError as exc:
        logger.warning("HF index refresh failed: %s", exc)
        if cached_models:
            meta["stale"] = True
            meta["error"] = str(exc)
            meta["count"] = len(cached_models)
            meta["fetched_at"] = cached_at or None
            return list(cached_models), meta
        if disk_cache.get("models"):
            meta["source"] = "disk"
            meta["stale"] = True
            meta["error"] = str(exc)
            meta["count"] = len(disk_cache["models"])
            meta["fetched_at"] = disk_cache.get("fetched_at")
            cache["fetched_at"] = disk_cache["fetched_at"]
            cache["models"] = disk_cache["models"]
            return list(disk_cache["models"]), meta
        meta["error"] = str(exc)
        return [], meta
    except httpx.HTTPError as exc:
        logger.warning("HF index HTTP error: %s", exc)
        if cached_models:
            meta["stale"] = True
            meta["error"] = str(exc)
            meta["count"] = len(cached_models)
            meta["fetched_at"] = cached_at or None
            return list(cached_models), meta
        meta["error"] = str(exc)
        return [], meta

    cache["fetched_at"] = now
    cache["models"] = results
    _save_disk_cache(settings, now, results)
    meta["count"] = len(results)
    meta["fetched_at"] = now
    return list(results), meta


def _disk_cache_path(settings: Any) -> Path | None:
    path = getattr(settings, "hf_index_cache_path", None)
    if path is None:
        return None
    return Path(path)


def _legacy_disk_cache_path() -> Path:
    return Path("data/hf_gguf_index.json")


def _load_disk_cache(settings: Any) -> dict[str, Any]:
    path = _disk_cache_path(settings)
    if path is None:
        legacy = _legacy_disk_cache_path()
        path = legacy if legacy.exists() else None
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def resolve_hf_search_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "GGUF"
    lower = q.lower()
    for prefix in ("capability:", "tag:", "fit:", "source:"):
        if lower.startswith(prefix):
            return "GGUF"
    for prefix in ("repo:", "publisher:"):
        if lower.startswith(prefix):
            return q[len(prefix) :].strip() or "GGUF"
    return q


def hf_sort_params(sort_by: str, sort_dir: str) -> dict[str, Any]:
    direction = -1 if sort_dir.lower() != "asc" else 1
    if sort_by == "last_modified":
        return {"sort": "lastModified", "direction": direction}
    if sort_by == "likes":
        return {"sort": "likes", "direction": direction}
    if sort_by in {"display_name", "alias", "publisher"}:
        return {"sort": "id", "direction": direction}
    return {"sort": "downloads", "direction": direction}


def _live_session_key(
    *,
    search: str,
    sort_by: str,
    sort_dir: str,
    owners: tuple[str, ...],
) -> str:
    owner_key = ",".join(sorted(owners))
    return f"{search}|{sort_by}|{sort_dir}|{owner_key}"


def _new_live_session(settings: Any, search: str) -> dict[str, Any]:
    phases = [search]
    phases.extend(settings.hf_discovery_queries)
    deduped_phases: list[str] = []
    seen: set[str] = set()
    for phase in phases:
        key = phase.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped_phases.append(key)
    return {
        "phases": deduped_phases,
        "phase_index": 0,
        "list_cursor": None,
        "list_item_index": 0,
        "gguf_count": 0,
        "seen_ids": set(),
        "exhausted": False,
        "checkpoints": {0: {"phase_index": 0, "list_cursor": None, "list_item_index": 0, "gguf_count": 0}},
    }


def _restore_checkpoint(session: dict[str, Any], offset: int) -> None:
    checkpoints: dict[int, dict[str, Any]] = session.get("checkpoints", {})
    start_key = 0
    for key in sorted(int(k) for k in checkpoints):
        if key <= offset:
            start_key = key
        else:
            break
    point = checkpoints.get(start_key) or checkpoints.get(0) or {}
    session["phase_index"] = int(point.get("phase_index", 0))
    session["list_cursor"] = point.get("list_cursor")
    session["list_item_index"] = int(point.get("list_item_index", 0))
    session["gguf_count"] = int(point.get("gguf_count", 0))
    session["exhausted"] = False


def _save_checkpoint(session: dict[str, Any], gguf_count: int) -> None:
    checkpoints: dict[int, dict[str, Any]] = session.setdefault("checkpoints", {})
    checkpoints[int(gguf_count)] = {
        "phase_index": int(session.get("phase_index", 0)),
        "list_cursor": session.get("list_cursor"),
        "list_item_index": int(session.get("list_item_index", 0)),
        "gguf_count": int(session.get("gguf_count", 0)),
    }


async def _try_build_entry_from_id(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    headers: dict[str, str],
    model_id: str,
    derive_capabilities,
    registry,
    catalogue,
) -> dict[str, Any] | None:
    details = await _fetch_model_details(client, base_url, model_id, headers)
    if details is None:
        return None
    siblings = details.get("siblings", [])
    gguf_files = [
        f.get("rfilename", "")
        for f in siblings
        if f.get("rfilename", "").lower().endswith(".gguf")
    ]
    filename = pick_preferred_gguf_filename([f for f in gguf_files if f])
    if filename is None:
        return None
    return build_live_model_entry(
        model_id=model_id,
        details=details,
        filename=filename,
        derive_capabilities=derive_capabilities,
        registry=registry,
        catalogue=catalogue,
    )


async def fetch_hf_live_slice(
    *,
    settings: Any,
    offset: int,
    limit: int,
    search: str,
    sort_by: str,
    sort_dir: str,
    derive_capabilities,
    registry,
    catalogue,
    sessions: dict[str, dict[str, Any]],
    force_refresh: bool = False,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    """Fetch a slice of GGUF models directly from Hugging Face (no full local index)."""
    meta: dict[str, Any] = {
        "mode": "live_page",
        "source": "huggingface_api",
        "stale": False,
        "error": None,
        "count": None,
        "fetched_at": time.time(),
        "hf_search": search,
        "hf_offset": offset,
        "hf_limit": limit,
    }
    if not settings.hf_discovery_enabled:
        meta["error"] = "hf_discovery_disabled"
        return [], False, meta

    owners = tuple(owner.lower() for owner in settings.hf_discovery_owners)
    session_key = _live_session_key(
        search=search,
        sort_by=sort_by,
        sort_dir=sort_dir,
        owners=owners,
    )
    if force_refresh or session_key not in sessions:
        sessions[session_key] = _new_live_session(settings, search)

    session = sessions[session_key]
    if force_refresh:
        sessions[session_key] = _new_live_session(settings, search)
        session = sessions[session_key]

    _restore_checkpoint(session, max(0, offset))

    headers = hf_request_headers(settings)
    timeout = httpx.Timeout(settings.hf_index_timeout_seconds)
    list_page_size = max(10, min(settings.hf_live_list_page_size, 1000))
    base_url = settings.hf_base_url.rstrip("/")
    sort_params = hf_sort_params(sort_by, sort_dir)

    results: list[dict[str, Any]] = []
    has_more = False
    target_offset = max(0, offset)
    target_end = target_offset + max(1, limit)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            while len(results) < limit and not session.get("exhausted"):
                phase_index = int(session.get("phase_index", 0))
                phases: list[str] = session.get("phases", [search])
                if phase_index >= len(phases):
                    session["exhausted"] = True
                    break

                phase_search = phases[phase_index]
                params: dict[str, Any] = {
                    "search": phase_search,
                    "limit": list_page_size,
                    **sort_params,
                }
                cursor = session.get("list_cursor")
                if cursor:
                    params["cursor"] = cursor

                response = await client.get(
                    f"{base_url}/api/models",
                    params=params,
                    headers=headers,
                )
                if response.status_code == 429:
                    raise HfIndexError(
                        "Hugging Face rate limit reached. Set ARCHGPU_BRIDGE_HF_TOKEN to a valid HF access token."
                    )
                response.raise_for_status()
                items = response.json()
                if not isinstance(items, list) or not items:
                    phase_index += 1
                    session["phase_index"] = phase_index
                    session["list_cursor"] = None
                    session["list_item_index"] = 0
                    if phase_index >= len(phases):
                        session["exhausted"] = True
                    continue

                item_index = int(session.get("list_item_index", 0))
                while item_index < len(items) and len(results) < limit:
                    item = items[item_index]
                    item_index += 1
                    session["list_item_index"] = item_index

                    model_id = item.get("id")
                    if not model_id:
                        continue
                    seen_ids: set[str] = session.setdefault("seen_ids", set())
                    if model_id in seen_ids:
                        continue
                    if item.get("private"):
                        continue
                    owner = model_id.split("/", 1)[0].lower() if "/" in model_id else ""
                    if owners and owner not in owners:
                        continue

                    entry = await _try_build_entry_from_id(
                        client,
                        base_url=base_url,
                        headers=headers,
                        model_id=model_id,
                        derive_capabilities=derive_capabilities,
                        registry=registry,
                        catalogue=catalogue,
                    )
                    if entry is None:
                        continue

                    seen_ids.add(model_id)
                    gguf_count = int(session.get("gguf_count", 0))
                    if gguf_count % 25 == 0:
                        _save_checkpoint(session, gguf_count)

                    if gguf_count >= target_offset and gguf_count < target_end:
                        results.append(entry)
                    session["gguf_count"] = gguf_count + 1

                session["list_cursor"] = _parse_next_cursor(response.headers.get("link"))
                if session["list_cursor"] is None:
                    phase_index = int(session.get("phase_index", 0)) + 1
                    session["phase_index"] = phase_index
                    session["list_cursor"] = None
                    session["list_item_index"] = 0
                    if phase_index >= len(phases):
                        session["exhausted"] = True

            has_more = not bool(session.get("exhausted"))

    except HfIndexError as exc:
        logger.warning("HF live page fetch failed: %s", exc)
        meta["error"] = str(exc)
        return results, bool(results), meta
    except httpx.HTTPError as exc:
        logger.warning("HF live page HTTP error: %s", exc)
        meta["error"] = str(exc)
        return results, bool(results), meta

    meta["count"] = int(session.get("gguf_count", 0))
    meta["exhausted"] = bool(session.get("exhausted"))
    meta["has_more"] = has_more
    return results, has_more, meta


def _save_disk_cache(settings: Any, fetched_at: float, models: list[dict[str, Any]]) -> None:
    path = _disk_cache_path(settings)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"fetched_at": fetched_at, "models": models}, default=str),
            encoding="utf-8",
        )
    except OSError:
        pass


def model_matches_query(model: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    q = query.strip().lower()
    if not q:
        return True

    field_prefixes = {
        "repo:": "repo",
        "publisher:": "publisher",
        "tag:": "tags",
        "capability:": "capabilities",
        "source:": "source",
        "fit:": "fit",
    }
    for prefix, field in field_prefixes.items():
        if q.startswith(prefix):
            needle = q[len(prefix) :].strip()
            if field == "publisher":
                return needle in str((model.get("metadata") or {}).get("publisher", "")).lower()
            if field == "source":
                return needle in str((model.get("metadata") or {}).get("source", "")).lower()
            if field == "fit":
                return needle in str((model.get("runtime_fit") or {}).get("recommendation", "")).lower()
            if field in {"tags", "capabilities"}:
                values = [str(v).lower() for v in model.get(field, [])]
                return any(needle in v for v in values)
            return needle in str(model.get(field, "")).lower()

    haystack = " ".join(
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
    return q in haystack


def sort_models(models: list[dict[str, Any]], sort_by: str, sort_dir: str) -> list[dict[str, Any]]:
    reverse = sort_dir.lower() != "asc"

    def _key(model: dict[str, Any]) -> Any:
        meta = model.get("metadata") or {}
        fit = model.get("runtime_fit") or {}
        if sort_by == "downloads":
            return int(meta.get("downloads") or 0)
        if sort_by == "likes":
            return int(meta.get("likes") or 0)
        if sort_by == "last_modified":
            return str(meta.get("last_modified") or "")
        if sort_by == "display_name":
            return str(model.get("display_name") or model.get("alias") or "").lower()
        if sort_by == "publisher":
            return str(meta.get("publisher") or "").lower()
        if sort_by == "fit":
            order = {"recommended": 0, "possible": 1, "not_recommended": 2, "unknown": 3}
            return order.get(str(fit.get("recommendation") or "unknown"), 99)
        return str(model.get("alias") or "").lower()

    return sorted(models, key=_key, reverse=reverse)


def paginate_models(
    models: list[dict[str, Any]],
    *,
    page: int | None,
    page_size: int | None,
) -> dict[str, Any]:
    total = len(models)
    if page is None and page_size is None:
        return {
            "models": models,
            "total": total,
            "page": 1,
            "page_size": total,
            "total_pages": 1 if total else 0,
            "has_prev": False,
            "has_next": False,
        }

    effective_page = max(1, page or 1)
    effective_page_size = max(1, min(page_size or 25, 200))
    total_pages = (total + effective_page_size - 1) // effective_page_size if total else 0
    if total_pages and effective_page > total_pages:
        effective_page = total_pages
    start = (effective_page - 1) * effective_page_size
    end = start + effective_page_size
    page_models = models[start:end]
    return {
        "models": page_models,
        "total": total,
        "page": effective_page,
        "page_size": effective_page_size,
        "total_pages": total_pages,
        "has_prev": effective_page > 1,
        "has_next": effective_page < total_pages,
    }
