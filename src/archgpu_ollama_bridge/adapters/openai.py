from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..services import AppServices


def build_openai_router(services: AppServices) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["openai"])

    @router.get("/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model.openai_name,
                    "object": "model",
                    "owned_by": "archgpu-bridge",
                }
                for model in services.registry.list_models()
            ],
        }

    @router.post("/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        model_name = payload.get("model")
        if not model_name:
            raise HTTPException(status_code=422, detail="Request body must include a model")

        try:
            target = await services.router.route(model_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if payload.get("stream"):
            stream = services.proxy_client.stream_post(
                base_url=target.base_url,
                path="/v1/chat/completions",
                payload=payload,
            )
            return StreamingResponse(stream, media_type="text/event-stream")

        try:
            response = await services.proxy_client.post_json(
                base_url=target.base_url,
                path="/v1/chat/completions",
                payload=payload,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return JSONResponse(status_code=response.status_code, content=response.json_body)

    return router
