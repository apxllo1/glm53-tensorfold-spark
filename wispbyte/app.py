[Reading 138 lines from start (total: 138 lines, 0 remaining)]

from __future__ import annotations

import json
import os
import secrets
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .model_backend import get_backend

APP_NAME = os.getenv("APP_NAME", "GLM Bot")
MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("MODEL_ID", "local-model"))
APP_API_KEY = os.getenv("APP_API_KEY", "")

app = FastAPI(title=APP_NAME, version="1.0.0")

MCP_TOOLS: dict[str, dict[str, Any]] = {}
MCP_ALLOWED = {
    x.strip() for x in os.getenv("MCP_ALLOWED_TOOLS", "").split(",") if x.strip()
}


def auth(authorization: str | None = Header(default=None)) -> None:
    if not APP_API_KEY:
        return
    expected = f"Bearer {APP_API_KEY}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="invalid API key")


@app.get("/health")
def health() -> dict[str, Any]:
    b = get_backend()
    return {"ok": True, "model": MODEL_NAME, "loaded": b.loaded}


@app.get("/v1/models", dependencies=[Depends(auth)])
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/tools", dependencies=[Depends(auth)])
def tools() -> dict[str, Any]:
    return {"tools": list(MCP_TOOLS.values())}


@app.post("/mcp/register", dependencies=[Depends(auth)])
async def register_tool(request: Request) -> dict[str, Any]:
    body = await request.json()
    name = str(body.get("name", "")).strip()
    if not name:
        raise HTTPException(400, "tool name required")
    if MCP_ALLOWED and name not in MCP_ALLOWED:
        raise HTTPException(403, "tool not allowed")
    MCP_TOOLS[name] = {
        "name": name,
        "description": str(body.get("description", "")),
        "input_schema": body.get("input_schema", {"type": "object", "properties": {}}),
    }
    return {"ok": True, "tool": MCP_TOOLS[name]}


@app.delete("/mcp/register/{name}", dependencies=[Depends(auth)])
def unregister_tool(name: str) -> dict[str, Any]:
    MCP_TOOLS.pop(name, None)
    return {"ok": True}


def _anthropic_response(result, model: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if result.text:
        content.append({"type": "text", "text": result.text})
    for i, tc in enumerate(result.tool_calls):
        content.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:20]}",
            "name": tc["name"],
            "input": tc.get("arguments", {}),
        })
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content or [{"type": "text", "text": ""}],
        "stop_reason": "tool_use" if result.tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": result.prompt_tokens,
            "output_tokens": result.completion_tokens,
        },
    }


@app.post("/v1/messages", dependencies=[Depends(auth)])
async def messages(request: Request):
    body = await request.json()
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise HTTPException(400, "messages must be a non-empty list")
    max_tokens = int(body.get("max_tokens", 1024))
    temperature = body.get("temperature", 0.2)
    tools = body.get("tools") or list(MCP_TOOLS.values())
    result = get_backend().generate(msgs, tools=tools, max_tokens=max_tokens, temperature=temperature)
    response = _anthropic_response(result, body.get("model", MODEL_NAME))

    if not body.get("stream"):
        return JSONResponse(response)

    # The CPU backend generates synchronously; stream a valid Anthropic-shaped SSE
    # response in a few chunks so Discord/agent clients can consume it normally.
    text_value = ""
    for part in response["content"]:
        if part["type"] == "text":
            text_value = part["text"]
            break

    def event(name: str, data: dict[str, Any]) -> str:
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def stream():
        yield event("message_start", {"type": "message_start", "message": {k: v for k, v in response.items() if k != "content"} | {"content": []}})
        if text_value:
            yield event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
            for i in range(0, len(text_value), 64):
                yield event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text_value[i:i+64]}})
            yield event("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield event("message_delta", {"type": "message_delta", "delta": {"stop_reason": response["stop_reason"], "stop_sequence": None}, "usage": response["usage"]})
        yield event("message_stop", {"type": "message_stop"})

    return StreamingResponse(stream(), media_type="text/event-stream")

[executed on device: Alexanders-MacBook-Pro.local (bd265cb2-62d5-47e5-937a-511d5a03b340)]