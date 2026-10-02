from __future__ import annotations

import json
import os
import secrets
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .llama_backend import MODEL_NAME, anthropic_messages_to_openai, anthropic_tools_to_openai, llama_health, openai_to_anthropic

APP_NAME = os.getenv("APP_NAME", "GLM Bot")
APP_API_KEY = os.getenv("APP_API_KEY", "")
LLAMA_BASE = os.getenv("LLAMA_BASE", f"http://127.0.0.1:{os.getenv('LLAMA_PORT', '8090')}")
app = FastAPI(title=APP_NAME, version="2.0.0")
MCP_TOOLS: dict[str, dict[str, Any]] = {}
MCP_ALLOWED = {x.strip() for x in os.getenv("MCP_ALLOWED_TOOLS", "").split(",") if x.strip()}


def auth(authorization: str | None = Header(default=None)) -> None:
    if not APP_API_KEY:
        return
    if not authorization or not secrets.compare_digest(authorization, f"Bearer {APP_API_KEY}"):
        raise HTTPException(status_code=401, detail="invalid API key")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "model": MODEL_NAME, "llama_server": await llama_health()}


@app.get("/v1/models", dependencies=[Depends(auth)])
def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]}


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
    MCP_TOOLS[name] = {"name": name, "description": str(body.get("description", "")), "input_schema": body.get("input_schema", {"type": "object", "properties": {}})}
    return {"ok": True, "tool": MCP_TOOLS[name]}


@app.delete("/mcp/register/{name}", dependencies=[Depends(auth)])
def unregister_tool(name: str) -> dict[str, Any]:
    MCP_TOOLS.pop(name, None)
    return {"ok": True}


@app.post("/v1/messages", dependencies=[Depends(auth)])
async def messages(request: Request):
    import httpx
    body = await request.json()
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "messages must be a non-empty list")
    payload: dict[str, Any] = {"model": body.get("model", MODEL_NAME), "messages": anthropic_messages_to_openai(body), "max_tokens": min(int(body.get("max_tokens", 256)), int(os.getenv("MAX_OUTPUT_TOKENS", "256"))), "stream": bool(body.get("stream", False))}
    if body.get("temperature") is not None:
        payload["temperature"] = float(body["temperature"])
    tools_in = body.get("tools") or list(MCP_TOOLS.values())
    if tools_in:
        payload["tools"] = anthropic_tools_to_openai(tools_in)
        payload["tool_choice"] = "auto"
    if not payload["stream"]:
        async with httpx.AsyncClient(timeout=float(os.getenv("MODEL_TIMEOUT", "900"))) as client:
            response = await client.post(f"{LLAMA_BASE}/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        return JSONResponse(openai_to_anthropic(data, str(body.get("model", MODEL_NAME))))

    async def stream():
        started: set[int] = set()
        stop_reason = "end_turn"
        usage = {"input_tokens": 0, "output_tokens": 0}
        message = {"id": "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant", "model": body.get("model", MODEL_NAME), "content": [], "stop_reason": None, "stop_sequence": None, "usage": usage}
        def evt(name: str, data: dict[str, Any]) -> str:
            return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        yield evt("message_start", {"type": "message_start", "message": message})
        async with httpx.AsyncClient(timeout=float(os.getenv("MODEL_TIMEOUT", "900"))) as client:
            async with client.stream("POST", f"{LLAMA_BASE}/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    choice = (chunk.get("choices") or [{}])[0]
                    finish = choice.get("finish_reason")
                    if finish:
                        stop_reason = "tool_use" if finish in {"tool_calls", "function_call"} else ("max_tokens" if finish == "length" else "end_turn")
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        if 0 not in started:
                            started.add(0)
                            yield evt("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
                        yield evt("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": delta["content"]}})
                    for call in delta.get("tool_calls") or []:
                        idx = int(call.get("index", 0)) + 1
                        fn = call.get("function") or {}
                        if idx not in started:
                            started.add(idx)
                            yield evt("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "tool_use", "id": call.get("id") or ("toolu_" + uuid.uuid4().hex[:20]), "name": fn.get("name", ""), "input": {}}})
                        if fn.get("arguments"):
                            yield evt("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}})
                    if chunk.get("usage"):
                        usage["input_tokens"] = int(chunk["usage"].get("prompt_tokens", 0) or 0)
                        usage["output_tokens"] = int(chunk["usage"].get("completion_tokens", 0) or 0)
        for index in sorted(started):
            yield evt("content_block_stop", {"type": "content_block_stop", "index": index})
        yield evt("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": usage})
        yield evt("message_stop", {"type": "message_stop"})
    return StreamingResponse(stream(), media_type="text/event-stream")
