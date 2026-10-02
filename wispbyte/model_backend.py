[Reading 163 lines from start (total: 163 lines, 0 remaining)]

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass
class GenerationResult:
    text: str
    tool_calls: list[dict[str, Any]]
    prompt_tokens: int = 0
    completion_tokens: int = 0


class TransformersBackend:
    """Small, CPU-safe Hugging Face backend.

    The exact model is configured with MODEL_ID. Nothing is hard-coded to the
    original DGX Spark/TensorFold runtime, so Wispbyte can host a normal model.
    """

    def __init__(self) -> None:
        self.model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
        self.revision = os.getenv("MODEL_REVISION") or None
        self.max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", "32768"))
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            trust_remote_code = os.getenv("MODEL_TRUST_REMOTE_CODE", "0") == "1"
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=trust_remote_code,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self._model.to("cpu")
            self._model.eval()

    def _render_tools(self, tools: list[dict[str, Any]]) -> str:
        if not tools:
            return ""
        safe = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        return (
            "\n\nTOOLS AVAILABLE:\n"
            f"{safe}\n"
            "When you need a tool, emit exactly one or more tags of the form "
            '<tool_call>{"name":"TOOL_NAME","arguments":{}}</tool_call>. '
            "Do not put markdown around tool_call tags. Tool arguments must be valid JSON."
        )

    def _render_messages(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        chunks: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get("type")
                        if ptype == "text":
                            text_parts.append(str(part.get("text", "")))
                        elif ptype == "tool_result":
                            text_parts.append(f"[tool_result {part.get('tool_use_id')}] {part.get('content', '')}")
                        elif ptype == "tool_use":
                            text_parts.append(
                                f"[tool_use {part.get('id')}] {part.get('name')} "
                                f"{json.dumps(part.get('input', {}), ensure_ascii=False)}"
                            )
                    else:
                        text_parts.append(str(part))
                content = "\n".join(text_parts)
            chunks.append(f"{role.upper()}:\n{content}")
        if not chunks or messages[-1].get("role") != "system":
            chunks.insert(0, "SYSTEM:\nYou are a helpful local AI assistant. Follow the user's request and use tools when needed.")
        chunks[0] += self._render_tools(tools)
        chunks.append("ASSISTANT:\n")
        return "\n\n".join(chunks)

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = 0.2,
    ) -> GenerationResult:
        self.load()
        import torch

        prompt = self._render_messages(messages, tools or [])
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature is not None and temperature > 0,
                temperature=max(float(temperature or 0.2), 1e-5),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        tool_calls = []
        for match in _TOOL_CALL_RE.finditer(text):
            try:
                obj = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("name"):
                args = obj.get("arguments", {})
                if isinstance(args, dict):
                    tool_calls.append({"name": obj["name"], "arguments": args})
        if tool_calls:
            clean = _TOOL_CALL_RE.sub("", text).strip()
        else:
            clean = text.strip()
        return GenerationResult(
            text=clean,
            tool_calls=tool_calls,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=int(new_tokens.shape[0]),
        )


_backend: TransformersBackend | None = None


def get_backend() -> TransformersBackend:
    global _backend
    if _backend is None:
        _backend = TransformersBackend()
    return _backend

[executed on device: Alexanders-MacBook-Pro.local (bd265cb2-62d5-47e5-937a-511d5a03b340)]