from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException, status

from .openai_models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)


GROQ_PREFIX_RE = re.compile(r"^\s*groq:\s*([^\s]+)\s*", re.IGNORECASE)


@dataclass(frozen=True)
class GroqRoute:
    model: str
    request: ChatCompletionRequest


def split_secret_values(value: str) -> list[str]:
    raw_values = re.split(r"[\s,;]+", value.strip())
    return [raw_value for raw_value in raw_values if raw_value]


def load_groq_keys(path: str | Path) -> list[str]:
    key_path = Path(path)
    if not key_path.exists():
        return []

    keys: list[str] = []
    for raw_line in key_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            _, value = line.split("=", 1)
            keys.extend(split_secret_values(value.strip().strip('"').strip("'")))
        else:
            keys.extend(split_secret_values(line))

    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def parse_groq_route(request: ChatCompletionRequest) -> GroqRoute | None:
    dumped_messages = [message.model_dump(mode="json") for message in request.messages]

    for index in range(len(dumped_messages) - 1, -1, -1):
        message = dumped_messages[index]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            match = GROQ_PREFIX_RE.match(content)
            if not match:
                continue
            message["content"] = content[match.end() :].lstrip()
            return GroqRoute(
                model=match.group(1),
                request=request.model_copy(
                    update={
                        "model": match.group(1),
                        "messages": [
                            ChatMessage.model_validate(item) for item in dumped_messages
                        ],
                    },
                    deep=True,
                ),
            )
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    continue
                text = str(part.get("text") or "")
                match = GROQ_PREFIX_RE.match(text)
                if not match:
                    continue
                part["text"] = text[match.end() :].lstrip()
                return GroqRoute(
                    model=match.group(1),
                    request=request.model_copy(
                        update={
                            "model": match.group(1),
                            "messages": [
                                ChatMessage.model_validate(item)
                                for item in dumped_messages
                            ],
                        },
                        deep=True,
                    ),
                )
    return None


class GroqKeyPool:
    def __init__(self, key_file: str | Path) -> None:
        self.key_file = str(key_file)
        self._keys: list[str] = []
        self._mtime_ns: int | None = None
        self._index = 0
        self._lock = asyncio.Lock()

    def has_keys(self) -> bool:
        self._refresh_if_needed()
        return bool(self._keys)

    async def next_key(self) -> str:
        async with self._lock:
            self._refresh_if_needed()
            if not self._keys:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"no Groq API keys found in {self.key_file}",
                )
            key = self._keys[self._index % len(self._keys)]
            self._index = (self._index + 1) % len(self._keys)
            return key

    def _refresh_if_needed(self) -> None:
        path = Path(self.key_file)
        mtime_ns = path.stat().st_mtime_ns if path.exists() else None
        if mtime_ns == self._mtime_ns:
            return
        self._keys = load_groq_keys(path)
        self._mtime_ns = mtime_ns
        if self._keys:
            self._index %= len(self._keys)
        else:
            self._index = 0


class GroqClient:
    def __init__(
        self,
        *,
        key_file: str | Path,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_seconds: float = 600.0,
    ) -> None:
        self.key_pool = GroqKeyPool(key_file)
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def has_keys(self) -> bool:
        return self.key_pool.has_keys()

    async def complete(self, route: GroqRoute) -> ChatCompletionResponse:
        api_key = await self.key_pool.next_key()
        payload = self._request_payload(route.request, stream=False)
        timeout = httpx.Timeout(self.timeout_seconds, connect=30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(api_key),
                json=payload,
            )
        if response.status_code >= 400:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Groq HTTP {response.status_code}: {response.text[:500]}",
            )

        result = response.json()
        message = ChatMessage(
            role="assistant",
            content=result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", ""),
        )
        return ChatCompletionResponse(
            id=str(result.get("id") or f"chatcmpl-{uuid.uuid4().hex}"),
            object=str(result.get("object") or "chat.completion"),
            created=int(result.get("created") or time.time()),
            model=route.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=message,
                    finish_reason=str(
                        result.get("choices", [{}])[0].get("finish_reason") or "stop"
                    ),
                )
            ],
            usage=self._usage(result.get("usage")),
        )

    async def stream(self, route: GroqRoute) -> AsyncIterator[str]:
        api_key = await self.key_pool.next_key()
        payload = self._request_payload(route.request, stream=True)
        timeout = httpx.Timeout(self.timeout_seconds, connect=30, read=self.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(api_key),
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    payload = {
                        "error": {
                            "message": f"Groq HTTP {response.status_code}: {body[:500]}",
                            "type": "groq_error",
                        }
                    }
                    yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        yield f"{line}\n\n"

    def _request_payload(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                message.model_dump(mode="json") for message in request.messages
            ],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_completion_tokens": request.max_tokens,
            "stream": stream,
        }
        return payload

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _usage(raw_usage: Any) -> ChatCompletionUsage:
        if not isinstance(raw_usage, dict):
            return ChatCompletionUsage()
        prompt_tokens = int(raw_usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(raw_usage.get("completion_tokens", 0) or 0)
        total_tokens = int(raw_usage.get("total_tokens", 0) or 0)
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        return ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
