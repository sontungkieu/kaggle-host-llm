from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException, status

from .openai_models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    chunk_payload,
    estimate_usage,
)
from .registry import WorkerRegistry


class Dispatcher:
    def __init__(self, registry: WorkerRegistry, job_timeout_seconds: float) -> None:
        self.registry = registry
        self.job_timeout_seconds = job_timeout_seconds

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        job_id, worker, pending = await self._send_job(request)
        content_parts: list[str] = []
        finish_reason = "stop"
        worker_usage: ChatCompletionUsage | None = None
        try:
            while True:
                message = await asyncio.wait_for(
                    pending.queue.get(), timeout=self.job_timeout_seconds
                )
                message_type = message.get("type")
                if message_type == "token_delta":
                    content_parts.append(str(message.get("content") or ""))
                elif message_type == "job_done":
                    final_content = message.get("content")
                    if final_content is not None:
                        content_parts = [str(final_content)]
                    finish_reason = str(message.get("finish_reason") or "stop")
                    worker_usage = parse_worker_usage(message)
                    break
                elif message_type == "job_error":
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=str(message.get("error") or "worker failed"),
                    )
        except TimeoutError as exc:
            await self.registry.fail_worker(worker.node_id, job_id, "job timed out")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="worker job timed out",
            ) from exc
        finally:
            await self.registry.release_job(worker.node_id, job_id)

        message = ChatMessage(role="assistant", content="".join(content_parts))
        usage = worker_usage or estimate_usage(
            messages=request.messages,
            completion=message.content,
        )
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=message,
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        job_id, worker, pending = await self._send_job(request)

        async def events() -> AsyncIterator[str]:
            try:
                while True:
                    message = await asyncio.wait_for(
                        pending.queue.get(), timeout=self.job_timeout_seconds
                    )
                    message_type = message.get("type")
                    if message_type == "token_delta":
                        payload = chunk_payload(
                            completion_id=completion_id,
                            created=created,
                            model=request.model,
                            content=str(message.get("content") or ""),
                        )
                        yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                    elif message_type == "job_done":
                        final_content = str(message.get("content") or "")
                        if final_content:
                            payload = chunk_payload(
                                completion_id=completion_id,
                                created=created,
                                model=request.model,
                                content=final_content,
                            )
                            yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                        payload = chunk_payload(
                            completion_id=completion_id,
                            created=created,
                            model=request.model,
                            content="",
                            finish_reason=str(message.get("finish_reason") or "stop"),
                        )
                        yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    elif message_type == "job_error":
                        payload = {
                            "error": {
                                "message": str(message.get("error") or "worker failed"),
                                "type": "worker_error",
                            }
                        }
                        yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                        yield "data: [DONE]\n\n"
                        break
            except TimeoutError:
                payload = {
                    "error": {
                        "message": "worker job timed out",
                        "type": "worker_timeout",
                    }
                }
                yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                await self.registry.release_job(worker.node_id, job_id)

        return events()

    async def _send_job(self, request: ChatCompletionRequest) -> tuple[str, Any, Any]:
        worker = await self.registry.select_worker(request.model)
        if worker is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"no healthy worker is available for model {request.model}",
            )

        job_id = f"job-{uuid.uuid4().hex}"
        pending = await self.registry.reserve_job(worker, job_id)
        payload = {
            "type": "job",
            "job_id": job_id,
            "model": request.model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
            "timeout": self.job_timeout_seconds,
        }
        try:
            await worker.websocket.send_json(payload)
        except Exception as exc:
            await self.registry.release_job(worker.node_id, job_id)
            await self.registry.unregister(worker.node_id, reason="worker send failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="failed to send job to worker",
            ) from exc
        return job_id, worker, pending


def parse_worker_usage(message: dict[str, Any]) -> ChatCompletionUsage | None:
    raw_usage = message.get("usage")
    if not isinstance(raw_usage, dict):
        raw_usage = message

    try:
        prompt_tokens = int(raw_usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(raw_usage.get("completion_tokens", 0) or 0)
        total_tokens = int(raw_usage.get("total_tokens", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        return None

    if prompt_tokens <= 0 and completion_tokens <= 0 and total_tokens <= 0:
        return None
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return ChatCompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
