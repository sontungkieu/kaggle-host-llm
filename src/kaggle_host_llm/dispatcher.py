from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException, status

from .groq_client import GroqClient, parse_groq_route
from .kaggle_log import log_kaggle_event
from .openai_models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    chunk_payload,
    estimate_usage,
    message_has_image,
)
from .ocr_models import OcrRequest, OcrResponse
from .registry import WorkerRegistry


class Dispatcher:
    def __init__(
        self,
        registry: WorkerRegistry,
        job_timeout_seconds: float,
        *,
        groq_key_file: str = ".secrets/groq_key.env",
        groq_base_url: str = "https://api.groq.com/openai/v1",
        groq_client: GroqClient | None = None,
    ) -> None:
        self.registry = registry
        self.job_timeout_seconds = job_timeout_seconds
        self.groq_client = groq_client or GroqClient(
            key_file=groq_key_file,
            base_url=groq_base_url,
            timeout_seconds=job_timeout_seconds,
        )

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        groq_route = parse_groq_route(request)
        if groq_route is not None:
            return await self.groq_client.complete(groq_route)

        self._ensure_worker_compatible(request)
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
                    log_kaggle_event(
                        "chat_worker_job_error",
                        level="error",
                        owner=worker.owner,
                        backend=worker.model,
                        node_id=worker.node_id,
                        accelerator=worker.accelerator,
                        served_model=worker.model,
                        message=str(message.get("error") or "worker failed"),
                        details={"job_id": job_id},
                    )
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=str(message.get("error") or "worker failed"),
                    )
        except TimeoutError as exc:
            log_kaggle_event(
                "chat_worker_timeout",
                level="error",
                owner=worker.owner,
                backend=worker.model,
                node_id=worker.node_id,
                accelerator=worker.accelerator,
                served_model=worker.model,
                message="worker job timed out",
                details={"job_id": job_id},
            )
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
        groq_route = parse_groq_route(request)
        if groq_route is not None:
            return self.groq_client.stream(groq_route)

        self._ensure_worker_compatible(request)
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
                log_kaggle_event(
                    "chat_worker_timeout",
                    level="error",
                    owner=worker.owner,
                    backend=worker.model,
                    node_id=worker.node_id,
                    accelerator=worker.accelerator,
                    served_model=worker.model,
                    message="worker job timed out",
                    details={"job_id": job_id},
                )
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

    async def ocr(self, request: OcrRequest) -> OcrResponse:
        job_id, worker, pending = await self._send_ocr_job(request)
        try:
            while True:
                message = await asyncio.wait_for(
                    pending.queue.get(), timeout=self.job_timeout_seconds
                )
                message_type = message.get("type")
                if message_type in {"ocr_done", "job_done"}:
                    result = message.get("result")
                    if not isinstance(result, dict):
                        result = {
                            "text": str(message.get("content") or ""),
                            "markdown": str(message.get("content") or ""),
                        }
                    return self._format_ocr_response(request, result)
                if message_type in {"ocr_error", "job_error"}:
                    log_kaggle_event(
                        "ocr_worker_job_error",
                        level="error",
                        owner=worker.owner,
                        backend=worker.model,
                        node_id=worker.node_id,
                        accelerator=worker.accelerator,
                        served_model=worker.model,
                        message=str(message.get("error") or "OCR worker failed"),
                        details={"job_id": job_id, "message_type": message_type},
                    )
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=str(message.get("error") or "OCR worker failed"),
                    )
        except TimeoutError as exc:
            log_kaggle_event(
                "ocr_worker_timeout",
                level="error",
                owner=worker.owner,
                backend=worker.model,
                node_id=worker.node_id,
                accelerator=worker.accelerator,
                served_model=worker.model,
                message="OCR job timed out",
                details={"job_id": job_id},
            )
            await self.registry.fail_worker(worker.node_id, job_id, "OCR job timed out")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="OCR worker job timed out",
            ) from exc
        finally:
            await self.registry.release_job(worker.node_id, job_id)

    async def _send_job(self, request: ChatCompletionRequest) -> tuple[str, Any, Any]:
        worker = await self.registry.select_worker(request.model)
        if worker is None:
            log_kaggle_event(
                "no_worker_available",
                level="warning",
                backend=request.model,
                served_model=request.model,
                message=f"no healthy worker is available for model {request.model}",
            )
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
            "messages": [message.model_dump(mode="json") for message in request.messages],
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
            log_kaggle_event(
                "worker_send_failed",
                level="error",
                owner=worker.owner,
                backend=worker.model,
                node_id=worker.node_id,
                accelerator=worker.accelerator,
                served_model=worker.model,
                message=repr(exc),
                details={"job_id": job_id},
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="failed to send job to worker",
            ) from exc
        return job_id, worker, pending

    async def _send_ocr_job(self, request: OcrRequest) -> tuple[str, Any, Any]:
        worker = await self.registry.select_worker(request.model)
        if worker is None:
            log_kaggle_event(
                "no_ocr_worker_available",
                level="warning",
                backend=request.model,
                served_model=request.model,
                message=f"no healthy OCR worker is available for model {request.model}",
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"no healthy OCR worker is available for model {request.model}",
            )

        job_id = f"ocr-job-{uuid.uuid4().hex}"
        pending = await self.registry.reserve_job(worker, job_id)
        payload = {
            "type": "ocr_job",
            "job_id": job_id,
            "model": request.model,
            "request": request.worker_payload(),
            "timeout": self.job_timeout_seconds,
        }
        try:
            await worker.websocket.send_json(payload)
        except Exception as exc:
            await self.registry.release_job(worker.node_id, job_id)
            await self.registry.unregister(worker.node_id, reason="OCR worker send failed")
            log_kaggle_event(
                "ocr_worker_send_failed",
                level="error",
                owner=worker.owner,
                backend=worker.model,
                node_id=worker.node_id,
                accelerator=worker.accelerator,
                served_model=worker.model,
                message=repr(exc),
                details={"job_id": job_id},
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="failed to send OCR job to worker",
            ) from exc
        return job_id, worker, pending

    @staticmethod
    def _format_ocr_response(request: OcrRequest, result: dict[str, Any]) -> OcrResponse:
        text = str(result.get("text") or "")
        markdown = str(result.get("markdown") or "")
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        raw_pages = result.get("pages") if isinstance(result.get("pages"), list) else []
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}

        pages = [
            {
                "index": page.get("index"),
                "markdown": page.get("markdown", ""),
            }
            for page in raw_pages
            if isinstance(page, dict)
        ]

        if request.return_format == "text":
            return OcrResponse(
                model=request.model,
                text=text,
                metadata=metadata,
            )
        if request.return_format == "markdown":
            return OcrResponse(
                model=request.model,
                text=text,
                markdown=markdown,
                pages=pages,
                metadata=metadata,
            )
        if request.return_format == "json":
            return OcrResponse(
                model=request.model,
                data=data,
                pages=pages,
                metadata=metadata,
            )
        return OcrResponse(
            model=request.model,
            text=text,
            markdown=markdown,
            data=data,
            pages=raw_pages,
            metadata=metadata,
        )

    @staticmethod
    def _ensure_worker_compatible(request: ChatCompletionRequest) -> None:
        if any(message_has_image(message.content) for message in request.messages):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "image input requires a Groq vision route; prefix the user message "
                    "with groq:<vision-model>"
                ),
            )


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
