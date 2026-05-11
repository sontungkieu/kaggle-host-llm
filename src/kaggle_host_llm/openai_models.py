from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ChatTextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatImageUrl(BaseModel):
    url: str = Field(min_length=1)


class ChatImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ChatImageUrl


ChatContentPart = ChatTextContentPart | ChatImageContentPart
ChatContent = str | list[ChatContentPart]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: ChatContent


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def require_user_message(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if not any(message.role == "user" for message in messages):
            raise ValueError("at least one user message is required")
        return messages


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    lexical_count = len(TOKEN_PATTERN.findall(text))
    char_count = math.ceil(len(text) / 4)
    return max(1, lexical_count, char_count)


def message_text(content: ChatContent) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, ChatTextContentPart):
            parts.append(part.text)
    return "\n".join(parts)


def message_has_image(content: ChatContent) -> bool:
    return not isinstance(content, str) and any(
        isinstance(part, ChatImageContentPart) for part in content
    )


def estimate_usage(
    *,
    messages: list[ChatMessage],
    completion: str,
) -> ChatCompletionUsage:
    prompt_tokens = 2
    for message in messages:
        prompt_tokens += 4
        prompt_tokens += estimate_text_tokens(message.role)
        prompt_tokens += estimate_text_tokens(message_text(message.content))

    completion_tokens = estimate_text_tokens(completion)
    return ChatCompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def chunk_payload(
    *,
    completion_id: str,
    created: int,
    model: str,
    content: str,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
