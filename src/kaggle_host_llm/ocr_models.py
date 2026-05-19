from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class OcrRequest(BaseModel):
    model: str = Field(default="paddleocr-ppstructurev3", min_length=1)
    image_url: str | None = None
    image_base64: str | None = None
    document_url: str | None = None
    document_base64: str | None = None
    filename: str | None = None
    mime_type: str | None = None
    return_format: Literal["markdown", "text", "json", "all"] = "markdown"
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_input(self) -> "OcrRequest":
        inputs = [
            self.image_url,
            self.image_base64,
            self.document_url,
            self.document_base64,
        ]
        if sum(1 for value in inputs if value) != 1:
            raise ValueError(
                "provide exactly one of image_url, image_base64, document_url, document_base64"
            )
        return self

    def worker_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OcrResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"ocr-{uuid.uuid4().hex}")
    object: str = "ocr.result"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    text: str = ""
    markdown: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    pages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
