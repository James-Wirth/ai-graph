# aigraph/core/messages.py
from __future__ import annotations
import datetime as dt
import uuid
from typing import Any, Dict
from pydantic import BaseModel, Field


class Message(BaseModel):
    type: str
    body: Any
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ts: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    headers: Dict[str, Any] = Field(default_factory=dict)

    def with_header(self, **kv) -> "Message":
        new = self.model_copy(deep=True)
        new.headers.update(kv)
        return new

    def child(self, *, type: str, body: Any, **extra_headers) -> "Message":
        corr = self.headers.get("correlation_id", self.id)
        return Message(
            type=type,
            body=body,
            headers={
                "correlation_id": corr,
                "parent_id": self.id,
                **extra_headers,
            },
        )
