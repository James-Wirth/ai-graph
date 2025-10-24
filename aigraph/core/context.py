# aigraph/core/context.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pydantic import BaseModel
from aigraph.interfaces.ollama import LLMInterface


@dataclass
class Context:
    run_vars: Dict[str, Any] = field(default_factory=dict)
    llm: Optional[LLMInterface] = None
    cfg: Optional[Any] = None

    def structured(
        self,
        *,
        model: type[BaseModel] | BaseModel,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        if self.llm is None:
            raise RuntimeError("No LLMInterface configured on context.")
        messages = [
            {
                "role": "system",
                "content": "You are a precise JSON generator. Return ONLY valid JSON.",
            }
        ]
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.llm.generate(messages=messages, response_model=model, temperature=temperature)
