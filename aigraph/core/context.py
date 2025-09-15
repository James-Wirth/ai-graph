# aigraph/core/context.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from aigraph.core.tools import ToolRegistry
from aigraph.interfaces.ollama import LLMInterface


@dataclass
class Artifact:
    kind: str
    data: Any
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Inventory:
    slots: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.slots.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.slots[key] = value


@dataclass
class Context:
    run_vars: Dict[str, Any] = field(default_factory=dict)
    inventory: Inventory = field(default_factory=Inventory)
    tools: Optional[ToolRegistry] = None
    llm: Optional[LLMInterface] = None
    cfg: Optional[Any] = None
    artifacts: List[Artifact] = field(default_factory=list)

    def emit(self, *, kind: str, data: Any, **meta) -> None:
        self.artifacts.append(Artifact(kind=kind, data=data, meta=meta))

    def call_tool(self, name: str, **payload):
        if not self.tools:
            raise RuntimeError("No ToolRegistry configured on context.")
        return self.tools.get(name).call(payload)

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
