from typing import Any, Dict, List, Type, Union
from pydantic import BaseModel, ValidationError

try:
    from ollama import chat
    from ollama._types import ResponseError 
except Exception:
    chat = None
    ResponseError = Exception  


def _ollama_schema_sanitize(schema: Dict[str, Any]) -> Dict[str, Any]:
    allowed_obj = {"type", "properties", "items", "required", "enum", "title", "additionalProperties"}
    banned = {"$defs", "$ref", "allOf", "anyOf", "oneOf", "pattern", "format", "const", "not", "definitions"}

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            node = {k: v for k, v in node.items() if k not in banned}
            node = {k: clean(v) for k, v in node.items()}
            t = node.get("type")
            if t == "object":
                node = {k: v for k, v in node.items() if k in allowed_obj}
            elif t == "array":
                node = {k: v for k, v in node.items() if k in {"type", "items"}}
            return node
        if isinstance(node, list):
            return [clean(x) for x in node]
        return node

    return clean(schema)


class LLMInterface:
    def generate(
        self,
        *,
        messages: List[Dict[str, Any]],
        response_model: Union[Type[BaseModel], BaseModel],
    ) -> str:
        raise NotImplementedError


class OllamaInterface(LLMInterface):
    def __init__(self, model: str = "llama3.1", temperature: float = 0.0):
        if chat is None:
            raise RuntimeError("Couldn't import ollama.chat (pip install -U ollama)")
        self.model = model
        self.temperature = temperature

    def _chat(self, messages: List[Dict[str, Any]], *, fmt: Any | None) -> str:
        resp = chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature},
            format=fmt,
        )
        return resp.get("message", {}).get("content", "")

    def generate(
        self,
        *,
        messages: List[Dict[str, Any]],
        response_model: Union[Type[BaseModel], BaseModel],
    ) -> str:
        model_cls: Type[BaseModel] = (
            response_model if isinstance(response_model, type) and issubclass(response_model, BaseModel)
            else response_model.__class__
        )

        full_schema = model_cls.model_json_schema()

        try:
            content = self._chat(messages, fmt=full_schema)
            model_cls.model_validate_json(content)
            return content
        except (ValidationError, ResponseError):
            pass

        try:
            clean_schema = _ollama_schema_sanitize(full_schema)
            content2 = self._chat(messages, fmt=clean_schema)
            model_cls.model_validate_json(content2)
            return content2
        except (ValidationError, ResponseError):
            pass

        messages2 = messages + [{
            "role": "system",
            "content": (
                "Respond with a single JSON object that matches this schema exactly. "
                "No prose, no code fences.\n\n" + str(full_schema)
            ),
        }]
        content3 = self._chat(messages2, fmt="json")
        model_cls.model_validate_json(content3)
        return content3
