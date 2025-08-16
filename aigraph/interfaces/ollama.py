from typing import Any, Dict, Type, Union
from pydantic import BaseModel, ValidationError


try:
    from ollama import chat
except ImportError:
    chat = None


def _ollama_schema_sanitize(schema: Dict[str, Any]) -> Dict[str, Any]:

    allowed_keys = {"type", "properties", "items", "required", "enum", "title"}  
    banned_keys = {"$defs", "$ref", "allOf", "anyOf", "oneOf", "pattern", "format", "const", "not"}

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            node = {k: v for k, v in node.items() if k not in banned_keys}
            node = {k: clean(v) for k, v in node.items()}

            if "type" in node and node.get("type") == "object":
                node = {k: v for k, v in node.items() if k in allowed_keys}
            elif "type" in node and node.get("type") == "array":
                keep = {"type", "items"}
                node = {k: v for k, v in node.items() if k in keep}
            return node
        if isinstance(node, list):
            return [clean(x) for x in node]
        return node

    return clean(schema)


class LLMInterface:
    def generate(self, prompt: str, response_model: Union[Type[BaseModel], BaseModel], **kwargs) -> str:
        raise NotImplementedError


class OllamaInterface(LLMInterface):
    def __init__(self, model: str = "llama3.1", temperature: float = 0.0):
        if chat is None:
            raise RuntimeError("Couldn't import ollama.chat")
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, response_model: Union[Type[BaseModel], BaseModel], **kwargs) -> str:
        model_cls: Type[BaseModel] = (
            response_model if isinstance(response_model, type) and issubclass(response_model, BaseModel)
            else response_model.__class__  
        )

        raw_schema = model_cls.model_json_schema()
        schema = _ollama_schema_sanitize(raw_schema)

        try:
            resp = chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
                format=schema,   
            )
        except Exception as e:
            resp = chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt + "\nReturn ONLY valid JSON."}],
                options={"temperature": self.temperature},
                format="json",
            )

        content = resp.get("message", {}).get("content", "")

        try:
            model_cls.model_validate_json(content)
        except ValidationError:
            raise

        return content
