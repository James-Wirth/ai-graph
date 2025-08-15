from typing import Any, Dict

try:
    from ollama import chat
except ImportError:  
    chat = None


class LLMInterface:
    def generate(self, prompt: str, response_schema: Dict[str, Any], **kwargs) -> str:
        raise NotImplementedError


class OllamaInterface(LLMInterface):
    def __init__(self, model: str = "llama3", temperature: float = 0.0):
        if chat is None:
            raise RuntimeError("Couldn't import ollama.chat")
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, response_schema: Dict[str, Any], **kwargs) -> str:
        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=response_schema,
            options={"temperature": self.temperature},
        )
        return response.message.content
