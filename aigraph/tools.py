import logging

from typing import Any, Dict, Optional
from pydantic import BaseModel


class ToolResult(BaseModel):
    name: str
    input: Dict[str, Any]
    output: Any
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class Tool:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"aigraph.tool.{self.name}")

    def call(self, input: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError


class EchoTool(Tool):
    def call(self, input: Dict[str, Any]) -> ToolResult:
        self.logger.info("EchoTool called with: %s", input)
        return ToolResult(name=self.name, input=input, output=input, success=True)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self._tools[name]
