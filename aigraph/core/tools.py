import logging
import inspect

from typing import Any, Dict, Optional, Callable
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)

class ToolResult(BaseModel):
    name: str
    input: Dict[str, Any]
    output: Any
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Tool:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"aigraph.tool.{self.name}")

    def call(self, input: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError
    

class FunctionTool(Tool):
    def __init__(self, fn: Callable, name: Optional[str] = None, description: Optional[str] = None):
        super().__init__(name or fn.__name__)
        self.fn = fn
        self.description = (description or fn.__doc__ or "").strip()
        self._signature = inspect.signature(fn)

    def call(self, input: Dict[str, Any]) -> ToolResult:
        payload = input or {}
        try:
            bound = self._signature.bind_partial(**payload)
            bound.apply_defaults()
            out = self.fn(*bound.args, **bound.kwargs)
            return ToolResult(name=self.name, input=payload, output=out, success=True)
        except TypeError as e:
            return ToolResult(name=self.name, input=payload, output=None, success=False, error=f"Bad arguments: {e}")
        except Exception as e:
            self.logger.exception("Tool '%s' crashed", self.name)
            return ToolResult(name=self.name, input=payload, output=None, success=False, error=str(e))


def tool(fn: Optional[Callable] = None, name: Optional[str] = None, description: Optional[str] = None):
    def wrap(_fn: Callable) -> FunctionTool:
        return FunctionTool(_fn, name=name, description=description)
    return wrap(fn) if fn else wrap


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self._tools[name]
