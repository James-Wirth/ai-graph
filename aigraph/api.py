from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, Literal

from pydantic import BaseModel, Field

from aigraph.core.agents import Agent, LLMAgent
from aigraph.core.graphs import graph_from_agents
from aigraph.core.runner import GraphRunner
from aigraph.core.tools import ToolCall, ToolRegistry, ToolResult
from aigraph.core.viz import mermaid_from_graph, render_workflow_graph
from aigraph.interfaces.ollama import OllamaInterface


class _RunScope:
    __slots__ = ("payload", "vars", "artifacts", "tool_registry", "app_cfg")

    def __init__(self) -> None:
        self.payload: Any = None
        self.vars: Dict[str, Any] = {}
        self.artifacts: List[Dict[str, Any]] = []
        self.tool_registry: Optional[ToolRegistry] = None
        self.app_cfg: Optional["AppConfig"] = None

_scope_var: ContextVar[_RunScope] = ContextVar("_aigraph_scope", default=_RunScope())

def _scope() -> _RunScope:
    return _scope_var.get()

def data() -> Any:
    return _scope().payload

def set_data(value: Any) -> None:
    _scope().payload = value

def vars() -> Dict[str, Any]:
    return _scope().vars

def tool(name: str, **kwargs) -> ToolResult:
    reg = _scope().tool_registry
    if not reg:
        raise RuntimeError("No ToolRegistry bound to run scope (tools unavailable in this context).")
    t = reg.get(name)
    return t.call(kwargs)

def emit(artifact: Dict[str, Any]) -> None:
    _scope().artifacts.append(artifact)

################################
# Node & Tool Metadata
################################

@dataclass
class ToolSpec:
    name: str
    mode: Literal["pre", "interactive"] = "interactive"  
    required: bool = False
    argmap: Dict[str, Any] = field(default_factory=dict)
    alias: Optional[str] = None
    max_calls: Optional[int] = None

@dataclass
class NodeDef:
    func: Callable
    kind: Literal["start", "step", "route", "end", "handoff"]
    name: str
    prompt: Optional[str] = None
    tools: List[ToolSpec] = field(default_factory=list)
    rounds: int = 0  
    next_default: Optional[str] = None
    cases: Dict[str, str] = field(default_factory=dict)
    route_selector: Optional[Callable[[Any, Dict[str, Any]], Optional[str]]] = None


################################
# AppConfig simply defines the key top-level variables for our app
################################

class AppConfig(BaseModel):
    name: str = "App"
    model: str = "llama3"
    temperature: float = 0.1
    tools: List[str] = []  