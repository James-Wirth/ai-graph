################################
# AIGraph "API"...

# This file contains the implementation of our decorator-driven approach
# for constructing the graph of agents

# Hopefully it's helpful...
################################

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, Literal

from pydantic import BaseModel, create_model

from aigraph.core.agents import Agent, LLMAgent
from aigraph.core.graphs import graph_from_agents
from aigraph.core.runner import GraphRunner
from aigraph.core.tools import ToolRegistry, ToolResult
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
        raise RuntimeError("No ToolRegistry in the run scope.")
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
    required: bool = False
    argmap: Dict[str, Any] = field(default_factory=dict)
    alias: Optional[str] = None

@dataclass
class NodeDef:
    func: Callable
    kind: Literal["start", "step", "route", "end", "handoff"]
    name: str
    prompt: Optional[str] = None
    tools: List[ToolSpec] = field(default_factory=list)
    next_default: Optional[str] = None
    cases: Dict[str, str] = field(default_factory=dict)
    route_selector: Optional[Callable[[Any, Dict[str, Any]], Optional[str]]] = None
    schema_hint: Literal["json", "fields", "schema", "none"] = "json"
    returns: Optional[type[BaseModel]] = None
    returns_required: bool = True

################################
# AppConfig simply defines the key top-level variables for our app
################################

class AppConfig(BaseModel):
    name: str = "App"
    model: str = "llama3"
    temperature: float = 0.1
    tools: List[str] = []  

_NODE_REGISTRY: Dict[str, NodeDef] = {}


def _func_name(f: Union[str, Callable]) -> str:
    return f if isinstance(f, str) else f.__name__

################################
# These classes define the decorators, i.e.

# @aigraph.start
# @aigraph.step
# @aigraph.route
# @aigraph.end
################################

class _StartDecorator:
    def __call__(self, fn: Callable) -> Callable:
        _NODE_REGISTRY[fn.__name__] = NodeDef(func=fn, kind="start", name=fn.__name__)
        return fn

class _StepDecorator:
    def __call__(
        self,
        fn: Optional[Callable] = None,
        *,
        prompt: Optional[str] = None,
        tools: Optional[List[Union[str, Dict[str, Any], ToolSpec]]] = None,
        next: Optional[Callable] = None,
        schema_hint: Literal["json", "fields", "schema", "none"] = "json",
        returns: Optional[type[BaseModel]] = None,
        returns_required: bool = True,
    ):
        def _normalize(ts: List[Union[str, Dict[str, Any], ToolSpec]]) -> List[ToolSpec]:
            out: List[ToolSpec] = []
            for t in ts:
                if isinstance(t, ToolSpec):
                    out.append(t)
                elif isinstance(t, str):
                    out.append(ToolSpec(name=t))
                elif isinstance(t, dict):
                    out.append(ToolSpec(**t))
                else:
                    raise TypeError(f"Unsupported tool spec: {t!r}")
            return out

        def wrap(f: Callable) -> Callable:
            _NODE_REGISTRY[f.__name__] = NodeDef(
                func=f,
                kind="step",
                name=f.__name__,
                prompt=prompt,
                tools=_normalize(list(tools or [])),
                next_default=_func_name(next) if next else None,
                schema_hint=schema_hint,
                returns=returns,
                returns_required=bool(returns_required),
            )
            return f
        return wrap if fn is None else wrap(fn)

class _RouteDecorator:
    def __call__(
        self,
        fn: Optional[Callable] = None,
        *,
        prompt: str,
        cases: Mapping[str, Union[str, Callable]],
        tools: Optional[List[Union[str, Dict[str, Any], ToolSpec]]] = None,
        route_selector: Optional[Callable[[Any, Dict[str, Any]], Optional[str]]] = None,
        schema_hint: Literal["json", "fields", "schema", "none"] = "json",
    ):
        def _normalize(ts: List[Union[str, Dict[str, Any], ToolSpec]]) -> List[ToolSpec]:
            out: List[ToolSpec] = []
            for t in ts:
                if isinstance(t, ToolSpec):
                    out.append(t)
                elif isinstance(t, str):
                    out.append(ToolSpec(name=t))
                elif isinstance(t, dict):
                    out.append(ToolSpec(**t))
                else:
                    raise TypeError(f"Unsupported tool spec: {t!r}")
            return out

        def wrap(f: Callable) -> Callable:
            _NODE_REGISTRY[f.__name__] = NodeDef(
                func=f,
                kind="route",
                name=f.__name__,
                prompt=prompt,
                tools=_normalize(list(tools or [])),
                cases={k: _func_name(v) for k, v in dict(cases).items()},
                route_selector=route_selector,
                schema_hint=schema_hint
            )
            return f
        return wrap if fn is None else wrap(fn)

class _EndDecorator:
    def __call__(self, fn: Callable) -> Callable:
        _NODE_REGISTRY[fn.__name__] = NodeDef(func=fn, kind="end", name=fn.__name__)
        return fn

    def handoff(self, fn: Callable) -> Callable:
        _NODE_REGISTRY[fn.__name__] = NodeDef(func=fn, kind="handoff", name=fn.__name__)
        return fn

start = _StartDecorator()
step = _StepDecorator()
route = _RouteDecorator()
end = _EndDecorator()


################################
# App is the main entrypoint that compiles the decorators into a real graph
# (by building each Agent instance and connecting them up)
################################

class App:
    def __init__(self, **kwargs) -> None:
        self.cfg = AppConfig(**kwargs)
        self._registry = ToolRegistry()
        self._compiled: Optional[Tuple[Any, str]] = None  

    def register_tool(self, tool) -> None:
        self._registry.register(tool)

    def _compile(self) -> Tuple[Any, str]:
        if not _NODE_REGISTRY:
            raise RuntimeError("No nodes declared. Use @ag.start/@ag.step/@ag.route/@ag.end decorators.")

        starts = [n for n in _NODE_REGISTRY.values() if n.kind == "start"]
        if len(starts) != 1:
            raise RuntimeError(f"Expected exactly one @ag.start node, found {len(starts)}.")
        start_def = starts[0]

        agents: Dict[str, Agent] = {}

        class _PayloadModel(BaseModel):
            payload: Any

        class _PyAgent(Agent):
            def __init__(self, node: NodeDef) -> None:
                super().__init__(node.name)
                self.node = node

            def process(self, input_model: BaseModel, context: Dict[str, Any]) -> BaseModel:
                _scope().payload = getattr(input_model, "payload", input_model)
                _scope().vars = context
                _scope().tool_registry = self_app._registry
                _scope().app_cfg = self_app.cfg
                self.node.func()
                return _PayloadModel(payload=_scope().payload)

        def _mk_llm_agent(node: NodeDef) -> LLMAgent:
            if node.kind == "route":
                from typing import Literal as TypingLiteral, Optional as TypingOptional
                choices = tuple(node.cases.keys())
                ChoiceLit = TypingLiteral[choices] if choices else str
                RouteOut = create_model("RouteOutput", choice=(TypingOptional[ChoiceLit], None))
                response_model = RouteOut
                edges = {k: v for k, v in node.cases.items()}
                route_field = None if node.route_selector else "choice"
            else:
                if node.returns is not None:
                    if node.returns_required:
                        payload_ann, default = node.returns, ...
                    else:
                        from typing import Optional as TypingOptional
                        payload_ann, default = TypingOptional[node.returns], None
                else:
                    from typing import Dict, Any, Optional as TypingOptional
                    payload_ann, default = TypingOptional[Dict[str, Any]], None

                StepOut = create_model("StepOutput", payload=(payload_ann, default))
                response_model = StepOut
                edges = {}
                route_field = None

            adapter = OllamaInterface(model=self_app.cfg.model, temperature=self_app.cfg.temperature)
            
            allowed_names = sorted({s.name for s in node.tools}) or list(self_app.cfg.tools)

            agent = LLMAgent(
                name=node.name,
                prompt_template=node.prompt or "You are an AI agent. Return JSON that conforms to the schema.",
                response_model=response_model,
                interface=adapter,
                edges=edges,
                route_field=route_field,
                allowed_tools=allowed_names,
                route_selector=node.route_selector,
                schema_hint=node.schema_hint
            )
            return agent

        self_app = self
        for name, node in _NODE_REGISTRY.items():
            if node.kind in {"start", "end", "handoff"}:
                agents[name] = _PyAgent(node)
            elif node.kind in {"step", "route"}:
                agents[name] = _mk_llm_agent(node)
            else:
                raise RuntimeError(f"Unknown node kind '{node.kind}' for node '{name}'.")

        G = graph_from_agents(agents, start=start_def.name)

        for node in _NODE_REGISTRY.values():
            if node.next_default:
                if node.next_default not in agents:
                    raise RuntimeError(f"@step(next=...) references unknown node '{node.next_default}'.")
                G.add_edge(node.name, node.next_default)

        for name, node in _NODE_REGISTRY.items():
            if name in G.nodes:
                G.nodes[name]["node_def"] = _to_runner_node(node)

        self._compiled = (G, start_def.name)
        return self._compiled

    def run(self, initial_payload: Any, *, stream: bool = False):
        G, start_node = self._compiled or self._compile()

        class _In(BaseModel):
            payload: Any

        runner = GraphRunner(G, tool_registry=self._registry, max_steps=32)
        out, ctx = runner.run(_In(payload=initial_payload), start_node)
        if isinstance(out, BaseModel):
            result = out.model_dump()
        else:
            result = out.payload if hasattr(out, "payload") else out
        return result, ctx

    def graph(self):
        G, start_node = self._compiled or self._compile()
        return _GraphView(G, start_node)

    def viz(self, *, history: Optional[List[Dict[str, Any]]] = None):
        G, start_node = self._compiled or self._compile()
        fig, _ = render_workflow_graph(G, start=start_node, history=history)
        class _Saver:
            def save(self, path: str) -> str:
                fig.savefig(path, bbox_inches="tight")
                return path
        return _Saver()

    def mermaid(self, *, history: Optional[List[Dict[str, Any]]] = None) -> str:
        G, start_node = self._compiled or self._compile()
        return mermaid_from_graph(G, start=start_node, history=history)

    def expect(self, **kwargs):
        class _Expect:
            def __init__(self, contains: Optional[List[str]] = None) -> None:
                self.contains = contains or []
            def check(self, text: str) -> bool:
                missing = [c for c in self.contains if c not in text]
                if missing:
                    raise AssertionError(f"Missing expected substrings: {missing}")
                return True
        return _Expect(**kwargs)

def _to_runner_node(node: NodeDef):
    class _RunnerNode:
        def __init__(self, nd: NodeDef):
            self.func = nd.func
            self.kind = nd.kind
            self.name = nd.name
            self.prompt = nd.prompt

            self.tools = []
            for s in nd.tools:
                d = {"name": s.name}
                if s.required:
                    d["required"] = True
                if s.argmap:
                    d["argmap"] = s.argmap
                if s.alias:
                    d["as"] = s.alias
                self.tools.append(d)
            self.cases = dict(nd.cases)
            self.route_selector = nd.route_selector
            self.next_default = nd.next_default
    rn = _RunnerNode(node)
    return rn

################################
# Small wrapper for nx.DiGraph
################################

class _GraphView:
    def __init__(self, G: Any, start: str) -> None:
        self.G = G
        self.start = start
    def show(self) -> Any:
        return self.G
    def override(self, node_fn: Callable, *, next: Callable):
        u, v = node_fn.__name__, next.__name__
        if not self.G.has_node(u) or not self.G.has_node(v):
            raise RuntimeError("Unknown nodes in override().")
        self.G.add_edge(u, v)
        return self

def _reset_nodes_registry() -> None:
    _NODE_REGISTRY.clear()

__all__ = [
    "start", "step", "route", "end",
    "data", "set_data", "vars", "tool", "emit",
    "App", "AppConfig", "ToolSpec", "_reset_nodes_registry",
]
