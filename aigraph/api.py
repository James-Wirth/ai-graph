################################
# AIGraph "API"...

# This file contains the implementation of our decorator-driven approach
# for constructing the graph of agents

# Hopefully it's helpful...
################################

from __future__ import annotations

import inspect

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, Literal, get_origin, get_args
from typing import Literal as TypingLiteral, Optional as TypingOptional

from pydantic import BaseModel, create_model

from aigraph.core.agents import Agent, LLMAgent
from aigraph.core.graphs import graph_from_agents
from aigraph.core.runner import GraphRunner
from aigraph.core.tools import ToolRegistry, ToolResult
from aigraph.core.viz import mermaid_from_graph, render_workflow_graph
from aigraph.interfaces.ollama import OllamaInterface

from aigraph.core.injection import FromPayload, FromVar, FromTool, Context, Payload, resolve_fn_args



class _RunScope:
    __slots__ = ("payload", "vars", "artifacts", "tool_registry", "app_cfg",
                 "llm_output", "node_returns_type")

    def __init__(self) -> None:
        self.payload: Any = None
        self.vars: Dict[str, Any] = {}
        self.artifacts: List[Dict[str, Any]] = []
        self.tool_registry: Optional[ToolRegistry] = None
        self.app_cfg: Optional["AppConfig"] = None
        self.llm_output: Any = None
        self.node_returns_type: Any = None

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


@dataclass
class Result:
    payload: Any
    artifacts: list[dict[str, Any]] | None = None


def _capture_sig(fn):
    specs = []
    sig = inspect.signature(fn)
    for p in sig.parameters.values():
        default = p.default if p.default is not inspect._empty else None
        specs.append((p.name, p.annotation, default))
    return specs


################################
# Node & Tool Metadata
################################

@dataclass
class ToolSpec:
    name: str
    required: bool = False
    argmap: Dict[str, Any] = field(default_factory=dict)
    alias: Optional[str] = None

class _AcceptLLM:
    pass

def accept_llm() -> _AcceptLLM:
    return _AcceptLLM()

def llm(typ: Optional[type[BaseModel]] = None):
    scope = _scope()
    obj = scope.llm_output
    if obj is None:
        return None
    T = typ or scope.node_returns_type
    if isinstance(T, type) and issubclass(T, BaseModel):
        return T.model_validate(obj)
    return obj

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
    param_specs: List[Tuple[str, Any]] = field(default_factory=list)

################################
# AppConfig simply defines the key top-level variables for our app
################################

class AppConfig(BaseModel):
    name: str = "App"
    model: str = "llama3"
    temperature: float = 0.1
    tools: List[str] = []  

_NODE_REGISTRY: Dict[str, NodeDef] = {}


def _func_name(f: Union[str, Callable, None]) -> Optional[str]:
    if f is None:
        return None
    return f if isinstance(f, str) else f.__name__

################################
# These classes define the decorators, i.e.

# @aigraph.start
# @aigraph.step
# @aigraph.route
# @aigraph.end
################################

class _StartDecorator:
    def __call__(self, fn: Optional[Callable] = None, *, next: Union[str, Callable, None] = None):
        def wrap(f: Callable) -> Callable:
            _NODE_REGISTRY[f.__name__] = NodeDef(
                func=f,
                kind="start",
                name=f.__name__,
                param_specs=_capture_sig(f),
                next_default=_func_name(next),   
            )
            _ensure_return_type(_NODE_REGISTRY[f.__name__], f)
            return f
        return wrap if fn is None else wrap(fn)

class _StepDecorator:
    def __call__(
        self,
        fn: Optional[Callable] = None,
        *,
        prompt: Optional[str] = None,
        tools: Optional[List[Union[str, Dict[str, Any], ToolSpec]]] = None,
        next: Optional[Union[str, Callable]] = None,
        schema_hint: Literal["json", "fields", "schema", "none"] = "json",
        returns: Optional[type[BaseModel]] = None,
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
            inferred_returns = returns
            ra = inspect.signature(f).return_annotation
            if inferred_returns is None and isinstance(ra, type) and issubclass(ra, BaseModel):
                inferred_returns = ra
            _NODE_REGISTRY[f.__name__] = NodeDef(
                func=f,
                kind="step",
                name=f.__name__,
                prompt=prompt,
                tools=_normalize(list(tools or [])),
                next_default=_func_name(next) if next else None,
                schema_hint=schema_hint,
                returns=inferred_returns,
                param_specs=_capture_sig(f),
            )
            _ensure_return_type(_NODE_REGISTRY[f.__name__], f)
            return f
        return wrap if fn is None else wrap(fn)

class _RouteDecorator:
    def __call__(
        self,
        fn: Optional[Callable] = None,
        *,
        prompt: str,
        cases: Optional[Mapping[str, Union[str, Callable]]] = None,
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
            ra = inspect.signature(f).return_annotation
            lit_opts = None
            if get_origin(ra) in {TypingLiteral, Literal}:
                lit_opts = tuple(map(str, get_args(ra)))
            
            if cases is None:
                if not lit_opts:
                    raise RuntimeError(
                        f"@route '{f.__name__}': provide either cases=... or a Literal[...] return type."
                    )
                cdict = {opt: opt for opt in lit_opts}
            else:
                cdict = {k: _func_name(v) for k, v in dict(cases).items()}
                if lit_opts and set(cdict.keys()) != set(lit_opts):
                    raise RuntimeError(
                        f"@route '{f.__name__}' cases {sorted(cdict)} must match return Literal {sorted(lit_opts)}"
                    )

            _NODE_REGISTRY[f.__name__] = NodeDef(
                func=f,
                kind="route",
                name=f.__name__,
                prompt=prompt,
                tools=_normalize(list(tools or [])),
                cases=cdict,
                route_selector=route_selector,
                schema_hint=schema_hint,
                param_specs=_capture_sig(f),
            )
            _ensure_return_type(_NODE_REGISTRY[f.__name__], f)
            return f
        return wrap if fn is None else wrap(fn)

class _EndDecorator:
    def __call__(self, fn: Callable) -> Callable:
        _NODE_REGISTRY[fn.__name__] = NodeDef(
            func=fn,
            kind="end",
            name=fn.__name__,
            param_specs=_capture_sig(fn),   
        )
        _ensure_return_type(_NODE_REGISTRY[fn.__name__], fn)
        return fn

    def handoff(self, fn: Callable) -> Callable:
        _NODE_REGISTRY[fn.__name__] = NodeDef(
            func=fn,
            kind="handoff",
            name=fn.__name__,
            param_specs=_capture_sig(fn),   
        )
        _ensure_return_type(_NODE_REGISTRY[fn.__name__], fn)
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
        self_app = self
        if not _NODE_REGISTRY:
            raise RuntimeError("No nodes declared. Use @ag.start/@ag.step/@ag.route/@ag.end decorators.")

        starts = [n for n in _NODE_REGISTRY.values() if n.kind == "start"]
        if len(starts) != 1:
            raise RuntimeError(f"Expected exactly one @ag.start node, found {len(starts)}.")
        start_def = starts[0]

        agents: Dict[str, Agent] = {}

        class _PayloadModel(BaseModel):
            payload: Any
            artifacts: list[dict[str, Any]] | None = None

        class _PyAgent(Agent):
            def __init__(self, node: NodeDef) -> None:
                super().__init__(node.name)
                self.node = node
                self._llm_agent = None
                if node.kind in {"start", "step", "end"} and node.prompt:
                    self._llm_agent = _mk_llm_agent(node)

            def process(self, input_model: BaseModel, context: Dict[str, Any]) -> BaseModel:
                _scope().payload = getattr(input_model, "payload", input_model)
                _scope().vars = context
                _scope().tool_registry = self_app._registry
                _scope().app_cfg = self_app.cfg
                _scope().artifacts = []
                _scope().llm_output = None
                _scope().node_returns_type = self.node.returns

                payload = _scope().payload
                tool_ns = context.get("tools") or {}
                args = resolve_fn_args(self.node.param_specs, payload, context, tool_ns)

                if self._llm_agent is not None and self.node.kind != "route":
                    prompt_text = self._llm_agent.build_prompt(input_model, context)
                    context.setdefault("_debug", {})["last_prompt"] = prompt_text
                    messages = self._llm_agent.build_messages(input_model, context)
                    
                    llm_obj = self._llm_agent._llm_round(messages)
                    parsed = getattr(llm_obj, "payload", llm_obj)
                    if isinstance(parsed, BaseModel):
                        parsed = parsed.model_dump()
                    _scope().llm_output = parsed

                rv = self.node.func(**args)

                from aigraph.api import _AcceptLLM  
                if isinstance(rv, Result):
                    payload_out = rv.payload; artifacts_out = rv.artifacts or []
                elif isinstance(rv, tuple) and len(rv) == 2:
                    payload_out, artifacts_out = rv
                elif isinstance(rv, _AcceptLLM):
                    payload_out, artifacts_out = _scope().llm_output, []
                else:
                    if rv is None:
                        raise RuntimeError(f"Node '{self.node.name}' returned None; explicit return required.")
                    payload_out, artifacts_out = rv, []
 

                scope_artifacts = _scope().artifacts or []
                all_artifacts = (artifacts_out or []) + scope_artifacts

                if self.node.returns and isinstance(self.node.returns, type) and issubclass(self.node.returns, BaseModel):
                    try:
                        payload_out = self.node.returns.model_validate(payload_out).model_dump()
                    except Exception as e:
                        raise TypeError(
                            f"Node '{self.node.name}' returned invalid payload for "
                            f"{self.node.returns.__name__}: {e}"
                        ) from e

                return _PayloadModel(payload=payload_out, artifacts=all_artifacts or None)

        def _mk_llm_agent(node: NodeDef) -> LLMAgent:
            if node.kind == "route":
                choices = tuple(node.cases.keys())
                if not choices:
                    raise RuntimeError(f"@route '{node.name}' has no cases; provide cases=... or a Literal[...] return type.")
                ChoiceLit = TypingLiteral[choices]
                RouteOut = create_model("RouteOutput", choice=(ChoiceLit, ...))  
                response_model = RouteOut
                edges = dict(node.cases)
                route_field = None if node.route_selector else "choice"
            else:
                if node.returns is None:
                    raise RuntimeError(f"[{node.name}] Missing return type; declare -> Model or returns=Model.")
                payload_ann, default = node.returns, ...
                
                StepOut = create_model("StepOutput", payload=(payload_ann, default))
                response_model = StepOut
                edges, route_field = {}, None

            adapter = OllamaInterface(model=self_app.cfg.model, temperature=self_app.cfg.temperature)
            allowed_names = sorted({s.name for s in node.tools}) or list(self_app.cfg.tools)

            def _prompt_builder(input_model: BaseModel, context: Dict[str, Any]) -> str:
                payload = getattr(input_model, "payload", input_model)
                tool_ns = (context.get("tools") or {})
                args_ns = resolve_fn_args(node.param_specs, payload, context, tool_ns)

                base_env = {
                    "input": payload,
                    "payload": payload,
                    "context": context,
                    "param": args_ns,
                    "tool": tool_ns,
                    **args_ns,  
                }

                from string import Template
                import json, re

                base = Template(node.prompt or "").safe_substitute(
                    input=json.dumps(payload, ensure_ascii=False, indent=2),
                    context=json.dumps(context, ensure_ascii=False, indent=2),
                )

                def get_path(obj, path):
                    cur = obj
                    for part in path.split("."):
                        if isinstance(cur, dict):
                            cur = cur.get(part)
                        else:
                            cur = getattr(cur, part, None)
                        if cur is None:
                            break
                    return cur

                def repl(m):
                    key = m.group(1)
                    val = get_path(base_env, key) if "." in key else base_env.get(key)
                    return "null" if val is None else json.dumps(val, ensure_ascii=False)

                return re.sub(r"\$\{([a-zA-Z_][\w\.]+)\}", repl, base)

            return LLMAgent(
                name=node.name,
                prompt_template=node.prompt or "You are an AI agent. Return JSON that conforms to the schema.",
                response_model=response_model,
                interface=adapter,
                edges=edges,
                route_field=route_field,
                route_selector=node.route_selector,
                schema_hint=node.schema_hint,
                prompt_builder=_prompt_builder,
            )

        for name, node in _NODE_REGISTRY.items():
            if node.kind == "route":
                agents[name] = _mk_llm_agent(node)          
            elif node.kind in {"start", "step", "end", "handoff"}:
                agents[name] = _PyAgent(node)                
            else:
                raise RuntimeError(f"Unknown node kind '{node.kind}' for node '{name}'.")

        G = graph_from_agents(agents, start=start_def.name)

        for node in _NODE_REGISTRY.values():
            if node.next_default:
                if node.next_default not in agents:
                    raise RuntimeError(f"@step(next=...) references unknown node '{node.next_default}'.")
                G.add_edge(node.name, node.next_default)

        for name, agent in agents.items():
            if isinstance(agent, LLMAgent) and getattr(agent, "edges", None):
                for _, target in agent.edges.items():
                    if target not in agents:
                        raise RuntimeError(
                            f"Route from '{name}' points to unknown node '{target}'. "
                            "Use a function name string that matches an @ag.step/@ag.end."
                        )

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
            self.param_specs = list(nd.param_specs)

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
    "FromPayload", "FromVar", "FromTool", "Context", "Payload", "Result",
    "accept_llm", "llm",
    "App", "AppConfig", "ToolSpec", "_reset_nodes_registry",
]


def _ensure_return_type(node: NodeDef, fn: Callable) -> None:
    ra = inspect.signature(fn).return_annotation
    has_ra = (ra is not inspect._empty)
    if node.kind == "route":
        ok = has_ra and (get_origin(ra) in {TypingLiteral, Literal})
        if not ok:
            raise RuntimeError(f"@route '{fn.__name__}' must declare Literal[...] return type.")
        return
    inferred = node.returns or (ra if has_ra else None)
    if inferred is None:
        raise RuntimeError(f"@{node.kind} '{fn.__name__}' must declare a return type "
                           f"(use -> MyModel or returns=MyModel).")
    node.returns = inferred