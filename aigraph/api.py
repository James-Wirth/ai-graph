from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    Generic,
    TypeVar,
    get_origin,
    get_args,
    Literal,
)

from pydantic import BaseModel, ValidationError, create_model

from aigraph.core.agents import Agent
from aigraph.core.graphs import graph_from_agents
from aigraph.core.runner import GraphRunner
from aigraph.core.tools import ToolRegistry, FunctionTool
from aigraph.interfaces.ollama import OllamaInterface, LLMInterface

################################
# RunScope (defines the global context)
# You can access this stuff in the function body of each node
################################

from contextvars import ContextVar


class _RunScope:
    __slots__ = ("payload", "vars", "artifacts", "tool_registry", "app_cfg", "llm_iface")

    def __init__(self) -> None:
        self.payload: Any = None
        self.vars: Dict[str, Any] = {}
        self.artifacts: List[Dict[str, Any]] = []
        self.tool_registry: Optional[ToolRegistry] = None
        self.app_cfg: Optional["AppConfig"] = None
        self.llm_iface: Optional[LLMInterface] = None


_scope_var: ContextVar[_RunScope] = ContextVar("_aigraph_scope", default=_RunScope())


def _scope() -> _RunScope:
    return _scope_var.get()


def data() -> Any:
    return _scope().payload


def set_data(value: Any) -> None:
    _scope().payload = value


def vars() -> Dict[str, Any]:
    return _scope().vars


################################
# This is to let you register tools
# (that you call manually in the function body, for now...)
################################

_PENDING_TOOLS: List[FunctionTool] = []


def tool(*args, **kwargs):
    if args and isinstance(args[0], str):
        name = args[0]
        payload = dict(kwargs)
        reg = _scope().tool_registry
        if not reg:
            raise RuntimeError("No ToolRegistry available in this run scope.")
        return reg.get(name).call(payload)

    name = kwargs.pop("name", None)
    description = kwargs.pop("description", None)

    def _decorator(fn: Callable) -> FunctionTool:
        ft = FunctionTool(fn, name=name, description=description)
        _PENDING_TOOLS.append(ft)
        return ft

    if args and callable(args[0]):
        return _decorator(args[0])
    return _decorator


def emit(artifact: Dict[str, Any]) -> None:
    _scope().artifacts.append(artifact)


################################
# LLM returns
################################


def llm(
    *,
    model: Type[BaseModel],
    prompt: str,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseModel:
    sc = _scope()
    if sc.app_cfg is None:
        raise RuntimeError("Application config missing in scope; cannot call llm().")
    if sc.llm_iface is None:
        sc.llm_iface = OllamaInterface(
            model=sc.app_cfg.model, temperature=temperature or sc.app_cfg.temperature
        )

    messages: List[Dict[str, str]] = []
    messages.append(
        {
            "role": "system",
            "content": "You are a precise JSON generator. Return ONLY valid JSON for the schema. No extra text, no code fences.",
        }
    )
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    raw = sc.llm_iface.generate(messages=messages, response_model=model)
    try:
        return model.model_validate_json(raw)
    except ValidationError as e:
        raise TypeError(f"LLM returned invalid JSON for {model.__name__}: {e}") from e


################################
# Routing node returns
################################

T = TypeVar("T")


class Route(Generic[T]):
    """Return from a @route function: choose(case) or delegate(prompt=...)."""

    def __init__(self, case: Optional[T], prompt: Optional[str] = None):
        self.case = case
        self.prompt = prompt

    @classmethod
    def choose(cls, case: T) -> "Route[T]":
        return cls(case=case, prompt=None)

    @classmethod
    def delegate(cls, *, prompt: str) -> "Route[T]":
        return cls(case=None, prompt=prompt)


################################
# Node metadata
################################


@dataclass
class NodeDef:
    func: Callable
    kind: Literal["start", "step", "route", "end"]
    name: str
    next_default: Optional[str] = None
    cases: List[str] = None
    returns: Optional[Type[BaseModel]] = None


################################
# AppConfig (general high-level variables)
################################


class AppConfig(BaseModel):
    name: str = "App"
    model: str = "llama3"
    temperature: float = 0.1


################################
# Decorators
# (We love to hate them and hate to love them)
################################

_NODE_REGISTRY: Dict[str, NodeDef] = {}


def _func_name(f: Union[str, Callable, None]) -> Optional[str]:
    if f is None:
        return None
    return f if isinstance(f, str) else f.__name__


class _StartDecorator:
    def __call__(self, fn: Optional[Callable] = None, *, next: Union[str, Callable, None] = None):
        def wrap(f: Callable) -> Callable:
            _ensure_single_param(f, "@start")
            nd = NodeDef(func=f, kind="start", name=f.__name__, next_default=_func_name(next))
            _ensure_return_type(nd, f)
            _NODE_REGISTRY[f.__name__] = nd
            return f

        return wrap if fn is None else wrap(fn)


class _StepDecorator:
    def __call__(
        self, fn: Optional[Callable] = None, *, next: Optional[Union[str, Callable]] = None
    ):
        def wrap(f: Callable) -> Callable:
            _ensure_single_param(f, "@step")
            nd = NodeDef(func=f, kind="step", name=f.__name__, next_default=_func_name(next))
            _ensure_return_type(nd, f)
            _NODE_REGISTRY[f.__name__] = nd
            return f

        return wrap if fn is None else wrap(fn)


class _RouteDecorator:
    def __call__(self, fn: Optional[Callable] = None, *, cases: List[str]):
        def wrap(f: Callable) -> Callable:
            _ensure_single_param(f, "@route")
            nd = NodeDef(func=f, kind="route", name=f.__name__, cases=list(cases or []))
            _ensure_route_return_type(nd, f)
            if not nd.cases:
                raise RuntimeError(f"@route '{f.__name__}' requires non-empty cases=[...].")
            _NODE_REGISTRY[f.__name__] = nd
            return f

        return wrap if fn is None else wrap(fn)


class _EndDecorator:
    def __call__(self, fn: Callable) -> Callable:
        _ensure_single_param(fn, "@end")
        nd = NodeDef(func=fn, kind="end", name=fn.__name__)
        _ensure_return_type(nd, fn)
        _NODE_REGISTRY[fn.__name__] = nd
        return fn


start = _StartDecorator()
step = _StepDecorator()
route = _RouteDecorator()
end = _EndDecorator()

################################
# Result
# (Maybe it's useful for this to be here explicitly)
################################


@dataclass
class Result:
    payload: Any
    artifacts: List[Dict[str, Any]] | None = None


################################
# App class (compiles the graph and does all the hard stuff, basically)
################################


class App:
    def __init__(self, **kwargs) -> None:
        self.cfg = AppConfig(**kwargs)
        self._registry = ToolRegistry()
        self._compiled: Optional[Tuple[Any, str]] = None

    def register_tool(self, tool: FunctionTool) -> None:
        self._registry.register(tool)

    def _compile(self) -> Tuple[Any, str]:
        if not _NODE_REGISTRY:
            raise RuntimeError("No nodes declared. Use @ag.start/@ag.step/@ag.route/@ag.end.")

        for t in _PENDING_TOOLS:
            self._registry.register(t)

        starts = [n for n in _NODE_REGISTRY.values() if n.kind == "start"]
        if len(starts) != 1:
            raise RuntimeError(f"Expected exactly one @ag.start node, found {len(starts)}.")
        start_def = starts[0]

        agents: Dict[str, Agent] = {}

        class _PayloadModel(BaseModel):
            payload: Any
            artifacts: List[Dict[str, Any]] | None = None

        class _PyAgent(Agent):
            def __init__(self, node: NodeDef) -> None:
                super().__init__(node.name)
                self.node = node

            def process(self, input_model: BaseModel, context: Dict[str, Any]) -> BaseModel:
                _scope().payload = getattr(input_model, "payload", input_model)
                _scope().vars = context
                _scope().tool_registry = self_app._registry
                _scope().app_cfg = self_app.cfg
                _scope().artifacts = []

                raw_payload = _scope().payload
                sig = inspect.signature(self.node.func)
                if len(sig.parameters) != 1:
                    raise TypeError(
                        f"Node '{self.node.name}' must accept exactly one argument (payload)."
                    )

                ((param_name, param),) = sig.parameters.items()
                param_ann = param.annotation
                payload_for_func = raw_payload
                payload_for_pass = raw_payload

                if isinstance(param_ann, type) and issubclass(param_ann, BaseModel):
                    try:
                        coerced = param_ann.model_validate(raw_payload)
                        payload_for_func = coerced
                        payload_for_pass = coerced.model_dump()
                    except Exception as e:
                        raise TypeError(
                            f"Node '{self.node.name}' expected payload of type {param_ann.__name__} but got incompatible input: {e}"
                        ) from e

                rv = self.node.func(payload_for_func)
                if self.node.kind == "route":
                    choice = None
                    if isinstance(rv, Route):
                        if rv.case is not None:
                            choice = str(rv.case)
                        else:
                            prompt = rv.prompt or ""
                            import enum

                            ChoicesEnum = enum.Enum("ChoicesEnum", {c: c for c in self.node.cases})
                            RouteOut = create_model("RouteOut", choice=(ChoicesEnum, ...))
                            obj = llm(model=RouteOut, prompt=prompt)
                            v = getattr(obj, "choice", None)
                            choice = getattr(v, "value", None)
                    elif isinstance(rv, str):
                        choice = rv
                    else:
                        try:
                            import enum

                            if isinstance(rv, enum.Enum):
                                choice = str(rv.value)
                        except Exception:
                            pass

                    if choice not in self.node.cases:
                        raise ValueError(
                            f"Route '{self.node.name}' chose invalid case '{choice}'. Allowed: {sorted(self.node.cases)}"
                        )
                    context.setdefault("_route", {})[self.node.name] = choice

                    from pydantic import BaseModel as _BM

                    passthrough = payload_for_pass
                    if isinstance(passthrough, _BM):
                        passthrough = passthrough.model_dump()
                    return _PayloadModel(payload=passthrough, artifacts=_scope().artifacts or None)

                if isinstance(rv, Result):
                    payload_out = rv.payload
                    artifacts_out = rv.artifacts or []
                else:
                    payload_out = rv
                    artifacts_out = []

                if (
                    self.node.returns
                    and isinstance(self.node.returns, type)
                    and issubclass(self.node.returns, BaseModel)
                ):
                    try:
                        validated = self.node.returns.model_validate(payload_out)
                        payload_out = validated.model_dump()
                    except Exception as e:
                        raise TypeError(
                            f"Node '{self.node.name}' returned invalid payload for {self.node.returns.__name__}: {e}"
                        ) from e

                scope_artifacts = _scope().artifacts or []
                all_artifacts = (artifacts_out or []) + scope_artifacts

                return _PayloadModel(payload=payload_out, artifacts=all_artifacts or None)

            def route(
                self,
                neighbors: List[Any],
                context: Dict[str, Any],
                last_output: Optional[BaseModel],
            ):
                if self.node.kind != "route":
                    return super().route(neighbors, context, last_output)
                if not neighbors:
                    return None, "No neighbors", 1.0
                choice = (context.get("_route") or {}).get(self.node.name)
                if choice is None:
                    return None, "No route choice recorded", 0.1
                target = choice
                if target not in neighbors:
                    raise AssertionError(
                        f"[{self.node.name}] Route for '{choice}' points to non-neighbor '{target}'."
                    )
                return target, f"Route chose '{choice}'", 0.9

        self_app = self
        for name, node in _NODE_REGISTRY.items():
            agents[name] = _PyAgent(node)

        G = graph_from_agents(agents, start=start_def.name)

        for node in _NODE_REGISTRY.values():
            if node.next_default:
                if node.next_default not in agents:
                    raise RuntimeError(
                        f"@{node.kind}(next=...) references unknown node '{node.next_default}'."
                    )
                G.add_edge(node.name, node.next_default)
            if node.kind == "route":
                for target in node.cases or []:
                    if target not in agents:
                        raise RuntimeError(
                            f"@route '{node.name}' case '{target}' points to unknown node. "
                            "Cases must be step/end function names."
                        )
                    G.add_edge(node.name, target)

        for name, node in _NODE_REGISTRY.items():
            if name in G.nodes:
                G.nodes[name]["node_def"] = _to_runner_node(node)

        self._compiled = (G, start_def.name)
        return self._compiled

    def run(self, initial_payload: Any, *, stream: bool = False):
        G, start_node = self._compiled or self._compile()

        class _In(BaseModel):
            payload: Any

        runner = GraphRunner(G, tool_registry=self._registry, max_steps=64)
        out, ctx = runner.run(_In(payload=initial_payload), start_node)

        if isinstance(out, BaseModel):
            md = out.model_dump()
            result = md["payload"] if "payload" in md else md
        else:
            result = out.payload if hasattr(out, "payload") else out

        return result, ctx

    def graph(self):
        G, start_node = self._compiled or self._compile()
        return _GraphView(G, start_node)

    def viz(self, *, history: Optional[List[Dict[str, Any]]] = None):
        from aigraph.core.viz import render_workflow_graph

        G, start_node = self._compiled or self._compile()
        fig, _ = render_workflow_graph(G, start=start_node, history=history)

        class _Saver:
            def save(self, path: str) -> str:
                fig.savefig(path, bbox_inches="tight")
                return path

        return _Saver()

    def mermaid(self, *, history: Optional[List[Dict[str, Any]]] = None) -> str:
        from aigraph.core.viz import mermaid_from_graph

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


################################
# Runner node view
################################


def _to_runner_node(node: NodeDef):
    class _RunnerNode:
        def __init__(self, nd: NodeDef):
            self.func = nd.func
            self.kind = nd.kind
            self.name = nd.name
            self.next_default = nd.next_default
            self.cases = list(nd.cases or [])

    return _RunnerNode(node)


################################
# Graph view
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


################################
# Validation helpers
################################


def _ensure_single_param(fn: Callable, where: str):
    sig = inspect.signature(fn)
    if len(sig.parameters) != 1:
        raise TypeError(f"{where} '{fn.__name__}' must accept exactly one argument (payload).")


def _ensure_return_type(node: NodeDef, fn: Callable) -> None:
    ra = inspect.signature(fn).return_annotation
    if ra is inspect._empty:
        raise RuntimeError(f"@{node.kind} '{fn.__name__}' must declare a return type (-> MyModel).")
    if not (isinstance(ra, type) and issubclass(ra, BaseModel)):
        raise RuntimeError(
            f"@{node.kind} '{fn.__name__}' must return a Pydantic BaseModel (found {ra})."
        )
    node.returns = ra


def _ensure_route_return_type(node: NodeDef, fn: Callable) -> None:
    ra = inspect.signature(fn).return_annotation
    if ra is inspect._empty:
        raise RuntimeError(
            f"@route '{fn.__name__}' must declare return type Route[Literal[...]] or Route[Enum]."
        )
    origin = get_origin(ra)
    if origin is not Route:
        raise RuntimeError(f"@route '{fn.__name__}' must return Route[...], not {ra}.")
    args = get_args(ra) or []
    if args:
        inner = args[0]
        inner_origin = get_origin(inner)
        if inner_origin is Literal:
            allowed = set(map(str, get_args(inner)))
            if node.cases and not set(node.cases).issubset(allowed):
                missing = set(node.cases) - allowed
                raise RuntimeError(
                    f"@route '{fn.__name__}' cases {sorted(missing)} not listed in Route[Literal[...]]."
                )


__all__ = [
    "start",
    "step",
    "route",
    "end",
    "data",
    "set_data",
    "vars",
    "tool",
    "emit",
    "Result",
    "Route",
    "llm",
    "App",
    "AppConfig",
]
