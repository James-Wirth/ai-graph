# aigraph/api.py
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

from aigraph.core.nodes import Node
from aigraph.core.graphs import graph_from_nodes
from aigraph.core.runner import MessageRunner
from aigraph.core.tools import ToolRegistry, FunctionTool, Tool
from aigraph.interfaces.ollama import OllamaInterface, LLMInterface
from aigraph.core.messages import Message

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


def call_tool(*args, **kwargs):
    if args and isinstance(args[0], str):
        name = args[0]
        payload = dict(kwargs)
        reg = _scope().tool_registry
        if not reg:
            raise RuntimeError("No ToolRegistry available in this run scope.")
        return reg.get(name).call(payload)
    raise RuntimeError(
        "Decorator-style @ag.tool is no longer supported. Use @app.tool on a specific App instance."
    )


def emit(artifact: Dict[str, Any]) -> None:
    _scope().artifacts.append(artifact)


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
        temp = sc.app_cfg.temperature if temperature is None else temperature
        sc.llm_iface = OllamaInterface(model=sc.app_cfg.model, temperature=temp)

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


@dataclass
class NodeDef:
    func: Callable[[Message], Optional[Message | List[Message]]]
    name: str
    consumes: List[str] | None = None
    emits: List[str] | None = None


class AppConfig(BaseModel):
    name: str = "App"
    model: str = "llama3"
    temperature: float = 0.1


def _ensure_single_param(fn: Callable, where: str):
    sig = inspect.signature(fn)
    if len(sig.parameters) != 1:
        raise TypeError(f"{where} '{fn.__name__}' must accept exactly one argument (Message).")


class App:
    def __init__(self, **kwargs) -> None:
        self.cfg = AppConfig(**kwargs)
        self._registry = ToolRegistry()
        self._compiled: Optional[Tuple[Any, Dict[str, Node]]] = None
        self._nodes: Dict[str, NodeDef] = {}

    def _register_node(self, nd: NodeDef) -> None:
        if nd.name in self._nodes:
            raise RuntimeError(f"node '{nd.name}' already registered on app '{self.cfg.name}'.")
        self._nodes[nd.name] = nd
        self._compiled = None

    def node(
        self, name: str, *, consumes: Optional[List[str]] = None, emits: Optional[List[str]] = None
    ):
        def _decorator(fn: Callable[[Message], Optional[Message | List[Message]]]) -> Callable:
            _ensure_single_param(fn, "@node")
            self._register_node(NodeDef(func=fn, name=name, consumes=consumes, emits=emits))
            return fn

        return _decorator

    def tool(
        self,
        fn: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        def _wrap(f: Callable):
            ft = FunctionTool(f, name=name, description=description)
            self._registry.register(ft)
            self._compiled = None
            return f

        return _wrap if fn is None else _wrap(fn)

    def register_tool(self, tool: Tool) -> None:
        self._registry.register(tool)

    def _compile(self):
        if not self._nodes:
            raise RuntimeError("No nodes declared. Use @app.node('name').")

        nodes: Dict[str, Node] = {}

        class _PyNode(Node):
            def __init__(
                self, name, fn, *, cfg: AppConfig, tools: ToolRegistry, consumes_hint, emits_hint
            ):
                consumes = list(consumes_hint or [name])
                super().__init__(name, consumes=consumes, emits=list(emits_hint or []))
                self._fn = fn
                self.cfg = cfg
                self.tools = tools

            def _normalize_out(self, rv: Any) -> List[Message]:
                if rv is None:
                    return []
                if isinstance(rv, Message):
                    outs = [rv]
                elif isinstance(rv, list) and all(isinstance(x, Message) for x in rv):
                    outs = rv
                else:
                    raise TypeError(
                        f"Node '{self.name}' must return Message, list[Message], or None."
                    )

                sc = _scope()
                normed: List[Message] = []
                for m in outs:
                    headers = dict(getattr(m, "headers", {}) or {})
                    if sc.artifacts:
                        prev = headers.get("artifacts", [])
                        headers["artifacts"] = list(prev) + list(sc.artifacts)
                    normed.append(Message(type=m.type, body=m.body, headers=headers))
                return normed

            def process(self, msg: Message, context: Dict[str, Any]) -> List[Message]:
                sc = _scope()
                sc.payload = msg.body
                sc.vars = context
                sc.tool_registry = self.tools
                sc.app_cfg = self.cfg
                sc.artifacts = []

                rv = self._fn(msg)
                return self._normalize_out(rv)

        for name, nd in self._nodes.items():
            nodes[name] = _PyNode(
                name,
                nd.func,
                cfg=self.cfg,
                tools=self._registry,
                consumes_hint=nd.consumes,
                emits_hint=nd.emits,
            )

        G = graph_from_nodes(nodes)

        for name, n in nodes.items():
            if name in G.nodes:
                G.nodes[name]["node_def"] = self._nodes[name]
                G.nodes[name]["consumes"] = list(getattr(n, "consumes", []))
                G.nodes[name]["emits"] = list(getattr(n, "emits", []))

        self._compiled = (G, nodes)
        return self._compiled

    def run(
        self,
        initial_payload: Any | List[Any] | None = None,
        *,
        seed_types: List[str] | None = None,
    ):
        G, _ = self._compiled or self._compile()
        runner = MessageRunner(G, max_steps=64)

        seeds = self._build_seeds(initial_payload, seed_types)
        return runner.run(seeds)

    def run_iter(
        self,
        initial_payload: Any | List[Any] | None = None,
        *,
        seed_types: List[str] | None = None,
    ):
        G, _ = self._compiled or self._compile()
        runner = MessageRunner(G, max_steps=64)

        seeds = self._build_seeds(initial_payload, seed_types)
        return runner.run_iter(seeds)

    def _build_seeds(
        self, initial_payload: Any | List[Any] | None, seed_types: List[str] | None
    ) -> List[Message]:
        def default_seed_type() -> str:
            return next(iter(self._nodes.keys()))

        if initial_payload is None:
            return []
        if isinstance(initial_payload, list):
            if seed_types and len(seed_types) == len(initial_payload):
                return [Message(type=t, body=b) for t, b in zip(seed_types, initial_payload)]
            t = seed_types[0] if seed_types else default_seed_type()
            return [Message(type=t, body=b) for b in initial_payload]
        t = seed_types[0] if seed_types else default_seed_type()
        return [Message(type=t, body=initial_payload)]

    def graph(self):
        G, _ = self._compiled or self._compile()
        return _GraphView(G)

    def viz(self, *, history: Optional[List[Dict[str, Any]]] = None, observed_only: bool = False):
        from aigraph.core.viz import render_workflow_graph

        G, _ = self._compiled or self._compile()
        fig, _ = render_workflow_graph(G, history=history, observed_only=observed_only)

        class _Saver:
            def save(self, path: str) -> str:
                fig.savefig(path, bbox_inches="tight")
                return path

        return _Saver()

    def mermaid(
        self, *, history: Optional[List[Dict[str, Any]]] = None, observed_only: bool = True
    ) -> str:
        from aigraph.core.viz import mermaid_from_graph

        G, _ = self._compiled or self._compile()
        return mermaid_from_graph(G, history=history, observed_only=observed_only)


class _GraphView:
    def __init__(self, G: Any) -> None:
        self.G = G

    def show(self) -> Any:
        return self.G


__all__ = ["data", "vars", "call_tool", "emit", "llm", "App", "AppConfig", "Message"]
