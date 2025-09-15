# aigraph/api.py
from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from aigraph.core.nodes import Node
from aigraph.core.graphs import graph_from_nodes
from aigraph.core.runner import MessageRunner
from aigraph.core.tools import ToolRegistry, FunctionTool, Tool
from aigraph.core.messages import Message
from aigraph.core.context import Context, Inventory
from aigraph.interfaces.ollama import LLMInterface


@dataclass
class NodeDef:
    func: Callable[[Message, Context], Optional[Message | List[Message]]]
    name: str
    consumes: List[str] | None = None
    emits: List[str] | None = None


class AppConfig(BaseModel):
    name: str = "App"
    model: str = "llama3"
    temperature: float = 0.1


def _ensure_sig_msg_ctx(fn: Callable, where: str):
    sig = inspect.signature(fn)
    if list(sig.parameters.keys()) != ["msg", "ctx"]:
        raise TypeError(f"{where} '{fn.__name__}' must accept (msg: Message, ctx: Context).")


class App:
    def __init__(self, **kwargs) -> None:
        self.cfg = AppConfig(**kwargs)
        self._registry = ToolRegistry()
        self._compiled_graph: Optional[Any] = None
        self._nodes: Dict[str, NodeDef] = {}
        self._llm_iface: Optional[LLMInterface] = None

    def with_llm(self, llm_iface: LLMInterface) -> "App":
        self._llm_iface = llm_iface
        return self

    def node(
        self, name: str, *, consumes: Optional[List[str]] = None, emits: Optional[List[str]] = None
    ):
        def _decorator(fn: Callable[[Message, Context], Optional[Message | List[Message]]]):
            _ensure_sig_msg_ctx(fn, "@node")
            if name in self._nodes:
                raise RuntimeError(f"node '{name}' already registered on app '{self.cfg.name}'.")
            self._nodes[name] = NodeDef(func=fn, name=name, consumes=consumes, emits=emits)
            self._compiled_graph = None
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
            self._compiled_graph = None
            return f

        return _wrap if fn is None else _wrap(fn)

    def register_tool(self, tool: Tool) -> None:
        self._registry.register(tool)

    def _compile(self):
        if not self._nodes:
            raise RuntimeError("No nodes declared. Use @app.node('name').")

        class _PyNode(Node):
            def __init__(
                self,
                name,
                fn,
                *,
                cfg: AppConfig,
                tools: ToolRegistry,
                llm: Optional[LLMInterface],
                consumes_hint,
                emits_hint,
            ):
                consumes = list(consumes_hint or [name])
                super().__init__(name, consumes=consumes, emits=list(emits_hint or []))
                self._fn = fn
                self.ctx = Context(
                    run_vars={}, inventory=Inventory(), tools=tools, llm=llm, cfg=cfg
                )

            def process(self, msg: Message, run_vars: Dict[str, Any]) -> List[Message]:
                self.ctx.run_vars = run_vars
                self.ctx.artifacts.clear()
                out = self._fn(msg, self.ctx)
                if out is None:
                    return []
                if isinstance(out, Message):
                    return [out]
                if isinstance(out, list) and all(isinstance(x, Message) for x in out):
                    return out
                raise TypeError(f"Node '{self.name}' must return Message, list[Message], or None.")

        nodes: Dict[str, Node] = {}
        for name, nd in self._nodes.items():
            nodes[name] = _PyNode(
                name,
                nd.func,
                cfg=self.cfg,
                tools=self._registry,
                llm=self._llm_iface,
                consumes_hint=nd.consumes,
                emits_hint=nd.emits,
            )

        G = graph_from_nodes(nodes)

        for name, n in nodes.items():
            if name in G.nodes:
                G.nodes[name]["kind"] = "node"
                G.nodes[name]["node"] = n
                G.nodes[name]["node_def"] = self._nodes[name]
                G.nodes[name]["consumes"] = list(getattr(n, "consumes", []))
                G.nodes[name]["emits"] = list(getattr(n, "emits", []))

        self._compiled_graph = G
        return G

    def _build_seeds(
        self, initial_payload: Any | List[Any] | None, seed_types: List[str] | None
    ) -> List[Message]:
        if initial_payload is None:
            return []

        def default_seed_type() -> str:
            return next(iter(self._nodes.keys()))

        if isinstance(initial_payload, list):
            if seed_types and len(seed_types) == len(initial_payload):
                return [Message(type=t, body=b) for t, b in zip(seed_types, initial_payload)]
            t = seed_types[0] if seed_types else default_seed_type()
            return [Message(type=t, body=b) for b in initial_payload]
        t = seed_types[0] if seed_types else default_seed_type()
        return [Message(type=t, body=initial_payload)]

    def run(
        self, initial_payload: Any | List[Any] | None = None, *, seed_types: List[str] | None = None
    ):
        G = self._compiled_graph or self._compile()
        runner = MessageRunner(G, max_steps=64)
        seeds = self._build_seeds(initial_payload, seed_types)
        return runner.run(seeds)

    def run_iter(
        self, initial_payload: Any | List[Any] | None = None, *, seed_types: List[str] | None = None
    ):
        G = self._compiled_graph or self._compile()
        runner = MessageRunner(G, max_steps=64)
        seeds = self._build_seeds(initial_payload, seed_types)
        return runner.run_iter(seeds)

    def graph(self):
        return self._compiled_graph or self._compile()

    def viz(self, *, history: Optional[List[Dict[str, Any]]] = None, observed_only: bool = False):
        from aigraph.core.viz import render_workflow_graph

        G = self._compiled_graph or self._compile()
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

        G = self._compiled_graph or self._compile()
        return mermaid_from_graph(G, history=history, observed_only=observed_only)


__all__ = ["App", "AppConfig", "Message"]
