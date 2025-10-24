# aigraph/api.py
from __future__ import annotations
import inspect
import networkx as nx
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from aigraph.core.nodes import Node
from aigraph.core.runner import MessageRunner
from aigraph.core.messages import Message
from aigraph.core.context import Context
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


class Blueprint:
    def __init__(self, *, name: str | None = None, prefix: str = "") -> None:
        self.name = name or "blueprint"
        self.prefix = prefix.rstrip(":")
        self._nodes: Dict[str, Dict[str, Any]] = {}

    def node(
        self,
        name: str,
        *,
        consumes: Optional[List[str]] = None,
        emits: Optional[List[str]] = None,
    ):
        def _decorator(fn: Callable[[Message, Context], Optional[Message | List[Message]]]):
            nodename = f"{self.prefix}:{name}" if self.prefix else name
            if nodename in self._nodes:
                raise RuntimeError(
                    f"node '{nodename}' already registered on blueprint '{self.name}'."
                )
            self._nodes[nodename] = dict(func=fn, name=nodename, consumes=consumes, emits=emits)
            return fn

        return _decorator

    def __export__(self):
        return self._nodes


class App:
    def __init__(self, **kwargs) -> None:
        self.cfg = AppConfig(**kwargs)
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
                llm: Optional[LLMInterface],
                consumes_hint,
                emits_hint,
            ):
                consumes = list(consumes_hint or [name])
                super().__init__(name, consumes=consumes, emits=list(emits_hint or []))
                self._fn = fn
                self.ctx = Context(run_vars={}, llm=llm, cfg=cfg)

            def process(self, msg: Message, run_vars: Dict[str, Any]) -> List[Message]:
                self.ctx.run_vars = run_vars
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
                llm=self._llm_iface,
                consumes_hint=nd.consumes,
                emits_hint=nd.emits,
            )

        G = nx.DiGraph()
        for n in nodes.values():
            G.add_node(n.name, kind="node", node=n)
            for t in n.consumes:
                G.add_node(t, kind="msg")
                G.add_edge(t, n.name)
            for t in n.emits:
                G.add_node(t, kind="msg")
                G.add_edge(n.name, t)

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

    def include_blueprint(self, module: Any) -> "App":
        if not hasattr(module, "__export__"):
            raise TypeError("include_blueprint expects an object with __export__().")
        nodes_meta = module.__export__()

        for name, meta in nodes_meta.items():
            if name in self._nodes:
                raise RuntimeError(f"node '{name}' already registered on app '{self.cfg.name}'.")
            self._nodes[name] = NodeDef(
                func=meta["func"],
                name=meta["name"],
                consumes=meta.get("consumes"),
                emits=meta.get("emits"),
            )
        self._compiled_graph = None
        return self


__all__ = ["App", "Blueprint", "AppConfig", "Message"]
