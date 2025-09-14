# aigraph/core/runner.py
from __future__ import annotations
import datetime as dt
import logging
import networkx as nx
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

from aigraph.core.bus import MessageBus
from aigraph.core.messages import Message


class ExecutionContext:
    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or str(uuid.uuid4())
        self.history: List[Dict[str, Any]] = []
        self.variables: Dict[str, Any] = {}
        self.created_at = dt.datetime.now(dt.UTC)

    def record_bus_event(
        self,
        *,
        node: Optional[str],
        consumed: Optional[str],
        emitted: List[str],
        details: str = "",
    ):
        self.history.append(
            {
                "timestamp": dt.datetime.now(dt.UTC).isoformat(),
                "node": node,
                "consumed": consumed,
                "emitted": emitted,
                "details": details,
            }
        )


@dataclass
class BusEvent:
    timestamp: str
    node: Optional[str]
    consumed: Optional[str]
    emitted_types: List[str]
    emitted_msgs: List[Message]
    details: str = ""


class MessageRunner:
    def __init__(self, graph: nx.DiGraph, max_steps: int = 64):
        self.graph = graph
        self.max_steps = max_steps
        self.logger = logging.getLogger("aigraph.runner")

    def run(self, initial_messages: List[Message]) -> Tuple[List[Message], ExecutionContext]:
        ctx = ExecutionContext()
        bus = MessageBus(max_steps=self.max_steps)

        for _, data in self.graph.nodes(data=True):
            if data.get("kind") == "node":
                n = data["node"]
                for t in n.consumes:
                    bus.subscribe(t, n)

        for m in initial_messages:
            bus.publish(m)

        emitted, history = bus.run(context=ctx.variables)
        for ev in history:
            ctx.record_bus_event(
                node=ev.get("node"),
                consumed=ev.get("consumed"),
                emitted=ev.get("emitted") or [],
                details=ev.get("details", ""),
            )
        return emitted, ctx

    def run_iter(
        self, initial_messages: List[Message]
    ) -> Generator[BusEvent, None, Tuple[List[Message], ExecutionContext]]:
        ctx = ExecutionContext()
        bus = MessageBus(max_steps=self.max_steps)

        for _, data in self.graph.nodes(data=True):
            if data.get("kind") == "node":
                n = data["node"]
                for t in n.consumes:
                    bus.subscribe(t, n)

        for m in initial_messages:
            bus.publish(m)

        try:
            for out, ev in bus.iter_run(context=ctx.variables):
                ctx.record_bus_event(
                    node=ev.get("node"),
                    consumed=ev.get("consumed"),
                    emitted=ev.get("emitted") or [],
                    details=ev.get("details", ""),
                )
                yield BusEvent(
                    timestamp=ev["timestamp"],
                    node=ev["node"],
                    consumed=ev["consumed"],
                    emitted_types=ev["emitted"],
                    emitted_msgs=out,
                    details=ev.get("details", ""),
                )
        except StopIteration as stop:
            emitted, _history = stop.value
            return emitted, ctx
