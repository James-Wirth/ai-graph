# aigraph/core/runner.py
from __future__ import annotations
import datetime as dt
import logging
import networkx as nx
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

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


def _due(m: Message) -> bool:
    at = m.headers.get("deliver_at")
    if not at:
        return True
    try:
        return dt.datetime.now(dt.UTC) >= dt.datetime.fromisoformat(at)
    except Exception:
        return True


class MessageRunner:
    def __init__(self, graph: nx.DiGraph, max_steps: int = 64):
        self.graph = graph
        self.max_steps = max_steps
        self.logger = logging.getLogger("aigraph.runner")

    def _log_event(self, ev: Dict[str, Any], outs: List[Message]) -> None:
        self.logger.debug(
            "[%s] %s :: %s -> %s",
            ev.get("timestamp"),
            ev.get("node") or "NO-NODE",
            ev.get("consumed"),
            ",".join(ev.get("emitted") or []),
        )
        for m in outs:
            corr = m.headers.get("correlation_id", m.id)
            self.logger.debug(
                "  emitted %s id=%s corr=%s parent=%s",
                m.type,
                m.id,
                corr,
                m.headers.get("parent_id"),
            )

    def _build_subscriptions(self) -> Dict[str, List[Any]]:
        subs: Dict[str, List[Any]] = defaultdict(list)
        for _, data in self.graph.nodes(data=True):
            node_obj = data.get("node")
            if node_obj is None:
                continue
            for t in getattr(node_obj, "consumes", []) or []:
                subs[t].append(node_obj)
        return subs

    def _merge_artifacts(self, outs: List[Message], node_obj: Any) -> List[Message]:
        ctx = getattr(node_obj, "ctx", None)
        if not ctx or not getattr(ctx, "artifacts", None):
            return outs
        merged: List[Message] = []
        artifacts_payload = [getattr(a, "__dict__", a) for a in ctx.artifacts]
        for om in outs:
            headers = dict(om.headers)
            prev = headers.get("artifacts", [])
            headers["artifacts"] = list(prev) + artifacts_payload
            merged.append(Message(type=om.type, body=om.body, headers=headers))
        ctx.artifacts.clear()
        return merged

    def _correlate(self, parent: Message, outs: List[Message]) -> List[Message]:
        corr = parent.headers.get("correlation_id", parent.id)
        fixed: List[Message] = []
        for om in outs:
            if om.headers.get("correlation_id"):
                fixed.append(om)
            else:
                fixed.append(om.with_header(correlation_id=corr, parent_id=parent.id))
        return fixed

    def run(self, initial_messages: List[Message]) -> Tuple[List[Message], ExecutionContext]:
        gen = self.run_iter(initial_messages)
        emitted: List[Message] = []
        ctx: Optional[ExecutionContext] = None
        try:
            while True:
                _ = next(gen)
        except StopIteration as stop:
            emitted, ctx = stop.value
        return emitted, ctx

    def run_iter(
        self, initial_messages: List[Message]
    ) -> Generator[BusEvent, None, Tuple[List[Message], ExecutionContext]]:
        ctx = ExecutionContext()
        subs = self._build_subscriptions()

        queue: deque[Message] = deque(initial_messages)
        seen: set[str] = set()
        emitted_all: List[Message] = []
        steps = 0

        def _record_and_yield(
            node_name: Optional[str],
            consumed: Optional[str],
            outs: List[Message],
            details: str = "",
        ) -> BusEvent:
            ev = {
                "timestamp": dt.datetime.now(dt.UTC).isoformat(),
                "node": node_name,
                "consumed": consumed,
                "emitted": [m.type for m in outs],
                "details": details,
            }
            ctx.record_bus_event(
                node=node_name, consumed=consumed, emitted=ev["emitted"], details=details
            )
            try:
                self._log_event(ev, outs)
            except Exception:
                pass
            return BusEvent(
                timestamp=ev["timestamp"],
                node=node_name,
                consumed=consumed,
                emitted_types=ev["emitted"],
                emitted_msgs=outs,
                details=details,
            )

        try:
            while queue and steps < self.max_steps:
                msg = queue[0]
                if not _due(msg):
                    queue.rotate(-1)
                    continue

                msg = queue.popleft()
                if msg.id in seen:
                    continue
                seen.add(msg.id)

                targets = list(subs.get(msg.type, []))
                if not targets:
                    ev = _record_and_yield(None, msg.type, [], "no-subscriber")
                    emitted_all.append(msg)
                    steps += 1
                    yield ev
                    continue

                for node_obj in targets:
                    try:
                        outs = node_obj.process(msg, ctx.variables) or []
                        if isinstance(outs, Message):
                            outs = [outs]
                        elif not isinstance(outs, list) or not all(
                            isinstance(x, Message) for x in outs
                        ):
                            raise TypeError(
                                f"Node '{getattr(node_obj, 'name', 'unknown')}' "
                                "must return Message, list[Message], or None."
                            )
                    except Exception as e:
                        outs = [
                            msg.child(
                                type="error.v1",
                                body={
                                    "node": getattr(node_obj, "name", None),
                                    "for": msg.type,
                                    "error": str(e),
                                },
                            )
                        ]

                    outs = self._correlate(msg, outs)
                    outs = self._merge_artifacts(outs, node_obj)
                    for om in outs:
                        queue.append(om)

                    ev = _record_and_yield(getattr(node_obj, "name", None), msg.type, outs, "")
                    emitted_all.extend(outs)
                    steps += 1
                    yield ev
                    if steps >= self.max_steps:
                        break
        finally:
            return emitted_all, ctx
