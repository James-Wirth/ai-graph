# aigraph/core/bus.py
from __future__ import annotations
import datetime as dt
import logging
from collections import deque, defaultdict
from typing import Dict, Generator, List, Tuple
from aigraph.core.messages import Message
from aigraph.core.nodes import Node


def _due(m: Message) -> bool:
    at = m.headers.get("deliver_at")
    if not at:
        return True
    try:
        return dt.datetime.now(dt.UTC) >= dt.datetime.fromisoformat(at)
    except Exception:
        return True


class MessageBus:
    def __init__(self, *, max_steps: int = 64):
        self.max_steps = max_steps
        self.logger = logging.getLogger("aigraph.bus")
        self.queue: deque[Message] = deque()
        self.seen: set[str] = set()
        self.subs: Dict[str, List[Node]] = defaultdict(list)

    def subscribe(self, typ: str, node: Node) -> None:
        self.subs[typ].append(node)

    def publish(self, *msgs: Message) -> None:
        for m in msgs:
            if m.id in self.seen:
                continue
            self.queue.append(m)

    def run(self, *, context: Dict[str, any]) -> Tuple[List[Message], List[Dict[str, any]]]:
        history: List[Dict[str, any]] = []
        emitted: List[Message] = []
        steps = 0

        while self.queue and steps < self.max_steps:
            msg = self.queue[0]
            if not _due(msg):
                self.queue.rotate(-1)
                continue

            msg = self.queue.popleft()
            if msg.id in self.seen:
                continue
            self.seen.add(msg.id)

            nodes = list(self.subs.get(msg.type, []))
            if not nodes:
                ev = {
                    "timestamp": msg.ts.isoformat(),
                    "node": None,
                    "consumed": msg.type,
                    "emitted": [],
                    "details": "no-subscriber",
                }
                history.append(ev)
                emitted.append(msg)
                steps += 1
                continue

            for n in nodes:
                try:
                    out = n.process(msg, context) or []
                except Exception as e:
                    out = [
                        msg.child(
                            type="error.v1",
                            body={"node": n.name, "for": msg.type, "error": str(e)},
                        )
                    ]

                corr = msg.headers.get("correlation_id", msg.id)
                out = [
                    (
                        om
                        if om.headers.get("correlation_id")
                        else om.with_header(correlation_id=corr, parent_id=msg.id)
                    )
                    for om in out
                ]

                self.publish(*out)
                ev = {
                    "timestamp": msg.ts.isoformat(),
                    "node": n.name,
                    "consumed": msg.type,
                    "emitted": [m.type for m in out],
                    "details": "",
                }
                history.append(ev)
                emitted.extend(out)
                steps += 1
                if steps >= self.max_steps:
                    break

        return emitted, history

    def iter_run(
        self, *, context: Dict[str, any]
    ) -> Generator[
        Tuple[List[Message], Dict[str, any]], None, Tuple[List[Message], List[Dict[str, any]]]
    ]:
        history: List[Dict[str, any]] = []
        emitted_all: List[Message] = []
        steps = 0
        try:
            while self.queue and steps < self.max_steps:
                msg = self.queue[0]
                if not _due(msg):
                    self.queue.rotate(-1)
                    continue

                msg = self.queue.popleft()
                if msg.id in self.seen:
                    continue
                self.seen.add(msg.id)

                nodes = list(self.subs.get(msg.type, []))
                if not nodes:
                    ev = {
                        "timestamp": msg.ts.isoformat(),
                        "node": None,
                        "consumed": msg.type,
                        "emitted": [],
                        "details": "no-subscriber",
                    }
                    history.append(ev)
                    emitted_all.append(msg)
                    steps += 1
                    yield [msg], ev
                    continue

                for n in nodes:
                    try:
                        out = n.process(msg, context) or []
                    except Exception as e:
                        out = [
                            msg.child(
                                type="error.v1",
                                body={"node": n.name, "for": msg.type, "error": str(e)},
                            )
                        ]

                    corr = msg.headers.get("correlation_id", msg.id)
                    out = [
                        (
                            om
                            if om.headers.get("correlation_id")
                            else om.with_header(correlation_id=corr, parent_id=msg.id)
                        )
                        for om in out
                    ]

                    self.publish(*out)
                    ev = {
                        "timestamp": msg.ts.isoformat(),
                        "node": n.name,
                        "consumed": msg.type,
                        "emitted": [m.type for m in out],
                        "details": "",
                    }
                    history.append(ev)
                    emitted_all.extend(out)
                    steps += 1
                    yield out, ev
                    if steps >= self.max_steps:
                        break
        finally:
            return emitted_all, history
