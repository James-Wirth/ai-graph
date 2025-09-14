# aigraph/core/nodes.py
from __future__ import annotations
import logging
from typing import Any, Dict, List, Sequence
from aigraph.core.messages import Message


class Node:
    def __init__(self, name: str, *, consumes: Sequence[str], emits: Sequence[str] | None = None):
        self.name = name
        self.consumes = list(consumes or [])
        self.emits = list(emits or [])
        self.logger = logging.getLogger(f"aigraph.node.{self.name}")

    def process(self, msg: Message, context: Dict[str, Any]) -> List[Message]:
        raise NotImplementedError
