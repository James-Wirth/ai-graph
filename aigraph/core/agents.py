import logging

from typing import Any, Dict, Optional, List, Tuple
from pydantic import BaseModel


class Agent:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"aigraph.agent.{self.name}")

    def process(self, input_model: BaseModel, context: Dict[str, Any]) -> BaseModel:
        raise NotImplementedError

    def route(
        self,
        neighbors: List[Any],
        context: Dict[str, Any],
        last_output: Optional[BaseModel],
    ) -> Tuple[Optional[Any], str, float]:
        if not neighbors:
            return None, "No neighbors", 1.0
        if len(neighbors) == 1:
            return neighbors[0], "Only neighbor", 1.0
        return None, "No routing protocol defined", 0.5


