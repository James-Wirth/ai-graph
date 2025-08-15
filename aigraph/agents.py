import logging

from typing import Any, Dict, Callable, Optional, List, Set, Tuple
from pydantic import BaseModel, ValidationError

from aigraph.interfaces.ollama import LLMInterface


class ToolCall(BaseModel):
    name: str
    input: Dict[str, Any] = {}


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
        return None, "No routing protocol defined", 1.0


class LLMAgent(Agent):

    def __init__(
        self,
        name: str,
        prompt_template: str,
        response_model: Any,
        interface: LLMInterface,
        *,

        edges: Optional[Dict[str, Any]] = None,        
        route_field: Optional[str] = None,
        route_selector: Optional[Callable[[BaseModel, Dict[str, Any]], Optional[str]]] = None,

        allowed_tools: Optional[List[str]] = None,
        max_tool_rounds: int = 0,
    ):
        super().__init__(name=name)
        self.prompt_template = prompt_template
        self.response_model = response_model
        self.adapter = interface

        self.edges = edges or {}
        self.route_field = route_field
        self.route_selector = route_selector

        self.allowed_tools: Set[str] = set(allowed_tools or [])
        self.max_tool_rounds = max(0, int(max_tool_rounds))

        if self.route_field and hasattr(response_model, 'model_fields'):
            field = response_model.model_fields.get(self.route_field)
            ann = getattr(field, "annotation", None) if field else None
            enum_args = getattr(ann, "__args__", None)
            if enum_args:
                options = {str(x) for x in enum_args}
                unknown = set(self.edges.keys()) - options
                if unknown:
                    raise ValueError(
                        f"[{self.name}] edges contain keys not in response model '{self.route_field}': {unknown}"
                    )

    def build_prompt(self, input_model: BaseModel, context: Dict[str, Any]) -> str:
        return self.prompt_template.format(input=input_model.model_dump(), context=context)

    def _llm_round(self, prompt: str, schema: Dict[str, Any]) -> BaseModel:
        raw = self.adapter.generate(prompt=prompt, response_schema=schema)
        self.logger.debug("Raw LLM response: %s", raw)
        try:
            return self.response_model.model_validate_json(raw)
        except ValidationError:
            self.logger.error("Validation failed. Raw output: %s", raw)
            raise

    def process(self, input_model: BaseModel, context: Dict[str, Any]) -> BaseModel:
        self.logger.info("Agent '%s' processing input: %s", self.name, input_model)
        prompt = self.build_prompt(input_model, context)
        schema = self.response_model.model_json_schema()
        return self._llm_round(prompt, schema)

    def route(
        self,
        neighbors: List[Any],
        context: Dict[str, Any],
        last_output: Optional[BaseModel],
    ) -> Tuple[Optional[Any], str, float]:
        
        if not neighbors:
            return None, "No neighbors", 1.0
        if len(neighbors) == 1 and not self.edges:
            return neighbors[0], "Only neighbor", 1.0

        key: Optional[str] = None
        if last_output is not None:
            if self.route_selector:
                key = self.route_selector(last_output, context)
            elif self.route_field and hasattr(last_output, self.route_field):
                key = getattr(last_output, self.route_field)

        if key is None:
            return None, "No route key produced", 0.5

        if self.edges:
            if key not in self.edges:
                return None, f"Unknown route key '{key}' for edges {sorted(self.edges)}", 0.1
            target = self.edges[key]
            if target not in neighbors:
                raise AssertionError(
                    f"[{self.name}] Edge for key '{key}' >> '{target}' not one of the actual neighbors {neighbors}"
                )
            return target, f"Agent chose route '{key}'", 0.9

        if key in neighbors:
            return key, f"Agent chose neighbor '{key}'", 0.9

        return None, f"Agent chose invalid neighbor '{key}'", 0.1
