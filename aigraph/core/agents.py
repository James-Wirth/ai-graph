import logging
import json
import enum

from typing import Any, Dict, Callable, Optional, List, Set, Tuple, get_args, get_origin
from pydantic import BaseModel, ValidationError
from string import Template

from aigraph.interfaces.ollama import LLMInterface


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

        prompt_builder: Optional[Callable[[BaseModel, Dict[str, Any]], str]] = None,

        allowed_tools: Optional[List[str]] = None,
        max_tool_rounds: int = 0,
    ):
        super().__init__(name=name)
        self.prompt_template = prompt_template
        self.prompt_builder = prompt_builder

        if isinstance(response_model, type):
            self.response_model_cls = response_model
        else:
            self.response_model_cls = response_model.__class__

        self.adapter = interface

        self.edges = edges or {}
        self.route_field = route_field
        self.route_selector = route_selector

        self.allowed_tools: Set[str] = set(allowed_tools or [])
        self.max_tool_rounds = max(0, int(max_tool_rounds))

        if self.route_field and hasattr(response_model, "model_fields"):
            fld = response_model.model_fields.get(self.route_field)
            if not fld:
                raise ValueError(f"[{self.name}] route_field '{self.route_field}' not found in response model.")
            ann = getattr(fld, "annotation", None)

            def _allowed(ann_type) -> Set[str]:
                if ann_type is None:
                    return set()
                origin = get_origin(ann_type)
                args = get_args(ann_type)
                if origin is Optional or (origin is getattr(__import__('typing'), 'Union', None) and type(None) in args):
                    return {v for a in args if a is not type(None) for v in _allowed(a)}
                from typing import Literal
                if origin is Literal:
                    return {str(v) for v in args}
                if isinstance(ann_type, type) and issubclass(ann_type, enum.Enum):
                    return {str(m.value) for m in ann_type}
                return set()

            opts = _allowed(ann)
            if opts:
                unknown = set(self.edges.keys()) - opts
                if unknown:
                    raise ValueError(f"[{self.name}] edges contain keys not in response model '{self.route_field}': {unknown}")

    def build_prompt(self, input_model: BaseModel, context: Dict[str, Any]) -> str:
        if self.prompt_builder:
            return self.prompt_builder(input_model, context)

        data = (
            input_model.model_dump()
            if hasattr(input_model, "model_dump")
            else getattr(input_model, "__dict__", input_model)
        )
        tmpl = Template(self.prompt_template)
        return tmpl.safe_substitute(
            input=json.dumps(data, ensure_ascii=False, indent=2),
            context=json.dumps(context, ensure_ascii=False, indent=2),
        )

    def build_messages(self, input_model: BaseModel, context: Dict[str, Any]) -> List[Dict[str, str]]:
        prompt = self.build_prompt(input_model, context)

        msgs: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a *precise* JSON generator. Return ONLY valid JSON for the schema."},
            {"role": "user", "content": prompt},
        ]

        last_tool = context.get("__tool_last__")
        if last_tool:
            msgs.append({
                "role": "tool",
                "content": json.dumps(last_tool, ensure_ascii=False),
            })

        return msgs

    def _llm_round(self, messages: List[Dict[str, str]]) -> BaseModel:
        raw = self.adapter.generate(messages=messages, response_model=self.response_model_cls)
        self.logger.debug("Raw LLM response: %s", raw)
        try:
            return self.response_model_cls.model_validate_json(raw)
        except ValidationError:
            self.logger.error("Validation failed. Raw output: %s", raw)
            raise

    def process(self, input_model: BaseModel, context: Dict[str, Any]) -> BaseModel:
        self.logger.info("Agent '%s' processing input: %s", self.name, input_model)
        messages = self.build_messages(input_model, context)
        return self._llm_round(messages)


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

        if key is not None:
            if isinstance(key, enum.Enum):
                key = key.value
            if not isinstance(key, str):
                key = str(key)

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
