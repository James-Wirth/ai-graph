import logging
import json
import enum
import re

from typing import Any, Dict, Union, Callable, Optional, List, Set, Tuple, get_args, get_origin
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
        schema_hint: str = "json"
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
        if self.edges:
            self.edges = {str(k.value if isinstance(k, enum.Enum) else k): v for k, v in self.edges.items()}

        self.route_field = route_field
        self.route_selector = route_selector
        self.schema_hint = (schema_hint or "json").lower()

        if self.route_field and hasattr(self.response_model_cls, "model_fields"):
            fld = self.response_model_cls.model_fields.get(self.route_field)
            if not fld:
                raise ValueError(f"[{self.name}] route_field '{self.route_field}' not found in response model.")
            ann = getattr(fld, "annotation", None)

            def _allowed(ann_type) -> Set[str]:
                if ann_type is None:
                    return set()
                origin = get_origin(ann_type)
                args = get_args(ann_type)
                if origin is Union and args and type(None) in args:
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

        data = (input_model.model_dump()
                if hasattr(input_model, "model_dump")
                else getattr(input_model, "__dict__", input_model))
        root = {"input": data, "context": context}

        base = Template(self.prompt_template).safe_substitute(
            input=json.dumps(data, ensure_ascii=False, indent=2),
            context=json.dumps(context, ensure_ascii=False, indent=2),
        )

        def _get_path(obj, path):
            cur = obj
            for part in path.split("."):
                if isinstance(cur, dict):
                    cur = cur.get(part)
                else:
                    cur = getattr(cur, part, None)
                if cur is None:
                    break
            return cur

        def repl(m):
            path = m.group(1)  
            val = _get_path(root, path)
            return "null" if val is None else json.dumps(val, ensure_ascii=False)

        return re.sub(r"\$\{([a-zA-Z_][\w\.]+)\}", repl, base)

    def build_messages(self, input_model: BaseModel, context: Dict[str, Any]) -> List[Dict[str, str]]:
        prompt = self.build_prompt(input_model, context)

        msgs: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a *precise* JSON generator. Return ONLY valid JSON for the schema. No code fences, no extra text."},
        ]
        hint = self._schema_hint_text(self.schema_hint)
        if hint:
            msgs.append({"role": "system", "content": hint})

        msgs.append({"role": "user", "content": prompt})
        return msgs
    
    def _schema_hint_text(self, style: str) -> str:
        try:
            cls = self.response_model_cls
            schema = cls.model_json_schema()
        except Exception:
            return ""

        style = (style or "json").lower()
        if style == "none":
            return ""
        if style == "schema":
                return "Output must match this JSON Schema:\n" + json.dumps(schema, ensure_ascii=False)
        if style == "fields":
            props = schema.get("properties", {}) or {}
            req = set(schema.get("required", []) or [])
            lines = []
            for k, v in props.items():
                t = v.get("type", "object")
                if isinstance(t, list):
                    t = "/".join(map(str, t))
                star = " (required)" if k in req else ""
                lines.append(f"- {k}: {t}{star}")
            return "Return a single JSON object with these fields:\n" + "\n".join(lines)
        props = (schema.get("properties") or {}).keys()
        shape = {k: None for k in props}
        return "Return only valid JSON of this shape:\n" + json.dumps(shape, ensure_ascii=False)

    def _llm_round(self, messages: List[Dict[str, str]]) -> BaseModel:
        raw = self.adapter.generate(messages=messages, response_model=self.response_model_cls)
        self.logger.debug("Raw LLM response (first 2k): %s ... (%d chars)",
                  raw[:2000], len(raw))
        try:
            return self.response_model_cls.model_validate_json(raw)
        except ValidationError:
            try:
                m = re.search(r"\{.*\}\s*$", raw, re.S)
                if m: return self.response_model_cls.model_validate_json(m.group(0))
            except Exception:
                pass
            self.logger.error("Validation failed. Raw output (first 2k): %s", raw[:2000])
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

        if isinstance(key, enum.Enum): 
            key = key.value
        key = None if key is None else str(key)

        if self.edges:
            if key not in self.edges:
                return None, f"Unknown route key '{key}' for edges {sorted(self.edges)}", 0.1
            target = self.edges[key]
            if target not in neighbors:
                raise ValueError(
                    f"[{self.name}] Edge for key '{key}' >> '{target}' not one of the actual neighbors {neighbors}"
                )
            return target, f"Agent chose route '{key}'", 0.9

        if key in neighbors:
            return key, f"Agent chose neighbor '{key}'", 0.9

        return None, f"Agent chose invalid neighbor '{key}'", 0.1
