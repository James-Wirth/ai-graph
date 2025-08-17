from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Annotated, get_origin, get_args
from pydantic import BaseModel

class Inject:
    __aigraph_inject__ = True

@dataclass(frozen=True)
class From(Inject):
    path: str  

@dataclass(frozen=True)
class FromVar(Inject):
    path: str  

@dataclass(frozen=True)
class ToolValue(Inject):
    name: str        
    field: str = "output"  

class Context(Inject): ...
class Payload(Inject): ...

def get_path(root: Any, path: str) -> Any:
    cur = root
    for part in (path or "").split("."):
        if part == "":
            continue
        cur = (cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None))
        if cur is None:
            break
    return cur

def _is_annotated_with(t, marker_type) -> tuple[bool, Any]:
    if get_origin(t) is Annotated:
        base, *extras = get_args(t)
        for e in extras:
            if isinstance(e, marker_type):
                return True, (base, e)
        return True, (base, None)
    return False, (t, None)

def _coerce(value, typ):
    try:
        if value is None:
            return None
        if isinstance(typ, type) and issubclass(typ, BaseModel):
            return typ.model_validate(value)
        return value
    except Exception:
        return value

def resolve_fn_args(param_specs, payload, ctx_vars, tool_ns) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def is_annotated_with(t, marker_type):
        if get_origin(t) is Annotated:
            base, *extras = get_args(t)
            for e in extras:
                if isinstance(e, marker_type):
                    return True, (base, e)
            return True, (base, None)
        return False, (t, None)

    def coerce(value, typ):
        try:
            if value is None:
                return None
            if isinstance(typ, type) and issubclass(typ, BaseModel):
                return typ.model_validate(value)
            return value
        except Exception:
            return value

    for spec in param_specs:
        if len(spec) == 2:
            name, ann = spec
            default = None
        else:
            name, ann, default = spec

        is_anno, (base, injected) = is_annotated_with(ann, Inject)
        if injected is None and isinstance(default, Inject):
            injected = default

        if injected is None:
            if ann is Context or default is Context or name in {"ctx", "context"}:
                out[name] = ctx_vars
                continue
            if ann is Payload or default is Payload or name == "payload":
                out[name] = payload
                continue

        if isinstance(injected, From):
            out[name] = coerce(get_path(payload, injected.path), base)
            continue
        if isinstance(injected, FromVar):
            out[name] = coerce(get_path(ctx_vars, injected.path), base)
            continue
        if isinstance(injected, ToolValue):
            t = tool_ns.get(injected.name) or {}
            out[name] = t if injected.field == "all" else (t.get(injected.field) if isinstance(t, dict) else None)
            continue

        out[name] = coerce(get_path(payload, name), ann if is_anno else ann)

    return out
